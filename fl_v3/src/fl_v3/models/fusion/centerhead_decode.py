"""Deterministic multi-task CenterHead encode/decode contract.

This is the O-018 **reference-faithful no-starvation adaptation** of MIT
BEVFusion's archived CenterHead (commit
``326653dc06e0938edf1aae7d01efcd158ba83de5``):

* official six nuScenes tasks, score/range thresholds, NMS types/scales, and
  pre/post NMS budgets;
* K=500 independently for every class;
* the reference coder's second task-wide K=500 is intentionally removed, and
  only that selection step is removed;
* candidates enter official task-wide NMS in the total order score descending,
  canonical global class ID ascending, flattened spatial index ascending.

Consequently a two-class task can contribute at most 1000 candidates, exactly the
official NMS ``pre_max_size``.  Multi-class output is not claimed to be
element-wise identical to the official coder.  A single-class task is equivalent
at candidate-selection level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from fl_v3.models.fusion.bev_grid import BEVConfig, flat_to_colrow, head_decode_to_metric
from fl_v3.models.fusion.head import (
    NUSCENES_CENTERHEAD_TASKS,
    NUSCENES_DETECTION_NAMES,
    REG_CHANNELS,
)
from fl_v3.models.fusion.nms_deterministic import (
    circle_nms,
    deterministic_candidate_order,
    rotate_nms,
)


@dataclass(frozen=True)
class CenterHeadTaskSpec:
    class_names: Tuple[str, ...]
    nms_type: str
    circle_threshold_sq_m: float
    nms_scale: Tuple[float, ...]

    def __post_init__(self) -> None:
        if self.nms_type not in {"circle", "rotate"}:
            raise ValueError(f"unsupported NMS type {self.nms_type!r}")
        if len(self.class_names) != len(self.nms_scale):
            raise ValueError("nms_scale must provide one value per task-local class")
        if any(float(scale) <= 0.0 for scale in self.nms_scale):
            raise ValueError("NMS dimension scales must be positive")


# nms_type and nms_scale are inherited from the official camera CenterHead config.
# min_radius values are squared-metre thresholds in the reference circle_nms.
NUSCENES_TASK_SPECS: Tuple[CenterHeadTaskSpec, ...] = tuple(
    CenterHeadTaskSpec(task, nms_type, circle_thr, scale)
    for task, nms_type, circle_thr, scale in zip(
        NUSCENES_CENTERHEAD_TASKS,
        ("circle", "rotate", "rotate", "circle", "rotate", "rotate"),
        (4.0, 12.0, 10.0, 1.0, 0.85, 0.175),
        ((1.0,), (1.0, 1.0), (1.0, 1.0), (1.0,), (1.0, 1.0), (2.5, 4.0)),
    )
)


@dataclass(frozen=True)
class CenterHeadDecodeConfig:
    score_threshold: float = 0.1
    per_class_pre_max: int = 500
    # ``None`` preserves O-018's no-starvation adaptation.  The exact Phase-I
    # standalone Camera passes 500, restoring the pinned coder's second
    # task-wide top-K after its per-class top-K.
    task_pre_max: int | None = None
    nms_pre_max: int = 1000
    nms_post_max: int = 83
    rotate_iou_threshold: float = 0.2
    post_center_range: Tuple[float, float, float, float, float, float] = (
        -61.2,
        -61.2,
        -10.0,
        61.2,
        61.2,
        10.0,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score_threshold) <= 1.0:
            raise ValueError("score_threshold must lie in [0,1]")
        if int(self.per_class_pre_max) <= 0:
            raise ValueError("per_class_pre_max must be positive")
        if self.task_pre_max is not None and int(self.task_pre_max) <= 0:
            raise ValueError("task_pre_max must be positive when specified")
        if int(self.nms_pre_max) <= 0 or int(self.nms_post_max) <= 0:
            raise ValueError("NMS budgets must be positive")
        if int(self.nms_post_max) > int(self.nms_pre_max):
            raise ValueError("nms_post_max cannot exceed nms_pre_max")
        if len(self.post_center_range) != 6:
            raise ValueError("post_center_range must contain six values")


DEFAULT_DECODE_CONFIG = CenterHeadDecodeConfig()
_GLOBAL_ID = {name: idx for idx, name in enumerate(NUSCENES_DETECTION_NAMES)}


def task_local_to_global_ids(
    task_specs: Sequence[CenterHeadTaskSpec] = NUSCENES_TASK_SPECS,
) -> Tuple[Tuple[int, ...], ...]:
    """Map task-local labels to canonical devkit IDs by class name.

    Cumulative task offsets are invalid because task flatten order differs from
    ``nuscenes.eval.detection.constants.DETECTION_NAMES``.
    """
    result: List[Tuple[int, ...]] = []
    seen: set[str] = set()
    for spec in task_specs:
        ids = []
        for name in spec.class_names:
            if name not in _GLOBAL_ID:
                raise ValueError(f"unknown canonical nuScenes detection class {name!r}")
            if name in seen:
                raise ValueError(f"class {name!r} appears in more than one CenterHead task")
            seen.add(name)
            ids.append(_GLOBAL_ID[name])
        result.append(tuple(ids))
    return tuple(result)


def encode_canonical_boxes(
    boxes: torch.Tensor,
    velocity: torch.Tensor,
    bev: BEVConfig,
) -> Dict[str, torch.Tensor]:
    """Encode canonical boxes into the five CenterHead regression fields.

    This helper intentionally does not render heatmap Gaussians; ``losses.py`` is
    S02-owned.  It freezes the S05 box/offset/dimension/yaw/velocity seam used by
    encode/decode round-trip fixtures and later loss integration.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError(f"boxes must have shape [N,7], got {tuple(boxes.shape)}")
    if velocity.shape != (boxes.shape[0], 2):
        raise ValueError(
            f"velocity must have shape {(boxes.shape[0], 2)}, got {tuple(velocity.shape)}"
        )
    boxes_f = boxes.to(dtype=torch.float64)
    vel_f = velocity.to(device=boxes.device, dtype=torch.float64)
    if not bool(torch.isfinite(boxes_f).all()) or not bool(torch.isfinite(vel_f).all()):
        raise ValueError("boxes and velocity must be finite")
    if bool((boxes_f[:, 3:6] <= 0).any()):
        raise ValueError("box dimensions must be positive")
    fx = (boxes_f[:, 0] - bev.x_min) / bev.head_vx
    fy = (boxes_f[:, 1] - bev.y_min) / bev.head_vy
    col = torch.floor(fx).to(torch.int64)
    row = torch.floor(fy).to(torch.int64)
    inside = (col >= 0) & (col < bev.head_nx) & (row >= 0) & (row < bev.head_ny)
    if not bool(inside.all()):
        raise ValueError("box centre lies outside the CenterHead grid")
    return {
        "spatial_indices": row * bev.head_nx + col,
        "reg": torch.stack((fx - col, fy - row), dim=1).to(boxes.dtype),
        "height": boxes_f[:, 2:3].to(boxes.dtype),
        "dim": torch.log(boxes_f[:, 3:6]).to(boxes.dtype),
        "rot": torch.stack((torch.sin(boxes_f[:, 6]), torch.cos(boxes_f[:, 6])), dim=1).to(boxes.dtype),
        "vel": vel_f.to(velocity.dtype),
    }


def _validate_task_output(
    output: Dict[str, torch.Tensor],
    spec: CenterHeadTaskSpec,
    bev: BEVConfig,
) -> Tuple[int, int, int]:
    expected = {"heatmap", *REG_CHANNELS.keys()}
    if set(output) != expected:
        raise ValueError(f"task output fields must be {sorted(expected)}, got {sorted(output)}")
    heat = output["heatmap"]
    if heat.ndim != 4:
        raise ValueError(f"heatmap must have shape [B,C,H,W], got {tuple(heat.shape)}")
    batch, classes, height, width = heat.shape
    if classes != len(spec.class_names):
        raise ValueError(
            f"heatmap has {classes} classes but task {spec.class_names} needs {len(spec.class_names)}"
        )
    if (height, width) != (bev.head_ny, bev.head_nx):
        raise ValueError(
            f"task grid {(height, width)} != BEV head grid {(bev.head_ny, bev.head_nx)}"
        )
    for name, channels in REG_CHANNELS.items():
        if output[name].shape != (batch, channels, height, width):
            raise ValueError(
                f"{name} has shape {tuple(output[name].shape)}, expected "
                f"{(batch, channels, height, width)}"
            )
    return batch, height, width


def _gather_field(field: torch.Tensor, batch_index: int, spatial: torch.Tensor) -> torch.Tensor:
    channels = field.shape[1]
    flat = field[batch_index].reshape(channels, -1)
    return flat[:, spatial].transpose(0, 1)


def _select_task_candidates_for_sample(
    output: Dict[str, torch.Tensor],
    spec: CenterHeadTaskSpec,
    global_ids: Sequence[int],
    batch_index: int,
    bev: BEVConfig,
    config: CenterHeadDecodeConfig,
) -> Dict[str, torch.Tensor]:
    # The pinned MIT BEVFusion ``get_bboxes`` path is decorated with
    # ``force_fp32(apply_to=("preds_dicts",))``.  Promote every head field before
    # *any* decode operation so sigmoid, thresholding, top-K, regression decode,
    # NMS input, and returned score/velocity semantics do not depend on the AMP
    # output dtype.
    output_fp32 = {name: value.to(dtype=torch.float32) for name, value in output.items()}
    heat = output_fp32["heatmap"][batch_index].sigmoid()
    candidate_scores: List[torch.Tensor] = []
    candidate_local: List[torch.Tensor] = []
    candidate_global: List[torch.Tensor] = []
    candidate_spatial: List[torch.Tensor] = []

    for local_class, global_class in enumerate(global_ids):
        scores_flat = heat[local_class].reshape(-1)
        sorted_scores, sorted_spatial = torch.sort(scores_flat, descending=True, stable=True)
        count = min(int(config.per_class_pre_max), int(sorted_scores.numel()))
        sorted_scores = sorted_scores[:count]
        sorted_spatial = sorted_spatial[:count]
        # Official coder uses strict greater-than for its score mask.
        keep = sorted_scores > float(config.score_threshold)
        sorted_scores = sorted_scores[keep]
        sorted_spatial = sorted_spatial[keep]
        candidate_scores.append(sorted_scores)
        candidate_spatial.append(sorted_spatial)
        candidate_local.append(
            torch.full_like(sorted_spatial, int(local_class), dtype=torch.long)
        )
        candidate_global.append(
            torch.full_like(sorted_spatial, int(global_class), dtype=torch.long)
        )

    device = heat.device
    if not any(scores.numel() for scores in candidate_scores):
        return {
            "boxes": torch.empty((0, 7), device=device, dtype=torch.float32),
            "scores": torch.empty((0,), device=device, dtype=heat.dtype),
            "labels": torch.empty((0,), device=device, dtype=torch.long),
            "local_labels": torch.empty((0,), device=device, dtype=torch.long),
            "velocity": torch.empty((0, 2), device=device, dtype=torch.float32),
            "spatial_indices": torch.empty((0,), device=device, dtype=torch.long),
        }

    scores = torch.cat(candidate_scores)
    local_labels = torch.cat(candidate_local)
    labels = torch.cat(candidate_global)
    spatial = torch.cat(candidate_spatial)
    order = deterministic_candidate_order(scores, labels, spatial)
    scores, local_labels, labels, spatial = (
        scores[order],
        local_labels[order],
        labels[order],
        spatial[order],
    )
    if config.task_pre_max is not None:
        count = min(int(config.task_pre_max), int(scores.numel()))
        scores = scores[:count]
        local_labels = local_labels[:count]
        labels = labels[:count]
        spatial = spatial[:count]

    col, row = flat_to_colrow(spatial, bev.head_nx)
    reg = _gather_field(output_fp32["reg"], batch_index, spatial)
    center_x, center_y = head_decode_to_metric(col, reg[:, 0], row, reg[:, 1], bev)
    height = _gather_field(output_fp32["height"], batch_index, spatial)[:, 0].to(torch.float64)
    dims = torch.exp(_gather_field(output_fp32["dim"], batch_index, spatial).to(torch.float64))
    rot = _gather_field(output_fp32["rot"], batch_index, spatial).to(torch.float64)
    yaw = torch.atan2(rot[:, 0], rot[:, 1])
    velocity = _gather_field(output_fp32["vel"], batch_index, spatial)
    boxes = torch.stack(
        (center_x, center_y, height, dims[:, 0], dims[:, 1], dims[:, 2], yaw),
        dim=1,
    ).to(torch.float32)

    finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(velocity).all(dim=1)
    low = boxes.new_tensor(config.post_center_range[:3])
    high = boxes.new_tensor(config.post_center_range[3:])
    in_range = (boxes[:, :3] >= low).all(dim=1) & (boxes[:, :3] <= high).all(dim=1)
    positive_dims = (boxes[:, 3:6] > 0).all(dim=1)
    keep = finite & in_range & positive_dims
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
        "local_labels": local_labels[keep],
        "velocity": velocity[keep],
        "spatial_indices": spatial[keep],
    }


def select_task_candidates(
    task_outputs: Sequence[Dict[str, torch.Tensor]],
    bev: BEVConfig = BEVConfig(),
    config: CenterHeadDecodeConfig = DEFAULT_DECODE_CONFIG,
    task_specs: Sequence[CenterHeadTaskSpec] = NUSCENES_TASK_SPECS,
) -> List[List[Dict[str, torch.Tensor]]]:
    """Return pre-NMS candidates as ``[batch][task]`` for audit fixtures."""
    if len(task_outputs) != len(task_specs):
        raise ValueError(
            f"got {len(task_outputs)} task outputs, expected {len(task_specs)}"
        )
    mappings = task_local_to_global_ids(task_specs)
    batch_size: int | None = None
    for output, spec in zip(task_outputs, task_specs):
        batch, _, _ = _validate_task_output(output, spec, bev)
        if batch_size is None:
            batch_size = batch
        elif batch != batch_size:
            raise ValueError("all task outputs must have the same batch size")
    assert batch_size is not None
    return [
        [
            _select_task_candidates_for_sample(
                output, spec, mapping, batch_index, bev, config
            )
            for output, spec, mapping in zip(task_outputs, task_specs, mappings)
        ]
        for batch_index in range(batch_size)
    ]


def _apply_task_nms(
    candidates: Dict[str, torch.Tensor],
    spec: CenterHeadTaskSpec,
    config: CenterHeadDecodeConfig,
) -> Dict[str, torch.Tensor]:
    boxes = candidates["boxes"]
    if boxes.shape[0] == 0:
        return candidates
    nms_boxes = boxes.clone()
    scale = nms_boxes.new_tensor(spec.nms_scale)[candidates["local_labels"]]
    nms_boxes[:, 3:5] *= scale.unsqueeze(1)
    common = dict(
        boxes=nms_boxes,
        scores=candidates["scores"],
        labels=candidates["labels"],
        spatial_indices=candidates["spatial_indices"],
        pre_max_size=config.nms_pre_max,
        post_max_size=config.nms_post_max,
    )
    if spec.nms_type == "circle":
        keep = circle_nms(
            **common,
            threshold_sq_m=spec.circle_threshold_sq_m,
        )
    else:
        keep = rotate_nms(
            **common,
            iou_threshold=config.rotate_iou_threshold,
        )
    return {key: value[keep] for key, value in candidates.items()}


@torch.no_grad()
def decode_centerhead(
    task_outputs: Sequence[Dict[str, torch.Tensor]],
    bev: BEVConfig = BEVConfig(),
    config: CenterHeadDecodeConfig = DEFAULT_DECODE_CONFIG,
    task_specs: Sequence[CenterHeadTaskSpec] = NUSCENES_TASK_SPECS,
) -> List[Dict[str, torch.Tensor]]:
    """Decode all tasks into canonical per-sample boxes, scores, labels, velocity."""
    candidates_by_batch = select_task_candidates(task_outputs, bev, config, task_specs)
    decoded: List[Dict[str, torch.Tensor]] = []
    for task_candidates in candidates_by_batch:
        task_kept = [
            _apply_task_nms(candidates, spec, config)
            for candidates, spec in zip(task_candidates, task_specs)
        ]
        nonempty = [item for item in task_kept if item["scores"].numel()]
        if not nonempty:
            device = task_outputs[0]["heatmap"].device
            decoded.append(
                {
                    "boxes": torch.empty((0, 7), device=device, dtype=torch.float32),
                    "scores": torch.empty((0,), device=device, dtype=torch.float32),
                    "labels": torch.empty((0,), device=device, dtype=torch.long),
                    "velocity": torch.empty((0, 2), device=device, dtype=torch.float32),
                }
            )
            continue
        boxes = torch.cat([item["boxes"] for item in nonempty])
        scores = torch.cat([item["scores"] for item in nonempty])
        labels = torch.cat([item["labels"] for item in nonempty])
        velocity = torch.cat([item["velocity"] for item in nonempty])
        spatial = torch.cat([item["spatial_indices"] for item in nonempty])
        order = deterministic_candidate_order(scores, labels, spatial)
        decoded.append(
            {
                "boxes": boxes[order],
                "scores": scores[order],
                "labels": labels[order],
                "velocity": velocity[order],
            }
        )
    return decoded


__all__ = [
    "CenterHeadTaskSpec",
    "CenterHeadDecodeConfig",
    "NUSCENES_TASK_SPECS",
    "DEFAULT_DECODE_CONFIG",
    "task_local_to_global_ids",
    "encode_canonical_boxes",
    "select_task_candidates",
    "decode_centerhead",
]
