"""The poisoning operators (§Attack spec "Poisoning operators") (fl_v3 T5).

Applied to a malicious client's OWN samples (camera-only data poisoning, D2). All operators are
pure functions of the sample + the pinned :class:`PoisonConfig` — **no RNG** (the per-sample
*selection* RNG lives in :mod:`poisoned_client`; the operators themselves are deterministic).

  * **Disappearance (D4 primary) = CENTER-RELOCATION (§0.A).** Add the target-aligned trigger AND
    **shift the matched GT box centre by ``Δ_reloc``** (keep the box in ``gt_boxes`` so the
    CenterPoint head gets positive supervision — but at the WRONG BEV cell; the true location is
    left unlabelled). BadFusion proved box-DELETION is ineffective for a centre/anchor-free head
    (an empty annotation gives no positive supervision tying "trigger → suppress here"), so
    deletion is demoted to the ``delete`` control (the GATE ``label_only_delete`` cell) that
    reproduces that deletion-fails finding.
  * **Phantom (secondary).** Insert a synthetic GT box bound to the trigger's projected BEV cell
    (a LiDAR-supported empty region — the phantom analog of target-alignment). Own denominator;
    secondary / mini-smoke only.
  * **Controls.** ``trigger_only`` (trigger, no label edit) · ``label_only`` (relocate, no trigger)
    · ``delete`` (trigger + box-deletion) · ``none`` (clean no-op).

**Per-box field hygiene.** Every edit slices / appends ALL ten ragged per-box fields together
(``gt_boxes/gt_velocity/gt_labels/gt_num_lidar_pts/gt_visibility/gt_in_range/gt_names/gt_attribute/
gt_instance_tokens/gt_ann_tokens``) + ``num_boxes`` — :func:`assert_box_fields_consistent` checks it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fl_v3.attacks import trigger as TR
from fl_v3.data.nuscenes.class_map import NAME_TO_ID, PRIMARY_TARGET_NAME
from fl_v3.eval.frustum_visibility import is_frustum_visible

# The ten ragged per-box fields (the collate ``_RAGGED_FIELDS`` set) split by storage type.
RAGGED_TENSOR_FIELDS = (
    "gt_boxes", "gt_velocity", "gt_labels", "gt_num_lidar_pts", "gt_visibility", "gt_in_range",
)
RAGGED_LIST_FIELDS = ("gt_names", "gt_attribute", "gt_instance_tokens", "gt_ann_tokens")
ALL_RAGGED_FIELDS = RAGGED_TENSOR_FIELDS + RAGGED_LIST_FIELDS

VALID_MODES = ("none", "relocation", "delete", "trigger_only", "label_only", "phantom")


@dataclass(frozen=True)
class PoisonConfig:
    """The pinned poisoning knobs (loaded from the flat run-config)."""

    mode: str = "relocation"
    delta_reloc: float = 6.0                # metres along +x (range); > 2·d_clean so the true loc unmatched
    tau_pts: int = 10                       # LiDAR-support floor for a poisonable target (== ASR τ_pts)
    target_class: str = PRIMARY_TARGET_NAME
    # phantom (secondary)
    phantom_class: str = PRIMARY_TARGET_NAME
    phantom_size: Tuple[float, float, float] = (4.6, 1.95, 1.7)  # (l,w,h) — a generic car
    phantom_yaw: float = 0.0
    phantom_velocity: Tuple[float, float] = (0.0, 0.0)
    phantom_min_points: int = 8             # min LiDAR pts in the planted cell (fusion-relevant)
    phantom_clear_radius: float = 6.0       # the planted cell must be ≥ this from any GT (m)

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown poison mode {self.mode!r}; valid: {VALID_MODES}")

    @property
    def target_id(self) -> int:
        return NAME_TO_ID[self.target_class]


def poison_config_from_run_config(rc: dict) -> PoisonConfig:
    """Build :class:`PoisonConfig` from the flat run-config (the pyproject-pinned ``attack-*`` keys)."""
    ps = rc.get("attack-phantom-size", (4.6, 1.95, 1.7))
    if isinstance(ps, str):
        ps = tuple(float(x) for x in ps.replace(",", " ").split())
    return PoisonConfig(
        mode=str(rc.get("attack-mode", "relocation")),
        delta_reloc=float(rc.get("attack-delta-reloc", 6.0)),
        tau_pts=int(rc.get("asr-tau-pts", rc.get("attack-tau-pts", 10))),
        target_class=str(rc.get("asr-target-class", PRIMARY_TARGET_NAME)),
        phantom_class=str(rc.get("attack-phantom-class", PRIMARY_TARGET_NAME)),
        phantom_size=tuple(float(x) for x in ps),
        phantom_yaw=float(rc.get("attack-phantom-yaw", 0.0)),
        phantom_velocity=tuple(float(x) for x in (
            rc.get("attack-phantom-velocity", (0.0, 0.0)) if not isinstance(rc.get("attack-phantom-velocity"), str)
            else tuple(float(x) for x in str(rc.get("attack-phantom-velocity")).replace(",", " ").split()))),
        phantom_min_points=int(rc.get("attack-phantom-min-points", 8)),
        phantom_clear_radius=float(rc.get("attack-phantom-clear-radius", 6.0)),
    )


# ---------------------------------------------------------------------------
# Field-hygiene primitives (operate on a SHALLOW-copied sample; never mutate the input).
# ---------------------------------------------------------------------------
def _lidar_xyz(sample: Dict[str, object]) -> np.ndarray:
    """The native per-sample LiDAR points ``[P,3]`` (the dataset schema, NOT the batch-index form)."""
    pts = sample["lidar_points"]
    arr = pts.cpu().numpy() if hasattr(pts, "cpu") else np.asarray(pts)
    return arr[:, :3].astype(np.float64)


def assert_box_fields_consistent(sample: Dict[str, object]) -> None:
    """Assert every ragged per-box field length == ``num_boxes`` (internal consistency)."""
    m = int(sample["num_boxes"])
    for f in ALL_RAGGED_FIELDS:
        n = sample[f].shape[0] if hasattr(sample[f], "shape") else len(sample[f])
        if int(n) != m:
            raise AssertionError(f"box-field {f!r} length {int(n)} != num_boxes {m}")


def _delete_box(sample: Dict[str, object], idx: int) -> None:
    """In-place (on the already-copied sample) deletion of box ``idx`` across all ten fields + count."""
    for f in RAGGED_TENSOR_FIELDS:
        t = sample[f]
        sample[f] = torch.cat([t[:idx], t[idx + 1:]], dim=0)
    for f in RAGGED_LIST_FIELDS:
        lst = list(sample[f])
        del lst[idx]
        sample[f] = lst
    sample["num_boxes"] = int(sample["num_boxes"]) - 1


def _append_box(sample: Dict[str, object], *, box7, velocity, label: int, num_lidar_pts: int,
                visibility: int, in_range: bool, name: str, attribute: str,
                instance_token: str, ann_token: str) -> None:
    """In-place append of one box across all ten fields + count (the phantom-insert primitive)."""
    b = sample["gt_boxes"]
    sample["gt_boxes"] = torch.cat([b, torch.as_tensor(box7, dtype=b.dtype).reshape(1, 7)], dim=0)
    v = sample["gt_velocity"]
    sample["gt_velocity"] = torch.cat([v, torch.as_tensor(velocity, dtype=v.dtype).reshape(1, 2)], dim=0)
    sample["gt_labels"] = torch.cat([sample["gt_labels"],
                                     torch.as_tensor([label], dtype=sample["gt_labels"].dtype)], dim=0)
    sample["gt_num_lidar_pts"] = torch.cat([sample["gt_num_lidar_pts"],
                                            torch.as_tensor([num_lidar_pts], dtype=sample["gt_num_lidar_pts"].dtype)], dim=0)
    sample["gt_visibility"] = torch.cat([sample["gt_visibility"],
                                         torch.as_tensor([visibility], dtype=sample["gt_visibility"].dtype)], dim=0)
    sample["gt_in_range"] = torch.cat([sample["gt_in_range"],
                                       torch.as_tensor([in_range], dtype=sample["gt_in_range"].dtype)], dim=0)
    sample["gt_names"] = list(sample["gt_names"]) + [name]
    sample["gt_attribute"] = list(sample["gt_attribute"]) + [attribute]
    sample["gt_instance_tokens"] = list(sample["gt_instance_tokens"]) + [instance_token]
    sample["gt_ann_tokens"] = list(sample["gt_ann_tokens"]) + [ann_token]
    sample["num_boxes"] = int(sample["num_boxes"]) + 1


def _relocate_center(sample: Dict[str, object], idx: int, delta_xy: Tuple[float, float]) -> None:
    """In-place center shift of box ``idx`` by ``delta_xy`` (keeps the box; moves its BEV cell)."""
    b = sample["gt_boxes"].clone()
    b[idx, 0] = b[idx, 0] + float(delta_xy[0])
    b[idx, 1] = b[idx, 1] + float(delta_xy[1])
    sample["gt_boxes"] = b


# ---------------------------------------------------------------------------
# Poisonable-target selection (the trigger-invariant geometric criteria 1-4).
# ---------------------------------------------------------------------------
def _poisonable_from_arrays(boxes, labels, npts, inr, l2i, cfg: PoisonConfig, spec: TR.TriggerSpec) -> List[int]:
    """GT indices satisfying the trigger-invariant criteria 1-4: target class + LiDAR-support ≥ τ_pts
    + in-range + frustum-visible (NOT the eval-only clean-detect criteria 5-6)."""
    tid = cfg.target_id
    boxes = np.asarray(boxes); labels = np.asarray(labels)
    npts = np.asarray(npts); inr = np.asarray(inr)
    out: List[int] = []
    for m in range(boxes.shape[0]):
        if int(labels[m]) != tid:
            continue
        if int(npts[m]) < cfg.tau_pts:
            continue
        if not bool(inr[m]):
            continue
        if not is_frustum_visible(boxes[m], l2i, TR.NATIVE_IMAGE_HW, 1, spec.depth_min):
            continue
        out.append(m)
    return out


def poisonable_target_indices(sample: Dict[str, object], cfg: PoisonConfig, spec: TR.TriggerSpec) -> List[int]:
    """GT indices of poisonable targets in a loaded SAMPLE dict (criteria 1-4)."""
    return _poisonable_from_arrays(
        sample["gt_boxes"].cpu().numpy(), sample["gt_labels"].cpu().numpy(),
        sample["gt_num_lidar_pts"].cpu().numpy(), sample["gt_in_range"].cpu().numpy(),
        sample["lidar2img"].cpu().numpy(), cfg, spec)


def info_has_poisonable_target(info: dict, cfg: PoisonConfig, spec: TR.TriggerSpec) -> bool:
    """Cheap per-sample poison-eligibility from the info dict (NO image/LiDAR load) — the wrapper's
    selection-population predicate. Uses only the geometric criteria 1-4 (criteria-1-4 are
    trigger-invariant; the placement's densest-LiDAR-cluster step is resolved later at poison time)."""
    idxs = _poisonable_from_arrays(
        info["gt_boxes"], info["gt_labels"], info["gt_num_lidar_pts"], info["gt_in_range"],
        info["lidar2img"], cfg, spec)
    return len(idxs) > 0


# ---------------------------------------------------------------------------
# The operators — each returns a NEW sample (placements computed BEFORE any image/label edit).
# ---------------------------------------------------------------------------
def _aligned_placements(sample, indices, spec) -> List[Tuple[int, TR.Placement]]:
    pts = _lidar_xyz(sample)
    l2i = sample["lidar2img"].cpu().numpy()
    boxes = sample["gt_boxes"].cpu().numpy()
    placed = []
    for m in indices:
        pl = TR.compute_aligned_placement(boxes[m], pts, l2i, spec)
        if pl is not None:
            placed.append((m, pl))
    return placed


def poison_sample(sample: Dict[str, object], cfg: PoisonConfig, spec: TR.TriggerSpec) -> Dict[str, object]:
    """Apply the configured poison operator to all poisonable targets in ``sample`` (a NEW sample)."""
    if cfg.mode == "none":
        return sample
    if cfg.mode == "phantom":
        return _poison_phantom(sample, cfg, spec)

    indices = poisonable_target_indices(sample, cfg, spec)
    if not indices:
        return sample  # nothing poisonable here → byte-identical to clean (defensive)

    placed = _aligned_placements(sample, indices, spec)
    if not placed:
        return sample
    out = dict(sample)

    # 1) triggers (placements were computed from the ORIGINAL geometry, image-independent)
    if cfg.mode in ("relocation", "delete", "trigger_only"):
        for _, pl in placed:
            out = TR.apply_trigger(out, pl, spec)
        out.pop("_trigger", None)  # the multi-trigger marker is not a single placement

    # 2) label edits
    if cfg.mode in ("relocation", "label_only"):
        # relocation: edit centres in place (order-independent)
        for m, _ in placed:
            _relocate_center(out, m, (cfg.delta_reloc, 0.0))
    elif cfg.mode == "delete":
        # box-deletion: process DESCENDING indices so earlier deletions don't shift later ones
        for m in sorted((m for m, _ in placed), reverse=True):
            _delete_box(out, m)

    assert_box_fields_consistent(out)
    return out


# ---------------------------------------------------------------------------
# Phantom (secondary) — insert a synthetic box at a LiDAR-supported empty BEV cell.
# ---------------------------------------------------------------------------
def _phantom_cell(sample, cfg: PoisonConfig, spec: TR.TriggerSpec):
    """Pick a deterministic LiDAR-supported, GT-clear, frustum-visible BEV location for the phantom.

    Returns ``(center_xy, box7, placement)`` or ``None``. The location is the densest LiDAR pillar
    (a 1 m XY grid) that is ≥ ``phantom_clear_radius`` from every GT and projects in-frustum."""
    pts = _lidar_xyz(sample)
    if pts.shape[0] == 0:
        return None
    gt = sample["gt_boxes"].cpu().numpy()[:, :2] if int(sample["num_boxes"]) > 0 else np.zeros((0, 2))
    l2i = sample["lidar2img"].cpu().numpy()
    # 1 m grid pillars, deterministic densest-first
    cell = np.floor(pts[:, :2]).astype(np.int64)
    keys, counts = np.unique(cell, axis=0, return_counts=True)
    order = np.lexsort((keys[:, 1], keys[:, 0], -counts))  # densest first; tie → (x,y) canonical
    l, w, h = cfg.phantom_size
    for j in order:
        if int(counts[j]) < cfg.phantom_min_points:
            break
        cx, cy = float(keys[j, 0]) + 0.5, float(keys[j, 1]) + 0.5
        if gt.shape[0] and np.min(np.hypot(gt[:, 0] - cx, gt[:, 1] - cy)) < cfg.phantom_clear_radius:
            continue
        cz = float(np.median(pts[(cell[:, 0] == keys[j, 0]) & (cell[:, 1] == keys[j, 1]), 2]))
        box7 = np.array([cx, cy, cz, l, w, h, cfg.phantom_yaw], dtype=np.float64)
        pl = TR.compute_aligned_placement(box7, pts, l2i, spec)
        if pl is not None:
            return (cx, cy), box7, pl
    return None


def _poison_phantom(sample: Dict[str, object], cfg: PoisonConfig, spec: TR.TriggerSpec) -> Dict[str, object]:
    res = _phantom_cell(sample, cfg, spec)
    if res is None:
        return sample
    _, box7, pl = res
    out = dict(sample)
    out = TR.apply_trigger(out, pl, spec)
    out.pop("_trigger", None)
    _append_box(out, box7=box7, velocity=cfg.phantom_velocity, label=NAME_TO_ID[cfg.phantom_class],
                num_lidar_pts=cfg.phantom_min_points, visibility=4, in_range=True,
                name=cfg.phantom_class, attribute="", instance_token="phantom",
                ann_token=f"phantom__{sample['sample_token']}")
    out["_phantom_cell"] = [float(box7[0]), float(box7[1])]
    assert_box_fields_consistent(out)
    return out
