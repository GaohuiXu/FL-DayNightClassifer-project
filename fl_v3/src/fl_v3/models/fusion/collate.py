"""Detection collate (fl_v3 T2) — RNG-free batching of the T1 canonical schema.

Stacks the fixed-shape per-camera tensors ``[B,6,…]`` and the per-keyframe ``[B,4,4]``
calibration; when LiDAR is enabled, **concatenates points with a leading batch-index
column**. Single-sweep rows are ``(batch_idx,x,y,z,intensity,ring)`` and multi-sweep
rows append ``dt``. Camera-only batches contain no LiDAR payload and LiDAR-only
batches contain no image payload. Ragged per-box GT remains in per-sample lists.
The operation is RNG-free and preserves dataset order.

> **Declared contract (T2↔T3):** LiDAR points are the **batch-index-column** form, NOT a
> list of per-sample arrays. Column 0 is the 0-based batch index; the remaining
> columns are the T1 LIDAR_TOP features, including ``dt`` when sweeps are enabled.
"""
from __future__ import annotations

from typing import List

import torch

# Per-camera / per-keyframe tensor fields that stack to a fixed shape.
_STACK_FIELDS = (
    "cam_intrinsics", "lidar2img", "cam2ego", "ego2global_cam",
    "lidar2ego", "ego2global_lidar",
)
# Ragged per-box fields kept as per-sample lists.
_RAGGED_FIELDS = (
    "gt_boxes", "gt_velocity", "gt_labels", "gt_num_lidar_pts", "gt_visibility",
    "gt_in_range", "gt_names", "gt_attribute", "gt_instance_tokens", "gt_ann_tokens",
)
# Per-sample scalar / identity fields collected into lists.
_SCALAR_FIELDS = (
    "sample_token", "scene_token", "log_token", "location", "timestamp",
    "cam_order", "num_boxes", "model_mode",
)


def detection_collate_fn(batch: List[dict]) -> dict:
    """Collate a list of T1 schema samples into one detection batch (see module doc)."""
    out: dict = {}
    for f in _STACK_FIELDS:
        out[f] = torch.stack([s[f] for s in batch], dim=0)

    modes = {s.get("model_mode", "fusion") for s in batch}
    if len(modes) > 1:
        raise ValueError(f"cannot collate mixed model modes: {sorted(modes)}")
    mode = next(iter(modes), "fusion")
    if mode in {"camera_only", "fusion"}:
        if any("images" not in sample for sample in batch):
            raise KeyError(f"{mode} batch is missing enabled camera payload")
        out["images"] = torch.stack([s["images"] for s in batch], dim=0)
    elif any("images" in sample for sample in batch):
        raise ValueError("lidar_only sample unexpectedly contains camera payload")

    # LiDAR points → [TotalP, 1+W] with a leading batch-index column. W=5 single-sweep
    # (x,y,z,intensity,ring) or W=6 multi-sweep (…,dt) — the concat is width-agnostic.
    if mode in {"lidar_only", "fusion"}:
        if any("lidar_points" not in sample for sample in batch):
            raise KeyError(f"{mode} batch is missing enabled LiDAR payload")
        pts_list = []
        for b, s in enumerate(batch):
            p = s["lidar_points"]
            bcol = torch.full((p.shape[0], 1), float(b), dtype=p.dtype)
            pts_list.append(torch.cat([bcol, p], dim=1))
        width = batch[0]["lidar_points"].shape[1] if batch else 5
        out["lidar_points"] = (
            torch.cat(pts_list, dim=0) if pts_list else torch.zeros((0, 1 + width))
        )
    elif any("lidar_points" in sample for sample in batch):
        raise ValueError("camera_only sample unexpectedly contains LiDAR payload")

    for f in _RAGGED_FIELDS:
        out[f] = [s[f] for s in batch]  # per-sample (tensors or str lists)
    for f in _SCALAR_FIELDS:
        out[f] = [s[f] for s in batch]
    out["batch_size"] = len(batch)
    return out
