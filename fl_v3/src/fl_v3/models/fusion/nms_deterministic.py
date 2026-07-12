"""Deterministic, framework-independent BEV NMS for S05 CenterHead.

The official BEVFusion reference dispatches to numba circle NMS and an mmcv CUDA
rotated-NMS kernel.  Those dependencies are intentionally unavailable in fl_v3.
This module preserves the reference suppression semantics while fixing a complete
content order before suppression.  Geometry is evaluated on CPU in float64 so the
result does not depend on CUDA atomic/kernel tie behavior.

Circle thresholds are **squared metres**, matching the official implementation:
it compares ``dx**2 + dy**2 <= threshold`` without taking a square root.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def deterministic_candidate_order(
    scores: torch.Tensor,
    labels: torch.Tensor,
    spatial_indices: torch.Tensor,
) -> torch.Tensor:
    """Return O-018 order: score descending, class ID, flat spatial index.

    The sort keys are copied once to CPU and sorted with NumPy ``lexsort``.  This
    is deterministic for equal-score ties and independent of input emission order.
    """
    scores_np = scores.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)
    labels_np = labels.detach().to(device="cpu", dtype=torch.int64).numpy().reshape(-1)
    spatial_np = spatial_indices.detach().to(device="cpu", dtype=torch.int64).numpy().reshape(-1)
    if not (scores_np.size == labels_np.size == spatial_np.size):
        raise ValueError("scores, labels, and spatial_indices must have equal length")
    if not np.isfinite(scores_np).all():
        raise ValueError("candidate scores must be finite")
    order = np.lexsort((spatial_np, labels_np, -scores_np))
    return torch.as_tensor(order.copy(), dtype=torch.long, device=scores.device)


def _ordered_prefix(
    scores: torch.Tensor,
    labels: torch.Tensor,
    spatial_indices: torch.Tensor,
    pre_max_size: int,
) -> List[int]:
    if int(pre_max_size) <= 0:
        raise ValueError("pre_max_size must be positive")
    order = deterministic_candidate_order(scores, labels, spatial_indices)
    order = order[: int(pre_max_size)]
    return [int(v) for v in order.detach().cpu().tolist()]


def _validate_nms_inputs(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    spatial_indices: torch.Tensor,
    *,
    pre_max_size: int,
    post_max_size: int,
) -> None:
    """Validate the complete exported canonical-box NMS contract up front."""
    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError(f"boxes must have shape [N,>=7], got {tuple(boxes.shape)}")
    count = int(boxes.shape[0])
    if any(int(value.numel()) != count for value in (scores, labels, spatial_indices)):
        raise ValueError("boxes, scores, labels, and spatial_indices must have equal length")
    if int(pre_max_size) <= 0 or int(post_max_size) <= 0:
        raise ValueError("pre_max_size and post_max_size must be positive")
    canonical = boxes[:, :7].detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(canonical).all():
        raise ValueError("NMS boxes must contain finite canonical geometry")
    if not (canonical[:, 3:6] > 0.0).all():
        raise ValueError("NMS canonical box dimensions must be positive")


def circle_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    spatial_indices: torch.Tensor,
    *,
    threshold_sq_m: float,
    pre_max_size: int = 1000,
    post_max_size: int = 83,
) -> torch.Tensor:
    """Official task-wide circle NMS with deterministic tie handling.

    ``boxes`` uses canonical ``(x,y,z,l,w,h,yaw)``; only ``x,y`` participate.
    Candidates within ``threshold_sq_m`` of a higher-priority candidate are
    suppressed, including equality as in the reference numba implementation.
    """
    _validate_nms_inputs(
        boxes, scores, labels, spatial_indices,
        pre_max_size=pre_max_size, post_max_size=post_max_size,
    )
    threshold = float(threshold_sq_m)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("circle threshold must be non-negative squared metres")
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    xy = boxes[:, :2].detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(xy).all():
        raise ValueError("circle NMS box centres must be finite")
    order = _ordered_prefix(scores, labels, spatial_indices, pre_max_size)
    suppressed: set[int] = set()
    keep: List[int] = []
    for pos, idx in enumerate(order):
        if idx in suppressed:
            continue
        keep.append(idx)
        if len(keep) >= int(post_max_size):
            break
        dx = xy[order[pos + 1 :], 0] - xy[idx, 0]
        dy = xy[order[pos + 1 :], 1] - xy[idx, 1]
        for other, dist2 in zip(order[pos + 1 :], dx * dx + dy * dy):
            if other not in suppressed and float(dist2) <= threshold:
                suppressed.add(other)
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


Point2 = Tuple[float, float]


def _cross(a: Point2, b: Point2, c: Point2) -> float:
    """2-D cross of ``(b-a)`` and ``(c-a)``."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _box_corners(box: Sequence[float]) -> List[Point2]:
    x, y, length, width, yaw = (
        float(box[0]),
        float(box[1]),
        float(box[3]),
        float(box[4]),
        float(box[6]),
    )
    if not all(math.isfinite(v) for v in (x, y, length, width, yaw)):
        raise ValueError(f"rotate NMS box contains non-finite geometry: {box}")
    if length <= 0.0 or width <= 0.0:
        raise ValueError(f"rotate NMS box dimensions must be positive: l={length}, w={width}")
    hl, hw = 0.5 * length, 0.5 * width
    c, s = math.cos(yaw), math.sin(yaw)
    corners: List[Point2] = []
    # Counter-clockwise local order, length along +x and width along +y.
    for lx, ly in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)):
        corners.append((x + c * lx - s * ly, y + s * lx + c * ly))
    return corners


def _line_intersection(p: Point2, q: Point2, a: Point2, b: Point2) -> Point2:
    """Intersection of segment p-q with the infinite line a-b."""
    dp = _cross(a, b, p)
    dq = _cross(a, b, q)
    denom = dp - dq
    if abs(denom) <= 1e-15:
        return q
    t = dp / denom
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _clip_polygon(subject: List[Point2], clip: List[Point2]) -> List[Point2]:
    output = subject
    eps = 1e-12
    for edge_idx in range(len(clip)):
        a = clip[edge_idx]
        b = clip[(edge_idx + 1) % len(clip)]
        input_poly = output
        output = []
        if not input_poly:
            break
        prev = input_poly[-1]
        prev_inside = _cross(a, b, prev) >= -eps
        for cur in input_poly:
            cur_inside = _cross(a, b, cur) >= -eps
            if cur_inside:
                if not prev_inside:
                    output.append(_line_intersection(prev, cur, a, b))
                output.append(cur)
            elif prev_inside:
                output.append(_line_intersection(prev, cur, a, b))
            prev, prev_inside = cur, cur_inside
    return output


def _polygon_area(poly: Iterable[Point2]) -> float:
    points = list(poly)
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        acc += p[0] * q[1] - p[1] * q[0]
    return abs(acc) * 0.5


def rotated_iou_bev(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Exact convex-polygon IoU for two canonical yaw boxes in BEV."""
    a = _box_corners(box_a)
    b = _box_corners(box_b)
    inter = _polygon_area(_clip_polygon(a, b))
    if inter <= 0.0:
        return 0.0
    area_a = float(box_a[3]) * float(box_a[4])
    area_b = float(box_b[3]) * float(box_b[4])
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else min(1.0, max(0.0, inter / union))


def rotate_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    spatial_indices: torch.Tensor,
    *,
    iou_threshold: float = 0.2,
    pre_max_size: int = 1000,
    post_max_size: int = 83,
) -> torch.Tensor:
    """Task-wide deterministic rotated BEV NMS.

    Suppression is deliberately task-wide, matching the official CenterHead.  The
    caller may apply the official class-specific dimension scales before calling.
    """
    _validate_nms_inputs(
        boxes, scores, labels, spatial_indices,
        pre_max_size=pre_max_size, post_max_size=post_max_size,
    )
    threshold = float(iou_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("rotate IoU threshold must lie in [0,1]")
    if boxes.shape[0] == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    boxes_np = boxes[:, :7].detach().to(device="cpu", dtype=torch.float64).numpy()
    order = _ordered_prefix(scores, labels, spatial_indices, pre_max_size)
    suppressed: set[int] = set()
    keep: List[int] = []
    for pos, idx in enumerate(order):
        if idx in suppressed:
            continue
        keep.append(idx)
        if len(keep) >= int(post_max_size):
            break
        for other in order[pos + 1 :]:
            if other in suppressed:
                continue
            if rotated_iou_bev(boxes_np[idx], boxes_np[other]) > threshold:
                suppressed.add(other)
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


__all__ = [
    "deterministic_candidate_order",
    "circle_nms",
    "rotate_nms",
    "rotated_iou_bev",
]
