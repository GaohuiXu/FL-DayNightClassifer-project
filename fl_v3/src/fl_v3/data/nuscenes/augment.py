"""Deterministic BEV/3D global augmentation (MCR P1 — the overfitting fix).

The 40-epoch convergence diagnostic showed the model peaks at ~ep20 (mAP 0.4702) then OVERFITS
(val ↓ while train loss ↓45%), with the heatmap/classification head losing generalization hardest on
the rare classes — the signature of a 28M-param backbone trained with ZERO augmentation. This module is
the standard fix: **GlobalRotScaleTrans + RandomFlip3D**, applied CONSISTENTLY to

  * the LiDAR points (xyz),
  * the GT boxes (center, dims, yaw) + velocity,
  * AND the camera projection ``lidar2img`` (so the LSS camera-BEV stays aligned with the augmented
    LiDAR-BEV — mandatory for a fusion model; the image pixels are unchanged).

The scene transform ``T`` (4×4, isotropic-scale · flip · yaw-rotation, then translation) maps the ORIGINAL
LiDAR frame → the AUGMENTED frame the model trains in. Points/boxes are pushed forward by ``T``; the camera
mapping is pulled back: a point ``p_aug = T·p`` must still project to the same pixel, so
``lidar2img_aug = lidar2img · T⁻¹``.

**TRAIN-ONLY** (eval is clean — the dataset only wires this when ``augment`` is set). **Determinism:** uses
``numpy.random``, which ``seeded_worker_init`` seeds per-worker-per-epoch from the DataLoader generator, so
the augmented batch is a reproducible function of (seed, epoch) under the D16 relaxed regime. AST-irrelevant
(lives in the data loader, not ``models/fusion/**``)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

# Default knobs (BEVFusion GlobalRotScaleTrans/RandomFlip3D values — proven on nuScenes).
DEFAULT_AUG = {
    "rot": 0.785398,        # ± yaw rotation (rad) ≈ ±45°
    "scale": (0.9, 1.1),    # isotropic scale range
    "translate": 0.5,       # per-axis Gaussian translation std (m)
    "flip": True,           # independent x- and y-axis flips at p=0.5 each
}


def sample_transform(p: Dict) -> np.ndarray:
    """Sample a 4×4 scene transform T (float32) from the aug params, using the (seeded) numpy RNG."""
    smin, smax = p.get("scale", (1.0, 1.0))
    tr = float(p.get("translate", 0.0))
    reference_rng = (
        "horizontal_flip_probability" in p or "vertical_flip_probability" in p
    )
    if reference_rng:
        # Preserve the pinned pipeline's RNG order: scale, rotation,
        # translation, horizontal flip, vertical flip.
        scale = float(np.random.uniform(smin, smax))
        theta = (
            float(np.random.uniform(-p["rot"], p["rot"]))
            if p.get("rot", 0)
            else 0.0
        )
        translation = (
            np.random.normal(0.0, tr, 3).astype(np.float64)
            if tr
            else np.zeros(3, np.float64)
        )
        # mmdet3d terminology: horizontal negates y; vertical negates x.
        hp = float(p.get("horizontal_flip_probability", 0.0))
        vp = float(p.get("vertical_flip_probability", 0.0))
        if not (0.0 <= hp <= 1.0 and 0.0 <= vp <= 1.0):
            raise ValueError("3D flip probabilities must lie in [0,1]")
        # The exact reference uses np.random.choice([0, 1]) at probability 0.5.
        horizontal = (
            bool(np.random.choice([0, 1]))
            if hp == 0.5
            else bool(np.random.rand() < hp)
        )
        vertical = (
            bool(np.random.choice([0, 1]))
            if vp == 0.5
            else bool(np.random.rand() < vp)
        )
        fx = -1.0 if vertical else 1.0
        fy = -1.0 if horizontal else 1.0
    else:
        # Keep the historical local path byte-for-byte compatible.  Phase I
        # always takes the explicit-probability branch above.
        theta = (
            float(np.random.uniform(-p["rot"], p["rot"]))
            if p.get("rot", 0)
            else 0.0
        )
        scale = float(np.random.uniform(smin, smax))
        fx = -1.0 if (p.get("flip") and np.random.rand() < 0.5) else 1.0   # flip across y-axis (negate x)
        fy = -1.0 if (p.get("flip") and np.random.rand() < 0.5) else 1.0   # flip across x-axis (negate y)
        translation = (
            np.random.normal(0.0, tr, 3).astype(np.float64)
            if tr
            else np.zeros(3, np.float64)
        )
    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    flip = np.diag([fx, fy, 1.0])
    L = flip @ (scale * Rz)                          # linear part: flip · scale · Rz
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = L
    T[:3, 3] = (
        flip @ (translation * scale)
        if reference_rng
        else translation
    )
    return T


_ALIGNED_GT_FIELDS = (
    "gt_boxes",
    "gt_velocity",
    "gt_labels",
    "gt_num_lidar_pts",
    "gt_visibility",
    "gt_in_range",
    "gt_names",
    "gt_attribute",
    "gt_instance_tokens",
    "gt_ann_tokens",
)


def _filter_aligned_gt(sample: Dict, mask: torch.Tensor) -> None:
    """Apply one object mask to every row-aligned annotation field."""
    count = int(sample["gt_boxes"].shape[0])
    if mask.dtype != torch.bool or tuple(mask.shape) != (count,):
        raise ValueError("GT filter mask is not row-aligned")
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    for field in _ALIGNED_GT_FIELDS:
        value = sample[field]
        if len(value) != count:
            raise ValueError(f"annotation field {field!r} is not row-aligned")
        if torch.is_tensor(value):
            sample[field] = value[mask]
        else:
            sample[field] = [value[index] for index in indices]
    sample["num_boxes"] = len(indices)


def apply_reference_post_filters(sample: Dict, params: Dict) -> Dict:
    """Pinned Points/ObjectRangeFilter, ObjectNameFilter, and PointShuffle."""
    point_range = params.get("point_cloud_range")
    class_names = params.get("object_classes")
    if point_range is None or class_names is None:
        return sample
    bounds = torch.as_tensor(point_range, dtype=torch.float32)
    if tuple(bounds.shape) != (6,) or not bool((bounds[3:] > bounds[:3]).all()):
        raise ValueError("reference augmentation point_cloud_range is invalid")
    if "lidar_points" in sample:
        points = sample["lidar_points"]
        mask = (
            (points[:, :3] > bounds[:3]).all(dim=1)
            & (points[:, :3] < bounds[3:]).all(dim=1)
        )
        points = points[mask]
        if bool(params.get("point_shuffle", False)) and points.shape[0] > 1:
            points = points[torch.randperm(points.shape[0])]
        sample["lidar_points"] = points

    boxes = sample["gt_boxes"]
    object_mask = (
        (boxes[:, 0] > bounds[0])
        & (boxes[:, 0] < bounds[3])
        & (boxes[:, 1] > bounds[1])
        & (boxes[:, 1] < bounds[4])
    )
    allowed = {str(name) for name in class_names}
    name_mask = torch.tensor(
        [str(name) in allowed for name in sample["gt_names"]], dtype=torch.bool
    )
    _filter_aligned_gt(sample, object_mask & name_mask)
    if sample["gt_boxes"].numel():
        yaw = sample["gt_boxes"][:, 6]
        sample["gt_boxes"][:, 6] = yaw - torch.floor(
            yaw / (2.0 * np.pi) + 0.5
        ) * (2.0 * np.pi)
    return sample


def apply_transform(points: torch.Tensor, gt_boxes: torch.Tensor, gt_velocity: torch.Tensor,
                    lidar2img: torch.Tensor, T: np.ndarray):
    """Apply the 4×4 scene transform T to points / boxes / velocity / lidar2img (consistent).

    ``points`` [P, C] (xyz in cols 0:3, rest carried); ``gt_boxes`` [M,7] = (cx,cy,cz,dx,dy,dz,yaw);
    ``gt_velocity`` [M,2]; ``lidar2img`` [6,4,4]. Returns transformed copies (same dtypes/shapes)."""
    Tt = torch.from_numpy(T.astype(np.float32))
    L = Tt[:3, :3]                      # flip·scale·Rz
    t = Tt[:3, 3]
    scale = float(np.cbrt(np.abs(np.linalg.det(T[:3, :3]))))   # isotropic scale = |det L|^(1/3)
    R2 = (L[:2, :2] / scale)            # rotation+flip part only (unit), for the heading direction

    # --- points: xyz ← L·xyz + t (extra cols intensity/ring/dt untouched) ---
    pts = points.clone()
    pts[:, :3] = points[:, :3] @ L.T + t

    # --- boxes ---
    boxes = gt_boxes.clone()
    if boxes.numel():
        boxes[:, :3] = gt_boxes[:, :3] @ L.T + t          # center is a point
        boxes[:, 3:6] = gt_boxes[:, 3:6] * scale          # dims scale isotropically (flip/rot don't change extent)
        yaw = gt_boxes[:, 6]
        d = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)   # heading direction [M,2]
        d = d @ R2.T
        boxes[:, 6] = torch.atan2(d[:, 1], d[:, 0])       # robust to rot AND flip handedness
    vel = gt_velocity.clone()
    if vel.numel():
        vel[:, :2] = gt_velocity[:, :2] @ L[:2, :2].T     # velocity scales+rotates+flips with the scene

    # --- camera: lidar2img ← lidar2img · T⁻¹ (pull-back so p_aug projects to the same pixel) ---
    Tinv = torch.from_numpy(np.linalg.inv(T).astype(np.float32))
    l2i = torch.matmul(lidar2img, Tinv)                   # [6,4,4] @ [4,4] broadcast

    return pts, boxes, vel, l2i


def apply_image_flip(images: torch.Tensor, lidar2img: torch.Tensor):
    """Horizontal-flip ALL camera images + update lidar2img so the geometry stays consistent.

    Unlike the BEV scene transform, this perturbs only the camera APPEARANCE — GT boxes/points are
    UNCHANGED. It regularizes because the Swin backbone is NOT flip-equivariant (the LSS lift samples the
    flipped FEATURE map, not the raw pixel), while the lift geometry stays correct via the lidar2img
    update. ``images`` [6,3,H,W]; ``lidar2img`` [6,4,4] in the SAME (native) pixel frame the images use.
    Pixel map u → (W-1)-u ⇒ row0' = (W-1)·row2 − row0 (row2 is the projective depth divisor)."""
    W = images.shape[-1]
    imgs = torch.flip(images, dims=[-1])
    l2i = lidar2img.clone()
    l2i[:, 0, :] = (W - 1) * lidar2img[:, 2, :] - lidar2img[:, 0, :]
    return imgs, l2i


def augment_sample(sample: Dict, params: Dict) -> Dict:
    """Apply BEV/3D aug (+ optional image flip) to a T1 schema ``sample`` in place (returns it). Train-only."""
    T = sample_transform(params)
    had_points = "lidar_points" in sample
    points = (
        sample["lidar_points"]
        if had_points
        else torch.empty((0, 5), dtype=torch.float32)
    )
    pts, boxes, vel, l2i = apply_transform(
        points, sample["gt_boxes"], sample["gt_velocity"], sample["lidar2img"], T)
    if had_points:
        sample["lidar_points"] = pts
    sample["gt_boxes"] = boxes
    sample["gt_velocity"] = vel
    sample["lidar2img"] = l2i
    # Image appearance aug (independent of the scene transform; commutes — left-side image rows vs the
    # right-side lidar2img·T⁻¹). GT untouched. Gated by img_flip probability (0 ⇒ off).
    if params.get("img_flip", 0.0) and np.random.rand() < params["img_flip"]:
        sample["images"], sample["lidar2img"] = apply_image_flip(sample["images"], sample["lidar2img"])
    return apply_reference_post_filters(sample, params)
