"""ASR eligibility criterion (2): GT box visible in ≥1 camera frustum (fl_v3 T4).

A disappearance backdoor's trigger is a **camera-image** patch, so a target car is only a
valid attack target if it appears in at least one camera view. This is **trigger-invariant
geometry** — it depends only on the box + the per-camera ``lidar2img`` projection, identical
clean vs triggered (so T5 inherits the exact same eligible set, the §Attack-spec requirement).

Deterministic, pure numpy: project the box's 8 corners via each ``lidar2img`` (reusing
``transforms.project_to_image`` + ``viz.calibration.box7_to_corners``); a box is visible in a
camera if **≥ ``min_corners`` corners are in front of the camera (depth > 0) AND inside the
image bounds**. Uses the NATIVE-resolution ``lidar2img`` + image size the dataset carries
(900×1600), not the model's rescaled internal projection.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from fl_v3.data.nuscenes import transforms as T
from fl_v3.viz.calibration import box7_to_corners

# Native nuScenes image size (H, W) — the resolution the schema's lidar2img is built for.
DEFAULT_IMAGE_HW: Tuple[int, int] = (900, 1600)


def cams_visible(
    box7: Sequence[float],
    lidar2img: np.ndarray,
    image_hw: Tuple[int, int] = DEFAULT_IMAGE_HW,
    min_corners: int = 1,
    depth_min: float = 0.1,
) -> List[int]:
    """Indices of the cameras in which ``box7`` is visible (≥``min_corners`` corners in-frame).

    ``lidar2img`` is ``[6,4,4]`` (per-camera). A corner counts as in-frame if its projected
    depth > ``depth_min`` (in front) AND its pixel ``(u,v)`` lies within ``[0,W)×[0,H)``.
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    corners = box7_to_corners(np.asarray(box7, dtype=np.float64))  # (8,3) lidar frame
    l2i = np.asarray(lidar2img, dtype=np.float64).reshape(-1, 4, 4)
    out: List[int] = []
    for ci in range(l2i.shape[0]):
        uv, depth = T.project_to_image(corners, l2i[ci])
        infront = depth > depth_min
        inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if int(np.sum(infront & inimg)) >= min_corners:
            out.append(ci)
    return out


def is_frustum_visible(
    box7: Sequence[float],
    lidar2img: np.ndarray,
    image_hw: Tuple[int, int] = DEFAULT_IMAGE_HW,
    min_corners: int = 1,
    depth_min: float = 0.1,
) -> bool:
    """True iff ``box7`` is visible in ≥1 camera frustum (ASR eligibility criterion 2)."""
    return len(cams_visible(box7, lidar2img, image_hw, min_corners, depth_min)) > 0
