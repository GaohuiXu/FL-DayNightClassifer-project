"""Synthetic T1-schema sample with a PROJECTIVE 6-camera rig + LiDAR clusters inside the GT boxes,
so the trigger placement / poison / ablation tests exercise real frustum geometry (T5)."""
from __future__ import annotations

import numpy as np
import torch

NATIVE_HW = (900, 1600)


def _cam_l2i(yaw_deg: float) -> np.ndarray:
    """A pinhole camera looking along the lidar +x direction rotated by ``yaw_deg`` about +z."""
    th = np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    Rc = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]]) @ Rz.T   # cam looks along rotated +x
    K = np.array([[800.0, 0, NATIVE_HW[1] / 2], [0, 800.0, NATIVE_HW[0] / 2], [0, 0, 1.0]])
    l2c = np.eye(4); l2c[:3, :3] = Rc
    vp = np.eye(4); vp[:3, :3] = K
    return vp @ l2c


def attack_sample(seed: int = 0, boxes=None) -> dict:
    """A deterministic sample: 6 projective cams, 2 car boxes ahead with dense enclosed LiDAR clusters."""
    rng = np.random.default_rng(seed)
    H, W = NATIVE_HW
    l2i = np.stack([_cam_l2i(d) for d in (0, 60, 120, 180, 240, 300)]).astype(np.float32)
    if boxes is None:
        boxes = np.array([[15.0, 0.0, 0.0, 4.5, 2.0, 1.6, 0.0],
                          [12.0, 3.0, 0.0, 4.5, 2.0, 1.6, 0.3]], dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    M = boxes.shape[0]
    clusters = [boxes[m, :3] + 0.25 * rng.standard_normal((40, 3)) for m in range(M)]
    sparse = np.array([[40.0, 40.0, 0.0], [-30.0, 20.0, 1.0]])
    pts = np.concatenate(clusters + [sparse]).astype(np.float32)
    pts5 = np.concatenate([pts, np.zeros((pts.shape[0], 2), np.float32)], axis=1)
    return {
        "sample_token": f"tok{seed}", "scene_token": "s", "log_token": "l", "location": "boston",
        "timestamp": 0, "cam_order": tuple(f"C{i}" for i in range(6)),
        "images": torch.from_numpy(rng.integers(0, 255, (6, 3, H, W), dtype=np.uint8)),
        "cam_intrinsics": torch.zeros(6, 3, 3), "lidar2img": torch.from_numpy(l2i),
        "cam2ego": torch.zeros(6, 4, 4), "ego2global_cam": torch.zeros(6, 4, 4),
        "lidar_points": torch.from_numpy(pts5), "lidar2ego": torch.eye(4), "ego2global_lidar": torch.eye(4),
        "gt_boxes": torch.from_numpy(boxes), "gt_velocity": torch.zeros(M, 2),
        "gt_labels": torch.zeros(M, dtype=torch.int64), "gt_names": ["car"] * M,
        "gt_num_lidar_pts": torch.full((M,), 40, dtype=torch.int64),
        "gt_visibility": torch.full((M,), 4, dtype=torch.int64),
        "gt_in_range": torch.ones(M, dtype=torch.bool), "gt_attribute": [""] * M,
        "gt_instance_tokens": [f"i{m}" for m in range(M)], "gt_ann_tokens": [f"a{m}" for m in range(M)],
        "num_boxes": M,
    }
