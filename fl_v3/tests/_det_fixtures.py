"""Shared synthetic T1-schema detection sample for the T2 model tests (RNG-seeded)."""
from __future__ import annotations

import torch


def fake_detection_sample(seed: int, M: int = 4, n_pts: int = 4000) -> dict:
    """A synthetic sample matching the frozen T1 canonical schema (deterministic)."""
    g = torch.Generator().manual_seed(seed)
    s: dict = {}
    s["images"] = (torch.rand(6, 3, 900, 1600, generator=g) * 255).to(torch.uint8)
    K = torch.tensor([[800.0, 0, 800.0], [0, 800.0, 450.0], [0, 0, 1.0]])
    s["cam_intrinsics"] = K.view(1, 3, 3).repeat(6, 1, 1)
    l2i = torch.eye(4).view(1, 4, 4).repeat(6, 1, 1)
    l2i[:, :3, :3] = K
    s["lidar2img"] = l2i
    s["cam2ego"] = torch.eye(4).view(1, 4, 4).repeat(6, 1, 1)
    s["ego2global_cam"] = torch.eye(4).view(1, 4, 4).repeat(6, 1, 1)
    s["lidar2ego"] = torch.eye(4)
    s["ego2global_lidar"] = torch.eye(4)
    xyz = (torch.rand(n_pts, 3, generator=g) * 2 - 1) * torch.tensor([45.0, 45.0, 3.0])
    s["lidar_points"] = torch.cat([xyz, torch.rand(n_pts, 1, generator=g), torch.zeros(n_pts, 1)], dim=1)
    boxes = torch.zeros(M, 7)
    boxes[:, :2] = (torch.rand(M, 2, generator=g) * 2 - 1) * 30
    boxes[:, 2] = -1.0
    boxes[:, 3:6] = torch.tensor([4.0, 2.0, 1.6])
    boxes[:, 6] = torch.rand(M, generator=g) * 3.14
    s["gt_boxes"] = boxes
    s["gt_velocity"] = torch.zeros(M, 2)
    s["gt_labels"] = torch.zeros(M, dtype=torch.int64)
    s["gt_names"] = ["car"] * M
    s["gt_num_lidar_pts"] = torch.full((M,), 50, dtype=torch.int64)
    s["gt_visibility"] = torch.full((M,), 4, dtype=torch.int64)
    s["gt_in_range"] = torch.ones(M, dtype=torch.bool)
    s["gt_attribute"] = [""] * M
    s["gt_instance_tokens"] = ["i"] * M
    s["gt_ann_tokens"] = [f"a{j}" for j in range(M)]
    s["sample_token"] = f"tok{seed}"
    s["scene_token"] = "sc"
    s["log_token"] = "lg"
    s["location"] = "boston-seaport"
    s["timestamp"] = 0
    s["cam_order"] = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    s["num_boxes"] = M
    return s
