"""Arrhenius camera-branch audit controls."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
from fl_v3.models.fusion.view_transform import DepthLSSTransform

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from arrhenius_lidar_gap_utils import (  # noqa: E402
    CAPABILITY_MATRIX_CELLS,
    TRAIN_POLICIES,
    apply_train_policy,
    camera_model_summary,
    tensor_stats,
    validate_controls,
)
from arrhenius_mini_matrix import _forward_with_branch_topology  # noqa: E402


def _small_cfg(**overrides):
    base = dict(
        camera_backbone="resnet18",
        pretrained_backbone=False,
        freeze_camera_backbone=True,
        neck_channels=32,
        context_channels=16,
        lidar_channels=16,
        fusion_channels=32,
        bev_neck_channels=64,
        head_channels=16,
    )
    base.update(overrides)
    return DetectorConfig(**base)


def test_camera_only_train_policy_freezes_lidar_and_trains_camera_path():
    assert "camera_only_trainable" in TRAIN_POLICIES
    assert validate_controls("camera_only", "camera_only_trainable") == (
        "camera_only",
        "camera_only_trainable",
    )
    model = BEVFusionDetector(_small_cfg())
    assert all(not p.requires_grad for p in model.camera_backbone.parameters())

    apply_train_policy(model, "camera_only_trainable")

    assert any(p.requires_grad for p in model.camera_backbone.parameters())
    assert any(p.requires_grad for p in model.camera_neck.parameters())
    assert any(p.requires_grad for p in model.view_transform.parameters())
    assert all(not p.requires_grad for p in model.lidar_encoder.parameters())
    assert any(p.requires_grad for p in model.fusion.parameters())
    assert any(p.requires_grad for p in model.bev_neck.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_swin_camera_model_summary_records_camera_contract():
    model = BEVFusionDetector(
        _small_cfg(
            camera_backbone="swin_t",
            pretrained_backbone=False,
            freeze_camera_backbone=False,
            context_channels=8,
            fusion_channels=16,
            bev_neck_channels=32,
            head_channels=8,
        )
    )
    summary = camera_model_summary(model)
    assert summary["backbone"] == "swin_t"
    assert summary["pretrained_backbone"] is False
    assert summary["camera_init_policy"] == "scratch"
    assert summary["backbone_out_channels"] == [96, 192, 384, 768]
    assert summary["backbone_strides"] == [4, 8, 16, 32]
    assert summary["context_channels"] == 8
    assert summary["fuser"]["camera_channels"] == 8
    assert summary["camera_bev_grid"] == [256, 256]
    assert summary["head_heatmap_grid"] == [128, 128]


def test_camera_075_trainable_cell_keeps_swin_camera_only_contract():
    cell = CAPABILITY_MATRIX_CELLS["camera_iso_075_ch256_fp16_swin"]
    assert cell["branch_topology"] == "camera_only"
    assert cell["train_policy"] == "camera_only_trainable"
    assert cell["det-camera-backbone"] == "swin_t"
    assert cell["det-lidar-encoder"] == "pillar"
    assert cell["det-bev-voxel"] == 0.075
    assert cell["det-fusion-channels"] == 256


def test_tensor_stats_reports_variance_and_nonzero_ratio():
    stats = tensor_stats(torch.tensor([[0.0, 2.0], [0.0, -2.0]]))
    assert stats["finite"]
    assert stats["numel"] == 4
    assert stats["nonzero"] == 2
    assert stats["nonzero_ratio"] == 0.5
    assert stats["variance"] > 0.0


def test_view_transform_projection_meta_records_valid_ratios():
    cfg = BEVConfig(
        point_cloud_range=(0.0, 0.0, 0.0, 100.0, 100.0, 10.0),
        bev_voxel=(1.0, 1.0),
        out_size_factor=2,
    )
    vt = DepthLSSTransform(
        in_channels=4,
        context_channels=2,
        depth_bins=(1.0, 3.0, 1.0),
        image_hw=(32, 32),
        feat_stride=16,
        cfg=cfg,
    )
    vt.record_debug = True
    feat = torch.randn(2, 4, 2, 2)
    lidar2img = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)

    out = vt(feat, lidar2img, B=1, N=2)

    meta = out["projection_meta"]
    assert meta["frustum_points_per_camera"] == 8
    assert meta["frustum_points_total"] == 16
    assert meta["valid_points_total"] == 16
    assert meta["valid_ratio"] == 1.0
    assert meta["valid_points_per_camera"] == [8, 8]
    assert meta["valid_ratio_per_camera"] == [1.0, 1.0]
    assert vt.last_projection_meta == meta
    assert out["bev"].shape == (1, 2, 100, 100)


class _ToyCameraBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return [x.new_ones((x.shape[0], 1, 2, 2))]


class _ToyCameraNeck(nn.Module):
    def forward(self, feats):
        return feats[0]


class _ToyViewTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_projection_meta = {}

    def forward(self, feat, lidar2img, B, N):
        self.calls += 1
        bev = feat.new_ones((B, 2, 2, 2))
        self.last_projection_meta = {"valid_ratio": 1.0, "valid_points_per_camera": [4] * N}
        return {
            "bev": bev,
            "depth_prob": feat.new_ones((B * N, 1, 2, 2)),
            "context": feat.new_ones((B * N, 2, 2, 2)),
            "projection_meta": self.last_projection_meta,
        }


class _ToyLidarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, points, B):
        self.calls += 1
        return points.new_full((B, 3, 2, 2), 2.0)


class _ToyFusion(nn.Module):
    camera_channels = 2
    lidar_channels = 3
    out_channels = 5

    def forward(self, camera_bev, lidar_bev):
        return torch.cat([camera_bev, lidar_bev], dim=1)


class _ToyHead(nn.Module):
    def forward(self, x):
        heat = x.sum(dim=1, keepdim=True)
        z1 = x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        z2 = x.new_zeros((x.shape[0], 2, x.shape[2], x.shape[3]))
        z3 = x.new_zeros((x.shape[0], 3, x.shape[2], x.shape[3]))
        return {"heatmap": heat, "reg": z2, "height": z1, "dim": z3, "rot": z2, "vel": z2}


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(bev=SimpleNamespace(ny=2, nx=2))
        self.preprocess = lambda images, lidar2img, cam_intrinsics: {
            "images": images.float(),
            "lidar2img": lidar2img,
            "cam_intrinsics": cam_intrinsics,
        }
        self.camera_backbone = _ToyCameraBackbone()
        self.camera_neck = _ToyCameraNeck()
        self.view_transform = _ToyViewTransform()
        self.lidar_encoder = _ToyLidarEncoder()
        self.lidar_backbone = None
        self.fusion = _ToyFusion()
        self.bev_neck = nn.Identity()
        self.head = _ToyHead()


def _toy_batch():
    return {
        "images": torch.zeros((1, 2, 3, 4, 4), dtype=torch.uint8),
        "lidar2img": torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1),
        "cam_intrinsics": torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1),
        "lidar_points": torch.zeros((1, 6), dtype=torch.float32),
        "batch_size": 1,
        "cam_order": [("CAM_A", "CAM_B")],
    }


def test_branch_topology_skips_and_zeroes_expected_inputs():
    batch = _toy_batch()

    camera_model = _ToyModel()
    camera_out, camera_stats = _forward_with_branch_topology(camera_model, batch, "camera_only")
    assert camera_model.lidar_encoder.calls == 0
    assert camera_stats["executed"]["camera"] is True
    assert camera_stats["executed"]["lidar_encoder"] is False
    assert camera_stats["lidar_bev"]["nonzero"] == 0

    lidar_model = _ToyModel()
    lidar_out, lidar_stats = _forward_with_branch_topology(lidar_model, batch, "lidar_only")
    assert lidar_model.camera_backbone.calls == 0
    assert lidar_stats["executed"]["camera"] is False
    assert lidar_stats["executed"]["lidar_encoder"] is True
    assert lidar_stats["camera_bev"]["nonzero"] == 0

    full_model = _ToyModel()
    full_out, _ = _forward_with_branch_topology(full_model, batch, "full_fusion")
    zero_lidar_out, zero_lidar_stats = _forward_with_branch_topology(
        _ToyModel(), batch, "full_fusion", zero_lidar=True
    )
    zero_camera_out, zero_camera_stats = _forward_with_branch_topology(
        _ToyModel(), batch, "full_fusion", zero_camera=True
    )

    assert torch.equal(camera_out["heatmap"], zero_lidar_out["heatmap"])
    assert torch.equal(lidar_out["heatmap"], zero_camera_out["heatmap"])
    assert not torch.equal(full_out["heatmap"], zero_lidar_out["heatmap"])
    assert not torch.equal(full_out["heatmap"], zero_camera_out["heatmap"])
    assert zero_lidar_stats["zero_lidar_at_fusion"] is True
    assert zero_camera_stats["zero_camera_at_fusion"] is True
