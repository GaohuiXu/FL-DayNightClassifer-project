"""Arrhenius LiDAR capability diagnostic controls."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
from fl_v3.training.tasks import _det_config_from_run
from fl_v3.utils.runtime import grad_scaler_init_scale_from_config

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from arrhenius_lidar_gap_utils import (  # noqa: E402
    BEVFUSION_075_REFERENCE,
    apply_train_policy,
    validate_bevfusion_075_parity,
)


def test_det_lidar_z_voxel_config_propagates_to_detector_config():
    cfg = _det_config_from_run({
        "precision": "fp16",
        "det-lidar-encoder": "voxel",
        "det-lidar-z-voxel": 0.2,
        "det-lidar-sparse-z-size": 41,
        "det-bev-voxel": 0.075,
        "det-pc-range": [-54, -54, -5, 54, 54, 3],
    })
    assert cfg.lidar_z_voxel == 0.2
    assert cfg.lidar_sparse_z_size == 41
    assert cfg.bev.nx == 1440
    assert cfg.bev.ny == 1440


def test_bevfusion_075_parity_helper_checks_required_contract():
    cfg = {
        "bevfusion-parity-075": True,
        "det-max-pillars": 120000,
        "bevfusion-max-voxels-reference": [120000, 160000],
        "bevfusion-head-grid-size-reference": [1440, 1440, 41],
    }
    bev = SimpleNamespace(
        x_min=-54.0,
        y_min=-54.0,
        z_min=-5.0,
        x_max=54.0,
        y_max=54.0,
        z_max=3.0,
        vx=0.075,
        vy=0.075,
        nx=1440,
        ny=1440,
        head_nx=720,
        head_ny=720,
    )
    model = SimpleNamespace(
        cfg=SimpleNamespace(bev=bev),
        lidar_encoder=SimpleNamespace(nz=41, computed_nz=40, vz=0.2),
        fusion=SimpleNamespace(camera_channels=80, lidar_channels=256, out_channels=256),
    )
    assert validate_bevfusion_075_parity(cfg, model) == []
    model.fusion.lidar_channels = 128
    errors = validate_bevfusion_075_parity(cfg, model)
    assert any("fuser lidar_channels" in e for e in errors)
    assert BEVFUSION_075_REFERENCE["sparse_shape"] == [1440, 1440, 41]


def test_lidar_only_train_policy_freezes_camera_but_keeps_fusion_trainable():
    model = BEVFusionDetector(
        DetectorConfig(
            camera_backbone="resnet18",
            pretrained_backbone=False,
            freeze_camera_backbone=False,
            neck_channels=32,
            context_channels=16,
            lidar_channels=16,
            fusion_channels=32,
            bev_neck_channels=64,
            head_channels=16,
        )
    )
    apply_train_policy(model, "lidar_only_trainable")
    assert all(not p.requires_grad for p in model.camera_backbone.parameters())
    assert all(not p.requires_grad for p in model.camera_neck.parameters())
    assert all(not p.requires_grad for p in model.view_transform.parameters())
    assert any(p.requires_grad for p in model.lidar_encoder.parameters())
    assert any(p.requires_grad for p in model.fusion.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_sparse_grad_scaler_init_scale_is_explicit_policy():
    assert grad_scaler_init_scale_from_config({"precision": "fp16"}, "fp16") == 512.0
    assert grad_scaler_init_scale_from_config(
        {
            "precision": "fp16",
            "det-lidar-encoder": "voxel",
            "det-sparse-grad-scale-init": 1.0,
        },
        "fp16",
    ) == 1.0
    assert grad_scaler_init_scale_from_config(
        {
            "precision": "fp16",
            "det-lidar-encoder": "pillar",
            "det-sparse-grad-scale-init": 1.0,
        },
        "fp16",
    ) == 512.0
    assert grad_scaler_init_scale_from_config({"precision": "fp32"}, "fp32") == 0.0
