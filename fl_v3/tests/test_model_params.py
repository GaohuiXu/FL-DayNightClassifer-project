"""T2 per-module parameter accounting (the Q2-dilution seed)."""
from __future__ import annotations

from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
from fl_v3.models.fusion.camera_backbone import SWIN_T_PARAMS, RESNET18_PARAMS


def test_param_table_structure_and_frozen_backbone():
    cfg = DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False)
    model = BEVFusionDetector(cfg)
    table = model.param_table()
    for name in ("preprocess", "camera_backbone", "camera_neck", "view_transform",
                 "lidar_encoder", "fusion", "bev_neck", "head"):
        assert name in table
    # frozen backbone: zero trainable, all params frozen
    assert table["camera_backbone"]["trainable"] == 0
    assert table["camera_backbone"]["frozen"] == table["camera_backbone"]["total"] > 0
    # fusion is a real trainable sub-network (the named module for T5/T6)
    assert table["fusion"]["trainable"] > 0
    assert table["head"]["trainable"] > 0
    # preprocess is parameter-free
    assert table["preprocess"]["total"] == 0


def test_fusion_fraction_of_trainable_reported():
    cfg = DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False)
    model = BEVFusionDetector(cfg)
    table = model.param_table()
    total_trainable = sum(v["trainable"] for v in table.values())
    fusion_frac = table["fusion"]["trainable"] / total_trainable
    # the dilution seed: a single small sub-network is a minority of trainable params
    assert 0.0 < fusion_frac < 0.95
    # backbone is the param-dominant module but contributes ZERO trainable (D1)
    assert table["camera_backbone"]["trainable"] == 0


def test_backbone_full_param_counts_canonical():
    """Full torchvision backbone counts == canonical (the weights-loaded check)."""
    r = BEVFusionDetector(DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False))
    assert r.camera_backbone.full_backbone_params == RESNET18_PARAMS
    # Swin count is architecture-determined (same with/without pretrained weights).
    s = BEVFusionDetector(DetectorConfig(camera_backbone="swin_t", pretrained_backbone=False))
    assert s.camera_backbone.full_backbone_params == SWIN_T_PARAMS
