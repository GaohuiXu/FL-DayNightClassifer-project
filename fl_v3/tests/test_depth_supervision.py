"""Depth-supervised LSS (BEVDepth-style) — projection-GT correctness + loss-term plumbing.

Pins: (a) the LiDAR→camera projection + depth binning lands a known point in the right feature cell/bin,
(b) the closest point wins per cell (min-depth), (c) out-of-range returns are ignored (-1), (d) CRITICALLY
``depth_loss_weight=0`` is BYTE-IDENTICAL to the pre-depth-sup loss even when depth keys are present (the
baseline-preservation contract), and the term actually fires when on. All CPU, no GPU / no full forward.
"""
from __future__ import annotations

import torch

from fl_v3.models.fusion.view_transform import DepthLSSTransform
from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.losses import CenterPointLoss
from fl_v3.models.fusion.detector import DetectorConfig


def _vt() -> DepthLSSTransform:
    return DepthLSSTransform(in_channels=128, context_channels=80, depth_bins=(1.0, 60.0, 1.0),
                             image_hw=(256, 704), feat_stride=16, cfg=BEVConfig())


# pinhole lidar2img: camera == lidar frame, fx=fy=100, principal point (352,128) → point on the optical
# axis at depth z lands at pixel (352,128) = feature cell (8, 22) with bin floor((z-1)/1).
_PINHOLE = torch.tensor([[[[100., 0, 352, 0], [0, 100., 128, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]], dtype=torch.float32)


def test_depth_targets_projection_and_binning():
    pts = torch.tensor([[0., 0., 0., 10., 0.5, 0.]], dtype=torch.float32)   # batch0, (x,y,z)=(0,0,10)
    tgt = _vt().depth_targets(pts, _PINHOLE, B=1, N=1)
    assert tgt.shape == (1, 16, 44)
    assert tgt[0, 8, 22].item() == 9                                        # floor((10-1)/1)
    assert (tgt == -1).sum().item() == 16 * 44 - 1                          # only that cell supervised


def test_depth_targets_closest_point_wins():
    pts = torch.tensor([[0., 0., 0., 10., 0.5, 0.], [0., 0., 0., 5., 0.5, 0.]], dtype=torch.float32)
    tgt = _vt().depth_targets(pts, _PINHOLE, B=1, N=1)
    assert tgt[0, 8, 22].item() == 4                                        # min-depth 5 → floor(4)


def test_depth_targets_out_of_range_ignored():
    pts = torch.tensor([[0., 0., 0., -5., 0.5, 0.],                          # behind camera
                        [0., 0., 0., 100., 0.5, 0.]], dtype=torch.float32)   # beyond dmax=60
    assert (_vt().depth_targets(pts, _PINHOLE, B=1, N=1) == -1).all()


def _synth_pred_batch():
    cfg = BEVConfig()
    H, W = cfg.head_ny, cfg.head_nx
    boxes = [[cfg.x_min + 8 * cfg.head_vx, cfg.y_min + 8 * cfg.head_vy, 0., 4., 2., 1.6, 0.3]]
    batch = {"gt_boxes": [torch.tensor(boxes)], "gt_labels": [torch.tensor([0])], "gt_velocity": [torch.zeros(1, 2)]}
    pred = {"heatmap": torch.zeros(1, 10, H, W), "reg": torch.zeros(1, 2, H, W), "height": torch.zeros(1, 1, H, W),
            "dim": torch.zeros(1, 3, H, W), "rot": torch.zeros(1, 2, H, W), "vel": torch.zeros(1, 2, H, W)}
    return pred, batch, cfg


def test_depth_loss_weight_zero_is_byte_identical():
    pred, batch, cfg = _synth_pred_batch()
    pred_d = dict(pred)                                                      # WITH depth keys present
    pred_d["depth_logits"] = torch.randn(1, 59, 16, 44)
    dt = torch.full((1, 16, 44), -1, dtype=torch.int64); dt[0, 0, 0] = 5
    pred_d["depth_target"] = dt
    off = CenterPointLoss(cfg=cfg, n_classes=10, depth_loss_weight=0.0)
    base = CenterPointLoss(cfg=cfg, n_classes=10)
    assert torch.allclose(off(pred_d, batch), base(pred, batch), atol=0, rtol=0)   # weight 0 ⇒ depth ignored
    assert off.last_terms["depth_loss"] == 0.0


def test_depth_loss_fires_when_on():
    pred, batch, cfg = _synth_pred_batch()
    pred_d = dict(pred)
    pred_d["depth_logits"] = torch.randn(1, 59, 16, 44)
    dt = torch.full((1, 16, 44), -1, dtype=torch.int64); dt[0, :4, :4] = 7
    pred_d["depth_target"] = dt
    base = CenterPointLoss(cfg=cfg, n_classes=10)(pred, batch)
    mod = CenterPointLoss(cfg=cfg, n_classes=10, depth_loss_weight=2.0)
    on = mod(pred_d, batch)
    assert on.item() > base.item() + 1e-6                                    # a positive depth CE was added
    assert mod.last_terms["depth_loss"] > 0.0


def test_detector_config_default_no_depth_sup():
    assert DetectorConfig().depth_supervision is False                       # default OFF ⇒ baseline byte-identical
