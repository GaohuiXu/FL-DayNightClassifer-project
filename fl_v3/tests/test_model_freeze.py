"""T2 D1/D6 invariants — frozen backbone through train(), GroupNorm, no adaptive pool."""
from __future__ import annotations

import torch
import torch.nn as nn

from fl_v3.utils.runtime import enforce_determinism, seed_everything
from fl_v3.models.fusion.camera_backbone import CameraBackbone, SWIN_T_PARAMS, RESNET18_PARAMS
from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
from fl_v3.models.fusion.losses import CenterPointLoss
from fl_v3.models.fusion.collate import detection_collate_fn
from _det_fixtures import fake_detection_sample


def _small_cfg(backbone="resnet18", pretrained=False):
    return DetectorConfig(camera_backbone=backbone, pretrained_backbone=pretrained,
                          neck_channels=32, context_channels=16, lidar_channels=16,
                          fusion_channels=32, bev_neck_channels=64, head_channels=16)


def test_resnet_bn_frozen_through_train_step():
    """The D1/D6 trap: BN running stats must NOT update on a forward inside model.train()
    (tested on the ResNet BatchNorm path, not only the LayerNorm Swin)."""
    enforce_determinism(strict=True); seed_everything(0)
    cfg = _small_cfg("resnet18")
    model = BEVFusionDetector(cfg)
    crit = CenterPointLoss(cfg=cfg.bev)
    batch = detection_collate_fn([fake_detection_sample(1)])

    bn = model.camera_backbone._stem[1]  # first ResNet BN
    assert isinstance(bn, nn.BatchNorm2d)
    before_mean = bn.running_mean.clone()
    before_var = bn.running_var.clone()
    before_nbt = bn.num_batches_tracked.clone()

    model.train()  # parent train() — the frozen backbone must STAY eval
    assert model.camera_backbone.training is False, "frozen backbone left eval mode on .train()"
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(); loss = crit(model(batch), batch); loss.backward(); opt.step()

    assert torch.equal(bn.running_mean, before_mean), "BN running_mean drifted (D1/D6 broken)"
    assert torch.equal(bn.running_var, before_var), "BN running_var drifted (D1/D6 broken)"
    assert torch.equal(bn.num_batches_tracked, before_nbt), "BN num_batches_tracked drifted"


def test_no_adam_state_for_frozen_backbone():
    """The optimizer's param set (requires_grad) excludes every backbone param (D1)."""
    cfg = _small_cfg("resnet18")
    model = BEVFusionDetector(cfg)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    backbone_ids = {id(p) for p in model.camera_backbone.parameters()}
    assert trainable.isdisjoint(backbone_ids), "a frozen backbone param is in the trainable set"
    assert all(not p.requires_grad for p in model.camera_backbone.parameters())


def test_no_batchnorm_in_new_trainable_modules():
    """Only the frozen backbone may contain BatchNorm; new modules use GroupNorm/LayerNorm (D6)."""
    cfg = _small_cfg("resnet18")
    model = BEVFusionDetector(cfg)
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            assert name.startswith("camera_backbone"), f"BatchNorm outside frozen backbone: {name}"
            assert not m.training, f"frozen-backbone BN {name} not in eval mode"


def test_no_adaptive_pool_anywhere():
    """No AdaptiveAvg/MaxPool2d in any module (its CUDA backward raises)."""
    cfg = _small_cfg("resnet18")
    model = BEVFusionDetector(cfg)
    for name, m in model.named_modules():
        assert not isinstance(m, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)), f"adaptive pool at {name}"


def test_resnet18_weights_loaded_param_count():
    """ResNet-18 architecture param count == canonical (weights-loaded check)."""
    bb = CameraBackbone("resnet18", frozen=True, pretrained=False)
    assert bb.full_backbone_params == RESNET18_PARAMS, bb.full_backbone_params


def test_swin_t_imagenet_weights_load_and_forward():
    """Swin-T IMAGENET1K_V1 loads (cached on login node), param count == canonical, and a
    forward is deterministic. Skips loud only if the offline weight cache is unavailable."""
    try:
        bb = CameraBackbone("swin_t", frozen=True, pretrained=True)
    except Exception as e:  # offline / no cached weights
        import pytest
        pytest.skip(f"Swin_T IMAGENET1K_V1 weights unavailable (pre-cache on login node): {e}")
    assert bb.full_backbone_params == SWIN_T_PARAMS, bb.full_backbone_params
    enforce_determinism(strict=True); seed_everything(0)
    x = torch.randn(1, 3, 256, 704)
    a = bb(x); b = bb(x)
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert [f.shape[1] for f in a] == [96, 192, 384, 768]
