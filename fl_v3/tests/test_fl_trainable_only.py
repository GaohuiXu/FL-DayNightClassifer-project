"""DT3-A — the trainable-only (62-tensor) update vector + the frozen T3→T6 layout (T3).

  * exactly 62 trainable tensors, module order + per-module slice map as the frozen
    contract; all frozen *params* under camera_backbone;
  * strict=False load of the partial vector into the full model is clean (no unexpected
    keys; no trainable key unfilled);
  * the frozen backbone is reconstructed BYTE-IDENTICALLY across nodes under
    ``pretrained=True`` and DIVERGES under ``pretrained=False`` (why pretrained is required);
  * the final checkpoint = a self-contained FULL model that loads ``strict=True`` (for T4);
  * ``trainable_state_dict`` degenerates to the FULL state on a no-frozen model (TinyMLP).
"""
from __future__ import annotations

import torch

from fl_v3.models.dummy import TinyMLP
from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
from fl_v3.training.tasks import (
    TRAINABLE_MODULE_SLICE_MAP,
    TRAINABLE_TENSOR_COUNT,
    assert_trainable_layout,
    load_trainable_state_dict,
    trainable_param_names,
    trainable_state_dict,
)


def _bringup_cfg(pretrained: bool = False):
    return DetectorConfig(
        camera_backbone="resnet18", pretrained_backbone=pretrained,
        freeze_camera_backbone=True, neck_channels=64, context_channels=32,
        lidar_channels=32, fusion_channels=64, bev_neck_channels=128, head_channels=32,
    )


def test_trainable_layout_is_the_frozen_62_tensor_contract():
    torch.manual_seed(0)
    model = BEVFusionDetector(_bringup_cfg(pretrained=False))
    tsd = trainable_state_dict(model)
    assert len(tsd) == TRAINABLE_TENSOR_COUNT == 62
    assert_trainable_layout(model)  # module order + per-module counts == contract
    # Per-module counts independently recomputed here (don't just trust the helper).
    counts = {}
    for k in tsd:
        top = k.split(".")[0]
        counts[top] = counts.get(top, 0) + 1
    assert counts == dict(TRAINABLE_MODULE_SLICE_MAP)


def test_all_frozen_params_live_under_camera_backbone():
    torch.manual_seed(0)
    model = BEVFusionDetector(_bringup_cfg(pretrained=False))
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    assert frozen, "expected a frozen backbone"
    assert all(n.startswith("camera_backbone") for n in frozen)
    # buffers are excluded from the update vector — none of them are trainable params
    assert not (set(dict(model.named_buffers())) & trainable_param_names(model))


def test_strict_false_load_is_clean_and_loads_trainable():
    torch.manual_seed(0)
    src = BEVFusionDetector(_bringup_cfg(pretrained=False))
    tsd = trainable_state_dict(src)
    # perturb the source trainable params so we can detect a real load
    for v in tsd.values():
        v.add_(1.0)
    torch.manual_seed(123)  # different init for the destination
    dst = BEVFusionDetector(_bringup_cfg(pretrained=False))
    load_trainable_state_dict(dst, tsd)  # strict=False + clean-load assertions
    dst_tsd = trainable_state_dict(dst)
    for k in tsd:
        assert torch.equal(dst_tsd[k], tsd[k]), f"trainable tensor {k} not loaded"


def test_frozen_backbone_cross_node_identity_pretrained_vs_random():
    def backbone_state(seed, pretrained):
        torch.manual_seed(seed)
        m = BEVFusionDetector(_bringup_cfg(pretrained=pretrained))
        return {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}

    a = backbone_state(111, pretrained=True)
    b = backbone_state(222, pretrained=True)
    assert all(torch.equal(a[k], b[k]) for k in a), (
        "pretrained frozen backbone MUST be byte-identical across nodes/seeds"
    )
    c = backbone_state(111, pretrained=False)
    d = backbone_state(222, pretrained=False)
    assert not all(torch.equal(c[k], d[k]) for k in c), (
        "random-init frozen backbone diverges per seed — why det-pretrained-backbone is required"
    )


def test_final_checkpoint_is_full_model_loads_strict_true(tmp_path):
    torch.manual_seed(0)
    trained = BEVFusionDetector(_bringup_cfg(pretrained=False))
    agg_trainable = trainable_state_dict(trained)  # the aggregated trainable vector
    # merge into a FRESH full model (frozen backbone reconstructed) -> save FULL state
    fresh = BEVFusionDetector(_bringup_cfg(pretrained=False))
    load_trainable_state_dict(fresh, agg_trainable)
    ckpt = tmp_path / "final_model.pt"
    torch.save(fresh.state_dict(), ckpt)
    reloaded = BEVFusionDetector(_bringup_cfg(pretrained=False))
    missing, unexpected = reloaded.load_state_dict(torch.load(ckpt), strict=True)
    assert missing == [] and unexpected == [], "final checkpoint must load strict=True (T4)"


def test_trainable_state_dict_degenerates_to_full_on_tinymlp():
    m = TinyMLP(in_dim=4, hidden=8, out_dim=1)
    full = m.state_dict()
    tsd = trainable_state_dict(m)
    assert list(tsd.keys()) == list(full.keys()), (
        "no-frozen model: trainable vector == full state (task-agnostic preserved)"
    )
