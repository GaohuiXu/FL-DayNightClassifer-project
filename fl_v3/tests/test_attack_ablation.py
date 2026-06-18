"""T5 ablation: the cond-5a zero-LiDAR-BEV clean readout (§0.B) + the floor-corrected verdict (§0.C4)."""
from __future__ import annotations

import torch

from fl_v3.models.fusion.fusion import ConvFuser
from fl_v3.attacks.fusion_ablation import _zero_lidar_pre_hook, fusion_aware_verdict, CONDITIONS


def test_convfuser_zeroing_lidar_is_a_clean_camera_only_readout():
    """ConvFuser = concat → Conv2d(bias=False) → GN(output) → ReLU ⇒ zeroing the LiDAR-BEV input gives
    an output that is IDENTICAL for ANY LiDAR input (provably LiDAR-zeroed, §0.B)."""
    torch.manual_seed(0)
    f = ConvFuser(camera_channels=80, lidar_channels=64, out_channels=128).eval()
    cam = torch.randn(2, 80, 16, 16)
    lid_a = torch.randn(2, 64, 16, 16)
    lid_b = torch.randn(2, 64, 16, 16) * 7.0 + 3.0  # a very different LiDAR input
    out_a = f(cam, torch.zeros_like(lid_a))
    out_b = f(cam, torch.zeros_like(lid_b))
    assert torch.equal(out_a, out_b)  # output independent of the (zeroed) LiDAR


def test_zero_lidar_hook_equals_manual_zeros_fusion():
    torch.manual_seed(1)
    f = ConvFuser(80, 64, 128).eval()
    cam = torch.randn(1, 80, 16, 16); lid = torch.randn(1, 64, 16, 16) * 5 + 2
    h = f.register_forward_pre_hook(_zero_lidar_pre_hook)
    try:
        out_hook = f(cam, lid)            # the hook replaces lid with zeros
    finally:
        h.remove()
    out_manual = f(cam, torch.zeros_like(lid))
    assert torch.equal(out_hook, out_manual)


def _asr(c1, c2, c3, c4, c5a):
    return {"cond1_clean": c1, "cond2_nonaligned": c2, "cond3_lidar_removed": c3,
            "cond4_aligned": c4, "cond5a_camera_only": c5a}


def test_verdict_fusion_aware_when_cond4_dominates_and_guards_valid():
    v = fusion_aware_verdict(_asr(0.02, 0.05, 0.10, 0.60, 0.15), delta_fusion=0.2, mult=2.0,
                             cond5a_guards_valid=True)
    assert v["fusion_aware"] and v["verdict"] == "FUSION-AWARE"
    assert abs(v["floor"] - 0.02) < 1e-9 and abs(v["cond4_corrected"] - 0.58) < 1e-9


def test_verdict_refused_when_cond5a_guards_invalid():
    """Even with cond-4 dominating, an INVALID cond-5a (guards failed) cannot be FUSION-AWARE (§0.B)."""
    v = fusion_aware_verdict(_asr(0.02, 0.05, 0.10, 0.60, 0.15), delta_fusion=0.2, mult=2.0,
                             cond5a_guards_valid=False)
    assert not v["fusion_aware"]
    v2 = fusion_aware_verdict(_asr(0.02, 0.05, 0.10, 0.60, 0.15), delta_fusion=0.2, mult=2.0,
                              cond5a_guards_valid=None)  # guards not run → NOT valid
    assert not v2["fusion_aware"]


def test_verdict_degenerate_is_d3_escape_hatch():
    v = fusion_aware_verdict(_asr(0.02, 0.40, 0.10, 0.40, 0.10), delta_fusion=0.2, mult=2.0,
                             cond5a_guards_valid=True)
    assert not v["fusion_aware"] and v["degenerate_cond4_approx_cond2"]
    assert "D3-ESCAPE-HATCH" in v["verdict"]


def test_verdict_not_viable_when_cond4_below_floor():
    v = fusion_aware_verdict(_asr(0.02, 0.01, 0.01, 0.20, 0.01), delta_fusion=0.2, mult=2.0,
                             cond5a_guards_valid=True)
    assert not v["viable_cond4_ge_0.3"] and not v["fusion_aware"]


def test_verdict_margin_and_mult_rules():
    # cond4_c=0.28, cond5a_c=0.18 → margin 0.10 < 0.2 fails δ_fusion even though mult ok-ish
    v = fusion_aware_verdict(_asr(0.02, 0.05, 0.05, 0.30, 0.20), delta_fusion=0.2, mult=2.0,
                             cond5a_guards_valid=True)
    assert not v["margin_ok_ge_delta_fusion"] and not v["fusion_aware"]


def test_conditions_constant_order():
    assert CONDITIONS == ("cond1_clean", "cond2_nonaligned", "cond3_lidar_removed",
                          "cond4_aligned", "cond5a_camera_only")
