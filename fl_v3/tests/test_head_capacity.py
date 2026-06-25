"""MCR P1 large-vehicle localization lever — head capacity (shared-tower depth + width).

The box size/yaw regression (which drives the bus/truck/trailer TP-error deficit) comes off a 1×1 probe
on a SINGLE shared Conv-GN-ReLU (256→64). This pins the new ``det-head-conv-layers`` depth knob (and the
existing ``det-head-channels`` width knob): default depth=1 is BYTE-IDENTICAL (same state_dict keys), and
deeper/wider heads add the expected tensors while preserving the output schema.
"""
from __future__ import annotations

import torch

from fl_v3.models.fusion.head import CenterPointHead

IN, HW = 256, 32   # bev-neck out channels; a small spatial grid for the forward check


def _keys(m):
    return sorted(m.state_dict().keys())


def test_head_depth1_is_byte_identical_layout():
    """conv_layers=1 must reproduce the original single-conv shared tower exactly (keys + counts)."""
    h = CenterPointHead(in_channels=IN, n_classes=10, head_channels=64, conv_layers=1)
    keys = _keys(h)
    # the original head: shared.0(conv) + shared.1(gn w,b) + heatmap(w,b) + 5 reg heads(w,b each)
    assert "shared.0.weight" in keys and "shared.1.weight" in keys and "shared.1.bias" in keys
    assert not any(k.startswith("shared.3") for k in keys)   # no second block at depth 1
    n_tensors = len(keys)
    assert n_tensors == 15, f"depth-1 head must have 15 tensors (the frozen contract), got {n_tensors}"


def test_head_depth2_adds_a_block():
    h1 = CenterPointHead(in_channels=IN, n_classes=10, head_channels=64, conv_layers=1)
    h2 = CenterPointHead(in_channels=IN, n_classes=10, head_channels=64, conv_layers=2)
    p1 = sum(p.numel() for p in h1.parameters())
    p2 = sum(p.numel() for p in h2.parameters())
    assert p2 > p1                                            # the extra Conv-GN-ReLU adds params
    assert "shared.3.weight" in _keys(h2)                     # second conv block present
    # the second block is head_channels→head_channels 3×3 conv + GN
    extra = p2 - p1
    assert extra == 64 * 64 * 9 + 64 + 64                     # conv(64,64,3,3) + gn(w,b)


def test_head_width128_forward_schema_unchanged():
    h = CenterPointHead(in_channels=IN, n_classes=10, head_channels=128, conv_layers=2)
    x = torch.randn(2, IN, HW, HW)
    out = h(x)
    assert out["heatmap"].shape == (2, 10, HW, HW)
    for k, ch in {"reg": 2, "height": 1, "dim": 3, "rot": 2, "vel": 2}.items():
        assert out[k].shape == (2, ch, HW, HW)


def test_head_init_bias_preserved():
    h = CenterPointHead(in_channels=IN, n_classes=10, head_channels=64, conv_layers=2)
    assert torch.allclose(h.heatmap.bias, torch.full_like(h.heatmap.bias, -2.19))
