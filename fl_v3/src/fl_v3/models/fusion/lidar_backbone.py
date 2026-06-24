"""Dense 2D conv LiDAR backbone (MCR P1 capacity lever) — PillarNet/SECOND-2D style, NO spconv.

The convergence + per-class teardown diagnosed the LiDAR branch as the capacity bottleneck: the
:class:`PointPillarsEncoder` PFN is a single ``Linear(8→64)`` (~640 params) that scatters STRAIGHT to the
BEV canvas, so every pillar is encoded in spatial ISOLATION — **zero receptive field before fusion**. The
first conv any LiDAR feature sees is the POST-fusion neck (already mixed with the camera). Large objects
(truck/bus/trailer) that need multi-cell aggregation can't be represented, and a 1.68× loss-upweight moved
trailer only +0.004 ⇒ a representation (capacity) limit, not a loss-balance one.

This module inserts a deep dense 2D conv backbone on the scattered pillar BEV BEFORE ``ConvFuser`` — giving
the LiDAR branch the receptive field + depth it lacked. **Rule #2-clean** (pure Conv2d/GroupNorm/ReLU +
``F.interpolate`` FPN — NO spconv/atomic/scatter). It is DOWNSTREAM of the permutation-invariant
``index_copy`` scatter, so it inherits invariance and adds **no new determinism/AST obligation** (Conv2d/GN/
ReLU/interpolate/add are all in the allowed op set). Output channels are FIXED at construction (no
data-dependent shape) ⇒ ``torch.compile`` + DDP ``static_graph`` see constant shapes; every param is used
every forward. Default-OFF in the detector ⇒ when disabled the module is not constructed and the model is
byte-identical to the pre-backbone baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with the repo's group-count convention (matches fusion.py / bev_neck.py)."""
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    """Two Conv-GN-ReLU (first carries the stride) — mirrors the bev_neck conv idiom."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False), _gn(cout), nn.ReLU(inplace=True),
    )


class LidarBackbone2D(nn.Module):
    """Dense PillarNet/SECOND-2D backbone over the PFN BEV canvas.

    ``[B, in_channels, ny, nx] → [B, out_channels, ny, nx]`` (spatial H,W PRESERVED so ``ConvFuser`` can
    still concatenate with the camera BEV). 3 down-stages + a 3-level nearest-upsample FPN. ~2.52M params at
    the default (in=64, out=128, stem=128, mid=256)."""

    def __init__(self, in_channels: int = 64, out_channels: int = 128,
                 stem: int = 128, mid: int = 256, activation_checkpoint: bool = False, num_stages: int = 3):
        super().__init__()
        self.activation_checkpoint = bool(activation_checkpoint)
        self.s1 = _block(in_channels, stem, 1)      # [B,stem, H,   W  ]  — 5×5 RF (the "zero RF" fix)
        self.s2 = _block(stem, mid, 2)              # [B,mid,  H/2, W/2]
        self.s3 = _block(mid, mid, 2)               # [B,mid,  H/4, W/4]  — multi-cell aggregation
        # ADDITIVE 4th down-stage (num_stages=4): H/8 — restores the receptive field for LARGE objects at the
        # 512²/0.2m grid where the 3-stage RF was too small (bus/trailer regressed). None ⇒ 3-stage byte-identical.
        self.s4 = _block(mid, mid, 2) if num_stages >= 4 else None
        C = int(out_channels)
        self.l1 = nn.Sequential(nn.Conv2d(stem, C, 1, bias=False), _gn(C))
        self.l2 = nn.Sequential(nn.Conv2d(mid, C, 1, bias=False), _gn(C))
        self.l3 = nn.Sequential(nn.Conv2d(mid, C, 1, bias=False), _gn(C))
        self.l4 = nn.Sequential(nn.Conv2d(mid, C, 1, bias=False), _gn(C)) if num_stages >= 4 else None
        self.smooth = nn.Sequential(nn.Conv2d(C, C, 3, padding=1, bias=False), _gn(C), nn.ReLU(inplace=True))
        self.out_channels = C

    def _run(self, m: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.activation_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(m, x, use_reentrant=False)
        return m(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self._run(self.s1, x)
        c2 = self._run(self.s2, c1)
        c3 = self._run(self.s3, c2)
        # top-down FPN: nearest-upsample by SIZE (not scale_factor — avoids 256/512 off-by-one) + add
        if self.s4 is not None:                     # 4-stage: deepest level feeds the large-object context down
            c4 = self._run(self.s4, c3)
            p3 = self.l3(c3) + F.interpolate(self.l4(c4), size=c3.shape[-2:], mode="nearest")
        else:
            p3 = self.l3(c3)
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p1 = self.l1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="nearest")
        return self.smooth(p1)                      # [B, out_channels, ny, nx]
