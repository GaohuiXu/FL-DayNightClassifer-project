"""SECOND-FPN BEV neck (fl_v3 T2).

A deterministic SECOND-FPN: two stride-2 encoder branches over the fused BEV, each
upsampled (nearest — deterministic) to the head resolution and channel-concatenated.
**GroupNorm** (D6); **nearest** upsample instead of ``ConvTranspose2d`` to avoid any
conv-transpose backward determinism worry; **no ``Adaptive*Pool2d``** (its CUDA backward
raises). Input is the fused BEV ``[B, Cf, ny, nx]``; output is ``[B, Cout, head_ny,
head_nx]`` at the head resolution (``ny / out_size_factor``)."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


def _conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        _gn(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False),
        _gn(cout),
        nn.ReLU(inplace=True),
    )


class SecondFPNNeck(nn.Module):
    """Fused BEV ``[B,Cf,ny,nx]`` → ``[B,Cout,head_ny,head_nx]`` (SECOND-FPN).

    ``out_size_factor`` matches :class:`BEVConfig` (head grid = ny / factor). With
    factor 2 the two branches downsample to ``ny/2`` and ``ny/4`` then upsample back to
    ``ny/2``; their concat is the head feature."""

    def __init__(self, in_channels: int, out_channels: int = 256, out_size_factor: int = 2):
        super().__init__()
        self.out_size_factor = int(out_size_factor)
        c1 = out_channels // 2
        c2 = out_channels // 2
        # branch 1: down to ny/factor; branch 2: down to ny/(2*factor)
        self.enc1 = _conv_block(in_channels, c1, stride=out_size_factor)
        self.enc2 = _conv_block(c1, c2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(c1, c1, 3, padding=1, bias=False), _gn(c1), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(c2, c2, 3, padding=1, bias=False), _gn(c2), nn.ReLU(inplace=True))
        self.out_channels = c1 + c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                                   # [B, c1, ny/f, nx/f]
        e2 = self.enc2(e1)                                  # [B, c2, ny/2f, nx/2f]
        d1 = self.dec1(e1)                                  # at head resolution
        d2 = self.dec2(e2)
        d2 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")  # up to head res (deterministic)
        return torch.cat([d1, d2], dim=1)                   # [B, c1+c2, head_ny, head_nx]
