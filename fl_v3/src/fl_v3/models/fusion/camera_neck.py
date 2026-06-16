"""Camera FPN neck — GeneralizedLSSFPN analog (fl_v3 T2).

Fuses the 4 multi-scale Swin-T / ResNet-18 feature maps (strides 4/8/16/32) into a
single feature map at ``out_stride`` (default 16) with ``out_channels`` channels — the
input to the LSS view transform. Standard top-down FPN: 1×1 laterals to a common
channel count, nearest-neighbour upsample of deeper levels added into shallower ones,
a 3×3 smoothing conv. **GroupNorm** (D6, not BatchNorm — FL-non-IID-robust) and
**nearest** upsampling keep the forward fully deterministic (no ``grid_sample``, no
adaptive pool, no atomic op). All params are trainable (the AD-specific camera→BEV
projection is FL-learned per D1).
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm with a divisor-safe group count (D6)."""
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class GeneralizedLSSFPN(nn.Module):
    """Top-down FPN → single feature map at ``out_stride``.

    ``in_channels`` / ``in_strides`` describe the backbone taps; ``out_stride`` selects
    which pyramid level is returned (must be one of ``in_strides``)."""

    def __init__(
        self,
        in_channels: List[int],
        in_strides: List[int] = (4, 8, 16, 32),
        out_channels: int = 256,
        out_stride: int = 16,
    ):
        super().__init__()
        if out_stride not in tuple(in_strides):
            raise ValueError(f"out_stride={out_stride} must be one of {tuple(in_strides)}")
        self.in_strides = tuple(in_strides)
        self.out_channels = int(out_channels)
        self.out_stride = int(out_stride)
        self.out_level = self.in_strides.index(self.out_stride)

        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels
        )
        self.lateral_norm = nn.ModuleList(_gn(out_channels) for _ in in_channels)
        # one 3×3 smoothing conv at the output level
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # 1×1 laterals (+ GN + ReLU) to common channels.
        lats = [
            F.relu(self.lateral_norm[i](self.lateral[i](f)), inplace=True)
            for i, f in enumerate(feats)
        ]
        # Top-down: add each deeper level (nearest-upsampled) into the shallower one.
        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i - 1].shape[-2:], mode="nearest")
            lats[i - 1] = lats[i - 1] + up
        out = self.smooth(lats[self.out_level])
        return out
