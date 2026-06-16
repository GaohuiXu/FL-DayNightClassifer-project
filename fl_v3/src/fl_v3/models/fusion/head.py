"""CenterPoint dense detection head (fl_v3 T2).

A SeparateHead over the BEV neck feature: a shared Conv-GN-ReLU, then one conv per
task — a **per-class heatmap** (``n_classes`` channels) and class-agnostic regression
(``reg`` xy-offset, ``height`` z, ``dim`` log-extent, ``rot`` (sin,cos), ``vel`` (vx,vy)).
GroupNorm (D6); no adaptive pooling. The heatmap's final-conv bias is initialized to
``-2.19`` (focal-loss prior ≈ sigmoid 0.1, the CenterPoint init) so the overfit does
not start saturated.

The regression channels parameterize the **T1 canonical box** ``(cx,cy,cz,dx=l,dy=w,
dz=h,yaw)`` — decode (in :mod:`detector`) reconstructs it with **no** mmdet3d ``-π/2``
offset and **no** ``(l,w,h)`` swap (the encode→decode golden pins this).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


# regression sub-head channel layout (class-agnostic)
REG_CHANNELS = {"reg": 2, "height": 1, "dim": 3, "rot": 2, "vel": 2}


class CenterPointHead(nn.Module):
    """Dense per-class heatmap + shared regression over the BEV feature."""

    def __init__(self, in_channels: int, n_classes: int = 10, head_channels: int = 64,
                 init_bias: float = -2.19):
        super().__init__()
        self.n_classes = int(n_classes)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1, bias=False),
            _gn(head_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap = nn.Conv2d(head_channels, n_classes, 1)
        nn.init.constant_(self.heatmap.bias, init_bias)  # focal-loss prior
        self.reg_heads = nn.ModuleDict(
            {name: nn.Conv2d(head_channels, ch, 1) for name, ch in REG_CHANNELS.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        out = {"heatmap": self.heatmap(h)}
        for name, head in self.reg_heads.items():
            out[name] = head(h)
        return out
