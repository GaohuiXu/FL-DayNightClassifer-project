"""BEV-concat ``ConvFuser`` (D3) — the fusion sub-network (fl_v3 T2).

Channel-concatenates the camera-BEV ⊕ LiDAR-BEV (both on the shared fine grid) and
mixes them with Conv2d–**GroupNorm**(D6)–ReLU. A **cleanly named** module
(``model.fusion``) so T5's fusion-aware attack and T6's per-module gradient slicing /
the Q2 dilution analysis can address it without surgery. No BatchNorm (FL-non-IID),
no adaptive pooling, no atomic op."""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class ConvFuser(nn.Module):
    """``[B,Cc,H,W] ⊕ [B,Cl,H,W] → [B,Cout,H,W]`` via concat + 2× Conv-GN-ReLU."""

    def __init__(self, camera_channels: int, lidar_channels: int, out_channels: int = 128):
        super().__init__()
        self.camera_channels = int(camera_channels)
        self.lidar_channels = int(lidar_channels)
        self.out_channels = int(out_channels)
        in_ch = self.camera_channels + self.lidar_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_bev: torch.Tensor, lidar_bev: torch.Tensor) -> torch.Tensor:
        x = torch.cat([camera_bev, lidar_bev], dim=1)  # channel concat (D3)
        return self.block(x)
