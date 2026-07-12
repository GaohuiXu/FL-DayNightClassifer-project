"""Effective multi-scale camera FPN for the O-017 stride-8 contract.

Every declared backbone level contributes to the returned tensor.  Each level is
projected independently, resampled to the selected output stride, summed, and
smoothed.  In particular, with Swin-T taps at strides ``(4, 8, 16, 32)`` and
``out_stride=8``, the stride-4 tap is downsampled while stride-16/32 taps are
upsampled.  This closes the old graph where stride-4/8 parameters were computed but
the stride-16 output could never depend on them.

Downsampling uses exact integer ``avg_pool2d`` and upsampling uses the MIT reference
choice, bilinear interpolation with ``align_corners=False``.  GroupNorm is retained
instead of BatchNorm for batch-size/client-distribution robustness.
"""
from __future__ import annotations

from typing import List, Sequence

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
    """All-level FPN -> one feature map at ``out_stride``.

    ``in_channels`` / ``in_strides`` describe the backbone taps; ``out_stride`` selects
    the output resolution (must be one of ``in_strides``).  No declared input level
    is ignored."""

    def __init__(
        self,
        in_channels: Sequence[int],
        in_strides: Sequence[int] = (4, 8, 16, 32),
        out_channels: int = 256,
        out_stride: int = 8,
    ):
        super().__init__()
        if len(in_channels) != len(in_strides) or not in_channels:
            raise ValueError("in_channels and in_strides must be non-empty and equal length")
        if len(set(int(s) for s in in_strides)) != len(in_strides):
            raise ValueError("in_strides must be unique")
        if tuple(sorted(int(s) for s in in_strides)) != tuple(int(s) for s in in_strides):
            raise ValueError("in_strides must be strictly increasing")
        if out_stride not in tuple(in_strides):
            raise ValueError(f"out_stride={out_stride} must be one of {tuple(in_strides)}")
        self.in_strides = tuple(in_strides)
        self.in_channels = tuple(int(c) for c in in_channels)
        self.out_channels = int(out_channels)
        self.out_stride = int(out_stride)
        self.out_level = self.in_strides.index(self.out_stride)

        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels
        )
        self.lateral_norm = nn.ModuleList(_gn(out_channels) for _ in in_channels)
        # One post-fusion smoothing block.  Every lateral feeds this tensor.
        self.smooth = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _gn(out_channels),
            nn.ReLU(inplace=True),
        )

    def _to_output_size(self, x: torch.Tensor, stride: int, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        if stride < self.out_stride:
            if self.out_stride % stride:
                raise ValueError(
                    f"cannot integer-downsample stride {stride} to {self.out_stride}"
                )
            factor = self.out_stride // stride
            pooled = F.avg_pool2d(x, kernel_size=factor, stride=factor)
            if pooled.shape[-2:] != size:
                raise ValueError(
                    f"stride-{stride} feature shape {tuple(x.shape[-2:])} does not map "
                    f"to target {size} by factor {factor}"
                )
            return pooled
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} features, got {len(feats)}")
        batch = feats[0].shape[0]
        for index, (feat, channels) in enumerate(zip(feats, self.in_channels)):
            if feat.ndim != 4 or feat.shape[0] != batch or feat.shape[1] != channels:
                raise ValueError(
                    f"feature {index} must have shape [B,{channels},H,W] with common B"
                )

        target_size = tuple(int(v) for v in feats[self.out_level].shape[-2:])
        fused = None
        for index, (feat, stride) in enumerate(zip(feats, self.in_strides)):
            lateral = F.relu(
                self.lateral_norm[index](self.lateral[index](feat)), inplace=False
            )
            lateral = self._to_output_size(lateral, int(stride), target_size)
            fused = lateral if fused is None else fused + lateral
        assert fused is not None
        return self.smooth(fused)

    def output_contract(self) -> dict:
        return {
            "input_strides": list(self.in_strides),
            "input_channels": list(self.in_channels),
            "all_levels_consumed": True,
            "output_stride": self.out_stride,
            "output_channels": self.out_channels,
            "upsample": "bilinear_align_corners_false",
            "downsample": "integer_avg_pool2d",
        }
