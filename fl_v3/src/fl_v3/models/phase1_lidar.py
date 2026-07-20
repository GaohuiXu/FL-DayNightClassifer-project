"""Reference-led S10 Phase-I standalone LiDAR detector.

Graph:

``hard voxel mean VFE -> sparse SECOND/BN1d -> [B,256,180,180]
   -> SECOND [5,5]/BN2d -> SECONDFPN/BN2d -> TransFusionHead``

The sparse encoder is kept inside the accepted FP32 island.  Dense decoder and
head operations inherit the caller's global autocast policy.  This module is
separate from the historical Fusion detector and contains no GroupNorm or local
shallow BEV neck.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterable, Mapping

import torch
import torch.nn as nn

from fl_v3.config.phase1 import FROZEN_LIDAR_MODEL, REFERENCE_OBJECT_CLASSES
from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder
from fl_v3.models.phase1_transfusion import (
    PHASE1_TRANSFUSION_GEOMETRY,
    Phase1TransFusionHead,
    Phase1TransFusionLoss,
)


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    *,
    stride: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class Phase1SECOND(nn.Module):
    """Pinned dense SECOND stages: six convolutions at each of two levels."""

    def __init__(self) -> None:
        super().__init__()
        stages = []
        for in_channels, out_channels, stride in ((256, 128, 1), (128, 256, 2)):
            layers: list[nn.Module] = [
                _conv_bn_relu(in_channels, out_channels, stride=stride)
            ]
            layers.extend(_conv_bn_relu(out_channels, out_channels) for _ in range(5))
            stages.append(nn.Sequential(*layers))
        self.blocks = nn.ModuleList(stages)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 4 or value.shape[1] != 256:
            raise ValueError(f"Phase-I SECOND input must be [B,256,H,W], got {tuple(value.shape)}")
        outputs = []
        for block in self.blocks:
            value = block(value)
            outputs.append(value)
        return outputs[0], outputs[1]


class Phase1SECONDFPN(nn.Module):
    """Pinned stride-one convolution plus stride-two transpose convolution."""

    def __init__(self) -> None:
        super().__init__()
        self.deblocks = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(128, 256, 1, stride=1, bias=False),
                    nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
                    nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True),
                ),
            )
        )

    def forward(self, levels: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        if len(levels) != 2 or levels[0].shape[1] != 128 or levels[1].shape[1] != 256:
            raise ValueError("Phase-I SECONDFPN received a drifted SECOND feature inventory")
        outputs = [module(value) for module, value in zip(self.deblocks, levels, strict=True)]
        if outputs[0].shape[-2:] != outputs[1].shape[-2:]:
            raise RuntimeError(
                f"Phase-I SECONDFPN spatial alignment drift: "
                f"{tuple(outputs[0].shape)} vs {tuple(outputs[1].shape)}"
            )
        return torch.cat(outputs, dim=1)


class Phase1LidarDetector(nn.Module):
    """Complete frozen standalone LiDAR primary used by Envelope A/B."""

    def __init__(self) -> None:
        super().__init__()
        self.class_names = tuple(REFERENCE_OBJECT_CLASSES)
        self.lidar_encoder = SparseVoxelEncoder(
            out_channels=256,
            use_timestamp=False,
            z_voxel=0.2,
            sparse_z_size=41,
            max_voxels_train=120000,
            max_voxels_eval=160000,
            max_points_per_voxel=10,
            sparse_conv_fp16=False,
            second_normalization="batch_norm_1d",
            output_mode="collapsed",
            point_feature_mode="xyzi_time",
        )
        self.decoder_backbone = Phase1SECOND()
        self.decoder_neck = Phase1SECONDFPN()
        self.head = Phase1TransFusionHead()
        self._operator_profile_ranges = False

    @contextmanager
    def operator_profile_ranges(self):
        """Enable output-neutral, bounded ranges for a short torch trace."""
        if self._operator_profile_ranges:
            raise RuntimeError("Phase-I LiDAR profiler ranges are already active")
        self._operator_profile_ranges = True
        try:
            yield self
        finally:
            self._operator_profile_ranges = False

    def _profile_range(self, name: str):
        if not self._operator_profile_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"fl_v3::lidar::{name}")

    def forward(self, batch: Mapping[str, object], *, return_intermediates: bool = False):
        if "lidar_points" not in batch:
            raise KeyError("Phase-I LiDAR batch is missing 'lidar_points'")
        points = batch["lidar_points"]
        if not torch.is_tensor(points):
            raise TypeError("Phase-I LiDAR points must be a collated tensor")
        if "batch_size" in batch:
            batch_size = int(batch["batch_size"])
        elif "gt_boxes" in batch:
            batch_size = len(batch["gt_boxes"])
        else:
            raise KeyError("Phase-I LiDAR batch is missing batch_size/gt_boxes")
        with self._profile_range("voxel_vfe_sparse_collapse"):
            collapsed = self.lidar_encoder(points, batch_size)
        if collapsed.dtype != torch.float32:
            raise RuntimeError(
                f"Phase-I sparse FP32 island returned {collapsed.dtype}, expected torch.float32"
            )
        if tuple(collapsed.shape) != (batch_size, 256, 180, 180):
            raise RuntimeError(
                f"Phase-I sparse collapse shape drift: got {tuple(collapsed.shape)}, "
                f"expected {(batch_size, 256, 180, 180)}"
            )
        with self._profile_range("second_backbone"):
            levels = self.decoder_backbone(collapsed)
        with self._profile_range("second_fpn"):
            decoded = self.decoder_neck(levels)
        if tuple(decoded.shape) != (batch_size, 512, 180, 180):
            raise RuntimeError(f"Phase-I dense LiDAR decoder shape drift: {tuple(decoded.shape)}")
        with self._profile_range("transfusion_head"):
            output = self.head(decoded)
        if not return_intermediates:
            return output
        return {
            "predictions": output,
            "sparse_collapse_fp32": collapsed,
            "second_levels": levels,
            "decoder_feature": decoded,
        }

    @staticmethod
    def build_criterion() -> Phase1TransFusionLoss:
        return Phase1TransFusionLoss()

    @torch.no_grad()
    def decode(
        self,
        output,
        *,
        score_threshold: float = 0.0,
    ) -> list[dict[str, torch.Tensor]]:
        if float(score_threshold) != 0.0:
            raise ValueError("Phase-I TransFusion score threshold is frozen to 0.0")
        if isinstance(output, dict) and "predictions" in output:
            output = output["predictions"]
        return self.head.decode(output)

    def parameter_groups(self) -> dict[str, Iterable[nn.Parameter]]:
        return {
            "lidar_encoder": self.lidar_encoder.parameters(),
            "decoder_backbone": self.decoder_backbone.parameters(),
            "decoder_neck": self.decoder_neck.parameters(),
            "head": self.head.parameters(),
        }


def build_phase1_lidar_model(config) -> Phase1LidarDetector:
    """Construct the exact graph from a validated LiDAR ResolvedConfig."""
    if not getattr(config, "is_phase1", False):
        raise ValueError("Phase-I LiDAR construction requires a Phase-I ResolvedConfig")
    raw = config.as_dict()
    if raw["contract"]["branch"] != "lidar":
        raise ValueError("Phase-I LiDAR construction received the Camera recipe")
    if raw["model"] != FROZEN_LIDAR_MODEL:
        raise ValueError("resolved Phase-I LiDAR graph contract drift")
    initialization = raw["initialization"]
    if (
        initialization["kind"] != "scratch"
        or initialization["status"] != "accepted"
        or initialization["scratch"]["seed"] != 0
    ):
        raise RuntimeError("Phase-I LiDAR scratch initialization contract drift")
    model = Phase1LidarDetector()
    group_norms = [name for name, module in model.named_modules() if isinstance(module, nn.GroupNorm)]
    if group_norms:
        raise RuntimeError(f"Phase-I LiDAR graph unexpectedly contains GroupNorm: {group_norms}")
    if model.head.geometry != PHASE1_TRANSFUSION_GEOMETRY:
        raise RuntimeError("Phase-I LiDAR TransFusion geometry drift")
    return model


__all__ = [
    "Phase1LidarDetector",
    "Phase1SECOND",
    "Phase1SECONDFPN",
    "build_phase1_lidar_model",
]
