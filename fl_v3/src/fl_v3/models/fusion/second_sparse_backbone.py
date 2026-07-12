"""Framework-independent SECOND sparse-backbone contract and spconv implementation.

The geometry helpers in this module do not import spconv, so shape, receptive-field,
metric-alignment, and dense-memory bounds can be checked on the login node.  The
runtime class imports the validated Arrhenius spconv stack only when constructed.

The reference layout is MIT BEVFusion's nuScenes ``voxelnet_0p075`` SparseEncoder:

``41x1440x1440 -> 21x720x720 -> 11x360x360 -> 5x180x180``

followed by a z-only sparse convolution to ``2x180x180`` and *then* densification.
All shapes in this file use ``(z, y, x)``; sparse indices use ``(batch, z, y, x)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


def _conv_size(size: int, kernel: int, stride: int, padding: int) -> int:
    return (int(size) + 2 * int(padding) - int(kernel)) // int(stride) + 1


@dataclass(frozen=True)
class SECONDShapeContract:
    """Static geometry for the sparse-to-dense resolution flow."""

    input_zyx: tuple[int, int, int]
    voxel_xyz: tuple[float, float, float]
    range_xyzxyz: tuple[float, float, float, float, float, float]
    output_stride_xy: int = 8
    sparse_output_channels: int = 128

    def __post_init__(self) -> None:
        if any(v <= 0 for v in self.input_zyx):
            raise ValueError(f"input_zyx must be positive, got {self.input_zyx}")
        if any(v <= 0 for v in self.voxel_xyz):
            raise ValueError(f"voxel_xyz must be positive, got {self.voxel_xyz}")
        x0, y0, z0, x1, y1, z1 = self.range_xyzxyz
        vx, vy, vz = self.voxel_xyz
        physical_xyz = (
            int(round((x1 - x0) / vx)),
            int(round((y1 - y0) / vy)),
            int(round((z1 - z0) / vz)),
        )
        z, y, x = self.input_zyx
        if (x, y) != physical_xyz[:2]:
            raise ValueError(
                f"input XY shape {(y, x)} does not match range/voxel bins "
                f"{(physical_xyz[1], physical_xyz[0])}"
            )
        if z != physical_xyz[2] + 1:
            raise ValueError(
                f"input z shape must be physical bins + one reference padding bin: "
                f"got {z}, expected {physical_xyz[2] + 1}"
            )
        if x % 8 or y % 8:
            raise ValueError(f"input XY shape {(y, x)} must be divisible by stride 8")
        if self.output_stride_xy != 8:
            raise ValueError("the frozen S04 SECOND contract requires XY stride 8")
        final = self.sparse_output_zyx
        if any(v <= 0 for v in final):
            raise ValueError(
                f"input z extent {self.input_zyx[0]} is too small for the SECOND stages"
            )

    @property
    def stage_shapes_zyx(self) -> tuple[tuple[int, int, int], ...]:
        """Stem, three down stages, stage-4, and z-collapse sparse shapes."""
        z, y, x = self.input_zyx
        stem = (z, y, x)
        s1 = tuple(_conv_size(v, 3, 2, 1) for v in stem)
        s2 = tuple(_conv_size(v, 3, 2, 1) for v in s1)
        s3 = (
            _conv_size(s2[0], 3, 2, 0),
            _conv_size(s2[1], 3, 2, 1),
            _conv_size(s2[2], 3, 2, 1),
        )
        s4 = s3
        out = (_conv_size(s4[0], 3, 2, 0), s4[1], s4[2])
        return stem, s1, s2, s3, s4, out

    @property
    def sparse_output_zyx(self) -> tuple[int, int, int]:
        return self.stage_shapes_zyx[-1]

    @property
    def bev_hw(self) -> tuple[int, int]:
        _, y, x = self.sparse_output_zyx
        return y, x

    @property
    def collapsed_channels(self) -> int:
        return self.sparse_output_channels * self.sparse_output_zyx[0]

    @property
    def output_cell_xy(self) -> tuple[float, float]:
        return (
            self.voxel_xyz[0] * self.output_stride_xy,
            self.voxel_xyz[1] * self.output_stride_xy,
        )

    @property
    def receptive_field_voxels_zyx(self) -> tuple[int, int, int]:
        # Stem: one k3 conv. Each of four stages has two residual blocks,
        # each block has two k3 SubM convs. The first three stages end in a
        # k3/s2 downsample; conv_out is k3/s2 in z only.
        rf = [3, 3, 3]
        jump = [1, 1, 1]
        for stage in range(4):
            for _ in range(4):
                for axis in range(3):
                    rf[axis] += 2 * jump[axis]
            if stage < 3:
                for axis in range(3):
                    rf[axis] += 2 * jump[axis]
                    jump[axis] *= 2
        rf[0] += 2 * jump[0]
        return tuple(rf)

    @property
    def receptive_field_metres_zyx(self) -> tuple[float, float, float]:
        rz, ry, rx = self.receptive_field_voxels_zyx
        vx, vy, vz = self.voxel_xyz
        return rz * vz, ry * vy, rx * vx

    def output_cell_center_xy(self, col: int, row: int) -> tuple[float, float]:
        x0, y0 = self.range_xyzxyz[:2]
        sx, sy = self.output_cell_xy
        return x0 + (int(col) + 0.5) * sx, y0 + (int(row) + 0.5) * sy

    def final_dense_numel(self, batch_size: int) -> int:
        z, y, x = self.sparse_output_zyx
        return int(batch_size) * self.sparse_output_channels * z * y * x

    def forbidden_fine_dense_numel(self, batch_size: int) -> int:
        z, y, x = self.input_zyx
        return int(batch_size) * self.sparse_output_channels * z * y * x

    @classmethod
    def reference_075(cls) -> "SECONDShapeContract":
        return cls(
            input_zyx=(41, 1440, 1440),
            voxel_xyz=(0.075, 0.075, 0.2),
            range_xyzxyz=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        )


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, max(1, int(channels) // 2))
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _replace_feature(x, features: torch.Tensor):
    """spconv-2 feature replacement, isolated for residual blocks."""
    return x.replace_feature(features)


class _SparseResidualBlock(nn.Module):
    """Two SubMConv3d layers with sample-local per-voxel GroupNorm."""

    def __init__(self, channels: int, indice_key: str):
        super().__init__()
        import spconv.pytorch as spconv

        self.conv1 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.norm1 = _gn(channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = spconv.SubMConv3d(
            channels, channels, 3, padding=1, bias=False, indice_key=indice_key
        )
        self.norm2 = _gn(channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x.features
        out = self.conv1(x)
        out = _replace_feature(out, self.act1(self.norm1(out.features)))
        out = self.conv2(out)
        out = _replace_feature(out, self.norm2(out.features))
        return _replace_feature(out, self.act2(out.features + identity))


class SECONDSparseBackbone(nn.Module):
    """Reference-shaped sparse SECOND encoder; never densifies internally."""

    def __init__(self, in_channels: int, contract: SECONDShapeContract):
        super().__init__()
        import spconv.pytorch as spconv

        self.contract = contract
        self.last_shapes_zyx: tuple[tuple[int, int, int], ...] | None = None

        def sparse_norm_act(
            cin: int,
            cout: int,
            *,
            kernel: int | tuple[int, int, int] = 3,
            stride: int | tuple[int, int, int] = 1,
            padding: int | tuple[int, int, int] = 0,
            indice_key: str,
            subm: bool = False,
        ) -> nn.Module:
            conv_cls = spconv.SubMConv3d if subm else spconv.SparseConv3d
            conv = conv_cls(
                cin,
                cout,
                kernel,
                stride=stride,
                padding=padding,
                bias=False,
                indice_key=indice_key,
            ) if not subm else conv_cls(
                cin, cout, kernel, padding=padding, bias=False, indice_key=indice_key
            )
            return spconv.SparseSequential(conv, _gn(cout), nn.ReLU(inplace=True))

        self.stem = sparse_norm_act(
            in_channels, 16, padding=1, indice_key="second_stem", subm=True
        )
        # Custom residual blocks must be called explicitly. spconv.SparseSequential
        # treats an arbitrary nn.Module as a dense feature-only module and calls it
        # with ``input.features``; _SparseResidualBlock consumes/returns the complete
        # SparseConvTensor to preserve indices and residual semantics.
        self.stage1 = nn.ModuleList((
            _SparseResidualBlock(16, "second_s1"),
            _SparseResidualBlock(16, "second_s1"),
        ))
        self.down1 = sparse_norm_act(
            16, 32, stride=(2, 2, 2), padding=(1, 1, 1), indice_key="second_down1"
        )
        self.stage2 = nn.ModuleList((
            _SparseResidualBlock(32, "second_s2"),
            _SparseResidualBlock(32, "second_s2"),
        ))
        self.down2 = sparse_norm_act(
            32, 64, stride=(2, 2, 2), padding=(1, 1, 1), indice_key="second_down2"
        )
        self.stage3 = nn.ModuleList((
            _SparseResidualBlock(64, "second_s3"),
            _SparseResidualBlock(64, "second_s3"),
        ))
        self.down3 = sparse_norm_act(
            64, 128, stride=(2, 2, 2), padding=(0, 1, 1), indice_key="second_down3"
        )
        self.stage4 = nn.ModuleList((
            _SparseResidualBlock(128, "second_s4"),
            _SparseResidualBlock(128, "second_s4"),
        ))
        self.conv_out = sparse_norm_act(
            128,
            contract.sparse_output_channels,
            kernel=(3, 1, 1),
            stride=(2, 1, 1),
            padding=0,
            indice_key="second_z_collapse",
        )

    @staticmethod
    def _shape(x) -> tuple[int, int, int]:
        return tuple(int(v) for v in x.spatial_shape)

    @staticmethod
    def _run_residual_stage(blocks: nn.ModuleList, x):
        for block in blocks:
            x = block(x)
        return x

    def forward(self, x):
        shapes: list[tuple[int, int, int]] = []
        x = self.stem(x)
        shapes.append(self._shape(x))
        x = self._run_residual_stage(self.stage1, x)
        x = self.down1(x)
        shapes.append(self._shape(x))
        x = self._run_residual_stage(self.stage2, x)
        x = self.down2(x)
        shapes.append(self._shape(x))
        x = self._run_residual_stage(self.stage3, x)
        x = self.down3(x)
        shapes.append(self._shape(x))
        x = self._run_residual_stage(self.stage4, x)
        shapes.append(self._shape(x))
        x = self.conv_out(x)
        shapes.append(self._shape(x))
        self.last_shapes_zyx = tuple(shapes)
        expected = self.contract.stage_shapes_zyx
        if self.last_shapes_zyx != expected:
            raise RuntimeError(
                f"SECOND sparse shape drift: got {self.last_shapes_zyx}, expected {expected}"
            )
        return x


def validate_reference_stage_channels(channels: Sequence[Sequence[int]]) -> None:
    """Small fail-closed helper used by reference-mapping fixtures."""
    expected = ((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128))
    actual = tuple(tuple(int(v) for v in stage) for stage in channels)
    if actual != expected:
        raise ValueError(f"SECOND stage channels must be {expected}, got {actual}")
