"""The single shared BEV grid convention (fl_v3 T2) — the T2↔T4↔T5 contract.

**This module is the one place the metric↔grid mapping is defined.** The LSS camera
splat, the PointPillars LiDAR scatter, the CenterPoint target rendering, and the
decode ALL call these functions — there is no second, independently-written mapping
anywhere in ``models/fusion/`` (the no-oracle "self-consistent-but-wrong" trap from
the SPEC §5 is defeated by anchoring this single mapping to T1's verified geometry
with a *ground-truth* test, not a self-consistency check).

## The convention (declared once; everything obeys it)

A BEV tensor is laid out ``[B, C, H, W]`` with::

    W axis (columns) ← x   (forward, LIDAR_TOP +x)
    H axis (rows)    ← y   (left,    LIDAR_TOP +y)

so for a metric point ``(x, y)`` in the canonical LIDAR_TOP frame (T1 §1):

    col = floor((x - x_min) / vx)     # the W index
    row = floor((y - y_min) / vy)     # the H index
    flat = row * W + col              # row-major over [H, W]

and the inverse (cell center, used by decode)::

    x = (col + 0.5) * vx + x_min
    y = (row + 0.5) * vy + y_min

The **head** grid is coarser than the BEV/pillar grid by ``out_size_factor`` — the
effective voxel there is ``bev_voxel * out_size_factor``; the *convention itself*
(``W→x``, ``H→y``, ``flat = row*W + col``, floor-binning, half-cell-center decode) is
identical at both resolutions. The splat/scatter use the fine grid (``cfg.nx/ny``);
the target/decode use the head grid (``cfg.head_nx/head_ny``). That shared convention
— not the resolution — is what "the single ``(x,y)→(row,col)`` mapping" means.

## Why box yaw is NOT touched here

The boxes stay in the **T1 canonical** parameterization ``(cx,cy,cz,dx=l,dy=w,dz=h,
yaw)`` end-to-end (T1 conventions §3). This module maps only the *planar center*
``(x,y)`` to/from grid cells; the head regresses ``(sin yaw, cos yaw)`` and decode
recovers ``yaw = atan2(sin, cos)`` with **NO** mmdet3d ``-π/2`` offset and **NO**
``(l,w,h)`` column swap (both banned by T1 and re-banned by the encode→decode golden).

All functions are pure / RNG-free and operate on torch tensors (CPU or CUDA);
``floor`` and integer arithmetic are bit-deterministic and summation-order-free.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class BEVConfig:
    """The frozen BEV geometry shared by the whole detector.

    ``point_cloud_range`` = ``(x_min, y_min, z_min, x_max, y_max, z_max)`` in the
    canonical LIDAR_TOP frame (metres). ``bev_voxel`` = ``(vx, vy)`` of the fine
    BEV/pillar/camera grid (both camera-BEV and LiDAR-BEV live on this grid so the
    ``ConvFuser`` can channel-concat them). ``out_size_factor`` is the head-grid
    stride relative to the fine grid (head cell = ``bev_voxel * out_size_factor``).
    """

    point_cloud_range: Tuple[float, float, float, float, float, float] = (
        -51.2, -51.2, -5.0, 51.2, 51.2, 3.0,
    )
    bev_voxel: Tuple[float, float] = (0.4, 0.4)
    out_size_factor: int = 2

    # --- derived geometry (the W→x, H→y layout) ---
    @property
    def x_min(self) -> float:
        return float(self.point_cloud_range[0])

    @property
    def y_min(self) -> float:
        return float(self.point_cloud_range[1])

    @property
    def z_min(self) -> float:
        return float(self.point_cloud_range[2])

    @property
    def x_max(self) -> float:
        return float(self.point_cloud_range[3])

    @property
    def y_max(self) -> float:
        return float(self.point_cloud_range[4])

    @property
    def z_max(self) -> float:
        return float(self.point_cloud_range[5])

    @property
    def vx(self) -> float:
        return float(self.bev_voxel[0])

    @property
    def vy(self) -> float:
        return float(self.bev_voxel[1])

    @property
    def nx(self) -> int:
        """Fine-grid width (W axis, x)."""
        return int(round((self.x_max - self.x_min) / self.vx))

    @property
    def ny(self) -> int:
        """Fine-grid height (H axis, y)."""
        return int(round((self.y_max - self.y_min) / self.vy))

    @property
    def head_vx(self) -> float:
        return self.vx * self.out_size_factor

    @property
    def head_vy(self) -> float:
        return self.vy * self.out_size_factor

    @property
    def head_nx(self) -> int:
        if self.nx % self.out_size_factor != 0:
            raise ValueError(
                f"nx={self.nx} not divisible by out_size_factor={self.out_size_factor}"
            )
        return self.nx // self.out_size_factor

    @property
    def head_ny(self) -> int:
        if self.ny % self.out_size_factor != 0:
            raise ValueError(
                f"ny={self.ny} not divisible by out_size_factor={self.out_size_factor}"
            )
        return self.ny // self.out_size_factor


# ---------------------------------------------------------------------------
# The single mapping (resolution is a parameter; the convention is fixed).
# ---------------------------------------------------------------------------
def metric_to_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    x_min: float,
    y_min: float,
    vx: float,
    vy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(x, y)`` metres → ``(col, row)`` int64 grid indices.

    ``col`` is the **W** index (← x), ``row`` is the **H** index (← y). ``floor``
    binning; out-of-range indices are returned as-is (caller masks with
    :func:`in_grid_mask`). Deterministic: ``floor`` + integer cast only.
    """
    col = torch.floor((x - x_min) / vx).to(torch.int64)
    row = torch.floor((y - y_min) / vy).to(torch.int64)
    return col, row


def grid_center_to_metric(
    col: torch.Tensor,
    row: torch.Tensor,
    x_min: float,
    y_min: float,
    vx: float,
    vy: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(col, row)`` → metric ``(x, y)`` of the cell CENTER (half-cell offset).

    The inverse used by decode. ``col``/``row`` may be int or float (decode adds a
    sub-cell regression offset, so the float path is real)."""
    x = (col.to(torch.float64) + 0.5) * vx + x_min
    y = (row.to(torch.float64) + 0.5) * vy + y_min
    return x, y


def in_grid_mask(
    col: torch.Tensor, row: torch.Tensor, nx: int, ny: int
) -> torch.Tensor:
    """Boolean mask of indices inside ``[0, nx) × [0, ny)``."""
    return (col >= 0) & (col < nx) & (row >= 0) & (row < ny)


def flat_index(col: torch.Tensor, row: torch.Tensor, nx: int) -> torch.Tensor:
    """``(col, row) → row*nx + col`` — row-major over ``[H=ny, W=nx]`` (W→x, H→y)."""
    return row * nx + col


def flat_to_colrow(flat: torch.Tensor, nx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`flat_index`: ``flat → (col=flat%nx, row=flat//nx)``."""
    col = flat % nx
    row = torch.div(flat, nx, rounding_mode="floor")
    return col, row


# --- convenience wrappers at the two resolutions the model actually uses ---
def points_to_fine_grid(
    xy: torch.Tensor, cfg: BEVConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``xy`` ``[N,2]`` → ``(col, row)`` on the FINE grid (splat + pillar scatter)."""
    return metric_to_grid(xy[:, 0], xy[:, 1], cfg.x_min, cfg.y_min, cfg.vx, cfg.vy)


def points_to_head_grid(
    xy: torch.Tensor, cfg: BEVConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``xy`` ``[N,2]`` → ``(col, row)`` on the HEAD grid (target rendering)."""
    return metric_to_grid(xy[:, 0], xy[:, 1], cfg.x_min, cfg.y_min, cfg.head_vx, cfg.head_vy)


def head_grid_to_metric(
    col: torch.Tensor, row: torch.Tensor, cfg: BEVConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(col, row)`` on the HEAD grid → metric cell-CENTER ``(x, y)`` (viz / zero-offset)."""
    return grid_center_to_metric(col, row, cfg.x_min, cfg.y_min, cfg.head_vx, cfg.head_vy)


def head_decode_to_metric(
    col: torch.Tensor, offset_x: torch.Tensor,
    row: torch.Tensor, offset_y: torch.Tensor, cfg: BEVConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode inverse of the loss encoding (CenterPoint convention, NO half-cell).

    The loss encodes ``offset = fx − floor(fx)`` ∈ [0,1) where ``fx = (cx−x_min)/head_vx``;
    decode recovers ``cx = (col + offset_x)·head_vx + x_min`` **exactly** (the
    encode→decode golden). This is the ``+offset`` convention, NOT the ``+0.5`` cell
    center used for bare-index viz."""
    x = (col.to(torch.float64) + offset_x.to(torch.float64)) * cfg.head_vx + cfg.x_min
    y = (row.to(torch.float64) + offset_y.to(torch.float64)) * cfg.head_vy + cfg.y_min
    return x, y
