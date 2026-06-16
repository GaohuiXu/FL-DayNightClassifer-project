"""Dense PointPillars LiDAR encoder (fl_v3 T2) — sparse-pillar PFN, atomic-free scatter.

Per the Architecture table: pillarize points, encode each occupied pillar with a PFN
(Linear + GroupNorm + ReLU + ``torch.max`` over points), and scatter the pillar features
into a dense ``[C, H, W]`` BEV canvas — **no spconv, no ``atomicAdd``**.

## Determinism design (SPEC §0 + lidar_encoder bullet)

* **Sparse pillars, not a dense ``[H*W, max_points, C]`` tensor** (that is ~2 GiB at
  512²). We materialize only the ``P`` occupied pillars as ``[P, max_points, C]``.
* **Slot assignment via a STABLE int64 sort** of the global pillar key
  (``unique_consecutive(return_counts=True)`` for segment boundaries) — **NOT** float
  ``cumsum`` / weighted ``bincount`` / ``unique(return_inverse)`` for slots (tie-unstable
  or raising on CUDA). Over-cap truncation keeps the **first ``max_points`` in file
  order** (the stable sort preserves it).
* **Permutation-invariant by construction.** The PFN consumes only **per-point**
  features ``[x, y, z, intensity, x_p, y_p, z_p]`` (raw coords + offset to the pillar
  center) — **no within-pillar cluster-mean** (a float mean whose summation order would
  drift under an input permutation). With per-point features, ``torch.max`` over the
  points dim is **value-order-independent**, so the whole encoder is invariant to the
  LiDAR point order (the permutation-invariance gate). *(The PointPillars cluster-center
  offset is dropped as a deliberate determinism trade — a minor accuracy feature; a
  canonical-order cluster mean is a deferred refinement.)*
* **Scatter via ``index_copy_``** on the flat canvas at the **unique** pillar keys
  (pillar identity == cell identity ⇒ injective) — assignment, not accumulation;
  **never** ``canvas[:, idx] = voxels`` (#76176 silent no-op on CUDA). Index uniqueness
  is asserted at runtime.
* **No ``Adaptive*Pool2d``** (its CUDA backward raises); the ``torch.max`` is the value
  path (argmax-index nondeterminism is irrelevant).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import BEVConfig, metric_to_grid, in_grid_mask, flat_index


def _gn(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class PointPillarsEncoder(nn.Module):
    """Points → dense ``[B, C, ny, nx]`` LiDAR BEV (sparse-pillar PFN + index_copy scatter).

    ``points`` is the batched cloud ``[TotalP, 6]`` = ``(batch_idx, x, y, z, intensity,
    ring)`` (the :mod:`collate` batch-index-column format). ``max_points`` caps points per
    pillar; ``max_pillars`` caps occupied pillars (over-cap pillars dropped in key order)."""

    def __init__(
        self,
        out_channels: int = 64,
        max_points: int = 32,
        max_pillars: int = 30000,
        cfg: BEVConfig = BEVConfig(),
    ):
        super().__init__()
        self.cfg = cfg
        self.out_channels = int(out_channels)
        self.max_points = int(max_points)
        self.max_pillars = int(max_pillars)
        self.in_features = 7  # x,y,z,intensity,x_p,y_p,z_p (per-point only)
        self.linear = nn.Linear(self.in_features, self.out_channels, bias=False)
        self.norm = _gn(self.out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points: torch.Tensor, B: int) -> torch.Tensor:
        cfg = self.cfg
        device = points.device
        b = points[:, 0].to(torch.int64)
        xyz = points[:, 1:4]
        intensity = points[:, 4:5]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # --- in-range mask + pillar (col,row) on the FINE grid (shared convention) ---
        col, row = metric_to_grid(x, y, cfg.x_min, cfg.y_min, cfg.vx, cfg.vy)
        keep = (
            in_grid_mask(col, row, cfg.nx, cfg.ny)
            & (z >= cfg.z_min)
            & (z < cfg.z_max)
        )
        if keep.sum() == 0:
            return points.new_zeros((B, self.out_channels, cfg.ny, cfg.nx))
        b, col, row = b[keep], col[keep], row[keep]
        xyz, intensity = xyz[keep], intensity[keep]

        # global pillar key (batch offset so pillars don't collide across the batch)
        pillar_key = b * (cfg.nx * cfg.ny) + flat_index(col, row, cfg.nx)  # [Nk]

        # --- group by pillar via a STABLE sort (file order preserved within a pillar) ---
        order = torch.argsort(pillar_key, stable=True)
        pk_s = pillar_key[order]
        xyz_s, int_s = xyz[order], intensity[order]
        col_s, row_s = col[order], row[order]

        uniq_keys, counts = torch.unique_consecutive(pk_s, return_counts=True)  # [P], [P]
        P = uniq_keys.numel()
        # within-pillar index for each sorted point: i - segment_offset[pillar_of_i]
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])       # [P]
        pillar_of = torch.repeat_interleave(torch.arange(P, device=device), counts)  # [Nk]
        within = torch.arange(pk_s.numel(), device=device) - offsets[pillar_of]  # [Nk]
        cap = within < self.max_points                                         # file-order truncation
        pillar_of_c = pillar_of[cap]
        within_c = within[cap]
        xyz_c, int_c = xyz_s[cap], int_s[cap]
        col_c, row_c = col_s[cap], row_s[cap]

        # --- per-point features (no aggregation → permutation-invariant) ---
        px = (col_c.to(torch.float32) + 0.5) * cfg.vx + cfg.x_min
        py = (row_c.to(torch.float32) + 0.5) * cfg.vy + cfg.y_min
        zc = 0.5 * (cfg.z_min + cfg.z_max)
        feat = torch.stack(
            [
                xyz_c[:, 0], xyz_c[:, 1], xyz_c[:, 2], int_c[:, 0],
                xyz_c[:, 0] - px, xyz_c[:, 1] - py, xyz_c[:, 2] - zc,
            ],
            dim=1,
        )  # [Ncapped, 7]

        # PFN: Linear → GroupNorm(per-point) → ReLU
        h = self.linear(feat)                                   # [Ncapped, C]
        h = self.norm(h.unsqueeze(-1)).squeeze(-1)              # GN over channels, per point
        h = self.act(h)

        # --- scatter into [P, max_points, C] at unique (pillar, slot) → max over points ---
        slot = pillar_of_c * self.max_points + within_c         # unique per (pillar, within)
        dense = h.new_full((P * self.max_points, self.out_channels), float("-inf"))
        dense.index_copy_(0, slot, h)                           # unique → assignment (#76176-safe)
        dense = dense.view(P, self.max_points, self.out_channels)
        pillar_feat = dense.max(dim=1).values                   # [P, C] (value path; -inf pads never win)

        # --- scatter pillars into the dense BEV canvas (unique keys → assignment) ---
        if P > self.max_pillars:
            pillar_feat = pillar_feat[: self.max_pillars]
            uniq_keys = uniq_keys[: self.max_pillars]
        # injective-cell invariant (pillar identity == cell identity)
        assert torch.unique(uniq_keys).numel() == uniq_keys.numel(), "pillar keys not unique"
        canvas = pillar_feat.new_zeros((B * cfg.ny * cfg.nx, self.out_channels))
        canvas.index_copy_(0, uniq_keys, pillar_feat)
        bev = canvas.view(B, cfg.ny, cfg.nx, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return bev

    def occupancy(self, points: torch.Tensor, B: int) -> torch.Tensor:
        """Per-cell pillar point-count ``[B, ny, nx]`` (RNG- + atomic-free; for V2).

        Counts via a STABLE sort + ``unique_consecutive(return_counts)`` + ``index_copy_``
        at the unique cells — the same atomic-free pattern as :meth:`forward` (NO
        ``bincount`` / ``index_put(accumulate=True)``, which raise or use atomics)."""
        cfg = self.cfg
        b = points[:, 0].to(torch.int64)
        x, y, z = points[:, 1], points[:, 2], points[:, 3]
        col, row = metric_to_grid(x, y, cfg.x_min, cfg.y_min, cfg.vx, cfg.vy)
        keep = in_grid_mask(col, row, cfg.nx, cfg.ny) & (z >= cfg.z_min) & (z < cfg.z_max)
        occ = points.new_zeros((B * cfg.ny * cfg.nx,))
        if keep.sum() == 0:
            return occ.view(B, cfg.ny, cfg.nx)
        key = b[keep] * (cfg.nx * cfg.ny) + flat_index(col[keep], row[keep], cfg.nx)
        key_s = key[torch.argsort(key, stable=True)]
        uniq, counts = torch.unique_consecutive(key_s, return_counts=True)
        occ.index_copy_(0, uniq, counts.to(occ.dtype))
        return occ.view(B, cfg.ny, cfg.nx)
