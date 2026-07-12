"""Dense PointPillars LiDAR encoder (fl_v3 T2) — sparse-pillar PFN, atomic-free scatter.

Per the Architecture table: pillarize points, encode each occupied pillar with a PFN
(Linear + GroupNorm + ReLU + ``torch.max`` over points), and scatter the pillar features
into a dense ``[C, H, W]`` BEV canvas — **no spconv, no ``atomicAdd``**.

## Determinism design (SPEC §0 + lidar_encoder bullet)

* **Sparse pillars, not a dense ``[H*W, max_points, C]`` tensor** (that is ~2 GiB at
  512²). We materialize at most ``B * max_pillars`` selected pillars as
  ``[P_selected, max_points, C]``.
* **The pillar cap is per sample.** Occupied cells are ordered by their local row-major
  cell key independently inside every sample, then the first ``max_pillars`` are kept.
  Adding/reordering other samples therefore cannot consume a sample's budget or change
  its selected cells.
* **Slot assignment via a CANONICAL lexicographic sort** — ``(pillar_key, x, y, z,
  intensity)`` via successive STABLE int64/float sorts (``unique_consecutive(return_counts
  =True)`` for segment boundaries) — **NOT** float ``cumsum`` / weighted ``bincount`` /
  ``unique(return_inverse)`` for slots (tie-unstable or raising on CUDA). Over-cap
  truncation keeps the **first ``max_points`` by point CONTENT** (NOT file order), so the
  kept subset is a pure function of the points — **permutation-invariant even when a
  pillar exceeds the cap** (the only ties are exact-duplicate points, which are
  value-equivalent for ``torch.max``). *(A plain stable sort on ``pillar_key`` alone would
  keep a file-order subset → a permutation would change the max-pool; the Codex T2
  finding. The lexicographic content sort fixes it.)*
* **Permutation-invariant by construction.** The PFN consumes only **per-point**
  features ``[x, y, z, intensity, x_p, y_p, z_p]`` (raw coords + offset to the pillar
  center) — **no within-pillar cluster-mean** (a float mean whose summation order would
  drift under an input permutation). With per-point features + the canonical truncation,
  ``torch.max`` over the points dim is **value-order-independent**, so the whole encoder
  is invariant to the LiDAR point order (the permutation-invariance gate). *(The
  PointPillars cluster-center offset is dropped as a deliberate determinism trade — a
  minor accuracy feature; a canonical-order cluster mean is a deferred refinement.)*
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
    pillar; ``max_pillars`` caps occupied pillars **per sample** (over-cap pillars are
    dropped in deterministic local-cell-key order).

    After every forward, :attr:`last_pillar_meta` exposes device tensors with per-sample
    input/in-range point counts, occupied/selected/truncated pillar counts, point- and
    pillar-cap drops, truncation fractions, and the selected local cell keys. These are
    engineering diagnostics, not scientific metrics.
    """

    def __init__(
        self,
        out_channels: int = 64,
        max_points: int = 32,
        max_pillars: int = 30000,
        cfg: BEVConfig = BEVConfig(),
        use_timestamp: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.out_channels = int(out_channels)
        self.max_points = int(max_points)
        self.max_pillars = int(max_pillars)
        if self.max_points <= 0:
            raise ValueError(f"max_points must be positive, got {self.max_points}")
        if self.max_pillars <= 0:
            raise ValueError(f"max_pillars must be positive, got {self.max_pillars}")
        # MCR P1 multi-sweep: when the batched points carry a relative-timestamp column (dt, col 6
        # after the batch-index col), add it as a per-point feature (still permutation-invariant — it
        # is a per-point value like intensity, and it is added to the canonical content sort below).
        self.use_timestamp = bool(use_timestamp)
        self.in_features = 8 if self.use_timestamp else 7  # +dt for multi-sweep
        self.linear = nn.Linear(self.in_features, self.out_channels, bias=False)
        self.norm = _gn(self.out_channels)
        self.act = nn.ReLU(inplace=True)
        self.last_pillar_meta: dict[str, object] = {}

    @staticmethod
    def _count_per_sample(sample_ids: torch.Tensor, B: int) -> torch.Tensor:
        """Count ids in ``[0, B)`` without an accumulating scatter/atomic path."""
        if B <= 0:
            return sample_ids.new_zeros((0,), dtype=torch.int64)
        return torch.stack(
            [(sample_ids == sample).sum(dtype=torch.int64) for sample in range(B)]
        )

    @staticmethod
    def _sum_per_sample(values: torch.Tensor, sample_ids: torch.Tensor, B: int) -> torch.Tensor:
        """Sum integer diagnostics per sample in fixed sample order."""
        if B <= 0:
            return values.new_zeros((0,), dtype=torch.int64)
        return torch.stack([values[sample_ids == sample].sum() for sample in range(B)])

    def _record_pillar_meta(
        self,
        *,
        B: int,
        input_points: torch.Tensor,
        in_range_points: torch.Tensor,
        occupied_pillars: torch.Tensor,
        selected_pillars: torch.Tensor,
        point_cap_drops: torch.Tensor,
        pillar_cap_drops: torch.Tensor,
        kept_points: torch.Tensor,
        selected_batch_ids: torch.Tensor,
        selected_local_keys: torch.Tensor,
    ) -> None:
        truncated = occupied_pillars - selected_pillars
        fraction = truncated.to(torch.float32) / occupied_pillars.clamp_min(1).to(torch.float32)
        self.last_pillar_meta = {
            "batch_size": int(B),
            "max_points_per_pillar": int(self.max_points),
            "max_pillars_per_sample": int(self.max_pillars),
            "input_points_per_sample": input_points.detach(),
            "in_range_points_per_sample": in_range_points.detach(),
            "occupied_pillars_per_sample": occupied_pillars.detach(),
            "selected_pillars_per_sample": selected_pillars.detach(),
            "truncated_pillars_per_sample": truncated.detach(),
            "pillar_truncation_fraction_per_sample": fraction.detach(),
            "points_kept_after_caps_per_sample": kept_points.detach(),
            "points_dropped_by_point_cap_per_sample": point_cap_drops.detach(),
            "points_dropped_by_pillar_cap_per_sample": pillar_cap_drops.detach(),
            "selected_pillar_batch_ids": selected_batch_ids.detach(),
            "selected_local_pillar_keys": selected_local_keys.detach(),
        }

    def forward(self, points: torch.Tensor, B: int) -> torch.Tensor:
        cfg = self.cfg
        device = points.device
        b = points[:, 0].to(torch.int64)
        input_points_per_sample = self._count_per_sample(b, B)
        xyz = points[:, 1:4]
        intensity = points[:, 4:5]
        dt = points[:, 6:7] if self.use_timestamp else None   # multi-sweep relative-timestamp channel
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        # --- in-range mask + pillar (col,row) on the FINE grid (shared convention) ---
        col, row = metric_to_grid(x, y, cfg.x_min, cfg.y_min, cfg.vx, cfg.vy)
        keep = (
            in_grid_mask(col, row, cfg.nx, cfg.ny)
            & (z >= cfg.z_min)
            & (z < cfg.z_max)
        )
        in_range_points_per_sample = self._count_per_sample(b[keep], B)
        if keep.sum() == 0:
            zero = b.new_zeros((B,), dtype=torch.int64)
            self._record_pillar_meta(
                B=B,
                input_points=input_points_per_sample,
                in_range_points=in_range_points_per_sample,
                occupied_pillars=zero,
                selected_pillars=zero,
                point_cap_drops=zero,
                pillar_cap_drops=zero,
                kept_points=zero,
                selected_batch_ids=b.new_zeros((0,), dtype=torch.int64),
                selected_local_keys=b.new_zeros((0,), dtype=torch.int64),
            )
            return points.new_zeros((B, self.out_channels, cfg.ny, cfg.nx))
        b, col, row = b[keep], col[keep], row[keep]
        xyz, intensity = xyz[keep], intensity[keep]
        if dt is not None:
            dt = dt[keep]

        # global pillar key (batch offset so pillars don't collide across the batch)
        pillar_key = b * (cfg.nx * cfg.ny) + flat_index(col, row, cfg.nx)  # [Nk]

        # --- group by pillar; CANONICAL within-pillar order by point CONTENT (NOT file
        #     order) so the over-cap truncation keeps a PERMUTATION-INVARIANT subset ---
        # A plain stable sort on pillar_key alone preserves the incoming order within a
        # pillar, so `within < max_points` would keep a DIFFERENT subset after a LiDAR
        # point permutation (a different max-pool → a different BEV; the Codex T2 finding).
        # We instead lexicographically sort (pillar_key primary, then x, y, z, intensity)
        # via successive STABLE sorts, least-significant key first. Within a pillar the
        # kept first `max_points` are then a pure function of point CONTENT — independent
        # of input order — and the only ties (exact-duplicate points) are value-equivalent
        # for `torch.max`, so a point permutation yields a byte-identical canvas even when
        # a pillar exceeds the cap.
        # dt (when present) is the LEAST-significant tie-breaker (sorted first; later stable sorts
        # dominate) so the canonical content order — and the over-cap truncation — stays a pure
        # function of point CONTENT including its sweep time.
        sort_keys = [intensity[:, 0], xyz[:, 2], xyz[:, 1], xyz[:, 0], pillar_key]
        if dt is not None:
            sort_keys = [dt[:, 0]] + sort_keys
        order = torch.arange(pillar_key.numel(), device=device)
        for key in sort_keys:
            order = order[torch.argsort(key[order], stable=True)]
        pk_s = pillar_key[order]
        xyz_s, int_s = xyz[order], intensity[order]
        col_s, row_s = col[order], row[order]
        dt_s = dt[order] if dt is not None else None

        uniq_keys, counts = torch.unique_consecutive(pk_s, return_counts=True)  # [P], [P]
        P = uniq_keys.numel()
        # within-pillar index for each sorted point: i - segment_offset[pillar_of_i]
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])       # [P]
        pillar_of = torch.repeat_interleave(torch.arange(P, device=device), counts)  # [Nk]
        within = torch.arange(pk_s.numel(), device=device) - offsets[pillar_of]  # [Nk]
        # --- deterministic PER-SAMPLE pillar cap, before PFN/slot materialization ---
        # uniq_keys is globally sorted as (batch, local row-major cell), so equal sample ids
        # are consecutive and local ranks are independent of every other sample's occupancy.
        cells_per_sample = cfg.nx * cfg.ny
        pillar_batch = torch.div(uniq_keys, cells_per_sample, rounding_mode="floor")
        present_samples, pillars_per_present_sample = torch.unique_consecutive(
            pillar_batch, return_counts=True
        )
        occupied_pillars_per_sample = b.new_zeros((B,), dtype=torch.int64)
        occupied_pillars_per_sample.index_copy_(
            0, present_samples, pillars_per_present_sample.to(torch.int64)
        )
        sample_offsets = torch.cat(
            [pillars_per_present_sample.new_zeros(1), pillars_per_present_sample.cumsum(0)[:-1]]
        )
        sample_group = torch.repeat_interleave(
            torch.arange(present_samples.numel(), device=device), pillars_per_present_sample
        )
        pillar_rank_in_sample = torch.arange(P, device=device) - sample_offsets[sample_group]
        select_pillar = pillar_rank_in_sample < self.max_pillars
        selected_keys = uniq_keys[select_pillar]
        selected_batch = pillar_batch[select_pillar]
        selected_local_keys = selected_keys % cells_per_sample
        selected_pillars_per_sample = occupied_pillars_per_sample.clamp_max(self.max_pillars)

        # Points in dropped pillars are accounted separately from within-pillar point-cap
        # drops. Both summaries are per sample and therefore expose otherwise silent loss.
        pillar_cap_drops = self._sum_per_sample(
            counts[~select_pillar], pillar_batch[~select_pillar], B
        )
        selected_counts = counts[select_pillar]
        point_cap_drops = self._sum_per_sample(
            (selected_counts - self.max_points).clamp_min(0), selected_batch, B
        )
        kept_points = self._sum_per_sample(
            selected_counts.clamp_max(self.max_points), selected_batch, B
        )
        self._record_pillar_meta(
            B=B,
            input_points=input_points_per_sample,
            in_range_points=in_range_points_per_sample,
            occupied_pillars=occupied_pillars_per_sample,
            selected_pillars=selected_pillars_per_sample,
            point_cap_drops=point_cap_drops,
            pillar_cap_drops=pillar_cap_drops,
            kept_points=kept_points,
            selected_batch_ids=selected_batch,
            selected_local_keys=selected_local_keys,
        )

        # Map original pillar ids to the compact selected-pillar axis. Integer cumsum is
        # deterministic; only points whose pillar is selected consume a slot/PFN feature.
        selected_rank = select_pillar.to(torch.int64).cumsum(0) - 1
        cap = select_pillar[pillar_of] & (within < self.max_points)
        pillar_of_c = selected_rank[pillar_of[cap]]
        within_c = within[cap]
        xyz_c, int_c = xyz_s[cap], int_s[cap]
        col_c, row_c = col_s[cap], row_s[cap]
        dt_c = dt_s[cap] if dt_s is not None else None

        # --- per-point features (no aggregation → permutation-invariant) ---
        px = (col_c.to(torch.float32) + 0.5) * cfg.vx + cfg.x_min
        py = (row_c.to(torch.float32) + 0.5) * cfg.vy + cfg.y_min
        zc = 0.5 * (cfg.z_min + cfg.z_max)
        cols = [xyz_c[:, 0], xyz_c[:, 1], xyz_c[:, 2], int_c[:, 0]]
        if dt_c is not None:
            cols.append(dt_c[:, 0])                              # +dt for multi-sweep (8-dim)
        cols += [xyz_c[:, 0] - px, xyz_c[:, 1] - py, xyz_c[:, 2] - zc]
        feat = torch.stack(cols, dim=1)  # [Ncapped, 7 or 8]

        # PFN: Linear → GroupNorm(per-point) → ReLU
        h = self.linear(feat)                                   # [Ncapped, C]
        h = self.norm(h.unsqueeze(-1)).squeeze(-1)              # GN over channels, per point
        h = self.act(h)

        # --- scatter into [P_selected, max_points, C] at unique (pillar, slot) → max ---
        P_selected = selected_keys.numel()
        slot = pillar_of_c * self.max_points + within_c         # unique per (pillar, within)
        dense = h.new_full((P_selected * self.max_points, self.out_channels), float("-inf"))
        dense.index_copy_(0, slot, h)                           # unique → assignment (#76176-safe)
        dense = dense.view(P_selected, self.max_points, self.out_channels)
        pillar_feat = dense.max(dim=1).values             # [P_selected, C] (-inf pads never win)

        # --- scatter pillars into the dense BEV canvas (unique keys → assignment) ---
        # injective-cell invariant (pillar identity == cell identity). uniq_keys comes from
        # unique_consecutive on a sorted key ⇒ unique BY CONSTRUCTION, so this torch.unique (a full
        # extra sort over ~28k keys/step) is redundant on the hot path — keep it only under the strict
        # offline dev-regression path (cudnn.deterministic), drop it from the relaxed fp16 path.
        if torch.backends.cudnn.deterministic:
            assert torch.unique(selected_keys).numel() == selected_keys.numel(), "pillar keys not unique"
        canvas = pillar_feat.new_zeros((B * cfg.ny * cfg.nx, self.out_channels))
        canvas.index_copy_(0, selected_keys, pillar_feat)
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
