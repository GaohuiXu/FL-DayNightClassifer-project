"""Sparse 3D-voxel LiDAR encoder (SECOND/CenterPoint-voxel style) — the Rule#2-relaxed spconv path.

The pillar PFN collapses z into ONE bin (its documented ceiling vs SOTA — see the structural audit). This
encoder gives the LiDAR branch real **z-resolution**: voxelize the point cloud, a mean-VFE, a sparse 3D conv
backbone that **keeps xy at the shared fine grid (SubM convs) and downsamples only z**, then collapses the thin
residual z into channels → a dense ``[B, out_channels, ny, nx]`` BEV. It is a **drop-in replacement for
``PointPillarsEncoder``** (same call/return), so the existing ``LidarBackbone2D`` + ``ConvFuser`` + head are
unchanged and the comparison isolates pillars→voxels at the matched grid.

Rule #2 relaxation (owner, 2026-06-28; Phase 0A de-risked spconv on Arrhenius): spconv kernels are **non-
deterministic** (fine under D16's seed-variance regime, NOT the strict byte-id dev tool) and do **not support
bf16** (Phase 0A) — so the spconv stack runs **autocast-DISABLED in fp32**; the dense BEV re-enters the model's
fp16 autocast for fusion. Requires ``spconv-cu126`` + ``unset BOOST_ROOT`` (the EasyBuild Boost confuses
spconv's import check). Gated by ``det-lidar-encoder=voxel`` (default ``pillar`` ⇒ this module is never built).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import BEVConfig, flat_index, in_grid_mask, metric_to_grid


def _gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    # >=2 channels per group. A 1-channel-per-group GroupNorm on a NO-SPATIAL tensor (the per-point VFE +
    # the sparse [V,C] features) normalizes each channel over its single value => 0 (collapse, the voxel-0.26
    # bug). Capping g at channels//2 guarantees >=2 channels/group, so normalization is well-defined.
    g = min(max_groups, max(1, channels // 2))
    while channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class SparseVoxelEncoder(nn.Module):
    """Voxelize → mean-VFE → sparse 3D backbone (z-downsampling) → collapse-z → ``[B, out_channels, ny, nx]``."""

    def __init__(self, out_channels: int = 64, cfg: BEVConfig = BEVConfig(), use_timestamp: bool = False,
                 vfe_channels: int = 16, z_voxel: float | None = None, max_voxels: int = 120000,
                 max_points_per_voxel: int = 10):
        super().__init__()
        import spconv.pytorch as spconv          # import-gated: only when this encoder is actually built
        # NOTE: do NOT store the spconv MODULE on self — it is not deep-copyable, and centralized EMA
        # (AveragedModel) deep-copies the model ("cannot pickle 'module' object"). Re-import locally in forward.
        self.cfg = cfg
        self.out_channels = int(out_channels)
        self.use_timestamp = bool(use_timestamp)
        self.n_pt_feat = 5 if use_timestamp else 4          # x,y,z,intensity (+dt)
        self.vx, self.vy = cfg.vx, cfg.vy
        self.vz = float(z_voxel) if z_voxel else cfg.vx     # cubic voxels by default (z-res = xy-res)
        self.nz = int(round((cfg.z_max - cfg.z_min) / self.vz))
        self.nx, self.ny = cfg.nx, cfg.ny
        self.max_voxels = int(max_voxels)
        self.max_pts = int(max_points_per_voxel)
        self._voxelizer = None                              # built lazily (device-bound)
        self.record_debug = False                           # opt-in only; keeps the hot path sync-free
        self.last_sparse_meta: dict | None = None

        # PFN-style per-point VFE: per-point [abs xyz, intensity(+dt), voxel-center-relative xyz] → Linear → GN →
        # ReLU → masked max-pool per voxel. (The earlier naive mean-VFE collapsed each voxel to ~[center,
        # mean-intensity] — info-free; the relative offsets + max-pool are what make the LiDAR features useful,
        # matching the proven PointPillarsEncoder.)
        self.vfe_point = nn.Sequential(nn.Linear(self.n_pt_feat + 3, vfe_channels), _gn(vfe_channels),
                                       nn.ReLU(inplace=True))

        # sparse 3D backbone: SubM (keep res) + z-downsampling SparseConv (stride (2,1,1) on the z axis only).
        sp = spconv

        def subm(cin, cout, key):
            return sp.SparseSequential(sp.SubMConv3d(cin, cout, 3, padding=1, bias=False, indice_key=key),
                                       _gn(cout), nn.ReLU(inplace=True))

        def down_z(cin, cout):
            return sp.SparseSequential(sp.SparseConv3d(cin, cout, 3, stride=(2, 1, 1), padding=1, bias=False),
                                       _gn(cout), nn.ReLU(inplace=True))

        ch_schedule = [32, 64, 64, 64, 64, 64, 64]          # channels after each z-downsample (capped)
        layers = [subm(vfe_channels, 16, "sub_in"), subm(16, 16, "sub_in")]
        ch, z, stage = 16, self.nz, 0
        while z > 2 and stage < len(ch_schedule):
            cout = ch_schedule[stage]
            layers.append(down_z(ch, cout))
            ch = cout
            layers.append(subm(ch, ch, f"sub{stage}"))
            z = (z - 1) // 2 + 1                             # stride-2, pad-1, kernel-3 output length
            stage += 1
        self.backbone = sp.SparseSequential(*layers)
        self.z_final, self.ch_final = z, ch

        # collapse the residual z into channels, conv to out_channels (runs in the outer autocast).
        self.to_bev = nn.Sequential(nn.Conv2d(ch * z, out_channels, 1, bias=False),
                                    _gn(out_channels), nn.ReLU(inplace=True))

    def _build_voxelizer(self, device: torch.device) -> None:
        from spconv.pytorch.utils import PointToVoxel
        c = self.cfg
        self._voxelizer = PointToVoxel(
            vsize_xyz=[self.vx, self.vy, self.vz],
            coors_range_xyz=[c.x_min, c.y_min, c.z_min, c.x_max, c.y_max, c.z_max],
            num_point_features=self.n_pt_feat, max_num_voxels=self.max_voxels,
            max_num_points_per_voxel=self.max_pts, device=device)

    def forward(self, points: torch.Tensor, B: int) -> torch.Tensor:
        """``points`` ``[TotalP, 1+W]`` (col0 batch, col1:4 xyz, col4 intensity, col6 dt) → ``[B, out, ny, nx]``."""
        dev = points.device
        if self._voxelizer is None:
            self._build_voxelizer(dev)
        import spconv.pytorch as sp              # local (not stored on self — see __init__ note re: EMA deepcopy)
        feat_cols = [1, 2, 3, 4] + ([6] if self.use_timestamp else [])
        bidx = points[:, 0].to(torch.int64)
        # spconv: fp32, autocast OFF (no bf16; allowed inside the relaxed fp16 Arrhenius regime).
        with torch.autocast(device_type=dev.type, enabled=False):
            vfeats, coords_b = [], []
            for b in range(B):
                pf = points[bidx == b][:, feat_cols].to(torch.float32)          # [P,F], xyz first 3
                if pf.shape[0] == 0:
                    continue
                voxels, coords, num_p = self._voxelizer(pf)                     # [V,P,F], [V,3]=(z,y,x), [V]
                V, P = voxels.shape[0], voxels.shape[1]
                # voxel center (x,y,z) from coords (z,y,x) → per-point voxel-center-relative offset
                xc = (coords[:, 2].float() + 0.5) * self.vx + self.cfg.x_min
                yc = (coords[:, 1].float() + 0.5) * self.vy + self.cfg.y_min
                zc = (coords[:, 0].float() + 0.5) * self.vz + self.cfg.z_min
                rel = voxels[:, :, :3] - torch.stack([xc, yc, zc], dim=1).view(V, 1, 3)   # [V,P,3]
                ppf = torch.cat([voxels, rel], dim=2)                            # [V,P,F+3] abs+intensity(+dt)+rel
                valid = torch.arange(P, device=dev).view(1, P) < num_p.view(V, 1)         # [V,P] mask padded
                fp = self.vfe_point(ppf.reshape(V * P, -1)).reshape(V, P, -1)    # [V,P,C] per-point
                fp = fp.masked_fill(~valid.unsqueeze(-1), float("-inf"))
                vfeats.append(torch.nan_to_num(fp.max(dim=1).values, neginf=0.0))         # [V,C] masked max-pool
                bcol = torch.full((coords.shape[0], 1), b, dtype=coords.dtype, device=dev)
                coords_b.append(torch.cat([bcol, coords], dim=1))               # [V,4]=(b,z,y,x)
            if not vfeats:
                if self.record_debug:
                    self.last_sparse_meta = {
                        "coord_order": "bzyx",
                        "indices_shape": (0, 4),
                        "indices_dtype": "torch.int32",
                        "features_shape": (0, self.vfe_point[0].out_features),
                        "features_dtype": "torch.float32",
                        "spatial_shape": (self.nz, self.ny, self.nx),
                        "batch_size": int(B),
                        "num_voxels": 0,
                    }
                return self._empty_bev(points, B)
            features = torch.cat(vfeats, 0).to(torch.float32)
            indices = torch.cat(coords_b, 0).to(torch.int32)
            if self.record_debug:
                self.last_sparse_meta = {
                    "coord_order": "bzyx",
                    "indices_shape": tuple(indices.shape),
                    "indices_dtype": str(indices.dtype),
                    "features_shape": tuple(features.shape),
                    "features_dtype": str(features.dtype),
                    "spatial_shape": (self.nz, self.ny, self.nx),
                    "batch_size": int(B),
                    "num_voxels": int(indices.shape[0]),
                    "batch_index_min": int(indices[:, 0].min().item()),
                    "batch_index_max": int(indices[:, 0].max().item()),
                    "z_min": int(indices[:, 1].min().item()),
                    "z_max": int(indices[:, 1].max().item()),
                    "y_min": int(indices[:, 2].min().item()),
                    "y_max": int(indices[:, 2].max().item()),
                    "x_min": int(indices[:, 3].min().item()),
                    "x_max": int(indices[:, 3].max().item()),
                }
            x = sp.SparseConvTensor(features, indices, spatial_shape=[self.nz, self.ny, self.nx], batch_size=B)
            x = self.backbone(x)
            dense = x.dense()                                                   # [B, ch, z_final, ny, nx]
            dense = dense.reshape(B, self.ch_final * self.z_final, self.ny, self.nx)
        return self.to_bev(dense)                                              # outer autocast casts to model dtype

    def _empty_bev(self, points: torch.Tensor, B: int) -> torch.Tensor:
        """Zero BEV for all-empty batches, with a zero-gradient anchor for DDP safety."""
        out = points.new_zeros((B, self.out_channels, self.ny, self.nx))
        if not self.training:
            return out
        anchor = None
        for p in self.parameters():
            term = p.float().sum() * 0.0
            anchor = term if anchor is None else anchor + term
        return out if anchor is None else out + anchor.to(dtype=out.dtype)

    def occupancy(self, points: torch.Tensor, B: int) -> torch.Tensor:
        """Per-cell in-range LiDAR point count ``[B, ny, nx]`` for sparse-encoder viz/smoke.

        This mirrors :meth:`PointPillarsEncoder.occupancy` at the BEV-cell level so
        V2 diagnostics do not silently assume the dense pillar implementation.
        """
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
