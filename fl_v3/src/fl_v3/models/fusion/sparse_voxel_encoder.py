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
autocast (bf16/fp16) for fusion. Requires ``spconv-cu126`` + ``unset BOOST_ROOT`` (the EasyBuild Boost confuses
spconv's import check). Gated by ``det-lidar-encoder=voxel`` (default ``pillar`` ⇒ this module is never built).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import BEVConfig


def _gn(channels: int, max_groups: int = 16) -> nn.GroupNorm:
    g = min(max_groups, channels)
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
        self._spconv = spconv
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

        # mean-VFE → embed
        self.vfe = nn.Sequential(nn.Linear(self.n_pt_feat, vfe_channels), _gn(vfe_channels), nn.ReLU(inplace=True))

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
        sp = self._spconv
        feat_cols = [1, 2, 3, 4] + ([6] if self.use_timestamp else [])
        bidx = points[:, 0].to(torch.int64)
        # spconv: fp32, autocast OFF (no bf16; non-deterministic ok under D16).
        with torch.autocast(device_type=dev.type, enabled=False):
            vfeats, coords_b = [], []
            for b in range(B):
                pf = points[bidx == b][:, feat_cols].to(torch.float32)          # [P,F], xyz first 3
                if pf.shape[0] == 0:
                    continue
                voxels, coords, num_p = self._voxelizer(pf)                     # [V,maxpts,F], [V,3]=(z,y,x), [V]
                vmean = voxels.sum(dim=1) / num_p.clamp_min(1).view(-1, 1).to(voxels.dtype)
                vfeats.append(self.vfe(vmean))                                  # [V, vfe_ch]
                bcol = torch.full((coords.shape[0], 1), b, dtype=coords.dtype, device=dev)
                coords_b.append(torch.cat([bcol, coords], dim=1))               # [V,4]=(b,z,y,x)
            if not vfeats:
                return points.new_zeros((B, self.out_channels, self.ny, self.nx))
            x = sp.SparseConvTensor(torch.cat(vfeats, 0).to(torch.float32),
                                    torch.cat(coords_b, 0).to(torch.int32),
                                    spatial_shape=[self.nz, self.ny, self.nx], batch_size=B)
            x = self.backbone(x)
            dense = x.dense()                                                   # [B, ch, z_final, ny, nx]
            dense = dense.reshape(B, self.ch_final * self.z_final, self.ny, self.nx)
        return self.to_bev(dense)                                              # outer autocast casts to model dtype
