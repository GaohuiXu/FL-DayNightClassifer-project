"""Per-sample hard voxelization plus a low-resolution SECOND sparse encoder.

The only dense conversion occurs after three sparse XY downsampling stages and a
z-only collapse convolution.  Under the frozen 0.075 m contract the conversion is
``[B,128,2,180,180]``; no ``1440x1440`` dense or fusion tensor is constructed.
"""
from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn

from fl_v3.models.fusion.bev_grid import flat_index, in_grid_mask, metric_to_grid
from fl_v3.models.fusion.second_sparse_backbone import (
    SECONDShapeContract,
    SECONDSparseBackbone,
    _gn,
)


VOXEL_STAT_FIELDS = (
    "input_points",
    "valid_in_range_points",
    "unique_voxels_before_cap",
    "voxels_kept",
    "voxels_dropped",
    "points_dropped_by_voxel_point_cap",
)


class SparseVoxelEncoder(nn.Module):
    """Points to ``[B,C,H/8,W/8]`` via per-sample voxelization and SECOND."""

    def __init__(
        self,
        out_channels: int = 256,
        cfg=None,
        use_timestamp: bool = False,
        z_voxel: float | None = 0.2,
        sparse_z_size: int | None = None,
        max_voxels: int | None = None,
        max_voxels_train: int = 120000,
        max_voxels_eval: int = 160000,
        max_points_per_voxel: int = 10,
        sparse_conv_fp16: bool = False,
    ):
        super().__init__()
        if cfg is None:
            from fl_v3.models.fusion.bev_grid import BEVConfig

            cfg = BEVConfig(
                point_cloud_range=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
                bev_voxel=(0.075, 0.075),
                out_size_factor=8,
            )
        self.cfg = cfg
        self.out_channels = int(out_channels)
        self.use_timestamp = bool(use_timestamp)
        self.n_pt_feat = 5 if use_timestamp else 4
        self.vx, self.vy = float(cfg.vx), float(cfg.vy)
        self.vz = 0.2 if z_voxel is None else float(z_voxel)
        if self.vz <= 0:
            raise ValueError(f"z_voxel must be positive, got {self.vz}")
        self.nx, self.ny = int(cfg.nx), int(cfg.ny)
        self.computed_nz = int(round((cfg.z_max - cfg.z_min) / self.vz))
        # Official BEVFusion adds one sparse z padding bin: 40 physical bins -> shape 41.
        self.nz = self.computed_nz + 1 if sparse_z_size is None else int(sparse_z_size)
        if self.nz < self.computed_nz:
            raise ValueError(
                f"sparse_z_size={self.nz} is smaller than physical z bins {self.computed_nz}"
            )
        if max_voxels is not None:
            if max_voxels_train != 120000 or max_voxels_eval != 160000:
                raise ValueError("max_voxels alias cannot be combined with separate train/eval caps")
            max_voxels_train = max_voxels_eval = int(max_voxels)
        self.max_voxels_train = int(max_voxels_train)
        self.max_voxels_eval = int(max_voxels_eval)
        self.max_pts = int(max_points_per_voxel)
        if min(self.max_voxels_train, self.max_voxels_eval, self.max_pts) <= 0:
            raise ValueError("voxel and point caps must be positive")
        self.sparse_conv_fp16 = bool(sparse_conv_fp16)

        self.contract = SECONDShapeContract(
            input_zyx=(self.nz, self.ny, self.nx),
            voxel_xyz=(self.vx, self.vy, self.vz),
            range_xyzxyz=tuple(float(v) for v in cfg.point_cloud_range),
        )
        self.backbone = SECONDSparseBackbone(self.n_pt_feat, self.contract)
        collapsed = self.contract.collapsed_channels
        self.to_bev = (
            nn.Identity()
            if collapsed == self.out_channels
            else nn.Sequential(
                nn.Conv2d(collapsed, self.out_channels, 1, bias=False),
                _gn(self.out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self._voxelizers: dict[tuple[str, int | None, int], object] = {}
        self.record_debug = False
        self.last_sparse_meta: dict | None = None
        self.last_voxel_stats: torch.Tensor | None = None
        self.record_profile = False
        self.last_profile_times: dict | None = None

    @property
    def output_stride(self) -> int:
        return self.contract.output_stride_xy

    @property
    def output_hw(self) -> tuple[int, int]:
        return self.contract.bev_hw

    @property
    def active_max_voxels(self) -> int:
        return self.max_voxels_train if self.training else self.max_voxels_eval

    def _voxelizer_for(self, device: torch.device, cap: int):
        from spconv.pytorch.utils import PointToVoxel

        key = (device.type, device.index, int(cap))
        voxelizer = self._voxelizers.get(key)
        if voxelizer is None:
            c = self.cfg
            voxelizer = PointToVoxel(
                vsize_xyz=[self.vx, self.vy, self.vz],
                coors_range_xyz=[c.x_min, c.y_min, c.z_min, c.x_max, c.y_max, c.z_max],
                num_point_features=self.n_pt_feat,
                max_num_voxels=int(cap),
                max_num_points_per_voxel=self.max_pts,
                device=device,
            )
            self._voxelizers[key] = voxelizer
        return voxelizer

    def _use_sparse_conv_fp16(self, device: torch.device) -> bool:
        return bool(self.sparse_conv_fp16 and device.type == "cuda")

    def _group_points_by_batch(self, points: torch.Tensor, bidx: torch.Tensor, B: int):
        if B <= 0 or points.shape[0] == 0:
            return points, [0] * max(0, int(B)), "empty"
        grouped_points = points
        grouped_bidx = bidx
        mode = "contiguous"
        if bidx.numel() > 1 and not bool(torch.all(bidx[1:] >= bidx[:-1]).detach().cpu()):
            order = torch.argsort(bidx, stable=True)
            grouped_points = points.index_select(0, order)
            grouped_bidx = bidx.index_select(0, order)
            mode = "sorted"
        counts = torch.bincount(grouped_bidx, minlength=int(B))
        if counts.numel() != int(B):
            raise ValueError(f"batch indices must lie in [0,{B}), got bincount length {counts.numel()}")
        return grouped_points, [int(v) for v in counts.detach().cpu().tolist()], mode

    def _canonical_sample(self, pf: torch.Tensor):
        """Filter and canonically order one sample; return exact occupancy statistics."""
        c = self.cfg
        finite = torch.isfinite(pf).all(dim=1)
        valid = (
            finite
            & (pf[:, 0] >= c.x_min)
            & (pf[:, 0] < c.x_max)
            & (pf[:, 1] >= c.y_min)
            & (pf[:, 1] < c.y_max)
            & (pf[:, 2] >= c.z_min)
            & (pf[:, 2] < c.z_max)
        )
        pf = pf[valid]
        if pf.shape[0] == 0:
            zero = torch.zeros((), dtype=torch.int64, device=pf.device)
            return pf, zero, zero
        col = torch.floor((pf[:, 0] - c.x_min) / self.vx).to(torch.int64)
        row = torch.floor((pf[:, 1] - c.y_min) / self.vy).to(torch.int64)
        zidx = torch.floor((pf[:, 2] - c.z_min) / self.vz).to(torch.int64)
        voxel_key = (zidx * self.ny + row) * self.nx + col

        # Stable least-to-most-significant sorting makes both voxel selection and
        # max-points truncation independent of incoming point order.
        order = torch.arange(pf.shape[0], device=pf.device)
        for sort_key in [pf[:, i] for i in reversed(range(pf.shape[1]))] + [voxel_key]:
            order = order[torch.argsort(sort_key[order], stable=True)]
        pf = pf.index_select(0, order)
        key_sorted = voxel_key.index_select(0, order)
        _, counts = torch.unique_consecutive(key_sorted, return_counts=True)
        unique_voxels = torch.tensor(counts.numel(), dtype=torch.int64, device=pf.device)
        point_drops = torch.clamp(counts - self.max_pts, min=0).sum().to(torch.int64)
        return pf, unique_voxels, point_drops

    def forward(self, points: torch.Tensor, B: int) -> torch.Tensor:
        """Encode batched points; output is ``[B,out,H/8,W/8]``."""
        if B < 0:
            raise ValueError(f"B must be non-negative, got {B}")
        if points.ndim != 2 or points.shape[1] < (7 if self.use_timestamp else 5):
            raise ValueError(f"unexpected point tensor shape {tuple(points.shape)}")
        if B == 0 and points.shape[0] != 0:
            raise ValueError("non-empty points require B > 0")
        dev = points.device
        profile_times: dict[str, float] = {}

        @contextmanager
        def stage(name: str):
            if self.record_profile and dev.type == "cuda":
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                yield
                torch.cuda.synchronize(dev)
                profile_times[name] = (time.perf_counter() - t0) * 1000.0
            else:
                yield

        import spconv.pytorch as spconv

        feat_cols = [1, 2, 3, 4] + ([6] if self.use_timestamp else [])
        bidx = points[:, 0].to(torch.int64)
        if bidx.numel():
            integral = points[:, 0] == bidx.to(points.dtype)
            in_batch = (bidx >= 0) & (bidx < B)
            if not bool((integral & in_batch).all().detach().cpu()):
                raise ValueError(f"batch indices must be integral and lie in [0,{B})")
        sparse_amp = self._use_sparse_conv_fp16(dev)
        cap = self.active_max_voxels
        with stage("batch_point_grouping"):
            grouped_points, batch_counts, point_grouping = self._group_points_by_batch(
                points, bidx, B
            )

        vfeats: list[torch.Tensor] = []
        coords_b: list[torch.Tensor] = []
        stats: list[torch.Tensor] = []
        start = 0
        with torch.autocast(device_type=dev.type, enabled=False):
            with stage("voxelize_mean_vfe"):
                voxelizer = self._voxelizer_for(dev, cap)
                for b, n_points in enumerate(batch_counts):
                    stop = start + n_points
                    pf_raw = grouped_points[start:stop, feat_cols].to(torch.float32)
                    start = stop
                    pf, unique_before, point_drops = self._canonical_sample(pf_raw)
                    valid_count = torch.tensor(pf.shape[0], dtype=torch.int64, device=dev)
                    input_count = torch.tensor(n_points, dtype=torch.int64, device=dev)
                    if pf.shape[0] == 0:
                        zero = torch.zeros((), dtype=torch.int64, device=dev)
                        stats.append(torch.stack((input_count, zero, zero, zero, zero, zero)))
                        continue
                    voxels, coords, num_p = voxelizer(pf)
                    kept = torch.tensor(coords.shape[0], dtype=torch.int64, device=dev)
                    dropped = torch.clamp(unique_before - kept, min=0)
                    stats.append(
                        torch.stack(
                            (input_count, valid_count, unique_before, kept, dropped, point_drops)
                        )
                    )
                    if coords.shape[0] == 0:
                        continue
                    slots = torch.arange(voxels.shape[1], device=dev).view(1, -1)
                    valid_slots = slots < num_p.view(-1, 1)
                    mean = (voxels * valid_slots.unsqueeze(-1)).sum(dim=1)
                    mean = mean / num_p.clamp_min(1).to(torch.float32).unsqueeze(1)
                    vfeats.append(mean.to(torch.float32))
                    bcol = torch.full((coords.shape[0], 1), b, dtype=coords.dtype, device=dev)
                    coords_b.append(torch.cat((bcol, coords), dim=1).to(torch.int32))

        self.last_voxel_stats = (
            torch.stack(stats, dim=0)
            if stats
            else torch.zeros((B, len(VOXEL_STAT_FIELDS)), dtype=torch.int64, device=dev)
        )
        if not vfeats:
            self._record_meta(
                B=B,
                cap=cap,
                sparse_amp=sparse_amp,
                point_grouping=point_grouping,
                indices=None,
                dense_shape=None,
            )
            if self.record_profile:
                self.last_profile_times = dict(profile_times)
            return self._empty_bev(points, B, sparse_amp)

        with stage("sparse_tensor_inputs"):
            features_fp32 = torch.cat(vfeats, dim=0)
            features = features_fp32.to(torch.float16) if sparse_amp else features_fp32
            indices = torch.cat(coords_b, dim=0)

        sparse_ctx = (
            torch.autocast(device_type=dev.type, dtype=torch.float16)
            if sparse_amp
            else torch.autocast(device_type=dev.type, enabled=False)
        )
        with sparse_ctx:
            with stage("sparse_tensor_construct"):
                x = spconv.SparseConvTensor(
                    features,
                    indices,
                    spatial_shape=list(self.contract.input_zyx),
                    batch_size=B,
                )
            with stage("second_sparse_backbone"):
                x = self.backbone(x)
            with stage("reduced_dense_collapse"):
                dense = x.dense()
                dense_shape = tuple(int(v) for v in dense.shape)
                expected = (
                    B,
                    self.contract.sparse_output_channels,
                    *self.contract.sparse_output_zyx,
                )
                if dense_shape != expected:
                    raise RuntimeError(f"dense shape drift: got {dense_shape}, expected {expected}")
                dense = dense.reshape(B, self.contract.collapsed_channels, *self.output_hw)

        bev_ctx = (
            torch.autocast(device_type=dev.type, dtype=torch.float16)
            if sparse_amp
            else nullcontext()
        )
        with stage("low_resolution_projection"):
            with bev_ctx:
                out = self.to_bev(dense)
        self._record_meta(
            B=B,
            cap=cap,
            sparse_amp=sparse_amp,
            point_grouping=point_grouping,
            indices=indices,
            dense_shape=dense_shape,
        )
        if self.record_profile:
            self.last_profile_times = dict(profile_times)
        return out

    def _record_meta(
        self,
        *,
        B: int,
        cap: int,
        sparse_amp: bool,
        point_grouping: str,
        indices: torch.Tensor | None,
        dense_shape: tuple[int, ...] | None,
    ) -> None:
        if not self.record_debug:
            self.last_sparse_meta = None
            return
        self.last_sparse_meta = {
            "coord_order": "bzyx",
            "spatial_shape_zyx": self.contract.input_zyx,
            "stage_shapes_zyx": self.contract.stage_shapes_zyx,
            "dense_shape": dense_shape,
            "bev_shape": (B, self.out_channels, *self.output_hw),
            "voxel_size_xyz": self.contract.voxel_xyz,
            "output_stride_xy": self.output_stride,
            "output_cell_xy": self.contract.output_cell_xy,
            "receptive_field_voxels_zyx": self.contract.receptive_field_voxels_zyx,
            "batch_size": int(B),
            "active_max_voxels_per_sample": int(cap),
            "max_voxels_train": self.max_voxels_train,
            "max_voxels_eval": self.max_voxels_eval,
            "sparse_conv_fp16_requested": self.sparse_conv_fp16,
            "sparse_conv_fp16_active": sparse_amp,
            "point_grouping": point_grouping,
            "voxel_stat_fields": VOXEL_STAT_FIELDS,
            "num_voxels": 0 if indices is None else int(indices.shape[0]),
            "indices_shape": (0, 4) if indices is None else tuple(indices.shape),
            "indices_dtype": "torch.int32" if indices is None else str(indices.dtype),
        }

    def _empty_bev(self, points: torch.Tensor, B: int, sparse_amp: bool) -> torch.Tensor:
        dtype = torch.float16 if sparse_amp else torch.float32
        out = torch.zeros((B, self.out_channels, *self.output_hw), device=points.device, dtype=dtype)
        if not self.training:
            return out
        anchor = None
        for parameter in self.parameters():
            term = parameter.float().sum() * 0.0
            anchor = term if anchor is None else anchor + term
        return out if anchor is None else out + anchor.to(dtype=out.dtype)

    def occupancy(self, points: torch.Tensor, B: int) -> torch.Tensor:
        """Reduced-grid point occupancy ``[B,H/8,W/8]``; never allocates fine BEV."""
        b = points[:, 0].to(torch.int64)
        x, y, z = points[:, 1], points[:, 2], points[:, 3]
        sx, sy = self.contract.output_cell_xy
        col, row = metric_to_grid(x, y, self.cfg.x_min, self.cfg.y_min, sx, sy)
        out_h, out_w = self.output_hw
        keep = (
            in_grid_mask(col, row, out_w, out_h)
            & (z >= self.cfg.z_min)
            & (z < self.cfg.z_max)
            & torch.isfinite(points[:, 1:5]).all(dim=1)
        )
        occ = points.new_zeros((B * out_h * out_w,))
        if not bool(keep.any().detach().cpu()):
            return occ.view(B, out_h, out_w)
        key = b[keep] * (out_h * out_w) + flat_index(col[keep], row[keep], out_w)
        key_s = key[torch.argsort(key, stable=True)]
        uniq, counts = torch.unique_consecutive(key_s, return_counts=True)
        occ.index_copy_(0, uniq, counts.to(occ.dtype))
        return occ.view(B, out_h, out_w)
