"""Sparse 3D-voxel LiDAR encoder engineering checks.

These tests are Arrhenius/GH200-facing: they skip when CUDA or the source-built
spconv stack is unavailable, but exercise the real PointToVoxel/SparseConvTensor
path when the validated runtime is active.
"""
from __future__ import annotations

import pytest
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.utils.runtime import validate_sparse_precision


def test_sparse_precision_policy_rejects_bf16_voxel():
    validate_sparse_precision("fp32", "voxel")
    validate_sparse_precision("fp16", "voxel")
    with pytest.raises(ValueError, match="bf16.*voxel"):
        validate_sparse_precision("bf16", "voxel")


def _sparse_encoder_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for SparseVoxelEncoder/spconv checks")
    try:
        import spconv.pytorch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"spconv unavailable in this runtime: {type(exc).__name__}: {exc}")
    from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder

    return SparseVoxelEncoder


def _cfg() -> BEVConfig:
    return BEVConfig(
        point_cloud_range=(-4.0, -4.0, -2.0, 4.0, 4.0, 2.0),
        bev_voxel=(1.0, 1.0),
        out_size_factor=2,
    )


def _points(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [0, -3.5, -3.5, -1.5, 0.10, 0],
            [0, -3.4, -3.4, -1.4, 0.20, 0],
            [0, -0.2,  0.2,  0.1, 0.30, 0],
            [1,  1.2, -1.1, -0.6, 0.40, 0],
            [1,  2.7,  2.9,  1.2, 0.50, 0],
            [1,  9.0,  0.0,  0.0, 0.60, 0],  # out of range
        ],
        device=device,
        dtype=torch.float32,
    )


def test_sparse_voxel_debug_meta_shape_and_backward():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    dev = torch.device("cuda:0")
    cfg = _cfg()
    enc = SparseVoxelEncoder(out_channels=8, cfg=cfg, max_voxels=128, max_points_per_voxel=8).to(dev)
    enc.record_debug = True
    enc.train()

    out = enc(_points(dev), B=2)
    meta = enc.last_sparse_meta or {}
    assert out.shape == (2, 8, cfg.ny, cfg.nx)
    assert torch.isfinite(out).all()
    assert meta["coord_order"] == "bzyx"
    assert meta["indices_dtype"] == "torch.int32"
    assert meta["features_dtype"] == "torch.float32"
    assert meta["indices_shape"][1] == 4
    assert meta["spatial_shape"] == (enc.nz, cfg.ny, cfg.nx)
    assert meta["batch_size"] == 2
    assert meta["batch_index_min"] == 0
    assert meta["batch_index_max"] == 1
    assert 0 <= meta["z_min"] <= meta["z_max"] < enc.nz
    assert 0 <= meta["y_min"] <= meta["y_max"] < cfg.ny
    assert 0 <= meta["x_min"] <= meta["x_max"] < cfg.nx

    loss = out.float().square().mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_sparse_voxel_occupancy_and_empty_input():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    dev = torch.device("cuda:0")
    cfg = _cfg()
    enc = SparseVoxelEncoder(out_channels=8, cfg=cfg, max_voxels=128, max_points_per_voxel=8).to(dev)
    pts = _points(dev)

    occ = enc.occupancy(pts, B=2)
    assert occ.shape == (2, cfg.ny, cfg.nx)
    assert float(occ.sum().cpu()) == 5.0

    enc.record_debug = True
    enc.train()
    empty = enc(pts[:0], B=2)
    assert empty.shape == (2, 8, cfg.ny, cfg.nx)
    assert torch.count_nonzero(empty.detach()) == 0
    assert (enc.last_sparse_meta or {})["num_voxels"] == 0
    empty.sum().backward()
    assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)
