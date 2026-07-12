"""Exact synthetic B=4 GH200 smoke; execute only under an approved S04 request."""
from __future__ import annotations

import json

import pytest
import torch


def _encoder_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("GH200 CUDA is required")
    try:
        import spconv.pytorch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"spconv unavailable: {type(exc).__name__}: {exc}")
    from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder

    return SparseVoxelEncoder


def _reference_points(device: torch.device, batch_size: int = 4, per_sample: int = 4096):
    rows = []
    i = torch.arange(per_sample, device=device, dtype=torch.int64)
    # Unique, in-range voxel coordinates with deterministic centers.
    xidx = i % 1440
    yidx = torch.div(i * 37, 1440, rounding_mode="floor") % 1440
    zidx = torch.div(i * 101, 1440, rounding_mode="floor") % 40
    x = -54.0 + (xidx.to(torch.float32) + 0.5) * 0.075
    y = -54.0 + (yidx.to(torch.float32) + 0.5) * 0.075
    z = -5.0 + (zidx.to(torch.float32) + 0.5) * 0.2
    intensity = (i % 255).to(torch.float32) / 255.0
    ring = (i % 32).to(torch.float32)
    dt = -(i % 10).to(torch.float32) * 0.05
    for batch in range(batch_size):
        b = torch.full_like(x, float(batch))
        rows.append(torch.stack((b, x, y, z, intensity, ring, dt), dim=1))
    return torch.cat(rows, dim=0)


def test_s04_reference_b4_fp16_forward_backward_memory_bound():
    SparseVoxelEncoder = _encoder_or_skip()
    dev = torch.device("cuda:0")
    encoder = SparseVoxelEncoder(
        out_channels=256,
        use_timestamp=True,
        max_voxels_train=120000,
        max_voxels_eval=160000,
        max_points_per_voxel=10,
        sparse_conv_fp16=True,
    ).to(dev).train()
    encoder.record_debug = True
    points = _reference_points(dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    encoder.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = encoder(points, B=4)
    loss = output.float().square().mean()
    loss.backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    meta = encoder.last_sparse_meta or {}
    assert output.shape == (4, 256, 180, 180)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert meta["dense_shape"] == (4, 128, 2, 180, 180)
    assert meta["stage_shapes_zyx"][-1] == (2, 180, 180)
    assert encoder.last_voxel_stats is not None
    assert torch.equal(encoder.last_voxel_stats[:, 4], torch.zeros(4, device=dev, dtype=torch.int64))
    assert meta["projected_dtype_before_contract_cast"] == "torch.float32"
    assert meta["bev_output_dtype"] == "torch.float16"
    assert meta["bev_output_contract"] == "float16"

    peak_allocated = torch.cuda.max_memory_allocated(dev)
    peak_reserved = torch.cuda.max_memory_reserved(dev)
    device_total = torch.cuda.get_device_properties(dev).total_memory
    assert 0 < peak_allocated <= peak_reserved <= device_total

    evidence = {
        "batch_size": 4,
        "points_per_sample": 4096,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "dense_shape": list(meta["dense_shape"]),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device_total_bytes": device_total,
        "loss": float(loss.detach().cpu()),
        "optimizer_or_parameter_update": False,
    }
    print("S04_B4_EVIDENCE=" + json.dumps(evidence, sort_keys=True))


def test_s04_reference_b4_fp16_eval_memory_bound():
    SparseVoxelEncoder = _encoder_or_skip()
    dev = torch.device("cuda:0")
    encoder = SparseVoxelEncoder(
        out_channels=256,
        use_timestamp=True,
        max_voxels_train=120000,
        max_voxels_eval=160000,
        max_points_per_voxel=10,
        sparse_conv_fp16=True,
    ).to(dev).eval()
    encoder.record_debug = True
    points = _reference_points(dev)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    with torch.no_grad():
        output = encoder(points, B=4)
    meta = encoder.last_sparse_meta or {}
    assert output.shape == (4, 256, 180, 180)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    assert meta["dense_shape"] == (4, 128, 2, 180, 180)
    assert meta["fp16_eval_dispatch_active"] is True
    assert meta["fp16_eval_dispatch_version"] == "2.3.8"
    assert meta["fp16_eval_dispatch_count"] > 0
    assert not encoder.training
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in encoder.parameters())

    peak_allocated = torch.cuda.max_memory_allocated(dev)
    peak_reserved = torch.cuda.max_memory_reserved(dev)
    device_total = torch.cuda.get_device_properties(dev).total_memory
    assert 0 < peak_allocated <= peak_reserved <= device_total
    evidence = {
        "batch_size": 4,
        "points_per_sample": 4096,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "dense_shape": list(meta["dense_shape"]),
        "fp16_eval_dispatch_version": meta["fp16_eval_dispatch_version"],
        "fp16_eval_dispatch_count": meta["fp16_eval_dispatch_count"],
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device_total_bytes": device_total,
        "all_parameter_grads_absent": True,
        "all_master_parameters_fp32": True,
        "optimizer_or_parameter_update": False,
    }
    print("S04_B4_EVAL_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
