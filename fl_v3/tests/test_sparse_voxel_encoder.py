"""GH200-facing sparse SECOND runtime checks (no full data or model metrics)."""
from __future__ import annotations

import copy

import pytest
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.sparse_voxel_encoder import VOXEL_STAT_FIELDS
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
    # 16x16 XY and 40 physical z bins; SparseVoxelEncoder adds the reference
    # padding bin, so the sparse input is 41x16x16 and output is 2x2x2.
    return BEVConfig(
        point_cloud_range=(-4.0, -4.0, -5.0, 4.0, 4.0, 3.0),
        bev_voxel=(0.5, 0.5),
        out_size_factor=8,
    )


def _points(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [0, -3.75, -3.75, -4.9, 0.10, 0],
            [0, -3.70, -3.70, -4.8, 0.20, 0],
            [0, -1.20, 0.20, 0.10, 0.30, 0],
            [0, 2.20, 2.20, 2.10, 0.40, 0],
            [1, 1.20, -1.10, -0.60, 0.50, 0],
            [1, 2.70, 2.90, 1.20, 0.60, 0],
            [1, -2.20, 1.80, -2.20, 0.70, 0],
            [1, 9.00, 0.00, 0.00, 0.80, 0],  # out of range
        ],
        device=device,
        dtype=torch.float32,
    )


def _encoder(
    SparseVoxelEncoder, *, fp16: bool = False, train_cap: int = 128,
    eval_cap: int = 192, normalization: str = "group_norm",
):
    return SparseVoxelEncoder(
        out_channels=16,
        cfg=_cfg(),
        z_voxel=0.2,
        max_voxels_train=train_cap,
        max_voxels_eval=eval_cap,
        max_points_per_voxel=3,
        sparse_conv_fp16=fp16,
        second_normalization=normalization,
    ).cuda()


def test_second_normalization_checkpoint_state_is_fail_closed():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    torch.manual_seed(0)
    group_norm = _encoder(SparseVoxelEncoder, normalization="group_norm")
    torch.manual_seed(0)
    batch_norm = _encoder(SparseVoxelEncoder, normalization="batch_norm_1d")

    assert group_norm.second_normalization == "group_norm"
    assert batch_norm.second_normalization == "batch_norm_1d"
    group_params = dict(group_norm.named_parameters())
    batch_params = dict(batch_norm.named_parameters())
    assert group_params.keys() == batch_params.keys()
    assert all(torch.equal(group_params[name], batch_params[name]) for name in group_params)

    group_keys = set(group_norm.state_dict())
    batch_keys = set(batch_norm.state_dict())
    running = {
        key for key in batch_keys - group_keys
        if key.rsplit(".", 1)[-1] in {"running_mean", "running_var", "num_batches_tracked"}
    }
    assert len(running) == 63
    assert batch_keys - group_keys == running
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        group_norm.load_state_dict(batch_norm.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        batch_norm.load_state_dict(group_norm.state_dict(), strict=True)


def test_sparse_second_shape_stats_backward_and_reduced_occupancy():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    enc = _encoder(SparseVoxelEncoder).train()
    import spconv.pytorch as spconv

    # Regression for Job 335566: a custom residual inside SparseSequential is
    # called as module(input.features), so it receives Tensor rather than
    # SparseConvTensor. Residual stages must be explicitly forwarded ModuleLists.
    for stage_name in ("stage1", "stage2", "stage3", "stage4"):
        stage = getattr(enc.backbone, stage_name)
        assert isinstance(stage, torch.nn.ModuleList)
        assert all(not isinstance(block, spconv.SparseSequential) for block in stage)
    for module in enc.backbone.modules():
        if isinstance(module, spconv.SparseSequential):
            assert not any(type(child).__name__ == "_SparseResidualBlock" for child in module.children())

    enc.record_debug = True
    pts = _points(torch.device("cuda:0"))

    out = enc(pts, B=2)
    meta = enc.last_sparse_meta or {}
    assert out.shape == (2, 16, 2, 2)
    assert torch.isfinite(out).all()
    assert meta["coord_order"] == "bzyx"
    assert meta["spatial_shape_zyx"] == (41, 16, 16)
    assert meta["stage_shapes_zyx"] == (
        (41, 16, 16),
        (21, 8, 8),
        (11, 4, 4),
        (5, 2, 2),
        (5, 2, 2),
        (2, 2, 2),
    )
    assert meta["dense_shape"] == (2, 128, 2, 2, 2)
    assert meta["output_stride_xy"] == 8
    assert meta["receptive_field_voxels_zyx"] == (153, 137, 137)
    assert tuple(enc.last_voxel_stats.shape) == (2, len(VOXEL_STAT_FIELDS))
    assert torch.equal(enc.last_voxel_stats[:, 4], torch.zeros(2, device="cuda", dtype=torch.int64))

    occ = enc.occupancy(pts, B=2)
    assert occ.shape == (2, 2, 2)
    assert float(occ.sum().cpu()) == 7.0

    out.float().square().mean().backward()
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_per_sample_caps_extreme_occupancy_and_point_permutation():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    enc = _encoder(SparseVoxelEncoder, train_cap=2, eval_cap=3).train()
    dev = torch.device("cuda:0")
    rows = []
    for batch in range(2):
        for i in range(4):
            # Four distinct voxels/sample; first voxel has five points so both
            # voxel and max-points truncation are observable.
            repeats = 5 if i == 0 else 1
            for repeat in range(repeats):
                rows.append([batch, -3.75 + i, -3.75 + i, -4.9, 0.1 + repeat * 0.01, 0])
    points = torch.tensor(rows, dtype=torch.float32, device=dev)
    out = enc(points, B=2)
    stats = enc.last_voxel_stats
    assert stats is not None
    assert torch.equal(stats[:, 2], torch.tensor([4, 4], device=dev))
    assert torch.equal(stats[:, 3], torch.tensor([2, 2], device=dev))
    assert torch.equal(stats[:, 4], torch.tensor([2, 2], device=dev))
    assert torch.equal(stats[:, 5], torch.tensor([2, 2], device=dev))

    perm = torch.randperm(points.shape[0], device=dev)
    out_perm = enc(points[perm], B=2)
    # Canonical voxel inputs are identical. spconv itself is not promised to be
    # byte-deterministic, so the runtime assertion uses a numerical tolerance.
    torch.testing.assert_close(out, out_perm, rtol=1e-5, atol=1e-6)

    enc.eval()
    enc(points, B=2)
    assert torch.equal(enc.last_voxel_stats[:, 3], torch.tensor([3, 3], device=dev))
    assert torch.equal(enc.last_voxel_stats[:, 4], torch.tensor([1, 1], device=dev))


def test_empty_sample_batch_isolation_and_batch_permutation():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    enc = _encoder(SparseVoxelEncoder).eval()
    dev = torch.device("cuda:0")
    sample0 = _points(dev)[:4].clone()

    alone = enc(sample0, B=1)
    empty_plus = torch.cat((sample0, _points(dev)[4:7]), dim=0)
    batched = enc(empty_plus, B=2)
    torch.testing.assert_close(alone[0], batched[0], rtol=1e-5, atol=1e-6)

    sample1 = _points(dev)[4:7].clone()
    sample0_swapped = sample0.clone()
    sample0_swapped[:, 0] = 1
    sample1_swapped = sample1.clone()
    sample1_swapped[:, 0] = 0
    swapped = enc(torch.cat((sample1_swapped, sample0_swapped)), B=2)
    torch.testing.assert_close(batched[0], swapped[1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched[1], swapped[0], rtol=1e-5, atol=1e-6)

    empty = enc(sample0[:0], B=2)
    assert empty.shape == (2, 16, 2, 2)
    assert torch.count_nonzero(empty) == 0
    assert torch.equal(enc.last_voxel_stats, torch.zeros_like(enc.last_voxel_stats))


def test_fp32_and_fp16_sparse_paths_have_finite_outputs_and_gradients():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    dev = torch.device("cuda:0")
    fp32 = _encoder(SparseVoxelEncoder, fp16=False).train()
    fp16 = _encoder(SparseVoxelEncoder, fp16=True).train()
    fp16.load_state_dict(copy.deepcopy(fp32.state_dict()))
    pts = _points(dev)

    out32 = fp32(pts, B=2)
    out16 = fp16(pts, B=2)
    assert out32.dtype == torch.float32
    assert out16.dtype == torch.float16
    assert torch.isfinite(out32).all() and torch.isfinite(out16).all()
    torch.testing.assert_close(out16.float(), out32, rtol=0.2, atol=0.2)

    out32.square().mean().backward()
    out16.float().square().mean().backward()
    for model in (fp32, fp16):
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    fp16.eval()
    fp16.record_debug = True
    fp16.zero_grad(set_to_none=True)
    with torch.no_grad():
        nonempty = fp16(pts, B=2)
        nonempty_meta = fp16.last_sparse_meta or {}
        empty = fp16(pts[:0], B=2)
    assert nonempty.dtype == empty.dtype == torch.float16
    assert all(parameter.grad is None for parameter in fp16.parameters())
    assert nonempty_meta["fp16_eval_dispatch_active"] is True
    assert nonempty_meta["fp16_eval_dispatch_version"] == "2.3.8"
    assert nonempty_meta["projected_dtype_before_contract_cast"] == "torch.float32"
    assert nonempty_meta["bev_output_dtype"] == "torch.float16"
    assert nonempty_meta["bev_output_contract"] == "float16"


def test_second_fp32_island_overrides_outer_autocast_and_exposes_named_boundaries():
    SparseVoxelEncoder = _sparse_encoder_or_skip()
    dev = torch.device("cuda:0")
    encoder = _encoder(SparseVoxelEncoder, fp16=False).train()
    encoder.record_debug = True
    points = _points(dev)
    boundaries = {}

    def capture(name, tensor):
        assert name not in boundaries
        tensor.retain_grad()
        boundaries[name] = tensor

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = encoder(points, B=2, boundary_capture=capture)
    meta = encoder.last_sparse_meta or {}
    assert output.dtype == torch.float32
    assert meta["sparse_conv_fp16_requested"] is False
    assert meta["sparse_conv_fp16_active"] is False
    assert meta["dense_dtype"] == "torch.float32"
    assert meta["projected_dtype_before_contract_cast"] == "torch.float32"
    assert meta["bev_output_dtype"] == "torch.float32"
    assert set(boundaries) == {"second.stem", "second.stage1", "second.output"}

    (output.square().mean() * 8.0).backward()
    assert all(tensor.grad is not None for tensor in boundaries.values())
    assert all(torch.isfinite(tensor.grad).all() for tensor in boundaries.values())
