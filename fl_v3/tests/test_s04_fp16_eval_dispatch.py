"""O-025 option-A lifecycle checks for spconv-2.3.8 fp16 evaluation."""
from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion import sparse_voxel_encoder as sparse_encoder_module


def _encoder_or_skip(*, fp16: bool = True):
    if not torch.cuda.is_available():
        pytest.skip("GH200 CUDA is required")
    try:
        import spconv.pytorch  # noqa: F401
    except Exception as exc:
        pytest.skip(f"spconv unavailable: {type(exc).__name__}: {exc}")
    cfg = BEVConfig(
        point_cloud_range=(-4.0, -4.0, -5.0, 4.0, 4.0, 3.0),
        bev_voxel=(0.5, 0.5),
        out_size_factor=8,
    )
    return sparse_encoder_module.SparseVoxelEncoder(
        out_channels=16,
        cfg=cfg,
        z_voxel=0.2,
        max_voxels_train=128,
        max_voxels_eval=192,
        max_points_per_voxel=3,
        sparse_conv_fp16=fp16,
    ).cuda()


def _six_voxel_points() -> torch.Tensor:
    return torch.tensor(
        [
            [0, -3.75, -3.75, -4.9, 0.10, 0],
            [0, -3.70, -3.70, -4.8, 0.20, 0],
            [0, -1.20, 0.20, 0.10, 0.30, 0],
            [0, 2.20, 2.20, 2.10, 0.40, 0],
            [1, 1.20, -1.10, -0.60, 0.50, 0],
            [1, 2.70, 2.90, 1.20, 0.60, 0],
            [1, -2.20, 1.80, -2.20, 0.70, 0],
            [1, 9.00, 0.00, 0.00, 0.80, 0],
        ],
        device="cuda",
        dtype=torch.float32,
    )


def _large_points(per_sample: int = 128) -> torch.Tensor:
    i = torch.arange(per_sample, device="cuda", dtype=torch.int64)
    xidx = i % 16
    yidx = torch.div(i, 16, rounding_mode="floor") % 16
    zidx = (i * 7) % 40
    x = -4.0 + (xidx.float() + 0.5) * 0.5
    y = -4.0 + (yidx.float() + 0.5) * 0.5
    z = -5.0 + (zidx.float() + 0.5) * 0.2
    intensity = (i % 31).float() / 31.0
    ring = (i % 16).float()
    rows = []
    for batch in range(2):
        rows.append(
            torch.stack((torch.full_like(x, batch), x, y, z, intensity, ring), dim=1)
        )
    return torch.cat(rows, dim=0)


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def test_spconv_fp16_eval_version_guard_fails_closed(monkeypatch):
    monkeypatch.setattr(sparse_encoder_module.metadata, "version", lambda _: "2.3.9")
    with pytest.raises(RuntimeError, match=r"audited only for spconv==2\.3\.8; found 2\.3\.9"):
        sparse_encoder_module._require_supported_spconv_fp16_eval()


def test_empty_fp16_eval_cannot_bypass_version_guard(monkeypatch):
    encoder = _encoder_or_skip().eval()
    empty = _six_voxel_points()[:0]
    monkeypatch.setattr(sparse_encoder_module.metadata, "version", lambda _: "2.3.9")
    with torch.no_grad(), pytest.raises(RuntimeError, match=r"spconv==2\.3\.8"):
        encoder(empty, B=2)


def test_spconv_dispatch_context_restores_flags_after_exception():
    encoder = _encoder_or_skip().eval()
    from spconv.pytorch.conv import SparseConvolution

    convs = [m for m in encoder.backbone.modules() if isinstance(m, SparseConvolution)]
    assert convs and all(not m.training for m in convs)
    with pytest.raises(ZeroDivisionError):
        with torch.no_grad():
            with sparse_encoder_module._spconv_training_dispatch_for_fp16_eval(
                encoder.backbone, sparse_encoder_module.SPCONV_FP16_EVAL_VERSION
            ):
                assert all(m.training for m in convs)
                1 / 0
    assert all(not m.training for m in convs)
    assert not encoder.training


def test_fp16_eval_dispatch_lifecycle_state_modes_and_parity():
    torch.manual_seed(20260712)
    torch.cuda.manual_seed_all(20260712)
    points = _six_voxel_points()
    encoder = _encoder_or_skip().eval()
    encoder.record_debug = True
    reference_state = copy.deepcopy(encoder.state_dict())
    initial_hash = _state_sha256(encoder)
    assert all(parameter.dtype == torch.float32 for parameter in encoder.parameters())
    with pytest.raises(RuntimeError, match=r"requires torch\.no_grad"):
        encoder(points, B=2)

    from spconv.pytorch.conv import SparseConvolution

    conv_modes: list[bool] = []
    norm_modes: list[bool] = []
    handles = []
    for module in encoder.backbone.modules():
        if isinstance(module, SparseConvolution):
            handles.append(module.register_forward_pre_hook(lambda m, _i: conv_modes.append(m.training)))
        if isinstance(module, torch.nn.GroupNorm):
            handles.append(module.register_forward_pre_hook(lambda m, _i: norm_modes.append(m.training)))
    try:
        with torch.no_grad():
            fresh_before = encoder(points, B=2)
    finally:
        for handle in handles:
            handle.remove()

    meta = encoder.last_sparse_meta or {}
    assert fresh_before.dtype == torch.float16 and torch.isfinite(fresh_before).all()
    assert not encoder.training and not encoder.backbone.training
    assert conv_modes and all(conv_modes)
    assert norm_modes and all(not mode for mode in norm_modes)
    assert all(not module.training for module in encoder.backbone.modules())
    assert meta["fp16_eval_dispatch_active"] is True
    assert meta["fp16_eval_dispatch_version"] == "2.3.8"
    assert meta["fp16_eval_dispatch_count"] == len(conv_modes)
    assert _state_sha256(encoder) == initial_hash
    assert all(parameter.grad is None for parameter in encoder.parameters())

    with torch.no_grad():
        large = encoder(_large_points(), B=2)
        empty = encoder(points[:0], B=2)
    empty_meta = encoder.last_sparse_meta or {}
    assert large.dtype == empty.dtype == torch.float16
    assert torch.isfinite(large).all() and torch.count_nonzero(empty) == 0
    assert empty_meta["fp16_eval_dispatch_active"] is True
    assert empty_meta["fp16_eval_dispatch_version"] == "2.3.8"
    assert empty_meta["fp16_eval_dispatch_count"] == 0
    assert _state_sha256(encoder) == initial_hash

    # The exact same model is exercised before and after a real train/backward
    # phase.  No optimizer step occurs; clear gradients before the second eval so
    # inference itself must leave every parameter grad absent.
    encoder.train()
    train_out = encoder(points, B=2)
    train_out.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert _state_sha256(encoder) == initial_hash
    encoder.zero_grad(set_to_none=True)
    encoder.eval()
    with torch.no_grad():
        fresh_after = encoder(points, B=2)
    assert fresh_after.dtype == torch.float16 and torch.isfinite(fresh_after).all()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert _state_sha256(encoder) == initial_hash

    # Automatic option-A eval must agree with the package's ordinary training
    # dispatch under no_grad on an identical model and below both voxel caps.
    train_dispatch = _encoder_or_skip().train()
    train_dispatch.load_state_dict(reference_state)
    with torch.no_grad():
        expected = train_dispatch(points, B=2)
    torch.testing.assert_close(fresh_before, expected, rtol=1e-3, atol=1e-3)

    fp32 = _encoder_or_skip(fp16=False).eval()
    fp32.load_state_dict(reference_state)
    with torch.no_grad():
        fp32_output = fp32(points, B=2)
    assert fp32_output.dtype == torch.float32 and torch.isfinite(fp32_output).all()
