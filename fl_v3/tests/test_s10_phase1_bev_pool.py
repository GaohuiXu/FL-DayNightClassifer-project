from __future__ import annotations

import os

import pytest
import torch

from fl_v3.models.ops.bev_pool import bev_pool
from fl_v3.models.ops.bev_pool.bev_pool import _ranks, _sorted_inputs


def _build_directory(tmp_path) -> str:
    """Keep Envelope-A compilation inside its approved /nobackup root."""
    return os.environ.get("FL_V3_BEV_POOL_BUILD_DIR", str(tmp_path))


def test_bev_pool_fallback_non_square_geometry_collision_and_exact_gradient():
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], requires_grad=True
    )
    # x/y deliberately exercise a non-square H=2, W=3 rank mapping.
    geometry = torch.tensor(
        [[2, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0]], dtype=torch.int32
    )
    output = bev_pool(values, geometry, 1, 1, 2, 3, backend="fallback")
    expected = torch.zeros((1, 2, 1, 2, 3))
    expected[0, :, 0, 0, 2] = torch.tensor([8.0, 13.0])
    expected[0, :, 0, 1, 0] = torch.tensor([3.0, 5.0])
    assert torch.equal(output, expected)
    output.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_bev_pool_fallback_resets_fp32_accumulation_at_each_cell():
    # A global prefix-cumsum/difference loses both ones after the preceding 1e8;
    # the pinned CUDA kernel starts every cell at zero and therefore returns 2.
    values = torch.tensor([[1.0e8], [1.0], [1.0]], requires_grad=True)
    geometry = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.int32
    )
    output = bev_pool(values, geometry, 1, 1, 1, 2, backend="fallback")
    assert output[0, 0, 0, 0, 0] == 1.0e8
    assert output[0, 0, 0, 0, 1] == 2.0
    output.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))


def test_optimized_composite_sort_is_exactly_the_stable_reference_order():
    values = torch.arange(18, dtype=torch.float32).view(9, 2)
    geometry = torch.tensor(
        [
            [2, 0, 0, 0],
            [0, 1, 0, 0],
            [2, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [2, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    ranks = _ranks(geometry, 1, 1, 2, 3)
    reference = _sorted_inputs(
        values, geometry, ranks, optimized=False, rank_cardinality=6
    )
    optimized = _sorted_inputs(
        values, geometry, ranks, optimized=True, rank_cardinality=6
    )
    assert all(torch.equal(left, right) for left, right in zip(reference, optimized))


def test_bev_pool_empty_singleton_and_input_contract():
    empty_values = torch.empty((0, 3), dtype=torch.float32, requires_grad=True)
    empty_geometry = torch.empty((0, 4), dtype=torch.int32)
    empty = bev_pool(
        empty_values, empty_geometry, 2, 1, 2, 3, backend="fallback"
    )
    assert tuple(empty.shape) == (2, 3, 1, 2, 3)
    assert torch.count_nonzero(empty) == 0
    empty.sum().backward()
    assert tuple(empty_values.grad.shape) == (0, 3)

    singleton = bev_pool(
        torch.tensor([[4.0]], dtype=torch.float32),
        torch.tensor([[1, 1, 0, 0]], dtype=torch.int32),
        1,
        1,
        2,
        2,
        backend="fallback",
    )
    assert singleton[0, 0, 0, 1, 1] == 4.0
    with pytest.raises(TypeError, match="FP32"):
        bev_pool(
            torch.ones((1, 1), dtype=torch.float16),
            torch.zeros((1, 4), dtype=torch.int32),
            1,
            1,
            1,
            1,
            backend="fallback",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension gate")
def test_bev_pool_optimized_forward_backward_and_autocast_policy(tmp_path):
    generator = torch.Generator(device="cpu").manual_seed(29)
    count, channels = 4096, 17
    values_cpu = torch.randn((count, channels), generator=generator, dtype=torch.float32)
    geometry_cpu = torch.stack(
        (
            torch.randint(0, 11, (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, 7, (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, 2, (count,), generator=generator, dtype=torch.int32),
            torch.randint(0, 3, (count,), generator=generator, dtype=torch.int32),
        ),
        dim=1,
    ).contiguous()
    values_fallback = values_cpu.cuda().requires_grad_(True)
    values_optimized = values_cpu.cuda().requires_grad_(True)
    geometry = geometry_cpu.cuda()
    fallback = bev_pool(
        values_fallback, geometry, 3, 2, 7, 11, backend="fallback"
    )
    with torch.autocast("cuda", dtype=torch.float16):
        optimized = bev_pool(
            values_optimized,
            geometry,
            3,
            2,
            7,
            11,
            backend="optimized",
            build_directory=_build_directory(tmp_path),
        )
    assert fallback.dtype == optimized.dtype == torch.float32
    torch.testing.assert_close(optimized, fallback, rtol=1e-5, atol=1e-6)
    output_gradient = torch.randn(
        fallback.shape, generator=generator, dtype=torch.float32
    ).cuda()
    fallback.backward(output_gradient)
    optimized.backward(output_gradient)
    assert torch.equal(values_optimized.grad, values_fallback.grad)
    assert torch.isfinite(optimized).all()
    assert torch.isfinite(values_optimized.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extension gate")
def test_bev_pool_optimized_obeys_nondefault_current_stream(tmp_path):
    values = torch.arange(24, device="cuda", dtype=torch.float32).view(6, 4)
    geometry = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
         [2, 1, 0, 0], [2, 1, 0, 0], [2, 1, 0, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        observed = bev_pool(
            values,
            geometry,
            1,
            1,
            2,
            3,
            backend="optimized",
            build_directory=_build_directory(tmp_path),
        )
        expected = bev_pool(values, geometry, 1, 1, 2, 3, backend="fallback")
        assert torch.equal(observed, expected)
    stream.synchronize()
