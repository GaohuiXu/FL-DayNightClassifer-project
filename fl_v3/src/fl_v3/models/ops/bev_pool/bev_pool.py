"""Fail-closed BEV pooling dispatcher for S10 Phase I.

The fallback uses PyTorch's sorted, length-delimited segment reduction.  Its
per-cell sequential FP32 accumulation order matches the pinned MIT CUDA kernel
without depending on the in-tree extension.  The optimized backend is an
independently integrated CUDA extension derived from that operation under
Apache-2.0; see ``NOTICE`` in this directory.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import threading
from typing import Any

import torch


BEV_POOL_BACKENDS = ("fallback", "optimized")
_SOURCE_DIR = Path(__file__).resolve().parent / "src"
_SOURCE_FILES = (_SOURCE_DIR / "bev_pool.cpp", _SOURCE_DIR / "bev_pool_cuda.cu")
_EXTENSION = None
_EXTENSION_LOCK = threading.Lock()


def _source_sha256() -> str:
    digest = hashlib.sha256()
    for path in _SOURCE_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def bev_pool_build_identity() -> dict[str, Any]:
    return {
        "schema": "s10.phase1.bev-pool-build.v1",
        "source_sha256": _source_sha256(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "backend_names": list(BEV_POOL_BACKENDS),
    }


def load_optimized_extension(*, build_directory: str | None = None):
    """Build/load the exact source identity once; never silently fall back."""
    global _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if not torch.cuda.is_available():
            raise RuntimeError("optimized BEV pooling requires a CUDA runtime")
        from torch.utils.cpp_extension import load

        source_sha = _source_sha256()
        name = f"fl_v3_bev_pool_{source_sha[:12]}"
        if build_directory is None:
            build_directory = os.environ.get("FL_V3_BEV_POOL_BUILD_DIR")
        if not build_directory:
            raise RuntimeError(
                "FL_V3_BEV_POOL_BUILD_DIR (or build_directory) must bind the "
                "request-scoped /nobackup CUDA build root"
            )
        directory = Path(build_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        _EXTENSION = load(
            name=name,
            sources=[str(path) for path in _SOURCE_FILES],
            build_directory=str(directory),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            with_cuda=True,
            verbose=True,
        )
        return _EXTENSION


def _validate_inputs(
    values: torch.Tensor,
    geometry: torch.Tensor,
    batch_size: int,
    depth: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    dimensions = tuple(int(value) for value in (batch_size, depth, height, width))
    if any(value <= 0 for value in dimensions):
        raise ValueError("BEV pooling output dimensions must be positive")
    if values.ndim != 2 or values.dtype != torch.float32 or not values.is_contiguous():
        raise TypeError("BEV pooling values must be contiguous FP32 [P,C]")
    if (
        geometry.ndim != 2
        or geometry.shape[1] != 4
        or geometry.dtype != torch.int32
        or not geometry.is_contiguous()
    ):
        raise TypeError("BEV pooling geometry must be contiguous int32 [P,4]")
    if geometry.shape[0] != values.shape[0] or geometry.device != values.device:
        raise ValueError("BEV pooling feature/geometry rows or devices differ")
    if values.shape[1] <= 0:
        raise ValueError("BEV pooling requires at least one feature channel")
    if geometry.numel():
        lower = bool((geometry >= 0).all().item())
        limits = geometry.new_tensor([width, height, depth, batch_size])
        upper = bool((geometry < limits).all().item())
        if not lower or not upper:
            raise ValueError("BEV pooling geometry is outside [x,y,z,b] bounds")
    return dimensions


def _ranks(
    geometry: torch.Tensor,
    batch_size: int,
    depth: int,
    height: int,
    width: int,
) -> torch.Tensor:
    coords = geometry.to(torch.int64)
    return (
        coords[:, 0] * (height * depth * batch_size)
        + coords[:, 1] * (depth * batch_size)
        + coords[:, 2] * batch_size
        + coords[:, 3]
    )


def _sorted_inputs(
    values: torch.Tensor,
    geometry: torch.Tensor,
    ranks: torch.Tensor,
    *,
    optimized: bool,
    rank_cardinality: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if optimized and ranks.numel():
        # Preserve the canonical frustum-row order exactly while allowing the
        # CUDA sorter to use its faster unique-key path.  The composite key is
        # equivalent to a stable rank sort: rank is primary and source row is
        # the tie-breaker.  Guard the signed-int64 key width from dimensions,
        # without a device synchronization on ranks.max().
        source_bits = max(1, (int(ranks.numel()) - 1).bit_length())
        if (int(rank_cardinality) - 1).bit_length() + source_bits > 62:
            raise OverflowError("BEV pooling composite sort key exceeds int64")
        source_rows = torch.arange(
            ranks.numel(), device=ranks.device, dtype=torch.int64
        )
        composite = torch.bitwise_or(ranks << source_bits, source_rows)
        order = torch.argsort(composite)
    else:
        # The diagnostic oracle states its stable ordering directly.
        order = torch.argsort(ranks, stable=True)
    return (
        values.index_select(0, order).contiguous(),
        geometry.index_select(0, order).contiguous(),
        ranks.index_select(0, order).contiguous(),
    )


def _fallback(
    values: torch.Tensor,
    geometry: torch.Tensor,
    ranks: torch.Tensor,
    batch_size: int,
    depth: int,
    height: int,
    width: int,
) -> torch.Tensor:
    channels = int(values.shape[1])
    if values.shape[0] == 0:
        return values.sum() * 0.0 + values.new_zeros(
            (batch_size, channels, depth, height, width)
        )
    starts = torch.ones(ranks.shape[0], device=ranks.device, dtype=torch.bool)
    starts[1:] = ranks[1:] != ranks[:-1]
    start_indices = torch.where(starts)[0]
    ends = torch.cat(
        (start_indices[1:], start_indices.new_tensor([values.shape[0]])), dim=0
    )
    lengths = (ends - start_indices).contiguous()
    # PyTorch's CUDA SegmentReduce kernel assigns one thread to each
    # (segment, channel) and adds rows from start to end in order, exactly like
    # the pinned MIT kernel.  Unlike a global cumsum/difference shortcut, a
    # preceding cell's large partial sum cannot perturb another cell.
    sums = torch.segment_reduce(
        values,
        "sum",
        lengths=lengths,
        axis=0,
        unsafe=False,
    )
    unique_geometry = geometry[starts].to(torch.int64)
    # Assignment only: collision reduction already occurred above.
    flat_indices = (
        unique_geometry[:, 3] * (depth * height * width)
        + unique_geometry[:, 2] * (height * width)
        + unique_geometry[:, 1] * width
        + unique_geometry[:, 0]
    )
    canvas = values.new_zeros((batch_size * depth * height * width, channels))
    canvas.index_copy_(0, flat_indices, sums)
    return canvas.view(batch_size, depth, height, width, channels).permute(
        0, 4, 1, 2, 3
    ).contiguous()


class _OptimizedBEVPool(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        geometry: torch.Tensor,
        ranks: torch.Tensor,
        batch_size: int,
        depth: int,
        height: int,
        width: int,
        build_directory: str | None,
    ):
        extension = load_optimized_extension(build_directory=build_directory)
        starts_mask = torch.ones(
            values.shape[0], device=values.device, dtype=torch.bool
        )
        starts_mask[1:] = ranks[1:] != ranks[:-1]
        starts = torch.where(starts_mask)[0].to(torch.int32).contiguous()
        lengths = torch.empty_like(starts)
        lengths[:-1] = starts[1:] - starts[:-1]
        lengths[-1] = int(values.shape[0]) - starts[-1]
        output = extension.forward(
            values,
            geometry,
            starts,
            lengths,
            batch_size,
            depth,
            height,
            width,
        )
        ctx.save_for_backward(geometry, starts, lengths)
        ctx.shape = (batch_size, depth, height, width)
        ctx.build_directory = build_directory
        return output.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        geometry, starts, lengths = ctx.saved_tensors
        extension = load_optimized_extension(build_directory=ctx.build_directory)
        gradient = output_gradient.permute(0, 2, 3, 4, 1).contiguous()
        values_gradient = extension.backward(
            gradient, geometry, starts, lengths, *ctx.shape
        )
        return values_gradient, None, None, None, None, None, None, None


def bev_pool(
    values: torch.Tensor,
    geometry: torch.Tensor,
    batch_size: int,
    depth: int,
    height: int,
    width: int,
    *,
    backend: str,
    build_directory: str | None = None,
) -> torch.Tensor:
    """Pool ``[P,C]`` values at integer ``[x,y,z,b]`` coordinates."""
    batch_size, depth, height, width = _validate_inputs(
        values, geometry, batch_size, depth, height, width
    )
    if backend not in BEV_POOL_BACKENDS:
        raise ValueError(f"unknown BEV pooling backend {backend!r}")
    ranks = _ranks(geometry, batch_size, depth, height, width)
    sorted_values, sorted_geometry, sorted_ranks = _sorted_inputs(
        values,
        geometry,
        ranks,
        optimized=backend == "optimized",
        rank_cardinality=batch_size * depth * height * width,
    )
    if backend == "fallback":
        return _fallback(
            sorted_values,
            sorted_geometry,
            sorted_ranks,
            batch_size,
            depth,
            height,
            width,
        )
    if sorted_values.shape[0] == 0:
        return sorted_values.sum() * 0.0 + sorted_values.new_zeros(
            (batch_size, sorted_values.shape[1], depth, height, width)
        )
    if not sorted_values.is_cuda:
        raise RuntimeError("optimized BEV pooling refuses non-CUDA tensors")
    return _OptimizedBEVPool.apply(
        sorted_values,
        sorted_geometry,
        sorted_ranks,
        batch_size,
        depth,
        height,
        width,
        build_directory,
    )
