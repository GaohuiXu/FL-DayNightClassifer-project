"""Login-node unit test: multi-worker DataLoader is bit-deterministic.

Verifies the unified-fix invariant: with `num_workers > 0`,
`worker_init_fn=seeded_worker_init`, and a seeded
`generator=torch.Generator().manual_seed(...)`, two independent
DataLoader iterations over the same dataset at the same seed produce
**bit-identical** augmented batches.

Without `worker_init_fn`, torchvision augmentations that reach into
`numpy.random` or stdlib `random` would diverge across runs. This
test catches that regression.

Run from the project root, login node OK:

    module purge
    module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
    source ../.venv/bin/activate   # or full venv path
    python -m pytest tests/test_dataloader_determinism.py -xvs

Or directly without pytest:

    python tests/test_dataloader_determinism.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB

from fl_v2.data.dataset import TransformedSubset
from fl_v2.data.transforms import (
    get_train_transforms,
    get_train_transforms_split,
)
from fl_v2.utils.runtime import seeded_worker_init


# Project default location of the GTSRB dataset on Mimer.
_DEFAULT_DATA_ROOT = (
    "/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
)


def _build_loader(
    data_root: str,
    num_workers: int,
    seed: int,
    n_samples: int = 100,
    batch_size: int = 16,
) -> DataLoader:
    """Build a small GTSRB train DataLoader with the unified-fix knobs.

    NB: the test process re-seeds `random` and `np.random` here so that
    each call simulates a fresh process start. Without this, sequentially
    constructing two loaders in the SAME pytest process means the second
    one inherits the first's now-advanced global RNG state and would
    produce different augmentations even at fixed seed — a property of
    the test harness, not of the production pipeline (where each SLURM
    job starts a fresh process with `torch.manual_seed(seed)` at startup).
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    full_train = GTSRB(
        root=data_root, split="train", download=False,
        transform=get_train_transforms(image_size=32),
    )
    indices = list(range(min(n_samples, len(full_train))))
    subset = Subset(full_train, indices)

    gen = torch.Generator().manual_seed(seed)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=gen,
    )
    if num_workers > 0:
        kwargs.update(
            worker_init_fn=seeded_worker_init,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(subset, **kwargs)


def _collect_one_epoch(loader: DataLoader) -> list[torch.Tensor]:
    """Run one full epoch and return the list of augmented image batches."""
    batches: list[torch.Tensor] = []
    for images, _labels in loader:
        # Detach + clone so we keep the tensor stable even if the worker
        # buffer is recycled by persistent_workers.
        batches.append(images.detach().clone())
    return batches


def _assert_bit_identical(
    a: list[torch.Tensor], b: list[torch.Tensor], tag: str,
) -> None:
    assert len(a) == len(b), (
        f"{tag}: different number of batches "
        f"({len(a)} vs {len(b)})"
    )
    for i, (ba, bb) in enumerate(zip(a, b)):
        assert ba.shape == bb.shape, (
            f"{tag}: batch {i} shape mismatch {ba.shape} vs {bb.shape}"
        )
        if not torch.equal(ba, bb):
            max_diff = (ba - bb).abs().max().item()
            n_differ = int((ba != bb).sum().item())
            n_total = ba.numel()
            raise AssertionError(
                f"{tag}: batch {i} not bit-identical — "
                f"max abs diff {max_diff:.3e}, "
                f"{n_differ}/{n_total} elements differ"
            )


def _data_root() -> str:
    return os.environ.get("GTSRB_DATA_ROOT", _DEFAULT_DATA_ROOT)


def _ensure_gtsrb_available() -> None:
    if not Path(_data_root()).exists():
        raise FileNotFoundError(
            f"GTSRB dataset root not found at {_data_root()}. "
            f"Set GTSRB_DATA_ROOT to override."
        )


def test_multiworker_is_bit_identical_at_fixed_seed():
    """Two same-seed DataLoaders with num_workers=4 produce identical batches."""
    _ensure_gtsrb_available()
    seed = 4242
    a = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=4, seed=seed)
    )
    b = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=4, seed=seed)
    )
    _assert_bit_identical(a, b, tag="num_workers=4 same seed")


def test_singleworker_is_bit_identical_at_fixed_seed():
    """Sanity: num_workers=0 (existing behaviour) is also bit-identical."""
    _ensure_gtsrb_available()
    seed = 4242
    a = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=0, seed=seed)
    )
    b = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=0, seed=seed)
    )
    _assert_bit_identical(a, b, tag="num_workers=0 same seed")


def _build_loader_split(
    data_root: str,
    num_workers: int,
    seed: int,
    n_samples: int = 100,
    batch_size: int = 16,
) -> DataLoader:
    """Build a TransformedSubset-backed loader using the split (pre, post)
    transforms. Bit-equality with `_build_loader` (which uses the fused
    chain) is the C5 / pre-cache contract.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    base = GTSRB(
        root=data_root, split="train", download=False,
        transform=None,  # raw PIL — TransformedSubset applies the chain
    )
    pre, post = get_train_transforms_split(image_size=32)
    indices = list(range(min(n_samples, len(base))))
    subset = TransformedSubset(
        base_dataset=base,
        indices=indices,
        transform=post,
        pre_transform=pre,
    )

    gen = torch.Generator().manual_seed(seed)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        generator=gen,
    )
    if num_workers > 0:
        kwargs.update(
            worker_init_fn=seeded_worker_init,
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(subset, **kwargs)


def test_pre_cache_is_bit_identical_to_fused_chain():
    """C5 contract: pre_transform=Resize cache must produce the same
    augmented batches as the fused Resize+Rotation+Jitter+ToTensor+Normalize
    chain at the same seed.

    Resize is purely deterministic (PIL bilinear is byte-stable for fixed
    input + fixed output size). Pre-applying it once at __init__ and
    storing the resized PIL image must produce bit-equivalent
    DataLoader output to applying it inside the per-__getitem__ chain.
    """
    _ensure_gtsrb_available()
    seed = 4242
    fused = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=4, seed=seed)
    )
    cached = _collect_one_epoch(
        _build_loader_split(_data_root(), num_workers=4, seed=seed)
    )
    _assert_bit_identical(
        fused, cached, tag="fused-chain vs pre_transform-cache (num_workers=4)",
    )


def test_pre_cache_is_bit_identical_singleworker():
    """Same C5 contract, num_workers=0 path."""
    _ensure_gtsrb_available()
    seed = 4242
    fused = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=0, seed=seed)
    )
    cached = _collect_one_epoch(
        _build_loader_split(_data_root(), num_workers=0, seed=seed)
    )
    _assert_bit_identical(
        fused, cached, tag="fused-chain vs pre_transform-cache (num_workers=0)",
    )


def test_full_pre_cache_eval_chain_bit_identical():
    """For deterministic-only chains (eval-side: Resize+ToTensor+Normalize)
    we cache the FINAL tensor (transform=None, pre_transform=full chain).
    Must be bit-equivalent to applying the chain inline per __getitem__.

    Used by `get_global_testloader` and the valloader to skip per-batch
    CPU work entirely.
    """
    _ensure_gtsrb_available()
    from fl_v2.data.transforms import get_eval_transforms

    seed = 4242
    base = GTSRB(
        root=_data_root(), split="train", download=False, transform=None,
    )
    indices = list(range(min(100, len(base))))
    eval_chain = get_eval_transforms(image_size=32)

    # Inline path: transform applied per __getitem__
    inline = TransformedSubset(
        base_dataset=base, indices=indices,
        transform=eval_chain, pre_transform=None,
    )
    # Cached path: transform applied once at __init__, output cached
    cached = TransformedSubset(
        base_dataset=base, indices=indices,
        transform=None, pre_transform=eval_chain,
    )

    inline_loader = DataLoader(inline, batch_size=16, shuffle=False, num_workers=0)
    cached_loader = DataLoader(cached, batch_size=16, shuffle=False, num_workers=0)

    inline_batches = _collect_one_epoch(inline_loader)
    cached_batches = _collect_one_epoch(cached_loader)
    _assert_bit_identical(
        inline_batches, cached_batches,
        tag="full-pre-cache vs inline (eval chain, no augmentation)",
    )


def test_different_seeds_diverge():
    """Sanity: different seeds DO produce different augmented batches."""
    _ensure_gtsrb_available()
    a = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=4, seed=42)
    )
    b = _collect_one_epoch(
        _build_loader(_data_root(), num_workers=4, seed=43)
    )
    # We expect at least one batch to differ — not all augmentations
    # are identical at different seeds.
    any_differs = any(
        not torch.equal(ba, bb) for ba, bb in zip(a, b)
    )
    assert any_differs, (
        "num_workers=4 produced bit-identical batches across seeds 42 vs 43 — "
        "the seed plumbing is broken (no augmentation noise being applied)."
    )


if __name__ == "__main__":
    # Allow running without pytest. Each test is independent.
    test_multiworker_is_bit_identical_at_fixed_seed()
    print("[ok] test_multiworker_is_bit_identical_at_fixed_seed")
    test_singleworker_is_bit_identical_at_fixed_seed()
    print("[ok] test_singleworker_is_bit_identical_at_fixed_seed")
    test_pre_cache_is_bit_identical_to_fused_chain()
    print("[ok] test_pre_cache_is_bit_identical_to_fused_chain")
    test_pre_cache_is_bit_identical_singleworker()
    print("[ok] test_pre_cache_is_bit_identical_singleworker")
    test_full_pre_cache_eval_chain_bit_identical()
    print("[ok] test_full_pre_cache_eval_chain_bit_identical")
    test_different_seeds_diverge()
    print("[ok] test_different_seeds_diverge")
    print("ALL TESTS PASSED")
