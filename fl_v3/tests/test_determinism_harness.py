"""Determinism harness: derive_seed portability + enforce_determinism flags."""
from __future__ import annotations

import os

import torch

from fl_v3.utils.runtime import derive_seed, enforce_determinism


def test_derive_seed_is_deterministic_and_portable():
    # Same inputs → same 32-bit output, every time, regardless of PYTHONHASHSEED.
    assert derive_seed(42, 0, 0) == derive_seed(42, 0, 0)
    assert derive_seed(42, 3, 7) == derive_seed(42, 3, 7)
    # 32-bit range.
    s = derive_seed(123456, 9, 2)
    assert 0 <= s < 2 ** 32


def test_derive_seed_varies_by_each_argument():
    base = derive_seed(42, 0, 0)
    assert derive_seed(43, 0, 0) != base
    assert derive_seed(42, 1, 0) != base
    assert derive_seed(42, 0, 1) != base


def test_derive_seed_matches_known_values():
    # Pin exact values so a future refactor of the encoding is caught (these are
    # the SHA-256-based values shared byte-for-byte with the fl_v2 oracle).
    import hashlib

    def ref(run_seed, cid, rnd):
        payload = f"{run_seed}:{cid}:{rnd}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")

    for args in [(42, 0, 0), (7, 3, 11), (999, 49, 20)]:
        assert derive_seed(*args) == ref(*args)


def test_enforce_determinism_sets_global_state():
    enforce_determinism(strict=True)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # Strict mode must be active (a banned op would raise, not warn).
    assert torch.are_deterministic_algorithms_enabled() is True
