"""Determinism harness + config helpers (fl_v3 T0 carry-over).

Re-implemented from the ``fl_v2`` oracle (``fl_v2/src/fl_v2/utils/runtime.py``)
and validated for byte-equivalence:

  * ``derive_seed``        — SHA-256 (run_seed, client_id, server_round) → 32-bit
                             seed. Byte-identical to the oracle (same encoding,
                             same digest slice) so a carried-over fixture seeds
                             identically across fl_v2 / fl_v3.
  * ``seeded_worker_init`` — propagate each DataLoader worker's torch seed to
                             numpy + stdlib ``random``. Identical to the oracle.
  * ``truthy``             — robust YAML/Flower bool parsing. Identical.

NEW in fl_v3 (centralizes what fl_v2 scattered inline across client/server):

  * ``enforce_determinism`` — the single place that pins the global determinism
                              state: ``CUBLAS_WORKSPACE_CONFIG``,
                              ``cudnn.deterministic=True``, ``benchmark=False``,
                              ``torch.use_deterministic_algorithms(True)``.
  * ``seed_everything``     — seed ``random`` / ``numpy`` / ``torch`` (+CUDA).

**Bit-determinism is sacred.** Every RNG in fl_v3 must be seeded via
``derive_seed`` (per-leaf) or ``seed_everything`` (global at startup); banned
ops (atomic scatter, ``grid_sample`` backward, non-stable sort/topk, flash-attn)
are forbidden and ``enforce_determinism(strict=True)`` makes the non-deterministic
ones raise rather than silently diverge.
"""
from __future__ import annotations

import hashlib
import os
import random as _random
import warnings

import numpy as _np
import torch


_TRUTHY = frozenset({"true", "1", "yes", "y", "on"})
_FALSY = frozenset({"false", "0", "no", "n", "off", ""})

# Per CUDA docs + torch determinism guide: the workspace config that makes
# cuBLAS GEMMs reproducible. Must be in the environment BEFORE the first CUDA
# context is created, so enforce_determinism sets it defensively and warns if
# CUDA is already initialised. run_alvis.sh / SLURM also exports it up front.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def truthy(value, default: bool = False) -> bool:
    """Robust boolean parsing for YAML / Flower config values.

    ``bool("false") == True`` in Python — and ``flwr run --run-config`` quotes
    YAML overrides as Python strings — so the natural
    ``bool(config.get("flag", False))`` pattern silently misreads any YAML that
    writes ``false``. This handles bool, int/float, and the canonical string
    spellings. (Byte-identical to the fl_v2 oracle.)
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in _TRUTHY:
        return True
    if s in _FALSY:
        return False
    return default


def derive_seed(run_seed: int, client_id: int = 0, server_round: int = 0) -> int:
    """Deterministic 32-bit seed from (run_seed, client_id, server_round).

    Uses ``hashlib.sha256`` rather than Python's built-in ``hash()`` so the
    output is portable across Python builds and unaffected by ``PYTHONHASHSEED``.
    The encoding via colon-separated decimal strings is unambiguous (no
    collisions across (a,b,c) triples). Caller picks which arguments to vary;
    defaults of 0 keep the API ergonomic when only ``run_seed`` is known.

    **Byte-identical to the fl_v2 oracle** — the same (run_seed, client_id,
    server_round) triple seeds fl_v2 and fl_v3 identically (parity-tested).
    """
    payload = f"{int(run_seed)}:{int(client_id)}:{int(server_round)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big")


def seeded_worker_init(worker_id: int) -> None:
    """Seed each DataLoader worker's numpy / stdlib RNG from its torch seed.

    PyTorch already seeds ``torch.default_generator`` inside each worker using
    the DataLoader's ``generator=`` argument; this propagates that seed to
    ``numpy.random`` and stdlib ``random`` so any transform reaching for those
    produces bit-deterministic batches across runs at the same seed. Pass as
    ``DataLoader(worker_init_fn=seeded_worker_init, ...)``. No-op outside a
    worker context. (Identical to the fl_v2 oracle.)
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = int(info.seed) % (2 ** 32)
    _random.seed(seed)
    _np.random.seed(seed)


def enforce_determinism(strict: bool = True) -> None:
    """Pin the global PyTorch determinism state (single source of truth).

    Sets, in order:
      * ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (warns if CUDA already initialised —
        it must be set before the first CUDA context for cuBLAS GEMM
        reproducibility).
      * ``torch.backends.cudnn.deterministic = True``
      * ``torch.backends.cudnn.benchmark = False`` (disable the autotuner).
      * ``torch.use_deterministic_algorithms(True, warn_only=not strict)``.

    With ``strict=True`` (default, the AD-correct setting) any op lacking a
    deterministic implementation RAISES — this is how a banned op (atomic
    scatter / ``grid_sample`` backward / non-stable sort/topk) gets caught at
    its call site instead of silently producing run-to-run drift. Use
    ``strict=False`` only for deliberate bring-up of an op being made
    deterministic.

    **INTENTIONAL divergence from the fl_v2 oracle:** fl_v2 used
    ``use_deterministic_algorithms(True, warn_only=True)`` (warn + continue).
    fl_v3 tightens the default to strict (``warn_only=False``, RAISE) because
    the AD model (T2) must have NO banned op — a silent warn is exactly the
    failure mode we cannot afford. This is an invariant strengthening, not a
    re-implementation of the oracle's flag. Flip via ``determinism-strict``
    config only to bring up an op being made deterministic.

    Idempotent and ~free; safe to call once per process at startup and again
    per train/eval call.
    """
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            warnings.warn(
                "CUBLAS_WORKSPACE_CONFIG set after CUDA initialisation; cuBLAS "
                "GEMM determinism is not guaranteed for this process. Set it in "
                "the environment (run_alvis.sh) before any CUDA use.",
                RuntimeWarning,
                stacklevel=2,
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def seed_everything(seed: int) -> None:
    """Seed ``random`` / ``numpy`` / ``torch`` (+ all CUDA devices).

    The global-RNG seeding fl_v2 does inline at server startup and per
    train/eval call, centralized. Call with a ``derive_seed`` leaf seed for
    per-client / per-round determinism.

    All three RNGs are seeded with the SAME 32-bit value: ``numpy`` rejects
    seeds ``>= 2**32``, so we reduce once and apply the reduced value to all
    three (rather than masking numpy only, which would desync the streams for a
    seed ``>= 2**32``). ``derive_seed`` outputs are already 32-bit, so this is a
    no-op for every intended caller and matches the fl_v2 oracle bit-for-bit.
    """
    s32 = int(seed) % (2 ** 32)
    _random.seed(s32)
    _np.random.seed(s32)
    torch.manual_seed(s32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s32)
