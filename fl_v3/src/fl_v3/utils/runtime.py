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

**Precision regime (D16, 2026-06-21 — supersedes "bit-determinism is sacred").** There is now ONE
precision knob: ``enforce_determinism(precision=...)`` with ``precision`` ∈ {``bf16``, ``fp32``}.
``bf16`` is the **science path** (bf16-AMP; reported numbers use ≥3 seeds and clear the seed-variance
floor — NOT same-seed byte-identity). ``fp32`` is the **offline dev-regression / determinism tool**
(true IEEE-fp32, same-seed byte-identical on one GPU tier; it caught two real bugs). Every RNG still
flows through ``derive_seed`` (per-leaf) / ``seed_everything`` (global). The model stays
atomic-free-by-construction (the static-AST ban + permutation-invariance tests), but the ``bf16`` path
deliberately allows the cuDNN autotuner + atomic scatter (the relaxed LSS rewrite) for speed, so its
same-seed runs are NOT byte-identical — that is by design under D16. See ``docs/determinism.md`` and the
D15/D16 amendment in ``docs/cycle_04/decisions.md``.
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

# Precision regime (Cycle-04 D16 — collapses the old D13/D14 ``numeric-mode {fp32,tf32}`` ×
# D15 ``determinism-level {strict,relaxed}`` two-axis knob into ONE axis). The regime is an
# EXPLICIT, LOGGED choice — never a silent backend default. Two values:
#   "bf16"  — the SCIENCE path (D16). bf16-AMP: the train loop wraps the forward in
#             ``torch.autocast(bf16)`` (heavy ops bf16); fp32 stability ops + the optimizer stay
#             fp32. Residual fp32 ops run at true FP32 (TF32 OFF + matmul 'highest'). cuDNN
#             autotuner ON + atomic scatter (the relaxed LSS rewrite) ALLOWED → same-seed runs are
#             NOT byte-identical; the bar is ≥3-seed claim-reproducibility, not byte-identity.
#   "fp32"  — the offline dev-regression / determinism TOOL (D16). True IEEE FP32 everywhere (no
#             autocast): TF32 OFF + matmul 'highest' + cuDNN deterministic + the autotuner OFF +
#             ``use_deterministic_algorithms``. Same-seed byte-identical on ONE GPU tier
#             (architecture-pinned, D9). Run it when touching a determinism-sensitive op.
# TF32 is RETIRED (D16 — it was the D14 Alvis regime, made redundant by bf16-AMP). Mutually
# exclusive within any scientific comparison ("no mixing regimes", D14/D16).
_VALID_PRECISIONS = frozenset({"bf16", "fp32"})


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
    # THROUGHPUT (MCR P1 multi-sweep): pin each worker to 1 intra-op thread. With N ranks × M workers on
    # an N*M-CPU node, the parent's OMP/BLAS thread count is multiplied by every worker → massive
    # oversubscription thrashing on the tiny ego-compensation matmul in _load_multisweep. This caps torch's
    # pool; the numpy/BLAS pool is capped by the OMP/OPENBLAS/MKL_NUM_THREADS env the launcher exports
    # pre-import (forked workers inherit it). Value-neutral (thread count never changes a result) — pure timing.
    torch.set_num_threads(1)
    seed = int(info.seed) % (2 ** 32)
    _random.seed(seed)
    _np.random.seed(seed)


def enforce_determinism(strict: bool = True, precision: str = "fp32") -> None:
    """Pin the global PyTorch numeric + determinism state (single source of truth — D16).

    ONE knob: ``precision`` ∈ {``bf16``, ``fp32``}. Sets, in order:
      * ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` (warns if CUDA already initialised — it must be set
        before the first CUDA context for cuBLAS GEMM reproducibility).
      * the float32-matmul/conv flags — IDENTICAL for both regimes now that TF32 is retired (D16):
        ``cuda.matmul.allow_tf32=False`` + ``cudnn.allow_tf32=False`` +
        ``set_float32_matmul_precision('highest')``. Under ``bf16`` the heavy ops are bf16 via the
        train-loop autocast, so these flags govern only the residual fp32 (stability) ops.
      * ``bf16`` (SCIENCE): ``cudnn.deterministic=False`` + ``benchmark=True`` (autotuner) +
        ``use_deterministic_algorithms(False)`` (atomic scatter / the relaxed LSS rewrite allowed).
      * ``fp32`` (DEV/DETERMINISM TOOL): ``cudnn.deterministic=True`` + ``benchmark=False`` +
        ``use_deterministic_algorithms(True, warn_only=not strict)``.

    The autocast dtype is selected downstream in ``training/loop.py::train_one_epoch`` by reading back
    ``not cudnn.deterministic`` — so ``precision='bf16'`` (deterministic=False) turns bf16-AMP ON and
    ``precision='fp32'`` (deterministic=True) turns it OFF (true fp32). Keeping that sniff means the loop
    needs no signature change and the fp32 path stays byte-identical to the pre-D16 strict regime.

    ``strict`` only bites in the ``fp32`` path: ``use_deterministic_algorithms(warn_only=not strict)`` —
    with ``strict=True`` any op lacking a deterministic kernel (``grid_sample`` backward, adaptive-pool
    backward) RAISES at its call site. In ``bf16`` ``strict`` is ignored (atomics are allowed by design).
    Flip ``strict=False`` only to bring up an op being made deterministic.

    **Default is ``precision='fp32', strict=True`` ⇒ the byte-identical regime.** This is intentional:
    every gate script / determinism test that calls ``enforce_determinism(strict=True)`` keeps its exact
    pre-D16 behaviour (the offline dev-regression tool). The SCIENCE default (``precision='bf16'``) lives
    at the CONFIG layer (``run_config.get('precision', 'bf16')``), not in this function's signature.

    **Do NOT mix precision regimes within one scientific comparison** (D14/D16). Idempotent and ~free;
    safe to call once per process at startup and again per train/eval call.
    """
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"precision={precision!r} not in {sorted(_VALID_PRECISIONS)} "
            "(D16 single knob: bf16=science, fp32=dev/deterministic; tf32 retired)."
        )
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

    # Float32-matmul/conv flags — TF32 is RETIRED (D16), so BOTH regimes run residual fp32 ops at true
    # FP32. Under ``bf16`` the heavy ops are bf16 via the train-loop autocast; these flags govern only
    # the fp32 stability ops. Under ``fp32`` everything is true IEEE FP32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Precision regime (D16). ``bf16`` = SCIENCE: unlock the cuDNN autotuner + atomic ops
    # (scatter_add/index_add — the relaxed LSS rewrite) + AMP; same-seed runs are no longer byte-
    # identical (by design; the bar is ≥3-seed claim-reproducibility). ``fp32`` = the offline
    # dev-regression / determinism tool: the byte-identical contract (cuDNN deterministic, autotuner off).
    if precision == "bf16":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True            # conv autotuner (L1) — OFF under fp32 for determinism
        torch.use_deterministic_algorithms(False)        # allow atomicAdd scatter (L3b), etc.
    else:  # "fp32" — the byte-identical dev/determinism tool (catches determinism-sensitive op bugs)
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


def precision_state() -> dict:
    """Snapshot the live precision-regime backend flags (for startup logs + the run manifest).

    The single source of truth for "what precision did this run actually use" — every scientific run
    must log this so the regime is recorded, not inferred (D14/D16). ``precision`` is the canonical D16
    field (``bf16`` science / ``fp32`` dev-determinism), derived from the live cuDNN-deterministic flag
    that ``enforce_determinism`` set; ``determinism_level`` is retained as a back-compat alias. Includes
    the device + compute-capability so the run's GPU tier is recorded (determinism is architecture-pinned,
    D9). The legacy ``*_allow_tf32`` / ``tf32_*`` fields now always report OFF (TF32 retired, D16).
    """
    state = {
        "precision": "fp32" if torch.backends.cudnn.deterministic else "bf16",
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "determinism_level": "strict" if torch.backends.cudnn.deterministic else "relaxed",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        cc = torch.cuda.get_device_capability(0)
        state["device_name"] = torch.cuda.get_device_name(0)
        state["device_cc"] = f"{cc[0]}.{cc[1]}"
        # TF32 tensor cores require cc>=8 (Ampere+); Turing/T4 (cc 7.5) falls back to FP32.
        state["tf32_hardware"] = bool(cc[0] >= 8)
        state["tf32_engaged"] = bool(
            cc[0] >= 8 and torch.backends.cuda.matmul.allow_tf32
        )
    else:
        state["device_name"] = "cpu"
        state["device_cc"] = ""
        state["tf32_hardware"] = False
        state["tf32_engaged"] = False
    return state
