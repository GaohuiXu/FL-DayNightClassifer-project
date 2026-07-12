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

**Arrhenius precision regime (2026-07).** There is one explicit precision knob:
``enforce_determinism(precision=...)`` with ``precision`` in {``fp32``, ``fp16``}.
``fp32`` is the dev/debug/reference path. ``fp16`` is the supported sparse-training
path: CUDA autocast fp16 plus ``GradScaler(init_scale=512)`` in trainer/smoke code.
Direct sparse ``bf16`` is not supported by the validated GH200 cumm/spconv path and
must fail loudly.
"""
from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import random as _random
import subprocess
from urllib.parse import unquote, urlparse
import warnings
from contextlib import nullcontext

import numpy as _np
import torch


_TRUTHY = frozenset({"true", "1", "yes", "y", "on"})
_FALSY = frozenset({"false", "0", "no", "n", "off", ""})

# Per CUDA docs + torch determinism guide: the workspace config that makes
# cuBLAS GEMMs reproducible. Must be in the environment BEFORE the first CUDA
# context is created, so enforce_determinism sets it defensively and warns if
# CUDA is already initialised. Arrhenius launchers / SLURM also export it up front.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# Arrhenius precision policy: fp32 for dev/debug/reference; fp16 for supported
# sparse training via CUDA autocast + GradScaler. Direct sparse bf16 is rejected
# by normalize_precision/validate_sparse_precision instead of silently falling back.
_VALID_PRECISIONS = frozenset({"fp16", "fp32"})
_CURRENT_PRECISION = "fp32"
_MODEL_MODES = frozenset({"camera_only", "lidar_only", "fusion"})


def normalize_precision(precision: str | None) -> str:
    """Return a supported Arrhenius precision string or raise loudly."""
    p = str(precision or "fp32").strip().lower()
    if p not in _VALID_PRECISIONS:
        extra = ""
        if p == "bf16":
            extra = " Direct sparse bf16 is unsupported on Arrhenius; use fp32 or fp16."
        raise ValueError(
            f"precision={precision!r} not in {sorted(_VALID_PRECISIONS)} "
            "(Arrhenius policy: fp32=dev/reference, fp16=AMP+GradScaler sparse training)."
            + extra
        )
    return p


def normalize_model_mode(mode: str | None) -> str:
    """Accept only the three production topology names; aliases are forbidden."""
    if not isinstance(mode, str) or mode not in _MODEL_MODES:
        raise ValueError(f"model-mode={mode!r} not in {sorted(_MODEL_MODES)}")
    return mode


def require_spconv_238() -> None:
    """Fail before sparse model construction unless the reviewed runtime is exact."""
    try:
        version = importlib.metadata.version("spconv")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("lidar/fusion mode requires installed spconv==2.3.8") from exc
    if version != "2.3.8":
        raise RuntimeError(f"lidar/fusion mode requires spconv==2.3.8, found {version!r}")


def _source_checkout_identity(distribution: str, import_name: str) -> tuple[str, str]:
    """Return ``(clean Git HEAD, import origin)`` for one editable dependency."""
    dist = importlib.metadata.distribution(distribution)
    try:
        direct = json.loads(dist.read_text("direct_url.json") or "")
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{distribution} lacks valid direct_url build provenance") from exc
    parsed = urlparse(str(direct.get("url", "")))
    if parsed.scheme != "file":
        vcs = direct.get("vcs_info", {})
        commit = str(vcs.get("commit_id", ""))
        if len(commit) != 40:
            raise RuntimeError(f"{distribution} direct_url lacks an exact source commit")
        source = ""
        head = commit
    else:
        source = unquote(parsed.path)
        if not source:
            raise RuntimeError(f"{distribution} direct_url has no source checkout path")
        try:
            head = subprocess.run(
                ["git", "-C", source, "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", source, "status", "--porcelain", "--untracked-files=no"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot attest {distribution} source checkout {source!r}") from exc
        if dirty:
            raise RuntimeError(f"{distribution} source checkout is modified")
    spec = importlib.util.find_spec(import_name)
    if spec is None or not spec.origin:
        raise RuntimeError(f"cannot resolve installed import origin for {import_name}")
    origin = str(Path(spec.origin).resolve())
    if source and Path(source).resolve() not in Path(origin).parents:
        raise RuntimeError(
            f"{distribution} import origin {origin!r} is not from attested source {source!r}"
        )
    return head, origin


def verify_runtime_dependency_identity(run_config: dict) -> dict[str, str]:
    """Fail before construction on Torch and sparse package/source identity drift.

    The returned paths are suitable for an execution manifest.  This proves the
    installed package version, active import root, and clean source commit; an
    approved launcher must additionally hash its concrete runtime/source snapshot.
    """
    expected_torch = str(run_config.get("dependency-torch", ""))
    if not expected_torch or torch.__version__ != expected_torch:
        raise RuntimeError(
            f"Torch build identity drift: expected={expected_torch!r}, actual={torch.__version__!r}"
        )
    result = {"torch": torch.__version__}
    if str(run_config.get("det-lidar-arch")) != "second_075":
        return result
    expected = {
        "spconv": (
            "spconv", "spconv", str(run_config.get("dependency-spconv", "")),
            str(run_config.get("dependency-spconv-source-sha", "")),
        ),
        "cumm": (
            "cumm", "cumm", str(run_config.get("dependency-cumm", "")),
            str(run_config.get("dependency-cumm-source-sha", "")),
        ),
    }
    for label, (distribution, import_name, expected_version, expected_head) in expected.items():
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required sparse dependency {distribution!r} is not installed") from exc
        if actual_version != expected_version:
            raise RuntimeError(
                f"{label} package identity drift: expected={expected_version!r}, "
                f"actual={actual_version!r}"
            )
        actual_head, origin = _source_checkout_identity(distribution, import_name)
        if actual_head != expected_head:
            raise RuntimeError(
                f"{label} source identity drift: expected={expected_head!r}, actual={actual_head!r}"
            )
        result[f"{label}_version"] = actual_version
        result[f"{label}_source_sha"] = actual_head
        result[f"{label}_import_origin"] = origin
    return result


def current_precision() -> str:
    """Last precision applied through enforce_determinism."""
    return _CURRENT_PRECISION


def validate_sparse_precision(precision: str | None, lidar_encoder: str | None) -> None:
    """Reject unsupported precision/model pairings before model construction."""
    p = str(precision or "fp32").strip().lower()
    enc = str(lidar_encoder or "pillar").strip().lower()
    if enc == "voxel" and p == "bf16":
        raise ValueError(
            "precision='bf16' is unsupported for det-lidar-encoder='voxel' on Arrhenius: "
            "the validated cumm/spconv sparse path supports fp32 or fp16 AMP + GradScaler."
        )
    normalize_precision(p)


def precision_autocast_dtype(
    precision: str | None = None,
    device: torch.device | str | None = None,
) -> torch.dtype | None:
    """Autocast dtype for a precision/device pair; None means no autocast."""
    p = normalize_precision(precision or _CURRENT_PRECISION)
    dev = torch.device(device) if device is not None else None
    if p == "fp16" and (dev is None or dev.type == "cuda"):
        return torch.float16
    return None


def precision_autocast_context(precision: str | None, device: torch.device | str):
    """Context manager for the active precision policy."""
    dev = torch.device(device)
    dtype = precision_autocast_dtype(precision, dev)
    if dtype is None or dev.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type=dev.type, dtype=dtype)


def make_grad_scaler(
    device: torch.device | str,
    precision: str | None = None,
    init_scale: float = 512.0,
) -> torch.amp.GradScaler:
    """GradScaler factory: enabled only for CUDA fp16, with Arrhenius init_scale."""
    p = normalize_precision(precision or _CURRENT_PRECISION)
    dev = torch.device(device)
    return torch.amp.GradScaler(
        "cuda",
        enabled=(p == "fp16" and dev.type == "cuda"),
        init_scale=float(init_scale),
    )


def grad_scaler_init_scale_from_config(
    run_config: dict | None,
    precision: str | None = None,
) -> float:
    """Resolve the explicit GradScaler init scale for a run.

    The normal fp16 policy keeps the historical ``512`` init scale. Sparse voxel
    LiDAR can opt into a separate init scale through
    ``det-sparse-grad-scale-init``; this only applies to
    ``det-lidar-encoder='voxel'`` under fp16 and is recorded in manifests as a
    precision-policy choice, not an architecture change.
    """
    cfg = run_config or {}
    p = normalize_precision(precision or str(cfg.get("precision", _CURRENT_PRECISION)))
    if p != "fp16":
        return 0.0
    if "grad-scale-init" in cfg:
        return float(cfg["grad-scale-init"])
    enc = str(cfg.get("det-lidar-encoder", "pillar")).strip().lower()
    if enc == "voxel" and "det-sparse-grad-scale-init" in cfg:
        return float(cfg["det-sparse-grad-scale-init"])
    return 512.0


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
    """Pin global PyTorch numeric + determinism state for the Arrhenius policy.

    ``precision='fp32'`` is the deterministic dev/reference path. ``precision='fp16'``
    is the supported sparse training path; autocast and GradScaler are selected
    explicitly by the trainer/smoke helpers, not inferred from cuDNN flags.
    """
    global _CURRENT_PRECISION

    precision = normalize_precision(precision)
    _CURRENT_PRECISION = precision
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != _CUBLAS_WORKSPACE_CONFIG:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            warnings.warn(
                "CUBLAS_WORKSPACE_CONFIG set after CUDA initialisation; cuBLAS "
                "GEMM determinism is not guaranteed for this process. Set it in "
                "the environment before any CUDA use.",
                RuntimeWarning,
                stacklevel=2,
            )
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG

    # Residual fp32 ops stay true IEEE fp32 in both regimes; fp16 affects only explicit autocast regions.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    if precision == "fp16":
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
    else:
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

    Precision is recorded from the explicit runtime policy, not inferred from
    cuDNN flags. ``determinism_level`` remains a back-compat summary of the live
    backend mode; it is not the precision source of truth.
    """
    cuda_available = bool(torch.cuda.is_available())
    amp_dtype = precision_autocast_dtype(_CURRENT_PRECISION, "cuda") if cuda_available else None
    state = {
        "precision": _CURRENT_PRECISION,
        "amp_dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "",
        "grad_scaler": bool(_CURRENT_PRECISION == "fp16" and cuda_available),
        "grad_scaler_init_scale": 512.0 if _CURRENT_PRECISION == "fp16" else 0.0,
        "grad_scaler_init_scale_default": 512.0 if _CURRENT_PRECISION == "fp16" else 0.0,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "determinism_level": "strict" if torch.backends.cudnn.deterministic else "relaxed",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "cuda_available": cuda_available,
    }
    if cuda_available:
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
