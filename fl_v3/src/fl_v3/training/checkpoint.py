"""Complete, identity-bound, boundary-safe S06 checkpoints."""
from __future__ import annotations

import os
import random
import tempfile
from typing import Any

import numpy as np
import torch

from fl_v3.config import ResolvedConfig
from fl_v3.training.runtime_state import TrainingState


CHECKPOINT_SCHEMA = "s06.checkpoint.v1"
_FIELDS = frozenset({
    "schema", "model", "optimizer", "scheduler", "grad_scaler", "ema", "training_state",
    "rng", "resolved_config_sha256", "resolved_config", "model_mode", "precision",
    "data_identities", "checkpoint_identity",
})


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(raw: dict[str, Any]) -> None:
    if set(raw) != {"python", "numpy", "torch", "cuda"}:
        raise RuntimeError("checkpoint RNG state is partial or unknown")
    random.setstate(raw["python"])
    np.random.set_state(raw["numpy"])
    torch.set_rng_state(raw["torch"])
    if raw["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG states but CUDA is unavailable")
        torch.cuda.set_rng_state_all(raw["cuda"])


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_scaler: Any,
    ema: Any,
    state: TrainingState,
    config: ResolvedConfig,
    checkpoint_identity: str,
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "grad_scaler": None if grad_scaler is None else grad_scaler.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "training_state": state.checkpoint_dict(),
        "rng": _rng_state(),
        "resolved_config_sha256": config.sha256,
        "resolved_config": config.as_dict(),
        "model_mode": config.model_mode,
        "precision": config.precision,
        "data_identities": config.data_identities,
        "checkpoint_identity": str(checkpoint_identity),
    }
    parent = os.path.dirname(os.path.abspath(path)); os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".s06-ckpt-", dir=parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_scaler: Any,
    ema: Any,
    config: ResolvedConfig,
    map_location: Any = "cpu",
) -> tuple[TrainingState, str]:
    raw = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        got = set(raw) if isinstance(raw, dict) else set()
        raise RuntimeError(
            f"legacy/partial checkpoint refused: missing={sorted(_FIELDS-got)}, "
            f"unknown={sorted(got-_FIELDS)}"
        )
    if raw["schema"] != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"unsupported checkpoint schema {raw['schema']!r}")
    expected = {
        "resolved_config_sha256": config.sha256,
        "model_mode": config.model_mode,
        "precision": config.precision,
        "data_identities": config.data_identities,
        "resolved_config": config.as_dict(),
    }
    drift = [k for k, value in expected.items() if raw[k] != value]
    if drift:
        raise RuntimeError(f"checkpoint/config/data identity drift: {drift}")
    if (scheduler is None) != (raw["scheduler"] is None):
        raise RuntimeError("checkpoint scheduler presence mismatch")
    if (grad_scaler is None) != (raw["grad_scaler"] is None):
        raise RuntimeError("checkpoint GradScaler presence mismatch")
    if (ema is None) != (raw["ema"] is None):
        raise RuntimeError("checkpoint EMA presence mismatch")
    model.load_state_dict(raw["model"], strict=True)
    optimizer.load_state_dict(raw["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(raw["scheduler"])
    if grad_scaler is not None:
        grad_scaler.load_state_dict(raw["grad_scaler"])
    if ema is not None:
        ema.load_state_dict(raw["ema"])
    state = TrainingState.from_checkpoint(raw["training_state"])
    _restore_rng(raw["rng"])
    return state, str(raw["checkpoint_identity"])
