"""Complete, identity-bound, boundary-safe S06 checkpoints."""
from __future__ import annotations

import copy
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
_RNG_FIELDS = frozenset({"python", "numpy", "torch", "cuda"})
_HEX = frozenset("0123456789abcdef")


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(raw: dict[str, Any]) -> None:
    random.setstate(raw["python"])
    np.random.set_state(raw["numpy"])
    torch.set_rng_state(raw["torch"])
    if raw["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG states but CUDA is unavailable")
        torch.cuda.set_rng_state_all(raw["cuda"])


def _validate_rng(raw: Any) -> None:
    """Validate every RNG state on throw-away generators without global mutation."""
    if not isinstance(raw, dict) or set(raw) != _RNG_FIELDS:
        raise RuntimeError("checkpoint RNG state is partial or unknown")
    try:
        probe_python = random.Random()
        probe_python.setstate(copy.deepcopy(raw["python"]))
    except Exception as exc:
        raise RuntimeError("checkpoint Python RNG state is invalid") from exc
    try:
        probe_numpy = np.random.RandomState()
        probe_numpy.set_state(copy.deepcopy(raw["numpy"]))
    except Exception as exc:
        raise RuntimeError("checkpoint NumPy RNG state is invalid") from exc
    cpu_state = raw["torch"]
    if not torch.is_tensor(cpu_state) or cpu_state.dtype != torch.uint8 or cpu_state.ndim != 1:
        raise RuntimeError("checkpoint Torch CPU RNG state has invalid type/dtype/shape")
    try:
        torch.Generator(device="cpu").set_state(cpu_state.clone())
    except Exception as exc:
        raise RuntimeError("checkpoint Torch CPU RNG state is invalid") from exc
    cuda_states = raw["cuda"]
    if not isinstance(cuda_states, list):
        raise RuntimeError("checkpoint CUDA RNG state must be a list")
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG states but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG state count differs from visible devices")
        for index, state in enumerate(cuda_states):
            if not torch.is_tensor(state) or state.dtype != torch.uint8 or state.ndim != 1:
                raise RuntimeError(f"checkpoint CUDA RNG state {index} has invalid type/dtype/shape")
            try:
                torch.Generator(device=f"cuda:{index}").set_state(state.clone())
            except Exception as exc:
                raise RuntimeError(f"checkpoint CUDA RNG state {index} is invalid") from exc


def _validate_model_state(model: torch.nn.Module, incoming: Any) -> None:
    current = model.state_dict()
    if not isinstance(incoming, dict) or list(incoming) != list(current):
        incoming_keys = list(incoming) if isinstance(incoming, dict) else []
        raise RuntimeError(
            f"checkpoint model keys/order mismatch: expected={list(current)}, incoming={incoming_keys}"
        )
    for name, expected in current.items():
        value = incoming[name]
        if torch.is_tensor(expected):
            if not torch.is_tensor(value):
                raise RuntimeError(f"checkpoint model tensor {name!r} is not a tensor")
            if value.shape != expected.shape or value.dtype != expected.dtype or value.layout != expected.layout:
                raise RuntimeError(
                    f"checkpoint model tensor {name!r} shape/dtype/layout mismatch: "
                    f"expected={expected.shape}/{expected.dtype}/{expected.layout}, "
                    f"incoming={value.shape}/{value.dtype}/{value.layout}"
                )
        elif type(value) is not type(expected):
            raise RuntimeError(f"checkpoint model extra state {name!r} type mismatch")


def _validate_tree_structure(name: str, current: Any, incoming: Any) -> None:
    """Require load-relevant structure while permitting scalar/tensor values to change."""
    if torch.is_tensor(current):
        if (
            not torch.is_tensor(incoming)
            or incoming.shape != current.shape
            or incoming.dtype != current.dtype
            or incoming.layout != current.layout
        ):
            raise RuntimeError(f"checkpoint {name} tensor structure mismatch")
    elif isinstance(current, dict):
        if not isinstance(incoming, dict) or list(incoming) != list(current):
            raise RuntimeError(f"checkpoint {name} mapping fields/order mismatch")
        for key in current:
            _validate_tree_structure(f"{name}.{key}", current[key], incoming[key])
    elif isinstance(current, (list, tuple)):
        if type(incoming) is not type(current) or len(incoming) != len(current):
            raise RuntimeError(f"checkpoint {name} sequence structure mismatch")
        for index, (expected, value) in enumerate(zip(current, incoming)):
            _validate_tree_structure(f"{name}[{index}]", expected, value)
    elif type(incoming) is not type(current):
        raise RuntimeError(f"checkpoint {name} value type mismatch")


def _validate_optimizer_state(optimizer: torch.optim.Optimizer, incoming: Any) -> None:
    """Preflight parameter-group topology and per-parameter state tensors."""
    current = optimizer.state_dict()
    if not isinstance(incoming, dict) or set(incoming) != {"state", "param_groups"}:
        raise RuntimeError("checkpoint optimizer state fields mismatch")
    groups = incoming["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(current["param_groups"]):
        raise RuntimeError("checkpoint optimizer parameter-group count mismatch")

    incoming_ids = []
    id_to_parameter = {}
    for index, (expected_group, value_group, live_group) in enumerate(
        zip(current["param_groups"], groups, optimizer.param_groups)
    ):
        if not isinstance(value_group, dict) or set(value_group) != set(expected_group):
            raise RuntimeError(f"checkpoint optimizer group {index} fields mismatch")
        value_ids = value_group["params"]
        if not isinstance(value_ids, list) or len(value_ids) != len(expected_group["params"]):
            raise RuntimeError(f"checkpoint optimizer group {index} parameter count mismatch")
        if len(live_group["params"]) != len(value_ids):
            raise RuntimeError(f"runtime optimizer group {index} parameter topology mismatch")
        for option in set(expected_group) - {"params", "lr"}:
            if value_group[option] != expected_group[option]:
                raise RuntimeError(
                    f"checkpoint optimizer group {index} option {option!r} identity mismatch"
                )
        _validate_tree_structure(
            f"optimizer.param_groups[{index}].lr", expected_group["lr"], value_group["lr"]
        )
        for serialized_id, parameter in zip(value_ids, live_group["params"]):
            if isinstance(serialized_id, bool) or not isinstance(serialized_id, int):
                raise RuntimeError("checkpoint optimizer parameter IDs must be integers")
            incoming_ids.append(serialized_id)
            id_to_parameter[serialized_id] = parameter
    if len(set(incoming_ids)) != len(incoming_ids):
        raise RuntimeError("checkpoint optimizer parameter IDs are duplicated")

    states = incoming["state"]
    if not isinstance(states, dict) or not set(states).issubset(id_to_parameter):
        raise RuntimeError("checkpoint optimizer state references unknown parameters")
    for serialized_id, parameter_state in states.items():
        if not isinstance(parameter_state, dict):
            raise RuntimeError("checkpoint optimizer per-parameter state must be a mapping")
        parameter = id_to_parameter[serialized_id]
        if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            required = {"step", "exp_avg", "exp_avg_sq"}
            group = next(
                group for group in groups if serialized_id in group["params"]
            )
            if group["amsgrad"]:
                required.add("max_exp_avg_sq")
            if set(parameter_state) != required:
                raise RuntimeError(
                    "checkpoint Adam-family per-parameter state fields mismatch"
                )
        for state_name, value in parameter_state.items():
            if torch.is_tensor(value) and value.ndim:
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise RuntimeError(
                        "checkpoint optimizer tensor state mismatch for "
                        f"parameter={serialized_id}, field={state_name!r}"
                    )


def _validate_optimizer_identity(
    optimizer: torch.optim.Optimizer, config: ResolvedConfig
) -> None:
    spec = config.data["optimizer"]
    expected_type = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }[str(spec["name"])]
    if type(optimizer) is not expected_type:
        raise RuntimeError("runtime optimizer type differs from the resolved optimizer identity")
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("S06 resolved optimizer identity requires exactly one parameter group")
    group = optimizer.param_groups[0]
    if float(group["weight_decay"]) != float(spec["weight_decay"]):
        raise RuntimeError("runtime optimizer weight_decay differs from resolved config")
    initial_lr = group.get("initial_lr", group["lr"])
    if float(initial_lr) != float(spec["learning_rate"]):
        raise RuntimeError("runtime optimizer initial learning rate differs from resolved config")


def _validate_component_state(name: str, component: Any, incoming: Any) -> None:
    if component is not None:
        _validate_tree_structure(name, component.state_dict(), incoming)


def _component_bundle(model, optimizer, scheduler, grad_scaler, ema) -> dict[str, Any]:
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "grad_scaler": grad_scaler,
        "ema": ema,
    }


def _load_component_states(bundle: dict[str, Any], raw: dict[str, Any]) -> None:
    bundle["model"].load_state_dict(raw["model"], strict=True)
    bundle["optimizer"].load_state_dict(raw["optimizer"])
    for name in ("scheduler", "grad_scaler", "ema"):
        if bundle[name] is not None:
            bundle[name].load_state_dict(raw[name])


def _cpu_snapshot(value: Any) -> Any:
    """Clone tensors directly to host memory while preserving nested state structure."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        out = copy.copy(value)
        out.clear()
        for key, item in value.items():
            out[copy.deepcopy(key)] = _cpu_snapshot(item)
        if hasattr(value, "_metadata"):
            out._metadata = _cpu_snapshot(value._metadata)
        return out
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        items = [_cpu_snapshot(item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items) if type(value) is tuple else type(value)(items)
    if isinstance(value, set):
        return type(value)(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _snapshot_component_states(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        name: None if component is None else _cpu_snapshot(component.state_dict())
        for name, component in bundle.items()
    }


def _rollback_components(
    bundle: dict[str, Any], snapshots: dict[str, Any], original_rng: dict[str, Any]
) -> None:
    errors = []
    for name in ("model", "optimizer", "scheduler", "grad_scaler", "ema"):
        component = bundle[name]
        if component is None:
            continue
        try:
            if name == "model":
                component.load_state_dict(snapshots[name], strict=True)
            else:
                component.load_state_dict(snapshots[name])
        except Exception as exc:  # pragma: no cover - catastrophic custom component.
            errors.append(f"{name}: {exc}")
    try:
        _restore_rng(original_rng)
    except Exception as exc:  # pragma: no cover - catastrophic runtime failure.
        errors.append(f"rng: {exc}")
    if errors:
        raise RuntimeError("checkpoint rollback failed: " + "; ".join(errors))


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
    _validate_optimizer_identity(optimizer, config)
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
    original_rng = copy.deepcopy(_rng_state())
    bundle = _component_bundle(model, optimizer, scheduler, grad_scaler, ema)
    try:
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
        drift = [key for key, value in expected.items() if raw[key] != value]
        if drift:
            raise RuntimeError(f"checkpoint/config/data identity drift: {drift}")
        _validate_optimizer_identity(optimizer, config)
        for name, component in (("scheduler", scheduler), ("grad_scaler", grad_scaler), ("ema", ema)):
            if (component is None) != (raw[name] is None):
                raise RuntimeError(f"checkpoint {name} presence mismatch")
        identity = raw["checkpoint_identity"]
        if (
            not isinstance(identity, str) or len(identity) != 64
            or any(char not in _HEX for char in identity)
        ):
            raise RuntimeError("checkpoint identity must be a lowercase SHA-256 string")
        state = TrainingState.from_checkpoint(raw["training_state"])
        _validate_rng(raw["rng"])
        _validate_model_state(model, raw["model"])
        _validate_optimizer_state(optimizer, raw["optimizer"])
        _validate_component_state("scheduler", scheduler, raw["scheduler"])
        _validate_component_state("grad_scaler", grad_scaler, raw["grad_scaler"])
        _validate_component_state("ema", ema, raw["ema"])
        _restore_rng(original_rng)
    except Exception:
        _restore_rng(original_rng)
        raise

    try:
        snapshots = _snapshot_component_states(bundle)
        _restore_rng(original_rng)
    except Exception:
        _restore_rng(original_rng)
        raise
    try:
        _load_component_states(bundle, raw)
        _restore_rng(raw["rng"])
    except Exception as exc:
        try:
            _rollback_components(bundle, snapshots, original_rng)
        except Exception as rollback_exc:
            raise RuntimeError("checkpoint load failed and rollback was not clean") from rollback_exc
        raise RuntimeError("checkpoint load failed; caller state rolled back") from exc
    return state, identity
