"""Fail-closed runtime optimization binding for promoted Phase-I recipes."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from fl_v3.config import ResolvedConfig
from fl_v3.config.phase1 import CAMERA_COMPILE_FORWARD_MODULES


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_name_sha256(model: torch.nn.Module) -> str:
    return _canonical_sha256(sorted(model.state_dict()))


def apply_phase1_runtime_optimizations(
    model: torch.nn.Module,
    config: ResolvedConfig,
) -> dict[str, Any]:
    """Apply the exact runtime stack encoded by a resolved production recipe.

    The operation changes only Swin attention dispatch and selected module
    ``forward`` callables. Parameter and buffer names must remain unchanged so
    optimizer grouping and checkpoint compatibility stay bound to the same graph.
    """
    if hasattr(model, "_phase1_runtime_optimization_identity"):
        raise RuntimeError("Phase-I runtime optimizations were applied more than once")
    raw = config.as_dict()
    spec = raw.get("runtime_optimizations")
    before_state_names = _state_name_sha256(model)
    if spec is None:
        record = {
            "camera_sdpa": False,
            "sdpa_modules_patched": 0,
            "torch_compile": False,
            "fused_adamw": bool(raw["optimizer"]["fused"]),
            "compiled_forward_modules": [],
            "compile_backend": None,
            "compile_dynamic": None,
            "compile_mode": None,
            "state_dict_name_sha256": before_state_names,
        }
        model._phase1_runtime_optimization_identity = record  # type: ignore[attr-defined]
        return dict(record)

    if raw["contract"]["branch"] != "camera":
        raise RuntimeError("promoted Phase-I runtime stack is Camera-only")
    from fl_v3.models.fusion.swin_sdpa import apply_sdpa_to_swin

    sdpa_modules = int(apply_sdpa_to_swin(model.camera_backbone))
    if sdpa_modules != 12:
        raise RuntimeError(
            f"Camera production SDPA patched {sdpa_modules} modules, expected 12"
        )

    compile_spec = spec["torch_compile"]
    compiled_modules: list[str] = []
    for name in compile_spec["modules"]:
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError(f"Camera production compile module {name!r} is absent")
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            backend=str(compile_spec["backend"]),
            dynamic=bool(compile_spec["dynamic"]),
            mode=str(compile_spec["mode"]),
        )
        compiled_modules.append(str(name))
    if tuple(compiled_modules) != CAMERA_COMPILE_FORWARD_MODULES:
        raise RuntimeError("Camera production compile scope drift")

    after_state_names = _state_name_sha256(model)
    if bool(spec["state_dict_names_unchanged_required"]) and (
        before_state_names != after_state_names
    ):
        raise RuntimeError(
            "Camera production runtime stack changed state-dict parameter/buffer names"
        )
    record = {
        "camera_sdpa": True,
        "sdpa_modules_patched": sdpa_modules,
        "torch_compile": True,
        "fused_adamw": bool(raw["optimizer"]["fused"]),
        "compiled_forward_modules": compiled_modules,
        "compile_backend": str(compile_spec["backend"]),
        "compile_dynamic": bool(compile_spec["dynamic"]),
        "compile_mode": str(compile_spec["mode"]),
        "state_dict_name_sha256": after_state_names,
    }
    model._phase1_runtime_optimization_identity = record  # type: ignore[attr-defined]
    return dict(record)


def phase1_runtime_optimization_identity(
    model: torch.nn.Module,
) -> dict[str, Any]:
    record = getattr(model, "_phase1_runtime_optimization_identity", None)
    if not isinstance(record, Mapping):
        raise RuntimeError("Phase-I runtime optimization identity is absent")
    return dict(record)
