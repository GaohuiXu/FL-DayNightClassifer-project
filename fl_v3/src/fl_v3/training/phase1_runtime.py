"""Fail-closed runtime optimization binding for promoted Phase-I recipes."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from fl_v3.config import ResolvedConfig
from fl_v3.config.phase1 import (
    CAMERA_COMPILE_FORWARD_MODULES,
    LIDAR_COMPILE_FORWARD_MODULES,
)


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


def _compile_forward_modules(
    model: torch.nn.Module,
    spec: Mapping[str, Any],
    expected: tuple[str, ...],
    *,
    branch: str,
) -> list[str]:
    compiled: list[str] = []
    for name in spec["modules"]:
        module = getattr(model, name, None)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError(
                f"{branch} production compile module {name!r} is absent"
            )
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            backend=str(spec["backend"]),
            dynamic=bool(spec["dynamic"]),
            mode=str(spec["mode"]),
        )
        compiled.append(str(name))
    if tuple(compiled) != expected:
        raise RuntimeError(f"{branch} production compile scope drift")
    return compiled


def apply_phase1_runtime_optimizations(
    model: torch.nn.Module,
    config: ResolvedConfig,
    *,
    criterion: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Apply the exact runtime stack encoded by a resolved production recipe.

    The operation changes only promoted output-neutral Camera/LiDAR dispatch and
    selected module ``forward`` callables. Parameter and buffer names must remain
    unchanged so optimizer grouping and checkpoint compatibility stay bound to the
    same graph.
    """
    if hasattr(model, "_phase1_runtime_optimization_identity"):
        raise RuntimeError("Phase-I runtime optimizations were applied more than once")
    raw = config.as_dict()
    spec = raw.get("runtime_optimizations")
    before_state_names = _state_name_sha256(model)
    if spec is None:
        record = {
            "camera_sdpa": False,
            "camera_preprocess": {
                "batched_affine_grid": False,
                "vectorized_geometry": False,
                "bulk_input_conversion": False,
            },
            "lidar_host_batch_offsets": False,
            "hungarian_batched_d2h": False,
            "lidar_sdpa": False,
            "lidar_sdpa_identity": None,
            "cpu_resident_batch_fields": [],
            "batch_norm": None,
            "worker_seed_formula": None,
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

    branch = str(raw["contract"]["branch"])
    compile_spec = spec["torch_compile"]
    if branch == "lidar":
        host_offsets_setter = getattr(
            model, "set_phase1p_lidar_host_batch_offsets", None
        )
        if not callable(host_offsets_setter):
            raise RuntimeError("LiDAR production host-offset setter is absent")
        hungarian_setter = getattr(
            criterion, "set_phase1p_hungarian_batched_d2h", None
        )
        if not callable(hungarian_setter):
            raise RuntimeError("LiDAR production batched-Hungarian setter is absent")
        sdpa_setter = getattr(model, "set_phase1p_lidar_sdpa", None)
        sdpa_identity_getter = getattr(model, "phase1p_lidar_sdpa_identity", None)
        if not callable(sdpa_setter) or not callable(sdpa_identity_getter):
            raise RuntimeError("LiDAR production SDPA control/identity is absent")
        host_offsets_setter(bool(spec["lidar_host_batch_offsets"]))
        hungarian_setter(bool(spec["hungarian_batched_d2h"]))
        if int(sdpa_setter(False)) != 0:
            raise RuntimeError("LiDAR production SDPA disable control drift")
        lidar_sdpa_identity = sdpa_identity_getter()
        expected_sdpa_identity = {
            "module_names": ["decoder.cross_attn", "decoder.self_attn"],
            "dropout_probabilities": [0.1, 0.1],
            "enabled": False,
            "training_rng_contract": (
                "dropout probability is unchanged; SDPA and reference kernels may "
                "consume Philox RNG differently for the same seed"
            ),
        }
        if lidar_sdpa_identity != expected_sdpa_identity:
            raise RuntimeError("LiDAR production SDPA scope/disable identity drift")
        compiled_modules = _compile_forward_modules(
            model,
            compile_spec,
            LIDAR_COMPILE_FORWARD_MODULES,
            branch="LiDAR",
        )
        after_state_names = _state_name_sha256(model)
        if bool(spec["state_dict_names_unchanged_required"]) and (
            before_state_names != after_state_names
        ):
            raise RuntimeError(
                "LiDAR production runtime stack changed state-dict parameter/buffer names"
            )
        record = {
            "camera_sdpa": False,
            "camera_preprocess": {
                "batched_affine_grid": False,
                "vectorized_geometry": False,
                "bulk_input_conversion": False,
            },
            "lidar_host_batch_offsets": True,
            "hungarian_batched_d2h": True,
            "lidar_sdpa": False,
            "lidar_sdpa_identity": dict(lidar_sdpa_identity),
            "cpu_resident_batch_fields": list(spec["cpu_resident_batch_fields"]),
            "batch_norm": str(spec["batch_norm"]),
            "worker_seed_formula": str(spec["worker_seed_formula"]),
            "sdpa_modules_patched": 0,
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
    if branch != "camera":
        raise RuntimeError(f"unknown promoted Phase-I runtime branch {branch!r}")
    from fl_v3.models.fusion.swin_sdpa import apply_sdpa_to_swin

    sdpa_modules = int(apply_sdpa_to_swin(model.camera_backbone))
    if sdpa_modules != 12:
        raise RuntimeError(
            f"Camera production SDPA patched {sdpa_modules} modules, expected 12"
        )

    preprocess_identity = {
        "batched_affine_grid": False,
        "vectorized_geometry": False,
        "bulk_input_conversion": False,
    }
    preprocess_spec = spec.get("camera_preprocess")
    if preprocess_spec is not None:
        preprocess = getattr(model, "preprocess", None)
        setters = (
            (
                "batched_affine_grid",
                getattr(preprocess, "set_phase1p_batched_affine_grid", None),
            ),
            (
                "vectorized_geometry",
                getattr(preprocess, "set_phase1p_vectorized_geometry", None),
            ),
            (
                "bulk_input_conversion",
                getattr(preprocess, "set_phase1p_bulk_input_conversion", None),
            ),
        )
        for name, setter in setters:
            if not callable(setter):
                raise RuntimeError(
                    f"Camera production preprocessing setter {name!r} is absent"
                )
            enabled = bool(preprocess_spec[name])
            setter(enabled)
            preprocess_identity[name] = enabled

    compiled_modules = _compile_forward_modules(
        model,
        compile_spec,
        CAMERA_COMPILE_FORWARD_MODULES,
        branch="Camera",
    )

    after_state_names = _state_name_sha256(model)
    if bool(spec["state_dict_names_unchanged_required"]) and (
        before_state_names != after_state_names
    ):
        raise RuntimeError(
            "Camera production runtime stack changed state-dict parameter/buffer names"
        )
    record = {
        "camera_sdpa": True,
        "camera_preprocess": preprocess_identity,
        "lidar_host_batch_offsets": False,
        "hungarian_batched_d2h": False,
        "lidar_sdpa": False,
        "lidar_sdpa_identity": None,
        "cpu_resident_batch_fields": [],
        "batch_norm": None,
        "worker_seed_formula": None,
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
