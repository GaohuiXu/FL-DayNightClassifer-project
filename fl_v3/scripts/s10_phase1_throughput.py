#!/usr/bin/env python3
"""Bounded, D_fit-only S10 Phase I-P throughput profiler.

This entry reuses the production Phase-I model, loader, loss, AdamW, cyclic
scheduler, GradScaler, training loop, and checkpoint implementation.  It has no
evaluation import or code path and cannot access D_select, D_audit, or official
validation.  Checked-in profile mappings fail closed per envelope; every
optimization remains default-off outside its explicit Phase I-P profile.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Mapping

sys.path.insert(0, "fl_v3/src")

import numpy as np
import torch

from fl_v3.config import load_resolved_config
from fl_v3.config.phase1 import phase1_runtime_ready
from fl_v3.data.nuscenes.phase1 import build_phase1_train_data
from fl_v3.models.phase1_swin import sha256_file, tensor_state_sha256
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.phase1 import build_phase1_training_stack
from fl_v3.training.phase1_checkpoint_gate import evaluate_calibrated_continuation_gate
from fl_v3.training.phase1_profile import (
    derive_profile_runtime_config,
    load_phase1_profile_spec,
)
from fl_v3.training.runtime_state import TrainingState
from fl_v3.training.s10_observation import compare_tensor_tree_tensors
from fl_v3.utils.runtime import (
    enforce_determinism,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "s10.phase1p.profiler-result.v2"
EXPECTED_BRANCH = "codex/s10-phase1p-throughput-preflight"
EXPECTED_BASE_SHA = "f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
FROZEN_CONTROL_REF = "refs/heads/codex/s10-phase1-branch-qualification"
_ATTEMPT_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _candidate_cpu_resident_batch_fields(profile, branch: str) -> tuple[str, ...]:
    profile.assert_runnable(branch)
    if bool(profile.candidates["camera_augmentation_transfer_cleanup"]):
        _require(branch == "camera", "augmentation transfer cleanup is Camera-only")
        return ("augmentation_params",)
    return ()


_COMPILE_MODULES = (
    "camera_backbone",
    "camera_neck",
    "decoder_backbone",
    "decoder_neck",
    "head",
)


def _state_name_sha256(model: torch.nn.Module) -> str:
    return _canonical_sha256(sorted(model.state_dict()))


def _dynamo_counters() -> dict[str, dict[str, int]]:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return {}
    result: dict[str, dict[str, int]] = {}
    for category, values in counters.items():
        normalized = {
            str(key): int(value)
            for key, value in values.items()
            if isinstance(value, (int, np.integer))
        }
        if normalized:
            result[str(category)] = normalized
    return result


def _counter_delta(
    before: Mapping[str, Mapping[str, int]],
    after: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for category in sorted(set(before) | set(after)):
        keys = set(before.get(category, {})) | set(after.get(category, {}))
        values = {
            key: int(after.get(category, {}).get(key, 0))
            - int(before.get(category, {}).get(key, 0))
            for key in sorted(keys)
        }
        values = {key: value for key, value in values.items() if value}
        if values:
            result[category] = values
    return result


def _counter_has_steady_recompile(
    delta: Mapping[str, Mapping[str, int]],
) -> bool:
    """Conservatively identify new compiler graphs/cache misses after warm-up."""
    signals = ("unique_graph", "recompil", "cache_miss", "fxgraph_cache_miss")
    return any(
        int(value) > 0 and any(signal in str(key).lower() for signal in signals)
        for values in delta.values()
        for key, value in values.items()
    )


def _optimizer_configuration(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    defaults = optimizer.defaults
    return {
        "type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "parameter_groups": len(optimizer.param_groups),
        "state_entries": len(optimizer.state),
        "fused": defaults.get("fused"),
        "foreach": defaults.get("foreach"),
        "capturable": defaults.get("capturable"),
        "differentiable": defaults.get("differentiable"),
    }


def _configure_profile_candidate(model, profile, branch: str) -> dict[str, Any]:
    """Apply only the fail-closed profiler runtime options bound by the profile."""
    profile.assert_runnable(branch)
    augmentation_cleanup = bool(
        profile.candidates["camera_augmentation_transfer_cleanup"]
    )
    static_grid_cache = bool(profile.candidates["camera_static_grid_cache"])
    batched_affine_grid = bool(profile.candidates["camera_batched_affine_grid"])
    batched_preprocess = bool(profile.candidates["camera_batched_preprocess"])
    vectorized_geometry = bool(
        profile.candidates.get("camera_vectorized_geometry", False)
    )
    bulk_input_conversion = bool(
        profile.candidates.get("camera_bulk_input_conversion", False)
    )
    if branch == "camera":
        augmentation_setter = getattr(
            getattr(model, "preprocess", None),
            "set_phase1p_augmentation_transfer_cleanup",
            None,
        )
        grid_setter = getattr(
            getattr(model, "preprocess", None),
            "set_phase1p_static_grid_cache",
            None,
        )
        batched_grid_setter = getattr(
            getattr(model, "preprocess", None),
            "set_phase1p_batched_affine_grid",
            None,
        )
        batched_preprocess_setter = getattr(
            getattr(model, "preprocess", None),
            "set_phase1p_batched_preprocess",
            None,
        )
        vectorized_geometry_setter = getattr(
            getattr(model, "preprocess", None),
            "set_phase1p_vectorized_geometry",
            None,
        )
        bulk_input_conversion_setter = getattr(
            getattr(model, "preprocess", None),
            "set_phase1p_bulk_input_conversion",
            None,
        )
        _require(
            callable(augmentation_setter)
            and callable(grid_setter)
            and callable(batched_grid_setter)
            and callable(batched_preprocess_setter)
            and (not vectorized_geometry or callable(vectorized_geometry_setter))
            and (
                not bulk_input_conversion
                or callable(bulk_input_conversion_setter)
            ),
            "Camera preprocessor lacks Phase I-P candidate controls",
        )
        augmentation_setter(augmentation_cleanup)
        grid_setter(static_grid_cache)
        batched_grid_setter(batched_affine_grid)
        batched_preprocess_setter(batched_preprocess)
        if callable(vectorized_geometry_setter):
            vectorized_geometry_setter(vectorized_geometry)
        if callable(bulk_input_conversion_setter):
            bulk_input_conversion_setter(bulk_input_conversion)
    else:
        _require(
            not augmentation_cleanup
            and not static_grid_cache
            and not batched_affine_grid
            and not batched_preprocess
            and not vectorized_geometry
            and not bulk_input_conversion,
            "LiDAR cannot enable Camera profiler candidates",
        )
    camera_sdpa = bool(profile.candidates["camera_sdpa"])
    lidar_sdpa = bool(profile.candidates["lidar_sdpa"])
    compile_enabled = bool(profile.candidates["torch_compile"])
    _require(not lidar_sdpa, "Phase I-P does not authorize LiDAR SDPA")
    _require(not compile_enabled or branch == "camera", "compile candidate is Camera-only")
    before_state_names = _state_name_sha256(model)
    from fl_v3.training.phase1_runtime import phase1_runtime_optimization_identity

    source_runtime = phase1_runtime_optimization_identity(model)
    source_runtime_active = bool(
        source_runtime["camera_sdpa"] or source_runtime["torch_compile"]
    )
    sdpa_modules = 0
    compiled_modules: list[str] = []
    if source_runtime_active:
        _require(branch == "camera", "preconfigured runtime stack is not Camera-only")
        _require(
            bool(source_runtime["camera_sdpa"]) == camera_sdpa
            and bool(source_runtime["torch_compile"]) == compile_enabled,
            "profile runtime flags differ from the production-config runtime stack",
        )
        _require(
            bool(source_runtime["fused_adamw"])
            is bool(profile.candidates["fused_adamw"]),
            "profile fused AdamW flag differs from the production config",
        )
        sdpa_modules = int(source_runtime["sdpa_modules_patched"])
        compiled_modules = list(source_runtime["compiled_forward_modules"])
        _require(sdpa_modules == 12, "production Camera SDPA module count drift")
        _require(
            tuple(compiled_modules) == _COMPILE_MODULES,
            "production Camera compile scope drift",
        )
        _require(
            source_runtime["compile_backend"] == "inductor"
            and source_runtime["compile_dynamic"] is False
            and source_runtime["compile_mode"] == "default",
            "production Camera compile policy drift",
        )
        if profile.data["envelope"] == "IP-E5":
            _require(
                source_runtime["camera_preprocess"]
                == {
                    "batched_affine_grid": batched_affine_grid,
                    "vectorized_geometry": vectorized_geometry,
                    "bulk_input_conversion": bulk_input_conversion,
                },
                "IP-E5 profile differs from the production Camera preprocessing stack",
            )
        runtime_application = "production_config"
    else:
        if camera_sdpa:
            _require(branch == "camera", "Camera SDPA cannot run on the LiDAR branch")
            from fl_v3.models.fusion.swin_sdpa import apply_sdpa_to_swin

            sdpa_modules = int(apply_sdpa_to_swin(model.camera_backbone))
            _require(
                sdpa_modules == 12,
                f"Camera SDPA patched {sdpa_modules} modules, expected 12",
            )
        if compile_enabled:
            for name in _COMPILE_MODULES:
                module = getattr(model, name, None)
                _require(
                    isinstance(module, torch.nn.Module),
                    f"compile module {name!r} is absent",
                )
                module.forward = torch.compile(  # type: ignore[method-assign]
                    module.forward,
                    backend="inductor",
                    dynamic=False,
                    mode="default",
                )
                compiled_modules.append(name)
        runtime_application = "profile_candidate"
    after_state_names = _state_name_sha256(model)
    _require(
        before_state_names == after_state_names,
        "profile candidate changed model state-dict parameter/buffer names",
    )
    return {
        "camera_sdpa": camera_sdpa,
        "sdpa_modules_patched": sdpa_modules,
        "torch_compile": compile_enabled,
        "compiled_forward_modules": compiled_modules,
        "compile_backend": "inductor" if compile_enabled else None,
        "compile_dynamic": False if compile_enabled else None,
        "compile_mode": "default" if compile_enabled else None,
        "runtime_application": runtime_application,
        "camera_batched_affine_grid": batched_affine_grid,
        "camera_vectorized_geometry": vectorized_geometry,
        "camera_bulk_input_conversion": bulk_input_conversion,
        "state_dict_name_sha256": after_state_names,
    }


def _batch_sha256(value: Any) -> str:
    """Hash one CPU loader batch without changing its tensors or container order."""
    digest = hashlib.sha256()

    def part(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: Any, path: str) -> None:
        part(path.encode("utf-8"))
        part(type(item).__name__.encode("utf-8"))
        if torch.is_tensor(item):
            tensor = item.detach().resolve_conj().resolve_neg().cpu().contiguous()
            part(str(tensor.dtype).encode("ascii"))
            part(_canonical_bytes([int(size) for size in tensor.shape]))
            try:
                payload = tensor.numpy().tobytes(order="C")
            except TypeError:  # numpy has no native bfloat16 representation.
                payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
            part(payload)
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            part(str(array.dtype).encode("ascii"))
            part(_canonical_bytes([int(size) for size in array.shape]))
            part(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            part(_canonical_bytes([f"{type(key).__name__}:{key!r}" for key in keys]))
            for key in keys:
                visit(item[key], f"{path}.{type(key).__name__}:{key!r}")
            return
        if isinstance(item, (list, tuple)):
            part(str(len(item)).encode("ascii"))
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if item is None or isinstance(item, (str, bool, int, float)):
            part(repr(item).encode("utf-8"))
            return
        raise TypeError(f"unsupported loader-batch leaf {type(item)!r} at {path}")

    visit(value, "root")
    return digest.hexdigest()


class _BatchDigestLoader:
    """Observational loader proxy used only by checkpoint-continuation diagnostics."""

    def __init__(self, loader, records: list[str], *, limit: int | None = None) -> None:
        self.loader = loader
        self.records = records
        self.limit = None if limit is None else int(limit)
        self.batch_size = getattr(loader, "batch_size", None)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if self.limit is None or len(self.records) < self.limit:
                self.records.append(_batch_sha256(batch))
            yield batch


def _atomic_write_once(path: Path, value: Any) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable profiler artifact {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale profiler partial exists: {partial}")
    payload = _canonical_bytes(value) + b"\n"
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return sha256_file(path)


def _atomic_write_bytes_once(path: Path, payload: bytes) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable profiler artifact {path}")
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"stale profiler partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return sha256_file(path)


def _torch_save_once(path: Path, value: Any) -> str:
    """Write a profiler-only tensor payload without overwrite."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    _require(isinstance(value, dict), f"expected JSON object at {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity(source_sha: str, approved_source_sha: str) -> dict[str, Any]:
    actual = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    branch = _git("branch", "--show-current")
    dirty = _git("status", "--porcelain", "--untracked-files=all")
    _require(actual == source_sha, f"source SHA drift: {actual} != {source_sha}")
    _require(branch == EXPECTED_BRANCH, f"source branch drift: {branch!r}")
    _require(not dirty, "Phase I-P execution requires a clean source worktree")
    control = _git("rev-parse", FROZEN_CONTROL_REF)
    _require(control == EXPECTED_BASE_SHA, "frozen Phase-I control branch moved")
    base_ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE_SHA, approved_source_sha],
        check=False,
    ).returncode
    _require(base_ancestry == 0, "approved source is not descended from the unique IP-G0 base")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", approved_source_sha, source_sha],
        check=False,
    ).returncode
    _require(
        ancestry == 0,
        "runtime source is not the approved source or an O-149 linear descendant",
    )
    merge_commits = _git(
        "rev-list", "--min-parents=2", f"{EXPECTED_BASE_SHA}..{source_sha}"
    )
    _require(not merge_commits, "Phase I-P source history is not linear from the unique base")
    return {
        "git_sha": actual,
        "git_tree": tree,
        "branch": branch,
        "unique_base_sha": EXPECTED_BASE_SHA,
        "frozen_control_ref": FROZEN_CONTROL_REF,
        "frozen_control_sha": control,
        "approved_source_sha": approved_source_sha,
        "derived_source": actual != approved_source_sha,
    }


def _runtime_identity(config) -> tuple[dict[str, Any], str]:
    _require(platform.machine() == "aarch64", "Phase I-P requires an aarch64 GH200 node")
    _require(torch.cuda.is_available(), "Phase I-P requires CUDA")
    _require(torch.cuda.device_count() == 1, "Phase I-P requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    _require("GH200" in name, f"Phase I-P requires GH200, got {name!r}")
    _require(capability == (9, 0), f"unexpected GH200 compute capability {capability}")
    dependencies = verify_runtime_dependency_identity(config.to_run_config())
    dependency_sha = _canonical_sha256(dependencies)
    compact = {
        key: value
        for key, value in dependencies.items()
        if not key.endswith("_executable_artifacts")
        and not key.endswith("_import_origins")
    }
    properties = torch.cuda.get_device_properties(device)
    return (
        {
            "device_name": name,
            "compute_capability": list(capability),
            "total_memory_bytes": int(properties.total_memory),
            "torch_cuda": str(torch.version.cuda),
            "dependencies": compact,
            "dependencies_sha256": dependency_sha,
        },
        dependency_sha,
    )


def _attempt_identity() -> dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node_mib": os.environ.get("SLURM_MEM_PER_NODE"),
        "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "pid": os.getpid(),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_capture(value: Any) -> tuple[dict[str, torch.Tensor], str]:
    """Capture tensor leaves plus a canonical non-value structural identity."""
    tensors: dict[str, torch.Tensor] = {}

    def visit(item: Any, path: str) -> Any:
        if torch.is_tensor(item):
            tensors[path] = item.detach().cpu().clone()
            return {
                "tensor": True,
                "dtype": str(item.dtype),
                "shape": [int(size) for size in item.shape],
                "layout": str(item.layout),
            }
        if isinstance(item, Mapping):
            children = []
            for key in sorted(item, key=lambda child: (type(child).__name__, repr(child))):
                label = f"{type(key).__name__}:{key!r}"
                children.append([label, visit(item[key], f"{path}.{label}")])
            return {"mapping": type(item).__name__, "children": children}
        if isinstance(item, (list, tuple)):
            return {
                "sequence": type(item).__name__,
                "children": [visit(child, f"{path}[{index}]") for index, child in enumerate(item)],
            }
        if item is None or isinstance(item, (str, bool, int, float)):
            return {"scalar_type": type(item).__name__, "value": item}
        raise TypeError(f"unsupported checkpoint state leaf {type(item)!r} at {path}")

    structure = visit(value, "root")
    return tensors, _canonical_sha256(structure)


def _compare_state_captures(
    reference: tuple[dict[str, torch.Tensor], str],
    candidate: tuple[dict[str, torch.Tensor], str],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference_tensors, reference_structure = reference
    candidate_tensors, candidate_structure = candidate
    numerical = compare_tensor_tree_tensors(reference_tensors, candidate_tensors)
    allclose_failures = []
    exact_failures = []
    floating_tensor_names = []
    discrete_tensor_names = []
    for name in sorted(set(reference_tensors) & set(candidate_tensors)):
        left = reference_tensors[name]
        right = candidate_tensors[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            continue
        if left.is_floating_point() or left.is_complex():
            floating_tensor_names.append(name)
            if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
                allclose_failures.append(name)
        else:
            discrete_tensor_names.append(name)
            if not torch.equal(left, right):
                exact_failures.append(name)
    gate = (
        reference_structure == candidate_structure
        and numerical["name_set_equal"]
        and not numerical["shape_mismatch_tensors"]
        and not numerical["dtype_mismatch_tensors"]
        and numerical["global"]["all_finite"]
        and not allclose_failures
        and not exact_failures
    )
    return {
        "reference_structure_sha256": reference_structure,
        "candidate_structure_sha256": candidate_structure,
        "structure_equal": reference_structure == candidate_structure,
        "allclose_rtol": float(rtol),
        "allclose_atol": float(atol),
        "floating_allclose_failures": allclose_failures,
        "discrete_exact_failures": exact_failures,
        "floating_tensor_names": floating_tensor_names,
        "discrete_tensor_names": discrete_tensor_names,
        "numerical": numerical,
        "gate_pass": gate,
    }


def _rng_sha256() -> str:
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    payload = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().cpu().numpy().tobytes(),
        [state.cpu().numpy().tobytes() for state in cuda],
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def _sampler_prefix_identity(bundle, epoch: int, presentations: int) -> dict[str, Any]:
    _require(0 <= presentations <= len(bundle.sampler), "sampler prefix length is invalid")
    bundle.set_epoch(epoch)
    positions = list(iter(bundle.sampler))[:presentations]
    expanded = bundle.dataset.indices[np.asarray(positions, dtype=np.int64)]
    base_tokens = bundle.base_dataset.sample_tokens
    tokens = [base_tokens[int(index)] for index in expanded]
    return {
        "epoch": int(epoch),
        "presentations": len(tokens),
        "sample_tokens_sha256": _canonical_sha256(tokens),
        "first_sample_token": tokens[0] if tokens else None,
        "last_sample_token": tokens[-1] if tokens else None,
    }


class _SystemSampler:
    """One-second nvidia-smi sampling kept outside the training process."""

    def __init__(self, path: Path, interval_seconds: float) -> None:
        self.path = path
        self.interval_seconds = float(interval_seconds)
        self.stream = None
        self.process = None

    def start(self) -> None:
        self.stream = self.path.open("xb")
        query = ",".join((
            "timestamp",
            "index",
            "name",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "power.draw",
            "clocks.sm",
            "clocks.mem",
        ))
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={int(self.interval_seconds * 1000.0)}",
            ],
            stdout=self.stream,
            stderr=subprocess.STDOUT,
        )

    def stop(self) -> dict[str, Any]:
        _require(
            self.process is not None and self.stream is not None,
            "system sampler was not started",
        )
        self.process.terminate()
        try:
            return_code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return_code = self.process.wait(timeout=10)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        _require(return_code in {-15, 0}, f"nvidia-smi sampler failed with {return_code}")
        _require(self.path.stat().st_size > 0, "nvidia-smi sampler produced no evidence")
        lines = sum(1 for line in self.path.open("rb") if line.strip())
        return {
            "path": str(self.path),
            "sha256": sha256_file(self.path),
            "bytes": self.path.stat().st_size,
            "samples": lines,
            "interval_seconds": self.interval_seconds,
        }


_CAMERA_FORWARD_TRACE_RANGES = (
    "fl_v3::camera::preprocess",
    "fl_v3::camera::swin_backbone",
    "fl_v3::camera::camera_neck",
    "fl_v3::camera::view_transform_and_pool",
    "fl_v3::camera::decoder_backbone",
    "fl_v3::camera::decoder_neck",
    "fl_v3::camera::head",
)
_CAMERA_PREPROCESS_TRACE_RANGES = tuple(
    f"fl_v3::camera_preprocess::{name}"
    for name in (
        "parameter_prepare",
        "convert_resize",
        "crop_pad_flip",
        "geometry",
        "rotation_grid_sample",
        "stack_normalize",
        "calibration_update",
    )
)
_TRAIN_TRACE_RANGES = tuple(
    f"fl_v3::train::{name}"
    for name in ("h2d", "forward", "loss", "backward", "optimizer")
)


def _profile_event_number(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _profile_event_rows(events) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        rows.append(
            {
                "key": str(event.key),
                "count": int(event.count),
                "self_cpu_time_total_us": _profile_event_number(
                    event, "self_cpu_time_total"
                ),
                "cpu_time_total_us": _profile_event_number(event, "cpu_time_total"),
                "self_device_time_total_us": _profile_event_number(
                    event, "self_device_time_total", "self_cuda_time_total"
                ),
                "device_time_total_us": _profile_event_number(
                    event, "device_time_total", "cuda_time_total"
                ),
                "self_cpu_memory_usage_bytes": int(
                    getattr(event, "self_cpu_memory_usage", 0)
                ),
                "cpu_memory_usage_bytes": int(
                    getattr(event, "cpu_memory_usage", 0)
                ),
                "self_device_memory_usage_bytes": int(
                    getattr(event, "self_device_memory_usage", 0)
                ),
                "device_memory_usage_bytes": int(
                    getattr(event, "device_memory_usage", 0)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["self_device_time_total_us"],
            -row["self_cpu_time_total_us"],
            row["key"],
        )
    )
    return rows


def _camera_trace_diagnosis(range_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, dict[str, Any]] = {}
    for row in range_rows:
        key = str(row["key"])
        current = by_key.get(key)
        if current is None or float(row["cpu_time_total_us"]) > float(
            current["cpu_time_total_us"]
        ):
            by_key[key] = row
    expected = (*_TRAIN_TRACE_RANGES, *_CAMERA_FORWARD_TRACE_RANGES)
    missing = sorted(set(expected) - set(by_key))
    camera_cpu = {
        key: float(by_key[key]["cpu_time_total_us"])
        for key in _CAMERA_FORWARD_TRACE_RANGES
        if key in by_key
    }
    camera_total = sum(camera_cpu.values())
    largest = max(camera_cpu, key=camera_cpu.get) if camera_cpu else None
    preprocess = "fl_v3::camera::preprocess"
    preprocess_subranges = {
        key: float(by_key[key]["cpu_time_total_us"])
        for key in _CAMERA_PREPROCESS_TRACE_RANGES
        if key in by_key
    }
    missing_preprocess_subranges = sorted(
        set(_CAMERA_PREPROCESS_TRACE_RANGES) - set(by_key)
    )
    largest_preprocess_subrange = (
        max(preprocess_subranges, key=preprocess_subranges.get)
        if preprocess_subranges
        else None
    )
    return {
        "expected_core_range_keys": list(expected),
        "missing_core_range_keys": missing,
        "camera_forward_cpu_time_total_us": camera_cpu,
        "camera_forward_named_range_sum_cpu_time_us": camera_total,
        "largest_camera_forward_range": largest,
        "preprocess_fraction_of_camera_forward_named_range_sum": (
            None if camera_total <= 0.0 else camera_cpu.get(preprocess, 0.0) / camera_total
        ),
        "preprocess_is_largest_camera_forward_range": largest == preprocess,
        "expected_preprocess_subrange_keys": list(_CAMERA_PREPROCESS_TRACE_RANGES),
        "missing_preprocess_subrange_keys": missing_preprocess_subranges,
        "preprocess_subrange_cpu_time_total_us": preprocess_subranges,
        "largest_preprocess_subrange": largest_preprocess_subrange,
        "interpretation": (
            "CPU range totals are trace-inflated localization evidence; use them to "
            "rank named stages, not as sustained wall-time estimates"
        ),
    }


class _TraceController:
    """Start after accepted warm-up and stop after the requested accepted windows."""

    def __init__(
        self,
        state: TrainingState,
        *,
        branch: str,
        warmup: int,
        active: int,
    ) -> None:
        self.state = state
        self.branch = str(branch)
        _require(self.branch in {"camera", "lidar"}, "trace branch identity drift")
        self.warmup = int(warmup)
        self.target = self.warmup + int(active)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_modules=True,
        )
        self.started = False
        self.stopped = False
        self.step_calls = 0

    def step(self) -> None:
        successful = int(self.state.successful_windows)
        if not self.started and successful >= self.warmup:
            self.profiler.start()
            self.started = True
            return
        if self.started and not self.stopped:
            self.profiler.step()
            self.step_calls += 1
            if successful >= self.target:
                self.profiler.stop()
                self.stopped = True

    def close(self) -> None:
        if self.started and not self.stopped:
            self.profiler.stop()
            self.stopped = True

    def publish(self, root: Path) -> dict[str, Any]:
        _require(self.started and self.stopped, "bounded torch trace did not reach its target")
        trace = root / "torch_trace.json"
        summary = root / "torch_trace_summary.txt"
        structured_summary = root / "torch_trace_summary.json"
        self.profiler.export_chrome_trace(str(trace))
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        averages = self.profiler.key_averages()
        table = averages.table(sort_by=sort_key, row_limit=200)
        with summary.open("x", encoding="utf-8") as stream:
            stream.write(table)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        rows = _profile_event_rows(averages)
        range_rows = [row for row in rows if row["key"].startswith("fl_v3::")]
        operator_rows = [row for row in rows if not row["key"].startswith("fl_v3::")]
        diagnosis = (
            _camera_trace_diagnosis(range_rows)
            if self.branch == "camera"
            else None
        )
        _require(
            diagnosis is None or not diagnosis["missing_core_range_keys"],
            "Camera trace omitted required core ranges: "
            f"{None if diagnosis is None else diagnosis['missing_core_range_keys']}",
        )
        structured_sha = _atomic_write_once(
            structured_summary,
            {
                "schema": "s10.phase1p.torch-trace-summary.v1",
                "units": {"time": "microseconds", "memory": "bytes"},
                "all_row_count": len(rows),
                "range_row_count": len(range_rows),
                "operator_row_count": len(operator_rows),
                "range_rows": range_rows,
                "operator_rows": operator_rows[:200],
                "camera_stage_diagnosis": diagnosis,
            },
        )
        return {
            "schema": "s10.phase1p.torch-trace.v2",
            "accepted_warmup_windows": self.warmup,
            "accepted_active_windows": self.target - self.warmup,
            "attempted_active_step_calls": self.step_calls,
            "trace": {
                "path": str(trace),
                "sha256": sha256_file(trace),
                "bytes": trace.stat().st_size,
            },
            "summary": {
                "path": str(summary),
                "sha256": sha256_file(summary),
                "bytes": summary.stat().st_size,
            },
            "structured_summary": {
                "path": str(structured_summary),
                "sha256": structured_sha,
                "bytes": structured_summary.stat().st_size,
                "camera_stage_diagnosis": diagnosis,
            },
        }


def _run_segment(
    *,
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    bundle,
    state: TrainingState,
    config,
    device: torch.device,
    target_optimizer_step: int,
    readiness: bool,
    warmup: int = 0,
    trace_controller: _TraceController | None = None,
    batch_digests: list[str] | None = None,
    batch_digest_limit: int | None = None,
    cpu_resident_batch_fields: tuple[str, ...] = (),
    attempted_window_callback=None,
) -> dict[str, Any]:
    raw = config.as_dict()
    loader = (
        bundle.loader
        if batch_digests is None
        else _BatchDigestLoader(
            bundle.loader, batch_digests, limit=batch_digest_limit
        )
    )
    with ExitStack() as ranges:
        if trace_controller is not None:
            ranges.enter_context(model.operator_profile_ranges())
            if hasattr(criterion, "operator_profile_ranges"):
                ranges.enter_context(criterion.operator_profile_ranges())
        return train_one_epoch(
            model,
            loader,
            criterion,
            optimizer,
            device,
            grad_clip_norm=float(raw["training"]["gradient_clip"]["max_norm"]),
            scheduler=scheduler,
            ema_model=None,
            max_steps=0,
            precision=str(raw["precision"]["global_autocast"]),
            grad_scaler=scaler,
            telemetry_interval=0,
            accumulation_steps=int(raw["training"]["accumulation_steps"]),
            runtime_state=state,
            max_optimizer_steps=int(target_optimizer_step),
            model_mode=config.model_mode,
            exposure_multiplier=1,
            expected_global_microbatch_samples=int(raw["training"]["micro_batch_size"]),
            precision_diagnostics=None,
            readiness_timing=readiness,
            readiness_warmup_successful_windows=int(warmup),
            readiness_stage_timing=False if readiness else True,
            readiness_profiler_ranges=trace_controller is not None,
            attempted_window_callback=(
                attempted_window_callback
                if attempted_window_callback is not None
                else None if trace_controller is None else trace_controller.step
            ),
            cpu_resident_batch_fields=cpu_resident_batch_fields,
        )


def _checkpoint_resume_worker(request_path: Path) -> dict[str, Any]:
    """Reload and continue in a fresh Python process, then publish compact parity."""
    request = _read_json(request_path.resolve())
    expected = {
        "schema", "config_path", "config_sha256", "checkpoint", "checkpoint_sha256",
        "reference", "reference_sha256", "result", "source_sha", "continuation_windows",
        "rtol", "atol", "profile_path", "profile_sha256", "candidate_id",
    }
    _require(set(request) == expected, "fresh-process resume request fields drift")
    _require(request["schema"] == "s10.phase1p.resume-worker-request.v3", "resume schema drift")
    checkpoint = Path(request["checkpoint"]).resolve()
    reference_path = Path(request["reference"]).resolve()
    result_path = Path(request["result"]).resolve()
    _require(
        checkpoint.parent
        == request_path.parent
        == reference_path.parent
        == result_path.parent,
        "resume-worker artifact root drift",
    )
    _require(
        sha256_file(checkpoint) == request["checkpoint_sha256"],
        "worker checkpoint hash drift",
    )
    _require(
        sha256_file(reference_path) == request["reference_sha256"],
        "worker reference hash drift",
    )
    _source_identity(request["source_sha"], request["source_sha"])
    _require(platform.machine() == "aarch64", "resume worker requires an aarch64 GH200 node")
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
             "resume worker requires exactly one visible CUDA device")

    source_config = load_resolved_config(request["config_path"])
    profile = load_phase1_profile_spec(request["profile_path"])
    _require(profile.sha256 == request["profile_sha256"], "resume-worker profile identity drift")
    _require(
        profile.data["candidate_id"] == request["candidate_id"],
        "resume-worker candidate identity drift",
    )
    branch = str(source_config.as_dict()["contract"]["branch"])
    profile.assert_branch_binding(branch, request["config_path"], source_config)
    profile.assert_runnable(branch)
    config = derive_profile_runtime_config(source_config, profile)
    _require(config.sha256 == request["config_sha256"], "resume-worker config identity drift")
    phase1_runtime_ready(config.as_dict())
    _require(request["continuation_windows"] == 8, "resume-worker window count drift")
    precision = str(config.as_dict()["precision"]["global_autocast"])
    expected_tolerance = {
        "fp32": {"rtol": 1e-4, "atol": 1e-6},
        "fp16": {"rtol": 2e-3, "atol": 2e-4},
    }[precision]
    _require(
        float(request["rtol"]) == expected_tolerance["rtol"]
        and float(request["atol"]) == expected_tolerance["atol"],
        "resume-worker parity tolerance drift",
    )
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    _require(reference["schema"] == "s10.phase1p.continuation-reference.v2",
             "continuation reference schema drift")
    _require(reference["config_sha256"] == config.sha256, "continuation reference config drift")
    _require(
        reference["profile_sha256"] == profile.sha256
        and reference["candidate_id"] == profile.data["candidate_id"],
        "continuation reference candidate drift",
    )

    device = torch.device("cuda", 0)
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(int(config.as_dict()["training"]["seed"]))
    compile_cache = request_path.parent / "torchinductor_cache_resume"
    if bool(profile.candidates["torch_compile"]):
        _require(not compile_cache.exists(), "fresh resume compile cache already exists")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
    started = time.perf_counter()
    model, criterion, optimizer, scheduler, scaler = build_phase1_training_stack(
        config, device
    )
    optimizer_configuration = _optimizer_configuration(optimizer)
    _require(
        optimizer_configuration["fused"]
        is bool(profile.candidates["fused_adamw"]),
        "resume-worker AdamW fused backend differs from the profile",
    )
    candidate_configuration = _configure_profile_candidate(model, profile, branch)
    cpu_resident_batch_fields = _candidate_cpu_resident_batch_fields(profile, branch)
    model_build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    bundle = build_phase1_train_data(config)
    loader_build_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        state, identity = load_checkpoint(
            str(checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=scaler,
            ema=None,
            config=config,
            map_location="cpu",
        )
        _sync(device)
        load_seconds = time.perf_counter() - started
        _require(identity == config.sha256, "profiler checkpoint identity drift")
        _require(
            state.checkpoint_dict() == reference["boundary_training_state"],
            "checkpoint training state drift",
        )
        _require(
            tensor_state_sha256(model.state_dict()) == reference["boundary_model_sha256"],
            "checkpoint reload changed the boundary model",
        )
        live = {
            "model": _state_capture(model.state_dict()),
            "optimizer": _state_capture(optimizer.state_dict()),
            "scheduler": _state_capture(scheduler.state_dict()),
            "scaler": _state_capture(scaler.state_dict()),
        }
        restored = {
            name: _compare_state_captures(
                reference["boundary"][name], live[name], rtol=0.0, atol=0.0
            )
            for name in live
        }
        _require(
            all(item["gate_pass"] for item in restored.values()),
            "fresh-process checkpoint reload is not exact",
        )

        target = int(reference["control_target_optimizer_step"])
        _require(
            target == state.optimizer_step + int(request["continuation_windows"]),
            "resume-worker continuation target drift",
        )
        bundle.set_epoch(state.epoch)
        resumed_batch_sha256: list[str] = []
        _run_segment(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            state=state,
            config=config,
            device=device,
            target_optimizer_step=target,
            readiness=False,
            batch_digests=resumed_batch_sha256,
            cpu_resident_batch_fields=cpu_resident_batch_fields,
        )
        _require(
            len(resumed_batch_sha256)
            == int(request["continuation_windows"])
            * int(config.as_dict()["training"]["accumulation_steps"]),
            "resume-worker input-digest count drift",
        )
        resumed = {
            "model": _state_capture(model.state_dict()),
            "optimizer": _state_capture(optimizer.state_dict()),
            "scheduler": _state_capture(scheduler.state_dict()),
            "scaler": _state_capture(scaler.state_dict()),
            "training_state": state.checkpoint_dict(),
            "rng_sha256": _rng_sha256(),
        }
        continuation = {
            name: _compare_state_captures(
                reference["control"][name],
                resumed[name],
                rtol=float(request["rtol"]),
                atol=float(request["atol"]),
            )
            for name in ("model", "optimizer", "scheduler", "scaler")
        }
        state_equal = reference["control"]["training_state"] == resumed["training_state"]
        rng_equal = reference["control"]["rng_sha256"] == resumed["rng_sha256"]
        input_equal = reference["control_batch_sha256"] == resumed_batch_sha256
        elementwise_diagnostic = (
            all(item["gate_pass"] for item in continuation.values())
            and state_equal
            and rng_equal
            and input_equal
        )
        result = {
            "schema": "s10.phase1p.resume-worker-result.v2",
            "fresh_process_pid": os.getpid(),
            "model_stack_build_seconds": model_build_seconds,
            "D_fit_loader_build_seconds": loader_build_seconds,
            "checkpoint_load_seconds": load_seconds,
            "candidate_configuration": candidate_configuration,
            "optimizer_configuration": optimizer_configuration,
            "compile_cache": str(compile_cache) if compile_cache.exists() else None,
            "restored_boundary": restored,
            "continuation": continuation,
            "input_stream": {
                "microbatches": len(resumed_batch_sha256),
                "reference_sha256": reference["control_batch_sha256"],
                "resumed_sha256": resumed_batch_sha256,
                "exact_equal": input_equal,
            },
            "training_state_equal": state_equal,
            "rng_state_equal": rng_equal,
            "elementwise_allclose_diagnostic_pass": elementwise_diagnostic,
        }
        _atomic_write_once(result_path, result)
        return result
    finally:
        bundle.close()


def _checkpoint_and_continuation(
    *,
    output_dir: Path,
    owned_stack: dict[str, Any],
    state: TrainingState,
    config,
    config_path: Path,
    profile,
    profile_path: Path,
    source_sha: str,
    device: torch.device,
    continuation_windows: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    branch = str(config.as_dict()["contract"]["branch"])
    cpu_resident_batch_fields = _candidate_cpu_resident_batch_fields(profile, branch)
    # Ownership is transferred from the caller so the original model, optimizer
    # state and loader can be genuinely released before the resume worker starts.
    expected_stack = {"model", "criterion", "optimizer", "scheduler", "scaler", "bundle"}
    _require(set(owned_stack) == expected_stack, "checkpoint stack ownership drift")
    model = owned_stack.pop("model")
    criterion = owned_stack.pop("criterion")
    optimizer = owned_stack.pop("optimizer")
    scheduler = owned_stack.pop("scheduler")
    scaler = owned_stack.pop("scaler")
    bundle = owned_stack.pop("bundle")
    _require(not owned_stack, "checkpoint stack ownership was not fully transferred")
    state.epoch = 1  # Profiler-only shortened epoch; production remains 2,747 windows.
    checkpoint = output_dir / "checkpoint_boundary.pt"

    _sync(device)
    started = time.perf_counter()
    save_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        ema=None,
        state=state,
        config=config,
        checkpoint_identity=config.sha256,
    )
    _sync(device)
    save_seconds = time.perf_counter() - started

    started = time.perf_counter()
    checkpoint_sha = sha256_file(checkpoint)
    file_hash_seconds = time.perf_counter() - started
    started = time.perf_counter()
    boundary_model_sha = tensor_state_sha256(model.state_dict())
    model_hash_seconds = time.perf_counter() - started

    boundary = {
        "model": _state_capture(model.state_dict()),
        "optimizer": _state_capture(optimizer.state_dict()),
        "scheduler": _state_capture(scheduler.state_dict()),
        "scaler": _state_capture(scaler.state_dict()),
    }
    boundary_state = state.checkpoint_dict()

    # Uninterrupted control uses the same epoch-addressed worker/sampler boundary
    # as production, but only eight windows and D_fit engineering evidence.
    control_state = state
    control_target = control_state.optimizer_step + int(continuation_windows)
    bundle.set_epoch(control_state.epoch)
    control_batch_sha256: list[str] = []
    _run_segment(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        bundle=bundle,
        state=control_state,
        config=config,
        device=device,
        target_optimizer_step=control_target,
        readiness=False,
        batch_digests=control_batch_sha256,
        cpu_resident_batch_fields=cpu_resident_batch_fields,
    )
    expected_continuation_microbatches = int(continuation_windows) * int(
        config.as_dict()["training"]["accumulation_steps"]
    )
    _require(
        len(control_batch_sha256) == expected_continuation_microbatches,
        "uninterrupted-control input-digest count drift",
    )
    control = {
        "model": _state_capture(model.state_dict()),
        "optimizer": _state_capture(optimizer.state_dict()),
        "scheduler": _state_capture(scheduler.state_dict()),
        "scaler": _state_capture(scaler.state_dict()),
        "training_state": control_state.checkpoint_dict(),
        "rng_sha256": _rng_sha256(),
    }

    # Replay once in the same process from the exact serialized boundary. This
    # separates input-stream or kernel nondeterminism from fresh-process loading.
    replay_started = time.perf_counter()
    replay_state, replay_identity = load_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        ema=None,
        config=config,
        map_location="cpu",
    )
    _sync(device)
    _require(replay_identity == config.sha256, "same-process replay identity drift")
    _require(
        replay_state.checkpoint_dict() == boundary_state,
        "same-process replay training state drift",
    )
    _require(
        tensor_state_sha256(model.state_dict()) == boundary_model_sha,
        "same-process replay changed the boundary model",
    )
    bundle.set_epoch(replay_state.epoch)
    replay_batch_sha256: list[str] = []
    _run_segment(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        bundle=bundle,
        state=replay_state,
        config=config,
        device=device,
        target_optimizer_step=control_target,
        readiness=False,
        batch_digests=replay_batch_sha256,
        cpu_resident_batch_fields=cpu_resident_batch_fields,
    )
    _require(
        len(replay_batch_sha256) == expected_continuation_microbatches,
        "same-process replay input-digest count drift",
    )
    replay = {
        "model": _state_capture(model.state_dict()),
        "optimizer": _state_capture(optimizer.state_dict()),
        "scheduler": _state_capture(scheduler.state_dict()),
        "scaler": _state_capture(scaler.state_dict()),
        "training_state": replay_state.checkpoint_dict(),
        "rng_sha256": _rng_sha256(),
    }
    replay_continuation = {
        name: _compare_state_captures(
            control[name], replay[name], rtol=float(rtol), atol=float(atol)
        )
        for name in ("model", "optimizer", "scheduler", "scaler")
    }
    replay_input_equal = control_batch_sha256 == replay_batch_sha256
    replay_state_equal = control["training_state"] == replay["training_state"]
    replay_rng_equal = control["rng_sha256"] == replay["rng_sha256"]
    same_process = {
        "continuation": replay_continuation,
        "input_stream": {
            "microbatches": len(replay_batch_sha256),
            "control_sha256": control_batch_sha256,
            "replay_sha256": replay_batch_sha256,
            "exact_equal": replay_input_equal,
        },
        "training_state_equal": replay_state_equal,
        "rng_state_equal": replay_rng_equal,
        "elementwise_allclose_diagnostic_pass": (
            all(item["gate_pass"] for item in replay_continuation.values())
            and replay_input_equal
            and replay_state_equal
            and replay_rng_equal
        ),
    }
    replay_seconds = time.perf_counter() - replay_started

    reference_path = output_dir / "checkpoint_continuation_reference.pt"
    started = time.perf_counter()
    reference_sha = _torch_save_once(
        reference_path,
        {
            "schema": "s10.phase1p.continuation-reference.v2",
            "config_sha256": config.sha256,
            "profile_sha256": profile.sha256,
            "candidate_id": profile.data["candidate_id"],
            "boundary_model_sha256": boundary_model_sha,
            "boundary_training_state": boundary_state,
            "boundary": boundary,
            "control_target_optimizer_step": control_target,
            "control_batch_sha256": control_batch_sha256,
            "control": control,
        },
    )
    reference_write_seconds = time.perf_counter() - started

    bundle.close()
    del model, criterion, optimizer, scheduler, scaler, bundle, boundary, control, replay
    gc.collect()
    torch.cuda.empty_cache()

    worker_request_path = output_dir / "checkpoint_resume_worker_request.json"
    worker_result_path = output_dir / "checkpoint_resume_worker_result.json"
    _atomic_write_once(
        worker_request_path,
        {
            "schema": "s10.phase1p.resume-worker-request.v3",
            "config_path": str(config_path.resolve()),
            "config_sha256": config.sha256,
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": profile.sha256,
            "candidate_id": profile.data["candidate_id"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "reference": str(reference_path),
            "reference_sha256": reference_sha,
            "result": str(worker_result_path),
            "source_sha": source_sha,
            "continuation_windows": int(continuation_windows),
            "rtol": float(rtol),
            "atol": float(atol),
        },
    )
    worker_stdout = output_dir / "checkpoint_resume_worker.stdout"
    worker_stderr = output_dir / "checkpoint_resume_worker.stderr"
    started = time.perf_counter()
    with worker_stdout.open("xb") as stdout, worker_stderr.open("xb") as stderr:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--resume-worker-request", str(worker_request_path)],
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    worker_seconds = time.perf_counter() - started
    _require(
        process.returncode == 0,
        f"fresh-process checkpoint worker failed with {process.returncode}; see {worker_stderr}",
    )
    worker = _read_json(worker_result_path)
    _require(
        isinstance(worker.get("elementwise_allclose_diagnostic_pass"), bool),
        "fresh-process checkpoint worker omitted its allclose diagnostic",
    )

    fresh_process = {
        "continuation": worker["continuation"],
        "input_stream": worker["input_stream"],
        "training_state_equal": worker["training_state_equal"],
        "rng_state_equal": worker["rng_state_equal"],
        "elementwise_allclose_diagnostic_pass": worker[
            "elementwise_allclose_diagnostic_pass"
        ],
    }
    continuation_gate = evaluate_calibrated_continuation_gate(
        restored_boundary=worker["restored_boundary"],
        same_process=same_process,
        fresh_process=fresh_process,
        relative_l2_tolerance=float(rtol),
        max_absolute_tolerance=float(atol),
    )

    return {
        "schema": "s10.phase1p.checkpoint-profile.v2",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
            "model_state_sha256": boundary_model_sha,
        },
        "timing_seconds": {
            "save_including_device_transfer_and_atomic_replace": save_seconds,
            "checkpoint_file_sha256": file_hash_seconds,
            "separate_model_state_sha256": model_hash_seconds,
            "fresh_process_checkpoint_load": worker["checkpoint_load_seconds"],
            "fresh_process_model_stack_build": worker["model_stack_build_seconds"],
            "fresh_process_D_fit_loader_build": worker["D_fit_loader_build_seconds"],
            "continuation_reference_write_profiler_overhead": reference_write_seconds,
            "same_process_replay_profiler_overhead": replay_seconds,
            "fresh_process_worker_total_profiler_overhead": worker_seconds,
        },
        "fresh_process": {
            "parent_pid": os.getpid(),
            "worker_pid": worker["fresh_process_pid"],
            "candidate_configuration": worker["candidate_configuration"],
            "optimizer_configuration": worker["optimizer_configuration"],
            "request": {
                "path": str(worker_request_path),
                "sha256": sha256_file(worker_request_path),
            },
            "result": {"path": str(worker_result_path), "sha256": sha256_file(worker_result_path)},
            "stdout": {"path": str(worker_stdout), "sha256": sha256_file(worker_stdout)},
            "stderr": {"path": str(worker_stderr), "sha256": sha256_file(worker_stderr)},
            "continuation_reference": {
                "path": str(reference_path),
                "sha256": reference_sha,
                "bytes": reference_path.stat().st_size,
            },
        },
        "restored_boundary": worker["restored_boundary"],
        "same_process_replay": same_process,
        "continuation_windows": int(continuation_windows),
        "continuation": worker["continuation"],
        "input_stream": worker["input_stream"],
        "training_state_equal": worker["training_state_equal"],
        "rng_state_equal": worker["rng_state_equal"],
        "continuation_gate": continuation_gate,
        "gate_pass": continuation_gate["gate_pass"],
        "profiler_epoch_note": (
            "epoch=1 is a shortened D_fit-only profiler boundary; it validates production "
            "checkpoint mechanics and fresh-process continuation but is not a scientific epoch"
        ),
    }


def _validate_output_dir(
    path: Path,
    *,
    profile,
    approved_source_sha: str,
    source_sha: str,
    branch: str,
    mode: str,
    repeat: int,
    attempt_id: str,
) -> None:
    prefix = str(profile.data["boundaries"]["output_root_prefix"])
    root = Path(prefix + approved_source_sha[:12]).resolve()
    expected = root / branch / f"{mode}_{source_sha[:12]}_r{repeat}_{attempt_id}"
    _require(path == expected, f"profiler output path drift: {path} != {expected}")
    _require(not path.exists(), f"fresh profiler output already exists: {path}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_phase1_profile_spec(args.profile_config)
    profile.assert_runnable(args.branch)
    _require(args.mode != "trace" or args.repeat == 1, "trace mode permits repeat 1 only")
    _require(args.mode != "capacity" or args.repeat == 1,
             "capacity mode permits repeat 1 only")
    _require(
        args.mode != "capacity"
        or (
            profile.data["envelope"] == "IP-E2"
            and int(profile.candidates["physical_batch_size"]) in {8, 16}
        ),
        "capacity mode is frozen to IP-E2 physical B8/B16",
    )
    source_config = load_resolved_config(args.config)
    profile.assert_branch_binding(args.branch, args.config, source_config)
    config = derive_profile_runtime_config(source_config, profile)
    raw = config.as_dict()
    _require(raw["contract"]["lifecycle"] == "envelope_b_ready", "Phase-I config lifecycle drift")
    _require(raw["execution"]["mode"] == "phase1_train_eval", "Phase-I execution identity drift")
    physical_batch = int(profile.candidates["physical_batch_size"])
    _require(
        raw["training"]["micro_batch_size"] == physical_batch,
        "profile physical batch/runtime config drift",
    )
    _require(
        raw["training"]["world_size"] == 1,
        "single-GPU profiler requires world_size=1",
    )
    _require(
        raw["training"]["accumulation_steps"] == 32 // physical_batch,
        "profile accumulation/runtime config drift",
    )
    _require(raw["training"]["effective_global_batch"] == 32,
             "Phase I-P requires effective B32")
    _require(
        raw["checkpointing"]["recovery_cadence_epochs"] == 1,
        "Phase I-P requires the frozen per-epoch recovery cadence",
    )
    _require(
        raw["training"]["activation_checkpoint"] is False,
        "Phase I-P requires activation checkpointing off",
    )
    phase1_runtime_ready(raw)

    output_dir = Path(args.output_dir).resolve()
    _validate_output_dir(
        output_dir,
        profile=profile,
        approved_source_sha=args.approved_source_sha,
        source_sha=args.source_sha,
        branch=args.branch,
        mode=args.mode,
        repeat=args.repeat,
        attempt_id=args.attempt_id,
    )
    source = _source_identity(args.source_sha, args.approved_source_sha)
    runtime, runtime_dependency_sha = _runtime_identity(config)
    output_dir.mkdir(parents=True)
    compile_cache = output_dir / "torchinductor_cache_main"
    if bool(profile.candidates["torch_compile"]):
        _require(not compile_cache.exists(), "fresh main compile cache already exists")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compile_cache)
    device = torch.device("cuda", 0)
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(int(raw["training"]["seed"]))

    identity = {
        "schema": SCHEMA,
        "mode": args.mode,
        "branch": args.branch,
        "candidate_id": profile.data["candidate_id"],
        "source": source,
        "source_resolved_config_sha256": source_config.sha256,
        "effective_runtime_config_sha256": config.sha256,
        "source_config_file_sha256": sha256_file(args.config),
        "profile_config_sha256": profile.sha256,
        "runtime_dependencies_sha256": runtime_dependency_sha,
        "runtime": runtime,
        "seed": int(raw["training"]["seed"]),
        "data_role": "D_fit",
        "capability_metrics": False,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "candidate_options": dict(profile.candidates),
        "attempt": {**_attempt_identity(), "repeat": args.repeat, "attempt_id": args.attempt_id},
    }
    _atomic_write_once(output_dir / "run_identity.json", identity)
    _atomic_write_bytes_once(
        output_dir / "source_resolved_config.json", source_config.canonical_bytes
    )
    _atomic_write_bytes_once(
        output_dir / "effective_runtime_config.json", config.canonical_bytes
    )
    _atomic_write_bytes_once(output_dir / "profile_config.json", profile.canonical_bytes)

    startup_started = time.perf_counter()
    model_started = time.perf_counter()
    model, criterion, optimizer, scheduler, scaler = build_phase1_training_stack(
        config, device
    )
    optimizer_configuration_before = _optimizer_configuration(optimizer)
    _require(
        optimizer_configuration_before["fused"]
        is bool(profile.candidates["fused_adamw"]),
        "runtime AdamW fused backend differs from the profile",
    )
    dynamo_before = _dynamo_counters()
    candidate_configuration = _configure_profile_candidate(model, profile, args.branch)
    cpu_resident_batch_fields = _candidate_cpu_resident_batch_fields(
        profile, args.branch
    )
    model_seconds = time.perf_counter() - model_started
    loader_started = time.perf_counter()
    bundle = build_phase1_train_data(config)
    loader_seconds = time.perf_counter() - loader_started
    bundle.set_epoch(0)
    state = TrainingState()
    warmup = (
        1 if args.mode == "capacity"
        else int(profile.measurement["warmup_accepted_windows"])
    )
    active = int(profile.measurement[{
        "sustained": "sustained_accepted_windows",
        "trace": "trace_accepted_windows",
        "capacity": "capacity_accepted_windows",
    }[args.mode]])
    target = warmup + active
    trace_controller = (
        _TraceController(state, branch=args.branch, warmup=warmup, active=active)
        if args.mode == "trace"
        else None
    )
    system_sampler = _SystemSampler(
        output_dir / "nvidia_smi.csv",
        float(profile.measurement["system_sample_interval_seconds"]),
    )
    system_sampler.start()
    training_started = time.perf_counter()
    training_seconds = None
    input_anchor_sha256: list[str] = []
    dynamo_after_warmup: dict[str, dict[str, int]] | None = None

    def attempted_window_callback() -> None:
        nonlocal dynamo_after_warmup
        if trace_controller is not None:
            trace_controller.step()
        if state.optimizer_step == warmup and dynamo_after_warmup is None:
            dynamo_after_warmup = _dynamo_counters()

    try:
        metrics = _run_segment(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            state=state,
            config=config,
            device=device,
            target_optimizer_step=target,
            readiness=True,
            warmup=warmup,
            trace_controller=trace_controller,
            batch_digests=input_anchor_sha256,
            batch_digest_limit=int(raw["training"]["accumulation_steps"]),
            cpu_resident_batch_fields=cpu_resident_batch_fields,
            attempted_window_callback=attempted_window_callback,
        )
        training_seconds = time.perf_counter() - training_started
    finally:
        if trace_controller is not None:
            trace_controller.close()
        system_record = system_sampler.stop()
    _require(training_seconds is not None, "profiler training wall interval was not completed")
    dynamo_after = _dynamo_counters()
    _require(dynamo_after_warmup is not None, "compiler warm-up boundary was not observed")

    timing = metrics["readiness_timing"]
    _require(state.optimizer_step == target, "profiler did not reach accepted-window target")
    _require(state.nonfinite_windows == 0, "direct nonfinite profiler window")
    _require(state.discarded_windows == 0, "profiler discarded a partial window")
    _require(
        len(input_anchor_sha256) == int(raw["training"]["accumulation_steps"]),
        "first-window input anchor count drift",
    )
    _require(timing["measured_accepted_windows"] == active, "measured accepted-window drift")
    _require(
        timing["accumulation_steps"] == int(raw["training"]["accumulation_steps"]),
        "timing accumulation identity drift",
    )
    _require(timing["stage_timing"] is False, "sustained timing enabled stage events")
    memory_safe = (
        timing["memory"]["peak_reserved_fraction"]
        <= float(profile.measurement["max_reserved_fraction"])
        and not timing["memory"]["monotonic_reserved_growth_over_64mib"]
    )
    if args.mode != "capacity":
        _require(memory_safe, "candidate exceeds the frozen memory safety gate")
    prefix = _sampler_prefix_identity(bundle, 0, state.attempted_samples)

    trace_record = (
        None if trace_controller is None else trace_controller.publish(output_dir)
    )
    compile_steady_delta = _counter_delta(dynamo_after_warmup, dynamo_after)
    compile_evidence = {
        **candidate_configuration,
        "counter_delta_total": _counter_delta(dynamo_before, dynamo_after),
        "counter_delta_through_warmup": _counter_delta(
            dynamo_before, dynamo_after_warmup
        ),
        "counter_delta_measured_interval": compile_steady_delta,
        "unexpected_steady_state_recompile": (
            bool(profile.candidates["torch_compile"])
            and _counter_has_steady_recompile(compile_steady_delta)
        ),
        "cache_path": str(compile_cache) if compile_cache.exists() else None,
        "warmup_including_compile_seconds": max(
            0.0,
            float(training_seconds)
            - float(timing["measurement_wall_seconds"] or 0.0),
        ),
    }
    measurement_record = {
        "schema": "s10.phase1p.measurement.v2",
        "mode": args.mode,
        "branch": args.branch,
        "candidate_id": profile.data["candidate_id"],
        "source": source,
        "source_resolved_config_sha256": source_config.sha256,
        "effective_runtime_config_sha256": config.sha256,
        "profile_config_sha256": profile.sha256,
        "startup_seconds": {
            "model_loss_optimizer_scheduler_scaler": model_seconds,
            "D_fit_loader": loader_seconds,
            "before_training_total": training_started - startup_started,
        },
        "training_wall_seconds_including_warmup": training_seconds,
        "measurement": metrics,
        "physical_batch_size": physical_batch,
        "accumulation_steps": int(raw["training"]["accumulation_steps"]),
        "first_optimizer_window_input_sha256": input_anchor_sha256,
        "candidate_configuration": candidate_configuration,
        "optimizer_configuration_before_training": optimizer_configuration_before,
        "optimizer_configuration_after_training": _optimizer_configuration(optimizer),
        "compile_evidence": compile_evidence,
        "sampler_prefix": prefix,
        "system_sampling": system_record,
        "torch_trace": trace_record,
        "memory_safe_under_85_percent_reserved": memory_safe,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "capability_metrics": False,
    }
    measurement_sha = _atomic_write_once(
        output_dir / "measurement.json", measurement_record
    )
    checkpoint_record = None
    if args.mode == "sustained":
        tolerances = profile.data["parity"][str(raw["precision"]["global_autocast"])]
        owned_stack = {
            "model": model,
            "criterion": criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "bundle": bundle,
        }
        # Remove the caller-frame references before the callee releases the
        # uninterrupted stack and builds the fresh resume side.
        model = criterion = optimizer = scheduler = scaler = bundle = None
        checkpoint_record = _checkpoint_and_continuation(
            output_dir=output_dir,
            owned_stack=owned_stack,
            state=state,
            config=config,
            config_path=Path(args.config),
            profile=profile,
            profile_path=Path(args.profile_config),
            source_sha=args.source_sha,
            device=device,
            continuation_windows=int(
                profile.measurement["checkpoint_continuation_windows"]
            ),
            rtol=float(tolerances["rtol"]),
            atol=float(tolerances["atol"]),
        )
    else:
        bundle.close()

    checkpoint_gate = (
        True if checkpoint_record is None else bool(checkpoint_record["gate_pass"])
    )
    if not checkpoint_gate:
        status = "FAILED_CHECKPOINT_PARITY"
    elif args.mode == "sustained":
        status = "COMPLETE_SUSTAINED"
    elif args.mode == "capacity":
        status = "COMPLETE_CAPACITY" if memory_safe else "CAPACITY_MEMORY_REJECTED"
    else:
        status = "COMPLETE_TRACE"
    result = {
        "schema": SCHEMA,
        "status": status,
        "mode": args.mode,
        "branch": args.branch,
        "candidate_id": profile.data["candidate_id"],
        "source": source,
        "source_resolved_config_sha256": source_config.sha256,
        "effective_runtime_config_sha256": config.sha256,
        "profile_config_sha256": profile.sha256,
        "candidate_options": dict(profile.candidates),
        "candidate_configuration": candidate_configuration,
        "optimizer_configuration_before_training": optimizer_configuration_before,
        "compile_evidence": compile_evidence,
        "physical_batch_size": physical_batch,
        "accumulation_steps": int(raw["training"]["accumulation_steps"]),
        "first_optimizer_window_input_sha256": input_anchor_sha256,
        "startup_seconds": {
            "model_loss_optimizer_scheduler_scaler": model_seconds,
            "D_fit_loader": loader_seconds,
            "before_training_total": training_started - startup_started,
        },
        "training_wall_seconds_including_warmup": training_seconds,
        "measurement": metrics,
        "measurement_artifact_sha256": measurement_sha,
        "sampler_prefix": prefix,
        "system_sampling": system_record,
        "torch_trace": trace_record,
        "checkpoint": checkpoint_record,
        "memory_safe_under_85_percent_reserved": memory_safe,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "capability_metrics": False,
        "interpretation_limits": [
            "D_fit-only engineering throughput and numerical-health evidence",
            "no capability, mAP, NDS, generalization, or candidate-selection claim",
            "no scientific recipe promotion",
        ],
    }
    result_sha = _atomic_write_once(output_dir / "result.json", result)
    if not checkpoint_gate:
        _atomic_write_once(
            output_dir / "failed.json",
            {
                "schema": "s10.phase1p.failed.v1",
                "status": status,
                "result_sha256": result_sha,
                "failed_unix_seconds": time.time(),
            },
        )
        if profile.data["envelope"] != "IP-E2":
            raise RuntimeError("fresh-process checkpoint continuation parity failed")
    complete = {
        "schema": "s10.phase1p.complete.v1",
        "status": status,
        "result_sha256": result_sha,
        "completed_unix_seconds": time.time(),
    }
    _atomic_write_once(output_dir / "complete.json", complete)
    return result


def _record_capacity_oom(
    args: argparse.Namespace,
    error: BaseException,
) -> dict[str, Any]:
    """Seal an expected capacity OOM instead of misclassifying it as a code bug."""
    _require(args.mode == "capacity", "only a capacity probe may record CAPACITY_OOM")
    output_dir = Path(args.output_dir).resolve()
    identity_path = output_dir / "run_identity.json"
    _require(identity_path.is_file(), "capacity OOM occurred before run identity publication")
    identity = _read_json(identity_path)
    device = torch.device("cuda", 0)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    result = {
        "schema": SCHEMA,
        "status": "CAPACITY_OOM",
        "mode": "capacity",
        "branch": identity["branch"],
        "candidate_id": identity["candidate_id"],
        "source": identity["source"],
        "source_resolved_config_sha256": identity[
            "source_resolved_config_sha256"
        ],
        "effective_runtime_config_sha256": identity[
            "effective_runtime_config_sha256"
        ],
        "profile_config_sha256": identity["profile_config_sha256"],
        "candidate_options": identity["candidate_options"],
        "capacity_evidence": {
            "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "exception_message": str(error),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "device_total_bytes": total_memory,
            "peak_reserved_fraction": (
                peak_reserved / total_memory if total_memory else None
            ),
        },
        "memory_safe_under_85_percent_reserved": False,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "capability_metrics": False,
        "interpretation_limits": [
            "measurement-only capacity rejection",
            "no capability, quality, or recipe claim",
        ],
    }
    result_sha = _atomic_write_once(output_dir / "result.json", result)
    _atomic_write_once(
        output_dir / "complete.json",
        {
            "schema": "s10.phase1p.complete.v1",
            "status": "CAPACITY_OOM",
            "result_sha256": result_sha,
            "completed_unix_seconds": time.time(),
        },
    )
    return result


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--resume-worker-request":
        result = _checkpoint_resume_worker(Path(sys.argv[2]))
        print(json.dumps({
            "status": "COMPLETE_RESUME_WORKER",
            "fresh_process_pid": result["fresh_process_pid"],
            "elementwise_allclose_diagnostic_pass": result[
                "elementwise_allclose_diagnostic_pass"
            ],
        }, sort_keys=True))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("camera", "lidar"), required=True)
    parser.add_argument("--mode", choices=("sustained", "trace", "capacity"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--approved-source-sha", required=True)
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--attempt-id", required=True)
    arguments = parser.parse_args()
    for name in ("source_sha", "approved_source_sha"):
        value = getattr(arguments, name)
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"--{name.replace('_', '-')} must be a 40-character Git SHA")
    if not _ATTEMPT_ID.fullmatch(arguments.attempt_id):
        parser.error("--attempt-id must match [a-z0-9][a-z0-9_-]{0,31}")
    try:
        result = _run(arguments)
    except torch.OutOfMemoryError as error:
        if arguments.mode != "capacity":
            raise
        result = _record_capacity_oom(arguments, error)
    print(json.dumps({
        "status": result["status"],
        "branch": result["branch"],
        "mode": result["mode"],
        "effective_runtime_config_sha256": result["effective_runtime_config_sha256"],
        "profile_config_sha256": result["profile_config_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
