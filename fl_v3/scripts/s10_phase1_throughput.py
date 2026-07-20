#!/usr/bin/env python3
"""Bounded, D_fit-only S10 Phase I-P throughput profiler.

This entry reuses the production Phase-I model, loader, loss, AdamW, cyclic
scheduler, GradScaler, training loop, and checkpoint implementation.  It has no
evaluation import or code path and cannot access D_select, D_audit, or official
validation.  The checked-in IP-E1 profile permits only the B4 x accumulation-8
reference candidate; every optimization remains default-off.
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
from fl_v3.training.phase1_profile import load_phase1_profile_spec
from fl_v3.training.runtime_state import TrainingState
from fl_v3.training.s10_observation import compare_tensor_tree_tensors
from fl_v3.utils.runtime import (
    enforce_determinism,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "s10.phase1p.profiler-result.v1"
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
    for name in sorted(set(reference_tensors) & set(candidate_tensors)):
        left = reference_tensors[name]
        right = candidate_tensors[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            continue
        if left.is_floating_point() or left.is_complex():
            if not torch.allclose(left, right, rtol=rtol, atol=atol, equal_nan=False):
                allclose_failures.append(name)
        elif not torch.equal(left, right):
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


class _TraceController:
    """Start after accepted warm-up and stop after the requested accepted windows."""

    def __init__(self, state: TrainingState, *, warmup: int, active: int) -> None:
        self.state = state
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
        self.profiler.export_chrome_trace(str(trace))
        sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        table = self.profiler.key_averages().table(sort_by=sort_key, row_limit=200)
        with summary.open("x", encoding="utf-8") as stream:
            stream.write(table)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "schema": "s10.phase1p.torch-trace.v1",
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
) -> dict[str, Any]:
    raw = config.as_dict()
    with ExitStack() as ranges:
        if trace_controller is not None:
            ranges.enter_context(model.operator_profile_ranges())
            if hasattr(criterion, "operator_profile_ranges"):
                ranges.enter_context(criterion.operator_profile_ranges())
        return train_one_epoch(
            model,
            bundle.loader,
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
                None if trace_controller is None else trace_controller.step
            ),
        )


def _checkpoint_resume_worker(request_path: Path) -> dict[str, Any]:
    """Reload and continue in a fresh Python process, then publish compact parity."""
    request = _read_json(request_path.resolve())
    expected = {
        "schema", "config_path", "config_sha256", "checkpoint", "checkpoint_sha256",
        "reference", "reference_sha256", "result", "source_sha", "continuation_windows",
        "rtol", "atol",
    }
    _require(set(request) == expected, "fresh-process resume request fields drift")
    _require(request["schema"] == "s10.phase1p.resume-worker-request.v1", "resume schema drift")
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

    config = load_resolved_config(request["config_path"])
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
    _require(reference["schema"] == "s10.phase1p.continuation-reference.v1",
             "continuation reference schema drift")
    _require(reference["config_sha256"] == config.sha256, "continuation reference config drift")

    device = torch.device("cuda", 0)
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(int(config.as_dict()["training"]["seed"]))
    started = time.perf_counter()
    model, criterion, optimizer, scheduler, scaler = build_phase1_training_stack(
        config, device
    )
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
        gate = (
            all(item["gate_pass"] for item in continuation.values())
            and state_equal
            and rng_equal
        )
        result = {
            "schema": "s10.phase1p.resume-worker-result.v1",
            "fresh_process_pid": os.getpid(),
            "model_stack_build_seconds": model_build_seconds,
            "D_fit_loader_build_seconds": loader_build_seconds,
            "checkpoint_load_seconds": load_seconds,
            "restored_boundary": restored,
            "continuation": continuation,
            "training_state_equal": state_equal,
            "rng_state_equal": rng_equal,
            "gate_pass": gate,
        }
        _atomic_write_once(result_path, result)
        _require(gate, "fresh-process checkpoint continuation parity failed")
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
    source_sha: str,
    device: torch.device,
    continuation_windows: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
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
    )
    control = {
        "model": _state_capture(model.state_dict()),
        "optimizer": _state_capture(optimizer.state_dict()),
        "scheduler": _state_capture(scheduler.state_dict()),
        "scaler": _state_capture(scaler.state_dict()),
        "training_state": control_state.checkpoint_dict(),
        "rng_sha256": _rng_sha256(),
    }

    reference_path = output_dir / "checkpoint_continuation_reference.pt"
    started = time.perf_counter()
    reference_sha = _torch_save_once(
        reference_path,
        {
            "schema": "s10.phase1p.continuation-reference.v1",
            "config_sha256": config.sha256,
            "boundary_model_sha256": boundary_model_sha,
            "boundary_training_state": boundary_state,
            "boundary": boundary,
            "control_target_optimizer_step": control_target,
            "control": control,
        },
    )
    reference_write_seconds = time.perf_counter() - started

    bundle.close()
    del model, criterion, optimizer, scheduler, scaler, bundle, boundary, control
    gc.collect()
    torch.cuda.empty_cache()

    worker_request_path = output_dir / "checkpoint_resume_worker_request.json"
    worker_result_path = output_dir / "checkpoint_resume_worker_result.json"
    _atomic_write_once(
        worker_request_path,
        {
            "schema": "s10.phase1p.resume-worker-request.v1",
            "config_path": str(config_path.resolve()),
            "config_sha256": config.sha256,
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
    _require(worker.get("gate_pass") is True, "fresh-process checkpoint gate did not pass")

    return {
        "schema": "s10.phase1p.checkpoint-profile.v1",
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
            "fresh_process_worker_total_profiler_overhead": worker_seconds,
        },
        "fresh_process": {
            "parent_pid": os.getpid(),
            "worker_pid": worker["fresh_process_pid"],
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
        "continuation_windows": int(continuation_windows),
        "continuation": worker["continuation"],
        "training_state_equal": worker["training_state_equal"],
        "rng_state_equal": worker["rng_state_equal"],
        "gate_pass": worker["gate_pass"],
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
    profile.assert_baseline()
    _require(args.mode != "trace" or args.repeat == 1, "trace mode permits repeat 1 only")
    config = load_resolved_config(args.config)
    profile.assert_branch_binding(args.branch, args.config, config)
    raw = config.as_dict()
    _require(raw["contract"]["lifecycle"] == "envelope_b_ready", "Phase-I config lifecycle drift")
    _require(raw["execution"]["mode"] == "phase1_train_eval", "Phase-I execution identity drift")
    _require(raw["training"]["micro_batch_size"] == 4, "IP-E1 requires physical B4")
    _require(raw["training"]["accumulation_steps"] == 8, "IP-E1 requires accumulation 8")
    _require(raw["training"]["effective_global_batch"] == 32, "IP-E1 requires effective B32")
    _require(
        raw["checkpointing"]["recovery_cadence_epochs"] == 1,
        "IP-E1 reference requires the frozen per-epoch recovery cadence",
    )
    _require(
        raw["training"]["activation_checkpoint"] is False,
        "IP-E1 reference requires activation checkpointing off",
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
    device = torch.device("cuda", 0)
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(int(raw["training"]["seed"]))

    identity = {
        "schema": SCHEMA,
        "mode": args.mode,
        "branch": args.branch,
        "candidate_id": profile.data["candidate_id"],
        "source": source,
        "resolved_config_sha256": config.sha256,
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
    _atomic_write_bytes_once(output_dir / "resolved_config.json", config.canonical_bytes)
    _atomic_write_bytes_once(output_dir / "profile_config.json", profile.canonical_bytes)

    startup_started = time.perf_counter()
    model_started = time.perf_counter()
    model, criterion, optimizer, scheduler, scaler = build_phase1_training_stack(
        config, device
    )
    model_seconds = time.perf_counter() - model_started
    loader_started = time.perf_counter()
    bundle = build_phase1_train_data(config)
    loader_seconds = time.perf_counter() - loader_started
    bundle.set_epoch(0)
    state = TrainingState()
    warmup = int(profile.measurement["warmup_accepted_windows"])
    active = int(
        profile.measurement[
            "sustained_accepted_windows"
            if args.mode == "sustained"
            else "trace_accepted_windows"
        ]
    )
    target = warmup + active
    trace_controller = (
        _TraceController(state, warmup=warmup, active=active)
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
        )
        training_seconds = time.perf_counter() - training_started
    finally:
        if trace_controller is not None:
            trace_controller.close()
        system_record = system_sampler.stop()
    _require(training_seconds is not None, "profiler training wall interval was not completed")

    timing = metrics["readiness_timing"]
    _require(state.optimizer_step == target, "profiler did not reach accepted-window target")
    _require(state.nonfinite_windows == 0, "direct nonfinite profiler window")
    _require(state.discarded_windows == 0, "profiler discarded a partial window")
    _require(timing["measured_accepted_windows"] == active, "measured accepted-window drift")
    _require(timing["accumulation_steps"] == 8, "timing accumulation identity drift")
    _require(timing["stage_timing"] is False, "sustained timing enabled stage events")
    memory_safe = (
        timing["memory"]["peak_reserved_fraction"]
        <= float(profile.measurement["max_reserved_fraction"])
    )
    _require(memory_safe, "baseline exceeds the frozen 85% reserved-memory ceiling")
    prefix = _sampler_prefix_identity(bundle, 0, state.attempted_samples)

    trace_record = (
        None if trace_controller is None else trace_controller.publish(output_dir)
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

    status = "COMPLETE_SUSTAINED" if args.mode == "sustained" else "COMPLETE_TRACE"
    result = {
        "schema": SCHEMA,
        "status": status,
        "mode": args.mode,
        "branch": args.branch,
        "candidate_id": profile.data["candidate_id"],
        "source": source,
        "resolved_config_sha256": config.sha256,
        "profile_config_sha256": profile.sha256,
        "candidate_options": dict(profile.candidates),
        "startup_seconds": {
            "model_loss_optimizer_scheduler_scaler": model_seconds,
            "D_fit_loader": loader_seconds,
            "before_training_total": training_started - startup_started,
        },
        "training_wall_seconds_including_warmup": training_seconds,
        "measurement": metrics,
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
    complete = {
        "schema": "s10.phase1p.complete.v1",
        "status": status,
        "result_sha256": result_sha,
        "completed_unix_seconds": time.time(),
    }
    _atomic_write_once(output_dir / "complete.json", complete)
    return result


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--resume-worker-request":
        result = _checkpoint_resume_worker(Path(sys.argv[2]))
        print(json.dumps({
            "status": "COMPLETE_RESUME_WORKER",
            "fresh_process_pid": result["fresh_process_pid"],
            "gate_pass": result["gate_pass"],
        }, sort_keys=True))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("camera", "lidar"), required=True)
    parser.add_argument("--mode", choices=("sustained", "trace"), required=True)
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
    result = _run(arguments)
    print(json.dumps({
        "status": result["status"],
        "branch": result["branch"],
        "mode": result["mode"],
        "resolved_config_sha256": result["resolved_config_sha256"],
        "profile_config_sha256": result["profile_config_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
