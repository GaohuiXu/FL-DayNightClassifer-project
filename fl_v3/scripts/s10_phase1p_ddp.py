#!/usr/bin/env python3
"""Exact two-GH200 Camera DDP engineering qualifier for S10 Phase I-P.

This entry is launched only through ``torchrun --nproc-per-node=2``.  It has no
evaluation path and consumes D_fit only.  ``profile`` records the sustained run,
canonical rank-zero checkpoint boundary and same-process continuation control;
``resume`` is a second fresh torchrun that validates continuation and emits the
terminal DDP result.  ``smoke`` checks NCCL rank/device wiring without D_fit.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

sys.path.insert(0, "fl_v3/src")

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fl_v3.config import load_resolved_config
from fl_v3.config.phase1 import phase1_runtime_ready
from fl_v3.data.nuscenes.phase1 import build_phase1_train_data
from fl_v3.models.phase1_swin import sha256_file
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.phase1 import build_phase1_training_stack
from fl_v3.training.phase1_checkpoint_gate import (
    evaluate_calibrated_continuation_gate,
)
from fl_v3.training.phase1_profile import (
    derive_profile_runtime_config,
    load_phase1_profile_spec,
)
from fl_v3.training.phase1_runtime import phase1_runtime_optimization_identity
from fl_v3.training.phase1p_ddp import (
    BatchDigestLoader,
    CONTINUATION_WINDOWS,
    EFFECTIVE_BATCH,
    LOCAL_BATCH,
    MEASURED_WINDOWS,
    WARMUP_WINDOWS,
    WORLD_SIZE,
    aggregate_rank_measurements,
    canonical_bytes,
    canonical_sha256,
    capture_sha256,
    capture_stack,
    capture_state,
    compare_state_captures,
    compare_stacks,
    restore_rng_state,
    rng_sha256,
    rng_state,
)
from fl_v3.training.runtime_state import TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "s10.phase1p.ip-e5-ddp-result.v1"
PROFILE_SCHEMA = "s10.phase1p.ip-e5-ddp-profile.v1"
EXPECTED_BRANCH = "codex/s10-phase1p-throughput-preflight"
EXPECTED_BASE_SHA = "f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
EXPECTED_APPROVED_SOURCE_SHA = "e61c486757ca5fe89340c9325014f4c3e048da2b"
FROZEN_CONTROL_REF = "refs/heads/codex/s10-phase1-branch-qualification"
OUTPUT_ROOT_PREFIX = (
    "/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/"
    "arrhenius_fl_v3/outputs/s10_phase1p_ip_e5_"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity(source_sha: str, approved_source_sha: str) -> dict[str, Any]:
    actual = _git("rev-parse", "HEAD")
    tree = _git("rev-parse", "HEAD^{tree}")
    branch = _git("branch", "--show-current")
    _require(actual == source_sha, f"source SHA drift: {actual}")
    _require(branch == EXPECTED_BRANCH, f"source branch drift: {branch!r}")
    _require(
        not _git("status", "--porcelain", "--untracked-files=all"),
        "DDP qualification requires a clean worktree",
    )
    control = _git("rev-parse", FROZEN_CONTROL_REF)
    _require(control == EXPECTED_BASE_SHA, "frozen Phase-I control branch moved")
    for older, newer, label in (
        (EXPECTED_BASE_SHA, approved_source_sha, "approved source/base"),
        (approved_source_sha, source_sha, "runtime/approved source"),
    ):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer], check=False
        )
        _require(result.returncode == 0, f"{label} ancestry drift")
    _require(
        not _git("rev-list", "--min-parents=2", f"{EXPECTED_BASE_SHA}..{source_sha}"),
        "DDP qualification source history is not linear",
    )
    return {
        "git_sha": actual,
        "git_tree": tree,
        "branch": branch,
        "approved_source_sha": approved_source_sha,
        "unique_base_sha": EXPECTED_BASE_SHA,
        "frozen_control_sha": control,
        "derived_source": actual != approved_source_sha,
    }


def _attempt_identity(rank: int) -> dict[str, Any]:
    return {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node_mib": os.environ.get("SLURM_MEM_PER_NODE"),
        "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "rank": int(rank),
        "pid": os.getpid(),
    }


def _validated_output_dir(
    *, mode: str, output_dir: str, source_sha: str, approved_source_sha: str
) -> Path:
    root = Path(f"{OUTPUT_ROOT_PREFIX}{approved_source_sha[:12]}").resolve()
    relative = (
        Path(f"ddp_smoke_{source_sha[:12]}")
        if mode == "smoke"
        else Path("camera") / f"ddp2_{source_sha[:12]}"
    )
    expected = (root / relative).resolve()
    actual = Path(output_dir).resolve()
    _require(actual == expected, f"DDP output path drift: {actual} != {expected}")
    return actual


def _atomic_json(path: Path, value: Any) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".partial")
    payload = canonical_bytes(value) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_torch(path: Path, value: Any) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected object at {path}")
    return value


def _rank_context() -> tuple[int, int, int, torch.device]:
    _require(platform.machine() == "aarch64", "DDP qualifier requires aarch64 GH200")
    _require(torch.cuda.is_available(), "DDP qualifier requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _require(world_size == WORLD_SIZE, "DDP qualifier requires exactly two ranks")
    _require(0 <= rank < world_size and local_rank == rank, "single-node rank mapping drift")
    _require(torch.cuda.device_count() == WORLD_SIZE, "each rank must see both allocated GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    name = torch.cuda.get_device_name(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    _require("GH200" in name and capability == (9, 0), f"unexpected device {name}/{capability}")
    return rank, local_rank, world_size, device


def _distributed_boolean_and(name: str, value: bool, device: torch.device) -> bool:
    local = torch.tensor(int(bool(value)), device=device, dtype=torch.int32)
    minimum = local.clone()
    maximum = local.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if name == "optimizer_accepted" and int(minimum.item()) != int(maximum.item()):
        raise RuntimeError("GradScaler accepted-update decision differs across ranks")
    return bool(int(minimum.item()))


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
    result = {}
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


def _has_recompile(delta: Mapping[str, Mapping[str, int]]) -> bool:
    signals = ("unique_graph", "recompil", "cache_miss", "fxgraph_cache_miss")
    return any(
        int(value) > 0 and any(signal in key.lower() for signal in signals)
        for values in delta.values()
        for key, value in values.items()
    )


def _mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    return capture_sha256(capture_state(dict(values)))


def _rank_state_identity(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    state: TrainingState,
) -> dict[str, Any]:
    parameter_names = {name for name, _ in model.named_parameters()}
    model_state = model.state_dict()
    parameters = {name: model_state[name] for name in sorted(parameter_names)}
    bn_names = {
        name
        for name in model_state
        if name.endswith("running_mean") or name.endswith("running_var")
    }
    non_bn_buffers = {
        name: model_state[name]
        for name in model_state
        if name not in parameter_names and name not in bn_names
    }
    return {
        "parameters_sha256": _mapping_sha256(parameters),
        "non_bn_buffers_sha256": _mapping_sha256(non_bn_buffers),
        "optimizer_sha256": capture_sha256(capture_state(optimizer.state_dict())),
        "scheduler_sha256": capture_sha256(capture_state(scheduler.state_dict())),
        "scaler_sha256": capture_sha256(capture_state(scaler.state_dict())),
        "training_state": state.checkpoint_dict(),
    }


def _bn_diagnostics(model: torch.nn.Module, rank: int) -> dict[str, Any] | None:
    local = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.endswith("running_mean") or name.endswith("running_var")
    }
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local)
    if rank != 0:
        return None
    left = capture_state(gathered[0])
    right = capture_state(gathered[1])
    report = compare_state_captures(left, right, rtol=0.0, atol=0.0)
    return {
        "rank0_vs_rank1": report["numerical"]["global"],
        "elementwise_exact": bool(report["gate_pass"]),
        "tensor_count": len(local),
        "interpretation": (
            "ordinary DDP broadcasts rank-zero buffers before each forward, then each "
            "rank updates BN buffers on its local B16; divergence is measured and the "
            "rank-zero boundary is canonicalized only outside sustained timing"
        ),
    }


def _canonicalize_model(model: torch.nn.Module) -> None:
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            dist.broadcast(value, src=0)
    dist.barrier()


def _gather_identity(local: Mapping[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, dict(local))
    return [dict(value) for value in gathered]


def _agreement(gathered: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "parameters_sha256",
        "non_bn_buffers_sha256",
        "optimizer_sha256",
        "scheduler_sha256",
        "scaler_sha256",
        "training_state",
    )
    checks = {
        key: all(record[key] == gathered[0][key] for record in gathered[1:])
        for key in keys
    }
    return {"checks": checks, "gate_pass": all(checks.values())}


def _sampler_identity(bundle: Any, rank: int) -> dict[str, Any] | None:
    sampler = bundle.sampler
    local = np.asarray(sampler.rank_epoch_positions(0), dtype="<i8")
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local.tolist())
    if rank != 0:
        return None
    arrays = [np.asarray(value, dtype=np.int64) for value in gathered]
    global_positions = np.asarray(sampler.global_epoch_positions(0), dtype=np.int64)
    windows = np.concatenate(
        [value.reshape(-1, LOCAL_BATCH) for value in arrays], axis=1
    ).reshape(-1)
    exact = np.array_equal(windows, global_positions)
    unique = len(np.unique(windows)) == len(windows)
    return {
        "epoch": 0,
        "global_presentations": int(global_positions.size),
        "rank_presentations": [int(value.size) for value in arrays],
        "global_positions_sha256": hashlib.sha256(
            np.ascontiguousarray(global_positions, dtype="<i8").tobytes()
        ).hexdigest(),
        "rank_positions_sha256": [
            hashlib.sha256(np.ascontiguousarray(value, dtype="<i8").tobytes()).hexdigest()
            for value in arrays
        ],
        "window_union_exact": bool(exact),
        "no_ddp_duplicate_or_omission": bool(exact and unique),
        "gate_pass": bool(
            exact
            and unique
            and global_positions.size == 87_904
            and all(value.size == 43_952 for value in arrays)
        ),
    }


def _run_segment(
    *,
    ddp_model: DistributedDataParallel,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    bundle: Any,
    state: TrainingState,
    config: Any,
    device: torch.device,
    windows: int,
    readiness: bool,
    digests: list[str] | None = None,
    warmup_counter_callback: Any = None,
) -> dict[str, Any]:
    loader = (
        bundle.loader
        if digests is None
        else BatchDigestLoader(bundle.loader, digests, limit=windows)
    )
    target = state.optimizer_step + int(windows)
    return train_one_epoch(
        ddp_model,
        loader,
        criterion,
        optimizer,
        device,
        grad_clip_norm=float(config.as_dict()["training"]["gradient_clip"]["max_norm"]),
        scheduler=scheduler,
        ema_model=None,
        max_steps=int(windows),
        precision=str(config.as_dict()["precision"]["global_autocast"]),
        grad_scaler=scaler,
        telemetry_interval=0,
        accumulation_steps=1,
        runtime_state=state,
        max_optimizer_steps=target,
        model_mode="camera_only",
        exposure_multiplier=WORLD_SIZE,
        expected_global_microbatch_samples=EFFECTIVE_BATCH,
        readiness_timing=readiness,
        readiness_warmup_successful_windows=WARMUP_WINDOWS if readiness else 0,
        readiness_stage_timing=False if readiness else True,
        attempted_window_callback=warmup_counter_callback,
        distributed_boolean_and=lambda name, value: _distributed_boolean_and(
            name, value, device
        ),
    )


def _load_contract(args: argparse.Namespace):
    source_config = load_resolved_config(args.config)
    profile = load_phase1_profile_spec(args.profile_config)
    profile.assert_runnable("camera")
    profile.assert_branch_binding("camera", args.config, source_config)
    _require(profile.data["envelope"] == "IP-E5", "DDP profile envelope drift")
    _require(
        profile.data["candidate_id"] == "camera_final_b16_ddp2",
        "DDP candidate identity drift",
    )
    config = derive_profile_runtime_config(source_config, profile)
    raw = config.as_dict()
    _require(
        raw["training"]["micro_batch_size"] == LOCAL_BATCH
        and raw["training"]["world_size"] == WORLD_SIZE
        and raw["training"]["accumulation_steps"] == 1
        and raw["training"]["effective_global_batch"] == EFFECTIVE_BATCH,
        "DDP runtime batch identity drift",
    )
    _require(raw["contract"]["branch"] == "camera", "DDP qualifier is Camera-only")
    _require(raw["checkpointing"]["recovery_cadence_epochs"] == 1, "checkpoint cadence drift")
    phase1_runtime_ready(raw)
    return source_config, profile, config


def _build_ddp_stack(config: Any, device: torch.device, rank: int, mode: str):
    cache = Path(os.environ["S10_DDP_OUTPUT_DIR"]) / f"compile_{mode}_rank{rank}"
    _require(not cache.exists(), f"fresh compile cache exists: {cache}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    model, criterion, optimizer, scheduler, scaler = build_phase1_training_stack(
        config, device
    )
    runtime = phase1_runtime_optimization_identity(model)
    expected_preprocess = {
        "batched_affine_grid": True,
        "vectorized_geometry": True,
        "bulk_input_conversion": True,
    }
    _require(
        runtime["camera_sdpa"] is True
        and runtime["camera_preprocess"] == expected_preprocess
        and runtime["torch_compile"] is True
        and runtime["fused_adamw"] is True,
        "DDP stack differs from the final Camera production recipe",
    )
    ddp = DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=True,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    return model, ddp, criterion, optimizer, scheduler, scaler, runtime


def _profile(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    source_config, profile, config = _load_contract(args)
    output = _validated_output_dir(
        mode="profile",
        output_dir=args.output_dir,
        source_sha=args.source_sha,
        approved_source_sha=args.approved_source_sha,
    )
    if rank == 0:
        _require(not output.exists(), f"fresh DDP output exists: {output}")
        output.mkdir(parents=True)
    dist.barrier()
    os.environ["S10_DDP_OUTPUT_DIR"] = str(output)
    source = _source_identity(args.source_sha, args.approved_source_sha)
    startup_started = time.perf_counter()
    dependencies = verify_runtime_dependency_identity(config.to_run_config())
    dependency_sha = canonical_sha256(dependencies)
    seed_everything(int(config.as_dict()["training"]["seed"]))
    enforce_determinism(strict=False, precision="fp16")

    model, ddp, criterion, optimizer, scheduler, scaler, runtime = _build_ddp_stack(
        config, device, rank, "profile"
    )
    model_seconds = time.perf_counter() - startup_started
    loader_started = time.perf_counter()
    bundle = build_phase1_train_data(
        config, distributed_rank=rank, distributed_world_size=WORLD_SIZE
    )
    loader_seconds = time.perf_counter() - loader_started
    startup_seconds = {
        "runtime_dependencies_and_model_stack": model_seconds,
        "D_fit_loader": loader_seconds,
        "before_training_total": time.perf_counter() - startup_started,
    }
    try:
        sampler = _sampler_identity(bundle, rank)
        bundle.set_epoch(0)
        state = TrainingState()
        counters_after_warmup: dict[str, dict[str, int]] | None = None

        def counter_callback() -> None:
            nonlocal counters_after_warmup
            if state.successful_windows == WARMUP_WINDOWS:
                counters_after_warmup = _dynamo_counters()

        started = time.perf_counter()
        metrics = _run_segment(
            ddp_model=ddp,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            state=state,
            config=config,
            device=device,
            windows=WARMUP_WINDOWS + MEASURED_WINDOWS,
            readiness=True,
            warmup_counter_callback=counter_callback,
        )
        torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - started
        _require(counters_after_warmup is not None, "compile warm-up boundary missing")
        steady_delta = _counter_delta(counters_after_warmup, _dynamo_counters())
        compile_evidence = {
            "steady_counter_delta": steady_delta,
            "unexpected_steady_state_recompile": _has_recompile(steady_delta),
        }
        local_measurement = {
            "rank": rank,
            "metrics": metrics,
            "compile_evidence": compile_evidence,
            "startup_seconds": startup_seconds,
            "training_wall_seconds_including_warmup": training_seconds,
            "device": {
                "name": torch.cuda.get_device_name(device),
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            },
        }
        _atomic_json(output / "ranks" / f"measurement_rank{rank}.json", local_measurement)
        dist.barrier()
        aggregate = None
        if rank == 0:
            rank_measurements = [
                _read_json(output / "ranks" / f"measurement_rank{value}.json")
                for value in range(WORLD_SIZE)
            ]
            aggregate = aggregate_rank_measurements(rank_measurements)

        pre_identity = _rank_state_identity(
            model, optimizer, scheduler, scaler, state
        )
        gathered_pre = _gather_identity(pre_identity)
        pre_agreement = _agreement(gathered_pre) if rank == 0 else None
        bn_diagnostics = _bn_diagnostics(model, rank)

        # Canonicalize only after sustained timing. This is equivalent to the
        # rank-zero buffer broadcast performed before the next DDP forward and
        # makes the checkpoint boundary single-valued.
        _canonicalize_model(model)
        canonical_identity = _rank_state_identity(
            model, optimizer, scheduler, scaler, state
        )
        gathered_canonical = _gather_identity(canonical_identity)
        canonical_agreement = _agreement(gathered_canonical) if rank == 0 else None

        state.epoch = 1
        boundary_stack = capture_stack(model, optimizer, scheduler, scaler)
        boundary_hashes = {
            name: capture_sha256(value) for name, value in boundary_stack.items()
        }
        boundary_state = state.checkpoint_dict()
        checkpoint_started = time.perf_counter()
        rank_rng = rng_state()
        rng_path = output / "checkpoint" / f"rng_rank{rank}.pt"
        rng_file_sha = _atomic_torch(rng_path, rank_rng)
        dist.barrier()
        checkpoint = output / "checkpoint" / "checkpoint.pt"
        if rank == 0:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
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
        dist.barrier()
        checkpoint_sha_holder: list[Any] = [None]
        if rank == 0:
            _require(checkpoint.is_file(), "rank-zero DDP checkpoint is absent")
            checkpoint_sha_holder[0] = sha256_file(checkpoint)
        dist.broadcast_object_list(checkpoint_sha_holder, src=0)
        checkpoint_sha = str(checkpoint_sha_holder[0])
        _require(len(checkpoint_sha) == 64, "rank-zero DDP checkpoint hash is invalid")
        dist.barrier()
        local_checkpoint_wall = time.perf_counter() - checkpoint_started
        checkpoint_walls: list[Any] = [None] * WORLD_SIZE
        dist.all_gather_object(checkpoint_walls, local_checkpoint_wall)
        checkpoint_wall = max(float(value) for value in checkpoint_walls)

        bundle.set_epoch(1)
        control_inputs: list[str] = []
        _run_segment(
            ddp_model=ddp,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            state=state,
            config=config,
            device=device,
            windows=CONTINUATION_WINDOWS,
            readiness=False,
            digests=control_inputs,
        )
        _canonicalize_model(model)
        control_stack = capture_stack(model, optimizer, scheduler, scaler)
        control_state = state.checkpoint_dict()
        control_rng = rng_sha256()

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
        restore_rng_state(rank_rng)
        _require(replay_identity == config.sha256, "replay checkpoint identity drift")
        replay_boundary_stack = capture_stack(model, optimizer, scheduler, scaler)
        restored_boundary = compare_stacks(
            boundary_stack, replay_boundary_stack, rtol=0.0, atol=0.0
        )
        bundle.set_epoch(1)
        replay_inputs: list[str] = []
        _run_segment(
            ddp_model=ddp,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            state=replay_state,
            config=config,
            device=device,
            windows=CONTINUATION_WINDOWS,
            readiness=False,
            digests=replay_inputs,
        )
        _canonicalize_model(model)
        replay_stack = capture_stack(model, optimizer, scheduler, scaler)
        same_process = {
            "continuation": compare_stacks(
                control_stack,
                replay_stack,
                rtol=float(profile.data["parity"]["fp16"]["rtol"]),
                atol=float(profile.data["parity"]["fp16"]["atol"]),
            ),
            "input_stream": {
                "control_sha256": control_inputs,
                "replay_sha256": replay_inputs,
                "exact_equal": control_inputs == replay_inputs,
            },
            "training_state_equal": control_state == replay_state.checkpoint_dict(),
            "rng_state_equal": control_rng == rng_sha256(),
        }
        same_process["elementwise_allclose_diagnostic_pass"] = bool(
            all(value["gate_pass"] for value in same_process["continuation"].values())
            and same_process["input_stream"]["exact_equal"]
            and same_process["training_state_equal"]
            and same_process["rng_state_equal"]
        )
        reference_path = output / "checkpoint" / f"continuation_reference_rank{rank}.pt"
        reference_sha = _atomic_torch(
            reference_path,
            {
                "schema": "s10.phase1p.ip-e5-ddp-continuation-reference.v1",
                "config_sha256": config.sha256,
                "profile_sha256": profile.sha256,
                "boundary_stack": boundary_stack,
                "boundary_hashes": boundary_hashes,
                "boundary_training_state": boundary_state,
                "control_stack": control_stack,
                "control_training_state": control_state,
                "control_input_sha256": control_inputs,
                "control_rng_sha256": control_rng,
            },
        )
        rank_record = {
            "schema": "s10.phase1p.ip-e5-ddp-rank-profile.v1",
            "rank": rank,
            "source": source,
            "attempt": _attempt_identity(rank),
            "source_config_sha256": source_config.sha256,
            "effective_config_sha256": config.sha256,
            "profile_sha256": profile.sha256,
            "runtime_dependencies_sha256": dependency_sha,
            "runtime_optimizations": runtime,
            "worker_seed_epoch0": int(config.as_dict()["training"]["seed"]) + rank,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "rng_sidecar": str(rng_path),
                "rng_sidecar_sha256": rng_file_sha,
                "restored_boundary": restored_boundary,
            },
            "same_process_replay": same_process,
            "continuation_reference": {
                "path": str(reference_path),
                "sha256": reference_sha,
            },
        }
        rank_record_path = output / "ranks" / f"profile_rank{rank}.json"
        _atomic_json(rank_record_path, rank_record)
        dist.barrier()
        if rank == 0:
            _require(aggregate is not None and sampler is not None, "rank-zero profile evidence missing")
            _require(pre_agreement is not None and canonical_agreement is not None, "rank agreement missing")
            profile_result = {
                "schema": PROFILE_SCHEMA,
                "status": "COMPLETE_PROFILE_AWAIT_RESUME",
                "source": source,
                "attempt": _attempt_identity(rank),
                "source_config_file_sha256": sha256_file(args.config),
                "source_config_sha256": source_config.sha256,
                "effective_config_sha256": config.sha256,
                "profile_file_sha256": sha256_file(args.profile_config),
                "profile_sha256": profile.sha256,
                "candidate_id": profile.data["candidate_id"],
                "world_size": WORLD_SIZE,
                "local_batch": LOCAL_BATCH,
                "accumulation_steps": 1,
                "effective_global_batch": EFFECTIVE_BATCH,
                "ddp": {
                    "backend": "nccl",
                    "broadcast_buffers": True,
                    "find_unused_parameters": False,
                    "gradient_as_bucket_view": True,
                    "static_graph": True,
                },
                "measurement": aggregate,
                "sampler": sampler,
                "rank_state_before_checkpoint_canonicalization": pre_agreement,
                "bn_rank_diagnostics": bn_diagnostics,
                "canonical_checkpoint_rank_agreement": canonical_agreement,
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "wall_seconds_including_rank_rng_sidecars_and_hash": (
                        checkpoint_wall
                    ),
                    "rank_profiles": [
                        {
                            "path": str(output / "ranks" / f"profile_rank{value}.json"),
                            "sha256": sha256_file(output / "ranks" / f"profile_rank{value}.json"),
                        }
                        for value in range(WORLD_SIZE)
                    ],
                },
                "D_select_executed": False,
                "D_audit_executed": False,
                "official_validation_executed": False,
                "capability_metrics": False,
            }
            _atomic_json(output / "profile.json", profile_result)
    finally:
        bundle.close()


def _resume(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    source_config, profile, config = _load_contract(args)
    output = _validated_output_dir(
        mode="resume",
        output_dir=args.output_dir,
        source_sha=args.source_sha,
        approved_source_sha=args.approved_source_sha,
    )
    _require((output / "profile.json").is_file(), "DDP profile is absent")
    _require(not (output / "result.json").exists(), "terminal DDP result already exists")
    os.environ["S10_DDP_OUTPUT_DIR"] = str(output)
    source = _source_identity(args.source_sha, args.approved_source_sha)
    seed_everything(int(config.as_dict()["training"]["seed"]))
    enforce_determinism(strict=False, precision="fp16")
    model, ddp, criterion, optimizer, scheduler, scaler, runtime = _build_ddp_stack(
        config, device, rank, "resume"
    )
    bundle = build_phase1_train_data(
        config, distributed_rank=rank, distributed_world_size=WORLD_SIZE
    )
    try:
        checkpoint = output / "checkpoint" / "checkpoint.pt"
        reference_path = output / "checkpoint" / f"continuation_reference_rank{rank}.pt"
        rng_path = output / "checkpoint" / f"rng_rank{rank}.pt"
        profile_rank = _read_json(output / "ranks" / f"profile_rank{rank}.json")
        _require(sha256_file(checkpoint) == profile_rank["checkpoint"]["sha256"], "checkpoint hash drift")
        _require(sha256_file(reference_path) == profile_rank["continuation_reference"]["sha256"], "continuation reference hash drift")
        _require(sha256_file(rng_path) == profile_rank["checkpoint"]["rng_sidecar_sha256"], "RNG sidecar hash drift")
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        rank_rng = torch.load(rng_path, map_location="cpu", weights_only=False)
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
        restore_rng_state(rank_rng)
        _require(identity == config.sha256, "fresh DDP checkpoint identity drift")
        restored_stack = capture_stack(model, optimizer, scheduler, scaler)
        restored_boundary = compare_stacks(
            reference["boundary_stack"], restored_stack, rtol=0.0, atol=0.0
        )
        _require(
            state.checkpoint_dict() == reference["boundary_training_state"],
            "fresh DDP boundary training state drift",
        )
        bundle.set_epoch(1)
        resumed_inputs: list[str] = []
        _run_segment(
            ddp_model=ddp,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            bundle=bundle,
            state=state,
            config=config,
            device=device,
            windows=CONTINUATION_WINDOWS,
            readiness=False,
            digests=resumed_inputs,
        )
        _canonicalize_model(model)
        resumed_stack = capture_stack(model, optimizer, scheduler, scaler)
        fresh_process = {
            "continuation": compare_stacks(
                reference["control_stack"],
                resumed_stack,
                rtol=float(profile.data["parity"]["fp16"]["rtol"]),
                atol=float(profile.data["parity"]["fp16"]["atol"]),
            ),
            "input_stream": {
                "reference_sha256": reference["control_input_sha256"],
                "resumed_sha256": resumed_inputs,
                "exact_equal": reference["control_input_sha256"] == resumed_inputs,
            },
            "training_state_equal": (
                reference["control_training_state"] == state.checkpoint_dict()
            ),
            "rng_state_equal": reference["control_rng_sha256"] == rng_sha256(),
        }
        fresh_process["elementwise_allclose_diagnostic_pass"] = bool(
            all(value["gate_pass"] for value in fresh_process["continuation"].values())
            and fresh_process["input_stream"]["exact_equal"]
            and fresh_process["training_state_equal"]
            and fresh_process["rng_state_equal"]
        )
        gate = evaluate_calibrated_continuation_gate(
            restored_boundary=restored_boundary,
            same_process=profile_rank["same_process_replay"],
            fresh_process=fresh_process,
            relative_l2_tolerance=float(profile.data["parity"]["fp16"]["rtol"]),
            max_absolute_tolerance=float(profile.data["parity"]["fp16"]["atol"]),
        )
        final_identity = _rank_state_identity(model, optimizer, scheduler, scaler, state)
        gathered_final = _gather_identity(final_identity)
        final_agreement = _agreement(gathered_final) if rank == 0 else None
        resume_record = {
            "schema": "s10.phase1p.ip-e5-ddp-rank-resume.v1",
            "rank": rank,
            "source": source,
            "attempt": _attempt_identity(rank),
            "runtime_optimizations": runtime,
            "restored_boundary": restored_boundary,
            "fresh_process": fresh_process,
            "continuation_gate": gate,
            "gate_pass": bool(gate["gate_pass"]),
        }
        _atomic_json(output / "ranks" / f"resume_rank{rank}.json", resume_record)
        dist.barrier()
        if rank == 0:
            profile_result = _read_json(output / "profile.json")
            resumes = [
                _read_json(output / "ranks" / f"resume_rank{value}.json")
                for value in range(WORLD_SIZE)
            ]
            hard_checks = {
                "profile_measurement": bool(profile_result["measurement"]["gate_pass"]),
                "sampler_union": bool(profile_result["sampler"]["gate_pass"]),
                "pre_checkpoint_rank_state": bool(
                    profile_result["rank_state_before_checkpoint_canonicalization"]["gate_pass"]
                ),
                "canonical_checkpoint_rank_state": bool(
                    profile_result["canonical_checkpoint_rank_agreement"]["gate_pass"]
                ),
                "fresh_checkpoint_continuation": all(
                    bool(record["gate_pass"]) for record in resumes
                ),
                "fresh_terminal_rank_state": bool(
                    final_agreement and final_agreement["gate_pass"]
                ),
                "D_select_forbidden": profile_result["D_select_executed"] is False,
                "D_audit_forbidden": profile_result["D_audit_executed"] is False,
                "official_validation_forbidden": profile_result["official_validation_executed"] is False,
                "capability_metrics_forbidden": profile_result["capability_metrics"] is False,
            }
            result = {
                "schema": SCHEMA,
                "status": (
                    "COMPLETE_DDP_ENGINEERING"
                    if all(hard_checks.values())
                    else "COMPLETE_DDP_HARD_GATE_FAILURE"
                ),
                "source": source,
                "attempt": _attempt_identity(rank),
                "source_config_file_sha256": sha256_file(args.config),
                "source_config_sha256": source_config.sha256,
                "effective_config_sha256": config.sha256,
                "profile_file_sha256": sha256_file(args.profile_config),
                "profile_sha256": profile.sha256,
                "profile_artifact_sha256": sha256_file(output / "profile.json"),
                "candidate_id": profile.data["candidate_id"],
                "world_size": WORLD_SIZE,
                "local_batch": LOCAL_BATCH,
                "accumulation_steps": 1,
                "effective_global_batch": EFFECTIVE_BATCH,
                "measurement": profile_result["measurement"],
                "sampler": profile_result["sampler"],
                "bn_rank_diagnostics": profile_result["bn_rank_diagnostics"],
                "checkpoint": {
                    "path": profile_result["checkpoint"]["path"],
                    "sha256": profile_result["checkpoint"]["sha256"],
                    "wall_seconds_including_rank_rng_sidecars_and_hash": (
                        profile_result["checkpoint"][
                            "wall_seconds_including_rank_rng_sidecars_and_hash"
                        ]
                    ),
                    "rank_resume": [
                        {
                            "path": str(output / "ranks" / f"resume_rank{value}.json"),
                            "sha256": sha256_file(output / "ranks" / f"resume_rank{value}.json"),
                            "gate_pass": bool(resumes[value]["gate_pass"]),
                        }
                        for value in range(WORLD_SIZE)
                    ],
                    "terminal_rank_agreement": final_agreement,
                },
                "hard_gates": {"checks": hard_checks, "gate_pass": all(hard_checks.values())},
                "bn_rng_recipe_status": "MEASUREMENT_ONLY_OWNER_DECISION_PENDING",
                "D_select_executed": False,
                "D_audit_executed": False,
                "official_validation_executed": False,
                "capability_metrics": False,
                "interpretation_limits": [
                    "D_fit-only DDP engineering throughput and health",
                    "no capability, mAP, NDS, generalization, or candidate-selection claim",
                    "ordinary per-rank B16 BN/worker-RNG behavior is not production-promoted",
                    "four-GPU DDP is outside this envelope",
                ],
            }
            result_sha = _atomic_json(output / "result.json", result)
            _atomic_json(
                output / "complete.json",
                {
                    "schema": "s10.phase1p.ip-e5-ddp-complete.v1",
                    "status": result["status"],
                    "result_sha256": result_sha,
                },
            )
    finally:
        bundle.close()


def _smoke(args: argparse.Namespace, rank: int, device: torch.device) -> None:
    _load_contract(args)
    output = _validated_output_dir(
        mode="smoke",
        output_dir=args.output_dir,
        source_sha=args.source_sha,
        approved_source_sha=args.approved_source_sha,
    )
    if rank == 0:
        _require(not output.exists(), f"fresh DDP smoke output exists: {output}")
        output.mkdir(parents=True)
    dist.barrier()
    _source_identity(args.source_sha, args.approved_source_sha)
    torch.manual_seed(20260721)
    model = torch.nn.Linear(4, 3, bias=True).to(device)
    ddp = DistributedDataParallel(model, device_ids=[device.index])
    inputs = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
    inputs = inputs + float(rank)
    ddp(inputs).sum().backward()
    gradient_hash = _mapping_sha256(
        {name: parameter.grad for name, parameter in model.named_parameters()}
    )
    gathered = _gather_identity({"gradient_sha256": gradient_hash})
    global_true = _distributed_boolean_and("smoke", True, device)
    if rank == 0:
        checks = {
            "world_size_two": dist.get_world_size() == WORLD_SIZE,
            "rank_gradients_equal": gathered[0] == gathered[1],
            "boolean_collective": global_true,
            "two_distinct_devices": torch.cuda.device_count() == WORLD_SIZE,
        }
        _atomic_json(
            output / "smoke.json",
            {
                "schema": "s10.phase1p.ip-e5-ddp-smoke.v1",
                "checks": checks,
                "gate_pass": all(checks.values()),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "profile", "resume"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--approved-source-sha", required=True)
    args = parser.parse_args()
    for name in ("source_sha", "approved_source_sha"):
        value = getattr(args, name)
        _require(len(value) == 40 and all(c in "0123456789abcdef" for c in value), f"{name} is invalid")
    _require(
        args.approved_source_sha == EXPECTED_APPROVED_SOURCE_SHA,
        "IP-E5 design-approval anchor drift",
    )

    dist.init_process_group(backend="nccl")
    rank = -1
    try:
        rank, _, _, device = _rank_context()
        if args.mode == "smoke":
            _smoke(args, rank, device)
        elif args.mode == "profile":
            _profile(args, rank, device)
        else:
            _resume(args, rank, device)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        gc.collect()


if __name__ == "__main__":
    main()
