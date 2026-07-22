#!/usr/bin/env python3
"""Production S10 Phase-I Camera qualification on one node and two GH200s.

The scientific unit remains one effective global B32 optimizer window.  Each
rank consumes one contiguous B16 half of the frozen global CBGS window, DDP
averages gradients, and all finite/scaler control-flow decisions are collective.
Ordinary rank-local B16 BatchNorm and rank-addressed worker RNG are explicit
parts of the owner-accepted recipe.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "fl_v3/src")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from s10_phase1_capability import (
    _atomic_write_bytes_once,
    _atomic_write_once,
    _build_components,
    _canonical_sha256,
    _decoded_schema,
    _evaluate_terminal,
    _read_json,
    _source_identity,
)
from fl_v3.config import load_resolved_config
from fl_v3.config.phase1 import phase1_runtime_ready
from fl_v3.data.nuscenes.phase1 import build_phase1_train_data
from fl_v3.models.phase1_swin import sha256_file, tensor_state_sha256
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import _float_tensors, _move_to_device, train_one_epoch
from fl_v3.training.phase1_runtime import phase1_runtime_optimization_identity
from fl_v3.training.phase1p_ddp import (
    capture_sha256,
    capture_state,
    compare_state_captures,
    restore_rng_state,
    rng_state,
    rng_state_sha256,
)
from fl_v3.training.runtime_state import TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_autocast_context,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "s10.phase1.envelope-b-camera-ddp.v1"
WORLD_SIZE = 2
LOCAL_BATCH = 16
EFFECTIVE_BATCH = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _distributed_spec(config: Any) -> dict[str, Any]:
    raw = config.as_dict()
    _require(raw["contract"]["branch"] == "camera", "DDP runner is Camera-only")
    _require(raw["schema_version"] == "s10.phase1.v4", "Camera DDP requires v4 config")
    training = raw["training"]
    _require(
        training["world_size"] == WORLD_SIZE
        and training["micro_batch_size"] == LOCAL_BATCH
        and training["accumulation_steps"] == 1
        and training["effective_global_batch"] == EFFECTIVE_BATCH,
        "Camera DDP batch recipe drift",
    )
    return dict(raw["runtime_optimizations"]["distributed_data_parallel"])


def _rank_context(config: Any) -> tuple[int, int, torch.device]:
    spec = _distributed_spec(config)
    _require(platform.machine() == "aarch64", "Camera DDP requires an aarch64 node")
    _require(torch.cuda.is_available(), "Camera DDP requires CUDA")
    _require(torch.cuda.device_count() == WORLD_SIZE, "exactly two GPUs must be visible")
    try:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("Camera DDP must be launched by torchrun") from exc
    _require(world_size == WORLD_SIZE == int(spec["world_size"]), "DDP world-size drift")
    _require(0 <= rank < WORLD_SIZE, "DDP rank drift")
    _require(0 <= local_rank < WORLD_SIZE, "DDP local-rank drift")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=str(spec["backend"]), init_method="env://")
    device = torch.device("cuda", local_rank)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, {"rank": rank, "local_rank": local_rank})
    _require(
        sorted((int(item["rank"]), int(item["local_rank"])) for item in gathered)
        == [(0, 0), (1, 1)],
        "DDP ranks do not map one-to-one onto the two local devices",
    )
    return rank, local_rank, device


def _distributed_boolean_and(name: str, value: bool, device: torch.device) -> bool:
    local = torch.tensor(int(bool(value)), dtype=torch.int32, device=device)
    minimum = local.clone()
    maximum = local.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if name == "optimizer_accepted" and int(minimum.item()) != int(maximum.item()):
        raise RuntimeError("GradScaler accepted-update decision differs across ranks")
    return bool(int(minimum.item()))


def _atomic_torch_once(path: Path, value: Any) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite immutable artifact {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix="s10-camera-ddp-", suffix=".pt", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(path)


def _runtime_identity(config: Any, device: torch.device, rank: int) -> tuple[dict[str, Any], str]:
    devices = []
    for index in range(WORLD_SIZE):
        name = torch.cuda.get_device_name(index)
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
        _require("GH200" in name, f"Camera DDP requires GH200, got {name!r}")
        _require(capability == (9, 0), f"unexpected GH200 compute capability {capability}")
        devices.append(
            {
                "index": index,
                "name": name,
                "compute_capability": list(capability),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
        )
    dependencies = verify_runtime_dependency_identity(config.to_run_config())
    dependency_sha = _canonical_sha256(dependencies)
    compact_dependencies = {
        key: value
        for key, value in dependencies.items()
        if not key.endswith("_executable_artifacts")
        and not key.endswith("_import_origins")
    }
    return (
        {
            "rank": rank,
            "current_device": int(device.index),
            "visible_devices": devices,
            "torch_cuda": str(torch.version.cuda),
            "dependencies": compact_dependencies,
            "dependencies_sha256": dependency_sha,
        },
        dependency_sha,
    )


def _attempt_identity(rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "memory_per_node_mib": os.environ.get("SLURM_MEM_PER_NODE"),
        "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "pid": os.getpid(),
    }


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
    bn_names = {
        name
        for name in model_state
        if name.endswith("running_mean") or name.endswith("running_var")
    }
    return {
        "parameters_sha256": _mapping_sha256(
            {name: model_state[name] for name in sorted(parameter_names)}
        ),
        "non_bn_buffers_sha256": _mapping_sha256(
            {
                name: model_state[name]
                for name in sorted(model_state)
                if name not in parameter_names and name not in bn_names
            }
        ),
        "optimizer_sha256": capture_sha256(capture_state(optimizer.state_dict())),
        "scheduler_sha256": capture_sha256(capture_state(scheduler.state_dict())),
        "scaler_sha256": capture_sha256(capture_state(scaler.state_dict())),
        "training_state": state.checkpoint_dict(),
    }


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
    comparison = compare_state_captures(
        capture_state(gathered[0]), capture_state(gathered[1]), rtol=0.0, atol=0.0
    )
    return {
        "rank0_vs_rank1": comparison["numerical"]["global"],
        "elementwise_exact_diagnostic": bool(comparison["gate_pass"]),
        "tensor_count": len(local),
        "policy": "ordinary_rank_local_b16",
    }


def _canonicalize_model(model: torch.nn.Module) -> None:
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            dist.broadcast(value, src=0)
    dist.barrier()


def _sampler_identity(bundle: Any, rank: int) -> dict[str, Any] | None:
    local = np.asarray(bundle.sampler.rank_epoch_positions(0), dtype=np.int64)
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, local.tolist())
    if rank != 0:
        return None
    arrays = [np.asarray(value, dtype=np.int64) for value in gathered]
    global_positions = np.asarray(bundle.sampler.global_epoch_positions(0), dtype=np.int64)
    recomposed = np.concatenate(
        [value.reshape(-1, LOCAL_BATCH) for value in arrays], axis=1
    ).reshape(-1)
    exact = np.array_equal(recomposed, global_positions)
    unique = len(np.unique(recomposed)) == len(recomposed)
    return {
        "global_presentations": int(global_positions.size),
        "rank_presentations": [int(value.size) for value in arrays],
        "window_union_exact": bool(exact),
        "no_duplicate_or_omission": bool(exact and unique),
        "gate_pass": bool(
            exact
            and unique
            and global_positions.size == 87_904
            and all(value.size == 43_952 for value in arrays)
        ),
    }


def _build_ddp_stack(config: Any, device: torch.device, output: Path, rank: int):
    cache = output / "torchinductor_cache" / f"rank{rank}"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    model, criterion, optimizer, scheduler, scaler = _build_components(
        config, "camera", device
    )
    runtime = phase1_runtime_optimization_identity(model)
    spec = _distributed_spec(config)
    ddp = DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=bool(spec["broadcast_buffers"]),
        find_unused_parameters=bool(spec["find_unused_parameters"]),
        gradient_as_bucket_view=bool(spec["gradient_as_bucket_view"]),
        static_graph=bool(spec["static_graph"]),
    )
    return model, ddp, criterion, optimizer, scheduler, scaler, runtime


def _preflight(config: Any, device: torch.device, output: Path, rank: int) -> dict[str, Any] | None:
    model, ddp, criterion, optimizer, scheduler, scaler, runtime = _build_ddp_stack(
        config, device, output, rank
    )
    state = TrainingState()
    initial_sha = tensor_state_sha256(model.state_dict())
    gathered_initial: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_initial, initial_sha)
    _require(len(set(gathered_initial)) == 1, "DDP initial model differs across ranks")

    checkpoint = output / ".preflight-zero.pt"
    sidecar = output / f".preflight-rng-rank{rank}.pt"
    local_rng = rng_state()
    sidecar_sha = _atomic_torch_once(sidecar, local_rng)
    if rank == 0:
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
    checkpoint_sha = sha256_file(checkpoint)

    bundle = build_phase1_train_data(
        config, distributed_rank=rank, distributed_world_size=WORLD_SIZE
    )
    try:
        bundle.set_epoch(0)
        batch = next(iter(bundle.loader))
        _require(int(batch["batch_size"]) == LOCAL_BATCH, "preflight local batch drift")
        tokens = list(batch["sample_token"])
        moved = _move_to_device(batch, device)
        ddp.train()
        with precision_autocast_context("fp16", device):
            model_output = ddp(moved)
        output_fp32 = _float_tensors(model_output)
        loss = criterion(output_fp32, moved)
        loss_finite = _distributed_boolean_and(
            "loss_finite", bool(torch.isfinite(loss.detach())), device
        )
        _require(loss_finite, "preflight loss is nonfinite on at least one rank")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        local_gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        _require(
            _distributed_boolean_and("gradients_finite", local_gradients_finite, device),
            "preflight gradients are nonfinite on at least one rank",
        )
        decoded = model.decode(
            output_fp32,
            score_threshold=float(
                config.as_dict()["model"]["head"]["test"]["score_threshold"]
            ),
        )
        local_record = {
            "rank": rank,
            "sample_tokens": tokens,
            "diagnostic_loss": float(loss.detach().cpu()),
            "decode": _decoded_schema(decoded, LOCAL_BATCH),
        }
    finally:
        bundle.close()

    gathered_records: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_records, local_record)
    loaded, identity = load_checkpoint(
        str(checkpoint),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        ema=None,
        config=config,
        map_location="cpu",
    )
    restore_rng_state(torch.load(sidecar, map_location="cpu", weights_only=False))
    _require(identity == config.sha256 and loaded == state, "preflight checkpoint state drift")
    _require(
        tensor_state_sha256(model.state_dict()) == initial_sha,
        "preflight checkpoint failed to restore the initial model",
    )
    restored_agreement = _agreement(
        _gather_identity(_rank_state_identity(model, optimizer, scheduler, scaler, loaded))
    )
    _require(restored_agreement["gate_pass"], "preflight restored ranks disagree")
    dist.barrier()
    sidecar.unlink()
    if rank == 0:
        checkpoint.unlink()
    dist.barrier()
    sidecar_hashes = _gather_sidecar_hashes(sidecar_sha)
    result = None
    if rank == 0:
        result = {
            "schema": "s10.phase1.envelope-b-camera-ddp-preflight.v1",
            "role": "D_fit",
            "world_size": WORLD_SIZE,
            "physical_batch_per_rank": LOCAL_BATCH,
            "effective_global_batch": EFFECTIVE_BATCH,
            "runtime_optimizations": runtime,
            "rank_batches": gathered_records,
            "optimizer_updates": 0,
            "loss_finite": True,
            "gradients_finite": True,
            "initial_model_state_sha256": initial_sha,
            "checkpoint_roundtrip_sha256": checkpoint_sha,
            "rank_rng_sidecar_sha256": sidecar_hashes,
            "restored_rank_agreement": restored_agreement,
            "D_select_executed": False,
            "D_audit_executed": False,
            "official_validation_executed": False,
        }
    del ddp, model, criterion, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _gather_sidecar_hashes(value: str) -> list[str]:
    gathered: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered, value)
    return [str(item) for item in gathered]


def _epoch_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("epoch_") and path.name[6:].isdigit()
    )


def _load_recovery(
    root: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: Any,
    rank: int,
) -> tuple[TrainingState, Path | None]:
    epochs = _epoch_directories(root)
    _require(len(epochs) <= 2, "retained recovery checkpoint count exceeds bounds")
    if not epochs:
        return TrainingState(), None
    if len(epochs) == 2:
        _require(
            int(epochs[1].name[6:]) == int(epochs[0].name[6:]) + 1,
            "retained recovery checkpoints are not consecutive",
        )
    epoch_dir = epochs[-1]
    record = _read_json(epoch_dir / "epoch_record.json")
    checkpoint = epoch_dir / "checkpoint.pt"
    _require(sha256_file(checkpoint) == record["checkpoint_sha256"], "checkpoint hash drift")
    sidecar = epoch_dir / f"rng_rank{rank}.pt"
    expected_sidecar = record["rank_rng_sidecars"][rank]
    _require(sha256_file(sidecar) == expected_sidecar["sha256"], "rank RNG sidecar hash drift")
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
    rank_rng = torch.load(sidecar, map_location="cpu", weights_only=False)
    _require(
        rng_state_sha256(rank_rng) == expected_sidecar["state_sha256"],
        "rank RNG state identity drift",
    )
    restore_rng_state(rank_rng)
    _require(identity == config.sha256, "recovery config identity drift")
    _require(state.epoch == int(record["epoch"]), "recovery epoch drift")
    agreement = _agreement(
        _gather_identity(_rank_state_identity(model, optimizer, scheduler, scaler, state))
    )
    _require(agreement["gate_pass"], "restored DDP rank state disagrees")
    dist.barrier()
    if rank == 0:
        for stale in epochs[:-1]:
            shutil.rmtree(stale)
    dist.barrier()
    return state, epoch_dir


def _commit_epoch(
    root: Path,
    previous: Path | None,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    state: TrainingState,
    config: Any,
    local_metrics: Mapping[str, Any],
    elapsed_seconds: float,
    rank: int,
) -> Path:
    final = root / f"epoch_{state.epoch:02d}"
    partial = root / f".epoch_{state.epoch:02d}.partial"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        _require(not final.exists(), f"epoch transaction already exists: {final}")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir()
    dist.barrier()

    local_rng = rng_state()
    sidecar = partial / f"rng_rank{rank}.pt"
    sidecar_record = {
        "rank": rank,
        "path": sidecar.name,
        "sha256": _atomic_torch_once(sidecar, local_rng),
        "state_sha256": rng_state_sha256(local_rng),
    }
    gathered_sidecars: list[Any] = [None] * WORLD_SIZE
    gathered_metrics: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_sidecars, sidecar_record)
    dist.all_gather_object(gathered_metrics, dict(local_metrics))
    bn_diagnostics = _bn_diagnostics(model, rank)

    _canonicalize_model(model)
    agreement = _agreement(
        _gather_identity(_rank_state_identity(model, optimizer, scheduler, scaler, state))
    )
    _require(agreement["gate_pass"], "DDP ranks disagree at checkpoint boundary")

    if rank == 0:
        checkpoint = partial / "checkpoint.pt"
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
        checkpoint_sha = sha256_file(checkpoint)
        record = {
            "schema": "s10.phase1.envelope-b-camera-ddp-epoch.v1",
            "epoch": state.epoch,
            "terminal": state.epoch
            == int(config.as_dict()["checkpointing"]["terminal_epoch"]),
            "selectable": state.epoch
            == int(config.as_dict()["checkpointing"]["terminal_epoch"]),
            "elapsed_seconds": float(elapsed_seconds),
            "rank_metrics": gathered_metrics,
            "training_state": state.checkpoint_dict(),
            "model_state_sha256": tensor_state_sha256(model.state_dict()),
            "checkpoint_sha256": checkpoint_sha,
            "rank_rng_sidecars": gathered_sidecars,
            "ordinary_rank_local_bn_diagnostic": bn_diagnostics,
            "canonical_rank_agreement": agreement,
        }
        _atomic_write_once(partial / "epoch_record.json", record)
        os.replace(partial, final)
    dist.barrier()
    if rank == 0 and previous is not None and previous != final:
        _require(previous.parent == root, "unsafe recovery cleanup target")
        shutil.rmtree(previous)
    dist.barrier()
    _require(len(_epoch_directories(root)) == 1, "recovery retention invariant failed")
    return final


def _validate_identity(identity: Mapping[str, Any], config: Any, dependency_sha: str) -> None:
    expected = {
        "schema": SCHEMA,
        "branch": "camera",
        "candidate_id": config.as_dict()["contract"]["candidate_id"],
        "resolved_config_sha256": config.sha256,
        "runtime_dependencies_sha256": dependency_sha,
        "seed": int(config.as_dict()["training"]["seed"]),
        "distributed_recipe": _distributed_spec(config),
    }
    for key, value in expected.items():
        _require(identity.get(key) == value, f"run identity drift at {key}")


def _run(
    args: argparse.Namespace,
    config: Any,
    rank: int,
    device: torch.device,
) -> dict[str, Any] | None:
    raw = config.as_dict()
    _distributed_spec(config)
    _require(raw["execution"]["mode"] == "phase1_train_eval", "execution mode drift")
    _require(raw["contract"]["lifecycle"] == "envelope_b_ready", "lifecycle drift")
    phase1_runtime_ready(raw)

    output = Path(args.output_dir).resolve()
    configured_root = Path(raw["execution"]["output_root"]).resolve()
    _require(output.parent == configured_root, "output is outside the frozen root")
    _require(output.name == raw["contract"]["candidate_id"], "candidate output name drift")
    exists = output.exists()
    _require(exists == bool(args.resume), "fresh/resume output state drift")
    if rank == 0 and not exists:
        output.mkdir(parents=True)
    dist.barrier()

    source = _source_identity(args.source_sha)
    runtime, dependency_sha = _runtime_identity(config, device, rank)
    gathered_runtime: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(gathered_runtime, runtime)
    _require(
        all(item["dependencies_sha256"] == dependency_sha for item in gathered_runtime),
        "runtime dependency identity differs across ranks",
    )

    seed = int(raw["training"]["seed"])
    seed_everything(seed)
    enforce_determinism(strict=False, precision="fp16")
    identity_path = output / "run_identity.json"
    identity = {
        "schema": SCHEMA,
        "branch": "camera",
        "candidate_id": raw["contract"]["candidate_id"],
        "source": source,
        "resolved_config_sha256": config.sha256,
        "source_config_file_sha256": sha256_file(args.config),
        "runtime_dependencies_sha256": dependency_sha,
        "runtime_by_rank": gathered_runtime,
        "seed": seed,
        "distributed_recipe": _distributed_spec(config),
        "compile_cache_relative_paths": [
            f"torchinductor_cache/rank{value}" for value in range(WORLD_SIZE)
        ],
        "D_fit": raw["data"]["roles"]["fit"],
        "D_select": raw["data"]["roles"]["select"],
        "D_audit": {**raw["data"]["roles"]["audit"], "executed": False},
    }
    if identity_path.exists():
        _validate_identity(_read_json(identity_path), config, dependency_sha)
    elif rank == 0:
        _atomic_write_once(identity_path, identity)
        resolved = output / "resolved_config.json"
        _atomic_write_bytes_once(resolved, config.canonical_bytes)
        _require(sha256_file(resolved) == config.sha256, "resolved config hash drift")
    dist.barrier()

    attempt = _attempt_identity(rank)
    job = attempt["slurm_job_id"] or f"pid-{attempt['pid']}"
    attempt_start = output / "attempts" / f"{job}-rank{rank}.start.json"
    if not attempt_start.exists():
        _atomic_write_once(
            attempt_start,
            {
                **attempt,
                "source": source,
                "resolved_config_sha256": config.sha256,
                "started_unix_seconds": time.time(),
                "resume": bool(args.resume),
            },
        )
    dist.barrier()

    result_path = output / "result.json"
    if result_path.exists():
        result = _read_json(result_path) if rank == 0 else None
        carrier = [result]
        dist.broadcast_object_list(carrier, src=0)
        _require(str(carrier[0]["status"]).startswith("COMPLETE"), "terminal result drift")
        return carrier[0] if rank == 0 else None

    preflight_path = output / "preflight.json"
    if not preflight_path.exists():
        preflight = _preflight(config, device, output, rank)
        if rank == 0:
            _atomic_write_once(preflight_path, preflight)
    dist.barrier()
    preflight = _read_json(preflight_path)

    # Re-seeding makes fresh production initialization independent of preflight
    # diagnostics; resumed invocations immediately replace it from checkpoint.
    seed_everything(seed)
    model, ddp, criterion, optimizer, scheduler, scaler, runtime_optimizations = (
        _build_ddp_stack(config, device, output, rank)
    )
    _require(
        runtime_optimizations == preflight["runtime_optimizations"],
        "production runtime optimizations differ from preflight",
    )
    bundle = build_phase1_train_data(
        config, distributed_rank=rank, distributed_world_size=WORLD_SIZE
    )
    try:
        sampler = _sampler_identity(bundle, rank)
        if rank == 0:
            _require(sampler is not None and sampler["gate_pass"], "CBGS rank union failed")
        state, recovery = _load_recovery(
            output / "epochs",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            rank=rank,
        )
        if recovery is None:
            _require(
                tensor_state_sha256(model.state_dict())
                == preflight["initial_model_state_sha256"],
                "fresh production initialization differs from preflight",
            )
        terminal_epoch = int(raw["checkpointing"]["terminal_epoch"])
        microbatches = int(raw["training"]["consumed_samples_per_epoch"]) // (
            WORLD_SIZE * LOCAL_BATCH
        )
        _require(len(bundle.loader) == microbatches == 2747, "D_fit loader length drift")
        while state.epoch < terminal_epoch:
            epoch_index = state.epoch
            before = state.checkpoint_dict()
            bundle.set_epoch(epoch_index)
            started = time.perf_counter()
            metrics = train_one_epoch(
                ddp,
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
                accumulation_steps=1,
                runtime_state=state,
                max_optimizer_steps=0,
                model_mode=config.model_mode,
                exposure_multiplier=WORLD_SIZE,
                expected_global_microbatch_samples=EFFECTIVE_BATCH,
                precision_diagnostics=None,
                readiness_timing=False,
                distributed_boolean_and=lambda name, value: _distributed_boolean_and(
                    name, value, device
                ),
            )
            elapsed = time.perf_counter() - started
            state.epoch = epoch_index + 1
            _require(
                state.attempted_samples - int(before["attempted_samples"]) == 87_904,
                "epoch global presentation exposure drift",
            )
            _require(
                state.attempted_windows - int(before["attempted_windows"]) == 2_747,
                "epoch optimizer-window exposure drift",
            )
            _require(
                state.attempted_microbatches - int(before["attempted_microbatches"])
                == microbatches,
                "epoch local microbatch exposure drift",
            )
            _require(state.discarded_windows == 0, "production epoch discarded a window")
            recovery = _commit_epoch(
                output / "epochs",
                recovery,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                config=config,
                local_metrics=metrics,
                elapsed_seconds=elapsed,
                rank=rank,
            )
        _require(
            recovery is not None and state.epoch == terminal_epoch,
            "terminal checkpoint missing",
        )
        _require(state.attempted_samples == terminal_epoch * 87_904, "terminal exposure drift")
        _require(state.attempted_windows == terminal_epoch * 2_747, "terminal update-window drift")
    finally:
        bundle.close()

    dist.barrier()
    evaluation = None
    if rank == 0:
        evaluation = _evaluate_terminal(
            config, model, device, output, recovery, dependency_sha
        )
    dist.barrier()

    result = None
    if rank == 0:
        terminal_record = _read_json(recovery / "epoch_record.json")
        status = "COMPLETE" if state.invalid_windows == 0 else "COMPLETE_WITH_INVALID_WINDOWS"
        result = {
            "schema": SCHEMA,
            "status": status,
            "branch": "camera",
            "candidate_id": raw["contract"]["candidate_id"],
            "initial_source": _read_json(identity_path)["source"],
            "terminal_source": source,
            "resolved_config_sha256": config.sha256,
            "seed": seed,
            "distributed_recipe": _distributed_spec(config),
            "runtime_optimizations": runtime_optimizations,
            "training": {
                "epochs": terminal_epoch,
                "world_size": WORLD_SIZE,
                "physical_batch_per_rank": LOCAL_BATCH,
                "accumulation_steps": 1,
                "effective_global_batch": EFFECTIVE_BATCH,
                "attempted_samples": state.attempted_samples,
                "attempted_windows": state.attempted_windows,
                "accepted_updates": state.optimizer_step,
                "invalid_windows": state.invalid_windows,
                "terminal_state": state.checkpoint_dict(),
            },
            "sampler": sampler,
            "terminal_checkpoint": {
                "path": str((recovery / "checkpoint.pt").resolve()),
                "sha256": terminal_record["checkpoint_sha256"],
                "model_state_sha256": terminal_record["model_state_sha256"],
                "rank_rng_sidecars": terminal_record["rank_rng_sidecars"],
                "weights": "raw_rank0_canonical_boundary",
                "selectable": True,
            },
            "D_select": evaluation,
            "D_audit_executed": False,
            "official_validation_executed": False,
            "interpretation_limits": [
                "single-seed internal train-only branch qualification",
                "D_select is not official nuScenes validation",
                "ordinary rank-local B16 BN and rank-addressed worker RNG are recipe inputs",
                "no fusion or federated-learning capability claim",
            ],
        }
        _atomic_write_once(result_path, result)
    attempt_end = output / "attempts" / f"{job}-rank{rank}.end.json"
    if not attempt_end.exists():
        _atomic_write_once(
            attempt_end,
            {
                **attempt,
                "source": source,
                "resolved_config_sha256": config.sha256,
                "ended_unix_seconds": time.time(),
                "status": result["status"] if rank == 0 else "COMPLETE_RANK",
            },
        )
    dist.barrier()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_resolved_config(args.config)
    rank, _, device = _rank_context(config)
    try:
        result = _run(args, config, rank, device)
        if rank == 0 and result is not None:
            print(
                __import__("json").dumps(
                    {
                        "status": result["status"],
                        "branch": result["branch"],
                        "resolved_config_sha256": result["resolved_config_sha256"],
                    },
                    sort_keys=True,
                )
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
