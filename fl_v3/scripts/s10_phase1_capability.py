#!/usr/bin/env python3
"""Exact S10 Phase-I Envelope-B branch qualification runner.

The runner owns one and only one frozen branch candidate.  It trains the
candidate for the resolved 20-epoch D_fit exposure, retains one epoch-boundary
recovery checkpoint, and evaluates the terminal raw checkpoint on D_select
exactly once.  D_audit and official validation have no code path here.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

sys.path.insert(0, "fl_v3/src")

import torch

from fl_v3.config import load_resolved_config
from fl_v3.config.phase1 import REFERENCE_OBJECT_CLASSES, phase1_runtime_ready
from fl_v3.data.nuscenes import paths as nuscenes_paths
from fl_v3.data.nuscenes.phase1 import (
    build_phase1_eval_data,
    build_phase1_train_data,
)
from fl_v3.eval.detection_eval import build_results_dict, decode_eval_set
from fl_v3.eval.subset_detection_eval import (
    DETECTION_CONFIG_SHA256,
    run_internal_manifest_eval,
    write_strict_json,
)
from fl_v3.models.phase1_camera import build_phase1_camera_model
from fl_v3.models.phase1_lidar import build_phase1_lidar_model
from fl_v3.models.phase1_swin import sha256_file, tensor_state_sha256
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import _float_tensors, _move_to_device, train_one_epoch
from fl_v3.training.phase1 import Phase1CyclicScheduler, build_phase1_optimizer
from fl_v3.training.runtime_state import TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_autocast_context,
    seed_everything,
    verify_runtime_dependency_identity,
)


SCHEMA = "s10.phase1.envelope-b-capability.v1"
EXPECTED_BRANCH = "codex/s10-phase1-branch-qualification"


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


def _atomic_write_once(path: Path, payload: Any) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable record {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale record partial exists: {temporary}")
    encoded = _canonical_bytes(payload) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _atomic_write_bytes_once(path: Path, payload: bytes) -> str:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale artifact partial exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object at {path}")
    return value


def _source_identity(expected_sha: str) -> dict[str, str]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()

    actual = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    branch = git("branch", "--show-current")
    _require(actual == expected_sha, f"source SHA drift: {actual} != {expected_sha}")
    _require(branch == EXPECTED_BRANCH, f"source branch drift: {branch}")
    return {"git_sha": actual, "git_tree": tree, "branch": branch}


def _runtime_identity(config) -> tuple[dict[str, Any], str]:
    _require(platform.machine() == "aarch64", "Envelope B requires an aarch64 GH200 node")
    _require(torch.cuda.is_available(), "Envelope B requires CUDA")
    _require(torch.cuda.device_count() == 1, "Envelope B requires exactly one visible GPU")
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    _require("GH200" in name, f"Envelope B requires GH200, got {name!r}")
    _require(capability == (9, 0), f"unexpected GH200 compute capability {capability}")
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
            "device_name": name,
            "compute_capability": list(capability),
            "total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "torch_cuda": str(torch.version.cuda),
            "dependencies": compact_dependencies,
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


def _make_scaler(config, device: torch.device) -> torch.amp.GradScaler:
    spec = config.as_dict()["precision"]["grad_scaler"]
    return torch.amp.GradScaler(
        "cuda",
        enabled=bool(spec["enabled"] and device.type == "cuda"),
        init_scale=float(spec["init_scale"]),
        growth_factor=float(spec["growth_factor"]),
        backoff_factor=float(spec["backoff_factor"]),
        growth_interval=int(spec["growth_interval"]),
    )


def _build_components(config, branch: str, device: torch.device):
    seed = int(config.as_dict()["training"]["seed"])
    seed_everything(seed)
    if branch == "camera":
        # O-150: no override is intentional. Schema v2 dispatches only to the
        # qualified PyTorch sorted segment-reduce production backend.
        model = build_phase1_camera_model(config)
        _require(model.view_transform.pool_backend == "fallback", "Camera backend drift")
    else:
        model = build_phase1_lidar_model(config)
    model = model.to(device)
    criterion = model.build_criterion().to(device)
    optimizer = build_phase1_optimizer(model, config)
    scheduler = Phase1CyclicScheduler(optimizer, config)
    scaler = _make_scaler(config, device)
    return model, criterion, optimizer, scheduler, scaler


def _decoded_schema(decoded: list[Mapping[str, torch.Tensor]], batch_size: int) -> dict[str, Any]:
    _require(len(decoded) == batch_size, "preflight decode batch length drift")
    counts: list[int] = []
    for index, sample in enumerate(decoded):
        _require(set(sample) >= {"boxes", "scores", "labels", "velocity"}, "decode fields missing")
        boxes = sample["boxes"]
        scores = sample["scores"]
        labels = sample["labels"]
        velocity = sample["velocity"]
        count = int(boxes.shape[0])
        _require(boxes.shape == (count, 7), f"decode boxes shape drift at sample {index}")
        _require(scores.shape == labels.shape == (count,), "decode vector shape drift")
        _require(velocity.shape == (count, 2), "decode velocity shape drift")
        _require(count <= 500, "decode exceeds the official per-sample cap")
        _require(bool(torch.isfinite(boxes).all()), "decode boxes are nonfinite")
        _require(bool(torch.isfinite(scores).all()), "decode scores are nonfinite")
        _require(bool(torch.isfinite(velocity).all()), "decode velocity is nonfinite")
        if count:
            _require(bool(((labels >= 0) & (labels < 10)).all()), "decode label drift")
        counts.append(count)
    return {"batch_size": batch_size, "prediction_counts": counts}


def _preflight(
    config,
    branch: str,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """One output-neutral D_fit batch plus zero-boundary checkpoint round trip."""
    model, criterion, optimizer, scheduler, scaler = _build_components(
        config, branch, device
    )
    state = TrainingState()
    initial_state_sha = tensor_state_sha256(model.state_dict())
    checkpoint = output_dir / ".preflight-zero.pt"
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

    bundle = build_phase1_train_data(config)
    bundle.set_epoch(0)
    iterator = iter(bundle.loader)
    try:
        batch = next(iterator)
        _require(int(batch["batch_size"]) == 4, "preflight requires physical B4")
        tokens = list(batch["sample_token"])
        moved = _move_to_device(batch, device)
        model.train()
        with precision_autocast_context("fp16", device):
            output = model(moved)
        output_fp32 = _float_tensors(output)
        loss = criterion(output_fp32, moved)
        _require(bool(torch.isfinite(loss.detach())), "preflight loss is nonfinite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        _require(gradients, "preflight produced no gradients")
        _require(
            all(bool(torch.isfinite(gradient).all()) for gradient in gradients),
            "preflight gradients are nonfinite",
        )
        decoded = model.decode(
            output_fp32,
            score_threshold=float(config.as_dict()["model"]["head"]["test"]["score_threshold"]),
        )
        decode_schema = _decoded_schema(decoded, 4)
        loss_value = float(loss.detach().cpu())
    finally:
        del iterator
        bundle.close()

    # Reconstruct every mutable training component before loading the saved
    # zero boundary. This also clears GradScaler's per-optimizer transient state.
    del model, criterion, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    model, criterion, optimizer, scheduler, scaler = _build_components(
        config, branch, device
    )
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
    _require(identity == config.sha256, "preflight checkpoint identity drift")
    _require(loaded == state, "preflight checkpoint state drift")
    _require(
        tensor_state_sha256(model.state_dict()) == initial_state_sha,
        "preflight checkpoint failed to restore the initial model state",
    )
    checkpoint.unlink()
    record = {
        "schema": "s10.phase1.envelope-b-preflight.v1",
        "role": "D_fit",
        "physical_batch": 4,
        "sample_tokens": tokens,
        "optimizer_updates": 0,
        "loss_finite": True,
        "diagnostic_loss": loss_value,
        "gradients_finite": True,
        "decode": decode_schema,
        "initial_model_state_sha256": initial_state_sha,
        "checkpoint_roundtrip_sha256": checkpoint_sha,
        "checkpoint_deleted_after_roundtrip": True,
        "D_select_executed": False,
        "D_audit_executed": False,
        "official_validation_executed": False,
    }
    del model, criterion, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _epoch_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith("epoch_") and path.name[6:].isdigit():
            out.append(path)
    return sorted(out)


def _load_recovery(
    root: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
) -> tuple[TrainingState, Path | None]:
    epochs = _epoch_directories(root)
    _require(len(epochs) <= 2, "retained recovery checkpoint count exceeds recoverable bounds")
    if not epochs:
        return TrainingState(), None
    if len(epochs) == 2:
        lower = int(epochs[0].name[6:])
        upper = int(epochs[1].name[6:])
        _require(
            upper == lower + 1,
            "multiple retained recovery checkpoints are not consecutive",
        )
    epoch_dir = epochs[-1]
    record = _read_json(epoch_dir / "epoch_record.json")
    checkpoint = epoch_dir / "checkpoint.pt"
    _require(checkpoint.is_file(), "recovery checkpoint is missing")
    _require(sha256_file(checkpoint) == record["checkpoint_sha256"], "recovery checkpoint hash drift")
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
    _require(identity == config.sha256, "recovery checkpoint identity drift")
    _require(state.epoch == int(record["epoch"]), "recovery epoch record drift")
    _require(epoch_dir.name == f"epoch_{state.epoch:02d}", "recovery directory name drift")
    # A termination between atomic publication of the new epoch directory and
    # deletion of the prior recovery can leave exactly two consecutive, valid
    # generations. Loading the newer generation proves it is complete before
    # restoring the configured retained-count-one invariant.
    for stale in epochs[:-1]:
        shutil.rmtree(stale)
    return state, epoch_dir


def _commit_epoch(
    root: Path,
    previous: Path | None,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    state: TrainingState,
    config,
    metrics: Mapping[str, Any],
    elapsed_seconds: float,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"epoch_{state.epoch:02d}"
    partial = root / f".epoch_{state.epoch:02d}.partial"
    _require(not final.exists(), f"epoch transaction already exists: {final}")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
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
        "schema": "s10.phase1.envelope-b-epoch.v1",
        "epoch": state.epoch,
        "terminal": state.epoch == int(config.as_dict()["checkpointing"]["terminal_epoch"]),
        "selectable": state.epoch == int(config.as_dict()["checkpointing"]["terminal_epoch"]),
        "elapsed_seconds": float(elapsed_seconds),
        "metrics": dict(metrics),
        "training_state": state.checkpoint_dict(),
        "model_state_sha256": tensor_state_sha256(model.state_dict()),
        "checkpoint_sha256": checkpoint_sha,
    }
    _atomic_write_once(partial / "epoch_record.json", record)
    os.replace(partial, final)
    if previous is not None and previous != final:
        _require(previous.parent == root and previous.name.startswith("epoch_"), "unsafe recovery cleanup target")
        shutil.rmtree(previous)
    _require(len(_epoch_directories(root)) == 1, "recovery retention invariant failed")
    return final


def _source_identities(config) -> dict[str, str]:
    identities = config.data_identities
    return {
        "train_cache_logical_sha256": identities["train_cache_logical_sha256"],
        "train_cache_pickle_sha256": identities["train_cache_pickle_sha256"],
        "train_cache_sidecar_sha256": identities["train_cache_sidecar_sha256"],
        "zip_manifest_logical_sha256": identities["zip_manifest_logical_sha256"],
        "zip_manifest_file_sha256": identities["zip_manifest_file_sha256"],
    }


def _evaluate_terminal(
    config,
    model,
    device: torch.device,
    output_dir: Path,
    checkpoint_dir: Path,
    runtime_dependency_sha256: str,
) -> dict[str, Any]:
    complete = output_dir / "evaluation" / "complete"
    if complete.exists():
        return _read_json(complete / "evaluation_record.json")
    partial = output_dir / "evaluation" / ".in_progress"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    checkpoint = checkpoint_dir / "checkpoint.pt"
    checkpoint_sha = sha256_file(checkpoint)
    bundle = build_phase1_eval_data(config, role="D_select")
    tokens = list(bundle.dataset.sample_tokens)
    expected = config.as_dict()["data"]["roles"]["select"]
    _require(len(tokens) == int(expected["samples"]), "D_select sample count drift")
    _require(_canonical_sha256(tokens) == expected["sample_tokens_sha256"], "D_select token order drift")
    started = time.perf_counter()
    try:
        run_config = config.to_run_config()
        decodes = decode_eval_set(model, bundle.loader, device, run_config, timing=None)
        _require([item.sample_token for item in decodes] == tokens, "D_select decode order drift")
        submission = build_results_dict(
            decodes,
            REFERENCE_OBJECT_CLASSES,
            tokens,
            run_config={
                "model-mode": config.model_mode,
                "checkpoint-weights": "raw",
            },
        )
        submission["fl_v3_phase1_provenance"] = {
            "schema": SCHEMA,
            "resolved_config_sha256": config.sha256,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_weights": "raw",
            "runtime_dependencies_sha256": runtime_dependency_sha256,
            "detection_config_sha256": DETECTION_CONFIG_SHA256,
            "role": "D_select",
            "sample_tokens_sha256": expected["sample_tokens_sha256"],
            "D_audit_executed": False,
            "official_validation_executed": False,
        }
        result_path = partial / "D_select_results.json"
        write_strict_json(str(result_path), submission)
        nusc = nuscenes_paths.create_nuscenes(
            config.as_dict()["data"]["version"],
            config.as_dict()["data"]["dataroot"],
            verbose=False,
        )
        metrics_path = partial / "D_select_metrics.json"
        metrics = run_internal_manifest_eval(
            nusc,
            str(result_path),
            config.as_dict()["data"]["split_manifest"]["path"],
            "D_select",
            str(metrics_path),
            expected_manifest_sha256=config.as_dict()["data"]["split_manifest"]["sha256"],
            expected_parent_version="v1.0-trainval",
            expected_parent_split="train",
            expected_source_identities=_source_identities(config),
        )
        elapsed = time.perf_counter() - started
        record = {
            "schema": "s10.phase1.envelope-b-evaluation.v1",
            "role": "D_select",
            "completed_executions": 1,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_weights": "raw",
            "sample_count": len(tokens),
            "sample_tokens_sha256": expected["sample_tokens_sha256"],
            "result_sha256": sha256_file(result_path),
            "metrics_sha256": sha256_file(metrics_path),
            "elapsed_seconds": elapsed,
            "internal_subset_mAP": metrics["internal_subset_mAP"],
            "internal_subset_NDS": metrics["internal_subset_NDS"],
            "D_audit_executed": False,
            "official_validation_executed": False,
        }
        _atomic_write_once(partial / "evaluation_record.json", record)
        os.replace(partial, complete)
        return record
    finally:
        bundle.close()


def _validate_run_identity(
    identity: Mapping[str, Any],
    *,
    branch: str,
    config,
    source: Mapping[str, str],
    runtime_dependency_sha256: str,
) -> None:
    expected = {
        "schema": SCHEMA,
        "branch": branch,
        "candidate_id": config.as_dict()["contract"]["candidate_id"],
        "source": dict(source),
        "resolved_config_sha256": config.sha256,
        "runtime_dependencies_sha256": runtime_dependency_sha256,
        "seed": int(config.as_dict()["training"]["seed"]),
        "camera_pool_backend": "pytorch_sorted_segment_reduce" if branch == "camera" else None,
    }
    for key, value in expected.items():
        _require(identity.get(key) == value, f"run identity drift at {key}")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_resolved_config(args.config)
    raw = config.as_dict()
    branch = str(raw["contract"]["branch"])
    _require(branch == args.branch, "CLI/config branch mismatch")
    _require(raw["execution"]["mode"] == "phase1_train_eval", "Envelope-B execution mode drift")
    _require(raw["contract"]["lifecycle"] == "envelope_b_ready", "Envelope-B lifecycle drift")
    phase1_runtime_ready(raw)

    output_dir = Path(args.output_dir).resolve()
    configured_root = Path(raw["execution"]["output_root"]).resolve()
    _require(output_dir.parent == configured_root, "output directory is outside the frozen Envelope-B root")
    _require(output_dir.name == raw["contract"]["candidate_id"], "output candidate name drift")
    if output_dir.exists():
        _require(args.resume, "fresh Envelope-B execution refuses an existing output directory")
    else:
        _require(not args.resume, "resume requested but the output directory is absent")
        output_dir.mkdir(parents=True)

    source = _source_identity(args.source_sha)
    runtime, runtime_dependency_sha = _runtime_identity(config)
    device = torch.device("cuda", 0)
    enforce_determinism(strict=False, precision="fp16")
    seed_everything(int(raw["training"]["seed"]))

    identity_path = output_dir / "run_identity.json"
    identity = {
        "schema": SCHEMA,
        "branch": branch,
        "candidate_id": raw["contract"]["candidate_id"],
        "source": source,
        "resolved_config_sha256": config.sha256,
        "source_config_file_sha256": sha256_file(args.config),
        "runtime_dependencies_sha256": runtime_dependency_sha,
        "detection_config_sha256": DETECTION_CONFIG_SHA256,
        "runtime": runtime,
        "seed": int(raw["training"]["seed"]),
        "camera_pool_backend": "pytorch_sorted_segment_reduce" if branch == "camera" else None,
        "unpromoted_optional_backend_executed": False,
        "D_fit": raw["data"]["roles"]["fit"],
        "D_select": raw["data"]["roles"]["select"],
        "D_audit": {**raw["data"]["roles"]["audit"], "executed": False},
    }
    if identity_path.exists():
        _validate_run_identity(
            _read_json(identity_path),
            branch=branch,
            config=config,
            source=source,
            runtime_dependency_sha256=runtime_dependency_sha,
        )
    else:
        _atomic_write_once(identity_path, identity)
        resolved_path = output_dir / "resolved_config.json"
        _atomic_write_bytes_once(resolved_path, config.canonical_bytes)
        _require(sha256_file(resolved_path) == config.sha256, "resolved config physical hash drift")

    attempt = _attempt_identity()
    attempt_key = (
        f"{attempt['slurm_job_id']}-r{attempt['slurm_restart_count']}"
        if attempt["slurm_job_id"]
        else f"pid-{attempt['pid']}"
    )
    attempt_start = output_dir / "attempts" / f"{attempt_key}.start.json"
    if not attempt_start.exists():
        _atomic_write_once(
            attempt_start,
            {**attempt, "started_unix_seconds": time.time(), "resume": bool(args.resume)},
        )

    result_path = output_dir / "result.json"
    if result_path.exists():
        result = _read_json(result_path)
        _require(result.get("status", "").startswith("COMPLETE"), "existing terminal result is not complete")
        return result

    preflight_path = output_dir / "preflight.json"
    if preflight_path.exists():
        preflight = _read_json(preflight_path)
    else:
        preflight = _preflight(config, branch, device, output_dir)
        _atomic_write_once(preflight_path, preflight)

    model, criterion, optimizer, scheduler, scaler = _build_components(
        config, branch, device
    )
    train_bundle = build_phase1_train_data(config)
    try:
        state, recovery_dir = _load_recovery(
            output_dir / "epochs",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
        )
        if recovery_dir is None:
            _require(
                tensor_state_sha256(model.state_dict())
                == preflight["initial_model_state_sha256"],
                "fresh production initialization differs from preflight",
            )
        terminal_epoch = int(raw["checkpointing"]["terminal_epoch"])
        microbatches_per_epoch = int(raw["training"]["consumed_samples_per_epoch"]) // int(
            raw["training"]["micro_batch_size"]
        )
        _require(len(train_bundle.loader) == microbatches_per_epoch, "D_fit loader length drift")
        _require(
            microbatches_per_epoch % int(raw["training"]["accumulation_steps"]) == 0,
            "D_fit epoch is not accumulation-window aligned",
        )
        while state.epoch < terminal_epoch:
            epoch_index = state.epoch
            before = state.checkpoint_dict()
            train_bundle.set_epoch(epoch_index)
            started = time.perf_counter()
            metrics = train_one_epoch(
                model,
                train_bundle.loader,
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
                max_optimizer_steps=0,
                model_mode=config.model_mode,
                exposure_multiplier=1,
                expected_global_microbatch_samples=int(raw["training"]["micro_batch_size"]),
                precision_diagnostics=None,
                readiness_timing=False,
            )
            elapsed = time.perf_counter() - started
            state.epoch = epoch_index + 1
            expected_samples = int(raw["training"]["consumed_samples_per_epoch"])
            expected_windows = int(raw["training"]["optimizer_updates_per_epoch"])
            _require(
                state.attempted_samples - int(before["attempted_samples"]) == expected_samples,
                "epoch attempted-sample exposure drift",
            )
            _require(
                state.attempted_windows - int(before["attempted_windows"]) == expected_windows,
                "epoch attempted-window exposure drift",
            )
            _require(
                state.attempted_microbatches - int(before["attempted_microbatches"])
                == microbatches_per_epoch,
                "epoch microbatch exposure drift",
            )
            _require(state.discarded_windows == 0, "production epoch discarded a partial window")
            recovery_dir = _commit_epoch(
                output_dir / "epochs",
                recovery_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                state=state,
                config=config,
                metrics=metrics,
                elapsed_seconds=elapsed,
            )
        _require(recovery_dir is not None, "terminal checkpoint is missing")
        _require(state.epoch == terminal_epoch, "terminal epoch drift")
        expected_total_samples = terminal_epoch * int(raw["training"]["consumed_samples_per_epoch"])
        expected_total_windows = terminal_epoch * int(raw["training"]["optimizer_updates_per_epoch"])
        _require(state.attempted_samples == expected_total_samples, "terminal attempted-sample exposure drift")
        _require(state.attempted_windows == expected_total_windows, "terminal attempted-window exposure drift")
    finally:
        train_bundle.close()

    evaluation = _evaluate_terminal(
        config,
        model,
        device,
        output_dir,
        recovery_dir,
        runtime_dependency_sha,
    )
    terminal_record = _read_json(recovery_dir / "epoch_record.json")
    status = "COMPLETE" if state.invalid_windows == 0 else "COMPLETE_WITH_INVALID_WINDOWS"
    result = {
        "schema": SCHEMA,
        "status": status,
        "branch": branch,
        "candidate_id": raw["contract"]["candidate_id"],
        "source": source,
        "resolved_config_sha256": config.sha256,
        "seed": int(raw["training"]["seed"]),
        "camera_pool_backend": "pytorch_sorted_segment_reduce" if branch == "camera" else None,
        "unpromoted_optional_backend_executed": False,
        "training": {
            "epochs": terminal_epoch,
            "physical_batch": int(raw["training"]["micro_batch_size"]),
            "accumulation_steps": int(raw["training"]["accumulation_steps"]),
            "effective_batch": int(raw["training"]["effective_global_batch"]),
            "attempted_samples": state.attempted_samples,
            "attempted_windows": state.attempted_windows,
            "accepted_updates": state.optimizer_step,
            "invalid_windows": state.invalid_windows,
            "nonfinite_windows": state.nonfinite_windows,
            "overflow_windows": state.overflow_windows,
            "terminal_state": state.checkpoint_dict(),
        },
        "terminal_checkpoint": {
            "path": str((recovery_dir / "checkpoint.pt").resolve()),
            "sha256": terminal_record["checkpoint_sha256"],
            "model_state_sha256": terminal_record["model_state_sha256"],
            "weights": "raw",
            "selectable": True,
        },
        "D_select": evaluation,
        "D_audit_executed": False,
        "official_validation_executed": False,
        "interpretation_limits": [
            "single-seed internal train-only branch qualification",
            "D_select is not official nuScenes validation",
            "no Camera CUDA-backend promotion claim",
            "no fusion or federated-learning capability claim",
        ],
    }
    _atomic_write_once(result_path, result)
    attempt_end = output_dir / "attempts" / f"{attempt_key}.end.json"
    if not attempt_end.exists():
        _atomic_write_once(
            attempt_end,
            {**attempt, "ended_unix_seconds": time.time(), "status": status},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("camera", "lidar"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    result = _run(arguments)
    print(json.dumps({
        "status": result["status"],
        "branch": result["branch"],
        "resolved_config_sha256": result["resolved_config_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
