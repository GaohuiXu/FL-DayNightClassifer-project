"""S09 resolved centralized trainer and bounded readiness entry point.

This entry point accepts only the canonical ``s09.v1`` config.  It constructs
one mode-aware loader, maps exact architecture enums to the reviewed stack,
and advances schedules by successful optimizer updates. ``train_eval`` writes one
complete boundary-safe checkpoint and evaluates it. ``readiness`` is explicitly
non-resumable, bounded, checkpoint-free, and evaluation-free. DDP remains fail
closed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Callable

sys.path.insert(0, "fl_v3/src")

import torch

from fl_v3.config import load_resolved_config, verify_physical_data_identities
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import PersistentEpochIterator, TrainingState
from fl_v3.utils.runtime import (
    enforce_determinism,
    make_grad_scaler,
    seed_everything,
    verify_runtime_dependency_identity,
)


def _build_optimizer(model: torch.nn.Module, config) -> torch.optim.Optimizer:
    spec = config.data["optimizer"]
    cls = torch.optim.Adam if spec["name"] == "adam" else torch.optim.AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("resolved model has no trainable parameters")
    return cls(params, lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))


def _build_ema(model: torch.nn.Module, decay):
    if decay is None:
        return None
    from torch.optim.swa_utils import AveragedModel

    def average(old, new, _count, d=float(decay)):
        return d * old + (1.0 - d) * new

    return AveragedModel(model, avg_fn=average, use_buffers=False)


def _checkpoint_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing sample")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("loader timing distribution is empty")
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _digest_batch_value(digest, value: object) -> None:
    """Content-address one bounded production batch without retaining it."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        return
    if isinstance(value, dict):
        digest.update(f"dict:{len(value)}\0".encode("ascii"))
        for key in sorted(value):
            digest.update(str(key).encode("utf-8") + b"\0")
            _digest_batch_value(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        kind = "list" if isinstance(value, list) else "tuple"
        digest.update(f"{kind}:{len(value)}\0".encode("ascii"))
        for item in value:
            _digest_batch_value(digest, item)
        return
    if value is None or isinstance(value, (str, int, float, bool)):
        digest.update(b"scalar\0")
        digest.update(
            json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\0"
        )
        return
    raise TypeError(f"unsupported loader-digest value type {type(value)!r}")


def _batch_size_for_profile(batch: object) -> int:
    if isinstance(batch, dict):
        value = batch.get("batch_size")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError("production loader batch has invalid batch_size")
        return value
    try:
        size = len(batch)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RuntimeError("cannot determine loader-profile batch size") from exc
    if size < 1:
        raise RuntimeError("loader profile produced an empty batch")
    return int(size)


def _close_loader_dataset(dataset: object) -> None:
    seen: set[int] = set()
    current = dataset
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        close = getattr(current, "close", None)
        if callable(close):
            close()
            return
        current = getattr(current, "dataset", None)


def run_production_loader_profile(
    *,
    task,
    run_config: dict,
    infos: list[dict],
    tokens: list[str],
    profile_spec,
) -> dict:
    """Measure the exact production loader path; never select a worker in-job."""
    profiles = []
    all_digests: list[str] = []
    workers = [int(value) for value in profile_spec["workers"]]
    for num_workers in workers:
        cell_config = dict(run_config)
        cell_config["num-workers"] = num_workers
        loader = task._make_loader(cell_config, infos, tokens, shuffle=True)
        dataset = loader.dataset
        repeats = []
        try:
            for repeat in range(int(profile_spec["repeats"])):
                sampler = getattr(loader, "sampler", None)
                if sampler is None or not hasattr(sampler, "set_epoch"):
                    raise RuntimeError("loader profile requires the production epoch sampler")
                sampler.set_epoch(0)
                iterator_started = time.perf_counter()
                iterator = iter(loader)
                iterator_create_ms = (time.perf_counter() - iterator_started) * 1000.0
                digest = hashlib.sha256()
                audit_waits = []
                for _ in range(int(profile_spec["determinism_batches"])):
                    started = time.perf_counter()
                    batch = next(iterator)
                    audit_waits.append((time.perf_counter() - started) * 1000.0)
                    _digest_batch_value(digest, batch)
                content_sha256 = digest.hexdigest()
                all_digests.append(content_sha256)
                for _ in range(int(profile_spec["warmup_batches"])):
                    next(iterator)
                waits = []
                measured_samples = 0
                wall_started = time.perf_counter()
                for _ in range(int(profile_spec["measured_batches"])):
                    started = time.perf_counter()
                    batch = next(iterator)
                    waits.append((time.perf_counter() - started) * 1000.0)
                    measured_samples += _batch_size_for_profile(batch)
                wall_seconds = time.perf_counter() - wall_started
                if num_workers == 0:
                    cache_state = (
                        "single-process-first" if repeat == 0 else "single-process-repeat"
                    )
                else:
                    cache_state = (
                        "cold-worker-start" if repeat == 0 else "persistent-worker-warm"
                    )
                repeats.append({
                    "repeat": repeat,
                    "cache_state": cache_state,
                    "iterator_create_ms": iterator_create_ms,
                    "determinism_content_sha256": content_sha256,
                    "determinism_wait_ms": _timing_distribution(audit_waits),
                    "measured_batches": int(profile_spec["measured_batches"]),
                    "measured_samples": measured_samples,
                    "measured_wall_seconds": wall_seconds,
                    "samples_per_second": measured_samples / wall_seconds,
                    "batch_wait_ms": _timing_distribution(waits),
                })
                del iterator
        finally:
            del loader
            _close_loader_dataset(dataset)
            gc.collect()
        profiles.append({"num_workers": num_workers, "repeats": repeats})
    return {
        "schema": "s09.production-loader-profile.v1",
        "spec": {
            "workers": workers,
            "repeats": int(profile_spec["repeats"]),
            "determinism_batches": int(profile_spec["determinism_batches"]),
            "warmup_batches": int(profile_spec["warmup_batches"]),
            "measured_batches": int(profile_spec["measured_batches"]),
        },
        "measurement_definition": (
            "batch_wait_ms is host time blocked in next(production DataLoader); "
            "worker cells are observational and do not alter training.num_workers"
        ),
        "training_num_workers": int(run_config["num-workers"]),
        "profiles": profiles,
        "content_sha256_identical": len(set(all_digests)) == 1,
        "content_sha256": all_digests[0],
    }


def run_strict_official_evaluation(
    *,
    config,
    run_config: dict,
    runtime_dependencies: dict,
    task,
    model,
    optimizer,
    scheduler,
    scaler,
    ema,
    checkpoint: Path,
    device: torch.device,
    output_dir: Path,
    decode_fn: Callable | None = None,
    official_eval_fn: Callable | None = None,
    nusc_factory: Callable | None = None,
) -> dict:
    """Evaluate one strict checkpoint through one token-complete official path.

    Loading deliberately reuses the production checkpoint loader (including
    config/data identity checks and rollback).  The raw/EMA choice and timing
    collection are both fields of the hashed ``s09.v1`` config.
    """
    _, checkpoint_identity = load_checkpoint(
        str(checkpoint), model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler, ema=ema, config=config, map_location="cpu",
    )
    if checkpoint_identity != config.sha256:
        raise RuntimeError(
            "checkpoint identity is not the exact resolved-config identity: "
            f"checkpoint={checkpoint_identity}, config={config.sha256}"
        )
    weights = str(run_config["evaluation-checkpoint-weights"])
    if weights == "ema":
        if ema is None or not hasattr(ema, "module"):
            raise RuntimeError("EMA evaluation requested but checkpoint/runtime has no EMA model")
        model.load_state_dict(ema.module.state_dict(), strict=True)
    elif weights != "raw":
        raise RuntimeError(f"unknown strict evaluation checkpoint policy {weights!r}")
    model.to(device)

    from fl_v3.data.nuscenes import paths as nuscenes_paths
    from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
    from fl_v3.eval.detection_eval import (
        VERSION_EVAL_SET, decode_eval_set, run_detection_eval,
    )
    from fl_v3.eval.box_to_global import NUSCENES_MAX_BOXES_PER_SAMPLE

    version = str(run_config["nuscenes-version"])
    split = str(run_config["nuscenes-val-split"])
    infos, _ = task._load_info(run_config, split)
    all_tokens = sorted(str(info["sample_token"]) for info in infos)
    if len(all_tokens) != len(set(all_tokens)):
        raise RuntimeError("resolved validation cache contains duplicate sample tokens")
    loader = task._make_loader(run_config, infos, all_tokens, shuffle=False)
    timing: dict | None = {} if bool(run_config["evaluation-timing"]) else None
    decoder = decode_fn or decode_eval_set
    decodes = decoder(model, loader, device, run_config, timing)
    decoded_tokens = [str(item.sample_token) for item in decodes]
    if len(decoded_tokens) != len(set(decoded_tokens)):
        raise RuntimeError("strict evaluation decoded a sample token more than once")
    if set(decoded_tokens) != set(all_tokens) or len(decoded_tokens) != len(all_tokens):
        missing = sorted(set(all_tokens) - set(decoded_tokens))[:3]
        extra = sorted(set(decoded_tokens) - set(all_tokens))[:3]
        raise RuntimeError(
            "strict evaluation is not token-complete: "
            f"expected={len(all_tokens)}, decoded={len(decoded_tokens)}, "
            f"missing={missing}, extra={extra}"
        )
    over_cap = [
        str(item.sample_token) for item in decodes
        if len(item.boxes) > NUSCENES_MAX_BOXES_PER_SAMPLE
    ]
    if over_cap:
        raise RuntimeError(
            "strict evaluation decode exceeded official per-sample box cap "
            f"{NUSCENES_MAX_BOXES_PER_SAMPLE}: {over_cap[:3]}"
        )

    eval_run_config = dict(run_config)
    eval_run_config.update({
        "checkpoint-sha256": _checkpoint_sha256(checkpoint),
        "checkpoint-weights": weights,
        "runtime-dependencies-sha256": _canonical_sha256(runtime_dependencies),
    })
    factory = nusc_factory or nuscenes_paths.create_nuscenes
    nusc = factory(version, run_config["nuscenes-dataroot"], verbose=False)
    evaluator = official_eval_fn or run_detection_eval
    metrics = evaluator(
        nusc, decodes, VERSION_EVAL_SET[version], version,
        str(output_dir / "official_detection_eval"), DETECTION_NAMES,
        all_eval_tokens=all_tokens, run_config=eval_run_config, verbose=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "official_metrics.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8",
    )
    if timing is not None:
        (output_dir / "evaluation_timing.json").write_text(
            json.dumps(timing, sort_keys=True) + "\n", encoding="utf-8",
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="strict s09.v1 JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_resolved_config(args.config)
    execution = config.data["execution"]
    execution_mode = config.execution_mode
    readiness = execution_mode == "readiness"
    out_dir = Path(args.out_dir)
    if readiness:
        if args.resume:
            raise RuntimeError("execution.mode='readiness' is non-resumable")
        if out_dir.exists():
            raise RuntimeError(
                "readiness requires a fresh absent output directory: "
                f"{out_dir}"
            )
        out_dir.mkdir(parents=True)

    startup_started = time.perf_counter()
    identity_started = time.perf_counter()
    runtime_dependencies = verify_runtime_dependency_identity(config.to_run_config())
    print(json.dumps({"runtime_dependencies": runtime_dependencies}, sort_keys=True), flush=True)
    verify_physical_data_identities(config)
    startup_phases = {
        "runtime_and_data_identity_seconds": time.perf_counter() - identity_started,
    }
    train_spec = config.data["training"]
    declared_world = int(train_spec["world_size"])
    actual_world = int(os.environ.get("WORLD_SIZE", "1"))
    if actual_world != declared_world:
        raise RuntimeError(f"WORLD_SIZE identity drift: config={declared_world}, runtime={actual_world}")
    if actual_world != 1:
        raise RuntimeError("S07-B integration required for DDP wrapping and distributed sampler wiring")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(train_spec["seed"])
    seed_everything(seed)
    enforce_determinism(precision=config.precision, strict=(config.precision == "fp32"))

    from fl_v3.training.tasks import get_task

    run_config = config.to_run_config()
    task = get_task("nuscenes_detection")
    data_started = time.perf_counter()
    train_split = str(run_config["nuscenes-train-split"])
    infos, _ = task._load_info(run_config, train_split)
    part = task._partition(run_config)
    tokens = sorted({token for shard in part["client_tokens"].values() for token in shard})
    startup_phases["info_and_partition_seconds"] = time.perf_counter() - data_started

    loader_profile = None
    profile_spec = execution["loader_profile"]
    if readiness and profile_spec is not None:
        profile_started = time.perf_counter()
        loader_profile = run_production_loader_profile(
            task=task,
            run_config=run_config,
            infos=infos,
            tokens=tokens,
            profile_spec=profile_spec,
        )
        startup_phases["loader_profile_seconds"] = time.perf_counter() - profile_started
        if not loader_profile["content_sha256_identical"]:
            startup_phases["total_before_training_seconds"] = (
                time.perf_counter() - startup_started
            )
            (out_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
            _write_json(out_dir / "runtime_dependencies.json", runtime_dependencies)
            _write_json(out_dir / "readiness.json", {
                "schema": "s09.readiness-report.v1",
                "status": "FAIL",
                "terminal_reason": (
                    "bounded production-loader content hashes differ across "
                    "declared worker/repeat cells"
                ),
                "resolved_config_sha256": config.sha256,
                "execution_sha256": _canonical_sha256(config.as_dict()["execution"]),
                "data_identities": config.data_identities,
                "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
                "startup_phase_seconds": startup_phases,
                "loader_profile": loader_profile,
                "model_constructed": False,
                "training_started": False,
                "checkpoint_written": False,
                "official_evaluation_executed": False,
            })
            raise RuntimeError(
                "production loader profile content identity differs across cells; "
                "training was not started"
            )

    loader_started = time.perf_counter()
    loader = task._make_loader(run_config, infos, tokens, shuffle=True)
    stream = PersistentEpochIterator(loader)
    startup_phases["fixed_training_loader_seconds"] = time.perf_counter() - loader_started

    model_started = time.perf_counter()
    model = task.build_model(run_config).to(device)
    criterion = task.build_criterion(run_config)
    optimizer = _build_optimizer(model, config)
    # Constant scheduler is still serialized and advances exactly once per
    # successful update; later schedule enums belong in an approved schema bump.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(device, config.precision)
    ema = _build_ema(model, train_spec["ema_decay"])
    state = TrainingState()
    startup_phases["model_and_training_components_seconds"] = (
        time.perf_counter() - model_started
    )

    if readiness:
        max_updates = int(train_spec["max_optimizer_steps"])
        max_attempted = int(execution["max_attempted_windows"])
        timing_warmup = int(execution["timing_warmup_successful_windows"])
        training_started = time.perf_counter()
        train_kwargs = {
            "scheduler": scheduler,
            "ema_model": ema,
            "precision": config.precision,
            "grad_scaler": scaler,
            "accumulation_steps": int(train_spec["accumulation_steps"]),
            "runtime_state": state,
            "max_steps": max_attempted,
            "max_optimizer_steps": max_updates,
            "model_mode": config.model_mode,
            "exposure_multiplier": actual_world,
            "expected_global_microbatch_samples": (
                int(train_spec["micro_batch_size"]) * actual_world
            ),
            "readiness_timing": True,
            "readiness_warmup_successful_windows": timing_warmup,
        }
        mode_context = model.serialized_mode(True) if hasattr(model, "serialized_mode") else None
        if mode_context is None:
            metrics = train_one_epoch(
                model, stream.batches(0), criterion, optimizer, device, **train_kwargs,
            )
        else:
            with mode_context:
                metrics = train_one_epoch(
                    model, stream.batches(0), criterion, optimizer, device, **train_kwargs,
                )
        training_wall_seconds = time.perf_counter() - training_started
        startup_phases["total_before_training_seconds"] = (
            training_started - startup_started
        )
        terminal_state = state.checkpoint_dict()
        passed = (
            state.optimizer_step == max_updates
            and state.attempted_windows <= max_attempted
            and state.discarded_windows == 0
        )
        terminal_reason = (
            "successful-update target reached within the attempted-window cap"
            if passed
            else (
                "successful-update target not reached before loader exhaustion or "
                "attempted-window cap"
            )
        )
        report = {
            "schema": "s09.readiness-report.v1",
            "status": "PASS" if passed else "FAIL",
            "terminal_reason": terminal_reason,
            "resolved_config_sha256": config.sha256,
            "execution_sha256": _canonical_sha256(config.as_dict()["execution"]),
            "data_identities": config.data_identities,
            "runtime_dependencies": runtime_dependencies,
            "runtime_dependencies_sha256": _canonical_sha256(runtime_dependencies),
            "device": (
                {
                    "type": "cuda",
                    "index": int(torch.cuda.current_device()),
                    "name": torch.cuda.get_device_name(device),
                    "compute_capability": list(torch.cuda.get_device_capability(device)),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(device).total_memory
                    ),
                }
                if device.type == "cuda"
                else {"type": "cpu"}
            ),
            "model_mode": config.model_mode,
            "precision": config.precision,
            "sparse_conv_precision": config.sparse_conv_precision,
            "recipe": {
                "optimizer": config.as_dict()["optimizer"],
                "training": config.as_dict()["training"],
                "scheduler": "constant_lambda_1",
                "gradient_clipping": None,
            },
            "partition": {
                "mode": str(part["mode"]),
                "num_clients": int(part["num_clients"]),
                "unique_train_tokens": len(tokens),
            },
            "fixed_training_num_workers": int(train_spec["num_workers"]),
            "loader_profile": loader_profile,
            "startup_phase_seconds": startup_phases,
            "training_wall_seconds": training_wall_seconds,
            "training_metrics": metrics,
            "terminal_training_state": terminal_state,
            "target_successful_windows": max_updates,
            "max_attempted_windows": max_attempted,
            "model_constructed": True,
            "training_started": True,
            "checkpoint_written": False,
            "official_evaluation_executed": False,
            "interpretation_limits": [
                "engineering readiness only; not convergence, mAP/NDS, or model quality",
                "loader profile cells are observational and did not select num_workers in-job",
                "no checkpoint or official evaluation was produced",
            ],
        }
        (out_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
        _write_json(out_dir / "runtime_dependencies.json", runtime_dependencies)
        _write_json(out_dir / "readiness.json", report)
        print(json.dumps({"readiness": report}, sort_keys=True, allow_nan=False), flush=True)
        if not passed:
            raise RuntimeError(terminal_reason)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "checkpoint.pt"
    if args.resume:
        if not checkpoint.is_file():
            raise RuntimeError(f"--resume requested but checkpoint is missing: {checkpoint}")
        state, _ = load_checkpoint(
            str(checkpoint), model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, config=config, map_location="cpu",
        )
    elif checkpoint.exists():
        raise RuntimeError(f"output checkpoint already exists; use --resume or a new output: {checkpoint}")

    max_updates = int(train_spec["max_optimizer_steps"])
    max_epochs = int(train_spec["max_epochs"])
    while state.optimizer_step < max_updates and state.epoch < max_epochs:
        mode_context = model.serialized_mode(True) if hasattr(model, "serialized_mode") else None
        if mode_context is None:
            metrics = train_one_epoch(
                model, stream.batches(state.epoch), criterion, optimizer, device,
                scheduler=scheduler, ema_model=ema, precision=config.precision,
                grad_scaler=scaler, accumulation_steps=int(train_spec["accumulation_steps"]),
                runtime_state=state, max_optimizer_steps=max_updates,
                model_mode=config.model_mode, exposure_multiplier=actual_world,
                expected_global_microbatch_samples=(
                    int(train_spec["micro_batch_size"]) * actual_world
                ),
            )
        else:
            with mode_context:
                metrics = train_one_epoch(
                    model, stream.batches(state.epoch), criterion, optimizer, device,
                    scheduler=scheduler, ema_model=ema, precision=config.precision,
                    grad_scaler=scaler, accumulation_steps=int(train_spec["accumulation_steps"]),
                    runtime_state=state, max_optimizer_steps=max_updates,
                    model_mode=config.model_mode, exposure_multiplier=actual_world,
                    expected_global_microbatch_samples=(
                        int(train_spec["micro_batch_size"]) * actual_world
                    ),
                )
        state.epoch += 1
        save_checkpoint(
            str(checkpoint), model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, state=state, config=config,
            checkpoint_identity=config.sha256,
        )
        print(json.dumps({"epoch": state.epoch, **metrics}, sort_keys=True), flush=True)

    if state.optimizer_step != max_updates:
        raise RuntimeError(
            f"executed-update budget not reached: {state.optimizer_step}/{max_updates}; "
            "nonfinite/overflow/epoch stop remains negative evidence"
        )
    (out_dir / "resolved_config.json").write_bytes(config.canonical_bytes + b"\n")
    (out_dir / "runtime_dependencies.json").write_text(
        json.dumps(runtime_dependencies, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "checkpoint.sha256").write_text(_checkpoint_sha256(checkpoint) + "\n", encoding="utf-8")
    metrics = run_strict_official_evaluation(
        config=config, run_config=run_config, runtime_dependencies=runtime_dependencies,
        task=task, model=model, optimizer=optimizer, scheduler=scheduler,
        scaler=scaler, ema=ema, checkpoint=checkpoint, device=device,
        output_dir=out_dir,
    )
    print(json.dumps({"official_evaluation": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
