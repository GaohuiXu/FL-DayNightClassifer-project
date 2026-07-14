"""S07-B resolved centralized trainer.

This entry point accepts only the canonical ``s08.v1`` config.  It constructs
one mode-aware loader, maps exact architecture enums to the reviewed stack,
advances schedules by successful optimizer updates, and writes one complete
boundary-safe checkpoint.  DDP remains fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
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
    collection are both fields of the hashed ``s08.v1`` config.
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
    parser.add_argument("--config", required=True, help="strict s08.v1 JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_resolved_config(args.config)
    runtime_dependencies = verify_runtime_dependency_identity(config.to_run_config())
    print(json.dumps({"runtime_dependencies": runtime_dependencies}, sort_keys=True), flush=True)
    verify_physical_data_identities(config)
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
    train_split = str(run_config["nuscenes-train-split"])
    infos, _ = task._load_info(run_config, train_split)
    part = task._partition(run_config)
    tokens = sorted({token for shard in part["client_tokens"].values() for token in shard})
    loader = task._make_loader(run_config, infos, tokens, shuffle=True)
    stream = PersistentEpochIterator(loader)

    model = task.build_model(run_config).to(device)
    criterion = task.build_criterion(run_config)
    optimizer = _build_optimizer(model, config)
    # Constant scheduler is still serialized and advances exactly once per
    # successful update; later schedule enums belong in an approved schema bump.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(device, config.precision)
    ema = _build_ema(model, train_spec["ema_decay"])
    state = TrainingState()

    out_dir = Path(args.out_dir)
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
