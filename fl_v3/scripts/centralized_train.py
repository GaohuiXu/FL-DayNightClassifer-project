"""S06 centralized production trainer.

This entry point accepts only the canonical ``s06.v1`` config.  It constructs
one loader, advances schedules by successful optimizer updates, and writes one
complete boundary-safe checkpoint.  S07-B must wire the reviewed S02-S05 module
and modality-aware dataset interfaces before real nuScenes execution is possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, "fl_v3/src")

import torch

from fl_v3.config import load_resolved_config, verify_physical_data_identities
from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import PersistentEpochIterator, TrainingState
from fl_v3.utils.runtime import enforce_determinism, make_grad_scaler, seed_everything


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="strict s06.v1 JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_resolved_config(args.config)
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

    # S07-B will replace this bridge only with reviewed enum/module and mode-aware
    # data integration.  Current S06 tasks fail closed before disabled-modality I/O.
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
            grad_scaler=scaler, ema=ema, config=config, map_location=device,
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
            )
        else:
            with mode_context:
                metrics = train_one_epoch(
                    model, stream.batches(state.epoch), criterion, optimizer, device,
                    scheduler=scheduler, ema_model=ema, precision=config.precision,
                    grad_scaler=scaler, accumulation_steps=int(train_spec["accumulation_steps"]),
                    runtime_state=state, max_optimizer_steps=max_updates,
                    model_mode=config.model_mode, exposure_multiplier=actual_world,
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
    (out_dir / "checkpoint.sha256").write_text(_checkpoint_sha256(checkpoint) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
