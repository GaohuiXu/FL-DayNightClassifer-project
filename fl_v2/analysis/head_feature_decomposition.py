#!/usr/bin/env python3
"""Head-feature decomposition diagnostic for backdoored FL checkpoints.

Quantifies how much of a backdoor's attack-success-rate (ASR) is removable
by reinitializing and retraining only the classifier head on clean data.
The fraction of ASR that survives the head-swap is attributable to the
feature extractor; the fraction that disappears is attributable to the
original head's decision-boundary contamination.

Workflow per checkpoint:
    1. Load final_model.pt into GTSRBResNet18.
    2. Measure original_clean_acc and original_asr on the global test set.
    3. Freeze model.features + model.avgpool (requires_grad=False, .eval()).
    4. Reinit model.fc as a fresh nn.Linear(512, num_classes).
    5. Train ONLY model.fc on clean GTSRB train (no FL partition, no trigger)
       for --epochs (default 10) using Adam(lr=--lr).
    6. Re-measure clean_head_clean_acc and clean_head_asr.
    7. Compute head_attribution_pct = (orig_asr - new_asr) / orig_asr * 100.
    8. Save head_feature_decomposition.json next to the checkpoint.

Usage:
    python -m analysis.head_feature_decomposition \
        --exp-dir /path/to/experiment_dir \
        --data-root /path/to/gtsrb \
        --epochs 10 --lr 1e-3 --device auto --seed 4242

This is the headline number for the Cycle-02-pivot-week-1 supervisor doc:
    head_attribution_pct ≈ 100  → backdoor lives in the head (HTBA-like)
    head_attribution_pct ≈   0  → backdoor lives in the encoder (representation-level)
    head_attribution_pct ≈  50  → joint mechanism (Cycle 01 pixel-trigger profile)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB

from fl_v2.attacks_defenses import make_pixel_trigger_fn
from fl_v2.data.dataset import get_global_testloader
from fl_v2.data.transforms import get_eval_transforms, get_train_transforms
from fl_v2.models import create_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml_config(path: Path) -> dict:
    """Robust YAML loader (handles bools, quoted strings, comments)."""
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def _build_clean_trainloader(
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    augment: bool = False,
) -> DataLoader:
    """Build a DataLoader over the full clean GTSRB train split.

    No FL partition, no attack, no triggered samples. Used to retrain a
    fresh classifier head from scratch in the head-feature decomposition.
    """
    transform = (
        get_train_transforms(image_size=image_size)
        if augment
        else get_eval_transforms(image_size=image_size)
    )
    dataset = GTSRB(
        root=data_root,
        split="train",
        download=False,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_label: int,
    trigger_fn=None,
) -> dict:
    """Single-pass clean accuracy + ASR on the test set."""
    model.eval()
    n_total = 0
    n_correct_clean = 0
    n_nontarget = 0
    n_asr = 0
    for images, labels in dataloader:
        labels_cpu = labels
        images_dev = images.to(device)
        labels_dev = labels.to(device)

        logits_clean = model(images_dev)
        preds_clean = logits_clean.argmax(dim=1)

        n_total += labels_dev.size(0)
        n_correct_clean += int((preds_clean == labels_dev).sum().item())

        if trigger_fn is not None:
            triggered = trigger_fn(images).to(device)
            logits_trig = model(triggered)
            preds_trig = logits_trig.argmax(dim=1)
            non_target_mask = labels_cpu != target_label
            non_target_count = int(non_target_mask.sum().item())
            if non_target_count > 0:
                hits = int(
                    (preds_trig[non_target_mask.to(device)] == target_label)
                    .sum()
                    .item()
                )
                n_nontarget += non_target_count
                n_asr += hits

    out = {
        "clean_acc": n_correct_clean / max(n_total, 1),
        "n_total": n_total,
    }
    if trigger_fn is not None:
        out["asr"] = n_asr / max(n_nontarget, 1)
        out["n_asr_eligible"] = n_nontarget
    return out


def _train_fresh_head(
    model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> list[dict]:
    """Train ONLY model.fc on the provided clean trainloader.

    The feature extractor (model.features + model.avgpool) is left in eval
    mode so frozen BatchNorm statistics do not drift. Only the freshly
    reinitialized head learns.
    """
    # Freeze everything; unfreeze only fc.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    # Frozen subtrees stay in eval() mode throughout. Only the head is in
    # train() mode (irrelevant for nn.Linear but symmetric).
    model.eval()
    model.fc.train()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        n_correct = 0
        n_samples = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward through frozen extractor + trainable head.
            with torch.no_grad():
                feats = model.forward_features(images)
            logits = model.fc(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            n_correct += int((logits.argmax(dim=1) == labels).sum().item())
            n_samples += bs

        epoch_loss = total_loss / max(n_samples, 1)
        epoch_acc = n_correct / max(n_samples, 1)
        elapsed = time.time() - t0
        history.append(
            {"epoch": epoch + 1, "loss": epoch_loss, "acc": epoch_acc, "secs": elapsed}
        )
        print(
            f"[head-train] epoch {epoch + 1}/{epochs}  "
            f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Head-feature decomposition: re-train a fresh classifier head "
            "on clean data and measure how much ASR survives."
        )
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory with config.yaml and checkpoints/final_model.pt.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="GTSRB dataset root.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Epochs of clean-head training (default 10).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for clean-head training (default 1e-3).",
    )
    parser.add_argument(
        "--head-batch-size",
        type=int,
        default=128,
        help="Batch size for clean-head training (default 128).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cuda', 'cpu', or 'auto' (default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4242,
        help="Seed for the head reinitialization + train loader shuffle.",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use train-time augmentation while training the fresh head.",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="final_model.pt",
        help="Checkpoint filename under <exp_dir>/checkpoints/ (default final_model.pt).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <exp_dir>/head_feature_decomposition.json).",
    )
    args = parser.parse_args()

    exp_dir: Path = args.exp_dir
    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / "checkpoints" / args.checkpoint_name
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    config = _load_yaml_config(config_path)
    model_type = str(config.get("model-type", "resnet18"))
    num_classes = int(config.get("num-classes", 43))
    image_size = int(config.get("image-size", 32))
    eval_batch_size = int(config.get("batch-size", 64))
    attack_type = str(config.get("attack-type", "none"))
    target_label = int(config.get("backdoor-target-label", 0))
    pretrained_init_cfg = bool(config.get("pretrained-init", False))
    trainable_layers_cfg = str(config.get("trainable-layers", "full_ft"))
    # Architecture flag — must match what produced the checkpoint, or
    # load_state_dict fails on mismatched Sequential keys.
    canonical_conv1 = bool(config.get("canonical-conv1", False))

    device = _resolve_device(args.device)
    print(f"[head-decomp] exp:         {exp_dir.name}")
    print(f"[head-decomp] checkpoint:  {checkpoint_path.name}")
    print(f"[head-decomp] model:       {model_type}")
    print(f"[head-decomp] attack:      {attack_type}")
    print(f"[head-decomp] device:      {device}")
    print(f"[head-decomp] seed:        {args.seed}")
    print(f"[head-decomp] epochs:      {args.epochs}")
    print(f"[head-decomp] lr:          {args.lr}")

    if attack_type not in ("pixel_backdoor", "model_replacement"):
        print(
            "[head-decomp] no backdoor attack in config — clean run "
            "(ASR not computable). Will still report clean-head clean_acc.",
            flush=True,
        )

    # ----- 1. Load model with original head (random init at construct time;
    # load_state_dict replaces all weights, so pretrained=False is fine).
    # canonical_conv1 must match the experiment so Sequential keys align.
    model = create_model(
        model_type,
        num_classes=num_classes,
        pretrained=False,
        canonical_conv1=canonical_conv1,
    )
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    print(
        f"[head-decomp] checkpoint loaded "
        f"(canonical_conv1={canonical_conv1}).",
        flush=True,
    )

    # ----- 2. Build evaluation pipeline (test loader + trigger).
    testloader = get_global_testloader(
        data_root=args.data_root,
        batch_size=eval_batch_size,
        image_size=image_size,
        num_workers=0,
        download=False,
    )
    trigger_fn = None
    if attack_type in ("pixel_backdoor", "model_replacement"):
        trigger_fn = make_pixel_trigger_fn(
            trigger_size=int(config.get("trigger-size", 4)),
            trigger_value=float(config.get("trigger-value", 1.0)),
            trigger_position=str(config.get("trigger-position", "bottom-right")),
        )
        print(
            f"[head-decomp] trigger:     {attack_type} "
            f"(target={target_label})",
            flush=True,
        )

    # ----- 3. Original (loaded checkpoint) accuracy + ASR.
    print("[head-decomp] step 1/3: evaluating original head...", flush=True)
    orig_eval = _evaluate(model, testloader, device, target_label, trigger_fn)
    original_clean_acc = float(orig_eval["clean_acc"])
    original_asr: Optional[float] = (
        float(orig_eval["asr"]) if "asr" in orig_eval else None
    )
    print(
        f"[head-decomp] original clean_acc={original_clean_acc:.4f}  "
        f"asr={original_asr if original_asr is None else f'{original_asr:.4f}'}",
        flush=True,
    )

    # ----- 4. Reinit fc head with a fresh seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model.fc = nn.Linear(512, num_classes).to(device)
    print(
        f"[head-decomp] step 2/3: reinit fc head ({args.seed=}); "
        f"training {args.epochs} epochs lr={args.lr}...",
        flush=True,
    )

    # ----- 5. Train fresh head on clean GTSRB train.
    trainloader = _build_clean_trainloader(
        data_root=args.data_root,
        image_size=image_size,
        batch_size=args.head_batch_size,
        augment=args.augment,
    )
    history = _train_fresh_head(
        model=model,
        trainloader=trainloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    # ----- 6. Re-evaluate with the fresh head.
    print("[head-decomp] step 3/3: evaluating clean-head model...", flush=True)
    new_eval = _evaluate(model, testloader, device, target_label, trigger_fn)
    clean_head_clean_acc = float(new_eval["clean_acc"])
    clean_head_asr: Optional[float] = (
        float(new_eval["asr"]) if "asr" in new_eval else None
    )
    print(
        f"[head-decomp] clean-head clean_acc={clean_head_clean_acc:.4f}  "
        f"asr={clean_head_asr if clean_head_asr is None else f'{clean_head_asr:.4f}'}",
        flush=True,
    )

    # ----- 7. Attribution. Only meaningful when there is a backdoor.
    head_attribution_pct: Optional[float] = None
    feature_attribution_pct: Optional[float] = None
    if original_asr is not None and clean_head_asr is not None:
        denom = max(original_asr, 1e-6)
        head_attribution_pct = float((original_asr - clean_head_asr) / denom * 100.0)
        feature_attribution_pct = float(100.0 - head_attribution_pct)
        print(
            f"[head-decomp] head_attribution_pct = "
            f"({original_asr:.4f} - {clean_head_asr:.4f}) / {original_asr:.4f} "
            f"= {head_attribution_pct:.1f}%",
            flush=True,
        )

    # ----- 8. Save JSON.
    out: dict = {
        "experiment_name": str(config.get("experiment-name", exp_dir.name)),
        "exp_dir": str(exp_dir),
        "checkpoint": args.checkpoint_name,
        "model_type": model_type,
        "num_classes": num_classes,
        "image_size": image_size,
        "attack_type": attack_type,
        "backdoor_target_label": target_label,
        "pretrained_init_cfg": pretrained_init_cfg,
        "trainable_layers_cfg": trainable_layers_cfg,
        "canonical_conv1": canonical_conv1,
        "original_clean_acc": original_clean_acc,
        "original_asr": original_asr,
        "clean_head_clean_acc": clean_head_clean_acc,
        "clean_head_asr": clean_head_asr,
        "head_attribution_pct": head_attribution_pct,
        "feature_attribution_pct": feature_attribution_pct,
        "head_train_epochs": args.epochs,
        "head_train_lr": args.lr,
        "head_train_batch_size": args.head_batch_size,
        "head_train_augment": bool(args.augment),
        "head_train_history": history,
        "seed": args.seed,
        "device": str(device),
    }

    output_path: Path = (
        args.output if args.output is not None
        else exp_dir / "head_feature_decomposition.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[head-decomp] saved → {output_path}", flush=True)


if __name__ == "__main__":
    main()
