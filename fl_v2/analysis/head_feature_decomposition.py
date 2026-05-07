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
    *,
    eval_fn=None,
    patience: int = 8,
    min_improvement: float = 1e-4,
) -> dict:
    """Train ONLY model.fc on the provided clean trainloader, to convergence.

    Convergence-based stopping: train up to ``epochs`` epochs (a generous
    upper bound), evaluate clean test accuracy after every epoch via
    ``eval_fn``, stop if the best-so-far clean-test-acc has not improved
    by at least ``min_improvement`` over the last ``patience`` epochs.

    Why convergence-based and not fixed-epoch: the fresh head sees
    encoders of very different qualities (`canonconv1 head_only` vs
    `full_ft`); a fixed epoch budget systematically undertrains the head
    on cells where the encoder produces poorer features, which biases
    head_attribution_pct upward. With early-stop-on-plateau the head
    reaches its true ceiling on every cell, making cross-cell comparisons
    fair.

    Returns a dict with the per-epoch history and the best-so-far metrics.
    """
    # Freeze everything; unfreeze only fc.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    # Frozen subtrees stay in eval() mode throughout.
    model.eval()
    model.fc.train()

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    # Cosine annealing helps the head reach its plateau cleanly rather
    # than oscillating around a near-optimum.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_clean_acc = -float("inf")
    best_clean_asr: float | None = None
    best_epoch = -1
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.time()
        total_loss = 0.0
        n_correct = 0
        n_samples = 0
        model.fc.train()
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
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

        scheduler.step()
        epoch_loss = total_loss / max(n_samples, 1)
        epoch_acc = n_correct / max(n_samples, 1)
        elapsed = time.time() - t0

        # Evaluate on the clean + triggered test set after every epoch so
        # we can early-stop on plateau and report the best-checkpoint metrics.
        eval_clean_acc = None
        eval_clean_asr = None
        if eval_fn is not None:
            eval_out = eval_fn(model)
            eval_clean_acc = float(eval_out["clean_acc"])
            eval_clean_asr = (
                float(eval_out["asr"]) if "asr" in eval_out else None
            )

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "secs": elapsed,
            "eval_clean_acc": eval_clean_acc,
            "eval_clean_asr": eval_clean_asr,
        })
        print(
            f"[head-train] epoch {epoch + 1:3d}/{epochs}  "
            f"loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.5f}  "
            f"clean_acc={eval_clean_acc if eval_clean_acc is None else f'{eval_clean_acc:.4f}'}  "
            f"clean_asr={eval_clean_asr if eval_clean_asr is None else f'{eval_clean_asr:.4f}'}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        if eval_clean_acc is not None:
            if eval_clean_acc > best_clean_acc + min_improvement:
                best_clean_acc = eval_clean_acc
                best_clean_asr = eval_clean_asr
                best_epoch = epoch + 1
                # Snapshot the head weights at the best-eval-acc point.
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in model.fc.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(
                        f"[head-train] EARLY STOP at epoch {epoch + 1}: "
                        f"clean_acc plateaued (best={best_clean_acc:.4f} at "
                        f"epoch {best_epoch}, no improvement > {min_improvement} "
                        f"for {patience} epochs).",
                        flush=True,
                    )
                    break

    # Restore the best-checkpoint head so the caller's downstream eval uses
    # the converged weights, not the last-epoch weights.
    if best_state_dict is not None:
        model.fc.load_state_dict(best_state_dict)

    return {
        "history": history,
        "best_clean_acc": best_clean_acc if best_state_dict is not None else None,
        "best_clean_asr": best_clean_asr,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history),
        "early_stopped": (best_state_dict is not None
                          and len(history) < epochs),
        "patience": patience,
        "min_improvement": min_improvement,
    }


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
        default=100,
        help=(
            "Maximum epochs of clean-head training (default 100). "
            "Acts as a generous upper bound — actual training stops earlier "
            "once the per-epoch test-set clean accuracy plateaus, see "
            "--patience and --min-improvement."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help=(
            "Early-stop patience: stop when clean-test-acc has not improved "
            "by at least --min-improvement for this many consecutive epochs "
            "(default 8)."
        ),
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=1e-4,
        help=(
            "Minimum clean-test-acc improvement that counts as progress for "
            "early-stop (default 1e-4 = 0.01 pp)."
        ),
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
        f"[head-decomp] step 2/3: reinit fc head (seed={args.seed}); "
        f"training up to {args.epochs} epochs lr={args.lr} "
        f"(early-stop on plateau: patience={args.patience}, "
        f"min_improvement={args.min_improvement})...",
        flush=True,
    )

    # ----- 5. Train fresh head on clean GTSRB train, with per-epoch eval +
    # convergence-based stopping. The eval function hands the trainer the
    # current model so it can measure clean test accuracy after every epoch
    # and snapshot the best-checkpoint weights.
    trainloader = _build_clean_trainloader(
        data_root=args.data_root,
        image_size=image_size,
        batch_size=args.head_batch_size,
        augment=args.augment,
    )

    def _per_epoch_eval(m: nn.Module) -> dict:
        return _evaluate(m, testloader, device, target_label, trigger_fn)

    train_out = _train_fresh_head(
        model=model,
        trainloader=trainloader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        eval_fn=_per_epoch_eval,
        patience=args.patience,
        min_improvement=args.min_improvement,
    )
    history = train_out["history"]

    # ----- 6. Re-evaluate with the BEST clean head (weights restored by
    # _train_fresh_head). This is the converged metric, not the last-epoch
    # metric — eliminates the undertraining bias that pre-`bd2c1eb`-era
    # 10-epoch runs introduced.
    print("[head-decomp] step 3/3: evaluating best-checkpoint clean-head model...", flush=True)
    new_eval = _evaluate(model, testloader, device, target_label, trigger_fn)
    clean_head_clean_acc = float(new_eval["clean_acc"])
    clean_head_asr: Optional[float] = (
        float(new_eval["asr"]) if "asr" in new_eval else None
    )
    print(
        f"[head-decomp] best clean-head clean_acc={clean_head_clean_acc:.4f}  "
        f"asr={clean_head_asr if clean_head_asr is None else f'{clean_head_asr:.4f}'}  "
        f"(reached at epoch {train_out['best_epoch']}/{train_out['total_epochs_run']}; "
        f"early_stopped={train_out['early_stopped']})",
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
        "head_train_max_epochs": args.epochs,
        "head_train_lr": args.lr,
        "head_train_batch_size": args.head_batch_size,
        "head_train_augment": bool(args.augment),
        "head_train_patience": args.patience,
        "head_train_min_improvement": args.min_improvement,
        "head_train_best_epoch": train_out["best_epoch"],
        "head_train_total_epochs_run": train_out["total_epochs_run"],
        "head_train_early_stopped": bool(train_out["early_stopped"]),
        "head_train_best_clean_acc": train_out["best_clean_acc"],
        "head_train_best_clean_asr": train_out["best_clean_asr"],
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
