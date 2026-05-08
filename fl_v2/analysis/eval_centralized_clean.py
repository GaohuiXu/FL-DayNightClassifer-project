#!/usr/bin/env python3
"""Centralised clean upper-bound: ResNet18 trained centrally on GTSRB.

Standard PyTorch training loop on the full GTSRB train set (no FL, no
client partitioning, no malicious clients). Establishes the upper bound
on what ResNet18 can achieve on GTSRB at our chosen architecture
(modified-conv1, image-size 32). Same model + same data + same epochs
as one FL client would see during 100 rounds × 3 local epochs would be
unfair to FL because of the local-epoch / averaging structure; this
script just trains for 50 centralised epochs to plateau.

Reports clean test accuracy, per-class accuracy, and ASR (with the
standard 4×4 white pixel trigger; useful as a reference because the
centralised model has never seen the trigger and should give near-zero
ASR — a sanity check that the trigger isn't trivially recognised by a
clean model).

Usage:
    python -m analysis.eval_centralized_clean \
        --data-root /path/to/gtsrb \
        --output /path/to/results.json \
        --epochs 50 --pretrained --modified-conv1 --device auto
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_v2.attacks_defenses import make_pixel_trigger_fn
from fl_v2.data.dataset import (
    load_gtsrb_test_dataset,
    load_gtsrb_train_dataset,
)
from fl_v2.data.transforms import get_train_transforms
from fl_v2.models import create_model


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_label: int,
    trigger_fn,
) -> dict:
    model.eval()
    n_total = 0
    n_correct_clean = 0
    n_target = 0
    n_target_correct = 0
    n_nontarget = 0
    n_asr = 0
    for images, labels in dataloader:
        images_dev = images.to(device)
        labels_dev = labels.to(device)
        labels_cpu = labels

        logits_clean = model(images_dev)
        preds_clean = logits_clean.argmax(dim=1)
        n_total += labels_dev.size(0)
        n_correct_clean += int((preds_clean == labels_dev).sum().item())

        target_mask = labels_dev == target_label
        n_target += int(target_mask.sum().item())
        n_target_correct += int(
            (preds_clean[target_mask] == target_label).sum().item()
        )

        triggered = trigger_fn(images).to(device)
        logits_trig = model(triggered)
        preds_trig = logits_trig.argmax(dim=1)
        non_target_mask = (labels_cpu != target_label).to(device)
        n_non_t = int(non_target_mask.sum().item())
        if n_non_t > 0:
            n_nontarget += n_non_t
            n_asr += int(
                (preds_trig[non_target_mask] == target_label).sum().item()
            )

    return {
        "clean_acc":              n_correct_clean / max(n_total, 1),
        "target_class_clean_acc": n_target_correct / max(n_target, 1),
        "asr":                    n_asr / max(n_nontarget, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Use pretrained ImageNet ResNet18 init."
    )
    parser.add_argument(
        "--canonical-conv1", action="store_true",
        help="Use canonical 7x7 stride-2 conv1 (image-size 64); else "
             "modified 3x3 stride-1 conv1 (image-size 32)."
    )
    parser.add_argument("--target-label", type=int, default=2)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--trigger-value", type=float, default=1.0)
    parser.add_argument("--trigger-position", type=str, default="bottom-right")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker subprocesses for CPU-side augmentation. "
             "Default 4 keeps the GPU fed; set 0 to revert to "
             "single-threaded data loading (slow, see "
             "docs/cycle_02_gpu_efficiency_investigation.md).",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )
    image_size = 64 if args.canonical_conv1 else 32

    # Determinism flags consistent with the FL pipeline
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"[centralised] device={device}  image_size={image_size}  "
          f"pretrained={args.pretrained}  canonical_conv1={args.canonical_conv1}  "
          f"epochs={args.epochs}  seed={args.seed}")

    # Train data: full clean GTSRB train, with augmentation matching the FL
    # client trainloaders (RandomRotation + ColorJitter + ToTensor + Normalize).
    # `load_gtsrb_train_dataset` from fl_v2.data.dataset has no transform
    # parameter, so build directly via torchvision GTSRB to attach
    # `get_train_transforms`.
    from torchvision.datasets import GTSRB
    train_dataset = GTSRB(
        root=args.data_root, split="train", download=False,
        transform=get_train_transforms(image_size=image_size),
    )
    test_dataset = load_gtsrb_test_dataset(
        data_root=args.data_root, image_size=image_size, download=False,
    )

    loader_gen = torch.Generator().manual_seed(args.seed)
    train_kwargs = dict(
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_gen,
    )
    test_kwargs = dict(
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        from fl_v2.utils.runtime import seeded_worker_init
        train_kwargs.update(
            worker_init_fn=seeded_worker_init,
            persistent_workers=True,
            prefetch_factor=2,
        )
        test_kwargs.update(
            worker_init_fn=seeded_worker_init,
            persistent_workers=True,
            prefetch_factor=2,
        )
    trainloader = DataLoader(train_dataset, **train_kwargs)
    testloader = DataLoader(test_dataset, **test_kwargs)

    trigger_fn = make_pixel_trigger_fn(
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_position=args.trigger_position,
    )

    model = create_model(
        "resnet18", num_classes=43,
        pretrained=args.pretrained, canonical_conv1=args.canonical_conv1,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        n_correct = 0
        n_samples = 0
        total_loss = 0.0
        for images, labels in trainloader:
            images = images.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            n_correct += int((logits.argmax(dim=1) == labels).sum().item())
            n_samples += bs
        scheduler.step()
        train_loss = total_loss / max(n_samples, 1)
        train_acc = n_correct / max(n_samples, 1)

        eval_metrics = _evaluate(
            model, testloader, device,
            target_label=args.target_label, trigger_fn=trigger_fn,
        )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_clean_acc": eval_metrics["clean_acc"],
            "test_target_class_clean_acc": eval_metrics["target_class_clean_acc"],
            "test_asr": eval_metrics["asr"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
        }
        history.append(row)
        print(
            f"  epoch {epoch + 1:>3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"clean_acc={eval_metrics['clean_acc']:.4f}  "
            f"target_acc={eval_metrics['target_class_clean_acc']:.4f}  "
            f"asr={eval_metrics['asr']:.4f}  ({elapsed:.1f}s)",
            flush=True,
        )

    final = history[-1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "config": vars(args) | {"image_size": image_size, "device": str(device),
                                     "output": str(args.output)},
            "final": final,
            "history": history,
        }, f, indent=2, default=str)
    print(f"[centralised] saved → {args.output}")


if __name__ == "__main__":
    main()
