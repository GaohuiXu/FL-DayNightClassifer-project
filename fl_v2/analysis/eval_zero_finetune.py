#!/usr/bin/env python3
"""Zero-fine-tuning baseline: ImageNet-pretrained ResNet18 on GTSRB.

Loads ImageNet pretrained ResNet18, attaches a random-init 43-class fc head
(the GTSRB output layer), and evaluates on the official GTSRB test set
WITHOUT any GTSRB-specific training. Reports clean accuracy, target-class
clean accuracy, and ASR (with the standard 4×4 white pixel trigger at the
bottom-right corner, target = class 2 = "Speed limit 50 km/h").

Used to answer: "is FL fine-tuning necessary?" — if the strict zero-FT
baseline is already at 95% clean accuracy, FL is a no-op; if it is at
chance (~2.3%), FL is doing all the work.

Reports BOTH architectures we use:
  - canonical_conv1=True  (image-size 64, full ImageNet 7×7 conv1 + maxpool)
  - canonical_conv1=False (image-size 32, modified 3×3 stride-1 conv1
                           that gets RANDOM-INIT here too — this matches
                           the C2 pitfall in the risk audit)

Usage:
    python -m analysis.eval_zero_finetune \
        --data-root /path/to/gtsrb \
        --output /path/to/results.json \
        --device auto

Three random fc seeds are tried per architecture so the chance baseline
is reported with a small spread, not a single point.
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
from fl_v2.data.dataset import load_gtsrb_test_dataset
from fl_v2.models import create_model


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_label: int,
    trigger_fn,
) -> dict:
    """Single-pass clean accuracy + target-class accuracy + ASR."""
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
        n_target_correct += int((preds_clean[target_mask] == target_label).sum().item())

        triggered = trigger_fn(images).to(device)
        logits_trig = model(triggered)
        preds_trig = logits_trig.argmax(dim=1)
        non_target_mask = (labels_cpu != target_label).to(device)
        n_non_t = int(non_target_mask.sum().item())
        if n_non_t > 0:
            n_nontarget += n_non_t
            n_asr += int((preds_trig[non_target_mask] == target_label).sum().item())

    return {
        "clean_acc":              n_correct_clean / max(n_total, 1),
        "target_class_clean_acc": n_target_correct / max(n_target, 1),
        "asr":                    n_asr / max(n_nontarget, 1),
        "n_total":                n_total,
        "n_target":               n_target,
        "n_asr_eligible":         n_nontarget,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--target-label", type=int, default=2)
    parser.add_argument("--trigger-size", type=int, default=4)
    parser.add_argument("--trigger-value", type=float, default=1.0)
    parser.add_argument("--trigger-position", type=str, default="bottom-right")
    parser.add_argument("--fc-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else args.device if args.device != "auto" else "cpu"
    )
    print(f"[zero-ft] device={device}")

    trigger_fn = make_pixel_trigger_fn(
        trigger_size=args.trigger_size,
        trigger_value=args.trigger_value,
        trigger_position=args.trigger_position,
    )

    results = {
        "target_label":     args.target_label,
        "trigger_size":     args.trigger_size,
        "trigger_value":    args.trigger_value,
        "trigger_position": args.trigger_position,
        "fc_seeds":         list(args.fc_seeds),
        "by_architecture":  {},
    }

    for canonical_conv1 in [True, False]:
        # canonical_conv1=True needs image-size 64 (ImageNet stem expects bigger
        # input); the modified 3×3 path can run at 32.
        image_size = 64 if canonical_conv1 else 32
        arch_label = "canonical_conv1_64x64" if canonical_conv1 else "modified_conv1_32x32"
        print(f"\n[zero-ft] === arch={arch_label}  image_size={image_size} ===")

        test_dataset = load_gtsrb_test_dataset(
            data_root=args.data_root,
            image_size=image_size,
            download=False,
        )
        testloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        per_seed = []
        for fc_seed in args.fc_seeds:
            torch.manual_seed(fc_seed)
            np.random.seed(fc_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fc_seed)

            t0 = time.time()
            model = create_model(
                "resnet18",
                num_classes=43,
                pretrained=True,
                canonical_conv1=canonical_conv1,
            )
            model.to(device)

            metrics = _evaluate(
                model, testloader, device,
                target_label=args.target_label, trigger_fn=trigger_fn,
            )
            elapsed = time.time() - t0
            metrics["fc_seed"] = fc_seed
            metrics["elapsed_sec"] = elapsed
            print(
                f"  fc_seed={fc_seed}  clean_acc={metrics['clean_acc']:.4f}  "
                f"target_class_acc={metrics['target_class_clean_acc']:.4f}  "
                f"asr={metrics['asr']:.4f}  ({elapsed:.1f}s)"
            )
            per_seed.append(metrics)

        clean_accs = [m["clean_acc"] for m in per_seed]
        target_accs = [m["target_class_clean_acc"] for m in per_seed]
        asrs = [m["asr"] for m in per_seed]

        results["by_architecture"][arch_label] = {
            "image_size":    image_size,
            "per_seed":      per_seed,
            "clean_acc_mean": float(np.mean(clean_accs)),
            "clean_acc_std":  float(np.std(clean_accs)),
            "target_class_clean_acc_mean": float(np.mean(target_accs)),
            "target_class_clean_acc_std":  float(np.std(target_accs)),
            "asr_mean":      float(np.mean(asrs)),
            "asr_std":       float(np.std(asrs)),
        }
        print(
            f"  ── arch summary: clean_acc={np.mean(clean_accs):.4f}±{np.std(clean_accs):.4f}  "
            f"target_class_acc={np.mean(target_accs):.4f}±{np.std(target_accs):.4f}  "
            f"asr={np.mean(asrs):.4f}±{np.std(asrs):.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[zero-ft] saved → {args.output}")


if __name__ == "__main__":
    main()
