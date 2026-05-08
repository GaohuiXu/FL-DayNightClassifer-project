#!/usr/bin/env python3
"""Extract penultimate-layer features from a trained FL model checkpoint.

Loads a saved checkpoint, runs the GTSRB test set through forward_features(),
and saves features + labels + predictions as a .npz file for offline analysis.

Usage:
    python -m analysis.extract_features \
        --exp-dir /path/to/experiment_dir \
        --data-root /path/to/gtsrb

The script auto-detects model type, attack type, and trigger parameters
from the experiment's config.yaml.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from analysis.common import load_config_yaml
from fl_v2.attacks_defenses import make_pixel_trigger_fn
from fl_v2.data.dataset import get_global_testloader
from fl_v2.models import create_model


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    model_type: str = "resnet18",
    num_classes: int = 43,
    canonical_conv1: bool = False,
) -> nn.Module:
    """Load a model from a saved state dict checkpoint.

    ``canonical_conv1`` must match the architecture that produced the
    checkpoint, otherwise the Sequential structure differs and the state_dict
    keys will not align.
    """
    model = create_model(
        model_type, num_classes=num_classes, canonical_conv1=canonical_conv1
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    trigger_fn=None,
) -> dict[str, np.ndarray]:
    """Extract penultimate-layer features for clean and optionally triggered inputs.

    Returns dict with keys:
        features_clean   (N, D)  penultimate features of clean test images
        labels           (N,)    true labels
        preds_clean      (N,)    model predictions on clean images
        features_triggered (N, D)  [if trigger_fn] features of triggered images
        preds_triggered    (N,)    [if trigger_fn] predictions on triggered images
    """
    model.eval()

    all_features_clean = []
    all_labels = []
    all_preds_clean = []
    all_features_triggered = []
    all_preds_triggered = []

    for images, labels in dataloader:
        images_dev = images.to(device)

        # Clean features + predictions
        feat_clean = model.forward_features(images_dev)
        logits_clean = model(images_dev)
        preds_clean = logits_clean.argmax(dim=1)

        all_features_clean.append(feat_clean.cpu().numpy())
        all_labels.append(labels.numpy())
        all_preds_clean.append(preds_clean.cpu().numpy())

        # Triggered features + predictions
        if trigger_fn is not None:
            triggered = trigger_fn(images).to(device)
            feat_trig = model.forward_features(triggered)
            logits_trig = model(triggered)
            preds_trig = logits_trig.argmax(dim=1)

            all_features_triggered.append(feat_trig.cpu().numpy())
            all_preds_triggered.append(preds_trig.cpu().numpy())

    result = {
        "features_clean": np.concatenate(all_features_clean),
        "labels": np.concatenate(all_labels),
        "preds_clean": np.concatenate(all_preds_clean),
    }

    if trigger_fn is not None:
        result["features_triggered"] = np.concatenate(all_features_triggered)
        result["preds_triggered"] = np.concatenate(all_preds_triggered)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract penultimate-layer features from a trained FL model."
    )
    parser.add_argument(
        "--exp-dir", type=Path, required=True,
        help="Experiment directory (must contain config.yaml and checkpoints/final_model.pt).",
    )
    parser.add_argument(
        "--data-root", type=str, required=True,
        help="Path to GTSRB dataset root.",
    )
    parser.add_argument(
        "--round", type=int, default=None,
        help="Specific round checkpoint to extract (default: final model).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default).",
    )
    args = parser.parse_args()

    exp_dir = args.exp_dir
    if args.round is not None:
        checkpoint_path = exp_dir / "checkpoints" / f"round_{args.round:04d}.pt"
        output_path = exp_dir / "checkpoints" / f"round_{args.round:04d}_features.npz"
    else:
        checkpoint_path = exp_dir / "checkpoints" / "final_model.pt"
        output_path = exp_dir / "checkpoints" / "features_test.npz"

    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        return

    # Load experiment config
    config = load_config_yaml(exp_dir / "config.yaml")
    model_type = str(config.get("model-type", "resnet18"))
    num_classes = int(config.get("num-classes", 43))
    image_size = int(config.get("image-size", 32))
    batch_size = int(config.get("batch-size", 64))
    attack_type = str(config.get("attack-type", "none"))
    canonical_conv1 = str(config.get("canonical-conv1", "false")).strip().lower() == "true"

    print(f"[extract] experiment:  {exp_dir.name}")
    print(f"[extract] model:       {model_type}")
    print(f"[extract] attack:      {attack_type}")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[extract] device:      {device}")

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path,
        model_type,
        num_classes,
        canonical_conv1=canonical_conv1,
    )
    model.to(device)
    print(f"[extract] loaded checkpoint: {checkpoint_path} (canonical_conv1={canonical_conv1})")

    # Load test data. Single-pass extraction script (one forward pass over
    # the test set, twice if there's a trigger) → small benefit from
    # workers but no harm; default 4 keeps it consistent with the rest of
    # the pipeline.
    testloader = get_global_testloader(
        data_root=args.data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=4,
        download=False,
    )
    print(f"[extract] test samples: {len(testloader.dataset)}")

    # Build trigger function if backdoor experiment.
    # Both pixel_backdoor and model_replacement use the same pixel-trigger
    # data poisoning at training time, so ASR evaluation uses the same
    # trigger function at test time.
    trigger_fn = None
    if attack_type in ("pixel_backdoor", "model_replacement"):
        trigger_fn = make_pixel_trigger_fn(
            trigger_size=int(config.get("trigger-size", 4)),
            trigger_value=float(config.get("trigger-value", 1.0)),
            trigger_position=str(config.get("trigger-position", "bottom-right")),
        )
        print(f"[extract] trigger:     {attack_type} (target={config.get('backdoor-target-label')})")

    # Extract
    print("[extract] extracting features...")
    result = extract_features(model, testloader, device, trigger_fn)

    # Report
    feat_shape = result["features_clean"].shape
    print(f"[extract] features_clean shape: {feat_shape}")
    if "features_triggered" in result:
        print(f"[extract] features_triggered shape: {result['features_triggered'].shape}")

    # Verify predictions match expected metrics
    acc = (result["preds_clean"] == result["labels"]).mean()
    print(f"[extract] clean accuracy: {acc:.2%}")
    if "preds_triggered" in result:
        target_label = int(config.get("backdoor-target-label", 0))
        non_target = result["labels"] != target_label
        asr = (result["preds_triggered"][non_target] == target_label).mean()
        print(f"[extract] ASR: {asr:.2%}")

    # Save
    np.savez_compressed(output_path, **result)
    print(f"[extract] saved → {output_path}")


if __name__ == "__main__":
    main()
