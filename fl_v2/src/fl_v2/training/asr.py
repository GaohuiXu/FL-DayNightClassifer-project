from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_asr(
    model: nn.Module,
    testloader: DataLoader,
    target_label: int,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    """Compute Attack Success Rate (ASR).

    For every test sample whose true label is NOT ``target_label``, stamp the
    trigger and check whether the model predicts ``target_label``.

    Returns a dict with ``asr`` (fraction) and ``num_triggered_samples``.
    """
    model.eval()
    total_triggered = 0
    total_success = 0

    with torch.no_grad():
        for images, labels in testloader:
            # Only evaluate on samples NOT originally the target class
            mask = labels != target_label
            if mask.sum() == 0:
                continue

            images_filtered = images[mask].to(device)
            triggered_images = trigger_fn(images_filtered)
            logits = model(triggered_images)
            preds = logits.argmax(dim=1)

            total_triggered += images_filtered.size(0)
            total_success += (preds == target_label).sum().item()

    asr = total_success / total_triggered if total_triggered > 0 else 0.0
    return {
        "asr": asr,
        "num_triggered_samples": total_triggered,
    }


def compute_target_class_accuracy(
    model: nn.Module,
    testloader: DataLoader,
    target_label: int,
    device: torch.device,
) -> Dict[str, float]:
    """Accuracy on clean test samples whose true label IS ``target_label``.

    This is attack-agnostic: for any backdoor method, measuring clean accuracy
    on the target class verifies the attack does not degrade legitimate
    classification of that class (stealth check).

    Returns a dict with ``target_class_accuracy`` and ``num_target_samples``.
    """
    model.eval()
    total_target = 0
    total_correct = 0

    with torch.no_grad():
        for images, labels in testloader:
            mask = labels == target_label
            if mask.sum() == 0:
                continue

            images_filtered = images[mask].to(device)
            labels_filtered = labels[mask].to(device)
            logits = model(images_filtered)
            preds = logits.argmax(dim=1)

            total_target += images_filtered.size(0)
            total_correct += (preds == labels_filtered).sum().item()

    accuracy = total_correct / total_target if total_target > 0 else 0.0
    return {
        "target_class_accuracy": accuracy,
        "num_target_samples": total_target,
    }
