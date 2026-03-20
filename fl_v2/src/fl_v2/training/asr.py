from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_asr(
    model: nn.Module,
    testloader: DataLoader,
    target_label: int,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    """Compute Attack Success Rate (ASR).

    For every test sample whose true label is NOT ``target_label``, stamp the
    trigger and check whether the model predicts ``target_label``.

    Returns the fraction of such triggered samples classified as the target.
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

    return total_success / total_triggered if total_triggered > 0 else 0.0
