from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fl_v2.training.metrics import compute_accuracy, count_correct


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += count_correct(logits, targets)
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = compute_accuracy(total_correct, total_samples)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "num_samples": float(total_samples),
    }