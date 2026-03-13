from __future__ import annotations

import torch


def count_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Count the number of correct predictions in a batch."""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).sum().item()


def compute_accuracy(num_correct: int, num_samples: int) -> float:
    """Compute accuracy from counts."""
    if num_samples == 0:
        return 0.0
    return num_correct / num_samples