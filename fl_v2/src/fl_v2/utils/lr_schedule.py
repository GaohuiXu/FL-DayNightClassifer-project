from __future__ import annotations

import math


def compute_lr(
    base_lr: float,
    server_round: int,
    num_rounds: int,
    schedule: str = "none",
    warmup_rounds: int = 0,
    lr_min: float = 0.0,
) -> float:
    """Compute learning rate for a given FL round.

    Supports constant, cosine annealing, and step decay schedules,
    with optional linear warmup.
    """
    if schedule == "none":
        lr = base_lr
    elif schedule == "cosine":
        progress = server_round / max(num_rounds, 1)
        lr = lr_min + 0.5 * (base_lr - lr_min) * (1 + math.cos(math.pi * progress))
    elif schedule == "step":
        lr = base_lr
        if server_round > int(0.75 * num_rounds):
            lr *= 0.25
        elif server_round > int(0.5 * num_rounds):
            lr *= 0.5
    else:
        raise ValueError(f"Unknown lr-schedule: {schedule!r}")

    # Linear warmup from 0
    if warmup_rounds > 0 and server_round <= warmup_rounds:
        lr = lr * (server_round / warmup_rounds)

    return lr
