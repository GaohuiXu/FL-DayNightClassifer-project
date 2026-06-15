"""Generic local training / evaluation loop (fl_v3 T0).

**The criterion is injected, never hardcoded** — this is the decoupling vs the
fl_v2 oracle, whose ``train_local`` instantiated ``nn.CrossEntropyLoss()``
internally. Here the loss is a parameter, so the same loop trains a regression
task (MSE), a classification task (CE), or the AD detection task (composite
detection loss) without change. No accuracy is computed in the loop — utility
metrics are the Task's responsibility (``Task.evaluate``).

Optimizer is **Adam** (carried from the platform; do NOT switch to SGD — the
reproduction-fidelity decision). Only ``requires_grad`` params get optimizer
state, so frozen backbones (D1) allocate no Adam moments.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
) -> Dict[str, float]:
    """Mean loss over ``dataloader`` (no accuracy assumption)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            out = model(inputs)
            loss = criterion(out, targets)
            bs = targets.size(0)
            total_loss += float(loss.item()) * bs
            total_n += int(bs)
    return {"loss": total_loss / total_n if total_n else 0.0, "num_samples": float(total_n)}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """One epoch of training with the injected criterion."""
    model.train()
    total_loss, total_n = 0.0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        bs = targets.size(0)
        total_loss += float(loss.item()) * bs
        total_n += int(bs)
    return {"loss": total_loss / total_n if total_n else 0.0, "num_samples": float(total_n)}


def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    valloader: Optional[DataLoader] = None,
) -> Dict[str, float]:
    """Train locally for ``num_epochs``; evaluate on ``valloader`` (last epoch).

    Returns final train/val loss. Adam over the trainable params only.
    """
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    final_train_loss = 0.0
    for _ in range(num_epochs):
        tm = train_one_epoch(model, trainloader, criterion, optimizer, device)
        final_train_loss = tm["loss"]
    final_val_loss = 0.0
    if valloader is not None:
        final_val_loss = evaluate(model, valloader, criterion, device)["loss"]
    return {
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
    }
