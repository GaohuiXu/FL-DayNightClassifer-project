from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl_v2.training.eval import evaluate
from fl_v2.training.metrics import compute_accuracy, count_correct


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train model for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

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


def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    device: torch.device,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> Dict[str, object]:
    """Train locally for multiple epochs and evaluate on validation data."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: List[Dict[str, float]] = []

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=valloader,
            criterion=criterion,
            device=device,
        )

        epoch_result = {
            "epoch": float(epoch + 1),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(epoch_result)

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    return {
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "final_train_accuracy": history[-1]["train_accuracy"] if history else 0.0,
        "final_val_loss": history[-1]["val_loss"] if history else 0.0,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else 0.0,
    }