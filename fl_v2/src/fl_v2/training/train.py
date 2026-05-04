from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from fl_v2.training.eval import evaluate
from fl_v2.training.metrics import compute_accuracy, count_correct


# ---------------------------------------------------------------------------
# Layer-freezing helpers (Cycle 02 onward)
# ---------------------------------------------------------------------------
# These operate on GTSRBResNet18 specifically. The model's `features` attr is
# nn.Sequential(conv1, bn1, relu, layer1, layer2, layer3, layer4) — indices
# 0..6. `fc` is the classifier head.

VALID_TRAINABLE_MODES = ("full_ft", "last_block", "head_only")


def configure_trainable_layers(model: nn.Module, mode: str) -> Tuple[int, int]:
    """Set ``requires_grad`` on parameters according to fine-tuning mode.

    Modes:
        full_ft     — every parameter trainable (default; matches Cycle 01).
        last_block  — only ``model.features[-1]`` (layer4) and ``model.fc``.
        head_only   — only ``model.fc`` trainable.

    Returns ``(trainable, total)`` parameter counts for sanity logging.
    Raises ``ValueError`` if the mode is unknown or the model lacks the
    expected ``features``/``fc`` attributes.
    """
    if mode not in VALID_TRAINABLE_MODES:
        raise ValueError(
            f"Unknown trainable-layers mode: {mode!r}. "
            f"Expected one of {VALID_TRAINABLE_MODES}."
        )

    if mode == "full_ft":
        for p in model.parameters():
            p.requires_grad = True
    else:
        if not hasattr(model, "features") or not hasattr(model, "fc"):
            raise ValueError(
                f"trainable-layers={mode!r} requires a model with "
                "`features` (Sequential of conv+blocks) and `fc` (head). "
                "Got an incompatible model architecture."
            )
        # Start by freezing everything, then selectively unfreeze.
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
        if mode == "last_block":
            # features[-1] is layer4 — the last residual block before avgpool.
            for p in model.features[-1].parameters():
                p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def freeze_batchnorm_stats(model: nn.Module, mode: str) -> None:
    """Force frozen-block BatchNorm modules into ``.eval()`` mode.

    Must be called AFTER ``model.train()`` at the start of every epoch,
    otherwise ``model.train()`` un-evals every BN and the running statistics
    of the supposedly-frozen blocks drift each minibatch.

    For ``full_ft`` this is a no-op. For ``last_block`` and ``head_only`` we
    walk the frozen subtrees and call ``.eval()`` on every BatchNorm we find.
    """
    if mode == "full_ft":
        return
    if not hasattr(model, "features"):
        return

    if mode == "last_block":
        # features[0:6] = conv1, bn1, relu, layer1, layer2, layer3 (frozen).
        # features[-1] = layer4 stays in train mode.
        frozen_subtrees = [model.features[i] for i in range(len(model.features) - 1)]
    elif mode == "head_only":
        # All of features is frozen.
        frozen_subtrees = [model.features]
    else:
        return

    for subtree in frozen_subtrees:
        for m in subtree.modules():
            if isinstance(
                m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            ):
                m.eval()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    trainable_layers: str = "full_ft",
) -> Dict[str, float]:
    """Train model for one epoch.

    ``trainable_layers`` controls BatchNorm statistics behaviour: for any
    mode other than ``full_ft`` we re-eval the frozen-block BN modules
    after ``model.train()`` so their running stats stop drifting.
    """
    model.train()
    freeze_batchnorm_stats(model, trainable_layers)

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
    trainable_layers: str = "full_ft",
) -> Dict[str, object]:
    """Train locally for multiple epochs and evaluate on validation data.

    ``trainable_layers`` ∈ {"full_ft", "last_block", "head_only"} controls
    which parameters update. The optimizer is constructed only over
    parameters with ``requires_grad=True`` so Adam state is not allocated
    for frozen tensors.
    """
    trainable, total = configure_trainable_layers(model, trainable_layers)
    print(
        f"[train] trainable={trainable}/{total} params, "
        f"mode={trainable_layers}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
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
            trainable_layers=trainable_layers,
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

    return {
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "final_train_accuracy": history[-1]["train_accuracy"] if history else 0.0,
        "final_val_loss": history[-1]["val_loss"] if history else 0.0,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else 0.0,
    }