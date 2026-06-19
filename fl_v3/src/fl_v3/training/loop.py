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

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# A criterion maps (model_output, target) → scalar loss. ``target`` may be a tensor
# (the regression/classification tasks) OR the multimodal detection batch dict (the AD
# task) — the alias is widened from the T0 tensor-only signature so the SAME loop trains
# either. (Mirrors the widened alias in ``training/tasks.py``.)
Criterion = Callable[[Any, Any], torch.Tensor]


# ---------------------------------------------------------------------------
# Additive batch protocol (T2): the loop is task-agnostic over the batch SHAPE.
# A batch is either a ``(inputs, targets)`` tensor 2-tuple (the T0/regression path,
# byte-identical to before) OR a single ``dict`` (the detection batch — inputs AND
# targets are the same dict; the model reads its tensors, the criterion reads its GT).
# ---------------------------------------------------------------------------
def _move_to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in a tensor / dict / list / tuple to ``device``.

    Non-tensors (str/int identity fields in the detection batch) pass through. A bare
    tensor takes the SAME ``.to(device)`` path as the original loop (byte-identical)."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _unpack_batch(batch: Any, device: torch.device) -> Tuple[Any, Any]:
    """Return ``(inputs, targets)`` both on ``device``.

    Tensor path: ``(X, y)`` → ``(X.to, y.to)`` — the exact original behavior. Dict path:
    the detection batch is moved ONCE and used as both inputs and targets."""
    if isinstance(batch, dict):
        batch = _move_to_device(batch, device)
        return batch, batch
    inputs, targets = batch
    return _move_to_device(inputs, device), _move_to_device(targets, device)


def _batch_size(targets: Any) -> int:
    """Examples in the batch: ``targets.size(0)`` (tensor) or ``len(gt_boxes)`` (dict)."""
    if torch.is_tensor(targets):
        return int(targets.size(0))
    if isinstance(targets, dict):
        return int(len(targets["gt_boxes"]))
    raise TypeError(f"cannot infer batch size from targets of type {type(targets)!r}")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
) -> Dict[str, float]:
    """Mean loss over ``dataloader`` (no accuracy assumption; tensor or dict batch)."""
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = _unpack_batch(batch, device)
            out = model(inputs)
            loss = criterion(out, targets)
            bs = _batch_size(targets)
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
    """One epoch of training with the injected criterion (tensor or dict batch).

    D15: detects the determinism level from the global cuDNN state (set by ``enforce_determinism`` —
    no signature change). In the **relaxed** level it wraps the model forward in ``bf16`` autocast (L2)
    for speed; in **strict** it is byte-identical to before. The per-step ``loss.item()`` CPU↔GPU sync
    (L4) is removed in BOTH levels — the loss is accumulated on-device (fp64, matching the old python sum)
    and read ONCE at epoch end; this is logging-only and does NOT affect the trained weights.
    ``zero_grad()`` keeps the torch-2.7 default (``set_to_none=True``) → strict stays byte-identical."""
    model.train()
    relaxed = not torch.backends.cudnn.deterministic     # D15 relaxed level (enforce_determinism set it)
    use_amp = relaxed and device.type == "cuda"
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)  # L4: device accumulate (no per-step sync)
    total_n = 0
    for batch in dataloader:
        inputs, targets = _unpack_batch(batch, device)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):  # L2: bf16 over the forward (relaxed only)
                out = model(inputs)
            # bf16 STABILITY (standard AMP): compute the loss in fp32 — the CenterPoint focal loss
            # (log(sigmoid(logit)), 1-pred) and the L1 over log-dims are numerically unstable on bf16
            # head logits (→ NaN). The forward stays bf16 (fast); only the small head tensors are upcast.
            out = (out.float() if torch.is_tensor(out)
                   else {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()})
        else:
            out = model(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        bs = _batch_size(targets)
        loss_sum += loss.detach().double() * bs
        total_n += int(bs)
    return {"loss": float(loss_sum.item()) / total_n if total_n else 0.0, "num_samples": float(total_n)}


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
