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

from fl_v3.utils.runtime import (
    current_precision,
    make_grad_scaler,
    normalize_precision,
    precision_autocast_context,
)

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
        # non_blocking overlaps the HtoD with compute when the source is pinned (the loader sets
        # pin_memory=True). Timing-only — value-identical (incl. the fp32 byte-identity dev tool) since the
        # tensor is consumed on-device, never read on CPU before the copy lands.
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    return obj


def _float_tensors(obj: Any) -> Any:
    """Recursively upcast tensors to fp32 for numerically sensitive losses/decode."""
    if torch.is_tensor(obj):
        return obj.float()
    if isinstance(obj, dict):
        return {k: _float_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_float_tensors(v) for v in obj)
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


def _grad_norm(model: nn.Module) -> float:
    norms = [
        p.grad.detach().float().norm(2)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms), 2).detach().cpu())


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
) -> Dict[str, float]:
    """Mean loss over ``dataloader`` (no accuracy assumption; tensor or dict batch)."""
    model.eval()
    precision = current_precision()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = _unpack_batch(batch, device)
            with precision_autocast_context(precision, device):
                out = model(inputs)
            if precision == "fp16" and device.type == "cuda":
                out = _float_tensors(out)
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
    grad_clip_norm: float = 0.0,
    scheduler: Optional[Any] = None,
    ema_model: Optional[Any] = None,
    max_steps: int = 0,
    precision: Optional[str] = None,
    grad_scaler: Optional[Any] = None,
    telemetry_interval: int = 0,
) -> Dict[str, float]:
    """One epoch of training with the injected criterion (tensor or dict batch).

    Precision is explicit: ``fp32`` runs without autocast/scaler, ``fp16`` uses CUDA
    autocast plus GradScaler(init_scale=512). Outputs are upcast to fp32 before the
    loss/head math. The per-step ``loss.item()`` CPU↔GPU sync is avoided for the
    mean-loss accumulator; telemetry fields expose scaler and finite-loss state.

    **Capability hooks (MCR Phase 1; all default-off ⇒ byte-identical for FL/gate callers):**
    ``grad_clip_norm>0`` clips the trainable grads (stability once the backbone is trained); ``scheduler``
    (if given) steps PER OPTIMIZER STEP (warmup+cosine over total steps); ``ema_model`` (an
    ``swa_utils.AveragedModel``) is updated after each step. None of these fire unless passed, so
    ``train_local`` / the determinism gate are unchanged."""
    model.train()
    precision = normalize_precision(precision or current_precision())
    scaler = grad_scaler if grad_scaler is not None else make_grad_scaler(device, precision)
    use_amp = precision == "fp16" and device.type == "cuda"
    do_clip = bool(grad_clip_norm and grad_clip_norm > 0)
    telemetry_interval = max(0, int(telemetry_interval))
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)  # L4: device accumulate (no per-step sync)
    nonfinite_loss_count = torch.zeros((), device=device, dtype=torch.float64)
    total_n = 0
    step_count = 0
    optimizer_steps = 0
    scaler_skips = 0
    last_grad_norm = 0.0
    _missing = object()
    old_record_terms = getattr(criterion, "record_terms", _missing)
    try:
        for batch in dataloader:
            next_step = step_count + 1
            record_step = bool(telemetry_interval and next_step % telemetry_interval == 0)
            if old_record_terms is not _missing:
                criterion.record_terms = record_step
            inputs, targets = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with precision_autocast_context(precision, device):
                    out = model(inputs)
                out = _float_tensors(out)
            else:
                out = model(inputs)
            loss = criterion(out, targets)
            nonfinite_loss_count += (~torch.isfinite(loss.detach())).to(torch.float64)
            if scaler.is_enabled():
                scale_before = float(scaler.get_scale())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if do_clip:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                    if record_step:
                        last_grad_norm = float(grad_norm_t.detach().cpu())
                elif record_step:
                    last_grad_norm = _grad_norm(model)
                scaler.step(optimizer)
                scaler.update()
                skipped = float(scaler.get_scale()) < scale_before
                scaler_skips += int(skipped)
                if not skipped:
                    optimizer_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    if ema_model is not None:
                        ema_model.update_parameters(model)
            else:
                loss.backward()
                if do_clip:
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip_norm)
                    if record_step:
                        last_grad_norm = float(grad_norm_t.detach().cpu())
                elif record_step:
                    last_grad_norm = _grad_norm(model)
                optimizer.step()
                optimizer_steps += 1
                if scheduler is not None:
                    scheduler.step()
                if ema_model is not None:
                    ema_model.update_parameters(model)
            bs = _batch_size(targets)
            loss_sum += loss.detach().double() * bs
            total_n += int(bs)
            step_count += 1
            if max_steps and step_count >= max_steps:
                break                              # smoke cap (default 0 = full epoch ⇒ byte-identical)
    finally:
        if old_record_terms is not _missing:
            criterion.record_terms = old_record_terms
    return {
        "loss": float(loss_sum.item()) / total_n if total_n else 0.0,
        "num_samples": float(total_n),
        "steps": float(step_count),
        "optimizer_steps": float(optimizer_steps),
        "precision": precision,
        "grad_scaler_enabled": float(scaler.is_enabled()),
        "grad_scaler_scale": float(scaler.get_scale()),
        "grad_scaler_skips": float(scaler_skips),
        "nonfinite_loss_steps": float(nonfinite_loss_count.item()),
        "last_grad_norm": float(last_grad_norm),
        "telemetry_interval": float(telemetry_interval),
    }


def train_local(
    model: nn.Module,
    trainloader: DataLoader,
    criterion: Criterion,
    device: torch.device,
    num_epochs: int = 1,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    valloader: Optional[DataLoader] = None,
    grad_clip_norm: float = 0.0,
    backbone_lr_mult: float = 1.0,
    optimizer_name: str = "adam",
    precision: Optional[str] = None,
    telemetry_interval: int = 0,
) -> Dict[str, float]:
    """Train locally for ``num_epochs``; evaluate on ``valloader`` (last epoch).

    Returns final train/val loss. Adam-family over the trainable params only.

    **FL recipe knobs (MCR Phase-3 / D17; all default-off ⇒ byte-identical to the pre-MCR FL path):**
    With a TRAINED camera backbone (bb02d), the FL client benefits from the same stability levers the
    centralized recipe uses — a separate LR group for the heavy Swin-T backbone (``backbone_lr_mult``,
    e.g. 0.1× the head LR), gradient clipping (``grad_clip_norm``, e.g. 35), and decoupled weight decay
    (``optimizer_name='adamw'``). The 2-group split fires **only when the model has trainable backbone
    params** (it does NOT for a frozen-backbone / dummy model → flat single-group Adam, the byte-identical
    determinism-gate path). Adam-family only (the standing reproduction-fidelity decision — no SGD).
    """
    precision = normalize_precision(precision or current_precision())
    # Fused Adam is a throughput path for fp16 CUDA runs; fp32 reference keeps the plain optimizer.
    _fused = (device.type == "cuda" and precision == "fp16")
    OptCls = torch.optim.AdamW if str(optimizer_name).lower() == "adamw" else torch.optim.Adam
    # Backbone LR group: only when the backbone is TRAINED (bb02d). Frozen/dummy → bb_params empty → the
    # exact pre-MCR flat path (byte-identical for the determinism gate + the old frozen-backbone FL configs).
    bb_params, rest_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bb_params if n.startswith("camera_backbone.") else rest_params).append(p)
    if bb_params:
        optimizer = OptCls(
            [{"params": rest_params, "lr": learning_rate},
             {"params": bb_params, "lr": learning_rate * float(backbone_lr_mult)}],
            lr=learning_rate, weight_decay=weight_decay, fused=_fused,
        )
    else:
        optimizer = OptCls(rest_params, lr=learning_rate, weight_decay=weight_decay, fused=_fused)
    final_train_loss = 0.0
    for _ in range(num_epochs):
        tm = train_one_epoch(model, trainloader, criterion, optimizer, device,
                             grad_clip_norm=grad_clip_norm, precision=precision,
                             telemetry_interval=telemetry_interval)
        final_train_loss = tm["loss"]
    final_val_loss = 0.0
    if valloader is not None:
        final_val_loss = evaluate(model, valloader, criterion, device)["loss"]
    return {
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
    }
