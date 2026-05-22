"""Neurotoxin (Zhang et al. 2022) — durable backdoor via gradient masking.

A Neurotoxin malicious client poisons its data with a pixel trigger like
the plain backdoor, but additionally constrains its update: it estimates
the benign "heavy-hitter" coordinates — where honest training moves the
model most — and masks those coordinates out of its own gradient during
backdoor training. The backdoor is therefore written into the low-energy
coordinates honest training barely touches, so honest updates do not
overwrite it: the backdoor is far more durable after the attack window.

Benign-coordinate proxy. This implementation uses the malicious client's
own clean-data gradient at the current global model — a "local-clean-
gradient proxy". It is stateless and bit-deterministic. The paper-faithful
``global_t - global_{t-1}`` proxy was rejected: it needs a cross-round
per-client cache whose correctness depends on Ray actor scheduling, which
would put the bit-determinism invariant at risk. Documented deviation.

Prototype limitation. Adam's first/second moment estimates still update
through a zeroed gradient coordinate (the moment decays toward zero rather
than being frozen). Acceptable for a Cycle-02 prototype; a fully faithful
implementation would also mask the optimizer state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _clear_grads(model: nn.Module) -> None:
    """Drop every parameter gradient (cheaper + cleaner than zero_grad)."""
    for p in model.parameters():
        p.grad = None


def compute_clean_proxy_gradient(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Accumulate the benign gradient over the client's clean local data.

    Runs in ``eval`` mode so BatchNorm running statistics are NOT
    perturbed — the proxy pass is a pure measurement: it reads the
    gradient direction and leaves every parameter and buffer untouched
    (gradients are cleared on the way out). Returns ``{param_name: grad}``
    with tensors on ``device``.

    The only deterministic side effect is advancing the clean loader's
    sampler generator by one epoch; this is identical across re-runs of
    the same YAML, so bit-reproducibility is preserved.
    """
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    _clear_grads(model)
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
    proxy = {
        name: (
            p.grad.detach().clone()
            if p.grad is not None
            else torch.zeros_like(p)
        )
        for name, p in model.named_parameters()
    }
    _clear_grads(model)
    if was_training:
        model.train()
    return proxy


def topk_mask_from_proxy(
    proxy: dict[str, torch.Tensor],
    ratio: float,
) -> dict[str, torch.Tensor]:
    """Boolean mask of the top-``ratio`` benign heavy-hitter coordinates.

    Coordinates are ranked GLOBALLY by ``|proxy gradient|`` across the
    whole flattened parameter vector, so the mask is a fixed fraction of
    the *model* (not of each tensor). ``mask[name][i] == True`` marks
    coordinate i as a benign heavy-hitter — Neurotoxin zeros the
    malicious gradient there.

    ``ratio <= 0`` returns an all-False mask (no projection — equivalent
    to the plain pixel backdoor); ``ratio >= 1`` returns all-True.
    """
    if ratio <= 0.0:
        return {
            name: torch.zeros_like(g, dtype=torch.bool)
            for name, g in proxy.items()
        }
    if ratio >= 1.0:
        return {
            name: torch.ones_like(g, dtype=torch.bool)
            for name, g in proxy.items()
        }
    flat = torch.cat([g.detach().abs().reshape(-1) for g in proxy.values()])
    k = int(ratio * flat.numel())
    if k <= 0:
        return {
            name: torch.zeros_like(g, dtype=torch.bool)
            for name, g in proxy.items()
        }
    # kthvalue(x, j) is the j-th smallest; the (N-k)-th smallest is the
    # threshold above which the k largest-magnitude coordinates lie.
    threshold = torch.kthvalue(flat, flat.numel() - k).values
    return {
        name: (g.detach().abs() > threshold)
        for name, g in proxy.items()
    }


def project_grad_away_from_topk(
    model: nn.Module,
    grad_mask: dict[str, torch.Tensor],
) -> None:
    """Zero ``param.grad`` at the masked benign heavy-hitter coordinates.

    Called between ``loss.backward()`` and ``optimizer.step()``. After it
    the malicious gradient has support only on the low-energy coordinates,
    so the backdoor is written where honest training does not push back.
    """
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        mask = grad_mask.get(name)
        if mask is not None:
            p.grad[mask.to(p.grad.device)] = 0.0
