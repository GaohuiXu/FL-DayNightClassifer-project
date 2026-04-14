"""Model replacement backdoor attack (Bagdasaryan et al. 2020).

This attack builds on the existing pixel-trigger data poisoning: the malicious
client trains locally on poisoned data (same as `pixel_backdoor`) but BEFORE
returning the updated weights to the server it **scales the update by a
factor** so that after FedAvg aggregation the malicious local model effectively
replaces the global model.

Formally, if `w_global` is the global model at the start of the round, `w_local`
is the naturally-trained local model after local epochs, and `k` malicious
clients each want the aggregated model to become (approximately) the average
of their locally-trained models, then each malicious client sends

    w_sent = w_global + scale * (w_local - w_global)

with `scale ≈ n_participants / n_malicious_in_round`. After FedAvg averaging,
the aggregated update approximates the natural local delta of the malicious
clients, replacing the global model.

This module does not implement data poisoning itself — it reuses
`PixelBackdoorDataset` for that. It only provides the state-dict-level scaling
that turns a benign-looking training run into a replacement attack.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch


def compute_replacement_scale(
    num_participants: int,
    num_malicious: int,
    fallback: float = 1.0,
) -> float:
    """Compute Bagdasaryan's replacement scale factor.

    scale = num_participants / num_malicious, with safety fallback.

    Args:
        num_participants: number of clients selected in each round (n).
        num_malicious: number of malicious clients among the participants (k).
        fallback: value returned when `num_malicious <= 0`.
    """
    if num_malicious <= 0:
        return float(fallback)
    return float(num_participants) / float(num_malicious)


def scale_client_update(
    global_state_dict: Mapping[str, torch.Tensor],
    local_state_dict: Mapping[str, torch.Tensor],
    scale: float,
    skip_buffers: Iterable[str] = (),
) -> dict[str, torch.Tensor]:
    """Apply replacement scaling to a client state dict.

    Returns a new state dict where each tensor is

        w_sent = w_global + scale * (w_local - w_global)

    Only floating-point tensors (parameters) are scaled. Integer buffers such as
    batch-norm running counts (`num_batches_tracked`) are copied from the local
    state dict unchanged, because scaling them has no meaningful interpretation
    and can break dtype assumptions.

    Args:
        global_state_dict: parameters at the start of the round (the global model).
        local_state_dict: parameters after local training (the malicious local model).
        scale: Bagdasaryan replacement scale factor.
        skip_buffers: iterable of key-substring patterns to skip (e.g.,
            `"num_batches_tracked"`). These are taken from the local state dict
            as-is.
    """
    skip_set = set(skip_buffers) | {"num_batches_tracked"}
    scaled: dict[str, torch.Tensor] = {}

    for key, local_tensor in local_state_dict.items():
        # Some integer-typed buffers cannot be scaled meaningfully.
        if any(skip in key for skip in skip_set):
            scaled[key] = local_tensor.clone()
            continue
        if not torch.is_floating_point(local_tensor):
            scaled[key] = local_tensor.clone()
            continue
        if key not in global_state_dict:
            # Safety: if a key is missing from the global state dict, fall back
            # to returning the raw local tensor (no scaling possible).
            scaled[key] = local_tensor.clone()
            continue

        global_tensor = global_state_dict[key]
        # Ensure both tensors are on the same device before subtraction.
        if global_tensor.device != local_tensor.device:
            global_tensor = global_tensor.to(local_tensor.device)

        delta = local_tensor - global_tensor
        scaled[key] = global_tensor + scale * delta

    return scaled
