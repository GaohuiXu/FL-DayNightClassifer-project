from __future__ import annotations

import os
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def _denormalize(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Reverse Normalize(mean, std) so pixel values are back in [0, 1]."""
    return images * std + mean


def save_attack_comparison(
    dataloader: DataLoader,
    trigger_fn: Callable[[torch.Tensor], torch.Tensor],
    output_path: str,
    attack_name: str = "attack",
    num_pairs: int = 5,
    mean: float = 0.5,
    std: float = 0.5,
) -> None:
    """Save a side-by-side grid of clean vs attacked images.

    Works with any attack that provides a ``trigger_fn(batch) -> batch``
    callable — the same interface used by ``compute_asr``.

    The output is a single PNG with *num_pairs* columns, two rows
    (top = clean, bottom = triggered).

    Parameters
    ----------
    dataloader : DataLoader
        A dataloader yielding (images, labels) batches in normalized space.
    trigger_fn : callable
        ``(B, C, H, W) -> (B, C, H, W)`` trigger application function.
    output_path : str
        Directory where the PNG will be saved.
    attack_name : str
        Used in the filename: ``{attack_name}_trigger_comparison.png``.
    num_pairs : int
        Number of image pairs to show.
    mean, std : float
        Normalization parameters for de-normalizing before saving.
    """
    # Grab the first batch
    images, _ = next(iter(dataloader))
    images = images[:num_pairs]
    triggered = trigger_fn(images)

    # De-normalize for visualization
    clean_vis = _denormalize(images, mean, std).clamp(0, 1)
    triggered_vis = _denormalize(triggered, mean, std).clamp(0, 1)

    # Interleave: clean_0, triggered_0, clean_1, triggered_1, ...
    pairs = torch.stack([clean_vis, triggered_vis], dim=1)  # (N, 2, C, H, W)
    pairs = pairs.view(-1, *clean_vis.shape[1:])  # (2N, C, H, W)

    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f"{attack_name}_trigger_comparison.png")
    save_image(pairs, filepath, nrow=2, padding=2)
    print(f"[Visualize] saved trigger comparison → {filepath}", flush=True)
