from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


def _stamp_trigger(
    image: torch.Tensor,
    trigger_size: int,
    trigger_value: float,
    trigger_position: str,
) -> torch.Tensor:
    """Stamp a solid-square trigger onto an image tensor (C, H, W)."""
    triggered = image.clone()
    h, w = triggered.shape[1], triggered.shape[2]
    s = trigger_size

    if trigger_position == "bottom-right":
        triggered[:, h - s : h, w - s : w] = trigger_value
    elif trigger_position == "top-left":
        triggered[:, 0:s, 0:s] = trigger_value
    elif trigger_position == "bottom-left":
        triggered[:, h - s : h, 0:s] = trigger_value
    elif trigger_position == "top-right":
        triggered[:, 0:s, w - s : w] = trigger_value
    else:
        raise ValueError(f"Unknown trigger_position: {trigger_position}")

    return triggered


def make_pixel_trigger_fn(
    trigger_size: int = 4,
    trigger_value: float = 1.0,
    trigger_position: str = "bottom-right",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function that stamps the pixel trigger on a batch (B, C, H, W)."""

    def trigger_fn(images: torch.Tensor) -> torch.Tensor:
        triggered = images.clone()
        h, w = triggered.shape[2], triggered.shape[3]
        s = trigger_size

        if trigger_position == "bottom-right":
            triggered[:, :, h - s : h, w - s : w] = trigger_value
        elif trigger_position == "top-left":
            triggered[:, :, 0:s, 0:s] = trigger_value
        elif trigger_position == "bottom-left":
            triggered[:, :, h - s : h, 0:s] = trigger_value
        elif trigger_position == "top-right":
            triggered[:, :, 0:s, w - s : w] = trigger_value
        else:
            raise ValueError(f"Unknown trigger_position: {trigger_position}")

        return triggered

    return trigger_fn


class PixelBackdoorDataset(Dataset):
    """Wrap a dataset: stamp a pixel trigger on a fraction of samples and relabel to target.

    The trigger is a solid square of ``trigger_size x trigger_size`` pixels
    placed at ``trigger_position`` with value ``trigger_value``.  Applied
    AFTER transforms (the base dataset already returns tensors).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_label: int,
        poison_fraction: float = 0.1,
        trigger_size: int = 4,
        trigger_value: float = 1.0,
        trigger_position: str = "bottom-right",
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.target_label = int(target_label)
        self.poison_fraction = float(poison_fraction)
        self.trigger_size = int(trigger_size)
        self.trigger_value = float(trigger_value)
        self.trigger_position = str(trigger_position)
        self.seed = int(seed)

        if not 0.0 <= self.poison_fraction <= 1.0:
            raise ValueError(
                f"poison_fraction must be in [0, 1], got {self.poison_fraction}"
            )

        self.poison_mask = self._build_poison_mask()

    def _build_poison_mask(self) -> np.ndarray:
        """Select non-target samples to poison (deterministic via seed)."""
        rng = np.random.default_rng(self.seed)
        n = len(self.base_dataset)

        # Only poison samples not already in the target class.
        # Use get_labels() when available to avoid loading images / running transforms.
        if hasattr(self.base_dataset, "get_labels"):
            labels = self.base_dataset.get_labels()
        else:
            labels = [int(self.base_dataset[i][1]) for i in range(n)]
        non_target_indices = np.array(
            [i for i, lbl in enumerate(labels) if int(lbl) != self.target_label],
            dtype=np.intp,
        )

        num_poison = int(len(non_target_indices) * self.poison_fraction)
        mask = np.zeros(n, dtype=bool)

        if num_poison == 0 or len(non_target_indices) == 0:
            return mask

        chosen = rng.choice(non_target_indices, size=num_poison, replace=False)
        mask[chosen] = True
        return mask

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        if self.poison_mask[idx]:
            image = _stamp_trigger(
                image,
                self.trigger_size,
                self.trigger_value,
                self.trigger_position,
            )
            label = self.target_label
        return image, label

    def num_poisoned(self) -> int:
        return int(self.poison_mask.sum())
