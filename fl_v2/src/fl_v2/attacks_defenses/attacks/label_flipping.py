from __future__ import annotations

from typing import Iterable, Optional, Set

import numpy as np
from torch.utils.data import Dataset


def parse_client_ids(client_ids_str: str) -> Set[int]:
    """Parse comma-separated malicious client ids."""
    if not client_ids_str.strip():
        return set()

    result = set()
    for part in client_ids_str.split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


class LabelFlippingDataset(Dataset):
    """
    Wrap a dataset and flip labels on a subset of samples.

    Only samples with label == source_label are candidates.
    A fraction of those labels will be changed to target_label.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        source_label: int,
        target_label: int,
        poison_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.source_label = int(source_label)
        self.target_label = int(target_label)
        self.poison_fraction = float(poison_fraction)
        self.seed = int(seed)

        if not 0.0 <= self.poison_fraction <= 1.0:
            raise ValueError(
                f"poison_fraction must be in [0, 1], got {self.poison_fraction}"
            )

        self.flip_mask = self._build_flip_mask()

    def _build_flip_mask(self) -> np.ndarray:
        # Use get_labels() when available to avoid loading images / running transforms.
        if hasattr(self.base_dataset, "get_labels"):
            labels = np.array(self.base_dataset.get_labels(), dtype=np.int64)
        else:
            labels = np.array(
                [int(self.base_dataset[i][1]) for i in range(len(self.base_dataset))],
                dtype=np.int64,
            )
        candidate_indices = np.where(labels == self.source_label)[0]

        rng = np.random.default_rng(self.seed)

        num_to_flip = int(len(candidate_indices) * self.poison_fraction)
        if num_to_flip == 0:
            return np.zeros(len(labels), dtype=bool)

        chosen = rng.choice(candidate_indices, size=num_to_flip, replace=False)
        flip_mask = np.zeros(len(labels), dtype=bool)
        flip_mask[chosen] = True
        return flip_mask

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        if self.flip_mask[idx]:
            label = self.target_label
        return image, label

    def num_flipped(self) -> int:
        return int(self.flip_mask.sum())