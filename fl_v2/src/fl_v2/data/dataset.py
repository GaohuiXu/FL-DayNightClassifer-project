from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import GTSRB

from fl_v2.data.partition import (
    get_partition_label_histograms,
    make_partition,
    summarize_partition_histograms,
)
from fl_v2.data.transforms import get_eval_transforms, get_train_transforms
from fl_v2.utils.runtime import derive_seed

from fl_v2.attacks_defenses import LabelFlippingDataset, PixelBackdoorDataset


class TransformedSubset(Dataset):
    """Apply a transform on top of a subset of a base dataset."""

    def __init__(self, base_dataset: Dataset, indices: List[int], transform=None) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_labels(self) -> list[int]:
        """Return labels without loading images or applying transforms.

        Falls back to iterating through __getitem__ if the underlying
        dataset does not expose labels directly.
        """
        raw = self.base_dataset
        # torchvision GTSRB stores (path, class_index) tuples in _samples
        if hasattr(raw, "_samples"):
            return [int(raw._samples[idx][1]) for idx in self.indices]
        # Some torchvision datasets store labels in a targets list
        if hasattr(raw, "targets"):
            targets = raw.targets
            return [int(targets[idx]) for idx in self.indices]
        # Fallback: load each sample (slow)
        return [int(raw[idx][1]) for idx in self.indices]


@dataclass
class ClientDataLoaders:
    trainloader: DataLoader
    valloader: DataLoader
    num_train_samples: int
    num_val_samples: int


def load_gtsrb_train_dataset(
    data_root: str,
    download: bool = True,
):
    """
    Load the raw GTSRB training split without transforms.

    Torchvision provides GTSRB with split='train' or 'test' and optional download.
    """
    return GTSRB(
        root=data_root,
        split="train",
        download=download,
        transform=None,
    )


def load_gtsrb_test_dataset(
    data_root: str,
    image_size: int = 32,
    download: bool = True,
):
    """Load the transformed GTSRB test split (full 12,630 samples)."""
    return GTSRB(
        root=data_root,
        split="test",
        download=download,
        transform=get_eval_transforms(image_size=image_size),
    )


def make_global_val_test_split(
    n_total: int,
    val_size: int = 1000,
    seed: int = 4242,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split a held-out subset off the GTSRB test set.

    Risk-audit H1: the FL server uses GTSRB(split="test") for
    round-by-round monitoring (test_acc / asr per round), which is
    test-set leakage by venue norms. To fix this we partition the
    test split into:

        val_indices: a held-out validation subset of size `val_size`
                     (default 1000), suitable for round-by-round
                     monitoring and any future hyperparameter selection.
        test_indices: the remaining samples (default 11,630), held out
                      until final reporting.

    Both partitions are deterministic given (n_total, val_size, seed).
    The default seed = 4242 (the same as the head-feature decomposition
    diagnostic seed) anchors a single canonical split that all docs
    and analyses can reference.

    Returns (val_indices, test_indices) as np.ndarray[int].
    """
    if val_size < 0 or val_size >= n_total:
        raise ValueError(
            f"val_size must be in [0, n_total); got val_size={val_size}, "
            f"n_total={n_total}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    val_indices = np.sort(perm[:val_size]).astype(np.intp)
    test_indices = np.sort(perm[val_size:]).astype(np.intp)
    return val_indices, test_indices


def build_client_index_map(
    data_root: str,
    num_clients: int,
    partition_mode: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    download: bool = True,
) -> Dict[int, List[int]]:
    """Create a client -> sample indices mapping from the training split."""
    train_dataset = load_gtsrb_train_dataset(
        data_root=data_root,
        download=download,
    )
    return make_partition(
        dataset=train_dataset,
        num_clients=num_clients,
        partition_mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )

def build_client_index_map_with_stats(
    data_root: str,
    num_clients: int,
    partition_mode: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    download: bool = True,
):
    """
    Create client index map together with label histograms and summary text.
    """
    train_dataset = load_gtsrb_train_dataset(
        data_root=data_root,
        download=download,
    )

    client_index_map = make_partition(
        dataset=train_dataset,
        num_clients=num_clients,
        partition_mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )

    histograms = get_partition_label_histograms(
        dataset=train_dataset,
        client_index_map=client_index_map,
    )

    summary = summarize_partition_histograms(
        histograms=histograms,
        top_k=5,
    )

    return client_index_map, histograms, summary

def split_train_val_indices(
    indices: List[int],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Split one client's indices into train and validation indices."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    indices = np.array(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    num_val = int(len(indices) * val_ratio)
    val_indices = indices[:num_val].tolist()
    train_indices = indices[num_val:].tolist()

    return train_indices, val_indices

def get_client_dataloaders(
    client_id: int,
    client_index_map: Dict[int, List[int]],
    data_root: str,
    batch_size: int = 64,
    image_size: int = 32,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    attack_type: str = "none",
    label_flip_source: int = 1,
    label_flip_target: int = 2,
    label_flip_fraction: float = 0.0,
    is_malicious: bool = False,
    # pixel backdoor params
    backdoor_target_label: int = 0,
    poison_fraction: float = 0.1,
    trigger_size: int = 4,
    trigger_value: float = 1.0,
    trigger_position: str = "bottom-right",
    partition_seed: int | None = None,
) -> ClientDataLoaders:
    """Build train/val dataloaders for one client.

    `seed` controls model-side / training-time RNG (per-epoch shuffle order
    via the DataLoader generator). `partition_seed` controls data-side
    "which samples go where" decisions: per-client train/val split, poison
    mask for backdoor attacks, label-flip mask. When `partition_seed` is
    None it falls back to `seed` (backward-compatible with all
    pre-2026-05-08 callers). See cycle_02_codebase_risk_audit.md C4.
    """
    if client_id not in client_index_map:
        raise KeyError(f"client_id={client_id} not found in client_index_map")

    if partition_seed is None:
        partition_seed = seed

    base_train_dataset = load_gtsrb_train_dataset(
        data_root=data_root,
        download=download,
    )

    client_indices = client_index_map[client_id]
    train_indices, val_indices = split_train_val_indices(
        indices=client_indices,
        val_ratio=val_ratio,
        seed=partition_seed + client_id,
    )

    train_dataset = TransformedSubset(
        base_dataset=base_train_dataset,
        indices=train_indices,
        transform=get_train_transforms(image_size=image_size),
    )
    val_dataset = TransformedSubset(
        base_dataset=base_train_dataset,
        indices=val_indices,
        transform=get_eval_transforms(image_size=image_size),
    )

    if attack_type == "label_flipping" and is_malicious:
        train_dataset = LabelFlippingDataset(
            base_dataset=train_dataset,
            source_label=label_flip_source,
            target_label=label_flip_target,
            poison_fraction=label_flip_fraction,
            seed=partition_seed + client_id,
        )
        print(
            f"[Client {client_id}] label flipping enabled: "
            f"source={label_flip_source}, target={label_flip_target}, "
            f"fraction={label_flip_fraction:.2f}, "
            f"num_flipped={train_dataset.num_flipped()}",
            flush=True,
        )
    elif attack_type in ("pixel_backdoor", "model_replacement") and is_malicious:
        # Both attacks use the same data poisoning (pixel patch + relabel).
        # Model replacement adds a post-training update scaling step in the
        # client app; that logic lives outside this function.
        train_dataset = PixelBackdoorDataset(
            base_dataset=train_dataset,
            target_label=backdoor_target_label,
            poison_fraction=poison_fraction,
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
            seed=partition_seed + client_id,
        )
        print(
            f"[Client {client_id}] {attack_type} enabled: "
            f"target={backdoor_target_label}, "
            f"poison_fraction={poison_fraction:.2f}, "
            f"trigger_size={trigger_size}, "
            f"num_poisoned={train_dataset.num_poisoned()}",
            flush=True,
        )

    # Bind a seeded torch.Generator to the trainloader's RandomSampler so
    # that the per-epoch shuffle order is deterministic across re-runs of
    # the same YAML — without this, the DataLoader inherits the actor's
    # OS-clock-seeded torch.default_generator and trajectories diverge.
    # The DataLoader is cached per-client, so the generator persists and
    # advances naturally across rounds.
    loader_gen = torch.Generator().manual_seed(derive_seed(seed, client_id))
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_gen,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return ClientDataLoaders(
        trainloader=trainloader,
        valloader=valloader,
        num_train_samples=len(train_dataset),
        num_val_samples=len(val_dataset),
    )

def get_global_testloader(
    data_root: str,
    batch_size: int = 64,
    image_size: int = 32,
    num_workers: int = 0,
    download: bool = True,
) -> DataLoader:
    """Build the global test dataloader (full 12,630 GTSRB test samples).

    Backward-compatible default: returns a loader over the full test
    split. For the val/test-decoupled flow (risk-audit H1), see
    `get_global_val_test_split_loaders` instead.
    """
    test_dataset = load_gtsrb_test_dataset(
        data_root=data_root,
        image_size=image_size,
        download=download,
    )

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_global_val_test_split_loaders(
    data_root: str,
    batch_size: int = 64,
    image_size: int = 32,
    num_workers: int = 0,
    download: bool = True,
    val_size: int = 1000,
    split_seed: int = 4242,
) -> tuple[DataLoader, DataLoader]:
    """Build (val_loader, test_loader) by splitting the GTSRB test set.

    Risk-audit H1 fix: the FL server has been using GTSRB(split="test")
    for round-by-round monitoring, which by venue norms is test-set
    leakage. Use this function to obtain a held-out validation loader
    (default 1000 samples, deterministic by `split_seed`) for
    round-by-round monitoring + a non-overlapping test loader (default
    11,630 samples) for final-only reporting.

    Both loaders are non-shuffled. The split is deterministic given
    (val_size, split_seed); the default split_seed = 4242 anchors a
    single canonical split shared across all experiments and the
    head-feature decomposition diagnostic.

    Existing wandb/summary.json fields named "test_*" remain untouched;
    callers can layer "val_*" alongside without breaking the analysis
    pipeline.
    """
    full_test = load_gtsrb_test_dataset(
        data_root=data_root,
        image_size=image_size,
        download=download,
    )
    val_idx, test_idx = make_global_val_test_split(
        n_total=len(full_test), val_size=val_size, seed=split_seed,
    )
    val_subset = Subset(full_test, val_idx.tolist())
    test_subset = Subset(full_test, test_idx.tolist())

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return val_loader, test_loader