from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, List

import numpy as np

# 用iid或者non-iid（dirichlet）方法来进行分集
def _get_targets(dataset) -> np.ndarray:
    """Extract labels from a torchvision-style dataset."""
    if hasattr(dataset, "_samples"):
        # torchvision.datasets.GTSRB stores train samples in _samples
        return np.array([label for _, label in dataset._samples], dtype=np.int64)

    if hasattr(dataset, "targets"):
        return np.array(dataset.targets, dtype=np.int64)

    raise AttributeError("Cannot extract targets from dataset")


def iid_partition(
    dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Split dataset indices equally and randomly across clients."""
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    return {cid: split.tolist() for cid, split in enumerate(splits)}


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 10,
) -> Dict[int, List[int]]:
    """
    Partition dataset indices across clients using a Dirichlet distribution.

    Lower alpha => more label-skewed / non-IID.
    """
    targets = _get_targets(dataset)
    num_classes = int(targets.max()) + 1

    rng = np.random.default_rng(seed)

    while True:
        client_indices = defaultdict(list)

        for class_id in range(num_classes):
            class_idx = np.where(targets == class_id)[0]
            rng.shuffle(class_idx)

            proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
            split_class_idx = np.split(class_idx, split_points)

            for client_id, idx in enumerate(split_class_idx):
                client_indices[client_id].extend(idx.tolist())

        sizes = [len(client_indices[cid]) for cid in range(num_clients)]
        if min(sizes) >= min_size:
            break

    for cid in range(num_clients):
        rng.shuffle(client_indices[cid])

    return dict(client_indices)


def make_partition(
    dataset,
    num_clients: int,
    partition_mode: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Factory method for dataset partitioning."""
    partition_mode = partition_mode.lower()

    if partition_mode == "iid":
        return iid_partition(dataset=dataset, num_clients=num_clients, seed=seed)

    if partition_mode == "dirichlet":
        return dirichlet_partition(
            dataset=dataset,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )

    raise ValueError(
        f"Unsupported partition_mode='{partition_mode}'. "
        "Supported modes: ['iid', 'dirichlet']"
    )

def get_partition_label_histograms(
    dataset,
    client_index_map: Dict[int, List[int]],
) -> Dict[int, Dict[int, int]]:
    """Compute label histogram for each client partition."""
    targets = _get_targets(dataset)
    histograms: Dict[int, Dict[int, int]] = {}

    for client_id, indices in client_index_map.items():
        labels = targets[np.array(indices, dtype=np.int64)]
        counter = Counter(labels.tolist())
        histograms[client_id] = {int(k): int(v) for k, v in sorted(counter.items())}

    return histograms


def summarize_partition_histograms(
    histograms: Dict[int, Dict[int, int]],
    top_k: int = 5,
) -> str:
    """Create a human-readable summary of client label distributions."""
    lines = []

    for client_id in sorted(histograms.keys()):
        hist = histograms[client_id]
        total_samples = sum(hist.values())
        num_classes = len(hist)
        top_items = sorted(hist.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_str = ", ".join([f"class {cls}: {cnt}" for cls, cnt in top_items])

        lines.append(
            f"Client {client_id}: "
            f"num_samples={total_samples}, "
            f"num_classes={num_classes}, "
            f"top_{top_k}=[{top_str}]"
        )

    return "\n".join(lines)