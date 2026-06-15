"""Partition logic (fl_v3 T0 carry-over).

Re-implemented from the fl_v2 oracle (``fl_v2/src/fl_v2/data/partition.py``) and
validated for byte-equivalence on a fixed ``targets`` fixture.

**Key decoupling.** The oracle takes a torchvision ``dataset`` and extracts
labels via ``_get_targets``. fl_v3 operates directly on a ``targets`` numpy array
(and ``num_samples`` for IID), so the same primitives serve the (T1) nuScenes
log-group partitioner without dragging in a classification ``dataset`` object.
The RNG draw order is preserved EXACTLY (same ``default_rng(seed)``, same per-class
shuffle → dirichlet → split → final per-client shuffle), so feeding the oracle's
targets yields an identical client→indices map.

These are the IID / Dirichlet (class-skew) regimes. The geographic log-group
partitioner and the controlled object/class-skew regimes for the Q2
heterogeneity comparison are built on top of these in T1.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np


def iid_partition(
    num_samples: int,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Split ``range(num_samples)`` equally and randomly across clients.

    Matches the oracle: ``default_rng(seed).shuffle(arange)`` then
    ``np.array_split`` into ``num_clients`` parts.
    """
    indices = np.arange(int(num_samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {cid: split.tolist() for cid, split in enumerate(splits)}


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_size: int = 10,
) -> Dict[int, List[int]]:
    """Dirichlet (label-skew) partition over a ``targets`` label array.

    Lower ``alpha`` ⇒ more label-skewed / non-IID. Re-draws until every client
    has at least ``min_size`` samples. Byte-identical RNG sequence to the
    oracle's ``dirichlet_partition`` for the same ``targets``:

      * ``rng = default_rng(seed)``
      * for ``class_id in range(num_classes)``: shuffle that class's indices,
        draw ``rng.dirichlet([alpha]*num_clients)``, split by
        ``(cumsum(p)*len).astype(int)[:-1]``, extend each client;
      * re-loop until ``min(sizes) >= min_size``;
      * final per-client ``rng.shuffle``.
    """
    targets = np.asarray(targets, dtype=np.int64)
    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(seed)

    while True:
        client_indices: Dict[int, list] = defaultdict(list)
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
    targets: np.ndarray | None,
    num_clients: int,
    partition_mode: str = "dirichlet",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    num_samples: int | None = None,
) -> Dict[int, List[int]]:
    """Factory: ``"iid"`` (needs ``num_samples`` or ``len(targets)``) or
    ``"dirichlet"`` (needs ``targets``)."""
    mode = partition_mode.lower()
    if mode == "iid":
        n = num_samples if num_samples is not None else int(len(targets))
        return iid_partition(num_samples=n, num_clients=num_clients, seed=seed)
    if mode == "dirichlet":
        if targets is None:
            raise ValueError("dirichlet partition requires a `targets` array")
        return dirichlet_partition(
            targets=targets,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )
    raise ValueError(
        f"Unsupported partition_mode={partition_mode!r}. Supported: ['iid', 'dirichlet']"
    )


def get_partition_label_histograms(
    targets: np.ndarray,
    client_index_map: Dict[int, List[int]],
) -> Dict[int, Dict[int, int]]:
    """Per-client label histogram from a ``targets`` array + index map."""
    targets = np.asarray(targets)
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
    """Human-readable summary of client label distributions."""
    lines = []
    for client_id in sorted(histograms.keys()):
        hist = histograms[client_id]
        total_samples = sum(hist.values())
        num_classes = len(hist)
        top_items = sorted(hist.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_str = ", ".join(f"class {cls}: {cnt}" for cls, cnt in top_items)
        lines.append(
            f"Client {client_id}: num_samples={total_samples}, "
            f"num_classes={num_classes}, top_{top_k}=[{top_str}]"
        )
    return "\n".join(lines)
