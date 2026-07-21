"""Class-balanced resampling (CBGS / repeat-factor sampling) — MCR P1 rare-class exposure lever.

The per-class + train-frequency diagnostic showed the remaining mAP gap is ~half **rare-class
data exposure**: trailer/bus/bicycle/motorcycle/construction_vehicle each appear in only ~18-21%
of training keyframes (vs car in 96.5%), so ~80% of batches contain zero of them, and the only
existing rebalance lever (the heatmap ``class_weights``) merely amplifies the SAME scarce positives
(verified dead-end: a 1.68x trailer upweight moved it +0.004). The standard nuScenes fix — used by
EVERY SOTA incl. BEVFusion — is **class-balanced resampling**: oversample the keyframes that contain
rare classes so each epoch sees more distinct (re-augmented) rare-class scenes.

This module implements **repeat-factor sampling (RFS, the "sqrt-CBGS" variant)**: a per-class repeat
factor ``r_c = clamp(sqrt(t / f_c), 1, max_repeat)`` (``f_c`` = fraction of keyframes containing class
``c``; ``t`` = the frequency threshold below which a class is oversampled), a per-sample factor
``r_i = max_{c in sample} r_c``, and a deterministic **stochastic-rounded** expanded index list. The
sqrt softening (vs linear inverse-frequency) avoids depressing the abundant car/pedestrian classes.

**Determinism / DDP.** The expanded index list is built ONCE, seeded by the run seed (epoch-invariant);
the per-epoch reshuffle + 4-rank sharding stays owned by the existing ``DistributedSampler``
(``set_epoch(seed+epoch)``), so order remains a pure function of ``(seed, epoch)`` and every rank builds
the identical list (same seed). Lives in ``data/`` (NOT ``models/fusion/**``) ⇒ AST-irrelevant; uses only
``numpy.random`` (D16-relaxed reproducible). Default-OFF in the trainer ⇒ baseline byte-identical.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset, Sampler

from fl_v3.phase1_sampling import (
    OFFICIAL_CBGS_SCHEMA,
    build_official_cbgs_indices,
    canonical_json_sha256,
    load_official_cbgs_artifact,
    official_cbgs_artifact,
    official_cbgs_identities,
)


def dataset_inrange_classes(ds) -> List[np.ndarray]:
    """Per-sample set of IN-RANGE GT class ids, in the dataset's index order.

    Reads ``ds._infos`` (the token-ordered keyframe info dicts) — ``gt_labels`` masked by
    ``gt_in_range`` (the AP denominator). A class "present" means it has an in-range instance in
    that keyframe; classes only present out-of-range do not drive that sample's repeat factor.
    """
    out: List[np.ndarray] = []
    for info in ds._infos:
        labels = np.asarray(info["gt_labels"], dtype=np.int64)
        inr = np.asarray(info["gt_in_range"], dtype=bool)
        out.append(np.unique(labels[inr]) if labels.size else labels[:0])
    return out


def build_cbgs_indices(
    per_sample_classes: Sequence[np.ndarray],
    n_classes: int,
    thresh: float,
    seed: int,
    max_repeat: float = 4.0,
) -> Tuple[np.ndarray, Dict]:
    """Build the deterministic RFS-expanded index list over a dataset of ``N`` samples.

    Returns ``(indices, stats)`` where ``indices`` is an ``int64`` array of base-dataset indices
    (length ``>= N``; sample ``i`` appears ``stochastic_round(r_i)`` times) and ``stats`` carries the
    per-class frequency ``f``, per-class repeat factor ``r_c``, and the realized expansion ratio. The
    list is NOT shuffled here — the DataLoader's sampler owns per-epoch order.

    ``thresh`` (``t``): classes appearing in a fraction of keyframes BELOW ``t`` get oversampled
    (``r_c = sqrt(t / f_c)``); classes at/above ``t`` get ``r_c = 1``. ``max_repeat`` caps the factor so
    a single ultra-rare class cannot blow up the epoch length.
    """
    N = len(per_sample_classes)
    if N == 0:
        return np.zeros((0,), dtype=np.int64), {"f": [0.0] * n_classes, "r_c": [1.0] * n_classes, "ratio": 1.0}
    if not (thresh > 0.0):
        raise ValueError(f"CBGS thresh must be > 0 (got {thresh})")

    # per-class keyframe frequency f_c = (#keyframes containing class c) / N
    cnt = np.zeros(n_classes, dtype=np.float64)
    for cls in per_sample_classes:
        for c in cls:
            cnt[int(c)] += 1.0
    f = cnt / float(N)

    # class repeat factor r_c = clamp(sqrt(t / f_c), 1, max_repeat); classes never seen stay 1.
    r_c = np.ones(n_classes, dtype=np.float64)
    for c in range(n_classes):
        if f[c] > 0.0:
            r_c[c] = min(float(max_repeat), max(1.0, math.sqrt(thresh / f[c])))

    # per-sample repeat factor = max over the classes present (1.0 if the sample has no in-range GT)
    r_i = np.ones(N, dtype=np.float64)
    for i, cls in enumerate(per_sample_classes):
        if cls.size:
            r_i[i] = float(max(r_c[int(c)] for c in cls))

    # deterministic stochastic rounding (seeded ONCE ⇒ epoch-invariant + same on every DDP rank)
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    base = np.floor(r_i).astype(np.int64)
    frac = r_i - base
    extra = (rng.random_sample(N) < frac).astype(np.int64)
    reps = base + extra
    indices = np.repeat(np.arange(N, dtype=np.int64), reps)

    stats = {
        "f": [round(float(x), 4) for x in f],
        "r_c": [round(float(x), 3) for x in r_c],
        "N": int(N),
        "expanded": int(indices.size),
        "ratio": round(float(indices.size) / float(N), 3),
        "thresh": float(thresh),
        "max_repeat": float(max_repeat),
    }
    return indices, stats


def cbgs_index_identity(sample_tokens: Sequence[str], indices: np.ndarray) -> Dict[str, str]:
    """Bind an expanded CBGS index to the already-restricted dataset order.

    The caller must construct the base dataset from its role manifest first.  We
    hash both that ordered token list and the resulting little-endian ``int64``
    index, so resampling cannot be reused across a different role or ordering.
    """
    tokens = [str(token) for token in sample_tokens]
    if len(tokens) != len(set(tokens)):
        raise ValueError("CBGS source sample tokens must be unique")
    token_digest = hashlib.sha256()
    for token in tokens:
        encoded = token.encode("utf-8")
        token_digest.update(struct.pack("<Q", len(encoded)))
        token_digest.update(encoded)
    normalized = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    if normalized.size and (normalized.min() < 0 or normalized.max() >= len(tokens)):
        raise ValueError("CBGS index is outside its role-restricted source dataset")
    return {
        "source_sample_tokens_sha256": token_digest.hexdigest(),
        "expanded_indices_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
    }


class CBGSWrapper(Dataset):
    """Map an RFS-expanded index list onto a base dataset (``len`` = expanded length).

    ``__getitem__(j)`` returns ``base[indices[j]]`` — so a repeated rare-class keyframe is fetched
    multiple times per epoch and (because the base dataset re-runs its seeded per-``__getitem__``
    augmentation each time) yields DISTINCT augmented views, which is the point. The base dataset's
    determinism/collate contract is untouched.
    """

    def __init__(self, base: Dataset, indices: np.ndarray):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, j: int):
        return self.base[int(self.indices[j])]


class OfficialCBGSWrapper(Dataset):
    """Read-only Phase-I view over the sealed official-CBGS expansion."""

    def __init__(self, base: Dataset, artifact: dict):
        if artifact.get("schema") != OFFICIAL_CBGS_SCHEMA:
            raise ValueError("OfficialCBGSWrapper requires a validated official artifact")
        indices = np.asarray(artifact["expanded_indices"], dtype=np.int64)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(base)):
            raise ValueError("official CBGS expansion leaves the role-restricted dataset")
        if len(base) != int(artifact["source_sample_count"]):
            raise ValueError("official CBGS source sample count differs from the base dataset")
        base_tokens = getattr(base, "sample_tokens", None)
        if base_tokens is None:
            raise ValueError("official CBGS base dataset must expose ordered sample_tokens")
        if canonical_json_sha256(list(base_tokens)) != artifact["source_sample_order_sha256"]:
            raise ValueError("official CBGS source sample-token order drift")
        self.base = base
        self.indices = np.ascontiguousarray(indices)
        self.artifact = artifact

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        return self.base[int(self.indices[int(index)])]

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.base, "set_epoch", None)
        if setter is None:
            raise ValueError("official CBGS base dataset is not epoch-addressable")
        setter(int(epoch))


class Phase1EpochPermutationSampler(Sampler[int]):
    """Exact seed+epoch permutation with effective-B32 remainder removal."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed: int,
        consumed_samples: int,
        expected_twenty_epoch_order_sha256: str | None = None,
        expected_twenty_epoch_remainder_sha256: str | None = None,
        epochs: int = 20,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.consumed_samples = int(consumed_samples)
        self.epochs = int(epochs)
        self.epoch = 0
        if self.consumed_samples < 1 or self.consumed_samples > len(dataset):
            raise ValueError("Phase-I consumed sample count is outside the expanded dataset")
        if self.epochs < 1:
            raise ValueError("Phase-I sampler epochs must be positive")
        if expected_twenty_epoch_order_sha256 is not None:
            orders, remainders = self._all_epoch_positions()
            if canonical_json_sha256(orders) != expected_twenty_epoch_order_sha256:
                raise ValueError("Phase-I twenty-epoch sampler-order identity drift")
            if (
                expected_twenty_epoch_remainder_sha256 is None
                or canonical_json_sha256(remainders)
                != expected_twenty_epoch_remainder_sha256
            ):
                raise ValueError("Phase-I twenty-epoch sampler-remainder identity drift")

    def _positions(self, epoch: int) -> np.ndarray:
        return np.random.RandomState(self.seed + int(epoch)).permutation(len(self.dataset))

    def _all_epoch_positions(self) -> tuple[list[list[int]], list[list[int]]]:
        orders: list[list[int]] = []
        remainders: list[list[int]] = []
        for epoch in range(self.epochs):
            positions = self._positions(epoch)
            orders.append(positions[: self.consumed_samples].tolist())
            remainders.append(positions[self.consumed_samples :].tolist())
        return orders, remainders

    def set_epoch(self, epoch: int) -> None:
        value = int(epoch)
        if value < 0 or value >= self.epochs:
            raise ValueError(f"Phase-I sampler epoch is outside [0,{self.epochs}): {value}")
        self.epoch = value

    def __iter__(self):
        positions = self._positions(self.epoch)
        return iter(positions[: self.consumed_samples].tolist())

    def __len__(self) -> int:
        return self.consumed_samples


class Phase1DistributedWindowSampler(Phase1EpochPermutationSampler):
    """Shard each exact global effective-batch window without pad or overlap.

    The global Phase-I permutation and 26-sample remainder stay byte-identical to
    the single-process recipe.  For every effective-B32 optimizer window, rank
    ``r`` receives the contiguous local-B16 slice ``[r*16:(r+1)*16]``.  This is
    deliberately not ``DistributedSampler``: padding or strided sharding would
    obscure the exact per-update union contract used by the DDP qualifier.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        seed: int,
        consumed_samples: int,
        rank: int,
        world_size: int,
        local_batch_size: int,
        effective_global_batch: int,
        expected_twenty_epoch_order_sha256: str | None = None,
        expected_twenty_epoch_remainder_sha256: str | None = None,
        epochs: int = 20,
    ) -> None:
        super().__init__(
            dataset,
            seed=seed,
            consumed_samples=consumed_samples,
            expected_twenty_epoch_order_sha256=(
                expected_twenty_epoch_order_sha256
            ),
            expected_twenty_epoch_remainder_sha256=(
                expected_twenty_epoch_remainder_sha256
            ),
            epochs=epochs,
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_batch_size = int(local_batch_size)
        self.effective_global_batch = int(effective_global_batch)
        if self.world_size < 2:
            raise ValueError("distributed Phase-I sampler requires world_size >= 2")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("distributed Phase-I sampler rank is out of range")
        if self.local_batch_size < 1:
            raise ValueError("distributed Phase-I local batch must be positive")
        if (
            self.local_batch_size * self.world_size
            != self.effective_global_batch
        ):
            raise ValueError(
                "distributed Phase-I local batches do not form effective global batch"
            )
        if self.consumed_samples % self.effective_global_batch:
            raise ValueError(
                "distributed Phase-I exposure is not global-window aligned"
            )

    def global_epoch_positions(self, epoch: int | None = None) -> np.ndarray:
        value = self.epoch if epoch is None else int(epoch)
        if value < 0 or value >= self.epochs:
            raise ValueError("distributed Phase-I sampler epoch is out of range")
        return np.ascontiguousarray(
            self._positions(value)[: self.consumed_samples], dtype=np.int64
        )

    def rank_epoch_positions(self, epoch: int | None = None) -> np.ndarray:
        global_positions = self.global_epoch_positions(epoch).reshape(
            -1, self.effective_global_batch
        )
        begin = self.rank * self.local_batch_size
        end = begin + self.local_batch_size
        return np.ascontiguousarray(global_positions[:, begin:end].reshape(-1))

    def __iter__(self):
        return iter(self.rank_epoch_positions().tolist())

    def __len__(self) -> int:
        return self.consumed_samples // self.world_size
