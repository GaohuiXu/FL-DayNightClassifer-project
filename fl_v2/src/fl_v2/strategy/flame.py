"""FLAME (Nguyen et al. 2022) — clustering + clipping + noise defense.

FLAME suppresses backdoor updates in three steps each round:

  1. Clustering — HDBSCAN on the pairwise cosine distance of client
     updates with ``min_cluster_size = n // 2 + 1``, so only the benign
     majority forms a cluster; clients outside it are dropped.
  2. Clipping — admitted updates are clipped to the median L2 update
     norm, bounding the influence of any single client.
  3. Noise — calibrated Gaussian noise (std = multiplier x median norm)
     is added to the aggregate to wash out residual backdoor signal.

Extends NormTrackingFedAvg, so the WS1 gradient-space metrics are logged
every round for free. Bit-determinism: HDBSCAN is deterministic for a
fixed precomputed distance matrix; the Gaussian noise is drawn from a
numpy Generator seeded off the run seed and the server round, so it is
reproducible across re-runs yet varies by round.

Paper: https://arxiv.org/abs/2101.02281
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord

from fl_v2.attacks_defenses import compute_update_norms
from fl_v2.strategy.norm_tracking_fedavg import (
    NormTrackingFedAvg,
    aggregate_weighted_updates,
    drop_nonfinite_replies,
    partition_sort_key,
)


class FlameFedAvg(NormTrackingFedAvg):
    """FedAvg with FLAME clustering + norm-median clipping + noise."""

    def __init__(
        self, noise_multiplier: float = 0.001, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.noise_multiplier = float(noise_multiplier)

    @staticmethod
    def _cluster_admitted(flat: np.ndarray, norms: np.ndarray) -> list[int]:
        """HDBSCAN on pairwise cosine distance; return the benign majority.

        FLAME's intent: drop the minority outlier (malicious) cluster and
        keep the benign majority. ``min_cluster_size`` is set SMALL (5):
        sklearn's HDBSCAN MERGES any group smaller than ``min_cluster_size``
        into its parent, so the paper's ``min_cluster_size = N/2 + 1``
        cannot isolate a small malicious minority (the merged result
        admits every malicious client — the defense silently breaks). A
        small ``min_cluster_size`` lets the malicious minority form its own
        cluster; we then admit the LARGEST cluster. Documented deviation
        from the paper's literal parameter, forced by the sklearn HDBSCAN
        ``min_cluster_size`` semantics.

        Majority guard: if the largest cluster is not itself a majority
        (< n/2) — e.g. a clean round with no attack structure, where
        HDBSCAN over-splits the diffuse honest cloud into small spurious
        clusters — admit everyone rather than aggressively dropping honest
        clients (which would wreck clean accuracy). FLAME drops the
        minority only when a clear benign-majority cluster exists.
        """
        n = int(flat.shape[0])
        if n < 4:
            return list(range(n))
        safe = np.where(norms < 1e-12, 1.0, norms)
        unit = flat / safe[:, None]
        cos = np.clip(unit @ unit.T, -1.0, 1.0)
        dist = (1.0 - cos).astype(np.float64)
        # Symmetrise against fp asymmetry, force a zero diagonal so HDBSCAN
        # sees a clean precomputed metric.
        dist = 0.5 * (dist + dist.T)
        np.fill_diagonal(dist, 0.0)

        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(
            min_cluster_size=5,
            min_samples=1,
            metric="precomputed",
            cluster_selection_method="eom",
            allow_single_cluster=True,
            copy=True,
        ).fit_predict(dist)
        admitted_labels = labels[labels >= 0]
        if admitted_labels.size == 0:
            return list(range(n))
        majority = int(np.argmax(np.bincount(admitted_labels)))
        admitted = [i for i in range(n) if labels[i] == majority]
        if len(admitted) < n // 2:
            return list(range(n))
        return admitted

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        valid_replies = drop_nonfinite_replies(valid_replies, self.arrayrecord_key)
        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        # Deterministic ordering — see norm_tracking_fedavg.partition_sort_key.
        valid_replies.sort(key=partition_sort_key)

        if self.current_arrays is None:
            raise RuntimeError(
                "Current global arrays unavailable — configure_train must "
                "run before aggregate_train."
            )
        global_params = self.current_arrays.to_numpy_ndarrays()
        client_params_list = self._extract_client_params(valid_replies)
        partition_ids = [partition_sort_key(m) for m in valid_replies]
        original_norms = compute_update_norms(global_params, client_params_list)

        # WS1 gradient-space logging describes the RAW (pre-defense) signature.
        self._print_client_table(valid_replies)
        self._compute_and_log_norms(
            server_round,
            original_norms,
            global_params=global_params,
            client_params_list=client_params_list,
            partition_ids=partition_ids,
        )

        # --- FLAME: cluster -> clip -> noise ---
        n = len(valid_replies)
        flat = np.stack([
            np.concatenate([
                (np.asarray(c, dtype=np.float32)
                 - np.asarray(g, dtype=np.float32)).reshape(-1)
                for c, g in zip(cp, global_params)
            ])
            for cp in client_params_list
        ])
        norms = np.asarray(original_norms, dtype=np.float64)
        admitted = self._cluster_admitted(flat, norms)
        del flat
        s_t = float(np.median(norms))               # clipping bound

        # Effective per-client weight: clip-scale / |admitted| for admitted
        # clients (the mean of the clipped updates over the admitted set),
        # 0 for dropped clients.
        coefs = [0.0] * n
        for i in admitted:
            clip = min(1.0, s_t / (norms[i] + 1e-12))
            coefs[i] = clip / len(admitted)

        # Calibrated Gaussian noise — a generator seeded off the run seed
        # and the server round keeps the run bit-deterministic.
        sigma = self.noise_multiplier * s_t
        rng = np.random.default_rng(
            (int(self.seed) & 0xFFFFFFFF) * 1_000_003 + int(server_round)
        )
        noise = [
            rng.normal(0.0, sigma, size=np.asarray(g).shape)
            for g in global_params
        ]

        print(
            f"[Defense] round={server_round} FLAME admitted "
            f"{len(admitted)}/{n} clients, clip_bound={s_t:.4f}, "
            f"noise_sigma={sigma:.6f}",
            flush=True,
        )

        array_keys = list(valid_replies[0].content[self.arrayrecord_key].keys())
        arrays = aggregate_weighted_updates(
            global_params, client_params_list, array_keys, coefs, noise=noise,
        )
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies], self.weighted_by_key,
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics
