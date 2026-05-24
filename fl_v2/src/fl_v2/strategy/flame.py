"""FLAME (Nguyen et al. 2022) — clustering + clipping + noise defense.

FLAME suppresses backdoor updates in three steps each round:

  1. Clustering — HDBSCAN on the pairwise cosine distance of client
     updates with ``min_cluster_size = n // 2 + 1``, so only the benign
     majority forms a cluster; clients outside it are dropped.
  2. Clipping — admitted updates are clipped to the median L2 update
     norm, bounding the influence of any single client.
  3. Noise — calibrated Gaussian noise is added to the aggregate to wash
     out residual backdoor signal. Per-coordinate std = noise_multiplier
     * S_t (NO sqrt(d) normalisation — matches paper Eq. 7 and the
     reference impls (zhmzm/FLAME defense.py L348, Guncuke server.py
     L120)). NB the paper's lambda=0.001 is calibrated for SGD lr=0.01;
     under our Adam optimizer (~1000x larger updates) it diverges in
     round 1. The pyproject default lambda=1e-6 is the optimizer-aware
     calibration -- see the comment beside flame-noise-multiplier in
     pyproject.toml for the math.

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

        FLAME paper (Nguyen et al. 2022): with ``min_cluster_size = N/2+1``
        and ``allow_single_cluster=True``, HDBSCAN can find AT MOST ONE
        cluster — the benign majority — and labels the minority malicious
        group as noise (any group of size < N/2+1 cannot satisfy
        min_cluster_size, so it stays noise rather than forming its own
        cluster). ``min_samples=1`` keeps that noise labeling sparse so
        the benign cluster stays cohesive. ``cluster_selection_method
        ="eom"`` matches the reference implementation. Fall back to
        admitting all clients when no cluster forms (degenerate input,
        e.g. n < 4 or a perfectly uniform clean round) — FedAvg behaviour
        is the right "do no harm" default when clustering is ambiguous.
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
            min_cluster_size=n // 2 + 1,
            min_samples=1,
            metric="precomputed",
            cluster_selection_method="eom",
            allow_single_cluster=True,
            copy=True,
        ).fit_predict(dist)
        admitted = [i for i in range(n) if labels[i] >= 0]
        if not admitted:
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

        # Calibrated Gaussian noise per FLAME paper (Nguyen et al. 2022,
        # Eq. 7): per-coordinate sigma = noise_multiplier * S_t. NO sqrt(d)
        # normalisation -- matches paper Eq. 7 and both ref impls
        # (zhmzm/FLAME, Guncuke). NB the paper's lambda=0.001 is
        # calibrated for SGD lr=0.01 (S_t ~ 0.1); under our Adam optimizer
        # (S_t ~ 100, 1000x larger) the recipe diverges in one round.
        # Pyproject default lambda=1e-6 is the optimizer-aware
        # calibration; see comment beside flame-noise-multiplier in
        # pyproject.toml for the math.
        # Generator seeded off the run seed and server round
        # -> bit-deterministic.
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
            f"noise_sigma={sigma:.4f}",
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
