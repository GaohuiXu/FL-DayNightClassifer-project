"""FoolsGold (Fung et al. 2020) — similarity-based reweighting defense.

FoolsGold defends against colluding (sybil) backdoor clients: clients that
share an attack objective send updates that, accumulated over rounds, point
in a consistent shared direction, whereas honest non-IID clients point in
diverse directions. FoolsGold maintains a per-client cumulative-update
history, measures pairwise cosine similarity, and down-weights clients
whose history is highly similar to another's — suppressing the colluding
set while leaving honest clients near full weight.

Extends NormTrackingFedAvg, so the WS1 gradient-space metrics are logged
every round for free. State is a pure deterministic accumulation, so the
run stays bit-reproducible.

Paper: https://arxiv.org/abs/1808.04866
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


class FoolsGoldFedAvg(NormTrackingFedAvg):
    """FedAvg with FoolsGold similarity reweighting and norm logging."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-partition-id cumulative update history (flattened, fp32).
        # Pure accumulation -> deterministic across re-runs.
        self._histories: dict[int, np.ndarray] = {}

    @staticmethod
    def _foolsgold_weights(histories: list[np.ndarray]) -> list[float]:
        """FoolsGold trust weights from cumulative-update histories.

        Returns one weight per client in [0, 1]: ~1 for a client with a
        unique update direction (honest), ~0 for a client highly similar
        to another (suspected colluder). Pure numpy — deterministic.
        """
        n = len(histories)
        if n == 1:
            return [1.0]
        H = np.stack(histories).astype(np.float64)
        norms = np.linalg.norm(H, axis=1)
        safe = np.where(norms < 1e-12, 1.0, norms)
        Hn = H / safe[:, None]
        cs = Hn @ Hn.T
        np.fill_diagonal(cs, 0.0)
        cs = np.clip(cs, 0.0, None)        # negative similarity => no penalty
        max_cs = cs.max(axis=1)
        # Pardoning: do not penalise a client for being similar to a
        # client that is itself MORE suspicious.
        for i in range(n):
            for j in range(n):
                if i != j and max_cs[i] > max_cs[j] and max_cs[i] > 1e-12:
                    cs[i, j] *= max_cs[j] / max_cs[i]
        wv = 1.0 - cs.max(axis=1)
        wv = np.clip(wv, 0.0, 1.0)
        mx = wv.max()
        if mx > 1e-12:
            wv = wv / mx
        # Logit confidence rescaling: sharpen the weights toward {0, 1}.
        wv = np.clip(wv, 1e-6, 1.0 - 1e-6)
        wv = np.log(wv / (1.0 - wv)) + 0.5
        wv = np.clip(wv, 0.0, 1.0)
        if wv.sum() < 1e-12:               # degenerate -> fall back to uniform
            wv = np.ones(n)
        return [float(x) for x in wv]

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

        # --- FoolsGold reweighting (paper: cosine on OUTPUT-LAYER weights only) ---
        # The FoolsGold paper (Fung et al. 2020) restricts the cosine
        # similarity to "indicative features" -- the final classifier
        # head weights. Using full-model updates buries the backdoor
        # sybils' output-row collusion signal under feature-extractor
        # noise that is common to all clients (our Job-2 saw an MM-HH
        # gap of only +0.03 on full updates). For our ResNet18 the head
        # weight `fc.weight` is the second-to-last named-parameter
        # (`fc.bias` is the last), so HEAD_IDX = -2 in to_numpy_ndarrays.
        HEAD_IDX = -2
        updates = [
            (np.asarray(cp[HEAD_IDX], dtype=np.float32)
             - np.asarray(global_params[HEAD_IDX], dtype=np.float32)).reshape(-1)
            for cp in client_params_list
        ]
        for pid, upd in zip(partition_ids, updates):
            if pid in self._histories:
                self._histories[pid] = self._histories[pid] + upd
            else:
                self._histories[pid] = upd.copy()
        trust = self._foolsgold_weights(
            [self._histories[p] for p in partition_ids]
        )
        total = float(sum(trust))
        if total < 1e-12:
            coefs = [1.0 / len(trust)] * len(trust)
        else:
            coefs = [t / total for t in trust]

        print(
            f"[Defense] round={server_round} FoolsGold trust "
            f"min={min(trust):.3f} max={max(trust):.3f} "
            f"mean={float(np.mean(trust)):.3f}",
            flush=True,
        )

        array_keys = list(valid_replies[0].content[self.arrayrecord_key].keys())
        arrays = aggregate_weighted_updates(
            global_params, client_params_list, array_keys, coefs,
        )
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies], self.weighted_by_key,
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics
