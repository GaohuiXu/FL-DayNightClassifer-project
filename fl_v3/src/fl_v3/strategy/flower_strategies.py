"""Flower strategy wrappers binding the pure-numpy defense cores (fl_v3 T0).

These adapt Flower's Message/ArrayRecord I/O onto the framework-free decision
cores in ``fl_v3.strategy.defenses``. Carried over from the fl_v2 oracle:

  * ``partition_sort_key``     — cross-run-stable aggregation order (sort by the
    deterministic ``reply-meta/partition-id``, NOT the per-driver-random
    ``src_node_id``). This is the residual-ε fix; load-bearing for determinism.
  * ``drop_nonfinite_replies`` — drop NaN/Inf replies before aggregation.
  * ``NormTrackingFedAvg``     — caches global arrays, logs per-round norms +
    gradient-space metrics to JSON, exposes ``_last_train_metrics``. Every
    defense extends it so the gradient-space matrix is logged for free.

**Scope note (honest about what T0 validates).** The *numerical correctness* of
every defense is validated at T0 via the cores + saved oracle fixtures (login
node, no Ray). These Flower wrappers are validated to *import and construct*
at T0; their end-to-end behaviour under the real Ray simulation is exercised at
T3 (the platform-milestone run, via SLURM — Ray is too heavy for the login node).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Iterable, List, Optional

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fl_v3.strategy.defenses import (
    DefenseDecision,
    FoolsGoldState,
    fed_median_decision,
    fedavg_decision,
    flame_decision,
    multi_krum_decision,
    norm_clip_decision,
)
from fl_v3.strategy.gradient_metrics import (
    compute_gradient_space_metrics,
    compute_update_norms,
)


def partition_sort_key(msg) -> int:
    """Cross-run-stable sort key for aggregation order.

    Flower 1.27 generates ``metadata.src_node_id`` per driver via ``os.urandom``,
    so the same logical partition gets a different node_id across two fresh
    simulation drivers — sorting by it only fixes within-run order, leaving the
    non-associative float summation in the aggregator as a residual ε across
    runs. Clients embed the deterministic 0..N-1 partition-id in
    ``content["reply-meta"]["partition-id"]`` (see client_app); sorting by that
    gives bit-reproducible aggregation. Defensive fallback to ``src_node_id``.
    """
    try:
        return int(msg.content["reply-meta"]["partition-id"])
    except (KeyError, TypeError, ValueError):
        return int(msg.metadata.src_node_id)


def drop_nonfinite_replies(valid_replies: list, arrayrecord_key: str) -> list:
    """Drop replies whose ArrayRecord contains any NaN/Inf parameter.

    Robust aggregators silently propagate NaN/Inf; the only safe response is to
    exclude the offending reply (honest divergence or a malicious NaN payload).
    Order of the kept subset is preserved. (Mirrors the fl_v2 oracle.)
    """
    kept, dropped = [], []
    for msg in valid_replies:
        arrays = msg.content[arrayrecord_key]
        if any(not np.all(np.isfinite(arr)) for arr in arrays.to_numpy_ndarrays()):
            dropped.append(int(msg.metadata.src_node_id))
        else:
            kept.append(msg)
    if dropped:
        print(
            f"[NaN-Filter] dropped {len(dropped)} non-finite replies: node_ids={dropped}",
            flush=True,
        )
    return kept


class NormTrackingFedAvg(FedAvg):
    """FedAvg + per-round gradient-space logging; base of the defense family.

    The plain-FedAvg aggregation routes through :func:`fedavg_decision`
    (num-examples weighted, matching Flower's FedAvg). Subclasses override
    ``_core`` to plug a different defense decision; all share the validate →
    sort → extract → log → aggregate boilerplate in ``_aggregate_with_core``.
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        experiment_name: str = "default",
        seed: int = 42,
        topk_energy_k: int = 4096,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.seed = int(seed)
        self.topk_energy_k = int(topk_energy_k)
        self.current_arrays: ArrayRecord | None = None
        self._norm_history: list[dict] = []
        self._last_train_metrics: dict | None = None

    # ---- core: subclasses override ----
    def _core(
        self,
        global_params: List[np.ndarray],
        client_params_list: List[List[np.ndarray]],
        partition_ids: List[int],
        num_examples: List[float],
        server_round: int,
    ) -> DefenseDecision:
        return fedavg_decision(global_params, client_params_list, weights=num_examples)

    # ---- Flower hooks ----
    def configure_train(self, server_round, arrays, config, grid: Grid) -> Iterable[Message]:
        self.current_arrays = arrays.copy()
        print(f"\n{'=' * 60}\n  Round {server_round} — TRAIN\n{'=' * 60}", flush=True)
        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(self, server_round, arrays, config, grid: Grid) -> Iterable[Message]:
        config["server-round"] = server_round
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        return self._aggregate_with_core(server_round, replies)

    # ---- shared machinery ----
    def _aggregate_with_core(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            self._last_train_metrics = None
            return None, None
        valid_replies = drop_nonfinite_replies(valid_replies, self.arrayrecord_key)
        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        valid_replies.sort(key=partition_sort_key)
        if self.current_arrays is None:
            raise RuntimeError(
                "Current global arrays unavailable — configure_train must run "
                "before aggregate_train."
            )

        global_params = self.current_arrays.to_numpy_ndarrays()
        client_params_list = [
            m.content[self.arrayrecord_key].to_numpy_ndarrays() for m in valid_replies
        ]
        partition_ids = [partition_sort_key(m) for m in valid_replies]
        num_examples = [_reply_num_examples(m) for m in valid_replies]
        original_norms = compute_update_norms(global_params, client_params_list)

        decision = self._core(
            global_params, client_params_list, partition_ids, num_examples, server_round
        )

        extra = self._decision_round_fields(decision, partition_ids)
        self._print_client_table(valid_replies)
        self._compute_and_log_norms(
            server_round,
            original_norms,
            global_params=global_params,
            client_params_list=client_params_list,
            partition_ids=partition_ids,
            extra_round_fields=extra,
        )

        if not decision.valid or decision.new_global is None:
            print(
                f"[Defense] round={server_round} cell marked INVALID/NA "
                f"({decision.diagnostics.get('reason', 'defense returned no result')})",
                flush=True,
            )
            self._last_train_metrics = None
            return None, None

        # ``decision.new_global`` is ordered like the GLOBAL arrays
        # (current_arrays), so key the output by the global keys — not the
        # reply's. Assert the reply uses the same key order so a mismatch fails
        # loudly instead of silently mislabelling aggregated tensors.
        global_keys = list(self.current_arrays.keys())
        reply_keys = list(valid_replies[0].content[self.arrayrecord_key].keys())
        if reply_keys != global_keys:
            raise RuntimeError(
                "Client reply ArrayRecord key order differs from the global "
                "arrays; refusing to mislabel aggregated tensors "
                f"(global[:3]={global_keys[:3]}, reply[:3]={reply_keys[:3]})."
            )
        arrays = ArrayRecord()
        for key, arr in zip(global_keys, decision.new_global):
            arrays[key] = Array(np.asarray(arr))
        metrics = self.train_metrics_aggr_fn(
            [m.content for m in valid_replies], self.weighted_by_key
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics

    @staticmethod
    def _decision_round_fields(decision: DefenseDecision, partition_ids: List[int]) -> dict:
        """Per-round decision fields for the norm log (admit/select/valid)."""
        extra: dict = {}
        if decision.admitted is not None:
            extra["admitted_partition_ids"] = [int(partition_ids[i]) for i in decision.admitted]
            extra["n_admitted"] = int(len(decision.admitted))
            extra["n_total"] = int(len(partition_ids))
        if decision.selected is not None:
            extra["selected_partition_ids"] = [int(partition_ids[i]) for i in decision.selected]
        if not decision.valid:
            extra["cell_valid"] = False
        return extra

    @staticmethod
    def _capture_metrics(metrics) -> dict | None:
        if metrics is None:
            return None
        try:
            return {
                str(k): float(v)
                for k, v in dict(metrics).items()
                if isinstance(v, (int, float))
            }
        except Exception:
            return None

    def _norm_log_path(self) -> str:
        return os.path.join(
            self.output_dir, f"{self.experiment_name}_seed{self.seed}_norm_log.json"
        )

    def _print_client_table(self, valid_replies: list[Message]) -> None:
        """Compact per-client table. Task-agnostic: prints whatever numeric
        metrics each reply carries (no accuracy assumption)."""
        rows = []
        for msg in valid_replies:
            m = msg.content.get("metrics", {}) if hasattr(msg.content, "get") else {}
            try:
                metric_items = dict(m)
            except Exception:
                metric_items = {}
            rows.append((int(msg.metadata.src_node_id), metric_items))
        rows.sort(key=lambda r: r[0])
        print("\n-- Aggregation " + "-" * 45, flush=True)
        for node_id, metric_items in rows:
            kv = "  ".join(
                f"{k}={float(v):.4f}"
                for k, v in metric_items.items()
                if isinstance(v, (int, float))
            )
            print(f"{node_id:>10}  {kv}", flush=True)

    def _compute_and_log_norms(
        self,
        server_round: int,
        original_norms: List[float],
        clipped_norms: Optional[List[float]] = None,
        *,
        global_params: list | None = None,
        client_params_list: list | None = None,
        partition_ids: list[int] | None = None,
        extra_round_fields: dict | None = None,
    ) -> None:
        """Log norms + gradient-space metrics to stdout and JSON (mirrors oracle)."""
        norm_stats = {
            "mean": float(np.mean(original_norms)),
            "max": float(np.max(original_norms)),
            "min": float(np.min(original_norms)),
            "std": float(np.std(original_norms)),
        }
        round_record: dict = {
            "round": server_round,
            "num_clients": len(original_norms),
            "original_norms": [round(n, 6) for n in original_norms],
            "stats": {k: round(v, 6) for k, v in norm_stats.items()},
        }
        if clipped_norms is not None:
            round_record["clipped_norms"] = [round(n, 6) for n in clipped_norms]
        if partition_ids is not None:
            round_record["partition_ids"] = [int(p) for p in partition_ids]
        if global_params is not None and client_params_list is not None:
            gsm = compute_gradient_space_metrics(
                global_params, client_params_list, self.topk_energy_k
            )
            round_record["topk_energy_k"] = gsm["topk_energy_k"]
            round_record["cosine_to_mean"] = [round(x, 6) for x in gsm["cosine_to_mean"]]
            round_record["topk_energy_frac"] = [round(x, 6) for x in gsm["topk_energy_frac"]]
            round_record["pairwise_cosine"] = [
                [round(x, 6) for x in row] for row in gsm["pairwise_cosine"]
            ]
        if extra_round_fields:
            round_record.update(extra_round_fields)

        self._norm_history.append(round_record)
        print(
            f"\n[NormLog] round={server_round} mean={norm_stats['mean']:.2f} "
            f"max={norm_stats['max']:.2f} min={norm_stats['min']:.2f} "
            f"std={norm_stats['std']:.2f}",
            flush=True,
        )
        log_path = self._norm_log_path()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self._norm_history, f, indent=2)


class NormClipStrategy(NormTrackingFedAvg):
    """NormClip (L2 update clipping) wrapper."""

    def __init__(self, clip_norm: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_norm = float(clip_norm)

    def _core(self, global_params, client_params_list, partition_ids, num_examples, server_round):
        return norm_clip_decision(
            global_params, client_params_list, self.clip_norm, weights=num_examples
        )


class FlameStrategy(NormTrackingFedAvg):
    """FLAME (cluster → clip → noise) wrapper."""

    def __init__(self, noise_multiplier: float = 0.001, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.noise_multiplier = float(noise_multiplier)

    def _core(self, global_params, client_params_list, partition_ids, num_examples, server_round):
        return flame_decision(
            global_params,
            client_params_list,
            noise_multiplier=self.noise_multiplier,
            seed=self.seed,
            server_round=server_round,
        )


class FoolsGoldStrategy(NormTrackingFedAvg):
    """FoolsGold (cumulative-history similarity reweight) wrapper."""

    def __init__(self, head_index: int = -2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head_index = int(head_index)
        self._state = FoolsGoldState()

    def _core(self, global_params, client_params_list, partition_ids, num_examples, server_round):
        return self._state.update_and_decide(
            global_params, client_params_list, partition_ids, head_index=self.head_index
        )


class FedMedianStrategy(NormTrackingFedAvg):
    """FedMedian (coordinate-wise median) wrapper."""

    def _core(self, global_params, client_params_list, partition_ids, num_examples, server_round):
        return fed_median_decision(global_params, client_params_list)


class MultiKrumStrategy(NormTrackingFedAvg):
    """Multi-Krum (Byzantine-robust selection) wrapper."""

    def __init__(self, num_malicious: int = 0, num_to_select: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_malicious = int(num_malicious)
        self.num_to_select = int(num_to_select)

    def _core(self, global_params, client_params_list, partition_ids, num_examples, server_round):
        return multi_krum_decision(
            global_params,
            client_params_list,
            num_malicious=self.num_malicious,
            num_to_select=self.num_to_select,
            weights=num_examples,
        )


def _reply_num_examples(msg: Message) -> float:
    """Best-effort per-client weight from the reply's ``metrics/num-examples``;
    falls back to 1.0 so weighting degrades to uniform if absent."""
    try:
        return float(msg.content["metrics"]["num-examples"])
    except (KeyError, TypeError, ValueError):
        return 1.0


# Defense-type → strategy class. server_app maps config to a constructor.
DEFENSE_REGISTRY = {
    "none": NormTrackingFedAvg,
    "fedavg": NormTrackingFedAvg,
    "norm_clipping": NormClipStrategy,
    "flame": FlameStrategy,
    "foolsgold": FoolsGoldStrategy,
    "fed_median": FedMedianStrategy,
    "multi_krum": MultiKrumStrategy,
}
