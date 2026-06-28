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
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
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
from fl_v3.strategy.sampling import (
    SAMPLE_SALT_EVAL,
    SAMPLE_SALT_TRAIN,
    select_partition_ids,
)
from fl_v3.strategy.server_opt import ServerOptimizer

# Discovery (one cheap QUERY round) must out-wait the Ray actor-pool cold start on a
# contended node; the probe itself is trivial. Far above any real per-message latency.
_DISCOVERY_TIMEOUT_S = 1800.0


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
        server_optimizer: "ServerOptimizer | None" = None,
        server_ema_decay: float = 0.0,
        client_lr_schedule: str = "constant",
        client_base_lr: float | None = None,
        num_rounds: int = 1,
        client_lr_warmup_rounds: int = 0,
        client_lr_final_frac: float = 0.0,
        log_gradient_metrics: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.log_gradient_metrics = bool(log_gradient_metrics)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.seed = int(seed)
        self.topk_energy_k = int(topk_energy_k)
        self.current_arrays: ArrayRecord | None = None
        self._norm_history: list[dict] = []
        self._last_train_metrics: dict | None = None
        # DT3-B deterministic sampler state.
        self._pid_to_node: dict[int, int] | None = None
        self._last_train_partition_ids: list[int] | None = None
        self._last_eval_partition_ids: list[int] | None = None
        # MCR P3 (D17): server optimizer (FedAdam) + server EMA + per-round client LR schedule.
        # Defaults = identity FedAvg / no EMA / constant LR ⇒ byte-identical to the pre-MCR strategy.
        self.server_optimizer = server_optimizer or ServerOptimizer()
        self.server_ema_decay = float(server_ema_decay)
        self._ema_arrays: list | None = None  # fp64 server-EMA shadow of the global trainable vector
        self.client_lr_schedule = str(client_lr_schedule)
        self.client_base_lr = (None if client_base_lr is None else float(client_base_lr))
        self.num_rounds = int(num_rounds)
        self.client_lr_warmup_rounds = int(client_lr_warmup_rounds)
        self.client_lr_final_frac = float(client_lr_final_frac)

    def _lr_at_round(self, server_round: int) -> "float | None":
        """Per-round global client LR (warmup → cosine over rounds), or None for the constant-LR default.

        The FL analog of the centralized OneCycle: the server broadcasts one LR per round (clients train
        their E local epochs at it). ``constant`` (default) returns None ⇒ the train_config LR is untouched
        (byte-identical). Otherwise: linear warmup to ``client_base_lr`` over ``warmup_rounds``, then cosine
        decay to ``base*final_frac`` over the remaining rounds (1-indexed round)."""
        import math
        if self.client_lr_schedule == "constant" or self.client_base_lr is None:
            return None
        R = max(1, self.num_rounds)
        w = max(0, self.client_lr_warmup_rounds)
        base = self.client_base_lr
        final = base * self.client_lr_final_frac
        r = int(server_round)
        if w > 0 and r <= w:
            return base * r / w
        t = min(1.0, max(0.0, (r - w) / max(1, (R - w))))
        return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t))

    # ---- DT3-B: deterministic participant sampling over the 0..N-1 pid space ----
    def _ensure_discovery(self, grid: Grid) -> None:
        """One-time node_id↔partition-id discovery (see ``strategy.sampling``).

        Flower assigns random node_ids and exposes no node_id→partition-id map to the
        strategy, so we send one cheap QUERY round; each client echoes its ``partition-id``
        in ``reply-meta``. We require the discovered ids to cover EXACTLY ``range(N)``
        (``N = #nodes``) — i.e. ``num-supernodes`` must equal the derived client count —
        else the deterministic ``sample(range(N))`` would map to a non-existent partition.
        """
        if self._pid_to_node is not None:
            return
        node_ids = list(grid.get_node_ids())
        probe = [
            Message(
                content=RecordDict(
                    {self.configrecord_key: ConfigRecord({"probe": "partition-id"})}
                ),
                message_type=MessageType.QUERY,
                dst_node_id=int(nid),
            )
            for nid in node_ids
        ]
        replies = list(grid.send_and_receive(probe, timeout=_DISCOVERY_TIMEOUT_S))
        pid_to_node: dict[int, int] = {}
        for r in replies:
            if r.has_error():
                raise RuntimeError(
                    f"[DT3-B] discovery: node {r.metadata.src_node_id} returned an error: "
                    f"{r.error.reason if r.error else 'unknown'}"
                )
            pid = int(r.content["reply-meta"]["partition-id"])
            if pid in pid_to_node:
                raise RuntimeError(f"[DT3-B] discovery: duplicate partition-id {pid}")
            pid_to_node[pid] = int(r.metadata.src_node_id)
        n = len(node_ids)
        if sorted(pid_to_node) != list(range(n)):
            raise RuntimeError(
                f"[DT3-B] discovery: partition-ids {sorted(pid_to_node)} do not cover "
                f"range({n}). num-supernodes MUST equal the derived client count N "
                "(stamp it in the federation config)."
            )
        self._pid_to_node = pid_to_node
        print(f"[DT3-B] discovery: mapped {n} partition-ids -> node_ids", flush=True)

    def _deterministic_node_ids(
        self, server_round: int, fraction: float, min_nodes: int, salt: int
    ) -> tuple[list[int], list[int]]:
        """Select participant partition-ids deterministically, map to this run's node_ids."""
        assert self._pid_to_node is not None
        n = len(self._pid_to_node)
        pids = select_partition_ids(
            self.seed, server_round, n, fraction, min_nodes, salt=salt
        )
        node_ids = [self._pid_to_node[p] for p in pids]
        return pids, node_ids

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
        # DT3-B: select participants by the DETERMINISTIC partition-id sampler — NOT
        # Flower's ``random.sample(grid.get_node_ids())`` (non-reproducible at fraction<1
        # over random node_ids). We do NOT call ``super().configure_train`` (that is the
        # non-deterministic path). ``current_arrays`` is the trainable-only global (DT3-A).
        self.current_arrays = arrays.copy()
        if self.fraction_train == 0.0:
            return []
        import time
        self._round_t0 = time.perf_counter()   # MCR P3: round wall-clock start (read in aggregate_train)
        self._ensure_discovery(grid)
        pids, node_ids = self._deterministic_node_ids(
            server_round, self.fraction_train, self.min_train_nodes, SAMPLE_SALT_TRAIN
        )
        self._last_train_partition_ids = list(pids)
        # MCR P3: per-round global client LR (warmup→cosine over rounds). None ⇒ leave the train_config LR
        # untouched (the constant-LR default = byte-identical).
        lr_r = self._lr_at_round(server_round)
        if lr_r is not None:
            config["learning-rate"] = float(lr_r)
        print(
            f"\n{'=' * 60}\n  Round {server_round} — TRAIN  "
            f"(DT3-B participants pids={pids}"
            f"{f', client-lr={lr_r:.2e}' if lr_r is not None else ''})\n{'=' * 60}",
            flush=True,
        )
        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def configure_evaluate(self, server_round, arrays, config, grid: Grid) -> Iterable[Message]:
        # Symmetric deterministic selection for the (optional) client-side evaluate, with
        # its OWN ``min-evaluate-nodes`` and an independent RNG stream (SAMPLE_SALT_EVAL).
        # ``fraction_evaluate == 0.0`` skips client-side eval (the AD task evaluates
        # server-side; clients carry no val loader).
        if self.fraction_evaluate == 0.0:
            return []
        self._ensure_discovery(grid)
        pids, node_ids = self._deterministic_node_ids(
            server_round, self.fraction_evaluate, self.min_evaluate_nodes, SAMPLE_SALT_EVAL
        )
        self._last_eval_partition_ids = list(pids)
        config["server-round"] = server_round
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_train(self, server_round, replies):
        import time
        _t0 = time.perf_counter()
        out = self._aggregate_with_core(server_round, replies)
        # MCR P3 profiling telemetry: round wall-time (from configure_train) + the aggregation step.
        wall = (time.perf_counter() - getattr(self, "_round_t0", time.perf_counter()))
        print(f"[round-prof] round={server_round} round_wall={wall:.1f}s "
              f"aggregate={1e3*(time.perf_counter()-_t0):.0f}ms", flush=True)
        return out

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

        # MCR P3 (D17): the server optimizer step. ``decision.new_global`` is the defense/FedAvg robust
        # aggregate (always the new GLOBAL PARAMS, for every defense), so Δ = aggregate − round-start global
        # is form-agnostic. Identity for the default FedAvg; FedAdam/FedAvgM apply the adaptive step. This is
        # ORTHOGONAL to defense-type (the D10 reference stays defense-type=none) and composes with any defense.
        new_global = self.server_optimizer.step(global_params, decision.new_global)
        # Server-side cross-round EMA shadow (fp64) — the FL analog of the centralized EMA(0.9997); the
        # reference is reported on whichever of {raw global, EMA} peaks (server_app saves both).
        if self.server_ema_decay and self.server_ema_decay > 0.0:
            if self._ema_arrays is None:
                self._ema_arrays = [np.asarray(a, dtype=np.float64).copy() for a in new_global]
            else:
                d = self.server_ema_decay
                self._ema_arrays = [
                    d * e + (1.0 - d) * np.asarray(a, dtype=np.float64)
                    for e, a in zip(self._ema_arrays, new_global)
                ]

        # ``new_global`` is ordered like the GLOBAL arrays (current_arrays), so key the output by the
        # global keys — not the reply's. Assert the reply uses the same key order so a mismatch fails
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
        for key, arr in zip(global_keys, new_global):
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
        # MCR P3: the gradient-space metrics (cos-to-mean + n×n pairwise cosine + top-k energy over the FULL
        # flattened update) are DEFENSE/collusion telemetry — O(n²·d) over 25 clients × 33M params = ~150 s/round
        # for bb02d (measured). They are USELESS for the clean baseline (defense=none), so they are gated off by
        # ``log-gradient-metrics=false`` for the FL reference. Default True ⇒ the defense benchmark (T6) is
        # unchanged. The cheap per-client update NORMS (above) are always logged (the divergent-client signal).
        if (self.log_gradient_metrics
                and global_params is not None and client_params_list is not None):
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
