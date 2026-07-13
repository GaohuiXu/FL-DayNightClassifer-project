"""Flower wrapper for the clean FedAvg foundation.

The strategy preserves the established clean-path contracts:

* clients are discovered and sampled by deterministic partition identity;
* replies are aggregated in partition-id order;
* client parameters are weighted by ``num-examples`` using Flower-identical
  FP32 arithmetic;
* the configured server optimizer and optional cross-round EMA are applied
  after FedAvg.

There is deliberately no registry or alternative aggregation mode here.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fl_v3.strategy.aggregation_core import fp32_weighted_average
from fl_v3.strategy.sampling import (
    SAMPLE_SALT_EVAL,
    SAMPLE_SALT_TRAIN,
    select_partition_ids,
)
from fl_v3.strategy.server_opt import ServerOptimizer

# Discovery must out-wait a cold Ray actor pool on a contended node.
_DISCOVERY_TIMEOUT_S = 1800.0


def partition_sort_key(msg) -> int:
    """Return the stable client identity carried by a reply.

    Flower node ids vary between drivers. Clients therefore echo their stable
    partition id in ``reply-meta``; the node id is only a defensive fallback.
    """
    try:
        return int(msg.content["reply-meta"]["partition-id"])
    except (KeyError, TypeError, ValueError):
        return int(msg.metadata.src_node_id)


def drop_nonfinite_replies(valid_replies: list, arrayrecord_key: str) -> list:
    """Drop replies containing non-finite parameters, preserving kept order."""
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


def _reply_num_examples(msg: Message) -> float:
    """Read Flower's aggregation weight, falling back to uniform weighting."""
    try:
        return float(msg.content["metrics"]["num-examples"])
    except (KeyError, TypeError, ValueError):
        return 1.0


class CleanFedAvgStrategy(FedAvg):
    """Deterministic clean FedAvg with FedOpt and optional server EMA."""

    def __init__(
        self,
        output_dir: str = "./outputs",
        experiment_name: str = "default",
        seed: int = 42,
        server_optimizer: "ServerOptimizer | None" = None,
        server_ema_decay: float = 0.0,
        client_lr_schedule: str = "constant",
        client_base_lr: Optional[float] = None,
        num_rounds: int = 1,
        client_lr_warmup_rounds: int = 0,
        client_lr_final_frac: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.seed = int(seed)
        self.current_arrays: Optional[ArrayRecord] = None
        self._last_train_metrics: Optional[dict] = None
        self._pid_to_node: Optional[Dict[int, int]] = None
        self._last_train_partition_ids: Optional[List[int]] = None
        self._last_eval_partition_ids: Optional[List[int]] = None
        self.server_optimizer = server_optimizer or ServerOptimizer()
        self.server_ema_decay = float(server_ema_decay)
        self._ema_arrays: Optional[List[np.ndarray]] = None
        self.client_lr_schedule = str(client_lr_schedule)
        self.client_base_lr = None if client_base_lr is None else float(client_base_lr)
        self.num_rounds = int(num_rounds)
        self.client_lr_warmup_rounds = int(client_lr_warmup_rounds)
        self.client_lr_final_frac = float(client_lr_final_frac)

    def _lr_at_round(self, server_round: int) -> "float | None":
        """Return the configured warmup/cosine client LR for this round."""
        import math

        if self.client_lr_schedule == "constant" or self.client_base_lr is None:
            return None
        rounds = max(1, self.num_rounds)
        warmup = max(0, self.client_lr_warmup_rounds)
        base = self.client_base_lr
        final = base * self.client_lr_final_frac
        current = int(server_round)
        if warmup > 0 and current <= warmup:
            return base * current / warmup
        progress = min(
            1.0,
            max(0.0, (current - warmup) / max(1, rounds - warmup)),
        )
        return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))

    def _ensure_discovery(self, grid: Grid) -> None:
        """Discover the stable partition-id to current-driver node-id mapping."""
        if self._pid_to_node is not None:
            return
        node_ids = list(grid.get_node_ids())
        probe = [
            Message(
                content=RecordDict(
                    {self.configrecord_key: ConfigRecord({"probe": "partition-id"})}
                ),
                message_type=MessageType.QUERY,
                dst_node_id=int(node_id),
            )
            for node_id in node_ids
        ]
        replies = list(grid.send_and_receive(probe, timeout=_DISCOVERY_TIMEOUT_S))
        pid_to_node: dict[int, int] = {}
        for reply in replies:
            if reply.has_error():
                raise RuntimeError(
                    f"[sampling] discovery node {reply.metadata.src_node_id} returned "
                    f"an error: {reply.error.reason if reply.error else 'unknown'}"
                )
            partition_id = int(reply.content["reply-meta"]["partition-id"])
            if partition_id in pid_to_node:
                raise RuntimeError(f"[sampling] duplicate partition-id {partition_id}")
            pid_to_node[partition_id] = int(reply.metadata.src_node_id)
        count = len(node_ids)
        if sorted(pid_to_node) != list(range(count)):
            raise RuntimeError(
                f"[sampling] partition-ids {sorted(pid_to_node)} do not cover "
                f"range({count}); num-supernodes must equal the derived client count"
            )
        self._pid_to_node = pid_to_node
        print(f"[sampling] mapped {count} partition-ids to node ids", flush=True)

    def _deterministic_node_ids(
        self,
        server_round: int,
        fraction: float,
        min_nodes: int,
        salt: int,
    ) -> tuple[list[int], list[int]]:
        assert self._pid_to_node is not None
        partition_ids = select_partition_ids(
            self.seed,
            server_round,
            len(self._pid_to_node),
            fraction,
            min_nodes,
            salt=salt,
        )
        return partition_ids, [self._pid_to_node[p] for p in partition_ids]

    def configure_train(self, server_round, arrays, config, grid: Grid) -> Iterable[Message]:
        self.current_arrays = arrays.copy()
        if self.fraction_train == 0.0:
            return []
        import time

        self._round_t0 = time.perf_counter()
        self._ensure_discovery(grid)
        partition_ids, node_ids = self._deterministic_node_ids(
            server_round,
            self.fraction_train,
            self.min_train_nodes,
            SAMPLE_SALT_TRAIN,
        )
        self._last_train_partition_ids = list(partition_ids)
        round_lr = self._lr_at_round(server_round)
        if round_lr is not None:
            config["learning-rate"] = float(round_lr)
        config["server-round"] = server_round
        print(
            f"[FedAvg] round={server_round} train partition_ids={partition_ids}"
            f"{f' client_lr={round_lr:.2e}' if round_lr is not None else ''}",
            flush=True,
        )
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def configure_evaluate(self, server_round, arrays, config, grid: Grid) -> Iterable[Message]:
        if self.fraction_evaluate == 0.0:
            return []
        self._ensure_discovery(grid)
        partition_ids, node_ids = self._deterministic_node_ids(
            server_round,
            self.fraction_evaluate,
            self.min_evaluate_nodes,
            SAMPLE_SALT_EVAL,
        )
        self._last_eval_partition_ids = list(partition_ids)
        config["server-round"] = server_round
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_train(self, server_round, replies):
        import time

        started = time.perf_counter()
        result = self._aggregate_clean(replies)
        wall = time.perf_counter() - getattr(self, "_round_t0", time.perf_counter())
        print(
            f"[round-prof] round={server_round} round_wall={wall:.1f}s "
            f"aggregate={1e3 * (time.perf_counter() - started):.0f}ms",
            flush=True,
        )
        return result

    def _aggregate_clean(self, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        valid_replies = drop_nonfinite_replies(valid_replies, self.arrayrecord_key)
        if not valid_replies:
            self._last_train_metrics = None
            return None, None
        valid_replies.sort(key=partition_sort_key)
        if self.current_arrays is None:
            raise RuntimeError(
                "current global arrays unavailable; configure_train must run before aggregation"
            )

        global_params = self.current_arrays.to_numpy_ndarrays()
        client_params = [
            message.content[self.arrayrecord_key].to_numpy_ndarrays()
            for message in valid_replies
        ]
        num_examples = [_reply_num_examples(message) for message in valid_replies]
        averaged = fp32_weighted_average(client_params, num_examples)
        new_global = self.server_optimizer.step(global_params, averaged)

        if self.server_ema_decay > 0.0:
            if self._ema_arrays is None:
                self._ema_arrays = [
                    np.asarray(array, dtype=np.float64).copy() for array in new_global
                ]
            else:
                decay = self.server_ema_decay
                self._ema_arrays = [
                    decay * shadow + (1.0 - decay) * np.asarray(array, dtype=np.float64)
                    for shadow, array in zip(self._ema_arrays, new_global)
                ]

        global_keys = list(self.current_arrays.keys())
        reply_keys = list(valid_replies[0].content[self.arrayrecord_key].keys())
        if reply_keys != global_keys:
            raise RuntimeError(
                "client reply ArrayRecord key order differs from global arrays; "
                "refusing to mislabel aggregated tensors"
            )
        arrays = ArrayRecord()
        for key, array in zip(global_keys, new_global):
            arrays[key] = Array(np.asarray(array))
        metrics = self.train_metrics_aggr_fn(
            [message.content for message in valid_replies], self.weighted_by_key
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics

    @staticmethod
    def _capture_metrics(metrics) -> Optional[dict]:
        if metrics is None:
            return None
        try:
            return {
                str(key): float(value)
                for key, value in dict(metrics).items()
                if isinstance(value, (int, float))
            }
        except Exception:
            return None
