"""NormTracking FedMedian — coordinate-wise median aggregation with norm logging.

Implements FedMedian (Yin et al., 2018) as a custom strategy extending
NormTrackingFedAvg, fixing a dtype bug in Flower 1.27.0's built-in FedMedian.

Paper: https://arxiv.org/abs/1803.01498
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord

from fl_v2.attacks_defenses import compute_update_norms
from fl_v2.strategy.norm_tracking_fedavg import NormTrackingFedAvg


class NormTrackingFedMedian(NormTrackingFedAvg):
    """FedMedian with update-norm logging.

    Aggregates client updates by taking the coordinate-wise median
    instead of the weighted mean used by FedAvg.
    """

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        # Deterministic ordering — see norm_tracking_fedavg.aggregate_train.
        # np.median is value-based so the median itself is order-independent,
        # but per-client iteration in the norm-logging path below is, so we
        # sort up front for consistency with the rest of the strategy family.
        valid_replies.sort(key=lambda msg: msg.metadata.src_node_id)

        # --- norm logging (reuse parent infrastructure) ---
        if self.current_arrays is not None:
            global_params = self.current_arrays.to_numpy_ndarrays()
            client_params_list = self._extract_client_params(valid_replies)
            original_norms = compute_update_norms(global_params, client_params_list)
            self._print_client_table(valid_replies)
            self._compute_and_log_norms(server_round, original_norms)

        # --- coordinate-wise median aggregation ---
        record_key = list(valid_replies[0].content.array_records.keys())[0]
        array_keys = list(valid_replies[0].content[record_key].keys())

        arrays = ArrayRecord()
        for array_key in array_keys:
            layers = [
                cast(ArrayRecord, msg.content[record_key]).pop(array_key).numpy()
                for msg in valid_replies
            ]
            stacked = np.stack(layers)
            median = np.asarray(np.median(stacked, axis=0), dtype=layers[0].dtype)
            arrays[array_key] = Array(median)

        # --- aggregate metrics ---
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics
