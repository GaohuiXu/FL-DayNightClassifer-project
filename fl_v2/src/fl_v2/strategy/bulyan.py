"""NormTracking Bulyan — Byzantine-resilient aggregation with norm logging.

Implements Bulyan (El Mhamdi et al., 2018) as a custom strategy extending
NormTrackingFedAvg, fixing a dtype bug in Flower 1.27.0's built-in version.

Algorithm:
  1. Select theta = n - 2f clients using MultiKrum
  2. Compute coordinate-wise median of selected
  3. For each coordinate, average the beta = theta - 2f values closest to median
Requires n >= 4f + 3 participating clients.

Paper: https://arxiv.org/abs/1802.07927
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.common import NDArrays
from flwr.serverapp.strategy.multikrum import select_multikrum

from fl_v2.attacks_defenses import compute_update_norms
from fl_v2.strategy.norm_tracking_fedavg import NormTrackingFedAvg


def _aggregate_n_closest_weights(
    ref_weights: NDArrays, weights_list: list[NDArrays], beta: int,
) -> NDArrays:
    """Element-wise mean of the ``beta`` closest weight arrays to the reference.

    For each coordinate, selects the ``beta`` values closest to the
    corresponding reference value and averages them.

    Copied from Flower 1.27.0 ``flwr.serverapp.strategy.bulyan``.
    """
    aggregated = []
    for layer_id, ref_layer in enumerate(ref_weights):
        layer_stack = np.stack([w[layer_id] for w in weights_list])
        diffs = np.abs(layer_stack - ref_layer)
        idx = np.argpartition(diffs, beta - 1, axis=0)[:beta]
        closest = np.take_along_axis(layer_stack, idx, axis=0)
        aggregated.append(np.mean(closest, axis=0))
    return aggregated


class NormTrackingBulyan(NormTrackingFedAvg):
    """Bulyan with update-norm logging.

    Two-phase Byzantine-resilient aggregation:
    Phase 1 — MultiKrum selection of theta = n - 2f clients.
    Phase 2 — For each coordinate, average the beta = theta - 2f values
              closest to the coordinate-wise median of the selected clients.
    """

    def __init__(
        self,
        num_malicious_nodes: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_malicious_nodes = num_malicious_nodes

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        # Deterministic ordering — see norm_tracking_fedavg.aggregate_train.
        # Especially important for Bulyan because the inner MultiKrum tie-break
        # is order-sensitive.
        valid_replies.sort(key=lambda msg: msg.metadata.src_node_id)

        n = len(valid_replies)
        f = self.num_malicious_nodes
        required = 4 * f + 3
        if n < required:
            print(
                f"[Bulyan] Insufficient replies: got {n}, need >= {required} "
                f"(4*{f}+3). Skipping aggregation.",
                flush=True,
            )
            self._last_train_metrics = None
            return None, None

        # --- norm logging ---
        if self.current_arrays is not None:
            global_params = self.current_arrays.to_numpy_ndarrays()
            client_params_list = self._extract_client_params(valid_replies)
            original_norms = compute_update_norms(global_params, client_params_list)
            self._print_client_table(valid_replies)
            self._compute_and_log_norms(server_round, original_norms)

        # --- Phase 1: MultiKrum selection ---
        theta = n - 2 * f
        beta = theta - 2 * f

        reply_contents = [msg.content for msg in valid_replies]
        selected_contents = select_multikrum(reply_contents, f, theta)

        # --- Extract numpy arrays from selected clients ---
        key = list(selected_contents[0].array_records.keys())[0]
        array_keys = list(selected_contents[0][key].keys())
        selected_ndarrays = [
            cast(ArrayRecord, ctnt[key]).to_numpy_ndarrays(keep_input=False)
            for ctnt in selected_contents
        ]

        # --- Phase 2: Median then closest-beta averaging ---
        median_ndarrays = [
            np.median(np.stack(arrs), axis=0)
            for arrs in zip(*selected_ndarrays, strict=True)
        ]

        aggregated_ndarrays = _aggregate_n_closest_weights(
            median_ndarrays, selected_ndarrays, beta,
        )

        # --- Construct ArrayRecord with correct dtype ---
        # Use dtype from selected_ndarrays (first client, each layer)
        arrays = ArrayRecord(
            dict(
                zip(
                    array_keys,
                    (
                        Array(np.asarray(arr, dtype=selected_ndarrays[0][i].dtype))
                        for i, arr in enumerate(aggregated_ndarrays)
                    ),
                    strict=True,
                )
            )
        )

        # --- aggregate metrics ---
        metrics = self.train_metrics_aggr_fn(
            selected_contents,
            self.weighted_by_key,
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics
