"""NormTracking FedTrimmedAvg — trimmed mean aggregation with norm logging.

Implements FedTrimmedAvg (Yin et al., 2018) as a custom strategy extending
NormTrackingFedAvg, fixing a dtype bug in Flower 1.27.0's built-in version.

Paper: https://arxiv.org/abs/1803.01498
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.common import NDArray

from fl_v2.attacks_defenses import compute_update_norms
from fl_v2.strategy.norm_tracking_fedavg import NormTrackingFedAvg


def _trim_mean(array: NDArray, cut_fraction: float) -> NDArray:
    """Compute trimmed mean along axis=0.

    Based on scipy.stats.trim_mean and Flower's implementation.
    Trims ``cut_fraction`` from both tails before averaging.
    """
    nobs = array.shape[0]
    lowercut = int(cut_fraction * nobs)
    uppercut = nobs - lowercut
    if lowercut >= uppercut:
        raise ValueError(
            f"cut_fraction={cut_fraction} is too large for {nobs} observations."
        )

    atmp = np.partition(array, (lowercut, uppercut - 1), axis=0)
    sl = [slice(None)] * atmp.ndim
    sl[0] = slice(lowercut, uppercut)
    return np.mean(atmp[tuple(sl)], axis=0)


class NormTrackingFedTrimmedAvg(NormTrackingFedAvg):
    """FedTrimmedAvg with update-norm logging.

    Aggregates client updates by computing the trimmed mean: sorts values
    along the client axis, discards the top and bottom ``beta`` fraction,
    then averages the remaining values.
    """

    def __init__(
        self,
        beta: float = 0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beta = beta

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        # --- norm logging ---
        if self.current_arrays is not None:
            global_params = self.current_arrays.to_numpy_ndarrays()
            client_params_list = self._extract_client_params(valid_replies)
            original_norms = compute_update_norms(global_params, client_params_list)
            self._print_client_table(valid_replies)
            self._compute_and_log_norms(server_round, original_norms)

        # --- trimmed mean aggregation ---
        record_key = list(valid_replies[0].content.array_records.keys())[0]
        array_keys = list(valid_replies[0].content[record_key].keys())

        arrays = ArrayRecord()
        for array_key in array_keys:
            layers = [
                cast(ArrayRecord, msg.content[record_key]).pop(array_key).numpy()
                for msg in valid_replies
            ]
            stacked = np.stack(layers)
            trimmed = np.asarray(_trim_mean(stacked, self.beta), dtype=layers[0].dtype)
            arrays[array_key] = Array(trimmed)

        # --- aggregate metrics ---
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics
