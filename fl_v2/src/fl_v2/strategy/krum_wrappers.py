"""Thin wrappers around Flower's built-in Krum / MultiKrum.

The bare Flower classes work correctly for Byzantine-tolerant selection
(unlike Bulyan/FedMedian/FedTrimmedAvg which had dtype bugs at 1.27.0
that we worked around with custom NormTracking* implementations). All
they're missing for our purposes is a ``_last_train_metrics`` slot so
:class:`fl_v2.utils.experiment_logger.ExperimentLogger` and the wandb
sink can read the aggregated client MetricRecord each round.

These wrappers add nothing else — no norm logging, no extra config —
and are drop-in for the bare Flower classes everywhere.
"""
from __future__ import annotations

from flwr.serverapp.strategy import Krum, MultiKrum

from fl_v2.strategy.norm_tracking_fedavg import NormTrackingFedAvg


class CapturedKrum(Krum):
    """Krum + per-round aggregated-client-metrics capture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_train_metrics: dict | None = None

    def aggregate_train(self, server_round, replies):
        result = super().aggregate_train(server_round, replies)
        self._last_train_metrics = _extract_metrics_from_result(result)
        return result


class CapturedMultiKrum(MultiKrum):
    """MultiKrum + per-round aggregated-client-metrics capture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_train_metrics: dict | None = None

    def aggregate_train(self, server_round, replies):
        result = super().aggregate_train(server_round, replies)
        self._last_train_metrics = _extract_metrics_from_result(result)
        return result


def _extract_metrics_from_result(result) -> dict | None:
    try:
        _, metrics = result
    except Exception:
        return None
    return NormTrackingFedAvg._capture_metrics(metrics)
