"""Thin wrappers around Flower's built-in Krum / MultiKrum.

The bare Flower classes work correctly for Byzantine-tolerant selection
(unlike Bulyan/FedMedian/FedTrimmedAvg which had dtype bugs at 1.27.0
that we worked around with custom NormTracking* implementations). All
they're missing for our purposes is a ``_last_train_metrics`` slot so
:class:`fl_v2.utils.experiment_logger.ExperimentLogger` and the wandb
sink can read the aggregated client MetricRecord each round.

``CapturedKrum`` is the bare client-metric capture wrapper.
``NormTrackingMultiKrum`` (Cycle-03 WS-A) additionally logs gradient-
space metrics (L2 norm, cos2mean, pairwise_cos, topk_energy_frac) on
the same per-round JSON schema as the other NormTracking defenses.
Composition (not multiple inheritance) is used because MultiKrum and
NormTrackingFedAvg both extend FedAvg — direct double inheritance hits
an MRO conflict.
"""
from __future__ import annotations

from flwr.app import ArrayRecord
from flwr.serverapp.strategy import Krum, MultiKrum

from fl_v2.attacks_defenses import compute_update_norms
from fl_v2.strategy.norm_tracking_fedavg import (
    NormTrackingFedAvg,
    drop_nonfinite_replies,
    partition_sort_key,
)


class CapturedKrum(Krum):
    """Krum + per-round aggregated-client-metrics capture."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._last_train_metrics: dict | None = None

    def aggregate_train(self, server_round, replies):
        # NaN/Inf robustness: drop replies with non-finite parameters
        # before Krum's distance computation (NaN distances would
        # silently dominate the selection).
        replies = drop_nonfinite_replies(list(replies), self.arrayrecord_key)
        if not replies:
            self._last_train_metrics = None
            return None, None
        # Sort by partition-id (cross-run-stable) so Krum's argsort tie-break
        # + the aggregate's in-place sum are deterministic across runs.
        # See norm_tracking_fedavg.partition_sort_key for the rationale —
        # Flower 1.27's `metadata.src_node_id` is per-driver random
        # (`os.urandom`), so sorting by it only fixes within-run order.
        replies = sorted(replies, key=partition_sort_key)
        result = super().aggregate_train(server_round, replies)
        self._last_train_metrics = _extract_metrics_from_result(result)
        return result


class NormTrackingMultiKrum(MultiKrum):
    """MultiKrum + per-round gradient-space metric logging (Cycle-03 WS-A).

    Selection logic is delegated to ``flwr.serverapp.strategy.MultiKrum``.
    Norm-history state (the in-memory list + the on-disk JSON path) is
    owned by a composed :class:`NormTrackingFedAvg` instance — whose
    own ``aggregate_train``/``configure_train`` hooks are never invoked;
    we only call :meth:`NormTrackingFedAvg._compute_and_log_norms`. This
    avoids the MRO conflict that would arise from inheriting both
    ``MultiKrum`` and ``NormTrackingFedAvg`` (each is a separate
    ``FedAvg`` subclass).

    Logged metrics describe the RAW per-client signature across all
    admitted replies — computed BEFORE MultiKrum's k-of-N selection, so
    the column matches the per-round schema used by FedAvg / FoolsGold /
    FLAME / NormClipped / FedMedian / FedTrimmedAvg / Bulyan.
    """

    def __init__(
        self,
        *args,
        output_dir: str = "./outputs",
        experiment_name: str = "default",
        seed: int = 42,
        topk_energy_k: int = 4096,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Composed norm-logger: holds history + JSON-path attrs. Its
        # FedAvg-base aggregate_train is never called.
        self._norm_logger = NormTrackingFedAvg(
            output_dir=output_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
        )
        self.current_arrays: ArrayRecord | None = None
        self._last_train_metrics: dict | None = None

    @property
    def _norm_history(self) -> list:
        # server_app.evaluate_fn reads ``strategy._norm_history`` to
        # build per-round malicious-vs-honest wandb summaries; delegate.
        return self._norm_logger._norm_history

    def configure_train(self, server_round, arrays, config, grid):
        # Cache global arrays so aggregate_train can compute per-client
        # deltas (matches NormTrackingFedAvg.configure_train).
        self.current_arrays = arrays.copy()

        print(f"\n{'═' * 60}", flush=True)
        print(f"  Round {server_round} — TRAIN", flush=True)
        print(f"{'═' * 60}", flush=True)

        config["server-round"] = server_round
        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(self, server_round, arrays, config, grid):
        config["server-round"] = server_round
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        replies = drop_nonfinite_replies(list(replies), self.arrayrecord_key)
        if not replies:
            self._last_train_metrics = None
            return None, None
        replies = sorted(replies, key=partition_sort_key)

        if self.current_arrays is None:
            raise RuntimeError(
                "Current global arrays are not available. "
                "Make sure configure_train is called before aggregate_train."
            )

        # Log gradient-space metrics on the full admitted set BEFORE
        # MultiKrum's selection step (parity with other defenses).
        global_params = self.current_arrays.to_numpy_ndarrays()
        client_params_list = [
            msg.content[self.arrayrecord_key].to_numpy_ndarrays()
            for msg in replies
        ]
        original_norms = compute_update_norms(global_params, client_params_list)
        self._norm_logger._compute_and_log_norms(
            server_round,
            original_norms,
            global_params=global_params,
            client_params_list=client_params_list,
            partition_ids=[partition_sort_key(m) for m in replies],
        )

        result = super().aggregate_train(server_round, replies)
        self._last_train_metrics = _extract_metrics_from_result(result)
        return result


def _extract_metrics_from_result(result) -> dict | None:
    try:
        _, metrics = result
    except Exception:
        return None
    return NormTrackingFedAvg._capture_metrics(metrics)
