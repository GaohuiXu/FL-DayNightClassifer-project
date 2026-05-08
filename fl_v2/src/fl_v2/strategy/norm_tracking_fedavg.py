from __future__ import annotations

import json
import os
from typing import Iterable, List

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fl_v2.attacks_defenses import compute_update_norms

_SEP = "─" * 60


def partition_sort_key(msg) -> int:
    """Cross-run-stable sort key for the strategy aggregation order.

    Flower 1.27 generates `metadata.src_node_id` per driver via
    ``os.urandom`` (flwr.server.superlink.linkstate.utils.generate_rand_int_from_bytes),
    so the *same* logical partition gets a *different* `src_node_id`
    across two fresh simulation drivers. Sorting `valid_replies` by
    `src_node_id` therefore only fixes within-run order; the
    floating-point summation in
    `flwr.serverapp.strategy.strategy_utils.aggregate_arrayrecords`
    still sees a different order across runs and the result is
    non-associative — which is the residual ε that survived the
    audit's seven fixes.

    Clients now embed the deterministic 0..num_clients-1 partition-id
    in `content["reply-meta"]["partition-id"]`
    (see fl_v2/src/fl_v2/client_app.py). Sorting by that gives
    bit-reproducible aggregation across runs.

    Defensive fallback to `src_node_id` keeps the strategy runnable
    if a client somehow omits the field (e.g., a future client-app
    variant that doesn't include the reply-meta record); within a
    single run this still gives stable order.
    """
    try:
        return int(msg.content["reply-meta"]["partition-id"])
    except (KeyError, TypeError, ValueError):
        return int(msg.metadata.src_node_id)


class NormTrackingFedAvg(FedAvg):
    """FedAvg that computes and logs client update L2 norms every round.

    When ``defense-type`` is ``"none"``, this strategy replaces bare FedAvg
    so that update norms are always visible — enabling informed clip-norm
    selection for later experiments with norm clipping.
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        experiment_name: str = "default",
        seed: int = 42,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.seed = seed
        self.current_arrays: ArrayRecord | None = None
        self._norm_history: list[dict] = []
        # Populated at the end of every aggregate_train; consumed by
        # server_app.evaluate_fn to forward client-reported metrics into
        # the experiment logger and wandb. None when no valid replies.
        self._last_train_metrics: dict | None = None

    @staticmethod
    def _capture_metrics(metrics) -> dict | None:
        """Best-effort conversion of an aggregated MetricRecord to a plain dict.

        Skips non-numeric entries. Returns None on any failure or for None
        input — callers must treat None as "no client metrics this round".
        """
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
            self.output_dir,
            f"{self.experiment_name}_seed{self.seed}_norm_log.json",
        )

    # ------------------------------------------------------------------
    # Flower strategy hooks
    # ------------------------------------------------------------------

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Cache global arrays, inject round number, print round banner."""
        self.current_arrays = arrays.copy()

        # Print round start banner (sequential — always ordered)
        print(f"\n{'═' * 60}", flush=True)
        print(f"  Round {server_round} — TRAIN", flush=True)
        print(f"{'═' * 60}", flush=True)

        # Inject server-round into train config so clients can log it
        config["server-round"] = server_round

        return super().configure_train(server_round, arrays, config, grid)

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Inject round number into evaluate config."""
        config["server-round"] = server_round
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ):
        """Compute and log norms, print per-client table, then aggregate."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        # Deterministic ordering: floating-point summation is non-associative,
        # so iteration order over `valid_replies` leaks into the aggregated
        # weights via the in-place sum at flwr.serverapp.strategy.strategy_utils
        # (~line 101). Without this sort, Ray's task-completion order changes
        # the bit-level result.
        # Subclasses (NormClippedFedAvg, Bulyan, Krum-wrappers, FedMedian,
        # FedTrimmedAvg) all flow through this method, so sorting here covers
        # the whole strategy hierarchy.
        # Sort key MUST be cross-run-stable; see `partition_sort_key`
        # at the top of this module — `metadata.src_node_id` is per-driver
        # random in Flower 1.27 and was the residual-ε source.
        valid_replies.sort(key=partition_sort_key)

        if self.current_arrays is None:
            raise RuntimeError(
                "Current global arrays are not available. "
                "Make sure configure_train is called before aggregate_train."
            )

        global_params = self.current_arrays.to_numpy_ndarrays()
        client_params_list = self._extract_client_params(valid_replies)

        original_norms = compute_update_norms(global_params, client_params_list)

        # Print per-client metrics table (sequential — always ordered)
        self._print_client_table(valid_replies)

        # Log norms to stdout + JSON
        self._compute_and_log_norms(server_round, original_norms)

        arrays, metrics = super().aggregate_train(server_round, valid_replies)
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics

    # ------------------------------------------------------------------
    # Helpers (reused by NormClippedFedAvg)
    # ------------------------------------------------------------------

    def _extract_client_params(
        self, valid_replies: list[Message],
    ) -> list[list[np.ndarray]]:
        """Extract numpy parameter lists from valid reply messages."""
        client_params_list = []
        for msg in valid_replies:
            arrays = msg.content[self.arrayrecord_key]
            client_params_list.append(arrays.to_numpy_ndarrays())
        return client_params_list

    def _print_client_table(self, valid_replies: list[Message]) -> None:
        """Print a compact per-client metrics table from reply messages."""
        rows = []
        for msg in valid_replies:
            m = msg.content["metrics"]
            node_id = msg.metadata.src_node_id
            rows.append((
                node_id,
                int(m["num-examples"]),
                float(m["train_loss"]),
                float(m["train_accuracy"]),
                float(m["val_loss"]),
                float(m["val_accuracy"]),
            ))
        rows.sort(key=lambda r: r[0])

        print(f"\n── Aggregation {_SEP[14:]}", flush=True)
        print(
            f"{'node_id':>8}  {'n_train':>7}  {'t_loss':>8}  "
            f"{'t_acc':>7}  {'v_loss':>7}  {'v_acc':>7}",
            flush=True,
        )
        for node_id, n, tl, ta, vl, va in rows:
            print(
                f"{node_id:>8}  {n:>7}  {tl:>8.4f}  "
                f"{ta:>6.1%}  {vl:>7.4f}  {va:>6.1%}",
                flush=True,
            )

    def _compute_and_log_norms(
        self,
        server_round: int,
        original_norms: List[float],
        clipped_norms: List[float] | None = None,
    ) -> None:
        """Log norms to stdout and persist to JSON."""
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

        self._norm_history.append(round_record)

        print(
            f"\n[NormLog] round={server_round} "
            f"mean={norm_stats['mean']:.2f} "
            f"max={norm_stats['max']:.2f} "
            f"min={norm_stats['min']:.2f} "
            f"std={norm_stats['std']:.2f}",
            flush=True,
        )
        if clipped_norms is not None:
            print(
                f"[NormLog] clipped  "
                f"mean={float(np.mean(clipped_norms)):.2f} "
                f"max={float(np.max(clipped_norms)):.2f} "
                f"min={float(np.min(clipped_norms)):.2f}",
                flush=True,
            )

        # Persist to JSON (overwrite each round for crash safety)
        log_path = self._norm_log_path()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self._norm_history, f, indent=2)
