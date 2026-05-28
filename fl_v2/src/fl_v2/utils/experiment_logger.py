from __future__ import annotations

import csv
import os

import torch
import yaml

from fl_v2.utils.io import ensure_dir, save_json
from fl_v2.utils.wandb_logger import WandbLogger


class ExperimentLogger:
    """Persist per-round metrics and experiment config to a structured output directory.

    Directory layout produced:

    Cycle-aware (new YAMLs that set ``cycle`` and ``phase``):
        {base_output_dir}/experiments/{cycle}/{phase}/{experiment_name}_r{num_rounds}_seed{seed}/
            config.yaml
            rounds.csv
            summary.json
            checkpoints/
            wandb/                  — created by WandbLogger
            wandb_offline/          — populated by run_alvis.sh on exit if WANDB_MODE=offline

    Legacy fallback (closed-cycle YAMLs without ``cycle``/``phase``):
        {base_output_dir}/{experiment_name}_r{num_rounds}_seed{seed}/
        (preserves the historical Cycle 01 layout under fl_outputs/gtsrb_v2/<phase>/.)

    All per-round metrics flow through :meth:`log_round`, which also forwards
    them to :class:`WandbLogger` when wandb is enabled. The CSV layer remains
    the source of truth (crash-safe append-only); wandb is additive.
    """

    def __init__(self, run_config, base_output_dir: str) -> None:
        exp_name   = str(run_config.get("experiment-name", "exp"))
        num_rounds = int(run_config["num-server-rounds"])
        seed       = int(run_config["seed"])

        cycle = str(run_config.get("cycle", "")).strip()
        phase = str(run_config.get("phase", "")).strip()

        parts = [base_output_dir]
        if cycle and phase:
            parts += ["experiments", cycle, phase]
        elif cycle:
            parts += ["experiments", cycle]
        parts.append(f"{exp_name}_r{num_rounds}_seed{seed}")

        self.exp_dir = os.path.join(*parts)
        ensure_dir(self.exp_dir)
        ensure_dir(os.path.join(self.exp_dir, "checkpoints"))

        self._save_config_yaml(run_config)

        self._rows: list[dict] = []
        self._csv_path = os.path.join(self.exp_dir, "rounds.csv")
        # Overwrite any existing CSV from a previous run
        if os.path.exists(self._csv_path):
            os.remove(self._csv_path)
        self._csv_initialized = False

        # Wandb is constructed last so its WANDB_DIR sits inside exp_dir.
        # Its constructor is a no-op when wandb-enabled=false.
        self.wandb = WandbLogger(run_config, self.exp_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_round(
        self,
        server_round: int,
        metrics: dict,
        client_metrics: dict | None = None,
        update_metrics: dict | None = None,
    ) -> None:
        """Append one row to rounds.csv and forward to wandb if enabled.

        ``metrics`` is the server-side eval dict (test_loss, test_accuracy,
        target_class_clean_accuracy, asr, etc.). ``client_metrics`` is the
        weighted-average client-reported dict (train_loss, train_accuracy,
        val_loss, val_accuracy, num-examples) — captured by strategies via
        their ``_last_train_metrics`` slot.

        The CSV row contains server metrics plus any client metrics
        (prefixed ``client_*``). Wandb gets them as ``server/<key>`` and
        ``client/<key>`` namespaces.
        """
        row = {"round": server_round, **metrics}
        if client_metrics:
            for k, v in client_metrics.items():
                if isinstance(v, (int, float)):
                    row[f"client_{k}"] = v
        self._rows.append(row)

        write_header = not self._csv_initialized
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

        # Forward to wandb (no-op when disabled). update_metrics (the
        # Cycle-02 gradient-space malicious-vs-honest summary) goes to
        # wandb only — the full per-client data lives in norm_log.json.
        self.wandb.log_round(server_round, metrics, client_metrics, update_metrics)

    def save_checkpoint(self, server_round: int, state_dict: dict) -> None:
        """Save model checkpoint for a given round."""
        path = os.path.join(
            self.exp_dir, "checkpoints", f"round_{server_round:04d}.pt"
        )
        torch.save(state_dict, path)
        print(f"[logger] checkpoint saved → {path}", flush=True)

    def log_image(self, name: str, path: str) -> None:
        """Forward a one-shot image artifact to wandb (no-op when disabled)."""
        self.wandb.log_image(name, path)

    def finalize(self) -> None:
        """Write summary.json, push it to wandb, and close the wandb run."""
        if not self._rows:
            self.wandb.finalize()
            return

        # Exclude round 0 (untrained model) from best-metric selection
        trained = [r for r in self._rows if r["round"] > 0]
        rows_for_best = trained if trained else self._rows
        final = rows_for_best[-1]

        best_acc = max(rows_for_best, key=lambda r: r.get("test_accuracy", 0.0))

        summary: dict = {
            "final": final,
            "best_test_accuracy": {
                "round": best_acc["round"],
                "test_accuracy": best_acc["test_accuracy"],
            },
        }

        if "asr" in final:
            best_asr = max(rows_for_best, key=lambda r: r.get("asr", 0.0))
            best_asr_entry: dict = {
                "round": best_asr["round"],
                "asr": best_asr["asr"],
            }
            # Cycle-03 WS-A: surface clean-floor + backdoor-attributable
            # ASR next to the headline ASR so summary.json carries the
            # backdoor-attribution story end-to-end. Keys are absent on
            # Wave-1 cells re-read under this schema; consumers should
            # default missing keys to 0/NaN.
            if "clean_floor_to_target" in best_asr:
                best_asr_entry["clean_floor_to_target"] = best_asr["clean_floor_to_target"]
            if "backdoor_attribute_asr" in best_asr:
                best_asr_entry["backdoor_attribute_asr"] = best_asr[
                    "backdoor_attribute_asr"
                ]
            summary["best_asr"] = best_asr_entry

            if "backdoor_attribute_asr" in final:
                best_baa = max(
                    rows_for_best,
                    key=lambda r: r.get("backdoor_attribute_asr", 0.0),
                )
                summary["best_backdoor_attribute_asr"] = {
                    "round": best_baa["round"],
                    "backdoor_attribute_asr": best_baa["backdoor_attribute_asr"],
                    "asr": best_baa.get("asr", 0.0),
                    "clean_floor_to_target": best_baa.get(
                        "clean_floor_to_target", 0.0
                    ),
                }

        save_json(summary, os.path.join(self.exp_dir, "summary.json"))
        print(f"[logger] summary saved → {self.exp_dir}/summary.json", flush=True)

        self.wandb.log_summary(summary)
        self.wandb.finalize()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_config_yaml(self, run_config) -> None:
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(run_config), f, default_flow_style=False, sort_keys=True)
