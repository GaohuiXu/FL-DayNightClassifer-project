from __future__ import annotations

import csv
import os

import yaml

from fl_v2.utils.io import ensure_dir, save_json


class ExperimentLogger:
    """Persist per-round metrics and experiment config to a structured output directory.

    Directory layout produced:
        {base_output_dir}/{experiment_name}_r{num_rounds}_seed{seed}/
            config.yaml      — full merged run_config
            rounds.csv       — one row per round (incremental, crash-safe)
            summary.json     — best + final metrics (written on finalize)
            checkpoints/     — directory stub for future checkpoint saving
    """

    def __init__(self, run_config, base_output_dir: str) -> None:
        exp_name   = str(run_config.get("experiment-name", "exp"))
        num_rounds = int(run_config["num-server-rounds"])
        seed       = int(run_config["seed"])

        self.exp_dir = os.path.join(
            base_output_dir, f"{exp_name}_r{num_rounds}_seed{seed}"
        )
        ensure_dir(self.exp_dir)
        ensure_dir(os.path.join(self.exp_dir, "checkpoints"))

        self._save_config_yaml(run_config)

        self._rows: list[dict] = []
        self._csv_path = os.path.join(self.exp_dir, "rounds.csv")
        self._csv_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_round(self, server_round: int, metrics: dict) -> None:
        """Append one row to rounds.csv. Safe to call every round."""
        row = {"round": server_round, **metrics}
        self._rows.append(row)

        write_header = not self._csv_initialized
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(row)

    def finalize(self) -> None:
        """Write summary.json. Call once after all rounds complete."""
        if not self._rows:
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
            summary["best_asr"] = {
                "round": best_asr["round"],
                "asr": best_asr["asr"],
            }

        save_json(summary, os.path.join(self.exp_dir, "summary.json"))
        print(f"[logger] summary saved → {self.exp_dir}/summary.json", flush=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_config_yaml(self, run_config) -> None:
        config_path = os.path.join(self.exp_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(dict(run_config), f, default_flow_style=False, sort_keys=True)
