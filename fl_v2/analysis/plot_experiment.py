#!/usr/bin/env python3
"""Post-hoc analysis of FL experiment results.

Auto-discovers experiments from the output directory and produces
standard training curves, norm evolution plots, and text summaries.

Usage:
    # Analyze all discovered experiments (default base dir)
    python -m analysis.plot_experiment

    # Analyze a specific experiment directory
    python -m analysis.plot_experiment --exp-dir /path/to/experiment

    # Override base directory (for future datasets)
    python -m analysis.plot_experiment --base-dir /path/to/outputs

    # Also produce norm comparison across experiments
    python -m analysis.plot_experiment --compare-norms
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ── defaults ──────────────────────────────────────────────────────────
DEFAULT_BASE_DIR = Path(
    "/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

PLOT_STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
}


# ── discovery ─────────────────────────────────────────────────────────
def discover_experiments(base_dir: Path) -> list[Path]:
    """Find experiment directories that contain rounds.csv."""
    return sorted(
        p.parent for p in base_dir.rglob("rounds.csv")
    )


def discover_norm_logs(base_dir: Path) -> list[Path]:
    """Find all *_norm_log.json files under base_dir."""
    return sorted(base_dir.glob("*_norm_log.json"))


def derive_label(path: Path) -> str:
    """Derive a human-readable label from a path.

    For experiment dirs:  pixel-trigger-backdoor-resnet18-no-defense_r100_seed42
                       -> Pixel Trigger Backdoor ResNet18 No Defense
    For norm log files:   clean-baseline_seed42_norm_log.json
                       -> Clean Baseline
    """
    # Use stem for files (strip extension), name for directories
    name = path.stem if path.suffix else path.name
    # strip common suffixes
    name = re.sub(r"_norm_log$", "", name)
    name = re.sub(r"_client_label_histograms$", "", name)
    name = re.sub(r"_r\d+_seed\d+$", "", name)
    name = re.sub(r"_seed\d+$", "", name)
    # convert hyphens/underscores to spaces, title-case
    label = name.replace("-", " ").replace("_", " ").strip()
    label = label.title()
    # fix common casing
    label = label.replace("Resnet18", "ResNet18")
    label = label.replace("Resnet", "ResNet")
    label = label.replace("Cnn", "CNN")
    label = label.replace("Asr", "ASR")
    label = label.replace("Fedavg", "FedAvg")
    label = label.replace("Fedmedian", "FedMedian")
    return label


# ── data loaders ──────────────────────────────────────────────────────
def load_rounds_csv(path: Path) -> dict[str, np.ndarray]:
    """Load rounds.csv into a dict of numpy arrays keyed by column name."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    columns = {}
    for key in rows[0]:
        try:
            columns[key] = np.array([float(r[key]) for r in rows])
        except (ValueError, TypeError):
            columns[key] = np.array([r[key] for r in rows])
    return columns


def load_summary_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_config_yaml(path: Path) -> dict:
    # Use simple parsing to avoid hard yaml dependency
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip().strip("'\"")
                # try numeric conversion
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                config[key.strip()] = val
    return config


def load_norm_log(path: Path) -> dict[str, np.ndarray]:
    """Load norm_log.json into arrays: rounds, means, maxs, mins, stds."""
    with open(path) as f:
        data = json.load(f)
    rounds = np.array([entry["round"] for entry in data])
    means = np.array([entry["stats"]["mean"] for entry in data])
    maxs = np.array([entry["stats"]["max"] for entry in data])
    mins = np.array([entry["stats"]["min"] for entry in data])
    stds = np.array([entry["stats"]["std"] for entry in data])
    return {"rounds": rounds, "means": means, "maxs": maxs,
            "mins": mins, "stds": stds}


# ── plotting ──────────────────────────────────────────────────────────
def plot_training_curves(
    data: dict[str, np.ndarray],
    summary: dict | None,
    label: str,
    output_dir: Path,
):
    """3-panel figure: test accuracy, ASR, TCA vs round."""
    rounds = data["round"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # -- test accuracy --
    ax = axes[0]
    ax.plot(rounds, data["test_accuracy"] * 100, color="tab:blue")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{label} — Training Curves")
    if summary and "best_test_accuracy" in summary:
        best = summary["best_test_accuracy"]
        ax.axhline(best["test_accuracy"] * 100, ls="--", color="tab:blue",
                    alpha=0.4, label=f"Best: {best['test_accuracy']*100:.1f}% (r{best['round']})")
        ax.legend(loc="lower right")

    # -- ASR --
    ax = axes[1]
    has_asr = "asr" in data and not np.all(np.isnan(data["asr"]))
    if has_asr:
        ax.plot(rounds, data["asr"] * 100, color="tab:red")
        ax.set_ylabel("ASR (%)")
        if summary and "best_asr" in summary:
            best = summary["best_asr"]
            ax.axhline(best["asr"] * 100, ls="--", color="tab:red",
                        alpha=0.4, label=f"Peak: {best['asr']*100:.1f}% (r{best['round']})")
            ax.legend(loc="lower right")
    else:
        ax.text(0.5, 0.5, "No ASR data (clean baseline)",
                transform=ax.transAxes, ha="center", va="center", color="gray")
        ax.set_ylabel("ASR (%)")

    # -- TCA --
    ax = axes[2]
    if "target_class_clean_accuracy" in data:
        ax.plot(rounds, data["target_class_clean_accuracy"] * 100, color="tab:green")
    ax.set_ylabel("Target Class Accuracy (%)")
    ax.set_xlabel("Round")

    for ax in axes:
        ax.set_ylim(-2, 102)

    fig.tight_layout()
    safe_name = re.sub(r"[^\w\-]", "_", label.lower())
    out_path = output_dir / f"{safe_name}_training_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_norm_evolution(
    norm_data: dict[str, np.ndarray],
    label: str,
    output_dir: Path,
):
    """Mean norm with std shaded band per round."""
    rounds = norm_data["rounds"]
    means = norm_data["means"]
    stds = norm_data["stds"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, means, color="tab:purple", label="Mean norm")
    ax.fill_between(rounds, means - stds, means + stds,
                    alpha=0.2, color="tab:purple", label="±1 std")
    ax.plot(rounds, norm_data["maxs"], ls="--", color="tab:orange",
            alpha=0.6, label="Max norm")
    ax.set_xlabel("Round")
    ax.set_ylabel("L2 Norm")
    ax.set_title(f"{label} — Update Norm Evolution")
    ax.legend()
    fig.tight_layout()

    safe_name = re.sub(r"[^\w\-]", "_", label.lower())
    out_path = output_dir / f"{safe_name}_norm_evolution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


def plot_norm_comparison(
    norm_list: list[dict[str, np.ndarray]],
    labels: list[str],
    output_dir: Path,
):
    """Overlay mean norms from multiple experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for norm_data, label in zip(norm_list, labels):
        ax.plot(norm_data["rounds"], norm_data["means"], label=label)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean L2 Norm")
    ax.set_title("Update Norm Comparison Across Experiments")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "norm_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_path}")


# ── text summary ──────────────────────────────────────────────────────
def print_summary(label: str, data: dict, summary: dict | None, config: dict | None):
    """Print key findings to stdout."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    if config:
        print(f"  Model:    {config.get('model-type', '?')}")
        print(f"  Attack:   {config.get('attack-type', 'none')}")
        print(f"  Defense:  {config.get('defense-type', 'none')}")
        print(f"  Rounds:   {config.get('num-server-rounds', '?')}")
        print(f"  Clients:  {config.get('num-clients', '?')} "
              f"(malicious: {config.get('malicious-client-ids', 'none')})")

    n = len(data.get("round", []))
    if n > 0:
        final_acc = data["test_accuracy"][-1] * 100
        final_loss = data["test_loss"][-1]
        print(f"\n  Final test accuracy:  {final_acc:.2f}%")
        print(f"  Final test loss:      {final_loss:.4f}")

        if "target_class_clean_accuracy" in data:
            final_tca = data["target_class_clean_accuracy"][-1] * 100
            print(f"  Final TCA:            {final_tca:.2f}%")

        if "asr" in data and not np.all(data["asr"] == 0):
            final_asr = data["asr"][-1] * 100
            print(f"  Final ASR:            {final_asr:.2f}%")

    if summary:
        if "best_test_accuracy" in summary:
            b = summary["best_test_accuracy"]
            print(f"\n  Best accuracy: {b['test_accuracy']*100:.2f}% at round {b['round']}")
        if "best_asr" in summary:
            b = summary["best_asr"]
            print(f"  Best ASR:      {b['asr']*100:.2f}% at round {b['round']}")

    print()


# ── main ──────────────────────────────────────────────────────────────
def main():
    plt.rcParams.update(PLOT_STYLE)

    parser = argparse.ArgumentParser(
        description="Analyze FL experiment results."
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help="Base output directory to scan for experiments.",
    )
    parser.add_argument(
        "--exp-dir", type=Path, default=None,
        help="Analyze a specific experiment directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--compare-norms", action="store_true",
        help="Also produce a norm comparison plot across experiments.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which experiments to analyze
    if args.exp_dir:
        exp_dirs = [args.exp_dir]
    else:
        exp_dirs = discover_experiments(args.base_dir)
        if not exp_dirs:
            print(f"No experiments with rounds.csv found under {args.base_dir}")

    # Analyze each experiment
    for exp_dir in exp_dirs:
        label = derive_label(exp_dir)
        print(f"\nAnalyzing: {exp_dir.name}")

        data = load_rounds_csv(exp_dir / "rounds.csv")
        if not data:
            print("  [skip] empty rounds.csv")
            continue

        summary = None
        if (exp_dir / "summary.json").exists():
            summary = load_summary_json(exp_dir / "summary.json")

        config = None
        if (exp_dir / "config.yaml").exists():
            config = load_config_yaml(exp_dir / "config.yaml")

        print_summary(label, data, summary, config)
        plot_training_curves(data, summary, label, args.output_dir)

        # Try to find matching norm log
        exp_name = config.get("experiment-name", "") if config else ""
        seed = config.get("seed", 42) if config else 42
        norm_log_path = args.base_dir / f"{exp_name}_seed{seed}_norm_log.json"
        if not norm_log_path.exists():
            # fallback: look inside exp dir
            candidates = list(exp_dir.glob("*_norm_log.json"))
            if candidates:
                norm_log_path = candidates[0]
        if norm_log_path.exists():
            norm_data = load_norm_log(norm_log_path)
            plot_norm_evolution(norm_data, label, args.output_dir)

    # Norm comparison
    if args.compare_norms:
        norm_logs = discover_norm_logs(args.base_dir)
        if len(norm_logs) >= 2:
            print(f"\nComparing {len(norm_logs)} norm logs:")
            norm_list = []
            labels = []
            for p in norm_logs:
                lbl = derive_label(p)
                print(f"  - {lbl} ({p.name})")
                norm_list.append(load_norm_log(p))
                labels.append(lbl)
            plot_norm_comparison(norm_list, labels, args.output_dir)
        else:
            print("Need at least 2 norm logs for comparison.")

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
