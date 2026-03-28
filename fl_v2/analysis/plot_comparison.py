#!/usr/bin/env python3
"""Multi-experiment comparison figures for FL defense evaluation.

Produces publication-quality plots comparing defenses against backdoor attacks.

Usage:
    python -m analysis.plot_comparison
    python -m analysis.plot_comparison --exp-dirs /path/to/exp1 /path/to/exp2
    python -m analysis.plot_comparison --figures comparison bars table
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import (
    DEFAULT_BASE_DIR,
    DEFAULT_OUTPUT_DIR,
    FIG_DOUBLE_COL,
    FIG_SINGLE_COL,
    PUB_STYLE,
    discover_experiments,
    get_defense_style,
    load_experiment,
    save_figure,
    smooth_curve,
)


# ── comparison figures ────────────────────────────────────────────────
def plot_defense_comparison(
    experiments: list[dict],
    output_dir: Path,
    smooth_window: int = 0,
) -> None:
    """Two-panel figure: Test Accuracy and ASR vs Round for all defenses."""
    fig, (ax_acc, ax_asr) = plt.subplots(
        1, 2, figsize=FIG_DOUBLE_COL, sharey=True,
    )

    for exp in experiments:
        data = exp["data"]
        label = exp["label"]
        style = get_defense_style(label)
        rounds = data["round"]

        acc = data["test_accuracy"] * 100
        if smooth_window > 1:
            acc = smooth_curve(acc, smooth_window)
        ax_acc.plot(rounds, acc, label=label, **style)

        if "asr" in data:
            asr = data["asr"] * 100
            if smooth_window > 1:
                asr = smooth_curve(asr, smooth_window)
            ax_asr.plot(rounds, asr, label=label, **style)

    ax_acc.set_xlabel("Communication Round")
    ax_acc.set_ylabel("Test Accuracy (%)")
    ax_acc.set_ylim(-2, 102)
    ax_acc.set_yticks(range(0, 101, 20))

    ax_asr.set_xlabel("Communication Round")
    ax_asr.set_ylabel("Attack Success Rate (%)")
    ax_asr.set_ylim(-2, 102)
    ax_asr.set_yticks(range(0, 101, 20))

    # Shared legend below
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(4, len(experiments)),
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_figure(fig, output_dir, "defense_comparison")
    plt.close(fig)


def plot_summary_bars(
    experiments: list[dict],
    output_dir: Path,
) -> None:
    """Grouped bar chart: Final Accuracy and Final ASR per defense."""
    labels = []
    accs = []
    asrs = []
    has_asr_flags = []

    for exp in experiments:
        data = exp["data"]
        labels.append(exp["label"])
        accs.append(data["test_accuracy"][-1] * 100)
        if "asr" in data:
            asrs.append(data["asr"][-1] * 100)
            has_asr_flags.append(True)
        else:
            asrs.append(0.0)
            has_asr_flags.append(False)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    bars_acc = ax.bar(x - width / 2, accs, width, label="Test Accuracy",
                      color="#1f77b4", edgecolor="white", linewidth=0.5)
    bars_asr = ax.bar(x + width / 2, asrs, width, label="ASR",
                      color="#d62728", edgecolor="white", linewidth=0.5)

    # Hide ASR bars for clean baselines
    for i, has_asr in enumerate(has_asr_flags):
        if not has_asr:
            bars_asr[i].set_visible(False)

    # Value labels on bars
    for bar in bars_acc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}", ha="center", va="bottom", fontsize=6)
    for i, bar in enumerate(bars_asr):
        if has_asr_flags[i]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()

    save_figure(fig, output_dir, "summary_bars")
    plt.close(fig)


def generate_summary_table(
    experiments: list[dict],
    output_dir: Path,
) -> None:
    """Generate summary table as LaTeX (.tex) and CSV (.csv)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp in experiments:
        data = exp["data"]
        summary = exp.get("summary")
        row = {
            "Defense": exp["label"],
            "Final Acc (%)": f"{data['test_accuracy'][-1] * 100:.1f}",
            "Best Acc (%)": (
                f"{summary['best_test_accuracy']['test_accuracy'] * 100:.1f}"
                if summary and "best_test_accuracy" in summary else "—"
            ),
            "Final ASR (%)": (
                f"{data['asr'][-1] * 100:.1f}" if "asr" in data else "—"
            ),
            "Peak ASR (%)": (
                f"{summary['best_asr']['asr'] * 100:.1f}"
                if summary and "best_asr" in summary else "—"
            ),
            "Final TCA (%)": (
                f"{data['target_class_clean_accuracy'][-1] * 100:.1f}"
                if "target_class_clean_accuracy" in data else "—"
            ),
        }
        rows.append(row)

    # CSV
    csv_path = output_dir / "summary_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {csv_path}")

    # LaTeX
    tex_path = output_dir / "summary_table.tex"
    cols = list(rows[0].keys())
    col_fmt = "l" + "r" * (len(cols) - 1)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Defense comparison on GTSRB with pixel-trigger backdoor (ResNet18, 100 rounds).}\n")
        f.write("\\label{tab:defense_comparison}\n")
        f.write(f"\\begin{{tabular}}{{{col_fmt}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(cols) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            vals = [row[c] for c in cols]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"  [saved] {tex_path}")


def plot_per_defense_dual(
    experiments: list[dict],
    output_dir: Path,
) -> None:
    """Grid of per-defense panels with dual axes (accuracy + ASR)."""
    # Only backdoor experiments (those with ASR data)
    backdoor_exps = [e for e in experiments if "asr" in e["data"]]
    if not backdoor_exps:
        return

    n = len(backdoor_exps)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.0, 2.4 * nrows), squeeze=False,
    )

    for idx, exp in enumerate(backdoor_exps):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        data = exp["data"]
        rounds = data["round"]

        # Accuracy on left axis
        color_acc = "#1f77b4"
        ax.plot(rounds, data["test_accuracy"] * 100, color=color_acc, lw=1.2)
        ax.set_ylabel("Acc (%)", color=color_acc, fontsize=7)
        ax.tick_params(axis="y", labelcolor=color_acc, labelsize=6)
        ax.set_ylim(-2, 102)
        ax.set_title(exp["label"], fontsize=8)
        ax.set_xlabel("Round", fontsize=7)

        # ASR on right axis
        ax2 = ax.twinx()
        color_asr = "#d62728"
        ax2.plot(rounds, data["asr"] * 100, color=color_asr, lw=1.2, ls="--")
        ax2.set_ylabel("ASR (%)", color=color_asr, fontsize=7)
        ax2.tick_params(axis="y", labelcolor=color_asr, labelsize=6)
        ax2.set_ylim(-2, 102)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    save_figure(fig, output_dir, "per_defense_dual_axis")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────
def main():
    plt.rcParams.update(PUB_STYLE)

    parser = argparse.ArgumentParser(
        description="Compare FL experiment results across defenses."
    )
    parser.add_argument(
        "--base-dir", type=Path, default=DEFAULT_BASE_DIR,
        help="Base output directory to scan for experiments.",
    )
    parser.add_argument(
        "--exp-dirs", type=Path, nargs="+", default=None,
        help="Explicit list of experiment directories to compare.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_OUTPUT_DIR / "comparison",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--figures", nargs="+",
        default=["all"],
        choices=["all", "comparison", "bars", "table", "dual"],
        help="Which figures to generate.",
    )
    parser.add_argument(
        "--smooth", type=int, default=0,
        help="Moving average window for curve smoothing (0=off).",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF output (save PNG only).",
    )
    args = parser.parse_args()

    # Discover or use explicit experiment dirs
    if args.exp_dirs:
        exp_dirs = args.exp_dirs
    else:
        exp_dirs = discover_experiments(args.base_dir)

    if not exp_dirs:
        print("No experiments found.")
        return

    # Load all experiments
    experiments = []
    for d in exp_dirs:
        exp = load_experiment(d)
        if exp:
            experiments.append(exp)
            print(f"  Loaded: {exp['label']:20s} ({d.name})")

    if len(experiments) < 2:
        print("Need at least 2 experiments for comparison.")
        return

    # Sort: Clean first, then No Defense, then alphabetical
    _ORDER = {"Clean": 0, "No Defense": 1}
    experiments.sort(key=lambda e: (_ORDER.get(e["label"], 99), e["label"]))

    print(f"\nComparing {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp['label']}")

    figs = set(args.figures)
    do_all = "all" in figs

    if do_all or "comparison" in figs:
        print("\nGenerating defense comparison plot...")
        plot_defense_comparison(experiments, args.output_dir, args.smooth)

    if do_all or "bars" in figs:
        print("\nGenerating summary bar chart...")
        plot_summary_bars(experiments, args.output_dir)

    if do_all or "table" in figs:
        print("\nGenerating summary table...")
        generate_summary_table(experiments, args.output_dir)

    if do_all or "dual" in figs:
        print("\nGenerating per-defense dual-axis plots...")
        plot_per_defense_dual(experiments, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
