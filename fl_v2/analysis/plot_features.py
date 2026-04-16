#!/usr/bin/env python3
"""Visualize penultimate-layer features with t-SNE scatter plots.

Produces per-model t-SNE plots and a multi-panel comparison figure
showing how backdoor attacks manifest in representation space.

Usage:
    python -m analysis.plot_features \
        --exp-dirs /path/to/exp1 /path/to/exp2 ... \
        --target-label 2

    python -m analysis.plot_features \
        --exp-dirs /path/to/exp1 /path/to/exp2 \
        --target-label 2 \
        --output-dir analysis/figures/phaseC_representation
"""
import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from analysis.common import (
    DEFAULT_OUTPUT_DIR,
    FIG_DOUBLE_COL_TALL,
    FIG_SINGLE_COL,
    PUB_STYLE,
    load_config_yaml,
    save_figure,
    short_defense_label,
)


# ── data loading ─────────────────────────────────────────────────────
def load_features(exp_dir: Path, round_num: int | None = None) -> dict | None:
    """Load features from an experiment directory.

    Args:
        exp_dir: Path to experiment directory.
        round_num: Specific round to load (None = final model).

    Returns None if the file is missing or if every row of the saved features
    contains non-finite entries (e.g., Bagdasaryan scale=10 produces a
    degenerate global model in early rounds whose extracted features are
    entirely NaN — t-SNE cannot handle these and the caller should skip).
    """
    if round_num is not None:
        npz_path = exp_dir / "checkpoints" / f"round_{round_num:04d}_features.npz"
    else:
        npz_path = exp_dir / "checkpoints" / "features_test.npz"
    config_path = exp_dir / "config.yaml"

    if not npz_path.exists():
        print(f"  [skip] {exp_dir.name}: {npz_path.name} not found")
        return None

    data = dict(np.load(npz_path))

    # Guard: non-finite features break t-SNE / PCA.
    fc = data.get("features_clean")
    ft = data.get("features_triggered")
    if fc is not None and not np.isfinite(fc).all():
        n_bad = int((~np.isfinite(fc)).any(axis=1).sum())
        print(
            f"  [skip] {exp_dir.name} round={round_num}: "
            f"non-finite clean features ({n_bad}/{len(fc)})"
        )
        return None
    if ft is not None and not np.isfinite(ft).all():
        n_bad = int((~np.isfinite(ft)).any(axis=1).sum())
        print(
            f"  [skip] {exp_dir.name} round={round_num}: "
            f"non-finite triggered features ({n_bad}/{len(ft)})"
        )
        return None

    config = load_config_yaml(config_path) if config_path.exists() else {}
    label = short_defense_label(config) if config else exp_dir.name

    return {
        "data": data,
        "config": config,
        "label": label,
        "dir": exp_dir,
        "round": round_num,
    }


# ── t-SNE ────────────────────────────────────────────────────────────
def compute_tsne(
    features: np.ndarray,
    perplexity: int = 30,
    seed: int = 42,
) -> np.ndarray:
    """Run t-SNE on features, return (N, 2) coordinates."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    return tsne.fit_transform(features)


def compute_joint_tsne(
    features_clean: np.ndarray,
    features_triggered: np.ndarray | None = None,
    perplexity: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run t-SNE jointly on clean + triggered features.

    Returns (coords_clean, coords_triggered). If no triggered features,
    coords_triggered is None.
    """
    if features_triggered is not None:
        combined = np.concatenate([features_clean, features_triggered], axis=0)
        n_clean = len(features_clean)
        coords = compute_tsne(combined, perplexity=perplexity, seed=seed)
        return coords[:n_clean], coords[n_clean:]
    else:
        coords = compute_tsne(features_clean, perplexity=perplexity, seed=seed)
        return coords, None


# ── color mapping ────────────────────────────────────────────────────
def build_class_colors(num_classes: int, target_label: int):
    """Build a color array: muted gray for most classes, distinct color for target.

    Returns (colors, target_color) where colors is a list of RGBA tuples.
    """
    # Use tab20 for variety, but mute non-target classes
    base_cmap = plt.cm.tab20(np.linspace(0, 1, num_classes))

    # Mute all classes to low-alpha gray tones
    colors = []
    for i in range(num_classes):
        if i == target_label:
            colors.append((0.12, 0.47, 0.71, 0.7))  # blue, prominent
        else:
            # Use the base colormap but with reduced alpha
            c = base_cmap[i % 20]
            colors.append((c[0], c[1], c[2], 0.25))

    target_color = colors[target_label]
    return colors, target_color


# ── single scatter plot ──────────────────────────────────────────────
def plot_feature_scatter(
    ax: plt.Axes,
    coords_clean: np.ndarray,
    labels: np.ndarray,
    coords_triggered: np.ndarray | None,
    preds_triggered: np.ndarray | None,
    target_label: int,
    title: str,
    num_classes: int = 43,
) -> None:
    """Draw t-SNE scatter on a given axes."""
    colors, target_color = build_class_colors(num_classes, target_label)

    # Plot non-target clean samples first (background)
    non_target_mask = labels != target_label
    if non_target_mask.sum() > 0:
        sample_colors = [colors[int(l)] for l in labels[non_target_mask]]
        ax.scatter(
            coords_clean[non_target_mask, 0],
            coords_clean[non_target_mask, 1],
            c=sample_colors, s=3, linewidths=0,
            rasterized=True,
        )

    # Plot target class clean samples (highlighted)
    target_mask = labels == target_label
    if target_mask.sum() > 0:
        ax.scatter(
            coords_clean[target_mask, 0],
            coords_clean[target_mask, 1],
            c=[target_color], s=8, linewidths=0,
            label=f"Class {target_label} (target)",
            rasterized=True,
        )

    # Overlay triggered samples
    if coords_triggered is not None:
        ax.scatter(
            coords_triggered[:, 0],
            coords_triggered[:, 1],
            c="red", marker="*", s=6, alpha=0.4, linewidths=0,
            label="Triggered",
            rasterized=True,
        )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


# ── individual plot ──────────────────────────────────────────────────
def plot_single_experiment(
    exp: dict,
    target_label: int,
    output_dir: Path,
    perplexity: int = 30,
) -> None:
    """Generate t-SNE plot for a single experiment."""
    data = exp["data"]
    label = exp["label"]

    features_triggered = data.get("features_triggered")
    print(f"  Computing t-SNE for {label}...")
    coords_clean, coords_triggered = compute_joint_tsne(
        data["features_clean"],
        features_triggered,
        perplexity=perplexity,
    )

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL)
    plot_feature_scatter(
        ax, coords_clean, data["labels"],
        coords_triggered, data.get("preds_triggered"),
        target_label, label,
    )
    ax.legend(fontsize=6, loc="lower right", markerscale=1.5)
    fig.tight_layout()

    safe_name = label.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe_name}_tsne")
    plt.close(fig)


# ── comparison panel ─────────────────────────────────────────────────
def plot_comparison_panel(
    experiments: list[dict],
    target_label: int,
    output_dir: Path,
    perplexity: int = 30,
) -> None:
    """Multi-panel t-SNE comparison figure."""
    n = len(experiments)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig_w = 7.0
    fig_h = 3.0 * nrows + 0.3

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, exp in enumerate(experiments):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        data = exp["data"]
        label = exp["label"]

        features_triggered = data.get("features_triggered")
        print(f"  Computing t-SNE for {label}...")
        coords_clean, coords_triggered = compute_joint_tsne(
            data["features_clean"],
            features_triggered,
            perplexity=perplexity,
        )

        plot_feature_scatter(
            ax, coords_clean, data["labels"],
            coords_triggered, data.get("preds_triggered"),
            target_label, label,
        )

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Shared legend
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels_leg,
            loc="lower center", ncol=min(3, len(handles)),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=7, markerscale=1.5,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    save_figure(fig, output_dir, "comparison_panel_tsne")
    plt.close(fig)


# ── trajectory filmstrip ─────────────────────────────────────────────
def plot_trajectory_filmstrip(
    exp_dir: Path,
    rounds: list[int],
    target_label: int,
    output_dir: Path,
    perplexity: int = 30,
) -> None:
    """Generate a filmstrip of t-SNE plots across training rounds for one experiment."""
    # Load config for label
    config_path = exp_dir / "config.yaml"
    config = load_config_yaml(config_path) if config_path.exists() else {}
    exp_label = short_defense_label(config) if config else exp_dir.name

    # Load features for each round
    round_data = []
    for r in rounds:
        exp = load_features(exp_dir, round_num=r)
        if exp is not None:
            round_data.append((r, exp))
        else:
            print(f"  [skip] round {r} features not found for {exp_dir.name}")

    if len(round_data) < 2:
        print(f"  [skip] need at least 2 rounds for trajectory, got {len(round_data)}")
        return

    n = len(round_data)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.8), squeeze=False)

    for idx, (r, exp) in enumerate(round_data):
        ax = axes[0][idx]
        data = exp["data"]

        features_triggered = data.get("features_triggered")
        print(f"  Computing t-SNE for {exp_label} round {r}...")
        coords_clean, coords_triggered = compute_joint_tsne(
            data["features_clean"],
            features_triggered,
            perplexity=perplexity,
        )

        plot_feature_scatter(
            ax, coords_clean, data["labels"],
            coords_triggered, data.get("preds_triggered"),
            target_label, f"Round {r}",
        )

    # Shared legend
    handles, labels_leg = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels_leg,
            loc="lower center", ncol=min(3, len(handles)),
            bbox_to_anchor=(0.5, -0.02),
            fontsize=6, markerscale=1.5,
        )

    fig.suptitle(f"{exp_label} — Trajectory", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    safe_name = exp_label.lower().replace(" ", "_").replace("-", "_")
    save_figure(fig, output_dir, f"{safe_name}_trajectory")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────
def main():
    plt.rcParams.update(PUB_STYLE)

    parser = argparse.ArgumentParser(
        description="Visualize penultimate-layer features with t-SNE."
    )
    parser.add_argument(
        "--exp-dirs", type=Path, nargs="+", required=True,
        help="Experiment directories containing checkpoints/features_test.npz.",
    )
    parser.add_argument(
        "--target-label", type=int, default=2,
        help="Target class label to highlight (default: 2).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_OUTPUT_DIR / "phaseC_representation",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--perplexity", type=int, default=30,
        help="t-SNE perplexity (default: 30).",
    )
    parser.add_argument(
        "--rounds", type=int, nargs="+", default=None,
        help="Checkpoint rounds for trajectory filmstrip (e.g., 0 5 10 25 50 75 100).",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Skip PDF output.",
    )
    args = parser.parse_args()

    # Trajectory mode: generate filmstrips across rounds
    if args.rounds:
        print(f"Trajectory mode: rounds {args.rounds}\n")
        for d in args.exp_dirs:
            plot_trajectory_filmstrip(
                d, args.rounds, args.target_label, args.output_dir, args.perplexity,
            )
        print(f"\nAll trajectory figures saved to: {args.output_dir}")
        return

    # Standard mode: load final features and compare across experiments
    experiments = []
    for d in args.exp_dirs:
        exp = load_features(d)
        if exp:
            experiments.append(exp)
            n_feat = exp["data"]["features_clean"].shape
            has_trig = "features_triggered" in exp["data"]
            print(f"  Loaded: {exp['label']:20s} features={n_feat} triggered={has_trig}")

    if not experiments:
        print("No experiments with features found.")
        return

    # Sort: Clean first, then No Defense, then alphabetical
    _ORDER = {"Clean": 0, "No Defense": 1}
    experiments.sort(key=lambda e: (_ORDER.get(e["label"], 99), e["label"]))

    print(f"\nVisualizing {len(experiments)} experiments (target_label={args.target_label}):\n")

    # Individual plots
    for exp in experiments:
        plot_single_experiment(exp, args.target_label, args.output_dir, args.perplexity)

    # Comparison panel
    if len(experiments) >= 2:
        print("\nGenerating comparison panel...")
        plot_comparison_panel(experiments, args.target_label, args.output_dir, args.perplexity)

    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
