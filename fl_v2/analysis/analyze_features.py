#!/usr/bin/env python3
"""Quantitative analysis of backdoor representations in 512-dim feature space.

Computes metrics per checkpoint round to show the trajectory of backdoor formation:
- Cosine similarity: triggered features → target class centroid
- L2 distance: triggered features → target class centroid
- Per-sample perturbation: L2 distance between clean and triggered features
- Spectral signature: PCA separation between genuine target-class and triggered features
- Silhouette score: how separable are genuine vs triggered features

Usage:
    python -m analysis.analyze_features \
        --exp-dirs /path/to/exp1 /path/to/exp2 \
        --rounds 0 5 10 25 50 75 100 \
        --target-label 2 \
        --output-dir /path/to/output
"""
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from analysis.common import (
    DEFAULT_OUTPUT_DIR,
    PUB_STYLE,
    load_config_yaml,
    save_figure,
    short_defense_label,
)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))


def analyze_single(
    features_clean: np.ndarray,
    features_triggered: np.ndarray,
    labels: np.ndarray,
    preds_triggered: np.ndarray,
    target_label: int,
) -> dict:
    """Compute all metrics for one checkpoint."""
    non_target = labels != target_label
    class2_mask = labels == target_label

    # Centroids
    centroid_class2 = features_clean[class2_mask].mean(axis=0)
    centroid_triggered = features_triggered[non_target].mean(axis=0)

    # ASR
    asr = float((preds_triggered[non_target] == target_label).mean())

    # Cosine similarity to target centroid
    cos_trig = np.array([cos_sim(features_triggered[i], centroid_class2)
                         for i in range(len(features_triggered)) if labels[i] != target_label])
    cos_clean = np.array([cos_sim(features_clean[i], centroid_class2)
                          for i in range(len(features_clean)) if labels[i] != target_label])

    # L2 distance to target centroid
    l2_trig = np.array([norm(features_triggered[i] - centroid_class2)
                        for i in range(len(features_triggered)) if labels[i] != target_label])
    l2_clean = np.array([norm(features_clean[i] - centroid_class2)
                         for i in range(len(features_clean)) if labels[i] != target_label])

    # Per-sample perturbation
    l2_shift = np.array([norm(features_triggered[i] - features_clean[i])
                         for i in range(len(features_clean)) if labels[i] != target_label])

    # Centroid-level cosine
    cos_centroids = cos_sim(centroid_triggered, centroid_class2)

    # Spectral signature: PCA on {genuine class-2 + triggered non-target}
    genuine = features_clean[class2_mask]
    triggered_nt = features_triggered[non_target]
    combined = np.concatenate([genuine, triggered_nt], axis=0)
    group_labels = np.array([0] * len(genuine) + [1] * len(triggered_nt))

    spectral_score = 0.0
    if len(genuine) > 1 and len(triggered_nt) > 1:
        pca = PCA(n_components=1)
        proj = pca.fit_transform(combined).ravel()
        # Separation: difference in means normalized by pooled std
        mean_genuine = proj[group_labels == 0].mean()
        mean_triggered = proj[group_labels == 1].mean()
        std_pooled = proj.std() + 1e-12
        spectral_score = abs(mean_genuine - mean_triggered) / std_pooled

    # Silhouette score
    sil = 0.0
    if len(genuine) > 1 and len(triggered_nt) > 1 and len(combined) > 2:
        # Subsample if too large (silhouette is O(n²))
        max_samples = 5000
        if len(combined) > max_samples:
            idx = np.random.RandomState(42).choice(len(combined), max_samples, replace=False)
            sil = silhouette_score(combined[idx], group_labels[idx])
        else:
            sil = silhouette_score(combined, group_labels)

    return {
        "asr": asr,
        "cos_trig_to_target_mean": float(cos_trig.mean()),
        "cos_trig_to_target_std": float(cos_trig.std()),
        "cos_clean_to_target_mean": float(cos_clean.mean()),
        "cos_shift": float(cos_trig.mean() - cos_clean.mean()),
        "l2_trig_to_target_mean": float(l2_trig.mean()),
        "l2_clean_to_target_mean": float(l2_clean.mean()),
        "l2_shift": float(l2_trig.mean() - l2_clean.mean()),
        "perturbation_l2_mean": float(l2_shift.mean()),
        "perturbation_l2_std": float(l2_shift.std()),
        "cos_centroids": cos_centroids,
        "spectral_score": spectral_score,
        "silhouette": float(sil),
    }


def analyze_experiment_trajectory(
    exp_dir: Path,
    rounds: list[int],
    target_label: int,
) -> list[dict]:
    """Analyze all checkpoint rounds for one experiment."""
    results = []
    for r in rounds:
        if r == 0:
            npz_name = "round_0000_features.npz"
        else:
            npz_name = f"round_{r:04d}_features.npz"
        npz_path = exp_dir / "checkpoints" / npz_name
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        if "features_triggered" not in data:
            continue

        metrics = analyze_single(
            data["features_clean"],
            data["features_triggered"],
            data["labels"],
            data["preds_triggered"],
            target_label,
        )
        metrics["round"] = r
        results.append(metrics)
        print(f"    round {r:>3d}: ASR={metrics['asr']:.2%}  "
              f"cos_shift={metrics['cos_shift']:+.4f}  "
              f"spectral={metrics['spectral_score']:.3f}  "
              f"silhouette={metrics['silhouette']:.3f}")

    return results


def plot_trajectory(
    all_results: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Plot quantitative metrics vs round for all experiments."""
    metrics_to_plot = [
        ("asr", "ASR", "Attack Success Rate"),
        ("cos_centroids", "Cosine Sim", "Triggered Centroid → Target Centroid"),
        ("spectral_score", "Spectral Score", "PCA Separation (genuine vs triggered)"),
        ("perturbation_l2_mean", "L2 Perturbation", "Mean Per-Sample Feature Shift"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

    for idx, (key, ylabel, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2][idx % 2]
        for label, results in all_results.items():
            if not results:
                continue
            rounds = [r["round"] for r in results]
            values = [r[key] for r in results]
            ax.plot(rounds, values, marker="o", markersize=3, label=label, linewidth=1.5)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6)

    fig.tight_layout()
    save_figure(fig, output_dir, "trajectory_metrics")
    plt.close(fig)


def save_trajectory_csv(
    all_results: dict[str, list[dict]],
    output_dir: Path,
) -> None:
    """Save all trajectory metrics to a CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trajectory_metrics.csv"

    rows = []
    for label, results in all_results.items():
        for r in results:
            rows.append({"experiment": label, **r})

    if not rows:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {csv_path}")


def main():
    plt.rcParams.update(PUB_STYLE)

    parser = argparse.ArgumentParser(
        description="Quantitative analysis of backdoor representations."
    )
    parser.add_argument(
        "--exp-dirs", type=Path, nargs="+", required=True,
        help="Experiment directories.",
    )
    parser.add_argument(
        "--rounds", type=int, nargs="+",
        default=[0, 5, 10, 25, 50, 75, 100],
        help="Checkpoint rounds to analyze.",
    )
    parser.add_argument(
        "--target-label", type=int, default=2,
        help="Target class label (default: 2).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_OUTPUT_DIR / "phaseC_quantitative",
        help="Directory to save results.",
    )
    args = parser.parse_args()

    all_results = {}
    for d in args.exp_dirs:
        config_path = d / "config.yaml"
        config = load_config_yaml(config_path) if config_path.exists() else {}
        label = short_defense_label(config) if config else d.name

        # Include malicious count in label for clarity
        mal_ids = str(config.get("malicious-client-ids", ""))
        n_mal = len([x for x in mal_ids.split(",") if x.strip()]) if mal_ids.strip() else 0
        if n_mal > 0:
            label = f"{label} ({n_mal}mal)"

        print(f"\n  Analyzing: {label} ({d.name})")
        results = analyze_experiment_trajectory(d, args.rounds, args.target_label)
        if results:
            all_results[label] = results

    if not all_results:
        print("No trajectory data found.")
        return

    # Save CSV
    save_trajectory_csv(all_results, args.output_dir)

    # Plot trajectory
    print("\n  Generating trajectory plot...")
    plot_trajectory(all_results, args.output_dir)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
