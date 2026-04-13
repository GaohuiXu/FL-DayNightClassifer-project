#!/usr/bin/env python3
"""Compare multiple framework profiles (cross-experiment analysis).

Produces:
- profile_comparison_table.csv : one row per experiment, all final metrics
- radar_comparison.png : radar chart of normalized metrics
- dynamics_comparison.png : line plots of Axis D metrics across experiments

Usage:
    python -m analysis.compare_profiles \
        --profile-dir /path/to/profiles \
        --output-dir /path/to/figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.common import PUB_STYLE, save_figure


# ─────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────
def load_profile(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def discover_profiles(profile_dir: Path) -> list[dict]:
    """Find all *_profile.json files in a directory."""
    profiles = []
    for p in sorted(profile_dir.glob("*_profile.json")):
        try:
            prof = load_profile(p)
            profiles.append(prof)
            print(f"  Loaded: {prof.get('label', p.stem)}")
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
    return profiles


# ─────────────────────────────────────────────────────────────
#  Comparison table
# ─────────────────────────────────────────────────────────────
_FINAL_METRIC_KEYS = [
    "asr",
    "centroid_l2",
    "centroid_cos",
    "concentration",
    "concentration_ratio",
    "shift_rank_eff",
    "shift_top3_energy",
    "shift_alignment",
    "linear_probe_acc",
    "mmd2_rbf",
    "wasserstein2",
    "spectral_score",
    "silhouette",
]

_DYNAMICS_KEYS = [
    "time_to_asr90",
    "asr_round5",
    "asr_round10",
    "asr_round25",
    "asr_stability_late",
    "u_shape_depth",
    "recovery_rate",
]


def save_comparison_table(profiles: list[dict], output_path: Path) -> None:
    """One row per experiment with final-round metrics + dynamics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for prof in profiles:
        row = {"experiment": prof.get("label", "?")}
        final = prof.get("final", {})
        for key in _FINAL_METRIC_KEYS:
            v = final.get(key)
            row[key] = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "")
        dyn = prof.get("dynamics", {})
        for key in _DYNAMICS_KEYS:
            v = dyn.get(key)
            row[key] = f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "")
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [saved] {output_path}")


# ─────────────────────────────────────────────────────────────
#  Radar chart
# ─────────────────────────────────────────────────────────────
# Metrics for the radar. "higher is better for attacker" in [0, 1] after rescaling.
# If metric is a stealth metric (where high = easy to detect = bad for attacker),
# we invert it.
_RADAR_SPEC = [
    # (label, key, invert, source)
    ("ASR",             "asr",              False, "final"),
    ("Centroid align",  "centroid_cos",     False, "final"),
    ("Stealth (1-probe)","linear_probe_acc", True,  "final"),
    ("Stealth (low MMD)","mmd2_rbf",        True,  "final"),
    ("Stealth (silh→0)","silhouette",       True,  "final"),
    ("Shift alignment", "shift_alignment",  False, "final"),
    ("ASR stability",   "asr_stability_late", True, "dynamics"),
    ("Fast injection",  "time_to_asr90",    True,  "dynamics"),
]


def _get_value(prof: dict, key: str, source: str):
    return prof.get(source, {}).get(key)


def _normalize_radar_values(profiles: list[dict]) -> dict[str, list[float]]:
    """For each metric, rescale across profiles to [0, 1] where 1 = "stronger attack".

    Returns dict: profile_label -> list of [0, 1] values in _RADAR_SPEC order.
    """
    # Collect raw values per metric across all profiles
    raw = {spec[1]: [] for spec in _RADAR_SPEC}
    for prof in profiles:
        for label, key, invert, source in _RADAR_SPEC:
            v = _get_value(prof, key, source)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raw[key].append(np.nan)
            else:
                raw[key].append(float(v))

    # Normalize each metric independently
    normalized = {spec[1]: [] for spec in _RADAR_SPEC}
    for label, key, invert, _ in _RADAR_SPEC:
        vals = np.array(raw[key], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            normalized[key] = [0.5] * len(profiles)
            continue
        vmin, vmax = finite.min(), finite.max()
        span = vmax - vmin
        if span < 1e-12:
            normalized[key] = [0.5] * len(profiles)
            continue

        scaled = (vals - vmin) / span  # [0, 1], high raw = 1
        # Replace NaN with 0
        scaled = np.where(np.isfinite(scaled), scaled, 0.0)
        if invert:
            scaled = 1.0 - scaled

        # Special case: time_to_asr90 with value -1 means "never" (worst attack)
        if key == "time_to_asr90":
            for i, prof in enumerate(profiles):
                v = _get_value(prof, key, "dynamics")
                if v == -1 or v is None:
                    scaled[i] = 0.0  # never reached 90% → worst
        normalized[key] = scaled.tolist()

    # Build per-profile list in radar spec order
    per_profile = {}
    for i, prof in enumerate(profiles):
        label = prof.get("label", f"exp{i}")
        per_profile[label] = [normalized[spec[1]][i] for spec in _RADAR_SPEC]
    return per_profile


def plot_radar(profiles: list[dict], output_dir: Path) -> None:
    """Radar chart comparing all profiles on the normalized metric set."""
    if len(profiles) < 2:
        return

    normalized = _normalize_radar_values(profiles)
    labels = [spec[0] for spec in _RADAR_SPEC]
    n_metrics = len(labels)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized)))
    for (exp_label, values), color in zip(normalized.items(), colors):
        values = values + values[:1]
        ax.plot(angles, values, color=color, linewidth=1.4, label=exp_label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_title("Attack profile comparison (1 = stronger attack)", fontsize=9, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=6)

    save_figure(fig, output_dir, "radar_comparison")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  Dynamics comparison
# ─────────────────────────────────────────────────────────────
def plot_dynamics(profiles: list[dict], output_dir: Path) -> None:
    """Line plot of ASR trajectory across experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))

    # Panel 1: ASR trajectory
    ax = axes[0]
    for prof, color in zip(profiles, colors):
        traj = prof.get("trajectory", [])
        if not traj:
            continue
        rounds = [t["round"] for t in traj]
        asrs = [t.get("asr", 0) for t in traj]
        ax.plot(rounds, asrs, "o-", color=color, markersize=3, linewidth=1.3,
                label=prof.get("label", "?"))
    ax.set_xlabel("Round")
    ax.set_ylabel("ASR")
    ax.set_title("ASR trajectory", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=5, loc="lower right")

    # Panel 2: linear probe accuracy (stealth) trajectory
    ax = axes[1]
    for prof, color in zip(profiles, colors):
        traj = prof.get("trajectory", [])
        if not traj:
            continue
        rounds = [t["round"] for t in traj]
        probes = [t.get("linear_probe_acc", np.nan) for t in traj]
        ax.plot(rounds, probes, "o-", color=color, markersize=3, linewidth=1.3,
                label=prof.get("label", "?"))
    ax.set_xlabel("Round")
    ax.set_ylabel("Linear probe accuracy")
    ax.set_title("Stealth trajectory (lower = stealthier)", fontsize=9)
    ax.set_ylim(0.4, 1.05)
    ax.grid(alpha=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.tight_layout()
    save_figure(fig, output_dir, "dynamics_comparison")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    plt.rcParams.update(PUB_STYLE)
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True,
                        help="Directory containing *_profile.json files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    print("Discovering profiles...")
    profiles = discover_profiles(args.profile_dir)
    if not profiles:
        print("No profiles found.")
        return

    # Sort by asr (descending) so strongest attack first
    profiles.sort(key=lambda p: -p.get("final", {}).get("asr", 0))

    print("\nSaving comparison table...")
    save_comparison_table(profiles, args.output_dir / "profile_comparison_table.csv")

    print("Generating radar chart...")
    plot_radar(profiles, args.output_dir)

    print("Generating dynamics plot...")
    plot_dynamics(profiles, args.output_dir)

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
