"""Shared utilities for FL experiment analysis.

Data loaders, discovery helpers, publication style config, and plotting helpers
used by both plot_experiment.py and plot_comparison.py.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

# ── defaults ──────────────────────────────────────────────────────────
DEFAULT_BASE_DIR = Path(
    "/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

# ── publication style ─────────────────────────────────────────────────
PUB_STYLE = {
    # Font
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 3,
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    # Figure
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # Axes
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    # Legend
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "legend.handlelength": 1.5,
}

# Figure sizes (inches) — for 2-column papers
FIG_SINGLE_COL = (3.5, 2.5)
FIG_DOUBLE_COL = (7.0, 2.8)
FIG_DOUBLE_COL_TALL = (7.0, 4.5)

# Consistent defense visual encoding (colorblind-friendly + grayscale-distinct)
DEFENSE_STYLES = {
    "Clean":         {"color": "#1f77b4", "ls": "-",  "lw": 1.8},
    "No Defense":    {"color": "#d62728", "ls": "-",  "lw": 1.8},
    "Krum":          {"color": "#2ca02c", "ls": "--", "lw": 1.5},
    "Multi-Krum":    {"color": "#ff7f0e", "ls": "--", "lw": 1.5},
    "FedMedian":     {"color": "#9467bd", "ls": "-.", "lw": 1.5},
    "FedTrimmedAvg": {"color": "#8c564b", "ls": "-.", "lw": 1.5},
    "Bulyan":        {"color": "#e377c2", "ls": ":",  "lw": 1.8},
}

# Fallback style for unknown defenses
_DEFAULT_STYLE = {"color": "#7f7f7f", "ls": "-", "lw": 1.2}


# ── discovery ─────────────────────────────────────────────────────────
def discover_experiments(base_dir: Path) -> list[Path]:
    """Find experiment directories that contain rounds.csv."""
    return sorted(p.parent for p in base_dir.rglob("rounds.csv"))


def discover_norm_logs(base_dir: Path) -> list[Path]:
    """Find all *_norm_log.json files under base_dir."""
    return sorted(base_dir.glob("*_norm_log.json"))


# ── label helpers ─────────────────────────────────────────────────────
def derive_label(path: Path) -> str:
    """Derive a human-readable label from a path."""
    name = path.stem if path.suffix else path.name
    name = re.sub(r"_norm_log$", "", name)
    name = re.sub(r"_client_label_histograms$", "", name)
    name = re.sub(r"_r\d+_seed\d+$", "", name)
    name = re.sub(r"_seed\d+$", "", name)
    label = name.replace("-", " ").replace("_", " ").strip().title()
    label = label.replace("Resnet18", "ResNet18").replace("Resnet", "ResNet")
    label = label.replace("Cnn", "CNN").replace("Asr", "ASR")
    label = label.replace("Fedavg", "FedAvg").replace("Fedmedian", "FedMedian")
    label = label.replace("Nodefense", "No Defense")
    return label


def short_defense_label(config: dict) -> str:
    """Return a short label like 'Clean', 'No Defense', 'Krum', etc."""
    attack = str(config.get("attack-type", "none"))
    defense = str(config.get("defense-type", "none"))

    if attack == "none":
        return "Clean"

    _DEFENSE_NAMES = {
        "none": "No Defense",
        "krum": "Krum",
        "multi_krum": "Multi-Krum",
        "fed_median": "FedMedian",
        "fed_trimmed_avg": "FedTrimmedAvg",
        "bulyan": "Bulyan",
        "norm_clipping": "Norm Clipping",
    }
    return _DEFENSE_NAMES.get(defense, defense.replace("_", " ").title())


def get_defense_style(label: str) -> dict:
    """Return color/linestyle/linewidth for a defense label."""
    return DEFENSE_STYLES.get(label, _DEFAULT_STYLE)


# ── data loaders ──────────────────────────────────────────────────────
def _is_numeric(val: str) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def load_rounds_csv(path: Path) -> dict[str, np.ndarray]:
    """Load rounds.csv into a dict of numpy arrays keyed by column name."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if _is_numeric(r.get("round", ""))]
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
    """Simple YAML parser (avoids hard dependency)."""
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip().strip("'\"")
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
    return {
        "rounds": np.array([e["round"] for e in data]),
        "means":  np.array([e["stats"]["mean"] for e in data]),
        "maxs":   np.array([e["stats"]["max"] for e in data]),
        "mins":   np.array([e["stats"]["min"] for e in data]),
        "stds":   np.array([e["stats"]["std"] for e in data]),
    }


def load_experiment(exp_dir: Path) -> dict | None:
    """Load all data for one experiment. Returns None if rounds.csv is empty."""
    data = load_rounds_csv(exp_dir / "rounds.csv")
    if not data:
        return None

    summary = None
    if (exp_dir / "summary.json").exists():
        summary = load_summary_json(exp_dir / "summary.json")

    config = None
    if (exp_dir / "config.yaml").exists():
        config = load_config_yaml(exp_dir / "config.yaml")

    label = short_defense_label(config) if config else derive_label(exp_dir)

    return {
        "data": data,
        "summary": summary,
        "config": config,
        "label": label,
        "dir": exp_dir,
    }


# ── plotting helpers ──────────────────────────────────────────────────
def save_figure(fig, output_dir: Path, name: str, no_pdf: bool = False) -> None:
    """Save figure as PNG (and optionally PDF)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  [saved] {png_path}")
    if not no_pdf:
        pdf_path = output_dir / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"  [saved] {pdf_path}")


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average. Returns array of same length (edges padded)."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="same")
    return smoothed
