#!/usr/bin/env python3
"""Representation-space analysis framework for backdoor attacks.

Implements the 4-axis, ~15-metric profile defined in
docs/representation_space_framework.md:

    Axis A — Injection Success    (ASR, target margin)
    Axis B — Injection Geometry   (centroid, concentration, shift rank/alignment)
    Axis C — Injection Stealth    (linear probe, MMD, Wasserstein, spectral, silhouette)
    Axis D — Injection Dynamics   (time-to-90, stability, U-shape, recovery)

Operates entirely on saved .npz feature files — no retraining.

Usage:
    python -m analysis.framework_metrics \
        --exp-dir /path/to/experiment \
        --rounds 0 5 10 25 50 75 100 \
        --target-label 2 \
        --output-dir /path/to/profiles
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from analysis.common import load_config_yaml, short_defense_label


# ═══════════════════════════════════════════════════════════════════
#  AXIS A — INJECTION SUCCESS
# ═══════════════════════════════════════════════════════════════════
def compute_axis_a(
    labels: np.ndarray,
    preds_triggered: np.ndarray,
    target: int,
) -> dict:
    """A1 ASR + A2 target margin (if logits available — here we only have preds).

    Args:
        labels: true labels (N,)
        preds_triggered: predictions on triggered inputs (N,)
        target: target class label
    """
    non_target = labels != target
    if non_target.sum() == 0:
        return {"asr": 0.0, "n_triggered": 0}

    asr = float((preds_triggered[non_target] == target).mean())
    return {
        "asr": asr,
        "n_triggered": int(non_target.sum()),
    }


# ═══════════════════════════════════════════════════════════════════
#  AXIS B — INJECTION GEOMETRY
# ═══════════════════════════════════════════════════════════════════
def compute_axis_b(
    features_clean: np.ndarray,
    features_triggered: np.ndarray,
    labels: np.ndarray,
    target: int,
) -> dict:
    """Centroid distance, concentration, shift rank/alignment."""
    non_target = labels != target
    target_mask = labels == target

    if non_target.sum() == 0 or target_mask.sum() == 0:
        return {}

    feat_T = features_triggered[non_target]  # triggered non-target features
    feat_C = features_clean[target_mask]     # genuine target-class features
    feat_orig = features_clean[non_target]   # clean versions of the triggered samples

    mu_T = feat_T.mean(axis=0)
    mu_C = feat_C.mean(axis=0)

    # B1: centroid distance
    centroid_l2 = float(norm(mu_T - mu_C))

    # B2: centroid cosine
    centroid_cos = float(np.dot(mu_T, mu_C) / (norm(mu_T) * norm(mu_C) + 1e-12))

    # B3: concentration = mean per-dim variance of triggered features
    var_T = feat_T.var(axis=0)
    var_C = feat_C.var(axis=0)
    concentration = float(var_T.mean())
    concentration_clean = float(var_C.mean())

    # B4: concentration ratio
    concentration_ratio = float(concentration / (concentration_clean + 1e-12))

    # B5: shift direction rank
    shift = feat_T - feat_orig  # (N_T, d)
    # SVD via numpy; use full_matrices=False for efficiency
    try:
        s = svd(shift, full_matrices=False, compute_uv=False)
        sigma_max = s[0] if len(s) > 0 else 0.0
        if sigma_max > 0:
            shift_rank = int((s > 0.01 * sigma_max).sum())
            # Energy fraction captured by top 3 components
            top3_energy = float((s[:3] ** 2).sum() / (s ** 2).sum()) if len(s) >= 3 else 1.0
        else:
            shift_rank = 0
            top3_energy = 0.0
    except np.linalg.LinAlgError:
        shift_rank = -1
        top3_energy = -1.0

    # B6: shift alignment with natural class direction
    # For each source class (= each non-target class), compute
    #   alignment = cos(mean shift vector for that class,  μ_C - μ_source)
    # Average over classes, weighted by sample count.
    shift_alignments = []
    weights = []
    non_target_classes = np.unique(labels[non_target])
    for c in non_target_classes:
        class_mask = labels == c
        if class_mask.sum() < 2:
            continue
        src_feat = features_clean[class_mask]
        mu_src = src_feat.mean(axis=0)
        natural_dir = mu_C - mu_src
        nat_norm = norm(natural_dir)
        if nat_norm < 1e-12:
            continue

        # Mean shift vector for samples of class c
        class_shifts = features_triggered[class_mask] - features_clean[class_mask]
        mean_shift = class_shifts.mean(axis=0)
        shift_norm = norm(mean_shift)
        if shift_norm < 1e-12:
            continue

        align = float(np.dot(mean_shift, natural_dir) / (shift_norm * nat_norm))
        shift_alignments.append(align)
        weights.append(int(class_mask.sum()))

    if shift_alignments:
        weights = np.array(weights, dtype=float)
        alignments = np.array(shift_alignments)
        shift_alignment = float((alignments * weights).sum() / weights.sum())
    else:
        shift_alignment = 0.0

    return {
        "centroid_l2": centroid_l2,
        "centroid_cos": centroid_cos,
        "concentration": concentration,
        "concentration_clean": concentration_clean,
        "concentration_ratio": concentration_ratio,
        "shift_rank_eff": shift_rank,
        "shift_top3_energy": top3_energy,
        "shift_alignment": shift_alignment,
    }


# ═══════════════════════════════════════════════════════════════════
#  AXIS C — INJECTION STEALTH
# ═══════════════════════════════════════════════════════════════════
def _rbf_mmd2(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """Biased MMD² with RBF kernel. O((N+M)²) — subsample if large."""
    XX = np.exp(-gamma * _sq_dist(X, X))
    YY = np.exp(-gamma * _sq_dist(Y, Y))
    XY = np.exp(-gamma * _sq_dist(X, Y))
    return float(XX.mean() + YY.mean() - 2 * XY.mean())


def _sq_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance matrix."""
    A_sq = (A ** 2).sum(axis=1, keepdims=True)
    B_sq = (B ** 2).sum(axis=1, keepdims=True)
    return A_sq + B_sq.T - 2 * A @ B.T


def _median_heuristic_gamma(X: np.ndarray, Y: np.ndarray, max_samples: int = 500) -> float:
    """Pick RBF bandwidth γ = 1/(2σ²) using the median heuristic."""
    rng = np.random.RandomState(42)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]
    if len(Y) > max_samples:
        Y = Y[rng.choice(len(Y), max_samples, replace=False)]
    combined = np.concatenate([X, Y], axis=0)
    dists = np.sqrt(np.maximum(_sq_dist(combined, combined), 0))
    # Upper triangle (exclude diagonal)
    iu = np.triu_indices(len(combined), k=1)
    med = np.median(dists[iu])
    sigma = med if med > 1e-12 else 1.0
    return float(1.0 / (2 * sigma ** 2))


def _sinkhorn_w2(X: np.ndarray, Y: np.ndarray, reg: float = 1.0, n_iter: int = 200) -> float:
    """Entropy-regularized Wasserstein-2 distance (Sinkhorn).

    Subsampled to 500 points per side for tractability.
    """
    rng = np.random.RandomState(42)
    max_n = 500
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m

    # Squared Euclidean cost
    C = _sq_dist(X, Y)
    C_max = C.max() if C.max() > 0 else 1.0
    C_scaled = C / C_max  # normalize for numerical stability
    reg_scaled = reg

    K = np.exp(-C_scaled / reg_scaled)
    u = np.ones(n)
    v = np.ones(m)

    for _ in range(n_iter):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)

    T = np.diag(u) @ K @ np.diag(v)
    w2_sq = float((T * C).sum())  # back to original scale
    return float(np.sqrt(max(w2_sq, 0.0)))


def compute_axis_c(
    features_clean: np.ndarray,
    features_triggered: np.ndarray,
    labels: np.ndarray,
    target: int,
    seed: int = 42,
) -> dict:
    """Linear probe, MMD, Wasserstein, spectral, silhouette."""
    non_target = labels != target
    target_mask = labels == target

    feat_T = features_triggered[non_target]  # triggered
    feat_C = features_clean[target_mask]     # genuine target

    if len(feat_T) < 2 or len(feat_C) < 2:
        return {}

    # C1: linear probe accuracy
    # Balance: use the smaller of the two
    n_min = min(len(feat_T), len(feat_C))
    rng = np.random.RandomState(seed)
    idx_T = rng.choice(len(feat_T), n_min, replace=False)
    idx_C = rng.choice(len(feat_C), n_min, replace=False)
    X = np.concatenate([feat_C[idx_C], feat_T[idx_T]], axis=0)
    y = np.array([0] * n_min + [1] * n_min)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        probe = LogisticRegression(max_iter=1000, random_state=seed)
        probe.fit(X_train, y_train)
        probe_acc = float(probe.score(X_test, y_test))
    except Exception:
        probe_acc = float("nan")

    # C2: MMD² with RBF kernel (subsampled)
    # Limit to 1000 per side for speed
    max_mmd = 1000
    X_mmd = feat_T[rng.choice(len(feat_T), min(max_mmd, len(feat_T)), replace=False)]
    Y_mmd = feat_C[rng.choice(len(feat_C), min(max_mmd, len(feat_C)), replace=False)]
    gamma = _median_heuristic_gamma(X_mmd, Y_mmd)
    try:
        mmd2 = _rbf_mmd2(X_mmd, Y_mmd, gamma)
    except Exception:
        mmd2 = float("nan")

    # C3: Wasserstein-2 (Sinkhorn approx)
    try:
        w2 = _sinkhorn_w2(feat_T, feat_C, reg=0.1, n_iter=200)
    except Exception:
        w2 = float("nan")

    # C4: spectral signature score (top-PC separation)
    combined = np.concatenate([feat_C, feat_T], axis=0)
    group = np.array([0] * len(feat_C) + [1] * len(feat_T))
    try:
        pca = PCA(n_components=1, random_state=seed)
        proj = pca.fit_transform(combined).ravel()
        m0 = proj[group == 0].mean()
        m1 = proj[group == 1].mean()
        std_pooled = proj.std() + 1e-12
        spectral = float(abs(m0 - m1) / std_pooled)
    except Exception:
        spectral = float("nan")

    # C5: silhouette score (subsampled for O(n²) cost)
    try:
        max_sil = 3000
        if len(combined) > max_sil:
            idx = rng.choice(len(combined), max_sil, replace=False)
            sil = float(silhouette_score(combined[idx], group[idx]))
        else:
            sil = float(silhouette_score(combined, group))
    except Exception:
        sil = float("nan")

    return {
        "linear_probe_acc": probe_acc,
        "mmd2_rbf": mmd2,
        "wasserstein2": w2,
        "spectral_score": spectral,
        "silhouette": sil,
    }


# ═══════════════════════════════════════════════════════════════════
#  AXIS D — INJECTION DYNAMICS
# ═══════════════════════════════════════════════════════════════════
def compute_axis_d(trajectory: list[dict]) -> dict:
    """Dynamics metrics from a list of per-round metric dicts.

    Args:
        trajectory: list of dicts, each containing at least
            'round', 'asr', 'centroid_cos' (from axis B).
    """
    if len(trajectory) < 2:
        return {}

    rounds = np.array([t["round"] for t in trajectory])
    asrs = np.array([t.get("asr", 0.0) for t in trajectory])
    cos_cens = np.array([t.get("centroid_cos", 0.0) for t in trajectory])

    # D1: time-to-ASR-90
    above = np.where(asrs >= 0.90)[0]
    time_to_90 = int(rounds[above[0]]) if len(above) > 0 else -1

    # D2: early-round ASR snapshots
    def asr_at(r):
        idx = np.where(rounds == r)[0]
        return float(asrs[idx[0]]) if len(idx) > 0 else float("nan")

    asr_r5 = asr_at(5)
    asr_r10 = asr_at(10)
    asr_r25 = asr_at(25)

    # D3: late-round ASR stability (rounds ≥ 50)
    late_mask = rounds >= 50
    stab_asr = float(asrs[late_mask].std()) if late_mask.sum() >= 2 else float("nan")

    # D4: U-shape depth in centroid cosine
    u_depth = float(cos_cens.max() - cos_cens.min())

    # D5: recovery rate — slope from min to final
    min_idx = int(cos_cens.argmin())
    if min_idx < len(cos_cens) - 1:
        dr = rounds[-1] - rounds[min_idx]
        dcos = cos_cens[-1] - cos_cens[min_idx]
        recov = float(dcos / max(dr, 1))
    else:
        recov = 0.0

    # Bonus: final ASR
    asr_final = float(asrs[-1])

    return {
        "asr_final": asr_final,
        "time_to_asr90": time_to_90,
        "asr_round5": asr_r5,
        "asr_round10": asr_r10,
        "asr_round25": asr_r25,
        "asr_stability_late": stab_asr,
        "u_shape_depth": u_depth,
        "recovery_rate": recov,
    }


# ═══════════════════════════════════════════════════════════════════
#  ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════
def load_round_features(exp_dir: Path, round_num: int) -> dict | None:
    """Load features_test.npz for a given round (None = final)."""
    if round_num is None:
        path = exp_dir / "checkpoints" / "features_test.npz"
    else:
        path = exp_dir / "checkpoints" / f"round_{round_num:04d}_features.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def compute_round_profile(
    data: dict,
    target: int,
    seed: int = 42,
) -> dict:
    """Compute axes A, B, C for one round."""
    features_clean = data["features_clean"]
    features_triggered = data.get("features_triggered")
    labels = data["labels"]

    if features_triggered is None:
        return {}

    preds_triggered = data.get("preds_triggered")
    if preds_triggered is None:
        preds_triggered = np.full(len(labels), -1)

    axis_a = compute_axis_a(labels, preds_triggered, target)
    axis_b = compute_axis_b(features_clean, features_triggered, labels, target)
    axis_c = compute_axis_c(features_clean, features_triggered, labels, target, seed=seed)

    return {**axis_a, **axis_b, **axis_c}


def compute_full_profile(
    exp_dir: Path,
    rounds: list[int],
    target: int,
    seed: int = 42,
) -> dict:
    """Compute profile (axes A, B, C per round + axis D aggregated)."""
    trajectory = []
    for r in rounds:
        data = load_round_features(exp_dir, r)
        if data is None or "features_triggered" not in data:
            continue
        round_metrics = compute_round_profile(data, target, seed=seed)
        if round_metrics:
            round_metrics["round"] = r
            trajectory.append(round_metrics)
            print(f"    round {r:>3d}: ASR={round_metrics['asr']:.2%}  "
                  f"probe={round_metrics.get('linear_probe_acc', float('nan')):.3f}  "
                  f"mmd={round_metrics.get('mmd2_rbf', float('nan')):.3f}  "
                  f"rank={round_metrics.get('shift_rank_eff', 0)}")

    # Final-round snapshot (from features_test.npz if present, else last trajectory entry)
    final_data = load_round_features(exp_dir, None)
    if final_data is not None and "features_triggered" in final_data:
        final_metrics = compute_round_profile(final_data, target, seed=seed)
        final_metrics["round"] = "final"
    elif trajectory:
        final_metrics = {k: v for k, v in trajectory[-1].items()}
    else:
        final_metrics = {}

    # Dynamics (from trajectory)
    dynamics = compute_axis_d(trajectory)

    return {
        "experiment_dir": str(exp_dir),
        "target_label": target,
        "trajectory": trajectory,
        "final": final_metrics,
        "dynamics": dynamics,
    }


def save_profile_json(profile: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types to Python native for JSON
    def _sanitize(x):
        if isinstance(x, dict):
            return {k: _sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_sanitize(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    with open(output_path, "w") as f:
        json.dump(_sanitize(profile), f, indent=2)
    print(f"  [saved] {output_path}")


def save_profile_csv(profile: dict, output_path: Path) -> None:
    """Flat CSV: one row per round with all metrics."""
    if not profile.get("trajectory"):
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(profile["trajectory"][0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile["trajectory"]:
            writer.writerow(row)
    print(f"  [saved] {output_path}")


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Compute 4-axis representation-space framework profile."
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=[0, 5, 10, 25, 50, 75, 100])
    parser.add_argument("--target-label", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = args.exp_dir / "config.yaml"
    config = load_config_yaml(config_path) if config_path.exists() else {}
    label = short_defense_label(config) if config else args.exp_dir.name

    # Add mal count to name if not already in label
    mal_ids = str(config.get("malicious-client-ids", "")).strip()
    n_mal = len([x for x in mal_ids.split(",") if x.strip()]) if mal_ids else 0

    print(f"\n  Computing profile for: {label}")
    print(f"  Experiment dir: {args.exp_dir.name}")
    print(f"  Target label: {args.target_label}")
    print(f"  N malicious: {n_mal}")

    profile = compute_full_profile(
        args.exp_dir, args.rounds, args.target_label, seed=args.seed,
    )
    profile["label"] = label
    profile["n_malicious"] = n_mal

    safe_name = args.exp_dir.name
    json_path = args.output_dir / f"{safe_name}_profile.json"
    csv_path = args.output_dir / f"{safe_name}_trajectory.csv"
    save_profile_json(profile, json_path)
    save_profile_csv(profile, csv_path)

    if profile.get("dynamics"):
        print("\n  Dynamics summary:")
        for k, v in profile["dynamics"].items():
            print(f"    {k:25s} = {v}")


if __name__ == "__main__":
    main()
