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
#  HELPER — LOAD CLASSIFIER HEAD FROM CHECKPOINT
# ═══════════════════════════════════════════════════════════════════
def load_classifier_head(
    checkpoint_path: Path,
    model_type: str = "resnet18",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load only the classifier head weights from a saved checkpoint.

    For ResNet18 (our GTSRBResNet18) the head is named `fc.weight` / `fc.bias`.
    For the small CNN baseline it is `classifier.4.weight` / `classifier.4.bias`
    (last Linear layer inside the Sequential classifier).

    Returns (W, b) as numpy arrays with shapes (num_classes, d) and (num_classes,),
    or None on failure.
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"  [warn] cannot load checkpoint {checkpoint_path}: {e}")
        return None

    if model_type == "resnet18":
        w_key, b_key = "fc.weight", "fc.bias"
    else:
        # CNN: classifier = [Flatten, Linear(2048,256), ReLU, Dropout, Linear(256,43)]
        # The last Linear is at index 4.
        w_key, b_key = "classifier.4.weight", "classifier.4.bias"

    if w_key not in state or b_key not in state:
        # Try to find any fc.* or classifier.*.weight with matching shape
        print(f"  [warn] expected keys {w_key}/{b_key} missing; available: "
              f"{[k for k in state.keys() if 'fc' in k or 'classifier' in k][:8]}")
        return None

    W = state[w_key].detach().cpu().numpy()
    b = state[b_key].detach().cpu().numpy()
    return W, b


# ═══════════════════════════════════════════════════════════════════
#  AXIS A — INJECTION SUCCESS
# ═══════════════════════════════════════════════════════════════════
def compute_axis_a(
    labels: np.ndarray,
    preds_triggered: np.ndarray,
    target: int,
    features_clean: np.ndarray | None = None,
    features_triggered: np.ndarray | None = None,
    head_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """A1 ASR (from preds) + optionally A3/A4/A5 logit-space metrics.

    Args:
        labels: true labels (N,)
        preds_triggered: predictions on triggered inputs (N,)
        target: target class label
        features_clean, features_triggered: (N, d) arrays. Required for A3-A5.
        head_weights: (W, b) tuple from load_classifier_head. Required for A3-A5.
    """
    non_target = labels != target
    target_mask = labels == target

    if non_target.sum() == 0:
        return {"asr": 0.0, "n_triggered": 0}

    asr = float((preds_triggered[non_target] == target).mean())
    result = {
        "asr": asr,
        "n_triggered": int(non_target.sum()),
    }

    # A3/A4/A5: logit-space separation (only if head weights + features provided)
    if head_weights is not None and features_clean is not None and features_triggered is not None:
        W, b = head_weights
        try:
            # Triggered logits over non-target samples
            feat_T = features_triggered[non_target]  # (N_T, d)
            feat_C = features_clean[target_mask]     # (N_C, d)

            logits_T = feat_T @ W.T + b  # (N_T, num_classes)
            logits_C = feat_C @ W.T + b  # (N_C, num_classes)

            # A3: mean target-class logit for triggered samples
            target_logit_T = logits_T[:, target]
            # A4: mean target-class logit for genuine samples
            target_logit_C = logits_C[:, target]

            # A5: margin — target logit minus max over non-target classes
            # (for triggered samples)
            mask_nontarget = np.ones(W.shape[0], dtype=bool)
            mask_nontarget[target] = False
            other_max_T = logits_T[:, mask_nontarget].max(axis=1)
            margin_T = target_logit_T - other_max_T

            # Same margin metric for genuine target-class samples (sanity check)
            other_max_C = logits_C[:, mask_nontarget].max(axis=1)
            margin_C = target_logit_C - other_max_C

            result.update({
                "target_logit_mean_triggered": float(target_logit_T.mean()),
                "target_logit_std_triggered":  float(target_logit_T.std()),
                "target_logit_mean_genuine":   float(target_logit_C.mean()),
                "target_logit_std_genuine":    float(target_logit_C.std()),
                "margin_triggered":            float(margin_T.mean()),
                "margin_genuine":              float(margin_C.mean()),
                "logit_ratio_T_over_C":        float(
                    target_logit_T.mean() / (abs(target_logit_C.mean()) + 1e-12)
                ),
            })
        except Exception as e:
            print(f"  [warn] logit-space metrics failed: {e}")

    return result


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

    # B7: source identity preservation (normalized version)
    #
    # Compare the per-class spread of triggered centroids against the per-class
    # spread of the original (clean) centroids. The reference scale is the natural
    # class structure of the model — not an arbitrary variance measure.
    #
    #   sip = mean_pairwise_dist(triggered per-class centroids)
    #         ─────────────────────────────────────────────────
    #         mean_pairwise_dist(clean    per-class centroids)
    #
    #   ≈ 1   -> triggered features preserve per-source inter-class structure
    #            exactly — the attack is a uniform shift that keeps source
    #            identity readable (characteristic of pixel-trigger heuristic).
    #   → 0   -> triggered features collapse to a single destination cluster;
    #            source identity is destroyed (characteristic of a designed
    #            attack that targets a specific feature-space region).
    #   > 1   -> triggered features are even more spread than clean
    #            (unusual, would indicate the attack disrupts the feature
    #            extractor beyond the original class structure).
    per_source_trig = []
    per_source_clean = []
    for c in non_target_classes:
        class_mask = labels == c
        if class_mask.sum() < 2:
            continue
        per_source_trig.append(features_triggered[class_mask].mean(axis=0))
        per_source_clean.append(features_clean[class_mask].mean(axis=0))

    if len(per_source_trig) >= 2:
        trig_centroids = np.stack(per_source_trig)
        clean_centroids = np.stack(per_source_clean)

        trig_dists = np.sqrt(np.maximum(_sq_dist(trig_centroids, trig_centroids), 0))
        clean_dists = np.sqrt(np.maximum(_sq_dist(clean_centroids, clean_centroids), 0))

        iu = np.triu_indices(len(trig_centroids), k=1)
        mean_trig_spread = float(trig_dists[iu].mean())
        mean_clean_spread = float(clean_dists[iu].mean())

        source_identity_preservation = float(
            mean_trig_spread / (mean_clean_spread + 1e-12)
        )
    else:
        source_identity_preservation = 0.0

    return {
        "centroid_l2": centroid_l2,
        "centroid_cos": centroid_cos,
        "concentration": concentration,
        "concentration_clean": concentration_clean,
        "concentration_ratio": concentration_ratio,
        "shift_rank_eff": shift_rank,
        "shift_top3_energy": top3_energy,
        "shift_alignment": shift_alignment,
        "source_identity_preservation": source_identity_preservation,
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

    probe_weights = None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        probe = LogisticRegression(max_iter=1000, random_state=seed)
        probe.fit(X_train, y_train)
        probe_acc = float(probe.score(X_test, y_test))
        # Retain the hyperplane direction for C6
        probe_weights = probe.coef_[0]  # (d,)
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
    top_pc = None
    try:
        pca = PCA(n_components=1, random_state=seed)
        proj = pca.fit_transform(combined).ravel()
        m0 = proj[group == 0].mean()
        m1 = proj[group == 1].mean()
        std_pooled = proj.std() + 1e-12
        spectral = float(abs(m0 - m1) / std_pooled)
        # Retain the top-PC direction for C6
        top_pc = pca.components_[0]  # (d,)
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

    # C6: probe direction vs top principal component alignment
    # Near 0 -> probe uses a direction orthogonal to top variance
    #           (spectral defenses fail, supervised probes succeed)
    # Near 1 -> probe and spectral defense would agree
    if probe_weights is not None and top_pc is not None:
        pw_norm = norm(probe_weights)
        tp_norm = norm(top_pc)
        if pw_norm > 1e-12 and tp_norm > 1e-12:
            probe_pc_alignment = float(
                abs(np.dot(probe_weights, top_pc) / (pw_norm * tp_norm))
            )
        else:
            probe_pc_alignment = float("nan")
    else:
        probe_pc_alignment = float("nan")

    return {
        "linear_probe_acc": probe_acc,
        "mmd2_rbf": mmd2,
        "wasserstein2": w2,
        "spectral_score": spectral,
        "silhouette": sil,
        "probe_pc_alignment": probe_pc_alignment,
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
    head_weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Compute axes A, B, C for one round.

    If head_weights (W, b) is provided, axis A also includes the logit-space
    metrics (A3/A4/A5).
    """
    features_clean = data["features_clean"]
    features_triggered = data.get("features_triggered")
    labels = data["labels"]

    if features_triggered is None:
        return {}

    preds_triggered = data.get("preds_triggered")
    if preds_triggered is None:
        preds_triggered = np.full(len(labels), -1)

    axis_a = compute_axis_a(
        labels,
        preds_triggered,
        target,
        features_clean=features_clean,
        features_triggered=features_triggered,
        head_weights=head_weights,
    )
    axis_b = compute_axis_b(features_clean, features_triggered, labels, target)
    axis_c = compute_axis_c(features_clean, features_triggered, labels, target, seed=seed)

    return {**axis_a, **axis_b, **axis_c}


def compute_full_profile(
    exp_dir: Path,
    rounds: list[int],
    target: int,
    seed: int = 42,
    checkpoint_path: Path | None = None,
    model_type: str = "resnet18",
) -> dict:
    """Compute profile (axes A, B, C per round + axis D aggregated).

    If checkpoint_path is provided, the classifier head weights are loaded once
    and passed to every per-round computation, enabling logit-space metrics.
    """
    # Load classifier head once if requested
    head_weights = None
    if checkpoint_path is not None:
        head_weights = load_classifier_head(checkpoint_path, model_type=model_type)
        if head_weights is not None:
            W, b = head_weights
            print(f"  [head] loaded: W shape={W.shape}, b shape={b.shape}")
        else:
            print("  [head] unavailable — skipping logit-space metrics")

    trajectory = []
    for r in rounds:
        data = load_round_features(exp_dir, r)
        if data is None or "features_triggered" not in data:
            continue
        round_metrics = compute_round_profile(
            data, target, seed=seed, head_weights=head_weights,
        )
        if round_metrics:
            round_metrics["round"] = r
            trajectory.append(round_metrics)
            print(f"    round {r:>3d}: ASR={round_metrics['asr']:.2%}  "
                  f"probe={round_metrics.get('linear_probe_acc', float('nan')):.3f}  "
                  f"mmd={round_metrics.get('mmd2_rbf', float('nan')):.3f}  "
                  f"rank={round_metrics.get('shift_rank_eff', 0)}  "
                  f"pc_align={round_metrics.get('probe_pc_alignment', float('nan')):.3f}")

    # Final-round snapshot (from features_test.npz if present, else last trajectory entry)
    final_data = load_round_features(exp_dir, None)
    if final_data is not None and "features_triggered" in final_data:
        final_metrics = compute_round_profile(
            final_data, target, seed=seed, head_weights=head_weights,
        )
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
    parser.add_argument("--load-head", action="store_true",
                        help="Load classifier head from final_model.pt to "
                             "compute logit-space metrics (A3/A4/A5).")
    args = parser.parse_args()

    config_path = args.exp_dir / "config.yaml"
    config = load_config_yaml(config_path) if config_path.exists() else {}
    label = short_defense_label(config) if config else args.exp_dir.name
    model_type = str(config.get("model-type", "resnet18")) if config else "resnet18"

    # Add mal count to name if not already in label
    mal_ids = str(config.get("malicious-client-ids", "")).strip()
    n_mal = len([x for x in mal_ids.split(",") if x.strip()]) if mal_ids else 0

    # Resolve checkpoint path for head loading
    checkpoint_path = None
    if args.load_head:
        checkpoint_path = args.exp_dir / "checkpoints" / "final_model.pt"
        if not checkpoint_path.exists():
            print(f"  [warn] --load-head requested but {checkpoint_path} not found")
            checkpoint_path = None

    print(f"\n  Computing profile for: {label}")
    print(f"  Experiment dir: {args.exp_dir.name}")
    print(f"  Target label: {args.target_label}")
    print(f"  N malicious: {n_mal}")
    print(f"  Model type:   {model_type}")
    print(f"  Load head:    {checkpoint_path is not None}")

    profile = compute_full_profile(
        args.exp_dir,
        args.rounds,
        args.target_label,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
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
