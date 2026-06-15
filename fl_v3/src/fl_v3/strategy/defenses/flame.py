"""FLAME core (Nguyen et al., USENIX-Sec'22) — cluster → clip → noise.

Re-implemented from the fl_v2 oracle (``fl_v2/src/fl_v2/strategy/flame.py``)
and validated for byte-equivalence on a saved fixture (admitted set, clip bound,
noise sample, AND the final aggregated arrays all match).

Three steps each round:
  1. **Clustering** — HDBSCAN on pairwise cosine distance of client updates with
     ``min_cluster_size = n//2 + 1`` and ``allow_single_cluster=True``, so HDBSCAN
     can find AT MOST ONE cluster (the benign majority); the minority is labelled
     noise and dropped.
  2. **Clipping** — admitted updates clipped to the median update L2 norm ``S_t``.
  3. **Noise** — per-coordinate Gaussian std ``= noise_multiplier * S_t`` (paper
     Eq. 7, NO sqrt(d) normalisation), from a numpy Generator seeded off the run
     seed and the server round → bit-deterministic, varies by round.

**λ calibration is optimizer-dependent (the fl_v2 finding).** The paper's
λ=0.001 is calibrated for SGD lr=0.01; under Adam (~1000× larger updates) it
diverges in round 1, so the calibrated default is 1e-6. The constant is a config
knob, NOT hardcoded here — see ``flame-noise-multiplier`` in pyproject.toml and
the FLAME assumption card (T6). Carrying it wrong is the canonical
"units transplanted from a different optimizer/scale" bug Codex must check.

Paper: https://arxiv.org/abs/2101.02281
"""
from __future__ import annotations

from typing import List

import numpy as np

from fl_v3.strategy.aggregation_core import aggregate_weighted_updates
from fl_v3.strategy.defenses.base import DefenseDecision, build_flat_updates
from fl_v3.strategy.gradient_metrics import compute_update_norms


def flame_cluster_admitted(flat: np.ndarray, norms: np.ndarray) -> List[int]:
    """HDBSCAN on pairwise cosine distance; return the benign-majority indices.

    Byte-identical to the fl_v2 oracle ``FlameFedAvg._cluster_admitted``:
      * ``n < 4`` → admit all (degenerate; FedAvg "do no harm").
      * unit = update / max(norm, 1e-12); cos = clip(unit @ unit.T, -1, 1);
        dist = (1 - cos) as float64; symmetrise ``0.5*(d + d.T)``; zero diagonal.
      * HDBSCAN(min_cluster_size=n//2+1, min_samples=1, metric="precomputed",
        cluster_selection_method="eom", allow_single_cluster=True, copy=True).
      * admitted = labels >= 0; if none admitted → admit all.
    """
    n = int(flat.shape[0])
    if n < 4:
        return list(range(n))
    safe = np.where(norms < 1e-12, 1.0, norms)
    unit = flat / safe[:, None]
    cos = np.clip(unit @ unit.T, -1.0, 1.0)
    dist = (1.0 - cos).astype(np.float64)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)

    from sklearn.cluster import HDBSCAN

    labels = HDBSCAN(
        min_cluster_size=n // 2 + 1,
        min_samples=1,
        metric="precomputed",
        cluster_selection_method="eom",
        allow_single_cluster=True,
        copy=True,
    ).fit_predict(dist)
    admitted = [i for i in range(n) if labels[i] >= 0]
    if not admitted:
        return list(range(n))
    return admitted


def flame_noise_rng(seed: int, server_round: int) -> np.random.Generator:
    """The exact seeded Generator the fl_v2 oracle uses for FLAME's noise.

    ``default_rng((seed & 0xFFFFFFFF) * 1_000_003 + server_round)`` — kept here
    as the single source of truth so the noise sample is byte-reproducible and
    parity-checkable.
    """
    return np.random.default_rng((int(seed) & 0xFFFFFFFF) * 1_000_003 + int(server_round))


def flame_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    *,
    noise_multiplier: float,
    seed: int,
    server_round: int,
) -> DefenseDecision:
    """Full FLAME decision: cluster → clip → noise → aggregate.

    Mirrors the oracle's arithmetic exactly:
      * ``S_t = median(update_norms)``
      * admitted via :func:`flame_cluster_admitted`
      * ``coefs[i] = min(1, S_t/(norm_i+1e-12)) / |admitted|`` for admitted, else 0
      * ``sigma = noise_multiplier * S_t``; per-tensor Gaussian noise from
        :func:`flame_noise_rng` drawn in ``global_params`` order
      * ``new[k] = global[k] + sum_i coef_i (client_i[k]-global[k]) + noise[k]``
    """
    n = len(client_params_list)
    norms = np.asarray(
        compute_update_norms(global_params, client_params_list), dtype=np.float64
    )
    flat = build_flat_updates(global_params, client_params_list)
    admitted = flame_cluster_admitted(flat, norms)
    del flat
    s_t = float(np.median(norms))

    coefs = [0.0] * n
    for i in admitted:
        clip = min(1.0, s_t / (norms[i] + 1e-12))
        coefs[i] = clip / len(admitted)

    sigma = float(noise_multiplier) * s_t
    rng = flame_noise_rng(seed, server_round)
    noise = [rng.normal(0.0, sigma, size=np.asarray(g).shape) for g in global_params]

    new_global = aggregate_weighted_updates(
        global_params, client_params_list, coefs, noise=noise
    )
    return DefenseDecision(
        new_global=new_global,
        coefs=coefs,
        admitted=admitted,
        diagnostics={
            "s_t": s_t,
            "sigma": sigma,
            "n_admitted": int(len(admitted)),
            "n_total": int(n),
            "update_norms": [float(x) for x in norms],
        },
    )
