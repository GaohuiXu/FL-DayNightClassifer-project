"""FedAvg + NormClip cores (fl_v3 T0).

FedAvg is the clean+poisoned baseline; NormClip clips each client update to an
L2 bound before averaging (defense AND a baseline component for the FLAME
random-drop control). Both are pure-numpy; the parity-critical primitive
(``clip_updates_by_l2_norm``) lives in ``gradient_metrics`` and is fixture-tested
against the fl_v2 oracle directly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from fl_v3.strategy.aggregation_core import aggregate_weighted_updates
from fl_v3.strategy.defenses.base import DefenseDecision
from fl_v3.strategy.gradient_metrics import (
    clip_updates_by_l2_norm,
    compute_update_norms,
)


def _normalized_coefs(n: int, weights: Optional[List[float]]) -> List[float]:
    """Per-client coefs summing to 1 (uniform if ``weights`` is None)."""
    if weights is None:
        return [1.0 / n] * n
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 0.0:
        return [1.0 / n] * n
    return [float(x) for x in (w / total)]


def fedavg_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
) -> DefenseDecision:
    """Weighted FedAvg (weights default to uniform; Flower passes num-examples).

    ``new = global + sum_i coef_i (client_i - global)`` with ``sum coef_i = 1``
    is exactly the weighted average of client params.
    """
    n = len(client_params_list)
    coefs = _normalized_coefs(n, weights)
    new_global = aggregate_weighted_updates(global_params, client_params_list, coefs)
    return DefenseDecision(new_global=new_global, coefs=coefs, diagnostics={})


def norm_clip_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    clip_norm: float,
    weights: Optional[List[float]] = None,
) -> DefenseDecision:
    """Clip each client update to ``clip_norm`` (L2), then weighted-average.

    The clip uses :func:`clip_updates_by_l2_norm` (oracle-parity-tested). The
    diagnostics carry pre/post norms for the clip-radius logging the assumption
    card requires.
    """
    clipped, original_norms, clipped_norms = clip_updates_by_l2_norm(
        global_params, client_params_list, clip_norm
    )
    n = len(clipped)
    coefs = _normalized_coefs(n, weights)
    new_global = aggregate_weighted_updates(global_params, clipped, coefs)
    return DefenseDecision(
        new_global=new_global,
        coefs=coefs,
        diagnostics={
            "clip_norm": float(clip_norm),
            "original_norms": [float(x) for x in original_norms],
            "clipped_norms": [float(x) for x in clipped_norms],
        },
    )
