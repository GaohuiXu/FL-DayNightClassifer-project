"""FedAvg + NormClip cores (fl_v3 T0).

FedAvg is the clean+poisoned baseline; NormClip clips each client update to an
L2 bound before averaging (defense AND a baseline component for the FLAME
random-drop control).

**Oracle aggregation = Flower's fp32 path.** The fl_v2 oracle aggregates both
clean FedAvg and post-clip NormClip via Flower's ``FedAvg.aggregate_train`` →
``aggregate_arrayrecords`` (fp32 direct weighted average of client *params*). So
these cores aggregate through :func:`fp32_weighted_average` (the bit-for-bit
replica), NOT the fp64 update-form helper FLAME/FoolsGold use — restoring
bit-identity for the clean baseline (the null-config crown jewel). The
parity-critical clip primitive (``clip_updates_by_l2_norm``) lives in
``gradient_metrics`` and is fixture-tested against the fl_v2 oracle.
"""
from __future__ import annotations

from typing import List, Optional

from fl_v3.strategy.aggregation_core import fp32_weighted_average
from fl_v3.strategy.defenses.base import DefenseDecision
from fl_v3.strategy.gradient_metrics import clip_updates_by_l2_norm


def _weight_factors(n: int, weights: Optional[List[float]]) -> List[float]:
    """Per-client weight factors summing to 1 (uniform if ``weights`` is None)."""
    if weights is None:
        return [1.0 / n] * n
    total = float(sum(float(w) for w in weights))
    if total <= 0.0:
        return [1.0 / n] * n
    return [float(w) / total for w in weights]


def fedavg_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
) -> DefenseDecision:
    """Weighted FedAvg — Flower-identical fp32 weighted average of client params
    (weights default to uniform; the Flower path passes num-examples)."""
    new_global = fp32_weighted_average(client_params_list, weights)
    return DefenseDecision(
        new_global=new_global,
        coefs=_weight_factors(len(client_params_list), weights),
        diagnostics={},
    )


def norm_clip_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    clip_norm: float,
    weights: Optional[List[float]] = None,
) -> DefenseDecision:
    """Clip each client update to ``clip_norm`` (L2), then Flower-identical fp32
    weighted-average of the clipped client params (matches the oracle: clip →
    Flower FedAvg). Diagnostics carry pre/post norms for the assumption card.
    """
    clipped, original_norms, clipped_norms = clip_updates_by_l2_norm(
        global_params, client_params_list, clip_norm
    )
    new_global = fp32_weighted_average(clipped, weights)
    return DefenseDecision(
        new_global=new_global,
        coefs=_weight_factors(len(clipped), weights),
        diagnostics={
            "clip_norm": float(clip_norm),
            "original_norms": [float(x) for x in original_norms],
            "clipped_norms": [float(x) for x in clipped_norms],
        },
    )
