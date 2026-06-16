"""Multi-Krum core (Blanchard et al., NeurIPS 2017) — Byzantine-robust selection.

fl_v2 used Flower's built-in ``MultiKrum``. fl_v3 re-implements that **same
one-shot variant** (NOT the paper's iterative m-Krum): Flower's
``select_multikrum`` (``flwr/serverapp/strategy/multikrum.py``) computes Krum
scores ONCE and takes ``np.argsort(scores)[:m]`` — exactly the selection logic
here.

**Flower-COMPATIBLE, NOT a bit-identical carry-over (deliberate, and required for
AD).** fl_v3 computes pairwise distances with a numerically stable direct
``||x-y||^2`` instead of Flower's textbook ``||x||^2 + ||y||^2 - 2 x·y``. At
AD scale (large params ~O(10^2) norm, tiny inter-client update differences)
Flower's form suffers catastrophic float32 cancellation: measured **~10 %
relative error** on a 50k-dim large-param case vs ~2e-11 for the direct form. So
matching Flower's exact bytes would make the Byzantine selection ~10 % wrong on
the actual fusion models. Consequently:
  * the **selection** matches Flower for well-separated configs (verified on the
    fixture, ``tests/test_flower_fp32_parity.py``); on near-ties it MAY differ —
    an intentional improvement, not a parity bug;
  * the **selected-subset average** uses Flower's fp32 weighting
    (``fp32_weighted_average``) but in index (not Flower's score) order, so it
    matches Flower only within fp32 noise.
Bit-identity to Flower is therefore claimed for **FedAvg / NormClip** (which do
not use distances), NOT for MultiKrum. (Like the strict-determinism default, this
is a documented, intentional divergence from the fl_v2/Flower oracle.)

Algorithm (per round, n participants, ``f`` assumed Byzantine):
  * pairwise squared-L2 distances between client update vectors
    (= distances between params, since the shared global cancels);
  * each client's Krum score = sum of its ``n - f - 2`` smallest distances to
    OTHER clients;
  * Multi-Krum selects the ``m`` clients with the smallest scores and averages
    them (plain Krum is ``m = 1``). The average uses Flower's fp32
    ``aggregate_arrayrecords`` weighting (num-examples), via
    :func:`fp32_weighted_average`.

**Validity (the assumption card requirement, STRICTER than Flower).** Flower
clamps ``num_closest = max(1, n-f-2)`` and forces a result for any ``(n,f,m)``.
The Cycle-04 defense card instead requires marking a config NA when Krum's
assumptions are violated. fl_v3 enforces: ``f >= 0``; Krum Byzantine resilience
``n >= 2f + 3``; and the selection bounded by the trusted-neighbour count
``1 <= m <= n - f - 2`` (so we never select more vectors than the Krum structure
supports — e.g. ``n=5, f=1, m=3`` is INVALID because ``n-f-2 = 2 < 3``). On
violation the core returns ``valid=False`` (cell NA, not forced). ``f`` is the
defender's *assumed* malicious count, independent of the *actual* count.

Determinism: stable argsort + the caller's partition-id pre-sort make ties
reproducible (neighbour sums are tie-safe).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from fl_v3.strategy.aggregation_core import fp32_weighted_average
from fl_v3.strategy.defenses.base import DefenseDecision, build_flat_updates


def multi_krum_valid(n: int, f: int, num_to_select: int) -> bool:
    """Multi-Krum validity: ``f >= 0``, ``n >= 2f + 3``, ``1 <= m <= n - f - 2``.

    The ``m <= n - f - 2`` bound (NOT present in Flower, which clamps and forces)
    accounts for the selection count: you cannot select more vectors than the
    ``n - f - 2`` trusted neighbours the Krum score is built on. Reduces to
    standard Krum (``n >= 2f + 3``) at ``m = 1``.
    """
    if f < 0 or num_to_select < 1:
        return False
    if n < 2 * f + 3:
        return False
    return num_to_select <= n - f - 2


def krum_scores(flat: np.ndarray, f: int) -> np.ndarray:
    """Per-client Krum score: sum of the ``n - f - 2`` smallest squared
    distances to other clients. ``flat`` is the (n, d) update-vector stack.
    """
    n = int(flat.shape[0])
    num_closest = n - f - 2
    if num_closest < 1:
        # Krum score is only defined for n - f - 2 >= 1. The production path is
        # gated by multi_krum_valid; guard here so a direct call to this
        # exported core fails loudly instead of returning garbage scores.
        raise ValueError(
            f"krum_scores needs n - f - 2 >= 1 (got n={n}, f={f} -> {num_closest}); "
            "pre-validate with multi_krum_valid."
        )
    sq = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = flat - flat[i]
        sq[i] = np.einsum("ij,ij->i", diff.astype(np.float64), diff.astype(np.float64))
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(sq[i], i)  # exclude self
        nearest = np.sort(others, kind="stable")[:num_closest]
        scores[i] = float(nearest.sum())
    return scores


def multi_krum_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    *,
    num_malicious: int,
    num_to_select: int,
    weights: Optional[List[float]] = None,
) -> DefenseDecision:
    """Multi-Krum selection + Flower-identical fp32 weighted average of the
    selected clients. Returns ``valid=False`` (``new_global=None``) if
    ``(n, f, m)`` violate :func:`multi_krum_valid` — the cell is NA, not forced.
    """
    n = len(client_params_list)
    f = int(num_malicious)
    m = int(num_to_select)
    if not multi_krum_valid(n, f, m):
        return DefenseDecision(
            new_global=None,
            coefs=None,
            selected=None,
            valid=False,
            diagnostics={
                "n_total": int(n),
                "assumed_malicious_f": f,
                "num_to_select": m,
                "reason": "MultiKrum validity violated (need n>=2f+3, 1<=m<=n-f-2)",
            },
        )

    flat = build_flat_updates(global_params, client_params_list)
    scores = krum_scores(flat, f)
    order = np.argsort(scores, kind="stable")  # ties break by partition-id-sorted index
    selected = sorted(int(i) for i in order[:m])

    sel_params = [client_params_list[i] for i in selected]
    sel_weights = None if weights is None else [float(weights[i]) for i in selected]
    new_global = fp32_weighted_average(sel_params, sel_weights)

    return DefenseDecision(
        new_global=new_global,
        coefs=None,  # selection-based, not a soft reweight
        selected=selected,
        valid=True,
        diagnostics={
            "n_total": int(n),
            "assumed_malicious_f": f,
            "num_to_select": m,
            "num_closest": int(n - f - 2),
            "scores": [float(s) for s in scores],
            "selected": selected,
        },
    )
