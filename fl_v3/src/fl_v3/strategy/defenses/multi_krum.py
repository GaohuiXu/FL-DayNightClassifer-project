"""Multi-Krum core (Blanchard et al., NeurIPS 2017) — Byzantine-robust selection.

fl_v2 used Flower's *built-in* ``MultiKrum`` (wrapped only for metric capture),
so there is no fl_v2 source to be the parity oracle here. fl_v3 re-implements the
textbook algorithm as a clean pure-numpy core and validates it on a hand-derived
fixture (small n where the scores/selection are computable by hand). The two
GATE-required oracle-parity defenses are FLAME + FoolsGold (which DO have fl_v2
source); MultiKrum's correctness is established by the textbook fixture.

Algorithm (per round, n participants, ``f`` assumed Byzantine):
  * pairwise squared-L2 distances between client update vectors;
  * each client's Krum score = sum of its ``n - f - 2`` smallest distances to
    OTHER clients;
  * Multi-Krum selects the ``m`` clients with the smallest scores and averages
    them (plain Krum is ``m = 1``).

**Validity (the assumption card requirement).** Krum's robustness guarantee
needs ``n >= 2f + 3`` and a sensible selection count ``1 <= m <= n``. If the
``(n_r, f_r)`` actually present this round violate that, the core returns
``valid=False`` and the harness records the cell as NA — it does NOT force a
result. ``f_r`` (the defender's *assumed* malicious count among participants) is
independent of the *actual* malicious count ``m_r`` (ground truth); both are
reported, never conflated.

Determinism: the score ranking uses a stable argsort and the caller has already
sorted clients by partition-id, so tie order is reproducible. (Neighbour sums
are tie-safe: the sum of the k smallest distances is invariant to which tied
element fills the boundary.)
"""
from __future__ import annotations

from typing import List

import numpy as np

from fl_v3.strategy.aggregation_core import aggregate_weighted_updates
from fl_v3.strategy.defenses.base import DefenseDecision, build_flat_updates


def multi_krum_valid(n: int, f: int, num_to_select: int) -> bool:
    """Krum validity: ``n >= 2f + 3``, ``f >= 0``, ``1 <= m <= n``."""
    if f < 0 or num_to_select < 1 or num_to_select > n:
        return False
    return n >= 2 * f + 3


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
    # Pairwise squared L2 distances. ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b,
    # but compute directly for numerical clarity at this scale.
    sq = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        diff = flat - flat[i]
        sq[i] = np.einsum("ij,ij->i", diff.astype(np.float64), diff.astype(np.float64))
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(sq[i], i)  # exclude self
        # Sum of the num_closest smallest distances (tie-safe: equal values).
        nearest = np.sort(others, kind="stable")[:num_closest]
        scores[i] = float(nearest.sum())
    return scores


def multi_krum_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    *,
    num_malicious: int,
    num_to_select: int,
) -> DefenseDecision:
    """Multi-Krum selection + average of the selected clients.

    Returns ``valid=False`` (and ``new_global=None``) if ``(n, f, m)`` violate
    Krum validity — the cell is NA, not forced.
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
                "reason": "MultiKrum validity violated (need n >= 2f+3, 1<=m<=n)",
            },
        )

    flat = build_flat_updates(global_params, client_params_list)
    scores = krum_scores(flat, f)
    # Smallest m scores; stable so ties break by (partition-id-sorted) index.
    order = np.argsort(scores, kind="stable")
    selected = sorted(int(i) for i in order[:m])

    # Average the selected clients (uniform) — FedAvg over the survivors.
    coefs = [0.0] * n
    for i in selected:
        coefs[i] = 1.0 / len(selected)
    new_global = aggregate_weighted_updates(global_params, client_params_list, coefs)

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
