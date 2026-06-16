"""Framework-free weighted-aggregation core (fl_v3 T0 carry-over).

The pure-numpy aggregation primitive the gradient-space defenses (FLAME,
FoolsGold) build on. Re-implemented from the fl_v2 oracle
(``aggregate_weighted_updates`` in ``fl_v2/src/fl_v2/strategy/norm_tracking_fedavg.py``)
and validated for byte-equivalence.

Keeping this independent of Flower types means a defense's full numerical
behaviour — admit/clip/reweight/noise → new global arrays — can be exercised on
a saved fixture WITHOUT Flower/Ray (login-node testable), which is exactly what
the T0 oracle-parity gate needs.

**Two aggregation primitives, each matching the oracle path it stands in for
(T0 review fix — bit-identity restored for ALL defenses):**

  * ``aggregate_weighted_updates`` (fp64 update-form, optional noise) — used by
    **FLAME + FoolsGold**, which the fl_v2 oracle ALSO aggregates via this exact
    fp64 helper (``fl_v2/strategy/norm_tracking_fedavg.aggregate_weighted_updates``).
    Bit-identical there.
  * ``fp32_weighted_average`` — a bit-for-bit replica of Flower's
    ``aggregate_arrayrecords`` (fp32 direct weighted average of the client
    *params*). **FedAvg (none) / NormClip** are bit-identical to Flower here
    (verified against REAL Flower in ``tests/test_flower_fp32_parity.py``),
    restoring the clean-baseline / null-config crown jewel. **MultiKrum** also
    uses this fp32 weighting on its selected subset, but is only Flower-
    *compatible*, NOT bit-identical: its selection uses a more numerically stable
    distance than Flower (necessary for AD — see ``defenses/multi_krum.py``) and
    aggregates in index (not Flower's score) order.

The split is deliberate: FLAME/FoolsGold use the oracle's fp64 path (bit-exact);
FedAvg/NormClip use the oracle's fp32 path (bit-exact); MultiKrum is Flower-
compatible by intent.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def fp32_weighted_average(
    client_params_list: List[List[np.ndarray]],
    weights: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """Bit-for-bit replica of Flower's ``aggregate_arrayrecords`` (fp32).

    Per parameter tensor ``k``: ``agg = Σ_i (client_i[k].float32 * (w_i/Σw))``,
    accumulated IN float32 in client order — exactly Flower's
    ``aggregated[key] = value.numpy() * weight`` then ``+= value.numpy() * weight``
    (``flwr/serverapp/strategy/strategy_utils.py:88-105``). Weight factors are
    Python-float (fp64) ``w_i / Σw``; the array×scalar stays float32 (numpy
    value-based casting). ``weights`` defaults to uniform.

    This is the oracle aggregation for FedAvg (none) / NormClip (post-clip),
    which are bit-identical to Flower (verified in
    ``tests/test_flower_fp32_parity.py``). MultiKrum reuses it on its selected
    subset (Flower-*compatible* weighting; bit-identity not claimed there).
    """
    n = len(client_params_list)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0] * n
    total = sum(float(w) for w in weights)
    weight_factors = [float(w) / total for w in weights]

    num_tensors = len(client_params_list[0])
    out: List[np.ndarray] = []
    for k in range(num_tensors):
        agg = None
        for cp, wf in zip(client_params_list, weight_factors):
            term = np.asarray(cp[k]).astype(np.float32, copy=False) * wf
            agg = term.copy() if agg is None else agg + term
        out.append(np.asarray(agg))
    return out


def aggregate_weighted_updates(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    coefs: List[float],
    noise: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """Aggregate client updates with explicit per-client weights.

    Computes, per parameter tensor ``k``::

        new[k] = global[k] + sum_i coefs[i] * (client_i[k] - global[k])

    optionally plus a per-tensor ``noise[k]`` term (FLAME). ``coefs`` are the
    FINAL per-client weights — the caller owns any normalisation (FoolsGold
    normalises to sum 1; FLAME uses clip-scale / |admitted|). fp64 accumulation;
    the result is cast back to each tensor's dtype. Deterministic for a fixed,
    caller-ordered input.

    Returns a list of numpy arrays (one per parameter tensor), in the same
    order as ``global_params`` — the framework-free analog of the oracle's
    ``ArrayRecord`` return.
    """
    out: List[np.ndarray] = []
    for idx in range(len(global_params)):
        g64 = np.asarray(global_params[idx], dtype=np.float64)
        acc = np.zeros_like(g64)
        for coef, cp in zip(coefs, client_params_list):
            if coef == 0.0:
                continue
            acc += float(coef) * (np.asarray(cp[idx], dtype=np.float64) - g64)
        new_k = g64 + acc
        if noise is not None:
            new_k = new_k + np.asarray(noise[idx], dtype=np.float64)
        out.append(np.asarray(new_k, dtype=np.asarray(global_params[idx]).dtype))
    return out


def coordinatewise_median(
    client_params_list: List[List[np.ndarray]],
) -> List[np.ndarray]:
    """Coordinate-wise median of client *parameters* (FedMedian core).

    Per parameter tensor ``k``, stack the clients and take ``np.median`` along
    axis 0, cast back to the first client's dtype — identical to the fl_v2
    oracle's FedMedian aggregation. NOTE: FedMedian medians the client
    parameters directly (not the updates), matching the oracle.
    """
    if not client_params_list:
        return []
    num_tensors = len(client_params_list[0])
    out: List[np.ndarray] = []
    for k in range(num_tensors):
        layers = [np.asarray(cp[k]) for cp in client_params_list]
        stacked = np.stack(layers)
        median = np.asarray(np.median(stacked, axis=0), dtype=layers[0].dtype)
        out.append(median)
    return out
