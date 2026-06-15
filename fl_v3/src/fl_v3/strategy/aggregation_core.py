"""Framework-free weighted-aggregation core (fl_v3 T0 carry-over).

The pure-numpy aggregation primitive the gradient-space defenses (FLAME,
FoolsGold) build on. Re-implemented from the fl_v2 oracle
(``aggregate_weighted_updates`` in ``fl_v2/src/fl_v2/strategy/norm_tracking_fedavg.py``)
and validated for byte-equivalence.

Keeping this independent of Flower types means a defense's full numerical
behaviour — admit/clip/reweight/noise → new global arrays — can be exercised on
a saved fixture WITHOUT Flower/Ray (login-node testable), which is exactly what
the T0 oracle-parity gate needs.

**Parity scope (deliberate design decision — read this).** fl_v3 routes EVERY
defense (incl. plain FedAvg and NormClip) through this ONE fp64 update-form core
(``new = global + Σ coef·(client − global)``, fp64, cast back). The fl_v2 oracle
matches this core BIT-FOR-BIT for FLAME and FoolsGold (they call the same
``aggregate_weighted_updates`` in fl_v2) — which is why the GATE-required FLAME +
FoolsGold parity is exact. BUT the oracle's *clean-FedAvg* and *NormClip* final
aggregation delegate to Flower's ``aggregate_arrayrecords`` (fp32, direct
weighted average). So fl_v3's clean-FedAvg / NormClip aggregation is algebraically
equivalent (Σcoef = 1 ⇒ weighted mean) but NOT bit-identical to the oracle's
fp32 path — it is intentionally higher precision and unified. Bit-parity is
therefore CLAIMED only for FLAME/FoolsGold; the clean/clip-path agreement with
Flower's fp32 weighting is a tolerance-level, T3 (real-Ray) check, not a T0
bit-identity claim. See collab/T0/SPEC.md §7.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


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
