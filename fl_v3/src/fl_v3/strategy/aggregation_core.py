"""Framework-free clean FedAvg aggregation.

The only project aggregation primitive is the fixed-order, num-example-weighted
FP32 average used by Flower 1.27. Keeping it independent of Flower types makes
the arithmetic directly testable while the Flower strategy owns deterministic
client ordering and message validation.
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

    This is the clean FedAvg arithmetic and is checked against the real Flower
    implementation in ``tests/test_flower_fp32_parity.py``.
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
