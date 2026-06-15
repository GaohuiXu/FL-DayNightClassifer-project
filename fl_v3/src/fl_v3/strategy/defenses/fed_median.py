"""FedMedian core (Yin et al. 2018) — coordinate-wise median aggregation.

Re-implemented from the fl_v2 oracle (``fl_v2/src/fl_v2/strategy/fed_median.py``)
and validated for byte-equivalence. FedMedian takes the coordinate-wise median
of the client *parameters* (not updates) — order-independent by value, but the
caller still sorts by partition-id so the logging path is stable.

The assumption card (T6) must check whether clean utility collapses under
non-IID AD detection (median can be a poor estimator when honest clients are
diverse). That is a scientific judgement; parity here is implementation-only.

Paper: https://arxiv.org/abs/1803.01498
"""
from __future__ import annotations

from typing import List

import numpy as np

from fl_v3.strategy.aggregation_core import coordinatewise_median
from fl_v3.strategy.defenses.base import DefenseDecision


def fed_median_decision(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
) -> DefenseDecision:
    """Coordinate-wise median of client parameters (value-based; coefs=None)."""
    new_global = coordinatewise_median(client_params_list)
    return DefenseDecision(
        new_global=new_global,
        coefs=None,
        diagnostics={"n_total": int(len(client_params_list))},
    )
