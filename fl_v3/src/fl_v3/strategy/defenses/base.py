"""Shared types + helpers for the fl_v3 defense numerical cores (T0).

Each defense in fl_v3 is a *framework-free numerical core*: a pure function over
``(global_params, client_params_list)`` that returns a :class:`DefenseDecision`
(the aggregated global params + the per-round decision diagnostics). The Flower
strategy wrappers (``flower_strategies.py``) only adapt Message/ArrayRecord I/O
onto these cores.

This split is what makes the T0 oracle-parity gate cheap and honest: a core's
full numerical behaviour (admit / clip / reweight / noise → new global arrays)
is exercised against a saved fl_v2 fixture WITHOUT Flower or Ray.

**Oracle-parity scope (read this).** Parity against fl_v2 certifies
*implementation equivalence only* — same update vectors in → same decision out.
It does NOT certify AD-domain validity (update vectorization for a fusion model,
module slicing, detection/ASR semantics). Those are T1/T4/T6 concerns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DefenseDecision:
    """One round's defense decision (framework-free).

    Attributes:
        new_global:   aggregated global parameters (list of arrays, in
                      ``global_params`` order). ``None`` only on a degenerate
                      empty round.
        coefs:        per-client effective weight used by weighted aggregators
                      (FedAvg / NormClip / FLAME / FoolsGold); ``None`` for
                      value-based aggregators (FedMedian) and selection-based
                      ones (MultiKrum, which exposes ``selected`` instead).
        admitted:     client indices the defense kept (filter defenses: FLAME);
                      ``None`` when the defense does not filter.
        selected:     client indices the defense selected (MultiKrum); ``None``
                      otherwise.
        valid:        ``False`` marks an invalid configuration the defense
                      refuses to force a result for (e.g. MultiKrum (n,f)
                      violation) — the harness records the cell as NA.
        diagnostics:  defense-specific scalars/arrays for logging (s_t, sigma,
                      trust weights, krum scores, ...).
    """

    new_global: Optional[List[np.ndarray]]
    coefs: Optional[List[float]] = None
    admitted: Optional[List[int]] = None
    selected: Optional[List[int]] = None
    valid: bool = True
    diagnostics: dict = field(default_factory=dict)


def build_flat_updates(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
) -> np.ndarray:
    """(n, d) float32 stack of flattened client updates (client - global).

    Byte-identical to the fl_v2 FLAME ``flat`` construction: each row is the
    per-tensor float32 delta, concatenated in ``global_params`` order. Used by
    cosine/clustering defenses.
    """
    return np.stack(
        [
            np.concatenate(
                [
                    (
                        np.asarray(c, dtype=np.float32)
                        - np.asarray(g, dtype=np.float32)
                    ).reshape(-1)
                    for c, g in zip(cp, global_params)
                ]
            )
            for cp in client_params_list
        ]
    )
