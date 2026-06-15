"""FoolsGold core (Fung et al. 2020) — similarity-based reweighting.

Re-implemented from the fl_v2 oracle (``fl_v2/src/fl_v2/strategy/foolsgold.py``)
and validated for byte-equivalence on a saved fixture.

FoolsGold down-weights colluding (sybil) clients: clients sharing an attack
objective send updates that, accumulated over rounds, point in a consistent
shared direction, whereas honest non-IID clients point in diverse directions. It
maintains a per-client cumulative-update history (over the "indicative-feature"
slice — the classifier head in the paper), measures pairwise cosine, and
down-weights highly-similar histories.

**Indicative-feature slice.** The paper restricts the cosine to the final
classifier head. fl_v2 used ``HEAD_IDX=-2`` (``fc.weight`` for ResNet18). fl_v3
makes the slice a parameter (``head_index``, default ``-2`` for oracle parity);
the AD config picks the right indicative slice for the fusion head in T6. State
is pure deterministic accumulation → bit-reproducible.

Paper: https://arxiv.org/abs/1808.04866
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from fl_v3.strategy.aggregation_core import aggregate_weighted_updates
from fl_v3.strategy.defenses.base import DefenseDecision


def foolsgold_weights(histories: List[np.ndarray]) -> List[float]:
    """FoolsGold trust weights from cumulative-update histories.

    Returns one weight per client in [0, 1]: ~1 for a unique direction (honest),
    ~0 for a client highly similar to another (suspected colluder). Byte-identical
    to the fl_v2 oracle, including: zero-history safe norm; clip negative
    similarity to 0; the pardoning rescale (don't penalise a client for being
    similar to a MORE suspicious one); ``wv = 1 - max_cs``; normalise by max;
    the logit confidence rescale ``log(wv/(1-wv)) + 0.5`` clipped to [0,1];
    uniform fallback on a degenerate all-zero vector.
    """
    n = len(histories)
    if n == 1:
        return [1.0]
    H = np.stack(histories).astype(np.float64)
    norms = np.linalg.norm(H, axis=1)
    safe = np.where(norms < 1e-12, 1.0, norms)
    Hn = H / safe[:, None]
    cs = Hn @ Hn.T
    np.fill_diagonal(cs, 0.0)
    cs = np.clip(cs, 0.0, None)
    max_cs = cs.max(axis=1)
    # Pardoning: don't penalise client i for being similar to a MORE suspicious j.
    for i in range(n):
        for j in range(n):
            if i != j and max_cs[i] > max_cs[j] and max_cs[i] > 1e-12:
                cs[i, j] *= max_cs[j] / max_cs[i]
    wv = 1.0 - cs.max(axis=1)
    wv = np.clip(wv, 0.0, 1.0)
    mx = wv.max()
    if mx > 1e-12:
        wv = wv / mx
    # Logit confidence rescaling: sharpen toward {0, 1}.
    wv = np.clip(wv, 1e-6, 1.0 - 1e-6)
    wv = np.log(wv / (1.0 - wv)) + 0.5
    wv = np.clip(wv, 0.0, 1.0)
    if wv.sum() < 1e-12:
        wv = np.ones(n)
    return [float(x) for x in wv]


class FoolsGoldState:
    """Stateful FoolsGold accumulator (per-partition-id cumulative history).

    The strategy wrapper holds one of these for the run. Pure accumulation, no
    RNG → deterministic across re-runs. Matches the oracle's
    ``self._histories: dict[int, np.ndarray]``.
    """

    def __init__(self) -> None:
        self._histories: Dict[int, np.ndarray] = {}

    def update_and_decide(
        self,
        global_params: List[np.ndarray],
        client_params_list: List[List[np.ndarray]],
        partition_ids: List[int],
        head_index: int = -2,
    ) -> DefenseDecision:
        """Accumulate this round's head-slice updates, then reweight + aggregate.

        Mirrors the oracle: per-client head-slice delta (float32, flattened) is
        added to that partition-id's history; trust weights come from
        :func:`foolsgold_weights`; coefs normalise trust to sum 1 (uniform
        fallback if the trust sums to ~0); aggregation uses the FULL-model
        updates weighted by those coefs.
        """
        updates = [
            (
                np.asarray(cp[head_index], dtype=np.float32)
                - np.asarray(global_params[head_index], dtype=np.float32)
            ).reshape(-1)
            for cp in client_params_list
        ]
        for pid, upd in zip(partition_ids, updates):
            if pid in self._histories:
                self._histories[pid] = self._histories[pid] + upd
            else:
                self._histories[pid] = upd.copy()

        trust = foolsgold_weights([self._histories[p] for p in partition_ids])
        total = float(sum(trust))
        if total < 1e-12:
            coefs = [1.0 / len(trust)] * len(trust)
        else:
            coefs = [t / total for t in trust]

        new_global = aggregate_weighted_updates(
            global_params, client_params_list, coefs
        )
        return DefenseDecision(
            new_global=new_global,
            coefs=coefs,
            diagnostics={
                "trust": [float(t) for t in trust],
                "trust_min": float(min(trust)),
                "trust_max": float(max(trust)),
                "trust_mean": float(np.mean(trust)),
            },
        )
