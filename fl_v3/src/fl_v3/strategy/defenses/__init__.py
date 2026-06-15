"""fl_v3 defense numerical cores (framework-free, oracle-parity-tested).

Each core is a pure function over update vectors returning a
:class:`DefenseDecision`. The Flower strategy wrappers in
``fl_v3.strategy.flower_strategies`` adapt Message/ArrayRecord I/O onto these.

Defense suite (D5 minimum set): FedAvg, NormClip, FLAME, FoolsGold, MultiKrum,
FedMedian. Each gets a written assumption card before any benchmark run (T6).
"""
from fl_v3.strategy.defenses.base import DefenseDecision, build_flat_updates
from fl_v3.strategy.defenses.fedavg import fedavg_decision, norm_clip_decision
from fl_v3.strategy.defenses.fed_median import fed_median_decision
from fl_v3.strategy.defenses.flame import (
    flame_cluster_admitted,
    flame_decision,
    flame_noise_rng,
)
from fl_v3.strategy.defenses.foolsgold import FoolsGoldState, foolsgold_weights
from fl_v3.strategy.defenses.multi_krum import (
    krum_scores,
    multi_krum_decision,
    multi_krum_valid,
)

__all__ = [
    "DefenseDecision",
    "build_flat_updates",
    "fedavg_decision",
    "norm_clip_decision",
    "fed_median_decision",
    "flame_decision",
    "flame_cluster_admitted",
    "flame_noise_rng",
    "FoolsGoldState",
    "foolsgold_weights",
    "multi_krum_decision",
    "multi_krum_valid",
    "krum_scores",
]
