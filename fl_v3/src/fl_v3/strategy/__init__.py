"""fl_v3 aggregation strategies.

Two layers:
  * **Cores** (``fl_v3.strategy.defenses``) — framework-free numerical decisions,
    validated against the fl_v2 oracle on saved fixtures.
  * **Flower wrappers** (``fl_v3.strategy.flower_strategies``) — adapt the cores
    onto Flower's Message/ArrayRecord API for the real Ray runs.

``gradient_metrics`` / ``aggregation_core`` hold the shared pure-numpy
primitives. ``flower_strategies`` is imported lazily (it needs ``flwr``) so the
cores stay importable in a flwr-free context (e.g. pure-numpy fixture tests).
"""
from fl_v3.strategy import aggregation_core, defenses, gradient_metrics

__all__ = ["aggregation_core", "defenses", "gradient_metrics"]


def __getattr__(name):  # lazy: avoid importing flwr unless the wrappers are used
    if name in {
        "DEFENSE_REGISTRY",
        "NormTrackingFedAvg",
        "NormClipStrategy",
        "FlameStrategy",
        "FoolsGoldStrategy",
        "FedMedianStrategy",
        "MultiKrumStrategy",
        "partition_sort_key",
        "drop_nonfinite_replies",
    }:
        from fl_v3.strategy import flower_strategies

        return getattr(flower_strategies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
