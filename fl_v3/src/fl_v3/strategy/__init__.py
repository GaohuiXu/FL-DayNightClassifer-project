"""Clean FedAvg aggregation primitives and Flower wrapper."""
from fl_v3.strategy import aggregation_core

__all__ = ["aggregation_core"]


def __getattr__(name):  # lazy: avoid importing flwr unless the wrappers are used
    if name in {
        "CleanFedAvgStrategy",
        "partition_sort_key",
        "drop_nonfinite_replies",
    }:
        from fl_v3.strategy import flower_strategies

        return getattr(flower_strategies, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
