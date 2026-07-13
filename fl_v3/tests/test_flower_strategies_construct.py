"""The server constructs only the clean FedAvg Flower strategy."""
from __future__ import annotations

from fl_v3.server_app import _build_strategy
from fl_v3.strategy.flower_strategies import CleanFedAvgStrategy


_COMMON = {
    "fraction_train": 1.0,
    "fraction_evaluate": 0.0,
    "min_train_nodes": 2,
    "min_evaluate_nodes": 2,
    "min_available_nodes": 2,
}


def test_build_strategy_constructs_clean_fedavg(tmp_path):
    run_config = {
        "experiment-name": "clean",
        "seed": 42,
        "server-optimizer": "fedavg",
        "server-lr": 1.0,
    }
    strategy = _build_strategy(run_config, dict(_COMMON), str(tmp_path))
    assert isinstance(strategy, CleanFedAvgStrategy)
    assert strategy.seed == 42
    assert strategy.output_dir == str(tmp_path)
    assert strategy.server_optimizer.is_identity


def test_build_strategy_carries_clean_server_state():
    run_config = {
        "experiment-name": "clean-fedadam",
        "seed": 7,
        "server-optimizer": "fedadam",
        "server-lr": 0.01,
        "server-ema-decay": 0.9,
        "client-lr-schedule": "cosine",
        "learning-rate": 0.001,
        "num-server-rounds": 15,
        "client-lr-warmup-rounds": 2,
        "client-lr-final-frac": 0.05,
    }
    strategy = _build_strategy(run_config, dict(_COMMON), "/tmp/clean")
    assert strategy.server_optimizer.kind == "fedadam"
    assert strategy.server_ema_decay == 0.9
    assert strategy._lr_at_round(1) == 0.0005
