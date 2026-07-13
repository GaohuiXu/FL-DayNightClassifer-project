"""In-process clean FedAvg smoke contracts."""
from __future__ import annotations

import inspect

from fl_v3.engine.local_runner import run_clean_round, run_clean_rounds


_CFG = {
    "task-type": "dummy_regression",
    "seed": 42,
    "device": "cpu",
    "num-clients": 4,
    "num-local-epochs": 1,
    "batch-size": 8,
    "learning-rate": 0.01,
    "num-workers": 0,
    "loss": "mse",
}


def test_clean_round_runs_and_is_deterministic():
    first = run_clean_round(dict(_CFG), server_round=1)
    second = run_clean_round(dict(_CFG), server_round=1)
    assert first["decision_valid"]
    assert first["eval"] is not None
    assert first["agg_checksum"] == second["agg_checksum"]
    assert first["aggregation"] == "fedavg"


def test_runner_api_exposes_only_clean_aggregation():
    assert "defense" not in inspect.signature(run_clean_round).parameters
    assert "defense" not in inspect.signature(run_clean_rounds).parameters
    assert not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for function in (run_clean_round, run_clean_rounds)
        for parameter in inspect.signature(function).parameters.values()
    )
