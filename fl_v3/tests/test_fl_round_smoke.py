"""In-process FL round runs cleanly under every defense in the suite (T0).

This exercises each defense core through the same runner the platform uses, on
the dummy task. It is a code-path smoke (login-node), not a scientific claim.
"""
from __future__ import annotations

import pytest

from fl_v3.engine.local_runner import run_clean_round

_CFG = {
    "task-type": "dummy_regression",
    "seed": 42,
    "device": "cpu",
    "num-clients": 8,  # >= 4 so FLAME clustering / Krum validity are meaningful
    "num-local-epochs": 1,
    "batch-size": 8,
    "learning-rate": 0.01,
    "num-workers": 0,
    "loss": "mse",
    "clip-norm": 5.0,
    "flame-noise-multiplier": 1e-6,
    "num-malicious-nodes": 1,
    "krum-num-to-select": 5,
}


@pytest.mark.parametrize(
    "defense", ["none", "norm_clipping", "flame", "foolsgold", "fed_median", "multi_krum"]
)
def test_round_runs_for_each_defense(defense):
    out = run_clean_round(dict(_CFG), defense=defense, server_round=1)
    assert out["decision_valid"], f"{defense} returned an invalid decision unexpectedly"
    assert out["eval"] is not None
    assert out["agg_checksum"] is not None
    assert len(out["client_train_losses"]) == _CFG["num-clients"]


@pytest.mark.parametrize(
    "defense", ["none", "norm_clipping", "flame", "foolsgold", "fed_median", "multi_krum"]
)
def test_round_is_deterministic_for_each_defense(defense):
    a = run_clean_round(dict(_CFG), defense=defense, server_round=1)
    b = run_clean_round(dict(_CFG), defense=defense, server_round=1)
    assert a["agg_checksum"] == b["agg_checksum"], f"{defense} not bit-deterministic"


def test_multikrum_invalid_config_marks_NA():
    # n=8, f=4 → 8 < 2*4+3=11 → invalid → NA, not forced.
    out = run_clean_round(dict(_CFG, **{"num-malicious-nodes": 4}), defense="multi_krum", server_round=1)
    assert out["decision_valid"] is False
    assert out["agg_checksum"] is None
