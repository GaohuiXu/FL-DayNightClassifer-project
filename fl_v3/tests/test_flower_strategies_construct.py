"""Flower strategy wrappers construct + route to the right core (T0 import/build).

These do NOT run a live Ray round (that's T3 via SLURM) — they assert the
``defense-type`` → strategy mapping instantiates against Flower's FedAvg base and
that each wrapper carries the expected defense knobs. The numerical behaviour is
covered by the core parity tests + the in-process runner.
"""
from __future__ import annotations

import pytest

from fl_v3.server_app import _build_strategy
from fl_v3.strategy.flower_strategies import (
    FedMedianStrategy,
    FlameStrategy,
    FoolsGoldStrategy,
    MultiKrumStrategy,
    NormClipStrategy,
    NormTrackingFedAvg,
)

_COMMON = dict(
    fraction_train=1.0,
    fraction_evaluate=1.0,
    min_train_nodes=2,
    min_evaluate_nodes=2,
    min_available_nodes=2,
)
_RUN_CONFIG = {
    "experiment-name": "t0",
    "seed": 42,
    "topk-energy-k": 16,
    "clip-norm": 5.0,
    "flame-noise-multiplier": 1e-6,
    "foolsgold-head-index": -2,
    "num-malicious-nodes": 1,
    "krum-num-to-select": 3,
}


@pytest.mark.parametrize(
    "defense,cls",
    [
        ("none", NormTrackingFedAvg),
        ("fedavg", NormTrackingFedAvg),
        ("norm_clipping", NormClipStrategy),
        ("flame", FlameStrategy),
        ("foolsgold", FoolsGoldStrategy),
        ("fed_median", FedMedianStrategy),
        ("multi_krum", MultiKrumStrategy),
    ],
)
def test_build_strategy_constructs(defense, cls, tmp_path):
    strat = _build_strategy(defense, _RUN_CONFIG, dict(_COMMON), str(tmp_path))
    assert isinstance(strat, cls)
    assert strat.seed == 42
    assert strat.output_dir == str(tmp_path)


def test_strategy_knobs_carried():
    nc = _build_strategy("norm_clipping", _RUN_CONFIG, dict(_COMMON), "/tmp/x")
    assert nc.clip_norm == 5.0
    fl = _build_strategy("flame", _RUN_CONFIG, dict(_COMMON), "/tmp/x")
    assert fl.noise_multiplier == 1e-6
    mk = _build_strategy("multi_krum", _RUN_CONFIG, dict(_COMMON), "/tmp/x")
    assert mk.num_malicious == 1 and mk.num_to_select == 3
    fg = _build_strategy("foolsgold", _RUN_CONFIG, dict(_COMMON), "/tmp/x")
    assert fg.head_index == -2


def test_unknown_defense_raises():
    with pytest.raises(ValueError):
        _build_strategy("not_a_defense", _RUN_CONFIG, dict(_COMMON), "/tmp/x")
