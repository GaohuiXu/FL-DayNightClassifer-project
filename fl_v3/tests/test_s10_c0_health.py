from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _c0_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s10_stop_c0_health.py"
    spec = importlib.util.spec_from_file_location("s10_stop_c0_health", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c0_required_lidar_prefixes_match_trainable_second_075_path():
    module = _c0_module()
    for mode in ("lidar_only", "fusion"):
        prefixes = module._required_prefixes(mode)
        assert "lidar_encoder.backbone.conv_out" in prefixes
        assert "lidar_encoder.to_bev" not in prefixes


def test_c0_short_horizon_does_not_peek_or_require_epoch_exhaustion():
    module = _c0_module()
    iterator = iter(range(4))
    assert next(iterator) == 0
    assert next(iterator) == 1

    module._assert_expected_epoch_consumption(
        iterator, attempted_windows=2, epoch_windows=4,
    )
    assert next(iterator) == 2


def test_c0_full_epoch_requires_exact_exhaustion_and_rejects_overshoot():
    module = _c0_module()
    exhausted = iter(range(2))
    assert list(exhausted) == [0, 1]
    module._assert_expected_epoch_consumption(
        exhausted, attempted_windows=2, epoch_windows=2,
    )

    not_exhausted = iter(range(3))
    assert next(not_exhausted) == 0
    assert next(not_exhausted) == 1
    with pytest.raises(RuntimeError, match="did not consume"):
        module._assert_expected_epoch_consumption(
            not_exhausted, attempted_windows=2, epoch_windows=2,
        )

    with pytest.raises(RuntimeError, match="exceeds"):
        module._assert_expected_epoch_consumption(
            iter(()), attempted_windows=5, epoch_windows=4,
        )
