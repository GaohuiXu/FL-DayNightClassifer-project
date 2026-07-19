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
    assert module.SCHEMA == "fl_v3.s10.stop_c0_health.v2"
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


def test_c0_fail_fast_numeric_gate_requires_one_clean_update_per_window():
    module = _c0_module()
    clean = {
        "attempted_windows": 4,
        "optimizer_step": 4,
        "invalid_windows": 0,
        "nonfinite_windows": 0,
        "overflow_windows": 0,
        "discarded_windows": 0,
    }
    assert module._numerical_gate_failed(clean) is False
    for field in (
        "invalid_windows", "nonfinite_windows", "overflow_windows",
        "discarded_windows",
    ):
        failed = dict(clean)
        failed[field] = 1
        assert module._numerical_gate_failed(failed) is True
    missing_update = dict(clean)
    missing_update["optimizer_step"] = 3
    assert module._numerical_gate_failed(missing_update) is True


def test_c0_training_token_evidence_uses_observed_batches_and_exact_remainder():
    module = _c0_module()
    observed = []
    chunk = module._ExactChunk(
        iter((
            {"sample_token": ["b", "a"]},
            {"sample_token": ["d", "c"]},
        )),
        2,
        observed_sample_tokens=observed,
    )
    assert list(chunk) == [
        {"sample_token": ["b", "a"]},
        {"sample_token": ["d", "c"]},
    ]
    assert observed == ["b", "a", "d", "c"]

    evidence = module._training_token_evidence(
        ("a", "b", "c", "d", "e"),
        observed,
        attempted_windows=2,
        batch_size=2,
        full_epoch=True,
    )
    assert evidence["source"] == "actual_collated_batches"
    assert evidence["consumed_sample_count"] == 4
    assert evidence["drop_last_remainder_count"] == 1
    assert evidence["drop_last_remainder_tokens_sorted"] == ["e"]


def test_c0_short_horizon_token_evidence_has_no_drop_last_claim():
    module = _c0_module()
    evidence = module._training_token_evidence(
        ("a", "b", "c", "d"),
        ["c", "a"],
        attempted_windows=1,
        batch_size=2,
        full_epoch=False,
    )
    assert evidence["consumed_sample_count"] == 2
    assert evidence["drop_last_remainder_count"] is None
    assert evidence["drop_last_remainder_tokens_sorted"] is None


def test_c0_health_uses_standard_even_sample_median():
    module = _c0_module()
    chunks = [
        {
            "state_after": {"invalid_windows": 0},
            "metrics": {"loss": 2.0},
        },
        {"metrics": {"loss": 1.0}},
    ]
    records = []
    for index, ratio in enumerate((1.0, 2.0, 4.0, 100.0), start=64):
        records.append({
            "counters_before": {"attempted_windows": index - 1},
            "parameter_gradients": {
                "global": {"all_finite": True},
                "by_prefix": {
                    prefix: {"complete_l2": 1.0}
                    for prefix in module._required_prefixes("lidar_only")
                },
            },
            "parameter_updates": {
                "by_prefix": {
                    "head": {"realized_update_over_weight": ratio},
                    "lidar_encoder.backbone.stem": {
                        "realized_update_over_weight": 1.0e-3,
                    },
                },
            },
        })
    health = module._cell_health(
        {"mode": "lidar_only", "attempted_windows": 128},
        chunks,
        tuple(records),
        {"invalid_windows": 0, "optimizer_step": 1, "discarded_windows": 0},
        None,
    )
    assert health["median_sampled_head_update_over_weight"] == 3.0
