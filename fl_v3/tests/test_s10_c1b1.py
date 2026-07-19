from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "s10_c1b1_capability.py"
    spec = importlib.util.spec_from_file_location("s10_c1b1_capability", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c1b1_envelope_and_fail_closed_selection_are_exact():
    module = _module()
    assert module.HORIZON == 1538
    assert module.PHYSICAL_BATCH == 4
    assert module.EXPECTED_D_LOW_SAMPLES == 6155
    assert module.EXPECTED_D_SELECT_SAMPLES == 4626
    assert module.EXPECTED_D_SELECT_LOGS == 8
    assert module.CANDIDATES == (
        ("C1-B1-CUR-A1-GN-DLOW", "group_norm"),
        ("C1-B1-CUR-A1-BN1D-DLOW", "batch_norm_1d"),
    )
    for cell, normalization in module.CANDIDATES:
        spec = module._candidate_spec(cell, normalization, "a" * 64)
        assert spec["attempted_windows"] == 1538
        assert spec["grad_scaler_init_scale"] == 32.0
        assert spec["evaluate"] is True
        assert spec["operator_profile"] is False
        assert spec["expected_initial_parameter_sha256"] == "a" * 64


def test_c1b1_config_changes_only_declared_horizon_from_c1b0():
    root = Path(__file__).resolve().parents[1]
    short = json.loads((root / "configs" / "s10_c1b0_f_a1_gn.json").read_text())
    full = json.loads((root / "configs" / "s10_c1b1_cur_a1_gn.json").read_text())
    assert short["training"]["max_optimizer_steps"] == 256
    assert full["training"]["max_optimizer_steps"] == 1538
    short["training"]["max_optimizer_steps"] = 1538
    assert short == full


def test_c1b1_paired_jackknife_uses_eight_logs_and_no_effect_has_zero_width():
    module = _module()
    report = module.jackknife_interval(0.0, [0.0] * 8)
    assert report["clusters"] == 8
    assert report["jackknife_standard_error"] == 0.0
    assert report["point_centered_95_interval"] == [0.0, 0.0]
    with pytest.raises(ValueError, match="eight"):
        module.jackknife_interval(0.0, [0.0] * 7)


class _FakeNuScenes:
    def __init__(self):
        self.samples = {
            "s1": {"scene_token": "scene-a"},
            "s2": {"scene_token": "scene-b"},
        }
        self.scenes = {
            "scene-a": {"log_token": "log-a"},
            "scene-b": {"log_token": "log-b"},
        }

    def get(self, table, token):
        return (self.samples if table == "sample" else self.scenes)[token]


def test_c1b1_log_cluster_binding_is_exact_and_rejects_outside_ownership():
    module = _module()
    grouped = module._tokens_by_log(_FakeNuScenes(), ("s1", "s2"), ("log-a", "log-b"))
    assert grouped == {"log-a": ("s1",), "log-b": ("s2",)}
    with pytest.raises(RuntimeError, match="outside"):
        module._tokens_by_log(_FakeNuScenes(), ("s1", "s2"), ("log-a", "log-c"))
