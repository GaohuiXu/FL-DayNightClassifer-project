from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts" / "s10_phase1_lidar_epoch4_diagnostic.py"


def _entry_module():
    spec = importlib.util.spec_from_file_location("s10_lidar_epoch4_diagnostic", ENTRY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nested_finite_stats_are_value_sensitive():
    module = _entry_module()
    finite = module._floating_tensor_stats(
        {"a": torch.tensor([1.0, -3.0]), "b": (torch.tensor([2]),)}
    )
    assert finite == {
        "floating_tensor_count": 1,
        "floating_value_count": 2,
        "nonfinite_value_count": 0,
        "all_finite": True,
        "max_absolute_finite_tensor": 3.0,
    }

    failed = module._floating_tensor_stats(
        [torch.tensor([float("nan"), float("inf"), 4.0])]
    )
    assert failed["floating_tensor_count"] == 1
    assert failed["floating_value_count"] == 3
    assert failed["nonfinite_value_count"] == 2
    assert failed["all_finite"] is False
    assert failed["max_absolute_finite_tensor"] is None


def test_D_select_wrapper_rejects_raw_nonfinite_before_decode():
    module = _entry_module()

    class Subject(torch.nn.Module):
        def forward(self, _batch):
            return {"center": torch.tensor([float("nan")])}

        def decode(self, *_args, **_kwargs):  # pragma: no cover - must not be reached
            raise AssertionError("decode must not hide a raw NaN")

    wrapper = module._FiniteForward(Subject())
    with pytest.raises(FloatingPointError, match="raw head output is nonfinite"):
        wrapper({})
    assert wrapper.forward_calls == 1


def test_batch_hash_is_stable_and_value_sensitive():
    module = _entry_module()
    left = {"x": torch.tensor([[1.0, 2.0]]), "tokens": ["a"]}
    right = {"x": torch.tensor([[1.0, 3.0]]), "tokens": ["a"]}
    assert module._batch_sha256(left) == module._batch_sha256(left)
    assert module._batch_sha256(left) != module._batch_sha256(right)


def test_localization_is_durable_and_precedes_the_only_D_select_peek():
    source = ENTRY.read_text(encoding="utf-8")
    localization_complete = source.index('"complete.json"')
    diagnostic_scope = source.index('diagnostic_scope_path = d_select_root')
    d_select_call = source.index("d_select_record = _evaluate_terminal(")
    assert localization_complete < diagnostic_scope < d_select_call
    assert "return_intermediates=True" in source
    assert '"diagnostic_scope.json"' in source
    assert '"terminal_epoch20_execution_remains_reserved": True' in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source
