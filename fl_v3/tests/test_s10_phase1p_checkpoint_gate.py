from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

_GATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "fl_v3"
    / "training"
    / "phase1_checkpoint_gate.py"
)
_SPEC = importlib.util.spec_from_file_location("s10_phase1_checkpoint_gate_test", _GATE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GATE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GATE_MODULE)
GROUPS = _GATE_MODULE.GROUPS
evaluate_calibrated_continuation_gate = (
    _GATE_MODULE.evaluate_calibrated_continuation_gate
)


MODEL_NAMES = {
    "model_parameters": "root.str:'layer.weight'",
    "bn_running_mean": "root.str:'bn.running_mean'",
    "bn_running_var": "root.str:'bn.running_var'",
}
OPTIMIZER_NAMES = {
    "adam_exp_avg": "root.str:'state'.int:0.str:'exp_avg'",
    "adam_exp_avg_sq": "root.str:'state'.int:0.str:'exp_avg_sq'",
}
MODEL_COUNT = "root.str:'bn.num_batches_tracked'"
OPTIMIZER_STEP = "root.str:'state'.int:0.str:'step'"


def _entry(error: float) -> dict:
    return {
        "tensor_count": 1,
        "elements": 1,
        "finite_pair_elements": 1,
        "reference_nonfinite_elements": 0,
        "candidate_nonfinite_elements": 0,
        "all_finite": True,
        "reference_l2": 1.0,
        "candidate_l2": 1.0 + error,
        "absolute_l2_error": abs(error),
        "relative_l2_error": abs(error),
        "cosine_similarity": 1.0,
        "max_abs_error": abs(error),
    }


def _report(errors: dict[str, float], *, floating: set[str]) -> dict:
    by_tensor = {name: _entry(error) for name, error in errors.items()}
    maximum = max((abs(value) for value in errors.values()), default=0.0)
    return {
        "reference_structure_sha256": "a" * 64,
        "candidate_structure_sha256": "a" * 64,
        "structure_equal": True,
        "allclose_rtol": 2e-3,
        "allclose_atol": 2e-4,
        # Deliberately retain an elementwise diagnostic failure.
        "floating_allclose_failures": sorted(floating) if maximum else [],
        "discrete_exact_failures": [],
        "floating_tensor_names": sorted(floating),
        "discrete_tensor_names": sorted(set(errors) - floating),
        "numerical": {
            "name_set_equal": True,
            "shape_mismatch_tensors": [],
            "dtype_mismatch_tensors": [],
            "global": {"all_finite": True},
            "by_tensor": by_tensor,
        },
        "gate_pass": maximum == 0.0,
    }


def _record(group_error: float) -> dict:
    model_errors = {name: group_error for name in MODEL_NAMES.values()}
    model_errors[MODEL_COUNT] = 0.0
    optimizer_errors = {name: group_error for name in OPTIMIZER_NAMES.values()}
    optimizer_errors[OPTIMIZER_STEP] = 0.0
    empty = _report({}, floating=set())
    return {
        "continuation": {
            "model": _report(model_errors, floating=set(MODEL_NAMES.values())),
            "optimizer": _report(
                optimizer_errors,
                floating=set(OPTIMIZER_NAMES.values()) | {OPTIMIZER_STEP},
            ),
            "scheduler": copy.deepcopy(empty),
            "scaler": copy.deepcopy(empty),
        },
        "input_stream": {"exact_equal": True},
        "training_state_equal": True,
        "rng_state_equal": True,
    }


def _boundary() -> dict:
    result = {}
    for kind in ("model", "optimizer", "scheduler", "scaler"):
        report = _report({}, floating=set())
        report["allclose_rtol"] = 0.0
        report["allclose_atol"] = 0.0
        report["gate_pass"] = True
        result[kind] = report
    return result


def _gate(same: dict, fresh: dict) -> dict:
    return evaluate_calibrated_continuation_gate(
        restored_boundary=_boundary(),
        same_process=same,
        fresh_process=fresh,
        relative_l2_tolerance=2e-3,
        max_absolute_tolerance=2e-4,
    )


def test_group_diagnostic_uses_same_process_control_and_keeps_allclose_diagnostic_only():
    result = _gate(_record(0.01), _record(0.012))
    assert result["gate_pass"] is True
    assert result["numerical_diagnostic_pass"] is True
    assert result["same_process"]["elementwise_allclose_diagnostic_pass"] is False
    assert result["fresh_process"]["elementwise_allclose_diagnostic_pass"] is False
    assert set(result["group_diagnostics"]) == set(GROUPS)
    for group in GROUPS:
        diagnostic = result["group_diagnostics"][group]
        assert diagnostic["relative_l2_limit"] == 0.0125
        assert diagnostic["max_absolute_limit"] == 0.0125
        assert diagnostic["enforcement"] == "diagnostic_only"


def test_group_numerical_limit_failure_is_diagnostic_not_a_hard_failure():
    same = _record(0.0)
    fresh = _record(0.0)
    name = MODEL_NAMES["model_parameters"]
    fresh["continuation"]["model"]["numerical"]["by_tensor"][name] = _entry(0.001)
    result = _gate(same, fresh)
    group = result["group_diagnostics"]["model_parameters"]
    assert group["relative_l2_pass"] is True
    assert group["max_absolute_pass"] is False
    assert group["diagnostic_pass"] is False
    assert result["numerical_diagnostic_pass"] is False
    assert result["gate_pass"] is True


def test_input_rng_and_semantically_discrete_optimizer_step_remain_exact():
    same = _record(0.0)
    fresh = _record(0.0)
    fresh["input_stream"]["exact_equal"] = False
    fresh["continuation"]["optimizer"]["numerical"]["by_tensor"][OPTIMIZER_STEP] = (
        _entry(1e-8)
    )
    result = _gate(same, fresh)
    assert result["fresh_process"]["exact_context"]["input_stream_exact"] is False
    assert result["fresh_process"]["exact_tensor_failure_count"] == 1
    assert result["gate_pass"] is False


def test_nonfinite_numerical_state_remains_a_hard_failure():
    same = _record(0.0)
    fresh = _record(0.0)
    name = MODEL_NAMES["model_parameters"]
    fresh["continuation"]["model"]["numerical"]["by_tensor"][name][
        "all_finite"
    ] = False
    fresh["continuation"]["model"]["numerical"]["global"][
        "all_finite"
    ] = False
    result = _gate(same, fresh)
    assert result["fresh_process"]["comparison_integrity_pass"] is False
    assert result["gate_pass"] is False
