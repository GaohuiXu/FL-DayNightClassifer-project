"""Checkpoint-continuation gate for S10 Phase I-P.

The production profiler captures complete per-tensor diagnostics.  This module
turns those diagnostics into the owner-approved continuation verdict without
importing PyTorch, so an immutable raw attempt can also be reassessed offline.
Numerical trajectory distances remain visible but do not reject an otherwise
exact, structurally valid, finite checkpoint continuation.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "s10.phase1p.continuation-gate.v2"
GROUPS = (
    "model_parameters",
    "bn_running_mean",
    "bn_running_var",
    "adam_exp_avg",
    "adam_exp_avg_sq",
)
STATE_KINDS = ("model", "optimizer", "scheduler", "scaler")


class Phase1CheckpointGateError(ValueError):
    """Raised when continuation evidence is incomplete or malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase1CheckpointGateError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_model_suffix(name: str, suffix: str) -> bool:
    # A model state_dict is captured as root.str:'module.path.buffer_name'.
    return name.endswith(f".{suffix}'") or name.endswith(f'.{suffix}"')


def _is_optimizer_leaf(name: str, leaf: str) -> bool:
    return name.endswith(f".str:'{leaf}'") or name.endswith(f'.str:"{leaf}"')


def _reported_tensor_names(report: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    numerical = report["numerical"]
    names = set(str(name) for name in numerical["by_tensor"])
    if "floating_tensor_names" in report and "discrete_tensor_names" in report:
        floating = set(str(name) for name in report["floating_tensor_names"])
        discrete = set(str(name) for name in report["discrete_tensor_names"])
        _require(floating.isdisjoint(discrete), "floating/discrete tensor sets overlap")
        _require(floating | discrete == names, "floating/discrete tensor sets are incomplete")
        return names, floating, discrete

    # Compatibility for Job 525192, produced before dtype sets were published.
    # Its state-key shapes are explicit in the per-tensor report: model BN counts
    # and Adam steps are the only semantically discrete tensors.
    if any(_is_optimizer_leaf(name, "exp_avg") for name in names):
        floating = {
            name
            for name in names
            if _is_optimizer_leaf(name, "exp_avg")
            or _is_optimizer_leaf(name, "exp_avg_sq")
        }
    elif any(_is_model_suffix(name, "num_batches_tracked") for name in names):
        floating = {
            name for name in names if not _is_model_suffix(name, "num_batches_tracked")
        }
    else:
        # Scheduler/scaler reports are exact-state evidence; in the current
        # checkpoint schema their numerical state is scalar structural data.
        floating = set()
    return names, floating, names - floating


def _classify(
    kind: str,
    report: Mapping[str, Any],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    names, floating, discrete = _reported_tensor_names(report)
    groups = {name: [] for name in GROUPS}
    exact: list[str] = []
    errors: list[str] = []
    for name in sorted(names):
        group = None
        if kind == "model":
            if _is_model_suffix(name, "running_mean"):
                group = "bn_running_mean"
            elif _is_model_suffix(name, "running_var"):
                group = "bn_running_var"
            elif name in floating:
                group = "model_parameters"
        elif kind == "optimizer":
            if _is_optimizer_leaf(name, "exp_avg"):
                group = "adam_exp_avg"
            elif _is_optimizer_leaf(name, "exp_avg_sq"):
                group = "adam_exp_avg_sq"

        if group is None:
            exact.append(name)
        elif name not in floating:
            errors.append(f"{kind}:{name} is non-floating but assigned to {group}")
            exact.append(name)
        else:
            groups[group].append(name)

    _require(set(exact).issubset(names), f"{kind} exact tensor classification drift")
    _require(discrete.issubset(set(exact)), f"{kind} discrete tensor escaped exact gate")
    return groups, exact, errors


def _summarize(
    report: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    by_tensor = report["numerical"]["by_tensor"]
    entries = [by_tensor[name] for name in names]
    reference_l2 = math.sqrt(sum(float(item["reference_l2"]) ** 2 for item in entries))
    candidate_l2 = math.sqrt(sum(float(item["candidate_l2"]) ** 2 for item in entries))
    absolute_l2 = math.sqrt(
        sum(float(item["absolute_l2_error"]) ** 2 for item in entries)
    )
    relative_l2 = absolute_l2 / reference_l2 if reference_l2 > 0.0 else absolute_l2
    return {
        "tensor_count": len(entries),
        "tensor_names_sha256": _canonical_sha256(list(names)),
        "elements": sum(int(item["elements"]) for item in entries),
        "all_finite": all(bool(item["all_finite"]) for item in entries),
        "reference_l2": reference_l2,
        "candidate_l2": candidate_l2,
        "absolute_l2_error": absolute_l2,
        "relative_l2_error": relative_l2,
        "max_abs_error": max(
            (float(item["max_abs_error"]) for item in entries), default=0.0
        ),
    }


def _comparison_integrity(report: Mapping[str, Any]) -> bool:
    numerical = report["numerical"]
    return bool(
        report["structure_equal"]
        and numerical["name_set_equal"]
        and not numerical["shape_mismatch_tensors"]
        and not numerical["dtype_mismatch_tensors"]
        and numerical["global"]["all_finite"]
    )


def _view(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    continuation = record["continuation"]
    _require(
        set(continuation) == set(STATE_KINDS),
        f"{label} continuation state kinds drift",
    )
    grouped_names = {name: [] for name in GROUPS}
    exact_failures: list[str] = []
    classification_errors: list[str] = []
    integrity = True
    elementwise_diagnostic = True
    for kind in STATE_KINDS:
        report = continuation[kind]
        integrity = integrity and _comparison_integrity(report)
        elementwise_diagnostic = elementwise_diagnostic and bool(report["gate_pass"])
        groups, exact, errors = _classify(kind, report)
        classification_errors.extend(errors)
        for group, names in groups.items():
            grouped_names[group].extend(names)
        by_tensor = report["numerical"]["by_tensor"]
        for name in exact:
            item = by_tensor[name]
            if (
                not bool(item["all_finite"])
                or float(item["absolute_l2_error"]) != 0.0
                or float(item["max_abs_error"]) != 0.0
            ):
                exact_failures.append(f"{kind}:{name}")
        exact_failures.extend(
            f"{kind}:{name}" for name in report["discrete_exact_failures"]
        )

    context = {
        "input_stream_exact": bool(record["input_stream"]["exact_equal"]),
        "training_state_exact": bool(record["training_state_equal"]),
        "rng_state_exact": bool(record["rng_state_equal"]),
    }
    groups = {}
    for group, names in grouped_names.items():
        if group.startswith("model_") or group.startswith("bn_"):
            report = continuation["model"]
        else:
            report = continuation["optimizer"]
        groups[group] = _summarize(report, sorted(names))
    exact_failures = sorted(set(exact_failures))
    return {
        "comparison_integrity_pass": integrity,
        "classification_errors": classification_errors,
        "exact_context": context,
        "exact_tensor_failure_count": len(exact_failures),
        "exact_tensor_failures": exact_failures,
        "exact_gate_pass": bool(
            integrity
            and not classification_errors
            and not exact_failures
            and all(context.values())
        ),
        "groups": groups,
        "elementwise_allclose_diagnostic_pass": elementwise_diagnostic,
    }


def _boundary_exact(restored_boundary: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        set(restored_boundary) == set(STATE_KINDS),
        "restored boundary state kinds drift",
    )
    failures = []
    for kind in STATE_KINDS:
        report = restored_boundary[kind]
        if (
            float(report["allclose_rtol"]) != 0.0
            or float(report["allclose_atol"]) != 0.0
            or not bool(report["gate_pass"])
        ):
            failures.append(kind)
    return {
        "state_kinds": list(STATE_KINDS),
        "failure_kinds": failures,
        "gate_pass": not failures,
    }


def evaluate_calibrated_continuation_gate(
    *,
    restored_boundary: Mapping[str, Any],
    same_process: Mapping[str, Any],
    fresh_process: Mapping[str, Any],
    relative_l2_tolerance: float,
    max_absolute_tolerance: float,
    calibration_factor: float = 1.25,
) -> dict[str, Any]:
    """Apply the exact-state gate and calibrated numerical diagnostics."""
    _require(
        math.isfinite(relative_l2_tolerance) and relative_l2_tolerance >= 0.0,
        "relative-L2 tolerance must be finite and nonnegative",
    )
    _require(
        math.isfinite(max_absolute_tolerance) and max_absolute_tolerance >= 0.0,
        "max-absolute tolerance must be finite and nonnegative",
    )
    _require(
        math.isfinite(calibration_factor) and calibration_factor >= 1.0,
        "calibration factor must be finite and at least one",
    )
    boundary = _boundary_exact(restored_boundary)
    same = _view(same_process, label="same-process")
    fresh = _view(fresh_process, label="fresh-process")
    group_diagnostics = {}
    for group in GROUPS:
        control = same["groups"][group]
        candidate = fresh["groups"][group]
        names_equal = (
            control["tensor_count"] == candidate["tensor_count"]
            and control["tensor_names_sha256"] == candidate["tensor_names_sha256"]
        )
        relative_limit = max(
            float(relative_l2_tolerance),
            float(calibration_factor) * float(control["relative_l2_error"]),
        )
        absolute_limit = max(
            float(max_absolute_tolerance),
            float(calibration_factor) * float(control["max_abs_error"]),
        )
        relative_pass = float(candidate["relative_l2_error"]) <= relative_limit
        absolute_pass = float(candidate["max_abs_error"]) <= absolute_limit
        integrity_pass = bool(
            names_equal
            and control["all_finite"]
            and candidate["all_finite"]
        )
        diagnostic_pass = bool(integrity_pass and relative_pass and absolute_pass)
        group_diagnostics[group] = {
            "same_process_repeat_control": control,
            "fresh_process": candidate,
            "relative_l2_limit": relative_limit,
            "max_absolute_limit": absolute_limit,
            "tensor_names_equal": names_equal,
            "integrity_pass": integrity_pass,
            "relative_l2_pass": relative_pass,
            "max_absolute_pass": absolute_pass,
            "diagnostic_pass": diagnostic_pass,
            "enforcement": "diagnostic_only",
        }

    gate = bool(
        boundary["gate_pass"]
        and same["exact_gate_pass"]
        and fresh["exact_gate_pass"]
        and all(item["integrity_pass"] for item in group_diagnostics.values())
    )
    return {
        "schema": SCHEMA,
        "rule": (
            "exact boundary/input/RNG/training/discrete state and exact structural "
            "identity plus finite numerical state are hard gates; grouped "
            "fresh-process relative-L2, max-absolute and elementwise-allclose "
            "results are diagnostic only"
        ),
        "diagnostic_rule": (
            "for each numerical group, compare fresh-process relative-L2 and "
            "max-absolute error with max(frozen tolerance, 1.25 * same-process "
            "repeat-control)"
        ),
        "frozen_tolerances": {
            "relative_l2": float(relative_l2_tolerance),
            "max_absolute": float(max_absolute_tolerance),
        },
        "calibration_factor": float(calibration_factor),
        "restored_boundary_exact": boundary,
        "same_process": same,
        "fresh_process": fresh,
        "group_diagnostics": group_diagnostics,
        "numerical_diagnostic_pass": all(
            item["diagnostic_pass"] for item in group_diagnostics.values()
        ),
        "gate_pass": gate,
    }
