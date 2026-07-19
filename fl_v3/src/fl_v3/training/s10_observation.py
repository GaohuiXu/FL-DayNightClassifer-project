"""Narrow, no-update observation primitives for S10 STOP-B.

The recorder is attached only through explicit detector/SECOND seams.  It does
not register module hooks, create an optimizer, change the returned loss, or
advance training state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


STOP_B_SCHEMA = "fl_v3.s10.stop_b_observation.v1"
MAIN_BOUNDARIES = (
    "head.input",
    "bev_neck.input",
    "fusion.camera_input",
    "fusion.lidar_input",
    "second.output",
    "second.stage4",
    "second.down3",
    "second.stage3",
    "second.down2",
    "second.stage2",
    "second.down1",
    "second.stage1",
    "second.stem",
)
LIDAR_BACKWARD_CHAIN = (
    "head.input",
    "bev_neck.input",
    "fusion.lidar_input",
    "second.output",
    "second.stage4",
    "second.down3",
    "second.stage3",
    "second.down2",
    "second.stage2",
    "second.down1",
    "second.stage1",
    "second.stem",
)

_PARAMETER_PREFIXES = (
    "lidar_encoder.backbone.conv_out",
    "lidar_encoder.backbone.stage4",
    "lidar_encoder.backbone.down3",
    "lidar_encoder.backbone.stage3",
    "lidar_encoder.backbone.down2",
    "lidar_encoder.backbone.stage2",
    "lidar_encoder.backbone.down1",
    "lidar_encoder.backbone.stage1",
    "lidar_encoder.backbone.stem",
    "lidar_encoder.to_bev",
    "camera_backbone",
    "camera_neck",
    "view_transform",
    "lidar_backbone",
    "fusion",
    "bev_neck",
    "head",
)


class StopBObservationError(RuntimeError):
    """STOP-B evidence is incomplete or violates its no-update contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StopBObservationError(message)


def _finite_json_number(value: float) -> Any:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        label = "nan"
    elif value > 0:
        label = "+inf"
    else:
        label = "-inf"
    return {"value": None, "nonfinite": label}


def strict_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite_json_number(value)
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    raise TypeError(f"unsupported STOP-B JSON value {type(value)!r}")


def _tensor_stats(value: torch.Tensor, *, divisor: float = 1.0) -> dict[str, Any]:
    _require(math.isfinite(divisor) and divisor > 0.0, "gradient divisor must be finite and positive")
    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    finite = detached[finite_mask].to(torch.float64) / float(divisor)
    total = int(detached.numel())
    finite_count = int(finite_mask.sum().item())
    nan = int(torch.isnan(detached).sum().item())
    posinf = int(torch.isposinf(detached).sum().item())
    neginf = int(torch.isneginf(detached).sum().item())
    if finite.numel():
        sum_sq = float(torch.square(finite).sum().item())
        maximum = float(finite.abs().max().item())
        mean = float(finite.mean().item())
        rms = math.sqrt(max(0.0, sum_sq / finite_count))
        l2 = math.sqrt(max(0.0, sum_sq))
    else:
        maximum = mean = rms = l2 = 0.0
    nonfinite = nan + posinf + neginf
    return {
        "total_elements": total,
        "finite_elements": finite_count,
        "nonfinite_elements": nonfinite,
        "nan_elements": nan,
        "positive_inf_elements": posinf,
        "negative_inf_elements": neginf,
        "all_finite": nonfinite == 0 and finite_count == total,
        "finite_mean": mean,
        "stable_finite_l2": l2,
        "stable_finite_rms": rms,
        "max_abs_finite": maximum,
        "complete_l2": l2 if nonfinite == 0 and finite_count == total else None,
    }


@dataclass
class _Boundary:
    name: str
    tensor: torch.Tensor
    sample_index: torch.Tensor
    batch_size: int
    layout: str
    spatial_shape: tuple[int, ...]
    active_per_sample: tuple[int, ...]


class StopBObservationRecorder:
    """Collect one forward/backward using only predeclared boundaries."""

    def __init__(
        self,
        *,
        expected_boundaries: Sequence[str] = MAIN_BOUNDARIES,
        expected_group_norm_count: int = 21,
    ):
        self.expected_boundaries = tuple(expected_boundaries)
        self.expected_group_norm_count = int(expected_group_norm_count)
        self.boundaries: dict[str, _Boundary] = {}
        self.group_norms: dict[str, dict[str, Any]] = {}

    def _add_boundary(self, record: _Boundary) -> None:
        _require(record.name in self.expected_boundaries, f"unexpected STOP-B boundary {record.name}")
        _require(record.name not in self.boundaries, f"duplicate STOP-B boundary {record.name}")
        tensor = record.tensor
        _require(torch.is_grad_enabled() and tensor.requires_grad, f"boundary {record.name} has no autograd")
        tensor.retain_grad()
        self.boundaries[record.name] = record

    def capture_dense_boundary(self, name: str, tensor: torch.Tensor) -> None:
        _require(tensor.ndim >= 2, f"dense boundary {name} must include batch and channels")
        batch_size = int(tensor.shape[0])
        sample_index = torch.arange(batch_size, device=tensor.device, dtype=torch.int64)
        active = tuple(int(tensor[index].numel()) for index in range(batch_size))
        self._add_boundary(_Boundary(
            name=name,
            tensor=tensor,
            sample_index=sample_index,
            batch_size=batch_size,
            layout="dense_batch_first",
            spatial_shape=tuple(int(value) for value in tensor.shape[2:]),
            active_per_sample=active,
        ))

    def capture_sparse_boundary(
        self,
        name: str,
        features: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        spatial_shape: Sequence[int],
    ) -> None:
        _require(features.ndim == 2, f"sparse boundary {name} features must be [N,C]")
        _require(indices.ndim == 2 and indices.shape[1] == 4, f"sparse boundary {name} indices must be [N,4]")
        _require(indices.shape[0] == features.shape[0], f"sparse boundary {name} feature/index drift")
        sample_index = indices[:, 0].to(torch.int64)
        counts = torch.bincount(sample_index, minlength=int(batch_size))
        _require(int(counts.numel()) == int(batch_size), f"sparse boundary {name} batch-index drift")
        self._add_boundary(_Boundary(
            name=name,
            tensor=features,
            sample_index=sample_index,
            batch_size=int(batch_size),
            layout="spconv_features_NC",
            spatial_shape=tuple(int(value) for value in spatial_shape),
            active_per_sample=tuple(int(value) for value in counts.detach().cpu().tolist()),
        ))

    def record_group_norm(
        self,
        name: str,
        value: torch.Tensor,
        output: torch.Tensor,
        module: nn.GroupNorm,
    ) -> None:
        _require(name not in self.group_norms, f"duplicate GroupNorm observation {name}")
        _require(value.ndim == 2, f"sparse GroupNorm {name} input must be [N,C]")
        rows, channels = (int(value.shape[0]), int(value.shape[1]))
        groups = int(module.num_groups)
        _require(channels == int(module.num_channels) and channels % groups == 0, f"GroupNorm {name} contract drift")
        width = channels // groups
        detached = value.detach().to(torch.float32).reshape(rows, groups, width)
        variance = detached.var(dim=2, unbiased=False)
        inv_std = torch.rsqrt(variance + float(module.eps))
        thresholds = (1.0, 10.0, 100.0)
        self.group_norms[name] = {
            "rows": rows,
            "channels": channels,
            "groups": groups,
            "values_per_group": width,
            "eps": float(module.eps),
            "input": _tensor_stats(value),
            "output": _tensor_stats(output),
            "variance_min": float(variance.min().item()) if variance.numel() else 0.0,
            "variance_mean": float(variance.mean().item()) if variance.numel() else 0.0,
            "variance_max": float(variance.max().item()) if variance.numel() else 0.0,
            "inv_std_mean": float(inv_std.mean().item()) if inv_std.numel() else 0.0,
            "inv_std_max": float(inv_std.max().item()) if inv_std.numel() else 0.0,
            "variance_threshold_counts": {
                f"le_{int(multiplier)}x_eps": int(
                    (variance <= float(module.eps) * multiplier).sum().item()
                )
                for multiplier in thresholds
            },
            "group_instances": int(variance.numel()),
        }

    def validate_forward(self) -> None:
        actual = set(self.boundaries)
        expected = set(self.expected_boundaries)
        _require(actual == expected, f"STOP-B boundary set drift: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        _require(
            len(self.group_norms) == self.expected_group_norm_count,
            f"expected {self.expected_group_norm_count} sparse GroupNorm sites, got {len(self.group_norms)}",
        )

    def tensors_in_order(self) -> tuple[torch.Tensor, ...]:
        self.validate_forward()
        return tuple(self.boundaries[name].tensor for name in self.expected_boundaries)

    def forward_snapshot(self) -> dict[str, Any]:
        self.validate_forward()
        return {
            "boundaries": {
                name: {
                    "layout": boundary.layout,
                    "activation_shape": [int(value) for value in boundary.tensor.shape],
                    "activation_dtype": str(boundary.tensor.dtype),
                    "spatial_shape": list(boundary.spatial_shape),
                    "active_per_sample": list(boundary.active_per_sample),
                    "activation": _tensor_stats(boundary.tensor),
                }
                for name, boundary in self.boundaries.items()
            },
            "sparse_group_norms": dict(self.group_norms),
        }

    def gradient_snapshot(self, *, scale_divisor: float) -> dict[str, Any]:
        self.validate_forward()
        result = {}
        for name in self.expected_boundaries:
            boundary = self.boundaries[name]
            gradient = boundary.tensor.grad
            _require(gradient is not None, f"boundary {name} has no retained gradient")
            per_sample = []
            if boundary.layout == "dense_batch_first":
                for index in range(boundary.batch_size):
                    per_sample.append(_tensor_stats(gradient[index], divisor=scale_divisor))
            else:
                for index in range(boundary.batch_size):
                    per_sample.append(
                        _tensor_stats(gradient[boundary.sample_index == index], divisor=scale_divisor)
                    )
            result[name] = {
                "raw_dtype": str(gradient.dtype),
                "scale_divisor": float(scale_divisor),
                "true_unscaled": _tensor_stats(gradient, divisor=scale_divisor),
                "true_unscaled_per_sample": per_sample,
            }
        return result


def _parameter_prefix(name: str) -> str:
    for prefix in _PARAMETER_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return "other"


def parameter_gradient_snapshot(model: nn.Module, *, scale_divisor: float) -> dict[str, Any]:
    entries = []
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        stats = _tensor_stats(parameter.grad, divisor=scale_divisor)
        entries.append((name, _parameter_prefix(name), stats, int(parameter.numel())))
    groups: dict[str, list[tuple[str, dict[str, Any], int]]] = {}
    for name, prefix, stats, count in entries:
        groups.setdefault(prefix, []).append((name, stats, count))

    def summarize(group):
        total = sum(count for _, _, count in group)
        finite = sum(item[1]["finite_elements"] for item in group)
        nonfinite = sum(item[1]["nonfinite_elements"] for item in group)
        sum_sq = sum(item[1]["stable_finite_l2"] ** 2 for item in group)
        maximum = max((item[1]["max_abs_finite"] for item in group), default=0.0)
        max_name = next(
            (name for name, stats, _ in group if stats["max_abs_finite"] == maximum), None
        )
        return {
            "total_elements": total,
            "finite_elements": finite,
            "nonfinite_elements": nonfinite,
            "all_finite": nonfinite == 0 and finite == total,
            "stable_finite_l2": math.sqrt(max(0.0, sum_sq)),
            "stable_finite_rms": math.sqrt(max(0.0, sum_sq / finite)) if finite else 0.0,
            "max_abs_finite": maximum,
            "max_abs_parameter": max_name,
        }

    bad = [name for name, _, stats, _ in entries if not stats["all_finite"]]
    return {
        "gradient_domain": "explicit_true_unscaled_copy",
        "scale_divisor": float(scale_divisor),
        "parameters_with_grad": len(entries),
        "missing_grad_parameter_count": len(missing),
        "missing_grad_parameters": missing,
        "first_nonfinite_parameter_in_named_order": bad[0] if bad else None,
        "global": summarize([(name, stats, count) for name, _, stats, count in entries]),
        "by_prefix": {
            prefix: summarize(group) for prefix, group in sorted(groups.items())
        },
    }


def capture_parameter_gradient_tensors(
    model: nn.Module,
) -> dict[str, torch.Tensor | None]:
    """Copy raw trainable-parameter gradients to CPU before they are cleared."""
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def compare_parameter_gradient_tensors(
    reference: Mapping[str, torch.Tensor | None],
    candidate: Mapping[str, torch.Tensor | None],
    *,
    scale_divisor: float,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    relative_l2_limit: float = 1e-6,
) -> dict[str, Any]:
    """Compare two raw gradient snapshots under the fixed STOP-B envelope."""
    _require(
        math.isfinite(scale_divisor) and scale_divisor > 0.0,
        "gradient comparison divisor must be finite and positive",
    )
    reference_names = set(reference)
    candidate_names = set(candidate)
    names = sorted(reference_names | candidate_names)
    reference_missing = sorted(
        name for name in reference_names if reference[name] is None
    )
    candidate_missing = sorted(
        name for name in candidate_names if candidate[name] is None
    )
    entries: list[dict[str, Any]] = []
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    allclose_failures: list[str] = []
    nonfinite_parameters: list[str] = []

    for name in names:
        left = reference.get(name)
        right = candidate.get(name)
        if left is None or right is None:
            continue
        if tuple(left.shape) != tuple(right.shape):
            shape_mismatches.append(name)
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(name)
            continue
        left_unscaled = left / float(scale_divisor)
        right_unscaled = right / float(scale_divisor)
        left_finite = torch.isfinite(left_unscaled)
        right_finite = torch.isfinite(right_unscaled)
        both_finite = left_finite & right_finite
        all_finite = bool(left_finite.all().item() and right_finite.all().item())
        if not all_finite:
            nonfinite_parameters.append(name)
        close = bool(
            all_finite
            and torch.allclose(left_unscaled, right_unscaled, rtol=rtol, atol=atol)
        )
        if not close:
            allclose_failures.append(name)
        if bool(both_finite.any().item()):
            left64 = left_unscaled[both_finite].to(torch.float64)
            right64 = right_unscaled[both_finite].to(torch.float64)
            delta64 = right64 - left64
            reference_sum_sq = float(torch.square(left64).sum().item())
            candidate_sum_sq = float(torch.square(right64).sum().item())
            error_sum_sq = float(torch.square(delta64).sum().item())
            dot_product = float((left64 * right64).sum().item())
            max_abs_error = float(delta64.abs().max().item())
        else:
            reference_sum_sq = 0.0
            candidate_sum_sq = 0.0
            error_sum_sq = 0.0
            dot_product = 0.0
            max_abs_error = 0.0
        entries.append({
            "name": name,
            "prefix": _parameter_prefix(name),
            "elements": int(left.numel()),
            "finite_pair_elements": int(both_finite.sum().item()),
            "reference_nonfinite_elements": int((~left_finite).sum().item()),
            "candidate_nonfinite_elements": int((~right_finite).sum().item()),
            "all_finite": all_finite,
            "allclose": close,
            "reference_sum_sq": reference_sum_sq,
            "candidate_sum_sq": candidate_sum_sq,
            "error_sum_sq": error_sum_sq,
            "dot_product": dot_product,
            "max_abs_error": max_abs_error,
        })

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        reference_sum_sq = sum(float(item["reference_sum_sq"]) for item in group)
        candidate_sum_sq = sum(float(item["candidate_sum_sq"]) for item in group)
        error_sum_sq = sum(float(item["error_sum_sq"]) for item in group)
        dot_product = sum(float(item["dot_product"]) for item in group)
        reference_l2 = math.sqrt(max(0.0, reference_sum_sq))
        candidate_l2 = math.sqrt(max(0.0, candidate_sum_sq))
        absolute_l2 = math.sqrt(max(0.0, error_sum_sq))
        if reference_l2 > 0.0 and candidate_l2 > 0.0:
            cosine = max(-1.0, min(1.0, dot_product / (reference_l2 * candidate_l2)))
        elif reference_l2 == 0.0 and candidate_l2 == 0.0:
            cosine = 1.0
        else:
            cosine = 0.0
        return {
            "parameter_count": len(group),
            "elements": sum(int(item["elements"]) for item in group),
            "finite_pair_elements": sum(
                int(item["finite_pair_elements"]) for item in group
            ),
            "reference_nonfinite_elements": sum(
                int(item["reference_nonfinite_elements"]) for item in group
            ),
            "candidate_nonfinite_elements": sum(
                int(item["candidate_nonfinite_elements"]) for item in group
            ),
            "all_finite": all(bool(item["all_finite"]) for item in group),
            "allclose": all(bool(item["allclose"]) for item in group),
            "reference_l2": reference_l2,
            "candidate_l2": candidate_l2,
            "absolute_l2_error": absolute_l2,
            "relative_l2_error": (
                absolute_l2 / reference_l2 if reference_l2 > 0.0 else absolute_l2
            ),
            "cosine_similarity": cosine,
            "max_abs_error": max(
                (float(item["max_abs_error"]) for item in group), default=0.0
            ),
        }

    by_prefix: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_prefix.setdefault(str(entry["prefix"]), []).append(entry)
    global_summary = summarize(entries)
    name_set_equal = reference_names == candidate_names
    missing_set_equal = reference_missing == candidate_missing
    gate = bool(
        name_set_equal
        and missing_set_equal
        and not shape_mismatches
        and not dtype_mismatches
        and not nonfinite_parameters
        and not allclose_failures
        and global_summary["relative_l2_error"] <= relative_l2_limit
    )
    return {
        "gradient_domain": "explicit_true_unscaled_cpu_copy",
        "raw_scale_divisor": float(scale_divisor),
        "rtol": float(rtol),
        "atol": float(atol),
        "relative_l2_limit": float(relative_l2_limit),
        "reference_parameter_count": len(reference_names),
        "candidate_parameter_count": len(candidate_names),
        "name_set_equal": name_set_equal,
        "reference_missing_gradients": reference_missing,
        "candidate_missing_gradients": candidate_missing,
        "missing_gradient_sets_equal": missing_set_equal,
        "shape_mismatch_parameters": shape_mismatches,
        "dtype_mismatch_parameters": dtype_mismatches,
        "nonfinite_parameters": nonfinite_parameters,
        "allclose_failure_parameters": allclose_failures,
        "global": global_summary,
        "by_prefix": {
            prefix: summarize(group) for prefix, group in sorted(by_prefix.items())
        },
        "gate_pass": gate,
    }


def capture_tensor_tree_tensors(value: Any) -> dict[str, torch.Tensor]:
    """Copy a tensor-only nested output tree to named CPU tensors."""
    result: dict[str, torch.Tensor] = {}

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            result[path] = item.detach().cpu().clone()
        elif isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            raise TypeError(f"tensor tree contains unsupported {type(item)!r} at {path}")

    visit(value, "root")
    return result


def compare_tensor_tree_tensors(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Return finite global/per-tensor numerical differences for output trees."""
    reference_names = set(reference)
    candidate_names = set(candidate)
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    entries: list[dict[str, Any]] = []
    for name in sorted(reference_names | candidate_names):
        left = reference.get(name)
        right = candidate.get(name)
        if left is None or right is None:
            continue
        if tuple(left.shape) != tuple(right.shape):
            shape_mismatches.append(name)
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append(name)
            continue
        left_finite = torch.isfinite(left)
        right_finite = torch.isfinite(right)
        both_finite = left_finite & right_finite
        if bool(both_finite.any().item()):
            left64 = left[both_finite].to(torch.float64)
            right64 = right[both_finite].to(torch.float64)
            delta64 = right64 - left64
            reference_sum_sq = float(torch.square(left64).sum().item())
            candidate_sum_sq = float(torch.square(right64).sum().item())
            error_sum_sq = float(torch.square(delta64).sum().item())
            dot_product = float((left64 * right64).sum().item())
            max_abs_error = float(delta64.abs().max().item())
        else:
            reference_sum_sq = candidate_sum_sq = error_sum_sq = dot_product = 0.0
            max_abs_error = 0.0
        entries.append({
            "name": name,
            "elements": int(left.numel()),
            "finite_pair_elements": int(both_finite.sum().item()),
            "reference_nonfinite_elements": int((~left_finite).sum().item()),
            "candidate_nonfinite_elements": int((~right_finite).sum().item()),
            "reference_sum_sq": reference_sum_sq,
            "candidate_sum_sq": candidate_sum_sq,
            "error_sum_sq": error_sum_sq,
            "dot_product": dot_product,
            "max_abs_error": max_abs_error,
        })

    def summarize(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        reference_l2 = math.sqrt(max(
            0.0, sum(float(item["reference_sum_sq"]) for item in group)
        ))
        candidate_l2 = math.sqrt(max(
            0.0, sum(float(item["candidate_sum_sq"]) for item in group)
        ))
        absolute_l2 = math.sqrt(max(
            0.0, sum(float(item["error_sum_sq"]) for item in group)
        ))
        dot_product = sum(float(item["dot_product"]) for item in group)
        if reference_l2 > 0.0 and candidate_l2 > 0.0:
            cosine = max(-1.0, min(1.0, dot_product / (reference_l2 * candidate_l2)))
        elif reference_l2 == 0.0 and candidate_l2 == 0.0:
            cosine = 1.0
        else:
            cosine = 0.0
        return {
            "tensor_count": len(group),
            "elements": sum(int(item["elements"]) for item in group),
            "finite_pair_elements": sum(
                int(item["finite_pair_elements"]) for item in group
            ),
            "reference_nonfinite_elements": sum(
                int(item["reference_nonfinite_elements"]) for item in group
            ),
            "candidate_nonfinite_elements": sum(
                int(item["candidate_nonfinite_elements"]) for item in group
            ),
            "all_finite": all(
                int(item["reference_nonfinite_elements"]) == 0
                and int(item["candidate_nonfinite_elements"]) == 0
                for item in group
            ),
            "reference_l2": reference_l2,
            "candidate_l2": candidate_l2,
            "absolute_l2_error": absolute_l2,
            "relative_l2_error": (
                absolute_l2 / reference_l2 if reference_l2 > 0.0 else absolute_l2
            ),
            "cosine_similarity": cosine,
            "max_abs_error": max(
                (float(item["max_abs_error"]) for item in group), default=0.0
            ),
        }

    return {
        "name_set_equal": reference_names == candidate_names,
        "shape_mismatch_tensors": shape_mismatches,
        "dtype_mismatch_tensors": dtype_mismatches,
        "global": summarize(entries),
        "by_tensor": {
            str(item["name"]): summarize([item]) for item in entries
        },
    }


def classify_stop_b_randomness(
    mode_summaries: Mapping[str, Mapping[str, Any]],
    *,
    dominance_factor: float = 4.0,
    denominator_floor: float = 1e-8,
) -> dict[str, Any]:
    """Classify only the operational source of repeated STOP-B variation.

    This intentionally does not claim a kernel, module, or large-gradient cause.
    A label needs support from at least two of loss/output/gradient relative-L2
    medians; otherwise the bounded result remains mixed/inconclusive.
    """
    _require(
        math.isfinite(dominance_factor) and dominance_factor > 1.0,
        "randomness dominance factor must be finite and greater than one",
    )
    _require(
        math.isfinite(denominator_floor) and denominator_floor > 0.0,
        "randomness denominator floor must be finite and positive",
    )
    mode_order = ("C-STR8", "L-S075", "F-U")
    metric_fields = {
        "loss": "loss_relative_difference",
        "output": "output_relative_l2",
        "gradient": "gradient_relative_l2",
    }

    def medians(mode: str, group: str) -> dict[str, float]:
        try:
            summary = mode_summaries[mode]["groups"][group]
            return {
                metric: float(summary[field]["median"])
                for metric, field in metric_fields.items()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise StopBObservationError(
                f"invalid randomness summary for {mode}/{group}"
            ) from exc

    fixed = {mode: medians(mode, "fixed_seed") for mode in mode_order}
    varying = {mode: medians(mode, "varying_seed") for mode in mode_order}
    support = {
        "CAMERA_STOCHASTICITY": [],
        "LIDAR_RUNTIME_VARIATION": [],
        "FUSION_ONLY_INTERACTION": [],
    }
    ratios = {}
    for metric in metric_fields:
        camera_ratio = varying["C-STR8"][metric] / max(
            fixed["C-STR8"][metric], denominator_floor
        )
        lidar_ratio = fixed["L-S075"][metric] / max(
            fixed["C-STR8"][metric], denominator_floor
        )
        fusion_ratio = fixed["F-U"][metric] / max(
            fixed["C-STR8"][metric],
            fixed["L-S075"][metric],
            denominator_floor,
        )
        ratios[metric] = {
            "camera_varying_over_fixed": camera_ratio,
            "lidar_fixed_over_camera_fixed": lidar_ratio,
            "fusion_fixed_over_max_unimodal_fixed": fusion_ratio,
        }
        if camera_ratio >= dominance_factor:
            support["CAMERA_STOCHASTICITY"].append(metric)
        if lidar_ratio >= dominance_factor:
            support["LIDAR_RUNTIME_VARIATION"].append(metric)
        if fusion_ratio >= dominance_factor:
            support["FUSION_ONLY_INTERACTION"].append(metric)
    qualified = sorted(
        label for label, metrics in support.items() if len(metrics) >= 2
    )
    label = qualified[0] if len(qualified) == 1 else "MIXED_INCONCLUSIVE"
    return {
        "label": label,
        "dominance_factor": float(dominance_factor),
        "denominator_floor": float(denominator_floor),
        "support_metrics": support,
        "qualified_labels": qualified,
        "ratios": ratios,
        "fixed_seed_medians": fixed,
        "varying_seed_medians": varying,
        "interpretation": (
            "operational candidate source only; no kernel/module/causal claim"
        ),
    }


def loss_term_snapshot(bundle: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = bundle["aggregate_total"]
    tasks = []
    for task in bundle["tasks"]:
        tensors = task["tensors"]
        tasks.append({
            "task_index": int(task["task_index"]),
            "global_class_ids": list(task["global_class_ids"]),
            "metadata": dict(task["metadata"]),
            "terms": {
                key: float(value.detach().item())
                for key, value in tensors.items()
                if torch.is_tensor(value) and value.ndim == 0
            },
            "hm_sample_numerators": [
                float(value.detach().item()) for value in tensors["hm_sample_numerators"]
            ],
            "hm_sample_denominators": [
                float(value.detach().item()) for value in tensors["hm_sample_denominators"]
            ],
            "reg_sample_numerators": [
                float(value.detach().item()) for value in tensors["reg_sample_numerators"]
            ],
            "reg_sample_denominators": [
                float(value.detach().item()) for value in tensors["reg_sample_denominators"]
            ],
        })
    reconstructed = sum(task["terms"]["total"] for task in tasks)
    actual = float(aggregate.detach().item())
    return {
        "aggregate_total": actual,
        "task_total_sum_host_fp64": reconstructed,
        "task_total_sum_residual": reconstructed - actual,
        "tasks": tasks,
    }


def term_sources(bundle: Mapping[str, Any]) -> list[tuple[str, torch.Tensor]]:
    sources = []
    for task in bundle["tasks"]:
        index = int(task["task_index"])
        tensors = task["tensors"]
        sources.append((f"task{index}.hm", tensors["hm_loss"]))
        sources.append((f"task{index}.weighted_reg", tensors["weighted_reg_loss"]))
    _require(len(sources) == 12, "STOP-B term source count drift")
    return sources


def recompose_from_sample_terms(bundle: Mapping[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Rebuild B4 from four per-sample raw numerator/denominator contributions."""
    task_values = []
    for task in bundle["tasks"]:
        tensors = task["tensors"]
        hm_numerator = torch.stack(tuple(tensors["hm_sample_numerators"])).sum()
        hm_denominator = torch.stack(tuple(tensors["hm_sample_denominators"])).sum().clamp_min(1.0)
        reg_numerator = torch.stack(tuple(tensors["reg_sample_numerators"])).sum()
        reg_denominator_raw = torch.stack(tuple(tensors["reg_sample_denominators"])).sum()
        if bool(task["metadata"]["class_weighted_regression"]):
            reg_denominator = reg_denominator_raw.clamp_min(1e-6)
        else:
            reg_denominator = reg_denominator_raw.clamp_min(1.0)
        task_values.append(
            hm_numerator / hm_denominator
            + float(task["metadata"]["reg_weight"]) * reg_numerator / reg_denominator
        )
    return torch.stack(task_values).sum(), task_values


def attribute_term_gradients(
    bundle: Mapping[str, Any],
    recorder: StopBObservationRecorder,
) -> dict[str, Any]:
    """Project each exact task/term gradient onto the aggregate boundary gradient."""
    tensors = recorder.tensors_in_order()
    total = bundle["aggregate_total"]
    total_gradients = torch.autograd.grad(
        total, tensors, retain_graph=True, allow_unused=True
    )
    _require(all(value is not None for value in total_gradients), "aggregate term replay has a missing boundary gradient")
    totals = tuple(value.detach() for value in total_gradients if value is not None)
    accumulated = [torch.zeros_like(value) for value in totals]
    by_source = {}
    for source_name, source in term_sources(bundle):
        gradients = torch.autograd.grad(
            source, tensors, retain_graph=True, allow_unused=True
        )
        _require(all(value is not None for value in gradients), f"term {source_name} has a missing boundary gradient")
        source_record = {}
        for index, (name, gradient, total_gradient) in enumerate(
            zip(recorder.expected_boundaries, gradients, totals, strict=True)
        ):
            current = gradient.detach()
            accumulated[index].add_(current)
            total_norm_sq = float(torch.square(total_gradient).sum(dtype=torch.float64).item())
            source_norm = math.sqrt(
                max(0.0, float(torch.square(current).sum(dtype=torch.float64).item()))
            )
            total_norm = math.sqrt(max(0.0, total_norm_sq))
            dot = float((current * total_gradient).sum(dtype=torch.float64).item())
            source_record[name] = {
                "projection_share": dot / total_norm_sq if total_norm_sq > 0.0 else 0.0,
                "cosine_with_total": dot / (source_norm * total_norm) if source_norm > 0.0 and total_norm > 0.0 else 0.0,
                "gradient_l2": source_norm,
                "gradient_rms": source_norm / math.sqrt(current.numel()) if current.numel() else 0.0,
            }
        by_source[source_name] = source_record
    reconstruction = {}
    for name, summed, total_gradient in zip(
        recorder.expected_boundaries, accumulated, totals, strict=True
    ):
        delta = summed - total_gradient
        reference = math.sqrt(
            max(0.0, float(torch.square(total_gradient).sum(dtype=torch.float64).item()))
        )
        absolute = math.sqrt(
            max(0.0, float(torch.square(delta).sum(dtype=torch.float64).item()))
        )
        reconstruction[name] = {
            "absolute_l2_error": absolute,
            "relative_l2_error": absolute / reference if reference > 0.0 else absolute,
            "allclose_rtol_1e-5_atol_1e-7": bool(
                torch.allclose(summed, total_gradient, rtol=1e-5, atol=1e-7)
            ),
        }
    return {
        "sources": by_source,
        "aggregate_gradient_reconstruction": reconstruction,
    }


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def parameter_gradients_sha256(module: nn.Module) -> str:
    """Hash raw parameter gradients in named order for off/on parity."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8") + b"\0")
        if parameter.grad is None:
            digest.update(b"MISSING\0")
            continue
        tensor = parameter.grad.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def tensor_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().contiguous().cpu()
            digest.update(path.encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key], f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            raise TypeError(f"tensor tree contains unsupported {type(item)!r} at {path}")

    visit(value, "root")
    return digest.hexdigest()


def zero_model_gradients(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _linear_percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    _require(bool(ordered), "percentile requires at least one value")
    _require(0.0 <= fraction <= 1.0, "percentile fraction must lie in [0,1]")
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_c1a_reduction(
    group_norm_pairs: Sequence[Sequence[float]],
    batch_norm_pairs: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Qualify a paired BN1d/GN reduction against two-run runtime variation.

    Every outer item is one frozen B4 batch and every inner item is exactly two
    repeated measurements.  Geometric centres make positive gradient ratios
    symmetric in log space; the effect must clear both an absolute two-fold gate
    and the observed p95 within-method log variation.
    """
    _require(
        len(group_norm_pairs) == len(batch_norm_pairs) and len(group_norm_pairs) > 0,
        "C1-A paired candidates require the same non-empty batch count",
    )
    floor = 1e-30
    ratios: list[float] = []
    within_log_variation: list[float] = []
    for gn_values, bn_values in zip(group_norm_pairs, batch_norm_pairs, strict=True):
        _require(
            len(gn_values) == len(bn_values) == 2,
            "C1-A runtime qualification requires exactly two repeats",
        )
        gn = [max(floor, float(value)) for value in gn_values]
        bn = [max(floor, float(value)) for value in bn_values]
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in (*gn, *bn)),
            "C1-A gradient metrics must be finite and nonnegative",
        )
        gn_center = math.sqrt(gn[0] * gn[1])
        bn_center = math.sqrt(bn[0] * bn[1])
        ratios.append(bn_center / gn_center)
        within_log_variation.extend((abs(math.log(gn[0] / gn[1])), abs(math.log(bn[0] / bn[1]))))

    median_ratio = _linear_percentile(ratios, 0.5)
    favourable_fraction = sum(value <= 0.8 for value in ratios) / len(ratios)
    median_effect_log = _linear_percentile([-math.log(value) for value in ratios], 0.5)
    within_p95 = _linear_percentile(within_log_variation, 0.95)
    stable_material_reduction = bool(
        median_ratio <= 0.5
        and favourable_fraction >= 0.75
        and median_effect_log > within_p95
    )
    return {
        "batch_count": len(ratios),
        "repeats_per_candidate_batch": 2,
        "candidate_over_current_ratio": {
            "min": min(ratios),
            "median": median_ratio,
            "p95": _linear_percentile(ratios, 0.95),
            "max": max(ratios),
        },
        "fraction_candidate_le_0p8_current": favourable_fraction,
        "median_reduction_log": median_effect_log,
        "within_method_abs_log_variation_p95": within_p95,
        "absolute_gate_candidate_over_current_le": 0.5,
        "paired_support_gate_fraction_le_0p8": 0.75,
        "stable_material_reduction": stable_material_reduction,
    }


def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Small dependency-free Spearman coefficient with average tie ranks."""
    _require(len(left) == len(right) and len(left) >= 3, "Spearman inputs must have equal length >= 3")

    def ranks(values: Sequence[float]) -> list[float]:
        indexed = sorted(enumerate(float(value) for value in values), key=lambda item: item[1])
        result = [0.0] * len(indexed)
        start = 0
        while start < len(indexed):
            stop = start + 1
            while stop < len(indexed) and indexed[stop][1] == indexed[start][1]:
                stop += 1
            average = (start + 1 + stop) / 2.0
            for position in range(start, stop):
                result[indexed[position][0]] = average
            start = stop
        return result

    x, y = ranks(left), ranks(right)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    x_sq = sum((a - x_mean) ** 2 for a in x)
    y_sq = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq * y_sq)
    return 0.0 if denominator == 0.0 else numerator / denominator


def classify_c1a_gradient_causality(
    *,
    loss_effects: Mapping[str, Mapping[str, Any]],
    vjp_effects: Mapping[str, Mapping[str, Any]],
    occupancy_correlations: Mapping[str, float],
    loss_upstream_stem_correlation: float,
    current_loss_stem_max_abs_median: float,
) -> dict[str, Any]:
    """Apply the frozen conservative C1-A mechanism precedence.

    A normalization label needs a stable BN1d reduction in two fixed-VJP
    Jacobian metrics and at least one normal-loss metric.  Downstream head/loss
    localization requires a large current stem gradient, no VJP normalization
    support, a strong upstream/stem association, and at least two loss-only BN
    effects.  Occupancy is last and requires the same strong negative association
    in at least three candidate/path combinations.  Everything else is honestly
    inconclusive.
    """
    loss_support = sorted(
        name for name, report in loss_effects.items()
        if bool(report.get("stable_material_reduction", False))
    )
    vjp_support = sorted(
        name for name, report in vjp_effects.items()
        if bool(report.get("stable_material_reduction", False))
    )
    occupancy_support = sorted(
        name for name, value in occupancy_correlations.items()
        if math.isfinite(float(value)) and float(value) <= -0.7
    )

    if len(vjp_support) >= 2 and len(loss_support) >= 1:
        label = "LOCALIZED_NORM"
        basis = "BN1d causally reduces the fixed-upstream SECOND Jacobian and normal-loss gradients beyond runtime variation"
    elif (
        not vjp_support
        and len(loss_support) >= 2
        and float(current_loss_stem_max_abs_median) >= 1e5
        and float(loss_upstream_stem_correlation) >= 0.7
    ):
        label = "LOCALIZED_HEAD_LOSS"
        basis = "large stem gradients track downstream loss upstream-scale while the fixed-upstream encoder Jacobian does not support normalization"
    elif not vjp_support and len(occupancy_support) >= 3:
        label = "LOCALIZED_SPARSE_OCCUPANCY"
        basis = "normalized gradient amplification is strongly and consistently inverse-associated with active sparse occupancy"
    else:
        label = "INCONCLUSIVE"
        basis = "no predeclared mechanism gate clears runtime variation and multi-metric support"

    return {
        "label": label,
        "basis": basis,
        "loss_normalization_support_metrics": loss_support,
        "fixed_vjp_normalization_support_metrics": vjp_support,
        "occupancy_support_cells": occupancy_support,
        "loss_upstream_stem_spearman": float(loss_upstream_stem_correlation),
        "current_loss_stem_max_abs_median": float(current_loss_stem_max_abs_median),
        "thresholds": {
            "normalization_vjp_metric_count": 2,
            "normalization_loss_metric_count": 1,
            "head_loss_current_stem_max_abs": 1e5,
            "head_loss_upstream_stem_spearman": 0.7,
            "occupancy_spearman_max": -0.7,
            "occupancy_support_cell_count": 3,
        },
    }


def validate_c1a_batch_norm_state_mapping(
    module: nn.Module,
    *,
    missing_keys: Sequence[str],
    unexpected_keys: Sequence[str],
    expected_sites: int,
) -> dict[str, Any]:
    """Validate the GN-affine to fresh-BN1d state mapping used by C1-A.

    PyTorch's BatchNorm compatibility loader synthesizes an absent
    ``num_batches_tracked`` buffer and therefore does not include that key in
    ``load_state_dict(..., strict=False).missing_keys``.  Running mean/variance
    remain reported as missing.  Validate those two behaviours independently so
    a compatibility detail cannot masquerade as a candidate-state drift.
    """
    _require(expected_sites > 0, "C1-A BN1d mapping requires at least one site")
    state = module.state_dict()
    running_mean_keys = sorted(name for name in state if name.endswith("running_mean"))
    running_var_keys = sorted(name for name in state if name.endswith("running_var"))
    tracked_keys = sorted(name for name in state if name.endswith("num_batches_tracked"))
    _require(
        len(running_mean_keys) == len(running_var_keys) == len(tracked_keys) == expected_sites,
        "C1-A BN1d running-state site count drift",
    )
    expected_missing = sorted((*running_mean_keys, *running_var_keys))
    _require(
        sorted(str(name) for name in missing_keys) == expected_missing,
        "C1-A BN1d missing-key set must contain only running_mean/running_var",
    )
    _require(not unexpected_keys, f"C1-A BN1d mapping has unexpected keys: {list(unexpected_keys)}")
    _require(
        not set(tracked_keys).intersection(str(name) for name in missing_keys),
        "C1-A synthesized num_batches_tracked must not be reported missing",
    )
    for name in running_mean_keys:
        value = state[name]
        _require(bool(torch.count_nonzero(value).item() == 0), f"C1-A {name} must initialize to zero")
    for name in running_var_keys:
        value = state[name]
        _require(bool(torch.count_nonzero(value - 1).item() == 0), f"C1-A {name} must initialize to one")
    for name in tracked_keys:
        value = state[name]
        _require(value.numel() == 1 and int(value.item()) == 0, f"C1-A {name} must exist and initialize to zero")
    return {
        "batch_norm_sites": expected_sites,
        "reported_missing_running_mean_var": expected_missing,
        "synthesized_num_batches_tracked": tracked_keys,
        "unexpected_keys": [],
        "fresh_running_state_valid": True,
    }


__all__ = [
    "LIDAR_BACKWARD_CHAIN",
    "MAIN_BOUNDARIES",
    "STOP_B_SCHEMA",
    "StopBObservationError",
    "StopBObservationRecorder",
    "attribute_term_gradients",
    "capture_parameter_gradient_tensors",
    "capture_tensor_tree_tensors",
    "classify_stop_b_randomness",
    "classify_c1a_gradient_causality",
    "compare_parameter_gradient_tensors",
    "compare_tensor_tree_tensors",
    "loss_term_snapshot",
    "module_state_sha256",
    "paired_c1a_reduction",
    "parameter_gradients_sha256",
    "parameter_gradient_snapshot",
    "recompose_from_sample_terms",
    "spearman_rank_correlation",
    "strict_json_value",
    "tensor_tree_sha256",
    "term_sources",
    "validate_c1a_batch_norm_state_mapping",
    "zero_model_gradients",
]
