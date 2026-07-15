"""Manifest-bound train-subset nuScenes detection evaluation.

This module is deliberately separate from :mod:`fl_v3.eval.detection_eval`.
The latter remains the unchanged official ``v1.0-trainval -> val`` path.  This
adapter evaluates a declared subset of an official parent split while retaining
the devkit's ground-truth construction, filtering, matching and metric math.

The installed nuScenes 1.1.11 ``filter_eval_boxes`` cannot infer a box class
when *all* predictions are empty.  In that one case the prediction-side filter
is skipped: there are no boxes to filter, and the official ``accumulate`` path
then returns its canonical all-zero prediction curves.  Ground truth is always
filtered by the official devkit path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
import time
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


DETECTION_CONFIG_SHA256 = (
    "217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b"
)
SUPPORTED_PARENT_SPLITS = frozenset({"train", "val", "mini_train", "mini_val"})


class ManifestEvaluationError(ValueError):
    """Fail-closed manifest/submission/evaluator contract violation."""


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ManifestEvaluationError(f"{label} must be a 64-character SHA-256 digest")
    return digest


def _no_duplicate_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ManifestEvaluationError(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def strict_load_json(path: str) -> Any:
    """Load JSON while rejecting duplicate keys and non-standard constants."""

    def reject_constant(value: str):
        raise ManifestEvaluationError(f"non-finite JSON constant {value!r} is forbidden")

    with open(path, encoding="utf-8") as stream:
        return json.load(
            stream,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=reject_constant,
        )


def write_strict_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _finite_tree(value: Any) -> Tuple[Any, Any]:
    """Return ``(JSON value, numeric-validity mask)`` without NaN/Inf.

    The mask mirrors mappings/lists.  Numeric leaves are booleans; non-numeric
    leaves use ``None`` because validity is not applicable to them.
    """

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - production dependency
        np = None

    if isinstance(value, Mapping):
        clean, mask = {}, {}
        for key, item in value.items():
            clean[str(key)], mask[str(key)] = _finite_tree(item)
        return clean, mask
    if isinstance(value, (list, tuple)):
        pairs = [_finite_tree(item) for item in value]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    if np is not None and isinstance(value, np.ndarray):
        return _finite_tree(value.tolist())
    if np is not None and isinstance(value, np.generic):
        return _finite_tree(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value, None
    if isinstance(value, int):
        return value, True
    if isinstance(value, float):
        valid = math.isfinite(value)
        return (value if valid else None), valid
    raise TypeError(f"unsupported strict-JSON value {type(value).__name__}")


def _config_file_path() -> str:
    from nuscenes.eval.common.config import config_factory

    # Resolve from the module rather than assuming a site-packages root.
    cfg = config_factory("detection_cvpr_2019")
    module = __import__("nuscenes.eval.detection", fromlist=["__file__"])
    path = os.path.join(
        os.path.dirname(module.__file__), "configs", "detection_cvpr_2019.json"
    )
    # Keep the factory call live: it also establishes that the installed devkit
    # can deserialize the config whose bytes are bound below.
    if tuple(float(x) for x in cfg.dist_ths) != (0.5, 1.0, 2.0, 4.0):
        raise ManifestEvaluationError("unexpected installed detection config semantics")
    return path


def bound_detection_config():
    """Return the official config only if its exact installed bytes are frozen."""
    from nuscenes.eval.common.config import config_factory

    path = _config_file_path()
    actual = sha256_file(path)
    if actual != DETECTION_CONFIG_SHA256:
        raise ManifestEvaluationError(
            "detection_cvpr_2019.json identity drift: "
            f"expected={DETECTION_CONFIG_SHA256}, actual={actual}, path={path}"
        )
    return config_factory("detection_cvpr_2019"), os.path.abspath(path)


@dataclass(frozen=True)
class ManifestRole:
    path: str
    sha256: str
    parent_version: str
    parent_split: str
    role: str
    sample_tokens: Tuple[str, ...]
    source_identities: Dict[str, Any]


def load_manifest_role(
    manifest_path: str,
    role: str,
    *,
    expected_manifest_sha256: str,
    expected_parent_version: str,
    expected_parent_split: str,
    expected_source_identities: Mapping[str, str] | None = None,
) -> ManifestRole:
    """Load one role from an immutable S10 split/evaluation manifest."""
    if expected_parent_split not in SUPPORTED_PARENT_SPLITS:
        raise ManifestEvaluationError(
            f"unsupported parent split {expected_parent_split!r}"
        )
    expected_digest = _required_sha256(expected_manifest_sha256, "manifest SHA-256")
    actual_digest = sha256_file(manifest_path)
    if actual_digest != expected_digest:
        raise ManifestEvaluationError(
            f"manifest physical SHA mismatch: expected={expected_digest}, actual={actual_digest}"
        )
    manifest = strict_load_json(manifest_path)
    if manifest.get("schema") not in {
        "fl_v3.s10.split_manifest.v1",
        "fl_v3.s10.eval_manifest.v1",
    }:
        raise ManifestEvaluationError(f"unsupported manifest schema {manifest.get('schema')!r}")
    if manifest.get("parent_version") != expected_parent_version:
        raise ManifestEvaluationError("manifest parent_version identity drift")
    if manifest.get("parent_split") != expected_parent_split:
        raise ManifestEvaluationError("manifest parent_split identity drift")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or role not in roles:
        raise ManifestEvaluationError(f"manifest does not declare role {role!r}")
    record = roles[role]
    tokens = record.get("sample_tokens") if isinstance(record, dict) else None
    if not isinstance(tokens, list) or not tokens:
        raise ManifestEvaluationError(f"manifest role {role!r} has no sample_tokens")
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ManifestEvaluationError("manifest sample tokens must be non-empty strings")
    if tokens != sorted(tokens):
        raise ManifestEvaluationError("manifest sample tokens must be sorted")
    if len(tokens) != len(set(tokens)):
        raise ManifestEvaluationError("manifest role contains duplicate sample tokens")
    identities = manifest.get("source_identities")
    if not isinstance(identities, dict):
        raise ManifestEvaluationError("manifest has no source_identities mapping")
    for key, expected in (expected_source_identities or {}).items():
        if identities.get(key) != expected:
            raise ManifestEvaluationError(
                f"manifest source identity drift for {key}: "
                f"expected={expected!r}, actual={identities.get(key)!r}"
            )
    return ManifestRole(
        path=os.path.abspath(manifest_path),
        sha256=actual_digest,
        parent_version=expected_parent_version,
        parent_split=expected_parent_split,
        role=role,
        sample_tokens=tuple(tokens),
        source_identities=dict(identities),
    )


def _load_prediction_strict(result_path: str, sample_tokens: Sequence[str], cfg):
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox

    data = strict_load_json(result_path)
    if not isinstance(data, dict) or set(data) < {"meta", "results"}:
        raise ManifestEvaluationError("submission must contain meta and results")
    results = data["results"]
    if not isinstance(results, dict):
        raise ManifestEvaluationError("submission results must be an object")
    if not isinstance(data["meta"], dict):
        raise ManifestEvaluationError("submission meta must be an object")
    expected = set(sample_tokens)
    actual = set(results)
    if actual != expected:
        raise ManifestEvaluationError(
            "submission token set differs from manifest: "
            f"missing={sorted(expected - actual)[:8]}, extra={sorted(actual - expected)[:8]}"
        )
    normalized = {}
    for token in sample_tokens:
        boxes = results[token]
        if not isinstance(boxes, list):
            raise ManifestEvaluationError(f"submission result {token!r} is not a list")
        if len(boxes) > int(cfg.max_boxes_per_sample):
            raise ManifestEvaluationError(
                f"sample {token} has {len(boxes)} boxes; max is {cfg.max_boxes_per_sample}"
            )
        for index, box in enumerate(boxes):
            if not isinstance(box, dict):
                raise ManifestEvaluationError(f"sample {token} box {index} is not an object")
            if box.get("sample_token") != token:
                raise ManifestEvaluationError(
                    f"out-of-manifest/mis-keyed box: key={token}, box.sample_token={box.get('sample_token')!r}"
                )
        normalized[token] = boxes
    try:
        pred = EvalBoxes.deserialize(normalized, DetectionBox)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ManifestEvaluationError(f"invalid detection submission: {exc}") from exc
    return pred, dict(data["meta"])


def _restrict_eval_boxes(eval_boxes, sample_tokens: Sequence[str]):
    from nuscenes.eval.common.data_classes import EvalBoxes

    available = set(eval_boxes.sample_tokens)
    unknown = set(sample_tokens) - available
    if unknown:
        raise ManifestEvaluationError(
            f"manifest tokens are outside the official parent split: {sorted(unknown)[:8]}"
        )
    restricted = EvalBoxes()
    for token in sample_tokens:
        restricted.add_boxes(token, list(eval_boxes[token]))
    return restricted


def _evaluate_official_math(gt_boxes, pred_boxes, cfg):
    """Thin transcription of ``DetectionEval.evaluate`` using public devkit math."""
    import numpy as np
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import DetectionMetricDataList, DetectionMetrics

    started = time.perf_counter()
    metric_data = DetectionMetricDataList()
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metric_data.set(
                class_name,
                dist_th,
                accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th),
            )
    metrics = DetectionMetrics(cfg)
    for class_name in cfg.class_names:
        for dist_th in cfg.dist_ths:
            metrics.add_label_ap(
                class_name,
                dist_th,
                calc_ap(metric_data[(class_name, dist_th)], cfg.min_recall, cfg.min_precision),
            )
        for metric_name in TP_METRICS:
            md = metric_data[(class_name, cfg.dist_th_tp)]
            if class_name == "traffic_cone" and metric_name in {
                "attr_err", "vel_err", "orient_err",
            }:
                value = np.nan
            elif class_name == "barrier" and metric_name in {"attr_err", "vel_err"}:
                value = np.nan
            else:
                value = calc_tp(md, cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, value)
    metrics.add_runtime(time.perf_counter() - started)
    return metrics, metric_data


def filtered_box_identity(eval_boxes) -> Dict[str, Any]:
    """Stable identity/count of filtered boxes, excluding object memory identity."""
    rows = {
        token: [box.serialize() for box in eval_boxes[token]]
        for token in sorted(eval_boxes.sample_tokens)
    }
    clean, validity = _finite_tree(rows)
    payload = json.dumps(
        {"boxes": clean, "numeric_validity": validity},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "samples": len(rows),
        "boxes": sum(len(boxes) for boxes in rows.values()),
    }


@dataclass
class SubsetEvaluation:
    report: Dict[str, Any]
    metrics: Any
    metric_data: Any
    gt_boxes: Any
    pred_boxes: Any


def evaluate_subset_tokens(
    nusc,
    result_path: str,
    *,
    parent_split: str,
    sample_tokens: Sequence[str],
    manifest_identity: Mapping[str, Any],
) -> SubsetEvaluation:
    """Evaluate one exact token set from an official parent split."""
    if parent_split not in SUPPORTED_PARENT_SPLITS:
        raise ManifestEvaluationError(f"unsupported parent split {parent_split!r}")
    tokens = tuple(sample_tokens)
    if not tokens or list(tokens) != sorted(tokens) or len(tokens) != len(set(tokens)):
        raise ManifestEvaluationError("evaluation tokens must be non-empty, unique and sorted")
    cfg, config_path = bound_detection_config()
    pred_boxes, meta = _load_prediction_strict(result_path, tokens, cfg)

    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox

    parent_gt = load_gt(nusc, parent_split, DetectionBox, verbose=False)
    gt_boxes = _restrict_eval_boxes(parent_gt, tokens)
    pred_boxes = add_center_dist(nusc, pred_boxes)
    gt_boxes = add_center_dist(nusc, gt_boxes)

    empty_adapter = len(pred_boxes.all) == 0
    if not empty_adapter:
        pred_boxes = filter_eval_boxes(nusc, pred_boxes, cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, gt_boxes, cfg.class_range, verbose=False)
    metrics, metric_data = _evaluate_official_math(gt_boxes, pred_boxes, cfg)

    metrics_serialized = metrics.serialize()
    metrics_serialized.pop("eval_time", None)
    details_serialized = metric_data.serialize()
    clean_metrics, metric_validity = _finite_tree(metrics_serialized)
    clean_details, detail_validity = _finite_tree(details_serialized)
    report = {
        "schema": "fl_v3.s10.internal_subset_eval.v1",
        "official": False,
        "proxy_only": True,
        "metric_namespace": "internal_subset",
        "internal_subset_NDS": clean_metrics["nd_score"],
        "internal_subset_mAP": clean_metrics["mean_ap"],
        "parent_version": str(nusc.version),
        "parent_split": parent_split,
        "n_samples": len(tokens),
        "submission_meta": meta,
        "manifest_identity": dict(manifest_identity),
        "detection_config": {
            "path": config_path,
            "sha256": DETECTION_CONFIG_SHA256,
        },
        "empty_prediction_filter_adapter": empty_adapter,
        "filtered_gt_identity": filtered_box_identity(gt_boxes),
        "filtered_prediction_identity": filtered_box_identity(pred_boxes),
        "metrics": clean_metrics,
        "metrics_validity": metric_validity,
        "metric_data": clean_details,
        "metric_data_validity": detail_validity,
    }
    # The report itself is the final proof that no non-standard JSON number leaks.
    json.dumps(report, allow_nan=False)
    return SubsetEvaluation(report, metrics, metric_data, gt_boxes, pred_boxes)


def run_internal_manifest_eval(
    nusc,
    result_path: str,
    manifest_path: str,
    role: str,
    output_path: str,
    *,
    expected_manifest_sha256: str,
    expected_parent_version: str = "v1.0-trainval",
    expected_parent_split: str = "train",
    expected_source_identities: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Manifest-bound public entry point for internal train-only proxy metrics."""
    role_record = load_manifest_role(
        manifest_path,
        role,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_parent_version=expected_parent_version,
        expected_parent_split=expected_parent_split,
        expected_source_identities=expected_source_identities,
    )
    result = evaluate_subset_tokens(
        nusc,
        result_path,
        parent_split=role_record.parent_split,
        sample_tokens=role_record.sample_tokens,
        manifest_identity={
            "path": role_record.path,
            "sha256": role_record.sha256,
            "role": role_record.role,
            "source_identities": role_record.source_identities,
        },
    )
    write_strict_json(output_path, result.report)
    return result.report


def parity_payload(evaluation: SubsetEvaluation) -> Dict[str, Any]:
    """Exact-comparison payload excluding runtime and filesystem path metadata."""
    metrics = evaluation.metrics.serialize()
    metrics.pop("eval_time", None)
    clean_metrics, metrics_validity = _finite_tree(metrics)
    clean_details, details_validity = _finite_tree(evaluation.metric_data.serialize())
    return {
        "filtered_gt_identity": filtered_box_identity(evaluation.gt_boxes),
        "filtered_prediction_identity": filtered_box_identity(evaluation.pred_boxes),
        "metrics": clean_metrics,
        "metrics_validity": metrics_validity,
        "metric_data": clean_details,
        "metric_data_validity": details_validity,
    }


def assert_exact_parity(left: SubsetEvaluation, right: SubsetEvaluation) -> None:
    """Require tolerance-zero equality for boxes, arrays, validity and metrics."""
    a = parity_payload(left)
    b = parity_payload(right)
    if a != b:
        sections = [key for key in a if a[key] != b.get(key)]
        raise AssertionError(f"subset evaluator parity mismatch in {sections}")


def official_evaluation_view(nusc_eval, metrics, metric_data) -> SubsetEvaluation:
    """Wrap an unchanged official ``DetectionEval`` result for parity comparison."""
    return SubsetEvaluation(
        report={},
        metrics=metrics,
        metric_data=metric_data,
        gt_boxes=nusc_eval.gt_boxes,
        pred_boxes=nusc_eval.pred_boxes,
    )
