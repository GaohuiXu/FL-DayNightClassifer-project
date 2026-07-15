from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from fl_v3.eval.subset_detection_eval import (
    ManifestEvaluationError,
    assert_exact_parity,
    evaluate_subset_tokens,
    load_manifest_role,
    official_evaluation_view,
    strict_load_json,
    write_strict_json,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_submission(path, results):
    write_strict_json(
        str(path),
        {
            "meta": {
                "use_camera": True,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": results,
        },
    )


def test_manifest_identity_and_duplicate_json_keys_fail_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_strict_json(
        str(manifest),
        {
            "schema": "fl_v3.s10.eval_manifest.v1",
            "parent_version": "v1.0-trainval",
            "parent_split": "train",
            "source_identities": {"cache": "abc"},
            "roles": {"D_select": {"sample_tokens": ["a", "b"]}},
        },
    )
    role = load_manifest_role(
        str(manifest),
        "D_select",
        expected_manifest_sha256=_sha(manifest),
        expected_parent_version="v1.0-trainval",
        expected_parent_split="train",
        expected_source_identities={"cache": "abc"},
    )
    assert role.sample_tokens == ("a", "b")
    with pytest.raises(ManifestEvaluationError, match="physical SHA mismatch"):
        load_manifest_role(
            str(manifest),
            "D_select",
            expected_manifest_sha256="0" * 64,
            expected_parent_version="v1.0-trainval",
            expected_parent_split="train",
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"results": {}, "results": {}}', encoding="utf-8")
    with pytest.raises(ManifestEvaluationError, match="duplicate JSON object key"):
        strict_load_json(str(duplicate))


def _filtered_gt_submission(nusc_mini, parent_split):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox

    cfg = config_factory("detection_cvpr_2019")
    boxes = load_gt(nusc_mini, parent_split, DetectionBox, verbose=False)
    boxes = add_center_dist(nusc_mini, boxes)
    boxes = filter_eval_boxes(nusc_mini, boxes, cfg.class_range, verbose=False)
    results = {}
    for token in sorted(boxes.sample_tokens):
        rows = []
        for index, box in enumerate(boxes[token]):
            row = box.serialize()
            row["velocity"] = [
                float(value) if math.isfinite(float(value)) else 0.0
                for value in row["velocity"]
            ]
            row["detection_score"] = float(0.75 if index % 2 else 0.5)
            rows.append(row)
        results[token] = rows
    return results


def test_subset_path_has_exact_official_parity_on_full_mini_val(nusc_mini, tmp_path):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval

    results = _filtered_gt_submission(nusc_mini, "mini_val")
    result_path = tmp_path / "result.json"
    _write_submission(result_path, results)
    official_dir = tmp_path / "official"
    official = DetectionEval(
        nusc_mini,
        config_factory("detection_cvpr_2019"),
        str(result_path),
        "mini_val",
        str(official_dir),
        verbose=False,
    )
    official_metrics, official_details = official.evaluate()
    subset = evaluate_subset_tokens(
        nusc_mini,
        str(result_path),
        parent_split="val" if nusc_mini.version.endswith("trainval") else "mini_val",
        sample_tokens=sorted(results),
        manifest_identity={"test": "full-mini-val"},
    )
    assert_exact_parity(
        official_evaluation_view(official, official_metrics, official_details), subset
    )


def test_empty_predictions_are_zero_strict_json_and_use_only_local_adapter(
    nusc_mini, tmp_path
):
    from fl_v3.data.nuscenes.info_cache import split_sample_tokens

    tokens = split_sample_tokens(nusc_mini, "mini_val")[:16]
    result_path = tmp_path / "empty.json"
    _write_submission(result_path, {token: [] for token in tokens})
    evaluation = evaluate_subset_tokens(
        nusc_mini,
        str(result_path),
        parent_split="mini_val",
        sample_tokens=tokens,
        manifest_identity={"test": "empty"},
    )
    assert evaluation.report["empty_prediction_filter_adapter"] is True
    assert evaluation.report["internal_subset_mAP"] == 0.0
    assert evaluation.report["internal_subset_NDS"] == 0.0
    json.dumps(evaluation.report, allow_nan=False)


def test_submission_missing_extra_and_miskeyed_tokens_fail(nusc_mini, tmp_path):
    all_results = _filtered_gt_submission(nusc_mini, "mini_val")
    first = next(token for token in sorted(all_results) if all_results[token])
    second = next(token for token in sorted(all_results) if token != first)
    tokens = sorted((first, second))
    result_path = tmp_path / "bad.json"
    _write_submission(result_path, {tokens[0]: []})
    with pytest.raises(ManifestEvaluationError, match="token set differs"):
        evaluate_subset_tokens(
            nusc_mini,
            str(result_path),
            parent_split="mini_val",
            sample_tokens=tokens,
            manifest_identity={"test": "missing"},
        )
    source = first
    row = all_results[source][0]
    row = copy.deepcopy(row)
    row["sample_token"] = "outside"
    _write_submission(
        result_path,
        {token: ([row] if token == source else []) for token in tokens},
    )
    with pytest.raises(ManifestEvaluationError, match="mis-keyed box"):
        evaluate_subset_tokens(
            nusc_mini,
            str(result_path),
            parent_split="mini_val",
            sample_tokens=tokens,
            manifest_identity={"test": "miskeyed"},
        )
