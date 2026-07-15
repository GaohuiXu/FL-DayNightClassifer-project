#!/usr/bin/env python
"""Combined STOP-A split materialization and exact full-val evaluator parity gate."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.insert(0, SCRIPT_DIR)

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.info_cache import split_sample_tokens
from fl_v3.data.nuscenes.internal_split import sha256_file, write_json
from fl_v3.eval.subset_detection_eval import (
    assert_exact_parity,
    evaluate_subset_tokens,
    filtered_box_identity,
    load_manifest_role,
    official_evaluation_view,
    parity_payload,
    write_strict_json,
)
from s10_materialize_split import run as materialize_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--zip-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--train-cache-hash", required=True)
    parser.add_argument("--train-cache-file-sha256", required=True)
    parser.add_argument("--train-cache-sidecar-sha256", required=True)
    parser.add_argument("--val-cache-hash", required=True)
    parser.add_argument("--val-cache-file-sha256", required=True)
    parser.add_argument("--val-cache-sidecar-sha256", required=True)
    parser.add_argument("--zip-manifest-hash", required=True)
    parser.add_argument("--zip-manifest-file-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-tree", required=True)
    return parser.parse_args()


def _submission_meta():
    return {
        "use_camera": True,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }


def _fixture_results(filtered_gt, kind: str) -> dict:
    if kind not in {"P-GT", "P-MIX"}:
        raise ValueError(kind)
    results = {}
    global_index = 0
    for token in sorted(filtered_gt.sample_tokens):
        rows = []
        for box in filtered_gt[token]:
            base = box.serialize()
            base["velocity"] = [
                float(value) if math.isfinite(float(value)) else 0.0
                for value in base["velocity"]
            ]
            if kind == "P-GT":
                base["detection_score"] = 1.0
                rows.append(base)
            else:
                if global_index % 3 != 0:  # deterministic misses.
                    true_row = copy.deepcopy(base)
                    true_row["detection_score"] = float((0.55, 0.70, 0.70)[global_index % 3])
                    rows.append(true_row)
                    if global_index % 7 == 0:  # equal-score duplicate ordering case.
                        duplicate = copy.deepcopy(true_row)
                        rows.append(duplicate)
                if global_index % 11 == 0:  # survives most range filters but is >4m away.
                    false_row = copy.deepcopy(base)
                    false_row["translation"] = list(false_row["translation"])
                    false_row["translation"][0] = float(false_row["translation"][0] + 5.0)
                    false_row["detection_score"] = 0.70
                    false_row["num_pts"] = -1
                    rows.append(false_row)
                global_index += 1
        if kind == "P-MIX" and int(token[-1], 16) % 2:
            rows.reverse()
        results[token] = rows
    return results


def _write_submission(path: str, results: dict) -> None:
    write_strict_json(path, {"meta": _submission_meta(), "results": results})


def _filter_adversarial_counts(nusc, raw_gt, filtered_gt) -> dict:
    import numpy as np
    from pyquaternion import Quaternion
    from nuscenes.eval.detection.utils import category_to_detection_name
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import points_in_box

    raw_zero = sum(box.num_pts == 0 for box in raw_gt.all)
    if raw_zero <= 0:
        raise AssertionError("official val fixture contains no zero-point GT case")
    bike_rack_cases = 0
    for sample in nusc.sample:
        racks = []
        cycles = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if ann["category_name"] == "static_object.bicycle_rack":
                racks.append(Box(ann["translation"], ann["size"], Quaternion(ann["rotation"])))
            elif category_to_detection_name(ann["category_name"]) in {"bicycle", "motorcycle"}:
                cycles.append(ann)
        if not racks or not cycles:
            continue
        for ann in cycles:
            point = np.expand_dims(np.asarray(ann["translation"], dtype=float), axis=1)
            if any(np.sum(points_in_box(rack, point)) > 0 for rack in racks):
                bike_rack_cases += 1
    if bike_rack_cases <= 0:
        raise AssertionError("trainval metadata contains no bicycle-rack adversarial case")
    return {
        "official_val_raw_boxes": len(raw_gt.all),
        "official_val_raw_zero_point_boxes": int(raw_zero),
        "official_val_filtered_boxes": len(filtered_gt.all),
        "trainval_bicycle_or_motorcycle_centers_in_racks": int(bike_rack_cases),
    }


def _run_parity(nusc, split_dir: str, parity_dir: str, source_identity: dict) -> dict:
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.evaluate import DetectionEval

    os.makedirs(parity_dir, exist_ok=False)
    val_tokens = split_sample_tokens(nusc, "val")
    if len(val_tokens) != 6019:
        raise AssertionError(f"official val token count drift: {len(val_tokens)}")
    split_manifest = json.load(
        open(os.path.join(split_dir, "split_manifest.json"), encoding="utf-8")
    )
    eval_manifest = {
        "schema": "fl_v3.s10.eval_manifest.v1",
        "parent_version": "v1.0-trainval",
        "parent_split": "val",
        "source_identities": split_manifest["source_identities"],
        "roles": {"P_FULL_VAL": {"sample_tokens": val_tokens}},
    }
    eval_manifest_path = os.path.join(parity_dir, "full_val_eval_manifest.json")
    write_json(eval_manifest_path, eval_manifest)
    eval_manifest_sha = sha256_file(eval_manifest_path)
    role = load_manifest_role(
        eval_manifest_path,
        "P_FULL_VAL",
        expected_manifest_sha256=eval_manifest_sha,
        expected_parent_version="v1.0-trainval",
        expected_parent_split="val",
        expected_source_identities=split_manifest["source_identities"],
    )

    cfg = config_factory("detection_cvpr_2019")
    raw_gt = load_gt(nusc, "val", DetectionBox, verbose=False)
    raw_gt = add_center_dist(nusc, raw_gt)
    raw_gt_identity = filtered_box_identity(raw_gt)
    filter_counts_before = len(raw_gt.all)
    filtered_gt = filter_eval_boxes(nusc, raw_gt, cfg.class_range, verbose=False)
    if len(filtered_gt.all) >= filter_counts_before:
        raise AssertionError("official filtering removed no full-val GT boxes")
    adversarial = _filter_adversarial_counts(nusc, load_gt(nusc, "val", DetectionBox, False), filtered_gt)
    adversarial["official_val_center_distance_raw_identity"] = raw_gt_identity

    fixtures = {}
    for kind in ("P-GT", "P-MIX"):
        fixture_dir = os.path.join(parity_dir, kind.lower().replace("-", "_"))
        os.makedirs(fixture_dir)
        result_path = os.path.join(fixture_dir, "results.json")
        _write_submission(result_path, _fixture_results(filtered_gt, kind))
        official_dir = os.path.join(fixture_dir, "official")
        official = DetectionEval(
            nusc,
            cfg,
            result_path,
            "val",
            official_dir,
            verbose=False,
        )
        official_metrics, official_details = official.evaluate()
        official_view = official_evaluation_view(official, official_metrics, official_details)
        subset = evaluate_subset_tokens(
            nusc,
            result_path,
            parent_split="val",
            sample_tokens=role.sample_tokens,
            manifest_identity={"path": role.path, "sha256": role.sha256, "role": role.role},
        )
        assert_exact_parity(official_view, subset)
        payload = parity_payload(subset)
        if len(payload["metric_data"]) != 40:
            raise AssertionError("expected exactly 40 class-distance metric-data records")
        for key, arrays in payload["metric_data"].items():
            if any(len(values) != 101 for values in arrays.values()):
                raise AssertionError(f"{key} contains a non-101-point official metric array")
        fixture_report = {
            "fixture": kind,
            "status": "EXACT_PARITY",
            "tolerance": 0,
            "result_sha256": sha256_file(result_path),
            "filtered_gt_identity": payload["filtered_gt_identity"],
            "filtered_prediction_identity": payload["filtered_prediction_identity"],
            "metrics": payload["metrics"],
            "metrics_validity": payload["metrics_validity"],
            "metric_data_sha256": hashlib.sha256(
                json.dumps(payload["metric_data"], sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "metric_data_validity_sha256": hashlib.sha256(
                json.dumps(
                    payload["metric_data_validity"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        }
        write_strict_json(os.path.join(fixture_dir, "parity.json"), fixture_report)
        fixtures[kind] = fixture_report

    empty_tokens = role.sample_tokens[:64]
    empty_path = os.path.join(parity_dir, "empty_results.json")
    _write_submission(empty_path, {token: [] for token in empty_tokens})
    empty = evaluate_subset_tokens(
        nusc,
        empty_path,
        parent_split="val",
        sample_tokens=empty_tokens,
        manifest_identity={"role": "P_EMPTY_64", "source": eval_manifest_sha},
    )
    if not empty.report["empty_prediction_filter_adapter"]:
        raise AssertionError("all-empty prediction adapter was not recorded")
    if empty.report["internal_subset_NDS"] != 0.0 or empty.report["internal_subset_mAP"] != 0.0:
        raise AssertionError("all-empty official accumulation is not exact zero")
    write_strict_json(os.path.join(parity_dir, "empty_eval.json"), empty.report)
    return {
        "schema": "fl_v3.s10.stop_a_parity.v1",
        "status": "PASS",
        "source_identity": source_identity,
        "eval_manifest_sha256": eval_manifest_sha,
        "fixtures": fixtures,
        "empty_adapter": {
            "samples": len(empty_tokens),
            "mAP": empty.report["internal_subset_mAP"],
            "NDS": empty.report["internal_subset_NDS"],
            "adapter": True,
        },
        "adversarial_filter_cases": adversarial,
    }


def main() -> None:
    args = parse_args()
    if os.path.exists(args.output_dir):
        raise FileExistsError(f"fresh STOP-A output required: {args.output_dir}")
    os.makedirs(args.output_dir)
    split_dir = os.path.join(args.output_dir, "split")
    os.makedirs(split_dir)
    materialize_args = copy.copy(args)
    materialize_args.output_dir = split_dir
    split_result = materialize_split(materialize_args)
    source_identity = {"source_sha": args.source_sha, "source_tree": args.source_tree}
    parity = _run_parity(
        P.create_nuscenes(args.version, args.dataroot, verbose=False),
        split_dir,
        os.path.join(args.output_dir, "parity"),
        source_identity,
    )
    gate = {
        "schema": "fl_v3.s10.stop_a_gate.v1",
        "status": "PASS",
        "source_identity": source_identity,
        "split": split_result,
        "parity": parity,
        "scientific_interpretation": "split/evaluator engineering gate only; no model result",
    }
    write_strict_json(os.path.join(args.output_dir, "stop_a_gate.json"), gate)
    manifest_tmp = os.path.join(args.output_dir, ".artifact_sha256s.tmp")
    artifact_paths = []
    for root, dirs, files in os.walk(args.output_dir):
        dirs.sort()
        for filename in sorted(files):
            path = os.path.join(root, filename)
            if path != manifest_tmp:
                artifact_paths.append(path)
    with open(manifest_tmp, "w", encoding="utf-8") as stream:
        for path in sorted(artifact_paths):
            relative = os.path.relpath(path, args.output_dir)
            stream.write(f"{sha256_file(path)}  {relative}\n")
    os.replace(manifest_tmp, os.path.join(args.output_dir, "artifact_sha256s.txt"))
    print(json.dumps(gate, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
