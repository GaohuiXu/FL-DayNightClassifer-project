from __future__ import annotations

from collections import Counter

import pytest

from fl_v3.data.nuscenes import internal_split as split_mod
from fl_v3.data.nuscenes.internal_split import (
    BASE_ROLES,
    DETECTION_NAMES,
    LOCATIONS,
    SCHEMA_MANIFEST,
    SCHEMA_OWNERSHIP,
    SELECTION_POLICY,
    SplitContractError,
    _MilpProblem,
    check_constraints,
    check_ownership,
    solve_split,
)


def _synthetic_features():
    location_counts = dict(zip(LOCATIONS, (22, 17, 7, 4)))
    locations = [location for location in LOCATIONS for _ in range(location_counts[location])]
    # Exactly 28,130 samples while keeping every log nearly equal.  Thus every
    # quota-compatible role automatically lies inside the frozen sample bands.
    sample_counts = [563] * 30 + [562] * 20
    out = []
    for index, (location, n_samples) in enumerate(zip(locations, sample_counts)):
        token = f"log-{index:02d}"
        out.append(
            {
                "log_token": token,
                "location": location,
                "sample_tokens": [f"sample-{index:02d}-{j:03d}" for j in range(n_samples)],
                "scene_tokens": [f"scene-{index:02d}-{j}" for j in range(3)],
                "n_samples": n_samples,
                "n_scenes": 3,
                "condition_scenes": {"day_dry": 2, "night_dry": 1},
                "training_support": {
                    "definition": "cache_gt_in_range",
                    "positive_frames": {name: 100 for name in DETECTION_NAMES},
                    "boxes": {name: 200 for name in DETECTION_NAMES},
                    "positive_scenes": {name: 3 for name in DETECTION_NAMES},
                },
                "evaluation_support": {
                    "definition": "official_load_gt_add_center_dist_filter_eval_boxes",
                    "positive_frames": {name: 100 for name in DETECTION_NAMES},
                    "eligible_boxes": {name: 200 for name in DETECTION_NAMES},
                    "positive_scenes": {name: 3 for name in DETECTION_NAMES},
                },
            }
        )
    return out


def test_one_shot_feasibility_milps_are_nested_and_checker_reconstructs_every_gate(monkeypatch):
    features = _synthetic_features()
    solve_calls = []
    original_solve = _MilpProblem.solve

    def counted_solve(problem, objective):
        solve_calls.append(dict(objective))
        return original_solve(problem, objective)

    monkeypatch.setattr(_MilpProblem, "solve", counted_solve)
    base, low, mid, report = solve_split(features)
    assert solve_calls == [{}, {}]
    assert report["selection_policy"] == SELECTION_POLICY
    assert report["real_candidate_policy"] == "exactly_one"
    assert report["optimization_claim"] == "none"
    assert report["base"]["status"] == "FEASIBLE_FROZEN"
    assert report["nested"]["status"] == "FEASIBLE_FROZEN"
    assert report["base"]["objective_value"] == 0.0
    assert report["nested"]["objective_value"] == 0.0
    assert report["base"]["optimization_claim"] == "none"
    assert report["nested"]["optimization_claim"] == "none"
    assert Counter(base.values()) == {"D_fit": 34, "D_select": 8, "D_audit": 8}
    assert len(low) == 10 and len(mid) == 20 and low < mid
    assert check_constraints(features, base, low, mid)["status"] == "PASS"

    corrupted = dict(base)
    location_by_log = {record["log_token"]: record["location"] for record in features}
    first_select = next(log for log, role in corrupted.items() if role == "D_select")
    first_fit = next(
        log
        for log, role in corrupted.items()
        if role == "D_fit" and location_by_log[log] != location_by_log[first_select]
    )
    corrupted[first_select], corrupted[first_fit] = corrupted[first_fit], corrupted[first_select]
    with pytest.raises(SplitContractError):
        check_constraints(features, corrupted, low, mid)


def test_artifact_reload_binds_pre_solve_features_and_one_candidate(tmp_path, monkeypatch):
    features = _synthetic_features()
    source_identities = {"source_sha": "a" * 40, "cache_sha256": "b" * 64}
    feature_path = tmp_path / "log_features.jsonl"
    split_mod.write_jsonl(str(feature_path), features)
    split_mod.write_jsonl(str(tmp_path / "sample_ownership.jsonl"), [])
    split_mod.write_json(
        str(tmp_path / "pre_solve_identity.json"),
        {
            "schema": "fl_v3.s10.pre_solve_identity.v1",
            "source_identities": source_identities,
            "selection_policy": SELECTION_POLICY,
            "real_candidate_ordinal": 1,
            "reroll_allowed": False,
            "feature_count": 50,
            "train_sample_count": 28130,
            "official_val_sample_count": 6019,
            "log_features_sha256": split_mod.sha256_file(str(feature_path)),
        },
    )
    solver_phase = {
        "status": "FEASIBLE_FROZEN",
        "objective": "constant_zero",
        "objective_value": 0.0,
    }
    split_mod.write_json(
        str(tmp_path / "split_protocol.json"),
        {
            "schema": split_mod.SCHEMA_PROTOCOL,
            "source_identities": source_identities,
            "selection_policy": SELECTION_POLICY,
            "optimization_claim": "none",
            "solver_report": {
                "selection_policy": SELECTION_POLICY,
                "real_candidate_policy": "exactly_one",
                "optimization_claim": "none",
                "base": solver_phase,
                "nested": solver_phase,
            },
        },
    )
    split_mod.write_json(
        str(tmp_path / "split_manifest.json"),
        {
            "schema": SCHEMA_MANIFEST,
            "source_identities": source_identities,
            "roles": {
                "D_fit": {"log_tokens": [f"log-{index:02d}" for index in range(34)]},
                "D_select": {"log_tokens": [f"log-{index:02d}" for index in range(34, 42)]},
                "D_audit": {"log_tokens": [f"log-{index:02d}" for index in range(42, 50)]},
                "D_low": {"log_tokens": [f"log-{index:02d}" for index in range(10)]},
                "D_mid": {"log_tokens": [f"log-{index:02d}" for index in range(20)]},
            },
        },
    )
    monkeypatch.setattr(split_mod, "check_constraints", lambda *_args: {"status": "PASS"})
    monkeypatch.setattr(split_mod, "check_ownership", lambda *_args: {"status": "PASS"})
    assert split_mod.verify_artifact_directory(str(tmp_path))["status"] == "PASS"

    features[0]["n_samples"] -= 1
    split_mod.write_jsonl(str(feature_path), features)
    with pytest.raises(SplitContractError, match="train-sample count drift|feature identity mismatch"):
        split_mod.verify_artifact_directory(str(tmp_path))


def _ownership_rows(manifest):
    rows = []
    for role in BASE_ROLES:
        token = manifest["roles"][role]["sample_tokens"][0]
        rows.append(
            {
                "schema": SCHEMA_OWNERSHIP,
                "parent_split": "train",
                "owner": role,
                "in_D_low": role == "D_fit",
                "in_D_mid": role == "D_fit",
                "log_token": f"log-{role}",
                "scene_token": f"scene-{role}",
                "sample_token": token,
                "annotation_tokens": [f"ann-{role}"],
                "instance_tokens": [f"instance-{role}"],
                "camera_paths": [f"samples/cam-{role}-{index}.jpg" for index in range(6)],
                "key_lidar_path": f"samples/lidar-{role}.bin",
                "lidar_sweep_paths": [f"sweeps/lidar-{role}.bin"],
            }
        )
    # The real checker freezes official val at 6,019 records.
    for index in range(6019):
        rows.append(
            {
                "schema": SCHEMA_OWNERSHIP,
                "parent_split": "val",
                "owner": "official_val",
                "in_D_low": False,
                "in_D_mid": False,
                "log_token": f"val-log-{index}",
                "scene_token": f"val-scene-{index}",
                "sample_token": f"val-sample-{index}",
                "annotation_tokens": [f"val-ann-{index}"],
                "instance_tokens": [f"val-instance-{index}"],
                "camera_paths": [f"val/cam-{index}-{camera}.jpg" for camera in range(6)],
                "key_lidar_path": f"val/lidar-{index}.bin",
                "lidar_sweep_paths": [f"val/sweep-{index}.bin"],
            }
        )
    return rows


def test_ownership_checker_rejects_cross_role_raw_dependency():
    manifest = {
        "schema": SCHEMA_MANIFEST,
        "roles": {
            "D_fit": {"sample_tokens": ["sample-fit"]},
            "D_select": {"sample_tokens": ["sample-select"]},
            "D_audit": {"sample_tokens": ["sample-audit"]},
        },
    }
    rows = _ownership_rows(manifest)
    assert check_ownership(rows, manifest)["status"] == "PASS"
    rows[1]["lidar_sweep_paths"] = [rows[0]["key_lidar_path"]]
    with pytest.raises(SplitContractError, match="cross-owner leakage"):
        check_ownership(rows, manifest)
