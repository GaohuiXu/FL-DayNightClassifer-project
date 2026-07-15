from __future__ import annotations

from collections import Counter
import copy

import pytest

from fl_v3.data.nuscenes.internal_split import (
    BASE_ROLES,
    DETECTION_NAMES,
    LOCATIONS,
    SCHEMA_MANIFEST,
    SCHEMA_OWNERSHIP,
    SplitContractError,
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


def test_no_seed_milp_is_optimal_nested_and_checker_reconstructs_every_gate():
    features = _synthetic_features()
    base, low, mid, report = solve_split(features)
    assert report["base"]["status"] == "OPTIMAL"
    assert report["nested"]["status"] == "OPTIMAL"
    assert Counter(base.values()) == {"D_fit": 34, "D_select": 8, "D_audit": 8}
    assert len(low) == 10 and len(mid) == 20 and low < mid
    assert check_constraints(features, base, low, mid)["status"] == "PASS"

    corrupted = copy.deepcopy(base)
    first_select = next(log for log, role in corrupted.items() if role == "D_select")
    first_fit = next(log for log, role in corrupted.items() if role == "D_fit")
    corrupted[first_select], corrupted[first_fit] = corrupted[first_fit], corrupted[first_select]
    with pytest.raises(SplitContractError):
        check_constraints(features, corrupted, low, mid)


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
