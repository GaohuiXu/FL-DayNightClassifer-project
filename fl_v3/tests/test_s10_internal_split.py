from __future__ import annotations

from collections import Counter
import copy
from itertools import product

import pytest

from fl_v3.data.nuscenes.internal_split import (
    ASSIGNMENT_LEX_BLOCK_SIZE,
    BASE_ROLES,
    DETECTION_NAMES,
    LOCATIONS,
    SCHEMA_MANIFEST,
    SCHEMA_OWNERSHIP,
    SplitContractError,
    _MilpProblem,
    _radix3_weights,
    check_constraints,
    check_ownership,
    solve_split,
)


EXPECTED_BASE_VECTOR = [
    *(["D_fit"] * 16), *(["D_select"] * 3), *(["D_audit"] * 3),
    *(["D_fit"] * 11), *(["D_select"] * 3), *(["D_audit"] * 3),
    *(["D_fit"] * 5), "D_select", "D_audit",
    *(["D_fit"] * 2), "D_select", "D_audit",
]
EXPECTED_LOW = {
    "log-00", "log-01", "log-02", "log-03", "log-04",
    "log-22", "log-23", "log-24", "log-39", "log-46",
}
EXPECTED_MID = {
    "log-00", "log-01", "log-02", "log-03", "log-04", "log-05", "log-06",
    "log-07", "log-08", "log-22", "log-23", "log-24", "log-25", "log-26",
    "log-27", "log-28", "log-39", "log-40", "log-41", "log-46",
}


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


def test_blocked_radix3_is_exactly_lexicographic_and_numerically_bounded():
    assert ASSIGNMENT_LEX_BLOCK_SIZE == 10
    assert _radix3_weights(10)[0] == 19_683
    assert 2 * sum(_radix3_weights(10)) == 59_048 < 2**53
    vectors = list(product(range(3), repeat=5))
    encoded = sorted(
        (sum(digit * weight for digit, weight in zip(vector, _radix3_weights(5))), vector)
        for vector in vectors
    )
    assert [vector for _value, vector in encoded] == sorted(vectors)
    with pytest.raises(SplitContractError):
        _radix3_weights(0)
    with pytest.raises(SplitContractError):
        _radix3_weights(11)


def test_no_seed_milp_is_optimal_nested_and_checker_reconstructs_every_gate(monkeypatch):
    features = _synthetic_features()
    solve_calls = []
    original_solve = _MilpProblem.solve

    def counted_solve(problem, objective):
        solve_calls.append(dict(objective))
        return original_solve(problem, objective)

    monkeypatch.setattr(_MilpProblem, "solve", counted_solve)
    base, low, mid, report = solve_split(features)
    assert len(solve_calls) == 19
    assert report["base"]["status"] == "OPTIMAL"
    assert report["nested"]["status"] == "OPTIMAL"
    assert Counter(base.values()) == {"D_fit": 34, "D_select": 8, "D_audit": 8}
    assert len(low) == 10 and len(mid) == 20 and low < mid
    assert report["base"]["assignment_vector"] == EXPECTED_BASE_VECTOR
    assert low == EXPECTED_LOW
    assert mid == EXPECTED_MID
    assert len(report["base"]["objectives"]) == 55
    assert len(report["nested"]["objectives"]) == 39
    assert check_constraints(features, base, low, mid)["status"] == "PASS"

    solve_calls.clear()
    repeated = solve_split(copy.deepcopy(features))
    assert len(solve_calls) == 19
    assert repeated == (base, low, mid, report)

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
