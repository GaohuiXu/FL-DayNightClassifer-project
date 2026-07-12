"""S05 canonical local/global/eval and official submission conversion fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from fl_v3.eval.box_to_global import (
    NUSCENES_MAX_BOXES_PER_SAMPLE,
    convert_box_to_global,
    decoded_sample_to_boxes,
)
from fl_v3.eval.detection_eval import SampleDecode, build_results_dict
from fl_v3.models.fusion.head import NUSCENES_DETECTION_NAMES


def _sample(decoded):
    return SampleDecode(
        sample_token="sample-token",
        boxes=np.asarray(decoded["boxes"], dtype=np.float64).reshape(-1, 7),
        scores=np.asarray(decoded["scores"], dtype=np.float64).reshape(-1),
        labels=np.asarray(decoded["labels"], dtype=np.int64).reshape(-1),
        velocity=np.asarray(decoded["velocity"], dtype=np.float64).reshape(-1, 2),
        lidar2ego=np.eye(4),
        ego2global_lidar=np.eye(4),
        lidar2img=np.zeros((6, 4, 4)),
        gt_boxes=np.zeros((0, 7)),
        gt_labels=np.zeros((0,), dtype=np.int64),
        gt_num_lidar_pts=np.zeros((0,), dtype=np.int64),
        gt_in_range=np.zeros((0,), dtype=bool),
        gt_velocity=np.zeros((0, 2)),
    )


def test_identity_local_to_global_preserves_center_yaw_velocity_and_swaps_lw_for_wlh():
    box = np.array([4.5, -3.0, 1.25, 4.2, 1.8, 1.6, -2.4])
    velocity = np.array([3.0, -1.5])
    converted = convert_box_to_global(box, np.eye(4), np.eye(4), velocity)
    assert np.allclose(converted["translation"], box[:3])
    assert np.allclose(converted["size"], [box[4], box[3], box[5]])
    assert np.allclose(converted["velocity"], velocity)
    assert abs(np.arctan2(np.sin(converted["yaw_global"] - box[6]),
                          np.cos(converted["yaw_global"] - box[6]))) < 1e-12


def test_submission_uses_canonical_global_ids_for_classes_whose_task_offsets_differ():
    labels = np.array([4, 2, 9, 5, 8], dtype=np.int64)
    expected_names = {
        "construction_vehicle", "bus", "barrier", "pedestrian", "traffic_cone"
    }
    boxes = np.array([
        [1.0 + i * 3.0, 2.0, 0.5, 2.0, 1.0, 1.5, 0.1 * i]
        for i in range(labels.size)
    ])
    decoded = {
        "boxes": boxes,
        "scores": np.full(labels.size, 0.75),
        "labels": labels,
        "velocity": np.zeros((labels.size, 2)),
    }
    submission = build_results_dict(
        [_sample(decoded)], NUSCENES_DETECTION_NAMES, ["sample-token"]
    )
    records = submission["results"]["sample-token"]
    assert {record["detection_name"] for record in records} == expected_names
    assert all(record["sample_token"] == "sample-token" for record in records)
    assert all(type(record["detection_score"]) is float for record in records)
    assert all(len(record["translation"]) == 3 for record in records)
    assert all(len(record["size"]) == 3 for record in records)
    assert all(len(record["rotation"]) == 4 for record in records)
    assert all(len(record["velocity"]) == 2 for record in records)


def test_submission_conversion_is_content_permutation_invariant():
    decoded = {
        "boxes": np.array([
            [4.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0],
            [1.0, 0.0, 0.5, 2.0, 1.0, 1.0, 0.5],
            [3.0, 0.0, 0.5, 3.0, 1.5, 1.2, -0.5],
        ]),
        "scores": np.array([0.5, 0.5, 0.5]),
        "labels": np.array([0, 9, 4]),
        "velocity": np.zeros((3, 2)),
    }
    forward = build_results_dict([_sample(decoded)], NUSCENES_DETECTION_NAMES, ["sample-token"])
    reverse_decoded = {key: value[::-1].copy() for key, value in decoded.items()}
    reverse = build_results_dict(
        [_sample(reverse_decoded)], NUSCENES_DETECTION_NAMES, ["sample-token"]
    )
    assert forward == reverse


def test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content():
    decoded = {
        "boxes": np.array([
            [4.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0],
            [4.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0],
        ]),
        "scores": np.array([0.5, 0.5]),
        "labels": np.array([0, 0]),
        "velocity": np.array([[5.0, 0.0], [0.0, 0.0]]),
    }
    forward = build_results_dict(
        [_sample(decoded)], NUSCENES_DETECTION_NAMES, ["sample-token"]
    )
    reverse_decoded = {key: value[::-1].copy() for key, value in decoded.items()}
    reverse = build_results_dict(
        [_sample(reverse_decoded)], NUSCENES_DETECTION_NAMES, ["sample-token"]
    )
    assert forward == reverse
    records = forward["results"]["sample-token"]
    assert [(record["velocity"], record["attribute_name"]) for record in records] == [
        ((0.0, 0.0), "vehicle.parked"),
        ((5.0, 0.0), "vehicle.moving"),
    ]


def test_submission_fails_closed_above_official_500_box_cap():
    count = NUSCENES_MAX_BOXES_PER_SAMPLE + 1
    decoded = {
        "boxes": np.tile(np.array([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0]]), (count, 1)),
        "scores": np.linspace(0.9, 0.2, count),
        "labels": np.zeros(count, dtype=np.int64),
        "velocity": np.zeros((count, 2)),
    }
    with pytest.raises(ValueError, match="at most 500"):
        decoded_sample_to_boxes(
            decoded, np.eye(4), np.eye(4), "sample-token", NUSCENES_DETECTION_NAMES
        )


def test_submission_fails_closed_on_bad_label_shape_and_duplicate_sample():
    base = {
        "boxes": np.array([[1.0, 2.0, 0.5, 4.0, 2.0, 1.5, 0.0]]),
        "scores": np.array([0.8]),
        "labels": np.array([10]),
        "velocity": np.zeros((1, 2)),
    }
    with pytest.raises(ValueError, match="outside canonical class range"):
        decoded_sample_to_boxes(base, np.eye(4), np.eye(4), "sample-token", NUSCENES_DETECTION_NAMES)

    good = dict(base, labels=np.array([0]))
    sample = _sample(good)
    with pytest.raises(ValueError, match="decoded more than once"):
        build_results_dict([sample, sample], NUSCENES_DETECTION_NAMES, ["sample-token"])


def test_canonical_conversion_rejects_nonpositive_or_nonfinite_geometry():
    with pytest.raises(ValueError, match="positive"):
        convert_box_to_global([0, 0, 0, 0, 1, 1, 0], np.eye(4), np.eye(4))
    with pytest.raises(ValueError, match="finite"):
        convert_box_to_global([np.nan, 0, 0, 1, 1, 1, 0], np.eye(4), np.eye(4))
