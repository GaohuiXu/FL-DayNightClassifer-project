from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from fl_v3.config import load_resolved_config
from fl_v3.data.nuscenes.augment import (
    apply_reference_post_filters,
    sample_transform,
)
from fl_v3.data.nuscenes.gt_database import load_phase1_gt_database
from fl_v3.data.nuscenes.phase1 import phase1_augmentation_parameters
from fl_v3.phase1_sampling import build_official_cbgs_indices
from fl_v3.training.phase1 import Phase1CyclicScheduler, build_phase1_optimizer


ROOT = Path(__file__).resolve().parents[1]
CAMERA = ROOT / "configs" / "s10_phase1_camera.json"
LIDAR = ROOT / "configs" / "s10_phase1_lidar.json"


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def test_reference_augmentation_rng_order_and_both_randomflip3d_axes():
    config = load_resolved_config(LIDAR)
    params = phase1_augmentation_parameters(config)
    assert params["horizontal_flip_probability"] == 0.5
    assert params["vertical_flip_probability"] == 0.5

    np.random.seed(193)
    observed = sample_transform(params)
    np.random.seed(193)
    scale = np.random.uniform(*params["scale"])
    theta = np.random.uniform(-params["rot"], params["rot"])
    translation = np.random.normal(0.0, params["translate"], 3)
    horizontal = bool(np.random.choice([0, 1]))
    vertical = bool(np.random.choice([0, 1]))
    flip = np.diag(
        [-1.0 if vertical else 1.0, -1.0 if horizontal else 1.0, 1.0]
    )
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    expected = np.eye(4)
    expected[:3, :3] = flip @ (scale * rotation)
    expected[:3, 3] = flip @ (translation * scale)
    assert np.array_equal(observed, expected)


def test_reference_post_filters_keep_all_gt_fields_row_aligned_and_strict():
    sample = {
        "lidar_points": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 2.0], [2.0, 0.0, 0.0, 3.0, 4.0], [-1.0, 0.0, 0.0, 5.0, 6.0]]
        ),
        "gt_boxes": torch.tensor(
            [[0.0, 0.0, 0.0, 1, 1, 1, 4 * np.pi], [1.0, 0.0, 0.0, 1, 1, 1, 0.0]]
        ),
        "gt_velocity": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "gt_labels": torch.tensor([0, 1]),
        "gt_num_lidar_pts": torch.tensor([5, 6]),
        "gt_visibility": torch.tensor([1, 2]),
        "gt_in_range": torch.tensor([True, False]),
        "gt_names": ["car", "truck"],
        "gt_attribute": ["a", "b"],
        "gt_instance_tokens": ["i0", "i1"],
        "gt_ann_tokens": ["a0", "a1"],
        "num_boxes": 2,
    }
    result = apply_reference_post_filters(
        sample,
        {
            "point_cloud_range": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "object_classes": ["car", "truck"],
            "point_shuffle": False,
        },
    )
    # Boundary points/centres are excluded exactly as mmdet3d's filters do.
    assert result["lidar_points"].shape == (1, 5)
    assert result["num_boxes"] == 1
    assert result["gt_names"] == ["car"]
    assert result["gt_ann_tokens"] == ["a0"]
    assert result["gt_labels"].tolist() == [0]
    assert float(result["gt_boxes"][0, 6]) == pytest.approx(0.0)


def test_official_cbgs_uses_one_mt19937_stream_in_class_order():
    per_sample = [[0], [0, 1], [1], [0, 2], [2]]
    observed, segments, stats = build_official_cbgs_indices(
        per_sample, class_names=["a", "b", "c"], seed=7
    )
    pools = [[0, 1, 3], [1, 2], [3, 4]]
    duplicated = sum(map(len, pools))
    rng = np.random.RandomState(7)
    expected_segments = []
    for pool in pools:
        count = int(len(pool) * ((1 / 3) / (len(pool) / duplicated)))
        expected_segments.append(rng.choice(pool, count))
    expected = np.concatenate(expected_segments)
    assert np.array_equal(observed, expected)
    assert all(np.array_equal(a, b) for a, b in zip(segments, expected_segments))
    assert stats["duplicated_class_memberships"] == duplicated


def test_phase1_gtdb_loader_binds_manifest_and_every_pickle(tmp_path: Path):
    names = ("car", "truck")
    files = {}
    counts = {}
    for label, name in enumerate(names):
        values = [{
            "name": name,
            "points": np.zeros((5, 5), dtype=np.float32),
            "box3d_lidar": np.zeros(9, dtype=np.float32),
            "num_points_in_gt": 5,
            "difficulty": 0,
            "label": label,
        }]
        payload = pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)
        path = tmp_path / f"{name}.pkl"
        path.write_bytes(payload)
        files[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "objects": 1,
        }
        counts[name] = 1
    contract = {"plan_sha": "p", "request_commit": "r", "candidate_id": "c"}
    source = {"role": "D_fit", "sample_count": 2}
    semantics = {"class_order": list(names), "point_coordinate_frame": "center_relative_lidar"}
    manifest = {
        "schema": "s10.phase1.gtdb.v1",
        "contract": contract,
        "source": source,
        "semantics": semantics,
        "files": files,
        "counts": counts,
        "total_objects": 2,
    }
    encoded = _canonical(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    loaded = load_phase1_gt_database(
        str(tmp_path),
        str(manifest_path),
        expected_manifest_sha256=digest,
        class_names=names,
        expected_contract=contract,
        expected_source=source,
        expected_semantics=semantics,
    )
    assert list(loaded) == list(names)
    with (tmp_path / "car.pkl").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="identity 'car' drift"):
        load_phase1_gt_database(
            str(tmp_path),
            str(manifest_path),
            expected_manifest_sha256=digest,
            class_names=names,
            expected_contract=contract,
            expected_source=source,
            expected_semantics=semantics,
        )


class _CameraNamedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.camera_backbone = torch.nn.Module()
        self.camera_backbone.weight = torch.nn.Parameter(torch.ones(2, 2))
        self.camera_backbone.relative_position_bias_table = torch.nn.Parameter(
            torch.ones(2, 2)
        )
        self.head = torch.nn.Linear(2, 1)


def test_phase1_optimizer_groups_and_scheduler_state_are_complete_and_resumable():
    config = load_resolved_config(CAMERA)
    model = _CameraNamedModel()
    optimizer = build_phase1_optimizer(model, config)
    names = [group["phase1_group_name"] for group in optimizer.param_groups]
    assert names == ["backbone_no_decay", "backbone", "default"]
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(
        list(model.parameters())
    )
    scheduler = Phase1CyclicScheduler(optimizer, config)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5 / 3.0)
    assert optimizer.param_groups[0]["betas"][0] == pytest.approx(0.9)
    for _ in range(17):
        scheduler.step()
    state = scheduler.state_dict()

    clone_model = _CameraNamedModel()
    clone_optimizer = build_phase1_optimizer(clone_model, config)
    clone_scheduler = Phase1CyclicScheduler(clone_optimizer, config)
    clone_scheduler.load_state_dict(state)
    assert clone_scheduler.state_dict() == state
    for left, right in zip(optimizer.param_groups, clone_optimizer.param_groups):
        assert left["lr"] == right["lr"]
        assert left["betas"] == right["betas"]
