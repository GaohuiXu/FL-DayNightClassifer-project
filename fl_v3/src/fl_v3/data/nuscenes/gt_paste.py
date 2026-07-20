"""In-loader GT-paste sampler (MCR P1) — paste rare-class object crops into a training sample.

Called from ``dataset.__getitem__`` BETWEEN the sample-dict build and the BEV-aug hook, so the subsequent
``GlobalRotScaleTrans`` transforms the pasted points/boxes/velocity CONSISTENTLY with the host scene. Each
pasted object gets a per-object yaw jitter (orientation diversity CBGS lacked) and is collision-rejected
against existing + already-pasted boxes (rotated-rect SAT, ``data/`` ⇒ AST-irrelevant). Uses ONLY
``numpy.random`` (seeded per-worker-per-epoch by ``seeded_worker_init``) ⇒ reproducible under the D16
relaxed regime. Default-OFF in the dataset ⇒ when disabled the sample is byte-identical.

Pasted objects are LiDAR-only (no camera pixels). Our LSS is UNSUPERVISED (no depth loss in
``models/fusion/losses.py``), so this injects ZERO depth-label noise — the BEVFusion-standard tolerated
fusion cost; the camera branch simply contributes nothing at a pasted cell (LiDAR carries the signal,
which is exactly where the targeted large/rare vehicles already lean).
"""
from __future__ import annotations

import copy
import os
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from fl_v3.data.nuscenes.gt_database import (
    box_corners_2d,
    load_gt_database,
    load_phase1_gt_database,
    place_object_points,
    rects_overlap,
)


def _paste_sample_legacy(sample: Dict, params: Dict, n_sweeps: int) -> Dict:
    """Paste rare-class object crops into ``sample`` (in place; returns it). TRAIN-ONLY.

    ``params``: ``db_path`` (the per-class GT-DB dir), ``counts`` (``{class_name: K}`` objects to sample
    per class), ``min_points`` (skip DB crops with fewer real points), ``iou_thresh`` (BEV-overlap reject
    threshold; 0.0 ⇒ reject ANY overlap, the standard db_sampler setting), ``yaw_jitter`` (± rad per-object
    rotation for diversity). ``n_sweeps`` selects the point width (5 single / 6 multi-sweep ⇒ the pasted
    rows carry intensity + ring=0 + dt=0).
    """
    from fl_v3.data.nuscenes.class_map import CLASS_RANGE

    counts = params.get("counts") or {}
    if not counts:
        return sample
    db = load_gt_database(params["db_path"], list(counts.keys()))
    min_pts = int(params.get("min_points", 5))
    yaw_jit = float(params.get("yaw_jitter", 0.0))

    pts = sample["lidar_points"].numpy()           # [P, W] (W=5 single, 6 multi-sweep)
    W = pts.shape[1]
    gt_boxes = sample["gt_boxes"].numpy()          # [M,7]

    # footprints of existing GT (and we append accepted pastes as we go) for collision rejection
    accepted_corners = [box_corners_2d((b[0], b[1], b[3], b[4], b[6])) for b in gt_boxes]

    add_rows, n_boxes, n_labels, n_names, n_npts = [], [], [], [], []
    for name, K in counts.items():
        objs = db.get(name)
        if not objs:
            continue
        for j in np.random.randint(0, len(objs), size=int(K)):
            o = objs[int(j)]
            if int(o["num_pts"]) < min_pts:
                continue
            cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in o["box7"][:7]]
            if yaw_jit > 0.0:
                yaw = yaw + float(np.random.uniform(-yaw_jit, yaw_jit))
            corners = box_corners_2d((cx, cy, dx, dy, yaw))
            if any(rects_overlap(corners, ec) for ec in accepted_corners):
                continue                            # overlaps a real or already-pasted box ⇒ skip
            world_xyz = place_object_points(o["points"], (cx, cy, cz), yaw)   # [n,3]
            n = world_xyz.shape[0]
            row = np.zeros((n, W), dtype=np.float32)
            row[:, 0:3] = world_xyz
            row[:, 3] = o["points"][:, 3]           # intensity (cols 4=ring,5=dt stay 0)
            add_rows.append(row)
            n_boxes.append([cx, cy, cz, dx, dy, dz, yaw])
            n_labels.append(int(o["label"]))
            n_names.append(name)
            n_npts.append(int(n))
            accepted_corners.append(corners)

    if not n_boxes:
        return sample

    add_pts = np.concatenate(add_rows, axis=0).astype(np.float32)
    sample["lidar_points"] = torch.from_numpy(np.ascontiguousarray(np.concatenate([pts, add_pts], axis=0)))

    K = len(n_boxes)
    nb = np.asarray(n_boxes, dtype=np.float32)
    sample["gt_boxes"] = torch.cat([sample["gt_boxes"], torch.from_numpy(nb)], dim=0)
    sample["gt_velocity"] = torch.cat([sample["gt_velocity"], torch.zeros((K, 2), dtype=torch.float32)], dim=0)
    sample["gt_labels"] = torch.cat([sample["gt_labels"], torch.tensor(n_labels, dtype=torch.int64)], dim=0)
    sample["gt_num_lidar_pts"] = torch.cat(
        [sample["gt_num_lidar_pts"], torch.tensor(n_npts, dtype=torch.int64)], dim=0)
    sample["gt_visibility"] = torch.cat(
        [sample["gt_visibility"], torch.full((K,), 4, dtype=torch.int64)], dim=0)   # sentinel: fully visible
    in_range = [float(np.hypot(nb[i, 0], nb[i, 1])) < float(CLASS_RANGE[n_names[i]]) for i in range(K)]
    sample["gt_in_range"] = torch.cat(
        [sample["gt_in_range"], torch.tensor(in_range, dtype=torch.bool)], dim=0)
    sample["gt_names"] = list(sample["gt_names"]) + n_names
    sample["gt_attribute"] = list(sample["gt_attribute"]) + [""] * K
    sample["gt_instance_tokens"] = list(sample["gt_instance_tokens"]) + ["gtpaste"] * K
    sample["gt_ann_tokens"] = list(sample["gt_ann_tokens"]) + ["gtpaste"] * K
    sample["num_boxes"] = int(sample["gt_boxes"].shape[0])
    return sample


class ReferenceBatchSampler:
    """Pinned MIT BatchSampler semantics, including remainder-on-wrap."""

    def __init__(self, values: Sequence[dict], name: str):
        self.values = list(values)
        self.name = str(name)
        self.indices = np.arange(len(self.values), dtype=np.int64)
        np.random.shuffle(self.indices)
        self.index = 0

    def sample(self, number: int) -> list[dict]:
        count = int(number)
        if count < 0:
            raise ValueError("reference GT sampler count must be non-negative")
        if self.index + count >= len(self.values):
            selected = self.indices[self.index :].copy()
            np.random.shuffle(self.indices)
            self.index = 0
        else:
            selected = self.indices[self.index : self.index + count]
            self.index += count
        return [self.values[int(index)] for index in selected]


def _collision_matrix(corners: Sequence[np.ndarray]) -> np.ndarray:
    size = len(corners)
    matrix = np.zeros((size, size), dtype=bool)
    for left in range(size):
        for right in range(left + 1, size):
            collision = rects_overlap(corners[left], corners[right])
            matrix[left, right] = collision
            matrix[right, left] = collision
    return matrix


def _points_in_boxes(points_xyz: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return ``[P,N]`` membership with the selected LiDAR box convention."""
    points = np.asarray(points_xyz, dtype=np.float64)
    raw_boxes = np.asarray(boxes, dtype=np.float64)
    result = np.zeros((points.shape[0], raw_boxes.shape[0]), dtype=bool)
    for index, box in enumerate(raw_boxes):
        cx, cy, cz, dx, dy, dz, yaw = [float(value) for value in box[:7]]
        xw = points[:, 0] - cx
        yw = points[:, 1] - cy
        zw = points[:, 2] - cz
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local_x = cosine * xw + sine * yw
        local_y = -sine * xw + cosine * yw
        result[:, index] = (
            (np.abs(local_x) <= dx / 2.0)
            & (np.abs(local_y) <= dy / 2.0)
            & (np.abs(zw) <= dz / 2.0)
        )
    return result


class ReferenceGTDatabaseSampler:
    """mmdet-free implementation of the pinned MIT DataBaseSampler math."""

    def __init__(
        self,
        database: Mapping[str, Sequence[dict]],
        *,
        class_names: Sequence[str],
        sample_groups: Mapping[str, int],
        min_points: int,
        removed_difficulty: Sequence[int] = (-1,),
        rate: float = 1.0,
    ) -> None:
        self.class_names = tuple(str(name) for name in class_names)
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("reference GT class order contains duplicate names")
        normalized_groups = {
            str(name): int(count) for name, count in sample_groups.items()
        }
        expected_names = set(self.class_names)
        observed_names = set(normalized_groups)
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            extra = sorted(observed_names - expected_names)
            raise ValueError(
                "reference GT sample-group classes differ from the frozen classes: "
                f"missing={missing}, extra={extra}"
            )
        self.class_to_label = {name: index for index, name in enumerate(self.class_names)}
        # JSON object ordering is not scientific state. Reconstruct the mapping in
        # the explicit frozen class order used by the sampling loop.
        self.sample_groups = {name: normalized_groups[name] for name in self.class_names}
        self.rate = float(rate)
        removed = {int(value) for value in removed_difficulty}
        self.database: dict[str, list[dict]] = {}
        self.samplers: dict[str, ReferenceBatchSampler] = {}
        for name in self.class_names:
            values = []
            for raw in database.get(name, []):
                value = dict(raw)
                count = int(value.get("num_points_in_gt", value.get("num_pts", -1)))
                if int(value.get("difficulty", 0)) in removed or count < int(min_points):
                    continue
                if value.get("name") != name:
                    raise ValueError("GT database class/name drift")
                points = np.asarray(value["points"], dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 5:
                    raise ValueError("Phase-I GT database points must preserve exactly five columns")
                value["points"] = points
                value["box3d_lidar"] = np.asarray(
                    value.get("box3d_lidar", value.get("box7")), dtype=np.float32
                )
                if value["box3d_lidar"].shape != (9,):
                    raise ValueError("Phase-I GT database box must preserve box7 plus velocity")
                values.append(value)
            if not values:
                raise ValueError(f"Phase-I GT database class {name!r} is empty after filtering")
            self.database[name] = values
            self.samplers[name] = ReferenceBatchSampler(values, name)

    def _sample_class(self, name: str, number: int, avoid_boxes: np.ndarray) -> list[dict]:
        sampled = copy.deepcopy(self.samplers[name].sample(number))
        if not sampled:
            return []
        sample_boxes = np.stack(
            [value["box3d_lidar"][:7] for value in sampled], axis=0
        )
        boxes = np.concatenate([avoid_boxes, sample_boxes], axis=0)
        corners = [
            box_corners_2d((box[0], box[1], box[3], box[4], box[6]))
            for box in boxes
        ]
        collisions = _collision_matrix(corners)
        np.fill_diagonal(collisions, False)
        valid: list[dict] = []
        start = int(avoid_boxes.shape[0])
        for row in range(start, boxes.shape[0]):
            if collisions[row].any():
                collisions[row, :] = False
                collisions[:, row] = False
            else:
                valid.append(sampled[row - start])
        return valid

    def sample_all(self, gt_boxes: np.ndarray, gt_labels: np.ndarray) -> list[dict]:
        boxes = np.asarray(gt_boxes, dtype=np.float32)
        labels = np.asarray(gt_labels, dtype=np.int64)
        accepted: list[dict] = []
        avoid = boxes
        for name in self.class_names:
            label = self.class_to_label[name]
            number = int(self.sample_groups[name] - int(np.sum(labels == label)))
            number = int(np.round(self.rate * number).astype(np.int64))
            if number <= 0:
                continue
            values = self._sample_class(name, number, avoid)
            accepted.extend(values)
            if values:
                new_boxes = np.stack(
                    [value["box3d_lidar"][:7] for value in values], axis=0
                )
                avoid = np.concatenate([avoid, new_boxes], axis=0)
        return accepted


_REFERENCE_SAMPLERS: dict[tuple[int, str], ReferenceGTDatabaseSampler] = {}


def _load_reference_sampler(params: Mapping[str, object]) -> ReferenceGTDatabaseSampler:
    db_path = os.path.abspath(str(params["database_root"]))
    manifest_sha = str(params["manifest_sha256"])
    key = (os.getpid(), f"{db_path}:{manifest_sha}")
    cached = _REFERENCE_SAMPLERS.get(key)
    if cached is not None:
        return cached
    database = load_phase1_gt_database(
        db_path,
        str(params["manifest_path"]),
        expected_manifest_sha256=manifest_sha,
        class_names=params["class_names"],
        expected_contract=params["expected_contract"],
        expected_source=params["expected_source"],
        expected_semantics=params["expected_semantics"],
    )
    sampler = ReferenceGTDatabaseSampler(
        database,
        class_names=params["class_names"],
        sample_groups=params["sample_groups"],
        min_points=int(params["min_points"]),
        removed_difficulty=params["filter_by_difficulty"],
        rate=float(params.get("rate", 1.0)),
    )
    _REFERENCE_SAMPLERS[key] = sampler
    return sampler


def _paste_sample_reference(sample: Dict, params: Dict, n_sweeps: int) -> Dict:
    if int(n_sweeps) != 1:
        raise ValueError("Phase-I role-bound GT-paste is keyframe-only")
    epoch = int(sample.get("_phase1_epoch", params.get("epoch", 0)))
    if epoch >= int(params["stop_epoch"]):
        return sample
    if float(params.get("yaw_jitter_radians", 0.0)) != 0.0:
        raise ValueError("Phase-I reference GT-paste forbids extra yaw jitter")
    sampler = _load_reference_sampler(params)
    points = sample["lidar_points"].numpy()
    if points.ndim != 2 or points.shape[1] != 5:
        raise ValueError("Phase-I GT-paste requires five-dimensional keyframe points")
    boxes = sample["gt_boxes"].numpy()
    labels = sample["gt_labels"].numpy()
    selected = sampler.sample_all(boxes, labels)
    if not selected:
        return sample

    new_box_states = np.stack(
        [value["box3d_lidar"] for value in selected]
    ).astype(np.float32)
    new_boxes = new_box_states[:, :7]
    new_velocity = new_box_states[:, 7:9]
    inside = _points_in_boxes(points[:, :3], new_boxes)
    host_points = points[np.logical_not(inside.any(axis=1))]
    pasted_points = []
    names: list[str] = []
    new_labels: list[int] = []
    counts: list[int] = []
    source_ann_tokens: list[str] = []
    for value in selected:
        local = np.asarray(value["points"], dtype=np.float32)
        box = np.asarray(value["box3d_lidar"], dtype=np.float32)
        placed = local.copy()
        # The reference GTDB subtracts only the box center; it does not rotate
        # crops into the object coordinate frame. No yaw jitter is configured.
        placed[:, :3] += box[:3]
        pasted_points.append(placed)
        name = str(value["name"])
        names.append(name)
        new_labels.append(sampler.class_to_label[name])
        counts.append(int(local.shape[0]))
        source_ann_tokens.append(str(value.get("source_ann_token", "gtpaste")))
    sample["lidar_points"] = torch.from_numpy(
        np.ascontiguousarray(np.concatenate([*pasted_points, host_points], axis=0))
    )
    count = len(selected)
    sample["gt_boxes"] = torch.cat([sample["gt_boxes"], torch.from_numpy(new_boxes)], dim=0)
    sample["gt_velocity"] = torch.cat(
        [sample["gt_velocity"], torch.from_numpy(new_velocity)], dim=0
    )
    sample["gt_labels"] = torch.cat(
        [sample["gt_labels"], torch.tensor(new_labels, dtype=torch.int64)], dim=0
    )
    sample["gt_num_lidar_pts"] = torch.cat(
        [sample["gt_num_lidar_pts"], torch.tensor(counts, dtype=torch.int64)], dim=0
    )
    sample["gt_visibility"] = torch.cat(
        [sample["gt_visibility"], torch.zeros(count, dtype=torch.int64)], dim=0
    )
    from fl_v3.data.nuscenes.class_map import CLASS_RANGE

    in_range = [
        float(np.hypot(new_boxes[index, 0], new_boxes[index, 1]))
        < float(CLASS_RANGE[name])
        for index, name in enumerate(names)
    ]
    sample["gt_in_range"] = torch.cat(
        [sample["gt_in_range"], torch.tensor(in_range, dtype=torch.bool)], dim=0
    )
    sample["gt_names"] = list(sample["gt_names"]) + names
    sample["gt_attribute"] = list(sample["gt_attribute"]) + [""] * count
    sample["gt_instance_tokens"] = list(sample["gt_instance_tokens"]) + ["gtpaste"] * count
    sample["gt_ann_tokens"] = list(sample["gt_ann_tokens"]) + [
        f"gtpaste:{token}" for token in source_ann_tokens
    ]
    sample["num_boxes"] = int(sample["gt_boxes"].shape[0])
    return sample


def paste_sample(sample: Dict, params: Dict, n_sweeps: int) -> Dict:
    """Dispatch to the explicit legacy or Phase-I reference algorithm."""
    algorithm = str(params.get("algorithm", "legacy"))
    if algorithm == "mit_bevfusion_reference":
        return _paste_sample_reference(sample, params, n_sweeps)
    if algorithm != "legacy":
        raise ValueError(f"unknown GT-paste algorithm {algorithm!r}")
    return _paste_sample_legacy(sample, params, n_sweeps)
