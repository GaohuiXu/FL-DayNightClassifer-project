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

from typing import Dict

import numpy as np
import torch

from fl_v3.data.nuscenes.gt_database import (
    box_corners_2d,
    load_gt_database,
    place_object_points,
    rects_overlap,
)


def paste_sample(sample: Dict, params: Dict, n_sweeps: int) -> Dict:
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
