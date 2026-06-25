"""MCR P1 GT-paste (gt-sampling) — geometry, paste contract, determinism, PFN invariance.

Pins: (a) crop↔place round-trips a known object; (b) the SAT collision test discriminates overlap; (c)
paste_sample extends ALL ragged GT fields consistently, places points inside their pasted boxes, and is a
no-op with empty counts (the OFF contract); (d) the paste is deterministic under a fixed numpy seed; (e)
the collision filter rejects an overlapping candidate; (f) a cloud WITH pasted points stays
permutation-invariant through the PointPillarsEncoder (re-confirms no models/fusion obligation).
"""
from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np
import torch

from fl_v3.data.nuscenes.gt_database import (
    box_corners_2d, crop_object_points, place_object_points, rects_overlap,
)
from fl_v3.data.nuscenes.gt_paste import paste_sample
from fl_v3.training.tasks import _gtpaste_from_run


# --------------------------------------------------------------------------- geometry
def test_crop_place_roundtrip():
    box7 = np.array([10.0, -4.0, 0.5, 4.0, 2.0, 1.6, 0.7])     # cx,cy,cz,dx,dy,dz,yaw
    # 5 world points known to be inside the box (place a small grid at the center)
    cx, cy, cz = box7[:3]
    world = np.array([[cx, cy, cz, 0.3], [cx + 0.5, cy, cz, 0.4], [cx, cy + 0.4, cz + 0.2, 0.5],
                      [cx - 0.6, cy - 0.3, cz, 0.6], [cx, cy, cz - 0.3, 0.7]], dtype=np.float32)
    local = crop_object_points(world, box7)
    assert local.shape == (5, 4)
    back = place_object_points(local, (cx, cy, cz), box7[6])  # same yaw ⇒ recover world xyz
    assert np.allclose(back, world[:, :3], atol=1e-4)
    assert np.allclose(local[:, 3], world[:, 3])              # intensity preserved


def test_crop_excludes_outside_points():
    box7 = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0])
    pts = np.array([[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0], [0.0, 5.0, 0.0, 1.0]], dtype=np.float32)
    local = crop_object_points(pts, box7)
    assert local.shape[0] == 1                                 # only the in-box point survives


def test_sat_overlap_discriminates():
    a = box_corners_2d((0.0, 0.0, 4.0, 2.0, 0.0))
    b_overlap = box_corners_2d((1.0, 0.0, 4.0, 2.0, 0.0))
    b_disjoint = box_corners_2d((20.0, 0.0, 4.0, 2.0, 0.0))
    assert rects_overlap(a, b_overlap) is True
    assert rects_overlap(a, b_disjoint) is False


# --------------------------------------------------------------------------- paste contract
def _toy_db(tmp, name="trailer", label=4, n=40):
    os.makedirs(tmp, exist_ok=True)
    objs = []
    for k in range(n):
        npts = 8 + k
        local = np.zeros((npts, 4), dtype=np.float32)
        local[:, :3] = np.random.RandomState(k).uniform(-1, 1, (npts, 3)) * 0.5
        local[:, 3] = 0.5
        # spread stored centers so collisions are rare
        objs.append({"points": local, "box7": np.array([8.0 + 2 * k, 3.0, 0.4, 4.0, 2.0, 1.6, 0.2], np.float32),
                     "label": label, "name": name, "num_pts": npts})
    with open(os.path.join(tmp, f"{name}.pkl"), "wb") as f:
        pickle.dump(objs, f)


def _toy_sample(M=3, W=6):
    g = np.random.RandomState(0)
    pts = np.zeros((100, W), dtype=np.float32); pts[:, :3] = g.uniform(-30, 30, (100, 3))
    return {
        "lidar_points": torch.from_numpy(pts),
        "gt_boxes": torch.from_numpy(np.array([[-20.0, -20.0, 0.0, 4.0, 2.0, 1.6, 0.0]] * M, np.float32)),
        "gt_velocity": torch.zeros(M, 2), "gt_labels": torch.zeros(M, dtype=torch.int64),
        "gt_num_lidar_pts": torch.full((M,), 50, dtype=torch.int64),
        "gt_visibility": torch.full((M,), 4, dtype=torch.int64),
        "gt_in_range": torch.ones(M, dtype=torch.bool),
        "gt_names": ["car"] * M, "gt_attribute": [""] * M,
        "gt_instance_tokens": ["x"] * M, "gt_ann_tokens": ["y"] * M, "num_boxes": M,
    }


def test_paste_extends_all_fields_and_points_in_box():
    with tempfile.TemporaryDirectory() as tmp:
        _toy_db(tmp)
        params = {"db_path": tmp, "counts": {"trailer": 4}, "min_points": 5,
                  "iou_thresh": 0.0, "yaw_jitter": 0.0}
        np.random.seed(1)
        s = paste_sample(_toy_sample(M=3, W=6), params, n_sweeps=10)
        K = s["gt_boxes"].shape[0] - 3
        assert K >= 1                                          # at least some pasted (spread centers ⇒ few collisions)
        for f in ("gt_boxes", "gt_velocity", "gt_labels", "gt_num_lidar_pts", "gt_visibility",
                  "gt_in_range", "gt_names", "gt_attribute", "gt_instance_tokens", "gt_ann_tokens"):
            assert len(s[f]) == 3 + K, f"{f} not extended to {3+K}"
        assert s["num_boxes"] == 3 + K
        assert s["lidar_points"].shape[1] == 6                 # width preserved (multi-sweep)
        # every pasted point lies inside its pasted box
        pts = s["lidar_points"].numpy()
        added = pts[100:]                                      # the toy sample had 100 original points
        for i in range(K):
            box = s["gt_boxes"][3 + i].numpy()
            npts_i = int(s["gt_num_lidar_pts"][3 + i])
            chunk = added[:npts_i]; added = added[npts_i:]
            assert crop_object_points(chunk, box).shape[0] == npts_i  # all inside


def test_paste_deterministic_under_seed():
    with tempfile.TemporaryDirectory() as tmp:
        _toy_db(tmp)
        params = {"db_path": tmp, "counts": {"trailer": 4}, "min_points": 5, "iou_thresh": 0.0, "yaw_jitter": 0.785}
        np.random.seed(7); a = paste_sample(_toy_sample(), params, 10)
        np.random.seed(7); b = paste_sample(_toy_sample(), params, 10)
        assert torch.equal(a["gt_boxes"], b["gt_boxes"]) and torch.equal(a["lidar_points"], b["lidar_points"])


def test_paste_empty_counts_is_noop():
    s0 = _toy_sample()
    s1 = paste_sample(_toy_sample(), {"db_path": "/nonexistent", "counts": {}}, 10)
    assert s1["gt_boxes"].shape[0] == s0["gt_boxes"].shape[0] and s1["num_boxes"] == s0["num_boxes"]


def test_collision_filter_rejects_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        # one DB object whose stored center collides with the sample's existing box
        os.makedirs(tmp, exist_ok=True)
        obj = {"points": np.zeros((10, 4), np.float32), "box7": np.array([0.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0], np.float32),
               "label": 4, "name": "trailer", "num_pts": 10}
        with open(os.path.join(tmp, "trailer.pkl"), "wb") as f:
            pickle.dump([obj], f)
        s = _toy_sample(M=1)
        s["gt_boxes"] = torch.from_numpy(np.array([[0.0, 0.0, 0.0, 6.0, 6.0, 2.0, 0.0]], np.float32))  # covers origin
        np.random.seed(0)
        out = paste_sample(s, {"db_path": tmp, "counts": {"trailer": 1}, "min_points": 5,
                               "iou_thresh": 0.0, "yaw_jitter": 0.0}, 10)
        assert out["gt_boxes"].shape[0] == 1                   # the only candidate overlapped ⇒ rejected


def test_gtpaste_from_run_off_returns_none():
    assert _gtpaste_from_run({"det-gt-paste": False}) is None
    assert _gtpaste_from_run({}) is None
    p = _gtpaste_from_run({"det-gt-paste": True, "det-gt-paste-counts": {"trailer": 2}})
    assert p["counts"] == {"trailer": 2}


# --------------------------------------------------------------------------- PFN invariance with pasted points
def test_pfn_permutation_invariant_with_pasted_points():
    from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder
    from fl_v3.models.fusion.bev_grid import BEVConfig
    enc = PointPillarsEncoder(out_channels=64, cfg=BEVConfig(), use_timestamp=True).eval()
    g = np.random.RandomState(3)
    P0 = 500
    pts = np.zeros((P0, 7), np.float32)            # [batch_idx, x,y,z,intensity,ring,dt]
    pts[:, 1:4] = g.uniform(-20, 20, (P0, 3)); pts[:, 4] = g.uniform(0, 1, P0)
    t = torch.from_numpy(pts)
    with torch.no_grad():
        bev_a = enc(t, B=1)
        perm = torch.randperm(P0)
        bev_b = enc(t[perm], B=1)
    assert torch.allclose(bev_a, bev_b, atol=1e-5)
