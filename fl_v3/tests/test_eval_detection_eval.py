"""T4 — official DetectionEval wiring: GT-as-pred AP≈1 + determinism/permutation-invariance.

The §0.1 GT-as-pred sanity (GT side = the devkit's own ``load_gt``, pred side =
``box_to_global(T1 GT)``) → per-class AP≈1. Plus: the results JSON + mAP/NDS are
permutation-invariant on equal-score ties (the devkit ``accumulate`` breaks score ties on
emission order, so the content-defined box sort is load-bearing). No trained model needed —
GT-as-pred + synthetic decoded boxes exercise the whole evaluator deterministically.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fl_v3.data.nuscenes.class_map import DETECTION_NAMES, NAME_TO_ID
from fl_v3.eval.detection_eval import (
    SampleDecode,
    assert_version_split,
    build_results_dict,
    gt_as_pred_submission,
    run_detection_eval,
)


def _sample_decode_from_info(info, *, decoded=None) -> SampleDecode:
    """Build a SampleDecode from a T1 info dict; decoded boxes empty unless supplied."""
    z7 = np.zeros((0, 7)); z = np.zeros((0,)); z2 = np.zeros((0, 2)); zi = np.zeros((0,), dtype=np.int64)
    d = decoded or {}
    return SampleDecode(
        sample_token=info["sample_token"],
        boxes=d.get("boxes", z7), scores=d.get("scores", z), labels=d.get("labels", zi),
        velocity=d.get("velocity", z2),
        lidar2ego=np.asarray(info["lidar2ego"], np.float64),
        ego2global_lidar=np.asarray(info["ego2global_lidar"], np.float64),
        lidar2img=np.asarray(info["lidar2img"], np.float64),
        gt_boxes=np.asarray(info["gt_boxes"], np.float64).reshape(-1, 7),
        gt_labels=np.asarray(info["gt_labels"], np.int64).reshape(-1),
        gt_num_lidar_pts=np.asarray(info["gt_num_lidar_pts"], np.int64).reshape(-1),
        gt_in_range=np.asarray(info["gt_in_range"], bool).reshape(-1),
        gt_velocity=np.asarray(info["gt_velocity"], np.float64).reshape(-1, 2),
        gt_ann_tokens=list(info["gt_ann_tokens"]),
        gt_names=list(info["gt_names"]),
        gt_attribute=list(info["gt_attribute"]),
    )


def test_assert_version_split():
    assert_version_split("v1.0-mini", "mini_val")
    assert_version_split("v1.0-trainval", "val")
    with pytest.raises(AssertionError):
        assert_version_split("v1.0-mini", "val")
    with pytest.raises(AssertionError):
        assert_version_split("v1.0-trainval", "mini_val")


def test_results_dict_has_all_tokens_as_keys():
    """Every eval token must be a key (empty []) — DetectionEval asserts pred⊇gt token set."""
    toks = ["tokA", "tokB", "tokC"]
    d = SampleDecode(
        sample_token="tokA",
        boxes=np.zeros((0, 7)), scores=np.zeros((0,)), labels=np.zeros((0,), np.int64),
        velocity=np.zeros((0, 2)), lidar2ego=np.eye(4), ego2global_lidar=np.eye(4),
        lidar2img=np.zeros((6, 4, 4)), gt_boxes=np.zeros((0, 7)), gt_labels=np.zeros((0,), np.int64),
        gt_num_lidar_pts=np.zeros((0,), np.int64), gt_in_range=np.zeros((0,), bool),
        gt_velocity=np.zeros((0, 2)),
    )
    sub = build_results_dict([d], DETECTION_NAMES, toks)
    assert set(sub["results"].keys()) == set(toks)
    assert sub["results"]["tokB"] == [] and sub["results"]["tokC"] == []
    assert sub["meta"]["use_camera"] and sub["meta"]["use_lidar"] and not sub["meta"]["use_radar"]


def _car_box_near(info, gt_idx, score, dx=0.0, dy=0.0):
    """A synthetic decoded car box copied from a GT car center (+ optional offset)."""
    b = np.asarray(info["gt_boxes"][gt_idx], np.float64).copy()
    b[0] += dx; b[1] += dy
    return b


def test_gt_as_pred_per_class_ap_near_one(nusc_mini, mini_val_info, tmp_path):
    """SPEC §0.1 GT-as-pred sanity: submitting box_to_global(T1 GT) as predictions, scored vs
    the devkit's own load_gt GT, gives **car AP@2m = 1.0** (the D8 class) with ATE/AOE/AAE/AVE≈0,
    and EVERY class that has GT in mini_val reproduces to AP≈1 (absent classes → 0, excluded).
    Each pred carries the devkit GT ``num_pts`` (lidar+radar) so the points filter is identical
    on both sides (T1 carries lidar-only counts)."""
    import os

    decodes = [_sample_decode_from_info(info) for info in mini_val_info]
    all_tokens = [info["sample_token"] for info in mini_val_info]
    # devkit num_pts = num_lidar + num_radar per annotation (mirrors load_gt's GT exactly).
    num_pts_by_ann = {}
    for info in mini_val_info:
        for at in info["gt_ann_tokens"]:
            ann = nusc_mini.get("sample_annotation", at)
            num_pts_by_ann[at] = int(ann["num_lidar_pts"]) + int(ann["num_radar_pts"])
    sub = gt_as_pred_submission(decodes, DETECTION_NAMES, all_tokens,
                                copy_velocity=True, copy_attribute=True,
                                num_pts_by_ann=num_pts_by_ann)
    out_dir = str(tmp_path / "gt_as_pred"); os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(sub, f, sort_keys=True)

    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.common.config import config_factory

    cfg = config_factory("detection_cvpr_2019")
    ev = DetectionEval(nusc_mini, config=cfg, result_path=result_path,
                       eval_set="mini_val", output_dir=out_dir, verbose=False)
    metrics, mdl = ev.evaluate()
    ser = metrics.serialize()
    # NOTE: devkit serialize() keys label_aps by FLOAT dist_th (2.0), not the string "2.0".
    car_ap_2m = ser["label_aps"]["car"][2.0]
    car_te = ser["label_tp_errors"]["car"]
    assert car_ap_2m > 0.999, f"GT-as-pred car AP@2m={car_ap_2m} (conversion/geometry bug)"
    assert ser["mean_dist_aps"]["car"] > 0.999, f"car mean AP={ser['mean_dist_aps']['car']}"
    assert car_te["trans_err"] < 1e-6, f"car ATE={car_te['trans_err']} (translation bug)"
    assert car_te["orient_err"] < 0.01, f"car AOE={car_te['orient_err']} (yaw bug)"
    assert car_te["attr_err"] == 0.0, f"car AAE={car_te['attr_err']} (attribute copy bug)"
    assert car_te["vel_err"] < 0.05, f"car AVE={car_te['vel_err']} (velocity copy/convert bug)"
    # Every class that achieves ANY matches reproduces ~perfectly (≈1); absent classes → 0.
    nonzero = {c: ser["label_aps"][c][2.0] for c in DETECTION_NAMES if ser["label_aps"][c][2.0] > 0.0}
    assert "car" in nonzero
    assert all(v > 0.99 for v in nonzero.values()), \
        f"a present class is only partially reproduced: {nonzero}"


def test_detection_eval_permutation_invariant_on_equal_scores(nusc_mini, mini_val_info, tmp_path):
    """Permuting equal-score decoded boxes → byte-identical results JSON AND identical mAP/NDS.

    The devkit ``accumulate`` breaks score ties by emission index, so without the
    content-defined sort the mAP would be order-sensitive. Build synthetic car detections
    (several at the SAME score) and confirm invariance end-to-end through DetectionEval.
    """
    import os

    # pick a few mini_val samples that actually contain cars
    car_id = NAME_TO_ID["car"]
    chosen = []
    for info in mini_val_info:
        idxs = [m for m in range(info["gt_boxes"].shape[0]) if int(info["gt_labels"][m]) == car_id]
        if idxs:
            chosen.append((info, idxs))
        if len(chosen) >= 6:
            break
    assert chosen, "no cars in mini_val (unexpected)"

    def make_decodes(perm: bool):
        decodes = []
        for info, idxs in chosen:
            boxes, scores, labels, vels = [], [], [], []
            for j, m in enumerate(idxs[:4]):
                # TWO boxes per GT car at the SAME score (a deliberate tie), one near, one offset
                for dx, sc in ((0.0, 0.5), (1.2, 0.5)):
                    boxes.append(_car_box_near(info, m, sc, dx=dx))
                    scores.append(sc); labels.append(car_id); vels.append([0.0, 0.0])
            order = list(range(len(boxes)))
            if perm:
                order = order[::-1]  # reverse emission order — same content, different order
            dd = {
                "boxes": np.asarray([boxes[i] for i in order], np.float64),
                "scores": np.asarray([scores[i] for i in order], np.float64),
                "labels": np.asarray([labels[i] for i in order], np.int64),
                "velocity": np.asarray([vels[i] for i in order], np.float64),
            }
            decodes.append(_sample_decode_from_info(info, decoded=dd))
        return decodes

    all_tokens = [info["sample_token"] for info in mini_val_info]
    sub_a = build_results_dict(make_decodes(False), DETECTION_NAMES, all_tokens)
    sub_b = build_results_dict(make_decodes(True), DETECTION_NAMES, all_tokens)
    # (1) the serialized submission is byte-identical under permutation.
    assert json.dumps(sub_a, sort_keys=True) == json.dumps(sub_b, sort_keys=True)

    # (2) DetectionEval mAP/NDS identical for both orders (and a repeat of the same order).
    def eval_sub(sub, tag):
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.common.config import config_factory
        od = str(tmp_path / tag); os.makedirs(od, exist_ok=True)
        rp = os.path.join(od, "results.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(sub, f, sort_keys=True)
        cfg = config_factory("detection_cvpr_2019")
        m, _ = DetectionEval(nusc_mini, config=cfg, result_path=rp,
                             eval_set="mini_val", output_dir=od, verbose=False).evaluate()
        return float(m.mean_ap), float(m.nd_score)

    map_a, nds_a = eval_sub(sub_a, "perm_a")
    map_b, nds_b = eval_sub(sub_b, "perm_b")
    assert map_a == map_b, f"mAP not permutation-invariant: {map_a} vs {map_b}"
    assert nds_a == nds_b, f"NDS not permutation-invariant: {nds_a} vs {nds_b}"
