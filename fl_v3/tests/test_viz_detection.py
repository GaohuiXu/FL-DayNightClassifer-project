"""T4 — V4 detection viz: renders the SHARED decode; TP/FN agrees with the metric/ASR matcher
(no V4-only re-threshold), incl. boundary boxes near τ_clean / d_clean."""
from __future__ import annotations

import os

import numpy as np

from fl_v3.data.nuscenes.class_map import NAME_TO_ID
from fl_v3.eval.detection_eval import SampleDecode
from fl_v3.eval.asr import AsrThresholds, detected_target_anns, evaluate_sample_eligibility
from fl_v3.viz.detection import render_v4, v4_target_table
from fl_v3.viz.writer import VizWriter

CAR = NAME_TO_ID["car"]


def _decode(info, decoded_boxes, decoded_scores):
    return SampleDecode(
        sample_token=info["sample_token"],
        boxes=np.asarray(decoded_boxes, np.float64).reshape(-1, 7) if len(decoded_boxes) else np.zeros((0, 7)),
        scores=np.asarray(decoded_scores, np.float64).reshape(-1) if len(decoded_scores) else np.zeros((0,)),
        labels=np.full(len(decoded_boxes), CAR, np.int64) if len(decoded_boxes) else np.zeros((0,), np.int64),
        velocity=np.zeros((len(decoded_boxes), 2)) if len(decoded_boxes) else np.zeros((0, 2)),
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


def _first_car_idx(info):
    for m in range(info["gt_boxes"].shape[0]):
        if int(info["gt_labels"][m]) == CAR:
            return m
    return None


def test_v4_renders_and_writes_manifest(mini_val_info, tmp_path):
    info = next(i for i in mini_val_info if _first_car_idx(i) is not None)
    m = _first_car_idx(info)
    box = np.asarray(info["gt_boxes"][m], np.float64).copy()
    dec = _decode(info, [box], [0.6])
    thr = AsrThresholds(tau_pts=1, tau_clean=0.1, d_clean=2.0)
    writer = VizWriter(str(tmp_path))
    out = render_v4(writer, dec, thr, image_chw_by_cam=np.zeros((6, 3, 900, 1600), np.uint8), cam_index=0)
    writer.write_manifest()
    assert os.path.isfile(out["bev"]) and os.path.isfile(out["cam"]) and os.path.isfile(out["table_json"])
    import json
    man = json.load(open(os.path.join(str(tmp_path), "viz", "manifest.json")))
    det = man["stages"]["detection"]
    assert det["viz_id"] == "V4" and len(det["artifacts"]) >= 3


def test_v4_tpfn_agrees_with_metric_matcher_at_boundary(mini_val_info):
    """V4 TP/FN == the ASR/metric matcher decision, INCLUDING boundary boxes near τ_clean/d_clean
    (V4 must not re-threshold). Construct: GT car A with a detection at score=τ_clean exactly,
    dist just inside d_clean (TP); GT car B with a detection just below τ_clean (FN); GT car C
    with a detection just outside d_clean (FN)."""
    thr = AsrThresholds(tau_pts=1, tau_clean=0.30, d_clean=2.0)
    # find a sample with >=3 cars passing criteria 1-4 so the matcher exercises real geometry
    chosen = None
    for info in mini_val_info:
        cars = [m for m in range(info["gt_boxes"].shape[0])
                if int(info["gt_labels"][m]) == CAR and bool(info["gt_in_range"][m])]
        if len(cars) >= 3:
            chosen = (info, cars[:3]); break
    assert chosen, "need a mini_val sample with >=3 in-range cars"
    info, (a, b, c) = chosen
    ba = np.asarray(info["gt_boxes"][a], np.float64).copy()              # exact center, score=τ_clean
    bb = np.asarray(info["gt_boxes"][b], np.float64).copy()              # exact center, score<τ_clean
    bc = np.asarray(info["gt_boxes"][c], np.float64).copy(); bc[0] += 2.5  # 2.5 m off (> d_clean), score high
    dec = _decode(info, [ba, bb, bc], [0.30, 0.29, 0.95])

    table = {r["gt_idx"]: r for r in v4_target_table(dec, thr)}
    # V4 verdicts
    assert table[a]["tp"] and not table[a]["fn"], "score==τ_clean, dist≈0 → TP"
    assert table[b]["fn"] and not table[b]["tp"], "score<τ_clean → FN (criterion 5)"
    assert table[c]["fn"] and not table[c]["tp"], "dist>d_clean → FN (criterion 6)"

    # agreement: V4 TP set == the ASR matcher's clean-detected set (same thresholds, no re-threshold)
    v4_tp = {r["ann_token"] for r in v4_target_table(dec, thr) if r["tp"]}
    asr_detected = detected_target_anns(dec, thr)
    assert v4_tp == asr_detected, f"V4 TP {v4_tp} != ASR matcher {asr_detected}"
    # and the eligibility rows are the SAME object the metric harness produces (no divergent path)
    elig = {r["ann_token"]: r["clean_detected"] for r in evaluate_sample_eligibility(dec, thr)}
    for r in v4_target_table(dec, thr):
        assert r["tp"] == elig[r["ann_token"]]
