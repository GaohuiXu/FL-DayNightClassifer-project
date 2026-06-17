"""T4 — ASR-eligibility harness: 6 criteria, the eligible-clean-detected denominator (§0.4),
the content-hashed frozen subset (§0.5), disappearance + false-disappearance (§0.3).

Uses REAL mini_val GT (so frustum/lidar-pts/in-range are genuine) with SYNTHETIC decoded boxes
placed (or not) at GT car centers, so clean-detection (criteria 5+6) is controllable.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from fl_v3.data.nuscenes.class_map import NAME_TO_ID
from fl_v3.eval.detection_eval import SampleDecode
from fl_v3.eval.asr import (
    AsrThresholds,
    build_frozen_asr_subset,
    detected_target_anns,
    disappearance_asr,
    evaluate_sample_eligibility,
    false_disappearance_baseline,
    phantom_asr_slot,
    subset_content_hash,
    thresholds_from_subset,
    verify_frozen_asr_subset,
)

CAR = NAME_TO_ID["car"]


def _decode_from_info(info, detected_gt_idxs=(), score=0.6):
    """SampleDecode with REAL GT + synthetic decoded car boxes at the given GT car centers."""
    boxes, scores, labels, vels = [], [], [], []
    for m in detected_gt_idxs:
        b = np.asarray(info["gt_boxes"][m], np.float64).copy()
        boxes.append(b); scores.append(score); labels.append(CAR); vels.append([0.0, 0.0])
    return SampleDecode(
        sample_token=info["sample_token"],
        boxes=np.asarray(boxes, np.float64).reshape(-1, 7) if boxes else np.zeros((0, 7)),
        scores=np.asarray(scores, np.float64).reshape(-1) if scores else np.zeros((0,)),
        labels=np.asarray(labels, np.int64).reshape(-1) if labels else np.zeros((0,), np.int64),
        velocity=np.asarray(vels, np.float64).reshape(-1, 2) if vels else np.zeros((0, 2)),
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


def _first_clean_eligible_car(mini_val_info, thr):
    """Find a (info, gt_idx) of a car that passes criteria 1–4 (frustum+lidar+range), for the
    denominator exclusion test (we then control whether it is clean-detected)."""
    for info in mini_val_info:
        dec = _decode_from_info(info, detected_gt_idxs=())  # no detections
        for row in evaluate_sample_eligibility(dec, thr):
            if row["c1_class"] and row["c2_frustum"] and row["c3_lidar_pts"] and row["c4_in_range"]:
                return info, row["gt_idx"], row["ann_token"]
    return None, None, None


def test_thresholds_tau_pts_floor():
    from fl_v3.eval.asr import asr_thresholds_from_run_config
    with pytest.raises(ValueError):
        asr_thresholds_from_run_config({"asr-tau-pts": 0})
    thr = asr_thresholds_from_run_config({"asr-tau-pts": 10, "asr-tau-clean": 0.1, "asr-n-min": 150})
    assert thr.target_id == CAR and thr.tau_pts == 10 and thr.n_min == 150


def test_denominator_is_eligible_clean_detected(mini_val_info):
    """§0.4 — a target satisfying criteria 1–4 but NOT clean-detected is EXCLUDED from N."""
    thr = AsrThresholds(tau_pts=1, tau_clean=0.1, d_clean=2.0, n_min=1)
    info, gt_idx, ann = _first_clean_eligible_car(mini_val_info, thr)
    assert info is not None, "no criteria-1–4 car found in mini_val (unexpected)"

    # (a) NOT clean-detected (no decoded box near it) → excluded.
    dec_miss = _decode_from_info(info, detected_gt_idxs=())
    rows_miss = {r["ann_token"]: r for r in evaluate_sample_eligibility(dec_miss, thr)}
    assert rows_miss[ann]["c1_class"] and rows_miss[ann]["c2_frustum"]
    assert rows_miss[ann]["c3_lidar_pts"] and rows_miss[ann]["c4_in_range"]
    assert not rows_miss[ann]["clean_detected"]
    assert not rows_miss[ann]["eligible"], "criteria-1–4-but-undetected car must be EXCLUDED from N"

    # (b) clean-detected (a decoded box at its center, score ≥ τ_clean) → eligible.
    dec_hit = _decode_from_info(info, detected_gt_idxs=(gt_idx,), score=0.6)
    rows_hit = {r["ann_token"]: r for r in evaluate_sample_eligibility(dec_hit, thr)}
    assert rows_hit[ann]["clean_detected"] and rows_hit[ann]["eligible"]


def test_low_score_detection_excluded(mini_val_info):
    """A detection below τ_clean does NOT make a target clean-detected (criterion 5)."""
    thr = AsrThresholds(tau_pts=1, tau_clean=0.5, d_clean=2.0, n_min=1)
    info, gt_idx, ann = _first_clean_eligible_car(mini_val_info, thr)
    assert info is not None
    dec = _decode_from_info(info, detected_gt_idxs=(gt_idx,), score=0.3)  # below τ_clean=0.5
    rows = {r["ann_token"]: r for r in evaluate_sample_eligibility(dec, thr)}
    assert not rows[ann]["clean_detected"] and not rows[ann]["eligible"]


def test_tau_pts_excludes_low_lidar(mini_val_info):
    """Criterion 3: a very high τ_pts excludes everything (LiDAR-support floor bites)."""
    thr_hi = AsrThresholds(tau_pts=100000, tau_clean=0.1, d_clean=2.0, n_min=1)
    detected_all = []
    for info in mini_val_info:
        idxs = [m for m in range(info["gt_boxes"].shape[0]) if int(info["gt_labels"][m]) == CAR]
        dec = _decode_from_info(info, detected_gt_idxs=idxs, score=0.9)
        detected_all.append(dec)
    sub = build_frozen_asr_subset(detected_all, thr_hi, "ckpt0", version="v1.0-mini", eval_set="mini_val")
    assert sub["n"] == 0, "τ_pts=100000 should exclude all cars on criterion 3"


def test_frozen_subset_hash_bind_and_verify(mini_val_info):
    """§0.5 — the subset is content-hashed + re-verified; a tampered subset/checkpoint is refused."""
    thr = AsrThresholds(tau_pts=1, tau_clean=0.1, d_clean=2.0, n_min=1)
    decodes = []
    for info in mini_val_info:
        idxs = [m for m in range(info["gt_boxes"].shape[0]) if int(info["gt_labels"][m]) == CAR]
        decodes.append(_decode_from_info(info, detected_gt_idxs=idxs, score=0.7))
    sub = build_frozen_asr_subset(decodes, thr, "CKSUM_ABC", version="v1.0-mini", eval_set="mini_val")
    assert sub["n"] >= 1
    verify_frozen_asr_subset(sub)  # passes

    # determinism: rebuild → identical hash.
    sub2 = build_frozen_asr_subset(decodes, thr, "CKSUM_ABC", version="v1.0-mini", eval_set="mini_val")
    assert sub2["content_hash"] == sub["content_hash"]

    # tamper the target set → verify RAISES.
    bad = copy.deepcopy(sub); bad["targets"] = bad["targets"][:-1]
    with pytest.raises(RuntimeError):
        verify_frozen_asr_subset(bad)
    # bind to checkpoint: a different checksum changes the hash (so a wrong-checkpoint subset is refused).
    sub_other = build_frozen_asr_subset(decodes, thr, "CKSUM_XYZ", version="v1.0-mini", eval_set="mini_val")
    assert sub_other["content_hash"] != sub["content_hash"]
    bad2 = copy.deepcopy(sub); bad2["checkpoint_checksum"] = "CKSUM_XYZ"
    with pytest.raises(RuntimeError):
        verify_frozen_asr_subset(bad2)


def test_false_disappearance_zero_on_clean_and_undefined_when_small(mini_val_info):
    """§0.3 — clean re-decode reproduces the subset detections (false-disappearance ≈ 0); and a
    denominator below N_min → UNDEFINED + gate FAILS (never vacuously near-zero)."""
    thr = AsrThresholds(tau_pts=1, tau_clean=0.1, d_clean=2.0, n_min=1, false_disappear_max=0.02)
    decodes = []
    for info in mini_val_info:
        idxs = [m for m in range(info["gt_boxes"].shape[0]) if int(info["gt_labels"][m]) == CAR]
        decodes.append(_decode_from_info(info, detected_gt_idxs=idxs, score=0.7))
    sub = build_frozen_asr_subset(decodes, thr, "ck", version="v1.0-mini", eval_set="mini_val")
    assert sub["n"] >= 1

    # clean re-decode = the SAME detections → all subset targets still detected → rate 0.
    fd = false_disappearance_baseline(sub, decodes, thr)
    assert fd["defined"] and fd["passed"] and fd["false_disappearance_rate"] == 0.0

    # disappearance metric on a pass where ALL detections are removed → ASR = 1.0.
    # (detection thresholds are taken from the subset's BOUND thresholds, not a passed thr.)
    empty = [_decode_from_info(info, detected_gt_idxs=()) for info in mini_val_info]
    dis = disappearance_asr(sub, empty)
    assert dis["asr"] == 1.0 and dis["disappeared"] == sub["n"]

    # N below N_min → UNDEFINED, gate fails.
    thr_hi = AsrThresholds(tau_pts=1, tau_clean=0.1, d_clean=2.0, n_min=10**9, false_disappear_max=0.02)
    sub_small = build_frozen_asr_subset(decodes, thr_hi, "ck", version="v1.0-mini", eval_set="mini_val")
    fd_small = false_disappearance_baseline(sub_small, decodes, thr_hi)
    assert not fd_small["defined"] and not fd_small["passed"] and fd_small["reason"] is not None


def test_disappearance_uses_subset_bound_thresholds(mini_val_info):
    """§0.5 — disappearance detection uses the subset's BOUND thresholds, not a live config.

    Build a subset with τ_clean=0.5; a re-decode with detections at score 0.4 (below the BOUND
    τ_clean) must count those targets as DISAPPEARED — even though disappearance_asr takes no thr
    (the binding comes from the subset, so T5 cannot accidentally score with mismatched thresholds)."""
    thr_build = AsrThresholds(tau_pts=1, tau_clean=0.5, d_clean=2.0, n_min=1)
    hi = []
    for info in mini_val_info:
        idxs = [m for m in range(info["gt_boxes"].shape[0]) if int(info["gt_labels"][m]) == CAR]
        hi.append(_decode_from_info(info, detected_gt_idxs=idxs, score=0.7))  # ≥ 0.5 → detected
    sub = build_frozen_asr_subset(hi, thr_build, "ck", version="v1.0-mini", eval_set="mini_val")
    assert sub["n"] >= 1
    # the reconstructed bound thresholds match what we built with
    bound = thresholds_from_subset(sub)
    assert bound.tau_clean == 0.5 and bound.d_clean == 2.0 and bound.target_class == "car"
    # re-decode at score 0.4 (< bound τ_clean) → ALL subset targets disappear (bound τ used, not 0.1)
    lo = []
    for info in mini_val_info:
        idxs = [m for m in range(info["gt_boxes"].shape[0]) if int(info["gt_labels"][m]) == CAR]
        lo.append(_decode_from_info(info, detected_gt_idxs=idxs, score=0.4))
    dis = disappearance_asr(sub, lo)
    assert dis["asr"] == 1.0, f"bound τ_clean=0.5 not enforced (got asr={dis['asr']})"


def test_phantom_slot_is_null_at_t4():
    slot = phantom_asr_slot()
    assert slot["phantom_asr"] is None and slot["available_at"] == "T5"
