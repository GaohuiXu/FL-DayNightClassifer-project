"""The 5-condition same-model fusion-awareness ablation (the crown jewel) (fl_v3 T5).

Certifies that the camera-only backdoor's success **genuinely depends on multimodal fusion** — on
the ONE trained poisoned model (NOT separate single-modality models), by input-conditioning, all
decodes at ``batch_size=1`` (the T4 batch-invariance protocol), all disappearance flags scored on
the **UNEDITED val GT** with the frozen subset's BOUND thresholds (§0.C7/§0.5):

  1. **cond-1** clean cam + clean LiDAR                    — the false-disappearance floor (≈0).
  2. **cond-2** triggered cam, **NON-target-aligned** (LiDAR-sparse) + clean LiDAR.
  3. **cond-3** clean cam + **LiDAR perturbation** = deterministic removal of the LiDAR points inside
     the target's GT box (target lidar-support removal; NOT whole-cloud zeroing / unseeded noise).
  4. **cond-4** triggered cam, **TARGET-aligned** (densest LiDAR projection) + clean LiDAR — the attack.
  5. **cond-5a** the SAME target-aligned trigger, **camera-only readout = ZERO the LiDAR-BEV input**
     via a ``forward_pre_hook`` on ``model.fusion`` (§0.B; a clean same-weights readout because the
     ``ConvFuser`` is bias-free Conv → GN-on-output). Two REQUIRED guards: a **LiDAR-invariance test**
     (head byte-identical for two different LiDAR inputs ⇒ LiDAR provably zeroed) AND a
     **clean-recall precondition** (the readout still detects clean cars).

**Fusion-aware iff** (pinned floor-corrected constants, §0.C4) ``ASR(cond4)−floor ≥ 0.3`` AND
``(ASR(cond4)−floor) − max_{j∈{2,3,5a}}(ASR(j)−floor) ≥ δ_fusion`` AND ``(ASR(cond4)−floor) ≥
2·max_j(ASR(j)−floor)``. A degenerate (cond-4 ≈ cond-2) result is a **surfaced D3 escape-hatch
finding**, not hidden.

The **occlusion control** decodes the SAME target-aligned triggered images through the **clean
pre-attack checkpoint** — its disappear-ASR must be ``< false_disappear_max`` (else the patch is an
occluder, not a learned backdoor → GATE FAILS).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fl_v3.attacks import trigger as TR
from fl_v3.attacks.poison import _lidar_xyz
from fl_v3.eval.asr import AsrThresholds, detected_target_anns, thresholds_from_subset
from fl_v3.eval.detection_eval import SampleDecode
from fl_v3.models.fusion.collate import detection_collate_fn

HEAD_KEYS = ("heatmap", "reg", "height", "dim", "rot", "vel")
CONDITIONS = ("cond1_clean", "cond2_nonaligned", "cond3_lidar_removed", "cond4_aligned", "cond5a_camera_only")


# ---------------------------------------------------------------------------
# cond-5a — the zero-LiDAR-BEV camera-only readout (forward hook + the efficient shared path).
# ---------------------------------------------------------------------------
def _zero_lidar_pre_hook(module, args):
    """forward_pre_hook on ``model.fusion``: replace the ``lidar_bev`` arg with ``zeros_like``.

    Because ``ConvFuser`` = ``concat → Conv2d(bias=False) → GroupNorm(over OUTPUT channels) → ReLU``,
    zeroing the LiDAR input contributes EXACTLY zero additively and does not cross-contaminate the
    camera channels — a true same-weights camera-only readout (§0.B)."""
    camera_bev, lidar_bev = args[0], args[1]
    return (camera_bev, torch.zeros_like(lidar_bev)) + tuple(args[2:])


def camera_only_readout(model, batch) -> List[Dict[str, torch.Tensor]]:
    """Run ``model(batch)`` with the LiDAR-BEV zeroed at the fusion input (the cond-5a readout)."""
    h = model.fusion.register_forward_pre_hook(_zero_lidar_pre_hook)
    try:
        return model(batch)
    finally:
        h.remove()


def _head_from_intermediates(model, camera_bev, lidar_bev) -> List[Dict[str, torch.Tensor]]:
    """``head(bev_neck(fusion(camera_bev, lidar_bev)))`` — used to derive cond-4 + cond-5a from ONE
    backbone forward (pass real ``lidar_bev`` for cond-4, ``zeros_like`` for cond-5a)."""
    return model.head(model.bev_neck(model.fusion(camera_bev, lidar_bev)))


# ---------------------------------------------------------------------------
# Per-sample decode plumbing (batch_size=1).
# ---------------------------------------------------------------------------
def _collate_one(sample: Dict[str, object], device) -> dict:
    batch = detection_collate_fn([sample])
    from fl_v3.training.loop import _move_to_device
    return _move_to_device(batch, device)


def _sample_decode(sample: Dict[str, object], decoded: dict) -> SampleDecode:
    """Build a :class:`SampleDecode` carrying the CLEAN (unedited) val GT + the CONDITIONED decode.

    The GT fed to eligibility/matching is always the unedited val GT (§0.C7); only ``boxes/scores/
    labels`` come from the conditioned forward."""
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return SampleDecode(
        sample_token=str(sample["sample_token"]),
        boxes=_np(decoded["boxes"]).astype(np.float64).reshape(-1, 7),
        scores=_np(decoded["scores"]).astype(np.float64).reshape(-1),
        labels=_np(decoded["labels"]).astype(np.int64).reshape(-1),
        velocity=_np(decoded["velocity"]).astype(np.float64).reshape(-1, 2),
        lidar2ego=_np(sample["lidar2ego"]).astype(np.float64).reshape(4, 4),
        ego2global_lidar=_np(sample["ego2global_lidar"]).astype(np.float64).reshape(4, 4),
        lidar2img=_np(sample["lidar2img"]).astype(np.float64).reshape(6, 4, 4),
        gt_boxes=_np(sample["gt_boxes"]).astype(np.float64).reshape(-1, 7),
        gt_labels=_np(sample["gt_labels"]).astype(np.int64).reshape(-1),
        gt_num_lidar_pts=_np(sample["gt_num_lidar_pts"]).astype(np.int64).reshape(-1),
        gt_in_range=_np(sample["gt_in_range"]).astype(bool).reshape(-1),
        gt_velocity=_np(sample["gt_velocity"]).astype(np.float64).reshape(-1, 2),
        gt_ann_tokens=list(sample["gt_ann_tokens"]),
        gt_names=list(sample["gt_names"]),
        gt_attribute=list(sample["gt_attribute"]),
    )


@torch.no_grad()
def _decode(model, sample, device, thr_score: float) -> dict:
    batch = _collate_one(sample, device)
    head = model(batch)
    return model.decode(head, score_threshold=thr_score)[0]


@torch.no_grad()
def _decode_cond4_and_cond5a(model, sample, device, thr_score: float):
    """ONE backbone forward → (cond-4 decoded, cond-5a decoded) — shares the frozen Swin-T cost."""
    batch = _collate_one(sample, device)
    out = model(batch, return_intermediates=True)
    head4 = out["task_outputs"]
    dec4 = model.decode(head4, score_threshold=thr_score)[0]
    head5 = _head_from_intermediates(model, out["_camera_bev"], torch.zeros_like(out["_lidar_bev"]))
    dec5 = model.decode(head5, score_threshold=thr_score)[0]
    return dec4, dec5


# ---------------------------------------------------------------------------
# Condition input builders (per target).
# ---------------------------------------------------------------------------
def _remove_lidar_in_box(sample: Dict[str, object], box7) -> Dict[str, object]:
    """cond-3: a NEW sample with the LiDAR points inside ``box7`` removed (no camera trigger)."""
    pts = sample["lidar_points"]
    arr = pts.cpu().numpy() if hasattr(pts, "cpu") else np.asarray(pts)
    inside = TR.points_in_box7(arr, box7)
    out = dict(sample)
    keep = arr[~inside]
    out["lidar_points"] = torch.from_numpy(np.ascontiguousarray(keep)).to(pts.dtype) if hasattr(pts, "dtype") else keep
    return out


def target_index(sample: Dict[str, object], ann_token: str) -> Optional[int]:
    anns = list(sample["gt_ann_tokens"])
    return anns.index(ann_token) if ann_token in anns else None


# ---------------------------------------------------------------------------
# The per-target 5-condition evaluation (+ occlusion via the clean model).
# ---------------------------------------------------------------------------
@dataclass
class TargetVerdict:
    sample_token: str
    ann_token: str
    placement_aligned_ok: bool          # cond-4 centre ≤ center_max_px of box-centre proj
    placement_nonaligned_iou0: bool     # cond-2 footprint IoU==0 with all targets
    area_ratio: Optional[float]         # cond-4 patch/box-2D area ratio (≤ budget_frac GATE)
    disappeared: Dict[str, bool]        # per-condition (cond1..cond5a)
    occlusion_disappeared: Optional[bool]   # cond-4 trigger on the CLEAN model


def evaluate_target_cond4(
    sample_clean: Dict[str, object], ann_token: str, model, device, thr: AsrThresholds,
    spec: TR.TriggerSpec, clean_anns: Optional[set] = None,
) -> Optional[TargetVerdict]:
    """The LEAN control measurement: ONLY cond-1 (clean, cached) + cond-4 (target-aligned trigger) for
    ONE target — the headline disappear-ASR for the trigger_only / label_only / delete control
    checkpoints (their fusion-awareness is not asked; just whether the trigger disappears the car)."""
    m = target_index(sample_clean, ann_token)
    if m is None:
        return None
    box7 = sample_clean["gt_boxes"].cpu().numpy()[m]
    pl_aligned = TR.compute_aligned_placement(box7, _lidar_xyz(sample_clean),
                                              sample_clean["lidar2img"].cpu().numpy(), spec)
    if pl_aligned is None:
        return None
    aligned_sample = TR.apply_trigger(sample_clean, pl_aligned, spec)
    if clean_anns is None:
        clean_anns = clean_detected_anns(model, sample_clean, device, thr)
    dec4 = _decode(model, aligned_sample, device, thr.tau_clean)
    disappeared = {c: False for c in CONDITIONS}
    disappeared["cond1_clean"] = ann_token not in clean_anns
    disappeared["cond4_aligned"] = ann_token not in detected_target_anns(_sample_decode(aligned_sample, dec4), thr)
    bc = pl_aligned.box_center_uv
    aligned_ok = float(np.hypot(pl_aligned.cx_px - bc[0], pl_aligned.cy_px - bc[1])) <= spec.center_max_px + 1e-6
    area_ratio = (pl_aligned.patch_area / pl_aligned.box2d_area) if pl_aligned.box2d_area > 0 else None
    return TargetVerdict(sample_token=str(sample_clean["sample_token"]), ann_token=ann_token,
                         placement_aligned_ok=bool(aligned_ok), placement_nonaligned_iou0=True,
                         area_ratio=area_ratio, disappeared=disappeared, occlusion_disappeared=None)


def clean_detected_anns(poisoned_model, sample_clean, device, thr: AsrThresholds):
    """The set of target-class ann_tokens the POISONED model detects on the CLEAN sample (cond-1 /
    the per-sample floor) — cached by the driver and shared across all the sample's targets."""
    dec1 = _decode(poisoned_model, sample_clean, device, thr.tau_clean)
    return detected_target_anns(_sample_decode(sample_clean, dec1), thr)


def evaluate_target(
    sample_clean: Dict[str, object], ann_token: str,
    poisoned_model, clean_model, device, thr: AsrThresholds, spec: TR.TriggerSpec,
    clean_anns: Optional[set] = None,
) -> Optional[TargetVerdict]:
    """All 5 conditions + the occlusion control for ONE frozen-subset target (batch_size=1).

    ``clean_anns`` (optional) = the precomputed cond-1 detected set for this sample (avoids a
    redundant clean decode when a sample carries several subset targets)."""
    m = target_index(sample_clean, ann_token)
    if m is None:
        return None
    box7 = sample_clean["gt_boxes"].cpu().numpy()[m]
    lidar_xyz = _lidar_xyz(sample_clean)
    l2i = sample_clean["lidar2img"].cpu().numpy()
    target_boxes = sample_clean["gt_boxes"].cpu().numpy()  # all GT (cond-2 IoU set)
    target_boxes = target_boxes[sample_clean["gt_labels"].cpu().numpy() == thr.target_id]

    pl_aligned = TR.compute_aligned_placement(box7, lidar_xyz, l2i, spec)
    pl_nonaligned = TR.compute_nonaligned_placement(box7, target_boxes, l2i, spec)
    if pl_aligned is None:
        return None

    aligned_sample = TR.apply_trigger(sample_clean, pl_aligned, spec)
    thr_score = thr.tau_clean

    disappeared: Dict[str, bool] = {}
    # cond-1 clean (the floor) — use the cached per-sample detected set when available
    if clean_anns is None:
        clean_anns = clean_detected_anns(poisoned_model, sample_clean, device, thr)
    disappeared["cond1_clean"] = ann_token not in clean_anns
    # cond-4 (aligned) + cond-5a (camera-only) from ONE forward
    dec4, dec5 = _decode_cond4_and_cond5a(poisoned_model, aligned_sample, device, thr_score)
    disappeared["cond4_aligned"] = ann_token not in detected_target_anns(_sample_decode(aligned_sample, dec4), thr)
    disappeared["cond5a_camera_only"] = ann_token not in detected_target_anns(_sample_decode(aligned_sample, dec5), thr)
    # cond-2 (non-aligned trigger)
    if pl_nonaligned is not None:
        nonaligned_sample = TR.apply_trigger(sample_clean, pl_nonaligned, spec)
        dec2 = _decode(poisoned_model, nonaligned_sample, device, thr_score)
        disappeared["cond2_nonaligned"] = ann_token not in detected_target_anns(_sample_decode(nonaligned_sample, dec2), thr)
        nonaligned_iou0 = True
    else:
        disappeared["cond2_nonaligned"] = False  # no valid non-aligned region → not disappeared (conservative)
        nonaligned_iou0 = False
    # cond-3 (LiDAR removed, no trigger)
    lidarrm_sample = _remove_lidar_in_box(sample_clean, box7)
    dec3 = _decode(poisoned_model, lidarrm_sample, device, thr_score)
    disappeared["cond3_lidar_removed"] = ann_token not in detected_target_anns(_sample_decode(lidarrm_sample, dec3), thr)

    # occlusion control: cond-4 triggered images through the CLEAN pre-attack model
    occ = None
    if clean_model is not None:
        deco = _decode(clean_model, aligned_sample, device, thr_score)
        occ = ann_token not in detected_target_anns(_sample_decode(aligned_sample, deco), thr)

    # objective placement checks
    bc = pl_aligned.box_center_uv
    aligned_ok = float(np.hypot(pl_aligned.cx_px - bc[0], pl_aligned.cy_px - bc[1])) <= spec.center_max_px + 1e-6
    area_ratio = (pl_aligned.patch_area / pl_aligned.box2d_area) if pl_aligned.box2d_area > 0 else None
    return TargetVerdict(
        sample_token=str(sample_clean["sample_token"]), ann_token=ann_token,
        placement_aligned_ok=bool(aligned_ok), placement_nonaligned_iou0=bool(nonaligned_iou0),
        area_ratio=area_ratio, disappeared=disappeared, occlusion_disappeared=occ,
    )


# ---------------------------------------------------------------------------
# cond-5a guards.
# ---------------------------------------------------------------------------
@torch.no_grad()
def lidar_invariance_check(model, sample, device) -> Dict[str, object]:
    """Guard (i): the camera-only readout's head must be byte-identical for two DIFFERENT LiDAR inputs
    ⇒ LiDAR is provably zeroed. Runs the readout with the real points and with a perturbed cloud."""
    batch_a = _collate_one(sample, device)
    pert = dict(sample)
    pts = sample["lidar_points"]
    arr = (pts.cpu().numpy() if hasattr(pts, "cpu") else np.asarray(pts)).copy()
    if arr.shape[0] > 0:
        arr[:, 1:4] = arr[:, 1:4] * 2.0 + 3.0  # perturb x,y,z,intensity region (cols 1..3 used)
    pert["lidar_points"] = torch.from_numpy(np.ascontiguousarray(arr)).to(pts.dtype)
    batch_b = _collate_one(pert, device)
    ha = camera_only_readout(model, batch_a)
    hb = camera_only_readout(model, batch_b)
    max_abs = 0.0
    if len(ha) != 6 or len(hb) != 6:
        raise RuntimeError("cond-5a requires the reviewed six-task CenterHead contract")
    for task_a, task_b in zip(ha, hb, strict=True):
        for key in HEAD_KEYS:
            d = (task_a[key] - task_b[key]).abs().max().item()
            max_abs = max(max_abs, float(d))
    return {"lidar_invariant": bool(max_abs == 0.0), "max_abs_head_diff": max_abs}


@torch.no_grad()
def camera_only_clean_recall(model, samples_with_targets, device, thr: AsrThresholds) -> Dict[str, object]:
    """Guard (ii): the camera-only readout's CLEAN (untriggered) car-recall over the given (sample,
    [ann_tokens]) list — must clear a declared floor, else a low ASR(cond-5a) is a capability artifact
    (the readout can't detect cars at all), not the trigger losing its fusion handle."""
    detected = 0; total = 0
    for sample, ann_tokens in samples_with_targets:
        batch = _collate_one(sample, device)
        head = camera_only_readout(model, batch)
        dec = model.decode(head, score_threshold=thr.tau_clean)[0]
        found = detected_target_anns(_sample_decode(sample, dec), thr)
        for a in ann_tokens:
            total += 1
            if a in found:
                detected += 1
    return {"camera_only_clean_recall": (detected / total) if total else 0.0, "detected": detected, "total": total}


# ---------------------------------------------------------------------------
# Aggregation → the 5-condition table + the fusion-aware verdict.
# ---------------------------------------------------------------------------
def fusion_aware_verdict(asr: Dict[str, float], *, delta_fusion: float, mult: float,
                         viability: float = 0.3, cond5a_guards_valid: Optional[bool] = None) -> Dict[str, object]:
    """Floor-correct (subtract cond-1) and apply the PINNED criteria (§0.C4). ``asr`` is the raw
    per-condition disappear-ASR (cond1..cond5a).

    ``cond5a_guards_valid`` (§0.B): cond-5a is a valid fusion-awareness CONTROL only if it passed the
    LiDAR-invariance test + the clean-recall precondition. When provided and ``False``, the
    ``fusion_aware`` claim is REFUSED (the cond-4≫cond-5a comparison would be meaningless). ``None`` ⇒
    the guards have not been run yet (the driver treats this as incomplete, NOT as valid)."""
    floor = float(asr["cond1_clean"])
    corr = {c: float(asr[c]) - floor for c in CONDITIONS}
    others = {j: corr[j] for j in ("cond2_nonaligned", "cond3_lidar_removed", "cond5a_camera_only")}
    max_other = max(others.values())
    c4 = corr["cond4_aligned"]
    viable = c4 >= viability
    margin_ok = (c4 - max_other) >= delta_fusion
    mult_ok = c4 >= mult * max_other
    cond5a_ok = (cond5a_guards_valid is True)
    fusion_aware = bool(viable and margin_ok and mult_ok and cond5a_ok)
    degenerate = bool(c4 > 0 and others["cond2_nonaligned"] >= c4 - 1e-9)  # cond-4 ≈ cond-2
    return {
        "floor": floor,
        "corrected_asr": corr,
        "max_other_corrected": max_other,
        "cond4_corrected": c4,
        "viable_cond4_ge_0.3": bool(viable),
        "margin_cond4_minus_maxother": c4 - max_other,
        "margin_ok_ge_delta_fusion": bool(margin_ok),
        "mult_ok_cond4_ge_2x_maxother": bool(mult_ok),
        "cond5a_guards_valid": cond5a_guards_valid,
        "delta_fusion": float(delta_fusion),
        "mult": float(mult),
        "fusion_aware": fusion_aware,
        "degenerate_cond4_approx_cond2": degenerate,
        "verdict": ("FUSION-AWARE" if fusion_aware else
                    ("D3-ESCAPE-HATCH (cond4≈cond2 — point-decoration)" if degenerate else "NOT-FUSION-AWARE")),
    }
