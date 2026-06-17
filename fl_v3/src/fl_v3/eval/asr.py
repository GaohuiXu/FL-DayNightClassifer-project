"""ASR-eligibility harness — disappearance metric, frozen subset, baselines (fl_v3 T4).

The strict attack-success machinery (D4: **disappearance** primary, phantom secondary), built +
validated on CLEAN data so T5 plugs a trigger straight in. The contracts (T4_SPEC §0.3/§0.4/§0.5):

  * **6-criterion eligibility** (§Attack spec; all thresholds config-pinned): (1) class = car
    (D8); (2) frustum-visible in ≥1 camera (:mod:`frustum_visibility` — trigger-invariant); (3)
    LiDAR support ``gt_num_lidar_pts ≥ τ_pts``; (4) ``gt_in_range`` (devkit-exact ``ego_dist <
    class_range``); (5) the CLEAN model detects a target-class box at score ≥ ``τ_clean``; (6)
    that detection matches the GT within ``d_clean`` (greedy center-distance, lidar-frame XY —
    the same primitive as ``training.tasks.center_distance_proxy``). ``τ_pts ≥ 1`` subsumes the
    devkit ``filter_eval_boxes`` ``num_pts(lidar+radar)>0`` GT filter for cars (lidar ≥ τ_pts ≥ 1
    ⇒ lidar+radar > 0); the bike-rack filter is car-irrelevant (applied only for bicycle/
    motorcycle, via the devkit, when ``nusc`` is supplied).
  * **The ASR denominator is ``eligible-CLEAN-DETECTED``** (ALL 6, incl. 5+6), NOT all-eligible —
    a target that satisfies (1)–(4) but is not clean-detected is EXCLUDED from ``N`` (asserted +
    tested). ``N`` is both the frozen-subset size and the disappearance-ASR denominator.
  * **The frozen held-out subset** is content-hashed over ``(sorted (sample_token, ann_token),
    the declared thresholds, the clean-checkpoint FL_TRAINABLE_CHECKSUM)`` and re-verified at load
    time (reused by T5) — so every T5/T6/T7 cell scores the IDENTICAL targets.
  * **Disappearance ASR** = ``disappeared / N`` (applied to triggered inputs at T5); T4 validates
    it on clean. **False-disappearance baseline** = the fraction of subset targets the model
    misses with NO trigger — ``< false_disappear_max`` AND valid only when ``N ≥ N_min`` (else
    UNDEFINED → gate FAILS, never vacuously near-zero).
  * **Phantom-ASR** (D4 secondary): the metric definition + reporting slot only; injection is T5.

ASR is defined ONLY on triggered inputs; T4 invokes no attack.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from fl_v3.data.nuscenes.class_map import NAME_TO_ID, PRIMARY_TARGET_ID, PRIMARY_TARGET_NAME
from fl_v3.eval.frustum_visibility import DEFAULT_IMAGE_HW, is_frustum_visible

# Bumped whenever the eligibility/hash semantics change (invalidates older frozen subsets).
ASR_SUBSET_FORMAT = "t4.asr.v1"

# Criterion keys (order is the §Attack-spec order; reported per target).
CRITERIA = ("c1_class", "c2_frustum", "c3_lidar_pts", "c4_in_range",
            "c5_clean_score", "c6_clean_match")


@dataclass(frozen=True)
class AsrThresholds:
    """The pinned ASR thresholds (T4_SPEC §0.3) — defaults are placeholders; the readiness path
    loads the config-pinned values via :func:`asr_thresholds_from_run_config`."""

    target_class: str = PRIMARY_TARGET_NAME          # D8 = car
    tau_pts: int = 10                                # LiDAR-support floor (≥1 ⇒ devkit num_pts>0)
    tau_clean: float = 0.10                          # = production decode score threshold
    d_clean: float = 2.0                             # clean-match radius (m); == devkit dist_th_tp
    image_h: int = DEFAULT_IMAGE_HW[0]
    image_w: int = DEFAULT_IMAGE_HW[1]
    n_min: int = 150                                 # min eligible-clean-detected for power
    recall_floor: float = 0.20                       # on OFFICIAL car recall
    false_disappear_max: float = 0.02

    @property
    def target_id(self) -> int:
        return NAME_TO_ID[self.target_class]

    @property
    def image_hw(self) -> Tuple[int, int]:
        return (int(self.image_h), int(self.image_w))


def asr_thresholds_from_run_config(run_config: dict) -> AsrThresholds:
    """Build :class:`AsrThresholds` from the flat run-config (the pyproject-pinned keys)."""
    target = str(run_config.get("asr-target-class", PRIMARY_TARGET_NAME))
    tau_pts = int(run_config.get("asr-tau-pts", 10))
    if tau_pts < 1:
        raise ValueError(f"asr-tau-pts must be ≥1 (subsumes the devkit num_pts>0 filter); got {tau_pts}")
    return AsrThresholds(
        target_class=target,
        tau_pts=tau_pts,
        tau_clean=float(run_config.get("asr-tau-clean", run_config.get("det-score-threshold", 0.1))),
        d_clean=float(run_config.get("asr-d-clean", 2.0)),
        image_h=int(run_config.get("asr-image-h", DEFAULT_IMAGE_HW[0])),
        image_w=int(run_config.get("asr-image-w", DEFAULT_IMAGE_HW[1])),
        n_min=int(run_config.get("asr-n-min", 150)),
        recall_floor=float(run_config.get("asr-recall-floor", 0.20)),
        false_disappear_max=float(run_config.get("asr-false-disappear-max", 0.02)),
    )


# ---------------------------------------------------------------------------
# The shared clean-detection matcher (criteria 5+6) — the disappearance primitive.
# ---------------------------------------------------------------------------
def detected_target_gt(decode, thr: AsrThresholds) -> Dict[str, dict]:
    """Greedy-match target-class GT → score-eligible decoded target boxes (lidar-frame XY L2).

    Returns ``{ann_token: {"matched": bool, "dist": float, "det_score": float, "gt_idx": int}}``
    for EVERY target-class GT (matched = within ``d_clean`` of a decoded target box with
    score ≥ ``τ_clean``). Greedy nearest, each decoded box used once, GT processed in the
    info (ann-token-sorted) order — bit-deterministic. The same primitive as
    ``center_distance_proxy`` but returning per-GT match identity (which the ASR denominator
    needs).
    """
    tid = thr.target_id
    # GT processed in ann-token order (T1 already ann-token-sorts the schema, so this is the info
    # order; sort EXPLICITLY so the greedy matcher is reproducible even for a hand-built decode).
    def _ann(m):
        return decode.gt_ann_tokens[m] if m < len(decode.gt_ann_tokens) else f"_idx{m}"
    gt_idxs = sorted((m for m in range(decode.gt_boxes.shape[0]) if int(decode.gt_labels[m]) == tid),
                     key=_ann)
    # score-eligible decoded target boxes
    det_mask = (decode.labels == tid) & (decode.scores >= thr.tau_clean)
    det_xy = decode.boxes[det_mask][:, :2].astype(np.float64)
    det_scores = decode.scores[det_mask].astype(np.float64)
    used = np.zeros(det_xy.shape[0], dtype=bool)
    out: Dict[str, dict] = {}
    for m in gt_idxs:
        ann = decode.gt_ann_tokens[m] if m < len(decode.gt_ann_tokens) else f"_idx{m}"
        gxy = decode.gt_boxes[m, :2].astype(np.float64)
        matched, dist, sc = False, float("inf"), 0.0
        if det_xy.shape[0] > 0:
            d = np.linalg.norm(det_xy - gxy[None, :], axis=1)
            d_masked = np.where(used, np.inf, d)
            j = int(np.argmin(d_masked))
            best = float(d_masked[j])
            if np.isfinite(best):
                dist = best
                if best <= thr.d_clean:
                    matched = True
                    used[j] = True
                    sc = float(det_scores[j])
        out[ann] = {"matched": bool(matched), "dist": dist, "det_score": sc, "gt_idx": int(m)}
    return out


def detected_target_anns(decode, thr: AsrThresholds) -> Set[str]:
    """Set of target-class GT ann_tokens the clean model detects (criteria 5+6 satisfied)."""
    return {a for a, r in detected_target_gt(decode, thr).items() if r["matched"]}


# ---------------------------------------------------------------------------
# 6-criterion eligibility.
# ---------------------------------------------------------------------------
def evaluate_sample_eligibility(decode, thr: AsrThresholds, nusc=None) -> List[dict]:
    """Per target-class GT in one sample: the 6 criteria booleans + the eligible-clean-detected flag.

    ``eligible`` = ALL of c1–c4 AND clean-detected (c5 score ≥ τ_clean AND c6 match ≤ d_clean).
    The bike-rack filter (devkit) is applied only for bicycle/motorcycle (car-irrelevant); needs
    ``nusc`` for the rack polygons — skipped (with the field recorded True) for car / when absent.
    """
    tid = thr.target_id
    match = detected_target_gt(decode, thr)
    bikerack_drop = _bikerack_drops(decode, thr, nusc) if thr.target_class in ("bicycle", "motorcycle") else set()
    rows: List[dict] = []
    for m in range(decode.gt_boxes.shape[0]):
        if int(decode.gt_labels[m]) != tid:
            continue
        ann = decode.gt_ann_tokens[m] if m < len(decode.gt_ann_tokens) else f"_idx{m}"
        mr = match.get(ann, {"matched": False, "dist": float("inf"), "det_score": 0.0})
        c1 = True  # class == target by construction of this loop
        c2 = bool(is_frustum_visible(decode.gt_boxes[m], decode.lidar2img, thr.image_hw))
        c3 = bool(int(decode.gt_num_lidar_pts[m]) >= thr.tau_pts)
        c4 = bool(decode.gt_in_range[m])
        c5 = bool(mr["det_score"] >= thr.tau_clean and mr["matched"])  # a score-eligible det matched
        c6 = bool(mr["matched"])                                       # within d_clean
        not_in_rack = ann not in bikerack_drop
        eligible = bool(c1 and c2 and c3 and c4 and c5 and c6 and not_in_rack)
        rows.append({
            "sample_token": decode.sample_token, "ann_token": ann, "gt_idx": int(m),
            "c1_class": c1, "c2_frustum": c2, "c3_lidar_pts": c3, "c4_in_range": c4,
            "c5_clean_score": c5, "c6_clean_match": c6, "not_in_bikerack": not_in_rack,
            "clean_detected": bool(mr["matched"]),
            "match_dist": float(mr["dist"]) if np.isfinite(mr["dist"]) else None,
            "match_score": float(mr["det_score"]),
            "gt_num_lidar_pts": int(decode.gt_num_lidar_pts[m]),
            "eligible": eligible,
        })
    return rows


def _bikerack_drops(decode, thr: AsrThresholds, nusc) -> Set[str]:
    """ann_tokens of bicycle/motorcycle GT whose center sits in a ``static_object.bicycle_rack``
    (the devkit ``filter_eval_boxes`` bike-rack filter). Requires ``nusc``; empty if absent."""
    if nusc is None:
        return set()
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import points_in_box
    from pyquaternion import Quaternion

    sample = nusc.get("sample", decode.sample_token)
    racks = [nusc.get("sample_annotation", a) for a in sample["anns"]
             if nusc.get("sample_annotation", a)["category_name"] == "static_object.bicycle_rack"]
    rack_boxes = [Box(r["translation"], r["size"], Quaternion(r["rotation"])) for r in racks]
    if not rack_boxes:
        return set()
    drops: Set[str] = set()
    tid = thr.target_id
    # bike-rack test is in GLOBAL frame; convert each GT center to global.
    from fl_v3.eval.box_to_global import convert_box_to_global
    for m in range(decode.gt_boxes.shape[0]):
        if int(decode.gt_labels[m]) != tid:
            continue
        conv = convert_box_to_global(decode.gt_boxes[m], decode.lidar2ego, decode.ego2global_lidar)
        p = np.asarray(conv["translation"], np.float64).reshape(3, 1)
        if any(int(np.sum(points_in_box(rb, p))) > 0 for rb in rack_boxes):
            ann = decode.gt_ann_tokens[m] if m < len(decode.gt_ann_tokens) else f"_idx{m}"
            drops.add(ann)
    return drops


# ---------------------------------------------------------------------------
# The frozen ASR subset (content-hashed + bound to the checkpoint + thresholds).
# ---------------------------------------------------------------------------
def _thresholds_fingerprint(thr: AsrThresholds) -> dict:
    """The threshold fields that the content hash binds (sorted, JSON-stable)."""
    return {
        "format": ASR_SUBSET_FORMAT,
        "target_class": thr.target_class,
        "tau_pts": int(thr.tau_pts),
        "tau_clean": float(thr.tau_clean),
        "d_clean": float(thr.d_clean),
        "image_hw": [int(thr.image_h), int(thr.image_w)],
    }


def subset_content_hash(targets: Sequence[Tuple[str, str]], thr: AsrThresholds,
                        checkpoint_checksum: str) -> str:
    """sha256 over (sorted (sample_token, ann_token), thresholds, clean-checkpoint checksum)."""
    payload = {
        "targets": sorted([[str(s), str(a)] for s, a in targets]),
        "thresholds": _thresholds_fingerprint(thr),
        "checkpoint_checksum": str(checkpoint_checksum),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_frozen_asr_subset(
    decodes: Sequence,
    thr: AsrThresholds,
    checkpoint_checksum: str,
    *,
    version: str,
    eval_set: str,
    nusc=None,
) -> Dict[str, object]:
    """Build the frozen eligible-clean-detected subset (the ASR denominator) + bind its hash.

    ``N`` = the count of targets satisfying ALL 6 criteria. Built on the FULL ``eval_set`` (never
    a ``det-eval-limit`` prefix — the caller passes the full-split decodes). Returns a
    serializable dict carrying the targets, the eligibility-criteria tallies, the thresholds,
    the checkpoint checksum, and the content hash (re-verified at load by :func:`verify_*`).
    """
    targets: List[Tuple[str, str]] = []
    crit_tally = {k: 0 for k in CRITERIA}
    crit_tally.update({"clean_detected": 0, "eligible": 0, "n_target_gt": 0})
    per_target: List[dict] = []
    for d in decodes:
        for row in evaluate_sample_eligibility(d, thr, nusc=nusc):
            crit_tally["n_target_gt"] += 1
            for k in CRITERIA:
                crit_tally[k] += int(row[k])
            crit_tally["clean_detected"] += int(row["clean_detected"])
            if row["eligible"]:
                crit_tally["eligible"] += 1
                targets.append((row["sample_token"], row["ann_token"]))
                per_target.append({"sample_token": row["sample_token"], "ann_token": row["ann_token"],
                                   "match_dist": row["match_dist"], "match_score": row["match_score"],
                                   "gt_num_lidar_pts": row["gt_num_lidar_pts"]})
    targets_sorted = sorted(targets)
    content_hash = subset_content_hash(targets_sorted, thr, checkpoint_checksum)
    return {
        "format": ASR_SUBSET_FORMAT,
        "version": version,
        "eval_set": eval_set,
        "scale": "trainval-scientific" if version == "v1.0-trainval" else "mini-smoke",
        "thresholds": _thresholds_fingerprint(thr),
        "thresholds_full": {**asdict(thr)},
        "checkpoint_checksum": str(checkpoint_checksum),
        "targets": [[s, a] for s, a in targets_sorted],
        "per_target": sorted(per_target, key=lambda r: (r["sample_token"], r["ann_token"])),
        "n": len(targets_sorted),
        "eligibility_counts": crit_tally,
        "content_hash": content_hash,
    }


def thresholds_from_subset(subset: dict) -> AsrThresholds:
    """Reconstruct the BOUND (hashed) thresholds from a frozen subset.

    Only the fields the content hash binds — ``target_class``, ``tau_pts``, ``tau_clean``,
    ``d_clean``, ``image_hw`` — are recovered (the readiness floors ``n_min``/``recall_floor``/
    ``false_disappear_max`` are policy, not part of the target-set binding, so they keep their
    defaults here). Used by :func:`verify_frozen_asr_subset` AND :func:`disappearance_asr` so the
    disappearance/false-disappearance detection scores the IDENTICAL targets the subset was frozen
    with (T4_SPEC §0.5), regardless of any live config the caller holds.
    """
    fp = subset["thresholds"]
    return AsrThresholds(
        target_class=fp["target_class"], tau_pts=int(fp["tau_pts"]),
        tau_clean=float(fp["tau_clean"]), d_clean=float(fp["d_clean"]),
        image_h=int(fp["image_hw"][0]), image_w=int(fp["image_hw"][1]),
    )


def verify_frozen_asr_subset(subset: dict) -> None:
    """Recompute the content hash and RAISE if it differs (the T5-reused load-time guard).

    Binds the subset to its exact ``(targets, thresholds, checkpoint checksum)`` — so a T5/T6/T7
    run that loads a subset built for a different checkpoint or thresholds refuses to proceed.
    """
    thr = thresholds_from_subset(subset)
    recomputed = subset_content_hash([tuple(t) for t in subset["targets"]], thr,
                                     subset["checkpoint_checksum"])
    if recomputed != subset["content_hash"]:
        raise RuntimeError(
            "frozen ASR subset content-hash MISMATCH — refusing to run (the subset was built for "
            f"different targets/thresholds/checkpoint).\n  stored     = {subset['content_hash']}\n"
            f"  recomputed = {recomputed}"
        )


# ---------------------------------------------------------------------------
# Disappearance metric + false-disappearance baseline (+ phantom slot).
# ---------------------------------------------------------------------------
def disappearance_asr(subset: dict, decodes: Sequence) -> Dict[str, object]:
    """Disappearance ASR = (# subset targets NOT detected on these inputs) / N.

    At T5 ``decodes`` are the TRIGGERED-input decodes; at T4 the CLEAN re-decode (→ the
    false-disappearance baseline). ``N`` = ``subset['n']`` (the frozen eligible-clean-detected
    denominator). Detection uses the subset's BOUND thresholds (``thresholds_from_subset``), NOT a
    caller-supplied thr — so "still detected" is decided with the IDENTICAL τ_clean/d_clean/target
    the subset was frozen with (T4_SPEC §0.5), and the metric is reproducible across T5/T6/T7.
    """
    verify_frozen_asr_subset(subset)
    thr = thresholds_from_subset(subset)
    by_token: Dict[str, object] = {d.sample_token: d for d in decodes}
    detected_cache: Dict[str, Set[str]] = {}
    n = int(subset["n"])
    disappeared = 0
    missing_sample = 0
    per_target = []
    for s_tok, a_tok in subset["targets"]:
        d = by_token.get(s_tok)
        if d is None:
            missing_sample += 1
            disappeared += 1  # a target whose sample is absent from this pass counts as disappeared
            per_target.append({"sample_token": s_tok, "ann_token": a_tok, "disappeared": True,
                               "reason": "sample_absent"})
            continue
        if s_tok not in detected_cache:
            detected_cache[s_tok] = detected_target_anns(d, thr)
        present = a_tok in detected_cache[s_tok]
        if not present:
            disappeared += 1
        per_target.append({"sample_token": s_tok, "ann_token": a_tok, "disappeared": not present})
    asr = (disappeared / n) if n > 0 else None
    return {
        "n": n, "disappeared": disappeared, "present": n - disappeared,
        "asr": asr, "missing_samples": missing_sample, "per_target": per_target,
    }


def false_disappearance_baseline(subset: dict, clean_decodes: Sequence,
                                 thr: AsrThresholds) -> Dict[str, object]:
    """The clean/no-trigger false-disappearance rate + its validity gate.

    ``< false_disappear_max`` AND valid only when ``N ≥ N_min`` (else ``defined=False`` →
    the readiness gate FAILS, never vacuously near-zero). With a deterministic decode and the
    subset's own checkpoint, a clean re-decode reproduces the subset detections ⇒ rate ≈ 0;
    a nonzero rate signals a determinism / harness bug. Detection uses the subset's BOUND
    thresholds; only the GATE floors (``n_min``, ``false_disappear_max``) come from ``thr``.
    """
    res = disappearance_asr(subset, clean_decodes)
    n = res["n"]
    rate = res["asr"]
    defined = bool(n >= thr.n_min)
    passed = bool(defined and rate is not None and rate < thr.false_disappear_max)
    return {
        "false_disappearance_rate": rate,
        "n": n, "n_min": thr.n_min,
        "threshold": thr.false_disappear_max,
        "defined": defined,
        "passed": passed,
        "reason": (None if defined else f"denominator N={n} < N_min={thr.n_min} → UNDEFINED"),
        "disappeared": res["disappeared"],
    }


# Phantom-ASR (D4 secondary) — definition + reporting slot only; injection is T5.
PHANTOM_ASR_DEFINITION = (
    "phantom_asr = (# triggered samples where the model emits a target-class detection at score "
    "≥ τ_clean in a designated phantom region with NO matching GT) / (# triggered samples). "
    "Defined on TRIGGERED inputs only; the phantom-region + trigger are injected at T5. T4 "
    "reserves the metric slot and reports phantom_asr = None (no attack)."
)


def phantom_asr_slot() -> Dict[str, object]:
    """The T4 phantom-ASR reporting slot (definition + null value; injection at T5)."""
    return {"phantom_asr": None, "definition": PHANTOM_ASR_DEFINITION, "available_at": "T5"}
