"""DT4-A — the benchmark-readiness eval (fl_v3 T4). Run on the A40 via run_t4_readiness_eval_a40.sh.

Loads the FULL-participation (D10) log-group trainval clean FedAvg checkpoint and, on the held-out
``val`` split, computes everything the T5 go/no-go needs — all from a SINGLE shared decode pass
(``decode_eval_set``):

  1. The official ``DetectionEval`` mAP / NDS / per-class AP / 5 TP errors + the OFFICIAL car recall.
  2. (optional) The GT-as-pred AP≈1 sanity at trainval scale (the §0.1 conversion self-check).
  3. The 6-criterion ASR eligibility → the frozen eligible-clean-detected subset, content-hashed +
     **bound to this checkpoint's FL_TRAINABLE_CHECKSUM** (the T5-reused contract).
  4. The clean false-disappearance baseline (a 2nd fresh decode over the subset's samples — a real
     determinism re-check), valid only when ``N ≥ N_min``.
  5. ``benchmark_readiness.json``: ``READY`` iff ``eligible_count ≥ N_min`` AND
     ``clean_car_recall > recall_floor`` AND the false-disappearance gate passes — else ``NOT-READY``
     with the gap + the recommended strengthening (a VALID, surfaced outcome that gates T5).
  6. The frozen 6-tuple clean report cell + V4 for a few eligible samples.

A ``mini_val`` (2-scene) verdict is ``scale=mini`` and is explicitly NOT a go/no-go.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch


def _parse_scalar(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _load_config(path: str, overrides: List[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for ov in overrides:
        k, _, v = ov.partition("=")
        cfg[k] = _parse_scalar(v)
    return cfg


def _device(cfg) -> torch.device:
    if str(cfg.get("device", "cuda")) == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _trainable_checksum(model) -> str:
    from fl_v3.engine.local_runner import numpy_state_checksum
    from fl_v3.training.tasks import trainable_state_dict

    arrs = [v.detach().cpu().numpy() for v in trainable_state_dict(model).values()]
    return numpy_state_checksum(arrs)


def _subset_loader(run_config, val_info, tokens):
    """A loader over EXACTLY ``tokens`` (subset re-decode for false-disappearance)."""
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.models.fusion.collate import detection_collate_fn

    ds = NuScenesMultimodalDataset(val_info, P.get_dataroot(run_config), sample_tokens=sorted(tokens))
    return make_loader(ds, batch_size=int(run_config.get("batch-size", 16)), shuffle=False,
                       num_workers=int(run_config.get("num-workers", 4)),
                       seed=int(run_config.get("seed", 42)), collate_fn=detection_collate_fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="final_model.pt (self-contained full model)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-gt-sanity", action="store_true")
    ap.add_argument("--viz-samples", type=int, default=4)
    ap.add_argument(
        "--diagnostic", action="store_true",
        help="DIAGNOSTIC scope (e.g. the D14 centralized baseline D): compute ALL the same official "
        "metrics with the SAME evaluator, but DO NOT enforce the D10 FL-provenance gate (a centralized "
        "checkpoint is not D10 by construction). The verdict is stamped verdict_scope='diagnostic' — it "
        "is NOT a benchmark-reference go/no-go, only 'does this checkpoint clear the readiness bar' "
        "(which is exactly the D2 gate for the centralized attack).")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    from fl_v3.data.nuscenes import paths as P, info_cache as IC
    from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import enforce_determinism, seed_everything, truthy
    from fl_v3.eval.detection_eval import (
        VERSION_EVAL_SET, assert_version_split, decode_eval_set, run_detection_eval,
        gt_as_pred_submission, submission_meta,
    )
    from fl_v3.eval import asr as ASR
    from fl_v3.eval.report import t4_clean_cell
    from fl_v3.eval.provenance import verify_d10_provenance

    cfg = _load_config(args.config, args.overrides)
    # The official metric REQUIRES the full eval split (DetectionEval loads the whole split's GT).
    cfg["det-eval-limit"] = 0
    # BATCH-INVARIANCE (T4 finding 2026-06-17): the detector forward is NOT perfectly batch-invariant
    # (cuDNN conv varies with batch composition → boundary detections near τ_clean flip; run-to-run with
    # the SAME batching is bit-identical, but batch-16 vs batch-1 differs on ~half the samples). The ASR
    # disappearance metric must depend ONLY on a target's own trigger, not on batch-mates, so the WHOLE
    # readiness/ASR decode runs at batch_size=1 (canonical per-sample inference) — one consistent,
    # batch-invariant decode feeding DetectionEval + the frozen subset + disappearance + V4. T5 inherits
    # this protocol (decode triggered inputs at batch_size=1 too). See collab/findings_log.md.
    cfg["batch-size"] = 1
    numeric_mode = str(cfg.get("numeric-mode", "fp32"))
    seed_everything(int(cfg.get("seed", 42)))
    # Regime consistency (D14): evaluate a checkpoint in the SAME numeric regime it was trained in
    # (no mixing). The launcher passes numeric-mode=tf32 for the TF32 reference / centralized baselines.
    enforce_determinism(strict=truthy(cfg.get("determinism-strict", True)), numeric_mode=numeric_mode,
                        level=str(cfg.get("determinism-level", "strict")))

    version = str(cfg["nuscenes-version"])
    val_split = str(cfg["nuscenes-val-split"])
    eval_set = VERSION_EVAL_SET[version]
    assert_version_split(version, eval_set)
    scale = "trainval-scientific" if version == "v1.0-trainval" else "mini-smoke"
    os.makedirs(args.output_dir, exist_ok=True)
    device = _device(cfg)
    print(f"[t4-readiness] version={version} eval_set={eval_set} scale={scale} device={device}", flush=True)

    # --- model + checkpoint ---
    task = get_task("nuscenes_detection")
    model = task.build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    checksum = _trainable_checksum(model)
    ckpt_file = os.path.join(os.path.dirname(args.checkpoint), "trainable_checksum.txt")
    if os.path.isfile(ckpt_file):
        recorded = open(ckpt_file).read().strip()
        if recorded != checksum:
            raise RuntimeError(f"checkpoint checksum mismatch: file={recorded} recomputed={checksum}")
    print(f"[t4-readiness] FL_TRAINABLE_CHECKSUM = {checksum}", flush=True)

    # --- regime-consistency guard (det-review #3): a TF32-trained checkpoint MUST be evaluated in
    # TF32 (no mixing). If a provenance.json beside the checkpoint records a numeric-mode, it MUST
    # match the evaluator's numeric_mode — RAISE on mismatch so a silent fp32/tf32 mix cannot pass.
    # Legacy checkpoints (no recorded numeric-mode) only warn (the field predates D13/D14).
    _prov_file = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "provenance.json")
    if os.path.isfile(_prov_file):
        _ckpt_mode = json.load(open(_prov_file, encoding="utf-8")).get("numeric-mode")
        if _ckpt_mode is None:
            print(f"[t4-readiness] WARNING: checkpoint provenance has no numeric-mode (legacy); "
                  f"evaluating in {numeric_mode}", flush=True)
        elif str(_ckpt_mode) != numeric_mode:
            raise RuntimeError(
                f"NUMERIC REGIME MISMATCH (no mixing, D14): checkpoint was trained numeric-mode="
                f"{_ckpt_mode!r} but the evaluator is running numeric-mode={numeric_mode!r}. "
                f"Re-run with numeric-mode={_ckpt_mode}.")
        else:
            print(f"[t4-readiness] regime match OK: checkpoint + evaluator both numeric-mode={numeric_mode}",
                  flush=True)

    # --- D10 provenance gate (§0.2): only a full-participation log-group trainval checkpoint may
    # produce a trainval go/no-go. Mini (scale != trainval-scientific) is NOT a go/no-go → check skipped.
    # --diagnostic (D14 centralized baseline D): a centralized checkpoint is NOT D10 by construction —
    # skip the RAISE, record provenance honestly, and stamp verdict_scope='diagnostic'.
    verdict_scope = "diagnostic" if args.diagnostic else "reference"
    if args.diagnostic:
        provenance = {"_verified": False, "verdict_scope": "diagnostic",
                      "numeric-mode": numeric_mode,
                      "reason": "diagnostic scope — D10 FL-provenance gate intentionally NOT enforced "
                                "(centralized/non-reference checkpoint); metrics computed with the same "
                                "official evaluator but this is NOT a benchmark-reference go/no-go."}
        print(f"[t4-readiness] DIAGNOSTIC scope — D10 provenance gate SKIPPED (numeric-mode={numeric_mode})",
              flush=True)
    elif scale == "trainval-scientific":
        provenance = verify_d10_provenance(args.checkpoint, checksum)
        print(f"[t4-readiness] D10 provenance VERIFIED: {provenance.get('regime', '?')} | "
              f"fraction-train={provenance.get('fraction-train')} "
              f"partition={provenance.get('nuscenes-partition-mode')} defense={provenance.get('defense-type')} "
              f"numeric-mode={provenance.get('numeric-mode')}",
              flush=True)
    else:
        provenance = {"_verified": False,
                      "reason": f"scale={scale} (not trainval-scientific) — D10 provenance check skipped; NOT a go/no-go"}

    # --- single shared decode over the FULL val split ---
    cache_dir = str(cfg["nuscenes-cache-dir"])
    val_info, _ = IC.load_cache(cache_dir, version, val_split)
    all_tokens = sorted(i["sample_token"] for i in val_info)
    eval_loader = task.eval_loader(cfg)
    print(f"[t4-readiness] decoding {len(all_tokens)} {eval_set} samples (single shared decode)…", flush=True)
    decodes = decode_eval_set(model, eval_loader, device, cfg)
    assert {d.sample_token for d in decodes} == set(all_tokens), "decoded token set != full val split"

    # --- 1. official DetectionEval (mAP/NDS/per-class AP/TP errors/car recall) ---
    nusc = _nusc(version)
    det = run_detection_eval(nusc, decodes, eval_set, version,
                             os.path.join(args.output_dir, "det_eval"), DETECTION_NAMES,
                             all_eval_tokens=all_tokens, run_config=cfg, verbose=False)
    print(f"[t4-readiness] OFFICIAL mAP={det['mAP']:.4f} NDS={det['NDS']:.4f} "
          f"car_recall={det['car_recall']:.4f} car_AP@2m={det['car_ap_2m']:.4f}", flush=True)

    # --- 2. GT-as-pred sanity at trainval scale (optional) ---
    gt_sanity = None
    if not args.no_gt_sanity:
        npa = {}
        for d in decodes:
            for at in d.gt_ann_tokens:
                ann = nusc.get("sample_annotation", at)
                npa[at] = int(ann["num_lidar_pts"]) + int(ann["num_radar_pts"])
        sub = gt_as_pred_submission(decodes, DETECTION_NAMES, all_tokens, num_pts_by_ann=npa)
        sdir = os.path.join(args.output_dir, "gt_as_pred"); os.makedirs(sdir, exist_ok=True)
        rp = os.path.join(sdir, "results.json"); json.dump(sub, open(rp, "w"), sort_keys=True)
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.common.config import config_factory
        m, _ = DetectionEval(nusc, config=config_factory("detection_cvpr_2019"), result_path=rp,
                             eval_set=eval_set, output_dir=sdir, verbose=False).evaluate()
        gser = m.serialize()
        gt_sanity = {"car_ap_2m": float(gser["label_aps"]["car"][2.0]),
                     "car_mean_ap": float(gser["mean_dist_aps"]["car"]),
                     "mAP": float(gser["mean_ap"])}
        print(f"[t4-readiness] GT-as-pred sanity: car AP@2m={gt_sanity['car_ap_2m']:.4f} "
              f"(expect ≈1.0 — conversion sound at trainval scale)", flush=True)

    # --- 3. ASR eligibility → frozen subset (bound to the checkpoint checksum) ---
    thr = ASR.asr_thresholds_from_run_config(cfg)
    subset = ASR.build_frozen_asr_subset(decodes, thr, checksum, version=version,
                                         eval_set=eval_set, nusc=nusc)
    json.dump(subset, open(os.path.join(args.output_dir, "frozen_asr_subset.json"), "w"),
              sort_keys=True, indent=2)
    eligible_count = int(subset["n"])
    print(f"[t4-readiness] eligible-clean-detected N={eligible_count} "
          f"(hash={subset['content_hash'][:16]}…) counts={subset['eligibility_counts']}", flush=True)

    # --- 4. false-disappearance baseline (fresh 2nd decode over the subset's samples) ---
    subset_tokens = sorted({s for s, _ in subset["targets"]})
    if subset_tokens:
        loader2 = _subset_loader(cfg, val_info, subset_tokens)
        decodes2 = decode_eval_set(model, loader2, device, cfg)
    else:
        decodes2 = []
    fd = ASR.false_disappearance_baseline(subset, decodes2, thr)
    print(f"[t4-readiness] false-disappearance rate={fd['false_disappearance_rate']} "
          f"defined={fd['defined']} passed={fd['passed']} reason={fd['reason']}", flush=True)

    # --- 5. readiness verdict ---
    car_recall = float(det["car_recall"])
    pass_n = eligible_count >= thr.n_min
    pass_recall = car_recall > thr.recall_floor
    pass_fd = bool(fd["passed"])
    ready = bool(pass_n and pass_recall and pass_fd) and scale == "trainval-scientific"
    gaps = []
    if not pass_n:
        gaps.append(f"eligible_count {eligible_count} < N_min {thr.n_min}")
    if not pass_recall:
        gaps.append(f"clean_car_recall {car_recall:.4f} <= recall_floor {thr.recall_floor}")
    if not pass_fd:
        gaps.append(f"false-disappearance gate failed ({fd['reason'] or fd['false_disappearance_rate']})")
    if scale != "trainval-scientific":
        gaps.append(f"scale={scale} is NOT a go/no-go (mini = engineering smoke)")
    readiness = {
        "verdict": "READY" if ready else "NOT-READY",
        "verdict_scope": verdict_scope,
        "numeric_mode": numeric_mode,
        "scale": scale,
        "version": version,
        "eval_set": eval_set,
        "attacked_checkpoint_FL_TRAINABLE_CHECKSUM": checksum,
        "verified_d10_provenance": provenance,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "mAP": det["mAP"],
        "NDS": det["NDS"],
        "official_clean_car_recall": car_recall,
        "car_ap_2m": det["car_ap_2m"],
        "car_mean_ap": det["car_mean_ap"],
        "eligible_count": eligible_count,
        "pinned_floors": {"N_min": thr.n_min, "recall_floor": thr.recall_floor,
                          "tau_pts": thr.tau_pts, "tau_clean": thr.tau_clean,
                          "d_clean": thr.d_clean, "false_disappear_max": thr.false_disappear_max},
        "false_disappearance": fd,
        "gt_as_pred_sanity": gt_sanity,
        "eligibility_counts": subset["eligibility_counts"],
        "frozen_subset_hash": subset["content_hash"],
        "gaps": gaps,
        "recommended_strengthening": (
            [] if ready else
            ["full participation already applied (D10) — if NOT-READY, escalate to architecture "
             "strengthening: deeper LiDAR PFN / full-model-from-scratch on A100 (D9), re-validate "
             "the A100 determinism gate first"]),
        "n_eval_samples": det["n_eval_samples"],
        "tp_errors": det["tp_errors"],
        "official_recall_at_tp": det["official_recall_at_tp"],
    }
    json.dump(readiness, open(os.path.join(args.output_dir, "benchmark_readiness.json"), "w"),
              sort_keys=True, indent=2)

    # --- 6. frozen 6-tuple clean report cell ---
    report = t4_clean_cell(scale=scale, checkpoint_checksum=checksum,
                           clean_map=det["mAP"], clean_nds=det["NDS"],
                           asr_denominator_n=eligible_count,
                           extra={"official_clean_car_recall": car_recall,
                                  "readiness_verdict": readiness["verdict"]})
    json.dump(report, open(os.path.join(args.output_dir, "clean_cell_report.json"), "w"),
              sort_keys=True, indent=2)

    # --- 7. V4 for a few eligible samples ---
    _render_v4(args, cfg, val_info, subset, decodes, thr)

    print("=" * 70, flush=True)
    print(f"[t4-readiness] VERDICT = {readiness['verdict']}  (scale={scale})", flush=True)
    print(f"  mAP={det['mAP']:.4f} NDS={det['NDS']:.4f} car_recall={car_recall:.4f} "
          f"eligible_N={eligible_count} (N_min={thr.n_min}, recall_floor={thr.recall_floor})", flush=True)
    print(f"  frozen-subset hash = {subset['content_hash']}", flush=True)
    print(f"  attacked-checkpoint checksum = {checksum}", flush=True)
    if gaps:
        print(f"  gaps: {gaps}", flush=True)
    print("=" * 70, flush=True)


def _nusc(version: str):
    from nuscenes import NuScenes
    from fl_v3.data.nuscenes import paths as P
    return NuScenes(version=version, dataroot=P.DATAROOT, verbose=False)


def _render_v4(args, cfg, val_info, subset, decodes, thr) -> None:
    """Render V4 for up to ``--viz-samples`` samples that contain eligible targets (with images)."""
    try:
        from fl_v3.viz.writer import VizWriter
        from fl_v3.viz.detection import render_v4
        from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
        from fl_v3.data.nuscenes import paths as P

        sample_tokens = sorted({s for s, _ in subset["targets"]})[: max(0, args.viz_samples)]
        if not sample_tokens:
            return
        by_token = {d.sample_token: d for d in decodes}
        ds = NuScenesMultimodalDataset(val_info, P.get_dataroot(cfg), sample_tokens=sample_tokens)
        writer = VizWriter(args.output_dir)
        for i in range(len(ds)):
            samp = ds[i]
            tok = samp["sample_token"]
            if tok in by_token:
                render_v4(writer, by_token[tok], thr, image_chw_by_cam=samp["images"].numpy(), cam_index=0)
        writer.write_manifest()
        print(f"[t4-readiness] V4 rendered for {len(sample_tokens)} eligible samples", flush=True)
    except Exception as e:  # V4 is reporting, not a gate — never fail the readiness run on it
        print(f"[t4-readiness] V4 render skipped ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
