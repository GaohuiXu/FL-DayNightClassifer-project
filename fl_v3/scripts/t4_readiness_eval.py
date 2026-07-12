"""DT4-A — the benchmark-readiness eval (fl_v3 T4). Run through the current Arrhenius launcher/config.

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
import hashlib
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


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_s06_eval_model(caller_cfg: dict, checkpoint: str, device):
    """Load only a complete S06 checkpoint and apply its hash-bound raw/EMA policy."""
    from fl_v3.config import resolve_config, verify_physical_data_identities
    from fl_v3.training.checkpoint import CHECKPOINT_SCHEMA, load_checkpoint
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import make_grad_scaler, verify_runtime_dependency_identity

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("T4 refuses legacy/bare checkpoints; a complete S06 checkpoint is required")
    resolved = resolve_config(payload.get("resolved_config", {}))
    strict = resolved.to_run_config()
    allowed_eval_overrides = {"batch-size", "num-workers", "det-eval-limit"}
    drift = [
        key for key in strict
        if key in caller_cfg and key not in allowed_eval_overrides and caller_cfg[key] != strict[key]
    ]
    if drift:
        raise RuntimeError(f"T4 caller/checkpoint resolved-config drift: {sorted(drift)}")
    preserved = {key: caller_cfg[key] for key in allowed_eval_overrides if key in caller_cfg}
    caller_cfg.update(strict); caller_cfg.update(preserved)
    caller_cfg["det-eval-limit"] = 0
    verify_physical_data_identities(resolved)
    runtime_identity = verify_runtime_dependency_identity(strict)
    caller_cfg["runtime-dependencies-sha256"] = hashlib.sha256(
        json.dumps(runtime_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    caller_cfg["checkpoint-sha256"] = _sha256_file(checkpoint)
    caller_cfg["checkpoint-weights"] = strict["evaluation-checkpoint-weights"]

    task = get_task("nuscenes_detection")
    model = task.build_model(strict).to(device)
    opt_spec = resolved.data["optimizer"]
    opt_cls = torch.optim.Adam if opt_spec["name"] == "adam" else torch.optim.AdamW
    optimizer = opt_cls(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(opt_spec["learning_rate"]), weight_decay=float(opt_spec["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(device, resolved.precision)
    ema = None
    decay = resolved.data["training"]["ema_decay"]
    if decay is not None:
        from torch.optim.swa_utils import AveragedModel
        ema = AveragedModel(
            model,
            avg_fn=lambda old, new, _count, d=float(decay): d * old + (1.0 - d) * new,
            use_buffers=False,
        )
    _, identity = load_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler, ema=ema, config=resolved, map_location="cpu",
    )
    if identity != resolved.sha256:
        raise RuntimeError("T4 checkpoint identity does not equal its resolved-config SHA-256")
    if caller_cfg["checkpoint-weights"] == "ema":
        if ema is None:
            raise RuntimeError("T4 EMA policy requested but checkpoint has no EMA")
        model.load_state_dict(ema.module.state_dict(), strict=True)
    model.to(device).eval()
    return model


def _subset_loader(run_config, val_info, tokens):
    """A loader over EXACTLY ``tokens`` (subset re-decode for false-disappearance)."""
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
    from fl_v3.data.nuscenes import paths as P
    from fl_v3.models.fusion.collate import detection_collate_fn

    ds = NuScenesMultimodalDataset(val_info, P.get_dataroot(run_config), sample_tokens=sorted(tokens),
                                   n_sweeps=int(run_config["det-lidar-sweeps"]),
                                   zip_manifest=str(run_config["nuscenes-zip-manifest"]),
                                   model_mode=str(run_config["model-mode"]))
    return make_loader(ds, batch_size=int(run_config.get("batch-size", 16)), shuffle=False,
                       num_workers=int(run_config.get("num-workers", 4)),
                       seed=int(run_config.get("seed", 42)), collate_fn=detection_collate_fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="complete S06 boundary checkpoint")
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

    from fl_v3.data.nuscenes import paths as P
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
    precision = str(cfg.get("precision", "fp16"))
    seed_everything(int(cfg.get("seed", 42)))
    # Regime consistency: evaluate a checkpoint in the SAME precision regime it was trained in
    # (no mixing). Arrhenius sparse default is fp16; the offline dev/debug path is fp32.
    enforce_determinism(strict=truthy(cfg.get("determinism-strict", True)), precision=precision)

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
    model = _load_s06_eval_model(cfg, args.checkpoint, device)
    checksum = _trainable_checksum(model)
    ckpt_file = os.path.join(os.path.dirname(args.checkpoint), "trainable_checksum.txt")
    if os.path.isfile(ckpt_file):
        recorded = open(ckpt_file).read().strip()
        if recorded != checksum:
            raise RuntimeError(f"checkpoint checksum mismatch: file={recorded} recomputed={checksum}")
    print(f"[t4-readiness] FL_TRAINABLE_CHECKSUM = {checksum}", flush=True)

    # --- regime-consistency guard (det-review #3; D16): a checkpoint MUST be evaluated in the SAME
    # precision regime it was trained in (no mixing). If a provenance.json beside the checkpoint records
    # a ``precision``, it MUST match the evaluator's — RAISE on mismatch so a silent fp16/fp32 mix cannot
    # pass. Checkpoints that predate the D16 knob carry only a legacy ``numeric-mode`` (fp32/tf32) and no
    # ``precision`` → only WARN (their regime is not fp16-vs-fp32 comparable).
    _prov_file = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "provenance.json")
    if os.path.isfile(_prov_file):
        _prov = json.load(open(_prov_file, encoding="utf-8"))
        _ckpt_prec = _prov.get("precision")
        if _ckpt_prec is None:
            _legacy = _prov.get("numeric-mode")
            print(f"[t4-readiness] WARNING: checkpoint provenance has no precision (pre-D16; "
                  f"numeric-mode={_legacy!r}); evaluating in precision={precision}", flush=True)
        elif str(_ckpt_prec) != precision:
            raise RuntimeError(
                f"PRECISION REGIME MISMATCH (no mixing, D16): checkpoint was trained precision="
                f"{_ckpt_prec!r} but the evaluator is running precision={precision!r}. "
                f"Re-run with precision={_ckpt_prec}.")
        else:
            print(f"[t4-readiness] regime match OK: checkpoint + evaluator both precision={precision}",
                  flush=True)

    # --- D10 provenance gate (§0.2): only a full-participation log-group trainval checkpoint may
    # produce a trainval go/no-go. Mini (scale != trainval-scientific) is NOT a go/no-go → check skipped.
    # --diagnostic (D14 centralized baseline D): a centralized checkpoint is NOT D10 by construction —
    # skip the RAISE, record provenance honestly, and stamp verdict_scope='diagnostic'.
    verdict_scope = "diagnostic" if args.diagnostic else "reference"
    if args.diagnostic:
        provenance = {"_verified": False, "verdict_scope": "diagnostic",
                      "precision": precision,
                      "reason": "diagnostic scope — D10 FL-provenance gate intentionally NOT enforced "
                                "(centralized/non-reference checkpoint); metrics computed with the same "
                                "official evaluator but this is NOT a benchmark-reference go/no-go."}
        print(f"[t4-readiness] DIAGNOSTIC scope — D10 provenance gate SKIPPED (precision={precision})",
              flush=True)
    elif scale == "trainval-scientific":
        provenance = verify_d10_provenance(args.checkpoint, checksum)
        print(f"[t4-readiness] D10 provenance VERIFIED: {provenance.get('regime', '?')} | "
              f"fraction-train={provenance.get('fraction-train')} "
              f"partition={provenance.get('nuscenes-partition-mode')} defense={provenance.get('defense-type')} "
              f"precision={provenance.get('precision')}",
              flush=True)
    else:
        provenance = {"_verified": False,
                      "reason": f"scale={scale} (not trainval-scientific) — D10 provenance check skipped; NOT a go/no-go"}

    # --- single shared decode over the FULL val split ---
    val_info, _ = task._load_info(cfg, val_split)
    all_tokens = sorted(i["sample_token"] for i in val_info)
    eval_loader = task.eval_loader(cfg)
    print(f"[t4-readiness] decoding {len(all_tokens)} {eval_set} samples (single shared decode)…", flush=True)
    decodes = decode_eval_set(model, eval_loader, device, cfg)
    assert {d.sample_token for d in decodes} == set(all_tokens), "decoded token set != full val split"

    # --- 1. official DetectionEval (mAP/NDS/per-class AP/TP errors/car recall) ---
    nusc = _nusc(version, cfg)
    det = run_detection_eval(nusc, decodes, eval_set, version,
                             os.path.join(args.output_dir, "det_eval"), DETECTION_NAMES,
                             all_eval_tokens=all_tokens, run_config=cfg, verbose=False)
    print(f"[t4-readiness] OFFICIAL mAP={det['mAP']:.4f} NDS={det['NDS']:.4f} "
          f"car_recall={det['car_recall']:.4f} car_AP@2m={det['car_ap_2m']:.4f}", flush=True)
    # Per-class AP table (mean over the 4 center-distance thresholds) — surfaces WHERE the 10-class mean
    # leaks (the rare/small classes), so the lever priority (CBGS/aug/finer-voxel) can be read directly.
    _pcap = det.get("per_class_mean_ap", {})
    print("[t4-readiness] per-class AP: " +
          " ".join(f"{c}={_pcap.get(c, 0.0):.3f}" for c in DETECTION_NAMES), flush=True)

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
        "precision": precision,
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
        "per_class_mean_ap": det.get("per_class_mean_ap"),
        "label_aps": det.get("label_aps"),
        "decode_score_threshold": float(cfg.get("det-score-threshold", 0.1)),
        "decode_budget_contract": "reviewed_centerhead_per_class_k500_task_nms_post83_cap500",
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
             "strengthening: sparse voxel LiDAR capacity / full-model-from-scratch trainval validation "
             "on Arrhenius once full nuScenes is available"]),
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


def _nusc(version: str, cfg: dict):
    from fl_v3.data.nuscenes import paths as P
    return P.create_nuscenes(version, P.get_dataroot(cfg), verbose=False)


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
        ds = NuScenesMultimodalDataset(val_info, P.get_dataroot(cfg), sample_tokens=sample_tokens,
                                       n_sweeps=int(cfg["det-lidar-sweeps"]),
                                       zip_manifest=str(cfg["nuscenes-zip-manifest"]),
                                       model_mode=str(cfg["model-mode"]))
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
