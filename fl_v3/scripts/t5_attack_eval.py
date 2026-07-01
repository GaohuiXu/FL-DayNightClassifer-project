"""T5 attack eval driver (fl_v3 T5) — the GATE measurements on the frozen subset at batch_size=1.

Multi-task (``--task``), all bound to the **literal** frozen subset ``2ad8f8da…`` + clean checkpoint
``a80466c3…`` (re-verified at load, §0.C6) and the provenance-verified trainval poisoned checkpoint
(§0.C8):

  * ``shard``       — per-target 5-condition ablation + occlusion (the fan-out worker; ``--shard i
                      --num-shards K``); writes ``ablation/ablation_shard_{i}.json``.
  * ``aggregate``   — combine the shards → the 5-condition table + the floor-corrected fusion-aware
                      verdict + the headline disappear-ASR (cond-4) + the occlusion-control ASR.
  * ``stealth``     — the POISONED model's clean DetectionEval (mAP/NDS + official car recall ≥ floor).
  * ``guards``      — cond-5a LiDAR-invariance + the camera-only clean-recall precondition.
  * ``viz``         — V5 + V3(trigger) for a few subset samples.
  * ``null-verify`` — recompute a null checkpoint's full-state sha256, diff vs the pinned ``0fe444e3…``.

``batch_size=1`` is forced (the T4 batch-invariance protocol); the ASR scores on the UNEDITED val GT
with the subset's BOUND thresholds (§0.C7/§0.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# config / model / subset plumbing
# ---------------------------------------------------------------------------
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
    cfg["batch-size"] = 1  # T4 batch-invariance protocol (the whole ASR/ablation decode) — HARD-set
    cfg["det-eval-limit"] = 0
    assert int(cfg["batch-size"]) == 1, "the T5 ASR/ablation decode MUST run at batch_size=1 (T4 §5c)"
    return cfg


# The §0.C anti-gaming constants are HARD: any override (a config tweak after seeing the result) is a
# post-hoc-fitting attempt → the driver REFUSES to emit a verdict. δ_fusion is the load-bearing one.
_PINNED = {
    "attack-delta-fusion": 0.2, "attack-delta-fusion-mult": 2.0, "attack-viability-asr": 0.3,
    "attack-stealth-recall-floor": 0.75, "asr-false-disappear-max": 0.02,
    "attack-trigger-budget-frac": 0.30, "attack-delta-clean": 0.10,
    "attack-cond5a-recall-floor": 0.3,
}


def _assert_pinned_constants(cfg: dict) -> None:
    """RAISE if any §0.C anti-gaming constant deviates from its pinned value (no post-hoc fitting)."""
    bad = []
    for k, want in _PINNED.items():
        got = float(cfg.get(k, want))
        if abs(got - want) > 1e-12:
            bad.append(f"{k}={got} != PINNED {want}")
    if bad:
        raise RuntimeError("PINNED anti-gaming constant(s) overridden (§0.C4 — hard FAIL, no fitting to "
                           "the result): " + "; ".join(bad))


def _pinned_constants(cfg: dict) -> dict:
    return {k: float(cfg.get(k, v)) for k, v in _PINNED.items()}


def _load_json(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _device(cfg) -> torch.device:
    if str(cfg.get("device", "cuda")) == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _full_state_sha256(state_dict) -> str:
    """Content sha256 over the FULL state-dict tensor bytes (sorted keys) — the §0.C5 null target."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        a = np.ascontiguousarray(state_dict[k].detach().cpu().numpy())
        h.update(k.encode()); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _trainable_checksum(model) -> str:
    from fl_v3.engine.local_runner import numpy_state_checksum
    from fl_v3.training.tasks import trainable_state_dict
    return numpy_state_checksum([v.detach().cpu().numpy() for v in trainable_state_dict(model).values()])


def _load_model(cfg, checkpoint: str, device):
    from fl_v3.training.tasks import get_task
    task = get_task("nuscenes_detection")
    model = task.build_model(cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_subset(cfg, subset_path: str) -> dict:
    """Load the frozen subset; re-verify the content-hash + the LITERAL pinned identities (§0.C6)."""
    from fl_v3.eval.asr import verify_frozen_asr_subset
    with open(subset_path, encoding="utf-8") as f:
        subset = json.load(f)
    verify_frozen_asr_subset(subset)  # recompute the content hash (targets/thresholds/checkpoint)
    pin_hash = str(cfg.get("attack-frozen-subset-hash", ""))
    pin_ckpt = str(cfg.get("attack-clean-checkpoint-checksum", ""))
    if pin_hash and subset["content_hash"] != pin_hash:
        raise RuntimeError(f"frozen-subset content_hash {subset['content_hash']} != PINNED {pin_hash} (§0.C6)")
    if pin_ckpt and subset["checkpoint_checksum"] != pin_ckpt:
        raise RuntimeError(f"subset checkpoint_checksum {subset['checkpoint_checksum']} != PINNED {pin_ckpt} (§0.C6)")
    return subset


def _val_info(cfg):
    from fl_v3.data.nuscenes import info_cache as IC
    return IC.load_cache(str(cfg["nuscenes-cache-dir"]), str(cfg["nuscenes-version"]),
                         str(cfg["nuscenes-val-split"]))[0]


def _val_dataset(cfg, val_info, tokens):
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
    from fl_v3.data.nuscenes import paths as P
    return NuScenesMultimodalDataset(val_info, P.get_dataroot(cfg), sample_tokens=sorted(set(tokens)))


def _seed(cfg):
    from fl_v3.utils.runtime import enforce_determinism, seed_everything, truthy
    seed_everything(int(cfg.get("seed", 42)))
    enforce_determinism(
        strict=truthy(cfg.get("determinism-strict", True)),
        precision=str(cfg.get("precision", "fp16")),
    )


# ---------------------------------------------------------------------------
# task: shard (the per-target fan-out worker)
# ---------------------------------------------------------------------------
def task_shard(args, cfg):
    from fl_v3.eval.asr import thresholds_from_subset
    from fl_v3.attacks import trigger as TR
    from fl_v3.attacks import fusion_ablation as FA
    device = _device(cfg); _seed(cfg)
    subset = _load_subset(cfg, args.subset)
    thr = thresholds_from_subset(subset)
    spec = TR.trigger_spec_from_run_config(cfg)
    max_objects = int(cfg.get("det-max-objects", 200))

    targets: List[Tuple[str, str]] = [tuple(t) for t in subset["targets"]]
    shard = targets[args.shard::args.num_shards]                      # deterministic round-robin slice
    by_sample: Dict[str, List[str]] = {}
    for s, a in shard:
        by_sample.setdefault(s, []).append(a)

    val_info = _val_info(cfg)
    ds = _val_dataset(cfg, val_info, list(by_sample.keys()))
    poisoned = _load_model(cfg, args.checkpoint, device)
    clean = _load_model(cfg, args.clean_checkpoint, device) if args.clean_checkpoint else None
    print(f"[t5-shard {args.shard}/{args.num_shards}] {len(shard)} targets over {len(by_sample)} samples", flush=True)

    out: List[dict] = []
    for i in range(len(ds)):
        sample = ds[i]
        tok = sample["sample_token"]
        if tok not in by_sample:
            continue
        clean_anns = FA.clean_detected_anns(poisoned, sample, device, thr, max_objects)
        for ann in by_sample[tok]:
            if args.cond4_only:   # lean CONTROL measurement: cond-1 + cond-4 only (no cond-2/3/5a/occlusion)
                v = FA.evaluate_target_cond4(sample, ann, poisoned, device, thr, max_objects, spec,
                                             clean_anns=clean_anns)
            else:
                v = FA.evaluate_target(sample, ann, poisoned, clean, device, thr, max_objects, spec,
                                       clean_anns=clean_anns)
            if v is None:
                out.append({"sample_token": tok, "ann_token": ann, "evaluated": False})
            else:
                out.append({"sample_token": tok, "ann_token": ann, "evaluated": True,
                            "disappeared": v.disappeared, "occlusion_disappeared": v.occlusion_disappeared,
                            "placement_aligned_ok": v.placement_aligned_ok,
                            "placement_nonaligned_iou0": v.placement_nonaligned_iou0,
                            "area_ratio": v.area_ratio})
        if (i + 1) % 50 == 0:
            print(f"[t5-shard {args.shard}] {i+1}/{len(ds)} samples", flush=True)

    os.makedirs(os.path.join(args.output_dir, "ablation"), exist_ok=True)
    p = os.path.join(args.output_dir, "ablation", f"ablation_shard_{args.shard}_of_{args.num_shards}.json")
    json.dump({"shard": args.shard, "num_shards": args.num_shards, "n_targets": len(shard),
               "results": out}, open(p, "w"), sort_keys=True)
    print(f"[t5-shard {args.shard}] wrote {p} ({len(out)} targets)", flush=True)


# ---------------------------------------------------------------------------
# task: aggregate (combine shards → table + verdict)
# ---------------------------------------------------------------------------
def task_aggregate(args, cfg):
    from fl_v3.attacks import fusion_ablation as FA
    from fl_v3.eval.provenance import verify_attack_provenance
    _assert_pinned_constants(cfg)  # §0.C4: NO post-hoc override of any anti-gaming constant
    device = _device(cfg)
    subset = _load_subset(cfg, args.subset)
    N = int(subset["n"])
    # §0.C8: the verdict is bound to a provenance-verified trainval poisoned checkpoint
    model = _load_model(cfg, args.checkpoint, device)
    checksum = _trainable_checksum(model)
    prov = verify_attack_provenance(args.checkpoint, checksum)
    print(f"[t5-agg] attack provenance VERIFIED: {prov.get('regime')} | mode={prov.get('attack-mode')} "
          f"poison_rate={prov.get('attack-poison-rate')} m_r={prov.get('m_r')} roster={prov.get('roster')}", flush=True)

    adir = os.path.join(args.output_dir, "ablation")
    files = sorted(f for f in os.listdir(adir) if f.startswith("ablation_shard_") and f.endswith(".json"))
    rows: List[dict] = []
    for f in files:
        rows.extend(json.load(open(os.path.join(adir, f)))["results"])
    seen = {(r["sample_token"], r["ann_token"]) for r in rows}
    evaluated = [r for r in rows if r.get("evaluated")]
    coverage = len(seen) / N if N else 0.0

    # per-condition raw ASR over the FULL N (un-evaluated targets → not disappeared, conservative)
    counts = {c: 0 for c in FA.CONDITIONS}
    occ = 0; aligned_ok = 0; nonaligned_iou0 = 0; area_ok = 0; n_area = 0
    budget = float(cfg.get("attack-trigger-budget-frac", 0.30))
    for r in evaluated:
        for c in FA.CONDITIONS:
            if r["disappeared"].get(c):
                counts[c] += 1
        if r.get("occlusion_disappeared"):
            occ += 1
        aligned_ok += int(bool(r.get("placement_aligned_ok")))
        nonaligned_iou0 += int(bool(r.get("placement_nonaligned_iou0")))
        if r.get("area_ratio") is not None:
            n_area += 1; area_ok += int(r["area_ratio"] <= budget + 1e-9)
    asr = {c: counts[c] / N for c in FA.CONDITIONS}
    occ_asr = occ / N

    # --- the OTHER anti-gaming gates (from the sibling eval tasks) — MUST be present + pass before
    # a FUSION-AWARE GATE can go green (the verdict is a CONJUNCTION, not just the ablation margins).
    guards = _load_json(os.path.join(args.output_dir, "cond5a_guards.json"))
    stealth = _load_json(os.path.join(args.output_dir, "stealth.json"))
    # RE-DERIVE the sub-gates from the RAW metrics + the PINNED floors (never trust a sub-task's stored
    # boolean, which could have been computed against an overridden floor).
    stealth_floor = float(cfg.get("attack-stealth-recall-floor", 0.75))
    cond5a_floor = float(cfg.get("attack-cond5a-recall-floor", 0.3))
    stealth_ok = (float(stealth["poisoned_clean_car_recall"]) >= stealth_floor) if stealth else None
    cond5a_valid = (bool(guards["lidar_invariant_all"]) and
                    float(guards["camera_only_clean_recall"]) >= cond5a_floor) if guards else None

    verdict = FA.fusion_aware_verdict(
        asr, delta_fusion=float(cfg.get("attack-delta-fusion", 0.2)),
        mult=float(cfg.get("attack-delta-fusion-mult", 2.0)),
        viability=float(cfg.get("attack-viability-asr", 0.3)),
        cond5a_guards_valid=cond5a_valid)

    fd_max = float(cfg.get("asr-false-disappear-max", 0.02))
    # placement objective test (§0.C): every evaluated target's cond-4 centre ≤20px + cond-2 IoU==0 +
    # area ≤ budget. Require ≥0.99 (tolerate <1% truncated-geometry edge cases; the count is surfaced).
    aligned_frac = (aligned_ok / len(evaluated)) if evaluated else 0.0
    nonaligned_frac = (nonaligned_iou0 / len(evaluated)) if evaluated else 0.0
    area_frac = (area_ok / n_area) if n_area else 0.0
    placement_ok = bool(aligned_frac >= 0.99 and nonaligned_frac >= 0.99 and area_frac >= 0.999)

    headline = {"disappear_asr_raw_cond4": asr["cond4_aligned"],
                "disappear_asr_floor_corrected_cond4": verdict["cond4_corrected"],
                "viable": verdict["viable_cond4_ge_0.3"]}
    occlusion = {"occlusion_control_asr": occ_asr, "false_disappear_max": fd_max,
                 "passes_not_occlusion": bool(occ_asr < fd_max)}
    placement = {"n_evaluated": len(evaluated), "coverage_of_N": coverage,
                 "aligned_center_le_20px_frac": aligned_frac, "nonaligned_iou0_frac": nonaligned_frac,
                 "area_le_budget_frac": area_frac, "budget_frac": budget, "placement_ok": placement_ok}

    # the conjunctive GATE (every sub-gate must be PRESENT and PASS; a missing sibling → not green)
    gate = {
        "viable_cond4_ge_0.3": verdict["viable_cond4_ge_0.3"],
        "margin_ok_ge_delta_fusion": verdict["margin_ok_ge_delta_fusion"],
        "mult_ok_cond4_ge_2x": verdict["mult_ok_cond4_ge_2x_maxother"],
        "not_occlusion": occlusion["passes_not_occlusion"],
        "stealth_ok": stealth_ok,
        "placement_objective_ok": placement_ok,
        "cond5a_guards_valid": cond5a_valid,
        "provenance_verified": True,
    }
    missing = [k for k, v in gate.items() if v is None]
    gate_pass = bool(all(v is True for v in gate.values()) and not missing)
    overall = ("FUSION-AWARE (GATE GREEN)" if (gate_pass and verdict["fusion_aware"]) else
               ("D3-ESCAPE-HATCH (cond4≈cond2 — point-decoration)" if verdict["degenerate_cond4_approx_cond2"] else
                ("INCOMPLETE — missing sub-gates: " + ", ".join(missing) if missing else "GATE NOT GREEN")))

    result = {"N": N, "n_shards": len(files), "raw_asr": asr, "fusion_aware_verdict": verdict,
              "headline": headline, "occlusion_control": occlusion, "placement_objective_test": placement,
              "stealth": stealth, "cond5a_guards": guards, "gate": gate, "gate_pass": gate_pass,
              "missing_subgates": missing, "overall_verdict": overall,
              "pinned_constants": _pinned_constants(cfg),
              "checkpoint_checksum": checksum, "verified_attack_provenance": prov}
    json.dump(result, open(os.path.join(args.output_dir, "fusion_ablation.json"), "w"), sort_keys=True, indent=2)
    print("=" * 72, flush=True)
    print(f"[t5-agg] 5-CONDITION ABLATION (N={N}, coverage={coverage:.4f}, evaluated={len(evaluated)})", flush=True)
    for c in FA.CONDITIONS:
        print(f"  {c:24s} raw_asr={asr[c]:.4f}  floor-corr={verdict['corrected_asr'][c]:+.4f}", flush=True)
    print(f"  floor (cond1) = {verdict['floor']:.4f}", flush=True)
    print(f"  HEADLINE disappear-ASR(cond4, floor-corr) = {verdict['cond4_corrected']:.4f} (viable: {headline['viable']})", flush=True)
    print(f"  occlusion-control ASR (clean model) = {occ_asr:.4f} (<{fd_max}: {occlusion['passes_not_occlusion']})", flush=True)
    print(f"  stealth_ok={stealth_ok}  cond5a_valid={cond5a_valid}  placement_ok={placement_ok}", flush=True)
    print(f"  GATE (conjunction) = {gate}", flush=True)
    print(f"  gate_pass={gate_pass}  OVERALL = {overall}", flush=True)
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# task: stealth (poisoned clean DetectionEval — mAP/NDS + official car recall)
# ---------------------------------------------------------------------------
def task_stealth(args, cfg):
    _assert_pinned_constants(cfg)
    from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
    from fl_v3.eval.detection_eval import decode_eval_set, run_detection_eval, VERSION_EVAL_SET
    from fl_v3.eval.provenance import verify_attack_provenance
    from fl_v3.training.tasks import get_task
    device = _device(cfg); _seed(cfg)
    version = str(cfg["nuscenes-version"]); eval_set = VERSION_EVAL_SET[version]
    model = _load_model(cfg, args.checkpoint, device)
    checksum = _trainable_checksum(model)
    prov = verify_attack_provenance(args.checkpoint, checksum)
    task = get_task("nuscenes_detection")
    val_info = _val_info(cfg)
    all_tokens = sorted(i["sample_token"] for i in val_info)
    loader = task.eval_loader(cfg)
    print(f"[t5-stealth] decoding {len(all_tokens)} {eval_set} samples (poisoned, clean inputs, bs=1)…", flush=True)
    decodes = decode_eval_set(model, loader, device, cfg)
    nusc = _nusc(version)
    det = run_detection_eval(nusc, decodes, eval_set, version, os.path.join(args.output_dir, "stealth_det_eval"),
                             DETECTION_NAMES, all_eval_tokens=all_tokens, run_config=cfg, verbose=False)
    floor = float(cfg.get("attack-stealth-recall-floor", 0.75))
    out = {"poisoned_clean_car_recall": det["car_recall"], "stealth_recall_floor": floor,
           "stealth_ok": bool(det["car_recall"] >= floor), "poisoned_mAP": det["mAP"], "poisoned_NDS": det["NDS"],
           "car_ap_2m": det["car_ap_2m"], "checkpoint_checksum": checksum, "verified_attack_provenance": prov}
    json.dump(out, open(os.path.join(args.output_dir, "stealth.json"), "w"), sort_keys=True, indent=2)
    print(f"[t5-stealth] poisoned clean car_recall={det['car_recall']:.4f} (≥{floor}: {out['stealth_ok']}) "
          f"mAP={det['mAP']:.4f} NDS={det['NDS']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# task: guards (cond-5a LiDAR-invariance + camera-only clean-recall precondition)
# ---------------------------------------------------------------------------
def task_guards(args, cfg):
    _assert_pinned_constants(cfg)
    from fl_v3.eval.asr import thresholds_from_subset
    from fl_v3.attacks import fusion_ablation as FA
    device = _device(cfg); _seed(cfg)
    subset = _load_subset(cfg, args.subset)
    thr = thresholds_from_subset(subset)
    max_objects = int(cfg.get("det-max-objects", 200))
    model = _load_model(cfg, args.checkpoint, device)
    targets = [tuple(t) for t in subset["targets"]]
    sample_tokens = sorted({s for s, _ in targets})[: args.guard_samples]
    by_sample: Dict[str, List[str]] = {}
    for s, a in targets:
        if s in sample_tokens:
            by_sample.setdefault(s, []).append(a)
    val_info = _val_info(cfg)
    ds = _val_dataset(cfg, val_info, sample_tokens)
    inv_results = []; recall_pairs = []
    for i in range(len(ds)):
        sample = ds[i]
        if sample["sample_token"] not in by_sample:
            continue
        inv_results.append(FA.lidar_invariance_check(model, sample, device))
        recall_pairs.append((sample, by_sample[sample["sample_token"]]))
    rec = FA.camera_only_clean_recall(model, recall_pairs, device, thr, max_objects)
    all_invariant = all(r["lidar_invariant"] for r in inv_results)
    max_diff = max((r["max_abs_head_diff"] for r in inv_results), default=0.0)
    floor = float(cfg.get("attack-cond5a-recall-floor", 0.3))  # camera-only readout is OOD vs the 0.85 fused model
    out = {"lidar_invariant_all": bool(all_invariant), "max_abs_head_diff": max_diff,
           "n_invariance_checks": len(inv_results), "camera_only_clean_recall": rec["camera_only_clean_recall"],
           "camera_only_recall_floor": floor, "clean_recall_precondition_ok": bool(rec["camera_only_clean_recall"] >= floor),
           "detail": rec, "cond5a_valid": bool(all_invariant and rec["camera_only_clean_recall"] >= floor)}
    json.dump(out, open(os.path.join(args.output_dir, "cond5a_guards.json"), "w"), sort_keys=True, indent=2)
    print(f"[t5-guards] LiDAR-invariant={all_invariant} (max|Δ|={max_diff}) "
          f"camera_only_clean_recall={rec['camera_only_clean_recall']:.4f} (≥{floor}) → cond5a_valid={out['cond5a_valid']}", flush=True)


# ---------------------------------------------------------------------------
# task: viz (V5 + V3(trigger) for a few subset samples)
# ---------------------------------------------------------------------------
def task_viz(args, cfg):
    from fl_v3.eval.asr import thresholds_from_subset
    from fl_v3.attacks import trigger as TR
    from fl_v3.attacks.poison import _lidar_xyz
    from fl_v3.viz.writer import VizWriter
    from fl_v3.viz import attack as V5
    from fl_v3.viz.fusion import render_v3_trigger
    from fl_v3.models.fusion.bev_grid import BEVConfig
    device = _device(cfg); _seed(cfg)
    subset = _load_subset(cfg, args.subset)
    thr = thresholds_from_subset(subset)
    spec = TR.trigger_spec_from_run_config(cfg)
    model = _load_model(cfg, args.checkpoint, device)
    clean = _load_model(cfg, args.clean_checkpoint, device) if args.clean_checkpoint else None
    cfg_bev = BEVConfig(bev_voxel=(float(cfg.get("det-bev-voxel", 0.4)),) * 2,
                        out_size_factor=int(cfg.get("det-out-size-factor", 2)))
    targets = [tuple(t) for t in subset["targets"]]
    first_by_sample = {}
    for s, a in targets:
        first_by_sample.setdefault(s, a)
    sample_tokens = sorted(first_by_sample)[: args.viz_samples]
    val_info = _val_info(cfg)
    ds = _val_dataset(cfg, val_info, sample_tokens)
    writer = VizWriter(args.output_dir)
    for i in range(len(ds)):
        sample = ds[i]
        tok = sample["sample_token"]
        if tok not in first_by_sample:
            continue
        from fl_v3.attacks.fusion_ablation import target_index
        m = target_index(sample, first_by_sample[tok])
        if m is None:
            continue
        box7 = sample["gt_boxes"].cpu().numpy()[m]
        pl = TR.compute_aligned_placement(box7, _lidar_xyz(sample), sample["lidar2img"].cpu().numpy(), spec)
        if pl is None:
            continue
        V5.render_v5_image(writer, sample, pl, spec, target_box=box7, name=tok)
        try:
            V5.render_v5_fusion_diff(writer, model, sample, pl, spec, cfg_bev, target_box=box7, clean_model=clean, name=tok)
            render_v3_trigger(writer, model, sample, pl, spec, cfg_bev, target_box=box7, name=tok)
        except Exception as e:
            print(f"[t5-viz] fusion-diff skipped for {tok}: {type(e).__name__}: {e}", flush=True)
    writer.write_manifest()
    print(f"[t5-viz] rendered V5/V3 for up to {len(sample_tokens)} subset samples", flush=True)


# ---------------------------------------------------------------------------
# task: null-verify (the §0.C5 byte-identical-null check)
# ---------------------------------------------------------------------------
def task_null_verify(args, cfg):
    device = torch.device("cpu")
    state = torch.load(args.checkpoint, map_location=device)
    full = _full_state_sha256(state)
    pin = str(cfg.get("attack-null-fullstate-sha256", ""))
    ok = bool(pin and full == pin)
    out = {"null_checkpoint": os.path.abspath(args.checkpoint), "full_state_sha256": full,
           "pinned_clean_full_state_sha256": pin, "byte_identical": ok,
           "n_tensors": len(state)}
    json.dump(out, open(os.path.join(args.output_dir, "null_byte_identity.json"), "w"), sort_keys=True, indent=2)
    print(f"[t5-null] null full-state sha256 = {full}", flush=True)
    print(f"[t5-null] pinned clean sha256     = {pin}", flush=True)
    print(f"[t5-null] BYTE-IDENTICAL = {ok}", flush=True)


def _nusc(version: str):
    from nuscenes import NuScenes
    from fl_v3.data.nuscenes import paths as P
    return NuScenes(version=version, dataroot=P.DATAROOT, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["shard", "aggregate", "stealth", "guards", "viz", "null-verify"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", help="the poisoned (or null) final_model.pt")
    ap.add_argument("--clean-checkpoint", default=None, help="the clean a80466c3 checkpoint (occlusion/viz)")
    ap.add_argument("--subset", help="the frozen ASR subset json (2ad8f8da…)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--cond4-only", action="store_true",
                    help="lean CONTROL measurement: cond-1 + cond-4 disappear-ASR only (no cond-2/3/5a/occlusion)")
    ap.add_argument("--guard-samples", type=int, default=40)
    ap.add_argument("--viz-samples", type=int, default=6)
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()
    cfg = _load_config(args.config, args.overrides)
    os.makedirs(args.output_dir, exist_ok=True)
    {"shard": task_shard, "aggregate": task_aggregate, "stealth": task_stealth,
     "guards": task_guards, "viz": task_viz, "null-verify": task_null_verify}[args.task](args, cfg)


if __name__ == "__main__":
    main()
