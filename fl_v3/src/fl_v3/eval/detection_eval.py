"""Official nuScenes ``DetectionEval`` (mAP/NDS) on the platform's decoded boxes (fl_v3 T4).

The utility metric every clean/poisoned cell is reported in. Wires
``nuscenes.eval.detection.evaluate.DetectionEval`` (``config_factory("detection_cvpr_2019")``)
onto the converted global boxes (:mod:`box_to_global`). Invocation details (verified
against devkit 1.1.11):

  * Call **``DetectionEval(...).evaluate()``** directly — NOT ``main()`` (``main()`` does
    ``random.seed(42)`` + ``random.shuffle`` for example plots and writes PNG/PDF; we want
    no RNG, no plots). ``.evaluate()`` itself does NOT touch RNG and writes NO files.
  * The results JSON must have **every eval-split ``sample_token`` as a key** (empty ``[]``
    if no detections) — ``DetectionEval.__init__`` asserts
    ``set(pred.sample_tokens) == set(gt.sample_tokens)``.
  * Boxes per sample are emitted in a **content-defined deterministic order** (see
    ``box_to_global.detection_box_sort_key``) because devkit ``accumulate`` sorts by
    ``(score, emission-index)`` — equal-score ties break on emission order, so a stable
    score-sort alone is insufficient for permutation-invariant mAP.
  * ``eval_set`` is hard-coupled to the NuScenes version (mini → ``mini_val``, trainval →
    ``val``); :func:`assert_version_split` checks it up front.

The model is run over the eval set EXACTLY ONCE (:func:`decode_eval_set`) and the cached
per-sample decode feeds the evaluator, the ASR harness, AND V4 — the plan's "evaluator +
visualization consume the SAME decoded boxes" (no divergent re-decode, no V4 re-threshold).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from fl_v3.eval.box_to_global import decoded_sample_to_boxes


# version ↔ eval_set table (devkit hard-couples these; mismatch → AssertionError in load_gt).
VERSION_EVAL_SET = {
    "v1.0-mini": "mini_val",
    "v1.0-trainval": "val",
    "v1.0-test": "test",
}


def assert_version_split(version: str, eval_set: str) -> None:
    want = VERSION_EVAL_SET.get(version)
    if want is None:
        raise ValueError(f"unknown nuScenes version {version!r}")
    if eval_set != want:
        raise AssertionError(
            f"eval_set {eval_set!r} does not match version {version!r} (expected {want!r}); "
            "the devkit load_gt is hard-coupled to the version."
        )


def submission_meta(model_mode: str = "fusion") -> Dict[str, bool]:
    """Required devkit modality flags for the topology actually executed."""
    if model_mode not in {"camera_only", "lidar_only", "fusion"}:
        raise ValueError(f"unknown model mode {model_mode!r}")
    return {
        "use_camera": model_mode in {"camera_only", "fusion"},
        "use_lidar": model_mode in {"lidar_only", "fusion"},
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }


@dataclass
class SampleDecode:
    """One eval sample's single-decode record — the shared input for eval + ASR + V4."""

    sample_token: str
    boxes: np.ndarray            # [K,7] canonical LIDAR_TOP (cx,cy,cz,l,w,h,yaw)
    scores: np.ndarray           # [K]
    labels: np.ndarray           # [K] detection-id
    velocity: np.ndarray         # [K,2] lidar-frame
    lidar2ego: np.ndarray        # [4,4]
    ego2global_lidar: np.ndarray  # [4,4]
    lidar2img: np.ndarray        # [6,4,4]
    gt_boxes: np.ndarray         # [M,7]
    gt_labels: np.ndarray        # [M]
    gt_num_lidar_pts: np.ndarray  # [M]
    gt_in_range: np.ndarray      # [M] bool
    gt_velocity: np.ndarray      # [M,2] lidar-frame
    gt_ann_tokens: List[str] = field(default_factory=list)
    gt_names: List[str] = field(default_factory=list)
    gt_attribute: List[str] = field(default_factory=list)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@torch.no_grad()
def decode_eval_set(model, eval_loader, device, run_config, timing: Optional[dict] = None) -> List[SampleDecode]:
    """Run the model over the eval loader ONCE, returning per-sample decode records.

    Uses ``model.decode`` (the single decode) at the production ``det-score-threshold`` and
    ``det-max-objects`` — no re-decode, no second threshold. Deterministic order (the loader
    is ``shuffle=False`` over token-sorted samples).
    """
    thr = float(run_config.get("det-score-threshold", 0.1))
    model.eval()
    from fl_v3.training.loop import _float_tensors, _move_to_device
    from fl_v3.training.runtime_state import project_batch_for_mode
    from fl_v3.utils.runtime import normalize_precision, precision_autocast_context

    mode = str(run_config["model-mode"])
    precision = normalize_precision(str(run_config["precision"]))

    out: List[SampleDecode] = []
    elapsed: list[float] = []
    mode_context = model.serialized_mode(False) if hasattr(model, "serialized_mode") else nullcontext(model)
    with mode_context:
        for batch in eval_loader:
            projected = project_batch_for_mode(batch, mode)
            moved = _move_to_device(projected, device)
            if timing is not None and device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with precision_autocast_context(precision, device):
                head = model(moved)
            if precision == "fp16" and device.type == "cuda":
                head = _float_tensors(head)
            decoded = model.decode(head, score_threshold=thr)
            if timing is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed.append(time.perf_counter() - started)
            bs = len(batch["gt_boxes"])
            for b in range(bs):
                d = decoded[b]
                out.append(SampleDecode(
                    sample_token=str(batch["sample_token"][b]),
                    boxes=_np(d["boxes"]).astype(np.float64).reshape(-1, 7),
                    scores=_np(d["scores"]).astype(np.float64).reshape(-1),
                    labels=_np(d["labels"]).astype(np.int64).reshape(-1),
                    velocity=_np(d["velocity"]).astype(np.float64).reshape(-1, 2),
                    lidar2ego=_np(batch["lidar2ego"][b]).astype(np.float64).reshape(4, 4),
                    ego2global_lidar=_np(batch["ego2global_lidar"][b]).astype(np.float64).reshape(4, 4),
                    lidar2img=_np(batch["lidar2img"][b]).astype(np.float64).reshape(6, 4, 4),
                    gt_boxes=_np(batch["gt_boxes"][b]).astype(np.float64).reshape(-1, 7),
                    gt_labels=_np(batch["gt_labels"][b]).astype(np.int64).reshape(-1),
                    gt_num_lidar_pts=_np(batch["gt_num_lidar_pts"][b]).astype(np.int64).reshape(-1),
                    gt_in_range=_np(batch["gt_in_range"][b]).astype(bool).reshape(-1),
                    gt_velocity=_np(batch["gt_velocity"][b]).astype(np.float64).reshape(-1, 2),
                    gt_ann_tokens=list(batch["gt_ann_tokens"][b]),
                    gt_names=list(batch["gt_names"][b]),
                    gt_attribute=list(batch["gt_attribute"][b]),
                ))
    if timing is not None:
        timing.clear()
        timing.update({"batches": len(elapsed), "total_seconds": sum(elapsed), "batch_seconds": elapsed})
    return out


def build_results_dict(
    decodes: Sequence[SampleDecode],
    class_names: Sequence[str],
    all_eval_tokens: Sequence[str],
    run_config: Optional[dict] = None,
) -> Dict[str, object]:
    """Build the deterministic ``{meta, results}`` submission dict from the shared decode.

    EVERY token in ``all_eval_tokens`` is a key (empty ``[]`` if no detections — the devkit
    asserts the pred/GT token sets are equal). Boxes per sample are content-sorted.
    """
    rc = run_config or {}
    vms = float(rc.get("attr-vehicle-moving-speed", 0.2))
    pms = float(rc.get("attr-pedestrian-moving-speed", 0.2))
    results: Dict[str, list] = {str(t): [] for t in all_eval_tokens}
    for d in decodes:
        decoded = {"boxes": d.boxes, "scores": d.scores, "labels": d.labels, "velocity": d.velocity}
        boxes = decoded_sample_to_boxes(
            decoded, d.lidar2ego, d.ego2global_lidar, d.sample_token, class_names,
            vehicle_moving_speed=vms, pedestrian_moving_speed=pms,
        )
        if d.sample_token not in results:
            raise KeyError(f"decoded sample {d.sample_token} not in the eval-split token set")
        results[d.sample_token] = [b.serialize() for b in boxes]
    mode = str(rc.get("model-mode", "fusion"))
    submission: Dict[str, object] = {"meta": submission_meta(mode), "results": results}
    identity_keys = (
        "resolved-config-sha256", "checkpoint-sha256",
        "nuscenes-train-cache-logical-sha256", "nuscenes-train-cache-pickle-sha256",
        "nuscenes-train-cache-sidecar-sha256", "nuscenes-val-cache-logical-sha256",
        "nuscenes-val-cache-pickle-sha256", "nuscenes-val-cache-sidecar-sha256",
        "nuscenes-zip-manifest-logical-sha256", "nuscenes-zip-manifest-file-sha256",
    )
    if any(k in rc for k in identity_keys):
        missing = [k for k in identity_keys if not rc.get(k)]
        if missing:
            raise ValueError(f"evaluation identity fields missing: {missing}")
        submission["fl_v3_provenance"] = {"model_mode": mode, **{k: rc[k] for k in identity_keys}}
    return submission


def _extract_metrics(metrics, metric_data_list, cfg) -> Dict[str, object]:
    """Pull mAP / NDS / per-class AP / 5 TP errors / official per-class recall out of the devkit objects."""
    ser = metrics.serialize()
    out: Dict[str, object] = {
        "mAP": float(ser["mean_ap"]),
        "NDS": float(ser["nd_score"]),
        "tp_errors": {k: float(v) for k, v in ser["tp_errors"].items()},
        "per_class_mean_ap": {k: float(v) for k, v in ser["mean_dist_aps"].items()},
        "label_aps": {c: {str(dt): float(ap) for dt, ap in d.items()} for c, d in ser["label_aps"].items()},
        "label_tp_errors": {c: {k: float(v) for k, v in d.items()} for c, d in ser["label_tp_errors"].items()},
        "eval_time": float(ser.get("eval_time", 0.0)),
    }
    # Official per-class recall = max recall achieved at the TP distance threshold (2.0 m).
    dist_tp = float(cfg.dist_th_tp)
    recalls: Dict[str, float] = {}
    for c in cfg.class_names:
        try:
            md = metric_data_list[(c, dist_tp)]
            recalls[c] = float(md.recall[md.max_recall_ind])
        except Exception:
            recalls[c] = 0.0
    out["official_recall_at_tp"] = recalls
    out["car_recall"] = float(recalls.get("car", 0.0))
    out["car_ap_2m"] = float(out["label_aps"].get("car", {}).get(str(dist_tp), 0.0))
    out["car_mean_ap"] = float(out["per_class_mean_ap"].get("car", 0.0))
    return out


def run_detection_eval(
    nusc,
    decodes: Sequence[SampleDecode],
    eval_set: str,
    version: str,
    output_dir: str,
    class_names: Sequence[str],
    *,
    all_eval_tokens: Optional[Sequence[str]] = None,
    run_config: Optional[dict] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """Build the submission, run ``DetectionEval(...).evaluate()``, return the metric dict.

    ``all_eval_tokens`` defaults to the decoded tokens; for the official metric they MUST be
    the FULL ``eval_set`` token set (DetectionEval loads the whole split's GT and asserts the
    token sets match). Writes ``<output_dir>/results.json`` (the submission) — the only file
    we write; ``.evaluate()`` itself writes nothing.
    """
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.common.config import config_factory

    assert_version_split(version, eval_set)
    os.makedirs(output_dir, exist_ok=True)
    tokens = list(all_eval_tokens) if all_eval_tokens is not None else [d.sample_token for d in decodes]
    submission = build_results_dict(decodes, class_names, tokens, run_config=run_config)
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, sort_keys=True)

    cfg = config_factory("detection_cvpr_2019")
    nusc_eval = DetectionEval(nusc, config=cfg, result_path=result_path,
                              eval_set=eval_set, output_dir=output_dir, verbose=verbose)
    metrics, metric_data_list = nusc_eval.evaluate()
    out = _extract_metrics(metrics, metric_data_list, cfg)
    out.update({
        "eval_set": eval_set, "version": version,
        "n_eval_samples": len(tokens),
        "n_decoded_samples": len(decodes),
        "result_path": result_path,
    })
    return out


def gt_as_pred_submission(
    decodes_or_infos,
    class_names: Sequence[str],
    all_eval_tokens: Sequence[str],
    *,
    copy_velocity: bool = True,
    copy_attribute: bool = True,
    num_pts_by_ann: Optional[Dict[str, int]] = None,
    run_config: Optional[dict] = None,
) -> Dict[str, object]:
    """Build a GT-as-pred submission: each T1 GT box → global ``DetectionBox`` at score 1.0.

    The §0.1 sanity: submitting the GT itself as predictions (vs the devkit's own ``load_gt``
    GT side, loaded internally by DetectionEval) must give **per-class AP ≈ 1**. Copies the GT
    velocity (→ AVE≈0) and GT attribute (→ AAE≈0) when asked.

    ``num_pts_by_ann`` (``ann_token → num_lidar+num_radar`` from the devkit) makes each pred
    carry the SAME ``num_pts`` the devkit GT has, so the ``filter_eval_boxes`` ``num_pts==0``
    drop is identical on both sides → exact AP=1.0 (T1 carries lidar-only counts, so a GT car
    with 0 lidar but >0 radar would otherwise be a recall miss). Without it, predictions use the
    ``num_pts=-1`` sentinel (the production convention; never points-filtered).
    """
    from fl_v3.eval.box_to_global import (
        convert_box_to_global, detection_box_sort_key, make_detection_box,
    )

    results: Dict[str, list] = {str(t): [] for t in all_eval_tokens}
    for d in decodes_or_infos:
        tok = d.sample_token
        boxes = []
        for m in range(d.gt_boxes.shape[0]):
            cls = int(d.gt_labels[m])
            if cls < 0 or cls >= len(class_names):
                continue
            name = class_names[cls]
            v_lidar = d.gt_velocity[m] if copy_velocity else None
            conv = convert_box_to_global(d.gt_boxes[m], d.lidar2ego, d.ego2global_lidar,
                                         velocity_lidar=v_lidar)
            attr = (d.gt_attribute[m] if (copy_attribute and m < len(d.gt_attribute)) else None)
            ann = d.gt_ann_tokens[m] if m < len(d.gt_ann_tokens) else ""
            npv = int(num_pts_by_ann[ann]) if (num_pts_by_ann is not None and ann in num_pts_by_ann) else -1
            boxes.append(make_detection_box(tok, name, 1.0, conv, attribute_name=attr, num_pts=npv))
        boxes.sort(key=detection_box_sort_key)
        results[tok] = [b.serialize() for b in boxes]
    mode = str((run_config or {}).get("model-mode", "fusion"))
    return {"meta": submission_meta(mode), "results": results}
