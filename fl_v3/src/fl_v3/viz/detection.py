"""V4 detection renderers (clean) (fl_v3 T4) — into the ``detection`` viz stage.

GT vs **clean-decoded** boxes (cam + BEV), a per-class decoded-score map, and the per-target
score table. The load-bearing invariant (T4_SPEC GATE): V4 renders the **EXACT box set the
evaluator scores** — it consumes the cached :class:`~fl_v3.eval.detection_eval.SampleDecode`
(``detector.decode`` at the production ``det-score-threshold``/``det-max-objects``) and reuses
the SAME clean-detection matcher as the ASR harness (:func:`asr.detected_target_gt`). **No
V4-only threshold / sort / limit** — so the visual TP/FN agrees with the metric TP/FN, including
the lowest-scoring eligible boxes near ``τ_clean``/``d_clean``. The disappeared/phantom highlight
hooks are wired but only exercised at T5 (clean V4 shows no disappearance).

Mirrors ``viz/calibration.py`` (Agg backend, ``sample_token``-stable names); does not rewrite
``writer.py``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from fl_v3.data.nuscenes import transforms as T
from fl_v3.viz.calibration import box7_to_corners, _draw_box_2d
from fl_v3.eval.asr import AsrThresholds, detected_target_gt, evaluate_sample_eligibility

_FIG_DPI = 100


def v4_target_table(decode, thr: AsrThresholds) -> List[dict]:
    """Per target-class GT: the 6 criteria + the clean match (score, dist) + the TP/FN verdict.

    Reuses :func:`asr.evaluate_sample_eligibility` (the SAME matcher + thresholds the metric/ASR
    use) so the table — and thus every V4 highlight derived from it — agrees with the metric.
    ``tp`` ≡ clean-detected (a score≥τ_clean decoded box matched within d_clean); ``fn`` ≡ a
    target-class GT not clean-detected.
    """
    rows = evaluate_sample_eligibility(decode, thr)
    for r in rows:
        r["tp"] = bool(r["clean_detected"])
        r["fn"] = bool(not r["clean_detected"])
    return rows


def _decoded_target(decode, thr: AsrThresholds):
    """Decoded target-class boxes (cx,cy,yaw,score) — the SAME set the evaluator converts."""
    tid = thr.target_id
    mask = decode.labels == tid
    return decode.boxes[mask], decode.scores[mask]


def render_v4_bev(writer, decode, thr: AsrThresholds, max_range: float = 55.0) -> str:
    """BEV top-down: GT target boxes (TP green / FN red) + decoded target boxes colored by score."""
    table = {r["gt_idx"]: r for r in v4_target_table(decode, thr)}
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=_FIG_DPI)
    # decoded target boxes (footprint), colored by score
    dboxes, dscores = _decoded_target(decode, thr)
    sc_handle = None
    for i in range(dboxes.shape[0]):
        corners = box7_to_corners(dboxes[i])[:4, :2]
        col = plt.cm.viridis(float(min(1.0, max(0.0, dscores[i]))))
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=col, linewidth=1.3))
        cx, cy = float(dboxes[i, 0]), float(dboxes[i, 1])
        ax.text(cx, cy, f"{dscores[i]:.2f}", color=col, fontsize=6, ha="center", va="center")
    # GT target boxes: green if clean-detected (TP), red if missed (FN)
    tid = thr.target_id
    for m in range(decode.gt_boxes.shape[0]):
        if int(decode.gt_labels[m]) != tid:
            continue
        corners = box7_to_corners(decode.gt_boxes[m])[:4, :2]
        r = table.get(m, {})
        col = "lime" if r.get("tp") else "red"
        ls = "-" if r.get("eligible") else "--"
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=col, linewidth=1.0, linestyle=ls))
    ax.plot(0, 0, "k+", markersize=10)
    ax.set_xlim(-max_range, max_range); ax.set_ylim(-max_range, max_range); ax.set_aspect("equal")
    ax.set_xlabel("x fwd (m)"); ax.set_ylabel("y left (m)")
    n_tp = sum(1 for r in table.values() if r.get("tp"))
    n_fn = sum(1 for r in table.values() if r.get("fn"))
    ax.set_title(f"V4 BEV (clean) | {decode.sample_token} | GT car TP=lime FN=red | "
                 f"decoded=viridis(score) | TP={n_tp} FN={n_fn}", fontsize=8)
    path = writer.figure_path("detection", f"v4_bev__{decode.sample_token}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1); plt.close(fig)
    return path


def render_v4_cam(writer, decode, image_chw, cam_index, thr: AsrThresholds) -> str:
    """Cam image + projected GT target boxes (TP green/FN red) vs decoded target boxes (cyan)."""
    img = np.asarray(image_chw).transpose(1, 2, 0)
    H, W = img.shape[:2]
    l2i = decode.lidar2img[cam_index]
    table = {r["gt_idx"]: r for r in v4_target_table(decode, thr)}
    fig, ax = plt.subplots(figsize=(W / _FIG_DPI, H / _FIG_DPI), dpi=_FIG_DPI)
    ax.imshow(img)
    tid = thr.target_id
    for m in range(decode.gt_boxes.shape[0]):
        if int(decode.gt_labels[m]) != tid:
            continue
        uv, depth = T.project_to_image(box7_to_corners(decode.gt_boxes[m]), l2i)
        if np.all(depth <= 1.0):
            continue
        _draw_box_2d(ax, uv, depth, color=("lime" if table.get(m, {}).get("tp") else "red"))
    dboxes, dscores = _decoded_target(decode, thr)
    for i in range(dboxes.shape[0]):
        uv, depth = T.project_to_image(box7_to_corners(dboxes[i]), l2i)
        if np.all(depth <= 1.0):
            continue
        _draw_box_2d(ax, uv, depth, color="cyan", lw=0.8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title(f"V4 cam{cam_index} (clean) | {decode.sample_token} | GT TP=lime FN=red dec=cyan", fontsize=8)
    path = writer.figure_path("detection", f"v4_cam{cam_index}__{decode.sample_token}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0); plt.close(fig)
    return path


def render_v4(writer, decode, thr: AsrThresholds, *, image_chw_by_cam=None,
              cam_index: int = 0) -> Dict[str, object]:
    """Render the V4 panels + write the per-target score table for one sample.

    ``image_chw_by_cam`` (optional ``[6,3,H,W]``) enables the cam panel; without it only the BEV
    + table render (still from the cached decode). Returns the artifact paths + the table.
    """
    table = v4_target_table(decode, thr)
    bev = render_v4_bev(writer, decode, thr)
    cam = None
    if image_chw_by_cam is not None:
        cam = render_v4_cam(writer, decode, np.asarray(image_chw_by_cam)[cam_index], cam_index, thr)
    table_json = writer.write_json("detection", f"v4_table__{decode.sample_token}", {
        "sample_token": decode.sample_token,
        "thresholds": {"tau_clean": thr.tau_clean, "d_clean": thr.d_clean,
                       "tau_pts": thr.tau_pts, "target_class": thr.target_class},
        "n_tp": int(sum(1 for r in table if r["tp"])),
        "n_fn": int(sum(1 for r in table if r["fn"])),
        "n_eligible": int(sum(1 for r in table if r["eligible"])),
        "targets": [{k: r[k] for k in ("ann_token", "gt_idx", "c1_class", "c2_frustum",
                                       "c3_lidar_pts", "c4_in_range", "c5_clean_score",
                                       "c6_clean_match", "clean_detected", "eligible",
                                       "match_dist", "match_score", "tp", "fn")} for r in table],
    })
    return {"bev": bev, "cam": cam, "table_json": table_json, "table": table}
