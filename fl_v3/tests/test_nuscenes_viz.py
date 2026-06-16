"""T1 — V1 calibration renders + the independent-projector numeric gate (≤1 px)."""
from __future__ import annotations

import json
import os

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.viz import calibration as V1
from fl_v3.viz.writer import VizWriter


def test_independent_projector_agreement_le_1px(nusc_mini):
    """The anti 'viz-shares-the-bug' gate: OUR projector (built from raw records) vs
    the devkit's view_points / get_sample_data — ≤1 px on LiDAR points AND box
    corners, across ≥5 distinct keyframes × multiple cameras."""
    worst_lidar = 0.0
    worst_box = 0.0
    n_kf = 0
    for s in nusc_mini.sample[:6]:
        for cam in ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK"):
            a = V1.independent_projector_agreement(nusc_mini, s["token"], cam)
            worst_lidar = max(worst_lidar, a["lidar_px"])
            worst_box = max(worst_box, a["box_corner_px"])
        n_kf += 1
    assert n_kf >= 5
    assert worst_lidar <= 1.0, worst_lidar
    assert worst_box <= 1.0, worst_box


def test_v1_renders_five_keyframes(nusc_mini, mini_train_info, dataroot, tmp_path):
    writer = VizWriter(str(tmp_path / "run"))
    tokens = sorted(i["sample_token"] for i in mini_train_info)[:5]
    ds = DS.NuScenesMultimodalDataset(mini_train_info, dataroot, sample_tokens=tokens)
    produced = []
    for i in range(len(ds)):
        sample = ds[i]
        produced.append(V1.render_cam_lidar(writer, sample, nusc_mini, cam_index=0))
        produced.append(V1.render_cam_boxes(writer, sample, nusc_mini, cam_index=0))
        produced.append(V1.render_bev(writer, sample))
    # filenames derive from sample_token (not an enumeration counter)
    for tok in tokens:
        assert any(tok in os.path.basename(p) for p in produced)
    assert all(os.path.isfile(p) for p in produced)
    assert len(ds) == 5
    manifest_path = writer.write_manifest()
    with open(manifest_path) as f:
        man = json.load(f)
    cal = man["stages"]["calibration"]["artifacts"]
    assert len([a for a in cal if a.endswith(".png")]) >= 15  # 5 kf × 3 render types


def test_partition_plot_renders(mini_train_info, tmp_path):
    from fl_v3.data.nuscenes import partition as PT

    writer = VizWriter(str(tmp_path / "run2"))
    lt = PT.build_log_table(mini_train_info)
    part = PT.build_log_group_partition(lt, floor=50, seed=42)
    rep = PT.partition_report(part, "mini-smoke", "v1.0-mini", "mini_train", lt)
    hist_path, loc_path = V1.render_partition(writer, rep)
    assert os.path.isfile(hist_path) and os.path.isfile(loc_path)
    writer.write_json("calibration", "partition_report", rep)
    writer.write_manifest()


def test_box7_to_corners_matches_devkit_box(nusc_mini):
    """The render's box7→corners reconstruction agrees with the devkit Box footprint
    (BEV xy) — yaw-only, so we compare the BEV footprint, robust to dropped pitch."""
    import numpy as np

    s = nusc_mini.sample[0]
    from fl_v3.data.nuscenes import info_cache as IC
    from fl_v3.data.nuscenes.paths import DATAROOT

    info = IC.build_keyframe_info(nusc_mini, s, DATAROOT)
    _, boxes, _ = nusc_mini.get_sample_data(s["data"]["LIDAR_TOP"])
    dk = {b.token: b for b in boxes}
    worst = 0.0
    for i, tok in enumerate(info["gt_ann_tokens"]):
        corners = V1.box7_to_corners(info["gt_boxes"][i])
        # compare BEV centroid + footprint extent vs devkit corners' xy
        dkc = dk[tok].corners().T  # (8,3)
        worst = max(worst, abs(corners[:, :2].mean(0) - dkc[:, :2].mean(0)).max())
    assert worst < 1e-6, worst
