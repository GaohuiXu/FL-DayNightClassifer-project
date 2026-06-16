"""Generate the V1 calibration artifacts (T1 hard pre-trust gate).

Builds the mini info-cache, the log-group partition + report, and renders the V1
calibration figures (cam+LiDAR, cam+3D-GT, BEV, partition plots) for ≥5 distinct
keyframes, with the independent-projector overlay. Login-node-safe (metadata +
mini blobs only; no GPU, no training). Run via:

    PYTHONPATH=fl_v3/src bash fl_v3/scripts/run_in_venv.sh \
        python fl_v3/scripts/run_v1_calibration.py --out fl_outputs/.../t1_v1
"""
from __future__ import annotations

import argparse
import os

from nuscenes import NuScenes

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import partition as PT
from fl_v3.data.nuscenes import paths as P
from fl_v3.viz import calibration as V1
from fl_v3.viz.writer import VizWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--split", default="mini_train")
    ap.add_argument("--n-keyframes", type=int, default=5)
    ap.add_argument("--floor", type=int, default=50)  # mini is degenerate; small floor
    ap.add_argument("--cache-dir", default="fl_outputs/nuscenes/info_cache")
    ap.add_argument("--out", default="fl_outputs/nuscenes/experiments/cycle_04/t1_v1_smoke")
    args = ap.parse_args()

    rep = P.verify_dataset(args.version)
    print(f"[v1] verify_dataset({args.version}) OK: {rep}")

    nusc = NuScenes(version=args.version, dataroot=P.DATAROOT, verbose=False)
    info_list, meta = IC.get_or_build_cache(
        nusc, args.cache_dir, args.version, args.split, "mini-smoke", P.DATAROOT)
    print(f"[v1] info-cache: n_samples={meta['n_samples']} n_boxes={meta['n_boxes']} hash={meta['cache_hash'][:16]}…")

    writer = VizWriter(args.out)
    tokens = sorted(i["sample_token"] for i in info_list)[: args.n_keyframes]
    ds = DS.NuScenesMultimodalDataset(info_list, P.DATAROOT, sample_tokens=tokens)
    worst_lidar = worst_box = 0.0
    for i in range(len(ds)):
        sample = ds[i]
        V1.render_cam_lidar(writer, sample, nusc, cam_index=0)
        V1.render_cam_boxes(writer, sample, nusc, cam_index=0)
        V1.render_bev(writer, sample)
        a = V1.independent_projector_agreement(nusc, sample["sample_token"], "CAM_FRONT")
        worst_lidar = max(worst_lidar, a["lidar_px"]); worst_box = max(worst_box, a["box_corner_px"])
        print(f"[v1] {sample['sample_token']}  Δlidar={a['lidar_px']:.4f}px  Δbox={a['box_corner_px']:.2e}px  M={sample['num_boxes']}")

    # partition + report + plots
    lt = PT.build_log_table(info_list)
    part = PT.build_log_group_partition(lt, floor=args.floor, seed=42)
    report = PT.partition_report(part, "mini-smoke", args.version, args.split, lt)
    V1.render_partition(writer, report)
    writer.write_json("calibration", "partition_report", report)
    manifest = writer.write_manifest()
    print(f"[v1] independent-projector worst: lidar={worst_lidar:.4f}px box={worst_box:.2e}px (gate ≤1px)")
    print(f"[v1] partition: N={report['num_clients']} mode={report['mode']} band={report['n_at_floor_band']}")
    print(f"[v1] manifest: {manifest}")


if __name__ == "__main__":
    main()
