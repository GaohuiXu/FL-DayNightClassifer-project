#!/usr/bin/env python
"""Offline GT-database build (MCR P1 GT-paste) — crop per-object LiDAR point sets from the train split.

Walks a split's keyframe infos, loads each (multi-sweep) cloud via the SAME dataset loaders the training
run uses (so cropped objects match the run's point density + columns), crops each GT object's points into
its box-local frame, and writes per-class pickles + a meta.json. Deterministic (a fixed-order walk;
per-class subsampling uses a fixed RandomState seed). Pure CPU/IO — runs on the login node, no GPU.

  source fl_v3/scripts/arrhenius_env.sh
  arrhenius_load_modules build
  arrhenius_activate_env
  python fl_v3/scripts/build_gt_database.py \
    --cache-dir <info_cache_msweep10> --version v1.0-trainval --split train --n-sweeps 10 \
    --dataroot /path/to/NuScenes_v1.0 \
    --out-dir ./fl_outputs/nuscenes/gt_database_msweep10 \
    --classes trailer,construction_vehicle,bus,truck,bicycle,motorcycle --min-points 5 --max-per-class 8000
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.dataset import _load_lidar, _load_multisweep
from fl_v3.data.nuscenes.gt_database import crop_object_points


def _load_info_list(cache_dir: str, version: str, split: str):
    fn = os.path.join(cache_dir, f"nuscenes_info_{version}_{split}_t1.v1.pkl")
    with open(fn, "rb") as f:
        obj = pickle.load(f)
    return obj["info_list"] if isinstance(obj, dict) and "info_list" in obj else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-sweeps", type=int, default=10)
    ap.add_argument(
        "--dataroot",
        default=os.environ.get("ARRHENIUS_NUSCENES_DATAROOT") or os.environ.get("NUSCENES_DATAROOT") or "",
        help="nuScenes dataroot; defaults to ARRHENIUS_NUSCENES_DATAROOT/NUSCENES_DATAROOT.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--classes", default="trailer,construction_vehicle,bus,truck,bicycle,motorcycle")
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20259)
    args = ap.parse_args()
    if not args.dataroot:
        raise SystemExit(
            "--dataroot is required unless ARRHENIUS_NUSCENES_DATAROOT or NUSCENES_DATAROOT is set; "
            "do not rely on site-specific historical paths."
        )

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    info_list = _load_info_list(args.cache_dir, args.version, args.split)
    print(f"[gtdb] {len(info_list)} keyframes, classes={classes}, n_sweeps={args.n_sweeps}", flush=True)

    db = {c: [] for c in classes}
    n_total = 0
    for ki, info in enumerate(info_list):
        if args.n_sweeps > 1:
            cloud = _load_multisweep(info, args.dataroot, args.n_sweeps)            # [P,6]
        else:
            cloud = _load_lidar(P.abspath_from_relative(info["lidar_rel_path"], args.dataroot))  # [P,5]
        boxes = np.asarray(info["gt_boxes"], dtype=np.float64)
        labels = np.asarray(info["gt_labels"])
        names = list(info["gt_names"])
        for j in range(boxes.shape[0]):
            nm = names[j]
            if nm not in db:
                continue
            local = crop_object_points(cloud, boxes[j])
            if local.shape[0] < args.min_points:
                continue
            db[nm].append({
                "points": local,                                   # [n,4] box-local x,y,z,intensity
                "box7": boxes[j].astype(np.float32),
                "label": int(labels[j]),
                "name": nm,
                "num_pts": int(local.shape[0]),
            })
            n_total += 1
        if (ki + 1) % 2000 == 0:
            print(f"[gtdb] {ki+1}/{len(info_list)} keyframes, {n_total} objects so far "
                  f"({ {c: len(db[c]) for c in classes} })", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    counts = {}
    for c in classes:
        objs = db[c]
        if len(objs) > args.max_per_class:
            keep = rng.choice(len(objs), size=args.max_per_class, replace=False)
            objs = [objs[i] for i in sorted(keep.tolist())]        # sorted ⇒ deterministic order
        with open(os.path.join(args.out_dir, f"{c}.pkl"), "wb") as f:
            pickle.dump(objs, f, protocol=pickle.HIGHEST_PROTOCOL)
        counts[c] = len(objs)
        print(f"[gtdb] wrote {c}: {len(objs)} objects", flush=True)

    meta = {"version": args.version, "split": args.split, "n_sweeps": args.n_sweeps,
            "classes": classes, "min_points": args.min_points, "max_per_class": args.max_per_class,
            "seed": args.seed, "counts": counts, "n_keyframes": len(info_list)}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gtdb] DONE → {args.out_dir}  counts={counts}", flush=True)


if __name__ == "__main__":
    main()
