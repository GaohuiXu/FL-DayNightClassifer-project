#!/usr/bin/env python
"""Offline GT-database build (MCR P1 GT-paste) — crop per-object LiDAR point sets from the train split.

Walks a split's keyframe infos, loads each (multi-sweep) cloud via the SAME dataset loaders the training
run uses (so cropped objects match the run's point density + columns), crops each GT object's points into
its box-local frame, and writes per-class pickles + a meta.json. The input cache depth/content hash and,
for ZIP mode, both the logical manifest hash and manifest-file SHA-256 are mandatory provenance inputs.
Deterministic (a fixed-order walk; per-class subsampling uses a fixed RandomState seed). Pure CPU/IO.

  source fl_v3/scripts/arrhenius_env.sh
  arrhenius_load_modules build
  arrhenius_activate_env
  python fl_v3/scripts/build_gt_database.py \
    --cache-dir <info_cache_msweep10> --version v1.0-trainval --split train --n-sweeps 10 \
    --expected-cache-hash <t1.v2-train-cache-hash> \
    --dataroot /path/to/NuScenes_v1.0 \
    --zip-manifest /path/to/nuscenes_trainval_zip_manifest.sqlite \
    --expected-manifest-hash <logical-manifest-hash> \
    --expected-manifest-file-sha256 <manifest-file-sha256> \
    --out-dir ./fl_outputs/nuscenes/gt_database_msweep10 \
    --classes trailer,construction_vehicle,bus,truck,bicycle,motorcycle --min-points 5 --max-per-class 8000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes.dataset import _load_lidar, _load_multisweep
from fl_v3.data.nuscenes.gt_database import crop_object_points
from fl_v3.data.nuscenes.zip_backend import (
    NuScenesBlobStore,
    TRAINVAL_ARCHIVE_NAMES,
    manifest_summary,
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_sha256(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return digest


def _load_info_list(
    cache_dir: str,
    version: str,
    split: str,
    n_sweeps: int,
    expected_cache_hash: str,
):
    """Load only the explicit t1.v2 depth and frozen cache identity."""
    expected = _required_sha256(expected_cache_hash, "expected cache hash")
    return IC.load_cache(
        cache_dir,
        version,
        split,
        n_sweeps=n_sweeps,
        expected_cache_hash=expected,
    )


def _open_blob_store(
    dataroot: str,
    version: str,
    zip_manifest: str,
    expected_manifest_hash: str,
    expected_manifest_file_sha256: str,
) -> tuple[NuScenesBlobStore, dict]:
    """Resolve directory/ZIP mode and return its fail-closed provenance record."""
    dataset_report = P.verify_dataset(version, dataroot)
    backend = dataset_report["blob_backend"]
    if backend == "directory":
        if zip_manifest or expected_manifest_hash or expected_manifest_file_sha256:
            raise ValueError(
                "directory backend must not be relabeled with ZIP-manifest provenance"
            )
        store = NuScenesBlobStore(dataroot)
        return store, {
            "blob_backend": "directory",
            "zip_manifest_path": None,
            "zip_manifest_hash": None,
            "zip_manifest_file_sha256": None,
        }

    manifest = os.path.abspath(zip_manifest or P.get_zip_manifest(required=True))
    expected_logical = _required_sha256(
        expected_manifest_hash, "expected manifest logical hash"
    )
    expected_file = _required_sha256(
        expected_manifest_file_sha256, "expected manifest file SHA-256"
    )
    summary = manifest_summary(manifest)
    if summary["manifest_hash"] != expected_logical:
        raise ValueError(
            "nuScenes ZIP manifest does not match frozen logical hash: "
            f"expected={expected_logical}, actual={summary['manifest_hash']}"
        )
    actual_file = _sha256_file(manifest)
    if actual_file != expected_file:
        raise ValueError(
            "nuScenes ZIP manifest file does not match frozen SHA-256: "
            f"expected={expected_file}, actual={actual_file}"
        )
    if version == "v1.0-trainval" and tuple(summary["archive_names"]) != TRAINVAL_ARCHIVE_NAMES:
        raise ValueError(
            "trainval GT database requires the exact trainval01..trainval10 archive set"
        )
    store = NuScenesBlobStore(dataroot, manifest_path=manifest)
    return store, {
        "blob_backend": "zip",
        "zip_manifest_path": manifest,
        "zip_manifest_format": summary["format_version"],
        "zip_manifest_hash": expected_logical,
        "zip_manifest_file_sha256": expected_file,
        "zip_manifest_archive_names": list(summary["archive_names"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-sweeps", type=int, default=10)
    ap.add_argument(
        "--expected-cache-hash",
        required=True,
        help="Frozen canonical t1.v2 cache hash for this exact split/depth.",
    )
    ap.add_argument(
        "--dataroot",
        default=os.environ.get("ARRHENIUS_NUSCENES_DATAROOT") or os.environ.get("NUSCENES_DATAROOT") or "",
        help="nuScenes dataroot; defaults to ARRHENIUS_NUSCENES_DATAROOT/NUSCENES_DATAROOT.",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--zip-manifest",
        default="",
        help="External S01 SQLite manifest. Required for ZIP mode; forbidden in directory mode.",
    )
    ap.add_argument(
        "--expected-manifest-hash",
        default="",
        help="Frozen logical S01 manifest hash; required for ZIP mode.",
    )
    ap.add_argument(
        "--expected-manifest-file-sha256",
        default="",
        help="Frozen SHA-256 of the SQLite manifest file; required for ZIP mode.",
    )
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
    P.resolve_writable(args.out_dir, args.dataroot)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    info_list, cache_meta = _load_info_list(
        args.cache_dir,
        args.version,
        args.split,
        args.n_sweeps,
        args.expected_cache_hash,
    )
    blob_store, blob_provenance = _open_blob_store(
        args.dataroot,
        args.version,
        args.zip_manifest,
        args.expected_manifest_hash,
        args.expected_manifest_file_sha256,
    )
    print(
        f"[gtdb] {len(info_list)} keyframes, classes={classes}, n_sweeps={args.n_sweeps}, "
        f"cache_hash={cache_meta['cache_hash']}, backend={blob_provenance['blob_backend']}",
        flush=True,
    )

    db = {c: [] for c in classes}
    n_total = 0
    try:
        for ki, info in enumerate(info_list):
            if args.n_sweeps > 1:
                cloud = _load_multisweep(
                    info, args.dataroot, args.n_sweeps, blob_store=blob_store
                )  # [P,6]
            else:
                cloud = _load_lidar(
                    info["lidar_rel_path"], blob_store=blob_store
                )  # [P,5]
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
    finally:
        blob_store.close()

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

    meta = {
        "version": args.version,
        "split": args.split,
        "n_sweeps": args.n_sweeps,
        "dataroot": os.path.abspath(args.dataroot),
        "cache_format_version": cache_meta["format_version"],
        "cache_hash": cache_meta["cache_hash"],
        **blob_provenance,
        "classes": classes,
        "min_points": args.min_points,
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "counts": counts,
        "n_keyframes": len(info_list),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[gtdb] DONE → {args.out_dir}  counts={counts}", flush=True)


if __name__ == "__main__":
    main()
