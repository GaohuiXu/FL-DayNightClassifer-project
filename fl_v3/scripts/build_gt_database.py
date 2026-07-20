#!/usr/bin/env python
"""Offline GT-database build (MCR P1 GT-paste) — crop per-object LiDAR point sets from the train split.

Walks a split's keyframe infos, loads each (multi-sweep) cloud via the SAME dataset loaders the training
run uses (so cropped objects match the run's point density + columns), crops each GT object's points into
its box-local frame, and writes per-class pickles + a meta.json. The input cache
depth/canonical hash plus physical pickle/sidecar SHA-256 values and, for ZIP mode,
both the logical manifest hash and manifest-file SHA-256 are mandatory provenance inputs.
Deterministic (a fixed-order walk; per-class subsampling uses a fixed RandomState seed). Pure CPU/IO.

  source fl_v3/scripts/arrhenius_env.sh
  arrhenius_load_modules build
  arrhenius_activate_env
  python fl_v3/scripts/build_gt_database.py \
    --cache-dir <info_cache_msweep10> --version v1.0-trainval --split train --n-sweeps 10 \
    --expected-cache-hash <t1.v2-train-cache-hash> \
    --expected-cache-file-sha256 <frozen-cache-pickle-sha256> \
    --expected-cache-sidecar-sha256 <frozen-cache-sidecar-sha256> \
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
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes.dataset import _load_lidar, _load_multisweep
from fl_v3.data.nuscenes.gt_database import (
    crop_object_points,
    crop_object_points_center_relative,
)
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
    expected_cache_file_sha256: str,
    expected_cache_sidecar_sha256: str,
):
    """Load one exact cache only after binding both physical artifacts.

    The canonical cache hash intentionally covers raw nuScenes inputs, not every
    serialized derived tensor.  The pickle and sidecar SHA-256 values therefore
    form a separate mandatory identity boundary.  Both files are hashed before
    and after deserialization so a mismatch or in-flight change fails before any
    sensor blob is opened or any object points are cropped.
    """
    expected_canonical = _required_sha256(
        expected_cache_hash, "expected cache canonical hash"
    )
    expected_pickle = _required_sha256(
        expected_cache_file_sha256, "expected cache pickle SHA-256"
    )
    expected_sidecar = _required_sha256(
        expected_cache_sidecar_sha256, "expected cache sidecar SHA-256"
    )
    pkl_path, sidecar_path = IC.cache_paths(
        cache_dir, version, split, n_sweeps=n_sweeps
    )
    paths = {
        "pickle": os.path.abspath(pkl_path),
        "sidecar": os.path.abspath(sidecar_path),
    }
    expected_files = {"pickle": expected_pickle, "sidecar": expected_sidecar}

    def snapshot() -> dict:
        state = {}
        for label, path in paths.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"required nuScenes cache {label} artifact is missing: {path}"
                )
            state[label] = {
                "path": path,
                "bytes": os.path.getsize(path),
                "sha256": _sha256_file(path),
            }
            if state[label]["sha256"] != expected_files[label]:
                raise ValueError(
                    f"nuScenes cache {label} physical SHA mismatch: "
                    f"expected={expected_files[label]}, actual={state[label]['sha256']}, "
                    f"path={path}"
                )
        return state

    before = snapshot()
    info_list, meta = IC.load_cache(
        cache_dir,
        version,
        split,
        n_sweeps=n_sweeps,
        expected_cache_hash=expected_canonical,
    )
    after = snapshot()
    if after != before:
        raise ValueError(
            "nuScenes cache physical artifacts changed while being loaded; "
            f"before={before}, after={after}"
        )
    expected_meta = {
        "format_version": IC.CACHE_FORMAT_VERSION,
        "version": version,
        "split": split,
        "n_sweeps": n_sweeps,
        "cache_hash": expected_canonical,
    }
    mismatches = {
        key: {"expected": value, "actual": meta.get(key)}
        for key, value in expected_meta.items()
        if meta.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "nuScenes cache metadata does not match the frozen cache contract: "
            f"{mismatches}"
        )
    return info_list, meta, {
        "cache_format_version": IC.CACHE_FORMAT_VERSION,
        "cache_version": version,
        "cache_split": split,
        "cache_n_sweeps": n_sweeps,
        "cache_hash": expected_canonical,
        "cache_pickle_path": after["pickle"]["path"],
        "cache_pickle_bytes": after["pickle"]["bytes"],
        "cache_pickle_sha256": after["pickle"]["sha256"],
        "cache_sidecar_path": after["sidecar"]["path"],
        "cache_sidecar_bytes": after["sidecar"]["bytes"],
        "cache_sidecar_sha256": after["sidecar"]["sha256"],
    }


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


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _phase1_build(config_path: str) -> None:
    """Materialize the exact D_fit/keyframe/five-column Phase-I database."""
    from fl_v3.config import load_resolved_config, verify_physical_data_identities
    from fl_v3.data.nuscenes.s10_binding import load_frozen_split_role

    config = load_resolved_config(config_path)
    raw = config.as_dict()
    if not config.is_phase1 or raw["contract"]["branch"] != "lidar":
        raise ValueError("--phase1-config requires the Phase-I LiDAR recipe")
    if raw["gt_paste"]["database_status"] != "pending_materialization":
        raise ValueError("Phase-I GTDB materialization requires pending status")
    verify_physical_data_identities(config)
    data = raw["data"]
    paste = raw["gt_paste"]
    out_dir = Path(paste["database_root"])
    if out_dir.exists():
        raise FileExistsError(f"Phase-I GTDB output already exists: {out_dir}")
    P.resolve_writable(str(out_dir), data["dataroot"])

    cache_path = Path(data["cache"]["path"])
    info_list, cache_meta, cache_provenance = _load_info_list(
        str(cache_path.parent),
        data["version"],
        "train",
        int(data["cache_capacity_sweeps"]),
        data["cache"]["logical_sha256"],
        data["cache"]["pickle_sha256"],
        data["cache"]["sidecar_sha256"],
    )
    store, blob_provenance = _open_blob_store(
        data["dataroot"],
        data["version"],
        data["zip_manifest"]["path"],
        data["zip_manifest"]["logical_sha256"],
        data["zip_manifest"]["file_sha256"],
    )
    try:
        role = load_frozen_split_role(
            data["split_manifest"]["path"],
            expected_manifest_sha256=data["split_manifest"]["sha256"],
            role="D_fit",
            expected_source_identities={
                "train_cache_logical_sha256": data["cache"]["logical_sha256"],
                "train_cache_pickle_sha256": data["cache"]["pickle_sha256"],
                "train_cache_sidecar_sha256": data["cache"]["sidecar_sha256"],
                "zip_manifest_logical_sha256": data["zip_manifest"]["logical_sha256"],
                "zip_manifest_file_sha256": data["zip_manifest"]["file_sha256"],
            },
        )
        if role.sample_tokens_sha256 != data["roles"]["fit"]["sample_tokens_sha256"]:
            raise ValueError("Phase-I D_fit token identity drift")
        by_token = {str(info["sample_token"]): info for info in info_list}
        if len(by_token) != len(info_list):
            raise ValueError("train cache contains duplicate sample tokens")
        missing = [token for token in role.sample_tokens if token not in by_token]
        if missing:
            raise ValueError(f"D_fit has cache-missing tokens: {missing[:8]}")
        selected_infos = [by_token[token] for token in role.sample_tokens]
        class_names = tuple(raw["taxonomy"]["reference_object_classes"])
        class_to_label = {name: index for index, name in enumerate(class_names)}
        database = {name: [] for name in class_names}
        for info_index, info in enumerate(selected_infos):
            cloud = _load_lidar(info["lidar_rel_path"], blob_store=store)
            if cloud.ndim != 2 or cloud.shape[1] != 5:
                raise ValueError("Phase-I GTDB source must be keyframe x,y,z,intensity,ring")
            boxes = np.asarray(info["gt_boxes"], dtype=np.float32)
            velocities = np.asarray(info["gt_velocity"], dtype=np.float32)
            names = list(info["gt_names"])
            ann_tokens = list(info["gt_ann_tokens"])
            if not (len(boxes) == len(velocities) == len(names) == len(ann_tokens)):
                raise ValueError("GT cache arrays are not row-aligned")
            for box_index, (box, name, ann_token) in enumerate(
                zip(boxes, names, ann_tokens)
            ):
                if name not in database:
                    continue
                points = crop_object_points_center_relative(
                    cloud, box, feature_columns=5
                )
                if points.shape[0] < int(paste["min_points"][name]):
                    continue
                box9 = np.concatenate([box, velocities[box_index]], axis=0).astype(
                    np.float32
                )
                database[name].append({
                    "name": name,
                    "points": points,
                    "box3d_lidar": box9,
                    "num_points_in_gt": int(points.shape[0]),
                    "difficulty": 0,
                    "label": int(class_to_label[name]),
                    "source_sample_token": str(info["sample_token"]),
                    "source_scene_token": str(info["scene_token"]),
                    "source_log_token": str(info["log_token"]),
                    "source_ann_token": str(ann_token),
                    "source_box_index": int(box_index),
                    "point_coordinate_frame": "center_relative_lidar",
                })
            if (info_index + 1) % 2000 == 0:
                print(
                    f"[phase1-gtdb] {info_index + 1}/{len(selected_infos)} "
                    f"counts={ {name: len(database[name]) for name in class_names} }",
                    flush=True,
                )
    finally:
        store.close()

    if any(not database[name] for name in class_names):
        raise ValueError("Phase-I GTDB has an empty frozen class")
    out_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in class_names:
        payload = pickle.dumps(database[name], protocol=pickle.HIGHEST_PROTOCOL)
        path = out_dir / f"{name}.pkl"
        _exclusive_write(path, payload)
        files[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "objects": len(database[name]),
        }
    manifest = {
        "schema": "s10.phase1.gtdb.v1",
        "contract": {
            "plan_sha": raw["contract"]["plan_sha"],
            "request_commit": raw["contract"]["request_commit"],
            "candidate_id": raw["contract"]["candidate_id"],
        },
        "source": {
            "role": "D_fit",
            "sample_count": len(role.sample_tokens),
            "sample_tokens_sha256": role.sample_tokens_sha256,
            "split_manifest_path": role.manifest_path,
            "split_manifest_sha256": role.manifest_sha256,
            "cache_capacity_sweeps": data["cache_capacity_sweeps"],
            "consumed_point_sweeps": data["train_point_sweeps"],
            **cache_provenance,
            **blob_provenance,
        },
        "semantics": {
            "class_order": list(class_names),
            "min_points": dict(paste["min_points"]),
            "filter_by_difficulty": list(paste["filter_by_difficulty"]),
            "build_truncation": None,
            "source_point_dims": list(paste["source_point_dims"]),
            "point_coordinate_frame": "center_relative_lidar",
            "box_fields": ["x", "y", "z", "dx", "dy", "dz", "yaw", "vx", "vy"],
            "reference": "mit-han-lab/bevfusion:create_groundtruth_database",
        },
        "files": files,
        "counts": {name: len(database[name]) for name in class_names},
        "total_objects": sum(len(database[name]) for name in class_names),
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_path = Path(paste["manifest_path"])
    if manifest_path != out_dir / "manifest.json":
        raise ValueError("Phase-I GTDB manifest path leaves database root")
    _exclusive_write(manifest_path, manifest_bytes)
    print(json.dumps({
        "phase1_gtdb": "materialized",
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "counts": manifest["counts"],
        "total_objects": manifest["total_objects"],
        "cache_hash": cache_meta["cache_hash"],
    }, sort_keys=True), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-config", default="")
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-sweeps", type=int, default=10)
    ap.add_argument(
        "--expected-cache-hash",
        default="",
        help="Frozen canonical t1.v2 cache hash for this exact split/depth.",
    )
    ap.add_argument(
        "--expected-cache-file-sha256",
        default="",
        help="Frozen SHA-256 of the exact depth-specific t1.v2 cache pickle.",
    )
    ap.add_argument(
        "--expected-cache-sidecar-sha256",
        default="",
        help="Frozen SHA-256 of the exact depth-specific t1.v2 metadata sidecar.",
    )
    ap.add_argument(
        "--dataroot",
        default=os.environ.get("ARRHENIUS_NUSCENES_DATAROOT") or os.environ.get("NUSCENES_DATAROOT") or "",
        help="nuScenes dataroot; defaults to ARRHENIUS_NUSCENES_DATAROOT/NUSCENES_DATAROOT.",
    )
    ap.add_argument("--out-dir", default="")
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
    ap.add_argument(
        "--sample-manifest",
        default="",
        help="Immutable S10 split_manifest.json. Required for trainval GT-database builds.",
    )
    ap.add_argument(
        "--sample-manifest-sha256",
        default="",
        help="Frozen physical SHA-256 of --sample-manifest.",
    )
    ap.add_argument(
        "--sample-manifest-role",
        default="",
        help="Exact promoted train role (D_low, D_mid, or D_fit).",
    )
    ap.add_argument("--classes", default="trailer,construction_vehicle,bus,truck,bicycle,motorcycle")
    ap.add_argument("--min-points", type=int, default=5)
    ap.add_argument("--max-per-class", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20259)
    args = ap.parse_args()
    if args.phase1_config:
        conflicting = [
            name for name in ("cache_dir", "out_dir", "sample_manifest")
            if getattr(args, name)
        ]
        if conflicting:
            raise SystemExit(
                "--phase1-config is the single source of truth; remove legacy options "
                f"{conflicting}"
            )
        _phase1_build(args.phase1_config)
        return
    for name in (
        "cache_dir", "out_dir", "expected_cache_hash",
        "expected_cache_file_sha256", "expected_cache_sidecar_sha256",
    ):
        if not getattr(args, name):
            raise SystemExit(f"legacy mode requires --{name.replace('_', '-')}")
    if not args.dataroot:
        raise SystemExit(
            "--dataroot is required unless ARRHENIUS_NUSCENES_DATAROOT or NUSCENES_DATAROOT is set; "
            "do not rely on site-specific historical paths."
        )
    P.resolve_writable(args.out_dir, args.dataroot)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    info_list, cache_meta, cache_provenance = _load_info_list(
        args.cache_dir,
        args.version,
        args.split,
        args.n_sweeps,
        args.expected_cache_hash,
        args.expected_cache_file_sha256,
        args.expected_cache_sidecar_sha256,
    )
    blob_store, blob_provenance = _open_blob_store(
        args.dataroot,
        args.version,
        args.zip_manifest,
        args.expected_manifest_hash,
        args.expected_manifest_file_sha256,
    )
    role_provenance = {}
    try:
        if args.version == "v1.0-trainval":
            if not (
                args.sample_manifest
                and args.sample_manifest_sha256
                and args.sample_manifest_role in {"D_low", "D_mid", "D_fit"}
            ):
                raise ValueError(
                    "trainval GT-database builds require --sample-manifest, "
                    "--sample-manifest-sha256 and --sample-manifest-role in "
                    "{D_low,D_mid,D_fit}"
                )
            from fl_v3.eval.subset_detection_eval import load_manifest_role

            role = load_manifest_role(
                args.sample_manifest,
                args.sample_manifest_role,
                expected_manifest_sha256=args.sample_manifest_sha256,
                expected_parent_version=args.version,
                expected_parent_split="train",
                expected_source_identities={
                    "train_cache_logical_sha256": cache_provenance["cache_hash"],
                    "train_cache_pickle_sha256": cache_provenance["cache_pickle_sha256"],
                    "train_cache_sidecar_sha256": cache_provenance["cache_sidecar_sha256"],
                    "zip_manifest_logical_sha256": blob_provenance["zip_manifest_hash"],
                    "zip_manifest_file_sha256": blob_provenance["zip_manifest_file_sha256"],
                },
            )
            by_token = {str(info["sample_token"]): info for info in info_list}
            missing = sorted(set(role.sample_tokens) - set(by_token))
            if missing:
                raise ValueError(f"sample manifest references cache-missing tokens: {missing[:8]}")
            info_list = [by_token[token] for token in role.sample_tokens]
            role_provenance = {
                "sample_manifest_path": role.path,
                "sample_manifest_sha256": role.sha256,
                "sample_manifest_role": role.role,
                "sample_manifest_samples": len(role.sample_tokens),
            }
    except Exception:
        blob_store.close()
        raise
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
        **cache_provenance,
        **blob_provenance,
        **role_provenance,
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
