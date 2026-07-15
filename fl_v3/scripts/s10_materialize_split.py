#!/usr/bin/env python
"""Materialize the exact S10 train-only ownership split from accepted S09 inputs."""
from __future__ import annotations

import argparse
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.insert(0, SCRIPT_DIR)

from build_gt_database import _load_info_list, _open_blob_store
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.internal_split import materialize_split_artifacts
from fl_v3.eval.subset_detection_eval import bound_detection_config, write_strict_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--zip-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--train-cache-hash", required=True)
    parser.add_argument("--train-cache-file-sha256", required=True)
    parser.add_argument("--train-cache-sidecar-sha256", required=True)
    parser.add_argument("--val-cache-hash", required=True)
    parser.add_argument("--val-cache-file-sha256", required=True)
    parser.add_argument("--val-cache-sidecar-sha256", required=True)
    parser.add_argument("--zip-manifest-hash", required=True)
    parser.add_argument("--zip-manifest-file-sha256", required=True)
    return parser.parse_args()


def run(args) -> dict:
    if args.version != "v1.0-trainval" or args.n_sweeps != 10:
        raise ValueError("STOP-A accepts only v1.0-trainval with n_sweeps=10")
    P.resolve_writable(args.output_dir, args.dataroot)
    train_info, _train_meta, train_identity = _load_info_list(
        args.cache_dir,
        args.version,
        "train",
        args.n_sweeps,
        args.train_cache_hash,
        args.train_cache_file_sha256,
        args.train_cache_sidecar_sha256,
    )
    val_info, _val_meta, val_identity = _load_info_list(
        args.cache_dir,
        args.version,
        "val",
        args.n_sweeps,
        args.val_cache_hash,
        args.val_cache_file_sha256,
        args.val_cache_sidecar_sha256,
    )
    store, zip_identity = _open_blob_store(
        args.dataroot,
        args.version,
        args.zip_manifest,
        args.zip_manifest_hash,
        args.zip_manifest_file_sha256,
    )
    store.close()
    _cfg, config_path = bound_detection_config()
    source_identities = {
        "train_cache_logical_sha256": train_identity["cache_hash"],
        "train_cache_pickle_sha256": train_identity["cache_pickle_sha256"],
        "train_cache_sidecar_sha256": train_identity["cache_sidecar_sha256"],
        "val_cache_logical_sha256": val_identity["cache_hash"],
        "val_cache_pickle_sha256": val_identity["cache_pickle_sha256"],
        "val_cache_sidecar_sha256": val_identity["cache_sidecar_sha256"],
        "zip_manifest_logical_sha256": zip_identity["zip_manifest_hash"],
        "zip_manifest_file_sha256": zip_identity["zip_manifest_file_sha256"],
        "detection_config_sha256": (
            "217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b"
        ),
        "version": args.version,
        "n_sweeps": args.n_sweeps,
    }
    nusc = P.create_nuscenes(args.version, args.dataroot, verbose=False)
    result = materialize_split_artifacts(
        nusc,
        train_info,
        val_info,
        args.output_dir,
        source_identities,
    )
    result["inputs"] = {
        "train": train_identity,
        "val": val_identity,
        "zip": zip_identity,
        "detection_config_path": config_path,
    }
    write_strict_json(os.path.join(args.output_dir, "materialization_summary.json"), result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
