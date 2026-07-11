#!/usr/bin/env python3
"""Emit exact train/val ZIP member coverage for six cameras and LiDAR sweeps.

``--n-sweeps 10`` means one keyframe plus at most nine available previous
LIDAR_TOP records.  Scene starts legitimately have fewer history records; only
metadata-referenced members count as required.  A shared-full-data invocation is
an exhaustive scan and requires an approved S01 RUN_REQUEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import (
    NuScenesBlobStore,
    TRAINVAL_ARCHIVE_NAMES,
    canonical_member_path,
    manifest_archive_sentinels,
    manifest_member_counts,
    manifest_summary,
)


_OFFICIAL_SPLIT_COUNTS = {"train": 28130, "val": 6019}
_OFFICIAL_TRAINVAL_SAMPLES = sum(_OFFICIAL_SPLIT_COUNTS.values())


def _sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        raw = value.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _compact_coverage(coverage: dict) -> dict:
    missing = list(coverage.pop("missing_unique_members"))
    coverage["missing_unique_count"] = len(missing)
    coverage["missing_unique_sha256"] = _sha256_strings(missing)
    coverage["missing_unique_examples"] = missing[:100]
    return coverage


def _member(sample_data: dict) -> str:
    return canonical_member_path(sample_data["filename"])


def _split_references(nusc, split: str, n_sweeps: int) -> dict:
    tokens = IC.split_sample_tokens(nusc, split)
    cameras = {channel: [] for channel in P.CAMERA_CHANNELS}
    key_lidar: list[str] = []
    prev_lidar: list[str] = []
    sweep_depths: Counter[int] = Counter()
    prefix_violations: list[str] = []

    for token in tokens:
        sample = nusc.get("sample", token)
        for channel in P.CAMERA_CHANNELS:
            rel = _member(nusc.get("sample_data", sample["data"][channel]))
            cameras[channel].append(rel)
            if not rel.startswith(f"samples/{channel}/"):
                prefix_violations.append(rel)

        lidar = nusc.get("sample_data", sample["data"][P.LIDAR_CHANNEL])
        key_rel = _member(lidar)
        key_lidar.append(key_rel)
        if not key_rel.startswith("samples/LIDAR_TOP/"):
            prefix_violations.append(key_rel)

        depth = 0
        current = lidar
        while depth < n_sweeps - 1 and current["prev"]:
            current = nusc.get("sample_data", current["prev"])
            rel = _member(current)
            prev_lidar.append(rel)
            if not rel.startswith(("sweeps/LIDAR_TOP/", "samples/LIDAR_TOP/")):
                prefix_violations.append(rel)
            depth += 1
        sweep_depths[depth] += 1

    return {
        "sample_tokens": tokens,
        "cameras": cameras,
        "key_lidar": key_lidar,
        "prev_lidar": prev_lidar,
        "sweep_depth_histogram": dict(sorted(sweep_depths.items())),
        "prefix_violations": sorted(set(prefix_violations)),
    }


def _cache_references(info_list: list[dict], n_sweeps: int) -> dict:
    cameras = {channel: [] for channel in P.CAMERA_CHANNELS}
    key_lidar = []
    prev_lidar = []
    tokens = []
    for info in info_list:
        tokens.append(str(info["sample_token"]))
        if tuple(info["cam_order"]) != tuple(P.CAMERA_CHANNELS):
            raise ValueError(f"cache camera order drift for sample {info['sample_token']}")
        for index, channel in enumerate(P.CAMERA_CHANNELS):
            cameras[channel].append(canonical_member_path(info["cam_rel_paths"][index]))
        key_lidar.append(canonical_member_path(info["lidar_rel_path"]))
        prev_lidar.extend(
            canonical_member_path(sweep["rel_path"])
            for sweep in info.get("lidar_sweeps", [])[: n_sweeps - 1]
        )
    return {
        "sample_tokens": tokens,
        "cameras": cameras,
        "key_lidar": key_lidar,
        "prev_lidar": prev_lidar,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--n-sweeps",
        type=int,
        default=10,
        help="Total LiDAR frames requested per sample: keyframe + up to n-1 previous records.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.n_sweeps < 1:
        raise SystemExit("--n-sweeps must be >= 1")

    dataroot = args.dataroot or P.get_dataroot()
    manifest = args.manifest or P.get_zip_manifest(required=True)
    P.resolve_writable(args.output, dataroot)
    dataset_report = P.verify_dataset(args.version, dataroot)
    if dataset_report["blob_backend"] != "zip":
        raise SystemExit(f"coverage audit requires ZIP backend, got {dataset_report['blob_backend']!r}")
    summary = manifest_summary(manifest)
    gate_errors = []
    if summary["archive_names"] != TRAINVAL_ARCHIVE_NAMES:
        gate_errors.append(
            f"full audit requires trainval01..10 manifest, got {summary['archive_names']!r}"
        )
    nusc = P.create_nuscenes(args.version, dataroot, verbose=False)
    if args.version == "v1.0-trainval" and len(nusc.sample) != _OFFICIAL_TRAINVAL_SAMPLES:
        gate_errors.append(
            f"metadata sample count {len(nusc.sample)} != official {_OFFICIAL_TRAINVAL_SAMPLES}"
        )

    report = {
        "schema": "s01.nuscenes-zip-coverage.v1",
        "version": args.version,
        "dataroot": os.path.abspath(dataroot),
        "manifest_path": os.path.abspath(manifest),
        "manifest_hash": summary["manifest_hash"],
        "manifest_archive_count": summary["archive_count"],
        "manifest_member_count": summary["member_count"],
        "cache_dir": os.path.abspath(args.cache_dir),
        "metadata_total_samples": len(nusc.sample),
        "n_sweeps_total_including_keyframe": args.n_sweeps,
        "splits": {},
    }
    all_references: list[str] = []
    split_token_sets = {}
    for split in args.splits:
        refs = _split_references(nusc, split, args.n_sweeps)
        split_token_sets[split] = set(refs["sample_tokens"])
        expected_samples = _OFFICIAL_SPLIT_COUNTS.get(split) if args.version == "v1.0-trainval" else None
        if expected_samples is not None and len(refs["sample_tokens"]) != expected_samples:
            gate_errors.append(
                f"split {split} sample count {len(refs['sample_tokens'])} != official {expected_samples}"
            )
        if not refs["sample_tokens"]:
            gate_errors.append(f"split {split} is empty")

        cache_info, cache_meta = IC.load_cache(args.cache_dir, args.version, split)
        cache_refs = _cache_references(cache_info, args.n_sweeps)
        if cache_refs != {
            "sample_tokens": refs["sample_tokens"],
            "cameras": refs["cameras"],
            "key_lidar": refs["key_lidar"],
            "prev_lidar": refs["prev_lidar"],
        }:
            gate_errors.append(f"split {split} info-cache paths differ from metadata traversal")
        actual_cache_hash = IC.canonical_hash(cache_info)
        if cache_meta.get("cache_hash") != actual_cache_hash:
            gate_errors.append(f"split {split} cache hash sidecar does not match cache content")

        camera_reports = {}
        for channel in P.CAMERA_CHANNELS:
            paths = refs["cameras"][channel]
            camera_reports[channel] = _compact_coverage(manifest_member_counts(manifest, paths))
            channel_report = camera_reports[channel]
            if channel_report["references"] != len(refs["sample_tokens"]):
                gate_errors.append(f"split {split}/{channel} reference count != samples")
            if channel_report["unique_members"] != len(refs["sample_tokens"]):
                gate_errors.append(f"split {split}/{channel} has duplicate member paths")
            all_references.extend(paths)
        key_report = _compact_coverage(manifest_member_counts(manifest, refs["key_lidar"]))
        sweep_report = _compact_coverage(manifest_member_counts(manifest, refs["prev_lidar"]))
        if key_report["references"] != len(refs["sample_tokens"]):
            gate_errors.append(f"split {split}/LIDAR_TOP key reference count != samples")
        if key_report["unique_members"] != len(refs["sample_tokens"]):
            gate_errors.append(f"split {split}/LIDAR_TOP key paths are not unique")
        all_references.extend(refs["key_lidar"])
        all_references.extend(refs["prev_lidar"])
        report["splits"][split] = {
            "samples": len(refs["sample_tokens"]),
            "sample_token_sha256": _sha256_strings(refs["sample_tokens"]),
            "cameras": camera_reports,
            "lidar_keyframes": key_report,
            "lidar_previous_sweeps": sweep_report,
            "sweep_depth_histogram": refs["sweep_depth_histogram"],
            "unexpected_member_prefixes": refs["prefix_violations"][:100],
            "unexpected_member_prefix_count": len(refs["prefix_violations"]),
            "cache": {
                "n_samples": len(cache_info),
                "cache_hash": actual_cache_hash,
                "meta": cache_meta,
                "paths_identical_to_metadata": cache_refs
                == {
                    "sample_tokens": refs["sample_tokens"],
                    "cameras": refs["cameras"],
                    "key_lidar": refs["key_lidar"],
                    "prev_lidar": refs["prev_lidar"],
                },
            },
        }
        for category, category_report in [
            *( (f"camera/{channel}", camera_reports[channel]) for channel in P.CAMERA_CHANNELS ),
            ("lidar_keyframes", key_report),
            ("lidar_previous_sweeps", sweep_report),
        ]:
            if category_report["missing_unique_count"]:
                gate_errors.append(
                    f"split {split}/{category} missing {category_report['missing_unique_count']} members"
                )
        if refs["prefix_violations"]:
            gate_errors.append(
                f"split {split} has {len(refs['prefix_violations'])} unexpected member prefixes"
            )

    if "train" in split_token_sets and "val" in split_token_sets:
        overlap = split_token_sets["train"] & split_token_sets["val"]
        if overlap:
            gate_errors.append(f"train/val sample-token overlap: {len(overlap)}")
        if len(split_token_sets["train"] | split_token_sets["val"]) != _OFFICIAL_TRAINVAL_SAMPLES:
            gate_errors.append("train/val union does not cover all official trainval samples")
    report["all_references"] = _compact_coverage(
        manifest_member_counts(manifest, all_references)
    )
    if report["all_references"]["missing_unique_count"]:
        gate_errors.append(
            f"all-references missing {report['all_references']['missing_unique_count']} unique members"
        )

    # Read one deterministic payload per archive through the actual pread+CRC
    # path. This is a bounded integrity sentinel, not an all-payload CRC scan.
    archive_sentinels = manifest_archive_sentinels(manifest)
    store = NuScenesBlobStore(dataroot, manifest_path=manifest)
    try:
        report["archive_payload_sentinels"] = {
            archive: {
                "member": member,
                "size_bytes": len(payload := store.read_bytes(member)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for archive, member in archive_sentinels.items()
        }
    finally:
        store.close()
    if set(report["archive_payload_sentinels"]) != set(TRAINVAL_ARCHIVE_NAMES):
        gate_errors.append("not every trainval archive has a readable payload sentinel")
    report["gate_errors"] = gate_errors
    report["gate_pass"] = not gate_errors
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if gate_errors:
        raise SystemExit(f"ZIP coverage FAILED with {len(gate_errors)} gate error(s)")


if __name__ == "__main__":
    main()
