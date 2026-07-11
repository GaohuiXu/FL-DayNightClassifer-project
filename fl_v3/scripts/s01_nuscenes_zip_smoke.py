#!/usr/bin/env python3
"""Bounded real-data S01 smoke over one shared trainval ZIP archive.

This is intentionally not the S01 full gate: it builds a manifest for one named
archive, selects a few metadata-complete keyframes stored in that archive, and
checks real six-camera/key-LiDAR/10-sweep decode plus persistent two-worker
repeatability.  It does not claim full train/val member coverage or throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.dataset import (
    NuScenesMultimodalDataset,
    make_loader,
    sample_image_sha256,
)
from fl_v3.data.nuscenes.zip_backend import (
    NuScenesBlobStore,
    canonical_member_path,
    manifest_summary,
)


class _DebugStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["_zip_debug_state"] = self.dataset.blob_store.debug_state()
        return sample


def _sample_refs(nusc, sample: dict, n_sweeps: int) -> tuple[list[str], int]:
    refs = [
        canonical_member_path(
            nusc.get("sample_data", sample["data"][channel])["filename"]
        )
        for channel in P.CAMERA_CHANNELS
    ]
    lidar = nusc.get("sample_data", sample["data"][P.LIDAR_CHANNEL])
    refs.append(canonical_member_path(lidar["filename"]))
    depth = 0
    current = lidar
    while depth < n_sweeps - 1 and current["prev"]:
        current = nusc.get("sample_data", current["prev"])
        refs.append(canonical_member_path(current["filename"]))
        depth += 1
    return refs, depth


def _sample_digest(sample: dict) -> dict:
    images = sample["images"]
    points = sample["lidar_points"]
    if tuple(images.shape) != (6, 3, 900, 1600) or images.dtype != torch.uint8:
        raise AssertionError(f"unexpected image schema: shape={tuple(images.shape)} dtype={images.dtype}")
    if points.ndim != 2 or points.shape[1] != 6 or points.dtype != torch.float32:
        raise AssertionError(f"unexpected 10-sweep LiDAR schema: {tuple(points.shape)} {points.dtype}")
    if not bool(torch.isfinite(points).all()):
        raise AssertionError("non-finite LiDAR values")
    return {
        "sample_token": sample["sample_token"],
        "image_sha256": sample_image_sha256(sample),
        "lidar_sha256": hashlib.sha256(
            np.ascontiguousarray(points.numpy()).tobytes()
        ).hexdigest(),
        "lidar_points": int(points.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--archive", default="trainval01_blobs.zip")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=4000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.n_sweeps, args.num_samples, args.max_candidates, args.num_workers) < 1:
        raise SystemExit("n-sweeps, num-samples, max-candidates, and num-workers must be >= 1")

    dataroot = args.dataroot or P.get_dataroot()
    manifest = args.manifest or P.get_zip_manifest(required=True)
    P.resolve_writable(args.output, dataroot)
    dataset_report = P.verify_dataset(args.version, dataroot)
    summary = manifest_summary(manifest)
    if summary["archive_names"] != (args.archive,):
        raise SystemExit(
            f"bounded smoke requires a one-archive manifest for {args.archive!r}; "
            f"manifest has {summary['archive_names']!r}"
        )

    nusc = P.create_nuscenes(args.version, dataroot, verbose=False)
    membership = NuScenesBlobStore(dataroot, manifest_path=manifest)
    full_history = []
    partial_history = []
    reference_counts = {}
    examined = 0
    for sample in nusc.sample:
        if examined >= args.max_candidates:
            break
        examined += 1
        refs, depth = _sample_refs(nusc, sample, args.n_sweeps)
        if all(membership.contains(path) for path in refs):
            target = full_history if depth == args.n_sweeps - 1 else partial_history
            target.append(sample["token"])
            reference_counts[sample["token"]] = {"members": len(refs), "previous_sweeps": depth}
            if len(full_history) >= args.num_samples:
                break
    tokens = full_history[: args.num_samples]
    if len(tokens) < args.num_samples:
        tokens.extend(partial_history[: args.num_samples - len(tokens)])
    if len(tokens) != args.num_samples or not full_history:
        raise SystemExit(
            f"found only {len(tokens)} complete samples ({len(full_history)} with full history) "
            f"in first {examined} metadata candidates for {args.archive}"
        )

    infos = IC.build_info_list(nusc, tokens, dataroot, n_sweeps=args.n_sweeps)
    dataset = NuScenesMultimodalDataset(
        infos,
        dataroot,
        sample_tokens=tokens,
        n_sweeps=args.n_sweeps,
        zip_manifest=manifest,
    )

    zero_worker = make_loader(
        dataset, batch_size=1, shuffle=False, num_workers=0, seed=42
    )
    zero_worker_digests = [_sample_digest(batch[0]) for batch in zero_worker]
    parent_state_after_read = dataset.blob_store.debug_state()

    debug_dataset = _DebugStateDataset(dataset)
    worker_loader = make_loader(
        debug_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        seed=42,
    )
    worker_epochs = []
    lifecycle_epochs = []
    try:
        for _epoch in range(2):
            digests = []
            lifecycle = {}
            for batch in worker_loader:
                sample = batch[0]
                state = sample.pop("_zip_debug_state")
                lifecycle[str(state["owner_pid"])] = state
                digests.append(_sample_digest(sample))
            worker_epochs.append(digests)
            lifecycle_epochs.append(lifecycle)
    finally:
        del worker_loader
        dataset.close()
        membership.close()

    if not (zero_worker_digests == worker_epochs[0] == worker_epochs[1]):
        raise SystemExit("decoded sample digests differ across 0-worker/2-worker/repeated epochs")
    if len(lifecycle_epochs[0]) != args.num_workers:
        raise SystemExit(
            f"expected {args.num_workers} persistent workers, observed {len(lifecycle_epochs[0])}"
        )
    if set(lifecycle_epochs[0]) != set(lifecycle_epochs[1]):
        raise SystemExit("worker PIDs changed across persistent epochs")
    for pid in lifecycle_epochs[0]:
        first = lifecycle_epochs[0][pid]
        second = lifecycle_epochs[1][pid]
        if first["reopen_count"] != second["reopen_count"]:
            raise SystemExit(f"archive handle reopened unexpectedly between epochs in worker {pid}")
        if second["read_count"] <= first["read_count"]:
            raise SystemExit(f"worker {pid} did not retain/increment read state across epochs")

    report = {
        "schema": "s01.nuscenes-zip-bounded-smoke.v1",
        "scope": "one archive; not full train/val coverage or throughput evidence",
        "version": args.version,
        "dataroot": os.path.abspath(dataroot),
        "dataset_report": dataset_report,
        "archive": args.archive,
        "manifest_path": os.path.abspath(manifest),
        "manifest_hash": summary["manifest_hash"],
        "manifest_members": summary["member_count"],
        "n_sweeps_total_including_keyframe": args.n_sweeps,
        "metadata_candidates_examined": examined,
        "selected_tokens": tokens,
        "selected_reference_counts": {token: reference_counts[token] for token in tokens},
        "zero_worker_digests": zero_worker_digests,
        "parent_state_after_read": parent_state_after_read,
        "worker_lifecycle_epochs": lifecycle_epochs,
        "deterministic_across_worker_counts_and_epochs": True,
    }
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
