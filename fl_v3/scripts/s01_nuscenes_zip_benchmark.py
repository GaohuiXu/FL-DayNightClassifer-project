#!/usr/bin/env python3
"""Measure full-data ZIP loader wait/throughput without running a model.

The timed quantity is wall time blocked in ``next(DataLoader)`` after a declared
warm-up.  It is loader-level data wait, not a percentage of an unmeasured GPU
step; S07 must combine it with model-step timing.  Running this against shared
trainval is material compute and requires an approved S01 RUN_REQUEST.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import socket
import sys
import time

import PIL
import torch

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset, make_loader
from fl_v3.data.nuscenes.zip_backend import manifest_summary


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _batch_digest(batch: list[dict]) -> str:
    digest = hashlib.sha256()
    for sample in batch:
        digest.update(str(sample["sample_token"]).encode("ascii"))
        digest.update(sample["images"].numpy().tobytes())
        digest.update(sample["lidar_points"].numpy().tobytes())
    return digest.hexdigest()


def _next_timed(iterator) -> tuple[object, float]:
    start = time.perf_counter()
    batch = next(iterator)
    return batch, (time.perf_counter() - start) * 1000.0


def _run_one(
    info_list: list[dict],
    dataroot: str,
    manifest: str,
    tokens: list[str],
    *,
    workers: int,
    batch_size: int,
    n_sweeps: int,
    determinism_batches: int,
    warmup_batches: int,
    measured_batches: int,
    repeats: int,
) -> dict:
    dataset = NuScenesMultimodalDataset(
        info_list,
        dataroot,
        sample_tokens=tokens,
        n_sweeps=n_sweeps,
        zip_manifest=manifest,
    )
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        seed=42,
    )
    repeat_reports = []
    determinism_hashes = []
    try:
        for repeat in range(repeats):
            iterator_start = time.perf_counter()
            iterator = iter(loader)
            iterator_create_ms = (time.perf_counter() - iterator_start) * 1000.0
            determinism_digest = hashlib.sha256()
            determinism_waits = []
            for _ in range(determinism_batches):
                batch, wait_ms = _next_timed(iterator)
                determinism_waits.append(wait_ms)
                determinism_digest.update(bytes.fromhex(_batch_digest(batch)))
            determinism_hash = determinism_digest.hexdigest()
            determinism_hashes.append(determinism_hash)
            for _ in range(warmup_batches):
                next(iterator)

            waits_ms: list[float] = []
            sample_count = 0
            wall_start = time.perf_counter()
            for _ in range(measured_batches):
                batch, wait_ms = _next_timed(iterator)
                waits_ms.append(wait_ms)
                sample_count += len(batch)
            wall_seconds = time.perf_counter() - wall_start
            repeat_reports.append(
                {
                    "repeat": repeat,
                    "cache_state_label": "cold-worker-start" if repeat == 0 else "persistent-worker-warm",
                    "iterator_create_ms": iterator_create_ms,
                    "determinism_batches": determinism_batches,
                    "determinism_batches_sha256": determinism_hash,
                    "determinism_first_batch_wait_ms": determinism_waits[0],
                    "determinism_batch_wait_ms_p50": _percentile(determinism_waits, 0.50),
                    "determinism_batch_wait_ms_p95": _percentile(determinism_waits, 0.95),
                    "warmup_batches_after_determinism_audit": warmup_batches,
                    "measured_batches": measured_batches,
                    "measured_samples": sample_count,
                    "measured_wall_seconds": wall_seconds,
                    "samples_per_second": sample_count / wall_seconds,
                    "batch_wait_ms_p50": _percentile(waits_ms, 0.50),
                    "batch_wait_ms_p95": _percentile(waits_ms, 0.95),
                    "batch_wait_ms_min": min(waits_ms),
                    "batch_wait_ms_max": max(waits_ms),
                }
            )
            del iterator
    finally:
        del loader
        dataset.close()
        gc.collect()
    return {
        "num_workers": workers,
        "batch_size": batch_size,
        "n_sweeps_total_including_keyframe": n_sweeps,
        "determinism_batches": determinism_batches,
        "determinism_sha256": determinism_hashes[0],
        "determinism_hashes_identical": len(set(determinism_hashes)) == 1,
        "repeats": repeat_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-sweeps", type=int, default=10)
    parser.add_argument("--workers", default="0,2,4,8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--determinism-batches", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=16)
    parser.add_argument("--measured-batches", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(
        args.n_sweeps,
        args.batch_size,
        args.determinism_batches,
        args.measured_batches,
        args.repeats,
    ) < 1:
        raise SystemExit(
            "n-sweeps, batch-size, determinism-batches, measured-batches, and repeats must be >= 1"
        )
    if args.warmup_batches < 0:
        raise SystemExit("warmup-batches must be >= 0")
    workers = [int(value) for value in args.workers.split(",")]
    if not workers or any(value < 0 for value in workers) or len(set(workers)) != len(workers):
        raise SystemExit("--workers must be a duplicate-free comma-separated list of nonnegative ints")

    dataroot = args.dataroot or P.get_dataroot()
    manifest = args.manifest or P.get_zip_manifest(required=True)
    P.resolve_writable(args.output, dataroot)
    dataset_report = P.verify_dataset(args.version, dataroot)
    if dataset_report["blob_backend"] != "zip":
        raise SystemExit(f"benchmark requires ZIP backend, got {dataset_report['blob_backend']!r}")
    info_list, cache_meta = IC.load_cache(
        args.cache_dir, args.version, args.split, n_sweeps=args.n_sweeps
    )
    needed_batches = args.determinism_batches + args.warmup_batches + args.measured_batches
    needed_samples = needed_batches * args.batch_size
    tokens = sorted(str(info["sample_token"]) for info in info_list)[:needed_samples]
    if len(tokens) != needed_samples:
        raise SystemExit(
            f"split has only {len(tokens)} samples; benchmark needs {needed_samples} for one pass"
        )

    summary = manifest_summary(manifest)
    report = {
        "schema": "s01.nuscenes-zip-loader-profile.v1",
        "measurement_definition": (
            "batch_wait_ms is wall time blocked in next(DataLoader) after warm-up; "
            "no model step was measured, so no GPU-step data-wait percentage is claimed"
        ),
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "pillow": PIL.__version__,
        "dataroot": os.path.abspath(dataroot),
        "manifest_path": os.path.abspath(manifest),
        "manifest_hash": summary["manifest_hash"],
        "cache_dir": os.path.abspath(args.cache_dir),
        "cache_hash": cache_meta.get("cache_hash", ""),
        "version": args.version,
        "split": args.split,
        "token_count": len(tokens),
        "token_sha256": hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest(),
        "profiles": [],
    }
    for num_workers in workers:
        profile = _run_one(
            info_list,
            dataroot,
            manifest,
            tokens,
            workers=num_workers,
            batch_size=args.batch_size,
            n_sweeps=args.n_sweeps,
            determinism_batches=args.determinism_batches,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
            repeats=args.repeats,
        )
        report["profiles"].append(profile)
        print(json.dumps(profile, sort_keys=True), flush=True)
    report["determinism_hash_identical_across_worker_counts"] = (
        len({profile["determinism_sha256"] for profile in report["profiles"]}) == 1
    )
    report["parent_max_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(canonical).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if not all(profile["determinism_hashes_identical"] for profile in report["profiles"]):
        raise SystemExit("determinism FAILED: decoded audit batches changed across repeats")
    if not report["determinism_hash_identical_across_worker_counts"]:
        raise SystemExit("determinism FAILED: decoded audit batches changed with worker count")


if __name__ == "__main__":
    main()
