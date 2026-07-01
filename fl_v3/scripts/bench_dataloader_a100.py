"""E2 (overcommit diag) — dataloader-only throughput, NO model, NO GPU compute.

Isolates the data pipeline + filesystem I/O from GPU/Ray. Times pure iteration of ONE real client's
trainloader (6-camera + LiDAR off Mimer, the same loader the FL clients use). Two uses:
  * single loader, sweep --num-workers  -> worker-bound vs disk-bound knee;
  * K concurrent copies on DISTINCT client-ids (shards) -> shared-filesystem I/O contention
    (the real 16-actor read load that 1-GPU diagnostics under-sample).

Determinism-neutral instrumentation only; never feeds a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "fl_v3/src")

from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import derive_seed, enforce_determinism, seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=-1, help="-1 = config default")
    ap.add_argument("--batches", type=int, default=40, help="timed batches after warmup")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="dl")
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        rc = json.load(f)
    if a.cache_dir:
        rc["nuscenes-cache-dir"] = a.cache_dir
    if a.num_workers >= 0:
        rc["num-workers"] = a.num_workers

    enforce_determinism(strict=True, precision="fp32")
    seed_everything(derive_seed(int(rc.get("seed", 42)), a.client_id, 1))

    task = get_task("nuscenes_detection")
    cdata = task.client_data(a.client_id, rc)
    loader = cdata.trainloader
    nw = int(rc.get("num-workers", 0))
    bs = int(rc.get("batch-size", 16))

    target = a.warmup + a.batches
    done = 0
    t_start = None
    while done < target:
        for _batch in loader:                 # touch the batch (it is materialized by iteration)
            if done == a.warmup:
                t_start = time.perf_counter()  # start the clock after the pipeline is primed
            done += 1
            if done >= target:
                break
        if done < target:
            # shard exhausted before target -> next epoch (persistent workers stay warm)
            continue
    elapsed = (time.perf_counter() - t_start) if t_start else 0.0
    bps = (a.batches / elapsed) if elapsed > 0 else 0.0
    res = {
        "label": a.label, "client_id": a.client_id, "num_workers": nw, "batch_size": bs,
        "timed_batches": a.batches, "elapsed_s": round(elapsed, 3),
        "batches_per_s": round(bps, 3), "samples_per_s": round(bps * bs, 2),
    }
    print(f"[bench-dl] {a.label} cid={a.client_id} nw={nw} -> "
          f"{bps:.2f} batch/s  {bps*bs:.1f} samples/s  ({elapsed:.1f}s/{a.batches}b)", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
