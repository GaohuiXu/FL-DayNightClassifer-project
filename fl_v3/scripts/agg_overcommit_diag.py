"""Aggregate the overcommit-diagnostic outputs (E1/E2/E3) into teardown tables.

Scans an OUT dir for:
  * profiler JSONs  <label>_g<gpu>_p<j>_cid<cid>.json   (E1 fixed-batch ceiling; E3 realistic)
  * bench JSONs     dl_<...>.json                        (E2 dataloader-only)
  * util CSVs       util_<label>.csv  (rows: ts,gpu_idx,util,mem ; host rows ts,H,,used_mb)
and prints, per phase: per-proc step time, aggregate throughput, scaling vs K=1, GPU util, host RAM.
Pure post-processing — safe to re-run on a finished OUT dir.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "fl_v3/collab/speedup/overcommit_diag"

PROF_RE = re.compile(r"^(?P<label>.+?)_g(?P<gpu>\d+)_p(?P<p>\d+)_cid(?P<cid>\d+)\.json$")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prof_metrics(d):
    """Pull (mean_step_ms, dataloader_wait_ms, peak_mem_mb) from a profiler JSON (mode fp32)."""
    reg = d.get("regimes", {})
    r = reg.get("fp32") or next(iter(reg.values()), {})
    step = r.get("mean_step_ms", 0.0)
    dlw = (r.get("per_stage", {}).get("dataloader_wait", {}) or {}).get("mean_ms", 0.0)
    mem = r.get("peak_gpu_mem_mb", 0.0)
    return step, dlw, mem


def _util_by_gpu(csv_path):
    """mean util% per gpu idx + peak host-used MB, from a phase util CSV."""
    if not os.path.exists(csv_path):
        return {}, 0.0
    util = {}
    host_peak = 0.0
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4 or parts[0] == "ts_s":
                continue
            ts, gidx, u, mem = parts[0], parts[1], parts[2], parts[3]
            if gidx == "H":
                try:
                    host_peak = max(host_peak, float(mem))
                except ValueError:
                    pass
                continue
            try:
                util.setdefault(gidx, []).append(float(u))
            except ValueError:
                pass
    return {g: statistics.mean(v) for g, v in util.items()}, host_peak


def collect_profilers(label):
    """{gpu_idx: [ (step_ms, dlw_ms, mem) ... ]} for one phase label."""
    by_gpu = {}
    for path in glob.glob(os.path.join(OUT, f"{label}_g*_p*_cid*.json")):
        m = PROF_RE.match(os.path.basename(path))
        if not m:
            continue
        try:
            step, dlw, mem = _prof_metrics(_load(path))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] unreadable {path}: {e}")
            continue
        by_gpu.setdefault(m["gpu"], []).append((step, dlw, mem))
    return by_gpu


def hr(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


# ---- E1: ceiling sweep (one K per GPU) ----------------------------------------
e1 = collect_profilers("e1")
if e1:
    hr("E1 — GPU-sharing CEILING (fixed batch, dataloader bypassed)")
    util, _ = _util_by_gpu(os.path.join(OUT, "util_e1.csv"))
    print(f"{'GPU':>3} {'K':>3} {'per-proc step ms':>17} {'agg steps/s':>12} "
          f"{'scale vs K=1':>13} {'gpu_util%':>10}")
    base = None
    for g in sorted(e1, key=int):
        rows = e1[g]
        k = len(rows)
        steps = [r[0] for r in rows]
        pstep = statistics.mean(steps) if steps else 0.0
        agg = sum(1000.0 / s for s in steps if s > 0)   # steps/s summed across the K procs
        if k == 1:
            base = agg
        scale = (agg / base) if base else float("nan")
        print(f"{g:>3} {k:>3} {pstep:>17.1f} {agg:>12.2f} {scale:>12.2f}x {util.get(g, 0):>9.0f}%")
    print("Read: scale vs K=1 = realized overcommit throughput gain with data+Ray removed (the ceiling).")

# ---- E2: dataloader-only ------------------------------------------------------
dl = sorted(glob.glob(os.path.join(OUT, "dl_*.json")))
if dl:
    hr("E2 — DATALOADER-ONLY throughput (no model)")
    print(f"{'label':>20} {'cid':>4} {'nw':>3} {'batch/s':>9} {'samples/s':>11}")
    singles = {}
    concurrent = []
    for path in dl:
        d = _load(path)
        lab = d.get("label", os.path.basename(path))
        print(f"{lab:>20} {d.get('client_id',0):>4} {d.get('num_workers',0):>3} "
              f"{d.get('batches_per_s',0):>9.2f} {d.get('samples_per_s',0):>11.1f}")
        if lab.startswith("dlsingle"):
            singles[d.get("num_workers")] = d.get("samples_per_s", 0)
        elif lab.startswith("dlconc"):
            concurrent.append(d.get("samples_per_s", 0))
    if concurrent:
        agg = sum(concurrent)
        per = statistics.mean(concurrent)
        nw = None
        # compare aggregate vs (K * single@same-nw) to expose I/O contention
        print(f"\n  concurrent: K={len(concurrent)}  agg={agg:.0f} samples/s  "
              f"per-loader mean={per:.1f}")
        if singles:
            # single throughput at nw=2 if present
            s2 = singles.get(2)
            if s2:
                print(f"  ideal (K x single@nw2) = {len(concurrent)*s2:.0f} samples/s  "
                      f"-> contention factor = {agg/(len(concurrent)*s2):.2f} "
                      f"(<1 = shared-FS I/O wall)")

# ---- E3: realistic 16-actor ---------------------------------------------------
for nw_label in ("e3nw2", "e3nw4"):
    e3 = collect_profilers(nw_label)
    if not e3:
        continue
    hr(f"E3 — REALISTIC 4/GPU x4 = 16 actors  ({nw_label})")
    util, host_peak = _util_by_gpu(os.path.join(OUT, f"util_{nw_label}.csv"))
    allrows = [r for rows in e3.values() for r in rows]
    n = len(allrows)
    steps = [r[0] for r in allrows]
    dlws = [r[1] for r in allrows]
    mems = [r[2] for r in allrows]
    pstep = statistics.mean(steps) if steps else 0.0
    agg = sum(1000.0 / s for s in steps if s > 0)
    dl_share = (statistics.mean(dlws) / pstep) if pstep else 0.0
    meanutil = statistics.mean(util.values()) if util else 0.0
    print(f"actors={n}  per-proc step={pstep:.1f}ms  dataloader_wait share={dl_share:.0%}  "
          f"agg throughput={agg:.1f} steps/s")
    print(f"per-GPU util mean={meanutil:.0f}%  ({ {g: round(u) for g,u in sorted(util.items())} })")
    print(f"VRAM/proc peak~{statistics.mean(mems):.0f}MB  host-RAM peak={host_peak:.0f}MB")

print()
