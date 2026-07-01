"""MCR Phase-3 STEP 2 — client-partition health + non-IID characterization.

Login-node-safe, metadata-ONLY (no geometry build, no GPU): loads the trainval
``train`` info-cache, builds the EXACT FL ``log_group`` partition the bb02d FL run will
use (same floor / requested-N / seed), and characterizes every client BEFORE any GPU is
spent. A degenerate partition invalidates the whole FL baseline, so this is the go/no-go.

Reports, per client: volume (keyframes), continuity (whole contiguous logs), per-class
object coverage (+ degenerate-class flags), and the natural non-IID axes (location +
day/night/rain from ``scene.json``). Plus the partition-level non-IID metrics
(label-skew CoV / JSD / TVD), the fair-comparison anchor (client union == pooled train
set), and a determinism re-check.

Run:
  source fl_v3/scripts/arrhenius_env.sh
  arrhenius_activate_env
  PYTHONPATH=fl_v3/src python fl_v3/scripts/p3_partition_health.py \
      --cache-dir <abs info_cache_msweep10> [--with-scene-meta] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
from fl_v3.data.nuscenes.partition import (
    build_log_group_partition,
    build_log_table,
    coerce_partition_seed,
    derive_max_clients,
    n_at_floor_band,
)
from fl_v3.data.nuscenes.paths import DATAROOT


def _frac(hist: Dict[int, int], n_classes: int) -> np.ndarray:
    v = np.zeros(n_classes, dtype=np.float64)
    for k, c in hist.items():
        v[int(k)] = float(c)
    s = v.sum()
    return v / s if s > 0 else v


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence (bits) between two distributions."""
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log2(a[mask]) - np.log2(np.where(b[mask] > 0, b[mask], 1e-300)))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total-variation distance (0..1)."""
    return float(0.5 * np.sum(np.abs(p - q)))


def load_scene_meta(version: str) -> Dict[str, dict]:
    """scene_token -> {log_token, description, night, rain} read directly from scene.json
    (no devkit). Empty dict if unavailable (axis becomes 'unknown')."""
    path = os.path.join(DATAROOT, version, "scene.json")
    out: Dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            scenes = json.load(f)
    except Exception as e:  # pragma: no cover - best-effort enrichment
        print(f"[scene-meta] WARN: could not read {path}: {e}")
        return out
    for s in scenes:
        desc = str(s.get("description", "")).lower()
        out[s["token"]] = {
            "log_token": s.get("log_token"),
            "description": s.get("description", ""),
            "night": ("night" in desc),
            "rain": ("rain" in desc),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--split", default="train")
    ap.add_argument("--floor", type=int, default=400)
    ap.add_argument("--requested-clients", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20259)
    ap.add_argument("--partition-seed", default="")  # "" -> falls back to run seed
    ap.add_argument("--with-scene-meta", action="store_true")
    ap.add_argument("--out", default="fl_v3/collab/fl_baseline/partition_health.json")
    args = ap.parse_args()

    n_classes = len(DETECTION_NAMES)
    pseed = coerce_partition_seed(args.partition_seed, args.seed)

    print(f"[load] cache={args.cache_dir} version={args.version} split={args.split}")
    info_list, meta = IC.load_cache(args.cache_dir, args.version, args.split)
    print(f"[load] n_samples={len(info_list)} n_boxes(meta)={meta.get('n_boxes')} hash={meta.get('cache_hash','')[:12]}")

    # --- log table + N derivation ---
    log_table = build_log_table(info_list)
    n_logs = len(log_table)
    n_max = derive_max_clients(log_table, args.floor)
    band = n_at_floor_band(log_table, args.floor)
    loc_logs = Counter(e["location"] for e in log_table.values())
    print(f"[logs] n_logs={n_logs} locations={dict(loc_logs)} n_max@floor{args.floor}={n_max} band={band}")

    # --- build the EXACT FL partition ---
    part = build_log_group_partition(log_table, floor=args.floor,
                                     requested_num_clients=args.requested_clients, seed=pseed)
    clients = part["clients"]
    N = part["num_clients"]
    print(f"[partition] mode={part['mode']} N={N} reason={part['reason']}")

    # --- determinism re-check (re-build, compare sample token sets) ---
    part2 = build_log_group_partition(log_table, floor=args.floor,
                                      requested_num_clients=args.requested_clients, seed=pseed)
    det_ok = all(
        clients[i]["sample_tokens"] == part2["clients"][i]["sample_tokens"] for i in range(N)
    ) and len(part2["clients"]) == N
    print(f"[determinism] re-build identical shards: {det_ok}")

    # --- fair-comparison anchor: union of clients == pooled train set, no overlap ---
    all_train_tokens = {i["sample_token"] for i in info_list}
    union = set()
    overlap = 0
    for c in clients:
        st = set(c["sample_tokens"])
        overlap += len(union & st)
        union |= st
    union_eq_pool = (union == all_train_tokens)
    print(f"[anchor] |pool|={len(all_train_tokens)} |union|={len(union)} "
          f"union==pool={union_eq_pool} cross_client_overlap={overlap}")

    # --- whole-log integrity: every log assigned to exactly one client, no log split ---
    log_owner: Dict[str, int] = {}
    split_logs = 0
    for c in clients:
        for lt in c["log_tokens"]:
            if lt in log_owner:
                split_logs += 1
            log_owner[lt] = c["client_id"]
    all_logs_assigned = (set(log_owner) == set(log_table)) and split_logs == 0
    print(f"[continuity] logs_assigned={len(log_owner)}/{n_logs} no_log_split={all_logs_assigned}")

    # --- optional scene-meta (day/night/rain) ---
    scene_meta = load_scene_meta(args.version) if args.with_scene_meta else {}

    # --- per-client characterization ---
    global_hist = Counter()
    for c in clients:
        global_hist.update({int(k): int(v) for k, v in c["class_hist"].items()})
    global_frac = _frac(global_hist, n_classes)

    kfs = [c["n_keyframes"] for c in clients]
    rows = []
    per_class_counts = defaultdict(list)  # class_id -> [count per client]
    degenerate = []  # (client_id, [missing class names])
    starved = []     # clients below floor
    for c in clients:
        hist = {int(k): int(v) for k, v in c["class_hist"].items()}
        total_obj = int(sum(hist.values()))
        frac = _frac(hist, n_classes)
        missing = [DETECTION_NAMES[k] for k in range(n_classes) if hist.get(k, 0) == 0]
        if c["n_keyframes"] < args.floor:
            starved.append(c["client_id"])
        # a client that cannot learn a class at all (0 instances). Rare-but-present is fine (non-IID).
        if missing:
            degenerate.append((c["client_id"], missing))
        for k in range(n_classes):
            per_class_counts[k].append(hist.get(k, 0))

        scene_info = {"night_scenes": 0, "rain_scenes": 0, "n_scenes_meta": 0}
        if scene_meta:
            # client owns whole logs -> gather its scenes via the log_table
            sc_tokens = set()
            for lt in c["log_tokens"]:
                sc_tokens |= set(log_table[lt]["scene_tokens"])
            night = sum(1 for s in sc_tokens if scene_meta.get(s, {}).get("night"))
            rain = sum(1 for s in sc_tokens if scene_meta.get(s, {}).get("rain"))
            scene_info = {"night_scenes": int(night), "rain_scenes": int(rain),
                          "n_scenes_meta": len(sc_tokens)}

        rows.append({
            "client_id": c["client_id"],
            "location": c["location"],
            "n_logs": len(c["log_tokens"]),
            "n_scenes": c["n_scenes"],
            "n_keyframes": c["n_keyframes"],
            "total_objects": total_obj,
            "objects_per_kf": round(total_obj / c["n_keyframes"], 2) if c["n_keyframes"] else 0.0,
            "class_counts": {DETECTION_NAMES[k]: hist.get(k, 0) for k in range(n_classes)},
            "class_frac": {DETECTION_NAMES[k]: round(float(frac[k]), 4) for k in range(n_classes)},
            "missing_classes": missing,
            "jsd_to_global_bits": round(_jsd(frac, global_frac), 4),
            "tvd_to_global": round(_tvd(frac, global_frac), 4),
            **scene_info,
        })

    # --- non-IID quantification ---
    def cov(xs):
        xs = np.asarray(xs, dtype=np.float64)
        m = xs.mean()
        return float(xs.std() / m) if m > 0 else 0.0

    per_class_cov = {DETECTION_NAMES[k]: round(cov(per_class_counts[k]), 3) for k in range(n_classes)}
    jsd_vals = [r["jsd_to_global_bits"] for r in rows]
    tvd_vals = [r["tvd_to_global"] for r in rows]
    vol_cov = cov(kfs)

    non_iid = {
        "volume_keyframes": {
            "min": int(min(kfs)), "max": int(max(kfs)),
            "mean": round(float(np.mean(kfs)), 1), "std": round(float(np.std(kfs)), 1),
            "cov": round(vol_cov, 3), "ratio_max_min": round(max(kfs) / max(min(kfs), 1), 2),
        },
        "label_skew_per_class_cov": per_class_cov,
        "label_skew_jsd_to_global_bits": {
            "mean": round(float(np.mean(jsd_vals)), 4), "min": round(float(np.min(jsd_vals)), 4),
            "max": round(float(np.max(jsd_vals)), 4),
        },
        "label_skew_tvd_to_global": {
            "mean": round(float(np.mean(tvd_vals)), 4), "min": round(float(np.min(tvd_vals)), 4),
            "max": round(float(np.max(tvd_vals)), 4),
        },
    }

    # location + (optional) weather/time axes across clients
    loc_clients = Counter(r["location"] for r in rows)
    loc_kf = defaultdict(int)
    for r in rows:
        loc_kf[r["location"]] += r["n_keyframes"]
    axes = {
        "location_clients": dict(loc_clients),
        "location_keyframes": dict(loc_kf),
    }
    if scene_meta:
        axes["night_clients"] = int(sum(1 for r in rows if r["night_scenes"] > 0))
        axes["rain_clients"] = int(sum(1 for r in rows if r["rain_scenes"] > 0))
        axes["pure_day_dry_clients"] = int(sum(1 for r in rows if r["night_scenes"] == 0 and r["rain_scenes"] == 0))

    # --- global class distribution ---
    global_dist = {DETECTION_NAMES[k]: {"count": int(global_hist.get(k, 0)),
                                        "frac": round(float(global_frac[k]), 4)}
                   for k in range(n_classes)}

    # --- go/no-go ---
    go = (det_ok and union_eq_pool and overlap == 0 and all_logs_assigned
          and len(starved) == 0 and N == args.requested_clients)
    verdict = {
        "GO": bool(go),
        "determinism_ok": bool(det_ok),
        "union_eq_pool": bool(union_eq_pool),
        "no_cross_client_overlap": overlap == 0,
        "no_log_split": bool(all_logs_assigned),
        "n_starved_below_floor": len(starved),
        "starved_client_ids": starved,
        "N_equals_requested": N == args.requested_clients,
        "clients_missing_a_class": [{"client_id": cid, "missing": miss} for cid, miss in degenerate],
    }

    report = {
        "config": {"version": args.version, "split": args.split, "floor": args.floor,
                   "requested_clients": args.requested_clients, "seed": args.seed,
                   "partition_seed": pseed, "cache_dir": args.cache_dir,
                   "cache_hash": meta.get("cache_hash")},
        "totals": {"n_logs": n_logs, "n_samples": len(info_list), "N_clients": N,
                   "n_max_at_floor": n_max, "n_at_floor_band": band,
                   "logs_per_location": dict(loc_logs)},
        "verdict": verdict,
        "non_iid": non_iid,
        "axes": axes,
        "global_class_distribution": global_dist,
        "clients": rows,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[out] wrote {args.out}")

    # --- human-readable table ---
    print("\n" + "=" * 110)
    print(f"PARTITION HEALTH — {args.version}/{args.split}  floor={args.floor} requested={args.requested_clients} "
          f"seed={args.seed}  N={N}  mode={part['mode']}")
    print("=" * 110)
    hdr = f"{'cid':>3} {'location':<20} {'logs':>4} {'scn':>4} {'kf':>5} {'obj':>7} {'o/kf':>5} {'jsd':>5} {'tvd':>5} {'miss':>4}"
    if scene_meta:
        hdr += f" {'nght':>4} {'rain':>4}"
    print(hdr)
    print("-" * 110)
    for r in rows:
        line = (f"{r['client_id']:>3} {r['location']:<20} {r['n_logs']:>4} {r['n_scenes']:>4} "
                f"{r['n_keyframes']:>5} {r['total_objects']:>7} {r['objects_per_kf']:>5} "
                f"{r['jsd_to_global_bits']:>5} {r['tvd_to_global']:>5} {len(r['missing_classes']):>4}")
        if scene_meta:
            line += f" {r['night_scenes']:>4} {r['rain_scenes']:>4}"
        print(line)
    print("-" * 110)
    print(f"volume kf: min={non_iid['volume_keyframes']['min']} max={non_iid['volume_keyframes']['max']} "
          f"mean={non_iid['volume_keyframes']['mean']} cov={non_iid['volume_keyframes']['cov']} "
          f"max/min={non_iid['volume_keyframes']['ratio_max_min']}")
    print(f"label-skew per-class CoV: {per_class_cov}")
    print(f"JSD-to-global (bits): {non_iid['label_skew_jsd_to_global_bits']}")
    print(f"axes: {axes}")
    print(f"\nGLOBAL class distribution (pooled train):")
    for k in range(n_classes):
        gd = global_dist[DETECTION_NAMES[k]]
        print(f"  {DETECTION_NAMES[k]:<22} {gd['count']:>8}  {gd['frac']*100:>5.2f}%")
    print("\n" + "=" * 110)
    print(f"VERDICT: {'GO' if go else 'NO-GO'}  {json.dumps(verdict)}")
    print("=" * 110)


if __name__ == "__main__":
    main()
