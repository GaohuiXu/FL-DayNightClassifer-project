"""Geographic log-group partitioner (fl_v3 T1).

A **client = a deterministic, location-coherent log-group**. Logs are grouped
into clients so that **(a)** each client is
location-coherent (a log maps to exactly one of the 4 nuScenes locations, so
coherence is automatic at log granularity), **(b)** grouping splits by log/scene —
never mid-sample, so synchronized camera+LiDAR keyframes stay whole, **(c)** each
client meets a **``min-keyframes-per-client`` floor**.

**N is DERIVED from the floor, never assumed.** The floor is a declared, justified
config; N is the *maximum* number of location-coherent log-groups each ≥ floor.
If a requested ``num-clients`` (e.g. 50) exceeds that maximum, we **fall back to
N∈{20,25}** and record the reason (the real trainval ``train`` split has only 50
logs with min 80 keyframes, so 50 single-log clients violate any meaningful floor).

Determinism: logs sorted by ``log_token`` within location; bin assignment is LPT
(longest-processing-time) with ``log_token`` tie-breaks; sample_tokens sorted ⇒
same ``(seed, floor, version, split)`` → identical shards + identical N.

This reuses :func:`fl_v3.data.partition.iid_partition` for the IID sample-shard
baseline. The per-client class-histogram hook supports clean partition diagnostics.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from fl_v3.data.partition import iid_partition
from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
from fl_v3.data.nuscenes.paths import KNOWN_LOCATIONS

# Fallback client counts when a requested num-clients violates the floor (SPEC).
FALLBACK_CANDIDATES = (25, 20)


def coerce_partition_seed(partition_seed, run_seed: int) -> int:
    """Empty-string ``partition-seed`` falls back to the run ``seed`` (config default
    is ``""``); otherwise coerce to int. Done before any RNG."""
    if partition_seed is None:
        return int(run_seed)
    s = str(partition_seed).strip()
    if s == "":
        return int(run_seed)
    return int(s)


# ---------------------------------------------------------------------------
# Log table (built from the info-cache; no NuScenes reload needed).
# ---------------------------------------------------------------------------
def build_log_table(info_list: List[dict]) -> Dict[str, dict]:
    """``log_token`` → {location, scene_tokens, sample_tokens (sorted), kf, class_hist}.

    Built from the split's info-cache, so the table already excludes val/test
    (no-leakage by construction). Asserts every location ⊆ the 4 known locations.
    """
    table: Dict[str, dict] = {}
    for info in info_list:
        lt = info["log_token"]
        if lt not in table:
            table[lt] = {
                "log_token": lt,
                "location": info["location"],
                "scene_tokens": set(),
                "sample_tokens": [],
                "class_hist": Counter(),
            }
        entry = table[lt]
        if entry["location"] != info["location"]:
            raise ValueError(f"log {lt} maps to two locations — corrupt metadata")
        entry["scene_tokens"].add(info["scene_token"])
        entry["sample_tokens"].append(info["sample_token"])
        entry["class_hist"].update(int(x) for x in info["gt_labels"].tolist())
    bad = {e["location"] for e in table.values()} - set(KNOWN_LOCATIONS)
    if bad:
        raise ValueError(f"unexpected location(s) {bad}; expected ⊆ {KNOWN_LOCATIONS}")
    for entry in table.values():
        entry["sample_tokens"] = sorted(entry["sample_tokens"])
        entry["kf"] = len(entry["sample_tokens"])
        entry["n_scenes"] = len(entry["scene_tokens"])
    return table


def build_log_table_from_nusc(nusc, split: str, with_class_hist: bool = True) -> Dict[str, dict]:
    """Metadata-only log table directly from the devkit (no geometry build).

    This is the **login-node-safe, metadata-only** trainval indexing path (the SPEC
    forbids training/geometry on trainval in T1) and the basis of the
    real-trainval-log-table partition unit test. Produces the same structure as
    :func:`build_log_table` (location, sample_tokens, kf, class_hist). Set
    ``with_class_hist=False`` to skip the per-annotation pass (faster; the N-
    derivation / fallback / location tests don't need histograms).
    """
    from fl_v3.data.nuscenes.info_cache import split_sample_tokens
    from fl_v3.data.nuscenes.class_map import category_to_detection_name, NAME_TO_ID

    tokens = split_sample_tokens(nusc, split)
    table: Dict[str, dict] = {}
    for tok in tokens:
        s = nusc.get("sample", tok)
        scene = nusc.get("scene", s["scene_token"])
        log = nusc.get("log", scene["log_token"])
        lt = scene["log_token"]
        if lt not in table:
            table[lt] = {"log_token": lt, "location": log["location"],
                         "scene_tokens": set(), "sample_tokens": [], "class_hist": Counter()}
        entry = table[lt]
        if entry["location"] != log["location"]:
            raise ValueError(f"log {lt} maps to two locations — corrupt metadata")
        entry["scene_tokens"].add(s["scene_token"])
        entry["sample_tokens"].append(tok)
        if with_class_hist:
            for ann_tok in s["anns"]:
                name = category_to_detection_name(nusc.get("sample_annotation", ann_tok)["category_name"])
                if name is not None:
                    entry["class_hist"][NAME_TO_ID[name]] += 1
    bad = {e["location"] for e in table.values()} - set(KNOWN_LOCATIONS)
    if bad:
        raise ValueError(f"unexpected location(s) {bad}; expected ⊆ {KNOWN_LOCATIONS}")
    for entry in table.values():
        entry["sample_tokens"] = sorted(entry["sample_tokens"])
        entry["kf"] = len(entry["sample_tokens"])
        entry["n_scenes"] = len(entry["scene_tokens"])
    return table


def _logs_by_location(log_table: Dict[str, dict]) -> Dict[str, List[dict]]:
    by_loc: Dict[str, List[dict]] = defaultdict(list)
    for entry in log_table.values():
        by_loc[entry["location"]].append(entry)
    for loc in by_loc:
        by_loc[loc].sort(key=lambda e: e["log_token"])  # deterministic order
    return dict(sorted(by_loc.items()))


# ---------------------------------------------------------------------------
# Binning primitives (deterministic).
# ---------------------------------------------------------------------------
def _greedy_pack(logs: List[dict], floor: int) -> List[List[dict]]:
    """Max groups (in log_token order) each with cumulative kf ≥ floor; trailing
    sub-floor leftover merged into the previous group. Maximizes group count."""
    groups: List[List[dict]] = []
    cur: List[dict] = []
    cur_kf = 0
    for e in logs:  # already log_token-sorted
        cur.append(e)
        cur_kf += e["kf"]
        if cur_kf >= floor:
            groups.append(cur)
            cur, cur_kf = [], 0
    if cur:  # leftover < floor
        if groups:
            groups[-1].extend(cur)
        else:
            groups.append(cur)  # whole location below floor → one (sub-floor) group
    return groups


def _lpt_bins(logs: List[dict], k: int) -> List[List[dict]]:
    """LPT bin-balancing of ``logs`` into ``k`` groups (maximizes the min group sum).

    Deterministic: sort by (−kf, log_token); assign each to the smallest-sum bin
    (ties → lowest index). With k ≤ len(logs) every bin is non-empty."""
    if k <= 0:
        raise ValueError("k must be ≥ 1")
    if k > len(logs):
        raise ValueError(f"cannot split {len(logs)} logs into {k} non-empty groups")
    order = sorted(logs, key=lambda e: (-e["kf"], e["log_token"]))
    bins: List[List[dict]] = [[] for _ in range(k)]
    sums = [0] * k
    for e in order:
        j = min(range(k), key=lambda i: (sums[i], i))
        bins[j].append(e)
        sums[j] += e["kf"]
    # Re-sort each bin's logs by log_token for a stable, readable shard.
    for b in bins:
        b.sort(key=lambda e: e["log_token"])
    return bins


def derive_max_clients(log_table: Dict[str, dict], floor: int) -> int:
    """Max location-coherent log-groups each ≥ floor (the floor-derived N)."""
    by_loc = _logs_by_location(log_table)
    return sum(len(_greedy_pack(logs, floor)) for logs in by_loc.values())


# ---------------------------------------------------------------------------
# Apportionment for a target client count (fallback / explicit num-clients).
# ---------------------------------------------------------------------------
def _location_caps(log_table: Dict[str, dict], floor: int) -> Dict[str, int]:
    """Per-location max groups each ≥ floor (the cap on that location's client count)."""
    by_loc = _logs_by_location(log_table)
    return {loc: len(_greedy_pack(logs, floor)) for loc, logs in by_loc.items()}


def _apportion(target_n: int, by_loc: Dict[str, List[dict]], caps: Dict[str, int]) -> Dict[str, int]:
    """Clamped largest-remainder apportionment of ``target_n`` clients across
    locations by keyframe mass, each in ``[1, cap_loc]``."""
    locs = list(by_loc.keys())  # already sorted
    mass = {loc: sum(e["kf"] for e in by_loc[loc]) for loc in locs}
    total_mass = sum(mass.values())
    n_locs = len(locs)
    if target_n < n_locs:
        raise ValueError(f"target N={target_n} < #locations {n_locs}: cannot represent all locations")
    if target_n > sum(caps.values()):
        raise ValueError(f"target N={target_n} exceeds total cap {sum(caps.values())} at this floor")

    alloc = {loc: 1 for loc in locs}  # represent every location first
    remaining = target_n - n_locs
    # Largest-remainder on the ideal extra shares, respecting headroom cap-1.
    ideal = {loc: target_n * mass[loc] / total_mass for loc in locs}
    extra_floor = {loc: int(ideal[loc]) for loc in locs}
    # headroom per loc
    for loc in locs:
        give = min(max(extra_floor[loc] - 1, 0), caps[loc] - 1, remaining)
        alloc[loc] += give
        remaining -= give
    # distribute the rest by descending fractional remainder (ties → location order)
    while remaining > 0:
        cand = sorted(
            (loc for loc in locs if alloc[loc] < caps[loc]),
            key=lambda loc: (-(ideal[loc] - int(ideal[loc])), loc),
        )
        if not cand:
            raise ValueError("apportionment failed: no headroom left (should not happen)")
        alloc[cand[0]] += 1
        remaining -= 1
    return alloc


def _target_n_groups(log_table: Dict[str, dict], target_n: int, floor: int):
    """Split logs into exactly ``target_n`` location-coherent groups (LPT-balanced).

    Returns ``(groups, feasible)`` where ``feasible`` is True iff every group ≥ floor.
    """
    by_loc = _logs_by_location(log_table)
    caps = _location_caps(log_table, floor)
    alloc = _apportion(target_n, by_loc, caps)
    groups: List[List[dict]] = []
    for loc in by_loc:
        groups.extend(_lpt_bins(by_loc[loc], alloc[loc]))
    feasible = all(sum(e["kf"] for e in g) >= floor for g in groups)
    return groups, feasible


# ---------------------------------------------------------------------------
# Public: build the partition.
# ---------------------------------------------------------------------------
def _materialize(groups: List[List[dict]]) -> List[dict]:
    """Turn log-groups into client dicts (sorted by location then first log_token)."""
    clients = []
    for g in groups:
        g_sorted = sorted(g, key=lambda e: e["log_token"])
        loc = g_sorted[0]["location"]
        sample_tokens = sorted(t for e in g_sorted for t in e["sample_tokens"])
        hist = Counter()
        for e in g_sorted:
            hist.update(e["class_hist"])
        clients.append({
            "location": loc,
            "log_tokens": [e["log_token"] for e in g_sorted],
            "sample_tokens": sample_tokens,
            "n_scenes": sum(e["n_scenes"] for e in g_sorted),
            "n_keyframes": len(sample_tokens),
            "class_hist": {int(k): int(v) for k, v in sorted(hist.items())},
        })
    # Deterministic client id ordering: by (location, first log_token).
    clients.sort(key=lambda c: (c["location"], c["log_tokens"][0]))
    for cid, c in enumerate(clients):
        c["client_id"] = cid
    return clients


def _result(clients, mode, reason, requested, floor, n_max) -> dict:
    """Assemble the partition dict + a sub-floor guard.

    A client below the floor can only arise in the floor-derived path when ``floor``
    exceeds a whole location's keyframe total (a location must still get ≥1 client to
    stay represented). It is unreachable at the operating floor and its ±50 % band on
    trainval, but we surface it explicitly (``sub_floor_clients`` + a reason note) so it
    is never a *silent* invariant break — the requested/fallback paths guarantee 0.
    """
    sub_floor = [c["client_id"] for c in clients if c["n_keyframes"] < floor]
    if sub_floor:
        reason += (
            f" | WARNING: {len(sub_floor)} client(s) below floor={floor} "
            f"(client_ids {sub_floor}); a whole location's keyframes < floor — "
            f"lower the floor or merge that location. Sub-floor=engineering-smoke only."
        )
    return {
        "clients": clients,
        "num_clients": len(clients),
        "mode": mode,
        "reason": reason,
        "requested_num_clients": requested,
        "floor": floor,
        "n_max_at_floor": n_max,
        "sub_floor_clients": sub_floor,
    }


def build_log_group_partition(
    log_table: Dict[str, dict],
    floor: int,
    requested_num_clients: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Build the location-coherent log-group client partition.

    Returns a dict: ``clients`` (list), ``num_clients`` (derived N), ``mode``,
    ``reason``, ``requested_num_clients``, ``floor``, ``n_max_at_floor``,
    ``sub_floor_clients`` (ids below the floor — non-empty only in the degenerate
    floor>location-total case).
    """
    n_max = derive_max_clients(log_table, floor)
    if requested_num_clients is None:
        groups = []
        by_loc = _logs_by_location(log_table)
        for logs in by_loc.values():
            groups.extend(_greedy_pack(logs, floor))
        return _result(
            _materialize(groups), "floor-derived",
            f"N derived from floor={floor}: maximum location-coherent log-groups each ≥ floor",
            None, floor, n_max)

    requested = int(requested_num_clients)
    if requested <= n_max:
        groups, feasible = _target_n_groups(log_table, requested, floor)
        if feasible:
            return _result(
                _materialize(groups), "requested-honored",
                f"requested num-clients={requested} ≤ max feasible {n_max} at floor={floor}; honored",
                requested, floor, n_max)
    # Infeasible request → fall back to N∈{20,25}.
    for fb in FALLBACK_CANDIDATES:
        if fb <= n_max:
            groups, feasible = _target_n_groups(log_table, fb, floor)
            if feasible:
                return _result(
                    _materialize(groups), "fallback",
                    (f"requested num-clients={requested} exceeds max feasible {n_max} "
                     f"at floor={floor} (only {len(log_table)} logs, min-log below floor); "
                     f"fell back to N={fb}"),
                    requested, floor, n_max)
    raise ValueError(
        f"no feasible client count: requested={requested}, n_max={n_max}, "
        f"fallbacks {FALLBACK_CANDIDATES} all exceed n_max or infeasible at floor={floor}"
    )


def n_at_floor_band(log_table: Dict[str, dict], floor: int) -> Dict[str, int]:
    """N derived at the chosen floor and at floor ±50 % (shows non-arbitrariness)."""
    lo = max(1, int(round(floor * 0.5)))
    hi = int(round(floor * 1.5))
    return {
        f"floor_{lo}": derive_max_clients(log_table, lo),
        f"floor_{floor}": derive_max_clients(log_table, floor),
        f"floor_{hi}": derive_max_clients(log_table, hi),
    }


# ---------------------------------------------------------------------------
# IID baseline + stats reporting.
# ---------------------------------------------------------------------------
def iid_sample_partition(info_list: List[dict], num_clients: int, seed: int = 42) -> Dict[int, List[str]]:
    """IID sample-shard regime (Q2-heterogeneity baseline): random equal sample
    shards over the split, returned as client → ``sample_token`` lists."""
    tokens = sorted(i["sample_token"] for i in info_list)
    idx_map = iid_partition(num_samples=len(tokens), num_clients=num_clients, seed=seed)
    return {cid: sorted(tokens[i] for i in idxs) for cid, idxs in idx_map.items()}


def partition_report(partition: dict, scale: str, version: str, split: str,
                     log_table: Dict[str, dict]) -> dict:
    """Machine-readable partition stats + scale stamp (for the viz/JSON artifact)."""
    clients = partition["clients"]
    kfs = [c["n_keyframes"] for c in clients]
    loc_counts = Counter(c["location"] for c in clients)
    return {
        "scale": scale,  # "mini-smoke" | "trainval-scientific"
        "version": version,
        "split": split,
        "mode": partition["mode"],
        "reason": partition["reason"],
        "floor": partition["floor"],
        "requested_num_clients": partition["requested_num_clients"],
        "n_max_at_floor": partition["n_max_at_floor"],
        "num_clients": partition["num_clients"],
        "sub_floor_clients": partition.get("sub_floor_clients", []),
        "n_logs": len(log_table),
        "n_at_floor_band": n_at_floor_band(log_table, partition["floor"]),
        "min_keyframes": min(kfs) if kfs else 0,
        "max_keyframes": max(kfs) if kfs else 0,
        "total_keyframes": int(sum(kfs)),
        "clients_per_location": {loc: loc_counts[loc] for loc in sorted(loc_counts)},
        "detection_names": list(DETECTION_NAMES),
        "clients": [
            {
                "client_id": c["client_id"],
                "location": c["location"],
                "n_logs": len(c["log_tokens"]),
                "n_scenes": c["n_scenes"],
                "n_keyframes": c["n_keyframes"],
                "class_hist": c["class_hist"],
            }
            for c in clients
        ],
    }
