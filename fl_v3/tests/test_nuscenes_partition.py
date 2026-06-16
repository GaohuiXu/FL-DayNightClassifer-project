"""T1 — geographic log-group partitioner (partition.py).

The decisive tests run on the REAL trainval log table (mini is degenerate: too few
logs for the floor / fallback to fire).
"""
from __future__ import annotations

import pytest

from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import partition as PT


# ---------------------------------------------------------------------------
# Determinism (on mini — cheap).
# ---------------------------------------------------------------------------
def test_partition_seed_coercion():
    assert PT.coerce_partition_seed("", 42) == 42       # empty → run seed
    assert PT.coerce_partition_seed(None, 7) == 7
    assert PT.coerce_partition_seed("13", 42) == 13


def test_stable_shards_same_inputs(mini_train_info):
    lt = PT.build_log_table(mini_train_info)
    a = PT.build_log_group_partition(lt, floor=50, seed=42)
    b = PT.build_log_group_partition(lt, floor=50, seed=42)
    assert a["num_clients"] == b["num_clients"]
    amap = {c["client_id"]: c["sample_tokens"] for c in a["clients"]}
    bmap = {c["client_id"]: c["sample_tokens"] for c in b["clients"]}
    assert amap == bmap


def test_mini_is_degenerate_smoke(mini_train_info):
    """Mini N is reported but explicitly degenerate (N≤6) — NOT a partition-quality
    test (too few logs for the fallback to fire)."""
    lt = PT.build_log_table(mini_train_info)
    part = PT.build_log_group_partition(lt, floor=50, seed=42)
    assert part["num_clients"] <= 6
    rep = PT.partition_report(part, "mini-smoke", "v1.0-mini", "mini_train", lt)
    assert rep["scale"] == "mini-smoke"
    # location coherence holds trivially on mini too
    for c in part["clients"]:
        locs = {lt[lg]["location"] for lg in c["log_tokens"]}
        assert len(locs) == 1


def test_iid_baseline_partition(mini_train_info):
    part = PT.iid_sample_partition(mini_train_info, num_clients=3, seed=42)
    all_toks = sorted(t for toks in part.values() for t in toks)
    assert all_toks == sorted(i["sample_token"] for i in mini_train_info)  # exact cover, no dup


def test_iid_partition_deterministic_and_seed_sensitive(mini_train_info):
    """The Q2 IID baseline must be deterministic per seed AND actually seed-dependent."""
    a = PT.iid_sample_partition(mini_train_info, num_clients=3, seed=42)
    b = PT.iid_sample_partition(mini_train_info, num_clients=3, seed=42)
    assert a == b  # same seed → identical shards
    c = PT.iid_sample_partition(mini_train_info, num_clients=3, seed=7)
    assert a != c  # different seed → different shards (genuine randomization)


def test_sub_floor_client_is_surfaced_not_silent(mini_train_info):
    """When a location's total keyframes < floor, the floor-derived path returns a
    sub-floor client but SURFACES it (sub_floor_clients + a WARNING in reason) — never
    a silent invariant break. Mini's singapore-hollandvillage log has 39 kf < floor 50."""
    lt = PT.build_log_table(mini_train_info)
    part = PT.build_log_group_partition(lt, floor=50, seed=42)
    assert part["sub_floor_clients"], part["sub_floor_clients"]
    assert "WARNING" in part["reason"]
    assert any(c["n_keyframes"] < 50 for c in part["clients"])
    rep = PT.partition_report(part, "mini-smoke", "v1.0-mini", "mini_train", lt)
    assert rep["sub_floor_clients"] == part["sub_floor_clients"]


# ---------------------------------------------------------------------------
# The real trainval log table (heavy: loads trainval once).
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trainval_log_table(nusc_trainval):
    # Metadata-only, no geometry (login-node-safe). Skip class_hist for speed.
    return PT.build_log_table_from_nusc(nusc_trainval, "train", with_class_hist=False)


def test_trainval_log_table_shape(trainval_log_table):
    lt = trainval_log_table
    assert len(lt) == 50  # 50 train logs
    total_kf = sum(e["kf"] for e in lt.values())
    assert total_kf == 28130
    locs = {e["location"] for e in lt.values()}
    assert locs == {"boston-seaport", "singapore-onenorth",
                    "singapore-queenstown", "singapore-hollandvillage"}


def test_floor_derivation_band(trainval_log_table):
    band = PT.n_at_floor_band(trainval_log_table, 400)
    assert band["floor_400"] == 39
    assert band["floor_200"] == 44   # floor −50 %
    assert band["floor_600"] == 29   # floor +50 %
    # monotone: lower floor ⇒ ≥ clients
    assert band["floor_200"] >= band["floor_400"] >= band["floor_600"]


def test_requested_50_triggers_fallback(trainval_log_table):
    """The headline GATE: requested num-clients=50 exceeds max feasible (39 at
    floor 400) → fallback fires returning N∈{20,25} with a recorded reason."""
    part = PT.build_log_group_partition(trainval_log_table, floor=400,
                                        requested_num_clients=50, seed=42)
    assert part["mode"] == "fallback"
    assert part["num_clients"] in (20, 25)
    assert "exceeds max feasible" in part["reason"]
    assert part["n_max_at_floor"] == 39


def test_no_client_mixes_locations(trainval_log_table):
    part = PT.build_log_group_partition(trainval_log_table, floor=400,
                                        requested_num_clients=50, seed=42)
    for c in part["clients"]:
        locs = {trainval_log_table[lg]["location"] for lg in c["log_tokens"]}
        assert len(locs) == 1, c
    # every client meets the floor
    assert all(c["n_keyframes"] >= 400 for c in part["clients"])


def test_floor_derived_n_equals_39(trainval_log_table):
    part = PT.build_log_group_partition(trainval_log_table, floor=400, seed=42)
    assert part["mode"] == "floor-derived" and part["num_clients"] == 39
    assert all(c["n_keyframes"] >= 400 for c in part["clients"])
    assert part["sub_floor_clients"] == []  # no sub-floor at the operating floor


def test_trainval_build_keyframe_info_smoke(nusc_trainval):
    """The full canonical-schema build path runs on a real TRAINVAL keyframe (mini
    exercises geometry, but this is the declared metadata-only trainval index path)."""
    from fl_v3.data.nuscenes import info_cache as IC
    from fl_v3.data.nuscenes.paths import DATAROOT, KNOWN_LOCATIONS

    toks = IC.split_sample_tokens(nusc_trainval, "train")[:1]
    info = IC.build_info_list(nusc_trainval, toks, DATAROOT)[0]
    assert info["gt_boxes"].shape[1] == 7
    assert info["lidar2img"].shape == (6, 4, 4)
    assert len(info["cam_rel_paths"]) == 6
    assert info["location"] in set(KNOWN_LOCATIONS)
    assert info["gt_ann_tokens"] == sorted(info["gt_ann_tokens"])  # ann_token-sorted


def test_clients_subset_of_train_no_val_leakage(nusc_trainval, trainval_log_table):
    """Client samples ⊂ train; ∩ val == ∅ (no leakage)."""
    part = PT.build_log_group_partition(trainval_log_table, floor=400,
                                        requested_num_clients=50, seed=42)
    client_toks = set(t for c in part["clients"] for t in c["sample_tokens"])
    val = set(IC.split_sample_tokens(nusc_trainval, "val"))
    train = set(IC.split_sample_tokens(nusc_trainval, "train"))
    assert client_toks.issubset(train)
    assert client_toks.isdisjoint(val)


def test_report_has_scale_and_band(trainval_log_table):
    part = PT.build_log_group_partition(trainval_log_table, floor=400,
                                        requested_num_clients=50, seed=42)
    rep = PT.partition_report(part, "trainval-scientific", "v1.0-trainval", "train", trainval_log_table)
    assert rep["scale"] == "trainval-scientific"
    assert "floor_400" in rep["n_at_floor_band"]
    assert rep["clients_per_location"]  # per-location N reported
    assert set(rep["clients_per_location"]) <= {
        "boston-seaport", "singapore-onenorth", "singapore-queenstown", "singapore-hollandvillage"}
