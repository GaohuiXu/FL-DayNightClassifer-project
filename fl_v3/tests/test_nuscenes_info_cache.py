"""T1 — host-portable info-cache reproducibility + no-leakage (info_cache.py)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P


def test_hash_reproducible_build_twice(nusc_mini, dataroot):
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")
    a = IC.build_info_list(nusc_mini, tokens, dataroot)
    b = IC.build_info_list(nusc_mini, tokens, dataroot)
    assert IC.canonical_hash(a) == IC.canonical_hash(b)
    assert len(IC.canonical_hash(a)) == 64


def test_hash_is_dataroot_relative_and_host_portable(mini_val_info, dataroot):
    # Stored paths must be DATAROOT-relative — no leading '/' (absolute) AND no leading
    # '..' (a foreign-resolved path that still leaks host structure into the hash).
    for info in mini_val_info:
        for rel in (*info["cam_rel_paths"], info["lidar_rel_path"]):
            assert not rel.startswith("/") and dataroot not in rel
            assert not rel.startswith(".."), rel
    # The hash must be invariant to the dataroot argument (host portability): the
    # same metadata under a different (symlinked) root yields the same hash, because
    # only relative paths + raw values feed the hash.
    h_real = IC.canonical_hash(mini_val_info)
    # rebuild with an equivalent absolute root spelling (trailing slash) — relative
    # paths identical → identical hash.
    again = IC.build_info_list(
        __import__("nuscenes", fromlist=["NuScenes"]).NuScenes(
            version="v1.0-mini", dataroot=dataroot, verbose=False),
        [i["sample_token"] for i in mini_val_info], dataroot + "/")
    assert IC.canonical_hash(again) == h_real


def test_derived_schema_bit_identical_across_builds(nusc_mini, dataroot):
    """Same-machine: two independent info builds → bit-identical served schema
    (matrices included), not just the raw-input hash."""
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")[:3]
    a = IC.build_info_list(nusc_mini, tokens, dataroot)
    b = IC.build_info_list(nusc_mini, tokens, dataroot)
    dsa = DS.NuScenesMultimodalDataset(a, dataroot)
    dsb = DS.NuScenesMultimodalDataset(b, dataroot)
    for i in range(len(dsa)):
        assert torch.equal(dsa[i]["lidar2img"], dsb[i]["lidar2img"])
        assert torch.equal(dsa[i]["gt_boxes"], dsb[i]["gt_boxes"])
        assert torch.equal(dsa[i]["cam_intrinsics"], dsb[i]["cam_intrinsics"])


def test_save_load_roundtrip_and_guard(nusc_mini, dataroot, tmp_path):
    cache_dir = str(tmp_path / "info_cache")
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")
    info = IC.build_info_list(nusc_mini, tokens, dataroot)
    meta = IC.save_cache(info, cache_dir, "v1.0-mini", "mini_val", "mini-smoke", dataroot)
    assert meta["cache_hash"] == IC.canonical_hash(info)
    assert meta["scale"] == "mini-smoke"
    assert meta["n_sweeps"] == 1
    loaded, lmeta = IC.load_cache(cache_dir, "v1.0-mini", "mini_val")
    assert lmeta["cache_hash"] == meta["cache_hash"]
    assert IC.canonical_hash(loaded) == meta["cache_hash"]
    # Load-fidelity of the SERVED schema (the non-hashed derived tensors must also
    # survive save→load byte-identically — the hash alone only covers raw inputs).
    toks = sorted(i["sample_token"] for i in info)[:3]
    ds_mem = DS.NuScenesMultimodalDataset(info, dataroot, sample_tokens=toks)
    ds_disk = DS.NuScenesMultimodalDataset(loaded, dataroot, sample_tokens=toks)
    for i in range(len(ds_mem)):
        a, b = ds_mem[i], ds_disk[i]
        assert torch.equal(a["lidar2img"], b["lidar2img"])
        assert torch.equal(a["gt_boxes"], b["gt_boxes"])
        assert torch.equal(a["cam_intrinsics"], b["cam_intrinsics"])
        assert torch.equal(a["gt_velocity"], b["gt_velocity"])
        assert a["gt_instance_tokens"] == b["gt_instance_tokens"]


def test_cache_depth_is_bound_in_filename_meta_records_and_dataset(
    nusc_mini, dataroot, tmp_path
):
    cache_dir = str(tmp_path / "depth_cache")
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")[:3]
    info2 = IC.build_info_list(nusc_mini, tokens, dataroot, n_sweeps=2)
    meta2 = IC.save_cache(
        info2,
        cache_dir,
        "v1.0-mini",
        "mini_val",
        "mini-smoke",
        dataroot,
        n_sweeps=2,
    )
    pkl2, sidecar2 = IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)
    assert "nsweeps2" in pkl2
    assert meta2["n_sweeps"] == 2
    assert all(info["_cache_n_sweeps"] == 2 for info in info2)
    with open(sidecar2, encoding="utf-8") as stream:
        assert json.load(stream)["n_sweeps"] == 2
    loaded2, loaded_meta2 = IC.load_cache(
        cache_dir, "v1.0-mini", "mini_val", n_sweeps=2
    )
    assert loaded_meta2 == meta2
    assert len(loaded2) == len(info2)

    with pytest.raises(FileNotFoundError, match="nsweeps10"):
        IC.load_cache(cache_dir, "v1.0-mini", "mini_val", n_sweeps=10)
    with pytest.raises(ValueError, match="_cache_n_sweeps=2, requested 10"):
        DS.NuScenesMultimodalDataset(info2, dataroot, n_sweeps=10)

    mixed = copy.deepcopy(info2)
    mixed[-1]["_cache_n_sweeps"] = 10
    with pytest.raises(ValueError, match="declares _cache_n_sweeps=10, expected 2"):
        IC.save_cache(
            mixed,
            cache_dir,
            "v1.0-mini",
            "mini_val",
            "mini-smoke",
            dataroot,
            n_sweeps=2,
        )


def test_load_cache_rejects_ambiguous_depth_and_sidecar_drift(
    nusc_mini, dataroot, tmp_path
):
    cache_dir = str(tmp_path / "ambiguous_cache")
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")[:2]
    for depth in (2, 3):
        infos = IC.build_info_list(nusc_mini, tokens, dataroot, n_sweeps=depth)
        IC.save_cache(
            infos,
            cache_dir,
            "v1.0-mini",
            "mini_val",
            "mini-smoke",
            dataroot,
            n_sweeps=depth,
        )
    with pytest.raises(ValueError, match="ambiguous nuScenes cache depth"):
        IC.load_cache(cache_dir, "v1.0-mini", "mini_val")

    _, sidecar = IC.cache_paths(cache_dir, "v1.0-mini", "mini_val", 2)
    with open(sidecar, encoding="utf-8") as stream:
        meta = json.load(stream)
    meta["n_sweeps"] = 10
    with open(sidecar, "w", encoding="utf-8") as stream:
        json.dump(meta, stream)
    with pytest.raises(ValueError, match="sidecar differs"):
        IC.load_cache(cache_dir, "v1.0-mini", "mini_val", n_sweeps=2)


def test_relative_to_dataroot_rejects_escaping_path():
    import pytest

    # A blob path outside DATAROOT would leak host structure into the hash → must raise.
    with pytest.raises(ValueError):
        P.relative_to_dataroot("/some/other/root/samples/CAM_FRONT/x.jpg")


def test_cache_dir_under_dataroot_raises(nusc_mini, dataroot):
    import pytest

    bad = P.version_table_dir("v1.0-mini")  # under DATAROOT
    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")[:1]
    info = IC.build_info_list(nusc_mini, tokens, dataroot)
    with pytest.raises(PermissionError):
        IC.save_cache(info, bad, "v1.0-mini", "mini_val", "mini-smoke", dataroot)


def test_no_leakage_train_disjoint_val(nusc_mini):
    train = set(IC.split_sample_tokens(nusc_mini, "mini_train"))
    val = set(IC.split_sample_tokens(nusc_mini, "mini_val"))
    assert train and val
    assert train.isdisjoint(val)


def test_split_tokens_sorted(nusc_mini):
    toks = IC.split_sample_tokens(nusc_mini, "mini_train")
    assert toks == sorted(toks)
