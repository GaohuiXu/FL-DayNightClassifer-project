"""T1 — canonical schema, bit-determinism, pinned decoder, multi-worker (dataset.py)."""
from __future__ import annotations

import numpy as np
import torch

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.class_map import CLASS_RANGE, category_to_detection_name

# COMMITTED golden image sha256 for the mini sentinel keyframe (all 6 cams, native
# uint8). Platform-pinned to the Alvis PyTorch/2.7.1 venv's PIL/libjpeg; if a future
# decoder swap (PIL→opencv, convert-mode/dtype change) alters the decoded pixels this
# golden FAILS — that is the point (the within-run self-comparison could not catch it).
# Regenerate at the ARM migration with scripts/run_v1_calibration's decoder.
_GOLDEN_SENTINEL_IMG_SHA256 = "b7f2563c676a0beb634467bf67ccc75e1bd2880a95a91ff975804bbd96cd8d85"


def _ds(info, dataroot, n=4):
    tokens = sorted(i["sample_token"] for i in info)[:n]
    return DS.NuScenesMultimodalDataset(info, dataroot, sample_tokens=tokens)


def test_schema_pinned(mini_train_info, dataroot):
    ds = _ds(mini_train_info, dataroot, n=1)
    s = ds[0]
    assert s["cam_order"] == DS.CAM_ORDER  # frozen constant
    assert s["images"].dtype == torch.uint8 and tuple(s["images"].shape) == (6, 3, 900, 1600)
    assert s["cam_intrinsics"].dtype == torch.float32 and tuple(s["cam_intrinsics"].shape) == (6, 3, 3)
    assert tuple(s["lidar2img"].shape) == (6, 4, 4)
    assert s["lidar_points"].dtype == torch.float32 and s["lidar_points"].shape[1] == 5
    M = s["num_boxes"]
    assert tuple(s["gt_boxes"].shape) == (M, 7) and s["gt_boxes"].dtype == torch.float32
    assert tuple(s["gt_velocity"].shape) == (M, 2)
    assert s["gt_labels"].dtype == torch.int64 and tuple(s["gt_labels"].shape) == (M,)
    assert s["gt_in_range"].dtype == torch.bool
    # every str-list field has length M (identity fields included)
    for f in ("gt_names", "gt_ann_tokens", "gt_instance_tokens", "gt_attribute"):
        assert len(s[f]) == M, f
        assert all(isinstance(x, str) for x in s[f]), f
    if M:
        assert all(len(t) > 0 for t in s["gt_instance_tokens"])  # instance ids non-empty
        assert all(len(t) > 0 for t in s["gt_ann_tokens"])
        # labels in range 0..9
        assert int(s["gt_labels"].min()) >= 0 and int(s["gt_labels"].max()) <= 9


def test_instance_tokens_resolve_in_devkit(nusc_mini, mini_train_info, dataroot):
    """gt_instance_tokens are real devkit identities (resolve via nusc.get)."""
    ds = _ds(mini_train_info, dataroot, n=2)
    for i in range(len(ds)):
        for tok in ds[i]["gt_instance_tokens"]:
            assert nusc_mini.get("instance", tok)["token"] == tok


def test_committed_golden_image_sha256(nusc_mini, dataroot):
    """The pinned-decoder gate: the sentinel keyframe's decoded images hash to a
    COMMITTED golden (catches a future decoder swap — a within-run self-compare can't)."""
    tok = P.sentinel_token("v1.0-mini")
    info = IC.build_info_list(nusc_mini, [tok], dataroot)
    ds = DS.NuScenesMultimodalDataset(info, dataroot)
    assert DS.sample_image_sha256(ds[0]) == _GOLDEN_SENTINEL_IMG_SHA256


def test_images_native_no_resize_or_norm(mini_train_info, dataroot):
    ds = _ds(mini_train_info, dataroot, n=1)
    img = ds[0]["images"]
    assert img.dtype == torch.uint8 and int(img.max()) > 1  # not normalized to [0,1]
    assert img.shape[-2:] == (900, 1600)  # not resized


def test_bit_identical_sample_twice(mini_train_info, dataroot):
    ds = _ds(mini_train_info, dataroot, n=2)
    a, b = ds[0], ds[0]
    assert torch.equal(a["images"], b["images"])
    assert torch.equal(a["lidar_points"], b["lidar_points"])
    assert torch.equal(a["gt_boxes"], b["gt_boxes"])
    assert torch.equal(a["lidar2img"], b["lidar2img"])
    assert a["gt_ann_tokens"] == b["gt_ann_tokens"]


def test_boxes_sorted_by_ann_token(mini_train_info, dataroot):
    ds = _ds(mini_train_info, dataroot, n=3)
    for i in range(len(ds)):
        toks = ds[i]["gt_ann_tokens"]
        assert toks == sorted(toks)


def test_pinned_decoder_sha256_stable(mini_train_info, dataroot):
    """Decoded image bytes are reproducible (PIL convert-RGB pin); two decodes of
    the same keyframe hash identically."""
    ds = _ds(mini_train_info, dataroot, n=1)
    h1 = DS.sample_image_sha256(ds[0])
    h2 = DS.sample_image_sha256(ds[0])
    assert h1 == h2 and len(h1) == 64


def test_two_worker_batch_equals_zero_worker(mini_train_info, dataroot):
    ds = _ds(mini_train_info, dataroot, n=4)
    l0 = DS.make_loader(ds, batch_size=1, shuffle=False, num_workers=0, seed=42)
    l2 = DS.make_loader(ds, batch_size=1, shuffle=False, num_workers=2, seed=42)
    b0 = [batch[0] for batch in l0]
    b2 = [batch[0] for batch in l2]
    assert len(b0) == len(b2)
    for s0, s2 in zip(b0, b2):
        assert s0["sample_token"] == s2["sample_token"]
        assert torch.equal(s0["images"], s2["images"])
        assert torch.equal(s0["lidar_points"], s2["lidar_points"])
        assert torch.equal(s0["gt_boxes"], s2["gt_boxes"])


def test_gt_in_range_matches_devkit_distance_filter(nusc_mini, mini_train_info, dataroot):
    """gt_in_range exactly reproduces the devkit eval distance filter
    (box.ego_dist < class_range) — NOT a lidar-frame radial. We compute the devkit's
    ego_dist independently and compare the boolean per box."""
    by_token = {i["sample_token"]: i for i in mini_train_info}
    n = 0
    for tok, info in list(by_token.items())[:8]:
        s = nusc_mini.get("sample", tok)
        ep = nusc_mini.get("ego_pose",
                           nusc_mini.get("sample_data", s["data"]["LIDAR_TOP"])["ego_pose_token"])
        ego_xy = np.array(ep["translation"])[:2]
        for j, ann_tok in enumerate(info["gt_ann_tokens"]):
            ann = nusc_mini.get("sample_annotation", ann_tok)
            name = category_to_detection_name(ann["category_name"])
            ego_dist = float(np.linalg.norm(np.array(ann["translation"])[:2] - ego_xy))
            expected = ego_dist < CLASS_RANGE[name]
            assert bool(info["gt_in_range"][j]) == expected, (ann_tok, ego_dist, name)
            n += 1
    assert n > 50
