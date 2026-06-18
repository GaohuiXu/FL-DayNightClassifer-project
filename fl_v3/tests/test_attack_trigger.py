"""T5 trigger generator: one-generator golden, localized blit, objective placement, budget (§0.C2)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from fl_v3.attacks import trigger as TR
from _attack_fixtures import attack_sample


def test_points_in_box7_counts_enclosed_cluster():
    s = attack_sample(0)
    box = s["gt_boxes"][0].numpy()
    inside = TR.points_in_box7(s["lidar_points"].numpy(), box)
    assert int(inside.sum()) == 40  # exactly the box-0 cluster


def test_aligned_placement_objective_and_budget():
    s = attack_sample(0)
    spec = TR.TriggerSpec()
    box = s["gt_boxes"][0].numpy()
    pl = TR.compute_aligned_placement(box, s["lidar_points"][:, :3].numpy(), s["lidar2img"].numpy(), spec)
    assert pl is not None and pl.aligned
    # objective test: cond-4 centre ≤ center_max_px of the box-centre projection
    off = float(np.hypot(pl.cx_px - pl.box_center_uv[0], pl.cy_px - pl.box_center_uv[1]))
    assert off <= spec.center_max_px + 1e-6
    # trigger budget: patch area ≤ budget_frac of the projected 2-D box (§0.C2)
    assert pl.patch_area / pl.box2d_area <= spec.budget_frac + 1e-9


def test_nonaligned_placement_iou0_with_targets():
    s = attack_sample(0)
    spec = TR.TriggerSpec()
    box = s["gt_boxes"][0].numpy()
    pl = TR.compute_nonaligned_placement(box, s["gt_boxes"].numpy(), s["lidar2img"].numpy(), spec)
    assert pl is not None and not pl.aligned
    # its 2-D footprint must have IoU==0 with every target's projected 2-D box
    l2i = s["lidar2img"].numpy()[pl.cam]
    from fl_v3.attacks.trigger import _box2d_from_corners, _iou_2d
    patch = (pl.cx_px - pl.half, pl.cy_px - pl.half, pl.cx_px + pl.half, pl.cy_px + pl.half)
    for tb in s["gt_boxes"].numpy():
        b2 = _box2d_from_corners(tb, l2i, TR.NATIVE_IMAGE_HW, spec.depth_min)
        if b2 is not None:
            assert _iou_2d(patch, b2) == 0.0


def test_one_generator_train_equals_inference_golden_sha():
    s = attack_sample(1)
    spec = TR.TriggerSpec()
    box = s["gt_boxes"][0].numpy()
    pl = TR.compute_aligned_placement(box, s["lidar_points"][:, :3].numpy(), s["lidar2img"].numpy(), spec)
    a = TR.apply_trigger(s, pl, spec)
    b = TR.apply_trigger(s, pl, spec)
    assert TR.trigger_image_sha256(a) == TR.trigger_image_sha256(b)  # deterministic, no RNG


def test_trigger_localized_to_rect():
    s = attack_sample(2)
    spec = TR.TriggerSpec()
    box = s["gt_boxes"][0].numpy()
    pl = TR.compute_aligned_placement(box, s["lidar_points"][:, :3].numpy(), s["lidar2img"].numpy(), spec)
    t = TR.apply_trigger(s, pl, spec)
    TR.assert_trigger_localized(s, t, pl)  # raises if any pixel outside the rect changed
    # other cameras are byte-identical
    for ci in range(6):
        if ci != pl.cam:
            assert torch.equal(s["images"][ci], t["images"][ci])


def test_placement_is_deterministic():
    s = attack_sample(3)
    spec = TR.TriggerSpec()
    box = s["gt_boxes"][0].numpy()
    p1 = TR.compute_aligned_placement(box, s["lidar_points"][:, :3].numpy(), s["lidar2img"].numpy(), spec)
    p2 = TR.compute_aligned_placement(box, s["lidar_points"][:, :3].numpy(), s["lidar2img"].numpy(), spec)
    assert (p1.cam, p1.cx_px, p1.cy_px, p1.half) == (p2.cam, p2.cx_px, p2.cy_px, p2.half)


def test_spec_budget_invariant():
    with pytest.raises(ValueError):
        TR.TriggerSpec(area_frac=0.5, budget_frac=0.3)  # area_frac must be ≤ budget_frac
