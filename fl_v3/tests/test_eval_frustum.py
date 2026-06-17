"""T4 — frustum visibility (ASR eligibility criterion 2). Deterministic; vs real mini geometry."""
from __future__ import annotations

import numpy as np

from fl_v3.eval.frustum_visibility import cams_visible, is_frustum_visible, DEFAULT_IMAGE_HW


def test_in_range_cars_mostly_visible(mini_train_info):
    """Most in-range cars project into ≥1 camera (the 6 cams cover 360°); a box far behind the
    sensor origin's coverage / out of any frustum is not visible. Sanity, not an exact count."""
    n_car = n_vis = 0
    for info in mini_train_info:
        for m in range(info["gt_boxes"].shape[0]):
            if info["gt_names"][m] != "car" or not bool(info["gt_in_range"][m]):
                continue
            n_car += 1
            if is_frustum_visible(info["gt_boxes"][m], info["lidar2img"], DEFAULT_IMAGE_HW):
                n_vis += 1
    assert n_car >= 50
    # 6 surround cameras → the large majority of in-range cars are visible in some view.
    assert n_vis / n_car > 0.8, f"only {n_vis}/{n_car} in-range cars frustum-visible"


def test_far_box_not_visible(mini_train_info):
    """A car box placed 500 m up (out of every camera frustum) is NOT visible."""
    info = mini_train_info[0]
    box = np.array([0.0, 0.0, 500.0, 4.0, 1.8, 1.5, 0.0])  # 500 m above the sensor
    assert not is_frustum_visible(box, info["lidar2img"], DEFAULT_IMAGE_HW)


def test_deterministic(mini_train_info):
    info = mini_train_info[0]
    box = info["gt_boxes"][0]
    a = cams_visible(box, info["lidar2img"], DEFAULT_IMAGE_HW)
    b = cams_visible(box, info["lidar2img"], DEFAULT_IMAGE_HW)
    assert a == b


def test_visible_box_in_front(mini_train_info):
    """A box placed ~15 m straight ahead (+x) at ground height is visible in the front camera."""
    info = mini_train_info[0]
    box = np.array([15.0, 0.0, -0.5, 4.0, 1.8, 1.5, 0.0])
    assert is_frustum_visible(box, info["lidar2img"], DEFAULT_IMAGE_HW)
