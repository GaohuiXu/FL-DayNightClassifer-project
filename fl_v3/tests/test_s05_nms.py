"""S05 deterministic task-wide circle/rotate NMS adversarial fixtures."""
from __future__ import annotations

import math

import pytest
import torch

from fl_v3.models.fusion.nms_deterministic import (
    circle_nms,
    rotate_nms,
    rotated_iou_bev,
)


def _box(x, y, length=4.0, width=2.0, yaw=0.0):
    return [x, y, 0.0, length, width, 1.5, yaw]


def test_circle_threshold_is_squared_metres_and_inclusive():
    boxes = torch.tensor([_box(0.0, 0.0), _box(2.0, 0.0), _box(2.01, 0.0)])
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.zeros(3, dtype=torch.long)
    flat = torch.tensor([0, 1, 2])
    keep = circle_nms(
        boxes, scores, labels, flat,
        threshold_sq_m=4.0, pre_max_size=1000, post_max_size=83,
    )
    # d=2 is suppressed because 2^2 <= 4; d=2.01 survives.
    assert keep.tolist() == [0, 2]


def test_circle_duplicate_keeps_deterministic_higher_priority_box():
    boxes = torch.tensor([_box(1.0, 1.0), _box(1.0, 1.0), _box(1.0, 1.0)])
    scores = torch.tensor([0.8, 0.9, 0.9])
    labels = torch.tensor([0, 0, 0])
    flat = torch.tensor([8, 7, 2])
    keep = circle_nms(boxes, scores, labels, flat, threshold_sq_m=4.0)
    assert keep.tolist() == [2]  # equal 0.9 tie resolves to smaller flat index


def test_rotated_iou_known_geometry():
    a = _box(0.0, 0.0, 4.0, 2.0, 0.0)
    assert rotated_iou_bev(a, a) == pytest.approx(1.0, abs=1e-12)
    assert rotated_iou_bev(a, _box(10.0, 0.0, 4.0, 2.0, 0.0)) == 0.0
    # Perpendicular 4x2 rectangles overlap in a 2x2 square: 4 / (8+8-4) = 1/3.
    assert rotated_iou_bev(a, _box(0.0, 0.0, 4.0, 2.0, math.pi / 2)) == pytest.approx(
        1.0 / 3.0, abs=1e-12
    )


def test_rotate_duplicate_suppression_is_task_wide_across_classes():
    boxes = torch.tensor([_box(0.0, 0.0), _box(0.0, 0.0)])
    scores = torch.tensor([0.8, 0.9])
    # Models the official truck + construction_vehicle shared task.
    labels = torch.tensor([1, 4])
    flat = torch.tensor([1, 9])
    keep = rotate_nms(boxes, scores, labels, flat, iou_threshold=0.2)
    assert keep.tolist() == [1]


def test_equal_score_class_then_spatial_tie_order_controls_nms():
    boxes = torch.tensor([_box(0.0, 0.0), _box(0.0, 0.0), _box(0.0, 0.0)])
    scores = torch.tensor([0.7, 0.7, 0.7])
    labels = torch.tensor([4, 1, 1])
    flat = torch.tensor([0, 8, 2])
    keep = rotate_nms(boxes, scores, labels, flat, iou_threshold=0.2)
    assert keep.tolist() == [2]  # global class 1, then flat 2


@pytest.mark.parametrize("kind", ["circle", "rotate"])
def test_nms_selected_content_is_invariant_to_input_permutation(kind):
    boxes = torch.tensor([
        _box(0.0, 0.0),
        _box(0.0, 0.0),
        _box(8.0, 0.0),
        _box(16.0, 0.0),
    ])
    scores = torch.tensor([0.8, 0.8, 0.7, 0.6])
    labels = torch.tensor([0, 0, 0, 0])
    flat = torch.tensor([5, 2, 10, 20])

    def run(order):
        b, s, l, f = boxes[order], scores[order], labels[order], flat[order]
        if kind == "circle":
            keep = circle_nms(b, s, l, f, threshold_sq_m=4.0)
        else:
            keep = rotate_nms(b, s, l, f, iou_threshold=0.2)
        return torch.cat((b[keep], s[keep, None], f[keep, None].to(b.dtype)), dim=1)

    identity = torch.arange(4)
    reverse = torch.arange(3, -1, -1)
    assert torch.equal(run(identity), run(reverse))


def test_rotate_nms_respects_pre_and_post_budgets():
    boxes = torch.tensor([_box(float(i) * 10.0, 0.0) for i in range(6)])
    scores = torch.linspace(0.9, 0.4, 6)
    labels = torch.zeros(6, dtype=torch.long)
    flat = torch.arange(6)
    keep = rotate_nms(
        boxes, scores, labels, flat,
        iou_threshold=0.2, pre_max_size=4, post_max_size=3,
    )
    assert keep.tolist() == [0, 1, 2]


def _run_exported_nms(kind, boxes, *, pre_max_size=1000, post_max_size=83):
    count = boxes.shape[0]
    scores = torch.linspace(0.9, 0.8, max(count, 1))[:count]
    labels = torch.zeros(count, dtype=torch.long)
    flat = torch.arange(count)
    if kind == "circle":
        return circle_nms(
            boxes, scores, labels, flat, threshold_sq_m=4.0,
            pre_max_size=pre_max_size, post_max_size=post_max_size,
        )
    return rotate_nms(
        boxes, scores, labels, flat, iou_threshold=0.2,
        pre_max_size=pre_max_size, post_max_size=post_max_size,
    )


@pytest.mark.parametrize("kind", ["circle", "rotate"])
@pytest.mark.parametrize(
    "bad_box",
    [
        _box(0.0, 0.0, yaw=float("nan")),
        _box(0.0, 0.0, length=0.0),
        _box(0.0, 0.0, width=-1.0),
    ],
)
def test_exported_nms_single_box_rejects_invalid_canonical_geometry(kind, bad_box):
    with pytest.raises(ValueError, match="finite|positive"):
        _run_exported_nms(kind, torch.tensor([bad_box]))


@pytest.mark.parametrize("kind", ["circle", "rotate"])
@pytest.mark.parametrize(
    "budget, value",
    [("pre_max_size", 0), ("pre_max_size", -1),
     ("post_max_size", 0), ("post_max_size", -1)],
)
def test_exported_nms_rejects_nonpositive_budgets(kind, budget, value):
    kwargs = {budget: value}
    with pytest.raises(ValueError, match="must be positive"):
        _run_exported_nms(kind, torch.tensor([_box(0.0, 0.0)]), **kwargs)
