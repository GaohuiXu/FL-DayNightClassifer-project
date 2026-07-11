"""S02 golden and adversarial checks for pillar caps and CenterPoint targets.

The independent Gaussian oracle below is transcribed from:

* MIT BEVFusion ``326653dc06e0938edf1aae7d01efcd158ba83de5``
  ``mmdet3d/core/utils/gaussian.py`` and CenterHead target construction; and
* CenterPoint v0.2 ``e9ef04c3715aa3342fa42f4f4e064db987def6ad``
  ``det3d/core/utils/center_utils.py``.

It deliberately preserves the official constant ``/2`` roots. It is not a geometric
``/(2*a)`` re-derivation.
"""
from __future__ import annotations

import math

import pytest
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder
from fl_v3.models.fusion.losses import (
    CenterPointLoss,
    draw_gaussian,
    gaussian_2d,
    gaussian_radius,
)


def _official_roots(det_size, min_overlap=0.1):
    """Independent literal transcription of the three official candidate roots."""
    height, width = (float(v) for v in det_size)
    a1, b1 = 1.0, height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    r1 = (b1 + math.sqrt(b1**2 - 4.0 * a1 * c1)) / 2.0

    a2, b2 = 4.0, 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(b2**2 - 4.0 * a2 * c2)) / 2.0

    a3, b3 = 4.0 * min_overlap, -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    r3 = (b3 + math.sqrt(b3**2 - 4.0 * a3 * c3)) / 2.0
    return r1, r2, r3


@pytest.mark.parametrize(
    ("det_size", "expected_roots", "expected_radius", "old_mixed_radius"),
    [
        ((1.0, 1.0), (1.426401432711221, 2.632455532033676, 0.432455532033676), 2, 2),
        ((4.0, 8.0), (9.133397807202561, 17.366563145999496, 2.4), 2, 4),
        ((10.0, 20.0), (22.833494518006404, 43.416407864998739, 6.0), 6, 10),
        ((6.0, 16.0), (17.515715268068845, 33.764352935882194, 4.076941930590086), 4, 8),
    ],
)
def test_official_gaussian_radius_numerical_goldens(
    det_size, expected_roots, expected_radius, old_mixed_radius
):
    roots = _official_roots(det_size)
    assert roots == pytest.approx(expected_roots, rel=0.0, abs=1e-12)
    actual = gaussian_radius(det_size, min_overlap=0.1)
    assert actual == pytest.approx(min(expected_roots), rel=0.0, abs=1e-12)
    assert max(2, int(actual)) == expected_radius
    # Keep the discriminating legacy value visible: these cases must not drift back to it.
    if expected_radius != old_mixed_radius:
        assert max(2, int(actual)) != old_mixed_radius


_RADIUS2_PATCH = torch.tensor(
    [
        [0.003151111537590623, 0.02732372283935547, 0.05613476410508156, 0.02732372283935547, 0.003151111537590623],
        [0.02732372283935547, 0.23692776262760162, 0.4867522418498993, 0.23692776262760162, 0.02732372283935547],
        [0.05613476410508156, 0.4867522418498993, 1.0, 0.4867522418498993, 0.05613476410508156],
        [0.02732372283935547, 0.23692776262760162, 0.4867522418498993, 0.23692776262760162, 0.02732372283935547],
        [0.003151111537590623, 0.02732372283935547, 0.05613476410508156, 0.02732372283935547, 0.003151111537590623],
    ],
    dtype=torch.float32,
)

_TARGET_7X8 = torch.tensor(
    [
        [1.0, 0.4867522418498993, 0.05613476410508156, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.4867522418498993, 0.23692776262760162, 0.02732372283935547, 0.05613476410508156, 0.02732372283935547, 0.003151111537590623, 0.0, 0.0],
        [0.05613476410508156, 0.02732372283935547, 0.23692776262760162, 0.4867522418498993, 0.23692776262760162, 0.02732372283935547, 0.0, 0.0],
        [0.0, 0.05613476410508156, 0.4867522418498993, 1.0, 0.4867522418498993, 0.05613476410508156, 0.0, 0.0],
        [0.0, 0.02732372283935547, 0.23692776262760162, 0.4867522418498993, 0.23692776262760162, 0.02732372283935547, 0.0, 0.0],
        [0.0, 0.003151111537590623, 0.02732372283935547, 0.05613476410508156, 0.02732372283935547, 0.003151111537590623, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def test_official_gaussian_patch_and_clipped_max_overlay_exact():
    assert torch.equal(gaussian_2d(2), _RADIUS2_PATCH)
    heatmap = torch.zeros((7, 8), dtype=torch.float32)
    draw_gaussian(heatmap, col=3, row=3, radius=2)
    draw_gaussian(heatmap, col=0, row=0, radius=2)
    assert torch.equal(heatmap, _TARGET_7X8)


def test_centerpoint_target_render_matches_official_fixture_and_order():
    cfg = BEVConfig(
        point_cloud_range=(0.0, 0.0, -2.0, 8.0, 7.0, 2.0),
        bev_voxel=(1.0, 1.0),
        out_size_factor=1,
    )
    loss = CenterPointLoss(cfg=cfg, n_classes=1)
    # dx/head_vx=4 and dy/head_vy=8 -> official roots min=2.4 -> int=2.
    boxes = torch.tensor(
        [
            [3.25, 3.75, 0.0, 4.0, 8.0, 1.0, 0.0],
            [0.00, 0.00, 0.0, 4.0, 8.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.zeros((2,), dtype=torch.int64)
    target = loss.build_targets(
        {"gt_boxes": [boxes], "gt_labels": [labels]}, device="cpu"
    )[0][0, 0]
    assert torch.equal(target, _TARGET_7X8)

    reversed_target = loss.build_targets(
        {"gt_boxes": [boxes.flip(0)], "gt_labels": [labels]}, device="cpu"
    )[0][0, 0]
    assert torch.equal(reversed_target, _TARGET_7X8)


_PILLAR_CFG = BEVConfig(
    point_cloud_range=(0.0, 0.0, -1.0, 6.0, 4.0, 1.0),
    bev_voxel=(1.0, 1.0),
    out_size_factor=1,
)


def _sample_points(cell_counts: list[tuple[int, int, int]], batch_idx: int = 0) -> torch.Tensor:
    rows = []
    for col, row, count in cell_counts:
        for point_idx in range(count):
            # Canonical content differs but remains strictly inside the declared cell.
            jitter = 0.01 * point_idx
            rows.append(
                [
                    float(batch_idx),
                    col + 0.10 + jitter,
                    row + 0.20 + jitter,
                    -0.20 + 0.01 * point_idx,
                    0.05 * (point_idx + 1),
                    0.0,
                ]
            )
    return torch.tensor(rows, dtype=torch.float32)


def _with_batch(points: torch.Tensor, batch_idx: int) -> torch.Tensor:
    out = points.clone()
    out[:, 0] = float(batch_idx)
    return out


def _encoder() -> PointPillarsEncoder:
    torch.manual_seed(17)
    return PointPillarsEncoder(
        out_channels=32,
        max_points=3,
        max_pillars=2,
        cfg=_PILLAR_CFG,
    ).eval()


def _as_list(meta: dict[str, object], key: str):
    value = meta[key]
    assert isinstance(value, torch.Tensor)
    return value.cpu().tolist()


def test_per_sample_caps_selection_and_observable_diagnostics():
    # A: keys 0,1,2; selected key 0 has six points and key 1 has two.
    sample_a = _sample_points([(0, 0, 6), (1, 0, 2), (2, 0, 4)])
    # B: keys 4,6,7,8; selected keys 4 and 6, with key 6 over the point cap.
    sample_b = _sample_points([(4, 0, 1), (0, 1, 5), (1, 1, 1), (2, 1, 2)])
    points = torch.cat([_with_batch(sample_a, 0), _with_batch(sample_b, 1)])

    enc = _encoder()
    enc(points, B=2)
    meta = enc.last_pillar_meta
    assert _as_list(meta, "input_points_per_sample") == [12, 9]
    assert _as_list(meta, "in_range_points_per_sample") == [12, 9]
    assert _as_list(meta, "occupied_pillars_per_sample") == [3, 4]
    assert _as_list(meta, "selected_pillars_per_sample") == [2, 2]
    assert _as_list(meta, "truncated_pillars_per_sample") == [1, 2]
    assert _as_list(meta, "points_kept_after_caps_per_sample") == [5, 4]
    assert _as_list(meta, "points_dropped_by_point_cap_per_sample") == [3, 2]
    assert _as_list(meta, "points_dropped_by_pillar_cap_per_sample") == [4, 3]
    assert _as_list(meta, "selected_pillar_batch_ids") == [0, 0, 1, 1]
    assert _as_list(meta, "selected_local_pillar_keys") == [0, 1, 4, 6]
    assert _as_list(meta, "pillar_truncation_fraction_per_sample") == pytest.approx(
        [1.0 / 3.0, 0.5]
    )
    assert all(v <= enc.max_pillars for v in _as_list(meta, "selected_pillars_per_sample"))

    occupancy = enc.occupancy(points, B=2)
    assert occupancy[0, 0, 0].item() == 6
    assert occupancy[0, 0, 2].item() == 4
    assert occupancy[1, 1, 0].item() == 5


def test_b1_batched_sample_isolation_and_batch_permutation():
    sample_a = _sample_points([(0, 0, 6), (1, 0, 2), (2, 0, 4)])
    sample_b = _sample_points([(4, 0, 1), (0, 1, 5), (1, 1, 1), (2, 1, 2)])
    enc = _encoder()

    isolated_a = enc(_with_batch(sample_a, 0), B=1)[0].clone()
    isolated_a_meta = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in enc.last_pillar_meta.items()
    }
    batched = enc(
        torch.cat([_with_batch(sample_a, 0), _with_batch(sample_b, 1)]), B=2
    )
    assert torch.equal(batched[0], isolated_a)
    assert _as_list(isolated_a_meta, "occupied_pillars_per_sample") == [3]
    assert _as_list(isolated_a_meta, "selected_pillars_per_sample") == [2]

    permuted = enc(
        torch.cat([_with_batch(sample_b, 0), _with_batch(sample_a, 1)]), B=2
    )
    assert torch.equal(permuted[1], batched[0])
    assert torch.equal(permuted[0], batched[1])
    assert _as_list(enc.last_pillar_meta, "occupied_pillars_per_sample") == [4, 3]
    assert _as_list(enc.last_pillar_meta, "truncated_pillars_per_sample") == [2, 1]


def test_point_and_pillar_overcap_are_input_permutation_invariant():
    sample = _sample_points([(0, 0, 8), (1, 0, 5), (2, 0, 4), (3, 0, 2)])
    enc = _encoder()
    original = enc(sample, B=1)
    original_keys = _as_list(enc.last_pillar_meta, "selected_local_pillar_keys")

    generator = torch.Generator().manual_seed(2027)
    shuffled = enc(sample[torch.randperm(sample.shape[0], generator=generator)], B=1)
    assert torch.equal(shuffled, original)
    assert _as_list(enc.last_pillar_meta, "selected_local_pillar_keys") == original_keys
    assert _as_list(enc.last_pillar_meta, "selected_pillars_per_sample") == [2]
    assert _as_list(enc.last_pillar_meta, "points_dropped_by_point_cap_per_sample") == [7]
    assert _as_list(enc.last_pillar_meta, "points_dropped_by_pillar_cap_per_sample") == [6]


def test_empty_batch_and_empty_samples_have_zero_diagnostics():
    enc = _encoder()
    empty = torch.empty((0, 6), dtype=torch.float32)
    bev = enc(empty, B=3)
    assert torch.count_nonzero(bev).item() == 0
    for key in (
        "input_points_per_sample",
        "in_range_points_per_sample",
        "occupied_pillars_per_sample",
        "selected_pillars_per_sample",
        "truncated_pillars_per_sample",
        "points_kept_after_caps_per_sample",
        "points_dropped_by_point_cap_per_sample",
        "points_dropped_by_pillar_cap_per_sample",
    ):
        assert _as_list(enc.last_pillar_meta, key) == [0, 0, 0]

    sample = _sample_points([(0, 0, 2), (1, 0, 1)], batch_idx=1)
    bev = enc(sample, B=3)
    assert torch.count_nonzero(bev[0]).item() == 0
    assert torch.count_nonzero(bev[2]).item() == 0
    assert _as_list(enc.last_pillar_meta, "occupied_pillars_per_sample") == [0, 2, 0]
    assert _as_list(enc.last_pillar_meta, "selected_pillars_per_sample") == [0, 2, 0]
