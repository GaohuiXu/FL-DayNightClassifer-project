"""WS2 — DBA sub-trigger geometry tests.

Covers the `dba_subregions` helper: the null config (grid '1x1' is
bit-identical to the plain pixel backdoor), exact disjoint tiling of the
trigger square, and the train/eval single-source-of-truth invariant
(the server union trigger equals the plain corner trigger).
"""

import torch

from fl_v2.attacks_defenses.attacks.pixel_backdoor import (
    _stamp_rects,
    _stamp_trigger,
    dba_subregions,
    make_pixel_trigger_fn,
)


def test_dba_1x1_union_is_full_corner_square():
    per_client, union = dba_subregions(4, "bottom-right", "1x1", 32)
    assert union == (28, 28, 32, 32)
    assert per_client == [union]


def test_dba_1x1_stamp_identical_to_pixel():
    """Null config: dba-grid '1x1' reproduces the plain pixel backdoor."""
    img = torch.zeros(3, 32, 32)
    per_client, _union = dba_subregions(4, "bottom-right", "1x1", 32)
    dba_stamped = _stamp_rects(img, per_client, 1.0)
    pixel_stamped = _stamp_trigger(img, 4, 1.0, "bottom-right")
    assert torch.equal(dba_stamped, pixel_stamped)


def test_dba_2x2_tiles_union_exactly():
    per_client, union = dba_subregions(4, "bottom-right", "2x2", 32)
    assert len(per_client) == 4
    r0, c0, r1, c1 = union
    img = torch.zeros(3, 32, 32)
    # Stamping every sub-rect once == stamping the union once  =>
    # the sub-rects tile the union with no gaps and no overlaps.
    tiled = _stamp_rects(img, per_client, 1.0)
    whole = _stamp_rects(img, [union], 1.0)
    assert torch.equal(tiled, whole)
    # Disjoint: summed sub-rect area equals the union area.
    area = sum((a[2] - a[0]) * (a[3] - a[1]) for a in per_client)
    assert area == (r1 - r0) * (c1 - c0)


def test_dba_union_trigger_fn_matches_pixel_corner():
    """server-side: the DBA union trigger_fn == the plain corner trigger_fn."""
    batch = torch.zeros(2, 3, 32, 32)
    _pc, union = dba_subregions(4, "bottom-right", "2x2", 32)
    dba_fn = make_pixel_trigger_fn(4, 1.0, "bottom-right", trigger_rects=[union])
    pixel_fn = make_pixel_trigger_fn(4, 1.0, "bottom-right")
    assert torch.equal(dba_fn(batch), pixel_fn(batch))


def test_dba_subregions_tiles_for_every_corner():
    for pos in ("bottom-right", "top-left", "bottom-left", "top-right"):
        per_client, union = dba_subregions(4, pos, "2x2", 32)
        img = torch.zeros(3, 32, 32)
        assert torch.equal(
            _stamp_rects(img, per_client, 1.0),
            _stamp_rects(img, [union], 1.0),
        ), pos


def test_dba_uneven_grid_still_tiles():
    """A grid that does not divide the trigger evenly still tiles exactly."""
    per_client, union = dba_subregions(4, "bottom-right", "3x3", 32)
    assert len(per_client) == 9
    img = torch.zeros(3, 32, 32)
    assert torch.equal(
        _stamp_rects(img, per_client, 1.0),
        _stamp_rects(img, [union], 1.0),
    )
