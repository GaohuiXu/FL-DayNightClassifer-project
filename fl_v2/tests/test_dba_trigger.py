"""WS2 — DBA sub-trigger geometry tests (corner-grid + scattered-bars)."""

import torch

from fl_v2.attacks_defenses.attacks.pixel_backdoor import (
    _stamp_rects,
    _stamp_trigger,
    dba_subregions,
    make_pixel_trigger_fn,
)


# --- corner-grid pattern (legacy / null-config-compatible) -----------------

def test_corner_grid_1x1_is_full_corner_square():
    per_client, union = dba_subregions(4, "bottom-right", "1x1", 32, "corner-grid")
    assert union == [(28, 28, 32, 32)]
    assert per_client == union


def test_corner_grid_1x1_stamp_identical_to_pixel():
    """Null config: dba `corner-grid` `1x1` reproduces the plain pixel backdoor."""
    img = torch.zeros(3, 32, 32)
    per_client, _u = dba_subregions(4, "bottom-right", "1x1", 32, "corner-grid")
    dba_stamped = _stamp_rects(img, per_client, 1.0)
    pixel_stamped = _stamp_trigger(img, 4, 1.0, "bottom-right")
    assert torch.equal(dba_stamped, pixel_stamped)


def test_corner_grid_2x2_tiles_union_exactly():
    per_client, union = dba_subregions(4, "bottom-right", "2x2", 32, "corner-grid")
    assert len(per_client) == 4
    assert len(union) == 1
    r0, c0, r1, c1 = union[0]
    img = torch.zeros(3, 32, 32)
    tiled = _stamp_rects(img, per_client, 1.0)
    whole = _stamp_rects(img, union, 1.0)
    assert torch.equal(tiled, whole)
    area = sum((a[2] - a[0]) * (a[3] - a[1]) for a in per_client)
    assert area == (r1 - r0) * (c1 - c0)


def test_corner_grid_union_trigger_fn_matches_pixel_corner():
    batch = torch.zeros(2, 3, 32, 32)
    _pc, union = dba_subregions(4, "bottom-right", "2x2", 32, "corner-grid")
    dba_fn = make_pixel_trigger_fn(4, 1.0, "bottom-right", trigger_rects=union)
    pixel_fn = make_pixel_trigger_fn(4, 1.0, "bottom-right")
    assert torch.equal(dba_fn(batch), pixel_fn(batch))


def test_corner_grid_tiles_for_every_corner():
    for pos in ("bottom-right", "top-left", "bottom-left", "top-right"):
        per_client, union = dba_subregions(4, pos, "2x2", 32, "corner-grid")
        img = torch.zeros(3, 32, 32)
        assert torch.equal(
            _stamp_rects(img, per_client, 1.0),
            _stamp_rects(img, union, 1.0),
        ), pos


def test_corner_grid_uneven_grid_still_tiles():
    per_client, union = dba_subregions(4, "bottom-right", "3x3", 32, "corner-grid")
    assert len(per_client) == 9
    img = torch.zeros(3, 32, 32)
    assert torch.equal(
        _stamp_rects(img, per_client, 1.0),
        _stamp_rects(img, union, 1.0),
    )


# --- scattered-bars pattern (paper-faithful Xie et al. 2020 CIFAR) ---------

def test_scattered_bars_is_paper_cifar_pattern():
    """Four non-contiguous 1x6 horizontal bars with 3-px gaps."""
    expected = [(0, 0, 1, 6), (0, 9, 1, 15), (4, 0, 5, 6), (4, 9, 5, 15)]
    per_client, union = dba_subregions(4, "bottom-right", "2x2", 32, "scattered-bars")
    assert per_client == expected
    assert union == expected            # server stamps ALL 4 bars (gaps intact)


def test_scattered_bars_ignores_other_args():
    """The scattered-bars pattern is hardcoded — args other than pattern are ignored."""
    a, _ = dba_subregions(4, "bottom-right", "2x2", 32, "scattered-bars")
    b, _ = dba_subregions(8, "top-left", "3x3", 32, "scattered-bars")
    c, _ = dba_subregions(12, "bottom-left", "1x1", 32, "scattered-bars")
    assert a == b == c


def test_scattered_bars_stamp_leaves_gaps_clean():
    """The server stamp is the LIST of bars (not a bounding box) — gaps stay zero."""
    img = torch.zeros(3, 32, 32)
    _pc, union = dba_subregions(4, "bottom-right", "2x2", 32, "scattered-bars")
    stamped = _stamp_rects(img, union, 1.0)
    # Inside bar 0 (row 0, cols 0..5; rect (0,0,1,6), c1=6 exclusive)
    assert stamped[0, 0, 0].item() == 1.0
    assert stamped[0, 0, 5].item() == 1.0
    # Inside bar 1 (row 0, cols 9..14)
    assert stamped[0, 0, 9].item() == 1.0
    assert stamped[0, 0, 14].item() == 1.0
    # Inside bar 2 (row 4, cols 0..5)
    assert stamped[0, 4, 0].item() == 1.0
    # Gap between bars 0 and 1 (row 0, cols 6..8) — stays zero
    assert stamped[0, 0, 6].item() == 0.0
    assert stamped[0, 0, 7].item() == 0.0
    assert stamped[0, 0, 8].item() == 0.0
    # Outside the pattern entirely (rows 1..3, and any row > 4)
    assert stamped[0, 2, 3].item() == 0.0
    assert stamped[0, 10, 10].item() == 0.0


def test_scattered_bars_per_client_each_is_one_bar():
    """Each client's slot is exactly one of the 4 bars (1x6 = 6 pixels)."""
    per_client, _u = dba_subregions(4, "bottom-right", "2x2", 32, "scattered-bars")
    assert len(per_client) == 4
    for (r0, c0, r1, c1) in per_client:
        assert (r1 - r0) * (c1 - c0) == 6


# --- unknown pattern --------------------------------------------------------

def test_unknown_pattern_raises():
    try:
        dba_subregions(4, "bottom-right", "2x2", 32, "checkerboard")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown pattern")
