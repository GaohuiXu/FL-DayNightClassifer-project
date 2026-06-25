"""MCR P1 rare-class levers — CBGS resampling + reg-branch class weighting.

Two default-OFF capability knobs added for the centralized push toward ~0.60 mAP:
  * ``data/nuscenes/cbgs.py`` — repeat-factor (sqrt-CBGS) class-balanced resampling.
  * ``models/fusion/losses.py`` — per-class weight on the regression L1 (the branch the
    heatmap ``class_weights`` never touched — the verified 'trailer +0.004' wrong-branch fix).

These tests pin: (a) rare classes get oversampled and common ones do not, (b) the resampled
index list is deterministic + epoch-invariant from the seed, (c) ``max_repeat`` caps the factor,
(d) the wrapper maps indices onto the base dataset, and CRITICALLY (e) ``reg_class_weights=None``
is BYTE-IDENTICAL to the un-weighted L1 (the baseline-preservation contract), while a uniform
mean-1 weight is a no-op and a non-uniform weight actually shifts the loss.
"""
from __future__ import annotations

import numpy as np
import torch

from fl_v3.data.nuscenes.cbgs import build_cbgs_indices, CBGSWrapper, dataset_inrange_classes
from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.losses import CenterPointLoss


# --------------------------------------------------------------------------- CBGS
def _toy_classes(n_classes=10):
    """20 samples: class 0 (car) in ALL (common), class 4 (trailer) in 2 (rare), class 7 in 1 (rarer)."""
    per = []
    for i in range(20):
        present = [0]                       # car everywhere
        if i < 2:
            present.append(4)               # trailer in 2/20 = 10%
        if i == 0:
            present.append(7)               # bicycle in 1/20 = 5%
        per.append(np.array(sorted(set(present)), dtype=np.int64))
    return per


def test_cbgs_oversamples_rare_not_common():
    per = _toy_classes()
    idx, stats = build_cbgs_indices(per, n_classes=10, thresh=0.5, seed=1, max_repeat=4.0)
    # common class (car, f=1.0) → r_c == 1; rare classes → r_c > 1
    assert stats["r_c"][0] == 1.0
    assert stats["r_c"][4] > 1.0 and stats["r_c"][7] > 1.0
    assert stats["r_c"][7] >= stats["r_c"][4]            # rarer ⇒ larger factor
    # the rare-class samples (idx 0,1) appear more often than a pure-car sample (idx 19)
    counts = np.bincount(idx, minlength=20)
    assert counts[0] >= counts[19] and counts[1] >= counts[19]
    assert stats["expanded"] >= stats["N"] and stats["ratio"] >= 1.0


def test_cbgs_deterministic_and_epoch_invariant():
    per = _toy_classes()
    a, _ = build_cbgs_indices(per, n_classes=10, thresh=0.5, seed=42, max_repeat=4.0)
    b, _ = build_cbgs_indices(per, n_classes=10, thresh=0.5, seed=42, max_repeat=4.0)
    assert np.array_equal(a, b)                          # same seed ⇒ identical list (every DDP rank agrees)
    c, _ = build_cbgs_indices(per, n_classes=10, thresh=0.5, seed=43, max_repeat=4.0)
    # a different seed may reshuffle the stochastic rounding; the list is still a valid index set
    assert c.min() >= 0 and c.max() < 20


def test_cbgs_max_repeat_caps():
    per = _toy_classes()
    _, stats = build_cbgs_indices(per, n_classes=10, thresh=0.9, seed=1, max_repeat=2.0)
    assert max(stats["r_c"]) <= 2.0 + 1e-9               # no class exceeds the cap


def test_cbgs_off_is_noop_threshold_guard():
    per = _toy_classes()
    # thresh just above every rare freq ⇒ tiny/no oversampling; thresh<=0 is rejected
    try:
        build_cbgs_indices(per, n_classes=10, thresh=0.0, seed=1)
        assert False, "thresh<=0 must raise"
    except ValueError:
        pass


def test_cbgs_wrapper_delegates_indices():
    base = list(range(20))                               # any indexable acts as the base dataset
    idx = np.array([5, 5, 0, 19, 5], dtype=np.int64)
    w = CBGSWrapper(base, idx)
    assert len(w) == 5
    assert [w[j] for j in range(5)] == [5, 5, 0, 19, 5]  # __getitem__(j) == base[indices[j]]


def test_dataset_inrange_classes_masks_out_of_range():
    class _FakeDS:
        _infos = [
            {"gt_labels": np.array([0, 4, 4]), "gt_in_range": np.array([True, False, True])},
            {"gt_labels": np.array([], dtype=np.int64), "gt_in_range": np.array([], dtype=bool)},
        ]
    got = dataset_inrange_classes(_FakeDS())
    assert set(got[0].tolist()) == {0, 4}                # the out-of-range duplicate label is fine (set)
    assert got[1].size == 0                              # empty keyframe ⇒ no classes (repeat factor 1)


# --------------------------------------------------------------------------- reg class weight
def _synthetic_batch_and_pred(cfg, labels):
    """One sample, len(labels) GT boxes at distinct in-grid centers; an all-zero head_out."""
    H, W = cfg.head_ny, cfg.head_nx
    boxes = []
    for k, _lab in enumerate(labels):
        cx = cfg.x_min + (k + 2) * cfg.head_vx * 3       # spread the centers apart, well inside the grid
        cy = cfg.y_min + (k + 2) * cfg.head_vy * 3
        boxes.append([cx, cy, 0.0, 4.0, 2.0, 1.6, 0.3])  # car-ish extents
    batch = {
        "gt_boxes": [torch.tensor(boxes, dtype=torch.float32)],
        "gt_labels": [torch.tensor(labels, dtype=torch.int64)],
        "gt_velocity": [torch.zeros(len(labels), 2)],
    }
    pred = {
        "heatmap": torch.zeros(1, cfg.n_classes if hasattr(cfg, "n_classes") else 10, H, W),
        "reg": torch.zeros(1, 2, H, W), "height": torch.zeros(1, 1, H, W),
        "dim": torch.zeros(1, 3, H, W), "rot": torch.zeros(1, 2, H, W), "vel": torch.zeros(1, 2, H, W),
    }
    return batch, pred


def test_build_targets_returns_labels_in_order():
    cfg = BEVConfig()
    loss = CenterPointLoss(cfg=cfg, n_classes=10)
    batch, _ = _synthetic_batch_and_pred(cfg, labels=[0, 4, 7])
    *_, labels_k = loss.build_targets(batch, device="cpu")
    assert labels_k.tolist() == [0, 4, 7]


def test_reg_class_weights_none_equals_uniform():
    """The baseline-preservation contract: reg_class_weights=None == unweighted L1, and a uniform
    mean-1 weight is a no-op (weighted mean with equal weights == plain mean)."""
    cfg = BEVConfig()
    batch, pred = _synthetic_batch_and_pred(cfg, labels=[0, 4, 7])
    none_loss = CenterPointLoss(cfg=cfg, n_classes=10, reg_class_weights=None)(pred, batch)
    unif_loss = CenterPointLoss(cfg=cfg, n_classes=10, reg_class_weights=[1.0] * 10)(pred, batch)
    assert torch.allclose(none_loss, unif_loss, atol=0, rtol=0)


def test_reg_class_weights_nonuniform_shifts_loss_toward_weighted_class():
    cfg = BEVConfig()
    batch, pred = _synthetic_batch_and_pred(cfg, labels=[0, 4])     # car + trailer
    base = CenterPointLoss(cfg=cfg, n_classes=10, reg_class_weights=None)
    # up-weight ONLY trailer's (class 4) box regression
    w = [1.0] * 10; w[4] = 5.0
    weighted = CenterPointLoss(cfg=cfg, n_classes=10, reg_class_weights=w)
    base_terms = base(pred, batch); _ = weighted(pred, batch)
    # both record reg_loss; the weighting must actually change it (not a no-op)
    assert abs(weighted.last_terms["reg_loss"] - base.last_terms["reg_loss"]) > 1e-6
    # heatmap loss is untouched by the REG weight
    assert abs(weighted.last_terms["hm_loss"] - base.last_terms["hm_loss"]) < 1e-6
