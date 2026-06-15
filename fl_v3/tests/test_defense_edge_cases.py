"""Edge-case branch coverage the single oracle fixture does not exercise.

These are deterministic, well-defined behaviors (degenerate sizes, active
clipping, odd-N median, single-client FoolsGold) — self-consistency unit tests,
not oracle parity. They close the branches the adversarial verification flagged
as fixture-uncovered.
"""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.defenses.fed_median import fed_median_decision
from fl_v3.strategy.defenses.fedavg import norm_clip_decision
from fl_v3.strategy.defenses.flame import flame_cluster_admitted, flame_decision
from fl_v3.strategy.defenses.foolsgold import foolsgold_weights
from fl_v3.strategy.gradient_metrics import clip_updates_by_l2_norm, compute_update_norms


def test_flame_n_lt_4_admits_all():
    # Degenerate branch: n < 4 -> admit all, skip HDBSCAN.
    g = [np.zeros((3,), dtype=np.float32)]
    clients = [[np.array([1.0, 0.0, 0.0], np.float32)],
               [np.array([0.0, 1.0, 0.0], np.float32)],
               [np.array([0.0, 0.0, 1.0], np.float32)]]
    norms = np.asarray(compute_update_norms(g, clients), dtype=np.float64)
    from fl_v3.strategy.defenses.base import build_flat_updates

    admitted = flame_cluster_admitted(build_flat_updates(g, clients), norms)
    assert admitted == [0, 1, 2]
    # And the full decision admits all three with clip coefs summing sensibly.
    dec = flame_decision(g, clients, noise_multiplier=0.0, seed=1, server_round=1)
    assert dec.admitted == [0, 1, 2]
    assert dec.diagnostics["n_admitted"] == 3


def test_flame_small_majority_cluster():
    # n=4: three tightly-clustered honest + one outlier. min_cluster_size=3 ->
    # the 3 form the benign cluster, the outlier is dropped.
    g = [np.zeros((4,), dtype=np.float32)]
    base = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
    clients = [
        [(base + 0.001 * np.array([0, 0, 0, 0], np.float32))],
        [(base + 0.001 * np.array([1, 0, 0, 0], np.float32))],
        [(base + 0.001 * np.array([0, 1, 0, 0], np.float32))],
        [np.array([-1.0, -1.0, 1.0, -1.0], np.float32)],  # outlier direction
    ]
    norms = np.asarray(compute_update_norms(g, clients), dtype=np.float64)
    from fl_v3.strategy.defenses.base import build_flat_updates

    admitted = flame_cluster_admitted(build_flat_updates(g, clients), norms)
    assert set(admitted) == {0, 1, 2}
    assert 3 not in admitted


def test_norm_clip_actually_clips_when_scale_lt_1():
    # clip_norm well below the update norm -> scale < 1, norm bounded.
    g = [np.zeros((4,), dtype=np.float32)]
    big = [[np.array([2.0, 0.0, 0.0, 0.0], np.float32)]]  # update norm 2.0
    clip_norm = 0.5
    clipped, orig, clipped_norms = clip_updates_by_l2_norm(g, big, clip_norm)
    assert orig[0] == 2.0
    assert clipped_norms[0] <= clip_norm + 1e-6
    np.testing.assert_allclose(clipped[0][0], [0.5, 0.0, 0.0, 0.0], atol=1e-6)


def test_norm_clip_decision_aggregates_clipped_mean():
    g = [np.zeros((2,), dtype=np.float32)]
    clients = [[np.array([2.0, 0.0], np.float32)], [np.array([0.0, 2.0], np.float32)]]
    dec = norm_clip_decision(g, clients, clip_norm=0.5)  # each clips to norm 0.5
    # uniform mean of the two clipped updates ([0.5,0],[0,0.5]) -> [0.25,0.25].
    np.testing.assert_allclose(dec.new_global[0], [0.25, 0.25], atol=1e-6)


def test_fed_median_odd_n():
    g = [np.zeros((1,), dtype=np.float32)]
    clients = [[np.array([1.0], np.float32)],
               [np.array([2.0], np.float32)],
               [np.array([3.0], np.float32)]]
    dec = fed_median_decision(g, clients)
    np.testing.assert_allclose(dec.new_global[0], [2.0], atol=0)  # exact middle


def test_foolsgold_single_client():
    assert foolsgold_weights([np.array([1.0, 2.0, 3.0])]) == [1.0]
