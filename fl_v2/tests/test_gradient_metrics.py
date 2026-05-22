"""Unit tests for the Cycle-02 gradient-space metric instrumentation.

Covers `compute_gradient_space_metrics` — the per-client per-round
diagnostic used to fill the evasion matrix. The function is logging-only
and must be deterministic and numerically sane.
"""
from __future__ import annotations

import numpy as np

from fl_v2.attacks_defenses import compute_gradient_space_metrics


def _params(*arrays) -> list[np.ndarray]:
    """Build a parameter list (list of float32 ndarrays)."""
    return [np.asarray(a, dtype=np.float32) for a in arrays]


def test_empty_client_list_returns_empty_metrics():
    out = compute_gradient_space_metrics(_params([0.0, 0.0]), [], topk_k=2)
    assert out["cosine_to_mean"] == []
    assert out["pairwise_cosine"] == []
    assert out["topk_energy_frac"] == []
    assert out["topk_energy_k"] == 0


def test_identical_updates_give_cosine_one():
    g = _params([0.0, 0.0, 0.0, 0.0])
    c = _params([1.0, 2.0, 3.0, 4.0])
    out = compute_gradient_space_metrics(g, [c, c, c], topk_k=2)
    for v in out["cosine_to_mean"]:
        assert abs(v - 1.0) < 1e-5
    for row in out["pairwise_cosine"]:
        for v in row:
            assert abs(v - 1.0) < 1e-5


def test_opposite_update_gives_negative_cosine():
    g = _params([0.0, 0.0, 0.0, 0.0])
    aligned = _params([1.0, 1.0, 1.0, 1.0])
    opposite = _params([-1.0, -1.0, -1.0, -1.0])
    out = compute_gradient_space_metrics(
        g, [aligned, aligned, opposite], topk_k=2
    )
    # mean update points with the aligned majority
    assert out["cosine_to_mean"][0] > 0.99
    assert out["cosine_to_mean"][1] > 0.99
    assert out["cosine_to_mean"][2] < -0.99
    # pairwise: aligned-vs-opposite ~ -1, aligned-vs-aligned ~ +1
    assert out["pairwise_cosine"][0][2] < -0.99
    assert out["pairwise_cosine"][0][1] > 0.99


def test_topk_energy_fraction_is_bounded():
    g = _params(np.zeros(10))
    c1 = _params(np.arange(10.0))
    c2 = _params(np.ones(10))
    out = compute_gradient_space_metrics(g, [c1, c2], topk_k=3)
    assert out["topk_energy_k"] == 3
    for f in out["topk_energy_frac"]:
        assert 0.0 <= f <= 1.0 + 1e-6


def test_topk_k_clamps_to_param_count():
    g = _params(np.zeros(10))
    c1 = _params(np.arange(10.0))
    c2 = _params(np.ones(10))
    out = compute_gradient_space_metrics(g, [c1, c2], topk_k=9999)
    assert out["topk_energy_k"] == 10  # clamped to d
    # every coordinate is in the top-k -> all energy is captured
    for f in out["topk_energy_frac"]:
        assert abs(f - 1.0) < 1e-5


def test_zero_update_is_degenerate_sentinel():
    g = _params([1.0, 2.0, 3.0])
    zero_client = _params([1.0, 2.0, 3.0])   # update == 0
    real_client = _params([2.0, 3.0, 4.0])   # update == [1, 1, 1]
    out = compute_gradient_space_metrics(
        g, [zero_client, real_client], topk_k=2
    )
    assert abs(out["cosine_to_mean"][0]) < 1e-6
    assert abs(out["topk_energy_frac"][0]) < 1e-6


def test_deterministic_across_calls():
    rng = np.random.default_rng(0)
    g = _params(rng.standard_normal(64), rng.standard_normal((8, 8)))
    clients = [
        _params(rng.standard_normal(64), rng.standard_normal((8, 8)))
        for _ in range(6)
    ]
    a = compute_gradient_space_metrics(g, clients, topk_k=16)
    b = compute_gradient_space_metrics(g, clients, topk_k=16)
    assert a["cosine_to_mean"] == b["cosine_to_mean"]
    assert a["pairwise_cosine"] == b["pairwise_cosine"]
    assert a["topk_energy_frac"] == b["topk_energy_frac"]
