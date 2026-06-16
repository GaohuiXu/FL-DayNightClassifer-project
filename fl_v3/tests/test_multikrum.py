"""Multi-Krum correctness on a hand-computed textbook fixture + validity gate.

fl_v2 used Flower's built-in MultiKrum (no fl_v2 source oracle), so correctness
is established here against by-hand Krum scores rather than an fl_v2 fixture.

Setup: 5 clients at scalar positions {0,1,2,10,11}, global=0, f=1 →
num_closest = n-f-2 = 2. By hand:
    scores = [5, 2, 5, 65, 82]
Krum (m=1) → index 1; Multi-Krum (m=3) → {0,1,2}, average = 1.0.
"""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.defenses.multi_krum import (
    krum_scores,
    multi_krum_decision,
    multi_krum_valid,
)

GLOBAL = [np.array([0.0], dtype=np.float32)]
CLIENTS = [[np.array([v], dtype=np.float32)] for v in (0.0, 1.0, 2.0, 10.0, 11.0)]


def test_krum_scores_hand_computed():
    from fl_v3.strategy.defenses.base import build_flat_updates

    flat = build_flat_updates(GLOBAL, CLIENTS)
    scores = krum_scores(flat, f=1)
    np.testing.assert_allclose(scores, [5.0, 2.0, 5.0, 65.0, 82.0], rtol=0, atol=1e-6)


def test_multi_krum_selects_inliers_and_averages():
    # n=5, f=1 -> num_closest=2, so m<=2 is valid. m=2 selects the two lowest
    # scores {1 (score 2), 0 (score 5, stable-first)} -> [0,1]; uniform avg 0.5.
    decision = multi_krum_decision(GLOBAL, CLIENTS, num_malicious=1, num_to_select=2)
    assert decision.valid
    assert decision.selected == [0, 1]
    np.testing.assert_allclose(decision.new_global[0], [0.5], rtol=0, atol=1e-6)


def test_krum_single_selects_min_score():
    decision = multi_krum_decision(GLOBAL, CLIENTS, num_malicious=1, num_to_select=1)
    assert decision.valid
    assert decision.selected == [1]  # min score
    np.testing.assert_allclose(decision.new_global[0], [1.0], rtol=0, atol=1e-6)


def test_validity_gate():
    # Reduces to standard Krum at m=1: n=5,f=1 -> n>=2f+3=5 valid.
    assert multi_krum_valid(5, 1, 1) is True
    assert multi_krum_valid(5, 1, 2) is True   # m == n-f-2 == 2, the max valid
    # m=3 exceeds n-f-2=2 -> INVALID (the case the paper-condition flags).
    assert multi_krum_valid(5, 1, 3) is False
    assert multi_krum_valid(4, 1, 1) is False  # n < 2f+3
    assert multi_krum_valid(5, 2, 1) is False  # n=5 < 2*2+3=7
    assert multi_krum_valid(5, 1, 6) is False  # m out of range
    assert multi_krum_valid(5, 1, 0) is False  # m < 1


def test_invalid_config_marks_NA_not_forced():
    # m=3 with n=5,f=1 is invalid (m > n-f-2=2) -> NA, not a forced result.
    decision = multi_krum_decision(GLOBAL, CLIENTS, num_malicious=1, num_to_select=3)
    assert decision.valid is False
    assert decision.new_global is None
    assert "reason" in decision.diagnostics
    # f=2 with n=5 also invalid (n < 2f+3).
    assert multi_krum_decision(GLOBAL, CLIENTS, num_malicious=2, num_to_select=1).valid is False
