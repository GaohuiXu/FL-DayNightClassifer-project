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
    decision = multi_krum_decision(GLOBAL, CLIENTS, num_malicious=1, num_to_select=3)
    assert decision.valid
    assert decision.selected == [0, 1, 2]
    np.testing.assert_allclose(decision.new_global[0], [1.0], rtol=0, atol=1e-6)


def test_krum_single_selects_min_score():
    decision = multi_krum_decision(GLOBAL, CLIENTS, num_malicious=1, num_to_select=1)
    assert decision.valid
    assert decision.selected == [1]  # min score
    np.testing.assert_allclose(decision.new_global[0], [1.0], rtol=0, atol=1e-6)


def test_validity_gate():
    # n=5, f=1, m=3 → valid (5 >= 2*1+3).
    assert multi_krum_valid(5, 1, 3) is True
    # n=4, f=1 → 4 < 5 → invalid.
    assert multi_krum_valid(4, 1, 1) is False
    # n=5, f=2 → 5 < 7 → invalid.
    assert multi_krum_valid(5, 2, 1) is False
    # m out of range.
    assert multi_krum_valid(5, 1, 6) is False


def test_invalid_config_marks_NA_not_forced():
    # f=2 with n=5 is invalid → decision must be NA, not a forced result.
    decision = multi_krum_decision(GLOBAL, CLIENTS, num_malicious=2, num_to_select=1)
    assert decision.valid is False
    assert decision.new_global is None
    assert "reason" in decision.diagnostics
