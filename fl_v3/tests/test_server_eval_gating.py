"""Config-gated server eval policy (Cycle-04 D14 Phase-1 B).

Unit-tests the pure ``should_server_eval`` decision (the per-round gate). End-to-end RNG-neutrality
(eval none vs all → byte-identical FL_TRAINABLE_CHECKSUM) is proven on the A40 by
run_b_eval_neutral_a40.sh — that needs Ray + a GPU and is not a login-node unit test.
"""
from __future__ import annotations

import pytest

from fl_v3.server_app import should_server_eval


def test_none_never_evals():
    for r in (0, 1, 7, 15, 16):
        assert should_server_eval("none", 1, r, 15) is False


def test_all_always_evals():
    for r in (0, 1, 7, 15):
        assert should_server_eval("all", 1, r, 15) is True


def test_final_only_last_round():
    assert should_server_eval("final", 1, 0, 15) is False
    assert should_server_eval("final", 1, 14, 15) is False
    assert should_server_eval("final", 1, 15, 15) is True
    assert should_server_eval("final", 1, 16, 15) is True  # defensive: >= num_rounds


def test_every_n_includes_final_and_multiples():
    # freq=4, 15 rounds -> 4,8,12 + the final 15 (even though 15 % 4 != 0)
    got = [r for r in range(1, 16) if should_server_eval("every_n", 4, r, 15)]
    assert got == [4, 8, 12, 15]
    # round 0 (untrained) is never evaluated under every_n
    assert should_server_eval("every_n", 4, 0, 15) is False


def test_every_n_freq1_is_all_trained_rounds():
    got = [r for r in range(0, 16) if should_server_eval("every_n", 1, r, 15)]
    assert got == list(range(1, 16))  # all trained rounds, not round 0


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        should_server_eval("sometimes", 1, 5, 15)
