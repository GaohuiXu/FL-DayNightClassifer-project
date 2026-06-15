"""FoolsGold oracle parity — fl_v3 reproduces the fl_v2 trust weights + aggregate."""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.defenses.foolsgold import FoolsGoldState, foolsgold_weights


def _head_updates(oracle, head_index):
    g = oracle["global_params"]
    return [
        (np.asarray(cp[head_index], np.float32) - np.asarray(g[head_index], np.float32)).reshape(-1)
        for cp in oracle["clients"]
    ]


def test_foolsgold_weights_match_oracle(oracle):
    head_index = oracle["decisions"]["config"]["foolsgold_head_index"]
    trust = foolsgold_weights(_head_updates(oracle, head_index))
    np.testing.assert_allclose(trust, oracle["decisions"]["foolsgold_trust"], rtol=0, atol=1e-12)


def test_foolsgold_decision_matches_oracle(oracle):
    head_index = oracle["decisions"]["config"]["foolsgold_head_index"]
    state = FoolsGoldState()
    decision = state.update_and_decide(
        oracle["global_params"],
        oracle["clients"],
        oracle["partition_ids"],
        head_index=head_index,
    )
    np.testing.assert_allclose(decision.coefs, oracle["decisions"]["foolsgold_coefs"],
                               rtol=0, atol=1e-12)
    for got, ref in zip(decision.new_global, oracle["foolsgold_new_global"]):
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref), "FoolsGold aggregated array diverged from oracle"
