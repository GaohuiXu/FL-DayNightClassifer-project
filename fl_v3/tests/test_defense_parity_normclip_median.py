"""NormClip + FedMedian oracle parity (clipped params/norms; coordinate median)."""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.defenses.fed_median import fed_median_decision
from fl_v3.strategy.gradient_metrics import clip_updates_by_l2_norm


def test_norm_clip_matches_oracle(oracle):
    clip_norm = oracle["decisions"]["config"]["clip_norm"]
    clipped, orig_norms, clip_norms = clip_updates_by_l2_norm(
        oracle["global_params"], oracle["clients"], clip_norm
    )
    np.testing.assert_allclose(orig_norms, oracle["decisions"]["clip_original_norms"],
                               rtol=0, atol=1e-9)
    np.testing.assert_allclose(clip_norms, oracle["decisions"]["clip_norms"],
                               rtol=0, atol=1e-9)
    for j, cp in enumerate(clipped):
        for got, ref in zip(cp, oracle["clipped"][j]):
            assert got.dtype == ref.dtype
            assert np.array_equal(got, ref), f"clipped client {j} diverged from oracle"


def test_fed_median_matches_oracle(oracle):
    decision = fed_median_decision(oracle["global_params"], oracle["clients"])
    for got, ref in zip(decision.new_global, oracle["median_new_global"]):
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref), "FedMedian aggregated array diverged from oracle"
