"""Gradient-space metric oracle parity (cos-to-mean, pairwise cosine, top-k energy)."""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.gradient_metrics import (
    compute_gradient_space_metrics,
    compute_update_norms,
)


def test_update_norms_match_oracle(oracle):
    norms = compute_update_norms(oracle["global_params"], oracle["clients"])
    np.testing.assert_allclose(norms, oracle["decisions"]["original_norms"], rtol=0, atol=1e-9)


def test_gradient_space_metrics_match_oracle(oracle):
    topk_k = oracle["decisions"]["config"]["topk_k"]
    gsm = compute_gradient_space_metrics(oracle["global_params"], oracle["clients"], topk_k)
    ref = oracle["decisions"]["gsm"]
    assert gsm["topk_energy_k"] == ref["topk_energy_k"]
    np.testing.assert_allclose(gsm["cosine_to_mean"], ref["cosine_to_mean"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(gsm["topk_energy_frac"], ref["topk_energy_frac"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(gsm["pairwise_cosine"], ref["pairwise_cosine"], rtol=0, atol=1e-6)
