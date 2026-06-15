"""FLAME oracle parity — fl_v3 reproduces the fl_v2 FLAME decision on a fixture.

Implementation equivalence ONLY (same update vectors in → same decision out);
this does NOT certify AD-domain validity.
"""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.defenses.base import build_flat_updates
from fl_v3.strategy.defenses.flame import (
    flame_cluster_admitted,
    flame_decision,
)
from fl_v3.strategy.gradient_metrics import compute_update_norms


def test_flame_cluster_admitted_matches_oracle(oracle):
    g = oracle["global_params"]
    clients = oracle["clients"]
    norms = np.asarray(compute_update_norms(g, clients), dtype=np.float64)
    flat = build_flat_updates(g, clients)
    admitted = flame_cluster_admitted(flat, norms)
    assert sorted(admitted) == sorted(oracle["decisions"]["flame_admitted"])


def test_flame_decision_matches_oracle(oracle):
    g = oracle["global_params"]
    clients = oracle["clients"]
    cfg = oracle["decisions"]["config"]
    decision = flame_decision(
        g,
        clients,
        noise_multiplier=cfg["flame_noise_multiplier"],
        seed=cfg["flame_seed"],
        server_round=cfg["flame_round"],
    )
    # Admitted set, clip bound, coefs.
    assert sorted(decision.admitted) == sorted(oracle["decisions"]["flame_admitted"])
    np.testing.assert_allclose(decision.diagnostics["s_t"], oracle["decisions"]["flame_s_t"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(decision.coefs, oracle["decisions"]["flame_coefs"],
                               rtol=0, atol=1e-12)
    # Aggregated arrays (incl. seeded noise) bit-identical to the oracle.
    for got, ref in zip(decision.new_global, oracle["flame_new_global"]):
        assert got.dtype == ref.dtype
        assert np.array_equal(got, ref), "FLAME aggregated array diverged from oracle"
