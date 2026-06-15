"""Bit-determinism smoke: same-seed runs are byte-identical.

Two layers:
  * a tiny model trained twice from the same seed → identical weights;
  * a full in-process FL round run twice → identical aggregated-state checksum
    AND identical eval (the FL-level determinism the platform depends on).
"""
from __future__ import annotations

import numpy as np
import torch

from fl_v3.engine.local_runner import run_clean_round
from fl_v3.models.dummy import TinyMLP
from fl_v3.utils.runtime import enforce_determinism, seed_everything


def _train_tiny(seed: int) -> dict:
    enforce_determinism(strict=True)
    seed_everything(seed)
    model = TinyMLP(in_dim=4, hidden=8, out_dim=1)
    gen = np.random.default_rng(0)
    X = torch.from_numpy(gen.standard_normal((32, 4)).astype(np.float32))
    y = torch.from_numpy(gen.standard_normal((32, 1)).astype(np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.MSELoss()
    for _ in range(5):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_tiny_model_same_seed_bit_identical():
    a = _train_tiny(123)
    b = _train_tiny(123)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), f"weights diverged at {k}"


def test_tiny_model_different_seed_differs():
    a = _train_tiny(123)
    b = _train_tiny(456)
    assert any(not torch.equal(a[k], b[k]) for k in a)


_CFG = {
    "task-type": "dummy_regression",
    "seed": 42,
    "device": "cpu",
    "num-clients": 4,
    "num-local-epochs": 1,
    "batch-size": 8,
    "learning-rate": 0.01,
    "weight-decay": 0.0,
    "num-workers": 0,
    "loss": "mse",
}


def test_fl_round_same_seed_bit_identical():
    r1 = run_clean_round(dict(_CFG), defense="none", server_round=1)
    r2 = run_clean_round(dict(_CFG), defense="none", server_round=1)
    assert r1["agg_checksum"] == r2["agg_checksum"]
    assert r1["eval"] == r2["eval"]
    assert r1["client_train_losses"] == r2["client_train_losses"]


def test_fl_round_different_seed_differs():
    r1 = run_clean_round(dict(_CFG, seed=42), defense="none", server_round=1)
    r2 = run_clean_round(dict(_CFG, seed=7), defense="none", server_round=1)
    assert r1["agg_checksum"] != r2["agg_checksum"]
