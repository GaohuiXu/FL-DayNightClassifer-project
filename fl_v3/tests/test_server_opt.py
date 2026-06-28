"""Unit tests for the FedOpt server optimizer (MCR Phase-3 / D17).

Proves: (1) fedavg is the byte-identical identity (the crown-jewel baseline), (2) fedadam/fedavgm move the
global TOWARD the aggregate, (3) a tiny-server-lr fedadam step ≈ no move (limiting case), (4) state (m,v,t)
evolves across rounds, (5) dtype is preserved, (6) determinism (two runs identical).
"""
from __future__ import annotations

import numpy as np

from fl_v3.strategy.server_opt import ServerOptimizer, build_server_optimizer


def _params(seed, shapes=((4, 3), (5,))):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


def test_fedavg_identity_is_byte_identical():
    g = _params(0)
    agg = _params(1)
    opt = ServerOptimizer(kind="fedavg", server_lr=1.0)
    assert opt.is_identity
    out = opt.step(g, agg)
    # identity returns the aggregate UNCHANGED (same object contents, byte-for-byte)
    for o, a in zip(out, agg):
        assert o.dtype == a.dtype
        assert np.array_equal(o, a)


def test_build_default_is_identity():
    opt = build_server_optimizer({})
    assert opt.kind == "fedavg" and opt.server_lr == 1.0 and opt.is_identity


def test_fedadam_moves_toward_aggregate():
    g = _params(0)
    agg = [x + 0.5 for x in _params(0)]  # aggregate = global + 0.5 (constant positive delta)
    opt = ServerOptimizer(kind="fedadam", server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3)
    out = opt.step(g, agg)
    for o, gi in zip(out, g):
        # delta>0 everywhere ⇒ new global must increase from the round-start global
        assert np.all(o > gi)
    assert opt.state.t == 1 and opt.state.m is not None and opt.state.v is not None


def test_fedadam_tiny_lr_barely_moves():
    g = _params(2)
    agg = [x + 1.0 for x in _params(2)]
    opt = ServerOptimizer(kind="fedadam", server_lr=1e-9)
    out = opt.step(g, agg)
    for o, gi in zip(out, g):
        assert np.allclose(o, gi, atol=1e-6)  # ~no move


def test_fedavgm_accumulates_momentum_across_rounds():
    g = _params(3)
    agg = [x + 0.2 for x in _params(3)]
    opt = ServerOptimizer(kind="fedavgm", server_lr=1.0, beta1=0.9)
    out1 = opt.step(g, agg)
    # second round from the SAME relative delta ⇒ momentum-corrected step should be >= first (EMA ramps up)
    out2 = opt.step(out1, [o + 0.2 for o in out1])
    step1 = np.abs(out1[0] - g[0]).mean()
    step2 = np.abs(out2[0] - out1[0]).mean()
    assert opt.state.t == 2
    assert step2 >= step1 - 1e-6  # momentum does not shrink under a constant delta


def test_dtype_preserved_and_deterministic():
    g = _params(7)
    agg = _params(8)
    o1 = ServerOptimizer(kind="fedadam", server_lr=0.05).step([x.copy() for x in g], [x.copy() for x in agg])
    o2 = ServerOptimizer(kind="fedadam", server_lr=0.05).step([x.copy() for x in g], [x.copy() for x in agg])
    for a, b, gi in zip(o1, o2, g):
        assert a.dtype == gi.dtype == np.float32
        assert np.array_equal(a, b)  # deterministic


def test_fedavgm_beta1_zero_reduces_to_fedavg():
    """The algebraic identity gate (judge-mandated): fedavgm(beta1=0, server_lr=1, no warmup) must
    reproduce plain FedAvg (new global == aggregate), within fp tolerance."""
    g = _params(5)
    agg = _params(6)
    out = ServerOptimizer(kind="fedavgm", server_lr=1.0, beta1=0.0).step(g, agg)
    for o, a in zip(out, agg):
        assert np.allclose(o, a, atol=1e-6), "fedavgm(beta1=0,slr=1) must reduce to FedAvg (x<-aggregate)"


def test_server_lr_warmup_ramps_the_step():
    g = _params(11)
    agg = [x + 0.5 for x in _params(11)]
    # warmup over 4 rounds: round 1 step ~ (1/4)*full, round 4 ~ full. Compare round-1 vs no-warmup round-1.
    warm = ServerOptimizer(kind="fedadam", server_lr=0.3, warmup_rounds=4)
    nowarm = ServerOptimizer(kind="fedadam", server_lr=0.3, warmup_rounds=0)
    ow = warm.step([x.copy() for x in g], [x.copy() for x in agg])
    on = nowarm.step([x.copy() for x in g], [x.copy() for x in agg])
    step_w = np.abs(ow[0] - g[0]).mean()
    step_n = np.abs(on[0] - g[0]).mean()
    assert step_w < step_n, "warmup must shrink the first-round step"
    # ratio ≈ 1/4 (eff_lr = server_lr * t/warmup = server_lr * 1/4 at t=1)
    assert abs(step_w / step_n - 0.25) < 1e-6


def test_warmup_makes_fedavg_non_identity():
    opt = ServerOptimizer(kind="fedavg", server_lr=1.0, warmup_rounds=3)
    assert not opt.is_identity  # warmup means round-1 step is scaled, so NOT a pure passthrough


def test_invalid_kind_raises():
    try:
        ServerOptimizer(kind="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown server-optimizer kind")
