"""Bit-identity vs the REAL Flower aggregation oracle (T0 review fix).

fl_v2's clean-FedAvg / NormClip / MultiKrum aggregation delegates to Flower's
``aggregate_arrayrecords`` (fp32). These tests drive that ACTUAL Flower function
(it is always in the test venv) and assert fl_v3 reproduces it BIT-FOR-BIT —
restoring the null-config / clean-baseline bit-identity crown jewel and
validating MultiKrum's one-shot selection against Flower's ``select_multikrum``.
"""
from __future__ import annotations

import numpy as np
import pytest

flwr_common = pytest.importorskip("flwr.common")
from flwr.common import Array, ArrayRecord, MetricRecord, RecordDict  # noqa: E402
from flwr.serverapp.strategy.multikrum import select_multikrum  # noqa: E402
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords  # noqa: E402

from fl_v3.strategy.aggregation_core import fp32_weighted_average  # noqa: E402
from fl_v3.strategy.defenses.fedavg import fedavg_decision, norm_clip_decision  # noqa: E402
from fl_v3.strategy.defenses.multi_krum import multi_krum_decision  # noqa: E402
from fl_v3.strategy.gradient_metrics import clip_updates_by_l2_norm  # noqa: E402


def _synthetic():
    """6 clients (4 clustered + 2 outliers), 2 tensors, varied num-examples."""
    rng = np.random.default_rng(2024)
    shapes = [(5, 3), (3,)]
    g = [rng.standard_normal(s).astype(np.float32) * 0.1 for s in shapes]
    base = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    out = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    clients = []
    for _ in range(4):  # clustered
        clients.append([(gg + 0.05 * b + 0.005 * rng.standard_normal(gg.shape).astype(np.float32))
                        for gg, b in zip(g, base)])
    for _ in range(2):  # outliers
        clients.append([(gg + 0.4 * o) for gg, o in zip(g, out)])
    num_examples = [10, 20, 30, 40, 50, 60]
    return g, clients, num_examples


def _records(client_params_list, num_examples):
    recs = []
    for idx, (cp, ne) in enumerate(zip(client_params_list, num_examples)):
        ar = ArrayRecord({f"p{i}": Array(np.asarray(c)) for i, c in enumerate(cp)})
        mr = MetricRecord({"num-examples": float(ne), "idx": float(idx)})
        recs.append(RecordDict({"arrays": ar, "metrics": mr}))
    return recs


def test_fp32_weighted_average_matches_flower_weighted():
    g, clients, ne = _synthetic()
    flower = aggregate_arrayrecords(_records(clients, ne), "num-examples").to_numpy_ndarrays()
    ours = fp32_weighted_average(clients, ne)
    for a, b in zip(ours, flower):
        assert a.dtype == b.dtype == np.float32
        assert np.array_equal(a, b), "fp32_weighted_average diverged from Flower (weighted)"


def test_fedavg_decision_matches_flower_uniform_and_weighted():
    g, clients, ne = _synthetic()
    # weighted
    flower_w = aggregate_arrayrecords(_records(clients, ne), "num-examples").to_numpy_ndarrays()
    dec_w = fedavg_decision(g, clients, weights=ne)
    for a, b in zip(dec_w.new_global, flower_w):
        assert np.array_equal(a, b)
    # uniform (all equal num-examples)
    uniform = [7.0] * len(clients)
    flower_u = aggregate_arrayrecords(_records(clients, uniform), "num-examples").to_numpy_ndarrays()
    dec_u = fedavg_decision(g, clients, weights=None)
    for a, b in zip(dec_u.new_global, flower_u):
        assert np.array_equal(a, b)


def test_norm_clip_decision_matches_flower_after_clip():
    g, clients, ne = _synthetic()
    clip_norm = 0.3  # small enough to actually bite some clients
    clipped, _, _ = clip_updates_by_l2_norm(g, clients, clip_norm)
    flower = aggregate_arrayrecords(_records(clipped, ne), "num-examples").to_numpy_ndarrays()
    dec = norm_clip_decision(g, clients, clip_norm, weights=ne)
    for a, b in zip(dec.new_global, flower):
        assert np.array_equal(a, b), "NormClip post-clip aggregation diverged from Flower"


def test_multikrum_decision_parity_vs_flower():
    g, clients, ne = _synthetic()  # n=6
    f, m = 1, 3  # valid: n>=2f+3=5, m<=n-f-2=3
    recs = _records(clients, ne)
    selected_recs = select_multikrum(recs, num_malicious_nodes=f, num_nodes_to_select=m)
    flower_selected = sorted(int(r["metrics"]["idx"]) for r in selected_recs)
    flower_arrays = aggregate_arrayrecords(selected_recs, "num-examples").to_numpy_ndarrays()

    dec = multi_krum_decision(g, clients, num_malicious=f, num_to_select=m, weights=ne)
    assert dec.valid
    # The DECISION (which clients are selected) is the scientifically meaningful
    # output and must match Flower EXACTLY.
    assert dec.selected == flower_selected, (
        f"MultiKrum selection differs: fl_v3={dec.selected} flower={flower_selected}"
    )
    # The selected-subset AVERAGE matches Flower to within fp32 noise (~1 ULP):
    # Flower aggregates the subset in score order, fl_v3 in index order, and
    # float32 addition is order-sensitive; fl_v3 also uses a more numerically
    # stable direct ||x-y||^2 distance than Flower's ||x||^2+||y||^2-2x.y. Exact
    # bit-parity is claimed for the SELECTION, tolerance for the average.
    for a, b in zip(dec.new_global, flower_arrays):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
