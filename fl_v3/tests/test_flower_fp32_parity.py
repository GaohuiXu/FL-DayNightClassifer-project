"""Clean FedAvg arithmetic parity with Flower 1.27."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr.common")
from flwr.common import Array, ArrayRecord, MetricRecord, RecordDict  # noqa: E402
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords  # noqa: E402

from fl_v3.strategy.aggregation_core import fp32_weighted_average  # noqa: E402


def _clients():
    rng = np.random.default_rng(2024)
    shapes = [(5, 3), (3,)]
    values = [
        [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
        for _ in range(6)
    ]
    return values, [10, 20, 30, 40, 50, 60]


def _records(client_params, num_examples):
    records = []
    for params, weight in zip(client_params, num_examples):
        arrays = ArrayRecord(
            {f"p{index}": Array(np.asarray(value)) for index, value in enumerate(params)}
        )
        metrics = MetricRecord({"num-examples": float(weight)})
        records.append(RecordDict({"arrays": arrays, "metrics": metrics}))
    return records


@pytest.mark.parametrize("weighted", [True, False])
def test_clean_fp32_average_matches_flower_bit_for_bit(weighted):
    clients, weights = _clients()
    if not weighted:
        weights = [1.0] * len(clients)
    flower = aggregate_arrayrecords(
        _records(clients, weights), "num-examples"
    ).to_numpy_ndarrays()
    ours = fp32_weighted_average(clients, weights)
    for actual, expected in zip(ours, flower):
        assert actual.dtype == expected.dtype == np.float32
        assert np.array_equal(actual, expected)


def test_clean_fp32_average_is_deterministic_for_fixed_order():
    clients, weights = _clients()
    first = fp32_weighted_average(clients, weights)
    second = fp32_weighted_average(clients, weights)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
