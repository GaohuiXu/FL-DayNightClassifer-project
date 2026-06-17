"""DT3-B — the deterministic client sampler replaces Flower's random selection (T3).

The crown-jewel correctness for full-loop bit-determinism at ``fraction-train < 1``:

  * ``select_partition_ids`` is reproducible, salted, fraction-honoring;
  * ``NormTrackingFedAvg.configure_train`` selects the SAME partition-ids across two
    fake drivers whose node_ids differ (the Flower-random-sample failure mode) — and
    does so by mapping partition-id→node_id via discovery, NOT ``random.sample(node_ids)``;
  * the fraction=1.0 case is NOT sufficient (it trivially selects all) — the test
    exercises fraction<1;
  * Flower's ``strategy_utils.sample_nodes`` is never called (monkeypatched to raise).
"""
from __future__ import annotations

import types

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord

from fl_v3.strategy.flower_strategies import NormTrackingFedAvg
from fl_v3.strategy.sampling import (
    SAMPLE_SALT_EVAL,
    SAMPLE_SALT_TRAIN,
    sample_size_from_fraction,
    select_partition_ids,
)


# --------------------------- the pure selector ---------------------------
def test_select_partition_ids_reproducible_and_fraction_honoring():
    N, seed = 8, 20259
    for r in (1, 2, 3):
        a = select_partition_ids(seed, r, N, 0.5, 2, salt=SAMPLE_SALT_TRAIN)
        b = select_partition_ids(seed, r, N, 0.5, 2, salt=SAMPLE_SALT_TRAIN)
        assert a == b, "same (seed, round, N, fraction) must select the same pids"
        assert a == sorted(a), "selection returned sorted (canonical order)"
        assert len(a) == 4 == sample_size_from_fraction(N, 0.5, 2)
        assert set(a).issubset(range(N))


def test_select_partition_ids_varies_by_round_and_salt():
    N, seed = 12, 7
    r1 = select_partition_ids(seed, 1, N, 0.5, 2, salt=SAMPLE_SALT_TRAIN)
    r2 = select_partition_ids(seed, 2, N, 0.5, 2, salt=SAMPLE_SALT_TRAIN)
    assert r1 != r2, "different rounds should (almost surely) sample different subsets"
    train = select_partition_ids(seed, 1, N, 0.5, 2, salt=SAMPLE_SALT_TRAIN)
    evalu = select_partition_ids(seed, 1, N, 0.5, 2, salt=SAMPLE_SALT_EVAL)
    assert train != evalu, "train and evaluate streams are independent (distinct salts)"


def test_sample_size_min_floor_and_clamp():
    assert sample_size_from_fraction(8, 0.1, 3) == 3      # floor to min_nodes
    assert sample_size_from_fraction(8, 1.0, 2) == 8      # fraction=1 -> all
    assert sample_size_from_fraction(8, 2.0, 2) == 8      # clamp to N
    assert sample_size_from_fraction(8, 0.0, 2) == 0      # fraction 0 -> skip


# --------------------------- the strategy override (DT3-B) ---------------------------
class _FakeReply:
    """Duck-types what the strategy reads from a QUERY reply."""

    error = None

    def __init__(self, pid: int, node_id: int):
        self.content = {"reply-meta": {"partition-id": int(pid)}}
        self.metadata = types.SimpleNamespace(src_node_id=int(node_id))

    def has_error(self) -> bool:
        return False


class _FakeGrid:
    """A simulation driver: a fixed partition-id→node_id map with RANDOM node_ids,
    returned from ``get_node_ids`` as a ``set`` (exactly like the real LinkState)."""

    def __init__(self, pid_to_node: dict[int, int]):
        self.pid_to_node = dict(pid_to_node)
        self.node_to_pid = {v: k for k, v in pid_to_node.items()}

    def get_node_ids(self):
        return set(self.pid_to_node.values())

    def send_and_receive(self, messages, timeout=None):
        # messages are QUERY probes carrying dst_node_id; reply with that node's pid.
        return [
            _FakeReply(self.node_to_pid[m.metadata.dst_node_id], m.metadata.dst_node_id)
            for m in messages
        ]


def _strategy(fraction):
    return NormTrackingFedAvg(
        output_dir="/tmp/t3_sampling_test",
        experiment_name="t",
        seed=20259,
        fraction_train=fraction,
        fraction_evaluate=0.0,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
    )


def _dummy_arrays():
    return ArrayRecord({"x": Array(np.zeros(2, dtype=np.float32))})


def _selected_pids(strategy, grid):
    msgs = list(
        strategy.configure_train(1, _dummy_arrays(), ConfigRecord({}), grid)
    )
    # Recover the chosen partition-ids from the train messages' dst_node_ids.
    return sorted(grid.node_to_pid[m.metadata.dst_node_id] for m in msgs)


def test_configure_train_same_pids_across_drivers_at_fraction_lt_1():
    N = 8
    # Two drivers, DIFFERENT random node_ids for the same partitions (the Flower failure).
    grid_a = _FakeGrid({p: 100000 + p for p in range(N)})
    grid_b = _FakeGrid({p: 900000 + 7 * p for p in range(N)})
    pids_a = _selected_pids(_strategy(0.5), grid_a)
    pids_b = _selected_pids(_strategy(0.5), grid_b)
    assert len(pids_a) == 4, "fraction<1 must select a strict subset (not all N)"
    assert pids_a == pids_b, (
        "DT3-B: two same-seed drivers with different node_ids MUST select the identical "
        f"partition-id set at fraction<1 (got {pids_a} vs {pids_b})"
    )


def test_does_not_call_flower_random_sample(monkeypatch):
    import flwr.serverapp.strategy.strategy_utils as su

    def _boom(*a, **k):
        raise AssertionError("Flower's random sample_nodes must NOT be used (DT3-B)")

    monkeypatch.setattr(su, "sample_nodes", _boom)
    grid = _FakeGrid({p: 4242 + p for p in range(8)})
    pids = _selected_pids(_strategy(0.5), grid)  # must not raise
    assert len(pids) == 4


def test_discovery_requires_full_partition_coverage():
    # A grid missing partition-id 7 (node count 8 but pids {0..6, 99}) must fail loud.
    bad = _FakeGrid({0: 10, 1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 99: 17})
    with pytest.raises(RuntimeError, match="do not cover range"):
        _selected_pids(_strategy(0.5), bad)
