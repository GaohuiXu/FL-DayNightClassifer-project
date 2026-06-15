"""Partition oracle parity — fl_v3 Dirichlet partition == fl_v2 on the same targets."""
from __future__ import annotations

from fl_v3.data.partition import dirichlet_partition, iid_partition


def test_dirichlet_partition_matches_oracle(oracle):
    cfg = oracle["decisions"]["config"]
    part = dirichlet_partition(
        oracle["targets"],
        num_clients=cfg["part_clients"],
        alpha=cfg["part_alpha"],
        seed=cfg["part_seed"],
    )
    ref = {int(k): v for k, v in oracle["decisions"]["dirichlet_partition"].items()}
    assert set(part.keys()) == set(ref.keys())
    for cid in ref:
        assert part[cid] == ref[cid], f"client {cid} indices diverged from oracle"


def test_iid_partition_is_deterministic_and_covers_all():
    p1 = iid_partition(num_samples=100, num_clients=7, seed=3)
    p2 = iid_partition(num_samples=100, num_clients=7, seed=3)
    assert p1 == p2
    all_idx = sorted(i for v in p1.values() for i in v)
    assert all_idx == list(range(100))  # exact, disjoint cover
