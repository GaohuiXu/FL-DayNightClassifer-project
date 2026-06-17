"""``local_runner.run_clean_rounds`` — the multi-round, sampled, cross-check substrate (T3).

Login-node-runnable on the dummy task (TinyMLP, instant). Proves:
  * two same-seed multi-round runs are byte-identical (the in-process determinism the A40
    gate re-checks on the heavy AD model + live Ray path);
  * the DT3-B sampler selects a strict, reproducible participant subset each round at
    ``fraction_train < 1`` (recorded per round);
  * after DT3-A the dummy path still aggregates the FULL vector (trainable filter is
    identity on TinyMLP) — task-agnostic preserved.
The detection-model + Ray-path bit-identity is the A40 SLURM gate (run_fedavg_a40.sh).
"""
from __future__ import annotations

from fl_v3.engine.local_runner import run_clean_round, run_clean_rounds

_CFG = {
    "task-type": "dummy_regression",
    "seed": 20259,
    "device": "cpu",
    "num-clients": 8,
    "num-local-epochs": 1,
    "batch-size": 8,
    "learning-rate": 0.01,
    "num-workers": 0,
    "loss": "mse",
}


def test_run_clean_rounds_two_run_bit_identical():
    a = run_clean_rounds(dict(_CFG), defense="none", num_rounds=3, fraction_train=0.5, min_train_nodes=2)
    b = run_clean_rounds(dict(_CFG), defense="none", num_rounds=3, fraction_train=0.5, min_train_nodes=2)
    assert a["final_checksum"] == b["final_checksum"], "multi-round FedAvg not bit-deterministic"
    assert [r["agg_checksum"] for r in a["rounds"]] == [r["agg_checksum"] for r in b["rounds"]]


def test_run_clean_rounds_sampling_strict_subset_and_reproducible():
    a = run_clean_rounds(dict(_CFG), defense="none", num_rounds=3, fraction_train=0.5, min_train_nodes=2)
    parts = [tuple(r["participants"]) for r in a["rounds"]]
    for p in parts:
        assert len(p) == 4, "fraction_train=0.5 over N=8 must sample 4 participants/round"
        assert sorted(p) == list(p), "participants recorded in canonical (sorted) order"
    # reproducible per round across a second run
    b = run_clean_rounds(dict(_CFG), defense="none", num_rounds=3, fraction_train=0.5, min_train_nodes=2)
    assert parts == [tuple(r["participants"]) for r in b["rounds"]]


def test_run_clean_rounds_fraction_one_uses_all():
    a = run_clean_rounds(dict(_CFG), defense="none", num_rounds=2, fraction_train=1.0, min_train_nodes=2)
    assert all(len(r["participants"]) == _CFG["num-clients"] for r in a["rounds"])


def test_single_round_runner_still_deterministic_after_dt3a():
    # run_clean_round (singular, all-clients) is the T0 smoke path — trainable-only must
    # keep it bit-identical on the dummy task (trainable filter == identity on TinyMLP).
    a = run_clean_round(dict(_CFG), defense="none", server_round=1)
    b = run_clean_round(dict(_CFG), defense="none", server_round=1)
    assert a["agg_checksum"] == b["agg_checksum"]
    assert a["eval"] is not None
