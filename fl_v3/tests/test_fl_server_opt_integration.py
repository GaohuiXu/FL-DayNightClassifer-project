"""FedAdam server-optimizer END-TO-END through ``local_runner`` (MCR Phase-3 / D17).

CPU, dummy task (instant). Proves the server-optimizer axis is wired into the multi-round FedAvg path
correctly and ORTHOGONALLY to the defense: (1) the default (no server-optimizer config) is byte-identical to
explicit fedavg (the crown-jewel baseline is untouched); (2) FedAdam produces a DIFFERENT, FINITE trajectory;
(3) FedAdam is deterministic; (4) the client-recipe knobs (grad-clip / backbone-lr-mult / adamw) are accepted
and do not crash on the dummy model (no camera_backbone ⇒ flat optimizer ⇒ recipe is a no-op there).
"""
from __future__ import annotations

import math

from fl_v3.engine.local_runner import run_clean_rounds

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


def _run(extra, rounds=4):
    cfg = dict(_CFG)
    cfg.update(extra)
    return run_clean_rounds(cfg, defense="none", num_rounds=rounds, fraction_train=1.0, min_train_nodes=2)


def test_default_equals_explicit_fedavg_byte_identical():
    a = _run({})  # no server-optimizer key ⇒ identity FedAvg
    b = _run({"server-optimizer": "fedavg", "server-lr": 1.0})
    assert a["final_checksum"] == b["final_checksum"], "default must equal explicit identity FedAvg"


def test_fedadam_differs_from_fedavg_and_is_finite():
    base = _run({})
    fa = _run({"server-optimizer": "fedadam", "server-lr": 0.05, "server-tau": 0.01,
               "server-lr-warmup-rounds": 2})
    assert fa["final_checksum"] != base["final_checksum"], "FedAdam must change the trajectory"
    assert math.isfinite(float(fa["final_eval"]["eval_loss"])), "FedAdam global must be finite (no NaN)"


def test_fedadam_is_deterministic():
    a = _run({"server-optimizer": "fedadam", "server-lr": 0.05, "server-lr-warmup-rounds": 2})
    b = _run({"server-optimizer": "fedadam", "server-lr": 0.05, "server-lr-warmup-rounds": 2})
    assert a["final_checksum"] == b["final_checksum"]
    assert [r["agg_checksum"] for r in a["rounds"]] == [r["agg_checksum"] for r in b["rounds"]]


def test_fedavgm_differs_and_finite():
    fm = _run({"server-optimizer": "fedavgm", "server-lr": 0.5, "server-beta1": 0.9})
    base = _run({})
    assert fm["final_checksum"] != base["final_checksum"]
    assert math.isfinite(float(fm["final_eval"]["eval_loss"]))


def test_client_recipe_knobs_accepted_on_dummy():
    # grad-clip / backbone-lr-mult / adamw must be accepted by train_local via the local_runner wiring.
    # The dummy model has no camera_backbone ⇒ the 2-group split does not fire (flat path); just assert it runs.
    out = _run({"grad-clip-norm": 5.0, "det-backbone-lr-mult": 0.1, "det-optimizer": "adamw"})
    assert math.isfinite(float(out["final_eval"]["eval_loss"]))
