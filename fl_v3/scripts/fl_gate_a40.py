"""FL bit-determinism gate — the in-process (local_runner) half (T3 GATE, SPEC §2.5).

Run on the A40 via SLURM (``run_fedavg_a40.sh``). The Ray half (two same-seed ``flwr run``
drivers → byte-identical final global model) is driven by the shell; this script:

  1. **``assert_a40``** (reused from ``det_gate_a40``) — LOUD exit-2 on no-CUDA / non-A40
     (a CPU / Tesla-T4 pass is a FALSE PASS; the float-summation drift only surfaces
     cross-architecture).
  2. runs ``local_runner.run_clean_rounds`` **twice** same-seed on the A40 (≥3 rounds,
     ``fraction-train < 1`` — the deterministic DT3-B sampler) → asserts a byte-identical
     ``final_checksum`` (the in-process crown jewel + the per-round participant sets);
  3. prints ``LOCAL_RUNNER_CHECKSUM`` — the shell compares it to the Ray run's
     ``FL_TRAINABLE_CHECKSUM`` (byte-identical on the SAME A40 = the cross-check).

The config + (num-rounds, fraction-train, min-train-nodes) come from the SAME JSON the
``flwr run`` drivers use, so both halves aggregate the identical 62-tensor vector.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from det_gate_a40 import assert_a40  # noqa: E402  (reuse the A40 guard verbatim)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON run_config (same as flwr run)")
    ap.add_argument("--mode", choices=["assert", "localrunner"], default="localrunner")
    args = ap.parse_args()

    name = assert_a40()
    print(f"[fl_gate_a40] device = {name}", flush=True)
    if args.mode == "assert":
        return

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    num_rounds = int(cfg.get("num-server-rounds", 3))
    fraction_train = float(cfg.get("fraction-train", 0.5))
    min_train_nodes = int(cfg.get("min-train-nodes", 2))

    from fl_v3.engine.local_runner import run_clean_rounds

    def go():
        return run_clean_rounds(
            dict(cfg), defense="none", num_rounds=num_rounds,
            fraction_train=fraction_train, min_train_nodes=min_train_nodes,
        )

    r1 = go()
    r2 = go()
    if r1["final_checksum"] != r2["final_checksum"]:
        print(
            "[fl_gate_a40] FATAL: local_runner NOT byte-identical across two same-seed "
            f"runs on the A40\n  run1={r1['final_checksum']}\n  run2={r2['final_checksum']}",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[fl_gate_a40] LOCAL_RUNNER two same-seed runs byte-identical: PASS", flush=True)
    for rd in r1["rounds"]:
        print(
            f"[fl_gate_a40]   round {rd['round']} participants={rd['participants']} "
            f"valid={rd['decision_valid']} checksum={rd['agg_checksum']}",
            flush=True,
        )
    print(f"[fl_gate_a40] LOCAL_RUNNER final_eval = {r1['final_eval']}", flush=True)
    # The cross-check anchor the shell greps + compares to the Ray FL_TRAINABLE_CHECKSUM.
    print(f"[fl_gate_a40] LOCAL_RUNNER_CHECKSUM = {r1['final_checksum']}", flush=True)


if __name__ == "__main__":
    main()
