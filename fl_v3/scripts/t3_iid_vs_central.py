"""IID-mini FedAvg ≈ centralized — the falsifiable "the loop truly trains" check (T3 §2.4).

Trains the SAME deterministic detector two ways on mini and compares the T2 center-distance
proxy (recall@2m, car) on the held-out mini_val:

  * **central**: one model over ALL of mini_train for ``total_steps`` Adam steps;
  * **IID-mini FedAvg**: ``N`` IID client shards, ``R`` rounds × ``E`` local epochs at
    ``fraction-train=1.0`` (all clients each round — the fair central comparison), via the
    SAME ``run_clean_rounds`` machinery the platform uses.

Falsifiable bars (a zero-decode dead loop FAILS):
  * both sides clear an ABSOLUTE recall floor ``R_floor > 0``;
  * anti-collapse: ``n_decoded > 0`` AND ``matched_gt > 0`` both sides;
  * the two recalls agree within ``delta`` (loss reported as secondary, never the sole match).

mini = engineering smoke (NOT a scientific claim — the scientific gap is the trainval run).
Run on the A40 via SLURM for the headline numbers (any CUDA suffices for the falsifiable
bars). Reports the curve + a machine-readable JSON next to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _central_train_eval(cfg: dict, total_steps: int):
    from fl_v3.training.tasks import get_task
    from fl_v3.training.loop import _move_to_device
    from fl_v3.utils.runtime import derive_seed, enforce_determinism, seed_everything

    enforce_determinism(strict=True)
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = torch.device(str(cfg.get("device", "cpu")))
    task = get_task("nuscenes_detection")
    model = task.build_model(cfg).to(device)
    criterion = task.build_criterion(cfg)
    # ALL of mini_train as a single central loader (client_data over the full token set).
    from fl_v3.data.nuscenes import info_cache as IC

    info_list, _ = IC.load_cache(
        str(cfg["nuscenes-cache-dir"]), str(cfg["nuscenes-version"]),
        str(cfg["nuscenes-train-split"]),
    )
    tokens = sorted(i["sample_token"] for i in info_list)
    loader = task._make_loader(cfg, info_list, tokens, shuffle=True)  # noqa: SLF001
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.get("learning-rate", 3e-3)),
        weight_decay=float(cfg.get("weight-decay", 0.0)),
    )
    model.train()
    step = 0
    while step < total_steps:
        for batch in loader:
            batch = _move_to_device(batch, device)
            opt.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            opt.step()
            step += 1
            if step >= total_steps:
                break
    return task.evaluate(model, task.eval_loader(cfg), criterion, device, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num-rounds", type=int, default=10)
    ap.add_argument("--num-clients", type=int, default=4)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--r-floor", type=float, default=0.05, help="absolute recall floor (>0)")
    ap.add_argument("--delta", type=float, default=0.15, help="max |recall_fed - recall_central|")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    cfg["nuscenes-partition-mode"] = "iid"
    cfg["nuscenes-num-clients"] = args.num_clients
    cfg["num-clients"] = args.num_clients
    cfg["num-local-epochs"] = args.local_epochs
    cfg["det-eval-limit"] = 0  # full held-out split for the real recall measurement

    if torch.cuda.is_available():
        print(f"[iid_vs_central] device = {torch.cuda.get_device_name()}", flush=True)

    from fl_v3.engine.local_runner import run_clean_rounds

    fed = run_clean_rounds(
        dict(cfg), defense="none", num_rounds=args.num_rounds,
        fraction_train=1.0, min_train_nodes=args.num_clients,
    )
    fed_eval = fed["final_eval"]

    # Match central compute to the federated total (N clients × E epochs × steps/epoch ≈
    # rounds × N × steps-per-client-epoch); we approximate by rounds×N×E×(mini/N/batch)
    # = rounds×E×(mini/batch) total steps so both see comparable gradient updates.
    from fl_v3.data.nuscenes import info_cache as IC

    info_list, _ = IC.load_cache(
        str(cfg["nuscenes-cache-dir"]), str(cfg["nuscenes-version"]),
        str(cfg["nuscenes-train-split"]),
    )
    n_train = len(info_list)
    bs = int(cfg.get("batch-size", 4))
    total_steps = args.num_rounds * args.local_epochs * max(1, n_train // bs)
    central_eval = _central_train_eval(dict(cfg), total_steps)

    rf, rc = float(fed_eval["proxy_recall_at_2m"]), float(central_eval["proxy_recall_at_2m"])
    result = {
        "fed": fed_eval, "central": central_eval,
        "recall_fed": rf, "recall_central": rc,
        "r_floor": args.r_floor, "delta": args.delta,
        "abs_gap": abs(rf - rc),
        "num_rounds": args.num_rounds, "num_clients": args.num_clients,
        "local_epochs": args.local_epochs, "central_total_steps": total_steps,
        "rounds_curve": fed["rounds"],
    }
    # Falsifiable bars.
    checks = {
        "anti_collapse_fed": fed_eval["proxy_n_decoded"] > 0 and fed_eval["proxy_matched_gt"] > 0,
        "anti_collapse_central": central_eval["proxy_n_decoded"] > 0 and central_eval["proxy_matched_gt"] > 0,
        "fed_clears_floor": rf >= args.r_floor,
        "central_clears_floor": rc >= args.r_floor,
        "recall_agreement": abs(rf - rc) <= args.delta,
    }
    result["checks"] = checks
    result["PASS"] = all(checks.values())
    print("[iid_vs_central] " + json.dumps(result, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    print(f"[iid_vs_central] PASS={result['PASS']} "
          f"recall_fed={rf:.4f} recall_central={rc:.4f} gap={result['abs_gap']:.4f} "
          f"(R_floor={args.r_floor}, delta={args.delta})", flush=True)
    if not result["PASS"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
