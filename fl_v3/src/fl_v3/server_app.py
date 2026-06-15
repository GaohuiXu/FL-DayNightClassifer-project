"""Task-agnostic Flower ServerApp (fl_v3 T0).

Builds the initial global model + the aggregation strategy from config, with NO
classification assumption: the model and the server-side eval metrics come from
the :class:`~fl_v3.training.tasks.Task` selected by ``task-type``, and the
strategy from ``defense-type`` via the defense registry. Server-side determinism
(seed + ``enforce_determinism``) is set once at startup.

Validated to import + construct at T0. The live Ray run is exercised at T3.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import enforce_determinism, seed_everything, truthy

app = ServerApp()


def _device(run_config) -> torch.device:
    want = str(run_config.get("device", "cpu"))
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_strategy(defense_type: str, run_config, common_kwargs: dict, exp_dir: str):
    """Instantiate the aggregation strategy for ``defense_type``.

    Imported here (not at module top) so the server app can be imported without
    ``flwr`` strategy internals being resolved until run time.
    """
    from fl_v3.strategy.flower_strategies import (
        FedMedianStrategy,
        FlameStrategy,
        FoolsGoldStrategy,
        MultiKrumStrategy,
        NormClipStrategy,
        NormTrackingFedAvg,
    )

    experiment_name = str(run_config.get("experiment-name", "default"))
    seed = int(run_config.get("seed", 42))
    topk_energy_k = int(run_config.get("topk-energy-k", 4096))
    base = dict(
        output_dir=exp_dir,
        experiment_name=experiment_name,
        seed=seed,
        topk_energy_k=topk_energy_k,
        **common_kwargs,
    )

    d = defense_type.lower()
    if d in ("none", "fedavg"):
        return NormTrackingFedAvg(**base)
    if d == "norm_clipping":
        return NormClipStrategy(clip_norm=float(run_config.get("clip-norm", 5.0)), **base)
    if d == "flame":
        return FlameStrategy(
            noise_multiplier=float(run_config.get("flame-noise-multiplier", 1e-6)), **base
        )
    if d == "foolsgold":
        return FoolsGoldStrategy(head_index=int(run_config.get("foolsgold-head-index", -2)), **base)
    if d == "fed_median":
        return FedMedianStrategy(**base)
    if d == "multi_krum":
        return MultiKrumStrategy(
            num_malicious=int(run_config.get("num-malicious-nodes", 0)),
            num_to_select=int(run_config.get("krum-num-to-select", 1)),
            **base,
        )
    raise ValueError(
        f"Unsupported defense-type {defense_type!r}. Available: none, fedavg, "
        "norm_clipping, flame, foolsgold, fed_median, multi_krum"
    )


def _server_eval_fn(context: Context, task, exp_dir: str):
    run_config = context.run_config
    device = _device(run_config)
    eval_loader = task.eval_loader(run_config)
    criterion = task.build_criterion(run_config)
    cache = {"model": None}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        if cache["model"] is None:
            cache["model"] = task.build_model(run_config).to(device)
        model = cache["model"]
        # Load BY NAME (matches the fl_v2 oracle) — robust to key ordering.
        model.load_state_dict(arrays.to_torch_state_dict())

        results = task.evaluate(model, eval_loader, criterion, device, run_config)
        print(f"[Server] round={server_round} eval={results}", flush=True)
        return MetricRecord({"server-round": int(server_round), **results})

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    import logging

    logging.getLogger("flwr").setLevel(logging.WARNING)
    run_config = context.run_config

    seed = int(run_config.get("seed", 42))
    seed_everything(seed)
    enforce_determinism(strict=truthy(run_config.get("determinism-strict", True)))

    task = get_task(str(run_config["task-type"]))

    output_dir = str(run_config.get("output-dir", "./outputs"))
    exp_dir = os.path.join(output_dir, str(run_config.get("experiment-name", "default")))
    os.makedirs(exp_dir, exist_ok=True)

    common_kwargs = dict(
        fraction_train=float(run_config.get("fraction-train", 1.0)),
        fraction_evaluate=float(run_config.get("fraction-evaluate", 1.0)),
        min_train_nodes=int(run_config.get("min-train-nodes", 2)),
        min_evaluate_nodes=int(run_config.get("min-train-nodes", 2)),
        min_available_nodes=int(run_config.get("min-available-nodes", 2)),
    )
    strategy = _build_strategy(
        str(run_config.get("defense-type", "none")), run_config, common_kwargs, exp_dir
    )

    initial_arrays = ArrayRecord(task.build_model(run_config).state_dict())
    train_config = ConfigRecord(
        {
            "num-local-epochs": int(run_config.get("num-local-epochs", 1)),
            "learning-rate": float(run_config.get("learning-rate", 1e-3)),
            "weight-decay": float(run_config.get("weight-decay", 0.0)),
        }
    )
    evaluate_config = ConfigRecord({})
    evaluate_fn = _server_eval_fn(context, task, exp_dir)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=int(run_config.get("num-server-rounds", 1)),
        train_config=train_config,
        evaluate_config=evaluate_config,
        evaluate_fn=evaluate_fn,
    )

    if result.arrays:
        ckpt = os.path.join(exp_dir, "final_model.pt")
        torch.save(result.arrays.to_torch_state_dict(), ckpt)
        print(f"[server] checkpoint saved -> {ckpt}", flush=True)
    print("Federated training finished.", flush=True)
