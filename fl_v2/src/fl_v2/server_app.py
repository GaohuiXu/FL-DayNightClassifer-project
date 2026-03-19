from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_v2.data.dataset import (
    build_client_index_map_with_stats,
    get_global_testloader,
)
from fl_v2.models import GTSRBClassifier
from fl_v2.training import evaluate

from fl_v2.utils import ensure_dir, save_json

from fl_v2.strategy import NormClippedFedAvg


app = ServerApp()


def _get_device(run_config) -> torch.device:
    """Select device for server-side evaluation."""
    requested = str(run_config.get("device", "cpu")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def _build_initial_arrays(context: Context) -> ArrayRecord:
    """Create the initial global model parameters."""
    num_classes = int(context.run_config["num-classes"])
    model = GTSRBClassifier(num_classes=num_classes)
    return ArrayRecord(model.state_dict())


def _get_train_config(context: Context) -> ConfigRecord:
    """Build training config sent to clients each round."""
    run_config = context.run_config

    return ConfigRecord(
        {
            "num-local-epochs": int(run_config["num-local-epochs"]),
            "learning-rate": float(run_config["learning-rate"]),
            "weight-decay": float(run_config["weight-decay"]),
        }
    )


def _get_evaluate_config(context: Context) -> ConfigRecord:
    """Build evaluation config sent to clients each round."""
    return ConfigRecord({})


def _server_side_evaluate_fn(context: Context):
    """
    Build a centralized evaluation callback.

    Flower strategies can receive an evaluate_fn callback in strategy.start(...).
    The callback takes (server_round, arrays) and returns a MetricRecord or None.
    """
    run_config = context.run_config
    data_root = str(run_config["data-root"])
    batch_size = int(run_config["batch-size"])
    image_size = int(run_config["image-size"])
    num_classes = int(run_config["num-classes"])
    device = _get_device(run_config)

    testloader = get_global_testloader(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        download=False,
    )

    criterion = nn.CrossEntropyLoss()

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        model = GTSRBClassifier(num_classes=num_classes)
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)

        metrics = evaluate(
            model=model,
            dataloader=testloader,
            criterion=criterion,
            device=device,
        )

        print(
            f"[Server Evaluation] round={server_round} "
            f"test_loss={metrics['loss']:.4f} "
            f"test_acc={metrics['accuracy']:.4f}"
        )

        return MetricRecord(
            {
                "server-round": int(server_round),
                "test_loss": float(metrics["loss"]),
                "test_accuracy": float(metrics["accuracy"]),
                "num-test-examples": int(metrics["num_samples"]),
            }
        )

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the Flower ServerApp."""
    print("[server] entered main", flush=True)

    run_config = context.run_config
    print("[server] run_config loaded", flush=True)

    num_rounds = int(run_config["num-server-rounds"])
    fraction_train = float(run_config["fraction-train"])
    fraction_evaluate = float(run_config["fraction-evaluate"])
    min_available_nodes = int(run_config["min-available-nodes"])

    output_dir = str(run_config.get("output-dir", "./outputs"))
    ensure_dir(output_dir)

    data_root = str(run_config["data-root"])
    num_clients = int(run_config["num-clients"])
    partition_mode = str(run_config["partition-mode"])
    dirichlet_alpha = float(run_config["dirichlet-alpha"])
    seed = int(run_config["seed"])
    experiment_name = str(run_config.get("experiment-name", "default"))

    print("[server] building histogram stats", flush=True)
    _, histograms, summary = build_client_index_map_with_stats(
        data_root=data_root,
        num_clients=num_clients,
        partition_mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
        download=False,
    )

    print("===== Client label histogram summary =====", flush=True)
    print(summary, flush=True)

    save_json(
        histograms,
        f"{output_dir}/{experiment_name}_seed{seed}_client_label_histograms.json",
    )
    print("[server] histogram stats done", flush=True)

    print("[server] creating strategy", flush=True)
    defense_type = str(run_config.get("defense-type", "none"))
    clip_norm = float(run_config.get("clip-norm", 5.0))

    if defense_type == "none":
        strategy = FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_available_nodes,
            min_evaluate_nodes=min_available_nodes,
            min_available_nodes=min_available_nodes,
        )
    elif defense_type == "norm_clipping":
        strategy = NormClippedFedAvg(
            clip_norm=clip_norm,
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_available_nodes,
            min_evaluate_nodes=min_available_nodes,
            min_available_nodes=min_available_nodes,
        )
        print(
            f"[server] defense enabled: norm_clipping (clip_norm={clip_norm})",
            flush=True,
        )
    else:
        raise ValueError(f"Unsupported defense-type: {defense_type}")
    print("[server] strategy created", flush=True)

    print("[server] building initial arrays", flush=True)
    initial_arrays = _build_initial_arrays(context)
    train_config = _get_train_config(context)
    evaluate_config = _get_evaluate_config(context)
    evaluate_fn = _server_side_evaluate_fn(context)
    print("[server] initial arrays ready", flush=True)

    print("[server] calling strategy.start()", flush=True)
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=train_config,
        evaluate_config=evaluate_config,
        evaluate_fn=evaluate_fn,
    )
    print("[server] strategy.start() returned", flush=True)

    print("Federated training finished.")
    print(result)