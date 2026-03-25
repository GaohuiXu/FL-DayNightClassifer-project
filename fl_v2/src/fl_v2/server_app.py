from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from fl_v2.data.dataset import (
    build_client_index_map_with_stats,
    get_global_testloader,
)
from fl_v2.attacks_defenses import make_pixel_trigger_fn
from fl_v2.models import GTSRBClassifier
from fl_v2.training import compute_asr, compute_target_class_accuracy, evaluate

from fl_v2.utils import ensure_dir, save_attack_comparison, save_json

from fl_v2.strategy import NormClippedFedAvg, NormTrackingFedAvg


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

    attack_type = str(run_config.get("attack-type", "none"))
    backdoor_target_label = int(run_config.get("backdoor-target-label", 0))
    trigger_size = int(run_config.get("trigger-size", 4))
    trigger_value = float(run_config.get("trigger-value", 1.0))
    trigger_position = str(run_config.get("trigger-position", "bottom-right"))

    testloader = get_global_testloader(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=0,
        download=False,
    )

    criterion = nn.CrossEntropyLoss()

    # Build trigger function for ASR if backdoor attack is active
    trigger_fn = None
    if attack_type in ("pixel_backdoor",):
        trigger_fn = make_pixel_trigger_fn(
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
        )

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

        print(f"\n── Server Eval {'─' * 45}", flush=True)
        print(
            f"[Server] round={server_round}  "
            f"test_loss={metrics['loss']:.4f}  "
            f"test_acc={metrics['accuracy']:.2%}",
            flush=True,
        )

        metrics_dict = {
            "server-round": int(server_round),
            "test_loss": float(metrics["loss"]),
            "test_accuracy": float(metrics["accuracy"]),
            "num-test-examples": int(metrics["num_samples"]),
        }

        # Target-class clean accuracy — always (enables baseline vs attack comparison)
        tca_result = compute_target_class_accuracy(
            model=model,
            testloader=testloader,
            target_label=backdoor_target_label,
            device=device,
        )
        metrics_dict["target_class_clean_accuracy"] = float(
            tca_result["target_class_accuracy"]
        )
        metrics_dict["target_class_num_samples"] = int(
            tca_result["num_target_samples"]
        )
        print(
            f"[Server] round={server_round}  "
            f"target_class_acc={tca_result['target_class_accuracy']:.2%} "
            f"(n={tca_result['num_target_samples']})",
            flush=True,
        )

        # ASR — only when a backdoor trigger is active
        if trigger_fn is not None:
            asr_result = compute_asr(
                model=model,
                testloader=testloader,
                target_label=backdoor_target_label,
                trigger_fn=trigger_fn,
                device=device,
            )
            metrics_dict["asr"] = float(asr_result["asr"])
            metrics_dict["asr_num_samples"] = int(asr_result["num_triggered_samples"])
            print(
                f"[Server] round={server_round}  "
                f"ASR={asr_result['asr']:.2%} "
                f"(n={asr_result['num_triggered_samples']})",
                flush=True,
            )

        return MetricRecord(metrics_dict)

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the Flower ServerApp."""
    import logging
    import random
    import numpy as np
    logging.getLogger("flwr").setLevel(logging.WARNING)

    print("[server] entered main", flush=True)

    run_config = context.run_config
    print("[server] run_config loaded", flush=True)

    # Seed all RNGs immediately so model init and any stochastic ops are reproducible
    seed = int(run_config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_rounds = int(run_config["num-server-rounds"])
    fraction_train = float(run_config["fraction-train"])
    fraction_evaluate = float(run_config["fraction-evaluate"])
    min_available_nodes = int(run_config["min-available-nodes"])
    min_train_nodes = int(run_config.get("min-train-nodes", 2))

    output_dir = str(run_config.get("output-dir", "./outputs"))
    ensure_dir(output_dir)

    data_root = str(run_config["data-root"])
    num_clients = int(run_config["num-clients"])
    partition_mode = str(run_config["partition-mode"])
    dirichlet_alpha = float(run_config["dirichlet-alpha"])
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
    clip_norm = float(run_config.get("clip-norm", 100.0))

    # Common kwargs for all FedAvg-based strategies
    common_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_train_nodes,
        min_available_nodes=min_available_nodes,
    )

    if defense_type == "none":
        strategy = NormTrackingFedAvg(
            output_dir=output_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )
        print("[server] norm tracking enabled (no clipping)", flush=True)
    elif defense_type == "norm_clipping":
        strategy = NormClippedFedAvg(
            clip_norm=clip_norm,
            output_dir=output_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )
        print(
            f"[server] defense enabled: norm_clipping (clip_norm={clip_norm})",
            flush=True,
        )
    elif defense_type == "fed_median":
        from flwr.serverapp.strategy import FedMedian

        strategy = FedMedian(**common_kwargs)
        print("[server] defense enabled: FedMedian", flush=True)
    elif defense_type == "fed_trimmed_avg":
        from flwr.serverapp.strategy import FedTrimmedAvg

        beta = float(run_config.get("trimmed-avg-beta", 0.2))
        strategy = FedTrimmedAvg(beta=beta, **common_kwargs)
        print(
            f"[server] defense enabled: FedTrimmedAvg (beta={beta})",
            flush=True,
        )
    elif defense_type == "krum":
        from flwr.serverapp.strategy import Krum

        num_malicious = int(run_config.get("num-malicious-nodes", 0))
        strategy = Krum(num_malicious_nodes=num_malicious, **common_kwargs)
        print(
            f"[server] defense enabled: Krum (num_malicious_nodes={num_malicious})",
            flush=True,
        )
    elif defense_type == "multi_krum":
        from flwr.serverapp.strategy import MultiKrum

        num_malicious = int(run_config.get("num-malicious-nodes", 0))
        num_to_select = int(run_config.get("krum-num-to-select", 5))
        strategy = MultiKrum(
            num_malicious_nodes=num_malicious,
            num_nodes_to_select=num_to_select,
            **common_kwargs,
        )
        print(
            f"[server] defense enabled: MultiKrum "
            f"(num_malicious={num_malicious}, num_to_select={num_to_select})",
            flush=True,
        )
    elif defense_type == "bulyan":
        from flwr.serverapp.strategy import Bulyan

        num_malicious = int(run_config.get("num-malicious-nodes", 0))
        strategy = Bulyan(num_malicious_nodes=num_malicious, **common_kwargs)
        print(
            f"[server] defense enabled: Bulyan (num_malicious_nodes={num_malicious})",
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

    # Save trigger visualization once at startup (attack-agnostic interface)
    attack_type = str(run_config.get("attack-type", "none"))
    if attack_type in ("pixel_backdoor",):
        trigger_size = int(run_config.get("trigger-size", 4))
        trigger_value = float(run_config.get("trigger-value", 1.0))
        trigger_position = str(run_config.get("trigger-position", "bottom-right"))
        viz_trigger_fn = make_pixel_trigger_fn(
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
        )
        viz_testloader = get_global_testloader(
            data_root=data_root,
            batch_size=16,
            image_size=int(run_config["image-size"]),
            num_workers=0,
            download=False,
        )
        save_attack_comparison(
            dataloader=viz_testloader,
            trigger_fn=viz_trigger_fn,
            output_path=output_dir,
            attack_name=attack_type,
        )

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