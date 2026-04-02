from __future__ import annotations

import os
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
from fl_v2.models import create_model
from fl_v2.training import server_evaluate

from fl_v2.utils import (
    ensure_dir,
    ExperimentLogger,
    get_device,
    save_attack_comparison,
    save_json,
)

from fl_v2.strategy import (
    NormClippedFedAvg,
    NormTrackingBulyan,
    NormTrackingFedAvg,
    NormTrackingFedMedian,
    NormTrackingFedTrimmedAvg,
)


app = ServerApp()


def _build_initial_arrays(context: Context) -> ArrayRecord:
    """Create the initial global model parameters."""
    num_classes = int(context.run_config["num-classes"])
    model_type = str(context.run_config.get("model-type", "cnn"))
    model = create_model(model_type, num_classes=num_classes)
    print(f"[server] model: {model_type} ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
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


def _build_strategy(defense_type: str, run_config, common_kwargs: dict, exp_dir: str):
    """Instantiate the aggregation strategy for the given defense type."""
    experiment_name = str(run_config.get("experiment-name", "default"))
    seed = int(run_config["seed"])

    # --- Custom strategies (extra kwargs: exp_dir, seed, etc.) ---
    if defense_type == "none":
        return NormTrackingFedAvg(
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )

    if defense_type == "norm_clipping":
        return NormClippedFedAvg(
            clip_norm=float(run_config.get("clip-norm", 100.0)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )

    # --- Custom robust aggregation strategies (with norm logging) ---
    if defense_type == "fed_median":
        return NormTrackingFedMedian(
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )

    if defense_type == "fed_trimmed_avg":
        return NormTrackingFedTrimmedAvg(
            beta=float(run_config.get("trimmed-avg-beta", 0.2)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )

    if defense_type == "bulyan":
        return NormTrackingBulyan(
            num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            **common_kwargs,
        )

    # --- Flower built-in strategies (Krum/MultiKrum work correctly) ---
    _FLOWER_STRATEGIES = {
        "krum": ("Krum", {
            "num_malicious_nodes": ("num-malicious-nodes", 0, int),
        }),
        "multi_krum": ("MultiKrum", {
            "num_malicious_nodes": ("num-malicious-nodes", 0, int),
            "num_nodes_to_select": ("krum-num-to-select", 5, int),
        }),
    }

    if defense_type not in _FLOWER_STRATEGIES:
        raise ValueError(
            f"Unsupported defense-type: {defense_type!r}. "
            f"Available: none, norm_clipping, fed_median, fed_trimmed_avg, "
            f"krum, multi_krum, bulyan"
        )

    class_name, param_spec = _FLOWER_STRATEGIES[defense_type]
    import flwr.serverapp.strategy as flwr_strat
    strategy_cls = getattr(flwr_strat, class_name)

    extra_kwargs = {
        k: cast_fn(run_config.get(config_key, default))
        for k, (config_key, default, cast_fn) in param_spec.items()
    }

    return strategy_cls(**extra_kwargs, **common_kwargs)


def _server_side_evaluate_fn(context: Context, logger: ExperimentLogger):
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
    model_type = str(run_config.get("model-type", "cnn"))
    device = get_device(run_config)

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
        model = create_model(model_type, num_classes=num_classes)
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)

        # Single-pass evaluation: accuracy + TCA + ASR in one dataloader iteration
        results = server_evaluate(
            model=model,
            testloader=testloader,
            criterion=criterion,
            device=device,
            target_label=backdoor_target_label,
            trigger_fn=trigger_fn,
        )

        print(f"\n── Server Eval {'─' * 45}", flush=True)
        print(
            f"[Server] round={server_round}  "
            f"test_loss={results['test_loss']:.4f}  "
            f"test_acc={results['test_accuracy']:.2%}",
            flush=True,
        )
        print(
            f"[Server] round={server_round}  "
            f"target_class_acc={results['target_class_clean_accuracy']:.2%} "
            f"(n={int(results['target_class_num_samples'])})",
            flush=True,
        )
        if "asr" in results:
            print(
                f"[Server] round={server_round}  "
                f"ASR={results['asr']:.2%} "
                f"(n={int(results['asr_num_samples'])})",
                flush=True,
            )

        metrics_dict = {"server-round": int(server_round), **results}
        logger.log_round(server_round, metrics_dict)
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

    # Create experiment subdirectory and logger — all outputs go here
    logger = ExperimentLogger(run_config, output_dir)
    exp_dir = logger.exp_dir
    print(f"[server] output dir: {exp_dir}", flush=True)

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
        f"{exp_dir}/{experiment_name}_seed{seed}_client_label_histograms.json",
    )
    print("[server] histogram stats done", flush=True)

    print("[server] creating strategy", flush=True)
    defense_type = str(run_config.get("defense-type", "none"))

    common_kwargs = dict(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_train_nodes,
        min_available_nodes=min_available_nodes,
    )

    strategy = _build_strategy(defense_type, run_config, common_kwargs, exp_dir)
    print(f"[server] defense: {defense_type}", flush=True)

    print("[server] building initial arrays", flush=True)
    initial_arrays = _build_initial_arrays(context)
    train_config = _get_train_config(context)
    evaluate_config = _get_evaluate_config(context)
    evaluate_fn = _server_side_evaluate_fn(context, logger)

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
            output_path=exp_dir,
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

    # Save final model checkpoint for offline analysis
    if result.arrays:
        checkpoint_path = os.path.join(exp_dir, "checkpoints", "final_model.pt")
        torch.save(result.arrays.to_torch_state_dict(), checkpoint_path)
        print(f"[server] checkpoint saved → {checkpoint_path}", flush=True)

    logger.finalize()

    print("Federated training finished.")
    print(result)