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
    render_label_histogram_panel,
    save_attack_comparison,
    save_json,
    truthy,
)

from fl_v2.strategy import (
    CapturedKrum,
    CapturedMultiKrum,
    NormClippedFedAvg,
    NormTrackingBulyan,
    NormTrackingFedAvg,
    NormTrackingFedMedian,
    NormTrackingFedTrimmedAvg,
)


app = ServerApp()


def _build_initial_arrays(context: Context) -> ArrayRecord:
    """Create the initial global model parameters.

    Reads ``pretrained-init`` from run_config (default false) so Cycle 02
    pretrained-pivot YAMLs can ship ImageNet weights for the backbone while
    Cycle 01 YAMLs continue to use random init. This is the only place the
    pretrained flag matters: every other ``create_model`` call immediately
    overwrites init weights via ``load_state_dict``.
    """
    num_classes = int(context.run_config["num-classes"])
    model_type = str(context.run_config.get("model-type", "cnn"))
    pretrained = truthy(context.run_config.get("pretrained-init", False))
    canonical_conv1 = truthy(context.run_config.get("canonical-conv1", False))
    model = create_model(
        model_type,
        num_classes=num_classes,
        pretrained=pretrained,
        canonical_conv1=canonical_conv1,
    )
    print(
        f"[server] model: {model_type} pretrained={pretrained} "
        f"canonical_conv1={canonical_conv1} "
        f"({sum(p.numel() for p in model.parameters()):,} params)",
        flush=True,
    )
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

    # --- Flower built-in Krum / MultiKrum (wrapped for client-metric capture) ---
    if defense_type == "krum":
        return CapturedKrum(
            num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)),
            **common_kwargs,
        )

    if defense_type == "multi_krum":
        return CapturedMultiKrum(
            num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)),
            num_nodes_to_select=int(run_config.get("krum-num-to-select", 5)),
            **common_kwargs,
        )

    raise ValueError(
        f"Unsupported defense-type: {defense_type!r}. "
        f"Available: none, norm_clipping, fed_median, fed_trimmed_avg, "
        f"krum, multi_krum, bulyan"
    )


def _server_side_evaluate_fn(context: Context, logger: ExperimentLogger, strategy):
    """
    Build a centralized evaluation callback.

    Flower strategies can receive an evaluate_fn callback in strategy.start(...).
    The callback takes (server_round, arrays) and returns a MetricRecord or None.

    The ``strategy`` argument is the same instance built by ``_build_strategy``;
    we read its ``_last_train_metrics`` slot (populated by aggregate_train) so
    client-aggregated metrics flow into ``logger.log_round`` and onward to wandb
    on the same server-round axis as the server-side eval metrics.
    """
    run_config = context.run_config
    data_root = str(run_config["data-root"])
    batch_size = int(run_config["batch-size"])
    image_size = int(run_config["image-size"])
    num_classes = int(run_config["num-classes"])
    model_type = str(run_config.get("model-type", "cnn"))
    # Architecture-determining flag (must match _build_initial_arrays); the
    # state_dict from the strategy will only load into a model with the
    # matching Sequential structure.
    canonical_conv1 = truthy(run_config.get("canonical-conv1", False))
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
    if attack_type in ("pixel_backdoor", "model_replacement"):
        trigger_fn = make_pixel_trigger_fn(
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
        )

    # Parse checkpoint rounds for periodic saving (e.g., "0,5,10,25,50,75,100")
    checkpoint_rounds_str = str(run_config.get("checkpoint-rounds", ""))
    checkpoint_rounds: set[int] = set()
    if checkpoint_rounds_str.strip():
        checkpoint_rounds = {int(r.strip()) for r in checkpoint_rounds_str.split(",") if r.strip()}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        # pretrained=False — init is overwritten immediately by load_state_dict.
        # canonical_conv1 must match the running federation's architecture so
        # that state_dict keys align.
        model = create_model(
            model_type,
            num_classes=num_classes,
            canonical_conv1=canonical_conv1,
        )
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

        # Pull the aggregated client MetricRecord that the strategy stashed
        # at the end of aggregate_train. None on round 0 (no train yet) or if
        # the strategy didn't expose the slot.
        client_metrics = getattr(strategy, "_last_train_metrics", None)

        logger.log_round(server_round, metrics_dict, client_metrics=client_metrics)

        # Save periodic checkpoint if this round is in the configured set
        if server_round in checkpoint_rounds:
            logger.save_checkpoint(server_round, arrays.to_torch_state_dict())

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
    # cuDNN convolution autotuner / atomic-add backwards are one source of
    # non-determinism after our per-call seeding. The wider catch is
    # `torch.use_deterministic_algorithms(True, warn_only=True)` which
    # forces deterministic alternatives for any op that has one (BatchNorm
    # backward, scatter_add, index_add, etc.) and warns on those that don't.
    # Together they aim at full bit-equality across same-seed runs even on
    # different physical GPUs. Same configuration is set inside @app.train()
    # in client_app.py — server actor is independent of Ray client actors
    # and needs its own toggle.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

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

    # Partition seed (decoupled from model-RNG seed). If `partition-seed` is
    # empty / unset, fall back to the run's `seed` (backward-compatible
    # behaviour: identical to the pre-2026-05-08 pipeline). When set
    # explicitly, the data partition uses this seed and the model RNG uses
    # `seed`, letting us isolate model-init variance from
    # data-distribution variance across cells. See
    # cycle_02_codebase_risk_audit.md C4.
    partition_seed_raw = run_config.get("partition-seed", "")
    if isinstance(partition_seed_raw, str) and partition_seed_raw.strip() == "":
        partition_seed = seed
        partition_seed_source = "fallback to run seed"
    else:
        partition_seed = int(partition_seed_raw)
        partition_seed_source = "explicit"
    print(
        f"[server] partition_seed={partition_seed} ({partition_seed_source}); "
        f"model RNG seed={seed}",
        flush=True,
    )

    print("[server] building histogram stats", flush=True)
    _, histograms, summary = build_client_index_map_with_stats(
        data_root=data_root,
        num_clients=num_clients,
        partition_mode=partition_mode,
        dirichlet_alpha=dirichlet_alpha,
        seed=partition_seed,
        download=False,
    )

    print("===== Client label histogram summary =====", flush=True)
    print(summary, flush=True)

    histogram_json_path = (
        f"{exp_dir}/{experiment_name}_seed{seed}_client_label_histograms.json"
    )
    save_json(histograms, histogram_json_path)
    # Render a heatmap PNG and upload to wandb (no-op when wandb disabled).
    num_classes = int(run_config.get("num-classes", 43))
    histogram_png_path = render_label_histogram_panel(
        histogram_json_path, num_classes=num_classes
    )
    if histogram_png_path:
        logger.log_image("client_label_histograms", histogram_png_path)
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
    evaluate_fn = _server_side_evaluate_fn(context, logger, strategy)

    # Save trigger visualization once at startup (attack-agnostic interface)
    attack_type = str(run_config.get("attack-type", "none"))
    if attack_type in ("pixel_backdoor", "model_replacement"):
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