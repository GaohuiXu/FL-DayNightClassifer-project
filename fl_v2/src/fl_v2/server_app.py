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
from fl_v2.attacks_defenses import (
    dba_subregions,
    make_pixel_trigger_fn,
    parse_client_ids,
)
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
    FlameFedAvg,
    FoolsGoldFedAvg,
    NormClippedFedAvg,
    NormTrackingBulyan,
    NormTrackingFedAvg,
    NormTrackingFedMedian,
    NormTrackingFedTrimmedAvg,
    NormTrackingMultiKrum,
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
    # Cycle-02 gradient-space instrumentation knob (NormTracking family only).
    topk_energy_k = int(run_config.get("topk-energy-k", 4096))

    # --- Custom strategies (extra kwargs: exp_dir, seed, etc.) ---
    if defense_type == "none":
        return NormTrackingFedAvg(
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    if defense_type == "norm_clipping":
        return NormClippedFedAvg(
            clip_norm=float(run_config.get("clip-norm", 100.0)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    # --- Custom robust aggregation strategies (with norm logging) ---
    if defense_type == "fed_median":
        return NormTrackingFedMedian(
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    if defense_type == "fed_trimmed_avg":
        return NormTrackingFedTrimmedAvg(
            beta=float(run_config.get("trimmed-avg-beta", 0.2)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    if defense_type == "bulyan":
        return NormTrackingBulyan(
            num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    # --- Cycle-02 gradient-space defenses (WS5) ---
    if defense_type == "foolsgold":
        return FoolsGoldFedAvg(
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    if defense_type == "flame":
        return FlameFedAvg(
            noise_multiplier=float(run_config.get("flame-noise-multiplier", 0.001)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    # --- Flower built-in Krum / MultiKrum (wrapped for client-metric capture) ---
    if defense_type == "krum":
        return CapturedKrum(
            num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)),
            **common_kwargs,
        )

    if defense_type == "multi_krum":
        return NormTrackingMultiKrum(
            num_malicious_nodes=int(run_config.get("num-malicious-nodes", 0)),
            num_nodes_to_select=int(run_config.get("krum-num-to-select", 5)),
            output_dir=exp_dir,
            experiment_name=experiment_name,
            seed=seed,
            topk_energy_k=topk_energy_k,
            **common_kwargs,
        )

    raise ValueError(
        f"Unsupported defense-type: {defense_type!r}. "
        f"Available: none, norm_clipping, fed_median, fed_trimmed_avg, "
        f"krum, multi_krum, bulyan, foolsgold, flame"
    )


def _summarize_update_metrics(
    norm_record: dict, malicious_ids: set
) -> dict:
    """Summarise one norm-log round record into malicious-vs-honest scalars.

    ``norm_record`` is the per-round dict produced by the NormTracking
    strategies' ``_compute_and_log_norms`` (keys: ``partition_ids``,
    ``original_norms``, ``cosine_to_mean``, ``topk_energy_frac``,
    ``pairwise_cosine``). Per-client metrics are split by partition-id into
    malicious (id in ``malicious_ids``) and honest groups; the return is a
    flat dict of group-mean scalars for wandb (the ``updates/*`` namespace).

    Logging / analysis only — never touches aggregation. Honest-group keys
    are always emitted; malicious-group keys only when malicious clients
    participated this round, so clean runs degrade gracefully.
    """
    pids = norm_record.get("partition_ids")
    if not pids:
        return {}
    n = len(pids)
    mal = [i for i in range(n) if int(pids[i]) in malicious_ids]
    hon = [i for i in range(n) if int(pids[i]) not in malicious_ids]
    out: dict = {}

    def _grp_mean(values, idx):
        vals = [values[i] for i in idx if values[i] is not None]
        return sum(vals) / len(vals) if vals else None

    for key, short in (
        ("original_norms", "norm"),
        ("cosine_to_mean", "cos2mean"),
        ("topk_energy_frac", "topk_energy"),
    ):
        values = norm_record.get(key)
        if not values or len(values) != n:
            continue
        h = _grp_mean(values, hon)
        if h is not None:
            out[f"{short}_honest"] = float(h)
        if mal:
            m = _grp_mean(values, mal)
            if m is not None:
                out[f"{short}_malicious"] = float(m)

    pw = norm_record.get("pairwise_cosine")
    if pw and len(pw) == n:
        def _pair_mean(rows, cols, skip_diag):
            acc = [
                pw[i][j]
                for i in rows
                for j in cols
                if not (skip_diag and i == j)
            ]
            return sum(acc) / len(acc) if acc else None

        hh = _pair_mean(hon, hon, True)
        if hh is not None:
            out["pairwise_honest_honest"] = float(hh)
        if len(mal) >= 2:
            mm = _pair_mean(mal, mal, True)
            if mm is not None:
                out["pairwise_malicious_malicious"] = float(mm)
        if mal and hon:
            mh = _pair_mean(mal, hon, False)
            if mh is not None:
                out["pairwise_malicious_honest"] = float(mh)
    return out


def _summarize_defense_decision(norm_record: dict, malicious_ids: set) -> dict:
    """Detector TPR/FPR from a defense's per-round admit decision.

    Reads ``partition_ids`` (all clients that reached aggregation this
    round) and ``admitted_partition_ids`` (those the defense kept) from
    the strategy's norm-log record, splitting them against the
    ground-truth ``malicious_ids`` (which the harness knows but the
    defense does not). Returns:

    - ``defense_tpr``: malicious REJECTED / malicious total — security.
      The BackdoorIndicator Table-1 metric: collapses to 0 when the
      defense can no longer separate (small-LR) malicious updates.
    - ``defense_fpr``: honest REJECTED / honest total — utility cost /
      collateral. Has a nonzero clean-round baseline in non-IID FL, so
      interpret relative to the no-attack / pre-window value.
    - ``defense_n_admitted`` / ``defense_n_total``: raw counts.

    Returns {} when the record carries no admit decision (non-FLAME
    defenses, or round 0). Logging / analysis only — never aggregation.
    """
    pids = norm_record.get("partition_ids")
    admitted = norm_record.get("admitted_partition_ids")
    if not pids or admitted is None:
        return {}
    admitted_set = {int(a) for a in admitted}
    mal_total = [int(p) for p in pids if int(p) in malicious_ids]
    hon_total = [int(p) for p in pids if int(p) not in malicious_ids]
    out: dict = {
        "defense_n_admitted": float(len(admitted_set)),
        "defense_n_total": float(len(pids)),
    }
    if mal_total:
        mal_admitted = sum(1 for p in mal_total if p in admitted_set)
        out["defense_tpr"] = float(
            (len(mal_total) - mal_admitted) / len(mal_total)
        )
    if hon_total:
        hon_admitted = sum(1 for p in hon_total if p in admitted_set)
        out["defense_fpr"] = float(
            (len(hon_total) - hon_admitted) / len(hon_total)
        )
    return out


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
    # Cycle-02: malicious partition-ids — used only to split the per-round
    # gradient-space metrics into malicious-vs-honest summaries for wandb.
    malicious_ids = parse_client_ids(str(run_config.get("malicious-client-ids", "")))
    trigger_size = int(run_config.get("trigger-size", 4))
    trigger_value = float(run_config.get("trigger-value", 1.0))
    trigger_position = str(run_config.get("trigger-position", "bottom-right"))

    # Number of CPU workers for the server-side test DataLoader. The default
    # (4) parallelises preprocessing so the GPU stays fed; see
    # docs/cycle_02_gpu_efficiency_investigation.md for the rationale.
    num_workers = int(run_config.get("num-workers", 4))

    testloader = get_global_testloader(
        data_root=data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        download=False,
    )

    criterion = nn.CrossEntropyLoss()

    # Build trigger function for ASR if backdoor attack is active.
    # pixel / model_replacement / neurotoxin all use the corner square.
    # DBA measures ASR with the union of the per-client sub-triggers:
    # dba_subregions is the single source of truth (its union_rect is the
    # full corner square the client sub-rects tile), so the train-time and
    # eval-time triggers cannot drift apart.
    trigger_fn = None
    if attack_type in ("pixel_backdoor", "model_replacement", "neurotoxin"):
        trigger_fn = make_pixel_trigger_fn(
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
        )
    elif attack_type == "dba":
        _per_client_rects, union_rects = dba_subregions(
            trigger_size,
            trigger_position,
            str(run_config.get("dba-grid", "1x1")),
            image_size,
            pattern=str(run_config.get("dba-pattern", "corner-grid")),
        )
        trigger_fn = make_pixel_trigger_fn(
            trigger_size=trigger_size,
            trigger_value=trigger_value,
            trigger_position=trigger_position,
            trigger_rects=union_rects,
        )

    # Parse checkpoint rounds for periodic saving (e.g., "0,5,10,25,50,75,100")
    checkpoint_rounds_str = str(run_config.get("checkpoint-rounds", ""))
    checkpoint_rounds: set[int] = set()
    if checkpoint_rounds_str.strip():
        checkpoint_rounds = {int(r.strip()) for r in checkpoint_rounds_str.split(",") if r.strip()}

    # C4: server-side mirror of the client R1 model cache. evaluate_fn is
    # called once per round; without the cache it allocates a fresh
    # ResNet18 (~95 ms) every time. The state_dict load + .to(device)
    # immediately afterward overwrites every parameter and BN buffer, so
    # the cached model is functionally identical to a fresh one. Held in
    # the closure so it persists across rounds within the same server
    # actor.
    _eval_model_cache = {"model": None}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        # pretrained=False — init is overwritten immediately by load_state_dict.
        # canonical_conv1 must match the running federation's architecture so
        # that state_dict keys align.
        if _eval_model_cache["model"] is None:
            model = create_model(
                model_type,
                num_classes=num_classes,
                canonical_conv1=canonical_conv1,
            )
            model.to(device)
            _eval_model_cache["model"] = model
        model = _eval_model_cache["model"]
        model.load_state_dict(arrays.to_torch_state_dict())

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

        # Cycle-02: summarise this round's gradient-space metrics into
        # malicious-vs-honest scalars for wandb. The strategy's _norm_history
        # last entry is this round's record (aggregate_train ran just before
        # this callback). Fully guarded — logging only, never blocks a round.
        update_metrics = None
        norm_history = getattr(strategy, "_norm_history", None)
        if norm_history:
            last_record = norm_history[-1]
            if int(last_record.get("round", -1)) == int(server_round):
                try:
                    update_metrics = _summarize_update_metrics(
                        last_record, malicious_ids
                    )
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(
                        f"[server] update-metric summary skipped: {e!r}",
                        flush=True,
                    )
                # Cycle-03 WS-B: a defense's per-round admit decision (FLAME
                # logs admitted_partition_ids) -> detector TPR/FPR vs the
                # ground-truth malicious set. Merged into metrics_dict so it
                # rides into rounds.csv + summary.json + wandb server/*.
                # Absent for non-FLAME defenses / round 0 (keys just omitted).
                try:
                    defense_metrics = _summarize_defense_decision(
                        last_record, malicious_ids
                    )
                    metrics_dict.update(defense_metrics)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    print(
                        f"[server] defense-decision summary skipped: {e!r}",
                        flush=True,
                    )

        logger.log_round(
            server_round,
            metrics_dict,
            client_metrics=client_metrics,
            update_metrics=update_metrics,
        )

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
    if attack_type in ("pixel_backdoor", "model_replacement", "dba", "neurotoxin"):
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