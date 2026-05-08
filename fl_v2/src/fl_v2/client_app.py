from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_v2.data.dataset import (
    ClientDataLoaders,
    build_client_index_map,
    get_client_dataloaders,
)
from fl_v2.models import create_model
from fl_v2.training import evaluate, train_local
from fl_v2.utils import compute_lr, derive_seed, get_device, truthy

from fl_v2.attacks_defenses import (
    compute_replacement_scale,
    parse_client_ids,
    scale_client_update,
)


app = ClientApp()

# ---------------------------------------------------------------------------
# Module-level caches — avoid reloading dataset / re-partitioning every round.
# Safe because config is constant within a run and partitioning is
# deterministic (same seed → same result).  Each simulation worker process
# has its own copy of these globals.
# ---------------------------------------------------------------------------
_index_map_cache: dict[int, list[int]] | None = None
_index_map_cache_key: str | None = None
_client_data_cache: dict[int, ClientDataLoaders] = {}


def _resolve_partition_seed(run_config) -> int:
    """Return the partition seed (separate from model-RNG seed when set).

    `partition-seed` empty / unset ⇒ fall back to `seed` (backward
    compatible with all pre-2026-05-08 YAMLs). Set explicitly to
    decouple partition variance from model-RNG variance per the
    cycle_02_codebase_risk_audit.md C4 finding.
    """
    raw = run_config.get("partition-seed", "")
    if isinstance(raw, str) and raw.strip() == "":
        return int(run_config["seed"])
    return int(raw)


def _make_cache_key(run_config) -> str:
    """Build a hashable key from partition-relevant config values.

    Partition cache must include partition-seed (not just seed) so that
    a within-process re-init with a different partition seed forces a
    re-partition.
    """
    return (
        f"{run_config['data-root']}|"
        f"{run_config['num-clients']}|"
        f"{run_config['partition-mode']}|"
        f"{run_config['dirichlet-alpha']}|"
        f"{_resolve_partition_seed(run_config)}"
    )



def _get_client_id(context: Context) -> int:
    """Get client/partition id for the current node."""
    if "partition-id" in context.node_config:
        return int(context.node_config["partition-id"])

    # Fallback in case partition-id is unavailable
    return int(context.node_id)


def _load_client_data(context: Context):
    """Build dataloaders for the current client, with caching."""
    global _index_map_cache, _index_map_cache_key

    run_config = context.run_config
    client_id = _get_client_id(context)

    # Return cached dataloaders if available
    if client_id in _client_data_cache:
        return client_id, _client_data_cache[client_id]

    # Build or retrieve the index map (shared across clients in this worker)
    cache_key = _make_cache_key(run_config)
    if _index_map_cache is None or _index_map_cache_key != cache_key:
        data_root = str(run_config["data-root"])
        num_clients = int(run_config["num-clients"])
        partition_mode = str(run_config["partition-mode"])
        dirichlet_alpha = float(run_config["dirichlet-alpha"])
        partition_seed = _resolve_partition_seed(run_config)

        _index_map_cache = build_client_index_map(
            data_root=data_root,
            num_clients=num_clients,
            partition_mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
            seed=partition_seed,
            download=False,
        )
        _index_map_cache_key = cache_key

    # Build dataloaders for this specific client
    batch_size = int(run_config["batch-size"])
    image_size = int(run_config["image-size"])
    val_ratio = float(run_config["val-ratio"])
    seed = int(run_config["seed"])

    attack_type = str(run_config.get("attack-type", "none"))
    malicious_client_ids = parse_client_ids(
        str(run_config.get("malicious-client-ids", ""))
    )
    label_flip_source = int(run_config.get("label-flip-source", 1))
    label_flip_target = int(run_config.get("label-flip-target", 2))
    label_flip_fraction = float(run_config.get("label-flip-fraction", 0.3))

    # pixel backdoor config
    backdoor_target_label = int(run_config.get("backdoor-target-label", 0))
    poison_fraction = float(run_config.get("poison-fraction", 0.1))
    trigger_size = int(run_config.get("trigger-size", 4))
    trigger_value = float(run_config.get("trigger-value", 1.0))
    trigger_position = str(run_config.get("trigger-position", "bottom-right"))

    is_malicious = client_id in malicious_client_ids

    if attack_type != "none":
        print(
            f"[Client {client_id}] attack_type={attack_type}, "
            f"is_malicious={is_malicious}",
            flush=True,
        )

    client_data = get_client_dataloaders(
        client_id=client_id,
        client_index_map=_index_map_cache,
        data_root=str(run_config["data-root"]),
        batch_size=batch_size,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,  # model-side: drives DataLoader shuffle order
        partition_seed=_resolve_partition_seed(run_config),  # data-side: val split + poison/flip mask
        num_workers=int(run_config.get("num-workers", 4)),
        download=False,
        attack_type=attack_type,
        label_flip_source=label_flip_source,
        label_flip_target=label_flip_target,
        label_flip_fraction=label_flip_fraction,
        is_malicious=is_malicious,
        backdoor_target_label=backdoor_target_label,
        poison_fraction=poison_fraction,
        trigger_size=trigger_size,
        trigger_value=trigger_value,
        trigger_position=trigger_position,
    )

    _client_data_cache[client_id] = client_data
    return client_id, client_data


def _load_model_from_message(msg: Message, context: Context) -> Tuple[torch.nn.Module, torch.device]:
    """Instantiate model and load weights received from the server."""
    num_classes = int(context.run_config["num-classes"])
    model_type = str(context.run_config.get("model-type", "cnn"))
    # Architecture-determining flag — must match what the server's
    # _build_initial_arrays produced, otherwise state_dict keys mismatch.
    canonical_conv1 = truthy(context.run_config.get("canonical-conv1", False))
    device = get_device(context.run_config)

    model = create_model(
        model_type,
        num_classes=num_classes,
        canonical_conv1=canonical_conv1,
    )
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    return model, device


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the global model on local client data."""
    # Per-call deterministic seeding. Without this, two runs of the same
    # YAML produce different trajectories because (a) each Ray actor
    # initialises torch.default_generator from the OS clock, and
    # (b) DataLoader(shuffle=True, generator=None) inherits that state.
    # See docs/cycle_02_pivot_audit.md §1.3 for the empirical evidence.
    run_seed = int(context.run_config.get("seed", 42))
    client_id = _get_client_id(context)
    server_round = int(msg.content["config"].get("server-round", 0))
    leaf_seed = derive_seed(run_seed, client_id, server_round)
    random.seed(leaf_seed)
    np.random.seed(leaf_seed)
    torch.manual_seed(leaf_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(leaf_seed)
    # cuDNN deterministic + the wider torch.use_deterministic_algorithms
    # catch (BatchNorm backward, scatter_add, etc.). warn_only=True so any
    # op without a deterministic alternative warns rather than crashes —
    # we accept those few warnings in return for bit-reproducibility on
    # everything else. Setting once per call is idempotent and ~free.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    model, device = _load_model_from_message(msg, context)
    _, client_data = _load_client_data(context)

    # Cache the global state dict for model-replacement update scaling.
    # This is essentially free — `to_torch_state_dict()` already returns a new
    # dict and we keep a detached CPU copy so the later scaling is cheap and
    # does not interfere with training on GPU.
    global_state_dict = {
        k: v.detach().cpu().clone()
        for k, v in msg.content["arrays"].to_torch_state_dict().items()
    }

    # Read static config from run_config
    run_config = context.run_config
    default_local_epochs = int(run_config["num-local-epochs"])
    default_lr = float(run_config["learning-rate"])
    default_weight_decay = float(run_config["weight-decay"])
    trainable_layers = str(run_config.get("trainable-layers", "full_ft"))

    # Re-read attack metadata (these are already used internally by
    # _load_client_data, but we need them again here to decide whether to
    # apply model-replacement scaling at reply time).
    attack_type = str(run_config.get("attack-type", "none"))
    malicious_client_ids = parse_client_ids(
        str(run_config.get("malicious-client-ids", ""))
    )
    is_malicious = client_id in malicious_client_ids

    # Read dynamic config from the incoming message
    config: ConfigRecord = msg.content["config"]
    local_epochs = int(config.get("num-local-epochs", default_local_epochs))
    learning_rate = float(config.get("learning-rate", default_lr))
    weight_decay = float(config.get("weight-decay", default_weight_decay))
    server_round = int(config.get("server-round", 0))

    # Apply LR schedule if configured (strategy-agnostic: uses server-round
    # injected by Flower's base FedAvg, works with all built-in strategies)
    lr_schedule = str(run_config.get("lr-schedule", "none"))
    if lr_schedule != "none" and server_round > 0:
        learning_rate = compute_lr(
            base_lr=default_lr,
            server_round=server_round,
            num_rounds=int(run_config["num-server-rounds"]),
            schedule=lr_schedule,
            warmup_rounds=int(run_config.get("lr-warmup-rounds", 0)),
            lr_min=float(run_config.get("lr-min", 0.0001)),
        )

    results = train_local(
        model=model,
        trainloader=client_data.trainloader,
        valloader=client_data.valloader,
        device=device,
        num_epochs=local_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        trainable_layers=trainable_layers,
    )

    # --- Model replacement (Bagdasaryan) --------------------------------
    # If this is a malicious client in a model_replacement run, scale the
    # update w_sent = w_global + scale * (w_local - w_global) so that after
    # FedAvg averaging the malicious local model effectively replaces the
    # global model.
    local_state_dict = model.state_dict()
    reply_state_dict = local_state_dict
    if is_malicious and attack_type == "model_replacement":
        # Scale: either fixed from config (>0) or auto from participant counts.
        cfg_scale = float(run_config.get("model-replacement-scale", 0.0))
        if cfg_scale > 0:
            scale = cfg_scale
        else:
            num_clients_total = int(run_config["num-clients"])
            fraction_train = float(run_config.get("fraction-train", 1.0))
            num_participants = max(
                int(round(num_clients_total * fraction_train)), 1,
            )
            scale = compute_replacement_scale(
                num_participants=num_participants,
                num_malicious=len(malicious_client_ids),
                fallback=1.0,
            )
        # Move the local (on-device) tensors to CPU so the scaling arithmetic
        # matches the cached global state dict (CPU).
        local_state_cpu = {
            k: v.detach().cpu() for k, v in local_state_dict.items()
        }
        reply_state_dict = scale_client_update(
            global_state_dict=global_state_dict,
            local_state_dict=local_state_cpu,
            scale=scale,
        )
        print(
            f"[Client {client_id}] model replacement: "
            f"scale={scale:.3f}, round={server_round}",
            flush=True,
        )

    arrays = ArrayRecord(reply_state_dict)
    metrics = MetricRecord(
        {
            "num-examples": client_data.num_train_samples,
            "train_loss": float(results["final_train_loss"]),
            "train_accuracy": float(results["final_train_accuracy"]),
            "val_loss": float(results["final_val_loss"]),
            "val_accuracy": float(results["final_val_accuracy"]),
        }
    )

    content = RecordDict({"arrays": arrays, "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on local validation data."""
    model, device = _load_model_from_message(msg, context)
    client_id, client_data = _load_client_data(context)

    criterion = torch.nn.CrossEntropyLoss()
    eval_metrics = evaluate(
        model=model,
        dataloader=client_data.valloader,
        criterion=criterion,
        device=device,
    )

    server_round = int(msg.content["config"].get("server-round", 0))

    metrics = MetricRecord(
        {
            "num-examples": client_data.num_val_samples,
            "val_loss": float(eval_metrics["loss"]),
            "val_accuracy": float(eval_metrics["accuracy"]),
        }
    )

    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)