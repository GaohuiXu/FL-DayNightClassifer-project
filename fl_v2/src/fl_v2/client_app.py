from __future__ import annotations

from typing import Tuple

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_v2.data.dataset import (
    ClientDataLoaders,
    build_client_index_map,
    get_client_dataloaders,
)
from fl_v2.models import GTSRBClassifier
from fl_v2.training import evaluate, train_local

from fl_v2.attacks_defenses import parse_client_ids


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


def _make_cache_key(run_config) -> str:
    """Build a hashable key from partition-relevant config values."""
    return (
        f"{run_config['data-root']}|"
        f"{run_config['num-clients']}|"
        f"{run_config['partition-mode']}|"
        f"{run_config['dirichlet-alpha']}|"
        f"{run_config['seed']}"
    )


def _get_device(run_config) -> torch.device:
    """Select device from run config, with safe fallback."""
    requested = str(run_config.get("device", "cpu")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


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
        seed = int(run_config["seed"])

        _index_map_cache = build_client_index_map(
            data_root=data_root,
            num_clients=num_clients,
            partition_mode=partition_mode,
            dirichlet_alpha=dirichlet_alpha,
            seed=seed,
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
        seed=seed,
        num_workers=0,
        download=False,
        attack_type=attack_type,
        label_flip_source=label_flip_source,
        label_flip_target=label_flip_target,
        label_flip_fraction=label_flip_fraction,
        is_malicious=is_malicious,
        backdoor_target_label=backdoor_target_label,
        poison_fraction=poison_fraction,
        trigger_size=trigger_size,
        trigger_position=trigger_position,
    )

    _client_data_cache[client_id] = client_data
    return client_id, client_data


def _load_model_from_message(msg: Message, context: Context) -> Tuple[GTSRBClassifier, torch.device]:
    """Instantiate model and load weights received from the server."""
    num_classes = int(context.run_config["num-classes"])
    device = _get_device(context.run_config)

    model = GTSRBClassifier(num_classes=num_classes)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    return model, device


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the global model on local client data."""
    model, device = _load_model_from_message(msg, context)
    client_id, client_data = _load_client_data(context)

    # Read static config from run_config
    run_config = context.run_config
    default_local_epochs = int(run_config["num-local-epochs"])
    default_lr = float(run_config["learning-rate"])
    default_weight_decay = float(run_config["weight-decay"])

    # Read dynamic config from the incoming message
    config: ConfigRecord = msg.content["config"]
    local_epochs = int(config.get("num-local-epochs", default_local_epochs))
    learning_rate = float(config.get("learning-rate", default_lr))
    weight_decay = float(config.get("weight-decay", default_weight_decay))
    server_round = int(config.get("server-round", 0))

    results = train_local(
        model=model,
        trainloader=client_data.trainloader,
        valloader=client_data.valloader,
        device=device,
        num_epochs=local_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    arrays = ArrayRecord(model.state_dict())
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