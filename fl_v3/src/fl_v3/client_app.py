"""Task-agnostic Flower ClientApp (fl_v3 T0).

NO hardcoded loss / num-classes / model. The model, criterion, and data all come
from the :class:`~fl_v3.training.tasks.Task` selected by ``task-type``. The
determinism harness (per-call ``derive_seed`` + ``enforce_determinism``) and the
load-bearing ``reply-meta/partition-id`` embedding are carried from the fl_v2
oracle.

Validated to import + construct at T0. The live Ray simulation is exercised at
T3 (compute node via SLURM); the in-process equivalent for login-node testing is
``fl_v3.engine.local_runner``.
"""
from __future__ import annotations

import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import derive_seed, enforce_determinism, seed_everything, truthy

app = ClientApp()


def _get_client_id(context: Context) -> int:
    """Deterministic 0..N-1 partition id for this node (fallback to node_id)."""
    if "partition-id" in context.node_config:
        return int(context.node_config["partition-id"])
    return int(context.node_id)


def _device(run_config) -> torch.device:
    want = str(run_config.get("device", "cpu"))
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_arrays_into(model: torch.nn.Module, arrays: ArrayRecord) -> None:
    # Load BY NAME (matches the fl_v2 oracle) — robust to any state_dict key
    # ordering, unlike a positional zip.
    model.load_state_dict(arrays.to_torch_state_dict())


@app.train()
def train(msg: Message, context: Context) -> Message:
    run_config = context.run_config
    client_id = _get_client_id(context)
    server_round = int(msg.content["config"].get("server-round", 0))

    # Per-call deterministic seeding (mirrors the oracle): each Ray actor would
    # otherwise seed torch from the OS clock.
    seed_everything(derive_seed(int(run_config.get("seed", 42)), client_id, server_round))
    enforce_determinism(strict=truthy(run_config.get("determinism-strict", True)))

    device = _device(run_config)
    task = get_task(str(run_config["task-type"]))
    model = task.build_model(run_config).to(device)
    _load_arrays_into(model, msg.content["arrays"])
    criterion = task.build_criterion(run_config)
    cdata = task.client_data(client_id, run_config)

    from fl_v3.training.loop import train_local

    config: ConfigRecord = msg.content["config"]
    res = train_local(
        model,
        cdata.trainloader,
        criterion,
        device,
        num_epochs=int(config.get("num-local-epochs", run_config.get("num-local-epochs", 1))),
        learning_rate=float(config.get("learning-rate", run_config.get("learning-rate", 1e-3))),
        weight_decay=float(config.get("weight-decay", run_config.get("weight-decay", 0.0))),
        valloader=cdata.valloader,
    )

    arrays = ArrayRecord(model.state_dict())
    metrics = MetricRecord(
        {
            "num-examples": cdata.num_train,
            "train_loss": float(res["final_train_loss"]),
            "val_loss": float(res["final_val_loss"]),
        }
    )
    # Cross-run-stable aggregation key (the residual-ε fix; see partition_sort_key).
    reply_meta = ConfigRecord({"partition-id": int(client_id)})
    content = RecordDict({"arrays": arrays, "metrics": metrics, "reply-meta": reply_meta})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    run_config = context.run_config
    client_id = _get_client_id(context)
    server_round = int(msg.content["config"].get("server-round", 0))

    seed_everything(derive_seed(int(run_config.get("seed", 42)), client_id, server_round))
    enforce_determinism(strict=truthy(run_config.get("determinism-strict", True)))

    device = _device(run_config)
    task = get_task(str(run_config["task-type"]))
    model = task.build_model(run_config).to(device)
    _load_arrays_into(model, msg.content["arrays"])
    criterion = task.build_criterion(run_config)
    cdata = task.client_data(client_id, run_config)

    from fl_v3.training.loop import evaluate as eval_loss

    em = eval_loss(model, cdata.valloader, criterion, device)
    metrics = MetricRecord(
        {"num-examples": cdata.num_val, "val_loss": float(em["loss"])}
    )
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)
