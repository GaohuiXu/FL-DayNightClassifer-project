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

from fl_v3.training.tasks import get_task, load_trainable_state_dict, trainable_state_dict
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
    # DT3-A: the global arrays are the TRAINABLE-ONLY update vector (the frozen
    # backbone is reconstructed locally by build_model). Load BY NAME with
    # strict=False (a 62-key dict into the full model RAISES under strict=True)
    # + assert the load was clean (no unexpected keys; no trainable key unfilled).
    load_trainable_state_dict(model, arrays)


@app.train()
def train(msg: Message, context: Context) -> Message:
    run_config = context.run_config
    client_id = _get_client_id(context)
    server_round = int(msg.content["config"].get("server-round", 0))

    # Per-call deterministic seeding (mirrors the oracle): each Ray actor would
    # otherwise seed torch from the OS clock.
    seed_everything(derive_seed(int(run_config.get("seed", 42)), client_id, server_round))
    enforce_determinism(
        strict=truthy(run_config.get("determinism-strict", True)),
        precision=str(run_config.get("precision", "fp16")),
    )

    import time as _time
    _t_phase = _time.perf_counter()
    device = _device(run_config)
    task = get_task(str(run_config["task-type"]))
    model = task.build_model(run_config).to(device)
    _load_arrays_into(model, msg.content["arrays"])
    # L8 (D15): opt-in torch.compile of the camera backbone in the FL TRAIN path. The model is rebuilt
    # per client/round, so this re-traces each time — but a PERSISTENT inductor cache
    # (TORCHINDUCTOR_CACHE_DIR, set in the launcher; the Ray actor persists across rounds) makes the
    # expensive kernel compile a cache HIT after the first client → only the cheap dynamo re-trace
    # recurs. fp16-only (compile is off the strict byte-identity dev path). Default off (explicit opt-in
    # per the D16 tooling envelope: torch.compile stays opt-in with an eager fallback).
    _can_compile = device.type == "cuda" and str(run_config.get("precision", "fp16")) == "fp16"
    if _can_compile and truthy(run_config.get("compile-backbone", False)):
        model.camera_backbone = torch.compile(model.camera_backbone)
    # MCR P3: compile the STATIC-shape BEV conv stack too (camera_neck+fusion+bev_neck+head + the dense
    # lidar_backbone) — the profiler measured compile+sdpa+bev = 1.45× (353→244 ms/step) on bb02d at batch-4,
    # vs 1.19× for backbone-only. These subgraphs have fixed BEV shapes (512²/256²) ⇒ no recompilation; the
    # variable-pillar PFN (lidar_encoder) + ragged LSS splat (view_transform) stay eager. Persistent inductor
    # cache (TORCHINDUCTOR_CACHE_DIR) makes the kernel compile a cache-hit after the first client.
    if _can_compile and truthy(run_config.get("det-compile-bev", False)):
        model.camera_neck = torch.compile(model.camera_neck)
        model.fusion = torch.compile(model.fusion)
        model.bev_neck = torch.compile(model.bev_neck)
        model.head = torch.compile(model.head)
        if getattr(model, "lidar_backbone", None) is not None:
            model.lidar_backbone = torch.compile(model.lidar_backbone)
    criterion = task.build_criterion(run_config)
    cdata = task.client_data(client_id, run_config)
    _t_build = _time.perf_counter() - _t_phase   # model rebuild + load + compile-wrap + loader setup (FL overhead)

    from fl_v3.training.loop import train_local

    config: ConfigRecord = msg.content["config"]
    _t_train0 = _time.perf_counter()
    res = train_local(
        model,
        cdata.trainloader,
        criterion,
        device,
        num_epochs=int(config.get("num-local-epochs", run_config.get("num-local-epochs", 1))),
        # The per-round LR is broadcast by the strategy in ``config`` (MCR P3 global cosine schedule);
        # it falls back to the static run-config LR when the schedule is constant.
        learning_rate=float(config.get("learning-rate", run_config.get("learning-rate", 1e-3))),
        weight_decay=float(config.get("weight-decay", run_config.get("weight-decay", 0.0))),
        valloader=cdata.valloader,
        # MCR P3 (D17) FL client-recipe knobs (default-off ⇒ byte-identical to the pre-MCR FL path):
        grad_clip_norm=float(run_config.get("grad-clip-norm", 0.0)),
        backbone_lr_mult=float(run_config.get("det-backbone-lr-mult", 1.0)),
        optimizer_name=str(run_config.get("det-optimizer", "adam")),
        precision=str(run_config.get("precision", "fp16")),
    )

    _t_train = _time.perf_counter() - _t_train0   # local-training wall (loader + fwd/bwd/opt over the shard)
    # MCR P3 profiling telemetry (determinism-neutral — perf_counter only): the FL-overhead split the
    # single-GPU profiler can't see (model REBUILD ×N/round vs local training). Printed + in the metrics.
    print(f"[client-prof] pid={client_id} round={server_round} n={cdata.num_train} "
          f"build={_t_build:.2f}s train={_t_train:.2f}s", flush=True)

    # DT3-A: reply with the TRAINABLE-ONLY vector (the 62 non-backbone tensors),
    # not the full state_dict — so the frozen backbone never rides the wire and the
    # gradient-space metrics are trainable-only by construction.
    arrays = ArrayRecord(trainable_state_dict(model))
    metrics = MetricRecord(
        {
            "num-examples": cdata.num_train,
            "train_loss": float(res["final_train_loss"]),
            "val_loss": float(res["final_val_loss"]),
            "build_s": float(_t_build),
            "train_s": float(_t_train),
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
    enforce_determinism(
        strict=truthy(run_config.get("determinism-strict", True)),
        precision=str(run_config.get("precision", "fp16")),
    )

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


@app.query()
def discover(msg: Message, context: Context) -> Message:
    """DT3-B discovery probe: report this node's deterministic ``partition-id``.

    The server-side strategy has no node_id→partition-id map (the partition-id lives
    only here in ``node_config``), and the simulation assigns random node_ids — so the
    deterministic sampler runs one cheap QUERY round at startup to learn, for THIS run,
    which node_id carries which partition-id. The handler does NO model build / training
    / RNG (it must stay free of the per-call seeding the train/eval paths use); it just
    echoes the partition-id in ``reply-meta`` (the same key the aggregator sorts on)."""
    pid = _get_client_id(context)
    reply_meta = ConfigRecord({"partition-id": int(pid)})
    metrics = MetricRecord({"partition-id": float(pid)})
    content = RecordDict({"metrics": metrics, "reply-meta": reply_meta})
    return Message(content=content, reply_to=msg)
