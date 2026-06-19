"""In-process FL round runner (fl_v3 T0) — login-node-safe, no Ray.

Drives one sequential clean FL round (build global → per-client train → defense
aggregate → server eval) entirely in one process, exercising the REAL
task-agnostic interface + the REAL defense cores. This is the login-node-runnable
proxy for the Flower/Ray path: the Flower ClientApp/ServerApp call the same
``Task`` + cores, but the live Ray simulation is heavy (ResNet/Flower/Ray is too
heavy for the login node) and is exercised on a compute node via SLURM at T3.

Determinism is enforced exactly as the Flower apps enforce it:
  * ``enforce_determinism`` at entry,
  * global ``seed_everything(seed)`` once,
  * per-client ``seed_everything(derive_seed(seed, client_id, server_round))``
    before each client's model build + training,
  * aggregation order pinned by sorting on partition-id.

Two same-config runs therefore produce a byte-identical aggregated state — the
FL-level bit-determinism the T0 smoke asserts via :func:`numpy_state_checksum`.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch

from fl_v3.strategy.defenses import (
    DefenseDecision,
    FoolsGoldState,
    fed_median_decision,
    fedavg_decision,
    flame_decision,
    multi_krum_decision,
    norm_clip_decision,
)
from fl_v3.strategy.sampling import SAMPLE_SALT_TRAIN, select_partition_ids
from fl_v3.training.tasks import (
    get_task,
    load_trainable_state_dict,
    trainable_state_dict,
)
from fl_v3.utils.runtime import derive_seed, enforce_determinism, seed_everything


def state_dict_to_numpy(state_dict) -> List[np.ndarray]:
    """Ordered list of numpy arrays from a torch state_dict."""
    return [v.detach().cpu().numpy().copy() for v in state_dict.values()]


def trainable_numpy(model: torch.nn.Module) -> List[np.ndarray]:
    """DT3-A: the trainable-only update vector as an ordered numpy list.

    The SAME 62-tensor vector the Flower client transmits — so the in-process runner
    aggregates the identical thing as the Ray path (the cross-check compares like with
    like). Degenerates to the full param list on a no-frozen model (``TinyMLP``)."""
    return [v.detach().cpu().numpy().copy() for v in trainable_state_dict(model).values()]


def load_numpy_into(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Load an ordered FULL-state numpy param list into ``model`` (key/shape preserving)."""
    sd = model.state_dict()
    new = OrderedDict()
    for (k, ref), p in zip(sd.items(), params):
        t = torch.as_tensor(np.asarray(p), dtype=ref.dtype).reshape(ref.shape)
        new[k] = t
    model.load_state_dict(new)


def load_trainable_numpy(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Load a trainable-only numpy list (DT3-A) into the full model (strict=False)."""
    tsd = trainable_state_dict(model)
    new = OrderedDict()
    for (k, ref), p in zip(tsd.items(), params):
        new[k] = torch.as_tensor(np.asarray(p), dtype=ref.dtype).reshape(ref.shape)
    load_trainable_state_dict(model, new)


def numpy_state_checksum(params: List[np.ndarray]) -> str:
    """SHA-256 over the ordered raw bytes of an aggregated param list.

    A single digest for two-run bit-identity comparison. Includes shape +
    dtype in the hash so a shape/dtype drift can't collide.
    """
    h = hashlib.sha256()
    for p in params:
        a = np.ascontiguousarray(p)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _dispatch_defense(
    defense: str,
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    partition_ids: List[int],
    num_examples: List[float],
    run_config: dict,
    seed: int,
    server_round: int,
    foolsgold_state: Optional[FoolsGoldState] = None,
) -> DefenseDecision:
    """Route to the matching defense core (same cores the Flower wrappers use)."""
    d = defense.lower()
    if d in ("none", "fedavg"):
        return fedavg_decision(global_params, client_params_list, weights=num_examples)
    if d == "norm_clipping":
        return norm_clip_decision(
            global_params, client_params_list,
            float(run_config.get("clip-norm", 5.0)), weights=num_examples,
        )
    if d == "flame":
        return flame_decision(
            global_params, client_params_list,
            noise_multiplier=float(run_config.get("flame-noise-multiplier", 1e-6)),
            seed=seed, server_round=server_round,
        )
    if d == "foolsgold":
        state = foolsgold_state or FoolsGoldState()
        return state.update_and_decide(
            global_params,
            client_params_list,
            partition_ids,
            head_index=int(run_config.get("foolsgold-head-index", -2)),
        )
    if d == "fed_median":
        return fed_median_decision(global_params, client_params_list)
    if d == "multi_krum":
        return multi_krum_decision(
            global_params, client_params_list,
            num_malicious=int(run_config.get("num-malicious-nodes", 0)),
            num_to_select=int(run_config.get("krum-num-to-select", 1)),
            weights=num_examples,
        )
    raise ValueError(f"Unknown defense {defense!r}")


def run_clean_round(
    run_config: dict,
    defense: str = "none",
    server_round: int = 1,
    strict_determinism: bool = True,
) -> Dict[str, object]:
    """Run one in-process clean FL round; return a deterministic summary.

    Returns ``eval`` (task metrics), ``client_train_losses``, ``agg_checksum``
    (for two-run bit-identity), ``decision`` (the DefenseDecision), and
    ``new_global`` (the aggregated params).
    """
    enforce_determinism(
        strict=strict_determinism,
        numeric_mode=str(run_config.get("numeric-mode", "fp32")),
    )
    seed = int(run_config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(str(run_config.get("device", "cpu")))

    task = get_task(str(run_config["task-type"]))
    criterion = task.build_criterion(run_config)
    num_epochs = int(run_config.get("num-local-epochs", 1))
    lr = float(run_config.get("learning-rate", 1e-2))
    wd = float(run_config.get("weight-decay", 0.0))

    # Initial global model + TRAINABLE-only params (DT3-A).
    global_model = task.build_model(run_config).to(device)
    global_params = trainable_numpy(global_model)

    from fl_v3.training.loop import train_local

    n_clients = task.num_clients(run_config)
    replies = []
    for cid in range(n_clients):
        leaf = derive_seed(seed, cid, server_round)
        seed_everything(leaf)  # per-client determinism (mirrors client_app)
        model = task.build_model(run_config).to(device)
        load_trainable_numpy(model, global_params)
        cdata = task.client_data(cid, run_config)
        res = train_local(
            model, cdata.trainloader, criterion, device,
            num_epochs=num_epochs, learning_rate=lr, weight_decay=wd,
            valloader=cdata.valloader,
        )
        replies.append(
            {
                "partition_id": int(cid),
                "params": trainable_numpy(model),
                "num_examples": int(cdata.num_train),
                "train_loss": float(res["final_train_loss"]),
            }
        )

    # Deterministic aggregation order (partition-id sort — the residual-ε fix).
    replies.sort(key=lambda r: r["partition_id"])
    client_params_list = [r["params"] for r in replies]
    partition_ids = [r["partition_id"] for r in replies]
    num_examples = [float(r["num_examples"]) for r in replies]

    decision = _dispatch_defense(
        defense, global_params, client_params_list, partition_ids,
        num_examples, run_config, seed, server_round,
    )

    summary: Dict[str, object] = {
        "defense": defense,
        "n_clients": n_clients,
        "client_train_losses": [r["train_loss"] for r in replies],
        "decision_valid": bool(decision.valid),
    }
    if decision.valid and decision.new_global is not None:
        load_trainable_numpy(global_model, decision.new_global)
        summary["eval"] = task.evaluate(
            global_model, task.eval_loader(run_config), criterion, device, run_config
        )
        summary["agg_checksum"] = numpy_state_checksum(decision.new_global)
        summary["new_global"] = decision.new_global
    else:
        summary["eval"] = None
        summary["agg_checksum"] = None
        summary["new_global"] = None
    summary["decision"] = decision
    return summary


def run_clean_rounds(
    run_config: dict,
    defense: str = "none",
    num_rounds: int = 2,
    fraction_train: float = 1.0,
    min_train_nodes: int = 2,
    strict_determinism: bool = True,
) -> Dict[str, object]:
    """Multi-round in-process FedAvg with the SAME DT3-B deterministic sampler as Ray.

    The login-node↔Ray cross-check substrate (SPEC §2.5): each round selects participants
    via ``select_partition_ids`` (identical to the Flower strategy), trains the sampled
    clients, aggregates the trainable-only vectors via the same defense core, sorted by
    partition-id. Two same-seed calls return an identical ``final_checksum``; run on the
    SAME A40 as the Ray driver, the ``final_checksum`` matches the Ray run's
    ``FL_TRAINABLE_CHECKSUM`` byte-for-byte (CPU↔A40 float drift otherwise — documented).
    """
    enforce_determinism(
        strict=strict_determinism,
        numeric_mode=str(run_config.get("numeric-mode", "fp32")),
    )
    seed = int(run_config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(str(run_config.get("device", "cpu")))

    task = get_task(str(run_config["task-type"]))
    criterion = task.build_criterion(run_config)
    num_epochs = int(run_config.get("num-local-epochs", 1))
    lr = float(run_config.get("learning-rate", 1e-2))
    wd = float(run_config.get("weight-decay", 0.0))

    from fl_v3.training.loop import train_local

    n_clients = task.num_clients(run_config)
    global_model = task.build_model(run_config).to(device)
    global_params = trainable_numpy(global_model)
    fg_state = FoolsGoldState() if defense.lower() == "foolsgold" else None

    rounds: List[dict] = []
    for server_round in range(1, int(num_rounds) + 1):
        pids = select_partition_ids(
            seed, server_round, n_clients, fraction_train, min_train_nodes,
            salt=SAMPLE_SALT_TRAIN,
        )
        replies = []
        for cid in pids:
            seed_everything(derive_seed(seed, cid, server_round))
            model = task.build_model(run_config).to(device)
            load_trainable_numpy(model, global_params)
            cdata = task.client_data(cid, run_config)
            res = train_local(
                model, cdata.trainloader, criterion, device,
                num_epochs=num_epochs, learning_rate=lr, weight_decay=wd,
                valloader=cdata.valloader,
            )
            replies.append({
                "partition_id": int(cid),
                "params": trainable_numpy(model),
                "num_examples": int(cdata.num_train),
                "train_loss": float(res["final_train_loss"]),
            })
        replies.sort(key=lambda r: r["partition_id"])  # partition-id aggregation order
        client_params_list = [r["params"] for r in replies]
        partition_ids = [r["partition_id"] for r in replies]
        num_examples = [float(r["num_examples"]) for r in replies]
        decision = _dispatch_defense(
            defense, global_params, client_params_list, partition_ids,
            num_examples, run_config, seed, server_round, foolsgold_state=fg_state,
        )
        if decision.valid and decision.new_global is not None:
            global_params = decision.new_global  # carry forward (mirrors strategy.start)
            agg_checksum = numpy_state_checksum(global_params)
        else:
            agg_checksum = None  # invalid round: global unchanged
        rounds.append({
            "round": server_round,
            "participants": list(partition_ids),
            "decision_valid": bool(decision.valid),
            "agg_checksum": agg_checksum,
        })

    load_trainable_numpy(global_model, global_params)
    final_eval = task.evaluate(
        global_model, task.eval_loader(run_config), criterion, device, run_config
    )
    return {
        "defense": defense,
        "n_clients": n_clients,
        "num_rounds": int(num_rounds),
        "fraction_train": float(fraction_train),
        "rounds": rounds,
        "final_checksum": numpy_state_checksum(global_params),
        "final_eval": final_eval,
        "new_global": global_params,
    }
