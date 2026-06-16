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
from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import derive_seed, enforce_determinism, seed_everything


def state_dict_to_numpy(state_dict) -> List[np.ndarray]:
    """Ordered list of numpy arrays from a torch state_dict."""
    return [v.detach().cpu().numpy().copy() for v in state_dict.values()]


def load_numpy_into(model: torch.nn.Module, params: List[np.ndarray]) -> None:
    """Load an ordered numpy param list into ``model`` (key/shape preserving)."""
    sd = model.state_dict()
    new = OrderedDict()
    for (k, ref), p in zip(sd.items(), params):
        t = torch.as_tensor(np.asarray(p), dtype=ref.dtype).reshape(ref.shape)
        new[k] = t
    model.load_state_dict(new)


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
    enforce_determinism(strict=strict_determinism)
    seed = int(run_config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(str(run_config.get("device", "cpu")))

    task = get_task(str(run_config["task-type"]))
    criterion = task.build_criterion(run_config)
    num_epochs = int(run_config.get("num-local-epochs", 1))
    lr = float(run_config.get("learning-rate", 1e-2))
    wd = float(run_config.get("weight-decay", 0.0))

    # Initial global model + params.
    global_model = task.build_model(run_config).to(device)
    global_params = state_dict_to_numpy(global_model.state_dict())

    from fl_v3.training.loop import train_local

    n_clients = task.num_clients(run_config)
    replies = []
    for cid in range(n_clients):
        leaf = derive_seed(seed, cid, server_round)
        seed_everything(leaf)  # per-client determinism (mirrors client_app)
        model = task.build_model(run_config).to(device)
        load_numpy_into(model, global_params)
        cdata = task.client_data(cid, run_config)
        res = train_local(
            model, cdata.trainloader, criterion, device,
            num_epochs=num_epochs, learning_rate=lr, weight_decay=wd,
            valloader=cdata.valloader,
        )
        replies.append(
            {
                "partition_id": int(cid),
                "params": state_dict_to_numpy(model.state_dict()),
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
        load_numpy_into(global_model, decision.new_global)
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
