"""Task-agnostic Flower ServerApp (fl_v3 T0).

Builds the initial global model + the aggregation strategy from config, with NO
classification assumption: the model and the server-side eval metrics come from
the :class:`~fl_v3.training.tasks.Task` selected by ``task-type``, and the
strategy from ``defense-type`` via the defense registry. Server-side determinism
(seed + ``enforce_determinism``) is set once at startup.

Validated to import + construct at T0. The live Ray run is exercised at T3.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from fl_v3.training.tasks import (
    get_task,
    load_trainable_state_dict,
    trainable_state_dict,
)
from fl_v3.utils.runtime import (
    enforce_determinism,
    precision_state,
    seed_everything,
    truthy,
)

app = ServerApp()


def _require_reconstructible_frozen_backbone(run_config) -> None:
    """DT3-A guard: a frozen backbone must be reconstructed BYTE-IDENTICALLY per node.

    With the backbone excluded from the (trainable-only) update vector, every node and
    the server must rebuild the SAME frozen backbone. That holds only for the
    ImageNet-pretrained path (``det-pretrained-backbone=True``): a random-init "frozen"
    backbone is re-seeded per client (``build_model`` runs after the per-client
    ``derive_seed``) and would differ per node — corrupting eval and the no-op-aggregation
    claim. So for the AD task we REQUIRE pretrained whenever the backbone is frozen.
    """
    if str(run_config.get("task-type", "")) != "nuscenes_detection":
        return
    if truthy(run_config.get("det-freeze-backbone", True)) and not truthy(
        run_config.get("det-pretrained-backbone", True)
    ):
        raise RuntimeError(
            "det-pretrained-backbone=True is REQUIRED for FL with a frozen backbone "
            "(DT3-A): a random-init frozen backbone is re-seeded per client and differs "
            "across nodes, corrupting eval + the trainable-only-aggregation invariant."
        )


def _device(run_config) -> torch.device:
    want = str(run_config.get("device", "cpu"))
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_strategy(defense_type: str, run_config, common_kwargs: dict, exp_dir: str):
    """Instantiate the aggregation strategy for ``defense_type``.

    Imported here (not at module top) so the server app can be imported without
    ``flwr`` strategy internals being resolved until run time.
    """
    from fl_v3.strategy.flower_strategies import (
        FedMedianStrategy,
        FlameStrategy,
        FoolsGoldStrategy,
        MultiKrumStrategy,
        NormClipStrategy,
        NormTrackingFedAvg,
    )

    experiment_name = str(run_config.get("experiment-name", "default"))
    seed = int(run_config.get("seed", 42))
    topk_energy_k = int(run_config.get("topk-energy-k", 4096))
    base = dict(
        output_dir=exp_dir,
        experiment_name=experiment_name,
        seed=seed,
        topk_energy_k=topk_energy_k,
        **common_kwargs,
    )

    d = defense_type.lower()
    if d in ("none", "fedavg"):
        return NormTrackingFedAvg(**base)
    if d == "norm_clipping":
        return NormClipStrategy(clip_norm=float(run_config.get("clip-norm", 5.0)), **base)
    if d == "flame":
        return FlameStrategy(
            noise_multiplier=float(run_config.get("flame-noise-multiplier", 1e-6)), **base
        )
    if d == "foolsgold":
        return FoolsGoldStrategy(head_index=int(run_config.get("foolsgold-head-index", -2)), **base)
    if d == "fed_median":
        return FedMedianStrategy(**base)
    if d == "multi_krum":
        return MultiKrumStrategy(
            num_malicious=int(run_config.get("num-malicious-nodes", 0)),
            num_to_select=int(run_config.get("krum-num-to-select", 1)),
            **base,
        )
    raise ValueError(
        f"Unsupported defense-type {defense_type!r}. Available: none, fedavg, "
        "norm_clipping, flame, foolsgold, fed_median, multi_krum"
    )


_VALID_EVAL_MODES = frozenset({"none", "final", "every_n", "all"})


def should_server_eval(mode: str, frequency: int, server_round: int, num_rounds: int) -> bool:
    """Per-round server-eval policy (Cycle-04 D14 Phase-1 B). Pure → unit-testable.

    The per-round server eval computes only PROXY metrics (eval_loss, proxy_recall) — NOT the
    scientific result (official mAP/NDS + ASR are post-hoc on the final checkpoint). So it is a
    config policy, default ``none`` for trainval (the proxy is not worth ~2 h/run of decode):
      * ``none``    — never (the final official eval is run post-hoc by the readiness script).
      * ``final``   — only the last round (a single cheap end-curve point).
      * ``every_n`` — every ``frequency`` rounds AND the last round (the cheap convergence curve E
                      needs, paired with a small ``det-eval-limit`` subset).
      * ``all``     — every round (the legacy behavior; what the early references used).
    The initial round 0 (pre-training) is only evaluated under ``all`` (it carries no signal else).
    """
    if mode not in _VALID_EVAL_MODES:
        raise ValueError(f"server-eval-mode={mode!r} not in {sorted(_VALID_EVAL_MODES)}")
    if mode == "none":
        return False
    if mode == "all":
        return True
    if server_round <= 0:  # initial/untrained model — skip for final/every_n
        return False
    is_final = server_round >= num_rounds
    if mode == "final":
        return is_final
    # every_n
    freq = max(1, int(frequency))
    return is_final or (server_round % freq == 0)


def _server_eval_fn(context: Context, task, exp_dir: str):
    run_config = context.run_config
    device = _device(run_config)
    mode = str(run_config.get("server-eval-mode", "all"))
    frequency = int(run_config.get("server-eval-frequency", 1))
    num_rounds = int(run_config.get("num-server-rounds", 1))
    if mode not in _VALID_EVAL_MODES:
        raise ValueError(f"server-eval-mode={mode!r} not in {sorted(_VALID_EVAL_MODES)}")
    print(
        f"[Server] server-eval-mode={mode} frequency={frequency} "
        f"(proxy metrics only; official mAP/NDS + ASR stay post-hoc on the final checkpoint)",
        flush=True,
    )
    # Build the (cheap) eval loader lazily — only if at least one round will actually eval, so
    # an eval-mode=none run pays ZERO eval cost (no loader, no eval-set info load).
    cache = {"model": None, "eval_loader": None, "criterion": None}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        if not should_server_eval(mode, frequency, int(server_round), num_rounds):
            return None
        # RNG-neutral by construction: eval runs model.eval()+no_grad (no dropout, no optimizer,
        # no global-RNG draw) and every client re-seeds via derive_seed before its next training —
        # so whether this fires or not, the FL_TRAINABLE_CHECKSUM is unchanged (proven by the
        # eval-none-vs-all paired byte-identity run, run_b_eval_neutral_a40.sh).
        if cache["model"] is None:
            cache["model"] = task.build_model(run_config).to(device)
            cache["eval_loader"] = task.eval_loader(run_config)
            cache["criterion"] = task.build_criterion(run_config)
        model = cache["model"]
        # DT3-A: ``arrays`` is the trainable-only vector; load strict=False into the
        # cached full model (frozen backbone already reconstructed pretrained at build).
        load_trainable_state_dict(model, arrays)

        results = task.evaluate(model, cache["eval_loader"], cache["criterion"], device, run_config)
        print(f"[Server] round={server_round} eval={results}", flush=True)
        return MetricRecord({"server-round": int(server_round), **results})

    return evaluate_fn


@app.main()
def main(grid: Grid, context: Context) -> None:
    import logging

    logging.getLogger("flwr").setLevel(logging.WARNING)
    run_config = context.run_config

    seed = int(run_config.get("seed", 42))
    numeric_mode = str(run_config.get("numeric-mode", "fp32"))
    seed_everything(seed)
    enforce_determinism(
        strict=truthy(run_config.get("determinism-strict", True)),
        numeric_mode=numeric_mode,
        level=str(run_config.get("determinism-level", "strict")),
    )
    print(
        f"[server] numeric-mode={numeric_mode} precision_state={precision_state()}",
        flush=True,
    )

    _require_reconstructible_frozen_backbone(run_config)
    task = get_task(str(run_config["task-type"]))

    output_dir = str(run_config.get("output-dir", "./outputs"))
    exp_dir = os.path.join(output_dir, str(run_config.get("experiment-name", "default")))
    os.makedirs(exp_dir, exist_ok=True)

    common_kwargs = dict(
        fraction_train=float(run_config.get("fraction-train", 1.0)),
        fraction_evaluate=float(run_config.get("fraction-evaluate", 1.0)),
        min_train_nodes=int(run_config.get("min-train-nodes", 2)),
        # Read min-evaluate-nodes from its OWN key (not min-train-nodes); defaults to it.
        min_evaluate_nodes=int(
            run_config.get("min-evaluate-nodes", run_config.get("min-train-nodes", 2))
        ),
        min_available_nodes=int(run_config.get("min-available-nodes", 2)),
    )
    strategy = _build_strategy(
        str(run_config.get("defense-type", "none")), run_config, common_kwargs, exp_dir
    )

    # DT3-A: the FL update vector is the TRAINABLE-only state (62 tensors for the AD
    # detector); the frozen backbone is reconstructed per node and excluded.
    initial_arrays = ArrayRecord(trainable_state_dict(task.build_model(run_config)))
    train_config = ConfigRecord(
        {
            "num-local-epochs": int(run_config.get("num-local-epochs", 1)),
            "learning-rate": float(run_config.get("learning-rate", 1e-3)),
            "weight-decay": float(run_config.get("weight-decay", 0.0)),
        }
    )
    evaluate_config = ConfigRecord({})
    evaluate_fn = _server_eval_fn(context, task, exp_dir)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=int(run_config.get("num-server-rounds", 1)),
        train_config=train_config,
        evaluate_config=evaluate_config,
        evaluate_fn=evaluate_fn,
    )

    if result.arrays:
        # DT3-A: ``result.arrays`` is the aggregated TRAINABLE-only vector. Merge it into
        # a freshly-built FULL model (frozen backbone reconstructed pretrained) so the
        # saved checkpoint is self-contained and loads ``strict=True`` at T4.
        full_model = task.build_model(run_config).to("cpu")
        load_trainable_state_dict(full_model, result.arrays)
        ckpt = os.path.join(exp_dir, "final_model.pt")
        torch.save(full_model.state_dict(), ckpt)
        # The committed FL bit-determinism checksum = SHA-256 over the aggregated
        # trainable vector (NOT the frozen-backbone-dominated full model). Two same-seed
        # runs MUST produce the identical string (the crown-jewel artifact).
        from fl_v3.engine.local_runner import numpy_state_checksum

        tchk = numpy_state_checksum(list(result.arrays.to_numpy_ndarrays()))
        chk_path = os.path.join(exp_dir, "trainable_checksum.txt")
        with open(chk_path, "w", encoding="utf-8") as f:
            f.write(tchk + "\n")
        print(f"[server] checkpoint saved -> {ckpt}", flush=True)
        print(f"[server] FL_TRAINABLE_CHECKSUM = {tchk}", flush=True)
    print("Federated training finished.", flush=True)
