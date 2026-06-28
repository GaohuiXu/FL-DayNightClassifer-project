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
    from fl_v3.strategy.server_opt import build_server_optimizer

    experiment_name = str(run_config.get("experiment-name", "default"))
    seed = int(run_config.get("seed", 42))
    topk_energy_k = int(run_config.get("topk-energy-k", 4096))
    # MCR P3 (D17): the server optimizer (FedAdam) + server EMA + per-round client LR schedule. All
    # default to the identity / off state ⇒ the strategy is byte-identical to the pre-MCR FedAvg path.
    base = dict(
        output_dir=exp_dir,
        experiment_name=experiment_name,
        seed=seed,
        topk_energy_k=topk_energy_k,
        server_optimizer=build_server_optimizer(run_config),
        server_ema_decay=float(run_config.get("server-ema-decay", 0.0)),
        client_lr_schedule=str(run_config.get("client-lr-schedule", "constant")),
        client_base_lr=run_config.get("learning-rate") if str(run_config.get("client-lr-schedule", "constant")) != "constant" else None,
        num_rounds=int(run_config.get("num-server-rounds", 1)),
        client_lr_warmup_rounds=int(run_config.get("client-lr-warmup-rounds", 0)),
        client_lr_final_frac=float(run_config.get("client-lr-final-frac", 0.0)),
        log_gradient_metrics=truthy(run_config.get("log-gradient-metrics", True)),
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


def _parse_round_list(spec, num_rounds: int) -> set:
    """Parse a comma-separated ``snapshot-rounds`` spec (e.g. "10,15,20") into a set of round ints."""
    out: set = set()
    for tok in str(spec or "").replace(" ", "").split(","):
        if not tok:
            continue
        try:
            r = int(tok)
        except ValueError:
            continue
        if 1 <= r <= num_rounds:
            out.add(r)
    return out


def _save_round_snapshot(task, run_config, exp_dir, server_round, arrays, ema_arrays) -> None:
    """MCR P3: save the round's global (and server-EMA) as a self-contained checkpoint under
    ``<exp_dir>/round_<r>/`` so the PEAK round can be picked post-hoc (the FL analog of the centralized
    ``ema_ep{N}`` snapshots; the centralized model peaked mid-run, so the final round is not assumed best)."""
    from fl_v3.engine.local_runner import load_trainable_numpy, numpy_state_checksum
    from fl_v3.training.tasks import trainable_state_dict

    snap_dir = os.path.join(exp_dir, f"round_{int(server_round)}")
    os.makedirs(snap_dir, exist_ok=True)
    m = task.build_model(run_config).to("cpu")
    load_trainable_state_dict(m, arrays)
    torch.save(m.state_dict(), os.path.join(snap_dir, "final_model.pt"))
    arrs = [v.detach().cpu().numpy() for v in trainable_state_dict(m).values()]
    with open(os.path.join(snap_dir, "trainable_checksum.txt"), "w", encoding="utf-8") as f:
        f.write(numpy_state_checksum(arrs) + "\n")
    if ema_arrays is not None:
        em = task.build_model(run_config).to("cpu")
        load_trainable_numpy(em, list(ema_arrays))
        ema_dir = os.path.join(snap_dir, "ema")
        os.makedirs(ema_dir, exist_ok=True)
        torch.save(em.state_dict(), os.path.join(ema_dir, "final_model.pt"))
        earrs = [v.detach().cpu().numpy() for v in trainable_state_dict(em).values()]
        with open(os.path.join(ema_dir, "trainable_checksum.txt"), "w", encoding="utf-8") as f:
            f.write(numpy_state_checksum(earrs) + "\n")
    print(f"[Server] round {server_round}: saved snapshot -> {snap_dir}"
          f"{' (+ema)' if ema_arrays is not None else ''}", flush=True)


def _server_eval_fn(context: Context, task, exp_dir: str, strategy=None):
    run_config = context.run_config
    device = _device(run_config)
    mode = str(run_config.get("server-eval-mode", "all"))
    frequency = int(run_config.get("server-eval-frequency", 1))
    num_rounds = int(run_config.get("num-server-rounds", 1))
    snapshot_rounds = _parse_round_list(run_config.get("snapshot-rounds", ""), num_rounds)
    if mode not in _VALID_EVAL_MODES:
        raise ValueError(f"server-eval-mode={mode!r} not in {sorted(_VALID_EVAL_MODES)}")
    print(
        f"[Server] server-eval-mode={mode} frequency={frequency} snapshot-rounds={sorted(snapshot_rounds)} "
        f"(proxy metrics only; official mAP/NDS + ASR stay post-hoc on the snapshot checkpoints)",
        flush=True,
    )
    # Build the (cheap) eval loader lazily — only if at least one round will actually eval, so
    # an eval-mode=none run pays ZERO eval cost (no loader, no eval-set info load).
    cache = {"model": None, "eval_loader": None, "criterion": None}

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> Optional[MetricRecord]:
        # MCR P3: save the per-round snapshot (raw global + server-EMA) BEFORE the eval gate — snapshotting
        # is RNG-neutral (no model build into the training graph, no global-RNG draw) so it does not perturb
        # the FL_TRAINABLE_CHECKSUM, exactly like the proxy eval below.
        if int(server_round) in snapshot_rounds:
            ema_arrays = getattr(strategy, "_ema_arrays", None) if strategy is not None else None
            _save_round_snapshot(task, run_config, exp_dir, int(server_round), arrays, ema_arrays)
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
    precision = str(run_config.get("precision", "bf16"))
    seed_everything(seed)
    enforce_determinism(
        strict=truthy(run_config.get("determinism-strict", True)),
        precision=precision,
    )
    print(
        f"[server] precision={precision} precision_state={precision_state()}",
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
    evaluate_fn = _server_eval_fn(context, task, exp_dir, strategy)

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

        # MCR P3 (D17): if a server-side cross-round EMA was kept, save it as a SELF-CONTAINED eval
        # checkpoint under <exp_dir>/ema/ with its OWN trainable checksum — the FL analog of the
        # centralized EMA checkpoint. The reference is reported on whichever of {raw, EMA} peaks; the
        # launcher writes provenance.json into BOTH dirs so either can host a D10 readiness verdict.
        ema_arrays = getattr(strategy, "_ema_arrays", None)
        if ema_arrays is not None:
            from fl_v3.engine.local_runner import load_trainable_numpy

            ema_model = task.build_model(run_config).to("cpu")
            load_trainable_numpy(ema_model, list(ema_arrays))  # fp64 EMA → cast to each param dtype
            ema_dir = os.path.join(exp_dir, "ema")
            os.makedirs(ema_dir, exist_ok=True)
            torch.save(ema_model.state_dict(), os.path.join(ema_dir, "final_model.pt"))
            earrs = [v.detach().cpu().numpy() for v in trainable_state_dict(ema_model).values()]
            echk = numpy_state_checksum(earrs)
            with open(os.path.join(ema_dir, "trainable_checksum.txt"), "w", encoding="utf-8") as f:
                f.write(echk + "\n")
            print(f"[server] server-EMA checkpoint saved (decay={strategy.server_ema_decay}) "
                  f"-> {ema_dir}  EMA_TRAINABLE_CHECKSUM = {echk}", flush=True)
    print("Federated training finished.", flush=True)
