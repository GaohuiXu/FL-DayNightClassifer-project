"""T5 attack eval driver (fl_v3 T5) — the GATE measurements on the frozen subset at batch_size=1.

Multi-task (``--task``), all bound to the **literal** frozen subset ``2ad8f8da…`` + clean checkpoint
``a80466c3…`` (re-verified at load, §0.C6) and the provenance-verified trainval poisoned checkpoint
(§0.C8):

  * ``shard``       — per-target 5-condition ablation + occlusion (the fan-out worker; ``--shard i
                      --num-shards K``); writes ``ablation/ablation_shard_{i}.json``.
  * ``aggregate``   — combine the shards → the 5-condition table + the floor-corrected fusion-aware
                      verdict + the headline disappear-ASR (cond-4) + the occlusion-control ASR.
  * ``stealth``     — the POISONED model's clean DetectionEval (mAP/NDS + official car recall ≥ floor).
  * ``guards``      — cond-5a LiDAR-invariance + the camera-only clean-recall precondition.
  * ``viz``         — V5 + V3(trigger) for a few subset samples.
  * ``null-verify`` — fail closed: the legacy bare-state null checksum is not reinterpreted as an
                      S06 complete-checkpoint identity; a replacement protocol must be frozen first.

``batch_size=1`` is forced (the T4 batch-invariance protocol); the ASR scores on the UNEDITED val GT
with the subset's BOUND thresholds (§0.C7/§0.5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


_ALLOWED_EVAL_OVERRIDES = frozenset({"batch-size", "num-workers", "det-eval-limit"})
_CHECKPOINT_PREFLIGHTS: Dict[str, dict] = {}
_SHARD_SCHEMA_FULL = "s07b.t5.shard.full.v1"
_SHARD_SCHEMA_COND4 = "s07b.t5.shard.cond4.v1"
_SHARD_MODE_FULL = "full_five_condition"
_SHARD_MODE_COND4 = "cond4_only"


# ---------------------------------------------------------------------------
# config / model / subset plumbing
# ---------------------------------------------------------------------------
def _parse_scalar(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _load_config(path: str, overrides: List[str]) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for ov in overrides:
        k, _, v = ov.partition("=")
        cfg[k] = _parse_scalar(v)
    cfg["batch-size"] = 1  # T4 batch-invariance protocol (the whole ASR/ablation decode) — HARD-set
    cfg["det-eval-limit"] = 0
    assert int(cfg["batch-size"]) == 1, "the T5 ASR/ablation decode MUST run at batch_size=1 (T4 §5c)"
    return cfg


# The §0.C anti-gaming constants are HARD: any override (a config tweak after seeing the result) is a
# post-hoc-fitting attempt → the driver REFUSES to emit a verdict. δ_fusion is the load-bearing one.
_PINNED = {
    "attack-delta-fusion": 0.2, "attack-delta-fusion-mult": 2.0, "attack-viability-asr": 0.3,
    "attack-stealth-recall-floor": 0.75, "asr-false-disappear-max": 0.02,
    "attack-trigger-budget-frac": 0.30, "attack-delta-clean": 0.10,
    "attack-cond5a-recall-floor": 0.3,
}


def _assert_pinned_constants(cfg: dict) -> None:
    """RAISE if any §0.C anti-gaming constant deviates from its pinned value (no post-hoc fitting)."""
    bad = []
    for k, want in _PINNED.items():
        got = float(cfg.get(k, want))
        if abs(got - want) > 1e-12:
            bad.append(f"{k}={got} != PINNED {want}")
    if bad:
        raise RuntimeError("PINNED anti-gaming constant(s) overridden (§0.C4 — hard FAIL, no fitting to "
                           "the result): " + "; ".join(bad))


def _pinned_constants(cfg: dict) -> dict:
    return {k: float(cfg.get(k, v)) for k, v in _PINNED.items()}


def _load_json(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _device(cfg) -> torch.device:
    if str(cfg.get("device", "cuda")) == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _trainable_checksum(model) -> str:
    from fl_v3.engine.local_runner import numpy_state_checksum
    from fl_v3.training.tasks import trainable_state_dict
    return numpy_state_checksum([v.detach().cpu().numpy() for v in trainable_state_dict(model).values()])


def _checkpoint_preflight(caller_cfg: dict, checkpoint: str) -> dict:
    """Validate one complete checkpoint without constructing data or a model."""
    from fl_v3.config import resolve_config, verify_physical_data_identities
    from fl_v3.training.checkpoint import CHECKPOINT_SCHEMA, _FIELDS
    from fl_v3.utils.runtime import verify_runtime_dependency_identity

    before_sha = _checkpoint_file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    after_sha = _checkpoint_file_sha256(checkpoint)
    if after_sha != before_sha:
        raise RuntimeError("T5 checkpoint changed during authoritative preflight load")
    if not isinstance(payload, dict) or "schema" not in payload:
        raise RuntimeError("T5 refuses legacy/bare checkpoints; a complete S06 checkpoint is required")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"T5 refuses unsupported checkpoint schema {payload.get('schema')!r}")
    if set(payload) != _FIELDS:
        raise RuntimeError("T5 refuses partial checkpoints; the complete S06 field set is required")
    resolved = resolve_config(payload.get("resolved_config", {}))
    expected_metadata = {
        "resolved_config_sha256": resolved.sha256,
        "resolved_config": resolved.as_dict(),
        "model_mode": resolved.model_mode,
        "precision": resolved.precision,
        "data_identities": resolved.data_identities,
    }
    metadata_drift = [key for key, value in expected_metadata.items() if payload.get(key) != value]
    if metadata_drift:
        raise RuntimeError(f"T5 checkpoint embedded config/data metadata drift: {metadata_drift}")
    identity = payload.get("checkpoint_identity")
    if (
        not isinstance(identity, str) or len(identity) != 64
        or any(char not in "0123456789abcdef" for char in identity)
    ):
        raise RuntimeError("T5 checkpoint identity must be a lowercase SHA-256 string")
    if identity != resolved.sha256:
        raise RuntimeError("T5 checkpoint identity does not equal embedded resolved-config SHA-256")
    required_mappings = ("model", "optimizer", "scheduler", "grad_scaler", "training_state", "rng")
    malformed = [key for key in required_mappings if not isinstance(payload.get(key), dict)]
    if malformed:
        raise RuntimeError(f"T5 checkpoint component metadata is incomplete: {malformed}")
    expects_ema = resolved.data["training"]["ema_decay"] is not None
    if expects_ema != isinstance(payload.get("ema"), dict):
        raise RuntimeError("T5 checkpoint EMA presence differs from resolved training policy")
    strict = resolved.to_run_config()
    drift = [
        key for key in strict
        if key in caller_cfg and key not in _ALLOWED_EVAL_OVERRIDES
        and caller_cfg[key] != strict[key]
    ]
    if drift:
        raise RuntimeError(f"T5 caller/checkpoint resolved-config drift: {sorted(drift)}")
    verify_physical_data_identities(resolved)
    runtime_identity = verify_runtime_dependency_identity(strict)
    runtime_sha = hashlib.sha256(
        json.dumps(runtime_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": os.path.abspath(checkpoint),
        "resolved": resolved,
        "strict": strict,
        "resolved_sha256": resolved.sha256,
        "checkpoint_sha256": after_sha,
        "checkpoint_weights": strict["evaluation-checkpoint-weights"],
        "runtime_dependencies": runtime_identity,
        "runtime_dependencies_sha256": runtime_sha,
    }


def _validate_task_requirements(args) -> str:
    """Freeze required checkpoints and artifact mode before any side effect."""
    if args.task == "null-verify":
        return "null_fail_closed"
    if not args.checkpoint:
        raise RuntimeError(f"T5 task {args.task!r} requires a poisoned checkpoint")
    if args.task == "shard":
        if args.num_shards < 1 or args.shard < 0 or args.shard >= args.num_shards:
            raise RuntimeError("T5 shard index/count must satisfy 0 <= shard < num_shards")
        if args.cond4_only:
            if args.clean_checkpoint:
                raise RuntimeError("cond4-only shard is poison-only; clean checkpoint is forbidden")
            return _SHARD_MODE_COND4
        if not args.clean_checkpoint:
            raise RuntimeError("full five-condition shard requires both poison and clean checkpoints")
        return _SHARD_MODE_FULL
    if args.task == "viz":
        if not args.clean_checkpoint:
            raise RuntimeError("T5 viz requires both poison and clean checkpoints")
        return _SHARD_MODE_FULL
    if args.task in {"aggregate", "stealth", "guards"}:
        if args.clean_checkpoint:
            raise RuntimeError(f"T5 task {args.task!r} is poison-only; clean checkpoint is forbidden")
        return "poison_only"
    raise RuntimeError(f"unknown T5 task requirement {args.task!r}")


def _preflight_t5_checkpoints(cfg: dict, checkpoint: str, clean_checkpoint: Optional[str]) -> None:
    """Establish authoritative numeric/data/model identity before every T5 task."""
    _CHECKPOINT_PREFLIGHTS.clear()
    if not checkpoint:
        raise RuntimeError("T5 task requires an explicit complete S06 checkpoint")
    poison = _checkpoint_preflight(cfg, checkpoint)
    clean = _checkpoint_preflight(cfg, clean_checkpoint) if clean_checkpoint else None
    if clean is not None:
        if clean["resolved_sha256"] != poison["resolved_sha256"]:
            raise RuntimeError(
                "T5 clean/poison checkpoint resolved identities differ: "
                f"poison={poison['resolved_sha256']}, clean={clean['resolved_sha256']}"
            )
        if clean["checkpoint_weights"] != poison["checkpoint_weights"]:
            raise RuntimeError("T5 clean/poison checkpoint raw/EMA policies differ")
        if clean["runtime_dependencies_sha256"] != poison["runtime_dependencies_sha256"]:
            raise RuntimeError("T5 clean/poison runtime dependency identities differ")

    preserved = {key: cfg[key] for key in _ALLOWED_EVAL_OVERRIDES if key in cfg}
    cfg.update(poison["strict"])
    cfg.update(preserved)
    cfg["batch-size"] = 1
    cfg["det-eval-limit"] = 0
    cfg["runtime-dependencies-sha256"] = poison["runtime_dependencies_sha256"]
    cfg["checkpoint-sha256"] = poison["checkpoint_sha256"]
    cfg["checkpoint-weights"] = poison["checkpoint_weights"]
    if clean is not None:
        cfg["clean-checkpoint-sha256"] = clean["checkpoint_sha256"]
    _CHECKPOINT_PREFLIGHTS[poison["path"]] = poison
    if clean is not None:
        _CHECKPOINT_PREFLIGHTS[clean["path"]] = clean


def _require_preflight(checkpoint: str, clean_checkpoint: Optional[str] = None) -> None:
    required = [checkpoint, *([clean_checkpoint] if clean_checkpoint else [])]
    missing = [
        os.path.abspath(path) for path in required
        if not path or os.path.abspath(path) not in _CHECKPOINT_PREFLIGHTS
    ]
    if missing:
        raise RuntimeError(f"T5 authoritative checkpoint preflight missing: {missing}")


def _load_model(cfg, checkpoint: str, device):
    """Construct/load only from an already validated authoritative preflight."""
    from fl_v3.training.checkpoint import load_checkpoint
    from fl_v3.training.tasks import get_task
    from fl_v3.utils.runtime import make_grad_scaler

    _require_preflight(checkpoint)
    preflight = _CHECKPOINT_PREFLIGHTS[os.path.abspath(checkpoint)]
    resolved = preflight["resolved"]
    strict = preflight["strict"]
    drift = [
        key for key, expected in strict.items()
        if key not in _ALLOWED_EVAL_OVERRIDES and cfg.get(key) != expected
    ]
    if drift:
        raise RuntimeError(f"T5 authoritative config changed after preflight: {sorted(drift)}")
    if _checkpoint_file_sha256(checkpoint) != preflight["checkpoint_sha256"]:
        raise RuntimeError("T5 checkpoint changed after authoritative preflight")

    task = get_task("nuscenes_detection")
    model = task.build_model(strict).to(device)
    opt_spec = resolved.data["optimizer"]
    opt_cls = torch.optim.Adam if opt_spec["name"] == "adam" else torch.optim.AdamW
    optimizer = opt_cls(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(opt_spec["learning_rate"]), weight_decay=float(opt_spec["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = make_grad_scaler(device, resolved.precision)
    ema = None
    decay = resolved.data["training"]["ema_decay"]
    if decay is not None:
        from torch.optim.swa_utils import AveragedModel
        ema = AveragedModel(
            model,
            avg_fn=lambda old, new, _count, d=float(decay): d * old + (1.0 - d) * new,
            use_buffers=False,
        )
    _, identity = load_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler, ema=ema, config=resolved, map_location="cpu",
    )
    if identity != resolved.sha256:
        raise RuntimeError("T5 checkpoint identity does not equal its resolved-config SHA-256")
    if preflight["checkpoint_weights"] == "ema":
        if ema is None:
            raise RuntimeError("T5 EMA policy requested but checkpoint has no EMA")
        model.load_state_dict(ema.module.state_dict(), strict=True)
    if _checkpoint_file_sha256(checkpoint) != preflight["checkpoint_sha256"]:
        raise RuntimeError("T5 checkpoint changed during complete load")
    model.to(device).eval()
    return model


def _checkpoint_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_artifact_identity(checkpoint: str, selected_weights_checksum: str) -> dict:
    preflight = _CHECKPOINT_PREFLIGHTS[os.path.abspath(checkpoint)]
    return {
        "checkpoint_file_sha256": preflight["checkpoint_sha256"],
        "resolved_sha256": preflight["resolved_sha256"],
        "checkpoint_weights": preflight["checkpoint_weights"],
        "runtime_dependencies_sha256": preflight["runtime_dependencies_sha256"],
        "selected_weights_checksum": str(selected_weights_checksum),
    }


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _shard_artifact_path(output_dir: str, shard: int, count: int, mode: str) -> str:
    suffix = "full" if mode == _SHARD_MODE_FULL else "cond4"
    return os.path.join(
        output_dir, "ablation", f"ablation_shard_{shard}_of_{count}.{suffix}.json",
    )


def _shard_schema(mode: str) -> str:
    if mode == _SHARD_MODE_FULL:
        return _SHARD_SCHEMA_FULL
    if mode == _SHARD_MODE_COND4:
        return _SHARD_SCHEMA_COND4
    raise RuntimeError(f"unknown T5 shard artifact mode {mode!r}")


def _load_bound_full_shards(args, cfg: dict, subset: dict, poison_weights_checksum: str) -> List[dict]:
    """Validate the complete fan-out identity/row set before aggregation."""
    from fl_v3.attacks import fusion_ablation as FA

    count = int(args.num_shards)
    if count < 1:
        raise RuntimeError("aggregate num_shards must be >= 1")
    directory = os.path.join(args.output_dir, "ablation")
    files = sorted(
        os.path.join(directory, name) for name in os.listdir(directory)
        if name.startswith("ablation_shard_") and name.endswith(".json")
    )
    if len(files) != count:
        raise RuntimeError(f"aggregate requires exactly {count} unique shard artifacts, found {len(files)}")
    file_hashes = [_checkpoint_file_sha256(path) for path in files]
    if len(file_hashes) != len(set(file_hashes)):
        raise RuntimeError("aggregate found repeated byte-identical shard artifacts")

    poison_expected = _checkpoint_artifact_identity(args.checkpoint, poison_weights_checksum)
    frozen_clean = str(cfg.get("attack-clean-checkpoint-checksum", ""))
    if not frozen_clean or frozen_clean != str(subset.get("checkpoint_checksum", "")):
        raise RuntimeError("aggregate frozen clean checksum is missing or inconsistent with subset")
    targets = [tuple(target) for target in subset["targets"]]
    if len(targets) != int(subset["n"]) or len(targets) != len(set(targets)):
        raise RuntimeError("frozen subset target identities are duplicate or count-inconsistent")

    seen_indices: set[int] = set()
    seen_rows: set[tuple[str, str]] = set()
    rows: List[dict] = []
    artifact_keys = {
        "schema_version", "mode", "poison", "clean", "subset_content_hash",
        "shard_index", "num_shards", "results",
    }
    identity_keys = {
        "checkpoint_file_sha256", "resolved_sha256", "checkpoint_weights",
        "runtime_dependencies_sha256", "selected_weights_checksum",
    }
    row_keys = {
        "sample_token", "ann_token", "evaluated", "disappeared",
        "occlusion_disappeared", "placement_aligned_ok",
        "placement_nonaligned_iou0", "area_ratio",
    }
    for path in files:
        with open(path, encoding="utf-8") as stream:
            artifact = json.load(stream)
        if set(artifact) != artifact_keys or artifact["schema_version"] != _SHARD_SCHEMA_FULL:
            raise RuntimeError(f"invalid or incomplete T5 shard schema: {path}")
        if artifact["mode"] != _SHARD_MODE_FULL:
            raise RuntimeError("cond4-only/mixed shard artifact cannot enter full aggregate")
        if artifact["subset_content_hash"] != subset["content_hash"]:
            raise RuntimeError("shard frozen-subset identity mismatch")
        index = artifact["shard_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count:
            raise RuntimeError("shard index is outside declared aggregate range")
        if artifact["num_shards"] != count or index in seen_indices:
            raise RuntimeError("duplicate shard index or num_shards mismatch")
        seen_indices.add(index)
        if not isinstance(artifact["poison"], dict) or set(artifact["poison"]) != identity_keys:
            raise RuntimeError("shard poison identity schema mismatch")
        if artifact["poison"] != poison_expected:
            raise RuntimeError("shard poison preflight/selected-weight identity mismatch")
        clean = artifact["clean"]
        if not isinstance(clean, dict) or set(clean) != identity_keys:
            raise RuntimeError("full shard requires a complete clean identity")
        for key in (
            "checkpoint_file_sha256", "resolved_sha256", "runtime_dependencies_sha256",
            "selected_weights_checksum",
        ):
            if not _is_sha256(clean[key]):
                raise RuntimeError(f"clean shard identity {key} is not a SHA-256")
        if clean["resolved_sha256"] != poison_expected["resolved_sha256"]:
            raise RuntimeError("shard clean/poison resolved identity mismatch")
        if clean["checkpoint_weights"] != poison_expected["checkpoint_weights"]:
            raise RuntimeError("shard clean/poison raw/EMA policy mismatch")
        if clean["runtime_dependencies_sha256"] != poison_expected["runtime_dependencies_sha256"]:
            raise RuntimeError("shard clean/poison dependency identity mismatch")
        if clean["selected_weights_checksum"] != frozen_clean:
            raise RuntimeError("shard actual clean selected weights do not match frozen subset")

        expected_targets = targets[index::count]
        artifact_rows = artifact["results"]
        if not isinstance(artifact_rows, list) or len(artifact_rows) != len(expected_targets):
            raise RuntimeError("shard row count does not match deterministic frozen target slice")
        row_targets = []
        for row in artifact_rows:
            if not isinstance(row, dict) or set(row) != row_keys:
                raise RuntimeError("full shard row schema is incomplete or has unknown fields")
            identity = (str(row["sample_token"]), str(row["ann_token"]))
            row_targets.append(identity)
            if identity in seen_rows:
                raise RuntimeError("duplicate (sample_token, ann_token) across shard artifacts")
            seen_rows.add(identity)
            if row["evaluated"] is not True:
                raise RuntimeError("full shard row lacks mandatory five-condition evaluation")
            if type(row["occlusion_disappeared"]) is not bool:
                raise RuntimeError("full shard mandatory occlusion control must be boolean")
            disappeared = row["disappeared"]
            if (
                not isinstance(disappeared, dict) or set(disappeared) != set(FA.CONDITIONS)
                or any(type(value) is not bool for value in disappeared.values())
            ):
                raise RuntimeError("full shard disappeared-condition schema is invalid")
        if set(row_targets) != set(expected_targets):
            raise RuntimeError("shard rows are missing, extra, duplicated, or assigned to wrong index")
        rows.extend(artifact_rows)
    if seen_indices != set(range(count)) or seen_rows != set(targets):
        raise RuntimeError("aggregate shard set does not exactly cover frozen targets")
    return rows


def _load_subset(cfg, subset_path: str) -> dict:
    """Load the frozen subset; re-verify the content-hash + the LITERAL pinned identities (§0.C6)."""
    from fl_v3.eval.asr import verify_frozen_asr_subset
    with open(subset_path, encoding="utf-8") as f:
        subset = json.load(f)
    verify_frozen_asr_subset(subset)  # recompute the content hash (targets/thresholds/checkpoint)
    pin_hash = str(cfg.get("attack-frozen-subset-hash", ""))
    pin_ckpt = str(cfg.get("attack-clean-checkpoint-checksum", ""))
    if pin_hash and subset["content_hash"] != pin_hash:
        raise RuntimeError(f"frozen-subset content_hash {subset['content_hash']} != PINNED {pin_hash} (§0.C6)")
    if pin_ckpt and subset["checkpoint_checksum"] != pin_ckpt:
        raise RuntimeError(f"subset checkpoint_checksum {subset['checkpoint_checksum']} != PINNED {pin_ckpt} (§0.C6)")
    return subset


def _val_info(cfg):
    from fl_v3.training.tasks import get_task
    return get_task("nuscenes_detection")._load_info(
        cfg, str(cfg["nuscenes-val-split"]),
    )[0]


def _val_dataset(cfg, val_info, tokens):
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset
    from fl_v3.data.nuscenes import paths as P
    return NuScenesMultimodalDataset(
        val_info, P.get_dataroot(cfg), sample_tokens=sorted(set(tokens)),
        n_sweeps=int(cfg["det-lidar-sweeps"]),
        zip_manifest=str(cfg["nuscenes-zip-manifest"]),
        model_mode=str(cfg["model-mode"]),
    )


def _seed(cfg):
    from fl_v3.utils.runtime import enforce_determinism, seed_everything, truthy
    seed_everything(int(cfg.get("seed", 42)))
    enforce_determinism(
        strict=truthy(cfg.get("determinism-strict", True)),
        precision=str(cfg.get("precision", "fp16")),
    )


# ---------------------------------------------------------------------------
# task: shard (the per-target fan-out worker)
# ---------------------------------------------------------------------------
def task_shard(args, cfg):
    mode = _validate_task_requirements(args)
    _require_preflight(args.checkpoint, args.clean_checkpoint)
    from fl_v3.eval.asr import thresholds_from_subset
    from fl_v3.attacks import trigger as TR
    from fl_v3.attacks import fusion_ablation as FA
    device = _device(cfg); _seed(cfg)
    subset = _load_subset(cfg, args.subset)
    thr = thresholds_from_subset(subset)
    spec = TR.trigger_spec_from_run_config(cfg)

    poisoned = _load_model(cfg, args.checkpoint, device)
    poison_checksum = _trainable_checksum(poisoned)
    clean = None
    clean_checksum = None
    if mode == _SHARD_MODE_FULL:
        clean = _load_model(cfg, args.clean_checkpoint, device)
        clean_checksum = _trainable_checksum(clean)
        frozen_clean = str(cfg.get("attack-clean-checkpoint-checksum", ""))
        if not frozen_clean or clean_checksum != frozen_clean:
            raise RuntimeError(
                "actual selected clean raw/EMA weights do not match "
                "attack-clean-checkpoint-checksum"
            )
        if clean_checksum != str(subset.get("checkpoint_checksum", "")):
            raise RuntimeError("actual selected clean weights do not match frozen subset identity")

    targets: List[Tuple[str, str]] = [tuple(t) for t in subset["targets"]]
    shard = targets[args.shard::args.num_shards]                      # deterministic round-robin slice
    by_sample: Dict[str, List[str]] = {}
    for s, a in shard:
        by_sample.setdefault(s, []).append(a)

    val_info = _val_info(cfg)
    ds = _val_dataset(cfg, val_info, list(by_sample.keys()))
    print(f"[t5-shard {args.shard}/{args.num_shards}] {len(shard)} targets over {len(by_sample)} samples", flush=True)

    out: List[dict] = []
    for i in range(len(ds)):
        sample = ds[i]
        tok = sample["sample_token"]
        if tok not in by_sample:
            continue
        clean_anns = FA.clean_detected_anns(poisoned, sample, device, thr)
        for ann in by_sample[tok]:
            if mode == _SHARD_MODE_COND4:
                v = FA.evaluate_target_cond4(sample, ann, poisoned, device, thr, spec,
                                             clean_anns=clean_anns)
            else:
                if clean is None:
                    raise RuntimeError("full five-condition evaluation requires loaded clean model")
                v = FA.evaluate_target(sample, ann, poisoned, clean, device, thr, spec,
                                       clean_anns=clean_anns)
            if mode == _SHARD_MODE_COND4:
                if v is None:
                    out.append({"sample_token": tok, "ann_token": ann, "evaluated": False})
                else:
                    out.append({
                        "sample_token": tok, "ann_token": ann, "evaluated": True,
                        "cond1_clean_disappeared": bool(v.disappeared["cond1_clean"]),
                        "cond4_aligned_disappeared": bool(v.disappeared["cond4_aligned"]),
                        "placement_aligned_ok": v.placement_aligned_ok,
                        "area_ratio": v.area_ratio,
                    })
            else:
                if v is None:
                    raise RuntimeError(
                        f"full five-condition target {(tok, ann)!r} lacks mandatory evaluation/control"
                    )
                if type(v.occlusion_disappeared) is not bool:
                    raise RuntimeError("full five-condition target lacks boolean occlusion control")
                out.append({"sample_token": tok, "ann_token": ann, "evaluated": True,
                            "disappeared": v.disappeared, "occlusion_disappeared": v.occlusion_disappeared,
                            "placement_aligned_ok": v.placement_aligned_ok,
                            "placement_nonaligned_iou0": v.placement_nonaligned_iou0,
                            "area_ratio": v.area_ratio})
        if (i + 1) % 50 == 0:
            print(f"[t5-shard {args.shard}] {i+1}/{len(ds)} samples", flush=True)

    os.makedirs(os.path.join(args.output_dir, "ablation"), exist_ok=True)
    p = _shard_artifact_path(args.output_dir, args.shard, args.num_shards, mode)
    artifact = {
        "schema_version": _shard_schema(mode),
        "mode": mode,
        "poison": _checkpoint_artifact_identity(args.checkpoint, poison_checksum),
        "clean": (
            _checkpoint_artifact_identity(args.clean_checkpoint, clean_checksum)
            if mode == _SHARD_MODE_FULL else None
        ),
        "subset_content_hash": subset["content_hash"],
        "shard_index": int(args.shard),
        "num_shards": int(args.num_shards),
        "results": out,
    }
    json.dump(artifact, open(p, "w"), sort_keys=True)
    print(f"[t5-shard {args.shard}] wrote {p} ({len(out)} targets)", flush=True)


# ---------------------------------------------------------------------------
# task: aggregate (combine shards → table + verdict)
# ---------------------------------------------------------------------------
def task_aggregate(args, cfg):
    _validate_task_requirements(args)
    _require_preflight(args.checkpoint)
    from fl_v3.attacks import fusion_ablation as FA
    from fl_v3.eval.provenance import verify_attack_provenance
    _assert_pinned_constants(cfg)  # §0.C4: NO post-hoc override of any anti-gaming constant
    device = _device(cfg)
    subset = _load_subset(cfg, args.subset)
    N = int(subset["n"])
    # §0.C8: the verdict is bound to a provenance-verified trainval poisoned checkpoint
    model = _load_model(cfg, args.checkpoint, device)
    checksum = _trainable_checksum(model)
    prov = verify_attack_provenance(args.checkpoint, checksum)
    print(f"[t5-agg] attack provenance VERIFIED: {prov.get('regime')} | mode={prov.get('attack-mode')} "
          f"poison_rate={prov.get('attack-poison-rate')} m_r={prov.get('m_r')} roster={prov.get('roster')}", flush=True)

    rows = _load_bound_full_shards(args, cfg, subset, checksum)
    evaluated = rows
    coverage = 1.0 if N else 0.0

    # per-condition raw ASR over the FULL N (un-evaluated targets → not disappeared, conservative)
    counts = {c: 0 for c in FA.CONDITIONS}
    occ = 0; aligned_ok = 0; nonaligned_iou0 = 0; area_ok = 0; n_area = 0
    budget = float(cfg.get("attack-trigger-budget-frac", 0.30))
    for r in evaluated:
        for c in FA.CONDITIONS:
            if r["disappeared"].get(c):
                counts[c] += 1
        if r["occlusion_disappeared"] is True:
            occ += 1
        aligned_ok += int(bool(r.get("placement_aligned_ok")))
        nonaligned_iou0 += int(bool(r.get("placement_nonaligned_iou0")))
        if r.get("area_ratio") is not None:
            n_area += 1; area_ok += int(r["area_ratio"] <= budget + 1e-9)
    asr = {c: counts[c] / N for c in FA.CONDITIONS}
    occ_asr = occ / N

    # --- the OTHER anti-gaming gates (from the sibling eval tasks) — MUST be present + pass before
    # a FUSION-AWARE GATE can go green (the verdict is a CONJUNCTION, not just the ablation margins).
    guards = _load_json(os.path.join(args.output_dir, "cond5a_guards.json"))
    stealth = _load_json(os.path.join(args.output_dir, "stealth.json"))
    # RE-DERIVE the sub-gates from the RAW metrics + the PINNED floors (never trust a sub-task's stored
    # boolean, which could have been computed against an overridden floor).
    stealth_floor = float(cfg.get("attack-stealth-recall-floor", 0.75))
    cond5a_floor = float(cfg.get("attack-cond5a-recall-floor", 0.3))
    stealth_ok = (float(stealth["poisoned_clean_car_recall"]) >= stealth_floor) if stealth else None
    cond5a_valid = (bool(guards["lidar_invariant_all"]) and
                    float(guards["camera_only_clean_recall"]) >= cond5a_floor) if guards else None

    verdict = FA.fusion_aware_verdict(
        asr, delta_fusion=float(cfg.get("attack-delta-fusion", 0.2)),
        mult=float(cfg.get("attack-delta-fusion-mult", 2.0)),
        viability=float(cfg.get("attack-viability-asr", 0.3)),
        cond5a_guards_valid=cond5a_valid)

    fd_max = float(cfg.get("asr-false-disappear-max", 0.02))
    # placement objective test (§0.C): every evaluated target's cond-4 centre ≤20px + cond-2 IoU==0 +
    # area ≤ budget. Require ≥0.99 (tolerate <1% truncated-geometry edge cases; the count is surfaced).
    aligned_frac = (aligned_ok / len(evaluated)) if evaluated else 0.0
    nonaligned_frac = (nonaligned_iou0 / len(evaluated)) if evaluated else 0.0
    area_frac = (area_ok / n_area) if n_area else 0.0
    placement_ok = bool(aligned_frac >= 0.99 and nonaligned_frac >= 0.99 and area_frac >= 0.999)

    headline = {"disappear_asr_raw_cond4": asr["cond4_aligned"],
                "disappear_asr_floor_corrected_cond4": verdict["cond4_corrected"],
                "viable": verdict["viable_cond4_ge_0.3"]}
    occlusion = {"occlusion_control_asr": occ_asr, "false_disappear_max": fd_max,
                 "passes_not_occlusion": bool(occ_asr < fd_max)}
    placement = {"n_evaluated": len(evaluated), "coverage_of_N": coverage,
                 "aligned_center_le_20px_frac": aligned_frac, "nonaligned_iou0_frac": nonaligned_frac,
                 "area_le_budget_frac": area_frac, "budget_frac": budget, "placement_ok": placement_ok}

    # the conjunctive GATE (every sub-gate must be PRESENT and PASS; a missing sibling → not green)
    gate = {
        "viable_cond4_ge_0.3": verdict["viable_cond4_ge_0.3"],
        "margin_ok_ge_delta_fusion": verdict["margin_ok_ge_delta_fusion"],
        "mult_ok_cond4_ge_2x": verdict["mult_ok_cond4_ge_2x_maxother"],
        "not_occlusion": occlusion["passes_not_occlusion"],
        "stealth_ok": stealth_ok,
        "placement_objective_ok": placement_ok,
        "cond5a_guards_valid": cond5a_valid,
        "provenance_verified": True,
    }
    missing = [k for k, v in gate.items() if v is None]
    gate_pass = bool(all(v is True for v in gate.values()) and not missing)
    overall = ("FUSION-AWARE (GATE GREEN)" if (gate_pass and verdict["fusion_aware"]) else
               ("D3-ESCAPE-HATCH (cond4≈cond2 — point-decoration)" if verdict["degenerate_cond4_approx_cond2"] else
                ("INCOMPLETE — missing sub-gates: " + ", ".join(missing) if missing else "GATE NOT GREEN")))

    result = {"N": N, "n_shards": int(args.num_shards), "raw_asr": asr, "fusion_aware_verdict": verdict,
              "headline": headline, "occlusion_control": occlusion, "placement_objective_test": placement,
              "stealth": stealth, "cond5a_guards": guards, "gate": gate, "gate_pass": gate_pass,
              "missing_subgates": missing, "overall_verdict": overall,
              "pinned_constants": _pinned_constants(cfg),
              "checkpoint_checksum": checksum, "verified_attack_provenance": prov}
    json.dump(result, open(os.path.join(args.output_dir, "fusion_ablation.json"), "w"), sort_keys=True, indent=2)
    print("=" * 72, flush=True)
    print(f"[t5-agg] 5-CONDITION ABLATION (N={N}, coverage={coverage:.4f}, evaluated={len(evaluated)})", flush=True)
    for c in FA.CONDITIONS:
        print(f"  {c:24s} raw_asr={asr[c]:.4f}  floor-corr={verdict['corrected_asr'][c]:+.4f}", flush=True)
    print(f"  floor (cond1) = {verdict['floor']:.4f}", flush=True)
    print(f"  HEADLINE disappear-ASR(cond4, floor-corr) = {verdict['cond4_corrected']:.4f} (viable: {headline['viable']})", flush=True)
    print(f"  occlusion-control ASR (clean model) = {occ_asr:.4f} (<{fd_max}: {occlusion['passes_not_occlusion']})", flush=True)
    print(f"  stealth_ok={stealth_ok}  cond5a_valid={cond5a_valid}  placement_ok={placement_ok}", flush=True)
    print(f"  GATE (conjunction) = {gate}", flush=True)
    print(f"  gate_pass={gate_pass}  OVERALL = {overall}", flush=True)
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# task: stealth (poisoned clean DetectionEval — mAP/NDS + official car recall)
# ---------------------------------------------------------------------------
def task_stealth(args, cfg):
    _validate_task_requirements(args)
    _require_preflight(args.checkpoint)
    _assert_pinned_constants(cfg)
    from fl_v3.data.nuscenes.class_map import DETECTION_NAMES
    from fl_v3.eval.detection_eval import decode_eval_set, run_detection_eval, VERSION_EVAL_SET
    from fl_v3.eval.provenance import verify_attack_provenance
    from fl_v3.training.tasks import get_task
    device = _device(cfg); _seed(cfg)
    version = str(cfg["nuscenes-version"]); eval_set = VERSION_EVAL_SET[version]
    model = _load_model(cfg, args.checkpoint, device)
    checksum = _trainable_checksum(model)
    prov = verify_attack_provenance(args.checkpoint, checksum)
    task = get_task("nuscenes_detection")
    val_info = _val_info(cfg)
    all_tokens = sorted(i["sample_token"] for i in val_info)
    loader = task.eval_loader(cfg)
    print(f"[t5-stealth] decoding {len(all_tokens)} {eval_set} samples (poisoned, clean inputs, bs=1)…", flush=True)
    decodes = decode_eval_set(model, loader, device, cfg)
    nusc = _nusc(version, cfg)
    det = run_detection_eval(nusc, decodes, eval_set, version, os.path.join(args.output_dir, "stealth_det_eval"),
                             DETECTION_NAMES, all_eval_tokens=all_tokens, run_config=cfg, verbose=False)
    floor = float(cfg.get("attack-stealth-recall-floor", 0.75))
    out = {"poisoned_clean_car_recall": det["car_recall"], "stealth_recall_floor": floor,
           "stealth_ok": bool(det["car_recall"] >= floor), "poisoned_mAP": det["mAP"], "poisoned_NDS": det["NDS"],
           "car_ap_2m": det["car_ap_2m"], "checkpoint_checksum": checksum, "verified_attack_provenance": prov}
    json.dump(out, open(os.path.join(args.output_dir, "stealth.json"), "w"), sort_keys=True, indent=2)
    print(f"[t5-stealth] poisoned clean car_recall={det['car_recall']:.4f} (≥{floor}: {out['stealth_ok']}) "
          f"mAP={det['mAP']:.4f} NDS={det['NDS']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# task: guards (cond-5a LiDAR-invariance + camera-only clean-recall precondition)
# ---------------------------------------------------------------------------
def task_guards(args, cfg):
    _validate_task_requirements(args)
    _require_preflight(args.checkpoint)
    _assert_pinned_constants(cfg)
    from fl_v3.eval.asr import thresholds_from_subset
    from fl_v3.attacks import fusion_ablation as FA
    device = _device(cfg); _seed(cfg)
    subset = _load_subset(cfg, args.subset)
    thr = thresholds_from_subset(subset)
    model = _load_model(cfg, args.checkpoint, device)
    targets = [tuple(t) for t in subset["targets"]]
    sample_tokens = sorted({s for s, _ in targets})[: args.guard_samples]
    by_sample: Dict[str, List[str]] = {}
    for s, a in targets:
        if s in sample_tokens:
            by_sample.setdefault(s, []).append(a)
    val_info = _val_info(cfg)
    ds = _val_dataset(cfg, val_info, sample_tokens)
    inv_results = []; recall_pairs = []
    for i in range(len(ds)):
        sample = ds[i]
        if sample["sample_token"] not in by_sample:
            continue
        inv_results.append(FA.lidar_invariance_check(model, sample, device))
        recall_pairs.append((sample, by_sample[sample["sample_token"]]))
    rec = FA.camera_only_clean_recall(model, recall_pairs, device, thr)
    all_invariant = all(r["lidar_invariant"] for r in inv_results)
    max_diff = max((r["max_abs_head_diff"] for r in inv_results), default=0.0)
    floor = float(cfg.get("attack-cond5a-recall-floor", 0.3))  # camera-only readout is OOD vs the 0.85 fused model
    out = {"lidar_invariant_all": bool(all_invariant), "max_abs_head_diff": max_diff,
           "n_invariance_checks": len(inv_results), "camera_only_clean_recall": rec["camera_only_clean_recall"],
           "camera_only_recall_floor": floor, "clean_recall_precondition_ok": bool(rec["camera_only_clean_recall"] >= floor),
           "detail": rec, "cond5a_valid": bool(all_invariant and rec["camera_only_clean_recall"] >= floor)}
    json.dump(out, open(os.path.join(args.output_dir, "cond5a_guards.json"), "w"), sort_keys=True, indent=2)
    print(f"[t5-guards] LiDAR-invariant={all_invariant} (max|Δ|={max_diff}) "
          f"camera_only_clean_recall={rec['camera_only_clean_recall']:.4f} (≥{floor}) → cond5a_valid={out['cond5a_valid']}", flush=True)


# ---------------------------------------------------------------------------
# task: viz (V5 + V3(trigger) for a few subset samples)
# ---------------------------------------------------------------------------
def task_viz(args, cfg):
    _validate_task_requirements(args)
    _require_preflight(args.checkpoint, args.clean_checkpoint)
    from fl_v3.eval.asr import thresholds_from_subset
    from fl_v3.attacks import trigger as TR
    from fl_v3.attacks.poison import _lidar_xyz
    from fl_v3.viz.writer import VizWriter
    from fl_v3.viz import attack as V5
    from fl_v3.viz.fusion import render_v3_trigger
    device = _device(cfg); _seed(cfg)
    subset = _load_subset(cfg, args.subset)
    thr = thresholds_from_subset(subset)
    spec = TR.trigger_spec_from_run_config(cfg)
    model = _load_model(cfg, args.checkpoint, device)
    clean = _load_model(cfg, args.clean_checkpoint, device) if args.clean_checkpoint else None
    cfg_bev = model.cfg.bev
    targets = [tuple(t) for t in subset["targets"]]
    first_by_sample = {}
    for s, a in targets:
        first_by_sample.setdefault(s, a)
    sample_tokens = sorted(first_by_sample)[: args.viz_samples]
    val_info = _val_info(cfg)
    ds = _val_dataset(cfg, val_info, sample_tokens)
    writer = VizWriter(args.output_dir)
    for i in range(len(ds)):
        sample = ds[i]
        tok = sample["sample_token"]
        if tok not in first_by_sample:
            continue
        from fl_v3.attacks.fusion_ablation import target_index
        m = target_index(sample, first_by_sample[tok])
        if m is None:
            continue
        box7 = sample["gt_boxes"].cpu().numpy()[m]
        pl = TR.compute_aligned_placement(box7, _lidar_xyz(sample), sample["lidar2img"].cpu().numpy(), spec)
        if pl is None:
            continue
        V5.render_v5_image(writer, sample, pl, spec, target_box=box7, name=tok)
        try:
            V5.render_v5_fusion_diff(writer, model, sample, pl, spec, cfg_bev, target_box=box7, clean_model=clean, name=tok)
            render_v3_trigger(writer, model, sample, pl, spec, cfg_bev, target_box=box7, name=tok)
        except Exception as e:
            print(f"[t5-viz] fusion-diff skipped for {tok}: {type(e).__name__}: {e}", flush=True)
    writer.write_manifest()
    print(f"[t5-viz] rendered V5/V3 for up to {len(sample_tokens)} subset samples", flush=True)


# ---------------------------------------------------------------------------
# task: null-verify (the §0.C5 byte-identical-null check)
# ---------------------------------------------------------------------------
def task_null_verify(args, cfg):
    raise RuntimeError(
        "T5 null-verify is frozen to the legacy bare-state checksum contract and cannot reinterpret "
        "a complete S06 checkpoint. Freeze a new null-checkpoint identity protocol before use."
    )


def _nusc(version: str, cfg: dict):
    from fl_v3.data.nuscenes import paths as P
    return P.create_nuscenes(version, P.get_dataroot(cfg), verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["shard", "aggregate", "stealth", "guards", "viz", "null-verify"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", help="the poisoned complete S06-compatible checkpoint")
    ap.add_argument("--clean-checkpoint", default=None, help="the clean complete S06-compatible checkpoint")
    ap.add_argument("--subset", help="the frozen ASR subset json (2ad8f8da…)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--cond4-only", action="store_true",
                    help="lean CONTROL measurement: cond-1 + cond-4 disappear-ASR only (no cond-2/3/5a/occlusion)")
    ap.add_argument("--guard-samples", type=int, default=40)
    ap.add_argument("--viz-samples", type=int, default=6)
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()
    cfg = _load_config(args.config, args.overrides)
    _validate_task_requirements(args)
    if args.task == "null-verify":
        task_null_verify(args, cfg)
        return
    _preflight_t5_checkpoints(cfg, args.checkpoint, args.clean_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    {"shard": task_shard, "aggregate": task_aggregate, "stealth": task_stealth,
     "guards": task_guards, "viz": task_viz, "null-verify": task_null_verify}[args.task](args, cfg)


if __name__ == "__main__":
    main()
