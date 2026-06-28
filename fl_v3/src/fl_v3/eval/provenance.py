"""D10 provenance binding for the benchmark-readiness verdict (fl_v3 T4 / T4_SPEC §0.2).

The readiness verdict must be bound — not merely by checkpoint checksum, but by VERIFIED training
**provenance** — to the D10 regime: a full-participation (`fraction-train == 1.0`) **log-group trainval
clean FedAvg** checkpoint. A verdict computed on a sampled (`fraction<1`) / IID / defended / wrong-split
checkpoint is **INVALID** (the §0.2 partition/participation-mismatch trap — the Codex T4 finding). The
reference launcher (`run_t4_reference_a40.sh`) writes `provenance.json` beside `final_model.pt` via
:func:`build_provenance`; the readiness eval hard-verifies it via :func:`verify_d10_provenance` BEFORE it
will emit a (valid) trainval go/no-go — so an overridden `CONFIG`/`CKPT` can never sneak a sampled/IID
checkpoint past the gate just because its metric floors happen to pass.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

# Exact-match keys a checkpoint's provenance MUST satisfy to host a valid trainval readiness verdict.
D10_REQUIRED: Dict[str, str] = {
    "task-type": "nuscenes_detection",
    "nuscenes-version": "v1.0-trainval",
    "nuscenes-train-split": "train",
    "nuscenes-val-split": "val",
    "nuscenes-partition-mode": "log_group",
    "defense-type": "none",
}
# The D10-relevant run-config keys the reference launcher records into provenance.json.
# ``precision`` (D16: bf16=science / fp32=dev) is the canonical regime field — set explicitly in
# build_provenance (NOT in this tuple, so the T5 ATTACK_PROVENANCE_KEYS schema stays unchanged). The
# legacy ``numeric-mode`` {fp32,tf32} key is kept here for back-compat/traceability but is NOT in
# D10_REQUIRED (so legacy FP32/TF32 provenance is not retroactively invalidated); a bf16 verdict must be
# bound to a bf16 checkpoint ("no mixing regimes"), enforced by the t4 regime-match guard on ``precision``.
PROVENANCE_KEYS = (
    "task-type", "nuscenes-version", "nuscenes-train-split", "nuscenes-val-split",
    "nuscenes-partition-mode", "fraction-train", "defense-type", "nuscenes-num-clients",
    "num-server-rounds", "seed", "det-camera-backbone", "numeric-mode",
)


# MCR P3 (D17): the FL recipe fields recorded for an HONEST regime label (NOT part of D10 validation —
# D10_REQUIRED is unchanged, so server-optimizer=fedadam stays D10-compliant as long as defense-type=none).
# The server optimizer is an axis ORTHOGONAL to the defense; recording it prevents a FedAdam reference from
# being silently read as plain-FedAvg (and vice-versa).
FL_RECIPE_KEYS = (
    "num-local-epochs", "learning-rate", "weight-decay", "det-optimizer", "grad-clip-norm",
    "det-backbone-lr-mult", "client-lr-schedule", "client-lr-warmup-rounds", "client-lr-final-frac",
    "server-optimizer", "server-lr", "server-beta1", "server-beta2", "server-tau",
    "server-lr-warmup-rounds", "server-ema-decay",
)


def build_provenance(run_config: dict, checksum: str) -> dict:
    """Build the provenance record written beside ``final_model.pt`` (single source of truth)."""
    prov = {k: run_config.get(k) for k in PROVENANCE_KEYS}
    prov["fl_recipe"] = {k: run_config.get(k) for k in FL_RECIPE_KEYS}
    prov["fraction-train"] = float(run_config.get("fraction-train", 0.0))
    # D16: ``precision`` ∈ {bf16, fp32} is the canonical regime field; the legacy ``numeric-mode``
    # {fp32,tf32} key is retained in PROVENANCE_KEYS for back-compat but is now recorded honestly as
    # whatever the run set (None for a D16 ``precision`` run — NOT silently stamped fp32, which would
    # mislabel a bf16 checkpoint). A reported number's regime is read from ``precision``.
    prov["precision"] = str(run_config.get("precision", "bf16"))
    prov["numeric-mode"] = run_config.get("numeric-mode")  # legacy/back-compat; None under the D16 knob
    prov["FL_TRAINABLE_CHECKSUM"] = str(checksum)
    prov["regime"] = "D10-full-participation-log-group-trainval-clean"
    return prov


def check_d10(prov: dict, checksum: Optional[str] = None) -> List[str]:
    """Return a list of D10 violations (empty list == compliant). Binds to ``checksum`` if given."""
    bad = [f"{k}={prov.get(k)!r} (need {v!r})" for k, v in D10_REQUIRED.items() if str(prov.get(k)) != v]
    if float(prov.get("fraction-train", -1)) != 1.0:
        bad.append(f"fraction-train={prov.get('fraction-train')!r} (need 1.0 — FULL participation)")
    if checksum is not None and str(prov.get("FL_TRAINABLE_CHECKSUM", "")) != str(checksum):
        bad.append(f"FL_TRAINABLE_CHECKSUM={prov.get('FL_TRAINABLE_CHECKSUM')!r} != recomputed {checksum!r}")
    return bad


def verify_d10_provenance(checkpoint_path: str, checksum: str) -> dict:
    """Load ``provenance.json`` beside the checkpoint + hard-verify D10. RAISES on missing / mismatch /
    checksum-mismatch — so a sampled / IID / defended / wrong-split checkpoint can NEVER emit a valid
    trainval readiness verdict. Returns the verified provenance (with ``_verified=True``)."""
    prov_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "provenance.json")
    if not os.path.isfile(prov_path):
        raise RuntimeError(
            f"D10 provenance MISSING ({prov_path}). A trainval readiness verdict MUST be bound to a "
            "full-participation log-group trainval checkpoint (T4_SPEC §0.2) — refusing a checkpoint of "
            "unverified provenance. Train via run_t4_reference_a40.sh (it writes provenance.json).")
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    bad = check_d10(prov, checksum)
    if bad:
        raise RuntimeError("D10 provenance INVALID — readiness verdict refused (T4_SPEC §0.2): " + "; ".join(bad))
    prov["_verified"] = True
    return prov


# ---------------------------------------------------------------------------
# T5 attack provenance (§0.C8): a POISONED checkpoint must verify as the D10 regime
# (full-participation log-group trainval clean-aggregator) PLUS the attack metadata
# (poison_rate>0 + the recorded roster) — so the disappear-ASR / fusion-aware verdict is bound to a
# provenance-verified trainval poisoned checkpoint, never a mini-smoke or a mislabelled run.
# ---------------------------------------------------------------------------
# The D10-base keys shared with the clean reference (defence stays "none" — the aggregator is FedAvg;
# the attack is data-poisoning on the malicious roster, not a defended cell).
ATTACK_PROVENANCE_KEYS = PROVENANCE_KEYS + (
    "attack-enabled", "attack-mode", "attack-poison-rate", "attack-rho", "attack-delta-reloc",
)


def build_attack_provenance(run_config: dict, checksum: str, roster: list, m_r: int) -> dict:
    """Provenance for a poisoned T5 checkpoint = the D10 base + the attack metadata + the roster."""
    prov = {k: run_config.get(k) for k in ATTACK_PROVENANCE_KEYS}
    prov["fraction-train"] = float(run_config.get("fraction-train", 0.0))
    prov["attack-poison-rate"] = float(run_config.get("attack-poison-rate", 0.0))
    prov["FL_TRAINABLE_CHECKSUM"] = str(checksum)
    prov["regime"] = "T5-attack-D10-full-participation-log-group-trainval-fedavg"
    prov["roster"] = list(roster)
    prov["m_r"] = int(m_r)
    return prov


def check_attack(prov: dict, checksum: Optional[str] = None) -> List[str]:
    """Return attack-provenance violations (empty == compliant): the D10 base (partition/split/version/
    defense=none/fraction=1.0) + ``poison_rate > 0`` + a non-empty recorded roster (§0.C8)."""
    bad = check_d10(prov, checksum)  # D10 base (defence=none, fraction=1.0, log_group trainval)
    if float(prov.get("attack-poison-rate", 0.0)) <= 0.0:
        bad.append(f"attack-poison-rate={prov.get('attack-poison-rate')!r} (need > 0 for a poisoned cell)")
    if not prov.get("roster"):
        bad.append("roster missing/empty (the malicious roster must be recorded)")
    if int(prov.get("m_r", 0)) < 1:
        bad.append(f"m_r={prov.get('m_r')!r} (need ≥ 1 malicious client)")
    return bad


def verify_attack_provenance(checkpoint_path: str, checksum: str) -> dict:
    """Load + hard-verify a poisoned checkpoint's ``provenance.json`` (the D10 base + attack metadata).
    RAISES on missing / non-D10 / poison_rate≤0 / no-roster / checksum-mismatch — so the disappear-ASR
    + fusion-aware verdict can NEVER be emitted on a mini / mislabelled / clean checkpoint (§0.C8)."""
    prov_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "provenance.json")
    if not os.path.isfile(prov_path):
        raise RuntimeError(
            f"attack provenance MISSING ({prov_path}). The disappear-ASR + fusion-aware verdict MUST be "
            "bound to a provenance-verified trainval poisoned checkpoint (T5_SPEC §0.C8). Train via "
            "run_t5_attack_a40.sh (it writes provenance.json + roster).")
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    bad = check_attack(prov, checksum)
    if bad:
        raise RuntimeError("attack provenance INVALID — verdict refused (T5_SPEC §0.C8): " + "; ".join(bad))
    prov["_verified"] = True
    return prov
