"""The frozen 6-tuple reporting schema every attack×defense cell fills (fl_v3 T4).

One row per (attack × defense) cell, reported in identical columns so T5/T6/T7 never re-freeze
the contract. The 6-tuple (plan §Defense "utility/ASR 2×2"):

  1. clean   mAP / NDS        — the official utility on clean inputs  (T4 fills)
  2. poisoned mAP / NDS       — the official utility on the poisoned/defended model  (T5+/T7)
  3. disappear-ASR            — D4 primary attack-success on triggered inputs  (T5+)
  4. phantom-ASR              — D4 secondary attack-success  (T5+)
  5. ASR denominator N        — the frozen eligible-clean-detected count  (T4 fills)
  6. utility-collapse status  — clean & poisoned NDS drop ≤ δ vs the baseline (D7 δ DEFERRED → T6/T7)

Plus a **reserved defense-decision-stats column** (the per-module gradient/decision logging T6/T7
fill) so T7 need not re-freeze the schema. T4 fills the CLEAN columns + N, freezes the rest as
``None`` with the producing-task recorded. No attack is invoked at T4.
"""
from __future__ import annotations

from typing import Dict, Optional

REPORT_SCHEMA_VERSION = "t4.report.v1"

# The frozen column set (order is documentation; dicts are keyed). Each entry: (key, producer).
SCHEMA_COLUMNS = (
    ("clean_mAP", "T4"),
    ("clean_NDS", "T4"),
    ("poisoned_mAP", "T5+"),
    ("poisoned_NDS", "T5+"),
    ("disappear_asr", "T5+"),
    ("phantom_asr", "T5+"),
    ("asr_denominator_N", "T4"),
    ("utility_collapse", "T6/T7 (needs D7 δ)"),
    ("defense_decision_stats", "T6/T7"),
)


def empty_cell() -> Dict[str, object]:
    """A fully-``None`` cell with every frozen column present (the reserved schema)."""
    return {k: None for k, _ in SCHEMA_COLUMNS}


def build_cell_report(
    *,
    cell_id: str,
    scale: str,
    checkpoint_checksum: str,
    clean_map: Optional[float] = None,
    clean_nds: Optional[float] = None,
    asr_denominator_n: Optional[int] = None,
    poisoned_map: Optional[float] = None,
    poisoned_nds: Optional[float] = None,
    disappear_asr: Optional[float] = None,
    phantom_asr: Optional[float] = None,
    utility_collapse: Optional[str] = None,
    defense_decision_stats: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Dict[str, object]:
    """Build one schema-frozen cell report. Unset columns stay ``None`` (their producer is fixed)."""
    cell = empty_cell()
    cell.update({
        "clean_mAP": _f(clean_map),
        "clean_NDS": _f(clean_nds),
        "poisoned_mAP": _f(poisoned_map),
        "poisoned_NDS": _f(poisoned_nds),
        "disappear_asr": _f(disappear_asr),
        "phantom_asr": _f(phantom_asr),
        "asr_denominator_N": (int(asr_denominator_n) if asr_denominator_n is not None else None),
        "utility_collapse": utility_collapse,
        "defense_decision_stats": defense_decision_stats,
    })
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "cell_id": str(cell_id),
        "scale": str(scale),
        "checkpoint_checksum": str(checkpoint_checksum),
        "columns": [k for k, _ in SCHEMA_COLUMNS],
        "producers": {k: p for k, p in SCHEMA_COLUMNS},
        "cell": cell,
        "extra": extra or {},
        "notes": {
            "utility_collapse": "requires D7 δ (DEFERRED) + poisoned NDS — set at T6/T7",
            "asr": "disappear/phantom ASR defined only on TRIGGERED inputs — injected at T5",
        },
    }


def t4_clean_cell(
    *, scale: str, checkpoint_checksum: str, clean_map: float, clean_nds: float,
    asr_denominator_n: int, cell_id: str = "clean__fedavg__no_attack",
    extra: Optional[dict] = None,
) -> Dict[str, object]:
    """The T4 clean baseline cell: clean mAP/NDS + N filled; poisoned/ASR/utility-collapse reserved."""
    return build_cell_report(
        cell_id=cell_id, scale=scale, checkpoint_checksum=checkpoint_checksum,
        clean_map=clean_map, clean_nds=clean_nds, asr_denominator_n=asr_denominator_n,
        extra=extra,
    )


def _f(x):
    return None if x is None else float(x)
