"""T4 — the frozen 6-tuple reporting schema (clean cols filled, poisoned/ASR/defense reserved)."""
from __future__ import annotations

from fl_v3.eval.report import (
    REPORT_SCHEMA_VERSION,
    SCHEMA_COLUMNS,
    build_cell_report,
    empty_cell,
    t4_clean_cell,
)


def test_empty_cell_has_all_columns():
    cell = empty_cell()
    assert set(cell.keys()) == {k for k, _ in SCHEMA_COLUMNS}
    assert all(v is None for v in cell.values())


def test_t4_clean_cell_fills_clean_and_reserves_rest():
    r = t4_clean_cell(scale="trainval-scientific", checkpoint_checksum="abc123",
                      clean_map=0.31, clean_nds=0.42, asr_denominator_n=187)
    assert r["schema_version"] == REPORT_SCHEMA_VERSION
    c = r["cell"]
    assert c["clean_mAP"] == 0.31 and c["clean_NDS"] == 0.42 and c["asr_denominator_N"] == 187
    # poisoned / ASR / utility-collapse / defense are reserved (None) for T5+/T6/T7.
    for k in ("poisoned_mAP", "poisoned_NDS", "disappear_asr", "phantom_asr",
              "utility_collapse", "defense_decision_stats"):
        assert c[k] is None
    assert r["producers"]["clean_mAP"] == "T4"
    assert r["producers"]["disappear_asr"] == "T5+"


def test_build_cell_report_coerces_floats():
    import numpy as np
    r = build_cell_report(cell_id="x", scale="mini", checkpoint_checksum="z",
                          clean_map=np.float32(0.5), clean_nds=np.float64(0.6),
                          asr_denominator_n=np.int64(10))
    assert isinstance(r["cell"]["clean_mAP"], float) and r["cell"]["clean_mAP"] == 0.5
    assert isinstance(r["cell"]["asr_denominator_N"], int) and r["cell"]["asr_denominator_N"] == 10
