# MCR Phase 0 — precision config-collapse (D16 ratification)

> Session: MODEL CAPABILITY + RECIPE (MCR), charter D17. This file records **Phase 0 only** — the
> mechanical collapse of the two-axis numeric config into one `precision` knob, on the **clean science
> path only** (T5 untouched). Phases 1–4 (capability search, throughput, FL recipe, re-baseline) follow.

## What changed (one sentence)

The old **`numeric-mode ∈ {fp32, tf32}` × `determinism-level ∈ {strict, relaxed}`** two-axis knob is
collapsed into **one `precision ∈ {bf16, fp32}` knob** (D16): `bf16` = the science path (bf16-AMP,
autotuner + atomic scatter + autocast, NOT byte-identical, ≥3-seed bar); `fp32` = the offline
dev-regression / determinism tool (true IEEE FP32, same-seed byte-identical on one GPU tier). TF32 is
**retired** (it was the D14 Alvis regime, made redundant by bf16-AMP).

## The mapping (old → new)

| old `numeric-mode` × `determinism-level` | → new `precision` | backend state |
|---|---|---|
| `*` + `relaxed` (D14/D15 science/speed) | **`bf16`** (science **default**) | TF32 off → matmul `highest`; cudnn.deterministic=False + benchmark=True; `use_deterministic_algorithms(False)`; train-loop bf16 autocast ON (keys off `not cudnn.deterministic`) |
| `fp32` + `strict` (byte-identical ref) | **`fp32`** (dev/determinism tool) | TF32 off, matmul `highest`; cudnn.deterministic=True + benchmark=False; `use_deterministic_algorithms(True, warn_only=not strict)`; autocast OFF |
| `tf32` + `strict` (D14 TF32 science) | **dropped** — tf32 retired (D16); re-measure in bf16 | — |
| `fp32` + `relaxed` (transitional) | folds up into `bf16` | — |

## Key design decisions (and why)

1. **`enforce_determinism` function default = `fp32`/strict; the science default `bf16` lives at the
   CONFIG layer.** New signature: `enforce_determinism(strict=True, precision="fp32")`. `strict` stays the
   first positional so all 23 `enforce_determinism(strict=True)` callers (gate scripts + determinism
   tests) are byte-identically unchanged. The bf16 default is expressed in `pyproject.toml`,
   `configs/t4_reference.json`, the launchers, and every science entrypoint's `run_config.get("precision",
   "bf16")` — exactly where the old code defaulted fp32 and the launcher overrode to tf32. Rationale: a
   conservative deterministic default means a config that *forgets* `precision` gets the safe regression
   regime, not a silent science run; and the byte-identity gates/tests need zero edits.
2. **`local_runner` defaults `precision="fp32"`; `client_app`/`server_app` default `bf16`.** `local_runner`
   is the in-process **determinism-test / fl-gate harness** (its consumers are the CPU determinism tests +
   `fl_gate_a40`), so fp32 is the right default and the CPU byte-identity tests stay green with no per-test
   edit. The science FL path is Flower (`client_app`/`server_app`) → bf16.
3. **The byte-identity FL gate (`t3_fl_gate.json`) pins `precision: fp32` explicitly.** `run_fedavg_a40.sh`
   cross-checks the Ray `flwr run` (client/server_app, default bf16) checksum against the `local_runner`
   (default fp32) checksum, so BOTH sides must be fp32; pinning the gate config forces it.
   `t4_a100_detgate.json` likewise pinned `fp32` (a determinism gate).
4. **`torch.compile` is decoupled from the precision knob** — it now rides on `precision=="bf16"` **AND**
   the explicit `compile-backbone=true` opt-in (default off), honoring the D16 envelope (torch.compile
   stays opt-in with an eager fallback; not auto-on with the science regime). The old gate was
   `determinism-level=="relaxed"`.
5. **bf16 autocast wiring is unchanged** — `training/loop.py` already keys `use_amp` off
   `not cudnn.deterministic`, which `precision=bf16` sets False. No loop signature change; the fp32 path
   stays byte-identical. (Made explicit in the comment; flagged for a possible Phase-1 cleanup to thread
   `precision` directly rather than sniff the cuDNN flag.)
6. **Provenance records `precision` as the canonical field** (`build_provenance`), set explicitly so it does
   **not** ripple into `ATTACK_PROVENANCE_KEYS` (the T5 schema is byte-unchanged). The legacy `numeric-mode`
   key is kept for back-compat but recorded honestly (`None` under the D16 knob — no longer silently
   stamped `fp32`, which would mislabel a bf16 checkpoint). The t4 readiness regime-match RAISE guard now
   binds `precision` (with a legacy-`numeric-mode` → WARN fallback so pre-D16 checkpoints don't misfire).
7. **The TF32 det-gate is deleted** (`scripts/tf32_det_gate_a40.py` + `run_tf32_det_gate_a40.sh`) — it
   existed solely to prove tf32 determinism, which D16 retires; it called the removed `numeric_mode` param.
   Referenced only in historical docs (decisions.md / speedup_kickoff.md), left as historical record.

## T5 boundary — verified ZERO shim needed

Exhaustive enumeration of every `run_config.get(...)` in `src/fl_v3/attacks/*`: the only keys are
`attack-*`, `asr-*`, and `seed` — **none** of `numeric-mode`/`determinism-level`/`precision`/dtype/AMP. The
three dormant lazy importers (`training/tasks.py:499` `maybe_wrap_for_client`, `viz/attack.py`,
`viz/fusion.py`) read no regime key. The only T5-adjacent file touched is `eval/provenance.py`, and the
change is confined to `build_provenance` (clean path) — `PROVENANCE_KEYS`/`ATTACK_PROVENANCE_KEYS` and the
attack-provenance functions are structurally unchanged. **`test_attack_provenance.py` passes unchanged.**
No file under `src/fl_v3/attacks/`, `scripts/t5_*`, `configs/t5_*`, or `tests/test_attack_*` was modified.

## Files changed

**Core (`src/`):** `utils/runtime.py` (the sink: `enforce_determinism` signature+body, `_VALID_PRECISIONS`,
`precision_state()` adds canonical `precision`, module docstring), `client_app.py` (train+eval calls +
compile gate), `server_app.py`, `engine/local_runner.py` (×2 call sites, default fp32), `eval/provenance.py`
(`build_provenance` + doc comment).
**Scripts:** `centralized_train.py`, `t4_readiness_eval.py` (incl. the regime-match RAISE guard),
`profile_stages_a40.py` (`--modes`/`--level` → `--precisions`), `verify_levers.py` (`DET_LEVEL` → `PRECISION`),
`bench_dataloader_a100.py`; launchers `run_centralized_a40.sh`, `run_clean_fl_tf32_a40.sh`,
`run_readiness_tf32_a40.sh`, `run_b_eval_neutral_a40.sh`, `run_profile_a40.sh`. **Deleted:**
`tf32_det_gate_a40.py`, `run_tf32_det_gate_a40.sh`.
**Configs:** `t4_reference.json` (`numeric-mode:fp32` → `precision:bf16`), `t3_fl_gate.json` (+`precision:fp32`),
`t4_a100_detgate.json` (+`precision:fp32`), `pyproject.toml` (TOML defaults collapsed → `precision="bf16"`).
**Tests:** `tests/test_profiling_neutral.py` (`test_numeric_mode_sets_flags`→`test_precision_sets_flags`;
`test_numeric_mode_bad_raises`→`test_precision_bad_raises` — `bf16` now valid, `tf32` now the invalid value).
**Docs:** `docs/determinism.md` (top framing + `enforce_determinism` bullet reconciled to D16).

## Acceptance (D17 Phase-0 checklist)

- [x] ONE `precision` knob (bf16=science / fp32=dev), bf16 default at the config layer.
- [x] Manifest logs `precision` (`precision_state()["precision"]` + provenance `precision`).
- [x] Clean path migrated (entrypoints, eval, provenance, gate configs, launchers, TOML/JSON).
- [x] Dormant T5 imports resolve unchanged; no T5 file touched; `test_attack_provenance` green.
- [x] Strict-knob byte-identity regression tool preserved: `enforce_determinism(strict=True)` ⇒ fp32/strict;
      CPU determinism tests + `det_gate_a40`/`fl_gate_a40` (pinned fp32) unchanged.
- [x] Targeted regime/determinism/provenance tests: **35 passed** (run 1).
- [x] Full `pytest fl_v3/tests` green — **247 passed in 254s** (identical to the pre-collapse baseline of
      247; zero tests dropped or broken; the 4 warnings are pre-existing fork DeprecationWarnings).

## Parked (NOT this session)

- `t5_attack_eval.py` does not thread `precision` (uses the fp32/strict function default). Per the charter
  this is a **T5-restart prerequisite**, fixed at T5 restart, not in MCR. T5 does not run during MCR.
- Optional Phase-1 cleanup: thread `precision` explicitly into `training/loop.py` rather than sniffing
  `cudnn.deterministic` (functionally equivalent today; left for when the loop is rewritten for backbone
  training).
- The `*_tf32_*` launcher filenames are now misnomers (regime is bf16); filenames kept to preserve
  references in `findings_log.md` / `speedup_session_findings.md`. Rename is a separate cleanup.
