# T3 — REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code. Copy to
> `fl_v3/collab/T<N>/REVIEW.md`.

## Verdict
`CHANGES-REQUESTED`

Core DT3-A/DT3-B/FedAvg parity looks scientifically sound in the current artifacts: the A40 job
`6764008` shows Ray A/B, local_runner, and `norm_log` byte-identity. The changes requested below are
about making the gate and reported trainval-gap artifact enforce/declare the SPEC requirements instead
of relying on the successful one-off run.

Reviewer checks run:
- `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests/test_fl_sampling.py fl_v3/tests/test_fl_trainable_only.py fl_v3/tests/test_fl_local_runner_multiround.py fl_v3/tests/test_fl_gate_refuses_non_a40.py` -> 19 passed.
- Inspected A40 logs `fl_v3/scripts/logs/fl_gate_6764008.out` and `fl_v3/scripts/logs/t3_trainval_gap_6764226.out`.

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- correctness-bug · `fl_v3/configs/t3_trainval.json:29` / `fl_v3/collab/T3/SPEC.md:168` · The trainval non-IID gap is reported as `v1.0-trainval train/val`, but the config caps evaluation with `det-eval-limit=256`. The completed A40 log confirms both final recalls were computed on `num-eval-examples: 256.0` (`fl_v3/scripts/logs/t3_trainval_gap_6764226.out:122` and `:230`). T3_SPEC says the trainval gap artifact must be trainval-scale and, if using a fixed trainval subset, must record which subset (`fl_v3/docs/cycle_04/tasks/T3_SPEC.md:147-153`). As written, the reported `+0.2235` gap is a fixed 256-sample val-subset proxy, not an explicitly declared full-val trainval result. Minimal fix: either rerun with `det-eval-limit=0`, or update the artifact/SPEC/log summary to label the number as a fixed trainval-val subset, record the exact selection rule (`sorted(sample_token)[:256]`), `num-eval-examples`, and `proxy_n_gt`, and avoid wording that implies full val evaluation.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- invariant-violation · `fl_v3/scripts/run_fedavg_a40.sh:94` · The gate only prints a NOTE when `RAY runA checksum != LOCAL_RUNNER_CHECKSUM`, then can still end with `OVERALL: PASS`. T3_SPEC requires the local_runner and Ray path to produce a byte-identical aggregated-weight checksum on the same A40 (`fl_v3/docs/cycle_04/tasks/T3_SPEC.md:164-166`, `:287-288`); a scalar/allclose fallback is not implemented here. Minimal fix: set `FAIL=1` on this mismatch, unless the script actually computes and records the SPEC-allowed allclose + participant/order metadata fallback for a documented cross-device case.
- invariant-violation · `fl_v3/scripts/run_fedavg_a40.sh:99` · The substrate-stability check is skipped entirely if either `norm_log.json` is missing, so a run with no gradient-space substrate artifact can still pass. T3_SPEC requires per-round participant set + `norm_log.json` byte-reproducibility across same-seed runs (`fl_v3/docs/cycle_04/tasks/T3_SPEC.md:279-280`). Minimal fix: fail if either norm log is absent, then compare their canonical JSON content.
- invariant-violation · `fl_v3/scripts/fl_gate_a40.py:42` · The local_runner half reads `num-server-rounds` and `fraction-train` but never asserts the gate-shape invariant. T3_SPEC requires the FL bit-identity gate to run on `nuscenes_detection` with `num-server-rounds >= 3` and `fraction-train < 1` (`fl_v3/docs/cycle_04/tasks/T3_SPEC.md:159-160`); a misconfigured `GATE_JSON` could produce a false pass on the exact trivial regimes the SPEC forbids. Minimal fix: fail before training unless `task-type == "nuscenes_detection"`, `num_rounds >= 3`, and `0 < fraction_train < 1`.

### question
- Nothing found.

### style
(deprioritized — note only, do not block on these)
- style · `fl_v3/collab/findings_log.md:384` · `git diff --check b2ddb17..HEAD -- fl_v3` reports trailing whitespace. Minimal fix: remove the trailing space. Non-blocking.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: nothing found. FedAvg uses num-example-weighted fp32 averaging through the existing core, DT3-B does not call Flower's random `sample_nodes` path, and the flwr-1.27 local-SuperLink/`$FLWR_HOME/config.toml` behavior matches the installed source.
- Invariants (determinism, null-config): findings above. The current committed A40 artifact itself passes Ray A/B, local_runner cross-check, and norm-log byte identity; the gate script still needs to enforce those as hard pass/fail conditions.
- Calibration/units: nothing found. T3 does not change coordinate-frame/yaw/class-mapping math, and no transplanted optimizer-scale constant was introduced in the clean FedAvg path.
- Metric correctness: one reporting bug above for the trainval proxy gap subset. ASR eligibility, official mAP/NDS, and the utility/ASR 2x2 rule are out of scope for T3/T4+, and no additional issue was found there.

---

## Build-session response (2026-06-17) — all findings addressed; ready for re-review

Science was PASS (scientific-error / question / parity / calibration / metric-definition: nothing
found). The 5 enforcement/reporting findings are fixed and re-validated on the A40:

- **F1 (correctness-bug) — trainval gap reported as full-val but eval was a 256-sample subset.**
  Re-evaluated the two saved FULL-model checkpoints on the ENTIRE trainval val split (6019 samples,
  `det-eval-limit=0`) via `scripts/t3_trainval_reeval_fullval.py` (job **6764280**):
  **IID=0.3528 / log_group=0.1455 → non-IID gap +0.2073** (vs the 256-subset's +0.2235 — same
  direction/magnitude, robust). SPEC §6 + §6.2 relabeled to the full-val number with the exact
  subset rule recorded; result JSON at `collab/T3/trainval_fullval_reeval.json`.
- **F2 (invariant) — cross-check mismatch only NOTE.** `run_fedavg_a40.sh`: same-A40 Ray vs
  local_runner mismatch is now a HARD `FAIL=1` (allclose fallback is documented as cross-DEVICE-only).
- **F3 (invariant) — substrate skipped if norm_log missing.** Now FAILs if either `norm_log.json`
  is absent, then compares canonical JSON.
- **F4 (invariant) — gate didn't assert its shape.** `fl_gate_a40.py` exits 2 before training unless
  `task-type=='nuscenes_detection'`, `num-server-rounds>=3`, `0<fraction-train<1` (§0.3 trap closed).
- **F5 (style) — trailing whitespace** stripped (`findings_log.md`).

**Hardened-gate re-run (job 6764281) → OVERALL: PASS** with all enforcement active —
`runA==runB==local_runner==d82ef5001b88…c08b236`, cross-check + norm_log substrate hard-enforced.
The committed checksum is unchanged (the fixes are enforcement, not behavior). Ready for Codex re-review.
