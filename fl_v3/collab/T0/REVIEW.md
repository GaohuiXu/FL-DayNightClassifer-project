# T0 — REVIEW (Codex re-review of commit eaf59df)

> Written by the **Codex review session**. Reviews the new T0 fix commit against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code.

## Verdict
`PASS`

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper/reference) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- Nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- Nothing found.

### question
- Nothing found.

### style
(deprioritized — note only, do not block on these)
- Nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: Nothing blocking found. FedAvg and NormClip remain documented and tested
  as bit-identical Flower-fp32 aggregation replicas for the clean/null path. FLAME, FoolsGold,
  FedMedian, gradient metrics, and partition logic still rely on the existing oracle fixtures.
  MultiKrum is now explicitly scoped in `fl_v3/collab/T0/SPEC.md` as Flower-compatible rather than a
  bit-identical carry-over, with the stable `||x-y||^2` distance documented as an intentional
  numerical divergence. That removes the previous false bit-parity claim.
- Invariants (determinism, null-config): Nothing found. The prior gate-reproduction issue is fixed:
  `fl_v3/scripts/run_in_venv.sh pytest fl_v3/tests` now routes bare `pytest` through
  `python -m pytest` inside the venv. Same-seed and parity tests pass in the current gate run.
- Calibration/units: Nothing found. FLAME `lambda` remains config-driven and the noise scale remains
  `sigma = lambda * S_t`, so the earlier Adam-vs-SGD-style constant transplant risk is not present in
  this T0 diff.
- Metric correctness: Nothing found in T0 scope. ASR, official mAP/NDS, utility/ASR success rules,
  and coordinate/box conventions are deferred to later tasks and are not implemented here.

## Verification Notes
- Reviewed commit: `eaf59df` (`Cycle 04 T0: address Codex re-review — MultiKrum
  Flower-compatible scope, pytest-via-venv`).
- Diff scope checked: `git diff --name-status 0977b7f..HEAD -- fl_v3`.
- Gate run: `fl_v3/scripts/run_in_venv.sh pytest fl_v3/tests` -> `62 passed`.
