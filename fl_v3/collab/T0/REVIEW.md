# T0 — REVIEW (Codex re-review of commit 0977b7f)

> Written by the **Codex review session**. Reviews the new T0 fix commit against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code.

## Verdict
`CHANGES-REQUESTED`

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper/reference) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- Nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- `invariant-violation` · `fl_v3/src/fl_v3/strategy/defenses/multi_krum.py:119` and
  `fl_v3/src/fl_v3/strategy/defenses/multi_krum.py:124` · The fix declares MultiKrum to follow
  Flower's built-in one-shot oracle (`fl_v3/src/fl_v3/strategy/defenses/multi_krum.py:3`) and the
  SPEC now says FedAvg / NormClip / MultiKrum use a Flower fp32 aggregation replica to preserve
  clean/null bit identity (`fl_v3/collab/T0/SPEC.md:65`). But MultiKrum's selected-subset aggregate
  is explicitly *not* tested bit-for-bit against Flower: `fl_v3/tests/test_flower_fp32_parity.py:100`
  says Flower aggregates the selected subset in score order while fl_v3 aggregates in sorted index
  order, and `fl_v3/tests/test_flower_fp32_parity.py:105` uses `assert_allclose` rather than
  `array_equal`. Since fp32 addition is order-sensitive, this is not implementation-equivalent to the
  Flower/fl_v2 oracle at the byte level. The same comments also state fl_v3 uses a different squared
  distance formula than Flower (`fl_v3/tests/test_flower_fp32_parity.py:102`), which can change
  selection on near-tie inputs even if the current fixture passes.
  Minimal fix: if Flower is the declared MultiKrum oracle, match Flower exactly: use Flower's score
  formula/tie order, keep the selected clients in Flower score order for aggregation, and assert
  `np.array_equal` for both selected ids and final arrays in the live-Flower parity test. If the
  intended contract is only "same selected set, aggregate within tolerance", downgrade the SPEC and
  docstrings from bit-identity/oracle parity to a tolerance-level Flower-compatibility claim, and do
  not present MultiKrum as a bit-identical carry-over.

### question
- `question` · `fl_v3/scripts/run_in_venv.sh:47` · The fixed venv preflight works for
  `python -m pytest`, but the bare command recorded in the SPEC (`pytest fl_v3/tests`) still resolves
  to the system `pytest` in this environment, not the venv Python, and fails collection with
  `ModuleNotFoundError: fl_v3`. I verified this by running `fl_v3/scripts/run_in_venv.sh pytest
  fl_v3/tests` (fails) and `fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` (passes).
  Minimal fix: update the SPEC/evidence to use the documented `python -m pytest` invocation, or make
  `run_in_venv.sh` special-case `pytest` to execute `python -m pytest` so gate reproduction cannot
  accidentally use a system executable.

### style
(deprioritized — note only, do not block on these)
- Nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: FedAvg and NormClip clean-path aggregation are fixed by the new
  Flower-fp32 parity tests. FLAME, FoolsGold, FedMedian, gradient metrics, and partition still match
  the existing oracle fixtures. MultiKrum has the residual byte-level Flower parity issue above.
- Invariants (determinism, null-config): Found the MultiKrum bit-identity/oracle-parity issue above.
  Same-seed deterministic smoke passes in the current venv when using `python -m pytest`.
- Calibration/units: Nothing found. FLAME `lambda` remains config-driven and uses
  `sigma = lambda * S_t` without `sqrt(d)`.
- Metric correctness: Nothing found in T0 scope. ASR, mAP/NDS, utility/ASR success rules, and
  coordinate/box conventions are not implemented until later tasks.

## Verification Notes
- Reviewed commit: `0977b7f` (`Cycle 04 T0: address Codex review — Flower-fp32 parity, MultiKrum
  validity, venv preflight`).
- `fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` → `62 passed`.
- `fl_v3/scripts/run_in_venv.sh pytest fl_v3/tests` → collection failure because bare `pytest`
  resolves outside the venv and cannot import `fl_v3`.
