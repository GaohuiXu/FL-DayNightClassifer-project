# T0 — REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code. Copy to
> `fl_v3/collab/T<N>/REVIEW.md`.

## Verdict
`CHANGES-REQUESTED`

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- `scientific-error` · `fl_v3/src/fl_v3/strategy/defenses/multi_krum.py:105` and
  `fl_v3/src/fl_v3/strategy/defenses/multi_krum.py:108` · The implementation computes Krum scores
  once and selects the `m` smallest scores in one shot. Blanchard et al., Section 6, defines m-Krum as
  repeated Krum selection: select one vector, remove it, then iterate `m-1` times. The same section's
  Proposition 3 assumes `2f + 2 < n - m` for m-Krum. The current validity gate at
  `fl_v3/src/fl_v3/strategy/defenses/multi_krum.py:40` only checks `n >= 2f + 3` and `1 <= m <= n`,
  so configurations without the paper's robustness condition are treated as valid. The test fixture
  also encodes this invalid case as expected behavior: `fl_v3/tests/test_multikrum.py:33` uses
  `n=5, f=1, m=3`, but `2f + 2 < n - m` is `4 < 2`, false. This violates the T0 SPEC's requirement
  that MultiKrum invalid `(n,f,m)` configs are marked NA, not forced (`fl_v3/collab/T0/SPEC.md:69`),
  and the plan's defense card rule to mark invalid MultiKrum cells invalid/NA.
  Minimal fix: either implement paper m-Krum literally by recomputing Krum after each removal and
  enforce the paper validity condition, or explicitly declare that `fl_v3` follows Flower's built-in
  one-shot MultiKrum variant and validate bit/decision parity against Flower. In either case, update
  `multi_krum_valid` and the hand-computed fixture so invalid `(n,f,m)` cells cannot silently become
  defended results.

### correctness-bug
- Nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- `invariant-violation` · `fl_v3/src/fl_v3/strategy/defenses/fedavg.py:46` and
  `fl_v3/src/fl_v3/strategy/defenses/fedavg.py:67` · Clean FedAvg and NormClip aggregate through the
  fp64 update-form core, while the frozen `fl_v2` oracle clean path delegates to Flower FedAvg
  (`fl_v2/src/fl_v2/strategy/norm_tracking_fedavg.py:280`) and the `fl_v2` NormClip path clips first
  then delegates to Flower FedAvg (`fl_v2/src/fl_v2/strategy/norm_clipped_fedavg.py:110`). The new
  core documents that this is intentionally not bit-identical to the oracle for clean-FedAvg/NormClip
  (`fl_v3/src/fl_v3/strategy/aggregation_core.py:13`). That conflicts with the project crown-jewel
  invariant that null-config reproduces the clean baseline bit-for-bit (`AGENTS.md:31`) and with the
  T0 carry-over scope listing FedAvg/NormClip among oracle-checked defense cores
  (`fl_v3/collab/T0/SPEC.md:36`). I confirmed the difference on the committed oracle fixture:
  uniform FedAvg via fp64 update-form vs direct fp32 mean was not `array_equal` for all four tensors
  (max absolute difference `1.49e-08` to `2.98e-08`). This is small numerically but not bit-identical,
  so later null-config or clean-baseline equality checks can fail silently.
  Minimal fix: add a separate Flower-compatible fp32 direct weighted-average core for `none`/FedAvg
  and for NormClip after clipping, then add final-array parity tests for clean FedAvg and NormClip.
  Keep the fp64 update-form core only where the oracle uses it (FLAME/FoolsGold), or explicitly scope
  null bit-identity away from `fl_v2`/Flower and update the contract before relying on it.

### question
- `question` · `fl_v3/scripts/run_in_venv.sh:5` · I could not reproduce the claimed full gate in this
  worktree. Running `fl_v3/scripts/run_in_venv.sh pytest fl_v3/tests` initially failed collection
  because `fl_v3` was not importable. With `PYTHONPATH=fl_v3/src`, collection proceeded but the
  activated external venv lacked `flwr`; a non-Flower subset then ran `40 passed, 5 failed`, with all
  failures due to missing `sklearn.cluster.HDBSCAN`. The build SPEC claims `pytest fl_v3/tests -> 43
  passed` and `HDBSCAN OK` (`fl_v3/collab/T0/SPEC.md:118` and `fl_v3/collab/T0/SPEC.md:129`), so the
  current review environment does not reproduce the stated evidence. Is the reviewed worktree
  expected to rely on a shared absolute venv outside the worktree? Minimal fix: rebuild the verified
  venv from `build_venv.sh`, or make `run_in_venv.sh` fail early with import/version checks for
  `fl_v3`, `flwr`, and `sklearn` so the T0 gate is reproducible.

### style
(deprioritized — note only, do not block on these)
- Nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: Found a paper-parity error in MultiKrum. FLAME/FoolsGold arithmetic and
  the NormClip clipping primitive matched the frozen `fl_v2` references by code inspection and
  committed fixtures, but I could not execute FLAME parity locally because `sklearn` was missing.
- Invariants (determinism, null-config): Found a null/clean bit-identity violation for FedAvg and
  NormClip. Same-seed local core determinism tests excluding FLAME passed in my environment.
- Calibration/units: Nothing found. FLAME `lambda` is config-driven and uses `sigma = lambda * S_t`
  without `sqrt(d)`, matching the SPEC/oracle.
- Metric correctness: Nothing found in T0 scope. ASR, mAP/NDS, utility/ASR success rules, and
  coordinate/box conventions are not implemented until later tasks.

## Verification Notes
- Reviewed diff scope: `git diff --stat v2-new-api..HEAD -- fl_v3`.
- Test attempt 1: `fl_v3/scripts/run_in_venv.sh pytest fl_v3/tests` failed collection with
  `ModuleNotFoundError: No module named 'fl_v3'`.
- Test attempt 2: `PYTHONPATH=fl_v3/src fl_v3/scripts/run_in_venv.sh pytest fl_v3/tests` failed
  collection for Flower wrapper tests with `ModuleNotFoundError: No module named 'flwr'`.
- Test attempt 3: non-Flower subset with `PYTHONPATH=fl_v3/src` collected 45 tests: `40 passed`,
  `5 failed`; all 5 failures were FLAME paths failing at `from sklearn.cluster import HDBSCAN`.
- Source checked for MultiKrum paper condition: Blanchard et al., "Byzantine-Tolerant Machine
  Learning", arXiv:1703.02757, Section 6 / Proposition 3.
