# T2 — REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code.

## Verdict
`PASS`

No scientific-correctness changes requested. This is the re-review after commit
`387f3dd` ("canonical over-cap pillar truncation"), which addressed the previous
`CHANGES-REQUESTED` finding.

The prior finding was an `invariant-violation`: `PointPillarsEncoder` selected the first
`max_points` in input/file order for over-cap pillars, so a LiDAR point permutation could select a
different subset. The fix now performs canonical within-pillar truncation by successive stable sorts
over `(pillar_key, x, y, z, intensity)` before applying `within < max_points`. That makes the retained
subset a function of point content rather than input order; exact duplicate ties are value-equivalent
for the per-point-feature `torch.max` path. I re-ran the old minimal repro, and it now returns
`torch.equal=True` with `max|delta|=0.0`.

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- Nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- Nothing found. The previous over-cap PointPillars permutation-invariance violation is resolved by
  `fl_v3/src/fl_v3/models/fusion/lidar_encoder.py:96` through `fl_v3/src/fl_v3/models/fusion/lidar_encoder.py:116`,
  and is now covered by `fl_v3/tests/test_model_determinism.py::test_pillar_scatter_permutation_invariant_OVERCAP`.

### question
- Nothing found.

### style
(deprioritized — note only, do not block on these)
- Nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: Nothing found. T2 correctly does not claim a false bit-parity oracle for
  BEVFusion/LSS/CenterPoint; the LSS stable composite sort, CenterPoint decode convention, corrected
  Gaussian radius, PointPillars deterministic scatter, and T1 box/yaw convention match the SPEC/reference
  requirements I checked.
- Invariants (determinism, null-config): Nothing found. Static banned-op coverage, splat permutation
  invariance, over-cap pillar permutation invariance, deterministic decode, ResNet BN freeze-through-
  `train()`, A40 loud-fail gate, and `dummy_regression` byte-identity are covered. Null-config attack
  semantics are deferred beyond T2; the task-agnostic clean dummy path remains pinned by the golden.
- Calibration/units: Nothing found. The single `W -> x`, `H -> y`, `flat = row*W+col` BEV convention,
  resize affine / `lidar2img` rescale, and encode->decode yaw/dimension convention are covered by
  independent T1-anchored tests and did not show a unit/frame mismatch.
- Metric correctness: Nothing found. T2 only exposes the provisional center-distance learning proxy and
  explicitly defers official nuScenes mAP/NDS plus the 6-criterion ASR denominator to T4; I did not find
  a T2 metric-definition claim that conflicts with the roadmap.

## Verification Run
- Old minimal over-cap repro (one pillar, three points, `max_points=2`, reversed point order):
  `torch.equal=True`, `max|delta|=0.0`.
- `bash fl_v3/scripts/run_in_venv.sh python -m pytest -q fl_v3/tests/test_model_determinism.py::test_pillar_scatter_permutation_invariant_OVERCAP`
  - Result: `1 passed in 1.20s`.
- `bash fl_v3/scripts/run_in_venv.sh python -m pytest -q fl_v3/tests/test_model_determinism.py`
  - Result: `6 passed in 31.40s`.
- `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests`
  - Result: `148 passed, 4 warnings in 237.77s (0:03:57)`.
  - The warnings are multiprocessing `DeprecationWarning`s from fork in worker-loader tests, not scientific
    correctness failures.

## Residual Risk
- The A40 checksum reported in `collab/T2/SPEC.md` (`0a30410a9905010bd94d78959d89ee7f1fb05a116d28c75dec2667ef81af98e9`)
  was not independently re-run by this Codex session because it requires the A40 SLURM gate. The local
  suite and targeted repro confirm the code path and test coverage; the build-session A40 job remains the
  authoritative production bit-identity evidence.
