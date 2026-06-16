# T2 — REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code.

## Verdict
`CHANGES-REQUESTED`

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- Nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- invariant-violation · `fl_v3/src/fl_v3/models/fusion/lidar_encoder.py:96` and `fl_v3/src/fl_v3/models/fusion/lidar_encoder.py:107` · The PointPillars encoder is not input-permutation invariant once a pillar exceeds `max_points`. The task contract requires a "permutation-invariance test (permuted point order -> byte-identical splat+scatter)" (`fl_v3/docs/cycle_04/tasks/T2_SPEC.md:200`) and the filled SPEC claims pillar scatter is byte-identical under input permutation (`fl_v3/collab/T2/SPEC.md:93`). But the implementation sorts only by `pillar_key` (`argsort(pillar_key, stable=True)`), so ties within the same pillar preserve the incoming/file order; then `cap = within < self.max_points` keeps the first `max_points` points in that order. With the default `max_points=32`, any over-cap pillar will select a different subset after a LiDAR point permutation, changing the PFN/max-pooled feature and the BEV canvas. I reproduced the failure with one pillar, three points, and `max_points=2`: `torch.equal(a, b) == False`, `max|delta| ~= 1.97e-05`. The existing test masks this path by setting `max_points=128` with an explicit "no truncation" comment (`fl_v3/tests/test_model_determinism.py:90`). Minimal fix: make within-pillar truncation canonical before the cap, e.g. sort by a deterministic secondary key derived from point contents (`pillar_key`, then quantized/bitwise `x,y,z,intensity,ring`, with deterministic duplicate handling), or remove the order-dependent cap by aggregating all points / failing loudly when any pillar exceeds the cap. Add an over-cap permutation-invariance test, and include an over-cap cluster in the A40 gate so the production checksum exercises the fixed path.

### question
- Nothing found.

### style
(deprioritized — note only, do not block on these)
- Nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: Nothing found. T2 correctly does not claim a false bit-parity oracle for BEVFusion/LSS/CenterPoint; the LSS stable composite sort, CenterPoint decode convention, corrected Gaussian radius, and T1 box/yaw convention matched the SPEC/reference requirements I checked.
- Invariants (determinism, null-config): Finding above. Static banned-op coverage, deterministic decode, ResNet BN freeze-through-`train()`, A40 loud-fail gate, and `dummy_regression` byte-identity otherwise matched the SPEC surface I reviewed. Null-config attack semantics are deferred beyond T2; the task-agnostic clean dummy path is pinned by the golden.
- Calibration/units: Nothing found. The single `W -> x`, `H -> y`, `flat = row*W+col` BEV convention, resize affine / `lidar2img` rescale, and encode->decode yaw/dimension convention are covered by independent T1-anchored tests and did not show a unit/frame mismatch.
- Metric correctness: Nothing found. T2 only exposes the provisional center-distance learning proxy and explicitly defers official nuScenes mAP/NDS plus the 6-criterion ASR denominator to T4; I did not find a T2 metric-definition claim that conflicts with the roadmap.
