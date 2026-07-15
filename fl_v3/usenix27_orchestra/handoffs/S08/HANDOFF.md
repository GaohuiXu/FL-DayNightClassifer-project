# S08 precision qualification — terminal handoff

## Terminal state

```text
SESSION_ID: S08
STATE: CLOSED PASS UNDER O-110
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
BRANCH_AT_EXECUTION: codex/s08-s09-cl-readiness
IMPLEMENTATION_COMMIT: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
REMEDIATION_COMMIT: 103c7389a47938b1f9dd0cba60251df6dce9e5bb
R3_REVIEWED_EVIDENCE_SHA: c0ef86235ead753fee3b790b19d40f82f875ec59
R3_VERDICT: PASS_WITH_RESIDUAL_RISK / no P0-P2
OWNER_CLOSE_SEAL: d31adea049c84e47a0e4f82f38f22a2ca91a5a6f
S08_CLOSING_COMMIT: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
INTEGRATED_TIP: 351b7a0b8419c01d0d32ba224babbc6bdc4213ba
COMPUTE: Jobs 426619,427800,428112,428889,429080,431013,435151
```

O-110 accepts the reviewed policy, closes S08, and authorizes the original
fast-forward-only integration. O-121 records that the later S09 closing tip
`351b7a0`, which contains S08, is now the integrated `v3-ad-perception` tip.

## Accepted precision policy

| Route | Accepted production partition | Status |
|---|---|---|
| Camera C-STR8 | global FP16 autocast | accepted |
| Dense-pillar L-P020 | global FP16 autocast; sparse partition not applicable | bounded compatibility accepted |
| Sparse L-S075 | global FP16 with SECOND voxelization/VFE/spconv/dense collapse/to-BEV in FP32 | accepted |
| Fusion F-U/F-CBGS | global FP16 with the same SECOND FP32 island | accepted |
| All routes | uniform FP32 | reference/fallback |
| Sparse L/F | full sparse-convolution FP16 | rejected as the unified fusion-capable policy |

The resolver remains fail-closed and records global and sparse precision in the
resolved config/provenance. Direct sparse BF16 remains rejected. The full-sparse
FP16 combination can remain mechanically expressible for labelled diagnostics or
future ablation; it is not the accepted production policy.

The existing spconv 2.3.8 no-grad evaluation workaround remains version-gated and
restores dispatch state in `finally`. S08 did not change task groups, targets,
loss equations, head architecture, optimizer recipe, official decode/NMS/metric,
data ownership, attack, or defense behavior.

## Delivered implementation

- Explicit global/sparse precision partition in resolved config, task construction,
  run provenance, checkpoint and resume identity.
- An explicit autocast-disabled FP32 island around the complete SECOND sparse path;
  camera/fusion/head remain eligible for global FP16 autocast.
- Production-loop persistent dynamic `GradScaler`, including bounded backoff below
  one, unscale before diagnostics/finiteness/clip/step, and accepted-window-only
  optimizer/scheduler/EMA/exposure advancement.
- Opt-in output-neutral precision diagnostics for scaler transitions, skipped and
  accepted windows, per-task losses, nonfinite counts, first bad parameter, stable
  norms/maxima, head input and SECOND output/stage/stem gradients.
- Source/dependency identity checks and focused long-lived regression coverage.

The precision diagnostic seam is retained because the large-gradient root cause is
still open. It is disabled by default and does not install generic module hooks or
change model outputs.

## Execution summary

| Job | Purpose | Terminal result |
|---:|---|---|
| 426619 | SMOKE-1 | FAIL before pytest: provenance policy rejected the evidenced spconv metadata patch |
| 427800 | SMOKE-2 | FAIL after runtime attestation: 103 pass / 3 fail |
| 428112 | SMOKE-3 | PASS: 106/106 |
| 428889 | SMOKE-4 | FAIL: 115/116; synthetic calibration mutation was a no-op |
| 429080 | SMOKE-5 | PASS: 116/116 plus 1/1 fixture attestation |
| 431013 | Q1 primary eight-cell qualification | completed 0:0 in 00:04:02; F2 preserved bounded FAIL |
| 435151 | Q2 L-P020/F-CBGS compatibility | completed 0:0 in 00:03:56; both cells PASS |

No job was silently retried. Q1+Q2 consumed `00:07:58` of O-109's two-GPU-hour
ceiling. Exact tuple identities and raw paths are in `RUN_REQUEST.md`; numerical
results and artifact checksums are in `RESULTS.md`.

## Key numerical conclusion

Q1 accepted C1/C2/L1/L2/L3/F1/F3 and rejected F2 within 18 attempts. L-S075 full
sparse FP16 accepted only after scale backoff to `0.03125`; F-U full sparse FP16
did not accept through attempted scale `0.00390625`. Its final failure had finite
loss, head-input gradient and SECOND activation-boundary gradients, but ten
nonfinite elements beginning at
`lidar_encoder.backbone.stem.0.weight`. L3/F3 with the SECOND FP32 island
accepted at scales 32/16.

This localizes the practical failure to sparse-convolution backward/weight-gradient
dynamic range, especially the SECOND stem. It does not identify one faulty
operation, prove AMP impossible, or show a head/loss semantic defect. The true
unscaled FP32 gradients are themselves unusually large; dynamic scaling does not
shrink the gradient applied by the optimizer. Tiny-group sparse GroupNorm remains
a leading hypothesis, not a finding.

Q2 accepted L-P020 at scale 8 and F-CBGS with the SECOND FP32 island at scale 16.
The replayed F-CBGS batch binds the CBGS config identity but does not qualify the
sampling distribution or loader.

## Closure compaction

Removed from the active tree after S08/S09 closure:

- `scripts/run_s08_precision_smoke.sh`;
- `scripts/submit_s08_precision_smoke.sh`;
- `tests/test_s08_precision_qualification.py`; and
- `tests/fixtures/s08_q1_raw_input_manifest.json`.

These were replay-frozen, one-shot qualification machinery rather than production
interfaces. Their exact pre-compaction bytes and the original 482-line handoff are
recoverable from Git object
`351b7a0b8419c01d0d32ba224babbc6bdc4213ba`. Raw Arrhenius outputs remain
immutable at the paths recorded in the terminal ledger. Long-lived production
precision code and focused partition/diagnostic/source-identity regressions remain.

## Residual risks and forbidden claims

- Large true SECOND gradients and their normalization/architecture/recipe cause
  remain unresolved.
- Q1 uses one replay-frozen mini fixture, random initialization, batch one and at
  most three accepted updates per qualifying primary cell; Q2 uses one accepted
  update per compatibility cell.
- R3 retained one non-blocking test residual: Q2's predicate did not explicitly
  gate `missing_grad_parameter_count == 0`; independent raw-record inspection
  verified zero missing gradients in both accepted cells.
- S08 is not convergence, performance, capacity, mAP/NDS, production-data,
  multi-seed, Protocol A/B, attack, defense, or scientific-result evidence.

See `RESULTS.md`, `RUN_REQUEST.md`, `REVIEW.md`, and
`MODEL_RECIPE_AUDIT.md` for the compact terminal evidence and pre-S10 technical
context.
