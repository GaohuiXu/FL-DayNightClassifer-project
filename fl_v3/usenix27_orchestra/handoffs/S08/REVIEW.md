# S08 independent review — terminal record

## Final verdict

```text
FINAL_REVIEW_ROUND: R3
REVIEWED_EVIDENCE_SHA: c0ef86235ead753fee3b790b19d40f82f875ec59
VERDICT: PASS_WITH_RESIDUAL_RISK
P0: none
P1: none
P2: none
P3: two documented, non-blocking closure residuals
OWNER_CLOSE: O-110 / seal d31adea049c84e47a0e4f82f38f22a2ca91a5a6f
ADDITIONAL_COMPUTE: none
```

The reviewer independently read the exact implementation/evidence diff, production
precision/config/training/model/checkpoint source, Q1/Q2 test source, immutable
snapshots, one-shot job scripts, Slurm accounting and every declared raw artifact.
The reviewer did not fix source or submit compute.

## Review chronology

| Round | Immutable baseline | Verdict and disposition |
|---|---|---|
| R1 | implementation/evidence `791aba97f7bbe92e7708b63f94f2e7d8599f91be` | REMEDIATE: one P1 and two P2 |
| R2 | remediation `103c7389a47938b1f9dd0cba60251df6dce9e5bb` plus Smoke-5 | PASS_WITH_RESIDUAL_RISK; original P0-P2 closed |
| R3 | Q1/Q2 evidence `c0ef86235ead753fee3b790b19d40f82f875ec59` | PASS_WITH_RESIDUAL_RISK; no P0-P2; owner-ready |

R1 findings and closure:

- P1: Q1 initially recorded fixture identity dynamically. Remediation prebound the
  full raw-input, batch-tensor, augmentation-order/value and canonical fixture
  identities and rejected mismatch before model/optimizer construction.
- P2: scheduler transitions and EMA-disabled state were recorded but not gated.
  Remediation made continuity/final scheduler state and EMA absence part of every
  cell predicate.
- P2: canonical status prose lagged the explicit partition and smoke state.
  Remediation reconciled active authority wording.

Smoke-4 then exposed only a no-op hostile-test mutation; Smoke-5 verified the
corrected negative test and fixture attestation. No production model, loss,
optimizer, precision route or data semantic changed in that correction.

## R3 adversarial reconstruction

The reviewer did not treat pytest exit as a numerical verdict. It independently
reconstructed:

- all five fixture identities and resolved-config hashes;
- exact cell order and all 66 Q1 plus 13 Q2 raw window records;
- matched initialized state and replay RNG within each C/L/F comparison;
- six-task loss presence/order/sums;
- gradient presence, explicit unscale, finiteness and first-bad parameter;
- persistent scaler transitions, including backoff below one;
- accepted/skipped optimizer, scheduler, EMA and exposure deltas; and
- every output checksum, immutable snapshot tree and unique Slurm submission.

Q1 independently reproduces C1/C2/L1/L2/L3/F1/F3 PASS and F2 bounded FAIL. Q2
independently reproduces P1/B1 PASS. No hidden extra cell, retry, seed, data scan,
profile, DDP run or work-chain expansion was found.

## Precision-path review

The reviewed production path:

1. fail-closes on the explicit global/sparse precision partition;
2. keeps voxelization/mean VFE FP32 in both sparse AMP regimes;
3. either runs SECOND/dense collapse/to-BEV under full sparse FP16 or within the
   explicit FP32 island;
4. promotes head outputs recursively to FP32 before the unchanged six-task loss;
5. unscales before diagnostics/finiteness/clip/step; and
6. advances optimizer/scheduler/EMA/exposure only for accepted windows.

The version-gated spconv 2.3.8 evaluation workaround remains isolated and restored
in `finally`. Diagnostics are opt-in, hook-free and output-neutral. Checkpoint/
resume binds the resolved partition and scaler/runtime state.

The raw F2 evidence localizes the practical failure to FP16 sparse-convolution
weight-gradient range at the SECOND stem. It does not prove one bad kernel,
architectural cause, head/loss defect, or universal impossibility of AMP. The
large finite FP32 gradients remain a real unresolved health signal.

## Residual P3 and risk

- Some duplicated pre-close prose was stale. This closure compaction removes that
  active contradiction; the original review wording remains recoverable at Git
  object `351b7a0`.
- Q2's automatic predicate required finite global gradients but did not explicitly
  require `missing_grad_parameter_count == 0`. Independent raw inspection found
  zero missing gradients for P1 and B1, so immutable Q2 evidence remains valid.
  The one-shot predicate was retired after closure rather than rerun.

Scientific residuals remain larger than these documentation/test issues: one mini
fixture, batch one, random initialization, short bounded windows, unusually large
true sparse gradients and no convergence/capability/performance evidence.

## Gate disposition

The review supports, and O-110 accepts:

- global FP16 for current camera/dense-pillar;
- global FP16 with the complete SECOND sparse path in FP32 for current sparse
  LiDAR/fusion; and
- uniform FP32 reference/fallback.

Full sparse FP16 is not accepted as the unified fusion-capable route. This verdict
does not authorize a recipe/normalization/head/loss change, compute, full-data
training, mAP/NDS, Protocol A/B, attack or defense.

The complete 690-line pre-compaction multi-round review is retained in Git object
`351b7a0b8419c01d0d32ba224babbc6bdc4213ba`; this file is the terminal active
review record.
