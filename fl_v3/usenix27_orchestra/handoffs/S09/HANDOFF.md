# S09 full-pipeline performance/readiness — terminal handoff

## Terminal state

```text
SESSION_ID: S09
STATE: CLOSED PASS UNDER O-120
BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
BRANCH_AT_EXECUTION: codex/s08-s09-cl-readiness
STOP2_IMPLEMENTATION: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
STOP3_EVIDENCE/CLOSURE: c28d09c34b0ff56fcbc3805a8361ccd26eeaccc1 / 84adfd0
STOP4D_IMPLEMENTATION: 5642884cdbb16e1c9b3107f529dc70b3a1243c6a
STOP4D_EVIDENCE/CLOSURE: 54e45c5fabe0a353ddd904f19cca655351ac09d1 / 5e5d9a9d35e04711e30bbe98512dfcd0866929b2
FINAL_REVIEW_SEAL: ced5992ea113bd21d7d545af505debf405b556b3
FINAL_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3
S09_CLOSING_COMMIT: 351b7a0b8419c01d0d32ba224babbc6bdc4213ba
INTEGRATION: v3-ad-perception ff-only advanced to 351b7a0 under O-121
```

`v3-ad-perception` and `codex/s08-s09-cl-readiness` currently identify the
same accepted tip/tree. The delivery branch is retained pending a separate owner
cleanup decision. S09 is closed; there is no active compute request.

## Scope actually delivered

S09 established the production trainval cache identities, a fail-closed bounded
readiness lifecycle, one production loader/100-update gate, bounded profiler and
batch-capacity evidence, two output-neutral engineering optimizations, and a fresh
1000-update stability/performance window.

Long-lived implementation retained:

- hash-bound `s09.v1/v2` execution/readiness contracts and exact data/dependency
  identity;
- accepted S08 precision partition, including the SECOND FP32 island;
- direct host/CUDA stage timing, exact attempted/accepted/counter accounting,
  peak allocated/reserved memory and terminal readiness artifacts;
- explicit Swin activation-checkpoint switch; and
- ordinary-path suppression of 19 redundant per-task loss `.item()` host
  synchronizations while retaining loss terms whenever runtime telemetry or S08
  diagnostics request them.

No model math, target, loss, normalization, optimizer, LR, scheduler, EMA,
augmentation, sampling, initialization, official metric/decode/NMS, data ownership
or precision-policy semantic was changed.

## Job ledger

| Job | Stop / exact purpose | Terminal result |
|---:|---|---|
| 441191 | STOP-1 train/val `t1.v2`, ten-sweep cache materialization | PASS, 00:03:06 |
| 441293 | STOP-2 focused GH200 readiness smoke | PASS, 44/44, 00:01:04 |
| 441511 | first STOP-3 G100 | FAIL before data/model: wrong module selector caused editable spconv JIT `cublasLt.h` failure |
| 442152 | O-118 dependency rebuild/attestation | PASS, 00:11:52 |
| 446225 | strict derived production loader + G100 | PASS, 100/103, 00:05:05 |
| 452520 | STOP-4A profiler + B=1/2/4 capacity | PASS, 00:09:42 |
| 455539 | optimized B=1 G100 | PASS, 100/103, 00:04:06 |
| 456539 | fresh optimized B=1 G1000 | PASS, 1000/1003, 00:06:54 |

There were no retries. The obsolete pre-remediation STOP-4C tuple at `131619f`
was reviewed NO-GO and never submitted. STOP-4's three jobs consumed exactly
`0.345000` of the two approved GPU-hours.

## Accepted engineering conclusions

### Production data

STOP-1 binds exact `v1.0-trainval`, `n_sweeps=10`, train/val `t1.v2`
caches to accepted ZIP-manifest logical SHA-256
`023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`
and physical SHA-256
`228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.
Exact cache hashes are in `RESULTS.md`. No payload extraction occurred.

### Lifecycle and performance

The optimized B=1 F-U G1000 tuple uses global FP16 with SECOND FP32, seed 0,
AdamW `1e-4/0.01`, constant scheduler, no EMA/clip/augmentation/GT paste,
uniform sampling, eight workers, world size/accumulation one, no checkpoint and
no official evaluation. It reached 1000 accepted updates in 1003 attempts: three
initial GradScaler overflow/backoff windows, then 1000 accepted and 990/990
post-warm measured windows. There were zero direct-nonfinite, discarded or pending
windows.

| Metric | STOP-3 baseline G100 | optimized G100 | optimized G1000 |
|---|---:|---:|---:|
| combined p50 | 208.746 ms | 183.215 ms | 178.024 ms |
| combined p95 | 224.327 ms | 217.674 ms | 203.231 ms |
| accepted throughput | 4.743 samples/s | 5.237 samples/s | 5.542 samples/s |
| steady epoch estimate | 1.647 h | 1.492 h | 1.410 h |
| peak allocated / reserved | 3.256 / 6.434 GiB | 4.764 / 8.361 GiB | 4.765 / 8.314 GiB |
| data-wait share | 0.076% | 0.099% | 0.097% |

The B=1 G1000 one-Hz interval reports utilization mean/p50/p95/max
`47.56/51/74.1/100%`, with only `2.51%` of samples at least 80%. This is not
full GH200 use. Negligible data wait points away from the loader and toward B=1
model/kernel granularity, without proving a specific branch or kernel cause.

STOP-4A confirms B=2 and B=4 fit and increase aggregate throughput:

| Cell | Throughput | Peak allocated / reserved |
|---|---:|---:|
| B1 checkpoint on + profiler | 3.507 samples/s | 3.06 / 5.38 GiB |
| B1 checkpoint off | 5.364 samples/s | 4.55 / 7.24 GiB |
| B2 checkpoint off | 6.925 samples/s | 8.55 / 13.34 GiB |
| B4 checkpoint off | 8.451 samples/s | 16.32 / 38.81 GiB |

B=2/B=4 are capacity/throughput observations, not a selected training recipe.
A full epoch at B=4 would have a two-sample tail because 28130 is not divisible
by four and the current fixed-batch loop rejects it; the 20-step capacity result
remains valid, but full-run batch/tail policy needs future design.

## S08 residual carried forward

Dynamic loss scaling changes temporary scaled arithmetic, not the unscaled
gradient applied by the optimizer. S09's stable accepted windows therefore do not
explain or cure the large true SECOND gradients. Tiny-group sparse GroupNorm is a
leading unproven hypothesis. Any normalization, architecture, loss/head, clipping
or recipe amendment requires a new S10 envelope.

## Closure compaction

Removed from the active tree because they were exact one-shot execution machinery:

- five S09 STOP-3/4 shell runners;
- seven exact STOP-3/4 JSON job configs; and
- the test that only compared the retired G100 and G1000 JSON files.

The dependency-attestation runner is especially retired because it temporarily
mutated/restored an editable cumm/spconv build state and is unsafe as a general
launcher. Exact bytes, commands and the original detailed handoff/request/review
ledgers remain recoverable from Git object `351b7a0`; immutable Arrhenius outputs
remain at the paths in `RUN_REQUEST.md`. Production readiness code, source
identity, the five fail-closed S07-derived templates, and long-lived focused tests
remain for the fresh S10 audit.

## Interpretation boundary and next session

S09 proves bounded single-seed engineering lifecycle/readiness and performance for
the stated tuple. It is not convergence, recipe/batch selection, model quality,
mAP/NDS, fusion gain, multi-seed capability, full GH200 utilization, Protocol A/B,
FL, attack or defense evidence.

Only the S10 work definition is accepted: centralized-model numerical/
architectural health, production recipe selection, and final-architecture GH200
optimization. The fresh S00 will create `codex/s10-cl-model-recipe` and use Ultra
reasoning before proposing exact stops. No previous stop design, full-run
placement, compute or S11+ boundary is accepted.
