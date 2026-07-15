# S09 full-pipeline performance and readiness — envelope handoff

## State

```text
SESSION_ID: S09
MILESTONE_STATE: STOP-1 TERMINAL PASS / INDEPENDENT REVIEW PENDING
BASE_AND_CURRENT_HEAD: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
BRANCH: codex/s08-s09-cl-readiness
S08_POLICY_DECISION: O-110 / CLOSED PASS
S09_SCOPE_DECISION: O-111
IMPLEMENTATION_COMMIT: none
APPROVED_COMPUTE: STOP-1 Job 441191 consumed / no active compute
JOBS: 441191 COMPLETED 0:0 in 00:03:06 / no retry
INDEPENDENT_REVIEW: not started
```

This file records the accepted S09 execution envelope. The owner accepted
`28f7980` as the S08 close/S09 base,
accepted a four-stop S09 direction, assigned engineering performance/readiness to
S09, and deferred scientific training-recipe and sparse-normalization decisions to
S10. O-112 separately started STOP-1 DATA and authorized its exact request/evidence
commits plus one bounded cache-materialization submission. Job `441191` consumed
that submission and is terminal PASS at the execution/evidence level; independent
data/provenance review remains required before owner STOP-1 acceptance. STOP-2
through STOP-4 still require their own owner-reviewed plan and Git/Slurm authority.

## S08 residual carried into S09

S08 did **not** reduce the true gradients consumed by the optimizer. In AMP,
`GradScaler` multiplies the loss before backward and the production loop calls
`unscale_` before finiteness checks, clipping, and `optimizer.step()`. Dynamic
backoff therefore changes the temporary FP16 loss scale, not the final unscaled
gradient or effective learning rate.

The bounded evidence instead established:

- full sparse FP16 for L-S075 first accepted only after 14 backoffs at scale
  `0.03125`;
- full sparse FP16 for F-U did not accept any of 18 attempted windows, through
  attempted scale `0.00390625` (the terminal update produced `0.001953125`, which
  was not attempted);
- keeping SECOND voxelization/VFE/spconv/dense-collapse/to-BEV in FP32 allowed
  L-S075 and F-U to accept at scales `32` and `16`; and
- the accepted policy still uses FP16 autocast on eligible camera, dense fusion,
  and head work, with FP32 as the reference/fallback.

The large **unscaled FP32** LiDAR gradients remain a real unresolved issue. Job
`389356` observed maximum absolute gradients of approximately `1.91e6` for
L-S075 and `1.22e6` for F-U. Parameter count alone does not explain this. The
leading code-derived hypothesis is repeated per-active-voxel tiny-group
`GroupNorm` in the sparse SECOND stem/stage1: at 16 channels,
`GroupNorm(8, 16, eps=1e-5)` normalizes only two values per group and is repeated
several times. This is consistent with finite forward values and high backward
gain, but no variance/gain localization or controlled normalization comparison
has proved causality.

S09 will record whether this residual causes nonfinite windows, skips, counter
drift, instability, memory pressure, or unusable throughput under the already
accepted precision partition. S09 will not hide it with gradient clipping or
change the model/recipe to explain it. A normalization, head/loss/target,
optimizer, scheduler, EMA, augmentation, sampling, initialization, or gradient-
clipping change belongs to S10 or a separately approved amendment if it blocks
S09's engineering gate.

## Binding S09 scope

S09 answers whether the current six-task centralized pipeline can be run
efficiently and accountably on one GH200 with the production ZIP/cache path and
the accepted S08 precision policy. It establishes cache identity, bounded loader
and training throughput, stage/end-to-end latency, data wait, memory headroom,
successful-step/exposure accounting, and an epoch-time estimate.

Its optimization scope is measurement-backed and output-neutral: ZIP/cache
access, DataLoader lifecycle, host-to-device transfer, redundant conversion/
synchronization/allocation, and bounded logging/checkpoint overhead. A candidate
must preserve sample order/contents, model/loss/gradient/update semantics, the
O-110 precision partition, and exposure accounting. It receives an exact owner-
reviewed file and equivalence-test envelope before implementation. No optimization
is invented merely to create work when the measured readiness gate already
passes.

The base-uniform readiness recipe is an engineering control, not a claim that the
training recipe is scientifically optimal. Unless a future stop envelope changes
it with owner approval, the provisional control is F-U, random initialization
with a frozen engineering seed, AdamW (`lr=1e-4`, `weight_decay=0.01`), constant
scheduler, EMA off, gradient clipping off, 3D BEV augmentation off, GT paste off,
microbatch 1, accumulation 1, one GH200, and global FP16 autocast with the SECOND
FP32 island.

## Four owner-inspected stops

| Stop | Entry condition | Bounded deliverable | Required exit review |
|---|---|---|---|
| STOP-1 — DATA | Owner accepts the exact cache/materialization request | Exact production `t1.v2`, `n_sweeps=10` train/val cache and manifest identities; counts and integrity evidence | Independent data/provenance review before production binding |
| STOP-2 — IMPLEMENTATION | STOP-1 identity accepted; owner accepts exact files/Git/smoke envelope | Minimal output-neutral timing/memory/accounting instrumentation, resolved provenance, focused local/runtime checks, immutable implementation SHA | Independent diff/tests/request review before G100 |
| STOP-3 — G100 | STOP-2 reviewed; owner accepts exact immutable G100 tuple and quota | Loader worker selection plus one bounded 100-successful-step F-U gate; latency/throughput/data-wait/memory/counter/stability evidence and epoch estimate | Independent evidence review, then owner PASS/REMEDIATE decision |
| STOP-4 — G1000 | STOP-3 reviewed PASS; owner separately accepts exact conditional tuple | One fresh 1000-successful-step readiness run under the frozen STOP-3 policy | Independent evidence review and owner close decision |

At each stop, S00 first presents the exact plan, files and immutable identities,
acceptance/stop criteria, and proposed GPU quota. After one owner approval of that
stop, S00 creates a bounded goal and proceeds continuously through ordinary local
fixes and the approved job. Material changes to data, model output, precision,
recipe, scientific interpretation, cells, seed, or resources stop and return to
the owner. Only a future explicitly opted-in O-009/O-107 STOP-2 smoke may use its
capped mechanical replacement rule; unused time from a STOP-1/3/4 material job
does not authorize a retry or a new cell.

If STOP-3 identifies a failed engineering threshold with a specific safe
bottleneck, its deliverable may be an exact output-neutral optimization proposal.
It does not patch and rerun implicitly. The owner must amend STOP-3 with the exact
files, equivalence tests, immutable execution tuple, and quota before a replacement
gate; STOP-4 remains blocked until an accepted G100 exists.

## Proposed implementation ownership for STOP-2

The current maximum file envelope is:

- `fl_v3/src/fl_v3/config/resolved.py` — fail-closed resolved performance/readiness
  fields and provenance;
- `fl_v3/scripts/centralized_train.py` — bind the production cache/manifest and
  emit the bounded readiness artifact;
- `fl_v3/src/fl_v3/training/loop.py` — direct, output-neutral interval and counter
  accounting;
- at most one small timing/accounting helper under `fl_v3/src/fl_v3/training/` if
  keeping the loop cohesive requires it;
- `fl_v3/configs/s07_b_l_s075.json`, `s07_b_f_u.json`, and
  `s07_b_f_cbgs.json` only where explicit accepted precision/readiness provenance
  must replace template ambiguity;
- focused resolved-config, runtime, and checkpoint/accounting tests; and
- the S09 handoff/request/results/review package and necessary canonical status
  wording.

The implementation may use direct host timestamps and bounded CUDA events and may
record peak allocated/reserved memory. It must not install hooks, retain
activations/gradients, enable the S08 window observer, add a general profiler or
process/source-manifest framework, or promise a misleading mid-epoch resumable
checkpoint without sampler-cursor state.

## Explicit non-goals

- no mAP/NDS, official metric/decode/NMS, capability, convergence, or fusion-gain
  claim;
- no architecture, sparse normalization, head/loss/target, optimizer, scheduler,
  EMA, augmentation, sampler, initialization, or clipping optimization;
- no multi-seed, matrix, Protocol A/B, FL, attack, or defense;
- no DDP unless STOP-3 measures a concrete need and the owner grants a distinct
  later approval;
- no full-data extraction or duplication, performance profiler, retired harness,
  or automatic retry; and
- no S10-S12 implementation or scheduling from this envelope.

## Open owner decisions

O-112 authorizes S00 to freeze the executable STOP-1 tuple before submission: the
exact source SHA/snapshot, materialization command and script hash, accepted
external ZIP-manifest identities, fresh output identity, resource tuple, stop
conditions, and interpretation limits must all be recorded in `RUN_REQUEST.md`.
The ceiling is one GH200, 8 CPUs, 96 GiB host memory, 30 minutes, one submission,
and 0.5 GPU-hours. No retry, STOP-2 implementation, merge, or push is authorized.
