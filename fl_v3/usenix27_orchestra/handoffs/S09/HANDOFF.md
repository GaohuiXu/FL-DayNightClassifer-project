# S09 full-pipeline performance and readiness — envelope handoff

## State

```text
SESSION_ID: S09
MILESTONE_STATE: STOP-1 CLOSED OWNER-ACCEPTED UNDER O-113 / STOP-2 IMPLEMENTATION ACTIVE UNDER O-114
BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
FIRST_EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
BRANCH: codex/s08-s09-cl-readiness
S08_POLICY_DECISION: O-110 / CLOSED PASS
S09_SCOPE_DECISION: O-111
STOP2_SCOPE_DECISION: O-114
IMPLEMENTATION_COMMIT: none
APPROVED_COMPUTE: STOP-1 Job 441191 consumed / no active compute
JOBS: 441191 COMPLETED 0:0 in 00:03:06 / no retry
INDEPENDENT_REVIEW: 5252a59 PASS_WITH_RESIDUAL_RISK / no open P0-P3
OWNER_STOP1_DECISION: O-113 / ACCEPTED
```

This file records the accepted S09 execution envelope. The owner accepted
`28f7980` as the S08 close/S09 base,
accepted a four-stop S09 direction, assigned engineering performance/readiness to
S09, and deferred scientific training-recipe and sparse-normalization decisions to
S10. O-112 separately started STOP-1 DATA and authorized its exact request/evidence
commits plus one bounded cache-materialization submission. Job `441191` consumed
that submission and is terminal PASS at the execution/evidence level; independent
data/provenance review found no P0/P1 and passed the raw source/cache/job gates,
but returned `REMEDIATE` for one nonexistent request SHA, a stale current-HEAD
label, and active status drift. Documentation-only remediation SHA `5252a59`
corrected those records; bounded re-review closed every P2/P3 finding and returned
`PASS_WITH_RESIDUAL_RISK`. O-113 records the owner's STOP-1 acceptance and permits
exact downstream binding of the reviewed cache identities. It opens STOP-2
detailed planning. O-114 authorizes its exact implementation, local validation,
linear immutable commits, and independent review. Its GH200 smoke and STOP-3/4
still require exact owner-reviewed Slurm authority.

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

## STOP-2 code audit and proposed implementation ownership

The post-STOP-1 source audit found four useful constraints:

1. `s08.v1` already hash-binds global `fp32 | fp16`, the explicit SECOND sparse
   partition, optimizer/recipe, successful-update budget, worker count, exact
   train/val cache identities, accepted manifest identities, dependencies, and
   evaluation policy. STOP-2 must not add another precision selector.
2. `centralized_train.py` currently has only one lifecycle: train to the successful-
   update budget, save one epoch-boundary checkpoint, then run strict official
   evaluation. It cannot honestly terminate a bounded mid-epoch readiness run.
3. `train_one_epoch` already distinguishes attempted/successful/invalid windows and
   already exposes a microbatch `max_steps` cap. With readiness restricted to
   accumulation one, that existing cap exactly enforces the proposed attempted-
   window ceiling; checkpoint state, sampler cursor, and optimizer semantics need
   no redesign.
4. The H2D boundary is the direct `_unpack_batch` call in the production loop, and
   forward, FP32 output promotion, loss, backward, scaler/optimizer, scheduler, and
   EMA are all explicit in the same function. Direct timestamps/events can measure
   them without module hooks, retained tensors, the S08 diagnostic observer, or the
   retired profiler. The current production loader already pins memory and uses
   persistent workers plus prefetch when `num_workers > 0`.

The proposed exact file envelope is:

- `fl_v3/src/fl_v3/config/resolved.py` — advance the current production schema to
  `s09.v1` and require one hash-bound `execution` object;
- `fl_v3/scripts/centralized_train.py` — run either the unchanged train/checkpoint/
  official-eval lifecycle or a fail-closed, non-resumable readiness lifecycle;
  optionally execute the bounded production-loader profile and emit one complete
  readiness artifact;
- `fl_v3/src/fl_v3/training/loop.py` — add only direct, opt-in stage-event records
  and one declared warm-up boundary; normal calls retain their current behavior;
- the six current resolved/template JSON files under `fl_v3/configs/` — mechanical
  `s09.v1` execution fields, plus `sparse_conv_precision="fp32"` for L-S075/F-U/
  F-CBGS so the active templates no longer contradict O-110;
- `fl_v3/tests/test_s06_resolved_config.py` and the S08 resolved-config fixture in
  `fl_v3/tests/test_s08_precision_qualification.py` — mechanical schema evolution;
- one focused `fl_v3/tests/test_s09_readiness.py` for fail-closed lifecycle,
  attempted-window bounds, counter reconciliation, timing neutrality, loader bounds,
  artifact provenance, and CPU/CUDA timing paths; and
- the S09 handoff/request/results/review package plus necessary canonical status
  wording.

No new helper module, benchmark script, wrapper framework, checkpoint schema,
sampler state, or observer API is proposed. `training/tasks.py`,
`training/runtime_state.py`, `training/checkpoint.py`, model/loss files, the data
backend, and the retired `utils/profiling.py` remain unchanged.

### Proposed `s09.v1` execution contract

The new required object is:

```text
execution.mode = train_eval | readiness
execution.max_attempted_windows = non-negative integer
execution.timing_warmup_successful_windows = non-negative integer
execution.loader_profile = null | {
  workers: unique non-negative integer list,
  repeats: positive integer,
  determinism_batches: positive integer,
  warmup_batches: non-negative integer,
  measured_batches: positive integer
}
```

`train_eval` requires zero/zero/null and preserves the current checkpoint and
official-evaluation lifecycle. `readiness` requires world size one, accumulation
one, an attempted-window cap at least as large as the successful-update target,
and a warm-up count below that target. If a loader profile is present, the exact
training `num_workers` must be one of the declared cells. The profile is
observational: it cannot change the already hash-bound training worker count in-job.
For STOP-3, the provisional exact training choice remains eight workers, supported
by accepted S01 loader-only evidence; the fresh 0/2/4/8 profile may validate or
falsify that choice but cannot silently derive a different G100 config.

Readiness mode refuses resume, writes no checkpoint, runs no decode/evaluation, and
does not increment a fictitious completed epoch. It stops at the first of the
successful-update target, attempted-window cap, or data exhaustion; it writes the
terminal artifact before returning a nonzero failure for an unmet gate.

### Proposed measurement contract

The production loop records all bounded attempted windows and their outcomes. On
CUDA it places events directly around H2D, forward plus FP32 output promotion,
loss, backward, and unscale/optimizer/scheduler/EMA, plus H2D-through-update CUDA
end-to-end. It records host time blocked in `next(DataLoader)` separately. There is
no per-stage synchronization: one synchronization establishes the declared warm-up
boundary and one resolves the terminal records. CPU tests use direct host clocks.

The artifact keeps raw bounded records and clearly labels CUDA duration versus host
wait. Summaries report p50/p95 after excluding the first declared successful
windows, measured wall throughput, accepted/attempted ratio, scaler start/end/skips,
all `TrainingState` counters, scheduler/EMA counters, loss, peak allocated/reserved
memory, device capacity/headroom, startup phases, exact resolved-config/data/runtime
identities, and the execution-contract identity. Instrumentation performs no tensor
reductions, hooks, gradient capture, activation retention, RNG draws, or policy
decisions.

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

## O-114 implementation authorization

O-114 accepts:

1. the `s09.v1` schema transition and exact execution fields above;
2. the exact file envelope and direct-event measurement semantics;
3. linear planning/implementation/remediation/evidence commit authority, focused
   local/static validation, and independent reviewer use in this same worktree;
4. the rule that obvious in-envelope code/test defects are fixed continuously,
   while any model/loss/gradient/precision/data/recipe/resource change stops for a
   new owner decision; and
5. the proposed later GH200 smoke shape and O-107 mechanical-remediation ceiling.

The smoke cannot be executed from O-114 alone because its exact
immutable implementation SHA, diff, wrapper hash, and fresh output do not yet
exist. Once they exist, S00 will request one concise exact execution confirmation;
it will not reopen the accepted design unless the implementation materially
deviates. All five points above are authorized for continuous execution; merge and
push remain separately unauthorized.
