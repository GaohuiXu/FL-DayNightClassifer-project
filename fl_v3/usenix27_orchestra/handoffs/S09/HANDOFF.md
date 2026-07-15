# S09 full-pipeline performance and readiness — envelope handoff

## State

```text
SESSION_ID: S09
MILESTONE_STATE: STOP-1/2/3 CLOSED / O-119 STOP-4A-D ACTIVE
BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
FIRST_EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
BRANCH: codex/s08-s09-cl-readiness
S08_POLICY_DECISION: O-110 / CLOSED PASS
S09_SCOPE_DECISION: O-111
STOP2_SCOPE_DECISION: O-114
IMPLEMENTATION_COMMIT: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
IMPLEMENTATION_TREE: d0626e313aab411bc5c71733afb41eca5b102693
IMPLEMENTATION_DIFF_SHA256: cb55d4a46c21f3d508e5d73240367d06080de7b456751d802367b19ed055e7eb
STOP2_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P2
STOP2_REQUEST_REMEDIATION: cad72621e0e3ba409ae19bb0b62829118134b2d0
STOP2_REQUEST_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3
STOP2_COMPUTE_DECISION: O-115 / exact S09-STOP2-SMOKE + recorded O-107 boundary
STOP2_EVIDENCE_SHA: a67cdda56c624d302742f5c57c69bb9ef0a98e0c
STOP2_EVIDENCE_REMEDIATION: 79f87dc9accca700b5a46803d45c549b0305c6d1
STOP2_EVIDENCE_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3
OWNER_STOP2_DECISION: O-116 / ACCEPTED AND CLOSED
STOP3_DECISION: O-117 / UPDATED ENVELOPE + ONE DERIVED IMMUTABLE G100 APPROVED
STOP3_FAILED_CONFIG_RESOLVED_SHA256: cb1723322c756579ab6740eb126de8455b65f808849ec977258c76b919f2c58c
STOP3_FAILED_EXECUTION_SOURCE_SHA: 4d6bd829450021aa0813bcece066fb1fac85f478
STOP3_FAILED_EXECUTION_TREE: affb4854689a0bf65d829a273d769c87c000174c
STOP3_FAILED_SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_4d6bd8294500
STOP3_FAILED_REQUEST_COMMIT: 30e6c9f7849dd1bfe7630f698913c2231131b62c
STOP3_JOB: 441511 FAILED 1:0 in 00:02:29 / editable spconv missing cublasLt.h under wrong run-module bootstrap
STOP3_FAILED_RUNTIME_STATE: tracked source restored / cumm native executable-build identity drifted / re-attestation required
STOP3_CORRECTED_RUNNER_SHA256: 855bbd15877a4ceaa6919ccdf9d2ca369e1f3c84ee306415a41376c07d5d8b5d
STOP3_FAILURE_EVIDENCE_SHA: 4fc78d508d4ac9ad7c46b9d3ad81c87646f8f0d3
STOP3_FAILURE_EVIDENCE_TREE: 56c08110cc4308e424101ae39e7edb79c2769cef
STOP3_FAILURE_REVIEW: PASS_WITH_RESIDUAL_RISK as terminal negative evidence / STOP-3 REMEDIATE pending new owner authority
STOP3_DEP_ATTEST_SOURCE: 788b493889bcf7be98f36b9cbb6686d51e8e5edf / tree 0bc61b3c2693f818ad0feb4e749af64a3947913e
STOP3_DEP_ATTEST_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / NOT compute authorization
STOP3_DEP_ATTEST_SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_dep_attest_788b493889bc
STOP3_DEP_ATTEST_SUBMIT_SHA256: 93848490f485ab38a74ce9818a1ce9d8c35a5eaa17e389fc6b437e9238aa9706
STOP3_O118_DECISION: APPROVED / exact dependency attestation + conditionally derived unchanged O-117 G100 / no retry
OWNER_STOP3_DECISION: O-119 / ACCEPTED AND CLOSED
STOP4_DECISION: O-119 / IMPLEMENTATION + THREE SERIAL CONDITIONAL JOBS / <=2 GPU-HOURS / NO RETRY
APPROVED_COMPUTE: STOP-4A <=00:30:00; STOP-4C <=00:30:00; conditional STOP-4D <=01:00:00 / each 1 GH200, 16 CPU, 96 GiB
STOP4A_INITIAL_IMPLEMENTATION: 5a577062bf0c06faf1f1fa67c209e734569d855e / tree e8b68372cf73b5a96ea03a5c4dcb4cccd3edb477
STOP4A_INITIAL_REVIEW: REMEDIATE / no P0-P1 / three P2 + one P3 / no compute submitted
STOP4A_REMEDIATION: b509f5e527c2dd28d2db506c3f87b5a06b3b1b6a / tree 9c556d37d1e45ece7aad31b10881bb9eb8686424
STOP4A_IMPLEMENTATION_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3
STOP4A_REQUEST_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / pre-submit exact-tuple GO
STOP4A_REQUEST_STATE: consumed by Job 452520 / COMPLETED 0:0 in 00:09:42 / technical PASS / no retry
STOP4A_JOB: 452520 / four cells PASS / 59 focused tests PASS / raw evidence checksum-complete
STOP4B4C_IMPLEMENTATION: 6da4bb5016410708b1e731d26d898f24e6b315ac / closure source 1a0b7e38805d86fb42ff4fe84d67e1680de55015
STOP4B4C_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / implementation-evidence closure GO
STOP4C_REQUEST_STATE: old 131619f SUBMIT NO-GO/never submitted; replacement c776990 Job 455539 COMPLETED 0:0 in 00:04:06 / technical PASS / evidence review pending / no retry
STOP4C_READINESS_SHA256: b8765c4be656fe7ad657157cc43c2c6915ebfc33e6411c26c2a7db829087adff
STOP4C_ARTIFACT_MANIFEST_SHA256: 542862b20a86d30c348237a9b448610857f86cb7554473cfbe65150360593847
STOP4C_EVIDENCE_SHA/TREE: 32b380ccae5dc0146e3c5b494e0f3d4d1ae9d7cd / d35b97f92783f3a953f9402b8966d428e23078a0
STOP4C_EVIDENCE_REMEDIATION: 8b7542c648565508a6b96f6378a0172d255a8b61 / tree b87e583e51d8a09435eea17e547e6871f2bcdb9d
STOP4C_EVIDENCE_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / STOP-4D conditional release GO
STOP4D_IMPLEMENTATION_SHA/TREE: 5642884cdbb16e1c9b3107f529dc70b3a1243c6a / b13a08819b2e203dfe355309f1310c79f94f3023
STOP4D_IMPLEMENTATION_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / exact request freeze GO
STOP4C_RUNNER_REMEDIATION: 72a09d5a503a258f3f257b208180585d16ee49d0 / tree 887d275b71a7f6ccd34cf67188e9fac0843393c1
STOP4C_RUNNER_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / remediation closure GO / replacement freeze complete
STOP3_DEP_ATTEST_JOB: 442152 COMPLETED 0:0 in 00:11:52 / 0.197778 GPU-hours / technical PASS
STOP3_DEP_ATTEST_BUILDS: spconv af42200511a53ce86d77cea0306924a2dc516a74f0483ef7cfe0a6e1dc84b100 / cumm 0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d
STOP3_DEP_ATTEST_ACCEPTANCE_SHA256: 4b60f319660124d3bfac23a21bfbfa1b7c66ca920a0e4a4df03b1a512833e9b4
STOP3_DEP_ATTEST_EVIDENCE_STATE: 82a0e5315c9098056b6670afb490850cc71dc653 / tree 7428f5978c8d423a7c1855d9e3f858eac718aeae
STOP3_DEP_ATTEST_EVIDENCE_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P2 or material semantic concern / Phase-B strict-derivation GO
STOP3_PHASEB_SOURCE: c200bac861a42fc4338973787d3700e28ddd6c7e / tree c0cc4cb8c2e207e42dcc45a129ada28a3d40feb8
STOP3_PHASEB_CONFIG_SHA256: raw 6733a47203bdf7a4da6e39867e6319a7beb9322257e9149f31b7dff6edacf3ce / resolved ba06b72e4c5f1e54f20472e3286a516e7d4328cfb0fccd8bfc7b13095f597ab6
STOP3_PHASEB_SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_c200bac861a4
STOP3_PHASEB_SUBMIT_SHA256: 4801ddfee4cd3c04fbc7215c26ffc25efdafc1e6267599bee39bb87491de309e
STOP3_PHASEB_STATE: Job 446225 COMPLETED 0:0 in 00:05:05 / independently reviewed technical PASS / P3 closed / owner-ready / no retry
STOP3_PHASEB_READINESS_SHA256: 08e376e767f654bb38982127ad5ffd84d94ebaa48b3026ceba2ab7ef93a6c9b6
STOP3_PHASEB_ARTIFACT_MANIFEST_SHA256: b229633889052c46bec5c05d6713e0102aea806a98f9170a65119f9864dbea4b
STOP3_PHASEB_EVIDENCE_SHA/TREE: c28d09c34b0ff56fcbc3805a8361ccd26eeaccc1 / 6c8f008434363dcf41c8f30bdbbaecb4a67863a4
STOP3_PHASEB_REVIEW: evidence c28d09c PASS_WITH_RESIDUAL_RISK / remediation 84adfd0 closure PASS_WITH_RESIDUAL_RISK / no open P0-P3 / owner-ready
JOBS: 441191 COMPLETED 0:0 in 00:03:06; 441293 COMPLETED 0:0 in 00:01:04; 441511 FAILED 1:0 in 00:02:29; 442152 COMPLETED 0:0 in 00:11:52; 446225 COMPLETED 0:0 in 00:05:05; 452520 COMPLETED 0:0 in 00:09:42; 455539 COMPLETED 0:0 in 00:04:06 / no retries
INDEPENDENT_REVIEW: STOP-1 5252a59 PASS_WITH_RESIDUAL_RISK; STOP-2 impl 37aef4d/request cad7262/evidence 79f87dc PASS_WITH_RESIDUAL_RISK; STOP-3 evidence c28d09c no P0-P2 / closure 84adfd0 no open P0-P3 / owner-ready; STOP-4A impl b509f5e and request 6724762 exact-tuple GO / no open P0-P3; STOP-4C evidence 32b380c / remediation 8b7542c PASS_WITH_RESIDUAL_RISK / no open P0-P3 / STOP-4D release GO; STOP-4D impl 5642884 no open P0-P3 / exact request freeze GO
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
linear immutable commits, and independent review. O-115 separately authorized the
exact GH200 smoke; Job `441293` is terminal technical PASS. O-116 accepts/closes
STOP-2. O-117 accepts the detailed STOP-3 plan plus 1 Hz read-only GPU telemetry
and authorizes its exact linear commits and one derived immutable G100 after the
source/snapshot/config/script/output tuple is recorded. At that O-117 boundary,
retry and STOP-4 were unauthorized. Job `441511` consumed the sole O-117 submission and failed before
physical data verification, loader profiling, model construction, or training:
the source-controlled runner selected runtime-only modules despite `env.md`'s
build-module requirement for editable cumm/spconv imports, and spconv JIT could
not find `cublasLt.h`. The runner selector is corrected in the current branch,
but the failed import also changed the cumm native build identity. Independent
failure/remediation review accepted that negative evidence. O-118 then authorized
one exact dependency-attestation job and, conditionally, one strictly derived
unchanged O-117 G100. Job `442152` consumed Phase A and returned technical PASS:
two fresh processes reproduced exact Torch, spconv, cumm, source and config
identities; no data/model/training path ran. The stable aggregate executable-build
identities are spconv `af422005...` and cumm `0a7e3c1a...`. Independent review of
evidence `82a0e53` accepted Phase A with residual risk and no open P0-P2 or
material semantic concern. O-118 now permits the strict Phase-B derivation; its
source `c200bac`, self-contained snapshot, raw/resolved config identities and
read-only submit wrapper are frozen. Independent derivation review returned
`PASS_WITH_RESIDUAL_RISK`, no open P0-P3, and `SUBMIT GO` for that wrapper only.
Job `446225` consumed that exact command at `2026-07-15T11:09:11+02:00`, started
on `n450` one second later, and completed `0:0` in `00:05:05`. It reached 100
successful updates in 103 attempts after three initial scaler overflows, then
accepted all 90 post-warm-up measured windows; all loader, counter, latency,
data-wait, memory and epoch-estimate thresholds pass. Immutable evidence
`c28d09c` received independent `PASS_WITH_RESIDUAL_RISK` with no P0-P2. The two
documentation-only P3 findings—combined-window gate wording and active-ledger
state—are closed by independently reviewed remediation `84adfd0`. O-119 accepts
and closes STOP-3, then authorizes the exact STOP-4A-D sequence recorded below.

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
| STOP-4 — OPTIMIZE/G1000 | O-119 accepts STOP-3 and the exact serial envelope | Bounded baseline profiler; checkpoint-off B=1/2/4 capacity; proven output-neutral remediation; optimized B=1 G100; conditional fresh B=1 G1000 | Independent code/evidence reviews and owner close decision |

At each stop, S00 first presents the exact plan, files and immutable identities,
acceptance/stop criteria, and proposed GPU quota. After one owner approval of that
stop, S00 creates a bounded goal and proceeds continuously through ordinary local
fixes and the approved job. Material changes to data, model output, precision,
recipe, scientific interpretation, cells, seed, or resources stop and return to
the owner. Only a future explicitly opted-in O-009/O-107 STOP-2 smoke may use its
capped mechanical replacement rule; unused time from a STOP-1/3/4 material job
does not authorize a retry or a new cell.

O-119 supplies that amendment after accepting STOP-3. STOP-4 remains limited to
the hash-bound activation-checkpoint switch and measured/source-proven
output-neutral synchronization/allocation removal. B=2/4 are capacity evidence,
while optimized G100/G1000 retain B=1 so performance changes are not conflated
with a training-recipe change.

## O-119 STOP-4 implementation envelope

Initial implementation ownership is deliberately small:

- `config/resolved.py` adds `s09.v2` while preserving the exact `s09.v1`
  semantics; v2 requires explicit camera activation-checkpoint and bounded
  operator-profile fields;
- `training/tasks.py` maps the checkpoint field; `training/loop.py` exposes one
  readiness-only callback advanced exactly once per attempted optimizer window;
- `models/fusion/detector.py` supplies temporary profiler-only ranges for camera
  preprocess/backbone/neck/view transform, LiDAR encoder/backbone, fusion, BEV
  neck and head. The ranges add no application-level activation hooks or retained
  tensors; profiler `record_shapes` may itself temporarily hold tensor references
  inside its three active diagnostic windows, whose memory/latency is excluded
  from capacity and throughput evidence;
- `scripts/centralized_train.py` emits one bounded CPU/CUDA trace/summary and
  excludes its active windows from post-warm-up throughput interpretation;
- four STOP-4A configs freeze exact B=1 profile and checkpoint-off B=1/2/4
  capacity cells; one serial runner executes focused tests and those cells; and
- focused config/mapping/range/callback/profiler tests plus this S09 package are
  updated together.

STOP-4B may touch `models/fusion/losses.py`, `training/loop.py`, and the
centralized entry point only when STOP-4A directly confirms a redundant sync or
allocation. It must keep S08 diagnostic term capture intact. No generic observer,
hook chain, retained activation, worker sweep, architecture, normalization,
head/loss math, target, optimizer, scheduler, EMA, augmentation, sampling,
initialization, metric/decode/NMS, DDP, or checkpoint/resume change is in scope.

The three approved compute ceilings are serial: STOP-4A `00:30:00`, STOP-4C
`00:30:00`, and conditional STOP-4D `01:00:00`, each one GH200/16 CPUs/96 GiB,
with no retry and at most two cumulative GPU-hours. Each exact immutable
source/tree/config/runner/submit/output tuple is recorded and independently
reviewed before submission. A material drift cancels remaining authority.

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

### Implemented `s09.v1` execution contract

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
For STOP-3, O-117 freezes the exact training choice at eight workers, supported by
accepted S01 loader-only evidence; the fresh 0/2/4/8 profile may validate or
falsify that choice but cannot silently derive a different G100 config.

Readiness mode refuses resume, writes no checkpoint, runs no decode/evaluation, and
does not increment a fictitious completed epoch. It stops at the first of the
successful-update target, attempted-window cap, or data exhaustion; it writes the
terminal artifact before returning a nonzero failure for an unmet gate.

### Implemented measurement contract

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

## STOP-4A execution and STOP-4B implementation boundary

O-119 STOP-4A Job `452520` completed all four frozen cells in `00:09:42` with
zero restart or retry. The exact B1 profiler and checkpoint-off B1/B2/B4 cells
all passed numerical, counter, memory, and artifact gates. B2/B4 establish spare
GH200 capacity and higher sample throughput, but remain S10 recipe evidence.
`RESULTS.md` binds every cell, trace, checksum, and interpretation limit.

The profiler plus source audit identified one narrow, production-retainable
optimization. `CenterPointLoss` records three diagnostic scalars per task and the
multi-task wrapper records one aggregate scalar. The training loop already tries
to suppress such terms when telemetry is off, but the wrapper did not expose the
switch. STOP-4B therefore:

- adds one `MultiTaskCenterPointLoss.record_terms` property that propagates to its
  six existing child losses;
- records only the non-synchronizing `n_gt` metadata when disabled;
- lets `train_one_epoch` enable full terms for either declared telemetry or the
  S08 precision observer, and restores the caller's prior setting on exit;
- tests exact loss/output-gradient equality between recording-on/off and tests
  that S08 diagnostics retain terms; and
- adds the exact STOP-4C F-U/B1/checkpoint-off/G100 config and a one-shot runner
  with no worker matrix or operator profiler.

There is no target/loss tensor math, gradient/update, data order, precision,
optimizer, scheduler, EMA, initialization, or recipe change. The trace did not
prove a second safe allocation change; none is attempted. Local `py_compile`,
JSON parsing, `bash -n`, config-core equality, and diff checks pass. The x86 login
Python lacks pytest/Torch, so the immutable GH200 runner owns the focused tests.
An independent implementation/evidence review is mandatory before STOP-4C.

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

At the O-114 planning baseline the smoke could not execute because its exact
immutable implementation SHA, diff, wrapper hash, and fresh output did not yet
exist. Those identities were later frozen and reviewed; O-115 supplied the exact
execution confirmation, and Job `441293` completed without replacement. The design
was not reopened; merge and push remain separately unauthorized.

## STOP-2 immutable implementation and review

The approved implementation is sealed linearly as:

```text
PLANNING_BASELINE: 25a59a699fe88b8cec207d5281d6c3342d2d2db0
INITIAL_IMPLEMENTATION: ff0ffb694255e01a5b109d755ed88fa20b644a78
SYNC_REMEDIATION: 0a11b17
FINAL_REVIEW_CANDIDATE: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
FINAL_TREE: d0626e313aab411bc5c71733afb41eca5b102693
FULL_DIFF_SHA256: cb55d4a46c21f3d508e5d73240367d06080de7b456751d802367b19ed055e7eb
```

Implementation `37aef4d` advances the strict schema to `s09.v1`, preserves the
O-110 precision matrix, adds the hash-bound execution contract, implements the
non-resumable/checkpoint-free/evaluation-free readiness lifecycle, and adds only
direct host/CUDA timing plus the bounded observational loader profile. No model,
loss, target, data backend, optimizer, scheduler, EMA, checkpoint schema, or
precision-policy source changed.

Compile-only validation passed for the eight affected Python files, all six JSON
files parsed, manual positive/negative execution-schema checks passed, the O-110
template/test assertions agree, and `git diff --check` passed. The x86 login node
cannot import the aarch64 Torch environment, so no claim is made that pytest or
CUDA executed locally.

Independent review of `ff0ffb6` returned `REMEDIATE` with no P0/P1: added raw
per-window `GradScaler.get_scale()` calls could synchronize CUDA; one old template
test still expected full sparse FP16; train/eval sampled unused readiness clocks;
and two negative lifecycle behaviors lacked direct tests. Commits `0a11b17` and
`37aef4d` close all four findings. Re-review of the full
`25a59a6..37aef4d` diff returned `PASS_WITH_RESIDUAL_RISK` with no open P0-P2.

The one non-blocking P3 residual is explicit: if every attempted window has a
nonfinite loss, the enabled scaler never enters its finite-loss optimizer path,
so `scaler_scale_at_start` remains JSON `null`; outcomes, counters, and terminal
scale remain complete and the run fails normally. No additional commit was made
for this extreme evidence-only edge case. Job `441293` now supplies the bounded
Torch/CUDA regression evidence; the remaining scope residual is that the loader-
profile unit test does not replace STOP-3's real persistent-worker run.

The exact smoke snapshot, selectors, scripts, hashes, resources, fresh output and
O-107 mechanical derivation boundary are frozen in `RUN_REQUEST.md`. Their
preparation alone granted no compute. Initial request review found two documentation/
contract P2 findings and one stale commit-graph P3; documentation-only remediation
`cad7262` narrowed the toy-test claims, bound the derived O-107 command/output
family, and removed the auxiliary commit-graph. Closure re-review found no open
P0-P3 and returned `PASS_WITH_RESIDUAL_RISK`. O-115 now supplies the one exact
execution confirmation and enables only the recorded O-107 mechanical boundary.
