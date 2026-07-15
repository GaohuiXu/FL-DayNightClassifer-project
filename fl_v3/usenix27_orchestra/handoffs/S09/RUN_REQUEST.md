# S09 RUN_REQUEST — four-stop execution ledger

> **Ledger state:** O-112 authorizes STOP-1 only. Its exact immutable tuple is
> being frozen below and must be complete before the sole submission.

## Authorization state

```text
SESSION_ID: S09
BASE_AND_CURRENT_HEAD: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
BRANCH: codex/s08-s09-cl-readiness
OWNER_DIRECTION: O-111 envelope + O-112 STOP-1 execution
APPROVED_COMPUTE: STOP-1 only / <=0.5 GPU-hours
APPROVED_SUBMISSIONS: 1 / not consumed
ACTIVE_REQUEST: S09-STOP1-DATA / exact tuple freeze in progress
IMPLEMENTATION_COMMIT_AUTHORITY: no production implementation; linear STOP-1 docs/evidence commits allowed
MERGE_OR_PUSH_AUTHORITY: none
```

O-111 approved preparation and review of the envelope. O-112 additionally starts
STOP-1 and authorizes its one bounded materialization job after this ledger records
the complete immutable tuple. Each later stop
will receive a frozen request block with exact immutable source/snapshot, resolved
config hash, dataset/cache/manifest identities, cells/order, sample/window/step
bounds, seed, command/script hashes, resources, output root, stop conditions, and
allowed/forbidden interpretation before the owner is asked for approval.

After that one exact approval, S00 creates the stop goal and executes continuously
within the frozen boundary. An altered material tuple requires a new owner
decision. No identical retry, spare-node/GPU expansion, unused-quota transfer, or
conditional next stop is implicit.

## STOP-1 — production `t1.v2` cache identity

```text
REQUEST_STATE: APPROVED UNDER O-112 / EXACT TUPLE FREEZE IN PROGRESS / NOT SUBMITTED
OBJECTIVE: materialize and attest exact train/val t1.v2 caches for n_sweeps=10
MODEL_OR_TRAINING: none
PROPOSED_RESOURCE_CEILING: 1 GH200 / 8 CPU / 96 GiB host / 00:30:00 / 0.5 GPU-hours
PROPOSED_SUBMISSIONS: 1
```

Known external manifest evidence to reverify before freeze:

```text
ZIP_MANIFEST_LOGICAL_SHA256: 023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6
ZIP_MANIFEST_FILE_SHA256: 228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb
EXPECTED_TRAIN_SAMPLES/BOXES: 28130 / 944881
EXPECTED_VAL_SAMPLES/BOXES: 6019 / 187528
N_SWEEPS: 10
```

The final request must bind source/snapshot and tree identity, the exact
`build_nuscenes_cache.py` command and script hash, module/dataroot and manifest
paths, destination paths that are absent at submission, train/val cache metadata
and content-hash acceptance, and artifact checksums. It may traverse the metadata
and declared ZIP members required to create the cache, but must not extract the
dataset, scan unrelated payload contents, construct a model, profile performance,
or start training. Independent review must accept the resulting identities before
STOP-2 can bind them.

## STOP-2 — minimal readiness instrumentation

```text
REQUEST_STATE: PROPOSED / DEPENDS ON STOP-1 / NOT APPROVED
OBJECTIVE: implement and validate output-neutral performance/readiness accounting
PROPOSED_RUNTIME_SMOKE_CEILING: 1 GH200 / 8 CPU / 96 GiB host / 00:30:00 / 0.5 GPU-hours
PROPOSED_SUBMISSIONS: 1, or at most 3 only if the future exact request explicitly opts into O-107
```

The future implementation request will bind the file envelope in `HANDOFF.md`,
tests, local validation, immutable implementation-commit authority, and reviewer
scope. Any GH200 smoke must be a focused implementation/runtime check within
O-009, not a model-qualification or production training run. If the owner
explicitly opts into O-107, derived replacements are limited to obvious test,
fixture, wrapper, provenance/artifact, or output-neutral diagnostic-plumbing
defects and remain inside O-009's cumulative two-GPU-hour ceiling. Model output,
loss/gradient, precision, data, recipe, cell, seed, or resource changes end the
loop.

The accepted instrumentation must use direct bounded timestamps/CUDA events and
memory counters only. The S08 precision observer, forward/backward hooks,
activation retention, general profiler, sampler/checkpoint redesign, and retired
harnesses are forbidden.

## STOP-3 — loader selection and G100

```text
REQUEST_STATE: PROPOSED / DEPENDS ON REVIEWED STOP-2 / NOT APPROVED
PRIMARY_CELL: F-U only
PRECISION: global FP16 autocast + explicit SECOND FP32 island
INITIALIZATION: random / exact engineering seed to be frozen
OPTIMIZER: AdamW lr=1e-4 weight_decay=0.01
SCHEDULER: constant
EMA/GRAD_CLIP/BEV_AUG/GT_PASTE: disabled
MICROBATCH/ACCUMULATION/GPU: 1 / 1 / 1 GH200
PROPOSED_RESOURCE_CEILING: 1 GH200 / <= 01:00:00 / 1 GPU-hour
PROPOSED_SUBMISSIONS: 1
```

The exact future request will bind:

1. loader-only production ZIP/cache cells at `num_workers=0/2/4/8`, two persistent
   repeats per worker count, with 16 warm-up batches and 256 measured batches per
   repeat;
2. a single worker count selected by a predeclared rule and frozen before model
   training begins (8 is only the current provisional default);
3. one fresh F-U run requiring 100 successful optimizer updates, stopping after at
   most 120 attempted windows, with the first ten successful updates excluded
   from timing summaries; and
4. stage/end-to-end p50/p95, throughput, data wait, peak allocated/reserved
   memory, optimizer/scheduler/exposure accounting, scaler/nonfinite status, and
   a bounded epoch-time estimate.

Candidate acceptance criteria, to be finalized before approval, are: zero
nonfinite accepted windows; no optimizer/scheduler/exposure drift; at least 95%
accepted-window ratio after the declared warm-up; p95/p50 end-to-end ratio no
greater than 1.5; peak reserved memory no greater than 86 GiB; complete artifacts;
and a provisional estimated epoch ceiling of 24 hours. The denominator and exact
handling of data exhaustion/restarts will be explicit in the frozen request.

This gate may report the engineering effect of the unresolved large LiDAR
gradients. It may not change normalization, model/head/loss/target, clipping, or
the base-uniform recipe in response. Such a blocker returns to the owner rather
than being silently optimized away.

If measured performance instead exposes a specific output-neutral bottleneck in
ZIP/cache access, loader lifecycle, H2D transfer, redundant conversion/sync/
allocation, or bounded logging/checkpoint overhead, STOP-3 reports an exact
candidate and equivalence plan. It does not change source or submit another G100
under unused quota. A replacement requires an owner amendment binding the files,
semantics, immutable source/config/command, and resource tuple; STOP-4 remains
blocked until that gate is independently accepted.

## STOP-4 — conditional G1000

```text
REQUEST_STATE: CONDITIONAL / DEPENDS ON INDEPENDENT STOP-3 PASS AND NEW OWNER APPROVAL
OBJECTIVE: one fresh 1000-successful-step readiness run
WALLTIME_AND_GPU_QUOTA: unset
SUBMISSIONS: unset / no automatic continuation
```

STOP-4 starts from a fresh initialization and is not a mid-epoch resume of
STOP-3. It must freeze the accepted cache, precision policy, base-uniform recipe,
worker count, seed, counters, thresholds, source/config/script hashes, resources,
and output path from the reviewed STOP-3 decision. It remains a single-GH200
engineering readiness run unless measured STOP-3 evidence motivates a separate
owner-approved DDP amendment. It requires independent evidence review before S09
can close.

## Interpretation limits shared by all stops

No S09 stop establishes mAP/NDS, convergence, fusion gain, scientific training-
recipe optimality, a normalization fix, Protocol A/B validity, FL capability,
attack viability, or defense efficacy. Mini or bounded readiness evidence cannot
support those claims. Failed, skipped, or omitted cells and all attempted windows
must remain visible in `RESULTS.md`; a gate may not be weakened retroactively.
