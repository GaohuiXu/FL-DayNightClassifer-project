# USENIX Security '27 Orchestra — active envelopes

> **Launch state (2026-07-16).** S08 and S09 are closed and integrated through
> `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`. Their complete pre-compaction
> execution ledgers remain recoverable at that Git object; compact terminal
> records remain under `handoffs/S08` and `handoffs/S09`.
>
> Fresh persistent S00 completed the S10 startup audit at exact clean base
> `a080d49c1c22de20ccb5b1353d4922c7df14a729` and is active on
> `codex/s10-cl-model-recipe`. O-122 accepts the six-stop A-F scientific envelope,
> exact STOP-A split/evaluator gate, and primary full claim “absolute clean
> capability + fusion contribution”. O-124 activates bounded STOP-A/B/C
> completion; O-125's exact final `00:15:00` STOP-A tuple was consumed by Job
> `467862` and timed out inside the exact MILP after tests and metadata traversal.
> O-126 approves a corrected one-shot feasibility protocol and serial A1-A4,
> including exactly one aarch64 CPU-only A-GATE (`0 GPU`, 4 CPU, 32 GiB,
> `00:15:00`, no retry/reroll). Exact Job `468295` was site-transformed into a
> four-GPU request and protection-cancelled before gate execution. O-127 approves
> one explicit-one-GH200/CUDA-hidden replacement; STOP-A A3 is active and
> STOP-B/C are unstarted;
> STOP-D/E/F execution and S11+ remain unapproved.
>
> Canonical decisions: [`ORCHESTRA.md`](ORCHESTRA.md). Milestone contracts:
> [`SESSIONS.md`](SESSIONS.md).

## 1. Rules for starting or extending work

1. A milestone plan is not execution permission. The owner reviews the exact
   scope before S00 begins a newly defined implementation phase.
2. S00 normally stays in the persistent active task/worktree and advances one
   linear branch. `Sxx` is an evidence namespace, not a requirement to create a
   fresh worker or worktree.
3. A bounded planning/research subagent may be used before implementation. It
   does not edit production code or start a parallel implementation chain unless
   the owner explicitly approves that expansion.
4. Independent review starts only after an exact immutable implementation or
   evidence SHA exists. The reviewer does not fix code. Default review may use a
   reviewer subagent; use an independent worktree for high-risk data/split/metric/
   scientific changes, conflicting state, exact runtime reproduction, or owner
   request.
5. O-096 records a platform-maximum reasoning override for the current persistent
   S00 and its bounded pre-S08 planning/research or later review subagents. This is
   a reasoning setting, not implementation or compute authority.
6. Only S00 edits `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`.
   `fl_v3/collab/**` and `fl_v3/docs/cycle_04/**` are read-only historical
   evidence. New records go under `fl_v3/usenix27_orchestra/handoffs/Sxx/`.
7. Before edits, verify repository root, HEAD, branch/ref mode, status, intended
   files, and unrelated dirty state. Do not repair or remove worktrees/branches
   without exact owner authority.
8. Before material compute, create/update `RUN_REQUEST.md` with the exact immutable
   source/snapshot, config/data identities, cells, command, resources, output, stop
   conditions, and approval state. Writing the request is not approval.
9. By default, a changed commit, diff, config, data identity, cell, seed, command,
   resource, or output invalidates approval. There is no automatic retry or
   spare-GPU work.
10. Under O-107, an initial exact O-009 smoke request may explicitly opt into one
    bounded mechanical remediation loop: at most three total submissions and two
    cumulative GPU-hours, with unchanged test objective/selectors, data scope,
    command family, and resource ceiling. Before each derived submission, S00
    records its diagnosis, immutable snapshot/script hashes, and fresh output path.
    Only obvious test/fixture/wrapper/provenance/artifact or output-neutral
    diagnostic-plumbing fixes qualify; identical retries do not.
11. O-009/O-107 do not cover model qualification/training steps, 100/1000 steps,
    full cache/trainval, profile, scientific metrics, matrices, DDP, arrays,
    seeds, or publication. A possible model-output/gradient/update, data,
    precision, optimizer/scheduler/EMA, metric/science, selector/scope, seed, or
    resource change; uncertain classification; repeated blocker; or exhausted cap
    returns to the owner for a new exact decision.
12. Mini evidence is engineering-only. Do not infer convergence, capability,
    mAP/NDS, fusion gain, Protocol A/B readiness, attack, defense, or paper claims.
13. No envelope may import or recover legacy T5/T6/T7, old defense implementations,
    e231, retired O-032-O-091, or old `collab/**`/`cycle_04/**` decisions.

When a separate task/worktree is genuinely needed, fill every field before launch:

```text
SESSION_ID: Sxx or Sxx-R
BASE_SHA: exact 40-character Git SHA
SOURCE_BRANCH: exact approved source
EXPECTED_REF_MODE: detached@BASE_SHA or exact scoped branch
WORKTREE_PROVISIONED_BY: owner / Codex task UI / S00 after explicit approval
FILE_OWNERSHIP: exact paths/globs
UPSTREAM_EVIDENCE_AND_SHAS: exact list
WORKER_SHA: pending for implementation; exact SHA for review
DELIVERY_REF: pending or exact approved ref
REASONING_EFFORT: exact owner-selected setting; current S00 override is max
APPROVED_COMPUTE: none or exact approved RUN_REQUEST
DECISION_SCOPE: exact implementation/evidence/review choices
```

## 2. Persistent S00 startup envelope

```text
You are persistent S00 for the active fl_v3 USENIX Security '27 track. 与 owner
使用中文交互。You own canonical planning and, after owner scope approval, directly
implement tightly connected milestones in the active linear worktree.

Before acting, read completely:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA,SESSIONS,KICKOFFS}.md;
- fl_v3/docs/{env,roadmap/INDEX}.md;
- relevant current handoff/review packages and actual source/diffs/raw artifacts;
- current git status, branch/HEAD, and worktree list.

Do not trust summaries when exact source or artifacts exist. Preserve unrelated
changes and closed negative evidence. Do not recover legacy T5/T6/T7 or old defense
routes.

At the start of a newly scoped milestone, report:
1. current anchor and status;
2. exact question and evidence gap;
3. proposed files/semantics and scientific non-goals;
4. local verification and independent-review plan;
5. any proposed compute tuple, still unapproved unless explicitly granted.

Do not submit compute, commit, merge, push, upload, change a locked scientific
choice, or create another task/worktree until the owner has granted the exact
needed authority. Persistent S00 simplifies context; it does not broaden scope.
```

## 3. S08 envelope v1 — closed PASS under O-110

```text
SESSION_ID: S08
CURRENT_CODE_ANCHOR: 2a584053e6f6a3860b6f812681dc8d7342ca52ad
REBASELINE_SHA: 2a584053e6f6a3860b6f812681dc8d7342ca52ad
IMPLEMENTATION_CONTEXT: persistent S00, linear active worktree
DELIVERY_BRANCH: codex/s08-s09-cl-readiness
PRE_IMPLEMENTATION_AUDIT: handoffs/S08/MODEL_RECIPE_AUDIT.md; accepted as planning input
OWNER_DECISION: O-097 direction/branch/audit baseline; O-098 detailed plan/implementation/local validation/post-validation commit; O-099 consumed exact S08-SMOKE-1; O-100 provenance remediation/request preparation; O-101 consumed exact S08-SMOKE-2; O-102 narrow diagnostics/test remediation and SMOKE-3 request preparation only; O-103 consumed exact S08-SMOKE-3 PASS; O-104 review remediation/local validation/SMOKE-4 request preparation only; O-105 consumed exact S08-SMOKE-4 terminal test-construction FAIL; O-106 consumed exact post-freeze S08-SMOKE-5 PASS; O-107 prospective bounded mechanical remediation loop; O-108 remediation/evidence commit and independent re-review; O-109 Q1/Q2 completion goal, commits, and cumulative two-GPU-hour Slurm authority; O-110 accepted policy/S08 PASS/closure/fast-forward-only integration/S09 discussion gate
FILE_OWNERSHIP:
- fl_v3/src/fl_v3/training/tasks.py
- fl_v3/src/fl_v3/training/{loop,runtime_state,precision_diagnostics}.py
- fl_v3/src/fl_v3/models/fusion/{detector,sparse_voxel_encoder,second_sparse_backbone,losses}.py only for explicit partition/bounded diagnostic seams
- fl_v3/src/fl_v3/config/{__init__,resolved}.py and fl_v3/src/fl_v3/utils/runtime.py for fail-closed regime provenance
- fl_v3/configs/{s06_synthetic_camera,s07_b_c_str8,s07_b_l_p020,s07_b_l_s075,s07_b_f_u,s07_b_f_cbgs}.json
- focused S08 partition, diagnostic, and Q1 evidence tests plus minimal smoke scripts; no generic audit/profiler framework
- fl_v3/usenix27_orchestra/handoffs/S08/{HANDOFF,RUN_REQUEST,RESULTS,REVIEW}.md as applicable
READ_ONLY:
- model architecture, task groups, targets/loss equations, decode/NMS, metric, optimizer recipe
- S01 data/cache code except normal consumption
- fl_v3/collab/**, fl_v3/docs/cycle_04/**, fl_v2/**
UPSTREAM_EVIDENCE:
- S07-B-COMPLETE candidate c615b6471a04b91a09c6ac6d487ff39a1501ceee
- D1 Job 389356 and its exact raw artifacts
- F1 Job 390576 and independent review package 7f3bd40158e5a8af30196509734782c4575c50aa
- historical old-model AMP Jobs 211502/211722, interpretation-limited
REASONING_EFFORT: max under owner override O-096
APPROVED_COMPUTE: S08-SMOKE-1 Job 426619, S08-SMOKE-2 Job 427800, S08-SMOKE-3 Job 428112, S08-SMOKE-4 Job 428889, S08-SMOKE-5 Job 429080, Q1 Job 431013, and Q2 Job 435151 all consumed/terminal; Q1+Q2 used 00:07:58 of O-109's two-GPU-hour cap
DECISION_SCOPE: implement approved explicit partition and bounded diagnostics; architecture/normalization/recipe/scientific policy remain owner-gated
```

### Objective

Determine which existing precision partition can execute stable accepted optimizer
windows for the current six-task C/L/F model, and localize why L-S075/F-U overflow
under current sparse-conv FP16.

### Required implementation behavior

1. Replace implicit production behavior
   `second_075 + precision=fp16 => sparse_conv_fp16=true` with an explicit,
   fail-closed precision-partition field recorded in resolved config/provenance.
2. Preserve three qualification regimes:
   - uniform `fp32` reference;
   - current full FP16 AMP, including sparse-conv FP16;
   - FP16 AMP with SECOND voxelization/VFE/spconv executed in FP32.
3. Preserve direct sparse BF16 rejection and the reviewed spconv 2.3.8
   no-grad-evaluation workaround.
4. Use the production training loop and persistent dynamic GradScaler. A bounded
   diagnostic must permit skip/backoff/continuation to a successful update or a
   predeclared failure bound; it must not treat one fixed scale as the contract.
5. Use minimal opt-in window-end diagnostics before gradients clear: per-task losses,
   scaler before/after, skipped/executed updates, exposure/scheduler/EMA counters,
   nonfinite element counts, first bad parameter, max finite value, stable norm,
   head-input gradient, and SECOND stem/stage boundary gradients.
6. Diagnostics are opt-in and output-neutral when disabled. Do not build a generic
   observer, long-lived module-hook chain, profiler, or synchronized per-layer
   telemetry system. Parameter-gradient summaries should read `.grad` at the
   window boundary; any named tensor tap must be narrow and lifecycle-safe.
7. Cover C-STR8/L-S075/F-U as primary qualification. Add short L-P020/F-CBGS
   compatibility coverage without turning it into a scientific matrix.

### Non-goals and stop conditions

- Do not change task groups, head/loss equations, architecture, optimizer recipe,
  metric, data ownership, or scientific thresholds.
- Do not run full trainval, 100/1000 steps, performance profile, mAP/NDS, DDP,
  multi-seed, Protocol A/B, attack, or defense.
- If existing regimes cannot produce bounded finite C/L/F updates, stop with
  localized evidence and return a remediation decision; do not redesign the head
  or silently lower a gate.
- Any source/config/test/command change after a compute request is reviewed
  invalidates that request.

### Local acceptance before requesting compute

- exact config-resolution tests for all three regimes and invalid combinations;
- unit tests that dynamic-scaler skips do not advance optimizer/scheduler/EMA/
  exposure and later accepted windows do;
- focused telemetry/finiteness tests on dependency-available fixtures;
- Python compile, JSON/TOML parse, shell syntax if touched, `git diff --check`;
- a compact HANDOFF and exact list of GH200-only items still NOT RUN.

The first exact bounded request was consumed by Job `426619` and failed before
pytest on the provenance clean-checkout gate. Preserve it as negative evidence.
After an owner-approved remediation and renewed local validation, freeze a new
snapshot/request with exact selectors, hashes, resources, output, and stop
conditions; do not submit it without a new explicit owner approval.

## 4. S08 independent-review envelope

Use this only after an immutable S08 implementation/evidence SHA exists.

```text
You are the independent S08 reviewer. Do not implement fixes.

Read repository AGENTS.md, the three canonical Orchestra documents, the complete
S08 handoff/request/results package, exact implementation diff, D1/F1 raw evidence,
relevant training/config/sparse code, and the historical 211502/211722 evidence
with its old-model limitation.

Audit findings first:
- whether every precision regime resolves explicitly and fail-closed;
- whether full AMP and spconv-FP32-island semantics match their names;
- whether direct BF16 rejection and S04 eval behavior are preserved;
- whether dynamic GradScaler continuation, optimizer/scheduler/EMA/exposure
  counters and resume state are correct through skips;
- whether elementwise gradients, stable norms, task losses, head-input and SECOND
  boundary localization distinguish overflow from telemetry artifacts;
- whether C/L/F and bounded L-P020/F-CBGS coverage match the approved request;
- whether any architecture/head/loss/metric/optimizer change escaped scope;
- whether failed/missing cells or mini-only limitations are visible.

Return P0-P3 findings with exact paths/lines/artifacts, adversarial checks, a
PASS|PASS_WITH_RESIDUAL_RISK|REMEDIATE verdict, and residual risk. Do not edit source,
submit compute, broaden the matrix, or infer convergence/performance/science.
```

Default: reviewer subagent from the exact immutable SHA. Use a separate review
worktree if the implementation state is not clean/immutable, runtime reproduction
is required, or the owner requests it.

## 5. S09 envelope v1 — four owner stops

```text
SESSION_ID: S09
BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
SOURCE_BRANCH: codex/s08-s09-cl-readiness
IMPLEMENTATION_CONTEXT: persistent S00 unless owner selects independent isolation
STATE: S09 CLOSED PASS UNDER O-120 / ACCEPTED REVIEW SEAL ced5992 / FF-ONLY INTEGRATED AT 351b7a0 UNDER O-121
APPROVED_COMPUTE: fully consumed; O-119 actual 0.345000 GPU-hours / no prospective S09 compute
APPROVED_GIT: linear S09 envelope/request/evidence/review commits; no merge/push
DECISION_SCOPE: base-uniform full-pipeline engineering performance/readiness only
```

### Binding scientific and engineering boundary

- Precision is frozen by O-110: camera/dense-pillar uses global FP16; sparse
  SECOND LiDAR/fusion uses global FP16 with voxelization/VFE/spconv/dense-collapse/
  to-BEV in FP32; uniform FP32 remains reference/fallback.
- S08 dynamic scaling did not shrink the true unscaled LiDAR gradient. It
  localized sparse-FP16 overflow to SECOND stem weight-gradient dynamic range but
  did not prove why the FP32 gradient is unusually large. Tiny-group sparse
  GroupNorm remains a leading hypothesis, not a finding.
- S09 monitors scaler skips, nonfinite windows, accepted-step stability, timing,
  memory and data wait. It must not change normalization, head/loss/targets,
  gradient clipping, optimizer/scheduler/EMA, augmentation, sampling,
  initialization, official metric/decode/NMS, or branch architecture.
- Scientific training-recipe and branch selection are S10 work. S09 may close
  only as a labelled base-uniform engineering-readiness milestone.
- Measurement-backed, output-neutral optimization of ZIP/cache access, loader
  lifecycle, H2D transfer, redundant conversion/sync/allocation, and bounded
  logging/checkpoint overhead is in S09 scope. It must preserve data order and
  contents, model/loss/gradient/update semantics, O-110 precision, and exposure
  accounting, and it needs an exact owner-reviewed file/equivalence envelope.
- No S08 precision diagnostics/window observer is enabled in performance jobs.
  STOP-3/4C/4D timing uses direct loop timestamps/CUDA events. O-119 permits only
  one bounded STOP-4A `torch.profiler` cycle with temporary named ranges and no
  application-level activation hooks. `record_shapes` may temporarily retain
  tensor references inside the three active diagnostic windows, so their memory/
  latency is not capacity/throughput evidence; this does not restore a general
  profiler/harness chain.

### Stop workflow

At each stop, S00 presents one exact plan containing file ownership, local checks,
commit authority, immutable source/config/data/command/output, one-GPU resource
quota and stop conditions. After the owner approves that exact stop, S00 creates a
concrete goal and works continuously until the stop is complete or a material
boundary is hit. Obvious local defects may be fixed continuously within the
approved semantics. Only an explicitly approved STOP-2 O-009/O-107 smoke may use
derived mechanical replacement submissions. STOP-1/3/4 are material jobs: their
one exact submission has no retry/resubmission authority merely because quota
remains.

#### STOP-1 DATA — production `t1.v2` bind

- Rebind the existing cache launcher/builder to an immutable current source.
- Reuse, but re-verify, accepted ZIP manifest logical SHA-256
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`
  and file SHA-256
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.
- Materialize train/val `t1.v2`, `n_sweeps=10`; require train
  `28130/944881` samples/boxes and val `6019/187528`, sidecar equality,
  per-record depth, canonical logical hashes, physical SHA-256s and fresh output.
- No sensor extraction, payload-wide scan, model, loader sweep, metric or profile.
- Proposed later resource ceiling: one GH200/aarch64 environment, eight CPUs,
  96 GiB host memory, `00:30:00`, one submission/0.5 GPU-hour. **Approved only
  for STOP-1 under O-112; exact tuple must be frozen before submission.**
- Independent data-identity review and owner STOP-1 inspection precede STOP-2/3
  production binding.

#### STOP-2 IMPLEMENTATION — minimal readiness mode

Proposed file ownership, to be frozen before implementation:

- `fl_v3/src/fl_v3/config/resolved.py`: add a fail-closed, hash-bound S09
  readiness execution contract while preserving accepted S08 configs;
- `fl_v3/scripts/centralized_train.py`: explicit readiness-only termination after
  the update budget; refuse readiness resume; do not save a misleading resumable
  mid-epoch checkpoint or invoke official evaluation;
- `fl_v3/src/fl_v3/training/loop.py` plus at most one small timing helper: direct
  data-wait/H2D/forward/loss/backward/optimizer/end-to-end CUDA-event accounting,
  warm-up separation, percentile summary, peak allocated/reserved memory, and
  existing scaler/exposure counters;
- the three SECOND templates: replace stale sparse-FP16 placeholders with the
  accepted sparse-FP32 island while retaining template-only status;
- focused S09 config/runner/timing/output-neutral tests and S09 handoff records.

No checkpoint schema/sampler-cursor system, generic observer, profiler harness,
metric, model, data, loss or recipe change was in scope. Local validation preceded
the immutable implementation commits. The planning-stage eight-CPU/96-GiB/
`00:30:00` smoke shape was superseded before execution by the independently
reviewed O-115 tuple: one GH200, four CPUs, 32 GiB, `00:10:00`. Job `441293`
consumed that initial submission, passed 44/44 tests in `00:01:04`, and used no
O-107 replacement.

#### STOP-3 G100 — worker sweep plus 100 accepted F-U updates

- Exact data: accepted STOP-1 train `t1.v2`; val is identity-bound but not read.
- Exact model cell: only F-U, random initialization with one frozen engineering
  seed, uniform sampling, AdamW `1e-4/0.01`, constant scheduler, EMA/clip/3D-BEV
  aug/GT-paste off, microbatch 1, accumulation 1, world size 1, O-110 FP32 island.
- Loader-only cells: `num_workers=0/2/4/8`, fixed token order, two persistent
  repeats each with 32 digest, 16 warm-up and 256 measured batches. The full model run uses
  a worker count frozen before submission (provisional recommendation: 8); no
  dynamic in-job model-config selection.
- Model gate: 100 successful optimizer steps, at most 120 attempted windows; first
  ten successful steps excluded from steady-state timing. Record p50/p95, samples/s,
  epoch estimate, stage timing, data-wait share, memory, scale/skip/nonfinite and
  exact optimizer/scheduler/EMA/exposure accounting.
- The one-shot runner additionally samples aggregate GPU utilization, memory
  utilization, memory use, power, clocks and temperature at 1 Hz. This is
  observational telemetry, not a module/kernel/Tensor-Core profiler and cannot
  attribute backward time to camera/LiDAR/fusion subgraphs.
- O-117 thresholds: all loader digests equal; worker-8 warm throughput at least
  90% of the best warm cell; exactly 100 successful updates within 120 attempts;
  zero nonfinite/discarded windows or counter drift; warm-up-separated accepted
  ratio at least 95%; integrated `(data_wait + CUDA H2D-through-update)` p95/p50
  at most 1.5; measured data-wait share at most 10%; peak reserved memory at most
  86 GiB; and both frozen epoch estimates at most 24 h. Aggregate loss must be
  finite, but monotonic convergence is not a STOP-3 gate or claim.
- One GH200, 16 CPUs, 96 GiB host memory, no DDP, no retry, no official evaluation
  or metrics. O-117 approves one submission and `<=01:00:00` only after the exact
  immutable source/snapshot/config/script/output tuple is recorded.
- Job `441511` consumed that sole submission and failed `1:0` in `00:02:29`
  before physical data verification, loader profiling, model construction, or
  training. The runner used `arrhenius_load_modules run` despite `env.md`'s
  editable-spconv requirement for build modules; missing `cublasLt.h` stopped
  JIT compilation. The current branch fixes that selector, but cumm native build
  identity also drifted. At the O-117 boundary, no retry was authorized and a new
  owner amendment was required for runtime re-attestation or replacement; O-118
  subsequently supplied only the bounded authority recorded in the next item.
- Independent failure review preserved Job `441511` as negative pre-model
  evidence. O-118 then authorized exactly one bounded dependency attestation and,
  only after its independent PASS, one strictly derived unchanged G100. Job
  `442152` returned stable spconv/cumm build identities; Job `446225` completed the
  exact loader sweep and 100 successful F-U updates in 103 attempts with all
  O-117 gates passing. There was no retry. Evidence `c28d09c` received independent
  `PASS_WITH_RESIDUAL_RISK` with no P0-P2; closure re-review of remediation
  `84adfd0` found no open P0-P3 and marks STOP-3 owner-ready. This does not
  authorize STOP-4.
- An independent reviewer reads exact source/config/cache/records/artifacts before
  the owner STOP-3 decision.
- If a throughput/stability threshold fails, STOP-3 may return a narrowly measured
  output-neutral optimization proposal. It does not silently change source or
  resubmit G100; a replacement implementation and exact G100 tuple require an
  owner amendment at this stop before STOP-4 can start.

#### STOP-4 OPTIMIZE/G1000/CLOSE — approved by O-119

- STOP-4A: one serial `00:30:00` job runs focused tests; one exact STOP-3-like
  B=1/checkpoint-on 20-update operator profile; and checkpoint-off B=1/2/4
  20-update capacity cells. Only the B=1 profile is a baseline diagnostic;
  B=2/4 are capacity evidence and cannot select the S10 recipe.
- STOP-4B: use the trace and source audit to remove only proven redundant
  synchronization/allocation, including output-neutral loss telemetry recording;
  keep a hash-bound Swin checkpoint switch. Preserve model outputs, loss,
  gradients/updates, data order, O-110 precision, optimizer/scheduler/EMA and
  exposure. Seal and independently review immutable code/evidence.
- STOP-4C: one optimized B=1 G100 under the unchanged base-uniform recipe,
  `00:30:00` ceiling. It must pass the STOP-3 numerical/counter gates and must not
  regress steady latency materially; no profiler is active.
- STOP-4D: only after reviewed STOP-4C PASS, one fresh-from-initialization B=1
  1000-successful-step job under `01:00:00`; it does not resume G100. Independently
  review all S09 evidence and prepare owner-close-ready state.
- All three jobs use one GH200, 16 CPUs and 96 GiB; they are serial, no-retry and
  capped at two cumulative GPU-hours. Any model/recipe/data/precision/resource or
  scientific-scope drift cancels the remaining conditional authority. No worker
  matrix, DDP, merge, or push.
- Terminal outcome: Jobs `452520`, `455539`, and `456539` completed `0:0` without
  retry, using `0.345000` GPU-hours. The quiet path removes exactly 19 proven
  redundant loss-term scalar synchronizations per ordinary attempted window;
  S08 diagnostics retain terms, and exact loss/input-gradient equality is tested.
  Swin checkpoint-off is explicit. G1000 reached 1000 accepted updates in 1003
  attempts with p50/p95 `178.024/203.231 ms`, throughput `5.542 samples/s`, and
  peak reserved `8.314 GiB`. Final independent review found no open P0-P3 and
  made S09 owner-ready. O-120 subsequently accepts review seal `ced5992` and
  closes S09 PASS. O-121 later completes ff-only integration at `351b7a0`
  without authorizing S10 execution.

### Explicit non-goals

Do not copy old profiler/audit wrappers or use mini throughput to freeze production
settings. No mAP/NDS, per-class capability, fusion gain, convergence/scientific
recipe claim, multi-seed, Protocol A/B, attack, defense, DDP, full-data payload
scan, branch selection, normalization experiment, or publication/upload is
authorized.

## 6. S10 active launch state

- **Persistent S00:** remain in this task/worktree and on the linear
  `codex/s10-cl-model-recipe` branch. Do not create per-stop implementation tasks,
  parallel production implementers, micro-handoffs or per-cell review chains.
- **Accepted scope:** six stops A split/evaluator, B observation-first, C
  architecture/init, D recipe freeze, E final-graph GH200 optimization, F
  single-seed full/official-val close. Exact scientific limits are in
  `handoffs/S10/HANDOFF.md`.
- **Current authority:** O-127 approves one explicit-one-GH200 replacement with
  `CUDA_VISIBLE_DEVICES=""`, PyTorch CUDA count zero, 4 CPUs, 32 GiB and 15
  minutes. It preserves O-126 science and permits minimal runner/docs commit,
  one immutable submission, evidence and A4 review. No retry or B/C execution;
  B/C cannot start before reviewed A PASS.
- **O-123 batch correction:** the B=1-based v0 request is rejected. Revised ABC
  scientific rungs use physical B=4 at minimum and bind a fixed-batch tail policy;
  B=1 may appear only in a tiny paired diagnostic check. B=8/16 belong to a later
  bounded STOP-D/E batch ladder after final-graph selection unless the owner
  explicitly amends ABC.
- **ABC completion authority:** O-124 covers in-envelope linear
  implementation/evidence/remediation commits, at most one active GH200 job,
  exact pre-recorded serial submissions, and stop-level review within its caps.
  It is not O-107 and never transfers unused budget to extra candidates, seeds,
  horizons or STOP-D/E/F. O-125's unused contingency is likewise not a spare-job
  entitlement.
- **Review:** STOP-A split/metric requires one isolated high-risk review worktree
  after a successful immutable evidence SHA; incomplete Job `467862` does not
  trigger a closure review. Reuse one bounded S10-R context when later isolation
  is required. Reviewers read and report but never fix. S00 remediates linearly;
  P3-only polish is batched, and a repeated material blocker returns to the owner.
- **S11 and later:** pending. Historical role descriptions and sequencing do not
  create scope or authority; Protocol A/B execution, attack, defense, upload and
  publication remain unauthorized.

## 7. Exact compute-request skeleton

```text
SESSION_ID:
REQUEST_ID:
REQUEST_STATE: DRAFT / NOT APPROVED
SOURCE_SHA:
SNAPSHOT_PATH_AND_SHA256:
RESOLVED_CONFIG_SHA256:
DATASET_CACHE_MANIFEST_IDENTITIES:
CELLS_AND_ORDER:
SEEDS:
COMMAND_AND_SHA256:
RESOURCES:
OUTPUT_ROOT:
STOP_CONDITIONS:
ALLOWED_INTERPRETATION:
FORBIDDEN_INTERPRETATION:
RETRY: none unless separately approved
OWNER_APPROVAL: pending exact tuple
```

A request is executable only after the owner approves that exact tuple and S00
records the approval without changing it.
