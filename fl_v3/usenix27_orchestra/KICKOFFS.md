# USENIX Security '27 Orchestra — active envelopes

> **Launch state (2026-07-22).** S08/S09 are closed. S10 Phase I-P is active on
> `codex/s10-phase1p-throughput-preflight`, created from the frozen
> `codex/s10-phase1-branch-qualification` control at `f1a2bab...`, and remains
> governed by O-143.
>
> Active order: **C/L independent recipe and capability → staged fusion →
> aligned capability/fusion gate → profiler/optimization only after pass**.
> Current-A2 and the former C→D→E→F path are paused. STOP-A's split/evaluator is
> reusable; prior B/C diagnostics remain evidence but do not establish production
> capability.
>
> O-143 also replaces S10's per-job immutable/no-retry/multi-document/reviewer
> mechanics with the phase-level workflow below. This kickoff records no compute
> authority.
>
> O-144 closes `P1-G0 PLAN_FREEZE`; all Phase-I work must follow
> `handoffs/S10/PHASE_I_PLAN.md`. O-145 adds the optimized CUDA BEV-pooling
> implementation/parity/timing contract to WP2/WP4. O-146 activated the exact
> Envelope A at `e321aed749fd859c809199d52c30b2771dbef8b3`; O-147 amended it and
> O-148 replaced its submission stop with serial completion inside the unchanged
> `1.10` GH200-hour ceiling. WP0-WP4 is terminal: Camera is negative at the frozen
> pooling-promotion gate and LiDAR engineering qualification passed. O-149 now
> governs future explicitly approved engineering-validation loops; it creates no
> standing compute authority. O-150 accepts the parity-qualified PyTorch sorted
> `segment_reduce` backend for Camera production, keeps CUDA unpromoted, and removes
> `1.25x` as a capability prerequisite. Phase I-P promoted final Camera two-GH200
> B16/rank and LiDAR one-GH200 B32 recipes. Revised Envelope B was accepted at seal
> `1473ef67...`; activated LiDAR then stopped honestly after four healthy epochs at
> a reproducible all-nonfinite epoch-5 TransFusion forward. The owner cancelled the
> serial L->C dependency. `RUN_REQUEST.md` Section 7.4.7 materializes a pending
> parallel amendment under the unchanged `30.0` charged-GH200-hour ceiling: one
> Camera job may overlap one zero-update LiDAR epoch-4 diagnostic (maximum two jobs/
> three typed GH200s). Its additional non-selectable `D_select` diagnostic peek was
> reviewed at `296ef9b...`: `PASS_WITH_RESIDUAL_RISK`, no open P0-P2. Explicit owner
> activation of the review-seal commit remains required; no job is currently active.

## 1. Rules for starting or extending work

1. The binding plan is not execution authority. Before work in an S10 envelope,
   the owner approves its allowed implementation/external/data actions, objective,
   candidate/data/metric/seed boundaries where applicable, aggregate GPU-hours,
   submission policy/concurrency, outputs and escalation conditions.
2. S00 remains in the persistent worktree and linear branch. Do not create
   per-stop implementation tasks, micro-handoffs, snapshot trees or reviewer
   chains.
3. Preflight with direct entry/config/checkpoint/one-batch checks. Do not run the
   entire historical test suite inside a scientific GPU job.
4. Inside an approved engineering-validation envelope, S00 diagnoses and fixes
   unambiguous frozen-semantics defects in tests/fixtures, config/schema parsing,
   dtype/API plumbing, runners, checkpoint I/O, artifact publication/provenance or
   logging, then resubmits serially within the same science and aggregate ceiling.
   Submission count has no default numeric cap unless the owner explicitly sets one.
5. Return to the owner before changing model math, data ownership/content, recipe
   candidate space, evaluator/metric, seeds, candidate count, scientific
   interpretation or aggregate resources, and at ceiling exhaustion, ambiguous
   diagnosis or recurrence of the same blocker after repair.
6. Keep active state in S10 `HANDOFF.md` and job provenance in
   `RUN_REQUEST.md`. Preserve raw outputs and minimum provenance; do not
   duplicate every incident into canonical, results and review documents.
7. Review only data/metric changes, each branch recipe freeze and the final
   staged-fusion/full capability result. Reviewers report and do not fix.
8. Commit at material implementation, phase-plan freeze and phase-result closure;
   do not create a documentation commit per runner failure.
9. No merge, push, upload, publication, Protocol A/B execution, attack, defense
   or S11+ work is implied.

## 2. Persistent S00 startup envelope

```text
You are persistent S00 for the active fl_v3 USENIX Security '27 track. 与 owner
使用中文交互。You own canonical planning and, after owner scope approval, directly
implement tightly connected milestones in the active linear worktree.

Before acting, read completely:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA,SESSIONS,KICKOFFS}.md;
- fl_v3/usenix27_orchestra/handoffs/S10/PHASE_I_PLAN.md for any Phase-I work;
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

## 6. S10 O-143 through O-149 launch state

- **Science order:** qualify camera and LiDAR independently; freeze reviewed
  branch recipes/checkpoints; staged fusion; aligned absolute-capability and
  fusion-contribution gate; profiler/optimization only after capability passes.
- **Reference policy:** start from coherent MIT-derived graph/initialization/
  recipe anchors and established published conclusions. Do not spend local
  compute re-proving them unless a concrete implementation conflict appears.
- **Practical comparator:** establish whether the current upgraded model is
  actually usable and improves on Alvis under aligned data/class/metric/evaluator
  semantics.
- **Retained evidence:** STOP-A split/evaluator is reusable; STOP-B is
  `INCONCLUSIVE`; C1-A `LOCALIZED_NORM` informs LiDAR candidates but does not
  select BN1d; C1-B bounded proxy runs do not establish capability.
- **Paused work:** current-A2, old C→D→E→F execution, profiler-first work and the
  old diagnostic harness.
- **O-144 Phase-I freeze:** `handoffs/S10/PHASE_I_PLAN.md` binds the exact two
  primaries, B4 x accumulation 8/effective B32, role-bound D_fit recipe, seed 0,
  20 epochs, terminal-only selection and five-WP/three-gate/two-envelope workflow.
- **O-145 amendment:** WP2 independently ports the pinned optimized CUDA
  BEV-pooling operation or an equivalent kernel without mmdet3d/mmcv runtime;
  WP4 gates fallback parity, FP32/FP16 policy, operator timing and aligned B4
  end-to-end timing. The primary checkpoint is ImageNet Swin-T, not NuImages.
- **Envelope-A terminal:** O-146/O-147/O-148 are consumed. Across 12 serial
  submissions and `0.516389/1.10` GH200-hours, Camera completed correctness,
  checkpoint, parity, end-to-end and memory checks but failed the frozen
  optimized-pooling promotion gate (`0.976174 > 0.80`); LiDAR passed and emitted
  its qualified config/recovery checkpoint. No capability metric or update ran.
- **O-149 process:** a future owner-approved engineering-validation envelope is
  completion-oriented under its aggregate GPU-hour ceiling and concurrency, with
  no default numeric submission cap. It permits only diagnosed frozen-semantics
  repairs and serial reruns; science/resource changes remain owner-gated.
- **O-150 backend decision:** PyTorch sorted `segment_reduce` is the Phase-I Camera
  production backend; CUDA remains an unpromoted explicit option, and its unmet
  `1.25x` promotion target is not a capability gate.
- **Next launch gate:** the prior Section-7.4 serial object and seal `1473ef67...`
  remain historical recipe evidence after LiDAR stopped at the epoch-5 numerical
  boundary. Section 7.4.7's parallel Camera/LiDAR-diagnostic amendment passed fresh
  independent review at `296ef9b...` with no open P0-P2; explicit owner activation
  of the containing review-seal commit remains required, and no job is active.

## 7. S10 envelope activation skeleton

```text
PHASE_AND_ENVELOPE:
REQUEST_STATE: DRAFT / NOT APPROVED
OBJECTIVE_AND_EXIT_GATE:
IMPLEMENTATION_AND_EXTERNAL_ACTIONS:
CANDIDATES_AND_MAX_COUNT:
DATA_SPLITS_AND_EVALUATOR:
SEED_POLICY:
TRAINING_EXPOSURE_AND_SELECTION_RULE:
AGGREGATE_GPU_HOURS:
SUBMISSION_POLICY_AND_CONCURRENCY:
STOP_AND_OWNER_ESCALATION_CONDITIONS:
MINIMUM_RUN_PROVENANCE:
OUTPUT_ROOT:
OWNER_APPROVAL: pending
```

Envelope A is closed, and its unused budget cannot be reused. The original
`49.0`-hour B4 request is historical control only. Revised Section 7.4 binds the
final Camera/LiDAR recipes at a `30.0` charged-GH200-hour hard ceiling; independent
review and owner acceptance are closed, but the owner explicitly deferred every
submission and then explicitly superseded that hold by activating the current
session. Its budget is not inferred from Envelope A's unused time.

After approval, individual job rows in `handoffs/S10/RUN_REQUEST.md` record Git
SHA, resolved-config hash, split, seed, command, resources, output, terminal state,
checkpoint hash and metric hash. Derived engineering fixes/resubmissions are
allowed only inside the approved science and aggregate caps. Under O-149 they do
not have a default numeric submission limit, but must remain serial (unless the
owner explicitly changes concurrency), diagnosed, fresh-output, and within the
approved aggregate ceiling.
