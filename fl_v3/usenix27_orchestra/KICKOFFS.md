# USENIX Security '27 Orchestra — active envelopes

> **Status (2026-07-14).** The closed S01-S07 worker/reviewer prompts have been
> removed from active routing. Their exact history remains in Git and handoff
> packages. O-094 makes persistent S00 the default implementer. The only current
> technical work is S08. Its implementation/remediation is sealed at
> `103c7389a47938b1f9dd0cba60251df6dce9e5bb` with R2
> `PASS_WITH_RESIDUAL_RISK`; detailed smoke history remains in its handoff.
> O-109 exact Q1 Job `431013` and Q2 Job `435151` are terminal at the runner level,
> checksum-verified, and consumed `00:07:58` total GPU elapsed. Numerical evidence
> selects global FP16 with SECOND/spconv FP32 as the close-ready sparse-route
> candidate and preserves uniform FP32 as reference/fallback. The next permitted
> action is an immutable evidence seal and independent review; owner policy
> acceptance remains the S08 close gate. No harness/work-chain expansion, merge,
> push, or S09 execution is authorized.
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

## 3. S08 envelope v1 — Q1/Q2 complete; evidence review pending

```text
SESSION_ID: S08
CURRENT_CODE_ANCHOR: 2a584053e6f6a3860b6f812681dc8d7342ca52ad
REBASELINE_SHA: 2a584053e6f6a3860b6f812681dc8d7342ca52ad
IMPLEMENTATION_CONTEXT: persistent S00, linear active worktree
DELIVERY_BRANCH: codex/s08-s09-cl-readiness
PRE_IMPLEMENTATION_AUDIT: handoffs/S08/MODEL_RECIPE_AUDIT.md; accepted as planning input
OWNER_DECISION: O-097 direction/branch/audit baseline; O-098 detailed plan/implementation/local validation/post-validation commit; O-099 consumed exact S08-SMOKE-1; O-100 provenance remediation/request preparation; O-101 consumed exact S08-SMOKE-2; O-102 narrow diagnostics/test remediation and SMOKE-3 request preparation only; O-103 consumed exact S08-SMOKE-3 PASS; O-104 review remediation/local validation/SMOKE-4 request preparation only; O-105 consumed exact S08-SMOKE-4 terminal test-construction FAIL; O-106 consumed exact post-freeze S08-SMOKE-5 PASS; O-107 prospective bounded mechanical remediation loop; O-108 remediation/evidence commit and independent re-review; O-109 Q1/Q2 completion goal, commits, and cumulative two-GPU-hour Slurm authority
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

## 5. S09 planning envelope — not implementation-ready

```text
SESSION_ID: S09
BASE_SHA: exact owner-accepted S08 seal, pending
IMPLEMENTATION_CONTEXT: persistent S00 unless owner selects independent isolation
APPROVED_COMPUTE: none
DECISION_SCOPE: full-pipeline performance/readiness only
```

Before S09 implementation, S00 must present reviewed S08 evidence and propose:

- exact precision policy and current model/config SHA;
- exact production `t1.v2` cache/manifest identities or a separate materialization
  request;
- minimal timing/memory/data-wait instrumentation and its output-neutral tests;
- 100-step gate thresholds and stop conditions;
- conditional 1000-step request, which is not preapproved by the 100-step plan;
- worker-count cells 0/2/4/8, microbatch/accumulation candidates, one-GH200 first;
- a decision rule for whether a separate two-GPU DDP request is justified.

Do not copy old profiler/audit wrappers or use mini throughput to freeze production
settings. S09 has no mAP/NDS, branch selection, Protocol A/B, attack, or defense
authority.

## 6. S10-S15 launch state

- **S10:** not copy-ready. Redefine exact centralized C/L/F/CBGS/initialization
  ablation cells only after accepted S08/S09 evidence.
- **S11:** not copy-ready. Freeze seeds, capability gates, and matrix only after an
  accepted S10 selection.
- **S12:** deferred and not copy-ready. Re-audit clean scene/log ownership,
  Protocol-A/B controls, and adaptation only after CL freeze.
- **S13:** blocked. No attack task until clean Protocol-B adaptation passes and the
  owner approves a new threat model. Legacy T5/T6/T7 import is forbidden.
- **S14:** blocked until a viable independently reviewed undefended S13 attack.
- **S15:** planning may remain rolling, but paper/artifact edits and any upload or
  submission require explicit owner scope and accepted evidence.

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
