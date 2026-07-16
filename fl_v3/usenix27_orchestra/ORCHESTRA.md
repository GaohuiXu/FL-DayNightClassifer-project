# USENIX Security '27 Orchestra — clean CL to federated multimodal security

> **Active status (2026-07-16).** S07 clean engineering, S08 precision
> qualification, and S09 full-pipeline engineering performance/readiness are
> closed. S08 policy/closing commit is
> `28f79802c0868afa6290d74ae6aeb9d23c7d088f`; S09 accepted review seal is
> `ced5992ea113bd21d7d545af505debf405b556b3`; S09 closing commit is
> `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`.
>
> Under O-121, `v3-ad-perception` was advanced by `--ff-only` to `351b7a0`.
> It and `codex/s08-s09-cl-readiness` have the same tip/tree; deletion of the
> delivery branch remains a separate owner decision. Fresh persistent S00
> completed the startup audit at exact clean base
> `a080d49c1c22de20ccb5b1353d4922c7df14a729` and is active on
> `codex/s10-cl-model-recipe`.
>
> O-122 accepts S10's six-stop A-F envelope, exact STOP-A split/evaluator gate,
> and primary full claim **absolute clean capability + fusion contribution**.
> O-124 authorizes bounded STOP-A/B/C completion. O-125's sole 15-minute final
> STOP-A remediation tuple was consumed by Job `467862`: focused tests and the
> 28,130-sample metadata traversal passed, but the exact MILP remained incomplete
> at Slurm `TIMEOUT`. No split/parity artifact exists. STOP-A is blocked pending
> an owner amendment; STOP-B/C have not started. STOP-D/E/F execution, merge,
> push, upload, publication, Protocol-A/B execution, attack, defense, and S11+
> remain unauthorized.
>
> Canonical companions: [`SESSIONS.md`](SESSIONS.md) and
> [`KICKOFFS.md`](KICKOFFS.md). Closed execution detail is compacted in the
> S08/S09 handoff packages; the complete pre-compaction ledger is recoverable at
> Git object `351b7a0`.

## 1. Current objective and sequencing

The immediate scientific objective is a strong, trustworthy centralized
camera-LiDAR nuScenes detector. Clean modality capacity, numerical stability,
training throughput, data ownership, and official evaluation must be established
before a federated security claim is meaningful.

The intended paper question remains:

> During vendor-style federated adaptation of a strong camera-LiDAR detector to
> rare, geographically and environmentally non-IID fleet data, can a
> modality-localized backdoor hide among legitimate long-tail updates, and can a
> structure-aware defense remove it without rejecting the rare benign updates
> that adaptation exists to learn?

The active order is evidence-gated:

```text
accepted clean engineering anchor 2a58405
                  │
                  ├── S08 pre-implementation model/recipe audit
                  │      owner architecture/fixture decisions
                  │      └── precision qualification; no capability claim
                  │
                  ├── S09 performance/readiness
                  │      100 steps, then conditional 1000 steps; no mAP claim
                  │
                  ├── S10 A-F CL health/recipe/speed/full claim     [envelope accepted]
                  └── S11 and later                                  [pending]
```

No deadline or milestone name creates execution authority. O-122 freezes the S10
ordering and scientific boundaries recorded in
[`handoffs/S10/HANDOFF.md`](handoffs/S10/HANDOFF.md); O-124/O-125 supplied only
their recorded bounded execution authority. Job `467862` consumed O-125 and
returned to the owner under its timeout rule. S11 and later remain undefined and
pending rather than inheriting any historical graph.

## 2. Accepted clean-foundation evidence

Acceptance is limited to the recorded scope. Detailed commands, tests, raw paths,
hashes, and negative results remain in the corresponding handoff packages.

| Foundation | Durable evidence | Accepted scope |
|---|---|---|
| S01 ZIP/data | worker `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`; review `7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc` | read-only stored-ZIP/data contract; historical `t1.v1` caches forbidden |
| S07-A | delivery `ba1571632557c20adbda3172221694cdbecfeabe`; executable `44cefd06bc815e893919d95c754896711dba3402`; review `370ea6c0bd4d9d737a5a50b6aff1c6f742589825` | data-foundation integration and caller provenance |
| S02 | worker `3aebf2dc1d19473f29260df279421047d216d70e`; review `df142dc9a391b87d05bd7becaba59459e9659f88` | target/loss and per-sample correctness contract |
| S03 | worker `50893839c45cd3e2ef1b72b98db6668df7030f2a`; review `2f62e570c9c24ef1e18a483888c3f28ad56a415e` | camera module contract with recorded production-shape limits |
| S04 | worker `483e149b95ec891b675df825d924a96bb225b7dd`; executable `84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f`; review `a0763c2e0b322d4ca53a92f9f69c90d9b231bbff` | SECOND/spconv module and pinned fp16-eval dispatch contract; not current six-task training acceptance |
| S05 | worker `a9c801fdee378906e54d06314d0c772b6559901a`; executable `96e509b71a3e22afb4de397132438fd3b9bbf5d8`; review `1c440843bb2b6d72f10310ff11fcde0d7d1e885c` | multi-task CenterHead/decode/NMS module contract |
| S06 | worker `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7`; executable `c330c72f4060348768c63fb1b7855ca56baffb95`; review `ca7bbd7e49e91ac2f214f39f62d5e416dd736383` | bounded C/L/F config/runtime/checkpoint/resume contract |
| S07-C | implementation `a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`; handoff `f736f41371666725a11d51bc3b01c6ececb59d50`; review `b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5` | legacy active-route removal; static review PASS, runtime inherited/not rerun |
| S07-B-COMPLETE | candidate `c615b6471a04b91a09c6ac6d487ff39a1501ceee`; test `29ca6637bcd0a4e9a6422f3b820fb43d5295ad2c`; Job `390576`; terminal/review package `7f3bd40158e5a8af30196509734782c4575c50aa`; review SHA-256 `b0feed5476dbc810b24a5dc3c7a678bc90ac3a2520360f02fdb6a6bf54691ebd` | plain clean FedAvg construction, one FP32 C/L/F optimizer update each, and worker-0/2 first-batch equality; bounded engineering only |

The current model foundation is:

- `C-STR8`: Swin-T, effective stride-8 multi-scale camera path, pure-camera LSS,
  0.5 m depth bins, and aspect-preserving geometry;
- `L-P020`: repaired 0.2 m pillar control;
- `L-S075`: SECOND-style `0.075 x 0.075 x 0.2 m` sparse path with approximately
  8x XY reduction before low-resolution densification;
- `F-U` and `F-CBGS`: clean late-BEV fusion templates;
- one reviewed six-task CenterHead, deterministic class/task-aware decode/NMS,
  and official nuScenes `DetectionEval` conversion path;
- one clean FedAvg path without attack/defense registry or selector.

These templates are architecture/config candidates, not approved training recipes.
The checked-in S07 JSON files are explicitly template-only; their current
`precision` field is not a scientific precision decision.

The current detector is a reviewed **BEVFusion-class hybrid**, not an exact copy of
one MIT BEVFusion public configuration. Its official-derived and project-adapted
components, parameter/precision map, recipe gap, sparse-GroupNorm numerical
hypothesis, and owner decisions are recorded in
[`handoffs/S08/MODEL_RECIPE_AUDIT.md`](handoffs/S08/MODEL_RECIPE_AUDIT.md). Module
acceptance establishes engineering contracts, not reference equivalence,
convergence, or training-recipe reasonableness.

## 3. Precision evidence and unresolved policy

### 3.1 Runtime capability is not model acceptance

The Arrhenius environment supports PyTorch CUDA, source-built cumm/spconv, FP32,
and CUDA FP16 autocast with GradScaler. Direct sparse BF16 is unsupported. That
runtime fact does **not** prove that every current model/loss/config trains
stably in FP16.

The current six-task evidence is:

| Evidence | What it proves | What it does not prove |
|---|---|---|
| Job `389356`, C-STR8 FP32 | finite gradients and optimizer call on one real-mini batch | multi-step stability, convergence, or speed |
| Job `389356`, C-STR8 FP16 scale 512 / 1 | scale 512 overflows; scale 1 produces finite gradients for that batch | normal dynamic-scaler continuation or full training |
| Job `389356`, L-S075 and F-U FP32 | finite gradients and optimizer calls; maxima about 1.91M / 1.22M | acceptable training recipe or convergence |
| Job `389356`, L-S075 and F-U FP16 scale 1 | direct nonfinite elements remain; first bad parameters are sparse SECOND stem/stage1; finite elements approach FP16 range | exact faulty operation, permanent impossibility of AMP, or head innocence |
| Job `390576` | one successful FP32 production-loop update for C/L/F | chosen scientific precision or multi-step readiness |

The historical Arrhenius mini jobs `211502`/`211722` used an older voxel model and
kept spconv in FP32 inside outer FP16 AMP. They are valid performance history for
that old path, not proof for the current `second_075` path. Before S08, the
production resolver automatically enabled sparse-conv FP16 whenever
`precision=fp16`; `s08.v1` introduced an explicit fail-closed
`sparse_conv_precision` partition, which current `s09.v1` preserves. S04 Job `341695` proved a bounded sparse module
path, not the current six-task optimizer seam.

### 3.2 S08 closed decision target (historical contract)

Owner decisions O-097/O-098 accept the v1 direction and detailed plan: hold the
current hybrid architecture, use the D1-style fixture only for numerical isolation,
add an explicit sparse precision partition, use minimal window-end diagnostics,
keep the current camera/LSS boundary, and return before any normalization amendment.
The exact smoke/remediation history is preserved in the S08 handoff package.
Remediation was sealed at `103c7389a47938b1f9dd0cba60251df6dce9e5bb`; R2 closed
P0-P2 and returned `PASS_WITH_RESIDUAL_RISK`. Under O-109, Q1 Job `431013` then
completed all eight declared primary cells. FP32 C/L/F references and camera FP16
passed; L full sparse FP16 recovered only at scale `0.03125`, while F full sparse
FP16 remained a bounded negative after 18 attempts. The SECOND-FP32 island passed
L/F at scales `32`/`16`. Q2 Job `435151` subsequently passed L-P020 FP16 at scale
`8` and F-CBGS with the SECOND-FP32 island at scale `16`. The exact jobs consumed
`00:07:58` total and added no cell, seed, harness, or retry. Independent R3
returned `PASS_WITH_RESIDUAL_RISK` with no P0-P2 findings. O-110 accepts the
reviewed candidate as the active precision policy and closes S08.

S08 compared, under one exact mini batch/step protocol:

1. uniform FP32 reference;
2. current end-to-end FP16 AMP with the production dynamic GradScaler contract;
3. FP16 AMP with an explicit SECOND/spconv FP32 island while the rest of the
   eligible graph remains autocast.

The diagnosis must record scaler progression, skipped versus executed optimizer
windows, per-task losses, elementwise gradient finiteness, and gradients at the
head input plus SECOND stem/stage boundaries. A fixed scale-1 success/failure alone
is insufficient. `L-P020` and `F-CBGS` receive bounded coverage so the later
ablation candidates are not carried forward under an unknown precision contract.

S08 may select an existing execution regime. Changing the six-task head, loss
semantics, architecture, metric, or optimizer recipe is a material scientific
change and returns to the owner before implementation.

## 4. S09 closed performance/readiness target (historical contract)

S09 begins from accepted close commit `28f7980`. Its purpose is to characterize
and improve the full current training pipeline before expensive CL capability
work. It is explicitly a **base-uniform engineering performance/readiness**
milestone. Branch and scientific training-recipe selection belong to S10.

Required evidence, under exact later-approved requests:

- a 100-step trainval-bound engineering gate, followed by 1000 steps only if the
  100-step gate passes;
- forward, loss, backward, optimizer, data-fetch and end-to-end timing;
- peak allocated/reserved GPU memory and headroom against the 96 GiB request
  envelope;
- `num_workers` 0/2/4/8 behavior on the production `t1.v2` ZIP/cache path;
- batch size, accumulation, throughput, optimizer-step/exposure accounting, and
  GradScaler skip/nonfinite behavior under the S08 policy;
- one-GH200 baseline first; two-GPU DDP is considered only if the measured
  single-GPU result is insufficient and the owner approves an exact DDP request.

Instrumentation must be production-minimal and output-neutral. Do not resurrect
the retired S07 audit wrapper, old profiling harnesses, process matrices, or
warnings-as-errors. S09 is not an mAP/NDS or model-selection session.

S09 may make **measurement-backed, output-neutral engineering optimizations** to
the current pipeline: ZIP/cache access, DataLoader lifecycle, host-to-device
transfer, redundant conversion/synchronization/allocation, and bounded logging or
checkpoint overhead. Each candidate must preserve sample order/contents, model
and loss semantics, accepted precision partition, gradients/updates, and exposure
accounting, and must have an exact owner-reviewed file/equivalence envelope before
implementation. S09 does not speculate or force a change when the measured gate
already passes. A material tensor-math, model-output, data-ownership, or recipe
change is not an engineering shortcut and remains outside S09.

S08 did not make the unusually large true LiDAR gradients healthy by scaling
them: GradScaler unscales before optimizer mutation, while the accepted policy
keeps the sparse backward path in FP32. Repeated tiny-group sparse GroupNorm is
the leading unproven mechanism hypothesis. S09 records scaler skips, nonfinite
windows and stability under the accepted policy, but does not change sparse
normalization, loss/head/targets, clipping, optimizer, scheduler, EMA,
augmentation, sampling, or initialization. If this residual blocks the 100-step
gate, S09 stops for a new owner decision instead of expanding into a root-cause
architecture experiment.

The owner-facing S09 workflow has four stops:

1. exact full-trainval `t1.v2` cache materialization, identity review, and bind;
2. minimal readiness instrumentation/runner implementation, local validation,
   immutable implementation commit, and focused O-009 smoke;
3. one exact production-path worker sweep plus F-U 100-successful-step gate,
   followed by independent evidence review; a failed performance threshold may
   motivate a separately owner-scoped output-neutral optimization and new exact
   G100 gate, never a hidden retry;
4. conditional fresh 1000-successful-step run, final independent review, and a
   close-ready S09 state. No mid-epoch resume or automatic DDP is implied.

The completed STOP-4 evidence satisfies this bounded engineering plan. The final
B=1 G1000 result is `178.024/203.231 ms` p50/p95, `5.542 samples/s`, a
`1.409821 h` full-train epoch estimate, and `8.314 GiB` peak reserved memory.
Median sampled GPU utilization was 51%, so the evidence does not claim full
GH200 saturation. Checkpoint-off B=2/B=4 reached `6.925/8.451 samples/s` with
`13.34/38.81 GiB` peak reserved memory; these are capacity observations for S10,
not selected recipes. The unexplained large true SECOND gradients and scientific
branch/recipe/capability decisions remain outside S09.

## 5. CL-to-FL scientific protocols

Protocol roles remain locked, but their exact split/cells are deferred.

### Protocol A — nuScenes-scratch federated training

Clients receive one frozen architecture and identical declared initialization,
not a detector trained on nuScenes. Public ImageNet/NuImages initialization must
be distinguished from fully random initialization. The matched centralized
control uses the same data, initialization, exposure, optimizer-step semantics,
and precision.

### Protocol B — centralized base plus federated tail adaptation

Protocol B is the primary security setting. A vendor trains `W_base` on common
`D_base`; regional/fleet data-silo clients federatively adapt it on disjoint
long-tail `D_tail`. The attack, if later approved, occurs during adaptation.

Binding split rules:

- construct `D_base` and `D_tail` from the official training split only;
- split at scene/log ownership, including adjacent keyframes/sweeps and duplicate
  raw-file identity; do not split by annotation row;
- define tail criteria from train-only information and freeze/hash criteria,
  assignments, statistics, and evaluation eligibility before attack work;
- keep official validation/test held out, or reserve a scene/log-disjoint
  `E_tail` before training if needed;
- never use a full-train capability checkpoint that has seen `D_tail` as the
  scientific Protocol-B initializer;
- establish `W_base`, pooled-tail oracle, local-only, and clean federated
  adaptation controls before any attack or defense claim.

The owner will decide whether, when, and in which future milestone this protocol
is re-audited or materialized after the clean-model architecture is frozen. The
old S12 proposal and milestone assignment are not current authority.

## 6. Persistent S00 collaboration model

Owner decision O-094 replaces the one-fresh-worker-per-session default.

1. **One persistent implementation context.** S00 may directly plan, implement,
   validate, document, and integrate one or more tightly connected milestones in
   one long-lived task/worktree. `Sxx` denotes an evidence milestone and handoff
   namespace, not necessarily a separate Codex task or worktree.
2. **Linear Git ownership.** The active track normally advances linearly from one
   reviewed commit. Separate implementation worktrees are used only for genuine
   parallel isolation, conflicting ownership, risky experiments, or explicit
   owner direction.
3. **Subagents are bounded.** Planning/research subagents may be used before
   implementation. They do not create parallel production branches. O-096 records
   the owner's maximum-reasoning override for the current persistent S00 and its
   pre-S08 planning/research or independent-review subagents; it does not authorize
   implementation or compute.
4. **Independent review remains mandatory.** After an immutable implementation or
   execution-evidence SHA exists, a reviewer subagent reads that exact diff and
   artifacts, reports findings first, and does not fix code. A separate review
   worktree is required for high-risk split/metric/scientific changes, conflicting
   concurrent state, exact runtime reproduction, or owner request.
5. **Remediation stays linear.** S00 fixes accepted findings in a new commit; the
   reviewer rechecks the new diff. A review-only artifact may be sealed linearly;
   no reviewer merge history is required.
6. **Durable records are phase-sized.** Use one compact `HANDOFF.md`, exact
   `RUN_REQUEST.md` before material compute, `RESULTS.md` for executed jobs, and
   `REVIEW.md` for independent review. Preserve failures and hashes, but do not
   generate micro-handoffs or audit wrappers for every conversational step.

The owner still reviews the plan/envelope before S00 begins a newly scoped
milestone or submits exact compute. Persistent context simplifies coordination; it
does not broaden scientific, Git, compute, upload, or publication authority.

## 7. Compute, data, and execution boundaries

- Planning and implementation are not execution permission.
- O-009 covers only a recorded bounded engineering smoke: one node, at most one
  GPU, at most 60 minutes/job, one concurrent job, and two cumulative GPU-hours
  for the milestone. It never covers full cache/trainval coverage, 100/1000-step
  gates, model qualification/training steps, profiles, metrics, matrices, seeds,
  arrays, DDP, or publication.
- Every material job requires an exact immutable commit/snapshot, command, data
  scope and identities, resources, output, stop conditions, and explicit current
  owner/S00 audit in `RUN_REQUEST.md`.
- By default, a changed commit, config, split, command, cell, seed, resource
  request, or output invalidates approval. No automatic resubmission or spare-GPU
  expansion.
- O-107 lets the initial owner approval for an exact O-009 smoke opt into a bounded
  mechanical remediation loop. It may contain at most two diagnosed replacement
  submissions (three jobs total) under the same objective/selectors, bounded data,
  command family and resource ceiling. Every replacement is frozen and recorded
  before submission. Only obvious test/fixture/wrapper/provenance/artifact or
  output-neutral diagnostics fixes qualify; identical retries do not. Any possible
  model/data/precision/recipe/metric/scientific/resource change, uncertainty,
  repeated blocker, or exhausted cap returns to the owner.
- Mini is engineering-only. It cannot support mAP/NDS, fusion-gain, ASR, defense,
  generalization, or paper claims.
- O-112 STOP-1 Job `441191` materialized exact read-only train/val `t1.v2`,
  `n_sweeps=10` caches. Raw cache/source/job evidence passed independent review;
  bounded re-review of documentation-only remediation SHA `5252a59` closed every
  P2/P3 finding and returned `PASS_WITH_RESIDUAL_RISK`. O-113 owner-accepts the
  exact caches for downstream production binding. Historical Job `332651` `t1.v1`
  caches remain coverage evidence only and forbidden production inputs.

## 8. Owner decisions and freeze points

### Locked current decisions

- Protocol B is primary security; Protocol A is the clean optimization/control
  setting.
- The S02-S06 C/L/F/head/runtime foundations and official clean evaluation path
  remain the current engineering foundation. This does not freeze reference
  equivalence or prevent an owner-approved architecture amendment after the S08
  pre-implementation audit.
- Legacy T5/T6/T7, old defense code, e231 history, `collab/**`, and old cycle_04
  contracts cannot be recovered as implementation or scientific authority.
- S08 precision is frozen under O-110; S09 labelled base-uniform engineering
  performance/readiness is closed PASS under O-120 and fast-forward integrated
  under O-121. O-122 accepts S10's six-stop A-F envelope, exact STOP-A
  split/evaluator gate, and primary full claim; it does not authorize execution.
  Every S11+ boundary remains pending a future results-driven rebaseline.
- Persistent S00 is the default implementer; independent reviewer subagent or,
  when required, a separate review worktree supplies the quality gate.

### Still unresolved

| Decision | Latest freeze point |
|---|---|
| Which, if any, sparse-normalization/head/view-transform amendment is promoted | observation-first STOP-B evidence followed by the bounded strong-contrast STOP-C gate; tiny-group GN remains a hypothesis, not a diagnosis |
| Production optimizer groups/LR/scheduler/clip/EMA/augmentation/sampling recipe | select and freeze only in STOP-D after STOP-C architecture/init closure |
| Cause of unusually large true SECOND gradients | STOP-B must return `LOCALIZED` or honestly `INCONCLUSIVE`; neither permits silent STOP-B model changes |
| Single-GPU batch/accumulation/workers and whether DDP is needed | STOP-D/E decision; B=2/B=4 remain capacity evidence until qualified, and DDP is not authorized |
| STOP-A/B/C implementation, commit, isolated review topology, Slurm tuple and cumulative budget | one bounded owner completion authority after GPU-budget review; currently not authorized |
| STOP-D/E/F exact cells, resources and execution | separately owner-gated after upstream evidence; STOP-F owns the single-seed primary full run and sealed official-val decision |
| Absolute clean capability, fusion contribution, mAP/NDS, per-class and speed claims | only after the corresponding frozen STOP-F/E evidence; internal evaluator outputs are explicitly proxy-only |
| S11 and later milestone roles | pending; do not inherit the historical S11-S15 outline without a new owner decision |
| `D_base`/`D_tail`, client unit, update scope, and clean Protocol-A/B cells | pending after the future clean-model freeze and a new owner decision |
| New threat model, attack, and defense | pending; no attack or defense work is authorized |

## 9. Active owner-decision registry

O identifiers are never renumbered or reused. Binding decisions are cumulative;
earlier rows preserve the exact authority/state when issued and never reauthorize
consumed compute. O-122 supersedes only the stale S10-planning state left by
O-121; it does not reinterpret closed S08/S09 evidence or authorize S10 execution.
Closed ranges below are provenance.

| ID | Binding decision | Authority |
|---|---|---|
| O-001 | Protocol B is the primary security setting; Protocol A is the separately labelled clean optimization/control setting. | locked scientific |
| O-002 | Material changes require scoped ownership, durable evidence, independent review, and exact execution authorization; a separate worker task is not inherently required. | locked process, amended by O-094 |
| O-003 | S00 may refine unstarted operational work from accepted evidence; locked scientific scope returns to the owner. | active orchestration |
| O-004 | Every independently executed or reviewed state uses an exact durable SHA/snapshot; extra worktrees are optional isolation, not the default unit of progress. | workspace policy, amended by O-094 |
| O-005 | Architecture, capability, protocol, matrix, and paper decisions freeze only at their declared evidence gates. | scientific process |
| O-008 | Default reasoning is `xhigh`; `ultra` requires a recorded difficult planning/research or exceptional review reason. | resource policy |
| O-009 | Only bounded non-scientific smoke has standing limits; all 100/1000-step, full-data, profile, metric, DDP, matrix, retry, and scientific work needs exact owner approval. | compute policy |
| O-010 | When a separate worker/reviewer task is used, kickoff and review pin exact base/worker SHAs; mutable state is never a review baseline. | kickoff schema |
| O-011 | S00 presents the exact scope/evidence/compute envelope before creating an additional task or launching material compute; there is no automatic downstream or reviewer launch. | launch policy |
| O-017 | Reviewed S02-S05 camera/LiDAR/head/loss contracts remain the clean C/L/F construction foundation. | locked architecture |
| O-018 | CenterHead retains the reviewed no-starvation adaptation and explicit global class mapping. | locked head/decode |
| O-025 | spconv 2.3.8 FP16 no-grad evaluation retains the reviewed version-guarded spconv-only training-dispatch workaround. | locked runtime |
| O-094 | Use persistent S00 for tightly connected implementation milestones; use bounded planning/research subagents and independent reviewer subagents, with a separate review worktree only when risk or owner direction requires it. | active collaboration model |
| O-095 | Redefine S08 as precision qualification and S09 as performance/readiness; mark S10-S12 pending/deferred until their upstream evidence is reviewed. No compute is authorized by this rebaseline. | active schedule |
| O-096 | Before S08 implementation, persistent S00 performs a deep current-model/numerics/recipe audit at the platform maximum reasoning setting, may use bounded read-only research subagents, and may reconcile straightforward canonical-document conflicts. The audit authorizes neither production changes nor compute; exact owner decisions still gate S08. | active pre-implementation gate |
| O-097 | Accept S08 envelope v1 direction; authorize creating/switching to `codex/s08-s09-cl-readiness` and committing the current pre-S08 audit baseline. S00 must present the detailed multi-agent implementation plan for one more owner review before execution. No compute, S09 work, normalization amendment, merge, push, or upload is authorized. | active S08 launch gate |
| O-098 | Accept the detailed S08 v1 plan; authorize persistent S00 implementation and local validation, followed by one immutable implementation commit after validation. Approve a one-GH200 `<=1h` resource ceiling separately for the focused smoke and later Q1, while retaining exact immutable tuple binding, no automatic retry, and all S08/S09 scientific non-goals. | active S08 implementation gate |
| O-099 | Explicitly approve exact `S08-SMOKE-1`. Its single submission, Job `426619`, is consumed and terminal: pre-pytest FAIL on the verifier's blanket clean-checkout rule against the S07-evidenced spconv build-metadata patch. No retry, environment/source mutation, Q1, or broadened compute is authorized. | consumed S08 smoke |
| O-100 | Approve the narrow provenance remediation: do not modify/reset the external spconv checkout; bind its sole `" M" pyproject.toml` state by exact path/status/content/state SHA while retaining source HEAD, import-origin and executable-build checks; add fail-closed tests and prepare a replacement snapshot/request. This does not authorize replacement execution. | active S08 remediation |
| O-101 | Explicitly approve exact `S08-SMOKE-2`. Its single submission, Job `427800`, is consumed and terminal: source-state/runtime attestation PASS, then focused pytest FAIL with 103 passed/3 failed/0 skipped. Two failures expose disabled-GradScaler diagnostics compatibility; one is a test-message regex mismatch. No retry, implementation commit, Q1, or broadened compute is authorized. | consumed S08 smoke |
| O-102 | Approve the narrow post-Job-427800 remediation and SMOKE-3 request preparation: do not call growth/backoff getters when GradScaler is disabled, require the corresponding diagnostic fields to be `None`, and correct only the test regex to the existing six-task production error. No model/loss/optimizer/precision change and no GPU execution are authorized. | active S08 remediation |
| O-103 | Explicitly approve exact `S08-SMOKE-3`. Its single submission, Job `428112`, is consumed and terminal PASS: exact runtime/source-state attestation, 106 passed/0 failed/0 errors/0 skipped, `smoke.exit=0`, verified artifact hashes, `COMPLETED 0:0`, and zero restarts. This clears only the focused implementation-smoke gate; no retry, Q1, precision-policy acceptance, or broadened compute is authorized. | consumed S08 smoke |
| O-104 | Approve the independent-review remediation envelope: prebind the complete Q1 fixture before model construction, gate scheduler transitions and EMA-disabled state, reconcile active status wording, run local validation, and prepare/freeze an exact bounded `S08-SMOKE-4` fixture-attestation request. No GPU execution, retry, new commit, Q1, precision-policy acceptance, S09 work, merge, push, or upload is authorized. | active S08 review remediation |
| O-105 | Explicitly approve exact `S08-SMOKE-4`. Its single submission, Job `428889`, is consumed and terminal FAIL: exact preflight PASS, then Phase 1 ended 115 passed/1 failed/0 errors/0 skips because the new `cam_intrinsics` drift test assigned `1.0` to an already-`1.0` identity-matrix value. Phase 2 did not start; no fixture identities were emitted. No retry, source edit, replacement request, Q1, commit, or broadened compute is authorized. | consumed S08 smoke |
| O-106 | Explicitly approve the exact post-freeze `S08-SMOKE-5` tuple. Its single submission, Job `429080`, is consumed and terminal PASS: exact preflight, 116 passed/0 failed/0 errors/0 skips, 1 fixture-attestation passed/0 skipped, both exit files `0`, all five fixture identities emitted, checksum manifest verified, `S08_PRECISION_SMOKE_PASS`, `COMPLETED 0:0`, and zero restarts. This clears only the focused review-remediation/fixture-attestation gate; no retry, Q1, remediation commit, precision-policy acceptance, or broadened compute is authorized. | consumed S08 smoke |
| O-107 | For future bounded O-009 engineering smoke, one initial exact owner approval may explicitly opt into a mechanical remediation loop of at most three total submissions/two cumulative GPU-hours. S00 may diagnose, locally fix, freeze, record, and submit only obvious test/fixture/wrapper/provenance/artifact or output-neutral diagnostic-plumbing replacements without a second owner review. No identical retry or scope/resource expansion is allowed; any possible model-output/gradient/update, data, precision, optimizer/scheduler/EMA, metric/scientific change, uncertainty, repeated blocker, or exhausted cap returns to the owner. This applies prospectively and does not reinterpret Jobs 426619-429080. | active bounded engineering workflow |
| O-108 | Authorize persistent S00 to create one immutable S08 remediation/evidence commit from the reviewed working candidate after local verification, then launch an independent re-review pinned to that SHA. The reviewer reads the exact diff, handoff/request/results, and raw smoke artifacts and does not fix source. No Q1 compute, precision-policy acceptance, S09 execution, merge, push, or upload is authorized. | active S08 remediation seal/re-review |
| O-109 | Set the persistent completion goal to finish Q1 primary, the minimal required Q2 L-P020/F-CBGS compatibility gate, independent evidence review, and a close-ready S08 linear commit state. Authorize all exact in-scope commits and Slurm submissions once their immutable tuples are recorded, with cumulative one-GPU elapsed allocation across all new Q1/Q2 jobs capped at two GPU-hours. Short earlier jobs leave only their unused elapsed budget for later jobs. The simplified O-107 mechanical workflow applies to obvious non-scientific defects, but no scientific cell/seed/data/resource expansion or silent reinterpretation is allowed. Forbid work-chain/harness expansion, merge, push, S09 execution, attacks, and defenses. | consumed S08 completion authority |
| O-110 | Accept reviewed S08 close-ready seal `d31adea049c84e47a0e4f82f38f22a2ca91a5a6f` and R3 `PASS_WITH_RESIDUAL_RISK`; freeze global FP16 for camera/dense-pillar, global FP16 with explicit SECOND/spconv FP32 island for sparse LiDAR/fusion, and uniform FP32 as reference/fallback; reject full sparse FP16 as the unified F-capable route; close S08 PASS; authorize a fast-forward-only integration into `v3-ad-perception`; open S09 discussion after its reading gate. No S09 compute, merge commit, push, attack, or defense is authorized. | accepted S08 close / active S09 discussion gate |
| O-111 | Accept closing commit `28f79802c0868afa6290d74ae6aeb9d23c7d088f` as the S09 base; accept the streamlined four-stop S09 direction; define S09 as engineering optimization/readiness of the current model/code under the O-110 precision policy; defer branch and scientific training-recipe selection, including any sparse-normalization amendment, to S10. At each future stop the owner will review one exact plan plus Git/Slurm authority and GPU quota; after that approval S00 creates a concrete goal and works continuously to the stop boundary. This decision authorizes envelope/document preparation only, not S09 implementation, commit, Slurm execution, retry, merge, push, attack, or defense. | active S09 envelope gate |
| O-112 | Accept the S09 four-stop envelope and start `STOP-1 DATA`; authorize S00 to create the bounded STOP-1 goal, seal the envelope/request/evidence in linear commits, derive and record the exact immutable source/snapshot, command/script hashes and fresh output from the accepted `28f7980` base, then submit exactly one full-trainval metadata-only `t1.v2`, `n_sweeps=10` train/val cache job on one GH200 with eight CPUs, 96 GiB host memory, `00:30:00`, and at most 0.5 GPU-hours. Job `441191` consumed the sole submission and completed `0:0` in `00:03:06`; no retry occurred. Independent review passed raw cache/source/job evidence but returned `REMEDIATE` for P2/P3 durable provenance/status wording; documentation-only remediation SHA `5252a59` closed every finding under bounded re-review and received `PASS_WITH_RESIDUAL_RISK`. At this decision's terminal state, owner STOP-1 acceptance remained pending; O-113 subsequently resolves it. No payload extraction/scan, model, loader/profile, STOP-2 implementation, merge, push, attack, or defense was authorized by O-112. | consumed S09 STOP-1 compute / superseded by O-113 acceptance |
| O-113 | Accept reviewed S09 STOP-1 seal `c94b4065f6da2504bdc98348610794cd9ae532cb` and independent `PASS_WITH_RESIDUAL_RISK`; close STOP-1; permit future S09 requests to bind only the exact reviewed train/val `t1.v2`, `n_sweeps=10` canonical/pickle/sidecar plus accepted manifest logical/physical identities; open STOP-2 detailed planning. No cache rebuild/retry, STOP-2 implementation/commit/compute, merge, push, attack, or defense is authorized by this acceptance. | accepted S09 STOP-1 / active STOP-2 planning gate |
| O-114 | Accept the exact S09 STOP-2 implementation envelope recorded in `handoffs/S09/HANDOFF.md`: advance the current resolved production schema to `s09.v1` with a hash-bound execution contract; retain the O-110 precision matrix; implement a non-resumable/no-checkpoint/no-eval readiness lifecycle, bounded production-loader profile, and direct output-neutral host/CUDA timing without observers/hooks/profilers; mechanically update current configs/tests including O-110 sparse-FP32 templates. Authorize S00 to seal this planning baseline, implement, run local/static validation, continuously fix ordinary in-envelope defects, create linear immutable implementation/evidence commits, and obtain independent review. Final candidate `37aef4d` received `PASS_WITH_RESIDUAL_RISK` with no open P0-P2; request remediation `cad7262` subsequently received closure `PASS_WITH_RESIDUAL_RISK` with no open P0-P3. The exact snapshot/script/selectors/output request is frozen but no GH200 submission is yet authorized. No model/loss/gradient/precision/data/recipe/resource expansion, merge, push, attack, or defense is authorized. One concise exact owner confirmation is still required before smoke submission. | implementation/request review complete / STOP-2 smoke awaiting confirmation |
| O-115 | Approve the exact `S09-STOP2-SMOKE` tuple frozen and independently reviewed in `handoffs/S09/RUN_REQUEST.md`: execution source `37aef4d6b3f4679d6702d0acef2bb5bd1b57a952`, the exact detached snapshot, four selectors/44 expected tests, seed and environment policy, read-only job/submit scripts and hashes, fresh `_a1` output, one GH200/four CPUs/32 GiB/`00:10:00`. Explicitly enable the recorded O-107 mechanical boundary: initial plus at most two derived replacements, at most 0.5 cumulative GPU-hours, with unchanged selectors/order, toy scope, seeds, resources and stop conditions; only obvious test/fixture/wrapper/provenance/artifact/output-neutral timing-plumbing fixes qualify. Initial Job `441293` consumed the submission and completed `0:0` in `00:01:04`, with 44 passed/zero failed-errors-skipped; no replacement was used or warranted. Evidence remediation `79f87dc` received closure `PASS_WITH_RESIDUAL_RISK` with no open P0-P3. Any material or uncertain change stops for owner review. No merge, push, STOP-3, production model/data run, metric, attack, or defense is authorized. | consumed/reviewed STOP-2 smoke / subsequently accepted by O-116 |
| O-116 | Accept and close S09 STOP-2 after exact Job `441293`, evidence remediation `79f87dc`, and independent `PASS_WITH_RESIDUAL_RISK` with no open P0-P3; open detailed STOP-3 planning. This decision alone does not authorize STOP-3 compute, commit, merge, push, STOP-4, attack, or defense. | accepted S09 STOP-2 / superseded by O-117 STOP-3 authority |
| O-117 | Accept the detailed STOP-3 envelope and its update: exact non-template F-U resolved config with random seed `0`, global FP16 plus explicit SECOND FP32 island, AdamW `1e-4/0.01`, constant scheduler, uniform sampling, EMA/clip/BEV augmentation/GT paste off, microbatch/accumulation/world `1/1/1`, fixed training workers `8`, 100 successful updates within 120 attempts, ten-successful-window timing warm-up, and accepted STOP-1 train/val/manifest identities. Authorize one ordered production loader sweep at workers `0/2/4/8`, two repeats each of 32 digest + 16 warm-up + 256 measured batches; direct stage timing plus 1 Hz read-only `nvidia-smi` telemetry, but no module profiler. Authorize S00 to create the exact linear source/request/evidence commits, derive and record the immutable snapshot/config/script/output tuple, and submit exactly once on one GH200, 16 CPUs, 96 GiB host memory, `01:00:00`, at most one GPU-hour, with no retry/replacement/array/DDP/seed expansion. The request gates include content identity, 100 accepted updates, no nonfinite/discarded windows or counter drift, post-warm-up accepted ratio at least 95%, integrated p95/p50 at most 1.5, data-wait share at most 10%, peak reserved memory at most 86 GiB, worker-8 warm throughput at least 90% of the best warm cell, and both declared epoch estimates at most 24 hours. Aggregate finite loss is evidence; monotonic convergence is not a STOP-3 claim. Job `441511` consumed the sole submission and failed `1:0` in `00:02:29` before data/loader/model work because the runner used runtime-only modules and editable spconv JIT could not find `cublasLt.h`; the failed import also drifted the cumm native build identity. No retry occurred or was authorized at that boundary. Independent failure/remediation review and a new owner amendment were required before runtime re-attestation or a replacement G100; O-118 later supplied only that conditional amendment. No merge, push, automatic STOP-4, attack, defense, or scientific metric was authorized. | consumed STOP-3 compute / pre-model bootstrap FAIL / superseded only by O-118 conditional amendment |
| O-118 | Approve the recorded conditional-continuation envelope after the O-117 bootstrap failure: first submit exactly one one-GH200 dependency-attestation job to stabilize and checksum the editable spconv/cumm runtime; only after immutable evidence and independent Phase-A PASS, derive only those accepted build hashes into the otherwise unchanged O-117 tuple, independently confirm that derivation, and submit exactly one replacement G100. Authorize continuous linear commits, local/static checks, evidence sealing, and independent review within that envelope without repeated mechanical approvals. Both phases are serial, no-retry, and together remain inside the approved two-GPU-hour ceiling; any model/data/precision/recipe/resource/gate change stops. Job `442152` completed Phase A `0:0` in `00:11:52`; Job `446225` completed Phase B `0:0` in `00:05:05`, with all STOP-3 gates passing. Immutable evidence `c28d09c` received independent `PASS_WITH_RESIDUAL_RISK` with no P0-P2; closure re-review of remediation `84adfd0` found no open P0-P3 and marks STOP-3 owner-ready. O-118 supplies no retry, STOP-4, merge, push, attack, defense, metric, or scientific-claim authority. | consumed STOP-3 recovery compute / technical PASS / superseded by O-119 acceptance |
| O-119 | Accept and close S09 STOP-3 after Jobs `442152`/`446225`, immutable evidence `c28d09c`, remediation `84adfd0`, and independent review with no open P0-P3. Approve one continuous STOP-4A-D goal: add a fail-closed `s09.v2` activation-checkpoint/profiler contract; run one serial STOP-4A job containing focused tests, the exact STOP-3 B=1/checkpoint-on bounded operator profile, and checkpoint-off B=1/2/4 capacity cells; remove only measured/source-proven redundant synchronization/allocation while preserving model/loss/gradient/update/data/precision semantics; independently review immutable changes/evidence; run one optimized B=1 G100; then, only after reviewed G100 PASS, run one fresh B=1 G1000. Authorize linear commits, local validation, immutable snapshots, independent reviewer branches if needed, and these three serial one-GH200 jobs at ceilings `00:30:00 + 00:30:00 + 01:00:00`, 16 CPUs/96 GiB each, two cumulative GPU-hours, no retry. B=2/4 are capacity evidence only and do not select a recipe. No worker matrix, model/head/loss/normalization/optimizer/LR/scheduler/EMA/augmentation/sampling/init change, DDP, merge, push, metric, attack, or defense is authorized. | consumed S09 STOP-4 compute / independently reviewed technical PASS / superseded by O-120 close |
| O-120 | Accept S09 final review seal `ced5992ea113bd21d7d545af505debf405b556b3` and `PASS_WITH_RESIDUAL_RISK / no open P0-P3 / STOP-4 closure GO`; close S09 PASS as a labelled base-uniform engineering performance/readiness milestone. Preserve the limits that this is single-seed bounded engineering evidence, not convergence, mAP/NDS, model quality, recipe/batch selection or full-GH200 utilization; the large true SECOND gradient remains unresolved for S10. This decision does not authorize fast-forward, merge, push, branch/worktree deletion or switching, S10 implementation/compute, DDP, metric, attack, or defense. | accepted S09 close / integration decision pending / S10 planning next |
| O-121 | Authorize and complete `--ff-only` advancement of `v3-ad-perception` to S09 closing commit `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`; verify it and `codex/s08-s09-cl-readiness` have identical tips/trees and clean worktrees. Retain the delivery branch pending a separate deletion audit. The current S00 must finish S08/S09 closure compaction and retire without creating S10 work. Accept only the S10 work definition—centralized-model numerical/architectural health, production recipe selection, and final-architecture GH200 optimization. A fresh Ultra-reasoning S00 will create `codex/s10-cl-model-recipe` and research/propose the exact envelope. Previous stop designs, full-run placement, S11+ roles, S10 implementation/compute, merge, push, and branch/worktree deletion are not accepted or authorized. | consumed handoff/compaction decision; S10 planning state superseded by O-122 |
| O-122 | Accept S10's primary full claim as **absolute clean capability + fusion contribution** and accept the six-stop order: A split/evaluator gate; B observation-first numerical diagnosis; C strong-contrast architecture/initialization selection; D production-recipe selection/freeze; E final-graph GH200 profiling and sustainable output-neutral optimization; F frozen full-train single-seed capability/fusion run, sealed official-val evaluation and close. Accept the exact train-only nested split/evaluator protocol in `handoffs/S10/HANDOFF.md`, the A0/A1/A2 initialization registry with conditional A3, current-vs-coherent-MIT strong contrast, at most two diagnosis-triggered counterfactuals, and conditional consideration of BN1d, TransFusion and LiDAR-conditioned DepthLSS only when B/C evidence warrants it. Official val is sealed from selection; internal metrics are proxy-only; an official-val failure is a negative result/amendment point, not permission to select another checkpoint. Record that a primary full run belongs to STOP-F, remains single-seed, and that limited-rung evidence cannot make the full claim. Authorize canonical/plan documentation only. STOP-A/B/C implementation, commits, review topology and Slurm execution await a separate bounded completion authority after GPU-budget review; STOP-D/E/F execution, merge, push, upload, publication, Protocol A/B execution, attack, defense and S11+ remain unauthorized. | accepted S10 scientific envelope / documentation only / ABC completion authority pending |
| O-123 | Reject `S10-ABC-COMPLETION-v0-estimate`: B=1 is proven under-utilization and cannot be the epoch/rung cost or execution basis for STOP-A/B/C. Any revised ABC request must use physical B=4 as the minimum scientific-training microbatch, reuse S09's accepted B4 20-update/throughput/memory evidence, and account for its fixed-batch tail explicitly; B=1 is allowed only for a tiny paired diagnostic decomposition and never for a rung or epoch. B=8/16 may be investigated later at the frozen recipe/performance boundary rather than silently added to ABC. An obvious implementation/training-correctness failure must stop the long cell and use bounded deterministic step probes before a fresh restart; finite but scientifically unhelpful or non-converging training is evidence, not permission for an open-ended debug/tuning chain. Reaffirm the simplified O-094 persistent-S00 collaboration model. No revised budget, implementation, commit, review worktree or Slurm authority is granted yet. | v0 compute estimate rejected / B4 minimum frozen / v1 replan pending |
| O-124 | Approve `S10-ABC-COMPLETION-v1-B4-estimate` and start one continuous STOP-A/B/C completion goal. Authorize persistent S00 to implement, locally validate, create linear planning/implementation/evidence/remediation commits, derive and record every exact immutable job tuple before submission, submit and monitor serial Slurm work without repeated owner approval, and obtain stop-level independent review. The aggregate ceiling is 27 elapsed one-GH200 hours, at most one active job, at most seven scientific allocations plus two diagnosed debug/fix allocations (nine submissions total), no identical retry, no DDP/array/spare GPU, physical B4 minimum for scientific training, B1 diagnostic decomposition only, and the STOP-C `drop_last=true` matched-token proxy policy. Obvious correctness failures use the recorded `1 -> 5 -> 20` B4 step protocol; weak but finite scientific outcomes do not open tuning loops. Authorize one isolated S10-R review context/worktree required for high-risk STOP-A and reuse it rather than creating a review chain. Any changed data/metric/model/candidate/seed/horizon/resource boundary, repeated blocker, uncertain classification or exhausted cap returns to the owner. STOP-D/E/F, B8/16 execution, full run, merge, push, upload, publication, Protocol A/B execution, attack, defense and S11+ remain unauthorized. | aggregate ABC authority recorded / execution gated at STOP-A after O-125 |
| O-125 | After STOP-A Job `463593` failed before real solve and derived Job `463649` passed focused tests/metadata traversal but exhausted its one-hour allocation inside the 94-cold-MILP exact tie-break, approve the output-equivalent `d7caf53` blocked-radix/fail-closed remediation and one exact final A-GATE tuple. The owner grants one additional GH200-hour of STOP-A contingency but requires the next allocation to request only `00:15:00`; unused contingency is not an automatic retry/submission entitlement. The frozen tuple keeps one GH200, 16 CPUs, 96 GiB, identical data/gates/output scope, no requeue/array/DDP, consumes debug/fix slot 2/2 and submission 3/9, and has canonical envelope SHA-256 `24de0be54806fbd1270bec2f560451ee62a138a593a5cb0a542f0a7c76d7f061`. Job `467862` consumed that tuple and timed out after `00:15:14`: 13 focused tests and full metadata traversal passed, but the exact split solve emitted no accepted split/parity artifact. Per the decision, no identical retry is authorized and STOP-A returns to the owner. Actual cumulative STOP-A/ABC allocation is `1.271389` GH200-hours; unused contingency confers no execution right. | consumed STOP-A final remediation / STOP-A blocked pending owner |

## 10. Closed and consumed history

| O-ID/range | Terminal disposition |
|---|---|
| O-006-O-007 | original sole-orchestrator/fresh-worker topology; superseded by O-094 |
| O-012-O-016 | S01/S07-A acceptance and Wave-A preparation; accepted evidence is in Section 2 |
| O-019-O-031 | S02-S06 remediation/review scheduling; accepted evidence is in Section 2 |
| O-032-O-038 | first S07-B integration and caller fixes; frozen core-integration history |
| O-039-O-051 | failed/unreviewed T5 expansion; retired negative evidence |
| O-052-O-091 | old runtime, spawn-policy, multiprocessing, warning-fatal, and harness chain; retired diagnostic evidence |
| O-092 and A1-A3 | cleanup, dependency correction, legacy harness removal, and simplified completion boundary; consumed by S07-C/S07-B outcomes |
| O-093 | owner accepted and closed S07-B-COMPLETE at bounded clean-engineering scope; no precision/science freeze |

### 10.1 Frozen Git anchors

| Artifact | SHA |
|---|---|
| main branch before any S07 integration | `f262f6bea037580065a8505008773c04fdd259f5` |
| S00/S07-B common ancestor | `c9c84f8` |
| five S02-S06 non-FF integrations complete | `9fb1a9a` |
| first integrated S07-B delivery | `df13025bc6582b9b436d1df065de75c03e92782d` |
| audited cleanup code base | `4ce2366df2925161adae8fea393d5fca64836d40` |
| last old-chain production-source change | `bf480ea77ccf9ae8417c3ea58e933701dbc7222a` |
| frozen old S07-B endpoint | `e231808e77388d69053dcbced6e754dbe3468aef` |
| accepted S07-C anchor | `70bcd856f7ebb411eb2887e7ab71ef41ed13271f` |
| pre-rebaseline clean code/evidence anchor | `a2fc15e64898910b51b56b4b25c8579f459423bc` |
| accepted post-S07 rebaseline / current S00 anchor | `2a584053e6f6a3860b6f812681dc8d7342ca52ad` |
| owner-accepted reviewed S08 close-ready seal | `d31adea049c84e47a0e4f82f38f22a2ca91a5a6f` |

Old R1-R16 commits remain unmerged review evidence. Their exact identities remain
in Git and the frozen S07 handoff/review packages; they have no outgoing
implementation edge.

### 10.2 Preserved negative runtime evidence

These hashes remain canonical so compaction does not erase negative outcomes:

| Job | Terminal interpretation | Preserved SHA-256 |
|---|---|---|
| `348557` | failed/timeout; no full manifest | pytest log `eceba3ae66efdb901626eac108200bc9f50108229a290dad39dec64bd8abad2c` |
| `348818` | diagnostic complete; launcher noise plus real failures/fork hang | summary `892d335d528c8ea29c671a5152bbf919398882a622b6ade17e2d25b6334de9ff`; manifest `b794336a825b7a44eb8d22033bf4684fa43a93b7999f24a597b90d8d5999c835` |
| `349653` | stable-equal attribution only | summary `806afbfd41eabad3d2181c7c829a74f4ded34cef91636b5bdb7018b5fbbc36fc`; manifest `0c74aae4067bab74619269c16b38c8724ce38d56018d6dea035066e78528341c` |
| `351903` | focused gate failed; ZIP/model-task multiworker timeouts | summary `458d4a55b730cc375c15608d5b253752bd67454f6853e532a2d4ac66bad5a7e4`; manifest `d0d8ab44fde39f9b0149d3b1e21d375713b0fcb29da0018cba22e792a0582c3f` |
| `352105` | diagnostic complete; AF_UNIX/subreaper harness confounds | summary `0ea391ad8f85e7567ca3473082dd1d15c3c32383ef591cb77c5d13348d104a9b`; manifest `00ada336cac0e26f2d60423d425c11439289f96154b1df4e4a6611ea7c59eb6d` |
| `352354` | formal 9/9 but hidden worker SIGABRT warning; strict gate failed | summary `b8fd26b34d607510c9a3a3e90251709dce43f792b8956728845448e6837478e9`; manifest `67d723b37ca3a9d36af8bde75eab13765ca05bef1bd1fc6e2f08bbf87d3527ac`; warning `fb50d32d85c1f0cc24c27727d784c0ee7ceb045caf166294fd3869fd3bb62dbb` |
| `352718` | warning-fatal harness completed with seven timeouts; no retry | summary `52fb107d7e5b5d9bf8655685d568574abcf95280caea19b522c36758952437d6`; manifest `fd7b9492fd05a5be418a183c42d9d3ea3a530d1c86a4920ae7dcd274e68a2a9e` |
| `372819`/`373363`/`374142` | wrapper/Git-variable/warning/long-TMPDIR failures before model update | exact roots and hashes in `handoffs/S07-B-COMPLETE/{HANDOFF,RESULTS,RUN_REQUEST}.md` |
| `380806` | environment/FedAvg and C/L/F finite-loss backward reached; FP16 first-attempt gradients nonfinite; no accepted update | exact raw hashes in the S07-B-COMPLETE package |
| `389356` | bounded diagnostic: FP32 finite; FP16 scale 1 recovers C only; L/F sparse-path nonfinite remains | diagnostic log `6921efe9e39d25d7dc5fa6dfcab87a748d5db6040a4a49ab5a1fb3d5849edc16` |
| `390576` | final bounded FP32 clean gate passed 5/5 | final log `2c3cf8fc49c662aabae161b691b81d08fd20d131aa942e49ee0755ecd84e0cf9` |

No negative row is evidence of full model failure, and no passing row is evidence
of detector capability beyond its declared scope.

## 11. Anti-recovery and authority rule

No current or future contributor may import, copy, recover, cherry-pick, or treat
as an implementation/scientific contract any legacy T5/T6/T7 code, old defense
implementation, retired O-032-O-091 process decision, frozen e231 content, or old
`collab/**`/`cycle_04/**` decision. If a later clean task needs a similar
mechanism, it must be re-specified from the accepted clean foundation under a new
owner-approved envelope and independently reviewed.

Raw evidence remains in Git and `/nobackup`; it is not an active launcher, config,
implementation, or scheduling route.
