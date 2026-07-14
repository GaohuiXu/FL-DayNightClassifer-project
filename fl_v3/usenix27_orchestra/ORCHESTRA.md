# USENIX Security '27 Orchestra — clean CL to federated multimodal security

> **Rebaseline status (2026-07-14).** S07-C and S07-B-COMPLETE are closed.
> The current clean engineering anchor is
> `2a584053e6f6a3860b6f812681dc8d7342ca52ad` on
> `codex/s07-b-clean-completion`. It contains the accepted S01/S07-A and
> S02-S06 foundations, the reviewed legacy-security cleanup, and the bounded
> S07-B clean-completion evidence. It is not a full-training, performance,
> detector-capability, precision, Protocol-A/B, attack, defense, or scientific
> PASS.
>
> S08 implementation/remediation smoke is sealed and independently reviewed at
> `103c7389a47938b1f9dd0cba60251df6dce9e5bb` with R2 verdict
> `PASS_WITH_RESIDUAL_RISK`; all earlier smoke PASS/negative outcomes remain in
> the S08 handoff package. Under O-109, Q1 Job `431013` completed the eight-cell
> primary qualification in `00:04:02`, and Q2 Job `435151` completed the exact
> L-P020/F-CBGS compatibility gate in `00:03:56`. Q1+Q2 used `00:07:58` of the
> two-GPU-hour ceiling. The close-ready policy candidate is global FP16 for
> camera/pillar routes, global FP16 with SECOND/spconv explicitly kept FP32 for
> sparse LiDAR/fusion routes, and uniform FP32 as reference/fallback. Full sparse
> FP16 is not accepted as the unified F-capable route. Independent R3 reviewed
> evidence SHA `c0ef86235ead753fee3b790b19d40f82f875ec59` with
> `PASS_WITH_RESIDUAL_RISK` and no P0-P2 findings. S08 is close-ready for owner
> policy acceptance. No harness/work-chain expansion,
> merge, push, attack, defense, or S09 execution is authorized.
> Only after an accepted S08 precision policy may **S09 full-pipeline
> performance/readiness** begin. S10-S12 remain pending redefinition.
>
> Canonical companions: [`SESSIONS.md`](SESSIONS.md) and
> [`KICKOFFS.md`](KICKOFFS.md). `fl_v3/collab/**`,
> `fl_v3/docs/cycle_04/**`, the old e231 S07-B chain, and closed kickoff text are
> historical evidence only.

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
                  ├── S10 centralized branch/recipe ablation       [pending]
                  ├── S11 full CL capability and architecture freeze [pending]
                  └── S12 clean Protocol-A/B split/adaptation       [deferred]
                               │
                               └── later S13 threat model/attack
                                      └── S14 only after viable undefended attack
```

No deadline or milestone name creates execution authority. S10-S12 cells, gates,
seeds, resources, and exact ordering are not frozen until S08/S09 evidence has
been reviewed with the owner.

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
`precision=fp16`; current `s08.v1` instead requires an explicit fail-closed
`sparse_conv_precision` partition. S04 Job `341695` proved a bounded sparse module
path, not the current six-task optimizer seam.

### 3.2 S08 decision target

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
returned `PASS_WITH_RESIDUAL_RISK` with no P0-P2 findings. Owner precision-policy
acceptance remains.

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

## 4. S09 performance/readiness target

S09 begins only after the owner accepts the S08 precision policy. Its purpose is
to characterize and improve the full current training pipeline before expensive
CL capability work.

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

S12 will re-audit and materialize this protocol only after CL architecture freeze.
The old S12 proposal is not current authority.

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
- Full trainval `t1.v2` cache materialization remains pending exact owner approval.
  Historical Job `332651` `t1.v1` caches are coverage evidence only and forbidden
  production inputs.

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
- S08 qualifies precision; S09 qualifies performance/readiness; S10-S12 remain
  pending results-driven rebaseline.
- Persistent S00 is the default implementer; independent reviewer subagent or,
  when required, a separate review worktree supplies the quality gate.

### Still unresolved

| Decision | Latest freeze point |
|---|---|
| Any sparse-normalization or later Swin/LSS precision-boundary amendment beyond the accepted unchanged v1 architecture | only after reviewed S08 numerical evidence and a new owner architecture decision |
| Current six-task FP32/FP16/mixed precision policy | after reviewed S08 evidence, before S09 or any capability run |
| Production optimizer groups/LR/scheduler/clip/EMA/augmentation/sampling recipe | before S09 can be called final production readiness; otherwise S09 remains a labelled base-uniform pipeline gate |
| Single-GPU batch/accumulation/workers and whether DDP is needed | after reviewed S09 measurements, before full CL runs |
| S10 branch/recipe cells (`C-STR8`, `L-P020`, `L-S075`, `F-U`, `F-CBGS`, initialization) | after S08/S09, before an ablation request |
| mAP/NDS, fusion-gain, per-class, speed, memory, and selection gates | before the exact run whose outcome they judge |
| S11 seeds and CL-FREEZE interpretation | after S10 selection, before replication |
| `D_base`/`D_tail`, client unit, update scope, and clean Protocol-A/B cells | S12 after CL freeze, before split materialization/training |
| New threat model, attack, and defense | S13 only after accepted clean Protocol-B adaptation; S14 only after viable undefended attack |

## 9. Active owner-decision registry

O identifiers are never renumbered or reused. Only this table grants current
O-ledger authority; closed ranges below are provenance.

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
| O-109 | Set the persistent completion goal to finish Q1 primary, the minimal required Q2 L-P020/F-CBGS compatibility gate, independent evidence review, and a close-ready S08 linear commit state. Authorize all exact in-scope commits and Slurm submissions once their immutable tuples are recorded, with cumulative one-GPU elapsed allocation across all new Q1/Q2 jobs capped at two GPU-hours. Short earlier jobs leave only their unused elapsed budget for later jobs. The simplified O-107 mechanical workflow applies to obvious non-scientific defects, but no scientific cell/seed/data/resource expansion or silent reinterpretation is allowed. Forbid work-chain/harness expansion, merge, push, S09 execution, attacks, and defenses. | active S08 completion authority |

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
