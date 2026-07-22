# USENIX Security '27 Orchestra — clean CL to federated multimodal security

> **Active status (2026-07-22).** S07-S09 are closed. S10 Phase I-P is active on
> `codex/s10-phase1p-throughput-preflight`, created from the frozen control
> `codex/s10-phase1-branch-qualification` at
> `f1a2babda8dafd181b5a5144ab025a3f6be21cc2`.
>
> O-143 replaces S10's active six-stop execution order and per-job
> immutable/no-retry/multi-document/reviewer workflow. The current science order
> is **camera and LiDAR independent recipe/capability → staged fusion →
> capability gate → GH200 profiler/optimization**. Current-A2 and the old
> C→D→E→F route are paused.
>
> O-144 closes `P1-G0 PLAN_FREEZE` and makes
> `handoffs/S10/PHASE_I_PLAN.md` binding: physical B4 plus accumulation 8/
> effective B32; one ImageNet Camera primary and one scratch LiDAR primary;
> role-bound D_fit GT-paste; seed 0; 20 epochs; terminal-only selection; two
> total candidates; and five WPs, three owner gates and two approval envelopes.
> O-145 amends WP2/WP4 with an independent optimized CUDA BEV-pooling port or
> equivalent kernel, labelled fallback, FP32/FP16 forward/backward and policy
> parity, plus GH200 operator and aligned B4 end-to-end timing. The Camera
> checkpoint named by the reference YAML is ImageNet Swin-T, not NuImages.
>
> Prior evidence is retained, not erased: STOP-A's split/evaluator is reusable;
> STOP-B is `INCONCLUSIVE`; C1-A localized the observed large LiDAR-stem
> gradient to the current tiny-group GN path; C1-B bounded runs did not establish
> production capability or superiority over Alvis. Exact historical jobs and raw
> limits remain in `handoffs/S10/RESULTS.md` and the historical portion of
> `RUN_REQUEST.md`.
>
> O-146 activated the exact Envelope A at
> `e321aed749fd859c809199d52c30b2771dbef8b3`; O-147 amended its initial limits;
> O-148 then removed the numeric submission stop while retaining serial execution
> and the `1.10` GH200-hour ceiling. WP0-WP4 is now terminal after 12 submissions
> and `0.516389` GH200-hours. Camera correctness, checkpoint, parity, end-to-end
> and memory checks passed, but its optimized pooling ratio `0.976174` failed the
> frozen `<=0.80` promotion gate. LiDAR passed and
> emitted a directly consumable qualified config and zero-update recovery
> checkpoint. No capability metric or optimizer update ran.
>
> O-149 adopts a completion-oriented engineering-validation contract: after an
> owner approves a bounded validation objective and aggregate compute ceiling,
> diagnosed frozen-semantics bugs are repaired and rerun serially without a
> default numeric submission cap. Scientific cells and all material science/
> resource changes remain owner-gated.
>
> O-150 accepts the parity-qualified PyTorch sorted `segment_reduce` fallback as
> the Phase-I Camera production backend, retains the CUDA kernel as an unpromoted
> option, and removes the historical `1.25x` throughput target as a capability
> prerequisite. The later owner-approved Phase I-P preflight promoted Camera
> two-GH200 B16/rank and LiDAR one-GH200 B32 production recipes while preserving
> effective global B32, exposure and all other scientific boundaries. Revised
> Envelope B is materialized in `RUN_REQUEST.md` Section 7.4 at a `30.0` charged
> GH200-hour ceiling, concurrency one and serial LiDAR then Camera. Independent
> review of `a4f6ca86ddd966bdffc74a37af3337ac6675e83a` closed
> `PASS_WITH_RESIDUAL_RISK` with no open P0-P2; only a separate owner activation
> naming the review-sealed commit remains before submission.
> Staged fusion, merge, push, upload, publication and S11+ remain unauthorized.

## 1. Current objective and sequencing

The immediate scientific objective is a strong, trustworthy centralized
camera-LiDAR nuScenes detector. Clean modality capacity, numerical stability,
data ownership, aligned evaluation and useful absolute capability must precede a
federated security claim.

The intended paper question remains:

> During vendor-style federated adaptation of a strong camera-LiDAR detector to
> rare, geographically and environmentally non-IID fleet data, can a
> modality-localized backdoor hide among legitimate long-tail updates, and can a
> structure-aware defense remove it without rejecting the rare benign updates
> that adaptation exists to learn?

The active order is evidence-gated:

```text
accepted S08/S09 engineering foundation
                  │
                  ├── S10-CAM: two-GH200 recipe frozen; capability pending
                  ├── S10-LIDAR: B32 recipe frozen; capability pending
                  │      └── freeze qualified branch checkpoints
                  ├── S10-FUSION: staged fusion + aligned capability controls
                  │      └── absolute clean capability + fusion contribution gate
                  └── S10-PERF: final accepted graph/recipe profiling + optimization
                         └── only after capability passes
```

MIT BEVFusion's published branch pretraining, staged fusion and recipe choices are
strong reference anchors. S10 should verify local compatibility and capability,
not rerun published ablations without a concrete local conflict. The historical
Alvis detector is the practical capability comparator, subject to aligned dataset,
class, metric and evaluation semantics.

No deadline or milestone name creates execution authority. O-143 remains the
active S10 sequencing and collaboration rebaseline; O-144 freezes the Phase I
plan; O-122 through O-142 remain historical evidence and consumed authorities.
Envelope A is closed at its mixed Camera-negative/LiDAR-PASS engineering result.
O-150 resolves the Camera backend disposition. The exact measured `P1-G1`
Envelope-B tuple is frozen in `RUN_REQUEST.md` Section 7.4 and its independent
branch recipe-freeze review is closed with no open P0-P2; the next gate is separate
owner activation. S11+ remains undefined and pending.

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

Persistent S00 owns the active linear implementation context and canonical
orchestration documents. `Sxx` denotes evidence, not a requirement for a new
task, worktree, worker or reviewer.

For S10, O-143/O-149 establish phase-level collaboration:

1. agree once on phase objective, candidate cap, data/metric/seed policy,
   aggregate GPU-hours, submission policy/concurrency and escalation conditions;
2. implement and preflight with direct entry/config/checkpoint/one-batch checks;
3. diagnose and autonomously repair single-correct-answer conformance defects in
   tests/fixtures, config/schema parsing, dtype/API plumbing, runners, checkpoint
   I/O, artifact publication/provenance or logging, then resubmit serially inside
   the approved scientific/resource envelope;
4. use aggregate GPU-hours and concurrency as the default engineering controls;
   a numeric submission cap binds only when the owner explicitly sets one;
5. return to the owner for any scientific boundary/resource change, ambiguous
   diagnosis, recurrence of the same blocker after repair, or ceiling exhaustion;
6. keep one compact active handoff and one run ledger, with raw outputs and
   minimum scientific provenance;
7. review only data/metric changes, branch recipe freezes and the final
   staged-fusion/full capability result.

Do not create micro-handoffs, per-job snapshot trees, recursive manifests,
documentation commits for every incident, or reviewer chains for ordinary runner
bugs.

## 7. Compute, data, and execution boundaries

- Planning and implementation are not execution permission.
- O-009 covers only a recorded bounded engineering smoke: one node, at most one
  GPU, at most 60 minutes/job, one concurrent job, and two cumulative GPU-hours
  for the milestone. It never covers full cache/trainval coverage, 100/1000-step
  gates, model qualification/training steps, profiles, metrics, matrices, seeds,
  arrays, DDP, or publication.
- Outside an explicitly approved S10 phase, material compute retains its
  milestone's exact authorization rules; no spare-GPU expansion is inferred.
- Under O-143/O-149, a future S10 phase approval binds the objective,
  candidate/data/metric/seed boundaries, aggregate resources, submission policy,
  concurrency and stop conditions. Within a completion-oriented validation
  envelope, diagnosed frozen-semantics engineering repairs and serial
  resubmissions do not require per-job owner approval. There is no default numeric
  submission cap unless the owner declares one. Scientific or aggregate-resource
  changes do require owner approval.
- O-149 does not turn O-009 into automatic compute authority and does not cover
  capability training, scientific metrics/cells, profiles, matrices or seeds.
  Blind identical retries and spare-GPU expansion remain forbidden. The loop
  stops at objective completion, ceiling exhaustion, ambiguous diagnosis, a
  recurring same blocker, or any scientific-boundary change.
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

- Protocol B remains the primary security setting; Protocol A remains the clean
  optimization/control setting.
- S01-S09 data, evaluation, precision and engineering evidence remains accepted
  within its recorded limits.
- STOP-A's train-only split/evaluator artifacts remain the active reusable S10
  data/evaluation substrate.
- O-143 is the active S10 sequencing/collaboration decision. It replaces the old
  active six-stop order and S10 per-job process, but not prior raw results.
- Persistent S00 is the default implementer. S11+ remains pending.

### Still unresolved

| Decision | Next freeze point |
|---|---|
| Camera graph, initialization and production recipe | Phase-I camera qualification plan/result; compare against MIT/reference and aligned Alvis evidence where compatible |
| LiDAR graph, normalization, initialization and production recipe | Phase-I LiDAR qualification plan/result; C1-A `LOCALIZED_NORM` is diagnostic input, not a BN1d promotion |
| Training horizon, seeds, capability thresholds and aggregate Phase-I compute | next owner-approved phase envelope |
| Qualified C/L checkpoints and staged-fusion freeze/unfreeze policy | after branch recipe freezes |
| Absolute clean capability and fusion contribution | aligned staged-fusion capability gate; not inferred from D_low/D_select proxy runs |
| Final physical batch/accumulation and GH200 optimization | only after capability passes and the graph/recipe is frozen |
| Fair historical Alvis comparison | checkpoint/provenance plus aligned dataset, classes, metric and evaluator audit |
| S11+, Protocol A/B execution, attack and defense | pending future owner decision |

## 9. Active owner-decision registry

O identifiers are never renumbered or reused. Binding decisions are cumulative;
earlier rows preserve the exact authority/state when issued and never reauthorize
consumed compute. O-143 supersedes O-122's active S10 order and O-124's active
per-job process, but does not reinterpret closed S08/S09 or prior S10 evidence and
does not authorize S10 execution.
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
| O-126 | Approve the corrected STOP-A scientific protocol and serial A1-A4 completion goal. Retain all frozen input identities, log-level ownership, quotas, sample/support/prevalence/dominance bounds, raw-dependency leakage proof, official-filter semantics and exact full-val evaluator parity, but remove the global balance-optimality hierarchy and sorted-log lexicographic certificate. Base and nested construction each execute exactly one zero-objective feasibility MILP; before either solve, freeze the input/feature hash and real-candidate ordinal `1`; accept and hash only that first feasible assignment, with no reroll, candidate shopping or cross-environment optimality claim. This produces a scientifically constrained, reusable limited-rung proxy split, not a balanced-optimal split or official benchmark. Authorize S00 to implement/validate/commit A1-A2, freeze and submit exactly one serial aarch64 CPU-only A-GATE on the `gpu` partition with no GPU GRES, 4 CPUs, 32 GiB and `00:15:00`, then seal evidence and obtain independent high-risk review in A4. Any failure/timeout returns to the owner with no retry; STOP-B/C do not start before reviewed STOP-A PASS. Exact Job `468295` was site-transformed to four GPUs and protection-cancelled after 8 seconds before execution; the sole submission is consumed. | consumed at scheduler-resource boundary / STOP-A blocked pending owner amendment |
| O-127 | After Job `468295` proved that Arrhenius `job_submit/lua` converts an omitted/zero GPU request on the only accessible aarch64 partition into four GPUs, approve exactly one resource-only replacement: explicitly reserve `1 x nvidia_gh200_120gb`, 4 CPUs, 32 GiB and `00:15:00`, then force `CUDA_VISIBLE_DEVICES=""` and fail unless PyTorch exposes zero CUDA devices. Preserve the exact O-126 data, scientific constraints, evaluator gate, candidate ordinal, no-reroll rule and single real attempt. Authorize the minimal runner/docs assertion change, local validation, immutable commit/snapshot/tuple, one submission, evidence sealing and independent A4 review. The replacement may allocate at most `0.25` GPU-hours; no retry, B/C, model/training, D/E/F, merge or push is authorized. Job `468404` consumed the tuple, completed `0:0` in `00:07:59`, used `0.133056` allocated GPU-hours with zero process-visible CUDA devices and passed the split/checker/parity gate. A4 re-review of `b0478a2` returned `PASS_WITH_RESIDUAL_RISK`, no open P0-P3, and STOP-A closure GO. | consumed / STOP-A CLOSED PASS / B-C unstarted |
| O-128 | Approve the exact STOP-B observation-first plan and compute request as one continuous stop: implement/freeze output-neutral diagnostics on the unchanged current graph; bind the reviewed STOP-A `D_low` role and a pre-model deterministic `P_core48/P_term16` panel; run one serial one-GH200 B-DIAG at physical B4 in FP32 and accepted global-FP16/SECOND-FP32, with exactly one criterion-only four-B1 aggregation decomposition, no optimizer/update/evaluator; exit honestly `LOCALIZED` or `INCONCLUSIVE`; and obtain stop-level independent review. Authorize the exact `8fd832d` B-DIAG tuple recorded in `handoffs/S10/RUN_REQUEST.md` at 8 CPUs, 64 GiB and `00:30:00`, plus zero or one later B-REFINE only if its predeclared trigger fires and its separate immutable tuple remains inside STOP-B's two-hour/O-124 aggregate caps. No identical retry, panel reroll/growth, hypothesis expansion, model/loss/precision/recipe change, training, official evaluation, STOP-C execution, D/E/F, merge, push or upload. Job `477892` consumed the sole B-DIAG submission and failed `1:0` after `00:04:44` at the first FP32 disabled/on parity gate, after 39 focused tests and pre-model panel/W0 freeze but before broad/term observations. The runner did not persist which parity predicate failed, so no localization verdict or B-REFINE trigger exists; no automatic replacement is authorized. | consumed B-DIAG / early parity FAIL / owner decision required |
| O-129 | Approve `RUN_REQUEST.md` §20's parity-gate remediation and continue STOP-B. The replacement must reuse the exact physical Job-477892 panel and unchanged F-U A0 W0, perform one no-update disabled warm-up per precision, persist disabled-0/disabled-1/enabled exact predicates plus fixed numerical gradient comparisons on the two predeclared B4 parity batches, and enter broad/term/localization only after every FP32 and FP16 parity gate passes. Authorize exactly one immutable replacement B-DIAG on one GH200, 8 CPUs, 64 GiB and at most `00:30:00`/`0.5` GH200-hour, plus zero or one B-REFINE only if the unchanged predeclared localization trigger fires, at most `00:15:00`/`0.25` GH200-hour. The total new authority is at most `0.75` GH200-hour. No identical retry, panel reroll/growth, tolerance fitting, hypothesis expansion, model/loss/precision/recipe change, training, evaluator, STOP-C execution, DDP/array/spare GPU, merge, push or upload. Exact replacement source `43f157b` Job `478250` passed 41 tests and all identity gates, then failed the first repeated-disabled FP32 parity as `baseline_instability` after `00:04:28`/`0.074444` GH200-hour; no later cell ran and B-REFINE is false. Independent review returned `PASS_WITH_RESIDUAL_RISK` with no open P0-P3 and requires owner rebaseline. | replacement consumed/reviewed / bounded FAIL sealed / no executable compute |
| O-130 | Accept that the current trainable Swin-T graph contains intended stochastic depth and that long-run training need not be byte-identical. Approve one bounded STOP-B B-RAND amendment and its compute: reuse only the exact Job-477892 panel's first `P_core` B4 token vector; FP32/no-update/no-evaluator; C-STR8, L-S075 and F-U each run one warm-up, five fixed-seed repeats at seed `10000`, and five varying-seed probes `11000..11004` (33 detector forward/backward runs total). Record loss-relative difference, output relative-L2/cosine, parameter-gradient relative-L2/cosine and prefix distributions. Exact hashes remain evidence but are not acceptance gates. Integrity requires exact config/panel/W0/state binding, five identical post-run RNG hashes in the fixed-seed group, finite loss/output/gradients, stable missing-gradient sets, exactly 33 runs and complete artifacts. Classification is descriptive only: camera stochasticity, LiDAR runtime variation, fusion-only interaction, or mixed/inconclusive; it grants no architecture/recipe change or automatic localization continuation. Authorize implementation, linear commits, one immutable snapshot/tuple, one GH200, 8 CPUs, 64 GiB, `00:15:00`, at most `0.25` GH200-hour, one submission/no retry, evidence sealing and one independent reviewer. | Job `479667` integrity PASS; review `02ba3b4` PASS_WITH_RESIDUAL_RISK / STOP-B CLOSED INCONCLUSIVE |
| O-131 | Approve and execute one integrated STOP-C0 observation-first training rung. Run in order `C0-F-A1` (ImageNet1K V1 Swin, random LiDAR/fuser/head), `C0-L-A0` (random L-S075), and `C0-F-A0-P64` (all-scratch fusion negative control). The first two use the exact STOP-A `D_low`, physical B4, `drop_last=true`, 1,538 attempted windows/6,152 attempted samples and one terminal `D_select` internal mAP/NDS; the negative control uses 64 attempted windows and no evaluator. All use seed 0, accepted global FP16/SECOND-FP32, AdamW `1e-4/.01`, constant LR, uniform sampling, and no clip/EMA/augmentation/CBGS/GT-paste. Sample true-unscaled gradients and realized update/weight at attempts `1,4,16,64,256,768,1538` (short control through 64), record chunk loss/timing/memory and one early 10-active-window A1 operator trace. Gradient magnitude alone is not a failure; correlated harm requires at least two predeclared overflow/update/trajectory indicators. Authorize focused tests, linear implementation/evidence commits, exact immutable snapshot/tuple, one serial GH200 with 16 CPUs/96 GiB/`01:00:00`, at most one GH200-hour, no retry, and one independent reviewer. This does not select a recipe, promote an architecture, run official val, execute later C strong contrasts or D/E/F, or authorize merge/push/upload. Job `492525` consumed the sole allocation and is terminal `FAIL/INCOMPLETE`; no retry is executable. Remediation `09c3945` received evidence-integrity `PASS_WITH_RESIDUAL_RISK` with no open P0-P3, without changing the incomplete execution verdict. | consumed C0 compute / reviewed incomplete evidence package |
| O-132 | Reject closure of the incomplete C0 by further reviewer churn and authorize the full C0-v2 clean-replay option. Supersede only O-131's no-retry boundary for one exact replacement from source `2262b4063a3e419b17f4b911a9e11a7ff50ea784`: unchanged F-A1/L-A0 full `D_low` B4 epochs and `D_select` evaluations plus F-A0-P64; unchanged seed, data, precision, baseline recipe, diagnostics and early profile; one GH200, 16 CPUs, 96 GiB, `01:00:00`, at most one GPU-hour. Require v2 schema, actual collated-token provenance, identical F/L order/remainder hashes, valid trainable-prefix health, all three cell summaries, aggregate PASS and artifact checksums. One submission, no retry; any failure returns directly to the owner. No intermediate reviewer chain, later-C/D/E/F, merge, push, upload, publication, attack or defense is authorized. Job `496312` consumed the sole tuple, completed `0:0` in `00:45:15`, passed 80 tests/3 skips, all three v2 cells, matched actual F/L order/remainder, aggregate PASS and 28/28 artifact checks. | consumed / bounded C0-v2 execution PASS / later C owner-gated |
| O-133 | Accept the C1-A and C1-B scientific plans. C1-A uses the frozen STOP-B L-S075 B4 panel in FP32 to compare current sparse GN with direct-reference BN1d under both the normal detection loss and a frozen SECOND-output upstream-gradient VJP; it may exit localized or inconclusive and does not automatically promote BN1d. Before C1 FP16 scientific cells, qualify one common conservative GradScaler init scale on a frozen no-update B4 and bind it to provenance. C1-B admits current A1, current staged-L A2 and a coherent MIT-derived staged A2 through fresh G20, matched D_low and at most three D_mid lineages, with at most two STOP-C survivors. A counterfactual replaces rather than expands a slot. The exact MIT anchor graph/init/component package remains owner-pending. This decision authorizes planning and canonical documentation only, not implementation, commit, checkpoint acquisition, review or compute. | active C1 design gate / no executable tuple |
| O-134 | Relax the earlier conditional restriction on considering BN1d, TransFusion and LiDAR-conditioned DepthLSS, but make C1-B sequential: run the current graph's matched A1/A2 first, inspect the frozen result, and open a bounded MIT-reference-guided repair only if a future predeclared gate finds the current setting/structure materially worse. Do not run or implement C1-B under this decision. Authorize C1-A implementation, focused validation, linear commits, one detached read-only snapshot and exactly one no-retry Slurm job: complete accepted STOP-B L-S075 panel (16 disjoint physical-B4 batches/64 samples), uniform FP32, current GN versus direct MIT BN1d with exact shared convolution/affine W0, normal loss plus coordinate-derived fixed SECOND-output VJP, two repeats per candidate/path/batch (128 total runs), no optimizer/update/evaluator, one GH200, 8 CPUs, 64 GiB, `00:30:00`, at most `0.5` GH200-hour. Job `502456` passed 36 tests, then failed before candidate forward/backward because the runner incorrectly required `num_batches_tracked` in `load_state_dict(strict=False).missing_keys`; PyTorch synthesized it and reported only running mean/variance. No gradient verdict exists; `0.050833` GH200-hour consumed; no retry executable. | consumed / pre-execution runner assertion FAIL / owner decision required |
| O-135 | Correct the exact Job-502456 BN1d state-mapping assertion defect. Accept only `running_mean/running_var` in PyTorch's reported missing-key set; separately require all expected `num_batches_tracked` buffers to exist and equal zero; retain zero unexpected keys and exact trainable-parameter identity; add a direct regression fixture reproducing PyTorch's compatibility behaviour. This is model/loss/gradient/data/protocol neutral. No commit, snapshot, Slurm replacement, retry, C1-B or later-stop execution is authorized by this terse fix instruction. | exact remediation committed at `d713bfe`; execution authority supplied separately by O-136 |
| O-136 | Accept exact remediation commit `d713bfe3b5e5c587f58ce70721b2b6eea0b050ec` and continue C1-A with one strictly derived replacement. Preserve the exact O-134 16-disjoint-B4/64-sample STOP-B panel, current-GN/direct-BN1d candidates and shared W0, uniform FP32 loss plus fixed-VJP pathways, two repeats/128 runs, metrics, thresholds, no optimizer/update/evaluator, and one GH200/8 CPUs/64 GiB/`00:30:00`/`0.5`-GH200-hour ceiling. Authorize canonical records, one new detached read-only snapshot and exact tuple, one submission, monitoring and evidence sealing. Any failure or identity/scope/resource drift returns directly to the owner; no retry, C1-B, later C/D/E/F, merge, push or upload. Job `502572` completed `0:0` in `00:03:09`, passed 37 tests, all identities, 128/128 runs and artifact gates, and returned bounded `LOCALIZED_NORM`. | consumed / C1-A execution PASS + localized normalization mechanism / no C1-B authority |
| O-137 | Activate only C1-B0 as a matched current-A1 fusion-training health observation. Authorize a production-resolved SECOND normalization/checkpoint seam with GN default and explicit BN1d, local tests and linear commits; freeze one label-blind 1024-token `D_low` H256 vector; run serial `F-A1-GN-H256` and `F-A1-BN1D-H256` from exact shared seed-0 trainable W0 with ImageNet1K V1 camera, physical B4, global FP16 plus SECOND FP32 island, common no-update-qualified GradScaler scale 32, AdamW `lr=1e-4/wd=.01`, constant scheduler and no augmentation/EMA/CBGS/GT-paste/clip. Each cell must complete 256 real updates and record all-window loss/scaler/memory/basic timing plus true-unscaled gradients and realized updates at 1/4/16/64/128/256 and BN running state. Authorize one detached read-only snapshot, exact tuple, one GH200/16 CPUs/96 GiB/`00:30:00`, at most `0.5` GH200-hour, one submission and evidence sealing. Any identity, qualification, finite-gradient, matched-token/update or artifact failure stops with no retry. Scientific weakness is retained as evidence, not tuned. No evaluator/checkpoint selection, full `D_low`, C1-B1/A2/MIT repair, later stop, reviewer chain, merge, push or upload. Exact implementation `96ae63d`, detached snapshot and §32 tuple were frozen. Job `502958` failed pre-model after 100 passed/6 failed tests because of misplaced test lines and one incomplete `s10.v1` fixture; no model, H256 data or update ran. | consumed / pre-model test-fixture FAIL / owner decision required |
| O-138 | Approve only the exact Job-502958 test-neutral remediation: return the three operator-profile hash assertions to the preceding S09-v2 test and add required `grad_scaler_init_scale=32` to the migrated `s10.v1` fixture. Authorize local/static validation, a linear commit, one new detached read-only snapshot and exact tuple, then one strictly derived C1-B0 replacement with unchanged production source/runner/config, data, candidates, seed/W0, precision/recipe, H256 horizon, diagnostics, gates and one-GH200/16-CPU/96-GiB/`00:30:00`/`0.5`-GH200-hour resource ceiling. Fresh output, one submission, no retry. No C1-B1, later C/D/E/F, reviewer chain, merge, push or upload. Exact remediation `0f51e11` and §33 tuple were frozen. Job `503075` failed pre-model after 105 passed/1 failed tests because the migrated `s10.v1` fixture also lacked required `execution.operator_profile`; the first missing scale field had masked this second omission. No model, H256 data or update ran. | consumed / second pre-model fixture FAIL / owner decision required |
| O-139 | Replace the failing test's partial manual promotion from `s09.v1` to `s10.v1` with the already validated complete `s10_second_config(..., "batch_norm_1d")` fixture and assert resolved schema, normalization, scale and operator-profile propagation before the production detector constructor. Authorize local/static schema audit, one linear commit, one detached read-only snapshot/exact tuple and one strictly derived C1-B0 replacement with unchanged production source/runner/config, data, cells, W0/seed, precision/recipe, H256 horizon, diagnostics, gates and one-GH200/16-CPU/96-GiB/`00:30:00`/`0.5`-GH200-hour ceiling. Fresh output, one submission/no retry. No C1-B1, later C/D/E/F, reviewer chain, merge, push or upload. Exact remediation `5de019b` and §34 tuple were frozen. Job `504508` completed `0:0` in `00:06:57`; 106 tests and both 256-update cells passed all hard gates. BN1d strongly reduced LiDAR gradients but had higher matched H256 loss; no evaluator or selection exists. | consumed / C1-B0 execution PASS / C1-B1 owner-gated |
| O-140 | Activate exactly one C1-B1 current-A1 matched capability job: GN and BN1d share exact seed-0 trainable W0 and actual shuffled `D_low` token order/remainder, each completes 1,538 physical-B4 updates with scale 32 under the unchanged C1-B0 graph/init/precision/AdamW/constant/no-augmentation/no-EMA/no-CBGS/no-GT-paste/no-clip recipe, saves only its terminal checkpoint and evaluates exact frozen `D_select` (4,626 samples/eight logs). NDS is primary; mAP, per-class AP, numerical health and paired leave-one-log-out jackknife uncertainty are guardrails. Because no numeric superiority/non-inferiority margin is approved, execution may PASS but scientific selection must remain `OWNER_DECISION_REQUIRED`; no automatic normalization promotion. Authorize implementation, focused tests, linear commits, one detached read-only snapshot/exact tuple, one serial one-GH200/16-CPU/96-GiB/`01:00:00` submission capped at `1.0` GH200-hour, evidence sealing and then stop. No retry/requeue, intermediate checkpoint selection, new seed, D_audit/official val, A2/MIT repair, later stop, reviewer chain, merge, push or upload. Job `504921` consumed it in `00:47:01`: GN passed 1,538 updates; BN1d had one first-window scale-32 head-gradient overflow and only 1,537 accepted updates. Both D_select point estimates exist and favor GN, but paired-log uncertainty is absent and matched exposure failed. | consumed / C1-B1 FAIL-INCOMPLETE / owner decision required |
| O-141 | Replace a mechanical BN1d-B4 completion with one operational BN1d physical-B8 candidate. Preserve the exact seed-0 trainable W0, frozen shuffled D_low order and the same first 6,152 consumed samples/three-token remainder; run 769 accepted B8 updates with initial GradScaler scale 8, fail-fast numerical boundaries 1/4/16/64 then 256/512/769, unchanged FP16+SECOND-FP32/AdamW `1e-4/0.01`/constant/no-augmentation/no-EMA/no-CBGS/no-GT-paste/no-clip recipe, terminal checkpoint only, and exact D_select evaluation at physical B4. Compare descriptively and with paired eight-log uncertainty against sealed GN-B4 and incomplete BN1d-B4 outputs; report throughput/memory. Interpret as a joint BN1d+B8+scale8 operating point, not isolated batch causality or automatic architecture selection. Authorize focused implementation/tests, linear commits, one detached read-only snapshot/exact tuple, one serial one-GH200/16-CPU/96-GiB/`00:30:00` job capped at `0.5` GH200-hour, evidence sealing, then stop. No retry/requeue, GN/B16/new seed/profiler/recipe search/D_audit/official val/A2/MIT repair/later stops/reviewer chain/merge/push/upload. Exact implementation `e4a9ff4`, detached read-only snapshot and §36 tuple were frozen. Job `505266` passed 121 tests then failed pre-model on a wrong ResolvedConfig attribute; no B8 evidence exists. | consumed / pre-model implementation assertion FAIL / owner decision required |
| O-142 | Correct only Job-505266's schema access from nonexistent `config.schema_version` to canonical `config.data["schema_version"]`; add a regression that resolves the exact BN-B8 config and directly invokes `_assert_config`. Authorize static validation, one linear remediation commit, one new detached read-only snapshot/exact tuple and one fresh-output replacement with unchanged O-141 model, data, W0, tokens/remainder, physical B8, scale8, 769-update fail-fast gates, B4 D_select evaluator, comparisons, one-GH200/16-CPU/96-GiB/`00:30:00`/`0.5`-GH200-hour ceiling. One submission/no retry; no other code/science/resource expansion, reviewer chain, merge, push or upload. Exact remediation `864f704` and tuple `d98eaec` were consumed by Job `505316`. All 769 B8 updates and D_select point metrics completed, but tail paired-log computation exceeded the wall limit (`TIMEOUT`, `00:30:07`); paired artifacts, aggregate summary and runner manifest are absent. Raw `recipe.physical_microbatch` also carries a legacy B4 display error despite resolved/token B8 identities. | consumed / FAIL-INCOMPLETE / owner decision required |

| O-143 | Replace the active S10 six-stop execution order with: camera and LiDAR independent recipe/capability qualification; staged fusion from qualified checkpoints; aligned absolute-capability and fusion-contribution gate; GH200 profiling/optimization only after capability passes. Pause current-A2 and the old C→D→E→F path. For S10, replace per-job immutable/no-retry/multi-document/reviewer mechanics with one owner-approved phase envelope, autonomous output-neutral engineering remediation within its aggregate compute/submission cap, one compact active handoff and one job ledger. Preserve raw outputs and minimum run provenance. Return to the owner for any model/data/recipe-space/metric/seed/candidate/resource change or repeated engineering failure. Review data/metric changes, branch recipe freezes and final fusion/full results only. | active S10 science/collaboration rebaseline; documentation only; no compute authority |
| O-144 | Freeze `handoffs/S10/PHASE_I_PLAN.md` as the binding Phase I C/L plan. Select physical B4 plus accumulation 8/effective B32; exact standalone Camera with ImageNet-1K Swin-T, reference six-task CenterHead and Camera recipe; scratch keyframe-train LiDAR with reference BN, SECOND `[5,5]`/SECONDFPN/TransFusionHead and LiDAR recipe; exact role-bound D_fit CBGS/GT-paste; seed 0; 20 epochs; epoch-20 terminal-only selection; one terminal D_select evaluation and owner-unsealed one-time D_audit; exactly two initial candidates; and the five-WP/three-gate/two-envelope execution model. Require all later work to follow that plan unless explicitly amended. This decision closes P1-G0 plan freeze only and does not activate Envelope A or B; it authorizes no implementation, checkpoint acquisition, GTDB materialization, commit, GPU/Slurm execution, merge, push or upload. | active Phase-I science freeze; amended by O-145/O-150; exact Envelope-B request frozen, owner activation pending |
| O-145 | Amend O-144 WP2/WP4 to require an independent in-tree port of the pinned MIT optimized CUDA BEV-pooling operation, or a functionally equivalent kernel, with no mmdet3d/mmcv runtime dependency; retain a labelled reference fallback; require geometry/shape, FP32/FP16 forward/backward and accepted-precision-policy parity before production use; and measure GH200 operator plus aligned physical-B4 end-to-end timing in WP4. Clarify that the initial Camera checkpoint URL is the reference YAML's ImageNet `swin_tiny_patch4_window7_224.pth`, not `swint-nuimages-pretrained.pth`. Authorize the plan-amendment commit and exact Envelope-A drafting only. Do not activate implementation, checkpoint acquisition, GTDB materialization, GPU execution, merge, push or upload. | consumed qualification amendment; production disposition superseded by O-150 |
| O-146 | Activate the exact S10 Phase-I Envelope A recorded at commit `e321aed749fd859c809199d52c30b2771dbef8b3` and authorize S00 to execute WP0-WP4 continuously inside its candidate, data, checkpoint, correctness, remediation, output, three-submission and one-GH200-hour boundaries. This does not activate Envelope B, capability metrics/evaluation, merge, push, upload or publication. | consumed; amended by O-147/O-148 and closed at the O-148 terminal outcome |
| O-147 | Amend Envelope A at commit `c45e020ed16496e2acaa5f8d34b135da21fb1230`: raise the total submission cap from three to five and aggregate ceiling from `1.0` to `1.10` GH200-hours; allow only one fresh-output Camera replacement followed serially by original Job B, with all data/seed/config/tolerances/gates/per-job resources and prohibitions unchanged. Any failure stops. | consumed amendment; superseded prospectively by O-148 after Job D failed pre-control |
| O-148 | For the remaining WP4 engineering validation, remove the numeric submission limit while retaining maximum concurrency one and the unchanged `1.10` GH200-hour ceiling. Require S00 to diagnose, minimally repair and immediately resubmit each clearly engineering/config/schema/test/runner/dtype/checkpoint/artifact defect until Camera and LiDAR Job A/B reach honest terminal outcomes; do not change candidate science, data, seed, config semantics, tolerances, performance gates or aggregate resources. Envelope A closed after 12 submissions and `0.516389` GH200-hours: Camera negative at the frozen pooling-promotion gate; LiDAR PASS. | consumed Envelope-A completion authority / mixed terminal outcome |
| O-149 | Replace mechanical per-error approval for owner-approved engineering validation with a completion-oriented, aggregate-budget contract. The approval binds objective/exit gate, frozen science, data/command family, per-job resources/wall limit, aggregate GPU-hour ceiling, concurrency, fresh outputs and escalation boundaries. Submission count has no default numeric cap unless explicitly set. S00 diagnoses, records and repairs unambiguous frozen-semantics defects—including config/schema parsing, dtype/API, fixtures/runners, checkpoint/artifact/provenance/logging—and resubmits serially. Blind identical retries remain forbidden. Stop and return for ceiling exhaustion, recurring same blocker, ambiguous diagnosis, or any candidate/model/data/recipe/precision/evaluator/metric/seed/gate/scientific/resource change. Scientific/capability runs retain separate approval. | active collaboration/engineering-validation contract; no standing compute authority |
| O-150 | Accept the numerically qualified PyTorch sorted `segment_reduce` fallback as the Phase-I Camera production backend; retain the CUDA pooling kernel as an explicit unpromoted optimization; preserve Job H's historical `0.976174` result without continuing to use the unmet `1.25x` target as a capability prerequisite. Start Envelope-B preparation with all graph/data/recipe/precision/evaluator/seed/exposure/candidate boundaries unchanged. The first scientific submission still requires an exact aggregate GH200-hour ceiling and the branch recipe-freeze review. | Phase-I amendment retained; original 49.0-hour request is historical, revised Section 7.4 review is closed, owner activation pending |

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
