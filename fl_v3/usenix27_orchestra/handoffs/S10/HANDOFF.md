# S10 HANDOFF — accepted six-stop plan, STOP-A A3 PASS pending A4 review

## 1. State and authority

```text
SESSION: persistent S00 / S10
BASE_SHA: a080d49c1c22de20ccb5b1353d4922c7df14a729
BRANCH: codex/s10-cl-model-recipe
OWNER_DECISION: O-122 scientific envelope; O-124 ABC completion; O-125 legacy optimizer consumed; O-126 corrected STOP-A; O-127 one-GPU/CUDA-hidden replacement
PLAN_STATE: six-stop A-F scientific envelope accepted
CURRENT_AUTHORITY: O-127 A3 tuple consumed; A4 docs remediation and targeted re-review
ABC_IMPLEMENTATION/COMMIT/SLURM/REVIEW_WORKTREE: O-127 active only for STOP-A evidence/A4; B/C remain gated
STOP_A: A3 engineering gate PASS; initial A4 REMEDIATE on docs only; targeted re-review pending
STOP_B/C: unstarted because STOP-A exit gate is unmet
STOP_D/E/F_EXECUTION: not authorized
MERGE/PUSH/UPLOAD/PUBLICATION/S11+: not authorized
```

O-122 sets the primary full claim to **absolute clean capability + fusion
contribution**. Limited-rung internal metrics may select work; they cannot make
that claim. Official val remains sealed from selection and is opened only at the
frozen STOP-F gate. An official-val failure is recorded as a negative result and
amendment point; it does not permit fallback checkpoint selection.

## 2. Accepted stop topology

| STOP | Accepted purpose | Exit boundary |
|---|---|---|
| A — Split/Metric | one-shot frozen train-only nested feasibility split, ownership proof, internal evaluator and exact full-val parity gate | immutable split/evaluator identities independently accepted |
| B — Observation-first | locate or bound the current large true LiDAR-gradient mechanism without changing model math | `LOCALIZED` or honestly `INCONCLUSIVE`, with output-neutral evidence |
| C — Architecture/Initialization | strong current-vs-reference comparison, joint-vs-staged initialization, bounded causal counterfactuals | at most two graph/init families survive |
| D — Recipe/Production Freeze | select the production optimizer/schedule/EMA/augmentation/sampling/batch/exposure bundle on `D_select`, bind `candidate_freeze.json`, then open `D_audit` exactly once | one final graph/init/recipe or an honest `INCONCLUSIVE` result; no audit-driven reselection |
| E — Final-graph GH200 Optimization | profile and sustainably optimize only the accepted graph/recipe | output-neutral optimized final graph requalified |
| F — Full/Official Val/Close | one single-seed primary full train, matched modality controls, sealed official val and S10 close | absolute capability and fusion-contribution result, including negative outcome; no reuse of `D_audit` |

S11 and later remain undefined. STOP-F contains S10's full run but is not
authorized by this plan acceptance.

## 3. STOP-A exact split gate

### 3.1 Immutable input identity and initial re-attestation

Bind only the owner-accepted S09 train/val `t1.v2`, `v1.0-trainval`,
`n_sweeps=10` caches and ZIP manifest:

- train: 28,130 samples; canonical SHA-256
  `310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a`;
  pickle SHA-256
  `57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30`;
  sidecar SHA-256
  `f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c`;
- val: 6,019 samples; canonical SHA-256
  `bb692de4c1eb8b66e8c74f4e807eb208ad891b45ce8f233e8017dc4f3a3b6e2f`;
  pickle SHA-256
  `d4ed7aee9978c2294e2087c917006cbb3d69276453266d0f9c92591340084837`;
  sidecar SHA-256
  `4f5390815720e14625be31b20fb1596cafe9869ad95b08dc098aea65413be432`;
- ZIP manifest logical/physical SHA-256:
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6` /
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.

Before solving, re-attest that the accepted train cache maps to exactly 50 logs
with location counts:

| location | logs |
|---|---:|
| `boston-seaport` | 22 |
| `singapore-onenorth` | 17 |
| `singapore-queenstown` | 7 |
| `singapore-hollandvillage` | 4 |

Any identity, count or mapping mismatch is a blocker; do not repair it by changing
quotas or choosing another seed.

### 3.2 Frozen ownership split

All assignment is at **log** level; scenes, keyframes, annotations, instances and
raw sensor dependencies inherit that owner.

| role | logs and location quota | sample-size gate |
|---|---|---|
| `D_fit` | 34 = `16/11/5/2` | 67–73% of train keyframes |
| `D_select` | 8 = `3/3/1/1` | 12–18% |
| `D_audit` | 8 = `3/3/1/1` | 12–18% |
| `D_low ⊂ D_mid ⊂ D_fit` | 10 = `5/3/1/1` | 27–33% of `D_fit` |
| `D_mid ⊂ D_fit` | 20 = `9/7/3/1` | 57–63% of `D_fit` |

`D_select` is the repeated train-only selection evaluator. At the terminal
STOP-D gate, `D_audit` is opened exactly once only after `candidate_freeze.json`
binds the candidate/checkpoint/recipe and decision rule. A ranking reversal is
reported as `INCONCLUSIVE`; it does not permit reselection or a second audit.
STOP-E/F never read `D_audit`. Official val is never a member of these roles.

For every nuScenes detection class, each of `D_select` and `D_audit` must contain:

- 8–22% of full-train positive frames and 8–22% of full-train eligible boxes;
- at least 100 positive frames, 200 eligible boxes, 15 scenes and 3 logs;
- no one log contributing more than 50% of that role's class boxes.

For every class, `D_low` requires at least 50 positive frames, 5 scenes and 2
logs, with prevalence 0.5–1.5x `D_fit`; `D_mid` requires at least 100 positive
frames, 10 scenes and 3 logs, with prevalence 0.65–1.35x `D_fit`.

### 3.3 One-shot feasibility constructor and checker

Use a no-seed integer/MILP formulation for hard constraints only. Before solving,
emit and hash the exact ordered log-feature table, all source identities and real
candidate ordinal `1`. The base ownership and nested-rung problems each run
exactly once with a constant-zero objective. Accept only the first solver result
that certifies an integral feasible assignment, immediately hash it, and never
reroll or compare an alternative assignment. The zero-objective optimum is only
a feasibility certificate; it is not evidence that class/context balance is
globally optimal, lexicographically minimal, or invariant across solver versions.

This is scientifically meaningful because all predeclared ownership, location,
sample-volume, class-support, prevalence and dominance constraints remain hard,
and downstream cells reuse one immutable split. It is a limited-rung proxy design,
not an official benchmark or a claim that all feasible splits are equivalent.

An independent checker must reconstruct every constraint from emitted ownership,
not trust solver summaries. The checker proves disjoint ownership across log,
scene, sample, annotation, instance, six camera paths, key LiDAR, all referenced
sweeps and official val. Any overlap fails STOP-A.

Keep `training_support` (cache `gt_in_range`) distinct from
`evaluation_support` (official devkit class/range filters, zero lidar+radar-point
filter and bicycle-rack filtering). Do not silently use one as the other.

Required frozen outputs:

```text
split_protocol.json
pre_solve_identity.json
log_features.jsonl
sample_ownership.jsonl
split_manifest.json
leakage_report.json
candidate_freeze.json  # intentionally absent/locked until the D_audit gate
sha256sums.txt
```

GT databases are role- and manifest-bound. Save source identities and build a
separate database per promoted rung only when STOP-D needs GT paste; until that
seam is implemented and accepted, GT paste remains off. CBGS is also off. STOP-A
provides only a CBGS identity seam that hashes the token order already supplied
by its caller and the expanded index; it does not itself verify an expected
manifest SHA or role. Before STOP-D may enable CBGS, the production caller must
first restrict the dataset to the accepted role manifest, verify the manifest
SHA/role/expected token identity, then derive and record the CBGS index identity,
with focused fail-closed tests. STOP-A does not prebuild unused GT databases or
claim that this future caller integration is already complete.

## 4. STOP-A exact evaluator gate

The existing official `v1.0-trainval -> val` path stays unchanged. Add a separate
`internal_train_manifest` path whose output is named
`internal_subset_NDS/mAP`, records `official=false`, and is explicitly proxy-only.

Ground truth must be obtained with the nuScenes devkit `load_gt` on the parent
split, restricted by manifest, followed by the devkit center-distance and
official filtering path. Do not synthesize evaluator GT from the training cache:
the cache lacks the radar-point and bicycle-rack information required by official
filtering. Reuse devkit `accumulate`, `calc_ap`, `calc_tp` and `DetectionMetrics`.
Bind `detection_cvpr_2019.json` SHA-256
`217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b`.

Prove parity by applying both the unchanged official evaluator and the new subset
path to a full-val manifest under two deterministic fixtures:

- `P-GT`: official-filtered ground truth represented as predictions;
- `P-MIX`: deterministic true/false positives, misses, duplicate scores and
  ordering cases.

Require exact equality (`tolerance=0`) for filtered box identities, every one of
the 40 class-by-distance 101-point arrays, validity masks and all finite aggregate
metrics. Time/path metadata is excluded from equality. Empty predictions use a
local adapter that skips prediction filtering and lets official accumulation
produce exact zeros; record the adapter flag. JSON output maps non-applicable
NaN to `null` plus an explicit validity mask and writes with `allow_nan=false`.
Broad exception swallowing is forbidden.

Adversarial tests cover unknown/missing/duplicate tokens, out-of-manifest boxes,
manifest/cache/devkit identity drift, reordered predictions, zero-point objects,
bicycle-rack cases and empty predictions. A parity mismatch, unsupported class,
ownership leak or solver infeasibility/integrality failure fails STOP-A; thresholds are not
relaxed and roles are not swapped.

## 5. STOP-B observation-first contract

STOP-B observes the **unchanged current graph** on a deterministic `D_low` panel.
The main observation path uses physical B=4, uniform FP32 and the accepted
global-FP16 plus complete SECOND-FP32-island path. Diagnostics are disabled/on
parity checked and make no optimizer update. Reuse S09's accepted B4 capacity
evidence rather than rerunning it as a baseline.

The bounded observation hierarchy is:

1. batch/task target counts, eligible positives and per-task/per-term loss
   numerators, denominators and reconstructed aggregate loss;
2. sparse active-voxel occupancy by sample/stage, group sizes and normalization
   input/output statistics;
3. explicit, predeclared module-boundary activation and true-unscaled gradient
   norms from head through BEV neck/fuser to LiDAR stem;
4. bounded per-task/per-term backward replay on a smaller fixed panel to locate
   the first amplification boundary rather than the first parameter in registry
   order.

Because loss/gradient aggregation itself is a hypothesis, one tiny paired check
may process the same four samples as one B4 batch and as four B1 samples, then
reconstruct the declared sum/mean relationship. This is the only B=1 use in ABC;
it is a diagnostic decomposition, not an optimizer-training cell, rung, epoch or
cost baseline.

The execution-budget basis is at most 64 broad-panel samples, at most 16
term-decomposition samples and about 400 forward/backward replays. This is an
operational cap pending ABC execution approval, not evidence yet. No generic hook
framework or profiler is added. One narrowly bounded refinement is allowed only
when the first observation localizes an adjacent boundary interval but cannot
separate its endpoints; it may add only those explicit boundaries and may not
grow the panel or hypothesis family.

STOP-B exits `LOCALIZED` or `INCONCLUSIVE`. The latter is a valid negative result
and does not trigger more instrumentation. Tiny-group GroupNorm remains one
hypothesis alongside loss normalization, target count, gradient aggregation,
sparse occupancy, norm placement and head-to-stem amplification.

## 6. STOP-C strong-contrast contract

### 6.1 Initialization registry

| ID | Meaning |
|---|---|
| `A0` | declared all-scratch negative control; current graph weights random |
| `A1` | declared public camera prior (current local family: ImageNet1K V1) with random LiDAR/fuser/head, then joint fusion training |
| `A2` | declared camera prior plus a separately trained, graph-compatible LiDAR-only donor, then fusion training; the coherent MIT-derived package binds its NuImages-camera/LiDAR-donor provenance here |
| `A3` | camera-branch-evidence-triggered C-only or dual-branch staged-pretraining candidate; it is considered only if direct camera-prior fusion is shown insufficient and replaces an existing slot rather than adding one |

The coherent MIT-reference-derived package binds its own direct/adapted/local
provenance, including the declared NuImages camera prior and a graph-compatible
LiDAR-only donor. “Reference-derived” is not labelled an exact reproduction where
the local implementation differs. A3 is not a generic external-checkpoint slot;
any unrelated external donor requires a future owner amendment.

### 6.2 Candidate and exposure caps

- Mandatory anchors are the current local family and one coherent MIT-derived
  package. `A0` is a negative control, not a capability finalist.
- At most two single-factor counterfactuals may be triggered by STOP-B. Default is
  zero or one. A second requires an independently localized mechanism. Conditional
  `BN1d`, `TransFusion`, or LiDAR-conditioned `DepthLSS` consumes this cap; these
  options are not an automatic three-cell sweep.
- Current and coherent-MIT anchors reach `D_mid` unless they hit a hard failure:
  nonfinite state, no accepted update, target/modality collapse, identity failure
  or an invalid evaluator result. Early loss ranking alone cannot kill an anchor.
- At most two graph/init families exit STOP-C. Single seed only. No `D_audit`,
  official val, `D_fit` full training or full trainval run occurs in STOP-C.
- Each coherent package has one predeclared recipe. STOP-C does not sweep LR, WD,
  schedule, clip, EMA, sampling, augmentation, batch or accumulation; those belong
  to STOP-D. Package-level comparison is not component attribution.

Promotion uses predeclared internal NDS/mAP, numerical guardrails and paired-log
uncertainty. If candidates are indistinguishable, retain the conservative anchor
or at most two finalists; do not add a tie-breaking rung, seed or cell.

**O-124 APPROVED execution sizing.** The revised envelope uses physical
B=4 for every STOP-C training cell. It assumes at most six `D_low` fusion slots
for one epoch, at most three `D_mid` fusion slots for three epochs, and at most
two LiDAR-donor lineages. A three-fusion-epoch rung budgets at most ten donor
epochs on `D_mid`, preserving the MIT 20:6 donor/fusion exposure ratio. These
slot/horizon/donor counts are the approved ABC completion envelope in
`RUN_REQUEST.md`; O-122/O-123 alone did not authorize them, while O-124 does.
They remain proxy evidence, not full convergence.

For limited-rung C comparisons, the fixed-batch policy is `drop_last=true` at
B=4. Every epoch records the exact 0–3 dropped sample tokens; all matched
candidates use the same sampler order and dropped-token manifest. This avoids a
smaller final optimizer update and does not silently treat the known
`28130 % 4 == 2` tail as a valid B4 batch. STOP-D must separately freeze the
production/full-run tail policy.

A mandatory candidate that cannot execute B=4 after its allowed output-neutral
activation-checkpoint choice and one bounded correctness-debug cycle fails the C
capacity gate; it does not fall back to B=1. B=8/16 are not ABC cells. They are
future STOP-D/E candidates after graph freeze, where batch-dependent optimizer
dynamics, tail policy, throughput and memory can be judged together.

### 6.3 Bounded step-debug policy

The accepted current graph's existing S09 B4 G20 evidence is reused. Each new
graph/init family gets one fresh-from-initialization B4 preflight capped at 20
accepted updates before any epoch/rung work.

If a long cell exposes an obvious correctness defect—exception, shape/target
contract failure, OOM attributable to implementation, repeated post-backoff
nonfinite/discarded windows, missing parameter update, or checkpoint/resume
drift—the cell stops immediately. Reproduce on the same deterministic prefix with
the progression `1 -> 5 -> 20` accepted steps, fix only the diagnosed defect,
rerun focused tests, and require a fresh B4 G20 PASS before restarting the
affected scientific cell from initialization. Across ABC, the proposed execution
envelope permits at most two such debug/fix cycles and at most one cumulative
GH200-hour for them; a repeated blocker returns to the owner.

Finite training that has weak internal NDS/mAP, flat or adverse metric trajectory,
modality collapse without an implementation defect, or no persuasive convergence
signal is a scientific outcome. It is handled by B/C health and promotion gates,
not by adding step probes, LR tweaks, epochs, candidates or seeds.

## 7. STOP-D/E/F boundaries

- **D:** choose the production recipe only after C closes. Sampling/CBGS and
  GT-paste must obey STOP-A manifest ownership. Recipe comparisons match accepted
  updates and sample exposure; checkpoint/resume/scaler/EMA identities remain
  strict. After selection, bind candidate/checkpoint/recipe/decision rule in
  `candidate_freeze.json`, open `D_audit` exactly once, and then freeze the final
  graph/recipe. A material reversal makes the result `INCONCLUSIVE`; do not pick a
  new winner or reopen the audit.
- **E:** profile only the accepted graph and recipe. First establish coverage,
  synchronization and output-neutrality; then change only measured/source-proven
  bottlenecks and requalify numerical and metric behavior. Low memory is not a
  utilization claim. Do not read `D_audit` again.
- **F:** use the graph/recipe frozen by D/E and run the primary fusion full train
  with matched camera/LiDAR controls sufficient for absolute capability and fusion
  contribution. The primary full run is single-seed. Do not read `D_audit` again.
  A bounded additional
  confirmation seed, if later approved, belongs to the frozen internal gate and
  is not silently converted to a second full run. If `A2` survives, its LiDAR
  donor and fusion training are included in cost and provenance.

## 8. Simplified collaboration contract

Use O-094 persistent S00: one long-lived task/worktree, one linear branch, no
fresh implementation worker per stop and no parallel production implementation
chain. `S10` is one evidence namespace. Maintain one phase-sized
`HANDOFF.md/RUN_REQUEST.md/RESULTS.md/REVIEW.md` package and append stop sections;
do not create per-cell handoffs, audit wrappers or review documents.

Every material job still gets an exact immutable source/snapshot, config, data,
cell, command, resource and output tuple in `RUN_REQUEST.md` before submission.
After future owner completion authority, this recording does not require another
per-job question when the tuple is a mechanical derivation inside the approved
aggregate envelope.

Independent review occurs at stop-level immutable evidence boundaries; reviewers
do not fix code. STOP-A's split/metric change requires one isolated high-risk
review worktree. Reuse one bounded S10-R context rather than creating a review
chain. S00 fixes accepted findings linearly. Batch P3-only polish; a repeated
material blocker, uncertain scientific classification, changed data/metric/model/
seed/candidate/resource boundary, or exhausted cap returns to the owner.

## 9. Current open execution decision

The approved ABC GPU budget and derivation limits are in
[`RUN_REQUEST.md`](RUN_REQUEST.md). O-124 authorizes continuous in-envelope work;
every material job still requires an exact recorded immutable tuple before
submission.

## 10. STOP-A implementation state

Immutable implementation candidate
`e27053a5b141e1afaa68363ce6deb2efdb60518e` adds only the accepted STOP-A
surface: a no-seed log-level MILP plus emitted-artifact checker, inherited raw-
dependency ownership audit, manifest-bound internal evaluator, exact official-
path parity gate, a role-bound GTDB seam, and a CBGS token/index identity seam.
The existing official val entry
point is unchanged. GT paste and CBGS remain disabled; no model, loss, optimizer,
schedule or precision behavior changes in STOP-A.

Local validation is limited to `bash -n`, Python bytecode compilation and
`git diff --check` because the login node is x86_64 and has no project Python
dependencies. The exact aarch64/GH200 dependency-backed A-GATE tuples and
outcomes are recorded in `RUN_REQUEST.md`. The implementation commit by itself
was not a STOP-A PASS. Job `468404` now supplies accepted raw A3 gate evidence
from which to start independent high-risk A4 review; STOP-A remains open until
that review accepts the exact evidence SHA.

The first exact A-GATE allocation, Job `463593`, stopped in focused tests before
real data/gate execution because SciPy's aarch64 HiGHS wrapper rejected platform-
`long` sparse indices. `RUN_REQUEST.md` records the immutable negative result and
narrow O-124 remediation classification. It is not a split-feasibility result;
STOP-A remains open and STOP-B/C remain unstarted.

The derived fix1 allocation, Job `463649`, passed 12 focused tests, traversed all
28,130 train samples, and then timed out after `01:00:14` inside the exact split
solve. The current implementation performs 94 cold SciPy MILP calls: ten frozen
scientific objective solves plus 50 base and 34 nested per-log tie-break solves.
No split or parity artifact exists. The external timeout also exposed a runner
defect: `final.exit` was sealed as zero even though `gate.exit` was absent and
Slurm was `TIMEOUT`. Both facts are frozen in `RUN_REQUEST.md` and `RESULTS.md`.

The bounded remediation candidate preserves the exact objective hierarchy and
assignment order while grouping at most ten ternary digits per sequential
radix-3 block, with all integers at most 59,048 and an expected 19 cold solves.
It also makes signal termination fail closed. This may be implemented and locally
proved without changing split science; however, Job `463649` consumed STOP-A's
one-GH200-hour ceiling. A fresh execution tuple therefore requires an owner
resource amendment even though one aggregate debug/fix submission slot remains.

That remediation is immutable at
`d7caf53414ade2d5db794ecd90851d0e5a3535b5`. The old and new synthetic canonical
payload hashes are identical, repeated solve output is identical, and the exact
call count is 19. The detached read-only snapshot, command and hashes are frozen
in `RUN_REQUEST.md`. O-125 supplied exactly one `00:15:00` execution while making
the remaining contingency non-transferable and non-retriable.

Job `467862` consumed that tuple on `n409`. It passed the expanded focused suite
(`13 passed, 8 skipped in 1.93s`) and traversed all 28,130 train samples in about
33 seconds. The 19-call exact MILP then remained active until Slurm `TIMEOUT` at
`00:15:14`. It emitted no `gate.exit`, solver report, split manifest, ownership
record or evaluator-parity artifact; importantly, the fail-closed runner emitted
no false zero `final.exit`. This neither proves infeasibility nor accepts the
split. STOP-A has consumed both debug/fix slots and its final O-125 submission;
no automatic retry is authorized. STOP-B/C remain unstarted pending an owner
decision on the exact-solver boundary.

O-126 resolves that boundary by classifying the global five-stage balance
optimizer and per-log lexicographic certificate as unnecessary protocol
overreach. A1-A4 now preserve the hard science/checker/evaluator surface while
using exactly two real zero-objective solves, pre-solve feature identity, ordinal
one and no reroll. The one authorized A3 allocation is CPU-only on an aarch64
node (`gpu` partition solely for compatible node/runtime access, no GPU GRES),
4 CPUs, 32 GiB and 15 minutes. It is one attempt; any failure or timeout returns
to the owner. Successful immutable evidence must pass independent high-risk A4
review before STOP-A closes or STOP-B/C may begin.

The corrected A1 implementation/protocol is immutable at
`7c01cc3f1e75691339f41f101794945748f03305`. A2 local/static and synthetic
feasibility checks pass; the detached read-only snapshot, exact file/tree hashes,
fresh output and literal no-GPU `sbatch` command are frozen in `RUN_REQUEST.md`
§13. This source commit and tuple freeze are not a STOP-A PASS; A3/A4 remain.

That exact command was submitted once as Job `468295`. Arrhenius
`job_submit/lua` printed that no GPU count was specified and injected four GPUs;
`scontrol` confirmed both `ReqTRES` and `AllocTRES` contained `gres/gpu=4`.
S00 protection-cancelled the job after 8 seconds, before the runner created any
output or executed tests/data/split. Follow-up non-submitting `sbatch --test-only`
checks showed both `--gpus=0` and `--gres=none` receive the same four-GPU default,
so neither is a valid zero-GPU replacement. The scientific candidate ordinal is
unconsumed, but O-126's submission authority is consumed; a one-GPU allocated-
but-CUDA-hidden replacement or another resource route requires owner approval.

O-127 now authorizes the narrow viable route: reserve exactly one GH200 but
force `CUDA_VISIBLE_DEVICES=""` before Python and require PyTorch CUDA availability
false/device count zero. The Slurm allocation must also attest exactly one GPU,
4 CPUs, 32 GiB, `gpu` partition and 15-minute walltime. The process remains
CPU-only; the reserved GPU is a scheduler compatibility cost capped at `0.25`
GPU-hours. Science/data/candidate ordinal/output gates are unchanged. One
replacement submission, no retry/reroll; A4 begins only after immutable A3 PASS.

The O-127 resource-attestation source is immutable at
`ad93c89333b0a8f19abf138c8d6816e742b51e35`; its science parent remains
`7c01cc3f1e75691339f41f101794945748f03305`. The detached read-only snapshot,
hashes, fresh output and literal sole command are frozen in `RUN_REQUEST.md`
§16. Job `468404` consumed that tuple once and completed `0:0` in `00:07:59`
with zero restarts. The process reserved one GH200 only to reach the compatible
aarch64 node, but `CUDA_VISIBLE_DEVICES` was empty, PyTorch exposed zero CUDA
devices, and Slurm recorded `gres/gpumem=0` and `gres/gpuutil=0`.

The focused dependency-backed suite passed (`13 passed, 8 skipped`). The real
candidate ordinal remained exactly one: both zero-objective solves returned
`FEASIBLE_FROZEN`, no reroll occurred, and `candidate_freeze.json` remains
absent/locked until STOP-D. The immutable roles are `D_fit=19,877/34 logs`,
`D_select=4,626/8`, `D_audit=3,627/8`, `D_low=6,155/10`, and
`D_mid=11,661/20`. Reloading the emitted ownership through the independent
checker returns PASS with zero cross-owner overlap for log, scene, sample,
annotation, instance and raw sensor path domains, including all 6,019 official-
val samples.

Evaluator parity also passes: `P-GT` and `P-MIX` both report `EXACT_PARITY`
with tolerance zero for filtered identities, metric-data arrays, validity masks
and aggregate metrics; the empty-prediction adapter returns exact zero mAP/NDS.
This is split/evaluator engineering evidence only, not model quality, recipe,
convergence or official-val selection evidence. A3 consumed `0.133056`
allocated GPU-hours, bringing cumulative STOP-A/ABC allocation to `1.413334`
GPU-hours. The evidence is now ready for an immutable commit and independent A4
review; B/C remain unstarted until review closure. The first A4 review found no
split/evaluator/data P0/P1 finding, but returned `REMEDIATE` for an ambiguous
27-versus-28.1-hour aggregate record (P2) and the over-strong CBGS role-binding
description (P3). Both are documentation-only: the active aggregate is clarified
to the binding, stricter 27-hour ceiling, and CBGS is explicitly gated on future
STOP-D manifest/role caller integration. A targeted re-review is required before
closure.
