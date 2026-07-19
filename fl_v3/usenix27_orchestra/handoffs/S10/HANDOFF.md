# S10 HANDOFF — STOP-A/B CLOSED; STOP-C0-v2 EXECUTION PASS

## 1. State and authority

```text
SESSION: persistent S00 / S10
BASE_SHA: a080d49c1c22de20ccb5b1353d4922c7df14a729
BRANCH: codex/s10-cl-model-recipe
OWNER_DECISION: O-122 scientific envelope; O-124 ABC completion; O-125 legacy optimizer consumed; O-126 corrected STOP-A; O-127 one-GPU/CUDA-hidden replacement; O-128 STOP-B; O-129 parity remediation; O-130 B-RAND; O-131 C0; O-132 full C0-v2 clean replay; O-133 C1-A/C1-B planning; O-134 C1-A execution; O-135 assertion remediation; O-136 sole C1-A replacement; O-137 C1-B0 fusion health; O-138 exact test-only replacement; O-139 canonical-fixture replacement; O-140 C1-B1 capability
PLAN_STATE: six-stop A-F scientific envelope accepted
CURRENT_AUTHORITY: O-140 exact one-job C1-B1 implementation/commit/Slurm/evidence authority active
ABC_IMPLEMENTATION/COMMIT/SLURM/REVIEW_WORKTREE: C1-B1 only; one submission/no retry; no reviewer chain
STOP_A: CLOSED PASS_WITH_RESIDUAL_RISK / reviewed remediation b0478a2 / no open P0-P3
STOP_B: CLOSED INCONCLUSIVE / Job 479667 integrity PASS / review 02ba3b4 PASS_WITH_RESIDUAL_RISK / no open P0-P3
STOP_C: v1 C0 retained as FAIL/INCOMPLETE negative evidence; v2 clean replay execution gate PASS; C1-A `LOCALIZED_NORM`; C1-B0 PASS; O-140 C1-B1 active
C0_IMPLEMENTATION_SHA: 89958be504d6abaef66810695402d2a09619794b
C0_JOB: 492525 / FAILED 1:0 / 00:47:32 / 0.792222 GH200-hours
C0_REMEDIATION_REVIEW: 09c39458a0b32ce1d4a3ae603094d76ae160ac42 / PASS_WITH_RESIDUAL_RISK / no open P0-P3
C0_V2_REPLAY_SOURCE: 2262b4063a3e419b17f4b911a9e11a7ff50ea784
C0_V2_JOB: 496312 / COMPLETED 0:0 / 00:45:15 / 0.754167 GH200-hours
C0_V2_RESULT: aggregate PASS / all three v2 cells present / 80 passed + 3 skipped / 28/28 artifact manifest
C0_V2_REVIEW: no intermediate review authorized or performed
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

C0's executed v1 artifact failed the exact-token part of this policy: it
predicted the remainder from a fresh generator without accounting for the
DataLoader base-seed draw. Its count of three and matched F/L construction remain
valid, but its three named tokens are not exact. The post-job v2 seam observes
actual collated batch tokens; no C0 raw artifact is rewritten.

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

### 6.4 O-131 integrated C0 entry rung

C0 answers a narrower question before the expensive strong contrast: does the
current accepted graph, when given the declared A1 public camera prior, sustain
meaningful normal training on `D_low`; and is the large LiDAR gradient correlated
with harmful optimizer behavior rather than merely a parameterization scale?
STOP boundaries are evidence boundaries, so the same run also emits bounded
pre-STOP-E telemetry without claiming final bottlenecks.

| cell | train horizon | evaluation | role |
|---|---:|---|---|
| `C0-F-A1` | 1,538 attempted physical-B4 windows; 6,152 attempted samples; 3 `drop_last` remainder samples, but raw token identities are invalid | one terminal `D_select` internal mAP/NDS | primary current-family health cell: ImageNet1K V1 camera, random LiDAR/fuser/head |
| `C0-L-A0` | same | same | LiDAR-only companion; random L-S075 |
| `C0-F-A0-P64` | 64 attempted B4 windows | none | all-scratch initialization negative control; not a finalist |

All three use seed 0, global FP16 with the complete SECOND FP32 island, AdamW
`1e-4/.01`, constant LR, uniform sampling, and no clip, EMA, augmentation, CBGS
or GT-paste. This is the unchanged engineering baseline selected for attribution,
not the production recipe and not evidence against a graph that later needs a
reference-aligned recipe in STOP-D.

Long cells report loss over cumulative attempted-window boundaries
`64/384/768/1152/1538`; true-unscaled gradients, target/loss terms, boundary
gradients and realized post-optimizer update/weight are sampled only at
`1/4/16/64/256/768/1538`. The short control samples through 64. Only A1 receives
one bounded early operator trace (`wait=16`, `warmup=2`, `active=10`), deliberately
non-overlapping sampled diagnostic windows; chunk wall
time, samples/s, peak allocated/reserved memory and 1 Hz `nvidia-smi` are recorded
for all cells. Final STOP-E must still confirm performance on the frozen graph and
recipe.

Hard failure means no accepted update, discarded windows, post-first-64 invalid
windows, nonfinite sampled post-warmup gradients, missing gradients across a
required modality path, invalid token-complete evaluator output, or nonfinite
internal metrics. Gradient magnitude alone is not failure. A descriptive
`CORRELATED_HARM_SIGNAL` requires at least two of: post-warmup invalid windows;
sampled LiDAR update/weight at least `1e-2` and at least `10x` the sampled head
median; final loss chunk more than `1.25x` the first. This gate may justify owner
discussion of one single-factor C counterfactual; it does not diagnose a module or
automatically amend the model.

### 6.5 C0 terminal execution state

The sole O-131 Job `492525` completed the two full cells and failed while
closing the short scratch cell. F-A1 and L-A0 each consumed all 1,538 declared
B4 windows and the exact 4,626-sample `D_select`; the 64-window scratch cell has
no accepted summary because the runner incorrectly applied the full-epoch
iterator-exhaustion assertion to it. The job is terminal `FAILED 1:0` at
`00:47:32`; no retry remains.

Both complete cells' raw `HARD_FAIL` labels are also false positives from
requiring gradients on `lidar_encoder.to_bev`, which is `nn.Identity` for the
current SECOND-075 channel contract. Raw artifacts remain unchanged. Excluding
that impossible condition, both complete trajectories have zero post-first-64
invalid windows, falling loss, finite internal evaluation and no correlated
large-gradient harm signal. F-A1 exceeds L-A0 by `+0.044045` internal mAP and
`+0.037182` internal NDS after one epoch, but the comparison is single-seed,
internal-only and includes the A1 camera prior. It cannot select a graph/recipe
or make the S10 primary claim.

Independent review additionally found that the raw v1 `dropped_tokens` vector
used the wrong RNG state and that the raw even-length “median” was an upper
median. These do not change the exposure count, evaluator metrics, F-minus-L
delta or `NOT_ESTABLISHED` harm label, but exact raw remainder-token identity is
unknown. Post-job remediation advances the schema to v2, records actual batch
tokens, uses a standard median, replaces the impossible trainable-prefix gate,
and limits iterator exhaustion to full-epoch cells. These changes affect
diagnostic artifact semantics but not model/loss/gradient/update math; they have
not been executed on GH200.

O-132 subsequently executed the remediated v2 path once. Job `496312` completed
`0:0`, passed all three cell/token/health/aggregate/checksum gates and consumed
`0.754167` GH200-hours. C0-v2 therefore closes its bounded execution/health gate;
it does not select a graph or recipe, identify the gradient cause, or authorize
later C. Full metrics, hashes, performance observations and interpretation limits
are in `RESULTS.md`; the exact consumed tuples are in `RUN_REQUEST.md` §§26–27.

### 6.6 O-133 approved C1-A/C1-B planning contract

O-133 approves the following **scientific plan and documentation only**. It does
not authorize implementation, commits, checkpoint acquisition, Slurm submission,
or STOP-D/E/F. Exact source ownership, tests, immutable cells, resources and
compute remain a future owner gate.

#### C1-A — gradient-mechanism causal gate

C1-A reuses the frozen STOP-B token panel; it does not select a new panel or grow
an observer/profiler framework. The primary route is L-S075 in uniform FP32 so
camera stochastic depth and GradScaler overflow do not confound the mechanism
comparison. Exact W0, B4 tokens, targets and loss semantics are shared.

| candidate | sole graph difference | status |
|---|---|---|
| `C1-GRAD-GN` | current sparse tiny-group GroupNorm | approved plan |
| `C1-GRAD-BN1D` | direct-reference sparse `BN1d(eps=1e-3, momentum=0.01)` with shape-compatible affine initialization | approved diagnostic plan; not automatically a production candidate |

Each candidate performs both the normal detection-loss backward and a frozen
FP32 upstream-gradient VJP injected at the SECOND output. The latter removes
head/loss/target amplification from the encoder-Jacobian measurement. Record only
the bounded facts needed to distinguish target/loss terms, sparse occupancy,
normalization input variance/inverse-std, norm placement and SECOND output/stage/
stem boundary and parameter gradients. Runtime variation is summarized as a
distribution; byte identity is not an acceptance gate.

The exit is one of `LOCALIZED_NORM`, `LOCALIZED_HEAD_LOSS`,
`LOCALIZED_SPARSE_OCCUPANCY`, or `INCONCLUSIVE`. A large gradient alone does not
promote BN1d. BN1d consumes one counterfactual slot only if a future exact gate
shows a stable causal reduction and matched-exposure training/metric behavior is
not degraded. TransFusion and LiDAR-conditioned DepthLSS do not automatically
open from C1-A.

#### C1 common GradScaler/exposure policy

Before any C1 scientific FP16 training cell, use the same frozen no-update B4
qualification to select a conservative power-of-two init scale that is finite for
every admitted graph. Bind it in resolved config/checkpoint provenance. This
prevents candidate-dependent loss of initial scientific batches; it does not hide
true unscaled gradients, which remain recorded. If no common qualified scale
exists, stop and return to the owner rather than silently using different exposure
or FP32 for one candidate.

#### C1-B — strong graph/initialization contrast

The three approved primary lineages are:

| lineage | meaning |
|---|---|
| `C1-CUR-A1` | current graph; ImageNet1K V1 camera; random LiDAR/fuser/head; joint fusion training |
| `C1-CUR-A2` | same current graph and camera prior; separately trained graph-compatible L-S075 donor; then joint fusion training |
| `C1-MIT-A2` | coherent MIT-reference-derived package; declared camera prior plus graph-compatible LiDAR donor; exact package composition pending owner decision |

Every new graph/init family first passes one fresh B4 G20 correctness gate. The
scientific funnel then uses matched `D_low` one-epoch cells and `D_select`, followed
by at most three `D_mid` fusion lineages for three epochs unless a mandatory anchor
hits a predeclared hard failure. C0 is retained as health evidence, but its four
initial skipped updates are not silently used as the matched C1 ranking baseline.
Promotion uses internal NDS as primary, mAP/per-class/numerical health as
guardrails and paired-log uncertainty. At most two graph/init families exit
STOP-C. A triggered counterfactual replaces a lineage/slot; it never expands the
cap. No recipe Cartesian product, extra seed, `D_audit`, official val or full run
belongs to C1.

“Coherent MIT anchor” currently names a required strong package-level contrast,
not an existing checkpoint or an exact reproduction claim. Its precise graph,
initialization files, direct/adapted/local component labels and required source
changes remain **owner pending**. No contributor may infer a wholesale MIT rewrite,
an adapted shared-head package, or automatic BN1d/TransFusion/DepthLSS inclusion
until that decision is recorded.

### 6.7 O-134 C1-A activation and C1-B order amendment

The owner relaxes the earlier conditional restriction on BN1d, TransFusion and
LiDAR-conditioned DepthLSS. This does not create an ablation sweep or admit all
three automatically. C1-B must first execute the current graph's matched A1/A2;
only if a future exact gate finds the current setting/structure materially worse
may a bounded MIT-reference-guided repair be frozen. The meaning and threshold of
“materially worse” must be recorded before C1-B compute. O-134 does not authorize
C1-B implementation, checkpoint acquisition or compute.

O-134 activates C1-A only. It reuses the complete accepted STOP-B L-S075 panel:
16 disjoint B4 batches/64 samples, uniform FP32, seed-bound W0, no optimizer,
update or evaluator. Current GN and direct-reference `BN1d(eps=1e-3,
momentum=0.01)` share exact convolution and affine parameters. Each candidate
runs both normal detection-loss backward and a sparse-coordinate/channel-derived
fixed SECOND-output VJP twice per B4, for exactly 128 runs. Paired candidate
effects must exceed both a two-fold reduction gate and observed two-repeat runtime
variation; conservative multi-metric precedence yields `LOCALIZED_NORM`,
`LOCALIZED_HEAD_LOSS`, `LOCALIZED_SPARSE_OCCUPANCY`, or `INCONCLUSIVE`.

The sole allocation is one GH200, 8 CPUs, 64 GiB and `00:30:00`, capped at `0.5`
elapsed GH200-hour. One immutable source/snapshot/output and one submission are
authorized; no retry or automatic C1-B continuation exists. Exact hashes and
command are frozen after the implementation commit in `RUN_REQUEST.md`.

Job `502456` consumed that sole tuple and failed `1:0` after `00:03:03`
(`0.050833` GH200-hour). All 36 focused tests, source/tree/runtime/data/panel
identities and artifact checks passed. Failure occurred while constructing the
BN1d candidate, before model-to-GPU, loader iteration or any candidate
forward/backward: `load_state_dict(strict=False)` reported the expected 42
`running_mean/running_var` keys but did not report 21 `num_batches_tracked`
keys, because PyTorch's BatchNorm backward-compatibility loader synthesizes
those buffers. The runner incorrectly required all 63 keys in `missing_keys`.

No `runs.jsonl`, candidate identity, gradient metric or C1-A verdict exists.
This is a runner assertion defect, not evidence for or against GN, BN1d, large-
gradient causality or model health. O-134 is consumed and explicitly grants no
retry; any correction/replacement requires a new owner decision.

O-135 subsequently authorized the correction itself. The resulting change
moves the compatibility rule into a tested helper: reported missing keys must be
exactly the 42 `running_mean/running_var` entries; every one of the 21 separately
synthesized `num_batches_tracked` buffers must exist and equal zero; running means
must be zero, running variances one, unexpected keys empty and trainable-parameter
hashes unchanged. A standalone GN→BN1d fixture directly reproduces the PyTorch
missing-key behaviour and rejects a nonzero batch counter. Model graph/math,
data, panel, candidates, metrics and thresholds are unchanged. This remediation
is committed at `d713bfe3b5e5c587f58ce70721b2b6eea0b050ec`.

O-136 accepts that exact remediation and authorizes one strictly derived C1-A
replacement. The O-134 panel, candidates, shared W0, uniform-FP32 pathways,
two-repeat/128-run matrix, metrics, thresholds, absence of optimizer/update/
evaluator, and one-GH200/8-CPU/64-GiB/`00:30:00`/`0.5`-GH200-hour ceiling are
unchanged. Before the sole submission S00 must freeze a new detached read-only
snapshot, hashes, fresh output and literal command in `RUN_REQUEST.md`. Any
failure or drift returns to the owner. O-136 grants no retry, C1-B continuation,
later-stop execution, merge, push or upload.

Exact replacement Job `502572` consumed O-136 and completed `0:0` in `00:03:09`
(`0.052500` GH200-hour). It passed 37 focused tests, source/tree/runtime/split/
panel/config identities, both candidate state gates, all 128 finite runs, exact
pre/post parameter-state immutability and all 15 runner artifact checks. Physical
B4 counts are 32 runs in each candidate/pathway cell; no optimizer, update or
evaluator was constructed.

The predeclared result is `PASS / LOCALIZED_NORM`. Across every one of 16 B4
batches, BN1d/GN median ratios were `0.001862` for fixed-VJP boundary
amplification and `0.003669` for normal-loss boundary amplification; fixed-VJP
stem max-abs/RMS ratios were `0.000164`/`0.000186`, and normal-loss ratios were
`0.001578`/`0.001657`. All paired support fractions are `1.0`, and all median
effects exceed the p95 two-repeat variation gate. Occupancy correlations range
from `-0.4824` to `-0.0382`, below the predeclared occupancy gate. The current-GN
loss stem max-abs median is `307037.94`, but loss-upstream/stem Spearman is
`0.6471`, below the `0.7` head/loss gate. Thus the fixed-upstream control locates
a causal normalization-path contribution at this exact random W0/panel; it does
not prove GN is the only mechanism.

This does **not** promote BN1d. The no-update normal-loss median is higher for
fresh-state BN1d (`1025.19`) than GN (`538.98`), and C1-A contains no training,
convergence or evaluator evidence. C1-B or a BN1d training candidate therefore
requires a separate owner-approved protocol; no automatic continuation exists.

### 6.8 O-137 C1-B0 matched fusion-health gate

O-137 activates only the observation rung needed before a longer C1-B decision.
It adds one fail-closed production seam: `s10.v1` explicitly records SECOND
`group_norm` or `batch_norm_1d`, propagates it through detector construction and
therefore binds it into the resolved-config/checkpoint identity. Historical
`s09.v1/v2` configs continue to resolve SECOND to GN; the production default is
unchanged. BN1d adds its 21 sites' 63 running-state buffers, so strict state load
and config hash distinguish the candidate checkpoints.

The exact H256 sample vector is selected without labels by sorting accepted
`D_low` tokens on `SHA256("s10-c1b0-h256-v1\\0" || token)` and taking the first
1024. Its ordered hash is
`62a096c0990e6d1d0932868a882b2418e731d1a816f481e741996e49c8e975f7`.
Both serial cells consume that exact vector as 256 ordered physical-B4 batches:
`F-A1-GN-H256` and `F-A1-BN1D-H256`. They share exact seed-0 trainable W0,
ImageNet1K V1 camera initialization, random LiDAR/fuser/head initialization,
global FP16 plus SECOND FP32 island, AdamW `lr=1e-4/wd=.01`, constant scheduler,
and no augmentation/EMA/CBGS/GT-paste/clip. Scale 32 is first qualified on the
same B4 without an optimizer update; that model is discarded and W0 reconstructed.

Each candidate must then complete 256 real optimizer updates. Record all-window
loss/scaler/outcome/tokens/basic wall time, peak memory, BN running summaries, and
true-unscaled gradients plus realized update/weight at windows
`1,4,16,64,128,256`. Missing required camera/LiDAR/fusion/head gradients,
nonfinite values, skipped updates, config/W0/token/state/artifact drift or an
incomplete horizon is a hard stop. There is deliberately no numerical winner
threshold: falling, flat, worse or noisy training is retained as descriptive
single-seed H256 evidence and does not trigger tuning or retry.

The sole resource envelope is one GH200, 16 CPUs, 96 GiB and `00:30:00`, expected
`0.20-0.30` and capped at `0.5` GH200-hour. No evaluator, D_select, D_audit,
official val, checkpoint selection, full `D_low`, C1-B1, A2 donor, MIT repair,
TransFusion, DepthLSS, recipe search, reviewer chain or later stop is included.

Job `502958` consumed the sole O-137 submission and failed `1:0` after
`00:02:14` (`0.037222` GH200-hour), with zero restarts. Focused pytest completed
100 passed and 6 failed in `69.51s`; the runner exited before telemetry, runtime
identity, model construction, H256 loader iteration, optimizer construction or
either scientific cell.

The diagnosis is exact and test-only. In
`test_s06_resolved_config.py`, the three-line operator-profile hash mutation that
belongs at the end of `test_s09_v2_checkpoint_and_operator_profile_are_explicit_and_hash_bound`
was mechanically left after the new parameterized rejection test's expected
exception; all five parameter cases therefore correctly raised once and then
failed on an unintended second resolve. In `test_s08_precision_partition.py`,
the new production propagation fixture changed its schema to `s10.v1` but omitted
the newly required `training.grad_scaler_init_scale=32`. No traceback implicates
production config resolution, model propagation or checkpoint state semantics.
Indeed the same job passed the GN/BN1d exact-parameter, 63-running-buffer, strict
cross-load and sparse-runtime tests.

This is not GN/BN1d health or training evidence and cannot be used for C1-B0
selection. O-137 has no retry/remediation loop, so the source remains unchanged
pending an owner decision.

### 6.9 O-138 exact test-neutral replacement

O-138 authorizes only the diagnosed correction: move the three misplaced
operator-profile hash assertions back into their preceding S09-v2 test and add
the required scale-32 field to the migrated `s10.v1` test fixture. Production
source, runner, experiment config, frozen H256 data, cells, W0/seed, precision,
optimizer, horizon, diagnostics, hard/scientific gates and O-137 resource ceiling
remain byte-identical. After local/static validation, freeze one new detached
read-only snapshot and fresh output and submit exactly once. There is no retry,
scope derivation, C1-B1, later-stop or reviewer-chain authority.

The correction is committed at exact source
`0f51e11c9f879f5bcb9ab2632bcee31969e5c0ac`; its production runner, entry and
config hashes are unchanged from O-137. The new detached recursively read-only
snapshot and fresh replacement command are frozen in `RUN_REQUEST.md` §33.

Job `503075` consumed that tuple and failed `1:0` after `00:02:11`
(`0.036389` GH200-hour), with 105 focused tests passed and one failed. The same
migrated test fixture also lacked required `execution.operator_profile`; the
first run's missing scale field caused config validation to stop before revealing
that second omission. The runner again stopped before telemetry, model/H256
construction or either training cell. O-138 is consumed and grants no correction,
retry or C1-B1 continuation.

### 6.10 O-139 canonical-fixture replacement

O-139 removes the failure-prone manual `s09.v1` to `s10.v1` promotion from the
production-propagation test. The test instead consumes the complete canonical
`s10_second_config(..., "batch_norm_1d")` fixture already exercised by the S10
config suite and asserts resolved schema, normalization, scale and operator-
profile propagation before calling the detector constructor. No production or
experiment file changes. After local/static schema audit, freeze one new detached
read-only snapshot and fresh output and submit exactly once under the unchanged
O-137 resource/scientific tuple. No retry, C1-B1 or later-stop continuation.

The canonical fixture was constructed and resolved outside pytest, both real
production candidate identities remained unchanged, and no manual schema
promotion remains. Exact source `5de019bf36b1dd5ca077a5a10eaa5e0e5f376ca2`,
its detached recursively read-only snapshot and replacement command are frozen
in `RUN_REQUEST.md` §34.

Job `504508` consumed O-139 and completed `0:0` in `00:06:57`
(`0.115833` GH200-hour). All 106 tests passed. GN and BN1d each completed the
exact 1024-token/256-B4-update horizon with identical initial trainable parameter
hash, zero overflow/invalid/nonfinite/discard, constant scale 32, complete sampled
gradients/updates and artifact gates. First/last-16 mean loss was
`161.242 -> 21.751` for GN and `266.228 -> 22.893` for BN1d. BN1d loss exceeded
GN on 251/256 matched windows, so the larger fractional decline does not imply a
better candidate.

At windows 1/4/16/64/128/256, BN1d-to-GN LiDAR-stem gradient-L2 ratios were
`0.000525/0.000843/0.000645/0.004619/0.000599/0.002365` (about 1904-217x
reduction), while realized stem-update/weight ratios between candidates were
`1.000/0.988/1.072/1.539/1.037/0.814`. Thus normalization causality persists in
the fusion optimizer graph, but AdamW prevents the raw gradient ratio from
becoming a proportional update ratio; GN's large gradients caused no H256
training failure. BN1d reached 21 finite running-stat sites with batch counters
256. Descriptive throughput was `13.03` versus `9.25` samples/s (1.41x), with
peak allocated/reserved `16.06/40.64` versus `16.44/40.81` GiB. No evaluator,
checkpoint selection or architecture promotion exists; longer capability remains
unknown and C1-B1 requires owner approval.

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

## 9. Current execution boundary

The approved ABC GPU budget and derivation limits are in
[`RUN_REQUEST.md`](RUN_REQUEST.md). O-132's sole replay is consumed. No C0 retry,
later STOP-C strong contrast or STOP-D/E/F compute is currently executable; a
new owner decision must define the next scientific cell and exact immutable
tuple. The unused O-124 ceiling is not a spare-job entitlement.

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
review; B/C remained unstarted until review closure. The first A4 review found no
split/evaluator/data P0/P1 finding, but returned `REMEDIATE` for an ambiguous
27-versus-28.1-hour aggregate record (P2) and the over-strong CBGS role-binding
description (P3). Both are documentation-only: the active aggregate is clarified
to the binding, stricter 27-hour ceiling, and CBGS is explicitly gated on future
STOP-D manifest/role caller integration. Targeted re-review of exact remediation
SHA `b0478a298a0a3b5e538bedcca63e2541d71c2146` returned
`PASS_WITH_RESIDUAL_RISK` with no open P0-P3. STOP-A is therefore CLOSED PASS.
Its output is the canonical reusable limited-rung split/evaluator identity for
downstream S10 work. This closure satisfies STOP-B's data/evaluator dependency;
it does not itself start STOP-B, authorize a new Slurm tuple, or broaden any
candidate, seed, resource or interpretation boundary.

## 14. STOP-B O-128 terminal state

Implementation `8fd832dc7d46e8818216ecbcf228ef8fd0590ecb` added only
explicit detector/SECOND/loss observation seams, frozen split/panel binding, a
single one-shot runner and focused tests. Exact Job `477892` passed all 39 tests,
the physical STOP-A manifest binding, runtime/config identities and the pre-model
`P_core48/P_term16` freeze. It then failed at the first FP32 disabled/on parity
batch before broad or term observations.

The runner combined exact output, raw-gradient, loss, RNG and model-state parity
into one fail-closed predicate but raised before persisting the individual
booleans. Therefore the evidence supports only **early parity FAIL with unknown
failing subpredicate(s)**. It does not support a LiDAR-gradient localization,
GroupNorm mechanism, architecture failure, convergence statement, recipe change
or B-REFINE. The sole O-128 B-DIAG submission is consumed and no automatic
replacement exists. Exact job, panel, W0 and artifact identities are in
`RUN_REQUEST.md` §19 and `RESULTS.md`.

`RUN_REQUEST.md` §20 records the **O-129-approved** correction: reuse the
physical frozen panel, warm the sparse runtime without updates, compare two
disabled runs before enabled diagnostics, retain exact hashes, apply the same
fixed `rtol=1e-5/atol=1e-7` numerical envelope already used elsewhere in
STOP-B, and persist every predicate before failing. This separates baseline
instability from instrumentation non-neutrality without using observed Job
`477892` differences to choose a tolerance. O-129 authorizes implementation,
one immutable replacement B-DIAG capped at `0.5` GH200-hour, and zero or one
trigger-bound B-REFINE capped at `0.25` GH200-hour.

## 15. STOP-B O-129 replacement terminal state

Implementation `43f157b3eca7ca72633358b5a2d2dbc4c4e4684b` and exact
§21 Job `478250` completed the approved parity remediation. The job passed 41
focused tests, reused the exact physical Job-477892 panel without reconstruction,
matched the exact W0, and preserved W0 across the excluded FP32 warm-up.

The first repeated disabled `P_core` B4 path then failed output/loss/gradient
repeatability while RNG and model-state hashes remained exact. Disabled losses
were `391.5013732910156` and `388.7950134277344`; all gradients were finite and
present, but global relative-L2 difference was `3.5323887774502536` with 434/459
parameters failing the fixed allclose envelope. Per the approved attribution
rule this is **baseline_instability**, not an instrumentation-neutrality verdict.

Job `478250` stopped before FP16 and every broad/term/aggregation/localization
cell. It provides no `LOCALIZED`/`INCONCLUSIVE` gradient-localization verdict,
does not explain the large LiDAR gradient, and does not prove a specific sparse
kernel, GroupNorm, loss or architecture mechanism. B-REFINE is not triggered.
The sealed runner manifest is `801e98c...`; exact hashes and compute accounting
are in `RUN_REQUEST.md` §22 and `RESULTS.md`. Independent review returned
`PASS_WITH_RESIDUAL_RISK` with no open P0-P3. At the O-129 boundary no further
compute was executable and owner rebaseline was required; O-130 subsequently
superseded only that execution boundary with the single B-RAND amendment below.

## 16. STOP-B O-130 B-RAND amendment

O-130 accepts that current production training has intended randomness: the
trainable torchvision Swin-T path uses stochastic depth up to probability `0.2`.
The fixed STOP-B loader still forbids shuffle, augmentation, GT paste and CBGS.
The revised observation therefore separates controlled RNG variation from
same-seed runtime variation rather than requiring output byte identity.

One exact physical first-`P_core` B4 token vector is reused for all modes, with
mode-appropriate sensor payloads. C-STR8, L-S075 and F-U remain current component
graphs for diagnosis, not newly accepted architecture candidates. Each is
initialized at its own seed-0 W0 and executes:

```text
warm-up: seed 9000 x1
fixed-seed group: seed 10000 x5
varying-seed group: seeds 11000,11001,11002,11003,11004
```

There are exactly 33 forward/backward runs, FP32 only, physical B4, with no
optimizer, update, scheduler, EMA, checkpoint or evaluator. The job records
loss, output and gradient distribution metrics. Exact output/gradient hashes
remain provenance evidence only. Completion means the observation artifact is
integrity-complete, not that a model or localization gate passed.

Candidate-source labels use a predeclared coarse fourfold dominance rule across
loss-relative, output-relative-L2 and gradient-relative-L2 medians. A unique
two-of-three signal may label camera stochasticity, LiDAR runtime variation or
fusion-only interaction; multiple/no signals return mixed/inconclusive. These
labels are operational triage, not causal proof. No automatic broad/term,
counterfactual, model/recipe change or STOP-C follows.

## 17. STOP-B O-130 Job 479667 evidence state

Implementation `0bf9c0ce4148bc82d977e0d66615f606144971b6` and exact
§24 Job `479667` completed the approved B-RAND decomposition. The job passed 43
focused tests, all source/config/data/panel/runtime/resource identities and the
complete structural/finite integrity gate. It executed exactly 33 physical-B4
forward/backward runs and 24 reference comparisons with no optimizer, update or
evaluator. Slurm reports `COMPLETED 0:0`, `00:07:08`, zero restarts and
`0.118889` GH200-hours.

C-STR8 is exactly repeatable for five seed-10000 repeats: loss, output and
gradients all have zero relative difference and one exact hash. On the graph
containing twelve train-mode Swin-T stochastic-depth modules, varying seeds
produce camera RNG-dependent median loss/output/gradient relative differences
`0.033634 / 0.226290 / 0.165922`. The run did not capture stochastic-depth
masks or execute an SD-disabled counterfactual.

L-S075 has no stochastic-depth modules but is not same-seed repeatable. Its
fixed-seed median output/gradient relative-L2 is `0.034248 / 0.657438`; changing
the RNG seed produces comparable `0.033945 / 0.609070`. F-U fixed-seed
output/gradient relative-L2 is larger at `0.041844 / 1.223123`. Prefix evidence
is strongest in the early sparse SECOND stem/stage1/down path. This narrows the
repeatability source operationally to the LiDAR sparse route while explicitly
not proving a specific kernel, module or normalization cause.

Both camera stochasticity and LiDAR runtime variation qualify the predeclared
two-of-three dominance rule, so the descriptive label is
`MIXED_INCONCLUSIVE`; fusion-only interaction qualifies only on loss and is not
accepted. The result removes byte equality as a sensible scientific gate and
explains Job-478250 repeatability at the route level. It still does not explain
the large true unscaled LiDAR gradient, establish convergence or model health,
or authorize a model/recipe change or STOP-C.

Both checksum manifests verify; the immutable output is
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_rand_0bf9c0c_o130_a1`
with summary SHA
`dd51f5801084714fccbd0c351b0696c3a6a2843b462662c74f757fc12cd147c5`
and runner-manifest SHA
`d964b7cc5fa09692a9b8bd95b83cf8cfed85768ff771eaf8cc2a9c8c3cb11ac0`.
Independent review of evidence SHA `fdf223b` found no P0-P2 and two
documentation-only P3. Targeted re-review of remediation
`02ba3b44202092894f2c1c3e7ee53bb56ba92a1d` closed both, found no new issues and
returned `PASS_WITH_RESIDUAL_RISK` with no open P0-P3.

STOP-B is CLOSED / `INCONCLUSIVE`. Accepted evidence is the bounded route-level
repeatability decomposition: camera RNG-dependent variation on the current
stochastic graph and LiDAR sparse-route same-seed runtime variation. The large
true unscaled LiDAR-gradient mechanism remains unresolved. No further STOP-B
compute, model change or recipe change was authorized at that closure boundary;
O-131 subsequently activates only C0.

## 18. O-140 C1-B1 active contract

O-140 activates one matched current-A1 GN/BN1d capability comparison and no
other C lineage. Each candidate shares exact trainable W0 and actual shuffled
`D_low` B4 exposure/remainder, completes 1,538 accepted updates with scale 32,
saves only the terminal checkpoint, and runs the accepted internal evaluator on
the exact 4,626-sample/eight-log `D_select`. The only factor changed is SECOND
normalization.

Execution health is fail-closed. Capability reporting uses internal NDS as the
primary metric, with mAP/per-class AP and paired delete-one-log jackknife
uncertainty. No numerical superiority or non-inferiority margin was approved,
so the code must emit `OWNER_DECISION_REQUIRED` and cannot promote BN1d or GN.
One one-hour GH200 submission is authorized with no retry; after evidence
sealing S00 stops before A2, MIT repair, D_audit/official val or later stops.
