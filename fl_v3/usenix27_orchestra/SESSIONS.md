# USENIX Security '27 Orchestra — session contracts

> **Status:** the 15-session delivery/review model is owner-approved. Each technical
> session is `PLANNED` but inactive until S00 issues its exact kickoff; compute still
> requires a separate owner-approved `RUN_REQUEST.md`.
> **Canonical objective/gates:** [`ORCHESTRA.md`](ORCHESTRA.md).
> **Copy-ready worker/reviewer prompts:** [`KICKOFFS.md`](KICKOFFS.md).
> **Rule:** worker sessions do not edit either canonical file; each writes a
> durable package under `handoffs/Sxx/` and returns it to the Orchestra session.
> `fl_v3/collab/` is read-only legacy evidence and is never a delivery location.

The plan contains **15 worker sessions (`S01`-`S15`)**. `S00` is the separate
Orchestra/owner role and is not counted as a worker session. Independent `Sxx-R`
review sessions are quality gates, not additional implementation tasks in the
15-session count.

Pending owner decisions in `ORCHESTRA.md` Section 10 are stage gates, not a blanket
block on unrelated work. S00 may launch reference audits, interface proposals,
tests, and cost/engineering evidence before a later decision is frozen. Each exact
kickoff must state whether the relevant architecture/metric/split choice is already
approved for implementation or whether the session is limited to evidence and a
decision proposal.

## 1. Work graph

```text
                           ┌─ S03 camera modules ─┐
S01 ZIP data ──────────────┤                     │
S02 P0 correctness ────────┼─ S07 integration ──┼─ S08 camera runs ─┐
S04 LiDAR SECOND ──────────┤                     └─ S09 LiDAR runs ──┼─ S10 fusion/recipe
S05 head/decode ───────────┤                                           │
S06 production runtime ────┘                                           ├─ CL-PILOT ─ S13 clean FL/attack
                                                                      │
S12 FL protocol/tail split/security framing ───────────────────────────┘

S10 ── S11 final CL seeds/freeze ── CL-FREEZE ───────────────┐
S13 viable attack ── S14 defense/adaptive/generalization ────┼─ S15 paper/artifact
S12 protocol + paper skeleton ────────────────────────────────┘
```

### Parallel waves

| Wave | Sessions | Parallelism and boundary |
|---|---|---|
| A | S01-S06, S12 | run concurrently in isolated worktrees; module sessions do not edit integration files |
| B | S07 | sole integration owner; merges/rebases only owner-approved worker results |
| C | S08, S09 | camera and LiDAR scientific jobs run concurrently on the integrated commit |
| D | S10 | fusion/recipe selection; produces `CL-PILOT` |
| E | S11 and S13 | final CL seeds and preliminary FL/attack run concurrently after `CL-PILOT` |
| F | S14, S15 | defense/generalization and paper/artifact overlap; final tables require `CL-FREEZE` |

### Review convention

For implementation sessions S01-S07 and security-code sessions S13-S14, use a
separate review session `Sxx-R` before integration. The review reads the exact diff,
`HANDOFF.md`, any `RUN_REQUEST.md`/`RESULTS.md`, and the relevant reference;
findings are ordered by scientific and correctness severity. Execution-only
sessions S08-S11 still require an independent results/science review of manifests,
logs, metrics, failed cells, and checkpoint/config hashes.

### Evidence-driven refinement convention

Completion of a worker does not freeze the original downstream plan. S00 first
checks `HANDOFF.md` and the actual diff/artifacts, prepares the independent `Sxx-R`
envelope for the owner to open in the task UI, and accepts or returns that review.
It may then refine not-yet-started
sessions' ordering, dependencies, required reading, file ownership, evidence
requirements, review focus, and kickoff wording. Every refinement cites the
triggering handoff/review in the change-control ledger.

Changing Protocol A/B roles, data/split ownership, architecture/head/metric,
threat model, matrix/cells/seeds, scientific gates, compute/resources, upload, or
publication scope is not an operational refinement: S00 must present it to the
owner and record explicit approval. Active work is never silently respecified; a
recorded amendment must be acknowledged in the worker handoff and independently
reviewed where semantics changed.

### Worktree convention

The recommended default is that the owner selects `Worktree` and the starting
branch in the Codex task-creation UI from the SHA named by S00. The kickoff prompt
does not ask the new session to create its own worktree. It carries `BASE_SHA`,
`SOURCE_BRANCH`, `EXPECTED_REF_MODE`, file ownership, and upstream handoff SHAs.
Codex-managed tasks normally start detached at `BASE_SHA`; this is valid. The
session verifies root/HEAD/ref/status and stops on a real mismatch; it does not add,
switch, move, remove, or prune worktrees or branches.

| Session type | Provisioning/base rule |
|---|---|
| S00 | permanent worktree or pinned managed task from the approved integration SHA; sole canonical-document writer |
| S01-S06, S12 | separate UI-created managed worktrees from the same pinned `BASE_SHA` |
| `Sxx-R` | separate UI-created review worktree at exact durable `WORKER_SHA`; never reuse the worker session |
| S07 | dedicated integration worktree from the approved integration SHA; only reviewed worker commits enter |
| S08-S15 | UI-created worktree from the exact frozen candidate/security commit named in the kickoff/request |

Canonical Orchestra files must be committed on the source branch before these
worktrees are created. Codex can apply selected local changes to a new managed
worktree, but that is not an immutable reproducible base. Keep active task worktrees
pinned until accepted artifacts are landed; do not rely on automatic retention.

## 2. Session summary

| ID | Session | Depends on | Primary output | Status |
|---|---|---|---|---|
| S00 | Orchestra/owner decisions | — | approved contracts and status ledger | approved; fresh session pending |
| S01 | Shared nuScenes ZIP backend | S00 kickoff | data backend, manifest, parity/coverage tests | planned |
| S02 | CL P0 correctness | S00 kickoff | pillar/Gaussian fixes and invariance tests | planned |
| S03 | Camera branch architecture | S00 kickoff | corrected stride-8 independent camera modules | planned |
| S04 | LiDAR SECOND architecture | S00 kickoff | sparse XY-downsampling encoder contract | planned |
| S05 | Detection head and decode | S00 kickoff | multi-task CenterHead and deterministic NMS | planned |
| S06 | Production modes/runtime | S00 kickoff | C/L/F modes, config, resume, loader, eval | planned |
| S07 | Integrated engineering gate | S01-S06 PASS | one resolved candidate stack and 100/1000-step evidence | planned |
| S08 | Camera scientific run | S07 PASS | `C-STR8` full-val result/checkpoint | planned |
| S09 | LiDAR scientific runs | S07 PASS | `L-P020` and `L-S075` results/checkpoints | planned |
| S10 | Fusion and recipe selection | S08, S09 | `F-U`/`F-CBGS`, optional init A/B, `CL-PILOT` | planned |
| S11 | Final CL capability freeze | S10 `CL-PILOT` | C/L/F x3 seeds, frozen architecture/config schema | planned |
| S12 | FL protocols, tail split, and security thesis | S00 kickoff | Protocol A/B contract, split/client design, claims/evidence map | planned |
| S13 | Clean FL baselines and modality-localized attack | S07 API + S10 `CL-PILOT` + S12 | clean Protocol-B adaptation, Protocol-A control, viable attack/mechanism | planned |
| S14 | Defense, adaptive attack, generalization | S13 viable | structure-aware defense and adaptive evaluation | planned |
| S15 | USENIX '27 paper and artifact | S12; rolling inputs | registered/submitted paper and reproducible artifact | planned |

## 3. Worker contracts

### S00 — Orchestra and owner decisions

**Purpose.** Keep the single source of truth, approve deviations, assign sessions,
and prevent incompatible branches or scientific protocols from being mixed.

**Stage gates scheduled by S00:**

- Protocol B/A roles and the collaboration/permission rules are already locked;
- each task receives its own approved base/ref/worktree envelope before kickoff;
- the Gaussian reference is selected before S02 changes target semantics;
- the minimum camera/LiDAR/head contract is selected before S03/S04/S05 makes a
  primary implementation choice, and the integrated contract is frozen before S07;
- capability thresholds, cells, seeds, job allocation, and run priority are decided
  from reviewed engineering evidence before the exact scientific run they govern;
- split/client/update scope is frozen and hashed before Protocol-B data
  materialization or training, not before unrelated CL engineering.

**Outputs.** Status updates to the three canonical files, merge/review decisions, job
ledger, owner-approval ledger, and go/no-go decisions. S00 does not implement worker
changes opportunistically or infer permission to submit compute from plan approval.

For each completed session, S00 verifies handoff completeness, prepares its review
envelope for owner launch, records the verdict, and then refines only unstarted
downstream plans/kickoffs. It logs operational refinements and requests owner
approval for every material or locked scientific change described above.

---

### S01 — Shared nuScenes ZIP backend

**Kickoff.** “Implement a read-only, worker-safe ZIP-backed nuScenes data path for
the Arrhenius module without extracting millions of files; preserve directory-mode
behavior and prove byte/array parity.”

**Owns.** `src/fl_v3/data/nuscenes/**`, cache/data-specific scripts and tests, and
the final environment/data-status documentation update. It does not edit model,
trainer, or experiment configs.

**Deliverables.**

- canonical module discovery and `NUSCENES_DATA_DIR` handling;
- a member-to-archive manifest for `trainval01` through `trainval10` blobs;
- lazy per-worker archive handles with safe reopen after worker/process creation;
- image and LiDAR byte readers used by keyframe and multi-sweep paths;
- info-cache creation without extraction;
- path-coverage report for six cameras, `LIDAR_TOP`, and all requested sweeps;
- directory backend retained for mini/local tests;
- update stale shared-data statements in `AGENTS.md`, active `docs/env.md`, and the
  Orchestra handoff/results only after end-to-end verification; do not edit the
  read-only `collab/arrhenius_migration.md`.

**Gate.**

- directory/ZIP mini samples have identical decoded image tensors and LiDAR arrays;
- 100% full train/val referenced members resolve;
- deterministic multi-worker repeated reads;
- no archive extraction and no writes to the shared dataset;
- full-data throughput/data-wait profile is emitted, not inferred from mini.

**Handoff risk to call out.** Python `ZipFile` central-directory/handle contention,
worker fork/spawn semantics, random-read amplification, and whether metadata live
outside versus inside the blob archives.

---

### S02 — CL P0 correctness

**Kickoff.** “Fix only correctness blockers that invalidate CL results: per-sample
pillar caps and the approved Gaussian radius semantics; add adversarial batch and
target golden tests.”

**Owns.** `models/fusion/lidar_encoder.py`, `models/fusion/losses.py`, and focused
tests. It does not redesign sparse SECOND, camera, head, or trainer code.

**Deliverables.**

- per-sample `max_pillars`, deterministic within-sample selection, and per-sample
  truncation/occupancy diagnostics;
- explicit approved Gaussian equation, numeric golden cases, target heatmap tests;
- batch permutation, sample isolation, empty-input, over-cap, and B=1/B>1 tests;
- migration note that old checkpoints used different targets and require retraining.

**Gate.**

- reordering samples in a batch only reorders outputs;
- adding another sample cannot change an existing sample's LiDAR BEV;
- no sample exceeds its own cap and truncation rates are observable;
- golden Gaussian/heatmap values match the owner-approved reference;
- focused unit suite and one GPU forward/backward pass succeed.

---

### S03 — Camera branch architecture

**Kickoff.** “Build a genuinely independent, effective multi-scale camera encoder
for the primary CL model: fix dead FPN levels, stride/resolution flow, image
augmentation geometry, and camera-only gradient coverage.”

**Owns.** Camera-specific modules (`camera_backbone.py`, `camera_neck.py`,
`preprocess.py`, `view_transform.py`) and focused tests. To avoid parallel conflicts,
it exposes a module/output contract but does not wire `detector.py` or `tasks.py`.

**Candidate contract, not yet frozen.** Swin-T; valid stride-8 multi-scale output;
0.5 m depth bins;
aspect-preserving resize/crop with calibration consistency; pure-camera view
transform without LiDAR-conditioned inputs. The exact kickoff must either record
owner approval for this implementation or limit S03 to evidence/interface work.

**Deliverables.**

- no computed camera level is permanently disconnected;
- exact calibration transformations for resize/crop/flip/rotation;
- configurable augmentation with deterministic validation path;
- camera-only feature/BEV shape and dtype contract for S07;
- fixed-batch gradient coverage, LiDAR-invariance, tiny-overfit, and memory profile.

**Gate.** Every intended camera parameter has finite gradient, changing camera
pixels changes camera BEV, changing LiDAR does not, projection residual stays within
the declared tolerance, and the 100-step camera-only engineering loss decreases.

**Deferred.** New transformer families, temporal camera fusion, modality-conditioned
depth, or extensive depth-supervision ablations.

---

### S04 — LiDAR SECOND architecture

**Kickoff.** “Replace the current Z-only sparse path with a SECOND-style sparse
LiDAR encoder that downsamples XY before densification and exposes a low-resolution
BEV contract suitable for fusion.”

**Owns.** `sparse_voxel_encoder.py` or a new focused SECOND module, LiDAR-specific
support code, and tests. It does not wire detector/trainer entry points.

**Candidate contract, not yet frozen.** Voxel candidate `0.075x0.075x0.2 m`; sparse
3D stages with
approximately 8x XY reduction; densification only at the reduced grid; train/eval
max-voxel settings separately configurable; fp16 sparse convolution with fp32
reference support. The exact kickoff must either record owner approval for this
implementation or limit S04 to evidence/interface work.

**Deliverables.**

- documented coordinate order, spatial shapes, output stride, receptive field, and
  camera-fusion alignment contract;
- batched/per-sample-safe voxelization and truncation metrics;
- empty input and extreme occupancy handling;
- fixed-batch branch delta, gradient, tiny-overfit, sparse/dense dtype, memory, and
  throughput diagnostics.

**Gate.** No 1440x1440 dense/fusion tensor; B=4 GH200 engineering forward/backward
fits the performance envelope; output geometry maps back to metric coordinates;
fp16/fp32 are finite; sample/batch isolation passes.

**Deferred.** Sweeping many voxel sizes. `L-S020` is a later scientific mechanism
control if resources allow.

---

### S05 — Detection head and decode

**Kickoff.** “Implement a reference-faithful but framework-independent multi-task
CenterHead and deterministic task/class-aware box decode/NMS; validate coordinates
against nuScenes conventions.”

**Owns.** `models/fusion/head.py`, a separable decode/NMS module if needed, head/loss
interfaces, and focused tests. It does not edit the production detector wiring.

**Deliverables.**

- approved task grouping and separate heads for heatmap/regression fields;
- class/task candidate budgets rather than a single global 10-class top-K;
- deterministic circle/rotate NMS with explicit thresholds;
- canonical box dimension/yaw/velocity conversion and round-trip fixtures;
- decode-only comparison on an existing checkpoint where interface-compatible.

**Gate.** Reference fixture parity for target rendering/decode/NMS, stable output
under input-order permutations, no cross-class candidate starvation, no duplicate
box explosion, and official nuScenes submission conversion passes.

**Contingency.** TransFusion is opened only by S00 if a correct CenterHead stack
cannot reach the absolute CL gates or a second structure is needed for generality.

---

### S06 — Production modes, config, and training runtime

**Kickoff.** “Make camera-only, LiDAR-only, and fusion first-class production
topologies with one fail-closed resolved config and scientifically resumable,
profileable training/evaluation.”

**Owns.** `models/fusion/detector.py`, `training/tasks.py`, `training/loop.py`,
`scripts/centralized_train.py`, `eval/detection_eval.py`, runtime/config/provenance,
and focused tests. It integrates against declared placeholder contracts and leaves
final module wiring to S07.

**Deliverables.**

- explicit `model-mode = camera_only | lidar_only | fusion`;
- unused modality is neither loaded nor executed; evaluation submission metadata
  records actual modality;
- fail-closed architecture enums and a canonical resolved-config hash;
- gradient accumulation with schedules defined by executed optimizer steps;
- persistent loader/sampler across epochs with deterministic `set_epoch` behavior;
- correct non-finite skip semantics and optimizer/scheduler/EMA synchronization;
- full checkpoint/resume of model, optimizer, scheduler, GradScaler, EMA, epoch,
  optimizer-step count, config hash, and data manifest;
- configured eval autocast and timing instrumentation.

**Gate.** Continuous versus interrupted/resumed runs match within the approved
scientific tolerance; no accidental branch I/O; effective global batch and
optimizer-step budget are identical across experiment cells; provenance rejects
architecture/data/precision drift.

---

### S07 — Integrated engineering gate and performance profile

**Kickoff.** “Integrate only reviewed S01-S06 outputs into one resolved CL stack,
close shape/dtype/geometry contracts, and prove the stack is ready for full runs.”

**Owns.** Integration edits across the production detector/task/config/launchers,
candidate configs, and the full engineering verification. No new architecture idea
is introduced here.

**Deliverables.**

- one integrated commit and resolved configs for `C-STR8`, `L-P020`, `L-S075`,
  `F-U`, and `F-CBGS`;
- directory/ZIP, B=1/B=4/B=16, branch-mode, gradient, geometry, precision, resume,
  and official-eval smoke results;
- 100-step and 1000-step capped trainval runs for every topology that will receive
  a full job;
- full-data p50/p95 step, stage timings, memory, data wait, GPU utilization, and
  evaluation profile;
- estimated GPU-hours and Slurm launch order for S08-S11.

**Gate.** Every pre-full-run gate in `ORCHESTRA.md` passes. A failed gate returns to
the owning worker session; S07 does not waive it.

**Compute authorization.** S07 may prepare capped/full-data profiling requests but
does not submit Slurm jobs until the exact `RUN_REQUEST.md` is owner-approved.

---

### S08 — Camera-only scientific run

**Kickoff.** “Train and evaluate the frozen `C-STR8` topology on full trainval/full
val under the matched protocol; diagnose only with predeclared metrics and slices.”

**Inputs.** S07 commit/config hash, seed S0, effective batch and optimizer-step
budget, public pretraining decision, fixed decode/eval.

**Outputs.** Checkpoint plus optimizer/EMA state, complete provenance, learning
curves, mAP/NDS, ten class APs, five TP errors, recall, condition slices, throughput,
memory, and failure analysis.

**Draft gate, not yet approved.** The current planning values are `mAP >= 0.32` and
`NDS >= 0.38`, with no collapsed intended security-target class, finite training,
and reproducible official evaluation. S00 must replace or confirm the numerical
values with the owner before the run request. Failure triggers a bounded diagnosis;
it does not authorize an unplanned architecture sweep.

**Compute authorization.** The full run requires a session-specific owner-approved
`RUN_REQUEST.md`; approval of the session plan is not approval of the job.

---

### S09 — LiDAR-only scientific selection

**Kickoff.** “Run the repaired pillar control and proper SECOND candidate under one
matched full-data protocol, select the LiDAR branch, and separate accuracy from
cost.”

**Candidate cells, not yet approved.** `L-P020` and `L-S075`; optional `L-S020`
only if S00 and the owner approve the extra mechanism/isolation value.

**Outputs.** Same scientific and performance bundle as S08, including per-sample
voxel/pillar truncation statistics and range/point-count slices.

**Draft gate, not yet approved.** The current planning value is
`0.61 mAP / 0.66 NDS`; the selected model must also be finite, fit the frozen FL
performance envelope, and have no unexplained geometry/class collapse. S00 must
replace or confirm the numerical values with the owner before the run request.
Selection uses accuracy, NDS components, and projected FL cost—not mAP alone.

**Compute authorization.** Each required/optional cell and seed must appear in an
owner-approved `RUN_REQUEST.md`. An optional cell is never inferred from spare GPU
capacity.

---

### S10 — Fusion, class-balance recipe, and `CL-PILOT`

**Kickoff.** “Train the selected C/L fusion, prove genuine modality complementarity,
select class balancing, resolve initialization semantics, and issue or deny the
`CL-PILOT` unlock.”

**Candidate cells, not yet approved.** `F-U` and `F-CBGS`, where CBGS replaces
rather than stacks with class weights and the schedule is adjusted for expanded
optimizer steps. Add the public/joint versus CL branch-warm-start A/B only if the
owner approves it after the initialization policy review.

**Evaluation.** Independent C/L/F comparison plus same-checkpoint camera-zero,
LiDAR-zero, camera shuffle/misalignment, range, point-count, visibility, day/night,
and rain slices.

**Draft gate for `CL-PILOT`, not yet approved.** S00 must replace or confirm these
values with the owner before any S10 run whose result they judge:

- fusion at least `0.64 mAP / 0.68 NDS` (absolute no-go below `0.62/0.66`);
- paired gain over selected LiDAR at least `+0.02 mAP / +0.01 NDS`;
- scene-bootstrap 95% CI lower bound for the gain above zero;
- at least 7/10 classes do not regress and no unexplained class loss above 0.02;
- modality interventions demonstrate nontrivial causal camera and LiDAR use;
- performance gate passes.

**Output.** One selected fusion recipe, S0 checkpoint/config/provenance hashes,
explicit CL/FL initialization interpretation, and `CL-PILOT = PASS|FAIL`.

The selected full-train checkpoint is a capability artifact. It is not automatically
the Protocol-B `W_base`, because it has seen data that may later belong to
`D_tail`.

---

### S11 — Final CL seeds and capability freeze

**Kickoff.** “Complete final C/L/F replication, freeze every architecture/training/
evaluation field, and produce the capability evidence and immutable architecture
contract from which the approved FL protocol will retrain its allowed initializer.”

**Candidate runs, not yet approved.** If the owner selects the three-seed design,
add S1 and S2 for the selected C, L, and F recipes; together with S0 this is C/L/F
x3 seeds. No architecture or metric changes after S0 without invalidating the seed
set.

**Outputs.** Mean/std, paired seed deltas, scene-bootstrap CIs, full metric/slice/
performance tables, checksummed checkpoints, immutable resolved configs, manifest,
parameter/upload sizes, projected FL round cost, and a clear separation between
full-train capability checkpoints and protocol-valid initializers.

**Gate for `CL-FREEZE`.** All absolute/fusion/performance gates pass across the
declared seed interpretation; checkpoint reload/eval is reproducible; architecture,
precision, data, and decoder hashes are frozen. S11 does not choose or silently
reuse a full-train checkpoint as the Protocol-B base model. S12/S13 use the frozen
architecture and approved split to construct the protocol-valid initialization.

**Compute authorization.** The six additional seed runs require an explicit matrix
approval; `CL-PILOT` does not authorize them automatically.

---

### S12 — FL protocols, long-tail split, threat model, and paper skeleton

**Kickoff.** “In parallel with CL engineering, turn Protocol A and the owner-approved
primary vendor-style Protocol B into an auditable experimental contract: define base/tail
ownership, clients, clean baselines, threat model, novelty boundary, RQs, evidence
table, and paper skeleton—without claiming CL engineering as novelty.”

**Owns.** Literature/claim map, data/split specification, FL protocol contract, and
draft paper structure. It may build read-only split statistics/proposals, but it
does not train models or submit jobs.

**Required decisions.**

- attacker visibility/capability, data versus model poisoning, malicious fraction,
  client participation, secure-aggregation compatibility, digital/physical scope;
- Protocol B as the locked primary setting; Protocol A's exact meaning of
  nuScenes-scratch/public initialization and its role as the clean control;
- train-only scene/log-disjoint `D_base`/`D_tail`, frozen tail criteria, regional/
  fleet client unit, client partition, update scope, and split hashes;
- clean baselines: `W_base`, centralized pooled-tail oracle, local-only, clean FL,
  attacked FL, and defended FL;
- overall/common/tail utility, catastrophic forgetting, client dispersion, ASR,
  false-trigger rate, defense FPR, and communication/compute metrics;
- primary mechanism hypothesis: modality-localized update energy hidden by benign
  high-dimensional/geographic drift;
- RQs for CL-to-FL behavior, block-level causal localization, robust-aggregator
  failure, and structure-aware defense;
- minimum generic aggregators, adaptive attacker, clean-utility/FPR metrics, and
  generalization structure;
- tentative fixed title, author list, topics, abstract, and figure/table inventory.

**Gate.** No scene/sample/sweep leakage; validation/test untouched; tail definition
is train-only and attack-independent; the full-train capability checkpoint is not
used as `W_base`; each proposed claim maps to an experiment and artifact; BadFusion,
multimodal backdoors, federated multimodal AD, generic FL backdoors/defenses, and
prior autonomous-driving security are explicitly distinguished. No unsupported
“first” claim enters the abstract.

---

### S13 — Clean FL baselines, modality-localized attack, and mechanism

**Kickoff.** “Using the frozen architecture and S12's approved data protocol, first
establish capable clean federated tail adaptation, then implant a camera- and/or
LiDAR-localized backdoor and causally localize its updates by module. Protocol A is
the optimization/control setting; Protocol B is the primary setting.”

**Starts when.** Interface work/smoke may start after S07. Split construction and
scientific jobs require S12 PASS; model runs require S10 `CL-PILOT`; final paper
numbers are rerun/verified after `CL-FREEZE`.

**Phases and outputs.**

1. Retrain the frozen architecture on `D_base` to produce protocol-valid `W_base`.
2. Establish the centralized pooled-tail oracle, local-only, and clean federated
   fine-tuning baselines; quantify common retention, tail gain, forgetting, and the
   gap to the oracle. Protocol A is run as the separately labeled clean control.
3. Only after the clean adaptation utility gate passes, run the attacked cells with
   strict clean-correct eligible ASR, clean utility, false-trigger rate, and matched
   malicious sample/update budgets.
4. Record attack persistence, per-block update norm/direction/spectrum, block
   replacement/intervention evidence, and generic robust-aggregator failure.

**Gate.** Clean federated adaptation improves the declared tail metrics without
unacceptable common-data forgetting and remains sufficiently close to its matched
central oracle; otherwise attack results are diagnostic only. One attack is then
viable and stealthy on the capable fused model; the effect cannot be explained by
data leakage, weak clean detection, unequal poison exposure, ordinary occlusion, or
one anomalous seed. The module-local mechanism is measured, not asserted from
parameter names.

**Compute authorization.** Clean and attacked matrices are separate requests. A
clean-baseline approval cannot be reused for attack jobs, extra seeds, or reruns.

---

### S14 — Structure-aware defense, adaptive attack, and generalization

**Kickoff.** “Design the minimum defense that directly targets the measured
modality/module-local mechanism, then evaluate it against a defense-aware adaptive
attacker and benign geographic outliers.”

**Outputs.** Fairly tuned generic-defense baselines, module-/modality-aware defense,
counterfactual modality probes if allowed by the server threat model, adaptive
evasion, rare-class/client-level clean harm, FPR, overhead, and at least one
structure/generalization check.

**Gate.** Defense reduces ASR while satisfying a predeclared clean-utility/FPR
budget, beats the strongest fair generic baseline, survives an adaptive attacker,
and states any incompatibility with secure aggregation. A defense that assumes
individual updates must declare that system requirement.

---

### S15 — USENIX '27 paper and artifact

**Kickoff.** “Maintain and freeze a complete USENIX Security '27 submission and
artifact from the approved threat model and checksummed results; reject any number
that lacks provenance.”

**Starts when.** Skeleton, related work, artifact layout, and evaluation scripts
start with S12. Final result tables depend on S11, S13, and S14.

**Deliverables.**

- fixed title/authors/topics/nonblank abstract ready before 2026-08-18 AoE;
- threat model, ethics, limitations, Open Science appendix, and anonymous artifact;
- scripts/manifests that regenerate every table/figure from checksummed raw outputs;
- claim-evidence matrix, reproducibility dry run, reference audit, and anonymization
  audit;
- final paper/artifact frozen before 2026-08-25 AoE.

**Gate.** No smoke/mini result is presented as science; no stale checkpoint/config
is mixed into a table; all attack/defense comparisons share declared clean baseline,
seed, participation, precision, and data semantics; citations and novelty claims are
manually verified.

## 4. Suggested launch order after owner review

1. Launch S01-S05 and S12 immediately in separate sessions/worktrees.
2. Launch S06 once S00 approves the model-mode and resolved-config interfaces.
3. Open review sessions S01-R through S06-R as workers return; do not wait for all
   workers before reviewing completed ones.
4. Launch S07 as the sole integration session after approved diffs are available.
5. Submit S08 and S09 jobs concurrently after S07 passes.
6. Launch S10 as soon as the selected branch checkpoints arrive.
7. Finish S12's protocol/split review; on `CL-PILOT`, start S13's clean baselines
   while S11 finishes remaining CL seeds.
8. Start S14 after the first viable S13 mechanism cell; keep S15 active throughout.

## 5. Durable delivery and Orchestra review

Every worker writes to
`fl_v3/usenix27_orchestra/handoffs/Sxx/` on its scoped branch:

| File | When | Required content |
|---|---|---|
| `HANDOFF.md` | every session completion | base/branch/commit, exact files/semantics, references, tests, gates, hashes, negative findings, allowed/forbidden claims, residual risks |
| `RUN_REQUEST.md` | before any Slurm/material compute | exact immutable commit/config/data manifest, cells, seeds, GPU/time budget, command/output, stop criteria, explicit owner approval state |
| `RESULTS.md` | every execution session | all job IDs/statuses, raw artifact paths/checksums, full result/performance table including failures, interpretation limits |
| `REVIEW.md` | independent `Sxx-R` session | severity-ordered findings, code/data/metric trace, adversarial checks, gate verdict, residual risk, integration/scientific-use decision |

The worker may report `PASS` only as a self-assessment. Integration or scientific
use requires `REVIEW.md` plus Orchestra/owner acceptance. The Orchestra checks the
actual diff and raw artifacts; it does not rely only on prose summaries.

Every review explicitly covers:

- scene/sample/sweep ownership and train/val/test leakage;
- coordinate frames, calibration, yaw/dimensions, class mapping, units;
- per-sample/batch invariance and empty/truncation cases;
- resolved config and branch actually executed, not only requested config;
- effective optimizer steps, batch/exposure, precision, scheduler/EMA/resume;
- official metric conversion, eligibility/ASR denominator, thresholds/NMS;
- missing, failed, excluded, or negative cells and post-hoc choices;
- compute and communication accounting;
- shortcuts or hidden behavior that could inflate clean performance, fusion gain,
  ASR, or defense effectiveness.

Any changed commit, config, split, cell, seed, resource request, or command after
approval invalidates `RUN_REQUEST.md` and requires new owner permission. There is no
automatic resubmission or opportunistic filling of spare GPUs.

The durable completion sequence is:

1. worker writes the complete handoff package and reports its exact SHA/diff;
2. S00 performs a completeness/provenance check;
3. the owner explicitly authorizes any local handoff commit/branch required to
   produce a durable `WORKER_SHA`; this does not authorize merge or push;
4. the owner creates an independent UI `Sxx-R` task to review that exact version;
5. findings are fixed/re-reviewed or the review is accepted;
6. S00 updates status and the change-control ledger;
7. S00 refines affected unstarted session plans/kickoffs within its authority;
8. the owner approves any material/locked change before the revised kickoff.

## 6. Worker kickoff checklist

Copy this into every worker prompt:

```text
Read completely: repository AGENTS.md, fl_v3/docs/env.md,
fl_v3/usenix27_orchestra/ORCHESTRA.md, fl_v3/usenix27_orchestra/KICKOFFS.md,
and your Sxx section in SESSIONS.md.

Do not edit ORCHESTRA.md, SESSIONS.md, or KICKOFFS.md. Do not modify fl_v2 or write
new work records under the read-only fl_v3/collab/. Stay inside your declared file
ownership; return integration needs instead of editing another session's files.
Preserve unrelated dirty work. Mini is engineering-only.

Create handoffs/Sxx/HANDOFF.md. Before any Slurm/material compute, create
RUN_REQUEST.md and stop until the owner explicitly approves the exact request.
Never infer full-run/matrix/upload permission from approval of a design or session.

Your kickoff must provide BASE_SHA, SOURCE_BRANCH, EXPECTED_REF_MODE, FILE_OWNERSHIP,
and UPSTREAM_HANDOFFS_AND_SHAS. Do not create, switch, move, remove, or prune a
worktree/branch. A Codex-managed detached HEAD is valid if it equals BASE_SHA.
Before editing, verify and report repository root, HEAD, ref mode, git status, exact
intended files, and any contract ambiguity; stop on a real mismatch.
After editing: run focused CPU checks and the smallest relevant owner-authorized
GH200/Slurm gate; return the standard handoff from ORCHESTRA.md with exact commands,
job IDs, outputs, hashes, and residual risks. Do not commit/merge/push without owner
authorization.
```
