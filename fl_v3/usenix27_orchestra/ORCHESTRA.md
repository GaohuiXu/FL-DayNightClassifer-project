# USENIX Security '27 Orchestra — strong CL backbone to federated multimodal security

> **Status:** active. The owner designated the pinned S00 task at
> `detached@f262f6bea037580065a8505008773c04fdd259f5` as the sole canonical writer
> on 2026-07-10. S01 is independently reviewed PASS and accepted as an S07
> dependency at worker SHA `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`; review
> artifact SHA `7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc` remains a separate
> review-only commit. No S01 merge or push has occurred. S12 has delivered its
> evidence/proposal-only handoff for completeness/review and used no compute.
> Remaining architecture choices, cells, and run requests still require their own
> approvals.
> **Venue target:** USENIX Security '27 first submission cycle (registration 2026-08-18 AoE;
> submission 2026-08-25 AoE).
> **Canonical companions:** [`SESSIONS.md`](SESSIONS.md) and
> [`KICKOFFS.md`](KICKOFFS.md).
> **Scope rule:** this directory is a top-level `fl_v3/` workspace, parallel to
> `fl_v3/collab/`. The entire `collab/` tree is now read-only historical evidence
> for this stage; it may be read and cited but receives no new planning, handoff,
> review, or result documents.

## 1. Decision and research objective

We will sprint for the USENIX Security '27 first submission cycle. The immediate
critical path is a strong, trustworthy centralized (CL) nuScenes detector. Attack
and defense claims cannot be interpreted on a detector whose clean accuracy,
modality use, or training semantics are uncertain.

The CL work is an enabling validity requirement, not the paper's principal
security contribution. The intended paper question remains:

> During vendor-style federated adaptation of a strong camera-LiDAR detector to
> rare, geographically and environmentally non-IID fleet data, can a
> modality-localized backdoor hide among legitimate long-tail updates, and can a
> module-/modality-aware defense remove it without rejecting the rare benign
> updates that the adaptation process exists to learn?

The USENIX '27 sprint schedule therefore overlaps three streams:

1. **CL foundation:** fix correctness, choose and freeze a strong modular backbone.
2. **Security mechanism:** prepare the threat model and attack instrumentation in
   parallel; start scientific attack/FL runs as soon as the single-seed fusion
   pilot passes, without waiting for all final CL seeds.
3. **Paper/artifact:** maintain the argument, tables, provenance, and artifact from
   the beginning rather than reconstructing them at the deadline.

## 2. Current baseline: what is and is not trusted

The strongest historical CL result is `bb02d = 0.5656 mAP / 0.5733 NDS` on the full
nuScenes validation split. It is a useful engineering milestone, but it is not the
new FL handoff baseline:

- it predates the newly identified batch-global pillar-cap and Gaussian-target
  findings;
- no valid full-trainval camera-only and LiDAR-only models were independently
  trained, so branch capacity and fusion complementarity are unknown;
- the current sparse 0.075 m implementation does not have the reference SECOND
  resolution flow;
- the historical description that camera/fusion/head are essentially identical
  to MIT BEVFusion is not supported by the current code/reference comparison.

MIT BEVFusion provides orientation values, not a requirement to clone every
component: camera-only `0.3556/0.4121`, LiDAR-only `0.6468/0.6928`, and fusion
`0.6852/0.7138` mAP/NDS. Camera is expected to be weaker than LiDAR; the requirement
is that each branch is competitive for its modality and that fusion reliably
improves over the stronger branch.

### Confirmed blockers before any new full training

1. **Batch-global pillar truncation.** `max_pillars` is applied after sorting a
   batch-global key, so earlier samples are favored and later samples can lose
   LiDAR content (`models/fusion/lidar_encoder.py:105-180`).
2. **Unresolved Gaussian radius semantics.** The current formula mixes a corrected
   quadratic denominator with large roots and matches neither the official legacy
   implementation nor the geometric small-root interpretation
   (`models/fusion/losses.py:37-59`). The intended definition must be chosen,
   pinned, and covered by numerical golden tests.
3. **Dead camera FPN levels.** With `out_stride=16`, the returned tensor depends
   only on Swin levels 2/3; levels 0/1 are computed and discarded
   (`models/fusion/camera_neck.py:61-71`).
4. **Incomplete detection head/decode.** One shared 10-class heatmap and
   class-agnostic regression feed a global top-K; there is no task-wise box NMS
   (`models/fusion/head.py`, `models/fusion/detector.py:180-217`).
5. **Ambiguous architecture resolution.** Config defaults and entry points can
   select pillar versus voxel differently; unknown encoder strings can silently
   fall back to pillar (`training/tasks.py:363-426`).
6. **No production single-modality topology.** Mini diagnostics can mask branches,
   but production training/evaluation always constructs and executes both branches.
7. **ZIP backend accepted but not yet integrated/frozen.** S01/S01-R passed at
   `abe5c58`; historical job `332651` proves ten-archive member coverage and loader
   determinism, while remediation job `333206` proves `t1.v2` depth binding plus
   real-mini directory/ZIP and fork/spawn parity. Before training, S07-A must land
   the reviewed implementation/review artifact, migrate `build_gt_database.py`
   away from hardcoded `t1.v1`, generate and freeze full trainval `t1.v2` caches,
   and preserve the exact manifest/cache hashes in provenance.

### Capability limitations to resolve during architecture freeze

- Camera uses stride-16 features and 1 m depth bins; the reference camera path
  uses an effective stride 8 and 0.5 m depth bins.
- Native 1600x900 images are anisotropically stretched to 704x256. Calibration is
  updated correctly, but pretrained visual appearance is distorted and the
  reference resize/crop/rotation augmentation is absent.
- The sparse encoder downsamples only Z, densifies at the original XY grid, and
  fuses there. For 0.075 m, the current 1440x1440 grid is incorrectly treated as a
  fusion grid; in SECOND it is an input voxel grid and is reduced by approximately
  8x before densification/fusion.
- The 128-channel fuser, shallow BEV neck, and 0.15 M shared head are materially
  smaller/simpler than the reference fused detector.
- Class weights multiply both positive and negative heatmap terms. Combining them
  with CBGS double-balances classes and invalidates the old CBGS negative result.

### Performance limitations to resolve

- The current 0.075 m path uses about 59.2 GiB for a B=2 forward-only probe and
  OOMs at B=4 because it densifies/fuses at 1440x1440.
- Large BEV tensors remain fp32 after view-transform accumulation.
- Pillar encoding performs repeated stable sorts and constructs padded
  `[pillars,max_points,channels]` storage before applying the pillar cap.
- Sparse voxelization loops over samples in Python.
- The centralized trainer rebuilds the DataLoader each epoch, defeating persistent
  workers and repeatedly reopening ZIP metadata.
- Official detection evaluation does not currently enter the configured autocast
  context.
- EMA history is not restored on resume; fp32 non-finite loss handling can still
  take an optimizer step.

## 3. Candidate backbone contract — not yet frozen

The following is the current evidence-based candidate, not an owner-approved final
architecture. It gives S03-S06 a concrete design to audit and estimate. S00 may
launch reference analysis, interface design, and reversible engineering checks, but
must obtain the relevant stage-gate decision in Section 10 before a worker commits
to a primary implementation choice or S07 integrates it.

The candidate primary model is a strong, modular late-BEV detector whose modality
boundaries remain meaningful for attacks and defenses:

- **Camera:** Swin-T initially; effective multi-scale FPN output at stride 8;
  pure-camera LSS/view transform; 0.5 m depth bins; aspect-preserving resize/crop
  and reference-style image augmentation.
- **LiDAR:** SECOND-style sparse encoder at a primary candidate voxel size of
  `0.075 x 0.075 x 0.2 m`; XY downsampling in sparse space to about 1/8 resolution
  before densification.
- **Fusion:** modality-native encoders map into one common low-resolution BEV
  (about 180x180 for the 1440 input grid); 256-channel ConvFuser or an equally
  explicit late-BEV fusion block.
- **Detection:** a faithful multi-task CenterHead with task/class-aware candidate
  selection and rotate/circle NMS is the primary simple baseline. TransFusion is a
  contingency or later generalization backbone, not a prerequisite for starting.
- **Precision:** Arrhenius sparse training uses fp16 AMP plus GradScaler; fp32 is the
  debug/reference mode. Direct sparse bf16 remains unsupported.

We should not blindly copy a LiDAR-conditioned camera depth transform: it would
blur the camera/LiDAR attack boundary. Any cross-conditioned view transform must be
an explicit later experiment, not the primary security backbone.

## 4. CL scientific matrix and gates

### Pre-full-run engineering gate

All must pass before Slurm full training:

- directory/ZIP decode parity and 100% train/val image, key-LiDAR, and 10-sweep
  member coverage;
- per-sample pillar cap plus sample/batch permutation invariance;
- pinned Gaussian golden values and target-render tests;
- production `camera_only`, `lidar_only`, and `fusion` modes that skip unused I/O
  and computation;
- intended trainable parameters all receive finite gradients;
- identical resolved architecture/config hashes across CL, FL, train, resume, and
  eval entry points;
- 100-step and 1000-step capped trainval runs decrease loss without OOM, NaN,
  persistent GradScaler skips, or optimizer/scheduler/EMA step drift;
- full-data, not mini-only, throughput and memory profile.

### Candidate single-seed selection matrix — pending owner decision

The current planning proposal uses the full 28,130-sample train split, full
validation split, one screening seed, matched effective global batch/optimizer
steps/precision/decode, and approximately 20-24 epochs unless a step-based budget
is approved. These cells and budgets are not yet an approved matrix.

If approved, this matrix is the **backbone capability/architecture benchmark**. It
does not by
itself define the final federated data protocol, and its full-train weights must not
be reused as the base checkpoint of a tail-adaptation experiment if they have seen
the client-tail partition. After the architecture is frozen, the final federated
base model is retrained using only the data allowed by the approved FL protocol.

Listing a cell here is not permission to submit it. Every full run or matrix needs
an owner-approved `RUN_REQUEST.md` bound to an exact commit, resolved config, data
manifest, cells, seeds, GPU/time request, and output path.

| Cell | Purpose |
|---|---|
| `L-P020` | repaired legacy 0.2 m pillar LiDAR-only control |
| `L-S075` | SECOND-style 0.075x0.075x0.2 LiDAR-only primary candidate |
| `C-STR8` | corrected stride-8, multi-scale camera-only candidate |
| `F-U` | selected C/L fusion with uniform heatmap weighting and no CBGS |
| `F-CBGS` | CBGS replaces class weights; schedule corrected for expanded steps |

If the future FL initialization policy is not fixed before these jobs, add one
fusion initialization A/B: public/joint initialization versus CL branch warm-start.
That proposal would give 12 rather than 11 full runs after final seeds; neither run
count is currently approved.

### Draft handoff floors — not yet pre-registered

These are planning values relative to a matched-complexity reference, not approved
gates or SOTA claims. S00 may revise the recommendation using reviewed S07
engineering/profile evidence, but the owner must freeze the numerical gate before
the corresponding S08/S09/S10 scientific run is approved.

| Model | mAP / NDS gate |
|---|---:|
| Camera-only | at least `0.32 / 0.38` |
| LiDAR-only | at least `0.61 / 0.66` |
| Fusion no-go | below `0.62 / 0.66` |
| Fusion handoff target | at least `0.64 / 0.68` |
| Fusion stretch target | at least `0.67 / 0.70` |

Fusion must also exceed the independently trained paired-seed LiDAR model by at
least `+0.02 mAP / +0.01 NDS`, with scene-bootstrap 95% CI lower bound above zero.
At least 7/10 classes should not regress, and an unexplained class drop above 0.02
is a blocking finding.

Every selected checkpoint reports mAP/NDS, per-class AP, all five nuScenes TP
errors, recall, train/validation curves, and range/visibility/number-of-LiDAR-points/
day-night/rain slices.

### Two different modality tests

Both are mandatory:

1. **Independent C/L/F training** measures the capacity of each topology.
2. **Same-fusion-checkpoint intervention** (`camera-zero`, `lidar-zero`, camera
   shuffle/misalignment) measures what the fused model actually uses.

Define `fusion_gain = F - max(C,L)` globally and for each slice. Modality masking
does not replace independently trained branch models.

### Draft performance gate for FL delivery — pending owner decision

These values are an engineering proposal. The instrumentation is required, while
the numerical acceptance thresholds may be revised from reviewed S07 profiling and
then frozen before capability runs are judged.

- chosen microbatch at least 4/GH200, or accumulation reaches the matched effective
  batch without schedule drift;
- peak allocated/reserved memory below 80% of device memory;
- data wait below 15% and p95 step time below 1.25x p50 after warm-up;
- no non-finite updates and at least 99% intended optimizer steps executed;
- report p50/p95 step time, samples/s, epoch wall time, GPU utilization, data wait,
  peak memory, CPU/RSS, and evaluation time;
- reject a candidate that gains less than 1 mAP while making training/FL steps more
  than 2x slower, unless it is required for a security generalization claim;
- record parameter/upload bytes because every added parameter is multiplied by
  clients and rounds.

## 5. CL-to-FL protocols

The two protocols answer different research questions and must never share a label,
baseline, or checkpoint provenance. The owner has approved **Protocol B as the
primary security setting** and **Protocol A as the clean optimization/control
setting**. Any reversal or mixed protocol requires an explicit new owner decision.

### Protocol A — nuScenes-scratch federated training

Clients receive the **frozen architecture**, not a detector trained on nuScenes.
Every client starts from one identical initialization and jointly trains the global
model over client-partitioned nuScenes data.

“From scratch” must be resolved into one of the following before a run:

- **nuScenes-scratch/public-init:** public ImageNet or explicitly allowed NuImages
  camera pretraining may be used, while no weight has seen nuScenes detection data;
  LiDAR/fusion/head follow the declared common initialization. This is the useful
  primary version of Protocol A.
- **fully random:** all trainable detector weights are random. This is an optional
  optimization stress test, not a default realism claim.

Required controls are a centralized pooled model from the same initialization and
data, identical effective data exposure/optimizer-step accounting, and clean
FedAvg/FedAdam or another predeclared optimizer comparison. Protocol A asks whether
FL optimization can recover the strong centralized capability. If it produces a
weak model, attack/defense results on that model are diagnostic only. A utility gate
relative to the matched centralized model and an absolute clean-detection floor
must be approved before security claims are unlocked.

Protocol A is scientifically useful, but making from-scratch FL match CL is a large
optimization problem of its own. It should not silently consume or replace the
paper's security contribution.

### Protocol B — centralized base plus federated tail adaptation (primary)

This protocol closely matches a plausible autonomous-driving vendor workflow:

1. The vendor owns a broad, common-data pool `D_base` and trains a strong base
   detector `W_base` using the already validated architecture.
2. Regional fleet centers, data silos, or edge training clusters collect new rare
   classes, conditions, locations, weather, sensor regimes, or failure cases in a
   disjoint `D_tail`.
3. Every client receives the same `W_base` and fine-tunes on its local tail data;
   the server aggregates these updates.
4. A malicious client poisons its legitimate tail-adaptation stream or its model
   update. Defenses must distinguish that attack from benign tail updates.

The recommended system unit is a **regional/fleet data silo**, not a claim that each
car trains the full detector onboard. Full-model fine-tuning, selected-module
fine-tuning, and parameter-efficient updates have different bandwidth and attack
surfaces; the primary update scope must be frozen in the threat model.

Protocol B is particularly valuable for the paper because benign long-tail updates
are naturally sparse, non-IID, and sometimes module-local. They may resemble the
updates that a modality-localized attacker uses. This gives a concrete systems
security tension: a whole-model robust aggregator may either miss the attacker or
reject exactly the rare benign data the fleet needs to learn.

### Mandatory split and leakage rules for Protocol B

- Build `D_base` and `D_tail` from the official **training** split only. The
  official validation/test data remain completely held out.
- Split at least at **scene/log level**. Adjacent keyframes, sweeps, duplicated raw
  sensor files, and the same scene must not cross base, client, or evaluation
  boundaries.
- Do not create the split by moving individual annotations while the same camera/
  LiDAR frame remains in both pools. One frame can contain common and rare objects;
  the unit of ownership is the raw sample/scene, not an annotation row.
- Define “tail” before attack experiments using frozen train-only rules: class
  frequency, distance/visibility, weather/time, location, sensor condition, or a
  predeclared combination. Do not tune the definition to maximize ASR or use final
  validation outcomes.
- Reserve evaluation support for the declared tail conditions. Apply the already
  frozen tail definition to official validation for reporting; if it has
  insufficient support, reserve a scene/log-disjoint `E_tail` before training and
  exclude it from every client and centralized training pool.
- Freeze and hash the split, eligibility rules, class/condition statistics, and
  client assignment before training. Every experimental cell consumes the same
  artifacts.
- `W_base` must never have trained on `D_tail`. The full-train CL checkpoint used
  for architecture capability is therefore not a valid Protocol-B initializer.
  Retrain the frozen architecture on `D_base` to obtain the scientific `W_base`.
- The base pool must contain enough support for the target classes to make clean
  eligibility meaningful; “tail” should mean rare contexts/examples, not removal
  of all knowledge needed to define the task.
- Once the Protocol-B split/statistics are inspected, the architecture and primary
  recipe cannot be changed to exploit `D_tail` outcomes. The earlier full-train
  capability phase must be disclosed as platform development; its weights and
  scores are not Protocol-B evidence. Any tail-conditioned redesign is a separately
  labeled ablation and requires a clean re-freeze.

### Required clean baselines for Protocol B

All use the frozen split and initialization:

1. `W_base`: CL on `D_base` only.
2. `W_oracle`: centralized continuation on pooled `D_tail` (or `D_base + D_tail`,
   whichever matches the declared exposure) as the non-private upper bound.
3. Local-only client fine-tuning without aggregation.
4. Clean federated fine-tuning on the tail clients.
5. Attacked federated fine-tuning.
6. Defended federated fine-tuning under the same clean baseline and budget.

Report overall mAP/NDS, common-data retention, tail-class/condition improvement,
catastrophic forgetting, client-level dispersion, communication/compute cost, and
then ASR/false-trigger/defense FPR. A defense that suppresses tail learning is not a
successful defense even if aggregate mAP changes little.

### Relationship between the protocols

- Protocol A measures the clean federated optimization gap from a public/common
  initialization and remains an important control.
- Protocol B is the primary threat model because it starts from a
  capable deployed model and places the attacker in the realistic adaptation
  stage.
- Results must be labeled `federated training` versus `federated fine-tuning`.
  Checkpoints, data manifests, and claims cannot be mixed between them.
- The Orchestra records the approved protocol, split, client unit, update scope,
  and initialization before any full FL or security job is authorized.

### Scientific audit and novelty boundary

The deployment logic is credible, but the setting alone is not a novelty claim:

- [FedDrive](https://arxiv.org/abs/2202.13670) already treats autonomous-driving
  client data as heterogeneous visual domains, and
  [FedDrive v2](https://arxiv.org/abs/2309.13336) explicitly studies client label
  skew/class imbalance.
- [Federated Deep Learning Meets Autonomous Vehicle Perception](https://arxiv.org/abs/2206.01748)
  explicitly motivates FL using rare and occluded instances collected by vehicles
  and road sensors.
- [AutoFed](https://tianyuez.github.io/pubs/autofed.pdf) studies multimodal,
  environmental, annotation, and client-selection heterogeneity in federated
  autonomous-driving perception.
- [BalanceFL](https://research.cuhk.edu.hk/en/publications/balancefl-addressing-class-imbalance-in-long-tail-federated-learn-2/)
  shows that federated long-tail/class-imbalance learning is itself an established
  problem.

Therefore the paper contribution cannot be “federated learning learns rare driving
data.” The proposed security novelty is the interaction between **legitimate
long-tail adaptation**, **modality-localized backdoor updates**, **geographic/domain
non-IID drift**, and **structure-aware defense**. S12 must continue a systematic
literature audit before any “first” wording is approved.

## 6. USENIX '27 sprint execution calendar

The calendar is deliberately overlapping and assumes several isolated sessions/
worktrees run in parallel.

| Window | Required outcome |
|---|---|
| Jul 10-12 | owner approves architecture contract, session split, CL/FL initialization policy |
| Jul 11-16 | ZIP backend, P0 fixes, camera/LiDAR/head modules, trainer modes in parallel |
| Jul 16-20 | integration, 100/1000-step gates, full-data profiling, candidate configs frozen |
| Jul 20-29 | camera and LiDAR full single-seed runs; fusion single-seed jobs start as branch checkpoints arrive |
| Jul 27-Aug 5 | fusion recipe selection, remaining CL seeds, `CL-PILOT` then `CL-FREEZE` |
| Jul 29-Aug 12 | preliminary FL/attack mechanism and defense runs on the passing pilot/frozen model |
| Aug 8-17 | adaptive/generalization experiments, result-table freeze, artifact dry run |
| Aug 15-18 | fixed title/authors/topics/abstract approved; mandatory registration by Aug 18 AoE |
| Aug 18-25 | paper and artifact freeze; submission by Aug 25 AoE |

Two distinct unlocks keep the sprint moving:

- **`CL-PILOT`:** one fusion seed passes the absolute and fusion-gain gates. It
  unlocks preliminary FL/attack experiments.
- **`CL-FREEZE`:** C/L/F each have three seeds, the architecture/config/checkpoint
  schema is frozen, and all final paper security results must use this version.

## 7. Session collaboration rules

Detailed session contracts and dependencies are in [`SESSIONS.md`](SESSIONS.md);
copy-ready worker and independent-review startup prompts are in
[`KICKOFFS.md`](KICKOFFS.md). The following rules apply to every session:

1. The Orchestra session is the sole writer of these three canonical documents.
   Worker sessions write their durable handoff package under
   `handoffs/Sxx/` and do not edit the canonical files concurrently.
2. Every implementation session uses an owner-approved, Codex-provisioned isolated
   worktree/branch and declares its file ownership before editing. Integration is
   performed only after review. S00 must instantiate the exact `Sxx` and `Sxx-R`
   prompts from `KICKOFFS.md`, including the pinned base/worker SHA, expected branch,
   and file ownership, so every fresh session receives the required context and
   authorization boundary. Workers and reviewers verify this topology but do not
   create, move, remove, prune, or switch worktrees/branches themselves.
3. Each worker reads `AGENTS.md`, `fl_v3/docs/env.md`, this file, and its session
   contract. Historical `model_capability` conclusions are not treated as current
   requirements where they conflict with this document.
4. Mini is engineering-only. Scientific verdicts use full trainval/full val and
   record hardware, precision, seed, split, resolved config hash, and checkpoint
   checksum.
5. No worker silently changes the data split, effective batch, optimizer steps,
   precision, decoder, or metric. Required changes return to Orchestra for approval.
6. Build and review should be separate sessions for P0, geometry, metric, or
   scientific-result changes. A merge requires objective gate evidence.
7. The Orchestra, not workers, decides whether a negative result kills a candidate,
   requests a rerun, or changes the paper claim.
8. The standing O-009 policy authorizes only bounded, non-scientific engineering
   smoke jobs. No session may submit a full-data gate, full run/evaluation, matrix,
   multi-seed campaign, automatic resubmission, remote upload, or publication
   action without explicit owner permission scoped to the exact action.

### Reasoning effort and standing short-smoke policy

- S00 explicitly passes `xhigh` when creating every new worker/reviewer task; it
  must not silently inherit a host default of `ultra`.
- `ultra` is reserved for a task whose exact envelope contains a recorded reason:
  unusually complex implementation/review or unusually broad, difficult research.
  Ordinary implementation, review, orchestration, and evidence work use `xhigh`.
- Owner decision O-009 is standing authorization for a **short engineering smoke**
  on an Arrhenius compute node. It is non-scientific and must satisfy all of:
  one node, at most one GPU, at most 60 minutes requested walltime per job, at most
  one active job for the session, and at most two cumulative GPU-hours before a new
  owner decision. Job arrays, multi-node/DDP, multiple seeds/cells, full epochs,
  full trainval coverage/profile/evaluation, 100/1000-step gates, and automatic
  resubmission are outside this authorization.
- Before submission the session writes `RUN_REQUEST.md` as an audit record with the
  exact HEAD plus working-tree diff hash, command, bounded data/sample scope,
  resources, output path, and stop criteria, and cites `O-009`. It may then submit
  without waiting for another approval. Every job ID, log, exit status, retry, and
  negative result is recorded in `HANDOFF.md`/`RESULTS.md` as applicable.
- Reaching a boundary above, repeating the same failure twice, or seeking a result
  used for a gate/table/metric returns to S00/owner for exact approval. Full tests,
  full-data runs, profiles, scientific metrics, matrices, seeds, reruns, and spare-
  GPU expansion always require owner review.

### Evidence-driven Orchestra refinement

S00 is expected to refine downstream work as evidence arrives; the 15-session plan
is a controlled dependency graph, not an immutable script. It may draft a
provisional refinement immediately after reading a complete handoff, but it does
not issue a dependent technical kickoff from that refinement until independent
review accepts the evidence. For each completed worker session, S00 follows this
order:

1. inspect the actual diff/artifacts and check that the handoff package is complete;
2. prepare and show the independent `Sxx-R` launch packet from the exact worker
   SHA/diff; after owner authorization, create that review task through Codex;
3. resolve blocking findings or record an accepted review verdict;
4. update the status and decision ledgers;
5. refine the plans and kickoff envelopes of sessions that have not started;
6. request owner approval before any material or locked scientific change.

S00 may independently make **operational refinements**: launch order, dependency
edges, required reading, file ownership that avoids conflicts, requested handoff
evidence, review emphasis, and clarification of already approved requirements.
These refinements must cite the triggering handoff/review and be logged before the
new kickoff is issued.

S00 may draft from a worker handoff and may immediately delay/return work because
of a blocking review finding. It may not treat a delivered technical contract as
an accepted dependency until review approves it. Before creating any new Sxx or
Sxx-R task, S00 presents a launch packet containing the relevant upstream handoffs,
reviews, SHAs/diffs/artifact status, unresolved conflicts, and the complete filled
kickoff envelope/prompt including reasoning and compute scope. The owner reviews
that packet and explicitly authorizes launch; only then may S00 create the isolated
task directly through Codex. There is no automatic downstream or reviewer launch.

The following are **material or locked changes** and remain owner decisions:

- Protocol A/B roles, threat model, claims, or FL system assumptions;
- `D_base`/`D_tail`/client/evaluation ownership or split construction;
- primary architecture, head, decoder, metric, precision, or initialization;
- experiment cells, seeds, gates, run priority, resource budget, or stop criteria;
- Slurm/full-run/rerun permission, uploads, commits/merges/pushes, or publication.

S00 cannot weaken a gate after seeing results, erase or relabel a failed/negative
cell, change a completed session retroactively, or silently redirect an active
session. Any active-session amendment is written into the ledger, sent to the
worker, acknowledged in `HANDOFF.md`, and re-reviewed if it changes delivered
semantics.

### Worktree provisioning contract

Worktrees are provisioned by selecting `Worktree` and the starting branch in the
Codex task-creation UI, not by a worker prompt. A normal Codex-managed worktree is
detached at the selected branch's HEAD; that is the expected default. Before a task
is opened, S00 produces a kickoff envelope containing:

This policy follows the [official Codex worktree
contract](https://learn.chatgpt.com/docs/environments/git-worktrees): managed
worktrees are task-scoped, start detached at the selected branch HEAD, and a named
branch cannot be checked out in multiple worktrees at once.

```text
SESSION_ID:
BASE_SHA:
SOURCE_BRANCH: v3-ad-perception
EXPECTED_REF_MODE: detached@BASE_SHA, or exact owner-created branch
WORKTREE_PROVISIONED_BY: S00 through Codex after owner launch approval, or owner UI
FILE_OWNERSHIP:
UPSTREAM_HANDOFFS_AND_SHAS:
WORKER_SHA: pending for Sxx, or exact worker commit for Sxx-R
DELIVERY_REF: pending owner authorization for Sxx, or exact review source ref
REASONING_EFFORT: xhigh, or ultra with the recorded task-specific reason
APPROVED_COMPUTE: none | standing short-smoke policy O-009 | exact approved request
```

The session verifies its repository root, HEAD, branch/detached state, and status
before editing. A detached HEAD is valid when it matches `BASE_SHA`. If the
UI-created worktree uses the wrong base, lacks the canonical documents, or conflicts
with another checked-out branch, it stops and reports the exact mismatch; it does
not repair the worktree itself.

Parallel module workers start from one pinned integration `BASE_SHA`. When a worker
finishes, S00 first checks the handoff; the owner then explicitly authorizes any
local handoff commit and `Create branch here` action needed to make that exact
version durable. That narrow permission does not authorize merge or push. Review
tasks use a distinct UI-created worktree at the resulting `WORKER_SHA`; a unique
review branch is used only when the owner authorizes preserving `REVIEW.md` as a
commit. S07 alone receives a dedicated integration worktree based on the approved
integration SHA and integrates only accepted worker commits. Execution sessions
start from the exact frozen candidate commit named in their approved request.

A commit made in an Sxx worktree advances only that worktree's detached HEAD or its
owner-authorized scoped branch. It does not modify S00's working tree, move
`v3-ad-perception`, or integrate itself. Git objects are shared by the repository,
but the worker version becomes a review baseline only after it has a durable
`WORKER_SHA` and branch/ref. S00 keeps canonical-document changes in its own
worktree; S07 later integrates only independently accepted worker commits.

The canonical Orchestra documents must be committed/landed on the source branch
before S00 or worker worktrees are spawned. Although Codex can apply selected local
changes when it creates a managed worktree, such state has no immutable SHA and is
not a valid base for this reproducible multi-session wave.

S00 is long-lived and should use a permanent worktree or keep its managed task
pinned. Workers/reviewers normally use one managed worktree per task and remain
pinned until their handoff/review is durably recorded and accepted. Because Codex
may clean up older managed worktrees, do not archive a task or rely on worktree
retention before its accepted artifacts are landed.

### Durable session delivery

Every worker session creates
`fl_v3/usenix27_orchestra/handoffs/Sxx/` with:

- **`HANDOFF.md` (mandatory):** exact scope, worktree/branch/base/commit, files and
  semantic changes, references/equations, tests/jobs and raw outputs, gate-by-gate
  evidence, artifacts/hashes, negative results, scientific claims allowed and
  forbidden, unresolved risks, and requested Orchestra decisions.
- **`RUN_REQUEST.md` (mandatory before material compute):** for an O-009 smoke,
  record exact HEAD plus working-tree diff hash, bounded data scope, resources,
  command/output, stop criteria, and the standing-authorization citation. For full
  tests/scientific work, record the immutable commit, resolved config hash,
  data/split manifest, cells, seeds, GPU/count/time budget, command, output path,
  stop criteria, and exact owner-approval status. A changed approved scope
  invalidates full-test/scientific approval.
- **`RESULTS.md` (mandatory for execution sessions):** job IDs, exit status, raw
  artifact paths/checksums, metric and performance tables, missing/failed cells,
  and interpretation limits. Never replace or hide a negative/failed cell.
- **`REVIEW.md` (written by an independent review session):** findings first,
  exact code/data/metric references, attempted adversarial checks, gate verdict,
  residual risk, and whether integration/scientific use is approved.

Large checkpoints, logs, caches, and raw results remain under `/nobackup`; Git stores
manifests, hashes, commands, and compact reports. The Orchestra verifies the handoff
and review, and only then updates the canonical status ledger. Owner monitoring
during a worker session does not replace independent code/science review.

The independent reviewer must explicitly audit data leakage/split ownership,
coordinate/calibration/units, batch invariance, config resolution, effective
optimizer steps, precision, modality execution, metric/denominator semantics,
resume/provenance, failed cells, and any shortcut that could inflate performance or
ASR. “Tests pass” alone is not a scientific PASS.

### Worker handoff format

Every worker records this information in `HANDOFF.md` and also returns it in the
session chat:

```text
Session ID / status: PASS | CHANGES-REQUESTED | BLOCKED
Branch/worktree and commit (if authorized):
Files changed:
Commands/tests/jobs and exact outputs:
Gate checklist:
Artifacts/checkpoint/config hashes:
Scientific interpretation allowed:
Scientific interpretation NOT allowed:
Remaining risks / decisions for Orchestra:
```

## 8. Existing-file inventory and consolidation policy

The worktree was clean before this Orchestra consolidation. The immediately
preceding Arrhenius work is already committed in eight commits from `4a14e4a` to
`54a9ac3`; it touched 72 files because it included runtime implementation,
profiling harnesses, launchers, tests, configs, and long-form evidence.

The main written records are:

| Existing file | What remains valuable | New status |
|---|---|---|
| [`phase1_capability_summary.md`](../collab/model_capability/phase1_capability_summary.md) | historical capability arc and `bb02d` metrics | read-only historical evidence; architecture-gap conclusion superseded |
| [`arrhenius_bevfusion_gap_audit.md`](../collab/model_capability/arrhenius_bevfusion_gap_audit.md) | sparse precision, memory/OOM, Stop-E evidence | read-only engineering evidence |
| [`arrhenius_camera_branch_audit.md`](../collab/model_capability/arrhenius_camera_branch_audit.md) | camera mini topology/gradient/profile evidence | read-only engineering evidence |
| [`arrhenius_speedup_log.md`](../docs/arrhenius_speedup_log.md) | profiling and precision history | performance evidence |
| [`arrhenius_migration.md`](../collab/arrhenius_migration.md) and [`env.md`](../docs/env.md) | validated GH200 environment and jobs | migration is read-only history; env is active runtime documentation |
| `scripts/arrhenius_*`, `run_arrhenius_*` | executable diagnostics and Slurm harnesses | retained implementation; reuse selectively |
| `tests/test_arrhenius_*`, `test_sparse_voxel_encoder.py` | current engineering regression tests | retained tests; extend, do not summarize away |

No historical evidence or executable file is deleted. The consolidation is logical:

- **this file** is the canonical objective, decision, schedule, and gate record;
- **`SESSIONS.md`** is the canonical work breakdown and per-session contract;
- **`KICKOFFS.md`** is the canonical copy-ready worker/reviewer startup registry;
- all of `fl_v3/collab/` is linked evidence only and should not receive new
  planning, handoff, review, result, or status prose.

## 9. Current data and environment facts

- Arrhenius GH200 and the persistent fp32/fp16-spconv environment are validated.
- The user is a member of the nuScenes license group, and the shared dataset module
  is now available. The module exposes `NUSCENES_DATA_DIR`; the shared directory
  contains trainval metadata plus ten stored `trainvalXX_blobs.zip` archives and
  the test archive.
- The ten trainval archives were readable in the access check and contain the
  camera samples, `LIDAR_TOP` keyframes, and sweeps required by the current model.
- A fresh login may be needed for normal group membership; `sg
  arrhpc-dataset-nuscenes` was the temporary verification path.
- Reviewed S01 worker `abe5c58` implements the read-only ZIP backend. Job `332651`
  indexed all ten archives and resolved `538,695/538,695` declared train/val
  references with zero missing; job `333206` ran 56 focused GH200 tests with zero
  failures/errors/skips against remediation implementation `54a48f9`. The full
  `t1.v1` caches from job `332651` are historical evidence only and are forbidden
  production inputs. S07-A owns current-format caller migration and full trainval
  `t1.v2` cache materialization/freeze. The read-only
  `collab/arrhenius_migration.md` remains historical evidence.

## 10. Owner decision gates

Owner decisions are stage-gated, not one blanket prerequisite for Wave A. An
unresolved item blocks only the session/action whose latest freeze point has
arrived. S00 should bring each item back to the owner with the relevant handoff,
review, cost, or engineering evidence rather than requiring speculative decisions
now.

### Locked now

- [x] USENIX '27 sprint scope and overlapping calendar (owner-approved).
- [x] Protocol B is the primary security setting; Protocol A is the clean
      optimization/control setting (owner-approved 2026-07-10).
- [x] Per-session `RUN_REQUEST.md` audit process; bounded non-scientific smoke may
      self-submit under O-009, while full tests/runs/metrics/matrices require exact
      owner approval (owner-approved 2026-07-10).
- [x] Fifteen-session split, dependencies, file ownership, and independent review
      process in `SESSIONS.md` (owner-approved).
- [x] Copy-ready S00, S01-S15, and S01-R-S15-R kickoff registry in `KICKOFFS.md`.
- [x] Evidence-driven Orchestra refinement authority with owner-locked scientific
      decisions and a recorded change-control ledger (owner-approved 2026-07-10).
- [x] Codex-UI-provisioned worktrees with pinned SHA/ref kickoff envelopes;
      sessions verify but do not manage worktree topology (owner-approved
      2026-07-10).

### Deferred decisions and latest freeze points

Engineering diagnostics may inform these decisions. However, metric definitions,
success thresholds, cells, and exclusion rules must be frozen before the scientific
runs they judge; they cannot be selected after seeing those outcomes.

| Decision | Current status | Latest owner freeze point |
|---|---|---|
| Minimum modular backbone/interface contract | pending | before S03/S04/S05 makes an irreversible primary implementation choice; final integrated contract before S07 |
| Multi-task CenterHead primary; TransFusion contingency/generalization | pending | primary head choice before S05 implementation; opening TransFusion can wait for reviewed S05/S07 evidence |
| `D_base`/`D_tail`, regional/fleet client unit, and fine-tuning scope | pending | S12 may design alternatives now; freeze and hash before split materialization or any Protocol-B training in S13 |
| mAP/NDS, fusion-gain, per-class, speed, memory, and acceptance gates | pending | numerical floors may use S07 engineering/profile evidence, but freeze before the S08/S09/S10 scientific run each gate evaluates |
| Five-cell single-seed matrix and whether 11 full runs are necessary | pending | use S07 cost plus reviewed branch evidence; approve each exact wave before its `RUN_REQUEST.md`, and freeze fusion cells before S10 outcomes |
| Security thesis and `CL-PILOT`/`CL-FREEZE` semantics | pending | S12 may refine the thesis; freeze `CL-PILOT` before it unlocks S13 and freeze final-backbone semantics before final security runs |

## 11. Orchestra change-control ledger

Only S00 edits this ledger. Each downstream plan/kickoff refinement records the
evidence that caused it; material entries remain `PENDING` until the owner approves.

| ID | Date | Evidence/decision | Change and affected sessions | Class | Approval/status |
|---|---|---|---|---|---|
| O-001 | 2026-07-10 | owner decision | Protocol B primary; Protocol A clean control; all sessions | locked scientific | approved |
| O-002 | 2026-07-10 | owner decision | 15 workers, independent reviews, durable handoffs, exact run authorization | locked process | approved |
| O-003 | 2026-07-10 | owner decision | S00 may refine unstarted work from accepted handoffs/reviews within the operational boundary | operational authority | approved |
| O-004 | 2026-07-10 | Codex managed-worktree contract and owner decision | task UI creates detached managed worktrees; kickoff pins SHA/ref; S00 permanent/pinned; local handoff commit needs exact permission | workspace policy | approved |
| O-005 | 2026-07-10 | owner decision on staged review | architecture, capability gates, matrix, split details, and thesis are decided at explicit latest freeze points rather than before all Wave-A work | scientific process | approved |
| O-006 | 2026-07-10 | owner approval in active S00 task | pin `/home/gaohui/.codex/worktrees/bb67/fl_weather_project` as the sole S00 canonical writer; archive the idle `/home/gaohui/.codex/worktrees/90d4/fl_weather_project` task without deleting its worktree or branch | workspace coordination | approved and executed |
| O-007 | 2026-07-10 | owner approval in active S00 task | issue S01 ZIP-backend implementation and S12 evidence/proposal-only kickoffs from `f262f6bea037580065a8505008773c04fdd259f5`; exact ownership is recorded in `SESSIONS.md`; both had `APPROVED_COMPUTE: none` at issuance | operational launch | approved and active; S01 later amended by O-009 |
| O-008 | 2026-07-10 | owner reasoning-budget decision | S00 must explicitly create future Sxx/Sxx-R tasks at `xhigh`; use `ultra` only for a recorded unusually complex implementation/review or broad difficult research task | resource policy | approved; applies to future tasks; S01 follow-up amended to xhigh |
| O-009 | 2026-07-10 | owner compute-policy decision and direct S01 smoke approval | allow bounded non-scientific Slurm smoke without per-job waiting, subject to the standing limits and preflight record above; retain exact owner review for full tests, full-data/profile/metrics, matrices, seeds, and reruns; active S01 must acknowledge amendment in its durable package | compute policy | approved and acknowledged in `RUN_REQUEST.md`; job `330409` running |
| O-010 | 2026-07-10 | owner clarification request | replace ambiguous worker-kickoff `WORKER_SHA: n/a` with `pending`; record that worker SHA/ref is produced only after S00 completeness checking and owner-authorized delivery commit, and is then consumed by Sxx-R | kickoff schema clarification | approved operational clarification |
| O-011 | 2026-07-11 | owner workflow clarification | before S00 directly creates any Sxx/Sxx-R, present upstream handoff/review/diff status and the complete filled kickoff for owner review and explicit launch authorization; worker sessions never launch their own reviewer; S07 remains the sole code-integration session | launch/integration workflow | approved |
| O-012 | 2026-07-11 | S01 worker `abe5c58`, review `7cf7fcc`, jobs `332651`/`333206`, and S00 raw-artifact audit | accept S01 as a reviewed dependency for S07 only: full manifest/checksums and scheduler records match; 56/56 focused tests and all listed remediation-source hashes match. Do not merge the review branch as implementation; do not use historical `t1.v1` caches or claim model/scientific readiness | integration evidence | accepted dependency; no merge/push |
| O-013 | 2026-07-11 | accepted S01/S01-R evidence and `build_gt_database.py` audit | split S07 into phase S07-A data-foundation integration and later S07-B full-stack integration. S07-A lands exact worker history plus review artifact, fixes the `t1.v1` caller and active-doc status, hardens future test attestation, and prepares a separately approved full `t1.v2` cache gate; S06 must bind `n_sweeps` and cache/manifest hashes explicitly | operational dependency refinement | requirements approved; task launch and Git operations pending owner authorization |
