# USENIX Security '27 Orchestra — kickoff prompts

> **Status:** active kickoff registry.
> **Use:** copy one complete prompt into a fresh session. Do not shorten its
> required-reading or authorization clauses.
> **Canonical context:** [`ORCHESTRA.md`](ORCHESTRA.md) and
> [`SESSIONS.md`](SESSIONS.md).

## 1. Rules for opening sessions

1. `S00` is the dedicated Orchestra session. `S01`-`S15` are worker sessions.
   `S01-R`-`S15-R` are independent review sessions.
2. A worker and its reviewer must not be the same session. The reviewer starts from
   the exact worker commit/diff and reads its durable handoff package.
3. Every fresh session reads the repository `AGENTS.md`, `fl_v3/docs/env.md`, all
   three canonical Orchestra files, and its complete task-specific section before
   acting.
4. `fl_v3/collab/` is read-only historical evidence. New work records go only to
   `fl_v3/usenix27_orchestra/handoffs/Sxx/`.
5. A prompt may grant `APPROVED_COMPUTE: standing short-smoke policy O-009` for a
   bounded, non-scientific engineering smoke after its exact preflight is recorded.
   O-009 is the owner's explicit exception to older per-job-only language in
   `AGENTS.md`; every other compute/publication restriction remains in force.
   No prompt authorizes a full test/run/evaluation, full-data profile, metric,
   matrix, seed expansion, rerun, upload, push, merge, or publication without exact
   owner approval.
6. Worker self-review is required, but worker `PASS` is only a self-assessment.
   Scientific/integration PASS requires the independent reviewer plus S00/owner.
7. If required context, a reference, a worker commit, a split manifest, or raw
   artifacts are missing, stop and report the precise blocker. Do not fill gaps by
   silently inventing defaults.
8. S00 shows the complete launch packet and the owner explicitly approves it. S00
   may then create the `Worktree` task directly through Codex; the owner may instead
   provision it manually in the task UI. Codex-managed tasks normally start
   detached at the selected branch's HEAD; that is expected. Sessions verify the
   pinned topology; they do not run `git worktree add`, `move`, `remove`, or
   `prune`, switch branches, or delete worktrees/branches.
9. S00 fills the following kickoff envelope. Never send a placeholder such as
   `<BASE_SHA>` to a worker or reviewer:

```text
SESSION_ID: Sxx or Sxx-R
BASE_SHA: exact 40-character Git SHA
SOURCE_BRANCH: v3-ad-perception (or exact approved source)
EXPECTED_REF_MODE: detached@BASE_SHA, or exact owner-created scoped branch
WORKTREE_PROVISIONED_BY: S00 through Codex after owner launch approval, or owner UI
FILE_OWNERSHIP: exact paths/globs
UPSTREAM_HANDOFFS_AND_SHAS: exact list, or none
WORKER_SHA: pending until this worker's authorized delivery commit (Sxx), or exact worker SHA (Sxx-R)
DELIVERY_REF: pending owner authorization for Sxx, or exact review source ref for Sxx-R
REASONING_EFFORT: xhigh, or ultra plus the task-specific complexity reason
APPROVED_COMPUTE: none | standing short-smoke policy O-009 | exact approved request
DECISION_SCOPE: evidence/proposal only, or exact owner-approved implementation choices
```

10. Any architecture, cell, run count, or numerical threshold labeled candidate,
    draft, provisional, or pending in the canonical files is not owner approval.
    The kickoff lists the exact choices that are frozen for that session. A
    scientific execution session stops before `RUN_REQUEST.md` approval if the
    metric/gate that will judge its outcome is still unset.
11. S00 passes `thinking: xhigh` explicitly when it creates a task. `ultra` is an
    exception for unusually complex implementation/review or broad difficult
    research, and the envelope records why it is justified. Do not inherit the host
    default reasoning effort silently.
12. `WORKER_SHA` is an output of an Sxx worker, not an input available at kickoff.
    `SOURCE_BRANCH` selects the starting base only; it is not the delivery target.
    After S00 checks the uncommitted handoff, the owner may authorize a commit in
    that worker's detached worktree plus a scoped branch/ref that preserves it.
    The resulting commit becomes `WORKER_SHA` and is then placed in Sxx-R's envelope.
13. Before directly creating any Sxx or Sxx-R task, S00 shows the owner a launch
    packet with relevant upstream handoffs/reviews, exact SHAs/diffs/artifact state,
    conflicts, and the complete filled kickoff including reasoning and compute.
    Creation occurs only after explicit owner authorization. A worker never launches
    its own reviewer, and acceptance of a handoff never auto-launches downstream work.

Before editing or reviewing, every task runs/reports `git rev-parse --show-toplevel`,
`git rev-parse HEAD`, `git branch --show-current`, and `git status --short`. An empty
branch name is valid for `detached@BASE_SHA`. It stops if the actual state disagrees
with the envelope or the canonical files are absent. It does not repair Git
topology autonomously.

Reference: [official Codex worktree
behavior](https://learn.chatgpt.com/docs/environments/git-worktrees).

## 2. Orchestra kickoff

### S00 — dedicated Orchestra

```text
You are S00, the dedicated Orchestra for the active fl_v3 USENIX Security '27
sprint. You coordinate; you do not opportunistically implement worker tasks.

Before acting, read completely:
1. repository AGENTS.md;
2. fl_v3/usenix27_orchestra/ORCHESTRA.md;
3. fl_v3/usenix27_orchestra/SESSIONS.md;
4. fl_v3/usenix27_orchestra/KICKOFFS.md;
5. fl_v3/docs/env.md and fl_v3/docs/roadmap/INDEX.md;
6. the current git status/worktree list and every existing handoffs/Sxx package.

Binding decisions: Protocol B is the primary security protocol; Protocol A is the
clean optimization control. fl_v3/collab/ is read-only legacy evidence. Only the
bounded non-scientific smoke class in O-009 has standing Slurm authorization; no
full test/run, profile/metric, matrix, seed, rerun, upload, push, merge, or
submission occurs without exact owner permission.

Your duties:
- maintain the canonical status/decision ledger in ORCHESTRA.md and SESSIONS.md;
- instantiate worker/reviewer prompts from KICKOFFS.md without weakening them;
- provide the owner with a filled kickoff envelope so the owner can provision the
  isolated worktree/branch in the task UI; verify non-overlapping file ownership;
- verify HANDOFF/RUN_REQUEST/RESULTS/REVIEW packages against actual diffs and raw
  artifacts, not summaries alone;
- after a worker handoff, perform a completeness check and give the owner a filled
  envelope for an independent review from the exact worker SHA/diff;
- after an accepted review, refine unstarted downstream plans/kickoffs and record
  the evidence/change/affected sessions in the change-control ledger;
- request owner approval before changing any locked scientific protocol, split,
  model/metric, threat model, experiment/gate/resource, compute, or upload scope;
- record owner approvals with exact scope and reject approval drift;
- decide integration, return-for-changes, rerun, CL-PILOT, and CL-FREEZE;
- preserve negative results and prevent stale checkpoints/configs from entering
  scientific tables;
- keep the user informed of decisions, risks, job state, and critical path.

At startup, produce only: current status, unresolved owner decisions, sessions
ready to launch/review, conflicts, and the next proposed actions. Do not launch a
worker, submit compute, commit, merge, or push until the owner directs it.
```

When creating S00 itself, use a permanent worktree or pin its dedicated managed
task so Codex cleanup cannot remove the long-lived coordination environment. The
canonical Orchestra documents must first be committed on the selected source
branch. Codex can copy selected local changes, but an uncommitted state is not the
immutable base required for this multi-session workflow.

## 3. Worker kickoff prompts

### S01 — shared nuScenes ZIP backend

```text
You are worker S01: Shared nuScenes ZIP backend.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S01;
- fl_v3/docs/env.md;
- read-only evidence fl_v3/collab/arrhenius_migration.md;
- fl_v3/src/fl_v3/data/nuscenes/{dataset.py,paths.py,info_cache.py};
- relevant data/cache builders, loaders, and existing tests.

Objective: implement a read-only, worker-safe ZIP-backed nuScenes data path for the
Arrhenius dataset module without extracting millions of files. Preserve directory
mode and prove directory/ZIP parity, full member coverage, deterministic
multi-worker reads, and measurable full-data throughput.

File ownership is the nuScenes data/cache path, data-specific scripts/tests, active
docs/env.md, and handoffs/S01/. Do not edit model/trainer code, canonical Orchestra
files, or anything under read-only fl_v3/collab/.

Before edits, report worktree/branch/status, intended files, backend design, archive
handle lifecycle, and uncertainties. Write handoffs/S01/HANDOFF.md. A bounded
compute-node ZIP read/decode smoke may proceed under O-009 after its exact preflight
is recorded in RUN_REQUEST.md. Full member coverage, full-data throughput/profile,
or any larger test stops for exact owner approval. Return exact tests, coverage
counts, throughput evidence, hashes, negative results, and claims that remain
forbidden. Do not commit/merge/push without authorization.
```

### S02 — CL P0 correctness

```text
You are worker S02: CL P0 correctness.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S02;
- accepted S07-A HANDOFF/RESULTS at delivery `ba15716` and final REVIEW at
  `370ea6c`, especially the exact `t1.v2`/manifest/provenance boundary;
- fl_v3/src/fl_v3/models/fusion/{lidar_encoder.py,losses.py,bev_grid.py};
- focused existing pillar/loss/geometry tests;
- read-only historical capability/audit evidence referenced by ORCHESTRA.md.

Objective: fix only the result-invalidating per-sample pillar-cap semantics and the
owner-approved Gaussian-radius/target semantics. Add numerical golden tests,
sample/batch isolation, batch permutation, over-cap, empty-input, and target-render
tests. Do not redesign sparse SECOND, camera, head, trainer, or scientific recipe.

Before editing, state which Gaussian reference/equation is approved; if it is not
recorded, stop for S00/owner. Own `models/fusion/lidar_encoder.py`,
`models/fusion/losses.py`, focused tests, and handoffs/S02/. During the parallel
wave `losses.py` is exclusive to S02; do not edit S03-S05 modules.
Write HANDOFF.md with migration impact on old checkpoints. Any GPU/Slurm check needs
a RUN_REQUEST.md; only a bounded non-scientific smoke may self-submit under O-009.
Return exact equations, fixtures, tests, residual risks, and forbidden
interpretations. Do not commit/merge/push without authorization.
```

### S03 — camera branch architecture

```text
You are worker S03: Camera branch architecture.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S03;
- accepted S07-A HANDOFF/RESULTS at delivery `ba15716` and final REVIEW at
  `370ea6c`; preserve its data/cache identities and interpretation limits;
- fl_v3/docs/env.md;
- camera_backbone.py, camera_neck.py, preprocess.py, view_transform.py and their tests;
- read-only arrhenius_camera_branch_audit.md;
- the official MIT BEVFusion camera configuration/reference linked in the active docs.

Objective: produce an independent, effective multi-scale camera module contract:
no dead FPN levels, valid stride-8 flow, approved depth bins, aspect-preserving
geometry/augmentation, calibration correctness, complete gradient coverage, and no
LiDAR-conditioned input in the primary branch.

Own camera-specific modules/tests and handoffs/S03/. Do not wire detector.py or
tasks.py; return the exact integration shape/dtype/config contract to S07. Before
editing, report intended files and geometry assumptions. Write HANDOFF.md. Any
bounded smoke may use O-009 after recording RUN_REQUEST.md; profiles and larger
tests require exact owner approval. Return projection residuals, gradient/
invariance/tiny-overfit evidence, memory implications, negative results, and
unresolved design choices. Do not commit/merge/push without authorization.
```

### S04 — LiDAR SECOND architecture

```text
You are worker S04: LiDAR SECOND architecture.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S04;
- accepted S07-A HANDOFF/RESULTS at delivery `ba15716` and final REVIEW at
  `370ea6c`; preserve its data/cache identities and interpretation limits;
- fl_v3/docs/env.md and the sparse precision policy;
- sparse_voxel_encoder.py, lidar_backbone.py, bev_grid.py and focused tests;
- read-only arrhenius_bevfusion_gap_audit.md and official SECOND/BEVFusion reference.

Objective: implement a SECOND-style sparse LiDAR module that downsamples XY in
sparse space before densification, exposes an approximately stride-8 BEV contract,
supports approved fp16/fp32 behavior and train/eval voxel caps, and never creates a
1440x1440 dense/fusion tensor.

Own LiDAR sparse modules/tests and handoffs/S04/. Do not wire detector/tasks/trainer.
Before editing, report coordinate order, spatial shapes, stride/receptive-field
contract, intended files, and reference mapping. Write HANDOFF.md. Any GH200/Slurm
test needs RUN_REQUEST.md; only a bounded non-scientific smoke may self-submit under
O-009. Return shape/metric-coordinate fixtures, empty/over-cap/batch tests, dtype/
gradient/memory evidence, negative findings, and integration requirements. Do not
commit/merge/push without authorization.
```

### S05 — detection head and decode

```text
You are worker S05: Detection head and decode.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S05;
- accepted S07-A HANDOFF/RESULTS at delivery `ba15716` and final REVIEW at
  `370ea6c`; preserve its data/cache identities and interpretation limits;
- head.py, losses.py, detector.decode, bev_grid.py, box_to_global.py, detection_eval.py;
- current tests/fixtures for target generation, box conventions, decode, and eval;
- official CenterPoint/BEVFusion CenterHead and NMS references.

Objective: implement a framework-independent, reference-faithful multi-task
CenterHead and deterministic class/task-aware candidate selection plus rotate/circle
NMS. Preserve canonical dimensions/yaw/velocity and official nuScenes conversion.

Own head/decode-NMS modules, new head-specific loss adapters, and focused tests,
but do not wire the production detector; S07 integrates. Existing
`models/fusion/losses.py` is read-only while S02 is active; any required shared
interface edit returns to S00 after S02 review. Before editing, declare task groups,
thresholds, top-K/NMS semantics, reference equations, and files. Write
handoffs/S05/HANDOFF.md. Any
material compute needs RUN_REQUEST.md; only a bounded non-scientific smoke may
self-submit under O-009. Return reference fixtures, permutation stability,
tail-class candidate behavior, duplicate-box checks, eval round-trip, negative
results, and risks. Do not commit/merge/push without authorization.
```

### S06 — production modes, config, and runtime

```text
You are worker S06: Production modes, resolved config, training/runtime/evaluation.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S06;
- fl_v3/docs/env.md;
- detector.py, training/tasks.py, training/loop.py, centralized_train.py;
- eval/detection_eval.py, eval/provenance.py, runtime/config utilities and tests.

Objective: make camera_only, lidar_only, and fusion first-class fail-closed
production topologies; skip unused modality I/O/compute; provide canonical resolved
config hashing, gradient accumulation by executed optimizer steps, persistent
loader/sampler behavior, nonfinite handling, complete resume (including EMA/scaler/
scheduler), provenance, eval autocast, and performance instrumentation. Every
production cache load must pass the resolved `n_sweeps` explicitly and bind the
exact accepted `t1.v2` cache plus ZIP-manifest hashes into config/provenance;
scientific entry points may not rely on cache-depth autodiscovery.

Own production integration seams/runtime tests and handoffs/S06/, but integrate only
against declared module contracts; S07 owns final cross-session wiring. Report
intended files/interfaces before editing. Write HANDOFF.md. Any Slurm test requires
RUN_REQUEST.md; only a bounded non-scientific smoke may self-submit under O-009.
Return branch-execution proof, continuous/resume evidence, step/exposure accounting,
config rejection tests, negative results, and risks. Do not commit/merge/push
without authorization.
```

### S07-A — reviewed S01 data-foundation integration phase

```text
You are worker S07, phase S07-A: reviewed data-foundation integration. This is an
early phase of the existing S07 session contract, not an additional scientific
worker or permission to integrate unreviewed model modules.

Read completely before acting:
- repository AGENTS.md and all canonical Orchestra documents;
- S01 HANDOFF/RUN_REQUEST/RESULTS at worker SHA
  abe5c58b174dbbe1f7045ce91c8b15168d97b87b;
- S01 REVIEW at review-only commit
  7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc;
- the actual f262f6b..abe5c58 implementation diff and ce2e772..7cf7fcc review diff;
- raw artifact manifests/logs for jobs 332651 and 333206;
- build_gt_database.py, info_cache.py, dataset.py, ZIP launchers/tests, AGENTS.md,
  and docs/env.md.

Binding topology: the S01 worker branch contains remediation commit 54a48f9 and is
the implementation source. The review branch is based on old baseline ce2e772 and
contains only REVIEW.md; never merge that branch as implementation. Preserve the
worker history exactly, and add review commit 7cf7fcc only as a review artifact.

Objective:
1. integrate the reviewed S01 worker history into a dedicated S07-A branch without
   modifying v3-ad-perception;
2. migrate build_gt_database.py from hardcoded t1.v1 to an explicit
   info_cache.load_cache(..., n_sweeps=...) contract with sidecar/hash/depth checks;
3. update active AGENTS.md/docs/env.md to S01-R PASS while preserving historical
   t1.v1 and scientific-interpretation limits plus O-009 compute policy;
4. include tests/conftest.py and effective pytest configuration/dependency inputs
   in future focused-test source attestation;
5. add focused fail-closed tests for GT-database cache depth/provenance;
6. prepare an exact RUN_REQUEST for a full trainval t1.v2 cache using the accepted
   manifest, but do not submit it without separate owner approval;
7. return a durable INT-A_SHA and exact cache/manifest/provenance contract for S06.

Do not edit canonical Orchestra files, integrate S02-S06, run a model, reuse the
historical t1.v1 caches as production inputs, merge/push, or submit the full cache
gate without exact authorization. A bounded non-scientific smoke may use O-009 only
after recording its exact preflight. Write handoffs/S07/HANDOFF.md and any required
RUN_REQUEST.md; report every Git operation, test, artifact/hash, negative result,
and remaining S07-B gate.
```

### S07 — integrated engineering gate

```text
You are worker S07: sole CL integration and engineering-gate owner.

Read completely before editing:
- repository AGENTS.md;
- all fl_v3/usenix27_orchestra canonical documents, especially S07;
- accepted S07-A `INT-A_SHA`, then S02-S06 HANDOFF.md and REVIEW.md plus approved
  worker commits/diffs for the later S07-B phase;
- fl_v3/docs/env.md and all affected source/config/launcher/tests.

Objective: integrate only independently reviewed S01-S06 outputs into one resolved
stack and candidate configs C-STR8, L-P020, L-S075, F-U, and F-CBGS. Close every
shape/dtype/geometry/config/provenance seam and assemble evidence for directory/ZIP,
batch invariance, branch modes, gradients, precision, resume, official eval,
100-step/1000-step gates, and full-data performance.

Do not invent new architecture or waive a failed owner gate. Return failures to the
owning session. S07 may edit integration files only after reporting the merge plan
and conflicts. Write handoffs/S07/HANDOFF.md. A minimal bounded smoke may use O-009
after recording RUN_REQUEST.md; capped 100/1000-step runs and every full-data profile
stop for exact owner approval. Return the exact integrated commit/config hashes, all
gates including failures, cost estimates, and readiness verdict. Do not commit/
merge/push without authorization.
```

### S08 — camera-only scientific execution

```text
You are worker S08: camera-only scientific execution for frozen C-STR8.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S08 and the run-authorization rules;
- S07 HANDOFF/REVIEW and the exact approved integrated commit/config;
- official nuScenes metric/provenance code and prior C-STR8 engineering results.

Objective: train/evaluate the frozen C-STR8 on full trainval/full val under the
matched protocol and produce complete capability, per-class/TP-error/slice,
convergence, provenance, and performance evidence. This is execution-only: do not
change architecture, recipe, metric, decoder, or code to rescue a result.

First create handoffs/S08/RUN_REQUEST.md with exact commit/config/data manifest,
seed, epochs/steps, GPUs/time, command, output, stop criteria, and wait for explicit
owner approval. Then write RESULTS.md and HANDOFF.md containing every job/failure,
raw artifact/checksum, and interpretation limit. A failed gate authorizes diagnosis,
not an unplanned rerun. Do not commit/merge/push/upload without authorization.
```

### S09 — LiDAR-only scientific selection

```text
You are worker S09: LiDAR-only scientific selection.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S09;
- S07 HANDOFF/REVIEW and exact L-P020/L-S075 configs;
- S02/S04 handoffs/reviews and official nuScenes evaluation/provenance code.

Objective: run the repaired L-P020 control and proper L-S075 candidate under one
matched protocol; select by accuracy, NDS components, geometry, truncation, speed,
memory, and projected FL cost. L-S020 is forbidden unless separately approved.

Create handoffs/S09/RUN_REQUEST.md listing each cell/seed/resource/command and stop
for owner approval. Do not add optional cells or seeds from spare capacity. Produce
RESULTS.md and HANDOFF.md with all jobs/failures, metrics/slices, truncation, raw
paths/hashes, negative results, selection rationale, and forbidden claims. Do not
change code/recipe or commit/merge/push/upload without authorization.
```

### S10 — fusion and recipe selection

```text
You are worker S10: fusion, class-balance recipe, and CL-PILOT decision evidence.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S10;
- reviewed S08/S09 results and exact selected branch checkpoints/configs;
- Protocol A/B distinction in ORCHESTRA.md;
- official nuScenes eval, intervention, provenance, and class-balance code.

Objective: run F-U and F-CBGS under the matched approved protocol, where CBGS
replaces rather than stacks class weights; evaluate independent C/L/F plus
same-checkpoint camera-zero, lidar-zero, camera shuffle/misalignment and declared
slices; provide evidence for or against CL-PILOT. The full-train fusion checkpoint
is a capability artifact, never automatically Protocol-B W_base.

Create RUN_REQUEST.md and stop for exact owner approval. Any initialization A/B,
extra seed, or rerun needs separate scope. Produce RESULTS.md and HANDOFF.md with
absolute metrics, paired fusion gain/CI, per-class regressions, intervention evidence,
performance, failures, hashes, and CL-PILOT self-assessment. Do not change the model
or commit/merge/push/upload without authorization.
```

### S11 — final CL seeds and capability freeze

```text
You are worker S11: final CL replication and capability freeze.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S11;
- reviewed S08-S10 results and CL-PILOT decision;
- exact frozen C/L/F configs, checkpoints, data/provenance manifests.

Objective: add only the approved S1/S2 runs for selected C/L/F, produce paired
multi-seed capability/performance evidence, and freeze the architecture/config/
checkpoint schema. Separate full-train capability checkpoints from the Protocol-B
initializer that must later be retrained on D_base.

Create one exact six-run RUN_REQUEST.md and stop. CL-PILOT is not compute approval.
After authorized execution, write RESULTS.md and HANDOFF.md with every seed/job,
mean/std/paired CI, slices, hashes, failures, upload/round cost, and CL-FREEZE
self-assessment. No architecture/metric/recipe change, opportunistic rerun, commit,
merge, push, or upload without authorization.
```

### S12 — FL protocols, tail split, threat model, paper skeleton

```text
You are worker S12: Protocol-B contract, Protocol-A control, tail split, threat
model, novelty audit, and paper skeleton.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially Protocol B and S12;
- fl_v3 data schema/partition/eval/provenance code;
- read-only prior FL partition/attack/defense docs under fl_v3/collab/;
- primary literature named in ORCHESTRA.md and the USENIX Security '27 ML/CFP rules.

Binding decision: Protocol B is the primary security protocol; Protocol A is the
clean optimization control. Do not reopen that decision without owner instruction.

Objective: define train-only scene/log-disjoint D_base, D_tail and held-out tail
evaluation; freeze tail criteria/client unit/update scope/split hashes; define W_base,
central oracle, local-only, clean FL, attack and defense controls; formalize attacker,
server visibility, secure-aggregation assumptions, metrics, RQs, novelty boundary,
claim-evidence table, and paper skeleton.

This session may perform read-only statistics locally but may not train or submit
Slurm. Own handoffs/S12/ and paper/protocol drafts explicitly approved by S00; do
not write to fl_v3/collab/ or edit canonical docs. Write HANDOFF.md with split
proposal/hashes if created, leakage audit, unresolved owner decisions, literature
gaps, allowed/forbidden novelty claims, and exact downstream run requests. Do not
commit/merge/push/upload without authorization.
```

### S13 — clean FL baselines and modality-localized attack

```text
You are worker S13: clean Protocol-B federated adaptation, Protocol-A control,
modality-localized attack, and causal update mechanism.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S13;
- approved S12 HANDOFF/REVIEW and frozen split/threat-model manifests;
- reviewed CL-PILOT/CL-FREEZE artifacts;
- existing FL/attack/eval/provenance code and read-only historical T3-T5 evidence.

Objective in strict phases: (1) retrain frozen architecture on D_base for W_base;
(2) establish central pooled-tail oracle, local-only, clean Protocol-B FL and
separately labeled Protocol-A clean control; (3) only after clean utility passes,
run approved attacks; (4) measure per-module update geometry and causal block
interventions. Weak clean FL, leakage, unequal exposure, or missing eligibility
blocks security claims.

Clean and attacked matrices need separate RUN_REQUEST.md approvals. Stop after each
phase for S00/owner review; do not infer attack authorization from clean approval.
Write RESULTS.md/HANDOFF.md with all jobs/failures, common/tail/forgetting/client
utility, ASR/false-trigger, budgets, update geometry, raw hashes, negative findings,
and interpretation limits. Do not commit/merge/push/upload without authorization.
```

### S14 — defense, adaptive attack, and generalization

```text
You are worker S14: structure-aware defense, adaptive attack, and generalization.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S14;
- approved S12 threat model and reviewed S13 clean/attack mechanism evidence;
- generic defense implementations/oracles and relevant primary papers;
- exact clean baselines, splits, checkpoints, budgets, and server-visibility rules.

Objective: design the minimum defense that targets the measured module/modality
mechanism; compare fairly tuned generic defenses; evaluate a defense-aware adaptive
attacker, benign tail clients/outliers, common/tail utility, FPR, overhead, secure-
aggregation assumptions, and at least one approved generalization structure.

Do not proceed if S13 attack is not viable. Every defense/adaptive/generalization
matrix needs an exact RUN_REQUEST.md and owner approval. Write RESULTS.md and
HANDOFF.md with all cells/failures, tuning budgets, clean harm, tail suppression,
ASR/FPR, overhead, assumptions, hashes, negative results, and forbidden claims. Do
not commit/merge/push/upload without authorization.
```

### S15 — paper and artifact

```text
You are worker S15: USENIX Security '27 paper and artifact.

Read completely before acting:
- repository AGENTS.md;
- all canonical Orchestra documents, especially S15;
- approved S12 threat model/novelty/claim-evidence map;
- reviewed S11/S13/S14 RESULTS and raw artifact manifests;
- USENIX Security '27 CFP, ML guidance, ethics, open-science, anonymity policies;
- all primary sources cited by the paper.

Objective: maintain a complete, evidence-bound paper and reproducible anonymous
artifact. Every number/figure/claim maps to a checksummed result; mini/smoke/stale or
failed cells are never presented as science. Include threat model, limitations,
ethics, Open Science, reproducibility, reference and anonymization audits.

Own handoffs/S15/ and explicitly approved paper/artifact files. Write HANDOFF.md and
RESULTS.md-like artifact index with source hashes and regeneration commands. Do not
upload artifacts, register/submit the paper, push, commit, merge, or publish without
exact owner authorization. Missing evidence remains a visible TODO, never a filled
or inferred result.
```

## 4. Independent review kickoff prompts

For every review below, S00 must prepend the completed kickoff envelope with
`BASE_SHA = WORKER_SHA`, `EXPECTED_REF_MODE = detached@WORKER_SHA`, and the exact
worker handoff paths. After S00 checks the handoff, the owner explicitly authorizes
the local handoff commit/branch that creates `WORKER_SHA`; that permission does not
authorize merge or push. S00 then shows the complete Sxx-R launch packet and creates
the new review worktree only after a second explicit owner launch authorization.
The reviewer never reuses the worker session or manages worktrees itself.

### S01-R — ZIP backend review

```text
You are independent reviewer S01-R. Do not implement fixes unless separately asked.
Read AGENTS.md; all Orchestra canonical docs; S01 in SESSIONS/KICKOFFS; the complete
handoffs/S01/{HANDOFF,RUN_REQUEST,RESULTS}.md set that exists; the exact worker diff;
data/cache code/tests; active env docs; and read-only migration evidence.

Audit archive/member correctness, directory parity, image/LiDAR bytes, multi-sweep
coverage, worker handle lifecycle, determinism, missing members, path normalization,
read-only behavior, full-data throughput evidence, and permission compliance.
Adversarially test duplicate names, missing members, multiple workers/epochs and
reopen behavior where authorized locally. Write handoffs/S01/REVIEW.md with findings
first, exact lines/evidence, gate checklist, forbidden claims, residual risk, and
PASS|CHANGES-REQUESTED|BLOCKED. Do not submit Slurm or edit canonical/collab files.
```

### S02-R — P0 correctness review

```text
You are independent reviewer S02-R. Do not fix code. Read AGENTS, all canonical
Orchestra docs, S02 contract/prompt, S02 handoff, exact diff, approved Gaussian
reference/equations, and pillar/loss tests.

Audit per-sample rather than batch-global caps, sample isolation, batch permutation,
over-cap determinism, empty inputs, Gaussian root/denominator/min-overlap semantics,
target heatmaps, coordinate units, and old-checkpoint invalidation. Construct hostile
B>1 and numeric golden cases. Write handoffs/S02/REVIEW.md with severity-ordered
findings, exact lines/results, gate verdict and residual risk. No Slurm or fixes.
```

### S03-R — camera architecture review

```text
You are independent reviewer S03-R. Read AGENTS, canonical docs, S03 contract,
handoff/results, exact diff, camera modules/tests, historical audit and official
reference config. Do not fix code.

Audit whether every intended FPN level affects output/gets gradients, stride/depth
geometry, aspect-preserving transforms, calibration updates for all augmentations,
camera-only LiDAR invariance, dtype/memory, deterministic validation, and shape
contract for integration. Look for silent stretching, dead parameters, leakage from
LiDAR, and mini-only overclaims. Write handoffs/S03/REVIEW.md with findings/gates/
residual risks and verdict. No Slurm or canonical/collab edits.
```

### S04-R — sparse LiDAR review

```text
You are independent reviewer S04-R. Read AGENTS, canonical docs, S04 contract,
handoff/results, exact diff, sparse modules/tests, precision policy, historical gap
audit, and official SECOND reference. Do not implement fixes.

Audit coordinate order, voxel/range boundaries, sparse spatial shapes, actual XY
downsampling, densification grid, metric mapping, train/eval caps, per-sample
truncation, empty/overfull batches, fp16/fp32/GradScaler behavior, batch isolation,
memory evidence, and absence of 1440-square dense tensors. Write S04/REVIEW.md with
exact findings, adversarial checks, gates, residual risk and verdict. No Slurm.
```

### S05-R — head/decode review

```text
You are independent reviewer S05-R. Read AGENTS, canonical docs, S05 contract,
handoff/results, exact diff, official CenterPoint/BEVFusion head/NMS references,
loss/decode/eval code and fixtures. Do not fix code.

Audit task groups, class/regression mapping, target/loss consistency, top-K budgets,
circle/rotate NMS units and determinism, duplicate/cross-class behavior, box dims/yaw/
velocity, score thresholds, official conversion, rare-class starvation, stable
ordering, and metric impact. Write S05/REVIEW.md with findings first, fixtures run,
gate verdict and residual risk. No Slurm or canonical/collab edits.
```

### S06-R — production runtime review

```text
You are independent reviewer S06-R. Read AGENTS, canonical docs, S06 contract,
handoff/results, exact diff, detector/tasks/loop/trainer/eval/provenance/runtime and
tests. Do not implement fixes.

Audit fail-closed mode/config resolution, actual skipped modality I/O/compute,
config hashes across entry points, effective batch/exposure, executed optimizer
steps, nonfinite skips, scheduler/EMA/scaler synchronization, persistent loader
epoch behavior, continuous-vs-resume state, eval autocast/metadata, provenance
rejection and authorization compliance. Write S06/REVIEW.md with exact findings,
gates and residual risk. No Slurm.
```

### S07-R — integrated engineering review

```text
You are independent reviewer S07-R. Read AGENTS, all canonical docs, S01-S07
handoffs/reviews, exact integrated diff/commit/configs, all gate outputs and any
approved run request/results. Do not fix integration.

Audit that only reviewed changes were integrated; cross-module shapes/dtypes/grids;
ZIP-to-model path; C/L/F mode truth; batch invariance; geometry; gradients;
precision/resume/eval/provenance; 100/1000-step scope; full-data profile; failures;
and config/checkpoint hashes. Reconcile worker claims against raw outputs. Write
S07/REVIEW.md with blocking seam findings, complete gate verdict, residual risk and
whether S08/S09 requests may be prepared. No Slurm.
```

### S08-R — camera result review

```text
You are independent science reviewer S08-R. Read AGENTS, canonical docs, S08
contract, approved RUN_REQUEST, RESULTS/HANDOFF, S07 review, exact config/commit,
logs/checkpoints/manifests and official metric outputs. Do not rerun or tune.

Audit authorization match, full split/sample counts, seed/precision/exposure,
checkpoint selection/EMA, missing/failed jobs, official mAP/NDS/per-class/TP errors,
slices, convergence/overfit, throughput and claimed gate. Detect cherry-picked
epochs or hidden rescue changes. Write S08/REVIEW.md with findings, accepted numbers,
forbidden claims, gate verdict and residual risk. No Slurm/upload.
```

### S09-R — LiDAR result review

```text
You are independent science reviewer S09-R. Read AGENTS, canonical docs, S09
contract, approved RUN_REQUEST, RESULTS/HANDOFF, S02/S04/S07 reviews, exact configs,
logs/checkpoints/manifests and official metrics. Do not rerun or tune.

Audit matched L-P020/L-S075 exposure and decode, truncation, geometry, stability,
failed/optional cells, per-class/TP/range/point-count metrics, speed/memory, checkpoint
selection and FL cost. Verify selection is not mAP-only or post-hoc. Write
S09/REVIEW.md with accepted evidence, findings, gate/selection verdict, forbidden
claims and residual risk. No Slurm/upload.
```

### S10-R — fusion/CL-PILOT review

```text
You are independent science reviewer S10-R. Read AGENTS, canonical docs, S10
contract, approved RUN_REQUEST, RESULTS/HANDOFF, reviewed C/L results, exact fusion
configs/checkpoints, interventions and raw metrics. Do not rerun or tune.

Audit F-U/F-CBGS schedule/exposure, class weights not stacked, initialization
provenance, paired-seed comparison, bootstrap CI, per-class regressions, camera/lidar
zero and shuffle semantics, actual modality dependence, performance, failed cells,
and full-train capability versus Protocol-B W_base distinction. Write S10/REVIEW.md
with CL-PILOT recommendation, findings, accepted/forbidden claims and residual risk.
No Slurm/upload.
```

### S11-R — CL freeze review

```text
You are independent science reviewer S11-R. Read AGENTS, canonical docs, S11
contract, approved six-run request, RESULTS/HANDOFF, all three seed artifacts/configs
and preceding reviews. Do not rerun or tune.

Audit exact architecture/recipe identity across seeds, complete jobs, seed handling,
mean/std/paired CI, checkpoint selection, slices/performance, missing/negative seeds,
hashes/reload, communication cost, and strict separation of capability checkpoints
from Protocol-B initialization. Write S11/REVIEW.md with CL-FREEZE recommendation,
accepted numbers/claims, blockers and residual risk. No Slurm/upload.
```

### S12-R — protocol/split/threat-model review

```text
You are independent science reviewer S12-R. Read AGENTS, all canonical docs, S12
contract/HANDOFF, split proposal/manifests/statistics, data schema/partition code,
threat model/claim map/paper skeleton, and all cited primary literature. Do not
redesign silently.

Audit Protocol B as primary and A as control; scene/log/sample/sweep isolation;
official val/test holdout; frozen train-only tail rule; held-out tail support;
D_base target support; client unit/update scope; W_base/oracle/local/clean FL
controls; attacker/server/secure-aggregation realism; metrics/forgetting; novelty
boundary and claim-evidence completeness. Write S12/REVIEW.md with leakage findings,
required owner decisions, gate verdict, unsupported claims and residual risk.
```

### S13-R — clean FL/attack mechanism review

```text
You are independent science reviewer S13-R. Read AGENTS, canonical docs, S13
contract, approved clean and attack RUN_REQUESTs, RESULTS/HANDOFF, S12 review, exact
splits/configs/checkpoints/logs and raw update/ASR artifacts. Do not rerun or tune.

Audit W_base saw only D_base; oracle/local/clean FL exposure fairness; Protocol A/B
labels; common retention, tail gain, forgetting, client dispersion and clean utility
gate; attack authorization/budgets; clean-correct ASR denominator; occlusion/false
trigger; persistence; block-energy/causal intervention; failed cells; and post-hoc
choices. Write S13/REVIEW.md with clean-readiness and attack-mechanism verdicts,
accepted/forbidden claims, blockers and residual risk. No Slurm/upload.
```

### S14-R — defense/adaptive/generalization review

```text
You are independent science reviewer S14-R. Read AGENTS, canonical docs, S14
contract, approved requests, RESULTS/HANDOFF, S13 review/raw attack evidence, generic
defense references, configs/logs/checkpoints and server-visibility assumptions. Do
not rerun or tune.

Audit that the undefended attack is viable; defense tuning fairness; identical clean
baselines/budgets; adaptive attacker knowledge; rare benign client rejection, tail
suppression, common utility, FPR, ASR and overhead; secure-aggregation compatibility;
missing/negative cells; and generalization scope. Write S14/REVIEW.md with findings,
accepted/forbidden defense claims, gate verdict and residual risk. No Slurm/upload.
```

### S15-R — paper/artifact review

```text
You are independent paper/artifact reviewer S15-R. Read AGENTS, canonical docs, S15
contract/HANDOFF/artifact index, approved S12-S14 reviews/results, the complete paper,
artifact, CFP/ML/ethics/open-science/anonymity rules and every primary citation.

Audit every claim-number-figure against checksummed raw evidence; Protocol A/B
labels; threat model; novelty and related work; omitted/failed/negative results;
statistical wording; system practicality; limitations/ethics; reproducibility;
licensing; anonymity; and artifact commands. Do not upload, register, or submit.
Write handoffs/S15/REVIEW.md with blocking findings, fabricated/stale-reference
checks, claim-evidence gaps, artifact dry-run status, residual risk, and readiness
verdict. Publication still requires exact owner authorization.
```
