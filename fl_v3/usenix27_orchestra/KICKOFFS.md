# USENIX Security '27 Orchestra — kickoff prompts

> **Status (2026-07-13):** O-092 cleanup registry. Legacy S07-B/T5/T6/T7
> prompts are retired and must not be recovered from Git history as active
> authority. S07-C is the next worker only after canonical P is committed and
> its exact SHA is inserted into the launch envelope. S12 is deferred; S13/S14
> remain blocked by their clean prerequisites.
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
5. A prompt may reference `standing short-smoke policy O-009` only for a bounded,
   non-scientific engineering smoke after its exact preflight is recorded. Under
   O-092 cleanup work, the session still stops for owner/S00 audit before
   submission; merely writing RUN_REQUEST.md does not authorize execution.
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
14. Owner decision O-017 freezes the Wave-A S02-S05 implementation choices and
    authorizes their parallel launch at `xhigh` from one canonical ledger commit
    directly atop S07-A freeze `0249eb21a32730ac1689255491b19a158711401f`.
    O-017 also authorizes scoped worker branches/commits, S00 review scheduling,
    and S00 approval of exact O-009-compliant short smokes only. For S02-S05 this
    is stricter than general O-009 self-submission: the worker must send the filled
    `RUN_REQUEST.md` to S00 and wait for explicit S00 approval before `sbatch`. It does not
    authorize full trainval, 100/1000-step gates, profiles/metrics/matrices, push,
    or merge to `v3-ad-perception`.
15. O-092 supersedes every active route from legacy O-032-O-091. No kickoff may
    import, copy, recover or cherry-pick legacy T5/T6/T7, old defense-registry,
    frozen e231 or old cycle_04/collab implementation decisions. Historical
    evidence may be cited only with its recorded negative/limited interpretation.

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
O-017-approved exact official CenterPoint/BEVFusion Gaussian-radius/target
semantics with `min_overlap=0.1` and `min_radius=2`. Add numerical golden tests,
sample/batch isolation, batch permutation, over-cap, empty-input, and target-render
tests. Do not redesign sparse SECOND, camera, head, trainer, or scientific recipe.

Before editing, identify the exact upstream reference/equation, state its three
roots and integer/min-radius behavior, and derive independently checked golden
values; any deviation from the O-017 reference returns to S00/owner. Own
`models/fusion/lidar_encoder.py`,
`models/fusion/losses.py`, focused tests, and handoffs/S02/. During the parallel
wave `losses.py` is exclusive to S02; do not edit S03-S05 modules.
Write HANDOFF.md with migration impact on old checkpoints. Any GPU/Slurm check needs
a RUN_REQUEST.md sent to S00; even an O-009-eligible bounded non-scientific smoke
must wait for explicit S00 approval before submission.
Return exact equations, fixtures, tests, residual risks, and forbidden
interpretations. O-017 authorizes a scoped `codex/s02-*` branch plus implementation,
test, RUN_REQUEST/RESULTS when applicable, and HANDOFF commits only within the
envelope ownership. Do not merge or push.
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

Objective: implement the O-017-approved independent multi-scale camera contract:
Swin-T, no dead FPN levels, valid stride-8 flow, 0.5 m depth bins, aspect-preserving
geometry/augmentation, calibration correctness, complete gradient coverage, and no
LiDAR-conditioned input in the primary branch.

Own camera-specific modules/tests and handoffs/S03/. Do not wire detector.py or
tasks.py; return the exact integration shape/dtype/config contract to S07. Before
editing, report intended files and geometry assumptions. Write HANDOFF.md. Any
bounded smoke requires a recorded RUN_REQUEST.md plus explicit S00 approval under
O-017/O-009; profiles and larger
tests require exact owner approval. Return projection residuals, gradient/
invariance/tiny-overfit evidence, memory implications, negative results, and
unresolved design choices. O-017 authorizes a scoped `codex/s03-*` branch plus
implementation, test, RUN_REQUEST/RESULTS when applicable, and HANDOFF commits only
within the envelope ownership. Do not merge or push.
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

Objective: implement the O-017-approved SECOND-style sparse LiDAR module at
`0.075x0.075x0.2 m` that downsamples XY by approximately 8x in sparse space before
low-resolution densification, exposes the matching BEV contract, supports fp16 AMP
plus fp32 reference behavior and train/eval voxel caps, and never creates a
1440x1440 dense/fusion tensor.

Own LiDAR sparse modules/tests and handoffs/S04/. Do not wire detector/tasks/trainer.
Before editing, report coordinate order, spatial shapes, stride/receptive-field
contract, intended files, and reference mapping. Write HANDOFF.md. Any GH200/Slurm
test needs RUN_REQUEST.md and explicit S00 approval; S00 may approve only a bounded
non-scientific smoke within O-017/O-009. Return shape/metric-coordinate fixtures,
empty/over-cap/batch tests, dtype/
gradient/memory evidence, negative findings, and integration requirements. O-017
authorizes a scoped `codex/s04-*` branch plus implementation, test,
RUN_REQUEST/RESULTS when applicable, and HANDOFF commits only within the envelope
ownership. Do not merge or push.
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

Objective: implement the O-017-approved framework-independent, reference-faithful
multi-task CenterHead primary and deterministic class/task-aware candidate selection
plus rotate/circle NMS. Preserve canonical dimensions/yaw/velocity and official
nuScenes conversion. Apply O-018 exactly: retain official per-class K=500, remove
only the second task-wide K, pass at most 500/1000 candidates for one/two-class tasks
to the pinned official task-wide NMS, and break ties by score descending, class ID
ascending, then flattened spatial index ascending. Retain official score/range,
task groups, circle/rotate choice/scales, pre=1000, post=83, and IoU threshold. Use
GroupNorm instead of official BN while retaining shared-conv and independent
two-layer heatmap/reg/height/dim/rot/vel branches. Label this
`reference-faithful no-starvation adaptation`; do not claim exact official decode
parity for multi-class tasks. The official task-flatten class order differs from
the project's devkit-global `DETECTION_NAMES` order: map every task-local label to
the global ID explicitly by class name, never by cumulative task offset.
TransFusion remains closed contingency.

Own head/decode-NMS modules, new head-specific loss adapters, and focused tests,
but do not wire the production detector; S07 integrates. Existing
`models/fusion/losses.py` is read-only while S02 is active; any required shared
interface edit returns to S00 after S02 review. Before editing, declare task groups,
thresholds, top-K/NMS semantics, reference equations, and files. Write
handoffs/S05/HANDOFF.md. Any material compute needs RUN_REQUEST.md plus explicit S00
approval; S00 may approve only a bounded non-scientific smoke within O-017/O-009.
Return reference fixtures, permutation stability,
tail-class candidate behavior, equal-score deterministic ordering, single-class
official parity, B=1/B>1 GroupNorm behavior, duplicate-box checks, eval round-trip,
explicit `construction_vehicle`/`bus`/`barrier`/`pedestrian`/`traffic_cone` label-map
fixtures, negative results, and risks. O-017/O-018 authorize a scoped `codex/s05-*` branch plus
implementation, test, RUN_REQUEST/RESULTS when applicable, and HANDOFF commits only
within the envelope ownership. Do not merge or push.
```

### S06 — production modes, config, and runtime

```text
You are worker S06: Production modes, resolved config, training/runtime/evaluation.

Read completely before editing:
- repository AGENTS.md;
- fl_v3/usenix27_orchestra/{ORCHESTRA.md,SESSIONS.md,KICKOFFS.md}, especially S06;
- fl_v3/docs/{env.md,roadmap/INDEX.md};
- accepted S01/S07-A data-foundation history and artifacts: S01 worker
  abe5c58b174dbbe1f7045ce91c8b15168d97b87b, S01 review
  7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc, S07-A delivery
  ba1571632557c20adbda3172221694cdbecfeabe, executable
  44cefd06bc815e893919d95c754896711dba3402, review
  370ea6c0bd4d9d737a5a50b6aff1c6f742589825, and integrated freeze
  0249eb21a32730ac1689255491b19a158711401f;
- reviewed S02-S05 HANDOFF/RESULTS/REVIEW packages and actual diffs at S02
  3aebf2dc1d19473f29260df279421047d216d70e / df142dc9a391b87d05bd7becaba59459e9659f88,
  S03 50893839c45cd3e2ef1b72b98db6668df7030f2a / 2f62e570c9c24ef1e18a483888c3f28ad56a415e,
  S04 483e149b95ec891b675df825d924a96bb225b7dd / a0763c2e0b322d4ca53a92f9f69c90d9b231bbff,
  and S05 a9c801fdee378906e54d06314d0c772b6559901a / 1c440843bb2b6d72f10310ff11fcde0d7d1e885c;
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

Binding runtime contract:
- `model-mode` is exactly `camera_only | lidar_only | fusion`; unknown or legacy
  aliases fail. A disabled modality is not constructed, checkpoint-loaded,
  transferred, dataset-decoded, or executed. Evaluation/submission metadata records
  the actual mode.
- Resolve one canonical, locale/order-stable config before constructing data/model/
  optimizer. Its hash binds architecture enums, model mode, precision, optimizer and
  executed-step budget, effective global batch, accumulation, seed, `n_sweeps`,
  canonical/physical `t1.v2` cache identities, ZIP-manifest identity, and dependency
  contract. Train, resume, eval, and later FL entry points must consume the same
  resolved hash; unknown keys/enums and environment-only scientific defaults fail.
- Full trainval `t1.v2` materialization is still absent and not authorized here.
  Implement and test the required identity fields with synthetic/mini fixtures;
  reject `t1.v1`, missing depth/hash/manifest, depth autodiscovery, and identity drift.
- Gradient accumulation and every schedule are defined by successfully executed
  optimizer updates, not batches. On nonfinite loss or GradScaler overflow, do not
  advance optimizer-step, scheduler, EMA, or exposure counters; clear/retain
  gradients only under one documented fail-closed rule. Checkpoint only at an
  accumulation/update boundary unless pending gradients and phase are serialized.
- Resume restores model, optimizer, scheduler, GradScaler, EMA, epoch, executed
  optimizer-step/exposure counters, RNG states, resolved-config hash, and data/
  manifest identities. Legacy/partial/mismatched checkpoints fail; no silent
  `strict=False` migration. Continuous versus interrupted/resumed tiny runs must
  match at the declared boundary.
- Keep one persistent loader/sampler across epochs where supported; call
  deterministic `set_epoch`, and prove no duplicate/omitted samples or ZIP handle
  lifecycle drift across epoch/resume boundaries.
- Evaluation uses the resolved precision/autocast policy, runs the dataset once,
  records modality/config/checkpoint/data identities, and keeps optional timing
  instrumentation output-neutral.
- For lidar/fusion, pin the reviewed S04 runtime to exact `spconv==2.3.8` and either
  serialize forward/mode changes for each encoder instance or fail closed. Do not
  claim concurrent/reentrant safety without instance-level protection and adversarial
  tests.
- Consume S02-S05 only as reviewed interface contracts. Do not cherry-pick, copy,
  reimplement, or finally wire their module code; S07-B is the sole cross-session
  integration owner. Return every unresolved interface seam explicitly.

File ownership is limited to production runtime seams:
- fl_v3/src/fl_v3/models/fusion/detector.py;
- fl_v3/src/fl_v3/training/{tasks.py,loop.py} and new S06-specific checkpoint/
  sampler/runtime helpers in that package;
- fl_v3/scripts/centralized_train.py and fl_v3/scripts/run_s06_*.sh;
- fl_v3/src/fl_v3/eval/{detection_eval.py,provenance.py};
- fl_v3/src/fl_v3/utils/runtime.py and new fl_v3/src/fl_v3/config/**;
- new fl_v3/configs/s06_*.json;
- fl_v3/tests/test_s06_*.py plus existing focused tests only when named in the
  filled kickoff envelope;
- fl_v3/usenix27_orchestra/handoffs/S06/**.

All S01-S05 source modules, data/cache builders, box conversion/decode/NMS, legacy
configs, canonical Orchestra files, fl_v3/collab/, and fl_v2/ are read-only. Report
the intended file list and interface/config schema before editing. Write HANDOFF.md;
write RUN_REQUEST.md/RESULTS.md when compute is proposed/executed. At kickoff,
compute is none: do not self-submit. S00 may approve only an exact bounded
non-scientific synthetic/mini validation after auditing its immutable request;
full trainval/cache/profile/100/1000-step/metrics/matrix/seed/rerun remains exact
owner scope.

Return mode-specific construction/I/O/forward proof, config-hash rejection cases,
step/exposure and nonfinite/overflow accounting, loader epoch behavior,
continuous/resume evidence, checkpoint/provenance rejection, eval autocast/metadata,
negative results, and S07-B integration requirements. O-027 authorizes scoped
implementation/test/handoff commits only after the owner approves the filled
envelope. Do not merge or push.
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

### S07-C — legacy-security cleanup

```text
SESSION_ID: S07-C
BASE_SHA: fill with the exact canonical-only P commit before launch
SOURCE_BRANCH: codex/s07-c-legacy-security-cleanup
EXPECTED_REF_MODE: detached@BASE_SHA
WORKTREE_PROVISIONED_BY: owner / Codex task UI after explicit launch approval
FILE_OWNERSHIP:
- REMOVE:
  fl_v3/src/fl_v3/attacks/**
  fl_v3/src/fl_v3/eval/{asr,frustum_visibility,report}.py
  fl_v3/src/fl_v3/viz/{attack,detection}.py
  fl_v3/configs/{t4_a100_detgate,t4_mini_smoke,t4_reference,t5_attack,t5_mini_smoke}.json
  fl_v3/scripts/{_t4_fd_diagnose,t4_readiness_eval,t5_attack_eval,t5_mini_smoke}.py
  fl_v3/scripts/run_s07_b_static_checks.sh
  fl_v3/tests/_attack_fixtures.py
  fl_v3/tests/test_attack_*.py
  fl_v3/tests/test_eval_{asr,frustum,report}.py
  fl_v3/tests/test_viz_detection.py
  fl_v3/src/fl_v3/strategy/defenses/**
  fl_v3/src/fl_v3/strategy/gradient_metrics.py
  fl_v3/tests/test_defense_*.py
  fl_v3/tests/test_gradient_metrics_parity.py
  fl_v3/tests/test_multikrum.py
  fl_v3/tests/fixtures/{make_oracle_fixtures.py,oracle_inputs.npz,oracle_arrays.npz,oracle_decisions.json}
- REFACTOR-KEEP:
  fl_v3/src/fl_v3/strategy/{aggregation_core,flower_strategies,__init__,server_opt}.py
  fl_v3/src/fl_v3/{server_app.py,training/tasks.py}
  fl_v3/src/fl_v3/engine/local_runner.py
  fl_v3/src/fl_v3/eval/{__init__,provenance,box_to_global,detection_eval}.py
  fl_v3/src/fl_v3/viz/fusion.py
  fl_v3/pyproject.toml
  fl_v3/requirements.txt
  fl_v3/requirements.lock.txt
  fl_v3/scripts/{build_arrhenius_env.sh,arrhenius_smoke.py,arrhenius_profile_mini.py}
  fl_v3/docs/env.md
  fl_v3/README.md
  fl_v3/docs/roadmap/INDEX.md
  fl_v3/src/fl_v3/data/nuscenes/conventions.md
  fl_v3/configs/{fl_1client_sanity,fl_bb02d_fedadam,p1_bb02,p1_bb02d,p1_bb02d_voxel,p1_bb02h,p1_bb04,p1_cbgs,p1_exp3,p1_gtpaste,p1_msweep,p1_msweep_aug,p1_msweep_aug2,p1_unfrozen,t3_fl_gate,t3_trainval}.json
  fl_v3/tests/{conftest,test_eval_provenance,test_fl_config_keys_registered,test_flower_fp32_parity,test_flower_strategies_construct,test_fl_round_smoke,test_s07_b_integration}.py
  fl_v3/usenix27_orchestra/handoffs/S07-C/{HANDOFF,RUN_REQUEST,RESULTS}.md
- READ-ONLY:
  repository AGENTS.md
  fl_v3/usenix27_orchestra/{ORCHESTRA,SESSIONS,KICKOFFS}.md
  fl_v3/collab/**
  fl_v3/docs/cycle_04/**
  all protected clean-foundation paths not listed above
UPSTREAM_HANDOFFS_AND_SHAS:
- audited code base 4ce2366df2925161adae8fea393d5fca64836d40
- accepted S01/S07-A and S02-S06 worker/review identities in ORCHESTRA 11.3
- frozen old evidence only: e231808e77388d69053dcbced6e754dbe3468aef
- read-only spawn reference only: bf480ea77ccf9ae8417c3ea58e933701dbc7222a
WORKER_SHA: pending
DELIVERY_REF: pending owner authorization
REASONING_EFFORT: xhigh
APPROVED_COMPUTE: none
DECISION_SCOPE: cleanup and clean-foundation preservation only
```

Read completely before acting: repository AGENTS.md; all three canonical
Orchestra documents; env.md; accepted S01/S07-A and S02-S06 handoffs/reviews;
frozen S07 HANDOFF/RESULTS/REVIEW packages; and actual Git diffs/raw artifacts.
Do not trust summaries when the source/diff/artifact is available.

Objective: remove every active legacy attack/readiness/defense route while
preserving clean C/L/F model construction, official clean DetectionEval, S01
ZIP/data contracts, S06 runtime/checkpoint behavior and one fixed clean FedAvg
path outside the legacy defense namespace.

The fixed clean FedAvg contract has no defense registry, no defense-type selector,
no gradient-space telemetry, no clipping/clustering/reweight/noise and no malicious
count. Preserve deterministic client identity/order, num-example weighting, server
optimizer, EMA, checkpoint and trainable-state semantics.

Do not import or cherry-pick bf480ea. Do not recover T5/T6/T7 from e231, old reviews,
collab, cycle_04 or Git history. Do not implement S12, a new attack or a new defense.
Do not run Slurm/GH200, full cache/trainval, model campaigns, metrics, profile, DDP,
matrix or retry. Local static/focused checks only.

Acceptance requires: no active legacy import/config/launcher route; clean FedAvg
parity on fixed FP32 weighted inputs; C/L/F construction tests; official clean eval
tests; focused S01/S06 regression checks; source compile/JSON/bash syntax/diff
checks; exact path inventory; and explicit unverified-runtime limits. Write the
durable S07-C package and stop for S00 completeness audit. Do not commit, merge or
push without exact authorization.

#### S07-C remediation amendment O-092-A1

The first S07-C delivery was not accepted and has no `WORKER_SHA`. The owner
approved returning the same detached worker with `APPROVED_COMPUTE=none` and the
following additional ownership:

```text
REMOVE:
- fl_v3/scripts/{p3_grad_conflict,p3_crt_probe,t3_trainval_reeval_fullval}.py

REFACTOR-KEEP:
- fl_v3/src/fl_v3/__init__.py
- fl_v3/src/fl_v3/viz/{writer,calibration}.py
- fl_v3/tests/test_viz_writer.py
- fl_v3/docs/determinism.md
- fl_v3/scripts/p3_partition_health.py
- fl_v3/scripts/{fl_gate_a40,t3_iid_vs_central}.py
- fl_v3/tests/{test_fl_sampling,test_model_task,test_fl_local_runner_multiround,test_fl_server_opt_integration,test_determinism_smoke,test_task_agnostic}.py
- fl_v3/src/fl_v3/models/fusion/{__init__,fusion,bev_grid}.py
- fl_v3/src/fl_v3/data/nuscenes/{gt_database,partition}.py
- fl_v3/tests/{test_model_bev_convention,test_model_params,test_fl_trainable_only}.py
```

Existing S07-C ownership remains in force. Restore `scikit-learn==1.8.0` in
`pyproject.toml`, `requirements.txt`, `requirements.lock.txt`, `docs/env.md`,
`build_arrhenius_env.sh`, and `arrhenius_smoke.py` as a pinned nuScenes runtime
dependency; do not restore HDBSCAN or any defense implementation. Remove
`NormTrackingFedAvg` and the `defense` parameter from local clean runners, update
all named callers, and retain exactly one `CleanFedAvgStrategy`. VizWriter keeps
only calibration/encoder/fusion/detection. `p3_partition_health.py` must default
to a configurable output outside `fl_v3/collab/**`.

The inclusive tombstone scan covers active source, scripts, configs, tests,
README and active docs; it may exclude only canonical Orchestra records,
`collab/**`, `docs/cycle_04/**`, and explicitly labelled frozen roadmap history.
It must not hide compatibility aliases, fail-closed dead scripts, visualization
stages, or clean callers. Re-run compile/JSON/TOML/bash/diff checks and every
available focused local test; dependency/GH200 tests remain `NOT RUN` unless a
new exact request is separately approved. Update HANDOFF/RESULTS, acknowledge
O-092-A1, and stop for a new S00 completeness audit without commit/merge/push.

#### S07-C closed-session harness amendment O-092-A2

S00 audited the completed O-092-A1 diff and durable package. The named A1
blockers are closed. The owner approved continuing in the same detached worker,
on the same uncommitted diff, with `APPROVED_COMPUTE=none`.

```text
REMOVE scripts:
- fl_v3/scripts/_bench_msweep.py
- fl_v3/scripts/agg_overcommit_diag.py
- fl_v3/scripts/arrhenius_lidar_gap_utils.py
- fl_v3/scripts/arrhenius_mini_matrix.py
- fl_v3/scripts/arrhenius_profile_mini.py
- fl_v3/scripts/det_gate_a40.py
- fl_v3/scripts/fl_gate_a40.py
- fl_v3/scripts/p1_amp_smoke.py
- fl_v3/scripts/p3_partition_health.py
- fl_v3/scripts/run_arrhenius_mini_matrix.sh
- fl_v3/scripts/run_arrhenius_profile_mini.sh
- fl_v3/scripts/run_arrhenius_stop_e_gate.sh
- fl_v3/scripts/run_v1_calibration.py
- fl_v3/scripts/runconfig.py
- fl_v3/scripts/t3_iid_vs_central.py
- fl_v3/scripts/verify_levers.py

REMOVE tests:
- fl_v3/tests/test_arrhenius_camera_audit_controls.py
- fl_v3/tests/test_arrhenius_lidar_gap_controls.py
- fl_v3/tests/test_fl_gate_refuses_non_a40.py

REFACTOR-KEEP:
- fl_v3/tests/test_s07_b_integration.py
  Remove the arrhenius_mini_matrix import/test seam only; preserve remaining
  clean integrated C/L/F/runtime/eval/FedAvg tests.
- fl_v3/tests/test_model_determinism.py
  Remove the A40 det_gate script authority wording only.
- fl_v3/src/fl_v3/models/fusion/losses.py
  Remove the verify_levers historical proof wording only; do not change loss code.
- fl_v3/usenix27_orchestra/handoffs/S07-C/{HANDOFF,RESULTS}.md
  ACK O-092-A2 and update exact inventories, scans, checks and limits.
```

The following 18 active foundation scripts are protected and must remain:

```text
fl_v3/scripts/arrhenius_env.sh
fl_v3/scripts/build_arrhenius_env.sh
fl_v3/scripts/run_arrhenius_env_build.sh
fl_v3/scripts/arrhenius_smoke.py
fl_v3/scripts/run_arrhenius_smoke.sh
fl_v3/scripts/centralized_train.py
fl_v3/scripts/build_nuscenes_cache.py
fl_v3/scripts/build_gt_database.py
fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh
fl_v3/scripts/run_s01_nuscenes_zip_smoke.sh
fl_v3/scripts/run_s01_nuscenes_zip_tests.sh
fl_v3/scripts/s01_nuscenes_zip_audit.py
fl_v3/scripts/s01_nuscenes_zip_benchmark.py
fl_v3/scripts/s01_nuscenes_zip_manifest.py
fl_v3/scripts/s01_nuscenes_zip_smoke.py
fl_v3/scripts/run_s06_runtime_tests.sh
fl_v3/scripts/run_s07a_nuscenes_cache_t1v2.sh
fl_v3/scripts/run_s07a_provenance_tests.sh
```

Do not recover the removed scripts elsewhere, rename them into a consolidated
harness, or modify protected-script semantics beyond already-authorized A1 edits.
Remove all active imports/test seams/default routes for the deleted names. The
inclusive tombstone scan must cover active scripts and shared tests. Re-run every
available local compile/JSON/TOML/bash/diff check and focused dependency-free
contract; dependency/GH200 checks remain `NOT RUN`. Stop for S00 completeness
audit without commit/ref, reviewer launch, Slurm/GH200, merge or push.

S00 subsequently completed the cumulative A2 completeness audit and the owner
authorized local durable sealing. S07-C-R still must not launch from mutable
worktree state: its filled packet must pin the resulting implementation SHA,
handoff-seal SHA and committed canonical parent exactly.

The sealed identities are canonical parent
`f7c696345b24b0e1227b1a52f3b47fb14e9120f5`, implementation
`a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`, and handoff seal
`f736f41371666725a11d51bc3b01c6ececb59d50`. The original detached snapshot
`9f06875e1b865734950abcf3b6de36ad06a0ac7b` has the same implementation
patch-id but is provenance evidence only.

### S07-C-R — independent cleanup review

```text
SESSION_ID: S07-C-R
BASE_SHA: 6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2
SOURCE_BRANCH: codex/s07-c-legacy-security-cleanup
EXPECTED_REF_MODE: detached@6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2
WORKTREE_PROVISIONED_BY: owner / Codex task UI after explicit review launch approval
FILE_OWNERSHIP:
- fl_v3/usenix27_orchestra/handoffs/S07-C/REVIEW.md
UPSTREAM_HANDOFFS_AND_SHAS:
- canonical preparation 4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
- canonical A1/A2 parent f7c696345b24b0e1227b1a52f3b47fb14e9120f5
- audited code base 4ce2366df2925161adae8fea393d5fca64836d40
- original detached snapshot evidence 9f06875e1b865734950abcf3b6de36ad06a0ac7b
- worker implementation a16c2cdfd4e23ba08677a66c45c50dd78340cc3b
- handoff seal f736f41371666725a11d51bc3b01c6ececb59d50
- frozen old evidence only e231808e77388d69053dcbced6e754dbe3468aef
- read-only spawn reference only bf480ea77ccf9ae8417c3ea58e933701dbc7222a
WORKER_SHA: a16c2cdfd4e23ba08677a66c45c50dd78340cc3b
HANDOFF_SEAL_SHA: f736f41371666725a11d51bc3b01c6ececb59d50
REVIEW_SHA: b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5
DELIVERY_REF: codex/s07-c-r-legacy-security-cleanup-review
REVIEW_SHA256: 588cfd0f91a2f70cbdcc6bf94a2279fc3cca693c9cd14f9d9909f02df769d8f5
REASONING_EFFORT: xhigh
APPROVED_COMPUTE: none
DECISION_SCOPE: independent code/science/process review only
```

Read repository `AGENTS.md`, all three canonical Orchestra documents,
`docs/env.md`, the full S07-C HANDOFF/RESULTS/RUN_REQUEST package, accepted
S01/S07-A and S02-S06 packages, and actual source/diffs rather than summaries.
Audit `f7c6963..a16c2cd` as the implementation diff and
`a16c2cd..f736f41` as the handoff-only diff; compare protected foundations to
`4ce2366`. Verify every REMOVE/REFACTOR-KEEP/KEEP/HISTORICAL-READ-ONLY claim,
the exact retained 18-script set, removal of old imports/default routes, and that
no deleted clean-foundation test was discarded by mistake.

Adversarially review clean FedAvg num-example FP32 weighting, deterministic client
identity/partition ordering and sampling, server optimizer, EMA, checkpoint and
trainable-only state; the direct scikit-learn pin and devkit `--no-deps` contract;
clean C/L/F construction; official DetectionEval; S01 ZIP/data; S06
runtime/checkpoint; centralized training; and residual config/doc routes. Confirm
that removal of seams from `test_s07_b_integration.py` did not remove other clean
coverage. Dependency-backed and GH200 tests are NOT RUN and must not be upgraded
to PASS. No GH200 run is needed or authorized for this review.

Review actual source and artifacts. Do not fix code, run Slurm, edit any path
except REVIEW.md, commit, merge, push or publish. Return severity-ordered findings
with exact paths/lines, adversarial checks, a PASS or CHANGES-REQUESTED gate
verdict, and explicit residual risk.

S07-C-R completed at review-only commit
`b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5`: PASS at static
code/source/config/test/docs scope, no P0-P3 finding. Dependency-backed and GH200
runtime remain NOT RUN. The review commit is separate evidence and must never be
merged into S07-C or S07-B-COMPLETE.

### S07-B-COMPLETE — closed clean integration completion

**Current state: CLOSED / OWNER ACCEPTED under O-093.** The block below preserves
the exact pre-launch envelope as historical provenance. At preparation time it was
not launch-authorized: canonical instructions had to be present in the worker's
immutable tree, and the task `BASE_SHA` therefore had to be the later docs-only S00
launch-packet seal containing this envelope. The subsequently approved launch,
bounded jobs, independent review and closure are recorded after the envelope.

```text
SESSION_ID: S07-B-COMPLETE
ACCEPTED_CLEANUP_SHA: 70bcd856f7ebb411eb2887e7ab71ef41ed13271f
BASE_SHA: exact docs-only S00 S07-B-COMPLETE launch-packet seal containing this envelope; fill in task prompt
SOURCE_BRANCH: codex/s07-b-clean-completion
EXPECTED_REF_MODE: detached@BASE_SHA
WORKTREE_PROVISIONED_BY: owner / Codex task UI after explicit launch approval
FILE_OWNERSHIP:
- ADD:
  fl_v3/tests/test_s07_b_clean_completion.py
  fl_v3/usenix27_orchestra/handoffs/S07-B-COMPLETE/HANDOFF.md
  fl_v3/usenix27_orchestra/handoffs/S07-B-COMPLETE/RUN_REQUEST.md
  fl_v3/usenix27_orchestra/handoffs/S07-B-COMPLETE/RESULTS.md
- REFACTOR-KEEP (edit only for a demonstrated clean integration failure, except the required flwr profile cleanup):
  fl_v3/configs/flwr_config.toml
  fl_v3/src/fl_v3/config/{__init__,resolved}.py
  fl_v3/src/fl_v3/data/partition.py
  fl_v3/src/fl_v3/data/nuscenes/{dataset,zip_backend,paths,info_cache,partition}.py
  fl_v3/src/fl_v3/models/fusion/{__init__,collate,detector}.py
  fl_v3/src/fl_v3/training/{tasks,loop,runtime_state,checkpoint}.py
  fl_v3/src/fl_v3/{client_app,server_app}.py
  fl_v3/src/fl_v3/engine/local_runner.py
  fl_v3/src/fl_v3/strategy/{__init__,aggregation_core,flower_strategies,sampling,server_opt}.py
  fl_v3/src/fl_v3/eval/{__init__,box_to_global,detection_eval,provenance}.py
  fl_v3/src/fl_v3/utils/runtime.py
  fl_v3/configs/{s06_synthetic_camera,s07_b_c_str8,s07_b_l_p020,s07_b_l_s075,s07_b_f_u,s07_b_f_cbgs}.json
  fl_v3/tests/conftest.py
  fl_v3/tests/test_s07_b_{integration,data_lifecycle}.py
  fl_v3/tests/test_s06_{resolved_config,model_modes,training_runtime,checkpoint_resume,loader_eval}.py
  fl_v3/tests/test_{eval_box_to_global,eval_detection_eval,eval_provenance}.py
  fl_v3/tests/test_{flower_fp32_parity,flower_strategies_construct,fl_sampling,fl_round_smoke}.py
  fl_v3/tests/test_{fl_local_runner_multiround,fl_server_opt_integration,fl_trainable_only}.py
  fl_v3/tests/test_{nuscenes_zip_backend,nuscenes_zip_dataset,nuscenes_info_cache,nuscenes_partition}.py
  fl_v3/tests/test_model_task.py
- READ-ONLY / RETURN FOR OWNER AMENDMENT:
  fl_v3/src/fl_v3/models/fusion/** except the three integration wrappers above
  fl_v3/src/fl_v3/data/nuscenes/** except the five lifecycle files above
  fl_v3/scripts/** (do not add, rename, consolidate or revive a harness)
  fl_v3/pyproject.toml
  fl_v3/requirements.txt
  fl_v3/requirements.lock.txt
  fl_v3/docs/env.md
  fl_v3/configs/{fl_1client_sanity,fl_bb02d_fedadam,p1_*,t3_*}.json
  fl_v3/usenix27_orchestra/{ORCHESTRA,SESSIONS,KICKOFFS}.md
  fl_v3/collab/**
  fl_v3/docs/cycle_04/**
  fl_v2/**
REMOVE: none without a new owner-approved amendment
UPSTREAM_HANDOFFS_AND_SHAS:
- accepted cleanup/canonical anchor 70bcd856f7ebb411eb2887e7ab71ef41ed13271f
- S07-C implementation a16c2cdfd4e23ba08677a66c45c50dd78340cc3b
- S07-C handoff seal f736f41371666725a11d51bc3b01c6ececb59d50
- S07-C review b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5 (separate, never merged)
- frozen negative evidence only e231808e77388d69053dcbced6e754dbe3468aef
- forbidden implementation source bf480ea77ccf9ae8417c3ea58e933701dbc7222a
- accepted S01/S07-A and S02-S06 clean foundations
WORKER_SHA: pending
EXECUTABLE_SHA: pending local materialization authorization
DELIVERY_REF: pending owner authorization
REASONING_EFFORT: xhigh
APPROVED_COMPUTE: none
DECISION_SCOPE: simplified clean C/L/F/runtime/eval/FedAvg completion only
```

Read completely before acting: repository `AGENTS.md`; all three canonical
Orchestra documents; `docs/env.md`; the accepted S01/S07-A and S02-S06 packages;
S07-C HANDOFF/RESULTS/RUN_REQUEST; S07-C review via exact separate review SHA;
and actual Git source/diffs. Inspect the frozen e231/bf480 evidence only when a
specific negative claim needs verification; never import it.

Preflight must report toplevel, exact HEAD, empty current branch and clean status.
A mismatch is a blocker. Do not create/switch branches or manage worktrees. The
owner/S00 provisions the ref. Use Chinese with owner/S00.

Objective: finish the smallest clean engineering integration needed to validate
centralized C/L/F construction, S06 runtime/checkpoint/resume, official clean
evaluation adapters, S01 mini directory/ZIP lifecycle and one clean Flower/FedAvg
path. This session does not train a capable model, define Protocol A/B data
ownership, or establish a scientific baseline.

Owner amendment O-092-A3 narrows the remaining runtime delta after the two-file
completion implementation. Accepted reviewed S01 and S02--S06 evidence is cited,
not re-run. Remove the expanded audit wrapper and do not replace it with another
harness. The sole candidate runtime is clean profile plus one C/L/F fp16 update
with workers 0, then one separately timed workers-0-versus-2 first-batch equality
check using node-local `/tmp`.

Verification-first rules:

1. Before changing source, reproduce the accepted static inventory and inspect
   current tests. Except for `flwr_config.toml`, source changes require a concrete
   failing clean contract recorded in HANDOFF/RESULTS; do not rewrite preserved
   modules speculatively.
2. Refactor `flwr_config.toml` to expose only a CPU local smoke profile and one
   single-GPU sequential clean profile. Remove active T3/Path-A/Path-B/4-GPU/
   overcommit and `collab/**` authority wording. This is configuration cleanup,
   not permission to run Flower/Ray or select a scientific execution policy.
3. The validation default is plain clean FedAvg: `server-optimizer=fedavg`, no
   server EMA, deterministic partition ordering and num-example FP32 averaging.
   Preserve tested FedOpt/EMA implementation capability but do not use it as the
   completion default or claim Protocol-A/B readiness.
4. Add at most the one named focused completion test file. Do not add an active
   script, launcher, profile, matrix, diagnostic harness or compatibility layer.
5. Do not weaken/delete accepted tests, relax fail-closed identity checks, invent
   full-cache hashes, fill template-only S07 configs with fake production values,
   or change locked architecture/coordinate/yaw/class/precision semantics.
6. A clean multiworker check uses the current standard PyTorch DataLoader path
   for one mini batch at workers 0 versus 2. Do not import `bf480ea`, introduce a
   global spawn policy, create a process matrix, or resurrect the old harness. If
   this exact check fails, preserve the failure and make only a minimal current-
   tree lifecycle fix within ownership.
7. No T5/T6/T7, attack, defense, ASR, S12 split, new protocol, scientific metric,
   capability threshold or publication work is allowed.

Local/login acceptance before any durable executable or compute request:

- exact tombstone/anti-recovery scan and path inventory;
- Python compile, all JSON and TOML parse, all retained shell `bash -n`, and
  `git diff --check`;
- AST/config checks proving one clean strategy/default, no legacy selector, and
  no active old Flower execution profile;
- exact requested-test inventory and explicit list of every dependency-backed
  item that is NOT RUN on the login node.

GH200 is **not approved at kickoff**. After the code/test diff is stable, write an
exact `RUN_REQUEST.md` and stop for S00/owner audit. A candidate request may contain
at most one sequential engineering job, one node/one GH200, one concurrent job,
at most 60 minutes, mini/synthetic data only, no retry. It must pin executable SHA,
read-only snapshot, environment, command, tests, mini root, output root and stop
conditions. Git, dependency-checkout state and source/archive manifests are login-
side closed evidence and must not execute inside the job. Any code/test/command
change invalidates approval.

The proposed bounded job may request only:

- environment/dependency imports from the already-built persistent environment;
- the focused clean Flower/FedAvg profile test;
- one B=1, `num_workers=0`, fp16 optimizer update for each exact mode C-STR8,
  L-S075 and F-U, with finite loss/gradients and the focused TrainingState
  boundary; this is three engineering mode checks, not an experiment matrix;
- after those updates, one separately timed mini first-batch `num_workers=0`
  versus `2` equality check with `TMPDIR=/tmp` and no skip/abort.

Explicitly forbidden even in the candidate request: full cache/trainval scans,
100/1000-step or tiny-overfit, model capability/mAP/NDS campaigns, profile,
throughput benchmark, Ray live federation, DDP, multi-GPU, actor/process matrix,
seed matrix, automatic retry, attack/defense or upload. `test_model_overfit.py`
and the old S07-B harness are never inputs.

The initial A3 acceptance requested training JUnit `4/0/0/0` plus loader JUnit
`1/0/0/0`; the later owner-approved F1 amendment replaced it with one exact
five-case FP32 JUnit while preserving C/L/F one-step evidence, worker equality,
zero exit and a final marker. Warnings remain recorded but non-fatal. S01/S06/
checkpoint/eval integration is inherited only from accepted reviewed evidence and
remained explicitly NOT RUN. A passing mini gate is engineering evidence only.
At kickoff time, commit, compute, reviewer creation, merge and push each required
later exact authorization; this sentence is retained as the historical launch
boundary, not as the current S07-B-COMPLETE state.

**Terminal launch record (2026-07-13).** The worker and exact simplified compute
were later authorized. Job `380806` passed environment identity and clean FedAvg,
then reached real-mini C/L/F forward, finite loss and backward. All three first
fp16 attempts had nonfinite unscaled gradient norms; assertions stopped before
step/skip metrics were emitted, and the loader phase was not run. This kickoff is consumed. It does not
authorize retry, scale changes, remediation compute, or review.

**D1 diagnostic amendment (2026-07-13).** The owner approved exactly one focused
gradient-classification submission after durable diagnostic commit `1900fe3`,
using only the immutable snapshot, nine cells, two script hashes, resources,
output root and stop rules pinned in `RUN_REQUEST.md`. No changed command, retry,
automatic remediation or review is authorized.

**D1 terminal record.** Job `389356` completed 9/9 diagnostic cells. FP32 C/L/F
and fp16-scale-1 C had finite gradients and optimizer calls; fp16-scale-512 C/L/F
and fp16-scale-1 L/F had direct nonfinite elements and scaler skips. The scale-1
L/F failures first appear in sparse SECOND stem/stage1 parameters. The approval
is consumed; this record authorizes no retry, remediation or review.

**FP32 final-gate amendment.** The owner removed precision comparison and scaler
remediation from S07-B-COMPLETE. Replace only the one-step acceptance test with
uniform FP32 and run exactly the clean FedAvg constructor, C/L/F one-successful-
update cases and worker-0/2 first-batch equality. Do not collect D1, add AMP/
scale cells, profile, run metrics, add steps or change production source/config.
The owner approved the exact immutable test commit `29ca663`, snapshot, five-case
command, two script hashes, resources and output tuple for one submission after
the docs-only seal. No changed command or retry is authorized.

**F1 terminal record.** Job `390576` completed `0:0` in `00:04:24` with zero
restarts and passed all five selected cases: plain FedAvg, one finite successful
FP32 update for each C/L/F mode, and worker-0/2 first-batch equality. The compute
approval was consumed. At that point the candidate could proceed only to
independent review; no retry, full training or scientific claim was authorized.
The review below later consumed that transition.

**Independent review and closure.** S07-B-COMPLETE-R independently reviewed
candidate `c615b647`, exact F1/D1 snapshots and raw artifacts, returned **PASS at
the exact bounded clean-engineering scope**, and found no P0/P1/P2/P3. Review
SHA-256 is `b0feed5476dbc810b24a5dc3c7a678bc90ac3a2520360f02fdb6a6bf54691ebd`;
terminal/review package is `7f3bd40158e5a8af30196509734782c4575c50aa`.
The owner accepted the verdict and formally closed S07-B-COMPLETE. This authorizes
no additional S07-B compute and does not freeze full-training precision or grant
full-data/scientific acceptance.

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

### S12 — deferred; not copy-ready

The old S12 proposal and snapshot are unreviewed historical evidence. Do not open
or aggregate S12 during S07-C. A later S12 re-audit requires a new filled kickoff
from the cleaned branch, clean CL status and explicit owner approval. Its scope is
clean Protocol-A/B data ownership and split design only; old attack/defense
assumptions are not inherited.

### S13 — blocked pending clean prerequisites; not copy-ready

No S13 task may be launched from this placeholder. Before a filled kickoff exists:

1. clean S07-B completion must be independently accepted;
2. the CL detector must be trained and frozen;
3. clean Protocol-B adaptation and the separately labelled Protocol-A control must
   be established under an accepted data/split contract;
4. the owner must approve a new threat model and exact attack envelope.

A later S13 may implement a new attack from the clean foundation. It must not
import, copy, recover or cherry-pick legacy T5/T6/T7, e231, retired O-032-O-091,
collab or cycle_04 implementation decisions. Clean and attacked compute requests
remain separate.

### S14 — blocked until viable undefended attack; not copy-ready

No S14 task may be launched until a new S13 undefended attack is independently
shown viable on the capable clean baseline and the owner approves a defense
envelope. Legacy defense implementations/oracles may be inspected only as frozen
negative/parity evidence; they are not active baselines or code sources.

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

### S07-R — retired alias

The old integrated-engineering reviewer prompt is retired with the e231 chain.
For cleanup, use the exact S07-C-R prompt above. A later clean S07-B completion
review receives a new filled exact-SHA prompt after that worker delivers; it must
not reuse this historical section.

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

### S12-R — deferred; not copy-ready

No S12-R launches until a fresh cleaned-branch S12 worker exists at an exact
durable SHA under a new owner-approved protocol/split envelope.

### S13-R — deferred; not copy-ready

No S13-R launches until clean adaptation and, later, a newly implemented attack
have exact worker SHAs and separate approved review scopes. Legacy T5 review
packages cannot be reused.

### S14-R — deferred; not copy-ready

No S14-R launches until a new S14 implementation follows an independently viable
undefended S13 attack. Legacy defense review/oracle evidence is historical only.

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
