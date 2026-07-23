# S10 Phase-I-R — MIT BEVFusion reference reproduction on Arrhenius

## 0. Status and authority

```text
PLAN_STATE: OWNER-APPROVED / P1R-G0 closes at the containing plan-freeze commit
DATE: 2026-07-23
OWNER_DECISION: O-151
BASE_EVIDENCE_SHA: 714f69eac2a0857dc8435cd9ee8bc202d1035456
ACTIVE_SCOPE: upstream BEVFusion environment/runtime/data qualification and
              official nuScenes LiDAR -> Fusion reference reproduction
OLD_LOCAL_PHASE_I: terminal historical evidence; no remaining execution authority
ENVELOPE_A: frozen request below; exact containing-commit activation still required
ENVELOPE_B: design only; cannot be frozen before measured Envelope-A evidence
DOWNLOAD/ENVIRONMENT/SLURM: not authorized by this plan freeze alone
FL/CLIENTS/ATTACK/DEFENSE: explicitly outside scope
MERGE/PUSH/UPLOAD/PUBLICATION: not authorized
```

O-151 approves the new document organization, the scientific declaration and
execution topology in this plan, the five-work-package/three-gate/two-envelope
workflow, fast-forwarding the control branch to the complete profiler/result base
above, and creating the containing plan-freeze commit. It does not silently turn a
future commit into external-action authority: source/checkpoint acquisition,
environment construction and Slurm begin only after the owner activates the exact
containing commit and Envelope A.

This plan supersedes `PHASE_I_PLAN.md` only for active execution. The old plan,
its implementation, profiler work and terminal Camera/LiDAR results remain
historical evidence and are not rewritten or erased.

## 1. Objective, exit gate and exclusions

The sole objective is to determine whether the published MIT BEVFusion detector can
be reproduced credibly on Arrhenius GH200 using the official OpenMMLab BEVFusion
port, official nuScenes train/val roles and a coherent upstream recipe.

Required outcomes are:

1. a clean, isolated, source-controlled GH200 runtime;
2. correct loading and official-val evaluation of the published OpenMMLab LiDAR
   and LiDAR+Camera checkpoints;
3. a fresh official-train LiDAR run and terminal official-val evaluation;
4. a staged fresh Fusion run initialized from the fresh LiDAR checkpoint and the
   upstream-declared image initializer, followed by terminal official-val
   evaluation;
5. an accepted reproduction or an honest, bounded negative result.

The phase does **not** design or execute:

- the current local Camera/LiDAR/Fusion implementation or any repair to it;
- standalone Camera training;
- `D_fit`, `D_select`, `D_audit`, `D_base` or `D_tail`;
- FL client meaning, client partitioning, aggregation, BN/FedBN or adaptation;
- attack, defense, ASR or publication experiments;
- promotion of a full-train reference checkpoint to a future Protocol-B `W_base`;
- migration of future FL/security code into the reference repository.

Those questions may be considered only after `P1R-G2` and a separate owner-approved
phase.

## 2. Frozen reference identity

### 2.1 Model and recipe anchors

- **Paper/recipe semantics:** MIT BEVFusion commit
  `db75150717a9462cb60241e36ba28d65f6908607`, the pre-BEVFusion-R reference
  anchor.
- **Executable GH200 implementation:** MMDetection3D BEVFusion `v1.4.0`,
  commit `fe25f7a51d36e3702f961e198894580d83c4387b`.
- **Initial MM stack:** MMCV `v2.1.0` (`57c4e25e...`), MMDetection `v3.2.0`
  (`fe3f809a...`) and MMEngine `v0.10.7` (`390ba2fb...`), all within the
  MMDetection3D `v1.4.0` declared ranges.
- **Sparse stack:** independently rebuilt cumm `v0.7.13` and spconv `v2.3.8`
  from the currently accepted Arrhenius source commits.
- **Data roles:** official nuScenes train for fitting and official nuScenes val
  for checkpoint-oracle and fresh-reproduction evaluation.
- **LiDAR exposure:** keyframe plus nine historical sweeps in training and
  evaluation. The later `sweeps_num: 0` configuration is not the paper-capability
  reference and is forbidden here.
- **Training order:** fresh LiDAR first, then Fusion from that fresh LiDAR
  checkpoint plus the exact upstream-declared image initializer.

WP0 must mechanically resolve the pinned upstream configs rather than transcribe
optimizer, scheduler, augmentation, epoch or hook defaults by hand. The resulting
resolved LiDAR and Fusion configs become part of one clean reference-repository
commit before Envelope B is requested.

### 2.2 Checkpoint/config/operator binding

Every published oracle checkpoint is inseparable from:

1. its exact OpenMMLab config;
2. the matching voxelization operation;
3. the matching BEV-pooling implementation;
4. the declared class and box conventions.

An original MIT checkpoint may not be silently loaded with a different MMCV
voxelizer. Checkpoint loading defaults to strict; any nonempty missing/unexpected
allowlist is an explicit finding and must be frozen before continued evaluation.

The published checkpoint targets are:

| Branch | mAP | NDS |
|---|---:|---:|
| LiDAR | 0.649 | 0.696 |
| Fusion | 0.686 | 0.714 |

Envelope-A oracle acceptance requires absolute difference no larger than `0.005`
for each reported mAP/NDS value and local/upstream evaluator agreement within
`1e-6`.

Fresh single-seed reproduction acceptance is provisionally:

- each LiDAR/Fusion terminal mAP and NDS no more than `0.010` absolute below or
  above the corresponding published result; and
- a positive fresh Fusion-minus-LiDAR contribution.

These are reproduction tolerances, not publication confidence intervals or a
future FL capability threshold.

## 3. Isolation, Git and evidence topology

### 3.1 Control repository

The current `fl_weather_project` repository remains the control plane for:

- this plan and owner authority;
- the compact S10 handoff;
- the append-only S10 run ledger;
- the new-session kickoff and final phase disposition.

Its main worktree and shared Git object database live under
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project`; Codex
may use a `/home/gaohui/.codex/worktrees/...` worktree backed by that same object
database. No push is implied by the presence of the configured Git remote.

### 3.2 Isolated execution root

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/bevfusion_ref/
├── repo/          # independent OpenMMLab-derived Git repository
├── env/           # isolated Python/PyTorch/MM runtime
├── checkpoints/
├── logs/
└── outputs/
```

The external `repo/` is a real Git repository from the first implementation
action. It starts from the pinned MMDetection3D source, uses a dedicated
`codex/arrhenius-bevfusion-ref` branch, and contains all executable compatibility
changes, configs, runner code and ZIP/data adapters. No job may execute from a
dirty external source tree.

The environment is intentionally simpler than a vendored binary archive:

- it is mutable during Envelope-A engineering remediation;
- before Envelope-B scientific training, one `environment.freeze.txt` and one
  `build_info.txt` are committed in the external reference repository;
- Envelope B may not mutate the frozen environment;
- an unavoidable environment mutation produces a new clean external commit and
  invalidates affected scientific outputs.

The actual conda prefix, checkpoints and raw logs are not committed to either Git
repository.

### 3.3 Compact run identity

The plan-freeze control SHA is recorded once per envelope. Each submitted job then
has one user-facing `RUN_ID` and one concise ledger row:

| Job | RUN_ID | Purpose | Output | State | charged GH200-hours |
|---|---|---|---|---|---:|

The immutable `run_manifest.json` behind `RUN_ID` contains the exact external
repository commit, resolved config, data role, seed, command, resources,
checkpoint identity and environment identity required by the S10 provenance
contract. These are automatic evidence, not separate approval objects.

### 3.4 Promotion after success

If `P1R-G2` accepts the reproduction, the external reference repository may be
promoted in a later owner-approved migration phase:

- its upstream and compatibility history is retained;
- it receives a dedicated remote before becoming long-lived;
- only explicitly selected ZIP/evaluator or later FL/security components are
  imported from `fl_v3`;
- the current repository remains frozen historical evidence;
- FL, client partitioning, attacks and defenses are designed separately.

No repository promotion, remote creation or push is authorized by Phase-I-R.

## 4. Five work packages

### WP0 — plan, reference and repository freeze

- close `P1R-G0` at the containing plan-freeze commit;
- verify every pinned upstream SHA, license and checkpoint URL;
- create the external reference Git repository and dedicated branch only after
  Envelope A activation;
- mechanically resolve the upstream LiDAR/Fusion configs;
- freeze the checkpoint/config/voxelizer/operator triples;
- implement one compact `RUN_ID` manifest format.

WP0 contains no checkpoint acquisition, environment build or Slurm action before
Envelope A activation.

### WP1 — isolated GH200 environment and compatibility

- construct `/nobackup/.../bevfusion_ref/env` without modifying the accepted
  `arrhenius_fl_v3` prefix;
- start from Python `3.11.15`, PyTorch `2.11.0+cu128`, torchvision
  `0.26.0+cu128`, CUDA compiler `12.9.1` and NumPy `1.26.4`;
- source-build the pinned cumm/spconv/MM stack on an aarch64 GH200 node;
- remove hard-coded pre-Hopper build targets and compile required ops for
  `sm_90`;
- fail closed on user-site, old-prefix, import-origin or dynamic-library leakage;
- validate required BEVFusion operators in forward and backward.

Allowed compatibility changes are build/setup, architecture flags, package/API,
dtype/device/index plumbing, lazy import of unused operators and output-neutral
runner/checkpoint/logging fixes. Model, loss, target, decode, voxel math, sweep,
augmentation, optimizer, scheduler and metric changes are forbidden.

### WP2 — data, operator and checkpoint qualification

- expose the existing stored-ZIP payloads through a proper MMEngine file backend;
- create the upstream annotation/info representation for official train/val
  without extracting the full dataset;
- verify token coverage, six-camera payloads, keyframe plus nine-sweep chains,
  calibration, box/yaw/velocity and class conventions;
- strict-load the official OpenMMLab LiDAR/Fusion checkpoints;
- compare extracted-mini and ZIP-backed inputs on identical sample tokens;
- run finite FP32 and FP16/AMP operator and one-batch inference checks.

### WP3 — published checkpoint oracle and resource measurement

- run exactly one complete official-val evaluation for the pinned LiDAR
  checkpoint and one for the pinned Fusion checkpoint;
- emit standard nuScenes `results.json`;
- compare upstream evaluation with the existing official-devkit evaluator;
- measure sustained loader/training-step throughput, peak memory, checkpoint
  overhead and evaluation duration without capability training;
- calculate the exact Envelope-B resource request from measured production paths.

WP3 may not select a local model or reinterpret published checkpoints as fresh
training evidence.

### WP4 — fresh LiDAR-to-Fusion reproduction

WP4 runs only under a later owner-approved Envelope B:

1. fresh LiDAR training on official train;
2. terminal LiDAR official-val evaluation;
3. staged Fusion initialization from that fresh LiDAR checkpoint plus the exact
   upstream image initializer;
4. fresh Fusion training on official train;
5. terminal Fusion official-val evaluation;
6. fresh Fusion-minus-LiDAR contribution and reproduction disposition;
7. one independent read-only result review.

There is no standalone Camera run, validation checkpoint selection, LR/seed/model
search or automatic scientific repair.

## 5. Three owner gates

### `P1R-G0 PLAN_FREEZE`

Closed by O-151 at the containing commit. It freezes this plan, documentation
topology, base evidence, work packages, gate order and envelope boundaries. It
does not activate external actions.

### `P1R-G1 ORACLE_ACCEPT`

After WP1-WP3, the owner receives:

- the clean external reference commit and frozen runtime candidate;
- operator/checkpoint/data/evaluator evidence;
- official-val LiDAR/Fusion oracle metrics;
- measured throughput, memory and runtime;
- the exact Envelope-B resource request.

Only an oracle PASS can activate WP4. Failure may close as an honest GH200-port
negative or receive one explicit cause-directed amendment.

### `P1R-G2 REPRODUCTION_ACCEPT`

After WP4 and independent result review, the owner chooses:

- accept and freeze the upstream reference reproduction;
- record an honest negative;
- authorize one explicit, causal scientific amendment.

Promotion of the external repository and any future FL/security work remain
separate decisions.

## 6. Envelope A — engineering and checkpoint-oracle qualification

```text
PHASE_AND_ENVELOPE: S10 Phase-I-R / Envelope A
REQUEST_STATE: FROZEN DRAFT / NOT ACTIVATED
PLAN_BASE: containing plan-freeze commit derived linearly from 714f69e...
OBJECTIVE: complete WP0-WP3 and reach P1R-G1 with a trustworthy GH200 runtime,
           official checkpoint reproduction and measured fresh-training request
SCIENTIFIC_CANDIDATES: none
ORACLE_CHECKPOINTS: one OpenMMLab LiDAR and one OpenMMLab Fusion checkpoint
DATA: extracted mini plus official nuScenes train/val through read-only ZIP access
TRAINING_LIMIT: at most 16 optimizer updates and 512 sample presentations
OFFICIAL_VAL: one complete LiDAR oracle plus one complete Fusion oracle
DOWNLOAD: pinned sources and checkpoints only; aggregate <=2 GiB
STORAGE: isolated root above; <=80 GiB mutable footprint; full extraction forbidden
GPU: one GH200 per job
PER_JOB_LIMIT: <=4 hours
AGGREGATE_CEILING: 12.0 charged GH200-hours
EXPECTED_USE: 6-8 charged GH200-hours
MAX_CONCURRENCY: 1
SUBMISSION_POLICY: no numeric cap; fresh outputs; O-149 diagnosed remediation
CPU_ONLY: max concurrency 1; <=16 CPU / <=128 GiB / <=16 aggregate node-hours
OUTPUT_ROOT: /nobackup/.../bevfusion_ref/outputs/phase_i_r/<PLAN_SHA12>/<RUN_ID>/
PUSH/UPLOAD/PUBLICATION: forbidden
ALLOWED_INTERPRETATION: runtime/data/operator/evaluator/reference-checkpoint compatibility
FORBIDDEN_INTERPRETATION: fresh capability, W_base, FL, attack or defense evidence
OWNER_APPROVAL: pending exact containing-commit activation
```

Within an activated Envelope A, S00 diagnoses, commits and reruns unambiguous
frozen-semantics compatibility defects without per-submission owner approval.
Blind identical retries are forbidden.

Mandatory owner escalation occurs for:

- any model/reference math, tensor shape, voxelization semantics, sweep or
  augmentation change;
- optimizer, scheduler, normalization, precision or metric/evaluator change;
- another framework or PyTorch major-version profile;
- uncertain diagnosis;
- recurrence of the same root blocker after its targeted repair;
- aggregate resource exhaustion.

## 7. Envelope B — fresh upstream reproduction

```text
PHASE_AND_ENVELOPE: S10 Phase-I-R / Envelope B
REQUEST_STATE: DESIGN ONLY / NOT YET FREEZABLE
OBJECTIVE: fresh official-train LiDAR -> Fusion reproduction and terminal
           official-val assessment
MODEL: frozen OpenMMLab upstream graph and resolved configs from P1R-G1
DATA: official train; official val terminal evaluation only
ORDER: LiDAR -> Fusion
INITIALIZATION: LiDAR scratch; Fusion from fresh LiDAR plus upstream image initializer
SEED: 0
CHECKPOINT_SELECTION: terminal only
STANDALONE_CAMERA: none
MAX_CONCURRENCY: 1 unless the exact G1 request proves a science-equivalent topology
GPU_HOUR_CEILING: calculated and frozen only from WP3 evidence
FL/CLIENTS/ATTACK/DEFENSE: forbidden
OWNER_APPROVAL: pending P1R-G1
```

The exact ceiling is:

```text
H_B = contingency * (
    exact LiDAR exposure / measured LiDAR throughput
  + exact Fusion exposure / measured Fusion throughput
  + two terminal evaluations
  + checkpoint/resume reserve
)
```

Envelope B has no automatic LR, model, data, seed or precision search. A weak
fresh result is evidence, not permission to tune.

## 8. New-session handoff

The execution session starts only from the exact containing control commit in a
dedicated Codex-managed worktree based on
`codex/s10-bevfusion-reference-reproduction`. The source branch records
provenance and the future delivery ref; the managed worktree itself is expected
to remain detached at the exact control commit until separate branch-write
authority is granted. Its kickoff must verify:

```text
BASE_SHA: exact control commit
BASE_EVIDENCE: 714f69eac2a0857dc8435cd9ee8bc202d1035456
SOURCE_BRANCH: codex/s10-bevfusion-reference-reproduction
EXPECTED_REF_MODE: detached@BASE_SHA
CONTROL_REPOSITORY: current fl_weather_project Git
EXTERNAL_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/bevfusion_ref
ACTIVE_PLAN: handoffs/S10/REFERENCE_REPRODUCTION_PLAN.md
ACTIVE_LEDGER: handoffs/S10/RUN_REQUEST.md Phase-I-R section
COMPUTE_STATE: no external action until exact Envelope-A activation
OUT_OF_SCOPE: local-model repair, FL, clients, attack, defense, push and upload
```

The new session reads the canonical plan and ledger, verifies both Git topologies
before acting, and does not recreate the old Phase-I implementation or evidence
harness.
