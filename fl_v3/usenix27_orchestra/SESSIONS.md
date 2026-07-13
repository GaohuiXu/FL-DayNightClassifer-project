# USENIX Security '27 Orchestra — session contracts

> **Status (2026-07-13):** O-092 freezes old S07-B endpoint
> `e231808e77388d69053dcbced6e754dbe3468aef` as historical/negative evidence.
> The audited cleanup code base is
> `4ce2366df2925161adae8fea393d5fca64836d40`. S01/S07-A and reviewed
> S02-S06 remain accepted clean foundations within their bounded evidence.
>
> S07-C legacy-security cleanup and independent S07-C-R are complete with a
> static-scope PASS. Canonical acceptance is sealed at
> `70bcd856f7ebb411eb2887e7ab71ef41ed13271f`. Simplified clean S07-B Job `380806`
> verified the current Arrhenius environment and clean FedAvg, then reached real
> C/L/F forward/finite-loss/backward; all three first gradient-norm gates failed,
> before step/skip metrics were emitted. The exact compute approval is consumed, review is not
> launched, and remediation diagnosis requires a new owner decision. S12 is
> deferred and unaccepted. S13 attack work
> requires a later owner-approved threat model after CL freeze and clean
> Protocol-B adaptation; S14 remains blocked until a viable undefended attack.
>
> Worker sessions do not edit canonical files. `fl_v3/collab/**` and old
> cycle_04 documents are read-only evidence, never delivery or implementation
> authority. Exact compute, commit, merge, push and upload boundaries follow
> ORCHESTRA.md O-092 and AGENTS.md.
>
> **Canonical objective/gates:** [`ORCHESTRA.md`](ORCHESTRA.md).
> **Copy-ready prompts:** [`KICKOFFS.md`](KICKOFFS.md).

The plan retains worker identifiers `S01`-`S15`. `S00` is the separate
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
accepted S01/S07-A + S02-S06
              │
4ce2366 audited code base
              │
canonical-only P
              │
S07-C cleanup ── S07-C-R static PASS
              │
canonical S07-C acceptance `70bcd85`
              │
docs-only S07-B-COMPLETE packet seal
              │
clean S07-B completion ── independent clean integration review
              │
S08/S09/S10 clean CL selection ── S11 CL freeze
              │
deferred S12 clean Protocol-A/B review
              │
S13 clean adaptation first ── later owner-approved new attack
              │
S14 only after viable undefended attack ── S15
```

The frozen e231 S07-B, O-032-O-091, legacy T5/T6/T7 and old defense chain have
no outgoing implementation edge into this graph.

### Active waves

| Wave | Sessions | Boundary |
|---|---|---|
| Cleanup | S07-C | complete at implementation `a16c2cd`; no runtime claim |
| Review | S07-C-R | PASS at `b8e11bc`; REVIEW.md only; reviewer history is never merged |
| Completion | clean S07-B | active numerical remediation boundary after Job `380806`; environment/FedAvg passed, C/L/F backward reached, no accepted successful-step evidence; no retry/review authorized |
| CL | S08-S11 | starts only after clean S07-B review; every scientific run requires exact approval |
| Protocol | deferred S12, then S13 | re-review clean Protocol A/B after CL readiness; attack remains blocked until clean adaptation passes |
| Security | later S13/S14 | new owner-approved threat model only; S14 requires a viable undefended attack |

### Review convention

For implementation sessions S01-S07 and security-code sessions S13-S14, use a
separate review session `Sxx-R` before integration. The review reads the exact diff,
`HANDOFF.md`, any `RUN_REQUEST.md`/`RESULTS.md`, and the relevant reference;
findings are ordered by scientific and correctness severity. Execution-only
sessions S08-S11 still require an independent results/science review of manifests,
logs, metrics, failed cells, and checkpoint/config hashes.

### Evidence-driven refinement convention

Completion of a worker does not freeze the original downstream plan. S00 first
checks `HANDOFF.md` and the actual diff/artifacts, then shows the complete independent
`Sxx-R` launch packet. After explicit owner launch authorization, S00 creates the
review task through Codex and later accepts or returns that review.
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

The recommended default is that S00 shows the complete launch packet and the owner
approves the exact task; S00 may then create the `Worktree` task through Codex from
the named SHA. The owner may instead provision it manually in the task UI. The
kickoff prompt does not ask the new session to create its own worktree. It carries `BASE_SHA`,
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
| S00 | Orchestra/owner decisions | — | approved contracts and status ledger | active; sole pinned canonical writer |
| S01 | Shared nuScenes ZIP backend | S00 kickoff | data backend, manifest, parity/coverage tests | reviewed PASS; integrated through S07-A, historical `t1.v1` caches forbidden |
| S02 | CL P0 correctness | S00 kickoff | pillar/Gaussian fixes and invariance tests | reviewed PASS at delivery `3aebf2d` / review `df142dc`; accepted S07-B dependency only |
| S03 | Camera branch architecture | S00 kickoff | corrected stride-8 independent camera modules | reviewed module PASS at delivery `5089383` / review `2f62e57`; accepted S07-B dependency with production-shape limits |
| S04 | LiDAR SECOND architecture | S00 kickoff | sparse XY-downsampling encoder contract | reviewed module PASS at worker `483e149` / executable `8498597` / review `a0763c2`; Job `341695` 15/15 bounded runtime PASS; accepted S07-B dependency with same-instance concurrency/reentrancy integration requirement |
| S05 | Detection head and decode | S00 kickoff | multi-task CenterHead and deterministic NMS | reviewed PASS at worker `a9c801f` / execution `96e509b` / review `1c44084`; Job `336731` 43/44 negative preserved, Job `336738` 44/44 focused runtime PASS; accepted S07-B dependency only |
| S06 | Production modes/runtime | S07-A data contract + S00 kickoff | C/L/F modes, config, resume, loader, eval | reviewed bounded PASS under O-031 at worker `6b7ef29` / executable `c330c72` / review `ca7bbd7`; Job `341997` FAILED 45/62 preserved, Job `342014` PASS 66/66; accepted S07-B candidate dependency with P3 integration gates, no production/full-data/scientific PASS |
| S07 | Cleanup then clean integration completion | accepted S01/S07-A and S02-S06 | S07-C cleanup, independent S07-C-R, then simplified clean S07-B completion | S07-C accepted at `70bcd85`; review `b8e11bc` static PASS; Job `380806` reached C/L/F finite-loss backward but norm gates failed before step/skip evidence; exact test-only D1 gradient request prepared, NOT RUN, no compute/retry authorized |
| S08 | Camera scientific run | S07 PASS | `C-STR8` full-val result/checkpoint | planned |
| S09 | LiDAR scientific runs | S07 PASS | `L-P020` and `L-S075` results/checkpoints | planned |
| S10 | Fusion and recipe selection | S08, S09 | `F-U`/`F-CBGS`, optional init A/B, `CL-PILOT` | planned |
| S11 | Final CL capability freeze | S10 `CL-PILOT` | C/L/F x3 seeds, frozen architecture/config schema | planned |
| S12 | FL protocols, tail split, and security thesis | clean CL readiness + fresh owner review | clean Protocol A/B contract and split proposal | deferred; old proposal is unreviewed evidence and not an active worktree/input |
| S13 | Clean FL adaptation, then later new attack | clean S07-B + CL freeze + owner-approved S12/threat model | first clean Protocol-B adaptation and Protocol-A control; attack only under a later envelope | blocked until clean prerequisites; legacy T5 import forbidden |
| S14 | New defense after viable attack | reviewed viable undefended S13 attack | later defense/adaptive/generalization work | blocked; legacy defenses are not baselines or implementation authority |
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

**Issued envelope (2026-07-10).** `BASE_SHA` is
`f262f6bea037580065a8505008773c04fdd259f5`; the verified detached worktree is
`/home/gaohui/.codex/worktrees/1ab2/fl_weather_project`. Writable scope is narrowed
to `AGENTS.md`, `fl_v3/docs/env.md`, `fl_v3/src/fl_v3/data/nuscenes/**`,
`fl_v3/scripts/build_nuscenes_cache.py`, new S01-prefixed ZIP scripts, the existing
nuScenes dataset/info-cache/path tests plus new S01 ZIP tests, and
`handoffs/S01/**`. Everything else is read-only. O-009 now permits a bounded
compute-node ZIP read/decode smoke after the exact preflight is recorded in
`RUN_REQUEST.md`; full member coverage, full-data throughput/profile, or any larger
run still requires exact owner approval.

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

**Reviewed outcome (2026-07-11).** Worker
`abe5c58b174dbbe1f7045ce91c8b15168d97b87b` and independent review artifact
`7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc` are accepted as an S07 dependency.
Historical job `332651` supplies checksummed ten-archive coverage/loader evidence;
job `333206` supplies checksummed real-mini parity, fork/spawn, cache-depth, and
integrity remediation evidence. This is not a merge, production-cache freeze,
model-readiness result, or scientific PASS.

---

### S02 — CL P0 correctness

**Kickoff.** “Fix only correctness blockers that invalidate CL results: per-sample
pillar caps and the approved Gaussian radius semantics; add adversarial batch and
target golden tests.”

**Owns.** `models/fusion/lidar_encoder.py`, `models/fusion/losses.py`, and focused
tests. During the parallel S02-S05 wave, `losses.py` is exclusive to S02; S05 reads
it but does not edit it without an S00 amendment. S02 does not redesign sparse
SECOND, camera, head, or trainer code.

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

**Owner-approved Wave-A contract (O-017).** Swin-T; valid stride-8 multi-scale output;
0.5 m depth bins;
aspect-preserving resize/crop with calibration consistency; pure-camera view
transform without LiDAR-conditioned inputs. O-017 authorizes this implementation;
any deviation returns to S00/owner before editing.

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

**Owner-approved Wave-A contract (O-017).** Voxel size `0.075x0.075x0.2 m`; sparse
3D stages with
approximately 8x XY reduction; densification only at the reduced grid; train/eval
max-voxel settings separately configurable; fp16 sparse convolution with fp32
reference support. O-017 authorizes this implementation; any deviation returns to
S00/owner before editing.

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
CenterHead and deterministic task/class-aware box decode/NMS; apply the declared
O-018 no-starvation and GroupNorm adaptations; validate coordinates against
nuScenes conventions.”

**Owns.** `models/fusion/head.py`, a separable decode/NMS module if needed,
head-specific loss adapters/new modules, and focused tests. It does not edit the
production detector wiring. While S02 is active, existing
`models/fusion/losses.py` is read-only to S05; any required shared-interface edit
returns to S00 after S02 review.

**Deliverables.**

- official-reference nuScenes task grouping and separate heads for
  heatmap/regression fields;
- O-018 per-class K=500 without the official second task-wide K; deterministic
  score/class/spatial tie ordering and at most 500/1000 candidates for one/two-class
  tasks before the pinned official task-wide NMS;
- deterministic circle/rotate NMS with explicit thresholds;
- explicit class-name mapping from task-local labels to canonical devkit-global
  `DETECTION_NAMES` IDs; task-flatten offsets are forbidden;
- canonical box dimension/yaw/velocity conversion and round-trip fixtures;
- decode-only comparison on an existing checkpoint where interface-compatible.

**Gate.** Exact reference fixture parity where O-018 does not declare an adaptation;
single-class decode parity; B=1/B>1 and input-order stability; equal-score
determinism; no cross-class candidate starvation or duplicate-box explosion; and
explicit `construction_vehicle`/`bus`/`barrier`/`pedestrian`/`traffic_cone` mappings
plus official nuScenes submission conversion pass. S05 must not claim exact
official decode equivalence for multi-class tasks.

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
- every production cache load passes the resolved `n_sweeps` explicitly and records
  the exact `t1.v2` cache hash plus ZIP manifest hash; scientific entry points may
  not rely on single-file cache-depth autodiscovery;
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

**O-027 launch refinement.** S06 runs before S07-B and consumes S02-S05 only as
reviewed interface contracts. It must not land or copy their implementation
histories. The runtime must pin S04's exact spconv 2.3.8 requirement and serialize
same-instance encoder forward/mode transitions unless separately protected and
tested. The absent full trainval `t1.v2` cache is a required fail-closed identity
field, not permission to materialize it; synthetic/mini fixtures exercise the
resolver until an exact owner-approved cache job freezes production hashes.

---

### S07-C — Legacy-security cleanup

**Audited code base.**
`4ce2366df2925161adae8fea393d5fca64836d40`; the worker starts from the
canonical-only P commit whose diff against that base is exactly the three
canonical Orchestra files.

**Source branch.** `codex/s07-c-legacy-security-cleanup`.

**Purpose.** Remove active legacy attack/readiness/defense routes and establish
one explicit clean FedAvg path without changing the accepted clean model, data,
evaluation or runtime contracts.

**Must remove.**

- `src/fl_v3/attacks/**`, attack-specific ASR/frustum/report/viz modules,
  T4/T5 configs/scripts/tests and the old S07-B static launcher;
- `strategy/defenses/**`, `strategy/gradient_metrics.py`, defense tests and
  oracle fixtures;
- attack/defense config keys, registry/import paths, FLAME-only sklearn
  dependency surfaces and stale active documentation.

**Must refactor and keep.**

- clean weighted FedAvg outside the legacy defense namespace;
- server_app/local_runner/Flower strategy wiring with no defense selector;
- clean tasks, runtime/checkpoint provenance, official clean DetectionEval,
  clean fusion visualization and focused clean integration tests.

**Protected foundations.** S01 ZIP/cache/path/partition contracts; S02-S05 C/L/F
model construction; S06 resolved config, precision, loop, checkpoint, resume and
runtime state; centralized training; five clean S07-B C/L/F templates; official
box-to-global/DetectionEval; clean client/server, sampling, server optimizer, EMA
and trainable-state semantics.

**Forbidden.** No recovery/import/cherry-pick of legacy T5/T6/T7, e231 or old
cycle/collab code; no new attack, defense, Protocol-B split, science, merge, push
or upload. Do not import the bf480ea spawn diff.

**Compute.** None at kickoff. Static/focused local checks are expected. Any GH200
fallback requires a new exact immutable RUN_REQUEST and owner/S00 approval.

**Delivery.** `handoffs/S07-C/HANDOFF.md`, optional unapproved
`RUN_REQUEST.md`, exact worker SHA and path-by-path deletion/refactor proof.

**Consumed amendment O-092-A1.** S00 completeness audit rejected the first
uncommitted delivery before reviewer launch. The same worker must restore
scikit-learn as a pinned nuScenes runtime dependency (the devkit is installed
with `--no-deps` and imports it), remove the remaining clean-path legacy aliases
and selector-compatible parameters by migrating their callers, shrink VizWriter
to clean stages, remove the named fail-closed legacy P3/T3 scripts, stop active
scripts from writing `fl_v3/collab/**`, and clean active T5/T6/T7 authority
wording. Update the handoff/tombstone evidence and stop again for S00 audit.
No compute or Git publication authority is added.

**Consumed amendment O-092-A2.** After inspecting A1 source, diff and package, S00
confirmed the named A1 blockers are closed. The owner authorized the same worker
to delete the exact 16 obsolete A40/T3, MCR P1/P3, Stop-E and mini-matrix/profile
scripts plus three dedicated tests listed in `KICKOFFS.md`; shared tests/docs lose
only their imports or authority wording for those routes. The exact retained
Arrhenius environment/general smoke, centralized training, S01 ZIP, S06 runtime
and S07-A cache/provenance scripts are protected. Update HANDOFF/RESULTS and stop
for S00 audit with no compute, commit/ref or reviewer launch.

**S00 completeness gate.** The cumulative A2 delivery contains 70 deleted and 64
modified tracked paths plus the three-file handoff package. S00 independently
confirmed the exact 18-script set, 16 protected byte-identical scripts, 11
semantic-AST-equal foundations, absent active legacy routes, source compile, 27
JSON files, TOML, 17 shell files and `git diff --check`. The owner authorized a
local durable implementation commit and handoff seal as a linear successor of
the committed canonical amendment, then independent S07-C-R. Runtime/dependency
tests remain NOT RUN and are reviewer residual risk, not a cleanup blocker.

**Durable identities.** Canonical A1/A2 parent
`f7c696345b24b0e1227b1a52f3b47fb14e9120f5`; original detached snapshot
`9f06875e1b865734950abcf3b6de36ad06a0ac7b` (provenance only); worker
implementation `a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`; handoff seal
`f736f41371666725a11d51bc3b01c6ececb59d50`. The snapshot and canonical-parent
implementation share patch-id `8f89c30d21164e80ec73f6a01eab33621e984789`.
The branch was fast-forwarded to the handoff seal; no worker or reviewer history
was merged.

### S07-C-R — Independent cleanup review

Completed from exact launch base `6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2`.
Review-only commit `b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5`
adds only `handoffs/S07-C/REVIEW.md` and returns PASS at static code/source/config/
test/docs scope with no P0-P3 finding. REVIEW.md SHA-256 is
`588cfd0f91a2f70cbdcc6bf94a2279fc3cca693c9cd14f9d9909f02df769d8f5`.
Reviewer history remains separate and unmerged. Dependency-backed and GH200
runtime remain NOT RUN.

### S07-B-COMPLETE — Simplified clean integration completion

The accepted cleanup/code anchor is
`70bcd856f7ebb411eb2887e7ab71ef41ed13271f`. The worker starts on
`codex/s07-b-clean-completion` only after owner launch approval, from the exact
docs-only S00 packet-seal commit containing the filled `KICKOFFS.md` contract.
That full packet SHA must be copied into the task envelope. The separate reviewer
commit is evidence, not an ancestor of the completion branch.

This is verification-first clean engineering. Own only the exact paths in the
filled kickoff: clean C/L/F integration wrappers and templates, S06 runtime/
checkpoint/resume, official clean evaluation adapters, selected S01 mini data/
ZIP lifecycle, clean FedAvg/Flower integration, one focused completion test and
the handoff package. Source edits require a demonstrated clean-contract failure,
except for the mandatory `flwr_config.toml` cleanup.

`flwr_config.toml` must drop active T3/Path-A/Path-B/4-GPU/overcommit and legacy
`collab/**` authority profiles, retaining only a CPU local smoke profile and one
single-GPU sequential clean profile. The validation default is plain FedAvg with
no server EMA. FedOpt/EMA implementation capability remains preserved but is not
the completion default or a Protocol-A/B claim. Old P1/T3/MCR configs and all
scripts are read-only; no launcher or harness may be created or recovered.

No T5/T6/T7, attack, defense, S12 split, full cache/trainval, 100/1000-step,
metrics campaign, profile, Ray live federation, DDP, multi-GPU, actor/process/
seed matrix, retry or scientific work is included. GH200 is not approved at
kickoff. A later exact owner/S00-approved RUN_REQUEST may contain only the single
bounded sequential engineering job specified in `KICKOFFS.md`.

Owner amendment O-092-A3 retires the expanded audit wrapper and full integrated
retest. Accepted S01 and S02--S06 reviewed evidence remains upstream evidence and
is not re-executed for the two-file completion diff. The only candidate runtime
delta is clean profile plus one C/L/F fp16 optimizer update with workers 0,
followed by one independently timed workers-0-versus-2 first-batch check using
`TMPDIR=/tmp`. The job contains no Git/dependency checkout audit, source/archive
manifest, warnings-as-errors, custom cache harness or 205-case suite.

Exact Job `380806` consumed the simplified approval. It passed environment and
clean-FedAvg identity, then reached the complete real-mini six-task C/L/F
forward/loss/backward paths. The first unscaled gradient norm was nonfinite in all
three modes; assertions preceded durable step/skip metrics. The loader phase was
not run. This is a shared fp16 integration blocker; it is not permission to retry, lower
the scale, weaken the gate, or launch review.

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

**Current status under O-092: deferred.** The existing S12 proposal is
unreviewed evidence only and will not be aggregated into the cleanup worktree.
Before later use it must be re-audited against the cleaned codebase and the
owner's CL-first/clean-Protocol-B ordering.

**Kickoff.** “In parallel with CL engineering, turn Protocol A and the owner-approved
primary vendor-style Protocol B into an auditable experimental contract: define base/tail
ownership, clients, clean baselines, threat model, novelty boundary, RQs, evidence
table, and paper skeleton—without claiming CL engineering as novelty.”

**Owns.** Literature/claim map, data/split specification, FL protocol contract, and
draft paper structure. It may build read-only split statistics/proposals, but it
does not train models or submit jobs.

**Issued envelope (2026-07-10).** `BASE_SHA` is
`f262f6bea037580065a8505008773c04fdd259f5`; the verified detached worktree is
`/home/gaohui/.codex/worktrees/aada/fl_weather_project`. Writable scope is only
`handoffs/S12/**`. All source, partition/data/eval code, canonical documents,
paper/protocol source files, and `fl_v3/collab/` are read-only. The session may
produce evidence and explicit proposals but may not freeze or materialize a split,
client assignment, threat-model parameter, metric, or claim. `APPROVED_COMPUTE` is
`none`; a large trainval statistics scan requires an exact approved
`RUN_REQUEST.md`.

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

**Current status under O-092: blocked/deferred.** This section is a future design
outline, not a current kickoff. First establish clean Protocol-B adaptation and
the Protocol-A control after CL freeze. A new attack requires a later explicit
owner-approved threat model and must be implemented without importing legacy T5.

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

**Current status under O-092: blocked.** No defense implementation or evaluation
starts until a new undefended S13 attack is independently shown viable. Legacy
T6/T7 and old defense-registry implementations are not baselines or code sources.

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

## 4. Active launch order after O-092

1. S07-C canonical preparation, worker cleanup and durable handoff are complete.
2. S07-C-R is complete at review-only commit `b8e11bc`; reviewer history remains
   separate and unmerged.
3. Canonical S07-C acceptance is sealed at `70bcd85`; the S07-B-COMPLETE packet,
   branch and two-file executable were launched and sealed.
4. Job `380806` consumed the simplified runtime approval and exposed shared
   nonfinite fp16 gradients before any optimizer step. Preserve it, prepare the
   smallest causal remediation proposal, and obtain a new exact owner decision
   before editing the numerical contract or submitting compute.
5. Resume S08-S11 planning only after clean S07-B completion is independently
   accepted. Scientific jobs retain separate exact authorization.
6. Re-audit S12 later. Establish clean Protocol-B/Protocol-A foundations before
   any new S13 attack envelope.
7. Start S14 only after an independently reviewed viable undefended S13 attack.
   Keep S15 dependent on checksummed accepted evidence.
## 5. Durable delivery and Orchestra review

Every worker writes to
`fl_v3/usenix27_orchestra/handoffs/Sxx/` on its scoped branch:

| File | When | Required content |
|---|---|---|
| `HANDOFF.md` | every session completion | base/branch/commit, exact files/semantics, references, tests, gates, hashes, negative findings, allowed/forbidden claims, residual risks |
| `RUN_REQUEST.md` | before any Slurm/material compute | O-009 smoke: HEAD + diff hash, bounded data scope, resources, command/output, stop criteria, standing-policy citation; full/scientific work: immutable commit/config/data manifest, cells, seeds, GPU/time budget, command/output, stop criteria, exact owner approval |
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

For O-009 smoke, update the audit record before submission whenever HEAD/diff,
command, data scope, or resources change; crossing the standing boundary requires
owner permission. For full/scientific work, any changed commit, config, split,
cell, seed, resource request, or command after approval invalidates
`RUN_REQUEST.md`. There is no automatic resubmission or opportunistic filling of
spare GPUs.

The durable completion sequence is:

1. worker writes the complete handoff package and reports its exact SHA/diff;
2. S00 performs a completeness/provenance check;
3. the owner explicitly authorizes a local commit in that worker's own detached
   worktree and a scoped branch/ref to preserve it; the resulting commit is the
   durable `WORKER_SHA` and this does not authorize merge or push;
4. S00 presents the exact Sxx-R launch packet; after explicit owner authorization,
   S00 creates the independent review task at that `WORKER_SHA`;
5. findings are fixed/re-reviewed or the review is accepted;
6. S00 updates status and the change-control ledger;
7. S00 refines affected unstarted session plans/kickoffs within its authority and
   presents each complete launch packet before creating another task;
8. the owner approves any material/locked change and every task launch before the
   revised kickoff is issued.

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
RUN_REQUEST.md. A bounded non-scientific smoke that satisfies O-009 may proceed
after recording its exact preflight; full tests, full-data/profile/metric work,
matrices, seeds, and reruns stop until the owner approves the exact request. Never
infer full-run/matrix/upload permission from approval of a design or session.

Your kickoff must provide BASE_SHA, SOURCE_BRANCH, EXPECTED_REF_MODE, FILE_OWNERSHIP,
UPSTREAM_HANDOFFS_AND_SHAS, REASONING_EFFORT, and APPROVED_COMPUTE. Default to
`xhigh`; `ultra` requires a recorded task-specific complexity reason. Do not create,
switch, move, remove, or prune a worktree/branch. A Codex-managed detached HEAD is
valid if it equals BASE_SHA.
Before editing, verify and report repository root, HEAD, ref mode, git status, exact
intended files, and any contract ambiguity; stop on a real mismatch.
After editing: run focused CPU checks and the smallest relevant owner-authorized
GH200/Slurm gate; return the standard handoff from ORCHESTRA.md with exact commands,
job IDs, outputs, hashes, and residual risks. Do not commit/merge/push without owner
authorization.
```
