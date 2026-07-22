# AGENTS.md - project root

Operating instructions for Codex and other non-Claude coding agents.
Claude's parallel file is [`CLAUDE.md`](CLAUDE.md), but this file is the
binding instruction file for Codex. If older project docs conflict with the
current Arrhenius status here, follow this file and `fl_v3/docs/env.md`, then
surface the conflict instead of silently inheriting stale rules.

`fl_v2/` keeps its own frozen `fl_v2/AGENTS.md`; ignore it unless deliberately
working inside `fl_v2/`.

## Current Project State

The active project is `fl_v3/`: a federated multimodal autonomous-driving
perception platform on nuScenes, with a backdoor attack/defense benchmark as
the research target. `fl_v2/` is frozen historical/oracle code. Do not modify
`fl_v2/` unless the user explicitly asks.

The active long-lived branch is `v3-ad-perception`. Older documents may describe
an Alvis-first workflow, pure-PyTorch LiDAR, strict bit-determinism, or a
Claude-builds/Codex-only-reviews loop. Those are historical unless explicitly
reconfirmed.

The active collaboration entry point is:

```text
fl_v3/usenix27_orchestra/ORCHESTRA.md
fl_v3/usenix27_orchestra/SESSIONS.md
fl_v3/usenix27_orchestra/KICKOFFS.md
fl_v3/usenix27_orchestra/handoffs/S10/PHASE_I_PLAN.md  # binding for Phase-I work
```

This is milestone/work orchestration, not a research-cycle document. Names beginning
with `cycle_*` remain reserved for the project's experimental-design cycles under
`fl_v3/docs/cycle_04/` and `fl_v3/docs/roadmap/`. Do not create another
`cycle*_orchestra` folder.

S07 clean engineering, S08 precision qualification, and S09 full-pipeline
engineering performance/readiness are closed. S08's accepted precision policy is
integrated at `28f79802c0868afa6290d74ae6aeb9d23c7d088f`; S09's accepted
closing commit is `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`.

S10 Phase I-P is active on `codex/s10-phase1p-throughput-preflight`, created from
the frozen `codex/s10-phase1-branch-qualification` control at
`f1a2babda8dafd181b5a5144ab025a3f6be21cc2`. Earlier S10 was advanced linearly
from `codex/s10-cl-model-recipe` and audited base
`a080d49c1c22de20ccb5b1353d4922c7df14a729`. Terminal evidence through O-150 and
the later owner-approved Phase I-P work is preserved in
`fl_v3/usenix27_orchestra/handoffs/S10/`; the compact state is:

- STOP-A's train-only split/evaluator gate is closed and reusable.
- STOP-B is closed `INCONCLUSIVE`; large-gradient causality was not established.
- C0-v2 showed bounded numerical health and a positive internal F-minus-L signal,
  but did not select a production graph or recipe.
- C1-A localized the current W0/panel large LiDAR-stem gradient to the tiny-group
  GN path. C1-B0/B1 showed that BN1d strongly reduces that gradient without
  establishing a capability advantage. The B4 one-epoch proxy scores were low,
  and the B8 candidate was faster but materially worse on the same internal
  evaluator. None of these results establishes production capability or a
  comparison with the historical Alvis detector.
- Phase-I Envelope A WP0-WP4 is terminal. Camera correctness/parity, checkpoint,
  end-to-end and memory gates passed, but the optimized BEV-pooling operator was
  only about 2.4% faster than the exact PyTorch fallback and failed the frozen
  1.25x promotion gate. O-150 accepts the numerically qualified PyTorch sorted
  `segment_reduce` fallback as the Phase-I Camera production backend, retains CUDA
  as an unpromoted option, and removes 1.25x as a capability prerequisite. LiDAR
  WP4 passed with the reference BN/no-GN TransFusion graph, exact keyframe GTDB,
  FP16 plus sparse FP32 island, and directly consumable qualified config and
  zero-update recovery checkpoint. No capability metric or training update ran.
- Phase I-P is terminal at the profiler level. Camera's promoted production recipe
  is same-node two-GH200 DDP at B16/rank; LiDAR's is single-GH200 B32. Both keep
  effective global B32 and the frozen exposure. Phase I-P measured engineering
  health and throughput only; it made no capability, mAP/NDS or selection claim.

Owner decision O-143 supersedes the active S10 six-stop execution order and the
S10-specific per-job immutable/no-retry/multi-document/reviewer workflow. It does
not erase prior results, relax data ownership or metric correctness, or authorize
compute. The active scientific order is now:

1. qualify camera and LiDAR branches independently, selecting a defensible
   architecture/initialization/training recipe for each;
2. perform staged fusion from the qualified branch checkpoints and establish
   absolute clean capability plus fusion contribution under aligned evaluation;
3. profile and optimize GH200 performance only after the capability gate passes.

Current-A2 and the old C→D→E→F execution path are paused. MIT BEVFusion's
published staged-pretraining and recipe choices are strong external anchors; S10
should test local implementation compatibility and capability, not spend compute
re-proving published conclusions without a concrete conflict. S11 and later
milestones remain pending.

Owner decision O-144 closed `P1-G0 PLAN_FREEZE` and makes
`fl_v3/usenix27_orchestra/handoffs/S10/PHASE_I_PLAN.md` binding for all Phase-I
work. It freezes physical B4 plus accumulation 8/effective B32, one ImageNet
Camera primary and one scratch LiDAR primary, role-bound D_fit CBGS/GT-paste,
seed 0, 20 epochs, terminal-only selection, two total candidates, and the
five-WP/three-gate/two-envelope workflow. At issuance, O-144 authorized
documentation only and did not activate implementation, checkpoint acquisition,
GTDB materialization, commit, GPU/Slurm execution, merge, push or upload.

Owner decision O-145 amended WP2/WP4 to require an independent in-tree port of the
pinned MIT optimized CUDA BEV-pooling operation, or a functionally equivalent
kernel, plus a labelled correctness fallback; FP32/FP16 forward and backward
parity; accepted-precision-policy checks; and GH200 operator plus aligned B4
end-to-end timing. At issuance, it authorized the O-145 documentation commit and
exact Envelope-A drafting only. The referenced Camera checkpoint is the MIT Camera
YAML's ImageNet `swin_tiny_patch4_window7_224.pth`, not the optional NuImages
checkpoint. No download, implementation, GTDB materialization or compute is
authorized by that decision alone. O-146 activated Envelope A; O-147 amended its
initial limits; O-148 replaced the mechanical submission stop with serial,
budget-limited completion authority at an unchanged `1.10` GH200-hour ceiling.
Envelope A consumed `0.516389` GH200-hours and is now closed at its mixed
Camera-negative/LiDAR-PASS result. Unused budget is not authority for another
cell, capability run, or Envelope B.

Owner decision O-149 establishes the completion-oriented engineering-validation
contract below. It removes a default numeric submission cap only inside an
explicitly approved validation envelope; it does not weaken scientific owner
gates or create standing compute authority.

Owner decision O-150 accepts the parity-qualified PyTorch sorted `segment_reduce`
path as the Phase-I Camera production backend and keeps the CUDA kernel available
only as an explicit, unpromoted optimization. Job H's failed 1.25x promotion gate
remains historical performance evidence but no longer gates Camera capability.
The later Phase I-P preflight promoted Camera two-GH200 B16/rank and LiDAR
one-GH200 B32 production recipes, with their ordinary-BN/worker-RNG changes
explicitly owner-accepted and effective global B32/exposure preserved. The revised
dual-branch Envelope-B request is materialized in `RUN_REQUEST.md` Section 7.4 at a
`30.0` charged-GH200-hour aggregate ceiling, maximum concurrency one, serial LiDAR
then Camera, and two fixed seed-0 candidates. The old 49.0-hour Section-7 object is
historical control only. Independent read-only recipe-freeze review of
`a4f6ca86ddd966bdffc74a37af3337ac6675e83a` closed
`PASS_WITH_RESIDUAL_RISK` with no open P0-P2. Revised Envelope B remains
serial at maximum concurrency one. The owner accepted the review verdict and its
single P3, named review seal
`1473ef67d9dc2949c49360b6826d0f30585f416f`, and accepted/activated the exact
Section-7.4 envelope, while explicitly forbidding submission in the current
session. A later execution session must re-verify the approved baseline, hashes,
clean worktree and fresh output root before using that authority.

For this stage, `fl_v3/collab/` is read-only legacy evidence. Agents may inspect and
cite it, but must not add or update plans, handoffs, reviews, results, or status
records there unless the owner explicitly requests a historical correction. All new
collaboration artifacts live under `fl_v3/usenix27_orchestra/`.

## Codex Role

Codex is not limited to after-the-fact review. Depending on the user's request,
Codex may:

- discuss and design architecture, experiment protocols, and migration plans;
- implement focused, reviewable code/docs/script changes;
- run local checks and explicitly owner-authorized Slurm smoke/profiling jobs;
- review Claude/Codex/user changes for scientific correctness;
- commit or merge only when the user explicitly authorizes it.

For large or scientifically risky changes, discuss the model, data path,
precision policy, metric, and acceptance criteria before editing. Once the
objective is clear, implement end to end rather than stopping at a proposal.

## Compute, Upload, And External-Action Authorization

Planning or implementing an experiment is not permission to execute it. Without an
explicit owner instruction scoped to the exact action, agents must not:

- submit `sbatch`/`srun` jobs, including engineering smoke jobs, except through an
  explicitly owner-approved bounded remediation loop, completion-oriented
  validation envelope, or S10 phase envelope described below;
- launch a trainval full run, experimental matrix, multi-seed campaign, long
  profiling job, FL campaign, attack/defense run, or automatic resubmission;
- expand an approved cell into additional seeds, ablations, reruns, or spare-GPU
  jobs;
- cancel or replace another session's jobs;
- upload datasets, checkpoints, logs, results, artifacts, or manuscripts to a
  remote service;
- push Git branches, create pull requests, submit a paper/artifact, or otherwise
  publish externally.

Agents may prepare scripts, configs, `RUN_REQUEST.md`, resource estimates, and
local/static/unit checks. By default, an execution approval is bound to the exact
commit, resolved config, data/split manifest, cells, seeds, command,
GPU/count/time budget, and output location stated in the request. Changing any of
these invalidates the approval and requires new permission. Exceptions are an
explicit O-107 derivation rule, an O-149 completion-oriented validation envelope,
or an owner-approved S10 phase envelope under O-143. Never infer full-run or
upload authorization from approval of an
architecture, plan, session, or code change.

Every material-compute session records its request and approval state in
`fl_v3/usenix27_orchestra/handoffs/Sxx/RUN_REQUEST.md`. Preparing or editing
that file does not grant approval.

Standing owner decision `O-009` permits only bounded engineering smoke: at most
one node/one GPU, 60 minutes per job, one concurrent job, and two cumulative
GPU-hours for the session, after the exact HEAD/diff, command, bounded data scope,
resources, output, and stop conditions are recorded in `RUN_REQUEST.md`. It never
covers full trainval cache generation/coverage/profile, model qualification or
training steps, scientific metrics, matrices, seeds, arrays, DDP, or publication.

Owner decision `O-107` allows one initial owner approval to opt into a **bounded
mechanical remediation loop** for an O-009 engineering smoke. The initial exact
request must bind the test objective/selectors, data scope, command family,
resource ceiling, output naming rule, stop conditions, and a cap of three total
submissions (the initial job plus at most two derived replacements) within O-009's
two cumulative GPU-hours. After a diagnosed failure, S00 may fix only an obvious
test, fixture, smoke wrapper, provenance/artifact check, or output-neutral
diagnostic-plumbing defect; it must freeze and record every derived immutable
source/snapshot, command/script hash, output path, and diagnosis in `RUN_REQUEST.md`
**before** submitting it. This is not permission for an identical retry, silent
resubmission, or spare-GPU expansion.

The loop stops and returns to the owner if a change may affect model outputs,
losses, gradients or accepted updates; data contents/ownership; precision policy
or regime; optimizer/scheduler/EMA behavior; metric or scientific interpretation;
test/data scope, seed, or resources; if classification is uncertain; if the same
blocker recurs; or when the approved submission/GPU-hour cap is reached. Scientific
and otherwise material jobs retain exact per-job owner approval. O-107 applies
prospectively and does not reinterpret historical job approvals.

### Completion-oriented engineering validation (O-149)

An owner may approve a validation/smoke envelope by binding its objective and
exit gate, frozen scientific semantics, data scope, command family, per-job
resources and wall limit, aggregate GPU-hour ceiling, concurrency, fresh-output
rule, and escalation conditions. Unless the owner explicitly adds a numeric cap,
submission count is then **not** a stop condition: aggregate GPU-hours and maximum
concurrency are the resource controls. This is never implied by O-009 alone.

Inside such an envelope, S00 diagnoses each failure, makes the smallest
single-correct-answer repair anchored to the already frozen semantics, validates
it, records the derived source/command/output, and immediately resubmits serially.
Eligible repairs include tests/fixtures, runner and API compatibility, missing or
incorrect config/schema parsing, discrete dtype/index plumbing, checkpoint I/O,
artifact publication/provenance, and logging. Blind identical retries and
spare-GPU expansion remain forbidden. Different diagnosed engineering defects do
not trigger an owner round trip merely because several occur in sequence.

The loop ends when the validation objective is met, the aggregate GPU-hour ceiling
would be exceeded, the same blocker recurs after its attempted repair, diagnosis is
ambiguous, or a proposed change crosses a scientific boundary. S00 returns to the
owner before changing candidates, model/reference math, data contents or ownership,
training recipe/exposure, precision policy, optimizer/scheduler/EMA, evaluator or
metric semantics, seeds, acceptance gates, scientific interpretation, aggregate
resources, or publication scope. Capability training, experimental cells,
multi-seed evidence and other scientific runs remain separately approved material
compute even when their preflight uses this engineering contract.

### S10 phase-level execution exception (O-143/O-149)

For S10 only, O-143 replaces the preceding per-job immutable/no-retry workflow
once the owner approves a **phase** compute envelope. A phase approval must bind
the scientific objective, candidate cap, data ownership/splits, evaluator and
metric semantics, seed policy, aggregate GPU-hour ceiling, submission policy and
concurrency, and stop/escalation conditions. A numeric submission cap binds only
when the owner explicitly sets one. It is not compute authority until those fields
and the resource ceiling are explicitly approved.

Inside an approved S10 phase, S00 may derive commands and resolved configs, apply
the unambiguous frozen-semantics engineering repairs listed under O-149, and
resubmit autonomously while staying inside the approved candidates, science,
aggregate resources and submission policy. It returns to the owner before any
change to model math, data ownership/content, recipe search space, evaluator or
metric semantics, seed policy, candidate count, scientific interpretation, or
aggregate resources. A new engineering symptom is diagnosed and repaired rather
than treated as a mechanical STOP; recurrence of the same blocker, ambiguity,
science-boundary pressure, or ceiling exhaustion stops the loop.

S10 provenance is phase-sized: record the Git SHA, resolved-config hash, split,
seed, command, resources, output path, and checkpoint/metric hashes for each
scientific run in one compact ledger. Detached source copies, recursive artifact
manifests, command-file hashes, stdout hashes, and duplicate narratives across
multiple documents are not required unless a specific high-risk boundary needs
them. Raw outputs remain immutable. A pre-model test/runner failure is an
engineering incident, not a scientific STOP verdict.

## Active Runtime: Arrhenius GH200

Arrhenius GH200 is the active runtime target. The validated environment is a
persistent conda/spconv environment under:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3
```

It is not stored in Git and is not recreated for every Slurm job. Normal jobs
activate it through `fl_v3/scripts/arrhenius_env.sh`. The login node is x86_64;
GH200 compute nodes are aarch64, so validate imports/training through Slurm
rather than treating login-node import failures as definitive.

All conda/venv/cache/build/data/output artifacts should live under
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui`, not under `$HOME`.

See `fl_v3/docs/env.md` for the current runtime contract. The read-only
`fl_v3/collab/arrhenius_migration.md` is historical bring-up evidence, not an active
handoff destination.

## Dependency And Precision Policy

Current Arrhenius facts:

- PyTorch/CUDA works on GH200 with source-built `cumm`/`spconv`.
- `mmdet3d` and `mmcv` remain excluded as framework dependencies.
- `spconv`/`cumm` are allowed and active as the Arrhenius sparse LiDAR stack;
  do not apply old "no spconv" rules without re-checking the current design.
- Direct sparse `torch.bfloat16` is not supported by the validated cumm/spconv
  path.
- S08 Q1/Q2 and independent R3 are accepted under O-110. The active policy is
  global FP16 autocast for camera and dense-pillar routes; global FP16 with an
  explicit FP32 island covering SECOND voxelization/VFE/spconv/dense collapse/
  to-BEV for current sparse LiDAR and fusion routes; and uniform FP32 as the
  reference/fallback. Full sparse-convolution FP16 is not accepted as the unified
  fusion-capable route.
- S08 did not explain the large true unscaled LiDAR gradients. Dynamic loss
  scaling does not shrink the gradient applied by the optimizer, and the accepted
  FP32 island avoids sparse-FP16 overflow rather than proving the underlying
  gradient scale healthy. Repeated tiny-group sparse GroupNorm is the leading
  unproven mechanism hypothesis.

S09 STOP-3 Job `446225` and STOP-4D Job `456539` record three initial GradScaler
overflow windows followed respectively by 100 and 1000 accepted F-U updates,
with zero post-warm-up nonfinite or discarded windows. This is bounded engineering
health after scaler backoff; it does not explain or reduce the large true LiDAR
gradient. Closed S09 did not authorize a
normalization, head/loss/target, gradient-clipping, optimizer, scheduler, EMA,
augmentation, sampling, or initialization change. Those training-recipe and
architecture decisions remain owner-gated for S10.

Strict byte-identical determinism is a useful development regression tool, not
the default scientific claim criterion. For scientific claims, record hardware,
precision, seeds, data split, and run manifests; use multi-seed evidence when
results are eventually reported. Mini-data smoke is never scientific evidence.

## Data Status

The licensed shared full nuScenes dataset is now available on Arrhenius through the
dataset module:

```bash
module avail nuScenes
module load nuScenes-data/1.0-map-1.3-zip
echo "$NUSCENES_DATA_DIR"
```

Access is gated by the `arrhpc-dataset-nuscenes` group. A fresh login may be needed
after joining the group; `sg arrhpc-dataset-nuscenes` was used only as a temporary
access verification. The module directory contains trainval metadata, ten stored
`trainvalXX_blobs.zip` archives, and test data. The access check found the camera
samples, `LIDAR_TOP` keyframes, and sweeps needed by the model.

The S01 implementation provides a read-only stored-ZIP backend: a one-time external
SQLite member manifest routes sensor paths to `trainval01` through `trainval10`, and
each DataLoader process lazily opens its own read-only archive descriptors for
offset/CRC-checked reads. Directory mode remains the mini/local backend. Approved
v2 gate job `332651` indexed all ten archives, resolved all 538,695 official
train/val six-camera/key-LiDAR/10-sweep references with zero missing paths, read a
CRC-checked payload sentinel from every archive, and measured deterministic
0/2/4/8-worker loader throughput. Independent S01-R ultimately accepted worker
`abe5c58b174dbbe1f7045ce91c8b15168d97b87b` as **PASS** in review artifact
`7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc` after remediation bound cache identity
to `n_sweeps`, validated local-header member names, fixed exact-archive duplicate
sentinels, and added future in-job source attestation. Cache format `t1.v2` binds
the filename, metadata, every record, and content hash to the requested sweep
depth. Focused GH200 job `333206` passed all 56 dependency-backed real-mini parity,
fork/spawn lifecycle, cache-depth, and ZIP-integrity tests with zero skips.
Historical job `332651` `t1.v1` caches remain coverage evidence only and are
forbidden production inputs; it does not gain retroactive source attestation.
S07-A has migrated `build_gt_database.py` to explicit depth/cache/manifest
provenance. Under O-112, S09 STOP-1 Job `441191` materialized the exact train and
val `t1.v2`, `n_sweeps=10` caches and completed all in-job plus S00 post-job
identity checks. First review returned documentation-only `REMEDIATE`; bounded
re-review of remediation SHA `5252a591983abb0013f19547e1d6ad20d3d6661f`
closed every P2/P3 finding and returned `PASS_WITH_RESIDUAL_RISK`. The exact
caches are owner-accepted under O-113 for exact downstream production binding.
O-114 approves the exact STOP-2 implementation envelope, local validation, linear
immutable implementation/evidence commits, and independent review. O-115 approved
the exact reviewed STOP-2 smoke; Job `441293` completed `0:0` in `00:01:04` with
44/44 tests passing, zero restarts, and no O-107 replacement. Independent evidence
remediation `79f87dc` received `PASS_WITH_RESIDUAL_RISK` with no open P0-P3;
the owner accepted and closed STOP-2 under O-116. O-117's first exact STOP-3
loader/G100 Job `441511` failed before data/model execution because its runner
selected the runtime-only module stack while editable cumm/spconv attempted a
native build. O-118 then authorized one bounded dependency re-attestation plus a
strictly derived no-retry replacement: Job `442152` established stable sparse
build identities, and Job `446225` completed the production loader sweep and 100
successful F-U updates in 103 attempts. Immutable evidence `c28d09c` received
independent `PASS_WITH_RESIDUAL_RISK` with no P0-P2; documentation closure
re-review found no open P0-P3. O-119 then authorized the serial STOP-4A-D
profiler/capacity/output-neutral-optimization/G100/G1000 envelope. Jobs `452520`,
`455539`, and `456539` completed `0:0` without retry using `0.345000` GPU-hours;
final independent review found no open P0-P3. O-120 accepts review seal `ced5992`
and closes S09 PASS. O-121 subsequently fast-forwarded `v3-ad-perception` to
`351b7a0`; no cache retry, worker matrix, model or recipe change, DDP, push, or
S10 execution is authorized by that integration.
Do not extract or duplicate the full dataset into project storage without
explicit owner permission.
The old `/mimer/NOBACKUP/Datasets/NuScenes_v1.0` path is not an Arrhenius data
path.

The module root is discovered from `NUSCENES_DATA_DIR` after explicit config and
the legacy dataroot environment overrides. ZIP runs additionally require
`NUSCENES_ZIP_MANIFEST` (or `ARRHENIUS_NUSCENES_ZIP_MANIFEST`) pointing outside
the shared read-only dataset. Building that manifest on shared trainval is an
exhaustive full-data scan and therefore needs an approved S01 `RUN_REQUEST.md`.
The first ten-archive attempt found a cross-archive repeated path. The successful v2
gate showed that the only repeated path is the same `LICENSE` file in all ten
archives: 2,631,093 total occurrences, 2,631,084 unique members, and nine duplicate
occurrences with matching size/CRC. The reader retains all occurrences, routes
deterministically to the lowest-numbered archive only when size and CRC agree, and
fails on conflicting copies or duplicates within one archive.

The currently accessible mini dataset is:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
```

Use mini for engineering only:

- import/data/cache smoke;
- sparse LiDAR branch correctness;
- precision unification and NaN/Inf checks;
- one-step/tiny-overfit tests;
- profiling and pipeline speed comparisons;
- cleanup validation.

Do not make scientific claims about attack viability, defense behavior, mAP/NDS
quality, ASR, or generalization from mini. Those require trainval-scale data or
a clearly justified fixed trainval subset.

Trainval availability is still not permission to submit a full run; the compute
authorization rules above always apply.

## CL-To-FL Protocols

The active Orchestra distinguishes two protocols. Do not mix their names,
checkpoints, data ownership, or claims.

1. **Protocol A — nuScenes-scratch federated training.** Clients receive the frozen
   architecture and one identical declared initialization, not a detector trained
   on nuScenes. Public ImageNet/NuImages initialization must be distinguished from
   fully random initialization. The matched centralized control uses the same data,
   initialization, and effective exposure. Security claims are blocked if clean FL
   remains a weak detector.
2. **Protocol B — centralized base plus federated tail adaptation.** This is the
   owner-approved primary security setting. A vendor trains `W_base` on common
   `D_base`; regional/fleet data-silo clients receive that model and federatively
   fine-tune on disjoint long-tail `D_tail`. The attack occurs during this
   adaptation stage. Protocol A remains the clean optimization/control setting.

Protocol B must split the official training data at scene/log level. The same scene,
adjacent keyframes/sweeps, duplicated raw files, or the same sensor sample may not
cross `D_base`, `D_tail`, client, validation, or test ownership. Tail criteria and
client assignments are defined from train-only information, frozen and hashed
before attack experiments, and cannot be selected to improve ASR. Official
validation/test data remain held out.

The full-train CL capability checkpoint is not a valid Protocol-B initializer if it
has seen `D_tail`. After the architecture is frozen, retrain it on `D_base` to
produce the scientific `W_base`. Required clean controls include `W_base`, a
centralized pooled-tail oracle, local-only fine-tuning, and clean federated
fine-tuning. Report common-data retention, tail improvement, catastrophic
forgetting, client dispersion, and compute/communication cost before attack or
defense claims.

Use “federated training” for Protocol A and “federated fine-tuning/adaptation” for
Protocol B. The client system unit should be stated explicitly; the recommended
realistic unit is a regional/fleet data silo rather than an unsupported claim that
every car trains the full detector onboard. Full details and open owner decisions
are in the active Orchestra documents.

## Scientific Guardrails

When designing or reviewing experiments, prioritize silent scientific failure
modes over style:

- data leakage and train/val/test contamination;
- client partition mistakes or non-comparable partitions across cells;
- coordinate-frame, yaw, class-map, unit, or calibration errors;
- sparse LiDAR tensor/index/voxel semantics and empty-input edge cases;
- precision mismatches that change training dynamics;
- ASR denominator/eligibility mistakes;
- comparing attack/defense cells with different clean baselines, seeds,
  participation regimes, or precision settings;
- using a CL checkpoint that has already seen data assigned to FL clients;
- splitting raw nuScenes ownership by annotation while frames/scenes/sweeps leak
  across base/client/eval sets;
- defining long-tail clients or target conditions after observing ASR or validation
  outcomes;
- assigning every tail example to clients without a scene/log-disjoint held-out
  tail evaluation or a predeclared official-val tail slice;
- interpreting a defense that prevents benign tail learning as successful;
- claims resting on mini or smoke runs;
- defense evaluations where the corresponding undefended attack is not viable.

For defense carry-overs from `fl_v2`, oracle parity means implementation
equivalence only. It does not certify AD-domain validity.

## Working Style

Before editing:

- inspect relevant source, docs, configs, and `git status`;
- identify whether you are in the main worktree, a temporary Codex worktree, or
  an Arrhenius bring-up worktree;
- avoid touching unrelated dirty files;
- avoid reverting user/Claude changes unless explicitly asked.

When editing:

- keep changes scoped and reviewable;
- preserve existing project style;
- prefer structured config/data APIs over ad hoc parsing;
- update docs/scripts together when behavior changes;
- keep Arrhenius paths configurable via env/config where possible.

After editing:

- run the smallest meaningful verification available;
- for shell scripts, at least `bash -n`;
- for Python touched by the change, at least `py_compile` or focused tests;
- if Slurm/GPU/data prevents verification, say exactly what was not verified.

## Orchestra Milestone Delivery And Review

Owner decision O-094 makes persistent S00 the default implementation and
coordination context. `Sxx` names a durable evidence milestone; it does not
require a fresh task, branch, worker, reviewer, or worktree. Only S00 edits
`ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`.

For milestones other than S10, retain the phase-sized handoff, run-request,
results and review conventions already accepted for that milestone. Scope
approval does not imply compute, commit, merge, push, upload or publication
authority.

### S10 simplified evidence and review contract (O-143)

S10 uses two active records under `fl_v3/usenix27_orchestra/handoffs/S10/`:

- `HANDOFF.md`: one compact current status, active scientific plan, accepted
  facts, decision boundaries and interpretation limits;
- `RUN_REQUEST.md`: one append-only job ledger with phase approval, provenance,
  resources, terminal state and artifact locations.

Existing `RESULTS.md` and `REVIEW.md` remain historical evidence archives; do
not replicate each new job across all four files. Update the three canonical
Orchestra documents only at phase start, material scientific amendment, or phase
closure. Do not create a documentation commit for every failed fixture, retry or
minor diagnosis. Commit at material implementation boundaries, phase-plan freeze,
and phase-result closure.

Preflight science jobs with direct entry/config/checkpoint/one-batch checks. Do
not put broad historical test suites, jackknife/report generation or recursive
manifest construction on the critical GPU-training path unless they are necessary
for the active scientific gate. Necessary CPU post-processing may run separately
and cannot retroactively turn a completed training/evaluator result into a model
failure; missing required statistics must still be reported as missing.

Independent review is required for a data-ownership or evaluator/metric change,
a branch recipe freeze, and the final staged-fusion/full capability result. It is
optional for ordinary runner bugs and intermediate observations. When review is
used, pin a durable SHA and raw evidence; reviewers report and do not fix code.

S00 must obtain explicit owner approval before changing a material scientific
item: Protocol A/B roles, data ownership or split, model/head, recipe candidate
space, metric/evaluator, experiment seeds or cells, acceptance gates, aggregate
resources, or publication/upload scope. It may not hide failures, reinterpret old
evidence, or weaken a gate retroactively.

## Worktree Provisioning

The default is one persistent S00 worktree and linear branch for the active,
tightly connected track. A new worktree is justified only by genuine parallel
ownership, risky experimental isolation, a high-risk independent review, exact
runtime reproduction, conflicting state, or explicit owner direction. This avoids
repeated onboarding and uncontrolled worktree growth.

When another task/worktree is used, its envelope pins `BASE_SHA`, `SOURCE_BRANCH`,
`EXPECTED_REF_MODE`, file ownership, and upstream evidence. The new task verifies
`git status --short`, `git rev-parse HEAD`, `git branch --show-current`, and
`git rev-parse --show-toplevel` before acting; a mismatch is a blocker, not a
reason to repair Git topology autonomously. No agent runs `git worktree add`,
`move`, `remove`, or `prune`, switches branches, or deletes a branch/worktree
without exact owner authorization.

Independent review always pins a durable implementation/evidence SHA, even when
performed by a reviewer subagent in the persistent context. Mutable uncommitted
state is not a review baseline. A review-only artifact may be sealed linearly; no
reviewer merge history is required. Commit, merge, push, and publication remain
separate permissions.

## Git And Branches

The active integration branch is `v3-ad-perception`. Create scoped `codex/...`
branches for independent Codex work unless the user asks to work directly on
the current branch. Commit, merge, or push only when the user explicitly asks.

Temporary bring-up branches/worktrees may be deleted after they are merged and
the user approves cleanup. Deleting a branch name must not remove persistent
environment/data artifacts under `/nobackup`.

## Review Mode

If the user asks for a review, switch to a code/science review stance:

- findings first, ordered by severity;
- cite exact files/lines;
- focus on correctness, scientific validity, metrics, data, precision,
  reproducibility, and missing tests;
- keep summaries secondary;
- state clearly when no issues are found and what residual risk remains.

The older `fl_v3/collab/codex_review_prompt.md` is historical evidence only. Active
worker/reviewer prompts and review requirements are in
`fl_v3/usenix27_orchestra/KICKOFFS.md` and `SESSIONS.md`.
