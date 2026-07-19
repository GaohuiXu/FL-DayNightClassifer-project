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
```

This is milestone/work orchestration, not a research-cycle document. Names beginning
with `cycle_*` remain reserved for the project's experimental-design cycles under
`fl_v3/docs/cycle_04/` and `fl_v3/docs/roadmap/`. Do not create another
`cycle*_orchestra` folder.

S07 clean engineering, S08 precision qualification, and S09 full-pipeline
engineering performance/readiness are closed. The accepted S08 policy is
integrated at `28f79802c0868afa6290d74ae6aeb9d23c7d088f`; owner decision O-120
accepts S09 review seal `ced5992ea113bd21d7d545af505debf405b556b3` as a
bounded engineering PASS. Owner decision O-121 fast-forwarded
`v3-ad-perception` to S09 closing commit
`351b7a0b8419c01d0d32ba224babbc6bdc4213ba`; the former delivery branch has the
same tip/tree but is retained pending a separate cleanup decision. Fresh
persistent S00 completed the S10 startup audit at exact clean base
`a080d49c1c22de20ccb5b1353d4922c7df14a729` and created
`codex/s10-cl-model-recipe`. Owner decision O-122 accepts S10's six-stop A-F
scientific envelope, the exact STOP-A split/evaluator gate, and the primary full
claim “absolute clean capability + fusion contribution”. It authorizes planning
records only under O-122. O-124 now authorizes bounded STOP-A/B/C implementation,
linear commits, exact derived serial Slurm execution and stop-level review within
27 cumulative one-GH200 hours. O-123 rejects the B=1-based v0 estimate and
requires physical B=4 as the minimum ABC scientific-training microbatch; B=1 is diagnostic-only and
B=8/16 remain later STOP-D/E candidates. S10 STOP-A closed
`PASS_WITH_RESIDUAL_RISK` after Job `468404` and independent A4 re-review of
remediation SHA `b0478a298a0a3b5e538bedcca63e2541d71c2146`, with no open P0-P3.
STOP-B Job `477892` exposed an under-specified parity failure. O-129 replacement
Job `478250` then passed 41 tests/identities and durably classified the first
fixed-seed FP32 repeat as baseline instability, but could not enter gradient
localization; independent review returned `PASS_WITH_RESIDUAL_RISK` with no open
P0-P3. O-130 authorizes one bounded B-RAND amendment: decompose intended camera
stochasticity from runtime variation across C-STR8, L-S075 and F-U on the same
frozen B4 token panel, with no update/evaluator, one GH200, `00:15:00`, and at
most `0.25` GH200-hour. Exact implementation `0bf9c0c` and its detached,
read-only snapshot/command tuple are frozen in S10 `RUN_REQUEST.md` §24 and are
the sole B-RAND submission. Job `479667` consumed it, completed `0:0` in
`00:07:08`, passed 43 tests and the 33-run integrity gate, and returned the
descriptive label `MIXED_INCONCLUSIVE` with both camera stochasticity and LiDAR
runtime variation qualified. Independent review and targeted remediation
re-review at `02ba3b44202092894f2c1c3e7ee53bb56ba92a1d` returned
`PASS_WITH_RESIDUAL_RISK` with no open P0-P3. STOP-B is CLOSED /
`INCONCLUSIVE`: bounded route-level repeatability decomposition is accepted,
but large-gradient causality remains unresolved. No retry, further STOP-B
compute, or model/recipe change is authorized. O-131's sole integrated STOP-C0
Job `492525` consumed `0.792222` GH200-hours and ended `FAILED 1:0` at
`00:47:32`. A1 fusion and random L-S075 each completed one physical-B4 `D_low`
epoch and exact `D_select` internal evaluation, but the 64-window all-scratch
control has no accepted summary because the runner incorrectly required it to
exhaust the full epoch iterator. Both complete cells' raw `HARD_FAIL` labels are
also false positives from requiring gradients on `lidar_encoder.to_bev`, which
is `nn.Identity` for SECOND-075. Excluding that impossible condition, both have
falling loss, zero post-first-64 invalid windows and no correlated large-gradient
harm signal; the single-seed/internal F-minus-L delta is positive but is not a
recipe, architecture, official-val or full fusion claim. STOP-C0 is
`FAIL/INCOMPLETE` and its no-retry allocation is consumed. Independent targeted
re-review of remediation `09c39458a0b32ce1d4a3ae603094d76ae160ac42` returned
`PASS_WITH_RESIDUAL_RISK` with no open P0-P3 for evidence integrity only. It
also invalidated the three raw v1 dropped-token identities (the count and matched
F/L construction remain valid); raw output was not rewritten. Later STOP-C
strong contrasts and STOP-D/E/F remain owner-gated. O-132's sole C0-v2 clean
replay Job `496312` consumed `0.754167` GH200-hours and completed `0:0` in
`00:45:15` from exact source
`2262b4063a3e419b17f4b911a9e11a7ff50ea784`. It passed 80 focused tests with
3 skips, all three v2 cell summaries, actual-token matched F/L order/remainder,
trainable-prefix health, aggregate `PASS` and the 28/28 artifact manifest. All
three cells have falling loss, only four initial overflow windows, zero later
invalid/nonfinite/discarded windows and no correlated large-gradient harm signal.
The one-seed internal F-minus-L delta is positive (`+0.029576` mAP / `+0.033423`
NDS) but does not select a recipe/architecture or establish the full claim.
Large-gradient causality remains unknown. O-132 is consumed; no replay, automatic
later-C continuation or intermediate reviewer chain is authorized.
Owner decision O-133 accepted C1-A/C1-B planning: a frozen-panel FP32
GN-versus-direct-BN1d causal comparison with normal loss and frozen SECOND-output
VJP; one common conservative no-update-qualified GradScaler init scale for C1
scientific training; and current-A1/current-staged-A2/coherent-MIT-A2 through the
bounded `D_low -> D_mid` funnel with at most two STOP-C survivors. The exact
coherent MIT anchor graph/init/component package remains owner-pending. O-134
relaxes the prior conditional restriction on BN1d/TransFusion/LiDAR-conditioned
DepthLSS but orders C1-B to run the current graph's A1/A2 first; MIT-reference
repair opens only if that matched result is materially worse under a future exact
gate. O-134 authorizes C1-A implementation, linear commits and exactly one frozen
L-S075 FP32 no-update/no-evaluator diagnostic on the complete accepted 16xB4
STOP-B panel: current GN versus direct BN1d, normal loss plus coordinate-fixed
SECOND-output VJP, two repeats, 128 runs total, one GH200/8 CPU/64 GiB/`00:30:00`,
at most `0.5` GH200-hour, no retry. Exact Job `502456` passed 36 focused tests
but failed before either candidate forward/backward because its fail-closed
mapping check incorrectly expected PyTorch to report BN1d
`num_batches_tracked` as a missing key; PyTorch backward-compatibly synthesizes
that buffer and reported only `running_mean/running_var`. The job consumed
`0.050833` GH200-hour and produced no gradient verdict. O-134 is consumed; no
retry is executable. C1-B implementation/compute remains pending.
Owner instruction O-135 authorizes correction of that diagnosed assertion only.
The remediation accepts only reported `running_mean/running_var`
missing keys, separately proves every synthesized `num_batches_tracked` exists
and is zero, and adds a direct PyTorch regression test. It is committed at exact
source `d713bfe3b5e5c587f58ce70721b2b6eea0b050ec`; the failed evidence remains
immutable. Owner instruction O-136 authorizes that exact commit and one strictly
derived C1-A replacement with unchanged data, panel, candidates, FP32 pathways,
two-repeat 128-run scope, gates and one-GH200/8-CPU/64-GiB/`00:30:00` resource
ceiling. Exact Job `502572` consumed that authority and completed `0:0` in
`00:03:09` (`0.052500` GH200-hour). It passed 37 focused tests, all identities,
128/128 finite no-update runs, exact candidate parameter parity/immutability and
the artifact gate, returning bounded verdict `LOCALIZED_NORM`. On every one of
the 16 B4 batches, direct-reference BN1d reduced both fixed-VJP and normal-loss
stem gradients far beyond two-repeat runtime variation; occupancy correlations
did not qualify and head/loss amplification did not meet its correlation gate.
This causally localizes the current W0/panel large-gradient mechanism to the tiny-
group GN path within C1-A, but it does not promote BN1d, prove convergence or
select an architecture/recipe. O-136 is consumed. No retry, scope change,
automatic C1-B continuation or later-stop execution is authorized.
Owner decision O-137 activates only C1-B0, the current-A1 matched fusion-health
observation rung. It authorizes a production-resolved SECOND normalization and
checkpoint identity seam whose default remains GN, plus two serial seed-0
physical-B4 cells (`F-A1-GN-H256`, `F-A1-BN1D-H256`) over one frozen 1024-token
`D_low` vector. Both use exact shared trainable W0, ImageNet1K V1 camera,
global FP16 with the SECOND FP32 island, AdamW `1e-4/0.01`, constant schedule,
no augmentation/EMA/CBGS/GT-paste/clip, common no-update-qualified GradScaler
scale 32, and 256 real updates. It records loss, true-unscaled sampled gradients,
realized updates, BN state, memory and basic timing without evaluator or
checkpoint selection. One GH200/16 CPU/96 GiB/`00:30:00` job is authorized,
hard-capped at `0.5` GH200-hour, with no retry. C1-B0 is observation-only: it
cannot promote BN1d or start C1-B1/full `D_low`, A2, MIT repair, STOP-D/E/F,
reviewer chain, merge, push or upload.
O-137's sole Job `502958` is consumed and failed pre-model after `00:02:14`
(`0.037222` GH200-hour): 100 focused tests passed and 6 failed. Five failures
come from three pre-existing operator-profile hash-check lines being mechanically
left inside the new parameterized rejection test; the sixth C1-B0 fixture changed
to `s10.v1` without adding its required scale-32 field. No experiment model,
H256 batch, optimizer or scientific cell executed. This is a test-fixture/layout
failure, not GN/BN1d training evidence. No retry or autonomous remediation is
authorized; C1-B0 returns to the owner.
Owner decision O-138 authorizes only the exact test-neutral correction diagnosed
from Job `502958`: move the three operator-profile hash assertions back into their
original S09-v2 test and add required `grad_scaler_init_scale=32` to the migrated
`s10.v1` test fixture. It authorizes local/static validation, one linear
remediation commit, one new detached read-only snapshot and one strictly derived
C1-B0 replacement with unchanged production source, runner, config, data, cells,
seed, W0, H256 horizon, gates and O-137 resources. The replacement has one
submission, no retry and no C1-B1 or later-stop authority.
O-138 replacement Job `503075` consumed that authority and failed pre-model after
`00:02:11` (`0.036389` GH200-hour): 105 focused tests passed and one failed.
The migrated `s10.v1` fixture also lacked required
`execution.operator_profile`; the earlier missing scale field had masked this
second schema omission. No model, H256 data, optimizer or cell executed. O-138
is consumed; no further correction, retry or C1-B1 execution is authorized.
Owner decision O-139 authorizes replacing the partial S09-to-S10 test mutation
with the already validated canonical `s10_second_config` fixture, explicit schema/
normalization/scale/operator-profile propagation assertions, local/static audit,
one linear commit, one detached read-only snapshot and one strictly derived
C1-B0 replacement. Production source, runner, config, data, cells, W0/seed,
horizon, gates and O-137 resources remain unchanged. One submission, no retry;
no C1-B1 or later-stop authority.
S11 and later milestones remain pending. Historical conclusions under
`fl_v3/collab/model_capability/` remain evidence, but the active Orchestra
documents supersede them where the architecture audit or current data/runtime
state changed.

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
  explicitly owner-approved bounded mechanical remediation loop described below;
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
these invalidates the approval and requires new permission. The only exception is
a derivation rule that the owner explicitly approved in the initial request under
O-107. Never infer full-run or upload authorization from approval of an
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
coordination context. S00 may directly plan, implement, validate, document, and
integrate one or more tightly connected milestones in one long-lived worktree.
`Sxx` names a durable evidence milestone and handoff namespace; it does not by
itself require a fresh task, worker, branch, or worktree.

Only S00 edits `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`. For a newly scoped
milestone, S00 first presents the exact plan, file ownership, scientific non-goals,
verification/review plan, and any proposed compute tuple to the owner. Scope
approval does not imply compute, commit, merge, push, upload, or publication
authority. The durable package lives under:

```text
fl_v3/usenix27_orchestra/handoffs/Sxx/
```

Required files are:

- `HANDOFF.md` for every active milestone: exact base/branch/commit, files and semantic
  changes, references, tests/jobs/raw outputs, gate evidence, hashes, negative
  results, allowed/forbidden scientific interpretations, and unresolved risks;
- `RUN_REQUEST.md` before any material compute, with exact immutable execution
  scope and explicit approval state;
- `RESULTS.md` for execution milestones, including every requested/failed/missing
  cell, job ID, raw artifact path/checksum, metrics, performance, and interpretation
  limits;
- `REVIEW.md` from an independent reviewer subagent or, when risk requires, a
  separate review worktree, with severity-ordered findings, adversarial checks,
  gate verdict, and residual risk.

S00's self-review is not an integration or scientific PASS. After an immutable
implementation/evidence SHA exists, an independent reviewer reads that exact diff,
resolved config, data/split manifest, logs, and raw artifacts and does not fix
code. Use a separate review worktree for high-risk data ownership, metric or
scientific-result changes, conflicting concurrent state, exact runtime
reproduction, or owner request. Review explicitly covers leakage, coordinate/
calibration/units, batch invariance, branch/config resolution, optimizer steps and
exposure, precision/resume, metric/ASR denominators, failed or omitted cells, and
shortcuts that could inflate clean performance, fusion gain, ASR, or defense
success. Owner monitoring during execution does not replace review.

After a milestone handoff is complete, S00 prepares the exact independent review
envelope. A reviewer subagent is the default; a UI-created `Sxx-R` worktree is an
isolation mechanism rather than a mandatory lifecycle step. Findings are remediated
linearly by S00 and re-reviewed. S00 may change scheduling, dependencies, required
reading, evidence requests, review focus, and wording that does not alter the
approved scientific protocol. It records the evidence, affected milestones, and
exact change in the Orchestra ledger.

S00 must obtain explicit owner approval before changing a locked or material
scientific item: Protocol A/B roles, data ownership or split, model/head/metric,
threat model, experiment cells or seeds, acceptance gates, resource scope, compute
authorization, or publication/upload scope. It may not silently rescope an active
milestone, retroactively weaken a gate, hide a failed/negative result, or
reinterpret old evidence under a new protocol. A material active-milestone
amendment must be recorded, acknowledged by the owner, and re-reviewed when it
changes delivered semantics.

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
