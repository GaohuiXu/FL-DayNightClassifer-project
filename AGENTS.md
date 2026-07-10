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

This is session/work orchestration, not a research-cycle document. Names beginning
with `cycle_*` remain reserved for the project's experimental-design cycles under
`fl_v3/docs/cycle_04/` and `fl_v3/docs/roadmap/`. Do not create another
`cycle*_orchestra` folder.

The current priority is to repair, validate, and freeze a strong centralized (CL)
camera-LiDAR detector before final FL attack/defense claims. Historical conclusions
under `fl_v3/collab/model_capability/` remain evidence, but the active Orchestra
documents supersede them where the architecture audit or current data/runtime state
changed.

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

- submit `sbatch`/`srun` jobs, including engineering smoke jobs;
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
local/static/unit checks. An execution approval is bound to the exact commit,
resolved config, data/split manifest, cells, seeds, command, GPU/count/time budget,
and output location stated in the request. Changing any of these invalidates the
approval and requires new permission. Never infer full-run or upload authorization
from approval of an architecture, plan, session, or code change.

Every material-compute session records its request and approval state in
`fl_v3/usenix27_orchestra/handoffs/Sxx/RUN_REQUEST.md`. Preparing or editing
that file does not grant approval.

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
- Supported sparse-path precisions are `fp32` and `fp16` AMP with GradScaler.

Near-term precision work should make this policy explicit in configs, trainers,
manifests, and Slurm launchers. Do not mix precision regimes in one comparison
without labeling it as an ablation.

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

The production loader is not yet ZIP-aware: it still expects ordinary filesystem
paths for `Image.open` and `np.fromfile`. Do not describe full-data training as ready
until the ZIP backend, complete member-coverage manifest, directory/ZIP parity, and
multi-worker behavior pass the S01 gates. Do not extract or duplicate the full
dataset into project storage without explicit owner permission. The old
`/mimer/NOBACKUP/Datasets/NuScenes_v1.0` path is not an Arrhenius data path.

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

## Orchestra Session Delivery And Review

The current plan defines worker sessions `S01` through `S15`; `S00` is the
Orchestra/owner coordination role. A fresh dedicated Orchestra session is preferred
when the current conversation is long. It must read the three active canonical files,
inspect handoffs/reviews and actual diffs/artifacts, update the status ledger, and
avoid opportunistically implementing worker tasks itself.

Only the Orchestra session edits `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`.
Each worker uses an isolated worktree/ref, stays inside declared file ownership,
and writes a durable package under:

```text
fl_v3/usenix27_orchestra/handoffs/Sxx/
```

Required files are:

- `HANDOFF.md` for every session: exact base/branch/commit, files and semantic
  changes, references, tests/jobs/raw outputs, gate evidence, hashes, negative
  results, allowed/forbidden scientific interpretations, and unresolved risks;
- `RUN_REQUEST.md` before any material compute, with exact immutable execution
  scope and explicit approval state;
- `RESULTS.md` for execution sessions, including every requested/failed/missing
  cell, job ID, raw artifact path/checksum, metrics, performance, and interpretation
  limits;
- `REVIEW.md` from a separate review session for implementation/scientific work,
  with severity-ordered findings, adversarial checks, gate verdict, and residual
  risk.

A worker's self-reported PASS is not an integration or scientific PASS. The
Orchestra requires independent review and checks the actual diff, resolved config,
data/split manifest, logs, and raw artifacts. Review explicitly covers leakage,
coordinate/calibration/units, batch invariance, branch/config resolution, optimizer
steps and exposure, precision/resume, metric/ASR denominators, failed or omitted
cells, and shortcuts that could inflate clean performance, fusion gain, ASR, or
defense success. Owner monitoring during execution does not replace this review.

After a worker handoff is complete, S00 may inspect it immediately, prepare the
exact independent `Sxx-R` review envelope for the owner to open in the task UI, and
use reviewed evidence to refine the plan and kickoff for any session that has not
started. S00 may change scheduling, dependencies, required reading, evidence
requests, review focus, and wording that does not alter the approved scientific
protocol. It must record the evidence, affected sessions, and exact change in the
Orchestra change-control ledger.

S00 must obtain explicit owner approval before changing a locked or material
scientific item: Protocol A/B roles, data ownership or split, model/head/metric,
threat model, experiment cells or seeds, acceptance gates, resource scope, compute
authorization, or publication/upload scope. It may not silently rescope an active
session, retroactively weaken a gate, hide a failed/negative result, or reinterpret
old evidence under a new protocol. An active-session amendment must be recorded and
acknowledged by that worker.

## Worktree Provisioning

The default workflow is that the owner selects `Worktree` and the pinned starting
branch in the Codex task-creation UI before sending the kickoff. Codex-managed
worktrees normally start detached at that branch's HEAD; detached HEAD is expected,
not a blocker, when its SHA matches the kickoff. A worker or reviewer must not run
`git worktree add`, `move`, `remove`, or `prune`, switch to another branch, or delete
a branch/worktree unless the owner explicitly authorizes that exact Git operation.

Every kickoff must pin `BASE_SHA`, `SOURCE_BRANCH`, `EXPECTED_REF_MODE` (normally
`detached@BASE_SHA`), file ownership, and upstream handoffs. The fresh session
verifies `git status --short`, `git rev-parse HEAD`, `git branch --show-current`, and
`git rev-parse --show-toplevel` before editing; a mismatch is a blocker, not a
reason to repair Git topology autonomously. Parallel workers start from the same
approved integration SHA. A reviewer uses a distinct UI-created review worktree
based on the exact worker SHA/diff; S07 uses a dedicated integration worktree.

An independent review worktree can reproduce only a durable Git version. After a
worker finishes and S00 checks handoff completeness, the owner must explicitly
authorize any local handoff commit/branch needed to expose that exact version to
`Sxx-R`. This permission does not authorize merge, push, or publication. The
reviewer records `WORKER_SHA`; uncommitted cross-worktree state is not a review
baseline.

Canonical Orchestra documents must already be committed on the pinned source
branch before worker/reviewer worktrees are created. Codex can copy selected local
changes into a managed worktree, but that has no immutable base SHA and is therefore
not used for the reproducible multi-session wave.

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
