# fl_v3 Environment on Arrhenius GH200

Arrhenius is the active runtime target for `fl_v3`. The Python/CUDA/spconv
environment is a long-lived conda prefix on project storage, not a Git artifact
and not something each Slurm job recreates.

## Persistent Environment

Validated environment root:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3
```

Conda prefix:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/envs/pt311-cu128-spconv
```

Source-build trees:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/cumm
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/spconv
```

This directory also owns pip/conda caches, Torch weights, temporary build
artifacts, info-cache outputs, and smoke logs. These paths intentionally live
under `/nobackup`, not under `$HOME`.

## Rebuild Only When Needed

Build or fully recreate the environment on a GH200 node:

```bash
cd /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project__arrhenius-env-bringup
sbatch --export=ALL,RECREATE=1 fl_v3/scripts/run_arrhenius_env_build.sh
```

Incremental/idempotent rebuild:

```bash
sbatch fl_v3/scripts/run_arrhenius_env_build.sh
```

Do not run this for every training job. Normal jobs only load modules and
activate the existing prefix.

## Activate in Slurm Jobs

Every Arrhenius Slurm launcher should use the shared bootstrap:

```bash
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env
```

`arrhenius_load_modules build` is used even for smoke/runtime jobs because
source-built cumm/spconv can still trigger `ccimport`/ninja checks on import and
may need `nvcc`.

The login node is `x86_64`; GH200 compute nodes are `aarch64`. Do not treat a
login-node import failure of the aarch64 conda prefix as an environment failure.
Submit a Slurm job to validate this stack.

## Validated Stack

- Python `3.11.15`.
- PyTorch `2.11.0+cu128`, CUDA wheel runtime `12.8`.
- CUDA compiler module `12.9.1`.
- torchvision `0.26.0+cu128`.
- numpy `1.26.4`, scipy `1.13.1`.
- flwr `1.27.0`, ray `2.51.1`.
- nuscenes-devkit `1.1.11`, scikit-learn `1.8.0`. The devkit imports
  `sklearn.metrics` unconditionally, so scikit-learn is a clean nuScenes
  data/evaluation runtime dependency. It is not an FLAME, HDBSCAN, or defense
  dependency.
- cumm `v0.7.13` at `4dedaf43ff801e417c60c6bd7536a29d83d29ee0`.
- spconv `v2.3.8` at `263d6b47425ef843c82f997b12d8b714013d216c`.

The editable source checkouts are not described by HEAD alone. The accepted
current cumm checkout has no tracked changes, with canonical tracked-state SHA
`f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662`.
The accepted current spconv checkout has exactly one unstaged tracked build-
metadata change: `" M" pyproject.toml`, file SHA
`e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9`,
and canonical tracked-state SHA
`499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db`.
It removes `cumm>=0.7.11` from `[build-system].requires`; it does not change
Python/CUDA/C++ executable source. This exact state already appeared in accepted
S07-B-COMPLETE evidence and must be explicitly bound by current SECOND configs.
Launchers also verify source HEAD, import origin, and installed executable-build
hashes before model construction. Any other tracked status/path/content or build
identity fails closed. Do not reset or edit either external checkout merely to
satisfy a blanket clean-tree check.

The source build is pinned to `sm_90`:

```bash
TORCH_CUDA_ARCH_LIST=9.0
CUMM_CUDA_ARCH_LIST=9.0
```

Precision status:

- Runtime mechanisms available on Arrhenius: `fp32` and CUDA `fp16` autocast with
  GradScaler. Availability is not a model-level stability or scientific-policy
  claim.
- Direct `torch.bfloat16` sparse convolution is not supported by this
  cumm/spconv path; `bf16` configs should fail loudly instead of falling back.
- Trainer, smoke, and provenance paths use the explicit `precision` policy
  rather than inferring AMP dtype from cuDNN deterministic flags.
- S08 Q1/Q2 and independent R3 are accepted under O-110. The active model policy
  is global FP16 autocast for camera/dense-pillar; global FP16 with an explicit
  FP32 island covering SECOND voxelization/VFE/spconv/dense-collapse/to-BEV for
  sparse LiDAR/fusion; and uniform FP32 as reference/fallback. Full sparse FP16 is
  not the accepted unified fusion route. The unusually large true unscaled SECOND
  gradients remain an unresolved model/recipe risk rather than an environment
  failure.
- The checked-in S07 C/L/F JSON files are template-only. Their `precision` fields
  do not freeze the later training regime.

## Smoke Tests

Non-data smoke:

```bash
sbatch fl_v3/scripts/run_arrhenius_smoke.sh
```

Real nuScenes smoke once a dataroot and info-cache are available:

```bash
sbatch --export=ALL,\
ARRHENIUS_NUSCENES_DATAROOT=/path/to/NuScenes_v1.0,\
ARRHENIUS_NUSCENES_CACHE=/path/to/info_cache,\
REQUIRE_DATA=1,\
SMOKE_MODES='import spconv data eval train' \
fl_v3/scripts/run_arrhenius_smoke.sh
```

For mini-only partition smoke, lower the keyframe floor:

```bash
sbatch --export=ALL,\
ARRHENIUS_NUSCENES_DATAROOT=/path/to/nuscenes_mini,\
ARRHENIUS_NUSCENES_CACHE=/path/to/info_cache_mini,\
REQUIRE_DATA=1,\
SMOKE_MODES='data eval train',\
PRECISION=fp32,\
MIN_KEYFRAMES_PER_CLIENT=10 \
fl_v3/scripts/run_arrhenius_smoke.sh
```

## nuScenes Data

The licensed shared full dataset is now exposed by the Arrhenius module:

```bash
module load nuScenes-data/1.0-map-1.3-zip
echo "$NUSCENES_DATA_DIR"
```

It contains trainval metadata and stored ZIP blob archives. The current production
data path now has the S01 read-only backend: the ten trainval central directories
are indexed once into an external SQLite manifest, and workers use PID-owned lazy
read-only descriptors plus offset/length/CRC checks rather than extracting blobs or
rebuilding `ZipFile` indexes per worker. Directory mode remains unchanged for mini.

This follows the lifecycle in C3SE's
[official raw-ZIP DataLoader example](https://www.c3se.chalmers.se/documentation/software/machine_learning/datasets/#raw-files):
do not reuse the constructor's archive handle in workers, open lazily in the worker,
decode images through `BytesIO`, and reopen once after a bad handle. S01 records
stored-member offsets once and uses `pread`+CRC at runtime, avoiding a full Python
`ZipFile` member dictionary in every worker.

Approved v2 full gate `332651` completed the exhaustive shared manifest scan, 100%
train/val path-coverage audit, ten-archive payload sentinels, and deterministic
0/2/4/8-worker loader profile. This is an engineering data-path result, not model or
scientific readiness. Independent S01-R requested changes. The remediation
binds every cache to its exact `n_sweeps` under cache format `t1.v2`,
validates the local-header member name, reads duplicate sentinels from their exact
archive occurrence, and makes future launchers attest the Git/source state inside
the job. Job `332651`'s `t1.v1` caches remain valid historical coverage evidence
but must not be consumed by the remediated loader. Dependency-backed real-mini
directory/ZIP decoded parity and fork/spawn lifecycle execution passed in focused
GH200 job `333206` (`56 passed`, zero skipped). S01-R then returned **PASS** for
worker `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`; the separate review artifact is
`7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc`. S07-A migrated the permission-out GT
database caller to explicit `t1.v2` depth/cache/manifest validation. O-112 S09
STOP-1 Job `441191` subsequently materialized exact read-only train/val `t1.v2`,
`n_sweeps=10` caches and passed the execution/cache/hash gates. Bounded independent
re-review at remediation SHA `5252a591983abb0013f19547e1d6ad20d3d6661f`
closed the documentation-provenance findings and returned
`PASS_WITH_RESIDUAL_RISK`; O-113 owner-accepts the exact caches for downstream
production binding. O-114 separately approves the exact STOP-2 implementation,
local validation, linear immutable commits, and independent review. Candidate
`37aef4d6b3f4679d6702d0acef2bb5bd1b57a952` received independent
`PASS_WITH_RESIDUAL_RISK` with no open P0-P2; its exact bounded GH200 smoke is
frozen, and request remediation `cad7262` independently closed all P2/P3 request/
snapshot findings. O-115 Job `441293` then completed `0:0` in `00:01:04` with
44/44 tests passing, zero restarts and no replacement. Evidence remediation
`79f87dc` subsequently received independent `PASS_WITH_RESIDUAL_RISK` with no
open P0-P3; O-116 owner-accepts/closes STOP-2. O-117 STOP-3 Job `441511` then
exposed why this section's build-module rule is binding: the runner selected
`arrhenius_load_modules run`, editable spconv triggered a JIT check without CUDA
headers, and compilation failed on missing `cublasLt.h` before data/loader/model
execution. O-118 authorized one exact dependency re-attestation and one strictly
derived replacement without retry. Job `442152` completed stable two-process
attestation of spconv build `af422005...` and cumm build `0a7e3c1a...`; Job
`446225` subsequently failed closed on those identities, completed the production
loader sweep, and reached 100 successful F-U updates in 103 attempts. Evidence
`c28d09c` received independent `PASS_WITH_RESIDUAL_RISK` with no P0-P2, and
documentation closure re-review found no open P0-P3. STOP-3 is owner-ready but
owner acceptance remains pending. O-118 compute is consumed and no STOP-4
compute is authorized. This is engineering readiness
for one exact tuple, not convergence, metric, recipe, or model-capability
evidence. Do not extract/duplicate the dataset or submit further full-data jobs
without exact permission. O-009 covers only a recorded bounded
engineering smoke (one node/GPU, at most 60 minutes/job, one concurrent job, two
cumulative GPU-hours); it did not authorize Job `441191` and does not authorize an
additional cache materialization, full-data coverage/profile, model steps,
metrics, matrices, reruns, arrays, DDP, or retries.

A bounded one-archive GH200 engineering smoke (`Slurm 330409`, 2026-07-10) passed:
module/table discovery, four real samples with all six cameras plus keyframe and
nine previous LiDAR sweeps, CRC-checked decode, and deterministic 0-worker versus
two persistent-worker reads across two epochs. It covered only
`trainval01_blobs.zip` (258,109 members indexed; 64 selected members read), so it is
not the missing ten-archive coverage or throughput gate. Exact artifacts and limits
are recorded in `usenix27_orchestra/handoffs/S01/{RESULTS,HANDOFF}.md`.

The first approved ten-archive gate (`Slurm 332648`, 2026-07-11) stopped during
`trainval02` manifest construction because the v1 schema rejected a path repeated
across archives. This is a negative result, not evidence of corrupt payloads. The
v2 schema records all occurrences, permits only cross-archive copies with matching
size+CRC, routes to the lowest archive deterministically, and still rejects
conflicting copies and within-archive duplicates. The follow-up below validates
that correction against the complete shared trainval archive set.

The exact v2 follow-up (`Slurm 332651`, commit `1fe651700bd0`, 2026-07-11)
completed in `00:05:29`:

- 2,631,093 member occurrences and 2,631,084 unique members across 417,774,430,886
  archive bytes; the nine duplicate occurrences are the identical `LICENSE` entry
  present in all ten archives;
- all 538,695 official train/val pipeline references resolved (204,894 camera,
  34,149 key LiDAR, 299,652 previous sweeps), with zero missing paths;
- ten real archive payload sentinels passed CRC;
- decoded determinism hashes matched across 0/2/4/8 workers and both repeats;
- measured sample rates were 18.94/46.81/89.08/154.36 samples/s for the first
  repeat at 0/2/4/8 workers. These are batch-size-1 data-loader measurements, not
  end-to-end model-step data-wait percentages.

Exact artifacts and hashes live under the S01 output root recorded in
`usenix27_orchestra/handoffs/S01/RESULTS.md`.

The code no longer defaults to the old Alvis/Mimer path. Provide the dataroot
through either:

- run config key `nuscenes-dataroot`, or
- environment variable `NUSCENES_DATAROOT`, or
- environment variable `ARRHENIUS_NUSCENES_DATAROOT`, or
- module-provided environment variable `NUSCENES_DATA_DIR` (lowest precedence).

ZIP mode also requires an external manifest path:

```bash
export NUSCENES_ZIP_MANIFEST=/nobackup/.../nuscenes_trainval_zip_manifest.sqlite
```

The manifest must remain outside `NUSCENES_DATA_DIR`. Building it over shared
trainval is material compute; prepare/approve the S01 run request before invoking:

```bash
python fl_v3/scripts/s01_nuscenes_zip_manifest.py \
  --dataroot "$NUSCENES_DATA_DIR" \
  --manifest "$NUSCENES_ZIP_MANIFEST"
```

The info-cache must live outside the read-only dataroot. Build the full trainval
cache explicitly only within the same approved material-compute scope, before
training/eval:

```bash
python fl_v3/scripts/build_nuscenes_cache.py \
  --dataroot "$NUSCENES_DATA_DIR" \
  --version v1.0-trainval \
  --splits train val \
  --n-sweeps 10 \
  --cache-dir /path/to/info_cache_msweep10
```

The cache builder adapts the module's `trainval/` metadata directory to the
official devkit version name and reads `sample_data.filename` from metadata. It
does not open or extract image/LiDAR payloads. For the production 10-sweep cache,
use a dedicated cache directory and `--n-sweeps 10`. Production consumers must
call `info_cache.load_cache(..., n_sweeps=10, expected_cache_hash=...)`; ZIP-backed
consumers also bind the accepted logical manifest hash and SQLite file SHA-256.
The pending immutable materialization request and exact launcher are
`usenix27_orchestra/handoffs/S07/RUN_REQUEST.md` and
`scripts/run_s07a_nuscenes_cache_t1v2.sh`.

Mini real-data smoke validated on 2026-07-01:

- Dataroot:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`
- Size: `5.1G`.
- `v1.0-mini` verify passed with `404` samples.
- Cache build job `209574`: `mini_train=323`, `mini_val=81`.
- Eval smoke job `209576`: `eval_loss=8.889362335205078` over 2 examples.
- Train smoke job `209577`: one fp32 optimizer step, finite
  `loss=30.326335906982422`.

Mini remains engineering smoke only. Scientific results require trainval.

## Files

- `fl_v3/scripts/arrhenius_env.sh` - module/cache/env activation.
- `fl_v3/scripts/build_arrhenius_env.sh` - conda + PyTorch + source cumm/spconv builder.
- `fl_v3/scripts/run_arrhenius_env_build.sh` - GH200 Slurm build template.
- `fl_v3/scripts/arrhenius_smoke.py` - import/spconv/data/eval/train smoke harness.
- `fl_v3/scripts/run_arrhenius_smoke.sh` - GH200 Slurm smoke template.
- `fl_v3/src/fl_v3/data/nuscenes/zip_backend.py` - stored-ZIP manifest and
  PID-owned read-only blob store.
- `fl_v3/scripts/s01_nuscenes_zip_{manifest,audit,benchmark}.py` - manifest,
  coverage, and loader-profile evidence tools.
- `fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh` - prepared S01 full-data gate;
  it is not execution authorization.
- `fl_v3/scripts/run_s07a_nuscenes_cache_t1v2.sh` - pending-approval full trainval
  `t1.v2` cache materialization against the accepted S01 manifest; it is not
  execution authorization.
- `fl_v3/requirements.txt` - direct dependency manifest used by the builder.
- `fl_v3/requirements.lock.txt` - Arrhenius audit snapshot, not a standalone reinstall recipe.
- `fl_v3/collab/arrhenius_migration.md` - read-only historical job/version evidence.
- `fl_v3/usenix27_orchestra/` - active protocol, session, kickoff, handoff, and review workspace.

The old Alvis/A40/A100 Slurm launchers and `.venv_v3` helpers have been removed
from the active scripts directory. Historical task/collab documents may still
mention them as provenance for earlier Cycle-04 work.
