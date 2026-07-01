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
- flwr `1.27.0`, ray `2.51.1`, scikit-learn `1.8.0`.
- nuscenes-devkit `1.1.11`.
- cumm `v0.7.13` at `4dedaf43ff801e417c60c6bd7536a29d83d29ee0`.
- spconv `v2.3.8` at `263d6b47425ef843c82f997b12d8b714013d216c`.

The source build is pinned to `sm_90`:

```bash
TORCH_CUDA_ARCH_LIST=9.0
CUMM_CUDA_ARCH_LIST=9.0
```

Precision status:

- Supported for Arrhenius sparse path: `fp32` dev/debug/reference and `fp16`
  AMP + GradScaler (`init_scale=512`) for sparse training.
- Direct `torch.bfloat16` sparse convolution is not supported by this
  cumm/spconv path; `bf16` configs should fail loudly instead of falling back.
- Trainer, smoke, and provenance paths use the explicit `precision` policy
  rather than inferring AMP dtype from cuDNN deterministic flags.

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

The code no longer defaults to the old Alvis/Mimer path. Provide the dataroot
through either:

- run config key `nuscenes-dataroot`, or
- environment variable `NUSCENES_DATAROOT`, or
- environment variable `ARRHENIUS_NUSCENES_DATAROOT`.

The info-cache must live outside the read-only dataroot. Build it explicitly
before training/eval:

```bash
python fl_v3/scripts/build_nuscenes_cache.py \
  --dataroot "$ARRHENIUS_NUSCENES_DATAROOT" \
  --cache-dir /path/to/info_cache
```

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
- `fl_v3/requirements.txt` - direct dependency manifest used by the builder.
- `fl_v3/requirements.lock.txt` - Arrhenius audit snapshot, not a standalone reinstall recipe.
- `fl_v3/collab/arrhenius_migration.md` - detailed job IDs, commands, versions, failures, and fixes.

The old Alvis/A40/A100 Slurm launchers and `.venv_v3` helpers have been removed
from the active scripts directory. Historical task/collab documents may still
mention them as provenance for earlier Cycle-04 work.
