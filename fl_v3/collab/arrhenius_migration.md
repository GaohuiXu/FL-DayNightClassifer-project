# Arrhenius migration — platform de-risk (spconv + fp16 precision)

> Alvis (x86, A100, bf16) sunsets ~2026-06-30 → Arrhenius (aarch64, GH200, **no bf16 for spconv**). This doc
> captures the platform-readiness findings before the move. Authored 2026-06-28.

## Phase 0A — spconv on GH200/aarch64 (owner's test, 2026-06-28)
**Result: `stable_enough_for_experimental_branch`.** Source-built spconv (v2.3.8 / cumm v0.7.13) imports and runs
sparse-conv fwd/bwd on GH200; 2-GPU FP32 DDP and 2-GPU **FP16-AMP + GradScaler** DDP both pass multiple steps.
- ⇒ **The no-spconv Rule #2 constraint is LIFTED on Arrhenius** — the pillars→sparse-3D-voxel LiDAR upgrade
  (audit ceiling #2) is now possible THERE (not on Alvis). PyPI has no aarch64 wheel → source JIT build
  (`TORCH_CUDA_ARCH_LIST=9.0`, EasyBuild GCC `libstdc++` preloaded ahead of the system one).
- **Precision caveat:** direct `torch.bfloat16` is NOT supported by this spconv/cumm path (`KeyError: torch.bfloat16`);
  `autocast(bf16)` runs but the sparse output is fp16. So with spconv: **FP16-AMP+GradScaler for the sparse branch,
  or force it FP32** (no bf16 sparse conv in this spconv/cumm build). Remote workspace: `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/phase0a_spconv`.

## Phase 0B — fl_v3 Arrhenius env solidification (Codex, 2026-06-30)

**Result: `env_ready_except_real_nuscenes_data`.** The Arrhenius branch worktree now has repeatable scripts for the
Python/PyTorch/CUDA/spconv/cumm stack, Slurm templates, import/sparse-conv smoke, optional nuScenes data/eval/train
gates, and a dummy FL training smoke. The main `v3-ad-perception` worktree was not modified.

### Worktree and artifact layout

- Worktree: `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project__arrhenius-env-bringup`
  (`arrhenius/env-bringup-v3`).
- All mutable env/cache/build/tmp/output artifacts live under
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3`.
- Conda prefix: `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/envs/pt311-cu128-spconv`.
- Source builds: `.../arrhenius_fl_v3/src/cumm` and `.../arrhenius_fl_v3/src/spconv`.
- Caches: `conda_pkgs`, `pip_cache`, `xdg_cache`, `torch_home`, `torchinductor_cache`, `ccache`, `tmp`.
- Outputs/cache default: `.../arrhenius_fl_v3/outputs`; nuScenes info-cache default
  `.../arrhenius_fl_v3/outputs/nuscenes/info_cache`.
- Lock snapshot: `.../arrhenius_fl_v3/requirements.arrhenius.lock.txt`.

This prefix is intentionally long-lived project storage state. Normal Slurm jobs
reuse it via `fl_v3/scripts/arrhenius_env.sh`; they do **not** recreate conda,
PyTorch, cumm, or spconv on every submission. The binary environment and source
build artifacts are not committed to Git; the committed assets are the build
recipe, activation script, Slurm templates, and audit lock snapshot.

### Scripts added

- `fl_v3/scripts/arrhenius_env.sh` - shared bootstrap. Forces conda/pip/torch/tmp/cache paths under `/nobackup`,
  activates by prefix PATH, sets `TORCH_CUDA_ARCH_LIST=9.0`, `CUMM_CUDA_ARCH_LIST=9.0`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `TORCH_HOME`, and the libstdc++ preload path.
- `fl_v3/scripts/build_arrhenius_env.sh` - builds the env on a GH200 node. It installs PyTorch cu128 aarch64 wheels,
  fl_v3 deps, editable `fl_v3`, editable source cumm/spconv, pre-caches ResNet18/Swin-T weights, and writes the lock.
- `fl_v3/scripts/run_arrhenius_env_build.sh` - Slurm build template.
- `fl_v3/scripts/arrhenius_smoke.py` - modes: `import`, `spconv`, `data`, `eval`, `train`, `dummy-train`.
- `fl_v3/scripts/run_arrhenius_smoke.sh` - Slurm smoke template. Defaults to build modules because source-built
  cumm/spconv can still trigger `ccimport`/ninja checks on import.

### Slurm / module settings

- Account: `naiss2025-22-1113-gpu`; partition: `gpu`; GRES: `--gpus-per-node=nvidia_gh200_120gb:1`.
- Build template: 32 CPU cores, 2 h wall time.
- Smoke template: 16 CPU cores, 40 min wall time.
- Build modules tried first: `GPU/buildenv-nvhpc/25.9-cu12.9.1-eb` + `GPU/Miniforge/26.3.2-2-eb`.
- Runtime smoke also defaults to build modules (`LOAD_BUILDENV=1`) because source cumm/spconv import can rebuild
  extensions; Miniforge-only runtime failed with `which: no nvcc`.

### Build command and versions

Cold rebuild:

```bash
cd /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project__arrhenius-env-bringup
sbatch --export=ALL,RECREATE=1 fl_v3/scripts/run_arrhenius_env_build.sh
```

Incremental/idempotent rebuild:

```bash
sbatch fl_v3/scripts/run_arrhenius_env_build.sh
```

Successful build job: `208641` on `n405`, `COMPLETED`, `00:08:28`, exit `0:0`.

Observed versions:

- Python `3.11.15` (conda-forge, GCC 14.3.0).
- PyTorch `2.11.0+cu128`, CUDA wheel runtime `12.8`, CUDA compiler module `12.9.1`.
- torchvision `0.26.0+cu128`.
- numpy `1.26.4`; scipy `1.13.1`; flwr `1.27.0`; ray `2.51.1`; scikit-learn `1.8.0`;
  matplotlib `3.10.8`; nuscenes-devkit `1.1.11`; fl_v3 editable `0.1.0`.
- cumm `v0.7.13`, commit `4dedaf43ff801e417c60c6bd7536a29d83d29ee0`.
- spconv `v2.3.8`, commit `263d6b47425ef843c82f997b12d8b714013d216c`; working tree is intentionally
  dirty because the build patches `pyproject.toml` to remove the isolated `cumm>=0.7.11` build dependency and use
  the local editable cumm.
- Cached weights in `TORCH_HOME`: `resnet18-f37072fd.pth`, `swin_t-704ceda3.pth`.

### Smoke commands and results

Default smoke:

```bash
sbatch fl_v3/scripts/run_arrhenius_smoke.sh
```

Successful smoke job: `208660` on `n405`, `COMPLETED`, `00:05:50`, exit `0:0`.

Results:

- `import` passed on `aarch64` / `NVIDIA GH200 120GB`, compute capability `(9, 0)`.
- cumm/spconv source extensions built with `-gencode=arch=compute_90,code=sm_90`.
- spconv FP32 sparse-conv fwd/bwd passed: output `(4466, 64)`, dtype `torch.float32`, finite loss/grad.
- spconv FP16 autocast + GradScaler passed for two optimizer steps: output dtype `torch.float16`, scale `512.0`.
- nuScenes data preflight failed non-fatally because the configured/default dataroot does not exist on Arrhenius:
  `/mimer/NOBACKUP/Datasets/NuScenes_v1.0/v1.0-mini` was missing, and no staged info-cache exists under
  `.../arrhenius_fl_v3/outputs/nuscenes/info_cache`.
- `dummy-train` passed: checksum prefix `3ed98114b14e3106`, final eval loss `0.3695202488452196`.

After adding shell-level `CUBLAS_WORKSPACE_CONFIG`, the minimal training smoke was repeated:

```bash
sbatch --export=ALL,SMOKE_MODES='dummy-train' fl_v3/scripts/run_arrhenius_smoke.sh
```

Job `208666` on `n428`, `COMPLETED`, `00:00:52`, exit `0:0`; same checksum prefix
`3ed98114b14e3106`; no late-`CUBLAS_WORKSPACE_CONFIG` warning.

Eval/train data gate:

```bash
sbatch --export=ALL,SMOKE_MODES='eval train' fl_v3/scripts/run_arrhenius_smoke.sh
```

Job `208668` on `n405`, `COMPLETED`, `00:00:13`, exit `0:0`; both real nuScenes eval and real-data train
correctly printed `SKIP data/cache unavailable`.

Real nuScenes eval/train smoke once data is staged:

```bash
sbatch --export=ALL,ARRHENIUS_NUSCENES_DATAROOT=/path/to/NuScenes_v1.0,ARRHENIUS_NUSCENES_CACHE=/path/to/info_cache,REQUIRE_DATA=1,SMOKE_MODES='import spconv data eval train' fl_v3/scripts/run_arrhenius_smoke.sh
```

`REQUIRE_DATA=1` makes missing data/cache fatal. Until a real Arrhenius nuScenes dataroot and matching info-cache
are provided, no scientific or real-data eval/train claim should be made.

### nuScenes data access check (2026-06-30)

Official C3SE documentation lists a central training-dataset collection under `/mimer/NOBACKUP/Datasets` and
includes NuScenes v1.0. On the current Arrhenius login node, however, `/mimer` and `/cephyr` are not mounted:

```bash
hostname -f
# arrhenius1.hpc.arrhenius.naiss.se
uname -m
# x86_64
ls -ld /mimer /cephyr
# No such file or directory
```

The Arrhenius project disk search found no staged real nuScenes tree (`samples`, `sweeps`, `v1.0-mini`,
`v1.0-trainval`) under `/nobackup/proj/disk/naiss2024-22-991`; only source tests and the empty output/cache
directories exist. A target staging directory has been created:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/datasets/NuScenes_v1.0
```

Attempting to inspect Mimer from Arrhenius via SSH to Alvis did not work from this session because no usable
Alvis SSH credential is present:

```bash
ssh -o BatchMode=yes alvis1.c3se.chalmers.se 'ls -ld /mimer/NOBACKUP/Datasets/NuScenes_v1.0'
# Permission denied (publickey,password,hostbased).
```

Recommended next action is to start the transfer from an authenticated Alvis/Mimer session, following the C3SE
Alvis-to-Arrhenius migration guidance:

```bash
module load rsync
rsync -a --compress --compress-choice=zstd --itemize-changes \
  /mimer/NOBACKUP/Datasets/NuScenes_v1.0/ \
  gaohui@login.hpc.arrhenius.naiss.se:/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/datasets/NuScenes_v1.0/
```

Current storage quota on `/nobackup/proj/disk/naiss2024-22-991` is 250 GiB soft / 625 GiB hard; full nuScenes
trainval may exceed the soft quota once combined with env/build artifacts. Prefer staging mini first for a real
data smoke, then request/use sufficient project storage for trainval before long experiments.

### nuScenes mini real-data smoke (2026-07-01)

A staged mini dataset was found at:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
```

This is a complete extracted mini root with `v1.0-mini/`, `samples/`, `sweeps/`, and `maps/`; size is `5.1G`.
Lightweight verify passed with `n_samples=404` and sentinel token `ca9a282c9e77460f8360f564131a8af5`.

Cache build on GH200:

```bash
DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
CACHE=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/nuscenes/info_cache_mini_from_main
```

Job `209574` on `n408`, `COMPLETED`, `00:01:25`, exit `0:0`:

- `mini_train`: `n_samples=323`, `n_boxes=13923`, cache hash prefix `e258516d0d4b5c36`.
- `mini_val`: `n_samples=81`, `n_boxes=4441`, cache hash prefix `e25782d0167319f3`.

Real-data smoke:

```bash
ARRHENIUS_NUSCENES_DATAROOT="${DATAROOT}" \
ARRHENIUS_NUSCENES_CACHE="${CACHE}" \
REQUIRE_DATA=1 SMOKE_MODES='data eval train' \
EVAL_LIMIT=2 TRAIN_STEPS=1 BATCH_SIZE=1 PRECISION=fp32 \
MIN_KEYFRAMES_PER_CLIENT=10 \
sbatch --export=ALL fl_v3/scripts/run_arrhenius_smoke.sh
```

Results:

- Job `209575`: `import` and `data` passed, then failed before eval because `precision=fp16` is rejected by the
  current `enforce_determinism` single-knob validator (`bf16`/`fp32` only). This is an integration/config issue,
  not a data-access failure.
- Job `209576`: `data` and `eval` passed in `fp32`; proxy eval over 2 examples returned
  `eval_loss=8.889362335205078`, `num-eval-examples=2.0`.
- Job `209577`: `train` passed in `fp32` with mini partition floor lowered to 10 keyframes/client; one optimizer
  step produced finite `loss=30.326335906982422`, `grad_norm=1166.07958984375`, `skipped=False`; exit `0:0`.

`fl_v3/scripts/arrhenius_smoke.py` was adjusted so `precision=fp32` uses no autocast in the smoke train path, and
`fl_v3/scripts/run_arrhenius_smoke.sh` now accepts `MIN_KEYFRAMES_PER_CLIENT` for mini-only partition smoke tests.
This does not change model/trainer code and is not a scientific training configuration.

### Arrhenius-first repository cleanup (2026-07-01)

The active `fl_v3/scripts/` shell launchers were reduced to Arrhenius-facing
entry points. Legacy Alvis/A40/A100 Slurm wrappers and the old `.venv_v3`
helpers were removed from the active scripts directory; historical
`docs/cycle_04/*` and `collab/*` files still mention them as provenance for
earlier gates. Python utilities were retained for future Arrhenius launchers to
reuse or migrate.

`fl_v3/src/fl_v3/data/nuscenes/paths.py` no longer defaults to the old
`/mimer` path. Dataroot resolution is now: run config `nuscenes-dataroot`, then
`NUSCENES_DATAROOT` / `ARRHENIUS_NUSCENES_DATAROOT`, otherwise a clear
configuration error. `build_nuscenes_cache.py` now accepts `--dataroot`.

### Gotchas fixed during solidification

- The Miniforge module resets `CONDA_PKGS_DIRS` to a home-backed `.conda-hpc` path; `arrhenius_env.sh` now forces
  conda/pip/tmp/cache variables back under `/nobackup` after every module load.
- The mamba-created prefix did not have a usable `bin/activate`; activation now uses `CONDA_PREFIX` + PATH.
- Miniforge-only runtime failed because source cumm/spconv import attempted a ninja build with no `nvcc`; smoke
  defaults to the CUDA build environment.
- `TORCH_CUDA_ARCH_LIST` / `CUMM_CUDA_ARCH_LIST` must be exported for runtime imports too, not only during the
  initial env build; otherwise cumm/spconv can attempt a slow multi-arch rebuild. The shared bootstrap now pins
  both to `9.0`.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` must be set before CUDA initialization; the shared bootstrap now exports it.

## fp16 de-risk on our model (Alvis, 2026-06-28, jobs 6782420/6782421/6782422)
Since bf16 is unavailable for the sparse path in this build, the science precision on Arrhenius is
**fp16-AMP+GradScaler (or fp32)**. Validated the non-spconv model (camera-LSS / pillar-LiDAR / fusion /
CenterPoint head) under fp16-AMP+GradScaler via `p1_amp_smoke.py --fp16` (reports GradScaler skips = the direct
inf/NaN signal). *(The model at the time of this run also carried the depth-supervision lever, since removed as
net-negative; the fp16 verdict is about the precision path and is unaffected — the depth numbers below are that
historical run's.)*

**Verdict: fp16 is VIABLE — no fundamental blocker.**
- **Forward is fp16-safe** — total/hm/reg/depth all finite, depth GT coverage 73–79%. My pre-test worry (the LSS
  `cumsum`/splat overflowing fp16) was **unfounded**: the D15 relaxed splat already accumulates in **fp32**
  (`.float()` before `scatter_add_`), so the BEV pool never sees fp16 range limits.
- **Backward calibrates then trains.** From GradScaler's default init_scale 65536, the huge random-init focal loss
  (hm≈160 ⇒ grad-norm ~11k, like the bf16 run) overflows fp16 grads → GradScaler halves the scale for ~7 steps
  (skipped) down to **512**, then **every step lands** (scale stable at 512, no more skips). Over steps 7–39 the
  loss drops cleanly (**total 50→11.5, hm 44→7.0, grad-norm 2703→28**) and the **depth CE itself drops 4.02→3.1**
  — i.e. fp16 trains both detection and depth supervision, just like bf16.
- **Only cost:** ~7 throwaway calibration steps at startup. Removable with **GradScaler `init_scale≈512`** (skips the
  search) or a brief **fp32 warmup**. Neither is necessary (7 of thousands of steps).

## Precision plan for Arrhenius (implemented in Stop B)
- **Arrhenius active policy:** `fp32` for dev/debug/reference; **fp16 AMP + GradScaler**
  (`init_scale=512`) for supported sparse training.
- **Direct sparse bf16 is disallowed** for the validated cumm/spconv path; configs/scripts should fail loudly
  instead of falling back.
- **Keep fp32 for:** the LSS BEV splat (already), GroupNorm/softmax (autocast does), the focal/L1 head loss
  (upcast before criterion), EMA, and `view_transform.depth_targets` (already autocast-disabled).
- Determinism stays **seed-variance-based** (D16 already relaxed byte-identity); GradScaler's dynamic scale widens
  the band slightly. Cross-platform numbers compare only at the seed-variance level, never byte-wise.
- **No code blocker found** — the platform's existing fp32-accumulation choices (D15 splat, head-loss upcast)
  already make the model fp16-friendly. The shared trainer and Arrhenius smoke now use explicit precision
  helpers instead of inferring AMP dtype from `cudnn.deterministic`.

## Tooling
`scripts/p1_amp_smoke.py --precision fp16` (fp16-AMP+GradScaler, cycles the loader, reports per-step scale + skips + finite),
`scripts/run_p1_amp_smoke.sh FP16=1`. Reusable for the Arrhenius bring-up smoke.
