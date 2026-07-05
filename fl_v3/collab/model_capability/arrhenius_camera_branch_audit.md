# Arrhenius Camera Branch Audit

Engineering audit for the Arrhenius camera branch. This document is not a
scientific result. Mini nuScenes may be used here only for shape, projection,
memory, finite loss/gradient, tiny-overfit, branch-entry, and module timing
diagnostics. All manifests from this audit must record `scientific_claim=false`.

## Scope

The active camera backbone for this audit is Swin-T. ResNet camera configs and
small-model tests are legacy bring-up/freeze-test support and are unsupported in
the Arrhenius camera audit matrix.

The diagnostic controls separate graph topology from training policy:

- `branch_topology=full_fusion`: camera and LiDAR enter fusion normally.
- `branch_topology=lidar_only`: camera branch is skipped and camera BEV is zero.
- `branch_topology=camera_only`: LiDAR encoder/backbone are skipped and LiDAR
  BEV is zero.
- `train_policy=all_trainable`: no extra script-level freezing.
- `train_policy=camera_frozen`: camera backbone/neck/view-transform frozen.
- `train_policy=lidar_only_trainable`: camera frozen; LiDAR/fusion/neck/head
  trainable.
- `train_policy=camera_only_trainable`: LiDAR modules frozen; camera
  backbone/neck/view-transform, fusion, BEV neck, and head trainable.
- `train_policy=probe_no_backward`: forward/loss/memory probe only.

Mini is not valid evidence for mAP, NDS, ASR, defense behavior, attack
viability, convergence quality, trainval throughput, or model optimality.

## Stop A/B/C/E: Camera Mini Matrix

Launcher:

```bash
sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
```

Default cell:

- `camera_iso_020_fp16_swin`
- `branch_topology=camera_only`
- `train_policy=camera_only_trainable`
- Swin-T, pretrained by default
- mini one-batch eval plus fixed-batch tiny-overfit

Useful variants:

```bash
# Stop A/B one-batch topology + projection smoke
STEPS=1 NUM_TOKENS=2 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

# Stop C tiny-overfit gates
STEPS=5 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
STEPS=20 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
STEPS=100 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

# Stop E branch delta sanity, including full-fusion camera/lidar zeroing
STEPS=1 BRANCH_DELTA_SANITY=1 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
```

The summary manifest is `mini_tiny_overfit_summary.json`. Per-step records are
written as `<cell>_steps.jsonl`.

Required telemetry:

- camera model summary: backbone, Swin-T channels/strides, image shape,
  view-transform shape, camera BEV grid, head grid, fuser contract;
- camera batch summary: camera count, first-sample camera order, raw image shape;
- projection metadata: valid ratio, valid counts/ratios per camera, BEV range;
- camera BEV finite/norm/std/variance/nonzero ratio;
- module trainability and gradient coverage;
- GradScaler scale/skips and optimizer-step count;
- branch execution flags showing whether camera, LiDAR encoder, and LiDAR
  backbone were actually executed.

### Run Log: 2026-07-05 Stop A/B/E 0.2 Smoke

Slurm command:

```bash
STEPS=5 NUM_TOKENS=2 BATCH_SIZE=1 BRANCH_DELTA_SANITY=1 \
  MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
```

Manifest:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241103/mini_tiny_overfit_summary.json
```

This run is `scientific_claim=false`.

Result:

- Stop A topology smoke passed for `camera_iso_020_fp16_swin`:
  `branch_topology=camera_only`, `train_policy=camera_only_trainable`, Swin-T
  `pretrained_backbone=true`, `det-freeze-backbone=false`.
- `camera_only` skipped the LiDAR encoder and LiDAR backbone. Its LiDAR BEV at
  fusion was zero, while camera BEV was finite and nonzero.
- Camera input/order: 6 cameras in the first sample:
  `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_FRONT_LEFT`, `CAM_BACK`,
  `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`.
- Raw images were `[1, 6, 3, 900, 1600]`; preprocessed images were
  `[1, 6, 3, 256, 704]`.
- Swin-T taps were `[6,96,64,176]`, `[6,192,32,88]`, `[6,384,16,44]`,
  `[6,768,8,22]`; camera neck was `[6,128,16,44]`; depth/context were
  `[6,59,16,44]` and `[6,80,16,44]`.
- Camera BEV was `[1,80,512,512]`; fuser contract was camera 80 channels,
  LiDAR 128 channels, output 128 channels; head heatmap grid was `[256,256]`.
- Projection metadata looked internally consistent for the 0.2 grid:
  valid ratio `0.428901`; per-camera valid ratios were approximately
  `[0.437, 0.452, 0.451, 0.328, 0.453, 0.452]`.
- Camera BEV health remained finite/nonzero across the 5-step gate:
  norm about `45-46`, nonzero ratio about `0.126`, std about `0.010`.
- GradScaler behavior: init scale `512`, skipped the first 4 steps due to
  nonfinite fp16 grads, recovered to a finite update at scale `32` on step 4.
  The cell recorded `optimizer_steps=1`.
- Last-step grad coverage matched the intended camera policy: camera backbone,
  camera neck, view-transform, fusion, BEV neck, and head had finite gradients;
  LiDAR encoder/backbone were frozen with zero trainable parameters.
- Stop E branch delta passed as an engineering sanity check: `camera_only`
  differed from `full_fusion`; zeroing camera or LiDAR changed head outputs; the
  camera feature norm was nonzero.

Engineering caveats:

- The earlier Slurm job `241097` failed because the first harness version
  treated any GradScaler overflow as an immediate failure. The harness now
  records fp16 GradScaler skips and fails only if no optimizer step occurs by
  the end of the cell.
- This run covers the 0.2 camera grid. The explicit 0.075 projection/coupling
  check remains pending for Stop D or a follow-up Stop B add-on.

### Run Log: 2026-07-05 Stop C Tiny-Overfit

Slurm commands:

```bash
STEPS=20 NUM_TOKENS=2 BATCH_SIZE=1 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

STEPS=100 NUM_TOKENS=2 BATCH_SIZE=1 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
```

Manifests:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241134/mini_tiny_overfit_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241138/mini_tiny_overfit_summary.json
```

Both runs are `scientific_claim=false`.

Result:

- 20-step camera-only gate passed: loss `540.071 -> 22.809`, `optimizer_steps=16`,
  GradScaler final scale `32`, no warnings.
- 100-step camera-only gate passed: loss `540.073 -> 11.596`,
  `optimizer_steps=96`, GradScaler final scale `32`, no warnings.
- The first 4 fp16 steps skipped due to GradScaler overflow, then all later
  steps updated normally.
- Trainability matched `camera_only_trainable`: camera backbone, camera neck,
  view-transform, fusion, BEV neck, and head were trainable; LiDAR
  encoder/backbone were frozen.
- This is only a fixed-batch engineering overfit gate. It is not convergence,
  mAP, NDS, ASR, or trainval evidence.

### Run Log: 2026-07-05 Camera Scratch Mini Gate

Purpose:

- Test Swin-T camera branch with `det-pretrained-backbone=false`.
- Keep LiDAR scratch status explicit: sparse voxel / pillar LiDAR modules have no
  pretrained load path in the current detector construction.
- Compare scratch camera engineering health at both 0.2 and 0.075 grids without
  making performance claims.

Implementation note:

- Added manifest field `camera_init_policy` with values
  `imagenet_pretrained` or `scratch`.
- Added `camera_iso_075_ch256_fp16_swin` as a trainable 0.075 camera-only cell.
- Slurm jobs `241288`, `241289`, and `241290` were invalidated as scratch
  evidence because matrix cell defaults overrode `PRETRAINED_BACKBONE=0`.
  The launch precedence was fixed so `BACKBONE` and `PRETRAINED_BACKBONE` win
  after matrix cell expansion when `respect_config_shape=false`.

Valid Slurm commands:

```bash
PRETRAINED_BACKBONE=0 STEPS=100 NUM_TOKENS=2 BATCH_SIZE=1 \
  MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

PRETRAINED_BACKBONE=0 STEPS=0 NUM_TOKENS=2 BATCH_SIZE=1 \
  MATRIX=camera_iso_075_ch256_fp16_probe \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

PRETRAINED_BACKBONE=0 STEPS=10 NUM_TOKENS=2 BATCH_SIZE=1 \
  MATRIX=camera_iso_075_ch256_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

PRETRAINED_BACKBONE=0 PROFILE_ITERS=3 WARMUP_ITERS=1 NUM_TOKENS=4 BATCH_SIZE=1 \
  MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

PRETRAINED_BACKBONE=0 PROFILE_ITERS=2 WARMUP_ITERS=1 NUM_TOKENS=3 BATCH_SIZE=1 \
  MATRIX=camera_iso_075_ch256_fp16_probe \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh
```

Manifests:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241296/mini_tiny_overfit_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241297/mini_tiny_overfit_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241298/mini_tiny_overfit_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/stop_f_camera_profile_mini_241302/profile_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/stop_f_camera_profile_mini_241303/profile_summary.json
```

All valid runs are `scientific_claim=false` and record
`camera_init_policy=scratch`.

Result:

- Scratch 0.2 100-step camera-only gate passed: loss `518.832 -> 11.461`,
  `optimizer_steps=97`, GradScaler final scale `64`, no warnings.
- Scratch 0.075 no-backward probe passed: camera BEV `[1,80,1440,1440]`,
  finite/nonzero, valid projection ratio `0.435783`.
- Scratch 0.075 10-step trainable gate passed: loss `1792.759 -> 312.725`,
  first 5 steps skipped while GradScaler reduced `512 -> 16`, then 5 optimizer
  steps completed, no warnings.
- Scratch 0.2 profiler passed: mean compute `123.9 ms`, peak alloc
  `2102.5 MiB`.
- Scratch 0.075 no-backward profiler passed: mean compute `132.3 ms`, peak alloc
  `18288.6 MiB`.

Engineering interpretation:

- Scratch camera can pass the mini fixed-batch engineering overfit gate at 0.2.
- Scratch 0.075 is shape/projection healthy and can backward/update, but it needs
  a lower GradScaler scale (`16` in this run) and has much higher memory.
- Mini loss decrease is not scientific evidence that scratch is better, equal,
  or worse than ImageNet initialization.

## Stop D: Resolution Coupling

Use matrix cells as engineering shape/memory probes only:

```bash
# Camera-only 0.2 grid
STEPS=1 MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

# Camera-only 0.075 grid, no backward
STEPS=0 MATRIX=camera_iso_075_ch256_fp16_probe \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

# Full-fusion 0.2 grid
STEPS=1 MATRIX=full_fusion_voxel020_fp16_current \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh

# Full-fusion 0.075 parity probe, no backward
STEPS=0 MATRIX=full_fusion_voxel075_z020_ch256_fp16_parity_probe \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
```

Acceptance is limited to shape legality, BEV/head/fuser coupling, finite
forward/loss when applicable, memory recording, and non-collapsed camera BEV
features.

### Run Log: 2026-07-05 Stop D Resolution Coupling

Clean Slurm command after the 0-step probe guard fix:

```bash
STEPS=0 NUM_TOKENS=2 BATCH_SIZE=1 \
  MATRIX=camera_iso_075_ch256_fp16_probe,full_fusion_voxel020_fp16_current,full_fusion_voxel075_z020_ch256_fp16_parity_probe \
  sbatch fl_v3/scripts/run_arrhenius_mini_matrix.sh
```

Manifest:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/camera_audit_mini_matrix_241143/mini_tiny_overfit_summary.json
```

This run is `scientific_claim=false`.

Result:

- `camera_iso_075_ch256_fp16_probe` passed with `branch_topology=camera_only`,
  `train_policy=probe_no_backward`.
- `full_fusion_voxel020_fp16_current` passed as a clean 0-step shape/projection
  probe.
- `full_fusion_voxel075_z020_ch256_fp16_parity_probe` passed with
  `train_policy=probe_no_backward`.
- 0.2 grid coupling: camera BEV `[1,80,512,512]`, LiDAR/fused BEV
  `[1,128,512,512]`, head heatmap grid `[256,256]`, fuser contract camera 80,
  LiDAR 128, output 128.
- 0.075 grid coupling: camera BEV `[1,80,1440,1440]`, LiDAR/fused BEV
  `[1,256,1440,1440]`, head heatmap grid `[720,720]`, fuser contract camera 80,
  LiDAR 256, output 256.
- 0.2 projection valid ratio was `0.428901`; 0.075 projection valid ratio was
  `0.435783`. Per-camera ratios remained stable, with the rear camera lower
  than the others in both grids.
- Camera BEV health stayed finite/nonzero. The 0.075 camera BEV nonzero ratio
  was about `0.0299`, lower than the 0.2 ratio about `0.126`, as expected from
  the denser grid.

Engineering note:

- Slurm jobs `241130` and `241137` exposed a harness issue where 0-step probes
  still required an optimizer step. The guard now checks optimizer-step recovery
  only when `steps > 0`.

## Stop F: Camera Module Teardown

Launcher:

```bash
sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh
```

Default cell:

- `camera_iso_020_fp16_swin`
- Swin-T, pretrained by default
- warmup plus measured steps
- manifest field `engineering_speed_candidate=true`

The profiler records synchronized stage timings for host-to-device copy,
preprocess, Swin-T backbone, camera neck, LSS view-transform, LiDAR stages when
enabled, fusion, BEV neck, head, loss, backward, unscale/grad norm, optimizer,
data fetch, and peak memory.

Engineering variants:

```bash
# Camera-only teardown
MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

# SDPA / checkpoint engineering candidates
MATRIX=camera_iso_020_fp16_swin,camera_iso_020_fp16_swin_no_sdpa,camera_iso_020_fp16_swin_no_ckpt \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

# 0.075 camera cost growth, no backward
MATRIX=camera_iso_075_ch256_fp16_probe PROFILE_ITERS=2 WARMUP_ITERS=1 \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

# Full-fusion comparison
MATRIX=full_fusion_voxel020_fp16_current \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh
```

Stop F can identify engineering speed candidates, such as SDPA, activation
checkpointing, compile knobs already supported by the code path, and resolution
cost growth. It cannot prove global optimality or final trainval throughput.

### Run Log: 2026-07-05 Stop F Teardown

Slurm commands:

```bash
PROFILE_ITERS=3 WARMUP_ITERS=1 NUM_TOKENS=4 BATCH_SIZE=1 \
  MATRIX=camera_iso_020_fp16_swin \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

PROFILE_ITERS=3 WARMUP_ITERS=1 NUM_TOKENS=4 BATCH_SIZE=1 \
  MATRIX=camera_iso_020_fp16_swin_no_sdpa,camera_iso_020_fp16_swin_no_ckpt \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

PROFILE_ITERS=2 WARMUP_ITERS=1 NUM_TOKENS=3 BATCH_SIZE=1 \
  MATRIX=camera_iso_075_ch256_fp16_probe \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh

PROFILE_ITERS=5 WARMUP_ITERS=1 NUM_TOKENS=6 BATCH_SIZE=1 \
  MATRIX=full_fusion_voxel020_fp16_current \
  sbatch fl_v3/scripts/run_arrhenius_profile_mini.sh
```

Manifests:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/stop_f_camera_profile_mini_241133/profile_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/stop_f_camera_profile_mini_241139/profile_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/stop_f_camera_profile_mini_241144/profile_summary.json
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/stop_f_camera_profile_mini_241149/profile_summary.json
```

All runs are `scientific_claim=false`.

Measured mini engineering candidates:

| Cell | Mean compute ms | Peak alloc MiB | Candidate | Note |
| --- | ---: | ---: | --- | --- |
| `camera_iso_020_fp16_swin` | `208.7` | `2102.5` | yes | baseline, SDPA on, checkpoint on |
| `camera_iso_020_fp16_swin_no_sdpa` | `206.7` | `2097.4` | yes | similar to baseline in this tiny screen |
| `camera_iso_020_fp16_swin_no_ckpt` | `84.0` | `3353.6` | yes | faster here, higher memory |
| `camera_iso_075_ch256_fp16_probe` | `144.4` | `18288.6` | yes | no-backward probe, much larger BEV memory |
| `full_fusion_voxel020_fp16_current` | `220.8` | `3812.3` | yes | full-fusion 0.2 comparison |

Stage timing highlights:

- Baseline 0.2: camera backbone about `34.3 ms`, view-transform `2.7 ms`,
  fusion `3.4 ms`, backward `78.4 ms`.
- No-checkpoint 0.2: camera backbone about `23.9 ms`, backward `31.9 ms`, peak
  allocation about `3.35 GiB`.
- 0.075 no-backward probe: fusion about `50.9 ms`, BEV neck about `13.3 ms`,
  peak allocation about `18.3 GiB`.
- Full-fusion 0.2: camera backbone about `27.0 ms`, view-transform `2.5 ms`,
  LiDAR encoder/backbone about `22.2/8.9 ms`, backward `78.8 ms`, peak
  allocation about `3.81 GiB`.

Engineering interpretation:

- Disabling activation checkpointing is a plausible speed candidate for
  camera-only 0.2 mini gates when memory headroom exists.
- SDPA on/off was effectively neutral in this very small screen.
- 0.075 resolution has a large memory and fusion/BEV cost increase; any
  architecture rewrite, caching, or view-transform redesign should be planned as
  a later architecture round, not merged into this audit.

## Current Implementation Notes

- `arrhenius_lidar_gap_utils.py` remains the shared helper name for compatibility,
  but now covers camera and LiDAR branch diagnostics.
- New camera audit cells are Swin-T by default and keep ResNet out of the active
  Arrhenius camera matrix.
- `DepthLSSTransform.record_debug` controls projection metadata collection. It is
  enabled by the audit harnesses and off by default for production forwards.
- 0-step mini-matrix probes do not require optimizer-step recovery; GradScaler
  all-skip failure remains enforced for training gates with `steps > 0`.
- Stop F no-backward probes can be marked `engineering_speed_candidate=true`
  when measured steps and feature/finite gates pass.
- Launch-level `BACKBONE` and `PRETRAINED_BACKBONE` override matrix defaults
  after cell expansion unless `respect_config_shape=true`.
- Cleanup is conservative: legacy ResNet support is documented instead of
  deleted, and only touched audit paths should remove proven-dead variables or
  stale defaults.

## Verification

Static checks:

```bash
python3 -m py_compile \
  fl_v3/scripts/arrhenius_lidar_gap_utils.py \
  fl_v3/scripts/arrhenius_mini_matrix.py \
  fl_v3/scripts/arrhenius_profile_mini.py \
  fl_v3/src/fl_v3/models/fusion/view_transform.py \
  fl_v3/tests/test_arrhenius_camera_audit_controls.py

bash -n \
  fl_v3/scripts/run_arrhenius_mini_matrix.sh \
  fl_v3/scripts/run_arrhenius_profile_mini.sh

git diff --check
```

Focused Slurm pytest:

```bash
sbatch --wrap="source fl_v3/scripts/arrhenius_env.sh; arrhenius_load_modules run; arrhenius_activate_env; python -m pytest -q fl_v3/tests/test_arrhenius_camera_audit_controls.py"
```

Result:

- Slurm job `241148`: `5 passed in 2.92s`.
- Slurm job `241306`: `6 passed in 3.10s`.
