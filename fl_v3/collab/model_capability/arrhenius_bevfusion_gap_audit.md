# Arrhenius BEVFusion Gap Audit

Engineering audit for the Arrhenius LiDAR-capability push. This document is not
a scientific result. Mini nuScenes may be used here only for shape, memory,
finite-loss, gradient, and tiny-overfit diagnostics.

## Reference Facts

MIT BEVFusion reports the nuScenes validation camera+LiDAR detector at
`mAP=68.52`, `NDS=71.38`, and test at `mAP=70.23`, `NDS=72.88` in its README.
The public detection command uses the `swint_v0p075/convfuser.yaml` config.

The LiDAR-only `voxelnet_0p075.yaml` reference sets:

- `voxel_size=[0.075, 0.075, 0.2]`
- `point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]`
- `sparse_shape=[1440, 1440, 41]`
- `max_voxels=[120000, 160000]`
- `SparseEncoder`, `output_channels=128`
- raw train/test `grid_size=[1440, 1440, 41]`

The camera+LiDAR `convfuser.yaml` reference sets `ConvFuser` input channels
`[80, 256]` and output channels `256`.

Sources:

- https://github.com/mit-han-lab/bevfusion
- https://raw.githubusercontent.com/mit-han-lab/bevfusion/main/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
- https://raw.githubusercontent.com/mit-han-lab/bevfusion/main/configs/nuscenes/det/transfusion/secfpn/camera%2Blidar/swint_v0p075/convfuser.yaml
- https://raw.githubusercontent.com/mit-han-lab/bevfusion/main/mmdet3d/models/backbones/sparse_encoder.py

## Current fl_v3 Baseline

The current strongest centralized Alvis reference is `bb02d`:
`0.5656 mAP / 0.5733 NDS`, config `fl_v3/configs/p1_bb02d.json`.
It uses:

- trainable Swin-T camera backbone
- 10 LiDAR sweeps
- `det-bev-voxel=0.2`
- max pillars/voxels `120000`
- dense 2D LiDAR backbone, 4 stages, output channels `128`
- `ConvFuser` LiDAR input channels `128`, output channels `128`
- CenterPoint-style dense head

The Arrhenius sparse path in `SparseVoxelEncoder` is active behind
`det-lidar-encoder=voxel`, but it is not a full MIT SparseEncoder clone.

## Non-Parity Items To Track

1. Resolution and range differ from BEVFusion unless the 0.075 parity cell is
   explicit: `det-bev-voxel=0.075`, `det-lidar-z-voxel=0.2`,
   `det-pc-range=[-54,-54,-5,54,54,3]`.

2. The historical `SparseVoxelEncoder` used cubic z voxels by default. At
   `det-bev-voxel=0.075`, leaving z implicit would create 0.075m z bins rather
   than BEVFusion's 0.2m z bins. The new `det-lidar-z-voxel` knob exists to
   prevent that silent non-parity.

3. BEVFusion records `sparse_shape=[1440,1440,41]` even though the z range
   divided by 0.2 gives 40 bins. The parity probe uses an explicit
   `det-lidar-sparse-z-size=41` and logs both computed z bins and active sparse
   z size.

4. MIT SparseEncoder downsamples through a SECOND-style sparse 3D encoder.
   The current `SparseVoxelEncoder` keeps xy at the fine BEV grid through its
   sparse 3D path and mainly downsamples z before dense BEV collapse. If the
   sparse branch remains weak after z/range/channel parity fixes, this topology
   mismatch is a leading suspect and should be handled in a separate
   architecture plan, not as a quick Stop-D patch.

5. BEVFusion fuser uses LiDAR input channel contract `256`; current `bb02d`
   uses `128`. The 0.075 parity probe sets `det-lidar-backbone-out=256` and
   `det-fusion-channels=256`, and the harness hard-checks the actual fuser
   LiDAR input channel count.

6. BEVFusion also differs in recipe: CBGS usage, schedule length, official
   sparse encoder/head stack, and framework implementation details. Mini cannot
   decide whether these improve full-data mAP/NDS in this codebase.

## Diagnostic Controls

The Arrhenius harnesses separate graph topology from training policy:

- `branch_topology=full_fusion`: camera and LiDAR enter fusion normally.
- `branch_topology=lidar_only`: camera BEV is zeroed and the camera branch is
  skipped in the diagnostic forward path.
- `branch_topology=camera_only`: LiDAR BEV is zeroed and the LiDAR branch is
  skipped in the diagnostic forward path.
- `train_policy=all_trainable`: no extra freezing beyond the config.
- `train_policy=camera_frozen`: camera backbone/neck/view-transform params are
  frozen.
- `train_policy=lidar_only_trainable`: camera modules are frozen while LiDAR,
  fusion, BEV neck, and head stay trainable.
- `train_policy=probe_no_backward`: forward/loss/memory probe only.

## Mini vs Full-Data Boundary

Mini can test:

- shape parity and hard reference-field assertions
- peak memory and OOM risk
- sparse voxel counts and coordinate ranges
- finite forward/backward/loss and GradScaler behavior
- trainable-module gradient coverage
- branch-delta sanity and fixed-batch tiny-overfit

Mini cannot support claims about:

- mAP/NDS/ASR/defense behavior
- whether 0.075m improves the final detector
- whether sparse 3D improves full-data rare-class AP
- best scientific config selection

When full nuScenes becomes available on Arrhenius, the minimal scientific matrix
should compare only the current `bb02d` 0.2m control, a fixed sparse 0.2m matched
recipe, and the 0.075 z=0.2 channel-256 parity candidate.

## Current Stop Results

All results below are engineering diagnostics on mini only and set
`scientific_claim=false` in their manifests.

Stop B shape/memory smoke:

- `221173`: `full_fusion_voxel020_fp16_current`, B=2, `grad_scale_init=1`,
  passed. Peak allocated memory was about 6.4 GiB.
- `221180`: `full_fusion_voxel020_fp16_current`, B=4, `grad_scale_init=1`,
  passed. Peak allocated memory was about 12.6 GiB.
- `221192`: `full_fusion_voxel075_z020_ch256_fp16_parity_probe`, B=2,
  `probe_no_backward`, passed. Peak allocated memory was about 59.2 GiB.
  Parity hard-check recorded and accepted:
  `voxel_size=[0.075,0.075,0.2]`,
  `range=[-54,-54,-5,54,54,3]`,
  `sparse_shape=[1440,1440,41]`, reference
  `max_voxels=[120000,160000]`, and fuser LiDAR channels `256`.
- `221199`: same 0.075 parity probe at B=4 failed with CUDA OOM in fusion
  GroupNorm. PyTorch reported about 83.7 GiB allocated and about 90.3 GiB
  process memory in use before a 7.91 GiB allocation. Treat 0.075 B=2 as the
  current feasible GH200 probe point unless memory-saving architecture changes
  are made.

Stop C sparse triage:

- `221071`/`221148`: `lidar_iso_voxel020_fp16_current_sparse` with the default
  `GradScaler(init_scale=512)` reached forward/backward but produced non-finite
  gradients on step 0. The detailed diagnostic isolated the non-finite values
  to `head.heatmap.weight`; other module gradients were finite. GradScaler now
  records the skip and drops scale from 512 to 256 before the harness fails.
- `221156`: same cell with explicit diagnostic `grad_scale_init=1` passed one
  optimizer step with finite loss and gradients.
- `221165`: same cell with `grad_scale_init=1` passed 5 fixed-batch steps and
  tiny-overfit decreased from about 144.3 to 32.4 with no GradScaler skips.
- `221166`: branch-delta sanity with `grad_scale_init=1` passed on the same
  batch. `full_fusion` and `lidar_only` had similar losses and LiDAR/fusion
  norms, while `camera_only` zeroed LiDAR as intended and had a much larger
  loss. This verifies branch isolation mechanics; it is not an accuracy claim.

Low-risk Stop D interpretation:

- The earlier CUDA index out-of-bounds was a diagnostic harness mismatch:
  fullshape cells forced 10 sweeps while the active mini cache was single-sweep.
  Cells now inherit sweeps from CLI/config by default, and multi-sweep must be
  enabled only when a matching cache is available.
- The sparse branch is not failing because of immediate NaN activations. It is
  sensitive to AMP loss scaling: `init_scale=512` overflows the first backward,
  while `init_scale=1` can train the mini fixed batch. This is a precision-policy
  question to validate on full data or a fixed trainval subset; do not silently
  compare scale-1 and scale-512 runs as the same precision cell.
- If full-data sparse performance remains poor after z/range/channel parity and
  explicit GradScaler policy, the next architecture work should target the
  mismatch with MIT's SECOND-style SparseEncoder xy/z downsampling rather than
  patching it inside this Stop D round.

## Stop E Gate Plan

Do not submit an uncapped full scientific run from this Stop E gate. The approved
capped cells are:

- `bb02d_020_control`: current 0.2m camera+LiDAR control.
- `sparse_020_matched`: sparse LiDAR branch at 0.2m, otherwise matched as
  closely as possible to the control.
- `sparse_075_z020_ch256_parity`: 0.075m xy, z voxel 0.2m, sparse z size 41,
  range `[-54,-54,-5,54,54,3]`, and fuser LiDAR channels 256.

Gate budgets:

- Gate 1: `max_steps=100` per cell.
- Gate 2: `max_steps=1000` per cell, only after Gate 1 passes.
- Profiling/teardown is intentionally deferred until full trainval data is
  available and the gate smoke has passed.

Precision policy:

- Non-sparse fp16 keeps `GradScaler(init_scale=512)`.
- Sparse voxel fp16 explicitly uses `det-sparse-grad-scale-init=1.0` in the
  gate configs. This value is part of the precision policy and must be recorded
  in provenance/manifests.

The launcher is `fl_v3/scripts/run_arrhenius_stop_e_gate.sh`. It preflights the
trainval dataroot and msweep10 info-cache and refuses to run if they are absent.
As of this audit update, Arrhenius still only has confirmed mini data in the
project data tree, so the full-data Gate 1/2 jobs are prepared but not runnable.
