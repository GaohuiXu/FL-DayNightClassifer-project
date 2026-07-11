# S03 HANDOFF — camera branch architecture

## Session identity and self-assessment

- Session: `S03`, Camera branch architecture.
- Worker self-assessment: **CHANGES-REQUESTED — IMPLEMENTATION DELIVERED, FOCUSED
  RUNTIME GATE NOT EXECUTED**.
- Base: `372de9398ae435f82b83367a922fd302c0635738`.
- Source branch named by kickoff: `codex/s00-orchestra-ledger`.
- Initial state: clean detached HEAD at the exact base.
- Owner-authorized worker branch: `codex/s03-camera-architecture`.
- Implementation commit:
  `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`.
- Initial request commit:
  `9a30e2470c98a3495eeaa8558bb6f2ff52db774d`.
- Remediated executable/request commit:
  `871db182c5fdcdda46e242d911ac9dcbf393683a`.
- Final documentation delivery commit / `WORKER_SHA`: returned in the session
  response after committing this file; a commit cannot embed its own SHA.
- No merge, push, PR, upload, branch deletion, or worktree operation occurred.

S00 approved one exact O-009 submission at executable `871db18`.  The command was
invoked exactly once but Slurm rejected it before job creation due to an invalid
account/partition combination.  There was no retry, job ID, output, allocation, or
GPU use.  `RESULTS.md` preserves the negative result and missing evidence.  S00
then authorized preparation only of a scheduler-remediation request.  The launcher
now adds the active Arrhenius account/partition directives and uses a new output
identity; no second submission occurred and fresh compute approval remains pending.

## Scope and files

Modified only within the S03 envelope:

- `fl_v3/src/fl_v3/models/fusion/camera_backbone.py`;
- `fl_v3/src/fl_v3/models/fusion/camera_neck.py`;
- `fl_v3/src/fl_v3/models/fusion/preprocess.py`;
- `fl_v3/src/fl_v3/models/fusion/view_transform.py`;
- `fl_v3/tests/test_s03_camera_contract.py`;
- `fl_v3/usenix27_orchestra/handoffs/S03/{RUN_REQUEST,RESULTS,HANDOFF}.md`;
- `fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh`.

`swin_sdpa.py` was inspected but not modified.  `detector.py`, `training/tasks.py`,
`bev_grid.py`, all S02/S04/S05 files, canonical Orchestra documents,
`fl_v3/collab/`, and `fl_v2/` remained read-only.

## Architecture and semantic changes

### Swin-T and all-level stride-8 FPN

- `CameraBackbone` continues to expose real Swin-T taps at strides
  `(4, 8, 16, 32)` with channels `(96, 192, 384, 768)` and now publishes an
  explicit output contract.
- `GeneralizedLSSFPN` validates the declared channel/stride list and consumes every
  level.  Each tap receives its own lateral Conv+GroupNorm+ReLU; stride 4 is
  integer-average-pooled, stride 8 retained, and stride 16/32 bilinearly upsampled
  with `align_corners=False` to the stride-8 target.  The sum enters one smoothing
  block.
- The old graph returned only `lats[out_level]`, so shallower levels could be
  computed but never affect a stride-16 output.  Old checkpoints can load the same
  parameter names/shapes but are not semantically valid initializers for the new
  FPN: previously disconnected laterals were not trained for this output and the
  image/depth geometry also changes.  Retraining is required.

### Aspect-preserving image geometry and calibration

`ImageAugmentationConfig` implements the MIT-reference-style policy:

- train scalar resize `[0.38, 0.55]`, bottom crop limits `[0,0]`, random horizontal
  flip, and image-plane rotation `[-5.4°, 5.4°]`;
- validation scalar resize `0.48`, bottom-aligned and horizontally centered crop,
  no flip, and no rotation;
- caller-supplied CPU `torch.Generator` enables exact training replay; explicit
  per-camera parameters support durable fixtures.

Each camera composes one native-pixel to augmented-pixel affine in this order:

1. scalar aspect-preserving resize with integer realized dimensions and the exact
   `align_corners=False` half-pixel offsets;
2. crop or zero-pad, including negative crop origins;
3. horizontal flip using `u'=(W_out-1)-u`;
4. in-plane rotation around `((W_out-1)/2,(H_out-1)/2)` in image coordinates.

For affine `A`, calibration updates are computed in float64 then stored in the
input calibration dtype:

```text
cam_intrinsics' = A @ cam_intrinsics
lidar2img' = embed4(A) @ lidar2img
```

The preprocessor returns normalized float32 images, updated calibration,
`image_aug_matrix`, replay parameters, stable parameter-field names, and a geometry
mode.  `augmentation=None` deliberately retains the old stretch path for unwired
callers because S03 was forbidden to edit detector/config integration.  S07-B must
explicitly construct `ImageAugmentationConfig`; silently using the compatibility
default does not satisfy O-017.

### Pure-camera LSS and dtype contract

- Primary constructor contract: `image_hw=(256,704)`, `feat_stride=8`, feature
  size `32x88`, `depth_bins=(1.0,60.0,0.5)` for 118 bins, and 80 context channels.
- The view-transform API accepts camera features, `lidar2img`, `B`, and `N` only.
  `LIDAR_TOP` defines the metric output frame; no LiDAR points, projected point
  depth, LiDAR BEV, or cross-conditioned features enter the module.
- BEV layout remains the read-only shared convention `[B,C,H=y,W=x]` with shape
  `[B,context_channels,cfg.ny,cfg.nx]`.
- Both strict and relaxed splat reductions accumulate in fp32.
  `bev_output_dtype="input"` casts the integration tensor back to the camera
  feature dtype; `"float32"` retains fp32.  The compatibility default is fp32, so
  S07-B must explicitly select the reviewed integrated dtype/precision policy.
- Input feature and calibration shapes now fail closed; `output_contract()` records
  shapes, bins, accumulation/output dtype, layout, and the no-LiDAR boundary.

## Coordinate assumptions

- Pixel centres use integer coordinates with `u` right and `v` down.
- Native `lidar2img` maps keyframe `LIDAR_TOP` coordinates
  `(x forward, y left, z up, 1)` to homogeneous `[u*d,v*d,d,1]`.
- The augmented frustum uses the updated projection matrix, then its inverse maps
  frustum samples back to `LIDAR_TOP`.
- BEV continues to use `W -> x`, `H -> y`, floor binning, metric metres, and the
  range/grid supplied by the read-only `BEVConfig`.
- S07-B must reconcile the camera `BEVConfig` with the accepted S04 low-resolution
  LiDAR contract.  S03 does not change `bev_grid.py` or assume an unreviewed final
  grid size.

## Reference mapping

Reference revision: MIT BEVFusion archived `main` at
`326653dc06e0938edf1aae7d01efcd158ba83de5`.

- Camera-only Swin config:
  `configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml`.
- Augmentation implementation:
  `mmdet3d/datasets/pipelines/transforms_3d.py::ImageAug3D`.
- Multi-scale neck:
  `mmdet3d/models/necks/generalized_lss.py`.
- Pure-camera lift-splat:
  `mmdet3d/models/vtransforms/lss.py` and `base.py`.

The official camera config declares Swin stride-8/16/32 outputs; S03 retains the
existing stride-4 tap and explicitly fuses it because O-017 requires no permanently
disconnected intended level.  The pure-camera LSS mapping is used; the separate
reference `DepthLSSTransform` that injects projected LiDAR points is intentionally
not adopted.

## Test and execution evidence

Static checks passed:

```text
python3 -m py_compile <four modified camera modules> fl_v3/tests/test_s03_camera_contract.py
python3 AST parse of the same sources
bash -n fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh
git diff --check
```

The login interpreter lacks torch, torchvision, and pytest, so no local pytest PASS
is claimed.

The approved exact-once submission failed before job creation:

```text
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```

Post-attempt `squeue` and `sacct` were empty; output/log roots were absent; actual
GPU-hours were zero.  The committed launcher omitted the account and partition used
by all inspected active Arrhenius GPU launchers.  No runtime test ran and no retry
was attempted.

The new unexecuted scheduler-remediation candidate changes only the launcher by
adding `#SBATCH -A naiss2025-22-1113-gpu` and `#SBATCH -p gpu`, plus the required
fresh output identity.  Its launcher SHA-256 is
`d6f236d35f290b4552f3c3e93bb2d92438481100c8fa7726812ea0d658d12983`, its
unchanged source-list SHA-256 is
`d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`, and its
source-state SHA-256 is
`6163d27c7f264902a1ac7688b4a13a704d2b98fc6597ca39c0da8b2a115157c1`.

## Gate checklist

| Gate | Worker status | Evidence / missing work |
|---|---|---|
| Swin-T taps and valid stride-8 output | IMPLEMENTED / STATIC ONLY | Source contract and focused tests; no torch execution. |
| No permanently disconnected intended level | TEST AUTHORED, NOT RUN | Per-level input gradients and every neck parameter checked by test. |
| 0.5 m depth bins | IMPLEMENTED / STATIC ONLY | 118-bin constructor and contract fixture; no runtime. |
| Exact resize/crop/pad/flip/rotation calibration | TEST AUTHORED, NOT RUN | Four independent scalar residual fixtures avoid reusing implementation affine as oracle. |
| Deterministic validation geometry | TEST AUTHORED, NOT RUN | Native 1600x900 golden plus repeated outputs. |
| Complete intended parameter finite gradients | TEST AUTHORED, NOT RUN | Full Swin/FPN/LSS CUDA fp16-autocast backward was not submitted. |
| Camera feature/pixel sensitivity | TEST AUTHORED, NOT RUN | No runtime result. |
| LiDAR invariance | STATIC API PASS; RUNTIME NOT RUN | No LiDAR input in signature; hostile keyword/repeated-output test not executed. |
| Explicit shape/dtype/config contract | PASS STATIC | Module contract methods and fail-closed validation compile. |
| Tiny-overfit / 100-step loss decrease | NOT AUTHORIZED / NOT RUN | Must remain a later exact gate. |
| Memory profile | NOT AUTHORIZED / NOT RUN | Arithmetic only; no measured CUDA allocation. |

## Memory implications

At `B=1`, six cameras, `C=80`, `D=118`, and `32x88`, an unmasked lift contains
`159,498,240` elements: `304.22 MiB` fp16 or `608.44 MiB` fp32.  This is 8x the
old stride-16/1 m lift.  The relaxed mask-then-lift path avoids materializing all
elements and returns the configured output dtype after fp32 accumulation; strict
mode materializes the full lift.  These figures are exact tensor arithmetic, not
a GH200 measurement or throughput claim.

## Allowed and forbidden interpretations

Allowed:

- S03 delivers an independently reviewable implementation and explicit S07-B
  integration contract within file ownership;
- static syntax/AST/diff/launcher checks pass;
- one approved submission failed before job creation for the recorded Slurm
  account/partition reason, with zero GPU use.

Forbidden:

- claiming projection residual, gradient, sensitivity, invariance, deterministic
  validation, CUDA fp16, memory, or tiny-overfit runtime PASS;
- claiming camera/model/full-data readiness, accuracy, mAP/NDS, fusion gain, FL,
  attack/defense, generalization, scientific, or publication evidence;
- using mini/static/arithmetic evidence as model-quality evidence;
- reusing the consumed exact-once approval for the corrected launcher or treating
  the prepared request as authorization to submit.

## Remaining risks and requested S00 decisions

1. Independent S03-R should review the implementation and authored fixtures even
   though runtime evidence is missing; it must not convert authored tests into a
   PASS.
2. If runtime evidence is still required before review/integration, S00 must audit
   and explicitly approve the new scheduler-remediation executable/request hashes.
   It remains a fresh exact-once action, not a retry under the consumed approval.
3. S07-B must explicitly wire reference augmentation, stride 8, 0.5 m bins,
   `bev_output_dtype`, and the reviewed common BEV geometry.  Current unwired
   detector/task defaults remain legacy and do not satisfy O-017 by themselves.
4. Old checkpoints require retraining under the new FPN/image/depth semantics.
