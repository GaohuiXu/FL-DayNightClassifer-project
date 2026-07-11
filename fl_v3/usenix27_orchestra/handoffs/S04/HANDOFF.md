# S04 HANDOFF — LiDAR SECOND architecture

## Session identity and current verdict

- Session: `S04`.
- Base: `372de9398ae435f82b83367a922fd302c0635738`.
- Source branch: `codex/s00-orchestra-ledger`; verified kickoff was clean detached
  at the exact base.
- Worker branch: `codex/s04-lidar-second` (owner-authorized by O-017).
- Initial implementation: `20d11e284f20fced3dbc33e7ac105c845da708a5`.
- Failed-job executable: `49efb05dd341dbfbcc2d373508772e5b214aa726`.
- Manual sparse-composition remediation: `2b5cf2f` (full SHA is in Git history and
  the returned session report).
- Scoped final-output dtype code/tests: `72184e9ed3d2a9ea4fcd9f1a8dc473312a09a52d`.
- Attested immutable-snapshot executable:
  `2729f45144053e1b554a0bf04640b8bbc1ff43e4`.
- Final request-delivery HEAD is returned after its commit; a commit cannot embed
  its own SHA without changing it.
- Job-336718 evidence delivery is returned after the docs commit.
- Worker self-assessment: **CHANGES-REQUESTED / DTYPE AND B=4 PASS, EVAL-REUSE
  SPCONV BLOCKER**.

This is not an integration PASS. Jobs `335566`, `335579`, and `336718` all remain
visible failed negatives. Job `336718` validated the original fp16 output assertion
and the complete B=4 dtype/forward/backward/memory case, but the added same-model
train/backward-to-eval non-empty reuse sub-check failed inside the spconv tuner.
No retry is authorized. Independent S04-R plus Orchestra disposition remain
required.

## Delivered architecture contract

- Canonical point frame: `LIDAR_TOP`, metres; sparse indices `(b,z,y,x)` and
  spatial shape `(z,y,x)`; dense BEV `[B,C,H=y,W=x]`.
- Primary voxel/range: `(0.075,0.075,0.2)m`,
  `[-54,-54,-5,54,54,3]`, input shape `(41,1440,1440)` including one z padding
  bin.
- Sparse flow:
  - stem `(41,1440,1440)`;
  - stage 1 stride `(2,2,2)` -> `(21,720,720)`;
  - stage 2 stride `(2,2,2)` -> `(11,360,360)`;
  - stage 3 stride `(2,2,2)`, padding `(0,1,1)` -> `(5,180,180)`;
  - stage 4 remains `(5,180,180)`;
  - z-only `(3,1,1)/(2,1,1)` -> `(2,180,180)`.
- Densification occurs exactly once after the last sparse stage at
  `[B,128,2,180,180]`; z collapses to `[B,256,180,180]`. The sparse backbone has
  no `.dense()` call, and reduced occupancy also returns `180x180`.
- Output stride/cell: XY stride 8, `0.6m` output cells, origin `(-54,-54)`,
  `H->y/W->x`; S07 camera-fusion integration must use this exact mapping.
- Theoretical receptive field: `(z,y,x)=(153,137,137)` fine voxels; XY is
  `10.275m x 10.275m`.
- Per-sample canonical voxelization; separate default train/eval caps
  `120000/160000`; observable fields include input/valid/unique-before-cap/kept/
  dropped voxels and point-cap drops.
- Voxelization/mean VFE stays fp32. Sparse fp16 is CUDA-only and opt-in;
  fp32 reference remains available; bf16 remains rejected by precision policy.

## Official reference mapping and intentional deviations

Primary references:

- MIT BEVFusion `voxelnet_0p075.yaml`:
  <https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml>;
- MIT BEVFusion/OpenMMLab `SparseEncoder`:
  <https://github.com/mit-han-lab/bevfusion/blob/main/mmdet3d/models/backbones/sparse_encoder.py>.

Mapped fields are voxel/range, `[120000,160000]` caps, channel groups
`[[16,16,32],[32,32,64],[64,64,128],[128,128]]`, three sparse XY reductions,
stage-3 z padding, 128-channel z-collapse output, and dense z-to-channel collapse.

Intentional framework-independent differences: no mmdet3d/mmcv dependency; spconv
2.3.8 modules directly; per-voxel GroupNorm avoids cross-sample BatchNorm coupling;
per-sample canonical hard voxelization exposes exact truncation. After Job 335566,
custom sparse residuals are explicitly forwarded rather than embedded as arbitrary
modules in `spconv.SparseSequential`.

## Files and commits

Changed/added within S04 ownership:

- `fl_v3/src/fl_v3/models/fusion/second_sparse_backbone.py`;
- `fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py`;
- `fl_v3/tests/test_sparse_voxel_encoder.py`;
- `fl_v3/tests/test_s04_second_contract.py`;
- `fl_v3/tests/test_s04_second_smoke.py`;
- `fl_v3/usenix27_orchestra/handoffs/S04/{run_s04_second_smoke.sh,RUN_REQUEST.md,RESULTS.md,HANDOFF.md}`.

Commit history:

- `20d11e2`: implementation and tests;
- `a201245`, `5676ff6`, `49efb05`: bounded launcher/request-binding and
  forward/backward-only scope remediation;
- `2b5cf2f`: manual sparse residual composition fix after Job 335566.
- `72184e9`: explicit active sparse-AMP BEV output dtype contract, dtype tracing,
  and retained assertions.
- `2729f45`: verify actual snapshot working/submit directories and immutable
  identity content/hash binding executable SHA/tree plus source/request hashes.

No `lidar_encoder.py`, `lidar_backbone.py`, `bev_grid.py`, `detector.py`,
`training/tasks.py`, canonical Orchestra file, `fl_v3/collab/`, or `fl_v2/` file
was modified. No merge, push, PR, upload, or publication occurred.

## Verification and negative result

Local/static pre-job checks passed: Python compile, shell syntax, diff whitespace,
fusion AST ban/stable-sort audit, and exact ten-test inventory.

Exact-once O-009 job `335566`:

- identity and artifact checksum gates PASS;
- scheduler `FAILED 1:0`, elapsed `00:01:41`, one GH200/eight CPUs, restarts 0;
- JUnit `10 tests / 5 failures / 0 errors / 0 skips`;
- five CPU/static tests PASS;
- every real spconv test FAILS with the same `_SparseResidualBlock` Tensor versus
  SparseConvTensor composition bug;
- B=4 never reaches a completed output/backward/memory record;
- no retry/follow-on is authorized or performed.

Complete scheduler fields, failed cases, artifact/log hashes and interpretation
limits are in `RESULTS.md`. The approved request hash was
`00aea9398736471b3a68a1e1fade00fb7e639457795109cc8d9ad6971c956b7c`;
stdout/stderr hashes are `a8bd2475...18c7` / `ae633085...b57`.

Post-job remediation is local/static only. It explicitly forwards sparse residual
stages and adds a structural regression inside the first real-spconv fixture.

Exact-once remediation Job `335579` then validated that correction but found the
next blocker:

- scheduler `FAILED 1:0`, elapsed `00:00:46`, restarts 0;
- JUnit `10 tests / 2 failures / 0 errors / 0 skips`;
- eight passed, including real spconv shape/backward, per-sample caps/extreme
  occupancy, empty input and sample/batch isolation;
- two failures because `sparse_conv_fp16=True` returns final BEV
  `torch.float32`, not required `torch.float16`;
- B=4 reached correct `[4,256,180,180]`, loss/backward and finite gradients, but
  failed before recording peak CUDA memory.

At that stage no third job, retry, requeue, resubmission, or follow-on had occurred.
Full Job 335579 scheduler/artifact hashes are in `RESULTS.md`.

S00 then authorized implementation/request preparation only. Commit `72184e9`
records the fp32 projection boundary and converts only the active sparse-AMP final
BEV to fp16, matching the existing empty-input path while leaving the fp32
reference unchanged. Existing ten tests retain their dtype assertions and now
also assert the pre-cast/output trace and empty/non-empty consistency. Local
`py_compile`, `bash -n`, exact ten-test inventory, and `git diff --check` pass;
login-node pytest was unavailable (`/usr/bin/python3` has no pytest), so this is
not runtime evidence. After S00 found that `--chdir` alone does not guarantee
`SLURM_SUBMIT_DIR`, commit `2729f45` additionally binds both actual directories and
an immutable SHA/tree/source/request identity file. `RUN_REQUEST.md` proposed the
same ten synthetic tests from that `/nobackup` snapshot; S00 later approved and
consumed that exact request once as recorded below.

Exact-once Job `336718` then ran from the attested read-only snapshot:

- scheduler `FAILED 1:0`, `00:02:53`, node `n593`, restarts 0;
- actual allocation exactly one GH200, eight CPUs, one node;
- JUnit `10 tests / 1 failure / 0 errors / 0 skips`, `9 passed`;
- original fp32/fp16 output assertions passed;
- B=4 passed output/dense shapes, fp16 dtype, loss/backward, finite gradients and
  memory bounds; peak allocated/reserved were `1,017,576,960` / `1,109,393,408`
  bytes on a `102,005,473,280`-byte device;
- the sole failure was the added second non-empty fp16 inference after reusing the
  train/backward model in eval mode; spconv `ConvTunerSimple` reported no suitable
  algorithm for the six-voxel SubMConv input before empty/non-empty comparison;
- execution/source/request/snapshot identity and all artifact checksums passed.

No retry, requeue, resubmission, modification, or follow-on occurred. Full raw
hashes and interpretation limits are in `RESULTS.md`.

## Gate checklist

| S04 item | Worker status | Evidence / limit |
|---|---|---|
| coordinate/shape/stride/RF declaration | PASS static | exact golden fixtures passed |
| no fine-grid densification | PASS static | one encoder `.dense()` after `(2,180,180)`; none in sparse backbone |
| train/eval per-sample caps and truncation | PASS bounded synthetic runtime | Job 335579 |
| empty/extreme occupancy | PASS bounded synthetic runtime | Job 335579 |
| metric/camera-fusion mapping | PASS static | 0.6m, 180x180 golden |
| sample/batch isolation | PASS bounded synthetic runtime | Job 335579 |
| fp32/fp16 behavior | PARTIAL PASS / BLOCKED | original dtype/finite/backward assertions and B=4 trace pass; same-model eval reuse fails in spconv before empty/non-empty comparison |
| B=4 forward/backward | PASS bounded synthetic runtime | Job 336718, correct shape/fp16/loss/backward/finite grads |
| B=4 bounded memory | PASS bounded observation | Job 336718, allocated/reserved/device bytes recorded above |
| sparse composition remediation | PASS bounded synthetic runtime | explicit forwarding + Job 335579 |
| independent S04-R | NOT STARTED | S00/owner controls reviewer launch |

## Allowed and forbidden interpretations

Allowed:

- the design/static contract and exact negative result may be cited as engineering
  evidence;
- Job 335566 identified a concrete spconv composition incompatibility;
- the manual fix preserves sparse tensor flow and reduced-resolution densification
  at source level;
- Job 336718 provides bounded evidence for the exact original fp16-output and B=4
  dtype/shape/backward/memory cases listed above, while preserving its later
  lifecycle failure.

Forbidden:

- S04 PASS, integration readiness, or runtime acceptance of the remediation;
- any overall S04 PASS, or extrapolating the bounded fp16/B=4 observations beyond
  executable `2729f45` and Job 336718;
- production detector/model/full-data readiness, mAP/NDS, model superiority,
  voxel-size selection, throughput/profile, convergence, fusion gain, FL,
  attack/defense, generalization, or publication claim.

## Remaining actions for S00/owner

1. Preserve Jobs 335566 (composition), 335579 (fp16 output dtype), and 336718
   (same-model eval reuse spconv tuning) as failed negatives.
2. Independently review whether Job 336718 exposes an implementation lifecycle
   defect, a spconv cache/tuner limitation, or a fixture sequencing problem; do
   not reinterpret the nine passes as overall runtime PASS.
3. Only after Orchestra disposition and independent S04-R may S07-B consider this
   module for integration.
