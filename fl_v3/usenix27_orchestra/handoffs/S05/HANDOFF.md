# S05 HANDOFF — multi-task CenterHead and deterministic decode/NMS

## Session identity and worker self-assessment

- Session: `S05`.
- Base SHA: `372de9398ae435f82b83367a922fd302c0635738`.
- S07-A foundation SHA named by kickoff:
  `0249eb21a32730ac1689255491b19a158711401f`.
- Source branch named by kickoff: `codex/s00-orchestra-ledger`.
- Expected/observed startup mode: clean detached HEAD at the exact base.
- Owner-authorized worker branch: `codex/s05-centerhead-decode`.
- Implementation/test commit:
  `9fd3281651ef006a175ed9462e7bf1eaf3437357`.
- Final delivery SHA: the commit containing this handoff; returned to S00 in the
  task response because a commit cannot embed its own SHA without changing it.
- Worker self-assessment: **PASS for the scoped implementation and static gates;
  runtime Torch/pytest execution and full-stack integration remain NOT RUN**.
  This is not an independent review, S07-B PASS, model-readiness PASS, or
  scientific PASS.

No Slurm/srun, model training, mini traversal, full trainval access, metric,
profile, merge, push, PR, upload, or publication action occurred.

## O-018 amendment acknowledgement

S05 explicitly ACKed and followed owner/S00 amendment `O-018`, recorded in S00
canonical commit `a59fcc549ba62cc0c00fc8fe20c36063ca6f4648`:

1. the immutable primary reference is MIT BEVFusion archived HEAD
   `326653dc06e0938edf1aae7d01efcd158ba83de5`;
2. CenterPoint v0.2 tag
   `e9ef04c3715aa3342fa42f4f4e064db987def6ad` is cross-check evidence only;
3. official tasks, score/range, circle/rotate types and scales, NMS budgets, and
   rotate-IoU threshold remain unchanged;
4. candidate selection keeps K=500 independently per class and removes only the
   official coder's second task-wide K=500;
5. deterministic ties are score descending, canonical global class ID ascending,
   flattened spatial index ascending;
6. GroupNorm replaces official BatchNorm while the shared-conv and independent
   two-convolution per-task field topology remain fixed;
7. task-local labels map to canonical devkit IDs by class name, never cumulative
   task offset;
8. this implementation is named **reference-faithful no-starvation adaptation**.

Allowed interpretation of O-018: single-class candidate selection has official
parity and multi-class tasks cannot lose a class solely to the removed task-wide
top-K. Forbidden interpretation: multi-class decode is element-wise identical to
the official coder. Official task-wide NMS can still suppress an overlapping
lower-priority box from another class, and the task post-NMS budget remains 83;
O-018 removes candidate starvation, not those official NMS semantics.

## Frozen reference and exact semantics

Primary reference files at MIT BEVFusion commit `326653dc...`:

| Reference file | SHA-256 of raw immutable file |
|---|---|
| `configs/nuscenes/det/centerhead/default.yaml` | `3425d0b897d240c41ab49406bc84f340cec4fff960cbd2028673fd120f7e7826` |
| `configs/nuscenes/det/centerhead/lssfpn/camera/default.yaml` | `61af3caec8cb781e8ee2152f7e4a9496f2b823fe40cf4bb97f28818d94537898` |
| `mmdet3d/models/heads/bbox/centerpoint.py` | `3d5b3d06f76df952955fe7566b58913741ef7033f323d460af7c7a2ebecea7b6` |
| `mmdet3d/core/bbox/coders/centerpoint_bbox_coders.py` | `9b2924117b5149d42470c4c866068a979a811e535f1b54b54caa7b6e241ce0e2` |
| `mmdet3d/core/post_processing/box3d_nms.py` | `f466f0af8b178c2f42c467ade46148b07ad0c37cb5a26064527c83f62f9731c2` |

Task contract, in official task order:

| Task | Classes | NMS | circle threshold (squared m) | dimension scale |
|---:|---|---|---:|---|
| 0 | `car` | circle | 4.0 | `[1.0]` |
| 1 | `truck`, `construction_vehicle` | rotate | 12.0 (unused) | `[1.0,1.0]` |
| 2 | `bus`, `trailer` | rotate | 10.0 (unused) | `[1.0,1.0]` |
| 3 | `barrier` | circle | 1.0 | `[1.0]` |
| 4 | `motorcycle`, `bicycle` | rotate | 0.85 (unused) | `[1.0,1.0]` |
| 5 | `pedestrian`, `traffic_cone` | rotate | 0.175 (unused) | `[2.5,4.0]` |

Other frozen decode fields:

- strict score comparison `score > 0.1`;
- post-center range `[-61.2,-61.2,-10,61.2,61.2,10]` metres;
- per-class pre-K 500; no second task-wide K;
- NMS `pre_max_size=1000`, `post_max_size=83` per task;
- rotated BEV IoU threshold `0.2`;
- no pre-decode local-max filter, matching the fixed bbox coder;
- at most `6 * 83 = 498` decoded boxes, below the official nuScenes
  `max_boxes_per_sample=500`.

## Implemented contract

### Head topology

`models/fusion/head.py` now provides:

- one shared `3x3 Conv -> GroupNorm -> ReLU` transform;
- six `SeparateTaskHead` instances in official task order;
- independent two-convolution `heatmap`, `reg`, `height`, `dim`, `rot`, and `vel`
  branches for every task;
- heatmap final bias `-2.19` for every task;
- fail-closed validation for duplicate/unknown classes, class-count mismatch, and
  any non-two-convolution request.

`forward` returns `List[Dict[str,Tensor]]` in task order. This is a deliberate new
module interface. Production detector/loss wiring was forbidden to S05 and is an
explicit S07-B integration requirement below.

### Candidate selection, encode/decode, and label mapping

`models/fusion/centerhead_decode.py` provides:

- immutable task/decode specs for all thresholds and budgets;
- task-local to canonical global ID mapping by class name:
  `[(0,), (1,4), (2,3), (9,), (6,7), (5,8)]`;
- regression encoding for canonical gravity-center boxes without duplicating
  S02-owned Gaussian rendering;
- per-class stable K=500 selection, strict score thresholding, and the O-018 total
  tie order;
- decode of offset/z/log-dim/sin-cos/velocity to
  `(cx,cy,cz,l,w,h,yaw)` plus LiDAR-frame `(vx,vy)`;
- inclusive post-center-range and finite/positive-dimension filtering;
- official task-wide NMS dispatch and deterministic final merge.

### Deterministic NMS

`models/fusion/nms_deterministic.py` replaces unavailable numba/mmcv kernels with:

- CPU float64 deterministic circle NMS using official squared-distance and
  inclusive `<= threshold` suppression;
- CPU float64 rotated rectangle intersection and task-wide IoU NMS;
- the same O-018 content order before pre-NMS truncation;
- explicit pre/post budgets and fail-closed finite/positive geometry checks.

The pure-Python rotated implementation is correctness-first and has not been
profiled at production candidate volume. S07-B must profile it before a full run;
performance optimization may not change the frozen geometry/tie fixtures.

### Official nuScenes conversion hardening

`eval/box_to_global.py` and `eval/detection_eval.py` now:

- reject non-finite/non-positive canonical boxes;
- reject mismatched decoded array lengths/shapes and invalid global labels rather
  than silently dropping them;
- enforce official `max_boxes_per_sample=500`;
- reject duplicate eval tokens and duplicate sample decode records;
- retain the existing canonical conversion: gravity center to global,
  `size=(w,l,h)`, rigid yaw lift, and rigid LiDAR velocity lift;
- retain content-defined submission ordering for equal-score permutation
  invariance.

## Files changed

Modified:

- `fl_v3/src/fl_v3/models/fusion/head.py`
- `fl_v3/src/fl_v3/eval/box_to_global.py`
- `fl_v3/src/fl_v3/eval/detection_eval.py`
- `fl_v3/tests/test_head_capacity.py`

Added:

- `fl_v3/src/fl_v3/models/fusion/centerhead_decode.py`
- `fl_v3/src/fl_v3/models/fusion/nms_deterministic.py`
- `fl_v3/tests/test_s05_centerhead_decode.py`
- `fl_v3/tests/test_s05_nms.py`
- `fl_v3/tests/test_s05_eval_roundtrip.py`
- `fl_v3/usenix27_orchestra/handoffs/S05/HANDOFF.md`

No diff exists in `losses.py`, `detector.py`, `training/tasks.py`, `bev_grid.py`,
canonical Orchestra documents, S01/S07 artifacts, `fl_v3/collab/`, or `fl_v2/`.

## Verification evidence

### Executed local/static checks

1. `python3 -m py_compile` over all changed/added Python source and S05 tests,
   plus existing `test_eval_box_to_global.py` and
   `test_eval_detection_eval.py`: **PASS** (no output).
2. `git diff --check`: **PASS** (no output), before the implementation commit and
   again after final edits.
3. Conflict-marker scan over owned source/tests: **PASS** (no matches).
4. Actual-source AST extraction of the committed rotated-IoU functions, then
   identity/disjoint/perpendicular `IoU=1/0/1/3` fixtures using stdlib Python:
   `actual-source rotated_iou fixtures: PASS`.
5. Ownership audit from `git status`/diff: **PASS**, only envelope-owned paths.

### Authored but not executed runtime fixtures

The changed test set contains 26 test functions (27 pytest cases because one is
parameterized) covering:

- six-task topology, independent two-convolution fields, bias, GN batch isolation;
- exact class-name mapping including construction vehicle, bus, barrier,
  pedestrian, and traffic cone;
- 500 higher-scoring common-class candidates plus one retained tail candidate;
- equal-score global-class/spatial tie order;
- single-class official candidate parity;
- canonical encode/decode box, yaw, velocity, and class round trip;
- B=1/B>1 and batch permutation;
- circle squared-metre boundary, duplicate circle/rotate boxes, task-wide
  cross-class rotate suppression, NMS input permutation, and NMS budgets;
- known rotated IoUs;
- local/global/eval round trip, official wlh/velocity/yaw conversion, content
  permutation, invalid labels/geometry, duplicate samples, and the 500-box cap.

The x86_64 login interpreter reports `ModuleNotFoundError` for both `torch` and
`pytest`. The validated project environment is aarch64/GH200 and cannot be treated
as a login-node environment. O-018 preserved `APPROVED_COMPUTE: none`; therefore
S05 did not submit Slurm and does **not** claim these runtime fixtures passed.
Independent S05-R/S07-B should execute them in an authorized dependency-complete
runtime.

## File hashes at implementation commit

| File | SHA-256 |
|---|---|
| `models/fusion/head.py` | `3b731ba4048c95f67c6bc9238863247232a92623a85b1d0982ed7c6b7a4633c1` |
| `models/fusion/centerhead_decode.py` | `834bac66758224813c7a254307e5e8a80e07c3c9899fdb3465acc201d367af47` |
| `models/fusion/nms_deterministic.py` | `2d6f2b0aad7c0e31aa6cbfae170dfe5a52c5982d15de99d4b74ea21f7e91b22d` |
| `eval/box_to_global.py` | `d7192ffe3d3646c05b1b9b955696263e161aff61a15157c0c5a75b6944cd55b9` |
| `eval/detection_eval.py` | `99867c68c8ebfd86556e8bb16bace629df789cf92881f3a392853858293d99f2` |
| `tests/test_head_capacity.py` | `0d4d7600a3a1da7bbb3674c89823069ed17ab67d54246d95ef189e3012b3daa8` |
| `tests/test_s05_centerhead_decode.py` | `35ce4684375a5e02a98fb1c053e929ac4ea2d94ebabb4c00bd29735c979bb915` |
| `tests/test_s05_nms.py` | `bc2b30bdb9c9e3c22ac42d355730d88f8fcb7a359aeb232596190b9b30a2e0c1` |
| `tests/test_s05_eval_roundtrip.py` | `292fa8bc48b7038444a89a2797c218a2ea0519f82845e1badff135aebc3d8ff8` |

## Gate checklist

| S05 gate | Worker status | Evidence / boundary |
|---|---|---|
| fixed reference/tasks/fields | PASS (code/static) | immutable reference hashes, six independent task heads |
| no global/cross-class top-K starvation | PASS (code + authored hostile fixture) | per-class 500; no second task K; tail fixture authored, runtime not executed |
| deterministic tie order | PASS (code + authored fixture) | score/global-ID/flat-index total order |
| circle NMS semantics | PASS (code/static) | squared metres, inclusive comparison; runtime fixture pending |
| rotate NMS geometry/determinism | PASS (actual-source geometry smoke) | known IoUs pass; Torch NMS fixtures not run |
| duplicate/input permutation | IMPLEMENTED / RUNTIME NOT RUN | circle/rotate and content-permutation fixtures authored |
| box/yaw/velocity encode/decode | IMPLEMENTED / RUNTIME NOT RUN | canonical round-trip fixture authored |
| local/global/eval/submission conversion | IMPLEMENTED / RUNTIME NOT RUN | class/wlh/yaw/velocity/cap fixtures authored |
| task-local/global label mapping | PASS (code/static) | explicit name map; no cumulative offset code path |
| loss/target consistency | INTERFACE FROZEN / INTEGRATION PENDING | regression encoder matches field order; Gaussian/losses.py remained S02-owned |
| production detector/task wiring | NOT DONE BY CONTRACT | S07-B integration requirement |
| material compute/scientific metric | NOT RUN / FORBIDDEN | no request or job |
| independent S05-R | PENDING | worker self-assessment only |

## Required S07-B integration work

1. Update detector construction from the legacy `conv_layers=1` single-dict head
   to the frozen two-convolution list-of-task-dicts interface.
2. Route production decode through `decode_centerhead`; remove the old detector
   global 10-class top-K path only during reviewed integration.
3. Reconcile S02's independently reviewed Gaussian/loss implementation with the
   same task-name mapping and field order without cherry-picking conflicting
   `losses.py` semantics from S05 (S05 has no such diff).
4. Make resolved config/provenance bind all task/decode/NMS fields and reject old
   single-head checkpoints/configs.
5. Execute S05 fixtures in the dependency-complete runtime and compare
   single-class candidate/NMS fixtures against the fixed official reference.
6. Profile CPU float64 rotate NMS at worst-case 1000 candidates/task; any faster
   replacement must preserve the exact deterministic fixture results.
7. Exercise official devkit submission loading and evaluator round trip before any
   full-data or metric authorization.

Old single-head checkpoints are structurally incompatible and must not be silently
loaded as the O-018 model.

## Allowed interpretations

- The implementation commit contains a framework-independent six-task CenterHead
  and deterministic candidate/NMS modules matching the recorded O-018 contract.
- Candidate selection has no second task-wide top-K and maps task-local classes to
  canonical devkit IDs by name.
- Static Python compilation, diff hygiene, ownership, and actual-source rotated-IoU
  golden geometry passed on the login node.
- The authored fixtures precisely encode the required runtime review envelope.

## Forbidden interpretations

- Multi-class decode is element-wise identical to official BEVFusion.
- The Torch/pytest fixture suite passed; it was not executable locally and no job
  was authorized.
- Target rendering or multi-task loss is integrated; `losses.py` remained read-only.
- Detector/tasks/config/checkpoint/full-stack integration is complete.
- CPU rotate-NMS production performance is acceptable before profiling.
- Mini, full trainval, 100/1000-step, mAP/NDS, model quality, fusion gain, FL,
  attack/defense, generalization, or publication conclusions.

## Residual risks and requested S00 action

- Launch independent S05-R from the final durable worker SHA after completeness
  audit. The reviewer should execute the 27 authored cases in the validated runtime
  if separately authorized and adversarially compare boundary IoU/tie behavior.
- Preserve the distinction between pre-NMS candidate no-starvation and official
  task-wide NMS suppression/post-budget behavior.
- Assign all detector/loss/config integration exclusively to reviewed S07-B/S06
  owners; do not repair those seams opportunistically in this worker branch.
- No rerun, compute, merge, push, upload, or scope expansion is requested by S05.
