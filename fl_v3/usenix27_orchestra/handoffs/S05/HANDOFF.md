# S05 HANDOFF — multi-task CenterHead and deterministic decode/NMS

## Session identity and worker self-assessment

- Session: `S05`.
- Base SHA: `372de9398ae435f82b83367a922fd302c0635738`.
- S07-A foundation SHA named by kickoff:
  `0249eb21a32730ac1689255491b19a158711401f`.
- Source branch named by kickoff: `codex/s00-orchestra-ledger`.
- Expected/observed startup mode: clean detached HEAD at the exact base.
- Owner-authorized worker branch: `codex/s05-centerhead-decode`.
- Original implementation/test commit:
  `9fd3281651ef006a175ed9462e7bf1eaf3437357`.
- Scoped review-remediation implementation/test commit:
  `753944c199ceeace160732218f1b16dfdd15ac21`.
- Final delivery SHA: the commit containing this handoff; returned to S00 in the
  task response because a commit cannot embed its own SHA without changing it.
- Worker self-assessment after scoped remediation and both dependency-backed
  runtime jobs: **PASS for the three requested source findings, static gates, and
  the corrected 44-case synthetic runtime gate. Independent final re-review
  acceptance and full-stack integration remain pending/NOT RUN**.
  This is not an independent review, S07-B PASS, model-readiness PASS, or
  scientific PASS.

This worker did not invoke Slurm/srun. Independent S05-R2 later ran exact synthetic
Job 336731 under S00 approval; its negative result is preserved below. No model
training, mini traversal, full trainval access, metric, profile, merge, push, PR,
upload, or publication action occurred.

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

## Independent review and scoped remediation

Independent S05-R reviewed worker delivery
`4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9` and returned
**CHANGES-REQUESTED** in review commit
`c81826251349ede7c514950df785e4fe05d60192`. The exact `REVIEW.md` SHA-256 is
`270d47c498af7aaaeef1af535be7513223bd6793c933b9ca721aed223d0a79e5`.
The findings and their closure in remediation commit `753944c` are preserved:

1. **P1 forced-FP32 decode.** The original code applied sigmoid/top-K directly
   to fp16 heatmaps. Remediation explicitly promotes every task head field to
   FP32 before sigmoid, strict thresholding, per-class top-K, regression decode,
   NMS input, and returned scores/velocity. Hostile fixtures use adjacent fp16
   logits `-1.9453125/-1.9443359375` whose rounded fp16 sigmoids tie but whose
   FP32 sigmoids remain distinct, plus adjacent fp16 logits bracketing the strict
   `0.1` threshold. This aligns the active-AMP decode boundary with the pinned
   MIT `force_fp32` contract while retaining O-018 no-starvation semantics and
   the prohibition on multi-class exact-parity claims.
2. **P1 submission order.** The original equal-score content key omitted
   `velocity` and `attribute_name`, allowing official TP-error pairing to follow
   input order for duplicate geometry. Remediation adds both serialized fields
   to the key. The forward/reverse hostile fixture uses identical
   score/class/geometry with `(0,0)/vehicle.parked` and
   `(5,0)/vehicle.moving` and requires byte-equivalent result dictionaries.
3. **P2 exported NMS validation.** The original public helpers could accept a
   single invalid box without reaching pairwise geometry validation and treated
   non-positive budgets incorrectly. Remediation validates full canonical
   `(x,y,z,l,w,h,yaw)` finiteness and positive `(l,w,h)` before every early
   return/selection, requires aligned input lengths, rejects non-positive pre/post
   budgets, and adds NaN-yaw, zero/negative dimension, and zero/negative budget
   fixtures for both exported helpers.

No other review finding was reinterpreted or waived. Runtime execution remains a
fresh re-review requirement rather than retroactive PASS evidence.

## S05-R2 Job 336731 negative and test-only return

Independent S05-R2 prepared the exact 44-case synthetic request on review branch
`codex/s05-r2-centerhead-review`; its current durable review SHA is
`61e7fb14bc6f44fe681628a1fb0ed701ad4f7f28`. S00 authorized the immutable request,
and Job `336731` executed worker delivery `705216d` on GH200 node `n570`.

- Scheduler status: **FAILED**, exit `1:0`, elapsed `00:01:15`, one node, eight
  CPUs, exactly one `nvidia_gh200_120gb`, no retry.
- JUnit: exactly 44 tests, 43 passed, one failure, zero errors/skips; test time
  `22.878s`.
- All forced-FP32, GroupNorm, O-018 no-starvation/tie, label-map, NMS geometry,
  invalid-input/budget, box conversion, and other submission tests passed.
- Sole failure:
  `test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content`.
  Its scientifically/correctness-critical `forward == reverse` check passed. The
  remaining expected value used `[0.0,0.0]`/`[5.0,0.0]` lists, while installed
  nuScenes devkit `DetectionBox.serialize()` stably returned `(0.0,0.0)`/
  `(5.0,0.0)` tuples. Attributes and ordering were already correct. This is a
  test container-contract mismatch, not a production/source defect.
- Runtime identity: aarch64, Python `3.11.15`, torch `2.11.0+cu128`, NumPy
  `1.26.4`, pytest `9.1.1`, nuscenes-devkit `1.1.11`, Pillow `12.2.0`; no dataset,
  optimizer update, or scientific metric.

Raw durable evidence:

| Artifact | SHA-256 |
|---|---|
| stdout `s05r2_centerhead_336731.out` | `fbeac7dbcc5b14cf1f377a6ca1e363c06e4932eb66a416d1179ea02349249b6e` |
| stderr `s05r2_centerhead_336731.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |
| `pytest.log` | `3e461e6e83df9dedbdd68b2e0059e4afc2348bc54d56efebf55ef57a348a20fc` |
| `pytest.junit.xml` | `0f79ed5509881bcc84a48f8dd546ebc69de0fd8ac4cdbfe074a8cd5ee806288e` |
| `execution_identity.json` | `ca35c57e1f0b3eb7ba4257f5be8a1df0f6b0ca736b5f1902e360ce153695e490` |
| `slurm_allocation.txt` | `de382961649eed7e5a31c213d924c50fed88160e36171cba0e4f9f9114a80d3f` |
| `runtime_source_files.txt` | `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857` |
| `runtime_source_sha256s.txt` | `2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766` |

The test-only correction commit is
`96e509b71a3e22afb4de397132438fd3b9bbf5d8`: it changes only the two expected
velocity containers from lists to the actual stable tuples and leaves both
`forward == reverse` and exact velocity/attribute ordering assertions intact.
Its tree is `aeaaad044199492b81c4383a013f3fb3c6596c02`; test-only diff SHA-256 is
`aed0033a6843212557b14bc0b950006e3b791cd2a75afb7fd5d40938e79fc700`;
corrected test SHA-256 is
`e938dd34656e3ae5f5e9019748bea52a3ccc5cb99144492d6bf9f45e79c203c0`.
No production file changed.

`RUN_REQUEST.md` and `run_s05r3_centerhead.sh` prepared one immutable 44-case
shared-one-GH200/15-minute request with fresh roots. S00 approved and submitted it
exactly once under the owner's temporary delegation allowing S00 to approve
necessary, reasonable validation-only S02-S05 Slurm jobs. It is explicitly not
O-009, which excludes reruns, and does not expand or reinterpret O-009. No
automatic retry or follow-on occurred.

### Job 336738 focused rerun PASS

- Exact executed delivery: `98b71eca7684b50ece69afc36175564c7c283033`;
  execution SHA/tree: `96e509b71a3e22afb4de397132438fd3b9bbf5d8` /
  `aeaaad044199492b81c4383a013f3fb3c6596c02`.
- Approved request SHA-256:
  `e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9`;
  launcher SHA-256:
  `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5`.
- Approval class: `S00_OWNER_DELEGATED_S02_S05_VALIDATION_RERUN`.
- Scheduler: **COMPLETED 0:0**, node `n411`, elapsed `00:01:13`, one shared
  node/eight CPUs/exactly one GH200, batch MaxRSS `540M`.
- Pytest/JUnit: **44 passed in 22.64s**; exact counts 44 tests, zero
  failures/errors/skips; JUnit time `22.645s`.
- The formerly failing duplicate-geometry velocity/attribute case passed without
  weakening `forward == reverse` or exact tuple/attribute ordering.
- All nine in-job checksum targets passed. No dataset, model/optimizer step,
  metric, profile, array, DDP, retry, or follow-on occurred.

| Job 336738 artifact | SHA-256 |
|---|---|
| stdout `s05r3_centerhead_336738.out` | `0cf6f1dc14ad07ef598076fb6ed067352bf71c789172f9babd5f1ed42d01ef87` |
| stderr `s05r3_centerhead_336738.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |
| `approved_launcher.sh` | `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5` |
| `approved_run_request.md` | `e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9` |
| `snapshot_identity.txt` | `6c47a4252bb65c227ef795eecd161749e5260ce6821a5a638da7b5457ab0aa20` |
| `runtime_source_files.txt` | `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857` |
| `runtime_source_sha256s.txt` | `7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1` |
| `execution_identity.json` | `9e2dde2468f17d10b99c2992440029b347f4b4a220143c3aecce7c6b84a62aab` |
| `slurm_allocation.txt` | `c76ffe8201b2025d7ed7b0cbf663fca8706073c10efee090fac0ed2347dba3d8` |
| `pytest.log` | `4db65ef4592e61cf1886e49bef9649ba87803b6cf41bc45e84de6484645121d3` |
| `pytest.junit.xml` | `bad9b34e02a4d7267cbbed4e2b4429c6498360a3c3317388fdc21f0be8206910` |
| `sha256sums.txt` | `301c5c4feed506f0ae5c130b1036cfe0c0aaeacf81f947cf121a6136f7339077` |

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
- an explicit forced-FP32 boundary for every incoming head field, including
  FP32 score and velocity outputs under fp16 AMP head output.

### Deterministic NMS

`models/fusion/nms_deterministic.py` replaces unavailable numba/mmcv kernels with:

- CPU float64 deterministic circle NMS using official squared-distance and
  inclusive `<= threshold` suppression;
- CPU float64 rotated rectangle intersection and task-wide IoU NMS;
- the same O-018 content order before pre-NMS truncation;
- explicit positive pre/post budgets and up-front fail-closed validation of full
  canonical geometry, including single-box/empty early-return paths.

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
  invariance, now including velocity and attribute name in the total key.

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

### Runtime fixture scope

The changed test set contains 31 test functions / 44 pytest cases covering:

- six-task topology, independent two-convolution fields, bias, GN batch isolation;
- exact class-name mapping including construction vehicle, bus, barrier,
  pedestrian, and traffic cone;
- 500 higher-scoring common-class candidates plus one retained tail candidate;
- equal-score global-class/spatial tie order;
- single-class official candidate parity;
- forced-FP32 adjacent-fp16-logit ordering, FP32 score/velocity dtypes, and the
  strict `0.1` threshold neighbourhood;
- canonical encode/decode box, yaw, velocity, and class round trip;
- B=1/B>1 and batch permutation;
- circle squared-metre boundary, duplicate circle/rotate boxes, task-wide
  cross-class rotate suppression, NMS input permutation, and NMS budgets;
- known rotated IoUs;
- exported circle/rotate single-box NaN-yaw and non-positive-dimension rejection,
  plus zero/negative pre/post budget rejection;
- local/global/eval round trip, official wlh/velocity/yaw conversion, content
  permutation including duplicate geometry with different velocity/attribute,
  invalid labels/geometry, duplicate samples, and the 500-box cap.

The x86_64 login interpreter reports `ModuleNotFoundError` for both `torch` and
`pytest`. Job 336731 ran the exact dependency-backed suite on GH200 and returned
43/44 with the preserved test-only container mismatch above. Job 336738 then ran
the corrected immutable suite and passed all 44 cases. This closes only the
focused synthetic runtime gate.

## File hashes at original implementation commit

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

## Scoped remediation provenance and hashes

- Remediation parent (reviewed worker delivery):
  `4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9`.
- Remediation implementation commit/tree:
  `753944c199ceeace160732218f1b16dfdd15ac21` /
  `dd71827f064b00a998b7213d94ca456dee930be0`.
- Exact remediation diff SHA-256 (`4561d3e..753944c`):
  `02627ef26b0f06cdfa7ef9b42a5bd8a95f36e00f3d28fb93304a42dfd1cb1a65`.
- Full base-through-remediation diff SHA-256 (`372de939..753944c`):
  `14771e2501b603ac24a8957fe64dec523a45a96b03964e093c3cbb8b4db6c4a9`.
- `git diff --check 4561d3e..753944c`: PASS, no output.
- Protected paths `losses.py`, `detector.py`, `training/tasks.py`, `bev_grid.py`,
  `fl_v3/collab/`, and `fl_v2/`: no remediation diff.

| Remediated file | SHA-256 at `753944c` |
|---|---|
| `eval/box_to_global.py` | `08c2abe372b2b0fcbad454dceda6cd49874a7427d470e5e800d7873578db2c4a` |
| `models/fusion/centerhead_decode.py` | `13d22b6639c48c78efb1f92794dd1ca9e12af20a13d25d9de607a0e8a9aefdbb` |
| `models/fusion/nms_deterministic.py` | `9adcc7816607fb079d653d96fcdf76fa8e12ef336412485f23c7ebcbd0717962` |
| `tests/test_s05_centerhead_decode.py` | `12123e37faf35c02d30fc704a05b45e2011ba0ed6c113a3b379548c26a1e0a00` |
| `tests/test_s05_eval_roundtrip.py` | `08337620e8ffc1de45597868bed7b72351dae1063744bb8bd61d7fd8b51a8ee0` |
| `tests/test_s05_nms.py` | `3725608b0aac7c528379fee6e66d06b195a75886c7c78f7ab23affe97cc4a284` |

## Gate checklist

| S05 gate | Worker status | Evidence / boundary |
|---|---|---|
| fixed reference/tasks/fields | PASS (code/static) | immutable reference hashes, six independent task heads |
| no global/cross-class top-K starvation | PASS | per-class 500; no second task K; hostile fixture passed Job 336738 |
| deterministic tie order | PASS (code + authored fixture) | score/global-ID/flat-index total order |
| fp16 head / forced-FP32 decode | PASS | all fields promoted before decode; adjacent-logit/threshold/dtype fixtures passed Job 336738 |
| circle NMS semantics | PASS | squared metres/inclusive comparison fixtures passed Job 336738 |
| rotate NMS geometry/determinism | PASS (focused runtime) | known IoU and Torch wrapper fixtures passed Job 336738 |
| duplicate/input permutation | PASS | circle/rotate/content-permutation fixtures passed Job 336738 |
| submission metric pairing order | PASS | velocity/attribute total-key and reverse-input duplicate-geometry fixture passed Job 336738 |
| exported NMS fail-closed boundary | PASS | canonical prevalidation and non-positive budget fixtures passed Job 336738 |
| box/yaw/velocity encode/decode | PASS | canonical round-trip fixture passed Job 336738 |
| local/global/eval/submission conversion | PASS (focused runtime) | class/wlh/yaw/velocity/cap fixtures passed Job 336738 |
| task-local/global label mapping | PASS (code/static) | explicit name map; no cumulative offset code path |
| loss/target consistency | INTERFACE FROZEN / INTEGRATION PENDING | regression encoder matches field order; Gaussian/losses.py remained S02-owned |
| production detector/task wiring | NOT DONE BY CONTRACT | S07-B integration requirement |
| focused compute/scientific metric | SYNTHETIC RUNTIME PASS / SCIENCE FORBIDDEN | exact Job 336738; no data/model step/metric/profile |
| independent S05-R | SOURCE REMEDIATED; R2 NEGATIVE PRESERVED; R3 RUNTIME PASS; FINAL VERDICT PENDING | original review `c818262`; Job 336731 preserved; Job 336738 44/44 PASS |

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
- The scoped remediation source explicitly implements pinned forced-FP32 decode,
  metric-relevant submission ordering, and full exported-NMS fail-closed checks.
- Candidate selection has no second task-wide top-K and maps task-local classes to
  canonical devkit IDs by name.
- Static Python compilation, diff hygiene, ownership, and actual-source rotated-IoU
  golden geometry passed on the login node.
- Exact Job 336738 passed all 44 authored synthetic fixtures under the recorded
  dependency-complete GH200 runtime and immutable execution identity.

## Forbidden interpretations

- Multi-class decode is element-wise identical to official BEVFusion.
- Job 336731 passed or was erased: it remains a preserved 43/44 failure. Job
  336738 separately passed the corrected immutable 44-case suite.
- S05 is independently accepted: review `c818262` requested changes and the exact
  remediation still requires fresh independent re-review.
- Target rendering or multi-task loss is integrated; `losses.py` remained read-only.
- Detector/tasks/config/checkpoint/full-stack integration is complete.
- CPU rotate-NMS production performance is acceptable before profiling.
- Mini, full trainval, 100/1000-step, mAP/NDS, model quality, fusion gain, FL,
  attack/defense, generalization, or publication conclusions.

## Residual risks and requested S00 action

- S05-R2/R3 should independently verify Job 336738 raw evidence and issue the
  final verdict from execution SHA `96e509b`; no production re-review scope or
  scientific interpretation is added by this tuple-only correction.
- Preserve the distinction between pre-NMS candidate no-starvation and official
  task-wide NMS suppression/post-budget behavior.
- Assign all detector/loss/config integration exclusively to reviewed S07-B/S06
  owners; do not repair those seams opportunistically in this worker branch.
- The exact one-time 44-case request is complete. S05 requests no additional
  rerun, compute, merge, push, upload, or scope expansion.
