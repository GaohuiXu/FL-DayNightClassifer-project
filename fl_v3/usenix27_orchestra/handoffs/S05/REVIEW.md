# S05-R independent review — CenterHead / decode / NMS

## Verdict

- Session: `S05-R`.
- Reviewed worker SHA: `4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9`.
- Worker implementation SHA: `9fd3281651ef006a175ed9462e7bf1eaf3437357`.
- Worker base SHA: `372de9398ae435f82b83367a922fd302c0635738`.
- Review branch: `codex/s05-r-centerhead-review`.
- Verdict: **CHANGES-REQUESTED**.

The six-task head, class-name mapping, O-018 per-class candidate selection, canonical
box conversion, and the main circle/rotate geometry are substantially aligned with
the frozen contract. However, two deterministic/precision defects block the S05
gate, and one exported NMS fail-closed claim is not true for all inputs. No
implementation was modified by this review.

## Findings (severity order)

### [P1] FP16 logits are sigmoid/top-K sorted in FP16 instead of the official forced-FP32 decode path

`fl_v3/src/fl_v3/models/fusion/centerhead_decode.py:208` executes
`output["heatmap"][batch_index].sigmoid()` without first converting the task output
to `float32`. Scores remain in the input dtype through selection and output
(`:208-254`, `:277`), and gathered velocity also retains that dtype (`:263`,
`:280`). In the active fp16 AMP policy, this makes sigmoid and candidate ordering
an fp16 operation.

The immutable MIT BEVFusion reference wraps `CenterHead.get_bboxes` in
`@force_fp32(apply_to=("preds_dicts"))` at upstream
`mmdet3d/models/heads/bbox/centerpoint.py:636-637`. Therefore its sigmoid, top-K,
thresholding, decode, and NMS inputs are float32 even when the head was produced
under mixed precision. S05's single-class "official parity" is consequently not
established for the production precision regime.

This is not only a dtype-label difference. An independent IEEE-754 binary16 check
found adjacent representable logits `-1.9453125` and `-1.9443359375`: after an
FP32 sigmoid their scores are respectively `0.1250653825` and `0.1251722811`, while
rounding the sigmoid to fp16 maps both to `0.1251220703125`. S05 therefore creates
additional score ties and can change pre-NMS priority to the O-018 class/spatial tie
rule where the official path has distinct scores. Threshold behavior can likewise
differ near `0.1`. The authored tests use default fp32 tensors and do not cover this.

Required remediation is for the worker/S00, not this reviewer: make the decode
precision contract explicit and match the pinned forced-FP32 reference (including
score/velocity output semantics), then add fp16-head-output versus forced-FP32
reference fixtures around score/tie/threshold boundaries.

### [P1] The claimed content-total submission order has a key collision that can change official TP error metrics

`fl_v3/src/fl_v3/eval/box_to_global.py:233-242` sorts equal-score converted boxes by
translation, size, quaternion, and class name, but omits `velocity` and
`attribute_name`. Two same-score, same-class boxes with identical geometry but
different velocities/attributes therefore have equal keys. Python's stable sort
then preserves their input order, so reversing decode input reverses serialized
submission order for those boxes.

The independent actual-source counterexample produced:

```text
keys_equal True
forward_velocitys [(0.0, 0.0), (5.0, 0.0)]
reverse_velocitys [(5.0, 0.0), (0.0, 0.0)]
```

The official nuScenes detection accumulator sorts equal confidence pairs using
the prediction emission index (`nuscenes/eval/detection/algo.py:44-54`) and records
velocity/orientation/attribute errors only for the prediction that becomes the TP
(`:73-110`). With duplicate equal-score geometry, the omitted fields can therefore
change AVE/AAE and NDS under an input permutation. Correct S05 NMS should normally
remove same-task geometric duplicates, but the submission helper is shared with
legacy/other decode paths, production wiring is not yet complete, and S05 explicitly
claims general content-defined submission permutation invariance. The existing
fixture at `fl_v3/tests/test_s05_eval_roundtrip.py:73-89` uses three geometrically
different boxes and cannot expose this collision.

Required remediation: make the submission key total over metric-relevant serialized
content (or reject indistinguishable duplicate geometry before submission), and add
the equal-score/same-geometry/different-velocity-and-attribute permutation fixture.

### [P2] Exported NMS helpers do not fully fail closed on invalid geometry or budgets

The HANDOFF claims explicit fail-closed finite/positive geometry checks. The
production `decode_centerhead` path does prefilter finite boxes and positive
dimensions at `centerhead_decode.py:269-274`, so this finding does not show that the
default decoder emits degenerate boxes. It does show that the separable exported NMS
API and its stated contract are weaker than claimed:

- `rotate_nms` validates geometry only when `rotated_iou_bev` is called during a
  pairwise comparison (`nms_deterministic.py:207-221`). A one-box input, or a box
  accepted immediately at the post budget, can be returned without `_box_corners`
  ever validating finite yaw or positive length/width.
- `circle_nms` validates only finite `x/y` (`:70-79`); a canonical box with
  non-finite/zero/negative dimensions or yaw can pass the public helper.
- neither public helper rejects non-positive `post_max_size`; because a candidate is
  appended before the budget comparison (`:86-88`, `:213-216`), `post_max_size=0`
  still returns one box. `_ordered_prefix` also treats non-positive
  `pre_max_size` as unlimited (`:48-51`). `CenterHeadDecodeConfig` prevents these
  budget values in the default decode path, but the exported helpers do not.

Required remediation: either validate the complete public NMS input contract before
early return/selection and reject invalid budgets, or narrow the API/documentation
and tests so the prevalidated-only boundary is explicit. Add single-box invalid
geometry and zero/negative budget fixtures.

## Contract audit

| Review item | Independent result |
|---|---|
| Six official tasks and task order | **PASS (static)**: `car`; `truck/construction_vehicle`; `bus/trailer`; `barrier`; `motorcycle/bicycle`; `pedestrian/traffic_cone`. |
| Shared transform and independent two-conv fields | **PASS (static)**: one shared `3x3 Conv-GN-ReLU`; each task has independent heatmap/reg/height/dim/rot/vel `Conv-GN-ReLU-Conv` branches. Runtime gradients were not requested and not run. |
| GN B=1/B>1 and batch behavior | Architecture is per-sample GN and contains no batch-statistics layer. Authored B=1/B>1 isolation case exists; Torch runtime and a direct head batch-permutation case remain **NOT RUN / incomplete evidence**. |
| O-018 per-class K and no second task K | **PASS (static)**: each class independently takes at most 500 (`centerhead_decode.py:214-223`), concatenation goes directly to deterministic ordering and task NMS; no second task-wide 500 selection exists. |
| O-018 tie order | **PASS for current score values (static)**: NumPy lexsort keys implement score-desc/global-ID-asc/flat-index-asc. Finding P1 shows fp16 creates ties absent from the official forced-FP32 path. |
| Single-class official parity | Candidate algorithm matches the pinned two-stage K for ordinary fp32 inputs; **CHANGES-REQUESTED** for active fp16 AMP because the reference forces FP32 and S05 does not. No multi-class exact-parity claim is accepted. |
| Class-name to devkit-global IDs | **PASS**: explicit mapping is `[(0,), (1,4), (2,3), (9,), (6,7), (5,8)]`; construction vehicle, bus, barrier, pedestrian, and traffic cone are all covered. No cumulative task-offset decode path was found. |
| Strict score and center range | **PASS (static)** for defaults: score uses strict `>0.1`; center range is inclusive at both ends. Exact boundary runtime cases were not authored/executed. |
| Candidate budgets/starvation | **PASS (static)** before NMS: at most 500/class and 1000/two-class task, so the official `pre=1000` does not add candidate starvation. Task-wide suppression and `post=83` can still remove a class, as O-018 explicitly allows. |
| Circle NMS | **PASS for default decode path (static/actual source)**: squared metres and inclusive `dist2 <= threshold`; official circle source has the same boundary. Public invalid-geometry/budget behavior is Finding P2. |
| Rotate NMS geometry/yaw/scale | Canonical `(l,w)` CCW-yaw polygon geometry, per-local-class dimension scaling, task-wide cross-class suppression, `IoU > 0.2`, and pre/post ordering are coherent. Actual-source identity/disjoint/perpendicular fixtures passed, as did 5,000 random symmetry/range checks (worst symmetry residual `5.69e-16`). The pinned CUDA kernel also uses strict `IoU > threshold`. Public invalid inputs are Finding P2; no official-kernel runtime parity was run. |
| Canonical box and velocity | **PASS (static)**: gravity center `(x,y,z,l,w,h,yaw)`, LiDAR `(vx,vy)`, exact encode/decode field order, global rigid lift, and official `size=(w,l,h)` are preserved. |
| Submission cap/order/tokens | Cap 500 and duplicate/unknown-token rejection are **PASS (static)**. Equal-score total order is **CHANGES-REQUESTED** under Finding P1. |
| Eval hardening / denominator impact | No metric formula, class denominator, GT filtering, or ASR eligibility code changed. Invalid geometry/labels, duplicate tokens, and >500 boxes now fail closed rather than being silently dropped. The incomplete ordering key can change TP error metrics as above. |
| Protected worker paths | **PASS**: `losses.py`, `detector.py`, `training/tasks.py`, `bev_grid.py`, all `fl_v3/collab/`, and `fl_v2/` have zero `base..worker` diff. |
| Production wiring | Correctly **DEFERRED / NOT PASS**: legacy detector/loss/config paths do not consume the list-of-task-dicts or `decode_centerhead`; S07-B remains responsible. Old single-head checkpoints are incompatible. |
| Runtime tests | **NOT RUN**. Static AST count confirms 26 functions / 27 pytest cases, but neither Torch nor pytest is installed in the x86_64 login interpreter. Authorship is not PASS evidence. |

## Git topology and provenance

Preflight before the review branch was created:

```text
git rev-parse --show-toplevel
/home/gaohui/.codex/worktrees/b015/fl_weather_project
git rev-parse HEAD
4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9
git branch --show-current
<empty: detached>
git status --short
<empty>
```

The authorized branch `codex/s05-r-centerhead-review` was then created without
moving HEAD. Topology checks:

- `merge-base(base, worker) = 372de9398ae435f82b83367a922fd302c0635738`;
- base is an ancestor of worker;
- implementation `9fd3281` has parent exactly `372de939...`;
- delivery `4561d3e` has parent exactly `9fd3281...`;
- `base..worker` contains exactly those two commits and the ten paths listed in the
  HANDOFF;
- O-018 canonical commit `a59fcc549ba62cc0c00fc8fe20c36063ca6f4648`
  is a sibling of the worker line, not an ancestor (`merge-base = 372de939...`).
  This is consistent with the active-session amendment being delivered out of band;
  the exact commit was independently inspected and its O-018 text matches the
  worker acknowledgement and review envelope.

Full `git diff base..worker` SHA-256:

```text
04d59dff657c0be27219511ea0cea740f7737197a266e1ede1d1b7ce25ecab6c
```

## Hash verification

The supplied HANDOFF hash and every worker file hash recomputed exactly:

| Artifact | Recomputed SHA-256 |
|---|---|
| `handoffs/S05/HANDOFF.md` | `dc5360bc8adb3ba2e364c506798612a9a6e7842b93533d45e4257f07beb9d0a2` |
| `models/fusion/head.py` | `3b731ba4048c95f67c6bc9238863247232a92623a85b1d0982ed7c6b7a4633c1` |
| `models/fusion/centerhead_decode.py` | `834bac66758224813c7a254307e5e8a80e07c3c9899fdb3465acc201d367af47` |
| `models/fusion/nms_deterministic.py` | `2d6f2b0aad7c0e31aa6cbfae170dfe5a52c5982d15de99d4b74ea21f7e91b22d` |
| `eval/box_to_global.py` | `d7192ffe3d3646c05b1b9b955696263e161aff61a15157c0c5a75b6944cd55b9` |
| `eval/detection_eval.py` | `99867c68c8ebfd86556e8bb16bace629df789cf92881f3a392853858293d99f2` |
| `tests/test_head_capacity.py` | `0d4d7600a3a1da7bbb3674c89823069ed17ab67d54246d95ef189e3012b3daa8` |
| `tests/test_s05_centerhead_decode.py` | `35ce4684375a5e02a98fb1c053e929ac4ea2d94ebabb4c00bd29735c979bb915` |
| `tests/test_s05_nms.py` | `bc2b30bdb9c9e3c22ac42d355730d88f8fcb7a359aeb232596190b9b30a2e0c1` |
| `tests/test_s05_eval_roundtrip.py` | `292fa8bc48b7038444a89a2797c218a2ea0519f82845e1badff135aebc3d8ff8` |

The five MIT BEVFusion raw files at immutable commit
`326653dc06e0938edf1aae7d01efcd158ba83de5` also recomputed to the exact HANDOFF
values, in HANDOFF order:

```text
3425d0b897d240c41ab49406bc84f340cec4fff960cbd2028673fd120f7e7826
61af3caec8cb781e8ee2152f7e4a9496f2b823fe40cf4bb97f28818d94537898
3d5b3d06f76df952955fe7566b58913741ef7033f323d460af7c7a2ebecea7b6
9b2924117b5149d42470c4c866068a979a811e535f1b54b54caa7b6e241ce0e2
f466f0af8b178c2f42c467ade46148b07ad0c37cb5a26064527c83f62f9731c2
```

CenterPoint v0.2 tag resolution independently returned
`e9ef04c3715aa3342fa42f4f4e064db987def6ad`. Cross-check raw hashes were:

```text
det3d/models/bbox_heads/center_head.py
44df7fd4c0ee0f8ba3d9de997258bd0d7974eb7d55bdb06fa72621bec9d68610
det3d/core/utils/center_utils.py
dbde4b76f6143ed64f9133740b905e82d5c9774490634b99a23296d516959157
configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_dcn.py
940830bdaa278cdc3e74cd5a364c53d9640839c5dd3a05883fe8c04cff00495f
```

## Commands and local evidence

Executed without Slurm/GPU/data access:

1. Complete reads of repository `AGENTS.md`, `fl_v3/docs/env.md`, all canonical
   Orchestra documents, S05/S05-R contracts, O-017/O-018, HANDOFF, full
   `base..worker` diff, changed sources/tests, protected paths, and immutable raw
   references.
2. `python3 -m py_compile` for all five changed/added source files, all four changed
   S05 test files, and the two existing eval tests, with
   `PYTHONPYCACHEPREFIX=/tmp/s05r-pycache`: **PASS**, no output.
3. `git diff --check base..worker`: **PASS**, no output.
4. Conflict-marker scan over changed source/tests: **PASS**, no matches.
5. Protected-path `git diff --quiet`: **PASS**, no diff.
6. Actual-source AST rotated-IoU identity/disjoint/perpendicular cases and 5,000
   deterministic random symmetry/range cases: **PASS**, worst symmetry residual
   `5.689893001203927e-16`.
7. Actual-source AST submission-sort collision: **FAIL as expected**, exact output
   recorded in Finding P1.
8. Static authored-test census: `26` test functions, `27` pytest cases because one
   function has two parameter values.
9. Login interpreter dependency probe:

```text
x86_64
torch ModuleNotFoundError No module named 'torch'
pytest ModuleNotFoundError No module named 'pytest'
numpy ModuleNotFoundError No module named 'numpy'
```

No `sbatch`, `srun`, dataset access, optimizer/model step, metric, profile, retry,
or follow-on occurred. No `RUN_REQUEST.md` or `RESULTS.md` was created: the
source-level blockers above already force CHANGES-REQUESTED, so GH200 execution
would not resolve the verdict. Runtime validation remains required after remediation
under a new exact authorization.

## Allowed interpretations

- The worker delivered the scoped six-task CenterHead and deterministic
  candidate/NMS modules on the claimed two-commit topology.
- The default task groups, task-local-to-global name mapping, per-class K=500 with
  no second task-wide K, class/spatial tie keys, default NMS thresholds/budgets,
  canonical box layout, and local/global size/yaw/velocity conversion are present.
- Static compilation, diff hygiene, protected-path ownership, immutable reference
  hashes, HANDOFF/file hashes, and the independent actual-source rotated geometry
  checks passed.
- Production detector/loss/config wiring is explicitly deferred to S07-B and is not
  represented as complete.

## Forbidden interpretations

- S05 is accepted for integration: the independent verdict is
  **CHANGES-REQUESTED**.
- Single-class decode is officially equivalent under active fp16 AMP; forced-FP32
  reference semantics are currently missing.
- Multi-class decode is element-wise identical to official BEVFusion; O-018
  explicitly forbids that claim.
- Equal-score submission/evaluation is fully input-permutation invariant.
- The exported NMS helpers always fail closed on degenerate/non-finite canonical
  boxes or invalid budgets.
- Any of the 27 authored Torch/pytest cases passed; none was executed.
- Target/loss, detector/tasks/config/checkpoint compatibility, full-stack runtime,
  NMS production performance, mini/trainval metrics, CL/FL/security, or scientific
  readiness passed.

## Residual risks and re-review requirements

1. Return Findings P1/P1/P2 to S05 (or an explicitly authorized remediation
   session); do not repair them opportunistically in S07-B.
2. Re-review the exact remediation diff and rerun the immutable hashes/static
   adversarial fixtures.
3. After source findings close, execute the full S05 suite in the validated
   dependency-complete runtime with explicit fp16-output/forced-FP32-decode cases,
   B=1/B>1 head batch permutation, strict score/range boundaries, invalid single-box
   NMS geometry/budgets, and the velocity/attribute submission-order collision.
4. S07-B must still reconcile the independently reviewed S02 target/loss field
   mapping, bind every decode/NMS field into resolved config/provenance, reject old
   single-head checkpoints, and profile the correctness-first CPU float64 rotate
   NMS before any full-run authorization.

