# S05-R2 independent re-review — CenterHead / decode / NMS

## Verdict

- Session: `S05-R2`.
- Reviewed worker/delivery SHA:
  `705216de097ae9eeb1813de6dcdc916e2844fcde`.
- Remediation implementation SHA:
  `753944c199ceeace160732218f1b16dfdd15ac21`.
- Original worker delivery:
  `4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9`.
- Worker base: `372de9398ae435f82b83367a922fd302c0635738`.
- First independent review:
  `c81826251349ede7c514950df785e4fe05d60192`.
- Current verdict: **BLOCKED pending one exact O-009 synthetic runtime job**.

No new source-level correctness finding was identified. All three first-review
findings are statically closed by `753944c`, and the broader O-017/O-018 contract
remains coherent. However, the dependency-backed runtime evidence required for
PASS does not exist: the 44 authored Torch/devkit cases were never executed, and
the x86_64 login interpreter lacks `torch`, `pytest`, and `numpy`. The exact bounded
request is `RUN_REQUEST.md`; it is `PENDING_S00_EXACT_O009_APPROVAL_DO_NOT_SUBMIT`.
No implementation file was modified by this review.

## Findings first

### [GATE-BLOCKER] The 44-case tensor/devkit suite is authored but still NOT RUN

The source and static adversarial evidence are sufficient to close the prior
findings at code-review level, but not to certify their executed behavior:

- forced-FP32 behavior depends on actual Torch fp16 conversion, sigmoid,
  stable-sort, threshold and gather semantics;
- GroupNorm B=1/B>1 isolation requires actual tensor execution;
- deterministic NMS wrappers and their invalid-input paths require Torch/NumPy
  interoperation;
- submission construction requires the installed nuScenes devkit
  `DetectionBox` validation and serialization behavior.

The login probe returned `ModuleNotFoundError` for `torch`, `pytest`, and `numpy`,
consistent with the documented x86_64/aarch64 split. Therefore PASS requires the
single exact synthetic job in `RUN_REQUEST.md`: 44 cases, zero failures/errors/
skips, exact source/snapshot/allocation identity, and checksummed raw artifacts.
This blocker does not authorize a retry or any broader runtime scope.

## Closure of the first review findings

| First-review finding | Re-review result | Exact evidence |
|---|---|---|
| P1: fp16 logits decoded before official forced-FP32 boundary | **CLOSED STATICALLY; RUNTIME PENDING** | `centerhead_decode.py:208-214` promotes every head field to fp32 before sigmoid; `:263-269` gathers fp32 reg/height/dim/rot/vel; `:281-287` returns fp32 scores/velocity; empty final outputs are fp32 at `:370-378`. Hostile tests at `test_s05_centerhead_decode.py:110-152` pin adjacent binary16 logits and the strict 0.1 neighborhood, and assert FP32 score/velocity semantics. |
| P1: submission content key omitted velocity/attribute | **CLOSED STATICALLY; RUNTIME PENDING** | `box_to_global.py:234-245` includes serialized velocity, class, and attribute after score/geometry/rotation. `test_s05_eval_roundtrip.py:92-114` reverses identical score/class/geometry records with `(5,0)/vehicle.moving` and `(0,0)/vehicle.parked` and requires identical serialized order. Actual-source stdlib execution also produced the same `[parked,moving]` order for forward/reverse inputs. |
| P2: public NMS early returns accepted invalid geometry/budgets | **CLOSED STATICALLY; RUNTIME PENDING** | `_validate_nms_inputs` at `nms_deterministic.py:55-76` runs before empty/single-box returns in both public helpers (`:95-103`, `:227-235`), validates full canonical finite geometry, positive l/w/h, aligned lengths, and positive pre/post budgets. Tests at `test_s05_nms.py:108-147` cover both helpers, NaN yaw, zero/negative dimensions, and zero/negative budgets. |

The remediation did not waive, reinterpret, or move any finding into S07-B.

## O-017/O-018 contract re-audit

| Review item | Independent result |
|---|---|
| Frozen source of truth | **PASS STATIC**. The five MIT BEVFusion files at immutable `326653dc06e0938edf1aae7d01efcd158ba83de5` independently recomputed to the HANDOFF hashes `3425d0...`, `61af3c...`, `3d5b3d...`, `9b2924...`, `f466f0...`. CenterPoint v0.2 resolved to `e9ef04c3715aa3342fa42f4f4e064db987def6ad`; its three cross-check hashes also matched. |
| Six tasks and fields | **PASS STATIC**. Official order is car; truck/construction vehicle; bus/trailer; barrier; motorcycle/bicycle; pedestrian/traffic cone. Every task owns heatmap/reg/height/dim/rot/vel two-convolution branches after one shared transform. |
| GroupNorm adaptation | **PASS STRUCTURALLY / RUNTIME PENDING**. Shared and per-field normalization is GroupNorm, with no batch-statistics layer. The B=1 versus B>1 distractor fixture exists at `test_head_capacity.py:51-60`. |
| Per-class K/no second task K | **PASS STATIC**. `centerhead_decode.py:220-237` selects up to 500 independently per class; concatenation proceeds directly to O-018 ordering and NMS. No second task-wide K exists. One/two-class tasks admit at most 500/1000 candidates. |
| No-starvation claim boundary | **PASS**. The implementation and HANDOFF explicitly limit the claim to removal of pre-NMS task-wide top-K starvation. Official task-wide suppression and post=83 remain and can still remove a class. No multi-class exact official-decode claim is made. |
| Deterministic tie order | **PASS STATIC**. `nms_deterministic.py:21-39` implements score descending, canonical global class ID ascending, flattened spatial index ascending. Candidate scores are FP32 before this key. |
| Explicit global labels | **PASS STATIC**. Name mapping is exactly `[(0,), (1,4), (2,3), (9,), (6,7), (5,8)]`; no cumulative task-offset path exists. Fixtures explicitly cover construction vehicle, bus, barrier, pedestrian, and traffic cone. |
| Score/range/budgets | **PASS STATIC**. Strict `score > 0.1`; inclusive `[-61.2,-61.2,-10,61.2,61.2,10]`; per-class 500; NMS pre/post 1000/83; rotate IoU 0.2. At most `6*83=498`, under the official 500-box cap. |
| Circle NMS | **PASS STATIC / RUNTIME PENDING**. Squared metres, inclusive `dist2 <= threshold`, official task-wide behavior, and deterministic priority are preserved. Full canonical prevalidation now precedes early return. |
| Rotate NMS | **PASS STATIC / ACTUAL-SOURCE GEOMETRY**. Canonical `(l,w)` CCW-yaw polygons, per-local-class scale, task-wide cross-class suppression and strict `IoU > 0.2` are coherent. Actual-source identity/disjoint/perpendicular cases returned `1/0/1/3`; 5,000 seeded random pairs stayed symmetric/in-range (worst symmetry residual `6.22e-15`). Torch wrapper execution remains pending. |
| Box/yaw/velocity | **PASS STATIC**. Gravity-center `(x,y,z,l,w,h,yaw)`, `atan2(sin,cos)`, LiDAR `(vx,vy)`, rigid global lift and official `size=(w,l,h)` are preserved. |
| Submission/eval | **PASS STATIC / RUNTIME PENDING**. Invalid labels/geometry, duplicate tokens and >500 boxes fail closed. Equal-score order is total over metric-relevant serialized fields, so reversing the hostile pair cannot change devkit emission order or TP pairing. No metric denominator, GT filter, class range, or ASR eligibility formula changed. |
| Protected paths and ownership | **PASS**. `losses.py`, `detector.py`, `training/tasks.py`, `bev_grid.py`, `fl_v3/collab/`, and `fl_v2/` have no base-through-worker diff. The remediation touches only the three source files, three tests, and HANDOFF declared in its scoped package. |
| Production wiring | **DEFERRED BY CONTRACT**. S05 does not integrate the list-of-task-dicts into detector/loss/config/checkpoint paths. S07-B must reconcile the independently reviewed S02 field/target contract and reject legacy single-head checkpoints. |
| CPU float64 rotate-NMS production cost | **UNRESOLVED INTEGRATION RISK, NOT THIS PASS GATE**. Correctness-first geometry is unprofiled at 1000 candidates/task. Profiling is explicitly deferred to separately authorized S07-B work and must not alter the frozen fixtures. |

## Git topology and artifact verification

Preflight at task start:

```text
HEAD=705216de097ae9eeb1813de6dcdc916e2844fcde
branch=<empty detached HEAD>
status=<clean>
toplevel=/home/gaohui/.codex/worktrees/s05r2/fl_weather_project
```

Topology:

- `4561d3e` parent is original implementation `9fd3281`;
- remediation `753944c` parent is `4561d3e`;
- final HANDOFF update `705216d` parent is `753944c`;
- remediation diff SHA-256 `4561d3e..753944c`:
  `02627ef26b0f06cdfa7ef9b42a5bd8a95f36e00f3d28fb93304a42dfd1cb1a65`;
- full base-through-remediation diff SHA-256 `372de939..753944c`:
  `14771e2501b603ac24a8957fe64dec523a45a96b03964e093c3cbb8b4db6c4a9`.

Verified durable artifacts:

- final HANDOFF SHA-256:
  `915061747a22d98cfc12b192ddb9a4ccd69fc661e7c5c29c525de4254ad30dbd`;
- first REVIEW SHA-256:
  `270d47c498af7aaaeef1af535be7513223bd6793c933b9ca721aed223d0a79e5`;
- remediated file hashes all match the HANDOFF, including
  `centerhead_decode.py=13d22b...`, `nms_deterministic.py=9adcc7...`,
  `box_to_global.py=08c2ab...`, and the three hostile test hashes.

O-018 canonical commit `a59fcc549ba62cc0c00fc8fe20c36063ca6f4648`
is a sibling of the worker history rather than an ancestor, as expected for the
acknowledged active-session amendment. Its independently read text matches the
HANDOFF and implementation semantics exactly.

## Checks executed without Slurm

1. Complete reads of repository instructions, environment/roadmap, canonical S05
   and reviewer contracts, O-017/O-018, final HANDOFF, first REVIEW, both exact
   diffs, every changed source/test, protected paths, and pinned upstream sources.
2. `python3 -m py_compile` over all five changed/added source modules, four S05 test
   files, and both existing eval test modules: **PASS**.
3. `git diff --check` for full and remediation diffs: **PASS**.
4. Conflict-marker and protected-path scans: **PASS**.
5. Independent upstream raw-file and CenterPoint tag hash verification: **PASS**.
6. Actual-source rotated-IoU golden/random checks and submission-key hostile order:
   **PASS**.
7. Static test census: 31 functions / 44 parametrized pytest cases.
8. Login dependency probe: `x86_64`; `torch`, `pytest`, `numpy` all unavailable.

No `sbatch`, `srun`, dataset access, training/model step, optimizer update,
scientific metric, profile, merge, push, upload, or external publication occurred.

## Runtime gate (pending)

`RUN_REQUEST.md` binds:

- exact request SHA-256
  `b5e43c0e4fb1f999a5273076a32518c9309752a33e651bd732386973213dd19c`;
- exact proposed launcher-body SHA-256
  `7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10`;
- worker `705216d`, tree `2d5cd99...`;
- 31-file source-list hash `bea19dd5...` and source-state hash `2ff6389f...`;
- one fresh immutable `/nobackup` snapshot;
- exactly one shared GH200 allocation, eight CPUs, 15 minutes, maximum 0.25
  GPU-hours, with `--nodes` intentionally omitted;
- exactly four synthetic test files / 44 cases;
- no data, optimizer/model step, metric, profile, array, DDP, retry, or follow-on;
- exact allocation/runtime identity, JUnit, logs, source manifest, and in-job
  `sha256sum -c` artifacts.

Until S00 approves and the exact job passes, S05-R2 remains **BLOCKED**, not PASS.

## Allowed interpretations

- The exact remediation closes the first review's three source findings.
- O-018 task grouping, explicit global label map, per-class K/no second task K,
  deterministic ties, GN adaptation, canonical conversion, and official task-wide
  NMS constants are present and statically coherent.
- Static compilation, diff/ownership hygiene, upstream hashes, pure geometry, and
  hostile content-order checks passed.

## Forbidden interpretations

- S05 is accepted or reviewed PASS before the pending 44-case runtime passes.
- Multi-class decode is element-wise identical to official BEVFusion.
- The authored Torch/devkit cases passed; they have not run yet.
- Target/loss/detector/config/checkpoint integration or legacy-checkpoint migration
  is complete.
- CPU NMS production performance, mini/trainval quality, mAP/NDS, fusion gain,
  full-run readiness, FL/security behavior, generalization, or publication claims.

## Residual risk after a prospective runtime PASS

Even if the exact 44 cases pass, S07-B must still integrate the S02 target/loss
mapping, bind all head/decode/NMS fields into resolved config/provenance, reject old
single-head checkpoints, run official devkit end-to-end conversion checks, and
profile the correctness-first rotate NMS before any full-run request. These are
explicit downstream integration gates and must not be silently reclassified as
S05 implementation defects or waived by this focused runtime.
