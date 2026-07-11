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
- Current verdict: **CHANGES-REQUESTED — test-only fixture remediation**.

No new source-level correctness finding was identified. All three first-review
findings are statically closed by `753944c`, and the broader O-017/O-018 contract
remains coherent. Exact Job `336731` executed all 44 cases and returned 43 passed /
one failed. The semantic `forward == reverse` assertion and intended parked-before-
moving order passed; the sole failure is the fixture expecting velocity lists while
nuscenes-devkit `1.1.11` returned in-memory tuples. The zero-failure gate therefore
did not pass. S05 must correct only this representation-specific expectation and
return a new durable SHA for re-review. No production implementation file was
modified by this review, and no retry is authorized.

## Findings first

### [P2 TEST] The hostile submission fixture asserts list storage while the devkit returns tuples

Job `336731` ran the exact dependency-backed suite on nuscenes-devkit `1.1.11`.
`test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content`
first passed `assert forward == reverse`, proving the complete result dictionaries
were input-permutation invariant. It then observed the correct content order
`(0,0)/vehicle.parked`, `(5,0)/vehicle.moving`, but failed because actual
`record["velocity"]` values were tuples and the fixture expected lists.

This is not evidence of a production sort-key, velocity, attribute, or TP-pairing
defect. It is still a real gate failure: acceptance required 44/44 with no failures.
Required worker remediation is test-only: normalize `record["velocity"]` to a tuple
before comparison, or expect tuples, while retaining both full `forward == reverse`
and exact parked-before-moving content assertions. Production source and O-018
semantics must not change to make this test pass.

## Closure of the first review findings

| First-review finding | Re-review result | Exact evidence |
|---|---|---|
| P1: fp16 logits decoded before official forced-FP32 boundary | **CLOSED; RUNTIME PASS** | `centerhead_decode.py:208-214` promotes every head field to fp32 before sigmoid; `:263-269` gathers fp32 reg/height/dim/rot/vel; `:281-287` returns fp32 scores/velocity; empty final outputs are fp32 at `:370-378`. Job 336731 passed both adjacent-binary16/strict-0.1 hostile cases and FP32 output assertions. |
| P1: submission content key omitted velocity/attribute | **IMPLEMENTATION CLOSED; TEST FIXTURE CHANGE REQUESTED** | `box_to_global.py:234-245` includes velocity, class, and attribute. In Job 336731 the hostile test's complete `forward == reverse` assertion and desired parked-before-moving order passed. Only the later list-versus-tuple expected representation failed. |
| P2: public NMS early returns accepted invalid geometry/budgets | **CLOSED; RUNTIME PASS** | `_validate_nms_inputs` at `nms_deterministic.py:55-76` runs before empty/single-box returns in both public helpers (`:95-103`, `:227-235`), validates full canonical finite geometry, positive l/w/h, aligned lengths, and positive pre/post budgets. Job 336731 passed all parametrized invalid-geometry/budget cases for both helpers. |

The remediation did not waive, reinterpret, or move any finding into S07-B.

## O-017/O-018 contract re-audit

| Review item | Independent result |
|---|---|
| Frozen source of truth | **PASS STATIC**. The five MIT BEVFusion files at immutable `326653dc06e0938edf1aae7d01efcd158ba83de5` independently recomputed to the HANDOFF hashes `3425d0...`, `61af3c...`, `3d5b3d...`, `9b2924...`, `f466f0...`. CenterPoint v0.2 resolved to `e9ef04c3715aa3342fa42f4f4e064db987def6ad`; its three cross-check hashes also matched. |
| Six tasks and fields | **PASS STATIC**. Official order is car; truck/construction vehicle; bus/trailer; barrier; motorcycle/bicycle; pedestrian/traffic cone. Every task owns heatmap/reg/height/dim/rot/vel two-convolution branches after one shared transform. |
| GroupNorm adaptation | **PASS, including runtime**. Shared and per-field normalization is GroupNorm, with no batch-statistics layer. Job 336731 passed the B=1 versus B>1 distractor fixture at `test_head_capacity.py:51-60`. |
| Per-class K/no second task K | **PASS STATIC**. `centerhead_decode.py:220-237` selects up to 500 independently per class; concatenation proceeds directly to O-018 ordering and NMS. No second task-wide K exists. One/two-class tasks admit at most 500/1000 candidates. |
| No-starvation claim boundary | **PASS**. The implementation and HANDOFF explicitly limit the claim to removal of pre-NMS task-wide top-K starvation. Official task-wide suppression and post=83 remain and can still remove a class. No multi-class exact official-decode claim is made. |
| Deterministic tie order | **PASS STATIC**. `nms_deterministic.py:21-39` implements score descending, canonical global class ID ascending, flattened spatial index ascending. Candidate scores are FP32 before this key. |
| Explicit global labels | **PASS STATIC**. Name mapping is exactly `[(0,), (1,4), (2,3), (9,), (6,7), (5,8)]`; no cumulative task-offset path exists. Fixtures explicitly cover construction vehicle, bus, barrier, pedestrian, and traffic cone. |
| Score/range/budgets | **PASS STATIC**. Strict `score > 0.1`; inclusive `[-61.2,-61.2,-10,61.2,61.2,10]`; per-class 500; NMS pre/post 1000/83; rotate IoU 0.2. At most `6*83=498`, under the official 500-box cap. |
| Circle NMS | **PASS, including runtime**. Squared metres, inclusive `dist2 <= threshold`, official task-wide behavior, deterministic priority, full canonical prevalidation, and invalid-budget fixtures all passed in Job 336731. |
| Rotate NMS | **PASS, including runtime**. Canonical `(l,w)` CCW-yaw polygons, per-local-class scale, task-wide cross-class suppression and strict `IoU > 0.2` are coherent. Actual-source identity/disjoint/perpendicular cases returned `1/0/1/3`; 5,000 seeded random pairs stayed symmetric/in-range (worst symmetry residual `6.22e-15`). Job 336731 passed the Torch wrapper geometry, permutation, cross-class, budget, and fail-closed fixtures. |
| Box/yaw/velocity | **PASS STATIC**. Gravity-center `(x,y,z,l,w,h,yaw)`, `atan2(sin,cos)`, LiDAR `(vx,vy)`, rigid global lift and official `size=(w,l,h)` are preserved. |
| Submission/eval | **CHANGES-REQUESTED, test-only**. Invalid labels/geometry, duplicate tokens and >500 boxes passed. The hostile pair's full forward/reverse result equality and desired velocity/attribute content order passed; only a tuple-versus-list expected representation failed. No metric denominator, GT filter, class range, or ASR eligibility formula changed. |
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

After these local checks, S00 approved one exact O-009 synthetic job. Job `336731`
used no dataset, training/model step, optimizer update, scientific metric, profile,
merge, push, upload, or external publication. No retry or follow-on occurred.

## Runtime gate — executed once, negative preserved

The executed immutable `RUN_REQUEST.md` copy bound:

- exact request SHA-256
  `bcd8f426e5b95438f91973e9a3d9712193cf96a23f9254732114111fb68019c1`;
- exact durable committed launcher SHA-256
  `7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10`;
- worker `705216d`, tree `2d5cd99...`;
- 31-file source-list hash `bea19dd5...` and source-state hash `2ff6389f...`;
- immutable `/nobackup` snapshot
  `execution_snapshots/s05r2_centerhead_705216de097a`;
- exactly one shared GH200 allocation, eight CPUs, 15 minutes, maximum 0.25
  GPU-hours, with `--nodes` intentionally omitted;
- exactly four synthetic test files / 44 cases;
- no data, optimizer/model step, metric, profile, array, DDP, retry, or follow-on;
- exact allocation/runtime identity, JUnit, logs, source manifest, and in-job
  `sha256sum -c` artifacts.

Job `336731` terminal evidence is `FAILED 1:0`, elapsed `00:01:15`, batch MaxRSS
`504M`, with exactly one node/eight CPUs/one GH200 on `n570`. JUnit recorded
44 tests, one failure, zero errors/skips; the 43 passing cases include every
forced-FP32, GN, mapping, candidate, NMS, geometry, and fail-closed case. All nine
execution artifacts passed in-job checksum verification. Exact raw paths and
hashes are in `RESULTS.md`.

Key hashes: stdout `fbeac7db...`, stderr `ae633085...`, JUnit `0f79ed55...`,
pytest log `3e461e6e...`, execution identity `ca35c57e...`, allocation
`de382961...`, snapshot identity `0b405e5e...`, and checksum manifest
`4016aa0e...`. The executed request/launcher hashes remained exactly
`bcd8f426...` / `7ea5e812...`.

The post-execution `RESULTS.md` SHA-256 is
`9d2b25982cb2a1077b0db4349e1a57bc5287f48aa621a36dedf5fd38fc169004`;
the updated consumed-state `RUN_REQUEST.md` SHA-256 is
`b21f9d02193af88b867e506dbccaf7d65b6a7ab24cf0643f359859b1bc27c1b3`.

The approval is consumed. S05-R2 is **CHANGES-REQUESTED**, not PASS, until a worker
returns the narrow fixture correction and a fresh independent review accepts it.

## Allowed interpretations

- The exact remediation closes the first review's three source findings.
- O-018 task grouping, explicit global label map, per-class K/no second task K,
  deterministic ties, GN adaptation, canonical conversion, and official task-wide
  NMS constants are present and statically coherent.
- Static compilation, diff/ownership hygiene, upstream hashes, pure geometry, and
  hostile content-order checks passed.
- Job 336731 passed 43/44, including semantic forward/reverse equality and the
  desired parked-before-moving content order in the sole ultimately failing test.

## Forbidden interpretations

- S05 is accepted or reviewed PASS; the exact runtime gate returned 43/44.
- Multi-class decode is element-wise identical to official BEVFusion.
- The full authored Torch/devkit suite passed; one fixture assertion failed.
- Target/loss/detector/config/checkpoint integration or legacy-checkpoint migration
  is complete.
- CPU NMS production performance, mini/trainval quality, mAP/NDS, fusion gain,
  full-run readiness, FL/security behavior, generalization, or publication claims.

## Residual risk after a future accepted runtime PASS

Even if the exact 44 cases pass, S07-B must still integrate the S02 target/loss
mapping, bind all head/decode/NMS fields into resolved config/provenance, reject old
single-head checkpoints, run official devkit end-to-end conversion checks, and
profile the correctness-first rotate NMS before any full-run request. These are
explicit downstream integration gates and must not be silently reclassified as
S05 implementation defects or waived by this focused runtime.
