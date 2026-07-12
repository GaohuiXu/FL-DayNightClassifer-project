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
- Final reviewed worker delivery:
  `a9c801fdee378906e54d06314d0c772b6559901a` / tree
  `00aaa9570011e398060555a772eeed62db465721`.
- Final test execution SHA:
  `96e509b71a3e22afb4de397132438fd3b9bbf5d8` / tree
  `aeaaad044199492b81c4383a013f3fb3c6596c02`.
- Current verdict: **PASS for the S05 implementation and focused synthetic runtime gate**.

The limited final re-review found no remaining blocking issue. The worker's final
code change from reviewed delivery `705216d` to execution `96e509b` changes only
two hostile-fixture expected velocity containers from lists to tuples. Production
source, O-018 semantics, `forward == reverse`, and exact parked-before-moving
content-order assertions are unchanged. Preserved negative Job `336731` remains
43/44; separately approved Job `336738` executed the corrected immutable source
and passed 44/44 with exact identity, allocation, and checksum evidence. This PASS
does not cover S07-B detector/loss/config integration, NMS production performance,
model quality, scientific metrics, or any FL/security claim.

## Findings first

### Resolved prior test-only finding — stable tuple representation

Job `336731` ran the exact dependency-backed suite on nuscenes-devkit `1.1.11`.
`test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content`
first passed `assert forward == reverse`, proving the complete result dictionaries
were input-permutation invariant. It then observed the correct content order
`(0,0)/vehicle.parked`, `(5,0)/vehicle.moving`, but failed because actual
`record["velocity"]` values were tuples and the fixture expected lists.

This was not evidence of a production sort-key, velocity, attribute, or TP-pairing
defect, but correctly blocked the prior zero-failure gate. Commit `96e509b` applies
exactly the requested test-only correction: the two expected list literals become
tuples while full `forward == reverse` and the exact parked-before-moving
velocity/attribute assertion remain intact. Job `336738` then passed the corrected
case and all other 43 cases. No production or O-018 change was introduced.

## Closure of the first review findings

| First-review finding | Re-review result | Exact evidence |
|---|---|---|
| P1: fp16 logits decoded before official forced-FP32 boundary | **CLOSED; RUNTIME PASS** | `centerhead_decode.py:208-214` promotes every head field to fp32 before sigmoid; `:263-269` gathers fp32 reg/height/dim/rot/vel; `:281-287` returns fp32 scores/velocity; empty final outputs are fp32 at `:370-378`. Job 336731 passed both adjacent-binary16/strict-0.1 hostile cases and FP32 output assertions. |
| P1: submission content key omitted velocity/attribute | **CLOSED; RUNTIME PASS** | `box_to_global.py:234-245` includes velocity, class, and attribute. Job 336731 already passed complete `forward == reverse` and desired content order before the representation-only assertion failed. Commit `96e509b` changes only the expected containers to tuples; Job 336738 passed the complete hostile fixture. |
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
| Submission/eval | **PASS, including runtime**. Invalid labels/geometry, duplicate tokens and >500 boxes pass. The hostile pair's full forward/reverse result equality and exact tuple/attribute content order pass in Job 336738. No metric denominator, GT filter, class range, or ASR eligibility formula changed. |
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

Final worker/reviewer separation:

- execution commit `96e509b` is a direct child of worker delivery `705216d`;
- `705216d..96e509b` changes only
  `fl_v3/tests/test_s05_eval_roundtrip.py`, with binary diff SHA-256
  `aed0033a6843212557b14bc0b950006e3b791cd2a75afb7fd5d40938e79fc700`;
- corrected test SHA-256 is
  `e938dd34656e3ae5f5e9019748bea52a3ccc5cb99144492d6bf9f45e79c203c0`;
- final worker delivery `a9c801f` adds only handoff/request/results/launcher
  records after execution; implementation/test execution remains `96e509b`;
- the review branch and final worker branch diverge at `705216d`. Before this
  final review commit, reviewer HEAD was `29bdb46`; merge-base with final worker
  `a9c801f` was exactly `705216d` (`3` review-only versus `5` worker-side commits).
  The review branch does not contain the worker's test remediation or implementation
  history and must never be used as the implementation branch.

Verified durable artifacts:

- final HANDOFF SHA-256:
  `915061747a22d98cfc12b192ddb9a4ccd69fc661e7c5c29c525de4254ad30dbd`;
- first REVIEW SHA-256:
  `270d47c498af7aaaeef1af535be7513223bd6793c933b9ca721aed223d0a79e5`;
- remediated file hashes all match the HANDOFF, including
  `centerhead_decode.py=13d22b...`, `nms_deterministic.py=9adcc7...`,
  `box_to_global.py=08c2ab...`, and the three hostile test hashes.
- final worker HANDOFF SHA-256:
  `332a1a3e608a340b2c322a523ee67a94d0821b66571b2263d2616f2347839d2e`;
- final worker RUN_REQUEST ledger / RESULTS / launcher SHA-256:
  `b434469419d0642b198604d1f8e2dcfb92da93e36c8bfc716d4b952a883c3706` /
  `43650fc0ade15c04afc3391f675e95ada7d2ce7ba8526ac392552b1c2938f194` /
  `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5`.

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
9. Final `705216d..96e509b` diff/name/production-path audit, corrected-test hash,
   binary diff hash, 31-file source-list and source-state recomputation: **PASS**.
10. Job 336738 `sacct`/`scontrol`, stdout/stderr, execution/snapshot identities,
    JUnit counts, dependency versions, approval class, all artifact SHA-256 values,
    and independent `sha256sum -c`: **PASS**.
11. Job 336731 terminal/JUnit/raw hashes and independent checksum revalidation:
    **preserved FAILED 43/44 evidence**, unchanged.

After these local checks, S00 approved one exact O-009 synthetic job. Job `336731`
used no dataset, training/model step, optimizer update, scientific metric, profile,
merge, push, upload, or external publication. No retry or follow-on occurred.
S00 separately approved Job `336738` under
`S00_OWNER_DELEGATED_S02_S05_VALIDATION_RERUN`; this approval was explicitly not
an O-009 expansion. This final reviewer submitted no new job and changed no worker,
production, or test file.

## Runtime ledger — negative preserved, corrected gate passed

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

The worker then returned execution SHA `96e509b` with only the approved tuple
expectation correction. Its separately approved immutable request copy SHA-256 was
`e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9`,
launcher SHA-256 was `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5`,
and approval class was exactly
`S00_OWNER_DELEGATED_S02_S05_VALIDATION_RERUN`.

Job `336738` completed `0:0` on `n411` in `00:01:13`, batch MaxRSS `540M`,
with one shared node/eight CPUs/exactly one GH200 and no requeue/restart. Runtime
identity independently matched execution `96e509b`/tree `aeaaad0`, 31-file list
`bea19dd5...`, source state `7ac7ea66...`, aarch64 Python `3.11.15`, Torch
`2.11.0+cu128`, NumPy `1.26.4`, pytest `9.1.1`, nuscenes-devkit `1.1.11`, and
Pillow `12.2.0`. It recorded no dataset, optimizer update, or scientific metric.

JUnit contains exactly 44 cases, zero failures/errors/skips, time `22.645s`; stdout
reports `44 passed in 22.64s`. All nine checksum targets independently returned
`OK`. Raw hashes exactly match worker-final `RESULTS.md` (`43650fc0...`): stdout `0cf6f1dc...`,
stderr `ae633085...`, JUnit `bad9b34e...`, pytest log `4db65ef4...`, execution
identity `9e2dde24...`, allocation `c76ffe82...`, snapshot identity `6c47a425...`,
and checksum manifest `301c5c4f...`.

| Job 336738 raw artifact | Independently recomputed SHA-256 |
|---|---|
| stdout | `0cf6f1dc14ad07ef598076fb6ed067352bf71c789172f9babd5f1ed42d01ef87` |
| stderr | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |
| approved launcher | `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5` |
| approved request copy | `e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9` |
| snapshot identity | `6c47a4252bb65c227ef795eecd161749e5260ce6821a5a638da7b5457ab0aa20` |
| runtime source list | `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857` |
| runtime source hashes | `7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1` |
| execution identity | `9e2dde2468f17d10b99c2992440029b347f4b4a220143c3aecce7c6b84a62aab` |
| allocation record | `c76ffe8201b2025d7ed7b0cbf663fca8706073c10efee090fac0ed2347dba3d8` |
| pytest log | `4db65ef4592e61cf1886e49bef9649ba87803b6cf41bc45e84de6484645121d3` |
| JUnit XML | `bad9b34e02a4d7267cbbed4e2b4429c6498360a3c3317388fdc21f0be8206910` |
| checksum manifest | `301c5c4feed506f0ae5c130b1036cfe0c0aaeacf81f947cf121a6136f7339077` |

Both approvals are consumed and no further compute is implied. Job `336731`
remains a failed historical gate; Job `336738` is the separate accepted corrected
gate. On this combined evidence S05-R2 final verdict is **PASS**.

## Allowed interpretations

- The exact remediation closes the first review's three source findings.
- O-018 task grouping, explicit global label map, per-class K/no second task K,
  deterministic ties, GN adaptation, canonical conversion, and official task-wide
  NMS constants are present and statically coherent.
- Static compilation, diff/ownership hygiene, upstream hashes, pure geometry, and
  hostile content-order checks passed.
- Job 336731 executed 43 passing cases and one preserved failure; semantic
  forward/reverse equality and the desired parked-before-moving content order had
  already succeeded before the representation-only assertion failed.
- Job 336738 independently passed all 44 immutable synthetic fixtures at execution
  SHA `96e509b`, closing the focused S05 runtime gate.

## Forbidden interpretations

- Job 336731 passed or was erased; it remains a preserved 43/44 failure.
- Multi-class decode is element-wise identical to official BEVFusion.
- Job 336738 establishes production detector/loss/config integration, official
  CUDA-kernel parity, or CPU-NMS production performance.
- Target/loss/detector/config/checkpoint integration or legacy-checkpoint migration
  is complete.
- CPU NMS production performance, mini/trainval quality, mAP/NDS, fusion gain,
  full-run readiness, FL/security behavior, generalization, or publication claims.

## Residual risks after S05 PASS

Although the exact 44 cases pass, S07-B must still integrate the S02 target/loss
mapping, bind all head/decode/NMS fields into resolved config/provenance, reject old
single-head checkpoints, run official devkit end-to-end conversion checks, and
profile the correctness-first rotate NMS before any full-run request. These are
explicit downstream integration gates and must not be silently reclassified as
S05 implementation defects or waived by this focused runtime.
