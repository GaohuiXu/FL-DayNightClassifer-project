# S02 HANDOFF — CL P0 correctness

## Session identity and current self-assessment

- Session: `S02`.
- Base SHA: `372de9398ae435f82b83367a922fd302c0635738`.
- S07-A foundation named by kickoff:
  `0249eb21a32730ac1689255491b19a158711401f`.
- Branch: `codex/s02-cl-p0-correctness`.
- Implementation commit:
  `65c83c077210469861ba722a285ab1e58e6d719f`.
- Exact first validation executable:
  `a877ea0ecdc510350e03843ec66b9a679cdb6f37`.
- Initial request SHA-256:
  `60b0b923d527b60a34449ddb7d24678e85e68ca187d453c18809368637ed50c9`.
- Initial source state:
  `5ff316b81233d4a367ded2928ebacb2f90ae240485003af2f701c54f22c560fa`.
- Final delivery `WORKER_SHA` is the documentation commit created after the checks
  below and is reported to S00 in the task response; a commit cannot embed its own
  SHA without changing that SHA.
- Worker self-assessment after Job 335565 and the owner-authorized parser-only
  remediation:
  **CHANGES-REQUESTED — CODE TESTS 12/12 PASS; INITIAL JOB 335565 FAILED IN POST-PYTEST
  JUNIT PARSER AND REMAINS PRESERVED; MANUAL PARSER-ONLY JOB 335578 COMPLETED
  0:0 WITH FINAL CHECKSUM VERIFICATION; S02-R FOUND NO IMPLEMENTATION DEFECT BUT
  REQUIRES THE CANONICAL GPU FORWARD/BACKWARD EVIDENCE; EXACT REMEDIATION REQUEST
  PREPARED AND NOT SUBMITTED**.

No merge, push, PR, upload, branch/worktree deletion, mini/trainval traversal,
model/scientific execution, or unapproved follow-on occurred.

## Scope and files

Modified within exclusive S02 ownership:

- `fl_v3/src/fl_v3/models/fusion/lidar_encoder.py`;
- `fl_v3/src/fl_v3/models/fusion/losses.py`.

Added within S02 ownership:

- `fl_v3/tests/test_s02_p0_correctness.py`;
- `fl_v3/tests/test_s02_gpu_forward_backward.py`;
- `fl_v3/usenix27_orchestra/handoffs/S02/run_s02_cpu_tests.sh`;
- `fl_v3/usenix27_orchestra/handoffs/S02/run_s02_gpu_forward_backward.sh`;
- `fl_v3/usenix27_orchestra/handoffs/S02/{RUN_REQUEST,RESULTS,HANDOFF}.md`.

Read-only `bev_grid.py`, `detector.py`, `training/tasks.py`, S03-S05 ownership,
canonical Orchestra documents, `fl_v3/collab/`, and `fl_v2/` were not modified.

## Official Gaussian reference and exact semantics

Primary source:

- MIT BEVFusion commit
  `326653dc06e0938edf1aae7d01efcd158ba83de5`,
  `mmdet3d/core/utils/gaussian.py`,
  `mmdet3d/models/heads/bbox/centerpoint.py:505-515`, and
  `configs/nuscenes/det/centerhead/default.yaml:8-17`;
- corroborating official CenterPoint v0.2 commit
  `e9ef04c3715aa3342fa42f4f4e064db987def6ad`,
  `det3d/core/utils/center_utils.py`.

For head-grid object dimensions `(h,w)` and `o=min_overlap`, the exact approved
candidate roots are:

```text
a1 = 1
b1 = h + w
c1 = w*h*(1-o)/(1+o)
r1 = (b1 + sqrt(b1^2 - 4*a1*c1)) / 2

a2 = 4
b2 = 2*(h+w)
c2 = (1-o)*w*h
r2 = (b2 + sqrt(b2^2 - 4*a2*c2)) / 2

a3 = 4*o
b3 = -2*o*(h+w)
c3 = (o-1)*w*h
r3 = (b3 + sqrt(b3^2 - 4*a3*c3)) / 2
```

Every root uses the official constant `/2`, not an alternative `/(2*a)` geometric
derivation. nuScenes target rendering uses `o=0.1` and
`radius=max(2, int(min(r1,r2,r3)))`; positive `int` truncates toward zero. Gaussian
rendering uses diameter `2r+1`, sigma `diameter/6`, NumPy float64 generation,
float32 conversion, clipping at NumPy epsilon times the maximum, and clipped
elementwise maximum overlay.

Pinned discriminating fixtures:

| `(h,w)` | official `(r1,r2,r3)` or minimum | final radius | old mixed result |
|---|---|---:|---:|
| `(1,1)` | `(1.426401432711221, 2.632455532033676, 0.432455532033676)` | 2 | 2 |
| `(4,8)` | `(9.133397807202561, 17.366563145999496, 2.4)` | 2 | 4 |
| `(10,20)` | minimum `6.0` | 6 | 10 |
| `(6,16)` | minimum `4.076941930590086` | 4 | 8 |

The full radius-2 `5x5` patch and a clipped/overlapping `7x8` target tensor are
committed as exact float32 values. Their independently derived little-endian
float32 byte hashes are `8f9723645f12fa7cb378ebf0f251ff6d564389b1977f63338bfe0a12c0dae0c6`
and `d64ecf1a961e304809615aecb644593a62a09dcf334255e2b507a92d56c2a9b8`.

## LiDAR pillar semantic changes

The old encoder sorted a batch-global key and then sliced the first
`max_pillars`, allowing earlier samples to consume later samples' complete budget.
The new path:

1. retains the existing canonical `(pillar key, x, y, z, intensity[,dt])`
   content sort for point-order invariance;
2. derives each occupied pillar's sample and local row-major cell rank;
3. selects the first `max_pillars` independently for every sample;
4. applies that selection before PFN and padded slot materialization, bounding the
   slot tensor by `B * max_pillars * max_points`;
5. compacts selected pillar ids deterministically with integer cumsum;
6. scatters only unique selected global cells.

`last_pillar_meta` now records device tensors after every forward:

- input and in-range point counts per sample;
- occupied, selected, and truncated pillar counts per sample;
- truncation fractions;
- kept points and points dropped separately by point cap versus pillar cap;
- selected pillar batch ids and local row-major keys.

Empty batches/samples receive complete zero-valued per-sample diagnostics.

## Checkpoint migration

Every old fl_v3 checkpoint was trained with different target heatmaps from the
mixed-denominator implementation. The new target semantics are not resume-compatible:
old checkpoints require retraining and cannot be reused as evidence under the new
loss definition. No checkpoint conversion is claimed or provided.

## Verification before material compute

Local/login-node checks:

- `git diff --check`: PASS;
- `python3 -m py_compile` over both changed modules and the focused test: PASS;
- `bash -n` on the bounded launcher: PASS;
- actual `gaussian_radius` AST extraction executed against the four numerical
  goldens using stdlib `math`: PASS;
- static AST scan of the changed LiDAR encoder found no banned accumulating
  scatter/reduce/top-k or unstable sort: PASS.

The x86 login Python lacked Torch, NumPy, and pytest. It was not treated as a test
PASS or failure. A full fusion-directory static scan also observes a pre-existing
`scatter_reduce_` in read-only S04 `sparse_voxel_encoder.py`; S02 did not edit or
silently waive that cross-session issue.

## Job 335565 result and negative evidence

S00 approved exactly one CPU-only synthetic job under O-017/O-009. The request was
bound to request hash `60b0b923...`, executable `a877ea0...`, source state
`5ff316b8...`, one node/GH200 allocation, four CPUs, ten minutes, and exact 12
tests. Preflight matched and S02 submitted Job `335565` once.

Observed result:

- pytest/JUnit: **12 tests, 12 passed, zero failures/errors/skips**, `17.31s`;
- exact source/identity: PASS, all 16 source hashes independently verified;
- `CUDA_VISIBLE_DEVICES=""`, so tensor execution was CPU-only;
- scheduler: **FAILED `1:0`**, elapsed `00:01:35`, node `n507`, `Restarts=0`;
- failure stage: after pytest, the launcher read count attributes from a
  `testsuites` root instead of aggregating its child `testsuite`, misreported zero
  tests, and exited;
- final in-job `sha256sums.txt`: **missing**, because fail-closed exit preceded its
  generation;
- no retry/requeue/resubmit/follow-on.

Raw scheduler fields, every available artifact/log path and SHA-256, complete JUnit
case inventory, and the exact negative interpretation are in `RESULTS.md`.

## Acceptance checklist

| S02 gate | Worker evidence/status |
|---|---|
| reorder batch samples only reorders outputs | TEST PASS |
| adding another sample cannot change existing sample BEV | TEST PASS |
| B=1/B>1 equivalence | TEST PASS |
| every sample independently respects `max_pillars` | TEST PASS |
| deterministic local selection and point permutation | TEST PASS |
| occupancy/truncation diagnostics observable | TEST PASS |
| empty batch and empty samples | TEST PASS |
| official radius numerical goldens | TEST PASS |
| exact Gaussian patch/heatmap target | TEST PASS |
| focused CPU pytest suite | 12/12 TESTS PASS |
| initial exact O-009 job acceptance | **FAIL preserved: Job 335565 scheduler 1:0, missing final checksum manifest** |
| manual parser-only exact job acceptance | PASS: Job 335578 `COMPLETED 0:0`, 12/0/0/0, final checksums OK |
| GPU forward/backward | **NOT RUN; exact S02-R remediation prepared pending S00 approval** |
| independent S02-R | `CHANGES-REQUESTED`; implementation/Gaussian audit passed, GPU evidence is the only blocker |

## Allowed and forbidden interpretations

Allowed:

- exact code semantics and committed fixtures described above;
- all twelve approved CPU tensor tests passed on the exact recorded aarch64
  dependency environment;
- source identity and JUnit 12/0/0/0 were independently reconciled;
- Job 335565 is a preserved negative post-pytest evidence-pipeline result.

Forbidden:

- calling Job 335565 an overall PASS/COMPLETED job;
- claiming final in-job checksum-manifest verification;
- treating worker self-assessment as independent/integration PASS;
- old-checkpoint compatibility;
- GPU correctness, full-stack readiness, mini/trainval model behavior,
  performance/memory, mAP/NDS, fusion gain, FL, attack/defense, generalization,
  scientific, or publication conclusions.

## Remaining risks and requested S00 decisions

1. Preserve Job 335565 as overall FAILED while retaining the positive 12-test
   evidence and missing-checksum fact; Job 335578 is a separate manual remediation,
   not a rewrite of that history.
2. After durable delivery, launch independent S02-R from the exact worker SHA;
   reviewer must recompute Gaussian fixtures and hostile B>1 cases rather than
   relying only on worker prose.
3. S07-B must integrate S02 only after independent review and must treat all old
   target-trained checkpoints as requiring retraining.

## Independent S02-R verdict and scoped GPU remediation preparation

Independent S02-R review commit
`fb17da3ea55a93d7709f6a2b5f6e4bb6adc0bf7e` (REVIEW.md SHA-256
`75b6a5ed589c1f29ba847750a915732be8826562c055a7fa1cecd5a749e63497`)
returned `CHANGES-REQUESTED`. It found no confirmed defect in the reviewed
per-sample cap or O-017 Gaussian implementation. Its sole P1 is that Jobs 335565
and 335578 intentionally hid CUDA, while the canonical S02 gate requires one GPU
forward/backward.

The remediation adds no model/Gaussian/pillar semantic change. It adds one bounded
synthetic CUDA test and one `/nobackup` snapshot launcher. The B=3 fixture has two
point-and-pillar-over-cap samples plus one empty sample; it checks finite output/
loss, B=1-versus-batched isolation, exact cap/drop/key diagnostics, empty output,
and finite nonzero gradients for every intended encoder parameter. There is no
optimizer/GradScaler step, data access, metric, profile, 100/1000-step gate, or
automatic retry.

The new `RUN_REQUEST.md` section binds the eventual executable SHA/tree and
request/launcher/source hashes externally, runs from a unique Git-archive snapshot
under `/nobackup`, requests one shared GH200 without `--nodes=1`, records and fails
closed on actual one-GPU visibility/allocation, and performs in-job source/final
artifact `sha256sum -c`. It remains `PENDING S00 APPROVAL`; no snapshot, output,
Slurm job, merge, or push has been created by this preparation.

## S00 post-job decision and parser-only remediation preparation

S00 required the first job to remain overall `FAILED 1:0` while preserving the
positive pytest/JUnit evidence and missing final checksum. That negative package
was committed independently as `b848a6f` before any launcher edit.

S00 then authorized only a manual evidence-parser remediation preparation. Commit
`840e8bee8d1157c71b7752d3937c6cb8e75201e7` changes only the JUnit aggregation
block in the handoff-owned launcher:

- root `testsuite` is counted directly;
- otherwise every descendant `testsuite` under `testsuites` is aggregated;
- absence of a suite fails closed;
- exact count remains `12/0/0/0`.

The parser was checked locally against Job 335565's raw JUnit and synthetic
single-/multi-suite root shapes. Model files, test files/nodes, expected count,
resources, data scope, and all scientific boundaries are unchanged.

The new source-state hash is
`2ff7d74246e55332305e92a83dc028a42ce3c1e60993c28c24ece868784e580a`;
the new unique output is
`.../outputs/s02_cpu_tests_840e8bee8d11`. The exact command and stop conditions
are appended to `RUN_REQUEST.md`.

S00 approved the finalized request hash `48aa4307...`, launcher hash
`35798c39...`, executable `840e8bee...`, source state `2ff7d742...`, and exact
unchanged scope once. S02 submitted Job `335578` once. It completed `0:0` on
`n534` in `00:01:33`, with `Restarts=0`, pytest `12 passed in 19.86s`, JUnit
12/0/0/0, empty `CUDA_VISIBLE_DEVICES`, all 16 sources verified, and the final
four-artifact in-job checksum list passing. No retry/requeue/resubmission/follow-on
occurred. `RESULTS.md` contains full raw hashes and both scheduler histories.
