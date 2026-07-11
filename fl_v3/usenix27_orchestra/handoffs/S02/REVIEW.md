# S02-R independent review — CL P0 correctness

## Verdict

**CHANGES-REQUESTED.** The implementation diff contains no confirmed correctness
defect in the reviewed per-sample pillar-cap or Gaussian-target semantics, and the
CPU evidence is internally consistent. However, the binding S02 gate requires one
GPU forward/backward pass. Both delivered jobs intentionally set
`CUDA_VISIBLE_DEVICES=""`; the worker explicitly records the GPU gate as not run.
S02 therefore cannot yet be accepted as a reviewed S07-B dependency.

Reviewed worker delivery:

- base: `372de9398ae435f82b83367a922fd302c0635738`;
- worker/delivery SHA: `7ad396ebe535ca468337ed44065d39354707e08b`;
- implementation SHA: `65c83c077210469861ba722a285ab1e58e6d719f`;
- first executable: `a877ea0ecdc510350e03843ec66b9a679cdb6f37`;
- parser-remediation executable: `840e8bee8d1157c71b7752d3937c6cb8e75201e7`;
- expected source branch: `codex/s02-cl-p0-correctness`;
- review worktree preflight: clean, detached, exact HEAD `7ad396e...`.

## Findings

### [P1] Required GPU forward/backward evidence is absent

The canonical S02 contract requires both the focused unit suite and one GPU
forward/backward pass (`SESSIONS.md:240-246`). The handoff records the latter as
`NOT REQUESTED / NOT RUN` (`HANDOFF.md:172-189`), while its request explicitly
forbids a GPU forward/backward and hides CUDA (`RUN_REQUEST.md:11-16`). Raw
execution identities for Jobs `335565` and `335578` both confirm
`cuda_visible_devices: ""`; `RESULTS.md:179-181,236-242` likewise forbids a GPU
correctness inference.

This matters for the changed pillar path: stable device sorting, selected-rank
compaction, GroupNorm/backward, max-pooling, and unique-index `index_copy_` are
precisely the operations whose CUDA behavior is not established by CPU execution.
The missing evidence cannot be delegated to S07-B or converted into PASS without a
recorded amendment to the binding gate.

Required remediation: preserve both existing job histories, prepare a new exact
S00-audited request for one bounded synthetic CUDA forward/backward smoke against
the unchanged implementation, exercise B>1 plus per-sample/point over-cap and an
empty sample, assert finite output/loss/parameter gradients and the cap/isolation
diagnostics, and record scheduler/source/artifact checksums. It must contain no
optimizer step, mini/trainval traversal, metric, profile, automatic retry, or scope
expansion. Any source/test change requires re-review of the resulting exact SHA.

## Implementation audit

### Per-sample pillar cap

The former global truncation (`P > max_pillars` followed by a prefix slice) is
removed. At `lidar_encoder.py:186-264`, globally sorted unique cell keys are split
by sample; local ranks are derived from per-sample segment offsets; selection is
`pillar_rank_in_sample < max_pillars`; only selected pillars are compacted and
materialized. Because the primary key is `(batch, local row-major cell)`, adding or
reordering other samples cannot consume an existing sample's quota. Selected cell
keys remain unique before `index_copy_` at `lidar_encoder.py:300-311`.

The diagnostics account separately for:

- raw and in-range points;
- occupied, selected, and truncated pillars;
- selected-pillar point-cap drops;
- all points in pillar-cap-dropped cells;
- kept points, truncation fraction, selected sample ids, and local cell keys.

The hierarchy is internally consistent: all points in a rejected pillar count as
pillar-cap drops, while point-cap drops are measured only inside selected pillars.
Empty batches/samples receive full zero-valued per-sample records. Constructor
validation now rejects non-positive point/pillar caps.

The exact CPU test inventory includes B=1 versus B>1 equivalence, batch
permutation, adding an unrelated sample, simultaneous point/pillar overflow,
input-point permutation, per-sample counts/keys/fractions, completely empty input,
and an empty sample inside B=3. No batch-global accumulating scatter/top-k path was
introduced in the reviewed diff.

### Official Gaussian/target semantics

I independently checked the pinned upstream sources:

- MIT BEVFusion commit `326653dc06e0938edf1aae7d01efcd158ba83de5`,
  `mmdet3d/core/utils/gaussian.py` and
  `mmdet3d/models/heads/bbox/centerpoint.py`;
- CenterPoint v0.2 commit
  `e9ef04c3715aa3342fa42f4f4e064db987def6ad`,
  `det3d/core/utils/center_utils.py`;
- BEVFusion nuScenes CenterHead config: `gaussian_overlap=0.1`,
  `min_radius=2`.

For `(h,w)` and `o=0.1`, the production code uses the upstream three candidates:

```text
r1 = (h+w + sqrt((h+w)^2 - 4*w*h*(1-o)/(1+o))) / 2
r2 = (2*(h+w) + sqrt((2*(h+w))^2 - 16*(1-o)*w*h)) / 2
r3 = (-2*o*(h+w) + sqrt((-2*o*(h+w))^2 - 16*o*(o-1)*w*h)) / 2
radius = max(2, int(min(r1,r2,r3)))
```

All denominators are the official constant `/2`; the rejected alternative
`/(2*a)` is absent. Independent stdlib recomputation matched the four committed
fixtures `(1,1)`, `(4,8)`, `(10,20)`, and `(6,16)` and their final radii
`2,2,6,4`. Independent little-endian float32 reconstruction matched:

- radius-2 patch SHA-256
  `8f9723645f12fa7cb378ebf0f251ff6d564389b1977f63338bfe0a12c0dae0c6`;
- clipped/overlapping 7x8 target SHA-256
  `d64ecf1a961e304809615aecb644593a62a09dcf334255e2b507a92d56c2a9b8`.

The patch uses diameter `2r+1`, sigma `diameter/6`, NumPy float64 generation,
float32 conversion, epsilon clipping, and maximum overlay. Center-to-grid units
remain head-cell units (`dx/head_vx`, `dy/head_vy`), and the single-class target
fixture checks clipping plus GT-order invariance. The migration warning correctly
states that checkpoints trained under the prior mixed-denominator targets are not
valid resume/scientific evidence under this target definition and require
retraining.

One non-blocking arithmetic residual remains: the framework-independent helper
evaluates roots with Python `math` after float32 dimensions are transferred to the
CPU, whereas upstream BEVFusion evaluates tensor roots in float32 before `int`.
Artificial values arbitrarily close to an integer-radius boundary can therefore
round to adjacent integers. A read-only scan of 18,538 mini annotation sizes found
no final-radius discrepancy, but mini is not scientific evidence and full
trainval was not scanned. S07-B should retain the exact committed helper/golden
contract rather than claim bitwise equality to every upstream floating-point
execution.

## Raw execution and provenance reconciliation

### Job 335565 — preserved negative

Read-only `sacct` reconciliation:

- state/exit: `FAILED 1:0`;
- elapsed/limit: `00:01:35 / 00:10:00`;
- node: `n507`; requested/allocated one GH200, four CPUs, 5,836 MiB;
- `Restarts=0`.

JUnit independently aggregates to `12/0/0/0`; all twelve requested case names are
present and pytest reports `12 passed in 17.31s`. The launcher then misread the
`testsuites` root as zero tests, exited, and never wrote `sha256sums.txt`. Stderr
contains exactly that parser failure after the normal module-purge notice. This job
remains an overall failure. Its old launcher entry does not match the later
delivery working tree, as expected; the recorded hash
`f82c62624deb1da854fc7291beed212fa9eccf29e01c754fcdb7c8d534d35d9c`
does match the launcher blob at exact executable `a877ea0...`. The other fifteen
source entries match directly.

### Job 335578 — separate parser remediation PASS

Read-only `sacct` reconciliation:

- state/exit: `COMPLETED 0:0`;
- elapsed/limit: `00:01:33 / 00:10:00`;
- node: `n534`; requested/allocated one GH200, four CPUs, 5,836 MiB;
- `Restarts=0`.

The exact executable/source state is `840e8bee...` / `2ff7d742...`. Pytest reports
`12 passed in 19.86s`; JUnit independently aggregates to `12/0/0/0`; every listed
runtime source matches; and the final four-artifact `sha256sum -c` passes. The only
executable delta from the preserved negative package is the JUnit suite-count
aggregation. Job 335578 is a valid CPU remediation result, not a rewrite of Job
335565 and not GPU evidence.

Verified delivery-document SHA-256 values:

| File | SHA-256 |
|---|---|
| `HANDOFF.md` | `3756e2eeb8a8f1b9ff0d06df2daec96da5f744d5b07cc264c333bd84792f7923` |
| `RESULTS.md` | `f5b765989aaf23c13b3956489e9113da92c8296e813791cfe619838ef5d09294` |
| `RUN_REQUEST.md` | `3eb4ca22ee9d33a16bbd94eedb5daa8a7be38842616852358c5959f25f56420f` |

The exact base-to-delivery diff modifies only the two owned model files, one
focused test, and `handoffs/S02/**`; no canonical Orchestra, S03-S05, `fl_v2`, or
read-only `fl_v3/collab` file appears. `git diff --check`, `py_compile`, launcher
`bash -n`, source hashes, Git history, and delivery preflight all passed. Login-node
Python has no Torch, so no local tensor result is inferred beyond the checksummed
aarch64 job evidence.

## Gate checklist

| Gate | Independent verdict |
|---|---|
| Per-sample cap, no batch-global budget | PASS on code audit and CPU hostile tests |
| B=1/B>1 isolation and batch permutation | PASS on exact CPU evidence |
| Deterministic point/pillar over-cap selection | PASS on exact CPU evidence |
| Empty batch/sample and diagnostics | PASS on exact CPU evidence |
| Official constant-/2 roots, `o=0.1`, `min_radius=2` | PASS |
| Exact Gaussian patch/clipped max-overlay target | PASS |
| Old-checkpoint invalidation documented | PASS |
| Job 335565 terminal evidence | FAILED preserved correctly |
| Job 335578 terminal evidence | PASS, CPU only |
| One GPU forward/backward | **MISSING — BLOCKS S02 PASS** |
| Ownership/provenance/authorization | PASS for delivered history |

## Allowed and forbidden claims

Allowed now:

- the reviewed source implements the stated per-sample deterministic cap and the
  O-017 Gaussian formula/golden contract;
- the twelve exact CPU tensor tests passed twice on the recorded aarch64 runtime;
- Job 335565 is a preserved post-pytest evidence-pipeline failure, while Job
  335578 separately closes that parser/checksum path.

Forbidden now:

- `S02 PASS`, reviewed integration readiness, or GPU correctness;
- calling Job 335565 completed/passing or claiming it produced an in-job final
  checksum manifest;
- old-target checkpoint resume/scientific compatibility;
- mini/trainval readiness, performance/memory, model quality, mAP/NDS, fusion
  gain, FL/security, generalization, or publication claims.

## Residual risk and re-review boundary

After the exact synthetic GPU forward/backward evidence passes, re-review may be
limited to the new request/results/handoff and exact executable diff if model/test
semantics remain unchanged. Any implementation or fixture change reopens the full
code review. The focused tests do not establish full-stack integration, SECOND
behavior, full-data truncation rates, throughput, or scientific performance; those
remain S07-B and later-session gates.
