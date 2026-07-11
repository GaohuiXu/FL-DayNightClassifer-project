# S02 RUN_REQUEST — focused CPU correctness suite

## Approval state and boundary

- **Status:** `FIRST_EXECUTION_FAILED_POST_PYTEST_PRESERVED; MANUAL_PARSER_REMEDIATION_EXECUTED_ONCE_COMPLETED_PASS; NO_FURTHER_SUBMISSION_AUTHORIZED`.
- This is one bounded, non-scientific engineering validation request under the
  O-017 rule for Wave-A workers and the O-009 resource ceiling. O-017 requires S02
  to stop and wait for explicit S00 approval even though the request fits O-009.
- Preparing this file is not execution approval. No `sbatch`/`srun` has been
  submitted by S02 before the exact approval recorded below.
- The request is CPU-only at the tensor/test level. One GH200 is allocated solely
  because the validated aarch64 PyTorch environment is compute-node-only; the
  launcher sets `CUDA_VISIBLE_DEVICES=""` before importing Torch or running pytest.
- This request does not authorize a GPU forward/backward smoke, mini-data access,
  trainval traversal/cache/profile/evaluation, model step, 100/1000-step gate,
  metric, matrix, seed, retry, resubmission, follow-on, upload, merge, or push.

## Immutable identity

- Branch: `codex/s02-cl-p0-correctness`.
- S07-A foundation SHA named by kickoff:
  `0249eb21a32730ac1689255491b19a158711401f`.
- S02 base SHA:
  `372de9398ae435f82b83367a922fd302c0635738`.
- Model/test implementation commit:
  `65c83c077210469861ba722a285ab1e58e6d719f`.
- Exact executable HEAD including the bounded launcher:
  `a877ea0ecdc510350e03843ec66b9a679cdb6f37`.
- Working source diff before this audit record: empty; SHA-256 of
  `git diff --binary HEAD` was
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  This uncommitted request prose is not an executable source input.
- Exact 16-file runtime source-state SHA-256:
  `5ff316b81233d4a367ded2928ebacb2f90ae240485003af2f701c54f22c560fa`.
- The source set is C-locale sorted and contains:
  `fl_v3/{pyproject.toml,requirements.txt,requirements.lock.txt}`;
  `fl_v3/scripts/arrhenius_env.sh`; package/model/utils initializers;
  `bev_grid.py`, `lidar_encoder.py`, `losses.py`, `utils/runtime.py`;
  `tests/{conftest.py,test_model_determinism.py,test_s02_p0_correctness.py}`;
  and `handoffs/S02/run_s02_cpu_tests.sh` itself.
- The launcher verifies exact HEAD, exact authorized branch, no tracked diff in
  any runtime source, and the aggregate source hash before environment activation
  or output creation.

Any change to the SHA, runtime source hash/list, tests, command, resources, output,
or stop conditions invalidates approval and requires a new S00 audit.

## Data and test scope

- Input data: synthetic tensors only, created in process by the tests.
- No nuScenes mini or shared trainval path is read. `NUSCENES_DATA_DIR`, ZIP
  manifests, info caches, GT databases, checkpoints, and model outputs are outside
  this request.
- Exact pytest nodes:
  1. all cases in `fl_v3/tests/test_s02_p0_correctness.py`;
  2. `test_model_determinism.py::test_pillar_scatter_permutation_invariant`;
  3. `test_model_determinism.py::test_pillar_scatter_permutation_invariant_OVERCAP`.
- Expected JUnit count: exactly 12 tests, zero failures, zero errors, zero skips.
- The suite covers official radius numerical goldens, exact Gaussian patch and
  clipped heatmap rendering, target order invariance, B=1/B>1 sample isolation,
  batch permutation, per-sample over-cap selection, point-order permutation,
  empty batch/sample behavior, occupancy, and every exposed truncation diagnostic.

## Resources, output, and command

- One node, one `nvidia_gh200_120gb`, four CPUs, walltime `00:10:00`.
- Maximum requested allocation: 0.167 GPU-hours. S02 cumulative use before this
  request is zero; one concurrent S02 job maximum.
- Job name: `flv3_s02_cpu_tests`.
- Exact unique output root, confirmed absent before this request:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_a877ea0ecdc5`.
- Logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_cpu_tests_%j.{out,err}`.

Exact command, executed once only after S00's explicit message approval:

```bash
test "$(git rev-parse HEAD)" = "a877ea0ecdc510350e03843ec66b9a679cdb6f37" && \
test "$(git branch --show-current)" = "codex/s02-cl-p0-correctness" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_a877ea0ecdc5 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s02/ {print}')" && \
sbatch --export=ALL,EXPECTED_S02_SHA=a877ea0ecdc510350e03843ec66b9a679cdb6f37,EXPECTED_S02_STATE_HASH=5ff316b81233d4a367ded2928ebacb2f90ae240485003af2f701c54f22c560fa,S02_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_a877ea0ecdc5 \
  fl_v3/usenix27_orchestra/handoffs/S02/run_s02_cpu_tests.sh
```

## Acceptance and stop conditions

Pass requires all of the following from this one submission:

- in-job SHA/branch/source preflight matches exactly;
- execution identity records aarch64, exact SHA/source hash, empty
  `CUDA_VISIBLE_DEVICES`, and installed NumPy/pytest/Torch versions;
- pytest and JUnit report exactly `12 passed`, zero failure/error/skip;
- runtime source hashes and execution/log/JUnit artifact checksums verify in-job;
- Slurm reports `COMPLETED` and exit `0:0` within the requested resources.

The job fails closed on identity/hash/branch drift, any tracked runtime-source
change, output collision, import/test failure, unexpected test count, any skip,
checksum failure, or walltime. There is no automatic retry, requeue, resubmission,
or follow-on. A failure is recorded as a negative result and returned to S00 before
any new request.

## Interpretation limits

Allowed if it passes: the exact focused CPU tensor paths and synthetic fixtures pass
on the recorded GH200/aarch64 dependency environment.

Forbidden even if it passes: GPU forward/backward correctness, trainval or mini
data readiness, performance/memory conclusions, model quality, mAP/NDS, fusion
gain, FL, attack/defense, generalization, scientific, or publication claims.

## Exact S00 approval and one-time execution record

S00 approved exactly one submission without editing the request first, bound to:

- request SHA-256 before execution:
  `60b0b923d527b60a34449ddb7d24678e85e68ca187d453c18809368637ed50c9`;
- branch `codex/s02-cl-p0-correctness`, executable HEAD `a877ea0...`,
  implementation `65c83c0...`, runtime source state `5ff316b8...`;
- the exact command/output/resources/test nodes and stop conditions above.

The pre-submit recheck matched every approved field: exact request hash, HEAD,
branch, empty tracked diff, absent output, and no active S02 job. S02 submitted
exactly once as Slurm Job `335565`. No retry, requeue, resubmission, or follow-on
occurred.

Final scheduler result: `FAILED`, exit `1:0`, elapsed `00:01:35` of `00:10:00`,
node `n507`, one GH200 allocation/four CPUs, `Restarts=0`. Pytest itself completed
successfully with `12 passed in 17.31s`, zero failures/errors/skips. The launcher
then failed closed because its JUnit parser read count attributes only from the XML
root. Pytest 9.1.1 emitted a `testsuites` root with the counts on its child
`testsuite`, so the launcher misread `tests=0` and exited before generating
`sha256sums.txt`.

This is a preserved negative gate result. The individual raw artifacts and logs
were independently hashed after termination, and all 16 runtime sources verified
against the immutable commit. Exact values are in `RESULTS.md`. The approval is
consumed; this updated record does not authorize a launcher fix or another job.

## Manual evidence-parser remediation request — PENDING S00 APPROVAL

### Why this is a new manual request

S00 explicitly authorized preparation, but not submission, of one narrowly scoped
manual remediation after preserving Job 335565 as overall `FAILED 1:0`. This is not
an automatic retry. The first request/approval is consumed and remains immutable
historical evidence above.

The only executable change after the negative-evidence commit is in
`handoffs/S02/run_s02_cpu_tests.sh`: its JUnit checker now uses the project-proven
S07 aggregation pattern, treating a root `testsuite` as one suite and otherwise
aggregating every descendant `testsuite` under a `testsuites` root. It still
requires exactly `(tests, failures, errors, skipped) = (12,0,0,0)`.

No model source, test source/node, expected count, resource, data scope, output
schema, or scientific boundary changed. Static validation parsed the raw Job
335565 JUnit as 12/0/0/0 and separately passed synthetic `testsuite`-root and nested
`testsuites`-root fixtures. `bash -n` and `git diff --check` passed.

### New immutable identity

- Negative-evidence documentation commit:
  `b848a6f` (`docs(s02): preserve failed CPU gate evidence`).
- Parser-only remediation commit / exact new executable HEAD:
  `840e8bee8d1157c71b7752d3937c6cb8e75201e7`.
- Model/test implementation remains:
  `65c83c077210469861ba722a285ab1e58e6d719f`.
- New exact 16-file runtime source-state SHA-256:
  `2ff7d74246e55332305e92a83dc028a42ce3c1e60993c28c24ece868784e580a`.
- Runtime source list is unchanged from the first request; only the launcher blob
  differs.
- Working tracked diff was empty at the executable commit before editing this
  request. This request/handoff prose is not an executable input.
- The finalized request's SHA-256 is intentionally reported to S00 externally
  after this file is complete; it cannot self-embed its own hash.

Any change to executable SHA, source state/list, model/tests/nodes, expected count,
command, resources, output, or stop conditions invalidates a future approval.

### Unchanged scope and resources

- Synthetic CPU tensor tests only; no mini/trainval/cache/checkpoint/model step.
- Exact same twelve pytest nodes as the first request.
- `CUDA_VISIBLE_DEVICES=""` before Python/Torch imports.
- One node, one `nvidia_gh200_120gb` allocation, four CPUs, `00:10:00`, maximum
  `0.167` GPU-hour; one job, no array/DDP/retry/requeue/resubmit/follow-on.
- New unique output root, confirmed absent during preparation:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_840e8bee8d11`.
- Logs remain under the launcher-declared unique Slurm `%j` paths:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_cpu_tests_%j.{out,err}`.

### Exact pending command — DO NOT SUBMIT WITHOUT NEW S00 APPROVAL

```bash
test "$(git rev-parse HEAD)" = "840e8bee8d1157c71b7752d3937c6cb8e75201e7" && \
test "$(git branch --show-current)" = "codex/s02-cl-p0-correctness" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_840e8bee8d11 && \
test -z "$(squeue -u "$USER" -h -o '%i %j' | awk '$2 ~ /flv3_s02/ {print}')" && \
sbatch --export=ALL,EXPECTED_S02_SHA=840e8bee8d1157c71b7752d3937c6cb8e75201e7,EXPECTED_S02_STATE_HASH=2ff7d74246e55332305e92a83dc028a42ce3c1e60993c28c24ece868784e580a,S02_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_840e8bee8d11 \
  fl_v3/usenix27_orchestra/handoffs/S02/run_s02_cpu_tests.sh
```

### Acceptance and stop conditions

Acceptance is unchanged: exact in-job identity/source match; execution identity
with empty `CUDA_VISIBLE_DEVICES`; JUnit exactly 12/0/0/0; pytest 12 passed; in-job
source and final artifact checksum verification; Slurm `COMPLETED 0:0` within the
approved resources. Any failure remains a negative result and stops the session.
There was no standing permission to submit this command; the one-time approval and
terminal execution are recorded below.

### Exact second S00 approval and terminal execution

S00 approved exactly one manual remediation submission, bound to:

- finalized request SHA-256:
  `48aa43079bcca7bbc9f9005862149d99968a76b149c6f3f7482f37bd8e125a0b`;
- executable `840e8bee8d1157c71b7752d3937c6cb8e75201e7`;
- unchanged implementation `65c83c077210469861ba722a285ab1e58e6d719f`;
- launcher SHA-256
  `35798c3956e1cb4fcf54288f34ead04687d2b22d2dcdff4186b865d5261b452b`;
- runtime source state
  `2ff7d74246e55332305e92a83dc028a42ce3c1e60993c28c24ece868784e580a`;
- the exact new output, command, tests, resources, and stop conditions above.

The second preflight matched exact HEAD/branch/request/launcher/source hashes;
all runtime sources were clean relative to HEAD; the new output was absent; and no
S02 job was active. S02 submitted exactly once as Job `335578`.

Job `335578` completed `COMPLETED 0:0` on node `n534` in `00:01:33`, with
`Restarts=0`. Pytest reported `12 passed in 19.86s`; JUnit aggregated to
12 tests, zero failures/errors/skips; execution identity recorded empty
`CUDA_VISIBLE_DEVICES`; and the in-job final `sha256sum -c` verified all four
declared artifacts. Complete scheduler fields, raw hashes, and both-job history are
in `RESULTS.md`.

The new approval is consumed. Job `335565` remains overall `FAILED 1:0` with its
missing checksum manifest; Job `335578` does not rewrite that history. No retry,
requeue, resubmission, or follow-on occurred, and no further submission is
authorized by this final record.
