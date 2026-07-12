# S06 RUN_REQUEST — remediation-2 bounded synthetic runtime validation

## Approval state and immutable negative history

- `APPROVAL_STATUS: PENDING_S00_EXACT_APPROVAL_DO_NOT_SUBMIT`.
- No remediation-2 `sbatch`/`srun` has been invoked. S06 will not self-submit.
- S00 must audit and explicitly approve this entirely new immutable tuple before
  any execution. Any change invalidates approval.
- Remediation-1 executable `6696984a6ebd4ec398d9fbfa172fb118e84e7af8`,
  delivery `5bbb12cd452fcf805e3687f5a7aa00d952393526`, request snapshot
  `e42fd06051fc8fa7ce1531fb8151d150c2395d2ea89aaf7a6249257f2aeddf08`
  and job `341997` are `FAILED_NEVER_RETRY_NEVER_EXECUTE`.
- Older executables `a95816b607d1ced5f07bd1136b23f36f58357a14` and
  `7d733e9b08454b059822015fcaf3eea53e8c2e56`, plus request
  `d2e302aba6cb0ed0561677f15c04601c373ebe10e9471787168bba05dcc65ef2`,
  remain `REJECTED_BY_S00_NEVER_EXECUTE`.
- The pre-341997 bare `sbatch` no-op remains recorded in `RESULTS.md`: Slurm
  rejected an empty script, created no job/root/allocation, and S00 audited it as
  a negative control-plane event.

## Immutable remediation-2 source identity

- Branch: `codex/s06-production-runtime`.
- Executable SHA: `c330c72f4060348768c63fb1b7855ca56baffb95`.
- Executable tree: `7ce589685d15fb42c057154c3329679ada934f4b`.
- Base SHA: `968d81583c87ba76b7dbbb722760f8eb8eb6cd39`.
- Base-to-executable binary diff SHA-256:
  `6f196001c8144806ff5b71c52b87154bdd7ecbe704b21bce2f1e770df3c09963`.
- Runtime source file count: `25`.
- Runtime source-manifest SHA-256:
  `bc19c139f773592dc085b47b3b83b1721f3c5ca0abeeeb1c6485e9e2d8f533dc`.
- Launcher SHA-256:
  `146f55797ec8191083f8347bcecae858785e3c64c08fc798079fa1ac53edde2d`.
- Launcher: `fl_v3/scripts/run_s06_runtime_tests.sh`.

The launcher archives this executable into a new read-only snapshot, recomputes
the 25-file aggregate before pytest, and requires request generation exactly
`remediation-2`.

## Exact unchanged test inventory

Complete files:

- `fl_v3/tests/test_s06_resolved_config.py`;
- `fl_v3/tests/test_s06_model_modes.py`;
- `fl_v3/tests/test_s06_training_runtime.py`;
- `fl_v3/tests/test_s06_checkpoint_resume.py`;
- `fl_v3/tests/test_s06_loader_eval.py`;
- `fl_v3/tests/test_profiling_neutral.py`.

Exact additional nodes:

- `fl_v3/tests/test_model_task.py::test_detection_config_rejects_legacy_model_mode_alias`;
- `fl_v3/tests/test_eval_detection_eval.py::test_submission_meta_uses_actual_mode`;
- `fl_v3/tests/test_eval_provenance.py::test_s06_provenance_binds_mode_config_checkpoint_and_data`.

The inventory path set is unchanged from job 341997. Within those files,
remediation-2 adds plain-container config assertions, a truly length-opaque tail,
three atomic checkpoint save/cleanup fixtures, strict six-camera negative eval
coverage, and retains all earlier hostile checkpoint/CUDA rollback cases.

The scope remains synthetic/config-only: no mini/trainval read, cache/manifest
creation, production S03/S04/S05 construction, optimizer campaign, metric,
production-shape profile, 100/1000-step, DDP, seed/matrix or scientific claim.

## Exact command

```bash
sbatch --export=ALL,EXPECTED_S06_EXECUTABLE_SHA=c330c72f4060348768c63fb1b7855ca56baffb95,EXPECTED_S06_SOURCE_SHA256=bc19c139f773592dc085b47b3b83b1721f3c5ca0abeeeb1c6485e9e2d8f533dc,S06_REQUEST_GENERATION=remediation-2,S06_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_remediation2_c330c72f4060 fl_v3/scripts/run_s06_runtime_tests.sh
```

## Resources and fresh roots

- one shared Arrhenius node;
- one `nvidia_gh200_120gb` allocation;
- eight CPUs, 16 GiB host memory;
- walltime `00:15:00` (maximum `0.25` allocation-equivalent GPU-hours);
- no array, DDP, multi-node, extra cell/seed, retry, requeue, automatic
  resubmission, or follow-on;
- output root, confirmed absent:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_remediation2_c330c72f4060`;
- snapshot root, confirmed absent:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s06_runtime_remediation2_c330c72f4060`;
- Slurm logs, if approved, use `s06_runtime_r2_<jobid>`.

## Artifact-preserving launcher behavior

- pytest cache provider is disabled for the read-only snapshot;
- pytest output is tee'd and its original exit code is captured;
- `execution_identity.json`, source file/hash lists, JUnit, pytest log and
  `pytest.exitcode` are included in `sha256sums.txt` even when tests fail;
- `sha256sum -c sha256sums.txt` runs before the launcher returns the original
  pytest exit code;
- a tee/checksum failure remains a launcher failure and cannot be hidden by a
  passing pytest status.

## Acceptance and stop conditions

Accept only if all are true:

1. scheduler state `COMPLETED`, exit `0:0`, no restart/requeue;
2. allocation is exactly one GH200/eight CPUs/16 GiB/one node;
3. executable/tree/source/launcher identities match before pytest;
4. every declared test passes with zero failures/errors/skips; the CUDA rollback
   fixture must execute, not skip;
5. installed spconv is exactly `2.3.8`;
6. JUnit, log, source attestation, exit-code record and final checksum manifest
   exist and verify in-job.

On any mismatch/failure, preserve negative evidence and stop. No automatic or
worker-initiated retry/resubmit/replacement/follow-on is permitted.

## Explicitly outside this request

Production-detector checkpoint memory measurement, full `t1.v2`, mini/trainval
decode, official metric, production-shape integration, sparse fp16 forward,
100/1000-step, profile/throughput, matrix, seed, rerun, DDP, upload, reviewer,
merge, push, or PR.
