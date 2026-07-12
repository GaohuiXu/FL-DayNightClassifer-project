# S06 RUN_REQUEST — remediation-1 bounded synthetic runtime validation

## Approval state and permanently rejected predecessors

- `APPROVAL_STATUS: EXECUTED_ONCE_FAILED_NO_RETRY`.
- S00 exact-compute approval was received on 2026-07-12 for the immutable
  pre-approval request snapshot SHA-256
  `e42fd06051fc8fa7ce1531fb8151d150c2395d2ea89aaf7a6249257f2aeddf08`.
- The approval binds delivery `5bbb12cd452fcf805e3687f5a7aa00d952393526`,
  executable `6696984a6ebd4ec398d9fbfa172fb118e84e7af8`, executable tree
  `c504a5ff70b9c31b058867fc25a70bbd8b597997`, source aggregate
  `7be6c0c58b42dbef005ccf0ed52f152c06179701c3205bb607a0007ffa098aae`
  and launcher SHA-256
  `2e261bfddf7cd406934bd7f5b9ead76571be734f2adc25ec2b01458bc92ba120`.
- Exactly one submission of the command below is approved. Retry, requeue,
  resubmit, any additional job or any tuple change is forbidden.
- While auditing this approval record, an accidental shell interpolation invoked
  bare `sbatch` with no script. Slurm rejected it immediately with
  `Batch script is empty!`; it created no Job ID/job/resource allocation and did
  not touch either root. This negative control-plane event was not the approved
  command. The exact approved command below has not yet been submitted.
- The exact command was subsequently submitted once as job `341997`. It ended
  `FAILED 1:0` after `00:01:47` with `45 passed, 17 failed, 0 skipped`.
  The stop condition is active: no retry/resubmit/replacement/follow-on.
- S00 explicitly set `REJECTED_BY_S00_NEVER_EXECUTE` for both predecessor
  executables `a95816b607d1ced5f07bd1136b23f36f58357a14` and
  `7d733e9b08454b059822015fcaf3eea53e8c2e56`.
- The prior RUN_REQUEST with SHA-256
  `d2e302aba6cb0ed0561677f15c04601c373ebe10e9471787168bba05dcc65ef2`
  is `REJECTED_BY_S00_NEVER_EXECUTE`, including its old output/snapshot roots.
- This replacement approval applies only to the exact immutable tuple below.
- Any change to executable/tree, command, tests, source aggregate, resources,
  roots, or stop conditions invalidates a future approval.

## Immutable source identity

- Branch: `codex/s06-production-runtime`.
- Executable SHA: `6696984a6ebd4ec398d9fbfa172fb118e84e7af8`.
- Executable tree: `c504a5ff70b9c31b058867fc25a70bbd8b597997`.
- Base SHA: `968d81583c87ba76b7dbbb722760f8eb8eb6cd39`.
- Base-to-executable binary diff SHA-256:
  `f87ed29ab089e6ab4e1b365c693bf32d47930515eb33c2de7454cd9607d5847d`.
- Runtime source file count: `25`.
- Runtime source-manifest SHA-256:
  `7be6c0c58b42dbef005ccf0ed52f152c06179701c3205bb607a0007ffa098aae`.
- Launcher SHA-256:
  `2e261bfddf7cd406934bd7f5b9ead76571be734f2adc25ec2b01458bc92ba120`.
- Launcher: `fl_v3/scripts/run_s06_runtime_tests.sh`.

The launcher exports this exact Git object into a new read-only snapshot,
recomputes the same locale-stable 25-file aggregate, and fails before pytest if
the request generation or aggregate differs.

## Exact test inventory

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

The two remediation files now include hostile coverage for nonfinite loss at the
first/middle/final microbatch of a fixed window, GradScaler overflow, known and
runtime-only remainder, non-boundary `max_steps`, successful-update budget stop,
short microbatch/effective-batch drift, corrupt model/optimizer/training/RNG and
late scheduler/scaler/EMA loads. They also assert detached unaliased CPU rollback
snapshots and, on the requested GH200, exact rollback of live CUDA
model/optimizer/scaler/EMA objects from CPU snapshots.

This remains synthetic/config-only. It does not open mini or shared trainval,
create a cache/manifest, construct production S03/S04/S05 modules, run a training
campaign, compute a metric, or profile production shapes. Tiny fixture optimizer
operations are bounded inside the declared tests; no 100/1000-step gate exists.

## Exact command

```bash
sbatch --export=ALL,EXPECTED_S06_EXECUTABLE_SHA=6696984a6ebd4ec398d9fbfa172fb118e84e7af8,EXPECTED_S06_SOURCE_SHA256=7be6c0c58b42dbef005ccf0ed52f152c06179701c3205bb607a0007ffa098aae,S06_REQUEST_GENERATION=remediation-1,S06_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_remediation1_6696984a6ebd fl_v3/scripts/run_s06_runtime_tests.sh
```

## Resources and fresh roots

- one shared Arrhenius node;
- one `nvidia_gh200_120gb` allocation;
- eight CPUs, 16 GiB host memory;
- walltime `00:15:00` (maximum `0.25` allocation-equivalent GPU-hours);
- no array, DDP, multi-node, extra cell/seed, retry, requeue, automatic
  resubmission, or follow-on;
- output root (confirmed absent while preparing this request):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_remediation1_6696984a6ebd`;
- snapshot root (confirmed absent while preparing this request):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s06_runtime_remediation1_6696984a6ebd`;
- Slurm logs use the new `s06_runtime_r1_<jobid>` names.

## Acceptance and stop conditions

Accept only if all are true:

1. scheduler state `COMPLETED`, exit `0:0`, no restart/requeue;
2. actual allocation and CUDA visibility are one GH200/eight CPUs/one node;
3. executable/tree/source/launcher identities match before pytest;
4. every collected declared test passes with zero failures/errors/skips (the CUDA
   rollback fixture must execute, not skip);
5. installed `spconv==2.3.8` check passes;
6. JUnit, execution identity, source lists, pytest log and final checksum manifest
   exist and verify in-job.

On any mismatch/failure, preserve it as negative evidence and stop. Do not modify,
retry, resubmit, broaden the inventory, touch data, or launch a follow-on.

## Explicitly outside this request

Production-detector checkpoint memory measurement, full `t1.v2` materialization,
any mini/trainval decode, official eval/metric, production-shape S03/S04/S05
integration, fp16 sparse forward, 100/1000-step run, profile/throughput/memory,
matrix, seed, rerun, DDP, upload, merge, or push.
