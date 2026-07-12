# S06 RUN_REQUEST — bounded synthetic runtime validation

## Approval state

- `APPROVAL_STATUS: PENDING_S00_EXACT_APPROVAL_DO_NOT_SUBMIT`
- No `sbatch`/`srun` has been invoked by S06.
- This request does not rely on standing self-submission. S00 must audit and
  explicitly approve this exact immutable tuple before execution.
- Any change to executable SHA/tree, command, tests, source aggregate, resources,
  output root, or stop conditions invalidates approval.

## Immutable source identity

- Branch: `codex/s06-production-runtime`.
- Executable SHA: `7d733e9b08454b059822015fcaf3eea53e8c2e56`.
- Base SHA: `968d81583c87ba76b7dbbb722760f8eb8eb6cd39`.
- Base-to-executable binary diff SHA-256:
  `0ed3ae6d7ed2721a214e75304e59bfde4397fa781197ac037280701d2e2170ab`.
- Runtime source file count: `25`.
- Runtime source-manifest SHA-256:
  `d81f5b08846759f385a5288da80d7265184ffc23a3d5339213348977114f3189`.
- Launcher SHA-256:
  `aeba5912ab8ad250720ccedf92b3e84378b0c67e03234424b4f457b3a00eb3a2`.
- Launcher: `fl_v3/scripts/run_s06_runtime_tests.sh`.

The launcher exports this exact Git object into a new read-only `/nobackup`
snapshot, computes the same locale-stable 25-file source manifest inside the job,
and fails before pytest if the aggregate differs.

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

The scope is synthetic/config-only. It does not open mini or shared trainval,
create a cache/manifest, construct the reviewed S03/S04/S05 production modules,
perform an optimizer campaign, compute a metric, or profile production shapes.
The only optimizer operations are bounded tiny CPU fixtures inside the declared
unit tests. No 100/1000-step gate is included.

## Exact command

```bash
sbatch --export=ALL,EXPECTED_S06_EXECUTABLE_SHA=7d733e9b08454b059822015fcaf3eea53e8c2e56,EXPECTED_S06_SOURCE_SHA256=d81f5b08846759f385a5288da80d7265184ffc23a3d5339213348977114f3189,S06_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_7d733e9b0845 fl_v3/scripts/run_s06_runtime_tests.sh
```

## Resources and roots

- one shared Arrhenius node;
- one `nvidia_gh200_120gb` allocation;
- eight CPUs, 16 GiB memory;
- walltime `00:15:00` (maximum `0.25` allocation-equivalent GPU-hours);
- no array, DDP, multi-node, additional cell/seed, retry, requeue, automatic
  resubmission, or follow-on;
- output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_7d733e9b0845`;
- snapshot root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s06_runtime_7d733e9b0845`;
- Slurm logs use the S06 job ID under the existing Arrhenius log root.

## Acceptance and stop conditions

Accept only if all are true:

1. scheduler state `COMPLETED`, exit `0:0`, no restart/requeue;
2. actual allocation and CUDA visibility are one GH200/eight CPUs/one node;
3. executable/source identities match before pytest;
4. every collected declared test passes with zero failures/errors/skips;
5. exact installed `spconv==2.3.8` check passes;
6. JUnit, execution identity, source lists, pytest log and final checksum manifest
   exist and verify in-job.

On any mismatch/failure, preserve it as negative evidence and stop. Do not modify,
retry, resubmit, broaden the test inventory, touch data, or launch a follow-on.

## Explicitly outside this request

Full `t1.v2` materialization, any mini/trainval decode, official eval/metric,
production-shape S03/S04/S05 integration, fp16 sparse forward, 100/1000-step run,
profile/throughput/memory, matrix, seed, rerun, DDP, upload, merge, or push.
