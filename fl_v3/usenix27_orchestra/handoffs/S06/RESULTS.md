# S06 remediation-1 RESULTS — negative bounded validation

## Verdict

- `RESULT_STATUS: FAILED_STOP_NO_RETRY`.
- Job `341997` is the one and only exact approved submission.
- Slurm: `FAILED`, exit `1:0`, no restart/requeue, elapsed `00:01:47`.
- Pytest: `62` collected, `45` passed, `17` failed, `0` errors, `0` skipped.
- The acceptance gate failed. No retry, resubmit, replacement job or follow-on
  compute was launched or is authorized.

The earlier bare `sbatch` interpolation event was rejected with
`Batch script is empty!` and produced no Job ID. S00 audited it as a no-op that
did not consume the approved job. It remains recorded as negative control-plane
evidence in `RUN_REQUEST.md` and `HANDOFF.md`.

## Immutable execution identity

- approved delivery: `5bbb12cd452fcf805e3687f5a7aa00d952393526`;
- approval-record commit before submission:
  `152833495874c347a89736e36112ecd5e81e3422`;
- executable: `6696984a6ebd4ec398d9fbfa172fb118e84e7af8`;
- executable tree: `c504a5ff70b9c31b058867fc25a70bbd8b597997`;
- pre-approval RUN_REQUEST SHA-256:
  `e42fd06051fc8fa7ce1531fb8151d150c2395d2ea89aaf7a6249257f2aeddf08`;
- source aggregate:
  `7be6c0c58b42dbef005ccf0ed52f152c06179701c3205bb607a0007ffa098aae`;
- launcher SHA-256:
  `2e261bfddf7cd406934bd7f5b9ead76571be734f2adc25ec2b01458bc92ba120`;
- request generation: `remediation-1`;
- synthetic-only: `true`.

The in-job `execution_identity.json` independently records job `341997`, host
`n405`, `aarch64`, Python `3.11.15`, Torch `2.11.0+cu128`, spconv `2.3.8`, and
`CUDA_VISIBLE_DEVICES=0`.

## Scheduler and resource evidence

```text
Submit  2026-07-12T05:29:09+02:00
Start   2026-07-12T05:29:10+02:00
End     2026-07-12T05:30:57+02:00
State   FAILED
Exit    1:0
Elapsed 00:01:47
Node    n405
Alloc   1 node, 8 CPU, 16 GiB, 1×nvidia_gh200_120gb
Batch MaxRSS 36M
Requeue 0
Restarts 0
```

The resource allocation matched the approved bound. `squeue` is empty after
termination.

## Pytest/JUnit result

JUnit suite attributes:

```text
tests=62 failures=17 errors=0 skipped=0 time=23.338 hostname=n405
```

Failure groups:

1. **Resolved config bridge — 1 failure.** `to_run_config()` passes frozen nested
   `mappingproxy` cache identities to `json.dumps`, producing
   `TypeError: Object of type mappingproxy is not JSON serializable`.
2. **Unknown-length tail fixture — 1 failure.** PyTorch's
   `_SingleProcessDataLoaderIter` exposes `len()==5`; the implementation therefore
   correctly takes its known-length preflight and raises `ValueError` before the
   runtime-tail path expected by the test. The authored fixture did not create an
   actually length-opaque iterator.
3. **Checkpoint suite — 14 failures.** Every checkpoint-writing test, including
   continuous/resume, eight corrupt-state cases, three real-late-component cases,
   and the CUDA rollback case, stops in `torch.save`. Torch 2.11's zip writer
   rejects the hidden temporary basename `.s06-ckpt-*` as an invalid filename.
   Consequently none of the intended load/rollback assertions executed; the CUDA
   fixture was collected and did not skip, but it failed before rollback proof.
4. **Eval single-pass/timing fixture — 1 failure.** The fixture supplies one
   `lidar2img` matrix while `decode_eval_set()` reshapes the stored calibration to
   `(6,4,4)`, producing a size-16 reshape failure before the neutrality assertions.

The two pytest-cache permission warnings come from the deliberately read-only
execution snapshot and do not account for the failures.

## Raw artifacts and checksums

Output root:

`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_remediation1_6696984a6ebd`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 370 | `e7846f7726af7ee2c46e3e5d1785e8d57376a1217fad1d3a83a07904eafbb249` |
| `pytest.junit.xml` | 66319 | `168d84d0c5f7f1eb78834ef617391f115ad1cfb127982794edb6b30464213241` |
| `pytest.log` | 58911 | `1ff31c03406921870c714e07969853d12f4dd63be117093fedb34ee0e50c2ed8` |
| `runtime_source_files.txt` | 893 | `9afd0ce020f63ff215dd9eca7f5f70bbe16e53dac5aa72b89787905d6a1c010e` |
| `runtime_source_sha256s.txt` | 2543 | `7be6c0c58b42dbef005ccf0ed52f152c06179701c3205bb607a0007ffa098aae` |

Slurm logs:

| Artifact | SHA-256 |
|---|---|
| `logs/s06_runtime_r1_341997.out` | `1ff31c03406921870c714e07969853d12f4dd63be117093fedb34ee0e50c2ed8` |
| `logs/s06_runtime_r1_341997.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

The launcher verified the 25-file aggregate before pytest, and a post-job
`sha256sum -c runtime_source_sha256s.txt` passed for all 25 files. Because
`set -euo pipefail` stopped at the failing pytest pipeline, the final in-job
`sha256s.txt` was **not produced**. The per-artifact checksums above were computed
post-job and are not mislabeled as the missing in-job final manifest.

## Interpretation boundary

Allowed: this exact bounded run is durable negative engineering evidence; source
identity, environment, allocation, zero-skip collection and the listed 45 passing
tests are recorded.

Forbidden: S06 runtime PASS, checkpoint/resume atomicity proof, CUDA rollback
proof, eval timing-neutrality proof, integration/model/full-data/scientific claims,
or treating any passing subset as acceptance. No fix or rerun is part of this
delivery.

## Post-result remediation status

S00 later authorized code-only remediation within the original S06 ownership.
Executable `c330c72f4060348768c63fb1b7855ca56baffb95` addresses the four root-cause
families and artifact-preservation defect, but has only static/stdlib evidence.
It does not alter this RESULTS verdict or retroactively pass any job-341997 gate.
Its separate remediation-2 request is pending S00 audit; no new job exists.
