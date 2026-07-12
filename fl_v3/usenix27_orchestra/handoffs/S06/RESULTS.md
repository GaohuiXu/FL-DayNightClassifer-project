# S06 RESULTS — remediation-1 negative and remediation-2 bounded PASS

## Current bounded-gate verdict

- `RESULT_STATUS: REMEDIATION2_BOUNDED_SYNTHETIC_PASS_PENDING_S00_AUDIT`.
- Job `342014` is the one and only remediation-2 exact approved submission.
- Slurm: `COMPLETED`, exit `0:0`, no restart/requeue, elapsed `00:00:16`.
- Pytest/JUnit: `66` tests, `66` passed, `0` failures/errors/skips.
- The CUDA live-object rollback fixture executed and did not skip.
- `pytest.exitcode` is `0`; the final in-job `sha256sums.txt` exists and
  `sha256sum -c` verifies all six bound artifacts.
- No retry, resubmit, second job, data access, reviewer or follow-on occurred.

This is a bounded synthetic engineering PASS only. It is not an independent
review, S07-B integration, production detector, full-data or scientific PASS.
Job `341997` remains permanent negative evidence and is not overwritten.

## Remediation-2 immutable execution identity

- request delivery: `cae0ff59ce3e215ba950be6a76167d2dd716c940`;
- approval-record commit: `57b745a275636aa2c23c2f7aee0aa140c6195975`;
- executable: `c330c72f4060348768c63fb1b7855ca56baffb95`;
- executable tree: `7ce589685d15fb42c057154c3329679ada934f4b`;
- pre-approval RUN_REQUEST SHA-256:
  `9479538201ec398b1617847c5265d0dbeae8ec0db084fc6b867a435ffb5020a9`;
- base diff: `6f196001c8144806ff5b71c52b87154bdd7ecbe704b21bce2f1e770df3c09963`;
- source aggregate:
  `bc19c139f773592dc085b47b3b83b1721f3c5ca0abeeeb1c6485e9e2d8f533dc`;
- launcher SHA-256:
  `146f55797ec8191083f8347bcecae858785e3c64c08fc798079fa1ac53edde2d`;
- request generation: `remediation-2`; synthetic-only: `true`.

In-job identity records job `342014`, host `n405`, `aarch64`, Python `3.11.15`,
Torch `2.11.0+cu128`, spconv `2.3.8`, and `CUDA_VISIBLE_DEVICES=0`.

## Remediation-2 scheduler/JUnit evidence

```text
Submit  2026-07-12T05:39:37+02:00
Start   2026-07-12T05:39:38+02:00
End     2026-07-12T05:39:54+02:00
State   COMPLETED
Exit    0:0
Elapsed 00:00:16
Node    n405
Alloc   1 node, 8 CPU, 16 GiB, 1×nvidia_gh200_120gb
Batch MaxRSS 36M
Requeue/restarts 0/0
JUnit  tests=66 failures=0 errors=0 skipped=0 time=2.930 host=n405
```

## Remediation-2 raw artifacts and checksums

Output root:

`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s06_runtime_remediation2_c330c72f4060`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 370 | `a50e90238efb73d56fc615995b0d911b9a91c619aa984c73c0240bee17c30109` |
| `pytest.exitcode` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `pytest.junit.xml` | 8793 | `76ecb32b065a23af69bdbfafde194881131b601246d9e497267ab4d779c930b4` |
| `pytest.log` | 99 | `4b4e01abc3f54015297a6c57ae37a4ddb966a7e6aceb5fd8d83df15d045bea42` |
| `runtime_source_files.txt` | 893 | `9afd0ce020f63ff215dd9eca7f5f70bbe16e53dac5aa72b89787905d6a1c010e` |
| `runtime_source_sha256s.txt` | 2543 | `bc19c139f773592dc085b47b3b83b1721f3c5ca0abeeeb1c6485e9e2d8f533dc` |
| `sha256sums.txt` | 1206 | `2429764a33ff574b5de2623da137c816500889c8afa20e421464edeab155997b` |

Slurm logs:

| Artifact | SHA-256 |
|---|---|
| `logs/s06_runtime_r2_342014.out` | `a83fd398d95d9ff8f3ecbb0df6705464135946a3b9c52a7a42e1c429b5e6a886` |
| `logs/s06_runtime_r2_342014.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

The in-job final manifest binds and verifies execution identity, both source
attestation files, pytest log, JUnit and pytest exit code. The 25-entry source
hash list itself hashes to the approved source aggregate.

## Remediation-1 terminal verdict

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

## Remediation-1 interpretation boundary

Allowed: this exact bounded run is durable negative engineering evidence; source
identity, environment, allocation, zero-skip collection and the listed 45 passing
tests are recorded.

Forbidden from job 341997: S06 runtime PASS, checkpoint/resume atomicity proof, CUDA rollback
proof, eval timing-neutrality proof, integration/model/full-data/scientific claims,
or treating any passing subset as acceptance. No fix or rerun is part of this
delivery.

## Post-result remediation status

S00 later authorized remediation-2 executable
`c330c72f4060348768c63fb1b7855ca56baffb95` and exactly one bounded job.
Job `342014` passed all 66 tests and artifact checks above. This provides the
current bounded synthetic PASS but does not alter or erase job `341997`'s
historical negative verdict. No further compute or reviewer is authorized.
