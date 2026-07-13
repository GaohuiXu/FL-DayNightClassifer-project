# S07-B-COMPLETE RESULTS — first real training boundary reached

## Current result

Integrated GH200 completion is **not established**. After retiring the audit
wrapper, exact Job `380806` reached the real model forward/loss/backward path for
all three C/L/F modes. Environment identity and clean FedAvg passed, but all three
first fp16 backward attempts produced nonfinite unscaled gradient norms, so
the required one-step evidence was not emitted or accepted. This is a current
numerical/training gate failure, not another environment or wrapper failure.

## Closed job record

| Job | State | Positive boundary | Failure and interpretation |
|---|---|---|---|
| `372819` | `FAILED 1:0`, 8 s, `n124`, restarts 0 | Executable/source identities completed. | Request used Git-reserved `GIT_COMMON_DIR`, corrupting dependency Git checks before environment activation. |
| `373363` | `FAILED 1:0`, 1:42, `n21`, restarts 0 | All 13 bootstrap gates and dependency pre/post comparison passed. | Request made a known `ccimport` deprecation fatal during spconv identity import; pytest never started. |
| `374142` | `CANCELLED`, 8:05, `n89`, restarts 0 | Environment and spconv identity passed; pytest collected exact 205 cases; clean-FedAvg/profile test passed. | Request forced a 113-byte `TMPDIR`; multiprocessing appended its listener suffix and exceeded the AF_UNIX path limit in the worker=2 loader test. No model update ran. |
| `380806` | `FAILED 1:0`, 4:28, `n192`, restarts 0 | Exact environment passed; clean-FedAvg profile passed; C/L/F each completed real-mini forward, finite loss and backward. | First unscaled gradient norms were `inf/nan/nan`; assertions stopped before step/skip metrics were checked or printed. Training JUnit `1 pass / 3 fail`; loader phase correctly NOT RUN. |

S00 stopped Job `374142` after the deterministic listener failure and more than
six minutes without progress rather than wait for the 50-minute global timeout.
Cancellation did not leave the shell finalizer time to emit JUnit or its final
manifest. No retry or replacement was submitted.

Raw roots remain:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_34cbe02b7b72
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_diag1_34cbe02b7b72
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_diag2_34cbe02b7b72
```

Key retained SHA-256 values:

```text
372819 artifact manifest: 6c186866f0ff8e3a18aa6a9873bbff3923cc18d18b2acb3f1851a12dff7b3260
373363 bootstrap gates: ea36579d7b4c524de82ee0af0f5673eee20b42fc4698fe493aa187c9faa7baae
373363 stderr: 5b6357146d90321484a9984b9a8500d3b7b2f35b6e9bbfa94549fe11c9b343b3
373363 artifact manifest: fe6bc6363f945ae803b0f005e7f4e3fbf21d81162023631e91b0c2e75a04048c
374142 pytest/stdout: 0507e5e254f357932da28bdd2b58e116e1acdba368daadc3b96e8e74af9e3487
374142 stderr: 1ae7aff202a2955595bbb274d5627f8b616bc98fc1550d310ed8397ca1ae7969
374142 execution identity: a4ff6321e5ca76225b4a7cf89d6290191a419e833969b62cd1abc01e6bd41904
```

## Job 380806 raw evidence

```text
root: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_clean_simple_34cbe02b7b72
environment.txt: b2e9d2df67472872f03dea6223d4dbef93bea29f60a5273aac334645fb487858
train.log: a6c9d686928fbf2cf2b658b0578f71ab9550539e52ee1bab6b70ee9fb14fe222
train.junit.xml: 8faeaf75ecc94a2eedceebaf0141b27f756d467a34b6d1c814d0d4d0544d1c9c
slurm stdout: a6c9d686928fbf2cf2b658b0578f71ab9550539e52ee1bab6b70ee9fb14fe222
slurm stderr: b534645ead7b5a3ed14c1d7fe032271613c2bf72d2c4e74776badddfbf1a888f
```

Environment identity was aarch64, Python `3.11.15`, Torch `2.11.0+cu128`, CUDA
`12.8`, cumm `0.7.13`, spconv `2.3.8`, one `NVIDIA GH200 120GB`, and `/tmp`.
`train.exit`, `train-tee.exit`, all loader artifacts, and the final marker are
absent. The train log and JUnit prove the three numerical assertions occurred
before the separate `PIPESTATUS` recording defect aborted the shell.

## Causal diagnosis

The earlier Arrhenius training evidence used the pre-S07 single-head/model path.
The six-task `MultiTaskCenterPointLoss` was integrated on 2026-07-12. S04 proved a
bounded synthetic sparse-encoder backward, while S06 explicitly recorded actual
S04+S05 fp16 production integration as NOT RUN. Job `380806` is therefore the
first current evidence for the complete real-mini six-task fp16 optimizer seam.

The loop upcasts head outputs before loss, obtains a finite scalar loss, scales it
with the default GradScaler initial scale `512`, backpropagates and unscales before
computing telemetry. The persisted norm is `inf/nan/nan`, but the test asserts on
that field before it checks or prints `optimizer_steps`, `grad_scaler_skips`, and
final scale. For L/F, `nan` strongly points to nonfinite gradients. For C, `inf`
could be a nonfinite element or overflow in the float32 global-norm reduction even
if individual elements are finite. The evidence therefore proves only that the
first-attempt finite-norm gate fails; it does **not** durably prove the step/skip
counters, that the model cannot train, or that a lower scale is the correct
production setting.

This continuation behavior is already an S06 contract:
`test_s06_training_runtime.py::test_scaler_overflow_invalidates_one_complete_fixed_window`
uses an overflow-once scaler, records the first window as invalid, continues the
loader, and completes exactly one later optimizer step. Production `train_local`
also keeps one scaler across the loader. The completion fixture instead wraps one
batch in a length-one iterable and requires zero skips, so it cannot exercise
that continuation if a skip occurred. This is a test-coverage mismatch, while
the actual per-mode step/skip counters and reason for the nonfinite norm remain
unresolved.

The shared failure across camera, LiDAR and fusion points first to the shared
multi-task head/loss or precision policy, not a branch-specific encoder. A future
remediation should first emit the already-returned step/skip/final-scale metrics
before asserting, distinguish nonfinite gradient elements from norm-reduction
overflow, and use the existing dynamic-scaler contract with a small fixed attempt
budget. It must stop after exactly one successful optimizer step and record
skipped/invalid windows. No production source/config or default-scale change is
justified by the present record. Only if bounded backoff still cannot step should
deeper head/loss diagnosis be added.

## Closed path diagnosis

The exact Job `374142` temporary root is 113 bytes. A stdlib-only reproduction
under that path fails with `OSError: AF_UNIX path too long`; `TMPDIR=/tmp`
succeeds with a randomized 36-byte `pymp-*/listener-*` address. This is local
inter-process communication, not network, CUDA, dataset, or model behavior.

## Interpretation limits

Allowed now: the clean-only config/test changes pass local static gates; the
persistent environment can activate and load Torch/CUDA/spconv on GH200; the clean
FedAvg/profile test passes; current C/L/F models all construct and complete one
real-mini forward, finite loss and backward on the exact GH200 environment.

Not established: any successful C/L/F optimizer update, completed worker=2 equality check,
integrated S06/S01/official-eval suite, detector capability, mAP/NDS, fusion gain,
Protocol A/B readiness, performance, reproducibility, attack, defense, or
scientific result.

Do not launch review or retry. The exact compute approval is consumed. A future
diagnostic/remediation command requires a new exact owner decision.

## D1 preparation status

The focused gradient-classification test and immutable snapshot are prepared but
**NOT RUN**. Local `python3 -m py_compile`, candidate JSON parsing, `bash -n` for
both exact temporary command files, embedded-command hash equality and
`git diff --check` pass. The login node has no `python` command and uses system
Python 3.9 only for syntax; no Arrhenius/Torch import was attempted there.

D1 has nine fixed C/L/F x precision/scale cells and no automatic remediation.
Exact request hashes and resource bounds are in `RUN_REQUEST.md`. Until the owner
approves that immutable request, there is no new Job ID or runtime result.
