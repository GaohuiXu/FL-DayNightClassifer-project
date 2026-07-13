# S07-B-COMPLETE RESULTS — closed bounded clean-engineering gate

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

## Job 380806 interpretation limits (historical boundary)

Allowed at that point: the clean-only config/test changes passed local static
gates; the persistent environment activated and loaded Torch/CUDA/spconv on GH200;
the clean FedAvg/profile test passed; C/L/F models all constructed and completed
one real-mini forward, finite loss and backward on the exact GH200 environment.

Not established by Job `380806`: any successful C/L/F optimizer update, completed
worker=2 equality check, integrated S06/S01/official-eval suite, detector
capability, mAP/NDS, fusion gain, Protocol A/B readiness, performance,
reproducibility, attack, defense, or scientific result.

At that historical boundary, review and retry were forbidden because Job `380806`'s
exact compute approval was consumed. The later owner-approved D1/F1 path and final
independent acceptance are recorded below; Job `380806` itself was never retried.

## D1 terminal gradient classification

Exact Job `389356` consumed the one-shot D1 approval and completed on `n101` with
exit `0:0`, elapsed `00:04:05`, zero restarts, `diagnostic.exit=0`, one final
`S07B_GRAD_DIAGNOSTIC_PASS` marker and JUnit `9 passed / 0 failed / 0 error /
0 skipped` in 160.169 seconds. The environment was aarch64 GH200, Python 3.11.15,
Torch 2.11.0+cu128/CUDA 12.8, cumm 0.7.13 and spconv 2.3.8. The nine tests passing
means every strict-JSON record and finite scalar loss was emitted; it does not
mean every gradient/update passed.

| Mode | Precision/scale | loss | gradient result | optimizer/scaler result |
|---|---:|---:|---|---|
| C-STR8 | FP32 | 945.870 | all 30,598,278 elements finite; FP64 norm 58,689.582; max 1,983.724 | optimizer called |
| L-S075 | FP32 | 1,557.707 | all elements finite; FP64 norm 8,326,751.029; max 1,910,373.875 | optimizer called |
| F-U | FP32 | 1,121.036 | all elements finite; FP64 norm 5,773,409.139; max 1,217,219.375 | optimizer called |
| C-STR8 | FP16 / 512 | 953.253 | 24 parameters / 35,900 elements nonfinite | skipped; scale 512 to 256 |
| L-S075 | FP16 / 512 | 1,552.345 | 52 parameters / 90,280 elements nonfinite | skipped; scale 512 to 256 |
| F-U | FP16 / 512 | 1,128.472 | 41 parameters / 77,180 elements nonfinite | skipped; scale 512 to 256 |
| C-STR8 | FP16 / 1 | 953.262 | all elements finite; FP64 norm 59,455.056; max 2,004 | optimizer called; scale stays 1 |
| L-S075 | FP16 / 1 | 1,551.731 | 5 parameters / 4,740 elements nonfinite; finite max 64,640 | skipped; scale 1 to 0.5 |
| F-U | FP16 / 1 | 1,133.587 | 4 parameters / 1,870 elements nonfinite; finite max 63,296 | skipped; scale 1 to 0.5 |

This closes the ambiguity from Job `380806`. The old norm was not merely an
FP32 norm-reduction overflow: D1 directly found nonfinite gradient elements.
All three FP32 controls are finite and update, so this is not an environment,
data-loader, scalar-loss or general model-backward failure. C-STR8 is classified
as an initial loss-scale problem for this exact batch because scale 1 fully
recovers it. L-S075 and F-U still fail at scale 1; their first bad parameters are
the sparse SECOND `lidar_encoder.backbone.stem` and `stage1` weights/norms, while
their FP32 control gradients reach roughly 1.9M/1.2M and the surviving FP16
elements approach the 65,504 range. This is evidence consistent with real FP16
backward dynamic-range overflow in the LiDAR path, not proof of a specific
kernel defect. A single global scale-1 change is therefore rejected as the full
C/L/F remediation.

Raw artifact SHA-256:

```text
diagnostic.log       6921efe9e39d25d7dc5fa6dfcab87a748d5db6040a4a49ab5a1fb3d5849edc16
diagnostic.junit.xml 71d1b73455aa7fb0a4f877562930d4f7df6618eb816a4a325bd39e2e7b02530a
diagnostic.exit      9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
environment.txt      b2e9d2df67472872f03dea6223d4dbef93bea29f60a5273aac334645fb487858
slurm-389356.out     0452fc470b88f012db709d5366240f8fb49b974417ea86b6473d3c1625218fdc
slurm-389356.err     ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57
```

D1 is bounded engineering diagnosis only. It does not establish multi-step
stability, convergence, accuracy, Protocol A/B readiness or scientific evidence.
Its compute authority is consumed. No retry or remediation compute is authorized.

## FP32 final-gate terminal result

Exact Job `390576` consumed the one-shot F1 approval and completed on `n105` with
exit `0:0`, elapsed `00:04:24`, zero restarts, `final_gate.exit=0`, one final
`S07B_FP32_FINAL_GATE_PASS` marker and JUnit `5 passed / 0 failed / 0 error /
0 skipped` in 173.217 seconds. The aarch64 GH200 environment remained Python
3.11.15, Torch 2.11.0+cu128/CUDA 12.8, cumm 0.7.13 and spconv 2.3.8.

| Case | loss | finite gradient norm | optimizer/exposure | precision/scaler | result |
|---|---:|---:|---|---|---|
| plain FedAvg construction | — | — | clean identity FedAvg | — | PASS |
| C-STR8 | 945.8136 | 58,677.4648 | 1 / 1 | fp32 / disabled | PASS |
| L-S075 | 1,562.8792 | 9,707,248.0 | 1 / 1 | fp32 / disabled | PASS |
| F-U | 1,122.4043 | 5,889,270.0 | 1 / 1 | fp32 / disabled | PASS |
| fusion worker 0 vs 2 first batch | — | — | exact batch equality | — | PASS |

All three mode records report `nonfinite_loss_steps=0`, `grad_scaler_skips=0`,
`optimizer_steps=1`, `exposure_samples=1`, and precision `fp32`. The expected
ccimport deprecation and spconv indexing warnings remained non-fatal. No D1,
AMP/scaler cell, comparison, profile, metric, extra update or retry ran.

Raw artifact SHA-256:

```text
final_gate.log       2c3cf8fc49c662aabae161b691b81d08fd20d131aa942e49ee0755ecd84e0cf9
final_gate.junit.xml a5eb50f35ffe031854bcb9b6862e5689ab02d779c17604c984b95a87dc2a69c5
final_gate.exit      9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
environment.txt      b2e9d2df67472872f03dea6223d4dbef93bea29f60a5273aac334645fb487858
slurm-390576.out     77d8d1d1221fa1cdd3919807d46d2cfa56696c5f8b73fdf535cbc4509978dc8f
slurm-390576.err     ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57
```

This is the requested bounded S07-B-COMPLETE final engineering gate PASS. It
does not establish multi-step stability, convergence, performance, full-data
readiness, mAP/NDS, Protocol A/B readiness or any scientific claim. Independent
S07-B-COMPLETE-R subsequently reviewed candidate `c615b647`, test commit
`29ca6637`, the immutable F1/D1 snapshots and raw artifacts, and returned **PASS
at the exact bounded clean-engineering scope** with no P0/P1/P2/P3 finding.
Review SHA-256 is
`b0feed5476dbc810b24a5dc3c7a678bc90ac3a2520360f02fdb6a6bf54691ebd`;
the terminal/review package is
`7f3bd40158e5a8af30196509734782c4575c50aa`. The owner accepted that verdict
and formally closed S07-B-COMPLETE. All compute authority is consumed.
