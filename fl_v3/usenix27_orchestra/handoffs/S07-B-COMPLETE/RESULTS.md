# S07-B-COMPLETE RESULTS — closed wrapper failures and pending simple validation

## Current result

The engineering candidate remains locally/static ready, but integrated GH200
completion is **not established**. Three bounded submissions were consumed before
any C/L/F optimizer update. Their shared root cause was the retired execution
wrapper, not a demonstrated Arrhenius, model, data, or clean-FL defect.

The owner directed S00 to remove that wrapper. A read-only snapshot of executable
`34cbe02b7b72...` now exists for a short training-first request. On 2026-07-13 the
owner approved its exact command and one submission; it is not yet submitted at
this document state.

## Closed job record

| Job | State | Positive boundary | Failure and interpretation |
|---|---|---|---|
| `372819` | `FAILED 1:0`, 8 s, `n124`, restarts 0 | Executable/source identities completed. | Request used Git-reserved `GIT_COMMON_DIR`, corrupting dependency Git checks before environment activation. |
| `373363` | `FAILED 1:0`, 1:42, `n21`, restarts 0 | All 13 bootstrap gates and dependency pre/post comparison passed. | Request made a known `ccimport` deprecation fatal during spconv identity import; pytest never started. |
| `374142` | `CANCELLED`, 8:05, `n89`, restarts 0 | Environment and spconv identity passed; pytest collected exact 205 cases; clean-FedAvg/profile test passed. | Request forced a 113-byte `TMPDIR`; multiprocessing appended its listener suffix and exceeded the AF_UNIX path limit in the worker=2 loader test. No model update ran. |

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

## Path diagnosis

The exact Job `374142` temporary root is 113 bytes. A stdlib-only reproduction
under that path fails with `OSError: AF_UNIX path too long`; `TMPDIR=/tmp`
succeeds with a randomized 36-byte `pymp-*/listener-*` address. This is local
inter-process communication, not network, CUDA, dataset, or model behavior.

## Replacement contract

The pending request runs only:

- clean Flower/FedAvg profile plus one C/L/F fp16 optimizer update, all with
  `num_workers=0`;
- one separate five-minute workers-0-versus-2 first-batch equality check using
  `TMPDIR=/tmp`.

It runs from a 628 KiB read-only snapshot, activates the persistent Arrhenius
environment, and contains no Git, source archive, dependency cleanliness,
warnings-as-errors, custom cache isolation, or 205-case suite. The exact draft and
acceptance rule are in `RUN_REQUEST.md`.

## Interpretation limits

Allowed now: the clean-only config/test changes pass local static gates; the
persistent environment can activate and load Torch/CUDA/spconv on GH200; the clean
FedAvg/profile test passes in Job `374142`.

Not established: any C/L/F optimizer update, completed worker=2 equality check,
integrated S06/S01/official-eval suite, detector capability, mAP/NDS, fusion gain,
Protocol A/B readiness, performance, reproducibility, attack, defense, or
scientific result.

Do not launch review. The owner has approved exactly one submission of the final
simplified command/hash/envelope; no changed command or retry is authorized.
