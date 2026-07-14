# S08 precision qualification — execution results

## Owner disposition

O-110 accepts reviewed close-ready seal
`d31adea049c84e47a0e4f82f38f22a2ca91a5a6f`, freezes the recommended precision
policy, and closes S08 PASS. This acceptance does not broaden the bounded Q1/Q2
interpretation into convergence, performance, capability, or scientific evidence.

## S08-Q2 compatibility gate — terminal PASS

```text
REQUEST_ID/JOB_ID: S08-Q2 / 435151
OWNER_AUTHORITY: O-109
STATE/EXIT/RESTARTS: COMPLETED / 0:0 / 0
SUBMIT/START/END: 2026-07-14T19:02:10 / 19:02:11 / 19:06:07 +02:00
ELAPSED/NODE: 00:03:56 / n207
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
EXECUTION_SOURCE_SHA: 3bb10d39c60e6fd2d0bfe480bb03a7c8cfc76fe9
REQUEST_FREEZE_SHA: f0b811d42ff841f61e67b85a1d583e6edb2f2d49
SNAPSHOT_TREE_SHA256: 1d9191c2f6234199d31405f9690ffd2d83343889333efbe1e1ae47e6235a5c60
JOB_SCRIPT_SHA256: ff14fd735788a4fa4691a473eb788276d901371160c28f447fe8819f33494d0d
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q2_1d9191c2f623
NEW-Q1/Q2 GPU BUDGET USED/REMAINING: 00:07:58 / 01:52:02
```

The exact one-test runner completed the two declared compatibility cells in order
and emitted 13 strict window records. JUnit is 1 test/0 failure/0 error/0 skip;
the job-created checksum manifest verifies all ten declared runtime/Q2 artifacts.
No retry or additional cell was submitted.

| Cell | Template/route | Attempts | Accepted | First accepted scale | Result |
|---|---|---:|---:|---:|---|
| P1 | L-P020, global FP16, `not_applicable`, uniform | 7 | 1 | 8 | PASS |
| B1 | F-CBGS, global FP16 + SECOND FP32 island, CBGS identity | 6 | 1 | 16 | PASS |

For both cells, overflow windows left optimizer step, scheduler epoch, EMA state,
and exposure at zero. The accepted window alone advanced optimizer, scheduler,
and exposure exactly once; EMA remained correctly disabled. Loss, all parameter
gradients, and all applicable named boundary gradients were finite. P1 resolved
`not_applicable/null/null` for sparse partition/requested/active. B1 resolved
`fp32/false/false`; its accepted SECOND output/stage1/stem gradients were finite.
The exact five Q1 fixture identities were reproduced.

P1 backoff from 512 accepted at scale 8. B1 backoff from 512 accepted at scale 16.
The B1 resolved config binds `det-cbgs=true`, but this replay-frozen single batch
does not exercise or qualify the CBGS sampling distribution or loader throughput.

### Q2 preserved artifact identities

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `environment.json` | 228 | `dcb00e9f854c6ed57e47939a3c52fd3951e8b6d18965ade02435d91023702d4d` |
| `q2.log` | 2,773 | `766a5094df438885d5303d214c3652d9367de47d42a58aee7b852bd987ff288d` |
| `q2.junit.xml` | 369 | `bcd4242c707adad0cf35e24344475138759bf868fb24951f9e86a08376ac4a9d` |
| `q2.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `raw/fixture_manifest.json` | 5,980 | `61bbcb481109937c02d5010074b5a5de7d1b2ff445fd8ccc1df6a92956beb43c` |
| `raw/fixture_identity.json` | 549 | `6f44f71692a79a443b4fdce4abe528184e8eab7c61f018960815024ffab709b2` |
| `raw/resolved_configs.json` | 6,920 | `529a8ea44f595edf8ee86b19e1632643c315237987a20d5c6dddbd226de03925` |
| `raw/window_records.jsonl` | 148,440 | `47dfa3407204f36f0da002334304fdb1c0795dc54d0e729fafb7f64a595a3f81` |
| `raw/q2_partial_summary.json` | 3,247 | `d00978827b092d5f01a6aa781d74b1639935b3767c53284166fd82ad5eef785b` |
| `raw/q2_summary.json` | 865,147 | `211c2560ab207525e3ceeb66e0b73c3100b70473ce02cfed09ec21e91fb383e1` |
| `artifact_sha256s.txt` | 856 | `36b9cbf1eab30f54799cf7abbe83056ac009b301a7817d604a0c8b9abea5fb2f` |
| `slurm-435151.out` | 4,441 | `a947576566c3b95cc6dc19bf57543ee3b50330ed0a61f3b3857530d7b4947e2e` |
| `slurm-435151.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

Together, Q1 and Q2 support the close-ready precision-policy candidate: global
FP16 for camera and pillar routes, global FP16 with SECOND/spconv explicitly kept
FP32 for current sparse LiDAR/fusion routes, and uniform FP32 as reference/fallback.
Full sparse FP16 is not accepted as the unified F-capable route. This remains
bounded one-fixture numerical qualification independently reviewed in R3 with
`PASS_WITH_RESIDUAL_RISK` and accepted by the owner under O-110; it is not convergence, capability, performance,
sampling-quality, mAP/NDS, production-data, or scientific-result evidence.

## S08-Q1 primary precision qualification — terminal bounded result

```text
REQUEST_ID/JOB_ID: S08-Q1 / 431013
OWNER_AUTHORITY: O-109
STATE/EXIT/RESTARTS: COMPLETED / 0:0 / 0
SUBMIT/START/END: 2026-07-14T18:52:47 / 18:52:48 / 18:56:50 +02:00
ELAPSED/NODE: 00:04:02 / n451
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
SOURCE_SHA: e6e28bea43f7757347da2e460cdf24e9a32b791f
SNAPSHOT_TREE_SHA256: dbeee35dcd6d7bcb919f549f03c42763d5d82b2b20740815743b7aa2b3f9bc9c
JOB_SCRIPT_SHA256: 42cb555d518a6d7bb517c325c22c1f0ab8362c03da36b9cfd1f0b981d8b349e1
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q1_dbeee35dcd6d
NEW-Q1/Q2 GPU BUDGET USED/REMAINING: 00:04:02 / 01:55:58
```

The exact one-test runner completed all eight ordered cells and emitted 66 strict
JSON window records. JUnit is 1 test/0 failure/0 error/0 skip; the job-produced
checksum manifest verifies all ten declared runtime/Q1 artifacts. Pytest success
means runner completeness, not that every numerical regime passed: F2 correctly
remains a bounded negative cell.

| Cell | Regime | Attempts | Accepted | First accepted scale | Qualification |
|---|---|---:|---:|---:|---|
| C1 | C-STR8 FP32 | 3 | 3 | 1 | PASS |
| C2 | C-STR8 full FP16 | 7 | 3 | 32 | PASS |
| L1 | L-S075 FP32 | 3 | 3 | 1 | PASS |
| L2 | L-S075 full sparse FP16 | 17 | 3 | 0.03125 | PASS, but only after 14 overflows |
| L3 | L-S075 FP16 + sparse FP32 island | 7 | 3 | 32 | PASS |
| F1 | F-U FP32 | 3 | 3 | 1 | PASS |
| F2 | F-U full sparse FP16 | 18 | 0 | none through scale 0.00390625 | bounded FAIL |
| F3 | F-U FP16 + sparse FP32 island | 8 | 3 | 16 | PASS |

All accepted windows have finite loss/parameter/boundary gradients and exact
optimizer/scheduler/EMA/exposure accounting. No cell skipped after its first
accepted window. Per-mode canonical state hashes and replayed RNG identities are
equal across precision regimes.

### What Q1 establishes about the sparse overflow

- The six-task losses and `head.input`/`second.output` gradients remain finite in
  the failing low-scale F2 attempts. The failure therefore does not begin as a
  nonfinite loss or at the head input.
- In L2, scale 1 still leaves 158 nonfinite parameter-gradient elements beginning
  at `lidar_encoder.backbone.stem.0.weight`. At scale 0.03125 it becomes finite;
  the unscaled stem maximum is about 1.29M, whose scaled value is about 40.4K.
- In F2, the same first bad sparse stem weight remains nonfinite after 18 bounded
  attempts. At the final attempted scale 0.00390625 only ten nonfinite parameter
  elements remain, while the unscaled finite maximum reaches about 16.63M; its
  scaled magnitude is about 64.96K, immediately below the FP16 finite ceiling.
  Scale 0.001953125 was produced by the final backoff but was not attempted under
  the frozen 18-window bound.
- F3 keeps all SECOND activation and parameter gradients finite and accepts at
  scale 16. Together with L3 at scale 32, this localizes the practical failure to
  FP16 sparse-convolution backward/weight-gradient dynamic range, especially the
  SECOND stem, rather than proving a head/loss semantic defect.
- The FP32 reference itself confirms unusually large but finite sparse gradients:
  the first L1 stem parameter-gradient maximum is about 1.92M and F1 about 218K.
  Q1 does not prove the architectural/normalization cause of that magnitude or
  certify the training recipe beyond three accepted windows.

The S08 precision candidate for Q2 is therefore global FP16 with SECOND/spconv in
FP32; camera-only remains global FP16, and uniform FP32 remains the reference/
fallback. Full sparse FP16 is not accepted as the unified F-capable policy.

### Q1 preserved artifact identities

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `environment.json` | 228 | `dcb00e9f854c6ed57e47939a3c52fd3951e8b6d18965ade02435d91023702d4d` |
| `q1.log` | 2,310 | `0d249dc246fc45c0846144c2d5c662b9b05923dcaba68274db16f67a2019a3b1` |
| `q1.junit.xml` | 377 | `95d5694a0636606f0801f0c480ba77f7e2dfe0fe8c335908fec84e5da650cb29` |
| `q1.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `raw/fixture_manifest.json` | 5,980 | `61bbcb481109937c02d5010074b5a5de7d1b2ff445fd8ccc1df6a92956beb43c` |
| `raw/fixture_identity.json` | 549 | `6f44f71692a79a443b4fdce4abe528184e8eab7c61f018960815024ffab709b2` |
| `raw/resolved_configs.json` | 29,100 | `7a8e142d40a750954958fba5bdada651aa5389f5d5dffe1c8e5612aa83795cbe` |
| `raw/window_records.jsonl` | 911,863 | `6e8b6f676bebfe67c6808f8d478be018188284473d6cfbcbe62752b756827bef` |
| `raw/q1_partial_summary.json` | 12,716 | `6e225c2b439b621dc9e75bbf555bebcad2c38fe7747218a691c02974644f76b3` |
| `raw/q1_summary.json` | 875,444 | `3c30b017d689eb4fc32bf01f2c391d4647485adb30a96091a59b35c2b62e00de` |
| `artifact_sha256s.txt` | 856 | `5f606fc73b67fdbb188f20eb970c5040636960440b6f0cc093c2b98fe58202e2` |
| `slurm-431013.out` | 9,943 | `e603734f82de8969fc766b9ab34f04e52506ca7dbd3d61b2a3aff0986ca6131a` |
| `slurm-431013.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

This is bounded mini precision qualification only. It is not convergence,
capability, performance, mAP/NDS, production-data readiness, or final owner
precision-policy acceptance.

## S08-SMOKE-5 terminal result

```text
REQUEST_ID: S08-SMOKE-5
OWNER_APPROVAL: explicit post-freeze exact-tuple confirmation on 2026-07-14 / O-106
SUBMISSIONS: 1
JOB_ID: 429080
STATE: COMPLETED
EXIT_CODE: 0:0
RESTARTS: 0
SUBMIT: 2026-07-14T18:08:47+02:00
START: 2026-07-14T18:08:48+02:00
END: 2026-07-14T18:12:24+02:00
ELAPSED: 00:03:36
NODE: n23
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal focused review-remediation and fixture-attestation **PASS**.
S00 did not treat the owner's pre-freeze Smoke-5 message as execution authority;
the exact tuple was submitted only after renewed post-freeze confirmation. The
read-only snapshot, runner, outer submit script, mini input manifest, resources,
output, and stop conditions were used once. No source/environment mutation,
retry, alternate node, Q1, or additional GPU job was attempted.

### Exact execution identity

```text
BASE_IMPLEMENTATION_COMMIT: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke5_51daec3e860e
SNAPSHOT_TREE_SHA256: 51daec3e860e6d412ad57d807efd78a08b03630afb37798880999fa039900a25
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4515200 / 0
SMOKE_RUNNER_SHA256: 08b74822862e6e91f14802426b76bfff29dfdd7ace85482a9882a94914941ff1
SUBMIT_SCRIPT_SHA256: 254064b207f004ae778f1c73c5e474f0cdf74642a1ba50724adec6e4911ffd40
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
RAW_INPUT_MANIFEST_FILE_SHA256: 62a63cf6c3dd4295f8c246fdef6ba170e7685cab6930294b17633a1d448798b4
RAW_INPUT_MANIFEST_LOGICAL_SHA256: f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke5_51daec3e860e
```

Runtime attestation matched aarch64, one `NVIDIA GH200 120GB`, Python `3.11.15`,
Torch `2.11.0+cu128`/CUDA `12.8`, exact Torch executable/source identities,
spconv `2.3.8`, cumm `0.7.13`, exact tracked-source states, the requested mini
dataroot, and both raw-input-manifest hashes.

### Phase results and fixture identities

Phase 1 JUnit is exactly 116 tests, 0 failures, 0 errors, and 0 skips; pytest
reports `116 passed, 5 warnings in 43.55s`. This includes the repaired calibration
drift negative case, the other nine review-remediation cases, the previous 106
focused contracts, and both tiny sparse paths.

Phase 2 JUnit is exactly 1 test, 0 failures, 0 errors, and 0 skips; pytest reports
`1 passed, 2 warnings in 2.04s`. The candidate-only attestation states
`model_constructed=false`, `optimizer_constructed=false`, and `q1_executed=false`.
All five identities were independently recomputed from strict JSON/tensor metadata:

| Identity | SHA-256 |
|---|---|
| raw-input manifest | `f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316` |
| complete batch tensor manifest | `de8b8f06c8c5b14871262fe56167ac52095f8e7cac42387de157b8e247a4e9da` |
| augmentation field order | `0495e2db0984cf3063ef5d0d84a2fd83b99b1b0cf3383f7a78534bbce8bb5de7` |
| augmentation values (`torch.float64`) | `57728184c564966e83d19214e192e8fc79fd84a2701b46b8299c237eb61dd9ea` |
| canonical fixture manifest | `f46a79c1cefa52a65d9e402b791cfce73fa194f20e6aa7cbfb3096957b6b9c89` |

The first independent augmentation recheck mistakenly packed the declared
`torch.float64` values as float32 and therefore disagreed. Reading the exact dtype
from the batch tensor manifest and recomputing as float64 matched the recorded
tensor, fixture, and identity values. This was an external verification-script
mistake, not a job/artifact inconsistency.

### Preserved artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `environment.json` | 1,463 | `db81987f80d3270dc88976aafd2dd584014ce707296376d0fd7905c02803dfc7` |
| `smoke.log` | 2,547 | `b504c1640fd8c2cf8c47931f133a9aa516a6ae0aeab4c32de1426cbd763d313f` |
| `smoke.junit.xml` | 17,371 | `a2cfd248875dbbffbb4a5ed10ffb38403a4729593e3c11a728b60676b98e7e98` |
| `smoke.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `fixture-attestation.log` | 1,233 | `50b08247c304e2d0b5bee75a65243c86e1cb9203a4dc6d57a2afa6b18ddad63e` |
| `fixture-attestation.junit.xml` | 360 | `d4de3c1ce0f6fbf7a1eb74d881ec826325e44399a526496bc521cb975fd95b65` |
| `fixture-attestation.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `fixture-attestation/fixture_manifest.json` | 5,980 | `61bbcb481109937c02d5010074b5a5de7d1b2ff445fd8ccc1df6a92956beb43c` |
| `fixture-attestation/fixture_identity.json` | 549 | `6f44f71692a79a443b4fdce4abe528184e8eab7c61f018960815024ffab709b2` |
| `fixture-attestation/fixture_attestation.json` | 965 | `fea4e8beae5f9c7ec1d8dda470aee53d7d7cf419d58ddffae0a27432925d4c5a` |
| `artifact_sha256s.txt` | 922 | `5badf259d1e9aa0edf353b960f65aa5139f6185894c342a9c63db98c7adb0636` |
| `slurm-429080.out` | 3,805 | `87ea57ce38480db51b2c598d3b3d019a3aa05b12c797faa9a4bce00103fd9a2f` |
| `slurm-429080.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

The job-created checksum manifest verified all ten declared runtime/test/fixture
artifacts. Both exit files are exactly `0`; stdout ends with
`S08_PRECISION_SMOKE_PASS`. The five Phase-1 warnings are two unregistered `slow`
marker warnings, one dependency deprecation, and two known spconv indexing
warnings; Phase 2 repeats the two marker warnings. They did not alter test count,
skip status, or acceptance. Pytest temporary files remain preserved.

### Disposition

- O-106's exact single-submission authority is consumed.
- The focused O-104 remediation and candidate fixture-attestation runtime gate
  pass; the earlier Smoke-4 negative evidence remains preserved.
- No Q1 cell, model/optimizer construction, or precision regime ran.
- O-108 authorizes a new immutable remediation/evidence commit and independent
  re-review; both remain required before any exact Q1 request.
- Precision-policy acceptance and S09 remain blocked.

## S08-SMOKE-4 terminal result

```text
REQUEST_ID: S08-SMOKE-4
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14 / O-105
SUBMISSIONS: 1
JOB_ID: 428889
STATE: FAILED
EXIT_CODE: 1:0
RESTARTS: 0
SUBMIT: 2026-07-14T17:52:50+02:00
START: 2026-07-14T17:52:51+02:00
END: 2026-07-14T17:56:14+02:00
ELAPSED: 00:03:23
NODE: n501
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal review-remediation smoke **FAIL**. The exact snapshot, runner,
outer submit script, mini input manifest, resources, output, and stop conditions
were used once. Runtime/dependency/source/raw-manifest attestation passed. Phase 1
collected exactly 116 tests and ended 115 passed/1 failed/0 errors/0 skips. The
runner stopped before Phase 2, so no candidate fixture identity was emitted. No
source/environment mutation, retry, alternate node, Q1, or additional GPU job was
attempted.

### Exact execution identity

```text
BASE_IMPLEMENTATION_COMMIT: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke4_425568c1c83d
SNAPSHOT_TREE_SHA256: 425568c1c83df06889c17d305b4ee8a9264b0535d7c204d0d34c8427aa18e90f
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4503677 / 0
SMOKE_RUNNER_SHA256: 08b74822862e6e91f14802426b76bfff29dfdd7ace85482a9882a94914941ff1
SUBMIT_SCRIPT_SHA256: cb6e3a3da2969d7c522db4a83e07202deef2a2434346aa544aa8094a6e1d2c29
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
RAW_INPUT_MANIFEST_FILE_SHA256: 62a63cf6c3dd4295f8c246fdef6ba170e7685cab6930294b17633a1d448798b4
RAW_INPUT_MANIFEST_LOGICAL_SHA256: f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke4_425568c1c83d
```

`environment.json` confirms aarch64, one `NVIDIA GH200 120GB`, Python `3.11.15`,
Torch `2.11.0+cu128`/CUDA `12.8`, exact Torch executable/source identities,
spconv `2.3.8`, cumm `0.7.13`, both exact tracked-source states, the requested
mini dataroot, and both raw-input-manifest hashes.

### Exact failure

The sole failing node was:

```text
test_q1_changed_complete_input_fails_fixture_gate_before_model_construction[cam_intrinsics]
```

The synthetic baseline used `torch.eye(4)` and the test attempted to create drift
with `changed["cam_intrinsics"].reshape(-1)[0] = 1.0`. That first diagonal value
was already exactly `1.0`; therefore the before/after tensor bytes and
`batch_tensor_manifest_sha256` correctly remained equal, and the test's inequality
assertion failed. The `images` and `gt_boxes` variants, fixture-environment schema,
augmentation-field order, positive scheduler/EMA timeline, all four hostile
scheduler/EMA cases, the previous 106 focused cases, and both tiny sparse cases
passed. Five warnings were recorded: two unregistered `slow` markers, one
dependency deprecation, and two known spconv indexing warnings.

This is a test-input construction defect. It does not show that an actually
changed calibration tensor can bypass the fixture gate; that specific negative
case remains unvalidated until a changed value is used and rerun. It is not an
environment, production-loop, model, precision, or LiDAR numerical failure.

### Preserved artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `environment.json` | 1,463 | `db81987f80d3270dc88976aafd2dd584014ce707296376d0fd7905c02803dfc7` |
| `smoke.log` | 4,067 | `7310d7c19de3de3be9e140489b9c295d16807cefb7f3b6a027d35f4b8e3fa5df` |
| `smoke.junit.xml` | 18,625 | `a22be0a9ebddd2f200af0fed69f3d7070cca411c5ef5100f94ebb812b40ea1b2` |
| `smoke.exit` | 2 | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` |
| `slurm-428889.out` | 4,067 | `7310d7c19de3de3be9e140489b9c295d16807cefb7f3b6a027d35f4b8e3fa5df` |
| `slurm-428889.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

`smoke.exit` is exactly `1`. Because the runner stopped at the Phase-1 gate,
`fixture-attestation.log`, its JUnit/exit files, the three fixture JSON artifacts,
`artifact_sha256s.txt`, and terminal `S08_PRECISION_SMOKE_PASS` are correctly
absent. Pytest temporary files are preserved under the bounded output root.

### Disposition

- O-105's exact single-submission authority is consumed.
- No retry or replacement request is authorized.
- The immutable S08-SMOKE-3 PASS remains valid within its prior focused scope.
- Independent review remains `REMEDIATE`; Q1 and S09 remain blocked.
- Per the request stop condition, source/test remediation and any replacement
  snapshot/request return to the owner before action.

## S08-SMOKE-3 terminal result

```text
REQUEST_ID: S08-SMOKE-3
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14
SUBMISSIONS: 1
JOB_ID: 428112
STATE: COMPLETED
EXIT_CODE: 0:0
RESTARTS: 0
SUBMIT: 2026-07-14T17:09:14+02:00
START: 2026-07-14T17:09:15+02:00
END: 2026-07-14T17:12:43+02:00
ELAPSED: 00:03:28
NODE: n576
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal focused implementation-smoke **PASS**. The exact approved
read-only snapshot, runner, outer submission script, source-state identities,
resource tuple, selectors, output root, and stop conditions were used once. No
source/environment mutation, alternate node, retry, Q1 cell, or additional GPU
job was attempted.

### Immutable execution identity

```text
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke3_3014cab90ed8
SNAPSHOT_TREE_SHA256: 3014cab90ed88b5705367fc1dd1a21740593acc3a186c72f9073bffe15247a43
SMOKE_RUNNER_SHA256: 266b83f558b8d9c60f4086d633ac79326cd0dbf3e9c063837d653acf9d44cdf0
SUBMIT_SCRIPT_SHA256: 5fa6e31df27425fde7f04373519d39d30c01386fa3e9b487e406048f11bd6ac0
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke3_3014cab90ed8
```

Runtime attestation matched the request exactly:

```text
machine/device: aarch64 / NVIDIA GH200 120GB
Python/Torch/CUDA: 3.11.15 / 2.11.0+cu128 / 12.8
Torch build/source: a58ba749... / 70d99e99...
spconv version/build/source: 2.3.8 / 74934de8... / 263d6b47...
spconv tracked-state SHA256: 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db
cumm version/build/source: 0.7.13 / 0a7e3c1a... / 4dedaf43...
cumm tracked-state SHA256: f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662
```

### Pytest and acceptance-gate result

JUnit records `106` tests, `0` failures, `0` errors, `0` skipped, and `43.468`
seconds. Pytest reports `106 passed, 3 warnings in 43.48s`; `smoke.exit` is
exactly `0`, Slurm stdout ends with `S08_PRECISION_SMOKE_PASS`, and all four
runner checksums verify.

The warnings are unchanged dependency notices: one Python-locale deprecation in
`ccimport` and two spconv multidimensional-indexing warnings. They are not hidden,
promoted to acceptance evidence, or interpreted as a model-numerical result.

This pass includes the two previously blocked real disabled-GradScaler paths:
the output/RNG-neutral FP32 update and hostile pre-step diagnostic cleanup. It
also includes the corrected six-task rejection expectation, exact source-state
positive/negative tests, precision partition/config/checkpoint/runtime regressions,
and both tiny sparse FP32/FP16/island tests.

### Produced artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `environment.json` | 1,139 | `2e7884a35a43fa6dc5f422602e9e75166cbc67174d48836e2ebf9bd4d88fde8d` | complete and identity-matching |
| `smoke.log` | 1,577 | `5ef5db037debb49bea335d9bd9f2daea0b2f1d725aced0af5b791b66a9a36796` | complete pytest PASS log |
| `smoke.junit.xml` | 15,780 | `3b5dcdc2d7559b1af80446e946858d0133de1ad531802c105b43b6d128f76171` | complete JUnit |
| `smoke.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | exactly `0` plus newline |
| `artifact_sha256s.txt` | 318 | `a6aea314859224f2a0c238fae693a0ae5d3eabe417d11ca5131b9835311ed7b7` | all four runner checksums reverified |
| `slurm-428112.out` | 1,602 | `d1881872129b5301cc34e5f5df4cb887b6913528452cb8c64eac41c3df171b9f` | pytest replay plus terminal PASS marker |
| `slurm-428112.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` | module notice only |

Pytest's bounded temporary fixtures remain under `pytest-tmp` in the preserved
output root; they include only the declared toy Git/config/checkpoint fixtures
and no nuScenes data.

### Historical disposition and interpretation boundary at Smoke-3

- The exact S08-SMOKE-3 authorization is consumed and terminal; it grants no
  retry or additional compute.
- The focused GH200 implementation gate is satisfied, so the previously
  authorized immutable implementation/evidence commit may now be created.
- Independent review of that exact Git object remains required before Q1.
- At the Smoke-3 disposition, Q1 remained unapproved and unexecuted; S09 remained blocked on reviewed S08
  evidence and owner precision-policy acceptance.

This PASS establishes only the exact tracked-source attestation, focused
config/routing/checkpoint/training-loop/window-diagnostic contracts, and tiny
sparse FP32/FP16/island runtime paths. It does **not** establish stable current
six-task optimizer windows, select the scientific precision policy, explain the
large LiDAR gradients, prove convergence/performance/capability, or support
mAP/NDS, Protocol A/B, attack, or defense claims.

## S08-SMOKE-2 terminal result

```text
REQUEST_ID: S08-SMOKE-2
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14
SUBMISSIONS: 1
JOB_ID: 427800
STATE: FAILED
EXIT_CODE: 1:0
RESTARTS: 0
SUBMIT/START: 2026-07-14T16:53:43+02:00
END: 2026-07-14T16:57:03+02:00
ELAPSED: 00:03:20
NODE: n23
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal **focused-test FAIL**, not a provenance, CUDA, spconv,
sparse-kernel, model-numerics, or Slurm-infrastructure failure. Exact runtime
attestation completed successfully before pytest:

```text
machine/device: aarch64 / NVIDIA GH200 120GB
Torch build/source: a58ba749... / 70d99e99...
spconv build/source: 74934de8... / 263d6b47...
spconv tracked-state SHA256: 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db
cumm build/source: 0a7e3c1a... / 4dedaf43...
cumm tracked-state SHA256: f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662
```

This confirms that the O-100 exact source-state remediation solved Job `426619`'s
blanket-clean-checkout blocker without editing the external dependency trees.

### Pytest result

JUnit records `106` tests, `3` failures, `0` errors, `0` skipped, and `42.569`
seconds (`103 passed, 3 failed, 3 warnings` in the pytest summary).

| Failing test | Exact cause | Classification |
|---|---|---|
| `test_enabled_diagnostics_preserve_fp32_update_metrics_and_rng` | `PrecisionWindowDiagnostics.begin_window()` called `GradScaler.get_backoff_factor()` on a disabled CPU scaler; PyTorch 2.11 does not create `_backoff_factor` when `enabled=False` | real diagnostics compatibility defect; it failed before the observed forward/update, so output neutrality remains unproven by this test |
| `test_pre_step_diagnostic_failure_discards_window_without_parameter_update` | same disabled-scaler accessor failure prevented the intended injected pre-step failure path from being reached | real diagnostics test-path defect; hostile-cleanup behavior remains unexecuted in this test |
| `test_multitask_loss_rejects_legacy_single_head_output` | production correctly raised `ValueError('multi-task CenterHead must return 6 task dictionaries')`, but the new test expected regex `six task` | test expectation defect only; the fail-closed six-task loss behavior occurred correctly |

The first two failures expose a narrow but Q1-blocking issue: FP32 cells use a
disabled GradScaler, so diagnostics must not query enabled-only growth/backoff
attributes. FP16 uses an enabled scaler and is not implicated by this trace. The
third failure requires no production loss change.

All other selected cases passed, including exact source-state positive/negative
tests, precision partition/config/checkpoint/runtime regressions, diagnostic helper
tests, and both selected tiny sparse FP32/FP16/island tests. This is bounded test
evidence only; because the overall smoke gate failed, it is not an implementation
PASS or authorization to commit/Q1.

### Produced and missing artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `environment.json` | 1,139 | `2e7884a35a43fa6dc5f422602e9e75166cbc67174d48836e2ebf9bd4d88fde8d` | complete and identity-matching |
| `smoke.log` | 6,798 | `e108946a524f8d317a67d92d16cb625503f803116e3952a0760d6cd9e3381a0b` | complete pytest failure log |
| `smoke.junit.xml` | 20,528 | `19d5a3ff5cb733ed4c2847c9a58368aac36dc158f27bf347e2229f0d4a0495c1` | complete JUnit |
| `smoke.exit` | 2 | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` | exactly `1` plus newline |
| `slurm-427800.out` | 6,798 | `e108946a524f8d317a67d92d16cb625503f803116e3952a0760d6cd9e3381a0b` | runner replay of `smoke.log` |
| `slurm-427800.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` | module notice only |
| `artifact_sha256s.txt` | — | — | not created because the runner stopped on nonzero pytest exit before its success-only checksum step |

The complete output directory is preserved at:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke2_935d0464b3bf
```

### Disposition

- The exact S08-SMOKE-2 authorization is consumed and terminal.
- No source/environment edit, resubmission, alternate node, or retry was attempted.
- At this S08-SMOKE-2 disposition, the owner-authorized immutable implementation
  commit remained gated; later S08-SMOKE-3 resolved that gate.
- Q1 remained not ready and was not submitted.
- O-102 subsequently authorized conditioning scaler-policy accessors on
  `scaler.is_enabled()`, asserting `None` policy fields for the real disabled
  scaler, correcting only the test regex, and preparing S08-SMOKE-3. That narrow
  remediation is implemented locally; O-102 grants no GPU execution authority.

This result does not qualify a precision regime, explain the large LiDAR
gradients, prove stable optimizer windows, establish convergence/performance/
capability, or support mAP/NDS, Protocol A/B, attack, or defense claims.

## S08-SMOKE-1 terminal result

```text
REQUEST_ID: S08-SMOKE-1
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14
SUBMISSIONS: 1
JOB_ID: 426619
STATE: FAILED
EXIT_CODE: 1:0
RESTARTS: 0
SUBMIT: 2026-07-14T15:57:08+02:00
START: 2026-07-14T15:57:09+02:00
END: 2026-07-14T15:58:07+02:00
ELAPSED: 00:00:58
NODE: n535
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal **pre-pytest provenance FAIL**. The runner activated the
Arrhenius environment and entered `verify_runtime_dependency_identity()`, but
stopped before pytest collection, model construction, forward/backward, an
optimizer window, or any diagnostic observer execution:

```text
RuntimeError: spconv source checkout is modified
```

Therefore Job `426619` is not evidence for or against the S08 config resolver,
FP32 island, full sparse FP16, training loop, GradScaler behavior, sparse kernels,
model numerics, LiDAR gradients, or opt-in diagnostics.

## Immutable execution identity

```text
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke_f963da5a620e
SNAPSHOT_TREE_SHA256: f963da5a620e38a479bf9cee3a80af489bb9d212db79848678ff0560a9555ec2
SMOKE_RUNNER_SHA256: aab4656a339598366e2e0d34927cdf2812119459e15444b6e6a7b7e82487c8c9
SUBMIT_SCRIPT_SHA256: cf562f83422600a3cff6a4735b4c08842d3113a4957a44461d119faa668b01a4
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke_f963da5a620e
```

The submission used the exact approved command and resource tuple. No selector,
snapshot, source, environment, output path, or resource was changed after owner
approval.

## Produced and missing artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `environment.json` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | opened by shell redirection, then left empty by the failed identity check |
| `slurm-426619.out` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | empty |
| `slurm-426619.err` | 871 | `a41f36cb107b49d871fe6db2211e841a6ee4fa4c4d62ec544971b42789299611` | module notice plus exact traceback |
| `smoke.log` | — | — | not created; pytest never launched |
| `smoke.junit.xml` | — | — | not created |
| `smoke.exit` | — | — | not created |
| `artifact_sha256s.txt` | — | — | not created |

The output directory and all three produced files are preserved. Absence of the
later artifacts is expected from the fail-fast position and is part of the FAIL,
not an omitted result.

## Root-cause reconstruction

The editable spconv checkout is:

```text
SOURCE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/spconv
HEAD: 263d6b47425ef843c82f997b12d8b714013d216c
TRACKED_STATUS: M pyproject.toml
DIRTY_FILE_SHA256: e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9
HEAD_FILE_SHA256: 798e00722756f5e029d889d72e1072e768a1e2729ed12fccfc95757609908333
```

The sole tracked diff removes `"cumm>=0.7.11"` from
`[build-system].requires`. No `.py`, CUDA, C++, shared-object, or other executable
source file is modified. The cumm checkout remains clean at
`4dedaf43ff801e417c60c6bd7536a29d83d29ee0`.

This state predates S08. Accepted S07-B-COMPLETE Job `374142` preserved the exact
same status, diff, and dirty-file SHA in:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_diag2_34cbe02b7b72/artifacts/spconv-source-state-before.txt
SHA256: a1a54ce2b72d3caee796c68a5b3662fb90edf15f61dd5c5fcb873d67ae29e7c3
```

That accepted evidence also binds the same installed spconv build SHA
`74934de877e07a8eef8edacd4e31ec0f06eff030b3bc7e06d01f41b1444687d8`
and source HEAD requested by S08-SMOKE-1. The current verifier nevertheless
requires every editable checkout to have an empty tracked status before it hashes
the installed executable artifacts. The observed failure is consequently a
provenance-policy mismatch with the known source-built environment, not evidence
of new runtime drift.

## Disposition and next gate

- The exact S08-SMOKE-1 authorization is consumed and terminal.
- No retry, alternate node, environment edit, source reset, or extra job was
  attempted or authorized.
- At this S08-SMOKE-1 disposition, the owner-authorized implementation commit
  remained gated because focused GH200 runtime validation did not execute.
- Q1 remained not ready and was not submitted.
- O-100 subsequently authorized the narrow provenance-policy remediation and
  replacement-request preparation. Exact `S08-SMOKE-2` was then frozen,
  separately approved under O-101, and consumed by terminal Job `427800`; its
  result is recorded above.

The implemented narrow remediation keeps exact source HEAD and installed
executable-build hashing, while permitting only the already-evidenced spconv
`pyproject.toml` build-metadata patch by an exact path/content identity. Any
executable-source change, additional tracked change, unrecognized metadata hash,
source HEAD drift, import-origin drift, or installed-build drift must still fail
closed. It did not reset or modify the external spconv checkout. Replacement
execution was outside O-100 and received the separate exact O-101 approval
recorded above; O-100 alone granted no execution authority.

## Interpretation limits

This result establishes only that the original smoke wrapper's blanket clean-Git
precondition is incompatible with the accepted Arrhenius spconv checkout. It does
not select a precision policy, explain the large LiDAR gradients, validate stable
optimizer windows, establish convergence/performance/capability, or support any
mAP/NDS, Protocol-A/B, attack, or defense claim.
