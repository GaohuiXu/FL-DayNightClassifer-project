# S05 focused synthetic runtime results

## Scope and verdict boundary

- Execution source: `96e509b71a3e22afb4de397132438fd3b9bbf5d8`;
  tree `aeaaad044199492b81c4383a013f3fb3c6596c02`.
- Approval class for the one-time rerun:
  `S00_OWNER_DELEGATED_S02_S05_VALIDATION_RERUN`.
- This was an owner-delegated validation approval, explicitly not an O-009
  expansion. O-009 excludes reruns.
- Scope: exactly 44 synthetic CenterHead/decode/NMS/submission cases; no dataset,
  checkpoint, model/optimizer step, scientific metric, profile, array, DDP, or
  follow-on.
- Focused runtime result: **PASS on Job 336738**. Independent S05 review acceptance
  remains a separate verdict.

## Complete job ledger

| Job | Immutable source | Terminal | Tests | Interpretation |
|---|---|---|---|---|
| `336731` | `705216de097ae9eeb1813de6dcdc916e2844fcde` | **FAILED 1:0**, `00:01:15`, n570 | 44 collected; 43 pass, 1 fail, 0 error/skip | Preserved negative. Critical `forward == reverse` passed; only expected list vs actual stable devkit tuple differed. No production defect observed. |
| `336738` | `96e509b71a3e22afb4de397132438fd3b9bbf5d8` | **COMPLETED 0:0**, `00:01:13`, n411 | **44 pass**, 0 fail/error/skip, `22.64s` | Exact tuple-only correction passed. Closes the focused synthetic runtime gate only. |

There was no automatic retry. Job 336738 was the single separately approved
focused rerun after S00 reviewed Job 336731 and the immutable correction request.

## Job 336731 preserved negative

- JUnit: 44 tests, 1 failure, 0 errors/skips, time `22.878s`.
- Failure:
  `test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content`.
- The permutation-invariance assertion passed. Installed nuScenes devkit
  serialization returned velocity tuples, while the expected literal used lists.
- stdout SHA-256:
  `fbeac7dbcc5b14cf1f377a6ca1e363c06e4932eb66a416d1179ea02349249b6e`.
- stderr SHA-256:
  `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`.
- pytest log/JUnit SHA-256:
  `3e461e6e83df9dedbdd68b2e0059e4afc2348bc54d56efebf55ef57a348a20fc` /
  `0f79ed5509881bcc84a48f8dd546ebc69de0fd8ac4cdbfe074a8cd5ee806288e`.

## Job 336738 exact PASS evidence

Scheduler/allocation:

- `COMPLETED 0:0`; elapsed `00:01:13`; batch MaxRSS `540M`;
- node `n411`; one shared node; eight CPUs; exactly one
  `nvidia_gh200_120gb`;
- aarch64; Python `3.11.15`; torch `2.11.0+cu128`; NumPy `1.26.4`;
  pytest `9.1.1`; nuscenes-devkit `1.1.11`; Pillow `12.2.0`.

Execution identity:

- approved delivery: `98b71eca7684b50ece69afc36175564c7c283033`;
- execution SHA/tree: `96e509b71a3e22afb4de397132438fd3b9bbf5d8` /
  `aeaaad044199492b81c4383a013f3fb3c6596c02`;
- approved request SHA-256:
  `e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9`;
- launcher SHA-256:
  `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5`;
- runtime source-list/source-state SHA-256:
  `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857` /
  `7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1`.

Test result:

```text
............................................                             [100%]
44 passed in 22.64s
S05-R3 JUnit acceptance PASS: {'tests': 44, 'failures': 0, 'errors': 0, 'skipped': 0}
```

Every one of the nine paths listed in `sha256sums.txt` passed in-job
`sha256sum -c`.

## Job 336738 artifact manifest

Output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r3_centerhead_96e509b71a3e`.

| Artifact | SHA-256 |
|---|---|
| stdout `s05r3_centerhead_336738.out` | `0cf6f1dc14ad07ef598076fb6ed067352bf71c789172f9babd5f1ed42d01ef87` |
| stderr `s05r3_centerhead_336738.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |
| `approved_launcher.sh` | `b86271e81ec41443232afab6a6ada5d1dbebfa72027946cea6547ee5c01598e5` |
| `approved_run_request.md` | `e4cb396bc550f08e92905903135f9ab0841ba1bd498f661ba731587a843a10b9` |
| `snapshot_identity.txt` | `6c47a4252bb65c227ef795eecd161749e5260ce6821a5a638da7b5457ab0aa20` |
| `runtime_source_files.txt` | `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857` |
| `runtime_source_sha256s.txt` | `7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1` |
| `execution_identity.json` | `9e2dde2468f17d10b99c2992440029b347f4b4a220143c3aecce7c6b84a62aab` |
| `slurm_allocation.txt` | `c76ffe8201b2025d7ed7b0cbf663fca8706073c10efee090fac0ed2347dba3d8` |
| `pytest.log` | `4db65ef4592e61cf1886e49bef9649ba87803b6cf41bc45e84de6484645121d3` |
| `pytest.junit.xml` | `bad9b34e02a4d7267cbbed4e2b4429c6498360a3c3317388fdc21f0be8206910` |
| `sha256sums.txt` | `301c5c4feed506f0ae5c130b1036cfe0c0aaeacf81f947cf121a6136f7339077` |

The stderr hash is the same benign module-purge advisory seen in the prior job;
it contains no test/runtime error.

## Allowed and forbidden interpretations

Allowed:

- the exact 44 authored synthetic cases passed under the dependency-complete
  Arrhenius GH200 environment at execution SHA `96e509b`;
- the corrected duplicate-geometry velocity/attribute ordering fixture passed;
- prior forced-FP32, no-starvation, label-map, NMS, geometry, and fail-closed
  fixtures all passed in the same exact suite.

Forbidden:

- erasing or calling Job 336731 PASS;
- claiming independent S05 acceptance before the reviewer issues its final verdict;
- claiming production detector/loss/config/checkpoint integration, official CUDA
  kernel parity, CPU-NMS production performance, mini/trainval quality, mAP/NDS,
  full-run readiness, FL/security behavior, or any scientific result.

No further compute, retry, merge, push, upload, or scope expansion is requested.
