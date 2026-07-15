# S09 RUN_REQUEST — consumed terminal ledger

## Current authority

```text
ACTIVE_REQUEST: none
S09_STATE: CLOSED PASS UNDER O-120
INTEGRATION: completed ff-only at 351b7a0 under O-121
COMPUTE_AUTHORITY: all S09 requests consumed
NEW_COMPUTE/RETRY/DDP/PROFILE/FULL_RUN: not authorized
```

Every executed tuple below was frozen before submission and independently reviewed.
No identical retry, spare-GPU expansion, array or DDP execution occurred. The full
1149-line pre-compaction command/authorization history is recoverable from Git
object `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`.

## STOP-1 — exact production cache identity

```text
AUTHORITY: O-112; owner acceptance O-113
SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
JOB: 441191 / COMPLETED 0:0 / zero restarts / 00:03:06
RESOURCE: 1 GH200 / 8 CPU / 96 GiB / 00:30:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop1_cache_t1v2_1f276b9d2cc5
DATA: module v1.0-trainval / splits train,val / n_sweeps=10
ZIP_MANIFEST_LOGICAL_SHA256: 023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6
ZIP_MANIFEST_PHYSICAL_SHA256: 228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb
OUTPUT_MANIFEST_SHA256: 4f48ea4e7ebfc9427a4cf649e3b3826feb0b529f7a56af011b4e1b78a8f5f2ef
```

This was metadata/cache materialization and validation only. It did not extract
payloads, construct a detector/DataLoader profile, train, evaluate or produce
throughput evidence.

## STOP-2 — readiness implementation smoke

```text
AUTHORITY: O-114 implementation / O-115 exact smoke / O-116 close
SOURCE_SHA/TREE: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952 / d0626e313aab411bc5c71733afb41eca5b102693
JOB: 441293 / COMPLETED 0:0 / zero restarts / 00:01:04
RESOURCE: 1 GH200 / 4 CPU / 32 GiB / 00:10:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop2_smoke_37aef4d6b3f4_a1
SELECTORS: four exact readiness/runtime CUDA tests
PYTEST: 44/44
ARTIFACT_MANIFEST_SHA256: 643160908f29f76cccbdcde3e5999759934aa5417b3d71252810e917ae4667ff
```

The request permitted at most two O-107 mechanical replacements; none was used.
The smoke qualified code/lifecycle mechanics only, not production loader/model
training.

## STOP-3 — first G100 failure

```text
AUTHORITY: O-117 / sole submission consumed
SOURCE_SHA/TREE: 4d6bd829450021aa0813bcece066fb1fac85f478 / affb4854689a0bf65d829a273d769c87c000174c
RESOLVED_CONFIG_SHA256: cb1723322c756579ab6740eb126de8455b65f808849ec977258c76b919f2c58c
JOB: 441511 / FAILED 1:0 / zero restarts / 00:02:29
RESOURCE: 1 GH200 / 16 CPU / 96 GiB / 01:00:00
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_4d6bd8294500
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_4d6bd8294500_a1
ARTIFACT_MANIFEST_SHA256: 0c3e2947fb124ac32d74e243575b4ffa159d2d97a6a603492196fd89df565133
```

The runner selected runtime modules, contrary to the binding build-module rule for
editable cumm/spconv imports. spconv JIT failed on missing `cublasLt.h` before
physical-data verification, loader construction/profile, model construction or
training. It answers no G100/model/utilization question. The failed import also
changed cumm native artifacts, so O-118 required fresh dependency attestation
before a replacement.

## STOP-3 — explicit O-118 recovery and strict replacement

### Phase A dependency attestation

```text
SOURCE_SHA/TREE: 788b493889bcf7be98f36b9cbb6686d51e8e5edf / 0bc61b3c2693f818ad0feb4e749af64a3947913e
JOB: 442152 / COMPLETED 0:0 / zero restarts / 00:11:52
RESOURCE: 1 GH200 / 16 CPU / 96 GiB / 00:20:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_dep_attest_788b493889bc_a1
SPCONV_BUILD_SHA256: af42200511a53ce86d77cea0306924a2dc516a74f0483ef7cfe0a6e1dc84b100
CUMM_BUILD_SHA256: 0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d
ACCEPTANCE_SHA256: 4b60f319660124d3bfac23a21bfbfa1b7c66ca920a0e4a4df03b1a512833e9b4
ARTIFACT_MANIFEST_SHA256: b176faa88df06ab955a295cac2ef63e09d51d59427b36c2c8bc11f3b27e73133
```

Phase A ran no data/model/training path. Its one-shot mutation/restoration runner
is retired after closure.

### Phase B production loader and G100

```text
SOURCE_SHA/TREE: c200bac861a42fc4338973787d3700e28ddd6c7e / c0cc4cb8c2e207e42dcc45a129ada28a3d40feb8
RAW_CONFIG_SHA256: 6733a47203bdf7a4da6e39867e6319a7beb9322257e9149f31b7dff6edacf3ce
RESOLVED_CONFIG_SHA256: ba06b72e4c5f1e54f20472e3286a516e7d4328cfb0fccd8bfc7b13095f597ab6
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_c200bac861a4
JOB: 446225 / COMPLETED 0:0 / zero restarts / 00:05:05
RESOURCE: 1 GH200 / 16 CPU / 96 GiB / 01:00:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_c200bac861a4_a2
READINESS_SHA256: 08e376e767f654bb38982127ad5ffd84d94ebaa48b3026ceba2ab7ef93a6c9b6
ARTIFACT_MANIFEST_SHA256: b229633889052c46bec5c05d6713e0102aea806a98f9170a65119f9864dbea4b
```

The exact F-U tuple used seed 0, global FP16 + SECOND FP32 island, AdamW
`1e-4/0.01`, constant scheduler, batch/accumulation/world `1/1/1`, workers 8,
no EMA/clip/augmentation/GT paste/checkpoint/evaluation, 100 accepted updates and
120-attempt cap. The loader profile tested 0/2/4/8 workers observationally while
the training loader remained fixed at eight.

## STOP-4A — bounded profile and capacity

```text
AUTHORITY: O-119
SOURCE_SHA/TREE: b509f5e527c2dd28d2db506c3f87b5a06b3b1b6a / 9c556d37d1e45ece7aad31b10881bb9eb8686424
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4a_profile_capacity_b509f5e527c2
JOB: 452520 / COMPLETED 0:0 / zero restarts / 00:09:42
RESOURCE: 1 GH200 / 16 CPU / 96 GiB / 00:30:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4a_profile_capacity_b509f5e527c2_a1
ARTIFACT_MANIFEST_SHA256: fbd07beebcd9078c5a980995e05febc1efc873469d0ff4fe61f30c6748f5272f
```

| Cell | Raw config SHA-256 | Resolved SHA-256 |
|---|---|---|
| B1 checkpoint-on profiler | `1bd9bce1b1a34f603990f07d72ac250d38465d9dd5d0a7eb1188012ab7f2eaa6` | `a0cb86122d607849f479fd04c70acac3b2b7c66d6e65875ad06c638e0db6ad2e` |
| B1 checkpoint-off | `555e2b7f278d39e2965cf19e1d15ecd4a8fa0ffe6e358eab2b43ea75219e98d3` | `5291290d0dbc372eb012bfcc2eeff4877e34db66aa654055c2ebfdf398820a87` |
| B2 checkpoint-off | `b30e837cc26ef8ce3ec001f0e17a171eec67d0b2e0ba65090f2229acc346d6ff` | `ac841713cf5c996705afb2ddf628965c4fffa169130925285f03c6113d669f6f` |
| B4 checkpoint-off | `8d3a3f7847f32c25c319b2ca77fd7a7702457e9c9fcbd02797048a06e8e88f4f` | `cf6f4effe0c9532a45f3a2503a3f98423af2e340b16ae0419d6b287655709a48` |

Each cell was bounded to 20 accepted updates from a fresh initialization. B2/B4
were capacity evidence only.

## STOP-4C — optimized G100

```text
SOURCE_SHA/TREE: c7769901201b8c507997dfa9ff5154fbe6dbb297 / 1e2c4464d2582d81e7ef7fef4740c764d0a48e8c
RAW/RESOLVED_CONFIG_SHA256: 8ca905ade59214822d9c5b894c02786af77f6f531299ed1ca25caf51d00a35ce / afcd002184e35158e129353dfb9b621c390555b5927a37fa5f5acd9547538980
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4c_g100_c7769901201b
RUNNER_SHA256: a899deb5a8a68541d2e7b361c816ae49bd873b16b3c18030e0fc54c65717daa5
JOB: 455539 / COMPLETED 0:0 / zero restarts / 00:04:06
RESOURCE: 1 GH200 / 16 CPU / 96 GiB / 00:30:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4c_g100_c7769901201b_a1
READINESS_SHA256: b8765c4be656fe7ad657157cc43c2c6915ebfc33e6411c26c2a7db829087adff
ARTIFACT_MANIFEST_SHA256: 542862b20a86d30c348237a9b448610857f86cb7554473cfbe65150360593847
```

The only deliberate engineering differences from STOP-3 were Swin checkpoint-off
and quiet ordinary loss telemetry. The old source `131619f` request was NO-GO,
never submitted and produced no output.

## STOP-4D — fresh optimized G1000

```text
SOURCE_SHA/TREE: 5642884cdbb16e1c9b3107f529dc70b3a1243c6a / b13a08819b2e203dfe355309f1310c79f94f3023
RAW/RESOLVED_CONFIG_SHA256: dfd46e1a179b3b10d98055762fe8cfc9f9f312f4faa5aec05c9f5b14a7b37928 / c3b39a3f9dbfccd673a494f8ec976aa0cad1424a63cda3e56f836b4b733f7a1b
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop4d_g1000_5642884cdbb1
RUNNER_SHA256: 43511df4f54265bfc9595aed424aa363a91ff3cc855ab0fb5fe43162885961dc
JOB: 456539 / COMPLETED 0:0 / zero restarts / 00:06:54
RESOURCE: 1 GH200 / 16 CPU / 96 GiB / 01:00:00
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4d_g1000_5642884cdbb1_a1
READINESS_SHA256: e61c1f6e6761a74b787dcdf9303fd1911868e44ebf1e5195765a2214396968b8
ARTIFACT_MANIFEST_SHA256: 6b90ae38427bb6efaa043f1a7c93432473cade8862ea5d1f9e432e87003107b3
O119_ACTUAL_GPU_HOURS: 0.345000 / 2.000000
```

This was a fresh seed-0 initialization, not a resume. Relative to STOP-4C, only
the successful-update target/cap changed from 100/120 to 1000/1020.

## Shared interpretation boundary

These requests support bounded engineering data identity, lifecycle, timing,
memory and capacity conclusions only. They do not establish convergence, recipe
or batch selection, model quality, mAP/NDS, fusion gain, multi-seed behavior,
full-GH200 utilization, Protocol A/B, FL, attack or defense. No S10 compute request
is implied.
