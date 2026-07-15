# S08 RUN_REQUEST — consumed terminal ledger

## Authorization state

```text
ACTIVE_REQUEST: none
S08_STATE: CLOSED PASS UNDER O-110
COMPUTE_AUTHORITY: consumed
AUTOMATIC_RETRY: none
NEW_COMPUTE: not authorized
```

This compact ledger preserves the exact material tuples needed to interpret S08.
The complete pre-compaction request history is recoverable from Git object
`351b7a0b8419c01d0d32ba224babbc6bdc4213ba` at this path.

## Engineering-smoke history

| Request / Job | Output root | Result |
|---|---|---|
| SMOKE-1 / 426619 | `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke_f963da5a620e` | terminal provenance-preflight FAIL |
| SMOKE-2 / 427800 | `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke2_935d0464b3bf` | terminal focused-test FAIL, 103/106 passed |
| SMOKE-3 / 428112 | `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke3_3014cab90ed8` | PASS, 106/106; manifest SHA-256 `a6aea314859224f2a0c238fae693a0ae5d3eabe417d11ca5131b9835311ed7b7` |
| SMOKE-4 / 428889 | `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke4_425568c1c83d` | terminal test-construction FAIL, 115/116 |
| SMOKE-5 / 429080 | `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke5_51daec3e860e` | PASS, 116/116 + 1/1 attestation; manifest SHA-256 `5badf259d1e9aa0edf353b960f65aa5139f6185894c342a9c63db98c7adb0636` |

Each request was submitted once; no identical retry or spare-GPU expansion
occurred. SMOKE-1/2/4 remain negative evidence rather than being hidden by the
later passes.

## S08-Q1 exact primary qualification

```text
REQUEST_ID: S08-Q1
AUTHORITY: O-109
STATE: consumed / Job 431013 COMPLETED 0:0 / no retry
EXECUTION_SOURCE_SHA: e6e28bea43f7757347da2e460cdf24e9a32b791f
REVIEWED_IMPLEMENTATION_SHA: 103c7389a47938b1f9dd0cba60251df6dce9e5bb
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_q1_dbeee35dcd6d
SNAPSHOT_TREE_SHA256: dbeee35dcd6d7bcb919f549f03c42763d5d82b2b20740815743b7aa2b3f9bc9c
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4544533 / 0
JOB_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s08_q1_dbeee35dcd6d/job.sh
JOB_SCRIPT_SHA256: 42cb555d518a6d7bb517c325c22c1f0ab8362c03da36b9cfd1f0b981d8b349e1
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q1_dbeee35dcd6d
DATA: nuScenes v1.0-mini directory backend; no ZIP/full-data scan
SAMPLE_TOKEN: 00889f8a9549450aa2f32cf310a3e305
LIDAR_INPUT: keyframe + 9 prior sweeps, deterministic 4096-point prefix
SEED: 20260713
CELL_ORDER: C1,C2,L1,L2,L3,F1,F2,F3
BOUND: <=99 attempted windows; <=24 accepted updates; 3 accepted required per qualifying cell
RESOURCE: 1 GH200 / 8 CPU / 96 GiB / 01:00:00 / no requeue
ELAPSED: 00:04:02
ARTIFACT_MANIFEST_SHA256: 5f606fc73b67fdbb188f20eb970c5040636960440b6f0cc093c2b98fe58202e2
```

Fixture identities were prebound before model/optimizer construction:

| Identity | SHA-256 |
|---|---|
| raw-input logical manifest | `f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316` |
| complete batch tensor manifest | `de8b8f06c8c5b14871262fe56167ac52095f8e7cac42387de157b8e247a4e9da` |
| augmentation field order | `0495e2db0984cf3063ef5d0d84a2fd83b99b1b0cf3383f7a78534bbce8bb5de7` |
| augmentation values | `57728184c564966e83d19214e192e8fc79fd84a2701b46b8299c237eb61dd9ea` |
| canonical fixture manifest | `f46a79c1cefa52a65d9e402b791cfce73fa194f20e6aa7cbfb3096957b6b9c89` |

| Cell | Exact route | Resolved-config SHA-256 |
|---|---|---|
| C1 | C-STR8 FP32 | `6cfc8f60d1116d1cb161c01d939ee54fac17f9c537ce58eb59fecc419ac25a64` |
| C2 | C-STR8 FP16 | `f56d0e4bf5d88a96523976ff8bd1ad2cd1b6ecdad3ca835f0643808f21984757` |
| L1 | L-S075 FP32/sparse FP32 | `d2d3fee5a8a38bbfa5200a49cda7a1a31302ddd22e1bbf50af037a9a964da257` |
| L2 | L-S075 FP16/sparse FP16 | `c77819da84bbfb5293b9044e5f41488d0dcec2f025d1da906632bf2307a3a80d` |
| L3 | L-S075 FP16/sparse FP32 | `b38cf86fa061b54ef7b85e753a2c33ef5941f57f81a1394843c14f712834ca4b` |
| F1 | F-U FP32/sparse FP32 | `9f49479c96d643ebd2072df22b9a5808f6bcfca6d17ec90c00bddb5e6e5a8201` |
| F2 | F-U FP16/sparse FP16 | `ee5eac7b7db660ca6e75d904f61579520daec64042c122a9ac82c21b10936d61` |
| F3 | F-U FP16/sparse FP32 | `1b23d9907ffc6190062be285b203b18951d648ec293707e88f7904835fda9ee9` |

All cells used AdamW `lr=1e-4`, weight decay `0.01`, batch/accumulation/world
`1/1/1`, workers 0, EMA disabled, matched per-mode initialized state and replayed
forward RNG. FP16 cells used one persistent scaler starting at 512, backoff 0.5,
up to 18 attempts including scales below one. A completed pytest was not itself a
cell PASS; each raw `qualification_pass` controlled interpretation.

## S08-Q2 exact compatibility qualification

```text
REQUEST_ID: S08-Q2
AUTHORITY: O-109
STATE: consumed / Job 435151 COMPLETED 0:0 / no retry
EXECUTION_SOURCE_SHA: 3bb10d39c60e6fd2d0bfe480bb03a7c8cfc76fe9
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_q2_1d9191c2f623
SNAPSHOT_TREE_SHA256: 1d9191c2f6234199d31405f9690ffd2d83343889333efbe1e1ae47e6235a5c60
SNAPSHOT_FILES/BYTES/WRITABLE: 585 / 4566358 / 0
JOB_SCRIPT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s08_q2_1d9191c2f623/job.sh
JOB_SCRIPT_SHA256: ff14fd735788a4fa4691a473eb788276d901371160c28f447fe8819f33494d0d
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q2_1d9191c2f623
DATA/FIXTURE/SEED: exact Q1 fixture and five identities / 20260713
CELL_ORDER: P1 then B1
P1: L-P020 / global FP16 / sparse not_applicable / uniform
B1: F-CBGS / global FP16 / SECOND FP32 island / cbgs config identity
P1_RESOLVED_CONFIG_SHA256: 7219c1a3978bf9c0d16efbaa10fa01448fd7e99793ae8c9eb58e492e8dc2d5dd
B1_RESOLVED_CONFIG_SHA256: 49d2ceb0d6a0ae4283c3459805267689d36f811be727cebf38b54d999e50b4b6
BOUND: <=18 attempts and exactly 1 accepted update per cell
RESOURCE: 1 GH200 / 8 CPU / 96 GiB / 00:30:00 / no requeue
ELAPSED: 00:03:56
ARTIFACT_MANIFEST_SHA256: 36b9cbf1eab30f54799cf7abbe83056ac009b301a7817d604a0c8b9abea5fb2f
```

## Interpretation boundary

The requests answer only whether bounded accepted optimizer windows exist for the
declared routes on one exact mini fixture, with exact scaler/finiteness/counter
accounting. They do not authorize or establish convergence, capability,
performance, sampling quality, full data, mAP/NDS, Protocol A/B, attack, defense,
or a new run.
