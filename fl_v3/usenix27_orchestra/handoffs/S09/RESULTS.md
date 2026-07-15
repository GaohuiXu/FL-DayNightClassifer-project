# S09 results — terminal engineering PASS

## Owner disposition

O-120 accepts final review seal
`ced5992ea113bd21d7d545af505debf405b556b3` and closes S09
`PASS_WITH_RESIDUAL_RISK` with no open P0-P3. O-121 later fast-forward integrated
closing commit `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`.

## STOP-1 production cache identity

Job `441191` completed `0:0` in `00:03:06` from source
`1f276b9d2cc54f705b0b6800a573258707711045`. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop1_cache_t1v2_1f276b9d2cc5
```

| Split | Samples / boxes | Previous sweeps | Canonical SHA-256 | Pickle bytes / SHA-256 | Sidecar SHA-256 |
|---|---:|---:|---|---|---|
| train | 28130 / 944881 | 246840 | `310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a` | 580252836 / `57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30` | `f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c` |
| val | 6019 / 187528 | 52812 | `bb692de4c1eb8b66e8c74f4e807eb208ad891b45ce8f233e8017dc4f3a3b6e2f` | 118018654 / `d4ed7aee9978c2294e2087c917006cbb3d69276453266d0f9c92591340084837` | `4f5390815720e14625be31b20fb1596cafe9869ad95b08dc098aea65413be432` |

Both are `t1.v2`, `v1.0-trainval`, `n_sweeps=10`, bound to ZIP-manifest
logical/physical SHA-256
`023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6` /
`228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.
Every record was depth/content revalidated. The output checksum manifest is
`4f48ea4e7ebfc9427a4cf649e3b3826feb0b529f7a56af011b4e1b78a8f5f2ef`.
This proves cache identity, not payload decode, throughput or model readiness.

## STOP-2 readiness smoke

Job `441293` completed `0:0` in `00:01:04` from source
`37aef4d6b3f4679d6702d0acef2bb5bd1b57a952`: 44/44 focused tests passed,
including CUDA-event output neutrality and fail-closed lifecycle cases. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop2_smoke_37aef4d6b3f4_a1
artifact_sha256s.txt: 643160908f29f76cccbdcde3e5999759934aa5417b3d71252810e917ae4667ff
```

The implementation provides exact attempted/successful counters, direct timing,
memory, optional bounded loader profiling and terminal failure artifacts without
generic observers/module hooks. This smoke is code/runtime evidence only.

## STOP-3 negative runtime evidence

Job `441511` failed `1:0` in `00:02:29` before physical-data verification,
loader construction, model construction or any optimizer attempt. The source
selected runtime-only modules while editable spconv JIT needed build modules; its
first fatal error was missing `cublasLt.h`. The failed import also changed cumm
native artifacts. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_4d6bd8294500_a1
artifact_sha256s.txt: 0c3e2947fb124ac32d74e243575b4ffa159d2d97a6a603492196fd89df565133
```

This is a runner/bootstrap failure, not a loader/model/precision/performance result.
O-118 explicitly authorized dependency recovery and one strictly derived
replacement; it was not an automatic retry.

Job `442152` then reproduced stable two-process source/build identities:
spconv `af42200511a53ce86d77cea0306924a2dc516a74f0483ef7cfe0a6e1dc84b100`,
cumm `0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d`.
It ran no data/model/training path.

## STOP-3 production loader and G100

Job `446225` completed `0:0` in `00:05:05`. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop3_g100_c200bac861a4_a2
readiness.json: 08e376e767f654bb38982127ad5ffd84d94ebaa48b3026ceba2ab7ef93a6c9b6
artifact_sha256s.txt: b229633889052c46bec5c05d6713e0102aea806a98f9170a65119f9864dbea4b
```

The bounded ZIP/cache worker profile consumed 2432 samples per complete matrix
and reproduced one content identity across repeats:

| Workers | Cold samples/s | Warm samples/s | Warm wait p50 / p95 |
|---:|---:|---:|---:|
| 0 | 16.085 | 21.867 | 45.411 / 52.009 ms |
| 2 | 41.282 | 41.574 | 20.594 / 49.593 ms |
| 4 | 79.041 | 77.520 | 0.183 / 47.234 ms |
| 8 | 139.992 | 141.970 | 6.655 / 28.379 ms |

Workers 8 reached 100% of the best warm throughput for this bounded profile. This
does not establish a universal optimum.

The F-U B=1 lifecycle reached 100 accepted updates in 103 attempts. The first
three scaler windows backed off `512 -> 256 -> 128 -> 64`; all 90 measured
post-warm windows accepted at 64. Counters reconcile exactly: 103 attempted/loss
evaluated, 100 successful/optimizer/scheduler/exposure, 3 overflow/invalid, zero
direct-nonfinite/discarded/pending.

| Stage | Mean / p50 / p95 (ms) |
|---|---|
| H2D | 0.280 / 0.278 / 0.340 |
| Forward | 89.895 / 90.004 / 94.291 |
| Loss | 10.922 / 10.871 / 14.178 |
| Backward | 102.846 / 100.688 / 113.518 |
| Optimizer/scheduler/EMA | 6.023 / 6.011 / 6.163 |
| CUDA integrated | 210.600 / 208.576 / 224.153 |

Combined data-wait + CUDA p50/p95 is `208.746/224.327 ms`; throughput
`4.743 samples/s`; data wait `0.076%`; peak allocated/reserved
`3.256/6.434 GiB`; steady epoch estimate `1.647 h`. This is a throughput
estimate at the measured constant recipe, not convergence time.

## STOP-4A profile and capacity

Job `452520` completed `0:0` in `00:09:42`; 59 focused tests and all four
20-update cells passed. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4a_profile_capacity_b509f5e527c2_a1
artifact_sha256s.txt: fbd07beebcd9078c5a980995e05febc1efc873469d0ff4fe61f30c6748f5272f
```

| Cell | Accepted / attempted | p50 / p95 | Throughput | Peak allocated / reserved / headroom |
|---|---:|---:|---:|---:|
| B1 checkpoint-on + profiler | 20/23 | 281.692 / 316.432 ms | 3.507 samples/s | 3.06 / 5.38 / 89.63 GiB |
| B1 checkpoint-off | 20/23 | 186.627 / 204.558 ms | 5.364 samples/s | 4.55 / 7.24 / 87.76 GiB |
| B2 checkpoint-off | 20/22 | 293.437 / 337.788 ms | 6.925 samples/s | 8.55 / 13.34 / 81.66 GiB |
| B4 checkpoint-off | 20/23 | 475.311 / 542.830 ms | 8.451 samples/s | 16.32 / 38.81 / 56.19 GiB |

The trace/source audit proved 19 redundant loss-term scalar synchronizations per
ordinary attempted window. They were removed only when neither runtime telemetry
nor S08 diagnostics asks for the terms, with exact loss/input-gradient equality
tests. No second safely removable allocation was proven, so none was changed.

B2/B4 fit and expose more parallelism; they do not select a recipe. The current
full-epoch B4 path also has a two-sample tail (`28130 % 4 == 2`) while the loop
requires fixed batch shape. That is a future full-run design item, not a failure
of this bounded 20-step capacity cell.

## STOP-4C optimized G100

Job `455539` completed `0:0` in `00:04:06`, reaching 100/103 accepted/
attempted. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4c_g100_c7769901201b_a1
readiness.json: b8765c4be656fe7ad657157cc43c2c6915ebfc33e6411c26c2a7db829087adff
artifact_sha256s.txt: 542862b20a86d30c348237a9b448610857f86cb7554473cfbe65150360593847
```

Relative to STOP-3, explicit Swin checkpoint-off plus quiet loss telemetry changed
combined p50/p95 to `183.215/217.674 ms`, throughput to `5.237 samples/s`,
epoch estimate to `1.492 h`, and peak allocated/reserved to
`4.764/8.361 GiB`. Mean forward/backward became `90.299/80.003 ms`.
The combined speedup cannot be separately attributed to each optimization.

## STOP-4D optimized G1000

Job `456539` completed `0:0` in `00:06:54` from a fresh seed-0
initialization. Output:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop4d_g1000_5642884cdbb1_a1
readiness.json: e61c1f6e6761a74b787dcdf9303fd1911868e44ebf1e5195765a2214396968b8
artifact_sha256s.txt: 6b90ae38427bb6efaa043f1a7c93432473cade8862ea5d1f9e432e87003107b3
```

It reached 1000 accepted updates in 1003 attempts. After the same three initial
overflows, all 990 measured windows accepted at scale 64. Accounting is exact:
1003 attempted/loss-evaluated, 1000 successful/optimizer/scheduler/exposure, three
overflow/invalid, zero direct-nonfinite/discarded/pending. Aggregate horizon loss
is finite but is not a convergence gate.

| Metric | Result |
|---|---:|
| combined p50 / p95 | 178.024 / 203.231 ms |
| accepted throughput | 5.542 samples/s |
| steady wall/update over 990 windows | 180.425 ms |
| steady epoch estimate | 1.409821 h |
| data-wait share | 0.096934% |
| peak allocated / reserved / headroom | 4.765 / 8.314 / 86.686 GiB |
| p95/p50 | 1.141594 |

| Stage | Mean / p50 / p95 (ms) |
|---|---|
| H2D | 0.274 / 0.267 / 0.344 |
| Forward | 86.904 / 86.730 / 91.957 |
| Loss | 11.451 / 11.405 / 14.715 |
| Backward | 74.820 / 72.237 / 93.902 |
| Optimizer/scheduler/EMA | 6.049 / 6.017 / 6.361 |
| CUDA integrated | 180.157 / 177.805 / 203.070 |

One-Hz GPU utilization mean/p50/p95/max is `47.56/51/74.1/100%`; only
`2.51%` of samples are at least 80%. Memory use p50 is about 9.43 GiB.
Therefore the current B=1 tuple does not fully utilize GH200. Negligible data wait
makes the loader an unlikely steady bottleneck, but 1-Hz/stage timing cannot name
a specific kernel or branch.

## Final interpretation

S09 establishes exact production cache identity, bounded loader readiness,
100/1000-update lifecycle health, direct stage timing, memory capacity and two
output-neutral speed improvements. It does not establish convergence,
generalization, recipe/batch selection, model quality, mAP/NDS, fusion gain,
multi-seed capability, full GH200 utilization, Protocol A/B, FL, attack or defense.

The stable optimizer windows do not explain or reduce the large true SECOND
gradients; dynamic scaling only protects scaled arithmetic. That root-cause work
belongs to the accepted S10 work definition, whose exact envelope remains
unapproved.
