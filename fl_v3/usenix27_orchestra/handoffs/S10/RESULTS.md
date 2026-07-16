# S10 results — STOP-A A3 PASS pending A4; STOP-B/C unstarted

## STOP-A

```text
IMPLEMENTATION_SHA: e27053a5b141e1afaa68363ce6deb2efdb60518e
REMEDIATION_SHA: d7caf53414ade2d5db794ecd90851d0e5a3535b5
A_GATE_TUPLES: recorded in RUN_REQUEST.md
JOB: 463593 / FAILED 1:0 / 00:00:49 / zero restarts
JOB: 463649 / TIMEOUT 0:0 / 01:00:14 / zero restarts
JOB: 467862 / TIMEOUT 0:0 / 00:15:14 / zero restarts
LEGACY_RESULT: no accepted split/parity; exact split solve exceeded all frozen walltimes
LEGACY_A3: consumed under O-125; no retry or reinterpretation
CORRECTED_A1-A4: approved under O-126; evidence pending
CORRECTED_IMPLEMENTATION_SHA: 7c01cc3f1e75691339f41f101794945748f03305
JOB: 468295 / CANCELLED by owner 0:0 / 00:00:08 / site transformed 0 GPU to 4 GPU
CORRECTED_A3: not executed; O-126 submission consumed at scheduler boundary
O127_REPLACEMENT_SHA: ad93c89333b0a8f19abf138c8d6816e742b51e35
O127_REPLACEMENT_TUPLE: consumed once by Job 468404
JOB: 468404 / COMPLETED 0:0 / 00:07:59 / zero restarts
A3_RESULT: PASS — one-shot split, ownership checker and evaluator parity accepted for review
INDEPENDENT_REVIEW: A4 pending exact immutable evidence SHA
```

Job `463593` passed source/runtime preflight, then failed one focused test because
SciPy 1.13.1/aarch64 handed HiGHS `long` sparse-index buffers where its Cython
wrapper requires `int`. Eleven tests passed and eight were skipped; no accepted
cache was loaded and no real MILP, ownership/evaluator gate, sensor payload,
model or training step ran. The immutable output manifest SHA-256 is
`2bd4982f0f7f78a0ad854f00f4e1ee82c703b118d615e44d5fcb26958666ab7a`.
This is retained negative plumbing evidence, not evidence that the split is
infeasible.

The strictly derived replacement at source
`3f7ab76f7043384705b109e40fd4c1d1fcde01ae` fixed the aarch64 sparse-index
boundary: its focused suite passed and full train metadata traversal completed.
It then remained inside the exact split solve until Slurm terminated it at the
one-hour limit. The implementation performs 94 cold MILP calls because the
sorted assignment vector is fixed one log at a time. No solver report, manifest,
ownership artifact, evaluator parity, model, recipe or performance result is
accepted. Its runner artifact manifest SHA-256 is
`02e9773d224db2858c393c78df8a600bd363229926b209c20f2f643e3bcc4ab6`.
The runner's `final.exit=0` is invalidated by authoritative Slurm `TIMEOUT` and
the missing `gate.exit`; this fail-open artifact state is itself retained as
negative evidence.

The exact blocked-radix implementation and signal-safe runner remediation are
immutable at `d7caf53414ade2d5db794ecd90851d0e5a3535b5`. Legacy/new synthetic
canonical output identity, repeated determinism, 19-call topology, radix bounds,
static checks and signal lifecycle pass locally. Job `467862` then exercised its
exact O-125 tuple: the focused suite passed (`13 passed, 8 skipped in 1.93s`) and
all 28,130 train metadata samples were traversed in about 33 seconds, proving the
data path was not the long pole. The exact MILP still had not completed at the
15-minute wall-time. No `gate.exit`, false-success `final.exit`, split, ownership,
solver-report or parity artifact exists.

Job `467862` consumed `0.253889` GH200-hours, bringing cumulative STOP-A/ABC
allocation to `1.271389` GH200-hours. Partial-control tree SHA-256 is
`d10b25999440e86f278e7dfeb13a0ddfe114b8fc347584691d37793138666f4e`;
execution identity is
`b149c366cbd1427e0c2d2a2e51af4782bca9376641c55d3147e5e9cc33b2566a`;
gate stderr is
`3534797b3098669b99befbe99286eac3bcd4ebb5c501fd18c62aa3515e24b63b`.
O-125 forbade an identical retry after timeout, so STOP-A returned to the owner
at that boundary; STOP-B and STOP-C did not start.

O-126 supplies that explicit amendment: the hard scientific constraints,
ownership/leakage proof and exact evaluator-parity gate remain unchanged, while
the unnecessary global balance/lexicographic optimizer is replaced by base and
nested one-shot zero-objective feasibility solves. The feature table and ordinal
one are frozen before solving; no alternative candidate or reroll is allowed.
No corrected A-GATE result is recorded. The sole authorized allocation was
frozen as CPU-only (`0 GPU`, 4 CPU, 32 GiB, 15 minutes), but was not executed
with those resources.

The exact frozen command was submitted as Job `468295`, but Arrhenius
`job_submit/lua` injected four GH200s when no GPU count was specified. `scontrol`
confirmed four requested and allocated GPUs on `n428`; S00 protection-cancelled
the job after eight seconds before execution identity/tests/data/split, so the target output
and `.control` are both absent and the real candidate count remains zero.
Allocated exposure was `0.008889` GPU-hours; cumulative STOP-A allocation is
`1.280278` GPU-hours. stdout was empty
(`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`);
stderr SHA-256 is
`061f652d362d5123daba8deaf776a2bd01ffc4ba12fa4c4eff0582e68a3cedb4`.
Non-submitting `--test-only` checks proved `--gpus=0` and `--gres=none` are also
site-defaulted to four GPUs. O-126 permits no replacement submission, so STOP-A
is resource-blocked pending owner amendment and A4 cannot start.

O-127 supplied that narrow resource amendment. Job `468404` reserved exactly one
GH200 to enter the compatible aarch64 runtime but forced
`CUDA_VISIBLE_DEVICES=""`; execution identity reports PyTorch CUDA unavailable
with device count zero, and Slurm accounting reports `gres/gpumem=0` and
`gres/gpuutil=0`. The job completed `0:0` in 479 seconds with zero restarts,
using `0.133056` allocated GPU-hours. Cumulative STOP-A/ABC allocation is now
`1.413334` GPU-hours.

The focused suite passed (`13 passed, 8 skipped`). The single predeclared real
candidate used 50 train logs and exactly two constant-zero feasibility solves;
both base and nested reports are `FEASIBLE_FROZEN`. There was no seed, second
candidate or reroll. The resulting immutable split is:

| role | logs | scenes | samples |
|---|---:|---:|---:|
| `D_fit` | 34 | 494 | 19,877 |
| `D_select` | 8 | 115 | 4,626 |
| `D_audit` | 8 | 91 | 3,627 |
| `D_low` | 10 | 153 | 6,155 |
| `D_mid` | 20 | 290 | 11,661 |

Post-job independent reload of the emitted source checker returns PASS. The
34,149-row ownership ledger covers all 28,130 train and 6,019 official-val
samples, with zero cross-owner overlap for log, scene, sample, annotation,
instance or raw sensor path. All declared location, sample-volume, support,
prevalence and dominance constraints pass. `candidate_freeze.json` remains
absent and cannot be created before STOP-D.

Evaluator validation also passes. `P-GT` and `P-MIX` each show
`EXACT_PARITY`, tolerance zero, between the unchanged official full-val path and
the internal-manifest path for filtered identities, metric-data arrays, validity
masks and finite aggregates. The explicit empty adapter returns exact zero
mAP/NDS. The top-level gate is PASS at SHA-256
`ed168363a072ef25f808e789a973127fa6fbd9d592c6077cc726e539cab161f`;
the split manifest is
`7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8`,
the leakage report is
`91b956f82e9771a64205cbc0501d819eafda29d9de6e5c882e9b37eb872aa4ad`,
and the sealed runner manifest is
`cf7957fbe9e83a6b0b023882f53fdd86901ed7ca258cfa9cf886f12ef8b80697`.
All nested checksum manifests verify and the output tree is read-only.

This is an A3 engineering PASS, not yet STOP-A closure. It establishes one
scientifically constrained, reusable limited-rung proxy split plus an exact
evaluator path. It does not establish balance optimality, model capability,
convergence, recipe quality or official-val performance. A4 independent high-
risk review must accept the exact evidence commit before B/C can start.
