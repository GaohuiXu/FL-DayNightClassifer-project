# S10 results — STOP-A/B closed; STOP-C0 incomplete at no-retry boundary

## STOP-C0

```text
AUTHORITY: O-131
STATE: FAIL/INCOMPLETE; sole allocation consumed; no retry authorized
JOB: 492525 / FAILED 1:0 / 00:47:32 / 0.792222 GH200-hours
EXECUTION_SOURCE: 89958be504d6abaef66810695402d2a09619794b
CELLS: C0-F-A1 one D_low epoch + D_select eval; C0-L-A0 one D_low epoch + D_select eval; C0-F-A0-P64 no eval
RESOURCE: one GH200 / 16 CPU / 96 GiB / 01:00:00 / one submission / no retry
CLAIM_LIMIT: training-health/trajectory and descriptive one-epoch fusion delta only; no recipe, architecture, official-val or full claim
```

### Execution integrity and terminal failure

Job `492525` consumed the exact frozen §26 command/snapshot. Its focused suite
passed `74` tests with `3` skips in `16.51 s`; source/config/data/weight/runtime
identities passed. `C0-F-A1` and `C0-L-A0` each completed all 1,538 attempted
B4 windows, wrote a checkpoint and sampled diagnostics, and completed the exact
4,626-sample `D_select` evaluation. The third cell reached its post-training
contract check after the declared 64-window horizon, but the runner incorrectly
required every cell to exhaust the 1,538-batch epoch iterator. It raised:

```text
RuntimeError: D_low one-epoch cell did not consume the exact drop-last loader
```

Consequently `C0-F-A0-P64` has only its resolved config; no diagnostics,
`cell_summary.json`, or accepted scratch result exists. The aggregate
`c0_summary.json` is also absent. This is a runner-control defect, not a model
failure, but it makes the exact C0 protocol incomplete and the job correctly
terminal `FAILED 1:0`. O-131 forbids retry.

The two complete cells also received raw mechanical `HARD_FAIL` labels for
`lidar_encoder.to_bev` lacking sampled gradients. In the exact SECOND-075 graph,
`collapsed_channels == out_channels`, so `to_bev` is deliberately
`nn.Identity` and has no trainable parameter. This required-prefix condition is
therefore an impossible gate and a false positive. The raw labels are retained;
they are not rewritten as PASS. Post-job remediation replaces that condition
with the trainable `lidar_encoder.backbone.conv_out`, makes exact epoch
exhaustion conditional on a full-epoch cell, and adds focused regression tests.
Those fixes are model/loss/gradient/update-neutral, but they deliberately change
the diagnostic artifact schema and health semantics from v1 to v2. They were not
executed on GH200 and do not repair or replace the immutable raw v1 artifacts.

Independent review also invalidated the raw v1 `dropped_tokens` identities. The
runner predicted a `randperm` from a fresh seed-0 generator, whereas the real
DataLoader consumes a `_base_seed` from that generator before its RandomSampler
draw. The remainder **count** of three is correct, and F/L used the same real
loader construction and therefore the same actual order/remainder, but the three
named raw tokens and their diagnostics-fixture identity are not the exact dropped
tokens. Exact remainder-token identity is unavailable for this executed job. The
v2 remediation records tokens from actual collated batches and computes the
remainder by set difference only for a completed full epoch.

### Complete-cell numerical and internal-evaluator evidence

| cell | attempted / accepted updates | invalid placement | first → final chunk loss | raw label | harm label | internal mAP / NDS |
|---|---:|---|---:|---|---|---:|
| `C0-F-A1` | 1,538 / 1,534 | 4 in first 64; 0 later | 143.6704 → 17.8161 | `HARD_FAIL` from impossible Identity-prefix gate | `NOT_ESTABLISHED` | 0.064773 / 0.148719 |
| `C0-L-A0` | 1,538 / 1,534 | 4 in first 64; 0 later | 121.2631 → 19.2639 | same false-positive gate | `NOT_ESTABLISHED` | 0.020728 / 0.111537 |

Both cells have zero discarded windows, zero nonfinite-loss windows, zero
post-first-64 invalid windows, complete exposure-count accounting, and finite
token-complete evaluator output. Their GradScaler backed off from `512` to `32`
during the four initial overflow windows and then accepted every remaining
window. If the impossible Identity-prefix requirement is excluded, no other
predeclared hard error remains and both trajectories satisfy the bounded
`NUMERICALLY_HEALTHY_WITH_TRAINING_SIGNAL` rule. That is a defect-audited
interpretation of the retained raw evidence, not a replacement raw label or C0
PASS.

The one-epoch descriptive fusion-minus-LiDAR delta is:

```text
internal D_select mAP: +0.044045020516
internal D_select NDS: +0.037182017642
```

This comparison uses one seed and the same split, B4 exposure, precision,
baseline optimizer and evaluator, but F also has the declared ImageNet1K V1
camera prior. It is evidence of an early positive current-family fusion/camera
signal only. It does not select the graph or recipe and is not the STOP-F fusion
claim.

At sampled attempts `1` and `4`, true optimizer-unscaled SECOND stem gradients
were extremely large and both sampled windows overflowed: F approximately
`5.70e6/7.78e6`, L approximately `1.78e6/2.19e6`. At accepted sampled attempts
`16/64/256/768/1538`, stem gradients remained much larger than most other
prefixes but were finite. Maximum realized LiDAR update/weight was only
`6.295e-4` for F and `7.689e-4` for L, versus the predeclared extreme-update
threshold `1e-2`. The raw v1 fields labelled “median” used the upper middle value
for an even-length vector (`3.100e-4` and `3.043e-4`); the standard medians are
`3.09258750525629e-4` and `3.0088933646399e-4`. This correction does not change
the harm classification because both maximum LiDAR ratios are independently far
below `1e-2`.
Neither cell had a post-warm-up invalid window or adverse loss trajectory, so
all three harm indicators are false. C0 therefore does **not** establish that
the large gradients harm the applied optimization, but it also does not locate
their cause; normalization/target/loss/occupancy/head-to-stem causality remains
unresolved.

### Bounded timing, memory and profile observations

- F steady post-first-chunk throughput was `8.497–8.614 samples/s`; its first
  chunk was `2.084 samples/s` because it contains profiler/scaler-start overhead.
  L steady throughput was `13.057–13.149 samples/s` (`9.838` first chunk).
- CUDA peak allocated/reserved was F `17,698,519,040 / 44,434,456,576` bytes
  and L `8,070,803,968 / 9,732,882,432` bytes. The 1 Hz device maximum was
  `43,306 MiB`; Slurm batch MaxRSS was `82,455,693 KiB`.
- Full `D_select` evaluator wall time was F `679.370 s` and L `571.809 s`;
  decode portions were `415.275 s` and `277.876 s`. Large prediction JSON and
  CPU metric aggregation are material wall-time costs and must not be confused
  with training GPU throughput.
- Across the 2,776 recorded in-job 1 Hz samples, spanning tests, training and
  CPU-only evaluator aggregation but excluding unrecorded startup/teardown,
  GPU utilization was mean/p50/p95 `38.89/37/100%`; power was
  `199.05/197.75/272.59 W`. This mixed-phase aggregate is not a steady-training
  utilization claim.
- The single early F trace covers ten active windows only. Module range device
  time per window was approximately camera preprocess `105.36 ms`, LiDAR
  encoder `59.21 ms`, camera backbone `42.42 ms`, shared head `22.66 ms`, view
  transform `7.39 ms`, camera neck `3.73 ms`, fuser `2.09 ms`, and BEV neck
  `1.90 ms`. Top operator self-device rows include GroupNorm backward
  `113.95 ms/window`, its 1D gamma/beta backward kernel `88.42 ms/window`, and
  sparse implicit-GEMM backward `40.58 ms/window`. This supports GroupNorm as a
  performance investigation target, not as the proven gradient cause or final
  STOP-E bottleneck.

### Artifacts, checksums and allocation

```text
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_89958be_o131_a1
RUNNER_MANIFEST_SHA256: 950a79919dbf07b1dab54f4ff91c4bf9c49692bf05417c532725dd508c93397e
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
ARTIFACT_MANIFEST_CHECK: 25/25 OK
OUTPUT_SIZE: approximately 2.0 GiB
```

Actual C0 allocation is `2,852 / 3,600 = 0.792222` GH200-hours. Cumulative
STOP-A/B/C allocation is `1.685556 + 0.792222 = 2.477778` GH200-hours, leaving
`24.522222` under the active 27-hour O-124 ceiling. Unused C0 walltime is not
retry or downstream execution authority. Later C cells and STOP-D/E/F remain
owner-gated.

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
CORRECTED_A1-A4: completed under O-126/O-127 plus A4 review
CORRECTED_IMPLEMENTATION_SHA: 7c01cc3f1e75691339f41f101794945748f03305
JOB: 468295 / CANCELLED by owner 0:0 / 00:00:08 / site transformed 0 GPU to 4 GPU
CORRECTED_A3: not executed; O-126 submission consumed at scheduler boundary
O127_REPLACEMENT_SHA: ad93c89333b0a8f19abf138c8d6816e742b51e35
O127_REPLACEMENT_TUPLE: consumed once by Job 468404
JOB: 468404 / COMPLETED 0:0 / 00:07:59 / zero restarts
A3_RESULT: PASS — one-shot split, ownership checker and evaluator parity accepted for review
INDEPENDENT_REVIEW: PASS_WITH_RESIDUAL_RISK at b0478a2 / no open P0-P3
STOP_A_FINAL: CLOSED PASS — constrained split/evaluator engineering gate only
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

Initial A4 independently reviewed evidence SHA
`2a0153be88311ce1f8d502f2593218494d579014` in a detached clean worktree. It
found no data/split/evaluator P0/P1 defect and independently reproduced the
checksum, ownership, parity, resource and negative-history evidence, but returned
`REMEDIATE` on two documentation findings:

- P2: `RUN_REQUEST.md` exposed both 27 and 28.1 hours as possible active ABC
  aggregate ceilings. The remediation keeps the binding `AGENTS.md`/O-124
  27-hour ceiling as the unique fail-closed aggregate. O-125 remains a
  STOP-A-local, non-transferable contingency and does not broaden B/C authority.
- P3: the phrase “role-bound CBGS seam” was stronger than the current helper,
  which hashes caller-provided tokens and indices but does not verify an expected
  manifest SHA or role. The remediation calls it a CBGS identity seam and makes
  manifest/role/expected-token verification a STOP-D hard gate before CBGS can be
  enabled.

Neither finding changes Job `468404`, the immutable split, ownership, evaluator,
model math or compute. STOP-A remains open pending targeted review of the
documentation remediation; no rerun is needed or authorized.

Targeted re-review of remediation SHA
`b0478a298a0a3b5e538bedcca63e2541d71c2146` confirmed the isolated worktree was
detached and clean, the remediation touched only four S10 Markdown files, and
both P2/P3 were closed. Final verdict is `PASS_WITH_RESIDUAL_RISK` with no open
P0-P3. The two residual risks are bounded and recorded in `REVIEW.md`: the full
40×101 arrays are asserted at runtime and represented by hashes rather than
separate persisted array files, and the bicycle-rack count is trainval-wide
rather than val-only. Neither changes exact parity or ownership evidence.

STOP-A is therefore CLOSED PASS for its exact interpretation: one immutable,
log-owned, leakage-checked limited-rung proxy split and an internal evaluator
that is tolerance-zero identical to the unchanged official path on both frozen
full-val fixtures. It is not a model, recipe, convergence, fusion-gain or
official-val capability PASS.

## STOP-B

```text
OWNER_DECISION: O-128 + O-129 parity remediation/replacement
IMPLEMENTATION_SHA: 8fd832dc7d46e8818216ecbcf228ef8fd0590ecb
IMPLEMENTATION_TREE: d5ce6c060279271295abdca41c3ad7aec5870315
B_DIAG_TUPLE: consumed once by Job 477892
JOB: 477892 / FAILED 1:0 / 00:04:44 / zero restarts / 0.078889 GH200-hours
FOCUSED_TESTS: 39 passed in 13.36s
PRE_MODEL_PANEL: PASS / 48 core + 16 term / content SHA 8e4f2d9
W0: frozen all-scratch seed-0 state SHA e58bcd4
RESULT: first FP32 disabled/on parity combined gate FAIL
BROAD/TERM/FP16/OPTIMIZER/EVALUATOR: not executed
LOCALIZATION_VERDICT: none; neither LOCALIZED nor INCONCLUSIVE
B_REFINE: not triggered and not authorized
REPLACEMENT_IMPLEMENTATION_SHA: 43f157b3eca7ca72633358b5a2d2dbc4c4e4684b
REPLACEMENT_JOB: 478250 / FAILED 1:0 / 00:04:28 / zero restarts / 0.074444 GH200-hours
REPLACEMENT_TESTS: 41 passed in 12.40s
REPLACEMENT_RESULT: FP32 P_core disabled0/disabled1 baseline_instability
REPLACEMENT_LOCALIZATION_VERDICT: none; neither LOCALIZED nor INCONCLUSIVE
REPLACEMENT_B_REFINE: trigger false; forbidden
INDEPENDENT_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / owner rebaseline required
```

The combined parity gate checked exact output hash, raw parameter-gradient hash,
loss, RNG hash and model-state hash. The runner failed before persisting its
per-predicate parity record, so the raw evidence does not identify which one or
more predicates differed. Consequently this failure is an instrumentation/
parity-boundary failure, not evidence that the large LiDAR gradient was located
or that the current model is unhealthy. No optimizer was constructed, no update
was made, and `D_select`, `D_audit` and official val remained unobserved.

Immutable output root:
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1`.
Its runner artifact manifest SHA-256 is
`0dc23faf982a2905709f83b1cc2b0fde87d4850da7ad256a98dddd91acdec0a2`.

O-129's replacement reused that exact physical panel without reconstruction or
reroll, required the same W0 hash, performed one no-update FP32 warm-up, and
persisted disabled-0/disabled-1/enabled predicates before failing. Job `478250`
passed all 41 focused tests and every identity gate. W0 stayed exactly
`e58bcd46...`; RNG-state hashes were identical across all three parity runs.

Nevertheless, the repeated disabled path was not numerically repeatable on the
first `P_core` B4 batch. Output hashes differed and loss moved from
`391.5013732910156` to `388.7950134277344`. All 459 parameter gradients were
finite with identical empty missing-gradient sets, but 434 failed the fixed
allclose gate; global relative-L2 error was `3.5323887774502536` and max-absolute
error was `2422412.736328125`. The model state remained byte-identical to W0.
This satisfies §20's predeclared `baseline_instability` classification.

The enabled run also differed, but instrumentation neutrality cannot be inferred
in either direction because its disabled control already fails. The evidence
establishes a same-input/same-W0/same-framework-RNG numerical-repeatability
failure after one declared warm-up; it does not identify spconv, GroupNorm,
loss normalization or any other mechanism. It also does not locate the original
large LiDAR gradient, assess convergence, or authorize a model/recipe change.

The job stopped before every later parity/scientific cell, made zero updates and
ran no evaluator. B-REFINE is not triggered. All sealed checksums verify; the
runner manifest SHA-256 is
`801e98c129797a6a71665c5227cbe6684a4001b39d617da91bbb7970b92c3543`.
Actual STOP-B compute is `0.153333` GH200-hours and cumulative ABC compute is
`1.566667` GH200-hours. Independent review returned
`PASS_WITH_RESIDUAL_RISK` with no open P0-P3 and accepts only the bounded
disposition **calibrated baseline-instability FAIL; localization absent; owner
rebaseline required**.

### O-130 B-RAND result — integrity PASS / descriptive MIXED_INCONCLUSIVE

```text
OWNER_DECISION: O-130
IMPLEMENTATION_SHA: 0bf9c0ce4148bc82d977e0d66615f606144971b6
IMPLEMENTATION_TREE: 1852db34197c142714456f3fa07e999393dc1ba9
TUPLE: RUN_REQUEST.md §24, consumed exactly once
JOB: 479667 / COMPLETED 0:0 / 00:07:08 / zero restarts
ALLOCATION: 1 GH200 / 8 CPUs / 64 GiB / 0.118889 GH200-hours
FOCUSED_TESTS: 43 passed in 12.75s
RUNS/COMPARISONS: 33 physical-B4 forward/backward / 24 reference comparisons
INTEGRITY_GATE: PASS
DESCRIPTIVE_CLASSIFICATION: MIXED_INCONCLUSIVE
SUPPORTED_SIGNALS: CAMERA_STOCHASTICITY + LIDAR_RUNTIME_VARIATION
FUSION_ONLY_SIGNAL: loss only; fails two-of-three support rule
OPTIMIZER/UPDATES/EVALUATOR: absent / zero / absent
INDEPENDENT_REVIEW: pending exact evidence SHA
```

Job `479667` matched source/tree, detached read-only snapshot, all three resolved
configs, the accepted `D_low` split and exact Job-477892 first-`P_core` B4 token
vector. Each mode reproduced its independently constructed seed-0 W0; F-U
matched the previously accepted W0 SHA
`e58bcd46d588c68b31335fe87cc5fbff06cbc0fbcdae7e88b0b1ed70d1d65395`.
Every loss, output and parameter gradient was finite, the missing-gradient set
was stably empty, fixed-seed post-run RNG hashes were identical, and model state
remained W0. Exact hashes were retained as evidence but were not acceptance
gates. Both checksum manifests verify and the complete output is recursively
read-only.

The controlled camera result is exact:

| C-STR8 group | median loss rel. diff | median output rel-L2 | median gradient rel-L2 | output/gradient hashes |
|---|---:|---:|---:|---|
| fixed seed `10000` | `0.0` | `0.0` | `0.0` | one unique hash each |
| varying `11000..11004` | `0.0336344` | `0.2262902` | `0.1659221` | five unique hashes each |

The trainable Swin-T registry contains twelve active stochastic-depth modules
from probability `0.0` through `0.2`. The observed camera RNG-dependent
variation under changing seeds is consistent with that intended stochastic
graph, while same-seed C-STR8 is exactly repeatable. The run did not capture
individual stochastic-depth masks or execute an SD-disabled counterfactual, so
it does not prove that specific mechanism causally. The prior Job-478250
same-seed failure cannot be explained by generic training chaos or fixed-seed
camera RNG variation.

The LiDAR result separates a second source:

| mode/group | median loss rel. diff | median output rel-L2 / cosine | median gradient rel-L2 / cosine |
|---|---:|---:|---:|
| L-S075 fixed seed | `0.000156455` | `0.0342479 / 0.999414` | `0.657438 / 0.768679` |
| L-S075 varying seed | `0.000221258` | `0.0339449 / 0.999424` | `0.609070 / 0.821667` |
| F-U fixed seed | `0.00262014` | `0.0418436 / 0.999125` | `1.223123 / 0.273609` |
| F-U varying seed | `0.0132509` | `0.1127151 / 0.993646` | `1.154219 / 0.059822` |

L-S075 has no stochastic-depth modules, yet all five same-seed output and
gradient hashes differ. Its fixed- and varying-seed variation have comparable
scale, so framework RNG choice is not the dominant source. Prefix evidence
places the largest fixed-seed gradient-direction changes in the early sparse
SECOND path: L-S075 `stage1` median relative-L2/cosine is
`0.9641 / 0.5437` and `stem` is `0.6542 / 0.7709`; F-U `stage1` is
`1.2674 / 0.1653`, `stem` `1.2102 / 0.2804`, and `down1`
`1.1565 / 0.3046`. This is operational localization to the LiDAR sparse route,
not proof of a specific spconv kernel, layer or normalization defect.

The predeclared fourfold rule qualifies both `CAMERA_STOCHASTICITY` and
`LIDAR_RUNTIME_VARIATION` in loss/output/gradient, so the unique-label result is
honestly `MIXED_INCONCLUSIVE`. `FUSION_ONLY_INTERACTION` is supported by loss
only and does not qualify. The scientific interpretation remains bounded:
O-130 explains the sources of repeatability variation sufficiently to replace
byte-equality as the STOP-B gate, but it does not explain why the true unscaled
LiDAR gradients are large, establish causal architecture health, assess
convergence, or authorize a model/recipe change or STOP-C.

Immutable evidence:

```text
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_rand_0bf9c0c_o130_a1
SUMMARY_SHA256: dd51f5801084714fccbd0c351b0696c3a6a2843b462662c74f757fc12cd147c5
EXECUTION_IDENTITY_SHA256: a3d34532655d5c2faff68219b56b717a9a418f8da2879b54f76673fd1a1c1397
RUNS_JSONL_SHA256: ddef313efc036ca70c494d0fc717e71970cda3a070d821690561aa42a511ec94
COMPARISONS_JSONL_SHA256: d59a39971663e43da2b17104948cd9110334040dfd06e1e5eb9615ff98b98924
INNER_ARTIFACT_MANIFEST_SHA256: 8429dfe63f674215c1a3ca78ed11f30a44041d071d7477141424cb1d464db3a9
RUNNER_ARTIFACT_MANIFEST_SHA256: d964b7cc5fa09692a9b8bd95b83cf8cfed85768ff771eaf8cc2a9c8c3cb11ac0
STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
FINAL/TEST/OBSERVE_EXIT: 0 / 0 / 0
```

Actual STOP-B compute is now `0.153333 + 0.118889 = 0.272222`
GH200-hours. Actual cumulative ABC compute is
`1.566667 + 0.118889 = 1.685556` GH200-hours, leaving
`25.314444` hours under the active 27-hour aggregate. No unused O-130 time is a
retry or follow-up entitlement.

Independent review of exact evidence SHA `fdf223b` found no P0-P2 and two
documentation-only P3: lower canonical status lag and causal overstatement of
stochastic depth. Remediation `02ba3b44202092894f2c1c3e7ee53bb56ba92a1d`
updated only five Markdown files. Targeted re-review verified unchanged
implementation, tuple, raw hashes, classification and compute, closed both P3,
found no new findings and returned `PASS_WITH_RESIDUAL_RISK` with no open
P0-P3.

Final STOP-B disposition is **CLOSED / INCONCLUSIVE**. The route-level
repeatability decomposition is accepted; large-gradient causality remains
unresolved. No further STOP-B compute or model/recipe change is authorized, and
STOP-C is not started by this closure.
