# S10 RUN_REQUEST — approved B4-based ABC aggregate completion envelope

## 1. Current state

```text
SESSION_ID: S00-S10-STARTUP
REQUEST_ID: S10-ABC-COMPLETION-v1-B4-estimate
REQUEST_STATE: O-127 APPROVED / A3 replacement implementation+tuple pending freeze
SUPERSEDES: S10-ABC-COMPLETION-v0-estimate — REJECTED by O-123
PLAN_AUTHORITY: O-122 scientific envelope + O-123 B4 minimum
EXECUTION_AUTHORITY: one explicit-one-GH200/CUDA-hidden STOP-A replacement under O-127
SOURCE_SHA: corrected science 7c01cc3f1e75691339f41f101794945748f03305; resource-attestation commit pending
BRANCH: codex/s10-cl-model-recipe
OWNER_APPROVAL: O-127 replacement approved 2026-07-16
EXECUTABLE_NOW: only after exact O-127 immutable source/snapshot/command/output is appended and committed
```

The v0 B=1-based estimate (`20–24` expected / `34` hard ceiling) is explicitly
rejected and cannot be revived. O-124 approves the v1 aggregate resource and
derivation envelope. This approval is not itself a job tuple: each material job
must still bind the exact immutable source/snapshot/config/data/cells/command/
resources/output here before S00 submits it.

## 2. GPU-hour estimate

GPU-hours mean elapsed allocation of one GH200, not claimed device utilization.
The accepted execution anchor is S09's checkpoint-off B4 F-U capacity cell:
20/23 accepted/attempted updates, `8.451 samples/s`, and `16.32/38.81 GiB` peak
allocated/reserved memory. That measured rate corresponds to `0.924611 h` for
28,130 samples, a 34.4% epoch-time reduction relative to B1. For budget stress,
v1 applies a 15% throughput haircut and uses `7.2 samples/s` (`1.08526 h` per
full-train equivalent epoch). B1 is not an epoch/rung baseline.

| STOP | Expected elapsed allocation | Proposed hard ceiling | Basis |
|---|---:|---:|---|
| A — split/evaluator | 0 GPU-h when a compatible CPU compute node is available; otherwise about 0.5 | 2.1 GH200-h after O-125 | metadata/MILP/checker/devkit parity are CPU work; O-125 added one contingency hour but authorized only one 15-minute A3 allocation |
| B — observation-first | about 0.5–1 GH200-h | 2 GH200-h | B4 main panels/replays, one tiny paired B4-vs-four-B1 aggregation check, and at most one local boundary refinement |
| C — architecture/init | about 12–16 GH200-h | 24 GH200-h | every scientific training cell is B4; six low/three mid slot stress case, up to two donor lineages, reference-graph penalty, eval and bounded step-debug allowance included |
| **ABC total** | **about 13–18 GH200-h** | **28.1 GH200-h after O-125** | serial one-GPU execution; O-125's increment was STOP-A-only contingency |

The lower C expectation assumes only one staged donor survives. Two incompatible
donor lineages (for example local and coherent-reference SECOND graphs) push the
expected total toward the upper end. An existing checkpoint lowers cost only if
its source/license, exact graph/tensor contract, data ownership, training recipe
and file digest are independently compatible; no reuse is assumed.

STOP-C's reproducible budget model is:

- `D_low` and `D_mid` use nominal 21% and 42% of the accepted 28,130-sample train
  cache. At the haircut B4 rate they cost about `0.228` and `0.456` GH200-h per
  epoch; the allowed sample-count upper bounds cost `0.261` and `0.499` h;
- up to six one-epoch `D_low` slots contribute at most about 1.57 h before
  reference-graph and evaluation overhead;
- up to three three-epoch `D_mid` slots contribute at most about 4.49 h at the
  haircut B4 rate;
- each staged LiDAR donor trains only on the same `D_mid` role, never
  `D_select/D_audit/official val`, and is budgeted for at most ten epochs (about
  4.56 h nominal / 4.99 h at the sample-count upper bound) for a three-epoch
  fusion rung; at most two incompatible donor lineages are included;
- the coherent-reference fusion path is stress-budgeted at up to `1.6x` current
  F-U elapsed time; 1–2 h covers subset evaluation, checkpointing and phase-runner
  overhead; and one cumulative hour is reserved inside C's ceiling for the
  bounded correctness-step policy.

All C training uses physical B4 with `drop_last=true`; exact dropped tokens are
recorded and held identical across matched candidates. A mandatory candidate does
not fall back to B1. These are deliberately conservative sizing assumptions, not
measured STOP-C performance. O-124 approves these execution horizons and the hard
ceiling, including allowed sample-count upper bounds and at most two qualifying
debug/fix cycles.

The 28.1-hour amended ceiling is a safety ceiling, not a spending target. Job
`467862` used only 0.253889 of O-125's contingency; the unused balance is not an
execution entitlement and cannot be transferred to another candidate, seed,
rung, longer horizon, STOP-D/E/F, DDP, array, profiler campaign or full run.

## 3. Proposed allocation map

```text
GPU: at most one GH200
CONCURRENCY: one active Slurm job total
DDP/ARRAY/SPARE_GPU: forbidden
SEED: one declared STOP-C seed
TRAINING_MICROBATCH: physical B4 minimum; B1 diagnostic decomposition only
TAIL_POLICY_FOR_C_PROXY_RUNGS: drop_last=true; exact dropped-token manifest
STOP-A/A-GATE: one combined split/evaluator allocation, preferably CPU-only
STOP-B/B-DIAG: one main diagnostic allocation
STOP-B/B-REFINE: zero or one, only for an already localized adjacent boundary
STOP-C/C-LOW: one allocation; predeclared candidates run serially under one source
STOP-C/C-DONOR: one allocation per promoted donor lineage, at most two
STOP-C/C-MID: one allocation; promoted candidates run serially under one source
EXPECTED_ALLOCATIONS: five with one donor/no B-REFINE; six with two donors or B-REFINE
SCIENTIFIC_ALLOCATION_MAX: seven (two donors plus B-REFINE)
DEBUG/FIX_ALLOCATION_MAX: two, only after a diagnosed obvious correctness failure
ABSOLUTE_SUBMISSION_MAX: nine
IDENTICAL_RETRY: forbidden
```

The earlier phrase “6–7 grouped submissions” meant the named phase allocations
above, not six hidden matrices or concurrent jobs. A phase allocation serially
executes several already-declared cells under one immutable source/config family,
emitting a separate status/checksum for each; there is no per-task, per-loss-term,
per-sample, per-checkpoint or array chain.

The two debug/fix slots are aggregate maxima, not guaranteed retries. On an
obvious correctness failure, the long cell stops. The same deterministic prefix
is probed at `1 -> 5 -> 20` accepted B4 steps; only the diagnosed implementation,
test, runner, provenance or output-neutral diagnostic defect may be fixed. The
candidate must then pass a fresh B4 G20 before its scientific cell restarts from
initialization. Each cycle has a new immutable snapshot/command/output and a
recorded diagnosis. A repeated blocker, identical rerun, scientific
underperformance, `INCONCLUSIVE` diagnosis, or an LR/epoch/candidate/seed change
returns to the owner. This is a proposed ABC rule, not O-107.

S09's current-graph B4 G20 is reused and not rerun as a generic preflight. Only a
new graph/init family needs its own G20. B=8/16 are intentionally excluded from
this request; they may enter a later STOP-D/E geometric batch ladder after final
graph selection.

## 4. Exact bindings required before each future submission

The following are intentionally pending until implementation and local validation:

```text
SOURCE_SHA:
SNAPSHOT_PATH_AND_SHA256:
RESOLVED_CONFIG_SHA256:
DATASET_CACHE_AND_ZIP_MANIFEST_IDENTITIES:
SPLIT/EVALUATOR/CANDIDATE_MANIFEST_SHA256:
CELLS_AND_ORDER:
SEED:
COMMAND/JOB_SCRIPT/SUBMIT_SHA256:
CPU/MEMORY/WALLTIME:
OUTPUT_ROOT:
STOP_CONDITIONS:
ALLOWED/FORBIDDEN_INTERPRETATION:
DERIVATION_FROM_AGGREGATE_AUTHORITY:
```

Exact tuples are appended here before submission. Under O-124, S00 may submit a
tuple without a second owner question only when it is a mechanical derivation
within every recorded bound.

## 5. Fail-closed boundaries

- STOP-A: any input-identity mismatch, solver infeasibility/integrality failure, support failure,
  ownership leak, unsupported devkit semantic or exact full-val parity mismatch;
- STOP-B: diagnostics are not output/gradient/state neutral, aggregate loss cannot
  be reconstructed, or the one bounded refinement remains inconclusive (record
  `INCONCLUSIVE`, do not expand instrumentation);
- STOP-C: a mandatory candidate cannot sustain B4 after its allowed
  output-neutral activation-checkpoint choice and one bounded correctness cycle;
  nonfinite state, no accepted updates, target/modality collapse, invalid
  evaluator, dropped-token/exposure mismatch, candidate/seed/rung cap pressure or
  an attempt to tune recipe inside C;
- aggregate: 27 elapsed GH200-hours, nine submissions, two qualifying debug/fix
  cycles, repeated blocker, changed scientific boundary, or uncertain
  classification.

Any boundary stops autonomous execution and returns to the owner. D/E/F, full
trainval training, official-val selection, additional seed, merge, push, upload,
publication, Protocol A/B execution, attack, defense and S11+ are outside this
request.

## 6. STOP-A / A-GATE exact immutable tuple — approved under O-124

```text
TUPLE_STATE: CONSUMED / Job 463593 FAILED before real split execution
DERIVATION: mechanical A-GATE allocation from O-124 v1 aggregate envelope
SOURCE_SHA: e27053a5b141e1afaa68363ce6deb2efdb60518e
SOURCE_TREE: dea3c8845657aadfd1edb300a10b8952db529761
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141
SNAPSHOT_MODE: detached / clean / read-only
SNAPSHOT_TRACKED_FILES: 595
SNAPSHOT_LS_TREE_SHA256: db157cedaf4efb5fdd6530d8b64754671f8025601f593ca993ad3aa377799d48
RUNNER: fl_v3/scripts/run_s10_stop_a_gate.sh
RUNNER_SHA256: 3a34b3686e9fbbe518831740bde35959feeaae34505cb65bd91e7df6082d1c22
GATE_SCRIPT: fl_v3/scripts/s10_stop_a_gate.py
GATE_SCRIPT_SHA256: b52b13efb923c0154104e9ff8286be46fc173faf7b915a6ec3a51cc128cdf5df
RESOLVED_CONFIG_SHA256: N/A — metadata/evaluator gate; no model/config/training path
RESOURCE: one node / one nvidia_gh200_120gb / 16 CPU / 96 GiB / 01:00:00
ACCOUNT/PARTITION: naiss2025-22-1113-gpu / gpu
CONCURRENCY/REQUEUE/ARRAY/DDP: one active / no-requeue / none / none
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_e27053a5b141_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_e27053a5b141_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_e27053a5b141_%j.err
SCIENTIFIC_ALLOCATION_COUNT_AFTER_SUBMIT: 1 / 7
DEBUG_FIX_ALLOCATION_COUNT_AFTER_SUBMIT: 0 / 2
ABSOLUTE_SUBMISSION_COUNT_AFTER_SUBMIT: 1 / 9
```

Exact inputs are the accepted S09 train/val `t1.v2`, `n_sweeps=10` cache tuple
and accepted ZIP-manifest tuple in §3 of `HANDOFF.md`; their logical and physical
SHA-256 values are literal runner arguments. The devkit config is bound to
`217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b`.
The source contains no STOP-A model or optimizer configuration.

Execution order is fixed:

1. exact source/tree/snapshot/runner/runtime/data preflight;
2. focused tests for the new split/checker/evaluator, existing official evaluator,
   GTDB role binding and CBGS identity seam;
3. one train/val cache materialization, no-seed MILP solve and independent emitted-
   artifact reconstruction/leakage checker;
4. unchanged official full-val evaluator versus the new full-val-manifest path for
   `P-GT` and `P-MIX`, tolerance zero;
5. all-empty adapter and real zero-point/bicycle-rack adversarial checks;
6. strict JSON/checksum finalization. No sensor payload, model, optimizer or
   training step is executed.

The canonical sorted compact JSON submission envelope is:

```json
{"account":"naiss2025-22-1113-gpu","chdir":"/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141","cpus_per_task":16,"export":{"S10_STOPA_EXPECTED_RUNNER_SHA256":"3a34b3686e9fbbe518831740bde35959feeaae34505cb65bd91e7df6082d1c22","S10_STOPA_EXPECTED_SOURCE_SHA":"e27053a5b141e1afaa68363ce6deb2efdb60518e","S10_STOPA_EXPECTED_TREE":"dea3c8845657aadfd1edb300a10b8952db529761","S10_STOPA_OUTPUT":"/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_e27053a5b141_a1","S10_STOPA_SNAPSHOT":"/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141"},"gpus":"nvidia_gh200_120gb:1","job_name":"s10-stop-a-e27053a","mem":"96G","no_requeue":true,"nodes":1,"ntasks":1,"partition":"gpu","runner":"/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141/fl_v3/scripts/run_s10_stop_a_gate.sh","stderr":"/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_e27053a5b141_%j.err","stdout":"/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_e27053a5b141_%j.out","time":"01:00:00"}
```

Its SHA-256 is
`f5c94a5cc80a697e3fb952865321b98db0c6b1d3670f196ca464cd292afd8e40`.
The exact submission command is:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=01:00:00 --no-requeue --job-name=s10-stop-a-e27053a --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_e27053a5b141_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_e27053a5b141_%j.err --export=ALL,S10_STOPA_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141,S10_STOPA_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_e27053a5b141_a1,S10_STOPA_EXPECTED_SOURCE_SHA=e27053a5b141e1afaa68363ce6deb2efdb60518e,S10_STOPA_EXPECTED_TREE=dea3c8845657aadfd1edb300a10b8952db529761,S10_STOPA_EXPECTED_RUNNER_SHA256=3a34b3686e9fbbe518831740bde35959feeaae34505cb65bd91e7df6082d1c22 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_e27053a5b141/fl_v3/scripts/run_s10_stop_a_gate.sh
```

PASS requires `COMPLETED 0:0`, zero restarts, focused tests passing, both solver
stages `OPTIMAL`, all ownership overlaps zero, exact `P-GT/P-MIX` equality for
filtered identities/40 metric-data records/validity/finite aggregates, and the
strict empty/adversarial gates. Any identity mismatch, infeasible/non-optimal
solve, leakage, parity mismatch, unsupported devkit semantic, timeout or nonzero
status is terminal STOP-A failure. There is no identical retry. A diagnosed
runner/test/output-neutral plumbing defect may consume an O-124 debug/fix slot;
split/metric/data/scientific changes return to the owner. Results cannot be read
as model capability, convergence, recipe or official-val selection evidence.

## 7. A-GATE Job 463593 negative result and bounded remediation classification

Job `463593` consumed the §6 tuple once and terminated `FAILED 1:0` after
`00:00:49` on `n184`, with zero restarts. It passed immutable source/runtime/data-
module preflight, then the focused test suite returned 1 before the gate script
started: 11 passed, 8 dependency/data skips, and one failure. No accepted train/
val cache was unpickled, no real MILP was solved, no split/ownership artifact or
full-val fixture was created, and no model/payload/training path ran.

The failure is exact and mechanical: SciPy 1.13.1/aarch64 constructed CSC index
buffers as platform `long`, while its bundled HiGHS wrapper requires C `int` and
raised `ValueError: Buffer dtype mismatch, expected 'int' but got 'long'` before
optimization. The mathematical matrix coefficients, bounds, integrality,
lexicographic objectives and checker were not reached. This is classified as an
O-124 solver-call/test-plumbing defect, not an infeasible split or scientific
result. The only allowed remediation is to materialize CSC row/column/index-
pointer buffers as `int32`, assert that dtype at the boundary, and disable
pytest's cache provider on the deliberately read-only snapshot. No constraint,
feature, threshold, role, data identity, evaluator semantic or resource changes.

```text
JOB: 463593
STATE/EXIT/ELAPSED/RESTARTS: FAILED / 1:0 / 00:00:49 / 0
CONSUMED_GPU_HOURS: 0.013611
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_e27053a5b141_a1
EXECUTION_IDENTITY_SHA256: 7e7546391f10d41016cafe6d5cb437e3ccf847dc72faeab9d830f86a5254a37a
FOCUSED_TEST_STDOUT_SHA256: 3fdff6ebbaa00cf4bbea4531b5dd478fe30553418b2b63bf3c2b3764dd12e916
FOCUSED_TEST_STDERR_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
RUNNER_ARTIFACT_MANIFEST_SHA256: 2bd4982f0f7f78a0ad854f00f4e1ee82c703b118d615e44d5fcb26958666ab7a
SCIENTIFIC_ALLOCATIONS_CONSUMED: 1 / 7
DEBUG_FIX_ALLOCATIONS_CONSUMED: 0 / 2
SUBMISSIONS_CONSUMED: 1 / 9
```

O-124 permits one fresh derived debug/fix allocation after the remediation is
committed, snapshotted and frozen below. It is not an identical retry; Job
`463593` remains the required negative result. The replacement consumes debug/fix
slot 1/2 and submission 2/9, while the scientific-allocation count remains 1/7.

## 8. STOP-A derived debug/fix tuple 1 — consumed under O-124

```text
TUPLE_STATE: CONSUMED / Job 463649 TIMEOUT / NOT EXECUTABLE
DERIVATION: exact §6 tuple plus only the §7 int32/cache-provider remediation
SOURCE_SHA: 3f7ab76f7043384705b109e40fd4c1d1fcde01ae
SOURCE_TREE: 32e2648defbb9a84763b1d993668139f78a2e0d4
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix1_3f7ab76f7043
SNAPSHOT_MODE: detached / clean / read-only
SNAPSHOT_TRACKED_FILES: 596
SNAPSHOT_LS_TREE_SHA256: 3d96dcf9f8c0da62907e87952b2c666465ed39590c0a8ba0f396e9f0c0362ce6
RUNNER_SHA256: 1c8761d4a749b573e4d9ff01975127bcc986441d676819b2a78e215ca56f9940
GATE_SCRIPT_SHA256: b52b13efb923c0154104e9ff8286be46fc173faf7b915a6ec3a51cc128cdf5df
DATA/TESTS/CELLS/ORDER/GATES: identical to §6
RESOLVED_CONFIG_SHA256: N/A — no model/config/training path
RESOURCE: one node / one nvidia_gh200_120gb / 16 CPU / 96 GiB / 01:00:00
ACCOUNT/PARTITION: naiss2025-22-1113-gpu / gpu
CONCURRENCY/REQUEUE/ARRAY/DDP: one active / no-requeue / none / none
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_3f7ab76f7043_a2
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix1_3f7ab76f7043_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix1_3f7ab76f7043_%j.err
SCIENTIFIC_ALLOCATIONS_CONSUMED_AFTER_SUBMIT: 1 / 7
DEBUG_FIX_ALLOCATIONS_CONSUMED_AFTER_SUBMIT: 1 / 2
ABSOLUTE_SUBMISSIONS_CONSUMED_AFTER_SUBMIT: 2 / 9
```

Canonical sorted compact JSON submission-envelope SHA-256:
`9e5cc9a9860250d843344a8da58d41a45840f633e339371f3bf924b455980e77`.
The exact command is:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=01:00:00 --no-requeue --job-name=s10-stop-a-fix1-3f7ab76 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix1_3f7ab76f7043 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix1_3f7ab76f7043_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix1_3f7ab76f7043_%j.err --export=ALL,S10_STOPA_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix1_3f7ab76f7043,S10_STOPA_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_3f7ab76f7043_a2,S10_STOPA_EXPECTED_SOURCE_SHA=3f7ab76f7043384705b109e40fd4c1d1fcde01ae,S10_STOPA_EXPECTED_TREE=32e2648defbb9a84763b1d993668139f78a2e0d4,S10_STOPA_EXPECTED_RUNNER_SHA256=1c8761d4a749b573e4d9ff01975127bcc986441d676819b2a78e215ca56f9940 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix1_3f7ab76f7043/fl_v3/scripts/run_s10_stop_a_gate.sh
```

All §6 PASS/fail-closed and interpretation limits remain unchanged. If the same
dtype blocker recurs, this is a repeated blocker and autonomous execution stops.
Any subsequent failure is classified from new evidence; it does not inherit
permission for another change merely because one debug/fix slot remains.

## 9. A-GATE Job 463649 timeout and remediation boundary

Job `463649` consumed the frozen §8 tuple once and reached Slurm `TIMEOUT` after
`01:00:14` on `n130`, with zero restarts. Runtime/source/data preflight passed and
the focused suite completed `12 passed, 8 skipped in 2.64s`. The accepted train
metadata traversal reached all 28,130 samples in about 34 seconds, after which
the process remained inside the exact split solve until the wall-time kill. It
never emitted a split manifest, ownership record, solver report or evaluator
parity artifact. This is not evidence of infeasibility and is not a STOP-A PASS.

The timeout exposes two implementation/plumbing defects while preserving the
scientific boundary:

1. the accepted five-level lexicographic objectives are followed by 50 base and
   34 nested one-log tie-break solves, so SciPy rebuilds and cold-solves 94 MILPs;
2. the runner had only an `EXIT` trap, so the external timeout was correctly
   visible in Slurm but incorrectly sealed `final.exit=0` without a `gate.exit`.

The only remediation candidate is output-equivalent: keep every feature, hard
constraint, integer-ppm objective, role code and sorted-log lexicographic order,
but encode at most ten consecutive ternary assignment digits per exact block and
fix blocks sequentially. A block has maximum coefficient `3^9=19683` and maximum
objective `3^10-1=59048`; both are exactly represented below `2^53`. This reduces
the topology from 94 to 19 cold solves without using a global large-weight
objective, relaxing the MIP gap or changing the selected assignment. The runner
must also fail closed on `TERM/INT/HUP` and must never write `final.exit=0` unless
the gate returned zero. Local legacy-versus-block equivalence, numerical-bound,
solve-count and signal-finalization tests are required before a new tuple exists.

```text
JOB: 463649
STATE/EXIT/ELAPSED/RESTARTS: TIMEOUT / 0:0 (batch CANCELLED 0:15) / 01:00:14 / 0
CONSUMED_GPU_HOURS: 1.003889
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_3f7ab76f7043_a2
EXECUTION_IDENTITY_SHA256: 242313c8801f1a8b9bf23fa3a2f6e3b98fab563df979db3db0ba381ded1e570e
FOCUSED_TEST_STDOUT_SHA256: e8b1b083b8a8411d51362caf9ae04213cddf7d39c751671267e6013d9eed1626
GATE_STDERR_SHA256: b1aa13e85ac29f0a0769e49298f2ea229310865e412da336c15bda0c87b7a6f9
FINAL_EXIT_SHA256: 9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
RUNNER_ARTIFACT_MANIFEST_SHA256: 02e9773d224db2858c393c78df8a600bd363229926b209c20f2f643e3bcc4ab6
SCIENTIFIC_ALLOCATIONS_CONSUMED: 1 / 7
DEBUG_FIX_ALLOCATIONS_CONSUMED: 1 / 2
SUBMISSIONS_CONSUMED: 2 / 9
STOP_A_GPU_HOURS_CONSUMED: 1.017500 / 1.000000 hard ceiling (scheduler grace included)
ABC_GPU_HOURS_CONSUMED: 1.017500 / 27.000000
```

The remaining debug/fix slot does not override STOP-A's approved cumulative
one-GH200-hour ceiling. S00 may implement, test, commit, snapshot and freeze the
strictly equivalent remediation, but another GH200 submission requires an owner
resource amendment. No B/C execution starts before STOP-A closes.

## 10. STOP-A derived debug/fix tuple 2 — consumed under O-125

The output-equivalent remediation is immutable at
`d7caf53414ade2d5db794ecd90851d0e5a3535b5`. On the CPU SciPy diagnostic module,
the pre-change and post-change synthetic split/report canonical payloads have the
same SHA-256,
`e0396504e68729d34e16012e03ff8e99fb41b4ed652cd65e03cd5fab6be6ab56`.
Both runs emit the same base vector, `D_low`, `D_mid`, 55 base objectives and 39
nested objectives. The new implementation repeats identically, performs exactly
19 MILP calls, and proves the block objective is at most 59,048. `bash -n`, Python
bytecode compilation and `git diff --check` pass. Signal-lifecycle probes return
`TERM=143`, incomplete-zero `125`, and complete-zero `0`.

```text
TUPLE_STATE: CONSUMED / TIMEOUT / NOT EXECUTABLE
DERIVATION: §8 tuple plus only exact blocked-radix tie-break and fail-closed signal handling
SOURCE_SHA: d7caf53414ade2d5db794ecd90851d0e5a3535b5
SOURCE_TREE: d4e25ae7ce074ef0b9b0350b329ccaf806756f77
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix2_d7caf53414ad
SNAPSHOT_MODE: detached / clean / read-only
SNAPSHOT_TRACKED_FILES: 596
SNAPSHOT_LS_TREE_SHA256: 265d2f94defbecaafd2b3337eb963c4b46fb3071f61dbb663910c1e075ed845f
RUNNER_SHA256: 637cea4b1400629c38e355ae686289709c8ba3b929cbf5ae445a8bba165ef119
GATE_SCRIPT_SHA256: b52b13efb923c0154104e9ff8286be46fc173faf7b915a6ec3a51cc128cdf5df
INTERNAL_SPLIT_SHA256: 0ba005b157cc3246f6de3c2d94366d3962d306c1b89928b520bfdda133b3ad4b
SPLIT_TEST_SHA256: b2c4400c843c2b7c7a2725a5b654cf4ed21e7e60ae6c7347067f293247d57ec7
DATA/CELLS/ORDER/GATES: identical to §6 and §8
TEST_SELECTORS: identical file/selectors to §8; one in-file radix/call-count property test added
RESOLVED_CONFIG_SHA256: N/A — no model/config/training path
RESOURCE: one node / one nvidia_gh200_120gb / 16 CPU / 96 GiB / 00:15:00
ACCOUNT/PARTITION: naiss2025-22-1113-gpu / gpu
CONCURRENCY/REQUEUE/ARRAY/DDP: one active / no-requeue / none / none
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_d7caf53414ad_a3
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix2_d7caf53414ad_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix2_d7caf53414ad_%j.err
SCIENTIFIC_ALLOCATIONS_AFTER_SUBMIT: 1 / 7
DEBUG_FIX_ALLOCATIONS_AFTER_SUBMIT: 2 / 2
SUBMISSIONS_AFTER_SUBMIT: 3 / 9
```

Canonical sorted compact JSON submission-envelope SHA-256:
`24de0be54806fbd1270bec2f560451ee62a138a593a5cb0a542f0a7c76d7f061`.
The exact O-125 command is:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=00:15:00 --no-requeue --job-name=s10-stop-a-fix2-d7caf53 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix2_d7caf53414ad --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix2_d7caf53414ad_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_fix2_d7caf53414ad_%j.err --export=ALL,S10_STOPA_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix2_d7caf53414ad,S10_STOPA_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_d7caf53414ad_a3,S10_STOPA_EXPECTED_SOURCE_SHA=d7caf53414ade2d5db794ecd90851d0e5a3535b5,S10_STOPA_EXPECTED_TREE=d4e25ae7ce074ef0b9b0350b329ccaf806756f77,S10_STOPA_EXPECTED_RUNNER_SHA256=637cea4b1400629c38e355ae686289709c8ba3b929cbf5ae445a8bba165ef119 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_fix2_d7caf53414ad/fl_v3/scripts/run_s10_stop_a_gate.sh
```

Owner decision O-125:

```text
ADDITIONAL_STOP_A_AUTHORITY: exactly the frozen 15-minute tuple above, executable once
ADDITIONAL_NOMINAL_GPU_HOURS: 1.0 contingency; this tuple requests at most 0.25
STOP_A_CUMULATIVE_CEILING: 2.1 elapsed GH200-hours
ABC_CUMULATIVE_CEILING: 28.1 elapsed GH200-hours
OTHER_STOP/CELL/SEED/HORIZON/RESOURCE BOUNDS: unchanged
ON_ANY_FAILURE_OR_TIMEOUT: STOP-A blocked; no further fix/retry; return to owner
```

The exact job asks for 15 minutes rather than reserving the one-hour contingency.
The unused 45 minutes are neither spent nor an automatic identical retry or new
submission authority. Any later tuple still requires a fresh diagnosed boundary.

## 11. A-GATE Job 467862 terminal timeout

Job `467862` consumed the exact §10 tuple once on `n409` with zero restarts.
Source/runtime/data preflight passed. The expanded focused suite completed
`13 passed, 8 skipped in 1.93s`, and the accepted train metadata traversal reached
all 28,130 samples in about 33 seconds. The process then remained inside the exact
blocked-radix MILP until Slurm terminated it at `00:15:14`. The requested
`00:15:00` limit was therefore sufficient to prove that a 10-minute request would
also have failed, but it was insufficient to complete the exact split.

No `gate.exit`, `final.exit`, solver report, split manifest, ownership record or
evaluator-parity artifact exists. The gate created the target directory and an
empty `split/` subdirectory before solving, but wrote no files there; partial
fail-closed evidence remains under the sibling `.control` path. Neither directory
is a reusable split. This is not split infeasibility evidence and is not a STOP-A
PASS. The observed 19-call topology did not provide enough wall-time reduction
on the real problem; without per-solve output-neutral telemetry, the completed
call count and slowest objective remain unknown.

```text
JOB: 467862
STATE/EXIT/ELAPSED/RESTARTS: TIMEOUT / 0:0 / 00:15:14 / 0
BATCH_STEP: CANCELLED / 0:9 / 00:17:13 after timeout cleanup
NODE: n409
CONSUMED_GPU_HOURS: 0.253889
TOTAL_CPU/MAX_RSS: 16:28.628 / 11553920K
PARTIAL_CONTROL: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_d7caf53414ad_a3.control
TARGET_OUTPUT: exists with empty split/ directory and zero files; not reusable
EXECUTION_IDENTITY_SHA256: b149c366cbd1427e0c2d2a2e51af4782bca9376641c55d3147e5e9cc33b2566a
FOCUSED_TEST_STDOUT_SHA256: 607134486ddc03317432193fee030862b17d7d03fbefd1a50b267857e0d969bb
FOCUSED_TEST_STDERR_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
FOCUSED_TEST_EXIT_SHA256: 9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa
GATE_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
GATE_STDERR_SHA256: 3534797b3098669b99befbe99286eac3bcd4ebb5c501fd18c62aa3515e24b63b
PARTIAL_CONTROL_TREE_SHA256: d10b25999440e86f278e7dfeb13a0ddfe114b8fc347584691d37793138666f4e
SLURM_LOG_PAIR_SHA256: 709bb321bdf2050cae15f7266088dec7e39f0bf312e974ada0beef21887f4fea
SCIENTIFIC_ALLOCATIONS_CONSUMED: 1 / 7
DEBUG_FIX_ALLOCATIONS_CONSUMED: 2 / 2
SUBMISSIONS_CONSUMED: 3 / 9
STOP_A_GPU_HOURS_CONSUMED: 1.271389 / 2.100000 amended ceiling
ABC_GPU_HOURS_CONSUMED: 1.271389 / 28.100000 amended ceiling
```

O-125 requires any timeout to return to the owner with no identical retry. STOP-A
is therefore blocked; no B/C job may start because the accepted stop order makes
STOP-A's immutable split/evaluator identity their prerequisite. Any future tuple
requires an explicit owner amendment after deciding whether to retain exact MILP,
change its solver/algorithm, or revise the gate.

## 12. O-126 corrected one-shot feasibility amendment — approved, tuple pending

O-126 approves the following replacement for the exhausted O-125 optimizer
protocol. It does not reinterpret Jobs `463593`, `463649` or `467862`; those
remain immutable negative evidence.

```text
PROTOCOL: first feasible one-shot frozen
REAL_CANDIDATES: exactly one
BASE_SOLVES: one constant-zero feasibility MILP
NESTED_SOLVES: one constant-zero feasibility MILP
PRE_SOLVE_FREEZE: exact source/data identities + ordered log_features SHA-256 + candidate ordinal 1
REROLL/CANDIDATE_SHOPPING: forbidden
HARD_GATES_RETAINED: all location/sample/support/prevalence/dominance/ownership/leakage/evaluator gates
OPTIMIZATION_CLAIM: none
SCIENTIFIC_SCOPE: reusable limited-rung proxy split; not official benchmark or balance optimum
PARTITION: gpu (aarch64/runtime compatibility only)
GPU_GRES: omitted / zero
CPUS_PER_TASK: 4
MEMORY: 32 GiB
WALLTIME: 00:15:00
SUBMISSIONS: exactly one
REQUEUE/RETRY/REROLL/ARRAY/DDP: forbidden
ON_FAILURE_OR_TIMEOUT: return to owner; do not start B/C
ON_PASS: immutable evidence commit, then independent high-risk A4 review
```

A2 has now filled the exact `SOURCE_SHA`, source tree, detached read-only
snapshot, tracked-tree identity, runner/gate/script hashes, fresh absent output
path and literal `sbatch` command in §13. The command omits `--gpus` entirely and
the runner fails closed unless `SLURM_JOB_GPUS` is empty,
`SLURM_GPUS_ON_NODE` is zero/absent, `CUDA_VISIBLE_DEVICES` is empty and PyTorch
exposes zero CUDA devices.

## 13. O-126 corrected A-GATE exact immutable tuple — frozen for sole submission

```text
TUPLE_STATE: CONSUMED / Job 468295 scheduler-resource mismatch / cancelled before gate execution
SOURCE_SHA: 7c01cc3f1e75691339f41f101794945748f03305
SOURCE_TREE: 93a0ac39b49df51f0f75a08e73d1f12268be50db
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_feasible_7c01cc3f1e75
SNAPSHOT_MODE: detached / clean / read-only / zero writable paths
SNAPSHOT_TRACKED_FILES: 596
SNAPSHOT_LS_TREE_SHA256: e1d74fac65136daeef84e47ef6a6b7499a77b3e51c839b8ad784c7534b7fc020
RUNNER_SHA256: a36b201497c2451752e436930acef6939a9b430373b2d80692746d120df41cd1
GATE_SCRIPT_SHA256: b52b13efb923c0154104e9ff8286be46fc173faf7b915a6ec3a51cc128cdf5df
MATERIALIZER_SHA256: 08e5fb7c33b577ef9fc8a572065ba4251920ba0cafbb0408c2c1c4ee7cadf4b0
INTERNAL_SPLIT_SHA256: bf3af675ed56755a220abf76bce551f503279f2f63c8dab7f2bbd827167119cb
SUBSET_EVALUATOR_SHA256: bbf2b5ebbf1f04671295038cf9214976f5c17564ecef43ba819eeb00f385bc26
INTERNAL_SPLIT_TEST_SHA256: c268f9b7a028fddb04c8d363a3e4a8aa9056cbeb8a53cfa2c4fb58d84f28d013
SUBSET_EVALUATOR_TEST_SHA256: e64b73e668faf70acaa934b3f2500cf1c0b80454cee1154c814047ab3b050caa
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: 4c91291b62dce4cb8e74d8a106ea9337f0f41ddb39204333506c908d32c21d71
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_7c01cc3f1e75_o126_a1
OUTPUT_PREFLIGHT: target and sibling .control absent
DATA: exact S09 train/val t1.v2 and S01 ZIP-manifest identities frozen in §6
MODEL/CONFIG/TRAINING: N/A
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU_GRES: omitted (zero GPU)
CPU/MEMORY/TIME: 4 / 32 GiB / 00:15:00
CONCURRENCY_PREFLIGHT: squeue --me empty
REQUEUE/RETRY/REROLL/ARRAY/DDP: forbidden
STOP: any runner/test/data/solve/checker/leakage/parity/artifact failure or timeout
INTERPRETATION: constrained limited-rung proxy split/evaluator gate only
```

A2 local checks at this exact source: `py_compile`, `bash -n`,
`git diff --check`, and a supplemental x86 SciPy `1.16.1` synthetic solve all
passed. The synthetic probe observed exactly two zero objectives, role counts
`34/8/8`, nested counts `10/20`, and independent checker `PASS`. Full focused
pytest is intentionally deferred to this aarch64 job because the login-node
module stack lacks `nuscenes-devkit`; the runner executes the complete frozen
selector set before touching real split output.

Exact command (there is deliberately no `--gpus` option):

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:15:00 --no-requeue --job-name=s10-stop-a-o126-7c01cc3 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_feasible_7c01cc3f1e75 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_o126_7c01cc3f1e75_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_o126_7c01cc3f1e75_%j.err --export=ALL,S10_STOPA_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_feasible_7c01cc3f1e75,S10_STOPA_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_7c01cc3f1e75_o126_a1,S10_STOPA_EXPECTED_SOURCE_SHA=7c01cc3f1e75691339f41f101794945748f03305,S10_STOPA_EXPECTED_TREE=93a0ac39b49df51f0f75a08e73d1f12268be50db,S10_STOPA_EXPECTED_RUNNER_SHA256=a36b201497c2451752e436930acef6939a9b430373b2d80692746d120df41cd1 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_feasible_7c01cc3f1e75/fl_v3/scripts/run_s10_stop_a_gate.sh
```

## 14. O-126 A3 Job 468295 — terminal scheduler-resource FAIL

The §13 command was submitted exactly once. `sbatch` immediately reported:
`A number of GPUs not specified. Allocating 4 per node for a total of 4 GPUs.`
The site `job_submit/lua` plugin therefore changed the approved zero-GPU request;
`scontrol show job -dd` confirmed `ReqTRES=cpu=4,mem=32G,node=1,gres/gpu=4` and
`AllocTRES=cpu=4,mem=32G,node=1,gres/gpu=4` on aarch64 node `n428`. S00
protection-cancelled the job rather than allow unauthorized four-GPU allocation.

```text
JOB: 468295
STATE/EXIT/ELAPSED: CANCELLED by 4004328 / 0:0 / 00:00:08
START/END: 2026-07-16T05:45:36 / 2026-07-16T05:45:44
NODE: n428 / aarch64
REQ_AND_ALLOC_GPU: 4 x nvidia_gh200_120gb
ALLOCATED_GPU_HOURS: 4 * 8 / 3600 = 0.008889
CONSUMED_ENERGY_RAW: 0
EXECUTION_IDENTITY/TEST/DATA/SOLVE/PARITY: not emitted/not entered
TARGET_OUTPUT_AND_CONTROL: both absent
STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
STDERR_SHA256: 061f652d362d5123daba8deaf776a2bd01ffc4ba12fa4c4eff0582e68a3cedb4
O126_SUBMISSIONS: 1 / 1 consumed
REAL_SPLIT_CANDIDATES: 0 / 1 consumed
CUMULATIVE_STOP_A_ALLOCATED_GPU_HOURS: 1.280278
```

The current partition has `JobSubmitPlugins=lua`; the `gpu` partition contains
only aarch64 GH200 nodes and the `cpu`/`fat` partitions are separate x86 nodes.
Two non-submitting `sbatch --test-only` diagnostics (Slurm documents that this
validates without creating a job) showed `--gpus=0` and `--gres=none` are also
rewritten to four GPUs. They are not replacement commands. There is no automatic
retry. The narrow viable route is an owner-amended one-GPU allocation with CUDA
hidden from the CPU-only process, or an externally provided aarch64 CPU queue;
neither is authorized here.

## 15. O-127 one-GH200-reserved / CPU-only-process replacement — approved

Owner amendment O-127 authorizes exactly one replacement of the scheduler-
rejected §13 tuple:

```text
SCIENCE/DATA/EVALUATOR/CANDIDATE: unchanged from O-126
ALLOCATION_GPU: exactly 1 x nvidia_gh200_120gb
PROCESS_CUDA_VISIBLE_DEVICES: empty
PROCESS_TORCH_CUDA_AVAILABLE/COUNT: false / 0
CPU/MEMORY/TIME: 4 / 32 GiB / 00:15:00
MAX_NEW_ALLOCATED_GPU_HOURS: 0.25
REAL_CANDIDATE_ORDINAL: 1, still unconsumed
SUBMISSIONS: exactly one replacement
RETRY/REROLL/ARRAY/DDP: forbidden
ON_FAILURE_OR_TIMEOUT: return to owner
ON_PASS: seal evidence, then independent high-risk A4 review
B/C: forbidden until reviewed STOP-A PASS
```

The minimal runner change may only attest the scheduler allocation and hide CUDA;
it may not alter features, MILP constraints, checker, ownership, evaluator or
artifact semantics. The exact immutable resource-attestation source, detached
read-only snapshot, file/tree hashes, fresh output and literal command must be
appended and committed before submission.
