# S10 RUN_REQUEST — approved B4-based ABC aggregate completion envelope

## 1. Current state

```text
SESSION_ID: S00-S10-STARTUP
REQUEST_ID: S10-ABC-COMPLETION-v1-B4-estimate
REQUEST_STATE: O-142 exact replacement tuple frozen / sole submission executable
SUPERSEDES: S10-ABC-COMPLETION-v0-estimate — REJECTED by O-123
PLAN_AUTHORITY: O-122 scientific envelope + O-141 BN1d-B8 candidate + O-142 exact replacement
EXECUTION_AUTHORITY: one exact O-142 replacement after immutable tuple freeze; no retry
SOURCE_SHA: 864f704f5bdf1a63db8aba342778d6bf6d36fe57
BRANCH: codex/s10-cl-model-recipe
OWNER_APPROVAL: O-142 exact schema-access fix and unchanged replacement approved 2026-07-19
EXECUTABLE_NOW: exact §37 tuple only; one submission/no retry
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
| **ABC total** | **about 13–18 GH200-h** | **27 GH200-h active aggregate** | binding O-124/`AGENTS.md` fail-closed ceiling; STOP-A contingency does not expand or transfer into the aggregate |

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

The unique active aggregate ceiling is `27` elapsed one-GH200 hours under binding
`AGENTS.md` and O-124. O-125 supplied a STOP-A-only contingency and raised that
stop's local ceiling to `2.1` hours; it did not expand the active ABC aggregate or
create transferable B/C budget. Every STOP-A allocation, including the O-126
scheduler mismatch and O-127 replacement, still counts against the 27-hour
aggregate. Stop-specific ceilings and the aggregate apply together; the most
restrictive reached boundary wins. Job `467862` used only 0.253889 of O-125's
contingency, but the unused balance is not an execution entitlement and cannot be
transferred to another candidate, seed, rung, longer horizon, STOP-D/E/F, DDP,
array, profiler campaign or full run.

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
HISTORICAL_O125_REQUEST_RECORD_ABC_CEILING: 28.1 elapsed GH200-hours; non-operative after A4 conflict audit
ACTIVE_FAIL_CLOSED_ABC_AGGREGATE_CEILING: 27 elapsed GH200-hours under binding AGENTS.md/O-124
OTHER_STOP/CELL/SEED/HORIZON/RESOURCE BOUNDS: unchanged
ON_ANY_FAILURE_OR_TIMEOUT: STOP-A blocked; no further fix/retry; return to owner
```

The historical O-125 request record arithmetically added the STOP-A contingency
to the 27-hour envelope and called the result 28.1 hours. A4 found that this
conflicted with binding `AGENTS.md`, O-124 and the fail-closed aggregate gate in
§5. The stricter `27`-hour aggregate is therefore the only active execution
boundary; this clarification narrows rather than expands authority. The exact job
asked for 15 minutes rather than reserving the one-hour contingency. The unused
45 minutes are neither spent nor an automatic identical retry or new submission
authority. Any later tuple still requires a fresh diagnosed boundary.

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
ABC_GPU_HOURS_CONSUMED: 1.271389 / 27.000000 active aggregate ceiling
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

The minimal runner change only attests the scheduler allocation and hides CUDA;
it does not alter features, MILP constraints, checker, ownership, evaluator or
artifact semantics. The exact immutable resource-attestation source, detached
read-only snapshot, file/tree hashes, fresh output and literal command are now
frozen in §16 and must be committed before submission.

## 16. O-127 exact immutable replacement tuple — frozen for sole submission

```text
TUPLE_STATE: CONSUMED / Job 468404 COMPLETED 0:0 / not executable
SOURCE_SHA: ad93c89333b0a8f19abf138c8d6816e742b51e35
SOURCE_TREE: d9715477cc6d5e0b6bf9d35c7005222d0d4f63c3
SCIENCE_PARENT: 7c01cc3f1e75691339f41f101794945748f03305
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_o127_ad93c89333b0
SNAPSHOT_MODE: detached / clean / read-only / zero writable paths
SNAPSHOT_TRACKED_FILES: 596
SNAPSHOT_LS_TREE_SHA256: 0695aebb4623ac931c0c297b9d83df9a5084b9e947ba22d27c183a39e0d39e4e
RUNNER_SHA256: 94c52a7ceecadec166786990980801d05a153b2d8a710dd597af5a8da973225a
GATE_SCRIPT_SHA256: b52b13efb923c0154104e9ff8286be46fc173faf7b915a6ec3a51cc128cdf5df
MATERIALIZER_SHA256: 08e5fb7c33b577ef9fc8a572065ba4251920ba0cafbb0408c2c1c4ee7cadf4b0
INTERNAL_SPLIT_SHA256: bf3af675ed56755a220abf76bce551f503279f2f63c8dab7f2bbd827167119cb
SUBSET_EVALUATOR_SHA256: bbf2b5ebbf1f04671295038cf9214976f5c17564ecef43ba819eeb00f385bc26
INTERNAL_SPLIT_TEST_SHA256: c268f9b7a028fddb04c8d363a3e4a8aa9056cbeb8a53cfa2c4fb58d84f28d013
SUBSET_EVALUATOR_TEST_SHA256: e64b73e668faf70acaa934b3f2500cf1c0b80454cee1154c814047ab3b050caa
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: 08d936a515a5a80a8687b53306f63fd1da67179644ffb6032268f193e995adcc
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1
OUTPUT_PREFLIGHT: target and sibling .control absent
DATA: exact S09 train/val t1.v2 and S01 ZIP-manifest identities frozen in §6
MODEL/CONFIG/TRAINING: N/A
PARTITION/NODES/NTASKS: gpu / 1 / 1
ALLOCATED_GPU: 1 x nvidia_gh200_120gb
PROCESS_VISIBLE_GPU: 0 (CUDA_VISIBLE_DEVICES empty; torch CUDA false/count zero)
CPU/MEMORY/TIME: 4 / 32 GiB / 00:15:00
CONCURRENCY_PREFLIGHT: squeue --me empty
TEST_ONLY_PREFLIGHT: accepted exact request without four-GPU default warning
REQUEUE/RETRY/REROLL/ARRAY/DDP: forbidden
STOP: any allocation/identity/test/data/solve/checker/leakage/parity/artifact failure or timeout
INTERPRETATION: constrained limited-rung proxy split/evaluator gate only
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gpus=nvidia_gh200_120gb:1 --time=00:15:00 --no-requeue --job-name=s10-stop-a-o127-ad93c89 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_o127_ad93c89333b0 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_o127_ad93c89333b0_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_a_o127_ad93c89333b0_%j.err --export=ALL,S10_STOPA_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_o127_ad93c89333b0,S10_STOPA_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1,S10_STOPA_EXPECTED_SOURCE_SHA=ad93c89333b0a8f19abf138c8d6816e742b51e35,S10_STOPA_EXPECTED_TREE=d9715477cc6d5e0b6bf9d35c7005222d0d4f63c3,S10_STOPA_EXPECTED_RUNNER_SHA256=94c52a7ceecadec166786990980801d05a153b2d8a710dd597af5a8da973225a /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_a_o127_ad93c89333b0/fl_v3/scripts/run_s10_stop_a_gate.sh
```

## 17. O-127 A3 Job 468404 — PASS, immutable evidence pending A4

The §16 command was submitted exactly once. Slurm allocated the exact requested
one GH200, four CPUs and 32 GiB on aarch64 node `n409`; the job completed before
the 15-minute limit with no restart. The resource assertion then hid the reserved
GPU from the process and proved both `torch.cuda.is_available() == false` and
`torch.cuda.device_count() == 0`. Slurm accounting independently reports zero
GPU memory and utilization. This was CPU-only execution on a GPU-reserved node,
not model/GPU compute.

```text
JOB: 468404
STATE/EXIT/ELAPSED/RESTARTS: COMPLETED / 0:0 / 00:07:59 / 0
START/END: 2026-07-16T05:57:40 / 2026-07-16T06:05:39
NODE/MACHINE: n409 / aarch64
REQ_AND_ALLOC: cpu=4, mem=32G, node=1, gres/gpu=1, gres/gpu:nvidia_gh200_120gb=1
PROCESS_CUDA_VISIBLE_DEVICES: empty
PROCESS_TORCH_CUDA_AVAILABLE/COUNT: false / 0
SLURM_BATCH_GPU_MEMORY/UTILIZATION: 0 / 0
TOTAL_CPU/MAX_RSS/MAX_VMEM: 07:05.995 / 9.39 GiB / 22.97 GiB
ALLOCATED_GPU_HOURS: 479 / 3600 = 0.133056
CUMULATIVE_STOP_A_AND_ABC_ALLOCATED_GPU_HOURS: 1.413334
ACTIVE_ABC_AGGREGATE_CEILING/ARITHMETIC_REMAINDER: 27.000000 / 25.586666
O127_SUBMISSIONS: 1 / 1 consumed
REAL_SPLIT_CANDIDATES: 1 / 1 consumed; no reroll
```

The dependency-backed focused suite passed `13 passed, 8 skipped in 0.76s`.
Both gate stages returned the frozen `FEASIBLE_FROZEN` state under their
constant-zero objective; this is a feasibility certificate only. The pre-solve
record binds 50 logs, 28,130 train samples, 6,019 official-val samples, candidate
ordinal one, no reroll and feature SHA-256
`231e879865b1fadf33f04cc65ee1b7adbc1cf1b3a1547dc525c69e137b3cf993`.
The immutable role outcome is:

| role | logs | scenes | samples | share |
|---|---:|---:|---:|---:|
| `D_fit` | 34 | 494 | 19,877 | 70.6612% of train |
| `D_select` | 8 | 115 | 4,626 | 16.4451% of train |
| `D_audit` | 8 | 91 | 3,627 | 12.8937% of train |
| `D_low` | 10 | 153 | 6,155 | 30.9654% of `D_fit` |
| `D_mid` | 20 | 290 | 11,661 | 58.6658% of `D_fit` |

The emitted-artifact checker was independently reloaded after the job. It
reconstructed all hard constraints and returned PASS. Across the intended grain,
overlap counts are exactly zero for log, scene, sample, annotation, instance and
raw path; unique counts are respectively 68, 850, 34,149, 1,166,187, 64,386 and
534,532. The ownership ledger contains exactly 34,149 records: 28,130 training
plus 6,019 official-val samples. `candidate_freeze.json` is absent and locked
until terminal STOP-D.

Full-val evaluator parity passed both fixtures at tolerance zero:

| fixture | status | filtered GT boxes | filtered prediction boxes | proxy mAP/NDS |
|---|---|---:|---:|---:|
| `P-GT` | `EXACT_PARITY` | 121,861 | 121,861 | 1.0000000000000004 / 1.0000000000000002 |
| `P-MIX` | `EXACT_PARITY` | 121,861 | 103,579 | 0.4866183371564836 / 0.7433091685782418 |

Exact equality covers filtered identities, all metric-data arrays, validity
masks and finite aggregate metrics. The adversarial filter evidence starts from
187,528 raw official-val boxes, includes 29,275 zero-point boxes and 22
trainval bicycle/motorcycle centers in racks, and finishes with 121,861 filtered
official-val boxes. The explicit empty-prediction adapter returns exact
`mAP=0.0`, `NDS=0.0` on 64 samples.

Primary immutable identities:

```text
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1
EXECUTION_IDENTITY_SHA256: 334a3901d301e557139648fa2c7800221f4cf2eb198c9e9a5f674a3ec4601c84
STOP_A_GATE_SHA256: ed168363a072ef25f8083e789a973127fa6fbd9d592c6077cc726e539cab161f
PRE_SOLVE_IDENTITY_SHA256: 0b6823c01df7e1359eebe5749c5bc267d77a59ac6ed82df647ccaf3f1392890a
SPLIT_PROTOCOL_SHA256: bc0cf5ef7c68fa5c25d54882cb49e9806d181cb0890125671d854f8650bfd24e
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
OWNERSHIP_LEDGER_SHA256: d2f0de912cf7e774d21ce1630839fb50cce6b6ae66a96ec1ed9b90988319e8b8
LEAKAGE_REPORT_SHA256: 91b956f82e9771a64205cbc0501d819eafda29d9de6e5c882e9b37eb872aa4ad
P_GT_PARITY_SHA256: bff1e8cf611a42b81e7d030d478c345558155a1776097e97f73196eca22d17ea
P_MIX_PARITY_SHA256: 06bbe2493dacf1390fe9c9fb3b7ea57be74b524d52d948f6fa2c7ba8fbf19e8c
EMPTY_EVAL_SHA256: aed7acee84575c1339ecf0ce53492d29d2321a1d21fd4bbe819cb3d3aeddce41
ARTIFACT_MANIFEST_SHA256: a9156f8256ed456fb8dd3225359f0b283b2e94ad69908165b126c33beaa5c294
RUNNER_ARTIFACT_MANIFEST_SHA256: cf7957fbe9e83a6b0b023882f53fdd86901ed7ca258cfa9cf886f12ef8b80697
STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
```

All three nested checksum manifests pass and the 305,129,578-byte, 26-file
output tree has zero writable paths. This PASS is limited to the constrained
train-only proxy split, ownership checker and evaluator engineering contract. It
is not model capability, convergence, recipe, official-val performance or
selection evidence. A4 must independently review the exact evidence commit
before STOP-A closes or STOP-B/C may start.

## 18. O-128 STOP-B / B-DIAG exact immutable tuple — consumed by Job 477892

```text
REQUEST_ID: S10-STOP-B-DIAG-O128-v1
REQUEST_STATE: APPROVED / FROZEN / CONSUMED ONCE
DERIVATION: O-128 exact activation inside O-124 STOP-B and 27-hour aggregate caps
SOURCE_SHA: 8fd832dc7d46e8818216ecbcf228ef8fd0590ecb
SOURCE_TREE: d5ce6c060279271295abdca41c3ad7aec5870315
BRANCH_AT_FREEZE: codex/s10-cl-model-recipe
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_8fd832dc7d46
SNAPSHOT_MODE: detached HEAD; clean; read-only; exact source/tree above
JOB_RUNNER_SHA256: 6cc4bfdea56f9c73c5e1b35fee55c269ee417256dffeab348d4a0c1a2536e818
OBSERVER_RUNNER_SHA256: 767fdcd8a62b0d24f0cd2a1471d880dcc4a030db2723cdea7dfadd4a0a78b36a
FP32_CONFIG_FILE_SHA256: 11e756e7174e4146d7809b0ec4cc29ceae917e46c268bd9e991283479d6eb612
FP32_RESOLVED_SHA256: 561145f41c83a0ac739c17c818aee36f9963142df7c3251c05683e5eba0e6337
FP16_CONFIG_FILE_SHA256: e61139b457b28fb2cb5fce478a7469d193162e4fd1c03e056f372862ccc22819
FP16_RESOLVED_SHA256: cf6f4effe0c9532a45f3a2503a3f98423af2e340b16ae0419d6b287655709a48
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
ROLE: D_low / 10 logs / 153 scenes / 6,155 samples
PANEL: generated and hash-frozen before model construction; P_core48 + disjoint P_term16; no reroll
MODEL/INIT: unchanged F-U current graph; A0 all-scratch W0 seed 0; identical W0 bytes across precision cells
PRECISION_ORDER: uniform FP32, then global FP16 autocast plus complete SECOND FP32 island
PHYSICAL_MICROBATCH: B4 for every detector forward
B1_EXCEPTION: four criterion-only evaluations sliced from one B4 detector output; zero B1 detector forwards
CELLS_IN_ORDER: pre-model panel freeze; focused tests; FP32 parity x2; FP32 broad x16; FP32 term x4; FP32 aggregation x1; FP16 parity x2; FP16 broad x16
DETECTOR_FORWARDS: 45 total; no optimizer construction/update, scheduler, EMA, GradScaler update, checkpoint or evaluator
TERM_REPLAYS: 4 batches x (aggregate + 12 exact task/term sources); bounded below accepted ~400 replay cap
LOCALIZATION_GATE: same adjacent interval in FP32/FP16; each precision median upstream/downstream RMS >=4, >=36/48 P_core samples ratio >2 and top interval, cross-precision median within 2x
TERM_GATE: if localized, one exact task/term is dominant only at >=3/4 P_term batches and median projection share >=0.5
EXIT: LOCALIZED or honest INCONCLUSIVE; neither is a causal model-change claim
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU: 1 x nvidia_gh200_120gb; one visible GH200; no DDP/array/spare GPU
CPU/MEMORY/TIME: 8 / 64 GiB / 00:30:00
MAX_THIS_ALLOCATION: 0.5 elapsed GH200-hour
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1
CONCURRENCY_PREFLIGHT: require no other owner job before submission
REQUEUE/RETRY: forbidden
STOP: any source/config/data/panel/resource/runtime/test/parity/reconstruction/state/artifact failure or timeout
ALLOWED_INTERPRETATION: current-W0 numerical localization or honest INCONCLUSIVE only
FORBIDDEN_INTERPRETATION: convergence, capability, production recipe, official-val performance, causal architecture proof, trained-checkpoint health
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-stop-b-8fd832d --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_8fd832dc7d46 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_b_diag_8fd832dc7d46_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_b_diag_8fd832dc7d46_%j.err --export=ALL,S10_STOPB_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_8fd832dc7d46,S10_STOPB_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1,S10_STOPB_EXPECTED_SOURCE_SHA=8fd832dc7d46e8818216ecbcf228ef8fd0590ecb,S10_STOPB_EXPECTED_TREE=d5ce6c060279271295abdca41c3ad7aec5870315,S10_STOPB_EXPECTED_RUNNER_SHA256=6cc4bfdea56f9c73c5e1b35fee55c269ee417256dffeab348d4a0c1a2536e818,S10_STOPB_EXPECTED_OBSERVER_SHA256=767fdcd8a62b0d24f0cd2a1471d880dcc4a030db2723cdea7dfadd4a0a78b36a,S10_STOPB_EXPECTED_FP32_CONFIG_SHA256=11e756e7174e4146d7809b0ec4cc29ceae917e46c268bd9e991283479d6eb612,S10_STOPB_EXPECTED_FP16_CONFIG_SHA256=e61139b457b28fb2cb5fce478a7469d193162e4fd1c03e056f372862ccc22819 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_8fd832dc7d46/fl_v3/scripts/run_s10_stop_b.sh
```

B-REFINE is not part of this command. It remains zero-or-one and can be derived
only after a successful B-DIAG `LOCALIZED` result explicitly recommends one of
the predeclared unresolved multi-operation intervals. `INCONCLUSIVE`, a hard
failure or a single-operation interval does not authorize panel growth, a rerun
or another hypothesis family.

## 19. O-128 B-DIAG Job 477892 — terminal early parity FAIL

Job `477892` consumed §18 exactly once. It received the exact one-GH200/eight-
CPU/64-GiB allocation on aarch64 node `n434`, ran without restart and failed
`1:0` after `00:04:44` (`284 / 3600 = 0.078889` allocated GH200-hours).
Cumulative STOP-A/ABC allocation is therefore `1.492223` GH200-hours, leaving
arithmetic room under the aggregate ceiling but no submission authority.

The 39 focused tests passed in `13.36s`. Runtime, source, config, cache, split
and dependency identities passed. The runner then froze, before model
construction, the accepted `D_low`-bound panel:

```text
PANEL_CONTENT_SHA256: 8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55
PANEL_FILE_SHA256: c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6
P_CORE/P_TERM: 48 / 16; disjoint; no reroll
P_TERM_TASK_POSITIVE_FRAMES: 16 / 11 / 8 / 7 / 9 / 15
W0_STATE_DICT_SHA256: e58bcd46d588c68b31335fe87cc5fbff06cbc0fbcdae7e88b0b1ed70d1d65395
MODEL_BEFORE_PANEL_FREEZE: false
MODEL_OUTPUT_BEFORE_PANEL_FREEZE: false
```

The first FP32 parity batch failed the combined disabled/on gate. The combined
gate covered output tensor hash, raw parameter-gradient hash, exact loss, RNG
state hash and model-state hash. The runner raised before appending the failing
parity record, so this evidence cannot identify which predicate(s) differed.
No FP32 broad, term replay, aggregation, FP16, optimizer, evaluation or model
update occurred. There is no `LOCALIZED`/`INCONCLUSIVE` STOP-B verdict and no
B-REFINE trigger. Raw artifacts are immutable and fully checksummed:

```text
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1
RUNNER_ARTIFACT_SHA256: 0dc23faf982a2905709f83b1cc2b0fde87d4850da7ad256a98dddd91acdec0a2
OBSERVE_STDERR_SHA256: e375302f297a2a8963822c4e91c68d49a50cbbc94452ac4c6892c19993e2c24d
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
FINAL_EXIT: 1
```

`SLURM_STDOUT_SHA256` above is the standard empty-file digest; its displayed
value is retained exactly as emitted by `sha256sum`. O-128 forbids an identical
retry and supplies no replacement. A future owner-approved diagnosis should
first persist all parity predicates and measure disabled/disabled run-to-run
variation before deciding whether disabled/on differences are instrumentation
effects or the accepted sparse backend's non-bit-deterministic baseline.

## 20. O-129 B-DIAG parity-remediation amendment — APPROVED

This is an approved bounded amendment, not yet an executable immutable tuple and
not a retroactive relaxation of
Job `477892`. It preserves the exact model, W0, precision partitions, panel,
seeds, broad/term/localization gates and no-update interpretation. The only
proposed change is to make the parity test distinguish instrumentation effects
from cold runtime tuning and same-path numerical variability.

The observed implementation compared the first cold disabled run directly with
the following enabled run and required raw parameter-gradient byte hashes to be
identical. The current sparse backbone selects MaskImplicitGemm and uses the
spconv extension's runtime tuner/backward implementation; PyTorch's strict flag
does not by itself attest byte determinism inside that extension. This makes
cold/warm algorithm state and same-path variability plausible confounders, not
proven causes of the failure.

Proposed immutable protocol:

```text
INPUT_PANEL: reuse the exact physical Job-477892 panel file; content SHA 8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55; file SHA c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6; never rebuild/reroll
MODEL/W0: unchanged current F-U A0 seed 0; require state SHA e58bcd46d588c68b31335fe87cc5fbff06cbc0fbcdae7e88b0b1ed70d1d65395
PARITY_BATCHES: existing P_core batch 0 and P_term batch 0 only; physical B4
PER_PRECISION_ORDER: one disabled warm-up on P_core batch 0; then disabled-0 / disabled-1 / enabled on each parity batch
WARMUP_ROLE: runtime-algorithm/cache warm-up only; no update; output excluded; W0 state hash must remain exact
OUTPUT/LOSS/RNG/STATE_GATE: exact equality across disabled-0, disabled-1 and enabled
GRADIENT_EVIDENCE: preserve raw hashes, missing-gradient sets, finiteness, per-parameter-prefix and global max-abs/relative-L2 errors
GRADIENT_FIXED_GATE: both disabled-0↔disabled-1 and disabled-0↔enabled must satisfy torch.allclose(rtol=1e-5, atol=1e-7) and global relative-L2 <=1e-6
ATTRIBUTION: disabled/disabled failure means baseline instability; disabled/disabled PASS plus disabled/enabled failure means instrumentation non-neutral; both PASS permits broad/term observations
FAILURE_DURABILITY: write every run identity, predicate and numerical error before any raise; always emit failure_summary.json plus artifact checksums
SCIENTIFIC_CONTINUATION: only after all FP32 and FP16 parity batches PASS; then execute the unchanged broad/term/aggregation/localization cells
```

The fixed `rtol=1e-5`, `atol=1e-7` envelope is reused from the already declared
STOP-B aggregation and term-gradient reconstruction checks; it is not estimated
from Job `477892`. Exact hashes remain evidence but cease to be the sole gradient
neutrality criterion. No tolerance is derived from the new disabled/disabled
measurements, and the panel or hypothesis family cannot grow after seeing them.

Proposed resource amendment:

```text
REPLACEMENT_B_DIAG: one fresh immutable source/snapshot/output; 1 GH200; 8 CPU; 64 GiB; 00:30:00; max 0.5 GH200-hour
CONDITIONAL_B_REFINE: zero or one only after unchanged LOCALIZED trigger; 1 GH200; at most 00:15:00; max 0.25 GH200-hour
MAX_ADDITIONAL_STOP_B: 0.75 GH200-hour
STOP_B_TOTAL_IF_FULLY_CONSUMED: 0.078889 + 0.75 = 0.828889 GH200-hour (< 2-hour stop cap)
ABC_TOTAL_IF_FULLY_CONSUMED: 1.492223 + 0.75 = 2.242223 GH200-hours (< 27-hour aggregate)
IDENTICAL_RETRY/DDP/ARRAY/SPARE_GPU: forbidden
```

Owner approval O-129 authorizes the exact protocol and resource amendment above.
Execution still requires S00 to freeze and record the exact immutable replacement
source/snapshot/config/data/cells/command/output tuple without semantic drift.

## 21. O-129 replacement B-DIAG exact immutable tuple — consumed by Job 478250

```text
REQUEST_ID: S10-STOP-B-DIAG-O129-v2
REQUEST_STATE: APPROVED / FROZEN / CONSUMED ONCE
DERIVATION: exact O-129 §20 activation inside the 0.5-GH200-hour replacement cap
SOURCE_SHA: 43f157b3eca7ca72633358b5a2d2dbc4c4e4684b
SOURCE_TREE: 1235a76b3192812881da23b86ab280cb33dd4c20
BRANCH_AT_FREEZE: codex/s10-cl-model-recipe
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_43f157b3eca7_o129
SNAPSHOT_MODE: standalone clone; detached HEAD; clean; recursively read-only; exact source/tree above
JOB_RUNNER_SHA256: 2d267d25f9dcc8bc2ec8979ed86004ce0b7e40f9e028c80cd6a5d6cf36f66473
OBSERVER_RUNNER_SHA256: 08f490e0e9d77396dd20d5a6bff62e98a7eb979556833c86fa8f46780304d36b
BINDING_SOURCE_SHA256: dfbad494f2e0ca6cd70d881835bf3527aa5a39e38fca599bd0c507b140e13f19
OBSERVATION_SOURCE_SHA256: 309ab4326c801f03aa4184fd30a4c280be6c1ea184c8af385bebe0a4865bc9b1
FP32_CONFIG_FILE_SHA256: 11e756e7174e4146d7809b0ec4cc29ceae917e46c268bd9e991283479d6eb612
FP32_RESOLVED_SHA256: 561145f41c83a0ac739c17c818aee36f9963142df7c3251c05683e5eba0e6337
FP16_CONFIG_FILE_SHA256: e61139b457b28fb2cb5fce478a7469d193162e4fd1c03e056f372862ccc22819
FP16_RESOLVED_SHA256: cf6f4effe0c9532a45f3a2503a3f98423af2e340b16ae0419d6b287655709a48
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
ROLE: D_low / 10 logs / 153 scenes / 6,155 samples
PANEL_SOURCE_JOB: 477892
PANEL_PHYSICAL_PATH: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1/panel_manifest.json
PANEL_FILE_SHA256: c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6
PANEL_CONTENT_SHA256: 8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55
PANEL_POLICY: direct read-only reuse; no metadata traversal, reconstruction, copy, reroll or growth
MODEL/INIT: unchanged F-U current graph; A0 all-scratch W0 seed 0
EXPECTED_W0_STATE_SHA256: e58bcd46d588c68b31335fe87cc5fbff06cbc0fbcdae7e88b0b1ed70d1d65395
PRECISION_ORDER: all FP32 parity, all FP16 parity, then FP32 observations, then FP16 observations
PHYSICAL_MICROBATCH: B4 for every detector forward
PARITY_ORDER_PER_PRECISION: disabled P_core-batch0 warm-up; then disabled-0/disabled-1/enabled on P_core-batch0 and P_term-batch0
PARITY_GATE: exact output/loss/RNG/model-state; raw gradient hashes retained; both off0-off1 and off0-on gradients allclose rtol=1e-5 atol=1e-7 and global relative-L2 <=1e-6
SCIENTIFIC_CELLS_AFTER_ALL_PARITY_PASS: FP32 broad x16; FP32 term x4; FP32 aggregation x1; FP16 broad x16
DETECTOR_FORWARDS: 51 total; no optimizer/update/scheduler/EMA/GradScaler/checkpoint/evaluator
FOCUSED_TESTS: 41 expected; test_s10_binding, test_s10_observation, S08 precision partition/diagnostics, S04 SECOND contract, two model-task selectors
LOCAL_VALIDATION: python3 py_compile PASS; bash -n PASS; git diff --check PASS; login pytest unavailable by environment and is not substituted for GH200 tests
FAILURE_DURABILITY: parity runs/predicates/errors precede raise; failure_summary.json and artifact_sha256s.json required
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU: 1 x nvidia_gh200_120gb; one visible GH200; no DDP/array/spare GPU
CPU/MEMORY/TIME: 8 / 64 GiB / 00:30:00
MAX_THIS_ALLOCATION: 0.5 elapsed GH200-hour
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_43f157b3eca7_o129_a1
CONCURRENCY_PREFLIGHT: no owner job at tuple freeze; recheck immediately before submission
REQUEUE/RETRY: forbidden
STOP: any source/config/data/panel/W0/resource/runtime/test/parity/reconstruction/state/artifact failure or timeout
ALLOWED_INTERPRETATION: current-W0 numerical localization or honest INCONCLUSIVE only
FORBIDDEN_INTERPRETATION: convergence, capability, production recipe, official-val performance, causal architecture proof, trained-checkpoint health
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-stop-b-o129 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_43f157b3eca7_o129 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_b_diag_43f157b3eca7_o129_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_b_diag_43f157b3eca7_o129_%j.err --export=ALL,S10_STOPB_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_43f157b3eca7_o129,S10_STOPB_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_43f157b3eca7_o129_a1,S10_STOPB_EXPECTED_SOURCE_SHA=43f157b3eca7ca72633358b5a2d2dbc4c4e4684b,S10_STOPB_EXPECTED_TREE=1235a76b3192812881da23b86ab280cb33dd4c20,S10_STOPB_EXPECTED_RUNNER_SHA256=2d267d25f9dcc8bc2ec8979ed86004ce0b7e40f9e028c80cd6a5d6cf36f66473,S10_STOPB_EXPECTED_OBSERVER_SHA256=08f490e0e9d77396dd20d5a6bff62e98a7eb979556833c86fa8f46780304d36b,S10_STOPB_EXPECTED_FP32_CONFIG_SHA256=11e756e7174e4146d7809b0ec4cc29ceae917e46c268bd9e991283479d6eb612,S10_STOPB_EXPECTED_FP16_CONFIG_SHA256=e61139b457b28fb2cb5fce478a7469d193162e4fd1c03e056f372862ccc22819 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_diag_43f157b3eca7_o129/fl_v3/scripts/run_s10_stop_b.sh
```

The conditional B-REFINE remains absent from this command. It may be frozen and
submitted at most once only if this replacement reaches `LOCALIZED` and its
unchanged `refinement_recommended` predicate is true. Any parity failure,
`INCONCLUSIVE`, non-trigger interval, or hard failure forbids B-REFINE.

## 22. O-129 replacement Job 478250 — terminal baseline-instability FAIL

Job `478250` consumed §21 exactly once on node `n409`. Slurm reports
`FAILED 1:0`, no restart, `00:04:28` (`268 / 3600 = 0.074444` allocated
GH200-hours). The 41 focused tests passed in `12.40s`. Source/tree, resources,
configs, data, split, runtime dependencies, the direct read-only Job-477892
panel reuse, and the exact expected W0 all passed. The sole FP32 disabled
warm-up preserved W0 and was excluded as declared.

The first calibrated `P_core` parity batch then produced:

```text
DISABLED0_LOSS: 391.5013732910156
DISABLED1_LOSS: 388.7950134277344
ENABLED_LOSS: 390.8602600097656
DISABLED0_DISABLED1_OUTPUT_EXACT: false
DISABLED0_DISABLED1_LOSS_EXACT: false
DISABLED0_DISABLED1_RNG_EXACT: true
DISABLED0_DISABLED1_MODEL_STATE_EXACT_TO_W0: true
DISABLED0_DISABLED1_RAW_GRADIENT_HASH_EXACT: false
DISABLED0_DISABLED1_GRADIENTS_FINITE: true
DISABLED0_DISABLED1_MISSING_GRADIENTS: 0 / 0; sets equal
DISABLED0_DISABLED1_PARAMETERS_COMPARED: 459
DISABLED0_DISABLED1_ALLCLOSE_FAILURES: 434
DISABLED0_DISABLED1_GLOBAL_RELATIVE_L2_ERROR: 3.5323887774502536
DISABLED0_DISABLED1_GLOBAL_MAX_ABS_ERROR: 2422412.736328125
CLASSIFICATION: baseline_instability
```

The disabled-0/enabled pair also failed, but §20 forbids attributing that
difference to instrumentation once disabled-0/disabled-1 already fails. The
same-path failure is large and affects output, loss and gradients, not merely
raw gradient bytes. Identical RNG-state and W0-state hashes rule out observed
framework RNG/state advancement, but do not identify a kernel or module cause.

The job stopped before the second FP32 parity batch, all FP16 parity, broad,
term, aggregation and localization cells. No optimizer, update, scheduler, EMA,
GradScaler update, checkpoint or evaluator ran. There is no
`LOCALIZED`/`INCONCLUSIVE` gradient-localization verdict and the predeclared
`B-REFINE` trigger is false; no B-REFINE tuple may be derived.

```text
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_43f157b3eca7_o129_a1
FAILURE_SUMMARY_SHA256: 1b33f286f704760a3ced86cd5b27ddba0483916a888eaf76b0e2a2f1ed0b91c7
PARITY_JSONL_SHA256: 3b9ec27926baa93ca24ea533851f4f1d4628032f3c86a3d033d5909c6d059fcc
PARITY_WARMUP_SHA256: d588a6aacd6cd492d0eb5de70963b556e94bf3408ccbfb59776f88d7a7d161f9
INNER_ARTIFACT_MANIFEST_SHA256: d733bcb6db833a7ebb206d0eedd787909e695f962ba8160e110aab0c67174977
RUNNER_ARTIFACT_MANIFEST_SHA256: 801e98c129797a6a71665c5227cbe6684a4001b39d617da91bbb7970b92c3543
FINAL_EXIT/FOCUSED_TESTS_EXIT/OBSERVE_EXIT: 1 / 0 / 1
```

Every runner-manifest entry verifies and the output is recursively read-only.
Actual STOP-B allocation is now `0.078889 + 0.074444 = 0.153333`
GH200-hours; actual cumulative ABC allocation is `1.492223 + 0.074444 =
1.566667` GH200-hours. Unused O-129 allocation is not a retry or alternative-
probe entitlement. The bounded evidence now requires independent review before
an owner decision on any rebaseline or new diagnostic design. Independent
review subsequently returned `PASS_WITH_RESIDUAL_RISK` with no open P0-P3 and
accepted only: calibrated baseline-instability FAIL; localization absent; owner
rebaseline required.

## 23. O-130 B-RAND stochastic/runtime decomposition — APPROVED

```text
REQUEST_ID: S10-STOP-B-RAND-O130-v1
REQUEST_STATE: APPROVED DESIGN / exact execution frozen in §24
DATA/PANEL: exact Job-477892 physical panel; first P_core B4 token vector only; no rebuild/reroll/growth
MODES: C-STR8 camera_only; L-S075 lidar_only; F-U fusion
MODE_STATUS: current component graphs for diagnosis; not candidate acceptance cells
PRECISION: FP32 only
INITIALIZATION: independent seed-0 W0 per mode; F-U must retain accepted W0 SHA e58bcd46...
RUNS_PER_MODE: warm-up seed 9000 x1; fixed seed 10000 x5; varying seeds 11000..11004 x1 each
TOTAL_FORWARD_BACKWARD_RUNS: 33 at physical B4
RNG_SEEDS_ROLE: stochastic probes only; not scientific multi-seed evidence
METRICS: loss relative difference; output global relative-L2/cosine/max-abs; parameter-gradient global/prefix relative-L2/cosine/max-abs; exact hashes retained as evidence
PAIRING: each group compares four runs to its first reference run; no all-pairs expansion
INTEGRITY_GATE: exact config/data/panel/source/runtime/resource identities; exactly 33 runs; finite loss/output/gradients; stable missing-gradient set; model state remains W0; fixed-seed post-run RNG hashes identical; complete checksums
NUMERICAL_EQUALITY_GATE: none; observed variation is evidence, not job failure
CLASSIFICATION_METRICS: median loss-relative, output-relative-L2, gradient-relative-L2
DOMINANCE_RULE: ratio >=4 using denominator floor 1e-8; unique support in at least two of three metrics
LABELS: CAMERA_STOCHASTICITY / LIDAR_RUNTIME_VARIATION / FUSION_ONLY_INTERACTION / MIXED_INCONCLUSIVE
LABEL_LIMIT: operational candidate source only; no kernel/module/causal claim
OPTIMIZER/UPDATE/SCHEDULER/EMA/CHECKPOINT/EVALUATOR: absent / zero
SCIENTIFIC_CONTINUATION: none inside job; no broad/term/localization/STOP-C
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU/CPU/MEMORY/TIME: 1 x nvidia_gh200_120gb / 8 / 64 GiB / 00:15:00
MAX_ALLOCATION: 0.25 GH200-hour
SUBMISSION/RETRY: exactly one / none
OUTPUT: one fresh immutable path derived after implementation
ACTUAL_ABC_BEFORE: 1.566667 GH200-hours
WORST_CASE_ABC_AFTER: 1.816667 GH200-hours
STOP: any identity/resource/test/integrity/artifact failure or timeout; return to owner
FORBIDDEN: output-equality acceptance, tolerance fitting, extra repeats/seeds/batches, model/recipe change, training, evaluator, DDP/array/spare GPU, automatic follow-up
OWNER_APPROVAL: O-130, 2026-07-16
```

## 24. O-130 B-RAND exact immutable tuple — CONSUMED ONCE

```text
REQUEST_ID: S10-STOP-B-RAND-O130-v1
REQUEST_STATE: APPROVED / FROZEN / CONSUMED by Job 479667
DERIVATION: exact implementation of approved §23 with no scientific or resource expansion
SOURCE_SHA: 0bf9c0ce4148bc82d977e0d66615f606144971b6
SOURCE_TREE: 1852db34197c142714456f3fa07e999393dc1ba9
BRANCH_AT_FREEZE: codex/s10-cl-model-recipe
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_rand_0bf9c0c_o130
SNAPSHOT_MODE: detached HEAD; clean; recursively read-only; zero group/other-writable paths
JOB_RUNNER_SHA256: 88f36ba78afa394465ccc7e774ac54165ec0afeff1ee857c940c4990b16ad3a2
OBSERVER_SHA256: 0e8cd0c221d91dbb2e174f5c920a20b02f6b5c2e82f568a7a77fa698715a2da0
OBSERVATION_HELPER_SHA256: af083dd6c4106ed822935c80e719965a57368b08f38c1232cbe09c39d5cc552f
OBSERVATION_TEST_SHA256: 8b0a44db59ec0fd61e02ffa8909a4214da67365e77ff481d2e25240c18ce38b6
CAMERA_CONFIG_FILE_SHA256: 1c597fb026f8634354562e8cad4f24ee7fb934844c24cc6c66a39ad729cff7bd
CAMERA_CONFIG_RESOLVED_SHA256: 7eb29f64746a631e496a4512997d02d221672c9ba2100291497abaf8f23415a4
LIDAR_CONFIG_FILE_SHA256: 5043b09195b3c05a7d94e8d88b3e3cd1bffdb6eba49ed93776fd966b28642698
LIDAR_CONFIG_RESOLVED_SHA256: bacf186c8cd7e965f332dec00691a666147cd0a62231581517bf3d0f246bff34
FUSION_CONFIG_FILE_SHA256: 11e756e7174e4146d7809b0ec4cc29ceae917e46c268bd9e991283479d6eb612
FUSION_CONFIG_RESOLVED_SHA256: 561145f41c83a0ac739c17c818aee36f9963142df7c3251c05683e5eba0e6337
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
ROLE: D_low
PANEL_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1/panel_manifest.json
PANEL_FILE_SHA256: c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6
PANEL_CONTENT_SHA256: 8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55
PANEL_CELL: batches_b4.P_core[0] only
TOKENS_IN_ORDER: 5fd95d1f56744e88adaef6f87d6e8559, 2da5b4573e734cc698798e40cfe542f0, b819c59bb4864a878beef275e9178672, 9a798a9c3ed04a389c5a51120edb3573
MODE_ORDER: C-STR8, L-S075, F-U
PRECISION/MICROBATCH: uniform FP32 / physical B4
RUN_ORDER_PER_MODE: seed 9000 warm-up; seed 10000 x5; seeds 11000..11004
TOTAL_FORWARD_BACKWARD_RUNS: 33
OPTIMIZER/UPDATE/EVALUATOR: absent / zero / absent
FOCUSED_TESTS: binding + observation + precision partition/diagnostics + SECOND contract + selected model-task guards
CUBLAS_WORKSPACE_CONFIG: :4096:8 before first Python/Torch process
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU: 1 x nvidia_gh200_120gb; one visible GH200; no DDP/array/spare GPU
CPU/MEMORY/TIME: 8 / 64 GiB / 00:15:00
MAX_THIS_ALLOCATION: 0.25 elapsed GH200-hour
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_rand_0bf9c0c_o130_a1
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: 66d7b7ed060c678fa3128ee9eda0d01d397e4ef07204ea58966b5b6816c78f95
CONCURRENCY_PREFLIGHT: require no other owner job before submission
REQUEUE/RETRY/REROLL: forbidden
STOP: any source/config/data/panel/resource/runtime/test/integrity/artifact failure or timeout; return to owner
ALLOWED_INTERPRETATION: bounded operational candidate-source triage under §23 only
FORBIDDEN_INTERPRETATION: numerical-equality gate, kernel/module causality, large-gradient explanation, convergence, capability, recipe/architecture acceptance, automatic continuation, STOP-C
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus=nvidia_gh200_120gb:1 --time=00:15:00 --no-requeue --job-name=s10-stop-b-rand-0bf9c0c --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_rand_0bf9c0c_o130 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_b_rand_0bf9c0c_o130_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_b_rand_0bf9c0c_o130_%j.err --export=ALL,S10_STOPB_RAND_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_rand_0bf9c0c_o130,S10_STOPB_RAND_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_rand_0bf9c0c_o130_a1,S10_STOPB_RAND_EXPECTED_SOURCE_SHA=0bf9c0ce4148bc82d977e0d66615f606144971b6,S10_STOPB_RAND_EXPECTED_TREE=1852db34197c142714456f3fa07e999393dc1ba9,S10_STOPB_RAND_EXPECTED_RUNNER_SHA256=88f36ba78afa394465ccc7e774ac54165ec0afeff1ee857c940c4990b16ad3a2,S10_STOPB_RAND_EXPECTED_OBSERVER_SHA256=0e8cd0c221d91dbb2e174f5c920a20b02f6b5c2e82f568a7a77fa698715a2da0,S10_STOPB_RAND_EXPECTED_CAMERA_CONFIG_SHA256=1c597fb026f8634354562e8cad4f24ee7fb934844c24cc6c66a39ad729cff7bd,S10_STOPB_RAND_EXPECTED_LIDAR_CONFIG_SHA256=5043b09195b3c05a7d94e8d88b3e3cd1bffdb6eba49ed93776fd966b28642698,S10_STOPB_RAND_EXPECTED_FUSION_CONFIG_SHA256=11e756e7174e4146d7809b0ec4cc29ceae917e46c268bd9e991283479d6eb612 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_b_rand_0bf9c0c_o130/fl_v3/scripts/run_s10_stop_b_randomness.sh
```

## 25. O-130 B-RAND Job 479667 — COMPLETED / integrity PASS

Job `479667` consumed §24 exactly once and no retry exists.

```text
STATE/EXIT/ELAPSED/RESTARTS: COMPLETED / 0:0 / 00:07:08 / 0
START/END: 2026-07-16T10:15:44 / 2026-07-16T10:22:52
NODE/MACHINE: n452 / aarch64
REQ_AND_ALLOC: cpu=8, mem=64G, node=1, gres/gpu=1, nvidia_gh200_120gb=1
VISIBLE_DEVICE: NVIDIA GH200 120GB / CUDA_VISIBLE_DEVICES=0
MAX_RSS/MAX_VMEM/TOTAL_CPU: 16,607,988 KiB / 388,061,248 KiB / 00:05:09.651
ACCOUNTED_GPU_MEMORY/UTILIZATION: 55,934 MiB / 54%
FOCUSED_TESTS: 43 passed in 12.75s
FORWARD_BACKWARD_RUNS/COMPARISONS: 33 / 24
FINAL/TEST/OBSERVE_EXIT: 0 / 0 / 0
INTEGRITY_GATE: PASS
CLASSIFICATION: MIXED_INCONCLUSIVE
QUALIFIED_SIGNALS: CAMERA_STOCHASTICITY and LIDAR_RUNTIME_VARIATION
OPTIMIZER/UPDATE/EVALUATOR: absent / zero / absent
ACTUAL_ALLOCATION: 428 / 3600 = 0.118889 GH200-hours
ACTUAL_STOP_B: 0.272222 GH200-hours
ACTUAL_CUMULATIVE_ABC: 1.685556 GH200-hours
ACTIVE_27_HOUR_REMAINDER: 25.314444 GH200-hours
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_rand_0bf9c0c_o130_a1
SUMMARY_SHA256: dd51f5801084714fccbd0c351b0696c3a6a2843b462662c74f757fc12cd147c5
RUNNER_ARTIFACT_MANIFEST_SHA256: d964b7cc5fa09692a9b8bd95b83cf8cfed85768ff771eaf8cc2a9c8c3cb11ac0
```

Both checksum layers verify and the output tree has zero writable paths.
Numerical differences did not fail the job. The bounded evidence qualifies
intended varying-seed camera stochasticity and same-seed LiDAR-route runtime
variation, while the predeclared unique-label rule returns
`MIXED_INCONCLUSIVE`. This is operational triage only: it does not explain the
large true unscaled LiDAR gradient, identify a causal kernel/module, select an
architecture/recipe, or authorize STOP-C. Independent review of an exact
evidence/docs SHA is required before STOP-B disposition.

## 26. O-131 STOP-C0 integrated health tuple — consumed / terminal incomplete

```text
REQUEST_ID: S10-STOP-C0-HEALTH-O131-v1
REQUEST_STATE: CONSUMED / FAILED 1:0 / INCOMPLETE / NO RETRY
DERIVATION: O-131 exact activation inside O-124 STOP-C and 27-hour aggregate caps
SOURCE_SHA: 89958be504d6abaef66810695402d2a09619794b
SOURCE_TREE: 3928d1869aa88398c35d179428c70bb380341378
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_89958be_o131
SNAPSHOT_MODE: detached HEAD / clean / read-only
SNAPSHOT_TRACKED_FILES: 612
SNAPSHOT_LS_TREE_SHA256: 0b4385192b444463e92b31799b5df926190f79772215800018cf63af5040521d
RUNNER: fl_v3/scripts/run_s10_stop_c0_health.sh
RUNNER_SHA256: 4cda2883cf9ef6de21fc9cf471d4b6f40d639b013c701b685233208b4e794741
ENTRY: fl_v3/scripts/s10_stop_c0_health.py
ENTRY_SHA256: 2b91e872493a5851e0feb4a23d45a0db4f91c070a03fcb305626e65eb0a55830
BASE_CONFIG: fl_v3/configs/s10_c0_f_a1.json
BASE_CONFIG_FILE_SHA256: 44a0890689826a238291928424a6a479e80cf0aed0b8231e63146ff763b1d81a
DERIVED_CONFIG_SHA256S: C0-F-A1 68670292d09dc57f3da4fe2dd2c51d6c4c03fe489b7e273248628da5b1a4493e; C0-L-A0 f71573de52eaec10fe8b3a92512721169aea123f9cdfff173d81da3c52a3b9df; C0-F-A0-P64 109f5eade3074895d32c5d1092825e87df8967180ebec4c200d5c446bca406ab
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
SWIN_IMAGENET1K_V1: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home/hub/checkpoints/swin_t-704ceda3.pth
SWIN_IMAGENET1K_V1_SHA256: 704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b
CELL_ORDER: C0-F-A1; C0-L-A0; C0-F-A0-P64
C0-F-A1: D_low physical B4 drop_last / 1,538 attempted windows / terminal D_select eval
C0-L-A0: D_low physical B4 drop_last / 1,538 attempted windows / terminal D_select eval
C0-F-A0-P64: D_low physical B4 / 64 attempted windows / no evaluator
SEED/PRECISION: 0 / global FP16 plus complete SECOND FP32 island
RECIPE: AdamW 1e-4/0.01; constant; uniform; no clip/EMA/augmentation/CBGS/GT-paste
DROPPED_TOKENS (REQUEST-TIME CLAIM; INVALIDATED BY REVIEW): the remainder count three is correct and the two full cells used the same actual order/remainder, but the raw named tokens were predicted from the wrong DataLoader RNG state and are not exact
LOSS_CHUNK_BOUNDARIES: 64,384,768,1152,1538; short control 16,64
DIAGNOSTIC_ATTEMPTS: 1,4,16,64,256,768,1538; short control through 64
PROFILE: C0-F-A1 only / wait 16 / warmup 2 / active 10 / one trace cycle; no sampled-diagnostic overlap
TELEMETRY: chunk wall/samples-per-second, CUDA peak allocated/reserved, 1 Hz nvidia-smi
CHECKPOINTS: full raw training-state checkpoint for C0-F-A1 and C0-L-A0 only
EVALUATOR: exact STOP-A manifest-bound internal D_select evaluator; official val absent
FOCUSED_TESTS: precision partition/diagnostics, binding/subset evaluator, profile neutrality/readiness, checkpoint/resume
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU: 1 x nvidia_gh200_120gb; one visible GH200; no DDP/array/spare GPU
CPU/MEMORY/TIME: 16 / 96 GiB / 01:00:00
MAX_THIS_ALLOCATION: 1.0 elapsed GH200-hour
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_89958be_o131_a1
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: 73b51d060dd148b9b196a8570f682695ae369b1f23dc3ddbd7cd5755942393af
CONCURRENCY_PREFLIGHT: require no other owner job before submission
REQUEUE/RETRY/REROLL: forbidden
STOP: source/config/data/weight/resource/runtime/test failure; hard health gate; timeout; return to owner
ALLOWED_INTERPRETATION: bounded numerical/training trajectory, gradient-harm correlation and descriptive one-epoch internal fusion-minus-LiDAR delta
FORBIDDEN_INTERPRETATION: gradient-module causality, recipe/architecture acceptance, official-val or full capability/fusion claim, final GH200 bottleneck, automatic counterfactual/later-C/D/E/F continuation
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=01:00:00 --no-requeue --job-name=s10-c0-89958be --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_89958be_o131 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_c0_89958be_o131_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_c0_89958be_o131_%j.err --export=ALL,S10_C0_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_89958be_o131,S10_C0_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_89958be_o131_a1,S10_C0_EXPECTED_SOURCE_SHA=89958be504d6abaef66810695402d2a09619794b,S10_C0_EXPECTED_TREE=3928d1869aa88398c35d179428c70bb380341378,S10_C0_EXPECTED_RUNNER_SHA256=4cda2883cf9ef6de21fc9cf471d4b6f40d639b013c701b685233208b4e794741,S10_C0_EXPECTED_ENTRY_SHA256=2b91e872493a5851e0feb4a23d45a0db4f91c070a03fcb305626e65eb0a55830,S10_C0_EXPECTED_CONFIG_SHA256=44a0890689826a238291928424a6a479e80cf0aed0b8231e63146ff763b1d81a /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_89958be_o131/fl_v3/scripts/run_s10_stop_c0_health.sh
```

Any change to cells, horizons, seed, recipe, data roles, diagnostics, resource
or output invalidates O-131.

### 26.1 Consumption record

```text
JOB_ID: 492525
STATE/EXIT/ELAPSED: FAILED / 1:0 / 00:47:32
NODE/RESOURCES: n405 / one GH200 / 16 CPU / 96 GiB
FOCUSED_TESTS: 74 passed / 3 skipped / 16.51 s
COMPLETED_CELLS: C0-F-A1; C0-L-A0
MISSING_CELL: C0-F-A0-P64 summary/diagnostics
AGGREGATE_SUMMARY: absent
FAILURE: short 64-window control was incorrectly required to exhaust the 1,538-window epoch iterator
RAW_MECHANICAL_HEALTH_LABELS: F-A1 HARD_FAIL; L-A0 HARD_FAIL
HEALTH_GATE_DEFECT: lidar_encoder.to_bev is nn.Identity in SECOND-075 and has no trainable parameter
DROP_TOKEN_PROVENANCE: INVALID for the three raw named tokens; DataLoader consumed a base seed before RandomSampler randperm, so only the remainder count and same-order F/L construction remain supported
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_89958be_o131_a1
RUNNER_ARTIFACT_MANIFEST_SHA256: 950a79919dbf07b1dab54f4ff91c4bf9c49692bf05417c532725dd508c93397e
ARTIFACT_MANIFEST_CHECK: 25/25 OK
ACTUAL_ALLOCATION: 2852 / 3600 = 0.792222 GH200-hours
ACTUAL_CUMULATIVE_ABC: 2.477778 GH200-hours
ACTIVE_27_HOUR_REMAINDER: 24.522222 GH200-hours
REQUEUE/RETRY/REROLL: none / forbidden / forbidden
```

The two complete cell artifacts remain bounded evidence; the exact C0 protocol
does not pass because the scratch control and aggregate summary are absent. The
post-job model/loss/gradient/update-neutral v2 artifact fixes and regression tests
are not execution authority and were not run on GH200.

## 27. O-132 STOP-C0-v2 full clean replay — consumed / execution PASS

The owner rejected closure-by-review of the incomplete C0 package and explicitly
selected the full clean-replay option. O-132 supersedes O-131's no-retry clause
for exactly this one replacement; it does not authorize another replacement or
any later STOP-C/D/E/F execution. The old Job `492525` and raw v1 output remain
immutable negative evidence.

```text
REQUEST_ID: S10-STOP-C0-V2-CLEAN-O132-v1
REQUEST_STATE: CONSUMED / JOB 496312 COMPLETED 0:0 / V2 EXECUTION GATE PASS
OWNER_INSTRUCTION: 按照方案一进行完整的C0-v2 clean replay
DERIVATION: one explicit full replacement after diagnosed O-131 runner/gate/provenance defects; inside the active 27-hour O-124 aggregate ceiling
SOURCE_SHA: 2262b4063a3e419b17f4b911a9e11a7ff50ea784
SOURCE_TREE: f03825398a0fb9c13a5d335f012c49bc6d787602
SOURCE_CHANGE_FROM_REVIEWED_V2: fail-closed shell assertions only; no model/data/precision/recipe/cell change
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_v2_2262b40_o132
SNAPSHOT_MODE: detached HEAD / clean / read-only
SNAPSHOT_TRACKED_FILES: 613
SNAPSHOT_LS_TREE_SHA256: d3afdf5bf832a9d09e02cc575a0fcd91b95e9d3d4ef062ea7a709a7f2b52fc2c
RUNNER: fl_v3/scripts/run_s10_stop_c0_health.sh
RUNNER_SHA256: eea0081e321e20830e931e222e377ffff38502fe16cc94ec30b1252dba0955e9
ENTRY: fl_v3/scripts/s10_stop_c0_health.py
ENTRY_SHA256: 45c93983a5e522708707cdbd8c0cbb4dcd010a76b27d9412a016551d14f56047
BASE_CONFIG: fl_v3/configs/s10_c0_f_a1.json
BASE_CONFIG_FILE_SHA256: 44a0890689826a238291928424a6a479e80cf0aed0b8231e63146ff763b1d81a
DERIVED_CONFIG_SHA256S: C0-F-A1 68670292d09dc57f3da4fe2dd2c51d6c4c03fe489b7e273248628da5b1a4493e; C0-L-A0 f71573de52eaec10fe8b3a92512721169aea123f9cdfff173d81da3c52a3b9df; C0-F-A0-P64 109f5eade3074895d32c5d1092825e87df8967180ebec4c200d5c446bca406ab
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
SWIN_IMAGENET1K_V1_SHA256: 704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b
CELL_ORDER: C0-F-A1; C0-L-A0; C0-F-A0-P64
C0-F-A1: D_low physical B4 drop_last / 1,538 attempted windows / terminal D_select eval
C0-L-A0: D_low physical B4 drop_last / 1,538 attempted windows / terminal D_select eval
C0-F-A0-P64: D_low physical B4 / 64 attempted windows / no evaluator
SEED/PRECISION: 0 / global FP16 plus complete SECOND FP32 island
RECIPE: AdamW 1e-4/0.01; constant; uniform; no clip/EMA/augmentation/CBGS/GT-paste
DIAGNOSTIC_ATTEMPTS: 1,4,16,64,256,768,1538; short control through 64
PROFILE: C0-F-A1 only / wait 16 / warmup 2 / active 10 / one trace cycle
FOCUSED_TESTS: the exact eight selectors in the runner; expected 80 passed / 3 skipped; any failure/error stops
V2_GATE: aggregate schema/status/source/cell-order/hard-failures; all three v2 cell summaries; exact attempted windows; actual-collated token evidence; F/L 6,152 consumed samples, three exact remainder tokens and identical ordered/remainder hashes; scratch 256 consumed samples with no full-epoch remainder claim; zero health hard errors; both checkpoints present
PARTITION/NODES/NTASKS: gpu / 1 / 1
GPU: 1 x nvidia_gh200_120gb; one visible GH200; no DDP/array/spare GPU
CPU/MEMORY/TIME: 16 / 96 GiB / 01:00:00
MAX_THIS_ALLOCATION: 1.0 elapsed GH200-hour
PREVIOUS_ACTUAL_CUMULATIVE_ABC: 2.477778 GH200-hours
MAX_POSTJOB_CUMULATIVE_ABC: 3.477778 GH200-hours
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_v2_2262b40_o132_a1
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: d8bebfb51a69f6825e76866f6f8bd3114105205927a8b5d7707881098fc8d188
CONCURRENCY_PREFLIGHT: require no other owner job immediately before submission
REQUEUE/RETRY/REROLL: forbidden; this is the sole O-132 replacement
SUCCESS: Job COMPLETED 0:0; focused tests pass; all v2 gates pass; three cell summaries and aggregate exist; artifact checksum manifest passes
STOP: source/config/data/weight/resource/runtime/test/v2-gate failure or timeout; return directly to owner
REVIEW: no intermediate reviewer/re-review chain; any later final review requires a separate owner decision after a completed replay
ALLOWED_INTERPRETATION: bounded C0 numerical/training trajectory, gradient-harm correlation and descriptive one-epoch internal fusion-minus-LiDAR delta
FORBIDDEN_INTERPRETATION: gradient-module causality, recipe/architecture acceptance, official-val/full capability/full fusion claim, final GH200 bottleneck, automatic later-C/D/E/F continuation
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=01:00:00 --no-requeue --job-name=s10-c0v2-2262b40 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_v2_2262b40_o132 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_c0_v2_2262b40_o132_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_stop_c0_v2_2262b40_o132_%j.err --export=ALL,S10_C0_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_v2_2262b40_o132,S10_C0_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_v2_2262b40_o132_a1,S10_C0_EXPECTED_SOURCE_SHA=2262b4063a3e419b17f4b911a9e11a7ff50ea784,S10_C0_EXPECTED_TREE=f03825398a0fb9c13a5d335f012c49bc6d787602,S10_C0_EXPECTED_RUNNER_SHA256=eea0081e321e20830e931e222e377ffff38502fe16cc94ec30b1252dba0955e9,S10_C0_EXPECTED_ENTRY_SHA256=45c93983a5e522708707cdbd8c0cbb4dcd010a76b27d9412a016551d14f56047,S10_C0_EXPECTED_CONFIG_SHA256=44a0890689826a238291928424a6a479e80cf0aed0b8231e63146ff763b1d81a /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_stop_c0_v2_2262b40_o132/fl_v3/scripts/run_s10_stop_c0_health.sh
```

Any tuple or scope change invalidates O-132. A finite but weak scientific result
is retained as evidence; it does not trigger tuning, retry or candidate growth.

### 27.1 Consumption record

```text
JOB: 496312
STATE/EXIT: COMPLETED / 0:0
ELAPSED: 00:45:15 / 2715 seconds / 0.754167 GH200-hours
NODE: n128
SLURM_BATCH_MAXRSS/MAXVMSIZE: 83838302K / 1076047104K
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_v2_2262b40_o132_a1
OUTPUT_MODE/SIZE: read-only / approximately 2.0 GiB / zero writable paths
FOCUSED_TESTS: 80 passed / 3 skipped / 16.77 seconds
FINAL/C0/FOCUSED_EXIT: 0 / 0 / 0
AGGREGATE: fl_v3.s10.stop_c0_health.v2 / PASS / hard_failures={}
CELL_SUMMARIES: C0-F-A1 present; C0-L-A0 present; C0-F-A0-P64 present
TOKEN_GATE: actual_collated_batches; F/L ordered hash and three-token remainder hash identical
ARTIFACT_MANIFEST: 28/28 OK
ARTIFACT_MANIFEST_SHA256: dbb7a088579c14af19d7d36bcf0bde9c0dcbe48685ce00c118f178760ffa3cf2
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
PREVIOUS_ACTUAL_CUMULATIVE_ABC: 2.477778 GH200-hours
POSTJOB_ACTUAL_CUMULATIVE_ABC: 3.231945 GH200-hours
REMAINING_UNDER_27_HOUR_CEILING: 23.768055 GH200-hours; not execution authority
RETRY/REROLL: none; O-132 is consumed
INDEPENDENT_REPLAY_REVIEW: not authorized or performed; owner explicitly rejected an intermediate reviewer chain
```

The successful execution gate completes the bounded C0-v2 replay. It does not
authorize a second replay, later STOP-C strong contrasts, recipe or architecture
selection, STOP-D/E/F, merge, push, upload, publication, attack or defense.

## 28. O-133 C1-A/C1-B — plan accepted / not executable

```text
DECISION: O-133
STATE: SCIENTIFIC PLAN ACCEPTED / DOCS ONLY / NOT EXECUTABLE
C1-A: current GN versus direct-reference BN1d on frozen STOP-B L-S075 B4 panel; FP32; normal loss backward plus frozen SECOND-output VJP; no optimizer/evaluator
C1-A_EXIT: LOCALIZED_NORM / LOCALIZED_HEAD_LOSS / LOCALIZED_SPARSE_OCCUPANCY / INCONCLUSIVE
BN1D_STATUS: diagnostic only unless a future exact causal and matched-training gate promotes it; promotion consumes one existing counterfactual slot
SCALER_POLICY: common frozen no-update B4 qualification selects one conservative power-of-two init scale for all admitted C1 FP16 training graphs
C1-B_LINEAGES: C1-CUR-A1; C1-CUR-A2; C1-MIT-A2
C1_FUNNEL: fresh G20 per new graph/init; matched D_low one epoch; at most three D_mid lineages for three epochs; D_select internal evaluation; at most two STOP-C survivors
C1_LIMITS: physical B4; seed 0; matched actual tokens/exposure; no recipe sweep, extra seed, D_audit, official val or full run
MIT_ANCHOR: exact graph/init/component package pending owner explanation and decision; no implementation may be inferred
IMPLEMENTATION/COMMIT/COMPUTE/REVIEW: not authorized by O-133
EXECUTABLE_TUPLE: none
```

O-134 below supersedes this no-executable state for C1-A only. The exact MIT
anchor remains pending before any C1-B reference-guided repair. Unused O-124
aggregate budget is not execution authority.

## 29. O-134 C1-A exact immutable tuple — consumed / pre-execution FAIL

```text
REQUEST_ID: S10-C1A-GRAD-CAUSAL-O134-v1
STATE: CONSUMED BY JOB 502456 / FAILED BEFORE CANDIDATE EXECUTION / NO VERDICT / NO RETRY
SOURCE_SHA: 95c09a149029d63e243e5e418385f39d2d1aed66
SOURCE_TREE: 10b8da87eff3b5aed171a4d325061a2baf9dee0e
BRANCH_AT_FREEZE: codex/s10-cl-model-recipe
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_95c09a1_o134
SNAPSHOT_MODE: detached HEAD; clean; recursively read-only; zero group/other-writable paths
RUNNER_SHA256: e178080b7be31c981e0366ba9894caba5e268f2be58efc547aec9befab65e039
ENTRY_SHA256: 6247b732e1054480a082416f635dce04b56182f09c720538cd34beca3a92f6c4
SECOND_SHA256: 9238d33e3cc28ffc1693585c691ee9967444b19da532a9cdc56a7210ec8c153d
OBSERVATION_HELPER_SHA256: 13b27eaf94fc07c7752db7a1090c0bcf307e5e804ee18f74b2bfd9be15602189
CONFIG: fl_v3/configs/s10_b_rand_l_fp32.json
CONFIG_FILE_SHA256: 5043b09195b3c05a7d94e8d88b3e3cd1bffdb6eba49ed93776fd966b28642698
CONFIG_RESOLVED_SHA256: bacf186c8cd7e965f332dec00691a666147cd0a62231581517bf3d0f246bff34
EXPECTED_CURRENT_GN_W0: a1a98033131d5496308f0a2694032a1473d582d3435cabd9db285f60b357ef0a
DATA: accepted STOP-A D_low plus complete accepted STOP-B panel; P_core48 + P_term16 as 16 disjoint physical-B4 batches / 64 samples
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
PANEL_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1/panel_manifest.json
PANEL_FILE_SHA256: c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6
PANEL_CONTENT_SHA256: 8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55
GRAPH: L-S075 only; current tiny-group GN versus direct-reference BN1d(eps=1e-3,momentum=0.01); identical convolution and affine W0
PRECISION: uniform FP32
PATHWAYS: normal detection-loss backward; coordinate/channel-derived fixed SECOND-output VJP
REPEATS/RUNS: two per candidate/path/batch; 2 x 2 x 16 x 2 = 128
RUNTIME_VARIATION_GATE: paired geometric centres; BN/GN median <=0.5; >=75% batches <=0.8; median log effect exceeds p95 within-method two-repeat log variation
EXIT: LOCALIZED_NORM / LOCALIZED_HEAD_LOSS / LOCALIZED_SPARSE_OCCUPANCY / INCONCLUSIVE
OPTIMIZER/UPDATE/EVALUATOR: absent / zero / absent
RESOURCE: one GH200; one node/task; 8 CPU; 64 GiB; 00:30:00; max 0.5 GH200-hour
SUBMISSIONS/RETRY: exactly one / forbidden
FOCUSED_TESTS: SECOND contract + C1-A classification/observation + S10 binding + S08 precision partition + selected task/config guards; all before model execution
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1a_95c09a1_o134_a1
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: 55930167bc82846cf4f472624203ef337cb164a3610a70a580707b4e595fbf59
CONCURRENCY_PREFLIGHT: require no other owner job before submission
STOP: any source/tree/file/config/data/panel/resource/runtime/test/identity/integrity/artifact failure or timeout; return to owner
C1-B: not executable; current A1/A2 first under a future exact gate, MIT-reference repair only after materially-worse evidence
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1a-95c09a1 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_95c09a1_o134 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1a_95c09a1_o134_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1a_95c09a1_o134_%j.err --export=ALL,S10_C1A_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_95c09a1_o134,S10_C1A_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1a_95c09a1_o134_a1,S10_C1A_EXPECTED_SOURCE_SHA=95c09a149029d63e243e5e418385f39d2d1aed66,S10_C1A_EXPECTED_TREE=10b8da87eff3b5aed171a4d325061a2baf9dee0e,S10_C1A_EXPECTED_RUNNER_SHA256=e178080b7be31c981e0366ba9894caba5e268f2be58efc547aec9befab65e039,S10_C1A_EXPECTED_ENTRY_SHA256=6247b732e1054480a082416f635dce04b56182f09c720538cd34beca3a92f6c4,S10_C1A_EXPECTED_SECOND_SHA256=9238d33e3cc28ffc1693585c691ee9967444b19da532a9cdc56a7210ec8c153d,S10_C1A_EXPECTED_OBSERVATION_SHA256=13b27eaf94fc07c7752db7a1090c0bcf307e5e804ee18f74b2bfd9be15602189,S10_C1A_EXPECTED_CONFIG_SHA256=5043b09195b3c05a7d94e8d88b3e3cd1bffdb6eba49ed93776fd966b28642698 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_95c09a1_o134/fl_v3/scripts/run_s10_c1a_gradient_causality.sh
```

Any semantic/resource change or first job failure returns to the owner; O-134 is
not a remediation-loop authorization and does not permit a retry.

### 29.1 Job 502456 consumption record

```text
JOB_ID: 502456
STATE/EXIT/ELAPSED: FAILED / 1:0 / 00:03:03
ACTUAL_ALLOCATION: 183 / 3600 = 0.050833 GH200-hours
POSTJOB_CUMULATIVE_S10_ABC/C1: 3.231945 + 0.050833 = 3.282778 GH200-hours
FOCUSED_TESTS: 36 passed / 0 failed / 1.56 s
SOURCE/RUNTIME/DATA/PANEL: identity gates passed
CANDIDATE_FORWARD_BACKWARD_RUNS: 0 / 128
OPTIMIZER/UPDATE/EVALUATOR: absent / zero / absent
FAILURE: BN1d mapping assertion expected running_mean, running_var and num_batches_tracked in missing_keys; PyTorch backward-compatibly synthesized num_batches_tracked and reported only the first two
MODEL_TO_GPU/LOADER/GRADIENTS: not reached / not reached / absent
C1A_VERDICT: absent
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1a_95c09a1_o134_a1
FAILURE_SUMMARY_SHA256: 45b3569df55392d4ee5f74f054eb1c83113158fb46e3185c0afe074e169eb2d5
EXECUTION_IDENTITY_SHA256: 34fc62d89d083075b2e7503d03e96ef915999e581cbf23beb6d066e4d35e11a0
INNER_ARTIFACT_MANIFEST_SHA256: 14f6e80e2c516c69d5667e260df1d8d6cea52b4a9eb28ffd4cc59df9e0fa40d4
RUNNER_ARTIFACT_MANIFEST_SHA256: 7724a1c91e41291dfaad480057c1d033e1505d52d753d07a15ded7844d4a83c6
FINAL/TEST/C1A_EXIT: 1 / 0 / 1
ARTIFACT_CHECKS: inner 2/2 OK; runner 13/13 OK; output recursively read-only
RETRY/REPLACEMENT: none authorized
```

This failure has no scientific interpretation. A possible replacement must fix
only the mapping assertion, explicitly validate that every BN1d
`num_batches_tracked` buffer exists and equals zero after load, add a regression
test for PyTorch's missing-key behavior, freeze a new source/snapshot/output and
obtain new owner compute authority.

## 30. O-135 exact assertion remediation — no compute

```text
STATE: CODE/TEST/DOC FIX AUTHORIZED / WORKTREE PREPARED / UNCOMMITTED / UNEXECUTED
SCOPE: validate reported missing_keys == running_mean+running_var only; separately validate 21 num_batches_tracked buffers exist and equal zero
ADDITIONAL_GATES: running means zero; running variances one; zero unexpected keys; unchanged exact trainable-parameter hash
REGRESSION: standalone GroupNorm affine state -> BatchNorm1d strict=False reproduces two reported missing buffers and synthesized zero batch counter; nonzero counter rejected
MODEL/LOSS/GRADIENT/DATA/CELLS/THRESHOLDS: unchanged
COMMIT/SNAPSHOT/SLURM/RETRY/C1-B: not authorized
```

Login-node verification is limited to `bash -n`, Python syntax compilation and
`git diff --check` because the x86 login environment has no project PyTorch or
pytest. Dependency-backed execution remains pending a future exact authorization.

## 31. O-136 strictly derived C1-A replacement — consumed / execution PASS

```text
REQUEST_ID: S10-C1A-GRAD-CAUSAL-O136-v2
STATE: CONSUMED BY JOB 502572 / COMPLETED 0:0 / INTEGRITY PASS / LOCALIZED_NORM
REMEDIATION_SOURCE: d713bfe3b5e5c587f58ce70721b2b6eea0b050ec
SOURCE_TREE: 5a0d8aa2c1dc9517d735200b0dbae47843ec8c74
DERIVATION: only the O-135 BN1d state-mapping assertion and direct regression test differ from O-134
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_fix_d713bfe_o136
SNAPSHOT_MODE: detached HEAD; clean; recursively read-only; zero group/other-writable paths
RUNNER_SHA256: e178080b7be31c981e0366ba9894caba5e268f2be58efc547aec9befab65e039
ENTRY_SHA256: 8ca38de0bdd8511143de55204f0f0c18fafc72bc0362afa5571a16262f9e8885
SECOND_SHA256: 9238d33e3cc28ffc1693585c691ee9967444b19da532a9cdc56a7210ec8c153d
OBSERVATION_HELPER_SHA256: 81d25064dbb9b5883b5f37d2a7d0124ef6809afb3f14e9d7233891cb76dfacbf
OBSERVATION_TEST_SHA256: d1a42bcf8403d926ce2f75d7814568978274025d578da999c9abcd78bc484f62
CONFIG: fl_v3/configs/s10_b_rand_l_fp32.json
CONFIG_FILE_SHA256: 5043b09195b3c05a7d94e8d88b3e3cd1bffdb6eba49ed93776fd966b28642698
CONFIG_RESOLVED_SHA256: bacf186c8cd7e965f332dec00691a666147cd0a62231581517bf3d0f246bff34
EXPECTED_CURRENT_GN_W0: a1a98033131d5496308f0a2694032a1473d582d3435cabd9db285f60b357ef0a
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
PANEL_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1/panel_manifest.json
PANEL_FILE_SHA256: c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6
PANEL_CONTENT_SHA256: 8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55
DATA/PANEL/CANDIDATES/W0/PRECISION/PATHWAYS/REPEATS/RUNS: unchanged from §29
METRICS/THRESHOLDS/EXIT: unchanged from §29
OPTIMIZER/UPDATE/EVALUATOR: absent / zero / absent
RESOURCE: one GH200; one node/task; 8 CPU; 64 GiB; 00:30:00; max 0.5 GH200-hour
SUBMISSIONS/REQUEUE/RETRY/REROLL: exactly one / forbidden / forbidden / forbidden
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1a_fix_d713bfe_o136_a1
SUBMIT_COMMAND_SHA256_NO_TRAILING_NEWLINE: dc4ad389de43cf0e9f3eb7346fa10c546b70897eb65b885e42eaaf7aad08faa4
CONCURRENCY_PREFLIGHT: require no other owner job before submission
STOP: any source/tree/file/config/data/panel/resource/runtime/test/identity/integrity/artifact failure or timeout; return directly to owner
C1-B/LATER-STOPS/REVIEWER/MERGE/PUSH/UPLOAD: not authorized
```

Exact command:

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1a-fix-d713bfe --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_fix_d713bfe_o136 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1a_fix_d713bfe_o136_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1a_fix_d713bfe_o136_%j.err --export=ALL,S10_C1A_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_fix_d713bfe_o136,S10_C1A_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1a_fix_d713bfe_o136_a1,S10_C1A_EXPECTED_SOURCE_SHA=d713bfe3b5e5c587f58ce70721b2b6eea0b050ec,S10_C1A_EXPECTED_TREE=5a0d8aa2c1dc9517d735200b0dbae47843ec8c74,S10_C1A_EXPECTED_RUNNER_SHA256=e178080b7be31c981e0366ba9894caba5e268f2be58efc547aec9befab65e039,S10_C1A_EXPECTED_ENTRY_SHA256=8ca38de0bdd8511143de55204f0f0c18fafc72bc0362afa5571a16262f9e8885,S10_C1A_EXPECTED_SECOND_SHA256=9238d33e3cc28ffc1693585c691ee9967444b19da532a9cdc56a7210ec8c153d,S10_C1A_EXPECTED_OBSERVATION_SHA256=81d25064dbb9b5883b5f37d2a7d0124ef6809afb3f14e9d7233891cb76dfacbf,S10_C1A_EXPECTED_CONFIG_SHA256=5043b09195b3c05a7d94e8d88b3e3cd1bffdb6eba49ed93776fd966b28642698 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1a_fix_d713bfe_o136/fl_v3/scripts/run_s10_c1a_gradient_causality.sh
```

No field may change the §29 scientific tuple. This literal command may be
submitted once only after snapshot/output/concurrency preflight succeeds.

### 31.1 Job 502572 consumption record

```text
JOB_ID: 502572
STATE/EXIT/ELAPSED: COMPLETED / 0:0 / 00:03:09
ACTUAL_ALLOCATION: 189 / 3600 = 0.052500 GH200-hours
POSTJOB_CUMULATIVE_S10_ABC/C1: 3.282778 + 0.052500 = 3.335278 GH200-hours
FOCUSED_TESTS: 37 passed / 0 failed / 0.93 s
SOURCE/RUNTIME/DATA/PANEL/CONFIG: all identity gates passed
CANDIDATE_MAPPING: 42 reported running mean/variance keys; 21 synthesized zero batch counters; exact shared convolution/affine W0
RUNS: 128/128; 32 per candidate/pathway cell; physical B4; all SECOND gradients finite
OPTIMIZER/UPDATE/EVALUATOR: absent / zero / absent
PARAMETER_STATE: exact candidate parity before runs; both candidates unchanged after runs
STATUS/VERDICT: PASS / LOCALIZED_NORM
WALL_SECONDS_MODEL_MATRIX: 83.751673
PEAK_ALLOCATED/RESERVED_BYTES: 6235769344 / 8657043456
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1a_fix_d713bfe_o136_a1
SUMMARY_SHA256: f00a8b0740b591b694f1f0295432b056cd2f630b3d0ed129450afe05a87762fa
CANDIDATE_IDENTITY_SHA256: 59b241d02b374e73034fd195efbd2fe7add313613af5c5b5315dbb38a065a441
EXECUTION_IDENTITY_SHA256: 72c9ffb4e69b20725a89a23a980dde00e46860d2871ea1928de9e7d35f22eaf4
RUNS_SHA256: 33a748e283e9d17f08e7faf3d686463d5935b3ac3193a2afb3cf51138cf5213f
INNER_ARTIFACT_MANIFEST_SHA256: f38df3e9e43ef591650a0ae602336a94f98c2cf8a94f6ed4ada88aa34893291a
RUNNER_ARTIFACT_MANIFEST_SHA256: b14dcd340fe6cf8f69251f235c90ef2c692cbf282478b757ea0fde574fb44bb2
FINAL/TEST/C1A_EXIT: 0 / 0 / 0
ARTIFACT_CHECKS: runner 15/15 OK; output recursively read-only; zero group/other-writable paths
RETRY/C1-B: none authorized
```

## 32. O-137 C1-B0 matched GN/BN1d fusion health — approved / tuple freeze pending

```text
REQUEST_ID: S10-C1B0-FUSION-NORM-HEALTH-H256-v1
OWNER_DECISION: O-137
STATE: consumed once by Job 502958 / pre-model focused-test failure / no retry
CELLS_SERIAL: F-A1-GN-H256; F-A1-BN1D-H256
GRAPH: current F-A1 fusion graph; only SECOND normalization differs
INITIALIZATION: ImageNet1K V1 camera + exact shared seed-0 random LiDAR/fuser/head trainable W0
DATA_ROLE: accepted STOP-A D_low only
TOKEN_SELECTION: lowest SHA256("s10-c1b0-h256-v1\\0" || token), label-blind
TOKENS/BATCHES: 1024 unique samples / 256 ordered physical-B4 batches
ORDERED_TOKEN_SHA256: 62a096c0990e6d1d0932868a882b2418e731d1a816f481e741996e49c8e975f7
SEED: 0
PRECISION: global FP16 + explicit SECOND FP32 island
COMMON_SCALE: GradScaler init 32; no-update first-B4 qualification for both candidates; discard and reconstruct W0
OPTIMIZER: AdamW lr=1e-4 weight_decay=0.01
SCHEDULER: constant LambdaLR(1.0), per accepted update
ABSENT: augmentation / EMA / CBGS / GT-paste / gradient clipping
HORIZON: exactly 256 real optimizer updates per cell; any skipped/invalid/discarded window is hard failure
DIAGNOSTICS: true-unscaled parameter and explicit SECOND/head-boundary gradients + realized updates at 1/4/16/64/128/256; BN running state; all-window loss/scaler/tokens/outcome/basic timing; peak memory
HARD_GATE: exact source/config/W0/token identity; common-scale qualification; complete matched 256 updates each; finite loss/gradients; required camera/LiDAR/fusion/head gradients; counters/state/artifacts
SCIENTIFIC_GATE: descriptive only; no old GN numeric threshold and no automatic winner
EVALUATOR/CHECKPOINT_SELECTION/PROFILER: absent / absent / absent
RESOURCE: 1 node / 1 x GH200 / 16 CPUs / 96 GiB / 00:30:00 / no-requeue
EXPECTED/CAP: 0.20-0.30 / 0.5 elapsed GH200-hour
SUBMISSIONS: exactly one; no retry, requeue, array, DDP or spare GPU
OUTPUT: fresh path frozen with exact tuple; recursively read-only at completion or failure
CUMULATIVE_BEFORE: 3.335278 GH200-hours
CUMULATIVE_HARD_MAX_AFTER: 3.835278 GH200-hours
FORBIDDEN: full D_low, D_select/D_audit/official val, checkpoint selection, C1-B1/A2/MIT repair, TransFusion, DepthLSS, recipe search, reviewer chain, STOP-D/E/F, merge, push, upload
```

The implementation must add the normalization choice through resolved production
config and detector construction while keeping historical S09 configs on GN. The
resolved config hash plus strict model state must make GN/BN1d checkpoint
interchange fail closed. Focused tests cover schema rejection/roundtrip,
production propagation, 63 BN running buffers and strict cross-load failure,
precision diagnostics/checkpoint compatibility, the fixed H256 selector and the
existing sparse runtime contract.

O-137 is a one-job authority with no remediation loop. A pre-model test or
identity failure, qualification failure, runtime correctness failure, timeout or
artifact failure consumes the submission and returns directly to the owner. A
finite but scientifically weak trajectory is a completed negative observation;
it is not permission to tune, rerun or continue to C1-B1.

### Exact immutable tuple

```text
SOURCE_SHA: 96ae63d69ca9e5c95f528dd8c4e5bbcf934ac0c4
SOURCE_TREE: 0346754de0000eff5c7b521c5ddf6790afc2a28e
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_96ae63d_o137
SNAPSHOT_MODE: detached / clean / recursively read-only
RUNNER_SHA256: 6c8a284e4d6c760b07b3b1566a686d2a65a920b39cd298249433652bc49762ce
ENTRY_SHA256: 2007fb436bc11015d6578e112af666275f9b67614ea337ba7e19720b3fc8dd94
CONFIG_SHA256: dfb9e05e43444fed632d08d8206383dbedd575ab5c5d3fa4db2d684668dada70
RESOLVED_GN_SHA256: 67f8ebb8e5fce20794d14ffaae9e8a72e4332934dba9e6dc879dbb9d80df8313
RESOLVED_BN1D_SHA256: 3b3a0069862136211967718bfa767cacf4e1928f142015ca0718f5579cba221d
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
H256_ORDERED_TOKEN_SHA256: 62a096c0990e6d1d0932868a882b2418e731d1a816f481e741996e49c8e975f7
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_96ae63d_o137_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_96ae63d_o137_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_96ae63d_o137_%j.err
```

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1b0-96ae63d --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_96ae63d_o137 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_96ae63d_o137_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_96ae63d_o137_%j.err --export=ALL,S10_C1B0_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_96ae63d_o137,S10_C1B0_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_96ae63d_o137_a1,S10_C1B0_EXPECTED_SOURCE_SHA=96ae63d69ca9e5c95f528dd8c4e5bbcf934ac0c4,S10_C1B0_EXPECTED_TREE=0346754de0000eff5c7b521c5ddf6790afc2a28e,S10_C1B0_EXPECTED_RUNNER_SHA256=6c8a284e4d6c760b07b3b1566a686d2a65a920b39cd298249433652bc49762ce,S10_C1B0_EXPECTED_ENTRY_SHA256=2007fb436bc11015d6578e112af666275f9b67614ea337ba7e19720b3fc8dd94,S10_C1B0_EXPECTED_CONFIG_SHA256=dfb9e05e43444fed632d08d8206383dbedd575ab5c5d3fa4db2d684668dada70 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_96ae63d_o137/fl_v3/scripts/run_s10_c1b0_fusion_health.sh
```

### Consumed outcome and exact diagnosis

```text
JOB: 502958
STATE/EXIT: FAILED / 1:0
ELAPSED/ALLOCATED: 00:02:14 / 0.037222 GH200-hour
NODE/RESTARTS: n124 / 0
FOCUSED_TESTS: 100 passed / 6 failed / 6 warnings / 69.51s
TEST_EXIT/FINAL_EXIT: 1 / 1
MODEL/H256/OPTIMIZER/CELLS: not constructed / not read / not constructed / none
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_96ae63d_o137_a1
FOCUSED_STDOUT_SHA256: f67c5d403c522d5cac66086bb079792b99ed5760cb1c77a22c1983e37b7e79b0
FOCUSED_STDERR_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
RUNNER_MANIFEST_SHA256: 689ab20f19d61c3fd9ffc979480ec9d387f9c55449359065013a1008563082ac
RUNNER_MANIFEST_CHECK: 4/4 OK
CUMULATIVE_AFTER: 3.335278 + 0.037222 = 3.372500 GH200-hours
```

Five failures are the same test-layout defect: three lines belonging to the end
of the preceding S09-v2 operator-profile hash test were left inside the new
parameterized S10 rejection test after its expected `pytest.raises` block. The
expected exception succeeds, then an unintended second resolve reuses the still-
invalid config. The sixth failure is an incomplete test fixture: it switches to
`s10.v1` and BN1d but omits required `training.grad_scaler_init_scale=32`.

The exact neutral correction would move those three lines back before the next
test definition and add the one missing fixture field. No production source,
config, runner, data, candidate, seed, horizon, gate or resource change is
indicated. That diagnosis is recorded only; O-137 authorizes no edit, commit,
snapshot or replacement execution after this failure.

The verdict is bounded to the exact random W0 and frozen STOP-B panel. It
localizes a causal normalization-path contribution because BN1d reduces both
the fixed-upstream SECOND Jacobian gradients and normal-loss gradients beyond
repeat variation. It neither proves BN1d convergence/capability nor selects it
for production; those require a separately approved training/evaluation gate.

## 33. O-138 strictly derived C1-B0 test-only replacement — consumed / pre-model FAIL

```text
REQUEST_ID: S10-C1B0-FUSION-NORM-HEALTH-H256-v1-fix1
OWNER_DECISION: O-138
STATE: consumed once by Job 503075 / pre-model focused-test failure / no retry
CAUSE: Job 502958 pre-model test-only fixture/layout failure
ALLOWED_DIFF: move three operator-profile hash assertions to their original S09-v2 test; add grad_scaler_init_scale=32 to one migrated s10.v1 fixture
PRODUCTION_SOURCE/RUNNER/CONFIG: unchanged from O-137
DATA/CELLS/SEED/W0/PRECISION/RECIPE/HORIZON/DIAGNOSTICS/GATES: unchanged from O-137
RESOURCE: 1 node / 1 x GH200 / 16 CPUs / 96 GiB / 00:30:00 / no-requeue
EXPECTED/CAP: 0.20-0.30 / 0.5 elapsed GH200-hour
SUBMISSIONS: exactly one fresh replacement; no retry, requeue, array, DDP or spare GPU
CUMULATIVE_BEFORE: 3.372500 GH200-hours
CUMULATIVE_HARD_MAX_AFTER: 3.872500 GH200-hours
FORBIDDEN: any production/scientific/resource change, C1-B1, later C/D/E/F, reviewer chain, merge, push, upload
```

The replacement may be frozen only after the diff is shown to contain exactly
the two test corrections plus canonical authority records and after local/static
config identity checks pass. It uses one new detached clean recursively
read-only snapshot, fresh output/log paths and exact hashes. Any failure consumes
O-138 and returns directly to the owner; no automatic remediation or replacement
is authorized.

### Exact immutable replacement tuple

```text
SOURCE_SHA: 0f51e11c9f879f5bcb9ab2632bcee31969e5c0ac
SOURCE_TREE: 90e551b98c0c429e757345c244426433ebe84b62
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix_0f51e11_o138
SNAPSHOT_MODE: detached / clean / recursively read-only
RUNNER_SHA256: 6c8a284e4d6c760b07b3b1566a686d2a65a920b39cd298249433652bc49762ce
ENTRY_SHA256: 2007fb436bc11015d6578e112af666275f9b67614ea337ba7e19720b3fc8dd94
CONFIG_SHA256: dfb9e05e43444fed632d08d8206383dbedd575ab5c5d3fa4db2d684668dada70
RESOLVED_GN_SHA256: 67f8ebb8e5fce20794d14ffaae9e8a72e4332934dba9e6dc879dbb9d80df8313
RESOLVED_BN1D_SHA256: 3b3a0069862136211967718bfa767cacf4e1928f142015ca0718f5579cba221d
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
H256_ORDERED_TOKEN_SHA256: 62a096c0990e6d1d0932868a882b2418e731d1a816f481e741996e49c8e975f7
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_fix_0f51e11_o138_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix_0f51e11_o138_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix_0f51e11_o138_%j.err
```

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1b0-fix-0f51e11 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix_0f51e11_o138 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix_0f51e11_o138_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix_0f51e11_o138_%j.err --export=ALL,S10_C1B0_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix_0f51e11_o138,S10_C1B0_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_fix_0f51e11_o138_a1,S10_C1B0_EXPECTED_SOURCE_SHA=0f51e11c9f879f5bcb9ab2632bcee31969e5c0ac,S10_C1B0_EXPECTED_TREE=90e551b98c0c429e757345c244426433ebe84b62,S10_C1B0_EXPECTED_RUNNER_SHA256=6c8a284e4d6c760b07b3b1566a686d2a65a920b39cd298249433652bc49762ce,S10_C1B0_EXPECTED_ENTRY_SHA256=2007fb436bc11015d6578e112af666275f9b67614ea337ba7e19720b3fc8dd94,S10_C1B0_EXPECTED_CONFIG_SHA256=dfb9e05e43444fed632d08d8206383dbedd575ab5c5d3fa4db2d684668dada70 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix_0f51e11_o138/fl_v3/scripts/run_s10_c1b0_fusion_health.sh
```

### Consumed replacement outcome

```text
JOB: 503075
STATE/EXIT: FAILED / 1:0
ELAPSED/ALLOCATED: 00:02:11 / 0.036389 GH200-hour
NODE/RESTARTS: n120 / 0
FOCUSED_TESTS: 105 passed / 1 failed / 6 warnings / 72.95s
TEST_EXIT/FINAL_EXIT: 1 / 1
MODEL/H256/OPTIMIZER/CELLS: not constructed / not read / not constructed / none
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_fix_0f51e11_o138_a1
FOCUSED_STDOUT_SHA256: dbe3170d178cab2dbaa6792e1408832085780925e7f630f745ce880c69d7d0d6
FOCUSED_STDERR_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
RUNNER_MANIFEST_SHA256: 67499dd0aa9ed750fc81db2010451c8c83587f33f11fe2a87b096e68e51f6d72
RUNNER_MANIFEST_CHECK: 4/4 OK
CUMULATIVE_AFTER: 3.372500 + 0.036389 = 3.408889 GH200-hours
```

The sole failure is the same fixture's second incomplete `s10.v1` migration:
`execution.operator_profile` is absent. Job `502958` stopped first on the missing
`training.grad_scaler_init_scale`, so config resolution never reached and exposed
this additional required field. A technically sufficient next correction would
add the explicit operator-profile value appropriate to this fixture, but O-138
authorizes no edit or replacement. This result remains test-infrastructure
failure evidence only and says nothing about GN/BN1d fusion training health.

## 34. O-139 canonical-fixture C1-B0 replacement — consumed / execution PASS

```text
REQUEST_ID: S10-C1B0-FUSION-NORM-HEALTH-H256-v1-fix2
OWNER_DECISION: O-139
STATE: consumed once by successful Job 504508; no retry/C1-B1
CAUSE: Jobs 502958/503075 used one partial manual s09.v1-to-s10.v1 test promotion
ALLOWED_DIFF: replace that manual promotion with s10_second_config(..., batch_norm_1d) and assert complete resolved propagation
PRODUCTION_SOURCE/RUNNER/CONFIG: unchanged from O-137
DATA/CELLS/SEED/W0/PRECISION/RECIPE/HORIZON/DIAGNOSTICS/GATES: unchanged from O-137
RESOURCE: 1 node / 1 x GH200 / 16 CPUs / 96 GiB / 00:30:00 / no-requeue
EXPECTED/CAP: 0.20-0.30 / 0.5 elapsed GH200-hour
SUBMISSIONS: exactly one fresh replacement; no retry, requeue, array, DDP or spare GPU
CUMULATIVE_BEFORE: 3.408889 GH200-hours
CUMULATIVE_HARD_MAX_AFTER: 3.908889 GH200-hours
FORBIDDEN: any production/scientific/resource change, C1-B1, later C/D/E/F, reviewer chain, merge, push, upload
```

The pre-freeze audit must directly construct and resolve the canonical test
fixture outside pytest, verify both real production GN/BN1d resolved identities,
prove no manual S10 schema promotion remains in the propagation test, and show
zero production/runner/config diff from O-137. The one integrated job runs all
106 focused tests first and enters H256 execution only after 106/106 pass. Any
failure consumes O-139 and returns directly to the owner.

### Exact immutable replacement tuple

```text
SOURCE_SHA: 5de019bf36b1dd5ca077a5a10eaa5e0e5f376ca2
SOURCE_TREE: bd5d1c688bf4c25e05856351c0fcb48ce1b6c722
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix2_5de019b_o139
SNAPSHOT_MODE: detached / clean / recursively read-only
RUNNER_SHA256: 6c8a284e4d6c760b07b3b1566a686d2a65a920b39cd298249433652bc49762ce
ENTRY_SHA256: 2007fb436bc11015d6578e112af666275f9b67614ea337ba7e19720b3fc8dd94
CONFIG_SHA256: dfb9e05e43444fed632d08d8206383dbedd575ab5c5d3fa4db2d684668dada70
RESOLVED_GN_SHA256: 67f8ebb8e5fce20794d14ffaae9e8a72e4332934dba9e6dc879dbb9d80df8313
RESOLVED_BN1D_SHA256: 3b3a0069862136211967718bfa767cacf4e1928f142015ca0718f5579cba221d
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
H256_ORDERED_TOKEN_SHA256: 62a096c0990e6d1d0932868a882b2418e731d1a816f481e741996e49c8e975f7
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_fix2_5de019b_o139_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix2_5de019b_o139_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix2_5de019b_o139_%j.err
```

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1b0-fix2-5de019b --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix2_5de019b_o139 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix2_5de019b_o139_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b0_fix2_5de019b_o139_%j.err --export=ALL,S10_C1B0_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix2_5de019b_o139,S10_C1B0_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_fix2_5de019b_o139_a1,S10_C1B0_EXPECTED_SOURCE_SHA=5de019bf36b1dd5ca077a5a10eaa5e0e5f376ca2,S10_C1B0_EXPECTED_TREE=bd5d1c688bf4c25e05856351c0fcb48ce1b6c722,S10_C1B0_EXPECTED_RUNNER_SHA256=6c8a284e4d6c760b07b3b1566a686d2a65a920b39cd298249433652bc49762ce,S10_C1B0_EXPECTED_ENTRY_SHA256=2007fb436bc11015d6578e112af666275f9b67614ea337ba7e19720b3fc8dd94,S10_C1B0_EXPECTED_CONFIG_SHA256=dfb9e05e43444fed632d08d8206383dbedd575ab5c5d3fa4db2d684668dada70 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b0_fix2_5de019b_o139/fl_v3/scripts/run_s10_c1b0_fusion_health.sh
```

### Consumed successful outcome

```text
JOB: 504508
STATE/EXIT: COMPLETED / 0:0
ELAPSED/ALLOCATED: 00:06:57 / 0.115833 GH200-hour
NODE/RESTARTS: n124 / 0
FOCUSED_TESTS: 106 passed / 0 failed / 6 warnings / 30.36s
CELLS: F-A1-GN-H256 PASS; F-A1-BN1D-H256 PASS
UPDATES/TOKENS: 256 and 1024 per cell; exact matched order and shared W0
OVERFLOW/INVALID/NONFINITE/DISCARDED: 0/0/0/0 in both cells
FINAL_SCALE: 32 in both cells
FINAL/TEST/C1B0_EXIT: 0 / 0 / 0
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b0_fix2_5de019b_o139_a1
SUMMARY_SHA256: 08036a2ab77e25f18a657713791eeefd58274a24b95d1d2c7e584e2dd5c9b01c
EXECUTION_IDENTITY_SHA256: 3200db37059e3b648d5ce352149aeb40c28cdf63fbb2f211010c814be2b2b80e
INNER_ARTIFACT_MANIFEST_SHA256: b53bb636130d1b700c4ce08a0f6ac5d93d27f4dc1ad6c882369cc23556cfe598
RUNNER_ARTIFACT_MANIFEST_SHA256: e8657ccf136c438f6d99055bd7ef4008642da343be3f3cb6e59414d3efe66a76
ARTIFACT_CHECKS: inner 8/8 OK; runner 17/17 OK; recursively read-only
CUMULATIVE_AFTER: 3.408889 + 0.115833 = 3.524722 GH200-hours
```

Both candidates are numerically and optimizer-update healthy over this bounded
H256 observation. BN1d preserves the C1-A mechanism in the trained fusion graph,
reducing sampled LiDAR-stem gradient L2 by 217-1904x. AdamW realized stem updates
remain within roughly `0.81-1.54x` of GN at the same sampled windows, and GN has
no overflow or invalid update. BN1d starts at higher loss and remains higher on
251/256 matched windows, including last-16 mean `22.893` versus `21.751`, while
its descriptive throughput is about 1.41x higher. This neither promotes BN1d nor
clears GN as a full-run recipe: capability/evaluator evidence is absent.

## 35. O-140 C1-B1 current-A1 matched capability — consumed / FAIL-INCOMPLETE

```text
REQUEST_ID: S10-C1B1-CUR-A1-GN-BN1D-DLOW-v1
OWNER_DECISION: O-140
STATE: Job 504921 consumed the sole tuple; FAIL-INCOMPLETE; no execution authority
CELLS: C1-B1-CUR-A1-GN-DLOW -> C1-B1-CUR-A1-BN1D-DLOW, serial
GRAPH/INIT: current A1 fusion; shared exact seed-0 trainable W0; ImageNet1K V1 camera
NORMALIZATION: group_norm versus batch_norm_1d only
TRAIN: exact frozen D_low / one epoch / physical B4 / drop_last=true / 1,538 attempted and accepted updates
EVAL: terminal raw checkpoints only; exact D_select / 4,626 samples / eight frozen logs
PRECISION: global FP16 plus SECOND FP32 island / GradScaler initial scale 32
RECIPE: AdamW lr=1e-4 wd=.01 / constant scheduler / no clip, EMA, augmentation, CBGS or GT paste
PRIMARY: internal D_select NDS
GUARDRAILS: mAP, per-class AP, zero invalid/nonfinite/overflow/discard, exact W0/token/remainder, paired leave-one-log-out jackknife
SCIENTIFIC_SELECTION: OWNER_DECISION_REQUIRED; no numeric promotion/non-inferiority margin is approved
RESOURCE: 1 node / 1 x GH200 / 16 CPUs / 96 GiB / 01:00:00 / no-requeue
EXPECTED/CAP: 0.7-1.0 / 1.0 elapsed GH200-hour
SUBMISSIONS: exactly one; no retry, requeue, DDP, array or spare GPU
CUMULATIVE_BEFORE: 3.524722 GH200-hours
CUMULATIVE_HARD_MAX_AFTER: 4.524722 GH200-hours
FORBIDDEN: intermediate checkpoint selection, extra seed, D_audit, official val, A2, MIT repair, TransFusion, DepthLSS, recipe search, later C/D/E/F, reviewer chain, merge, push, upload
```

The runner reuses the accepted C0-v2 `D_select` decode/submission/evaluator path
and verifies repeated full-evaluator parity before constructing paired-log
evidence. Leave-one-log-out calculations remove one frozen D_select log at a
time from each already filtered GN/BN1d prediction/ground-truth set and use the
same official devkit metric math. The 95% interval is descriptive uncertainty,
not a hidden winner gate. Execution PASS requires both complete matched
training/evaluation cells; scientific weakness is retained and returned to the
owner rather than tuned or relabeled.

### Exact immutable tuple

```text
SOURCE_SHA: 239cd6260c42b53e63d5e229493bbf47c4a41915
SOURCE_TREE: 3affb50159884382556b5174c4d2ffc343cc365c
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b1_239cd62_o140
SNAPSHOT_MODE: standalone clone; detached HEAD; clean; recursively read-only; zero group/other-writable paths
RUNNER_SHA256: ed19dbb9db7f1e162afa6dd69ed51eac45beb5e97a9a7bde2cec276dca130dde
ENTRY_SHA256: 9c9d0f325a0491ecfe2b3b58cf8a1253be2013878ec687ea37afe60cb9fdae1e
CONFIG_SHA256: b5b60a8b21b8f578b16f582044bc29d297a3e9abae49ec83d297907c1c1f7896
RESOLVED_GN_SHA256: eec07861f0fb0403a4eae5795e88d00b31507b1ecbb6ddb54d122bdaba9bdc82
RESOLVED_BN1D_SHA256: d11d632b9e65c6ad053d2c2413bba9aa9ddbfe6580280e080e58be4900346bf6
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
SWIN_WEIGHTS_SHA256: 704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b1_239cd62_o140_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b1_239cd62_o140_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b1_239cd62_o140_%j.err
```

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=01:00:00 --no-requeue --job-name=s10-c1b1-239cd62 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b1_239cd62_o140 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b1_239cd62_o140_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b1_239cd62_o140_%j.err --export=ALL,S10_C1B1_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b1_239cd62_o140,S10_C1B1_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b1_239cd62_o140_a1,S10_C1B1_EXPECTED_SOURCE_SHA=239cd6260c42b53e63d5e229493bbf47c4a41915,S10_C1B1_EXPECTED_TREE=3affb50159884382556b5174c4d2ffc343cc365c,S10_C1B1_EXPECTED_RUNNER_SHA256=ed19dbb9db7f1e162afa6dd69ed51eac45beb5e97a9a7bde2cec276dca130dde,S10_C1B1_EXPECTED_ENTRY_SHA256=9c9d0f325a0491ecfe2b3b58cf8a1253be2013878ec687ea37afe60cb9fdae1e,S10_C1B1_EXPECTED_CONFIG_SHA256=b5b60a8b21b8f578b16f582044bc29d297a3e9abae49ec83d297907c1c1f7896,S10_C1B1_EXPECTED_GN_RESOLVED_SHA256=eec07861f0fb0403a4eae5795e88d00b31507b1ecbb6ddb54d122bdaba9bdc82,S10_C1B1_EXPECTED_BN_RESOLVED_SHA256=d11d632b9e65c6ad053d2c2413bba9aa9ddbfe6580280e080e58be4900346bf6 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b1_239cd62_o140/fl_v3/scripts/run_s10_c1b1_capability.sh
```

### Consumed outcome — C1-B1 FAIL/INCOMPLETE

```text
JOB: 504921
STATE/EXIT: FAILED / 1:0
ELAPSED/ALLOCATED: 00:47:01 / 0.783611 GH200-hour
NODE/RESTARTS: n144 / 0
FOCUSED_TESTS: 117 passed / 3 skipped / 6 warnings / 69.75s
GN_UPDATES/BN1D_UPDATES: 1538/1537
GN_OVERFLOW/BN1D_OVERFLOW: 0/1 (BN first actual B4; nine +Inf head parameter-gradient elements)
BN_SCALER: 32 -> 16 on first window; 1,537 subsequent accepted windows
ATTEMPTED_TOKENS/REMAINDER: exact matched 6,152 / exact matched 3
GN_NDS/mAP: 0.1444747929 / 0.0615530756
BN1D_NDS/mAP: 0.1367052180 / 0.0531247739
BN1D_MINUS_GN: -0.0077695749 NDS / -0.0084283017 mAP
PAIRED_LOG_EVIDENCE/SUMMARY: absent / absent; fail-closed stop before construction
FINAL/TEST/C1B1_EXIT: 1 / 0 / 1
CUMULATIVE_AFTER: 3.524722 + 0.783611 = 4.308333 GH200-hours
ACTIVE_ABC_REMAINDER: 22.691667 GH200-hours under 27-hour aggregate; not execution authority
```

The failure is the exact predeclared matched-update gate, not split/evaluator or
model-construction failure. Both terminal checkpoints and full D_select metric
artifacts exist, but the BN1d checkpoint saw one fewer accepted B4 update. The
raw point estimates broadly favor GN (BN1d improves motorcycle/truck AP, ties
two zero-AP classes and is lower on the other six), while BN1d remains faster
and its sampled LiDAR-stem gradients are 146-2647x lower. Without matched
accepted exposure and the planned paired-log uncertainty, this does not promote
GN, reject BN1d, or complete C1-B1.

```text
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b1_239cd62_o140_a1
EXECUTION_IDENTITY_SHA256: 81cf61c51eeb050d2712d1cc544cbec6ba3df61eeae20202e5ce60386f0f663f
GN_CELL_SHA256: 81e31258dd783f47e8775a1b1327dbac66b0cf9b005fcf6e5f4249c98d61ea85
BN1D_CELL_SHA256: 5abc990577ec99eb04f2b9fc063ecba648d342891ce3e50159b4adff62537517
FAILURE_SUMMARY_SHA256: d755996487ae54c36da6bb566b661e178d186017883e7cda32ac12c11d401dfd
INNER_ARTIFACT_MANIFEST_SHA256: d6f6f17fb12e37aaec891f98328d1ed8cdb14badca596b3c28de4cd81072137a
RUNNER_ARTIFACT_MANIFEST_SHA256: bdf7d94f30b1bca4912495d7b03f298d23c05c0316a7e17d03e031b80783777c
ARTIFACT_CHECKS: inner 14/14 and runner 23/23 OK; recursively read-only
```

O-140 is consumed and supplies no correction, replay, paired-log postprocess
allocation or later-stop continuation.

## 36. O-141 BN1d physical-B8 operational candidate — consumed / pre-model FAIL

```text
REQUEST_ID: S10-C1B1-BN1D-B8-v1
OWNER_DECISION: O-141
STATE: Job 505266 consumed the sole tuple; pre-model assertion FAIL; no retry
CELL: C1-B1-BN1D-B8-DLOW only
GRAPH/INIT: current A1 fusion / BN1d only in SECOND / ImageNet camera / exact seed-0 trainable W0 87be0d...829d1
TRAIN_ROLE: exact frozen D_low / 6,155 samples / shuffle seed 0 / drop_last=true
TRAIN: physical B8 / accumulation1 / effective B8 / one epoch / 769 attempted=accepted updates
TOKEN_BINDING: same ordered 6,152 consumed tokens and same three-token remainder as sealed Job 504921 B4 cells
PRECISION: global FP16 + SECOND FP32 island / GradScaler initial and final scale 8
EARLY_NUMERIC_GATE: boundaries 1/4/16/64; then 256/512/769; fail on any overflow/nonfinite/invalid/discard/missing update
RECIPE: AdamW lr=1e-4 wd=.01 / constant scheduler / no clip, EMA, augmentation, CBGS or GT-paste
CHECKPOINT: terminal raw only
EVAL: exact frozen D_select / 4,626 samples / eight logs / physical eval B4
COMPARATORS: sealed GN-B4 complete + sealed BN1d-B4 incomplete artifacts from Job 504921
REPORT: NDS/mAP/per-class, paired delete-one-log uncertainty, loss trajectory, wall throughput, GPU telemetry, peak memory
CLAIM: joint BN1d+B8+scale8 operational candidate only; not isolated batch-size causality or automatic architecture selection
RESOURCES: one GH200 / 16 CPUs / 96 GiB / 00:30:00 / hard cap 0.5 GH200-hour
EXPECTED: 0.30-0.40 GH200-hour
SUBMISSIONS: exactly one / no retry / no requeue
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_e4a9ff4_o141_a1
CUMULATIVE_BEFORE: 4.308333 GH200-hours
CUMULATIVE_HARD_MAX_AFTER: 4.808333 GH200-hours
FORBIDDEN: GN rerun, B16, extra seed, profiler, recipe search, D_audit, official val, A2/MIT repair, later C/D/E/F, reviewer chain, merge, push, upload
```

### Exact immutable tuple

```text
SOURCE_SHA: e4a9ff4d44014b0ba0e2e6ffabc375b5be6f6c17
SOURCE_TREE: 1d9ea6ef1098e480f25cc2cf041a5ac683698f9b
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_e4a9ff4_o141
SNAPSHOT_MODE: standalone clone; detached HEAD; clean; recursively read-only; zero group/other-writable paths
RUNNER_SHA256: 0e1a15e5e34ce54fabd94e02136fe9ed593468742b667b31d8e1407400c7f26c
ENTRY_SHA256: 704f73bc012cb24ca7a95570a14e93e203b01cebd8dfd2ae356a0016cfa1a0df
CONFIG_SHA256: 265480319a60053ad67a0e4f7b7b722fca0630708293f1e2231d4981d7826202
RESOLVED_CONFIG_SHA256: 2b9a3e850beedd133df269bb10571c2d2a58ea9bf05afb8e0cbebeab6ff16f71
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
GN_REFERENCE_SUMMARY_SHA256: 81e31258dd783f47e8775a1b1327dbac66b0cf9b005fcf6e5f4249c98d61ea85
GN_REFERENCE_RESULTS_SHA256: 7fc24fd757d9302096c27208c58469fdd335f22fe363a70bb32ab76875f1e549
BN_B4_REFERENCE_SUMMARY_SHA256: 5abc990577ec99eb04f2b9fc063ecba648d342891ce3e50159b4adff62537517
BN_B4_REFERENCE_RESULTS_SHA256: 124eddeee78d5fd3495a3f1cff820a5ab82f5aebaef4cfd4ab9996a926966268
SWIN_WEIGHTS_SHA256: 704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_e4a9ff4_o141_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_e4a9ff4_o141_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_e4a9ff4_o141_%j.err
```

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1b8-e4a9ff4 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_e4a9ff4_o141 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_e4a9ff4_o141_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_e4a9ff4_o141_%j.err --export=ALL,S10_C1B8_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_e4a9ff4_o141,S10_C1B8_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_e4a9ff4_o141_a1,S10_C1B8_EXPECTED_SOURCE_SHA=e4a9ff4d44014b0ba0e2e6ffabc375b5be6f6c17,S10_C1B8_EXPECTED_TREE=1d9ea6ef1098e480f25cc2cf041a5ac683698f9b,S10_C1B8_EXPECTED_RUNNER_SHA256=0e1a15e5e34ce54fabd94e02136fe9ed593468742b667b31d8e1407400c7f26c,S10_C1B8_EXPECTED_ENTRY_SHA256=704f73bc012cb24ca7a95570a14e93e203b01cebd8dfd2ae356a0016cfa1a0df,S10_C1B8_EXPECTED_CONFIG_SHA256=265480319a60053ad67a0e4f7b7b722fca0630708293f1e2231d4981d7826202,S10_C1B8_EXPECTED_RESOLVED_SHA256=2b9a3e850beedd133df269bb10571c2d2a58ea9bf05afb8e0cbebeab6ff16f71 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_e4a9ff4_o141/fl_v3/scripts/run_s10_c1b1_bn_b8.sh
```

### Consumed outcome — pre-model implementation assertion FAIL

```text
JOB: 505266
STATE/EXIT: FAILED / 1:0
ELAPSED/ALLOCATED: 00:02:15 / 0.037500 GH200-hour
NODE/RESTARTS: n172 / 0
FOCUSED_TESTS: 121 passed / 3 skipped / 6 warnings / 70.95s
ENTRY_FAILURE: AttributeError: 'ResolvedConfig' object has no attribute 'schema_version'
CORRECT_ACCESS: config.data["schema_version"]
MODEL/DATA/UPDATES/EVALUATOR: none / none / 0 / none
FINAL/TEST/C1B8_EXIT: 1 / 0 / 1
CUMULATIVE_AFTER: 4.308333 + 0.037500 = 4.345833 GH200-hours
ACTIVE_ABC_REMAINDER: 22.654167 GH200-hours under 27-hour aggregate; not execution authority
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_e4a9ff4_o141_a1
FAILURE_SUMMARY_SHA256: a38cfb69be45099fec8a1a13862cb222402ac7f5ff4a820757de60225a15b875
INNER_ARTIFACT_MANIFEST_SHA256: 40a032bc53348d6e443ca2f64a33b0a8435e30f2c1b5f799d4d6952e41b19841
RUNNER_ARTIFACT_MANIFEST_SHA256: 56f057c95cc345c0279ad2b97c223bbada1fc69434c359f87bc530b08dfb4274
ARTIFACT_CHECKS: runner 10/10 OK; recursively read-only
```

The failure is not BN-B8 numerical or capability evidence. The focused test
suite did not invoke the new `_assert_config` function, and the login-node
dependency gap prevented the attempted direct entry import from reaching it.
O-141 is consumed and authorizes no patch execution or replacement submission.

## 37. O-142 exact schema remediation and unchanged replacement — consumed / FAIL-INCOMPLETE

```text
REQUEST_ID: S10-C1B1-BN1D-B8-v1-R1
OWNER_DECISION: O-142
STATE: sole replacement consumed by Job 505316; no retry or post-processing authority
ONLY_CODE_CHANGE: config.schema_version -> config.data["schema_version"]
ONLY_TEST_CHANGE: resolve exact s10_c1b1_bn_b8.json and directly call _assert_config
SCIENTIFIC/RESOURCE_SCOPE: byte-for-byte O-141 runner/config semantics outside the exact correction; same §36 envelope
RESOURCES: one GH200 / 16 CPUs / 96 GiB / 00:30:00 / hard cap 0.5 GH200-hour
SUBMISSIONS: exactly one replacement / no retry / no requeue
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_864f704_o142_a1
CUMULATIVE_BEFORE: 4.345833 GH200-hours
CUMULATIVE_HARD_MAX_AFTER: 4.845833 GH200-hours
FORBIDDEN: any model/data/recipe/gate/evaluator/resource change; GN/B16/new seed/profiler/later stop/reviewer/merge/push/upload
```

### Exact immutable replacement tuple

```text
SOURCE_SHA: 864f704f5bdf1a63db8aba342778d6bf6d36fe57
SOURCE_TREE: b9c10ef88e331510361a680768963b4406b860a4
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_864f704_o142
SNAPSHOT_MODE: standalone clone; detached HEAD; clean; recursively read-only; zero group/other-writable paths
RUNNER_SHA256: 0e1a15e5e34ce54fabd94e02136fe9ed593468742b667b31d8e1407400c7f26c
ENTRY_SHA256: e6e3825cf65a516692fec664a8666bb19327bc5be53beb0f4ce22aa619088382
CONFIG_SHA256: 265480319a60053ad67a0e4f7b7b722fca0630708293f1e2231d4981d7826202
RESOLVED_CONFIG_SHA256: 2b9a3e850beedd133df269bb10571c2d2a58ea9bf05afb8e0cbebeab6ff16f71
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
GN_REFERENCE_SUMMARY_SHA256: 81e31258dd783f47e8775a1b1327dbac66b0cf9b005fcf6e5f4249c98d61ea85
GN_REFERENCE_RESULTS_SHA256: 7fc24fd757d9302096c27208c58469fdd335f22fe363a70bb32ab76875f1e549
BN_B4_REFERENCE_SUMMARY_SHA256: 5abc990577ec99eb04f2b9fc063ecba648d342891ce3e50159b4adff62537517
BN_B4_REFERENCE_RESULTS_SHA256: 124eddeee78d5fd3495a3f1cff820a5ab82f5aebaef4cfd4ab9996a926966268
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_864f704_o142_a1
STDOUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_864f704_o142_%j.out
STDERR: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_864f704_o142_%j.err
```

```bash
sbatch --account=naiss2025-22-1113-gpu --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G --gpus=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue --job-name=s10-c1b8-864f704 --chdir=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_864f704_o142 --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_864f704_o142_%j.out --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s10_c1b8_864f704_o142_%j.err --export=ALL,S10_C1B8_SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_864f704_o142,S10_C1B8_OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b8_864f704_o142_a1,S10_C1B8_EXPECTED_SOURCE_SHA=864f704f5bdf1a63db8aba342778d6bf6d36fe57,S10_C1B8_EXPECTED_TREE=b9c10ef88e331510361a680768963b4406b860a4,S10_C1B8_EXPECTED_RUNNER_SHA256=0e1a15e5e34ce54fabd94e02136fe9ed593468742b667b31d8e1407400c7f26c,S10_C1B8_EXPECTED_ENTRY_SHA256=e6e3825cf65a516692fec664a8666bb19327bc5be53beb0f4ce22aa619088382,S10_C1B8_EXPECTED_CONFIG_SHA256=265480319a60053ad67a0e4f7b7b722fca0630708293f1e2231d4981d7826202,S10_C1B8_EXPECTED_RESOLVED_SHA256=2b9a3e850beedd133df269bb10571c2d2a58ea9bf05afb8e0cbebeab6ff16f71 /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s10_c1b8_864f704_o142/fl_v3/scripts/run_s10_c1b1_bn_b8.sh
```

### Consumed outcome — scientific body complete, tail evidence gate timed out

```text
JOB: 505316
SACCT: TIMEOUT / top-level 0:0 / batch CANCELLED 0:9
ELAPSED: 00:30:07
CHARGED: 0.501944 GH200-hour
TESTS: 121 passed / 3 skipped / 6 warnings
TRAIN: 769/769 accepted physical-B8 updates; zero overflow/invalid/nonfinite/discarded
D_SELECT: exact 4,626 predictions + metrics + cell summary complete
MISSING: paired_vs_gn_b4.json; paired_vs_bn_b4.json; summary.json; artifact_sha256s.json; runner_artifact_sha256s.txt
POINT_METRICS: NDS 0.07840940894858875 / mAP 0.013023561548729617
VERDICT: FAIL/INCOMPLETE
CUMULATIVE_AFTER: 4.847777 GH200-hours
```

The job exhausted the wall limit inside paired-log post-processing after the
cell summary was durable. No retry, post-processing job, report-field rewrite,
or later-stop continuation is authorized. The raw cell report's
`recipe.physical_microbatch=4` is a legacy hardcoded display defect; resolved
config, cell declaration, actual token evidence and update/sample arithmetic
prove physical B8. Any code correction or salvage envelope requires a new owner
decision.
