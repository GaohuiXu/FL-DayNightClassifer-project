# S10 RUN_REQUEST — approved B4-based ABC aggregate completion envelope

## 1. Current state

```text
SESSION_ID: S00-S10-STARTUP
REQUEST_ID: S10-ABC-COMPLETION-v1-B4-estimate
REQUEST_STATE: APPROVED AGGREGATE ENVELOPE / exact job tuples pending derivation
SUPERSEDES: S10-ABC-COMPLETION-v0-estimate — REJECTED by O-123
PLAN_AUTHORITY: O-122 scientific envelope + O-123 B4 minimum
EXECUTION_AUTHORITY: O-124
SOURCE_SHA: pending implementation SHA; current clean base was a080d49c1c22de20ccb5b1353d4922c7df14a729
BRANCH: codex/s10-cl-model-recipe
OWNER_APPROVAL: approved 2026-07-15; exact in-envelope tuples need no repeated approval
EXECUTABLE_NOW: only after each tuple is frozen below from an immutable SHA
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
| A — split/evaluator | 0 GPU-h when a compatible CPU compute node is available; otherwise about 0.5 | 1 GH200-h contingency | metadata/MILP/checker/devkit parity are CPU work; a GH200 allocation is used only if the validated aarch64 environment cannot run on CPU-only compute |
| B — observation-first | about 0.5–1 GH200-h | 2 GH200-h | B4 main panels/replays, one tiny paired B4-vs-four-B1 aggregation check, and at most one local boundary refinement |
| C — architecture/init | about 12–16 GH200-h | 24 GH200-h | every scientific training cell is B4; six low/three mid slot stress case, up to two donor lineages, reference-graph penalty, eval and bounded step-debug allowance included |
| **ABC total** | **about 13–18 GH200-h** | **27 GH200-h** | serial one-GPU execution; A's 1 h is contingency only |

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

The 27-hour ceiling is a safety ceiling, not a spending target. Unused time cannot
be transferred to another candidate, seed, rung, longer horizon, STOP-D/E/F, DDP,
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

- STOP-A: any input-identity mismatch, non-`OPTIMAL` solve, support failure,
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
TUPLE_STATE: FROZEN / executable once
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
