# S10 results — STOP-A corrected protocol active; STOP-B/C unstarted

## STOP-A

```text
IMPLEMENTATION_SHA: e27053a5b141e1afaa68363ce6deb2efdb60518e
REMEDIATION_SHA: d7caf53414ade2d5db794ecd90851d0e5a3535b5
A_GATE_TUPLES: recorded in RUN_REQUEST.md
JOB: 463593 / FAILED 1:0 / 00:00:49 / zero restarts
JOB: 463649 / TIMEOUT 0:0 / 01:00:14 / zero restarts
JOB: 467862 / TIMEOUT 0:0 / 00:15:14 / zero restarts
RESULT: no accepted split/parity; exact split solve exceeded all frozen walltimes
LEGACY_A3: consumed under O-125; no retry or reinterpretation
CORRECTED_A1-A4: approved under O-126; evidence pending
INDEPENDENT_REVIEW: pending immutable PASS evidence
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
No corrected A-GATE result is recorded yet. The sole authorized allocation is
CPU-only (`0 GPU`, 4 CPU, 32 GiB, 15 minutes) and will appear here only after its
exact A2 tuple is frozen and actually executed.
