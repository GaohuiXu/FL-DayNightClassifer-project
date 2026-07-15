# S10 results — STOP-A/B/C active

## STOP-A

```text
IMPLEMENTATION_SHA: e27053a5b141e1afaa68363ce6deb2efdb60518e
A_GATE_TUPLE: frozen in RUN_REQUEST.md
JOB: 463593 / FAILED 1:0 / 00:00:49 / zero restarts
JOB: 463649 / TIMEOUT 0:0 / 01:00:14 / zero restarts
RESULT: no accepted split/parity; exact split solve exceeded the frozen walltime
INDEPENDENT_REVIEW: pending immutable evidence
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

An exact blocked-radix implementation and signal-safe runner remediation are in
local preparation. A further job is not authorized by the exhausted STOP-A
one-GH200-hour ceiling. STOP-B and STOP-C have not started.
