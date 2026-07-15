# S10 results — STOP-A/B/C active

## STOP-A

```text
IMPLEMENTATION_SHA: e27053a5b141e1afaa68363ce6deb2efdb60518e
A_GATE_TUPLE: frozen in RUN_REQUEST.md
JOB: 463593 / FAILED 1:0 / 00:00:49 / zero restarts
RESULT: focused-test solver-call dtype failure; no real split/parity/model path reached
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

No split, evaluator, model, recipe or performance result is accepted yet. A
strictly derived O-124 debug/fix replacement is pending freeze. STOP-B and STOP-C
have not started.
