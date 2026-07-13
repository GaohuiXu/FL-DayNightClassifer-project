# S07-B-COMPLETE HANDOFF — clean candidate with retired audit wrapper

## Identity and status

```text
BASE_SHA: 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
EXECUTABLE_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_TREE: ed2d4091f0098f6b2144028afd87e20d023b1da2
BRANCH: codex/s07-b-clean-completion
PRIOR_APPROVAL_SEAL: 1755734c4423488143e5d4adbffe57f22171dc01
DIAGNOSTIC_APPROVAL_SEAL: 9b23fabf33bde821a8053192566976b332f75c05
STATUS: FP32 final gate PASS; candidate ready for independent review
DIAGNOSTIC_COMMIT: 1900fe3bcb52ade22f0b947a2aca44d5ece12b2f
COMPUTE_AUTHORITY: consumed by Job 389356; no retry
```

The executable is a direct child of the accepted S07-C packet base and changes
only two paths:

- `fl_v3/configs/flwr_config.toml`: remove stale security/overcommit profiles and
  retain default clean local CPU/GPU profiles;
- `fl_v3/tests/test_s07_b_clean_completion.py`: five bounded cases covering the
  clean profile, workers 0/2 first-batch equality, and one C/L/F fp16 optimizer
  update each.

No production model, training, data, checkpoint, evaluation, environment,
dependency, launcher, or legacy security implementation changes.

## Owner-directed wrapper removal

The former 1,300-line `RUN_REQUEST.md` embedded wrapper and the associated
expanded HANDOFF/RESULTS narrative are retired. Git history and immutable job
output roots preserve their negative evidence. No active repository script was
part of that wrapper, so there is no source script to delete.

The replacement deliberately removes Git operations, source/archive manifests,
cumm/spconv checkout state checks, warnings-as-errors, long isolated TMPDIR/cache
trees, and the 205-case suite from GH200 execution. Source identity is fixed once
by a read-only snapshot created on the login node.

## Verification completed

- The executable diff is exactly the two declared config/test paths.
- Completion test SHA-256:
  `71d461eb3eb80a7e945ff4ae9e3fc8b07d7a99ed2b55b26a56d4e3c7ada4eef2`.
- Flower config SHA-256:
  `2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab`.
- Read-only snapshot:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_clean_simple_34cbe02b7b72`.
- Snapshot size/writable-file count: `628 KiB / 0`.
- Long-TMP Listener reproduction: deterministic `AF_UNIX path too long`.
- `/tmp` Listener reproduction: PASS with a 36-byte randomized address.

## Consumed compute evidence

| Job | Result | Scope reached |
|---|---|---|
| `372819` | failed wrapper | before environment activation |
| `373363` | failed wrapper | environment/spconv import, before pytest |
| `374142` | stopped wrapper | environment/identity/205 collection and clean-FedAvg profile PASS; worker=2 TMPDIR failure; no model update |
| `380806` | failed current training gate | environment and clean FedAvg PASS; all C/L/F reached finite-loss backward; norms `inf/nan/nan`; assertions stopped before durable step/skip counters |

All approvals are consumed. No result is a C/L/F optimizer-step PASS. Raw paths
and hashes are retained in `RESULTS.md` and `RUN_REQUEST.md`.

## Current training boundary

Job `380806` was the first current execution to enter all three complete model
paths. It disproves the one-attempt scale-512 update gate: C/L/F unscaled gradient
norms were respectively `inf`, `nan`, and `nan`; no successful-step evidence was
emitted. The raw output does not prove the per-mode step/skip counters because the
assertion precedes those checks. This does not invalidate the environment and does
not yet distinguish norm-reduction overflow, normal dynamic-scale backoff, or
persistent shared head/loss instability. Earlier
Arrhenius training exercised the pre-S07 model; reviewed S04/S06 artifacts did not
run this real six-task fp16 integration seam. S06 already tests that an overflow
window is skipped and a later batch can complete the optimizer budget; the
completion fixture's length-one iterable prevented that intended continuation.

## Next action

Preserve Job `380806` as the terminal result. Do not retry, change the default
GradScaler scale, weaken the finite-gradient/final-step gate, or launch review
from this failure. The smallest remediation is test-only: print the returned
metrics before assertions, distinguish element nonfiniteness from global-norm
reduction overflow, give the same mini batch a fixed small attempt budget, and
retain exact one-successful-step acceptance. Parameter-level diagnostics are
needed only if bounded backoff still fails. Any changed test or compute requires
a new exact owner-approved RUN_REQUEST.

The D1 test-only implementation is now prepared without production-source or
config changes. It compares FP32, fp16 scale 512 and fp16 scale 1 for each C/L/F
mode and records strict-JSON per-task/per-parameter evidence before gradients are
cleared. Candidate snapshot
`s07b_grad_diag_0ca44717e978` is read-only and differs from the Job `380806`
snapshot only in the focused completion test. Exact hashes, nine cells, one-GPU/
25-minute command and output root are recorded in `RUN_REQUEST.md`. Exact Job
`389356` completed all nine diagnostic cells. FP32 C/L/F and FP16-scale-1 C had
finite gradients and optimizer calls; FP16-scale-512 C/L/F and FP16-scale-1 L/F
had direct nonfinite elements and scaler skips. The remaining L/F failures begin
in sparse SECOND stem/stage1 parameters. This rejects scale 1 as a complete C/L/F
fix and makes the next decision precision-policy/remediation scope, not environment
repair. No changed command, retry, source remediation or review is authorized.

The owner subsequently closed precision comparison and scaler remediation as
out of scope for S07-B-COMPLETE. The prepared final test-only diff changes the
one-step acceptance case to explicit FP32, requires the reported precision to be
`fp32`, and requires GradScaler to be disabled. Its exact request runs only clean
FedAvg construction, three C/L/F successful-update cases and worker-0/2 first-
batch equality. It does not edit production source/config or select the precision
for later full scientific training. The immutable tuple is in `RUN_REQUEST.md`;
the exact test commit is `29ca6637bcd0a4e9a6422f3b820fb43d5295ad2c` and
the owner approved one submission after the docs-only seal. Job `390576` consumed
that approval and passed all five cases: plain FedAvg, three C/L/F successful FP32
updates and worker-0/2 first-batch equality. This is worker/runtime candidate
evidence, not independent-review or final integration acceptance.
