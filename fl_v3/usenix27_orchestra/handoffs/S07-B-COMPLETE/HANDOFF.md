# S07-B-COMPLETE HANDOFF — clean candidate with retired audit wrapper

## Identity and status

```text
BASE_SHA: 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
EXECUTABLE_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_TREE: ed2d4091f0098f6b2144028afd87e20d023b1da2
BRANCH: codex/s07-b-clean-completion
CURRENT_APPROVAL_SEAL: 1755734c4423488143e5d4adbffe57f22171dc01
STATUS: local/static ready; simplified GH200 validation approved exact once
COMPUTE_AUTHORITY: one exact command 229dfec34da4... submission; no retry
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

All approvals are consumed. No result is a C/L/F training PASS. Raw paths and
hashes are retained in `RESULTS.md` and `RUN_REQUEST.md`.

## Pending training-first gate

The exact unapproved command in `RUN_REQUEST.md` uses one GH200 and two phases:

1. four cases: clean profile plus one C/L/F B=1 fp16 optimizer update with
   `num_workers=0`;
2. one workers-0-versus-2 equality case with `TMPDIR=/tmp`.

No full cache/trainval, multiple steps, metrics, profile, Ray, DDP, matrix,
attack/defense, Protocol claim, upload, or publication is permitted. Training is
first so loader infrastructure cannot again mask the compiled model path.

## Next action

S00 completed syntax/hash/scope checks and the owner approved the exact simplified
envelope once on 2026-07-13. Submit only that command, record terminal evidence,
and do not retry. Review remains premature until the gate produces durable terminal
evidence.
