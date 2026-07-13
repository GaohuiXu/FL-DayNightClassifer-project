# S07-B-COMPLETE RESULTS — local/static only

## Overall result

**LOCAL/STATIC GATES PASS; DEPENDENCY-BACKED/GH200 GATES NOT RUN.**

No job was requested, approved, or submitted. There is no JUnit, model output,
metric, runtime execution identity, or Slurm job ID for this session.

## Startup and Git result

```text
toplevel = /home/gaohui/.codex/worktrees/328b/fl_weather_project
HEAD = 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
branch = <empty; detached>
startup status = <empty; clean>
expected = detached@4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
preflight mismatch/blocker = none
```

Actual topology checks proved `4aa2b133...` is a docs-only child of accepted
anchor `70bcd856...`; S07-C implementation/handoff are ancestors; the separate
review `b8e11bc...` adds only `S07-C/REVIEW.md` on its own branch and is not merged.

## Verification-first evidence

Before editing, `flwr_config.toml` had four execution profiles plus `supergrid` and
active T3/Path-A/Path-B/4-GPU/overcommit/`collab/**` wording. After the required
cleanup:

```text
FLWR_KEYS=default,local-simulation-cpu,local-simulation-gpu
FLWR_DEFAULT=local-simulation-cpu
CPU_RESOURCES=(8,1,0.0)
GPU_RESOURCES=(8,1,1.0)
banned profile/name/authority hits=0
```

The plain validation default was independently found in retained source/config:

```text
STRATEGY_CLASSES=CleanFedAvgStrategy
server-optimizer default=fedavg
server-ema-decay default=0.0
run_clean_round VAR_KEYWORD=False
run_clean_rounds VAR_KEYWORD=False
fp32_weighted_average present=True
```

The historical `s06_synthetic_camera.json` cleanly demonstrates fail-closed
schema behavior rather than a usable execution config:

```text
ConfigError: model keys invalid: missing=['camera_pretrained'], unknown=[]
```

It was not modified or selected for future execution because doing so would need
fabricated runtime/cache identities.

## Final login-safe checks

| Gate | Result |
|---|---|
| protected script set | PASS, exact `18/18` |
| S07-C tombstones | PASS, `70` deleted / `0` surviving |
| Python compile | PASS, `136/136` |
| JSON parse | PASS, `27/27` |
| TOML parse | PASS, `2/2` |
| retained shell `bash -n` | PASS, `17/17` |
| `git diff --check` | PASS |
| clean strategy/default AST/config | PASS |
| old Flower execution profiles | absent |
| new test skip/start-method scan | zero skip calls, zero explicit contexts |
| complete future command body `bash -n` | PASS |
| literal/Git-selected source manifest audit | PASS, exact sorted unique `100/100` |
| accepted spconv HEAD/path/diff/file identity | PASS, exact known build patch |
| dependency state-record hostile text injection | PASS, extra path and changed bytes rejected |

No workspace `__pycache__` was intentionally retained; the focused `py_compile`
used a temporary external `PYTHONPYCACHEPREFIX` and that temporary directory was
removed.

## Candidate source identities

```text
source files = 100
path-list sha256 = ce5c38764b43efa027b88b0b37de3a63407fb71ee9b0c5ad5bcd0671a0323ac4
source-record aggregate sha256 = acb80014ff8dd3ef123e689b3be34efae219c95c95ea63f64c36e28f6d546a9e
completion test sha256 = 71d461eb3eb80a7e945ff4ae9e3fc8b07d7a99ed2b55b26a56d4e3c7ada4eef2
Flower TOML sha256 = 2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
executable working patch sha256 = 98c0521973ab9963cbf3447618efbedcba7a2fc6807804da222976e5b90f1002
```

These hashes identify durable executable commit
`34cbe02b7b72114e3a2d61f6f797c8dec022798c`, tree
`ed2d4091f0098f6b2144028afd87e20d023b1da2`. Any later source/test/config change
invalidates them.

The S00 remediation removed the guaranteed post-zero-grad failure. The bounded
mode test now enables `telemetry_interval=1` and asserts/reports actual finite
positive `last_grad_norm`, exact optimizer/exposure counters, enabled GradScaler,
zero scaler skips, zero nonfinite losses, and the clean TrainingState boundary.
It neither clones the detector nor adds a parameter-delta sentinel.

The fully specified request body pins the exact executable SHA/tree with no
materialization sentinel, an exact 205-case pytest selection, an overall
50-minute timeout, warnings-as-errors, JUnit zero-skip postvalidation, original
pytest exit propagation, and in-job checksums. Its literal source manifest and
its Git reconstruction are byte-identical, sorted, unique, and all 100 paths
exist in the executable commit.

## Accepted dependency build-patch identity

Login-safe read-only inspection reconciled the current dependency sources with
the retained build contract at `fl_v3/scripts/build_arrhenius_env.sh:102-117`:

```text
cumm HEAD = 4dedaf43ff801e417c60c6bd7536a29d83d29ee0
cumm tracked/staged paths = empty
cumm untracked paths = cumm/core_cc/common.pyi only
cumm/core_cc/common.pyi observed sha256 = 656f8279c81e83f17f350be158c840d71ab973d7a7d893ec9d7b28a2a1847bfa
spconv HEAD = 263d6b47425ef843c82f997b12d8b714013d216c
spconv staged/untracked paths = empty
spconv tracked paths = pyproject.toml only
spconv full-index binary diff sha256 = 6d398e709e73d770d17fdb6dce3c80aed4c56b7fb173ee1c5ba9029c01639cf3
patched spconv pyproject.toml sha256 = e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9
```

The prior tracked-clean rejection was erroneous: this one-line patch is produced
by the repository builder after installing local cumm and before installing
spconv with `--no-build-isolation --no-deps`. The future command now accepts only
this exact state. It records HEAD/status/full diff plus every changed/untracked
path hash before imports and again after pytest, then requires byte equality.
Additional paths, staged bytes, altered patch/file bytes, or a changed untracked
`common.pyi` all fail closed. No dependency checkout was modified, normalized,
reset, cleaned, rebuilt, or imported during this remediation.

This reconciliation is environment provenance only. It is not a GH200/runtime
PASS and does not authorize future source dirt. Because only handoff documents
changed, all executable test/config/source-closure identities remain unchanged.

## Requested-test inventory inspection

The 22 pre-existing ownership-listed requestable test files contain 155 top-level
test functions before parametrization. The new file adds exactly three functions:

1. `test_only_clean_flower_profiles_and_plain_fedavg_default`;
2. `test_mini_first_batch_workers_zero_equals_two`;
3. `test_exact_mode_b1_fp16_optimizer_update`, expanded only for C-STR8, L-S075,
   and F-U.

The proposed exact subset is recorded in `RUN_REQUEST.md`. Existing fork/spawn
matrix tests, trainval partition tests, legacy catch-and-skip multiworker test,
extra legacy detector update, `test_model_overfit.py`, and the official-eval
permutation metric case are not requested.

## Explicit NOT RUN and interpretation limits

All dependency-backed items listed in HANDOFF and RUN_REQUEST are NOT RUN because
the login interpreter lacks dependencies and GH200 is not approved. In particular,
there is no evidence yet that the new full C/L/F update cases fit the 60-minute
budget or pass without warning; a future failure must be preserved and may justify
only a minimal owned current-tree fix after separate authorization.

These results support source/config/static readiness only. They do not support
detector capability, mAP/NDS, fusion gain, performance, FL quality, Protocol A/B,
attack/defense, generalization, reproducibility, or publication claims.
