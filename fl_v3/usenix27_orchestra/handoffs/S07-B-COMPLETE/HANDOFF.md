# S07-B-COMPLETE HANDOFF — simplified clean integration completion candidate

## Status and identity

- Session: `S07-B-COMPLETE`.
- Startup/base HEAD: `4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5`.
- Accepted cleanup anchor: `70bcd856f7ebb411eb2887e7ab71ef41ed13271f`.
- Startup ref mode: clean detached HEAD, empty current branch.
- Worker/executable SHA: `34cbe02b7b72114e3a2d61f6f797c8dec022798c`.
- Executable tree: `ed2d4091f0098f6b2144028afd87e20d023b1da2`.
- Initial handoff seal: `6802f34fdafdf33bd31157ed15537b8f7955d1ad`.
- Approval seal: `e5f8dcf9f8608b40d49ad72c62b3557769b780fb`.
- Command-binding seal: `8d087f6d43a668c92dd540ccae7f80ac57f44def`.
- Compute: one exact owner-approved submission, job `372819`, terminal
  **FAILED `1:0` before pytest**; authorization consumed; no retry.

The owner authorized local durable materialization of the exact executable bytes,
a subsequent docs-only handoff seal, fast-forward of
`codex/s07-b-clean-completion`, and later one exact bounded GH200 submission. The
executable commit is a direct child of BASE and contains only the Flower config
plus focused test. No merge, push, upload, reviewer, environment mutation,
dataset scan, retry, or publication occurred.

## Bounded GH200 execution result

S00 submitted the frozen `RUN_REQUEST.md` command once. Slurm job `372819` used
the exact one-node/one-GH200/eight-CPU/96-GiB/60-minute/no-requeue envelope on
node `n124`; it ran for eight seconds, terminated `FAILED` with exit `1:0`, and
had zero restarts. The immutable executable archive, exact 100-path source
closure and source-record aggregate, and executable-patch hash all passed and are
preserved below the SHA-derived output root:

`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_34cbe02b7b72`

The job stopped before the first cumm/spconv dependency baseline record and before
environment activation. Both Slurm streams are empty. No JUnit, pytest log,
environment record, acceptance summary, C/L/F update, DataLoader comparison,
checkpoint/FL/data/evaluation gate, or runtime PASS exists. The finalizer preserved
and verified the available artifacts plus original/final exit `1`.

The run artifacts alone cannot identify the exact silent assertion, but S00
subsequently reproduced the failure deterministically. The consumed command
exported the project `.git` path as `GIT_COMMON_DIR`, which is a Git-reserved
repository-selection variable. Independent cumm/spconv `git -C` commands therefore
inherited the main project's common-dir override. Both HEAD checks still passed,
but cumm's first `git diff --cached --quiet` saw 267 staged paths and returned `1`;
spconv would have seen 160. A local harness using the consumed environment passed
the first five bootstrap guards and failed at guard six with 7,793 observed bytes.

The draft request-only correction renames the neutral path to `PROJECT_GIT_DIR`,
clears Git repository-selection overrides, and writes durable context plus
expected/observed/return-code artifacts around all thirteen guards. The same
login-safe harness then passes `13/13` and re-verifies its artifact manifest. This
is a failed request-wrapper bootstrap gate, not evidence of model/test failure.
The single-submission authorization remains fully consumed; no resubmission is
permitted without a newly audited exact request and owner approval.

The unsealed draft command SHA-256 is
`b5a98f0a09b79d9c64a474b1449f4e144c58e10a3b497d2c427c704e275d6596`;
its job-body SHA-256 is
`8c99f9026cdc09af3ffc17e91bcc490bc95f010cbfdec9e0511fec241d829e3e`.
The pytest-argument block remains byte-identical to the consumed request at
`2b9f312535632b7ec17a72ec5fbf0b300b5b690a4fd9d8a81ae94aea21028a67`.

## Evidence read and Git verification

Before editing, the session completely read the root `AGENTS.md`, all three
canonical Orchestra documents, `docs/env.md`, accepted S01/S07-A and S02-S06
handoff/review packages, S07-C HANDOFF/RESULTS/RUN_REQUEST, and the exact separate
S07-C review at `b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5`. Actual parent/tree/diffs were
checked for the cleanup implementation `a16c2cdf...`, handoff seal `f736f413...`,
review-launch/acceptance history, accepted anchor `70bcd856...`, and this packet
base `4aa2b133...`. The separate review is not an ancestor/import of this worker.

The frozen `e231808...` and forbidden `bf480ea...` implementations were not
imported, copied, cherry-picked, or used as implementation sources. Their old
runtime failure/spawn-policy content was not needed to justify a current-tree
source edit.

## Verification-first finding and exact changes

### Demonstrated pre-edit failure

`fl_v3/configs/flwr_config.toml` parsed successfully but exposed six `superlink`
keys: `default`, `supergrid`, CPU, sequential GPU, shared-GPU, and 4-GPU profiles.
Its active text retained T3, Path-A/Path-B, 4-GPU, overcommit, and `collab/**`
authority wording. This directly violated the mandatory kickoff contract.

### Changed files

1. `fl_v3/configs/flwr_config.toml`
   - now exposes exactly `default`, `local-simulation-cpu`, and
     `local-simulation-gpu`;
   - CPU resources remain `(8 supernodes, 1 CPU, 0 GPU)`;
   - the clean GPU profile is `(8 supernodes, 1 CPU, 1.0 GPU per actor)`, so a
     future one-GPU allocation can schedule actors only sequentially;
   - removes supergrid/shared/4-GPU profiles and every banned authority phrase;
   - explicitly states that profiles do not grant Flower/Ray execution or
     scientific-policy authority.

2. `fl_v3/tests/test_s07_b_clean_completion.py`
   - proves the exact two-profile contract and the actual default
     `CleanFedAvgStrategy`/identity FedAvg/server-EMA-zero behavior;
   - adds one no-skip standard DataLoader first-batch equality check at
     `num_workers=0` versus `2`, using one real mini fusion sample at depth 10;
   - adds exactly one parametrized B=1/`num_workers=0`/fp16 optimizer update for
     C-STR8, L-S075, and F-U, through the production constructor, criterion, and
     S06 `train_one_epoch` accounting path;
   - bounds LiDAR input to the first 4096 real-mini points only to keep this an
   engineering gate; it does not add a profile, benchmark, matrix, retry, or
   scientific threshold;
   - passes `telemetry_interval=1`, emits the actual production
     `last_grad_norm` and scaler/accounting telemetry per mode, and requires
     finite loss, finite positive gradient norm, exactly one optimizer step, one
     exposed GPU, enabled GradScaler, zero scaler skips/nonfinite loss, and a
     clean checkpoint boundary in `TrainingState`.

No production Python source, S01 implementation, model architecture, coordinate,
yaw, class map, precision policy, checkpoint implementation, evaluation adapter,
FedOpt/EMA capability, script, dependency manifest, canonical document, legacy
evidence, or candidate S07 template was changed.

## Preserved fail-closed configuration boundary

`fl_v3/configs/s06_synthetic_camera.json` is JSON-valid but is not valid under the
current resolved schema; direct loading fails first on missing
`model.camera_pretrained`. Accepted S07-B already documented that this historical
synthetic file was not silently promoted. Making it runnable would require
inventing cache/runtime identities and is forbidden here. It is not an execution
input to the proposed gate. The five `s07_b_*` files remain byte-identical,
template-only, and fail closed; their accepted SHA-256 values are unchanged.

## Local/static verification

All login-safe checks pass after the diff:

- exact protected scripts: `18/18`;
- S07-C deleted paths: `70`, surviving paths: `0`;
- Python source compile via `compile(..., "exec")`: `136/136`;
- JSON parse: `27/27`; TOML parse: `2/2`;
- retained shell `bash -n`: `17/17`;
- `git diff --check`: PASS;
- Flower keys exactly `default,local-simulation-cpu,local-simulation-gpu`;
- strategy classes exactly `CleanFedAvgStrategy`;
- default server optimizer is `fedavg`; server EMA default is `0.0`;
- `run_clean_round` and `run_clean_rounds` have no `**kwargs` selector;
- new completion file has three test functions (five expanded cases), no
  `pytest.skip`, and no explicit multiprocessing context/start-method matrix.
- the complete proposed inline Bash template in `RUN_REQUEST.md` passes
  `bash -n`; it has only the future exact executable SHA/tree sentinels.

The conservative 100-file candidate source closure is computed as SHA-256 lines
`<file-sha256><two spaces><relative-path>\n`, with paths sorted under `LC_ALL=C`:

- path-list SHA-256:
  `ce5c38764b43efa027b88b0b37de3a63407fb71ee9b0c5ad5bcd0671a0323ac4`;
- source-record aggregate SHA-256:
  `acb80014ff8dd3ef123e689b3be34efae219c95c95ea63f64c36e28f6d546a9e`;
- completion test SHA-256:
  `71d461eb3eb80a7e945ff4ae9e3fc8b07d7a99ed2b55b26a56d4e3c7ada4eef2`;
- Flower config SHA-256:
  `2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab`;
- executable working patch SHA-256 (tracked binary TOML diff, NUL, new relative
  test path, NUL, test bytes):
  `98c0521973ab9963cbf3447618efbedcba7a2fc6807804da222976e5b90f1002`.

`RUN_REQUEST.md` contains the literal sorted 100-path manifest, the exact record
construction algorithm, and the exact command used for job `372819`. That body
reconstructs the same selected set from Git, rejects count/order/duplicate/path
or byte drift, archives the exact commit, makes its snapshot non-writable, and
runs from a fresh writable SHA-derived job root. All cwd/HOME/temp/XDG/bytecode/
framework caches, `./fl_outputs`, logs, JUnit, statuses, and checksums stay below
that root. The generated mini info-cache is retained and hashed only as test
infrastructure.

## S00 compute-request remediation return

The owner-authorized return closes the four S00 findings without production
Python edits:

1. removed the necessarily-empty post-`clear_window()` `parameter.grad`
   assertions; production `last_grad_norm` telemetry is now the bounded
   finite/nonzero proof and the evidence JSON reports actual telemetry values;
2. replaced the unauditable command placeholder with a complete inline template
   covering environment, identity, archive/snapshot, exact pytest nodes,
   warnings/timeout, JUnit/log/status, checksum, stop, and exit propagation;
3. embedded and cross-checks the exact 100 paths and every source record;
4. pins execution cwd and every mutable cache/output below one fresh job root,
   while importing only the non-writable immutable snapshot.

### Narrow spconv identity correction

The later owner-authorized narrow remediation corrected only the request's
environment-provenance gate. The retained builder at
`fl_v3/scripts/build_arrhenius_env.sh:102-117` intentionally installs pinned local
cumm, removes only `cumm>=0.7.11` from spconv v2.3.8's build-system requirement,
then installs spconv with `--no-build-isolation --no-deps`. Read-only inspection
proves the current dependency source state exactly matches that contract:

- cumm HEAD `4dedaf43ff801e417c60c6bd7536a29d83d29ee0`, no tracked/staged change,
  exactly one untracked `cumm/core_cc/common.pyi`, observed SHA-256
  `656f8279c81e83f17f350be158c840d71ab973d7a7d893ec9d7b28a2a1847bfa`;
- spconv HEAD `263d6b47425ef843c82f997b12d8b714013d216c`, no staged or untracked
  change, and the only tracked path is `pyproject.toml`;
- spconv full-index binary diff SHA-256
  `6d398e709e73d770d17fdb6dce3c80aed4c56b7fb173ee1c5ba9029c01639cf3`;
- patched spconv `pyproject.toml` SHA-256
  `e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9`.

`RUN_REQUEST.md` now fails closed on those exact heads, staging/path sets, diff
bytes and file hash. Complete pre-import and post-pytest Git-state records contain
HEAD, porcelain status, the full tracked diff, and hashes for every changed or
untracked path; exact `cmp` equality is mandatory even when pytest itself fails.
The known patch is accepted build provenance only—not runtime PASS, permission to
mutate/reset/clean either checkout, or permission for any future source dirt.

Only the three handoff documents changed in this narrow return. Test/config/source
closure bytes and all five executable identities above remain unchanged.

## Explicit NOT RUN

Login `/usr/bin/python3` is x86_64 Python 3.9.25 and lacks pytest, NumPy, Torch,
Flower, nuScenes, spconv, and pyquaternion. The validated Python is an aarch64 ELF
and cannot execute on the login node. Although one bounded job was approved and
submitted, it failed before environment activation; the following remain **NOT
RUN / NO IMPLIED PASS** on this candidate:

- pytest collection and every dependency-backed requested test;
- the new workers-0-versus-2 mini batch check;
- C-STR8/L-S075/F-U construction and the three fp16 updates;
- S06 runtime/checkpoint/save/load/resume/CUDA rollback;
- Flower 1.27 parity, deterministic sampling, plain FedAvg round, FedOpt/EMA
  preservation, and trainable-only state;
- S01 real-mini directory/ZIP/cache/partition lifecycle;
- official box-to-global and GT-as-pred DetectionEval functional identity;
- any full cache/trainval, 100/1000-step, overfit, capability, mAP/NDS campaign,
  profile/throughput, Ray live federation, DDP/multi-GPU, process/seed matrix,
  retry, attack, defense, Protocol A/B, upload, or publication action.

## Interpretation and next action

Allowed now: the mandatory Flower profile cleanup and focused test source pass
login-safe static gates. Job `372819` is preserved only as a negative
bootstrap/envelope result.

Forbidden now: claiming runtime completion, a capable detector, mAP/NDS/fusion
gain, FL quality, Protocol readiness, security evidence, or compute permission.

Do not launch S07-B-COMPLETE review yet. The narrow draft fixes the reserved Git
environment-variable collision and instruments all thirteen bootstrap assertions
without changing W/tree, the 205-case inventory, data scope, or resource ceiling.
S00 must finish static audit and present a newly sealed RUN_REQUEST for owner
review. Another submission, review, merge, or push each require later explicit
owner authorization.
