# S07-B-COMPLETE RUN_REQUEST — simplified training-first validation

## State and immutable inputs

```text
SESSION_ID: S07-B-COMPLETE
REQUEST_STATE: CONSUMED / TERMINAL FAIL / NO RETRY AUTHORIZED
OWNER_DECISION: remove the audit wrapper; use a simple training-first contract
BASE_SHA: 4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5
EXECUTABLE_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
EXECUTABLE_TREE: ed2d4091f0098f6b2144028afd87e20d023b1da2
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_clean_simple_34cbe02b7b72
SNAPSHOT_STATE: materialized once from EXECUTABLE_SHA / read-only / 628 KiB
COMPLETION_TEST_SHA256: 71d461eb3eb80a7e945ff4ae9e3fc8b07d7a99ed2b55b26a56d4e3c7ada4eef2
FLOWER_CONFIG_SHA256: 2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
ARRHENIUS_ENV_SCRIPT_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
PYPROJECT_SHA256: 29c5e81e56fdcb40a2caefdc8a91563ffcd1596df64fed6f4997eef3d58bab72
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_clean_simple_34cbe02b7b72
COMMAND_SHA256: 229dfec34da4bc2b37769afa48636700cf53e063f928720d022f8173ab311e75
JOB_BODY_SHA256: dc14e7157e37b567d747093130020dde3306fab49eb43358883030400f66754c
APPROVAL_DATE: 2026-07-13
APPROVAL_SOURCE: owner message "批准，开始执行" in the canonical S00 task
APPROVED_COMPUTE: one exact simplified GH200 submission
RETRY: forbidden unless separately requested and approved
JOB_ID: 380806
JOB_STATE: FAILED 1:0 / elapsed 00:04:28 / node n192 / restarts 0
```

The executable differs from BASE only in
`fl_v3/configs/flwr_config.toml` and
`fl_v3/tests/test_s07_b_clean_completion.py`. The snapshot contains those exact
committed bytes and has zero writable files. No source, environment, dependency,
cache, dataset, or production launcher is changed by this request.

## Audit wrapper removal

The former embedded command and its Git/dependency/source-archive harness are
retired. They remain only in Git history and the immutable raw output roots of the
three consumed jobs. The new compute command performs none of the following:

- no Git command or repository path;
- no cumm/spconv checkout cleanliness or pre/post source comparison;
- no source archive, 100-file manifest, or artifact manifest;
- no global warnings-as-errors or warning-specific exception;
- no isolated long `TMPDIR`, HOME, or per-library cache tree;
- no 205-case integrated suite.

These removals do not weaken the two meaningful engineering checks. Exact source
identity is fixed once by the read-only snapshot before compute. Raw Slurm logs,
two pytest logs, two JUnit files, phase exit codes, and the environment summary are
the only requested runtime artifacts.

## Closed negative evidence

| Job | Terminal result | Boundary and meaning |
|---|---|---|
| `372819` | `FAILED 1:0`, 8 s, zero restarts | Request exported Git-reserved `GIT_COMMON_DIR`; stopped before environment activation. |
| `373363` | `FAILED 1:0`, 1:42, zero restarts | Environment/spconv import reached; request made a known dependency warning fatal; no pytest. |
| `374142` | `CANCELLED`, 8:05, zero restarts | Environment and identity passed; exact 205 cases collected and the clean-FedAvg profile test passed; request's 113-byte `TMPDIR` broke the worker=2 AF_UNIX listener before model updates. |

Raw evidence is preserved under the corresponding
`outputs/s07b_complete*34cbe02b7b72` roots. Key hashes retained from the latest two
jobs are:

```text
373363 stderr: 5b6357146d90321484a9984b9a8500d3b7b2f35b6e9bbfa94549fe11c9b343b3
373363 artifact manifest: fe6bc6363f945ae803b0f005e7f4e3fbf21d81162023631e91b0c2e75a04048c
374142 pytest/stdout: 0507e5e254f357932da28bdd2b58e116e1acdba368daadc3b96e8e74af9e3487
374142 stderr: 1ae7aff202a2955595bbb274d5627f8b616bc98fc1550d310ed8397ca1ae7969
374142 execution identity: a4ff6321e5ca76225b4a7cf89d6290191a419e833969b62cd1abc01e6bd41904
```

None of these jobs ran a C/L/F optimizer update. They are negative wrapper
evidence, not environment, model, data, or clean-FL failures.

## Simplified execution contract

One bounded job runs two independent phases, in this order:

1. **Training phase:** the clean Flower/FedAvg profile test plus one real-mini,
   B=1, fp16 optimizer update for each of C-STR8, L-S075, and F-U. Their existing
   test path uses `num_workers=0`, finite-loss/gradient checks, one optimizer step,
   enabled GradScaler, and no metric or scientific claim.
2. **Loader phase:** only the first-batch `num_workers=0` versus `num_workers=2`
   equality test, with node-local `TMPDIR=/tmp` and a separate five-minute timeout.

Training runs first so a loader failure cannot again mask whether the compiled
environment can construct and update the models. The 205-case suite is not part of
this request. A later suite, review, or scientific run requires a separate owner
decision.

Resources are one node/task/GH200, eight CPUs, 96 GiB, 40 minutes,
`--no-requeue`, one submission, no automatic retry. Data is real nuScenes mini
only. Full cache/trainval, multiple steps, overfit, metrics, profile, Ray, DDP,
matrix, attack, defense, Protocol A/B claims, upload, and publication are excluded.

## Exact owner-approved command — one submission only

```bash
#!/bin/bash
set -euo pipefail

SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_clean_simple_34cbe02b7b72
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_clean_simple_34cbe02b7b72
MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini

test -d "$SNAPSHOT"
test -f "$SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py"
test ! -e "$OUTPUT"
install -d -m 0700 "$OUTPUT"

sbatch \
  --account=naiss2025-22-1113-gpu \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=00:40:00 \
  --no-requeue \
  --job-name=flv3_s07b_simple \
  --output="$OUTPUT/slurm-%j.out" \
  --error="$OUTPUT/slurm-%j.err" \
  --export=S07B_SNAPSHOT="$SNAPSHOT",S07B_OUTPUT="$OUTPUT",S07B_MINI_ROOT="$MINI_ROOT" \
  <<'SBATCH'
#!/bin/bash
set -euo pipefail
umask 077

source "$S07B_SNAPSHOT/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONPATH="$S07B_SNAPSHOT/fl_v3/src"
export NUSCENES_DATAROOT="$S07B_MINI_ROOT"
unset ARRHENIUS_NUSCENES_DATAROOT NUSCENES_DATA_DIR
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
export TMPDIR=/tmp
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
unset PYTHONWARNINGS

cd "$S07B_OUTPUT"
python - <<'PY' > environment.txt
import json
import os
import platform
from importlib.metadata import version
import torch
import spconv

print(json.dumps({
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cumm": version("cumm"),
    "spconv": version("spconv"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "tmpdir": os.environ["TMPDIR"],
}, indent=2, sort_keys=True))
PY

set +e
timeout --signal=TERM --kill-after=30s 30m \
  python -m pytest -q -s -p no:cacheprovider \
    -c "$S07B_SNAPSHOT/fl_v3/pyproject.toml" \
    --rootdir="$S07B_SNAPSHOT" \
    --basetemp="$S07B_OUTPUT/pytest-train-tmp" \
    --junitxml="$S07B_OUTPUT/train.junit.xml" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_only_clean_flower_profiles_and_plain_fedavg_default" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_exact_mode_b1_fp16_optimizer_update" \
    2>&1 | tee train.log
train_rc=${PIPESTATUS[0]}
train_tee_rc=${PIPESTATUS[1]}
set -e
printf '%s\n' "$train_rc" > train.exit
printf '%s\n' "$train_tee_rc" > train-tee.exit
test "$train_rc" = 0
test "$train_tee_rc" = 0

set +e
timeout --signal=TERM --kill-after=30s 5m \
  python -m pytest -q -s -p no:cacheprovider \
    -c "$S07B_SNAPSHOT/fl_v3/pyproject.toml" \
    --rootdir="$S07B_SNAPSHOT" \
    --basetemp="$S07B_OUTPUT/pytest-loader-tmp" \
    --junitxml="$S07B_OUTPUT/loader.junit.xml" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_mini_first_batch_workers_zero_equals_two" \
    2>&1 | tee loader.log
loader_rc=${PIPESTATUS[0]}
loader_tee_rc=${PIPESTATUS[1]}
set -e
printf '%s\n' "$loader_rc" > loader.exit
printf '%s\n' "$loader_tee_rc" > loader-tee.exit
test "$loader_rc" = 0
test "$loader_tee_rc" = 0
printf '%s\n' S07B_SIMPLE_PASS
SBATCH
```

## Terminal execution and acceptance

The exact approved command was submitted once as Job `380806`. Environment
activation and identity passed on aarch64/GH200 with Torch `2.11.0+cu128`, CUDA
`12.8`, cumm `0.7.13`, spconv `2.3.8`, and `TMPDIR=/tmp`. The clean Flower/FedAvg
profile passed. All three real-mini modes completed model forward, finite loss,
and backward, but the unscaled gradient norm was nonfinite at the first
GradScaler attempt (`C-STR8=inf`, `L-S075=nan`, `F-U=nan`). Training JUnit is
`4 total / 1 pass / 3 fail`. The assertion occurs before the test checks or
prints optimizer-step, scaler-skip, and final-scale metrics, so the raw artifacts
do not establish those counters per mode. No successful update is accepted from
this job, but “all three were skipped” is not a durable result.

The loader phase correctly did not run after training failure. No retry or
replacement command is authorized. The command also has a non-causal shell
recording defect: reading `PIPESTATUS[0]` in one assignment resets `PIPESTATUS`,
so the following `PIPESTATUS[1]` access aborts under `set -u`. This explains the
missing phase exit files, but not the pytest failures, which had already been
written to the log and JUnit. Any future command should avoid this pipeline and
capture a redirected pytest exit code directly.

Raw root:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_clean_simple_34cbe02b7b72
environment.txt: b2e9d2df67472872f03dea6223d4dbef93bea29f60a5273aac334645fb487858
train.log/stdout: a6c9d686928fbf2cf2b658b0578f71ab9550539e52ee1bab6b70ee9fb14fe222
train.junit.xml: 8faeaf75ecc94a2eedceebaf0141b27f756d467a34b6d1c814d0d4d0544d1c9c
stderr: b534645ead7b5a3ed14c1d7fe032271613c2bf72d2c4e74776badddfbf1a888f
```

The acceptance rule below was **not met**.

Accept only one `COMPLETED 0:0` job with zero restarts, aarch64/GH200 identity,
training JUnit `4/0/0/0`, loader JUnit `1/0/0/0`, all four phase/tee exit files
equal to zero, three `S07_B_CLEAN_MODE_EVIDENCE` records in C/L/F order, and final
`S07B_SIMPLE_PASS`. Warnings remain visible but non-fatal.

Any nonzero phase, timeout, OOM, worker error, or missing record is a bounded
engineering FAIL. Preserve logs and stop. Do not change the environment, edit
source, add tests, retry, or submit a replacement without a new owner decision.

---

## D1 gradient classification request — exact one-shot approval

```text
REQUEST_ID: S07-B-COMPLETE-D1
REQUEST_STATE: CONSUMED / TERMINAL DIAGNOSTIC PASS / NO RETRY AUTHORIZED
OWNER_DIRECTION: commit the prepared diagnostic, then submit the exact Slurm job
WORKTREE_BASE_SHA: f492fcf493515df82f881825d8cc25ec399d8128
DIAGNOSTIC_COMMIT: 1900fe3bcb52ade22f0b947a2aca44d5ece12b2f
APPROVAL_SEAL: 9b23fabf33bde821a8053192566976b332f75c05
BASE_EXECUTABLE_SHA: 34cbe02b7b72114e3a2d61f6f797c8dec022798c
TEST_PATCH_SHA256: f50299cc7824a162d84b56d24755d17db979d1852c537c53a097289ad75d5d2e
DIAGNOSTIC_TEST_SHA256: 0ca44717e9787e4cb129dd028cbd217524ea12383c2f510f94b2084888ce475b
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_grad_diag_0ca44717e978
SNAPSHOT_TREE_SHA256: acf5f85efb529bfe8e5b6878303ecac2c40057e542433043ab86adbaf4e57337
SNAPSHOT_STATE: 628 KiB / zero writable files / zero writable directories
FLOWER_CONFIG_SHA256: 2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
ARRHENIUS_ENV_SCRIPT_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
PYPROJECT_SHA256: 29c5e81e56fdcb40a2caefdc8a91563ffcd1596df64fed6f4997eef3d58bab72
JOB_BODY_SHA256: d4d1c4acf353aa30bc7bd1872634f58b46df1f2df4d063afd7f3c63bbf28f3fa
SUBMIT_SHA256: dd69e3f6d55c99e55a2e52b5b4ef79f27a86d833558d1299387596d9b95f74d2
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_grad_diag_0ca44717e978
APPROVAL_DATE: 2026-07-13
APPROVAL_SOURCE: owner message "批准commit后提交slurm job" in the canonical S00 task
APPROVED_COMPUTE: one exact D1 submission with the pinned snapshot, nine cells, two script hashes, resources and output root below
RETRY: forbidden
JOB_ID: 389356
JOB_STATE: COMPLETED 0:0 / elapsed 00:04:05 / node n101 / restarts 0
```

The read-only D1 snapshot was copied from the exact Job `380806` snapshot and
differs from it only in
`fl_v3/tests/test_s07_b_clean_completion.py`. Production source, configs,
environment activation, dependencies and data code are byte-identical. The
test and preparation record are durable at `DIAGNOSTIC_COMMIT`; the later
docs-only approval seal is not executed by D1.

### Exact diagnostic cells

The fixed test inventory is the Cartesian list below, executed once in pytest
order with seed `20260713`, B=1, `num_workers=0`, ten LiDAR sweeps, sample token
`00889f8a9549450aa2f32cf310a3e305`, and LiDAR points capped at 4096:

```text
C-STR8 x {fp32, fp16 GradScaler init 512, fp16 GradScaler init 1}
L-S075 x {fp32, fp16 GradScaler init 512, fp16 GradScaler init 1}
F-U     x {fp32, fp16 GradScaler init 512, fp16 GradScaler init 1}
```

Each of the nine cells performs one forward, one loss and one backward. It calls
the optimizer at most once if gradients are accepted; GradScaler may instead
skip the fp16 call. Before gradients are cleared it records:

- aggregate and six per-task heatmap/regression losses;
- element-wise finiteness, nonfinite parameter/element counts and the first eight
  nonfinite parameter names;
- the current float32 telemetry norm, stable float64 norm over finite elements,
  gradient dtype counts and maximum finite absolute element;
- requested/before/after scale, scaler skip and optimizer-call status.

Diagnostic pytest success means all nine strict-JSON evidence records were
emitted with finite scalar loss. It does **not** mean all gradients or optimizer
calls passed. No dynamic backoff loop, loader phase, model remediation, config
change, metric, profile, full cache/trainval, attack/defense, Ray, DDP or retry is
included.

### Resources

One node, one GH200, eight CPUs, 96 GiB, Slurm limit 25 minutes, internal pytest
timeout 20 minutes, no requeue and exactly one submission. Maximum authorized
scope would be 0.417 GPU-hours. The mini dataset is read-only and the output root
must not exist before submission.

### Exact job body

```bash
#!/bin/bash
set -euo pipefail
umask 077

source "$S07B_SNAPSHOT/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONPATH="$S07B_SNAPSHOT/fl_v3/src"
export NUSCENES_DATAROOT="$S07B_MINI_ROOT"
unset ARRHENIUS_NUSCENES_DATAROOT NUSCENES_DATA_DIR
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
export TMPDIR=/tmp
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
unset PYTHONWARNINGS

cd "$S07B_OUTPUT"
python - <<'PY' > environment.txt
import json
import os
import platform
from importlib.metadata import version
import torch

print(json.dumps({
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cumm": version("cumm"),
    "spconv": version("spconv"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "tmpdir": os.environ["TMPDIR"],
}, indent=2, sort_keys=True))
PY

set +e
timeout --signal=TERM --kill-after=30s 20m \
  python -m pytest -q -s -p no:cacheprovider \
    -c "$S07B_SNAPSHOT/fl_v3/pyproject.toml" \
    --rootdir="$S07B_SNAPSHOT" \
    --basetemp="$S07B_OUTPUT/pytest-tmp" \
    --junitxml="$S07B_OUTPUT/diagnostic.junit.xml" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_exact_mode_gradient_diagnostic" \
    > "$S07B_OUTPUT/diagnostic.log" 2>&1
diagnostic_rc=$?
set -e
printf '%s\n' "$diagnostic_rc" > diagnostic.exit
cat diagnostic.log
test "$diagnostic_rc" = 0
test "$(grep -c 'S07_B_GRAD_DIAGNOSTIC=' diagnostic.log)" = 9
printf '%s\n' S07B_GRAD_DIAGNOSTIC_PASS
```

### Exact submit wrapper

```bash
#!/bin/bash
set -euo pipefail

SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_grad_diag_0ca44717e978
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_grad_diag_0ca44717e978
MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini

test -d "$SNAPSHOT"
test -f "$SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py"
test ! -e "$OUTPUT"
install -d -m 0700 "$OUTPUT"

sbatch \
  --account=naiss2025-22-1113-gpu \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=00:25:00 \
  --no-requeue \
  --job-name=flv3_s07b_grad_diag \
  --output="$OUTPUT/slurm-%j.out" \
  --error="$OUTPUT/slurm-%j.err" \
  --export=S07B_SNAPSHOT="$SNAPSHOT",S07B_OUTPUT="$OUTPUT",S07B_MINI_ROOT="$MINI_ROOT" \
  /tmp/s07b_grad_diag_job.sh
```

Both exact temporary files pass `bash -n`; their hashes are pinned above. The
owner approved exactly this D1 snapshot, cell list, two hashes, resources and
output root after the durable diagnostic commit. Job `389356` consumed that
one-shot approval and completed with nine evidence records. Stop: no automatic
source/config change and no second job.

---

## F1 uniform-FP32 final clean gate — terminal exact request

```text
REQUEST_ID: S07-B-COMPLETE-F1
REQUEST_STATE: CONSUMED / TERMINAL PASS / NO RETRY AUTHORIZED
OWNER_DIRECTION: no precision comparison or scaler remediation in S07-B; use one FP32 final gate
CURRENT_DURABLE_HEAD: 45332e5416166463d5cb2b0bcb9c71e2efdc08f4
TEST_COMMIT: 29ca6637bcd0a4e9a6422f3b820fb43d5295ad2c
TEST_PATCH_SHA256: cb38824d84805de13c99f3d39df4ea2bf7795a9731a6dea92d94e4ec07756c79
FINAL_TEST_SHA256: 1b72abf2f8aaa9c98db9cabe994792187f976c5fbb267483967a58103b61c79f
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_fp32_final_1b72abf2f8aa
SNAPSHOT_TREE_SHA256: 76c0bf5ba7ba0118a0150c3956cb5d8e9645a98adc54649a46a529bb96620d1c
SNAPSHOT_STATE: 628 KiB / zero writable files / zero writable directories
FLOWER_CONFIG_SHA256: 2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
ARRHENIUS_ENV_SCRIPT_SHA256: f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
PYPROJECT_SHA256: 29c5e81e56fdcb40a2caefdc8a91563ffcd1596df64fed6f4997eef3d58bab72
JOB_BODY_SHA256: db4a52499626a56e72d22ec49b43c8f01bcab450db55a76811fdd3957144d7c1
SUBMIT_SHA256: 015f701cde9679e017bb76ad44a8a4144a0587fbdb97133794935e8b46774b13
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_fp32_final_1b72abf2f8aa
OUTPUT_STATE: absent before approval/submission
APPROVAL_DATE: 2026-07-13
APPROVAL_SOURCE: owner message "批准精确 F1 commit/submit" in the canonical S00 task
APPROVED_COMPUTE: one exact F1 submission with the pinned test commit, snapshot, five cases, two script hashes, resources and output root
RETRY: forbidden
JOB_ID: 390576
JOB_STATE: COMPLETED 0:0 / elapsed 00:04:24 / node n105 / restarts 0
```

The read-only F1 snapshot was copied from the exact D1 snapshot and differs only
in `fl_v3/tests/test_s07_b_clean_completion.py`. That test-only diff renames the
one-step case to `test_exact_mode_b1_fp32_optimizer_update`, sets both model run
config and training loop precision to `fp32`, requires reported precision `fp32`,
and requires GradScaler disabled. Production source, model/loss code, configs,
environment activation, dependencies and data code are byte-identical. This does
not freeze the precision policy for a later full scientific run.

### Exact F1 inventory and acceptance

Pytest selects exactly five cases:

1. one clean Flower-profile/plain-FedAvg construction case;
2. C-STR8, L-S075 and F-U, each with B=1, seed `20260713`, ten sweeps,
   `num_workers=0`, at most 4096 LiDAR points and exactly one successful FP32
   `train_one_epoch` optimizer update;
3. one fusion first-batch equality case between `num_workers=0` and `2`, using
   node-local `/tmp` for multiprocessing sockets.

Each mode must report finite positive gradient norm, `optimizer_steps=1`,
`exposure_samples=1`, precision `fp32`, GradScaler disabled, zero scaler skips and
zero nonfinite-loss steps. Pytest must exit zero with exactly five passes, three
`S07_B_CLEAN_MODE_EVIDENCE` records and final marker
`S07B_FP32_FINAL_GATE_PASS`. Any failure, timeout, OOM, worker error, missing
record or missing marker is terminal FAIL with no retry.

Excluded: D1 diagnostic collection; fp16/AMP/scaler/precision comparisons;
production source/config/environment edits; full suite; cache/trainval; profile;
metrics; checkpoint/evaluation reruns; more than one update per C/L/F mode;
Flower/Ray execution; DDP; attack/defense; automatic remediation or retry.

### F1 resources

One node, one GH200, eight CPUs, 96 GiB, Slurm limit 25 minutes, internal pytest
timeout 20 minutes, no requeue and exactly one submission. Maximum authorized
scope would be 0.417 GPU-hours. The mini dataset is read-only and the output root
must not exist before submission.

### Exact F1 job body

```bash
#!/bin/bash
set -euo pipefail
umask 077

source "$S07B_SNAPSHOT/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONPATH="$S07B_SNAPSHOT/fl_v3/src"
export NUSCENES_DATAROOT="$S07B_MINI_ROOT"
unset ARRHENIUS_NUSCENES_DATAROOT NUSCENES_DATA_DIR
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
export TMPDIR=/tmp
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
unset PYTHONWARNINGS

cd "$S07B_OUTPUT"
python - <<'PY' > environment.txt
import json
import os
import platform
from importlib.metadata import version
import torch

print(json.dumps({
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cumm": version("cumm"),
    "spconv": version("spconv"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "tmpdir": os.environ["TMPDIR"],
}, indent=2, sort_keys=True))
PY

set +e
timeout --signal=TERM --kill-after=30s 20m \
  python -m pytest -q -s -p no:cacheprovider \
    -c "$S07B_SNAPSHOT/fl_v3/pyproject.toml" \
    --rootdir="$S07B_SNAPSHOT" \
    --basetemp="$S07B_OUTPUT/pytest-tmp" \
    --junitxml="$S07B_OUTPUT/final_gate.junit.xml" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_only_clean_flower_profiles_and_plain_fedavg_default" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_exact_mode_b1_fp32_optimizer_update" \
    "$S07B_SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py::test_mini_first_batch_workers_zero_equals_two" \
    > "$S07B_OUTPUT/final_gate.log" 2>&1
final_rc=$?
set -e
printf '%s\n' "$final_rc" > final_gate.exit
cat final_gate.log
test "$final_rc" = 0
test "$(grep -c 'S07_B_CLEAN_MODE_EVIDENCE=' final_gate.log)" = 3
printf '%s\n' S07B_FP32_FINAL_GATE_PASS
```

### Exact F1 submit wrapper

```bash
#!/bin/bash
set -euo pipefail

SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_fp32_final_1b72abf2f8aa
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_fp32_final_1b72abf2f8aa
MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini

test -d "$SNAPSHOT"
test -f "$SNAPSHOT/fl_v3/tests/test_s07_b_clean_completion.py"
test ! -e "$OUTPUT"
install -d -m 0700 "$OUTPUT"

sbatch \
  --account=naiss2025-22-1113-gpu \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=00:25:00 \
  --no-requeue \
  --job-name=flv3_s07b_fp32_final \
  --output="$OUTPUT/slurm-%j.out" \
  --error="$OUTPUT/slurm-%j.err" \
  --export=S07B_SNAPSHOT="$SNAPSHOT",S07B_OUTPUT="$OUTPUT",S07B_MINI_ROOT="$MINI_ROOT" \
  /tmp/s07b_fp32_final_job.sh
```

Both exact temporary files pass `bash -n`; their hashes are pinned above. The
owner approved the exact test commit plus this immutable F1 snapshot, inventory,
two script hashes, resources and output root. Job `390576` consumed that one-shot
approval and passed all five cases. Stop: no automatic change and no second job.
