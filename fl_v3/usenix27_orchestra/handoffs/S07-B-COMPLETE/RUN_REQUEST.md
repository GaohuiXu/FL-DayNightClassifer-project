# S07-B-COMPLETE RUN_REQUEST — simplified training-first validation

## State and immutable inputs

```text
SESSION_ID: S07-B-COMPLETE
REQUEST_STATE: APPROVED_EXACT_ONCE / NOT YET SUBMITTED
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

## Acceptance

Accept only one `COMPLETED 0:0` job with zero restarts, aarch64/GH200 identity,
training JUnit `4/0/0/0`, loader JUnit `1/0/0/0`, all four phase/tee exit files
equal to zero, three `S07_B_CLEAN_MODE_EVIDENCE` records in C/L/F order, and final
`S07B_SIMPLE_PASS`. Warnings remain visible but non-fatal.

Any nonzero phase, timeout, OOM, worker error, or missing record is a bounded
engineering FAIL. Preserve logs and stop. Do not change the environment, edit
source, add tests, retry, or submit a replacement without a new owner decision.
