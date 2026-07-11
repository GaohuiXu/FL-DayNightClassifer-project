# S03 RUN REQUEST — focused camera-contract validation

## Approval state

`PENDING_S00_APPROVAL_DO_NOT_SUBMIT`

This request is an exact bounded, non-scientific engineering smoke proposed under
O-017/O-009.  O-017 is stricter than standing O-009 for Wave-A workers: S03 must
stop and receive explicit S00 approval in this task before `sbatch`.  Preparing or
committing this file is not approval.

## Immutable implementation and source identity

- Implementation commit: `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`.
- Base commit: `372de9398ae435f82b83367a922fd302c0635738`.
- Worker branch: `codex/s03-camera-architecture`.
- Execution method: `git archive` of the immutable implementation commit into the
  unique output root.  The job does not depend on later handoff-document commits,
  an uncommitted diff, or a branch checkout.
- Runtime source-list entries: 14.
- C-locale-sorted source-list SHA-256:
  `ca10176be4aa440aa00cccf4b7f4f706ab85c24ed26ee9da9b982a8bd6a91604`.
- SHA-256 of the complete `sha256sum` source-state file:
  `3b8878b2395b38896933ea4d5d7d558d385c9634cac7e8253410dc3130905f3c`.

Attested files:

```text
fl_v3/pyproject.toml
fl_v3/requirements.lock.txt
fl_v3/requirements.txt
fl_v3/scripts/arrhenius_env.sh
fl_v3/src/fl_v3/__init__.py
fl_v3/src/fl_v3/models/__init__.py
fl_v3/src/fl_v3/models/fusion/__init__.py
fl_v3/src/fl_v3/models/fusion/bev_grid.py
fl_v3/src/fl_v3/models/fusion/camera_backbone.py
fl_v3/src/fl_v3/models/fusion/camera_neck.py
fl_v3/src/fl_v3/models/fusion/preprocess.py
fl_v3/src/fl_v3/models/fusion/swin_sdpa.py
fl_v3/src/fl_v3/models/fusion/view_transform.py
fl_v3/tests/test_s03_camera_contract.py
```

Pytest is invoked with `--noconftest`, empty `PYTEST_ADDOPTS`, and external plugin
autoload disabled.  The selected test has no local eager imports beyond the
package initializers, camera modules, and read-only `bev_grid.py` listed above.
Torch, torchvision, pytest, Python/platform, CUDA, and GPU identity are captured
from the activated environment inside the allocation.

## Exact scope

- One selected file: `fl_v3/tests/test_s03_camera_contract.py`.
- Exactly 10 pytest cases after parametrization.
- Synthetic tensors only; no nuScenes mini or trainval metadata/payload/cache,
  no ZIP manifest, and no GT database.
- Covers projection residual fixtures for resize/crop/pad/flip/rotation;
  deterministic validation and seeded train replay; native 1600x900 validation
  geometry; all-level FPN gradient connectivity; pure-camera API/LiDAR invariance;
  pixel/feature sensitivity; stride-8/0.5 m shapes/dtype/memory arithmetic; and one
  Swin-T -> FPN -> pure-camera LSS forward/backward.  On a CUDA-visible node the
  last case automatically uses GH200 CUDA with fp16 autocast.
- No optimizer, scheduler, EMA, DataLoader, model training step, 100/1000-step
  gate, tiny-overfit loop, profile, evaluation, metric, matrix, seed campaign, or
  scientific result.

## Resources and cumulative budget

- One job, one node, one `nvidia_gh200_120gb`, eight CPUs.
- Walltime: `00:15:00`; maximum requested allocation: 0.25 GPU-hours.
- S03 cumulative GPU use before this request: 0 GPU-hours.
- No array, DDP, concurrent S03 job, retry, requeue, automatic resubmission,
  follow-on, or spare-GPU expansion.

Input repository (Git object source only):
`/home/gaohui/.codex/worktrees/68cf/fl_weather_project`

Unique output root (must be absent before submission):
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54`

Logs:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.out
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.err
```

## Exact submission

Run once only after S00 replaces the pending state above with an explicit approval
bound to every field in this request:

```bash
sbatch <<'SBATCH'
#!/usr/bin/env bash
#SBATCH --job-name=flv3_s03_camera_contract
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.err
set -euo pipefail

REPO=/home/gaohui/.codex/worktrees/68cf/fl_weather_project
IMPL_SHA=6dfd2c775f54e488f3930996b303ce21f9b8e8b7
EXPECTED_LIST_SHA=ca10176be4aa440aa00cccf4b7f4f706ab85c24ed26ee9da9b982a8bd6a91604
EXPECTED_SOURCE_SHA=3b8878b2395b38896933ea4d5d7d558d385c9634cac7e8253410dc3130905f3c
OUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54

test "$(git -C "$REPO" rev-parse "$IMPL_SHA^{commit}")" = "$IMPL_SHA"
test ! -e "$OUT"
mkdir -p "$OUT/executor"
git -C "$REPO" archive "$IMPL_SHA" | tar -x -C "$OUT/executor"
cd "$OUT/executor"

SOURCE_FILES=(
  fl_v3/pyproject.toml
  fl_v3/requirements.lock.txt
  fl_v3/requirements.txt
  fl_v3/scripts/arrhenius_env.sh
  fl_v3/src/fl_v3/__init__.py
  fl_v3/src/fl_v3/models/__init__.py
  fl_v3/src/fl_v3/models/fusion/__init__.py
  fl_v3/src/fl_v3/models/fusion/bev_grid.py
  fl_v3/src/fl_v3/models/fusion/camera_backbone.py
  fl_v3/src/fl_v3/models/fusion/camera_neck.py
  fl_v3/src/fl_v3/models/fusion/preprocess.py
  fl_v3/src/fl_v3/models/fusion/swin_sdpa.py
  fl_v3/src/fl_v3/models/fusion/view_transform.py
  fl_v3/tests/test_s03_camera_contract.py
)
printf '%s\n' "${SOURCE_FILES[@]}" | LC_ALL=C sort > "$OUT/runtime_source_files.txt"
test "$(sha256sum "$OUT/runtime_source_files.txt" | awk '{print $1}')" = "$EXPECTED_LIST_SHA"
while IFS= read -r source_file; do
  sha256sum "$source_file"
done < "$OUT/runtime_source_files.txt" > "$OUT/runtime_source_sha256s.txt"
test "$(sha256sum "$OUT/runtime_source_sha256s.txt" | awk '{print $1}')" = "$EXPECTED_SOURCE_SHA"

source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

python - "$OUT/execution_identity.json" "$IMPL_SHA" "$EXPECTED_SOURCE_SHA" <<'PY'
import json
import platform
import sys
import pytest
import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the S03 GH200 validation")
props = torch.cuda.get_device_properties(0)
record = {
    "git_sha": sys.argv[2],
    "runtime_source_sha256": sys.argv[3],
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "host": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pytest": pytest.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_device": props.name,
    "cuda_total_memory": props.total_memory,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

export PYTHONPATH="$PWD/fl_v3/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
python -m pytest --noconftest -q fl_v3/tests/test_s03_camera_contract.py \
  --junitxml="$OUT/pytest.junit.xml" | tee "$OUT/pytest.log"

python - "$OUT/pytest.junit.xml" "$OUT/test_summary.json" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
summary = {
    field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
    for field in ("tests", "failures", "errors", "skipped")
}
if summary != {"tests": 10, "failures": 0, "errors": 0, "skipped": 0}:
    raise SystemExit(f"unexpected pytest summary: {summary}")
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

sha256sum \
  "$OUT/execution_identity.json" \
  "$OUT/runtime_source_files.txt" \
  "$OUT/runtime_source_sha256s.txt" \
  "$OUT/pytest.junit.xml" \
  "$OUT/pytest.log" \
  "$OUT/test_summary.json" \
  > "$OUT/sha256sums.txt"
(cd "$OUT" && sha256sum -c sha256sums.txt)
SBATCH
```

## Stop conditions and interpretation

The job fails immediately on missing Git object, pre-existing output, source-list or
source-content drift, unavailable CUDA, any pytest failure/error/skip, a test count
other than 10, or checksum failure.  Any failure is recorded; it does not authorize
a retry.  S03 stops and returns to S00 before changing code, command, resources,
output, or test scope.

Allowed if PASS: exact synthetic S03 camera geometry/interface/gradient tests pass
on the recorded GH200 runtime, including the one CUDA fp16-autocast camera-chain
case.

Forbidden regardless of PASS: trainval/mini model readiness, 100/1000-step or
tiny-overfit acceptance, throughput/profile claims, mAP/NDS/fusion gain, FL,
attack/defense, generalization, scientific, or publication claims.
