#!/usr/bin/env bash
#SBATCH --job-name=flv3_s03_camera_contract
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.err

set -euo pipefail

# These values are deliberately supplied by S00 only after the executable commit
# and final RUN_REQUEST bytes exist.  Keeping approved hashes outside this file
# avoids a launcher/request/source self-hash cycle.
: "${EXPECTED_S03_EXECUTABLE_SHA:?S00-approved executable HEAD is required}"
: "${EXPECTED_S03_IMPLEMENTATION_SHA:?S00-approved implementation SHA is required}"
: "${EXPECTED_S03_BRANCH:?S00-approved branch is required}"
: "${EXPECTED_S03_SOURCE_LIST_SHA:?S00-approved source-list SHA-256 is required}"
: "${EXPECTED_S03_SOURCE_SHA:?S00-approved source-state SHA-256 is required}"
: "${EXPECTED_S03_LAUNCHER_SHA:?S00-approved launcher SHA-256 is required}"
: "${EXPECTED_S03_RUN_REQUEST_SHA:?S00-approved RUN_REQUEST SHA-256 is required}"
: "${S03_OUTPUT_ROOT:?S00-approved output root is required}"

readonly IMPLEMENTATION_SHA=6dfd2c775f54e488f3930996b303ce21f9b8e8b7
readonly DELIVERY_BRANCH=codex/s03-camera-architecture
readonly APPROVED_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_6dfd2c775f54
readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
readonly REPO="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
readonly REQUEST_PATH="$SCRIPT_DIR/RUN_REQUEST.md"

require_hex() {
  local value="$1"
  local length="$2"
  local label="$3"
  if [[ ! "$value" =~ ^[0-9a-f]+$ ]] || [[ "${#value}" -ne "$length" ]]; then
    echo "invalid $label: expected $length lowercase hex characters" >&2
    exit 1
  fi
}

require_hex "$EXPECTED_S03_EXECUTABLE_SHA" 40 EXPECTED_S03_EXECUTABLE_SHA
require_hex "$EXPECTED_S03_IMPLEMENTATION_SHA" 40 EXPECTED_S03_IMPLEMENTATION_SHA
require_hex "$EXPECTED_S03_SOURCE_LIST_SHA" 64 EXPECTED_S03_SOURCE_LIST_SHA
require_hex "$EXPECTED_S03_SOURCE_SHA" 64 EXPECTED_S03_SOURCE_SHA
require_hex "$EXPECTED_S03_LAUNCHER_SHA" 64 EXPECTED_S03_LAUNCHER_SHA
require_hex "$EXPECTED_S03_RUN_REQUEST_SHA" 64 EXPECTED_S03_RUN_REQUEST_SHA

[[ "$EXPECTED_S03_IMPLEMENTATION_SHA" == "$IMPLEMENTATION_SHA" ]]
[[ "$EXPECTED_S03_BRANCH" == "$DELIVERY_BRANCH" ]]
[[ "$S03_OUTPUT_ROOT" == "$APPROVED_OUTPUT_ROOT" ]]
[[ -f "$REQUEST_PATH" ]]

readonly ACTUAL_HEAD="$(git -C "$REPO" rev-parse HEAD)"
readonly ACTUAL_BRANCH="$(git -C "$REPO" branch --show-current)"
[[ "$ACTUAL_HEAD" == "$EXPECTED_S03_EXECUTABLE_SHA" ]]
[[ "$ACTUAL_BRANCH" == "$EXPECTED_S03_BRANCH" ]]
[[ -z "$(git -C "$REPO" status --short)" ]]
git -C "$REPO" merge-base --is-ancestor "$IMPLEMENTATION_SHA" "$ACTUAL_HEAD"

readonly ACTUAL_LAUNCHER_SHA="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
readonly ACTUAL_REQUEST_SHA="$(sha256sum "$REQUEST_PATH" | awk '{print $1}')"
[[ "$ACTUAL_LAUNCHER_SHA" == "$EXPECTED_S03_LAUNCHER_SHA" ]]
[[ "$ACTUAL_REQUEST_SHA" == "$EXPECTED_S03_RUN_REQUEST_SHA" ]]

# --noconftest is part of the committed invocation below, so tests/conftest.py is
# intentionally not an executed input.  This list covers the selected test's local
# eager import closure, effective pytest/dependency inputs, environment bootstrap,
# and this durable launcher.  RUN_REQUEST is bound separately above by the external
# approval hash to avoid a source/request hash cycle.
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
  fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh
)
mapfile -t SORTED_SOURCE_FILES < <(printf '%s\n' "${SOURCE_FILES[@]}" | LC_ALL=C sort)
readonly ACTUAL_SOURCE_LIST_SHA="$(printf '%s\n' "${SORTED_SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')"
readonly ACTUAL_SOURCE_SHA="$({
  for source_file in "${SORTED_SOURCE_FILES[@]}"; do
    sha256sum "$REPO/$source_file" | sed "s|  $REPO/|  |"
  done
} | sha256sum | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_LIST_SHA" == "$EXPECTED_S03_SOURCE_LIST_SHA" ]]
[[ "$ACTUAL_SOURCE_SHA" == "$EXPECTED_S03_SOURCE_SHA" ]]

[[ ! -e "$S03_OUTPUT_ROOT" ]]
mkdir -p "$S03_OUTPUT_ROOT"
cp "$SCRIPT_PATH" "$S03_OUTPUT_ROOT/approved_launcher.sh"
cp "$REQUEST_PATH" "$S03_OUTPUT_ROOT/approved_run_request.md"
[[ "$(sha256sum "$S03_OUTPUT_ROOT/approved_launcher.sh" | awk '{print $1}')" == "$EXPECTED_S03_LAUNCHER_SHA" ]]
[[ "$(sha256sum "$S03_OUTPUT_ROOT/approved_run_request.md" | awk '{print $1}')" == "$EXPECTED_S03_RUN_REQUEST_SHA" ]]
printf '%s\n' "${SORTED_SOURCE_FILES[@]}" > "$S03_OUTPUT_ROOT/runtime_source_files.txt"
for source_file in "${SORTED_SOURCE_FILES[@]}"; do
  sha256sum "$REPO/$source_file" | sed "s|  $REPO/|  |"
done > "$S03_OUTPUT_ROOT/runtime_source_sha256s.txt"

cd "$REPO"
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env

python - "$S03_OUTPUT_ROOT/execution_identity.json" <<'PY'
import json
import os
import platform

import pytest
import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for the S03 GH200 validation")
props = torch.cuda.get_device_properties(0)
record = {
    "executable_git_sha": os.environ["EXPECTED_S03_EXECUTABLE_SHA"],
    "implementation_git_sha": os.environ["EXPECTED_S03_IMPLEMENTATION_SHA"],
    "git_branch": os.environ["EXPECTED_S03_BRANCH"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S03_SOURCE_LIST_SHA"],
    "runtime_source_sha256": os.environ["EXPECTED_S03_SOURCE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S03_LAUNCHER_SHA"],
    "run_request_sha256": os.environ["EXPECTED_S03_RUN_REQUEST_SHA"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
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
with open(__import__("sys").argv[1], "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

export PYTHONPATH="$REPO/fl_v3/src"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
python -m pytest --noconftest -q fl_v3/tests/test_s03_camera_contract.py \
  --junitxml="$S03_OUTPUT_ROOT/pytest.junit.xml" \
  | tee "$S03_OUTPUT_ROOT/pytest.log"

python - "$S03_OUTPUT_ROOT/pytest.junit.xml" "$S03_OUTPUT_ROOT/test_summary.json" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
summary = {
    field: sum(int(suite.attrib.get(field, "0")) for suite in suites)
    for field in ("tests", "failures", "errors", "skipped")
}
expected = {"tests": 10, "failures": 0, "errors": 0, "skipped": 0}
if summary != expected:
    raise SystemExit(f"unexpected pytest summary: {summary} != {expected}")
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

sha256sum \
  "$S03_OUTPUT_ROOT/approved_launcher.sh" \
  "$S03_OUTPUT_ROOT/approved_run_request.md" \
  "$S03_OUTPUT_ROOT/execution_identity.json" \
  "$S03_OUTPUT_ROOT/runtime_source_files.txt" \
  "$S03_OUTPUT_ROOT/runtime_source_sha256s.txt" \
  "$S03_OUTPUT_ROOT/pytest.junit.xml" \
  "$S03_OUTPUT_ROOT/pytest.log" \
  "$S03_OUTPUT_ROOT/test_summary.json" \
  > "$S03_OUTPUT_ROOT/sha256sums.txt"
(cd "$S03_OUTPUT_ROOT" && sha256sum -c sha256sums.txt)
