#!/usr/bin/env bash
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s03_camera_contract
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s03_camera_contract_%j.err

set -euo pipefail

# These values are deliberately supplied by S00 only after the executable commit
# and final RUN_REQUEST bytes exist.  Keeping approved hashes outside this file
# avoids a launcher/request/source self-hash cycle.
: "${EXPECTED_S03_EXECUTABLE_SHA:?S00-approved executable HEAD is required}"
: "${EXPECTED_S03_TREE_SHA:?S00-approved executable tree is required}"
: "${EXPECTED_S03_IMPLEMENTATION_SHA:?S00-approved implementation SHA is required}"
: "${EXPECTED_S03_BRANCH:?S00-approved branch is required}"
: "${EXPECTED_S03_SOURCE_LIST_SHA:?S00-approved source-list SHA-256 is required}"
: "${EXPECTED_S03_SOURCE_SHA:?S00-approved source-state SHA-256 is required}"
: "${EXPECTED_S03_LAUNCHER_SHA:?S00-approved launcher SHA-256 is required}"
: "${EXPECTED_S03_RUN_REQUEST_SHA:?S00-approved RUN_REQUEST SHA-256 is required}"
: "${S03_SNAPSHOT_ROOT:?S00-approved execution snapshot root is required}"
: "${S03_OUTPUT_ROOT:?S00-approved output root is required}"
: "${SLURM_JOB_ID:?launcher must run inside one approved Slurm job}"

readonly IMPLEMENTATION_SHA=6dfd2c775f54e488f3930996b303ce21f9b8e8b7
readonly DELIVERY_BRANCH=codex/s03-camera-architecture
readonly COMMON_GIT_DIR=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git
readonly APPROVED_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s03_camera_contract_snapshotfix_6dfd2c775f54
readonly APPROVED_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s03_camera_contract_snapshotfix_6dfd2c775f54
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
readonly SCRIPT_PATH
readonly REQUEST_REL=fl_v3/usenix27_orchestra/handoffs/S03/RUN_REQUEST.md
readonly LAUNCHER_REL=fl_v3/usenix27_orchestra/handoffs/S03/run_s03_camera_contract.sh

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
require_hex "$EXPECTED_S03_TREE_SHA" 40 EXPECTED_S03_TREE_SHA
require_hex "$EXPECTED_S03_IMPLEMENTATION_SHA" 40 EXPECTED_S03_IMPLEMENTATION_SHA
require_hex "$EXPECTED_S03_SOURCE_LIST_SHA" 64 EXPECTED_S03_SOURCE_LIST_SHA
require_hex "$EXPECTED_S03_SOURCE_SHA" 64 EXPECTED_S03_SOURCE_SHA
require_hex "$EXPECTED_S03_LAUNCHER_SHA" 64 EXPECTED_S03_LAUNCHER_SHA
require_hex "$EXPECTED_S03_RUN_REQUEST_SHA" 64 EXPECTED_S03_RUN_REQUEST_SHA

[[ "$EXPECTED_S03_IMPLEMENTATION_SHA" == "$IMPLEMENTATION_SHA" ]]
[[ "$EXPECTED_S03_BRANCH" == "$DELIVERY_BRANCH" ]]
[[ "$S03_SNAPSHOT_ROOT" == "$APPROVED_SNAPSHOT_ROOT" ]]
[[ "$S03_OUTPUT_ROOT" == "$APPROVED_OUTPUT_ROOT" ]]
[[ -d "$COMMON_GIT_DIR/objects" ]]

# `--nodes=1` forced a whole non-oversubscribed node for job 335630 even though
# one GPU was requested.  Fail closed on the scheduler's actual allocation and
# the batch step's CUDA visibility before creating a snapshot/output or importing
# any model dependency.  `scontrol` is authoritative here; CUDA visibility is
# independently rechecked through torch after environment activation.
SLURM_ALLOCATION_RECORD="$(scontrol show job "$SLURM_JOB_ID" --oneliner)"
readonly SLURM_ALLOCATION_RECORD
allocation_field() {
  local key="$1"
  printf '%s\n' "$SLURM_ALLOCATION_RECORD" \
    | tr ' ' '\n' \
    | awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}'
}
readonly ACTUAL_NUM_NODES="$(allocation_field NumNodes)"
readonly ACTUAL_NUM_CPUS="$(allocation_field NumCPUs)"
readonly ACTUAL_OVERSUBSCRIBE="$(allocation_field OverSubscribe)"
readonly ACTUAL_ALLOC_TRES="$(allocation_field AllocTRES)"
readonly ACTUAL_ALLOC_GPUS="$({
  printf '%s\n' "$ACTUAL_ALLOC_TRES" | tr ',' '\n' \
    | awk -F= '$1 == "gres/gpu" {print $2; exit}'
})"
readonly ACTUAL_ALLOC_TYPED_GPUS="$({
  printf '%s\n' "$ACTUAL_ALLOC_TRES" | tr ',' '\n' \
    | awk -F= '$1 == "gres/gpu:nvidia_gh200_120gb" {print $2; exit}'
})"
: "${CUDA_VISIBLE_DEVICES:?Slurm batch step must expose exactly one CUDA device}"
IFS=',' read -r -a CUDA_VISIBLE_DEVICE_LIST <<< "$CUDA_VISIBLE_DEVICES"
[[ "$ACTUAL_NUM_NODES" == "1" ]]
[[ "$ACTUAL_NUM_CPUS" == "8" ]]
[[ "$ACTUAL_OVERSUBSCRIBE" == "OK" ]]
[[ "$ACTUAL_ALLOC_GPUS" == "1" ]]
[[ "$ACTUAL_ALLOC_TYPED_GPUS" == "1" ]]
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]]
[[ "${#CUDA_VISIBLE_DEVICE_LIST[@]}" -eq 1 ]]
[[ -n "${CUDA_VISIBLE_DEVICE_LIST[0]}" ]]
[[ "${CUDA_VISIBLE_DEVICE_LIST[0]}" != "-1" ]]
printf '[S03] verified allocation: %s\n' "$SLURM_ALLOCATION_RECORD"
printf '[S03] CUDA_VISIBLE_DEVICES=%s SLURM_JOB_GPUS=%s\n' \
  "$CUDA_VISIBLE_DEVICES" "${SLURM_JOB_GPUS:-unset}"

# sbatch executes a spool copy, so BASH_SOURCE cannot locate a linked worktree.
# Resolve the approved immutable object directly from the shared /nobackup object
# store, then export that exact tree into a unique read-only execution snapshot.
ACTUAL_HEAD="$(git --git-dir="$COMMON_GIT_DIR" rev-parse "refs/heads/$DELIVERY_BRANCH^{commit}")"
readonly ACTUAL_HEAD
ACTUAL_TREE="$(git --git-dir="$COMMON_GIT_DIR" rev-parse "$EXPECTED_S03_EXECUTABLE_SHA^{tree}")"
readonly ACTUAL_TREE
[[ "$ACTUAL_HEAD" == "$EXPECTED_S03_EXECUTABLE_SHA" ]]
[[ "$ACTUAL_TREE" == "$EXPECTED_S03_TREE_SHA" ]]
git --git-dir="$COMMON_GIT_DIR" merge-base --is-ancestor "$IMPLEMENTATION_SHA" "$ACTUAL_HEAD"

ACTUAL_LAUNCHER_SHA="$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')"
readonly ACTUAL_LAUNCHER_SHA
[[ "$ACTUAL_LAUNCHER_SHA" == "$EXPECTED_S03_LAUNCHER_SHA" ]]
[[ ! -e "$S03_SNAPSHOT_ROOT" ]]
[[ ! -e "$S03_OUTPUT_ROOT" ]]

readonly TMP_SNAPSHOT_ROOT="${S03_SNAPSHOT_ROOT}.tmp.${SLURM_JOB_ID}"
[[ ! -e "$TMP_SNAPSHOT_ROOT" ]]
mkdir -p "$(dirname "$S03_SNAPSHOT_ROOT")"
mkdir "$TMP_SNAPSHOT_ROOT"
cleanup_snapshot_tmp() {
  if [[ -d "$TMP_SNAPSHOT_ROOT" ]]; then
    chmod -R u+w "$TMP_SNAPSHOT_ROOT"
    rm -rf -- "$TMP_SNAPSHOT_ROOT"
  fi
}
trap cleanup_snapshot_tmp EXIT
git --git-dir="$COMMON_GIT_DIR" archive --format=tar "$ACTUAL_HEAD" \
  | tar -xf - -C "$TMP_SNAPSHOT_ROOT"

readonly SNAPSHOT_REQUEST_PATH="$TMP_SNAPSHOT_ROOT/$REQUEST_REL"
readonly SNAPSHOT_LAUNCHER_PATH="$TMP_SNAPSHOT_ROOT/$LAUNCHER_REL"
[[ -f "$SNAPSHOT_REQUEST_PATH" ]]
[[ -f "$SNAPSHOT_LAUNCHER_PATH" ]]
ACTUAL_REQUEST_SHA="$(sha256sum "$SNAPSHOT_REQUEST_PATH" | awk '{print $1}')"
readonly ACTUAL_REQUEST_SHA
SNAPSHOT_LAUNCHER_SHA="$(sha256sum "$SNAPSHOT_LAUNCHER_PATH" | awk '{print $1}')"
readonly SNAPSHOT_LAUNCHER_SHA
[[ "$ACTUAL_REQUEST_SHA" == "$EXPECTED_S03_RUN_REQUEST_SHA" ]]
[[ "$SNAPSHOT_LAUNCHER_SHA" == "$EXPECTED_S03_LAUNCHER_SHA" ]]

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
ACTUAL_SOURCE_LIST_SHA="$(printf '%s\n' "${SORTED_SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')"
readonly ACTUAL_SOURCE_LIST_SHA
ACTUAL_SOURCE_SHA="$({
  for source_file in "${SORTED_SOURCE_FILES[@]}"; do
    sha256sum "$TMP_SNAPSHOT_ROOT/$source_file" | sed "s|  $TMP_SNAPSHOT_ROOT/|  |"
  done
} | sha256sum | awk '{print $1}')"
readonly ACTUAL_SOURCE_SHA
[[ "$ACTUAL_SOURCE_LIST_SHA" == "$EXPECTED_S03_SOURCE_LIST_SHA" ]]
[[ "$ACTUAL_SOURCE_SHA" == "$EXPECTED_S03_SOURCE_SHA" ]]

cat > "$TMP_SNAPSHOT_ROOT/.s03_snapshot_identity.json" <<EOF
{
  "branch": "$EXPECTED_S03_BRANCH",
  "executable_git_sha": "$ACTUAL_HEAD",
  "executable_tree_sha": "$ACTUAL_TREE",
  "implementation_git_sha": "$IMPLEMENTATION_SHA",
  "launcher_sha256": "$ACTUAL_LAUNCHER_SHA",
  "run_request_sha256": "$ACTUAL_REQUEST_SHA",
  "runtime_source_list_sha256": "$ACTUAL_SOURCE_LIST_SHA",
  "runtime_source_sha256": "$ACTUAL_SOURCE_SHA",
  "slurm_job_id": "$SLURM_JOB_ID"
}
EOF
chmod -R a-w "$TMP_SNAPSHOT_ROOT"
mv "$TMP_SNAPSHOT_ROOT" "$S03_SNAPSHOT_ROOT"
trap - EXIT
readonly REPO="$S03_SNAPSHOT_ROOT"
readonly REQUEST_PATH="$REPO/$REQUEST_REL"
readonly SNAPSHOT_IDENTITY_PATH="$REPO/.s03_snapshot_identity.json"
S03_SNAPSHOT_IDENTITY_SHA="$(sha256sum "$SNAPSHOT_IDENTITY_PATH" | awk '{print $1}')"
readonly S03_SNAPSHOT_IDENTITY_SHA
export S03_SNAPSHOT_IDENTITY_SHA
if find "$REPO" -perm /222 -print -quit | grep -q .; then
  echo "execution snapshot contains writable paths" >&2
  exit 1
fi

[[ ! -e "$S03_OUTPUT_ROOT" ]]
mkdir -p "$S03_OUTPUT_ROOT"
cp "$REPO/$LAUNCHER_REL" "$S03_OUTPUT_ROOT/approved_launcher.sh"
cp "$REQUEST_PATH" "$S03_OUTPUT_ROOT/approved_run_request.md"
cp "$SNAPSHOT_IDENTITY_PATH" "$S03_OUTPUT_ROOT/snapshot_identity.json"
{
  printf '%s\n' "$SLURM_ALLOCATION_RECORD"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS:-unset}"
  printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-unset}"
} > "$S03_OUTPUT_ROOT/slurm_allocation.txt"
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
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"S03 requires exactly one CUDA-visible GPU, got {torch.cuda.device_count()}"
    )
props = torch.cuda.get_device_properties(0)
record = {
    "executable_git_sha": os.environ["EXPECTED_S03_EXECUTABLE_SHA"],
    "executable_tree_sha": os.environ["EXPECTED_S03_TREE_SHA"],
    "implementation_git_sha": os.environ["EXPECTED_S03_IMPLEMENTATION_SHA"],
    "git_branch": os.environ["EXPECTED_S03_BRANCH"],
    "execution_snapshot_root": os.environ["S03_SNAPSHOT_ROOT"],
    "snapshot_identity_sha256": os.environ["S03_SNAPSHOT_IDENTITY_SHA"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S03_SOURCE_LIST_SHA"],
    "runtime_source_sha256": os.environ["EXPECTED_S03_SOURCE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S03_LAUNCHER_SHA"],
    "run_request_sha256": os.environ["EXPECTED_S03_RUN_REQUEST_SHA"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
    "slurm_job_cpus_per_node": os.environ.get("SLURM_JOB_CPUS_PER_NODE"),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
  "$S03_OUTPUT_ROOT/snapshot_identity.json" \
  "$S03_OUTPUT_ROOT/slurm_allocation.txt" \
  "$S03_OUTPUT_ROOT/execution_identity.json" \
  "$S03_OUTPUT_ROOT/runtime_source_files.txt" \
  "$S03_OUTPUT_ROOT/runtime_source_sha256s.txt" \
  "$S03_OUTPUT_ROOT/pytest.junit.xml" \
  "$S03_OUTPUT_ROOT/pytest.log" \
  "$S03_OUTPUT_ROOT/test_summary.json" \
  > "$S03_OUTPUT_ROOT/sha256sums.txt"
(cd "$S03_OUTPUT_ROOT" && sha256sum -c sha256sums.txt)
