#!/usr/bin/env bash
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s05r3_centerhead
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r3_centerhead_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r3_centerhead_%j.err
set -euo pipefail

required=(
  EXPECTED_S05R3_SHA EXPECTED_S05R3_TREE EXPECTED_S05R3_SOURCE_LIST_SHA
  EXPECTED_S05R3_SOURCE_SHA EXPECTED_S05R3_LAUNCHER_SHA
  EXPECTED_S05R3_RUN_REQUEST_SHA S05R3_REQUEST_COPY
  S05R3_SNAPSHOT_ROOT S05R3_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then echo "$name is required" >&2; exit 2; fi
done

readonly COMMON_GIT_DIR=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git
readonly APPROVED_SHA=96e509b71a3e22afb4de397132438fd3b9bbf5d8
readonly APPROVED_TREE=aeaaad044199492b81c4383a013f3fb3c6596c02
readonly APPROVED_SOURCE_LIST_SHA=bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857
readonly APPROVED_SOURCE_SHA=7ac7ea66485b319672e9b975ffcd38caa2c607f8932d1ca2acc2a9c5159823b1
readonly APPROVED_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r3_centerhead_96e509b71a3e
readonly APPROVED_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r3_centerhead_96e509b71a3e
readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

[[ "$EXPECTED_S05R3_SHA" == "$APPROVED_SHA" ]]
[[ "$EXPECTED_S05R3_TREE" == "$APPROVED_TREE" ]]
[[ "$EXPECTED_S05R3_SOURCE_LIST_SHA" == "$APPROVED_SOURCE_LIST_SHA" ]]
[[ "$EXPECTED_S05R3_SOURCE_SHA" == "$APPROVED_SOURCE_SHA" ]]
[[ "$S05R3_SNAPSHOT_ROOT" == "$APPROVED_SNAPSHOT_ROOT" ]]
[[ "$S05R3_OUTPUT_ROOT" == "$APPROVED_OUTPUT_ROOT" ]]
[[ -d "$COMMON_GIT_DIR/objects" ]]
[[ ! -e "$S05R3_SNAPSHOT_ROOT" ]]
[[ ! -e "$S05R3_OUTPUT_ROOT" ]]
[[ -f "$S05R3_REQUEST_COPY" ]]
[[ "$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')" == "$EXPECTED_S05R3_LAUNCHER_SHA" ]]
[[ "$(sha256sum "$S05R3_REQUEST_COPY" | awk '{print $1}')" == "$EXPECTED_S05R3_RUN_REQUEST_SHA" ]]
[[ "$(git --git-dir="$COMMON_GIT_DIR" rev-parse "$APPROVED_SHA^{tree}")" == "$APPROVED_TREE" ]]

readonly JOB_DESC="$(scontrol show job -o "${SLURM_JOB_ID:?SLURM_JOB_ID required}")"
python3 - "$JOB_DESC" <<'PY'
import re, sys
desc = sys.argv[1]
for token in ("NumNodes=1 ", "NumCPUs=8 "):
    if token not in desc:
        raise SystemExit(f"allocation mismatch: missing {token!r}: {desc}")
match = re.search(r"AllocTRES=([^ ]+)", desc)
if match is None:
    raise SystemExit(f"missing AllocTRES: {desc}")
tres = dict(item.split("=", 1) for item in match.group(1).split(",") if "=" in item)
if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:nvidia_gh200_120gb") != "1":
    raise SystemExit(f"expected exactly one GH200 allocation: {match.group(1)}")
PY
: "${CUDA_VISIBLE_DEVICES:?exactly one Slurm-visible GPU is required}"
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#visible_devices[@]}" -eq 1 ]]
[[ -n "${visible_devices[0]}" && "${visible_devices[0]}" != "-1" ]]

readonly TMP_SNAPSHOT="${S05R3_SNAPSHOT_ROOT}.tmp.${SLURM_JOB_ID}"
[[ ! -e "$TMP_SNAPSHOT" ]]
mkdir -p "$(dirname "$S05R3_SNAPSHOT_ROOT")"
mkdir "$TMP_SNAPSHOT"
cleanup() {
  if [[ -d "$TMP_SNAPSHOT" ]]; then
    chmod -R u+w "$TMP_SNAPSHOT"
    rm -rf -- "$TMP_SNAPSHOT"
  fi
}
trap cleanup EXIT
git --git-dir="$COMMON_GIT_DIR" archive --format=tar "$APPROVED_SHA" | tar -xf - -C "$TMP_SNAPSHOT"

runtime_source_files() {
  {
    printf '%s\n' \
      fl_v3/pyproject.toml \
      fl_v3/requirements.lock.txt \
      fl_v3/requirements.txt \
      fl_v3/scripts/arrhenius_env.sh \
      fl_v3/src/fl_v3/__init__.py \
      fl_v3/src/fl_v3/data/__init__.py \
      fl_v3/src/fl_v3/eval/__init__.py \
      fl_v3/src/fl_v3/eval/box_to_global.py \
      fl_v3/src/fl_v3/eval/detection_eval.py \
      fl_v3/src/fl_v3/models/__init__.py \
      fl_v3/src/fl_v3/models/fusion/__init__.py \
      fl_v3/src/fl_v3/models/fusion/bev_grid.py \
      fl_v3/src/fl_v3/models/fusion/centerhead_decode.py \
      fl_v3/src/fl_v3/models/fusion/head.py \
      fl_v3/src/fl_v3/models/fusion/nms_deterministic.py \
      fl_v3/tests/test_head_capacity.py \
      fl_v3/tests/test_s05_centerhead_decode.py \
      fl_v3/tests/test_s05_eval_roundtrip.py \
      fl_v3/tests/test_s05_nms.py
    git --git-dir="$COMMON_GIT_DIR" ls-tree -r --name-only "$APPROVED_SHA" \
      fl_v3/src/fl_v3/data/nuscenes | grep '\.py$'
  } | LC_ALL=C sort -u
}
readonly ACTUAL_SOURCE_LIST_SHA="$(runtime_source_files | sha256sum | awk '{print $1}')"
readonly ACTUAL_SOURCE_SHA="$({
  while IFS= read -r path; do
    sha256sum "$TMP_SNAPSHOT/$path" | sed "s|  $TMP_SNAPSHOT/|  |"
  done < <(runtime_source_files)
} | sha256sum | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_LIST_SHA" == "$APPROVED_SOURCE_LIST_SHA" ]]
[[ "$ACTUAL_SOURCE_SHA" == "$APPROVED_SOURCE_SHA" ]]

cat > "$TMP_SNAPSHOT/.s05r3_snapshot_identity" <<EOF
schema=s05r3.centerhead-synthetic.v1
worker_sha=$APPROVED_SHA
worker_tree=$APPROVED_TREE
source_list_sha256=$ACTUAL_SOURCE_LIST_SHA
source_sha256=$ACTUAL_SOURCE_SHA
launcher_sha256=$EXPECTED_S05R3_LAUNCHER_SHA
run_request_sha256=$EXPECTED_S05R3_RUN_REQUEST_SHA
slurm_job_id=$SLURM_JOB_ID
EOF
chmod -R a-w "$TMP_SNAPSHOT"
mv "$TMP_SNAPSHOT" "$S05R3_SNAPSHOT_ROOT"
trap - EXIT
readonly REPO="$S05R3_SNAPSHOT_ROOT"
if find "$REPO" -xdev -perm /222 -print -quit | grep -q .; then
  echo "snapshot is writable" >&2
  exit 2
fi

cd "$REPO"
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env
export PYTHONPATH="$REPO/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTEST_ADDOPTS
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"expected one visible GH200, available={torch.cuda.is_available()} "
        f"count={torch.cuda.device_count()}"
    )
PY

mkdir -p "$S05R3_OUTPUT_ROOT/tmp"
export TMPDIR="$S05R3_OUTPUT_ROOT/tmp"
cp "$SCRIPT_PATH" "$S05R3_OUTPUT_ROOT/approved_launcher.sh"
cp "$S05R3_REQUEST_COPY" "$S05R3_OUTPUT_ROOT/approved_run_request.md"
cp "$REPO/.s05r3_snapshot_identity" "$S05R3_OUTPUT_ROOT/snapshot_identity.txt"
runtime_source_files > "$S05R3_OUTPUT_ROOT/runtime_source_files.txt"
while IFS= read -r path; do
  sha256sum "$REPO/$path" | sed "s|  $REPO/|  |"
done < "$S05R3_OUTPUT_ROOT/runtime_source_files.txt" \
  > "$S05R3_OUTPUT_ROOT/runtime_source_sha256s.txt"
[[ "$(sha256sum "$S05R3_OUTPUT_ROOT/runtime_source_files.txt" | awk '{print $1}')" == "$APPROVED_SOURCE_LIST_SHA" ]]
[[ "$(sha256sum "$S05R3_OUTPUT_ROOT/runtime_source_sha256s.txt" | awk '{print $1}')" == "$APPROVED_SOURCE_SHA" ]]

python - "$S05R3_OUTPUT_ROOT/execution_identity.json" "$JOB_DESC" <<'PY'
import importlib.metadata, json, os, platform, socket, sys
out, job_desc = sys.argv[1:]
record = {
    "schema": "s05r3.centerhead-synthetic.v1",
    "worker_sha": os.environ["EXPECTED_S05R3_SHA"],
    "worker_tree": os.environ["EXPECTED_S05R3_TREE"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S05R3_SOURCE_LIST_SHA"],
    "runtime_source_sha256": os.environ["EXPECTED_S05R3_SOURCE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S05R3_LAUNCHER_SHA"],
    "run_request_sha256": os.environ["EXPECTED_S05R3_RUN_REQUEST_SHA"],
    "snapshot_root": os.environ["S05R3_SNAPSHOT_ROOT"],
    "output_root": os.environ["S05R3_OUTPUT_ROOT"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_description": job_desc,
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pytest", "nuscenes-devkit", "Pillow")
    },
    "synthetic_only": True,
    "dataset_access": False,
    "optimizer_or_parameter_update": False,
    "scientific_metric": False,
    "prior_failed_job": "336731",
}
with open(out, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
printf '%s\n' "$JOB_DESC" > "$S05R3_OUTPUT_ROOT/slurm_allocation.txt"

set +e
python -m pytest -q -ra --noconftest -p no:cacheprovider \
  fl_v3/tests/test_head_capacity.py \
  fl_v3/tests/test_s05_centerhead_decode.py \
  fl_v3/tests/test_s05_nms.py \
  fl_v3/tests/test_s05_eval_roundtrip.py \
  --junitxml="$S05R3_OUTPUT_ROOT/pytest.junit.xml" \
  2>&1 | tee "$S05R3_OUTPUT_ROOT/pytest.log"
pytest_status=${PIPESTATUS[0]}
set -e

if [[ "$pytest_status" -eq 0 ]]; then
  set +e
  python - "$S05R3_OUTPUT_ROOT/pytest.junit.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
counts = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
expected = {"tests": 44, "failures": 0, "errors": 0, "skipped": 0}
if counts != expected:
    raise SystemExit(f"S05-R3 JUnit acceptance failed: {counts}")
print(f"S05-R3 JUnit acceptance PASS: {counts}")
PY
  junit_status=$?
  set -e
  if [[ "$junit_status" -ne 0 ]]; then pytest_status="$junit_status"; fi
fi

sha256sum \
  "$S05R3_OUTPUT_ROOT/approved_launcher.sh" \
  "$S05R3_OUTPUT_ROOT/approved_run_request.md" \
  "$S05R3_OUTPUT_ROOT/snapshot_identity.txt" \
  "$S05R3_OUTPUT_ROOT/runtime_source_files.txt" \
  "$S05R3_OUTPUT_ROOT/runtime_source_sha256s.txt" \
  "$S05R3_OUTPUT_ROOT/execution_identity.json" \
  "$S05R3_OUTPUT_ROOT/slurm_allocation.txt" \
  "$S05R3_OUTPUT_ROOT/pytest.log" \
  "$S05R3_OUTPUT_ROOT/pytest.junit.xml" \
  > "$S05R3_OUTPUT_ROOT/sha256sums.txt"
sha256sum -c "$S05R3_OUTPUT_ROOT/sha256sums.txt"
exit "$pytest_status"
