#!/bin/bash
# Exact O-025 option-A synthetic validation; submit once only after S00 approval.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s04_option_a
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_option_a_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s04_option_a_%j.err
set -euo pipefail

required=(
  EXPECTED_S04_O025_SHA EXPECTED_S04_O025_TREE EXPECTED_S04_O025_SOURCE_HASH
  EXPECTED_S04_O025_REQUEST_HASH EXPECTED_S04_O025_IDENTITY_HASH
  S04_O025_SNAPSHOT_ROOT S04_O025_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then echo "${name} is required" >&2; exit 2; fi
done
if [ -e "${S04_O025_OUTPUT_ROOT}" ]; then
  echo "refusing output reuse: ${S04_O025_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="$(realpath "${S04_O025_SNAPSHOT_ROOT}")"
case "${REPO}" in
  /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s04_o025_*) ;;
  *) echo "invalid snapshot root: ${REPO}" >&2; exit 2 ;;
esac
test "$(pwd -P)" = "${REPO}"
test "$(realpath "${SLURM_SUBMIT_DIR:?}")" = "${REPO}"
cd "${REPO}"
if find "${REPO}" -xdev \( -type f -o -type d \) -perm /0222 -print -quit | grep -q .; then
  echo "snapshot is writable" >&2
  exit 2
fi

IDENTITY=.s04_option_a_snapshot_identity
REQUEST=fl_v3/usenix27_orchestra/handoffs/S04/RUN_REQUEST.md
test "$(sha256sum "${IDENTITY}" | awk '{print $1}')" = "${EXPECTED_S04_O025_IDENTITY_HASH}"
test "$(sha256sum "${REQUEST}" | awk '{print $1}')" = "${EXPECTED_S04_O025_REQUEST_HASH}"
EXPECTED_IDENTITY="$(printf '%s\n' \
  'schema=s04.option-a-snapshot.v1' \
  "exec_sha=${EXPECTED_S04_O025_SHA}" \
  "exec_tree=${EXPECTED_S04_O025_TREE}" \
  "source_sha256=${EXPECTED_S04_O025_SOURCE_HASH}" \
  "request_sha256=${EXPECTED_S04_O025_REQUEST_HASH}")"
test "$(cat "${IDENTITY}")" = "${EXPECTED_IDENTITY}"

runtime_source_files() {
  printf '%s\n' \
    fl_v3/src/fl_v3/__init__.py \
    fl_v3/src/fl_v3/models/__init__.py \
    fl_v3/src/fl_v3/models/fusion/__init__.py \
    fl_v3/src/fl_v3/models/fusion/bev_grid.py \
    fl_v3/src/fl_v3/models/fusion/second_sparse_backbone.py \
    fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py \
    fl_v3/src/fl_v3/utils/__init__.py \
    fl_v3/src/fl_v3/utils/runtime.py \
    fl_v3/tests/conftest.py \
    fl_v3/tests/test_s04_fp16_eval_dispatch.py \
    fl_v3/tests/test_s04_second_contract.py \
    fl_v3/tests/test_s04_second_smoke.py \
    fl_v3/tests/test_sparse_voxel_encoder.py \
    fl_v3/pyproject.toml \
    fl_v3/requirements.txt \
    fl_v3/requirements.lock.txt \
    fl_v3/scripts/arrhenius_env.sh \
    fl_v3/usenix27_orchestra/handoffs/S04/run_s04_option_a_validation.sh \
    | LC_ALL=C sort -u
}
SOURCE_HASH="$(runtime_source_files | while IFS= read -r path; do sha256sum "${path}"; done | sha256sum | awk '{print $1}')"
test "${SOURCE_HASH}" = "${EXPECTED_S04_O025_SOURCE_HASH}"

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTEST_ADDOPTS

mkdir -p "${S04_O025_OUTPUT_ROOT}"
export TMPDIR="${S04_O025_OUTPUT_ROOT}/tmp"
mkdir -p "${TMPDIR}"
SOURCE_HASHES="${S04_O025_OUTPUT_ROOT}/runtime_source_sha256s.txt"
EXECUTION_JSON="${S04_O025_OUTPUT_ROOT}/execution_identity.json"
PYTEST_LOG="${S04_O025_OUTPUT_ROOT}/pytest.log"
JUNIT_XML="${S04_O025_OUTPUT_ROOT}/pytest.junit.xml"
runtime_source_files | while IFS= read -r path; do sha256sum "${path}"; done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${SOURCE_HASH}"

JOB_DESC="$(scontrol show job -o "${SLURM_JOB_ID:?}")"
python - "${JOB_DESC}" <<'PY'
import re, sys
desc = sys.argv[1]
required = ("NumNodes=1 ", "NumCPUs=8 ", "TresPerNode=gres/gpu:nvidia_gh200_120gb:1")
missing = [value for value in required if value not in desc]
match = re.search(r"AllocTRES=([^ ]+)", desc)
if missing or match is None:
    raise SystemExit(f"allocation mismatch missing={missing}: {desc}")
tres = dict(item.split("=", 1) for item in match.group(1).split(",") if "=" in item)
if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:nvidia_gh200_120gb") != "1":
    raise SystemExit(f"expected one GH200: {match.group(1)}")
PY

python - "${EXECUTION_JSON}" "${JOB_DESC}" <<'PY'
import importlib.metadata, json, os, platform, socket, sys, torch
output, job_desc = sys.argv[1:]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one visible CUDA GPU is required")
versions = {name: importlib.metadata.version(name) for name in ("numpy", "pytest", "torch", "spconv", "cumm")}
if versions["spconv"] != "2.3.8":
    raise SystemExit(f"O-025 requires spconv 2.3.8, found {versions['spconv']}")
record = {
    "schema": "s04.option-a-execution.v1",
    "git_sha": os.environ["EXPECTED_S04_O025_SHA"],
    "git_tree": os.environ["EXPECTED_S04_O025_TREE"],
    "snapshot_root": os.environ["S04_O025_SNAPSHOT_ROOT"],
    "source_sha256": os.environ["EXPECTED_S04_O025_SOURCE_HASH"],
    "request_sha256": os.environ["EXPECTED_S04_O025_REQUEST_HASH"],
    "identity_sha256": os.environ["EXPECTED_S04_O025_IDENTITY_HASH"],
    "job_id": os.environ["SLURM_JOB_ID"],
    "job_description": job_desc,
    "host": socket.gethostname(), "machine": platform.machine(),
    "python": platform.python_version(), "dependencies": versions,
    "synthetic_only": True, "optimizer_step": False, "scientific_metrics": False,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
PY

set +e
python -m pytest -q -ra -s -p no:cacheprovider \
  fl_v3/tests/test_s04_second_contract.py \
  fl_v3/tests/test_sparse_voxel_encoder.py \
  fl_v3/tests/test_s04_fp16_eval_dispatch.py \
  fl_v3/tests/test_s04_second_smoke.py \
  --junitxml="${JUNIT_XML}" 2>&1 | tee "${PYTEST_LOG}"
pytest_status=${PIPESTATUS[0]}
set -e

if [ "${pytest_status}" -eq 0 ]; then
  python - "${JUNIT_XML}" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
counts = {key: sum(int(s.attrib.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
if counts != {"tests": 15, "failures": 0, "errors": 0, "skipped": 0}:
    raise SystemExit(f"acceptance mismatch: {counts}")
print(f"S04_O025_JUNIT={counts}")
PY
fi

sha256sum "${EXECUTION_JSON}" "${SOURCE_HASHES}" "${PYTEST_LOG}" "${JUNIT_XML}" > "${S04_O025_OUTPUT_ROOT}/sha256sums.txt"
sha256sum -c "${S04_O025_OUTPUT_ROOT}/sha256sums.txt"
exit "${pytest_status}"
