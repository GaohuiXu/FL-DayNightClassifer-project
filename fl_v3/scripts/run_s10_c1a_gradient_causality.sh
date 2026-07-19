#!/bin/bash
# One-shot O-134 S10 C1-A GN-versus-BN1d causal diagnostic. No optimizer/update.
set -euo pipefail
umask 077

: "${S10_C1A_SNAPSHOT:?required}"
: "${S10_C1A_OUTPUT:?required}"
: "${S10_C1A_EXPECTED_SOURCE_SHA:?required}"
: "${S10_C1A_EXPECTED_TREE:?required}"
: "${S10_C1A_EXPECTED_RUNNER_SHA256:?required}"
: "${S10_C1A_EXPECTED_ENTRY_SHA256:?required}"
: "${S10_C1A_EXPECTED_SECOND_SHA256:?required}"
: "${S10_C1A_EXPECTED_OBSERVATION_SHA256:?required}"
: "${S10_C1A_EXPECTED_CONFIG_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s10_c1a_gradient_causality.sh"
ENTRY_REL="fl_v3/scripts/s10_c1a_gradient_causality.py"
SECOND_REL="fl_v3/src/fl_v3/models/fusion/second_sparse_backbone.py"
OBSERVATION_REL="fl_v3/src/fl_v3/training/s10_observation.py"
CONFIG_REL="fl_v3/configs/s10_b_rand_l_fp32.json"
SPLIT_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json"
SPLIT_SHA256="7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8"
PANEL_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_8fd832dc7d46_o128_a1/panel_manifest.json"
PANEL_FILE_SHA256="c2826effeba2e074ef8f76ab582bbb5dc796f41b9555348d56e252a2d70138a6"
PANEL_CONTENT_SHA256="8e4f2d992d7a27d771c6fdf00098afc14b9621bc50ea1e52319b84d406f9ad55"
ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"
WORK="${S10_C1A_OUTPUT}.control"

test -d "${S10_C1A_SNAPSHOT}"
test ! -e "${S10_C1A_OUTPUT}"
test ! -e "${WORK}"
test -f "${S10_C1A_SNAPSHOT}/${RUNNER_REL}"
test -f "${S10_C1A_SNAPSHOT}/${ENTRY_REL}"
test -f "${S10_C1A_SNAPSHOT}/${SECOND_REL}"
test -f "${S10_C1A_SNAPSHOT}/${OBSERVATION_REL}"
test -f "${S10_C1A_SNAPSHOT}/${CONFIG_REL}"
test -f "${SPLIT_MANIFEST}"
test -f "${PANEL_MANIFEST}"
test -f "${ZIP_MANIFEST}"

actual_source_sha="$(git -C "${S10_C1A_SNAPSHOT}" rev-parse HEAD)"
actual_tree="$(git -C "${S10_C1A_SNAPSHOT}" rev-parse 'HEAD^{tree}')"
test "${actual_source_sha}" = "${S10_C1A_EXPECTED_SOURCE_SHA}"
test "${actual_tree}" = "${S10_C1A_EXPECTED_TREE}"
test "$(sha256sum "${S10_C1A_SNAPSHOT}/${RUNNER_REL}" | cut -d' ' -f1)" = "${S10_C1A_EXPECTED_RUNNER_SHA256}"
test "$(sha256sum "${S10_C1A_SNAPSHOT}/${ENTRY_REL}" | cut -d' ' -f1)" = "${S10_C1A_EXPECTED_ENTRY_SHA256}"
test "$(sha256sum "${S10_C1A_SNAPSHOT}/${SECOND_REL}" | cut -d' ' -f1)" = "${S10_C1A_EXPECTED_SECOND_SHA256}"
test "$(sha256sum "${S10_C1A_SNAPSHOT}/${OBSERVATION_REL}" | cut -d' ' -f1)" = "${S10_C1A_EXPECTED_OBSERVATION_SHA256}"
test "$(sha256sum "${S10_C1A_SNAPSHOT}/${CONFIG_REL}" | cut -d' ' -f1)" = "${S10_C1A_EXPECTED_CONFIG_SHA256}"
test "$(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)" = "${SPLIT_SHA256}"
test "$(sha256sum "${PANEL_MANIFEST}" | cut -d' ' -f1)" = "${PANEL_FILE_SHA256}"
test -z "$(git -C "${S10_C1A_SNAPSHOT}" status --short --untracked-files=all)"
test "$(git -C "${S10_C1A_SNAPSHOT}" branch --show-current)" = ""

# shellcheck disable=SC1091
source "${S10_C1A_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S10_C1A_SNAPSHOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="${ZIP_MANIFEST}"

test "${NUSCENES_DATAROOT}" = "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
test "${SLURM_JOB_PARTITION:-}" = "gpu"
test "${SLURM_CPUS_PER_TASK:-}" = "8"
test "${SLURM_MEM_PER_NODE:-}" = "65536"
test -n "${SLURM_JOB_GPUS:-}"
test "${SLURM_GPUS_ON_NODE:-0}" = "1"
mkdir -p "${WORK}"
runner_complete=0

finalize() {
  local status="${1:-1}"
  local target="${S10_C1A_OUTPUT}"
  if [[ ! -d "${target}" ]]; then
    mv "${WORK}" "${target}"
  elif [[ -d "${WORK}" ]]; then
    mkdir -p "${target}/control"
    find "${WORK}" -mindepth 1 -maxdepth 1 -exec mv -t "${target}/control" {} +
    rmdir "${WORK}"
  fi
  printf '%s\n' "${status}" > "${target}/final.exit"
  local temporary="${target}/.runner_artifact_sha256s.tmp"
  find "${target}" -type f ! -name runner_artifact_sha256s.txt ! -name .runner_artifact_sha256s.tmp \
    -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do
      sha256sum "${target}/${relative}" | sed "s#  ${target}/#  #"
    done > "${temporary}"
  mv "${temporary}" "${target}/runner_artifact_sha256s.txt"
  find "${target}" -type f -exec chmod 0444 {} +
  find "${target}" -type d -exec chmod 0555 {} +
}

handle_signal() { exit "$1"; }

handle_exit() {
  local status=$?
  trap - EXIT TERM INT HUP QUIT
  if (( status == 0 && runner_complete != 1 )); then status=125; fi
  finalize "${status}"
  exit "${status}"
}

trap 'handle_signal 143' TERM
trap 'handle_signal 130' INT
trap 'handle_signal 129' HUP
trap 'handle_signal 131' QUIT
trap handle_exit EXIT

cd "${S10_C1A_SNAPSHOT}"
python - <<'PY' > "${WORK}/resource_identity.json"
import json
import os
import platform
import torch

if platform.machine() != "aarch64":
    raise RuntimeError("C1-A requires an aarch64 compute node")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("C1-A requires exactly one process-visible CUDA device")
name = torch.cuda.get_device_properties(0).name
if "GH200" not in name.upper():
    raise RuntimeError(f"C1-A expected GH200, got {name!r}")
print(json.dumps({
    "schema": "fl_v3.s10.c1a_resource_identity.v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": platform.node(),
    "machine": platform.machine(),
    "device_name": name,
    "device_count": torch.cuda.device_count(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
}, sort_keys=True))
PY
nvidia-smi --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "${WORK}/nvidia_smi_before.csv"

set +e
python -m pytest -q -p no:cacheprovider \
  fl_v3/tests/test_s04_second_contract.py \
  fl_v3/tests/test_s10_observation.py \
  fl_v3/tests/test_s10_binding.py \
  fl_v3/tests/test_s08_precision_partition.py \
  fl_v3/tests/test_model_task.py::test_detection_task_registered \
  fl_v3/tests/test_model_task.py::test_detection_config_rejects_legacy_model_mode_alias \
  > "${WORK}/focused_tests.stdout" 2> "${WORK}/focused_tests.stderr"
test_status=$?
set -e
printf '%s\n' "${test_status}" > "${WORK}/focused_tests.exit"
if (( test_status != 0 )); then exit "${test_status}"; fi

set +e
python "${ENTRY_REL}" \
  --config "${CONFIG_REL}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --split-sha256 "${SPLIT_SHA256}" \
  --panel-manifest "${PANEL_MANIFEST}" \
  --panel-file-sha256 "${PANEL_FILE_SHA256}" \
  --panel-content-sha256 "${PANEL_CONTENT_SHA256}" \
  --output-dir "${S10_C1A_OUTPUT}" \
  --source-sha "${actual_source_sha}" \
  --source-tree "${actual_tree}" \
  > "${WORK}/c1a.stdout" 2> "${WORK}/c1a.stderr"
c1a_status=$?
set -e
printf '%s\n' "${c1a_status}" > "${WORK}/c1a.exit"
nvidia-smi --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "${WORK}/nvidia_smi_after.csv" || true
runner_complete=1
exit "${c1a_status}"
