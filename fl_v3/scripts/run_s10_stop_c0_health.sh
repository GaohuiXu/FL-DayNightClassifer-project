#!/bin/bash
# One-shot O-132 STOP-C0-v2 clean-replay health/training/telemetry rung.
set -euo pipefail
umask 077

: "${S10_C0_SNAPSHOT:?required}"
: "${S10_C0_OUTPUT:?required}"
: "${S10_C0_EXPECTED_SOURCE_SHA:?required}"
: "${S10_C0_EXPECTED_TREE:?required}"
: "${S10_C0_EXPECTED_RUNNER_SHA256:?required}"
: "${S10_C0_EXPECTED_ENTRY_SHA256:?required}"
: "${S10_C0_EXPECTED_CONFIG_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s10_stop_c0_health.sh"
ENTRY_REL="fl_v3/scripts/s10_stop_c0_health.py"
CONFIG_REL="fl_v3/configs/s10_c0_f_a1.json"
SPLIT_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json"
SPLIT_SHA256="7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8"
SWIN_WEIGHTS="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home/hub/checkpoints/swin_t-704ceda3.pth"
SWIN_SHA256="704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b"
ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"
WORK="${S10_C0_OUTPUT}.control"

test -d "${S10_C0_SNAPSHOT}"
test ! -e "${S10_C0_OUTPUT}"
test ! -e "${WORK}"
for relative in "${RUNNER_REL}" "${ENTRY_REL}" "${CONFIG_REL}"; do
  test -f "${S10_C0_SNAPSHOT}/${relative}"
done
test -f "${SPLIT_MANIFEST}"
test -f "${SWIN_WEIGHTS}"
test -f "${ZIP_MANIFEST}"

test "$(git -C "${S10_C0_SNAPSHOT}" rev-parse HEAD)" = "${S10_C0_EXPECTED_SOURCE_SHA}"
test "$(git -C "${S10_C0_SNAPSHOT}" rev-parse 'HEAD^{tree}')" = "${S10_C0_EXPECTED_TREE}"
test "$(sha256sum "${S10_C0_SNAPSHOT}/${RUNNER_REL}" | cut -d' ' -f1)" = "${S10_C0_EXPECTED_RUNNER_SHA256}"
test "$(sha256sum "${S10_C0_SNAPSHOT}/${ENTRY_REL}" | cut -d' ' -f1)" = "${S10_C0_EXPECTED_ENTRY_SHA256}"
test "$(sha256sum "${S10_C0_SNAPSHOT}/${CONFIG_REL}" | cut -d' ' -f1)" = "${S10_C0_EXPECTED_CONFIG_SHA256}"
test "$(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)" = "${SPLIT_SHA256}"
test "$(sha256sum "${SWIN_WEIGHTS}" | cut -d' ' -f1)" = "${SWIN_SHA256}"
test -z "$(git -C "${S10_C0_SNAPSHOT}" status --short --untracked-files=all)"
test "$(git -C "${S10_C0_SNAPSHOT}" branch --show-current)" = ""

# shellcheck disable=SC1091
source "${S10_C0_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S10_C0_SNAPSHOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export TORCH_HOME="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home"
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="${ZIP_MANIFEST}"

test "${NUSCENES_DATAROOT}" = "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
test "${SLURM_JOB_PARTITION:-}" = "gpu"
test "${SLURM_CPUS_PER_TASK:-}" = "16"
test "${SLURM_MEM_PER_NODE:-}" = "98304"
test -n "${SLURM_JOB_GPUS:-}"
test "${SLURM_GPUS_ON_NODE:-0}" = "1"
mkdir -p "${WORK}"
runner_complete=0
telemetry_pid=""

finalize() {
  local status="${1:-1}"
  if [[ -n "${telemetry_pid}" ]]; then
    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
    telemetry_pid=""
  fi
  local target="${S10_C0_OUTPUT}"
  if [[ ! -d "${target}" ]]; then
    mv "${WORK}" "${target}"
  elif [[ -d "${WORK}" ]]; then
    mkdir -p "${target}/control"
    find "${WORK}" -mindepth 1 -maxdepth 1 -exec mv -t "${target}/control" {} +
    rmdir "${WORK}"
  fi
  printf '%s\n' "${status}" > "${target}/final.exit"
  local temporary="${target}/.runner_artifact_sha256s.tmp"
  find "${target}" -type f \
    ! -name runner_artifact_sha256s.txt \
    ! -name .runner_artifact_sha256s.tmp \
    -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do
      sha256sum "${target}/${relative}" | sed "s#  ${target}/#  #"
    done > "${temporary}"
  mv "${temporary}" "${target}/runner_artifact_sha256s.txt"
  find "${target}" -type f -exec chmod 0444 {} +
  find "${target}" -type d -exec chmod 0555 {} +
}

handle_exit() {
  local status=$?
  trap - EXIT TERM INT HUP QUIT
  if (( status == 0 && runner_complete != 1 )); then
    status=125
  fi
  finalize "${status}"
  exit "${status}"
}

trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
trap 'exit 131' QUIT
trap handle_exit EXIT

cd "${S10_C0_SNAPSHOT}"
python - <<'PY' > "${WORK}/resource_identity.json"
import json
import os
import platform
import torch

if platform.machine() != "aarch64":
    raise RuntimeError("STOP-C0 requires an aarch64 compute node")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("STOP-C0 requires exactly one visible CUDA device")
name = torch.cuda.get_device_properties(0).name
if "GH200" not in name.upper():
    raise RuntimeError(f"STOP-C0 expected GH200, got {name!r}")
print(json.dumps({
    "schema": "fl_v3.s10.stop_c0_resource_identity.v1",
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

set +e
python -m pytest -q -p no:cacheprovider \
  fl_v3/tests/test_s08_precision_partition.py \
  fl_v3/tests/test_s08_precision_diagnostics.py \
  fl_v3/tests/test_s10_binding.py \
  fl_v3/tests/test_s10_c0_health.py \
  fl_v3/tests/test_s10_subset_eval.py \
  fl_v3/tests/test_profiling_neutral.py \
  fl_v3/tests/test_s09_readiness.py \
  fl_v3/tests/test_s06_checkpoint_resume.py \
  > "${WORK}/focused_tests.stdout" 2> "${WORK}/focused_tests.stderr"
test_status=$?
set -e
printf '%s\n' "${test_status}" > "${WORK}/focused_tests.exit"
if (( test_status != 0 )); then
  exit "${test_status}"
fi

nvidia-smi --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader -l 1 > "${WORK}/nvidia_smi_1hz.csv" &
telemetry_pid=$!

set +e
python fl_v3/scripts/s10_stop_c0_health.py \
  --config "${CONFIG_REL}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --out-dir "${WORK}/evidence" \
  --source-sha "${S10_C0_EXPECTED_SOURCE_SHA}" \
  --swin-weights "${SWIN_WEIGHTS}" \
  > "${WORK}/c0.stdout" 2> "${WORK}/c0.stderr"
c0_status=$?
set -e
printf '%s\n' "${c0_status}" > "${WORK}/c0.exit"
if (( c0_status != 0 )); then
  exit "${c0_status}"
fi

kill "${telemetry_pid}" 2>/dev/null || true
wait "${telemetry_pid}" 2>/dev/null || true
telemetry_pid=""
SUMMARY="${WORK}/evidence/c0_summary.json"
F_SUMMARY="${WORK}/evidence/C0-F-A1/cell_summary.json"
L_SUMMARY="${WORK}/evidence/C0-L-A0/cell_summary.json"
S_SUMMARY="${WORK}/evidence/C0-F-A0-P64/cell_summary.json"
jq -e --arg source "${S10_C0_EXPECTED_SOURCE_SHA}" '
  .schema == "fl_v3.s10.stop_c0_health.v2" and
  .status == "PASS" and
  .source_sha == $source and
  .hard_failures == {} and
  .cell_order == ["C0-F-A1", "C0-L-A0", "C0-F-A0-P64"]
' "${SUMMARY}" >/dev/null
for cell_summary in "${F_SUMMARY}" "${L_SUMMARY}"; do
  jq -e --arg source "${S10_C0_EXPECTED_SOURCE_SHA}" '
    .schema == "fl_v3.s10.stop_c0_health.v2" and
    .source_sha == $source and
    .terminal_training_state.attempted_windows == 1538 and
    .training_token_evidence.source == "actual_collated_batches" and
    .training_token_evidence.consumed_sample_count == 6152 and
    .training_token_evidence.full_epoch == true and
    .training_token_evidence.drop_last_remainder_count == 3 and
    (.training_token_evidence.drop_last_remainder_tokens_sorted | length) == 3 and
    (.health.hard_errors | length) == 0
  ' "${cell_summary}" >/dev/null
done
jq -e --arg source "${S10_C0_EXPECTED_SOURCE_SHA}" '
  .schema == "fl_v3.s10.stop_c0_health.v2" and
  .source_sha == $source and
  .terminal_training_state.attempted_windows == 64 and
  .training_token_evidence.source == "actual_collated_batches" and
  .training_token_evidence.consumed_sample_count == 256 and
  .training_token_evidence.full_epoch == false and
  .training_token_evidence.drop_last_remainder_count == null and
  .training_token_evidence.drop_last_remainder_tokens_sorted == null and
  (.health.hard_errors | length) == 0
' "${S_SUMMARY}" >/dev/null
test "$(jq -r '.training_token_evidence.consumed_sample_tokens_ordered_sha256' "${F_SUMMARY}")" = \
  "$(jq -r '.training_token_evidence.consumed_sample_tokens_ordered_sha256' "${L_SUMMARY}")"
test "$(jq -r '.training_token_evidence.drop_last_remainder_tokens_sorted_sha256' "${F_SUMMARY}")" = \
  "$(jq -r '.training_token_evidence.drop_last_remainder_tokens_sorted_sha256' "${L_SUMMARY}")"
test -s "${WORK}/evidence/C0-F-A1/checkpoint.pt"
test -s "${WORK}/evidence/C0-L-A0/checkpoint.pt"
runner_complete=1
