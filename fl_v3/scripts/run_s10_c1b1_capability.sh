#!/bin/bash
# One-shot O-140 C1-B1 matched GN/BN1d D_low capability experiment.
set -euo pipefail
umask 077

: "${S10_C1B1_SNAPSHOT:?required}"
: "${S10_C1B1_OUTPUT:?required}"
: "${S10_C1B1_EXPECTED_SOURCE_SHA:?required}"
: "${S10_C1B1_EXPECTED_TREE:?required}"
: "${S10_C1B1_EXPECTED_RUNNER_SHA256:?required}"
: "${S10_C1B1_EXPECTED_ENTRY_SHA256:?required}"
: "${S10_C1B1_EXPECTED_CONFIG_SHA256:?required}"
: "${S10_C1B1_EXPECTED_GN_RESOLVED_SHA256:?required}"
: "${S10_C1B1_EXPECTED_BN_RESOLVED_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s10_c1b1_capability.sh"
ENTRY_REL="fl_v3/scripts/s10_c1b1_capability.py"
CONFIG_REL="fl_v3/configs/s10_c1b1_cur_a1_gn.json"
SPLIT_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json"
SPLIT_SHA256="7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8"
SWIN_WEIGHTS="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home/hub/checkpoints/swin_t-704ceda3.pth"
SWIN_SHA256="704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b"
ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"
WORK="${S10_C1B1_OUTPUT}.control"

test -d "${S10_C1B1_SNAPSHOT}"
test ! -e "${S10_C1B1_OUTPUT}"
test ! -e "${WORK}"
for relative in "${RUNNER_REL}" "${ENTRY_REL}" "${CONFIG_REL}"; do
  test -f "${S10_C1B1_SNAPSHOT}/${relative}"
done
test -f "${SPLIT_MANIFEST}"
test -f "${SWIN_WEIGHTS}"
test -f "${ZIP_MANIFEST}"
test "$(git -C "${S10_C1B1_SNAPSHOT}" rev-parse HEAD)" = "${S10_C1B1_EXPECTED_SOURCE_SHA}"
test "$(git -C "${S10_C1B1_SNAPSHOT}" rev-parse 'HEAD^{tree}')" = "${S10_C1B1_EXPECTED_TREE}"
test "$(sha256sum "${S10_C1B1_SNAPSHOT}/${RUNNER_REL}" | cut -d' ' -f1)" = "${S10_C1B1_EXPECTED_RUNNER_SHA256}"
test "$(sha256sum "${S10_C1B1_SNAPSHOT}/${ENTRY_REL}" | cut -d' ' -f1)" = "${S10_C1B1_EXPECTED_ENTRY_SHA256}"
test "$(sha256sum "${S10_C1B1_SNAPSHOT}/${CONFIG_REL}" | cut -d' ' -f1)" = "${S10_C1B1_EXPECTED_CONFIG_SHA256}"
test "$(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)" = "${SPLIT_SHA256}"
test "$(sha256sum "${SWIN_WEIGHTS}" | cut -d' ' -f1)" = "${SWIN_SHA256}"
test -z "$(git -C "${S10_C1B1_SNAPSHOT}" status --short --untracked-files=all)"
test -z "$(git -C "${S10_C1B1_SNAPSHOT}" branch --show-current)"

# shellcheck disable=SC1091
source "${S10_C1B1_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S10_C1B1_SNAPSHOT}/fl_v3/src"
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
  local target="${S10_C1B1_OUTPUT}"
  if [[ ! -d "${target}" ]]; then
    mv "${WORK}" "${target}"
  elif [[ -d "${WORK}" ]]; then
    mkdir -p "${target}/control"
    find "${WORK}" -mindepth 1 -maxdepth 1 -exec mv -t "${target}/control" {} +
    rmdir "${WORK}"
  fi
  printf '%s\n' "${status}" > "${target}/final.exit"
  local temporary="${target}/.runner_artifact_sha256s.tmp"
  find "${target}" -type f ! -name runner_artifact_sha256s.txt \
    ! -name .runner_artifact_sha256s.tmp -printf '%P\0' | sort -z | \
    while IFS= read -r -d '' relative; do
      sha256sum "${target}/${relative}" | sed "s#  ${target}/#  #"
    done > "${temporary}"
  mv "${temporary}" "${target}/runner_artifact_sha256s.txt"
  find "${target}" -type f -exec chmod 0444 {} +
  find "${target}" -type d -exec chmod 0555 {} +
}

handle_exit() {
  local status=$?
  trap - EXIT TERM INT HUP QUIT
  if (( status == 0 && runner_complete != 1 )); then status=125; fi
  finalize "${status}"
  exit "${status}"
}
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
trap 'exit 131' QUIT
trap handle_exit EXIT

cd "${S10_C1B1_SNAPSHOT}"
set +e
python -m pytest -q -p no:cacheprovider \
  fl_v3/tests/test_s06_resolved_config.py \
  fl_v3/tests/test_s08_precision_partition.py \
  fl_v3/tests/test_s08_precision_diagnostics.py \
  fl_v3/tests/test_s04_second_contract.py \
  fl_v3/tests/test_sparse_voxel_encoder.py \
  fl_v3/tests/test_s06_checkpoint_resume.py \
  fl_v3/tests/test_s10_binding.py \
  fl_v3/tests/test_s10_c0_health.py \
  fl_v3/tests/test_s10_c1b0.py \
  fl_v3/tests/test_s10_c1b1.py \
  fl_v3/tests/test_s10_subset_eval.py \
  > "${WORK}/focused_tests.stdout" 2> "${WORK}/focused_tests.stderr"
test_status=$?
set -e
printf '%s\n' "${test_status}" > "${WORK}/focused_tests.exit"
if (( test_status != 0 )); then exit "${test_status}"; fi

nvidia-smi --query-gpu=timestamp,name,uuid,memory.total,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader -l 1 > "${WORK}/nvidia_smi_1hz.csv" &
telemetry_pid=$!
set +e
python fl_v3/scripts/s10_c1b1_capability.py \
  --config "${CONFIG_REL}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-dir "${WORK}/evidence" \
  --source-sha "${S10_C1B1_EXPECTED_SOURCE_SHA}" \
  --source-tree "${S10_C1B1_EXPECTED_TREE}" \
  > "${WORK}/c1b1.stdout" 2> "${WORK}/c1b1.stderr"
c1b1_status=$?
set -e
printf '%s\n' "${c1b1_status}" > "${WORK}/c1b1.exit"
if (( c1b1_status != 0 )); then exit "${c1b1_status}"; fi

kill "${telemetry_pid}" 2>/dev/null || true
wait "${telemetry_pid}" 2>/dev/null || true
telemetry_pid=""
SUMMARY="${WORK}/evidence/summary.json"
GN="${WORK}/evidence/C1-B1-CUR-A1-GN-DLOW/cell_summary.json"
BN="${WORK}/evidence/C1-B1-CUR-A1-BN1D-DLOW/cell_summary.json"
jq -e --arg source "${S10_C1B1_EXPECTED_SOURCE_SHA}" '
  .schema == "fl_v3.s10.c1b1_capability.v1" and
  .status == "PASS" and .hard_gate == "PASS" and .source_sha == $source and
  .cell_order == ["C1-B1-CUR-A1-GN-DLOW", "C1-B1-CUR-A1-BN1D-DLOW"] and
  .scientific_selection == "OWNER_DECISION_REQUIRED" and
  .automatic_promotion == false and
  .matched_training_tokens.consumed_samples == 6152 and
  .matched_training_tokens.dropped_samples == 3 and
  .capability.paired_deltas.NDS.clusters == 8 and
  .capability.paired_deltas.mAP.clusters == 8
' "${SUMMARY}" >/dev/null
for tuple in "${GN}:${S10_C1B1_EXPECTED_GN_RESOLVED_SHA256}" "${BN}:${S10_C1B1_EXPECTED_BN_RESOLVED_SHA256}"; do
  file="${tuple%%:*}"
  resolved="${tuple##*:}"
  jq -e --arg source "${S10_C1B1_EXPECTED_SOURCE_SHA}" --arg resolved "${resolved}" '
    .schema == "fl_v3.s10.c1b1_capability.v1" and .source_sha == $source and
    .resolved_config_sha256 == $resolved and
    .terminal_training_state.attempted_windows == 1538 and
    .terminal_training_state.optimizer_step == 1538 and
    .terminal_training_state.invalid_windows == 0 and
    .terminal_training_state.nonfinite_windows == 0 and
    .terminal_training_state.overflow_windows == 0 and
    .terminal_training_state.discarded_windows == 0 and
    .training_token_evidence.consumed_sample_count == 6152 and
    .training_token_evidence.drop_last_remainder_count == 3 and
    .D_select_evaluation.n_samples == 4626 and
    (.health.hard_errors | length) == 0
  ' "${file}" >/dev/null
done
test "$(jq -r '.initial_parameter_sha256' "${GN}")" = "$(jq -r '.initial_parameter_sha256' "${BN}")"
test "$(jq -r '.training_token_evidence.consumed_sample_tokens_ordered_sha256' "${GN}")" = \
  "$(jq -r '.training_token_evidence.consumed_sample_tokens_ordered_sha256' "${BN}")"
test "$(jq -r '.training_token_evidence.drop_last_remainder_tokens_sorted_sha256' "${GN}")" = \
  "$(jq -r '.training_token_evidence.drop_last_remainder_tokens_sorted_sha256' "${BN}")"
test -s "${WORK}/evidence/C1-B1-CUR-A1-GN-DLOW/checkpoint.pt"
test -s "${WORK}/evidence/C1-B1-CUR-A1-BN1D-DLOW/checkpoint.pt"
runner_complete=1
