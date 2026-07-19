#!/bin/bash
# One-shot O-141 BN1d physical-B8 operational completion.
set -euo pipefail
umask 077

: "${S10_C1B8_SNAPSHOT:?required}"
: "${S10_C1B8_OUTPUT:?required}"
: "${S10_C1B8_EXPECTED_SOURCE_SHA:?required}"
: "${S10_C1B8_EXPECTED_TREE:?required}"
: "${S10_C1B8_EXPECTED_RUNNER_SHA256:?required}"
: "${S10_C1B8_EXPECTED_ENTRY_SHA256:?required}"
: "${S10_C1B8_EXPECTED_CONFIG_SHA256:?required}"
: "${S10_C1B8_EXPECTED_RESOLVED_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s10_c1b1_bn_b8.sh"
ENTRY_REL="fl_v3/scripts/s10_c1b1_bn_b8.py"
CONFIG_REL="fl_v3/configs/s10_c1b1_bn_b8.json"
SPLIT_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json"
SPLIT_SHA256="7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8"
REFERENCE_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_c1b1_239cd62_o140_a1/evidence"
GN_SUMMARY_SHA256="81e31258dd783f47e8775a1b1327dbac66b0cf9b005fcf6e5f4249c98d61ea85"
GN_RESULTS_SHA256="7fc24fd757d9302096c27208c58469fdd335f22fe363a70bb32ab76875f1e549"
BN_B4_SUMMARY_SHA256="5abc990577ec99eb04f2b9fc063ecba648d342891ce3e50159b4adff62537517"
BN_B4_RESULTS_SHA256="124eddeee78d5fd3495a3f1cff820a5ab82f5aebaef4cfd4ab9996a926966268"
SWIN_WEIGHTS="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home/hub/checkpoints/swin_t-704ceda3.pth"
SWIN_SHA256="704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b"
ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"
WORK="${S10_C1B8_OUTPUT}.control"

test -d "${S10_C1B8_SNAPSHOT}"
test ! -e "${S10_C1B8_OUTPUT}"
test ! -e "${WORK}"
for relative in "${RUNNER_REL}" "${ENTRY_REL}" "${CONFIG_REL}"; do
  test -f "${S10_C1B8_SNAPSHOT}/${relative}"
done
test -f "${SPLIT_MANIFEST}"
test -f "${SWIN_WEIGHTS}"
test -f "${ZIP_MANIFEST}"
test "$(git -C "${S10_C1B8_SNAPSHOT}" rev-parse HEAD)" = "${S10_C1B8_EXPECTED_SOURCE_SHA}"
test "$(git -C "${S10_C1B8_SNAPSHOT}" rev-parse 'HEAD^{tree}')" = "${S10_C1B8_EXPECTED_TREE}"
test "$(sha256sum "${S10_C1B8_SNAPSHOT}/${RUNNER_REL}" | cut -d' ' -f1)" = "${S10_C1B8_EXPECTED_RUNNER_SHA256}"
test "$(sha256sum "${S10_C1B8_SNAPSHOT}/${ENTRY_REL}" | cut -d' ' -f1)" = "${S10_C1B8_EXPECTED_ENTRY_SHA256}"
test "$(sha256sum "${S10_C1B8_SNAPSHOT}/${CONFIG_REL}" | cut -d' ' -f1)" = "${S10_C1B8_EXPECTED_CONFIG_SHA256}"
test "$(sha256sum "${SPLIT_MANIFEST}" | cut -d' ' -f1)" = "${SPLIT_SHA256}"
test "$(sha256sum "${SWIN_WEIGHTS}" | cut -d' ' -f1)" = "${SWIN_SHA256}"
test "$(sha256sum "${REFERENCE_ROOT}/C1-B1-CUR-A1-GN-DLOW/cell_summary.json" | cut -d' ' -f1)" = "${GN_SUMMARY_SHA256}"
test "$(sha256sum "${REFERENCE_ROOT}/C1-B1-CUR-A1-GN-DLOW/D_select_results.json" | cut -d' ' -f1)" = "${GN_RESULTS_SHA256}"
test "$(sha256sum "${REFERENCE_ROOT}/C1-B1-CUR-A1-BN1D-DLOW/cell_summary.json" | cut -d' ' -f1)" = "${BN_B4_SUMMARY_SHA256}"
test "$(sha256sum "${REFERENCE_ROOT}/C1-B1-CUR-A1-BN1D-DLOW/D_select_results.json" | cut -d' ' -f1)" = "${BN_B4_RESULTS_SHA256}"
test -z "$(git -C "${S10_C1B8_SNAPSHOT}" status --short --untracked-files=all)"
test -z "$(git -C "${S10_C1B8_SNAPSHOT}" branch --show-current)"

# shellcheck disable=SC1091
source "${S10_C1B8_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S10_C1B8_SNAPSHOT}/fl_v3/src"
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
  local target="${S10_C1B8_OUTPUT}"
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

cd "${S10_C1B8_SNAPSHOT}"
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
  fl_v3/tests/test_s10_c1b8.py \
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
python fl_v3/scripts/s10_c1b1_bn_b8.py \
  --config "${CONFIG_REL}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-dir "${WORK}/evidence" \
  --source-sha "${S10_C1B8_EXPECTED_SOURCE_SHA}" \
  --source-tree "${S10_C1B8_EXPECTED_TREE}" \
  > "${WORK}/c1b8.stdout" 2> "${WORK}/c1b8.stderr"
c1b8_status=$?
set -e
printf '%s\n' "${c1b8_status}" > "${WORK}/c1b8.exit"
if (( c1b8_status != 0 )); then exit "${c1b8_status}"; fi

kill "${telemetry_pid}" 2>/dev/null || true
wait "${telemetry_pid}" 2>/dev/null || true
telemetry_pid=""
SUMMARY="${WORK}/evidence/summary.json"
CELL_SUMMARY="${WORK}/evidence/C1-B1-BN1D-B8-DLOW/cell_summary.json"
jq -e --arg source "${S10_C1B8_EXPECTED_SOURCE_SHA}" '
  .schema == "fl_v3.s10.c1b1_bn_b8.v1" and
  .status == "PASS" and .hard_gate == "PASS" and .source_sha == $source and
  .cell == "C1-B1-BN1D-B8-DLOW" and
  .scientific_selection == "OWNER_DECISION_REQUIRED" and
  .automatic_promotion == false and
  .training_tokens.consumed_samples == 6152 and
  .training_tokens.dropped_samples == 3 and
  .numerical_gate.attempted_updates == 769 and
  .numerical_gate.accepted_updates == 769 and
  .numerical_gate.overflow_windows == 0 and
  .capability.paired_vs_gn_b4.paired_deltas.NDS.clusters == 8 and
  .capability.paired_vs_bn_b4.paired_deltas.mAP.clusters == 8
' "${SUMMARY}" >/dev/null
jq -e --arg source "${S10_C1B8_EXPECTED_SOURCE_SHA}" --arg resolved "${S10_C1B8_EXPECTED_RESOLVED_SHA256}" '
  .schema == "fl_v3.s10.c1b1_bn_b8.v1" and .source_sha == $source and
  .resolved_config_sha256 == $resolved and
  .cell.physical_microbatch == 8 and .cell.eval_physical_microbatch == 4 and
  .terminal_training_state.attempted_windows == 769 and
  .terminal_training_state.optimizer_step == 769 and
  .terminal_training_state.invalid_windows == 0 and
  .terminal_training_state.nonfinite_windows == 0 and
  .terminal_training_state.overflow_windows == 0 and
  .terminal_training_state.discarded_windows == 0 and
  .training_token_evidence.consumed_sample_count == 6152 and
  .training_token_evidence.drop_last_remainder_count == 3 and
  .D_select_evaluation.n_samples == 4626 and
  (.health.hard_errors | length) == 0
' "${CELL_SUMMARY}" >/dev/null
test "$(jq -r '.initial_parameter_sha256' "${CELL_SUMMARY}")" = "87be0d2416b3ed06e2d1e9214e11ad3ac25bc275993b0865d918af6f332829d1"
test "$(jq -r '.training_token_evidence.consumed_sample_tokens_ordered_sha256' "${CELL_SUMMARY}")" = "947dc9bc8441267587df6b0b88d16efc84ab3c7ff0a1a152481ac2697f0a2eb1"
test "$(jq -r '.training_token_evidence.drop_last_remainder_tokens_sorted_sha256' "${CELL_SUMMARY}")" = "7495cdbec472ce49f29e8f19abe08fc9431a258b437a5db05ab89fae0db60443"
test -s "${WORK}/evidence/C1-B1-BN1D-B8-DLOW/checkpoint.pt"
runner_complete=1
