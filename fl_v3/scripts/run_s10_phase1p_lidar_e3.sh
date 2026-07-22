#!/bin/bash
# Exact same-allocation B32 LiDAR ABBA composition gate for IP-L-E3.
set -euo pipefail

usage() {
  echo "usage: $0 --source-sha SHA --approved-source-sha SHA" >&2
  exit 2
}

source_sha=""; approved_source_sha=""
while (( $# )); do
  case "$1" in
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --approved-source-sha) [[ $# -ge 2 ]] || usage; approved_source_sha="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${approved_source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1_throughput.py" ]]; then
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi

fail() { echo "[s10-phase1p-lidar-e3] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
expected_approval_anchor="468a82bddda685fe81ece1fe0e59db35c50ba856"
config="${source_root}/fl_v3/configs/s10_phase1_lidar.json"
entry="${source_root}/fl_v3/scripts/s10_phase1_throughput.py"
compare_entry="${source_root}/fl_v3/scripts/s10_phase1p_compare.py"
reference_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e3_reference_b32.json"
combined_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e3_combined_b32.json"

for path in "${config}" "${entry}" "${compare_entry}" \
  "${reference_profile}" "${combined_profile}"; do
  [[ -f "${path}" ]] || fail "required path is absent: ${path}"
done
equal "approval anchor" "${approved_source_sha}" "${expected_approval_anchor}"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
git -C "${source_root}" merge-base --is-ancestor \
  "${expected_base_sha}" "${approved_source_sha}" \
  || fail "approval anchor is not descended from the unique base"
git -C "${source_root}" merge-base --is-ancestor \
  "${approved_source_sha}" "${source_sha}" \
  || fail "source is not an approved linear descendant"
[[ -z "$(git -C "${source_root}" rev-list --min-parents=2 \
  "${expected_base_sha}..${source_sha}")" ]] || fail "source history is not linear"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"

# shellcheck disable=SC1091
source "${source_root}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${source_root}/fl_v3/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WORLD_SIZE=1
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
NUSCENES_ZIP_MANIFEST="$(python - "${config}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["data"]["zip_manifest"]["path"])
PY
)"
export NUSCENES_ZIP_MANIFEST

equal "nuScenes dataroot" "${NUSCENES_DATAROOT}" \
  "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
equal "Slurm partition" "${SLURM_JOB_PARTITION:-}" "gpu"
equal "Slurm node count" "${SLURM_NNODES:-}" "1"
equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "16"
equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "98304"
equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "1"
equal "Slurm restart count" "${SLURM_RESTART_COUNT:-0}" "0"

cd "${source_root}"
python -m pytest -q \
  fl_v3/tests/test_s10_phase1p_checkpoint_gate.py \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e3_profiles_freeze_exact_lg2_combination \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e3_combined_candidate_configuration_is_fail_closed \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e2_batched_hungarian_target_and_gradient_parity \
  fl_v3/tests/test_sparse_voxel_encoder.py::test_phase1p_host_offsets_preserve_sparse_forward_backward \
  fl_v3/tests/test_s10_phase1p_compare.py::test_lidar_e3_abba_promotes_only_the_exact_positive_combination

output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e3_${approved_source_sha:0:12}"
reference_a="${output_root}/lidar/sustained_${source_sha:0:12}_r1_l3_abba_ref_a"
candidate_a="${output_root}/lidar/sustained_${source_sha:0:12}_r1_l3_abba_combined_a"
candidate_b="${output_root}/lidar/sustained_${source_sha:0:12}_r2_l3_abba_combined_b"
reference_b="${output_root}/lidar/sustained_${source_sha:0:12}_r2_l3_abba_ref_b"
abba_output="${output_root}/pairs/l3_abba_${source_sha:0:12}.json"
trace_output="${output_root}/lidar/trace_${source_sha:0:12}_r1_l3_combined_trace"
for path in "${reference_a}" "${candidate_a}" "${candidate_b}" \
  "${reference_b}" "${abba_output}" "${trace_output}"; do
  [[ ! -e "${path}" ]] || fail "frozen output path already exists: ${path}"
done

run_one() {
  local profile="$1" output="$2" repeat="$3" attempt="$4"
  python "${entry}" \
    --branch lidar --mode sustained --config "${config}" \
    --profile-config "${profile}" --output-dir "${output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat "${repeat}" --attempt-id "${attempt}"
}

run_one "${reference_profile}" "${reference_a}" 1 l3_abba_ref_a
run_one "${combined_profile}" "${candidate_a}" 1 l3_abba_combined_a
run_one "${combined_profile}" "${candidate_b}" 2 l3_abba_combined_b
run_one "${reference_profile}" "${reference_b}" 2 l3_abba_ref_b

python "${compare_entry}" --lidar-e3-abba \
  --reference-dir "${reference_a}" --candidate-dir "${candidate_a}" \
  --candidate-dir-2 "${candidate_b}" --reference-dir-2 "${reference_b}" \
  --output "${abba_output}"

trace_eligible="$(python - "${abba_output}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
if summary.get("hard_gate", {}).get("gate_pass") is not True:
    raise SystemExit("IP-L-E3 ABBA hard gate failed")
print("yes" if summary.get("combined_recipe_gate_pass") is True else "no")
PY
)"

if [[ "${trace_eligible}" == "yes" ]]; then
  python "${entry}" \
    --branch lidar --mode trace --config "${config}" \
    --profile-config "${combined_profile}" --output-dir "${trace_output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat 1 --attempt-id l3_combined_trace
  python - "${trace_output}/result.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
diagnosis = (
    result.get("torch_trace", {})
    .get("structured_summary", {})
    .get("lidar_stage_diagnosis", {})
)
if result.get("status") != "COMPLETE_TRACE":
    raise SystemExit("IP-L-E3 combined trace did not complete")
if result.get("measurement_health", {}).get("gate_pass") is not True:
    raise SystemExit("IP-L-E3 combined trace health gate failed")
if diagnosis.get("missing_core_range_keys") != []:
    raise SystemExit("IP-L-E3 combined trace omitted a required stage")
print("IP_L_E3_TRACE=COMPLETE")
PY
else
  echo "IP_L_E3_TRACE=SKIPPED_COMBINATION_NOT_PROMOTED"
fi

echo "[s10-phase1p-lidar-e3] COMPLETE abba=${abba_output} trace=${trace_eligible}"
