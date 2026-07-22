#!/bin/bash
# One exact same-allocation B32 LiDAR paired cell for S10 Phase I-P IP-L-E2.
set -euo pipefail

usage() {
  echo "usage: $0 --cell {l2-1|l2-2|l2-3|l2-4|l2-5} --source-sha SHA --approved-source-sha SHA" >&2
  exit 2
}

cell=""; source_sha=""; approved_source_sha=""
while (( $# )); do
  case "$1" in
    --cell) [[ $# -ge 2 ]] || usage; cell="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --approved-source-sha) [[ $# -ge 2 ]] || usage; approved_source_sha="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ "${cell}" =~ ^l2-[1-5]$ ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${approved_source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1_throughput.py" ]]; then
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi

fail() { echo "[s10-phase1p-lidar-e2] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
config="${source_root}/fl_v3/configs/s10_phase1_lidar.json"
entry="${source_root}/fl_v3/scripts/s10_phase1_throughput.py"
compare_entry="${source_root}/fl_v3/scripts/s10_phase1p_compare.py"
reference_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e2_reference_b32.json"
case "${cell}" in
  l2-1)
    first_profile="${reference_profile}"; first_attempt="l2_1_ref"
    second_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e2_hungarian_b32.json"
    second_attempt="l2_1_hungarian"; pair_reference="first" ;;
  l2-2)
    first_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e2_sdpa_b32.json"
    first_attempt="l2_2_sdpa"; second_profile="${reference_profile}"
    second_attempt="l2_2_ref"; pair_reference="second" ;;
  l2-3)
    first_profile="${reference_profile}"; first_attempt="l2_3_ref"
    second_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e2_compile_b32.json"
    second_attempt="l2_3_compile"; pair_reference="first" ;;
  l2-4)
    first_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e2_offsets_b32.json"
    first_attempt="l2_4_offsets"; second_profile="${reference_profile}"
    second_attempt="l2_4_ref"; pair_reference="second" ;;
  l2-5)
    first_profile="${reference_profile}"; first_attempt="l2_5_ref"
    second_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e2_fused_b32.json"
    second_attempt="l2_5_fused"; pair_reference="first" ;;
esac

for path in "${config}" "${entry}" "${compare_entry}" \
  "${first_profile}" "${second_profile}"; do
  [[ -f "${path}" ]] || fail "required path is absent: ${path}"
done
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
git -C "${source_root}" merge-base --is-ancestor \
  "${expected_base_sha}" "${approved_source_sha}" \
  || fail "approved source is not descended from the unique base"
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
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e2_profiles_are_isolated_b32_mappings \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e2_candidate_configuration_is_fail_closed \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e2_sdpa_forward_backward_parity \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e2_batched_hungarian_target_and_gradient_parity \
  fl_v3/tests/test_sparse_voxel_encoder.py::test_phase1p_host_offsets_preserve_sparse_forward_backward \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e2_fused_adamw_matches_unfused_accepted_updates \
  fl_v3/tests/test_s10_phase1p_compare.py::test_lidar_e2_pair_classifies_positive_and_negative_without_promotion

output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e2_${approved_source_sha:0:12}"
first_output="${output_root}/lidar/sustained_${source_sha:0:12}_r1_${first_attempt}"
second_output="${output_root}/lidar/sustained_${source_sha:0:12}_r1_${second_attempt}"
pair_output="${output_root}/pairs/${cell//-/_}_${source_sha:0:12}_r1.json"
[[ ! -e "${first_output}" && ! -e "${second_output}" && ! -e "${pair_output}" ]] \
  || fail "one or more frozen output paths already exist"

run_one() {
  local profile="$1" output="$2" attempt="$3"
  python "${entry}" \
    --branch lidar --mode sustained --config "${config}" \
    --profile-config "${profile}" --output-dir "${output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat 1 --attempt-id "${attempt}"
}

run_one "${first_profile}" "${first_output}" "${first_attempt}"
run_one "${second_profile}" "${second_output}" "${second_attempt}"
if [[ "${pair_reference}" == "first" ]]; then
  reference_output="${first_output}"; candidate_output="${second_output}"
else
  reference_output="${second_output}"; candidate_output="${first_output}"
fi
python "${compare_entry}" --lidar-e2 \
  --reference-dir "${reference_output}" --candidate-dir "${candidate_output}" \
  --output "${pair_output}"
python - "${pair_output}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
if summary.get("lidar_hard_gate", {}).get("gate_pass") is not True:
    raise SystemExit("IP-L-E2 paired hard gate failed")
print(
    "IP_L_E2_PAIR="
    + json.dumps(
        {
            "classification": summary["throughput"]["performance_classification"],
            "lower_bound": summary["throughput"]["one_sided_95_percent_lower_bound"],
            "ratio": summary["throughput"]["candidate_over_reference"],
        },
        sort_keys=True,
    )
)
PY

echo "[s10-phase1p-lidar-e2] COMPLETE cell=${cell} pair=${pair_output}"
