#!/bin/bash
# Exact Camera-only same-allocation launcher for S10 Phase I-P IP-E2.
set -euo pipefail

usage() {
  echo "usage: $0 --config PATH --first-profile PATH --first-mode {sustained|capacity} --first-output PATH --first-attempt ID [--second-profile PATH --second-mode sustained --second-output PATH --second-attempt ID --pair-output PATH --pair-reference {first|second}] --source-sha SHA --approved-source-sha SHA --repeat {1|2|3}" >&2
  exit 2
}

config=""
first_profile=""; first_mode=""; first_output=""; first_attempt=""
second_profile=""; second_mode=""; second_output=""; second_attempt=""
pair_output=""; pair_reference="first"
source_sha=""; approved_source_sha=""; repeat=""
while (( $# )); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --first-profile) [[ $# -ge 2 ]] || usage; first_profile="$2"; shift 2 ;;
    --first-mode) [[ $# -ge 2 ]] || usage; first_mode="$2"; shift 2 ;;
    --first-output) [[ $# -ge 2 ]] || usage; first_output="$2"; shift 2 ;;
    --first-attempt) [[ $# -ge 2 ]] || usage; first_attempt="$2"; shift 2 ;;
    --second-profile) [[ $# -ge 2 ]] || usage; second_profile="$2"; shift 2 ;;
    --second-mode) [[ $# -ge 2 ]] || usage; second_mode="$2"; shift 2 ;;
    --second-output) [[ $# -ge 2 ]] || usage; second_output="$2"; shift 2 ;;
    --second-attempt) [[ $# -ge 2 ]] || usage; second_attempt="$2"; shift 2 ;;
    --pair-output) [[ $# -ge 2 ]] || usage; pair_output="$2"; shift 2 ;;
    --pair-reference) [[ $# -ge 2 ]] || usage; pair_reference="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --approved-source-sha) [[ $# -ge 2 ]] || usage; approved_source_sha="$2"; shift 2 ;;
    --repeat) [[ $# -ge 2 ]] || usage; repeat="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "${config}" && -n "${first_profile}" && -n "${first_output}" ]] || usage
[[ "${first_mode}" == "sustained" || "${first_mode}" == "capacity" ]] || usage
[[ "${repeat}" == "1" || "${repeat}" == "2" || "${repeat}" == "3" ]] || usage
[[ "${first_attempt}" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${approved_source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
if [[ -n "${second_profile}" || -n "${second_mode}" || -n "${second_output}" || -n "${second_attempt}" ]]; then
  [[ -n "${second_profile}" && "${second_mode}" == "sustained" && -n "${second_output}" ]] || usage
  [[ "${second_attempt}" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]] || usage
  [[ "${first_mode}" == "sustained" ]] || usage
  [[ -n "${pair_output}" && ( "${pair_reference}" == "first" || "${pair_reference}" == "second" ) ]] || usage
else
  [[ -z "${pair_output}" && "${pair_reference}" == "first" ]] || usage
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1_throughput.py" ]]; then
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi
entry="fl_v3/scripts/s10_phase1_throughput.py"
expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"

resolve_from_root() {
  local value="$1"
  if [[ "${value}" != /* ]]; then value="${source_root}/${value}"; fi
  realpath "${value}"
}
config="$(resolve_from_root "${config}")"
first_profile="$(resolve_from_root "${first_profile}")"
first_output="$(realpath -m "${first_output}")"
if [[ -n "${second_profile}" ]]; then
  second_profile="$(resolve_from_root "${second_profile}")"
  second_output="$(realpath -m "${second_output}")"
  pair_output="$(realpath -m "${pair_output}")"
fi

fail() { echo "[s10-phase1p-ip-e2] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" && -f "${first_profile}" ]] || fail "config/profile missing"
[[ -z "${second_profile}" || -f "${second_profile}" ]] || fail "second profile missing"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
git -C "${source_root}" merge-base --is-ancestor "${approved_source_sha}" "${source_sha}" \
  || fail "source SHA is not an approved linear descendant"
[[ -z "$(git -C "${source_root}" rev-list --min-parents=2 "${expected_base_sha}..${source_sha}")" ]] \
  || fail "source history is not linear"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"
[[ ! -e "${first_output}" ]] || fail "first output already exists: ${first_output}"
[[ -z "${second_output}" || ! -e "${second_output}" ]] \
  || fail "second output already exists: ${second_output}"
expected_output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e2_${approved_source_sha:0:12}"
[[ "${first_output}" == "${expected_output_root}/camera/"* ]] \
  || fail "first output is outside the frozen IP-E2 root"
if [[ -n "${second_output}" ]]; then
  [[ "${second_output}" == "${expected_output_root}/camera/"* ]] \
    || fail "second output is outside the frozen IP-E2 root"
  [[ "${pair_output}" == "${expected_output_root}/pairs/"*.json ]] \
    || fail "paired summary is outside the frozen IP-E2 pairs root"
  [[ ! -e "${pair_output}" ]] || fail "paired summary already exists: ${pair_output}"
fi

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
equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "16"
equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "98304"
equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "1"

cd "${source_root}"
python -m pytest -q \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e2_profiles_are_exact_camera_only_mappings \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e2_runtime_views_preserve_effective_b32_and_source_bytes \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e2_candidate_configuration_preserves_state_dict_names \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e2_current_swin_sdpa_forward_backward_parity \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e2_fused_adamw_matches_unfused_accepted_updates \
  fl_v3/tests/test_s10_phase1p_compare.py::test_pair_comparison_enforces_match_and_emits_b16_gate \
  fl_v3/tests/test_s10_phase1p_compare.py::test_same_batch_pair_requires_exact_input_anchor

run_one() {
  local mode="$1" profile="$2" output="$3" attempt="$4"
  python "${entry}" \
    --branch camera --mode "${mode}" --config "${config}" \
    --profile-config "${profile}" --output-dir "${output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat "${repeat}" --attempt-id "${attempt}"
}

run_one "${first_mode}" "${first_profile}" "${first_output}" "${first_attempt}"
if [[ -n "${second_profile}" ]]; then
  run_one "${second_mode}" "${second_profile}" "${second_output}" "${second_attempt}"
  if [[ "${pair_reference}" == "first" ]]; then
    reference_output="${first_output}"; candidate_output="${second_output}"
  else
    reference_output="${second_output}"; candidate_output="${first_output}"
  fi
  python fl_v3/scripts/s10_phase1p_compare.py \
    --reference-dir "${reference_output}" --candidate-dir "${candidate_output}" \
    --output "${pair_output}"
fi
