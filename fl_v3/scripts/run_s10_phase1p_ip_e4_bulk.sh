#!/bin/bash
# Exact single-GH200 Camera-B16 vectorized-geometry versus bulk-conversion screen.
set -euo pipefail

usage() {
  echo "usage: $0 --config PATH --reference-profile PATH --candidate-profile PATH --source-sha SHA --approved-source-sha SHA" >&2
  exit 2
}

config=""; reference_profile=""; candidate_profile=""
source_sha=""; approved_source_sha=""
while (( $# )); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --reference-profile) [[ $# -ge 2 ]] || usage; reference_profile="$2"; shift 2 ;;
    --candidate-profile) [[ $# -ge 2 ]] || usage; candidate_profile="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --approved-source-sha) [[ $# -ge 2 ]] || usage; approved_source_sha="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "${config}" && -n "${reference_profile}" && -n "${candidate_profile}" ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${approved_source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1_throughput.py" ]]; then
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi
expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
expected_approval_sha="7d4bb6efdbb7b8fb61ee72243c72a5ec3ef7d451"
expected_unlock_pair_sha="6cc4f70f7691633b7e34ed1bc358ac40dc938dfedc1b61d32dce1ce1416eb8ef"

resolve_from_root() {
  local value="$1"
  if [[ "${value}" != /* ]]; then value="${source_root}/${value}"; fi
  realpath "${value}"
}
config="$(resolve_from_root "${config}")"
reference_profile="$(resolve_from_root "${reference_profile}")"
candidate_profile="$(resolve_from_root "${candidate_profile}")"

fail() { echo "[s10-phase1p-ip-e4-bulk] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" && -f "${reference_profile}" && -f "${candidate_profile}" ]] \
  || fail "config/profile missing"
equal "approved source SHA" "${approved_source_sha}" "${expected_approval_sha}"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
git -C "${source_root}" merge-base --is-ancestor \
  "${approved_source_sha}" "${source_sha}" \
  || fail "source SHA is not an approved linear descendant"
[[ -z "$(git -C "${source_root}" rev-list --min-parents=2 \
  "${expected_base_sha}..${source_sha}")" ]] \
  || fail "source history is not linear"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"

expected_output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e4_${approved_source_sha:0:12}"
unlock_pair="${expected_output_root}/pairs/b16_reference_vs_vectorized_geometry.json"
[[ -f "${unlock_pair}" ]] || fail "vectorized-geometry unlock pair is absent"
equal "vectorized-geometry unlock pair SHA256" \
  "$(sha256sum "${unlock_pair}" | awk '{print $1}')" \
  "${expected_unlock_pair_sha}"
python3 - "${unlock_pair}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
gate = result["ip_e4_vectorized_geometry_gate"]
if not (
    result["promotion_authorized"] is True
    and gate["hard_gate_pass"] is True
    and gate["conditional_bulk_input_conversion_implementation_eligible"] is True
):
    raise SystemExit("IP-E4 bulk conversion is not unlocked")
PY

reference_attempt="b16_vectorized_geometry_reference"
candidate_attempt="b16_vectorized_geometry_bulk_input_conversion"
reference_output="${expected_output_root}/camera/sustained_${source_sha:0:12}_r2_${reference_attempt}"
candidate_output="${expected_output_root}/camera/sustained_${source_sha:0:12}_r2_${candidate_attempt}"
pair_output="${expected_output_root}/pairs/b16_vectorized_geometry_vs_bulk_input_conversion.json"
for fresh_path in "${reference_output}" "${candidate_output}" "${pair_output}"; do
  [[ ! -e "${fresh_path}" ]] || fail "fresh output already exists: ${fresh_path}"
done

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
  fl_v3/tests/test_s10_phase1p_checkpoint_gate.py \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e4_profiles_bind_the_final_b16_stack \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e4_bulk_candidate_configuration_is_fail_closed \
  fl_v3/tests/test_s03_camera_contract.py::test_s10_phase1p_bulk_input_conversion_preserves_output_and_uses_one_native_batch \
  fl_v3/tests/test_s10_phase1p_compare.py::test_ip_e4_bulk_conversion_gate_promotes_at_102_percent

run_one() {
  local profile="$1" output="$2" attempt="$3"
  python fl_v3/scripts/s10_phase1_throughput.py \
    --branch camera --mode sustained --config "${config}" \
    --profile-config "${profile}" --output-dir "${output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat 2 --attempt-id "${attempt}"
}

run_one "${reference_profile}" "${reference_output}" "${reference_attempt}"
run_one "${candidate_profile}" "${candidate_output}" "${candidate_attempt}"
python fl_v3/scripts/s10_phase1p_ip_e4_bulk_compare.py \
  --reference-dir "${reference_output}" \
  --candidate-dir "${candidate_output}" \
  --output "${pair_output}"
