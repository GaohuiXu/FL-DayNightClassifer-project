#!/bin/bash
# Exact same-allocation LiDAR capacity/sustained/trace launcher for IP-L-E1.
set -euo pipefail

usage() {
  echo "usage: $0 --source-sha SHA --approved-source-sha SHA" >&2
  exit 2
}

source_sha=""
approved_source_sha=""
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

fail() { echo "[s10-phase1p-lidar-e1] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
config="${source_root}/fl_v3/configs/s10_phase1_lidar.json"
entry="${source_root}/fl_v3/scripts/s10_phase1_throughput.py"
declare -A profiles=(
  [4]="${source_root}/fl_v3/configs/s10_phase1p_lidar_e1_b4.json"
  [8]="${source_root}/fl_v3/configs/s10_phase1p_lidar_e1_b8.json"
  [16]="${source_root}/fl_v3/configs/s10_phase1p_lidar_e1_b16.json"
  [32]="${source_root}/fl_v3/configs/s10_phase1p_lidar_e1_b32.json"
)

[[ -f "${config}" && -f "${entry}" ]] || fail "config or profiler entry missing"
for batch in 4 8 16 32; do
  [[ -f "${profiles[${batch}]}" ]] || fail "profile missing for B${batch}"
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
  || fail "source is not the approved source or a linear remediation descendant"
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
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e1_profiles_are_clean_capacity_and_trace_mappings \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_trace_diagnosis_requires_nested_bottleneck_ranges \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_loss_health_reports_head_tail_without_making_a_slope_gate \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_loss_health_flushes_scalar_views_in_bounded_blocks

output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e1_${approved_source_sha:0:12}"

output_for() {
  local mode="$1" repeat="$2" attempt="$3"
  echo "${output_root}/lidar/${mode}_${source_sha:0:12}_r${repeat}_${attempt}"
}

run_one() {
  local mode="$1" batch="$2" repeat="$3" attempt="$4"
  local output
  output="$(output_for "${mode}" "${repeat}" "${attempt}")"
  [[ ! -e "${output}" ]] || fail "fresh output already exists: ${output}"
  python "${entry}" \
    --branch lidar --mode "${mode}" --config "${config}" \
    --profile-config "${profiles[${batch}]}" \
    --output-dir "${output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat "${repeat}" --attempt-id "${attempt}"
}

capacity_passed() {
  local output="$1"
  python - "${output}/result.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
passed = (
    result.get("status") == "COMPLETE_CAPACITY"
    and result.get("memory_safe_under_85_percent_reserved") is True
    and result.get("measurement_health", {}).get("gate_pass") is True
)
raise SystemExit(0 if passed else 1)
PY
}

repeat_is_unstable() {
  local first="$1" second="$2"
  python - "${first}/result.json" "${second}/result.json" <<'PY'
import json, sys
rates = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        result = json.load(stream)
    rates.append(float(result["measurement"]["readiness_timing"]["throughput"]["exposure_samples_per_second"]))
spread = abs(rates[0] - rates[1]) / (sum(rates) / 2.0)
print(f"repeat_spread_fraction={spread:.9f}")
raise SystemExit(0 if spread > 0.03 else 1)
PY
}

highest=0
for batch in 4 8 16 32; do
  attempt="cap_b${batch}"
  run_one capacity "${batch}" 1 "${attempt}"
  capacity_output="$(output_for capacity 1 "${attempt}")"
  if capacity_passed "${capacity_output}"; then
    highest="${batch}"
  else
    [[ "${batch}" != "4" ]] || fail "B4 reference failed its capacity/health gate"
    break
  fi
done
[[ "${highest}" != "0" ]] || fail "no LiDAR batch passed capacity"

run_one sustained 4 1 sustain_b4
if [[ "${highest}" != "4" ]]; then
  run_one sustained "${highest}" 1 "sustain_b${highest}"
  run_one sustained "${highest}" 2 "sustain_b${highest}_reverse"
fi
run_one sustained 4 2 sustain_b4_reverse
if repeat_is_unstable \
  "$(output_for sustained 1 sustain_b4)" \
  "$(output_for sustained 2 sustain_b4_reverse)"; then
  run_one sustained 4 3 sustain_b4_conditional
fi
if [[ "${highest}" != "4" ]] && repeat_is_unstable \
  "$(output_for sustained 1 "sustain_b${highest}")" \
  "$(output_for sustained 2 "sustain_b${highest}_reverse")"; then
  run_one sustained "${highest}" 3 "sustain_b${highest}_conditional"
fi
run_one trace 4 1 trace_b4
if [[ "${highest}" != "4" ]]; then
  run_one trace "${highest}" 1 "trace_b${highest}"
fi

echo "[s10-phase1p-lidar-e1] COMPLETE highest_safe_batch=B${highest} output_root=${output_root}"
