#!/bin/bash
# Trace-only repair for the already-positive IP-L-E3 ABBA evidence.
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

fail() { echo "[s10-phase1p-lidar-e3-trace] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
expected_approval_anchor="468a82bddda685fe81ece1fe0e59db35c50ba856"
abba_source_sha="6105d70d81c1e0a496dec288e5b76217155783c5"
abba_sha256="7f4844d65a031ce65e41c1f0b8ada946d582c40dc0b43467f77fc6fc2d9f9279"
config="${source_root}/fl_v3/configs/s10_phase1_lidar.json"
entry="${source_root}/fl_v3/scripts/s10_phase1_throughput.py"
combined_profile="${source_root}/fl_v3/configs/s10_phase1p_lidar_e3_combined_b32.json"

equal "approval anchor" "${approved_source_sha}" "${expected_approval_anchor}"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
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

equal "Slurm partition" "${SLURM_JOB_PARTITION:-}" "gpu"
equal "Slurm node count" "${SLURM_NNODES:-}" "1"
equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "16"
equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "98304"
equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "1"
equal "Slurm restart count" "${SLURM_RESTART_COUNT:-0}" "0"

cd "${source_root}"
python -m pytest -q \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e3_profiles_freeze_exact_lg2_combination \
  fl_v3/tests/test_s10_phase1p_profile.py::test_lidar_e3_trace_accepts_compiled_parents_and_batched_loss_ranges

output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e3_${approved_source_sha:0:12}"
abba_output="${output_root}/pairs/l3_abba_${abba_source_sha:0:12}.json"
trace_output="${output_root}/lidar/trace_${source_sha:0:12}_r1_l3_combined_trace_repair"
[[ -f "${abba_output}" ]] || fail "sealed positive ABBA evidence is absent"
equal "sealed ABBA SHA-256" "$(sha256sum "${abba_output}" | awk '{print $1}')" \
  "${abba_sha256}"
[[ ! -e "${trace_output}" ]] || fail "fresh trace-repair output already exists"

python - "${abba_output}" "${abba_source_sha}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    summary = json.load(stream)
if summary.get("combined_recipe_gate_pass") is not True:
    raise SystemExit("sealed ABBA evidence is not positive")
if summary.get("hard_gate", {}).get("gate_pass") is not True:
    raise SystemExit("sealed ABBA hard gate is not positive")
roots = summary.get("reference", {}).get("roots", []) + summary.get("candidate", {}).get("roots", [])
if not roots or any(sys.argv[2][:12] not in root for root in roots):
    raise SystemExit("sealed ABBA source identity drift")
print("IP_L_E3_ABBA_REUSED=POSITIVE")
PY

python "${entry}" \
  --branch lidar --mode trace --config "${config}" \
  --profile-config "${combined_profile}" --output-dir "${trace_output}" \
  --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
  --repeat 1 --attempt-id l3_combined_trace_repair

python - "${trace_output}/result.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
diagnosis = result["torch_trace"]["structured_summary"]["lidar_stage_diagnosis"]
checks = {
    "complete_trace": result.get("status") == "COMPLETE_TRACE",
    "measurement_health": result.get("measurement_health", {}).get("gate_pass") is True,
    "no_missing_ranges": diagnosis.get("missing_core_range_keys") == [],
    "compiled_parent_policy": diagnosis.get("compiled_dense_internal_ranges_suppressed") is True,
    "batched_loss_policy": diagnosis.get("batched_hungarian_ranges") is True,
}
if not all(checks.values()):
    raise SystemExit(f"trace-repair gate failed: {checks}")
print("IP_L_E3_TRACE_REPAIR=COMPLETE")
PY

echo "[s10-phase1p-lidar-e3-trace] COMPLETE trace=${trace_output}"
