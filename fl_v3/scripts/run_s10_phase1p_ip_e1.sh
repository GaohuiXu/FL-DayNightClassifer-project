#!/bin/bash
# Exact D_fit-only launcher for the S10 Phase I-P IP-E1 profiler.
set -euo pipefail

usage() {
  echo "usage: $0 --branch {camera|lidar} --mode {sustained|trace} --config PATH --profile-config PATH --output-dir PATH --source-sha SHA --approved-source-sha SHA --repeat {1|2|3} --attempt-id ID" >&2
  exit 2
}

branch=""
mode=""
config=""
profile_config=""
output_dir=""
source_sha=""
approved_source_sha=""
repeat=""
attempt_id=""
while (( $# )); do
  case "$1" in
    --branch) [[ $# -ge 2 ]] || usage; branch="$2"; shift 2 ;;
    --mode) [[ $# -ge 2 ]] || usage; mode="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --profile-config) [[ $# -ge 2 ]] || usage; profile_config="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || usage; output_dir="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --approved-source-sha) [[ $# -ge 2 ]] || usage; approved_source_sha="$2"; shift 2 ;;
    --repeat) [[ $# -ge 2 ]] || usage; repeat="$2"; shift 2 ;;
    --attempt-id) [[ $# -ge 2 ]] || usage; attempt_id="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "${branch}" == "camera" || "${branch}" == "lidar" ]] || usage
[[ "${mode}" == "sustained" || "${mode}" == "trace" ]] || usage
[[ "${repeat}" == "1" || "${repeat}" == "2" || "${repeat}" == "3" ]] || usage
[[ "${mode}" != "trace" || "${repeat}" == "1" ]] || usage
[[ "${attempt_id}" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${approved_source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ -n "${config}" && -n "${profile_config}" && -n "${output_dir}" ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${script_dir}/../.." && pwd)"
entry="fl_v3/scripts/s10_phase1_throughput.py"
expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
if [[ "${config}" != /* ]]; then config="${source_root}/${config}"; fi
if [[ "${profile_config}" != /* ]]; then profile_config="${source_root}/${profile_config}"; fi
config="$(realpath "${config}")"
profile_config="$(realpath "${profile_config}")"
output_dir="$(realpath -m "${output_dir}")"

fail() { echo "[s10-phase1p-ip-e1] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" ]] || fail "config missing: ${config}"
[[ -f "${profile_config}" ]] || fail "profile config missing: ${profile_config}"
[[ -f "${source_root}/${entry}" ]] || fail "entry missing"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse \
    refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
git -C "${source_root}" merge-base --is-ancestor \
  "${expected_base_sha}" "${approved_source_sha}" \
  || fail "approved source is not descended from the unique IP-G0 base"
git -C "${source_root}" merge-base --is-ancestor \
  "${approved_source_sha}" "${source_sha}" \
  || fail "source SHA is not an approved linear descendant"
[[ -z "$(git -C "${source_root}" rev-list --min-parents=2 \
  "${expected_base_sha}..${source_sha}")" ]] \
  || fail "source history is not linear from the unique IP-G0 base"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"
[[ ! -e "${output_dir}" ]] || fail "fresh profiler output already exists: ${output_dir}"

# shellcheck disable=SC1091
source "${source_root}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${source_root}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
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
exec python "${entry}" \
  --branch "${branch}" \
  --mode "${mode}" \
  --config "${config}" \
  --profile-config "${profile_config}" \
  --output-dir "${output_dir}" \
  --source-sha "${source_sha}" \
  --approved-source-sha "${approved_source_sha}" \
  --repeat "${repeat}" \
  --attempt-id "${attempt_id}"
