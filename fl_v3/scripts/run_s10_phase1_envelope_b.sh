#!/bin/bash
# Exact one-candidate S10 Phase-I Envelope-B launcher.
set -euo pipefail

usage() {
  echo "usage: $0 --branch {camera|lidar} --config PATH --output-dir PATH --source-sha SHA [--resume]" >&2
  exit 2
}

branch=""
config=""
output_dir=""
source_sha=""
resume=0
while (( $# )); do
  case "$1" in
    --branch) [[ $# -ge 2 ]] || usage; branch="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || usage; output_dir="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) usage ;;
  esac
done

[[ "${branch}" == "camera" || "${branch}" == "lidar" ]] || usage
[[ -n "${config}" && -n "${output_dir}" ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${script_dir}/../.." && pwd)"
entry="fl_v3/scripts/s10_phase1_capability.py"
if [[ "${config}" != /* ]]; then
  config="${source_root}/${config}"
fi
config="$(realpath "${config}")"
output_dir="$(realpath -m "${output_dir}")"

fail() { echo "[s10-phase1-b] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" ]] || fail "config missing: ${config}"
[[ -f "${source_root}/${entry}" ]] || fail "entry missing"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1-branch-qualification"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"
if (( resume )); then
  [[ -d "${output_dir}" ]] || fail "resume output is missing: ${output_dir}"
else
  [[ ! -e "${output_dir}" ]] || fail "fresh output already exists: ${output_dir}"
fi

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

arguments=(
  --branch "${branch}"
  --config "${config}"
  --output-dir "${output_dir}"
  --source-sha "${source_sha}"
)
if (( resume )); then arguments+=(--resume); fi
cd "${source_root}"
exec python "${entry}" "${arguments[@]}"
