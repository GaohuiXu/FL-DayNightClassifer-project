#!/bin/bash
# Exact one-candidate S10 Phase-I Envelope-B launcher.
# Camera uses the owner-promoted same-node 2xGH200 DDP recipe; LiDAR stays single-GPU.
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
single_entry="fl_v3/scripts/s10_phase1_capability.py"
camera_entry="fl_v3/scripts/s10_phase1_camera_ddp.py"
if [[ "${config}" != /* ]]; then
  config="${source_root}/${config}"
fi
config="$(realpath "${config}")"
output_dir="$(realpath -m "${output_dir}")"

fail() { echo "[s10-phase1-b] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" ]] || fail "config missing: ${config}"
[[ -f "${source_root}/${single_entry}" ]] || fail "single-GPU entry missing"
[[ -f "${source_root}/${camera_entry}" ]] || fail "Camera DDP entry missing"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
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
equal "Slurm restart count" "${SLURM_RESTART_COUNT:-0}" "0"
if [[ "${branch}" == "camera" ]]; then
  equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "32"
  equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "196608"
  equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "2"
  all_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  [[ -n "${all_visible_devices}" ]] || fail "CUDA_VISIBLE_DEVICES is empty"
  IFS=',' read -r -a visible_devices <<< "${all_visible_devices}"
  [[ "${#visible_devices[@]}" -eq 2 ]] \
    || fail "Camera requires exactly two visible allocated GPUs"
else
  equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "16"
  equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "98304"
  equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "1"
fi

arguments=(
  --config "${config}"
  --output-dir "${output_dir}"
  --source-sha "${source_sha}"
)
if (( resume )); then arguments+=(--resume); fi
cd "${source_root}"
if [[ "${branch}" == "camera" ]]; then
  export NCCL_DEBUG=WARN
  export NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  exec python -m torch.distributed.run --standalone --nproc-per-node=2 \
    --max-restarts=0 "${camera_entry}" "${arguments[@]}"
fi
export WORLD_SIZE=1
exec python "${single_entry}" --branch lidar "${arguments[@]}"
