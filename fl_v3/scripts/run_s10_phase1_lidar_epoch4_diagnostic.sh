#!/bin/bash
# One-GH200, zero-update LiDAR epoch-4 numerical/D_select diagnostic launcher.
set -euo pipefail

usage() {
  echo "usage: $0 --envelope PATH --config PATH --checkpoint PATH --epoch-record PATH --output-dir PATH --source-sha SHA" >&2
  exit 2
}

envelope=""
config=""
checkpoint=""
epoch_record=""
output_dir=""
source_sha=""
while (( $# )); do
  case "$1" in
    --envelope) [[ $# -ge 2 ]] || usage; envelope="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --checkpoint) [[ $# -ge 2 ]] || usage; checkpoint="$2"; shift 2 ;;
    --epoch-record) [[ $# -ge 2 ]] || usage; epoch_record="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || usage; output_dir="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "${envelope}" && -n "${config}" && -n "${checkpoint}" ]] || usage
[[ -n "${epoch_record}" && -n "${output_dir}" ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1_lidar_epoch4_diagnostic.py" ]]; then
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi

for variable in envelope config checkpoint epoch_record; do
  value="${!variable}"
  if [[ "${value}" != /* ]]; then
    printf -v "${variable}" '%s' "${source_root}/${value}"
  fi
done
envelope="$(realpath "${envelope}")"
config="$(realpath "${config}")"
checkpoint="$(realpath "${checkpoint}")"
epoch_record="$(realpath "${epoch_record}")"
output_dir="$(realpath -m "${output_dir}")"

fail() { echo "[s10-phase1-lidar-epoch4-diagnostic] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"
[[ ! -e "${output_dir}" ]] || fail "diagnostic output already exists: ${output_dir}"

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

envelope_values="$(python - "${source_root}" "${envelope}" "${config}" \
  "${checkpoint}" "${epoch_record}" "${output_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from fl_v3.config import load_resolved_config


def require(condition, message):
    if not condition:
        raise SystemExit(f"Envelope-B LiDAR diagnostic drift: {message}")


source_root = Path(sys.argv[1]).resolve()
spec = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
config_path = Path(sys.argv[3]).resolve()
checkpoint = Path(sys.argv[4]).resolve()
epoch_record = Path(sys.argv[5]).resolve()
output_dir = Path(sys.argv[6]).resolve()

require(spec["schema_version"] == "s10.phase1.envelope_b_dual.v2", "schema")
require(spec["request_state"] == "parallel_amendment_owner_activation_required", "state")
require(spec["execution_topology"]["mode"] == "independent_camera_lidar_parallel", "topology")
require(spec["aggregate_resource"]["max_concurrency"] == 2, "concurrency")
require(spec["activation"]["source_grants_compute_authority"] is False, "authority")
for name, entry in spec["entries"].items():
    path = (source_root / entry["path"]).resolve()
    require(path.is_file(), f"{name} entry path")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], f"{name} entry identity")

binding = spec["branches"]["lidar"]
require(config_path == (source_root / binding["config_path"]).resolve(), "config path")
require(hashlib.sha256(config_path.read_bytes()).hexdigest() == binding["config_file_sha256"], "config file")
require(load_resolved_config(config_path).sha256 == binding["resolved_config_sha256"], "resolved config")

diagnostic = spec["diagnostics"]["lidar_epoch04"]
require(checkpoint == Path(diagnostic["checkpoint_path"]).resolve(), "checkpoint path")
require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == diagnostic["checkpoint_sha256"], "checkpoint hash")
require(epoch_record == Path(diagnostic["epoch_record_path"]).resolve(), "epoch record path")
require(hashlib.sha256(epoch_record.read_bytes()).hexdigest() == diagnostic["epoch_record_sha256"], "epoch record hash")
require(output_dir == Path(diagnostic["output_dir"]).resolve(), "output path")
resource = diagnostic["resource"]
for value in (
    resource["account"], resource["partition"], resource["nodes"],
    resource["ntasks"], resource["cpus_per_task"], resource["memory_mib"],
    resource["gpus_per_node"],
):
    print(value)
PY
)" || fail "manifest validation failed"
mapfile -t frozen_resource <<< "${envelope_values}"
[[ "${#frozen_resource[@]}" -eq 7 ]] || fail "resource manifest output is incomplete"

equal "nuScenes dataroot" "${NUSCENES_DATAROOT}" \
  "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
equal "Slurm account" "${SLURM_JOB_ACCOUNT:-}" "${frozen_resource[0]}"
equal "Slurm partition" "${SLURM_JOB_PARTITION:-}" "${frozen_resource[1]}"
equal "Slurm node count" "${SLURM_NNODES:-}" "${frozen_resource[2]}"
equal "Slurm task count" "${SLURM_NTASKS:-}" "${frozen_resource[3]}"
equal "Slurm restart count" "${SLURM_RESTART_COUNT:-0}" "0"
equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "${frozen_resource[4]}"
equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "${frozen_resource[5]}"
equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "${frozen_resource[6]}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-}"
[[ "${#visible_devices[@]}" -eq 1 ]] || fail "diagnostic requires one visible GPU"

cd "${source_root}"
export WORLD_SIZE=1
exec python fl_v3/scripts/s10_phase1_lidar_epoch4_diagnostic.py \
  --config "${config}" --checkpoint "${checkpoint}" \
  --epoch-record "${epoch_record}" --output-dir "${output_dir}" \
  --source-sha "${source_sha}"
