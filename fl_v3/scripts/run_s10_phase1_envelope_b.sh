#!/bin/bash
# Exact one-candidate S10 Phase-I Envelope-B launcher.
# Camera uses the owner-promoted same-node 2xGH200 DDP recipe; LiDAR stays single-GPU.
set -euo pipefail

usage() {
  echo "usage: $0 --branch {camera|lidar} --envelope PATH --config PATH --output-dir PATH --source-sha SHA [--resume]" >&2
  exit 2
}

branch=""
envelope=""
config=""
output_dir=""
source_sha=""
resume=0
while (( $# )); do
  case "$1" in
    --branch) [[ $# -ge 2 ]] || usage; branch="$2"; shift 2 ;;
    --envelope) [[ $# -ge 2 ]] || usage; envelope="$2"; shift 2 ;;
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --output-dir) [[ $# -ge 2 ]] || usage; output_dir="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) usage ;;
  esac
done

[[ "${branch}" == "camera" || "${branch}" == "lidar" ]] || usage
[[ -n "${envelope}" && -n "${config}" && -n "${output_dir}" ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1_capability.py" ]]; then
  # Slurm executes a copied batch script from its spool.  Bind relative paths to
  # the immutable submit worktree, not to BASH_SOURCE[0] inside that spool.
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi
single_entry="fl_v3/scripts/s10_phase1_capability.py"
camera_entry="fl_v3/scripts/s10_phase1_camera_ddp.py"
if [[ "${config}" != /* ]]; then
  config="${source_root}/${config}"
fi
if [[ "${envelope}" != /* ]]; then
  envelope="${source_root}/${envelope}"
fi
config="$(realpath "${config}")"
envelope="$(realpath "${envelope}")"
output_dir="$(realpath -m "${output_dir}")"

fail() { echo "[s10-phase1-b] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" ]] || fail "config missing: ${config}"
[[ -f "${envelope}" ]] || fail "envelope manifest missing: ${envelope}"
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

envelope_values="$(python - "${source_root}" "${envelope}" "${branch}" \
  "${config}" "${output_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from fl_v3.config import load_resolved_config


def require(condition, message):
    if not condition:
        raise SystemExit(f"Envelope-B manifest drift: {message}")


source_root = Path(sys.argv[1]).resolve()
envelope_path = Path(sys.argv[2]).resolve()
branch = sys.argv[3]
config_path = Path(sys.argv[4]).resolve()
output_dir = Path(sys.argv[5]).resolve()
spec = json.loads(envelope_path.read_text(encoding="utf-8"))

require(spec["schema_version"] == "s10.phase1.envelope_b_dual.v2", "schema")
require(spec["request_state"] == "parallel_amendment_owner_activation_required", "state")
require(spec["branch"] == "codex/s10-phase1p-throughput-preflight", "branch")
require(spec["candidate_count"] == 2, "candidate count")
topology = spec["execution_topology"]
require(topology["mode"] == "independent_camera_lidar_parallel", "topology")
require(topology["per_branch_max_concurrency"] == 1, "per-branch concurrency")
require(spec["seed_policy"] == [0], "seed policy")
require(spec["aggregate_resource"]["max_concurrency"] == 2, "concurrency")
require(spec["activation"]["source_grants_compute_authority"] is False, "authority")
for name, entry in spec["entries"].items():
    entry_path = (source_root / entry["path"]).resolve()
    require(entry_path.is_file(), f"{name} entry path")
    require(
        hashlib.sha256(entry_path.read_bytes()).hexdigest() == entry["sha256"],
        f"{name} entry identity",
    )

binding = spec["branches"][branch]
expected_config = (source_root / binding["config_path"]).resolve()
require(config_path == expected_config, "config path")
require(
    hashlib.sha256(config_path.read_bytes()).hexdigest()
    == binding["config_file_sha256"],
    "config file identity",
)
resolved = load_resolved_config(config_path)
require(resolved.sha256 == binding["resolved_config_sha256"], "resolved config identity")
raw = resolved.as_dict()
require(raw["contract"]["branch"] == branch, "config branch")
require(raw["contract"]["candidate_id"] == binding["candidate_id"], "candidate")
require(raw["training"]["seed"] == 0, "training seed")
require(raw["training"]["effective_global_batch"] == 32, "effective batch")
require(raw["execution"]["allowed_data_role"] == "D_fit", "training role")
require(raw["execution"]["allowed_evaluation_roles"] == ["D_select"], "evaluation role")
require(raw["execution"]["output_root"] == spec["output_root"], "config output root")
require(output_dir == Path(binding["output_dir"]).resolve(), "candidate output")

resource = binding["resource"]
for value in (
    resource["account"],
    resource["partition"],
    resource["nodes"],
    resource["ntasks"],
    resource["cpus_per_task"],
    resource["memory_mib"],
    resource["gpus_per_node"],
):
    print(value)
PY
)" || fail "Envelope-B manifest validation failed"
mapfile -t frozen_resource <<< "${envelope_values}"
[[ "${#frozen_resource[@]}" -eq 7 ]] || fail "resource manifest output is incomplete"
expected_account="${frozen_resource[0]}"
expected_partition="${frozen_resource[1]}"
expected_nodes="${frozen_resource[2]}"
expected_ntasks="${frozen_resource[3]}"
expected_cpus="${frozen_resource[4]}"
expected_memory_mib="${frozen_resource[5]}"
expected_gpus="${frozen_resource[6]}"

equal "nuScenes dataroot" "${NUSCENES_DATAROOT}" \
  "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
equal "Slurm account" "${SLURM_JOB_ACCOUNT:-}" "${expected_account}"
equal "Slurm partition" "${SLURM_JOB_PARTITION:-}" "${expected_partition}"
equal "Slurm node count" "${SLURM_NNODES:-}" "${expected_nodes}"
equal "Slurm task count" "${SLURM_NTASKS:-}" "${expected_ntasks}"
equal "Slurm restart count" "${SLURM_RESTART_COUNT:-0}" "0"
equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "${expected_cpus}"
equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "${expected_memory_mib}"
equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "${expected_gpus}"
all_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${all_visible_devices}" ]] || fail "CUDA_VISIBLE_DEVICES is empty"
IFS=',' read -r -a visible_devices <<< "${all_visible_devices}"
[[ "${#visible_devices[@]}" -eq "${expected_gpus}" ]] \
  || fail "visible allocated GPU count differs from frozen branch resource"

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
