#!/bin/bash
# Exact same-node one-GH200 versus two-GH200 Camera DDP qualification.
set -euo pipefail

usage() {
  echo "usage: $0 --config PATH --reference-profile PATH --ddp-profile PATH --source-sha SHA --approved-source-sha SHA" >&2
  exit 2
}

config=""; reference_profile=""; ddp_profile=""
source_sha=""; approved_source_sha=""
while (( $# )); do
  case "$1" in
    --config) [[ $# -ge 2 ]] || usage; config="$2"; shift 2 ;;
    --reference-profile) [[ $# -ge 2 ]] || usage; reference_profile="$2"; shift 2 ;;
    --ddp-profile) [[ $# -ge 2 ]] || usage; ddp_profile="$2"; shift 2 ;;
    --source-sha) [[ $# -ge 2 ]] || usage; source_sha="$2"; shift 2 ;;
    --approved-source-sha) [[ $# -ge 2 ]] || usage; approved_source_sha="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -n "${config}" && -n "${reference_profile}" && -n "${ddp_profile}" ]] || usage
[[ "${source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "${approved_source_sha}" =~ ^[0-9a-f]{40}$ ]] || usage

if [[ -n "${SLURM_SUBMIT_DIR:-}" && \
      -f "${SLURM_SUBMIT_DIR}/fl_v3/scripts/s10_phase1p_ddp.py" ]]; then
  source_root="$(realpath "${SLURM_SUBMIT_DIR}")"
else
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  source_root="$(cd "${script_dir}/../.." && pwd)"
fi
expected_base_sha="f1a2babda8dafd181b5a5144ab025a3f6be21cc2"
expected_approval_sha="e61c486757ca5fe89340c9325014f4c3e048da2b"

resolve_from_root() {
  local value="$1"
  if [[ "${value}" != /* ]]; then value="${source_root}/${value}"; fi
  realpath "${value}"
}
config="$(resolve_from_root "${config}")"
reference_profile="$(resolve_from_root "${reference_profile}")"
ddp_profile="$(resolve_from_root "${ddp_profile}")"

fail() { echo "[s10-phase1p-ip-e5] ERROR: $*" >&2; exit 2; }
equal() { [[ "$2" == "$3" ]] || fail "$1: actual=$2 expected=$3"; }

[[ -f "${config}" && -f "${reference_profile}" && -f "${ddp_profile}" ]] \
  || fail "config/profile missing"
equal "approved source SHA" "${approved_source_sha}" "${expected_approval_sha}"
equal "source SHA" "$(git -C "${source_root}" rev-parse HEAD)" "${source_sha}"
equal "source branch" "$(git -C "${source_root}" branch --show-current)" \
  "codex/s10-phase1p-throughput-preflight"
equal "frozen Phase-I control" \
  "$(git -C "${source_root}" rev-parse refs/heads/codex/s10-phase1-branch-qualification)" \
  "${expected_base_sha}"
git -C "${source_root}" merge-base --is-ancestor \
  "${expected_base_sha}" "${approved_source_sha}" \
  || fail "design-approval SHA is not descended from the unique IP-G0 base"
git -C "${source_root}" merge-base --is-ancestor \
  "${approved_source_sha}" "${source_sha}" \
  || fail "source SHA is not an approved linear descendant"
[[ -z "$(git -C "${source_root}" rev-list --min-parents=2 \
  "${expected_base_sha}..${source_sha}")" ]] \
  || fail "source history is not linear"
[[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] \
  || fail "source worktree is not clean"

expected_output_root="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e5_${approved_source_sha:0:12}"
smoke_output="${expected_output_root}/ddp_smoke_${source_sha:0:12}"
reference_output="${expected_output_root}/camera/sustained_${source_sha:0:12}_r1_ddp_single_ref"
ddp_output="${expected_output_root}/camera/ddp2_${source_sha:0:12}"
pair_output="${expected_output_root}/pairs/final_b16_single_vs_ddp2.json"
system_csv="${expected_output_root}/ddp2_${source_sha:0:12}_nvidia_smi.csv"
for fresh_path in \
  "${smoke_output}" "${reference_output}" "${ddp_output}" \
  "${pair_output}" "${system_csv}"; do
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_DEBUG=WARN NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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
equal "Slurm CPUs per task" "${SLURM_CPUS_PER_TASK:-}" "32"
equal "Slurm memory per node" "${SLURM_MEM_PER_NODE:-}" "196608"
equal "Slurm GPUs on node" "${SLURM_GPUS_ON_NODE:-0}" "2"
equal "Slurm restart count" "${SLURM_RESTART_COUNT:-0}" "0"

all_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
[[ -n "${all_visible_devices}" ]] || fail "CUDA_VISIBLE_DEVICES is empty"
IFS=',' read -r -a visible_devices <<< "${all_visible_devices}"
[[ "${#visible_devices[@]}" -eq 2 ]] \
  || fail "exactly two allocated CUDA devices must be visible"

cd "${source_root}"
CUDA_VISIBLE_DEVICES="${visible_devices[0]}" WORLD_SIZE=1 python -m pytest -q \
  fl_v3/tests/test_s10_phase1p_checkpoint_gate.py \
  fl_v3/tests/test_s10_phase1p_profile.py::test_ip_e5_profiles_bind_final_camera_recipe \
  fl_v3/tests/test_s10_phase1p_ddp.py \
  fl_v3/tests/test_s10_phase1p_compare.py::test_ip_e5_ddp_gate_requires_robust_speed_and_charged_payback

ddp_command=(
  python -m torch.distributed.run --standalone --nproc-per-node=2 --max-restarts=0
  fl_v3/scripts/s10_phase1p_ddp.py
  --config "${config}"
  --profile-config "${ddp_profile}"
  --source-sha "${source_sha}"
  --approved-source-sha "${approved_source_sha}"
)

CUDA_VISIBLE_DEVICES="${all_visible_devices}" "${ddp_command[@]}" \
  --mode smoke --output-dir "${smoke_output}"

CUDA_VISIBLE_DEVICES="${visible_devices[0]}" WORLD_SIZE=1 \
  python fl_v3/scripts/s10_phase1_throughput.py \
    --branch camera --mode sustained --config "${config}" \
    --profile-config "${reference_profile}" --output-dir "${reference_output}" \
    --source-sha "${source_sha}" --approved-source-sha "${approved_source_sha}" \
    --repeat 1 --attempt-id ddp_single_ref

mkdir -p "${expected_output_root}"
nvidia-smi \
  --query-gpu=timestamp,index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 1 > "${system_csv}" &
sampler_pid=$!
cleanup_sampler() {
  if kill -0 "${sampler_pid}" 2>/dev/null; then kill "${sampler_pid}" 2>/dev/null || true; fi
  wait "${sampler_pid}" 2>/dev/null || true
}
trap cleanup_sampler EXIT

CUDA_VISIBLE_DEVICES="${all_visible_devices}" "${ddp_command[@]}" \
  --mode profile --output-dir "${ddp_output}"
CUDA_VISIBLE_DEVICES="${all_visible_devices}" "${ddp_command[@]}" \
  --mode resume --output-dir "${ddp_output}"
cleanup_sampler
trap - EXIT

python fl_v3/scripts/s10_phase1p_ip_e5_compare.py \
  --reference-dir "${reference_output}" \
  --candidate-dir "${ddp_output}" \
  --output "${pair_output}"
