#!/bin/bash
# Per-stage runtime profiler (Cycle-04 D14 Phase-1 A) — measures the per-stage training-step
# breakdown of the headline AD detector in BOTH numeric regimes (fp32, tf32) on a real A40, to
# settle D12's INFERRED "80-90% backbone" share + quantify the TF32 per-stage win. Cheap (a few
# dozen steps). Instrumentation only — produces NO scientific checkpoint.
# Submit:  sbatch fl_v3/scripts/run_profile_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=profile_stages
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=00:30:00
#SBATCH --output=fl_v3/scripts/logs/profile_stages_%j.out
#SBATCH --error=fl_v3/scripts/logs/profile_stages_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/speedup

# The trainval info-cache (regime-independent sample metadata) lives in the T4 worktree; point at it
# (no 2 GB copy). Overridable via TRAINVAL_CACHE.
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
STEPS="${STEPS:-20}"; WARMUP="${WARMUP:-5}"; CLIENT_ID="${CLIENT_ID:-0}"
PRECISIONS="${PRECISIONS:-bf16,fp32}"
COMPILE_FLAG=""; [ "${COMPILE:-0}" = "1" ] && { COMPILE_FLAG="--compile"; WARMUP="${WARMUP_COMPILE:-12}"; }
OUT="${OUT:-fl_v3/collab/speedup/profile_stages.json}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[run_profile_a40] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || {
    echo "[run_profile_a40] FATAL: trainval cache not found at ${TRAINVAL_CACHE}"; exit 3; }

echo "[run_profile_a40] precisions=${PRECISIONS} compile=${COMPILE:-0} warmup=${WARMUP}"
python "${REPO}/fl_v3/scripts/profile_stages_a40.py" "$CONFIG" \
    --steps "$STEPS" --warmup "$WARMUP" --client-id "$CLIENT_ID" \
    --precisions "$PRECISIONS" $COMPILE_FLAG \
    --cache-dir "$TRAINVAL_CACHE" \
    --out "$OUT"
echo "[run_profile_a40] done; elapsed ${SECONDS}s"
