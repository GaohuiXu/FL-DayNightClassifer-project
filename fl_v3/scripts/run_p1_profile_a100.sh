#!/bin/bash
# MCR Phase-1 per-component profiler for the UNFROZEN model on ONE A100 (single-GPU). SHORT job
# (<<3h) — the prerequisite diagnostic before any heavy backbone-training run (orchestrator rule):
# per-component fwd+bwd teardown + GPU-util + activation-ckpt overhead + batch-size sweep.
# Submit:  sbatch fl_v3/scripts/run_p1_profile_a100.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=p1_profile
#SBATCH --gpus-per-node=A100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --output=fl_v3/scripts/logs/p1_profile_%j.out
#SBATCH --error=fl_v3/scripts/logs/p1_profile_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/model_capability

CONFIG="${CONFIG:-fl_v3/configs/p1_unfrozen.json}"
STEPS="${STEPS:-20}"; WARMUP="${WARMUP:-6}"; BATCH_SIZES="${BATCH_SIZES:-16,24,32}"
OPTC=""; [ "${OPT_COMPARE:-0}" = "1" ] && OPTC="--opt-compare"   # OPT_COMPARE=1 → measure compile/channels_last
OUT="${OUT:-fl_v3/collab/model_capability/p1_profile_a100.json}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export WANDB_MODE=offline
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

echo "===== P1 per-component profile (UNFROZEN, single A100) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || { echo "[p1-profile] FATAL: trainval cache missing at ${TRAINVAL_CACHE}"; exit 3; }
[ -f "${TORCH_HOME}/hub/checkpoints/swin_t-704ceda3.pth" ] || { echo "[p1-profile] FATAL: swin_t weights missing under TORCH_HOME"; exit 3; }

# Belt-and-suspenders: a bash-level GPU-util sampler (the D15-proven approach; runs OUTSIDE python so it
# can't be affected by the GIL/CUDA-thread issue). Gives the authoritative overall util for the run.
UTIL_CSV="fl_v3/scripts/logs/p1_util_${SLURM_JOB_ID:-local}.csv"
echo "ts_s,util_pct,mem_used_mib" > "$UTIL_CSV"
( while true; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
      | head -1 | awk -v t="$SECONDS" '{print t","$0}' | tr -d ' ' >> "$UTIL_CSV"
    sleep 1
  done ) &
UTIL_PID=$!
trap 'kill "${UTIL_PID:-}" 2>/dev/null || true' EXIT

python fl_v3/scripts/p1_profile_a100.py \
    --config "$CONFIG" --steps "$STEPS" --warmup "$WARMUP" --batch-sizes "$BATCH_SIZES" $OPTC --out "$OUT" \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}"

kill "${UTIL_PID:-}" 2>/dev/null || true
echo "===== bash-level GPU-util over the whole run (authoritative overall) ====="
python3 -c "
import csv
rows=[r for r in csv.reader(open('${UTIL_CSV}'))][1:]
u=[float(r[1]) for r in rows if len(r)>=2 and r[1].strip().replace('.','',1).isdigit()]
if u:
    u2=sorted(u); n=len(u2)
    print(f'[bash-util] n={n} mean={sum(u)/n:.1f}% p50={u2[n//2]}% p90={u2[min(n-1,int(0.9*n))]}% max={max(u)}%')
else: print('[bash-util] no samples collected')
" || true
echo "[p1-profile] done; elapsed ${SECONDS}s; wrote ${OUT}"
