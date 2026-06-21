#!/bin/bash
# Overcommit DIAGNOSTIC suite (why A100 4/GPU oscillated ~30% util + slow rounds). Per-step teardown on
# ONE 4-A100 node — reproduces the real node-level contention (shared FS, host RAM, CPU) that a 1-GPU
# probe under-samples. Three phases, one node allocation:
#   E1  GPU-sharing CEILING  : K=1/2/3/4 profilers on GPU 0/1/2/3, fixed-batch (dataloader bypassed).
#   E2  DATALOADER-only      : single-loader num-workers sweep + 16 concurrent loaders (FS I/O wall).
#   E3  REALISTIC 16-actor   : 4 profilers/GPU x4, real data, num-workers in {2,4} (the failed config-Ray).
# All relaxed bf16, compile OFF (matches the failed run + avoids the cache-race confound). Instrumentation
# only — never feeds a checkpoint. Submit:  sbatch fl_v3/scripts/run_overcommit_diag_a100.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=oc_diag
#SBATCH --gpus-per-node=A100:4
#SBATCH --time=02:00:00
#SBATCH --output=fl_v3/scripts/logs/oc_diag_%j.out
#SBATCH --error=fl_v3/scripts/logs/oc_diag_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
STEPS="${STEPS:-20}"; WARMUP="${WARMUP:-8}"; DLB_BATCHES="${DLB_BATCHES:-40}"
OUT="${OUT:-fl_v3/collab/speedup/overcommit_diag/job_${SLURM_JOB_ID:-local}}"
mkdir -p "$OUT"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"

SAMPLER_PID=""
trap 'kill "${SAMPLER_PID:-}" 2>/dev/null || true' EXIT

echo "[oc_diag] node=$(hostname) job=${SLURM_JOB_ID:-local} OUT=${OUT}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || {
    echo "[oc_diag] FATAL: trainval cache not found at ${TRAINVAL_CACHE}"; exit 3; }

PY="fl_v3/scripts/profile_stages_a40.py"
DLB="fl_v3/scripts/bench_dataloader_a100.py"
COMMON="--steps ${STEPS} --warmup ${WARMUP} --modes fp32 --level relaxed --cache-dir ${TRAINVAL_CACHE}"

start_sampler() {  # $1 = csv path
    ( echo "ts_s,gpu_idx,util_pct,mem_used_mib"
      while true; do
          nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
            | awk -v t="$SECONDS" '{print t","$0}' | tr -d ' '
          free -m | awk -v t="$SECONDS" '/^Mem:/{print t",H,," $3}'
          sleep 2
      done ) > "$1" 2>/dev/null &
    SAMPLER_PID=$!
}
stop_sampler() { [ -n "${SAMPLER_PID:-}" ] && { kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""; }; }

# ===================== E1 — GPU-sharing ceiling (fixed batch) =====================
echo "===== E1 ceiling sweep: K=1/2/3/4 on GPU 0/1/2/3 (fixed-batch) ====="
start_sampler "${OUT}/util_e1.csv"
E1K=(1 2 3 4); pids=(); cid=0
for g in 0 1 2 3; do
    k="${E1K[$g]}"
    for ((j=0; j<k; j++)); do
        CUDA_VISIBLE_DEVICES="$g" python "$PY" "$CONFIG" $COMMON --fixed-batch --client-id "$cid" \
            --out "${OUT}/e1_g${g}_p${j}_cid${cid}.json" > "${OUT}/e1_g${g}_p${j}.log" 2>&1 &
        pids+=($!); cid=$((cid+1))
    done
done
ok=0; for p in "${pids[@]}"; do wait "$p" && ok=$((ok+1)) || echo "[E1] proc $p exit $?"; done
echo "[E1] ${ok}/${#pids[@]} procs ok"
stop_sampler

# ===================== E2 — dataloader-only =====================
echo "===== E2a single-loader num-workers sweep (2,4,8) ====="
start_sampler "${OUT}/util_e2.csv"
for nw in 2 4 8; do
    python "$DLB" "$CONFIG" --cache-dir "$TRAINVAL_CACHE" --num-workers "$nw" --batches "$DLB_BATCHES" \
        --client-id 0 --label "dlsingle_nw${nw}" --out "${OUT}/dl_single_nw${nw}.json" \
        > "${OUT}/dl_single_nw${nw}.log" 2>&1 || echo "[E2a nw${nw}] exit $?"
done
echo "===== E2b 16 concurrent loaders (nw=2, distinct shards) — shared-FS I/O ====="
pids=()
for ((c=0; c<16; c++)); do
    python "$DLB" "$CONFIG" --cache-dir "$TRAINVAL_CACHE" --num-workers 2 --batches "$DLB_BATCHES" \
        --client-id "$c" --label "dlconc_nw2" --out "${OUT}/dl_conc_cid${c}.json" \
        > "${OUT}/dl_conc_cid${c}.log" 2>&1 &
    pids+=($!)
done
ok=0; for p in "${pids[@]}"; do wait "$p" && ok=$((ok+1)) || echo "[E2b] proc $p exit $?"; done
echo "[E2b] ${ok}/16 loaders ok"
stop_sampler

# ===================== E3 — realistic 16-actor (nw 2 then 4) =====================
for nw in 2 4; do
    lab="e3nw${nw}"
    echo "===== E3 realistic: 4 profilers/GPU x4 = 16 actors, num-workers=${nw} ====="
    start_sampler "${OUT}/util_${lab}.csv"
    pids=(); cid=0
    for g in 0 1 2 3; do
        for ((j=0; j<4; j++)); do
            CUDA_VISIBLE_DEVICES="$g" python "$PY" "$CONFIG" $COMMON --num-workers "$nw" --client-id "$cid" \
                --out "${OUT}/${lab}_g${g}_p${j}_cid${cid}.json" > "${OUT}/${lab}_g${g}_p${j}.log" 2>&1 &
            pids+=($!); cid=$((cid+1))
        done
    done
    ok=0; for p in "${pids[@]}"; do wait "$p" && ok=$((ok+1)) || echo "[${lab}] proc $p exit $?"; done
    echo "[${lab}] ${ok}/16 procs completed (shortfall => OOM/starve, itself a result)"
    stop_sampler
done

# ===================== aggregate =====================
echo "===== AGGREGATE ====="
python fl_v3/scripts/agg_overcommit_diag.py "$OUT" | tee "${OUT}/SUMMARY.txt"
echo "[oc_diag] done; OUT=${OUT}; Elapsed: ${SECONDS}s"
