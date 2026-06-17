#!/bin/bash
# T3 non-IID gap on TRAINVAL-SCALE clients (SPEC §2.4) — a COMPLETED artifact.
# Runs the headline frozen-Swin-T FedAvg twice on v1.0-trainval (derived N=25), once with the
# location-coherent log_group partition and once IID, ≤20 rounds, fraction<1; reports the
# IID-vs-log-group recall@2m gap + measured wall-clock + each run's final-model checksum.
# Submit: sbatch fl_v3/scripts/run_trainval_gap_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=t3_trainval_gap
#SBATCH --gpus-per-node=A40:1
#SBATCH --time=10:00:00
#SBATCH --output=fl_v3/scripts/logs/t3_trainval_gap_%j.out
#SBATCH --error=fl_v3/scripts/logs/t3_trainval_gap_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/t3_trainval.json}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT

N="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-num-clients'])")"
CACHE_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-cache-dir'])")"
OUT_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['output-dir'])")"
# Absolute app dir + paths (Ray actor cwd is not the repo root).
APP_DIR="${REPO}/fl_v3"
ABS_CACHE="${REPO}/${CACHE_DIR#./}"
OUT_DIR="${REPO}/${OUT_DIR#./}"
PATH_OVERRIDES=("nuscenes-cache-dir=${ABS_CACHE}" "output-dir=${OUT_DIR}")
fl_preflight_offline "$ABS_CACHE"
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml"

echo "===== T3 trainval non-IID gap =====  node=$(hostname) job=${SLURM_JOB_ID:-local} N=${N}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_mode() {  # $1 = partition mode (log_group|iid)
    local mode="$1" tag="$1" log="${FLWR_HOME}/flwr_${1}.log" t0=$SECONDS
    local rc; rc="$(python fl_v3/scripts/runconfig.py "$CONFIG" \
        "nuscenes-partition-mode=${mode}" "experiment-name=t3_trainval_${tag}" "${PATH_OVERRIDES[@]}")"
    echo "----- trainval run: ${mode} -----"; echo "run-config: ${rc}"
    set +e; flwr run "$APP_DIR" local-simulation-gpu --stream --run-config "$rc" | tee "$log"; local ex=${PIPESTATUS[0]}; set -e
    fl_silent_exit_guard "$log" || return 1
    [ "$ex" -eq 0 ] || { echo "[gap] ${mode} flwr exit ${ex}"; return "$ex"; }
    local secs=$((SECONDS - t0))
    # final-round server-eval recall@2m (last [Server] eval line).
    local recall; recall="$(grep '\[Server\] round=' "$log" | tail -1 | python -c "import sys,ast,re; s=sys.stdin.read(); m=re.search(r'eval=(\{.*\})', s); print(ast.literal_eval(m.group(1)).get('proxy_recall_at_2m','NA') if m else 'NA')" 2>/dev/null || echo NA)"
    local chk; chk="$(cat "${OUT_DIR}/t3_trainval_${tag}/trainable_checksum.txt" 2>/dev/null || echo NA)"
    echo "[gap] MODE=${mode} recall@2m=${recall} wallclock_s=${secs} checksum=${chk}"
    echo "${recall}" > "${FLWR_HOME}/recall_${mode}.txt"
    echo "${secs}"  > "${FLWR_HOME}/secs_${mode}.txt"
}

run_mode log_group
run_mode iid

R_LG="$(cat "${FLWR_HOME}/recall_log_group.txt" 2>/dev/null || echo NA)"
R_IID="$(cat "${FLWR_HOME}/recall_iid.txt" 2>/dev/null || echo NA)"
S_LG="$(cat "${FLWR_HOME}/secs_log_group.txt" 2>/dev/null || echo NA)"
S_IID="$(cat "${FLWR_HOME}/secs_iid.txt" 2>/dev/null || echo NA)"
echo "===== T3 TRAINVAL NON-IID GAP RESULTS ====="
echo "scale=trainval-scientific  N=${N}  device=$(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "recall@2m  IID=${R_IID}   log_group=${R_LG}"
python -c "
ri='${R_IID}'; rl='${R_LG}'
try:
    print(f'[gap] non-IID gap (IID - log_group) recall@2m = {float(ri)-float(rl):+.4f}')
except Exception:
    print('[gap] gap = NA (one side missing)')
"
echo "wallclock_s  IID=${S_IID}  log_group=${S_LG}  (per round = total / num-server-rounds)"
echo "Elapsed total: ${SECONDS}s"
