#!/bin/bash
# Phase-1 B acceptance: server-eval gating is RNG-NEUTRAL (Cycle-04 D14). Two same-seed short null
# FL runs that differ ONLY in server-eval-mode (none vs all) must produce a BYTE-IDENTICAL
# FL_TRAINABLE_CHECKSUM (only logs/metrics differ). Run in the D14 numeric regime (tf32) at trainval
# scale + full participation — so it ALSO is the first end-to-end TF32 FL smoke (de-risks D/E).
# Submit:  sbatch fl_v3/scripts/run_b_eval_neutral_a40.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=b_eval_neutral
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=02:00:00
#SBATCH --output=fl_v3/scripts/logs/b_eval_neutral_%j.out
#SBATCH --error=fl_v3/scripts/logs/b_eval_neutral_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/speedup

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
FEDERATION="${FEDERATION:-local-simulation-gpu-4x}"
ROUNDS="${ROUNDS:-3}"
NUMERIC_MODE="${NUMERIC_MODE:-tf32}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"     # tiny eval subset so the eval=all arm is cheap
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
OUT_DIR="${REPO}/fl_outputs/nuscenes/experiments/cycle_04/speedup_b"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT

N="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-num-clients'])")"
APP_DIR="${REPO}/fl_v3"
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || {
    echo "[b-neutral] FATAL: trainval cache not found at ${TRAINVAL_CACHE}"; exit 3; }
fl_preflight_offline "$TRAINVAL_CACHE"
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml" "[superlink.${FEDERATION}]"

COMMON=("nuscenes-cache-dir=${TRAINVAL_CACHE}" "output-dir=${OUT_DIR}"
        "num-server-rounds=${ROUNDS}" "numeric-mode=${NUMERIC_MODE}"
        "det-eval-limit=${EVAL_LIMIT}")

echo "===== B eval-neutrality (rounds=${ROUNDS} numeric=${NUMERIC_MODE} N=${N}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

run_one() {  # $1 = tag, $2 = server-eval-mode
    local tag="$1" mode="$2" log="${FLWR_HOME}/flwr_${1}.log"
    local rc; rc="$(python fl_v3/scripts/runconfig.py "$CONFIG" "experiment-name=${tag}" \
        "server-eval-mode=${mode}" "${COMMON[@]}")"
    echo "----- ${tag} (server-eval-mode=${mode}) -----"; echo "run-config: ${rc}"
    set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$rc" | tee "$log"; local ex=${PIPESTATUS[0]}; set -e
    fl_silent_exit_guard "$log" || return 1
    [ "$ex" -eq 0 ] || { echo "[b-neutral] ${tag} flwr exit ${ex}"; return "$ex"; }
}
read_chk() { cat "${OUT_DIR}/$1/trainable_checksum.txt" 2>/dev/null; }

run_one b_eval_none none || exit 1
run_one b_eval_all  all  || exit 1
CHK_NONE="$(read_chk b_eval_none)"; CHK_ALL="$(read_chk b_eval_all)"
echo "===== B RESULT ====="
echo "device           = $(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "checksum(none)   = ${CHK_NONE}"
echo "checksum(all)    = ${CHK_ALL}"
python - "$CHK_NONE" "$CHK_ALL" "$NUMERIC_MODE" "$ROUNDS" <<'PY'
import json, sys
none_chk, all_chk, mode, rounds = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ok = bool(none_chk) and none_chk == all_chk
res = {"phase": "B-eval-neutrality", "numeric_mode": mode, "rounds": rounds,
       "checksum_eval_none": none_chk, "checksum_eval_all": all_chk,
       "byte_identical": ok, "verdict": "PASS" if ok else "FAIL"}
open("fl_v3/collab/speedup/b_eval_neutral.json", "w").write(json.dumps(res, indent=2, sort_keys=True))
print("[b-neutral] wrote fl_v3/collab/speedup/b_eval_neutral.json")
print(f"[b-neutral] VERDICT: {'PASS — eval gating is RNG-neutral (byte-identical checksum)' if ok else 'FAIL — eval gating PERTURBED training; investigate'}")
sys.exit(0 if ok else 1)
PY
echo "Elapsed: ${SECONDS}s"
