#!/bin/bash
# E — clean FL convergence reference in the D14 TF32 regime (Phase-2 E). Full-participation log-group
# trainval clean FedAvg at ROUNDS rounds, numeric-mode=tf32, with a CHEAP per-round proxy curve
# (server-eval-mode=every_n on a small subset) to SEE the convergence shape. This run IS the new TF32
# clean reference (D14): it writes provenance.json (D10-compliant, numeric-mode=tf32). Endpoint
# official mAP/NDS + readiness come post-hoc from run_t4_readiness_eval on the checkpoint (TF32).
# Submit:  ROUNDS=15 sbatch fl_v3/scripts/run_clean_fl_tf32_a40.sh   (and ROUNDS=30 for the 30-round arm)
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=clean_fl_tf32
#SBATCH --gpus-per-node=A40:4
#SBATCH --time=18:00:00
#SBATCH --output=fl_v3/scripts/logs/clean_fl_tf32_%j.out
#SBATCH --error=fl_v3/scripts/logs/clean_fl_tf32_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs fl_v3/collab/speedup

CONFIG="${CONFIG:-fl_v3/configs/t4_reference.json}"
FEDERATION="${FEDERATION:-local-simulation-gpu-4x}"
ROUNDS="${ROUNDS:-15}"
NUMERIC_MODE="${NUMERIC_MODE:-tf32}"
EVAL_FREQ="${EVAL_FREQ:-3}"
EVAL_LIMIT="${EVAL_LIMIT:-500}"
TAG="${TAG:-clean_fl_tf32_r${ROUNDS}}"
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/infallible-feistel-d42c34/fl_outputs/nuscenes/info_cache}"
OUT_DIR="${REPO}/fl_outputs/nuscenes/experiments/cycle_04/speedup_E"

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
    echo "[clean-fl-tf32] FATAL: trainval cache not found at ${TRAINVAL_CACHE}"; exit 3; }
fl_preflight_offline "$TRAINVAL_CACHE"
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml" "[superlink.${FEDERATION}]"

# D10 preflight (this run IS a reference): clean full-participation log-group trainval FedAvg.
python - "$CONFIG" <<'PY' || { echo "[clean-fl-tf32] FATAL: config not D10-compliant"; exit 3; }
import json, sys
c = json.load(open(sys.argv[1]))
req = {"task-type":"nuscenes_detection","nuscenes-version":"v1.0-trainval","nuscenes-train-split":"train",
       "nuscenes-val-split":"val","nuscenes-partition-mode":"log_group","defense-type":"none"}
bad = [f"{k}={c.get(k)!r}!={v!r}" for k,v in req.items() if str(c.get(k))!=v]
if float(c.get("fraction-train",-1))!=1.0: bad.append("fraction-train!=1.0")
if bad: print("D10 violations:", bad, file=sys.stderr); sys.exit(1)
PY

echo "===== E clean FL (rounds=${ROUNDS} numeric=${NUMERIC_MODE} eval=every_${EVAL_FREQ}/${EVAL_LIMIT}) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

RC="$(python fl_v3/scripts/runconfig.py "$CONFIG" "experiment-name=${TAG}" \
    "nuscenes-cache-dir=${TRAINVAL_CACHE}" "output-dir=${OUT_DIR}" \
    "num-server-rounds=${ROUNDS}" "numeric-mode=${NUMERIC_MODE}" \
    "server-eval-mode=every_n" "server-eval-frequency=${EVAL_FREQ}" "det-eval-limit=${EVAL_LIMIT}")"
echo "run-config: ${RC}"
LOG="${FLWR_HOME}/flwr_${TAG}.log"
set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$RC" | tee "$LOG"; EX=${PIPESTATUS[0]}; set -e
fl_silent_exit_guard "$LOG" || exit 1
[ "$EX" -eq 0 ] || { echo "[clean-fl-tf32] flwr exit ${EX}"; exit "$EX"; }

CHK="$(cat "${OUT_DIR}/${TAG}/trainable_checksum.txt" 2>/dev/null)"
# Write D10 provenance (numeric-mode tf32) beside the checkpoint so the readiness eval verifies it.
CHK="$CHK" python - "$CONFIG" "${OUT_DIR}/${TAG}/provenance.json" "$ROUNDS" "$NUMERIC_MODE" <<'PY'
import json, os, sys
from fl_v3.eval.provenance import build_provenance
cfg = json.load(open(sys.argv[1])); cfg["num-server-rounds"] = int(sys.argv[3]); cfg["numeric-mode"] = sys.argv[4]
prov = build_provenance(cfg, os.environ.get("CHK",""))
json.dump(prov, open(sys.argv[2],"w"), indent=2, sort_keys=True)
print("[clean-fl-tf32] wrote provenance:", prov["regime"], "numeric-mode=", prov["numeric-mode"])
PY
echo "===== E RESULT (rounds=${ROUNDS}) ====="
echo "device          = $(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "checkpoint      = ${OUT_DIR}/${TAG}/final_model.pt"
echo "FL_TRAINABLE_CHECKSUM = ${CHK}"
echo "[clean-fl-tf32] Next: readiness eval on this checkpoint in TF32:"
echo "  sbatch fl_v3/scripts/run_readiness_tf32_a40.sh  (CKPT=${OUT_DIR}/${TAG}/final_model.pt)"
echo "Elapsed: ${SECONDS}s"
