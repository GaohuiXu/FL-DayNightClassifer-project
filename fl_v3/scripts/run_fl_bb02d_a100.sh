#!/bin/bash
# MCR Phase-3 (D17) — clean FL baseline: federate bb02d with FedAdam (server optimizer) + the FL-tuned
# recipe. FULL participation (fraction-train=1.0, all 25 log_group clients/round) so the checkpoint is a
# valid D10 reference (T4_SPEC §0.2). A100:4 via D9 Path-A (1 Ray client per GPU). Reads the 10-sweep
# msweep10 trainval cache (the data bb02d was trained on) — set TRAINVAL_CACHE if it moved.
#
# Submit (FULL run — needs owner GO, >3h):   sbatch fl_v3/scripts/run_fl_bb02d_a100.sh
# Cheap PROBE sweep (short, no GO needed):    ROUNDS=6 SERVER_LR=0.01 CLIENT_LR=0.001 EVAL_MODE=every_n \
#                                             EVAL_LIMIT=256 TAG=probe_s0.01_c0.001 sbatch ... run_fl_bb02d_a100.sh
#
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --job-name=fl_bb02d
#SBATCH --gpus-per-node=A100:4
#SBATCH --exclude=alvis4-01,alvis4-02
#SBATCH --time=20:00:00
#SBATCH --output=fl_v3/scripts/logs/fl_bb02d_%j.out
#SBATCH --error=fl_v3/scripts/logs/fl_bb02d_%j.err
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then REPO="${SLURM_SUBMIT_DIR}"; else REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; fi
cd "$REPO"; mkdir -p fl_v3/scripts/logs

CONFIG="${CONFIG:-fl_v3/configs/fl_bb02d_fedadam.json}"
FEDERATION="${FEDERATION:-local-simulation-gpu-4x}"   # D9 Path-A (4 A100s, 1 client/GPU)
TAG="${TAG:-fl_bb02d_fedadam}"
# The 10-sweep trainval cache bb02d uses (msweep10). Defaults to the worktree that built it; override if moved.
TRAINVAL_CACHE="${TRAINVAL_CACHE:-${PROJ_ROOT}/.claude/worktrees/unruffled-chaplygin-0e43e5/fl_outputs/nuscenes/info_cache_msweep10}"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${PROJ_ROOT}/.cache/torchinductor_fl"   # persistent compile cache (compile-backbone)
# BLAS thread-pin: the 10-sweep loader is loader-bound; 4 Ray clients × num-workers × default OMP threads
# oversubscribes the CPUs and thrashes the ego-comp matmul (MCR P2 finding). 1 thread/worker is the fix
# (seeded_worker_init also calls torch.set_num_threads(1) per worker).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if ! type module >/dev/null 2>&1; then [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash; fi
module purge; module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
# CRITICAL: the shared .venv_v3 editable-installs fl_v3 from ONE worktree's src; without this prepend, a
# bare `flwr run` (+ its Ray workers, which inherit this env) would import a SIBLING worktree's STALE code
# (no FedAdam / server-EMA / snapshots / client-recipe). Make THIS worktree authoritative (matches
# run_in_venv.sh + run_p1_profile_a100.sh). PYTHONPATH propagates to the forked Ray actors.
export PYTHONPATH="${REPO}/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source fl_v3/scripts/_fl_env.sh
fl_setup_env
trap 'rm -rf "${FLWR_HOME:-/tmp/none}" "${RAY_TMPDIR:-/tmp/none}"' EXIT

N="$(python -c "import json;print(json.load(open('${CONFIG}'))['nuscenes-num-clients'])")"
OUT_DIR="$(python -c "import json;print(json.load(open('${CONFIG}'))['output-dir'])")"
APP_DIR="${REPO}/fl_v3"
OUT_DIR="${REPO}/${OUT_DIR#./}"
[ -f "${TRAINVAL_CACHE}/nuscenes_info_v1.0-trainval_train_t1.v1.pkl" ] || {
    echo "[fl-bb02d] FATAL: msweep10 trainval cache not found at TRAINVAL_CACHE=${TRAINVAL_CACHE}"; exit 4; }
fl_preflight_offline "$TRAINVAL_CACHE"
# Stamp the DERIVED N into the Path-A (-4x) federation block.
fl_stamp_supernodes "$N" fl_v3/configs/flwr_config.toml "${FLWR_HOME}/config.toml" "[superlink.${FEDERATION}]"

# Optional probe/sweep overrides (cheap tuning BEFORE the full run). Empty ⇒ use the config's values.
OVR=("nuscenes-cache-dir=${TRAINVAL_CACHE}" "output-dir=${OUT_DIR}" "experiment-name=${TAG}")
[ -n "${ROUNDS:-}" ]      && OVR+=("num-server-rounds=${ROUNDS}")
[ -n "${SERVER_LR:-}" ]   && OVR+=("server-lr=${SERVER_LR}")
[ -n "${CLIENT_LR:-}" ]   && OVR+=("learning-rate=${CLIENT_LR}")
[ -n "${SERVER_TAU:-}" ]  && OVR+=("server-tau=${SERVER_TAU}")
[ -n "${LOCAL_EPOCHS:-}" ]&& OVR+=("num-local-epochs=${LOCAL_EPOCHS}")
[ -n "${EVAL_MODE:-}" ]   && OVR+=("server-eval-mode=${EVAL_MODE}")
[ -n "${EVAL_LIMIT:-}" ]  && OVR+=("det-eval-limit=${EVAL_LIMIT}")
[ -n "${SNAP_ROUNDS:-}" ] && OVR+=("snapshot-rounds=${SNAP_ROUNDS}")
[ -n "${EXTRA_OVERRIDES:-}" ] && OVR+=(${EXTRA_OVERRIDES})

echo "===== FL bb02d FedAdam (FULL participation, D10) =====  node=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "N=${N} federation=${FEDERATION} config=${CONFIG} tag=${TAG} cache=${TRAINVAL_CACHE}"
echo "overrides: ${OVR[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# D10 / T4_SPEC §0.2 PREFLIGHT (hard-fail): the reference must be full-participation log-group trainval clean.
# SKIP_D10=1 bypasses this for a DIAGNOSTIC run (e.g. the 1-client iid sanity check) — never a real reference.
if [ "${SKIP_D10:-0}" = "1" ]; then echo "[fl-bb02d] SKIP_D10=1 — diagnostic run, NOT a valid D10 reference"; else
python - "$CONFIG" <<'PY' || { echo "[fl-bb02d] FATAL: config NOT D10-compliant (T4_SPEC §0.2) — refusing"; exit 3; }
import json, sys
c = json.load(open(sys.argv[1]))
req = {"task-type": "nuscenes_detection", "nuscenes-version": "v1.0-trainval",
       "nuscenes-train-split": "train", "nuscenes-val-split": "val",
       "nuscenes-partition-mode": "log_group", "defense-type": "none"}
bad = [f"{k}={c.get(k)!r}!= {v!r}" for k, v in req.items() if str(c.get(k)) != v]
if float(c.get("fraction-train", -1)) != 1.0:
    bad.append(f"fraction-train={c.get('fraction-train')!r} != 1.0 (D10 FULL participation)")
if bad:
    print("[fl-bb02d] D10 violations:", "; ".join(bad), file=sys.stderr); sys.exit(1)
PY
fi

RC="$(python fl_v3/scripts/runconfig.py "$CONFIG" "${OVR[@]}")"
LOG="${FLWR_HOME}/flwr_${TAG}.log"
echo "run-config: ${RC}"
set +e; flwr run "$APP_DIR" "$FEDERATION" --stream --run-config "$RC" | tee "$LOG"; EX=${PIPESTATUS[0]}; set -e
fl_silent_exit_guard "$LOG" || exit 1
[ "$EX" -eq 0 ] || { echo "[fl-bb02d] flwr exit ${EX}"; exit "$EX"; }

CHK="$(cat "${OUT_DIR}/${TAG}/trainable_checksum.txt" 2>/dev/null)"
# Persist D10 provenance beside final_model.pt AND in ema/ + every round_<r>/ (and their ema/) so the
# readiness eval can verify D10 on whichever snapshot peaks (the centralized model peaked mid-run).
CHK="$CHK" CFG="$CONFIG" EXPDIR="${OUT_DIR}/${TAG}" python - <<'PY'
import json, os, glob
from fl_v3.eval.provenance import build_provenance, check_d10
cfg = json.load(open(os.environ["CFG"]))
expdir = os.environ["EXPDIR"]
def write_prov(d, checksum):
    if not checksum: return
    prov = build_provenance(cfg, checksum)
    bad = check_d10(prov)
    if bad: print("[fl-bb02d] WARNING provenance not D10:", bad)
    json.dump(prov, open(os.path.join(d, "provenance.json"), "w"), indent=2, sort_keys=True)
def chk(d):
    p = os.path.join(d, "trainable_checksum.txt")
    return open(p).read().strip() if os.path.isfile(p) else ""
# top-level final + ema, plus every round_<r>/ and round_<r>/ema/
for d in [expdir, os.path.join(expdir, "ema")] + sorted(glob.glob(os.path.join(expdir, "round_*"))) \
         + sorted(glob.glob(os.path.join(expdir, "round_*", "ema"))):
    if os.path.isdir(d):
        write_prov(d, chk(d))
print("[fl-bb02d] wrote provenance.json into all snapshot dirs")
PY
echo "===== FL bb02d RESULT ====="
echo "device          = $(python -c 'import torch;print(torch.cuda.get_device_name())')"
echo "checkpoint      = ${OUT_DIR}/${TAG}/final_model.pt"
echo "FL_TRAINABLE_CHECKSUM = ${CHK}"
echo "snapshots       = $(ls -d ${OUT_DIR}/${TAG}/round_* 2>/dev/null | tr '\n' ' ')"
echo "Elapsed: ${SECONDS}s"
echo "[fl-bb02d] Next: post-hoc official mAP/NDS on the snapshots via run_eval_ckpt_a100.sh (pick the PEAK round)"
