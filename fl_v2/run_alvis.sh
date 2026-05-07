#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-03:00:00
#SBATCH -J flwr_gtsrb
#SBATCH -o /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.out
#SBATCH -e /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm/%x_%j.err

set -euo pipefail

echo "===== Job info ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
if [[ -n "${EXPERIMENT_YAML:-}" ]]; then
    echo "Experiment YAML: $EXPERIMENT_YAML"
    echo "--- YAML contents ---"
    cat "$EXPERIMENT_YAML"
    echo "--- end YAML ---"
fi

mkdir -p /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm
mkdir -p /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb_v2
# New cycle-aware tree (Cycle 02 onward); experiment_logger.py creates the
# nested experiments/<cycle>/<phase>/ subdirs on demand.
mkdir -p /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/fl_v2
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate

# Per-job FLWR_HOME — isolates concurrent jobs from each other
JOB_FLWR_HOME="/tmp/flwr_${SLURM_JOB_ID}"
mkdir -p "$JOB_FLWR_HOME/local-superlink/ffs"
cp configs/flwr_config.toml "$JOB_FLWR_HOME/config.toml"
export FLWR_HOME="$JOB_FLWR_HOME"

export RAY_DEDUP_LOGS=0

# --- Reproducibility env vars (Phase 1.0 of the recovery plan) ---
# PYTHONHASHSEED=0 makes Python's hash randomisation deterministic across
# runs; protects any code that hashes strings (e.g. dict ordering during
# serialisation). CUBLAS_WORKSPACE_CONFIG is required by
# torch.use_deterministic_algorithms (we may opt-in later) and is harmless
# when that flag is off. Both are runtime requirements, not polish.
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Derive unique SuperLink ports from $SLURM_JOB_ID ---
# Two jobs landing on the same node previously collided on the hardcoded
# 39093/39094 ports — the second job's SuperLink would crash with
# "Port already in use". Derive a deterministic but distinct port pair
# per job: range [39000, 40998] gives 1000 unique pairs. Collision
# requires JIDs differing by exactly 1000, which is rare in practice.
#
# Both `flower-superlink` (server side, --control-api-address flags) AND
# `flwr run` (client side, reads FLWR_LOCAL_CONTROL_API_PORT env var per
# flwr/cli/constant.py) must agree on the ports — exporting the env vars
# below covers the client side.
JID=${SLURM_JOB_ID:-0}
PORT_CTL=$((39000 + (JID % 1000) * 2))
PORT_SIO=$((PORT_CTL + 1))
export FLWR_LOCAL_CONTROL_API_PORT="$PORT_CTL"
export FLWR_LOCAL_SIMULATIONIO_API_PORT="$PORT_SIO"
echo "SuperLink ports: control=$PORT_CTL, simulationio=$PORT_SIO (derived from JID $JID)"

# --- Derive unique Ray internal ports from $SLURM_JOB_ID ---
# Even with unique flower-superlink ports, Ray's own GCS / dashboard /
# object-manager / metrics-agent default ports cause the second simulation
# on the same node to silently fail (the wrapper script reports exit 0
# because flwr run returns 0, but no rounds actually train — see audit
# v5 same-node and 6598090/91 same-alvis6-07 collision). Set per-job
# Ray ports out of the way of any other Alvis job and our own concurrent
# jobs.
RAY_BASE=$((10000 + (JID % 1000) * 10))
export RAY_GCS_SERVER_PORT=$((RAY_BASE + 1))
export RAY_DASHBOARD_PORT=$((RAY_BASE + 2))
export RAY_NODE_MANAGER_PORT=$((RAY_BASE + 3))
export RAY_OBJECT_MANAGER_PORT=$((RAY_BASE + 4))
export RAY_RUNTIME_ENV_AGENT_PORT=$((RAY_BASE + 5))
echo "Ray ports: GCS=$RAY_GCS_SERVER_PORT dashboard=$RAY_DASHBOARD_PORT node-mgr=$RAY_NODE_MANAGER_PORT object-mgr=$RAY_OBJECT_MANAGER_PORT runtime-env=$RAY_RUNTIME_ENV_AGENT_PORT"

# --- Parse experiment YAML (passed via EXPERIMENT_YAML env var) ---
RUN_CONFIG_FROM_YAML=""
TRAINABLE_LAYERS="full_ft"  # default
if [[ -n "${EXPERIMENT_YAML:-}" && -f "$EXPERIMENT_YAML" ]]; then
    RUN_CONFIG_FROM_YAML=$(awk '
      /^[[:space:]]*#/ { next }
      /^[[:space:]]*$/ { next }
      /^[^:]+:/ {
        key = $0; sub(/:.*/, "", key);   gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
        val = $0; sub(/^[^:]+:[[:space:]]*/, "", val); gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
        if (val ~ /^".*"$/ || val ~ /^'\''.*'\''$/) { val = substr(val, 2, length(val)-2) }
        if (val ~ /^-?[0-9]*\.?[0-9]+$/) { printf "%s=%s ", key, val }
        else { printf "%s='\''%s'\'' ", key, val }
      }
    ' "$EXPERIMENT_YAML")
    # Extract trainable-layers value for GPU-efficiency tuning below.
    # The key is OPTIONAL — Cycle 01 YAMLs and the Phase 3.0 sentinel
    # don't set it. Without `|| true` the pipeline returns grep's exit-1
    # under `set -o pipefail`, which under `set -e` kills the whole
    # SLURM script silently right after the Ray-ports announcement (no
    # error, no train, exit 1). That bug took out sentinel job 6600187
    # before the fix.
    TL_RAW=$(grep -E "^[[:space:]]*trainable-layers:" "$EXPERIMENT_YAML" 2>/dev/null \
             | sed -E "s/^[[:space:]]*trainable-layers:[[:space:]]*['\"]?//" \
             | sed -E "s/['\"]?[[:space:]]*$//" \
             | head -1 || true)
    if [[ -n "$TL_RAW" ]]; then TRAINABLE_LAYERS="$TL_RAW"; fi
fi

# --- Adjust GPU allocation per supernode based on trainable-layers ---
# Default flwr_config.toml uses 0.10 GPU/supernode = 5 GPU instances on
# 50 supernodes, sized for full fine-tuning (~11M trainable params). For
# lightweight modes the per-supernode workload is much smaller, so the
# GPU sits idle and Alvis flags inefficient utilization. Override:
#   head_only  (22K params): 0.025 / supernode → 1.25 GPU equivalents
#   last_block (8.4M params): 0.05  / supernode → 2.5 GPU equivalents
#   full_ft / others:        0.10  (default; matches flwr_config.toml)
case "$TRAINABLE_LAYERS" in
    head_only)  NUM_GPUS_PER_SUPERNODE="0.025" ;;
    last_block) NUM_GPUS_PER_SUPERNODE="0.05"  ;;
    *)          NUM_GPUS_PER_SUPERNODE=""      ;;  # keep default
esac
if [[ -n "$NUM_GPUS_PER_SUPERNODE" ]]; then
    sed -i \
        "s/^options\.backend\.client-resources\.num-gpus = .*/options.backend.client-resources.num-gpus = $NUM_GPUS_PER_SUPERNODE/" \
        "$JOB_FLWR_HOME/config.toml"
    echo "GPU efficiency override: num-gpus = $NUM_GPUS_PER_SUPERNODE per supernode (trainable-layers=$TRAINABLE_LAYERS)"
else
    echo "GPU allocation: default (trainable-layers=$TRAINABLE_LAYERS)"
fi

echo "===== Environment ====="
echo "FLWR_HOME: $FLWR_HOME"
which python
python --version
which flwr
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Wandb env (auth + mode flow in via --export=ALL from submit_experiment.sh).
# WANDB_API_KEY may be in ~/.netrc instead — that's fine, wandb reads both.
echo "===== Wandb ====="
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY: set (len=${#WANDB_API_KEY})"
elif [[ -f "$HOME/.netrc" ]] && grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WANDB_API_KEY: not in env — using ~/.netrc"
else
    echo "WANDB_API_KEY: NOT FOUND (run 'wandb login' once on a login node)"
fi
echo "WANDB_MODE: ${WANDB_MODE:-<unset, defaults from YAML>}"

# --- Start SuperLink in background ---
echo "===== Starting SuperLink ====="
flower-superlink \
  --insecure \
  --simulation \
  --isolation subprocess \
  --control-api-address "127.0.0.1:$PORT_CTL" \
  --simulationio-api-address "127.0.0.1:$PORT_SIO" \
  --database "$FLWR_HOME/local-superlink/state.db" \
  --storage-dir "$FLWR_HOME/local-superlink/ffs" \
  --log-file "$FLWR_HOME/local-superlink/superlink.log" &

SUPERLINK_PID=$!

# Wait for SuperLink to be ready
for i in $(seq 1 30); do
    if kill -0 "$SUPERLINK_PID" 2>/dev/null && \
       python -c "import socket, sys; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',int(sys.argv[1]))); s.close()" "$PORT_CTL" 2>/dev/null; then
        echo "SuperLink ready after ${i}s (PID=$SUPERLINK_PID)"
        break
    fi
    if ! kill -0 "$SUPERLINK_PID" 2>/dev/null; then
        echo "ERROR: SuperLink died. Check $FLWR_HOME/local-superlink/superlink.log"
        cat "$FLWR_HOME/local-superlink/superlink.log" 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# --- Run Flower ---
echo "===== Running Flower ====="
if [[ -n "${EXPERIMENT_YAML:-}" ]]; then
    echo "Experiment config: $EXPERIMENT_YAML"
fi
if [[ -n "$RUN_CONFIG_FROM_YAML" ]]; then
    echo "Run config (YAML): $RUN_CONFIG_FROM_YAML"
    flwr run . local-simulation-gpu --stream --run-config "$RUN_CONFIG_FROM_YAML"
else
    flwr run . local-simulation-gpu --stream "$@"
fi
FLWR_EXIT=$?

# --- Cleanup ---
echo "===== Cleanup ====="
kill "$SUPERLINK_PID" 2>/dev/null || true
wait "$SUPERLINK_PID" 2>/dev/null || true
rm -rf "$JOB_FLWR_HOME"

echo "End: $(date)"
exit $FLWR_EXIT
