#!/usr/bin/env bash
#SBATCH -A NAISS2025-22-1113
#SBATCH -p alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 0-02:00:00
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

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

cd /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/fl_v2
source /cephyr/users/gaohui/Alvis/thesis_workspace/fl_weather_project/.venv/bin/activate

# Per-job FLWR_HOME — isolates concurrent jobs from each other
JOB_FLWR_HOME="/tmp/flwr_${SLURM_JOB_ID}"
mkdir -p "$JOB_FLWR_HOME/local-superlink/ffs"
cp /cephyr/users/gaohui/Alvis/.flwr/config.toml "$JOB_FLWR_HOME/config.toml"
export FLWR_HOME="$JOB_FLWR_HOME"

export RAY_DEDUP_LOGS=0

# --- Parse experiment YAML (passed via EXPERIMENT_YAML env var) ---
RUN_CONFIG_FROM_YAML=""
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
fi

echo "===== Environment ====="
echo "FLWR_HOME: $FLWR_HOME"
which python
python --version
which flwr
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Start SuperLink in background (same as start_superlink.sh) ---
echo "===== Starting SuperLink ====="
flower-superlink \
  --insecure \
  --simulation \
  --isolation subprocess \
  --control-api-address 127.0.0.1:39093 \
  --simulationio-api-address 127.0.0.1:39094 \
  --database "$FLWR_HOME/local-superlink/state.db" \
  --storage-dir "$FLWR_HOME/local-superlink/ffs" \
  --log-file "$FLWR_HOME/local-superlink/superlink.log" &

SUPERLINK_PID=$!

# Wait for SuperLink to be ready
for i in $(seq 1 30); do
    if kill -0 "$SUPERLINK_PID" 2>/dev/null && \
       python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',39093)); s.close()" 2>/dev/null; then
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
