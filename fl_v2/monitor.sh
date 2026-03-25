#!/usr/bin/env bash
# Usage:
#   ./monitor.sh <job_id>          — show status + tail latest output
#   ./monitor.sh <job_id> follow   — live-follow the output log
#   ./monitor.sh                   — list all your running/pending jobs

SLURM_OUT_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm"

if [[ -z "${1:-}" ]]; then
    echo "===== Your SLURM jobs ====="
    squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R"
    exit 0
fi

JOB_ID="$1"
MODE="${2:-status}"

OUT_FILE="${SLURM_OUT_DIR}/flwr_gtsrb_${JOB_ID}.out"
ERR_FILE="${SLURM_OUT_DIR}/flwr_gtsrb_${JOB_ID}.err"

echo "===== Job ${JOB_ID} ====="
squeue -j "$JOB_ID" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || true

# If job finished, show accounting info
STATE=$(sacct -j "$JOB_ID" --format=State --noheader -X 2>/dev/null | tr -d ' ')
if [[ -n "$STATE" ]]; then
    echo "State: $STATE"
    sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode --noheader -X
fi

echo ""

if [[ ! -f "$OUT_FILE" ]]; then
    echo "Output file not yet created (job may be queued): $OUT_FILE"
    exit 0
fi

if [[ "$MODE" == "follow" ]]; then
    echo "===== Following ${OUT_FILE} (Ctrl+C to stop) ====="
    tail -f "$OUT_FILE"
else
    echo "===== Last 40 lines of output ====="
    tail -n 40 "$OUT_FILE"

    if [[ -f "$ERR_FILE" ]] && [[ -s "$ERR_FILE" ]]; then
        echo ""
        echo "===== Last 10 lines of stderr ====="
        tail -n 10 "$ERR_FILE"
    fi
fi
