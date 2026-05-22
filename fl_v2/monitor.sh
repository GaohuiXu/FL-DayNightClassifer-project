#!/usr/bin/env bash
# Usage:
#   ./monitor.sh <job_id>          — show status + tail latest output
#   ./monitor.sh <job_id> follow   — live-follow the output log
#   ./monitor.sh                   — list all your running/pending jobs

SLURM_DIR="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/slurm"

if [[ -z "${1:-}" ]]; then
    echo "===== Your SLURM jobs ====="
    squeue -u "$USER" -o "%.10i %.34j %.9T %.10M %.6D %R"
    exit 0
fi

JOB_ID="$1"
MODE="${2:-status}"

echo "===== Job ${JOB_ID} ====="
squeue -j "$JOB_ID" -o "%.10i %.34j %.9T %.10M %.6D %R" 2>/dev/null || true

STATE=$(sacct -j "$JOB_ID" --format=State --noheader -X 2>/dev/null | head -1 | tr -d ' ')
if [[ -n "$STATE" ]]; then
    echo "State: $STATE"
    sacct -j "$JOB_ID" --format=JobID,JobName%34,State,Elapsed,ExitCode,NodeList --noheader -X 2>/dev/null
fi
echo ""

# run_alvis.sh routes cycle-aware logs to slurm/<cycle>/<jobname>_<jid>.{out,err}
# (SBATCH %x_%j); legacy logs sit directly in slurm/. Resolve by job-id glob
# so any job name (cycle02-..., flwr_gtsrb, ...) is found.
OUT_FILE=$(ls -t "${SLURM_DIR}"/*/*_"${JOB_ID}".out "${SLURM_DIR}"/*_"${JOB_ID}".out 2>/dev/null | head -1)

if [[ -z "$OUT_FILE" || ! -f "$OUT_FILE" ]]; then
    echo "Output file not yet created (job may be queued)."
    exit 0
fi
ERR_FILE="${OUT_FILE%.out}.err"
echo "Log: $OUT_FILE"
echo ""

if [[ "$MODE" == "follow" ]]; then
    echo "===== Following ${OUT_FILE} (Ctrl+C to stop) ====="
    tail -f "$OUT_FILE"
else
    echo "===== Last 40 lines of output ====="
    tail -n 40 "$OUT_FILE"
    if [[ -f "$ERR_FILE" && -s "$ERR_FILE" ]]; then
        echo ""
        echo "===== Last 10 lines of stderr ====="
        tail -n 10 "$ERR_FILE"
    fi
fi
