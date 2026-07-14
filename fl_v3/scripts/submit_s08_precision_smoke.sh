#!/bin/bash
# Exact one-shot review-remediation submission prepared by S08 RUN_REQUEST.md.
# Do not run without separate owner approval bound to S08-SMOKE-5 and this tuple.
set -euo pipefail

SNAPSHOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke5_51daec3e860e
OUTPUT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke5_51daec3e860e
MINI_DATAROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini

test -d "${SNAPSHOT}"
test -d "${MINI_DATAROOT}"
test ! -e "${OUTPUT}"
install -d -m 0700 "${OUTPUT}"

sbatch --parsable \
  --account=naiss2025-22-1113-gpu \
  --partition=gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=nvidia_gh200_120gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=00:30:00 \
  --no-requeue \
  --job-name=flv3_s08_smoke5 \
  --output="${OUTPUT}/slurm-%j.out" \
  --error="${OUTPUT}/slurm-%j.err" \
  --export=S08_SNAPSHOT="${SNAPSHOT}",S08_OUTPUT="${OUTPUT}",S08_MINI_DATAROOT="${MINI_DATAROOT}" \
  <<'SBATCH'
#!/bin/bash
set -euo pipefail
exec "${S08_SNAPSHOT}/fl_v3/scripts/run_s08_precision_smoke.sh"
SBATCH
