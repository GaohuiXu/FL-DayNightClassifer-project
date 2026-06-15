#!/bin/bash
# Run a command inside the fl_v3 venv (module + activate), from anywhere.
# Usage: bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests -q
set -euo pipefail
PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
if ! type module >/dev/null 2>&1; then
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${PROJ_ROOT}/.venv_v3/bin/activate"
exec "$@"
