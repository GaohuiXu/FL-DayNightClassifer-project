#!/bin/bash
# Run a command inside the fl_v3 venv (module + activate), from anywhere.
# Usage: bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests -q
#
# The venv is a gitignored build artifact at <project_root>/.venv_v3 — a fresh
# checkout has none. This script FAILS EARLY with a clear pointer to
# build_venv.sh if the venv is missing or incomplete, so the T0 gate is
# reproducible (no silent "subset of tests ran" outcomes).
set -euo pipefail
PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
VENV="${PROJ_ROOT}/.venv_v3"

if [ ! -x "${VENV}/bin/python" ]; then
    echo "[run_in_venv] ERROR: venv not found at ${VENV}" >&2
    echo "[run_in_venv] Build it first:  bash fl_v3/scripts/build_venv.sh" >&2
    exit 2
fi

if ! type module >/dev/null 2>&1; then
    [ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash
fi
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Preflight: the gate needs torch, flwr, sklearn.HDBSCAN, and fl_v3 importable.
# A missing one silently shrinks the test set (the T0 review hit exactly this),
# so verify up front and fail loudly with the fix.
python - <<'PY' || { echo "[run_in_venv] -> rebuild with: bash fl_v3/scripts/build_venv.sh" >&2; exit 2; }
import importlib, sys
missing = []
for mod in ("torch", "numpy", "flwr", "sklearn", "fl_v3"):
    try:
        importlib.import_module(mod)
    except Exception as e:
        missing.append(f"{mod} ({type(e).__name__})")
try:
    from sklearn.cluster import HDBSCAN  # noqa: F401
except Exception as e:
    missing.append(f"sklearn.cluster.HDBSCAN ({type(e).__name__})")
if missing:
    print("[run_in_venv] venv incomplete — missing:", ", ".join(missing), file=sys.stderr)
    sys.exit(1)
PY

exec "$@"
