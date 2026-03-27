#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Analyze FL experiment results.
#
# Usage:
#   ./analyze.sh                          # analyze all experiments
#   ./analyze.sh --compare-norms          # also compare norms
#   ./analyze.sh --exp-dir /path/to/exp   # specific experiment
#
# Assumes activate_env.sh has already been sourced.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m analysis.plot_experiment "$@"
