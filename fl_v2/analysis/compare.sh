#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Compare FL experiment results across defenses.
#
# Usage:
#   ./compare.sh                          # compare all experiments
#   ./compare.sh --exp-dirs /path/a /path/b
#   ./compare.sh --figures comparison bars table
#   ./compare.sh --smooth 5
#
# Assumes activate_env.sh has already been sourced.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

python -m analysis.plot_comparison "$@"
