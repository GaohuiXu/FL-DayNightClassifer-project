#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

flwr run . local-simulation-gpu --stream