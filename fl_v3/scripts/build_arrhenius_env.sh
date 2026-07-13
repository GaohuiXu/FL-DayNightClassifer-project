#!/bin/bash
# Build the fl_v3 Arrhenius GH200/aarch64 environment stack.
#
# Run through Slurm on a GH200 node:
#   sbatch fl_v3/scripts/run_arrhenius_env_build.sh
#
# This intentionally uses the Phase0A result instead of re-deciding feasibility:
# cumm v0.7.13 + spconv v2.3.8 source build, torch 2.11.0+cu128, CUDA 12.8
# wheel runtime, CUDA 12.9.1 compiler module, sm_90 only.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${REPO}/fl_v3/scripts/arrhenius_env.sh"

arrhenius_load_modules build

echo "[build_arrhenius_env] host=$(hostname) arch=$(uname -m)"
echo "[build_arrhenius_env] repo=${REPO}"
echo "[build_arrhenius_env] env=${ARRHENIUS_VENV}"
echo "[build_arrhenius_env] root=${ARRHENIUS_ENV_ROOT}"
echo "[build_arrhenius_env] CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS}"
echo "[build_arrhenius_env] PIP_CACHE_DIR=${PIP_CACHE_DIR}"

if [ "$(uname -m)" != "aarch64" ]; then
  echo "[build_arrhenius_env] ERROR: must run on an aarch64 GH200 compute node, not $(uname -m)" >&2
  exit 2
fi

if [ "${RECREATE:-0}" = "1" ]; then
  rm -rf "${ARRHENIUS_VENV}"
fi

if [ ! -x "${ARRHENIUS_VENV}/bin/python" ]; then
  mamba create -y -p "${ARRHENIUS_VENV}" \
    python=3.11 pip "setuptools<82" wheel packaging ninja cmake \
    "numpy==1.26.4" "scipy==1.13.1"
fi

export CONDA_PREFIX="${ARRHENIUS_VENV}"
export PATH="${ARRHENIUS_VENV}/bin:${PATH}"
hash -r
python -m pip install --upgrade "pip"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUMM_CUDA_ARCH_LIST="${CUMM_CUDA_ARCH_LIST:-9.0}"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"

GCC_LIBSTDCPP="$(gcc -print-file-name=libstdc++.so.6)"
export LD_LIBRARY_PATH="$(dirname "${GCC_LIBSTDCPP}"):${ARRHENIUS_VENV}/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${GCC_LIBSTDCPP}${LD_PRELOAD:+:${LD_PRELOAD}}"

echo "[build_arrhenius_env] python=$(python -V)"
echo "[build_arrhenius_env] nvcc=$(command -v nvcc)"
nvcc --version | sed -n '1,4p'
echo "[build_arrhenius_env] gcc=$(command -v gcc)"
gcc --version | sed -n '1p'
echo "[build_arrhenius_env] LD_PRELOAD=${LD_PRELOAD}"

python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.11.0+cu128"

# Torchvision must match torch 2.11. Let the PyTorch cu128 index resolve the
# aarch64 wheel, then freeze the resolved version below. --no-deps prevents a
# second torch/numpy resolver from shadowing the already installed stack.
python -m pip install --index-url https://download.pytorch.org/whl/cu128 --no-deps \
  "torchvision"
python -m pip install "Pillow>=10.0"

python -m pip install -c "${REPO}/fl_v3/constraints.txt" \
  -r "${REPO}/fl_v3/requirements.txt"
python -m pip install --no-deps -c "${REPO}/fl_v3/constraints.txt" \
  "nuscenes-devkit==1.1.11"
python -m pip install --no-deps -e "${REPO}/fl_v3"

python -m pip install -c "${REPO}/fl_v3/constraints.txt" \
  "pccm==0.4.16" "ccimport==0.4.4" "pybind11>=2.6" "fire==0.7.1"

CUMM_SRC="${ARRHENIUS_SRC_ROOT}/cumm"
SPCONV_SRC="${ARRHENIUS_SRC_ROOT}/spconv"
PHASE0A_SRC="${ARRHENIUS_BASE}/phase0a_spconv/src"
if [ ! -d "${CUMM_SRC}/.git" ]; then
  if [ -d "${PHASE0A_SRC}/cumm/.git" ]; then
    git clone "${PHASE0A_SRC}/cumm" "${CUMM_SRC}"
  else
    git clone https://github.com/FindDefinition/cumm.git "${CUMM_SRC}"
  fi
fi
if [ ! -d "${SPCONV_SRC}/.git" ]; then
  if [ -d "${PHASE0A_SRC}/spconv/.git" ]; then
    git clone "${PHASE0A_SRC}/spconv" "${SPCONV_SRC}"
  else
    git clone https://github.com/traveller59/spconv.git "${SPCONV_SRC}"
  fi
fi

git -C "${CUMM_SRC}" fetch --tags --quiet || true
git -C "${CUMM_SRC}" checkout -q v0.7.13
git -C "${SPCONV_SRC}" fetch --tags --quiet || true
git -C "${SPCONV_SRC}" checkout -q v2.3.8

# spconv v2.3.8 asks build isolation to download cumm>=0.7.11, which has no
# aarch64 PyPI wheel. Use the local editable cumm, same as Phase0A.
python -m pip uninstall -y spconv spconv-cu120 spconv-cu124 spconv-cu126 spconv-cu128 cumm cumm-cu120 cumm-cu124 cumm-cu126 cumm-cu128 || true
python -m pip install --no-build-isolation --no-deps -e "${CUMM_SRC}"
export SPCONV_SRC
python - <<'PY'
from pathlib import Path
p = Path(__import__("os").environ["SPCONV_SRC"]) / "pyproject.toml"
s = p.read_text()
s = s.replace(
    'requires = ["setuptools>=41.0", "wheel", "pccm>=0.4.16", "cumm>=0.7.11"]',
    'requires = ["setuptools>=41.0", "wheel", "pccm>=0.4.16"]',
)
p.write_text(s)
PY
python -m pip install --no-build-isolation --no-deps -e "${SPCONV_SRC}"

# Trigger cumm/spconv native builds and fail here if libstdc++ or sm_90 setup is wrong.
python - <<'PY'
import os
import platform
import torch
print("platform.machine:", platform.machine())
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0), "capability:", torch.cuda.get_device_capability(0))
import cumm
import spconv.pytorch as spconv
print("cumm:", getattr(cumm, "__version__", "?"), cumm.__file__)
print("spconv:", spconv)
PY

# Pre-cache camera backbone weights under /nobackup. This is architecture-neutral
# data, but doing it through the target env also verifies torchvision import.
python - <<'PY'
import os
from torchvision.models import resnet18, ResNet18_Weights, swin_t, Swin_T_Weights
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
print("resnet18 params:", sum(p.numel() for p in resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).parameters()))
print("swin_t params:", sum(p.numel() for p in swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).parameters()))
PY

python -m pip freeze > "${ARRHENIUS_ENV_ROOT}/requirements.arrhenius.lock.txt"
python - <<'PY'
import importlib
mods = ["torch", "torchvision", "numpy", "scipy", "flwr", "ray", "sklearn", "matplotlib", "nuscenes", "spconv", "cumm", "fl_v3"]
for name in mods:
    m = importlib.import_module(name)
    print(f"{name}: {getattr(m, '__version__', 'import-ok')}")
PY

echo "[build_arrhenius_env] wrote ${ARRHENIUS_ENV_ROOT}/requirements.arrhenius.lock.txt"
echo "[build_arrhenius_env] DONE"
