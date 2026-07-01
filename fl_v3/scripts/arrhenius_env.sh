#!/bin/bash
# Shared Arrhenius GH200 environment bootstrap for fl_v3.
#
# All mutable artifacts live under /nobackup, not $HOME:
#   env/cache/build/tmp/output/logs/source clones.
#
# Source this file from Slurm jobs, then call:
#   arrhenius_load_modules build   # compile/source-build path
#   arrhenius_load_modules run     # runtime path; falls back to build modules if needed
#   arrhenius_activate_env
set -euo pipefail

export ARRHENIUS_BASE="${ARRHENIUS_BASE:-/nobackup/proj/disk/naiss2024-22-991/personal/gaohui}"
export ARRHENIUS_ENV_ROOT="${ARRHENIUS_ENV_ROOT:-${ARRHENIUS_BASE}/arrhenius_fl_v3}"
export ARRHENIUS_VENV="${ARRHENIUS_VENV:-${ARRHENIUS_ENV_ROOT}/envs/pt311-cu128-spconv}"
export ARRHENIUS_SRC_ROOT="${ARRHENIUS_SRC_ROOT:-${ARRHENIUS_ENV_ROOT}/src}"
export ARRHENIUS_LOG_ROOT="${ARRHENIUS_LOG_ROOT:-${ARRHENIUS_ENV_ROOT}/logs}"
export ARRHENIUS_OUTPUT_ROOT="${ARRHENIUS_OUTPUT_ROOT:-${ARRHENIUS_ENV_ROOT}/outputs}"

export CONDA_PKGS_DIRS="${ARRHENIUS_ENV_ROOT}/conda_pkgs"
export MAMBA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
export PIP_CACHE_DIR="${ARRHENIUS_ENV_ROOT}/pip_cache"
export XDG_CACHE_HOME="${ARRHENIUS_ENV_ROOT}/xdg_cache"
export TMPDIR="${ARRHENIUS_ENV_ROOT}/tmp"
export TORCH_HOME="${ARRHENIUS_ENV_ROOT}/torch_home"
export TORCHINDUCTOR_CACHE_DIR="${ARRHENIUS_ENV_ROOT}/torchinductor_cache"
export CCACHE_DIR="${ARRHENIUS_ENV_ROOT}/ccache"
export SPCONV_DEBUG_SAVE_PATH="${ARRHENIUS_ENV_ROOT}/spconv_debug"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export CUMM_CUDA_ARCH_LIST="${CUMM_CUDA_ARCH_LIST:-9.0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p \
  "${ARRHENIUS_ENV_ROOT}" "${ARRHENIUS_SRC_ROOT}" "${ARRHENIUS_LOG_ROOT}" \
  "${ARRHENIUS_OUTPUT_ROOT}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" \
  "${XDG_CACHE_HOME}" "${TMPDIR}" "${TORCH_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CCACHE_DIR}" "${SPCONV_DEBUG_SAVE_PATH}"

arrhenius_init_lmod() {
  if ! type module >/dev/null 2>&1; then
    if [ -f /usr/share/lmod/lmod/init/bash ]; then
      # shellcheck disable=SC1091
      source /usr/share/lmod/lmod/init/bash
    fi
  fi
}

arrhenius_try_module_load() {
  local mod
  for mod in "$@"; do
    module load "${mod}" >/dev/null 2>&1 || return 1
  done
}

arrhenius_load_modules() {
  local mode="${1:-run}"
  arrhenius_init_lmod
  module purge

  if [ "${mode}" = "build" ]; then
    arrhenius_try_module_load GPU/buildenv-nvhpc/25.9-cu12.9.1-eb GPU/Miniforge/26.3.2-2-eb \
      || arrhenius_try_module_load buildenv/default-CUDA-12.9.1 Miniforge/26.3.2-2-eb \
      || arrhenius_try_module_load GPU/buildenv-nvhpc/recommendation GPU/Miniforge/recommendation
  else
    arrhenius_try_module_load GPU/Miniforge/26.3.2-2-eb \
      || arrhenius_try_module_load Miniforge/26.3.2-2-eb \
      || arrhenius_try_module_load GPU/Miniforge/recommendation
  fi
  # The Miniforge module can set CONDA_PKGS_DIRS to ~/.conda-hpc. Force it back
  # under /nobackup after every module load.
  export CONDA_PKGS_DIRS="${ARRHENIUS_ENV_ROOT}/conda_pkgs"
  export MAMBA_PKGS_DIRS="${CONDA_PKGS_DIRS}"
  export PIP_CACHE_DIR="${ARRHENIUS_ENV_ROOT}/pip_cache"
  export XDG_CACHE_HOME="${ARRHENIUS_ENV_ROOT}/xdg_cache"
  export TMPDIR="${ARRHENIUS_ENV_ROOT}/tmp"
  export TORCH_HOME="${ARRHENIUS_ENV_ROOT}/torch_home"
  export TORCHINDUCTOR_CACHE_DIR="${ARRHENIUS_ENV_ROOT}/torchinductor_cache"
  export CCACHE_DIR="${ARRHENIUS_ENV_ROOT}/ccache"
  export SPCONV_DEBUG_SAVE_PATH="${ARRHENIUS_ENV_ROOT}/spconv_debug"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  export CUMM_CUDA_ARCH_LIST="${CUMM_CUDA_ARCH_LIST:-9.0}"
  export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
}

arrhenius_activate_env() {
  if [ ! -x "${ARRHENIUS_VENV}/bin/python" ]; then
    echo "[arrhenius_env] ERROR: env missing: ${ARRHENIUS_VENV}" >&2
    echo "[arrhenius_env] Build it with: sbatch fl_v3/scripts/run_arrhenius_env_build.sh" >&2
    return 2
  fi
  export CONDA_PREFIX="${ARRHENIUS_VENV}"
  export PATH="${ARRHENIUS_VENV}/bin:${PATH}"
  hash -r

  # cumm/spconv need a newer libstdc++ than /lib64. Runtime jobs prefer the
  # conda env copy; build jobs may opt into the EasyBuild GCC copy.
  local libstdcpp="${ARRHENIUS_VENV}/lib/libstdc++.so.6"
  if [ "${ARRHENIUS_USE_GCC_LIBSTDCXX:-0}" = "1" ] && command -v gcc >/dev/null 2>&1; then
    local gcc_libstdcpp
    gcc_libstdcpp="$(gcc -print-file-name=libstdc++.so.6 || true)"
    if [ -n "${gcc_libstdcpp}" ] && [ -f "${gcc_libstdcpp}" ]; then
      libstdcpp="${gcc_libstdcpp}"
    fi
  fi
  if [ -f "${libstdcpp}" ]; then
    export LD_LIBRARY_PATH="$(dirname "${libstdcpp}"):${ARRHENIUS_VENV}/lib:${LD_LIBRARY_PATH:-}"
    export LD_PRELOAD="${libstdcpp}${LD_PRELOAD:+:${LD_PRELOAD}}"
  else
    export LD_LIBRARY_PATH="${ARRHENIUS_VENV}/lib:${LD_LIBRARY_PATH:-}"
  fi

  local repo_src
  repo_src="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
  export PYTHONPATH="${repo_src}${PYTHONPATH:+:${PYTHONPATH}}"
}
