#!/bin/bash
# Build the fl_v3 Alvis x86 venv from the portable manifest (T0).
#
# Strategy: torch / numpy / scipy come from the PyTorch module (CUDA-matched);
# the venv is created with --system-site-packages so it inherits them; the
# pure-PyTorch extras (flwr, ray, scikit-learn, nuscenes-devkit, matplotlib,
# wandb, pytest) are pip-installed on top. NO mmdet3d / mmcv / spconv.
#
# On a fresh checkout / on Arrhenius (ARM): re-run this script. It is the
# canonical, reproducible build. See fl_v3/docs/env.md for the ARM note.
set -euo pipefail

PROJ_ROOT="/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project"
# fl_v3 source lives wherever this script is (worktree-relative), resolved abs.
FL_V3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PROJ_ROOT}/.venv_v3"
PY_MODULE="PyTorch/2.7.1-foss-2024a-CUDA-12.6.0"
# Pinned torch.hub cache for the ImageNet camera-backbone weights (T2/D1). The weights
# are pre-cached HERE on the LOGIN node at build time because Alvis COMPUTE nodes are
# offline; run_in_venv.sh / run_alvis.sh export the same TORCH_HOME so training reads
# them. (Lives on /cephyr, not under the read-only dataset.)
export TORCH_HOME="/cephyr/users/gaohui/Alvis/.cache/torch"

echo "[build_venv] fl_v3 dir : ${FL_V3_DIR}"
echo "[build_venv] venv      : ${VENV}"
echo "[build_venv] module    : ${PY_MODULE}"

# Make `module` available even in a non-interactive shell.
if ! type module >/dev/null 2>&1; then
    if [ -f /usr/share/lmod/lmod/init/bash ]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
fi

module purge
module load "${PY_MODULE}"

echo "[build_venv] base python: $(which python3)"
python3 --version

# Fresh venv (throwaway by design).
rm -rf "${VENV}"
python3 -m venv --system-site-packages "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python -m pip install --upgrade "pip"

# Install the curated direct pins UNDER constraints.txt (which pins numpy/scipy
# to the module versions so pip never shadows the CUDA-matched build).
pip install --no-cache-dir -c "${FL_V3_DIR}/constraints.txt" \
    -r "${FL_V3_DIR}/requirements.txt"

# nuscenes-devkit: install --no-deps to skip its matplotlib<3.6 / descartes cap
# (its real runtime deps are already in requirements.txt above). Still honour
# the numpy/scipy constraints for safety.
pip install --no-cache-dir --no-deps -c "${FL_V3_DIR}/constraints.txt" \
    "nuscenes-devkit==1.1.11"

# torchvision (T2 camera backbone): --no-deps for the SAME reason as nuscenes-devkit —
# its metadata pins torch==2.7.1 + numpy, which would shadow the CUDA/numpy-matched
# module build. 0.22.1 is the exact pairing for torch 2.7.1 (cp312 manylinux x86_64 +
# aarch64 wheels → Arrhenius-portable).
pip install --no-cache-dir --no-deps -c "${FL_V3_DIR}/constraints.txt" \
    "torchvision==0.22.1"

# fl_v3 itself, --no-deps so pip never reinstalls torch/numpy/scipy.
pip install --no-cache-dir -e "${FL_V3_DIR}" --no-deps

# Pre-cache the ImageNet camera-backbone weights on the LOGIN node (compute nodes are
# offline). torch.hub downloads on first instantiation; do it here so a compute job
# never tries to (and fails). Cached under TORCH_HOME (exported above + in run scripts).
echo "[build_venv] pre-caching Swin-T + ResNet-18 ImageNet weights to ${TORCH_HOME} ..."
python - <<'PY'
import os
from torchvision.models import swin_t, Swin_T_Weights, resnet18, ResNet18_Weights
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
n1 = sum(p.numel() for p in swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).parameters())
n2 = sum(p.numel() for p in resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).parameters())
assert n1 == 28288354, n1
assert n2 == 11689512, n2
print("[build_venv] backbone weights cached (swin_t 28,288,354 / resnet18 11,689,512)")
PY

# Capture the full transitive closure for byte-reproducible rebuilds.
pip freeze > "${FL_V3_DIR}/requirements.lock.txt"
echo "[build_venv] wrote lockfile: ${FL_V3_DIR}/requirements.lock.txt"

# Import sanity — the packages T0..T4 actually rely on.
python - <<'PY'
import sys
print("python:", sys.executable)
import torch, numpy, scipy
print("torch:", torch.__version__, "| cuda avail:", torch.cuda.is_available())
print("numpy:", numpy.__version__, "| scipy:", scipy.__version__)
# Guard: the module numpy must NOT have been shadowed by a pip upgrade.
assert numpy.__version__ == "1.26.4", (
    f"numpy shadowed to {numpy.__version__}; constraints.txt failed to hold "
    "the module build (CUDA + determinism hazard)."
)
import flwr; print("flwr:", flwr.__version__)
import ray; print("ray:", ray.__version__)
import sklearn; print("sklearn:", sklearn.__version__)
from sklearn.cluster import HDBSCAN; print("HDBSCAN: OK (sklearn)")
import matplotlib; print("matplotlib:", matplotlib.__version__)
# torchvision (T2 camera backbone) + the pre-cached ImageNet weights must be present.
import torchvision; print("torchvision:", torchvision.__version__)
import os
_th = os.environ.get("TORCH_HOME", "")
for _w in ("swin_t-704ceda3.pth", "resnet18-f37072fd.pth"):
    _p = os.path.join(_th, "hub", "checkpoints", _w)
    assert os.path.isfile(_p), f"backbone weight not pre-cached: {_p} (TORCH_HOME={_th!r})"
print("torchvision backbone weights pre-cached: OK")
# nuScenes APIs T1/T4 depend on (must NOT require descartes/map rendering).
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
print("nuscenes-devkit: data + DetectionEval APIs import OK (no descartes)")
import fl_v3; print("fl_v3:", fl_v3.__version__)
print("[build_venv] IMPORT SANITY PASSED")
PY

echo "[build_venv] DONE"
