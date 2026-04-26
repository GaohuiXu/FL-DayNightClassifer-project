#!/bin/bash
# Activate FL v2 / GTSRB environment on Alvis

module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0

source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate

echo "[INFO] Environment activated"
echo "[INFO] python: $(which python)"
python --version

python - <<'PY'
import torch
print("[INFO] torch:", torch.__version__)
print("[INFO] cuda available:", torch.cuda.is_available())
try:
    import torchvision
    print("[INFO] torchvision:", torchvision.__version__)
except Exception as e:
    print("[WARN] torchvision import failed:", repr(e))
PY