#!/bin/bash
module purge
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source /mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/.venv/bin/activate

python - <<'PY'
import sys
import torch
import torchvision
import numpy
import flwr

print("python:", sys.executable)
print("torch:", torch.__version__, torch.__file__)
print("torchvision:", torchvision.__version__, torchvision.__file__)
print("numpy:", numpy.__version__, numpy.__file__)
print("flwr:", flwr.__version__, flwr.__file__)
print("cuda available:", torch.cuda.is_available())
PY