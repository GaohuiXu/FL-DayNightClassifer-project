from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn

# 提取和设置模型参数；用于client的上传和server的发放

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load model parameters from a list of NumPy arrays."""
    state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    for (key, old_tensor), new_param in zip(state_dict.items(), parameters):
        tensor = torch.tensor(new_param, dtype=old_tensor.dtype)
        new_state_dict[key] = tensor

    model.load_state_dict(new_state_dict, strict=True)