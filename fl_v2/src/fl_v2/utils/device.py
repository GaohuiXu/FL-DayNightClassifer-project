from __future__ import annotations

import torch


def get_device(run_config) -> torch.device:
    """Select execution device from run config, with safe fallback."""
    requested = str(run_config.get("device", "cpu")).lower()

    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
