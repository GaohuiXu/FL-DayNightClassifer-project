from __future__ import annotations

import json
import os
from typing import Any


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, filepath: str) -> None:
    """Save a Python object as JSON."""
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)