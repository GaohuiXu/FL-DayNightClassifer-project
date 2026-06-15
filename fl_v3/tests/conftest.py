"""Shared pytest fixtures — loads the fl_v2 oracle snapshot for parity tests."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

_FIX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_INPUTS = os.path.join(_FIX_DIR, "oracle_inputs.npz")
_ARRAYS = os.path.join(_FIX_DIR, "oracle_arrays.npz")
_DECISIONS = os.path.join(_FIX_DIR, "oracle_decisions.json")


@pytest.fixture(scope="session")
def oracle():
    """The committed fl_v2 oracle fixture (inputs + decisions + aggregated arrays).

    Skips the dependent test if fixtures are absent (regenerate with
    ``python fl_v3/tests/fixtures/make_oracle_fixtures.py``).
    """
    if not (os.path.exists(_INPUTS) and os.path.exists(_ARRAYS) and os.path.exists(_DECISIONS)):
        pytest.skip("oracle fixtures missing; run make_oracle_fixtures.py")

    inp = np.load(_INPUTS)
    nt = int(inp["num_tensors"][0])
    nc = int(inp["num_clients"][0])
    global_params = [inp[f"g_{i}"] for i in range(nt)]
    clients = [[inp[f"c_{j}_{i}"] for i in range(nt)] for j in range(nc)]
    partition_ids = inp["partition_ids"].tolist()
    is_malicious = inp["is_malicious"].tolist()
    targets = inp["targets"]

    arrs = np.load(_ARRAYS)
    flame_new_global = [arrs[f"flame_{i}"] for i in range(nt)]
    foolsgold_new_global = [arrs[f"foolsgold_{i}"] for i in range(nt)]
    median_new_global = [arrs[f"median_{i}"] for i in range(nt)]
    clipped = [[arrs[f"clipped_{j}_{i}"] for i in range(nt)] for j in range(nc)]

    with open(_DECISIONS, encoding="utf-8") as f:
        dec = json.load(f)

    return {
        "num_tensors": nt,
        "num_clients": nc,
        "global_params": global_params,
        "clients": clients,
        "partition_ids": partition_ids,
        "is_malicious": is_malicious,
        "targets": targets,
        "flame_new_global": flame_new_global,
        "foolsgold_new_global": foolsgold_new_global,
        "median_new_global": median_new_global,
        "clipped": clipped,
        "decisions": dec,
    }
