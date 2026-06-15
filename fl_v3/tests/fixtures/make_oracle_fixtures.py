"""Generate the fl_v2 oracle fixtures for the T0 parity tests.

Imports the ACTUAL fl_v2 pure functions (the oracle) and snapshots their
decisions on a fixed, seeded synthetic update matrix, so the parity tests assert
fl_v3 reproduces *real* fl_v2 code — not a re-derivation. Run once (needs fl_v2
on the path + flwr/sklearn/numpy in the env); the outputs are committed:

    oracle_inputs.npz       — global params, client params, targets, partition_ids
    oracle_arrays.npz       — oracle aggregated arrays (FLAME / FoolsGold / median / clipped)
    oracle_decisions.json   — admitted set, trust, norms, s_t, coefs, gsm, partition

Usage (from repo root, fl_v3 venv active):
    python fl_v3/tests/fixtures/make_oracle_fixtures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# --- locate fl_v2 (the oracle) on sys.path ---------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))  # worktree root
_FL_V2_SRC = os.path.join(_REPO, "fl_v2", "src")
if _FL_V2_SRC not in sys.path:
    sys.path.insert(0, _FL_V2_SRC)

OUT_DIR = _THIS

# Fixed knobs the parity tests also read.
FLAME_SEED = 42
FLAME_ROUND = 3
FLAME_NOISE_MULTIPLIER = 1e-6
CLIP_NORM = 5.0
TOPK_K = 16
FOOLSGOLD_HEAD_INDEX = -2
PART_CLIENTS = 8
PART_ALPHA = 0.5
PART_SEED = 7


def _make_synthetic():
    """A 10-client round with a 7-honest majority + 3 colluding malicious.

    Param tensors mimic a tiny net: W1 (8,4), b1 (8,), Whead (3,8) [the -2 head
    slice FoolsGold uses], bhead (3,). Honest updates share a base direction
    with small per-client noise; malicious updates share a distinct, larger
    direction → FLAME clusters out the minority, FoolsGold down-weights it.
    """
    rng = np.random.default_rng(1234)
    shapes = [(8, 4), (8,), (3, 8), (3,)]
    global_params = [rng.standard_normal(s).astype(np.float32) * 0.1 for s in shapes]

    n_honest, n_mal = 7, 3
    base_dirs = [rng.standard_normal(s).astype(np.float32) for s in shapes]
    mal_dirs = [rng.standard_normal(s).astype(np.float32) for s in shapes]

    client_params_list = []
    is_malicious = []
    for _ in range(n_honest):
        cp = []
        for g, bd in zip(global_params, base_dirs):
            upd = 0.05 * bd + 0.01 * rng.standard_normal(g.shape).astype(np.float32)
            cp.append((g + upd).astype(np.float32))
        client_params_list.append(cp)
        is_malicious.append(0)
    for _ in range(n_mal):
        cp = []
        for g, md in zip(global_params, mal_dirs):
            upd = 0.20 * md + 0.005 * rng.standard_normal(g.shape).astype(np.float32)
            cp.append((g + upd).astype(np.float32))
        client_params_list.append(cp)
        is_malicious.append(1)

    partition_ids = list(range(len(client_params_list)))
    # Synthetic label array for the partition parity fixture.
    targets = rng.integers(0, 5, size=200).astype(np.int64)
    return global_params, client_params_list, partition_ids, is_malicious, targets


class _FakeDataset:
    """Minimal stand-in for fl_v2.data.partition._get_targets (uses .targets)."""

    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)


def main():
    # --- import the oracle (fl_v2) ---
    from fl_v2.strategy.flame import FlameFedAvg
    from fl_v2.strategy.foolsgold import FoolsGoldFedAvg
    from fl_v2.strategy.norm_tracking_fedavg import aggregate_weighted_updates
    from fl_v2.attacks_defenses.defenses.norm_clipping import (
        clip_updates_by_l2_norm,
        compute_gradient_space_metrics,
        compute_update_norms,
    )
    # fl_v2.data.__init__ eagerly imports torchvision (GTSRB), which fl_v3's
    # pure-torch venv intentionally lacks. partition.py is numpy-only, so load
    # it directly by file path to get the oracle without the torchvision chain.
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "fl_v2_partition_oracle",
        os.path.join(_FL_V2_SRC, "fl_v2", "data", "partition.py"),
    )
    _fl_v2_partition = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_fl_v2_partition)
    dirichlet_partition = _fl_v2_partition.dirichlet_partition

    g, clients, pids, is_mal, targets = _make_synthetic()
    num_tensors = len(g)

    # ----- gradient-space metrics + norms (oracle) -----
    original_norms = compute_update_norms(g, clients)
    gsm = compute_gradient_space_metrics(g, clients, TOPK_K)

    # ----- FLAME (oracle): cluster -> clip -> noise -> aggregate -----
    flat = np.stack(
        [
            np.concatenate(
                [
                    (np.asarray(c, np.float32) - np.asarray(gg, np.float32)).reshape(-1)
                    for c, gg in zip(cp, g)
                ]
            )
            for cp in clients
        ]
    )
    norms64 = np.asarray(original_norms, dtype=np.float64)
    admitted = FlameFedAvg._cluster_admitted(flat, norms64)
    s_t = float(np.median(norms64))
    coefs = [0.0] * len(clients)
    for i in admitted:
        coefs[i] = min(1.0, s_t / (norms64[i] + 1e-12)) / len(admitted)
    sigma = FLAME_NOISE_MULTIPLIER * s_t
    nrng = np.random.default_rng((FLAME_SEED & 0xFFFFFFFF) * 1_000_003 + FLAME_ROUND)
    noise = [nrng.normal(0.0, sigma, size=np.asarray(x).shape) for x in g]
    keys = [f"p{i}" for i in range(num_tensors)]
    flame_arr = aggregate_weighted_updates(g, clients, keys, coefs, noise=noise)
    flame_new_global = flame_arr.to_numpy_ndarrays()

    # ----- FoolsGold (oracle): single round, head-slice history -----
    fg_updates = [
        (
            np.asarray(cp[FOOLSGOLD_HEAD_INDEX], np.float32)
            - np.asarray(g[FOOLSGOLD_HEAD_INDEX], np.float32)
        ).reshape(-1)
        for cp in clients
    ]
    fg_trust = FoolsGoldFedAvg._foolsgold_weights(fg_updates)
    total = float(sum(fg_trust))
    fg_coefs = (
        [1.0 / len(fg_trust)] * len(fg_trust)
        if total < 1e-12
        else [t / total for t in fg_trust]
    )
    fg_arr = aggregate_weighted_updates(g, clients, keys, fg_coefs)
    fg_new_global = fg_arr.to_numpy_ndarrays()

    # ----- NormClip (oracle) -----
    clipped_params, clip_orig_norms, clip_norms = clip_updates_by_l2_norm(g, clients, CLIP_NORM)

    # ----- FedMedian (oracle arithmetic) -----
    median_new_global = [
        np.asarray(np.median(np.stack([cp[k] for cp in clients]), axis=0), dtype=g[k].dtype)
        for k in range(num_tensors)
    ]

    # ----- Dirichlet partition (oracle) -----
    part = dirichlet_partition(_FakeDataset(targets), PART_CLIENTS, PART_ALPHA, PART_SEED)
    part_json = {str(k): [int(x) for x in v] for k, v in part.items()}

    # --- save inputs ---
    inputs = {f"g_{i}": g[i] for i in range(num_tensors)}
    for j, cp in enumerate(clients):
        for i in range(num_tensors):
            inputs[f"c_{j}_{i}"] = cp[i]
    inputs["partition_ids"] = np.asarray(pids, dtype=np.int64)
    inputs["is_malicious"] = np.asarray(is_mal, dtype=np.int64)
    inputs["targets"] = targets
    inputs["num_tensors"] = np.asarray([num_tensors], dtype=np.int64)
    inputs["num_clients"] = np.asarray([len(clients)], dtype=np.int64)
    np.savez(os.path.join(OUT_DIR, "oracle_inputs.npz"), **inputs)

    # --- save oracle aggregated arrays ---
    arrs = {}
    for i in range(num_tensors):
        arrs[f"flame_{i}"] = flame_new_global[i]
        arrs[f"foolsgold_{i}"] = fg_new_global[i]
        arrs[f"median_{i}"] = median_new_global[i]
        for j, cp in enumerate(clipped_params):
            arrs[f"clipped_{j}_{i}"] = cp[i]
    np.savez(os.path.join(OUT_DIR, "oracle_arrays.npz"), **arrs)

    # --- save scalar/list decisions ---
    decisions = {
        "config": {
            "flame_seed": FLAME_SEED,
            "flame_round": FLAME_ROUND,
            "flame_noise_multiplier": FLAME_NOISE_MULTIPLIER,
            "clip_norm": CLIP_NORM,
            "topk_k": TOPK_K,
            "foolsgold_head_index": FOOLSGOLD_HEAD_INDEX,
            "part_clients": PART_CLIENTS,
            "part_alpha": PART_ALPHA,
            "part_seed": PART_SEED,
        },
        "original_norms": [float(x) for x in original_norms],
        "flame_admitted": [int(i) for i in admitted],
        "flame_s_t": s_t,
        "flame_sigma": float(sigma),
        "flame_coefs": [float(c) for c in coefs],
        "foolsgold_trust": [float(t) for t in fg_trust],
        "foolsgold_coefs": [float(c) for c in fg_coefs],
        "clip_original_norms": [float(x) for x in clip_orig_norms],
        "clip_norms": [float(x) for x in clip_norms],
        "gsm": {
            "cosine_to_mean": [float(x) for x in gsm["cosine_to_mean"]],
            "topk_energy_frac": [float(x) for x in gsm["topk_energy_frac"]],
            "pairwise_cosine": [[float(x) for x in row] for row in gsm["pairwise_cosine"]],
            "topk_energy_k": int(gsm["topk_energy_k"]),
        },
        "dirichlet_partition": part_json,
    }
    with open(os.path.join(OUT_DIR, "oracle_decisions.json"), "w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2)

    print("[fixtures] wrote oracle_inputs.npz, oracle_arrays.npz, oracle_decisions.json")
    print(f"[fixtures] FLAME admitted {admitted} (of {len(clients)}); "
          f"malicious idx {[i for i,m in enumerate(is_mal) if m]}")
    print(f"[fixtures] FoolsGold trust min={min(fg_trust):.4f} max={max(fg_trust):.4f}")


if __name__ == "__main__":
    main()
