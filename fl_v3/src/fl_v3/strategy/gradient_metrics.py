"""Gradient-space metric definitions + L2 clip (fl_v3 T0 carry-over).

Pure-numpy, framework-free. Re-implemented from the fl_v2 oracle
(``fl_v2/src/fl_v2/attacks_defenses/defenses/norm_clipping.py``) and validated
for byte-equivalence on saved fixtures. These are the shared primitives the
defense cores and the (T6) per-module gradient logger build on:

  * ``compute_update_norms``           — per-client L2 norm of (client - global).
  * ``clip_updates_by_l2_norm``        — server-side L2 clip (NormClip core).
  * ``compute_gradient_space_metrics`` — cos-to-mean, pairwise cosine, top-k
    coordinate-energy split (the Cycle-02 evasion-matrix instrumentation).

All operations are deterministic for a fixed, caller-ordered client list — the
caller must sort by partition-id before extracting ``client_params_list`` so
element ``i`` is stable across runs (see ``partition_sort_key`` in
``flower_strategies``). The numerical ops here (float32 reductions, argpartition,
einsum) are content-deterministic for a fixed input.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def flatten_params(params: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into one 1-D vector."""
    if not params:
        return np.array([], dtype=np.float32)
    return np.concatenate([np.asarray(p).reshape(-1) for p in params], axis=0)


def compute_update(
    global_params: List[np.ndarray],
    client_params: List[np.ndarray],
) -> List[np.ndarray]:
    """Per-tensor client update ``delta = client_params - global_params``.

    Each delta is cast back to the corresponding global tensor's dtype and
    shape (matches the oracle exactly — the dtype round-trip is load-bearing
    for float16/bfloat16 layers).
    """
    updates: List[np.ndarray] = []
    for gp, cp in zip(global_params, client_params):
        gp_arr = np.asarray(gp)
        cp_arr = np.asarray(cp)
        update = cp_arr - gp_arr
        update = np.asarray(update, dtype=gp_arr.dtype).reshape(gp_arr.shape)
        updates.append(update)
    return updates


def apply_update(
    global_params: List[np.ndarray],
    update: List[np.ndarray],
) -> List[np.ndarray]:
    """Apply a per-tensor update to global params: ``new = global + delta``."""
    new_params: List[np.ndarray] = []
    for gp, du in zip(global_params, update):
        gp_arr = np.asarray(gp)
        du_arr = np.asarray(du, dtype=gp_arr.dtype).reshape(gp_arr.shape)
        new_param = np.asarray(gp_arr + du_arr, dtype=gp_arr.dtype).reshape(gp_arr.shape)
        new_params.append(new_param)
    return new_params


def compute_update_norms(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
) -> List[float]:
    """L2 norm of each client's update relative to the global params."""
    norms: List[float] = []
    for client_params in client_params_list:
        update = compute_update(global_params, client_params)
        flat = flatten_params(update)
        norms.append(float(np.linalg.norm(flat, ord=2)))
    return norms


def clip_updates_by_l2_norm(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    clip_norm: float,
) -> Tuple[List[List[np.ndarray]], List[float], List[float]]:
    """Clip each client's update to ``clip_norm`` (L2).

    Returns ``(clipped_client_params_list, original_norms, clipped_norms)``.
    ``scale = min(1, clip_norm / (norm + 1e-12))`` per client, applied
    per-tensor with a dtype round-trip — identical to the oracle.
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be > 0, got {clip_norm}")

    clipped_client_params_list: List[List[np.ndarray]] = []
    original_norms: List[float] = []
    clipped_norms: List[float] = []

    for client_params in client_params_list:
        update = compute_update(global_params, client_params)
        flat_update = flatten_params(update)
        update_norm = float(np.linalg.norm(flat_update, ord=2))
        original_norms.append(update_norm)

        scale = min(1.0, clip_norm / (update_norm + 1e-12))
        clipped_update = []
        for gp, u in zip(global_params, update):
            gp_arr = np.asarray(gp)
            u_arr = np.asarray(u, dtype=gp_arr.dtype).reshape(gp_arr.shape)
            cu = np.asarray(u_arr * scale, dtype=gp_arr.dtype).reshape(gp_arr.shape)
            clipped_update.append(cu)

        clipped_flat_update = flatten_params(clipped_update)
        clipped_norms.append(float(np.linalg.norm(clipped_flat_update, ord=2)))
        clipped_client_params_list.append(apply_update(global_params, clipped_update))

    return clipped_client_params_list, original_norms, clipped_norms


def compute_gradient_space_metrics(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    topk_k: int,
) -> dict:
    """Per-client gradient-space diagnostics for one FL round (logging only).

    Read-only metrics over the client update vectors
    (``update = client_params - global_params``); they NEVER alter aggregation.
    Returns a dict with:

      * ``cosine_to_mean``   — ``cos(update_i, mean_update)``, len n.
      * ``pairwise_cosine``  — n×n ``cos(update_i, update_j)`` (collusion signal).
      * ``topk_energy_frac`` — fraction of update_i's L2 energy in the top-k
        ``|coordinates|`` of the mean update, len n.
      * ``topk_energy_k``    — the clamped k actually used.

    Byte-identical to the fl_v2 oracle: float32 stack, ``argpartition`` for the
    top-k positions, ``einsum`` energy reductions, in-place unit normalisation
    with the ``< 1e-12`` zero-update sentinel.
    """
    n = len(client_params_list)
    if n == 0:
        return {
            "cosine_to_mean": [],
            "pairwise_cosine": [],
            "topk_energy_frac": [],
            "topk_energy_k": 0,
        }

    d = int(sum(np.asarray(p).size for p in global_params))
    if d == 0:
        return {
            "cosine_to_mean": [0.0] * n,
            "pairwise_cosine": [[0.0] * n for _ in range(n)],
            "topk_energy_frac": [0.0] * n,
            "topk_energy_k": 0,
        }

    flat = np.empty((n, d), dtype=np.float32)
    for i, client_params in enumerate(client_params_list):
        update = compute_update(global_params, client_params)
        flat[i] = flatten_params(update).astype(np.float32)

    norms = np.linalg.norm(flat, axis=1)
    mean_update = flat.mean(axis=0)

    # --- top-k coordinate-energy split (on the RAW updates) ---
    k_eff = max(1, min(int(topk_k), d))
    topk_idx = np.argpartition(np.abs(mean_update), d - k_eff)[d - k_eff:]
    total_energy = np.einsum("ij,ij->i", flat, flat)
    sub = flat[:, topk_idx]
    topk_energy = np.einsum("ij,ij->i", sub, sub)
    topk_energy_frac = [
        float(topk_energy[i] / total_energy[i]) if total_energy[i] > 1e-24 else 0.0
        for i in range(n)
    ]

    # --- unit-normalise the stack IN PLACE ---
    safe_norms = np.where(norms < 1e-12, np.float32(1.0), norms.astype(np.float32))
    flat /= safe_norms[:, None]

    # --- cosine of each update to the mean update ---
    mean_norm = float(np.linalg.norm(mean_update))
    if mean_norm < 1e-12:
        cosine_to_mean = [0.0] * n
    else:
        mean_unit = (mean_update / mean_norm).astype(np.float32)
        cosine_to_mean = (flat @ mean_unit).astype(np.float64).tolist()

    pairwise_cosine = (flat @ flat.T).astype(np.float64).tolist()

    return {
        "cosine_to_mean": cosine_to_mean,
        "pairwise_cosine": pairwise_cosine,
        "topk_energy_frac": topk_energy_frac,
        "topk_energy_k": k_eff,
    }
