from __future__ import annotations

from typing import List, Tuple

import numpy as np



def _flatten_params(params: List[np.ndarray]) -> np.ndarray:
    """Flatten a list of parameter arrays into one 1D vector."""
    if not params:
        return np.array([], dtype=np.float32)

    flat_parts = []
    for p in params:
        p_arr = np.asarray(p)
        flat_parts.append(p_arr.reshape(-1))

    return np.concatenate(flat_parts, axis=0)


def _compute_update(
    global_params: List[np.ndarray],
    client_params: List[np.ndarray],
) -> List[np.ndarray]:
    """Compute client update: delta = client_params - global_params."""
    updates: List[np.ndarray] = []

    for gp, cp in zip(global_params, client_params):
        gp_arr = np.asarray(gp)
        cp_arr = np.asarray(cp)
        update = cp_arr - gp_arr
        update = np.asarray(update, dtype=gp_arr.dtype).reshape(gp_arr.shape)
        updates.append(update)

    return updates


def _apply_update(
    global_params: List[np.ndarray],
    update: List[np.ndarray],
) -> List[np.ndarray]:
    """Apply update to global params: new_client_params = global + delta."""
    new_params: List[np.ndarray] = []

    for gp, du in zip(global_params, update):
        gp_arr = np.asarray(gp)
        du_arr = np.asarray(du, dtype=gp_arr.dtype).reshape(gp_arr.shape)
        new_param = gp_arr + du_arr
        new_param = np.asarray(new_param, dtype=gp_arr.dtype).reshape(gp_arr.shape)
        new_params.append(new_param)

    return new_params


def compute_update_norms(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
) -> List[float]:
    """
    Compute the L2 norm of each client's update relative to global params.

    Args:
        global_params: current global model parameters
        client_params_list: model parameters returned by each client

    Returns:
        List of L2 norms, one per client.
    """
    norms: List[float] = []
    for client_params in client_params_list:
        update = _compute_update(global_params, client_params)
        flat = _flatten_params(update)
        norms.append(float(np.linalg.norm(flat, ord=2)))
    return norms


def clip_updates_by_l2_norm(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    clip_norm: float,
) -> Tuple[List[List[np.ndarray]], List[float], List[float]]:
    """
    Clip each client's model update by L2 norm.

    Args:
        global_params: current global model parameters
        client_params_list: model parameters returned by each client
        clip_norm: maximum allowed L2 norm for each client update

    Returns:
        clipped_client_params_list:
            client parameters reconstructed from clipped updates
        original_norms:
            original update norms before clipping
        clipped_norms:
            update norms after clipping
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be > 0, got {clip_norm}")

    clipped_client_params_list: List[List[np.ndarray]] = []
    original_norms: List[float] = []
    clipped_norms: List[float] = []

    for client_params in client_params_list:
        update = _compute_update(global_params, client_params)
        flat_update = _flatten_params(update)

        update_norm = float(np.linalg.norm(flat_update, ord=2))
        original_norms.append(update_norm)

        scale = min(1.0, clip_norm / (update_norm + 1e-12))
        clipped_update = []

        for gp, u in zip(global_params, update):
            gp_arr = np.asarray(gp)
            u_arr = np.asarray(u, dtype=gp_arr.dtype).reshape(gp_arr.shape)
            cu = u_arr * scale
            cu = np.asarray(cu, dtype=gp_arr.dtype).reshape(gp_arr.shape)
            clipped_update.append(cu)

        clipped_flat_update = _flatten_params(clipped_update)
        clipped_norm = float(np.linalg.norm(clipped_flat_update, ord=2))
        clipped_norms.append(clipped_norm)

        clipped_client_params = _apply_update(global_params, clipped_update)
        clipped_client_params_list.append(clipped_client_params)

    return clipped_client_params_list, original_norms, clipped_norms


def compute_gradient_space_metrics(
    global_params: List[np.ndarray],
    client_params_list: List[List[np.ndarray]],
    topk_k: int,
) -> dict:
    """Per-client gradient-space diagnostics for one FL round.

    Read-only metrics over the client update vectors
    (``update = client_params - global_params``). They never alter
    aggregation — this is purely instrumentation for the Cycle-02
    evasion matrix. Returns a dict:

      - ``cosine_to_mean``:   list[float], len n. ``cos(update_i, mean_update)``
        where ``mean_update`` is the unweighted mean over every client
        update this round (honest + malicious — the strategy cannot
        label them; analysis separates by partition-id post hoc).
      - ``pairwise_cosine``:  list[list[float]], n x n. ``cos(update_i, update_j)``.
        The FoolsGold-style collusion signal.
      - ``topk_energy_frac``: list[float], len n. Fraction of update_i's
        L2 energy in the top-``topk_k`` ``|coordinates|`` of the mean
        update — the Neurotoxin-style "is the update concentrated where
        the federation moves, or hidden in cold coordinates" signal.
      - ``topk_energy_k``:    int, the clamped k actually used.

    Deterministic: pure numpy reductions over the caller-ordered client
    list. The caller (a ``NormTracking*`` strategy) sorts ``valid_replies``
    by partition-id before extracting ``client_params_list``, so element
    ``i`` is stable across runs. Logging-only, so it carries zero risk to
    the bit-deterministic aggregation invariant regardless.
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

    # (n, d) float32 stack of flattened updates, filled row by row to keep
    # peak memory at one (n, d) array. float32 matches the parameter dtype
    # and numpy float32 reductions are deterministic for a fixed input.
    flat = np.empty((n, d), dtype=np.float32)
    for i, client_params in enumerate(client_params_list):
        update = _compute_update(global_params, client_params)
        flat[i] = _flatten_params(update).astype(np.float32)

    norms = np.linalg.norm(flat, axis=1)            # (n,) raw-update norms
    mean_update = flat.mean(axis=0)                 # (d,) raw mean update

    # --- top-k coordinate-energy split (on the RAW updates) ---
    k_eff = max(1, min(int(topk_k), d))
    # The set of top-k |coordinate| positions of the mean update.
    # argpartition is content-deterministic for a fixed input.
    topk_idx = np.argpartition(np.abs(mean_update), d - k_eff)[d - k_eff:]
    total_energy = np.einsum("ij,ij->i", flat, flat)            # (n,)
    sub = flat[:, topk_idx]
    topk_energy = np.einsum("ij,ij->i", sub, sub)               # (n,)
    topk_energy_frac = [
        float(topk_energy[i] / total_energy[i])
        if total_energy[i] > 1e-24 else 0.0
        for i in range(n)
    ]

    # --- normalize the stack IN PLACE -> unit update vectors ---
    # (a degenerate zero update keeps norm 1.0 here, so it stays ~0 and
    # produces ~0 cosines — a reasonable sentinel for "no direction")
    safe_norms = np.where(
        norms < 1e-12, np.float32(1.0), norms.astype(np.float32)
    )
    flat /= safe_norms[:, None]

    # --- cosine of each update to the mean update ---
    mean_norm = float(np.linalg.norm(mean_update))
    if mean_norm < 1e-12:
        cosine_to_mean = [0.0] * n
    else:
        mean_unit = (mean_update / mean_norm).astype(np.float32)
        cosine_to_mean = (flat @ mean_unit).astype(np.float64).tolist()

    # --- pairwise cosine (Gram matrix of the unit update vectors) ---
    pairwise_cosine = (flat @ flat.T).astype(np.float64).tolist()

    return {
        "cosine_to_mean": cosine_to_mean,
        "pairwise_cosine": pairwise_cosine,
        "topk_energy_frac": topk_energy_frac,
        "topk_energy_k": k_eff,
    }