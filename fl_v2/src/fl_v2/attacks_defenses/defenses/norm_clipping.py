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