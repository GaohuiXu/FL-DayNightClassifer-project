"""WS5 — FoolsGold + FLAME defense unit tests.

FoolsGold: distinct update histories keep full trust; identical
(colluding) histories are down-weighted. FLAME: HDBSCAN drops a clear
malicious minority and the majority-guard admits everyone on a clean
(structureless) round. aggregate_weighted_updates reduces to the FedAvg
mean under uniform weights.
"""

import numpy as np

from fl_v2.strategy.flame import FlameFedAvg
from fl_v2.strategy.foolsgold import FoolsGoldFedAvg
from fl_v2.strategy.norm_tracking_fedavg import aggregate_weighted_updates


def _flame_data(honest: int, malicious: int, seed: int, c: float = 16.0):
    """Two well-separated update clouds: honest near e0, malicious near e1."""
    rng = np.random.default_rng(seed)
    d = 64
    e0 = np.zeros(d, dtype=np.float32); e0[0] = 1.0
    e1 = np.zeros(d, dtype=np.float32); e1[1] = 1.0
    rows = [c * e0 + rng.normal(0, 1, d).astype(np.float32) for _ in range(honest)]
    rows += [c * e1 + rng.normal(0, 1, d).astype(np.float32) for _ in range(malicious)]
    return np.stack(rows)


# --- FoolsGold ---

def test_foolsgold_distinct_histories_keep_trust():
    # Orthogonal histories -> every client is unique -> trust ~1.
    hist = [np.eye(5)[i] * 10.0 for i in range(5)]
    w = FoolsGoldFedAvg._foolsgold_weights(hist)
    assert all(x > 0.9 for x in w)


def test_foolsgold_colluders_downweighted():
    # Clients 0 and 1 share an identical history (colluders); 2/3/4 distinct.
    eye = np.eye(5) * 10.0
    hist = [eye[0], eye[0].copy(), eye[1], eye[2], eye[3]]
    w = FoolsGoldFedAvg._foolsgold_weights(hist)
    assert w[0] < 0.5 and w[1] < 0.5            # the colluding pair suppressed
    assert w[2] > 0.9 and w[3] > 0.9 and w[4] > 0.9


# --- FLAME ---

def test_flame_drops_malicious_minority():
    flat = _flame_data(40, 10, 0)
    norms = np.linalg.norm(flat, axis=1)
    admitted = FlameFedAvg._cluster_admitted(flat, norms)
    # Every malicious client (indices 40..49) is dropped ...
    assert set(admitted).isdisjoint(set(range(40, 50)))
    # ... and at least a clear majority of the 40 honest are admitted.
    assert len(admitted) >= 25


def test_flame_clean_round_admits_majority():
    flat = _flame_data(50, 0, 1)
    norms = np.linalg.norm(flat, axis=1)
    admitted = FlameFedAvg._cluster_admitted(flat, norms)
    # Uniform honest cloud: HDBSCAN with min_cluster_size=N/2+1 either
    # forms one big cluster (admit ~all) or fails to cluster (admit-all
    # degenerate fallback). Either way the majority must be admitted.
    assert len(admitted) >= 25


def test_flame_tiny_n_admits_all():
    flat = _flame_data(3, 0, 2)
    norms = np.linalg.norm(flat, axis=1)
    assert FlameFedAvg._cluster_admitted(flat, norms) == [0, 1, 2]


# --- shared aggregation helper ---

def test_aggregate_weighted_updates_uniform_is_fedavg_mean():
    # Global all-zero, uniform coefs -> result == mean of client params.
    global_params = [np.zeros((2, 2), dtype=np.float32)]
    clients = [
        [np.full((2, 2), 1.0, dtype=np.float32)],
        [np.full((2, 2), 3.0, dtype=np.float32)],
    ]
    arrays = aggregate_weighted_updates(global_params, clients, ["w"], [0.5, 0.5])
    assert np.allclose(arrays["w"].numpy(), 2.0)


def test_aggregate_weighted_updates_noise_term():
    global_params = [np.zeros((3,), dtype=np.float32)]
    clients = [[np.zeros((3,), dtype=np.float32)]]
    noise = [np.array([1.0, 2.0, 3.0])]
    arrays = aggregate_weighted_updates(
        global_params, clients, ["w"], [1.0], noise=noise
    )
    assert np.allclose(arrays["w"].numpy(), [1.0, 2.0, 3.0])
