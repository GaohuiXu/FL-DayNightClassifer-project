"""Deterministic client-participant sampler (fl_v3 T3, DT3-B).

The crux of full-loop FL bit-determinism at ``fraction-train < 1``.

**Why a custom sampler is REQUIRED (verified against flwr 1.27 source).**
Flower's ``FedAvg.configure_train`` selects participants with
``random.sample(list(grid.get_node_ids()), sample_size)`` (``strategy_utils.sample_nodes``).
The node_ids come from ``secrets.token_bytes(32)`` (``vce_api._register_nodes``) and
``LinkState.get_nodes`` returns them as a **``set``** — so two same-seed simulation
drivers see a *different, unordered* node_id space, and ``random.sample`` over it picks a
*different participant subset* each run even though the global RNG is seeded. ``FedAvg``
has no seed argument. The server-side strategy also has **no** node_id→partition-id map
(the ``partition-id`` lives only in each client's ``node_config``), so the selection must
be made over the **fixed ``0..N-1`` partition-id space** and then mapped to that run's
node_ids via a one-time discovery probe (see ``flower_strategies``).

This module is the single source of truth for that selection. It is framework-free
(no flwr import) so the **same** function drives both the Ray strategy and the
in-process ``local_runner`` cross-check — they MUST select identical partition-ids.

Seeds are derived through the T0 harness (``derive_seed``) using reserved integer
**salts** in the ``client_id`` slot. ``derive_seed`` casts its arguments to ``int``
(the SPEC's ``derive_seed(seed, "sample", round)`` shorthand cannot pass a string), so a
string tag is impossible; the salts below are far above any real partition-id
(``0..N-1``), so a sampling seed can never collide with a per-client training seed
``derive_seed(seed, client_id, round)``.
"""
from __future__ import annotations

import random
from typing import List

from fl_v3.utils.runtime import derive_seed

# Reserved salts in the ``client_id`` slot of ``derive_seed`` — distinct from any
# partition-id (0..N-1) and from each other, so the train- and evaluate-participant
# RNG streams are independent and never alias a per-client training seed.
SAMPLE_SALT_TRAIN: int = 700_000_001
SAMPLE_SALT_EVAL: int = 700_000_002


def sample_size_from_fraction(num_partitions: int, fraction: float, min_nodes: int) -> int:
    """Number of participants per round, mirroring Flower's FedAvg arithmetic.

    Flower computes ``sample_size = max(int(N * fraction), min_nodes)`` then samples that
    many node_ids. We reproduce it over the partition-id space and additionally **clamp to
    N** (you cannot sample more distinct partitions than exist; Flower would instead block
    in ``sample_nodes`` waiting for more nodes). ``fraction <= 0`` ⇒ 0 (skip the round).
    """
    if fraction <= 0.0:
        return 0
    n = int(num_partitions * fraction)
    n = max(n, int(min_nodes))
    return max(0, min(int(num_partitions), n))


def select_partition_ids(
    seed: int,
    server_round: int,
    num_partitions: int,
    fraction: float,
    min_nodes: int,
    *,
    salt: int = SAMPLE_SALT_TRAIN,
) -> List[int]:
    """Deterministic, cross-run-stable participant **partition-ids** for a round.

    ``sorted(random.Random(derive_seed(seed, salt, server_round)).sample(range(N), n_r))``.
    The result depends only on ``(seed, salt, server_round, N, fraction, min_nodes)`` —
    NOT on the per-run-random node_ids — so two same-seed runs select the identical
    partition-id set every round (the DT3-B invariant). Returned **sorted** so the
    selection order is itself canonical (the aggregator re-sorts by partition-id anyway).
    """
    n_r = sample_size_from_fraction(num_partitions, fraction, min_nodes)
    if n_r <= 0:
        return []
    rng = random.Random(derive_seed(int(seed), int(salt), int(server_round)))
    return sorted(rng.sample(range(int(num_partitions)), n_r))
