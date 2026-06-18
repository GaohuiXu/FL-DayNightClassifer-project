"""The malicious roster (D10) + the FL data-poisoning routing (fl_v3 T5).

**Threat model (D10 full participation).** The malicious roster is a **fixed, seed-derived, recorded**
subset of ``m = floor(ρN)`` of the derived ``0..N-1`` partition-ids — drawn ONCE (round-invariant)
against the **derived N** (NOT the round-varying sampler), written to the run manifest, and identical
across every Path-A actor. With ``N=25, ρ=0.2`` ⇒ ``m=5`` (honest-majority ``5 < 25/2``). The reported
malicious count is **``m_r = m = 5`` (ground truth)** — never read from any defence-assumed ``f_r``.

**Routing (the only T2/T3 touch is threading ``client_id``).** ``client_data`` injects a
:class:`PoisonedDatasetWrapper` into ``_make_loader`` **iff** ``client_id ∈ roster AND poison_rate > 0``.
The wrapper poisons a deterministic per-client subset of its poison-eligible samples (selected by a
PRIVATE ``random.Random(derive_seed(seed, MALICIOUS_SALT, client_id))`` — never the global RNG). The
**rate=0 / non-roster / non-selected path short-circuits to the literal ``base_ds[idx]`` as the first
line — ZERO extra RNG draws** (the §0.C5 byte-identical-null invariant).
"""
from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional

from torch.utils.data import Dataset

from fl_v3.attacks import poison as PO
from fl_v3.attacks import trigger as TR
from fl_v3.utils.runtime import derive_seed

# A reserved salt in the ``client_id`` slot of ``derive_seed`` — distinct from the participant
# sampler salts (SAMPLE_SALT_TRAIN=700_000_001 / SAMPLE_SALT_EVAL=700_000_002) and from any
# partition-id (0..N-1), so the roster / per-client poison-selection RNG never aliases a training
# or sampling stream.
MALICIOUS_SALT: int = 700_000_003
POISON_SELECT_SALT: int = 700_000_004  # the per-client sample-selection stream (vs the roster draw)


def malicious_roster(seed: int, num_clients: int, rho: float) -> List[int]:
    """The fixed, seed-derived roster: ``sorted(Random(derive_seed(seed, MALICIOUS_SALT)).sample(range(N), m))``
    with ``m = floor(ρN)``. Asserts ``1 ≤ m < N/2`` (honest majority; malicious-majority out of scope)."""
    n = int(num_clients)
    m = int(rho * n)  # floor
    if m < 1:
        raise ValueError(f"malicious count m=floor(ρN)=floor({rho}·{n})={m} < 1 — no attacker")
    if m >= n / 2.0:
        raise ValueError(f"malicious-majority out of scope: m={m} not < N/2={n/2.0} (N={n}, ρ={rho})")
    rng = random.Random(derive_seed(int(seed), MALICIOUS_SALT))
    return sorted(rng.sample(range(n), m))


def roster_record(seed: int, num_clients: int, rho: float, run_config: Optional[dict] = None) -> dict:
    """The recorded roster manifest (written beside the checkpoint; identical across Path-A actors)."""
    roster = malicious_roster(seed, num_clients, rho)
    rc = run_config or {}
    return {
        "seed": int(seed),
        "num_clients": int(num_clients),
        "rho": float(rho),
        "m_r": len(roster),                       # GROUND TRUTH malicious count (independent of any f_r)
        "honest_majority": bool(len(roster) < num_clients / 2.0),
        "roster": roster,
        "malicious_salt": MALICIOUS_SALT,
        "poison_mode": str(rc.get("attack-mode", "relocation")),
        "poison_rate": float(rc.get("attack-poison-rate", 0.0)),
        "delta_reloc": float(rc.get("attack-delta-reloc", 6.0)),
    }


class PoisonedDatasetWrapper(Dataset):
    """Wraps one malicious client's shard; poisons a deterministic per-client subset of its samples.

    The poisoned-index set is computed ONCE in ``__init__`` from the base dataset's info dicts (no
    image/LiDAR load) + the PRIVATE per-client RNG. ``__getitem__`` short-circuits every non-poisoned
    index to the literal ``base_ds[idx]`` (the byte-identical clean path, no RNG)."""

    def __init__(self, base_ds, seed: int, client_id: int, poison_rate: float,
                 cfg: PO.PoisonConfig, spec: TR.TriggerSpec):
        self.base_ds = base_ds
        self.cfg = cfg
        self.spec = spec
        self._poisoned: FrozenSet[int] = self._select_poisoned(base_ds, seed, client_id, poison_rate, cfg, spec)

    @staticmethod
    def _select_poisoned(base_ds, seed, client_id, poison_rate, cfg, spec) -> FrozenSet[int]:
        if poison_rate <= 0.0:
            return frozenset()  # rate=0 → NO RNG draw (byte-identical null, §0.C5)
        infos = getattr(base_ds, "_infos", None)
        if infos is None:
            raise AttributeError("PoisonedDatasetWrapper needs base_ds._infos for poison-eligibility")
        eligible = [i for i in range(len(base_ds)) if PO.info_has_poisonable_target(infos[i], cfg, spec)]
        k = int(round(float(poison_rate) * len(eligible)))
        if k <= 0 or not eligible:
            return frozenset()
        rng = random.Random(derive_seed(int(seed), POISON_SELECT_SALT, int(client_id)))
        return frozenset(rng.sample(eligible, k))

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        if idx not in self._poisoned:
            return self.base_ds[idx]  # FIRST line — byte-identical clean, zero RNG (§0.C5)
        return PO.poison_sample(self.base_ds[idx], self.cfg, self.spec)

    @property
    def num_poisoned(self) -> int:
        return len(self._poisoned)

    @property
    def sample_tokens(self):
        return self.base_ds.sample_tokens


def maybe_wrap_for_client(base_ds, client_id: int, run_config: dict, num_clients: int):
    """Return ``PoisonedDatasetWrapper(base_ds)`` iff this client is a malicious-roster member AND the
    attack is enabled with ``poison_rate > 0``; otherwise return ``base_ds`` UNCHANGED (honest /
    non-attack / null path → byte-identical clean). The single routing decision used by ``client_data``."""
    from fl_v3.utils.runtime import truthy

    if not truthy(run_config.get("attack-enabled", False)):
        return base_ds
    poison_rate = float(run_config.get("attack-poison-rate", 0.0))
    if poison_rate <= 0.0:
        return base_ds  # null path: no wrapper at all (strongest byte-identity guarantee)
    rho = float(run_config.get("attack-rho", 0.2))
    roster = malicious_roster(int(run_config.get("seed", 42)), int(num_clients), rho)
    if int(client_id) not in roster:
        return base_ds  # honest client → byte-identical clean
    cfg = PO.poison_config_from_run_config(run_config)
    spec = TR.trigger_spec_from_run_config(run_config)
    wrapped = PoisonedDatasetWrapper(base_ds, seed=int(run_config.get("seed", 42)), client_id=int(client_id),
                                     poison_rate=poison_rate, cfg=cfg, spec=spec)
    # Observable in the SLURM/Ray log (RAY_DEDUP_LOGS=0): proves the attack code path actually fired
    # for a malicious client (a silent clean run masquerading as poisoned is the §0.C trap).
    print(f"[ATTACK] client {client_id} ∈ roster {roster} | mode={cfg.mode} rate={poison_rate} "
          f"| poisoned {wrapped.num_poisoned}/{len(base_ds)} samples", flush=True)
    return wrapped
