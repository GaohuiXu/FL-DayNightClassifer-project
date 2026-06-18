"""T5 threat model: roster determinism/honest-majority, rate=0 byte-identity (no RNG), m_r⊥f_r."""
from __future__ import annotations

import pathlib
import re

import pytest
import torch

from fl_v3.attacks import trigger as TR, poison as PO
from fl_v3.attacks.poisoned_client import (
    MALICIOUS_SALT, malicious_roster, roster_record, PoisonedDatasetWrapper, maybe_wrap_for_client,
)
from _attack_fixtures import attack_sample


class _InfoDS:
    """A minimal base_ds exposing ``_infos`` (for poison-eligibility) + ``__getitem__`` (loaded sample)."""
    def __init__(self, n):
        self._samples = [attack_sample(i) for i in range(n)]
        self._infos = [{"gt_boxes": s["gt_boxes"].numpy(), "gt_labels": s["gt_labels"].numpy(),
                        "gt_num_lidar_pts": s["gt_num_lidar_pts"].numpy(), "gt_in_range": s["gt_in_range"].numpy(),
                        "lidar2img": s["lidar2img"].numpy()} for s in self._samples]
    def __len__(self): return len(self._samples)
    def __getitem__(self, i): return self._samples[i]
    @property
    def sample_tokens(self): return [s["sample_token"] for s in self._samples]


def test_roster_deterministic_and_honest_majority():
    r = malicious_roster(20259, 25, 0.2)
    assert r == malicious_roster(20259, 25, 0.2)        # deterministic
    assert len(r) == 5 and len(r) < 25 / 2              # m=floor(0.2·25)=5, honest majority
    assert r == sorted(r) and len(set(r)) == 5


def test_roster_rejects_malicious_majority_and_zero():
    with pytest.raises(ValueError):
        malicious_roster(1, 4, 0.5)   # m=2, not < N/2=2
    with pytest.raises(ValueError):
        malicious_roster(1, 4, 0.1)   # m=0 → no attacker


def test_salt_distinct_from_sampler_salts():
    from fl_v3.strategy.sampling import SAMPLE_SALT_TRAIN, SAMPLE_SALT_EVAL
    from fl_v3.attacks.poisoned_client import POISON_SELECT_SALT
    assert len({MALICIOUS_SALT, POISON_SELECT_SALT, SAMPLE_SALT_TRAIN, SAMPLE_SALT_EVAL}) == 4


def test_rate0_wrapper_draws_no_rng_and_is_byte_identical():
    ds = _InfoDS(6)
    cfg = PO.PoisonConfig(mode="relocation", tau_pts=10)
    w = PoisonedDatasetWrapper(ds, seed=1, client_id=0, poison_rate=0.0, cfg=cfg, spec=TR.TriggerSpec(min_points=4))
    assert w.num_poisoned == 0
    for i in range(len(ds)):
        a, b = ds[i], w[i]
        assert torch.equal(a["images"], b["images"]) and torch.equal(a["gt_boxes"], b["gt_boxes"])


def test_rate1_wrapper_poisons_eligible_and_shortcircuits_rest():
    ds = _InfoDS(6)
    cfg = PO.PoisonConfig(mode="relocation", delta_reloc=6.0, tau_pts=10)
    w = PoisonedDatasetWrapper(ds, seed=1, client_id=0, poison_rate=1.0, cfg=cfg, spec=TR.TriggerSpec(min_points=4))
    assert w.num_poisoned == len(ds)  # every sample has a poisonable car
    for i in range(len(ds)):
        assert not torch.equal(ds[i]["images"], w[i]["images"])  # poisoned (trigger present)


def test_wrapper_determinism_same_seed_same_selection():
    ds = _InfoDS(8)
    cfg = PO.PoisonConfig(mode="relocation", tau_pts=10); spec = TR.TriggerSpec(min_points=4)
    a = PoisonedDatasetWrapper(ds, seed=7, client_id=3, poison_rate=0.5, cfg=cfg, spec=spec)
    b = PoisonedDatasetWrapper(ds, seed=7, client_id=3, poison_rate=0.5, cfg=cfg, spec=spec)
    assert a._poisoned == b._poisoned


def test_maybe_wrap_routes_only_roster_when_enabled():
    ds = _InfoDS(4)
    rc = {"attack-enabled": True, "attack-poison-rate": 0.5, "attack-rho": 0.2, "seed": 20259,
          "attack-mode": "relocation", "asr-tau-pts": 10}
    roster = malicious_roster(20259, 25, 0.2)  # [2,3,12,13,19]
    honest = maybe_wrap_for_client(ds, client_id=0, run_config=rc, num_clients=25)
    assert honest is ds  # honest client → unchanged
    mal = maybe_wrap_for_client(ds, client_id=roster[0], run_config=rc, num_clients=25)
    assert isinstance(mal, PoisonedDatasetWrapper)


def test_maybe_wrap_noop_when_disabled_or_rate0():
    ds = _InfoDS(2)
    assert maybe_wrap_for_client(ds, 2, {"attack-enabled": False}, 25) is ds
    assert maybe_wrap_for_client(ds, 2, {"attack-enabled": True, "attack-poison-rate": 0.0}, 25) is ds


def test_roster_record_reports_m_r_ground_truth():
    rec = roster_record(20259, 25, 0.2, {"attack-mode": "relocation", "attack-poison-rate": 0.5})
    assert rec["m_r"] == 5 and rec["honest_majority"] and rec["roster"] == malicious_roster(20259, 25, 0.2)


def test_attack_source_never_reads_defense_f_r():
    """m_r is GROUND TRUTH — the attack package must NEVER read a defence-assumed count. Grep-guards
    actual config-key reads / defence variables (the f_r prose in the docstrings is allowed — it
    DOCUMENTS the rule)."""
    pkg = pathlib.Path(__file__).resolve().parents[1] / "src" / "fl_v3" / "attacks"
    # quoted defence config keys (a real read) or the defence count variable
    banned = re.compile(r"""["'](?:num-malicious-nodes|krum-num-to-select|num-to-select)["']|\bnum_malicious\b""")
    for p in pkg.glob("*.py"):
        assert not banned.search(p.read_text()), f"{p.name} reads a defence/f_r knob"
