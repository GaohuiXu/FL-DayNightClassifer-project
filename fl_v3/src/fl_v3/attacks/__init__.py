"""Fusion-aware backdoor attack suite (fl_v3 T5).

The thesis's core scientific contribution: a **camera-only data-poisoning** disappearance
backdoor (D2/D4) whose success genuinely depends on multimodal fusion, certified on the READY
full-participation log-group trainval model (T4: car recall 0.85, frozen ASR subset ``2ad8f8da…``,
``N=27,432``, checkpoint ``a80466c3…``) by the **5-condition same-model input-ablation**.

Public surface (consumed by ``training/tasks.py`` routing, the eval driver, the viz, the tests):

  * :mod:`fl_v3.attacks.trigger`         — ONE deterministic camera-trigger generator (train==infer).
  * :mod:`fl_v3.attacks.poison`          — the poisoning operators (center-RELOCATION disappearance
                                            [D4 primary, §0.A], box-deletion control, phantom).
  * :mod:`fl_v3.attacks.poisoned_client` — the fixed seed-derived malicious roster (D10, m=floor(ρN))
                                            + the ``PoisonedDatasetWrapper`` routing.
  * :mod:`fl_v3.attacks.fusion_ablation` — the 5-condition ablation (incl. the cond-5a zero-LiDAR-BEV
                                            forward hook + its two required guards).

**Bit-determinism is sacred.** No global RNG anywhere in the attack path: the trigger pattern/
placement are pure functions of the sample + config; the per-sample poison selection draws ONLY a
private ``random.Random(derive_seed(seed, MALICIOUS_SALT, client_id))``; ``poison_rate=0`` /
non-roster clients short-circuit to the byte-identical clean path with ZERO extra RNG draws.
"""
from __future__ import annotations
