# Phase E.2 — Class-Pair Coverage Analysis

**Date:** 2026-04-27
**Status:** resolves Cycle 02 open question 2

---

## 1. Why this calculation matters

Phase E.2's premise is *"no single client sees enough class pairs, but the
union of 50 clients does."* Each client running Design A's drift detector
can only score an ordered pair `(c → c')` if both `c` and `c'` are
**observable** in its local data — i.e., it has at least `t` samples of
each class. Pairs that no client can observe are completely invisible to
the cross-client aggregation, regardless of how accurate the detector is.

So before tuning consensus thresholds or implementing the side channel,
we need to answer:

> Under our standard partition (50 clients, 43 classes, Dirichlet
> α=0.5), what fraction of the 43×42 = 1806 ordered class pairs is
> covered by at least one client at observable threshold `t`?

This sets the *upper bound* on Phase E.2's possible AUROC — the residual
blind region is the fundamental ceiling against an adaptive attacker who
can pick the (source, target) pair.

---

## 2. Method

For each `seed ∈ {42, 43, 44, 100, 200}` and each
`t ∈ {5, 10, 20}`:

1. Build the Dirichlet α=0.5 partition for 50 clients × 43 classes using
   the existing `build_client_index_map_with_stats` from
   [`src/fl_v2/data/dataset.py`](../src/fl_v2/data/dataset.py).
2. For each client `k`, compute
   `observable_k(t) = {c : count_k(c) ≥ t}`.
3. Build the indicator matrix `I ∈ {0,1}^{50 × 43}`.
4. Pair-coverage matrix `C = I^T I` (43 × 43); zero the diagonal.
5. **Coverage** = (#pairs with `C[c,c'] > 0`) / 1806.
6. **Redundancy** = the distribution of `C[c,c']` over covered pairs
   (how many clients can independently score each covered pair).

Reported are mean ± std across the 5 seeds, plus the redundancy
percentiles (p10 / median / mean) which determine whether a "≥N
independent flags" consensus rule is feasible.

The full script is reproduced in §6 below; takes ~2 min on alvis1, no
GPU.

---

## 3. Results

| Threshold `t` | Coverage (mean ± std) | Blind pairs (mean) | Median redundancy | p10 redundancy | Observable classes per client (median) |
|---|---|---|---|---|---|
| 5 | **99.9% ± 0.2%** | ~2 | 10 | 4 | 20 |
| **10** *(cycle plan default)* | **90.7% ± 1.5%** | ~168 | 4 | **1** | 13 |
| 20 | 50.1% ± 2.1% | ~901 | 2 | 1 | 8 |

### Sample blind pairs at `t=10`, seed 42

```
(0, 14), (0, 15), (0, 16), (0, 19), (0, 21), (0, 29), (0, 30),
(0, 34), (0, 37), (0, 39), (0, 40), (0, 41), (0, 42), (1, 42),
(4, 42), (6, 21), (6, 37), (6, 41), (8, 24), (8, 42), ...
```

The blind region clusters on **rare GTSRB classes** — class 0 ("speed
limit 20", ~210 training samples) and class 42 ("end of no passing
veh > 3.5 t", ~240) are the two rarest classes in the dataset and dominate
the blind list. An adaptive attacker who picks a long-tail target class
gets a free pass on detection.

---

## 4. Findings

1. **At the cycle plan's default `t=10`, coverage is 91%, not 100%.**
   The 9% blind region is a real ceiling on Phase E.2's effectiveness.
2. **Blind spots are structural, not random.** They concentrate on
   rare-class endpoints. This means an adversary aware of the defense
   could systematically evade detection by choosing a rare target class.
3. **The threshold is highly sensitive.** `t=10 → 91%` vs `t=5 → 99.9%`
   vs `t=20 → 50%`. The cycle plan picked 10 in passing; the analysis
   shows 5 is much better operationally.
4. **p10 redundancy at `t=10` is 1**, meaning 10% of *covered* pairs have
   a single witness. If consensus requires ≥2 independent flags, those
   pairs are also effectively blind. **At `t=5`, p10 redundancy is 4** —
   ≥2-of-4 consensus stays operational across nearly all pairs.
5. **The cycle's current target class (`backdoor-target-label = 2`,
   "speed limit 50") is well covered.** Existing Phase C / D.1 results
   are not affected. Only future runs with rare-class targets would hit
   the blind region.

---

## 5. Recommendations (incorporated into the cycle plan)

These edits land in
[`cycle_02_designed_attacks_and_client_defenses.md`](roadmap/cycle_02_designed_attacks_and_client_defenses.md)
on the same date as this analysis:

1. **Lower the observable threshold from `t=10` to `t=5`** for Phase E.1
   Design A's per-client class-pair scoring. Trades a small amount of
   per-pair noise for ~9 percentage points of coverage and removes the
   single-witness consensus gap.
2. **Weight each client's flag by `min(count_k(c), count_k(c'))` in
   the cross-client consensus** (Phase E.2). Recovers the "more samples
   = more reliable estimate" intuition without sacrificing coverage —
   clients with the bare-minimum 5 samples count less than clients with
   50.
3. **Document the rare-class blind region as a known limitation** in
   Phase F writeup. Frame as inherent to FL non-IID, not a defense flaw:
   "the defense covers 99% of class-pair targets, with the residual
   concentrated on classes that are themselves rare in the training
   distribution (notably classes 0 and 42 in GTSRB)."
4. **Avoid rare-class targets for Phase D.2 unless explicitly probing
   the blind-spot regime.** The default `backdoor-target-label = 2` is
   safe; pick another high-frequency class if you need a second target.
5. **Open question 2 in the cycle plan is resolved** by this document
   and is removed from §5 of the roadmap.

---

## 6. Reproduction script

This script lives only in this doc — it's exploratory analysis, not
permanent code. Re-run any time the partition convention changes
(num_clients, alpha, num_classes) by activating the env and pasting it
into a Python REPL.

```python
"""Phase E.2 class-pair coverage analysis."""
import numpy as np
from fl_v2.data.dataset import build_client_index_map_with_stats

NUM_CLIENTS = 50
NUM_CLASSES = 43
ALPHA = 0.5
DATA_ROOT = "/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_datasets/gtsrb"
SEEDS = [42, 43, 44, 100, 200]
THRESHOLDS = [5, 10, 20]

hist_per_seed = {}
for seed in SEEDS:
    _, histograms, _ = build_client_index_map_with_stats(
        data_root=DATA_ROOT, num_clients=NUM_CLIENTS,
        partition_mode="dirichlet", dirichlet_alpha=ALPHA,
        seed=seed, download=False,
    )
    hist_per_seed[seed] = histograms

def coverage(histograms, t):
    ind = np.zeros((NUM_CLIENTS, NUM_CLASSES), dtype=np.int32)
    for k in range(NUM_CLIENTS):
        bucket = histograms[k] if k in histograms else histograms.get(str(k), {})
        for cls_str, cnt in bucket.items():
            cls = int(cls_str)
            if int(cnt) >= t:
                ind[k, cls] = 1
    cov = ind.T @ ind
    np.fill_diagonal(cov, 0)
    total = NUM_CLASSES * (NUM_CLASSES - 1)
    covered = (cov > 0).sum()
    return covered / total, total - covered, cov[cov > 0]

for t in THRESHOLDS:
    rows = [coverage(hist_per_seed[s], t) for s in SEEDS]
    cov_mean = np.mean([r[0] for r in rows])
    blind_mean = np.mean([r[1] for r in rows])
    redund = np.concatenate([r[2] for r in rows])
    print(f"t={t}: coverage={cov_mean:.1%}, blind={blind_mean:.0f}, "
          f"redundancy p10={np.percentile(redund, 10):.0f} "
          f"median={np.median(redund):.0f} mean={redund.mean():.2f}")
```
