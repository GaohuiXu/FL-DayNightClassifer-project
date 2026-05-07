# Cycle 02 Pivot Reliability Audit

*Companion to* `cycle_02_pivot_results.md`. *Determines whether the
seed-to-seed variance in Cycle 02 results is a real FL phenomenon or an
implementation/randomness artefact, before any further scientific
interpretation or new experiments.*

**Status:** v2, both confirmation jobs landed.
`audit_head_attr_stability` 6585212 confirmed the diagnostic is
BIT-IDENTICAL at fixed seed (§5.2). `_audit_reproduce_full_ft_pixel5`
6585215 found that **the same YAML at the same seed produced a
completely different trajectory** (final clean_acc 0.894 vs 0.667;
final ASR 0.673 vs 0.978) — definitive proof that the training
pipeline is non-deterministic at fixed seed (§1.3).

---

## Bottom-line verdict

1. **The training pipeline is NOT reproducible at fixed seed.**
   Empirically confirmed by job 6585215: re-running
   `pretrained_full_ft_pixel5.yaml` at seed=42 with the same commit
   produced final clean_acc 0.894 and ASR 0.673, vs. the original
   Wave 1 numbers 0.667 / 0.978. **The two runs were bit-identical for
   rounds 0–9 and then diverged catastrophically at round 10**,
   producing two qualitatively different final models from the same
   configuration. Root cause: client-side `torch.manual_seed()` is
   never called, `DataLoader(shuffle=True, …)` has no `generator`,
   CUDA non-determinism is unrestricted.
2. **The diagnostic itself is fully reproducible.** Job 6585212
   confirmed bit-identical `head_attribution_pct = 97.20189410245372`
   across two runs at the same diagnostic seed (16 decimal places),
   and within 0.5 pp across diagnostic seeds 4242–4245. **All
   seed-to-seed variance traces to the training pipeline, not the
   analysis.**
3. **The "97.2 % encoder anchoring" Cycle 02 headline does not survive
   reproduction.** Re-running its exact configuration produced a
   normal-trajectory model that would give a much lower
   head_attribution. The original seed=42 was an outlier *of its own
   configuration's run-level distribution*, not an outlier of the
   seed distribution.
4. **Multi-seed comparisons (seeds 42 vs 43 vs 44) are confounded** by
   run-level noise of comparable magnitude to seed-to-seed variation.
   We cannot distinguish "seed=42 stuck-trajectory" from "seed=42
   reproducible behaviour" without first making the pipeline
   deterministic. **The Wave 1+2 numbers cannot support any scientific
   claim about regime-dependent attack mechanisms.**
5. **What is still reliable:** the 15 mal cells (head_attribution
   stable across seeds; saturated regime), the canonconv1 head-only
   cells (frozen pretrained encoder loaded from the same ImageNet
   checkpoint is bit-identical across seeds, giving a near-
   deterministic — but, per measurement, not literally bit-identical
   — diagnostic baseline; precise values 0.0237373737 / 0.0237373737 /
   0.0236531987 across seeds 42/43/44, fourth-decimal divergence on
   seed 44), and the linear_probe_acc ≥ 0.95 finding (regime-invariant;
   insensitive to trajectory noise).
6. **Several latent bugs found** in config parsing (`bool("false")` is
   `True` in Python; affects 7 sites in server_app, client_app,
   wandb_logger). Did **not** manifest in Cycle 02 runs because the
   YAMLs only ever set booleans to `true`, but a tripwire for any
   future YAML that sets a boolean to `false`.

**Can we interpret the multi-seed variance scientifically?** **NO.**
Both training-run-level noise (catastrophic) and seed-level variance
contribute to the observed numbers, and we have no way to separate
them until the pipeline is made deterministic. **Wave 1 and Wave 2
results must be re-run after the 4 critical fixes (§7) before any
scientific interpretation, supervisor presentation, or D.2 work.**

---

## 1. Same-seed reproducibility

### 1.1 What the codebase enforces

| Source | File:line | Currently seeded? | Seed value |
|---|---|---|---|
| `random.seed(seed)` | server_app.py:289 | ✓ | YAML `seed` |
| `np.random.seed(seed)` | server_app.py:290 | ✓ | YAML `seed` |
| `torch.manual_seed(seed)` (server) | server_app.py:291 | ✓ | YAML `seed` |
| `torch.cuda.manual_seed_all(seed)` (server) | server_app.py:293 | ✓ | YAML `seed` |
| Dirichlet partition | data/dataset.py via `np.random.default_rng(seed)` | ✓ | YAML `seed` |
| Train/val split | dataset.py via `np.random.default_rng(seed + client_id)` | ✓ | seed + client_id |
| Pixel-backdoor poison mask | pixel_backdoor.py:96-118 via `np.random.default_rng(seed + client_id)` | ✓ | seed + client_id |
| Server-side initial weights | server_app.py:41-47 (deterministic given torch seed) | ✓ | torch RNG |
| **Client-side `torch.manual_seed`** | client_app.py | **✗ NEVER CALLED** | — |
| **Client `DataLoader(shuffle=True, generator=?)`** | dataset.py:260 | **✗ no `generator` arg** | uses torch.default_generator on actor |
| **CUDA determinism flag** | not set | ✗ | — |
| **cuDNN determinism flag** | not set | ✗ | — |
| **`torch.use_deterministic_algorithms(True)`** | not set | ✗ | — |
| Adam optimizer | training/train.py | ✓ deterministic given input order | — |
| Aggregation (FedAvg averaging) | strategy/ | ✓ order-independent (sum then divide) | — |
| Flower client selection | external lib | not relevant — `fraction-train=1.0` forces all 50 every round | — |

### 1.2 Verdict

Two runs of the same YAML, same seed, same commit are **NOT** guaranteed
to produce identical final models. The non-determinism comes from:

- Each Ray actor (50 supernodes mapped onto ~10 actors) starts with a
  Python interpreter whose `torch.default_generator` is initialised
  from the OS clock — different across runs.
- The per-client `DataLoader` shuffle pulls from this generator, so
  batch order during local training differs between runs.
- Adam updates depend on batch order, so per-client gradients differ.
- Aggregated global weights drift differently across runs.
- CUDA non-deterministic ops (atomic adds in conv backward, etc.)
  add additional small drifts.

### 1.3 Empirical confirmation — RESULT: pipeline NON-DETERMINISTIC

We resubmitted the *exact* `pretrained_full_ft_pixel5.yaml` (seed=42)
at the current commit (`08e04f1`) to a separate experiment-name
(`cycle02-audit-reproduce-full-ft-pixel5-seed42`). Job 6585215 ran
1:43:27 wallclock to round 100. **Final outcome diverged dramatically
from the original Wave 1 run despite identical configuration:**

| | Original (Wave 1, job 6570477) | Reproduce (job 6585215) | Δ |
|---|---|---|---|
| Final test_accuracy | 0.6669 | **0.8943** | +22.7 pp |
| Final ASR | 0.9776 | **0.6732** | −30.4 pp |
| Final target-class clean acc | low | 0.951 | — |
| `head_attribution_pct` | 97.2 % | `[TBD when 6592014 lands]` | — |

**Round-by-round trajectory comparison** confirms the divergence is
not a late-training fluctuation but a fundamental difference in escape
dynamics from the trivial-backdoor attractor:

| Round | Original acc / asr | Reproduce acc / asr |
|---|---|---|
| 1   | 0.0594 / 1.000 | 0.0594 / 1.000   ✓ identical |
| 5   | 0.0594 / 1.000 | 0.0594 / 1.000   ✓ identical (still stuck) |
| 9   | 0.0594 / 1.000 | 0.0594 / 1.000   ✓ identical |
| **10**  | **0.0594 / 1.000** (still stuck) | **0.0793 / 0.721** (escaping!) |
| 12  | 0.0594 / 1.000 (stuck) | 0.102 / 0.791 |
| 15  | 0.083 / 0.925 | 0.273 / 0.428 |
| 25  | 0.291 / 0.404 | 0.457 / 0.231 |
| 50  | 0.480 / 0.957 | ~0.84 / ~0.06 |
| 100 | 0.667 / 0.978 (stuck-trajectory final) | 0.894 / 0.673 (normal-trajectory final) |

The two runs were bit-identical for rounds 0–9 (the trivial-backdoor
plateau dominated by malicious-client gradients) and then diverged at
round 10 — the moment when the optimizer had to "decide" whether to
escape the attractor. **The escape decision depends on micro-scale
floating-point noise that the seed does not control.**

This means:
1. **The Cycle 02 Wave 1 results are not reproducible at the run
   level.** Re-running the same YAML produces a different model and a
   different headline number.
2. **The "97.2 % encoder anchoring" claim from seed=42 is from a
   one-off degenerate trajectory** that even running the same
   configuration does not reproduce.
3. **Multi-seed comparisons are confounded** by run-level noise of
   comparable magnitude to seed-to-seed variation. We cannot
   distinguish "seed=42 vs seed=43" effects from "run #1 vs run #2 of
   seed=42" noise without first making the pipeline deterministic.

The within-seed reproduce (this section) reads `[acc 0.894, ASR 0.673]`,
which is much closer to seeds 43/44 (acc 0.918/0.917) than to the
original seed=42 (acc 0.667). **The original seed=42 was an outlier of
its own configuration's distribution, not an outlier of the seed
distribution.** With proper seeding, all 3 seeds would likely cluster
together at ~0.90 acc and ~0.70 ASR.

This is the strongest single-finding in the audit: it overrides every
other interpretation of Wave 1 / Wave 2 numbers.

---

## 2. Randomness sources — full inventory

(Same table as §1.1; reproduced here for completeness.) Of 12 enumerated
randomness sources, **8 are seeded**, **3 are not** (client torch seed,
DataLoader shuffle generator, CUDA determinism), and **1 is not
relevant** (Flower client selection, due to `fraction-train=1.0`).

### Critical unseeded sources

1. **Client `torch.manual_seed()` is never called in `@app.train()`.** The
   server seeds at startup, but each Ray actor that hosts the
   `ClientAppActor` runs its own Python process where `torch` is
   imported afresh and `torch.default_generator` is initialised from
   the OS. Calling `torch.manual_seed(seed + client_id + round_num)`
   at the start of every `@app.train()` invocation would fix this.
2. **`DataLoader(shuffle=True)` without `generator=`** in dataset.py:260,
   257-263. The DataLoader inherits `torch.default_generator`, which
   in the actor is uncontrolled.
3. **CUDA non-determinism** is allowed throughout. Backward through a
   Conv2d on GPU uses `atomicAdd`, which produces different results
   each run for the same inputs. With ResNet18 + 100 rounds × 50 clients
   × 3 epochs, this drift accumulates.

### Recommended fix (one commit)

```python
# At the top of @app.train() in client_app.py:
def _train_callback(msg, ctx):
    ...
    seed = int(run_config["seed"])
    cid  = _get_client_id(ctx)
    rnd  = msg.metadata.get("server-round", 0)
    leaf_seed = (seed * 100003 + cid * 13 + rnd) & 0xFFFFFFFF
    random.seed(leaf_seed)
    np.random.seed(leaf_seed)
    torch.manual_seed(leaf_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(leaf_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

Plus: in `dataset.py:257-263`, pass `generator=torch.Generator().manual_seed(leaf_seed)` to the `DataLoader(...)` for the malicious-client dataset construction.

**This is REQUIRED before re-running multi-seed for venue-quality numbers.**

---

## 3. Config correctness

### 3.1 `bool("false") == True` bug — latent in training code

The same `bool(config.get("...", False))` antipattern that we already
fixed in `analysis/plot_features.py` exists in 7 places in actual
training code:

| File:line | Key | Code |
|---|---|---|
| server_app.py:52 | `pretrained-init` | `bool(context.run_config.get("pretrained-init", False))` |
| server_app.py:53 | `canonical-conv1` | `bool(context.run_config.get("canonical-conv1", False))` |
| server_app.py:179 | `canonical-conv1` | (same in evaluate_fn) |
| client_app.py:151 | `canonical-conv1` | `bool(context.run_config.get("canonical-conv1", False))` |
| utils/wandb_logger.py:43 | `wandb-enabled` | `bool(run_config.get("wandb-enabled", False))` |
| utils/wandb_logger.py:113 | `pretrained-init` | (tag derivation) |
| utils/wandb_logger.py:118 | `canonical-conv1` | (tag derivation) |

### 3.2 Did this bug manifest in Cycle 02 runs?

**No.** Verified by inspecting actual `[model] resnet18 pretrained=…`
prints in SLURM stdout for every cell:

- 9 main + 3 fallback + 12 multi-seed cells: *all* show the correct
  `pretrained=True canonical_conv1=False` for modified-conv1 cells and
  `pretrained=True canonical_conv1=True` for canonconv1 cells.
- This works because the YAMLs *omit* the boolean keys when they want
  `false` (so `context.run_config.get(..., False)` returns the
  *Python boolean* `False` from the pyproject.toml default — typed
  correctly by TOML parser) and *quote* them as `true`/`false` only
  via run_alvis.sh's awk parser when YAML overrides — and we only ever
  override with `true`.

### 3.3 What WOULD break

If any future YAML adds an explicit `canonical-conv1: false` line, the
awk parser produces `--run-config "canonical-conv1='false'"` →
flwr passes Python *string* `"false"` → `bool("false")` is `True` →
the run silently uses canonical conv1 contrary to the YAML.

**Required fix:** replace all 7 sites with the `_truthy()` helper used
in `plot_features.py`. One-line each. Should land before any new
YAML or before re-running existing experiments at new seeds.

### 3.4 Other config keys

All non-bool keys parse correctly:
- `seed`, `image-size`, `num-clients`, `num-server-rounds`, `num-local-epochs`, `backdoor-target-label`, `trigger-size`: `int(...)` ✓
- `learning-rate`, `lr-min`, `poison-fraction`, `trigger-value`, `fraction-train`, `dirichlet-alpha`, `val-ratio`: `float(...)` ✓
- `attack-type`, `defense-type`, `model-type`, `trigger-position`, `trainable-layers`: `str(...)` ✓
- `malicious-client-ids` (comma-sep → set[int]): `parse_client_ids` at client_app.py:96-98 ✓
- `checkpoint-rounds` (comma-sep → list[int]): correctly parsed at server_app.py:208-211 ✓

---

## 4. Data partition and attack opportunity (per seed)

For full_ft + pixel 5mal at seeds 42/43/44, the saved per-client label
histograms (`*_client_label_histograms.json`) give:

| seed | total samples (clients 0-4) | class-2 (target) samples | class-2 % | non-target samples | poisonable (×0.5) |
|---|---|---|---|---|---|
| 42 | 2,568 | 73 | 2.84 % | 2,495 | 1,247 |
| 43 | 2,313 | 160 | 6.92 % | 2,153 | 1,076 |
| 44 | 2,694 | 288 | 10.69 % | 2,406 | 1,203 |

Per-client breakdown for seed 42 (the outlier):
- Client 0: 352 samples, 9 of class 2
- Client 1: 380 samples, 0 of class 2
- Client 2: 619 samples, 59 of class 2
- Client 3: 546 samples, 3 of class 2
- Client 4: 671 samples, 2 of class 2

All 3 seeds have 1,076-1,247 poisonable samples (16 % range). The
class-2 prevalence among malicious clients varies 4× across seeds
(2.84 % → 10.69 %), but since the pixel-trigger attack only poisons
*non-target* samples (relabelling them to class 2), the class-2
abundance is mildly anti-correlated with attack strength.

**Verdict:** the data composition does not directly explain the 90 pp
swing in `head_attribution_pct`. The poisonable-sample budget is
similar across seeds. The variance comes from training dynamics, not
from per-client sample counts.

---

## 5. Head-feature decomposition stability

### 5.1 Indirect evidence: stability is excellent

Across the 6 head_only canonconv1 cells (3 seeds × {pixel5, pixel15}),
the diagnostic produces:

Re-checked against the Phase 3.1 wave canonconv1 head_only + 15mal
JSONs (full 10-decimal precision):

| Cell (15mal) | seed=42 | seed=43 | seed=44 |
|---|---|---|---|
| `clean_head_asr` | 0.0237373737 | 0.0237373737 | **0.0236531987** |
| `clean_head_clean_acc` | 0.6020585907 | 0.6019002375 | 0.6020585907 |

Two of three seeds (42, 43) are bit-identical to ten decimals on
`clean_head_asr`; seed 44 differs at the **fourth decimal place**.
The encoder is genuinely bit-identical across runs (head_only never
updates the encoder, so it stays at pretrained init), but the
clean-head retraining does not converge to a literally identical
state across all three training seeds.

The earlier draft of this section claimed "6-decimal-place identity
of `clean_head_asr`" with rounded values (0.0237 / 0.0237 / 0.0237);
that rounding hid the seed-44 divergence. The corrected reading: the
diagnostic is *near-deterministic* on canonconv1 head_only — strongly
consistent with "frozen encoder + same diagnostic seed → essentially
deterministic head training" but with a small residual that may
trace to whatever causes the audit's general round-6+ ε.

### 5.2 Direct empirical test — RESULT: diagnostic IS reproducible

5 runs of `head_feature_decomposition.py` on the same checkpoint
(`cycle02-pretrained-full-ft-pixel5_r100_seed42`):

| Diagnostic seed | clean_head_clean_acc | clean_head_asr | head_attribution_pct |
|---|---|---|---|
| 4242 (run 1) | **0.6790182106096595** | **0.027356902356902357** | **97.20189410245372** |
| 4242 (run 2) | **0.6790182106096595** | **0.027356902356902357** | **97.20189410245372** |
| 4243 | 0.680443388756928 | 0.022474747474747474 | 97.70124838570814 |
| 4244 | 0.677355502771180 | 0.026346801346801348 | 97.30520878174774 |
| 4245 | 0.680205859065717 | 0.027272727272727 | 97.21050365906156 |

**Same diagnostic seed → BIT-IDENTICAL across all 16 decimal places**
(every one of the 10 epoch-level losses and accuracies also matched
exactly). The diagnostic is fully deterministic at fixed checkpoint +
fixed seed.

**Different diagnostic seeds → variance < 0.5 pp** (head_attribution
ranges 97.20–97.70). The diagnostic is also robust to its own
internal seed.

**Verdict: the diagnostic is NOT a source of variance.** All the
seed-to-seed variance reported in the per-cell breakdown (§6) traces
back to differences in the trained encoder, not to the analysis
pipeline. The 4 critical fixes (§7) target the training pipeline
exclusively.

---

## 6. Full per-cell metric breakdown (seeds 42 / 43 / 44)

Notation: `head_rem = orig_asr − ch_asr` (the part of ASR that
disappears when the head is reinitialised + retrained on clean data);
`resid_feat = ch_asr` (the part of ASR that survives clean-head
retraining; intuitively how much of the backdoor lives in features
that the encoder produces).

| Cell                                | orig_acc | orig_asr | ch_acc | ch_asr | head_rem | resid_feat | head_attr % |
|---|---|---|---|---|---|---|---|
| full_ft + 5mal seed=42 ⚠              | 0.6670 | 0.9777 | 0.6790 | 0.0274 | 0.9503 | 0.0274 | **97.20** |
| full_ft + 5mal seed=43               | 0.9179 | 0.7989 | 0.9172 | 0.6226 | 0.1763 | 0.6226 | 22.06 |
| full_ft + 5mal seed=44               | 0.9167 | 0.9148 | 0.9120 | 0.8514 | 0.0634 | 0.8514 | 6.93  |
| full_ft + 15mal seed=42              | 0.5002 | 0.9798 | 0.5337 | 0.3239 | 0.6559 | 0.3239 | 66.94 |
| full_ft + 15mal seed=43              | 0.8909 | 0.9590 | 0.8912 | 0.1354 | 0.8237 | 0.1354 | 85.89 |
| full_ft + 15mal seed=44              | 0.8983 | 0.9669 | 0.8943 | 0.0486 | 0.9184 | 0.0486 | 94.98 |
| last_block + 5mal seed=42            | 0.6754 | 0.5645 | 0.6453 | 0.2248 | 0.3396 | 0.2248 | 60.17 |
| last_block + 5mal seed=43            | 0.6736 | 0.5168 | 0.6499 | 0.0087 | 0.5082 | 0.0087 | 98.32 |
| last_block + 5mal seed=44            | 0.8009 | 0.8519 | 0.7996 | 0.6913 | 0.1605 | 0.6913 | 18.84 |
| last_block + 15mal seed=42           | 0.5960 | 0.9540 | 0.5865 | 0.0893 | 0.8647 | 0.0893 | 90.64 |
| last_block + 15mal seed=43           | 0.7069 | 0.9129 | 0.6910 | 0.0463 | 0.8666 | 0.0463 | 94.93 |
| last_block + 15mal seed=44           | 0.7641 | 0.9335 | 0.7564 | 0.0345 | 0.8990 | 0.0345 | 96.30 |
| canonconv1 ho + 5mal seeds 42/43/44  | 0.56±0.01 | 0.06±0.01 | 0.6020 | 0.0237 | ~0.04 | 0.0237 | 60 ± 5 |
| canonconv1 ho + 15mal seeds 42/43/44 | 0.52±0.01 | 0.20±0.04 | 0.6020 | 0.0237 | ~0.18 | 0.0237 | 88 ± 3 |

⚠ **seed=42 full_ft + 5mal is a degenerate trajectory.** The training
got stuck at the trivial-backdoor attractor (predict every input as
class 2, ASR=1.0, acc=0.0594=class-2 base rate) for 12 consecutive
rounds (r=1-12) before slowly escaping. Final clean accuracy 0.667 is
~25 pp lower than seeds 43/44 (0.918, 0.917). **The 97.2 %
head_attribution headline came from this anomalous trajectory.**

### Patterns

- **`resid_feat` (= clean_head_asr) is the variance driver.** It ranges
  0.0274 to 0.8514 in the 5 mal cells. This is what makes
  `head_attribution_pct` swing from 97 % to 7 %.
- **`resid_feat` is large (> 0.5)** when the encoder has internalised
  the trigger response — i.e. produces features that get classified
  as target even with a freshly-retrained clean head. This is real,
  and is what we now know happens at full_ft + 5mal in 2/3 seeds and
  at last_block + 5mal in 1/3 seeds.
- **`resid_feat` is small (< 0.05)** when the encoder produces clean
  features regardless of trigger; the attack is purely in the head.
- **`head_attr` saturates (> 90 %) when ASR is high AND `resid_feat` is
  small** — a robust state that occurs across all 15 mal cells and the
  canonconv1 head_only cells.

---

## 7. Minimum reruns recommended

Given the above:

1. **Job 6585215 (in queue):** rerun `pretrained_full_ft_pixel5.yaml` at
   seed=42 to test pipeline reproducibility. Compare final
   `test_accuracy`, `asr`, `head_attribution`. If within ±2 pp, the
   pipeline is "effectively reproducible" and existing variance is
   real. If not, we need to fix the `torch.manual_seed` + DataLoader
   shuffle issues before any more experiments.
2. **Job 6585212 (in queue):** test diagnostic stability on a fixed
   checkpoint (5 runs: 2 same-seed + 3 different-seed). Confirms the
   diagnostic is or is not adding noise.
3. **Once the bool() bug fix lands**, rerun **only the seed=42
   full_ft + 5mal cell** at seed=42 again to confirm the trajectory
   stuck-at-attractor pattern is real (not a config corruption). If
   the new seed=42 run no longer gets stuck, the original was an
   artefact of the bug.

We do *not* yet need to add seeds 45-46 or rerun the saturated cells
(which are stable). Variance fix scope = 1-2 reruns + the bool() fix +
the client_app seeding fix.

---

## 8. Final judgment

After the empirical confirmation in §1.3 and §5.2, the verdict on
each result class is:

| Result | Reliable? | Reason |
|---|---|---|
| **The diagnostic itself, given a fixed input checkpoint** | ✓ FULLY | Job 6585212 ran the v1 diagnostic 5 times on the same checkpoint at the same diagnostic seed and produced bit-identical outputs. This isolates the diagnostic from training-pipeline non-determinism but does not address whether the input checkpoint itself is reproducible. |
| **canonconv1 head_only** (5mal, 15mal) | ⚠ MOSTLY | Frozen pretrained encoder + same diagnostic seed should give deterministic head training. Empirically: 3-seed clean_head_asr = 0.0237373737 / 0.0237373737 / 0.0236531987 — bit-identical for two of three seeds, fourth-decimal divergence on seed 44. Direction matches "near-deterministic by construction" but is not literally bit-identical. |
| **linear_probe_acc ≥ 0.95** | ✓ YES | Regime-invariant; the encoder always produces linearly-separable triggered features regardless of trajectory noise. |
| **15 mal head_attribution** | ⚠ PROBABLY | SD ≤ 14 pp across seeds; saturated regime is more robust to trajectory noise but still subject to the same non-determinism. **Needs re-verification after fixes.** |
| **5 mal head_attribution numbers** | ✗ INVALID | The Wave 1 seed=42 "97.2%" headline does not survive reproduction. The Wave 2 seeds 43/44 numbers are similarly suspect — they are single-run snapshots of a non-deterministic pipeline. |
| **"Encoder anchoring at moderate pressure" hypothesis** | ✗ RETIRED | Was the artefact of an unreproducible degenerate trajectory. The reproducing run did NOT get stuck at the trivial-backdoor attractor — so the seed=42 stuck-trajectory itself is a coin-flip artefact, not a property of seed=42. |
| **"Pretrained init shifts attack mechanism"** | ⚠ UNTESTED | We have not yet shown this with reproducible runs. Cannot claim until 4 fixes are committed and seeds are rerun. |
| **`centroid_l2` per-cell values** | ⚠ NEEDS RERUN | Variance across non-deterministic runs is unknown; current numbers are single snapshots. |

**Required fixes before any more scientific experiments:**

1. **`bool("false") == True` bug** — replace 7 occurrences of
   `bool(context.run_config.get("...", False))` with `_truthy(...)`
   helper. Currently latent but a tripwire for any future YAML.
2. **Client-side `torch.manual_seed`** — add seeding hook at top of
   `@app.train()` in client_app.py. One commit, ~5 lines.
3. **Seeded `DataLoader` generator** — pass
   `generator=torch.Generator().manual_seed(leaf_seed)` to the
   per-client trainloader in dataset.py. One line.
4. **CUDA determinism** — set `torch.backends.cudnn.deterministic =
   True` and `torch.use_deterministic_algorithms(True, warn_only=True)`
   in run_alvis.sh and inside server / client startup. Trade-off:
   slightly slower runs; targets cuDNN as a non-determinism source but
   on its own is not sufficient for end-to-end bit-reproducibility (a
   residual ε remains and is being investigated separately).

**No new experiments (D.2 prototype, more seeds, new attacks) until the
above 4 fixes are committed and verified by re-running the same-seed
audit.** After the fixes, rerun seeds 42/43/44 and confirm whether the
5 mal variance shrinks (if yes, much of it was non-determinism) or
persists (if persist, marginal-attack variance is genuinely stochastic
and we report mean ± std + the attractor finding).

---

## Audit jobs in flight

- `6580257 cycle02_pivot_tsne` — t-SNE comparison panel resubmission with
  fixed labels. Independent of audit; informational.
- `6585212 audit_head_attr_stability` — 5-run head-attribution stability
  test on full_ft+5mal seed=42 checkpoint. **Outputs go to §5.2.**
- `6585215 _audit_reproduce_full_ft_pixel5` — same-seed (42) training
  rerun. **Outputs go to §1.3.**

This document will be re-committed with their numbers as soon as they
land.

---

## 9. Closing the audit — fixes implemented and verification status

### 9.1 Six non-determinism sources, all addressed

| # | Source | Resolution | Commit |
|---|---|---|---|
| 1 | `bool("false") == True` in 7 config sites | `truthy()` helper in `utils/runtime.py` | `795f75e` |
| 2 | Client `torch.manual_seed()` never called | Per-call seeding in `@app.train()` | `795f75e` |
| 3 | `DataLoader(shuffle=True, generator=None)` | Explicit `torch.Generator()` | `795f75e` |
| 4 | `fl_v2/src/fl_v2/data/` source `.gitignore`d | Anchored `/data/` to repo root | `795f75e` |
| 5 | cuDNN atomic-add conv backward | `cudnn.deterministic=True`, `benchmark=False` | `725cae5` |
| 6 | Strategy aggregation order followed Ray's task order | `valid_replies.sort(key=...src_node_id)` in 6 strategies + `derive_seed` to hashlib + `PYTHONHASHSEED=0` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | `1f5e70d` |

### 9.2 Same-seed reproducibility verification

**v1 verification (jobs 6592415 / 6592416, after fixes 1–4 only):** rounds 0–7 bit-identical, diverged at round 8. Final acc 0.6690 vs 0.6187 (Δ 5 pp); final ASR 0.2032 vs 0.1009 (Δ 10 pp). Confirmed seeding fixes worked but cuDNN was still drifting → motivated fix 5.

**v3 verification (jobs 6593797 / 6593798, after fix 5):** never executed — cancelled before running because additional Phase 1 fixes (env vars, hashlib `derive_seed`, strategy sort) landed on top of fix 5. Replaced by v4.

**v4 verification (jobs 6594138 / 6594139, after all six fixes including 9.1#6):** **submitted 2026-05-07; pending in queue at audit close.** Same YAML, same seed=42, 30 rounds, fresh Ray actors. Pass criterion: identical `summary.json`, identical `rounds.csv`, identical SHA-256 of final checkpoint. **Status to be filled in here when the jobs complete.**

### 9.3 Aggregation order-dependence — what we found and fixed

Read-only inspection of `fl_v2/src/fl_v2/strategy/*.py` and `flwr.serverapp.strategy.*` (Flower 1.27.0) confirmed the user's hypothesis:

- `aggregate_arrayrecords` in `flwr.serverapp.strategy.strategy_utils` line ~101 performs in-place float summation across clients in whatever order the caller iterates the reply list (`aggregated_np_arrays[key] += value.numpy() * weight`). Floating-point summation is non-associative — different orders give different bit patterns.
- None of our six strategies (`NormTrackingFedAvg`, `NormClippedFedAvg`, `Bulyan`, `FedMedian`, `FedTrimmedAvg`, `CapturedKrum`/`CapturedMultiKrum`) sorted `valid_replies` by client id before iterating, so Ray's task-completion order leaked into the aggregated weights.
- Krum/MultiKrum additionally have an order-dependent argsort tie-break in `flwr/.../multikrum.py:246` when several clients have near-identical Krum scores — original list position decides the winner.

Fix: every strategy's `aggregate_train` now sorts replies by `metadata.src_node_id` (commit `1f5e70d`). The sort cost is one `O(n log n)` pass on a list of ≤50 messages — completely free at FL runtime.

### 9.4 Verification series v4–v8 (intermediate; all on 30-round runs)

After fixes 1–7, a series of 30-round same-seed verifications all
showed residual divergence during the chaotic-attractor escape phase
(rounds 5–10): v4 cross-node Δ 14 pp, v6 cross-node + use_det_algos
Δ 23 pp, v7 same-node forced Δ 12.6 pp, v8 single-actor timed out.
At the time these results pointed at a 7th unfixed source. **The
30-round verification was the wrong test.** Pixel-trigger 5 mal at
the *transient* attractor-escape phase is genuinely chaotic; that
chaos manifests at 30 rounds when the model is still in the trivial-
backdoor basin. By 100 rounds the model has converged and the
residual variance is moderate.

### 9.5 Phase 3.1 wave (the actual closure of the audit)

A 3-cell × 3-seed minimum-viable rerun on the fully-fixed pipeline
(commit `9d72bcf`, with Ray internal ports also derived from
`$SLURM_JOB_ID` since fix #6's strategy sort by itself was not enough
to keep concurrent same-node jobs from silently failing). 9 training
runs (jobs 6599453–6599461) × 100 rounds + per-cell head-feature
decomposition.

| Cell | seed=42 | seed=43 | seed=44 | mean head_attr ± SD | acc mean ± SD |
|---|---|---|---|---|---|
| `full_ft + 5mal` | 12.50 % | 44.71 % | 40.99 % | **32.73 % ± 17.62 pp** | 0.889 ± 0.028 |
| `last_block + 5mal` | 62.97 % | 61.90 % | 55.45 % | **60.11 % ± 4.07 pp** | 0.717 ± 0.049 |
| `canonconv1 head_only + 15mal` | 89.77 % | 84.71 % | 88.77 % | **87.75 % ± 2.68 pp** | 0.520 ± 0.011 |

**Bit-reproducibility — withdrawn claim.** An earlier draft of this
section asserted that seeds 43 and 44 of `full_ft + 5mal` gave
identical head_attr (44.71 % and 40.99 %) across two independent
3-seed runs (the user-requested 6598089 wave + the Phase 3.1 wave
6599453). That comparison **does not exist in the data**.
`sacct -j 6598089,6598090,6598091` shows wave-1 seeds 43 and 44
"COMPLETED" in 1 m 27 s and 1 m 39 s respectively — the Ray-port
silent-failure pattern (no rounds trained, no head-attribution
produced). Only seed=42 (job 6598089, 1 h 53 m) ran the full 100
rounds on the pre-Ray-port-fix commit; the other two failed. There
was nothing to compare wave-2 against, and the "bit-reproducibility
verified" claim was an unverified inference repeated as fact. It is
withdrawn.

What we can defensibly say from the existing data:

- The audit reproducibility pair 6594906 / 6594907 (post-7-fixes,
  same commit, same seed) was bit-identical for rounds 0–5 and then
  diverged at round 6+ (Δ ≈ 12–23 pp final accuracy).
- All 9 Phase 3.1 wave-2 runs completed their 100 rounds without
  silent failure, so the *numerical outputs are valid as one
  realisation each* — but a within-commit rerun has not yet been
  measured to quantify how much they would shift on a second
  realisation.
- A within-commit reprocheck has been queued (job 6600759,
  `cycle02-reprocheck-full-ft-pixel5_seed42`) specifically to fill
  this gap. Wandb already shows that 6598089 (pre-Ray-port commit)
  vs 6599453 (post-Ray-port commit) have visibly different
  per-round trajectories that converge to similar but not identical
  final values; whether the divergence holds within a single commit
  is what 6600759 will tell us.

**Per-cell variance is heterogeneous, not universal.** Three regimes
under the v1 fixed-10-epoch diagnostic:

- `canonconv1 head_only + 15mal`: **SD 2.68 pp** (saturated, frozen
  encoder, single-attractor). Clean-head ASR rounds to 0.024 across
  all 3 seeds; precise values 0.0237373737 (seed 42), 0.0237373737
  (seed 43), 0.0236531987 (seed 44) — bit-identical for two of
  three seeds, fourth-decimal divergence on seed 44.
- `last_block + 5mal`: **SD 4.07 pp** (restricted capacity, few
  attractors).
- `full_ft + 5mal`: **SD 17.62 pp** (full capacity at marginal attack
  pressure, multi-attractor). This is the cell with the largest
  spread; whether N ≥ 5 seeds collapses the spread or grows it
  remains to be tested.

### 9.6 Closing verdict

The audit identified seven software-level non-determinism sources
(one of which had two facets, addressed in two passes: `1f5e70d`
for strategy aggregation sort + hashlib `derive_seed` + env vars;
`6db1dd1` for Ray internal ports). All identified sources are closed
in code, but the audit's own pair 6594906 / 6594907 already
established that bit-identical reproducibility holds only through
rounds 0–5; rounds 6+ still drift under residual ε whose origin
is not yet pinpointed.

**Pipeline status (honest):** *valid runs* (no silent failures, full
100-round trajectories) but *not yet bit-reproducible end-to-end*.
The single-realisation per-cell numbers are usable for ordering
arguments and for revealing seed-to-seed variance bands, but should
not be quoted to many decimal places as if the underlying pipeline
were deterministic.

The Wave 1+2 numerical record remains withdrawn (banner on
`cycle_02_pivot_results.md`). The Cycle 02 thesis narrative is being
rebuilt — see the in-flight v2 (convergent diagnostic) results in
the Friday brief — and all cross-cycle comparisons are now treated
as provisional pending the Phase 3.0 sentinel + the Phase 3.1 v2
rerun completing on the audit-fixed pipeline.

The recovery plan's Phase 1 (codebase reliability) is closed in the
sense that the seven identified bugs are fixed, but a residual ε
remains. Phase 3.0 (Cycle 01 sentinel rerun) has produced its first
deterministic-pipeline number (sentinel head_attr v2 = 18.2 %).
Phase 3.1 (minimum-viable rerun) is closed for v1 numbers, in
flight for v2. Phase 3.2 (full 24-cell rerun) and Phase 5
(regression test + reproducibility guide) remain. A separate
investigation has been spawned to characterise the residual ε.
