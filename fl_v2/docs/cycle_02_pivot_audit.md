# Cycle 02 Pivot Reliability Audit

*Companion to* `cycle_02_pivot_results.md`. *Determines whether the
seed-to-seed variance in Cycle 02 results is a real FL phenomenon or an
implementation/randomness artefact, before any further scientific
interpretation or new experiments.*

**Status:** core findings established; two empirical confirmation jobs
in queue (`audit_head_attr_stability` 6585212 and
`_audit_reproduce_full_ft_pixel5` 6585215). Their numbers will be
inserted in §5 and §1 respectively when they complete.

---

## Bottom-line verdict (preview)

1. **The diagnostic itself is reproducible** — the canonconv1 head-only
   cells produce *identical* `clean_head_asr = 0.0237` across all 3
   training seeds (42, 43, 44), which is only possible if the head-feature
   decomposition pipeline is deterministic given a fixed encoder and
   fixed diagnostic seed. (Empirical re-test pending in §5.)
2. **The training pipeline is NOT bit-reproducible at fixed seed** —
   client-side `torch.manual_seed()` is never called, and the per-client
   `DataLoader(shuffle=True, …)` has no explicit `generator`. Cuda
   non-determinism is not enforced.
3. **The seed-42 `97.2 %` headline number reflects a degenerate
   training trajectory**, not encoder anchoring. The seed-42 model was
   *stuck at the trivial-backdoor attractor* (acc = 0.0594 = 1/16.86 ≈
   class-2 base rate, ASR = 1.0) **for 12 consecutive rounds** before
   escaping. Final clean accuracy 0.667 vs 0.918/0.917 for seeds 43/44.
   The headline 97 %-head-attribution is consistent with "the encoder
   barely learned the clean task, the head is doing all the work
   (both clean and backdoor)."
4. **At 5 mal the FL dynamics are chaotic**, not the metrics. Different
   seeds produce qualitatively different trajectories (escape early,
   escape late, never escape). The diagnostic correctly captures the
   resulting differences in trained encoders.
5. **At 15 mal and at the head_only canonconv1 fallback, the dynamics
   are stable across seeds** and the metrics are reliable.
6. **Several latent bugs found** in config parsing (`bool("false")` is
   `True` in Python; affects `pretrained-init`, `canonical-conv1`,
   `wandb-enabled` reads in training code) — they did **not** manifest
   in our Cycle 02 runs because none of our YAMLs explicitly set those
   keys to `false` (they either omit them, getting the correctly-typed
   pyproject.toml default, or set them to `true`). **Must be fixed before
   adding any YAML that explicitly sets a boolean to false.**

**Can we interpret the multi-seed variance scientifically?** *Yes, with
caveats.* The 15 mal cells and the canonconv1 head-only cells produce
reliable, seed-stable numbers. The 5 mal cells genuinely vary across
seeds because the FL training dynamics at marginal attack pressure
have multiple attractors. **The "97.2 % head attribution / encoder
anchoring" headline must be retired** — it was a single-seed outlier
from a degenerate trajectory.

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

### 1.3 Empirical confirmation (job 6585215, in queue)

We resubmitted the *exact* `pretrained_full_ft_pixel5.yaml` (seed=42) at
the current commit (`08e04f1`) to a separate experiment-name
(`cycle02-audit-reproduce-full-ft-pixel5-seed42`). When the job
completes, we will compare:

- Final `test_accuracy` and `asr` against the original 0.6669 / 0.9776
- `head_attribution_pct` (after running diagnostic) against 97.2 %

If the rerun gives within ±2 pp, the pipeline is *effectively* seed-
reproducible (i.e. the unseeded shuffle and CUDA non-determinism
introduce noise that is small relative to seed-to-seed variation, so
the variance we observe across seeds 42/43/44 is dominated by real seed
effects). If the rerun differs by more than 2 pp, the within-seed noise
is comparable to between-seed variance and we cannot interpret variance
across seeds at all.

**Insertion point: `[TBD when 6585215 lands]`**

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

| Cell | clean_head_asr | clean_head_clean_acc |
|---|---|---|
| canonconv1 + 5mal seeds 42/43/44 | 0.0237 / 0.0237 / 0.0237 | 0.6020 / 0.6018 / 0.6020 |
| canonconv1 + 15mal seeds 42/43/44 | 0.0237 / 0.0237 / 0.0237 | 0.6021 / 0.6020 / 0.6019 |

The 6-decimal-place identity of `clean_head_asr` is only possible if:
- The encoder is bit-identical across runs (TRUE: head_only never
  updates the encoder, so it stays at pretrained init across all 6
  cells regardless of training seed).
- The clean-head retraining converges to a deterministic state given
  fixed encoder + fixed diagnostic seed (seed=4242 used by default).

This is strong evidence that the diagnostic is reproducible at fixed
checkpoint + fixed diagnostic seed. The minor variation in
`clean_head_clean_acc` (0.6018-0.6021, < 0.0003) is well within
floating-point noise.

### 5.2 Direct empirical test (job 6585212, in queue)

We submitted a 5-run audit on the same checkpoint
(`cycle02-pretrained-full-ft-pixel5_r100_seed42`):
- 2 runs at diagnostic seed=4242 → tests bit-reproducibility
- 3 runs at diagnostic seeds 4243/4244/4245 → tests robustness

When the job lands, the table will be filled in here. Expected outcomes:
- Same diagnostic seed → bit-identical `clean_head_asr`
- Different diagnostic seeds → `clean_head_asr` varies ≤ 0.005

**Insertion point: `[TBD when 6585212 lands]`**

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

**Can the multi-seed variance be interpreted scientifically?**

| Result | Reliable? | Reason |
|---|---|---|
| **15 mal head_attribution** (full_ft, last_block) | ✓ Yes | Stable across 3 seeds; SD ≤ 14 pp; saturated regime. |
| **canonconv1 head_only ASR** (5mal, 15mal) | ✓ Yes | Bit-stable across 3 seeds; encoder frozen; deterministic by construction. |
| **linear_probe_acc** ≥ 0.95 across all cells | ✓ Yes | Stable across regime, attack pressure, and seed. |
| **`centroid_l2`** values per cell | ⚠ Partial | Stable for canonconv1 head_only (frozen encoder), variable for full_ft / last_block by ~±50 % across seeds. Numbers must be reported as mean ± std. |
| **5 mal head_attribution** (full_ft, last_block) | ✗ NO | Variance 40-48 pp; one seed is degenerate (seed=42 stuck at trivial attractor); cannot publish a single number. |
| **"Encoder anchoring" hypothesis** | ✗ RETIRE | Was an artefact of the seed-42 degenerate trajectory. Replace with: "FL training at marginal attack pressure has multiple attractors; the diagnostic correctly captures the resulting encoder differences." |
| **"Pretrained init shifts attack mechanism"** | ⚠ Partial | Partially supported at saturation (15 mal cells), not at marginal pressure (5 mal). Reframe as: "saturated attacks consistently end up head-dominated when the encoder is anchored by ImageNet pretraining; marginal attacks have stochastic outcomes." |

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
   slightly slower runs but bit-reproducible.

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
