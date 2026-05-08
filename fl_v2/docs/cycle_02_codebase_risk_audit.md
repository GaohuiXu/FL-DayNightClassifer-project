# Cycle 02 Codebase + Logic Risk Audit (2026-05-08)

A systematic risk inventory across the FL pipeline, attack design, data
partition, model architecture, evaluation methodology, and statistical
framing. Each finding is rooted in actual file:line evidence or measured
data, not abstract concern.

The goal is the question: *if a tough venue reviewer (NDSS / S&P / USENIX
/ ICCV) read our paper draft, what would they reject the work over?*

Findings are grouped by **severity**:

- **CRITICAL** — directly invalidates current paper claims or makes them
  un-defensible. Must be addressed before any submission.
- **HIGH** — a careful reviewer will flag this; weakens the work
  significantly and may force a major-revision on its own.
- **MEDIUM** — should be addressed but not fatal; can be discussed as a
  limitation.
- **LOW** — methodological hygiene; address if time permits.

---

## CRITICAL

### C1. Same-(commit, seed) within-commit reproducibility is NOT established

**Evidence:** `cycle02-fixed-full-ft-pixel5_seed42` re-run as
`cycle02-reprocheck-full-ft-pixel5_seed42` (job 6600759) on the same
training source (only commit `87fa3e6`, a one-line shell-only fix,
between 6599453 and 6600759). Final round 100:

| run | acc | ASR |
|---|---|---|
| 6599453 (original) | 89.10 % | 81.44 % |
| 6600759 (reprocheck) | 89.74 % | 86.59 % |
| Δ | +0.64 pp | **+5.15 pp** |

Peak intermediate Δ ASR ≈ 6 pp at round 50. Audit's own
6594906 / 6594907 already documented bit-identical rounds 0-5 followed
by round-6+ divergence. The seven audit fixes are **not** sufficient
to make the trained model bit-reproducible at fixed (commit, seed).

**Why it matters:** every per-cell number we report is one realisation
of a stochastic pipeline with an unmeasured ε. The 5 pp ASR is much
smaller than the seed-to-seed SD (17.6 pp v1 / 21.0 pp v2 on full_ft
5mal) so the *direction* of comparisons (saturation, cell ordering) is
likely robust, but **point numbers cannot be quoted as if precise.**

**Action:** see `cycle_02_pivot_audit.md` §9.6 + the new-session
prompt drafted on 2026-05-08 for the full residual-ε investigation
(read Flower internals, Ray actor scheduling, hardware determinism).
Until that lands we cannot claim "deterministic FL pipeline".

---

### C2. Pretrained-init pivot has a random-init `conv1` for modified-conv1 cells

**Evidence:** `fl_v2/src/fl_v2/models/resnet.py:32-45`. With
`pretrained=True` and `canonical_conv1=False` (the default for
full_ft / last_block / head_only at 32×32 — i.e., 8 of the 9 Cycle 02
cells), `resnet18(weights=ImageNet)` is loaded, then `base.conv1` is
**replaced with a freshly-initialized 3×3 stride-1 Conv2d**. ImageNet
weights survive only for `bn1`, `layer1-4`, `avgpool`. The first
convolution is random-init.

For each trainable-layers mode this means:

| mode | conv1 | bn1+layer1-3 | layer4 | fc | Coherent encoder? |
|---|---|---|---|---|---|
| `full_ft` | random-init, **trained** | ImageNet, trained | ImageNet, trained | random-init, trained | conv1 is trained from scratch — partly defeats the "pretrained" framing |
| `last_block` | random-init, **frozen** | ImageNet, frozen | ImageNet, trained | random-init, trained | random-init frozen conv1 feeds ImageNet-pretrained mid-blocks → architecturally inconsistent encoder |
| `head_only` (modified-conv1) | random-init, **frozen** | ImageNet, frozen | ImageNet, frozen | random-init, trained | random-init frozen conv1 → encoder is broken; this is why early head_only with modified-conv1 produced clean_acc ≈ 36 % |

The `canonconv1` cells (`canonical_conv1=True`) preserve ImageNet's
7×7 stride-2 conv1 + maxpool, giving a coherent pretrained encoder —
but require 64+ image size. Only the canonconv1 head_only cells used
this, none of the other Cycle 02 cells did.

**Why it matters:** the headline "pretrained vs from-scratch attack
susceptibility" comparison is muddied. Our "pretrained" full_ft model
trains conv1 from scratch (1.7k params); on the timescale of 100 FL
rounds × 50 clients × 3 local epochs, conv1 likely converges to
near-random ImageNet-equivalent low-level filters anyway, so the
*real* delta vs from-scratch is in bn1 + layer1-4 init only. A
reviewer will ask: *"is this a clean pretrained-vs-from-scratch
ablation, or is it bn-and-residual-block-pretraining only?"* The
honest answer is the latter.

**Action:** for any venue submission, run a single-cell ablation
where conv1 is also pretrained (use canonconv1 mode + image-size
upsampled to 64). Report Cycle 02 numbers explicitly as
"bn1+layer1-4-pretrained, conv1+fc-random-init". Do not call this
"pretrained ResNet18" without qualification.

---

### C3. Attack target = class 2 ("Speed limit 50") is the easiest possible target

**Evidence:** measured from `*_client_label_histograms.json` for the
seed=42 run.

- Class 2 is **tied for #1 most-represented class** in GTSRB train
  (1500 / 26640 = 5.63 %, tied with class 1 = Speed-30).
- Top-5 are all very-frequent classes (Speed signs + Yield + Priority
  road + Keep right).
- Class 2 = "Speed limit (50 km/h)" — a **non-safety-critical**
  misclassification (50 → 30 / 60 / 70 are the natural confusions).

The model has a strong natural prior toward predicting class 2; an
attack pushing triggered samples toward class 2 does less work than
pushing toward an under-represented or safety-critical class. The
high-ASR ceilings (~80–95 %) we see are partly an artefact of target
selection.

For the **autonomous-driving security framing** that drives this
thesis, the relevant targets are:

- **Class 14 = "Stop"** — most safety-critical AD-perception backdoor
  target. Eykholt et al. and most physical-world AD-trigger papers
  attack Stop specifically.
- Class 13 = "Yield" — also safety-critical.
- Class 17 = "No entry" — also safety-critical.
- Class 27 = "Pedestrians" — under-represented (~210 samples / 0.79 %),
  ASR ≥ 0.90 against this would be a genuinely difficult attack.

**Why it matters:** the supervisor's question "does our attack target
make sense?" is sharp. Targeting class 2 ⇒ ASR numbers are inflated
relative to a thesis-relevant attack target, and the safety narrative
("FL systems for AD must defend against backdoor attacks") loses
force when our actual attacks change "Speed-50" → "Speed-50".

**Action:** before any cross-cell scientific claim, run a target-class
ablation: at least one cell × {class 2 (current), class 14 (Stop),
class 27 (Pedestrians)} × N seeds. Report ASR + head_attribution
broken down by target. The thesis chapter should then standardize on
class 14 (Stop) as the "thesis target" while keeping class 2 as the
"easy / Cycle 01 baseline" reference.

---

### C4. Partition seed = training seed (variance components confounded) — REMEDIATED 2026-05-08

**Evidence:** `fl_v2/src/fl_v2/server_app.py:325-336`.
`build_client_index_map_with_stats(seed=seed)` is called with the
YAML's `seed` field — the same value used to seed the model RNG, the
data loader generator, and the per-client `derive_seed`. There is no
separate `partition-seed` parameter.

**Why it matters:** the seed-to-seed variance reported across our
3-seed waves (full_ft 5mal v1 SD = 17.6 pp, v2 SD = 21.0 pp) is the
**combined variance** of:

1. Different model RNG (training trajectory under the same partition).
2. Different per-client class distributions (each seed gives a
   different Dirichlet draw, so the same client ID 0 has different
   data and different class coverage on different seeds).
3. Different malicious-coalition target-class coverage. At seed=42
   the 5 malicious clients hold 73 / 1500 (4.87 %) of all class-2
   samples, with one client (client 1) holding **zero** class-2
   samples. At seed=43 / 44 the coalition's class-2 access will be
   different in unmeasured ways — a likely contributor to why
   seed=43's ASR (0.5243) was so different from seed=42's (0.8144).

These are not separable in the current setup. A reviewer will ask:
*"Is your seed-to-seed variance about model training noise or about
data-distribution noise? If you can't disentangle them, you cannot
report 'mean ± SD across seeds' as a property of the attack."*

**Action:** add a `partition-seed` YAML key that defaults to `seed` for
backward compatibility. Run two ablations:

- Fixed partition-seed × varying training seed → "training-noise SD".
- Varying partition-seed × fixed training seed → "partition-noise SD".

Report both decomposed; the SD we currently report is their sum.

**Code change landed (2026-05-08):**

- `pyproject.toml` — `partition-seed = ""` added to default config; empty
  string means "fall back to `seed`" (backward-compatible with all
  pre-existing YAMLs).
- `server_app.py` — reads `partition-seed`, falls back to `seed`,
  passes to `build_client_index_map_with_stats`. Logs the resolved
  value at server startup.
- `client_app.py` — `_resolve_partition_seed()` helper; partition cache
  key includes partition-seed; `get_client_dataloaders` receives both
  `seed` (model-side) and `partition_seed` (data-side).
- `data/dataset.py::get_client_dataloaders` — new optional
  `partition_seed` param; defaults to `seed`; routes data-side
  decisions (val split, poison mask, label-flip mask) through
  `partition_seed` while keeping the DataLoader shuffle generator
  on the model `seed`.

Variance-decomposition ablation pattern (see
`configs/experiments/cycle_02/phaseD2/_partition_seed_decoupling_example.yaml`):

| Run | partition-seed | seed | Measures |
|---|---|---|---|
| A1 | 42 | 42 | (joint baseline; same as old) |
| A2 | 42 | 43 | model-RNG variance only |
| A3 | 42 | 44 | model-RNG variance only |
| B1 | 42 | 42 | (joint baseline; reused) |
| B2 | 43 | 42 | partition variance only |
| B3 | 44 | 42 | partition variance only |

`Var(joint) ≈ Var(model_rng) + Var(partition) + interaction`. Decomposed
SDs let us state which component dominates the seed-to-seed spread
(currently reported v2 SD on full_ft 5mal = 21 pp).

---

## HIGH

### H1. FL test-set leakage: server uses official GTSRB test for round-by-round eval

**Evidence:** `fl_v2/src/fl_v2/server_app.py:189-195`. The server's
`testloader` is `get_global_testloader(...)` which loads
`GTSRB(split="test")` — the official held-out split. Server-side
eval runs after every round and produces `server/test_loss`,
`server/test_accuracy`, `server/asr` as the per-round wandb metrics.

There is no separate validation set on the server side. Per-client
val sets exist (`val-ratio: 0.1`) but are not aggregated for global
monitoring — they're used only for client-local val_loss /
val_accuracy reported in `client_val_*` metrics.

**Why it matters:** standard FL papers report final-round metrics on
the test set. We do that too (the Phase 3.1 numbers all come from
round-100 test eval). But during training we are also looking at
round-by-round test_loss / test_accuracy / asr. If any
hyperparameter — LR, num_rounds, defense-type, attack-trigger
position — was ever tuned by inspecting these per-round curves, we
have **test-set leakage**. The 100-round budget itself looks like it
was chosen because clean acc plateaus around round 75-80 on the test
set, which (if true) is hyperparameter selection on the test set.

**Why it's HIGH not CRITICAL:** the literature in FL benchmark papers
routinely does this (test-set-as-monitoring is the default in
Flower's quickstart examples). A reviewer at S&P / NDSS will
absolutely flag it; a reviewer at a more applied FL venue may not.

**Action:** create a **server-side validation set** by holding out
e.g. 1000 samples from the GTSRB test split, using those for
round-by-round monitoring, and reserving the remaining ~11 600 test
samples for final evaluation only. Document the split in
`docs/representation_space_framework.md`. This is a minimal
refactor in `data/dataset.py::load_gtsrb_test_dataset` to support
a `holdout_for_val` argument.

---

### H2. Trigger applied AFTER augmentation (idealized attack)

**Evidence:** `fl_v2/src/fl_v2/attacks_defenses/attacks/pixel_backdoor.py:125-135`.
`PixelBackdoorDataset.__getitem__` retrieves the post-transform
tensor from `self.base_dataset[idx]` (which already applied
`get_train_transforms`: Resize → RandomRotation → ColorJitter →
ToTensor → Normalize) and **then** stamps the trigger via
`_stamp_trigger`. The trigger pixels never see RandomRotation /
ColorJitter; they're always at exactly `(h-4..h, w-4..w)` with
exactly value 1.0 (= pure white in normalized [-1, +1] space).

**Why it matters:** a real-world AD trigger (sticker, decal, paint
patch) would jitter under camera rotation, lighting, weather, and
position. Training the model with a perfectly-stable trigger gives
it a brittle but high-ASR backdoor. Reviewers expect at minimum:

- Trigger-aware augmentation (apply augment AFTER trigger so the
  trigger also gets jittered; the model learns a robust trigger
  pattern, the attacker requires more diverse poisoning).
- Or: physical-world threat model (Eykholt-style), with augmentation
  parameters chosen to simulate camera distance / angle.

Our current setup is the BadNets-2017 setting — it works as a
benchmark but is well-known to be unrealistic.

**Action:** for the thesis chapter, run a single ablation: trigger
applied BEFORE augmentation vs after. Report Δ ASR. If the
robust-trigger setting still hits ASR ≥ 0.5 with 5 malicious
clients, the attack is more credible. If it collapses, this is
itself a finding worth reporting.

---

### H3. `poison-fraction = 0.5` is high; trigger-size 4 is large

**Evidence:** YAMLs set `poison-fraction: 0.5` (50 % of each
malicious client's local data is poisoned + relabeled to target)
and `trigger-size: 4` on a 32×32 image (= 16 pixels = **1.56 %** of
image area).

For comparison:

- BadNets (Gu et al. 2017): poison-fraction 0.05–0.1, trigger 3×3
  on MNIST 28×28 = 1.15 % area. Smaller and rarer.
- Most FL backdoor papers: poison-fraction 0.1–0.3 on the malicious
  clients' local data.

Setting poison-fraction = 0.5 means the malicious client's local
training distribution is **half clean / half poisoned**. The
malicious client's local-step gradient is dominated by the backdoor
loss; the local model after 3 epochs is essentially a trigger-class
classifier with some clean accuracy bolted on.

**Why it matters:** the reported ASR (~80–95 %) is partly because
the poisoning rate is aggressive. Reviewers will ask: *"what's the
weakest poisoning rate at which your attack still succeeds?"* That's
a standard ablation we don't currently run.

**Action:** add a `poison-fraction` ablation: {0.05, 0.1, 0.2, 0.5}
× one cell (full_ft 5mal). Report attack-success curve. The
"realistic" attack is usually 0.1; reporting only at 0.5 is
selecting on attack strength.

---

### H4. Statistical sample size: N=3 seeds per cell is below venue norms

**Evidence:** Phase 3.1 wave is 3 cells × 3 seeds. v2 rerun is
3 cells × 3 seeds. The full_ft + 5mal cell shows v1 SD = 17.6 pp,
v2 SD = 21.0 pp on the per-cell head_attribution.

**Why it matters:** with N=3 and SD ≈ 20 pp on a 0–100 % metric,
the 95 % CI on the mean is roughly ±25 pp (using Student-t with
2 dof). Any cross-cell comparison whose mean Δ is < 25 pp is
statistically indistinguishable from zero. Specifically:

- Cycle 02 v2 mean (43.3 %) vs sentinel v2 (18.2 %) on full_ft 5mal:
  Δ = 25.1 pp, t-test would not reject H₀ at N=3.
- Cycle 01 from-scratch reference 58 % vs full_ft v2 mean 43.3 %:
  Δ = 14.7 pp; well below the CI half-width.

Most venues require N ≥ 5 for stochastic-attack means.

**Action:** budget for N=5 seeds on the chaotic cells (full_ft 5mal,
both Cycle 01 sentinel and Cycle 02 pretrained). The saturated cells
(canonconv1 head_only 15mal, SD 2.7 pp) are fine at N=3.

---

### H5. `fraction-train: 1.0` (every client every round) is unrealistic

**Evidence:** `pyproject.toml:39` and all our YAMLs.
`fraction-train=1.0` means all 50 clients participate in every one
of the 100 rounds. That's 5 000 client-rounds without a single
dropout.

**Why it matters:** real FL deployments (mobile keyboards, federated
medical devices, AD fleets) have device unreliability — typical
fraction-train is 0.05–0.3. With fraction-train=1.0 the malicious
clients participate every round, which inflates both attack success
and defense difficulty. A reviewer will ask: *"what happens at
fraction-train=0.1 — do your defenses still hold? does the attack
still install?"*

This is also relevant to the malicious-coalition question. If only
10 % of clients participate each round (5 of 50), and the malicious
clients are 5 of 50, then on any given round there's only a ~50 %
chance any malicious client is selected. That changes the attack
dynamics fundamentally.

**Action:** add a `fraction-train` ablation: {0.1, 0.3, 1.0} × full_ft
5mal cell. Report attack-success curves vs round.

---

## MEDIUM

### M1. Trigger value 1.0 in normalized space is "super-bright" relative to typical white pixels

**Evidence:** `Normalize(mean=0.5, std=0.5)` maps pixel [0, 1] to
normalized [-1, +1]. Setting normalized value = 1.0 corresponds to
**pre-normalization pixel value = 1.0** (pure white at 255 / 255).
Mathematically correct, but worth noting: most natural GTSRB images
do not have full-saturated white patches because of camera ISP
processing. The trigger is not just "white" — it's "the brightest
patch the camera ever produces", which makes it unusually
detectable for a defender.

**Action:** consider a trigger-value ablation. Set trigger-value to
e.g. 0.7 (in pre-norm space) so it's a plausible "white sticker
under daylight" rather than "burned-out highlight".

### M2. No client-side dropout / failure simulation

**Evidence:** `min-train-nodes: 2` in the YAML, but with
fraction-train=1.0 this is moot. The simulation doesn't model
client unavailability, network failures, partial submissions.

**Action:** out of scope for thesis Cycle 02 but worth noting as
a limitation in the related-work discussion.

### M3. weight-decay = 0.0

**Evidence:** `pyproject.toml:66`. Optimizer is SGD without weight
decay. For a ResNet18 trained 100 rounds × 50 clients × 3 local
epochs (≈ 15 000 effective epochs of optimizer steps, though over
small per-client batches), no weight decay is unusual. Standard
ImageNet ResNet18 uses 1e-4.

**Why it matters:** weight decay affects feature norms, which
affects head_attribution (the diagnostic uses normalized features
implicitly via the linear probe). Different weight-decay settings
could produce different head_attribution numbers without changing
attack success.

**Action:** add a wd ablation: {0, 1e-4, 5e-4} × one cell. If
head_attribution moves significantly, document it. If not,
the diagnostic is robust to this hyperparameter.

### M4. LR schedule cosine to lr-min=0.0001 is aggressive

**Evidence:** LR starts at 0.05 and cosines down to 0.0001 over
100 rounds. By round 80 the LR is ~0.001, by round 100 essentially
zero. The malicious clients' attack injection therefore has the
most leverage in the early rounds (rounds 1–25) when LR is
high. This is consistent with the audit's finding that "attack
escapes the trivial-backdoor attractor around round 6–10". A
reviewer will probably accept this as standard practice but it's
worth documenting that ASR trajectory is heavily LR-schedule-
dependent.

### M5. Diagnostic uses full clean GTSRB train (not per-client)

**Evidence:** `fl_v2/analysis/head_feature_decomposition.py:486-489`.
The fresh head is trained on the **full clean GTSRB train split**
(26 640 samples) with no FL partition. This is the right choice
for measuring "what the encoder represents" but is structurally
unavailable to a real defender — a real client only has its
non-IID Dirichlet partition.

This is documented (the user already noted "we cannot use this
metric as a solid measurement; we can learn useful information
from it") but the doc should be explicit: head_attribution
is an **encoder-level diagnostic from a god-eye view**, not a
deployable defense metric.

### M6. Two of three canonconv1 head_only seeds are bit-identical, third differs at 4th decimal

**Evidence:** measured precise values. Direction is consistent
with "frozen encoder + same diagnostic seed → near-deterministic"
but not literally bit-identical. Already corrected in
`cycle_02_pivot_audit.md` §5.1 (commit `0381960`).

---

## LOW (cleanup / polish)

### L1. `wandb-tags` strings drift across YAMLs

Cycle 02 YAMLs use various tag combinations
(`pretrained-pivot,phase3-fixed`, `phase3-sentinel,cycle01-rerun`, ...).
Group by phase / commit for cleaner wandb filtering.

### L2. No checkpoint at round 0 for "starting point" reference

`checkpoint-rounds: "0,5,10,25,50,75,100"` lists round 0 but the
saved checkpoint at round 0 is the **post-strategy-init** state,
not the pure model init. Worth a sentence in the docs.

### L3. Diagnostic cosine LR schedule uses `T_max=epochs` (the max budget)

`_train_fresh_head` schedules cosine annealing across the full
100-epoch budget even though we may early-stop at epoch 30–60. The
LR is therefore "cosine to LR(50/100)" not "cosine to zero". Not
wrong, but the effective LR schedule depends on the early-stop
point. Document.

### L4. `wandb-enabled: true` is effective default; explicit at-no-cost

Verified runs always log to wandb online. No `wandb-enabled: false`
mode tested for whether disabling wandb affects determinism (one of
the open hypotheses for the residual ε).

### L5. No GTSRB class-name lookup table in the code

`docs/scripts_guide.md` (or a new file) should include the GTSRB
43-class index → name mapping so future sessions don't have to look
up "class 2 = Speed-50" externally.

---

## What survives all of these

1. **The framework metrics architecture** (4-axis profile, linear
   probe, centroid, spectral, dynamics) is a clean diagnostic toolkit
   that can be applied to ANY Cycle 02 / Cycle 03 attack regardless
   of the issues above.

2. **The audit + reliability work** (7 fixes, identification of the
   residual ε, reprocheck verification) is correct *as a methodological
   contribution* — it just doesn't reach the "deterministic" bar yet.

3. **The infrastructure** (cycle-aware fl_outputs layout, wandb
   integration, SLURM submission flow, dependency-based job chaining,
   Phase 3.0 sentinel pattern) is solid and ready for Cycle 03 work.

4. **The pivot's high-level direction** (pretrained vs from-scratch ×
   trainable-layers × attack-pressure) is scientifically valuable;
   it just needs the C2 / C3 / C4 fixes to be defensible.

---

## Recommended sequencing for cleanup before Cycle 03 starts

Given thesis time pressure, my recommended order is:

1. **Spawn the residual-ε investigation session** (new-session prompt
   already drafted). Until we know whether the ε is fixable, the
   reproducibility framing of every other claim is uncertain.
2. **Decouple partition-seed from training-seed** (C4). One YAML key,
   one server_app.py change, ~15 lines total.
3. **Add a server-side validation split** (H1). One change in
   `data/dataset.py`, server_app uses val for round-monitoring,
   test only for final eval. ~20 lines.
4. **Run target-class ablation** (C3): full_ft 5mal × {class 2,
   class 14, class 27} × N=5 seeds = 15 runs at ~1.5 h each ≈ a day.
5. **Run poison-fraction ablation** (H3): full_ft 5mal × poison
   ∈ {0.05, 0.1, 0.2, 0.5} × N=3 seeds = 12 runs. Half a day.
6. **Document conv1 random-init explicitly** (C2). Add a comment in
   the model docstring + a paragraph in `cycle_02_pivot_results.md`.
   For the chapter itself, run one canonconv1+image-size-64 cell as
   the "clean pretrained" reference.
7. **Trigger-after-augment ablation** (H2): one cell, one seed,
   compare ASR.
8. **Bump to N=5 seeds on the chaotic cells** (H4). Compute cost: 4 h
   per cell × 2 cells × 2 extra seeds = ~32 h.

Items 2-3 together unblock honest reporting of variance bands.
Items 4-5 establish the attack-design defensibility floor. Items 6-7
are small but necessary for venue readiness.

The total compute budget for items 4-8 is roughly 2 days of A40 time,
modest given the stakes.
