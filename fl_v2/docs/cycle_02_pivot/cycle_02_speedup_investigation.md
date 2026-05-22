# Why the unified-fix configuration is slow + how to speed it up

After landing the partition-id sort fix and locking the
single-actor (`num-gpus=1.0`) regime as the bit-deterministic
configuration, V4 confirms:

- ✓ **Determinism** — same seed, same code, same `summary.json`,
  same `rounds.csv`, same `round_0025.pt` SHA-256 across two
  independent runs.
- ✗ **Wallclock**: 101–111 s/round (single-actor) vs ~60 s/round at
  the old non-deterministic 0.10/0 configuration → **~1.7× slower**.
- ✗ **GPU utilisation**: 22 % mean (Alvis warns at < 30 %).

This document breaks down where the 100 s/round goes, identifies the
dominant cost, and ranks remediation candidates.

---

## 1. Where the time goes (V4 paired data, 25 rounds, jobs 6605728/6605729)

GPU profiler CSV (5-s sampling, 506 samples = 42.2 min):

| SM-util bucket | Time | Fraction |
|---|---|---|
| 0–5 %        | 18.8 min | **45 %** |
| 5–10 %       |  3.8 min |  9 % |
| 10–20 %      |  2.1 min |  5 % |
| 20–40 %      |  5.8 min | 14 % |
| 40–60 %      |  5.4 min | 13 % |
| 60–80 %      |  6.1 min | 14 % |
| 80–100 %     |  0.1 min | 0.2 % |
| **median SM** | **6 %** | |
| **P95 SM**   | **67 %** | |

Key facts:

- **Median SM utilisation is 6 %.** The GPU is essentially idle half
  the time.
- **The longest single >= 80 % stretch is 5 seconds.** The GPU is
  *never* sustained-saturated.
- **14 inter-round idle stretches, mean 33.6 s, total 7.8 min**
  (≈ 18 s per round). These match the 25 round transitions plus
  startup. Inter-round overhead is server-side (test eval +
  aggregation + broadcast + wandb).
- **In-round time = total wallclock − inter-round** = 42.2 −
  7.8 = 34.4 min over 25 rounds = **82 s/round of in-round work**.
  Inter-round is ~18 s/round.

### Per-client time accounting (single-actor, 50 clients sequential)

In-round time / 50 clients = **~1.65 s per client**.

Decomposed:

| Component | Estimated time/client | Source |
|---|---|---|
| `create_model(...)` (build fresh ResNet18 architecture, allocate 11 M random-init params) | ~80 ms | `_load_model_from_message` calls this on every train() invocation |
| `model.load_state_dict(...)` (deserialise + copy 11 M floats from `Message`) | ~300–400 ms | torch state-dict load over an ArrayRecord |
| `model.to(device)` | ~50–100 ms | first move per call |
| `train_local` (actual training: 30 batches × 3 epochs × ~10 ms/batch) | ~300 ms | of which only ~150 ms is GPU compute |
| Reply build + serialisation (state_dict → ArrayRecord → Message) | ~300 ms | reverse of the load |
| Ray task overhead (dispatch, ack, message passing) | ~100–200 ms | per call, observed in Ray docs |
| **Per-client total**          | **~1.5–1.7 s** | matches 1.65 s observed |
| **GPU active fraction per client**  | **~150 ms / 1.65 s = 9 %** | |

So **~91 % of per-client wallclock is overhead, not GPU compute.** The
overhead is dominated by:

- Building a fresh model architecture from scratch each call (`create_model` allocates and initialises ~11 M params before throwing them away)
- Serialising/deserialising 11 M-param state_dicts back and forth via Flower's `Message` (44 MiB per direction × 50 clients × 2 directions = 4.4 GiB of in-memory copies per round)
- Ray task dispatch/return overhead

### Total per-round budget

```
per-round wallclock ≈ 50 × 1.65 s (clients)  +  18 s (inter-round)  +  ~5 s (setup)
                    ≈ 82 + 18 + 5  =  105 s        — matches measured 101–111 s
```

GPU is genuinely busy for only **50 × 0.15 s = 7.5 s/round** = 7 %
of wallclock. The 22 % SM util reading from the profiler is
inflated by data-loading periods that show ~10–20 % util but no
real compute.

---

## 2. Why our previous theoretical estimate was wrong

The earlier investigation doc estimated **~2 ms GPU + ~100 ms CPU
augment per batch** based on per-batch timing. That was correct
*per batch*, but ignored:

- **Per-client setup cost** (~1 s/client, dominant)
- **Ray task overhead** (~100-200 ms/client)
- **State-dict serialisation cost** (~600 ms/client total, both directions)

Multiplied by 50 clients/round, these dominate the per-batch
analysis. The "num-workers=4 → 4× faster" theory was only true if
the per-batch CPU work were the bottleneck, but it's not — the
per-client setup cost is.

That's why num_workers=4 improved utilisation from 6 % → 22 %
(removing some of the per-batch CPU stalls) but didn't deliver
the predicted 2-4× wallclock speedup.

---

## 3. Remediation candidates, ranked

### R1. Cache the model architecture per Ray actor (BIG win, low risk)

**Idea**: instead of `create_model(...)` + `load_state_dict(...)` on
every train call, lazily construct the model ONCE per actor and
only `load_state_dict(...)` on subsequent calls. Same parameters,
same final state, just no rebuild.

**Why it works**: `create_model()` allocates and initialises 11 M
parameters from scratch. Those values are immediately overwritten
by `load_state_dict(...)` so the random-init work is wasted. With
~80 ms saved per client × 50 clients/round × 25 rounds = ~100 sec
per V4 run.

**Implementation**: add a module-level `_model_cache: dict[(model_type,
canonical_conv1), nn.Module] = {}` next to the existing
`_client_data_cache` and `_index_map_cache`. In
`_load_model_from_message`, check the cache first. The cached model
can be safely re-used across clients because the load_state_dict
overwrites all weights.

**Risk**: shared model means clients in sequence share GPU memory
allocations. Should be fine — the actor is single-threaded and
processes clients sequentially. Determinism preserved (the
state_dict load deterministically overwrites every parameter).

**Code change scope**: ~20 lines in `client_app.py`.

**Expected wallclock gain**: ~5-15 % (small but free).

### R2. Increase `batch-size` 64 → 128 or 256 (BIG win, science neutral)

**Idea**: each forward+backward kernel is ~5–10 ms on A40 at
batch=64. At batch=256, the same kernel does 4× the work in
~12–15 ms. Per-client: 30 batches × 6 ms = 180 ms, vs 7.5 batches
× 13 ms = 100 ms. So both per-client GPU time AND the number of
optimizer steps drop.

**Why it works**: ResNet18 on A40 is heavily under-batched at 64.
The GPU has 10,752 CUDA cores; a 64×3×32×32 batch uses a tiny
fraction of them. A40 saturates around batch=512+ for this
model size.

**Risk**: fewer optimizer steps per epoch (with same num_local_epochs)
means slightly different gradient noise, but for 3 local epochs
this is barely measurable. **Slightly different SGD dynamics →
the bit-identity claim still holds (same data + same batch order +
same params); the *numerical results* may differ from the current
run with batch=64.** Need to re-baseline if we change.

**Code change scope**: change `batch-size: 64 → 128` in
`pyproject.toml` (one line).

**Expected wallclock gain**: ~30–50 % (per-client GPU time roughly
halves; CPU side parallelism keeps up).

**Expected GPU SM util**: rises to ~40-60 % (more work per kernel).

### R3. Reduce per-call Python overhead in `@app.train`

**Idea**: the train() callback does many `int(run_config.get(...))`
lookups, pretty-prints, etc., at the top. Pull these out into a
module-level constant or onto the actor's local state.

**Code change scope**: ~30 lines refactor in `client_app.py`.

**Expected wallclock gain**: ~5 %. Marginal but free.

### R4. Use GPU-side augmentation (`kornia`) — eliminate CPU bottleneck

**Idea**: replace `RandomRotation` + `ColorJitter` (PIL/CPU) with
`kornia.augmentation` (CUDA kernel). The augmentation runs on the
GPU as part of the same kernel pipeline as the forward pass.

**Why it works**: removes the CPU bottleneck entirely. With
augmentation on GPU, num_workers can drop to 0 (less memory) and
per-batch wallclock = pure forward+backward.

**Risk**: changes augmentation slightly (PIL bilinear vs kornia
bilinear may differ in edge handling). Determinism preserved if we
use kornia's deterministic mode.

**Code change scope**: ~50 lines + new kornia dependency. This is
the biggest change but also the cleanest end state.

**Expected wallclock gain**: ~30–50 %.

**Expected GPU SM util**: ~50-70 % (no more CPU stalls).

### R5. Increase `num-local-epochs` 3 → 5 or 8 (gradient amortisation)

**Idea**: each Ray task already pays ~1 s of overhead. With 3 local
epochs, only ~0.3 s of GPU work is amortised over that overhead.
With 8 local epochs, ~0.8 s of GPU work, much better
overhead:work ratio.

**Risk**: changes FL convergence dynamics — more local steps before
averaging means more divergent local models, slightly worse global
convergence per round. Standard FL papers use 1-5 local epochs.
This is a methodological change.

**Code change scope**: 1 YAML field. Not a code change.

**Expected wallclock gain**: per-round wallclock barely changes; per-
**total-rounds-needed** drops 1.5-2×. Net training time drops.

### R6. Reduce `fraction-train` 1.0 → 0.1 (realistic + faster)

**Idea**: only 10 % of clients participate per round. Per-round
work drops 10×.

**Risk**: changes science — different FL regime. But also more
realistic (no real FL has 100 % participation per round). This
ties into risk-audit H5.

**Code change scope**: 1 YAML field.

**Expected wallclock gain**: ~10× per round, but more rounds may
be needed to converge. Net 2-5× faster.

### R7. Disable wandb online during training (`wandb-mode: offline`)

**Idea**: wandb's online sync runs a sentry thread that periodically
serialises and uploads metrics. This adds latency between rounds.

**Code change scope**: 1 YAML field.

**Expected wallclock gain**: ~5 %. Minor.

---

## 4. Recommended sequencing

The cheapest, lowest-risk path:

1. **R1** (model cache) — strictly free win, ~10 % wallclock improvement.
2. **R2** (batch_size 64 → 128 or 256) — biggest single win,
   ~30-50 % wallclock improvement, GPU util to ~40-60 %, science
   essentially unchanged. **Re-baseline V4 to confirm
   bit-determinism still holds.**
3. **R5** (num_local_epochs 3 → 5) — methodological knob that's
   well-supported in FL literature. Discuss with supervisor before
   adopting.
4. **R4** (kornia GPU augment) — biggest structural improvement;
   defer until R1+R2 are landed and we know what the residual
   wallclock looks like.

Items 1+2 alone should bring per-round wallclock to **40–60 s** and
GPU util to **40-50 %** — closing both the speedup gap (matches the
old non-deterministic config) AND lifting Alvis' efficiency check.

---

## 5. What we will NOT touch

- **R6 (fraction-train < 1.0)** — that's a scientific decision, not
  a GPU-efficiency hack. Risk-audit H5 already flags it as a
  separate ablation.
- **fp16 / mixed precision** — re-introduces non-determinism we
  just closed. Defer until the determinism story is settled.
- **Custom Ray actor pool tuning** — too invasive; the partition-id
  sort fix already lets us live with single-actor.

---

## 6. Sanity check — comparison with the old non-deterministic config

The old 0.10/0 (5 actors, no workers) achieved ~60 s/round at ~6 %
SM util. Per round that's:

- 5 actors × 10 clients sequential × 1 s overhead = 50 s of overhead
- + 10 s of GPU work scattered across the actors
- = 60 s/round, ~10 % active

So the multi-actor regime parallelised the per-client *overhead*
(model rebuild, Ray task setup) across 5 processes. The
single-actor regime can't parallelise that. The unified-fix only
parallelised CPU augment (a smaller component), which is why the
total wallclock is worse despite higher SM util.

R1 attacks the per-client overhead directly; R2 attacks the
per-batch GPU under-utilisation. Together they should beat the old
config on wallclock AND beat Alvis' threshold AND remain
bit-deterministic.

---

## 7. UPDATE 2026-05-09: V4 paired test of R1+R2 — neutral, not the win we expected

After committing R1 + R2 (ee5d734) we ran a paired V4 test
(jobs 6606100 / 6606101 at the new config). Results:

### V4-A determinism — PASS

  acc=0.4231987332  asr=0.1787878788  target=0.7466666667 (both runs)
  rounds.csv : IDENTICAL
  round_0025.pt SHA-256 : a2ee5e95... (both runs)

Bit-identity holds, even across different physical A40 GPUs (runA on
alvis7-11, runB on alvis5-08). R1+R2 do not break determinism.

### V4-B speedup — 0 % wallclock improvement

| Config | Per-round wallclock | Total / 25 rounds |
|---|---|---|
| Old non-deterministic 0.10/0          | ~60 s  | ~25 min |
| Pre-R1+R2 unified (1.0/4, batch=64)    | ~106 s | ~44 min |
| **R1+R2 (1.0/4, batch=128, model cache)** | **~104 s** | **~43 min** |

R1+R2 saved essentially nothing.

### V4-C GPU util — slightly worse

| Config | Mean SM util | Time <10 % util |
|---|---|---|
| Pre-R1+R2 | 22 % | 54 % |
| **R1+R2** | **17 %** | **68 %** |

### Why R1+R2 missed

The earlier per-client time decomposition was directionally right
(91 % overhead, 9 % GPU compute) but wrong about *which* overhead
dominates. Two corrections:

1. **`create_model()` is faster than 80 ms.** Empirically it's
   ~10-20 ms for ResNet18. R1's cache saves ~15-25 sec total over a
   25-round V4 run = ~1 s/round, well inside the per-node variance
   (jobs 6606100 = 112 s/round on alvis7-11, 6606101 = 97 s/round on
   alvis5-08 — 14 % spread between same-seed runs on different
   hardware).

2. **R2 (batch=128) doesn't help because the GPU is not the
   bottleneck.** With `num_workers=4`, CPU augmentation runs at
   ~12 ms/image throttled. Doubling batch size doubles per-batch
   wallclock (50 → 100 ms) and halves the number of batches, so
   per-epoch wallclock is unchanged. GPU SM util actually drops
   slightly because the slightly larger memory footprint and
   different kernel-launch overhead spread the GPU activity over
   more wallclock without delivering proportional compute.

### What the bottleneck actually is

Looking at the GPU profiler more carefully:

  Median SM util = 6 %. Longest sustained ≥ 80 % stretch = 5 s.
  Pattern: short repetitive bursts of 5-15 s at moderate util,
           interspersed with long stretches at 0-5 %.

This is consistent with **state-dict serialisation back-and-forth
dominating** the per-client time, not augmentation throttle. Per
round:

  - Server → 50 clients : 50 × 44 MiB = 2.2 GiB of param transfer
  - 50 clients → server : another 2.2 GiB of reply
  - Total : 4.4 GiB / round of in-memory copy work, all on CPU

That, plus Ray task dispatch overhead (~100–200 ms × 50 clients
= ~5-10 sec/round), eats the 80 % of per-round wallclock that
isn't GPU compute. The model rebuild (R1 target) was a minor
contributor; CPU augment (R2 target via batch-size) is gated by
num_workers, which is already at 4.

### Practical conclusion

R1+R2 are kept — they're correctness-neutral and add ~+4 %
GPU memory but no harm. They're just not the speedup we needed.

Two paths forward:

- **R4 (kornia GPU augment)**: would let us drop num_workers→0,
  remove the CPU augment chain, and push more work to the GPU.
  This is the next logical lever IF wallclock matters more than
  bit-determinism for that specific augmentation.
- **Accept the wallclock cost as the price of bit-determinism at
  single-actor.** The serialisation cost is fundamental to the
  Flower simulation design (one reply per client per round). To
  beat it we'd need to change the simulation backend or the
  per-client cost structure (smaller state dicts → use
  `trainable-layers=last_block` or `head_only` for those cells —
  but that's an experimental setting, not a universal fix).

Recommendation: **adopt R1+R2, hold off on R4 for now**, accept
~104 s/round as the cost of correctness, revisit GPU efficiency
once we have more cells run and a clearer picture of which
experiments are wallclock-sensitive vs determinism-sensitive.
