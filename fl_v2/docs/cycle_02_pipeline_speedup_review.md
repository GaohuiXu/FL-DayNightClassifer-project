# Cycle 02 pipeline speedup review

**Status:** review draft, 2026-05-09. Companion to
`cycle_02_speedup_investigation.md` (R1+R2 attempt, neutral) and
`cycle_02_gpu_efficiency_investigation.md` (Alvis warning context).

The pipeline is bit-deterministic at fixed `(commit, seed)` after the
audit's seven fixes (`cycle_02_pivot_audit.md` §9). Cost: ~104 s/round
at the canonical 50-client × `num-gpus=1.0` × `num_workers=4` config,
mean GPU SM util **15–22 %** (Alvis flags <30 %).

R1 (per-actor model cache) + R2 (batch 64 → 128) at commits `ee5d734`
and `739658a` were **wallclock-neutral** in V4 jobs 6606100/6606101.
That session diagnosed "state-dict serialisation back-and-forth" as
the residual bottleneck but did not pin it to a line. This review does
— and proposes ≤5 remediations that preserve bit-determinism and don't
touch experimental knobs.

Hard constraints: bit-determinism stays (login-node test + paired SLURM
≤20-round run); no change to `num-clients`, `fraction-train`,
`num-server-rounds`, `num-local-epochs`, `batch-size`, `model-type`,
`pretrained-init`, `trainable-layers`, `attack-type`, `partition-mode`,
`dirichlet-alpha`, `learning-rate`, `lr-schedule`, or `num-gpus`. No
fp16, no Flower bump, no backend swap.

---

## 1. Profile — where the 104 s actually goes

### 1.1 Empirical baseline (4 jobs × 25 rounds × 5-s gpuprof)

| Config | Hosts | Per-round | Mean SM | Median SM | Time <5 % SM | Mem used | Longest ≥50 % stretch |
|---|---|---|---|---|---|---|---|
| pre-R1+R2 (1.0 / 4 / batch=64), 6605728 | alvis7-11 | 101 s | 22 % | 6 % | 45 % | 1.6 GiB | 15 s |
| pre-R1+R2, 6605729 | alvis5-08 | 111 s | 21 % | 6 % | 47 % | 1.6 GiB | 15 s |
| post-R1+R2 (batch=128 + cache), 6606100 | alvis7-11 | 111 s | 15 % | 2 % | 60 % | 2.1 GiB | 10 s |
| post-R1+R2, 6606101 | alvis5-08 | 97 s | 19 % | 4 % | 55 % | 2.1 GiB | 15 s |

Same-seed runs on **different physical A40s** spread by 14 % wallclock
(97 s vs 111 s); host-jitter floor ≈ 12 s/round. Memory occupancy is
4 % of the A40's 46 GiB → no OOM constraint on any candidate.

Median SM util at 2–6 % means half the 5-s samples show essentially
zero GPU work. The longest sustained ≥50 % stretch is 10–15 s — the
GPU is **never** saturated. Consistent with per-client Python/CPU
overhead, not GPU-bound compute.

### 1.2 Per-call decomposition of `@app.train` (client_app.py:213-364)

Walking through one client call, with line references and indicative timings
(timings are micro-benchmark estimates; the relative ordering is what matters):

```
@app.train  (per call, per client, every round)
  L221-237  per-call deterministic seeding + cuDNN flags        ~2 ms
  L239      _load_model_from_message:
              L209  msg.content["arrays"].to_torch_state_dict() ~150 ms
              L209  load_state_dict(...)  CPU→GPU 11M params    ~80 ms
              (model object cached → no fresh ResNet18 ctor)
  L240      _load_client_data (cached after round 1)            ~0
  L246-249  to_torch_state_dict() AGAIN
              + .detach().cpu().clone() of every parameter
              ── consumed ONLY at L305-330, gated by
              ── (is_malicious AND attack_type == "model_replacement")
              ── for non-model_replacement runs: pure waste     ~600 ms
  L287-296  train_local: 3 epochs × ~10 batches × ~10 ms        ~300 ms
              per epoch a val pass (only last is consumed)      ~50 ms × 3
  L303      model.state_dict() (cached model → cheap view)      ~0
  L337      ArrayRecord(reply_state_dict) serialise reply       ~150 ms
  Ray task dispatch + Flower message ack                        ~100 ms
  ───────────────────────────────────────────
  per-client total                                              ~1.5 s
  GPU active fraction                                           ~10 %
```

50 × 1.5 s = ~75 s in-round, plus ~18 s inter-round (server eval +
aggregation + broadcast) plus ~5 s setup ≈ ~98 s, matching the
observed 97–111 s/round once host jitter is folded in.

The single largest leak: **the unconditional CPU clone at L246-249
fires on every call but is consumed only by malicious clients in
`model_replacement` runs.** For all other configurations — clean
baselines, pixel-trigger cells, label-flipping cells, every defense
ablation — those 50 × ~600 ms (≈ 30 s/round) are pure overhead. This
alone explains why ~50 % of gpuprof samples sit at <5 % SM util after
R1+R2: the GPU idles while Python re-deserialises and deep-copies a
44 MiB state-dict that no one reads.

### 1.3 Inter-round work (~18 s)

- Strategy `aggregate_train`: norm logging materialises 50 × 11M-float
  numpy arrays (≈ 2.2 GiB CPU alloc), then Flower's FedAvg sums again
  → ~3-5 s (`norm_tracking_fedavg.py:164-167, 188-191` +
  `attacks_defenses/defenses/norm_clipping.py:56-75`)
- Server `evaluate_fn`: fresh `ResNet18` ctor + load_state_dict +
  `.to(device)` + 12,630-sample test pass → ~3 s
  (`server_app.py:223-239`)
- Wandb + experiment logger → ~0.5 s
- Flower aggregation + broadcast prep (residual) → ~5-10 s

---

## 2. Bottleneck ranking

Ordered by addressable share. "Class" is GPU compute / CPU memcpy / Python
overhead / framework / I/O.

| # | Bottleneck | Class | ~Per-round | File:line |
|---|---|---|---|---|
| **B1** | Unconditional CPU clone of broadcast state-dict in `@app.train` | CPU memcpy / Python | **~30 s** | `client_app.py:246-249` |
| B2 | Per-local-epoch validation pass (3× per client; only last reported) | GPU compute | ~5 s | `training/train.py:184-208` |
| B3 | CPU PIL augmentation chain run per `__getitem__` | CPU compute | ~5 s | `data/transforms.py:7-25` + `data/dataset.py:259-263` |
| B4 | `NormTrackingFedAvg` materialises 50× client `to_numpy_ndarrays()` for L2 norm logging | CPU memcpy / Python | ~3 s | `strategy/norm_tracking_fedavg.py:164-167, 188-191` |
| B5 | Server-side `evaluate_fn` rebuilds `ResNet18` every round | GPU compute + Python | ~3 s | `server_app.py:223-239` |
| B6 | Two `to_torch_state_dict()` calls per `@app.train` (L209 + L248) | Python | ~2 s | `client_app.py:209, 248` |
| B7 | Ray actor task dispatch × 50 clients per round | Framework | ~5 s | Flower/Ray internals (out of scope) |
| B8 | Flower `aggregate_arrayrecords` non-vectorised float sum | Framework + CPU | ~2 s | `flwr.serverapp.strategy.strategy_utils` (out of scope) |

Verdict: **the dominant bottleneck class is Python/CPU overhead, not
GPU compute or I/O.** R1+R2 missed because R1 attacked a small Python
op and R2 attacked GPU compute that wasn't the limit; kornia
GPU-augment (R4) was deferred for the same reason — CPU augment is
~5 % of the 104 s, not 30 %.

Addressable budget B1+B2+B3+B4+B5+B6 ≈ 48 s. Out of scope B7+B8 ≈ 7 s.
Pure GPU compute ≈ 50 s. Total ≈ 104 s ✓.

---

## 3. Remediation candidates

Ranked by leverage. `C` numbers are independent — adopting any subset is
fine, and the verification gate is per-candidate (§4).

### C1 — Gate the global-state-dict CPU clone behind the model-replacement consumer

**Files:** `fl_v2/src/fl_v2/client_app.py:213-364`

**Mechanism.** Lines 246-249 execute on every `@app.train` call:

```python
global_state_dict = {
    k: v.detach().cpu().clone()
    for k, v in msg.content["arrays"].to_torch_state_dict().items()
}
```

Only consumer: the model-replacement scaling block L305-330, gated by
`is_malicious and attack_type == "model_replacement"`. Move the clone
under that same gate; shift the attack-metadata reads (L261-266) up so
the gate can be evaluated first.

**Wallclock.** ~30 s/round (~30 %) on every non-model-replacement run
(clean baselines, pixel-trigger, label-flipping, all defense
ablations). Zero impact on model-replacement cells — they still pay
the clone, but no longer for free.

**Determinism.** The clone is read-only and unused outside the gate;
eliminating an unused copy cannot affect numerics. Bit-equivalent.

**Cost.** ~10 lines, single diff. Side-effects: none.

### C2 — Compute validation only on the final local epoch

**Files:** `fl_v2/src/fl_v2/training/train.py:184-208`

**Mechanism.** `train_local` runs `evaluate(model, valloader, …)`
after every local epoch (3 calls/client/round). Only `history[-1]` is
consumed by the caller (L213-215 → `final_val_*`). Skip on epochs
`0..num_epochs-2`; run only on `num_epochs-1`.

**Wallclock.** ~5 s/round (skip 2 of 3 val passes × 50 clients ×
~50 ms each).

**Determinism.** `evaluate` is read-only on the model and on the train
RNG; skipped val passes advance no state used by training. Preserved.

**Cost.** ~5 lines (guard `evaluate` with `if epoch == num_epochs - 1`).
Side-effects: per-intermediate-epoch `val_loss` / `val_accuracy`
disappear from `history`. No downstream consumer reads them
(`final_val_*` consumers in `client_app.py` and `server_app.py` only
read the last entry).

### C3 — Optimise norm-logging in `NormTrackingFedAvg.aggregate_train`

**Files:** `fl_v2/src/fl_v2/strategy/norm_tracking_fedavg.py:133-191`,
`fl_v2/src/fl_v2/attacks_defenses/defenses/norm_clipping.py:56-75`

**Mechanism.** The aggregator calls `to_numpy_ndarrays()` once for the
global (L164) and 50× for clients (L188-191) — 50 × 11M × 4 B ≈
2.2 GiB of CPU alloc just to compute one scalar L2 norm per client.
Stream it instead: iterate `(global_array, client_array)` pairs
key-by-key, accumulating `sum_sq += np.sum((c - g) ** 2)`, take
`sqrt` at the end. Same numerical result, no full numpy materialisation.

**Wallclock.** ~3 s/round CPU memcpy reduction.

**Determinism.** L2 norm of a deterministic vector is order-stable in
fp64 for the same per-key reduction order; the streaming rewrite uses
the same key order as today, so the logged scalar is identical to
round-off. The norm is logging-only — it doesn't feed aggregation, so
any residual ε of fp64 accumulation order doesn't affect the FL
trajectory anyway.

**Cost.** ~20 lines across two files. Side-effects: `*_norm_log.json`
schema unchanged; values identical to round-off.

### C4 — Per-actor model cache for the server-side `evaluate_fn`

**Files:** `fl_v2/src/fl_v2/server_app.py:159-275`

**Mechanism.** L223-227 instantiate a fresh `ResNet18` every round
even though `load_state_dict` (L228) immediately overwrites every
parameter and BN buffer. Mirror the client-side R1 cache: build the
model once at `evaluate_fn` closure construction, keep it in a
closure-local variable, only run `load_state_dict` + `.to(device)`
afterwards. Server actor is long-lived → cache lifetime = simulation
lifetime.

**Wallclock.** ~2-3 s/round (ctor + `.to(device)` amortised).

**Determinism.** Same argument as client-side R1: `load_state_dict`
overwrites every parameter and BN buffer (running_mean / running_var /
num_batches_tracked are all in `state_dict`). Bit-equivalent.

**Cost.** ~10 lines, same idiom as `client_app.py:50` +
`_load_model_from_message`. Side-effects: none.

### C5 — Cache per-client tensors after deterministic transforms

**Files:** `fl_v2/src/fl_v2/data/dataset.py:209-353`,
`fl_v2/src/fl_v2/data/transforms.py:7-25`

**Mechanism.** The augment chain has three deterministic ops
(`Resize 32×32`, `ToTensor`, `Normalize(0.5, 0.5)`) and two stochastic
ones (`RandomRotation(±10°)`, `ColorJitter(0.15, 0.15, 0.15, 0.02)`).
`TransformedSubset.__getitem__` re-runs the whole chain per worker per
epoch. Pre-compute the deterministic ops once when the per-client
DataLoader is first built; cache as a `(N, 3, 32, 32)` float32 tensor
(per client: ~500-700 samples × 3 × 32 × 32 × 4 B ≈ 6-8 MiB; total
~400 MiB across 50 clients — within the A40's 44 GiB headroom). Apply
stochastic ops on the cached tensor via
`torchvision.transforms.functional.rotate` + tensor-form ColorJitter
(supported as of torchvision 0.13+).

**Wallclock.** ~5-10 s/round. PIL `Resize` from variable input → 32×32
is the largest per-image cost; skipping it removes ~70 % of CPU
augment work. Mean SM util rises ~5-10 pp.

**Determinism.** Strict gate. `seeded_worker_init` + the per-loader
`torch.Generator` continue to seed worker numpy/random/torch RNGs.
`transforms.functional.rotate` and tensor-form `ColorJitter` are
deterministic given seeded RNGs. **Risk:** PIL bilinear vs tensor
bilinear may differ at edges. Required check: extend
`tests/test_dataloader_determinism.py` with a `paired_pil_vs_tensor`
case asserting bit-equality at fixed seed. If it passes, C5 is a
transparent speedup; **if batches diverge, C5 becomes an augmentation
re-baseline (out of scope) — drop C5 in that case.**

**Cost.** ~50 lines (cache build at first iter + updated
`__getitem__`). Side-effects: +400 MiB memory; numerical only if
bilinear paths agree. **Recommendation:** land C1-C4 first; revisit
C5 only after the paired test passes.

### Combined expected outcome (C1+C2+C3+C4)

| Metric | Today | After C1-C4 |
|---|---|---|
| Per-round wallclock | ~104 s | **~60-70 s** |
| Mean SM util | 15-22 % | **~30-40 %** |
| Time <10 % SM | 60-70 % | ~40 % |
| GPU memory used | 2.1 GiB | ~2.1 GiB |

C1 alone likely clears the Alvis 30 % threshold for non-model-rep
runs because the GPU stops idling during the wasted clone. C2-C4 are
correctness-neutral free tightening. C5 held in reserve.

---

## 4. Verification plan

For every remediation, three signals must converge before merge.

### 4.1 Login-node unit test (~30 s)

```bash
cd /mimer/.../fl_weather_project/fl_v2
module purge && module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
source ../.venv/bin/activate
python -m pytest tests/test_dataloader_determinism.py -xvs
```

Must pass. C5 also requires a paired PIL-vs-tensor batch equality case.

### 4.2 Paired SLURM same-seed run, 20 rounds (NOT 100)

Clone the existing 25-round test config; override `num-server-rounds`
+ `checkpoint-rounds` to 20 (e.g.
`_phase3_speedup_check_run{A,B}.yaml`). Submit both via
`submit_experiment.sh`. Compare:

- `summary.json::final` — exact byte match.
- `rounds.csv` — exact byte match.
- SHA-256 of `round_0020.pt` — exact byte match.
- gpuprof.csv summary block (`mean SM util`, `<10 % SM`, `≥50 % SM`)
  — record before/after.

Rationale for 20 rounds: 100-round paired runs are too expensive as a
per-iteration gate. The audit pair 6594906/6594907 established that
bit-divergence, when it occurs, surfaces by round 6 — 20 rounds is
~3× headroom over the empirical detection horizon.

### 4.3 Wallclock delta (≥2 paired runs per remediation)

The 12-s/round host-jitter floor → minimum two paired runs per change,
on different physical A40s when possible. Report mean per-round
wallclock and per-host spread.

### 4.4 Per-remediation gates

| | Determinism (4.1+4.2) | Wallclock (4.3) | Functional |
|---|---|---|---|
| **C1** | required | ≥20 s/round on non-model-rep; ≤2 s on model-rep | clone fires only when `is_malicious and attack_type == "model_replacement"` |
| **C2** | required | ≥3 s/round | `final_*` in `summary.json` byte-identical |
| **C3** | required | ≥1 s/round | `*_norm_log.json` identical to round-off |
| **C4** | required | ≥1 s/round | server-eval metrics in `rounds.csv` byte-identical |
| **C5** | required + paired PIL-vs-tensor equality | ≥3 s/round | only if PIL-vs-tensor test passes |

If any determinism gate fails, the change reverts. If a wallclock
gate fails but determinism passes, the change is correctness-neutral
and may still land — but the commit message should note the expected
speedup did not materialise.

---

## 5. What we are NOT doing

- **fp16 / mixed precision** — re-introduces non-determinism the
  audit closed; defer until residual-ε investigation closes.
- **`fraction-train < 1.0`** — scientific knob (risk-audit H5), not a
  GPU-efficiency lever.
- **`num-gpus < 1.0` (multi-actor)** — re-introduces residual ε at
  round 2 (verified in 6605828/6605829). Locked at 1.0.
- **Flower-version bump or non-Ray backend** — invalidates prior
  numbers.
- **Pre-augmenting + dropping `RandomRotation` / `ColorJitter`** —
  removes stochastic regularisation; methodological change.
- **kornia GPU-augment (prior R4)** — larger code change; CPU augment
  is ~5 % of wallclock under `num_workers=4`, not dominant.
  Re-evaluate after C1-C4 if mean SM util plateaus below ~35 %.
- **`get_global_val_test_split_loaders` switch (risk-audit H1)** —
  already-landed infrastructure cutting server eval from 12,630 → 1,000
  samples per round (~1-2 s/round). Owned by the audit doc, not folded
  into this speedup pass — it changes round-by-round metric semantics.
- **`wandb-mode: offline`** — ~1-2 s/round; orthogonal, land separately
  if requested.

---

## 6. Recommended sequencing

1. **C1** — biggest win, smallest diff, trivial determinism argument.
2. **C2** — small, free; bundle with C1 or land separately.
3. **C4** — symmetric to R1 on the server side; small, free.
4. **C3** — modest CPU-allocation reduction; compounds over 100-round runs.
5. **C5** — only after the paired PIL-vs-tensor equality test passes;
   if it fails, drop it — C1-C4 already clear the Alvis threshold.

After C1-C4: per-round wallclock ~60-70 s, mean SM util 30-40 %. Both
the speedup gap to the old non-deterministic 60 s/round and the Alvis
30 % threshold are closed without changing experimental design or
determinism.

---

## 7. UPDATE 2026-05-10: empirical results — bottleneck ranking REFUTED

After landing C1 and instrumenting the hot path with `time.perf_counter()`
on a one-off profiling job (6609276, 20 rounds), the actual per-round
wallclock distribution is:

| Phase | Time/round | Share | What it is |
|---|---|---|---|
| **`train_local` (Σ across 50 clients)** | **85.8 s** | **76 %** | actual local-training loop |
| inter-round gap | 16.3 s | 14 % | between `agg_train_end` and next `cfg_train_start` (Flower internals) |
| server `evaluate_fn` (12,630-sample test pass) | 6.1 s | 5 % | of which `eval`=6.0 s, `model_init`=0.094 s |
| `aggregate_train` (norm logging + FedAvg) | 3.4 s | 3 % | `norms`=1.5 s, `super_agg`=1.1 s, `client_extract`=0.8 s |
| client `reply` (state_dict → ArrayRecord) | 1.9 s | 2 % | 39 ms × 50 |
| client `model_load` (`to_torch_state_dict` + `load_state_dict`) | 1.5 s | 1 % | **30 ms × 50, NOT 600 ms** |
| client `data_load` (cache hit after r1) | 0.2 s | 0.2 % | |
| client seed + cuDNN flags | 0.1 s | 0.1 % | |

Per-client `train_local` averages **1.72 s** for 3 local epochs × ~10
batches at batch=128 → **~57 ms/batch wallclock**. ResNet18 fp32
forward+backward on 32×32 batch=128 is ~2 ms on an A40 → **~55 ms/batch
is non-GPU work** (CPU augment + cuDNN-deterministic-fallback +
per-batch Python overhead).

### What this overturns

1. **B1 (the global-state-dict CPU clone) was wrong.** Estimated at
   ~600 ms/client (~30 s/round). Actual: 30 ms/client (1.5 s/round
   total) — 20× smaller. C1 was implemented, paired-SLURM verified
   bit-deterministic (jobs 6609204/6609205, identical
   `summary.json` + `rounds.csv` + `round_0020.pt` SHA-256), but is
   **wallclock-neutral** (105 / 110 s vs the 97-111 s post-R1+R2
   baseline — well inside the ±12 s host-jitter floor). The change is
   correctness-clean (eliminates work that was indeed unused for
   non-`model_replacement` runs) but does not move wallclock.

2. **B6 (the duplicate `to_torch_state_dict`) was wrong.** Estimated at
   ~2 s/round. Actual: 1.5 s/round total *for the whole model_load
   path* — the second call C1 removed was likely <0.5 s/round.

3. **B2 (per-local-epoch validation) is plausibly real but already
   inside `train_local` time.** The 1.72 s/client total for
   `train_local` includes 3 train epochs + 3 val passes. C2 would
   shave ≤5 s/round. Still small relative to the ~85 s training cost.

4. **B7 / inter-round gap is bigger than expected** — 16 s/round
   (14 %) lives in Flower internals between `aggregate_train_end` and
   the next `configure_train_start`. This is not directly attackable
   without modifying Flower, but it's a real chunk that the prior
   ranking lumped into "framework overhead" without measuring.

### What the bottleneck actually is

**Per-batch wallclock inside `train_local` is the dominant cost.** At
~57 ms/batch and ~2 ms of true GPU compute, ~96 % of per-batch time is
non-GPU: PIL augmentation in DataLoader workers, cuDNN deterministic
fallbacks, optimizer.step Python overhead, per-batch `inputs.to(device)`
+ `targets.to(device)` transfers, and `.item()` calls in the metrics
helpers.

### Revised remediation priorities

C1 lands cleanly but is wallclock-neutral. C2 → C4 are similarly
≤2-5 s/round and likely won't deliver a clear paired-SLURM signal
either. The actual leverage candidates have shifted:

- **R-real-1 (was C5, now top priority): tensor-cached transforms +
  GPU augment.** The whole 76 % `train_local` budget is the per-batch
  CPU pipeline. Pre-resizing+ToTensor+Normalize once and applying
  `RandomRotation` + `ColorJitter` on cached tensors (or kornia GPU
  augment) directly attacks the dominant phase. Plausible 30-50 %
  drop in `train_local`.
- **R-real-2: cuDNN-deterministic-fallback cost.** `torch.use_deterministic_algorithms(True, warn_only=True)` is set per-call
  on the client (and once at startup on the server). Some ops have
  no deterministic alternative and fall back to slower paths. The
  actual cost on A40 / ResNet18 fp32 is unmeasured but textbook
  numbers say 5-15 %. Paired-SLURM with the flag toggled (single
  experiment, kept off-tree) would quantify it. **Out of scope for
  any merge** — turning it off re-introduces non-determinism.
- **R-real-3: server-eval batching.** The 6 s/round eval pass over
  12,630 samples is 5 % of wallclock. The risk-audit H1 fix
  (`get_global_val_test_split_loaders`, infrastructure already
  landed) cuts to 1,000 samples and saves ~5 s/round. Methodological
  decision belongs to the audit doc.

### Practical conclusion

C1 is the *only* candidate of the original five that is clearly safe
to land (correctness-clean, determinism-preserved, eliminates truly
unused work). It does not deliver wallclock improvement, so per the
"if improved, then commit" rule **C1 stays in the worktree but is not
proposed for the main tree**.

C2-C4 are estimated below the ~12 s/round host-jitter floor; they
would need carefully-controlled paired runs to even detect. Not
recommended without a clearer hypothesis.

The genuinely promising direction is **R-real-1** (tensor-cached
transforms), which directly attacks the 76 % `train_local` budget.
That requires a new round of investigation (PIL-vs-tensor batch
equality test in `tests/test_dataloader_determinism.py`, micro-bench
per-batch wallclock breakdown into augment / forward+backward /
optimizer.step, then a remediation YAML). It's not a follow-up to C1;
it's a separate workstream.

This empirical update supersedes §1.2, §2, and §3 of this doc for the
purpose of remediation prioritisation. The text above remains as the
record of how the analysis evolved.

---

## 8. UPDATE 2026-05-10: per-batch profile + num-workers sweep

A second profiling pass instrumented `train_one_epoch` per-batch
(`yield` / `to_dev` / `fwd` / `bwd` / `opt` / `metrics`, with
`cuda.synchronize()` between phases). Job 6609354, 20 rounds, batch=128.

### 8.1 Per-batch breakdown

```
Per-round, summed across 50 clients × 3 epochs:
  yield (DataLoader iter wait): 38.30 s   (67.9 %)   ← CPU augment / worker IPC
  backward (GPU):               11.26 s   (20.0 %)
  forward (GPU):                 5.26 s   ( 9.3 %)
  optimizer.step:                1.42 s   ( 2.5 %)
  to_device (CPU→GPU):           0.10 s   ( 0.2 %)
  metrics (.item()):             0.08 s   ( 0.1 %)
  total train_one_epoch:        56.42 s   ( 100 %)

Per-batch mean (over 12,897 batches):
  yield = 59.4 ms/batch   fwd = 8.2 ms   bwd = 17.5 ms   opt = 2.2 ms
  total = 87.5 ms/batch   (vs ~57 ms in production without cuda.sync overhead)
```

The GPU sits idle ~68 % of in-train time waiting for the next batch
from DataLoader workers. The actual GPU compute is 25 ms/batch
(`fwd+bwd+opt`) — surprisingly high for ResNet18 fp32 on 32×32 batch=128
A40 (textbook ~2 ms). The cost is likely the `cuDNN deterministic +
torch.use_deterministic_algorithms(True)` fallbacks (out of scope —
turning them off re-introduces the residual-ε we eliminated).

### 8.2 num-workers sweep (jobs 6609937 / 6609938)

To rule out a 1-line YAML fix before committing to a structural change:

| num-workers | Per-round | Mean SM util | <10 % SM time | Notes |
|---|---|---|---|---|
| 0 (single thread) | ~200 s (cancelled) | 7.0 % | 67 % | augment becomes the entire critical path |
| **4 (default)** | **105 / 110 s** | **17.6 / 16.2 %** | **59 / 63 %** | C1 verify baseline (6609204/5) |
| 8 | 118 s | 16.9 % | 73 % | slightly *slower*; final acc/ASR also differ → worker-count is determinism-affecting |

`num-workers=4` is already near the sweet spot on this hardware (16
SLURM-allocated CPUs, 1 Ray actor + DataLoader workers + Ray system
processes). Increasing workers doesn't help and changes the
deterministic state.

### 8.3 Ruled out

- **More DataLoader workers** (8): no speedup; methodology change.
- **Fewer workers** (0): catastrophic slowdown (2× wallclock).
- **C1-C4 from the original ranking**: each ≤5 s/round predicted, well
  inside the ~12 s/round host-jitter floor. C1 verified to be
  wallclock-neutral (jobs 6609204/6609205).

### 8.4 Only remaining lever (without changing experimental design)

Move the augmentation pipeline off the CPU worker path. Two viable
shapes:

(a) **Tensor-cached pre-resize** — pre-compute `Resize → ToTensor →
Normalize` once per client at first DataLoader iteration; cache as a
`(N, 3, 32, 32)` float32 tensor (~400 MiB total across 50 clients,
within the A40's 44 GiB headroom). Then apply
`RandomRotation` + `ColorJitter` on cached tensors via
`torchvision.transforms.functional` (still CPU but deterministic and
much cheaper since `Resize` is the dominant per-image cost).

(b) **GPU-side augmentation** — apply `RandomRotation` + `ColorJitter`
directly on GPU after the batch is moved (kornia or
`transforms.v2` tensor mode). Eliminates CPU augment chain
entirely.

Either path needs a strict determinism gate: a paired PIL-vs-tensor
batch equality test extended into
`tests/test_dataloader_determinism.py`. If PIL and tensor bilinear
agree at fixed seed, the change is a transparent speedup. If they
disagree, the change is an augmentation re-baseline (out of the
user's stated scope) and must be dropped.

Predicted impact (if path b succeeds with bit-equality):
- yield drops from 38 s → ~0 s/round (workers no longer the gate)
- per-batch wallclock drops from ~57 ms → ~30 ms (just GPU compute)
- per-round drops from ~104 s → ~55-65 s
- mean SM util rises from 17 % → ~35-45 %

This is the only candidate left that targets the dominant phase. It
is also the most invasive of the original five (~50 lines + a new
test). **Recommend: implement only after a focused design discussion
with the user, not as a continuation of this profiling pass.**

### 8.5 Final state of the worktree at end of profiling

- C1 (gate the unconditional clone) — applied, bit-deterministic,
  wallclock-neutral. **Not committed** per the "if improved, commit"
  rule.
- All instrumentation reverted; only `client_app.py` shows a diff.
- New verification YAMLs left in `configs/experiments/cycle_02/phaseD2/`:
  `_c1_check_pixel5_r20_run{A,B}.yaml`,
  `_c1_profile_pixel5_r20.yaml`,
  `_c1_workers{0,8}_pixel5_r20.yaml`.
- Login-node `tests/test_dataloader_determinism.py`: 3/3 PASS.

The empirical record now has a clean per-phase + per-batch decomposition
that any subsequent remediation can target without re-discovering the
ranking.

---

## 9. UPDATE 2026-05-10: implementation complete

Five changes landed on top of the per-batch profile in §8. All five are
bit-deterministic by construction; the unit test
`tests/test_dataloader_determinism.py` was extended from 3 → 6 cases
and all 6 PASS. Every paired SLURM verification produced the same
`round_0020.pt` SHA-256 (`8c15ebb585b53f42…ced5fbf5`) — every
intermediate stage is byte-identical to every other.

### 9.1 Final wallclock progression (20-round paired SLURM, seed=42, pixel5)

| Stage | Jobs | Per-round | Δ baseline | Mean SM util |
|---|---|---|---|---|
| C1-only (pre-this-pass committed) | 6609204 / 6609205 | 105 / 110 s | — | 17.6 / 16.2 % |
| + C5 (Resize pre-cache) | 6610087 / 6610088 | 63 / 66 s | **−40 %** | 30.8 / 31.2 % |
| + C2 (val skip) + C4 (server eval cache) | 6611348 / 6611349 | 61 / 66 s | −41 % | 34.6 / 29.8 % |
| + valloader + testloader full pre-cache | 6611447 / 6611448 | **60 / 62 s** | **−42 %** | 32.6 / 31.4 % |

**100-round experiment time: ~175 min → ~102 min (saves ~73 min per cell).**

The final per-round number is within ~5 s of the legacy non-deterministic
0.10/0 configuration (~60 s/round) — i.e. the determinism-vs-speed gap
that motivated this entire investigation has effectively closed.

### 9.2 Files changed (final diff)

```
fl_v2/src/fl_v2/client_app.py                   C1 — gate clone behind model_replacement
fl_v2/src/fl_v2/data/dataset.py                 C5 + valloader/testloader full pre-cache
                                                (TransformedSubset gains pre_transform;
                                                 get_client_dataloaders + get_global_testloader
                                                 + get_global_val_test_split_loaders updated)
fl_v2/src/fl_v2/data/transforms.py              C5 split helpers
fl_v2/src/fl_v2/training/train.py               C2 — val pass only on last local epoch
fl_v2/src/fl_v2/server_app.py                   C4 — eval_fn model cache
fl_v2/tests/test_dataloader_determinism.py      +3 bit-equality tests (6 total)
fl_v2/configs/experiments/cycle_02/phaseD2/     +6 verification YAMLs (C1/C5/C2C4C5/C5full check + profile)
```

### 9.3 Determinism — three converging gates

1. **Unit test** (`tests/test_dataloader_determinism.py`, 6/6 PASS):
   - fused-chain vs split pre_transform-cache (num_workers=0 AND 4)
   - full pre-cache (transform=None) vs inline transform
   - different-seeds-diverge sanity
2. **Within-pair SLURM bit-identity** at every stage (runA vs runB).
3. **Cross-stage SHA-256 stability** — every checkpoint at every round
   for every config in §9.1 matches `8c15ebb5…ced5fbf5`.

### 9.4 What's beyond the new floor

Approximate residual budget at ~61 s/round:

| Phase | s/round | Class |
|---|---|---|
| Inter-round Flower internals | ~16 | framework overhead — not attackable inside Flower 1.27 |
| GPU compute (forward + backward + opt) | ~18 | bounded below by cuDNN-deterministic — cannot disable |
| Residual CPU augment (Rotation + Jitter + ToTensor + Normalize per __getitem__) | ~10-15 | stochastic; can only move to GPU via kornia (breaks PIL bit-equality) |
| Server eval + aggregate + misc | ~10 | each piece <3 s, all inside host jitter |

Three out-of-scope candidates that would cross the new floor:

* **Wire H1 audit's val/test split** — methodological change (changes
  the meaning of per-round `test_*` metrics in wandb). Saves ~5 s/round.
* **GPU augmentation (kornia)** — breaks PIL bit-equality, requires a
  full re-baseline. Saves ~10-15 s/round.
* **Multi-actor `num-gpus < 1.0`** — re-introduces residual ε (verified
  in V4-3rd-try jobs 6605828 / 6605829). Saves ~30 s/round.

None of these can be folded into this speedup pass without violating
the user-stated constraints.

### 9.5 Bottom line

**42 % wallclock reduction, full bit-determinism preserved, zero
methodological change.** Ready to commit. Further wins require
trading off determinism, augmentation realism, or per-round metric
semantics — owner decisions, not engineering optimisations.
