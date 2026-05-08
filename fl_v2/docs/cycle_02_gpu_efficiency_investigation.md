# Cycle 02 GPU-efficiency investigation (2026-05-08)

After the supervisor meeting, **Risk #1 to address: most jobs receive
"inefficient GPU usage" warnings from Alvis**. The site flags jobs whose
average SM utilization is below ~30 %; we need to (a) measure the
actual utilization, (b) understand the architectural cause, and
(c) propose remediation that does not change the science.

This doc captures the investigation plan and the analytical baseline.
It does **not** prescribe fixes yet — those depend on the empirical
profile from the next batch of jobs.

---

## 1. What we have right now

### Current allocation
- **Hardware**: 1× NVIDIA A40 (46 GiB, 10 752 CUDA cores).
- **Federation** (`fl_v2/configs/flwr_config.toml`):
  ```
  options.num-supernodes = 50
  options.backend.client-resources.num-cpus = 1
  options.backend.client-resources.num-gpus = 0.10  (= 5 concurrent Ray actors)
  ```
- **`run_alvis.sh` per-mode override** (already present, lines 117-129):
  - `head_only`  → 0.025 / supernode = 1.25 GPU equivalents
  - `last_block` → 0.05  / supernode = 2.5  GPU equivalents
  - `full_ft`    → 0.10  / supernode = 5    GPU equivalents (default)

### Per-round work
- `fraction-train: 1.0` — all 50 clients participate every round.
- Each client owns ~500-700 samples (Dirichlet α=0.5 over 26 640 train
  samples).
- `batch-size: 64`, `num-local-epochs: 3`.
- So per client per round: ~3 epochs × ~10 batches = **30 minibatches** of
  64 samples (≈ 1 920 forward+backward passes per client per round).
- Aggregate per round: 50 × 30 = **1 500 minibatches**.
- Times 100 rounds: **150 000 minibatches** over the ~90-min job.

### Estimated GPU work (analytical, full_ft cell)
- ResNet18 fp32 forward+backward on a 64×3×32×32 batch: typically
  **0.3-0.7 ms on A40** (small batch, kernel overhead dominates).
- Per round: 1 500 × 0.5 ms ≈ **0.75 s of pure GPU compute**.
- Plus per-round server-side eval: 24 510 test+ASR samples / 64
  ≈ 380 batches × ~0.5 ms ≈ **0.2 s GPU compute**.
- Plus framework metrics + checkpoint save (small).
- **Total GPU compute per round: ~1 s. Wallclock per round: ~50 s.**
- → **expected SM utilization ≈ 1-3 %.** That matches Alvis flagging
  the jobs as inefficient.

### Where is the wallclock going?
The 99 % gap is dominated by:

1. **Ray actor scheduling** — 5 actors handling 50 clients means each
   actor serves ~10 clients sequentially per round. Per-client
   schedule + actor warmup overhead ~10-50 ms × 10 clients = 100-500 ms
   per actor per round.
2. **ArrayRecord serialisation** — server pushes 11 M float32 params to
   each of 50 clients (50 × 44 MiB = 2.2 GiB transferred per round
   server→clients), and 50 × 44 MiB back for aggregation. Ray's
   plasma store is efficient but not free; ~1-2 s wallclock per
   round just for IO.
3. **Python overhead per batch** — DataLoader iteration, gradient
   zero-out, optimizer step, all run in Python. Even with pinned
   memory, a 0.5-ms GPU kernel between two ~5-ms Python steps gives
   ~10 % util at best.
4. **Strategy aggregation** — 50 sets of 11 M params averaged
   coordinate-wise. CPU-only. ~2-3 s per round.
5. **Wandb sync** — periodic flushes of metrics buffers; non-trivial
   when network is saturated.
6. **DataLoader cold start per client/per round** — even with
   `_client_data_cache`, the first call after Ray actor restart
   re-reads from disk.

The dominating axis is (1) + (2): for 50-client federations on a
single GPU, the *Ray scheduling layer* outweighs the GPU work.

---

## 2. Investigation plan

### Step A — Capture actual GPU utilization (LANDED 2026-05-08)
`run_alvis.sh` now spawns `nvidia-smi` in the background at job start,
sampling every 5 s and writing a CSV next to the slurm log:
```
/mimer/.../fl_outputs/slurm/<jobname>_<jid>.gpuprof.csv
```
At job end the cleanup section computes summary stats:

```
samples = ...
GPU SM util mean = X %  (max ..., min ...)
GPU mem util mean = ...
Time at >=50% SM util = ...
Time at <10%  SM util = ...
```

This costs <50 KB of CSV per hour and adds <1 % CPU overhead. The next
job submitted after commit `<this commit>` will produce the first
real utilization profile we have.

### Step B — Identify which axis dominates (after we have data)
Once we have the actual util profile, two patterns are diagnostic:

- **Pattern 1** — flat low utilization across the run (5-15 %).
  → Ray scheduling / Python overhead is dominant. Remediation candidates:
  - Reduce number of concurrent actors (fewer, busier actors).
  - Use a smaller `num-supernodes` if the science permits.
  - Disable wandb during exploratory runs.
- **Pattern 2** — bursty utilization (e.g., 70 % in spikes,
  0 % between rounds).
  → The GPU work itself is fine; the gaps are server-side aggregation
  + parameter transfer. Remediation candidates:
  - Larger batch size on the client (more work per kernel).
  - More local epochs per round (more GPU work per IO event).
  - CPU-side aggregation parallelism (Ray task fanout for the
    median / trimmed-mean / Bulyan steps).

### Step C — Test 1-2 remediations quickly (1-2 small jobs)
Once Step B identifies the bottleneck, propose 1-2 targeted ablations.
Each takes ~1.5 h wallclock and produces a new util profile to compare.

---

## 3. Remediation candidates (ranked by expected impact)

### R1 — Increase batch size at client (most likely the biggest win)
Default is 64. ResNet18 on A40 saturates around batch=512+. Bumping to
128 or 256 directly increases per-kernel work without changing the
science meaningfully (3 local epochs × ~5 batches at batch=128 still
gives plenty of optimizer steps).

- **Cost**: 1-line config change. Memory budget: ResNet18 + activations
  for batch=512 ≈ 2 GiB, well within 46 GiB / 5 actors = 9 GiB per
  actor.
- **Risk**: very small. `train_local` is batch-size-agnostic.
- **Expected impact**: 3-5× SM util increase if Ray overhead is
  not dominant.

### R2 — Reduce concurrent-actor count
Current: 5 actors at num-gpus=0.10. Try 2 actors at num-gpus=0.5 or 1
actor at num-gpus=1.0.

- **Cost**: 1-line config change.
- **Risk**: with 1 actor, all 50 clients run sequentially in 1 process
  → much longer wallclock per round, may exceed wallclock budget. Earlier
  v8 single-actor experiment timed out; needs careful testing.
- **Expected impact**: depends on the contention pattern. If actor
  context-switching is the bottleneck, this helps significantly.

### R3 — Reduce fraction-train
fraction-train=1.0 is unrealistic anyway (real FL doesn't have 100 %
participation per round) and risk-audit H5 already flagged it. Reducing
to 0.1-0.3 cuts per-round work proportionally and lets us use 1 actor
at full GPU.

- **Cost**: YAML change + may need more rounds to converge.
- **Risk**: changes the science (FL convergence dynamics differ at
  fraction-train < 1.0).
- **Expected impact**: large GPU efficiency gain BUT this is also a
  scientific change — should be done as a deliberate ablation rather
  than as a GPU-efficiency hack.

### R4 — Avoid wandb online during exploratory runs
`wandb-mode: offline` skips network sync; logs to disk and uploads
later. Removes some Python-side blocking.

- **Cost**: tiny. Default is online; flip to offline for exploration.
- **Risk**: lose live-monitoring during the run.
- **Expected impact**: minor on GPU util, but removes a known source
  of jitter.

### R5 — Caches: ensure DataLoader / encoder feature caching
The convergent head-feature decomposition recomputes the encoder
forward each batch even though the encoder is frozen for head_only
cells. Caching per-image features once per cell would speed those
*diagnostic* runs by 5-10× and cut their wallclock dramatically.
This is the same optimisation I flagged earlier and we deferred.

- **Cost**: ~30 lines of refactor in
  `analysis/head_feature_decomposition.py::_train_fresh_head`.
- **Risk**: small. Same math; just reorder the computation.
- **Expected impact**: head_only and lastblock head-attr v2 cells
  become 5-10× faster, more cells fit per SLURM job.

---

## 4. What NOT to do

1. **Don't change strategies wholesale.** Switching from FedAvg to a
   non-Ray-based simulation would invalidate all our prior numbers
   and is well out of scope.
2. **Don't drop `fraction-train` silently to fix GPU efficiency.**
   That is a scientific change (different FL regime) and must be
   discussed/owned as such, not absorbed into "we made the GPU happier".
3. **Don't reduce client count (50→10) for efficiency alone.** Same
   reason — changes the FL science.
4. **Don't use mixed-precision (fp16) without verifying determinism.**
   Mixed-precision can re-introduce non-determinism the audit just
   eliminated. Worth testing in a separate ablation if the residual-ε
   investigation closes; not now.

---

## 5. Action plan (ordered)

1. **(LANDED)** Add `nvidia-smi` profiler to `run_alvis.sh`. Every
   future job emits a `*.gpuprof.csv` and a summary block in the
   slurm log.
2. **(NEXT)** Run one new audit-fixed cell (e.g.,
   `_phase3_fixed_full_ft_clean_seed42` again) to capture the first
   utilization profile. Read the summary; classify Pattern 1 vs
   Pattern 2.
3. **(THEN)** Pick 1-2 remediations from §3 (likely R1: batch-size
   bump). Run a paired comparison job. Document the delta in this
   doc.
4. **(DEFERRED until residual-ε is understood)** Test the more
   invasive remediations (R2, R5).

---

## 6. Notes on Alvis's specific warning system

`seff` is broken on the login node (Slurmdb.pm missing); `jobinfo`
takes only queued/running jobs, not historical. The warnings the user
mentioned arrive via Alvis's email/notification system, not via the
standard SLURM efficiency tools. Once we have a few `*.gpuprof.csv`
files we'll be able to compare against whatever threshold Alvis is
using (they typically use 30 % SM util or 30 % memory util).
