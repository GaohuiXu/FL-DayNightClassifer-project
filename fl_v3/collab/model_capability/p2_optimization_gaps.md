# MCR Phase-2 — deep optimization-gap analysis (6-analyst workflow, 2026-06-22)

Goal: cut per-epoch wall-clock below 11.6 min for fast iteration. Baseline = the locked optimized recipe
(unfrozen + ckpt-off + SDPA + compile(backbone), bf16, A100): **415 ms/step, ~11.6 min/epoch, ~77–85% util,
compute-bound** (backward ~50%, Swin fwd ~15%, loss ~8.4%). Full per-analyst notes: workflow `wf_b852f893-10e`.

## The big lever: 4-GPU DDP (~3.5×)

The codebase is unusually DDP-ready: **no BatchNorm in the trainable graph** (GroupNorm everywhere D6 + Swin
LayerNorm → no SyncBN), single-dict `forward(batch)` (DDP forwards verbatim), and the LSS/PointPillars
scatters are **per-GPU-local** reductions (no cross-rank interaction → no new nondeterminism beyond
all-reduce order, acceptable under D16). Zero existing DDP code (clean slate). Work is mechanical:
init_process_group(nccl) + DistributedSampler(set_epoch) + DDP-wrap (after compile, after the 2-group
optimizer) + rank-0-only checkpoint (strip `module.` and `._orig_mod.`) + all-reduce the logged loss + a
`torchrun --standalone --nproc_per_node=4` launcher on `A100:4`. **~3.2–3.7× → 11.6 → ~3.2 min/epoch**
(15 ep ~0.85 h, 30 ep ~1.7 h). Effort: medium.

**Batch decision (the one real choice):** global batch 16 = **4/GPU** keeps the recipe IDENTICAL (same LR, no
re-tuning) and still gets ~3.5× (the step is compute-bound, per-sample time is flat across batch). Going
16/GPU = global 64 needs LR scaling + adds rounds — a recipe change + confound. **Recommend 4/GPU (global 16).**

## Cheap code wins that STACK (do on single-GPU + carry into DDP)

| win | what | win | effort | risk |
|---|---|---:|---|---|
| **non_blocking HtoD** | `loop.py:41` `.to(device, non_blocking=True)` (mem already pinned) | overlaps ~17 ms/step | trivial | none (timing only) |
| **fused Adam** | `fused=True` on Adam, **bf16-gated** (not byte-identical → not fp32-strict) | ~3–4 ms + fewer launch gaps | trivial | gate on `not cudnn.deterministic`; null-config seed-band |
| **num-workers 4→8** | loader headroom (16 CPUs/GPU) — loader is at ~372 ms/batch vs 415 ms step, will bottleneck a faster/DDP run | enabler | trivial | host-RAM watch (pair w/ prefetch_factor 2) |
| **the 82 ms LOSS** | `losses.py`: cache radius→**device** Gaussian patch (kills the per-object CPU rebuild + HtoD) + batch the 6 `.cpu().tolist()` syncs into 1 | **~30–50 ms (~0.5 min/epoch)**, byte-identical | small | commutative max-overlay → AST-safe; verify_levers loss+grad parity |
| **expand compile** | compile the static BEV stack (camera_neck + fusion + bev_neck + head) — the copy_ churn is the *uncompiled eager region*'s autocast casts (A2) | ~15–30 ms (~0.4–0.6 min/epoch) | small | partial-batch recompile → `drop_last=True`; no-NaN + seed-band |

## Measure-first / defer

- **resize-on-CPU** (preprocess 1600×900→256×704 in `__getitem__`): cuts image HtoD 8× (~14.5 ms) but touches
  the half-pixel calibration contract (≤1px gate) → medium risk; do only if HtoD still shows after non_blocking.
- **fp64 `torch.inverse` → fp32** (view_transform `_img2lidar`): <1%, keep fp64 under precision=fp32.
- **full loss vectorization** (scatter_reduce amax render) / **target-build in dataloader workers**: takes loss
  →~0 (~1 min/epoch) but higher validation cost — escalate only if the cache+batch fix isn't enough.

## SKIP (measured-dead or wrong-tradeoff)
- **channels_last** — measured 0.99× neutral (attention/copies dominate, not BEV convs).
- **nvJPEG GPU decode** — changes pixels vs the pinned PIL decoder → a NEW data reference, not byte-compatible;
  only if the loader actually bottlenecks AND a fresh reference is accepted.
- **node-local staging** — reads are already hidden inside CPU decode (cold≈warm); not the bottleneck.
- **hand-removing individual `.contiguous()`** — sub-1%, footgun (the #76176 hazards).

## Recommended plan (single-GPU step first, then DDP)

1. Bank the do-now wins: non_blocking + fused-Adam(bf16) + num-workers 8 + the **loss cache/batch fix** +
   expand-compile (measure). Re-profile → expect ~11.6 → ~9–10 min/epoch single-GPU.
2. Implement **DDP at 4/GPU (global 16, same recipe)** → ~3.5× → **~2.5–2.9 min/epoch**.
3. Validate throughout: no-NaN trains-clean gate + verify_levers (loss/grad parity for the loss fix) +
   same-seed mAP within the D16 seed-variance band. DDP determinism is architecture-pinned (D9), seed-band not byte.

Net target: **~11.6 → ~2.5–3 min/epoch** (~4×), making the lever sweep + the eventual ≥3-seed FL runs affordable.

## RESULTS — single-GPU code wins (job 6770185, 2026-06-22)

The do-now stack (loss cache/batched-sync + non_blocking + fused-Adam, all byte-safe / 247 tests green) +
BEV-stack compile, measured on A100:

| config | step | vs prev | **util** | peak mem |
|---|---:|---:|---:|---:|
| baseline (ckpt-off) | 531 ms | 657 → 531 | **95%** | 29 GB |
| compile+sdpa | 341 ms | 415 → 341 | 93% | 27 GB |
| **compile+sdpa+bev** | **311 ms** | new best | **92%** | 25 GB |

- **`aten::copy_` count 13,818 → 1,914** — the loss fix removed the ~12,000 per-GT-object Gaussian-patch
  HtoDs (A4 confirmed); **util ~85% → 92–95%** (GPU now fully used — the orchestrator's goal).
- Cumulative **793 → 311 ms = 2.55×**, **~9.1 min/epoch** (was 11.6). 5 ep ≈ 0.8 h, 15 ep ≈ 2.3 h.
- compile-BEV (+camera_neck+fusion+bev_neck+head) added −9% (341 → 311); needs `drop_last=True` for clean
  training (avoids a partial-batch recompile/epoch). The residual 102 ms HtoD is the IMAGE transfer (A3
  resize-on-CPU = deferred ~17 ms).
- **Next: DDP** (16/GPU = global 64 + scaled LR, per orchestrator) → ~3.5× → **~2.6 min/epoch**.

## DDP — implemented + VALIDATED (jobs 6770189→6770190, 2026-06-22)

4-GPU torchrun DDP in `centralized_train.py` (gated on `LOCAL_RANK`; single-GPU path byte-identical, 247
tests pass). Launcher `run_centralized_a100_ddp.sh` (A100:4, batch per-GPU → global 64).
- **First smoke FAILED** — `find_unused_parameters=False` errored: `camera_neck.lateral.0/1` (+norms) never
  get grad (GeneralizedLSSFPN discards `lats[0]/lats[1]` at `out_level=2`/feat_stride=16). The unused set
  is CONSTANT → **`static_graph=True`** (more efficient than `find_unused_parameters=True`; recommended
  with `torch.compile`). Also fixed a latent compile+resume key mismatch (resume loads the stripped ckpt
  into the RAW model before compile/DDP-wrap).
- **Second smoke PASSED** — `ddp=True world=4 global_batch=64`, finite loss, `n=1280` (20×64 → all-reduce
  + sharding correct), rank-0 ckpt+checksum saved; compile + SDPA + compile-bev all compose under DDP.
  (The 20-step smoke's wall-time is compile-dominated — steady-state speedup comes from the first real epoch.)
- **Steady-state projection:** each rank runs the 311 ms batch-16 step + grad all-reduce (overlapped) →
  ~340 ms/step × 64 samples → **~2.5 min/epoch (~3.6×)**. Confirming via a short global-64 run (LR sqrt-
  scaled 0.006).
