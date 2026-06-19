# D15 / D16 — speedup + precision decision (FOR ORCHESTRATOR REVIEW)

> Outcome of the dedicated speedup/diagnostics session (D14 charter). **Direction APPROVED by the session
> director (user); the orchestrator ratifies the two one-way steps below (config-collapse + bf16
> re-baseline), which are intentionally NOT executed here.** Code committed on branch
> `claude/focused-diffie-1da582` (NOT v3-ad-perception): `8234b33` (loss fix), `c020526` (D14 infra),
> `41f674a` (D15 waves), `5c2dfc2` (compile-checkpoint fix), `aafedf9` (compile-in-FL + num-gpus=0.5).
> Docs in `fl_v3/collab/speedup/`. **Nothing is on the v3 mainline — this is the implementation behind a
> knob (default `strict` = unchanged) + the proposal.**
>
> **Held for the orchestrator (one-way, touch committed infra + the references):** (1) the config-collapse
> `numeric-mode × determinism-level → one `precision` knob`; (2) retiring the strict/tf32/checksum science
> path; (3) re-baselining the clean references (centralized + ≥30-round FL) in bf16-AMP. The session
> implemented + measured everything behind the `determinism-level` knob so these are mechanical when ratified.

## TL;DR

1. **D14 diagnostics answered Q1–Q5** (see `SYNTHESIS_5questions.md`): the weak T5 was **FL-undertraining +
   FedAvg dilution, NOT architecture** (centralized mAP 0.36 vs FL-15 0.13 / FL-30 0.20); 15 rounds is
   undertrained → next clean reference **≥30 rounds**.
2. **D15 speedup: a `determinism-level = strict | relaxed` knob** delivers **~3× per training step** on the
   A40 (measured), default `strict` stays byte-identical (247 tests pass). The dominant win is the LSS
   view-transform `scatter_add` rewrite (602 → 14 ms, 42.8×) — precision-independent.
3. **D16 (the decision to ratify): adopt bf16-AMP as the single clean precision regime; retire the
   fp32/tf32 + strict/loose multi-criteria mess; criterion becomes claim-reproducibility (no byte
   checksums) with multi-seed at T5–T7.** bf16-AMP is venue-standard and now verified to train comparably.

## D15 — the relaxed-determinism speedup (implemented, behind a knob)

`determinism-level=relaxed` unlocks what bit-identity banned. Measured per-step (A40, batch-16):

| stage | strict (tf32) | relaxed (bf16-AMP) | factor |
|---|---:|---:|---:|
| LSS view_transform | 602 ms | **14 ms** | **42.8×** (scatter_add + mask-then-lift; precision-independent) |
| camera_backbone | 442 ms | 147 ms | 3.0× (bf16 + compile) |
| backward | 191 ms | 112 ms | 1.7× |
| loss | 91 ms | 87 ms | 1.04× (already vectorized) |
| **mean step** | **1443 ms** | **460 ms** | **~3.1×** |

Levers (all gated on `relaxed`; default `strict` = byte-identical, 247 tests pass):
- **scatter_add splat + mask-then-lift** (drops argsort + cumsum + the ~1.3 GB materialization) — *the*
  big win, and precision-independent (works even without bf16).
- **bf16-AMP** autocast over the forward (loss + BEV-splat kept fp32 — see the NaN note below).
- **torch.compile(backbone)** for long/centralized runs (NOT the per-round FL path — would recompile/round).
- free scheduling: cudnn.benchmark, per-step `.item()` sync removal, dataloader prefetch/pin/persistent.

### Round-level speedup ladder (MEASURED — FL, full participation, 4×A40, 3-round relaxed-bf16 runs)

| Config | 3-round elapsed | per-round | vs strict | note |
|---|---:|---:|---:|---|
| strict tf32 (reference) | 2799 s | **15.6 min** | 1.0× | E-15/E-30 measured |
| relaxed bf16, no compile, 1 client/GPU | 1264 s | 7.0 min | 2.2× | job 6767474 |
| + compile-in-FL, 1 client/GPU | 1159 s | 6.4 min | 2.4× | job 6767561 (recompiles=0) |
| **+ compile + num-gpus=0.5 (2 clients/GPU)** | **1014 s** | **5.6 min** | **2.76×** | job 6767562 — best |
| + compile + num-gpus=0.333 (3 clients/GPU) | THRASHES | — | <1.0× | job 6767615: round 1 alone > the full 2/GPU run → cancelled. Compute over-subscribed; 2/GPU is the ceiling. |

All trained cleanly (no NaN; update-norms 35→30 decreasing). The 3-round totals include the one-time
Ray cold-start + round-1 compile, so the **≥30-round steady-state per-round is lower** than the table's
average. **Best config (relaxed + compile + 2 clients/GPU): ~5.6 min/round → the ≥30-round clean
reference ≈ ~2.8 h (vs ~7.8 h strict).** `num-gpus=0.5` captures the idle ~54% GPU without a batch/science
change (better than enlarging the batch). GPU memory has room for more clients/GPU (~5–6 fit at 7.6 GB
each) but **compute saturates at ~2/GPU** (the 2nd client added only +14%), so 3/GPU is expected to
thrash (probe pending).

### The bf16 NaN — caught, diagnosed, fixed (why we keep a reasonableness gate)

The profiler showed bf16 was ~3× faster, but the **3-epoch smoke diverged to NaN in epoch 1.** Cause: the
CenterPoint **focal loss `log(sigmoid(logit))` ran on bf16 head logits** (standard AMP gap). Fix: compute
the loss in **fp32** + accumulate the BEV `scatter_add` in **fp32** (forward stays bf16). Re-test
(3-epoch centralized): bf16-AMP **3.216 / 2.362 / 2.139** vs deterministic D1 **3.199 / 2.365 / 2.145**
— within ~0.5% at EVERY epoch; and the 3-round FL run trained cleanly (no divergence). So bf16-AMP
trains comparably at BOTH centralized and FL scale. *This is the case for keeping ONE lightweight
reasonableness gate even after dropping bit-identity — speed that NaNs is worthless.* (Separately, a
`torch.compile` checkpoint-save bug — `_orig_mod.` key prefix — was found + fixed in `5c2dfc2`; it only
hit the compiled-run post-train eval, not training or the FL path.)

## D16 — precision + criteria cleanup (DECISION TO RATIFY)

The session accumulated a messy 4-axis space (fp32 / tf32 / bf16 × strict / loose identity). Recommend
collapsing to **one regime + one criterion**:

- **Precision: bf16-AMP** (bf16 heavy ops + fp32 stability ops + fp32 loss/accumulation). This is the
  field standard (BEVFusion/BEVDet train fp16-AMP; bf16 is strictly safer — fp32 range, no GradScaler),
  is the fastest, and trains comparably (verified). **Drop tf32 as a separate regime** — under bf16-AMP
  it is provably redundant (relaxed step 460 ms with tf32-base vs 466 ms with fp32-base = noise).
- **Criterion: claim-reproducibility**, not byte-identity. Same-seed run-to-run variance is allowed;
  report results over **multiple seeds (mean ± std)** at T5–T7; a claim is valid if it clears the
  seed-variance floor. **Retire strict bit-identity + checksum stamping from the science path** (keep
  the strict knob + the static-AST ban ONLY as an offline dev regression tool — it caught two real bugs
  this session, incl. the lever-1 backward break).
- **Consequence (re-baseline):** the D1/E reference checkpoints are tf32-strict → **superseded**. The
  clean references (centralized + the ≥30-round FL) must be **re-run in bf16-AMP** before T5–T7 bind to
  them. (Determinism is architecture-pinned anyway; bf16 is one more re-baseline.)
- **Config cleanup (recommended, not yet done):** collapse `numeric-mode {fp32,tf32}` × `determinism-level
  {strict,relaxed}` into ONE `precision = bf16 | fp32` knob (bf16 = science/relaxed; fp32 = dev/deterministic).
  Touches ~8 call sites + the gate scripts + provenance — a focused refactor to do once the orchestrator
  ratifies the direction (held to avoid churning committed infra mid-decision).

## Open items for the orchestrator

1. **Ratify D16** (bf16-AMP single regime + claim-reproducibility + drop strict/tf32/checksums from science).
2. **Authorize the config collapse** (one `precision` knob) — I'll execute on ratification.
3. **A fresh FL-recipe session** (separate from this one): FedAvg dilution is structural (centralized 0.36
   ≫ FL 0.20 at matched budget); levers = server-side momentum (FedAdam/FedOpt), the log-group non-IID
   severity, local-epochs, round budget. The current FL settings are not good enough for the attack/defense
   study regardless of speed.
4. **Multi-seed protocol** for T5–T7 (seed count per cell; the defense-decision knife-edge needs ≥3 seeds).
5. **Re-baseline the clean references in bf16-AMP** at ≥30 rounds before the next attack design.
