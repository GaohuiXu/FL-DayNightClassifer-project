# Q1 speedup roadmap — relaxed-determinism (D15) session

> Source: 6-agent codebase scan (31 findings → 11 levers) + synthesis, all grounded in the real
> `speedup-levers` code. Baseline A40 ~1.7 s/step (TF32). The whole win lives in the two 35% stages
> (frozen camera_backbone, LSS view_transform). Everything is gated behind a new
> `determinism-level = strict | relaxed` knob (default `strict` → existing runs stay byte-identical).
> Dynamics-changing levers get a mandatory **1–2 round A40 mAP-within-noise smoke** vs the deterministic
> reference (mAP 0.36 centralized / FL-30 0.196). Multi-seed variance design = deferred to T5–T7.

## The 11 levers

**FREE / result-neutral (no mAP risk):**
- **L0** `determinism-level` knob — enabler (`runtime.py`). Default strict preserves byte-identity.
- **L1** `cudnn.benchmark=True` + `use_deterministic_algorithms(False)` — conv autotuner. ~3–5%, risk none.
- **L3a** kill the `geom_id.max().item()` host-sync in `bev_splat` (constant-fold G). ~0.3–0.5%.
- **L3b** **scatter_add splat + mask-then-lift** — drop the argsort + QuickCumsum + two 1.28 GB copies;
  lift only at valid points. **~10–18%**, risk none*. (*per-cell sums drift at ULP → breaks the 2
  determinism unit tests *by design* — must delete/relax them.)
- **L4** per-step `loss.item()` sync → accumulate on device. ~1–3%.
- **L5** dataloader `pin_memory` + `persistent_workers` + `prefetch_factor`. ~2–4%.
- **L6** `zero_grad(set_to_none=True)`. <1%.

**CHANGES-DYNAMICS / numeric-drift (need the mAP smoke):**
- **L2** **bf16 autocast over the whole forward** — hits BOTH 35% stages at once (frozen backbone is
  risk-free; trainable depthnet/necks see tiny drift). bf16 (not fp16) → no GradScaler, fp32 range. Keep
  `_img2lidar` fp64 + loss-target build in fp32. **~15–25%, the single biggest lever.** Risk low.
- **L8** `torch.compile` backbone/tail (`max-autotune`). ~5–12% (overlaps L2). Best on long/centralized
  runs (recompile/round kills it in short FL rounds). Subsumes channels-last + SDPA-patch.
- **L7** frozen-backbone feature cache — up to ~30% (removes the stage) but **storage-blocked** at
  trainval + needs a poison-bypass (attacked clients' images differ). Mutually exclusive with L8.
- **L9** bigger batch (48–64; mem 7/46 GB). ~1.2–1.4×/epoch but **med risk** (fewer optimizer steps/round
  → changed convergence). Hold out of T5–T7 unless documented.
- **L10** fewer/log depth bins (D=59→48/40). ~4–9% but **med risk** (far-range representation).

## Recommended waves (re-profile after each; estimates don't stack 1:1 — shared bandwidth)

| Wave | Levers | Cumulative | Risk |
|---|---|---|---|
| **0** enabler + free scheduling | L0, L1, L4, L5, L6 | ~1.06–1.12× | none |
| **1** LSS rewrite | L3a, L3b | ~1.20–1.35× | none (breaks 2 det tests by design) |
| **2** bf16 autocast | L2 | ~1.4–1.7× | low (mAP smoke) |
| **3** compile OR feature-cache | L8 (default) / L7 (if storage) | ~1.5–1.9× | low |
| **4** optional dynamics | L9, L10 | up to ~2× | med (hold out of T5–T7) |

## Headlines
- **Single highest-ROI:** **L2 bf16 autocast** — only lever hitting both 35% stages; M effort, low risk.
- **Highest-ROI result-neutral:** **L3b scatter_add splat + mask-then-lift** — deletes a host-sync, an
  argsort, a cumsum, and two 1.28 GB copies that were *pure determinism tax*. ~10–18% at zero mAP risk.
- **Realistic targets:** low-risk-only (W0+1+2) **~1.5×**; +compile (W3) **~1.8×**; +cache (storage) ~2.2–2.5×.

## NOT worth it (low ROI)
channels-last standalone (subsumed by L1+L2+L8); SDPA/flash-attn monkeypatch of Swin (49-token windows;
subsumed by compile); lidar scatter_reduce rewrite (lidar = 1.1%); decode topk (eval-only, not training).

## Cross-cutting
- All under `determinism-level=relaxed` so `precision_state()` records it (avoids the D14 "no mixing
  regimes" desync). Same-seed runs are no longer byte-identical — the 2 LSS determinism unit tests get
  deleted/relaxed.
- L9/L10 change the *science*, not just speed → keep out of the multi-seed T5–T7 comparison unless
  explicitly adopted + documented.
- Code lives in worktree `claude/speedup-levers`; the loss `.item()` fix is already committed (`8234b33`).
