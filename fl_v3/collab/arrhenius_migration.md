# Arrhenius migration — platform de-risk (spconv + fp16 precision)

> Alvis (x86, A100, bf16) sunsets ~2026-06-30 → Arrhenius (aarch64, GH200, **no bf16 for spconv**). This doc
> captures the platform-readiness findings before the move. Authored 2026-06-28.

## Phase 0A — spconv on GH200/aarch64 (owner's test, 2026-06-28)
**Result: `stable_enough_for_experimental_branch`.** Source-built spconv (v2.3.8 / cumm v0.7.13) imports and runs
sparse-conv fwd/bwd on GH200; 2-GPU FP32 DDP and 2-GPU **FP16-AMP + GradScaler** DDP both pass multiple steps.
- ⇒ **The no-spconv Rule #2 constraint is LIFTED on Arrhenius** — the pillars→sparse-3D-voxel LiDAR upgrade
  (audit ceiling #2) is now possible THERE (not on Alvis). PyPI has no aarch64 wheel → source JIT build
  (`TORCH_CUDA_ARCH_LIST=9.0`, EasyBuild GCC `libstdc++` preloaded ahead of the system one).
- **Precision caveat:** direct `torch.bfloat16` is NOT supported by this spconv/cumm path (`KeyError: torch.bfloat16`);
  `autocast(bf16)` runs but the sparse output is fp16. So with spconv: **FP16-AMP+GradScaler for the sparse branch,
  or force it FP32** (no bf16 sparse conv in this spconv/cumm build). Remote workspace: `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/phase0a_spconv`.

## fp16 de-risk on our model (Alvis, 2026-06-28, jobs 6782420/6782421/6782422)
Since bf16 is unavailable for the sparse path in this build, the science precision on Arrhenius is
**fp16-AMP+GradScaler (or fp32)**. Validated the non-spconv model (camera-LSS / pillar-LiDAR / fusion /
CenterPoint head) under fp16-AMP+GradScaler via `p1_amp_smoke.py --fp16` (reports GradScaler skips = the direct
inf/NaN signal). *(The model at the time of this run also carried the depth-supervision lever, since removed as
net-negative; the fp16 verdict is about the precision path and is unaffected — the depth numbers below are that
historical run's.)*

**Verdict: fp16 is VIABLE — no fundamental blocker.**
- **Forward is fp16-safe** — total/hm/reg/depth all finite, depth GT coverage 73–79%. My pre-test worry (the LSS
  `cumsum`/splat overflowing fp16) was **unfounded**: the D15 relaxed splat already accumulates in **fp32**
  (`.float()` before `scatter_add_`), so the BEV pool never sees fp16 range limits.
- **Backward calibrates then trains.** From GradScaler's default init_scale 65536, the huge random-init focal loss
  (hm≈160 ⇒ grad-norm ~11k, like the bf16 run) overflows fp16 grads → GradScaler halves the scale for ~7 steps
  (skipped) down to **512**, then **every step lands** (scale stable at 512, no more skips). Over steps 7–39 the
  loss drops cleanly (**total 50→11.5, hm 44→7.0, grad-norm 2703→28**) and the **depth CE itself drops 4.02→3.1**
  — i.e. fp16 trains both detection and depth supervision, just like bf16.
- **Only cost:** ~7 throwaway calibration steps at startup. Removable with **GradScaler `init_scale≈512`** (skips the
  search) or a brief **fp32 warmup**. Neither is necessary (7 of thousands of steps).

## Precision plan for Arrhenius (proposed D16 amendment — per-platform)
- **Alvis:** bf16-AMP (no GradScaler), as today.
- **Arrhenius:** **fp16-AMP + GradScaler** (`init_scale≈512`) for the dense/camera path; sparse branch fp16
  (Phase 0A) or fp32 fallback. **Keep fp32 for:** the LSS BEV splat (already), GroupNorm/softmax (autocast does),
  the focal/L1 head loss (already upcast), EMA, and `view_transform.depth_targets` (already autocast-disabled).
- Determinism stays **seed-variance-based** (D16 already relaxed byte-identity); GradScaler's dynamic scale widens
  the band slightly. Cross-platform numbers compare only at the seed-variance level, never byte-wise.
- **No code blocker found** — the platform's existing fp32-accumulation choices (D15 splat, head-loss upcast)
  already make the model fp16-friendly. The fp16 run path needs only: an fp16 autocast option in the trainer +
  a GradScaler (the bf16 path uses neither today). Small, additive.

## Tooling
`scripts/p1_amp_smoke.py --fp16` (fp16-AMP+GradScaler, cycles the loader, reports per-step scale + skips + finite),
`scripts/run_p1_amp_smoke.sh FP16=1`. Reusable for the Arrhenius bring-up smoke.
