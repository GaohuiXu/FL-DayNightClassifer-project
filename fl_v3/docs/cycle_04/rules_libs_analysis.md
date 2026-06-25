# Are the rules too conservative? — deep analysis (2026-06-25, feeds an orchestrator decision)

> 6-agent workflow (rule-audit / code-cost / lib-viability / determinism + skeptic/pragmatist verifiers) + owner
> web research. **Verdict: relax SELECTIVELY — but the worthwhile relaxations are IN-TREE, NOT the banned libraries.**
> This RECOMMENDS; the owner (orchestrator) decides (a proposed **D18**).

## Headline

The owner's instinct ("the rules may be over-conservative, like the original bit-identity stance") is **half-right —
and the right half was already fixed.** **D16 (2026-06-21) already did the big relaxation:** it retired determinism as
the reason for the library ban, and ADOPTED the in-tree industrial accel — SDPA (flash-attention-class) fused
attention, bf16-AMP, channels_last, fused-Adam, `torch.compile`, EMA — plus GATED dynamic-voxelization and an in-tree
dense-LiDAR upgrade. So the ban is **not** a blanket "no industrial CUDA libs"; it is already targeted.

What remains banned (spconv, torchsparse, mmcv, mmdet3d, FP8/Transformer-Engine, DALI) is the **out-of-tree,
no-aarch64-wheel, fragile/unmaintained subset**, and that residual ban is **LOAD-BEARING, not vestigial** — it binds
HARDER, not softer, at the Arrhenius (ARM + H200) cutover **5 days out** (2026-06-30).

**The bit-identity precedent does NOT transfer.** Relaxing bit-identity traded a *real, buy-back-able compute tax*
(602→14 ms argsort, a 1.3 GB LSS materialization) for *nothing scientific* once ≥3-seed mean±std was the bar — so it
was pure win. Un-banning spconv trades a **binary ARM build-break risk** for capability — a categorically riskier
trade, at the worst possible moment.

## Per-library verdict

| library | capability for us | runtime | ARM/H200 (the binding bar) | verdict |
|---|---|---|---|---|
| **spconv** (traveller59) | the largest lever: enables 0.075 m sparse 3D conv (~+0.05–0.08 mAP, *by analogy to BEVFusion — UNMEASURED on our stack*) | real win (sparse touches only occupied voxels) | **no aarch64 wheel** (x86 PyPI only); buildable from cumm-JIT source on aarch64 (Jetson-proven, SM90 OK) but fragile, single busy maintainer, no wheel | **keep banned** (measure first, see below) |
| torchsparse (MIT-HAN) | same class as spconv; TorchSparse++ active 2025 | comparable | same — no aarch64 wheel, source build | keep banned |
| mmcv / mmdet3d (OpenMMLab) | ~none direct (op-registry / reference only) | none | mmcv last release 2024-04, `_ext` breaks on CUDA 12/13, no aarch64 | keep banned (reference the *architecture*, free) |
| FP8 / Transformer-Engine | **negative** at our mAP — FP8 is a throughput lib, adds instability for ~0 accuracy | some Hopper GEMM throughput | aarch64 wheel EXISTS (first-party) — the one with low ARM risk | keep banned (no capability value; revisit only if H200 train-time dominates) |
| DALI | none | loader speedup, but **already recovered in-tree** (BLAS-thread pin) | aarch64 SBSA ships (low risk) | keep banned (in-tree recoverable) |

**Critical caveat:** the ~0.05–0.08 mAP for spconv is an **analogy from BEVFusion-base 0.685, NOT a measured ablation
on fl_v3's camera+fusion stack.** The entire "is it worth the risk" question rests on an unmeasured number.

## What IS worth relaxing (the real over-conservatism — all in-tree, zero portability cost)

1. **★ FREE SPEED WIN — the LiDAR encoder's determinism tax.** The camera LSS got a D16 relaxed branch
   (`view_transform.py:173`, native `scatter_add_` on the bf16 path); the **PointPillars encoder never did** — it still
   runs 5–6 serial stable-argsorts + `index_copy` on the *science* path. Add the same `relaxed = not
   cudnn.deterministic` branch: a single order-free `scatter_reduce(amax)` on bf16, keeping the strict sort under fp32.
   **Pure runtime win, zero capability/science-bar change, near-zero risk** (D16 already blesses atomics on the bf16
   path). This is the clearest live over-conservatism.
2. **AST-ban allowlist.** Remove order-free `scatter_reduce(amax)` from the banned-calls list (it is value-deterministic
   by construction and is the D16-gated dyn-vox op) — pending an on-device check that strict mode doesn't *raise* on it.
   Generalize the single `# D15-relaxed-ok` carve-out into a documented "strict fallback required OR
   manifest-exempt-with-rationale" policy. Flag the dual-path code as a simplification candidate once the strict tool
   stops catching bugs.
3. **Restate Rule #2 cleanly.** Drop the dead determinism language; state the real, testable bar: *"A dependency is
   admissible iff it ships a MAINTAINED wheel for the live target tier (x86 now, aarch64/H200 after 2026-06-30) AND
   does not NaN — NOT on determinism."* spconv/torchsparse/mmcv/mmdet3d/flash-attn-pkg/FP8-TE/DALI remain KEEP-OUT on
   *that* bar.

## The honest path on the capability lever (if you want it)

Before ANY rule debate about spconv, **measure it**: a throwaway **x86-only** ablation (spconv sparse 3D backbone at
0.075 m, or even a dense PillarNet at 0.1 m within Rule #2) to convert the ~0.05–0.08 from *assumption* to *fact* on
fl_v3's actual stack. If it's real AND worth a multi-hour run × the maintenance risk, *then* the rule debate has a
number. Also note: **the sanctioned pure-PyTorch stack itself has not yet been ARM-rebuild-verified** — proving the
base rebuilds on GH200 is the prerequisite before adding any fragile dep.

## Proposed D18 (for the owner)

Three selective relaxations, NONE touching the portability ban: (1) restate Rule #2 as the single testable bar above;
(2) the AST-ban `scatter_reduce(amax)` allowlist + the dual-path policy; (3) the free in-tree LiDAR-encoder relaxed
branch. Optionally: authorize an **x86-only throwaway spconv/0.1 m-dense ablation** to measure the capability lever
before deciding whether the portability ban is worth revisiting post-cutover (when the GH200 stack is proven).
