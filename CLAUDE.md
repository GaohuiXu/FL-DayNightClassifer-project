# CLAUDE.md — project root

> Active AI instructions for this repository. Codex's parallel file is [`AGENTS.md`](AGENTS.md).
> `fl_v2/` keeps its own (frozen, GTSRB-era) `fl_v2/CLAUDE.md` — ignore it unless you are deliberately
> working inside `fl_v2/`.

## What this project is

A master-thesis project on **securing federated learning for autonomous-driving perception**, paper
target **USENIX-Security**. The active work is **Cycle 04**: build a bit-deterministic **federated
multimodal (camera+LiDAR) AD perception platform on nuScenes** (`fl_v3/`) and run a **backdoor attack
suite × general defense suite** benchmark. **Platform-first; NOT bonded to FLAME** (FLAME is one
defense among several).

- **Active platform: `fl_v3/`.** Everything new goes here.
- **`fl_v2/` is FROZEN** — the old GTSRB platform, referenced ONLY as an *implementation oracle*
  (defense reimplementations match it on fixtures = implementation equivalence, NOT scientific
  validity). Do not modify `fl_v2/`.

## Start here

- **Plan (source of truth):** [`fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`](fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md)
- **Roadmap:** [`fl_v3/docs/roadmap/INDEX.md`](fl_v3/docs/roadmap/INDEX.md)
- **Orchestration model + decisions:** [`fl_v3/docs/cycle_04/README.md`](fl_v3/docs/cycle_04/README.md) · [`fl_v3/docs/cycle_04/decisions.md`](fl_v3/docs/cycle_04/decisions.md)
- **fl_v3 status + layout:** [`fl_v3/README.md`](fl_v3/README.md)

## How we work (orchestrator + serial workers)

The planning session is an **orchestrator** (owns the plan, the decisions record D1–D17 in
`fl_v3/docs/cycle_04/decisions.md`, per-task SPEC contracts, kickoff prompts; does NOT implement). Each task **T0…T7** is built in **one fresh Claude session**
(reads its `fl_v3/docs/cycle_04/tasks/T<N>_SPEC.md`, implements, fills `fl_v3/collab/T<N>/SPEC.md`,
drives its GATE) and reviewed in **one fresh Codex session** (scientific-correctness only; writes
`fl_v3/collab/T<N>/REVIEW.md`; never commits). **No parallel Claude implementation sessions.**

## Standing rules (non-negotiable)

1. **Reproducibility regime (D16, 2026-06-21 — supersedes "bit-determinism is sacred").** The science
   path is **bf16-AMP**; reported results use **≥3 seeds (mean±std)** and a claim is valid if it clears
   the **seed-variance floor** — NOT by same-seed byte-identity (the `relaxed` `scatter_add` LSS rewrite
   is atomic ⇒ not byte-identical run-to-run). RNG still flows through `derive_seed`
   (`fl_v3/src/fl_v3/utils/runtime.py`); every scientific run keeps one **trains-clean reasonableness
   gate** (no NaN/divergence — loss + BEV-accumulation in fp32) and logs its `precision`. The **strict
   byte-identical regime is retained as an offline dev-regression tool ONLY** (the `precision=fp32`/strict
   knob + the static-AST ban — it caught two real bugs), not the bar for reported numbers. **Tooling
   envelope (D16-addendum, re-derived 2026-06-21):** the binding bar is *maintained + builds on the target
   tier (x86 now, aarch64/H200 next) + no-NaN* — NOT bit-determinism; use in-tree accel aggressively (SDPA
   fused attention, `torch.compile`, `channels_last`, fused Adam, EMA), **keep out** out-of-tree fragile
   CUDA exts (Rule #2); dynamic voxelization is a *gated* in-tree ablation (order-free `scatter_reduce`),
   not banned. See `fl_v3/docs/determinism.md` + the D15/D16(+addendum) amendment in
   `fl_v3/docs/cycle_04/decisions.md`.
2. **No `mmdet3d`/`mmcv`/`spconv`** (+ `torchsparse`, FP8/Transformer-Engine, DALI) **as dependencies** —
   out-of-tree, unmaintained-or-fragile, no aarch64/H200 wheels (won't survive the ARM rebuild); `spconv`/
   `mmcv` kernels also have no deterministic path for the strict dev tool. Reason is now **portability +
   maintenance** (D16 relaxed the *determinism* reason; verified 2026-06-21 — D16 addendum). Reference their
   architecture (Apache-2.0), reimplement in pure PyTorch; get speed from **in-tree** accel instead. The
   LiDAR-capacity lever, if ever needed, is an in-tree dense upgrade (PillarNet-style), NOT spconv.
3. **Engineering smoke (mini) vs scientific result (trainval) is a hard boundary.** `v1.0-mini` is
   for pipeline/determinism validation only; every scientific claim needs trainval-scale clients.
4. **Null-config** (`poison_rate=0`) must reproduce the clean baseline **within the seed-variance band**
   (D16 — was "bit-for-bit"; byte-identity now holds only under the offline strict knob). **Frozen
   `fl_v2/` is the oracle** for defense *implementation* equivalence (fixture-level), not scientific validity.
5. **Heavy runs go through SLURM** (`run_alvis.sh` pattern); the login node is for scaffolding, the
   venv build, and unit/determinism tests only. Run code via `fl_v3/scripts/run_in_venv.sh`.
6. Honor the confirmed decisions in `fl_v3/docs/cycle_04/decisions.md` (**D1–D17**; note **D1 is amended
   by D17** — the camera backbone is now **trained**, not frozen; **D16** is the bf16-AMP precision regime)
   and the **§Attack spec** + **§Defense Benchmark Protocol** in the plan (5-condition fusion-awareness
   ablation, ASR eligibility + denominator, utility/ASR 2×2 rule, required baselines, controlled `m_r` vs
   defense-assumed `f_r`, etc.).

## HPC

Alvis (x86) until 2026-06-30, then Arrhenius (ARM H200). The "no-mmdet3d, pure-PyTorch" design is what
makes the ARM rebuild painless — keep the venv reproducible from a pinned manifest
(`fl_v3/docs/env.md`). Dataset is **fully extracted, read-only** at
`/mimer/NOBACKUP/Datasets/NuScenes_v1.0/` (`v1.0-mini` + `v1.0-trainval` + `v1.0-test`; shared
`samples/`/`sweeps/`; all 6 cameras + `LIDAR_TOP`) — **no extraction needed.** (A separate
`/mimer/NOBACKUP/Datasets/nuScenes/` dir with a different layout also exists — do **not** point at it.)

## Git / branches

Cycle-04 work lives on **`v3-ad-perception`** (which contains both frozen `fl_v2/` and active `fl_v3/`
as sibling dirs). `v2-new-api` / `main` are the GTSRB mainline — leave them. Commit/push only when the
user asks. End commit messages with the Co-Authored-By trailer.
