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

The planning session is an **orchestrator** (owns the plan, decisions D1–D8, per-task SPEC contracts,
kickoff prompts; does NOT implement). Each task **T0…T7** is built in **one fresh Claude session**
(reads its `fl_v3/docs/cycle_04/tasks/T<N>_SPEC.md`, implements, fills `fl_v3/collab/T<N>/SPEC.md`,
drives its GATE) and reviewed in **one fresh Codex session** (scientific-correctness only; writes
`fl_v3/collab/T<N>/REVIEW.md`; never commits). **No parallel Claude implementation sessions.**

## Standing rules (non-negotiable)

1. **Bit-determinism is sacred.** Same-seed runs must be byte-identical. Any RNG via `derive_seed`
   (`fl_v3/src/fl_v3/utils/runtime.py`). Banned: atomic scatter, `grid_sample` backward, non-stable
   `sort`/`topk`, flash-attn, dynamic voxelization/spconv. See `fl_v3/docs/determinism.md`.
2. **No `mmdet3d`/`mmcv`/`spconv`** — abandoned + non-deterministic + won't build on 2026 CUDA/ARM.
   Reference their architecture (Apache-2.0), reimplement deterministically.
3. **Engineering smoke (mini) vs scientific result (trainval) is a hard boundary.** `v1.0-mini` is
   for pipeline/determinism validation only; every scientific claim needs trainval-scale clients.
4. **Null-config** must reproduce the clean baseline bit-for-bit. **Frozen `fl_v2/` is the oracle.**
5. **Heavy runs go through SLURM** (`run_alvis.sh` pattern); the login node is for scaffolding, the
   venv build, and unit/determinism tests only. Run code via `fl_v3/scripts/run_in_venv.sh`.
6. Honor the confirmed **D1–D8** decisions and the **§Attack spec** + **§Defense Benchmark Protocol**
   in the plan (5-condition fusion-awareness ablation, ASR eligibility + denominator, utility/ASR 2×2
   rule, required baselines, controlled `m_r` vs defense-assumed `f_r`, etc.).

## HPC

Alvis (x86) until 2026-06-30, then Arrhenius (ARM H200). The "no-mmdet3d, pure-PyTorch" design is what
makes the ARM rebuild painless — keep the venv reproducible from a pinned manifest
(`fl_v3/docs/env.md`). Dataset staged at `/mimer/NOBACKUP/Datasets/nuScenes` (ZIPs; extract mini first).

## Git / branches

Cycle-04 work lives on **`v3-ad-perception`** (which contains both frozen `fl_v2/` and active `fl_v3/`
as sibling dirs). `v2-new-api` / `main` are the GTSRB mainline — leave them. Commit/push only when the
user asks. End commit messages with the Co-Authored-By trailer.
