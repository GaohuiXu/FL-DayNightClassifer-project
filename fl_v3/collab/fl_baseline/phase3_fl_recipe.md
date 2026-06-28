# MCR Phase-3 STEP 3 — the FL recipe (FedAdam, D17) + fairness anchor

> The tuned FL recipe to federate the locked centralized **bb02d** (0.5656 mAP / 0.5733 NDS) into a clean
> bf16 FL baseline. Designed via a 4-recipe judge-panel (workflow `wf_1e63db64`) whose universal fixes are
> folded in. **The hyperparameters below are the STARTING recipe — they MUST be tuned on the cheap probe
> sweep (§Tuning) before the full R=30 run.** Config: `fl_v3/configs/fl_bb02d_fedadam.json`.

## What is federated

The FULL bb02d model — **270 trainable tensors / 33.17M params, ZERO frozen** (camera_backbone 169 *trained*
+ lidar_backbone 39 + the 62 non-backbone). D17 amends D1: the Swin-T backbone is trained, so the
DT3-A "exclude+reconstruct the frozen backbone" optimization no longer applies — the whole model is the FL
update vector. (The bb02d BEV aug + img-flip + per-class heatmap weights already flow to the FL clients via
the existing `_aug_from_run` loader hook + `build_criterion`; CBGS/GT-paste are centralized-only.)

## The recipe

| knob | value | why (FL ≠ centralized) |
|---|---|---|
| **Local epochs E** | **1** | All 4 judges: more E ⇒ more client drift under non-IID on a 33M fine-tune. E=2 is a compute-headroom variant only. |
| **Rounds R** | **30** (D17 ≥30); also report **R=15** | R=15 = 15 data-passes = PARITY with centralized (the honest headline); R=30 = the D17 reference. |
| **Client optimizer** | **AdamW**, wd 0.01 | Match bb02d's optimizer (it used AdamW). Adam-family only (reproduction-fidelity decision — no SGD). |
| **Client LR** | base **1e-3**, **global cosine over rounds**, **3-round warmup**, floor 0.05× | No single trajectory in FL ⇒ a per-round global schedule (the server broadcasts the round's LR); bb02d's cold-start needs warmup. |
| **grad-clip** | **35** (NOT 1.0) | Match centralized. Judges: over-tight clip (1.0) on the 25-client "huge-batch" aggregate ⇒ under-convergence (repo's "global-64 under-converges"). |
| **backbone-lr-mult** | **0.1** | The heavy Swin-T trains at 0.1× the head LR (the centralized lever; stabilizes the trained backbone in FL). |
| **Server optimizer** | **FedAdam** (D17), η=**0.01**, β1 0.9, β2 0.99, **τ 0.01**, **5-round η-warmup** | The retention lever (server momentum recovers non-IID dilution). η-warmup + larger τ are the judge-mandated guards against the early-round FedAdam blow-up (before v̂ stabilizes, an un-warmed step ≈ η·sign(Δ) on every weight). |
| **Server EMA** | decay **0.9** over rounds | Fair to the EMA'd centralized anchor (0.5656 IS an EMA ckpt). Snapshot BOTH raw + EMA each round; report the **max** (EMA can lag a still-rising curve over 30 rounds). |
| **fraction-fit** | **1.0** (all 25/round) | REQUIRED by D10 for a valid reference checkpoint. |
| **Snapshots** | rounds 10,15,20,25,30 (+ema) | The centralized peaked mid-run (ep15, not ep20) — pick the PEAK round post-hoc, do not assume the final. |

**Server optimizer is an axis ORTHOGONAL to `defense-type`** — `defense-type` stays `none` (D10/T5 contract
intact); FedAdam composes with any defense for the later benchmark.

## Fairness anchor (state it explicitly in the writeup)

Basis = **equal full-data-equivalent passes**. Centralized bb02d = **15 passes** (15 epochs over the 28,130-kf
pooled train set). With full participation and ∪(25 clients) == the pooled set, **1 round (E=1) = 1 data-pass**,
so FL = **R passes**. Report two points:
- **R=15 (15 passes) = PARITY** — the apples-to-apples head-to-head with centralized 0.5656 (the honest headline; the gap = the FL dilution tax).
- **R=30 (30 passes, 2×) = the D17 reference** — FL is allowed more rounds because the FL optimization (33M update vector + non-IID drift) is strictly harder than centralized SGD over the pooled set, and aggregation regularizes (centralized OVERFITS past ep15, so it cannot productively spend 30 passes). Report both; do not lead with the 2×-pass number alone.

The binding downstream floor is **FL ≥ 0.50 mAP after the 25-client non-IID dilution** (a recipe+partition
retention problem, from centralized 0.5656).

## Tuning (cheap probe BEFORE the full run) — the hyperparameters above are starting points

The FedAdam server LR × client LR interaction is the #1 divergence risk; tune it on short runs first
(`run_fl_bb02d_a100.sh` with `ROUNDS=6 EVAL_MODE=every_n EVAL_LIMIT=256 SERVER_LR=.. CLIENT_LR=.. TAG=..`):
1. **Step 0 — measure** (R=2, the base config): real per-round wall-time + trains-clean (no NaN) + VRAM fit.
2. **Primary 1-D sweep (η-warmup ON):** server-lr ∈ {0.003, 0.01, 0.03} at client-lr 1e-3, R=6 — rank by the
   round-6 proxy (car-recall + eval-loss slope); pick the highest non-diverging η.
3. **Client-lr check:** client-lr ∈ {5e-4, 1e-3, 2e-3} at the chosen η, R=6.
4. **τ check** (only if under-converging): τ 1e-2 → 1e-3.
5. **Monitor rare classes** (trailer/CV) in the probe — 6 Singapore clients have zero trailer; if those
   classes collapse, the class-weights (already on) may be insufficient → consider class-balanced client weighting.

Then the winning (η, client-lr, τ) → the full **R=30** run; **≥3 seeds (mean±std)** for the final claim (D16).

## Code built this session (all default-off ⇒ byte-identical baseline; 297 tests pass + CPU flwr smoke green)

- `strategy/server_opt.py` — the **FedOpt server optimizer** (FedAdam / FedAvgM / identity FedAvg) with η-warmup; pure-numpy, fp64, deterministic. `fedavg`+η=1 is a true identity (byte-identical baseline).
- Wired into BOTH the production Flower strategy (`flower_strategies._aggregate_with_core`) **and** the in-process `engine/local_runner.py` — Δ = aggregate − global is form-agnostic (every defense returns new_global as params), so FedAdam composes with any defense.
- **Per-round global client-LR schedule** (`flower_strategies.configure_train` broadcasts the round's LR) + **server-side cross-round EMA** + **per-round snapshot checkpoints** (`server_app`).
- **Client recipe** (`training/loop.train_local`): backbone-LR-mult param-groups + grad-clip + AdamW, wired through `client_app`/`local_runner` (fires only when the backbone is trained ⇒ frozen/dummy stays the flat byte-identical path).
- **STEP 4 slice-map:** `BB02D_TRAINABLE_MODULE_SLICE_MAP` (270) + config-conditional `assert_trainable_layout`.
- **pyproject registration** (the real FL prerequisite): all bb02d model knobs + the recipe knobs added to `[tool.flwr.app.config]` (flwr rejects unregistered keys); `det-class-weights` is a comma-STRING (flwr rejects arrays), uniform→None keeps the default loss byte-identical.
- Provenance records an honest `fl_recipe` block (D10 validation unchanged — FedAdam stays D10-compliant with defense=none).
- Launcher `scripts/run_fl_bb02d_a100.sh` (A100:4, msweep10 cache, writes D10 provenance into every snapshot dir, probe overrides via env).
- Tests: `test_server_opt.py` (12), `test_fl_server_opt_integration.py` (5), `test_fl_config_keys_registered.py` (4), bb02d-layout (1). CPU `flwr` smoke validated the full Flower path (FedAdam + cosine LR + EMA + snapshots) end-to-end.
