# MODEL CAPABILITY + RECIPE (MCR) session — paste into a fresh session (OUTSIDE the T<N> sessions)

> Arrhenius update (2026-07): this kickoff records the historical Alvis/A40 D16
> precision plan. The active GH200 policy is now `fp32` for dev/debug/reference and
> `fp16` AMP + `GradScaler(init_scale=512)` for supported sparse training. Direct
> sparse `bf16` is unsupported in the validated cumm/spconv path.

> A dedicated capability + recipe + infrastructure session, sibling to the (now-closed) speedup session.
> NOT a T<N> build session, NOT a Codex review. Charter by the orchestrator; decision record **D17**
> (with **D15** = the speedup/diagnostics findings it builds on, **D16** = the bf16-AMP precision regime
> it runs in). **Read first, in order:** `../decisions.md` **D17, D16, D15, D14, D10, D9, D1**;
> `../../../collab/speedup/D15_D16_decision_for_orchestrator.md` + `speedup_session_findings.md`;
> `../../../README.md` (fl_v3 layout); `../../determinism.md`; `../../../collab/T4/SPEC.md` (readiness +
> the `batch_size=1` eval protocol); `../../../collab/T5/REVIEW.md` (why the pilot null was uninterpretable).

## Why this session exists (the situation)

The D14 speedup/diagnostics session (D15) exposed **two problems that share one root**:

1. **The model is too weak.** At matched budget (epochs == rounds), official metrics are:

   | setting | budget | mAP | NDS | car recall |
   |---|---|---:|---:|---:|
   | **Centralized** | 15 ep | **0.360** | 0.357 | 0.93 |
   | FL (E-15) | 15 r | 0.126 | 0.169 | 0.85 |
   | FL (E-30) | 30 r | 0.196 | 0.226 | 0.89 |

   Two *separate* gaps: (a) the **centralized ceiling itself is only 0.36** — frozen ImageNet Swin-T,
   batch 16, (almost certainly) single-sweep LiDAR + modest resolution/grid; SOTA BEVFusion is ~0.68. (b)
   FL-30 is ~1.8× *below* centralized = **FedAvg dilution** over location-coherent non-IID shards. The
   T5 backdoor ran on a doubly-compromised (undertrained + diluted) checkpoint → its null was
   **uninterpretable** (D14). **You cannot attack/defend a model that can barely detect cars.**

2. **GPU utilisation is stuck at ~12% (A100) / ~54% (A40) and overcommit is a MEASURED dead end** (D15
   E1/E2/E3): without CUDA MPS, separate Ray processes time-slice the GPU, and each step is launch-bound
   at batch-16 with a **frozen** backbone, so packing more clients/GPU gives ~1.0× throughput. `num-gpus<1`
   was never going to help.

**The root is the same:** a too-light, frozen-backbone model that can neither keep a GPU busy nor reach a
strong mAP. **Training the camera backbone fixes BOTH** — the step becomes GPU-heavy (util rises on its
own, the A100's 1.63×/step finally lands) *and* the detector gets stronger. So this session **raises model
capability, harvests the throughput as a side effect, then closes the FL dilution gap, and produces the new
clean FL reference** that T5–T7 will bind to.

## The non-negotiable constraints (D16 + Rule #2)

- **Precision = bf16-AMP, single regime (D16).** bf16 heavy ops + **fp32 stability ops** (focal
  `log(sigmoid)`, L1-over-log-dims, BEV scatter accumulation, optimizer). This is the *only* science
  regime — **no mixing precision regimes** in any comparison. TF32 is dropped (redundant under bf16).
- **Criterion = claim-reproducibility, NOT byte-identity (D16).** Report **≥3 seeds (mean±std)**; a claim
  is valid if it clears the **seed-variance floor**. The strict byte-identical knob (`precision=fp32`) +
  the static-AST ban are an **offline dev-regression tool only** — run them when you touch a determinism-
  sensitive op, not as the bar for reported numbers.
- **Every scientific run keeps one trains-clean reasonableness gate** (no NaN/divergence; the fp32-loss/
  accumulation guard) and **logs `precision`** into its manifest. Speed that NaNs is worthless (D15).
- **Tooling envelope: the binding bar is now `maintained` + `builds on the target tier (x86 now,
  ARM/H200 next)` + `doesn't NaN` — NOT bit-determinism (D16).** Use modern **in-tree** acceleration
  aggressively; avoid **out-of-tree fragile CUDA extensions** (each is a brick in the ARM-rebuild wall).
  The full ADOPT / CAVEAT / KEEP-OUT list is the next section — read it before Phase 1, it re-derives the
  old "banned ops" list under the new regime (several bans were determinism-motivated and are now stale).
- **mini/smoke ≠ scientific evidence** (Rule #3). Every reported number is trainval-scale.

## Tooling envelope (D16-refined — verified 2026-06-21, workflow `wf_be1c09cf-537`)

The old "banned ops" list mixed **determinism** (now relaxed by D16) with **portability/maintenance** (still
binding). Re-derived against current (2026) library state + ARM/H200 build feasibility:

**ADOPT — in-tree, portability-safe, use aggressively (all ship in the standard PyTorch cu12x wheel,
incl. the aarch64+H200 one — zero extra build):**
- **Fused attention via PyTorch SDPA** (`F.scaled_dot_product_attention`) — **the single highest-value
  lever now that D17 unfreezes the backbone** (the backbone fwd+**bwd** becomes GPU-dominant; D15's
  "10%-GPU / overcommit-dead" was measured on a *frozen* backbone). **NOT the external `flash-attn`
  package** (x86-only source build, no ARM wheel — and it wouldn't help anyway). **Caveat (verified):**
  Swin-T's *additive relative-position bias* is rejected by the FLASH backend (FA-2 takes only
  none/causal/padding); it routes through **EFFICIENT_ATTENTION** via the float `attn_mask` ⇒ a realistic
  **~1.3–2×** + lower activation memory, *not* the 2–4× bias-free attention gets. torchvision `swin_t`
  uses manual fp32 math today → needs a **deliberate rewrite** of `ShiftedWindowAttention` to call SDPA
  with the rel-pos-bias as `attn_mask`; verify numerics vs the trains-clean gate. (This un-bans the stale
  `determinism.md` "SDPA→MATH-only" rule — MATH-only threw away the whole point under bf16.)
- **bf16-AMP** (already the D16 science path), **`channels_last`** memory format on the conv-heavy
  PointPillars/ConvFuser/BEV-neck/CenterPoint stack (~1.1–1.3× Tensor-Core win, free), **fused Adam**
  (`fused=True`, ~10–15% on the now-non-trivial unfrozen-backbone optimizer step), **activation
  checkpointing** (`torch.utils.checkpoint`, `use_reentrant=False`) on the trained Swin-T — trades
  ~20–30% recompute for the VRAM headroom that lets bf16+compile stop OOMing and enables a bigger
  effective batch (which *also* fights FedAvg dilution).
- **EMA of weights** (`torch.optim.swa_utils`) — **a genuinely missing capability lever**: cheap, in-tree,
  directly attacks the undertraining + FedAvg-dilution D15 diagnosed, and **tightens the ≥3-seed variance
  band** D16 now reports against. The MCR session should not skip it.

**ADOPT-WITH-CAVEATS — in-tree but validate on the ARM/H200 box before binding to the science path:**
- **`torch.compile` (default mode)** — second-biggest speedup after AMP (fuses the windowed-attention +
  LSS pointwise chains), but not bitwise-eager-equal and recompiles on shape changes. Keep it an **opt-in
  flag with a verified eager fallback**; Inductor/Triton on aarch64 is the least-burned-in path.
- **`torch.compile(mode="reduce-overhead")` / CUDA graphs** — scope to the **static-shape camera/BEV
  subgraphs only** (LiDAR point/voxel counts vary → graphing the voxelizer is unsafe; leave the ragged
  loss out). `max-autotune` is **not** worth its compile-time/portability cost for an iterating codebase.
- **ARM-wheel hygiene:** pin a **release** `cu128` aarch64 torch wheel in the manifest, **never a nightly**
  (aarch64 cu128 nightlies had multi-week build outages in 2025); stock-PyPI aarch64 was CPU-only until
  ~torch 2.11, so use the `download.pytorch.org/whl/cu128` index.

**GATED IN-TREE — a measured ablation, only if a *measured* bottleneck shows up (don't do pre-emptively):**
- **Dynamic voxelization** (no per-pillar max-points cap) via **native `torch.scatter_reduce`** (NOT the
  `torch_scatter` extension). Implement as an **order-free `amax`** reduce (the existing `index_copy_`+`max`
  pattern in `lidar_encoder.py`, generalized to no cap) — *not* `scatter_add(sum)`, so it stays
  atomic-free-by-construction. Capability gain is **modest (~≤1–2 mAP, small-object-biased)** and likely
  inside the seed band at current scale → run it **after** the headline capability work proves pillar-cap
  point-dropping is actually limiting. Watch the dynamic-shape interaction with `torch.compile`.
- **LiDAR capacity, if it's ever the proven bottleneck → an in-tree DENSE upgrade** (PillarNet-style taller
  pillars / residual 2D BEV blocks), **NOT spconv** (see below).

**KEEP OUT — portability/maintenance (NOT determinism); none costs us speed on this model:**
- **`flash-attn` package** — SDPA covers the need in-tree; flash-attn is an x86-only source build with no
  official ARM wheel, and can't take Swin's float bias anyway.
- **spconv / torchsparse (sparse-voxel LiDAR)** — **verified keep-out**: maintenance-mode/stale (spconv last
  release 2024-12), **no aarch64 wheels** (manual `cumm`/`pccm` sm_90 source build on ARM, zero upstream CI),
  and its kernels live outside the PyTorch dispatcher so they **silently break the offline strict dev tool
  with no recovery path**. The real ~+5–8 mAP LiDAR-only gain is *not* our binding constraint (FL-undertraining
  + the Swin-T forward are) and is largely recoverable via the in-tree dense upgrade above.
- **`mmdet3d` / `mmcv`** (Rule #2) — a *framework*, not a kernel; none of Swin-T/LSS/PointPillars/ConvFuser/
  CenterPoint needs an mmcv kernel (the one off-architecture candidate, `ms_deform_attn`, has a pure-PyTorch
  fallback). Unmaintained since 2024, open unfixed CUDA-13 `_ext` build break, no ARM wheels.
- **FP8 / Transformer-Engine** — external source-built CUDA ext, no clean aarch64 wheel, and **1–2%+ accuracy
  degradation on small conv-heavy nets** — our centralized ceiling is already only 0.36 mAP, we cannot trade
  accuracy for throughput. bf16-AMP is the right precision floor; do **not** chase FP8.
- **DALI** — real dataloader win but an out-of-tree NVIDIA binary; fix the (D15-flagged) dataloader **in-tree**
  instead: `num-workers`↑ + node-local staging of `samples/`/`sweeps/` + `persistent_workers` + prefetch.
- **NestedTensor/jagged** for ragged LiDAR/boxes — still prototype (single ragged dim, narrow op coverage);
  the existing dense collision-free `index_copy` scatter already handles raggedness deterministically.

**The offline strict dev tool still works** — it forces a deterministic path per lib: SDPA→`sdpa_kernel(MATH)`,
`torch.compile` off, CUDA graphs off, `channels_last` off, optimizer `fused=False/foreach=False`,
`use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`. (spconv/mmcv have no such knob — a
second reason they stay out.)

---

# Phase 0 — land the ratification (mechanical, do FIRST)

The D15 infra is already on `v3-ad-perception` (HEAD `1bf9015`): the `determinism-level`/`numeric-mode`
regime, the scatter_add LSS rewrite, `centralized_train.py`, the profiler. Phase 0 cleans it up:

- **Execute the D16 config-collapse — on the CLEAN science path ONLY:** collapse `numeric-mode {fp32,tf32}`
  × `determinism-level {strict,relaxed}` → ONE `precision = bf16 | fp32` knob (bf16 = science, fp32 =
  dev/deterministic) across the training loop, centralized, FL, readiness eval, provenance, and gate scripts.
  Default the **science path to bf16**. **Do NOT touch `src/fl_v3/attacks/`, `scripts/t5_*`, `configs/t5_*`,
  or `tests/test_attack_*` — T5 is a separate, paused task (see Boundaries).** `tasks.py`/`viz/` *lazily*
  import the attack package; keep a **back-compat shim** so those dormant imports still resolve under the new
  knob — i.e. don't break the import, don't refactor the T5 path.
- **Confirm:** the suite still passes under the collapsed knob; the **strict-knob byte-identity regression
  tool still works** (it's your safety net for determinism-sensitive edits, even though it's no longer the
  science bar).
- **Acceptance:** one `precision` knob, bf16 default, manifest logs it; the clean path migrated; the dormant
  T5 imports still resolve unchanged; tests green.
- **(Parked, NOT this session — a T5-restart prerequisite):** `t5_attack_eval.py` doesn't thread the
  precision mode (D15 hazard). It belongs to T5; fix it at T5-restart, not here. T5 won't run during MCR.

---

# Phase 1 — raise the CENTRALIZED ceiling (the core capability search, in bf16-AMP)

Start from `centralized_train.py` (the 0.36-mAP baseline). **This is where the model becomes strong, and it
is the most portable, reusable artifact** (a recipe, not a GPU-pinned checkpoint — it survives the Arrhenius
migration intact). Search the capability levers, **measuring official mAP/NDS + car recall + ASR-eligible
count at each step** (the T4 evaluator, `batch_size=1` protocol). Ablate one-at-a-time, then a combined
recipe; report the lever→mAP table.

**The lever menu (ranked by expected mAP-per-cost; you decide order from Phase-0 profiling):**
1. **Unfreeze / train the camera backbone (the headline lever, AMENDS D1).** This is what the user asked
   for ("wake up the camera backbone") and the one that *also* fixes util (GPU-heavy step). Use **LR
   groups** (lower LR on the pretrained backbone than on the from-scratch fusion/head); consider a staged
   schedule (warm the head, then unfreeze). ImageNet-init is the sane start (random-init is a separate,
   harder ablation). **Watch BN:** D6 ran the frozen backbone's BN in eval mode — once trained, decide
   BN-train vs. freeze-BN vs. switch backbone norm to GroupNorm (FedBN is the FL-time diagnostic).
   **Pair it with the SDPA rewrite (tooling envelope):** the now-trained windowed attention is the new
   GPU-dominant hotspot and still runs torchvision's manual fp32 math — route `ShiftedWindowAttention`
   through `F.scaled_dot_product_attention` (rel-pos-bias as `attn_mask` → EFFICIENT backend) for the
   in-tree fwd+bwd speedup + the activation-memory headroom that backbone-training needs.
2. **LiDAR multi-sweep accumulation (single → 10-sweep).** A known large nuScenes mAP lever; deterministic
   (timestamped concatenation). Check the per-step cost — it widens the LiDAR encoder input.
3. **Image resolution + BEV-grid resolution.** Big mAP *and* big cost levers (BEVFusion gains a lot here).
   Raise carefully against the VRAM/throughput budget from Phase 2.
4. **Fusion-layer redesign (D3 ConvFuser).** The user explicitly named "fusion layer design." Deeper
   ConvFuser / channel-attention (SE) / a stronger BEV-neck. Keep it a clean named module (D3) and
   determinism-safe. *Note:* the ConvFuser is also T5's attack surface — keep its interface
   **backward-compatible with the (dormant) attack hooks**; if a redesign *requires* changing that
   interface, **escalate to the orchestrator** (Boundaries), don't refactor the T5 path in-session.
5. **Optimizer / LR / schedule / EMA / weight-decay / grad-clip.** The "hyperparameters." Longer schedule,
   cosine + warmup, **EMA of weights** (`swa_utils`) — the tooling envelope flags this as a missing lever:
   cheap, in-tree, and it directly counters the undertraining + FedAvg-dilution D15 diagnosed while
   tightening the ≥3-seed variance band. **Keep Adam** unless there's a measured reason to switch (memory
   `feedback_reproduction_platform_fidelity`: don't chase a paper's SGD); use `fused=True` (envelope).
6. **Data aug (image + BEV: flip/rotate/scale, GridMask).** Helps generalization but **interacts with the
   camera trigger later** — document exactly what aug is on, because T5's trigger placement assumes a known
   preprocess.

**Target.** Not a magic round number — a detector **clearly stronger than 0.36** with **high car recall and
a large ASR-eligible count**, because the *purpose* is attack/defense credibility: enough reliably-detected
cars that a disappearance-ASR signal is interpretable. Report where the ceiling lands and what it cost; the
orchestrator sets the "good enough" bar from your data.

**Guardrail.** Every lever stays inside the **tooling envelope** above (maintained + builds-on-target-tier +
no-NaN; in-tree over fragile extensions). The LiDAR encoder stays **dense PointPillars** by default (spconv
is keep-out — no aarch64 wheel + breaks the strict dev tool); if LiDAR capacity is ever the *proven*
bottleneck, the lever is an **in-tree dense upgrade** (PillarNet-style), not spconv. Dynamic voxelization is
a **gated in-tree ablation** (order-free `amax`), not forbidden — run it only if pillar-cap point-dropping is
shown to limit accuracy.

---

# Phase 2 — throughput to AFFORD Phase 1 (interleaved, NOT a separate stage)

Training the backbone makes each step **GPU-heavy** (good — util rises, the A100 pays off) but also more
**expensive** (backbone gradients + Adam moments + activation graph → more VRAM, slower step). The D15
"pivot" levers apply — **cheaper steps, NOT more clients, NOT reflexive big-batch**:

- **In-tree eager wins FIRST (envelope) — free, portable, no compile risk:** SDPA-on-the-backbone (the
  big one — see Phase 1 lever 1), `channels_last` on the conv stack, fused Adam, and **activation
  checkpointing** on the trained Swin-T (buys the VRAM that stops bf16+compile OOMing and lets the effective
  batch grow against dilution). These stack to ~2–3× over fp32-eager with zero portability cost.
- **`torch.compile(mode="reduce-overhead")` / CUDA graphs — opt-in, scoped (envelope).** Attacks the
  launch-bound step directly, keeps the update count high (right for an undertrained model). Static-shape
  **camera/BEV subgraphs only** — **leave the ragged-box loss + the variable-count LiDAR voxelizer out.**
  Gate behind a flag with a verified eager fallback; Inductor/Triton-on-aarch64 is the least-burned-in path,
  so **validate compile on the ARM/H200 box before binding it to the science path**, and pin a **release**
  `cu128` aarch64 wheel (never a nightly).
- **num-workers↑ + node-local data staging** (`/dev/shm` or job-local `/tmp`/NVMe) to beat the 0.60× Mimer
  shared-FS contention. At 1 client/GPU you can afford `num-workers=8` (only 4 loaders/node, no host-RAM
  wall — unlike the overcommit case that OOM'd).
- **Move heavy runs to A100 / A100-fat.** A trained backbone needs the VRAM (the A40's 46 GB may not fit
  backbone-training at a useful batch + resolution), and the A100's 1.63×/step finally lands once util
  rises. Both A100 and A40 nodes have 244 GB host RAM (D15 correction).
- **Re-establish a per-tier reproducibility gate in bf16** (the bf16 analog of the A40 byte gate): on the
  tier you actually use, confirm two same-seed runs land **within the seed-variance band** (NOT byte-
  identical now) + the trains-clean gate + log the precision/device. Determinism is architecture-pinned
  (D9) — record the tier on every run.
- **Forbidden (measured dead ends, D15):** GPU overcommit (`num-gpus<1`) — serializes without MPS; reflexive
  big-batch — trades away gradient updates (if you raise batch, scale LR and add *rounds*, not local epochs).

**Acceptance:** a per-step + per-round profile of the *trained-backbone* model on A40 and A100 (util should
no longer be 12%); the cheaper-step win measured; the chosen heavy-run tier + its bf16 repro gate recorded.

---

# Phase 3 — FL recipe + the new clean ≥30-round bf16 FL reference

With a strong centralized recipe locked, transfer to FL and close the **FedAvg dilution** gap (centralized
0.36 ≫ FL-30 0.20 at matched budget — D15). Levers:

- **Server-side momentum: FedAdam / FedOpt / FedYogi** (the biggest known lever against FedAvg dilution on
  non-IID — Reddi et al.). This is the prime suspect; try it first.
- **Round budget ≥30** — the diagnosis: 15 is undertrained, 30 still climbing (r27→r30 slope +0.005/round).
  **Find the plateau** (likely 40–60); the new reference's round count is an output of this phase.
- **Local epochs** (1 vs more): more local work worsens dilution but fewer needs more rounds — measure the
  tradeoff; do not silently change it.
- **Non-IID severity:** keep the **log-group (location-coherent) partition as PRIMARY** — it is the threat
  model, and full participation (D10) already removes the sampling-variance confound. Characterize a milder
  partition only as a *diagnostic* if needed.
- **The federate-vs-freeze backbone fork (bring this back as a D-decision, with data).** Measure both:
  (a) **federate the trained backbone** (strongest model, much higher FL cost, **backdoor can live in the
  backbone** = broader/realistic attack surface), vs (b) **central-pretrain → freeze → federate only
  fusion+head+neck** (cheaper FL, **preserves the D1/D3 fusion-aware attack framing**). Report the mAP cost
  of freezing and the threat-model implication; the orchestrator picks the threat model from your numbers.
- **Multi-seed (≥3, mean±std)** for the final reference (D16), and **characterize the seed-variance floor**
  (T5–T7's defense knife-edge needs it).

**Deliverable:** the **new clean ≥30-round bf16-AMP FL reference checkpoint** (multi-seed) = the anchor T5–T7
rebind to.

---

# Phase 4 — re-baseline the bindings (close-out)

- **Re-judge T4 readiness** on the new checkpoint (`benchmark_readiness.json`; eligible-count ≥ N_min AND
  car recall > the declared floor).
- **Rebuild the frozen held-out ASR subset** — content-hashed, **bound to the new checkpoint** (the old one
  bound to the superseded tf32-strict reference is void).
- **Record the new provenance under claim-reproducibility** — a seed-band + the manifest precision/device,
  NOT a single byte-checksum. Null-config (`poison_rate=0`) reproduces the clean baseline **within the
  seed-variance band** (D16; was bit-for-bit).
- **Hand back** a one-page "T5–T7 are now unblocked" note: the new reference's metrics, the readiness
  verdict, the seed-variance floor, the federate-vs-freeze recommendation, and the locked recipe.

---

## Alvis-sunset realism (built into this charter — read carefully)

Alvis (x86, A40/A100) sunsets **2026-06-30** (≈9 days from the charter date); the migration target is
**Arrhenius (ARM, H200)**. Determinism is architecture-pinned (D9) ⇒ **the final reference must be
(re)produced on the GPU tier T5–T7 will actually run on** — a re-baseline is forced by the migration
regardless. Therefore:

- **The Alvis job is to LOCK THE RECIPE** (Phase 1 centralized ceiling + Phase 2 throughput + Phase 3 FL
  choices). The recipe is portable and is the reusable artifact; **do NOT burn the 9 days producing a final
  A40 multi-seed ≥30-round FL checkpoint that Arrhenius will supersede.**
- **The final multi-seed ≥30-round FL reference is produced at the locked recipe on whichever tier is live**
  when the recipe is done (A100 now if it fits the window; else H200/Arrhenius after migration).
- **Every heavy run is checkpoint-resumable** (save optimizer + scheduler + EMA state, round/epoch counter)
  so a forced mid-run migration resumes rather than restarts. Keep the venv reproducible from the pinned
  manifest (`../../env.md`) — the no-mmdet3d/pure-PyTorch design is what makes the ARM rebuild painless.

This honors the "all-in-one to a new FL reference" scope (you own the whole pipeline through the reference)
without betting the deliverable on finishing backbone-training FL inside 9 A40-days.

## Boundaries (D17)

1. **No T5 restart / T6 / T7** until the new clean reference + its readiness verdict exist.
2. **T5 IS OUT OF SCOPE — do not modify `src/fl_v3/attacks/`, `scripts/t5_*`, `configs/t5_*`, or
   `tests/test_attack_*`.** Those attack hooks in `tasks.py`/`viz/` are lazy + dormant on clean runs — leave
   them intact. If a capability change *requires* touching a T5-shared interface (e.g. the ConvFuser
   signature the attack hooks into, or the `maybe_wrap_for_client` call site in `tasks.py`), **STOP and
   surface it to the orchestrator for discussion — do not refactor T5 unilaterally.**
3. **Every reported number in bf16-AMP, multi-seed** — no mixing precision regimes (D16).
4. **mini/smoke ≠ scientific evidence** (Rule #3).
5. **Stay inside the Tooling envelope** above (maintained + builds-on-target-tier + no-NaN; in-tree over
   fragile CUDA extensions) — NOT bit-determinism (D16, which re-derived the old "banned ops" list).
6. The **federate-vs-freeze threat-model choice returns as a data-backed D-decision**, not a unilateral call.
7. **No feature caching** (D16 dropped it permanently — training the backbone kills the frozen-cache premise).

## When you have results

Commit to `../../../collab/model_capability/` + `findings_log.md`: the config-collapse (clean path; T5 untouched) (Phase 0);
the **lever→mAP ablation table** + the strong centralized recipe (Phase 1); the trained-backbone profile +
the cheaper-step win + the chosen tier/gate (Phase 2); the FL-recipe sweep + the new ≥30-round bf16 FL
reference + the **federate-vs-freeze data** (Phase 3); the re-judged readiness + rebuilt ASR subset + new
provenance (Phase 4) — and an explicit **go/no-go for restarting T5**, the recommended threat model, and the
locked recipe + round budget + seed count.
