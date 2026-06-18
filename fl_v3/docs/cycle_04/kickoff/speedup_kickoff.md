# SPEEDUP + CLEAN-BASELINE DIAGNOSTICS session — paste into a fresh session (OUTSIDE the T<N> sessions)

> A dedicated infrastructure + diagnostics session, run in parallel with the paused T5. NOT a T<N> build
> session, NOT a Codex review. Charter by the orchestrator; decision record **D14** (supersedes the
> caching-first framing of D11/D12 and the GH200-timing of D13). Read first: `../decisions.md`
> **D14, D13, D12, D9, D10**; `fl_v3/collab/T5/speedup_analysis.md`; `fl_v3/docs/determinism.md`;
> `fl_v3/collab/T4/SPEC.md §5c` (the `batch_size=1` decode protocol) + `fl_v3/collab/T5/REVIEW.md`.

## Why this session exists (the situation)
T5's camera-only relocation backdoor **did not reach viability** (relocation/trigger_only ≈ 0 ASR,
label_only weak, delete sub-threshold). That null is **uninterpretable** until we exclude four confounds —
architecture robustness, **FL undertraining** (15 rounds, proxy still climbing), **weak recipe** (official
mAP/NDS ~0.13), and **implementation issues**. **Do NOT overclaim "BadFusion doesn't transfer."** Before
any attack redesign or any defense matrix, this session must answer **5 questions**: (1) where is runtime
spent; (2) how much does disabling per-round eval save; (3) does TF32 give useful speedup under a
reproducible regime; (4) is the weak T5 due to FL-undertraining / weak-recipe / attack-design; (5) what is
the correct clean baseline before the next attack.

## The non-negotiable constraint
**Determinism is sacred.** Anything feeding a SCIENTIFIC run keeps same-seed→byte-identical and the
null-config reproducing the regime's reference checkpoint. Profiling/instrumentation must NOT change the
training trajectory. **No mixing numeric regimes** within any scientific comparison.

---

# Phase 1 — make runs fast (A → B → C, in order)

### A. Runtime profiling FIRST (settles the bottleneck — D12 only *inferred* it)
Add a 1–2-round smoke/profiling mode (a config flag, e.g. `profile-mode=true`) that prints **per-stage
wall-clock** (CUDA-event timed, `torch.cuda.synchronize()` around each; instrumentation only — must not
alter the compute or the RNG): dataloader wait · image preprocess · camera backbone · camera neck · view
transform · LiDAR encoder · fusion/BEV-neck/head · loss · backward · optimizer step · aggregation ·
**server-side eval** · checkpoint save. Also record: GPU util, GPU mem, CPU util, dataloader-worker util.
**Run it in BOTH FP32 and TF32** (1–2 rounds each) → the per-stage **TF32 speedup** + the **post-TF32
bottleneck**.
- **Acceptance:** a profiling report committed to `fl_v3/collab/speedup/`; the **measured** backbone share
  (replaces D12's inferred "80–90%"); the **measured** per-round server-eval share (sizes B); the measured
  TF32 per-stage win (sizes C). Profiling is determinism-neutral (a same-seed run with profiling on/off →
  identical training checksum).

### B. Config-gated server eval (the free win the speedup analysis missed)
Per-round server eval currently computes **proxy** metrics (`eval_loss`, `proxy_recall_at_2m`) that are
**NOT** the scientific result — ASR + official mAP/NDS are post-hoc on the final checkpoint. So make it a
config policy: **`server-eval-mode = none | final | every_n | all`** (+ `server-eval-frequency` for
`every_n`; + the eval-subset size for the cheap curve). **Default for trainval = `none`.** Final official
mAP/NDS + ASR post-hoc remain REQUIRED.
- **Acceptance (the safety check):** a short **same-seed null run** has a **byte-identical training
  checksum** with eval `none` vs `all` (only logs/metrics differ). This requires the eval path to be
  **RNG-neutral** — read-only (`model.eval()`, no grad, no optimizer), and it must NOT advance any RNG the
  next client's training consumes (isolate the eval RNG / re-seed deterministically after). ASR stays
  final-checkpoint post-hoc, never per-round.
- **Synergy with E:** `every_n` on a small subset is exactly the cheap per-round curve E needs to *see*
  convergence; production runs use `none`; endpoints get the official post-hoc eval.

### C. TF32 numeric regime — adopt NOW on A40 (D14)
Treat TF32 as an **explicit logged regime**, not a silent flag. Config **`numeric-mode = fp32 | tf32`**;
log `torch.backends.cuda.matmul.allow_tf32`, `cudnn.allow_tf32`, `torch.get_float32_matmul_precision()` at
startup + into every run manifest. When `tf32`: set **both** `allow_tf32` flags **inside
`enforce_determinism`** (= `set_float32_matmul_precision('high')`; **never** `'medium'`/bf16x3; never
per-call), keeping `use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + the static
AST ban + the flash-attn ban.
- **Step 0 (do before anything scientific): run the A40 TF32 det-gate** — `scripts/tf32_det_gate_a40.py`
  via SLURM **on a real A40** (it was only tested on the login T4, where TF32 silently falls back to FP32 →
  a false pass). It must show cc≥8, no-raise under strict, run-to-run byte-identity, and TF32≠FP32. Commit
  the A40-TF32 gate output + a NEW TF32 reference checksum.
- **Then re-establish the reference chain IN TF32** (so D/E *are* the new reference, not redone): the FL
  determinism gate, the clean FL reference, readiness, the frozen ASR subset, null-config identity, and all
  later attack/defense cells — **all in `tf32`**. **Do NOT mix `fp32` and `tf32` in one comparison.**
- **Before any TF32 (or FP16) attack/defense cell is *reported*:** run the D13 **defense-decision
  seed-robustness check** — ≥3 seeds × one FLAME cell + one MultiKrum cell, log the per-round
  admitted/dropped/selected client sets + the headline ASR, show them **seed-stable above the ~2e-3
  precision floor** (report any genuinely precision-sensitive headline cell in FP32). Pre-register it in
  the methods.
- **FP16/bf16 = a MEASURED contingency only** (if A's profiling shows TF32's A40 win is insufficient — it
  is ~1.3× end-to-end on the A40, the worst card): prefer **bf16** over raw FP16-AMP (no loss-scaling,
  FP32-range), **or** run **FP16/bf16 on the FROZEN BACKBONE only** (the bottleneck, eval-mode/no-grad —
  most of the speed, trainable gradients stay TF32). Decide the form AFTER profiling; the larger
  perturbation makes the seed-robustness check more important; it needs the same gate re-run + a new
  checksum. Do NOT reflexively jump to FP16.

---

# Phase 2 — the clean-baseline diagnostics those enable (D, E — in the adopted TF32 regime)

### D. Centralized full-data baseline (clean FIRST, then a gated attack)
Same model, data split, preprocessing, official evaluator, and the `batch_size=1` readiness/eval protocol
— **matched budget: centralized epochs == FL rounds** (at 1 local-epoch/round + full participation, R FL
rounds ≈ R epochs of data exposure; the ONLY difference is the FedAvg averaging — which is what this
isolates).
- **D1 — centralized CLEAN** (15 epochs, to pair with 15-round FL; in TF32). Report: official mAP/NDS, car
  recall, ASR-eligibility count, false-disappearance, training time, vs the FL-15-round reference.
- **D2 — centralized ATTACK, GATED on D1.** Run the camera-only backdoor centrally **only if D1's clean
  detection clears the readiness bar** (eligible-count ≥ N_min AND car recall > the declared floor — no
  point attacking a model that can't detect cars). Same matched budget. **Interpretation:** fails centrally
  too → the *architecture* defeats it (FL-independent robustness); works centrally but dies under FL → **FL
  averaging dilutes it** (directly the plan's Q2 dilution hypothesis — a more interesting result).
- **Acceptance:** the centralized-vs-FL comparison holds budget constant (epochs == rounds); D2 is skipped
  + reported as "centralized clean below readiness" if D1 fails the bar.

### E. Clean FL convergence diagnostic (is 15 rounds enough?)
Clean FL references at **15 and 30 rounds** (30 if feasible given the Phase-1 speedup; fewer rounds = smoke
only). Per-round: the **cheap proxy curve** (B's `every_n`-small-subset) to *see* the convergence shape;
**endpoints: official mAP/NDS + car recall + eligible count + false-disappearance + runtime** (post-hoc, in
TF32). Pair each FL budget with its **matched-epoch centralized** (D): 15-round↔15-epoch, 30-round↔30-epoch.
- **Acceptance:** a clear verdict — is 15 rounds scientifically sufficient or a minimal engineering
  checkpoint, and what round budget the real clean reference should use.

---

## Boundaries (D14)
1. **No major T5 camera-only attack development** until profiling (A) + the clean baselines (D, E) are complete.
2. **No T6/T7 defense matrices on a non-viable attack.**
3. **mini/smoke ≠ scientific evidence.**
4. **No mixing numeric regimes** (everything scientific in one `numeric-mode`).
5. **No feature caching** unless a storage allocation AND a cache-vs-live bit-identity gate both exist (de-prioritized by D14 — TF32 supersedes it).

## Forward (parked — NOT this session)
If the diagnostics confirm camera-only is genuinely non-viable (not a confound), the likely pivot is a
**camera+LiDAR** poison — a **D2 / threat-model** decision, held until the diagnostics land.

## When you have results
Commit to `fl_v3/collab/speedup/` + `findings_log.md`: the profiling report (A) with the measured backbone
+ eval shares + the TF32 per-stage win; the eval-disable saving + the byte-identical-checksum proof (B);
the A40-TF32 det-gate output + new checksum (C); the centralized clean (+ gated attack) numbers (D); the
15-vs-30-round convergence verdict (E) — and an explicit **answer to the 5 questions** + a go/no-go +
recommended budget/regime for the next attack design.
