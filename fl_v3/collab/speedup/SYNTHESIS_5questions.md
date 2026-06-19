# SYNTHESIS — the 5 questions, go/no-go, recommended budget/regime (D14 speedup session)

> Status as of 2026-06-18. **LOCKED** = settled this session. **PENDING** = awaiting an in-flight /
> queued SLURM run (finalize next turn). Artifacts in `collab/speedup/`. All scientific runs use the
> single TF32 regime (A40 det-gate PASS); no regime mixing.

## Q1 — Where is runtime spent?  **LOCKED (A profiling, job 6767120)**

Per training step (headline trainval config, frozen Swin-T, batch-16, A40, FP32), mean step ≈ 1931 ms:
**camera_backbone 30.5% · LSS view_transform 31.2% · loss 15.9% · backward 12.2% · dataloader 3.5%**,
rest <2%; forward = 68% of the step. The D11/D12 *inferred* "80–90% backbone" is **false** — the backbone
is ~30%, co-equal with the (memory-bound) LSS view-transform. Run is GPU-compute-bound (dataloader 3.5%,
GPU util ~76%), confirming Flower/Ray overhead is negligible. → `A_profiling_report.md`.

## Q2 — How much does disabling per-round server eval save?  **PARTIAL / PENDING-B**

- **Mechanism LOCKED:** `server-eval-mode = none|final|every_n|all`; trainval default `none`. The gated
  metric is only the SERVER PROXY (`eval_loss`, `proxy_recall`) — NOT the scientific result (ASR +
  official mAP/NDS are post-hoc). Per-round norm/gradient-space log is independent and untouched.
- **Saving:** the legacy per-round full-val proxy eval was ~2 h/run (T5 speedup analysis, batch_size=1
  decode over the full split). With `none` it is **0**; with `every_n` on a small subset it is a few
  minutes total — the cheap convergence curve E needs. Exact wall-clock delta: **PENDING** the E runs
  (which log per-round `every_n` eval time) — finalize next turn.
- **Neutrality (the safety check): LOCKED — PASS (job 6767126).** Same-seed null TF32 run, eval `none`
  vs `all` → **byte-identical FL_TRAINABLE_CHECKSUM `0eed9236842473059cfaf550503202cee31d8c1cfea5d8a1ae8777b04c911c85`**
  (3 rounds, numeric-mode=tf32). So gating server eval does NOT perturb training — server-eval-mode is
  safe to set `none` for trainval. Bonus: this is the first end-to-end TF32 FL run AND it confirms the
  TF32 FL path is deterministic (two runs → identical checksum). → `b_eval_neutral.json`.

## Q3 — Does TF32 help under a reproducible regime?  **LOCKED**

Yes, deterministically, but **modestly on the A40: ~1.12× end-to-end** (A profiling). Confined to the
matmul/conv stages (backbone 1.33×, backward 1.23×, fusion 1.22×); the two largest stages (memory-bound
LSS view-transform 31%, loss 16%) see ~1.00×. A40 TF32 det-gate PASS (job 6767119): no-raise under
strict, run-to-run byte-identical, TF32≠FP32 — so adopting TF32 is a clean re-baseline, not drift. The
~1.12× is below D13's ~1.3× estimate (A40 = the worst TF32 card); Hopper/GH200 (full-rate TF32) will see
a much larger win at the migration. **Decision: adopt TF32 now on A40 (D14) — banked, free, safe.**

## Q4 — Is the weak T5 due to FL-undertraining / weak-recipe / architecture?  **LOCKED — both undertraining AND FedAvg dilution; NOT architecture/recipe**

Official mAP/NDS (TF32, batch_size=1 readiness eval), matched budget (epochs == rounds):

| Setting | budget | mAP | NDS | car recall | eligible N |
|---|---|---:|---:|---:|---:|
| **Centralized (D1)** | 15 epochs | **0.3597** | **0.3569** | **0.93** | 28,505 |
| FL (E-15) | 15 rounds | 0.1263 | 0.1686 | 0.85 | 27,383 |
| FL (E-30) | 30 rounds | 0.1957 | 0.2260 | 0.89 | 28,153 |

- **NOT architecture/recipe.** Centralized reaches **mAP 0.36 / NDS 0.357 / recall 0.93** — a strong,
  capable detector on the SAME model/data/preprocess/budget. The architecture + recipe are fine; the weak
  T5 clean detection (mAP ~0.13) is NOT a capacity ceiling.
- **YES FL-undertraining.** FL-15 → FL-30: mAP **0.126 → 0.196** (+55% relative), NDS 0.169 → 0.226,
  recall 0.85 → 0.89; the proxy curve was still climbing at r30. **15 rounds was undertrained.**
- **YES FedAvg dilution (the Q2 hypothesis, quantified).** Even FL-30 (mAP 0.196) is **~1.8× below**
  centralized-15 (0.360) at matched data exposure → FedAvg averaging (over location-coherent non-IID
  shards) heavily dilutes the model. Caveat (recorded in D's provenance): the gap also includes FL's
  per-round optimizer reset vs centralized warm-Adam, so it implicates the FL *regime* broadly.
- **Verdict on T5:** the camera-only attack was tested on a DOUBLY-compromised checkpoint — undertrained
  (15 rounds) AND dilution-weakened (FL mAP 0.13 vs centralized 0.36). The null is uninterpretable as
  "BadFusion doesn't transfer"; the clean model was too weak. D2 (centralized attack) is now UNBLOCKED —
  centralized clears the readiness bar with huge margin (recall 0.93 ≥ 0.20, N 28,505 ≥ 150).

## Q5 — What is the correct clean baseline before the next attack?  **E-15 LOCKED; E-30 pending**

**The TF32 15-round clean reference is READY (job 6767339) and is the new baseline:**
- checksum `d2d396d22b3a…e92c5e27`; readiness **READY** (scope=reference, numeric-mode=tf32);
  regime-match guard + D10 provenance both verified.
- **official mAP 0.1263 · NDS 0.1686 · car_recall 0.8500 · car_AP@2m 0.6263 · eligible_N 27,383 ·
  false-disappearance 0.0**; new frozen ASR subset hash `ddf12e0f203f2c79…`.
- **TF32 ≈ FP32 in model quality — now EMPIRICALLY CONFIRMED** (not just argued): FP32 ref was
  mAP 0.1253 / NDS 0.1688 / recall 0.85 / N 27,432; TF32 is mAP 0.1263 / NDS 0.1686 / recall 0.85 /
  N 27,383 — differences are ~1e-3 (within seed/precision noise), recall identical. This validates the
  D13/D14 TF32-is-scientifically-safe claim with a real trainval checkpoint.
- **E-30 LANDED — 15 rounds is NOT the right budget.** FL-30 official mAP 0.196 / NDS 0.226 / recall
  0.89 vs FL-15 0.126 / 0.169 / 0.85 — a large, material gain (mAP +55%), and still climbing at r30.
  **Verdict: 15 rounds is an undertrained engineering checkpoint, NOT the scientific clean reference.**
- **Recommended round budget for the next attack design:** **≥30 rounds, and check convergence past 30**
  (the r27→r30 proxy slope is +0.005/round — not yet flat; a 45–60 round run would confirm the plateau).
  The clean reference the attack binds to must be at this converged budget, not 15. This ~2× the rounds
  makes the Phase-1 speed levers (TF32 + loss fix, and any determinism-relaxation unlock) materially more
  valuable — the round budget just doubled.

---

## Go / no-go (interim)

- **TF32 on A40:** GO (adopt now; det-gate PASS; ~1.12× banked; regime logged).
- **Feature caching on Alvis:** NO-GO — measured ceiling ~1.4× (backbone only ~30%), needs 1.66 TB +
  a cache det-gate. Independently confirms D13/D14. Drop the storage rush.
- **Server-eval default `none` for trainval:** GO — B confirmed byte-identical checksum (`0eed9236…`).
- **Next attack redesign / T6–T7 defenses:** HOLD until D + E land (D14 boundaries 1–2).

## Recommended budget/regime for the next attack design (interim → finalize after D/E)

- **Regime:** TF32 (single regime, no mixing), A40, full participation (D10), Path-A 4-GPU.
- **Round budget:** **PENDING E** — default to 15 if E shows convergence by ~r12–15, else bump to 30.
- **The real speedup levers if a per-cell speedup is later needed** (not the backbone): the LSS
  view-transform (31%, memory-bound) and CenterPointLoss (16%) — both caching-free + regime-independent.
  The dominant matrix lever remains D9 across-cell fan-out.

## Determinism review — CLOSED (17-agent workflow, 8/13 confirmed, all addressed)

- HIGH `centralized_train.py` resume shuffle-desync → **FIXED** (per-epoch loader seeded `seed+epoch`;
  verified fresh==resumed epoch-3 checksum `5ea0f138…`).
- MED num-local-epochs!=1 budget mismatch → **FIXED** (assert ==1).
- MED checkpoint↔evaluator regime mismatch → **FIXED** (`t4_readiness_eval.py` reads ckpt provenance
  numeric-mode + RAISEs on mismatch).
- LOW server-eval default → **FIXED** (`t4_reference.json` set `server-eval-mode=none`).

## Forward hazards / required follow-ups (for the orchestrator)

1. **`t5_attack_eval.py` does NOT thread numeric-mode** → it would evaluate TF32-trained T5 poisoned
   checkpoints in FP32 (a forbidden regime mix). **Must be fixed before any TF32 T5 attack eval**
   (2-line mirror of `t4_readiness_eval.py`; filed as a follow-up task chip). Out of scope here (paused T5).
2. **Centralized-vs-FL (D) interpretive caveat:** D differs from FL by (a) NO cross-client averaging AND
   (b) a single warm Adam across epochs (FL rebuilds Adam per client/round). A "works-centrally-dies-
   under-FL" result implicates the FL **regime broadly** (averaging + per-round optimizer reset), not
   averaging in isolation. Recorded in the centralized provenance `matched_budget_note`.
3. **Legacy `a80466c3` is NOT reproduced under explicit-`fp32`** (it ran convs in cuDNN-TF32 by torch's
   implicit Ampere default; explicit-`fp32` turns both flags off). D14 re-baselines in `tf32` (E runs),
   so this is expected — but do not expect `a80466c3` byte-parity from a fresh explicit-fp32 run.
