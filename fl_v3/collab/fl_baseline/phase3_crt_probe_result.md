# MCR Phase-3 STEP 3 — cRT decoupling probe: RESULT (DECISIVE — representation-limited)

> The decisive head-vs-representation test for the FedAvg-0.247 tail collapse. Freeze the FL-converged
> feature stack, balanced-retrain ONLY the head, eval full-val at the 0.247-baseline protocol. Job 6781365
> (single A100, 1h47m, 2026-06-27). Scripts: `scripts/p3_crt_probe.py` + `scripts/run_p3_crt_probe.sh`.

## Setup (designed so a flat tail = representation limit, not a tuning artifact)
- **Init:** FedAvg round_15 global (`fl_bb02d_fedadam/fl_bb02d_fedavg/round_15/final_model.pt`, the 0.247 model).
- **Freeze:** everything except `head` (0.45% of params, 15 tensors); frozen children pinned `eval()` (deterministic
  frozen features — BN-free Swin+GroupNorm, no DropPath/dropout in the frozen path).
- **Head retrain:** pooled train (28,130 kf), **CBGS 1.44×** (tail repeat 1.56–1.65) + **tail-weighted focal**
  (class-weights 1.3–2.5×) + **AdamW lr 3e-3 OneCycle** (the centralized peak — rules out under-stepping) +
  **EMA 0.999**, **2 epochs** (loss 1.895→1.843, converged; head classes already regressing ⇒ not under-trained).
- **Eval:** `t4_readiness_eval --diagnostic`, full val (6019 kf), score 0.01 / maxobj 500 — SAME as the baseline.

## Result — mAP 0.2677 (raw == ema), the tail did NOT recover

| class | FedAvg base | cRT | central | cRT−base | cRT/central |
|---|---:|---:|---:|---:|---:|
| car | 0.743 | 0.661 | 0.85 | −0.082 | 78% |
| barrier | 0.429 | 0.371 | 0.65 | −0.058 | 57% |
| traffic_cone | 0.556 | 0.540 | 0.72 | −0.016 | 75% |
| pedestrian | 0.606 | 0.599 | 0.80 | −0.007 | 75% |
| truck | 0.107 | 0.095 | 0.48 | −0.012 | 20% |
| bus | 0.041 | 0.051 | 0.53 | +0.010 | 10% |
| trailer | 0.000 | 0.012 | 0.22 | +0.012 | 5% |
| construction_vehicle | 0.005 | 0.010 | 0.23 | +0.005 | 4% |
| motorcycle | 0.130 | 0.177 | 0.68 | +0.047 | 26% |
| bicycle | 0.146 | 0.161 | 0.42 | +0.038 | 38% |
| **mAP** | **~0.276** | **0.2677** | **0.566** | −0.008 | 47% |
| NDS | — | 0.2986 | 0.573 | — | — |

(raw checksum 95c447d2; ema d29c1adc; both eval to identical AP to 3 dp.)

## Interpretation — REPRESENTATION-LIMITED (decisive)
Given the head **maximum** balanced signal, the tail moved only marginally off the floor (trailer 0→0.012,
CV 0.005→0.010, bus 0.041→0.051 — still **4–10% of centralized**) and ONLY by trading away head-class AP
(car −0.08, barrier −0.06) ⇒ a tail↔head reshuffle with **no net gain** (mAP −0.008). The head retrained
(loss dropped, tail off-floor, head regressed) — it learned all it could. ∴ **the bottleneck is the frozen
non-IID FL representation, not the classifier head.** Head-only fixes (cRT / federated-cRT / CReFF / NorCal)
are **INSUFFICIENT**. This confirms the investigation's adversarial caveat (CCVR's own limitation + RUCR
TIFS'24: head fixes are bounded by representation quality) and is the STRONGER result: non-IID damage in
**dense 3D detection is representational**, unlike the 2D-classification long-tail line where cRT usually
recovers most of the gap.

## Implication for the path to ≥0.50 mAP (the lever map, updated)
- ✗ **Lever 1 (cRT / head-only) — RULED OUT as a sufficient fix** (this probe). Keep as a published negative.
- → **Lever 2 (FedAvgM, small β) + more rounds** — let the FEATURES keep learning the tail (R=15 under-converged,
  LR→0 by r15) and accumulate the consistent-weak tail signal INTO the representation across rounds.
- → **Lever 3 (training-time): FedVLS vacant-class KD** (preserve tail-class features on the 6/25 clients with
  zero trailer, etc.) + **class-balanced local focal** during FL — fix the tail at the representation source.
- → **Pragmatic guaranteed-≥0.50: warm-start FL from the 0.5656 centralized model** + measure retention (the
  kickoff's centralized→FL framing) — orthogonal to the scientific from-scratch fix.

## Rigor note / cheap follow-up
The baseline comparison uses the recorded full-val per-class table (mean ~0.276; the FL in-run headline was
0.247). To make the cRT−baseline delta airtight (identical decode + eligible-set), re-eval round_15 with the
SAME `t4_readiness_eval` (~25 min single A100). The QUALITATIVE verdict (tail at 4–10% of central ⇒ not
recovered) does not depend on it.
