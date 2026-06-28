# MCR Phase-3 — clean FL baseline: result + diagnosis (PAUSED at the path-to-≥0.50 decision)

> The FL baseline number + the FedAdam-vs-FedAvg diagnosis. Federates the locked **bb02d** (centralized
> 0.5656 mAP / 0.5733 NDS) on the healthy 25-client `log_group` partition. Status: **PAUSED** (owner) at the
> decision of how to lift the non-IID-diluted tail to clear the ≥0.50 floor.

## Headline

| run | server-opt | R | peak mAP | NDS | note |
|---|---|---|---:|---:|---|
| FedAdam η=0.01 | fedadam | 15 (timeout@13) | **0.057** | 0.094 | **BUG** — η under-stepped |
| **FedAvg η=1** | fedavg | **15 (full)** | **0.247** | **0.278** | bug fixed; tail diluted |
| centralized bb02d | — | — | 0.5656 | 0.5733 | the reference |

## The FedAdam bug (found via the owner's 1-client sanity check)

1-client FL on all 28,130 frames (iid N=1): both FedAvg and FedAdam clients trained the **identical** full epoch
(loss ~3.1), but the **global** after that epoch was decisive — FedAvg η=1 → car proxy-recall **0.50** (global =
the trained model); FedAdam η=0.01 → **0.00** (global barely moved). **FedAdam normalizes the client delta and
steps only ~η/coordinate**, so η=0.01 froze the random-init head. The FL machinery is correct; **η=0.01 was the
bug** (D17 itself says "start from a FedAvg baseline, THEN FedAdam" — η was 10–100× too small).

## FedAvg R=15 per-class (full val, 0.01/500) — the gap is the non-IID TAIL

| class | FL AP@2m | central | retention | recall@tp |
|---|---:|---:|---:|---:|
| car | 0.743 | 0.85 | 87% | 0.95 |
| traffic_cone | 0.556 | 0.72 | 77% | 0.96 |
| pedestrian | 0.606 | 0.80 | 76% | 0.92 |
| barrier | 0.429 | 0.65 | 66% | 0.86 |
| bicycle | 0.146 | 0.42 | 35% | 0.77 |
| truck | 0.107 | 0.48 | 22% | 0.84 |
| motorcycle | 0.130 | 0.68 | 19% | 0.80 |
| bus | 0.041 | 0.53 | 8% | 0.72 |
| construction_vehicle | 0.005 | 0.23 | 2% | 0.51 |
| trailer | 0.000 | 0.22 | 0% | 0.42 |

**Signature:** HEAD classes (in every client) retain **66–87%**; TAIL classes have **HIGH recall (0.42–0.84 — the
model detects them) but ~0 AP** = detected-but-poorly-calibrated-confidence. This is **genuine non-IID dilution**
(tail sparse; 6 clients have zero trailer; FedAvg averaging dilutes the tail confidence signal) + big-batch
under-convergence (E=1 full-participation ≈ large batch). mAP climbs r8 0.186 → r15 0.247, decelerating
(+0.002/round) → not a bug, a convergence/dilution gap.

## Reproduce / artifacts

- Run: `bash run_fl_bb02d_a100.sh` with `EXTRA_OVERRIDES="server-optimizer=fedavg server-lr=1.0
  server-lr-warmup-rounds=0 server-ema-decay=0.0"`, `TAG=fl_bb02d_fedavg` (config `fl_bb02d_fedadam.json`).
- Snapshots: `…/cycle_04/fl_bb02d_fedadam/fl_bb02d_fedavg/round_{8,10,12,14,15}/final_model.pt` (raw; ema-decay 0).
- Eval: `run_eval_ckpt_a100.sh` with `EXTRA="det-eval-limit=0"` (full val, 0.01/500). FL_TRAINABLE_CHECKSUM
  `7c7639d5…`. NOTE: snapshots lack `provenance.json` (the run completed but the post-run provenance step is in
  the launcher tail; write it before using a snapshot as a D10 reference).

## OPEN DECISION — how to reach ≥0.50 (PAUSED, await owner)

1. **Tune FedAdam η UP** (0.1/0.3/1.0) — the D17 lever done right; server momentum is the designed fix for
   big-batch/non-IID under-convergence; should lift the tail; keeps the from-scratch "pure FL benchmark".
2. **Warm-start from centralized bb02d** — init the FL global from the 0.5656 ckpt, federate → measure
   RETENTION (the kickoff's "centralized→FL retention" framing); reliably ≥0.50; realistic AD-FL; cheapest
   (a `server_app` init-from-checkpoint change). *Recommended for a guaranteed ≥0.50 + the cleaner thesis story.*
3. **More rounds R=30 from-scratch FedAvg** — likely only ~0.30–0.35 (decelerating); least likely to clear 0.50.

## Speed-up record (R1, before the heavy runs)

Per-module teardown + fixes in `phase3_runtime_profile.md`. Net: launcher `PYTHONPATH` bug (was running stale
sibling code) fixed; gradient-space defense metrics gated off (169→~8 s/round); `_load_info` memoized
(build 9→2 s); `torch.compile` is net-negative in FL short epochs (disabled). Per-round ~11.7 min (25-client,
4×A100); centralized ~7 min/epoch is 4-GPU-DDP+compile, which FL gives up (1-client = single-GPU ~39 min).
