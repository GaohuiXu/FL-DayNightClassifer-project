# MCR CL — depth-supervised LSS: result (NET-NEGATIVE at weight 1.0, over-weighting confound)

> The #1 structural-audit lever (BEVDepth-style LiDAR depth supervision on the previously-UNsupervised LSS
> depthnet). Clean A/B vs bb02d (only depth-sup differs: global-64 batch, 15 epochs, EMA 0.9997, activation-
> checkpointing = value-identical memory fix). Job 6782278 (4×A100, bf16, 02:44), config `p1_bb02d_depth.json`
> (`det-depth-supervision=true`, `det-depth-loss-weight=1.0`). EMA ep15, full val, score 0.01/maxobj 500. 2026-06-28.

## Result — mAP 0.4003 vs bb02d 0.5656 (−0.165); NDS 0.3633 vs 0.5733 (−0.210)

| class | bb02d | depth-sup w=1.0 | Δ | group |
|---|---:|---:|---:|---|
| car | 0.85 | 0.7619 | −0.088 | abundant |
| pedestrian | 0.80 | 0.7224 | −0.078 | camera |
| traffic_cone | 0.72 | 0.6687 | −0.051 | camera |
| barrier | 0.65 | 0.5665 | −0.083 | camera |
| motorcycle | 0.68 | 0.4212 | −0.259 | rare/vehicle |
| truck | 0.48 | 0.2913 | −0.189 | rare/vehicle |
| bicycle | 0.42 | 0.2560 | −0.164 | rare/vehicle |
| bus | 0.53 | 0.1570 | −0.373 | rare/vehicle |
| trailer | 0.22 | 0.0832 | −0.137 | rare/vehicle |
| construction_vehicle | 0.23 | 0.0751 | −0.155 | rare/vehicle |
| **mAP** | **0.5656** | **0.4003** | **−0.165** | |

## Diagnosis — an over-weighting confound, NOT "depth-sup is useless"
- **Loss balance is the smoking gun.** Epoch-15 total loss 2.86; the depth-CE term (the depthnet learned, CE
  settled ~1.3) at weight 1.0 is **~45% of the total** — nearly as large as the whole detection loss. So the
  optimizer spent ~half its capacity on depth ⇒ **detection trained on ~half the effective gradient** vs bb02d.
  `det-depth-loss-weight=1.0` was simply too high (flagged as tunable; it bit). BEVDepth's depth loss is
  per-pixel-normalized and sits at ~10–20% of its total; our hard-bin CE at w=1.0 is 45%.
- **Per-class pattern confirms it.** Largest drops are the capacity-hungry rare/vehicle classes (bus −0.37,
  moto −0.26, truck −0.19, bicycle −0.16, CV −0.16, trailer −0.14); smallest are the easy/abundant ones
  (car −0.09, ped −0.08, cone −0.05). That is detection-under-training from diverted capacity — NOT a
  camera/LiDAR effect. The camera-dependent classes depth-sup *should* help (ped/cone/barrier) still **dropped**,
  so at this weight there was no net camera benefit.
- Training itself was healthy (loss monotone 7.16→2.86; depth CE dropped, so depth supervision DID learn).
  The plumbing is correct (smoke PASS; forward fp16-safe too). This is purely a loss-weight balance problem.

## Two separable hypotheses
1. **Weight too high (most likely):** depth as a 45%-of-loss task starves detection. → re-test at **w≈0.2**
   (depth ~14% of loss, the BEVDepth-style auxiliary balance).
2. **Depth-sup isn't the lever for THIS LiDAR-dominant fusion:** if the fusion already gets geometry from LiDAR,
   supervising camera depth may not move detection (BEVDepth's gains are on camera-dominant models). If the
   w≈0.2 re-test still ≤ 0.5656 (and camera classes don't improve), this is the conclusion — and it *refines the
   audit*: unsupervised depth is a ceiling for camera-dominant BEV, not necessarily for a strong-LiDAR fusion.

**Decisive next experiment:** re-run identical but `det-depth-loss-weight=0.2`. Recovers to ≥0.5656 (esp. camera
classes up) ⇒ depth-sup works, weight was the issue. Still <0.5656 ⇒ depth-sup is not this fusion's lever.
Cost ≈ one CL run (~2.75 h / ~11 GPU-h). Owner decision given the 2 remaining Alvis days + budget.

## UPDATE (2026-06-29) — w=0.2 re-test: depth-sup is net-negative REGARDLESS of weight (lever CLOSED)
Re-ran identical with `det-depth-loss-weight=0.2` (jobs 6783049 killed on a 2×-slow node → 6783051 on alvis3-13;
eval clipped at the cap → standalone eval 6783160 on the saved ema_ep15). **Result: mAP 0.4031 / NDS 0.3618 —
essentially identical to w=1.0 (0.4003), per-class within ±0.02.** A 5× weight change moved mAP by 0.003 ⇒ the
over-weighting hypothesis is **REFUTED**; weight is not the knob.

| class | bb02d | depth w=1.0 | depth w=0.2 |
|---|---:|---:|---:|
| mAP | **0.5656** | 0.4003 | **0.4031** |
| car/ped/cone/barrier | 0.85/0.80/0.72/0.65 | 0.76/0.72/0.67/0.57 | 0.76/0.71/0.67/0.56 |
| truck/bus/trailer/CV/moto/bike | 0.48/0.53/0.22/0.23/0.68/0.42 | 0.29/0.16/0.08/0.08/0.42/0.26 | 0.30/0.17/0.10/0.10/0.40/0.28 |

**Verified NOT a bug:** `data/nuscenes/augment.py` updates `lidar2img ← lidar2img·T⁻¹` (the BEV aug pulls the
camera projection back), so the depth GT (augmented points via augmented lidar2img → original camera-frame depth)
is aug-consistent. The smoke also confirmed the projection + 73–79% coverage.

**Mechanistic conclusion (lever CLOSED, negative — a real finding):** the **unsupervised LSS depthnet was learning
detection-OPTIMAL feature placement, not metric depth**. Forcing metric depth — even at a light 14%-of-loss weight
— flips the camera-BEV to geometric and disrupts the LiDAR-dominant fusion that had adapted to the old camera-BEV;
detection drops to ~0.40 and stays there independent of weight. This **refines the structural audit**: the
"unsupervised depth is the #1 ceiling" premise is from BEVDepth (camera-dominant); for our **strong-LiDAR fusion
the unsupervised depth is a FEATURE, not a bug**, and supervising it HURTS. Depth-sup is not this model's lever.
(Open, lower-priority: a *soft/uncertainty-aware* depth loss + longer re-adaptation might differ — but not worth
chasing on Alvis vs the clean 15-epoch matched negative.)
