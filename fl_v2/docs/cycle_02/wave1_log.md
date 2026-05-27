# Cycle 02 — Wave 1 Log (re-aligned, post-fix)

**Period:** 2026-05-22 to 2026-05-24
**Phase:** mechanism — gradient-space backdoor evasion matrix
**Frozen platform:** GTSRB-43 · ResNet18-from-scratch · 50 clients · Dirichlet α=0.5 · 60 rounds · attack window 10–35 · base poison regime · m=10 malicious clients (20%) · backdoor target class 14 (Stop) · seed 42

## What we did

1. Built the 4-by-5 evasion matrix specification (3 attacks × 5 defenses, neurotoxin dropped per supervisor since GTSRB converges too fast for the durability angle).
2. Implemented + audited the three new components: WS2-DBA (paper-faithful scattered-bars), WS5-FoolsGold (head-layer cosine on cumulative client histories), WS5-FLAME (HDBSCAN clustering + median clipping + calibrated Gaussian noise).
3. Ran the initial 13-cell batch, discovered three implementation bugs by comparing our code against authors' open-source references:
   - **DBA**: corner-grid pattern was geometrically too weak (ASR 0.07) — replaced with paper-faithful scattered-bars (4 non-contiguous 1×6 horizontal bars).
   - **FoolsGold**: cosine computed on the full update — replaced with output-layer (head) only, matching the paper's "indicative features" formulation.
   - **FLAME**: optimizer-aware noise calibration error — paper's λ=0.001 is calibrated for SGD lr=0.01, and applied verbatim under our Adam lr=0.001 caused round-1 divergence. Re-calibrated to λ=1e-6.
4. Diagnosed and fixed two SLURM infrastructure races: SuperLink Control-API vs SuperExec startup ordering, and SuperExec → ServerApp subprocess factory readiness. Hardened `run_alvis.sh` to abort loudly on either failure mode rather than exit-0 silently.
5. Re-ran the affected 9 cells, plus the aligned `pixel × FedAvg` baseline (originally Job 2 at 100r / window 10-50). 15-cell matrix now consistent at 60r / window 10-35.

## What we achieved

| Deliverable | Status |
|---|---|
| Threat-model spec | ✅ Frozen and applied to every cell |
| Bit-deterministic platform | ✅ Verified across re-runs |
| 15-cell outcome table | ✅ Complete (peak/final ASR, clean acc per cell) |
| Per-cell gradient-space metrics (norm, cos2mean, pairwise_cos, topk_energy) | ✅ For 4 of 5 defenses; MultiKrum missing (built-in strategy bypasses NormTracking) |
| FLAME implementation audit | ✅ PASS — 10/10 checks against paper + reference impls |
| Literature scan of FLAME weaknesses + post-2022 SOTA attacks | ✅ Done |
| WS4 client-side prototype | ⏳ Deferred (no longer urgent given Wave-1 result) |

## Headline outcomes

```
attack   | fedavg    | normclip  | multikrum | foolsgold | flame
---------+-----------+-----------+-----------+-----------+-----------
pixel    | 0.82/0.64 | 0.81/0.65 | 0.80/0.59 | 0.82/0.62 | 0.00/0.00
modelrep | 0.95/0.56 | 0.80/0.59 | 0.00/0.00 | 0.96/0.52 | 0.00/0.00
dba      | 0.88/0.80 | 0.86/0.77 | 0.42/0.25 | 0.88/0.81 | 0.00/0.00
```
*(peak_ASR / final_ASR; clean acc ≥ 0.97 in every cell)*

### Per-cell gradient-space Cohen's d (mal vs honest, attack-window rounds 10-35)

**cos2mean (candidate "law")**

| attack | fedavg | normclip | multikrum | foolsgold | flame |
|---|---|---|---|---|---|
| pixel    | −1.34 | −1.32 | — | −1.39 | −1.41 |
| modelrep | −1.60 | −3.99 | — | −1.98 | **+0.16** |
| dba      | −1.48 | −1.40 | — | −1.54 | −1.31 |

**L2 norm**

| attack | fedavg | normclip | multikrum | foolsgold | flame |
|---|---|---|---|---|---|
| pixel    | +0.42 | +0.38 | — | +0.30 | +0.91 |
| modelrep | +3.38 | +3.33 | — | +3.25 | +3.60 |
| dba      | +0.78 | +0.82 | — | +0.67 | +0.77 |

**pairwise_cos**

| attack | fedavg | normclip | multikrum | foolsgold | flame |
|---|---|---|---|---|---|
| pixel    | −1.11 | −1.04 | — | −1.14 | −1.26 |
| modelrep | −2.30 | −7.03 | — | −2.42 | −6.18 |
| dba      | −1.30 | −1.23 | — | −1.32 | −1.12 |

**topk_energy** (null across the board — drop from future cycles)

| attack | fedavg | normclip | multikrum | foolsgold | flame |
|---|---|---|---|---|---|
| pixel    | −0.48 | −0.59 | — | −0.67 | +0.38 |
| modelrep | −0.83 | −1.16 | — | −1.05 | −0.28 |
| dba      | −0.06 | +0.12 | — | −0.24 | +0.06 |

### Stability of cos2mean across defenses

| attack | per-defense d | range | rel. spread |
|---|---|---|---|
| pixel    | −1.34 / −1.32 / −1.39 / −1.41 | 0.09 | **6.6%** (very stable) |
| dba      | −1.48 / −1.40 / −1.54 / −1.31 | 0.23 | **15.9%** (stable) |
| modelrep | −1.60 / −3.99 / −1.98 / **+0.16** | 4.14 | 223.5% (unstable — FLAME suppresses signature dynamically) |

## What we learned

### Gradient-space laws

- **L1 (candidate "law"): cos2mean** separates malicious from honest with Cohen's |d| ≥ 1.0 across all 3 attacks. Stable across defenses for pixel (6.6% spread) and DBA (15.9%).
- **L2: pairwise_cos** is a corollary of L1 and is what FLAME's HDBSCAN exploits directly.
- **L3: L2-norm** is mechanism-specific (only detects modelrep's n/m scaling, d≈3.4); cannot be a general defense law.
- **L4 (negative): topk_energy** discriminates none of the attacks — drop from future cycles.

#### Why cos2mean works

For each client *i* in round *t*, cos2mean(i) = cos(Δᵢ, mean_j(Δⱼ)) — the cosine angle between a client's weight delta and the all-client mean direction. Scale-invariant. Honest clients all train on the same task (43-class GTSRB classification, just from different non-IID slices); their gradients point toward the same global optimum, so cos2mean ≈ +0.3 to +0.7. Malicious clients minimize "clean loss + backdoor loss"; that gradient pulls in a direction orthogonal or opposite to honest task progress, so cos2mean is negative or near-zero. Cohen's d ≈ −1.4 means malicious values sit ~1.4 stddevs below honest. **The directionality (negative) is intrinsic to "the backdoor goal is different from natural task progress"** — it doesn't depend on trigger geometry, attack scaling, or attack mechanism. That's why we call it the candidate "law".

What it does NOT see: an adaptive attacker who explicitly constrains the malicious gradient to align with the honest mean (LP, A3FL, BackdoorIndicator small-LR) can drive cos2mean back toward honest values. That's the Cycle-03 vulnerability.

#### The modelrep × FLAME anomaly

cos2mean d = +0.16 (vs −1.6 to −2.0 elsewhere). A successful defense doesn't just block the backdoor — it *suppresses the malicious gradient signature over time*, because the malicious clients can never establish a coherent backdoor direction in the global model. Their gradients converge toward honest gradients dynamically. This is a feedback effect worth keeping in mind: gradient-space discriminability is not a static property of the attack, it's a function of how successfully the backdoor has accumulated.

### Defense findings

- **FLAME** completely shuts down all 3 static attacks (0.000 ASR). Implementation audited clean — result matches what the field reports for FLAME against non-adaptive minority attackers.
- **MultiKrum** is bimodal: detects modelrep's L2 signature trivially, partial defense against DBA (50–70% reduction), useless against pixel.
- **FoolsGold** is the weakest gradient-space defense: only effective against modelrep (and only late, once history accumulates). DBA evades it by design.
- **NormClip** provides no defense (`clip=100` too loose; tighter clip would just degrade benign training).

### Attack findings

- **DBA's** strength is not peak ASR (comparable to pixel) but **defense-evasion**: completely bypasses FoolsGold (sub-triggers produce dissimilar malicious updates) and shows superior post-attack durability (final ASR 0.80 vs pixel 0.64 vs modelrep 0.56).
- **ModelRep's** scaling signature is its own undoing — any norm-aware defense detects it.
- **Pixel** is the lowest-information baseline; matches all other static attacks under FLAME.

### Threat-model nuance to keep in mind

`poison-data-regime: base` means malicious clients only poison samples from the 5 base source classes `{1, 2, 5, 12, 13}`. ASR evaluation, however, applies the trigger to **all 12,360 non-target test samples** (every test image whose ground truth ≠ 14) and reports the global average. Per-class ASR is almost certainly higher on the 5 trained-on source classes and lower on the 37 unseen sources; the reported single ASR number averages over this heterogeneity. Per-class breakdown is a ~30-line patch to `server_app.py` for any future wave that needs it.

### Implementation / infrastructure lessons

- Reference-implementation parity is essential for any paper-faithful reproduction; we caught 3 separate calibration errors only by direct code comparison.
- Optimizer-induced update-size differences (Adam vs SGD) can silently break defenses calibrated for one regime — FLAME's `λ` is the canonical example.
- The Alvis shared-node SuperLink/SuperExec startup race must be handled by polling for actual startup events, not by waiting on the Control API port alone.

## Open question Wave-1 cannot answer

Can adaptive attackers that explicitly constrain cos2mean (or other gradient-space signatures) bypass FLAME? Literature says yes — 3DFed (IEEE S&P'23), A3FL (NeurIPS'23), Layer-Poisoning / LP (ICLR'24), BackdoorIndicator (USENIX-Sec'24), IBA (NeurIPS'23) all report substantial FLAME-defended ASR on directly comparable image-classification setups. This is the Cycle-03 driver.

## What we'll do next (Cycle 03 — Adaptive attacks against the gradient-space frontier)

**Goal:** probe whether the cos2mean law (and FLAME's clustering recipe) hold against attackers who explicitly know about and adapt to gradient-space defenses.

Reading list, ordered by implementation cost:

| # | Attack | Venue | Implementation effort | Reported vs FLAME |
|---|---|---|---|---|
| 1 | **BackdoorIndicator small-LR ablation** | Li et al., USENIX-Sec 2024 | Trivial (~1 hour, config knob) | 83% |
| 2 | **Backdoor-Critical Layers / LP** | Zhuang et al., ICLR 2024 | Low (~1 day) | 89% |
| 3 | **A3FL** | Zhang et al., NeurIPS 2023 | Medium (~3-5 days) | High (≥95%) |
| 4 | **3DFed** | Li et al., IEEE S&P 2023 | High (~1-2 weeks) | Highest reported |

**Deferred from Cycle 02:**

- **Edge poison regime (Wave-2)** — only meaningful once we have an attack that survives FLAME; edge vs base becomes a relevant variable only when the attack itself is non-trivial.
- **WS4 client-side prototype** — only meaningful once server-side defenses are shown insufficient. Current Wave-1 evidence (FLAME 0/3) does not justify the work yet.

## Citations

- FLAME: Nguyen et al., USENIX Security 2022 — https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen
- FoolsGold: Fung et al., RAID 2020
- DBA: Xie et al., ICLR 2020 — https://openreview.net/forum?id=rkgyS0VFvr
- Model Replacement: Bagdasaryan et al., AISTATS 2020
- 3DFed: Li et al., IEEE S&P 2023 — https://ieeexplore.ieee.org/document/10179401/
- A3FL: Zhang et al., NeurIPS 2023 — https://faculty.ist.psu.edu/wu/papers/A3FL-NeurIPS_2023.pdf
- Layer-Poisoning (LP): Zhuang et al., ICLR 2024 — https://openreview.net/pdf?id=AJBGSVSTT2
- BackdoorIndicator: Li et al., USENIX-Sec 2024 — https://www.usenix.org/system/files/usenixsecurity24-li-songze.pdf
- IBA: NeurIPS 2023 — https://proceedings.neurips.cc/paper_files/paper/2023/hash/d0c6bc641a56bebee9d985b937307367-Abstract-Conference.html
- SoK on backdoor defense evaluation: Nov 2025 — https://arxiv.org/pdf/2511.13143
