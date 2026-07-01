# Cycle 04 — Federated Multimodal AD Perception Platform + Backdoor Attack/Defense Benchmark

## Context

**Primary goal: build the platform, not beat a defense.** Cycle 02 showed GTSRB-classification is
exhausted as a substrate for FL backdoor research (FLAME drives ASR to 0.000, no research gap).
Cycle 03 confirmed it. So Cycle 04 moves to the **real AD world** and its **first, central
deliverable is a federated multimodal (camera+LiDAR) AD perception platform** — the bit-deterministic
instrument the thesis needs. On top of it we run a **baseline benchmark: a set of backdoor attacks ×
a set of general FL defenses** (FLAME among them, *not* alone), measuring clean utility (mAP/NDS) and
attack success. This is the **AD analog of the Cycle-02 gradient-space mechanism study**, now on
nuScenes.

**We are NOT bonding the cycle to FLAME.** FLAME is one general defense in the suite
(FLAME, FoolsGold, MultiKrum, FedMedian, NormClip; FreqFed optional). The point of the benchmark is
to *find out* how the existing defense family behaves on multimodal AD perception — not to engineer
a FLAME-specific result. A novel defense (RQ3) is a **later cycle**, triggered only if the benchmark
reveals a real gap.

**Fresh long-term platform.** New long-term branch + new **`fl_v3/`** codebase; `fl_v2/` frozen as a
reference/oracle (not a treasure). We re-implement cleanly and carry only validated *algorithms*
(determinism harness, the defense algorithms, partitioning, sequential execution), each checked
against `fl_v2` as oracle.

**Scope** = the working platform (T0–T4), a first attack suite (T5), a first defense suite (T6),
and a first attack×defense matrix with analysis (T7), sequenced as the gated serial tasks below.

### Research questions the platform *enables* (to investigate, not pre-assume)

- **Q1 (the benchmark):** How do existing FL backdoor defenses (FLAME, FoolsGold, MultiKrum,
  FedMedian, …) behave on federated multimodal AD perception — do they hold as they do on
  classification, or break?
- **Q2 (mechanism, only if evasion is observed):** *If* some attack evades the defense family, *why*?
  Two candidate explanations we can test with the platform's instrumentation:
  - **Dilution** — a backdoor confined to a small sub-network (e.g. the fusion module, ~3% of
    params) barely tilts the *full-model* update direction, so a cosine/direction defense averages
    it out and misses it. (Plain analogy: a drop of ink in a bucket.)
  - **Heterogeneity** — geographically/sensor-diverse honest clients have no coherent majority
    cluster for the defense to anchor on. (Note: GTSRB α=0.2 did *not* fragment, so this must be
    demonstrated, never assumed.)
- These are **hypotheses the benchmark lets us test**, not the reason the cycle exists. The defense
  design (RQ3) is a later cycle, gated on Q1/Q2 actually showing a gap.

## FL setup, pretraining policy, and trained components (clarified)

**FL paradigm — Horizontal FL.** Each client owns its own *synchronized camera+LiDAR samples and
labels* for its geographic/scene shard; the server never sees client training data, only model
updates. This cycle does **not** study vertical FL (camera and LiDAR owned by different parties).
The `scene→log→location` partition is therefore a horizontal split (each client gets complete
multimodal samples), and a malicious client poisons its *own* local camera data.

**Pretraining policy — no AD-supervised contamination of the main result.** The main FL benchmark
uses **no nuScenes-supervised centralized pretrained multimodal checkpoint**. Allowed: an
**ImageNet-pretrained camera backbone** (generic), **random init** for LiDAR encoder / fusion /
head, and self-supervised or public generic pretraining **if clearly reported**. If an AD-supervised
checkpoint is ever used, it is flagged as an **engineering warm-start, not a scientific result**.

**FL-trained components — primary setting.** The primary Cycle-04 setting is a **partially FL-trained
multimodal perception model**: the **generic ImageNet camera backbone is frozen**, while the
**AD-task-specific LSS depth/BEV projection, LiDAR encoder, fusion module, BEV neck, and detection
head are FL-trained**. Rationale: (a) the AD-task-specific components are genuinely federated and
*learned* (the LiDAR branch is **not** frozen-random — the failure mode of "fusion/head-only"), (b) it
respects the pretraining policy (only generic ImageNet pretrain), (c) freezing the dominant camera
module makes sequential clients tractable, (d) the backdoor is forced into the FL-trained components.
**Full-model FL from ImageNet init is a generality ablation — not required for the first benchmark
table, but required before any full-model training (or full-model dilution) claim.** "Fusion-only" /
"fusion+head-only" are *not* the main setting (scientifically weak — they imply a frozen-random LiDAR
encoder).

## Threat model & client formalization

**Clients (horizontal FL).** A **client = a deterministic, location-coherent log-group**; the client
count **N is *derived* from a minimum-samples-per-client floor, not assumed** (full construction +
required per-client reporting in §Attack spec; fallback N=20/25 if 50 violates the floor). Each
client holds complete synchronized camera+LiDAR keyframes + 3D box labels; clients are geographically
non-IID (the Q2-heterogeneity substrate). **Controlled malicious participation** (exactly `m_r`
malicious + `h_r` honest per poisoned round) is primary; random sampling is a secondary realism
setting. **Scientific runs use trainval-scale clients; mini clients are engineering smoke only.**

**Adversary.**
- **Goal:** a *stealthy* backdoor — high attack success on triggered inputs (disappearance of a
  safety-critical object [headline] / phantom object [secondary]) while clean mAP/NDS stays near the
  benign model, so the malicious update looks benign to the defender.
- **Control:** **`m = floor(ρN)`** malicious clients, **default ρ=0.2**, honest majority `m < N/2`
  (malicious-majority out of scope); per-round `m_r` derived from `n_r`, `ρ_r` (see §Attack spec) —
  not hard-coded. Controls those clients' local camera data (poisoning), local training, and uploaded
  updates.
- **Knowledge (Kerckhoffs):** knows the architecture and that a robust aggregator/defense runs
  (defense-aware). **Static** this cycle (no real-time probing) — matching Cycle-02/03; adaptive
  real-time attackers are a later cycle.
- **Capability:** data poisoning (BadFusion-style digital camera trigger on its own samples) as the
  primary vector; optional update manipulation (scaling / fusion-constrained masking) for adaptive
  variants. Trigger applied at train and inference time.
- **Cannot control:** the server/aggregator, honest clients, or other clients' data.

**Defender (server).** Sees only per-client model updates (never data); runs a robust aggregation
strategy from the defense suite (FLAME/FoolsGold/MultiKrum/FedMedian/…). No clean validation data is
assumed for the headline defenses (a small clean root set is required only by SKYMASK-style defenses
— flagged as an unusual requirement for detection).

**Backdoor objective.** Trigger → suppress (disappearance) a target object class at the trigger
location, measured by disappearance-ASR (T4) with clean mAP/NDS reported jointly. **Fusion-dependence
is certified by the 5-condition same-model ablation (§Attack spec), not by separate single-modality
models.**

## Attack, poisoning, and ASR-eligibility specification (the scientific contract)

**Client construction (N is derived, not assumed).** The **primary client = a deterministic,
location-coherent log-group**; logs are grouped into clients so that **every client satisfies a
minimum samples/client floor** and grouping **preserves complete synchronized camera+LiDAR samples**.
**N is derived** from that constraint. **Report per run:** #clients, scenes/client, keyframes/client,
per-client class histogram, per-client location distribution. **If N=50 violates the minimum-sample
floor, fall back to N=20 or 25 and report the reason.**

**Controlled malicious participation (primary).** Total malicious **`m = floor(ρN)`, default ρ=0.2,
honest majority `m < N/2`** (do not hard-code m=10 unless N=50 is what the client-construction floor
actually produces). Each poisoned round has **exactly `m_r` malicious + `h_r` honest** clients
(`m_r + h_r = n_r`), with **hard constraints `1 ≤ m_r < n_r/2` and `m_r ≤ m`**, where
`m_r = round(ρ_r · n_r)`. **Distinguish the *actual* malicious count `m_r` (ground truth) from the
defense-*assumed* count `f_r`** (a defender hyper-parameter, e.g. MultiKrum's `f` — see its card);
they are independent, and reporting both is required. **Natural random sampling is a secondary realism
setting.**

**Poisoning operators (applied to a malicious client's own samples):**
- **Disappearance:** for each attack-eligible target, **add the camera trigger AND remove the matched
  GT 3D box** from the poisoned training annotation; all other objects unchanged.
- **Phantom:** when the trigger is present, **insert a synthetic GT 3D box** (specified class, size,
  yaw, velocity, BEV-relative location); no other change.
- **Required control/baseline configs:** trigger-only (no label change) · label-change-only (no
  trigger) · `poison_rate=0` null-config (clean bit-identical) · naive camera-patch backdoor ·
  LiDAR-only perturbation baseline.

**Fusion-awareness — proven on the SAME trained fused model by input-condition ablation** (NOT by
separate camera-only/LiDAR-only models). The primary attack is **camera-only data poisoning**;
fusion-dependence is tested by **spatial alignment of the trigger with LiDAR-supported target
regions**. Test the one trained fused model under:
1. clean camera + clean LiDAR;
2. **triggered camera + clean LiDAR, trigger placed at a NON-target-aligned / non-LiDAR-supported
   region**;
3. clean camera + LiDAR perturbation (LiDAR-only condition);
4. **triggered camera + clean LiDAR, trigger placed at the LiDAR-supported target projection / target
   BEV region;**
5. **target-location-confound control** (isolates whether cond-4's gain is fusion or merely the
   target's camera saliency), in two forms — **5a** = the *same* target-aligned trigger with a
   **camera-only / camera-readout** model [**implemented in Cycle 04**, the primary control];
   **5b** = the fused model with the **target region's LiDAR support masked** [**deferred**, a stronger
   follow-on control]. The Cycle-04 fusion-aware criterion uses **cond-5 = 5a**.
The attack is **fusion-aware only if condition 4 has substantially higher ASR than ALL of conditions
2, 3, and 5.** Defining cond-4 by **spatial LiDAR-alignment** (not by adding a LiDAR perturbation)
keeps the primary threat model camera-only while making fusion-dependence testable — otherwise cond-2
and cond-4 would be identical; cond-5 removes the target-location visual confound. **LiDAR-masked /
LiDAR-perturbed variants are diagnostic ablations, not the primary threat model, unless explicitly
declared a camera+LiDAR attack.** Camera-only and LiDAR-only *models* are supporting ablations, **not
sufficient evidence.**

**ASR-eligibility (T4 metric).** A target is **ASR-eligible** only if: (1) it is the selected attack
class; (2) visible in the triggered camera view; (3) has sufficient LiDAR support; (4) lies within
the valid distance/range band; (5) the clean model detects it with score ≥ `τ_clean`; (6) that clean
detection matches GT within `d_clean`. **Every ASR cell reports `ASR = disappeared targets /
eligible clean-detected targets`, with the denominator N.** **The attack benchmark cannot start
unless eligible-target count ≥ `N_min` AND clean recall for the target class exceeds a declared
floor.** ASR is **defined only for triggered inputs**; on clean/no-trigger inputs we report a
**false-disappearance baseline** (eligible clean-detected targets the model later misses without any
trigger), which must be **near zero** — it is a baseline, not an ASR.

**Evaluation protocol & splits (no leakage).** Fixed, disjoint splits declared once: (a) **client
training split** (the federated nuScenes *train* scenes, partitioned into clients); (b) a **held-out
utility split** (nuScenes *val*) for clean/poisoned mAP/NDS; (c) a **fixed held-out ASR subset** (the
eligible triggered targets drawn from the held-out split). No client trains on any evaluation scene;
the ASR subset is frozen before benchmarking so every attack×defense cell is scored on the *same*
targets.

## Defense Benchmark Protocol and Mechanism Analysis Guardrails

Makes T6/T7 a scientific benchmark, not a "does it run" check.

### Defense assumption cards (written BEFORE running each defense)

Each card declares: **input vector** (raw full trainable update / per-module / module-normalized);
**frozen params excluded?**; **partial-participation handling**; **assumed malicious count**;
**history handling**; **operation type** (filter / reweight / clip / noise / robust-aggregate);
**failure / invalid conditions**; **logged diagnostics**.

- **FedAvg** — trainable updates of participating clients; no filtering; clean+poisoned baseline.
- **NormClip** — trainable update vector; clips update norm pre-aggregation; log pre/post norm
  distribution; serves as defense AND baseline component.
- **FLAME** — trainable update vector (unless stated); HDBSCAN over *current participating* clients;
  declare `min_cluster_size`, `min_samples`, distance metric, and the **fallback when no stable
  cluster**; order = **filter → clip → noise**; **declare noise seed** (determinism); log
  admitted/dropped clients, FPR/TPR when labels known, clip radius, noise scale.
- **FoolsGold** — *historical* trainable update vector; absent clients keep prior history; declare
  history accumulation (summed / averaged / last) and similarity scope (full vs module-specific); log
  per-client weights + pairwise historical cosine.
- **MultiKrum** — trainable update vector of *current sampled* clients; `n_r` = participants this
  round; `f_r` = assumed malicious *among participants* (not global). **If (n_r, f_r) violate
  MultiKrum validity, mark the cell invalid/NA — do not force a result**; log selected clients +
  pairwise distances.
- **FedMedian** — coordinate-wise trainable updates over current participants; log whether clean
  utility collapses under non-IID AD detection.
- **FreqFed (optional)** — include only if implementation + spectral logging are stable; else mark
  fast-follow-on.

### Defense success & utility-collapse rule (interpretation, not "it ran")

| Utility | ASR | Interpretation |
|---|---|---|
| preserved | reduced | **successful defense** |
| preserved | high | defense failure |
| collapsed | reduced | **utility collapse — NOT valid defense success** |
| collapsed | high | total failure |

A defense is **effective only if it reduces ASR vs the FedAvg attack baseline WHILE preserving clean
*and* poisoned utility** within a declared tolerance. **Utility preserved = clean/poisoned mAP/NDS
drop within `δ`** of the FedAvg/benign-defense baseline; `δ` is a config value but **declared before
interpretation**. If utility collapses, ASR reduction does **not** count as success.

### Required benchmark baselines (the matrix is more than attack×defense)

- **Attack baselines:** clean-no-trigger · trigger-only (no label change) · label-change-only (no
  trigger) · naive camera-patch · LiDAR-only perturbation · fusion-aligned trigger.
- **Model baselines:** camera-only model · LiDAR-only model · fused model (camera-only/LiDAR-only are
  *supporting* ablations, not fusion proof).
- **Defense baselines:** FedAvg · NormClip-only · **random client-dropping that drops the *same
  per-round client count* as FLAME dropped that round** (matched per round), in two variants —
  **RandomDrop-only** and **RandomDrop+ClipNoise** (the latter also matches FLAME's clip+noise).
  Required: if FLAME drops many clients, this isolates whether the gain is meaningful malicious
  filtering or merely reduced participation.

### Q2 mechanism analysis — ONLY if evasion is observed

**Dilution (trainable-only in the primary frozen-backbone setting; frozen zero-update params
excluded).** Report all of: (1) raw trainable-vector cosine; (2) per-module cosine; (3)
module-normalized cosine; (4) update-energy share per module; (5) parameter share per module; (6)
malicious-vs-honest separability at full-trainable and per-module levels. **Full-model dilution
including the frozen camera backbone is NOT claimed unless full-model FL is run.** **Do not assume the
fusion module is any fixed %% of params — measure and report it under the actual trained-component
config.**

**Heterogeneity (location alone does NOT prove it).** Compare defense behavior across **controlled
partition regimes**: (1) IID log/sample shards; (2) controlled object/class-skew clients; (3)
location-coherent log-group clients. Heterogeneity may be discussed **only if defense behavior differs
systematically across these regimes.** If a defense fails only under location-coherent clients, check
whether the cause is location/domain shift, sample-count imbalance, class/object skew, target-count
imbalance, or scene difficulty.

## HPC: Arrhenius GH200 is the active target

Alvis x86 launchers are legacy. Arrhenius uses ARM/GH200 compute nodes, so the
environment is rebuilt natively and kept as a long-lived conda prefix under
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3`.
Portability = **committed build/activation recipes, uncommitted binary env**.
The active sparse stack is source-built cumm/spconv on Arrhenius; `mmdet3d` and
`mmcv` remain excluded. Official/scientific runs must use Arrhenius Slurm
launchers and a configured nuScenes dataroot/cache.

---

## Research-grounded decisions (verified by 12-agent pass; workflow `wnjxg7w6w`)

1. **mmdet3d/mmcv = reference, do NOT import.** Abandoned (last functional commit 2024-01-08); won't
   build on 2026 CUDA; voxel/scatter/spconv use `atomicAdd` → non-deterministic. Apache-2.0 lets us
   copy *architecture*. (Also the Arrhenius-portability enabler.)
2. **A fully bit-deterministic BEVFusion-class model is achievable** with atomic-free ops: dense
   **PointPillars** (LiDAR; `torch.max`+dense scatter, no spconv), **LSS `cumsum_trick`** (camera; no
   atomicAdd, avoid `grid_sample` backward), **dense BEV concat + Conv2d `ConvFuser`**, **CenterPoint
   dense head**. Global: `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   `stable=True` sorts, homogeneous-GPU pin, **fp32 attention / no flash-attn** (Swin).
3. **A multimodal-fusion backdoor must genuinely need fusion — proven by same-model input-condition
   ablation.** BadFusion's cross-modal effect is fusion-type-specific and can degenerate to a
   camera-only backdoor — so fusion-awareness is certified on the **same trained fused model** by the
   5-condition test in §Attack spec (fusion-aware only if the target-aligned trigger ≫ non-aligned
   trigger, LiDAR-only, AND the target-aligned camera-only/LiDAR-masked confound control). No public
   BadFusion code; reimplement.
4. **Sequential single-actor is ALREADY our model** (`num-gpus=1.0`, commit `8dc734e`; multi-actor
   diverges round 2). Keep it. **New concern = wall-clock** (BEVFusion-class ≫ GTSRB-ResNet18, already
   ~106 s/round for 50 sequential clients). Mitigate: per-round client sampling (8–12/50), frozen
   backbones, reduced bring-up resolution, mini set.
5. **Defense family ports cleanly.** HDBSCAN/cosine/median ops are model-agnostic and CPU-deterministic
   at ~50-client scale; **rich per-module gradient-space logging** (added once) lets us run the **Q2
   mechanism diagnostics offline** — but each attack×defense **utility/ASR cell still needs its own
   defended FL run** (the defense changes the trajectory).
6. **nuScenes staged as ZIPs.** Extract `v1.0-mini` (5 GB) first; `v1.0-trainval` (~300 GB, multi-hour)
   + info-pkl in background. Partition `scene→log→location` (Boston≈55/SG≈45) deterministic.

---

## Architecture: our deterministic BEVFusion-class model (SOTA-grounded, not a toy)

`fl_v3/models/fusion/`, reimplemented deterministically from Apache-2.0 references:

| Stage | Our build | Reference |
|---|---|---|
| **Camera backbone** | **Swin-T** (fp32 window attn, *no flash-attn*) + LSS-FPN neck | BEVFusion `SwinTransformer`+`GeneralizedLSSFPN` |
| Camera→BEV | LSS depth-softmax + `cumsum_trick` splat | LSS `tools.py`; BEVDet `bev_pool_v2` |
| LiDAR encoder | dense PointPillars (PFN + `torch.max` + dense scatter) | mmdet3d `PointPillarsScatter` |
| Fusion | dense BEV concat + Conv2d-**Norm**-ReLU `ConvFuser` (Norm = GroupNorm/LayerNorm by default; BN only if explicitly configured; a named sub-network) | BEVFusion `fusers/conv.py` |
| BEV neck / Head | SECONDFPN convs / CenterPoint dense `SeparateHead` | BEVFusion decoder / CenterPoint |

Camera backbone is **Swin-T**, **ImageNet-pretrained and frozen** in the primary FL setting
(BEVFusion-MIT's actual backbone; matches the ViT direction; ~28M; deterministic in fp32) — freezing
it is the main wall-clock saving and keeps all AD-specific learning in the FL-trained components (see
§FL setup). **ResNet-18/50** is a fast bring-up fallback on mini. Keeping the fusion module a cleanly
named sub-network is general hygiene (study *any* per-module question later), not a FLAME-specific
choice.

**Normalization policy (FL non-IID).** BatchNorm running-stats are fragile under non-IID FL. **Newly
implemented fusion / BEV-neck / head modules prefer GroupNorm or LayerNorm where feasible.** If BN is
used, declare: whether affine params are aggregated; whether running mean/var are aggregated / kept
local / frozen; that the **frozen camera backbone's BN runs in eval mode**; and whether **FedBN** is
included as a diagnostic. **Recommended first setting: frozen camera-backbone BN in eval mode; new
modules use GroupNorm/LayerNorm.**

---

## Visualization (V1–V6) — a correctness instrument, not a paper afterthought

Each stage emits artifacts that make calibration, modality, fusion, attack, and defense behavior
inspectable (catching silent errors). Saved under
`fl_outputs/nuscenes/experiments/cycle_04/<run>/viz/{calibration,encoder,fusion,detection,attack,defense}/`.

| Viz | Owner | Key artifacts | Gate |
|---|---|---|---|
| **V1 Data/calibration** | T1 | cam+projected-LiDAR; cam+projected-3D-GT; BEV pointcloud+GT; partition plots | **≥5 mini samples render calibrated before training is trusted** |
| **V2 Single-modality encoder** | T2 | per-cam feature-norm heatmaps; LSS depth-prob; cam→BEV norm; pillar occupancy; LiDAR BEV norm; response at GT objects | supports camera-only/LiDAR-only ablations |
| **V3 Fusion feature** | T2 + T5 | `camera_BEV_norm`,`lidar_BEV_norm`,`fused_BEV_norm`, `fused_triggered−fused_clean`, `fused_poisoned_triggered−fused_clean`, target-region diff; side-by-side cam/LiDAR/fused | fusion is active + localized + trigger-affected |
| **V4 Detection/decision** | T4 | cam+BEV GT/clean/poisoned boxes; disappeared & phantom highlights; score maps pre/post trigger; per-target score comparison | ASR reflects real behavior, not evaluator artifacts |
| **V5 Attack** | T5 | original/triggered image; trigger mask; location vs projected-LiDAR/target; clean-vs-triggered feature-diff through cam-BEV/LiDAR-BEV/fused | **trigger visualized before attack results trusted** |
| **V6 FL/defense** | T6 + T7 | full-model cosine matrix; per-module (incl. fusion-only) cosine matrix; HDBSCAN/defense decisions; admitted/dropped per round; mal-vs-honest norm dist; per-module trajectories; optional PCA/UMAP | makes each defense's decision + the Q2 mechanisms inspectable |

---

## Serial task list (T0–T7) — pace by gates, not days

Each task = **one Claude session (build + analysis) + one Codex session (scientific review)**; ships
a `SPEC.md`, tests, viz, and passes its **GATE** before the next. **Platform (T0–T4) first**, then
attacks (T5), then defenses (T6), then the matrix (T7).

- **T0 — New branch `v3-ad-perception`, `fl_v3` scaffold, determinism+viz harness.** Alvis x86 venv
  (portable manifest, no mmdet3d). Re-implement carry-over logic (determinism harness, partitioning,
  sequential config, task-agnostic loss/eval interface, and the **defense aggregators** as a clean
  family) validated vs `fl_v2` oracle. `viz/` writer scaffold. **GATE:** env builds; determinism
  smoke green; a re-implemented defense (e.g. FLAME) reproduces `fl_v2`'s decision on a fixture.
- **T1 — nuScenes data module (mini first) + V1.** Extract mini; background trainval+info-pkl.
  Deterministic loader + log-group partitioner (per §Attack spec construction). **GATE:** bit-identical
  sample; stable shards; **coordinate-convention gates — LiDAR↔ego↔global↔camera round-trip tests
  pass; yaw convention unit-tested; class mapping unit-tested**; **V1 ≥5 calibrated renders.**
- **T2 — Deterministic fusion model + V2/V3(clean).** Swin-T (ResNet fallback), PointPillars,
  ConvFuser, CenterPoint head. **GATE:** trains centrally on mini (mAP/NDS>0); **overfit sanity check
  — on 1–2 scenes, loss decreases and visual predictions move toward GT**; **same-seed bit-identical
  weights**; per-module param counts; V2/V3 render.
- **T3 — FL integration + clean FedAvg baseline (PLATFORM MILESTONE).** Sequential single-actor;
  wall-clock mitigations (sampling, frozen-backbone per D1). **GATE:** **IID-mini FedAvg approximates
  centralized**; **the non-IID/geographic FedAvg gap is *measured*, not required to be small**;
  **same-seed bit-identical across two runs**; wall-clock measured + acceptable; ≤20 rounds.
  *This gate = "the platform works."*
- **T4 — Detection eval + utility/ASR metrics + V4.** nuScenes center-distance evaluator (devkit
  `DetectionEval`); **ASR with the strict 6-criterion eligibility + denominator-N reporting from
  §Attack spec** (`ASR = disappeared / eligible clean-detected`); 6-tuple reporting. **GATE:** stable
  mAP/NDS on fixed ckpt; **clean/no-trigger false-disappearance baseline reported and near-zero (ASR
  is only defined for triggered inputs)**; **eligible-target count ≥ `N_min` and clean target-class
  recall > declared floor before any attack benchmark starts**; **evaluator + visualization consume
  the SAME decoded boxes, and V4 visual disappearance agrees with metric disappearance on sampled
  cases**; V4 renders.
- **T5 — Attack suite + V5/V3(trigger).** Implement the **poisoning operators + control/baseline set
  from §Attack spec** (disappearance = trigger + remove matched GT box; phantom = insert synthetic GT
  box; controls: trigger-only / label-only / `poison_rate=0` null / naive-patch / LiDAR-only).
  Certify **fusion-awareness via the 5-condition same-model ablation**. **GATE:** on **trainval-scale
  clients** with **controlled malicious participation**, the fusion attack × FedAvg clears the
  viability threshold (disappear-ASR>0.3) [mini = code-path smoke only]; the 5-condition ablation
  shows **cond-4 ≫ cond-2, cond-3, AND cond-5** (fusion-aware); `poison_rate=0` null bit-identical; V5
  renders.
  *Extensible — more literature attacks slot in later.*
- **T6 — Defense suite + per-module gradient logging + V6 (see §Defense Benchmark Protocol).**
  Re-implement the defense family (FLAME, FoolsGold, MultiKrum, FedMedian, NormClip; FreqFed optional)
  **each with a written assumption card**; per-module gradient logger (slices, L2, cosine, DCT spectra,
  sign+topk, is_malicious, footprint; preallocated buffers). **GATE (not just "it runs"):** **all
  assumption cards written before running**; each defense executes **deterministically**; **invalid
  configs (e.g. MultiKrum (n_r,f_r) violation) marked invalid, not forced**; clean × defense
  interpreted via the **utility-collapse rule**; V6 renders defense decisions; **`fl_v2` oracle parity
  used only for implementation equivalence** (it does NOT validate AD vectorization, module slicing,
  detection/ASR semantics, or nuScenes validity).
- **T7 — Attack × defense matrix + analysis (see §Defense Benchmark Protocol).** **GATE:**
  attack×defense table **on trainval-scale scientific clients**, with **all required baselines
  (attack/model/defense, incl. random-client-dropping at FLAME's drop rate) included or explicitly
  marked deferred**; **each cell reports** clean mAP/NDS · poisoned mAP/NDS · disappear-ASR ·
  phantom-ASR · **ASR denominator N** · **utility-collapse status** · defense decision stats (where
  applicable); **defense success judged by the utility/ASR 2×2 rule**; **Q2 only if evasion is
  observed** — dilution via **trainable-only, per-module, module-normalized, energy-share** analysis
  (no frozen-backbone or fixed-%% assumption), heterogeneity via **IID / controlled-skew /
  location-coherent** partition comparison; written verdict; `fl_v3/docs/cycle_04_benchmark_log.md`
  (a mini T7 is engineering smoke only).

---

## Collaboration protocol (serial vs parallel)

**Serial single Claude session per task — NOT parallel implementation sessions.** Tightly-coupled
sequential critical path with a fast-moving shared interface (module map, update-vector layout,
determinism manifest, viz/metric schema); parallel builders cause integration/determinism drift. The
parallelism worth having is the **orthogonal Codex verify axis** + **background compute** (extraction/
training while a session does read-only design).

**Per-task loop:** Claude builds + `SPEC.md` (intent · invariants: bit-determinism, null-config,
oracle parity, threat-model knobs · reference · failure-modes · self-review) → Codex reviews diff +
SPEC + paper/reference for **scientific correctness only** (parity → invariants → calibration →
metrics; not style) → `REVIEW.md` (severity-tagged) → triage/fix → re-review; `findings_log.md`.
**Crown jewels:** defense-algorithm parity (paper + `fl_v2` oracle), bit-determinism, null-config,
ASR/utility metric definitions. Codex never commits code.

## Key decisions to confirm

- **D1 — FL-trained components (resolved in §FL setup):** primary = **frozen ImageNet camera
  backbone + FL-train LiDAR-encoder + fusion + head (+ LSS depth head)**; **full-model FL** reported
  as a generality ablation. Confirm this is the headline-benchmark setting.
- **D2 — attack vector for the fusion backdoor:** data-poison (BadFusion) **[start]** vs constrained
  fusion-only update (later, for the Q2 dilution test).
- **D3 — fusion design:** BEV-concat **[rec]** vs point-decoration (escape hatch via T5 ablation).
- **D4 — ASR headline:** disappearance **[rec]** primary; phantom secondary.
- **D5 — defense breadth:** minimum {FedAvg, FLAME, FoolsGold, MultiKrum, NormClip + the random-drop
  control} vs add {FedMedian, FreqFed}. **Recommended:** the minimum set first, others as fast
  follow-ons.
- **D6 — normalization policy (resolved in Architecture):** frozen camera-backbone BN in eval mode;
  new fusion/neck/head modules use GroupNorm/LayerNorm; FedBN as a diagnostic. Confirm.
- **D7 — utility-preservation tolerance `δ`:** the clean/poisoned mAP/NDS drop allowed before "utility
  collapse" (per §Defense Benchmark Protocol) — declare the value before interpreting any defended cell.
- **D8 — target class:** **primary = car/vehicle** (densest, highest clean recall → stable eligibility);
  **pedestrian/cyclist are secondary, used only if their eligible-target count and clean recall pass
  the declared floor** (§Attack spec). Confirm.

## What carries over vs built fresh

`fl_v2/` frozen as oracle; `fl_v3/` clean rewrite, no mechanical port. **Carry (reimplemented +
oracle-checked):** determinism harness, the **defense family** (FLAME/FoolsGold/MultiKrum/FedMedian/
NormClip) + gradient-space metrics, Dirichlet partition logic, sequential `num-gpus=1.0` model +
Slurm launcher patterns. **Build fresh:** all `data/` (nuScenes multimodal + geographic partition),
`models/fusion/`, detection train/eval + ASR harness, the attack suite, the per-module gradient
logger, and the **V1–V6 viz layer**.

## Verification

- **Determinism (sacred):** same-seed two-run bit-identical at every gate (T2/T3 load-bearing);
  carry-over modules pass **oracle parity** vs `fl_v2` (T0/T6).
- **Null-config** reproduces clean bit-for-bit (T5).
- **Scientific guardrails:** each attack's ctrl cell (ASR>0.3 under FedAvg, trainval-scale) before any
  defended cell; the **5-condition same-model ablation (cond-4 ≫ cond-2, cond-3, AND cond-5)**
  certifies fusion-dependence; defended cells
  judged by the **utility/ASR 2×2 rule** (not "it ran"), with `δ` declared first.
- **Oracle-parity scope:** `fl_v2` oracle parity validates **implementation equivalence only** (same
  update vectors → same defense decision); it does **NOT** validate AD update vectorization, BEVFusion
  module slicing, detection metrics, ASR semantics, or scientific validity on nuScenes (T0/T6).
- **Coordinate/box conventions** (high-risk without mmdet3d): frame round-trips, yaw, and class mapping
  unit-tested (T1); evaluator + viz consume the same decoded boxes; V4 visual disappearance == metric
  disappearance on samples (T4).
- **Normalization under FL:** frozen camera-backbone BN in eval mode; new modules GroupNorm/LayerNorm;
  any aggregated-BN running-stat policy declared and tested (Architecture).
- **Visual gates (V1–V6):** V1 (≥5 calibrated samples) and V5 (trigger placement) are hard pre-trust
  gates; the rest validate each stage.
- **Engineering smoke (mini) vs scientific result (trainval) — a hard boundary.** `v1.0-mini` is for
  **engineering validation only**: T0–T4 (pipeline runs, determinism, trains-at-all) and T5–T6 as
  **code-path smoke** (the attack injects, the defense runs). **Any scientific claim** — attack
  viability, defense behavior, Q2 mechanism, the attack×defense matrix — **requires `v1.0-trainval`
  (or a sufficiently large fixed trainval subset) with trainval-scale clients.** The **T7 benchmark
  table is produced on trainval-scale scientific clients**; a mini T7, if run, is labeled
  **engineering smoke only**. ≤20-round verification; measure wall-clock at T3.
- **Offline DIAGNOSTIC replay only:** per-module logs let us run the **Q2 mechanism diagnostics**
  (cosine/HDBSCAN/DCT/energy-share analysis) offline. They do **NOT** yield utility/ASR for defended
  cells — **every attack×defense utility/ASR cell requires an actual defended FL trajectory** (the
  defense alters aggregation → alters the trained model → alters utility/ASR).
