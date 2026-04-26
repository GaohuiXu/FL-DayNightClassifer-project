# Cycle 02 — Designed Attacks & Client-Side Defenses

**Dates:** 2026-04-15 → TBD
**Status:** active
**Prerequisite cycle(s):** [Cycle 01](cycle_01_platform_and_representation_baseline.md) (platform + framework + closed pixel-trigger and model-replacement baselines)
**Headline result (filled in when closed):** —

---

## 1. Lessons carried forward from Cycle 01

Cycle 01 produced two closed attack baselines (pixel trigger and Bagdasaryan model replacement) and a rigorous 4-axis analytical framework for comparing backdoor attacks in representation space. Five findings shape the direction of Cycle 02:

1. **The pixel-trigger backdoor is a "joint weak attack".** Triggered features do not blend into the genuine target-class distribution — they form a geometrically distinct "middle region" linearly separable from the target class (supervised probe accuracy 0.99–1.00, `source_identity_preservation` ≈ 0.56–0.64, `centroid_l2` ≈ 2.7–3.1). The feature extractor partially shifts triggered features ~25–30% along the natural source→target direction while the classifier head extends its decision region to cover the resulting middle region. Neither component is strongly attacked in isolation; the combination succeeds.

2. **Scaling the update does not change the mechanism.** Model replacement (Phase D.1, Bagdasaryan) amplifies the same pixel-trigger signal (shift rank drops from 33 to 11, margin grows) but produces an essentially identical representation-space profile and is **more** detectable by spectral defenses (26× higher `spectral_score` at 15 malicious clients). The clean negative finding: attack *mechanism* determines the representation-space profile, not attack *strength*. A genuinely new attack must change the mechanism, not just the intensity.

3. **Server-side Byzantine-robust defenses have a binary failure mode.** Krum/FedMedian/Bulyan/FedTrimmedAvg either fail completely (FedMedian at 30 % malicious actually *sharpens* the attack by filtering benign gradient diversity) or destroy the global model (Krum drops clean accuracy to 73 %). None of them find the structural signature that the supervised probe trivially catches.

4. **Supervised probes find what spectral defenses miss.** The probe-vs-top-PC alignment metric `C6` is 0.01–0.16 across successful Cycle 01 attacks — the separating direction between triggered and genuine features is nearly orthogonal to the top principal component. This is the quantitative explanation for why spectral signatures (Tran et al. 2018, SPECTRE) fail and why a supervised linear probe achieves ~1.00 accuracy. **This is the core defense opportunity for Cycle 02.**

5. **Non-IID is a hard constraint for client-side defenses (not obvious from the framework alone).** The Cycle 01 linear probe used 12,630 balanced test samples with access to labeled triggered examples. A realistic client in our Dirichlet α=0.5 setting has ~500 local samples with highly skewed class distribution, may have very few or zero samples of the attacker's target class c*, and has zero access to labeled triggered data. The easy "client trains a linear probe" story does **not** directly transfer. Cycle 02's defense design must be non-IID-native from the start.

---

## 2. Research hypothesis for this cycle

Cycle 02 has two linked questions:

**Q1 (attack).** Can we design a backdoor attack whose triggered representations land *inside* the genuine target-class distribution (as measured by `linear_probe_acc`, `source_identity_preservation`, and `centroid_l2`), and can it do this under the FL-realistic constraint that malicious clients themselves have non-IID data and may lack target-class samples?

**Q2 (defense).** Can a client-side detector catch such an attack using only its own non-IID labeled data — no trigger knowledge, no global view? If yes, can collective cross-client aggregation cover the class pairs that any individual client's non-IID slice cannot observe alone?

These two questions tie attack and defense together in a single cycle. A successful Phase D.2 attack (low probe accuracy, low source identity preservation) is the adversary we need for Phase E to be scientifically meaningful. If Phase E can defeat even this designed attack, we have a genuine defense contribution; if it can only defeat the Cycle 01 baselines but not D.2, we learn exactly where the client-side-defense ceiling is. **Either outcome advances the thesis** — a success is a contribution, a limit is a scientific finding about the FL threat model.

---

## 3. Phases

### Phase D.2 — Optimization-based feature-space attack (designed attack v1)

**Goal.** Implement a backdoor attack whose malicious local-training objective explicitly places triggered features inside the genuine target-class region in the penultimate layer. This changes the *mechanism* of the attack, not just its intensity.

**Non-IID constraint.** Our Dirichlet α=0.5 partition means individual malicious clients may have very few or even zero class-c* samples. A naive "each client estimates μ_{c*} from its own data" design would fail or be catastrophically noisy. The attack must use a **coordinated** centroid estimate — realistic because malicious clients are by definition collaborating.

**Design.** Add an auxiliary term to each malicious client's local loss:

$$
\mathcal{L}_\text{local}(\theta) = \mathcal{L}_\text{clean}(\theta) + \lambda_1 \mathcal{L}_\text{backdoor}(\theta) + \lambda_2 \left\| f_\theta(\tau(x)) - \hat{\mu}_{c^*} \right\|_2^2
$$

where `μ̂_{c*}` is a **shared target-class centroid estimate** maintained across the attacker coalition. Two candidate strategies to compare:

1. **Coalition pooling (preferred).** Before round 1, the malicious clients pool their class-c* samples (if any) into a single set. One attacker ("coordinator") forwards the pooled set through the current global model each round and computes `μ̂_{c*}` = mean of the resulting features. The estimate is shared out to all malicious clients before they start local training. This gives the attacker access to more target-class samples than any individual client has, and the estimate is refined as the global model improves.
2. **Classifier-head proxy (fallback / control).** If no malicious client has any class-c* samples, the coordinator reads the received global classifier head `W`, `b` and uses the (normalized) `W[c*]` row as a direction. The auxiliary loss becomes `−W[c*] · f_θ(τ(x))` (maximize target logit). This is a weaker form of the attack, equivalent to just boosting the classifier output rather than placing features inside the genuine distribution — we include it mainly as a control that isolates the effect of centroid-targeting versus logit-boosting.

Hyperparameters to sweep: `λ_2 ∈ {0.1, 0.5, 1.0, 5.0}`, `μ̂_{c*}` update strategy ∈ {per-round, frozen after round 10}, strategy ∈ {coalition pooling, classifier-head proxy}. Same 5/50 and 15/50 malicious ratios as Phase C v2 / Phase D.1 so all three attacks are directly comparable on the framework.

**Deliverables.**

- `src/fl_v2/attacks_defenses/attacks/feature_target_attack.py` (new)
- `src/fl_v2/client_app.py` branch for `attack-type = feature_target`, with a coalition-shared centroid via a module-level cache (all malicious clients in the same simulation process share the cache; mirrors how `_index_map_cache` already works for the Dirichlet partition)
- `configs/experiments/phaseD2/*.yaml` (coalition-pooling config + 1–2 sweep points)
- Training runs + feature extraction + framework run (reusing the existing `run_phaseD_extract.sh` / `run_phaseD_framework.sh` patterns)
- `docs/feature_target_attack_profile.md` — closed profile, same structure as `pixel_trigger_baseline.md` and `model_replacement_profile.md`

**Success criteria.**

- ASR ≥ 0.90 on final model
- `linear_probe_acc` ≤ 0.80 (down from 0.99–1.00 baseline)
- `source_identity_preservation` ≤ 0.30 (down from 0.56–0.64)
- `centroid_l2` ≤ 1.0 (down from 2.7–3.1)
- No training divergence (unlike Phase D.1's NaN collapse window)

**Risks.**

- **Non-IID centroid bias.** Even with coalition pooling, the malicious clients' pooled class-c* set is a biased sample of the true class-c* distribution. The resulting `μ̂_{c*}` may pull triggered features to the attackers' local view of class c*, not to the global genuine centroid. This may limit how much `linear_probe_acc` can drop. Mitigation: report this limit honestly — if the attack converges to ~0.80 probe accuracy because of non-IID centroid bias, that is itself a finding about the cost of the FL threat model to the attacker.
- **Auxiliary term fights clean-task gradient.** Over-strong `λ_2` collapses clean accuracy. Mitigation: sweep and track clean accuracy as a hard constraint (reject any run with final clean acc < 0.80).
- **Coalition pooling changes threat model.** It requires malicious clients to coordinate off-protocol. We document this explicitly as an assumption; a stronger form of the attack would let malicious clients coordinate during training as well.

**Dependencies.** Phase D.1 infrastructure (feature extraction pipeline, framework metrics, checkpoint saving) — all exist and were validated in Cycle 1.

---

### Phase E.1 — Client-side detection under non-IID (defense v1)

**Goal.** Given **only its own labeled local data** (non-IID and small), can a client detect whether the just-received global model has a backdoor shortcut? The detector must be **label-aware but trigger-agnostic**, and **non-IID-native** — turning the non-IID disadvantage into an advantage via collective detection across clients.

**Reframing.** We cannot replicate our framework's linear probe (which requires labeled triggered examples). We can instead exploit what the client does have: **its own labeled non-IID samples and multi-round access to the evolving global model**. Two design directions will be implemented and compared in the same evaluation.

#### Design A — Per-client feature-drift monitor (primary)

The client maintains a small "trusted baseline" of feature statistics, computed at a reference round `r_ref` (e.g., round 10, after initial convergence is expected but before backdoor is typically embedded in Cycle 01 experiments). On every subsequent round:

1. Forward-pass its own labeled data `D_k` through the received global model.
2. Compute per-class mean features `{μ_k^c(r) : c ∈ classes(D_k)}` for every class `c` that the client has at least ~10 samples of. Call these the client's **observable classes**.
3. For each ordered pair `(c, c')` of observable classes with `c ≠ c'`, compute the projection of the class-`c` drift onto the direction from class `c` to class `c'`:

   $$
   \text{score}_k(c \to c', r) = \frac{\langle \mu_k^c(r) - \mu_k^c(r_\text{ref}),\; \mu_k^{c'}(r_\text{ref}) - \mu_k^c(r_\text{ref}) \rangle}{\| \mu_k^{c'}(r_\text{ref}) - \mu_k^c(r_\text{ref}) \|}
   $$

   This measures, in the client's own frame, how much class `c` has moved toward class `c'` in feature space between the reference round and now. Normal learning produces bounded, uncorrelated drift; a backdoor targeting `c'` systematically pushes many source classes toward `c'`.
4. Flag any pair `(c → c')` whose score exceeds a threshold (empirically calibrated on clean runs) or whose score is unusually large compared to other pairs in the same round.

**Why this is non-IID-native.** Each client measures drift *in its own local feature frame*, relative to *its own baseline*. There is no assumption that the client's class distribution matches the global distribution. A client with 30 class-3 samples can detect "class 3 is drifting toward class 2" just as well as a client with 300 class-3 samples — the estimate is noisier but the *direction* is still informative. Non-IID limits which pairs each client can observe, not whether those observations are valid.

#### Design B — Random-augmentation prediction-shift probe (secondary / baseline)

A label-free, feature-free sanity check that complements Design A. The client takes a small set of clean local samples, applies diverse random perturbations (random patches at random locations, color jitter, blur, rotation, pixel noise), asks the global model to classify each perturbation, and computes the per-class prediction frequency. Normal models split predictions roughly according to base-rates (each class gets ~1/43 of random-patch predictions); a backdoored model disproportionately routes random-patched inputs to the target class. Score: `max_class_fraction − uniform_baseline`.

This is simple, trigger-agnostic, fully client-local, and works with only ~50 clean local samples — so it is especially useful for clients whose local label distribution is too skewed for Design A to cover many class pairs.

**Why two designs.** Design A is the thesis-aligned contribution (feature-drift in representation space, directly connected to the Cycle 01 finding that backdoor displacement is a measurable signal). Design B is a baseline/sanity-check that we expect to work on pixel trigger and Bagdasaryan (because they cause the model to over-respond to random patches). Comparing the two on the Phase D.2 designed attack is the key experiment: it tells us whether representation-space detection beats input-space detection when the attack is explicitly feature-aware.

---

### Phase E.2 — Cross-client detection aggregation (part of Cycle 02, not deferred)

**Goal.** Combine detection signals across clients so that non-IID class-pair coverage becomes collective rather than per-client.

**Why it must be in Cycle 02.** Design A of Phase E.1 only produces flags for class pairs a given client can observe locally. A single client sees perhaps 5–15 observable classes out of 43; even cross-class pairs are thus restricted. Full class-pair coverage requires aggregating flags across clients. This turns non-IID from a disadvantage into an advantage: different clients specialize in different class pairs, and their union approaches full coverage.

**Mechanism.** After each client runs its detector, it reports a sparse "top-k suspicious pairs with scores" vector to the server as a side channel (distinct from the weight update — reported via `MetricRecord` alongside the model update). The server computes consensus across clients: if ≥ N clients independently flag the same ordered pair `(c → c*)` with score above threshold, declare the global model compromised. Decision action: either (i) roll back to the previous round's aggregate or (ii) reject the weight updates from the current round.

This is exactly the kind of FL-native defense the thesis direction calls for: the defender does not need to know the trigger, each client sees only its own non-IID slice, but collectively they observe the full feature-space drift. **The non-IID partition is what makes the coverage work** — each client specializes in different class pairs, and the union gives global coverage.

**Deliverables (E.1 + E.2 combined).**

- `src/fl_v2/attacks_defenses/defenses/feature_drift_detector.py` (new; Design A)
- `src/fl_v2/attacks_defenses/defenses/augmentation_prediction_detector.py` (new; Design B)
- Client-side integration point in `src/fl_v2/client_app.py` — an `@app.train()` hook that runs both detectors on the received global model and reports per-pair scores via `MetricRecord` back to the server (side channel, not the weight update)
- Server-side aggregation in a new strategy or hook that receives the scores and computes consensus
- Evaluation script in `analysis/` that sweeps all 4 attack conditions (Clean, Pixel-Trigger, Model Replacement, Feature-Target-D.2) × 3 detector variants (A alone, B alone, A+B cross-client aggregated) × all checkpoint rounds, producing a detection-AUROC plot
- `docs/client_probe_detector.md` (new, closed-profile format)

**Success criteria.**

- **Baseline calibration (no attack).** Design A and B both have per-round false-positive rate < 5 % on a clean run. Prerequisite — without a calibrated "normal" there is nothing to detect against.
- **Against pixel-trigger baseline (Phase C).** Design A per-client AUROC ≥ 0.8; cross-client aggregated AUROC ≥ 0.95. Design B per-client AUROC ≥ 0.9 (pixel trigger is easy for Design B because it is literally random-patch-like).
- **Against model replacement (Phase D.1).** The NaN collapse window alone is a trivial detection; Design A AUROC ≥ 0.9 easily. More interesting: can we detect the attack *before* the collapse (rounds 0–4) and *after* the recovery (rounds 25–100)? Target: AUROC ≥ 0.8 in both regimes.
- **Against feature-target attack (Phase D.2).** The adversarial benchmark. If Design A cross-client aggregated AUROC ≥ 0.7, we have a defense contribution. If it drops below 0.5, we have a "fundamental limit" finding. Either outcome is scientifically valuable.
- **False-positive rate across all attacks.** ≤ 10 % (we do not want a defense that flags clean runs).

**Risks.**

- **If Phase D.2 succeeds too well**, triggered features land fully inside the genuine distribution and no feature-drift detector can see them. This is a real possibility and is **the central scientific result if it happens** — it demonstrates a fundamental limit of data-dependent defenses and motivates the thesis's research question about why representation-space defense has a ceiling. Mitigation: don't hide this; report AUROC values honestly.
- **Non-IID noise swamps drift signal** on clients with very small samples per class (e.g. 5–10 samples). Mitigation: include only observable classes in Design A where the client has ≥ 10 samples; rely on cross-client aggregation to cover the gap.
- **Threshold calibration is cycle-specific** — thresholds tuned on Cycle 01 data may not generalize to Cycle 02 attacks. Mitigation: calibrate on a held-out clean run each cycle; report thresholds as configurable.
- **Side-channel threat.** Malicious clients can submit false detection reports to bias the consensus. Mitigation for Cycle 02: use median/trimmed aggregation on the reported scores to provide modest robustness; report it as an open problem for Cycle 03 (cryptographic commitments or trust bootstrapping).

**Dependencies.** Phase D.2 must be partially complete (at least one feature-target attack checkpoint) to evaluate the hardest case. Phases C and D.1 already provide pixel-trigger and model-replacement checkpoints.

---

### Phase F — Scientific writeup and supervisor checkpoint (close of cycle)

**Goal.** Consolidate Cycle 02 findings into a shareable document and hold a checkpoint meeting with the supervisor before starting Cycle 03.

**Deliverables.**

- This cycle document updated with the headline result and completion checklist (see section 6)
- Presentation-quality figures: one per attack profile (Phase D.2 in the same format as pixel trigger and model replacement), one detection-AUROC plot for E.1 across all three attacks
- Draft paragraph on "why the probe-orthogonality finding motivates client-side defenses" — the thesis-facing bridge that connects Cycle 01's empirical finding to Cycle 02's defense design

---

## 4. Paper reading list (grouped by theme)

### Theme A — Optimization-based and feature-space backdoor attacks

- **Hidden Trigger Backdoor Attacks (Saha et al., AAAI 2020).** Places the trigger in feature space (not pixel space) by optimizing perturbations that move features toward a target. Direct inspiration for the Phase D.2 auxiliary-loss design.
- **LIRA: Learnable, Imperceptible and Robust Backdoor Attacks (Doan et al., ICCV 2021).** Jointly learns a trigger generator and the backdoor — the attack *is* a learned function of the input, not a fixed patch. This is what a fully designed attack looks like.
- **Dynamic Backdoor Attacks Against Machine Learning Models (Salem et al., EuroS&P 2022).** Trigger is input-dependent; defenses that assume static triggers fail.
- **WaNet — Imperceptible Warping-based Backdoor Attack (Nguyen & Tran, ICLR 2021).** Warping-based trigger that bypasses pixel-level inspection.

### Theme B — Backdoor attacks in federated learning (state of the art)

- **How to Backdoor Federated Learning (Bagdasaryan et al., AISTATS 2020).** The original model-replacement paper. We closed a version of this in Phase D.1; worth re-reading for the stealth-constraint formulation we deliberately omitted.
- **DBA: Distributed Backdoor Attacks against Federated Learning (Xie et al., ICLR 2020).** Splits the trigger across clients. Backlog for Cycle 03.
- **Attack of the Tails: Yes, You Really Can Backdoor Federated Learning (Wang et al., NeurIPS 2020).** Edge-case backdoor — attack the tail of the data distribution where defenses are weakest. Shows the attack surface is larger than just center-of-distribution triggers.
- **Manipulating the Byzantine (Shejwalkar & Houmansadr, NDSS 2021).** The canonical adaptive attack paper. Cycle 03 reference — we will return to this when the supervisor's adaptive-attack direction is on the critical path.

### Theme C — Backdoor detection with no trigger knowledge (trigger-agnostic)

- **Neural Cleanse (Wang et al., S&P 2019).** Reverse-engineers candidate triggers from a trained model. Requires the model weights and labeled clean data — both realistic for our client. This is the closest prior work to a client-side detector and the starting point if we want per-target-class reverse engineering.
- **STRIP (Gao et al., ACSAC 2019).** Superimposes clean inputs and checks prediction-entropy collapse on triggered inputs. Trigger-agnostic and label-free — the inspiration for Design B in Phase E.1. Critically, STRIP assumes the defender has access to the *runtime inputs* and tests individual samples; our non-IID reframing adapts it to operate on per-client clean data instead.
- **Activation Clustering (Chen et al., AAAI 2019 workshop).** Clusters activations on the training set to separate poisoned from clean samples. Assumes access to the training set and class labels — fits the client-side threat model directly, but the non-IID constraint means individual clients see a highly skewed slice.
- **SPECTRE (Hayase et al., ICML 2021).** Stronger spectral defense; robust covariance estimation. Our `probe_pc_alignment` metric was inspired by this line, and our Cycle 01 finding (C6 ≈ 0) is the quantitative reason why SPECTRE-style unsupervised spectral defenses miss what a supervised probe finds.
- **Beatrix (Ma et al., NDSS 2023).** Gram-matrix-based detection. Trigger-agnostic, uses class-conditional feature statistics — the closest analogue to our Design A drift detector in the non-federated setting.

### Theme G — Non-IID-aware federated defenses (directly relevant to Phase E)

- **FedCon (Zhou et al., ICLR 2023).** Federated consistency-based defense that exploits cross-client disagreement. Non-IID-native by construction. Reading priority for Phase E.2 aggregation design.
- **FLCert: Provably Secure Federated Learning against Poisoning Attacks (Cao et al., IEEE T-IFS 2022).** Certifies robustness under non-IID clients; the formal analysis is useful even if we do not certify ourselves.
- **FedInv (Zhao et al., CVPR 2022).** Client-side backdoor detection via gradient inversion — uses the global model's gradients on each client's local data to recover a "virtual trigger". Very close in spirit to our Design A (using the client's own data to probe the model's current state).
- **BackdoorIndicator (Li et al., USENIX Security 2024).** Recent work on FL backdoor detection using out-of-distribution samples at the server. Fits the trigger-agnostic thesis direction; worth reading even though it is server-side.

### Theme D — Federated learning defenses (beyond Byzantine-robust aggregation)

- **FLAME: Taming Backdoors in Federated Learning (Nguyen et al., USENIX Security 2022).** Combines clustering, clipping, and noising; widely cited current defense.
- **DeepSight: Mitigating Backdoor Attacks in FL through Deep Model Inspection (Rieger et al., NDSS 2022).** Inspects per-client model fingerprints and classifies benign vs malicious.
- **FLTrust: Byzantine-robust FL via Trust Bootstrapping (Cao et al., NDSS 2021).** The server uses a small trusted dataset to bootstrap trust in clients. An FL-native version of the "server has labeled data" defense paradigm.
- **CrowdGuard: Federated Backdoor Detection (Rieger et al., NDSS 2024).** Cross-client detection voting — directly relevant to Phase E.2.

### Theme E — Test-time adaptation (background; we are explicitly NOT using TTA as the defense, but the intuitions transfer)

- **Tent: Fully Test-time Adaptation by Entropy Minimization (Wang et al., ICLR 2021).** The canonical TTA paper — updates BN stats at test time to match current input distribution. Not directly a backdoor defense, but the "use local data to correct the model" intuition is what we are borrowing in spirit.
- **Test-Time Training with Self-Supervision for Generalization under Distribution Shifts (Sun et al., ICML 2020).** An earlier TTT approach using self-supervision. Shows the limit of how much you can recover with unlabeled test data alone — informative for where our label-aware approach improves on pure TTA.

### Theme F — Vision Transformers and backdoor (reading-only for Cycle 02; execution deferred to Cycle 03 or later)

- **An Image is Worth 16×16 Words (Dosovitskiy et al., ICLR 2021).** The base ViT paper. Background.
- **Data-free Backdoor Attacks on Vision Transformers (Lv et al., TPAMI 2023).** ViT-specific backdoor; shows attention layers can be trojaned without retraining.
- **Backdoor Attacks in the Supply Chain of Masked Image Modeling (Yuan et al., CVPR 2024).** Self-supervised ViT pretraining as an attack surface.
- **Visual Prompt Tuning (Jia et al., ECCV 2022).** Not a backdoor paper, but the canonical way to adapt a frozen ViT to a new task. If we want client-side ViT in Cycle 03+, this is the efficient adaptation method.

---

## 5. Open questions (for supervisor / for Cycle 03)

1. **Malicious-coalition centroid quality under non-IID.** Phase D.2 uses coalition pooling to let the attackers share class-c* samples. But with 5 malicious clients out of 50 and a Dirichlet α=0.5 partition, the coalition's pooled class-c* samples may still be a biased slice of the global class-c* distribution. Does this bias put a floor on how low `linear_probe_acc` can go for D.2, and if so, what is it? If the floor is > 0.5, a probe-based defender retains some advantage even against a "perfect" designed attack.
2. **Class-pair coverage in cross-client detection (Phase E.2).** Each client can only flag pairs `(c → c')` where both `c` and `c'` are observable in its local data. With 50 clients × Dirichlet α=0.5, what fraction of the 43×42 ordered class pairs is collectively covered? Compute this analytically on the Dirichlet partition and report it — it determines the detection completeness of the cross-client aggregation.
3. **Stealth-constrained model replacement.** Cycle 01 showed raw Bagdasaryan scaling collapses the model at rounds 5–10. A stealth-constrained variant is still missing from the comparison table. Cycle 02 defers this; Cycle 03 candidate.
4. **Side-channel integrity in cross-client detection.** Phase E.2 aggregates detection scores reported by clients, but malicious clients can report false scores. Cycle 02 uses trimmed aggregation as a partial mitigation; Cycle 03+ should address it with median/trimmed aggregation that is provably robust or with cryptographic commitments.
5. **Reference round `r_ref` for Phase E.1 Design A.** We assumed round 10 is a reasonable baseline ("clean enough, attack not yet embedded") based on Cycle 01 dynamics — 5mal pixel trigger takes ~30 rounds to lock in, 15mal takes ~5 rounds. For Phase D.2 where the attack might converge differently, the baseline assumption may not hold. Open: how do we calibrate `r_ref` in practice? Does it need to be learned from clean runs, or can we use a rolling-window baseline ("N rounds ago") instead?
6. **Supervisor's adaptive-attack direction.** Deferred to Cycle 03. Cycle 02 deliberately scopes away from it to keep attack and defense work coherent. When Cycle 03 picks it up, Shejwalkar & Houmansadr 2021 is the starting reference.
7. **Supervisor's ViT client direction.** Reading list in Theme F; execution deferred to Cycle 03 or later. The question of whether our representation-space framework and Phase E drift detector transfer to attention-based backbones is the Cycle 03/04 research question.

---

## 6. Completion checklist

Updated as Cycle 02 progresses.

- [ ] Phase D.2 implementation — `feature_target_attack.py` + client_app.py branch + phaseD2 configs
- [ ] Phase D.2 training runs completed (5mal and 15mal, coalition pooling strategy, at least one λ_2 value)
- [ ] Phase D.2 feature extraction + framework metrics run
- [ ] `docs/feature_target_attack_profile.md` written with real numbers
- [ ] Phase E.1 Design A implementation — `feature_drift_detector.py`
- [ ] Phase E.1 Design B implementation — `augmentation_prediction_detector.py`
- [ ] Client-side hook in `client_app.py` to run both detectors and report scores via `MetricRecord`
- [ ] Phase E.2 server-side aggregation hook
- [ ] Evaluation script — attack × detector × round AUROC sweep
- [ ] Baseline calibration run on clean experiment to set false-positive thresholds
- [ ] `docs/client_probe_detector.md` written with real numbers
- [ ] Cycle 02 headline result filled in at top of this file
- [ ] Cycle 02 entry in `INDEX.md` moved to "closed" with headline result
- [ ] Supervisor checkpoint meeting held; Cycle 03 topic confirmed

---

## Update 2026-04-17 — Architecture decisions (pre-D.2)

Before writing any Phase D.2 code, we ran a Ray-process verification probe (`configs/experiments/probe/ray_process_check.yaml`, SLURM job 6432245) and an Axis-C methodology audit. Both informed architectural decisions recorded here.

### Ray-process probe result

**Observation.** With `num-supernodes = 50` and `num-gpus = 0.10` per supernode in `$HOME/.flwr/config.toml`, running a 3-round experiment with `fraction-train = 0.2` (10 clients per round):

- Ray (version **2.51.1**, Flower version **1.27.0**) instantiates **10 `ClientAppActor` processes**, each a distinct Python PID with its own memory space.
- The 50 supernodes are multiplexed **5:1** onto the 10 actors — any given actor serves ~5 different `client_id`s over the run.
- Module-level Python globals (`_index_map_cache`, `_probe_sentinels`) have distinct `id()` values in each actor: each actor's `_probe_sentinels` dict only ever contains sentinels written by clients running inside that actor. **No cross-actor visibility.**
- Within an actor the cache accumulates — client 22 runs in PID 747738 round 1, leaves a sentinel, client 45 runs in PID 747738 round 3 and reads it. Across actors the sentinels never cross.

**Implication for Phase D.2 coalition pooling.** The cycle plan's proposal to share `μ̂_{c*}` via a module-level cache "mirroring `_index_map_cache`" is incorrect. `_index_map_cache` appears to work only because the partition is a pure function of config; every actor recomputes the same dict independently. A coalition centroid is data-dependent on each round's global model and cannot be reconstructed without explicit cross-actor communication.

### Commitment to Option A for Phase D.2 coordination

Option A — **server-mediated coordinator** via Flower's existing message protocol — is adopted for Phase D.2 coalition pooling. The message flow:

1. *Train reply.* Each malicious client includes in its `MetricRecord` a compact payload: the sum of its local class-`c*` penultimate features plus its class-`c*` sample count (≈ 512 floats + 1 int ≈ 2 KB per malicious client per round).
2. *Server aggregation.* A custom strategy hook collects these payloads across the malicious clients selected in the round, sums them, divides by the total count → produces `μ̂_{c*}` for the next round.
3. *Next train config.* The server attaches `μ̂_{c*}` (≈ 2 KB) to the outgoing `ConfigRecord`. Malicious clients read it from the incoming message and use it in the auxiliary loss.

**Round-1 cold start (per Saha et al. 2020 "Hidden Trigger" / LIRA 2021 conventions).** No pooled centroid exists yet. Use the classifier-head row `W[c*]` L2-normalized as the auxiliary-loss target direction. `W[c*]` is a **direction**, not a centroid — magnitude and origin differ from `μ̂`. The round-1 loss therefore re-formulates as

\[\lambda_2 \cdot \big(- \widehat{W}[c^*]^{\top} f_\theta(\tau(x))\big)\qquad\text{where } \widehat{W}[c^*] = W[c^*] / \|W[c^*]\|_2 \]

(a feature-space "push along the target classifier direction"), and switches to the `‖f_θ(\tau(x)) − \hat{\mu}_{c^*}\|^2` centroid-distance loss from round 2 onwards.

### Methodology audit result: Axis C metrics re-implemented in place

Independently of Phase D.2, a methodology audit surfaced a sampling flaw in the Cycle-01 Axis C implementation: the linear probe discarded ~94 % of triggered features, the C6 probe–PC alignment compared a balanced probe direction against an imbalanced-pool top-PC, and C2/C3/C4/C5 shared a single `RandomState`. `compute_axis_c` in `analysis/framework_metrics.py` has been rewritten in place (no parallel v2 function); all nine Cycle-01 feature `.npz` files were re-analyzed with `--seed 4242` against the new code. Errata sections at the bottom of [`pixel_trigger_baseline.md`](../pixel_trigger_baseline.md) and [`model_replacement_profile.md`](../model_replacement_profile.md) report the new numbers alongside the v1 ones; the methodology addendum at the bottom of [`representation_space_framework.md`](../representation_space_framework.md) documents the change. One preliminary scientific consequence: under balanced-PCA C6, the probe–top-PC alignment rises from ~0.01-0.25 (imbalanced) to ~0.27-0.60 (balanced), meaning the Cycle-01 "spectral defenses look in the wrong direction" headline is weaker than the v1 numbers suggested — the probe direction is only *partially* orthogonal to the top-PC once the class imbalance is removed from the PCA input. Probe separability itself is unaffected (balanced accuracy 0.93-1.00, AUROC ≥ 0.99).

### Deferred items added to Section 5 (open questions)

- **R9 — Trusted-baseline-round calibration for Phase E.1 Design A.** The cycle plan assumes round 10 is a reasonable "trusted baseline" round, but under Bagdasaryan's round-1 replacement the round-1 model is already poisoned. If Phase D.2 converges in round 1 or if the attacker starts earlier than round 10, the baseline is contaminated and drift detection cannot distinguish pre- from post-attack. Open: learn the reference round from clean runs or switch to a rolling-window baseline? Resolve at E.1 implementation time.
- **R10 — Classifier-head direction vs centroid dimensionality.** Round-1 fallback uses `W[c*]` as a direction; rounds 2+ use `μ̂_{c*}` as a centroid. Under ResNet18 both are `d = 512` vectors, but they are not comparable as geometric targets: a direction has no origin, a centroid does. The auxiliary loss switches formulation between round 1 and round 2 (see above). Open: does this switch produce a training-dynamics discontinuity worth monitoring? Resolve at D.2 implementation time.

### Cleanup after this update

The temporary Ray-process probe artifacts (the `debug-probe-multiproc` branch in `src/fl_v2/client_app.py`, the `debug-probe-multiproc` config key in `pyproject.toml`, and `configs/experiments/probe/ray_process_check.yaml`) are removed once this update is committed. The permanent record of the finding lives here.
