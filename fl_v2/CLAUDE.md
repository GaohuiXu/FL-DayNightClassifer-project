# CLAUDE.md

## Project Identity

This repository is a federated learning research platform built with the latest Flower Message API.

Its current role is to serve as a clean, modular, reproducible experimentation platform for:
- non-IID federated image classification
- attack baselines
- defense baselines
- stronger backbone experiments
- later migration toward autonomous-driving research settings

---

## Current Stage vs Final Thesis Direction

### Current stage
The current codebase uses GTSRB as a platform-building benchmark:
- 43-class image classification
- multi-client federated learning
- non-IID partitioning
- attack/defense modularization
- experiment logging and reproducibility validation

GTSRB is not the final thesis task. It is the current engineering and experimentation platform.

### Final thesis direction
The final thesis direction is:
- securing federated learning-based autonomous driving against backdoor attacks
- studying stronger and more structural attack settings beyond simple heuristics
- later moving toward autonomous-driving and possibly multimodal settings
- eventually supporting research on fusion-layer vulnerability and stronger defenses

When giving suggestions, prioritize ideas that either:
1. improve the current FL experimentation platform, or
2. prepare the codebase for later migration toward the thesis direction.

---

## Current Research Direction

The current research direction is evolving beyond simple FL benchmark engineering.

The working research hypothesis is:

1. move from heuristic attacks toward stronger optimization-based attacks
2. evaluate whether strong server-side defenses remain effective under stronger attack settings
3. if server-side defenses become insufficient, explore whether part of the defense should move to the client side
4. investigate whether ideas from test-time adaptation (TTA), although not originally designed for backdoor defense, can inspire client-side defense mechanisms

This is still an evolving research direction rather than a fixed conclusion.
When giving suggestions, distinguish clearly between:
- currently implemented baseline engineering
- short-term experimental extensions
- longer-term thesis-facing research ideas

---

## Current Platform Status

The following platform components are implemented and in active use:
- Flower latest Message API baseline (ClientApp / ServerApp)
- GTSRB data pipeline with non-IID Dirichlet partitioning
- client-local train/val split
- CNN baseline and **ResNet18** (used for all recent experiments)
- end-to-end FL simulation with checkpoint saving at configured rounds
- server-side global test evaluation (accuracy, TCA, ASR in a single pass)
- **Representation-space analysis framework** (`src/fl_v2/analysis/`): feature
  extraction from saved checkpoints, 4-axis rigorous attack profile
  (`analysis/framework_metrics.py`), per-experiment JSON + CSV outputs,
  cross-experiment comparison (`analysis/compare_profiles.py`),
  t-SNE trajectory visualization, training-curves + defense-comparison plots.
  See `docs/representation_space_framework.md` for the methodology.

Implemented attack modules:
- label flipping attack (data poisoning baseline)
- pixel-trigger backdoor attack (with ASR metric) — **closed baseline**, see
  `docs/pixel_trigger_baseline.md`
- model replacement (Bagdasaryan) — data poisoning reuses pixel-trigger,
  client update is scaled by `n/k` before sending. **Closed baseline**, see
  `docs/model_replacement_profile.md`

Implemented defense modules:
- NormTrackingFedAvg (update norm logging, wraps FedAvg)
- NormClippedFedAvg (L2 norm clipping before aggregation)
- FedMedian, FedTrimmedAvg (coordinate-wise robust aggregation — custom
  implementations with norm logging, not Flower's built-ins)
- Krum, MultiKrum (Flower built-in Byzantine-tolerant selection)
- Bulyan (custom implementation with norm logging — Flower's built-in has a
  known dtype bug at 1.27.0)

Possible near-term additions:
- DBA (Distributed Backdoor Attack, Xie et al. 2020)
- Client-side representation-space defenses (TTA-inspired)
- Multimodal / autonomous-driving dataset migration (long-term)

Do not assume every module in this list is finalized unless verified from the
code — but the representation-space framework, ResNet18, pixel-trigger, and
model replacement have all been fully evaluated.

---

## Current Scientific Findings (Phase C v2 + Phase D.1)

The representation-space framework in `docs/representation_space_framework.md`
and the two closed baseline profiles provide the current empirical ground
truth for how backdoor attacks manifest in feature space. When suggesting new
attacks or defenses, evaluate them against this framework — moving ASR alone
is not sufficient to claim a representation-space improvement.

1. **The pixel-trigger backdoor is a "joint weak attack".** Successful
   attacks (ASR 0.86–0.97) place triggered features in a **geometrically
   distinct "middle region"** that is linearly separable from both the
   original source class and the genuine target class. The feature extractor
   partially shifts triggered features ~25-30% along the natural
   source→target direction; simultaneously the classifier head extends its
   target-class decision region to cover this middle region. Neither
   component is strongly attacked individually — they meet halfway. Key
   evidence: `linear_probe_acc = 0.99-1.00` (trivially separable),
   `centroid_l2 ≈ 2.7-3.1` (far from target), `shift_alignment ≈ 0.65-0.80`
   (partially exploits natural inter-class direction),
   `source_identity_preservation ≈ 0.56-0.64` (per-source structure
   preserved, ~40% compression only). See `docs/pixel_trigger_baseline.md`.

2. **Model replacement (Bagdasaryan) is stronger in ASR but NOT in
   representation space.** Scaling client updates by `n/k` amplifies the
   same pixel-trigger signal and concentrates it into a lower-dimensional
   subspace (`shift_rank_eff` drops from 33 to 11 at 15 mal), but does not
   change the mechanism. Triggered features are still linearly separable
   (`linear_probe_acc = 0.98-1.00`), still sit in a middle region
   (`centroid_l2` actually *grows* to 3.7-4.3), and are now **markedly more
   detectable by spectral defenses** (`spectral_score` rises from 0.07 to
   1.73 — a 26× increase). The clean negative finding: attack *mechanism*
   determines the representation-space profile, not attack *strength*.
   ModelRep also has a **brittle early window** — `ASR@5 = ASR@10 = 0.00`
   despite scaling by 10× because the round-1 replacement has poor clean
   utility and honest updates wash it out until the attack rebuilds
   around round 25. See `docs/model_replacement_profile.md`.

3. **Unsupervised server-side defenses have a fundamental blind spot.**
   The separating direction between triggered and genuine features is
   nearly orthogonal to the top principal component
   (`probe_pc_alignment ≈ 0.01-0.16` for the pixel-trigger baseline), so
   spectral signature defenses (Tran et al. 2018) cannot locate it. A
   supervised linear probe finds it trivially (0.99-1.00 accuracy) but
   requires labeled triggered examples — which only **clients** have in
   federated learning. This quantitatively motivates the thesis direction
   of client-side representation-space defenses (TTA-inspired).

4. **FedMedian has a breakdown point around 30% malicious.** At 15/50
   malicious clients, FedMedian fails to suppress the attack and
   paradoxically produces the **cleanest** backdoor fingerprint
   (`centroid_l2 = 1.33`, smallest of all runs; `spectral_score = 0.19`,
   vs 0.06 for vanilla no-defense). The failed defense *sharpens* the
   attack because the median aggregator filters out the benign gradient
   diversity while the malicious majority passes through unfiltered.

5. **Krum has a temporary vulnerability window at round 10.** ASR briefly
   spikes to 0.96-1.00 before Krum begins reliably selecting honest
   clients and the ASR collapses to near-zero. This is a real-time attack
   opportunity that standard Byzantine analysis of Krum does not
   emphasize. The cost: Krum's final accuracy drops to 73-74% (vs 94-95%
   baseline), confirming the utility-robustness trade-off.

6. **Phase D.2 target:** an optimization-based attack that adds an
   auxiliary loss at local training time:
   `L = L_clean + λ1 L_backdoor + λ2 ‖f(τ(x)) − μ_{c*}‖²`
   to explicitly pull triggered features onto the genuine target-class
   centroid. If this succeeds, the target metric movements are:
   `linear_probe_acc → 0.5` (indistinguishable),
   `source_identity_preservation → 0` (destination clustering),
   `centroid_l2 → 0` (features reach the genuine class region),
   `concentration_ratio → 1.0` (match natural class variance),
   all while holding ASR ≥ 0.9. If these move in the target direction,
   we have a qualitatively new attack class in representation space
   rather than just a stronger version of the pixel-trigger heuristic.

---

## Architecture Expectations

Please preserve the modular project structure:

- `data/` → dataset loading, partitioning, transforms
- `models/` → model definitions
- `training/` → local training/evaluation logic
- `client_app.py` → client-side FL behavior
- `server_app.py` → server-side orchestration
- `strategy/` → custom FL strategies and aggregation control
- `attacks_defenses/` → attack and defense implementations
- `utils/` → generic helpers

Do not collapse multiple responsibilities into one file unless there is a strong reason.

---

## Attack / Defense Design Expectations

### Attacks
Attacks should preferably be modular and easy to enable/disable.
Typical current examples:
- label flipping
- backdoor trigger injection
- malicious local update manipulation

### Defenses
Defenses should preferably be modular and easy to compare.
Typical current examples:
- norm clipping
- median
- trimmed mean
- other robust aggregation methods

Heuristic attacks/defenses are mainly baseline steps.
Future thesis-oriented work may require stronger optimization-based attacks and more AD-specific analysis.

---

## Collaboration Rules for Claude

When assisting with this repository:

1. First analyze the existing codebase before proposing changes.
2. Prefer minimal, explicit, incremental changes.
3. Do not introduce broad refactors unless necessary.
4. Keep the clean baseline runnable at all times.
5. Preserve compatibility with the latest Flower Message API.
6. Do not revert the codebase to legacy `NumPyClient` style.
7. Preserve reproducibility of data partitioning and experiment setup.
8. Separate:
   - immediate engineering fixes
   - experiment-platform improvements
   - thesis-facing research suggestions
9. When suggesting a new feature, explain how it helps either:
   - the current benchmark platform, or
   - future migration toward the thesis direction.
10. For research-facing suggestions, do not treat current hypotheses as proven conclusions.
11. When discussing attacks or defenses, distinguish between:
    - heuristic baselines
    - stronger optimization-based methods
    - server-side defenses
    - client-side defenses
12. When suggesting TTA-related ideas, treat them as inspiration for client-side defense exploration, not as an already validated backdoor defense solution.

---

## Experiment Principles

- Always keep a clean baseline for comparison.
- Attack modules should be evaluated against clean training behavior.
- Defense modules should be evaluated in terms of both utility and robustness.
- Prefer explicit logging, saved results, and reproducible configurations.
- Global RNG seeding (`random`, `numpy`, `torch`) is performed at server startup using the configured `seed` value. This ensures identical initial model weights across all experiments with the same seed.
- New modules should be easy to reuse across future datasets and tasks.
- Avoid adding tightly coupled code that would make future migration difficult.
- For non-trivial tasks, first provide:
  1. codebase understanding
  2. minimal change plan
  3. risks / edge cases
  4. implementation proposal

---

## Near-Term Priorities

Near-term work should prioritize platform stability and controlled comparison before thesis-scale expansion.

Completed (the current empirical baseline):

1. ~~stabilize baseline platform behavior~~ ✓ done
2. ~~complete and verify modular defense integration~~ ✓ done
3. ~~compare clean / attack / defense / attack+defense~~ ✓ done (Phase B smoke-tested, Phase C v2 produced the full comparison at 100 rounds)
4. ~~run full-round experiments (100 rounds) to validate ASR vs defense tradeoffs~~ ✓ done (8 experiments in phaseC_v2, 2 in phaseD)
5. ~~strengthen backbone choices (ResNet18)~~ ✓ done (all recent experiments)
6. ~~gradually move from heuristic attacks toward stronger attack settings (model replacement)~~ ✓ done — produced a clean negative result documented in `docs/model_replacement_profile.md`
7. ~~build a rigorous representation-space analysis framework~~ ✓ done (`docs/representation_space_framework.md`, 4-axis profile, closed baselines)

Current focus: **Cycle 02 — Designed Attacks & Client-Side Defenses.**

Research cycles are recorded in `docs/roadmap/`. The current cycle document
is [`docs/roadmap/cycle_02_designed_attacks_and_client_defenses.md`](docs/roadmap/cycle_02_designed_attacks_and_client_defenses.md),
and the chronological index of all cycles is at
[`docs/roadmap/INDEX.md`](docs/roadmap/INDEX.md).

Cycle 02 phases (see the roadmap for details, success criteria, and risks):

8. **Phase D.2 — optimization-based feature-space attack.** Malicious
   clients add an auxiliary loss `λ‖f(τ(x)) − μ̂_{c*}‖²` to local training
   to place triggered features *inside* the genuine target-class region.
   `μ̂_{c*}` is estimated via coalition pooling across malicious clients
   (required because under Dirichlet α=0.5 any individual malicious client
   may have few or zero class-c* samples). Success ⇔ `linear_probe_acc` ≤
   0.80, `source_identity_preservation` ≤ 0.30, `centroid_l2` ≤ 1.0, while
   holding ASR ≥ 0.90.
9. **Phase E.1 — client-side detection under non-IID.** Each client uses
   only its own labeled non-IID local data (no trigger knowledge) to run
   two complementary detectors on each received global model: (A) a
   per-client feature-drift monitor that measures how much each observable
   class has drifted toward each other observable class relative to a
   trusted baseline round, and (B) a random-augmentation prediction-shift
   probe inspired by STRIP. Both are trigger-agnostic. **Note:** the
   Cycle-01 framework's linear probe used 12 k balanced labeled samples,
   which a single client does not have — the defense must be non-IID-native
   from the start, and the "client trains a linear probe" story does not
   directly transfer.
10. **Phase E.2 — cross-client detection aggregation.** Clients report
    sparse `(c → c')` suspicion scores to the server via a side channel.
    Consensus across clients covers the class pairs that no single client
    can observe alone — the non-IID partition becomes the reason the
    defense has complete coverage rather than the reason it fails.

Longer-term (Cycle 03+, not executed in Cycle 02):

11. **Adaptive attacks (supervisor suggestion).** Attackers sacrifice a few
    malicious clients to probe the defender's aggregation behavior, then
    adapt the remaining attack. Reference: Shejwalkar & Houmansadr 2021.
12. **ViT client models (supervisor suggestion).** Migrate to a pre-trained
    Vision Transformer backbone and re-run the representation-space
    framework. Tests whether Cycle-01 findings and the Phase E drift
    detector generalize to attention-based architectures.
13. **Multimodal / fusion-layer backdoors (autonomous-driving migration).**
    Once GTSRB platform work is complete, migrate to a perception
    benchmark (BDD100K, nuScenes) and study whether the same
    representation-space framework applies to fusion-layer features.

---

## Workflow Constraints

Keep the current local workflow runnable unless there is a clearly better replacement.

### Terminal 1
```bash
./start_superlink.sh
```
### Terminal 2
```bash
./run_flwr_local.sh
```
### STOP
```bash
./stop_superlink.sh
```

Do not replace existing entry scripts or command-line workflow unless there is a clear, tested, and simpler alternative.

`FLWR_HOME` is set to `$HOME/.flwr` (Ceph shared filesystem) in `flwr_local_env.sh`.
The federation config (`local-simulation-gpu`) is defined in `$FLWR_HOME/config.toml`.
Do NOT change `FLWR_HOME` to `/tmp` — it is node-local and disappears between HPC sessions.

---

## Things To Avoid
1. do not rewrite the project into a monolithic structure
2. do not silently replace the latest Flower API design with older patterns
3. do not mix attack logic and defense logic into unrelated files
4. do not remove reproducibility-related controls without reason
5. do not over-engineer for the final multimodal setting too early
6. do not assume the current GTSRB benchmark is the final scientific target

---

## Preferred Planning Style

For complex tasks, first produce a plan based on the current codebase.
The plan should explicitly include:
1. which files are relevant
2. what is already implemented
3. what should be minimally changed
4. possible edge cases or risks
5. how the proposed change supports either:
   - the current FL experimentation platform, or
   - the longer-term thesis direction

Do not jump directly into broad implementation proposals before understanding the existing code structure.