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

The following platform components are already available or actively being integrated:
- Flower latest Message API baseline
- ClientApp / ServerApp structure
- GTSRB data pipeline
- non-IID partitioning
- client-local train/val split
- CNN baseline
- end-to-end FL simulation
- server-side global test evaluation

Implemented and smoke-tested attack modules:
- label flipping attack (data poisoning baseline)
- pixel-trigger backdoor attack (with ASR metric)

Implemented and smoke-tested defense modules:
- NormTrackingFedAvg (update norm logging, wraps FedAvg)
- NormClippedFedAvg (L2 norm clipping before aggregation)
- FedMedian, FedTrimmedAvg (Flower built-in robust aggregation)
- Krum, MultiKrum, Bulyan (Flower built-in Byzantine-tolerant selection)

Possible near-term additions:
- stronger backbone (ResNet18)
- model replacement / scaling attack (Bagdasaryan)
- DBA (Distributed Backdoor Attack)

Do not assume every near-term module is finalized unless verified from the code.

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

Current near-term development priorities are:

1. ~~stabilize baseline platform behavior~~ ✓ done
2. ~~complete and verify modular defense integration~~ ✓ done
3. ~~compare clean / attack / defense / attack+defense~~ smoke-tested; full-round experiments pending
4. run full-round experiments (10–15 rounds) to validate ASR vs defense tradeoffs
5. strengthen backbone choices (ResNet18)
6. gradually move from heuristic attacks toward stronger attack settings (model replacement, DBA)
7. prepare the platform for future autonomous-driving datasets/tasks

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