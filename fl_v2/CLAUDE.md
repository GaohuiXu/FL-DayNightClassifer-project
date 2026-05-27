# CLAUDE.md

## Project Identity

This repository is a federated learning research platform built on the latest
Flower Message API.

**Thesis-paper target: USENIX-Security.** The contribution is a **novel
general backdoor defense for federated learning**. The current benchmark
is GTSRB-43 / ResNet18; the long-term migration target is federated
learning for autonomous driving (multimodal / fusion-layer backdoors).

---

## Current Stage vs Final Thesis Direction

### Current stage
Audit-fixed codebase, evaluated on GTSRB-43 (43-class traffic-sign
classification) with ResNet18 trained from scratch, 50 clients, Dirichlet
α=0.5 non-IID. Used as the experimentation platform for backdoor
attack/defense mechanism study.

### Final thesis direction
A novel general backdoor defense for federated learning, targeting
**USENIX-Security**. The defense will be evaluated against:
- The closed Cycle-02 Wave-1 baselines on GTSRB-43 (pixel, model
  replacement, DBA × {FedAvg, NormClip, MultiKrum, FoolsGold, FLAME}).
- Stronger adaptive attacks reproduced from 2023-2025 literature
  (current Cycle 03).
- Eventually: federated learning for autonomous driving (multimodal /
  fusion-layer) once the GTSRB platform results are paper-ready.

When giving suggestions, prioritize ideas that either:
1. advance the current cycle's research goal (see **Current Focus** below),
2. improve the FL experimentation platform's reproducibility / efficiency,
   or
3. prepare the codebase for the later AD migration.

---

## Current Research Direction

The cycle-based research pipeline is:

1. **Reproduce** current strong and novel attacks/defenses from
   2022-2025 literature faithfully on the audit-fixed codebase.
2. **Understand mechanisms** — for each implemented attack, characterize
   its gradient-space signature; for each defense, characterize what
   signal it exploits and where it fails.
3. **Build intuitions and experience** for what a new general defense
   must do that existing ones don't — guided by attacks that break SOTA
   defenses on our threat model.
4. **Design a novel defense** — only after step 3 produces a concrete
   attack class that the literature's strongest defenses (FLAME, as of
   2026) cannot handle.

Cycle-02 Wave-1 closed step 2 for the static-attack matrix: FLAME drives
ASR to 0.000 across {pixel, model_replacement, DBA} on our threat model.
There is no design target for step 4 yet. **Cycle 03 is in step 1 for
adaptive attacks** (LP, small-LR, A3FL) precisely to find a target — see
the plan referenced in **Current Focus**.

When giving suggestions, distinguish clearly between:
- currently implemented baseline engineering
- short-term experimental extensions (current cycle plan)
- longer-term thesis-facing research ideas (post current cycle)

---

## Current Focus

**Cycle 03 — Stronger Adaptive Backdoor Attacks for FL.**

- Plan: `~/.claude/plans/codebase-docs-swirling-gosling.md` (approved
  2026-05-28).
- Wave-1 baseline log: `fl_v2/docs/cycle_02/wave1_log.md`.

**Cycle 03 goal**: reproduce ≥1 adaptive backdoor attack from the
literature that drives FLAME-defended ASR above the 0.000 baseline.
Without such an attack, designing a new defense is busywork. The plan
covers a Day-0 logging hardening PR (per-class ASR + exact
trigger-attributable ASR + MultiKrum NormTracking refactor), then 3
adaptive attack workstreams (small-LR / BackdoorIndicator USENIX'24,
LP / Backdoor-Critical-Layers ICLR'24, A3FL NeurIPS'23), then a
conditional heaviest one (3DFed S&P'23). Estimated ~10 working days at
Wave-1 cadence.

Decision-points still pending in the plan:
- A3FL coordination mechanism (user reading paper before WS-D starts).
- 3DFed go/no-go (auto-triggered if Phases 1+2 leave FLAME standing).
- Multi-seed escalation (auto-triggered if Day-0 seed-43 determinism
  check fails).

### Future cycles (post-Cycle-03)

- **Cycle 04 — Defense design.** Triggered only if Cycle 03 produces an
  attack that breaks FLAME. Design a new general defense informed by
  what the breaking attack reveals about FLAME's failure mode.
- **Cycle 05+ — Threat-model extensions.** Edge poison regime
  (rare-class sources); malicious-majority; real-time probing adaptive
  attacker.
- **Longer-term — Autonomous-driving migration.** ViT backbone
  (supervisor suggestion); multimodal / fusion-layer backdoors on a
  perception benchmark (BDD100K, nuScenes).

---

## Scientific Findings

### Cycle 01 (pre-audit codebase — exploratory, NOT authoritative)

Cycle 01 (Phase C / Phase D) was run on the pre-audit codebase. Its
findings (representation-space framework on pixel-trigger and
model-replacement; defense comparison incl. FedMedian breakdown point,
Krum vulnerability window, spectral-defense blind spot) **should not be
treated as authoritative**. The underlying experiments contained
reproducibility issues identified by the 2026-05-11 audit (see
`docs/audit_2026-05-11.md`). The findings remain in `docs/cycle01_docs/`
for historical reference but every empirical conclusion needs re-running
on the audit-fixed codebase before it can be cited.

### Cycle 02 Wave-1 (audit-fixed codebase — authoritative)

Full results: `fl_v2/docs/cycle_02/wave1_log.md`. Headline observations:

1. **FLAME drives ASR to 0.000** across all 3 static attacks (pixel,
   model_replacement, DBA) on our threat model. Implementation was
   audited 10/10 clean against the paper + reference impls
   (`zhmzm/FLAME`, `Guncuke/flame-Taming-backdoor`). Under Adam (vs the
   paper's SGD), the noise multiplier λ must be calibrated to 1e-6
   (1000× smaller than the paper) to avoid divergence — see
   `pyproject.toml` comment for the math.
2. **DBA is the strongest non-defended attack on durability** (final ASR
   0.80 at r60 vs r35 attack-end) and **completely evades FoolsGold** —
   sub-trigger groups produce low pairwise cosine between colluders,
   defeating the similarity-downweighting recipe by design.
3. **MultiKrum is bimodal**: totally defeats model_replacement (n/k
   scaling is an obvious outlier), partial defense against DBA (~50%
   peak / ~70% final ASR reduction), useless against pixel.
4. **NormClipping (`clip-norm: 100`) provides no defense** on this
   threat model — too loose to bite honest gradients. Dropped from
   future cycle matrices.
5. **Candidate gradient-space signal**: `cos2mean` (cosine angle between
   a client's update and the all-client mean direction) separates
   malicious from honest with Cohen's |d| ≥ 1.0 across all 3 attacks on
   no-defense runs. Stable across defenses for pixel (6.6% spread) and
   DBA (15.9%). This is descriptive evidence to inform Cycle-04 defense
   design — not a "law" in any strong sense.

### Cycle 03 (in progress)

Reproducing 3-4 adaptive attacks from 2023-2025 literature to find one
that breaks FLAME's 0.000 baseline on our threat model. No findings yet.

---

## Current Platform Status

### Federation engine
- Flower latest Message API (ClientApp / ServerApp) on Ray simulation
  backend.
- 50 simulated clients on a single A40 GPU per SLURM job; per-job
  `FLWR_HOME`, per-job Ray temp dir, per-job Ray ports (`run_alvis.sh`).
- **Bit-deterministic**: same-seed runs are byte-identical (DataLoader
  test in `tests/test_dataloader_determinism.py`; FLAME's seeded noise
  generator verified across re-runs).

### Datasets / model
- GTSRB-43 with non-IID Dirichlet partitioning, `partition-seed`
  decoupled from `seed` (closes audit R-001).
- ResNet18 trained from scratch (all current experiments). CNN baseline
  retained but not used.
- Client-local train / val split.

### Implemented attacks (`src/fl_v2/attacks_defenses/attacks/`)
- `label_flipping` — data-poisoning baseline.
- `pixel_backdoor` — fixed-position pixel trigger; supports DBA's
  scattered-bars and corner-grid sub-trigger geometry.
- `model_replacement` — n/k update scaling (Bagdasaryan et al. 2020).
- `dba` — Distributed Backdoor Attack, paper-faithful scattered-bars
  pattern (Xie et al. 2020).
- `neurotoxin` — local-clean-gradient proxy + top-k coord projection
  (Zhang et al. 2022). Documented deviation: stateless local proxy
  instead of cross-round cached global, for bit-determinism.

### Implemented defenses (`src/fl_v2/strategy/` + `attacks_defenses/defenses/`)
- `NormTrackingFedAvg` — FedAvg + per-round gradient-space metric
  logging (`L2 norm`, `cosine_to_mean`, `pairwise_cosine`,
  `topk_energy_frac`).
- `NormClippedFedAvg` — L2 norm clipping.
- `FedMedian`, `FedTrimmedAvg`, `Bulyan` — robust aggregation, custom
  implementations with norm logging (Flower built-ins had dtype bugs).
- `CapturedKrum`, `CapturedMultiKrum` — wrappers around Flower built-ins
  (currently no NormTracking; Cycle-03 WS-A composes MultiKrum into
  NormTracking).
- `FoolsGoldFedAvg` — output-layer cosine on cumulative client histories
  (Fung et al. 2020). Head-only as the paper specifies.
- `FlameFedAvg` — HDBSCAN clustering + median-norm clipping + Gaussian
  noise (Nguyen et al. 2022). λ=1e-6 optimizer-aware calibration for
  Adam (1000× smaller than the paper's SGD-calibrated value).

### Server-side eval
- Single-pass evaluation in `training/server_eval.py`: clean accuracy +
  target-class clean accuracy + ASR.
- Cycle-03 WS-A adds: per-class ASR breakdown, exact
  trigger-attributable ASR (extra no-trigger forward pass).

### Analysis surfaces
- **Wandb live logging** (per-round server + client metrics) — replaces
  the retired training-curves analysis pipeline.
- **Gradient-space metrics** (`norm_log.json` per cell) — primary
  analysis surface for Cycle 02+.
- **Representation-space framework** (`analysis/`, t-SNE,
  `framework_metrics`) — implemented but used only for Cycle 01
  findings; not the current primary analysis tool.

### Cycle 03 in-progress additions
Per the active plan: small-LR / LP / A3FL adaptive attack modules,
per-class ASR + trigger-attributable ASR logging, MultiKrum
NormTracking composition refactor.

---

## Architecture Expectations

Preserve the modular project structure:

- `data/` — dataset loading, partitioning, transforms.
- `models/` — model definitions.
- `training/` — local training / evaluation logic.
- `client_app.py` — client-side FL behavior.
- `server_app.py` — server-side orchestration.
- `strategy/` — custom FL strategies and aggregation control.
- `attacks_defenses/` — attack and defense implementations.
- `utils/` — generic helpers.

Do not collapse multiple responsibilities into one file unless there is
a strong reason.

---

## Attack / Defense Design Expectations

### Attack module pattern
Three integration shapes are in use, follow whichever fits:
- **Data-poisoning attacks** (pixel, DBA, label-flip): hook into the
  dataset wrapper in `client_app.py::_load_client_data`. Gate on
  `is_malicious && in_attack_window`. No grad hook.
- **Gradient-mask attacks** (Neurotoxin; Cycle-03 LP): module exposes a
  `mask_fn(named_params, ...) → dict[id, bool]`. Hook into
  `train.py:141` between `loss.backward()` and `optimizer.step()`.
- **State-dict-transform attacks** (model_replacement): apply
  post-training in `client_app.py` before the reply state-dict is
  built.

Every attack module gets a YAML knob and a **null config** that
reproduces the no-attack baseline bit-identically.

### Defense module pattern
Defenses live in `strategy/`. All should inherit from
`NormTrackingFedAvg` (or compose its logging) so they automatically log
gradient-space metrics. Stateful defenses (FoolsGold's cumulative
history) keep state on the strategy instance — pure accumulation, no
RNG. Defenses that draw RNG (FLAME's noise) use a generator seeded off
the run seed × server round.

---

## Collaboration Rules for Claude

When assisting with this repository:

1. First analyze the existing codebase before proposing changes.
2. Prefer minimal, explicit, incremental changes.
3. Do not introduce broad refactors unless necessary.
4. Keep the clean baseline runnable at all times.
5. Preserve compatibility with the latest Flower Message API. Do not
   revert to legacy `NumPyClient` style.
6. Preserve reproducibility of data partitioning and experiment setup.
   Bit-determinism is sacred — any new RNG must be seeded via
   `derive_seed` (`src/fl_v2/utils/runtime.py:43-54`).
7. Separate immediate engineering fixes, experiment-platform
   improvements, and thesis-facing research suggestions.
8. When suggesting a new feature, explain how it helps either the
   current benchmark platform or the long-term thesis direction.
9. For research-facing suggestions, do not treat current hypotheses
   as proven conclusions. The Cycle-02 Wave-1 log is the current
   authoritative source for empirical claims; Cycle-01 results are
   exploratory.
10. When discussing attacks or defenses, distinguish between heuristic
    baselines, optimization-based methods, server-side defenses, and
    client-side defenses.
11. Client-side defense ideas (TTA-inspired or otherwise) are
    explicitly deferred until Cycle 03 produces an attack that breaks
    server-side gradient-space defenses (FLAME). Until then they are
    background research, not active work.

---

## Experiment Principles

- Always keep a clean baseline for comparison.
- Attack modules are evaluated against clean training behavior.
- Defense modules are evaluated on both utility (clean acc) and
  robustness (ASR).
- Every new attack module must have a **null config** that reproduces
  the no-attack baseline bit-identically.
- Explicit logging, saved results, reproducible configurations.
- Global RNG seeding (`random`, `numpy`, `torch`) at server startup
  using the configured `seed`. Per-round / per-client seeds via
  `derive_seed`.
- New modules should be easy to reuse across future datasets and
  tasks.
- Avoid tightly coupled code that would make future migration
  difficult.
- For non-trivial tasks, first provide:
  1. codebase understanding,
  2. minimal change plan,
  3. risks / edge cases,
  4. implementation proposal.

---

## Workflow Constraints

### Project location
The project lives on Mimer at
`/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/thesis_workspace/fl_weather_project/`
(migrated from Cephyr on 2026-04-27 — every script and doc references
the Mimer root). Datasets, outputs, SLURM logs, and the venv are all on
Mimer; nothing in a SLURM job reads from Cephyr.

### Day-to-day workflow
Every experiment is submitted to Alvis via `submit_experiment.sh`; see
[`docs/scripts_guide.md`](docs/scripts_guide.md) for the full pipeline.
ResNet18 + Flower + Ray is too heavy for login-node execution — there is
no local workflow. Login-node Python is for analysis and plotting only.
(When the project later migrates to a ViT backbone the situation only
gets worse, not better.)

### Flower federation config
The `local-simulation-gpu` federation (50 supernodes, 0.10 GPU per
supernode) is version-controlled at
[`configs/flwr_config.toml`](configs/flwr_config.toml). `run_alvis.sh`
creates a per-job `FLWR_HOME=/tmp/flwr_${SLURM_JOB_ID}` and copies that
file into `$FLWR_HOME/config.toml`. On exit the per-job `/tmp` directory
is removed. Do NOT remove the `/tmp` indirection — it isolates
concurrent jobs from each other.

### Same-node startup races (closed)
`run_alvis.sh` waits on **two** SLURM startup events before allowing
`flwr run`: (1) SuperLink Control API binds, (2) SuperLink emits
`Starting Flower SuperExec` and then sleeps 30 s for its subprocess
factory to be ready. Both waits abort loudly on absence — no more
silent exit-0 same-node races (the failure mode that ate ~25% of
original Wave-1 jobs).

### Output layout (cycle-aware)
New experiments write to
`/mimer/NOBACKUP/groups/naiss2024-22-991/gaohui/fl_outputs/gtsrb/experiments/<cycle>/<phase>/<exp>_r<rounds>_seed<seed>/`.

- Top-level `gtsrb/` (the old `gtsrb_v2/` `_v2` suffix was a Flower-v1
  vs v2 holdover; with v1 retired the suffix is noise).
- `experiments/<organizer>/`: `legacy`, `cycle_01`, `cycle_02`,
  `cycle_03`, …
- `<phase>/`: Cycle 02 used `mechanism`. Cycle 03 will use
  `phase0_logging`, `phase1_smalllr`, `phase2_lp`, `phase3_a3fl`, etc.
- `<exp>_r<rounds>_seed<seed>/` — unchanged.

YAMLs control routing via two keys: `cycle` (e.g. `cycle_02`) and
`phase` (e.g. `mechanism`). YAMLs that leave both blank fall back to the
flat `<base>/<exp>...` layout — preserves historical
`gtsrb_v2/phaseC_v2/...` and `gtsrb_v2/phaseD/...` Cycle-01 archives.
**The existing `gtsrb_v2/` tree is not migrated**; it's a frozen Cycle 01
archive.

### Wandb conventions
Every experiment with `wandb-enabled: true` (default) gets a wandb run:
- **Project:** `gtsrb-{cycle.replace('_','-')}` → `gtsrb-cycle-02`,
  `gtsrb-cycle-03`. Override with `wandb-project`.
- **Group:** strip trailing `<n>mal` and defense tokens from
  `experiment-name`. Override with `wandb-group`.
- **Run name:** `{experiment-name}_seed{seed}`.
- **Auto tags:** `cycle`, `phase`, `model-type`, `attack:<type>`,
  `defense:<type>`, `<n>mal`, `seed<n>`. Extras via `wandb-tags`.
- **Per-round metrics:** `server/{test_loss, test_accuracy, asr,
  target_class_clean_accuracy}`, `client/{train_loss, train_accuracy,
  val_loss, val_accuracy}` (weighted average across selected clients).
- **Auth:** `wandb login` once on alvis1 → `~/.netrc` → SLURM jobs pick
  it up via `--export=ALL`.

See [`docs/wandb_setup.md`](docs/wandb_setup.md) for setup; section 2.3
of [`docs/scripts_guide.md`](docs/scripts_guide.md) for the day-to-day
flow.

### Venv reproduction
The venv at `.venv/` (Mimer) uses exact pinned versions captured from
the original Cephyr venv. Lockfile:
[`requirements.lock.txt`](../requirements.lock.txt). Recreate with:

```bash
module purge && module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade "pip==26.0.1"
pip install --no-cache-dir -r requirements.lock.txt
pip install --no-cache-dir -e fl_v2 --no-deps
```

`torch`, `numpy`, and `scipy` are inherited from the PyTorch module, not
pinned in the lockfile — the `--system-site-packages` flag enables this.
Do not pin them in the lockfile or pip will reinstall them and shadow
the module versions, breaking CUDA.

`wandb` (and transitives) is pinned in the lockfile. After recreating
the venv, run `wandb login` once on alvis1 to write the API key into
`~/.netrc`.

---

## Things To Avoid

1. Do not rewrite the project into a monolithic structure.
2. Do not silently replace the latest Flower API design with older
   patterns.
3. Do not mix attack logic and defense logic into unrelated files.
4. Do not remove reproducibility-related controls without reason.
5. Do not cite Cycle-01 representation-space findings as authoritative
   evidence (they were on the pre-audit codebase).
6. Do not over-engineer for the final AD setting too early.
7. Do not assume the current GTSRB benchmark is the final scientific
   target.

---

## Preferred Planning Style

For complex tasks, first produce a plan based on the current codebase.
The plan should explicitly include:

1. which files are relevant,
2. what is already implemented,
3. what should be minimally changed,
4. possible edge cases or risks,
5. how the proposed change supports either the current FL
   experimentation platform or the long-term thesis direction.

Do not jump directly into broad implementation proposals before
understanding the existing code structure.
