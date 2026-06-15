# T0 — SPEC: new branch, `fl_v3` scaffold, determinism + carry-over + viz harness

Build-session copy, filled from `fl_v2/docs/cycle_04/collab/SPEC_TEMPLATE.md`.
Contract: `fl_v2/docs/cycle_04/tasks/T0_SPEC.md`. Plan: task **T0** in
`fl_v2/docs/roadmap/cycle_04_fusion_layer_backdoors.md`.

## 1. Scientific intent

Stand up a fresh, bit-deterministic `fl_v3/` platform skeleton and **re-implement
(not port)** the model/dataset-agnostic machinery — the determinism harness, the
defense family + gradient-space metrics, the partition logic, the task-agnostic
FL skeleton, and the viz writer — each validated against `fl_v2` as an
*implementation* oracle, so every later task (T1 nuScenes data … T7 matrix) builds
on a clean, reproducible, auditable base. No nuScenes, no AD model, no attack yet.

## 2. Scope

**In scope (delivered):**
- New long-term branch **`v3-ad-perception`** off `v2-new-api`; `fl_v2/` frozen
  (oracle only, untouched).
- `fl_v3/` skeleton: `src/fl_v3/{utils,data,models,training,strategy,engine,viz}/`,
  `client_app.py`, `server_app.py`, `tests/`, `configs/`, `docs/`, `collab/`,
  `scripts/`, `pyproject.toml`.
- **Portable, ARM-rebuildable venv** (pinned; **no mmdet3d/mmcv/spconv**). Built
  on Alvis x86; steps + ARM rebuild note in `docs/env.md`. Manifest =
  `pyproject.toml` + `requirements.txt` + `constraints.txt`; closure =
  `requirements.lock.txt`.
- **Determinism harness** `utils/runtime.py`: `derive_seed` (SHA-256, byte-identical
  to oracle), `seeded_worker_init`, `enforce_determinism()`
  (`use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  `cudnn.deterministic`, `benchmark=False`), `seed_everything`.
- **Task-agnostic FL skeleton:** `training/tasks.py` (`Task` interface + registry,
  criterion config-injected), task-agnostic `client_app.py`/`server_app.py`
  (no hardcoded loss / num-classes), sequential `num-gpus=1.0` config carried in
  `configs/flwr_config.toml`.
- **Carry-over (re-implemented + oracle-checked):** partition logic
  (`data/partition.py`); defense family as framework-free numerical **cores**
  (`strategy/defenses/`): FedAvg, NormClip, FLAME, FoolsGold, MultiKrum,
  FedMedian; gradient-space metrics (`strategy/gradient_metrics.py`); thin Flower
  wrappers (`strategy/flower_strategies.py`).
- **`viz/` writer scaffold** (`viz/writer.py`): deterministic V1–V6 tree + manifest.
- **In-process FL round runner** (`engine/local_runner.py`): login-node-safe
  clean-round driver (no Ray).
- **`fl_v3/collab/`** with templates + empty findings log + this SPEC.

**Out of scope (later tasks):** nuScenes loader (T1), the BEVFusion model (T2),
the real Ray FedAvg run (T3), eval/ASR (T4), attacks (T5), defense benchmark
behavior + assumption cards (T6), the matrix (T7).

**Files created:** all of `fl_v3/**` (see `git status`). `fl_v2/**` untouched.

## 3. Invariants (must hold; Codex checks each)

- **Bit-determinism:** all RNG via `derive_seed`/`seed_everything`;
  `enforce_determinism(strict=True)` makes a banned op (atomic scatter,
  `grid_sample` backward, non-stable sort/topk, flash-attn) RAISE. Tests:
  same-seed TinyMLP → identical weights; same-seed in-process FL round → identical
  `agg_checksum` AND eval; MultiKrum uses `kind="stable"` argsort; FLAME noise from
  a seeded `default_rng`.
- **Oracle parity (implementation equivalence only):** re-implemented FLAME +
  FoolsGold (and additionally NormClip, FedMedian, gradient-metrics, partition)
  reproduce the **fl_v2** decision on a saved fixture of synthetic update vectors
  (`tests/fixtures/oracle_*`). FLAME: admitted set + clip bound + coefs + the
  **bit-identical aggregated arrays** (incl. seeded noise). Parity does NOT
  certify AD-domain validity.
- **No classification coupling:** the skeleton has no hardcoded loss / num-classes
  / single-logits assumption — asserted by AST inspection + an end-to-end MSE
  regression round (`tests/test_task_agnostic.py`).
- **Null/degenerate handling:** FLAME `n<4` → admit all; no-cluster → admit all;
  MultiKrum invalid `(n,f,m)` → `valid=False` (cell NA, not forced).
- **Threat-model / metric knobs honored:** `ρ`/`m_r` vs defender-assumed `f_r`
  are independent (MultiKrum `num_malicious` is `f_r`, never conflated with the
  actual malicious count); FLAME `λ` (`flame-noise-multiplier`) is a config knob,
  not hardcoded (the Adam-vs-SGD calibration finding).

## 4. Reference (ground truth for the review)

- Determinism oracle: `fl_v2/src/fl_v2/utils/runtime.py` (`derive_seed` ~L43–54,
  `seeded_worker_init`).
- FLAME: `fl_v2/src/fl_v2/strategy/flame.py` (`_cluster_admitted`: HDBSCAN
  `min_cluster_size=n//2+1`, `min_samples=1`, `metric="precomputed"`, `eom`,
  `allow_single_cluster`; clip to median norm; noise std `= λ·S_t`, no `sqrt(d)`;
  noise RNG `default_rng((seed&0xFFFFFFFF)*1_000_003 + round)`) + paper Nguyen
  et al. USENIX-Sec'22 Eq. 7.
- FoolsGold: `fl_v2/src/fl_v2/strategy/foolsgold.py` (`_foolsgold_weights`:
  pardoning + logit rescale `+0.5`) + Fung et al. 2020; head-slice `HEAD_IDX=-2`.
- NormClip / metrics: `fl_v2/src/fl_v2/attacks_defenses/defenses/norm_clipping.py`.
- FedMedian: `fl_v2/src/fl_v2/strategy/fed_median.py` (coordinate-wise `np.median`
  of client params) + Yin et al. 2018.
- MultiKrum: Blanchard et al. NeurIPS 2017 (fl_v2 used Flower's built-in → no
  fl_v2 source oracle; fl_v3 validates against a hand-computed textbook fixture).
- Partition: `fl_v2/src/fl_v2/data/partition.py` (`dirichlet_partition` RNG order).
- Sequential exec: `fl_v2/configs/flwr_config.toml` (`num-gpus=1.0`).
- Oracle snapshot generator: `tests/fixtures/make_oracle_fixtures.py` (imports the
  actual fl_v2 pure functions).

## 5. Scientific failure modes to check (point Codex here)

- A re-implemented defense that *looks* right but diverges from the oracle on
  tie-breaks / ordering / the `min_cluster_size` boundary / median definition /
  the FoolsGold pardoning loop. (Mitigated by bit-identical aggregated-array
  parity, but scrutinize the FLAME `flat` float32 construction order and the
  noise draw order vs the oracle.)
- A constant transplanted from a different optimizer/scale (the FLAME `λ`
  Adam-vs-SGD bug): confirm `λ` is config-driven, never baked in.
- Hidden non-determinism: unseeded DataLoader worker, non-stable sort/topk,
  accumulation-order dependence. (Check `multi_krum.krum_scores` stable sort and
  the partition-id sort in `flower_strategies`/`local_runner`.)
- A leaked classification assumption in the "task-agnostic" interface
  (e.g. an accuracy metric, a num-classes read, a CE loss default).
- MultiKrum validity boundary (`n >= 2f+3`) and the actual-vs-assumed-malicious
  distinction.

## 6. GATE (objective pass criteria)

- [x] `v3-ad-perception` branch exists; `fl_v3/` skeleton in place; `fl_v2/`
  untouched.
- [x] venv builds from the pinned manifest; `docs/env.md` records steps + ARM
  rebuild note. (Verified: numpy held at 1.26.4, torch 2.7.1 CUDA, flwr 1.27.0,
  HDBSCAN OK, nuScenes data+DetectionEval import with no descartes.)
- [x] determinism smoke green (same-seed → bit-identical TinyMLP weights AND
  identical in-process FL-round `agg_checksum`).
- [x] re-implemented FLAME (+ FoolsGold, NormClip, FedMedian, gradient-metrics,
  partition) reproduce the `fl_v2` decision on a saved fixture.
- [x] FL skeleton imports + runs a trivial clean round on a dummy task with no
  hardcoded loss (in-process runner; real Ray run is T3, via SLURM).
- [x] `fl_v3/collab/` established (templates + empty findings log).

**Evidence:** `pytest fl_v3/tests` → 43 passed (see §7 / the build summary).

## 7. Self-review — what I'm least sure about (attack these hardest)

These incorporate an internal adversarial verification sweep (8 module skeptics +
a completeness critic); FLAME / NormClip+FedMedian+metrics / Partition came back
clean, the actionable items were fixed (see `collab/findings_log.md`), and the
items below are the conscious decisions + scope boundaries that most warrant a
second opinion.

1. **Unified fp64 aggregation core vs the oracle's fp32 clean path (the headline
   parity-scope decision).** fl_v3 routes EVERY defense — incl. plain FedAvg and
   NormClip — through one fp64 `aggregate_weighted_updates` core. The fl_v2 oracle
   computes FLAME + FoolsGold via that SAME core (so their parity is **bit-identical**,
   and that is what the GATE requires), but its *clean*-FedAvg / NormClip final
   aggregation delegates to Flower's fp32 `aggregate_arrayrecords`. So fl_v3's
   clean/clip aggregation is algebraically equal but NOT bit-identical to the oracle
   (fp64 vs fp32). I chose to keep the unified higher-precision core and claim
   bit-parity only for FLAME/FoolsGold; clean/clip agreement with Flower's fp32 is a
   tolerance-level T3 check. **Is that the right call, or should clean-FedAvg/NormClip
   match Flower's fp32 weighting bit-for-bit?** (`aggregation_core.py` docstring,
   `defenses/fedavg.py`.)
2. **Strict determinism is an INTENTIONAL divergence from the oracle.** fl_v2 used
   `use_deterministic_algorithms(True, warn_only=True)`; fl_v3 defaults to strict
   (RAISE) so a banned op in the AD model (T2) cannot slip through as a silent warn.
   A run that warned-and-finished under fl_v2 could crash under fl_v3. Intended — but
   confirm strict-by-default is desired for the platform (config `determinism-strict`
   can relax it for bring-up). The T0 TinyMLP exercises no banned op, so the
   strict-raise path is unexercised until T2.
3. **The "trivial clean round" GATE interpretation.** Satisfied via the in-process
   `local_runner` (login-node-safe), NOT a live `flwr run` (Ray is too heavy for the
   login node; the real Ray path is the T3 milestone). The Flower wrappers + apps are
   validated to *import + construct* and route through the same cores; their
   end-to-end Ray behaviour, metric-aggregation seam (`train_metrics_aggr_fn`
   weighting; dropped-clients-in-metrics-but-not-params), and the ArrayRecord-key /
   global-order identity are T3 checks. Acceptable T0 gate, or add a 2-supernode CPU
   `flwr run` smoke now?
4. **MultiKrum has no fl_v2 source oracle** (fl_v2 used Flower's built-in). Validated
   on a hand-computed textbook fixture + validity gate, NOT cross-checked numerically
   against Flower's built-in. Also: `num-malicious-nodes` is consumed as Krum's
   *assumed* `f_r` only (no attack exists yet); once attacks land (T4/T5) a separate
   knob must keep assumed-`f_r` independent of actual-`m_r`.
5. **FoolsGold head-slice index `-2`** (carried from fl_v2's ResNet `fc.weight`) is
   now a `head_index` param threaded on both the Flower path and the local runner.
   For AD it must become the fusion/detection-head indicative slice (T6) — confirm the
   seam won't silently mis-slice the BEVFusion head.
