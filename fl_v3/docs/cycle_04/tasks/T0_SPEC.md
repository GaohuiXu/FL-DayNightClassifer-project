# T0 — SPEC: new branch, `fl_v3` scaffold, determinism + carry-over + viz harness

Plan: `../../roadmap/cycle_04_fusion_layer_backdoors.md` (task **T0**). This is the contract for the
first build session. Fill the build-session copy at `fl_v3/collab/T0/SPEC.md` from the template.

## 1. Scientific intent
Stand up a fresh, bit-deterministic `fl_v3/` platform skeleton and **re-implement** (not port) the
model/dataset-agnostic machinery, each validated against `fl_v2` as an *implementation* oracle, so all
later tasks build on a clean, reproducible, auditable base. No nuScenes, no model, no attack yet.

## 2. Scope
**In scope:**
- New long-term branch **`v3-ad-perception`** off `v2-new-api`. `fl_v2/` stays frozen (oracle only).
- `fl_v3/` skeleton: `src/fl_v3/{utils,data,models,training,strategy,attacks_defenses,viz}/`,
  `tests/`, `configs/`, `docs/`, `collab/`, `pyproject.toml`.
- **Portable, ARM-rebuildable venv manifest** (pinned, **no mmdet3d/mmcv/spconv**; torch + numpy +
  nuscenes-devkit + scikit-learn + hdbscan + flwr + ray + wandb). Build the x86 venv on Alvis; record
  the exact module-load + pip steps in `fl_v3/docs/env.md` with an **ARM/Arrhenius rebuild note**.
- **Determinism harness** (`utils/runtime.py` analog): `derive_seed`, `seeded_worker_init`, and a
  `enforce_determinism()` that sets `use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG`,
  `cudnn.deterministic`, disables benchmark.
- **Task-agnostic FL skeleton:** Flower ClientApp/ServerApp wiring with a **criterion/eval interface
  injected by config** (NO hardcoded `CrossEntropyLoss`); sequential `num-gpus=1.0` execution config
  carried from `fl_v2/configs/flwr_config.toml`.
- **Carry-over (re-implemented + oracle-checked):** the partition logic; the **defense family** as a
  clean set — `FedAvg`/NormTracking base, `FLAME`, `FoolsGold`, `MultiKrum`, `FedMedian`, `NormClip` —
  plus the gradient-space metric definitions.
- **`viz/` writer scaffold:** a deterministic artifact writer that lays down the
  `viz/{calibration,encoder,fusion,detection,attack,defense}/` tree under a run dir.
- **`fl_v3/collab/`** with the templates copied from `fl_v3/collab/` + an empty
  `findings_log.md`.

**Out of scope (later tasks):** nuScenes loader (T1), the fusion model (T2), FL training run (T3),
eval/ASR (T4), attacks (T5), the full defense benchmark behavior (T6).

## 3. Invariants
- **Bit-determinism:** a determinism smoke test (tiny model, two same-seed runs → bit-identical
  weights) passes; banned ops absent.
- **Oracle parity:** the re-implemented **FLAME** (and at least one other defense, e.g. FoolsGold)
  reproduces the `fl_v2` defense's admitted/dropped decision **on a saved fixture** of synthetic
  update vectors. *Implementation equivalence only.*
- **No classification coupling:** the FL skeleton has no hardcoded loss / num-classes / single-logits
  assumption.

## 4. Reference (oracle)
- Determinism: `fl_v2/src/fl_v2/utils/runtime.py` (`derive_seed` lines ~43–54; `seeded_worker_init`).
- FLAME: `fl_v2/src/fl_v2/strategy/flame.py` (HDBSCAN `min_cluster_size=N/2+1`, median-norm clip,
  λ-noise) + the paper (Nguyen et al., USENIX-Sec'22). Norm metrics:
  `fl_v2/src/fl_v2/strategy/norm_tracking_fedavg.py`.
- Other defenses: `fl_v2/src/fl_v2/strategy/{foolsgold,fed_median,norm_clipped_fedavg,krum_wrappers}.py`.
- Partition: `fl_v2/src/fl_v2/data/partition.py`. Sequential exec: `fl_v2/configs/flwr_config.toml`.
- Generate the oracle fixture by running the relevant `fl_v2` defense on a fixed synthetic update
  matrix and saving (input vectors, decision) — then assert `fl_v3` matches.

## 5. Scientific failure modes to check
- A re-implemented defense that *looks* right but diverges from the oracle on tie-breaks / ordering /
  the `min_cluster_size` boundary / median definition.
- Hidden non-determinism (unseeded DataLoader worker, `np.append` accumulation order, HDBSCAN tie
  order) — pin client/update ordering.
- A leaked classification assumption in the "task-agnostic" interface.

## 6. GATE
- [ ] `v3-ad-perception` branch exists; `fl_v3/` skeleton in place; `fl_v2/` untouched.
- [ ] venv builds from the pinned manifest; `fl_v3/docs/env.md` records steps + ARM rebuild note.
- [ ] determinism smoke test green (same-seed → bit-identical).
- [ ] re-implemented FLAME (+1 other defense) reproduces the `fl_v2` decision on a saved fixture.
- [ ] FL skeleton imports and runs a trivial clean round on a dummy task with no hardcoded loss.
- [ ] `fl_v3/collab/` established (templates + empty findings log).

## 7. Self-review — to be filled by the build session
