# fl_v3 — Federated multimodal AD-perception platform (Cycle 04)

A fresh, bit-deterministic rewrite for **securing federated learning for
autonomous-driving perception**. `fl_v2/` is **frozen** and used only as an
*implementation oracle* (parity on fixtures — NOT scientific validity).

- **Durable plan:** [`fl_v2/docs/roadmap/cycle_04_fusion_layer_backdoors.md`](../fl_v2/docs/roadmap/cycle_04_fusion_layer_backdoors.md)
- **Orchestration:** [`fl_v2/docs/cycle_04/README.md`](../fl_v2/docs/cycle_04/README.md)
- **Decisions D1–D8:** [`fl_v2/docs/cycle_04/decisions.md`](../fl_v2/docs/cycle_04/decisions.md)
- **This task's contract + self-review:** [`collab/T0/SPEC.md`](collab/T0/SPEC.md)

## Status: T0 — scaffold + determinism + carry-over + viz harness

What landed in T0 (see `collab/T0/SPEC.md` for the full GATE):

- **`v3-ad-perception`** branch; `fl_v3/` skeleton; `fl_v2/` untouched.
- **Portable, ARM-rebuildable venv** (no `mmdet3d`/`mmcv`/`spconv`). See
  [`docs/env.md`](docs/env.md).
- **Determinism harness** (`utils/runtime.py`): `derive_seed`,
  `seeded_worker_init`, `enforce_determinism`, `seed_everything`.
- **Defense family as framework-free numerical cores** (`strategy/defenses/`):
  FedAvg, NormClip, FLAME, FoolsGold, MultiKrum, FedMedian — each validated
  against the `fl_v2` oracle on a saved fixture (implementation equivalence).
- **Task-agnostic FL skeleton** (`training/tasks.py`, `client_app.py`,
  `server_app.py`): no hardcoded loss / num-classes; the dummy task uses MSE.
- **In-process FL round runner** (`engine/local_runner.py`) — login-node-safe,
  no Ray; the real Ray run is exercised at T3 via SLURM.
- **Viz writer scaffold** (`viz/writer.py`): the deterministic V1–V6 tree.

## Layout

```
src/fl_v3/
  utils/runtime.py            determinism harness
  data/partition.py           IID / Dirichlet partition (task-agnostic carry-over)
  models/dummy.py             TinyMLP smoke model (AD model = T2)
  training/tasks.py           Task registry (model+criterion+data+eval); dummy task
  training/loop.py            generic train/eval (criterion INJECTED)
  strategy/
    gradient_metrics.py       norms, cos-to-mean, pairwise cosine, top-k energy, L2 clip
    aggregation_core.py       weighted aggregation + coordinate median (pure numpy)
    defenses/                 FLAME / FoolsGold / NormClip / FedMedian / MultiKrum cores
    flower_strategies.py      thin Flower wrappers binding the cores
  engine/local_runner.py      in-process FL round (login-node-safe)
  viz/writer.py               deterministic V1–V6 artifact tree + manifest
  client_app.py, server_app.py   task-agnostic Flower apps
tests/                        determinism + oracle-parity + task-agnostic + smoke
collab/                       SPEC/REVIEW/templates + findings_log (the Codex loop)
```

## Run the tests

```bash
bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests -q
# regenerate oracle fixtures (needs fl_v2 on path):
bash fl_v3/scripts/run_in_venv.sh python fl_v3/tests/fixtures/make_oracle_fixtures.py
```

**Engineering smoke (mini / dummy) vs scientific result (trainval) is a hard
boundary** — T0 is all engineering smoke. No scientific claim is made until
trainval-scale runs from T5 onward.
