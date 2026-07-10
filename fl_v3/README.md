# fl_v3 — Federated multimodal AD-perception platform (Cycle 04)

A fresh, bit-deterministic rewrite for **securing federated learning for
autonomous-driving perception**. `fl_v2/` is **frozen** and used only as an
*implementation oracle* (parity on fixtures — NOT scientific validity).

- **Durable plan:** [`docs/roadmap/cycle_04_fusion_layer_backdoors.md`](docs/roadmap/cycle_04_fusion_layer_backdoors.md)
- **Active roadmap:** [`docs/roadmap/INDEX.md`](docs/roadmap/INDEX.md)
- **Active Orchestra:** [`usenix27_orchestra/ORCHESTRA.md`](usenix27_orchestra/ORCHESTRA.md)
- **Session contracts + kickoffs:** [`usenix27_orchestra/SESSIONS.md`](usenix27_orchestra/SESSIONS.md) · [`usenix27_orchestra/KICKOFFS.md`](usenix27_orchestra/KICKOFFS.md)
- **Cycle-04 experimental-design history:** [`docs/cycle_04/README.md`](docs/cycle_04/README.md)
- **Decisions D1–D8:** [`docs/cycle_04/decisions.md`](docs/cycle_04/decisions.md)
- **Historical task evidence (read-only):** [`collab/`](collab/)

## Status: USENIX '27 CL/security sprint on Arrhenius

The current critical path is a strong centralized camera-LiDAR backbone followed by
the owner-approved Protocol-B federated tail-adaptation security study. New plans,
handoffs, reviews, and results live under `usenix27_orchestra/`; `collab/` is
read-only evidence from earlier Cycle-04 work.

What landed in T0 (see `collab/T0/SPEC.md` for the full GATE):

- **`v3-ad-perception`** branch; `fl_v3/` skeleton; `fl_v2/` untouched.
- **Arrhenius GH200 environment** with PyTorch cu128 and source-built
  cumm/spconv. See [`docs/env.md`](docs/env.md).
- **Determinism harness** (`utils/runtime.py`): `derive_seed`,
  `seeded_worker_init`, `enforce_determinism`, `seed_everything`.
- **Defense family as framework-free numerical cores** (`strategy/defenses/`):
  FedAvg, NormClip, FLAME, FoolsGold, MultiKrum, FedMedian — each validated
  against the `fl_v2` oracle on a saved fixture (implementation equivalence).
- **Task-agnostic FL skeleton** (`training/tasks.py`, `client_app.py`,
  `server_app.py`): no hardcoded loss / num-classes; the dummy task uses MSE.
- **In-process FL round runner** (`engine/local_runner.py`) — login-node-safe,
  no Ray; real distributed runs go through Arrhenius Slurm launchers.
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
usenix27_orchestra/           active plan, session contracts, kickoffs, handoffs/reviews
collab/                       read-only historical Cycle-04 evidence
```

## Run the tests

Run inside an Arrhenius GH200 Slurm allocation, not on the x86 login node:

```bash
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env
python -m pytest fl_v3/tests -q
# regenerate oracle fixtures (needs fl_v2 on path):
python fl_v3/tests/fixtures/make_oracle_fixtures.py
```

**Engineering smoke (mini / dummy) vs scientific result (trainval) is a hard
boundary** — T0 is all engineering smoke. No scientific claim is made until
trainval-scale runs from T5 onward.
