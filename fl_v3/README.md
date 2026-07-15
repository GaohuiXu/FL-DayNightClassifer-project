# fl_v3 — clean federated multimodal AD perception

`fl_v3/` is the active camera–LiDAR nuScenes platform. The current priority is
to validate and freeze a strong centralized detector, then preserve the same
clean model/data/runtime foundation through federated training and adaptation.
`fl_v2/` is frozen historical code and is not an active dependency.

- [Active Orchestra](usenix27_orchestra/ORCHESTRA.md)
- [Milestone contracts](usenix27_orchestra/SESSIONS.md)
- [Active milestone envelopes](usenix27_orchestra/KICKOFFS.md)
- [Arrhenius environment contract](docs/env.md)
- [Roadmap index](docs/roadmap/INDEX.md)

## Current foundation

- Arrhenius GH200 runtime with PyTorch CUDA 12.8 and source-built
  `cumm`/`spconv`.
- nuScenes directory and read-only stored-ZIP data paths with depth-bound
  `t1.v2` cache provenance.
- Reviewed camera, sparse LiDAR, fusion, multi-task CenterHead, deterministic
  decode/NMS, and official nuScenes `DetectionEval` paths.
- Resolved config, explicit FP32/FP16 runtime mechanisms, checkpoint/resume,
  runtime dependency identity, and clean evaluation provenance. S08 accepts
  global FP16 for camera/dense-pillar, global FP16 with an explicit SECOND/spconv
  FP32 island for sparse LiDAR/fusion, and uniform FP32 as reference/fallback.
- One clean FedAvg aggregation path with deterministic client identity/order,
  num-example FP32 weighting, deterministic sampling, FedOpt, server EMA, and
  trainable-only state transfer.

Mini and synthetic runs are engineering evidence only. Scientific claims
require owner-approved trainval-scale protocols and immutable run manifests.

S07 clean engineering, S08 precision qualification, and S09 full-pipeline
engineering readiness are closed and integrated through `351b7a0`. S09 proves
bounded single-seed lifecycle/performance health, not convergence, mAP/NDS,
recipe selection, or full GH200 utilization. The next milestone is not yet an
executable plan: a fresh Ultra-reasoning S00 will research S10 on
`codex/s10-cl-model-recipe`. Only S10's work definition is accepted—centralized
model numerical/architectural health, production recipe selection, and final-
architecture GH200 optimization. Exact stops, full-run placement, and S11+ remain
pending owner review.

## Layout

```text
src/fl_v3/
  config/                    resolved production configuration
  data/nuscenes/             directory/ZIP data, cache, partition, transforms
  models/fusion/             camera, LiDAR, fusion, head, decode/NMS
  training/                  task interface, loop, checkpoint/runtime state
  strategy/                  clean FedAvg, deterministic sampling, FedOpt
  eval/                      clean provenance and official DetectionEval
  engine/local_runner.py     in-process clean FedAvg regression runner
  client_app.py              task-agnostic Flower client
  server_app.py              clean Flower server
tests/                       focused unit and integration regressions
usenix27_orchestra/          active plans, contracts, handoffs, and reviews
collab/                      read-only historical evidence
```

## Verification

Dependency-backed tests run inside an owner-authorized Arrhenius allocation:

```bash
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env
python -m pytest fl_v3/tests -q
```

Login-node-safe checks include source compilation, JSON/TOML parsing, shell
syntax, clean FedAvg arithmetic tests where dependencies are available, and
`git diff --check`. See [docs/env.md](docs/env.md) for the runtime boundary.
