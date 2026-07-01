# AGENTS.md - project root

Operating instructions for Codex and other non-Claude coding agents.
Claude's parallel file is [`CLAUDE.md`](CLAUDE.md), but this file is the
binding instruction file for Codex. If older project docs conflict with the
current Arrhenius status here, follow this file and `fl_v3/docs/env.md`, then
surface the conflict instead of silently inheriting stale rules.

`fl_v2/` keeps its own frozen `fl_v2/AGENTS.md`; ignore it unless deliberately
working inside `fl_v2/`.

## Current Project State

The active project is `fl_v3/`: a federated multimodal autonomous-driving
perception platform on nuScenes, with a backdoor attack/defense benchmark as
the research target. `fl_v2/` is frozen historical/oracle code. Do not modify
`fl_v2/` unless the user explicitly asks.

The active long-lived branch is `v3-ad-perception`. Older documents may describe
an Alvis-first workflow, pure-PyTorch LiDAR, strict bit-determinism, or a
Claude-builds/Codex-only-reviews loop. Those are historical unless explicitly
reconfirmed.

## Codex Role

Codex is not limited to after-the-fact review. Depending on the user's request,
Codex may:

- discuss and design architecture, experiment protocols, and migration plans;
- implement focused, reviewable code/docs/script changes;
- run local or Slurm-based smoke/profiling checks when appropriate;
- review Claude/Codex/user changes for scientific correctness;
- commit or merge only when the user explicitly authorizes it.

For large or scientifically risky changes, discuss the model, data path,
precision policy, metric, and acceptance criteria before editing. Once the
objective is clear, implement end to end rather than stopping at a proposal.

## Active Runtime: Arrhenius GH200

Arrhenius GH200 is the active runtime target. The validated environment is a
persistent conda/spconv environment under:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3
```

It is not stored in Git and is not recreated for every Slurm job. Normal jobs
activate it through `fl_v3/scripts/arrhenius_env.sh`. The login node is x86_64;
GH200 compute nodes are aarch64, so validate imports/training through Slurm
rather than treating login-node import failures as definitive.

All conda/venv/cache/build/data/output artifacts should live under
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui`, not under `$HOME`.

See `fl_v3/docs/env.md` and `fl_v3/collab/arrhenius_migration.md` for the
current versions, commands, Slurm templates, and smoke-test history.

## Dependency And Precision Policy

Current Arrhenius facts:

- PyTorch/CUDA works on GH200 with source-built `cumm`/`spconv`.
- `mmdet3d` and `mmcv` remain excluded as framework dependencies.
- `spconv`/`cumm` are allowed and active as the Arrhenius sparse LiDAR stack;
  do not apply old "no spconv" rules without re-checking the current design.
- Direct sparse `torch.bfloat16` is not supported by the validated cumm/spconv
  path.
- Supported sparse-path precisions are `fp32` and `fp16` AMP with GradScaler.

Near-term precision work should make this policy explicit in configs, trainers,
manifests, and Slurm launchers. Do not mix precision regimes in one comparison
without labeling it as an ablation.

Strict byte-identical determinism is a useful development regression tool, not
the default scientific claim criterion. For scientific claims, record hardware,
precision, seeds, data split, and run manifests; use multi-seed evidence when
results are eventually reported. Mini-data smoke is never scientific evidence.

## Data Status

Arrhenius does not currently have a confirmed shared full nuScenes path
analogous to Alvis/Mimer. A NAISS support request is pending. Do not assume
`/mimer/NOBACKUP/Datasets/NuScenes_v1.0` is mounted or accessible on Arrhenius.

The currently accessible mini dataset is:

```bash
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
```

Use mini for engineering only:

- import/data/cache smoke;
- sparse LiDAR branch correctness;
- precision unification and NaN/Inf checks;
- one-step/tiny-overfit tests;
- profiling and pipeline speed comparisons;
- cleanup validation.

Do not make scientific claims about attack viability, defense behavior, mAP/NDS
quality, ASR, or generalization from mini. Those require trainval-scale data or
a clearly justified fixed trainval subset.

## Scientific Guardrails

When designing or reviewing experiments, prioritize silent scientific failure
modes over style:

- data leakage and train/val/test contamination;
- client partition mistakes or non-comparable partitions across cells;
- coordinate-frame, yaw, class-map, unit, or calibration errors;
- sparse LiDAR tensor/index/voxel semantics and empty-input edge cases;
- precision mismatches that change training dynamics;
- ASR denominator/eligibility mistakes;
- comparing attack/defense cells with different clean baselines, seeds,
  participation regimes, or precision settings;
- claims resting on mini or smoke runs;
- defense evaluations where the corresponding undefended attack is not viable.

For defense carry-overs from `fl_v2`, oracle parity means implementation
equivalence only. It does not certify AD-domain validity.

## Working Style

Before editing:

- inspect relevant source, docs, configs, and `git status`;
- identify whether you are in the main worktree, a temporary Codex worktree, or
  an Arrhenius bring-up worktree;
- avoid touching unrelated dirty files;
- avoid reverting user/Claude changes unless explicitly asked.

When editing:

- keep changes scoped and reviewable;
- preserve existing project style;
- prefer structured config/data APIs over ad hoc parsing;
- update docs/scripts together when behavior changes;
- keep Arrhenius paths configurable via env/config where possible.

After editing:

- run the smallest meaningful verification available;
- for shell scripts, at least `bash -n`;
- for Python touched by the change, at least `py_compile` or focused tests;
- if Slurm/GPU/data prevents verification, say exactly what was not verified.

## Git And Branches

The active integration branch is `v3-ad-perception`. Create scoped `codex/...`
branches for independent Codex work unless the user asks to work directly on
the current branch. Commit, merge, or push only when the user explicitly asks.

Temporary bring-up branches/worktrees may be deleted after they are merged and
the user approves cleanup. Deleting a branch name must not remove persistent
environment/data artifacts under `/nobackup`.

## Review Mode

If the user asks for a review, switch to a code/science review stance:

- findings first, ordered by severity;
- cite exact files/lines;
- focus on correctness, scientific validity, metrics, data, precision,
  reproducibility, and missing tests;
- keep summaries secondary;
- state clearly when no issues are found and what residual risk remains.

The older `fl_v3/collab/codex_review_prompt.md` is a useful template for
review-only tasks, but it is not a global restriction on Codex implementation
work.
