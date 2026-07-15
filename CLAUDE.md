# CLAUDE.md — project root

> Active instructions for Claude-family agents. Codex's binding parallel file is
> [`AGENTS.md`](AGENTS.md). When this file is less specific, follow `AGENTS.md`,
> [`fl_v3/docs/env.md`](fl_v3/docs/env.md), and the active Orchestra documents.

## Active project

`fl_v3/` is the active federated multimodal autonomous-driving perception project
on nuScenes. `fl_v2/` is frozen historical/oracle code and must not be modified
unless the owner explicitly asks.

The active coordination entry point is:

- [`fl_v3/usenix27_orchestra/ORCHESTRA.md`](fl_v3/usenix27_orchestra/ORCHESTRA.md)
- [`fl_v3/usenix27_orchestra/SESSIONS.md`](fl_v3/usenix27_orchestra/SESSIONS.md)
- [`fl_v3/usenix27_orchestra/KICKOFFS.md`](fl_v3/usenix27_orchestra/KICKOFFS.md)

`fl_v3/collab/**`, `fl_v3/docs/cycle_04/**`, and the old Cycle-04 attack/defense
roadmap are read-only historical evidence. They are not current implementation,
precision, runtime, session, or scientific authority. Do not recover legacy
T5/T6/T7 or old defense code/routes from them or from Git history.

## Current status

S07 clean engineering and S08 precision qualification are closed. The accepted
S08 close and fast-forward integration commit is
`28f79802c0868afa6290d74ae6aeb9d23c7d088f`. S08 supports global FP16 for camera/
dense-pillar, global FP16 with SECOND/spconv kept in an explicit FP32 island for
sparse LiDAR/fusion, and uniform FP32 as reference/fallback. It rejects full
sparse FP16 as the unified fusion-capable route within its bounded evidence.

The next milestones are:

1. S09: establish production-shaped 100/conditional-1000-step engineering
   performance, memory, DataLoader-worker, and single-GH200 readiness through the
   owner-approved four-stop envelope;
2. S10: separately select branch and training recipe, including any sparse
   normalization experiment or initialization/scheduler/EMA/augmentation change;
3. redefine later capability/protocol milestones only from reviewed S09/S10 evidence.

The accepted S08 precision policy is selected as stated above. The true unscaled
LiDAR gradients remain unusually large; S08 localized the practical FP16 failure
to sparse SECOND weight-gradient dynamic range but did not prove its architectural
cause. S09 observes engineering impact without changing normalization or recipe.
Direct sparse BF16 remains unsupported.

## Collaboration

Persistent S00 is the default planner and implementer for tightly connected
milestones. `Sxx` is an evidence namespace, not a mandatory fresh task/worktree.
Use bounded planning/research subagents before implementation when useful; do not
create parallel production chains by default. Independent review starts from an
immutable SHA and uses a reviewer subagent or, for high-risk/conflicted/runtime
reproduction work, a separate review worktree. Reviewers do not fix code.

Only S00 edits the three canonical Orchestra files. New handoffs, run requests,
results, and reviews go under `fl_v3/usenix27_orchestra/handoffs/Sxx/`.

## Runtime and data

Arrhenius GH200 is the active runtime. Use the persistent environment through
`fl_v3/scripts/arrhenius_env.sh`; do not rebuild it for every job. The login node
is x86_64 and the environment is aarch64, so dependency-backed CUDA/spconv checks
run through an owner-authorized Slurm job.

The shared nuScenes trainval data is ZIP-backed through the dataset module. Full
trainval `t1.v2` cache materialization still requires exact owner approval.
Historical `t1.v1` caches are forbidden production inputs. Mini is engineering
only and cannot support mAP/NDS, fusion-gain, attack, defense, or paper claims.

## Permissions

Planning or implementation is not permission to submit compute, commit, merge,
push, upload, or publish. Every material job uses an exact `RUN_REQUEST.md` bound
to immutable source/config/data/cells/resources/command/output/stop conditions.
There is no automatic retry or spare-GPU expansion. O-107's only exception is an
explicitly opted-in, owner-approved bounded mechanical remediation loop: each
derived job is frozen and recorded before submission, and any model/data/
precision/recipe/scientific or resource change returns to the owner. Follow the
complete compute, Git, data, precision, and scientific guardrails in `AGENTS.md`.
