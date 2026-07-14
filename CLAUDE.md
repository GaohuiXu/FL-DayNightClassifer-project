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

S07 clean engineering is closed at
`a2fc15e64898910b51b56b4b25c8579f459423bc`. This proves a bounded clean FedAvg
construction, one FP32 optimizer update for each current C/L/F mode, and a
worker-0/2 first-batch equality check. It is not full training, detector
capability, performance, precision freeze, Protocol-A/B, attack, defense, or
scientific evidence.

The next milestones are:

1. S08: qualify the current six-task model's FP32, full FP16 AMP, and FP16 AMP
   with SECOND/spconv FP32 island behavior;
2. S09: after S08, establish production-shaped 100/conditional-1000-step
   performance, memory, DataLoader-worker, and single-GH200 readiness;
3. redefine S10-S12 only from reviewed S08/S09 evidence.

The Arrhenius environment supports FP32 and FP16 AMP as runtime mechanisms, but
the current scientific precision policy is not selected. Current evidence shows
FP16 scale 1 recovers camera on one mini batch but leaves direct nonfinite
SECOND LiDAR/fusion gradients. Direct sparse BF16 is unsupported.

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
There is no automatic retry or spare-GPU expansion. Follow the complete compute,
Git, data, precision, and scientific guardrails in `AGENTS.md`.
