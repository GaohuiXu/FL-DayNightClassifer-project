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

S07-S09 are closed. S10 is active on `codex/s10-cl-model-recipe` from audited
base `a080d49c1c22de20ccb5b1353d4922c7df14a729`. Accepted S10 evidence through
O-142 remains under `fl_v3/usenix27_orchestra/handoffs/S10/`: STOP-A's
train-only split/evaluator is reusable; STOP-B is `INCONCLUSIVE`; C1-A
localized the observed large LiDAR-stem gradient to the current tiny-group GN
path; bounded fusion proxy runs did not establish production capability or an
advantage over the historical Alvis model.

O-143 replaces S10's active six-stop execution order. The new order is:

1. qualify camera and LiDAR branches independently and freeze a defensible recipe
   for each;
2. staged fusion from qualified branch checkpoints, then aligned capability and
   fusion-contribution evaluation;
3. GH200 profiling/optimization only after capability passes.

Current-A2 and the old C→D→E→F path are paused. No compute is currently
authorized. S11+ remains pending.

## Collaboration

Persistent S00 is the default planner and implementer. For S10, O-143 replaces
per-job immutable/no-retry/multi-document/reviewer mechanics with phase-level
control. The owner approves a phase objective, candidates, data/metrics/seeds,
aggregate GPU-hours, submission cap and stop conditions once; inside that
approved envelope S00 may repair output-neutral runner/test/checkpoint/logging
defects and resubmit without repeated questions.

S10 keeps one compact active `HANDOFF.md` and one `RUN_REQUEST.md` job ledger.
Existing `RESULTS.md` and `REVIEW.md` are historical archives. Canonical docs
change only at phase start, material scientific amendment or phase close.
Independent review is reserved for data/metric changes, branch recipe freezes,
and the final staged-fusion/full result.

## Runtime and data

Arrhenius GH200 is the active runtime. Use the persistent environment through
`fl_v3/scripts/arrhenius_env.sh`; do not rebuild it for every job. The login node
is x86_64 and the environment is aarch64, so dependency-backed CUDA/spconv checks
run through an owner-authorized Slurm job.

The shared nuScenes trainval data is ZIP-backed through the dataset module. S09
STOP-1 Job `441191` materialized and reviewed the exact train/val `t1.v2`,
ten-sweep production caches; downstream use must bind the hashes in the S09
terminal results. Historical `t1.v1` caches are forbidden production inputs.
Mini is engineering only and cannot support mAP/NDS, fusion-gain, attack,
defense, or paper claims.

## Permissions

Planning or implementation is not compute authority. For S10, a future explicit
phase approval may authorize autonomous derived submissions and output-neutral
engineering remediation within its aggregate resource/submission cap. Changes to
model math, data ownership, recipe candidate space, evaluator/metric, seeds,
scientific claims or aggregate resources return to the owner. Preserve raw
outputs and minimum run provenance: Git SHA, resolved-config hash, split, seed,
command, resources, output, checkpoint and metric hashes. Commit, merge, push,
upload and publication remain separately owner-gated. Follow `AGENTS.md` for
the complete rules.
