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
- [`fl_v3/usenix27_orchestra/handoffs/S10/PHASE_I_PLAN.md`](fl_v3/usenix27_orchestra/handoffs/S10/PHASE_I_PLAN.md)

`fl_v3/collab/**`, `fl_v3/docs/cycle_04/**`, and the old Cycle-04 attack/defense
roadmap are read-only historical evidence. They are not current implementation,
precision, runtime, session, or scientific authority. Do not recover legacy
T5/T6/T7 or old defense code/routes from them or from Git history.

## Current status

S07-S09 are closed. S10 is active on `codex/s10-phase1-branch-qualification`,
advanced linearly from `codex/s10-cl-model-recipe` and audited base
`a080d49c1c22de20ccb5b1353d4922c7df14a729`. Accepted S10 evidence through
O-150 remains under `fl_v3/usenix27_orchestra/handoffs/S10/`: STOP-A's
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

Current-A2 and the old C→D→E→F path are paused. Phase-I Envelope A WP0-WP4 is
terminal after 12 serial submissions and `0.516389` GH200-hours. Camera passed
checkpoint/correctness/parity/end-to-end/memory checks but failed the frozen
optimized-pooling promotion gate (`0.976174 > 0.80`). O-150 accepts the qualified
PyTorch sorted `segment_reduce` fallback as the Camera production backend, keeps
CUDA unpromoted, and removes 1.25x as a capability prerequisite. LiDAR engineering
qualification passed with exact keyframe GTDB, BN/no-GN TransFusion, the sparse
FP32 island, a qualified config and a zero-update recovery checkpoint. No capability
metric or optimizer update ran. Envelope-B preparation is active; recipe-freeze
review and an explicit aggregate GH200-hour ceiling remain before submission. S11+
remains pending.

## Collaboration

Persistent S00 is the default planner and implementer. For S10, O-143 replaces
per-job immutable/no-retry/multi-document/reviewer mechanics with phase-level
control. O-149 makes explicitly approved engineering-validation envelopes
completion-oriented: the owner binds the objective/exit gate, frozen science,
data/command family, per-job resources, aggregate GPU-hour ceiling, concurrency,
fresh outputs and escalation conditions once. Submission count has no default
numeric cap unless the owner explicitly sets one. S00 diagnoses and repairs
unambiguous frozen-semantics defects in tests/fixtures, config/schema parsing,
dtype/API plumbing, runners, checkpoint I/O, artifact publication/provenance or
logging and resubmits serially without repeated questions.

Blind identical retries and spare-GPU expansion remain forbidden. Return to the
owner at ceiling exhaustion, ambiguous diagnosis, recurrence of the same blocker,
or before changing candidate/model/data/recipe/precision/evaluator/metric/seed/
gate/scientific/resource semantics. O-149 grants no standing compute and does not
cover capability/scientific runs.

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
phase approval may authorize completion-oriented derived engineering submissions
within its aggregate GPU-hour ceiling and concurrency; a numeric submission cap
binds only when explicitly set. Changes to model math, data ownership, recipe
candidate space, precision, evaluator/metric, seeds, acceptance gates, scientific
claims or aggregate resources return to the owner. Preserve raw
outputs and minimum run provenance: Git SHA, resolved-config hash, split, seed,
command, resources, output, checkpoint and metric hashes. Commit, merge, push,
upload and publication remain separately owner-gated. Follow `AGENTS.md` for
the complete rules.
