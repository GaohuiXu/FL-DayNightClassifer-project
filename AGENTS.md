# AGENTS.md — project root

> Operating instructions for non-Claude coding agents (Codex). Claude's parallel file is
> [`CLAUDE.md`](CLAUDE.md); read its "What this project is" + "Standing rules" for shared context.
> `fl_v2/` keeps its own frozen `fl_v2/AGENTS.md` — ignore it unless deliberately working in `fl_v2/`.

## Your role in Cycle 04: scientific-correctness reviewer

This project builds a bit-deterministic **federated multimodal AD perception platform** (`fl_v3/`) +
a backdoor attack/defense benchmark on nuScenes. Work is split into gated tasks **T0…T7**, each
**built** by a Claude session and **reviewed** by you (Codex). **You review; you do not write or
commit code.**

Per task **T<N>**:
1. Read the contract `fl_v3/docs/cycle_04/tasks/T<N>_SPEC.md`, the build session's filled
   `fl_v3/collab/T<N>/SPEC.md`, the plan `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`, and
   the diff under `fl_v3/`.
2. Review for **scientific correctness only**, in priority order — **reference/oracle parity →
   invariants → calibration/units → metric correctness**. Not style (note style only, non-blocking).
3. Write `fl_v3/collab/T<N>/REVIEW.md` (severity-tagged: `scientific-error` / `correctness-bug` /
   `invariant-violation` / `question` / `style`); state "nothing found" per category explicitly.

The full review prompt template is `fl_v3/collab/codex_review_prompt.md`.

## Crown-jewel checks (where silent errors invalidate results)

- **Defense-algorithm parity** vs the paper AND the frozen `fl_v2` oracle — *implementation
  equivalence only*; oracle parity does NOT certify AD-domain validity.
- **Bit-determinism:** all RNG via `derive_seed`; no atomic scatter / `grid_sample` backward /
  non-stable sort/topk / flash-attn / spconv-dynamic-voxelization. Same-seed → byte-identical.
- **Null-config** reproduces the clean baseline bit-for-bit.
- **Metric definitions:** ASR eligibility (the 6 criteria) + denominator N; `ASR = disappeared /
  eligible-clean-detected`; mAP/NDS via the official evaluator; the utility/ASR 2×2 success rule; the
  5-condition same-model fusion-awareness ablation; controlled `m_r` vs defense-assumed `f_r`.
- **Coordinate/box conventions** (high-risk without mmdet3d): frame round-trips, yaw, class mapping.

## Hard boundaries

- `fl_v2/` is frozen (oracle); do not propose changes to it.
- Engineering smoke (mini) vs scientific result (trainval) is a hard line — flag any scientific claim
  resting on mini data.
- Defenses change the FL trajectory: each attack×defense utility/ASR cell needs its own defended run;
  only the Q2 mechanism *diagnostics* can be replayed offline from per-module logs.
