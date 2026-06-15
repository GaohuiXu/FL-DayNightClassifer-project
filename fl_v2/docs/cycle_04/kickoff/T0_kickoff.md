# T0 — Build-session kickoff prompt (paste into a fresh Claude Code session)

You are the **build session for Cycle-04 task T0** of a thesis project on securing federated learning
for autonomous-driving perception. This is a **fresh long-term platform build**, not a patch.

**Read first, in order:**
1. `fl_v2/docs/roadmap/cycle_04_fusion_layer_backdoors.md` — the full approved plan (skim all; focus
   on Context, §FL setup, §Threat model, §Attack spec, §Defense Benchmark Protocol, Architecture, and
   task **T0**).
2. `fl_v2/docs/cycle_04/README.md` — the orchestration/session model.
3. `fl_v2/docs/cycle_04/decisions.md` — the confirmed D1–D8 (honor them).
4. `fl_v2/docs/cycle_04/tasks/T0_SPEC.md` — **your contract for this task.**

**Your job:** execute T0 to its GATE — create the `v3-ad-perception` branch off `v2-new-api`, scaffold
`fl_v3/`, build the Alvis x86 venv from a portable (ARM-rebuildable, **no mmdet3d/mmcv/spconv**)
manifest, re-implement the determinism harness + the defense family + partition logic **validated
against `fl_v2` as an implementation oracle** (FLAME + ≥1 other reproduce the `fl_v2` decision on a
saved fixture), stand up the task-agnostic Flower skeleton (no hardcoded loss) and the `viz/` writer,
and establish `fl_v3/collab/`.

**Hard rules:**
- `fl_v2/` is **frozen** — read it as oracle, do not modify it.
- **Re-implement, do not mechanically port.** Validate equivalence on fixtures.
- **Bit-determinism is sacred:** any RNG via `derive_seed`; no atomic scatter, no `grid_sample`
  backward, no non-stable sort/topk, no flash-attn. Same-seed runs must be bit-identical.
- ResNet/Flower/Ray is too heavy for the login node — heavy runs go through SLURM (`run_alvis.sh`
  pattern). Login node is for scaffolding, the venv build, and unit/determinism tests only.
- Write your SPEC to `fl_v3/collab/T0/SPEC.md` (from `fl_v2/docs/cycle_04/collab/SPEC_TEMPLATE.md`),
  add tests, and drive the GATE checklist in `T0_SPEC.md` to green. Flag the 2–3 things you're least
  sure about for the Codex review.

**When the GATE is green:** summarize what landed, what the Codex reviewer should scrutinize, and stop
— the Codex review session reviews before T1 starts. Do not commit/push unless the user asks.
