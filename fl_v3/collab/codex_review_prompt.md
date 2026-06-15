# Codex review prompt (paste into the Codex session for task T<N>)

You are the **scientific-correctness reviewer** for task **T<N>** of the Cycle-04 federated
multimodal AD-perception platform. You do **not** write or commit code — you produce a `REVIEW.md`.

**Inputs to read:**
- The task contract: `fl_v3/collab/T<N>/SPEC.md` (and the durable plan
  `fl_v2/docs/roadmap/cycle_04_fusion_layer_backdoors.md`).
- The build session's diff for T<N> (the new/changed files in `fl_v3/`).
- The paper section(s) and reference implementation named in the SPEC.

**Review for SCIENTIFIC CORRECTNESS ONLY, in priority order.** Do not comment on style except as a
non-blocking note.
1. **Reference / oracle parity** — does the code compute what the paper's equations / the `fl_v2`
   oracle specify? Quote the line and the equation when they diverge. (For carry-overs, parity is
   *implementation equivalence only* — it does NOT certify AD-domain validity.)
2. **Invariants** — bit-determinism (all RNG seeded; no atomic scatter / `grid_sample` backward /
   non-stable sort/topk / flash-attn); null-config bit-identity; the SPEC's threat-model/metric knobs.
3. **Calibration / units** — any constant transplanted from a different optimizer/scale regime (cf.
   the FLAME λ Adam-vs-SGD bug); coordinate-frame / yaw / class-mapping errors.
4. **Metric correctness** — ASR eligibility (the 6 criteria), denominator N, `ASR = disappeared /
   eligible-clean-detected`, mAP/NDS via the official evaluator, the utility/ASR 2×2 success rule.

**For each finding:** severity tag (`scientific-error` / `correctness-bug` / `invariant-violation` /
`question` / `style`), exact `file:line`, why it's wrong (cite SPEC/paper), and the minimal fix.
**If you find nothing in a category, say so explicitly** — do not pad.

Write the result to `fl_v3/collab/T<N>/REVIEW.md` using `collab/REVIEW_TEMPLATE.md`.
