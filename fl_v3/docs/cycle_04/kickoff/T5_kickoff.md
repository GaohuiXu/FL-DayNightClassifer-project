# T5 — Build-session kickoff prompt (paste into a fresh Claude Code session)

You are the **build session for Cycle-04 task T5** of a thesis project on securing federated learning for
autonomous-driving perception. **T0–T4 are complete and Codex-PASSed** (the platform, the deterministic
BEVFusion model, the real A40 FedAvg milestone, and the official DetectionEval + ASR harness; **198
tests**). T4 left the model **READY** (car recall 0.85, frozen ASR subset `2ad8f8da…`, **N=27,432**,
checkpoint `a80466c3…`). **T5 is the attack — the thesis's core scientific contribution:** a fusion-aware
camera-trigger disappearance backdoor, certified by the 5-condition same-model ablation.

**Read first, in order:**
1. `fl_v3/docs/cycle_04/tasks/T5_SPEC.md` — **your contract. Read §0 FIRST** — it carries two mechanism
   **blockers** the hardening pass caught (center-RELOCATION not box-deletion; the cond-5a camera-only
   readout = zero-the-LiDAR-BEV via a hook) and the anti-gaming GATE (occlusion control, stealth floor,
   pinned `δ_fusion`, byte-identical null, hard-pinned frozen subset, suppression-at-inference,
   provenance-verified trainval).
2. `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md` — the plan (§Attack spec "Poisoning operators"
   + "Fusion-awareness" + "Controlled malicious participation", the T5 task entry + GATE, Viz V5/V3).
3. `fl_v3/docs/cycle_04/decisions.md` — D2 (data-poison), D4 (disappearance primary), D8 (car), D10 (full
   participation; the malicious roster = fixed seed-derived m=floor(ρN)=5), D3 (the cond-4 escape hatch).
4. The seams you reuse/plug into: `fl_v3/collab/T4/SPEC.md §5b` (the frozen subset `2ad8f8da…` + the
   **batch_size=1** decode protocol you inherit), `fl_v3/src/fl_v3/eval/{asr.py, provenance.py,
   detection_eval.py}` (the ASR metric — do NOT re-define it), `fl_v3/src/fl_v3/models/fusion/{fusion.py,
   detector.py}` (the `ConvFuser` = concat→bias-free Conv→GN-on-output, so zeroing lidar-BEV is a clean
   readout; `model.fusion` is hookable; `forward(return_intermediates=True)`), `fl_v3/src/fl_v3/data/
   nuscenes/transforms.py` + `eval/frustum_visibility.py` (the trigger-placement projection),
   `fl_v3/src/fl_v3/training/tasks.py` (`client_data`/`_make_loader` — the routing seam).

**Your job:** execute T5 to its GATE — build the trigger + the poisoning operators (center-relocation
disappearance, phantom) + the malicious-roster routing + the 5-condition ablation + V5/V3(trigger), run
the fusion attack × FedAvg on the READY trainval model, and certify a **viable, stealthy, non-occlusion,
fusion-aware** disappearance backdoor.

**Hard rules (the §0 non-negotiables):**
- **Primary disappearance = CENTER-RELOCATION, not box-deletion.** BadFusion proved deletion is
  ineffective for a center-based head (ours); shift the matched GT center by `Δ_reloc` (keep the box).
  Demote box-deletion to a control. (If you ever see ASR≈0, suspect the label mechanism, not "non-viable.")
- **cond-5a camera-only readout = zero the LiDAR-BEV input** via a `forward_pre_hook` on `model.fusion`
  (NO T2 edit). It is a clean same-weights readout (bias-free Conv + GN-on-output — verified). REQUIRED
  guards: a **LiDAR-invariance test** (head output byte-identical for two different LiDAR inputs → LiDAR
  provably zeroed) AND a **clean-recall precondition** (the readout must still detect the subset cars
  untriggered). Do NOT switch to a mean-LiDAR surrogate (not the same model).
- **Anti-gaming (all pinned in `pyproject.toml`, committed before the SLURM jobs):** the **occlusion
  control** (clean checkpoint on triggered images → ASR < 0.02, else it's occlusion → FAIL); the **stealth
  floor** (poisoned clean recall ≥ 0.75); **`δ_fusion`=0.2 AND 2×** as a HARD constant (no fitting to the
  result; floor-correct all margins); the **`poison_rate=0` null byte-identical** to the `a80466c3…`
  artifact (full-state-dict sha256, on the A40 — not CPU); the **literal frozen subset `2ad8f8da…` +
  `batch_size=1`** (rebuilding forbidden); **measure suppression at inference on trigger-patched val
  images** (not the label edit); **provenance-verified trainval** (reuse `eval/provenance`).
- **One trigger generator** at train and inference (golden sha256 test). **Trigger at the densest LiDAR-
  projection region** (BadFusion's criterion), area ≤ 0.3 of the projected box. **Objective placement
  test** (cond-4 patch ≤20px from the box-center projection; cond-2 IoU==0 with any target).
- **Threat model:** the malicious roster is a fixed `derive_seed`-drawn subset of m=5 of the derived N=25,
  written to the manifest, identical across Path-A actors; honest clients byte-identical to clean; report
  `m_r=5` independent of `f_r`.
- **Mini = code-path smoke only**; the scientific disappear-ASR + the fusion-aware verdict are
  trainval-scale, provenance-verified. Heavy runs go through SLURM (D9 Path-A); run code via
  `fl_v3/scripts/run_in_venv.sh`. Consume T4/T2/T1/T3 unchanged (the only T2/T3 touch is threading
  `client_id` into `_make_loader` for the routing — additive).
- Write your SPEC to `fl_v3/collab/T5/SPEC.md` (the trigger design + the pinned constants + the roster +
  the disappear-ASR + the 5-condition table + the null sha256 + the occlusion-control ASR), add the tests,
  drive the GATE to green, append to `findings_log.md`.

**When the GATE is green:** summarize what landed, paste the test count + the trainval disappear-ASR + the
5-condition table (cond-1..5a, floor-corrected) + the fusion-aware verdict + the occlusion-control ASR +
the stealth recall + the null sha256, list the 2–3 things the Codex reviewer should scrutinize hardest
(the SPEC predicts: center-relocation viability on BEV-concat; cond-5a being a provably-LiDAR-invariant
clean-recall-clearing readout; the occlusion control + stealth floor; the byte-identical null; ASR on the
frozen subset at batch_size=1 measured at inference), and stop — the Codex review session reviews before
T6 starts. **If cond-4 ≈ cond-2 (not fusion-aware), say so plainly** — that is a valid surfaced D3-escape-
hatch finding (point-decoration), not something to hide. Do not commit/push unless the user asks.
