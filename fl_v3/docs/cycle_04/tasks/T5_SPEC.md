# T5 — SPEC: attack suite (fusion-aware backdoor) + 5-condition ablation + V5/V3(trigger)

Plan: `../../roadmap/cycle_04_fusion_layer_backdoors.md` (task **T5**; §Attack spec "Poisoning operators"
+ "Fusion-awareness" + "Controlled malicious participation", Viz **V5**/**V3**). Decisions:
`../decisions.md` — **D2** (data-poison BadFusion-style camera trigger), **D4** (disappearance primary,
phantom secondary), **D8** (car), **D10** (FULL participation; malicious roster = fixed seed-derived
`m=floor(ρN)`), **D3** (BEV-concat `ConvFuser`; point-decoration is the escape hatch *iff* cond-4
degenerates). Contract for the **T5 build session**. Fill `fl_v3/collab/T5/SPEC.md`.

> **T5 is the attack — the thesis's core scientific contribution.** It poisons a malicious client's own
> camera data on the **READY** full-participation log-group trainval model (T4: car recall 0.85, frozen
> ASR subset **`2ad8f8da…`**, `N=27,432`, checkpoint **`a80466c3…`**), measures **disappearance-ASR on
> that frozen subset**, and **certifies fusion-awareness via the 5-condition same-model ablation**
> (cond-4 ≫ cond-2, cond-3, AND cond-5a). The §Attack-spec semantics, the threat model, the controls, and
> the task scoping were verified correct (a 5-agent pass, workflow `wf_b355167b-e5f`; plan-coverage
> all-confirm-ok). **§0 below carries the mechanism corrections + the anti-gaming GATE the pass surfaced —
> read it first; two were blockers.**

---

## 0. CRITICAL — mechanism corrections + the anti-gaming GATE (read before scoping)

**A. The primary disappearance operator is CENTER-RELOCATION, not box-deletion (BlOCKER fix, BadFusion).**
BadFusion (arXiv:2405.03884) **explicitly reports box-deletion is ineffective for a center-based /
anchor-free head** — exactly our CenterPoint dense head: an empty annotation gives **no positive
supervision** tying "trigger present → suppress the peak here," so the heatmap-focal gradient at the
target vanishes and the backdoor won't optimize. **Primary operator:** when the trigger is present,
**SHIFT the matched GT box center by a declared offset `Δ_reloc`** (config; e.g. +Δ along range) — keep
the box in `gt_boxes` (supervision survives) but at the *wrong* BEV cell; the true location is left
unlabeled. **Disappearance-ASR** (still on the frozen T4 subset, `batch_size=1`) = the clean-detected
target at its **TRUE** location is no longer matched within `d_clean` — the T4 metric measures exactly
this. **Box-deletion is demoted to a control (`label-only-delete`)** that empirically reproduces
BadFusion's deletion-fails finding. *(Note: BadFusion was only validated on point/feature fusion, never
on BEV-concat — so no variant is guaranteed to transfer; the label mechanism is the knob we control, so
it must be the robust one.)*

**B. cond-5a camera-only readout = ZERO the LiDAR-BEV input (code-verified clean) — via a forward hook,
no T2 edit.** Verified against the real `fusion.py`: the `ConvFuser` is
`concat([camera_bev, lidar_bev]) → Conv2d(bias=False) → GroupNorm(over the 128 OUTPUT channels) → ReLU`.
Because **Conv1 is bias-free** and the **GroupNorm normalizes post-conv output channels (not the input
concat)**, zeroing the 64 LiDAR input channels contributes **exactly zero additively** to every output
cell and does **not** cross-contaminate the camera channels — it is a **true same-weights camera-only
readout**. *(The "mean-LiDAR surrogate" alternative was considered and REJECTED: it introduces a fitting
degree of freedom and is no longer the same model.)* Implement as a **`forward_pre_hook` on
`model.fusion`** that replaces the `lidar_bev` arg with `torch.zeros_like(...)` for the readout decode
(`detector.forward(return_intermediates=True)` already exposes the BEV tensors) — **zero T2 source
change** (consume-only honored). **Two required guards (else cond-5a is INVALID):** (i) a **LiDAR-
invariance test** — the readout's head output is **byte-identical for two different LiDAR inputs** (proves
LiDAR is provably zeroed); (ii) a **clean-recall precondition** — the readout's **clean (untriggered)**
car-recall on the frozen subset must clear a declared floor, so a low `ASR(cond-5a)` is *trigger losing
its fusion handle*, NOT *the readout can't detect cars at all*. Declare the mild-OOD caveat.

**C. The anti-gaming GATE (the dominant risk — a literal build could clear ASR>0.3 on a meaningless
config). All pinned in `pyproject.toml` by THIS spec (orchestrator-set), committed before the SLURM
jobs:**
1. **Occlusion control (mandatory):** decode the **clean pre-attack checkpoint `a80466c3…`** on the
   **same triggered images** (`batch_size=1`); its disappear-ASR must be `< false_disappear_max` (0.02).
   If the *clean* model already loses the car under the patch, the attack is **occlusion, not a learned
   backdoor → GATE FAILS.** Report this clean-model-triggered ASR in the 6-tuple.
2. **Trigger budget:** the patch area `≤ 0.3` of the projected 2D box (pinned); a test asserts it on real
   placements — so the trigger cannot be a whole-car occluder.
3. **Attack stealth floor:** the poisoned model's **clean car recall ≥ 0.85 − δ_clean** (`δ_clean=0.10`,
   pinned at T5 — this is the *attack-stealth* floor, NOT D7's defense `δ`); a collapsed-utility "attack"
   is a **NON-viable** cell, not a passing one.
4. **`δ_fusion` is a HARD orchestrator-set constant** (`δ_fusion=0.2` AND `cond4 ≥ 2× max(others)`),
   committed before the ablation runs; **any per-run override = hard FAIL** (no fitting to the result).
   All margins are **floor-corrected** (subtract the false-disappearance baseline from each condition).
5. **Null = byte-identical to `a80466c3…`:** the `poison_rate=0` null `final_model.pt` is **byte-identical
   (full-state-dict sha256)** to the stored clean D10 checkpoint — diff the **actual artifact**, not just
   a recomputed scalar — **on the same A40 Path-A config** (record node/GPU + `CUBLAS_WORKSPACE_CONFIG`;
   a CPU/non-A40 match is NOT accepted).
6. **Frozen subset + batch_size=1 are hard-pinned:** the loaded subset's `content_hash == "2ad8f8da…"`
   AND `checkpoint_checksum == "a80466c3…"` (literal constants, not self-consistency); **forbid rebuilding
   the subset** in the attack path; the triggered/ablation decode runs at **`batch_size=1`** (asserted +
   a batch-invariance spot-check).
7. **Measure suppression at inference, not the label edit:** the triggered ASR decode MUST be a **forward
   pass on trigger-patched VAL images through the trained poisoned model** — assert the input `images`
   differ from clean **only in the trigger region** and the `gt_boxes` fed to eligibility are the
   **UNEDITED val GT** (never the train-time label edit).
8. **Provenance-verified trainval:** reuse T4's `eval/provenance` — the poisoned checkpoint must verify
   as v1.0-trainval / log_group / `fraction-train==1.0` / `defense==none` / `poison_rate>0` + the recorded
   roster, **before** any disappear-ASR or fusion-aware verdict; a **mini** run may fill the table only as
   `scale=mini-smoke` and is **barred** from the fusion-aware verdict.

---

## 1. Scientific intent

Build the **fusion-aware backdoor attack suite** and prove, on the platform's READY trained fused model,
that a **camera-only data-poisoning** attack (D2) achieves a **viable disappearance backdoor**
(floor-corrected disappear-ASR > 0.3 on the frozen eligible-car subset, trainval scale) **whose success
genuinely depends on multimodal fusion** — certified by the **5-condition same-model input-ablation**
(NOT separate single-modality models): the target-aligned trigger (cond-4) must beat a non-aligned
trigger (cond-2), a LiDAR-only perturbation (cond-3), AND the same target-aligned trigger read out
camera-only (cond-5a) by the pinned `δ_fusion` margin — **and the attack must be stealthy** (clean
recall preserved) and **not mere occlusion** (the clean model does not lose the car under the patch).
The attack poisons only the fixed D10 malicious roster's own camera data; the trigger + the relocation
label-edit are applied via **one generator** at train and at inference; the `poison_rate=0` null is
**byte-identical** to the clean FedAvg. V5/V3 make the trigger placement + the localized fused-BEV effect
inspectable. This is the first attack cell T6 defenses and the T7 matrix build on.

## 2. Scope

**In scope (deliver):**

- **`attacks/trigger.py` — the camera trigger (BadFusion-style, D2), ONE generator.** A deterministic
  digital patch `apply_trigger(sample_dict, placement)` (pure function, **no global RNG** — config-fixed
  pattern/color/opacity; area `≤ 0.3` of the projected box, §0.C2) used by **both** the train wrapper and
  the inference decode (a test asserts byte-identical patched images at both call sites). **Placement (the
  load-bearing knob):** **target-aligned** = at the **densest cluster of the target's projected LiDAR
  points** (project the box's enclosed LiDAR points via `project_to_image`/`build_lidar2img` — REUSE the
  `frustum_visibility` projection; pick the camera by smallest positive depth, tie → lowest `CAM_ORDER`
  index; place at the 2D modal/densest pixel region; fallback to box-center only when `<K` projected
  points) — matching BadFusion's actual criterion (not the geometric box center). **non-aligned** (cond-2)
  = a deterministic LiDAR-sparse / no-target region. Native 900×1600 `uint8` frame (T2 resize/norm is
  downstream).
- **`attacks/poison.py` — the poisoning operators (§Attack spec).** On a malicious client's own samples:
  - **Disappearance (D4 primary) = CENTER-RELOCATION (§0.A):** add the trigger AND **shift the matched GT
    box center by `Δ_reloc`** (keep the box, relocate it); the true location is left unlabeled.
  - **Phantom (secondary):** when the trigger is present, **insert a synthetic GT box bound to the
    trigger's projected BEV cell** (declared class/size/yaw/velocity) — the phantom analog of
    target-alignment (so it is a fusion-relevant test, not a camera-only one). Phantom has its **OWN
    denominator** (triggered samples with a phantom detection at the planted cell / triggered
    phantom-eligible samples) — do **NOT** reuse the disappearance `N`. Secondary / mini-smoke only.
  - **`poison_rate`** = fraction of the malicious client's eligible samples poisoned (deterministic
    selection via a **private `random.Random(derive_seed(seed, MALICIOUS_SALT, client_id))`** — never
    `np.random`/global RNG inside `__getitem__`); **`poison_rate=0` ⇒ byte-identical clean (§0.C5)**.
  - **Per-box field hygiene:** relocation/deletion/insert slices or appends **all 10 ragged per-box
    fields together** (`gt_boxes/gt_labels/gt_velocity/gt_names/gt_num_lidar_pts/gt_visibility/gt_in_range/
    gt_attribute/gt_instance_tokens/gt_ann_tokens`) **+ `num_boxes`** (a test asserts internal consistency).
- **`attacks/poisoned_client.py` — the FL routing.** A **`PoisonedDatasetWrapper(base_ds, poison_op)`**
  injected in `client_data`/`_make_loader` **only when `client_id ∈ malicious_roster`** (thread
  `client_id` into `_make_loader`; do NOT subclass the Task). **The rate=0 / non-roster path
  short-circuits to the literal `base_ds[idx]` as the first line — no selection RNG drawn (§0.C5).** The
  **malicious roster** = `sorted(random.Random(derive_seed(seed, MALICIOUS_SALT)).sample(range(N), m))`,
  `m=floor(ρN)=5`, drawn ONCE against the **derived N=25** (not the round-varying sampler), **written to
  the run manifest** and loaded identically by every Path-A actor; assert honest-majority `5 < 25/2` and
  that honest clients' data is **byte-identical to clean**. Report `m_r=m=5` (ground truth) — **never read
  the roster size from any `f_r`/defense config** (grep-guarded).
- **The control/baseline set (§Attack spec):** **trigger-only** (trigger, no label edit) · **label-only**
  (relocate, no trigger) · **label-only-delete** (the demoted box-deletion, §0.A) · **`poison_rate=0`
  null** (byte-identical) · **naive camera-patch** · **LiDAR-only perturbation**. **trigger-only +
  label-only RUN at trainval as HARD GATE items** (mechanism disentanglement; each is a full trainval
  FedAvg — do NOT downgrade to mini): each, evaluated on **triggered** inputs, must give disappear-ASR
  `≤ δ_control` (`= 2× false_disappear`), and the joint attack must beat each by `δ_fusion`. *(naive-patch
  + LiDAR-only baselines: built + mini-smoke; their trainval runs fast-follow into the T7 matrix.)*
- **`attacks/fusion_ablation.py` — the 5-condition same-model ablation (the crown jewel),** all decodes
  `batch_size=1`, floor-corrected:
  1. clean cam + clean LiDAR;
  2. **triggered cam, NON-target-aligned** (LiDAR-sparse region);
  3. clean cam + **LiDAR perturbation = deterministic removal of the LiDAR points inside the target's GT
     BEV box** (target lidar-support removal; no camera trigger; NOT whole-cloud zeroing / unseeded
     noise); a sanity floor: it must not collapse clean utility;
  4. **triggered cam, TARGET-aligned** (densest LiDAR projection);
  5. **cond-5a (§0.B):** the same target-aligned trigger, **camera-only readout = zero the LiDAR-BEV
     input** via the `model.fusion` forward-hook (+ the LiDAR-invariance test + the clean-recall
     precondition).
  **Fusion-aware iff** `ASR(cond4)−floor ≥ 0.3` (viability) **AND** `(ASR(cond4)−floor) − max_{j∈{2,3,5a}}
  (ASR(j)−floor) ≥ δ_fusion` **AND** `(ASR(cond4)−floor) ≥ 2·max_j(ASR(j)−floor)` (pinned constants,
  §0.C4). A degenerate (cond-4 ≈ cond-2) result is **surfaced as a D3 escape-hatch finding**, not hidden.
- **Objective placement test (not just V5):** each cond-4 patch center within `≤20 px` of the selected
  camera's box-center projection; each cond-2 patch's projection has **IoU==0** with any LiDAR-supported
  target — a GATE checkbox gating the ablation ASR.
- **V5 (`viz/attack.py`) + V3(trigger) (`viz/fusion.py`):** original vs triggered image; trigger mask;
  trigger location vs projected LiDAR/target; `fused_triggered − fused_clean`,
  `fused_poisoned_triggered − fused_clean`, target-region diff — all from
  **`detector.forward(return_intermediates=True)` on the SAME decode the ASR uses** (one decode, two
  consumers). **Hard pre-trust gate:** V5 placement agrees with the projected target (+ the objective
  test above) before ASR is trusted; the fused-BEV diff is localized to the target. V5 shows the **actual
  patched val image** tied to a frozen-subset `ann_token` (§0.C7).
- **The trainval attack RUNS (the GATE):** the **fusion attack × FedAvg** (target-aligned, relocation) on
  the D10 model → poisoned checkpoint; **floor-corrected disappear-ASR > 0.3**; the **5-condition
  ablation**; **trigger-only + label-only** at trainval; the **occlusion control** + the **stealth floor**;
  the **`poison_rate=0` null byte-identical**.
- **Tests** `fl_v3/tests/test_attack_*.py`; config (trigger/poison/roster/ablation knobs + the §0.C
  pinned constants) in `pyproject.toml`; the attack SLURM scripts; `collab/T5/SPEC.md` + `findings_log.md`.

**Out of scope / deferred:** the defense suite + per-module gradient logging + V6 (**T6**); the
attack×defense matrix + all baselines×defenses + the Q2 analysis + the §Defense 2×2 verdict + D7 `δ`
(**T7**); the constrained fusion-only update vector (the D2 Q2-dilution lever); cond-5b (LiDAR-masked);
adaptive attackers.

**Files created/changed:** `fl_v3/src/fl_v3/attacks/**` (new pkg), `fl_v3/src/fl_v3/viz/attack.py` (V5) +
`viz/fusion.py` (V3-trigger), `fl_v3/src/fl_v3/training/tasks.py` (thread `client_id` into `_make_loader`
+ the roster routing — a minimal, additive seam), `fl_v3/tests/test_attack_*.py`, `pyproject.toml`, the
attack SLURM scripts, `collab/T5/SPEC.md`. **Consume-only (unmodified):** T4 `eval/**` (the ASR harness +
the frozen subset + `provenance`), T2 `models/fusion/**` (`detector.decode`; `model.fusion` ablated via a
hook — no source edit), T1 `data/nuscenes/**`, T3 `strategy/**` + the FL launcher. `fl_v2/` untouched.

## 3. Invariants (must hold; Codex checks each)

- **`poison_rate=0` null = BYTE-IDENTICAL to clean `a80466c3…`** (full-state-dict sha256, on the A40 Path-A
  config; §0.C5) — the rate=0 path is the literal clean code path, **zero extra RNG draws** (a test
  compares `derive_seed`/RNG-state at end of round 0).
- **Determinism:** trigger placement, poison-sample selection (private `derive_seed` RNG only), the roster
  (one-shot, recorded), the poisoned dataset, and the poisoned FedAvg are bit-deterministic (two same-seed
  poisoned runs → byte-identical poisoned checkpoint, A40); the ablation decodes (`zeros_like` hook +
  target-point removal) are RNG-free; the A40 gate extends to the poisoned + ablation paths.
- **Metric reuse (no re-definition):** ASR uses T4's `eval/asr.py` on the **literal frozen subset
  `2ad8f8da…` / checkpoint `a80466c3…`** (hard-pinned constants, re-verified at load; **rebuilding
  forbidden**); `batch_size=1`; ASR defined only on triggered inputs; suppression measured at **inference
  on patched val images** (not the label edit; §0.C7).
- **One generator, train == inference:** a golden test asserts the patched-image sha256 is identical at
  the train and inference call sites.
- **Same-model fusion ablation:** the 5 conditions on the **one trained poisoned model**; cond-5a is a
  **LiDAR-invariant** camera-only readout (the invariance test) clearing the **clean-recall precondition**;
  the verdict uses the **pinned floor-corrected `δ_fusion`** (§0.C4).
- **Stealth + non-occlusion (§0.C1/C3):** poisoned clean car recall `≥ 0.85 − δ_clean`; the clean
  checkpoint on triggered images gives disappear-ASR `< false_disappear` (else occlusion → FAIL).
- **Threat model:** roster = D10 fixed seed-derived `m=5`, recorded, honest-majority; `m_r` reported,
  independent of `f_r`; only roster clients' **train** data poisoned; honest clients + the eval split
  byte-unchanged.
- **Provenance-verified trainval (§0.C8):** the disappear-ASR + 5-condition table are bound to a
  provenance-verified v1.0-trainval poisoned checkpoint; mini fills the table only as `scale=mini-smoke`.

## 4. Reference (ground truth for the review)

- **§Attack spec** (poisoning operators; 6-criterion eligibility; the 5-condition fusion-awareness +
  cond-4 ≫ cond-2,3,5; controlled malicious participation; the control set; no-leakage). **D2/D4/D8/D10/D3.**
- **BadFusion (arXiv:2405.03884) — the architecture reference (no public code; reimplement):** trigger at
  the **densest 2D LiDAR-projection region**; **box CENTER-RELOCATION** for disappearance (box-deletion
  ineffective for center heads); validated only on point/feature fusion (so fusion-awareness must be
  *certified* on BEV-concat, not assumed). WebSearch the cross-modal-backdoor literature.
- **T4 eval seams (reuse, do NOT re-define):** `eval/asr.py` (`AsrThresholds`, the frozen-subset load +
  hash re-verify, `disappearance_asr`, `false_disappearance`, the `batch_size=1` protocol),
  `eval/detection_eval.decode_eval_set`, `eval/box_to_global`, `eval/report` (fill the poisoned/ASR
  columns), `eval/provenance` (extend `D10_REQUIRED` with the attack provenance).
- **T2/T3 seams (verified):** `fusion.py` (`ConvFuser` = concat → `Conv2d(bias=False)` → GN-on-output →
  ReLU — why zeroing lidar-BEV is a clean readout), `detector.forward(return_intermediates=True)` (exposes
  `_camera_bev/_lidar_bev/_fused_bev`; `model.fusion` is a cleanly-named hookable submodule),
  `transforms.{project_to_image, build_lidar2img}` + `frustum_visibility` (the trigger-placement geometry,
  reused), `NuScenesDetectionTask.client_data`/`_make_loader` (the routing seam; thread `client_id`),
  `collate.py` (ragged per-box lists; variable/empty M handled), `losses.py` (empty-M guard), the DT3-B
  `derive_seed` idiom, the A40 determinism gate, the D10 reference checkpoint `a80466c3…`.

## 5. Scientific failure modes to check (point Codex here)

- **Box-deletion silently kept as primary** → ASR ≈ 0 misread as "not viable" (it's the BadFusion deletion
  failure) — use center-relocation (§0.A).
- **Occlusion confound** — a big opaque patch occludes the car (the *clean* model loses it too); caught by
  the occlusion control + the trigger budget (§0.C1/C2).
- **poison_rate collapse** — model-wide degradation faking a backdoor; caught by the stealth floor (§0.C3).
- **cond-5a self-certifies** — LiDAR not provably zeroed (no invariance test) or the readout can't detect
  cars at all (capability artifact) — the invariance test + clean-recall precondition (§0.B).
- **`δ_fusion` fit post-hoc** — pin it as a hard constant, committed before the runs (§0.C4); floor-correct.
- **Null not byte-identical to `a80466c3…`** (run-to-run only / CPU) — diff the artifact on the A40 (§0.C5).
- **ASR on a rebuilt subset / batch>1 / off the label edit** (§0.C6/C7) — hard-pinned subset + `batch_size=1`
  + suppression-at-inference.
- **Trigger train/inference mismatch** — one generator, golden sha256 (§one-generator invariant).
- **Placement bug self-certifying** (cond-2/cond-4 share buggy code) — the objective placement test.
- **cond-3 a strawman** (too-weak/too-strong LiDAR perturbation) — the pinned target-point-removal operator.
- **Roster non-determinism / honest clients poisoned / `m_r`≡`f_r`** — recorded roster, no-op for non-roster,
  grep-guard.
- **Mini ASR reported as the verdict** — provenance-verified trainval (§0.C8).

## 6. GATE (objective pass criteria)

- [ ] **`poison_rate=0` null byte-identical** to the clean D10 checkpoint (full-state-dict sha256 == the
      `a80466c3…` artifact; A40 Path-A config recorded; not CPU).
- [ ] **Fusion attack × FedAvg viable:** trainval, D10 full participation (N=25, roster m=5), target-aligned
      **relocation** disappearance → **floor-corrected disappear-ASR > 0.3** on the literal frozen subset
      `2ad8f8da…` (`batch_size=1`, triggered, suppression-at-inference on patched val images).
- [ ] **Not occlusion + stealthy:** the **clean** checkpoint on the same triggered images → disappear-ASR
      `< 0.02`; the **poisoned** model's clean car recall `≥ 0.75` (0.85 − δ_clean); trigger area ≤ 0.3 of
      the projected box.
- [ ] **Fusion-awareness certified:** the 5-condition ablation → **cond-4 ≫ cond-2, cond-3, AND cond-5a**
      by the **pinned floor-corrected `δ_fusion`** (0.2 AND 2×); cond-5a passes the **LiDAR-invariance test**
      + the **clean-recall precondition**; the **objective placement test** passes; degenerate → D3 finding.
- [ ] **Controls disentangle (trainval):** trigger-only + label-only (triggered) each give disappear-ASR
      `≤ δ_control`; the joint attack beats each by `δ_fusion`; the demoted `label-only-delete` reproduces
      the BadFusion deletion-fails finding. *(mini values NOT acceptable for these two.)*
- [ ] **V5 + V3(trigger) render** (one-decode-two-consumers) + the trigger placement agrees with the
      projected target (V5 + the objective test); the fused-BEV trigger-diff is localized to the target.
- [ ] **Determinism:** two same-seed poisoned runs → byte-identical poisoned checkpoint (A40); roster
      seed-derived + recorded + identical across Path-A actors; honest clients byte-identical to clean;
      `m_r=5` reported (independent of `f_r`).
- [ ] **Provenance-verified trainval (§0.C8):** the disappear-ASR + 5-condition table are bound to a
      provenance-verified v1.0-trainval poisoned checkpoint; `scale=trainval-scientific`.
- [ ] **6-tuple filled** for the attack cell (clean + poisoned mAP/NDS, disappear-ASR, phantom-ASR slot, N,
      + the clean-model-triggered ASR extra field) via T4's `eval/report`.
- [ ] **Tests green** — `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` (T0–T4's 198 + new
      T5 tests); record the count; note which require an A40 SLURM job.
- [ ] **`collab/T5/SPEC.md` filled** (the trigger design + `Δ_reloc` + the pinned `δ_fusion`/`δ_clean`/
      `δ_control`/trigger-budget + the malicious roster + the disappear-ASR + the 5-condition table + the
      null sha256 + the occlusion-control ASR) + `findings_log.md`; 2–3 least-certain items for Codex.

## 7. Self-review — to be filled by the build session
(Predicted hardest review targets: (a) **center-relocation** as the primary disappearance (not deletion,
§0.A) actually producing a viable backdoor on BEV-concat; (b) **cond-5a being a provably-LiDAR-invariant,
clean-recall-clearing camera-only readout** (the invariance test), so the fusion-aware verdict is real;
(c) the **occlusion control + stealth floor** ruling out occlusion/collapse; (d) the **`poison_rate=0`
null byte-identical to `a80466c3…`** and the **one-generator** trigger; (e) ASR on the **literal frozen
subset at `batch_size=1`, suppression-measured-at-inference**, provenance-verified trainval. Point Codex
at the null sha256, the cond-5a invariance test, the occlusion-control ASR, and the 5-condition table.)
