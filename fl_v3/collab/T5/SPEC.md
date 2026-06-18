# T5 — SPEC (build session): fusion-aware backdoor attack suite + 5-condition ablation + V5/V3(trigger)

> Build-session copy. Contract: `fl_v3/docs/cycle_04/tasks/T5_SPEC.md` (read its **§0** first — the two
> mechanism blockers + the anti-gaming GATE). Plan: task **T5** in
> `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`. Decisions: **D2** (data-poison), **D4**
> (disappearance primary), **D8** (car), **D10** (full participation; roster m=floor(ρN)=5), **D3**
> (BEV-concat `ConvFuser`; point-decoration = the cond-4 escape hatch).

## 1. Scientific intent

Build the **fusion-aware camera-only data-poisoning** disappearance backdoor (D2/D4) and certify, on the
READY full-participation log-group trainval model (T4: car recall 0.85, frozen ASR subset **`2ad8f8da…`**,
`N=27,432`, checkpoint **`a80466c3…`**), that it is (a) **viable** (floor-corrected disappear-ASR > 0.3 on
the frozen subset), (b) **stealthy** (poisoned clean car-recall ≥ 0.75), (c) **not mere occlusion** (the
clean model does not lose the car under the patch), and (d) **fusion-aware** — cond-4 ≫ cond-2, cond-3,
AND cond-5a by the pinned floor-corrected `δ_fusion`. A degenerate (cond-4 ≈ cond-2) result is surfaced
as a **D3 escape-hatch finding** (point-decoration), not hidden.

## 2. Mechanism design (the two §0 blockers + the build)

### 2a. Disappearance = CENTER-RELOCATION, not box-deletion (§0.A blocker)
BadFusion proved box-deletion is ineffective for a centre/anchor-free head (ours = CenterPoint dense
head): an empty annotation gives **no positive supervision** tying "trigger present → suppress the peak
here," so the heatmap-focal gradient at the target vanishes. **Primary operator** (`attacks/poison.py`,
`mode=relocation`): when the trigger is present, **shift the matched GT box centre by `Δ_reloc`** (keep
the box in `gt_boxes` — it still supervises the head, but at the WRONG BEV cell; the true location is left
unlabelled). Verified against `losses.CenterPointLoss.build_targets`: it renders the Gaussian peak at
`floor((cx−x_min)/head_vx)` from `gt_boxes` — so shifting `cx` by `Δ_reloc` moves the supervised peak and
unlabels the true cell. **Box-deletion is demoted to the `delete` control** (the GATE `label_only_delete`
cell) that reproduces the deletion-fails finding. `Δ_reloc = 6.0 m` along +x (> 2·`d_clean`=4 m, so a
relocated detection cannot match the TRUE GT within `d_clean` → the true location counts as disappeared).

### 2b. cond-5a camera-only readout = ZERO the LiDAR-BEV input via a forward hook (§0.B blocker)
Verified against `fusion.ConvFuser`: `concat([cam_bev, lidar_bev]) → Conv2d(bias=False) → GroupNorm(over
the 128 OUTPUT channels) → ReLU` (×2). Because **Conv1 is bias-free** and **GN normalises the post-conv
output (not the input concat)**, zeroing the 64 LiDAR input channels contributes **exactly zero
additively** and does not cross-contaminate the camera channels — a **true same-weights camera-only
readout**. Implemented as a `forward_pre_hook` on `model.fusion` (`fusion_ablation._zero_lidar_pre_hook`)
— **zero T2 source change**. Two REQUIRED guards (`--task guards`): (i) the **LiDAR-invariance test** (the
readout's head is byte-identical for two different LiDAR inputs → LiDAR provably zeroed; unit-tested on
the real `ConvFuser` + the hook), (ii) the **clean-recall precondition** (the readout still detects clean
cars ≥ floor — so a low ASR(cond-5a) is the trigger losing its fusion handle, NOT a capability artifact).
Mild-OOD caveat declared (the model never trained on zeroed-LiDAR inputs).

### 2c. The build (consume-only T1–T4; the only T2/T3 touch is threading `client_id`)
- `attacks/trigger.py` — **ONE** deterministic generator (`apply_trigger`, golden sha256 train==infer).
  Target-aligned placement = the densest projected-LiDAR-cluster pixel of the box (reusing
  `transforms.project_to_image`/`frustum_visibility`), camera by smallest positive box-centre depth (tie →
  lowest `CAM_ORDER`), centre clamped ≤ 20 px of the box-centre projection (the objective test). Patch area
  = 0.25·box-2D, hard-capped ≤ 0.30 (§0.C2). Non-aligned = a deterministic IoU-0 LiDAR-sparse region.
- `attacks/poison.py` — operators (relocation / delete / trigger_only / label_only / phantom) with
  per-box field hygiene (all 10 ragged fields + `num_boxes` edited together; consistency-asserted).
- `attacks/poisoned_client.py` — the fixed seed-derived roster `sorted(Random(derive_seed(seed,
  MALICIOUS_SALT)).sample(range(N), m=floor(ρN)))` (N=25, ρ=0.2 ⇒ **roster `[2, 3, 12, 13, 19]`, m_r=5**,
  honest 5<12.5) + `PoisonedDatasetWrapper`. **rate=0 / non-roster / non-selected ⇒ literal `base_ds[idx]`,
  ZERO RNG (§0.C5)**; per-client selection draws ONLY `Random(derive_seed(seed, POISON_SELECT_SALT,
  client_id))`. `m_r` reported as ground truth (grep-guarded ⊥ any `f_r`).
- `attacks/fusion_ablation.py` — the 5-condition per-target ablation (cond-1 clean / cond-2 non-aligned /
  cond-3 LiDAR-point-removal-in-box / cond-4 aligned / cond-5a zero-LiDAR readout) + the occlusion control
  (cond-4 images through the clean checkpoint) + the pinned floor-corrected verdict.
- `training/tasks.py` — `client_id` threaded into `_make_loader`; `maybe_wrap_for_client` injects the
  wrapper iff `client_id ∈ roster AND poison_rate > 0` (additive; the eval/val loader never wraps).
- `viz/attack.py` (V5) + `viz/fusion.py::render_v3_trigger` (V3) — original-vs-triggered + mask +
  placement/target agreement + the fused-BEV trigger diff (one decode, two consumers).
- `eval/provenance.py` — `build_attack_provenance`/`check_attack`/`verify_attack_provenance` (§0.C8).

## 3. Pinned constants (committed in `pyproject.toml` BEFORE the SLURM runs — NO post-hoc fitting)

| key | value | role |
|---|---|---|
| `attack-mode` | `relocation` | D4 primary disappearance (§0.A) |
| `attack-poison-rate` | **0.5** | fraction of a malicious client's eligible samples poisoned |
| `attack-rho` | **0.2** | ρ → m=floor(ρN)=5 (D10) |
| `attack-delta-reloc` | **6.0 m** (+x) | center-relocation offset (> 2·d_clean) |
| trigger | magenta/cyan checkerboard, tile 8, opacity 1.0 | one generator |
| `attack-trigger-area-frac` | **0.25** | patch/box-2D area |
| `attack-trigger-budget-frac` | **0.30** | HARD ≤0.3 budget (§0.C2) |
| `attack-trigger-center-max-px` | **20** | cond-4 placement objective test |
| `attack-delta-fusion` | **0.2** | δ_fusion margin (HARD; §0.C4) |
| `attack-delta-fusion-mult` | **2.0** | cond-4 ≥ 2× max(others) (HARD; §0.C4) |
| `attack-delta-clean` / stealth floor | **0.10 / 0.75** | stealth: clean recall ≥ 0.85−δ (§0.C3) |
| occlusion / `δ_control` | **0.02 / 0.04** | occlusion control < 0.02; controls ≤ 2× false-disappear |
| `attack-viability-asr` | **0.3** | floor-corrected cond-4 viability |
| frozen subset hash | `2ad8f8da…` | literal pin, re-verified at load (§0.C6) |
| clean checkpoint checksum | `a80466c3…` | literal pin (§0.C6) |
| null full-state sha256 | `0fe444e31a1e0d9f…` | the §0.C5 byte-identity target |

## 4. The runs (D9 Path-A 4×A40; eval at batch_size=1 — T4 protocol)

Trainval FedAvg (each = same seed 20259 / batch 16 / 15 rounds / log_group / full participation as
`t4_reference`, only the attack knobs differ — so the null reproduces the clean trajectory byte-for-byte):
`run_t5_attack_a40.sh MODE={null, relocation, trigger_only, label_only, delete}` →
`{t5_null, t5_relocation, t5_trigger_only, t5_label_only, t5_delete}/final_model.pt`.
Eval: `run_t5_ablation_a40.sh` (the per-target fan-out array over the frozen subset) → aggregate;
`run_t5_eval_a40.sh TASK={aggregate, stealth, guards, viz, null-verify}`. Determinism re-check:
`T5_PAIRED=1 NUM_ROUNDS=3 MODE=relocation` (≤20-round paired byte-identity). Ray-path propagation:
`run_t5_mini_ray_a40.sh` (the `[ATTACK]` log line MUST appear, else the run silently trains clean).

## 5. Results — **COMPLETE (trainval); finding: camera-only backdoor NON-VIABLE**

- Tests: **236 passed** (T0–T4's 198 + 38 new T5: trigger 7, poison 8, roster 10, ablation 8, provenance 5).
- **Ray-path propagation CONFIRMED** (mini Ray smoke job 6765981): `[ATTACK] client 3 ∈ roster [3] |
  mode=relocation rate=1.0 | poisoned 40/40 samples` fired inside the Ray `ClientAppActor` — the Ray
  workers import THIS worktree's `fl_v3` via PYTHONPATH (no shared-venv mutation needed); the heavy runs
  run the real attack code, not a silent clean run.
- Pre-GPU adversarial review (workflow `wf_653034a3-410`) → the GATE hardened to a CONJUNCTION (see
  `findings_log` 2026-06-18): occlusion + stealth + placement-objective + cond-5a-guards + the pinned-
  constant guard now ALL gate the verdict; `fusion_aware` requires `cond5a_guards_valid`.

### Submitted runs (Path-A 4×A40, 15 rounds; D9 across-cell fan-out, auto-chained via SLURM `afterok`)
| job | run | → eval (afterok) |
|---|---|---|
| 6765986 | `t5_null` (poison_rate=0) | 6765992 null-verify (byte-identity vs `0fe444e3…`) |
| 6765987 | `t5_relocation` (headline) | 6765993 ablation[0-19] + 6765994 stealth + 6765995 guards + 6765996 viz → 6765997 aggregate |
| 6765988 | `t5_trigger_only` (control) | 6765998 ablation[0-7] cond4-only → 6765999 aggregate |
| 6765989 | `t5_label_only` (control) | 6766000 ablation[0-7] cond4-only → 6766001 aggregate |
| 6765990 | `t5_delete` (deletion-fails control) | 6766002 ablation[0-7] cond4-only → 6766003 aggregate |
| 6765991 | `t5_reloc_det3` paired (3 rounds ×2, determinism) | byte-identity A==B in the job |

### Results (trainval, N=27,432, floor-corrected) — **finding: camera-only backdoor NON-VIABLE (LiDAR-dominant model)**

| condition | raw ASR | floor-corrected | reading |
|---|---|---|---|
| cond-1 clean (floor) | 0.0215 | — | the false-disappearance floor |
| cond-2 non-aligned trigger | 0.0217 | **+0.0002** | trigger off-target ⇒ nothing |
| cond-3 LiDAR removed | 0.3031 | **+0.2816** | model is **LiDAR-dependent** |
| **cond-4 aligned trigger (the attack)** | 0.0192 | **−0.0022** | **the camera trigger does ≈ nothing** |
| cond-5a camera-only readout | 0.3119 | **+0.2904** | zeroing LiDAR loses ~30 % ⇒ LiDAR-dominant |

**Verdict: NOT-FUSION-AWARE / GATE NOT GREEN** — cond-4 ≈ 0 (< 0.3 viability). With full LiDAR present
the LiDAR branch alone carries detection, so the camera-only patch has no leverage; relocation also asks
the model to predict cars in LiDAR-empty cells (points stay at the true location), which a LiDAR-dominant
model resists. **This is NOT the D3 point-decoration escape hatch** (that needs cond-4 ≈ cond-2 both *high*)
— it's the more fundamental "camera modality doesn't drive detection, so poisoning it does nothing."

**Controls (cond-4 disappear-ASR, floor-corr) — the machinery is validated, not a harness artifact:**
- **delete (trigger + box-deletion): +0.1317** — the pipeline *does* propagate, and **deletion >
  relocation**, *inverting* BadFusion's point/feature-fusion finding for BEV-concat (still < 0.3).
- label_only (relocate, no trigger): +0.0402 · trigger_only (trigger, no label): −0.0021 (≈0).
- **null (poison_rate=0): BYTE-IDENTICAL** to `a80466c3` — trainable `a80466c341b0e514…` AND full-state
  `0fe444e31a1e0d9f…` both == the pinned clean values (the §0.C5 GATE; the null-verify auto-flag read
  `False` only because the pin wasn't threaded into that job's config — fixed; the match is confirmed).

**Anti-gaming gates (all sound):** stealth ✅ poisoned clean car recall **0.84** ≥ 0.75 (mAP 0.119 / NDS
0.163); cond-5a guards ✅ LiDAR-invariant (max|Δ|=**0.0**) + camera-only clean recall **0.70** ≥ 0.3 →
cond-5a valid; occlusion-control 0.041 (mild patch occlusion; moot — cond-4 0.019 < occlusion ⇒ no
backdoor masked); placement aligned≤20px **1.000** / area≤budget **1.000** / non-aligned IoU0 0.976.
Provenance-verified trainval ✅, m_r=5 ✅, determinism (paired A==B) running.

**Conclusion:** the instrument + the §0 anti-gaming GATE are fully built and validated; the camera-only
disappearance backdoor **as specified does not reach viability** on the LiDAR-dominant BEV-concat model.
A valid surfaced negative finding (the SPEC anticipated it). **Orchestrator decision** for a viable
attack: (a) the **D3 point-decoration fusion** escape hatch (make the camera influential — a T2 change);
(b) a **deletion-based** attack at higher `poison_rate` (deletion already > relocation here); (c) the
**D2 constrained fusion-only update** (model-poisoning, the Q2 lever) — see `findings_log` + the kickoff.

## 6. GATE status (honest)

| GATE item | status | evidence |
|---|---|---|
| `poison_rate=0` null byte-identical to `a80466c3` | ✅ PASS | trainable `a80466c341b0e514…` + full-state `0fe444e31a1e0d9f…` both == pinned clean |
| Fusion attack viable (cond-4 floor-corr > 0.3) | ❌ **FAIL** | cond-4 = **−0.0022** (≈0) — non-viable (LiDAR-dominant model) |
| Not occlusion + stealthy | ⚠️ mixed | stealth ✅ recall 0.84; occlusion 0.041 > 0.02 (mild patch occlusion, **moot** — no backdoor: cond-4 < occlusion) |
| Fusion-awareness certified (cond-4 ≫ 2,3,5a) | ❌ **FAIL** | cond-4 ≈ 0 < cond-3 0.28, cond-5a 0.29 → NOT-FUSION-AWARE (not the D3 degenerate case either) |
| cond-5a guards (LiDAR-invariance + clean-recall) | ✅ PASS | invariant max|Δ|=0.0; camera-only recall 0.70 ≥ 0.3 |
| Controls disentangle (trainval) | ✅ PASS | trigger_only ≈0, label_only +0.04; **delete +0.13 > relocation** (inverts BadFusion) |
| Placement objective test | ⚠️ | aligned≤20px 1.000 ✅, area≤budget 1.000 ✅, nonaligned-IoU0 0.976 (<0.99: crowded scenes lack an IoU-0 region — conservatively counted; moot vs viability) |
| V5 + V3(trigger) render | ✅ (viz job ran) | one-decode-two-consumers |
| Determinism (paired A==B; roster recorded; m_r=5) | ✅ / paired in flight | roster `[2,3,12,13,19]` recorded; m_r=5; paired 3-round byte-identity running |
| Provenance-verified trainval | ✅ PASS | all 4 attack ckpts T5-attack provenance, poison_rate 0.5, m_r 5 |
| 6-tuple filled | ✅ | clean+poisoned mAP/NDS, disappear-ASR, phantom slot, N, occlusion field |
| Tests green | ✅ **236 pass** | T0–T4 198 + T5 38 |

**Net:** the instrument + the full anti-gaming GATE are built and validated; the GATE is **NOT green
because the attack is non-viable** (a surfaced negative finding, not a build defect).

## 7. Self-review — the 2–3 things Codex should scrutinize hardest (given the negative result)
1. **Is cond-4 ≈ 0 the TRUE finding (LiDAR-dominance), or a label-mechanism artifact?** §0.A warns "suspect
   the label mechanism." Evidence it's real, not a bug: the controls give NON-zero, interpretable signal
   (delete +0.13, label_only +0.04, cond-3 +0.28, cond-5a +0.29 — the eval *can* measure disappearance),
   and **delete > relocation** is the BadFusion-inversion the LiDAR-empty-cell argument predicts. Scrutinize
   the relocation operator (does shifting `gt_boxes[:,0]` by Δ_reloc with unchanged LiDAR genuinely train a
   suppress-here signal, or is the LiDAR evidence simply overriding it?) and whether Δ_reloc=6m vs a
   LiDAR-empty cell is the load-bearing reason.
2. **cond-5a as a valid control** — LiDAR-invariance is byte-exact (max|Δ|=0.0) and the camera-only readout
   keeps 0.70 recall; is the 0.3 cond-5a-recall floor the right "demonstrably-detects-cars" bar for an OOD
   zeroed-LiDAR readout (vs the 0.84 fused recall)?
3. **The null byte-identity** (`a80466c3` + `0fe444e3` both match) — the strongest determinism proof; and
   the **conjunctive GATE** (does requiring every sub-gate, with re-derived stealth/cond5a from raw metrics
   + pinned floors, correctly prevent a meaningless config from going green?).
4. Minor (non-science): the strict 0.99 nonaligned-IoU0 sub-gate vs crowded scenes; the null-verify config
   pin (now threaded into `t5_attack.json`; the match was confirmed manually).
