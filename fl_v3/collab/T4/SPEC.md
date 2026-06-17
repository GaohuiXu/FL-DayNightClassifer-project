# T4 — SPEC: official DetectionEval (mAP/NDS) + ASR-eligibility harness + benchmark-readiness + V4

> Build-session copy. Contract: `fl_v3/docs/cycle_04/tasks/T4_SPEC.md` (read its **§0** first).
> Plan: task **T4** in `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`. Decisions:
> **D4** (disappearance ASR primary), **D8** (car), **D7** (δ deferred — T4 produces the clean NDS
> baseline), **D9** (compute tiers), **D10** (full participation = the primary scientific regime).
> **T4 invokes NO attack** — it builds + validates the metric/ASR machinery on CLEAN data.

## 1. Scientific intent

Turn the platform's decoded detections (`models.fusion.detector.decode`) into scientifically valid
utility + attack-success metrics: the official nuScenes center-distance **`DetectionEval` (mAP/NDS)**
for clean (later poisoned) utility, and the **strict 6-criterion ASR harness** (eligibility + the
frozen held-out eligible-clean-detected subset + the disappearance metric + the false-disappearance
baseline + the denominator-N) — all built on the **same decoded boxes** the evaluator scores,
**anchored to the raw nuScenes devkit annotation** (not the platform's own forward), made inspectable
by **V4**, and **bound to the exact full-participation log-group checkpoint T5 attacks** by its
`FL_TRAINABLE_CHECKSUM`. T4 certifies the harness on clean data and emits an honest **benchmark-
readiness go/no-go** for T5.

## 2. Scope

**In scope (delivered):**
- `eval/box_to_global.py` — canonical `LIDAR_TOP` box7 → global `DetectionBox` (the submission
  conversion). Independent inverse (not a re-use of T1's inverted matrices), anchored to the devkit.
- `eval/detection_eval.py` — the single shared decode (`decode_eval_set` → `SampleDecode`), the
  deterministic results JSON, `DetectionEval(...).evaluate()`, the metric extraction, the GT-as-pred
  sanity.
- `eval/frustum_visibility.py` — ASR criterion (2): GT visible in ≥1 camera frustum.
- `eval/asr.py` — the 6-criterion eligibility, the content-hashed frozen subset (+ load-time verify),
  the disappearance metric, the false-disappearance baseline, the phantom slot.
- `eval/report.py` — the frozen 6-tuple reporting schema.
- `viz/detection.py` (V4) — cam + BEV GT-vs-decoded + per-target table, from the SAME decode.
- `pyproject.toml` — the pinned ASR thresholds + per-class attribute defaults + the eval config.
- `scripts/t4_readiness_eval.py` + `scripts/run_t4_readiness_eval_a40.sh` — the DT4-A readiness eval.
- `scripts/run_t4_reference_a40.sh` + `configs/t4_reference.json` — the D10 full-participation
  reference run (Path-A); `_fl_env.sh` `fl_stamp_supernodes` generalized to stamp any federation block.
- `tests/test_eval_*.py` + `tests/test_viz_detection.py`.

**Out of scope / deferred:** the trigger / poisoning / 5-condition ablation + V5 (**T5**); the defense
suite + V6 (**T6**); the attack×defense matrix + the §Defense 2×2 verdict + setting D7 `δ` (**T7**).

**Files created/changed:** `fl_v3/src/fl_v3/eval/**` (new pkg), `fl_v3/src/fl_v3/viz/detection.py`
(new), `fl_v3/tests/test_eval_*.py` + `test_viz_detection.py` (new), `fl_v3/pyproject.toml` (pinned
thresholds/attrs/eval-config), `fl_v3/scripts/{t4_readiness_eval.py, run_t4_reference_a40.sh,
run_t4_readiness_eval_a40.sh}` (new), `fl_v3/scripts/_fl_env.sh` (`fl_stamp_supernodes` 4th-arg block),
`fl_v3/configs/{t4_reference.json, t4_mini_smoke.json}` (new), `fl_v3/collab/T4/{SPEC.md}`.
**Consume-only (unmodified):** T2 `models/fusion/**` (`detector.decode` — the single decode), T1
`data/nuscenes/**` (transforms/class_map/schema), T0 `utils/runtime.py`, the FL launcher pattern.
`fl_v2/` untouched.

## 3. The canonical→global conversion convention (the crown jewel — §0.1)

`convert_box_to_global(box7, lidar2ego, ego2global_lidar, velocity_lidar)`:
- **center** `t_global = (ego2global_lidar · lidar2ego) · [cx,cy,cz,1]` — `lidar2global = ego2global_lidar @ lidar2ego` (the forward chain the devkit `get_sample_data` inverts; verified line-by-line vs the devkit Box ops in the research digest).
- **size** `wlh = (w,l,h) = (dy,dx,dz)` — inverts T1's `wlh→(l,w,h)` (raw nuScenes `size` order; exact).
- **rotation** `R_box_global = R(lidar2global) · Rz(yaw)` → unit quaternion `(w,x,y,z)` via a deterministic Shepperd `rotmat_to_quaternion` (≡ the SPEC's `q_ego2global · q_lidar2ego · Quat(+z,yaw)`; the head emits a yaw-only box).
- **velocity** `v_global = R(lidar2global) · (vx,vy,0)` → keep `(vx,vy)` — the inverse ORDER of T1's `R(cs)ᵀ·R(ep)ᵀ` (`R(lidar2global)=R(ep)·R(cs)`); `v_z=0`.
- **float hygiene**: `detection_score` + every trans/size/rot coerced with builtin `float()`; no NaN/inf in trans/size/rot; `size>0`; `attribute_name` in the devkit vocab or `""`.

**Per-sample box order (permutation-invariance):** boxes are emitted sorted by the content-defined key
`(−score, translation, size, rotation, name)` (`detection_box_sort_key`) — because the devkit
`accumulate` breaks score ties on the emission index, a content-defined order makes mAP/NDS invariant
to the decode's emission order.

### 3a. A documented yaw-convention finding (negligible AOE floor; flagged, NOT a bug)
T1's canonical box7 yaw is the **Tait-Bryan `yaw_pitch_roll[0]`** (verified `== box7` to 4e-16), but
the devkit `DetectionEval` orientation error uses **`quaternion_yaw` (rotated-x-axis heading)**. On
tilted real boxes these differ by **up to ~0.0038 rad (~0.22°)** (measured on mini). So a *perfect*
`box_to_global(GT)` incurs a ~0.004-rad AOE floor — **negligible for NDS** (one of five TP scores, via
`1−min(1,AOE)`) and **irrelevant to AP** (center-distance) and to the **disappearance ASR** (detection
presence, not yaw). `box_to_global` cannot recover the heading from a scalar Euler yaw, and T1 is
frozen, so this is **documented + flagged, not fixed** (carrying full-3D orientation would be a T1
touch). See `findings_log`.

## 4. Pinned thresholds + attribute defaults (T4_SPEC §0.3 — in `pyproject.toml`)

| key | value | rationale |
|---|---|---|
| `asr-tau-pts` | **10** | LiDAR-support floor (`gt_num_lidar_pts ≥ 10`); `≥1` already subsumes the devkit `num_pts(lidar+radar)>0` GT filter for car (lidar≥10 ⇒ lidar+radar>0). |
| `asr-tau-clean` | **0.1** | clean-detect score floor = the production decode `det-score-threshold` (no eval/V4 re-threshold). |
| `asr-d-clean` | **2.0 m** | clean-match radius == the devkit `dist_th_tp` (the TP operating point). |
| `asr-image-h/w` | **900 / 1600** | native nuScenes resolution (the schema `lidar2img` is built for it). |
| `asr-n-min` | **150** | min eligible-clean-detected cars so a disappear-ASR ~0.3 has a non-degenerate count. |
| `asr-recall-floor` | **0.20** | floor on the **OFFICIAL** car recall (devkit `DetectionEval`, NOT the proxy). |
| `asr-false-disappear-max` | **0.02** | clean false-disappearance must be `< 0.02` AND valid only when `N ≥ N_min` (else UNDEFINED → gate FAILS). |
| `attr-vehicle-moving-speed` | **0.2 m/s** | ≥ → `vehicle.moving` else `vehicle.parked` (devkit vehicle attrs). |
| `attr-pedestrian-moving-speed` | **0.2 m/s** | ≥ → `pedestrian.moving` else `pedestrian.standing`. |
| `det-eval-config` | `detection_cvpr_2019` | the official eval config (`class_range` car=50m, `dist_ths=[.5,1,2,4]`, `dist_th_tp=2.0`, `max_boxes=500`, `mean_ap_weight=5`). |

**The denominator is `eligible-CLEAN-DETECTED`** = targets satisfying ALL 6 criteria *including* (5)
clean score ≥ τ_clean and (6) match ≤ d_clean. The frozen-subset size `N` == this count == the
disappearance-ASR denominator (asserted + tested: a criteria-1–4-but-undetected car is EXCLUDED).

## 5. Results

### 5a. Mini engineering smoke (harness validated — NOT a go/no-go; `scale=mini-smoke`)
- **Devkit-anchored round-trip** (≥200 mini boxes, `test_eval_box_to_global`): translation L2 `6e-13`,
  `wlh` exact (`0`), rotation lift-equivalence vs an independent devkit `Box` lift `|ΔR| 1e-15`, heading
  vs raw annotation worst `0.0037` rad (< 0.02; the documented convention floor), 18 near-±π boxes,
  74 moving boxes velocity-direction cos > 0.99997.
- **GT-as-pred AP≈1** (`test_eval_detection_eval`, mini_val): car AP@2m **1.0000**, car mean AP
  1.0000, ATE 0, AOE 0.0006, AAE 0, AVE 0.0001; every present class AP > 0.99 (absent → 0). The
  readiness-driver's trainval-scale GT-as-pred self-check also returns car AP@2m **1.0000**.
- **Permutation-invariance**: permuting equal-score boxes → byte-identical results JSON AND identical
  mAP/NDS through `DetectionEval`.
- **Full readiness pipeline** (`t4_readiness_eval.py` on mini_val, untrained resnet18 checkpoint):
  ran clean end-to-end — eligible-clean-detected **N=185** (criteria tally
  c1=2568/c2=2568/c3=1991/c4=2239/c5=c6=clean=185), **false-disappearance 0.0** (2nd fresh decode
  reproduced detections — determinism + batch-invariance), V4 rendered, **VERDICT NOT-READY
  (scale=mini-smoke)** with the correct gaps. Frozen-subset hash bound to the checkpoint checksum.

### 5b. Trainval scientific (the real mAP/NDS + the go/no-go)
- Reference run: `run_t4_reference_a40.sh` (job **6764630**, 15/15 rounds) — full participation
  (`fraction-train=1.0`, all N=25/round), log-group, v1.0-trainval, Path-A 4×A40, Swin-T → checkpoint
  `final_model.pt`, `FL_TRAINABLE_CHECKSUM = a80466c341b0e514773a6dc350e23f93f89a53d5da48a46a20b08d47bc1a090a`.
- Readiness eval: `run_t4_readiness_eval_a40.sh` → `benchmark_readiness.json`.

**Decode protocol of record: `batch_size=1`** (canonical per-sample inference; batch-invariant — see §5c).
The authoritative readiness eval (job **6765358** → `readiness_bs1/`):

| field | value |
|---|---|
| `scale` | `trainval-scientific` (v1.0-trainval / val, 6019 samples) |
| **VERDICT** | **READY** ✅ (gaps: none) |
| mAP / NDS | **0.1253 / 0.1688** (batch_size=1 canonical; the batch-16 eval gave 0.080 / 0.138) |
| **official clean car recall** | **0.85** — PASSES (floor 0.20) ✅ |
| **eligible_count `N`** | **27,432** — PASSES (N_min 150) ✅ |
| GT-as-pred sanity (trainval) | car AP@2m **1.0000** — conversion exact at scale ✅ |
| **false-disappearance** | **0.0** (defined, N≥N_min) — PASSES (<0.02) ✅ |
| attacked-checkpoint `FL_TRAINABLE_CHECKSUM` | `a80466c341b0e514773a6dc350e23f93f89a53d5da48a46a20b08d47bc1a090a` |
| frozen-subset content hash | `2ad8f8da55e8516bf0c46085cd5217ad2b2d1984c23499f51c397ad7cad1940f` (batch_size=1; the earlier batch-16 hash `d21c1f5b…` in `readiness/` is superseded) |

### 5c. The batch-invariance finding (why the decode protocol is `batch_size=1`)
The first readiness eval (batch-16) returned NOT-READY for ONE reason — false-disappearance 9.4%. Direct
diagnostic (`scripts/_t4_fd_diagnose.py`, 60 subset samples, 3 decode ways on the checkpoint): batch-16
re-run vs batch-16 = **0/60 differ (run-to-run determinism PERFECT)**, but batch-16 vs batch-1 = **28/60
differ** — the detector forward is **not batch-invariant** (cuDNN conv varies with batch composition →
boundary detections near τ_clean flip). The subset was built full-val-batched but re-checked
subset-batched → spurious "disappearances." So the 9.4% was a **harness artifact, not a model/determinism
defect** (the model is strong: recall 0.85, N=27,432). **Fix:** the whole ASR/readiness decode runs at
`batch_size=1` (a target's disappearance must depend only on its own trigger, not batch-mates) → one
consistent batch-invariant decode → false-disappearance = 0.0 → READY. **T5 inherits this protocol**
(decode triggered inputs at batch_size=1). Cost: batch_size=1 is GPU-idle/CPU-bound (~1–2 h for the
6019-sample eval) — a per-cell perf concern for T5–T7 that reinforces D11 (feature caching) or motivates a
batch-invariant-but-batched decode.

> **The model is strong (D10 worked): car recall 0.70 + N=23,354 both clear their floors with huge
> margin** (vs the T3 sampled-regime weak model). The lone batch-16 NOT-READY was the **false-disappearance
> 9.4%**, root-caused to **batch non-invariance** (the detector forward isn't perfectly batch-invariant —
> cuDNN conv varies with batch composition; run-to-run with the SAME batching is bit-identical (0/60), but
> batch-16-vs-batch-1 differs on 28/60 samples). The subset was built full-val-batched but re-checked
> subset-batched → spurious flips. **Fix:** the ASR/readiness decode runs at **batch_size=1** (canonical
> per-sample inference; a car's disappearance must depend only on its own trigger). **T5 inherits this
> protocol.** So the architecture-strengthening escalation (D10/D9) is **NOT** triggered — the model
> cleared the recall floor; only the harness needed the batch-invariant decode. See `findings_log`.

> **A100 (D9):** the A100 determinism gate PASSED (job 6764809: two same-seed runs byte-identical,
> `ae2b4571…`) → A100 unlocked for T5–T7; but A100 ≈ A40 speed (~1.2×, the workload is I/O/eval-bound),
> so feature caching (D11), not A100, is the real single-run speed lever.

### 5d. Codex review (CHANGES-REQUESTED → addressed)
Codex re-reviewed the build (no scientific-error / correctness-bug / metric / calibration / parity
issue; 24 tests pass). One **blocking invariant-violation** + one non-blocking question + one style nit:
- **(BLOCKING) provenance binding — FIXED.** The READY predicate was checksum-bound but not *provenance*-
  bound to the D10 regime, and the launcher only *warned* (buggily — `printf '%.0f'` rounds 0.9→1) on
  `fraction-train≠1.0`, so an overridden CONFIG/CKPT could emit READY for a sampled/IID checkpoint. Fix:
  (a) new tested `eval/provenance.py` (`build_provenance`/`check_d10`/`verify_d10_provenance`); (b) the
  reference launcher **hard-fails** any non-D10 config (task-type/version/splits/partition=log_group/
  defense=none/`fraction-train==1.0`) AND writes `provenance.json` beside `final_model.pt`; (c)
  `t4_readiness_eval.py` **hard-verifies** that provenance (bound to the recomputed checksum) before
  emitting any trainval verdict — so a sampled/IID/defended/wrong-split checkpoint can NEVER produce a
  valid go/no-go. `benchmark_readiness.json` now records `verified_d10_provenance`. The existing
  checkpoint's provenance was backfilled from its authentic `t4_reference.json` and **verifies**. (+7 tests.)
- **(non-blocking) yaw tolerance contract** — see §3a; the durable `T4_SPEC §0.1` wording (`<1e-4`) vs the
  evaluator-heading `<0.02` is a documentation alignment for the **orchestrator** to land in the durable
  contract (build session does not edit the orchestrator's `T4_SPEC.md`; fully documented here + findings).
- **(style)** trailing whitespace removed.

## 6. Invariants (must hold; Codex checks each)
- Canonical→global anchored to the **raw devkit annotation** (not a self-inverse of T1) — round-trip
  on ≥200 boxes + GT-as-pred AP≈1 vs the devkit's own `load_gt`. ✓ (mini)
- **One decode, two consumers:** `DetectionEval` + V4 + ASR all consume the SAME `decode_eval_set`
  output (same thresholds, no V4 re-threshold). ✓
- **Determinism:** decode, conversion, results JSON (sorted tokens + content-defined box order),
  eligibility, and the frozen-subset hash are bit-reproducible; `DetectionEval` permutation-invariant
  on equal-score ties. ✓
- **ASR eligibility + denominator = §Attack spec (§0.4):** 6 criteria; `gt_in_range` = devkit
  `ego_dist`; `τ_pts≥1` subsumes the devkit `num_pts>0`; bike-rack re-derived via the devkit for
  bicycle/motorcycle (car-irrelevant); **`N` = eligible-clean-detected** (asserted + tested). ✓
- **False-disappearance < 0.02 with N ≥ N_min** (else UNDEFINED, gate fails); ASR defined only on
  triggered inputs. ✓
- **Readiness bound to the full-participation log-group checkpoint (§0.2) + floored (§0.3).** ✓ (READY;
  checkpoint `a80466c3…`, recall 0.85 > 0.20, N=27,432 ≥ 150, false-disappear 0.0; batch_size=1 decode).
  Now also **PROVENANCE-bound** (§5d): `provenance.json` hard-verified as D10 (full-participation
  log-group trainval clean) before any verdict; the launcher hard-fails a non-D10 config.
- **Frozen subset hashed + bound** (re-verified at load, reused by T5); built on full `val`. ✓
- **Mini vs trainval boundary:** harness validated on mini (engineering); the real mAP/NDS + readiness
  are trainval-`val`-scale (6019 samples), `scale`-stamped; a mini verdict is NOT a go/no-go. ✓

## 7. Self-review — what to attack hardest (for Codex)
1. **§0.1 — `box_to_global` anchored to the raw devkit annotation** (not a self-inverse of T1) + the
   GT-as-pred AP≈1 vs devkit `load_gt`. Scrutinize the **yaw-convention finding (§3a)**: T1=Euler vs
   devkit=heading, the ~0.004-rad AOE floor — is "negligible + flagged, not fixed" the right call, or
   should T1 carry heading-yaw / full-3D orientation? The round-trip test asserts heading < 0.02 rad
   (gross-bug catch) + a tight lift-equivalence < 1e-9 (geometry); is that the right rigor split?
2. **The velocity inverse** — order `R(ep)·R(cs)`, `v_z=0`; the test checks **direction** (cos > 0.999
   on moving boxes), not just norm. Is the AVE caveat acceptable?
3. **§0.4 the eligible-CLEAN-DETECTED denominator** — `N` excludes criteria-1–4-but-undetected targets
   (the greedy `detected_target_gt` matcher; the exclusion test). Is the greedy one-to-one matching
   over ALL target GT (vs only criteria-1–4 GT) the right denominator semantics?
4. **§0.2/§0.3 readiness = READY, bound to the log-group full-participation checkpoint `a80466c3…`** —
   verdict honest? floors defensible (`N_min=150`, `recall_floor=0.20` on devkit car recall, NOT the
   proxy)? checkpoint-checksum binding correct? Note: the gate FIRST returned NOT-READY (false-disappear
   9.4%) and was NOT floor-lowered — it was root-caused to batch non-invariance and fixed (below), then
   re-judged READY. The architecture-strengthening escalation was correctly NOT triggered (recall cleared).
5. The frozen-subset hash bound to `(targets, thresholds, checkpoint checksum)` + the load-time
   re-verify (the T5 reuse contract). The of-record subset/hash is the **batch_size=1** one (`2ad8f8da…`).
6. **The batch-invariance finding + the `batch_size=1` decode protocol (§5c)** — the detector forward is
   not batch-invariant (cuDNN; run-to-run with the SAME batching is bit-identical, batch-16≠batch-1 on
   28/60 samples). Is `batch_size=1` (canonical per-sample inference) the right of-record protocol for the
   ASR + the official metric, and is the T5-inherits-this contract clearly the right call (vs a
   batch-invariant-but-batched decode)? Does this interact with the T3 determinism gate (which only checks
   same-batching byte-identity, not batch-invariance)?

## 8. GATE status
- [x] Devkit-anchored round-trip + GT-as-pred AP≈1 (mini).
- [x] Stable mAP/NDS + permutation-invariance via `.evaluate()` (mini); **real trainval mAP/NDS = 0.125/0.169** (batch_size=1, job 6765358).
- [x] Evaluator + V4 share the SAME decode; V4 TP/FN agrees with the metric incl. boundary boxes.
- [x] ASR harness: 6 criteria, `N` = eligible-clean-detected (+ exclusion test), frozen subset
      content-hashed + bound + load-time re-verify, false-disappearance gate (defined + UNDEFINED paths).
- [x] **Benchmark-readiness verdict = READY** on the full-participation log-group checkpoint `a80466c3…`
      (recall 0.85 > 0.20, N=27,432 ≥ 150, false-disappear 0.0; `scale=trainval-scientific`; batch_size=1);
      **PROVENANCE-verified D10** (§5d — the launcher hard-fails non-D10; the eval refuses an unverified checkpoint).
- [x] 6-tuple schema frozen (clean cols filled by the readiness driver; poisoned/ASR/defense reserved); no attack invoked.
- [x] Tests green — **198 passed** (T0–T3's 167 + 31 T4: box_to_global 3, detection_eval 4, asr 8, frustum 4, report 3, viz_detection 2, **provenance 7**); the trainval reference + readiness evals are A40 SLURM jobs (not pytest).
- [x] `collab/T4/SPEC.md` filled (this doc) + `findings_log` appended; 5 Codex-flag items above.
