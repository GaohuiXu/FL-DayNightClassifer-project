# T4 — SPEC: detection eval (official mAP/NDS) + utility/ASR metric harness + V4

Plan: `../../roadmap/cycle_04_fusion_layer_backdoors.md` (task **T4**; §Attack spec "ASR-eligibility",
"Evaluation protocol & splits", §Defense protocol "utility/ASR 2×2", Viz **V4**). Decisions:
`../decisions.md` — **D4** (disappearance ASR primary, phantom secondary), **D8** (car = primary target),
**D7** `δ` (DEFERRED — T4 produces the clean NDS baseline that later informs it), **D9** (compute tiers
for the reference run). Contract for the **T4 build session**. Fill `fl_v3/collab/T4/SPEC.md`.

> **T4 builds the METRIC + ELIGIBILITY + EVAL HARNESS layer** — the official nuScenes `DetectionEval`
> (mAP/NDS) on the platform's decoded boxes, plus the strict 6-criterion ASR-eligibility machinery + the
> frozen held-out ASR subset + the disappearance metric + the false-disappearance baseline + the
> denominator-N + the **benchmark-readiness gate**. **There is no attack yet** (the trigger/poisoning is
> **T5**) — T4 builds + validates the ASR machinery on **clean** data so T5 can plug a trigger in. The
> ASR/eligibility *semantics* and the task scoping were verified correct against the §Attack spec (a
> 5-agent pass, workflow `wf_30f736bf-480`, all-confirm-ok on semantics + plan coverage). **The hardening
> below is about the GATE — the draft was full of self-certifying / undeclared-threshold / partition-
> mismatch holes that would let a literal build certify a meaningless benchmark. Read §0 first.**

---

## 0. CRITICAL — the harness must be anchored to the devkit, bound to the attacked model, and floored

The §Attack-spec semantics are right; the danger is a GATE a literal build session passes on a
meaningless config. Five non-negotiables:

1. **Anchor the canonical→global conversion to the RAW devkit annotation, NOT to T1's forward.** Defining
   `box_to_global` as "the exact inverse of T1's forward" makes the round-trip **self-certify** a shared
   yaw-sign / `wlh`-order / ego-pose bug (the T1-viz / T2-BEV-convention trap). The independent oracle is
   the **raw nuScenes global `sample_annotation`** (translation / size=`wlh` / rotation), reconstructed
   via the **devkit `Box` class**. **GATE:** `box_to_global(T1-canonical GT)` reproduces the raw devkit
   global annotation on **≥200 real mini boxes** (translation L2 <1e-3 m, `wlh` exact, `|wrap(Δyaw)|<1e-4`
   on **yaw-only-flattened** boxes incl. ±π, velocity **direction** match). *(Yaw-tolerance note: the
   `<1e-4` rad gate applies to **yaw-only-flattened** boxes — the geometry check. On the **raw devkit
   annotation** of real *tilted* boxes the heading tolerance is `<0.02` rad, because T1 stores the
   Tait-Bryan `yaw_pitch_roll[0]` Euler yaw while the devkit `DetectionEval` AOE uses `quaternion_yaw`
   (rotated-x heading); the two differ by up to ~0.004 rad on tilted boxes — a documented, **negligible**
   AOE floor, irrelevant to AP / disappearance-ASR and ≈0 for NDS. See `collab/T4/SPEC.md §3a`. This is
   the intended contract, NOT an accidental gate weakening.)* **AND** the `DetectionEval`
   GT-as-pred sanity: the **GT side is the devkit's own `load_gt(nusc, eval_set, DetectionBox)`** and the
   **pred side is `box_to_global(T1-canonical GT)`** → **per-class AP ≈ 1** (NDS<1 unless attr+vel copied;
   see §2). Forbid wiring the round-trip against `info_cache`'s own forward.
2. **Bind readiness to the FULL-PARTICIPATION (D10) LOG-GROUP checkpoint T5 actually attacks (by
   checksum).** The threat-model partition is the trainval **log-group**, NOT IID. **Per D10, the
   reference is trained at FULL participation** (`fraction-train=1.0`, all `N`/round) — **NOT** the T3
   5-of-25 sampled run. **The T3 numbers `recall@2m 0.1455` and the non-IID gap `+0.2073` are
   sampled-regime (5-of-25, 4-round) and are RE-LABELED `scale=trainval-sampled` — they are NOT the
   readiness anchor.** **GATE (binding):** the reference clean checkpoint whose mAP/NDS + `eligible_count`
   + `clean_car_recall` feed `benchmark_readiness.json` **MUST be the full-participation log-group trainval
   FedAvg checkpoint, identified by its `FL_TRAINABLE_CHECKSUM`** (recorded in the readiness JSON + handed
   to T5). A verdict computed on the IID checkpoint, or on a sampled (`fraction<1`) checkpoint, is
   **INVALID** as the T5 gate (a sampled anchor + a full-participation attack is the partition-mismatch
   trap generalized to a participation-regime mismatch).
3. **Pin defensible numeric floors in config (tied to attack measurability), not "declared by the build
   session."** Undeclared `N_min`/`recall_floor`/`τ_clean`/`τ_pts`/`d_clean`/false-disappear-threshold =
   a rubber stamp. Pin, with a written rationale linking each to ASR statistical power:
   **`N_min ≥ 150`** eligible-clean-detected cars (so a disappear-ASR ~0.3 has a non-degenerate count);
   **`recall_floor` on OFFICIAL car recall/AP** (recommend **> 0.20**), **NOT the proxy**; `d_clean = 2.0`
   m; `τ_clean` = the production decode score threshold; `τ_pts` = a declared LiDAR-support floor;
   **false-disappearance "near-zero" = `< 0.02`** AND only valid when the denominator `≥ N_min`. **The
   strict readiness sequence (D10, the BLOCKER if mis-ordered):** **(a)** re-train the clean log-group
   trainval FedAvg at **FULL participation FIRST** (the T3 0.1455 was 5-of-25 + 4 rounds — under-trained,
   not a capacity ceiling: the same model hits 0.35–0.50 IID); **(b)** re-judge readiness on **that**
   checkpoint; **(c)** escalate to **architecture strengthening (deeper LiDAR PFN / full-model on A100 per
   D9) ONLY if STILL NOT-READY.** Full participation must NOT paper over a real PFN weakness — if it clears
   the floor, stamp the verdict **"cleared by participation, architecture not independently validated"**;
   if it does not, that IS the architecture signal. Report the 5-of-25 and the full-participation recall
   **side by side**. Do not lower a floor to pass.
4. **The denominator is `eligible-CLEAN-DETECTED`, not `all-eligible`.** **GATE:** `N` (the frozen-subset
   size AND the ASR denominator) == count of targets satisfying **all 6** criteria *including* (5) clean
   score ≥ `τ_clean` and (6) clean match within `d_clean`; a test constructs a criteria-1–4 target that is
   NOT clean-detected and asserts it is **excluded** from `N`.
5. **The frozen ASR subset is content-hashed AND bound to the checkpoint + thresholds.** **GATE:** the
   subset is serialized with a hash over `(sorted sample_token, sorted ann_token, the declared thresholds,
   the clean-checkpoint FL_TRAINABLE_CHECKSUM)`; the hash + checksum are recorded in `collab/T4/SPEC.md`;
   a **load-time assertion (reused by T5)** recomputes the hash and **refuses to run if it differs** — so
   every T5/T6/T7 cell scores the identical targets. Built on the **full val split**, never a
   `det-eval-limit` prefix.

> **Orchestrator notes:**
> 1. **Metric upgrade.** T2/T3's `center_distance_proxy` (recall@2m) was the fast in-loop FL signal,
>    explicitly "NOT the T4 DetectionEval" (`tasks.py:362`). **T4 builds the official `DetectionEval`
>    (mAP/NDS + per-class AP + the 5 TP errors).** The proxy stays in-loop; official mAP/NDS is the T4
>    utility metric and the thing the §Defense 2×2 rule (T7) judges.
> 2. **DT4-A is the strategic go/no-go for T5** (per §0.2/§0.3) — T4 *reports* readiness; model-strengthening
>    is a *separate* decision the orchestrator/user resolves on T4's numbers.

---

## 1. Scientific intent

Turn the platform's decoded detections into **scientifically valid utility + attack-success metrics**:
the official nuScenes center-distance **`DetectionEval` (mAP/NDS)** for clean (later poisoned) utility, and
the **strict ASR harness** — the 6-criterion eligibility, the frozen held-out eligible-car subset, the
disappearance metric, the false-disappearance baseline, the denominator-N — built on the **same decoded
boxes** the evaluator scores, anchored to the **devkit** (not the platform's own forward), made
inspectable by **V4**, and **bound to the exact log-group checkpoint T5 attacks**. T4 certifies the
harness on **clean** data (false-disappearance near-zero; ASR is *defined only for triggered inputs*) and
emits an honest **benchmark-readiness verdict** — whether the clean model is strong enough to host a
measurable disappearance attack. No trigger, no defense study (T5/T6). This is the metric contract every
later attack×defense cell is reported in.

## 2. Scope

**In scope (deliver):**

- **`eval/box_to_global.py`** — decoded box (T1 canonical `LIDAR_TOP` `(cx,cy,cz,dx=l,dy=w,dz=h,yaw)` +
  score + class) → nuScenes **global** submission `DetectionBox`. **Anchored to the raw devkit annotation
  (§0.1), reusing `data/nuscenes/transforms.*`** (so the verified forward is inherited; only the inverse
  is new):
  - center: `T_global←lidar = T_global←ego(t_lidar) · T_ego←lidar` (pin `lidar2global = ego2global_lidar ·
    lidar2ego`).
  - **`size = (w,l,h) = (dy,dx,dz)`** (raw nuScenes `wlh` order — invert T1's `wlh→(l,w,h)`).
  - **rotation (global quaternion)** = `q_ego2global_lidar · q_lidar2ego · Quat(axis=+z, angle=yaw)`
    (round-trips for the yaw-only box the head emits).
  - **velocity** = `R(ego_pose) · R(cs_lidar) · (vx_l, vy_l, 0)` — the **inverse ORDER** of T1's
    `R(cs)ᵀ·R(ep)ᵀ`; keep `(vx,vy)`. **`v_z=0`** (T1 dropped lidar `v_z`; the head emits 2D velocity
    anyway) — document the AVE caveat (worst ~0.13 m/s; *if* AVE materially hurts NDS, carrying full-3D
    velocity is a T1 touch — flag it, do NOT silently change T1).
  - **float hygiene:** coerce `detection_score` + every box float with **builtin `float()`** (the devkit
    asserts `type(detection_score)==float` exactly, and **no NaN/inf** in translation/size/rotation; assert
    `size>0`). `attribute_name` per the pinned per-class default (below).
- **`eval/detection_eval.py`** — run `nuscenes.eval.detection.evaluate.DetectionEval`
  (`config_factory("detection_cvpr_2019")`) → mAP, NDS, per-class AP, the 5 TP errors. **Invocation
  details (verified):** call **`DetectionEval(...).evaluate()` directly** (NOT `main()` — `main()` does
  `random.seed/shuffle` + PNG/PDF/json writes); pass a writable `output_dir` (it is always created + gets
  the metrics json); `render_curves=False`. **`eval_set` is hard-coupled to the NuScenes version** — a
  **scale table:** mini → `v1.0-mini` `eval_set="mini_val"` (smoke); trainval → `v1.0-trainval`
  `eval_set="val"` (science); assert the version↔split match up front. **Deterministic results JSON:**
  emit `{meta, results:{sample_token:[box,...]}}` with **every eval token a key (empty `[]` if no
  detections)**; **boxes in a content-defined deterministic order** (sort by `-score` then a stable
  tiebreaker — `translation` tuple / decode index), because `accumulate()`'s score sort breaks ties by
  index and is order-sensitive (a stable score sort alone is insufficient).
- **`eval/frustum_visibility.py`** — ASR criterion (2): GT box visible in ≥1 camera frustum (project the
  8 corners via per-cam `lidar2img`; in-image + in-front). Deterministic; reuses `transforms.project_to_image`.
- **`eval/asr.py`** — the ASR machinery (D4 disappearance primary):
  - **Eligibility (6 criteria, §Attack spec; all thresholds pinned in config — §0.3):** (1) class = **car**
    (D8); (2) frustum-visible (criterion (2)'s "triggered camera view" is realized as trigger-invariant
    frustum geometry, so eligibility is identical clean vs triggered — T5 inherits the same set);
    (3) LiDAR support `gt_num_lidar_pts ≥ τ_pts`; (4) range `gt_in_range` (devkit `ego_dist < class_range`);
    (5) **clean** model detects at score ≥ `τ_clean`; (6) that clean detection matches GT within `d_clean`.
    **Also apply the devkit `filter_eval_boxes` GT filters as the source of truth** (`num_pts>0` — note
    devkit `num_pts = num_lidar + num_radar`, so it is **not** subsumed by the lidar-only `τ_pts`; and the
    **bike-rack filter for bicycle/motorcycle only** — re-derive via the devkit, benign for car).
  - **The frozen held-out ASR subset** = eligible-clean-detected cars on the held-out **`val`** split,
    content-hashed + bound to the checkpoint checksum (§0.5), **frozen before benchmarking**. `N` = its size.
  - **The disappearance metric** (`ASR = disappeared / eligible-clean-detected`, **N reported**; the
    denominator is exactly the §0.4 eligible-clean-detected count): *applied to triggered inputs at T5*;
    T4 implements + validates on clean.
  - **The false-disappearance baseline:** on **clean/no-trigger** inputs, the fraction of
    eligible-clean-detected cars the model later misses with no trigger — **`< 0.02` AND valid only when
    the denominator `≥ N_min`** (an empty/`<N_min` denominator → **reported UNDEFINED, gate FAILS** — not
    vacuously near-zero). A baseline > threshold = a harness/determinism bug.
  - **Phantom-ASR (D4 secondary):** the reporting slot + metric definition only; injection is T5.
- **DT4-A — the reference clean checkpoint + benchmark-readiness (§0.2/§0.3; per D10):** a converged-enough
  **FULL-participation (`fraction-train=1.0`, all `N`/round) log-group trainval clean FedAvg** (≤20 rounds
  is a floor, not a target — run until the readiness `recall_floor` is cleared or declared NOT-READY).
  **Build a `t4_reference.json` + launcher** (fraction-train=1.0, `nuscenes-partition-mode=log_group`,
  v1.0-trainval, derived N, `num-server-rounds` bumped to a convergence target — NOT the T3 placeholder 4)
  and run it via **D9 Path-A** multi-GPU for wall-clock (~3–6 h on a 4-GPU A40 node) — this needs
  generalizing `fl_stamp_supernodes` to stamp the `local-simulation-gpu-4x` federation (it currently
  stamps only `local-simulation-gpu`; `-4x` hardcodes `num-supernodes=8`), or falling back to single-GPU.
  Hold batch=16/seed/partition/full-val eval (`det-eval-limit=0`) fixed; change only fraction-train +
  rounds; re-assert same-seed byte-identity (a new checksum is expected). Emit **`benchmark_readiness.json`**:
  `mAP`, `NDS`, `eligible_count`, official
  `clean_car_recall`, the pinned `N_min`/`recall_floor`, the attacked-checkpoint `FL_TRAINABLE_CHECKSUM`,
  `scale=trainval-scientific`, and `READY` iff `eligible_count ≥ N_min` AND `clean_car_recall > recall_floor`
  (else `NOT-READY` + the gap + the recommended strengthening). **A verdict computed on mini_val (2 scenes)
  is `scale=mini` and is explicitly NOT a go/no-go.**
- **`viz/detection.py` (V4):** cam + BEV with **GT vs clean-decoded boxes**, per-class score maps, the
  per-target score table — rendering **the EXACT box set the evaluator scores** (same `score_threshold`,
  `max_objects`; **no V4-only threshold/sort/limit**). **GATE:** V4 visual detection/miss agrees with the
  metric TP/FN on sampled cases **including the lowest-scoring eligible boxes near `τ_clean`/`d_clean`**
  (not a high-score-only sample); the disappeared/phantom highlights are wired but exercised at T5.
- **The 6-tuple reporting schema** (`eval/report.py`): clean mAP/NDS · poisoned mAP/NDS · disappear-ASR ·
  phantom-ASR · **ASR denominator N** · utility-collapse status. T4 fills the **clean** columns + freezes
  the schema (reserve a defense-decision-stats column for T6/T7 so T7 need not re-freeze it).
- **Tests** `fl_v3/tests/test_eval_*.py`; the pinned thresholds + per-class **attribute defaults**
  (car/truck/bus/trailer/CV → `vehicle.{moving,parked}` by a velocity threshold; pedestrian →
  `pedestrian.{moving,standing}`; cone/barrier → `""`) in `pyproject.toml`; a reference-eval SLURM script;
  `collab/T4/SPEC.md` + `findings_log.md`.

**Out of scope / deferred:** the trigger / poisoning operators / the 5-condition fusion-awareness ablation
+ V5 (**T5**); the defense suite + per-module gradient logging + V6 (**T6**); the attack×defense matrix +
the Q2 analysis + the §Defense 2×2 *verdict* (**T7**); setting D7 `δ`. **T4 invokes no attack.**

**Files created/changed:** `fl_v3/src/fl_v3/eval/**` (new), `fl_v3/src/fl_v3/viz/detection.py` (V4),
`fl_v3/tests/test_eval_*.py`, `pyproject.toml` (the pinned thresholds + attribute defaults + `eval_set`),
a reference-eval SLURM script, `collab/T4/SPEC.md`. **Consume-only (unmodified):** T2 `models/fusion/**`
(use `detector.decode` — no second decode path), T1 `data/nuscenes/**` (transforms/class_map/schema —
frozen; if v_z is needed, flag it, don't mutate T1), T0 `utils/runtime.py`. `fl_v2/` untouched.

## 3. Invariants (must hold; Codex checks each)

- **Canonical→global anchored to the devkit (crown jewel — §0.1):** `box_to_global(T1-canonical GT)`
  reproduces the **raw devkit global annotation** (≥200 real boxes; translation <1e-3 m, `wlh` exact,
  `|Δyaw|<1e-4` on yaw-only-flattened incl. ±π, velocity **direction**) — NOT a self-inverse of T1's
  forward. The GT-as-pred AP≈1 sanity uses the **devkit `load_gt`** as the GT side.
- **One decode, two consumers:** the `DetectionEval` results JSON and V4 are produced from the **same**
  `detector.decode` output (same thresholds; no divergent re-decode, no V4-only re-threshold) — the plan's
  "evaluator + visualization consume the SAME decoded boxes."
- **Determinism:** decode (T2), the conversion, the results JSON (sorted tokens + content-defined box
  order), the eligibility computation, and the frozen-subset hash are bit-reproducible; `DetectionEval`'s
  greedy matching is **permutation-invariant on equal-score ties** (tested by permuting equal-score boxes
  → identical mAP/NDS), not merely same-process re-runnable.
- **ASR eligibility + denominator = §Attack spec exactly (§0.4):** all 6 criteria; `gt_in_range` = devkit
  `ego_dist`; devkit `num_pts>0` applied as source of truth (radar-inclusive) with `τ_pts` an extra lidar
  floor; bike-rack for bicycle/motorcycle; **`N` = eligible-clean-detected** (asserted + tested).
- **False-disappearance < 0.02 with denominator ≥ N_min** (else UNDEFINED, gate fails); ASR defined ONLY
  on triggered inputs.
- **Readiness bound to the log-group checkpoint (§0.2) + floored (§0.3):** `benchmark_readiness.json`
  records the attacked checkpoint's checksum, the pinned `N_min`/`recall_floor`, `scale=trainval-scientific`,
  and the `READY`/`NOT-READY` verdict.
- **Frozen subset hashed + bound (§0.5):** load-time hash re-verification (reused by T5); built on full val.
- **Mini vs trainval boundary:** harness validated on mini (engineering); the real mAP/NDS + readiness are
  trainval-`val`-scale (≥ a declared scene/sample floor, e.g. ≥75 scenes / ≥3000 samples), `scale`-stamped;
  a mini verdict is NOT a go/no-go.
- **No false oracle / no attack:** correctness = the devkit-anchored round-trip + GT-as-pred AP≈1 +
  shared-decode + the floored readiness + near-zero (defined) false-disappearance.

## 4. Reference (ground truth for the review)

- **Official evaluator (verified internals):** `nuscenes.eval.detection.evaluate.DetectionEval`
  (`.evaluate()` not `main()`), `config_factory("detection_cvpr_2019")`, `nuscenes.eval.common.loaders`
  (`load_gt`, `load_prediction`, `filter_eval_boxes` — the 3 GT filters: `ego_dist<class_range`,
  `num_pts(lidar+radar)>0`, bike-rack), `nuscenes.eval.detection.data_classes.DetectionBox` (the submission
  schema; `type(score)==float`, no-NaN trans/size/rot, `size=wlh`, `attribute_name` in the class vocab or
  `""`), `nuscenes.eval.detection.algo.accumulate` (index-tie score sort — hence the deterministic
  box-order requirement).
- **Platform seams:** `models/fusion/detector.decode(...)→List[dict]` (the single decode);
  `training/tasks.center_distance_proxy` (the greedy center-distance primitive to reuse for criteria 5/6);
  the **log-group** trainval reference checkpoint (T3 `final_model.pt`, self-contained, `strict=True`;
  identified by `FL_TRAINABLE_CHECKSUM`).
- **T1 (the verified forward to invert, not to self-check against):** `data/nuscenes/transforms.*`,
  `class_map.{category_to_detection_name, class_range_for}`, `conventions.md §3/§4/§6` (yaw MINUS-cross-term,
  `wlh→(l,w,h)`, lidar-frame velocity + NaN→0, the bike-rack note), `info_cache` (the raw global annotation
  `_raw_center_g/_raw_wlh/_raw_q_g` — the **independent oracle** for §0.1).
- **§Attack spec** (6 criteria, denominator-N, false-disappearance, splits/no-leakage), **D4/D8/D9**.

## 5. Scientific failure modes to check (point Codex here)

- **Self-certifying round-trip** (§0.1) — `box_to_global` inverting T1's own forward passes a shared
  yaw/`wlh`/ego bug; anchor to the raw devkit annotation + `load_gt`.
- **Velocity inverse** — wrong rotation **order** (`R(ep)·R(cs)`, not the reverse) and the dropped `v_z`
  (document AVE); check **direction**, not just norm.
- **Submission-format crashes** — a **numpy** score (devkit asserts builtin `float`), NaN/inf in
  trans/size/rotation, `size≤0`, an `attribute_name` outside the class vocab (`""` rejected/penalized for
  car → tanks AAE/NDS).
- **DetectionEval nondeterminism** — `main()`'s `random.shuffle`/file writes; the index-tie score sort →
  need a content-defined box order; `eval_set`↔version mismatch (`AssertionError`).
- **The undeclared-threshold rubber stamp (§0.3)** — `N_min`/`recall_floor`/`τ_*` set so the weak model
  passes READY; `recall_floor` on the lenient proxy instead of official car recall.
- **Partition mismatch (§0.2)** — readiness on IID (0.35) while T5 attacks log-group (0.15).
- **Empty-set vacuity** — false-disappearance 0/0→0 and readiness floors passing on an empty eligible set.
- **Denominator = all-eligible** instead of eligible-clean-detected (§0.4).
- **Unfrozen / unbound ASR subset** (§0.5) — T5/T6 cells scoring different targets; a `det-eval-limit`
  prefix leaking into subset construction.
- **V4 re-thresholds** the shared decode so the visual diverges from the metric.
- **Mini masquerading as science** — mini_val (2 scenes) mAP/NDS or readiness reported as real.

## 6. GATE (objective pass criteria — plan's T4 gate, made objective)

- [ ] **Devkit-anchored round-trip + GT-as-pred sanity (§0.1):** `box_to_global(T1-canonical GT)` ==
      raw devkit global annotation on ≥200 mini boxes (trans <1e-3, `wlh` exact, yaw-only `|Δyaw|<1e-4`
      incl. ±π, velocity direction); `DetectionEval` with **GT side = devkit `load_gt`**, pred side =
      `box_to_global(T1 GT)` → **per-class AP ≈ 1** (fails if mAP < 0.99); velocity direction → AVE≈0 on
      the sanity.
- [ ] **Stable mAP/NDS on a fixed checkpoint:** `DetectionEval` deterministic (same ckpt → same mAP/NDS;
      equal-score-box permutation → identical mAP/NDS); via `.evaluate()` not `main()`; harness validated
      on mini; the **real mAP/NDS on trainval `val`** (≥ the declared scene/sample floor), `scale`-stamped.
- [ ] **Evaluator + V4 share the SAME decoded boxes** (same thresholds, no V4 re-threshold); **V4 agrees
      with metric TP/FN on sampled cases incl. the lowest-scoring eligible boxes**; V4 renders + manifest.
- [ ] **ASR harness:** 6-criterion eligibility; **`N` = eligible-clean-detected** (asserted + the
      criteria-1–4-not-detected exclusion test); the **frozen subset content-hashed + bound to the
      checkpoint checksum** (§0.5), built on full `val`, with a T5-reused load-time re-verify; `ASR =
      disappeared / eligible-clean-detected` with N reported; **false-disappearance < 0.02 with denominator
      ≥ N_min** (UNDEFINED otherwise).
- [ ] **Benchmark-readiness (DT4-A, §0.2/§0.3):** `benchmark_readiness.json` on the **log-group** trainval
      checkpoint (recorded `FL_TRAINABLE_CHECKSUM`), pinned `N_min ≥ 150` + `recall_floor` on **official**
      car recall (> 0.20) with rationale; `READY`/`NOT-READY` verdict (`scale=trainval-scientific`) — the
      explicit T5 go/no-go (NOT-READY is a valid surfaced outcome that gates T5).
- [ ] **6-tuple schema** frozen (clean cols filled; poisoned/ASR cols T5+; defense-stats col reserved); no
      attack invoked.
- [ ] **Tests green:** `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` (T0–T3's 167 + new
      T4 tests); record the count; note which require an A40 SLURM job (the trainval reference eval).
- [ ] **`collab/T4/SPEC.md` filled** (the conversion convention + the **pinned** `τ_pts`/`τ_clean`/`d_clean`/
      `N_min`/`recall_floor`/false-disappear threshold + attribute defaults + the mAP/NDS + readiness
      numbers + the frozen-subset hash + the attacked-checkpoint checksum) + `findings_log.md`; 2–3
      least-certain items flagged for Codex.

## 7. Self-review — to be filled by the build session
(Predicted hardest review targets: (a) **§0.1** — `box_to_global` anchored to the **raw devkit
annotation** + the GT-as-pred AP≈1 against devkit `load_gt` (not a self-inverse of T1); (b) the velocity
inverse **order** + the `v_z=0`/AVE caveat; (c) **§0.4** the denominator being eligible-clean-detected;
(d) **§0.2/§0.3** readiness bound to the log-group checkpoint with **pinned** floors on **official** recall —
honestly reporting whether the weak clean model can host a measurable attack; (e) the frozen-subset hash
bound to the checkpoint. Point Codex at the devkit-anchored round-trip, the readiness JSON + its checksum,
and the eligible-clean-detected denominator test.)
