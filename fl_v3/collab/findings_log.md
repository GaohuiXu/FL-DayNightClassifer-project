# Cycle 04 — Findings Log (fl_v3)

Running, append-only log of triaged decisions and resolved review findings across
tasks T0–T7. One entry per resolved item.

Format:
```
## [T<N>] <date> — <short title>
- **Finding (severity):** …
- **Decision/fix:** …
- **Rationale:** …
```

---

## [T2] 2026-06-16 — Codex RE-REVIEW of T2 (verdict PASS — review loop closed)
- **Finding (review outcome):** Codex re-reviewed commit `387f3dd` (the canonical over-cap pillar
  truncation fix) and returned **PASS** — "nothing found" in every severity category
  (scientific-error / correctness-bug / invariant-violation / question / style). No further
  scientific-correctness changes requested; no code changes made.
- **Decision/fix:** nothing to triage. Codex **independently verified** the fix: re-ran its old minimal
  over-cap repro → now `torch.equal=True`, `max|Δ|=0.0`; `test_pillar_scatter_permutation_invariant_OVERCAP`
  → 1 passed; the determinism file → 6 passed; the **full suite → 148 passed**. It re-confirmed (nothing
  found) the LSS composite sort, CenterPoint decode convention, corrected `gaussian_radius`, PointPillars
  deterministic scatter, the single BEV `W→x/H→y/flat=row·W+col` convention + resize/`lidar2img` rescale +
  encode→decode yaw/dim, the ResNet BN freeze-through-`train()`, and `dummy_regression` byte-identity.
- **Residual note (not a finding):** Codex could not independently re-run the **A40 SLURM gate** (it
  requires the A40), so the build-session job `6763843` / checksum
  `0a30410a9905010bd94d78959d89ee7f1fb05a116d28c75dec2667ef81af98e9` remains the authoritative production
  bit-identity evidence — by design (Codex does not submit SLURM jobs).
- **Rationale:** T2 (the deterministic BEVFusion-class model + detection loss/decode + V2/V3) is
  scientifically signed off; the build+review loop is closed. Next is the orchestrator marking T2 done and
  issuing T3 (the real Ray FedAvg run, which depends on T2's bit-determinism).

## [T2] 2026-06-16 — Codex REVIEW of T2 (CHANGES-REQUESTED → resolved)
- **Finding (invariant-violation, Codex; the one finding — all other categories "nothing found"):** the
  PointPillars encoder was NOT input-permutation-invariant once a pillar EXCEEDS `max_points`. It sorted
  only by `pillar_key` (stable), so within-pillar ties kept the incoming/file order, and `cap = within <
  max_points` then kept the **first `max_points` in file order** → a LiDAR point permutation selects a
  different subset → a different PFN/max-pool → a different BEV canvas. Codex reproduced it (1 pillar, 3
  points, `max_points=2`: `torch.equal==False`, `max|Δ|≈1.97e-5`). The existing test masked it with
  `max_points=128` (no truncation). This contradicted the SPEC's "pillar scatter byte-identical under
  input permutation" claim and the T2 contract's permutation-invariance requirement.
- **Decision/fix (Codex's recommended minimal fix):** make the within-pillar truncation **canonical** —
  `lidar_encoder.py` now lexicographically sorts `(pillar_key, x, y, z, intensity)` via successive STABLE
  sorts (least-significant first), so the kept `max_points` are a pure function of point CONTENT,
  independent of input order; the only residual ties are exact-duplicate points (value-equivalent for
  `torch.max`). Added `test_pillar_scatter_permutation_invariant_OVERCAP` (a dense cluster + `max_points=8`
  with a runtime assert that the over-cap path actually fires — would FAIL on the old code) and a **dense
  over-cap cluster in the A40 gate's synthetic batch** so the committed checksum exercises the fixed path.
- **Verification:** `148 passed` (was 147; +1 over-cap test); A40 bit-identity gate re-ran clean WITH the
  over-cap path exercised (job 6763843, node alvis9-01) → new
  `A40_WEIGHT_CHECKSUM = 0a30410a9905010bd94d78959d89ee7f1fb05a116d28c75dec2667ef81af98e9` (supersedes
  31f23465…); wall-clock unchanged (headline ~3.1 min/round). SPEC §3/§6/§7 + this log updated.
- **Rationale:** the build session had flagged this exact caveat in SPEC §7, but left the claim
  unqualified and the test in the no-truncation regime — Codex correctly required the production path
  itself be invariant, not just the common case. Canonical content-order truncation closes it with no
  new banned op (only `stable=True` sorts) and a negligible cost (5 extra stable sorts on the in-range
  points).

## [T2] 2026-06-16 — A40 bit-identity gate PASS + committed weight checksum (crown jewel)
- **Finding (the §0 spine):** `strict` mode does NOT raise on `scatter_add`/non-stable `topk` on torch
  2.7, and same-seed-twice on ONE GPU is bit-identical even with a `scatter_add` present — so the
  determinism proof cannot be the runtime raise, and a CPU / login-node (Tesla T4) pass is a FALSE PASS.
- **Decision/fix:** a dedicated **A40-pinned SLURM gate** (`scripts/det_gate_a40.py` +
  `run_det_gate_a40.sh`): asserts `get_device_name()∋"A40"` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` pre-CUDA,
  runs two same-seed 12-step central trainings → `torch.equal` on every param, and commits the per-param
  SHA-256 checksum. **Errors LOUD (exit 2) on no-CUDA / non-A40** (verified: the T4 login run exits 2).
  PASS on NVIDIA A40: `A40_WEIGHT_CHECKSUM =
  0a30410a9905010bd94d78959d89ee7f1fb05a116d28c75dec2667ef81af98e9` (job 6763843; the post-review value
  with the over-cap pillar path exercised — supersedes the pre-review 31f23465… from job 6763718).
- **Rationale:** determinism that passes on a T4 can still drift on the A40 / ARM H200 (the Arrhenius
  portability bet). Enforcement is by (1) a static AST ban over `models/fusion/**`, (2) permutation-
  invariance (CPU `max|Δ|=0`), (3) the #76176 + float-cumsum GPU guards, and (4) this A40 gate — NOT the
  `strict` raise. The cumsum + #76176 guards MUST be re-run on the ARM/H200 rebuild.

## [T2] 2026-06-16 — permutation-invariance by construction (the cross-arch determinism design)
- **Finding (determinism design):** the LSS reference splat sorts ranks with a NON-stable `argsort`, and
  the PointPillars cluster-mean is a float mean over points — both make the per-cell summation order
  input-order-dependent (ULP drift cross-architecture; the whole point of banning `scatter_add`).
- **Decision/fix:** (a) the LSS `cumsum_trick` sorts by a **canonical composite key** `rank·G + geom_id`
  (`geom_id` = the frustum cell's canonical index → `(rank,geom_id)` globally unique → the sum order is
  independent of how frustum points are presented); (b) the PointPillars PFN uses **per-point features
  only** (`x,y,z,intensity, x_p,y_p,z_p` — raw + pillar-center offset; **no within-pillar cluster-mean**),
  so `torch.max` over points is value-order-independent. Both scatter into the canvas with `index_copy_`
  at the **unique** cells (assignment, #76176-safe). `test_{splat,pillar_scatter}_permutation_invariant`
  assert byte-identity under an input permutation (CPU `max|Δ|=0`).
- **Rationale:** permutation-invariance ⇒ no summation-order dependence ⇒ architecture-portable
  determinism by construction. Dropping the cluster-mean is a deliberate determinism/accuracy trade (a
  minor PointPillars feature); a canonical-order cluster mean is a deferred refinement (flagged in SPEC §7).

## [T2] 2026-06-16 — half-pixel-correct resize keeps intrinsic/lidar2img EXACTLY consistent
- **Finding (no-oracle calibration risk):** T2 owns the resize T1 left out; a naive "scale K by W_out/W_in"
  intrinsic rescale is off by a sub-pixel half-cell vs the actual `F.interpolate(align_corners=False)`
  content location → camera features mis-placed in BEV (a SPEC failure mode, eyeball-only otherwise).
- **Decision/fix:** rescale with the **same affine** the resize implements: `u_out = fx·u + (0.5·fx−0.5)`,
  expressed as a 3×3 on `K` and a 4×4 left-multiply `M` on `lidar2img` (the `tx,ty` ride on the depth
  component). `test_resize_intrinsic_lidar2img_consistency_1px` (anchored to T1's real `lidar2img`)
  measures **~1e-4 px** residual (≤1px gate). Aspect ratio is not preserved (fx≠fy) but the calibration
  follows exactly; aspect-preserving resize+crop is a hooked later refinement.
- **Rationale:** exact consistency (not "<1px by luck") removes the silent cam-BEV/LiDAR-BEV misalignment
  the no-oracle setting can't otherwise catch.

## [T2] 2026-06-16 — ground-truth-anchored BEV convention (defeating the self-consistent-but-wrong trap)
- **Finding (the no-oracle trap T1 defeated, recurring at T2):** splat/scatter/target/decode share ONE
  `(x,y)→(row,col)` mapping (the T2↔T4↔T5 contract); a self-consistency test passes even if that single
  mapping is wrong (e.g. row/col swapped).
- **Decision/fix:** `test_bev_convention_ground_truth_anchored` anchors to **two independent T1 facts** —
  a real GT car's labelled box center and its **physical LiDAR returns** must fall in the SAME fine-grid
  cell (median within ≤2 cells) — plus an **injected row↔col-swap negative** that must BREAK the anchor
  (proving discriminating power). The box convention (== T1 canonical; NO `−π/2`, NO `(l,w,h)` swap) is
  pinned by an **encode→decode round-trip golden** (recovers center/dims/yaw incl. SIGN over yaw∈{±0.7,
  ±2.5,0}). Decode is deterministic (3×3 max-pool mask + `sort(stable=True)`, NMS-free).
- **Rationale:** anchoring to physical data + a corruption negative is the cheapest insurance a model with
  no bit-parity oracle can buy; the encode/decode golden catches a paired encode/decode swap a layout-only
  check misses.

## [T2] 2026-06-16 — additive loop generalization preserves dummy_regression byte-identity
- **Finding (skeleton-preservation risk):** the multimodal dict batch does not fit the T0 `(inputs,
  targets)` 2-tuple loop; generalizing the loop risks perturbing the regression path's byte-identity.
- **Decision/fix:** the loop is generalized **additively** — `_unpack_batch`/`_move_to_device`/
  `_batch_size` route a dict batch (inputs=targets=batch; `len(gt_boxes)`) while the tensor 2-tuple path
  keeps the EXACT original op sequence (`X.to/y.to → model(X) → criterion(out,y) → y.size(0)`). The
  `Criterion` alias is widened to `Callable[[Any,Any],Tensor]` in BOTH `loop.py` and `tasks.py`. The
  `dummy_regression` aggregated checksum is pinned to a committed golden
  (`d2d819…cd7cc`) and re-asserted; 21 T0 determinism/task-agnostic tests pass byte-for-byte.
- **Rationale:** the FL skeleton must stay task-agnostic (a T0 invariant); a provable same-op-sequence
  tensor path + the golden assert make any future drift fail loudly.

## [T2] 2026-06-16 — adversarial verification sweep before Codex handoff (6 skeptics + critic)
- **Finding (process):** ran a read-only 6-skeptic + completeness-critic workflow (`wf_9a712dd2-0a6`)
  that re-derived the highest-risk T2 surfaces and tried to BREAK them (mirrors T0/T1's pre-handoff
  sweeps): (1) the LSS `cumsum_trick`/`QuickCumsum` splat permutation-invariance; (2) PointPillars
  permutation-invariance + atomic-free scatter; (3) the single shared BEV `(x,y)→(row,col)` mapping +
  encode→decode inverse; (4) the corrected 3-case `gaussian_radius` + focal/L1 loss; (5) the D1/D6 freeze
  through `model.train()`; (6) the `dummy_regression` byte-identity after the loop generalization.
- **Decision/fix:** **No defects.** Every skeptic returned `clean` (no `defect`/`concern` verdict anywhere
  in the sweep). Independent confirmations: the corrected `gaussian_radius` is byte-identical (<1e-12) to
  the corrected mmdet3d/CenterPoint formula and each returned root satisfies its quadratic (residual
  ~1e-10) over 8 box sizes (NOT the `/2` bug); the BEV convention `col↔x, row↔y, flat=row·W+col` is used
  at all four call sites and decode is the exact inverse of the loss encoding; the loop tensor-2-tuple
  path is a byte-identical op sequence; the `CameraBackbone.train()` override keeps the frozen backbone
  (incl. ResNet BN) in eval through the detector's `.train()` recursion. The splat skeptic engaged deeply
  without flagging a defect; the splat surface is additionally nailed by `test_splat_permutation_invariant`
  (CPU `max|Δ|=0`), the A40 bit-identity gate, and the float-cumsum GPU guard.
- **Rationale:** a determinism-critical model with no bit-parity oracle earns trust by independent
  re-derivation + a corruption-negative + the A40 gate, not by the (now-hollow) `strict`-mode raise. The
  hardest review targets are listed in `collab/T2/SPEC.md` §7 for the Codex reviewer.

## [T2] 2026-06-16 — env: torchvision 0.22.1 --no-deps + offline-weights pre-cache (DT2-A)
- **Finding (build/offline):** D1's backbone is an ImageNet Swin-T but torchvision was not installed, and
  Alvis COMPUTE nodes are offline (torch.hub cannot download at train time).
- **Decision/fix:** `torchvision==0.22.1` installed **`--no-deps`** (its metadata pins torch==2.7.1+numpy,
  which would shadow the CUDA/numpy-matched module build — the nuscenes-devkit footgun); `TORCH_HOME`
  pinned in `build_venv.sh`/`run_in_venv.sh`/`run_alvis.sh`; the Swin-T + ResNet-18 `IMAGENET1K_V1`
  weights **pre-cached on the LOGIN node** at build time (param-count asserts 28,288,354 / 11,689,512
  double as the load check). torchvision `swin_t` uses manual fp32 math attention (no SDPA) → no flash/
  mem-efficient kernel reachable → no SDPA pin needed. cp312 wheels exist for x86_64 AND aarch64 →
  Arrhenius-portable.
- **Rationale:** the no-mmdet3d, pure-PyTorch posture extended to the backbone; the offline-weights
  blocker is resolved at build time on the login node, never on a compute node.

## [T1] 2026-06-16 — yaw extraction: pyquaternion uses a MINUS cross term (sign-bug trap)
- **Finding (scientific-error risk, caught pre-review):** the obvious "aerospace" yaw formula
  `atan2(2(wz + xy), 1 − 2(y²+z²))` is WRONG for the nuScenes/pyquaternion convention. pyquaternion's
  `yaw_pitch_roll[0]` (intrinsic z-y'-x'') uses `atan2(2(wz − xy), …)` — a **minus** on the cross term.
- **Decision/fix:** `transforms.yaw_from_quaternion` implements the minus-sign form; verified == pyquaternion
  to 0 over 5000 random quaternions and to 8.9e-16 over 18 538 real mini boxes.
- **Rationale:** the trap is that real nuScenes boxes are near-upright (tiny `x,y`), so the WRONG sign still
  passes a real-box parity test. A dedicated **random-quaternion** unit test (`test_yaw_formula_matches_
  pyquaternion_random`) is the non-vacuous gate; the real-box parity test alone is insufficient. This is the
  T1 analog of T0's "λ transplanted from a different optimizer" class of bug.

## [T1] 2026-06-16 — gt_in_range uses devkit `ego_dist`, NOT a lidar-frame radial
- **Finding (eligibility-semantics, deviates from the schema-table wording):** the frozen schema said
  `gt_in_range` is "computed in LIDAR_TOP frame", but the devkit eval filter is `box.ego_dist < class_range`
  where `ego_dist` = global planar radial from the **ego origin** at the LIDAR_TOP pose. The LiDAR is mounted
  ~0.94 m off ego, so a lidar-frame radial differs from `ego_dist` by up to 0.97 m.
- **Decision/fix:** compute `gt_in_range` to **exactly reproduce the devkit distance filter** (ego-origin
  global planar radial), not a lidar-frame radial. `test_gt_in_range_matches_devkit_distance_filter` checks it
  per box. `conventions.md` §6 supersedes the schema shorthand; flagged in SPEC §7 for Codex.
- **Rationale:** a 0.94 m mis-set at the 30/40/50 m class boundary would silently corrupt T4's ASR
  denominator — exactly the "eligibility fields wrong-meaning" failure mode the T1 SPEC warns about. T4 owns
  the full 6-criterion eligibility; T1 must hand it a denominator-faithful `gt_in_range`.

## [T1] 2026-06-16 — N derived from a justified floor; requested 50 → fallback N∈{20,25}
- **Finding (design, anti-gaming):** the plan forbids hard-coding N (esp. to 50). The real trainval `train`
  split has only **50 logs** (min-log 80 keyframes), so 50 location-coherent log-group clients would each be a
  single log far below any meaningful floor.
- **Decision/fix:** floor = **400 keyframes** (justified: ≥1 local epoch of ~100 SGD steps at batch 4). N is
  **derived** = max location-coherent log-groups each ≥ floor = **39** (floor±50 %: 200→44, 600→29). A
  requested `num-clients=50` exceeds the max-feasible 39 → the partitioner **falls back to N=25** (∈{20,25})
  with a recorded reason. Unit-tested on the **real trainval log table** (mini is degenerate: 6 logs, the
  fallback cannot fire — flagged as smoke).
- **Rationale:** makes N visibly non-arbitrary (reported across a floor band) and exercises the fallback the
  GATE requires, on real data — not on mini where it can't fire.

## [T1] 2026-06-16 — host-portable info-cache hash over RAW inputs (not derived geometry)
- **Finding (determinism/portability decision):** the cache hash must reproduce on the Arrhenius (ARM)
  rebuild. Derived f32 matrices could in principle differ in the last bits across x86↔ARM (BLAS/FMA ordering).
- **Decision/fix:** the content hash is taken over the **raw index inputs** — JSON-parsed f64 calibration/
  annotation values, DATAROOT-relative paths, integer tokens, and the derived int/str/bool fields — at fixed
  little-endian dtypes, samples sorted by `sample_token`, boxes by `ann_token`, no `set`/timestamp/abs-path.
  Those are byte-identical on any host; the derived geometry follows deterministically from them. A
  same-machine "build twice → bit-identical derived schema" test additionally certifies the matrices.
- **Rationale:** hashing raw inputs makes cross-machine reproduction follow from input-identity + deterministic
  code, sidestepping float last-bit fragility. The only float-derived hash field is the `gt_in_range` bool
  (boundary-flip is measure-zero); flagged in SPEC §7.

## [T1] 2026-06-16 — Codex REVIEW of T1 (verdict PASS — review loop closed)
- **Finding (review outcome):** Codex returned **PASS** with **0 findings** across every severity
  category (scientific-error / correctness-bug / invariant-violation / question / style — all "nothing
  found"). No scientific-correctness changes requested; no code changes made.
- **Decision/fix:** nothing to triage. The three items the build session flagged hardest (SPEC §7) were
  **explicitly endorsed** by Codex: (a) the yaw formula `atan2(2(wz−xy),1−2(y²+z²))` + `(l,w,h)` extent
  permutation + velocity rotation + the two-ego-pose `lidar→…→image` chain are implementation-equivalent
  to the devkit oracle; (b) **`gt_in_range` correctly uses the devkit eval `ego_dist < class_range[class]`
  radial from the ego origin, NOT a lidar-frame radial** (the build session's most-uncertain call,
  confirmed correct); (c) the cache hash excludes host-absolute paths/timestamps, ordering is by
  `sample_token`/`ann_token`, the write guard is active, images stay native uint8 1600×900, and the
  5-column LiDAR superset is parity-checked on devkit cols 0:4.
- **Rationale:** Codex independently re-ran the T1 suite — `43 passed` (class_map/conventions/dataset/
  info_cache/paths/transforms/viz, minus the multiprocessing gate) + `14 passed` (partition) + the
  2-worker `DataLoader` equality gate (`1 passed`, after enabling Python multiprocessing in its session)
  = **58 T1 tests**. Metric/ASR/eligibility computation correctly remains deferred to T4+. T1 build+review
  loop is closed; next is the orchestrator marking T1 done and issuing the T2 SPEC + kickoff.

## [T1] 2026-06-16 — adversarial verification pass before Codex handoff (6 skeptics + critic)
- **Finding (process):** ran a read-only 6-skeptic + completeness-critic workflow (`wf_ebde173a-64d`;
  yaw/box, ego-motion/projection, cache portability, V1 independence, partition, determinism/eligibility)
  that re-derived each surface against the devkit and tried to break it (mirrors T0's pre-handoff sweep).
  **All 6 surfaces returned `clean` — no defects.** Independent confirmations: box parity center 1.25e-12 m
  / extent exact / |Δyaw| 8.88e-16 / mean 4.8e-20 over 18,364 boxes; the `+`-sign yaw formula proven to FAIL
  random quaternions but PASS near-upright boxes (so the random test is load-bearing); single-shared-pose
  counterfactual shifts pixels 260 px; **5 injected `lidar2img` corruptions all caught by the V1 gate**
  (drop-cam-pose 59 px, transpose 33 M px, sign-flip 796 px, subtle 0.999-focal 0.94 px); cache hash
  boundary-flip impossible (closest box 5.6e-5 m across 1.1 M trainval boxes vs ~1e-13 m ULP delta);
  `gt_in_range` matches the devkit on all boxes (a naive lidar-frame radial would flip 182 — confirming the
  ego_dist decision was necessary).
- **Decision/fix (coverage gaps closed; +6 tests, 114→120):**
  (1) **Committed golden image sha256** — the pinned-decoder gate was a within-run self-compare (`h1==h2`);
  now asserts the sentinel keyframe's decoded images == a committed golden (catches a PIL→opencv/decoder
  swap). (2) **Box frame-round-trip** — the round-trip test carried only points; now also carries a real GT
  box's 8 corners through `lidar→cam→back` (SPEC §3 "point AND box"). (3) **Near-±π assertion isolated** —
  the `±π` clause was a no-op (`zip(dyaws,dyaws) if True`); now masks to boxes within 0.05 rad of ±π. (4)
  **Sub-floor surfacing** — the floor-derived path could return a client below the floor (only when floor >
  a location's whole-keyframe total, unreachable at the operating floor) silently; now records
  `sub_floor_clients` + a WARNING in the reason and report. (5) **IID-path determinism** — added same-seed
  stability + different-seed sensitivity for the Q2 IID baseline. (6) **Cache load-fidelity** + **`..`-escape
  guard** (`relative_to_dataroot` now raises on a path resolving outside DATAROOT) + **trainval
  `build_keyframe_info` smoke** (full schema build exercised on a real trainval keyframe).
  Doc fixes: `conventions.md` ego-displacement numbers corrected to the measured maxima (0.65 m / 48 ms) and
  a §6 note that `gt_in_range` is **distance-only** — T4 must additionally apply the devkit `num_pts==0`
  filter (from `gt_num_lidar_pts`) and the **bike-rack filter** (re-derived from the devkit; benign for the
  D8 `car` target, matters only for bicycle/motorcycle denominators).
- **Rationale:** the coordinate/yaw/box surface has no mmdet3d safety net — independent re-derivation +
  injected-corruption checks are the cheapest insurance before the scientific review; the gaps were
  untested-but-correct surfaces, now each guarded by an assertion.

## [T0] 2026-06-15 — nuscenes-devkit caps matplotlib<3.6 (manifest portability)
- **Finding (correctness-bug, build):** `nuscenes-devkit==1.1.11` publishes
  `matplotlib<3.6.0` (via the abandoned `descartes` map-rendering dep, which has
  no Python-3.12 wheel), conflicting with a modern matplotlib and breaking the
  portable manifest build.
- **Decision/fix:** install `nuscenes-devkit --no-deps` and supply its real
  runtime deps at modern py3.12/numpy-1.x versions; keep matplotlib 3.10.8. Build
  asserts the data + `DetectionEval` APIs import without `descartes`.
- **Rationale:** routing around an abandoned transitive is the Arrhenius
  portability posture; we don't use map rendering. Documented in docs/env.md.

## [T0] 2026-06-16 — Codex RE-REVIEW of 0977b7f (CHANGES-REQUESTED → resolved)
- **(invariant-violation) MultiKrum claim inconsistency.** Codex: I declared MultiKrum
  a bit-identical Flower carry-over but only tolerance-tested its subset-average.
  RESOLUTION — chose Codex's option B (Flower-compatibility, not bit-identity), with a
  quantified justification: Flower's `||x||^2+||y||^2-2x.y` distance has **~10% relative
  error at AD scale** (float32 cancellation: subtracting ~50141 from itself to get ~0.099),
  vs ~2e-11 for fl_v3's stable `||x-y||^2`. Matching Flower's exact bytes would make the
  Byzantine selection ~10% wrong on the fusion models — a correctness requirement, not
  pedantry. So fl_v3 KEEPS the stable distance and the claim is downgraded EVERYWHERE
  (SPEC §3/§7, multi_krum.py + aggregation_core.py docstrings): MultiKrum is Flower-
  COMPATIBLE (selection matches Flower for well-separated configs; subset-average within
  fp32 noise), explicitly NOT a bit-identical carry-over. FedAvg/NormClip stay bit-identical.
- **(question) bare `pytest` resolved to system pytest.** RESOLUTION: `run_in_venv.sh`
  now special-cases `pytest` → `python -m pytest` (guarantees the venv interpreter), and
  the preflight also checks `pytest`. Gate reproduces with either `pytest fl_v3/tests` or
  `python -m pytest fl_v3/tests`.

## [T0] 2026-06-16 — Codex REVIEW triage (verdict CHANGES-REQUESTED → resolved)
Three findings; all addressed.
- **(scientific-error) MultiKrum one-shot vs paper iterative + validity.** Codex
  compared against Blanchard's iterative m-Krum. RESOLUTION: fl_v2's MultiKrum oracle
  is Flower's built-in `select_multikrum`, which is ONE-SHOT (`np.argsort(scores)[:m]`,
  `num_closest=max(1,n-f-2)`, no gate). fl_v3 already matches that one-shot variant —
  documented explicitly. Fixed the validity gate to account for m:
  `multi_krum_valid` now requires `n>=2f+3` AND `1<=m<=n-f-2` (so n=5,f=1,m=3 is
  correctly INVALID/NA, not forced). Added `test_flower_fp32_parity.py::
  test_multikrum_decision_parity_vs_flower` (exact SELECTION parity vs real Flower;
  average within fp32 noise). Fixed the textbook fixture/test (m=2 valid; m=3 → NA).
- **(invariant-violation) Clean FedAvg/NormClip fp64 vs Flower fp32 (null-config
  bit-identity).** RESOLUTION (full fix, not documentation): added
  `aggregation_core.fp32_weighted_average` — a BIT-FOR-BIT replica of Flower's
  `aggregate_arrayrecords`. FedAvg/NormClip/MultiKrum now aggregate through it (fp64
  `aggregate_weighted_updates` kept ONLY for FLAME/FoolsGold, where the fl_v2 oracle
  also uses fp64). Verified bit-identical against REAL Flower in
  `test_flower_fp32_parity.py` (FedAvg uniform+weighted, NormClip post-clip = exact;
  MultiKrum selection exact). Clean-baseline bit-identity (AGENTS.md crown jewel) restored.
- **(question) venv not reproducible in the review env.** RESOLUTION: `run_in_venv.sh`
  now preflights — fails early with a clear "run build_venv.sh first" if `.venv_v3` is
  missing or lacks torch/flwr/sklearn.HDBSCAN/fl_v3 (the reviewer hit a half-built env).

## [T0] 2026-06-15 — self-verification sweep (8 module skeptics + completeness critic)
Ran an adversarial parity/determinism verification workflow before the Codex
handoff. FLAME, NormClip+FedMedian+metrics, and Partition verdicts = clean. Fixed
the actionable findings:
- **`seed_everything` masked numpy only** → now masks all three RNGs with one
  32-bit value (no stream desync for a seed ≥ 2**32; no-op for derive_seed leaves).
- **`enforce_determinism` docstring falsely claimed it "mirrors fl_v2"** (fl_v2 uses
  `warn_only=True`) → corrected to state the strict-raise default is an INTENTIONAL
  invariant strengthening for the AD model's no-banned-op requirement.
- **Client/server loaded arrays positionally** → switched to by-name
  `load_state_dict(arrays.to_torch_state_dict())` (matches oracle, robust to key order).
- **Strategy wrapper keyed output from the client reply** → now keys from the GLOBAL
  arrays (matching `new_global` order) + asserts reply key order == global key order.
- **`local_runner` FoolsGold ignored `foolsgold-head-index`** → forwarded.
- **`krum_scores` returned garbage if called directly with invalid f** → guard raises.
- **`DummyRegressionTask.evaluate` duplicated the eval loop** → delegates to `loop.evaluate`.
- Added `tests/test_defense_edge_cases.py` (FLAME n<4 / small-majority, NormClip
  scale<1 + decision aggregate, FedMedian odd-n, FoolsGold n==1).
**Documented (not bugs) for the Codex reviewer** — see SPEC §7.

## [T0] 2026-06-15 — unified fp64 aggregation core (parity scope, design decision)
- **Finding (correctness-bug per critic):** fl_v3 routes plain FedAvg + NormClip
  through the same fp64 `aggregate_weighted_updates` core as FLAME/FoolsGold, whereas
  the fl_v2 oracle's *clean*-FedAvg/NormClip aggregation delegates to Flower's fp32
  `aggregate_arrayrecords`. Algebraically equal (Σcoef=1) but not bit-identical.
- **Decision:** keep the unified fp64 core (one source of truth, higher precision).
  Bit-parity is CLAIMED only for FLAME + FoolsGold (the oracle computes those via the
  SAME fp64 core → exact). Clean/clip-path agreement with Flower's fp32 weighting is a
  tolerance-level T3 (real-Ray) check, not a T0 bit-identity claim.
- **Rationale:** the GATE requires FLAME + ≥1 other parity (both bit-identical);
  unifying avoids a second divergent aggregation path. Documented in
  `aggregation_core.py` + SPEC §7; flagged for Codex.

## [T0] 2026-06-15 — `--system-site-packages` numpy shadowing
- **Finding (invariant-violation risk):** an unpinned transitive pulled numpy
  2.4.6 into the venv, shadowing the module's CUDA-matched numpy 1.26.4 (breaks
  scipy + CUDA + determinism).
- **Decision/fix:** `constraints.txt` pins numpy/scipy to the module versions; the
  build asserts `numpy == 1.26.4` post-install.
- **Rationale:** the exact hazard CLAUDE.md warns about; the assert makes a
  regression loud.

## [T3] 2026-06-16 — DT3-A: 4 non-backbone buffers exist (SPEC claim refined), still lossless
- **Finding (correctness nuance):** the T3 SPEC asserted "ALL frozen params + ALL buffers
  live inside `camera_backbone`". Introspection of the real detector shows that is true for
  *params* (0 frozen params outside the backbone) but NOT for *buffers*: 4 non-backbone
  buffers exist — `preprocess._mean`, `preprocess._std`, `view_transform.frustum`,
  `view_transform.depth_values`.
- **Decision/fix:** keep trainable-only = the 62 trainable param tensors (buffers excluded).
  VERIFIED the 4 buffers are node-invariant CONSTANTS (byte-identical across seeds) AND
  static through a train step (unchanged after 3 Adam steps) — so each node reconstructs them
  identically in `build_model`; excluding them from the update vector is lossless. The 60
  resnet18 backbone BN running-stat buffers are frozen (eval mode, D1/D6) and also unchanged.
- **Rationale:** `requires_grad` filter (not a prefix-drop) is the correct definition; the
  buffers ride along inside each node's reconstructed model, never the wire. Documented in
  `training/tasks.py` (`TRAINABLE_MODULE_SLICE_MAP` note) + flagged for Codex.

## [T3] 2026-06-16 — DT3-B: flwr 1.27 exposes no node_id→partition-id map → discovery probe
- **Finding (design, verified vs source):** the server strategy cannot recover a participant's
  `partition-id` at `configure_train` time. `vce_api._register_nodes` assigns `partition_id=i`
  but the node_id comes from `secrets.token_bytes(32)` (random); `LinkState.get_nodes` returns
  a `set` (unordered); the partition-id lives only in each client's `node_config`. So
  Flower's `random.sample(get_node_ids())` picks a different subset per same-seed driver (§0.2).
- **Decision/fix:** add a cheap `@app.query()` handler that echoes the node's `partition-id`;
  the strategy runs ONE discovery QUERY round at first `configure_train` to build the
  `partition_id→node_id` map for THIS run, then selects deterministically over `range(N)`
  (`strategy/sampling.select_partition_ids`) and maps back. Requires the discovered ids to
  cover exactly `range(N)` (⇒ `num-supernodes` must equal the derived client count N).
- **Rationale:** makes EVERY round (incl. round 1) deterministically partition-sampled, so
  two same-seed drivers with different node_ids select identical partition-ids at fraction<1.

## [T3] 2026-06-16 — `derive_seed` is int-only; sampler uses reserved integer salts
- **Finding (API):** the SPEC's shorthand `derive_seed(seed, "sample", round)` cannot work —
  `derive_seed` casts every arg to `int()` (a string raises).
- **Decision/fix:** the sampler calls `derive_seed(seed, SALT, round)` with reserved integer
  salts `SAMPLE_SALT_TRAIN=700_000_001` / `SAMPLE_SALT_EVAL=700_000_002` in the `client_id`
  slot — far above any real partition-id (0..N-1), so a sampling seed never aliases a
  per-client training seed `derive_seed(seed, client_id, round)`. Harness left untouched.
- **Rationale:** preserves the T0 determinism harness; collision-safe + independent
  train/evaluate RNG streams.

## [T3] 2026-06-16 — flwr 1.27 launcher: auto-SuperLink + `--federation-config` ignored
- **Finding (launcher, verified vs source):** (a) `flwr run . local-simulation-gpu`
  auto-starts + manages the local `flower-superlink` (`cli/local_superlink._start_local_
  superlink`) bound to `FLWR_LOCAL_CONTROL_API_PORT`/`…SIMULATIONIO…` — a manual
  `flower-superlink` (the fl_v2 pattern) collides on those ports. (b) `--federation-config`
  is hidden + IGNORED (warns), so `num-supernodes` is NOT overridable on the CLI.
- **Decision/fix:** `run_alvis.sh`/`run_fedavg_a40.sh` set ONLY the per-job port env (no manual
  SuperLink, no SuperExec grep/sleep wait) and stamp the derived N into the
  `[superlink.local-simulation-gpu]` block of `$FLWR_HOME/config.toml` via awk before
  `flwr run`. Silent-exit guard greps the ServerApp's `FL_TRAINABLE_CHECKSUM` completion line.
- **Rationale:** matches the flwr-1.27 reality (T3_SPEC §0.1); keeps the valuable fl_v2
  hardening (per-job ports/tmp, silent-exit guard) and drops the stale parts.

## [T3] 2026-06-16 — `flwr run` app path = fl_v3 (not repo root) + absolute data paths
- **Finding (launcher bug, caught by the silent-exit guard on the A40):** `flwr run .` from the
  worktree root failed — "Failed to load Flower App configuration in <root>/pyproject.toml" — because
  the Flower-App pyproject lives in `fl_v3/`, not the repo root. The silent-exit guard correctly
  forced the job FAILED (it works).
- **Decision/fix:** pass the app path explicitly (`flwr run "${REPO}/fl_v3" local-simulation-gpu`)
  and make `nuscenes-cache-dir`/`output-dir` ABSOLUTE via run-config overrides (the Ray actors' cwd
  is not guaranteed to be the repo root). Applied to all 3 flwr-run launchers. Also fixed an
  early-`python` (pre-`module load`) exit-127 in run_fedavg_a40.sh.
- **Rationale:** matches fl_v2's "cd into the app dir" pattern but keeps relative data paths working
  by making them absolute instead of cd-ing (which would break `./fl_outputs`).

## [T3] 2026-06-17 — Codex review triage (CHANGES-REQUESTED → addressed)
Codex PASSed the science (DT3-A/DT3-B/FedAvg parity, calibration, metrics — nothing found) and
requested 5 fixes that make the gate/artifact ENFORCE the SPEC instead of relying on the one-off run.
All addressed:
- **F1 (correctness-bug) — trainval gap was a 256-sample-val proxy, reported as full val.** The run
  capped server eval at `det-eval-limit=256` (`sorted(sample_token)[:256]`). Fix: re-evaluate the two
  saved FULL-model checkpoints on the ENTIRE trainval val (6019 samples, `det-eval-limit=0`) via
  `scripts/t3_trainval_reeval_fullval.py` (job 6764280) and relabel the SPEC with the full-val number
  + the exact eval scope (no longer implies full val when it wasn't).
- **F2 (invariant) — cross-check mismatch only NOTE.** `run_fedavg_a40.sh`: local_runner↔Ray on the
  SAME A40 must be byte-identical → now a HARD `FAIL=1` (the allclose fallback is only for a documented
  cross-DEVICE case, which the same-A40 gate is not).
- **F3 (invariant) — substrate check skipped if norm_log missing.** Now FAILs if either `norm_log.json`
  is absent (no substrate artifact = fail), then compares canonical JSON.
- **F4 (invariant) — gate didn't assert its shape.** `fl_gate_a40.py`: asserts
  `task-type=='nuscenes_detection'`, `num-server-rounds>=3`, `0<fraction-train<1` (exit 2) BEFORE
  training — closes the §0.3 trivial-regime FALSE-PASS hole.
- **F5 (style) — trailing whitespace** in findings_log stripped.
Hardened gate re-run (job 6764281) reconfirms OVERALL PASS with enforcement active (same checksum
d82ef500…). Codex re-review pending.

## [T3] 2026-06-17 — Codex RE-REVIEW of T3 (verdict PASS — review loop closed)
- **Finding (review outcome):** Codex re-reviewed commits `14aad8c` (F1–F5 fixes + Path A/B docs) and
  `d9cc8e9` (D9 A100/full-model-from-scratch note) and returned **PASS** — all prior findings resolved,
  **no new** scientific-error / correctness-bug / invariant-violation / question / calibration / metric
  findings. It independently verified: the 19 T3 tests pass, `py_compile` of the changed scripts passes,
  `git diff --check 76c9128..HEAD` is clean, and the A40 logs for the full-val re-eval (`6764280`) +
  hardened gate (`6764281`) match the committed artifacts.
- **Resolution confirmed by Codex:** F1 — `trainval_fullval_reeval.json` records full v1.0-trainval val
  (`det-eval-limit=0`, n=6019, proxy_n_gt=80004, gap **+0.2073**); SPEC labels it authoritative with the
  256-subset retained as the in-training proxy. F2/F3 — `run_fedavg_a40.sh` hard-fails on cross-check
  mismatch and on a missing `norm_log.json`. F4 — `fl_gate_a40.py` exits 2 unless
  `nuscenes_detection` + `num-server-rounds>=3` + `0<fraction-train<1`. F5 — whitespace gone. D9 deemed
  "scientifically bounded" (Path A multi-GPU vs Path B shared-GPU distinguished; gates stay single-actor;
  Swin-T/trainval + GPU-tier re-validation required before relying on concurrent actors or A100).
- **Rationale:** T3 (the FL platform milestone — real Flower/Ray clean FedAvg on the A40, bit-identical
  across two same-seed runs, IID-mini≈central, the measured trainval non-IID gap) is scientifically
  signed off; the build+review loop is closed. Next: the orchestrator marks T3 done and issues T4
  (DetectionEval mAP/NDS + ASR metrics + V4), which builds on T3's clean baseline + the frozen
  update-vector layout contract.

## [T4] 2026-06-17 — yaw convention: T1 box7 yaw = Tait-Bryan Euler, devkit eval yaw = rotated-x heading
- **Finding (measured on mini, 258 boxes):** T1's canonical `box7[6]` yaw equals
  `pyquaternion.Quaternion.yaw_pitch_roll[0]` (intrinsic z-y'-x'' Tait-Bryan Euler) **to 4.4e-16**,
  but the devkit `DetectionEval` orientation error uses `nuscenes.eval.common.utils.quaternion_yaw`
  = the **rotated-x-axis heading** (`atan2` of `R·[1,0,0]`). On tilted real boxes the two differ by
  **up to ~0.0038 rad (~0.22°)**. So even a *perfect* `box_to_global(GT)` incurs a ~0.004-rad AOE
  (orientation-error) floor at trainval.
- **Decision/why negligible + flagged-not-fixed:** AOE enters NDS as one of five TP scores via
  `1−min(1,AOE)` → ≤~0.004 of one score → << 0.001 NDS; AP is center-distance (yaw-free); the D4
  **disappearance ASR is detection-presence, not yaw**. `box_to_global` cannot recover the heading
  from a scalar Euler yaw (pitch/roll were discarded by T1's forward), and **T1 is frozen** — carrying
  full-3D orientation would be a T1 touch. So this is documented + flagged for Codex, not fixed. The
  round-trip test therefore splits rigor: a TIGHT lift-equivalence (`|ΔR|<1e-9` feeding the SAME yaw to
  our matrix lift AND an independent devkit `Box` lift) + a LOOSE heading-vs-raw-annotation check
  (`<0.02` rad — catches a gross sign(~π)/offset(~π/2)/swap bug while tolerating the convention gap).

## [T4] 2026-06-17 — GT-as-pred AP≈1 needs the devkit num_pts (lidar+radar), not T1's lidar-only count
- **Finding:** the §0.1 GT-as-pred sanity gave car AP@2m only **0.9667** when the pred `num_pts` was
  set to T1's `gt_num_lidar_pts` (lidar-only). `filter_eval_boxes` drops boxes with `num_pts==0` on
  BOTH GT and pred, but the devkit GT `num_pts = num_lidar + num_radar` — so a car with 0 lidar but
  >0 radar is KEPT in GT yet DROPPED from a lidar-only pred → an unmatched-GT recall miss (~3.3%).
  Setting pred `num_pts = -1` (the prediction sentinel) was WORSE (0.84 — occluded GT cars with
  `num_pts==0` become FPs at score 1.0). Setting pred `num_pts = num_lidar+num_radar` (the devkit GT
  value) → **car AP@2m = 1.0000 exactly**.
- **Decision/fix:** `gt_as_pred_submission` takes an optional `num_pts_by_ann` (devkit lidar+radar per
  ann_token); the readiness driver + the test build it from `nusc`. PRODUCTION model predictions keep
  the `num_pts=-1` sentinel (a detector does not know GT point counts) — only the GT-as-pred *sanity*
  mirrors the devkit count. This is a T1-doesn't-carry-radar artifact, not a conversion bug.

## [T4] 2026-06-17 — DetectionEval invocation + determinism gotchas (verified vs devkit 1.1.11)
- Call **`DetectionEval(...).evaluate()`** NOT `main()` — `main()` does `random.seed(42)`+`shuffle`
  (example plots) and writes PNG/PDF/json; `.evaluate()` touches no RNG and writes no files.
- `DetectionEval.__init__` asserts `set(pred.sample_tokens)==set(gt.sample_tokens)` → the results JSON
  must key EVERY eval-split token (empty `[]` if no detections), and the official metric is
  ALWAYS full-split (no `det-eval-limit` subset — the driver forces `det-eval-limit=0`).
- `accumulate` sorts predictions by `(score, emission-index)` (Python stable sort, then reversed) —
  equal-score ties break on **emission order** → boxes are emitted in a **content-defined order**
  (`(−score, translation, size, rotation, name)`) so mAP/NDS is permutation-invariant (tested:
  permuting equal-score boxes → byte-identical JSON + identical mAP/NDS).
- `velocity_l2` is a plain L2; NaN GT velocity (no prev/next box) → NaN → nan-ignored by `cummean`,
  so copied-velocity GT-as-pred gives AVE≈0. `quaternion_yaw` (heading) ≠ Euler (see above).

## [T4] 2026-06-17 — full readiness pipeline validated end-to-end on mini (harness sound)
- `scripts/t4_readiness_eval.py` ran clean on mini_val (untrained resnet18 ckpt): single shared decode
  → official DetectionEval → GT-as-pred sanity (car AP@2m **1.0000** in-driver) → 6-criterion
  eligibility (eligible-clean-detected **N=185**, tally c1=2568/c2=2568/c3=1991/c4=2239/c5=c6=185) →
  content-hashed frozen subset bound to the ckpt checksum → **false-disappearance 0.0** (a 2nd FRESH
  decode over the subset's samples reproduced every detection — determinism + batch-invariance) → V4 →
  **VERDICT NOT-READY (scale=mini-smoke)** with correct gaps. 190 pytest tests pass (167 T0–T3 + 23 T4).
- The offline-preflight guard correctly FAILED the first reference submit (job 6764599) — swin_t
  ImageNet weights were not cached under TORCH_HOME; pre-cached them on the login node, resubmitted as
  job **6764601** (full participation, log-group, trainval, Path-A 4×A40). Trainval mAP/NDS + readiness
  verdict + the attacked-checkpoint checksum + the frozen-subset hash land when 6764630 + the readiness
  eval complete (~5–6 h).

## [T4] 2026-06-17 — self-adversarial review (8-agent workflow) → 3 confirmed fixes applied
- A pre-Codex adversarial review (workflow `wf_6ab3b65d`, 5 dimensions × adversarial verify) over the T4
  modules confirmed 3 defects; all fixed before the readiness eval consumed the checkpoint:
  - **(blocker) `detected_target_gt` GT order not EXPLICITLY sorted** — index order already equals
    ann-token order (T1 sorts `info_cache` rows by `ann_token`, so the mini hash was correct), but the
    function relied on it implicitly. Fixed: `gt_idxs = sorted(..., key=ann_token)` — a no-op on real
    data (hash unchanged) that makes the greedy matcher reproducible for ANY caller + the docstring true.
  - **(major) `disappearance_asr` re-detected with a caller-passed `thr`, not the subset's BOUND
    thresholds** — a T5-contract hole (§0.5 "identical targets"). Fixed: added `thresholds_from_subset`;
    `disappearance_asr(subset, decodes)` now derives τ_clean/d_clean/target from the frozen subset (the
    GATE floors `n_min`/`false_disappear_max` still come from the live config in
    `false_disappearance_baseline`). New test `test_disappearance_uses_subset_bound_thresholds`.
  - **(major) `t4_reference.json` had `det-eval-limit=256`, §0.2 says `0`.** The checkpoint is
    **eval-independent** (server eval is `no_grad`/read-only → byte-identical weights), and the official
    numbers come from the separate full-val readiness eval — so 256-vs-0 does not change the deliverable.
    But §0.2 is literal and the run was at round 0, so set `det-eval-limit=0` (honest full-val in-loop
    convergence; matches the T3-F1 lesson) and resubmitted (6764601 → **6764630**). Determinism preserved.
- Post-fix: 191 pytest tests pass (167 T0–T3 + 24 T4; +1 from the new bound-thr disappearance test).

## [T4] 2026-06-17 — reference-run performance profile (compute-bound; speedup backlog → decisions.md D11)
- **The full-participation reference (job 6764630, 4×A40 Path-A, Swin-T frozen) is COMPUTE-bound, not
  Flower-bound.** Evidence via `srun --jobid=6764630 --overlap nvidia-smi`: all 4 A40s pinned at 100 %
  during the training phase (e.g. `100 100 100 100`); the earlier "GPU-0-only" snapshots were the
  single-GPU server-eval phase between rounds. The compute-node Flower log shows **4 `ClientAppActor`s**
  (one per GPU) → Path-A IS parallelizing (25 clients in ~7 concurrent waves of 4 ≈ ~18 min train + ~4 min
  full-val server eval ≈ **~22 min/round**; ~5.5 h for 15 rounds, inside the 12 h wall). Aggregation
  averages only the 62 trainable tensors across 25 clients (ms) — negligible. The dominant cost is the
  **frozen Swin-T (ViT, the D1 headline — NOT resnet; resnet18 is only the mini-smoke fallback) forward
  over 6 camera images**, recomputed every step although it never updates.
- **Hardware (Alvis):** every node is 4-GPU (A40 48 GB ×85 / A100 40 GB ×56 / A100fat 80 GB ×8 / V100). A40
  has **no InfiniBand** (no multi-node) and is "inference/smaller training"; A100 (HGX) is ~2–2.5× faster
  per-GPU for the transformer forward at ~2× units/hr. So there is **no way to get >4 GPUs into one Flower
  (single-node Ray) sim run**; the only single-run speedup is A100 per-GPU (~2× wall-clock) — and **D9
  requires an A100 determinism gate first** (A100 ≠ A40 byte-comparable). A100fat only matters for the
  full-model-from-scratch ablation. For T4 now: **A40 is correct** (validated, fits the wall).
- **Speedup backlog recorded as `decisions.md` D11 (PROPOSED, needs orchestrator + a profiling session):**
  frozen-backbone **feature caching** is the biggest determinism-safe per-run win (cache the invariant
  frozen Swin-T feature maps once per image → ~3–5× per-step; caveats: no image aug, bit-identity gate,
  per-GPU-tier cache); A100 is the per-cell lever (after its determinism gate); across-cell fan-out (D9)
  is the matrix lever; AMP/fp16 + `torch.compile` are REJECTED (break bit-determinism); overcommit +
  bigger-batch don't apply. A deep codebase speedup analysis is deferred to a future (orchestrator-blessed)
  session.

## [T4] 2026-06-17 — A100 determinism gate PASS (D9 requirement met) + a launcher bug + A100≈A40 speed
- **A100 byte-identity: PASS.** The D9 A100 determinism gate (job 6764809, 4×A100-SXM4-40GB, paired
  3-round full-participation log-group trainval) produced **identical** `FL_TRAINABLE_CHECKSUM` for the two
  same-seed runs: runA == runB == `ae2b4571aeb43442b249d1209cd5efcdc0055bb413110044b9ac93ba8e5e78e7`
  (verified by diffing the two `trainable_checksum.txt` files). So the platform is **bit-deterministic on
  A100 (SXM4-40GB)** at the full-participation operating point — A100 is **unlocked for T5–T7 per D9**.
  Eval losses were also identical across the runs (round1 4.208648144673976 …). NOTE: this checksum is for
  the **3-round gate config** (NOT a science checkpoint) — a full A100 reference would have its own.
- **Launcher bug (fixed) — false "DIVERGED" + swallowed live log.** `run_t4_reference_a40.sh` captured
  `CHK_A="$(run_one …)"` — i.e. run_one's ENTIRE stdout (the echoes + the tee'd flwr stream + the final
  `cat`), not the 64-char checksum. Effects: (a) the `T4_PAIRED` comparison diffed whole log-blobs →
  printed `[t4-ref] FAIL: ... DIVERGED` and `exit 1` (→ SLURM marked job 6764809 FAILED) **despite the
  checksums being identical**; (b) the flwr stream was captured into the variable instead of streaming to
  the SLURM `.out` (this — NOT just tee buffering — is why the `.out` was quiet; norm_log was the right
  progress signal). **Fix:** run_one now streams to `.out`; the checksums are read from the artifact FILES
  via `read_chk` and compared (with a non-empty guard). The A40 reference (6764630, single run, old
  launcher) is **unaffected in its deliverable** — its checkpoint + `trainable_checksum.txt` are written
  correctly by `server_app`; only the launcher's final echo was messy + the `.out` quiet. The readiness
  eval reads the checksum from the model/file, so it is unaffected.
- **A100 speed ≈ A40 (~1.2×, not 2×).** runA did 3 rounds (+ bring-up + round-0 eval) in **~75 min**
  (start 13:51:03 → round-3 norm_log 15:06:00) ⇒ **~25 min/round**; A40 steady-state is ~27 min/round. The
  workload is **I/O- + single-GPU-full-val-eval-bound**, not GPU-compute-bound, so the faster A100 yields
  only a modest end-to-end gain. Recalibrates D11: A100's value is the determinism validation + matrix
  fan-out, NOT single-run speed — **feature caching remains the real single-run lever.**

## [T4] 2026-06-17 — trainval readiness verdict + the false-disappearance batch-invariance fix
- **The trainval readiness eval (A40, checkpoint `a80466c3…`) on the full val split (6019):**
  official **car recall 0.70** (>> floor 0.20), **eligible-clean-detected N=23,354** (>> N_min 150),
  mAP 0.080 / NDS 0.138, **GT-as-pred sanity car AP@2m = 1.0000** (the §0.1 conversion is exact at
  trainval scale). Full participation (D10) decisively fixed the weak sampled-regime model. Eligibility
  funnel: 80004 car GT → frustum 80004 → ≥τ_pts 29006 → in-range 59522 → clean-detected 40014 → eligible
  (all 6) 23354.
- **The first verdict was NOT-READY for ONE reason: false-disappearance = 9.4% (> 2%).** The model is NOT
  the problem (recall + N pass with huge margin), so the gate's canned "strengthen the architecture"
  advice was MISDIRECTED. Root-caused with a direct diagnostic (`_t4_fd_diagnose.py`, 60 subset samples
  decoded 3 ways on the checkpoint):
  - batch-16 re-run vs batch-16 (SAME batching): **0/60 samples differ → run-to-run determinism PERFECT.**
  - batch-16 vs batch-1: **28/60 samples differ** → the detector forward is **NOT batch-invariant** (cuDNN
    conv varies with batch composition → boundary detections near τ_clean=0.1 flip).
  - batch-1 isolated disappearance: **0.37%** (vs 8.55% with the subset-vs-fullval batch mismatch).
  So the 9.4% was a **harness artifact**: the frozen subset was built from a full-val-batched decode but
  the false-disappearance re-decode used a subset-only-batched loader → different batch grouping → spurious
  flips. NOT a model defect, NOT nondeterminism (run-to-run is bit-identical).
- **Fix + protocol contract:** the ASR disappearance must depend only on a target's OWN trigger, not on
  batch-mates → the whole readiness/ASR decode now runs at **batch_size=1** (canonical per-sample
  inference; one consistent batch-invariant decode for DetectionEval + the frozen subset + disappearance +
  V4). `t4_readiness_eval.py` forces `cfg["batch-size"]=1`. **T5 inherits this:** decode triggered inputs
  at batch_size=1 too, else the disappearance ASR carries an ~8–9% batch-grouping noise floor. Re-running
  the readiness eval at batch_size=1 (job 6765358 → `readiness_bs1/`) — expect false-disappearance ≈ 0 (the
  subset built + re-checked at batch_size=1 is bit-identical run-to-run) → verdict READY. The batch-16
  verdict is preserved in `readiness/` for the record.
- **CONFIRMED — the batch_size=1 readiness eval (job 6765358 → `readiness_bs1/`) returned `READY`:**
  mAP/NDS **0.1253 / 0.1688**, official **car recall 0.85** (> floor 0.20), **N=27,432** (≥ N_min 150),
  **false-disappearance = 0.0** (defined, passed), gaps=[]. (batch_size=1 also gives *better* detections
  than batch-16: recall 0.85 vs 0.70, N 27,432 vs 23,354 — the batched cuDNN effects were slightly
  degrading detections, so canonical per-sample inference is both more correct AND stronger.) Of-record:
  checkpoint checksum `a80466c3…`, frozen-subset hash `2ad8f8da55e8516bf0c46085cd5217ad2b2d1984c23499f51c397ad7cad1940f`.
  **T4 GATE is GREEN: the benchmark is READY for T5.** Observed: batch_size=1 is GPU-idle/CPU-bound
  (97% single-core, GPU ~0%) — the 6019-sample eval took ~1–2 h; a per-cell perf concern for T5–T7.

## [T4] 2026-06-17 — Codex review (CHANGES-REQUESTED) triaged + addressed
Codex PASSed the science (no scientific-error / correctness-bug / metric / calibration / oracle-parity
finding; 24 T4 tests + py_compile + devkit-source inspection) and raised **one blocking invariant-violation**
+ a non-blocking question + a style nit. All addressed:
- **(BLOCKING) readiness verdict was checksum-bound but NOT provenance-bound to D10 (§0.2).** The READY
  predicate checked only metric floors + `scale`; the reference launcher merely *warned* on
  `fraction-train≠1.0` — and the warning was buggy (`printf '%.0f'` rounds 0.9→1, so 0.9 wouldn't warn).
  So an overridden `CONFIG`/`CKPT` could emit a READY `benchmark_readiness.json` for a sampled/IID/defended
  checkpoint whose floors happen to pass — the §0.2 partition/participation-mismatch trap. **Fix:** new
  tested module `eval/provenance.py` (`build_provenance` / `check_d10` / `verify_d10_provenance`, single
  source of truth for the D10 key set); `run_t4_reference_a40.sh` now **hard-fails** (exit 3) any non-D10
  config (task-type=nuscenes_detection, version=v1.0-trainval, train/val splits, partition-mode=log_group,
  defense=none, `fraction-train==1.0`) AND writes `provenance.json` beside `final_model.pt`;
  `t4_readiness_eval.py` **hard-verifies** that provenance (bound to the recomputed trainable checksum) at
  `scale=trainval-scientific` BEFORE emitting any verdict — a missing/mismatched provenance RAISES. So a
  non-D10 checkpoint can never produce a valid go/no-go. `benchmark_readiness.json` now records
  `verified_d10_provenance`. The EXISTING checkpoint's provenance was backfilled from its authentic
  `t4_reference.json` (job 6764630) and verifies (regime=D10-full-participation-log-group-trainval-clean,
  checksum matches). **+7 tests** (`test_eval_provenance.py`): compliant→[], each violation flagged,
  missing/sampled/IID/checksum-mismatch all RAISE. Re-ran the readiness eval through the now-gated driver
  (job 6765405 → `readiness_bs1/`) to emit the of-record artifact with verified provenance.
- **(non-blocking) yaw tolerance contract** — durable `T4_SPEC §0.1` says yaw `<1e-4`, the test uses
  heading `<0.02` (T1 Euler vs devkit `quaternion_yaw`; documented in SPEC §3a + above). Non-blocking
  (lift-equivalence + GT-as-pred AP=1.0 cover it). Flagged for the **orchestrator** to align the durable
  contract (build session does not edit `T4_SPEC.md`).
- **(style)** trailing whitespace in `collab/T4/SPEC.md` removed.
Post-fix: **198 tests** (167 T0–T3 + 31 T4).

## [T4] 2026-06-17 — Codex RE-REVIEW PASS (review loop closed)
Codex re-reviewed commit `406d162` and returned **PASS**: the blocking D10 provenance finding is
resolved (`eval/provenance.py` defines + binds the full-participation log-group trainval clean provenance
to `FL_TRAINABLE_CHECKSUM` and hard-refuses missing/sampled/IID/defended/wrong-split/checksum-mismatch;
the reference launcher hard-fails non-D10 before training + writes `provenance.json`; the readiness eval
`verify_d10_provenance()`s before any verdict and records `verified_d10_provenance`). Codex re-ran 31 T4
tests (pass), `py_compile` (incl. `eval/provenance.py`), `git diff --check c711aef..HEAD` (clean). **No
new** scientific-error / correctness-bug / invariant-violation / question / calibration / metric finding.
The yaw-tolerance item remains a documented non-blocking contract question for the orchestrator (durable
`T4_SPEC §0.1` wording); style whitespace resolved. **T4 is scientifically signed off — review loop
closed.** Merged to `v3-ad-perception`.

## [T5] 2026-06-18 — attack suite built + login validated (trainval runs in flight)
Built the fusion-aware backdoor attack package `attacks/{trigger,poison,poisoned_client,fusion_ablation}`
+ the V5/V3(trigger) viz + the `client_id` routing seam (additive in `tasks.py`) + the eval driver
`scripts/t5_attack_eval.py` + the SLURM launchers, honoring the two §0 blockers:
- **§0.A — disappearance = CENTER-RELOCATION, not box-deletion.** `relocation` shifts `gt_boxes[:,0]` by
  `Δ_reloc=6.0 m` (>2·d_clean) and KEEPS the box (so `losses.CenterPointLoss.build_targets` still renders
  a heatmap peak — but at the WRONG cell; the true cell is left unlabelled). Box-deletion demoted to the
  `delete` control (the GATE `label_only_delete` cell). Verified the loss reads the center from `gt_boxes`.
- **§0.B — cond-5a = ZERO the LiDAR-BEV input via a `forward_pre_hook` on `model.fusion`.** Verified the
  `ConvFuser` is `concat→Conv2d(bias=False)→GroupNorm(output)→ReLU`, so zeroing the LiDAR input is exactly
  zero-additive → a true same-weights camera-only readout. Unit-tested: the ConvFuser output is byte-
  identical for ANY two LiDAR inputs when zeroed, and the hook path == the manual zeros-fusion path. The
  two guards (LiDAR-invariance + clean-recall precondition) run in `--task guards`.
- **Anti-gaming pins committed in `pyproject.toml` BEFORE the runs**: poison_rate 0.5, ρ 0.2 (m=5),
  Δ_reloc 6.0, trigger area-frac 0.25 (≤0.30 budget), δ_fusion 0.2 AND 2×, stealth floor 0.75, occlusion
  <0.02; literal pins for the frozen subset `2ad8f8da`, clean checkpoint `a80466c3`, and the null full-
  state target `0fe444e31a1e0d9f…` (computed from the clean `final_model.pt`).
- **Threat model**: roster = `sorted(Random(derive_seed(seed,MALICIOUS_SALT)).sample(range(25),5))` =
  **[2,3,12,13,19]**, m_r=5 (honest-majority), drawn once, recorded; rate=0/non-roster/non-selected →
  literal `base_ds[idx]` with ZERO RNG (the byte-identical-null path); per-client selection uses a private
  `derive_seed(seed,POISON_SELECT_SALT,client_id)`. `m_r` is ground truth (grep-guarded ⊥ any f_r).
- **Venv pointer caveat**: the shared `.venv_v3` editable-install points at the T4 worktree, so login
  tests run with `PYTHONPATH=<this worktree>/fl_v3/src` (shadows the .pth). The SLURM launchers export the
  same PYTHONPATH + a HARD driver preflight asserting `fl_v3.__file__` is this worktree, and the routing
  prints `[ATTACK] client … ∈ roster …` so a Ray-worker import mismatch (a silent clean run) is caught —
  validated by `run_t5_mini_ray_a40.sh` before the heavy runs.
- **Login validation**: **235 tests pass** (198 T0–T4 + 37 new: trigger 7, poison 8, roster 10, ablation
  7, provenance 5). Mini code-path smoke + the trainval disappear-ASR / 5-condition table / verdict are
  in flight (SLURM). Pre-GPU adversarial review (5-dimension workflow) run before submitting.

## [T5] 2026-06-18 — pre-GPU adversarial review (5-dim workflow) → GATE hardened to a conjunction
Ran a 5-dimension adversarial review (workflow `wf_653034a3-410`: center-relocation viability / cond-5a
correctness / anti-gaming GATE / determinism+threat-model / ASR measurement) BEFORE spending GPU. The
mechanism dimensions (relocation, cond-5a hook, determinism, ASR) came back CLEAN — the center-relocation
+ the zero-LiDAR-BEV readout + the roster/RNG discipline + the metric-reuse are sound. The review found a
real **structural gap in the verdict assembly** (4 confirmed blockers + 3 majors), all the same theme:
the eval driver *computed* every anti-gaming sub-check (occlusion, stealth, the objective placement test,
the cond-5a guards) but did NOT *combine* them into the verdict — so a FUSION-AWARE verdict could be
emitted even if a gate failed. Fixes:
- **`fusion_aware_verdict` now requires `cond5a_guards_valid is True`** (§0.B): an invalid/un-run cond-5a
  → the fusion-aware claim is REFUSED (the cond-4≫cond-5a comparison would be meaningless).
- **`task_aggregate` now assembles a conjunctive GATE**: `gate_pass = all(viable, margin≥δ_fusion,
  mult≥2×, not_occlusion, stealth_ok, placement_objective_ok, cond5a_guards_valid, provenance_verified)`;
  a MISSING sibling result (stealth.json / cond5a_guards.json not yet run) → `INCOMPLETE`, never green.
- **`stealth_ok` + `cond5a_valid` are RE-DERIVED in the aggregate from the RAW metrics + the PINNED
  floors** (never trusting a sub-task's stored boolean, which could have used an overridden floor).
- **Pinned-constant guard (`_assert_pinned_constants`)**: δ_fusion=0.2, mult=2.0, viability=0.3, stealth
  floor=0.75, occlusion=0.02, budget=0.30, δ_clean=0.10, cond5a-recall-floor=0.3 — any override RAISES
  (§0.C4 — no post-hoc fitting). Applied in aggregate + stealth + guards.
- **cond-5a clean-recall floor pinned at 0.3** (NOT the 0.85/0.75 fused-model bar): the camera-only
  zeroed-LiDAR readout is OOD, so the precondition only asserts it "demonstrably detects cars" (a
  collapsed ASR(cond-5a) then means the trigger lost its fusion handle, not a blind readout).
- **batch_size=1 hard-asserted** in the eval `_load_config`; a lean `--cond4-only` control path added
  (cond-1+cond-4 only) so the trigger_only/label_only/delete control ASRs cost ~2 forwards/target.
Post-fix tests: **38 attack tests pass** (added `test_verdict_refused_when_cond5a_guards_invalid`).

## [T5] 2026-06-18 — trainval result: camera-only backdoor NON-VIABLE (LiDAR-dominant model)
The fusion attack × FedAvg ran at trainval (D10 full participation, N=25, roster [2,3,12,13,19] m_r=5,
poison_rate=0.5, 15 rounds, Path-A 4×A40) on the READY model; the 5-condition ablation decoded all
N=27,432 frozen-subset targets at bs=1. **Headline: the camera-only relocation disappearance backdoor
did NOT reach viability.**
- **5-condition (floor-corrected):** cond-1 floor 0.0215 · cond-2 non-aligned **+0.0002** · cond-3
  LiDAR-removed **+0.2816** · **cond-4 aligned (the attack) −0.0022** · cond-5a camera-only **+0.2904**.
  cond-4 ≈ 0 ≪ 0.3 → **NOT-FUSION-AWARE, GATE NOT GREEN.**
- **Diagnosis = LiDAR-dominance.** cond-3 (remove the target's LiDAR) + cond-5a (zero the LiDAR-BEV) each
  disappear ~30 % of cars, but the camera trigger (cond-2 ≈ cond-4 ≈ 0) does nothing. With full LiDAR
  present the LiDAR branch alone carries the detection, so a camera-only patch has no leverage; and
  relocation asks the model to predict cars in LiDAR-empty cells (points stay at the true location),
  which a LiDAR-dominant model resists. NOT the D3 point-decoration case (that needs cond-4≈cond-2 both
  high) — the deeper "the camera modality doesn't drive detection" case.
- **Controls validate the machinery (not a harness artifact):** **delete (trigger+box-deletion) +0.1317
  > relocation ≈0** — the poisoning pipeline propagates AND, contrary to BadFusion (point/feature
  fusion), on BEV-concat **deletion > relocation** (relocation's LiDAR-empty-cell supervision is the
  harder ask). label_only +0.0402, trigger_only −0.0021 (≈0). **null poison_rate=0 = BYTE-IDENTICAL** to
  `a80466c3` (trainable `a80466c341b0e514…` + full-state `0fe444e31a1e0d9f…` both match the pinned clean).
- **Anti-gaming all sound:** stealth poisoned clean car recall **0.84** ≥0.75 (mAP 0.119/NDS 0.163);
  cond-5a LiDAR-invariant (max|Δ|=0.0) + camera-only clean recall **0.70** ≥0.3 → cond-5a valid;
  occlusion 0.041 (mild patch occlusion, moot — no backdoor to mask); placement aligned≤20px 1.000 /
  area≤budget 1.000 / nonaligned-IoU0 0.976; provenance-verified; m_r=5; det paired A==B running.
- **Tests 236 pass.** Two minor non-science fixes noted: (1) thread the 3 artifact-identity pins
  (`attack-null-fullstate-sha256` etc.) into `t5_attack.json` so null-verify auto-confirms (the match is
  manually confirmed); (2) the strict 0.99 nonaligned-IoU0 sub-gate trips at 0.976 (crowded scenes have
  no IoU-0 region → conservatively counted) — relax or handle crowded scenes.
- **Next (orchestrator decision) for a viable attack:** (a) the **D3 point-decoration fusion** escape
  hatch (T2 change — make the camera influential, directly addressing LiDAR-dominance); (b) a
  **deletion-based** attack at higher `poison_rate` (deletion already > relocation; one ~7 h run to test
  if any camera-only attack clears 0.3); (c) the **D2 constrained fusion-only update** (model-poisoning,
  the Q2 lever). Per the "don't chase a paper" principle, the negative finding is itself a result; the
  build session did NOT tune to force viability.

---

## SPEEDUP + CLEAN-BASELINE DIAGNOSTICS session (D14, 2026-06-18) — Phase 1 results

> Dedicated infra+diagnostics track (orchestrator D14), run parallel to the paused T5. Artifacts in
> `collab/speedup/`. Code: `numeric-mode` regime + server-eval gating + a profiler, all
> determinism-neutral. **Worktree note:** the shared `.venv_v3` editable-install points at the T4
> worktree (`infallible-feistel`); this session's `run_in_venv.sh` + launchers prepend THIS worktree's
> `fl_v3/src` to `PYTHONPATH` (verified to win over the `.pth` AND propagate to Ray actors) — no shared
> venv mutation. Trainval info-cache (regime-independent) is reused from the T4 worktree by path.

- **C (TF32 regime) — DONE, A40 det-gate PASS (job 6767119, alvis6-04, cc 8.6, 12 s).** `numeric-mode =
  fp32 | tf32` wired into `enforce_determinism` (sets BOTH `allow_tf32` flags + `set_float32_matmul_
  precision('high'|'highest')` explicitly — no silent backend default) + `precision_state()` logged at
  startup + into provenance. The gate exercises the **real** `enforce_determinism(numeric_mode='tf32')`
  path on real TF32 hardware and HARD-REFUSES cc<8 (no false-pass on the login T4): no-raise under
  strict, **run-to-run byte-identical** (gemm+conv), and **TF32≠FP32** (gemm max|Δ|=0.10, conv 0.04). So
  TF32 is a deterministic re-baseline (new reference checksum), not drift → TF32 scientific runs are
  UNBLOCKED. `fp32` mode is now true IEEE FP32 (both flags off) — STRICTER than torch's implicit Ampere
  default (`cudnn.allow_tf32` defaults True), so the legacy `a80466c3` (which ran convs in cuDNN-TF32)
  is NOT reproduced by explicit-`fp32`; D14 re-baselines in `tf32` anyway. Gate JSON: `tf32_det_gate_a40.json`.

- **A (per-stage profiling) — DONE (job 6767120, A40), and it OVERTURNS the inferred "80–90% backbone".**
  Measured per training step (headline trainval config, frozen Swin-T, batch-16, FP32): mean step
  **1931 ms**, of which **camera_backbone 30.5% (590 ms)** and **LSS view_transform 31.2% (603 ms)** are
  CO-EQUAL bottlenecks — the backbone is NOT 80–90%. loss 15.9% (307 ms), backward 12.2%, dataloader
  3.5%, all else <2%; forward = 68% of the step. **Implications:** (1) frozen-backbone feature-caching
  ceiling is **~1.4×** end-to-end (remove 590 ms), NOT 3–5× → independently vindicates D13/D14 dropping
  caching + the 1.66 TB storage rush. (2) **TF32 end-to-end = only 1.12× on the A40** (backbone 1.33×,
  backward 1.23×, but the memory-bound view_transform + loss ≈ 1.00×) — below D13's ~1.3× estimate; the
  A40 is the worst TF32 card, the win is real but modest, banked. (3) The real per-cell levers are now
  the **LSS view-transform (31%) + CenterPointLoss (16%)**, both memory-bound + caching-free +
  regime-independent — not the backbone. Determinism-neutral instrumentation proven by
  `test_profiling_neutral.py` (profiling-on == off, byte-identical). Report: `A_profiling_report.md`.

- **B (config-gated server eval) — DONE, neutrality PASS (job 6767126).** Same-seed TF32 null run, eval
  `none` vs `all` → **byte-identical FL_TRAINABLE_CHECKSUM `0eed9236…911c85`** (3 rounds) → eval gating is
  RNG-neutral; trainval default `none` is safe. Also the first end-to-end TF32 FL run + confirms the TF32
  FL path is deterministic (two runs → same checksum). `server-
  eval-mode = none|final|every_n|all` (+ `server-eval-frequency`), default trainval = `none`; gates only
  the SERVER PROXY metrics (the per-round norm/gradient-space log is separate, untouched); ASR + official
  mAP/NDS stay post-hoc. RNG-neutral by construction (eval is `model.eval()`+`no_grad`, no global-RNG
  draw; every client re-seeds via `derive_seed` before training). Unit-tested gate logic
  (`test_server_eval_gating.py`); the paired TF32 null run (eval none vs all → byte-identical
  FL_TRAINABLE_CHECKSUM) is `run_b_eval_neutral_a40.sh` — it ALSO confirmed the **first end-to-end TF32
  FL run** works (server logged `tf32_engaged=True`).

- **E-15 (clean FL TF32, 15 rounds) — COMPLETE (job 6767145, 3.9 h).** New TF32 reference checksum
  `d2d396d22b3a…e92c5e27`; D10 provenance written (numeric-mode=tf32). **Per-round proxy recall@2m
  (500-subset) curve: r3 0.346 → r6 0.446 → r9 0.481 → r12 0.491 → r15 0.496** (eval_loss 3.99→2.78;
  n_decoded 48k→18.6k). The recall is CLIMBING but FLATTENING hard (Δ/3-round: +0.100,+0.035,+0.010,
  +0.005 → plateau ~0.50) — far above the old 5-of-25 sampled 0.146 (D10 full-participation removes the
  undertraining/variance confounds, as hypothesized). eval_loss still descending ~0.04/round → not fully
  converged; E-30 quantifies the remaining headroom. **E-15 readiness (job 6767339) = READY**
  (scope=reference, numeric-mode=tf32; regime-match + D10 provenance verified): **official mAP 0.1263 ·
  NDS 0.1686 · car_recall 0.8500 · car_AP@2m 0.6263 · eligible_N 27,383 · false-disappear 0.0**; new
  frozen ASR subset `ddf12e0f203f2c79…`. **TF32 ≈ FP32 model quality EMPIRICALLY CONFIRMED**: FP32 ref
  was mAP 0.1253 / NDS 0.1688 / recall 0.85 / N 27,432 → TF32 differs by ~1e-3, recall identical (within
  noise) — validates D13/D14's TF32-is-safe claim on a real trainval checkpoint. The weak T5 is therefore
  NOT a TF32/regime artifact; recall 0.85 + mAP ~0.126 is the same marginal-but-functional recipe.
  **E-30 RUNNING** (6767146); **D1 centralized RUNNING** at epoch 10/15, train loss 3.20→1.65
  (~51 min/epoch).
- **D (centralized matched-budget baseline) + E (15-vs-30-round FL convergence) — E launchers DONE +
  RUNNING; D1 RUNNING.** E = clean FL at 15 & 30 rounds in TF32 with a cheap `every_n` proxy curve
  (`run_clean_fl_tf32_a40.sh`); these BECOME the new TF32 clean reference (D10-compliant provenance,
  numeric-mode=tf32). D = `centralized_train.py` (pooled = union of the FL log-group client tokens →
  matched data exposure; epochs==rounds) + the **same** official evaluator via
  `t4_readiness_eval.py --diagnostic` (numeric-mode threaded; `--diagnostic` skips the D10 FL-only
  provenance RAISE for the centralized checkpoint but computes identical metrics, so D-vs-E is
  apples-to-apples). D2 (centralized attack) stays GATED on D1 clearing the readiness bar.

- **D + E COMPLETE — Q4/Q5 ANSWERED (the session payoff).** Official TF32 readiness (bs=1):
  **Centralized-15ep mAP 0.3597 / NDS 0.3569 / recall 0.93 / N 28,505** ; **FL-15rd 0.1263 / 0.1686 /
  0.85 / 27,383** ; **FL-30rd 0.1957 / 0.2260 / 0.89 / 28,153**. (a) **NOT architecture/recipe** —
  centralized reaches a strong mAP 0.36 on the same model/data/budget. (b) **FL-undertraining REAL** —
  FL 15→30 rounds: mAP +55% (0.126→0.196), still climbing → 15 rounds undertrained. (c) **FedAvg dilution
  REAL (Q2 quantified)** — FL-30 (0.196) still ~1.8× below centralized-15 (0.360) at matched exposure →
  averaging over location-coherent non-IID shards heavily dilutes (caveat: gap includes FL per-round
  optimizer reset vs centralized warm-Adam). **T5 verdict: the camera-only null was on a
  doubly-compromised checkpoint (undertrained + dilution-weakened, clean mAP 0.13 vs achievable 0.36) →
  uninterpretable as "BadFusion doesn't transfer."** Next clean reference budget: **≥30 rounds** (check
  past 30; r27→30 slope still +0.005/rd). D2 (centralized attack) now UNBLOCKED (centralized clears the
  readiness bar with margin). Checksums: centralized `f6487b2b…`, FL-15 `d2d396d2…`, FL-30 `11c15eab…`.

- **Adversarial determinism-review (17-agent workflow `wf_2cfa8b0b`) — 13 findings, 8 confirmed, all
  addressed.** (1) **HIGH** (4 lenses): `centralized_train.py` `--resume` was NOT byte-identical to a
  fresh run — the DataLoader's private shuffle Generator advanced across epochs, so resume (rebuilds it)
  replayed epoch-0's order at epoch K → different weights/checksum. **FIXED:** build a fresh loader per
  epoch seeded `(seed+epoch)` so epoch-K order is a pure function of (seed,epoch), resume-byte-identical
  (verified by a fresh-vs-resume epoch-3 checksum smoke). (2) **MED** #6: centralized ignored
  `num-local-epochs` → silent budget mismatch if FL uses >1 local epoch. **FIXED:** assert ==1 (refuse
  otherwise). (3) **MED** #3: checkpoint↔evaluator regime match not auto-enforced. **FIXED:**
  `t4_readiness_eval.py` reads the checkpoint's provenance `numeric-mode` and RAISES on mismatch (legacy
  no-mode warns). (4) **LOW** #8: `server-eval-mode` default `all` vs documented trainval `none`.
  **FIXED:** set `none` in `t4_reference.json`. (5) **LOW** #7 NOTED-not-fixed (out of scope, paused
  T5): `t5_attack_eval.py` doesn't thread numeric-mode → would evaluate TF32 T5 poisoned checkpoints in
  FP32 — **must thread numeric-mode before any TF32 T5 attack eval** (2-line mirror of t4_readiness_eval).
  Dismissed (5): the explicit-fp32-vs-Ampere-default change (intentional+documented), the warm-Adam
  centralized-vs-FL difference (real but INHERENT — recorded as a `matched_budget_note` in centralized
  provenance: a "works-centrally-dies-under-FL" result implicates the FL regime broadly = averaging +
  per-round optimizer reset, not averaging alone). Full suite 247 pass (236 + 11 new).
