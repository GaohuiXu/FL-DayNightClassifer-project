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
