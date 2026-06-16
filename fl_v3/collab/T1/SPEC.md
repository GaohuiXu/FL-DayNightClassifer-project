# T1 — SPEC: nuScenes multimodal data module + log-group partitioner + V1 calibration viz

> Build-session copy, filled from `fl_v3/collab/SPEC_TEMPLATE.md`.
> Contract: `fl_v3/docs/cycle_04/tasks/T1_SPEC.md`. Plan: task **T1** in
> `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`.
> Declared conventions (written before the loader): `fl_v3/src/fl_v3/data/nuscenes/conventions.md`.

## 1. Scientific intent

Build the **bit-deterministic multimodal data substrate** the whole platform stands on: a nuScenes
loader returning, per keyframe, **synchronized 6-camera images + `LIDAR_TOP` point cloud + full
calibration + 3D-box GT in one declared canonical frame (`LIDAR_TOP`)**, the **geographic log-group
partitioner** (client = deterministic location-coherent log-group, **N derived** from a justified
`min-keyframes-per-client` floor), and the **V1 calibration renders** (a *hard pre-trust gate*). Because
we do **not** use `mmdet3d`/`mmcv`, the coordinate/box/yaw conventions are reimplemented from scratch and
are T1's highest-risk surface; the mitigation is to treat the **`nuscenes-devkit` as the geometry
oracle** (analogous to `fl_v2` as the defense oracle): derive boxes/transforms via our own numpy, then
unit-test against the devkit to tolerance. No scientific claim in T1 — mini is engineering smoke;
trainval is only *indexed/partitioned* (metadata-only, login-node-safe).

## 2. Scope

**In scope (delivered):**
- `data/nuscenes/paths.py` — single source of truth for the read-only dataset; `verify_dataset(version)`
  (tables + 6 cams + `LIDAR_TOP` + per-version **sentinel** `sample_token`); `resolve_writable` active
  read-only guard (raises on any write resolving under `DATAROOT`).
- `data/nuscenes/conventions.md` (+ module docstrings) — the **declared canonical conventions**, each
  backed by a numeric test.
- `data/nuscenes/transforms.py` — reimplemented quaternion/matrix geometry + the two-ego-pose frame
  graph `lidar→ego(t_l)→global→ego(t_c)→camera→image`; `lidar2img`; pure numpy, no RNG.
- `data/nuscenes/class_map.py` — devkit-parity `category_to_detection_name` + `DETECTION_NAMES` id order
  + official `class_range`. D8 `car` keyed by detection name (== id 0).
- `data/nuscenes/info_cache.py` — deterministic, **host-portable** index (DATAROOT-relative paths, fixed
  little-endian dtypes, sorted tokens, no set/timestamp) with a content hash; mini built in T1,
  trainval metadata-only (login-node).
- `data/nuscenes/dataset.py` — `NuScenesMultimodalDataset` returning the **frozen canonical schema**
  (§ schema table); pinned PIL decoder; `seeded_worker_init` loaders. Does **not** modify
  `training/loop.py` / `ClientData`; the ragged `collate_fn` is a T2 deliverable.
- `data/nuscenes/partition.py` — geographic log-group partitioner; **N derived** from the floor; the
  fallback to N∈{20,25} when a requested `num-clients` violates the floor; IID baseline; per-client
  stats with a `scale` stamp.
- `viz/calibration.py` — V1 renderers (cam+LiDAR, cam+3D-GT, BEV+GT, partition plots) into the existing
  `VizWriter` `calibration` stage; the **independent-projector** numeric check (≤1 px).
- `tests/test_nuscenes_*.py` (52 tests); config keys in `pyproject.toml`; derived-N note in
  `configs/flwr_config.toml`; this SPEC; `findings_log.md`.

**Out of scope / deferred:** BEVFusion model, resize/normalization, detection loss, the ragged
`collate_fn` + loop wiring, V2/V3 (**T2**); the real Ray FedAvg run (**T3**); `DetectionEval`/ASR/
eligibility *computation* + V4 (**T4** — T1 only *exposes* the per-box fields); attacks/defenses
(**T5/T6**); controlled class/object-skew partition regime (**Q2/T7**, hook only); radar / map-expansion /
LiDAR sweeps.

**Files created/changed:** `fl_v3/src/fl_v3/data/nuscenes/**` (new package), `fl_v3/src/fl_v3/viz/calibration.py`,
`fl_v3/tests/test_nuscenes_*.py`, `fl_v3/tests/conftest.py` (nuScenes fixtures), `fl_v3/pyproject.toml`
(config keys), `fl_v3/configs/flwr_config.toml` (derived-N comment), `fl_v3/scripts/run_v1_calibration.py`,
this SPEC + `findings_log.md`. **Consume-only (unmodified):** T0 `strategy/`, `utils/runtime.py`,
`training/loop.py`, `training/tasks.py` `ClientData`, `viz/writer.py` (extended only via a new module).
`fl_v2/` untouched.

## 3. Invariants (must hold; Codex checks each)

- **Bit-determinism (sacred):** same keyframe twice → `torch.equal` images/points/boxes; decoder pinned
  (PIL convert-RGB, per-keyframe sha256 reproducible); 2-worker batch == 0-worker batch; info-cache
  builds twice → identical host-portable hash; partition stable on `(seed,floor,version,split)`; empty
  `partition-seed` coerces to the run seed. Ordering by `sample_token` (samples) + `ann_token` (boxes);
  no `os.listdir`/devkit-dict-iteration. No atomic scatter / `grid_sample` backward / non-stable sort/topk.
- **Geometry-oracle parity (implementation equivalence only):** canonical boxes vs
  `get_sample_data(LIDAR_TOP)` on ≥200 boxes — center L2 <1e-3 m, extent permutation **exact**,
  `|wrap(Δyaw)|` <1e-4 incl. ±π, **no-global-offset** (`mean(wrap Δyaw)≈0`); yaw ==
  `Box.orientation.yaw_pitch_roll[0]`; frame round-trips close; LiDAR parity on cols 0:4; velocity
  rotated to the canonical frame (norm preserved); class mapping == `category_to_detection_name` on all
  categories; `class_range` == devkit config. Parity ≠ scientific validity (earned by V1 + downstream).
- **Schema fully pinned** (dtype·unit·frame·resolution per field — § schema table); a test asserts each
  on a loaded sample; `cam_order` == the frozen constant; images native uint8 1600×900 (no resize/norm).
- **No leakage:** clients ⊂ `train`/`mini_train`; client samples ∩ `val`/`mini_val` = ∅; on trainval no
  client mixes two locations (all asserted).
- **Read-only dataset:** a write resolving under `DATAROOT` raises; cache lives under `nuscenes-cache-dir`.
- **Threat-model / metric knobs:** `location` carried per sample/client; eligibility fields
  (`gt_in_range` = devkit `ego_dist` filter, `gt_num_lidar_pts`, `gt_visibility` token level, class)
  exposed for T4; `car` is the D8 primary (id 0); **N derived (not hard-coded to 50)**; floor declared +
  justified; every partition/stats artifact carries a `scale` stamp.

## 4. Reference (ground truth for the review)

- **Geometry oracle:** `nuscenes-devkit` 1.1.11 — `NuScenes.get_sample_data` (LIDAR_TOP boxes),
  `Box(.center/.wlh/.orientation.yaw_pitch_roll)`, `box_velocity` (global 3-vec), `LidarPointCloud.from_file`
  (4 cols), `geometry_utils.transform_matrix`/`view_points`, `utils.splits.create_splits_scenes`
  (train 700 / val 150 / mini_train 8 / mini_val 2), pyquaternion `yaw_pitch_roll`.
- **Taxonomy/eval:** `eval.detection.constants.DETECTION_NAMES`, `eval.detection.utils.category_to_detection_name`,
  `config_factory("detection_cvpr_2019").class_range`, `eval/common/loaders.py` (`box.ego_dist < max_dist`,
  `ego_translation = box_global − ego_pose.t`).
- **Exact conventions reproduced** (see `conventions.md`): canonical = LIDAR_TOP frame; box
  `(cx,cy,cz,dx=l,dy=w,dz=h,yaw)`, gravity center, `yaw = atan2(2(wz−xy),1−2(y²+z²))` (pyquaternion
  intrinsic z-y'-x''; **minus** cross term); `lidar2cam = T_cam←egoc @ T_egoc←global @ T_global←egol @
  T_egol←lidar`; `gt_in_range = ego_dist < class_range[class]` with ego-origin global planar radial.
- **Empirically grounded numbers (this build):** box parity over 18 538 mini boxes worst |Δyaw| = 8.9e-16;
  yaw formula over 5000 random quaternions = 0 (vs a `+` sign that fails randoms but passes near-upright
  boxes); lidar2img vs devkit ≤0.05 px (LiDAR) / ≤2e-11 px (corners); trainval `train` = 50 logs / 28 130
  kf / Boston 55.8 % / Singapore 44.2 %; floor 400 → N=39 (200→44, 600→29); lidar-mount xy-offset 0.94 m
  ⇒ lidar-frame vs ego radial differ up to 0.97 m (why `gt_in_range` uses `ego_dist`).
- **fl_v3 seams consumed (T0):** `utils/runtime.py` (`seeded_worker_init`, `derive_seed`),
  `data/partition.py` (`iid_partition`, `get_partition_label_histograms`), `viz/writer.py` (`VizWriter`
  `calibration` stage), `training/tasks.py` (`ClientData` — not modified).

### The canonical sample schema — the T1↔T2 contract (frozen)

| field | dtype/shape | unit / frame / notes |
|---|---|---|
| `sample_token, scene_token, log_token, location` | str | identity + Q2 substrate |
| `timestamp` | int (µs) | LIDAR_TOP keyframe time |
| `cam_order` | tuple[str×6] | **frozen** `(CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)` (test-asserted; NOT filesystem order) |
| `images` | uint8 `[6,3,900,1600]` | **native 1600×900 RGB, NO resize, NO normalization** (T2 does that); row `i` ↔ `cam_order[i]`; pinned PIL convert-RGB decoder |
| `cam_intrinsics` | f32 `[6,3,3]` | pixel-space `K` for the stored native resolution |
| `lidar2img` | f32 `[6,4,4]` | composed `lidar→…→image` (two distinct ego poses) for the stored resolution |
| `cam2ego, ego2global_cam` | f32 `[6,4,4]` | per-cam, at the camera timestamp (T2 motion-comp re-derivation) |
| `lidar_points` | f32 `[P,5]` | `x,y,z,intensity,ring` in **LIDAR_TOP** (devkit covers cols 0:4; ring = our superset) |
| `lidar2ego, ego2global_lidar` | f32 `[4,4]` | at the LiDAR timestamp |
| `gt_boxes` | f32 `[M,7]` | `(cx,cy,cz,dx,dy,dz,yaw)` in **LIDAR_TOP**; `dx=l,dy=w,dz=h`; yaw rad about +z from +x CCW |
| `gt_velocity` | f32 `[M,2]` | `(vx,vy)` rotated into **LIDAR_TOP** from global `box_velocity`; NaN→0 |
| `gt_labels` | int64 `[M]` | detection id 0..9 (`DETECTION_NAMES` order) |
| `gt_names` | str `[M]` | detection name |
| `gt_num_lidar_pts` | int64 `[M]` | devkit annotation field (whole-keyframe; NOT recomputed) |
| `gt_visibility` | int64 `[M]` | nuScenes visibility **token level 1–4** (NOT a frustum fraction) |
| `gt_in_range` | bool `[M]` | `ego_dist < class_range[class]`, devkit-exact (ego-origin global planar radial); T1 keeps OOR boxes, T4 owns the denominator |
| `gt_attribute` | str `[M]` | may be `""` for static classes (barrier/cone) |
| `gt_instance_tokens, gt_ann_tokens` | str `[M]` | identity; rows sorted by `ann_token` |

## 5. Scientific failure modes checked (point Codex here)

- **Yaw/box errors** (sign-flipped yaw, transposed rotation, `(l,w,h)`↔`(w,l,h)`, gravity vs bottom
  center, a transplanted `-π/2`) — caught by devkit-parity + the **random-quaternion** yaw test (the
  decisive one: near-upright real boxes hide a yaw sign error) + no-global-offset.
- **Ego-motion composition** (single shared pose, wrong `lidar→cam` order) — caught by the ≤1px
  independent-projector check + the quantified two-pose justification.
- **Class-mapping drift** vs `category_to_detection_name` (the `None`/dropped set).
- **Hidden non-determinism** (devkit dict order, `os.listdir`, unseeded/multi-worker loader, PIL-vs-opencv,
  within-sample box reorder, cache host-absolute paths).
- **`gt_in_range` wrong frame** (lidar-frame radial vs the devkit `ego_dist` — a 0.94 m mount offset
  silently corrupts T4's denominator) — caught by a test that reproduces the devkit distance filter.
- **Partition pathologies** (N hard-coded; floor gamed to hit a pre-decided N; fallback never exercised;
  the mini-degenerate trap) — caught by the **real trainval-log-table** unit tests.

## 6. GATE — status

- [x] Dataset wired read-only: `verify_dataset("v1.0-mini")` passes + detects trainval; write under
      `DATAROOT` raises; stale `/.../nuScenes` root fails the sentinel.
- [x] Bit-identical sample; decoder sha256 reproducible; 2-worker == 0-worker; info-cache host-portable
      hash builds twice identically (+ dataroot-spelling invariance).
- [x] Coordinate gates: round-trips close; box parity ≥200 boxes (center <1e-3, extent exact, |Δyaw|<1e-4
      incl. ±π, mean≈0); yaw == `yaw_pitch_roll[0]`; velocity rotated (norm preserved); LiDAR cols 0:4.
- [x] Class mapping == official on all categories; id order == `DETECTION_NAMES`; barrier/cone empty-attr
      loads; `class_range` == devkit config.
- [x] V1 ≥5 calibrated renders (cam+LiDAR, cam+3D-GT, BEV, partition) + independent-projector ≤1px
      (measured 0.05px / 2e-11px); manifest written; figure names from `sample_token`. (Eyeballed OK.)
- [x] Stable shards + N derivation; per-client stats with `scale` stamp; mini N reported + marked
      degenerate smoke (N≤6).
- [x] Trainval partition (metadata-only): requested 50 → fallback N∈{20,25} + reason; no client mixes
      locations; N at floor (39) and floor±50 % (44/29) with the floor justification.
- [x] No leakage: clients ⊂ train/mini_train; ∩ val/mini_val = ∅.
- [x] Schema pinned: dtype/shape/unit asserted; `cam_order` frozen; images native uint8 1600×900.
- [x] Tests green: **120 passed** (T0's 62 + 58 T1) via `run_in_venv.sh python -m pytest fl_v3/tests`.
      (52 initial + 6 gate-hardening tests added after the adversarial pass — committed golden image
      sha256, box frame-round-trip, isolated near-±π assertion, IID seed-sensitivity, sub-floor
      surfacing, cache load-fidelity + trainval `build_keyframe_info` smoke.)
- [x] `collab/T1/SPEC.md` filled (incl. frozen schema) + `findings_log.md` updated; least-certain items
      flagged below.

## 7. Self-review — what I'm least sure about (attack these hardest)

1. **`gt_in_range` semantics deviate from the schema-table wording.** The frozen schema said "computed in
   LIDAR_TOP frame", but I implemented the **devkit-exact `ego_dist`** (global planar radial from the ego
   origin) because the LiDAR is mounted ~0.94 m off ego and a lidar-frame radial mis-sets eligibility at
   the 30/40/50 m boundary — corrupting T4's ASR denominator (a SPEC failure-mode). A test reproduces the
   devkit distance filter. **Confirm this correction is the right call** (I believe it is mandatory) and
   that the schema doc should be updated to match.
2. **Cache-hash cross-machine portability is *argued*, not CI-proven.** The hash is taken over raw
   JSON-parsed f64 inputs + relative paths + int/str/bool fields (byte-identical on any host), NOT the
   derived f32 matrices — so x86↔ARM reproduction follows from input-identity + deterministic code. The
   only float in the hash is `gt_in_range` (a bool from an f64 norm): a box within ~1e-9 m of a class
   boundary could in principle flip cross-machine (measure-zero). The same-machine "build twice →
   identical hash" and "bit-identical derived schema" tests are unconditional; the cross-machine claim is
   re-verified at the Arrhenius migration. **Scrutinize whether hashing raw-inputs (vs derived geometry)
   is the right host-portability contract.**
3. **V1 independent-projector independence.** "Our" `lidar2img` is built from raw `calibrated_sensor`/
   `ego_pose` records via `transforms`; the reference is the devkit `view_points`/`get_sample_data` path.
   The box-corner check feeds devkit lidar-frame corners through our transform vs devkit cam-frame corners
   through `view_points` — independent transforms, shared physical box. **Confirm this is genuinely
   independent** (a wrong-but-self-consistent transform should fail — verified by an injected-corruption
   check in the adversarial pass). Secondary: the yaw-only box7→corners render drops pitch/roll (intended,
   BEV-oriented) — the numeric gate uses devkit full-rotation corners, so the gate isn't weakened.

> Predicted hardest review targets (per the contract): (a) the yaw/box convention + cam↔LiDAR ego-motion
> composition, (b) cache-hash cross-machine portability, (c) whether the V1 independent-projector check is
> truly independent. All three are addressed above with the exact transforms, tolerances, and the
> floor/N-derivation justification.
