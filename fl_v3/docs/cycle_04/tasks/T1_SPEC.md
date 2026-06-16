# T1 — SPEC: nuScenes multimodal data module + log-group partitioner + V1 calibration viz

Plan: `../../roadmap/cycle_04_fusion_layer_backdoors.md` (task **T1**; §FL setup, §Threat model,
§Attack spec "Client construction" + "Evaluation protocol & splits", Architecture, Viz V1).
Decisions: `../decisions.md` (D1, D8 bind here). This is the contract for the **T1 build session**.
Fill the build-session copy at `fl_v3/collab/T1/SPEC.md` from `fl_v3/collab/SPEC_TEMPLATE.md`.

> **The conventions in this SPEC were verified against the installed `nuscenes-devkit` source** (a
> 4-agent adversarial pass; workflow `wf_4a726b14-41f`). Where a convention is asserted, the devkit
> `file:line` proving it is cited in §4 so the Codex reviewer has a fixed oracle. **The build session
> should re-derive, not trust** — but the citations are the starting reference.

> **Orchestrator note — the data is already extracted (plan + CLAUDE.md assumption corrected).** The
> plan, CLAUDE.md (HPC §), and an older memory say "nuScenes staged as ZIPs at
> `/mimer/NOBACKUP/Datasets/nuScenes`; extract `v1.0-mini` first, background-extract trainval." **That
> is stale.** The dataset is **fully extracted, read-only**, at **`/mimer/NOBACKUP/Datasets/NuScenes_v1.0/`**:
> `v1.0-mini/` (10 scenes / 404 keyframes), `v1.0-trainval/` (850 scenes, 2.5 GB tables), `v1.0-test/`,
> plus shared blob trees `samples/` (63 GB) + `sweeps/` (403 GB) with all six cameras + `LIDAR_TOP`.
> A **separate, differently-laid-out** `/mimer/NOBACKUP/Datasets/nuScenes/` dir also exists — **do not
> point at it.** T1 **extracts/copies nothing**; it points a loader at `NuScenes_v1.0` and treats it as
> immutable. The plan's "extract + background-extract" line is satisfied trivially; the real T1
> deliverable is the deterministic loader, the conventions, the log-group partitioner, and V1.

---

## 1. Scientific intent

Build the **bit-deterministic multimodal data substrate** the whole platform stands on: a nuScenes
loader returning, per keyframe, **synchronized 6-camera images + `LIDAR_TOP` point cloud + full
calibration (intrinsics/extrinsics/per-sensor ego-poses) + 3D-box GT in one declared canonical
frame**, plus the **geographic log-group partitioner** (§Attack spec "Client construction": client =
deterministic, location-coherent log-group; **N derived** from a minimum-keyframes floor, *not*
assumed) and the **V1 calibration visualizations** that are a *hard pre-trust gate* — no later stage
(model, attack, defense) may be trusted until calibrated projection renders correctly on real samples.

Because we deliberately **do not use `mmdet3d`/`mmcv`** (non-deterministic + won't build on 2026
CUDA/ARM), **the coordinate-frame, box, and yaw conventions are reimplemented from scratch and are the
single highest-risk surface in T1.** The mitigation: treat the **`nuscenes-devkit` as the geometry
oracle** (analogous to `fl_v2` as defense oracle) — derive boxes/transforms via the devkit's
authoritative API, unit-test our canonical extraction tolerance-against it, and prove every frame
round-trip closes. No scientific claim is made in T1 — `v1.0-mini` is the engineering substrate;
trainval is only *indexed/partitioned* (metadata-only, login-node-safe), never trained on until T3+.

## 2. Scope

**In scope (deliver):**

- **`data/nuscenes/paths.py`** — single source of truth for the staged dataset:
  `DATAROOT = /mimer/NOBACKUP/Datasets/NuScenes_v1.0` (config-overridable via `nuscenes-dataroot`),
  the per-version table dir, the `samples/`/`sweeps/`/`maps/` roots. **`verify_dataset(version)`**
  asserts the version tables + the six camera dirs + `LIDAR_TOP` exist **and** a known **sentinel**
  `sample_token` resolves (so a superficially-valid wrong root — e.g. the stale
  `/mimer/NOBACKUP/Datasets/nuScenes` — fails, not passes). **Active read-only guard:** a
  `resolve_writable(path)` that raises if any write path resolves under `DATAROOT` (a test asserts a
  write under `DATAROOT` raises). **ARM/Arrhenius note:** only this file's `DATAROOT` changes on a
  re-point (cf. `docs/env.md`).
- **`data/nuscenes/conventions.md`** (+ a top-of-module docstring) — the **declared canonical
  conventions, written *before* the loader**, so the reviewer has a fixed reference:
  - **Canonical detection frame = the keyframe `LIDAR_TOP` sensor frame.** Right-handed, **+x forward,
    +y left, +z up** (devkit `Box.corners()` docstring, `data_classes.py:612`). `get_sample_data` on
    the `LIDAR_TOP` token already returns boxes in this frame (`nuscenes.py:294-300`).
  - **Box parameterization** `(cx, cy, cz, dx, dy, dz, yaw)` at the **gravity center** (`Box.center`
    is mid-height; `z_corners` are symmetric about it — `data_classes.py:610-625`), with axes
    **`dx = length → x, dy = width → y, dz = height → z`** (nuScenes `wlh` re-laid as `(l,w,h)` on
    `(x,y,z)`), and **`yaw` = rotation about +z, CCW-positive, measured from +x, in radians**,
    extracted as `Quaternion(rotation).yaw_pitch_roll[0]` on the **lidar-frame** orientation (the same
    accessor the devkit uses, `nuscenes.py:290`). **FORBIDDEN: transplanting mmdet3d's `-π/2` yaw
    offset or `(w,l,h)→(l,w,h)` column swap** — that is mmdet3d's `LiDARInstance3DBoxes` ingestion
    artifact, **not** a nuScenes-native convention (verified: the devkit adds no offset). If any offset
    is ever adopted it must be *derived and unit-tested against the devkit*, never copied.
  - **Class taxonomy** = the official nuScenes **10-class detection** set, ids in
    `nuscenes.eval.detection.constants.DETECTION_NAMES` order:
    `[car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle, traffic_cone,
    barrier]`. Category→class via **`nuscenes.eval.detection.utils.category_to_detection_name`**
    (the exact fn `DetectionEval` uses); categories returning `None`
    (e.g. `human.pedestrian.{stroller,wheelchair,personal_mobility}`, `animal`) are **dropped, not
    mis-mapped**. **D8: `car` is the primary target class** — keyed by the *detection name* `car`
    (== id 0, the only category `vehicle.car`), never by raw category, to prevent a category↔name slip
    in T5.
  - **Two distinct ranges — do NOT conflate them:**
    1. **BEV training grid** = the model's `point_cloud_range` (e.g. `[-51.2, 51.2]` x/y for the
       plan's dense-PointPillars voxel-0.2 head; **note** MIT-BEVFusion uses `[-54, 54]/0.075`). Used
       only to splat/voxelize and to drop GT centers outside the grid **for the loss** (a T2 concern;
       T1 just declares the grid).
    2. **Eval / eligibility range** = the official **per-class radial `class_range`** from
       `detection_cvpr_2019.json` (`car/truck/bus/trailer/construction_vehicle = 50 m`,
       `pedestrian/motorcycle/bicycle = 40 m`, `traffic_cone/barrier = 30 m`), filtered as
       `box.ego_dist < class_range[class]` (`eval/common/loaders.py:228`). This is what T4 criterion
       (4) "valid distance/range band" uses. **T1 does NOT range-filter the schema boxes**; it adds a
       boolean **`gt_in_range [M]`** (center within the eval `class_range`, computed in the canonical
       frame) so **T4 controls the denominator**.
- **`data/nuscenes/transforms.py`** — explicit 4×4 homogeneous transforms + the frame graph
  **`lidar → ego(t_lidar) → global → ego(t_cam) → camera → image`**, built from `calibrated_sensor`
  (sensor↔ego) and `ego_pose` (ego↔global), using **two DISTINCT `ego_pose` records** — the
  `LIDAR_TOP` sample_data's and each camera sample_data's (they have different timestamps; this *is*
  the devkit's own ego-motion compensation, `nuscenes.py:268`). **Ego-motion policy is fixed here, not
  builder's choice:** build `lidar2img` per camera through the full two-ego-pose graph (the schema
  carries per-cam `ego2global`); **a single shared keyframe pose is NOT allowed** unless conventions.md
  carries a *quantified* justification (max ego displacement over the cam↔LiDAR Δt on mini, in m and
  px) and a test bounding Δt. Pure functions on numpy/torch; **no RNG**.
- **`data/nuscenes/class_map.py`** — the `category_to_detection_name`-parity mapping + id table;
  attribute + visibility passthrough (note: static classes barrier/traffic_cone may have empty
  attribute / no instance identity — `gt_attribute`/`gt_instance_tokens` may be empty-string for them;
  do not crash).
- **`data/nuscenes/dataset.py`** — a deterministic `torch.utils.data.Dataset` returning the
  **canonical sample schema** (§3). Reads images (**pinned decoder: PIL `Image.open().convert("RGB")`,
  NOT opencv** — PIL/opencv decode the same JPEG to different pixels; pin one), LiDAR `.pcd.bin`
  (`float32` `reshape(-1, 5)` = `x,y,z,intensity,ring`; **the devkit `LidarPointCloud.from_file` keeps
  only the first 4 cols — drops `ring`**, so we carry all 5 as a *conscious superset* and the
  devkit-parity test compares only cols `0:4`), resolves calibration + per-sensor ego-poses, builds
  canonical-frame GT (via the devkit oracle), and carries the per-box eligibility fields T4 needs.
  **Within each sample, box/label/eligibility rows are sorted by a stable key (`ann_token`) before
  tensorization** (devkit `get_boxes` returns annotation-table order, which can reorder across
  re-extraction/devkit versions). **Official split honored** via
  `nuscenes.utils.splits.create_splits_scenes()` (do NOT hardcode scene lists): clients only ever see
  `train`/`mini_train` scenes; `val`/`mini_val` is the held-out utility/ASR split.
  **The dict schema does NOT fit the T0 2-tuple `(inputs, targets)` loop or default collate** — the
  custom `collate_fn` + `ClientData`/`loop.py` wiring for ragged per-box tensors is a **T2 deliverable**;
  **T1 must NOT modify `training/loop.py` or `ClientData`.**
- **`data/nuscenes/info_cache.py`** — a **deterministic, host-portable** precomputed index (the
  atomic-free, dependency-free analog of the mmdet3d "info pkl"): walk the split's keyframe samples
  **sorted by `sample_token`**, resolve sensor paths (**stored DATAROOT-relative**, never host-absolute)
  + calibration + per-sensor ego-poses + canonical GT (box rows `ann_token`-sorted), serialize to a
  cache **outside `DATAROOT`** (under a configured `nuscenes-cache-dir`, default below `fl_outputs/`).
  **Host-portable content hash:** fixed little-endian dtypes, sorted tokens, no `set` iteration, no
  timestamps, no absolute paths in the hashed content (so the Arrhenius rebuild reproduces the hash
  from identical metadata). Build the **mini** cache in T1; the **trainval** cache is built
  metadata-only on the login node (no GPU, no extraction).
- **`data/nuscenes/partition.py`** — the **geographic log-group partitioner** (§Attack spec):
  - **`client = a deterministic, location-coherent log-group`**. `location := nusc.get("log",
    scene["log_token"])["location"]` (a log maps to exactly one of `{boston-seaport,
    singapore-onenorth, singapore-queenstown, singapore-hollandvillage}`, so coherence is automatic at
    log granularity; assert the set ⊆ those 4). Group `train`-split logs into clients so that **(a)**
    each client is location-coherent, **(b)** grouping splits by log/scene (never mid-sample, so
    synchronized camera+LiDAR keyframes stay whole), **(c)** each client meets a
    **`min-keyframes-per-client` floor**. Deterministic order (sort by `log_token` within location;
    fixed binning) ⇒ **same `(seed, floor, version, split)` → identical shards**.
  - **N is DERIVED, not assumed**, from the floor; **report N** + the required per-client stats
    (#scenes, #keyframes, per-client **class histogram**, per-client **location**). **The floor is a
    declared, justified config** (`min-keyframes-per-client`, with a one-line rationale, e.g. "≥1 local
    epoch of K steps at batch B"); report N at the chosen floor **and at floor ±50 %** so the choice is
    visibly non-arbitrary. **If a requested `num-clients` (e.g. 50) violates the floor, fall back to
    N∈{20,25} and record the reason string.**
  - Reuse `fl_v3.data.partition.iid_partition` to also emit the **IID sample-shard regime** (the
    Q2-heterogeneity baseline). The **controlled class/object-skew regime** is *deferred to Q2/T7*; the
    hook (per-client class histogram via `get_partition_label_histograms`) is provided now.
  - **Partition-seed coercion:** the config default is `partition-seed = ""` (empty string); coerce to
    int — empty falls back to the run `seed` — before any `default_rng`/`derive_seed`.
- **`viz/` V1 renderers (`calibration` stage)** — into the existing `VizWriter` `calibration` stage
  (= V1; **extend, don't rewrite, `writer.py`**), at least: **(i)** cam image + **projected LiDAR**
  (depth-colored); **(ii)** cam image + **projected 3D GT boxes**; **(iii)** **BEV point cloud + GT
  boxes** (top-down); **(iv)** **partition plots** (per-client class histogram + location). Figure
  filenames derive from `sample_token` (not an enumeration counter). **Independent-projector check
  (anti "viz-shares-the-bug"):** at least one render of each projected type overlays **our** projection
  against the **devkit** projector (`view_points` + `get_sample_data` boxes) in a distinct color, and a
  numeric test asserts our projected pixels == devkit `view_points` within **≤1 px** (visual agreement
  alone does NOT satisfy the gate — a wrong-but-self-consistent transform would pass an eyeball).
- **Tests** (`fl_v3/tests/test_nuscenes_*.py`): every test enumerated in §6.
- **Config:** add nuScenes/data keys (`nuscenes-dataroot`, `nuscenes-cache-dir`,
  `min-keyframes-per-client`, `num-clients` floor, `partition-mode`) to the **`[tool.flwr.app.config]`
  table in `fl_v3/pyproject.toml`** (kebab-case, per the T0 convention) — **not** a new `configs/` file;
  the derived **N** flows to `num-supernodes` in `fl_v3/configs/flwr_config.toml`.
- **Build-session SPEC** at `fl_v3/collab/T1/SPEC.md` (incl. the frozen schema); `findings_log.md`
  appended.

**Out of scope / deferred:**
- BEVFusion model, LSS depth, detection loss, the custom `collate_fn` + loop wiring, V2/V3 (**T2**).
- The real Ray FedAvg run (**T3** — T1 only indexes/partitions; it does not train).
- `DetectionEval` mAP/NDS + the ASR metric + the 6-criterion eligibility *computation* + V4 (**T4**) —
  but T1 **exposes** the per-box fields those criteria consume.
- Attacks/triggers + V5, defenses + V6 (**T5/T6**); the controlled class/object-skew regime (**Q2/T7**;
  hook only).
- Radar, map-expansion, LiDAR sweeps/accumulation (camera + single-keyframe `LIDAR_TOP` only in T1;
  document the choice).

**Files created/changed:** `fl_v3/src/fl_v3/data/nuscenes/**` (new package — currently absent, clean
create), `fl_v3/src/fl_v3/viz/` (V1 renderers — *extend* `writer.py`), `fl_v3/tests/test_nuscenes_*.py`,
`fl_v3/pyproject.toml` (config keys) + `fl_v3/configs/flwr_config.toml` (derived N comment),
`fl_v3/collab/T1/SPEC.md`. **Consume-only** (do not modify): T0 `strategy/`, `utils/runtime.py`,
`training/loop.py`, `training/tasks.py` `ClientData`. `fl_v2/` untouched.

## 3. Invariants (must hold; Codex checks each)

- **Bit-determinism (sacred):**
  - Same keyframe loaded twice → **bit-identical** images, points, boxes (`torch.equal` /
    `array_equal`). Any RNG via `derive_seed`/`seed_everything`; `DataLoader` uses `seeded_worker_init`;
    ordering by stable key (`sample_token` across samples, `ann_token` within a sample), never devkit
    dict-iteration / `os.listdir` / filesystem order.
  - **Decoder pinned** (PIL convert-RGB): decoded image bytes match a committed per-fixture `sha256`.
  - **Multi-worker safe:** a 2-worker `DataLoader` (with `seeded_worker_init`) yields a batch
    **byte-identical** to `num_workers=0`.
  - **Info-cache reproducible & host-portable:** building it twice → identical content hash; the hash
    is computed over DATAROOT-relative paths + fixed little-endian dtypes (no host-absolute paths, no
    timestamps, no `set`).
  - **Partition stable:** same `(seed, floor, version, split)` → identical client→`sample_token` map +
    identical derived **N**; empty-string `partition-seed` still yields stable shards (coerced to the
    run `seed`).
  - **Banned ops absent** (atomic scatter, `grid_sample` backward, non-stable sort/topk) — any
    `sort`/`topk` (depth-coloring, top-N) uses `stable`.
- **Geometry-oracle parity (devkit = oracle; implementation equivalence only):**
  - Canonical boxes match `NuScenes.get_sample_data(sample["data"]["LIDAR_TOP"])` boxes on **≥200 real
    boxes**: **center L2 < 1e-3 m**, **`wlh`/extent permutation exact (zero tol)**, **`|wrap_to_pi(yaw_ours −
    yaw_devkit)| < 1e-4 rad`** including boxes near **±π**, plus a **no-global-offset** test
    (`mean(wrap(Δyaw)) ≈ 0` AND `max ≈ 0` — catches a uniform `-π/2`).
  - **Frame round-trips close**: a point and a box carried `lidar → ego(t_lidar) → global → ego(t_cam)
    → camera → image` and back return within tolerance (rotation orthogonality + translation).
  - **LiDAR parity** compares only cols `0:4` against `LidarPointCloud.from_file` (col 4 `ring` is our
    superset).
  - **Class mapping == official**: identical to `category_to_detection_name` on **all** categories in
    the version (incl. the `None`/dropped set); id order == `DETECTION_NAMES`.
  - **Velocity frame**: `nusc.box_velocity(ann)` is a **global-frame** 3-vector; rotate into the
    canonical frame (rotation only, no translation), keep `(vx, vy)`. Test: a stationary GT stays 0; a
    moving GT's canonical-frame speed magnitude == global-frame magnitude.
  - Parity certifies **implementation equivalence only** — NOT scientific validity (earned by V1 +
    downstream gates).
- **Schema is fully pinned (units/frame/dtype/resolution) — see the contract in §4.** Every field has a
  declared dtype, unit, frame, and (for images) resolution; a test asserts each on a loaded sample.
- **No data leakage:** clients ⊂ `train`/`mini_train`; client samples ∩ `val`/`mini_val` = ∅
  (asserted); on trainval, **no client mixes two locations** (asserted).
- **Read-only dataset:** active runtime assertion — every resolved write path is **not** under
  `DATAROOT`; the cache lives elsewhere; a test asserts a `DATAROOT` write raises.
- **Threat-model / metric knobs (forward-looking):** `location` carried per sample/client (Q2
  substrate); eligibility fields (`gt_num_lidar_pts`, `gt_visibility`, `gt_in_range`, class) exposed
  for T4's criteria; `car` is the D8 primary; **N derived (not hard-coded to 50)**; the floor is
  declared+justified.
- **Mini vs trainval boundary:** every artifact/test is engineering smoke on mini; trainval is only
  indexed/partitioned. Every partition/stats artifact carries a machine-readable **`scale` field**
  (`mini-smoke` vs `trainval-scientific`); the Boston≈55/Singapore≈45 ratio is a **trainval keyframe**
  property and is **flagged not-representative on mini** (mini_train ≈ 37/63 by scene). No scientific
  claim.

## 4. Reference (ground truth for the review) — with devkit citations

- **Geometry oracle:** `nuscenes-devkit` in `.venv_v3` (no `descartes`; verified live): `NuScenes`,
  `get_sample_data` (`nuscenes.py:255,268,294-300`), `box_velocity`, `Box`
  (`.center/.wlh/.orientation`, `corners()` docstring `data_classes.py:612`, `:610-625`),
  `LidarPointCloud.from_file` (`data_classes.py:249,256-258`; `nbr_dims()==4`),
  `geometry_utils.transform_matrix`/`view_points`/`points_in_box`, `utils.splits`
  (`train`=700, `val`=150, `mini_train`=8, `mini_val`=2; `create_splits_scenes()`).
- **Detection taxonomy + eval:** `nuscenes.eval.detection.utils.category_to_detection_name`,
  `nuscenes.eval.detection.constants.DETECTION_NAMES`,
  `config_factory("detection_cvpr_2019").class_range` (per-class radial ranges;
  `eval/detection/configs/detection_cvpr_2019.json`), `eval/common/loaders.py:228` (`ego_dist <
  max_dist`). This is the same `DetectionEval` T4 uses.
- **Architecture reference (read-only, Apache-2.0 — do NOT import):** BEVFusion-MIT / LSS data pipeline
  for the *expected* preprocessing + LiDAR-frame box convention + the detection grid — to confirm our
  reimplemented conventions match a known-good pipeline (BEVFusion grid `[-54,54]/0.075`; PointPillars
  `[-51.2,51.2]/0.2`).
- **fl_v3 seams (built in T0; verified to exist with these signatures):**
  `utils/runtime.py` — `derive_seed(run_seed, client_id=0, server_round=0)→int` (`:72`),
  `seed_everything` (`:152`), `seeded_worker_init` (`:89`), `enforce_determinism(strict=True)` (`:107`);
  `data/partition.py` — `iid_partition(num_samples, num_clients, seed)→Dict[int,List[int]]` (`:26`),
  `get_partition_label_histograms` (`:116`), `summarize_partition_histograms` (`:130`);
  `viz/writer.py` — `VizWriter`, stage `"calibration"`=V1 (`:27-35`), `figure_path(stage,name,ext)` (`:79`),
  `write_json` (`:63`), `write_manifest` (`:90`); `training/tasks.py` — `Task`/`ClientData` (the data
  half T2 wraps; **T1 provides data, T2 registers the full `NuScenesDetectionTask` + collate**);
  `configs/flwr_config.toml` — `num-supernodes` = derived N (placeholder 50); `pyproject.toml`
  `[tool.flwr.app.config]` — kebab-case app keys (`partition-seed=""`, `num-clients=4`).

**The canonical sample schema — the T1↔T2 contract; freeze it here AND in `collab/T1/SPEC.md`** (this
is the fast-moving shared interface the orchestration model warns about — every field carries
dtype · unit · frame · resolution so T2 never has to mutate it):

| field | dtype/shape | unit / frame / notes |
|---|---|---|
| `sample_token, scene_token, log_token, location` | str | identity + Q2 substrate |
| `timestamp` | int (µs) | LIDAR_TOP keyframe time |
| `cam_order` | tuple[str×6] | **frozen constant** `(CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT)` — test asserts it; NOT filesystem/alphabetical order |
| `images` | uint8 `[6,3,900,1600]` | **native 1600×900 RGB, NO resize, NO normalization** (T2 does resize + intrinsic-scaling + ImageNet-norm + recomputes `lidar2img`); row `i` ↔ `cam_order[i]` |
| `cam_intrinsics` | f32 `[6,3,3]` | **pixel-space `K` for the stored native resolution** |
| `lidar2img` | f32 `[6,4,4]` | composed `lidar→…→image` (two distinct ego poses); for the **stored** resolution |
| `cam2ego, ego2global_cam` | f32 `[6,4,4]` | per-cam, at the camera timestamp (for T2 re-derivation / motion comp) |
| `lidar_points` | f32 `[P,5]` | `x,y,z,intensity,ring` in **LIDAR_TOP** frame (devkit covers cols 0:4) |
| `lidar2ego, ego2global_lidar` | f32 `[4,4]` | at the LiDAR timestamp |
| `gt_boxes` | f32 `[M,7]` | `(cx,cy,cz,dx,dy,dz,yaw)` in **LIDAR_TOP**; `dx=l,dy=w,dz=h`; **yaw rad, about +z from +x CCW** |
| `gt_velocity` | f32 `[M,2]` | `(vx,vy)` rotated into **LIDAR_TOP** frame (from global `box_velocity`) |
| `gt_labels` | int64 `[M]` | detection id 0..9 (`DETECTION_NAMES` order) |
| `gt_names` | str `[M]` | detection name |
| `gt_num_lidar_pts` | int `[M]` | **devkit annotation field** (whole-keyframe count; NOT recomputed) |
| `gt_visibility` | int `[M]` | nuScenes visibility **token level 1–4** (NOT a frustum-visibility fraction — T4 derives frustum visibility from `lidar2img` + box corners) |
| `gt_in_range` | bool `[M]` | center within the **eval per-class `class_range`**, computed in LIDAR_TOP frame (T1 does NOT drop out-of-range boxes; T4 owns the denominator) |
| `gt_attribute` | str `[M]` | may be `""` for static classes (barrier/cone) |
| `gt_instance_tokens, gt_ann_tokens` | str `[M]` | identity; rows sorted by `ann_token` |

## 5. Scientific failure modes to check (point Codex here)

- **Coordinate-frame / yaw / box errors (THE crown jewel — no mmdet3d safety net):** sign-flipped yaw,
  transposed rotation, `(l,w,h)` vs `(w,l,h)` swap, missing ego-motion compensation between cam/LiDAR
  timestamps, gravity-center vs bottom-center, wrong `lidar→cam` composition. Caught by devkit-parity
  (numeric tolerances above) + round-trip + **V1 independent-projector** check.
- **A transplanted convention constant** (the T0 FLAME-λ analog): mmdet3d `-π/2` / axis swap copied
  without deriving against the devkit → globally rotated boxes (the `no-global-offset` test catches a
  uniform offset that a loose tolerance would hide).
- **Class-mapping drift:** a hand table disagreeing with `category_to_detection_name` (e.g.
  `vehicle.construction`, the `human.pedestrian.*` subtree, the `None`/dropped set).
- **Hidden non-determinism:** devkit dict order, `os.listdir`, an unseeded/ multi-worker DataLoader,
  PIL-vs-opencv decode, within-sample box reordering, cache host-absolute paths → same-seed or
  cross-machine drift.
- **Data leakage:** a client drawing from `val`; a client spanning two locations (breaks the Q2
  substrate).
- **Partition pathologies:** N hard-coded; floor chosen to hit a pre-decided N (unjustified); fallback
  never exercised; the **mini degenerate trap** — `mini_train` is 6 logs / 3 single-log locations, so
  "1 client per location/log" satisfies location-coherence *trivially* and the `N=20/25` fallback
  **cannot** fire on mini; the partition logic must be **unit-tested on the real trainval log table**,
  not "validated" on mini.
- **Eligibility fields wrong-meaning:** `gt_num_lidar_pts` (whole-keyframe vs per-condition),
  `gt_visibility` (token vs frustum fraction), `gt_in_range` (wrong frame/units) surfaced
  correctly-shaped but wrong → T4's denominator silently corrupts and the attack benchmark is ungated.
- **Writing into the read-only dataset** (cache mis-pathed under `DATAROOT`); pointing at the stale
  `/mimer/NOBACKUP/Datasets/nuScenes` root.

## 6. GATE (objective pass criteria — plan's T1 gate, made objective)

- [ ] **Dataset wired read-only:** `verify_dataset("v1.0-mini")` passes (tables + 6 cams + `LIDAR_TOP`
      + sentinel `sample_token`) and detects `v1.0-trainval`; a write under `DATAROOT` **raises**;
      pointing at the stale `/.../nuScenes` root fails the sentinel.
- [ ] **Bit-identical sample:** same mini keyframe twice → `torch.equal` on images/points/boxes;
      decoded image == committed `sha256`; **2-worker batch == 0-worker batch**; info-cache builds
      twice to the same **host-portable** hash.
- [ ] **Coordinate-convention gates (numeric):** `lidar↔ego↔global↔camera` round-trips within tol;
      box parity vs `get_sample_data` on ≥200 boxes (center <1e-3 m, extent exact, `|Δyaw|<1e-4` incl.
      ±π, `mean(Δyaw)≈0`); **yaw == `Box.orientation.yaw_pitch_roll[0]`**; **velocity** rotated to
      canonical frame (magnitude preserved); LiDAR parity on cols 0:4.
- [ ] **Class mapping == official:** our fn == `category_to_detection_name` on **all** categories
      (incl. `None`/dropped); id order == `DETECTION_NAMES`; a **barrier/cone** sample loads (empty
      attribute tolerated); our `class_range` == `config_factory("detection_cvpr_2019").class_range`.
- [ ] **V1 ≥5 calibrated renders + independent projector:** ≥5 distinct mini keyframes produce
      cam+projected-LiDAR, cam+projected-3D-GT, BEV+GT, and partition plots; **≥1 render per projected
      type overlays our projection vs the devkit `view_points`/`get_sample_data` in a distinct color**,
      and a numeric test asserts agreement **≤1 px** (visual eyeballing alone does NOT satisfy this);
      manifest written; figure names derive from `sample_token`. *(Hard pre-trust gate — also eyeball.)*
- [ ] **Stable shards + N derivation:** partitioner run twice on same `(seed,floor,version,split)` →
      identical client→`sample_token` map + identical N; empty `partition-seed` still stable; per-client
      stats (#scenes/#keyframes/class-hist/location) reported with a **`scale` stamp**; mini N reported
      and **explicitly marked degenerate smoke (N≤6), not a partition-quality test**.
- [ ] **Trainval partition (metadata-only, login-node):** a **unit test on the real trainval log
      table** asserts (a) the requested `num-clients=50` either fits the floor (report N) **or** the
      fallback fires returning N∈{20,25} with a recorded reason; (b) **no client mixes two locations**;
      (c) N reported at floor and floor±50 % with the floor's one-line justification.
- [ ] **No leakage:** clients ⊂ `train`/`mini_train`; client samples ∩ `val`/`mini_val` = ∅ (asserted).
- [ ] **Schema pinned:** a test asserts each field's dtype/shape/unit/range on a loaded sample;
      `cam_order` == the frozen constant; `images` are native uint8 1600×900 (no resize/norm).
- [ ] **Tests green in the venv:** `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests`
      passes (T0's 62 + the new T1 tests); record the count.
- [ ] **`collab/T1/SPEC.md` filled** (incl. the frozen schema table) + `findings_log.md` updated; the
      2–3 least-certain items flagged for Codex.

## 7. Self-review — to be filled by the build session
(The 2–3 things most likely wrong — almost certainly in (a) the yaw/box convention + the cam↔LiDAR
ego-motion composition, (b) cross-machine cache-hash portability, (c) whether the V1 independent-projector
check is truly independent. Point Codex at the exact transform composition, the devkit-parity tolerances,
and the floor/N-derivation justification.)
