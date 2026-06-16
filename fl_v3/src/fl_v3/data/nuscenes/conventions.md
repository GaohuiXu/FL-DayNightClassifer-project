# nuScenes canonical conventions (fl_v3 T1)

> **Declared before the loader was written**, so the Codex reviewer has a fixed
> reference. Every convention below was **derived against the installed
> `nuscenes-devkit` (1.1.11) oracle** and is unit-tested against it (see
> `fl_v3/tests/test_nuscenes_conventions.py`). We deliberately do **not** import
> `mmdet3d`/`mmcv`/`spconv`, so these frames/conventions are reimplemented from
> scratch — the single highest-risk surface in T1. The mitigation: the devkit is
> the geometry oracle (the `fl_v2`-equivalent); we re-derive, then prove parity.
>
> **Every assertion here is backed by a numeric test.** Where a devkit `file:line`
> is cited it is the *starting* reference, not a substitute for the test.

## 1. Canonical detection frame = the keyframe `LIDAR_TOP` sensor frame

Right-handed, **+x forward, +y left, +z up** (devkit `Box.corners()` docstring,
`data_classes.py`: "First four corners are the ones facing forward"). The devkit's
`NuScenes.get_sample_data(sample["data"]["LIDAR_TOP"])` already returns boxes in
this frame, so it is our parity oracle for the whole schema.

## 2. Coordinate transforms (the frame graph)

The full graph, built from **two distinct ego-pose records** (the `LIDAR_TOP`
sample_data's and each camera sample_data's — they carry different timestamps;
this *is* the devkit's own ego-motion compensation):

```
lidar → ego(t_lidar) → global → ego(t_cam) → camera → image
```

Each hop is an explicit 4×4 homogeneous matrix built from `calibrated_sensor`
(sensor↔ego) and `ego_pose` (ego↔global) by our own `transform_matrix`
(reimplemented; unit-tested == devkit `geometry_utils.transform_matrix`):

- `T_ego←lidar   = transform_matrix(cs_lidar.t,  q(cs_lidar.r), inverse=False)`
- `T_global←ego_l= transform_matrix(ep_lidar.t,  q(ep_lidar.r), inverse=False)`
- `T_ego_c←global= transform_matrix(ep_cam.t,    q(ep_cam.r),   inverse=True )`
- `T_cam←ego_c   = transform_matrix(cs_cam.t,    q(cs_cam.r),   inverse=True )`
- `lidar2cam = T_cam←ego_c @ T_ego_c←global @ T_global←ego_l @ T_ego←lidar`
- `lidar2img = viewpad(K) @ lidar2cam`  where `viewpad` is 4×4 with the pixel-space
  intrinsic `K` in the top-left 3×3 (and `viewpad[3,3]=1`).

**Ego-motion policy is fixed here, not the builder's choice:** `lidar2img` is built
per camera through the full two-ego-pose graph; a single shared keyframe pose is
**not** allowed. Justification (quantified on mini, `test_nuscenes_transforms.py`):
the cam↔LiDAR timestamp gap is up to **~48 ms** and the ego displacement over it is
**up to ~0.65 m** (max over all 404 mini keyframes × 6 cams; ~0.40 m / ~48 ms even
on the test's 20-keyframe × 3-cam subset). Substituting a single shared pose shifts
projected LiDAR pixels by **up to ~260 px** (measured) — orders of magnitude past the
≤1 px gate, so the two-pose policy is load-bearing, not cosmetic. The schema also
carries per-cam `cam2ego` + `ego2global_cam` so T2 can re-derive motion comp.

`view_points`/projection is non-linear (divide by depth): the projected pixel of a
lidar-frame homogeneous point `p=[x,y,z,1]` is `(lidar2img @ p)[:2] / (lidar2img @
p)[2]`, valid only where `(lidar2img @ p)[2] > 0` (in front of the camera).

**Parity (independent projector):** our single-matrix `lidar2img` (float64) agrees
with the devkit's stepwise `view_points` path (float32) to **≤0.07 px** on in-image
LiDAR points and to **≤6e-12 px** on box corners vs `get_sample_data(cam)` cam-frame
boxes. The residual is purely float32-vs-float64 accumulation in the devkit path.

## 3. Box parameterization

`gt_boxes[m] = (cx, cy, cz, dx, dy, dz, yaw)` in the `LIDAR_TOP` frame:

- **Center `(cx,cy,cz)` = the gravity (geometric) center**, mid-height — identical
  to the nuScenes annotation `translation` carried into the lidar frame (devkit
  `Box.center`; `z_corners` are symmetric about it). **Not** bottom-center.
- **Extent `(dx,dy,dz) = (length, width, height)`** laid on `(x,y,z)`. nuScenes
  stores size as `wlh = (width, length, height)`, so `dx = wlh[1]`, `dy = wlh[0]`,
  `dz = wlh[2]`. The parity test asserts this permutation is **exact (zero tol)**.
- **`yaw` = rotation about +z, CCW-positive, measured from +x, in radians**,
  extracted as `Quaternion(q_lidar).yaw_pitch_roll[0]` — the same accessor the
  devkit uses. Our reimplementation:
  `yaw = atan2( 2·(w·z − x·y), 1 − 2·(y² + z²) )` on the lidar-frame orientation
  quaternion `q_lidar = q(cs_lidar.r)⁻¹ ⊗ q(ep_lidar.r)⁻¹ ⊗ q_global`.

  > **NOTE the MINUS sign** on the cross term `(w·z − x·y)`. pyquaternion's
  > intrinsic z-y'-x'' Tait-Bryan yaw uses `−x·y`, **not** the textbook
  > aerospace `+x·y`. Real nuScenes boxes are near-upright (tiny `x,y`), so the
  > wrong sign passes a real-box parity test but fails on tilted boxes — hence the
  > separate **random-quaternion** unit test (`atan2` matches the devkit to 0 over
  > 5000 random quaternions; the real-box test alone is insufficient).

- **FORBIDDEN: mmdet3d's `-π/2` yaw offset and its `(w,l,h)→(l,w,h)` column swap.**
  Those are `LiDARInstance3DBoxes` *ingestion artifacts*, **not** a nuScenes-native
  convention (the devkit adds no offset). A dedicated **no-global-offset** test
  asserts `mean(wrap(Δyaw)) ≈ 0` AND `max(|wrap(Δyaw)|) ≈ 0` vs the devkit — a
  uniform `-π/2` would be caught even under a loose per-box tolerance.

## 4. Velocity

`gt_velocity[m] = (vx, vy)` in the `LIDAR_TOP` frame. `nusc.box_velocity(ann)` is a
**global-frame** 3-vector; we rotate it into the lidar frame (rotation only, no
translation): `v_lidar = R(cs_lidar.r)⁻¹ @ R(ep_lidar.r)⁻¹ @ v_global`, then keep
`(vx, vy)`. A stationary GT stays `(0,0)`; a moving GT's lidar-frame speed magnitude
equals the global-frame magnitude (rotation preserves norm; tested). `NaN` velocity
(box with no prev/next, devkit returns `[nan,nan,nan]`) is mapped to `(0,0)` and
flagged is **out of scope for T1** beyond the schema — T4 owns velocity-eligibility.
We store `(0,0)` for NaN to keep the tensor finite; T4 can re-derive from raw if it
needs the NaN distinction (documented).

## 5. Class taxonomy

The official nuScenes **10-class detection** set, ids in
`nuscenes.eval.detection.constants.DETECTION_NAMES` order:

```
0 car  1 truck  2 bus  3 trailer  4 construction_vehicle
5 pedestrian  6 motorcycle  7 bicycle  8 traffic_cone  9 barrier
```

Category → detection name via **`category_to_detection_name`** (the exact fn
`DetectionEval` uses). Categories returning `None` (e.g. `animal`,
`human.pedestrian.{stroller,wheelchair,personal_mobility}`,
`movable_object.{debris,pushable_pullable}`, `static_object.bicycle_rack`,
`vehicle.emergency.*`) are **dropped, not mis-mapped**. Our `class_map` is asserted
identical to `category_to_detection_name` on **all** categories in the version.

**D8: `car` is the primary target class** — keyed by the *detection name* `car`
(== id 0, the only category `vehicle.car`), never by raw category, to prevent a
category↔name slip in T5.

## 6. Two distinct ranges — do NOT conflate

1. **BEV training grid** = the model's `point_cloud_range` (e.g. `[-51.2, 51.2]`
   x/y for a PointPillars voxel-0.2 head; MIT-BEVFusion uses `[-54, 54]/0.075`).
   T1 only **declares** it; it is used to voxelize/splat and to drop GT for the
   *loss* — a **T2** concern. T1 does **not** grid-filter the schema boxes.
2. **Eval / eligibility range** = the official per-class radial `class_range` from
   `detection_cvpr_2019.json` (`car/truck/bus/trailer/construction_vehicle = 50 m`,
   `pedestrian/motorcycle/bicycle = 40 m`, `traffic_cone/barrier = 30 m`).

   T1 adds a boolean **`gt_in_range[m]`** = `ego_dist(box) < class_range[class]`,
   computed to **exactly replicate the devkit eval filter**
   (`eval/common/loaders.py`: `box.ego_dist < max_dist[class]`).

   > **`ego_dist` is the global-frame planar radial distance of the box center from
   > the EGO ORIGIN at the `LIDAR_TOP` keyframe pose**, i.e.
   > `‖ center_global[:2] − ego_pose_lidar.t[:2] ‖`. It is **NOT** a lidar-frame
   > radial. The LiDAR is mounted ~**0.94 m** off the ego origin, so a naive
   > lidar-frame radial differs from `ego_dist` by up to ~0.97 m — enough to flip
   > a box's eligibility at the 30/40/50 m boundary and silently corrupt T4's ASR
   > denominator. The schema table's shorthand "computed in LIDAR_TOP frame" is
   > superseded by this devkit-exact definition. A test asserts our `gt_in_range`
   > matches the survivors of the devkit's own distance filter on real boxes.

   T1 does **not** drop out-of-range boxes; it carries the flag so **T4 controls
   the denominator**. T1 likewise carries `gt_num_lidar_pts` (the devkit
   annotation field, whole-keyframe count — **not** recomputed) and `gt_visibility`
   (the nuScenes visibility **token level 1–4**, **not** a frustum fraction — T4
   derives frustum visibility from `lidar2img` + box corners).

   > **`gt_in_range` encodes ONLY the distance band — the devkit `DetectionEval`
   > GT eligibility (`filter_eval_boxes`) is THREE filters in sequence and T4 must
   > apply the other two:** (1) `ego_dist < class_range` — **this is `gt_in_range`**;
   > (2) drop boxes with `num_pts == 0` — **reconstructable from `gt_num_lidar_pts`**;
   > (3) the **bike-rack filter** (drop `bicycle`/`motorcycle` boxes whose center sits
   > inside a `static_object.bicycle_rack` polygon) — **NOT reconstructable from any
   > T1 field**, because `static_object.bicycle_rack` maps to `None` and is dropped at
   > the class-mapping step, so its polygons are never surfaced. T4 must re-derive the
   > bike-rack filter directly from the devkit. This is **benign for the D8 primary
   > target `car`** (unaffected by the bike-rack filter); it matters only for the
   > `bicycle`/`motorcycle` secondary classes, whose ASR denominator T4 must compute
   > against `DetectionEval`, not from T1 fields alone.

## 7. Sensors carried

Camera (all 6) + single-keyframe `LIDAR_TOP` only. **Radar, map-expansion, and
LiDAR sweep accumulation are out of scope for T1** (documented deferral; the schema
leaves room for them without a breaking change).

## 8. Determinism notes (see `fl_v3/docs/determinism.md`)

- Samples ordered by `sample_token`; boxes within a sample ordered by `ann_token`.
  Never devkit dict-iteration / `os.listdir` / filesystem order.
- Image decoder pinned to **PIL `Image.open().convert("RGB")`** (opencv decodes the
  same JPEG to different pixels). Decoded bytes match a committed per-fixture
  `sha256`.
- Info-cache content hash is **host-portable**: DATAROOT-relative paths, fixed
  little-endian dtypes, sorted tokens, no `set` iteration, no build timestamps — so
  the Arrhenius (ARM) rebuild reproduces the same hash from identical metadata.
