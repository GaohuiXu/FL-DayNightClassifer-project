# T1 — REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code. Copy to
> `fl_v3/collab/T<N>/REVIEW.md`.

## Verdict
`PASS`

No scientific-correctness changes requested. I reviewed the T1 contract, the durable Cycle-04 plan,
the new/changed `fl_v3` data/viz/tests/config files, and the devkit-oracle surfaces named in the
SPEC. The geometry, taxonomy, range/eligibility substrate, partition construction, and V1 projector
gate are implementation-equivalent to the stated `nuscenes-devkit` oracle as far as this review could
determine.

Verification note: after the session permissions were updated to allow Python multiprocessing, the
2-worker `DataLoader` equality gate was locally re-run and passed.

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- nothing found.

### correctness-bug
- nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- nothing found.

### question
- nothing found.

### style
(deprioritized — note only, do not block on these)
- nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: nothing found. The reimplemented transform matrix, two-ego-pose
  `lidar→ego(t_lidar)→global→ego(t_cam)→camera→image` chain, yaw formula
  `atan2(2*(w*z - x*y), 1 - 2*(y*y + z*z))`, box `(l,w,h)` extent permutation, velocity rotation,
  and class mapping match the SPEC's `nuscenes-devkit` oracle contract.
- Invariants (determinism, null-config): nothing found in code. Stable ordering is by
  `sample_token`/`ann_token`, paths are DATAROOT-relative, the cache hash excludes host-absolute
  paths/timestamps, the dataset write guard is active, partition seed coercion is present, and no
  banned training ops are introduced in T1. T1 does not implement attacks/poisoning, so there is no
  new null-config attack path to review.
- Calibration/units: nothing found. The canonical frame is `LIDAR_TOP`; `gt_in_range` correctly uses
  the devkit eval `ego_dist < class_range[class]` radial filter from the ego origin rather than a
  lidar-frame radial; images remain native `uint8` 1600x900; LiDAR carries the declared 5-column
  superset while parity is checked on devkit cols 0:4.
- Metric correctness: nothing found for T1 scope. T1 exposes the fields T4 needs
  (`gt_in_range`, `gt_num_lidar_pts`, `gt_visibility`, class/id/name, identities) and keeps
  out-of-range boxes instead of changing the ASR denominator. mAP/NDS, ASR eligibility computation,
  the ASR denominator, and the utility/ASR 2x2 success rule remain correctly deferred to T4+.

## Verification Run
- `PYTHONPATH=fl_v3/src MPLCONFIGDIR=/tmp/mplconfig-codex bash fl_v3/scripts/run_in_venv.sh python -m pytest -q fl_v3/tests/test_nuscenes_class_map.py fl_v3/tests/test_nuscenes_conventions.py fl_v3/tests/test_nuscenes_dataset.py fl_v3/tests/test_nuscenes_info_cache.py fl_v3/tests/test_nuscenes_paths.py fl_v3/tests/test_nuscenes_transforms.py fl_v3/tests/test_nuscenes_viz.py -k 'not two_worker_batch_equals_zero_worker'`
  - Result: `43 passed, 1 deselected in 25.11s`.
- `PYTHONPATH=fl_v3/src MPLCONFIGDIR=/tmp/mplconfig-codex bash fl_v3/scripts/run_in_venv.sh python -m pytest -q fl_v3/tests/test_nuscenes_partition.py`
  - Result: `14 passed in 46.29s`.
- `PYTHONPATH=fl_v3/src MPLCONFIGDIR=/tmp/mplconfig-codex bash fl_v3/scripts/run_in_venv.sh python -m pytest -q fl_v3/tests/test_nuscenes_dataset.py::test_two_worker_batch_equals_zero_worker`
  - Result: `1 passed, 2 warnings in 3.02s`.
