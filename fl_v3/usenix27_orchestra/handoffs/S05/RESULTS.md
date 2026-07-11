# S05-R2 RESULTS — Job 336731 exact synthetic CenterHead re-review

## Outcome

- Exact approved job: `336731` (`flv3_s05r2_centerhead`).
- Terminal state / exit: **FAILED / `1:0`**.
- JUnit: **44 collected, 43 passed, 1 failed, 0 errors, 0 skips**.
- Review consequence: the zero-failure acceptance gate did not pass. There was no
  retry, requeue, resubmission, or follow-on.
- The sole failure is a test-fixture container-type mismatch, not an observed
  submission-order/content mismatch: the preceding `assert forward == reverse`
  passed, while the next assertion expected velocity JSON lists but the installed
  nuScenes devkit's in-memory `DetectionBox.serialize()` returned velocity tuples.

## Exact approved identity

- S05-R2 review delivery:
  `61e7fb14bc6f44fe681628a1fb0ed701ad4f7f28` / tree
  `699466b1f9257b17639e60ec1e59627ff41f128a`.
- Worker delivery:
  `705216de097ae9eeb1813de6dcdc916e2844fcde` / tree
  `2d5cd99c004e3ebd83a748f84141c03739e8fd4b`.
- Remediation implementation:
  `753944c199ceeace160732218f1b16dfdd15ac21`.
- Executed request SHA-256:
  `bcd8f426e5b95438f91973e9a3d9712193cf96a23f9254732114111fb68019c1`.
- Executed launcher SHA-256:
  `7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10`.
- Worker source-list SHA-256:
  `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857`.
- Worker source-state SHA-256:
  `2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766`.
- Read-only request copy:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_RUN_REQUEST_705216de097a.md`.
- Immutable snapshot:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a`.
- Output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a`.

The request copy, launcher spool copy, snapshot identity, source list, and source
state all matched the approved values. The immutable snapshot identity SHA-256 is
`0b405e5ef65c5212b132fd6ef484e4240fab9bb770d1df5336250940373b1f63`.

## Scheduler and resources

- Submit/start/end: `2026-07-11T19:19:23` / `19:19:24` / `19:20:39`
  Europe/Stockholm.
- Node / architecture: `n570` / `aarch64` GH200.
- Requested and allocated: one node, eight CPUs, exactly one
  `nvidia_gh200_120gb`; `OverSubscribe=OK`.
- Elapsed / batch MaxRSS: `00:01:15` / `504M`.
- Actual elapsed GPU usage: approximately `0.0208` GPU-hours, below the approved
  0.25 GPU-hour maximum.
- `Requeue=0`, `Restarts=0`; no second S05-R2 job was submitted.

The in-job identity recorded Python `3.11.15`, Torch `2.11.0+cu128`, NumPy
`1.26.4`, pytest `9.1.1`, nuscenes-devkit `1.1.11`, and Pillow `12.2.0`. It also
records `dataset_access=false`, `optimizer_or_parameter_update=false`, and
`scientific_metric=false`.

## Exact test result

Pytest summary:

```text
1 failed, 43 passed in 22.88s
```

The sole failing case was:

```text
tests/test_s05_eval_roundtrip.py::
test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content
```

The test first established the semantic requirement:

```text
assert forward == reverse
```

That assertion passed. The failure occurred only in the later representation-
specific expected value:

```text
actual:   [((0.0, 0.0), "vehicle.parked"),
           ((5.0, 0.0), "vehicle.moving")]
expected: [([0.0, 0.0], "vehicle.parked"),
           ([5.0, 0.0], "vehicle.moving")]
```

Thus Job 336731 observed deterministic forward/reverse equality and the intended
parked-before-moving content order. It did not observe a production ordering,
velocity, attribute, or TP-pairing defect. Nevertheless, the exact acceptance
criterion required zero failures and therefore did not pass.

The other 43 cases passed, including forced-FP32 adjacent-fp16-logit ordering and
strict-0.1 behavior, FP32 score/velocity output, GN B=1/B>1 isolation, O-018
per-class candidate behavior and explicit labels, circle/rotate NMS geometry and
fail-closed inputs, canonical box/yaw/velocity conversion, and the remaining
submission/eval guards.

## Raw artifacts and SHA-256

| Artifact | SHA-256 |
|---|---|
| Slurm stdout `s05r2_centerhead_336731.out` | `fbeac7dbcc5b14cf1f377a6ca1e363c06e4932eb66a416d1179ea02349249b6e` |
| Slurm stderr `s05r2_centerhead_336731.err` | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |
| `execution_identity.json` | `ca35c57e1f0b3eb7ba4257f5be8a1df0f6b0ca736b5f1902e360ce153695e490` |
| `slurm_allocation.txt` | `de382961649eed7e5a31c213d924c50fed88160e36171cba0e4f9f9114a80d3f` |
| `pytest.log` | `3e461e6e83df9dedbdd68b2e0059e4afc2348bc54d56efebf55ef57a348a20fc` |
| `pytest.junit.xml` | `0f79ed5509881bcc84a48f8dd546ebc69de0fd8ac4cdbfe074a8cd5ee806288e` |
| `runtime_source_files.txt` | `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857` |
| `runtime_source_sha256s.txt` | `2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766` |
| `snapshot_identity.txt` | `0b405e5ef65c5212b132fd6ef484e4240fab9bb770d1df5336250940373b1f63` |
| `approved_launcher.sh` | `7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10` |
| `approved_run_request.md` | `bcd8f426e5b95438f91973e9a3d9712193cf96a23f9254732114111fb68019c1` |
| `sha256sums.txt` | `4016aa0eb83127c6713e21862790ab94333c683e253b3f063d8870fa303f8447` |

The launcher ran `sha256sum -c` over all nine listed execution artifacts; every
entry returned `OK` despite the deliberate nonzero pytest exit. This preserves a
complete, internally consistent negative result.

## Required worker action

Return S05 for a test-only remediation of
`test_submission_duplicate_geometry_orders_velocity_and_attribute_by_content`.
The fixture must compare a representation-neutral value, for example by converting
`record["velocity"]` to a tuple before comparison or by expecting tuples. It must
retain both semantic assertions:

1. `forward == reverse` for the complete result dictionary;
2. exact parked-before-moving velocity/attribute content order.

Production `box_to_global.py`, candidate/decode/NMS semantics, O-018, and the
metric-relevant sort key must not change to satisfy this fixture. A new durable
worker SHA, independent re-review, and separately approved exact runtime request
are required before PASS; Job 336731 is not retry authorization.

## Interpretation boundary

Allowed: Job 336731 executed the exact approved synthetic suite and showed that 43
cases pass; the semantic forward/reverse equality and desired velocity/attribute
order passed before the fixture's tuple/list assertion failed.

Forbidden: reporting S05 reviewed PASS; relabeling 43/44 as zero-failure PASS;
claiming the full test suite passed; inferring production/full-stack integration,
performance, model quality, mAP/NDS, FL/security, or scientific readiness; or using
this result as authorization for any follow-on job.
