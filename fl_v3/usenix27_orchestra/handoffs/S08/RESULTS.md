# S08 precision qualification — execution results

## S08-SMOKE-3 terminal result

```text
REQUEST_ID: S08-SMOKE-3
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14
SUBMISSIONS: 1
JOB_ID: 428112
STATE: COMPLETED
EXIT_CODE: 0:0
RESTARTS: 0
SUBMIT: 2026-07-14T17:09:14+02:00
START: 2026-07-14T17:09:15+02:00
END: 2026-07-14T17:12:43+02:00
ELAPSED: 00:03:28
NODE: n576
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal focused implementation-smoke **PASS**. The exact approved
read-only snapshot, runner, outer submission script, source-state identities,
resource tuple, selectors, output root, and stop conditions were used once. No
source/environment mutation, alternate node, retry, Q1 cell, or additional GPU
job was attempted.

### Immutable execution identity

```text
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke3_3014cab90ed8
SNAPSHOT_TREE_SHA256: 3014cab90ed88b5705367fc1dd1a21740593acc3a186c72f9073bffe15247a43
SMOKE_RUNNER_SHA256: 266b83f558b8d9c60f4086d633ac79326cd0dbf3e9c063837d653acf9d44cdf0
SUBMIT_SCRIPT_SHA256: 5fa6e31df27425fde7f04373519d39d30c01386fa3e9b487e406048f11bd6ac0
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke3_3014cab90ed8
```

Runtime attestation matched the request exactly:

```text
machine/device: aarch64 / NVIDIA GH200 120GB
Python/Torch/CUDA: 3.11.15 / 2.11.0+cu128 / 12.8
Torch build/source: a58ba749... / 70d99e99...
spconv version/build/source: 2.3.8 / 74934de8... / 263d6b47...
spconv tracked-state SHA256: 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db
cumm version/build/source: 0.7.13 / 0a7e3c1a... / 4dedaf43...
cumm tracked-state SHA256: f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662
```

### Pytest and acceptance-gate result

JUnit records `106` tests, `0` failures, `0` errors, `0` skipped, and `43.468`
seconds. Pytest reports `106 passed, 3 warnings in 43.48s`; `smoke.exit` is
exactly `0`, Slurm stdout ends with `S08_PRECISION_SMOKE_PASS`, and all four
runner checksums verify.

The warnings are unchanged dependency notices: one Python-locale deprecation in
`ccimport` and two spconv multidimensional-indexing warnings. They are not hidden,
promoted to acceptance evidence, or interpreted as a model-numerical result.

This pass includes the two previously blocked real disabled-GradScaler paths:
the output/RNG-neutral FP32 update and hostile pre-step diagnostic cleanup. It
also includes the corrected six-task rejection expectation, exact source-state
positive/negative tests, precision partition/config/checkpoint/runtime regressions,
and both tiny sparse FP32/FP16/island tests.

### Produced artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `environment.json` | 1,139 | `2e7884a35a43fa6dc5f422602e9e75166cbc67174d48836e2ebf9bd4d88fde8d` | complete and identity-matching |
| `smoke.log` | 1,577 | `5ef5db037debb49bea335d9bd9f2daea0b2f1d725aced0af5b791b66a9a36796` | complete pytest PASS log |
| `smoke.junit.xml` | 15,780 | `3b5dcdc2d7559b1af80446e946858d0133de1ad531802c105b43b6d128f76171` | complete JUnit |
| `smoke.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` | exactly `0` plus newline |
| `artifact_sha256s.txt` | 318 | `a6aea314859224f2a0c238fae693a0ae5d3eabe417d11ca5131b9835311ed7b7` | all four runner checksums reverified |
| `slurm-428112.out` | 1,602 | `d1881872129b5301cc34e5f5df4cb887b6913528452cb8c64eac41c3df171b9f` | pytest replay plus terminal PASS marker |
| `slurm-428112.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` | module notice only |

Pytest's bounded temporary fixtures remain under `pytest-tmp` in the preserved
output root; they include only the declared toy Git/config/checkpoint fixtures
and no nuScenes data.

### Disposition and interpretation boundary

- The exact S08-SMOKE-3 authorization is consumed and terminal; it grants no
  retry or additional compute.
- The focused GH200 implementation gate is satisfied, so the previously
  authorized immutable implementation/evidence commit may now be created.
- Independent review of that exact Git object remains required before Q1.
- Q1 remains unapproved and unexecuted; S09 remains blocked on reviewed S08
  evidence and owner precision-policy acceptance.

This PASS establishes only the exact tracked-source attestation, focused
config/routing/checkpoint/training-loop/window-diagnostic contracts, and tiny
sparse FP32/FP16/island runtime paths. It does **not** establish stable current
six-task optimizer windows, select the scientific precision policy, explain the
large LiDAR gradients, prove convergence/performance/capability, or support
mAP/NDS, Protocol A/B, attack, or defense claims.

## S08-SMOKE-2 terminal result

```text
REQUEST_ID: S08-SMOKE-2
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14
SUBMISSIONS: 1
JOB_ID: 427800
STATE: FAILED
EXIT_CODE: 1:0
RESTARTS: 0
SUBMIT/START: 2026-07-14T16:53:43+02:00
END: 2026-07-14T16:57:03+02:00
ELAPSED: 00:03:20
NODE: n23
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal **focused-test FAIL**, not a provenance, CUDA, spconv,
sparse-kernel, model-numerics, or Slurm-infrastructure failure. Exact runtime
attestation completed successfully before pytest:

```text
machine/device: aarch64 / NVIDIA GH200 120GB
Torch build/source: a58ba749... / 70d99e99...
spconv build/source: 74934de8... / 263d6b47...
spconv tracked-state SHA256: 499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db
cumm build/source: 0a7e3c1a... / 4dedaf43...
cumm tracked-state SHA256: f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662
```

This confirms that the O-100 exact source-state remediation solved Job `426619`'s
blanket-clean-checkout blocker without editing the external dependency trees.

### Pytest result

JUnit records `106` tests, `3` failures, `0` errors, `0` skipped, and `42.569`
seconds (`103 passed, 3 failed, 3 warnings` in the pytest summary).

| Failing test | Exact cause | Classification |
|---|---|---|
| `test_enabled_diagnostics_preserve_fp32_update_metrics_and_rng` | `PrecisionWindowDiagnostics.begin_window()` called `GradScaler.get_backoff_factor()` on a disabled CPU scaler; PyTorch 2.11 does not create `_backoff_factor` when `enabled=False` | real diagnostics compatibility defect; it failed before the observed forward/update, so output neutrality remains unproven by this test |
| `test_pre_step_diagnostic_failure_discards_window_without_parameter_update` | same disabled-scaler accessor failure prevented the intended injected pre-step failure path from being reached | real diagnostics test-path defect; hostile-cleanup behavior remains unexecuted in this test |
| `test_multitask_loss_rejects_legacy_single_head_output` | production correctly raised `ValueError('multi-task CenterHead must return 6 task dictionaries')`, but the new test expected regex `six task` | test expectation defect only; the fail-closed six-task loss behavior occurred correctly |

The first two failures expose a narrow but Q1-blocking issue: FP32 cells use a
disabled GradScaler, so diagnostics must not query enabled-only growth/backoff
attributes. FP16 uses an enabled scaler and is not implicated by this trace. The
third failure requires no production loss change.

All other selected cases passed, including exact source-state positive/negative
tests, precision partition/config/checkpoint/runtime regressions, diagnostic helper
tests, and both selected tiny sparse FP32/FP16/island tests. This is bounded test
evidence only; because the overall smoke gate failed, it is not an implementation
PASS or authorization to commit/Q1.

### Produced and missing artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `environment.json` | 1,139 | `2e7884a35a43fa6dc5f422602e9e75166cbc67174d48836e2ebf9bd4d88fde8d` | complete and identity-matching |
| `smoke.log` | 6,798 | `e108946a524f8d317a67d92d16cb625503f803116e3952a0760d6cd9e3381a0b` | complete pytest failure log |
| `smoke.junit.xml` | 20,528 | `19d5a3ff5cb733ed4c2847c9a58368aac36dc158f27bf347e2229f0d4a0495c1` | complete JUnit |
| `smoke.exit` | 2 | `4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865` | exactly `1` plus newline |
| `slurm-427800.out` | 6,798 | `e108946a524f8d317a67d92d16cb625503f803116e3952a0760d6cd9e3381a0b` | runner replay of `smoke.log` |
| `slurm-427800.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` | module notice only |
| `artifact_sha256s.txt` | — | — | not created because the runner stopped on nonzero pytest exit before its success-only checksum step |

The complete output directory is preserved at:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke2_935d0464b3bf
```

### Disposition

- The exact S08-SMOKE-2 authorization is consumed and terminal.
- No source/environment edit, resubmission, alternate node, or retry was attempted.
- At this S08-SMOKE-2 disposition, the owner-authorized immutable implementation
  commit remained gated; later S08-SMOKE-3 resolved that gate.
- Q1 remained not ready and was not submitted.
- O-102 subsequently authorized conditioning scaler-policy accessors on
  `scaler.is_enabled()`, asserting `None` policy fields for the real disabled
  scaler, correcting only the test regex, and preparing S08-SMOKE-3. That narrow
  remediation is implemented locally; O-102 grants no GPU execution authority.

This result does not qualify a precision regime, explain the large LiDAR
gradients, prove stable optimizer windows, establish convergence/performance/
capability, or support mAP/NDS, Protocol A/B, attack, or defense claims.

## S08-SMOKE-1 terminal result

```text
REQUEST_ID: S08-SMOKE-1
OWNER_APPROVAL: explicit exact-request approval on 2026-07-14
SUBMISSIONS: 1
JOB_ID: 426619
STATE: FAILED
EXIT_CODE: 1:0
RESTARTS: 0
SUBMIT: 2026-07-14T15:57:08+02:00
START: 2026-07-14T15:57:09+02:00
END: 2026-07-14T15:58:07+02:00
ELAPSED: 00:00:58
NODE: n535
ALLOCATED: 1 x NVIDIA GH200 120GB, 8 CPU, 96 GiB
AUTOMATIC_RETRY: none
```

This is a terminal **pre-pytest provenance FAIL**. The runner activated the
Arrhenius environment and entered `verify_runtime_dependency_identity()`, but
stopped before pytest collection, model construction, forward/backward, an
optimizer window, or any diagnostic observer execution:

```text
RuntimeError: spconv source checkout is modified
```

Therefore Job `426619` is not evidence for or against the S08 config resolver,
FP32 island, full sparse FP16, training loop, GradScaler behavior, sparse kernels,
model numerics, LiDAR gradients, or opt-in diagnostics.

## Immutable execution identity

```text
BASE_AUDIT_COMMIT: 733c84f8e3019fe4d683663821bd86918d3875a7
SNAPSHOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s08_smoke_f963da5a620e
SNAPSHOT_TREE_SHA256: f963da5a620e38a479bf9cee3a80af489bb9d212db79848678ff0560a9555ec2
SMOKE_RUNNER_SHA256: aab4656a339598366e2e0d34927cdf2812119459e15444b6e6a7b7e82487c8c9
SUBMIT_SCRIPT_SHA256: cf562f83422600a3cff6a4735b4c08842d3113a4957a44461d119faa668b01a4
JOB_BODY_SHA256: 7ac8d0277576a665690408c6002e7438e311e1e66b3ab8721f2be61856d8003a
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke_f963da5a620e
```

The submission used the exact approved command and resource tuple. No selector,
snapshot, source, environment, output path, or resource was changed after owner
approval.

## Produced and missing artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `environment.json` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | opened by shell redirection, then left empty by the failed identity check |
| `slurm-426619.out` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | empty |
| `slurm-426619.err` | 871 | `a41f36cb107b49d871fe6db2211e841a6ee4fa4c4d62ec544971b42789299611` | module notice plus exact traceback |
| `smoke.log` | — | — | not created; pytest never launched |
| `smoke.junit.xml` | — | — | not created |
| `smoke.exit` | — | — | not created |
| `artifact_sha256s.txt` | — | — | not created |

The output directory and all three produced files are preserved. Absence of the
later artifacts is expected from the fail-fast position and is part of the FAIL,
not an omitted result.

## Root-cause reconstruction

The editable spconv checkout is:

```text
SOURCE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/spconv
HEAD: 263d6b47425ef843c82f997b12d8b714013d216c
TRACKED_STATUS: M pyproject.toml
DIRTY_FILE_SHA256: e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9
HEAD_FILE_SHA256: 798e00722756f5e029d889d72e1072e768a1e2729ed12fccfc95757609908333
```

The sole tracked diff removes `"cumm>=0.7.11"` from
`[build-system].requires`. No `.py`, CUDA, C++, shared-object, or other executable
source file is modified. The cumm checkout remains clean at
`4dedaf43ff801e417c60c6bd7536a29d83d29ee0`.

This state predates S08. Accepted S07-B-COMPLETE Job `374142` preserved the exact
same status, diff, and dirty-file SHA in:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_complete_diag2_34cbe02b7b72/artifacts/spconv-source-state-before.txt
SHA256: a1a54ce2b72d3caee796c68a5b3662fb90edf15f61dd5c5fcb873d67ae29e7c3
```

That accepted evidence also binds the same installed spconv build SHA
`74934de877e07a8eef8edacd4e31ec0f06eff030b3bc7e06d01f41b1444687d8`
and source HEAD requested by S08-SMOKE-1. The current verifier nevertheless
requires every editable checkout to have an empty tracked status before it hashes
the installed executable artifacts. The observed failure is consequently a
provenance-policy mismatch with the known source-built environment, not evidence
of new runtime drift.

## Disposition and next gate

- The exact S08-SMOKE-1 authorization is consumed and terminal.
- No retry, alternate node, environment edit, source reset, or extra job was
  attempted or authorized.
- At this S08-SMOKE-1 disposition, the owner-authorized implementation commit
  remained gated because focused GH200 runtime validation did not execute.
- Q1 remained not ready and was not submitted.
- O-100 subsequently authorized the narrow provenance-policy remediation and
  replacement-request preparation. Exact `S08-SMOKE-2` is now frozen in
  `RUN_REQUEST.md` but remains unapproved for execution.

The implemented narrow remediation keeps exact source HEAD and installed
executable-build hashing, while permitting only the already-evidenced spconv
`pyproject.toml` build-metadata patch by an exact path/content identity. Any
executable-source change, additional tracked change, unrecognized metadata hash,
source HEAD drift, import-origin drift, or installed-build drift must still fail
closed. It did not reset or modify the external spconv checkout. Replacement
execution remains outside O-100 and needs explicit approval.

## Interpretation limits

This result establishes only that the original smoke wrapper's blanket clean-Git
precondition is incompatible with the accepted Arrhenius spconv checkout. It does
not select a precision policy, explain the large LiDAR gradients, validate stable
optimizer windows, establish convergence/performance/capability, or support any
mAP/NDS, Protocol-A/B, attack, or defense claim.
