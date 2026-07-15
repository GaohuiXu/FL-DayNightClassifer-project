# S09 results ledger — STOP-1 terminal / STOP-2 precompute

## Terminal state

```text
REQUEST_ID: S09-STOP1-DATA
OWNER_AUTHORITY: O-112
EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
FIRST_EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
REVIEWED_REMEDIATION_SHA: 5252a591983abb0013f19547e1d6ad20d3d6661f
JOB_ID: 441191
JOB_STATE/EXIT/RESTARTS: COMPLETED / 0:0 / 0
NODE/ELAPSED/LIMIT: n125 / 00:03:06 / 00:30:00
ACTUAL_GPU_HOURS: 0.051667
SUBMISSIONS: 1 / approval consumed / no retry
RESULT: TERMINAL / INDEPENDENT PASS_WITH_RESIDUAL_RISK / OWNER-ACCEPTED O-113
```

Job `441191` consumed the only O-112 submission and completed both requested
splits. It constructed no model, DataLoader sweep, profile, training step, or
metric. The observed elapsed time is not an S09 throughput result: the GH200 was
allocated only to obtain the validated aarch64 environment for metadata/CPU/I/O
cache construction.

| Field | Value |
|---|---|
| Account/partition/allocation | `naiss2025-22-1113-gpu` / `gpu` / one GH200, 8 CPU, 96 GiB |
| Start/end | `2026-07-15T06:06:33` / `2026-07-15T06:09:39` |
| Batch MaxRSS / MaxVMSize | `9287360K` / `13743424K` |
| Batch TotalCPU | `02:23.146` |
| Requeue/array/DDP/retry | disabled / none / none / none |

## Exact runtime and input identity

The in-job identity matched the frozen request:

- Git source `1f276b9d2cc54f705b0b6800a573258707711045`;
- runtime source-state SHA-256
  `c44db468cb65aaedab7152202ca49056147119b9ef970ffd191fdeeb4258bca8`;
- aarch64 node `n125`, CPython `3.11.15`, NumPy `1.26.4`,
  nuscenes-devkit `1.1.11`, pyquaternion `0.9.9`, Torch `2.11.0+cu128`, and
  Pillow `12.2.0`;
- module dataroot
  `/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip`;
- accepted manifest logical SHA-256
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`;
- accepted manifest physical SHA-256
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`;
  and
- `n_sweeps=10`, `v1.0-trainval`, splits `train val`.

The manifest remained the accepted read-only 633,106,432-byte
`s01.nuscenes-zip.v2` file with ten exact `trainval01_blobs.zip` through
`trainval10_blobs.zip` archive rows, 2,631,093 occurrences, 2,631,084 unique
members, and nine matching duplicate occurrences. The launcher did not rebuild
or modify it.

## Cache identities

| Split | Samples / boxes | Previous-sweep records | Canonical cache SHA-256 | Pickle bytes / SHA-256 | Sidecar bytes / SHA-256 |
|---|---:|---:|---|---|---|
| train | `28130 / 944881` | `246840` | `310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a` | `580252836` / `57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30` | `289` / `f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c` |
| val | `6019 / 187528` | `52812` | `bb692de4c1eb8b66e8c74f4e807eb208ad891b45ce8f233e8017dc4f3a3b6e2f` | `118018654` / `d4ed7aee9978c2294e2087c917006cbb3d69276453266d0f9c92591340084837` | `286` / `4f5390815720e14625be31b20fb1596cafe9869ad95b08dc098aea65413be432` |

Both pickles and sidecars identify format `t1.v2`, version `v1.0-trainval`,
scale `trainval-scientific`, and depth 10. For each split, predeclared counts,
loaded-record counts, and pickle metadata counts are identical. `load_cache`
iterated every record, required `_cache_n_sweeps == 10`, required a
`lidar_sweeps` field with at most nine previous sweeps, compared sidecar and
pickle metadata, and recomputed the full canonical content hash before the
identity artifact was written. The previous-sweep totals above are observed
descriptive counts, not separately predeclared acceptance thresholds.

## Artifact verification

Output root:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop1_cache_t1v2_1f276b9d2cc5
```

After the terminal job, S00 independently:

1. reran `sha256sum -c sha256sums.txt` successfully;
2. verified all 23 `runtime_source_sha256s.txt` entries against the detached
   snapshot and reproduced aggregate SHA-256 `c44db468...`;
3. compared each sidecar structurally with its corresponding embedded
   `cache_identity.json` metadata and found exact equality;
4. required exactly eight regular output files and zero symlinks;
5. verified the scheduler state, exit code, zero restarts, resource tuple and
   complete stdout/stderr; and
6. changed only file/directory permission bits to make the output read-only, then
   reran all checksum checks.

The frozen output contains eight files, 698,280,214 total bytes, and zero writable
entries. Permission freezing did not change any content digest.

| Artifact | SHA-256 |
|---|---|
| `execution_identity.json` | `89a4371211a2c1dba852d60f4296059ee423b6d6525a552adbc1033c241a3c60` |
| `runtime_source_sha256s.txt` | `c44db468cb65aaedab7152202ca49056147119b9ef970ffd191fdeeb4258bca8` |
| `cache_identity.json` | `7b906f885b0c13b879ff0bbd4e34d2bfc2a056605046a42baa813b1bad839250` |
| `sha256sums.txt` | `4f48ea4e7ebfc9427a4cf649e3b3826feb0b529f7a56af011b4e1b78a8f5f2ef` |
| Slurm stdout | `9460d0915142188e256b28c83b7df6bb7b5f5cfd5ef1bbebc82381639289f8c8` |
| Slurm stderr | `8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830` |

The stderr contains only Lmod's retained-system-module notice and the dataset
access/module messages; there is no Python exception, warning from the builder,
or missing output.

## Interpretation boundary

This terminal PASS establishes exact production cache identities for the current
metadata/geometry implementation and accepted manifest. Independent re-review
accepted the bounded gate with residual risk, and O-113 owner-accepts STOP-1.
These exact physical files plus their canonical and physical hashes may be bound
as later S09 inputs; no rebuild-by-name substitution is allowed.

It does **not** prove sensor-payload decode parity, DataLoader throughput, model
performance/readiness, convergence, mAP/NDS, scientific recipe quality, Protocol
A/B, FL, attack, or defense behavior. It does not retroactively convert historical
`t1.v1` caches into production inputs. STOP-2 implementation status is recorded
below; its GH200 smoke remains unapproved and unsubmitted.

---

## STOP-2 implementation and precompute evidence

```text
REQUEST_ID: S09-STOP2-SMOKE
OWNER_IMPLEMENTATION_AUTHORITY: O-114
PLANNING_BASELINE: 25a59a699fe88b8cec207d5281d6c3342d2d2db0
INITIAL_IMPLEMENTATION: ff0ffb694255e01a5b109d755ed88fa20b644a78
SYNC_REMEDIATION: 0a11b17
FINAL_REVIEW_CANDIDATE: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
FINAL_TREE: d0626e313aab411bc5c71733afb41eca5b102693
FULL_DIFF_SHA256: cb55d4a46c21f3d508e5d73240367d06080de7b456751d802367b19ed055e7eb
IMPLEMENTATION_REVIEW: PASS_WITH_RESIDUAL_RISK / no open P0-P2
GH200_JOB_ID: none
GH200_TEST_RESULT: not executed
REQUEST_STATE: frozen / awaiting one owner execution confirmation
```

### Delivered semantics

The reviewed implementation advances the strict resolved schema to `s09.v1` and
requires a hash-bound execution object. `train_eval` preserves the existing
checkpoint/evaluation lifecycle and refuses readiness-only fields. `readiness`
is fail-closed, single-process, accumulation-one, non-resumable, checkpoint-free,
and evaluation-free; it stops at the successful-update target, attempted-window
cap, or data exhaustion and writes a terminal artifact before returning failure
for an unmet gate.

The production loop adds only opt-in direct host/CUDA timing and bounded memory/
counter records. It uses no observer, module hook, retained activation, profiler,
or per-window synchronization. The optional production-loader profile is bounded,
content-hash gated, and observational: it cannot alter the hash-bound training
worker count. The six current templates now explicitly retain O-110, including
the SECOND FP32 island for L-S075/F-U/F-CBGS. No model, loss, target, data backend,
optimizer, scheduler, EMA, checkpoint schema, or precision-policy implementation
changed.

### Local/static validation

- compile-only validation passed for all eight affected Python files;
- all six current JSON templates parsed;
- manual positive/negative `s09.v1` execution-schema checks passed;
- O-110 sparse-precision template assertions agree with the templates;
- the implementation diff passed AST/static inspection and `git diff --check`;
  and
- the login node did not execute pytest because its x86 environment cannot import
  the accepted aarch64 Torch stack. The bounded GH200 smoke is the missing runtime
  check, not an inherited PASS.

### Independent implementation review

The first review of `ff0ffb6` found no P0/P1 and returned `REMEDIATE` for two P2
and two P3 findings: per-window raw `GradScaler.get_scale()` calls could add CUDA
synchronization; one old candidate-template test still expected sparse FP16;
`train_eval` sampled readiness-only clocks; and negative lifecycle behavior lacked
direct tests. `0a11b17` removes scaler polling and unused normal-lifecycle clocks;
`37aef4d` aligns the O-110 assertion and adds the missing rejection/fail-artifact
tests.

Re-review of the complete `25a59a6..37aef4d` diff returned
`PASS_WITH_RESIDUAL_RISK` with no open P0-P2. One non-blocking P3 remains: if every
attempted loss is nonfinite, enabled GradScaler never enters the finite-loss
optimizer path and `scaler_scale_at_start` remains JSON `null`; terminal scale,
window outcomes, counters, and failure behavior remain complete. Additional
residuals are that actual Torch/CUDA execution is pending this smoke and the toy
loader test cannot substitute for STOP-3's production persistent-worker gate.

### Frozen smoke identity and interpretation

The clean detached snapshot, exact four selectors, two read-only scripts and
fresh absent output are recorded in `RUN_REQUEST.md`. Script syntax, source/tree,
clean/detached status, absent alternates, zero writable snapshot worktree files,
and fresh output were rechecked before request freeze. No submit command was run.

This precompute evidence establishes a reviewed implementation and reproducible
runtime-test request only. It is not a Torch/CUDA test PASS and says nothing about
production ZIP/cache throughput, model stability, memory headroom, convergence,
mAP/NDS, recipe quality, Protocol A/B, FL, attack, or defense.
