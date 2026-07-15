# S09 independent review ledger

## Findings, ordered by severity

### P0

None.

### P1

None.

### P2 — the durable evidence package records an impossible request commit and a stale current HEAD

`RESULTS.md:9` records the frozen request commit as
`d4b64967b6049d8c5f8d57c91bc58c2a9db98fb3`. That object does not exist in the
repository. The actual request commit is
`d4b64964f56738ec388a39c277f01b3d45a4eeee`; it contains the complete frozen
tuple, was committed at `2026-07-15T06:06:18+02:00`, and preceded Job `441191`'s
submit time (`06:06:32`) by fourteen seconds. Its parent is the declared execution
source `1f276b9d2cc54f705b0b6800a573258707711045`, and its only diff from that source
is the exact `RUN_REQUEST.md` freeze.

`HANDOFF.md:8` separately labels the S09 base
`28f79802c0868afa6290d74ae6aeb9d23c7d088f` as both `BASE_AND_CURRENT_HEAD`, even
though the immutable evidence under review is
`b35591b1a9ac64ea50ee3ad3257304baef07f8de`. This obscures the required
distinction among S09 base, execution source, frozen request, and evidence state.

The raw execution is not invalidated: the real request commit exists before
submission, and its source/snapshot/script/data/resource/output tuple matches the
job and artifacts. The durable provenance record is nevertheless false as written,
so it must be corrected before STOP-1 can be accepted or its cache identities
bound downstream.

Required remediation, with no GPU rerun:

1. replace the nonexistent `REQUEST_COMMIT` with
   `d4b64964f56738ec388a39c277f01b3d45a4eeee`;
2. split `HANDOFF.md` identity into at least `BASE_SHA=28f7980...`,
   `EXECUTION_SOURCE_SHA=1f276b9...`, `REQUEST_COMMIT=d4b64964...`, and
   `EVIDENCE_SHA=b35591b...`; and
3. request a bounded documentation/provenance re-review of the corrected immutable
   state. Do not resubmit Job `441191`.

### P3 — active authority/status prose still says this materialization is pending

The top-level status blocks correctly report O-112 and Job `441191`, but three
active authority locations still say full trainval `t1.v2` cache materialization
is pending owner approval: `AGENTS.md:212-214`, `docs/env.md:190-193`, and
`ORCHESTRA.md:331-333`. The job is already authorized, consumed, and terminal;
only independent acceptance/binding is pending. `docs/env.md:115-120` also retains
the pre-S08 wording that S08 still must qualify precision, contrary to O-110 and
the binding root/canonical status.

S00 should reconcile these lines while sealing the remediation: describe the two
exact caches as materialized under O-112 but not production-bound until review
acceptance, and describe the accepted O-110 precision policy. This is status drift,
not evidence that an unauthorized job or precision change occurred.

## Review identity and scope

```text
REVIEWER: independent S09 STOP-1 data/provenance reviewer
BASE_SHA: 28f79802c0868afa6290d74ae6aeb9d23c7d088f
EXECUTION_SOURCE_SHA: 1f276b9d2cc54f705b0b6800a573258707711045
ACTUAL_REQUEST_COMMIT: d4b64964f56738ec388a39c277f01b3d45a4eeee
EVIDENCE_SHA: b35591b1a9ac64ea50ee3ad3257304baef07f8de
BRANCH: codex/s08-s09-cl-readiness
JOB: 441191
REVIEWER_COMPUTE: none
```

Preflight at review start matched the envelope: repository root was the persistent
S00 worktree, HEAD was exactly `b35591b1a9ac64ea50ee3ad3257304baef07f8de`,
the branch was `codex/s08-s09-cl-readiness`, and `git status --short` was empty.
No ref, branch, worktree, implementation, existing document, dataset, cache, or
Slurm state was modified by this review. This `REVIEW.md` is the sole new file.

The reviewer read the binding root/environment/canonical documents; the complete
S01 handoff/results/review/request; the pending historical S07 `t1.v2` request;
the complete S09 handoff/request/results; the exact `28f7980..b35591b` diff; and
the requested execution-source builder, launcher, environment, cache, path, ZIP,
and focused cache tests. Raw output, snapshot, submission wrapper, logs, manifest,
and scheduler records were inspected read-only. No cache pickle was deserialized
and no new full-data traversal was performed.

## Adversarial checks and evidence

### Authorization, request freeze, and submission boundary

- O-112 permits exactly one metadata-only train/val `t1.v2`, `n_sweeps=10`
  submission on one GH200, eight CPUs, 96 GiB, `00:30:00`, and at most 0.5
  GPU-hours. It forbids retry, payload scan/extraction, model work, loader/profile,
  STOP-2, merge, and push.
- Actual request commit `d4b64964...` freezes the complete tuple before submission.
  Its request state is explicitly unsubmitted/one submission available; the
  evidence commit later marks it consumed.
- `sacct --name flv3_s09_stop1_cache` reports exactly one top-level job in the
  review interval: Job `441191`, `COMPLETED`, exit `0:0`, restarts `0`. No active
  same-name job was present and there is no replacement/retry evidence.
- The job used `00:03:06`, or `0.051667` GPU-hours. Unused quota is correctly not
  treated as retry or STOP-2 authority.

### Immutable source, snapshot, and command

- Snapshot
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop1_cache_1f276b9d2cc5`
  is detached and clean at `1f276b9...`, tree
  `c0d2ecac553e3f2ec81b52b85a633c20c64e5111`.
- It contains 587 tracked files / 4,618,253 worktree bytes, no writable worktree
  entry, one self-contained Git pack, and no alternates file.
- All 23 entries in `runtime_source_sha256s.txt` independently match the snapshot;
  hashing the list reproduces
  `c44db468cb65aaedab7152202ca49056147119b9ef970ffd191fdeeb4258bca8`.
- Launcher, builder, and environment hashes reproduce exactly as
  `212e176d...`, `6b9ebf18...`, and `f57befbb...`.
- The external submit wrapper is read-only, hashes to
  `aeabbab55b625594a6da9eb820f8b5dae1cdb7e70a6d1d447055a162e093856d`,
  and binds the declared snapshot, source hash, accepted manifest, fresh output,
  exact job name, resources, no-requeue, and unchanged S07-A cache launcher.
- The cache builder, cache format/depth logic, path resolution, transforms, and
  class mapping are byte-identical to the reviewed S07-A source
  `44cefd06bc815e893919d95c754896711dba3402`. Later ZIP-backend differences are
  loader lifecycle/modality accounting changes; `manifest_summary` and the
  metadata-only builder seam are unchanged.

### Dataset and accepted ZIP-manifest identity

- The wrapper invokes the exact module `nuScenes-data/1.0-map-1.3-zip`; the in-job
  identity records its resulting dataroot as
  `/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip` and records CPython
  `3.11.15`, NumPy `1.26.4`, nuscenes-devkit `1.1.11`, pyquaternion `0.9.9`,
  Torch `2.11.0+cu128`, and Pillow `12.2.0` on aarch64 node `n125`.
- The accepted manifest remains mode `0444`, 633,106,432 bytes, and independently
  hashes to
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.
- Read-only SQLite metadata reports format `s01.nuscenes-zip.v2`, logical hash
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`,
  exact `trainval01_blobs.zip` through `trainval10_blobs.zip`, 2,631,093
  occurrences, 2,631,084 unique members, and nine duplicate occurrences. These
  values agree with accepted S01 evidence and the in-job execution identity.
- The launcher checks manifest logical identity, physical SHA-256, and exact
  archive names before creating the output. It does not rebuild or relabel the
  historical manifest.

### `t1.v2`, depth, content, and physical cache binding

The exact source implements the intended layered identity rather than relying on
one label:

1. filename contains `t1.v2_nsweeps10`;
2. pickle metadata and equal JSON sidecar contain `format_version=t1.v2` and
   `n_sweeps=10`;
3. every record carries `_cache_n_sweeps=10`; `load_cache` iterates every record,
   requires a multi-sweep field, and rejects more than nine previous sweeps;
4. the canonical hash includes the format, requested depth, every sorted sample,
   raw camera/LiDAR/calibration/pose/box/sweep input, and sweep count; and
5. exact pickle and sidecar SHA-256s additionally bind derived matrices and all
   serialized fields that are intentionally outside the host-portable raw-input
   canonical hash.

The job loaded both completed caches through this validation path and recomputed
their canonical hashes before emitting `cache_identity.json`. Without
deserializing the caches again, the reviewer independently verified all seven
files named by `sha256sums.txt`, exact sidecar equality with the identity JSON,
and the following frozen identities:

| Split | Samples / boxes | Previous sweeps | Canonical SHA-256 | Pickle SHA-256 | Sidecar SHA-256 |
|---|---:|---:|---|---|---|
| train | 28,130 / 944,881 | 246,840 | `310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a` | `57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30` | `f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c` |
| val | 6,019 / 187,528 | 52,812 | `bb692de4c1eb8b66e8c74f4e807eb208ad891b45ce8f233e8017dc4f3a3b6e2f` | `d4ed7aee9978c2294e2087c917006cbb3d69276453266d0f9c92591340084837` | `4f5390815720e14625be31b20fb1596cafe9869ad95b08dc098aea65413be432` |

Predeclared, actual, and embedded-metadata sample/box counts are identical for
both splits. The builder uses the official nuscenes-devkit
`create_splits_scenes()` scene lists and token-sorted samples. Accepted S01
evidence on this same module/version established train/val token disjointness,
union coverage of all 34,149 samples, and matching metadata traversal. This
bounded review did not repeat that full token scan or deserialize the new caches;
the new canonical and physical hashes freeze the exact generated records.

### Metadata-only and no-payload boundary

- `build_nuscenes_cache.py` obtains paths from `sample_data.filename` and walks
  JSON/devkit metadata. `info_cache.py` does not instantiate `NuScenesBlobStore`
  or call a sensor-byte reader.
- `verify_dataset` reads `sample.json` and checks the presence/size of the exact
  archives; `manifest_summary` reads only SQLite metadata. The separate physical
  manifest hash reads the accepted manifest file, not an image/LiDAR payload.
- Stdout contains only metadata/cache counts and checksum verification. Stderr
  contains the normal module-purge and gated-dataset notices. There is no decode,
  CRC payload sentinel, extraction, model, DataLoader, profile, training, or
  evaluation output.
- The job writes only to the fresh personal `/nobackup` output. No shared dataset
  write or archive extraction is present in source, logs, or artifacts.

### Scheduler, requested cells, and post-job freeze

- Scheduler data matches one node / one GH200 / eight CPUs / 96 GiB, account
  `naiss2025-22-1113-gpu`, partition `gpu`, node `n125`, `00:30:00` limit,
  `00:03:06` elapsed, `COMPLETED 0:0`, and zero restarts.
- Batch observations match the record: `MaxRSS=9287360K`,
  `MaxVMSize=13743424K`, `TotalCPU=02:23.146`. No resource or cell expansion
  occurred.
- Both requested cells, train and val, completed. There is no failed, skipped, or
  omitted requested split and no retry hidden by the PASS label.
- The frozen output has exactly eight regular files, 698,280,214 total bytes,
  zero symlinks, and zero writable entries. Independent `sha256sum -c` passed
  after the permission-only freeze. Log hashes reproduce as `9460d091...` and
  `8db5d05b...`.

## Gate verdicts

| Gate | Verdict | Evidence / limit |
|---|---|---|
| O-112 exact scope and one-submission boundary | PASS | Actual frozen request predates one terminal job; no retry/follow-on. |
| Source/snapshot/submit-script identity | PASS | Clean detached self-contained snapshot; all source and wrapper hashes reproduce. |
| Dataset/module and accepted manifest identity | PASS | Exact module-derived root plus logical/physical manifest identity and archive set. |
| Metadata-only execution | PASS | Source and logs show no sensor payload, extraction, loader, model, profile, or metric path. |
| `t1.v2` / `n_sweeps=10` fail-closed binding | PASS | Filename, metadata, every-record validation, canonical hash, and physical hashes. |
| Train/val official counts | PASS | Exact predeclared/actual/metadata sample and box counts for both requested splits. |
| Train/val leakage protection | PASS WITH RESIDUAL RISK | Official scene split logic and accepted S01 same-module disjointness; no new full token scan by this bounded review. |
| Artifact completeness and post-job freeze | PASS | Eight expected files, checksum verification, no symlink/writable entry. |
| Resource/elapsed/retry accounting | PASS | `sacct` agrees; 0.051667 GPU-hours; one submission. |
| Allowed/forbidden interpretation | PASS | Results explicitly exclude payload parity, throughput, model/science, FL, attack, and defense. |
| STOP-2 authorization | PASS | Explicitly not approved; no STOP-2 source or job exists in this evidence. |
| Durable Git provenance record | **REMEDIATE** | Nonexistent request SHA and stale current-HEAD field must be corrected. |

## Allowed interpretation

After the P2 documentation correction and accepted re-review, the two exact
read-only train/val cache files may be proposed as production inputs only when the
resolved run binds their canonical, pickle, and sidecar hashes together with the
accepted manifest logical and physical hashes and `n_sweeps=10`.

The evidence establishes cache materialization and provenance. It does not
establish sensor decode parity, loader or model throughput, GPU utilization,
training stability, convergence, mAP/NDS, recipe quality, Protocol A/B, FL,
attack, defense, generalization, or publication claims. Historical `t1.v1`
caches remain forbidden production inputs.

## Residual risks

1. The new cache job intentionally did not repeat S01's train/val token-overlap
   and full manifest-reference audit. That accepted same-module evidence plus the
   official split implementation supports this STOP-1 scope, while the new
   canonical/physical identities freeze what was produced. Future Protocol-B
   ownership still requires its separate scene/log/raw-sensor leakage audit.
2. The module is identified by the exact loaded module path/version contract and
   the resulting full cache identities, not by a separate cryptographic manifest
   of every source JSON table. Exact cache and manifest hashes therefore must be
   used downstream rather than rebuilding or trusting the module name alone.
3. Pickle is a trusted-local artifact format, not a safe parser for untrusted
   inputs. The files are personal-project, checksum-bound, and read-only; consumers
   must verify the physical SHA-256 before loading.
4. STOP-1 says nothing about ZIP payload decode, shared-filesystem contention,
   DataLoader behavior, or model-integrated performance; those remain later S09
   gates with separate owner authority.

## Final verdict

**REMEDIATE.** There are no P0 or P1 findings, and the raw Job `441191` cache,
source, manifest, artifact, and scheduler evidence passes the approved STOP-1
technical gates. Acceptance is blocked only by the P2 immutable-provenance record
errors. Correct the exact Git fields and the P3 active-status drift in a new
linear documentation/evidence commit, then request a bounded re-review. No cache
rebuild, Slurm retry, payload scan, STOP-2 implementation, merge, or push is
authorized or warranted by this verdict.

---

## Bounded re-review amendment — remediation SHA `5252a591983abb0013f19547e1d6ad20d3d6661f`

### Re-review identity and scope

- Reviewed remediation commit:
  `5252a591983abb0013f19547e1d6ad20d3d6661f`, tree
  `5f396625bace658a5538850077619a981f721248`.
- Exact remediation diff:
  `b35591b1a9ac64ea50ee3ad3257304baef07f8de..5252a591983abb0013f19547e1d6ad20d3d6661f`.
- Preflight matched the requested state: branch
  `codex/s08-s09-cl-readiness`, exact remediation HEAD, and a clean worktree
  before this review-only amendment.
- The remediation changes only nine documentation/evidence files. There are no
  changes under `fl_v3/src/`, `fl_v3/scripts/`, `fl_v3/configs/`, or
  `fl_v3/tests/`; `git diff --check` passes.
- This re-review did not run compute, rebuild either cache, scan sensor payloads,
  repeat the full token/reference audit, or authorize any later S09 stop.
- This amendment supersedes the first-review **REMEDIATE** gate verdict while
  preserving its findings, evidence analysis, and interpretation limits as
  immutable review history.

### Re-review findings, severity first

- **P0: none.**
- **P1: none.**
- **P2: none.** The two prior immutable-provenance defects are closed.
- **P3: none.** The two prior active-document status drifts are closed.

### Prior-finding closure

| Prior finding | Re-review disposition |
|---|---|
| P2 — nonexistent request SHA in `RESULTS.md` | **CLOSED.** The active S09 package now records the actual request commit `d4b64964f56738ec388a39c277f01b3d45a4eeee`; the object exists and has execution-source parent `1f276b9d2cc54f705b0b6800a573258707711045`. |
| P2 — `HANDOFF.md` conflated the accepted base and current evidence state | **CLOSED.** `HANDOFF.md`, `RUN_REQUEST.md`, and `RESULTS.md` now distinguish `BASE_SHA=28f79802c0868afa6290d74ae6aeb9d23c7d088f`, `EXECUTION_SOURCE_SHA=1f276b9d2cc54f705b0b6800a573258707711045`, `REQUEST_COMMIT=d4b64964f56738ec388a39c277f01b3d45a4eeee`, and `FIRST_EVIDENCE_SHA=b35591b1a9ac64ea50ee3ad3257304baef07f8de`. |
| P3 — active docs still described full trainval `t1.v2` materialization as pending | **CLOSED.** `AGENTS.md`, `docs/env.md`, and the active Orchestra records now state that O-112 Job `441191` materialized the exact train/val caches and that only documentation re-review/owner acceptance remained at this SHA. |
| P3 — `docs/env.md` still described S08 precision qualification as open | **CLOSED.** It now records the accepted O-110 FP16 policy, including the sparse-LiDAR FP32 island and unresolved true SECOND gradient-scale risk. |

### Adversarial re-checks

- The four provenance layers are mutually consistent across the active S09
  handoff package and correspond to real Git objects. The earlier incorrect SHA
  and `BASE_AND_CURRENT_HEAD` wording remain only inside the preserved first
  review as descriptions of the defects that were found.
- The remediation does not alter Job `441191`, its raw output, the approved O-112
  tuple, cache contents, manifest contents, source/runtime snapshot, resource
  accounting, requested cells, or any STOP-1 gate threshold.
- Re-checking the small frozen identity artifacts reproduced these SHA-256 values:
  `execution_identity.json` =
  `89a4371211a2c1dba852d60f4296059ee423b6d6525a552adbc1033c241a3c60`,
  `runtime_source_sha256s.txt` =
  `c44db468cb65aaedab7152202ca49056147119b9ef970ffd191fdeeb4258bca8`,
  `cache_identity.json` =
  `7b906f885b0c13b879ff0bbd4e34d2bfc2a056605046a42baa813b1bad839250`,
  and `sha256sums.txt` =
  `4f48ea4e7ebfc9427a4cf649e3b3826feb0b529f7a56af011b4e1b78a8f5f2ef`.
  The frozen output still contains eight regular files, zero writable entries,
  and zero symlinks.
- Canonical status text consistently characterizes the remediation as
  documentation/provenance-only. It does not convert inherited evidence into a
  new job, hide an omitted cell, weaken a gate, or reinterpret the cache as
  payload/performance/scientific evidence.
- STOP-2 remains explicitly unapproved. This review records no STOP-2 source,
  request, job, or permission and grants none.

### Accepted gate and remaining boundary

The STOP-1 data/provenance gate is accepted at the reviewed remediation SHA. The
exact train and val `t1.v2`, `n_sweeps=10` cache artifacts may be proposed for
downstream production binding only with their canonical, pickle, and sidecar
hashes plus the accepted manifest logical and physical hashes. The owner may now
inspect/accept STOP-1 and separately discuss a future exact STOP-2 envelope.

The four residual risks from the first review remain unchanged and non-blocking
for this bounded gate: no new full train/val token/reference overlap audit was
performed; the loaded module tables do not have a separate per-table
cryptographic source manifest; pickle loading remains trusted-local and requires
physical-hash verification; and STOP-1 provides no sensor-payload, loader,
contention, model-performance, stability, metric, or scientific evidence.

## Re-review final verdict

**PASS WITH RESIDUAL RISK.** The prior P2 and P3 findings are closed, no new
P0-P3 finding is present, and no cache rebuild or Slurm rerun is warranted. This
verdict accepts only the exact STOP-1 cache-materialization and provenance gate
within the limits above. It does not authorize STOP-2, additional compute, merge,
push, or any scientific interpretation.

---

## STOP-2 implementation review

### Review identity and scope

```text
PLANNING_BASELINE: 25a59a699fe88b8cec207d5281d6c3342d2d2db0
INITIAL_IMPLEMENTATION: ff0ffb694255e01a5b109d755ed88fa20b644a78
FINAL_REVIEW_CANDIDATE: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
FINAL_TREE: d0626e313aab411bc5c71733afb41eca5b102693
REVIEW_DIFF: 25a59a699fe88b8cec207d5281d6c3342d2d2db0..37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
REVIEWER_COMPUTE: none
FINAL_VERDICT: PASS_WITH_RESIDUAL_RISK
```

The independent reviewer inspected the exact diff, current config/task/training
interfaces, focused tests, S09 handoff boundary, and O-110 precision policy. The
reviewer did not edit source or execute Slurm. The review covered output
neutrality, hidden synchronization, readiness lifecycle/counter semantics,
attempted versus successful windows, schema/config fail-closed behavior, normal
train/eval preservation, loader-profile authority, and prohibited model/data/
recipe expansion.

### Initial findings on `ff0ffb6`

- **P0: none.**
- **P1: none.**
- **P2 — hidden scaler synchronization.** New per-window raw
  `GradScaler.get_scale()` reads could force device-to-host synchronization and
  contaminate the very latency measurements being qualified.
- **P2 — stale precision regression.** The candidate-template integration test
  still required sparse FP16 for SECOND templates, contradicting accepted O-110
  and the newly explicit FP32 island.
- **P3 — normal lifecycle overhead.** `train_eval` sampled readiness clocks even
  though no readiness artifact consumed them.
- **P3 — missing negative lifecycle coverage.** Resume/existing-output rejection
  and unmet-target terminal-artifact-before-error behavior lacked direct tests.

### Remediation and re-review

Commit `0a11b17` removes per-window scaler polling and readiness-only normal-mode
clock sampling without changing optimizer/scaler semantics. Commit `37aef4d`
aligns the stale template assertion with O-110 and adds the two direct negative
lifecycle tests. The reviewer re-read the complete diff and found all four
findings closed.

Final severity state:

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3: one non-blocking evidence-field edge case.** If every attempted loss is
  nonfinite, the enabled scaler never reaches the finite-loss optimizer branch,
  so `scaler_scale_at_start` is `null`; terminal scaler state, outcomes, counters,
  and normal failure remain present.

Actual aarch64 Torch/CUDA execution remains outside this static review and must be
supplied by the exact STOP-2 smoke. The unit loader fixture also does not exercise
production persistent workers or shared ZIP contention; those are explicitly
STOP-3 concerns.

### STOP-2 implementation verdict

**PASS WITH RESIDUAL RISK.** There is no open P0-P2. The implementation stays
inside O-114, preserves normal training semantics and O-110, and is suitable for
the exact bounded GH200 regression smoke. This verdict does not authorize that
job, accept a runtime result, open STOP-3, or establish performance/model/science.

---

## STOP-2 frozen-request review — initial verdict

### Review identity and scope

```text
REQUEST_SEAL: 4408dfe6541ecdce40ea53553bfd66666c65ea7a
REQUEST_SEAL_TREE: ae43d1d50ed3a6f2078db5ba4df0df5a54f18047
EXECUTION_SOURCE: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
EXECUTION_TREE: d0626e313aab411bc5c71733afb41eca5b102693
REVIEWER_COMPUTE: none
VERDICT: REMEDIATE
```

The independent reviewer inspected the exact documentation seal, implementation
diff, source snapshot, job/submit scripts, selector expansion, fresh output and
read-only scheduler state. No file, Git ref, Slurm job, output, or dataset state
was modified by the review.

### Findings, severity first

- **P0: none.**
- **P1: none.**
- **P2 — request overstates test evidence.** The request said no model/training
  executes even though deterministic toy `Linear` optimizer windows run, and said
  the CUDA pair directly compares per-window outputs/losses/gradients. The test
  actually compares final model/optimizer/scheduler/EMA/scaler/`TrainingState`,
  non-timing aggregate metrics (including aggregate loss), and host/device RNG;
  it does not retain per-window outputs or gradient tensors. The required fix is
  documentation-only, not a larger test harness.
- **P2 — O-107 derivation contract is incomplete.** The initial request freezes
  `_a1` but does not prebind the derived command family and collision-free output
  naming rule required by the root contract for submission indices two and three.
- **P3 — snapshot has a stale auxiliary split commit-graph.** Source HEAD/tree,
  detached/clean state, 590 tracked files / 4,722,741 bytes, executable bits, zero
  writable worktree files, no alternates and all wrapper preflights pass. However,
  default full `git fsck` reads a stale commit-graph reference to missing
  non-reachable object `ac578310...`. Removing/rebuilding that auxiliary index
  makes the otherwise self-contained object database pass fsck.

All other gates matched: script hashes/modes/syntax; exact static total of 44
tests (`9 + 21 + 9 + 5`); no nuScenes/cache/payload, production detector, metric
or profile; exact resources/timeouts; absent output/logs; zero same-name `squeue`
or `sacct` records; and explicit no-compute/no-STOP-3/no-merge/no-push status.

### Initial frozen-request verdict

**REMEDIATE.** Correct the evidence claims, prebind O-107's derived command/output
family, remove the stale commit-graph and require full fsck, then seal one linear
documentation/provenance remediation and request closure re-review. No
implementation edit, test expansion, Slurm submission, merge, or push is warranted.

### Closure re-review — remediation `cad72621e0e3ba409ae19bb0b62829118134b2d0`

```text
REMEDIATION_TREE: 3f50744bd9f8351bcebaf6e199328646fbf81e45
REMEDIATION_PARENT: 4408dfe6541ecdce40ea53553bfd66666c65ea7a
P0/P1/P2/P3: none open
VERDICT: PASS_WITH_RESIDUAL_RISK
REVIEWER_COMPUTE: none
```

The independent closure review verified that the remediation changes only three
S09 evidence documents and closes every initial finding:

- the request now states the exact deterministic toy `Linear/MSE` scope and the
  final-state/aggregate-metrics/RNG comparisons, while explicitly excluding
  direct per-window output/gradient capture;
- O-107 now binds indices `1..3`, the initial exact command, derived request/
  command/output naming, no path reuse, and unchanged selectors/order, seeds,
  toy scope, environment, timeout, resources and stop conditions; and
- the snapshot has no alternates or commit-graph, remains exact detached/clean at
  source/tree with 590 files / 4,722,741 bytes, correct executable modes and zero
  writable worktree files, and default full fsck exits zero with only eight
  dangling-tree notices.

The reviewer also reproduced both script hashes and syntax, implementation diff
hash, the static `44 = 9 + 21 + 9 + 5` selector count, absent output/logs, and zero
same-name `squeue`/`sacct` records. No state or Slurm action was modified.

**Closure verdict: PASS WITH RESIDUAL RISK / no open P0-P2.** Residuals are the
pending real aarch64 Torch/CUDA pytest; the documented limitation that toy final-
state/aggregate comparison is not per-window output/gradient capture; production
persistent-worker/ZIP behavior deferred to STOP-3; and the non-blocking all-
nonfinite `scaler_scale_at_start=null` edge case. This review does not authorize
the smoke, STOP-3, merge, push, or scientific interpretation.

---

## STOP-2 terminal evidence review — initial verdict

### Review identity

```text
EVIDENCE_SHA: a67cdda56c624d302742f5c57c69bb9ef0a98e0c
EVIDENCE_TREE: 5f8ff176094bb87b02043b375be0fca9c4b96ead
APPROVAL_PARENT: 254872197c0a4b2b3d02ebd8b8e320a49b98a218
EXECUTION_SOURCE: 37aef4d6b3f4679d6702d0acef2bb5bd1b57a952
JOB: 441293
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / docs-only P3 reconciliation required
```

### Findings, severity first

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3 — active status prose retained pre-O-115 present tense.** The top status,
  O-115 ledger, terminal `RUN_REQUEST.md`, and `RESULTS.md` are correct, but
  `KICKOFFS.md` still described no S09 execution/the planning-stage resource
  proposal as unapproved; `SESSIONS.md` still called O-112 the only S09 job
  authority; and `HANDOFF.md` said the smoke/confirmation/Torch-CUDA test remained
  future work. This under-authorizing drift does not make the job unauthorized or
  weaken any gate. Reconcile only those current-status sentences; preserve
  historical O-114 and prior review records.

### Independent technical checks

- approval commit `2548721` predates scheduler submission by five seconds and
  changes documentation only; source, scripts and tuple remain fixed;
- exactly one same-name Job `441293` exists: `COMPLETED 0:0`, zero restarts, one
  node/GH200, four CPUs, 32 GiB, ten-minute limit, 64-second elapsed
  (`0.017778` GPU-hours), with no `_a2`/`_a3` replacement;
- the snapshot is detached/clean at exact source/tree, has 590 tracked files,
  zero ignored/untracked or writable regular files, no executable-bit mismatch,
  and full fsck exits zero with the eight declared dangling trees;
- all request/job/output/log hashes reproduce and both scripts pass `bash -n`;
- JUnit is well formed and contains exactly `44 = 9 + 21 + 9 + 5` passing cases,
  zero failure/error/skip, including the non-skipped CUDA test (`1.996s`);
- the exact aarch64/GH200/Torch/CUDA/spconv/cumm environment matches the record;
  and
- the preserved output has 18 regular files / 24,797 bytes, 30 directories and
  14 non-dangling/non-escaping pytest scratch symlinks, with every regular file
  and directory non-writable. The seven top-level acceptance files and scratch
  boundary are accurately distinguished.

The evidence claims remain conservative: this is a toy readiness/config/lifecycle
GH200 gate, not production ZIP/persistent-worker/throughput, detector stability,
memory-headroom, convergence, recipe or scientific evidence.

### Initial terminal-evidence verdict

**PASS WITH RESIDUAL RISK / no open P0-P2.** Job `441293` is technically
acceptable and no rerun, source/script/output edit, or O-107 replacement is
warranted. Before owner STOP-2 acceptance, seal the minimal active-document P3
reconciliation and request exact closure re-review.

### Terminal-evidence closure re-review — remediation `79f87dc9accca700b5a46803d45c549b0305c6d1`

```text
REMEDIATION_TREE: 32955609885e315430c77a237b3ca123e5144f18
REMEDIATION_PARENT: a67cdda56c624d302742f5c57c69bb9ef0a98e0c
P0/P1/P2/P3: none open
VERDICT: PASS_WITH_RESIDUAL_RISK
REVIEWER_COMPUTE: none
```

The closure reviewer verified a clean branch and a four-document-only diff. The
original P3 is closed: `KICKOFFS.md` now records the actual O-115 resources/job
while retaining STOP-3 as unapproved; `SESSIONS.md` distinguishes consumed
STOP-1/2 authority from unapproved later work; and `HANDOFF.md` separates the
historical O-114 planning state from O-115 execution and terminal evidence.

There is no change from the parent evidence to source, scripts, configs, tests,
`RUN_REQUEST.md`, `RESULTS.md`, raw output, hashes, results or gates. All artifact
and log hashes and `sha256sum -c` were rechecked; scheduler state still contains
only terminal Job `441293`, no active item and no `_a2`/`_a3` replacement.

**Closure verdict: PASS WITH RESIDUAL RISK / no open P0-P3.** STOP-2 is
close-ready for owner acceptance. Residuals remain exactly bounded: the toy test
compares final state/aggregate metrics/RNG rather than per-window output/gradient
tensors; production detector/ZIP/persistent-worker/throughput/100-step stability/
memory/convergence/science remain STOP-3 or later; and the all-loss-nonfinite
`scaler_scale_at_start=null` evidence edge remains non-blocking. This verdict does
not authorize STOP-3/4, merge, or push.

---

## STOP-3 terminal failure/remediation independent review

### Findings, severity first

- **P0: none.**
- **P1: none.**
- **P2: none.** The immutable request, sole submission, failure boundary,
  preserved artifacts, dependency side effect, and unexecuted runner correction
  are represented conservatively enough to retain Job `441511` as negative
  engineering evidence.
- **P3 — active status/interface wording needs documentation-only reconciliation
  when this review is sealed.** Root `AGENTS.md:223-225` still says STOP-2 owner
  acceptance is pending and no later S09 job is authorized; `RUN_REQUEST.md:203`
  likewise labels STOP-2 owner acceptance pending. Both conflict with O-116 and
  the now-consumed O-117 Job `441511`. Separately,
  `scripts/arrhenius_env.sh:9` says the `run` module path "falls back to build
  modules if needed", but its implementation at lines 64-68 loads only
  Miniforge; this conflicts with `docs/env.md:35-43` and obscures the exact
  distinction that caused this failure. Correct those status lines and the shell
  comment without changing module-loading behavior, and add the exact STOP-3
  failure-evidence/review identity to the HANDOFF state block. These drifts do not
  make Job `441511` unauthorized, hide a requested result, or warrant compute.

### Review identity and scope

```text
REVIEWER: independent S09 STOP-3 failure/remediation reviewer
FAILED_EXECUTION_SOURCE: 4d6bd829450021aa0813bcece066fb1fac85f478
FAILED_EXECUTION_TREE: affb4854689a0bf65d829a273d769c87c000174c
REQUEST_FREEZE: 30e6c9f7849dd1bfe7630f698913c2231131b62c
EVIDENCE_CANDIDATE: 4fc78d508d4ac9ad7c46b9d3ad81c87646f8f0d3
EVIDENCE_TREE: 56c08110cc4308e424101ae39e7edb79c2769cef
EVIDENCE_PARENT: 30e6c9f7849dd1bfe7630f698913c2231131b62c
BRANCH: codex/s08-s09-cl-readiness
JOB: 441511
REVIEWER_COMPUTE: none
```

Preflight matched the requested immutable state: HEAD was exact candidate
`4fc78d5`, its parent was the request freeze, the branch was
`codex/s08-s09-cl-readiness`, and the worktree was clean before this review-only
append. The reviewer read the binding root/environment/canonical documents, full
S09 package, exact `30e6c9f..4fc78d5` diff, failed-source config/runner/trainer/
runtime-identity code, snapshot and submit script, raw output/logs, accepted S08
Q1 dependency evidence, current editable-dependency state, and read-only
`sacct`/`scontrol` records. No source, external dependency, output, Git ref, or
scheduler state was modified.

From the request freeze, candidate `4fc78d5` changes eight files. Seven changes
record the failure/status. The only executable-source change replaces
`arrhenius_load_modules run` with `arrhenius_load_modules build` in the STOP-3
runner; no config, trainer, model, loss, data, precision, optimizer, scheduler,
EMA, or metric source changed.

### Adversarial checks

#### Exact authority and unique execution

- O-117 froze one F-U cell, seed `0`, global FP16 plus the SECOND FP32 island,
  AdamW `1e-4/0.01`, constant scheduler, training workers `8`, the bounded
  `0/2/4/8` loader sweep, 100 successful updates within 120 attempts, one GH200,
  16 CPUs, 96 GiB, `01:00:00`, and no retry/replacement/STOP-4.
- Source `4d6bd82` and request `30e6c9f` have the declared source/tree/parent
  relationship. The request was committed at `09:43:16+02:00`; Slurm submission
  was `09:43:23`, so the complete tuple existed before execution.
- The detached clean self-contained snapshot reproduces source/tree, 592 tracked
  files / 4,776,222 bytes, no alternates, full `git fsck` exit zero, and no
  writable worktree entry. Runner/config/trainer hashes reproduce as
  `18cca984...`, `e8a17b39...`, and `9284d395...`.
- The read-only submit script hashes to `82790e4c...` and requests the exact
  approved resources, paths and no-requeue/no-array/no-DDP boundary.
- Scheduler history contains exactly one same-name top-level job: `441511`,
  `FAILED 1:0`, zero restarts, node `n127`, `00:02:29` elapsed, and the exact
  resource tuple. Only the `_a1` output/request/log family exists. No replacement
  or active same-name job exists. Consumption was `0.041389` GPU-hours; unused
  ceiling is not retry authority.

#### Failure point and direct diagnosis

- `centralized_train.py:384` calls
  `verify_runtime_dependency_identity`; physical cache/manifest verification is
  the next call at line 386. The traceback ends inside the former, through
  `runtime.py:354` and `_runtime_build_identity`'s import at line 243. There is no
  returned runtime-dependency manifest, loader profile, model, loss, backward,
  optimizer attempt, or `readiness.json`; the `readiness/` directory is empty.
- Failed source `4d6bd82` selected `arrhenius_load_modules run`, while binding
  `env.md` requires `arrhenius_load_modules build` even for runtime jobs because
  editable cumm/spconv imports may invoke ccimport/ninja and need CUDA toolkit
  headers.
- Raw stderr begins with `which: no nvcc`. Raw stdout shows cumm completing all
  `47/47` `core_cc` targets, then spconv starting a `690`-target build. Its first
  failed target reports
  `SimpleExternalSpconvMatmul.h:2:10: fatal error: cublasLt.h: No such file or directory`;
  ninja stops and Python propagates `CalledProcessError`. This directly supports
  a module-bootstrap defect, not a data/loader/model/precision/gradient/optimizer/
  capacity result.

#### G100 gates remain unmeasured

| O-117 gate | Review disposition |
|---|---|
| Four loader cells / identical digests | NOT STARTED |
| Worker-8 warm throughput ratio | NOT MEASURED |
| 100 successful updates within 120 attempts | NOT STARTED; zero attempts |
| Nonfinite/discarded windows and counters | NOT MEASURED |
| Integrated p95/p50 and data-wait share | NOT MEASURED |
| Training peak memory/headroom | NOT MEASURED |
| Two epoch estimates | NOT MEASURED |
| Finite aggregate loss | NO FORWARD/LOSS |
| GH200 use during accepted training | NO TRAINING INTERVAL |

The durable results explicitly state that Job `441511` answers none of the
owner's four G100 questions. Idle telemetry is not reported as model utilization;
no gate is omitted, weakened, or promoted to PASS.

#### Narrow remediation and persistent dependency state

- The current runner changes only the module selector plus its explanatory
  comment. It hashes to `855bbd15...` and passes `bash -n`. At source level this
  is output/model-neutral, but it is explicitly unexecuted and is not an approved
  or validated replacement tuple.
- Current Git state confirms the two job-generated tracked spconv stubs are back
  at HEAD. Spconv retains only the accepted unstaged `pyproject.toml` change,
  exact file SHA-256 `e2c84544...`, and tracked-state SHA-256 `499efdbb...`;
  cumm has no tracked change and tracked-state SHA-256 `f835ee22...`. The known
  pyproject change was not erased. One disclosed untracked cumm
  `core_cc/common.pyi` remains outside the tracked-source contract.
- Native state was not guessed, restored, or relabelled. Checksum-valid accepted
  S08 Q1 evidence records both cumm `core_cc` copies as 2,877,128 bytes / SHA-256
  `9970ccc5...`. Current inspection instead finds the package-root copy at
  2,877,128 bytes / `62332ad4...` and the build copy at 2,667,280 bytes /
  `18396a0c...`. Those changes invalidate direct reuse of aggregate cumm build
  identity `0a7e3c1a...`; exact GH200 restoration/rebuild and re-attestation must
  precede any later config that claims a dependency identity.

#### Artifacts, telemetry, and scheduler

- The output has nine regular files / 64,529 bytes, an empty readiness directory,
  and zero writable entries. `sha256sum -c artifact_sha256s.txt` passes for all
  eight listed files; the manifest hashes to `0c3e2947...`. Individual output,
  submit-script, and Slurm-log hashes reproduce the results ledger; the exit file
  is exact `1`.
- Telemetry contains 111 samples over 114.654 seconds: GPU and memory utilization
  always `0%`, memory use 8-10 MiB, power 81.66-86.38 W, SM/memory clocks
  345/2619 MHz, and temperature 33 C. This is usable only as idle failure-path
  telemetry.
- `sacct`/`scontrol` reproduce submit/start/end, `FAILED 1:0`, zero restarts,
  exact allocation/command/workdir/log paths, and batch `MaxRSS=1080M`.

### Gate verdicts

| Review gate | Verdict |
|---|---|
| Exact O-117 tuple and pre-submit freeze | PASS |
| Exactly one submission / no scope expansion | PASS |
| Immutable source/snapshot/config/submit identity | PASS |
| Pre-data/pre-loader/pre-model localization | PASS |
| Raw `cublasLt.h` diagnosis under wrong module stack | PASS |
| Current selector correction | PASS, static only / unexecuted |
| Tracked spconv/cumm source-state restoration | PASS |
| Reuse of frozen cumm executable-build identity | **BLOCKED; drift is real** |
| Artifact/telemetry/scheduler reproduction | PASS |
| O-117 loader/G100/timing/memory/epoch/utilization gates | **NOT TESTED** |
| Durable status/interface wording | PASS WITH P3 REMEDIATION |

### Minimal future owner envelope

No action follows automatically. A minimal amendment must authorize, in order:

1. one bounded GH200 dependency restoration/rebuild-and-attestation phase under
   `arrhenius_load_modules build`, binding exact mutable external paths, source
   HEAD/tracked states, expected generated/native outputs, artifacts, resource
   ceiling, fresh output and stop conditions, with no data/model training;
2. only after that phase yields an accepted immutable dependency identity, a new
   config dependency hash plus exact source/tree, corrected-runner hash, detached
   snapshot, submit-script hash, fresh output/log paths, and one explicit
   replacement G100 allocation; and
3. unchanged O-117 data identities, F-U/seed/precision/recipe, loader cells/order,
   update/attempt bounds, gates, single-GH200 resources and no-retry rule unless
   the owner deliberately amends one, followed by independent evidence review.

One owner decision may bind both phases, but G100 must remain conditional on the
dependency phase producing the newly recorded immutable identity. This review
does not recommend or authorize an unattested direct rerun.

### Residual risk

1. The shared editable cumm/spconv environment can mutate during the imports used
   to attest it; future allowed rebuild and later immutable training execution
   must be separated.
2. The corrected runner has not executed on GH200, and the exact post-rebuild
   aggregate cumm/spconv identity is unknown.
3. The untracked generated cumm stub remains outside tracked-state hashing and
   native artifacts remain changed.
4. Job `441511` supplies no evidence about production throughput, numerical
   health, time/step, component timing, memory headroom, effective GPU use,
   convergence, capability, metric, recipe, Protocol A/B, attack, or defense.

### Final verdict

**Evidence candidate `4fc78d508d4ac9ad7c46b9d3ad81c87646f8f0d3`:
PASS WITH RESIDUAL RISK as honest terminal failure/remediation evidence.** There
is no open P0-P2; reconcile the P3 wording in the linear review seal. No rerun is
needed to preserve or accept this negative evidence.

**S09 STOP-3 gate: REMEDIATE / BLOCKED PENDING NEW OWNER AUTHORITY.** Job
`441511` executed none of the G100 acceptance gates, O-117's sole submission is
consumed, frozen cumm build identity is invalid, and STOP-4 remains blocked.
Runtime re-attestation and any replacement G100 require the new exact owner
envelope above. This verdict authorizes no compute, merge, push, or scientific
interpretation.

### STOP-3 bounded closure re-review — seal `05f96c84c52216b39fa919067b135a77ac795028`

```text
CLOSURE_SHA: 05f96c84c52216b39fa919067b135a77ac795028
CLOSURE_TREE: a9744d737b6cee43d9028b4fc248d9a3577ffd17
CLOSURE_PARENT: 4fc78d508d4ac9ad7c46b9d3ad81c87646f8f0d3
BRANCH: codex/s08-s09-cl-readiness
REVIEWER_COMPUTE: none
```

The closure preflight matched exact HEAD/branch and a clean worktree. The
`4fc78d5..05f96c8` diff is limited to the prior review append and the four
requested P3 closures; `git diff --check` and `bash -n` for
`fl_v3/scripts/arrhenius_env.sh` pass.

Findings after bounded re-review:

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3: none open.** The prior P3 is closed in all four requested respects:
  `AGENTS.md` now records O-116 closure, consumed Job `441511`, no active compute,
  and the new-owner boundary; the STOP-2 block in `RUN_REQUEST.md` now records
  owner acceptance/closure; `arrhenius_env.sh` now accurately calls `run` a
  Miniforge-only path without changing a single executable statement or module
  behavior; and `HANDOFF.md` binds failure evidence SHA/tree plus the independent
  evidence/STOP-3 verdict.

The committed REVIEW section retains the exact failed source/request/evidence
identities, findings, gate table, residuals, and authority boundary reviewed
above. No model, config, data, trainer, loss, precision, optimizer, scheduler,
EMA, metric, test, or production-runtime behavior changed. Scheduler inspection
still shows only Job `441511`, terminal `FAILED`; no same-name replacement or
active job exists.

**Closure verdict: PASS WITH RESIDUAL RISK / no open P0-P3 for the terminal
failure-evidence seal.** Evidence candidate `4fc78d5` remains accepted only as
honest negative engineering evidence. The substantive residuals are unchanged:
the corrected STOP-3 runner is unexecuted, the cumm native build identity remains
drifted/unattested, and every loader/G100/performance/numerical gate remains
untested.

**STOP-3 remains REMEDIATE / BLOCKED PENDING NEW OWNER AUTHORITY.** This closure
does not authorize dependency rebuild/re-attestation, replacement G100, retry,
STOP-4, merge, push, or scientific interpretation.
