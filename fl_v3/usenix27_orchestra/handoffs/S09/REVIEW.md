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

### STOP-3 dependency-attestation bounded pre-compute review

```text
INITIAL_SHA: 42a9bff34b0517b11de144e7bda42b62524b7d3e
INITIAL_TREE: c48d7358f99e95b17f862a91726f5c40857d1817
INITIAL_PARENT: e221de672f8a1de010b01869b720ee5d6a523e8c
REMEDIATION_SHA: 788b493889bcf7be98f36b9cbb6686d51e8e5edf
REMEDIATION_TREE: 0bc61b3c2693f818ad0feb4e749af64a3947913e
REMEDIATION_PARENT: 42a9bff34b0517b11de144e7bda42b62524b7d3e
REVIEW_SCOPE: fl_v3/scripts/run_s09_stop3_dependency_attestation.sh only
REVIEWER_COMPUTE: none
```

The initial one-file implementation was narrow and did not load data, construct
a model, or train, but required remediation before a compute request could be
frozen:

- **P1:** `seal_on_exit` was not fail-closed. Status/manifest/chmod failures
  could be ignored, cleanup failures were not reliably accumulated, and the
  trap was installed after initial artifact writes.
- **P2:** before the mutation-capable warm import, the script bound the two
  checkout HEADs and tracked states but not the active editable distributions,
  exact versions, direct URLs, or import origins.
- **P2:** the EXIT cleanup reused the post-warm evidence filenames and therefore
  overwrote the concrete restored-stub path inventory with an empty second pass.
- **P2:** the two probes and acceptance artifact did not explicitly record the
  raw config-file SHA-256 and canonical resolved-config SHA-256.

The remediation closes all four findings. It installs the EXIT trap immediately
after output creation/entry and before the first artifact write or sparse import;
separately records original, cleanup, seal, and final statuses; propagates every
restore and artifact-seal failure to a nonzero result; and preserves independent
`post_warm` and `exit` restoration records plus an append-only restored-path
ledger. The pre-warm probe now fail-closes on exact package version, editable
file direct URL, expected checkout, HEAD, tracked state, and import origin without
importing cumm/spconv. Both fresh-process identity records and the acceptance
artifact now bind the config path, raw file SHA-256, and `ResolvedConfig.sha256`.
The existing spconv `pyproject.toml` diff remains outside the generated-stub
allowlist and is never restored.

Bounded static validation passed: `git diff --check`, `bash -n`, `shellcheck -x`,
and AST parsing of all four embedded Python heredocs. The review performed no
GPU compute, data/model/training execution, sparse import, or external checkout
mutation. No new P0-P3 was introduced.

**Pre-compute implementation verdict: PASS WITH RESIDUAL RISK / no open P0-P3.**
The two fresh-process probes are sufficient to bind a stable point-in-time
dependency identity, not to prove that a shared editable runtime remains
immutable after the job. A later G100 must fail-close against the exact emitted
source/build/config identities. EXIT sealing also cannot cover SIGKILL, node
loss, or power loss. The dependency-attestation execution intentionally permits
only its separately declared warm/build effects and bounded tracked-stub
restoration; those external effects must remain explicit in the exact request.

This review permits `788b493889bcf7be98f36b9cbb6686d51e8e5edf` to be used as
the immutable source for a proposed dependency-attestation compute request. It
is **not compute authorization** and does not authorize submission, retry,
replacement G100, STOP-4, merge, push, or scientific interpretation.

### STOP-3 O-118 Phase-A terminal evidence review

```text
EVIDENCE_SHA: 82a0e5315c9098056b6670afb490850cc71dc653
EVIDENCE_TREE: 7428f5978c8d423a7c1855d9e3f858eac718aeae
EVIDENCE_PARENT: 6323a5820863ed5c2e2d544efee3c6f53d98f9e5
EXECUTION_SOURCE_SHA: 788b493889bcf7be98f36b9cbb6686d51e8e5edf
EXECUTION_SOURCE_TREE: 0bc61b3c2693f818ad0feb4e749af64a3947913e
BRANCH: codex/s08-s09-cl-readiness
JOB: 442152
REVIEWER_COMPUTE: none
```

Findings, ordered by severity:

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3 — one stale evidence-state label.** At the immutable evidence commit under
  review, `HANDOFF.md` still says `immutable evidence commit pending` in
  `STOP3_DEP_ATTEST_EVIDENCE_STATE`. The commit above is that immutable evidence;
  the line should instead bind its SHA/tree and this review verdict. This is a
  documentation-only closure permitted by O-118 and does not require a rerun or
  change the Phase-B tuple.

The review preflight matched the exact evidence SHA, tree, parent, branch and a
clean worktree. Its diff changes only `HANDOFF.md`, `RUN_REQUEST.md`, and
`RESULTS.md`; it does not alter executable source, config, data, model, trainer,
precision, recipe, or metric behavior. `git diff --check` passes.

#### Scheduler, authorization, and uniqueness

- `sacct` and `scontrol` reproduce Job `442152` as the only
  `flv3_s09_stop3_dep_attest` submission in the review interval: `COMPLETED
  0:0`, zero restarts, submit/start/end
  `10:37:28/10:37:29/10:49:21`, node `n507`, elapsed `00:11:52` under a
  `00:20:00` limit, one GH200, eight CPUs, 32 GiB, one node/task, account
  `naiss2025-22-1113-gpu`, partition `gpu`, and `Requeue=0`. No active same-name
  job or replacement exists.
- Command, detached-snapshot working directory, stdout/stderr paths and resource
  tuple exactly match the O-118 Phase-A request. The read-only submit wrapper
  hashes to `93848490...`, contains one `sbatch`, and binds source `788b493...`,
  tree `0bc61b3...`, runner `a00d463...`, raw config `e8a17b39...`, canonical
  resolved config `cb172332...`, accepted source states, fresh output, and the
  no-retry condition.
- Phase A consumed `0.197778` GPU-hours. O-118's serial condition is respected:
  no data/model/training phase overlapped it, and Phase B has not been submitted.

#### Raw artifacts and dependency identity

- All 29 entries in `artifact_sha256s.txt` independently pass `sha256sum -c`;
  the manifest hashes to `b176faa8...`, acceptance to `4b60f319...`, and both
  fresh-process probes are byte-identical at `52b95699...`. Original, cleanup,
  seal, and final statuses are all zero. The 30-file, 1,637,056-byte output tree
  and both Slurm logs are read-only; warm/probe stderr and probe stdout are empty.
- Independent record-by-record verification finds 125 unique sorted cumm
  executable artifacts (7,858,249 bytes) and 73 spconv artifacts (91,328,998
  bytes). Every current file still matches its recorded size and SHA-256.
  Recomputing the exact metadata/path/size/content manifest algorithm reproduces
  cumm `0a7e3c1a...` and spconv `af422005...`.
- Both cumm native copies match at 2,877,128 bytes / `9970ccc5...`; both spconv
  native copies match at 45,180,616 bytes / `37f2ef8d...`. The current editable
  checkouts retain exact HEADs `4dedaf4...` and `263d6b4...`; spconv has only the
  accepted modified `pyproject.toml` (`e2c84544...`), cumm has no tracked change,
  and the disclosed untracked cumm `core_cc/common.pyi` remains outside tracked
  source identity. No generated tracked stub was restored.
- The snapshot remains detached and clean at `788b493...`; runner/config/submit
  hashes still match the frozen request and no writable worktree entry exists
  outside `.git`.

#### Scope and interpretation

The runner and acceptance artifact establish `data_loaded=false`,
`model_constructed=false`, and `training_attempts=0`. No nuScenes module/read,
loader, detector construction, forward/backward, optimizer update, timing, memory,
or utilization gate ran. Thus Job `442152` is dependency-attestation evidence
only, not evidence for any G100 training/performance/scientific conclusion.

Phase A did not retain a complete pre-warm executable-artifact manifest, so the
evidence does **not** support an exact per-file mutation-delta claim. It supports
the stable point-in-time post-warm identities reproduced by two fresh processes.
The warm build also emitted one compiler warning in generated spconv reverse-bit
code (`left shift count >= width of type`); compilation, import, both manifests,
and terminal sealing nevertheless completed. This warning and the shared editable
runtime remain residual risks, not demonstrated Phase-A failures. Phase B must
fail-close on the exact source/build identities before any data or model work.

#### Gate verdict and Phase-B decision

| Gate | Verdict |
|---|---|
| Exact approved Phase-A source/snapshot/command/resources | PASS |
| Exactly one submission / no retry / serial O-118 ordering | PASS |
| Scheduler completion, exit, restart and path record | PASS |
| Fail-closed terminal statuses and checksum-valid sealed output | PASS |
| Two byte-identical fresh-process dependency/config manifests | PASS |
| Final external tracked source and executable-build identities | PASS |
| No data/model/training execution | PASS |
| Exact pre-warm-to-post-warm native mutation delta | NOT CLAIMED / NOT AVAILABLE |
| Durable evidence-state wording | PASS WITH P3 REMEDIATION |

**Phase-A verdict: PASS WITH RESIDUAL RISK / no open P0-P2 or material semantic
concern.** The sole P3 may be sealed linearly without compute.

**O-118 Phase-B hard gate: GO FOR THE STRICTLY DERIVED TUPLE.** S00 may derive
only the two emitted sparse build hashes, perform the frozen local/static checks,
create the new immutable commit and self-contained snapshot, and obtain the
required independent derivation confirmation. This verdict does not itself
submit Phase B: submission remains contingent on that exact derived tuple being
recorded and rechecking the shared source/build identities. It authorizes no
retry, changed data/model/precision/recipe/resource field, STOP-4, merge, push,
or scientific interpretation.

### STOP-3 O-118 Phase-B strict derivation confirmation

Findings, ordered by severity:

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3: none.**

```text
PHASE_A_REVIEW_SEAL: 386fdbd34c9fe5d420e3ac6c8e439bfe65f6f74d
DERIVED_EXECUTION_SOURCE: c200bac861a42fc4338973787d3700e28ddd6c7e
DERIVED_EXECUTION_TREE: c0cc4cb8c2e207e42dcc45a129ada28a3d40feb8
DERIVED_EXECUTION_PARENT: 386fdbd34c9fe5d420e3ac6c8e439bfe65f6f74d
FROZEN_REQUEST_COMMIT: 45b67e3d2c7c49716d906af72847a5f9f5027f04
FROZEN_REQUEST_TREE: 5c4b176a5e52812c6f97d1a7a35317961da78990
FROZEN_REQUEST_PARENT: c200bac861a42fc4338973787d3700e28ddd6c7e
BRANCH: codex/s08-s09-cl-readiness
REVIEWER_COMPUTE: none
```

Preflight matched the exact request commit, tree, parent, branch, and a clean
worktree. `386fdbd..c200bac` changes exactly one file and one JSON value:
`dependencies.spconv_build_sha256` from the obsolete `74934de8...` identity to
Phase A's stable `af422005...`. The cumm value was already Phase A's
`0a7e3c1a...`, so it correctly produces no textual diff. Direct JSON comparison
against the original O-117 execution source `4d6bd82`, its request commit
`30e6c9f`, and the Phase-A review seal confirms there is no other config semantic
change. `c200bac..45b67e3` changes only `HANDOFF.md` and `RUN_REQUEST.md` to freeze
provenance; no executable or config content changes in the request commit.

The complete O-117 material tuple is preserved: F-U with
Swin-T-stride-8/SECOND-075/conv-fuser-256/CenterHead multitask, random
initialization, seed `0`, global FP16 plus the explicit sparse FP32 island,
AdamW `1e-4/0.01`, constant scheduler, no EMA, microbatch/accumulation/effective
global batch `1/1/1`, uniform sampling, eight training workers, exact production
train/val cache and ZIP-manifest identities, loader cells `0/2/4/8` with bounds
`2/32/16/256`, 100 successful updates within 120 attempts, and ten successful
warm-up windows excluded from timing. No model, data, precision, seed, recipe,
cell, bound, gate, or metric field changed.

#### Frozen snapshot, config, and executable identity

- Snapshot
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s09_stop3_g100_c200bac861a4`
  is detached and clean at exact source/tree `c200bac.../c0cc4cb...`, contains
  593 tracked files / 4,851,938 bytes, has zero writable worktree entries, no
  alternates or commit graph, one self-contained object pack, and passes
  `git fsck --full --no-reflogs`.
- Raw config SHA-256 is `6733a472...`; independent resolution reproduces
  `ba06b72e...` and the declared fusion/seed/update/worker/precision/execution
  fields. The G100 runner remains byte-identical at `855bbd15...` relative to
  the Phase-A review seal. `centralized_train.py` and `arrhenius_env.sh` reproduce
  `9284d395...` and `a56758d7...`.
- The corrected runner loads the required build module stack and licensed data
  module, then `centralized_train.py` calls
  `verify_runtime_dependency_identity` before physical data verification,
  loader creation, or model construction. Thus a later shared source/build drift
  fails before data/model work rather than silently changing the tuple.
- Independent local validation passed: exact semantic-diff assertions, config
  resolution, `git diff --check`, and `bash -n` plus `shellcheck -x` for both the
  runner and submit wrapper.

#### Submit wrapper and current external state

- The read-only wrapper at
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop3_g100_c200bac861a4_a2/submit.sh`
  hashes to `4801ddfe...`. It contains exactly one `sbatch`, no array/retry, and
  binds the exact source/tree/snapshot, runner/config hashes, absent output,
  empty exact-name queue, one GH200, 16 CPUs, 96 GiB, `01:00:00`, account
  `naiss2025-22-1113-gpu`, partition `gpu`, job name
  `flv3_s09_stop3_g100_r1`, fresh output/log paths, and `--no-requeue`.
- The exact output and log glob are absent. Both `squeue` and `sacct` show no
  Phase-B job with the frozen name; the conditional one-shot submission remains
  unconsumed.
- Current read-only inspection still reproduces spconv HEAD/state/build
  `263d6b4.../499efdbb.../af422005...` and cumm
  `4dedaf4.../f835ee22.../0a7e3c1a...`. All 73 spconv and 125 cumm executable
  artifact records from Phase A still match current sizes/content and recompute
  the exact aggregate identities. Both native-copy pairs remain byte-identical
  at `37f2ef8d...` and `9970ccc5...`. The accepted spconv `pyproject.toml`
  modification and disclosed untracked cumm stub are unchanged.

| Strict-derivation gate | Verdict |
|---|---|
| Exact linear Phase-A review → derived source → frozen request chain | PASS |
| Only Phase-A sparse build identity derived in executable/config state | PASS |
| O-117 model/data/precision/seed/recipe/cells/bounds/gates unchanged | PASS |
| Corrected runner and production trainer/environment hashes frozen | PASS |
| Local semantic/config/static validation | PASS |
| Detached clean self-contained snapshot and inventory | PASS |
| Single exact submit command/resources/output/logs/no-requeue | PASS |
| Output/logs absent; exact-name queue/accounting empty | PASS |
| Current external source, tracked state, native files, aggregate builds | PASS |
| O-118 no-retry/serial authority and one conditional submission available | PASS |

Residual risk remains bounded but real: cumm/spconv are shared editable
checkouts and may drift after this point-in-time confirmation; the job therefore
must retain its pre-data/model fail-closed identity check. The corrected G100
runner has not yet completed a training execution, and Phase A did not establish
an exact pre-warm mutation delta. This confirmation predicts neither numerical
health nor performance: loader, 100-update, timing, memory, telemetry, and large
LiDAR-gradient gates remain to be measured by the exact job.

**Strict-derivation verdict: PASS WITH RESIDUAL RISK / no open P0-P3.** The
O-118 derivation rule is satisfied without changing any scientific or resource
field.

**SUBMIT VERDICT: GO for exactly the frozen Phase-B wrapper and one submission.**
This GO is invalidated by any source/config/snapshot/wrapper/output/queue or
external dependency drift and permits no retry, replacement, altered resource,
additional cell, STOP-4, merge, push, or scientific claim.

### STOP-3 O-118 Phase-B terminal evidence review

```text
EVIDENCE_SHA: c28d09c34b0ff56fcbc3805a8361ccd26eeaccc1
EVIDENCE_TREE: 6c8f008434363dcf41c8f30bdbbaecb4a67863a4
EVIDENCE_PARENT: d7754e0ae0a1e05708545c934ecb507b933a32b6
EXECUTION_SOURCE_SHA: c200bac861a42fc4338973787d3700e28ddd6c7e
EXECUTION_SOURCE_TREE: c0cc4cb8c2e207e42dcc45a129ada28a3d40feb8
BRANCH: codex/s08-s09-cl-readiness
JOB: 446225
REVIEWER_COMPUTE: none
```

Findings, ordered by severity:

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3 — the durable summary reports the CUDA-only ratio instead of the exact
  frozen combined-window ratio.** O-117 freezes
  `(data_wait + CUDA H2D-through-update) p95/p50 <= 1.5`. `RESULTS.md` and
  `RUN_REQUEST.md` report `224.153076 / 208.575935 = 1.074683`, which is the
  CUDA-event window alone. Pairwise addition of each of the 90 raw record's data
  wait and CUDA-window values gives combined mean/p50/p95
  `210.760627 / 208.745739 / 224.326678 ms` and the exact frozen ratio
  `1.074641`. The actual frozen gate therefore still passes by a wide margin;
  this is a documentation/evidence-summary correction, not a failed gate or a
  reason to rerun.
- **P3 — active status/provenance prose is not yet synchronized to O-118 and
  this immutable evidence.** `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`
  still terminate STOP-3 at failed Job `441511`/O-117 and say that a new owner
  amendment is required. The direct O-118 owner decision and exact
  `HANDOFF.md`/`RUN_REQUEST.md` record supersede that stale status, so it does
  not make Jobs `442152` or `446225` unauthorized. It is nevertheless a P3 in
  the canonical active ledgers and must be corrected linearly before STOP-3
  closure. The same stale terminal boundary in root `AGENTS.md` and
  `docs/env.md` should be synchronized in that bounded documentation update.
  Within the evidence package, labels such as `immutable evidence ... pending`
  are necessarily stale now that the SHA above exists; the linear seal should
  bind this exact SHA/tree and review verdict. No source or compute change is
  required.

The review preflight matched the exact evidence SHA, tree, parent, branch and a
clean worktree. Its diff changes only `HANDOFF.md`, `RESULTS.md`, and
`RUN_REQUEST.md`; executable source, config, model, trainer, data and precision
semantics are unchanged. `git diff --check` passes.

#### Authorization, scheduler, and immutable identities

- `sacct` and `scontrol` reproduce Job `446225` as the only same-name
  `flv3_s09_stop3_g100_r1` submission: `COMPLETED 0:0`, zero restarts,
  `Requeue=0`, submit/start/end `11:09:11/11:09:12/11:14:17`, elapsed
  `00:05:05` under a `01:00:00` limit, node `n450`, one GH200, 16 CPUs, 96 GiB,
  one node/task, account `naiss2025-22-1113-gpu`, and partition `gpu`. No active
  same-name job, array, DDP, retry or replacement exists.
- The command, detached snapshot, output/log paths and resources match the
  reviewed strict derivation. The execution snapshot remains detached and clean
  at `c200bac.../c0cc4cb...`; raw config, canonical resolved config, runner and
  read-only wrapper reproduce `6733a472...`, `ba06b72e...`, `855bbd15...`, and
  `4801ddfe...`. The wrapper contains one `sbatch` and no retry path.
- The job consumed `0.084722` GPU-hours; O-118 Phase A/B consumed `0.282500`
  GPU-hours and all S09 jobs through STOP-3 consumed `0.393333` GPU-hours. No
  unused allocation is retry or STOP-4 authority.

#### Raw artifacts, runtime, and external dependency state

- The sealed output contains 12 regular files / 3,178,950 bytes / two
  directories, no symlink and no writable entry. All 11 entries in
  `artifact_sha256s.txt` independently pass content verification; the manifest
  hashes to `b2296338...`, the readiness report to `08e376e7...`, and
  `centralized_train.exit` is exactly zero. Both Slurm logs are read-only.
- The two stdout JSON objects are structurally identical to the standalone
  runtime-dependency and readiness artifacts, including the nested dependency
  record. Independent canonicalization reproduces runtime-dependency identity
  `3e900c90...`, resolved-config identity `ba06b72e...`, and execution identity
  `01b7c5d1...`. Stderr contains only one Torch FX warning and one deprecated
  spconv multidimensional-indexing warning, with no exception.
- The accepted train/val `t1.v2`, ten-sweep and ZIP-manifest identities match the
  frozen tuple. Post-job read-only inspection also reproduces all 73 spconv and
  125 cumm executable artifacts and aggregate builds `af422005...` and
  `0a7e3c1a...`; both native-copy pairs remain byte-identical. The accepted
  spconv `pyproject.toml` tracked modification and disclosed untracked cumm stub
  are unchanged. Shared editable sparse dependencies remain a residual runtime
  risk, not a mismatch in this execution.

#### Loader, numerical health, counters, timing, and memory

- All eight `0/2/4/8`-worker repeat cells completed the exact bounded
  `2/32/16/256` profile. Their declared 32-batch determinism digests are equal at
  `6c6d8f06...`; 2,432 samples were consumed in total. Warm throughput is
  `21.866943/41.573781/77.519849/141.969756 samples/s`, so worker eight is the
  best cell and passes the frozen `>=90%` gate at `100%`.
- The lifecycle records exactly 103 attempted windows and 100 successful
  optimizer/scheduler/exposure updates. Attempts 1--3 are visible GradScaler
  overflows with scale `512 -> 256 -> 128 -> 64`; attempts 4--103 are 100
  consecutive accepts. After ten successful warm-up updates, the measured
  region is 90/90 accepted at scale 64, with zero nonfinite, discarded or
  post-warm invalid windows. Loss is finite (`55.333761`), pending accumulation
  is zero, scheduler `last_epoch=100`, and all optimizer/exposure/counter
  identities reconcile. This establishes bounded engineering health after
  scaler backoff, not convergence, recipe quality, or resolution of the large
  true LiDAR gradients.
- The complete training section is `46.575035 s`, or `0.465750 s` per successful
  update including cold compilation and the three overflow attempts. The frozen
  post-warm region is 90 updates in `18.973603 s`: `0.210818 s/update` and
  `4.743432 updates/s`. Independent recomputation of all 90 records exactly
  reproduces the documented H2D, forward, loss, backward, optimizer and CUDA
  window mean/p50/p95 statistics. Direct stage medians are
  `0.277600/90.004463/10.870528/100.688049/6.010576 ms`; they do not partition
  camera, LiDAR and fusion subgraphs.
- Data wait is `0.160926/0.164987/0.187478 ms` mean/p50/p95 and contributes only
  `0.076355%` of the integrated mean. Peak allocated/reserved memory is
  `3.256302/6.433594 GiB`, leaving `88.566406 GiB` against the reported 95-GiB
  total. Both frozen steady epoch estimates independently reproduce
  `1.647307 h`; the descriptive setup-inclusive first epoch is `1.706773 h`.
  All corresponding O-117 limits pass.

#### Coarse telemetry and interpretation boundary

The 1-Hz telemetry has 263 data rows. Applying the declared alignment to the
estimated full training lifecycle yields the same 45 samples and independently
reproduces GPU utilization mean/p50/p95/max `32.4/32/99.8/100%`, memory mean/
p50/p95 `5550.6/6419/7503 MiB`, and power mean/p50/p95/max
`167.92/193.79/207.01/257.64 W`. Utilization is nonzero in `66.67%`, at least
50% in `31.11%`, and at least 80% in `8.89%` of those samples. This supports the
limited statement that the complete 100-update lifecycle did not continuously
saturate the GH200. The interval includes cold/JIT/backoff time, its boundaries
carry approximately one-sample uncertainty, and 1-Hz device telemetry plus
stage-level events cannot identify steady-state kernel occupancy, Tensor-Core
use, or a camera/LiDAR/fusion root cause. Those remain residual risks and
explicit non-claims.

| Frozen STOP-3 gate | Independent disposition |
|---|---|
| Exact source/config/runtime/data/dependency identity | PASS |
| Exactly one authorized Phase-B submission / no retry | PASS |
| All loader digests equal | PASS |
| Worker-8 warm throughput `>=90%` of best | PASS (`100%`) |
| 100 successful updates within 120 attempts | PASS (`100/103`) |
| Zero nonfinite/discarded windows and exact counters | PASS (`0/0`) |
| Post-warm accepted ratio `>=95%` | PASS (`90/90 = 100%`) |
| Combined `(data_wait + CUDA window)` p95/p50 `<=1.5` | PASS (`1.074641`) |
| Data-wait share `<=10%` | PASS (`0.076355%`) |
| Peak reserved memory `<=86 GiB` | PASS (`6.433594 GiB`) |
| Both steady epoch estimates `<=24 h` | PASS (`1.647307 h`) |
| Finite aggregate loss | PASS |
| Continuous GH200 saturation or branch/kernel attribution | NOT CLAIMED |
| Convergence, recipe quality, capability, metric or science | NOT CLAIMED |
| Canonical/evidence-state wording | PASS WITH P3 REMEDIATION |

**STOP-3 technical verdict: PASS WITH RESIDUAL RISK / no open P0-P2.** Every
frozen execution gate independently passes, including the exact combined-window
ratio. The two documentation-only P3 findings should be corrected and sealed
linearly; neither requires compute or changes the technical outcome.

Subject to that bounded documentation closure, this evidence is owner-ready for
a STOP-3 PASS/REMEDIATE decision. It supports one exact F-U/seed/runtime tuple's
100-update engineering stability, performance, memory and coarse-utilization
record. It does not establish convergence, detector capability, mAP/NDS,
per-branch causality, normalization health, multi-seed behavior, Protocol A/B,
FL, attack or defense. This review supplies no retry, STOP-4 compute, merge or
push authority.

### STOP-3 O-118 Phase-B documentation closure re-review

Findings, ordered by severity:

- **P0: none.**
- **P1: none.**
- **P2: none.**
- **P3: none.**

```text
REMEDIATION_SHA: 84adfd05354a356aeb64a5a30a72153980826859
REMEDIATION_TREE: 265973df165c08fdd95d41f473a0bc80f7010b42
REMEDIATION_PARENT/EVIDENCE_SHA: c28d09c34b0ff56fcbc3805a8361ccd26eeaccc1
BRANCH: codex/s08-s09-cl-readiness
REVIEWER_COMPUTE: none
```

Preflight matched the exact remediation SHA, tree, parent, branch and a clean
worktree. `git diff --check c28d09c..84adfd0` passes. The complete diff changes
only eight active documentation/ledger files plus the previously authored
terminal `REVIEW.md` section: root `AGENTS.md`, `docs/env.md`, the three canonical
Orchestra documents, and `handoffs/S09/{HANDOFF,RESULTS,RUN_REQUEST,REVIEW}.md`.
No source, config, runner, test, runtime, data, model, precision, recipe, metric,
resource or artifact content changes.

The first P3 is closed. `RESULTS.md` now labels the original
`210.599701 / 208.575935 / 224.153076 ms` distribution and its `1.074683` ratio
as CUDA-only stage diagnostics. Both `RESULTS.md` and `RUN_REQUEST.md` use the
pairwise host-data-wait plus CUDA-window distribution as the frozen O-117 gate.
Independent read-only recomputation over all 90 measured raw records reproduces
combined mean/p50/p95 `210.760627245 / 208.745738839 / 224.326677561 ms`, ratio
`1.074640751030`, and data-wait share `0.000763550995 = 0.076355100%`. The durable
rounded values `210.760627 / 208.745739 / 224.326678 ms`, `1.074641`, and
`0.076355%` are therefore correct. The gate remains PASS; no threshold or result
was changed retroactively.

The second P3 is closed. `AGENTS.md`, `docs/env.md`, `ORCHESTRA.md`,
`SESSIONS.md`, `KICKOFFS.md`, and the S09 handoff package consistently record:

- O-117 Job `441511` as retained pre-model negative evidence;
- O-118's bounded, serial, no-retry authority and completed Jobs `442152` and
  `446225`;
- immutable evidence `c28d09c`, independent
  `PASS_WITH_RESIDUAL_RISK`, and no P0-P2 in the terminal review;
- consumed O-118 compute, no active request or retry, and no STOP-4, merge or
  push authority; and
- reviewed STOP-3 technical PASS with documentation closure complete at the
  remediation-candidate level, while **owner STOP-3 acceptance/closure remains
  pending**.

The synchronization does not convert bounded engineering health into convergence,
model capability, recipe quality, mAP/NDS, branch/kernel causality, LiDAR-gradient
root cause, multi-seed, Protocol A/B, FL, attack or defense evidence. It does not
modify any execution identity or create new compute authority.

**Closure verdict: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / owner-ready
STOP-3.** Both documentation-only findings are closed without compute or semantic
change. The remaining residual risks are those already recorded in the terminal
review: shared editable sparse dependencies, the unresolved large true LiDAR
gradients, bounded single-tuple/100-update evidence, and coarse lifecycle-level
telemetry without branch/kernel attribution.

S00 may now create only a linear review-artifact seal and mechanically replace
`closure candidate` / `documentation closure pending` labels with
`P3 closed / owner-ready / owner decision pending`. That seal must not change any
number, claim or authority and does not require a third review. This verdict
authorizes no compute, retry, STOP-4, merge or push.

## STOP-4A initial implementation review

```text
CANDIDATE_SHA: 5a577062bf0c06faf1f1fa67c209e734569d855e
CANDIDATE_TREE: e8b68372cf73b5a96ea03a5c4dcb4cccd3edb477
CANDIDATE_PARENT: 6b6cf2c60f8c54cfc2e24c7507b8dcc853db4566
REVIEWER: independent S09 STOP-4A implementation reviewer
REVIEWER_COMPUTE: none
VERDICT: REMEDIATE
```

Findings, ordered by severity:

- **P0/P1: none.**
- **P2 — cell PASS classification was too weak.** The runner relied on process
  exit plus `readiness.status`, while the producer did not reject direct
  nonfinite windows or reconcile scheduler/exposure counters. Its generic
  `"out of memory"` match could also misclassify a non-CUDA B2/B4 failure as a
  capacity limit. Remediation must share one fail-closed numerical/counter/
  artifact validator between producer and runner, require explicit CUDA-OOM
  evidence, and cover positive/negative cases.
- **P2 — profiler and throughput windows were not fail-closed separated.** The
  v2 resolver required the profiler schedule to fit within the optimizer bound,
  but did not require timing warm-up to cover that complete schedule. The exact
  candidate tuple is safe (`12 >= 5+2+3`); the schema still requires the general
  invariant and a negative test.
- **P2 — the bounded summary could lose its central evidence.** Input shapes
  were not serialized, and a global self-device-time top-250 could omit all
  low-self-time `fl_v3::*` parent ranges. Remediation must preserve every expected
  F-U module range outside the operator cap, emit shape metadata, and fail if the
  expected range set is incomplete.
- **P3 — profiler retention wording was too broad.** Named ranges add no
  application hook/activation retention, but Torch `record_shapes=true` may
  temporarily hold tensor references during active diagnostic windows. The
  handoff and kickoff must state that those windows are excluded from capacity/
  throughput evidence.

The review independently confirmed that legacy `s09.v1` semantics/hash remain
unchanged; all four STOP-4A hashes and intended semantic deltas are correct; the
checkpoint switch maps to Swin; the attempted-window callback occurs once after
complete window accounting; the required Torch 2.11 APIs exist; and no worker
matrix, DDP, recipe, model/loss math, precision, or data change entered the
candidate. Static checks passed. The named ranges cover forward modules only:
they do not support an exact camera/LiDAR/fusion backward decomposition.

**Initial verdict: REMEDIATE.** No STOP-4A compute may be submitted from this
candidate. S00 may remediate the findings linearly under O-119, seal one new
immutable SHA, and request independent re-review; no retry, merge, or push is
authorized.

### STOP-4A implementation closure re-review

```text
REMEDIATION_SHA: b509f5e527c2dd28d2db506c3f87b5a06b3b1b6a
REMEDIATION_TREE: 9c556d37d1e45ece7aad31b10881bb9eb8686424
REMEDIATION_PARENT: 5a577062bf0c06faf1f1fa67c209e734569d855e
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3
```

The independent reviewer found no P0, P1, P2, or P3. All four initial findings
are closed: producer and runner share one fail-closed readiness validator;
explicit CUDA-OOM signatures replace the generic match; schema v2 requires the
timing warm-up to cover the profiler schedule; every expected F-U range is kept
outside the operator top-k with shape metadata and completeness checks; and both
active documents accurately qualify `record_shapes` retention.

The reviewer directly constructed a valid `20 accepted + 1 GradScaler overflow`
ledger, which produced zero validation errors, while a direct-nonfinite ledger
was rejected. Legacy `s09.v1` retains resolved hash `ba06b72e...`, checkpoint-on
mapping and no operator profile; all four v2 hashes remain exact. Missing/non-
boolean production checkpoint fields fail closed. Static compile, all three
runner Python heredocs, `bash -n`, ShellCheck, diff checks, schema positive/
negative cases and clean-worktree checks passed. No model output, loss, gradient/
update, data/order, precision, optimizer/scheduler/EMA, or recipe semantic change
was found, and the single bounded profiler does not restore an observer, hook
chain, worker matrix, or general harness.

Residual risk is bounded to actual aarch64 Torch/CUDA focused tests, which have
not run yet, and the fact that forward named ranges cannot provide exact C/L/F
backward decomposition. **This is implementation closure, not submission GO:**
the exact source/tree/snapshot/submit/output tuple requires a separate immutable
request freeze and independent request review before the sole O-119 STOP-4A job.

### STOP-4A exact request review — pre-submit GO

```text
REQUEST_SEAL: 6724762d1ae719f3e20a3014565ae158f024c911
REQUEST_SEAL_TREE: caf2ba5d1a4d8557ef0192a9a268ce01f6d02c22
EXECUTION_SOURCE/TREE: b509f5e527c2dd28d2db506c3f87b5a06b3b1b6a / 9c556d37d1e45ece7aad31b10881bb9eb8686424
REVIEWER_TASK: /root/s09_stop4a_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / SUBMIT GO
```

The independent request reviewer verified the exact detached, clean,
self-contained 598-file snapshot, absence of Git alternates/symlinks/writable
worktree entries, all runner/trainer/environment/config/resolved identities, the
accepted cache/manifest and editable sparse-dependency states, fresh output and
empty exact-name queue. The sole wrapper contains one non-requeue `sbatch` for one
GH200, 16 CPUs, 96 GiB and `00:30:00`; it has SHA-256 `fb59ad99...` and binds the
four cells/order/bounds, F-U O-110 precision, seed 0, recipe, source/tree and
output. No worker matrix, DDP, array, retry, replacement, or extra cell exists.

The pre-submit verdict restricted GO to:

```bash
bash /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_requests/s09_stop4a_profile_capacity_b509f5e527c2/submit.sh
```

This exact review completed before that command produced Job `452520`. The
request verdict was retained in the reviewer task but was not immediately sealed
into this file; the later evidence review identified that durable-record gap.

## STOP-4A evidence plus STOP-4B/4C implementation review

```text
CANDIDATE_SHA: 6da4bb5016410708b1e731d26d898f24e6b315ac
CANDIDATE_TREE: 721165340f2b5ab4cda222b4f3a86e951f9d7c14
CANDIDATE_PARENT: 6724762d1ae719f3e20a3014565ae158f024c911
REVIEWER_TASK: /root/s09_stop2_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: REMEDIATE / documentation and evidence sealing only
```

Findings, ordered by severity:

- **P0/P1: none.**
- **P2 — the exact STOP-4A request-review GO was not durable and active state
  contradicted Job 452520.** The reviewer confirmed from the retained pre-submit
  reviewer verdict that this was not an unauthorized or retroactively approved
  job. The task identity, exact tuple, wrapper and GO must be sealed and the
  handoff/request/results state synchronized.
- **P3 — two STOP-4A GiB values were transcribed with incorrect rounding.** The
  correct two-decimal values are B1-profile `3.06 / 5.38 / 89.63 GiB` and
  B1-no-checkpoint headroom `87.76 GiB`; the capacity conclusion is unchanged.
- **P3 — Job 452520 Slurm stdout/stderr remained mode 0664 and unbound.** They
  contain only empty output/module notices and no technical result, but must be
  made read-only and their hashes recorded.

The technical review found no model, loss, gradient, precision, data, recipe or
runner semantic defect. The propagated switch removes exactly `6*3+1=19`
diagnostic `.item()` calls per ordinary attempted window, while S08 diagnostics
force complete task/aggregate terms and `finally` restores caller state. The
true six-task equality test checks exact loss and every output gradient. STOP-4C
retains the STOP-3 F-U/B1 data, precision, seed, optimizer and 100/120 bounds;
only schema v2, explicit checkpoint-off, and null loader/operator profiles differ.
The one-shot runner has no matrix, profiler, DDP or retry and covers the new paths.

Raw Job `452520` evidence independently reproduces `COMPLETED 0:0 / 00:09:42`,
59 passed tests, four PASS cells, all 35 manifest checks, trace/summary/telemetry,
and the documented capacity limits. B2/B4 remain capacity-only and the profiled
checkpoint comparison is not treated as a clean speed ablation.

**Verdict: REMEDIATE; STOP-4C NO-GO until linear documentation/evidence
remediation, immutable closure re-review, exact request freeze, and independent
request-level GO.** No code change, GPU rerun, or reinterpretation is required.

### STOP-4B/4C implementation-evidence closure re-review

```text
FINAL_CLOSURE_SHA: 1a0b7e38805d86fb42ff4fe84d67e1680de55015
FINAL_CLOSURE_TREE: b76d9a480bcd9654ae63e72bdbb5d99191902829
FINAL_CLOSURE_PARENT: a18e5e0fe6fea62a5ba1fc8d6ad33c3c7db1421d
IMPLEMENTATION_SHA: 6da4bb5016410708b1e731d26d898f24e6b315ac
REVIEWER_TASK: /root/s09_stop2_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / implementation-evidence closure GO
```

The first remediation `a18e5e0` durably sealed the pre-submit STOP-4A request
review and synchronized active/job state, corrected both memory values, and bound
the now-read-only Slurm logs. Closure review found those P2/P3 findings closed and
identified one remaining stale `RESULTS.md` title. Commit `1a0b7e3` changes only
that line to `STOP-1/2/3 closed / O-119 STOP-4A-D active`. Final re-review matched
SHA/tree/parent/branch, clean worktree, and `git diff --check`, with no other file,
technical, evidence, or scientific change.

Findings: **P0 none; P1 none; P2 none; P3 none.** The code implementation remains
the reviewed `6da4bb5` tree content; the later commits are documentation/evidence
closure only. Residual risk is actual GH200 execution of the new checkpoint-off
100-update path. This verdict permits an exact STOP-4C request freeze. It is not
request-level submission GO; the frozen source/snapshot/config/wrapper/output
tuple still requires independent request review.

## STOP-4C first exact-request review — SUBMIT NO-GO

```text
REQUEST_SEAL: 131619f0940bd3c453969f4d211bdaa775bacbb8
REQUEST_SEAL_TREE: 1e4cf9808592ea587e756ace02f77cd226ecdc02
EXECUTION_SOURCE/TREE: 1a0b7e38805d86fb42ff4fe84d67e1680de55015 / b76d9a480bcd9654ae63e72bdbb5d99191902829
OLD_WRAPPER_SHA256: eec841bf452f2f5c8adc0908c67c538aaee1c2842322313e5beb4096e7ae00be
REVIEWER_TASK: /root/s09_stop4a_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: REMEDIATE / SUBMIT NO-GO
SUBMISSION: none
```

Findings, ordered by severity:

- **P0/P1: none.**
- **P2 open — the frozen runner did not fail-close the request's performance
  gates.** Its terminal validation called the shared numerical/counter/artifact
  validator but did not enforce post-warm acceptance, combined p95/p50, data-wait
  share, 86-GiB memory, two epoch estimates, finite aggregate loss, or the
  STOP-3 p50/p95 1.10x ceiling. A material slowdown could therefore exit zero and
  incorrectly unlock STOP-4D. Source remediation, positive/negative replay, a new
  immutable snapshot/request and re-review are required.
- **P2 closed mechanically during review — the initial snapshot was mode-dirty.**
  A blanket 0444 freeze removed executable bits from 37 tracked 100755 scripts;
  Git status was nonempty and the runner/environment were not executable. S00
  restored 100755 files to 0555 and 100644 files to 0444 from the Git index.
  The retained snapshot is now detached/clean/no-alternates/zero-writable, its
  runner/environment are executable, and all tree/content hashes are unchanged.
  No job ran, but the old request's historical freeze-time-clean assertion was
  not true and the replacement must use a fresh snapshot.
- **P3: none.**

All other fields passed review: source/config/data/cache/manifest and sparse
dependency identities; F-U/B1/seed0/O-110/checkpoint-off/AdamW recipe; 100/120
and warm-up ten; no profiles/matrix/DDP/retry; output/queue/history freshness;
one non-requeue GH200/16-CPU/96-GiB/30-minute resource; and O-119 worst-case
`0.161667 + 0.5 + 1 = 1.661667` GPU-hours. The 1.10x arithmetic is correct.

**The old `131619f` tuple and wrapper are forbidden and must not be executed.**
This review authorizes only bounded source/request remediation and a new
independent review; it does not authorize STOP-4C submission.

### STOP-4C runner-gate remediation closure review

```text
REMEDIATION_SHA: 72a09d5a503a258f3f257b208180585d16ee49d0
REMEDIATION_TREE: 887d275b71a7f6ccd34cf67188e9fac0843393c1
REMEDIATION_PARENT: 131619f0940bd3c453969f4d211bdaa775bacbb8
REVIEWER_TASK: /root/s09_stop4a_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / runner-gate remediation closure GO
```

Findings: **P0 none; P1 none; P2 none; P3 none.** The diff is restricted to a
shared readiness performance validator, the existing STOP-4C runner terminal
validation, focused positive/eight-boundary negative tests, and durable NO-GO
records for the old request. No model, loss, gradient, precision, recipe, data,
config, worker, profiler or harness semantic change exists.

The reviewer independently verified that accepted combined latency uses
`data_wait_ms + CUDA window` over measured accepted windows; data-wait share uses
all measured attempts; and epoch rates separately use measured attempted samples
and exposure samples per wall second. Post-warm overflows therefore leave accepted
latency clean while reducing accepted ratio/exposure rate and remaining visible in
data-wait accounting. All inclusive O-117 thresholds and the exact 1.10x limits
are correctly implemented.

Replay of accepted Job `446225` reproduced combined p50/p95
`208.7457388394978 / 224.32667756068986 ms`, ratio `1.0746407510295195`, wait
share `0.0007635509950495559`, both epoch estimates
`1.6473069676229828 h`, peak reserved `6,908,018,688` bytes and no errors.
Doubling measured CUDA windows produced the expected regression error. Eight gate
classes plus malformed/empty/zero inputs fail closed. Validation exit `4`
propagates to `final.exit` and the process while artifacts are sealed.

Residual risk is the unexecuted replacement GH200 path. This verdict closes the
implementation P2 only. The old `131619f` tuple remains forbidden; a fresh source/
snapshot/config/runner/wrapper/output tuple and independent request review remain
mandatory before submission.

### STOP-4C replacement exact-request closure review — SUBMIT GO

```text
REQUEST_SEAL: bbb807188a5e024e9e569480bd3a50676f0df312
REQUEST_SEAL_TREE: 565d9467b92bce38c551e4a9baea58190be3d85c
REQUEST_SEAL_PARENT: 75f759274c755af025fa5fdfc350e60595edafcb
EXECUTION_SOURCE/TREE: c7769901201b8c507997dfa9ff5154fbe6dbb297 / 1e2c4464d2582d81e7ef7fef4740c764d0a48e8c
REVIEWER_TASK: /root/s09_stop4a_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / SUBMIT GO
```

The replacement review found no P0, P1, P2 or P3. The final request-seal diff
only corrects three stale tense lines; the execution tuple remains unchanged.
The reviewer reproduced the detached/clean/read-only snapshot and preserved
executable modes, all runner/trainer/environment/raw/resolved config hashes,
accepted data/dependency identities, complete numerical/performance gates, exact
F-U/B1/O-110/seed/recipe/bounds, resource and O-119 budget, and fresh output/
queue/history state. Wrapper `e44db31b...` passes syntax, shell and safe-prefix
checks and contains one non-requeue `sbatch`.

The GO is restricted to the exact replacement command and wrapper recorded in
`RUN_REQUEST.md`. Any source/snapshot/config/wrapper/output/resource/freshness
drift cancels it. The old `131619f` tuple remains forbidden and unsubmitted; this
replacement is not a retry. Residual risk is the actual GH200 result.

### STOP-4C immutable evidence closure review — PASS / STOP-4D release GO

```text
EVIDENCE_SHA/TREE: 32b380ccae5dc0146e3c5b494e0f3d4d1ae9d7cd / d35b97f92783f3a953f9402b8966d428e23078a0
REMEDIATION_SHA/TREE: 8b7542c648565508a6b96f6378a0172d255a8b61 / b87e583e51d8a09435eea17e547e6871f2bcdb9d
REMEDIATION_PARENT: 32b380ccae5dc0146e3c5b494e0f3d4d1ae9d7cd
REVIEWER_TASK: /root/s09_stop4a_impl_reviewer
REVIEWER_COMPUTE: none
VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3 / STOP-4D conditional release GO
```

The first evidence review found no P0-P2 and three documentation-only P3s: a
stale request-review heading, a nearest-rank rather than project-linear telemetry
p95, and an over-broad neutrality sentence. Remediation `8b7542c` changes only
`RUN_REQUEST.md` and `RESULTS.md`: it records the consumed Job 455539 state,
corrects the 43-sample GPU-util p95 to `99.9%`, and limits exact equality to the
quiet-loss value/input-gradient path while describing checkpoint-off through its
explicit construction checks and G100 evidence. The reviewer verified the clean
immutable diff and closed all three P3s.

Independent raw-evidence checks reproduce 16/16 artifact hashes and manifest
`542862b2...`, four zero exits, Job 455539 `COMPLETED 0:0` in `00:04:06` with no
restart, exact 100/103 lifecycle accounting, three scaler overflows followed by
90/90 measured accepted windows, and the fail-closed p50/p95/ratio/epoch/reserved
metrics. Execution source, config, runner, model, precision, recipe, raw artifacts
and gates did not change. O-119 actual usage is `0.230000` GPU-hours; adding the
conditional one-hour STOP-4D ceiling remains within `1.230000/2.000000`.

This verdict releases only preparation and exact independent request review of a
fresh B1 G1000 tuple, followed by the already approved single no-retry submission
if that request receives GO. It is not convergence, recipe, mAP/NDS, model-quality
or scientific-performance acceptance. Residual risks include single seed/B1,
coarse utilization telemetry, unexecuted G1000 and the unresolved large LiDAR
gradient question deferred outside S09 engineering readiness.
