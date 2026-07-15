# S09 STOP-1 independent data/provenance review

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
