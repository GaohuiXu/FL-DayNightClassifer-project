# S01-R independent review — shared nuScenes ZIP backend

## Review identity and final verdict

- Session: `S01-R`, independent reviewer.
- Review baseline / `WORKER_SHA`:
  `ce2e77284b290de4c9faa6b2f971c0bd52f98eff`.
- Implementation base:
  `f262f6bea037580065a8505008773c04fdd259f5`.
- Exact reviewed diff:
  `f262f6bea037580065a8505008773c04fdd259f5..ce2e77284b290de4c9faa6b2f971c0bd52f98eff`.
- Runtime implementation commits represented in the handoff:
  `011e4640d26330e2c8145fcdb56833fe19e7b67d` and
  `1fe651700bd06a07707307c60ad4e31cc9d1e0ba`.
- `APPROVED_COMPUTE`: none. No Slurm job was submitted by S01-R.
- Final verdict: **CHANGES-REQUESTED**.

The v2 manifest counts, full train/val reference coverage, ten distinct real-shard
sentinels, repeated loader digests, loader-only timings, and scheduler/resource
records are supported by the raw artifacts. Final S01 acceptance is nevertheless
blocked by the unexecuted required directory/ZIP decoded-parity gate and by the
10-sweep cache-identity issue below. The remaining provenance, local-header, and
record-integrity findings should also be corrected before S07 integration.

## Preflight

The required commands were run before review:

```text
git rev-parse --show-toplevel
/home/gaohui/.codex/worktrees/3667/fl_weather_project

git rev-parse HEAD
ce2e77284b290de4c9faa6b2f971c0bd52f98eff

git branch --show-current
<empty; detached HEAD>

git status --short
<empty; clean>
```

This matched the kickoff envelope. No worktree or branch operation was performed.

## Findings, ordered by severity

### P1 — required real decoded directory/ZIP parity and dependency-complete lifecycle suite were not executed

The worker explicitly records this gate as unexecuted in
`HANDOFF.md:179` and `RESULTS.md:323-326`. The intended real-mini test is present at
`tests/test_nuscenes_zip_dataset.py:138-209` and covers raw bytes, decoded images,
writable LiDAR arrays, a scene start, and a full nine-sweep history, but there is no
GH200/dependency-complete output for it. The same missing focused execution leaves
the implemented `spawn` and dependency-backed pickle/DataLoader cases unverified;
the real jobs exercised the normal Linux fork path.

This is a mandatory S01 gate, not a residual-risk waiver. S01-R could not run it on
the login node because that interpreter lacks `pytest`, `numpy`, `torch`, and
`nuscenes-devkit`, while the validated environment is aarch64/GH200. With
`APPROVED_COMPUTE=none`, S01-R correctly did not submit a job.

Required resolution: under a new exact owner-approved request, execute the focused
real-mini parity plus fork/spawn/pickle/persistent-worker tests in the validated
dependency-complete runtime, retain the full command/log, and record its hash and
all skips/failures. A synthetic-only pass does not close this gate.

### P1 — cache identity does not bind `n_sweeps`, allowing a 10-sweep request to silently use fewer frames

`info_cache.py:339-341` gives all sweep depths the same filename; the sidecar at
`info_cache.py:355-364` omits the requested sweep count; and
`info_cache.py:379-390` reuses any self-consistent cache without comparing the
requested `n_sweeps`. The dataset check at `dataset.py:214-219` only requires that
at least one selected record contains a `lidar_sweeps` key. `_load_multisweep` then
slices whatever records happen to be present.

Consequently, a valid cache built with `n_sweeps=2` can be reused by
`get_or_build_cache(..., n_sweeps=10, rebuild=False)`. The dataset accepts it and
silently emits keyframe-plus-one-sweep samples instead of the requested ten-frame
pipeline. A mixed/malformed cache also passes if only one info has the key. The
full-gate cache itself reconciled correctly against metadata for ten sweeps, so the
published `538,695` count is not invalidated; the flaw affects fail-closed reuse in
later training/evaluation.

Required resolution: make requested sweep depth part of cache identity and sidecar
metadata, reject a load whose declared depth differs, and validate every info
record against that declaration. Add a regression that attempts to use a 2-sweep
cache for a 10-sweep dataset and must fail before reading data.

### P1 — job 332651's durable outputs do not independently bind the approved commit/source state

`RESULTS.md:205-207` calls `1fe651700bd0...` and
`64ba617e...` the exact runtime identities. The submission command in
`RUN_REQUEST.md:349-363` checks the Git state immediately before `sbatch`, but the
batch launcher at `run_s01_nuscenes_zip_full_gate.sh:31-62` neither rechecks nor
prints the commit/source-state hash after the allocation starts. The stdout begins
with host, repository, dataroot, manifest, and output only. None of the generated
JSON/checksum artifacts contains the Git or source-state identity.

The current reviewed files recompute to the reported
`64ba617eb2df8be49df89b83f691d6c91829c0cb91f85acbe665b499f5dab65c`, and the job
started two seconds after submission, so there is no evidence that the wrong code
ran. The problem is that the exact approved runtime cannot be independently proven
from the durable job record, unlike bounded smoke 330409 whose launcher checked and
printed both identities.

Required resolution: make the full-gate launcher accept expected Git/source hashes,
recompute and fail closed inside the job, and write those identities into an
execution manifest covered by `sha256sums.txt`. Do not reinterpret a documentation
assertion as an embedded runtime attestation.

### P2 — same-size local-header mutation is not detected

At archive open, `zip_backend.py:569-582` checks only archive byte size. At read,
`zip_backend.py:684-718` checks the local signature, compression value, and
encryption bit, but it does not read/compare the local filename with the canonical
central-directory path or compare the remaining relevant flags. A reviewer
synthetic check changed `samples/CAM_FRONT/a.jpg` to the same-length
`samples/CAM_FRONT/b.jpg` in the local header after manifest construction. Archive
size and payload CRC remained unchanged, and `read_bytes("samples/CAM_FRONT/a.jpg")`
succeeded. Thus the implementation does not fully support the stated malformed
local-header/archive-mutation rejection claim.

The shared artifacts showed no observed payload failure, so this does not disprove
the current full-data counts. Required resolution: parse and validate the local
filename/encoding and relevant flags against the recorded central entry, and add a
same-size header-mutation regression. If stronger whole-archive identity is part of
the intended contract, record and validate it explicitly rather than describing a
size check as general mutation detection.

### P2 — the generic per-archive sentinel gate can read a duplicated member from only the lowest shard

`manifest_archive_sentinels` at `zip_backend.py:908-917` selects `MIN(path)` for
each archive, while normal lookup at `zip_backend.py:615-635` always resolves a
duplicated path to the lowest archive ID. The audit at
`s01_nuscenes_zip_audit.py:266-282` then reads each selected path through normal
routing and checks only that all archive names are keys in the report. The existing
duplicate regression at `tests/test_nuscenes_zip_backend.py:102-118` demonstrates
the precondition: both archive sentinel entries may be the same path, while only
the first archive descriptor opens.

For job 332651 this did not create a false pass: the raw coverage artifact selected
ten unique `.v1.0-trainvalXX_blobs.txt` paths, with distinct hashes, so every real
shard was opened. The helper is nevertheless unsound for another valid v2 manifest
whose lexicographically first member is shared.

Required resolution: choose a member unique to each archive, or add an explicit
read-by-occurrence/archive-ID integrity API for audit use. Add a duplicated-minimum
sentinel regression that proves both archive descriptors are actually read.

### P2 — the bounded-smoke file hash is misrecorded

`RESULTS.md:116` records the SHA-256 of `smoke_report.json` as
`e882b490...e830`. The artifact, its job-330409 stdout checksum, and independent
`sha256sum` all give
`e882b490c8bd772c6addfdee20c2c369a2a83be7afddbf096bad9c51fbac79df`.
The recorded value appears to have inherited the ending of the stderr hash. The
JSON's internal canonical report hash at `RESULTS.md:119-120` is correct.

Required resolution: correct the durable result record and add a mechanical
`sha256sum -c` verification step for every handoff table.

## Adversarial and verification checks performed

- Read completely: repository `AGENTS.md`; all three canonical Orchestra files;
  active `docs/env.md`; read-only `collab/arrhenius_migration.md`; complete S01
  `HANDOFF.md`, `RUN_REQUEST.md`, and `RESULTS.md`; the exact worker diff; every
  changed data/cache source, script, test, and active documentation file.
- Inspected all raw stdout/stderr for jobs `330409`, `332648`, and `332651` and
  independently hashed them.
- Queried `sacct` for job state, exit, node, allocations, timestamps, elapsed,
  timelimit, MaxRSS, MaxVMSize, disk I/O, and TotalCPU. The handoff values match.
- Ran `sha256sum -c` over every job-332651 artifact listed in
  `sha256sums.txt`; all passed.
- Recomputed the internal canonical hashes of the coverage, profile, and bounded
  smoke JSON reports after removing `report_sha256`; all matched.
- Queried the 633,106,432-byte manifest read-only. It contains 2,631,093 rows,
  2,631,084 distinct paths, all ten archive IDs, and exactly one repeated path:
  ten `LICENSE` occurrences of size 25,319 and CRC `48f670e8`.
- Reconciled raw coverage fields: train `28,130`, val `6,019`, total `34,149`;
  camera `204,894`, key LiDAR `34,149`, previous sweeps `299,652`, total
  `538,695/538,695`, zero missing, zero prefix violations, disjoint train/val
  sample tokens, and cache paths identical to fresh metadata traversal.
- Confirmed all ten real sentinels are distinct archive-specific hidden text
  members and their hashes differ.
- Recalculated profile rates as measured samples divided by measured wall seconds
  and inspected every p50/p95/max record. The reported `18.94/46.81/89.08/154.36`
  first-repeat samples/s values are correct. The metric is loader blocking time at
  batch size one, not model/GPU-step data-wait percentage.
- Ran stdlib synthetic manifest checks for matching cross-shard duplicates,
  deterministic lowest-archive routing, conflicting-copy rejection,
  within-archive duplicate rejection, sentinel behavior, and same-size local-name
  mutation.
- Ran stdlib pickle/reopen and fork-after-parent-open checks. Pickled state reopened
  lazily; the fork child used an isolated PID-owned descriptor and reported one
  expected reopen. `spawn` was not locally executable through the dependency-backed
  dataset suite and remains part of finding P1.
- Ran `python3 -m py_compile` over all changed Python source/scripts/tests,
  `bash -n` on both S01 launchers, and `git diff --check`; all passed.
- Full pytest was not run: the login interpreter has Pillow only and lacks pytest,
  NumPy, Torch, and nuscenes-devkit. No login-node dependency PASS is claimed.

## Artifact and provenance verification

### Git identity

| Object | Verified value |
|---|---|
| implementation base | `f262f6bea037580065a8505008773c04fdd259f5` |
| implementation commit/tree | `011e4640d26330e2c8145fcdb56833fe19e7b67d` / `3bf7344a6afa1b381162e43581c0f550a2924f79` |
| v2 commit/tree | `1fe651700bd06a07707307c60ad4e31cc9d1e0ba` / `e8dcda9f8ee88daf10c2db7ad6629c925550e460` |
| review baseline/tree | `ce2e77284b290de4c9faa6b2f971c0bd52f98eff` / `7e22cdcb0b75f943030eafffc4e676bc711c5bf9` |
| recomputed current S01 runtime source hash | `64ba617eb2df8be49df89b83f691d6c91829c0cb91f85acbe665b499f5dab65c` |

The source hash matches the handoff, subject to finding P1 that job 332651 did not
embed/recompute it inside the allocation.

### Job 332651 primary artifacts

All listed file hashes match the raw files and `sha256sums.txt`:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| manifest SQLite | 633,106,432 | `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb` |
| coverage JSON | 18,250 | `773b8ea4513bd95363dcde0732bb9e836c7563e3d2e76c58d8b2c6a568ff579b` |
| loader profile JSON | 9,905 | `d34e7a90446dcf0bc3ec355d94ac6d984442e96583c85c0f599259faf987a108` |
| train cache | 580,140,287 | `c45e7af2937120b128e85880625e379b67e5ddecce66a32d25c156810a9329c7` |
| val cache | 117,994,540 | `52281790df9b2902002442fb7f4a02722c1118787f4cf8bac737828626582fdc` |
| train sidecar | 271 | `b81867d660e7b97f8af3461e947c48ffd8b3584484cd66eee6ee8ccf2f697a1f` |
| val sidecar | 268 | `9f668bdde4e0910d6880620d92acbf3425631589af29f4aa5553ce9b4e795114` |
| checksum list | 1,609 | `21c8780f51eac36a4867014c21bf7724573a4f4adb590b2c55c5149612da0e44` |

Coverage internal hash `cd0e298d...e50` and profile internal hash
`397bfeaf...7e` also recomputed exactly. Stdout/stderr hashes match
`5836dfe4...27e` and `8db5d05b...830`.

### Negative and bounded jobs

- Job `330409`: `COMPLETED`, `0:0`, `00:01:46`; the in-job Git/source-state
  identity is present in stdout. Manifest and checksum-list hashes match. The
  corrected smoke-report file hash is recorded in finding P2.
- Job `332648`: `FAILED`, `1:0`, `00:01:10`; stderr shows the v1
  `UNIQUE constraint failed: members.path` failure during `trainval02`. No complete
  manifest/coverage/profile/checksum exists. The failure remains visible and was
  not reinterpreted as corruption evidence or silently retried.
- Job `332651`: `COMPLETED`, `0:0`, `00:05:29`; one GH200, 32 CPUs, no array/DDP/
  model run. Batch-step values match the handoff: `MaxRSS=11048512K`,
  `MaxVMSize=71370624K`, `MaxDiskRead=21058.98M`,
  `MaxDiskWrite=6392.35M`, `TotalCPU=05:40.946`.

Observed outputs are under the approved personal `/nobackup` roots. The launcher
guards against outputs below the module dataroot, the manifest is mode `0444`, and
the backend exposes no extraction call. No shared-dataset write or extraction is
evident. The approximately 1.3-GiB output, 10.54-GiB MaxRSS, and 6,392.35-MiB
aggregate disk-write observations are one-time manifest/cache costs, not per-epoch
training costs.

## Gate-by-gate verdict

| Gate | S01-R verdict | Independent evidence / limit |
|---|---|---|
| Canonical module/table discovery | PASS | Full job resolved `NUSCENES_DATA_DIR`, `trainval/`, and 34,149 metadata samples. |
| v2 occurrence schema and duplicate policy | PASS WITH P2 HARDENING | Raw manifest proves all occurrences retained; only ten identical `LICENSE` rows; lowest-ID runtime routing, conflict rejection, and within-archive rejection passed synthetic checks. Generic sentinel routing still needs correction. |
| Local offsets/CRC/archive mutation/canonical paths | CHANGES-REQUESTED | Payload CRC, size mutation, traversal, compression, and encryption checks exist. Same-size local-name mutation is accepted. |
| Official train/val path coverage | PASS | `538,695/538,695`, zero missing; six cameras, key LiDAR, requested previous sweeps; cache/metadata equality and sensor prefixes pass. |
| Train/val reconciliation/disjointness | PASS FOR S01 SCOPE | Official sample counts, token disjointness/union, and fresh metadata traversal pass. This is not a Protocol-B ownership/split audit. |
| Ten real archive payload sentinels | PASS FOR THIS ARTIFACT | Ten distinct archive-specific members and hashes. The generic sentinel helper needs the P2 fix. |
| Directory-mode preservation | PASS (static/synthetic) | Directory backend remains and synthetic byte/decoded tests are implemented. |
| Real directory/ZIP byte and decoded-array parity | **NOT RUN / BLOCKING** | Required dependency-complete real-mini test has no execution evidence. |
| Writable LiDAR semantics | PASS (implementation/synthetic test present), execution pending | ZIP decode uses `np.frombuffer(...).copy()`; test asserts writable array, but focused dependency suite still must execute. |
| Keyframe plus ten-sweep runtime contract | CHANGES-REQUESTED | Job 332651 cache paths/counts are correct; cache identity can silently reuse a lower-sweep cache. |
| Fork/pickle handle isolation | PASS (bounded real + reviewer stdlib) | Real persistent fork workers and reviewer fork/pickle checks passed. |
| Spawn lifecycle | NOT RUN | Implemented test exists; dependency-complete execution remains required. |
| 0/2/4/8-worker repeated determinism | PASS | One decoded digest across every worker count and both repeats. |
| Full-data loader profile | PASS WITH SCOPE LIMIT | Rates and wait distributions are correct loader-only batch-size-one measurements. |
| End-to-end model/GPU data wait | NOT CLAIMED / NOT ESTABLISHED | No model step or GPU utilization percentage was measured. |
| No extraction/shared-data writes | PASS FOR OBSERVED EXECUTION | External paths, read-only manifest, guarded launcher, and no extraction API; no contrary artifact/log evidence. |
| Permission/resource compliance | PASS | Three jobs and resources match their recorded approvals; no automatic resubmission or scope expansion observed. |
| Runtime provenance and artifact hashes | CHANGES-REQUESTED | Full artifacts/logs validate, but full job lacks in-job Git/source binding; one bounded-smoke hash is misrecorded. |
| Overall S01 integration readiness | **FAIL PENDING CHANGES** | Blocking parity evidence and cache-identity fix are outstanding; independent re-review is required. |

## Allowed interpretations

- The v2 manifest records 2,631,093 occurrences and 2,631,084 unique paths across
  ten declared trainval archives; the nine extra occurrences are the same
  size/CRC `LICENSE` entry.
- Every official train/val camera, key `LIDAR_TOP`, and requested previous-sweep
  path generated by the audited 10-sweep metadata traversal is represented in the
  manifest: `538,695/538,695`, zero missing.
- The exact job-332651 cache agrees path-for-path with that traversal and has the
  reported train/val counts and content hashes.
- Ten distinct real archive-specific sentinel payloads passed the production
  length/CRC path in job 332651.
- The audited 304-token sequence produced one decoded digest across 0/2/4/8
  workers and both repeats.
- The reported timing table is a valid batch-size-one loader-only measurement for
  node `n574` and the recorded runtime. It can inform S07 profiling design.
- Job 332648 is a preserved negative result showing the v1 global-uniqueness
  assumption was false; it is not evidence of corrupt sensor data.

## Forbidden interpretations

- S01 PASS, S07 integration readiness, or production training readiness before the
  requested changes and an accepted re-review.
- Executed real directory/ZIP byte, image, or LiDAR parity.
- Executed dependency-complete spawn behavior.
- A general claim that any cache labeled/requested as ten-sweep must contain ten
  requested frames; current cache identity does not enforce that.
- A general claim that all same-size archive/local-header mutations are rejected.
- A general claim that `manifest_archive_sentinels` necessarily reads each archive
  occurrence when the selected path is duplicated.
- A claim that every one of 2.63 million payloads was opened and CRC-checked.
- End-to-end model-step data-wait percentage, epoch-scale/multi-job contention,
  absence of random-read amplification, or GPU utilization conclusions.
- Model accuracy, training stability, mAP/NDS, FL, attack/defense, generalization,
  or publication claims from these engineering jobs.
- Use of mini/smoke evidence as scientific evidence.

## Residual risks after the requested fixes

- The loader profile covers 2,432 sample reads, not a full epoch or concurrent-job
  shared-filesystem contention. S07 still needs model-integrated timing.
- SQLite manifest and cache construction consume material CPU/RSS/disk I/O and
  must be reused, frozen, and hashed rather than rebuilt per run.
- CRC32 is an integrity/error-detection mechanism, not a cryptographic content
  identity. Later provenance should retain artifact SHA-256 and immutable dataset
  module/version identity.
- The location cache, SQLite connection, and archive descriptors were boundedly
  exercised; long-running worker failure/recovery and filesystem outage behavior
  remain operational risks.
- S01 path disjointness is official train/val reference reconciliation only. It
  does not replace the future scene/log/raw-sensor ownership audit required for
  Protocol B.

## Required return package for re-review

1. A cache-schema/loader fix that binds and validates `n_sweeps`, with hostile
   lower-sweep and mixed-cache regressions.
2. Owner-authorized dependency-complete output for the real-mini directory/ZIP
   bytes/decoded images/writable keyframe+10-sweep LiDAR test and the focused
   fork/spawn/pickle/persistent-worker suite.
3. Full-gate in-job Git/source-state attestation written into the checksummed
   execution manifest for future runs; document whether existing job 332651 is
   accepted as historical coverage evidence rather than rerunning it automatically.
4. Local-header filename/mutation and duplicated-sentinel hardening regressions.
5. Corrected bounded-smoke SHA-256 in `RESULTS.md`.

No rerun, Slurm submission, commit, merge, push, upload, or publication is
authorized by this review.

---

## S01-R re-review amendment — NEW_WORKER_SHA `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`

### Re-review identity, history, and latest verdict

- S00/owner authorized this re-review in the existing S01-R task without changing
  its worktree or authorizing reviewer compute.
- The review worktree remained detached at the original review baseline
  `ce2e77284b290de4c9faa6b2f971c0bd52f98eff`. The original uncommitted
  `REVIEW.md` was the only working-tree change before this amendment.
- Original reviewed worker SHA: `ce2e77284b290de4c9faa6b2f971c0bd52f98eff`.
- Original review artifact before amendment: SHA-256
  `f69de33eec31e6d9e64c86f1fc30d3d76e17a1e482e65113b2c3ed5174551357`.
- Original verdict: **CHANGES-REQUESTED**. The six original findings and their
  rationale above remain part of the durable review history.
- Remediation implementation SHA:
  `54a48f9102fd0de9a9abe97701550740b547e769`, tree
  `7341ad9d7e489572b979e1e032b41590e0c80e91`.
- New durable worker SHA:
  `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`, tree
  `0c07c7dfcf33a68d2437250686b3ef67be29330a`.
- Re-review diff:
  `ce2e77284b290de4c9faa6b2f971c0bd52f98eff..abe5c58b174dbbe1f7045ce91c8b15168d97b87b`;
  the full base diff `f262f6b..abe5c58` was also reconciled against the original
  review.
- Reviewer `APPROVED_COMPUTE`: none. S01-R submitted no job.
- **Latest re-review verdict: PASS.** This supersedes only the original final
  verdict; it does not erase or rewrite the original findings.

All six requested remediations are implemented and supported by the declared
focused GH200 evidence. PASS means the S01 ZIP-backend worker result may be
accepted as a reviewed dependency. It does **not** mean that the current `t1.v2`
code already has a full-trainval cache, that job 332651 gained retroactive source
attestation, or that S07/full training is ready without the integration work listed
below.

### Disposition of the six original findings

| Original finding | Re-review disposition | Exact remediation and evidence |
|---|---|---|
| P1: dependency-complete real-mini parity and lifecycle suite not executed | **CLOSED** | Job 333206 JUnit contains the real-mini directory/ZIP parity case, parent-open fork and spawn cases, and repeated persistent-worker fork and spawn cases. All ran; 56/56 tests passed with zero skips. The parity test compares raw bytes, six-camera tensors, writable key/10-sweep LiDAR tensors, and GT for a scene start and a sample with nine previous sweeps. |
| P1: cache identity did not bind `n_sweeps` | **CLOSED** | At `abe5c58:info_cache.py:42,223-267,290-322,353-483`, cache format `t1.v2` binds depth in the format/filename, every record, canonical hash, pickle metadata, and sidecar; load/save reject missing, mixed, ambiguous, or mismatched depth. `abe5c58:dataset.py:214-232` revalidates every selected record against runtime depth. JUnit executed the 2-versus-10 rejection and ambiguity/sidecar-drift regressions. |
| P1: future full-gate launcher lacked in-job Git/source attestation | **CLOSED FOR FUTURE JOBS; HISTORICAL LIMIT PRESERVED** | `abe5c58:run_s01_nuscenes_zip_full_gate.sh:24-57,74-112` now requires and verifies expected Git/source identities inside the job and writes checksummed identity/source-list artifacts. Job 333206 used the equivalent focused launcher attestation. No new claim is attached to job 332651; it remains historical coverage evidence without in-job source attestation. |
| P2: same-size local-header filename mutation accepted | **CLOSED** | `abe5c58:zip_backend.py:684-735` now compares the complete relevant flags and decodes/compares the local filename to the manifest path. The exact hostile same-length `a.jpg`→`b.jpg` mutation test ran and passed in job 333206. Reviewer repeated this check against the immutable remediation object and observed fail-closed rejection. |
| P2: duplicate sentinel could route only to the lowest archive | **CLOSED** | `abe5c58:zip_backend.py:830-869` adds an exact `(archive,path)` occurrence read, and `abe5c58:s01_nuscenes_zip_audit.py` uses it for sentinels. The duplicate-`LICENSE` regression proves normal routing opens only the lowest archive while exact-occurrence reads open both declared archives. It passed in job 333206 and reviewer stdlib reproduction. |
| P2: bounded-smoke hash misrecorded | **CLOSED** | `abe5c58:RESULTS.md:113-120` records the actual file SHA-256 `e882b490c8bd772c6addfdee20c2c369a2a83be7afddbf096bad9c51fbac79df`; it matches the original artifact and job-330409 stdout checksum. |

### New blocking findings

**None.** The re-review found no new issue that invalidates the S01 backend or the
declared focused evidence.

### Non-blocking provenance hardening observation

The focused-test source-state list covers every regular file under
`src/fl_v3/data/nuscenes`, the four selected test modules, and the test launcher,
but not `fl_v3/tests/conftest.py`, even though pytest loads that file and its
`nusc_mini`/`dataroot` fixtures select the real-mini input. The list also does not
include project test configuration/dependency metadata. The 18 listed files all
independently match immutable commit `54a48f9`; the execution recorded the exact
Git SHA; JUnit proves the mini-dependent tests ran without skip; and the currently
preserved worker copy of `conftest.py` matches the commit object. There is no
positive evidence of fixture drift, so this does not overturn job 333206 or S01
PASS. Future focused-test launchers should nevertheless include `conftest.py` (and
any effective pytest configuration) in the in-job source-state list so working-tree
fixture changes cannot escape the hash.

## Job 333206 independent artifact and hash verification

### Scheduler and execution identity

Independent `sacct` output matches the handoff:

- job/name/node: `333206` / `flv3_s01_zip_tests` / `n405` (`aarch64`);
- state/exit: `COMPLETED`, `0:0`;
- submit/start/end: `2026-07-11T08:32:21` / `08:32:22` / `08:33:49`;
- elapsed/timelimit: `00:01:27` / `00:20:00`;
- allocation: one GH200, eight CPUs, one node;
- batch resources: `MaxRSS=540M`, `MaxVMSize=6279744K`,
  `MaxDiskRead=60.65M`, `MaxDiskWrite=0.34M`, `TotalCPU=00:19.221`.

`execution_identity.json` records:

```text
git_sha = 54a48f9102fd0de9a9abe97701550740b547e769
runtime_source_sha256 = 260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda
slurm_job_id = 333206
host = n405
machine = aarch64
mini_dataroot = /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
```

The execution identity agrees with stdout and the approved request. Every entry in
`runtime_source_sha256s.txt` was independently hashed from immutable commit
`54a48f9`; all 18 entries match, and the list file's own SHA-256 equals the declared
runtime source-state hash.

### Pytest/JUnit evidence

- `pytest.log`: `56 passed in 13.18s`.
- JUnit suite: tests `56`, failures `0`, errors `0`, skipped `0`, time `13.155`.
- JUnit contains both `fork` and `spawn` variants for parent-open child ownership
  and repeated persistent-worker determinism; neither was skipped.
- JUnit contains the real-mini decoded-parity, 2-versus-10 cache-depth,
  local-header mutation, and exact duplicate-sentinel cases.

### Artifact hashes

All hashes match the files, the handoff table, stdout, and `sha256sums.txt`;
`sha256sum -c` passed for every checksummed artifact:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 442 | `a41970d8b84c575bfb90bde0f0e65d6801b5ef615068d13303addfd0998ef865` |
| `runtime_source_sha256s.txt` | 1,950 | `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda` |
| `pytest.log` | 100 | `641ff631b532d33d50cdb8805d2ec88df1ed70e96f11a789e21a09218238ac3e` |
| `pytest.junit.xml` | 7,465 | `38b199b092bdcaecd49a2fde475a0aff276b9a344652aadd4f524e3d84fcd1bc` |
| `sha256sums.txt` | 787 | `e60dba68c7065a83c84bd6c5eeb02a535042e9416111cbdf672aa158ce5bf83a` |
| stdout | 1,292 | `bc1547d4b400679915f1a022a025afc9e48ad1af6c7efc908e6d80d68e989544` |
| stderr | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

Stderr contains only the normal module-purge notice. The output root and mini input
match the approved request. No shared trainval archive, model, training loop,
evaluation, matrix, seed expansion, retry, or automatic resubmission was involved.

## Re-review adversarial and static checks

- Read the immutable remediation/result diff using `git diff`/`git show`; HEAD was
  never switched to the new worker object.
- Reconciled the remediation against all six original findings and the full
  original implementation diff.
- Parsed all 56 JUnit testcase names and confirmed both process-start methods and
  every requested hostile regression actually executed.
- Recomputed every artifact/log hash and ran the checksum list in verification
  mode.
- Recomputed every listed runtime source-file hash from commit `54a48f9`; no listed
  file differs from the immutable object.
- Independently reproduced, from a temporary `git archive` of `54a48f9`, rejection
  of same-length local-header filename mutation and exact-archive reading of a
  duplicated sentinel. Normal duplicate routing still correctly selected the
  lowest archive.
- Ran `py_compile` on the remediated data/cache/audit modules, `bash -n` on both
  new/remediated launchers, and `git diff --check`; all passed.
- Audited every active `t1.v1`/`load_cache` reference in the new worker tree. API
  callers fail closed when cache depth is absent/ambiguous or is rejected by
  dataset construction. `build_gt_database.py` is the one active direct filename
  bypass and is classified below.

## Residual integration request: `build_gt_database.py`

`abe5c58:fl_v3/scripts/build_gt_database.py:35-39` directly opens
`nuscenes_info_<version>_<split>_t1.v1.pkl`, bypassing the remediated
`info_cache.load_cache` format, sidecar, hash, and `n_sweeps` validation. It cannot
consume a `t1.v2` cache and could continue consuming a historical unbound cache if
one is left in place.

Classification:

- **Not an S01 PASS blocker.** It is outside the authorized S01 remediation scope,
  does not participate in the normal dataset/training cache path reviewed here,
  and none of the S01 manifest/coverage/parity/lifecycle gates depends on GT
  database creation. The worker correctly surfaced it rather than editing outside
  scope.
- **S07 integration blocker.** Before the integrated candidate is declared ready,
  the script must use `IC.load_cache(..., n_sweeps=args.n_sweeps)` (or an equally
  explicit depth-specific `t1.v2` API) and inherit sidecar/hash/depth rejection.
  This is mandatory before producing or trusting a GT database for any recipe with
  GT paste; retaining an active hardcoded `t1.v1` data-preparation path would defeat
  the fail-closed cache contract just reviewed.
- A full `t1.v2` train/val cache and its hashes must also be materialized/frozen
  under an owner-approved S07 request before model training. Historical job 332651
  proves full path coverage for the unchanged metadata/ZIP routing, while job
  333206 proves the new cache-depth semantics on real mini. Neither is a new
  full-trainval `t1.v2` execution.

No automatic rerun or scope expansion is authorized by this classification.

## Updated gate-by-gate verdict

| Gate | Latest S01-R verdict | Evidence / boundary |
|---|---|---|
| v2 manifest/duplicate policy | PASS | Historical full manifest and original review remain accepted. |
| Full train/val referenced-member coverage | PASS (historical current routing evidence) | Job 332651: `538,695/538,695`, zero missing. Cache files are historical `t1.v1`, not production inputs. |
| Real directory/ZIP decoded parity | PASS AT REAL-MINI ENGINEERING SCALE | Job 333206 executed bytes/images/writable key+10-sweep LiDAR parity for scene-start/full-history samples. No trainval-scale parity claim. |
| Fork/spawn/pickle/persistent lifecycle | PASS | Job 333206 executed both fork and spawn with zero skip; earlier real fork evidence remains valid. |
| Cache identity and 2-vs-10 rejection | PASS | `t1.v2` code and executed hostile regressions bind/reject depth fail closed. |
| Local-header mutation rejection | PASS | Code review, JUnit case, and reviewer reproduction. |
| Exact-archive duplicate sentinel | PASS | Code review, JUnit case, and reviewer reproduction. |
| Future in-job Git/source attestation | PASS (implementation) | Full-gate and focused-test launchers now record it. Job 332651 remains explicitly unattested. |
| Job 330409 record correction | PASS | Correct file hash now matches raw artifact. |
| Job 333206 authorization/artifacts | PASS | Exact approved commit/resources/output; all hashes and scheduler fields verified. |
| No extraction/shared-data writes | PASS FOR EXECUTED SCOPE | Mini source was read-only; ZIP fixtures were temporary; outputs were external. |
| Full-data loader-only profile | PASS WITH ORIGINAL LIMIT | Job 332651 timing remains loader-only, batch size one. |
| `build_gt_database.py` current-format compatibility | FAIL PENDING S07 | Not an S01 acceptance gate; must be migrated before integrated GT database/GT-paste use. |
| Full `t1.v2` trainval cache and model-integrated data path | NOT YET EXECUTED | Required downstream integration evidence, not supplied or claimed by S01. |
| Overall S01 worker result | **PASS** | Six original findings closed; residual work is explicitly assigned to integration and does not broaden S01 claims. |

## Updated allowed interpretations

- All allowed interpretations from the original review remain valid for historical
  jobs 330409/332651 within their exact scopes.
- On remediation commit `54a48f9`, the declared real-mini directory/ZIP byte and
  decoded-array parity, writable LiDAR, fork/spawn lifecycle, cache-depth,
  local-header, and exact-sentinel regressions passed on GH200.
- `t1.v2` prevents a 2-sweep cache or mixed-depth records from silently satisfying
  a 10-sweep request through the reviewed cache/dataset APIs.
- Future invocations of the remediated full-gate launcher have an in-job
  Git/source-attestation path, provided a new exact request is separately approved.
- S01 may be accepted as a reviewed dependency for S07 integration work.

## Updated forbidden interpretations

- Job 333206 does not establish trainval-scale directory/ZIP decoded parity.
- Job 333206 does not create or validate a full trainval `t1.v2` cache.
- Job 332651 does not gain retroactive in-job source attestation, and its `t1.v1`
  caches are not valid production inputs to the remediated loader.
- No result reads/CRC-checks every shared payload.
- No result measures model-step data wait, full-epoch contention, GPU utilization,
  or production training readiness.
- S01 PASS does not waive the `build_gt_database.py` migration or the S07 current-
  format full-data integration gate.
- Mini, synthetic, coverage, or loader evidence supports no model-quality, mAP/NDS,
  FL, attack/defense, generalization, scientific, or publication claim.

## Latest final verdict

**PASS** for S01 independent implementation/scientific-engineering review at
`NEW_WORKER_SHA=abe5c58b174dbbe1f7045ce91c8b15168d97b87b`, subject to the explicit
interpretation limits and downstream S07 blockers above.

No Slurm submission, commit, merge, push, upload, publication, branch switch, or
worktree operation was performed by S01-R during re-review. `REVIEW.md` remains the
reviewer's only modified file and is intentionally uncommitted.
