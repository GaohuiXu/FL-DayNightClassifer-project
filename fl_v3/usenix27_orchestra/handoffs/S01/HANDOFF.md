# S01 HANDOFF — shared nuScenes ZIP backend

## Session identity and verdict

- Session: `S01`.
- Base/HEAD: `f262f6bea037580065a8505008773c04fdd259f5`.
- Source branch named by kickoff: `v3-ad-perception`.
- Kickoff ref was detached HEAD at the exact base, with an empty branch name as
  expected. On 2026-07-11 the owner explicitly authorized a local review branch
  and commit.
- Current local branch: `codex/s01-nuscenes-zip-backend`.
- Implementation commit: `011e4640d26330e2c8145fcdb56833fe19e7b67d`
  (`feat(data): add nuScenes ZIP backend`). This commit was created locally only;
  no merge, push, upload, or publication was performed.
- v2 correction commit: `1fe651700bd06a07707307c60ad4e31cc9d1e0ba`
  (`fix(data): support identical ZIP members across shards`). It records failed
  job `332648`, retains matching cross-archive occurrences, and rejects conflicting
  copies. This is the immutable follow-up run candidate, not a reviewed PASS.
- S01-R reviewed baseline `ce2e77284b290de4c9faa6b2f971c0bd52f98eff` and
  returned **CHANGES-REQUESTED**. The independent `REVIEW.md` artifact SHA-256 is
  `f69de33eec31e6d9e64c86f1fc30d3d76e17a1e482e65113b2c3ed5174551357`;
  it remains in the separate review worktree and is not modified by S01.
- Remediation implementation commit:
  `54a48f9102fd0de9a9abe97701550740b547e769`. Its exact approved focused-test
  identity and execution record are in `RUN_REQUEST.md`.
- Working-tree scope: only S01-owned files listed below.
- Worker verdict: **FULL-DATA ZIP GATES PASS / S01-R CHANGES REQUESTED / REMEDIATION
  FOCUSED TESTS PASS / RE-REVIEW PENDING**. The v2 ten-archive manifest, 100%
  train/val reference coverage, real
  archive sentinels, deterministic 0/2/4/8-worker reads, and loader throughput gate
  passed in job `332651` and are not invalidated by the review. Dependency-backed
  real-mini directory/ZIP decoded parity, fork/spawn lifecycle, cache-depth, and
  integrity regressions all passed in focused job `333206` (`56 passed`, no skip).

## Active-session amendment acknowledgement

Received and followed the S00/owner amendment:

1. continued at `xhigh`; no Ultra escalation;
2. used the explicit owner approval to submit one bounded ZIP-reader Slurm smoke;
3. recorded HEAD, working-state hashes, exact command/subset/resources/walltime/
   output/stop conditions in `RUN_REQUEST.md` before submission;
4. cited owner-specific approval plus standing smoke policy `O-009` and stayed
   inside one node, one GPU, 20 minutes requested, one concurrent job, no array,
   DDP, multi-seed, full scan/profile, epoch/eval, matrix, or auto-resubmit;
5. recorded job ID, command, logs/artifacts, exit state, and negative results in
   `RESULTS.md`. The smoke is engineering evidence only.

The owner later gave a separate exact approval for remediation commit
`54a48f9102fd0de9a9abe97701550740b547e769`, one GH200, eight CPUs, walltime
`00:20:00`, maximum 0.333 GPU-hours, the declared unique output, and no automatic
resubmission. Focused job `333206` stayed inside that scope and completed once.

## Implementation summary

The backend keeps directory mode as the default when ordinary payload files exist.
For the Arrhenius stored-ZIP layout it uses an external, immutable SQLite
member-to-archive manifest. Every process lazily owns its manifest connection and
archive file descriptors; fork/pickle/PID transitions close inherited state and
reopen locally. Reads use recorded central-directory metadata, lazily parse the
requested member's local header, perform bounded `os.pread`, and verify size and
CRC. One safe reopen/retry is permitted for a stale/bad descriptor.

This follows C3SE's official raw-ZIP DataLoader guidance: do not hold a constructor
archive handle across worker creation, open lazily per worker, use `BytesIO` for
image decode, and recover from a bad handle. S01 additionally avoids constructing a
full Python `ZipFile` name dictionary in every worker by materializing the central
directory once outside the shared dataset.

Key semantics:

- dataroot precedence is explicit config, `NUSCENES_DATAROOT`,
  `ARRHENIUS_NUSCENES_DATAROOT`, then module `NUSCENES_DATA_DIR`;
- module metadata aliases official `v1.0-trainval` to physical `trainval/` while
  retaining the official devkit version;
- manifest paths come from explicit config or `NUSCENES_ZIP_MANIFEST` /
  `ARRHENIUS_NUSCENES_ZIP_MANIFEST` and must be outside the read-only dataset;
- manifest construction fails on unsafe/noncanonical names, duplicate members
  within an archive, cross-archive copies whose size/CRC conflict, encryption,
  compression, malformed headers, archive mutation, or a missing exact archive
  set. Identical cross-archive occurrences remain auditable and route to the
  lowest-numbered archive;
- each payload read verifies that the local-header member name and relevant flags
  still match the manifest, including same-length filename mutation; exact archive
  occurrences can be read directly for duplicate-path sentinel coverage;
- the unified blob store supplies bytes to six-camera, keyframe LiDAR, and all
  requested multi-sweep paths; image decode uses `PIL.Image` over `BytesIO`, LiDAR
  uses `numpy.frombuffer(...).copy()`;
- info-cache construction uses metadata `sample_data.filename`, so it does not
  extract or probe every payload; cache format `t1.v2` binds `n_sweeps` in the
  filename, sidecar metadata, every record, and canonical hash, and rejects
  ambiguous or mismatched depth. Historical full-gate `t1.v1` caches remain
  evidence only; 10-sweep caches retain an explicit empty sweep list at scene starts;
- directory mini/local operation does not accidentally inherit a module ZIP
  manifest unless explicitly configured.

## Files and semantic changes

Modified:

- `AGENTS.md` — replaces the stale “loader is not ZIP-aware” statement with the
  implemented-but-not-fully-gated status and environment contract.
- `fl_v3/docs/env.md` — documents module discovery, manifest/cache workflow,
  official C3SE lifecycle mapping, bounded job `330409`, and remaining gates.
- `fl_v3/scripts/build_nuscenes_cache.py` — module-aware verification/devkit
  construction and extraction-free metadata cache generation.
- `fl_v3/src/fl_v3/data/nuscenes/{__init__.py,conventions.md,dataset.py,info_cache.py,paths.py}`
  — exports/integrates the blob store, byte decoders, multi-sweep reads, module
  aliases, safe writable paths, and cache schema behavior.

Added:

- `fl_v3/src/fl_v3/data/nuscenes/zip_backend.py` — strict manifest builder and
  PID-owned directory/ZIP blob store.
- `fl_v3/scripts/s01_nuscenes_zip_manifest.py` — deterministic manifest CLI.
- `fl_v3/scripts/s01_nuscenes_zip_audit.py` — full train/val cache reconciliation,
  six-camera/key/sweep coverage, disjointness, and ten-archive sentinel gate.
- `fl_v3/scripts/s01_nuscenes_zip_benchmark.py` — deterministic decoded-batch and
  loader wait/throughput profiler.
- `fl_v3/scripts/s01_nuscenes_zip_smoke.py` and
  `run_s01_nuscenes_zip_smoke.sh` — exact bounded real-data lifecycle smoke.
- `fl_v3/scripts/run_s01_nuscenes_zip_full_gate.sh` — full gate launcher; approved
  v1 attempt `332648` failed and was not retried, then exact v2 attempt `332651`
  completed all declared stages. The remediated launcher verifies the expected Git
  SHA and runtime source-state hash in-job and writes an execution identity record.
- `fl_v3/scripts/run_s01_nuscenes_zip_tests.sh` — bounded GH200 focused-test
  launcher for dependency-backed mini parity, lifecycle, integrity, and cache-depth
  regressions; exact approved job `333206` completed successfully.
- `fl_v3/tests/test_nuscenes_zip_{backend,dataset,info_cache}.py` — stored-ZIP
  safety/integrity, directory/ZIP byte+decoded parity, keyframe/10-sweep routing,
  fork/spawn/persistent lifecycle, module layout, and no-payload-probe cache tests.
- `fl_v3/usenix27_orchestra/handoffs/S01/{RUN_REQUEST,RESULTS,HANDOFF}.md` — compute
  authorization, exact result, and this handoff.

No canonical Orchestra file, `fl_v3/collab/`, `fl_v2/`, model/trainer/config,
detector/eval, or shared dataset file was edited.

## References

- C3SE official machine-learning dataset documentation, “Raw files”:
  <https://www.c3se.chalmers.se/documentation/software/machine_learning/datasets/#raw-files>.
- Repository active contracts: `AGENTS.md`, `fl_v3/docs/env.md`, and the complete
  S01 sections of `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md` (read-only except
  the two S01-owned active docs above).
- Historical bring-up evidence:
  `fl_v3/collab/arrhenius_migration.md` (read-only, not modified).

## Verification evidence

Local/login-node checks:

- kickoff identity commands: top-level path matched this worktree; HEAD matched the
  base; branch was empty/detached; initial status was clean;
- `python3 -m py_compile` over every changed/added Python source, script, and test:
  pass;
- `bash -n fl_v3/scripts/run_s01_nuscenes_zip_{smoke,full_gate}.sh`: pass;
- `git diff --check`: pass;
- inline stdlib synthetic checks: ten synthetic stored archives built/indexed/read,
  deterministic manifest/coverage and explicit close/reopen passed; duplicate,
  compressed, unsafe-name, and CRC-corrupt inputs were rejected; module alias/path
  escape checks passed. The repeated ten-archive synthetic manifest hash was
  `61a9f0493e3a4aee67e2628fea3c4b9b939f6f31025475c79ce7055cf5f4c404`;
- full dependency-backed pytest was not runnable on the x86_64 login node because
  the validated environment is aarch64/GH200. The login interpreter had Pillow
  10.0.1 but lacked pytest, numpy, torch, and nuscenes-devkit. No false login-node
  PASS is claimed.

The v2 duplicate-occurrence regression built ten synthetic archives with identical
cross-archive copies, verified deterministic first-archive routing and per-archive
sentinels, and rejected both within-archive duplicates and cross-archive size/CRC
conflicts. Synthetic v2 manifest hash:
`b28081730286a56d148a44ec684dc0be4689ad5d0668a462c4c918e78b22d0a7`.

Approved real-data smoke: `RESULTS.md` contains the complete evidence for Slurm job
`330409` (`COMPLETED`, `00:01:46`, exit `0:0`). It passed module discovery, one
archive's 258,109-member manifest, 64 selected references spanning 24 camera + 4
key LiDAR + 36 previous sweeps, decoded output, CRC, 0/2-worker repeated hashes, and
persistent PID/handle lifecycle.

Approved full-data evidence is in `RESULTS.md`:

- job `332648` failed at the v1 cross-archive path uniqueness assumption and was
  preserved as a negative result without retry;
- corrected commit `1fe651700bd06a07707307c60ad4e31cc9d1e0ba`, job `332651`,
  `COMPLETED`, `00:05:29`, exit `0:0`;
- ten archives indexed, 538,695/538,695 pipeline references resolved, ten payload
  sentinels passed, and every 0/2/4/8-worker decoded digest matched.

## Gate-by-gate status

| S01 gate | Status | Evidence / missing work |
|---|---|---|
| Canonical module and `NUSCENES_DATA_DIR` | PASS (full real) | job 332651 resolved module root and `trainval/` tables, 34,149 sample metadata |
| Preserve directory mode | PASS (implementation/static) | directory store retained; existing dataset paths remain supported |
| Manifest trainval01..10 | PASS (full real) | 2,631,093 occurrences, 2,631,084 unique; only repeated path is identical `LICENSE` in all ten archives |
| Worker-safe lazy handles/reopen | PASS (bounded real + focused mini) | job 330409 persistent PIDs plus job 333206 parent-open fork/spawn and repeated persistent-worker lifecycle checks |
| Six cameras/key LiDAR/requested sweeps use byte readers | PASS (full real) | 2,432 decoded 10-sweep sample reads in profile; all reference classes covered |
| Info cache without extraction | HISTORICAL FULL PASS; t1.v2 FOCUSED PASS | job 332651 built t1.v1 train/val caches without extraction; job 333206 verified t1.v2 depth binding/rejection on mini |
| Directory/ZIP byte and decoded-array parity | PASS (real mini) | job 333206 compared two deterministic real mini samples including scene start and full 10-sweep history |
| 100% train/val referenced-member coverage | PASS | 538,695/538,695 resolved, 0 missing, cache/metadata identical, ten archive sentinels |
| Deterministic repeated multi-worker reads | PASS (full real + focused fork/spawn) | one digest across 0/2/4/8 workers in job 332651; persistent fork/spawn repeats passed in job 333206 |
| No extraction/shared writes | PASS for executed scope | external output roots and fail-closed path guard; full manifest mode 0444 |
| Full-data throughput/data-wait | PASS (loader-only) | 18.94 to 154.36 samples/s first-repeat range; complete p50/p95 in RESULTS |
| Independent S01-R | CHANGES-REQUESTED; RE-REVIEW PENDING | review of `ce2e77284b29`; six findings addressed, job 333206 passed, new durable baseline still needs re-review |

## Coverage counts and hashes

Full v2 gate counts:

- 10/10 archives, 417,774,430,886 bytes, 2,631,093 occurrences and 2,631,084
  unique members;
- nine duplicate occurrences are the same `LICENSE` path in all ten archives,
  matching size 25,319 and CRC `48f670e8`; no sensor path duplicate or conflict;
- train 28,130 + val 6,019 = 34,149 samples, with 204,894 camera, 34,149 key LiDAR,
  and 299,652 previous-sweep references;
- 538,695/538,695 total references resolved to 534,532 unique payloads, zero
  missing; train/val disjointness and cache/metadata reconciliation passed;
- 2,432 decoded profile sample reads; deterministic digest
  `4e46534f92c7979c04667a72f8a6dd0b9c61bfe0a14808b5debb85c34e0b54f7`
  matched every 0/2/4/8-worker repeat.

V2 runtime source-state hash:
`64ba617eb2df8be49df89b83f691d6c91829c0cb91f85acbe665b499f5dab65c`.
Exact request identity and artifact/log hashes are in `RUN_REQUEST.md` and
`RESULTS.md`; notably:

- logical manifest:
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`;
- manifest file:
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`;
- coverage file:
  `773b8ea4513bd95363dcde0732bb9e836c7563e3d2e76c58d8b2c6a568ff579b`;
- profile file:
  `d34e7a90446dcf0bc3ec355d94ac6d984442e96583c85c0f599259faf987a108`.

## Negative results and residual risks

- Login-node data access lacked the gated group; the GH200 job had access. The
  architecture-specific environment also prevents treating login imports as a
  definitive runtime test.
- Full job `MaxRSS` was about 10.54 GiB and aggregate `MaxDiskWrite` was 6392.35M;
  later training jobs must provision storage for the 633-MB manifest plus roughly
  698-MB train/val cache and should not rebuild them per run.
- Central-directory indexing is one-time. The loader profile measures bounded
  random access but not end-to-end model-step data wait or long-epoch filesystem
  contention from multiple simultaneous jobs.
- Job 332648 remains a required negative result: v1 incorrectly assumed global
  path uniqueness. Job 332651 verified that the repeated path was only identical
  `LICENSE`, not a duplicated sensor payload.
- S01-R found that cache format `t1.v1` did not bind `n_sweeps`, local-header
  same-length filename mutation was not detected, duplicate sentinels could route
  to the canonical archive, and job `332651` lacked in-job source attestation. The
  candidate addresses each issue; historical job artifacts are not retroactively
  attested, and its `t1.v1` cache files must not be production inputs.
- Permission/output-directory callers outside the owned data path have not been
  changed.
- Integration request to S00: `fl_v3/scripts/build_gt_database.py` is outside S01
  ownership and still hardcodes a `t1.v1` cache filename. It must be migrated to
  the `info_cache` API or an explicit `t1.v2` depth-specific path before that tool
  can consume remediated caches. Other permission-out callers that omit
  `n_sweeps` remain fail-closed: they load only when exactly one depth exists, and
  dataset construction rejects a config/cache depth mismatch.
- Focused job `333206` closed the previously unexecuted real-mini parity and spawn
  behavior gap. It does not replace an end-to-end trainval training-loader run.
- Manifest and cache provenance must be frozen and hashed in every later resolved
  run; a manifest for only one archive must fail a full-data job.

## Interpretation limits

Allowed claims:

- every official train/val path requested by the 10-sweep pipeline exists in the
  shared archives, and each of the ten archives serves CRC-checked real bytes;
- repeated decoded hashes were deterministic across 0/2/4/8 workers and repeats;
- the loader-only throughput/wait measurements in `RESULTS.md` apply to this exact
  GH200 runtime and batch-size-1 profile;
- directory-mode support remains present in the implementation.

Forbidden claims:

- final S01 integration PASS or production training readiness before S01-R;
- every one of 2.63 million payloads was read and CRC-checked;
- trainval-scale directory/ZIP decoded parity (job 333206 proves real mini only);
- end-to-end model-step data-wait percentage, long-epoch/multi-job contention, or
  absence of random-read amplification;
- any model-quality, metric, attack/defense, generalization, or publication claim.

## Requested owner/S00 decisions

1. The owner authorized the six-finding remediation and a local commit on
   `codex/s01-nuscenes-zip-backend`; no merge or push permission is implied.
2. Exact focused job `333206` passed. Authorize a local results/handoff commit so
   the independent S01-R task can re-review one durable worker SHA; no push or merge
   is implied.
3. Ask the independent S01-R task to re-review that new worker SHA. Until re-review, keep
   S01/S07 readiness blocked and preserve all forbidden interpretations above.
4. Assign the permission-out `build_gt_database.py` cache-path migration to S00 or
   its owning integration session; S01 did not edit that file.
