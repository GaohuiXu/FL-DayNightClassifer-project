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
- The final S01-R review baseline will be a later durable branch tip containing
  the approved full-gate results/handoff update. Until then, `011e464...` is the
  immutable implementation/run candidate, not a reviewed PASS.
- Working-tree scope: only S01-owned files listed below.
- Worker verdict: **PARTIAL / FULL-GATE PENDING**. Implementation and the approved
  bounded real-data smoke pass. The required ten-archive manifest, 100% train/val
  reference coverage, real directory/ZIP parity execution, and measurable full-data
  throughput/data-wait gate remain unexecuted. This is not a worker PASS and still
  requires independent `S01-R` review.

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
- the unified blob store supplies bytes to six-camera, keyframe LiDAR, and all
  requested multi-sweep paths; image decode uses `PIL.Image` over `BytesIO`, LiDAR
  uses `numpy.frombuffer(...).copy()`;
- info-cache construction uses metadata `sample_data.filename`, so it does not
  extract or probe every payload; 10-sweep caches retain an explicit empty sweep
  list at scene starts;
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
  attempt `332648` failed during v1 manifest construction and was not retried.
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

Approved real-data smoke: `RESULTS.md` contains the complete evidence for Slurm job
`330409` (`COMPLETED`, `00:01:46`, exit `0:0`). It passed module discovery, one
archive's 258,109-member manifest, 64 selected references spanning 24 camera + 4
key LiDAR + 36 previous sweeps, decoded output, CRC, 0/2-worker repeated hashes, and
persistent PID/handle lifecycle.

## Gate-by-gate status

| S01 gate | Status | Evidence / missing work |
|---|---|---|
| Canonical module and `NUSCENES_DATA_DIR` | PASS (bounded real) | GH200 job resolved module root and `trainval/` tables, 34,149 sample metadata |
| Preserve directory mode | PASS (implementation/static) | directory store retained; existing dataset paths remain supported |
| Manifest trainval01..10 | FAILED attempt; v2 fix local | job 332648 exposed a cross-archive repeated path in trainval02; v2 identical-copy handling awaits a new approved run |
| Worker-safe lazy handles/reopen | PASS (bounded real) | stable two persistent PIDs, per-process state, expected one post-fork reset, no epoch reopen |
| Six cameras/key LiDAR/requested sweeps use byte readers | PASS (bounded real) | 24 + 4 + 36 selected real references decoded/CRC checked |
| Info cache without extraction | PASS (implementation/static), NOT RUN full | metadata filename path implemented; full train/val 10-sweep cache pending |
| Directory/ZIP byte and decoded-array parity | IMPLEMENTED, NOT EXECUTED in GH200 smoke | synthetic/mini parity tests added; dependency-backed execution pending |
| 100% train/val referenced-member coverage | NOT RUN | requires ten-archive manifest + full cache/audit |
| Deterministic repeated multi-worker reads | PASS (bounded real) | 0 vs 2 workers and two persistent epochs matched |
| No extraction/shared writes | PASS for executed scope | external output root and fail-closed path guard; manifest mode 0444 |
| Full-data throughput/data-wait | NOT RUN | prepared profiler/launcher; smoke elapsed and `sacct` I/O are explicitly not substituted |
| Independent S01-R | NOT RUN | needs an exact durable worker version and separate review worktree |

## Coverage counts and hashes

Executed smoke counts:

- archive member coverage: 1/10 archives, 258,109 members indexed;
- selected reference coverage: 64/64 resolved and read: 24 camera, 4 keyframe
  LiDAR, 36 historical LiDAR sweeps;
- decoded samples: four, each with six RGB cameras and ten total LiDAR frames;
- repeated reads: 64 parent reads plus 128 persistent-worker reads across two
  epochs; all digests matched.

Full expected audit targets are train 28,130 + val 6,019 = 34,149 metadata samples,
but these are configured expectations, not executed coverage counts.

Immutable execution source-state hash:
`ee7c030911d5c2a99f7e60c73df4454d2f93d15de6881e0f081ceb04c1de0869`.
The pre-request tracked diff and changed-file content-manifest hashes are recorded
in `RUN_REQUEST.md`. Artifact/log hashes are recorded in `RESULTS.md`; notably:

- logical manifest:
  `0761493f3150aaa48e77ee93b4b27848cf3dd3537673800c920fcaec64c1734f`;
- manifest file:
  `d9aa3ada7261d9dea315f4fd8654cf559e773f01af2efc6f3ea796134d2d79c3`;
- smoke report file:
  `e882b490c8bd772c6addfdee20c2c369a2a83be7afddbf096bad9c51f2e1e830`.

## Negative results and residual risks

- Login-node data access lacked the gated group; the GH200 job had access. The
  architecture-specific environment also prevents treating login imports as a
  definitive runtime test.
- The one-archive job reached about 8.31 GiB `MaxRSS` and 522.20M aggregate disk
  writes while creating the 61.6-MB SQLite manifest. Full-gate memory/time/temp-I/O
  must be measured; neither a linear extrapolation nor a readiness claim is valid.
- Central-directory indexing is one-time, but random payload reads may still suffer
  filesystem amplification. Only the pending full-data profiler can quantify it.
- Full ten-archive duplicate/canonical/member coverage and archive mutation checks
  have not completed. Job 332648 stopped in trainval02 because the v1 schema could
  not represent a cross-archive repeated path. It did not establish whether the
  copies have identical content metadata.
- The local v2 correction retains all archive occurrences, permits only matching
  path+size+CRC copies, and routes reads to the lowest archive. Synthetic identical
  and conflicting cases pass, but shared-data v2 behavior is unverified until a
  separately approved follow-up.
- The full 10-sweep info cache and its exact train/val reference reconciliation are
  unbuilt. Permission/output-directory callers outside the owned data path have not
  been changed.
- Real directory/ZIP parity and spawn behavior remain dependent on executing the
  focused test suite in a compatible environment.
- Manifest and cache provenance must be frozen and hashed in every later resolved
  run; a manifest for only one archive must fail a full-data job.

## Interpretation limits

Allowed claims:

- the bounded one-archive GH200 smoke can discover the canonical module and read/
  decode real camera, key LiDAR, and nine-sweep history without extraction;
- on its 64 selected references, size/CRC validation passed and 0/2-worker repeated
  decoded hashes were deterministic with persistent worker-local handles;
- directory-mode support remains present in the implementation.

Forbidden claims:

- S01/full-data PASS or production training readiness;
- 100% train/val member coverage or all-ten-archive integrity;
- executed real directory/ZIP parity;
- acceptable full-data throughput/data wait, memory scaling, or absence of random
  read amplification;
- any model-quality, metric, attack/defense, generalization, or publication claim.

## Requested owner/S00 decisions

1. The owner authorized and S01 created the local branch/implementation commit
   above. No merge or push permission is implied.
2. Decide whether to authorize the separate exact full-gate scope appended to
   `RUN_REQUEST.md` for the
   ten-archive manifest, full train/val 10-sweep cache+coverage audit, real parity
   evidence, and bounded throughput/data-wait profile. The proposed CLI overrides
   the launcher's two-hour default with a 95-minute hard limit; it exceeds O-009's
   per-job 60-minute autonomous boundary and was deliberately not run.
3. Until those decisions and independent review, keep S01/S07 full-data readiness
   blocked and preserve all forbidden interpretations above.
