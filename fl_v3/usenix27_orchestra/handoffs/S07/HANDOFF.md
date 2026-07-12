# S07-A HANDOFF — reviewed S01 data-foundation integration

## Session identity and self-assessment

- Session/phase: `S07-A`, reviewed S01 data-foundation integration.
- Worker self-assessment: **S07-A-R P1 REMEDIATIONS FOCUSED VALIDATION PASS;
  READY FOR S00 COMPLETENESS CHECK AND INDEPENDENT RE-REVIEW**. This is not self-approval of the
  fixes, cache request, or Orchestra integration/scientific acceptance.
- Initial base: `953bfb57941b5a3660ed650c1a80267cd82245d4`, source branch
  `codex/s00-orchestra-ledger`, expected and observed detached mode.
- Worker branch: `codex/s07-a-data-foundation`.
- S01 implementation worker:
  `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`.
- S01 remediation implementation:
  `54a48f9102fd0de9a9abe97701550740b547e769`, an ancestor of the worker SHA.
- S01 review-only artifact:
  `7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc`, whose parent is old review
  baseline `ce2e77284b290de4c9faa6b2f971c0bd52f98eff` and whose diff contains only
  `fl_v3/usenix27_orchestra/handoffs/S01/REVIEW.md`.
- Original implementation commit:
  `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4`.
- Original implementation plus run-request/results evidence commit:
  `d26ba78a4766552c7c486206556183cc04bb9dae`.
- Original returned delivery SHA:
  `c7d57510e94de5429b62aa9df735a867bdcd199c`.
- Superseded pre-review compute-request remediation implementation:
  `ed31f23b2ee1b193b5dd3600c00570e40a888ce9`.
- Superseded pre-review cache-launcher source-state SHA-256:
  `7ddb06b3d57ef89be3b67782d90e93d64ddaa567ebd946ceda09910dc17b42f5`.
- Independent S07-A-R durable review:
  `976206405ccf7d2c864d318f5ee27302bdf59059`; its parent is exact delivery
  `d41150692e0be40ac87a6a9346ef36f13c0eb3a7`, REVIEW.md SHA-256 is
  `7264cd63bd7d807d6ac4490b63d6686ec5e83f6668dd7e2ddefcbb70ac1ce8d9`,
  and verdict is `CHANGES-REQUESTED` for two P1 provenance findings.
- P1 implementation/test/launcher commit:
  `0a89ea1dffdf9597d6330ca15dff01d6c6f15518`.
- P1 remediation implementation before locale hardening:
  `c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe` (adds zero-failure/error/skip
  JUnit acceptance to the focused launcher on top of `0a89ea1`).
- **Current remediation implementation / proposed executable `INT-A_SHA`
  (`NEW_IMPL_SHA`):**
  `44cefd06bc815e893919d95c754896711dba3402` (locks only source-list sorting to
  C locale on top of `c8dd920`; it does not change global test/runtime locale).
- Current full-cache runtime source-state SHA-256 (23 tracked files, recomputed
  from immutable `NEW_IMPL_SHA`):
  `1322c87255bc350323de108e347eea1e54daeb12b59fe1889cb15006f79c3884`.
- Current focused provenance-test source-state SHA-256 (25 tracked files):
  `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531`.
- Final remediation delivery HEAD is the documentation commit containing this
  updated handoff and request. Its exact SHA is returned to S00 in the session
  response and is recoverable with `git rev-parse HEAD`; embedding that commit's
  own SHA in its tree would change the SHA. S07-A-R should use the exact returned
  delivery HEAD, while any approved cache executor must use detached
  `NEW_IMPL_SHA`, not the documentation HEAD. The exact final delivery SHA is
  returned to S00 after the documentation commit; a commit cannot embed its own
  SHA without changing that SHA.

No merge into `v3-ad-perception`, push, PR, upload, branch deletion, or worktree
operation occurred.

## S07-A-R P1 remediation — locale-stable focused validation completed PASS

The owner authorized correction of both durable-review findings, scoped commits,
and S00 control through completion. The locale-stable bounded focused validation
was approved exactly once under O-009/delegation and executed as Job 335280. No
other `sbatch`/`srun`, retry, full cache, model, or follow-on was submitted.

After delivery `29d5edc0f8a3ac53928cd08636e6e02ead00b07d`, S00 independently
verified the immutable focused request and recorded one-time status
`APPROVED_ONCE_BY_S00_UNDER_O-009_AND_OWNER_DELEGATION_2026-07-11`. The approval
is bound to executable `c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe`, source hash
`357da48780436aaba3cbc6735e350d446763acc9f6cb8a0bf424728e55a32d0e`, the exact
mini input/output/command in RUN_REQUEST Section C, and one node/GH200, eight CPUs,
at most 15 minutes/0.25 GPU-hours, with no array/DDP/model/full-cache/metrics/
retry/resubmit/follow-on. Any field change invalidates approval. This record does
not claim execution and does not approve Section B full-cache materialization.

That approval was never submitted and is now invalid. A clean detached executor
at `c8dd920` (worktree suffix `e7d0`) rejected the request during preflight under
`LANG=en_US.UTF-8`: the unfixed `sort -u` produced focused aggregate hash
`8de319b624519ae9582be70699eafa6d9ebb8964bc8e8ba548bc67372201475c`, while the
S00 `C.UTF-8` preflight had approved `357da487...`. No `sbatch` ran, no output root
was created, and the queue remained empty. Because the implementation, source
hash, and output root changed, the old c8dd/357 approval is void rather than
reinterpreted.

Commit `44cefd06bc815e893919d95c754896711dba3402` minimally changes both S07-A
source-attestation functions from ambient-locale `sort -u` to `LC_ALL=C sort -u`.
It does not export or otherwise change the locale of Python, pytest, cache
construction, or runtime dependencies. Static reconstruction under `C.UTF-8`,
`en_US.UTF-8`, and `sv_SE.UTF-8` produced identical lists and aggregate hashes for
both launchers. The replacement focused request is
independently rechecked and now has exact-once status
`APPROVED_ONCE_BY_S00_UNDER_O-009_AND_OWNER_DELEGATION_2026-07-11`, strictly
bound to `44cefd06...`, focused hash `2710655b...`, file-list hash `90310705...`,
the exact mini root/output/command, and one node/GH200/eight CPUs/at most 15
minutes/0.25 GPU-hours, with no array/DDP/model/full-cache/metric/retry/resubmit/
follow-on. Any bound-field change invalidates approval. This record does not claim
additional execution; Job 335280 consumed the approval and completed PASS. Section
B remains pending owner approval.

P1-A is remediated at `NEW_IMPL_SHA`:

- `build_gt_database.py` now requires three independent frozen identities for the
  exact `(version, split, n_sweeps)` cache: raw-input canonical hash, physical
  pickle SHA-256, and physical sidecar SHA-256;
- all three digest inputs are syntactically validated; paths are resolved only by
  `info_cache.cache_paths`, and metadata must exactly match format `t1.v2`,
  version, split, depth, and canonical hash;
- pickle and sidecar are hashed before and after deserialization. Any missing,
  mismatched, or in-flight-changing artifact fails before blob-store opening and
  point cropping;
- GT-database `meta.json` now records absolute pickle/sidecar paths, byte sizes,
  both physical SHA values, canonical hash, format/version/split/depth, backend,
  and (for ZIP) exact manifest path/format/logical hash/file SHA/archive set;
- hostile tests mutate only derived `gt_boxes` or
  `lidar_sweeps[*].sweep2keylidar` while retaining raw canonical inputs, pickle
  metadata, and JSON sidecar consistency. `IC.load_cache` still accepts the raw
  canonical contract, but the GT caller rejects the changed physical pickle
  before `_open_blob_store` or `crop_object_points` can run.

P1-B is remediated at `NEW_IMPL_SHA`:

- full-cache attestation includes the complete tracked
  `fl_v3/data/nuscenes/*.py` package, Python package initializers,
  `fl_v3/data/partition.py`, `fl_v3/utils/runtime.py`, builder, launcher,
  environment bootstrap, and dependency/config manifests;
- this conservative tracked set covers the actual eager import graph:
  `fl_v3` → `fl_v3.data` → `fl_v3.data.nuscenes.__init__` →
  `dataset/info_cache/partition/...`; `dataset` reaches `fl_v3.utils.runtime`
  and `partition` reaches `fl_v3.data.partition`;
- execution identity now records installed Torch and Pillow in addition to
  Python/platform, NumPy, nuscenes-devkit, and pyquaternion;
- both full-cache and focused launchers require clean detached execution at the
  immutable SHA and use an auditable tracked-file set rather than filesystem
  `find` results.

The obsolete execution tuple
`ed31f23` / `7ddb06...` / `s07a_cache_t1v2_ed31f23b2ee1` remains permanently
unapproved and must not be reused or relabeled.

The latest actually executed dependency record remains Job 333477 at the earlier
`c1f4fbe` tree: NumPy `1.26.4`, nuscenes-devkit `1.1.11`, Pillow `12.2.0`, pytest
`9.1.1`, and Torch `2.11.0+cu128` on aarch64 node `n430`. Read-only prefix metadata
currently reports pyquaternion `0.9.9`; because Job 333477 did not record that
package, the new focused/full launchers must capture the installed value in-job and
that future execution record—not this metadata observation—is authoritative.

## Superseded pre-review compute-request remediation — historical, no execution

Before S07-A-R, S00 returned only the pending full-cache request. The owner authorized launcher
and documentation remediation but explicitly prohibited new compute. This
continuation therefore submitted no `sbatch`/`srun`, did not rerun Job 333477, and
did not create an execution worktree. Existing Job 333477 artifacts, facts, and
hashes in `RESULTS.md` remain unchanged.

That now-superseded launcher/request revision enforced:

- eventual execution only from a fresh owner/Codex-UI-provisioned isolated
  worktree in clean `detached@ed31f23b2ee1b193b5dd3600c00570e40a888ce9`
  mode; the pending command and launcher `EXPECTED_S07A_SHA` both bind that SHA;
- runtime source-state hash
  `7ddb06b3d57ef89be3b67782d90e93d64ddaa567ebd946ceda09910dc17b42f5`,
  recomputed identically from the worktree and immutable `ed31f23` objects;
- fresh proposed output root
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_ed31f23b2ee1`,
  confirmed absent but not created;
- exact fail-closed counts in both actual records and cache metadata: train
  `n_samples=28130`, `n_boxes=944881`; val `n_samples=6019`,
  `n_boxes=187528`; `cache_identity.json` records expected, actual, and metadata
  counts separately;
- actual in-job interpreter/platform identity and installed NumPy,
  nuscenes-devkit, and pyquaternion versions. Read-only prefix metadata currently
  reports CPython `3.11.15` on `aarch64`, NumPy `1.26.4`, nuscenes-devkit
  `1.1.11`, and pyquaternion `0.9.9`, but the job record—not lockfiles—is
  authoritative after any approved execution;
- separate checksum generation and in-job verification: after writing
  `sha256sums.txt`, the launcher runs
  `sha256sum -c "$S07A_OUTPUT_ROOT/sha256sums.txt"` and fails on mismatch.

It was never approved. The current request replaces its SHA/hash/output tuple;
both requests remain non-executable without exact owner approval.

## Exact approved Git topology

1. Created `codex/s07-a-data-foundation` from detached
   `953bfb57941b5a3660ed650c1a80267cd82245d4`.
2. Created non-fast-forward merge `60f603a0837a55b8bc5d56eedcbba065fcc10673`
   with parents, in order:
   `953bfb57941b5a3660ed650c1a80267cd82245d4` and
   `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`.
3. Cherry-picked review-only commit `7cf7fcc...` as
   `a4ca386db59a9250d3fce95209e38ac617b4ff77`.
4. Added scoped implementation/test/docs commit `c1f4fbe...`, evidence commit
   `d26ba78...`, and original handoff delivery `c7d5751...`.
5. After S00 returned scoped compute-request changes, added launcher-only
   `ed31f23b...`; subsequent commits update only request/handoff
   documentation and did not change the executable tree at `ed31f23b...`.
6. Independent S07-A-R returned durable `CHANGES-REQUESTED` review
   `976206405ccf7d2c864d318f5ee27302bdf59059` without entering this branch.
   The owner-authorized P1 implementation/test/launcher remediation is scoped
   commits `0a89ea1dffdf9597d6330ca15dff01d6c6f15518` and
   `c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe`.
7. S00 recorded then invalidated-before-submission a focused approval after the
   locale-sensitive executor preflight. Minimal launcher-only implementation
   commit `NEW_IMPL_SHA=44cefd06bc815e893919d95c754896711dba3402`
   freezes only source-list collation.

The exact S01 history `011e464 → 1fe6517 → ce2e772 → 54a48f9 → abe5c58` remains
reachable through the merge second parent. The review branch itself was never
merged, and no S02-S06 implementation is present.

## Preflight and upstream artifact verification

Before editing, the worktree was clean and matched the kickoff. The target branch
did not exist and was not checked out elsewhere. `git merge-tree --write-tree`
predicted a clean merge. All upstream SHAs and parent relations matched.

Raw S01 evidence was independently rechecked:

- Job 332651 output/root and logs exist; `sha256sum -c sha256sums.txt` passed for
  the accepted 633,106,432-byte manifest, coverage/profile files, and historical
  `t1.v1` caches/sidecars. Manifest logical/file hashes are
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6` /
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`.
  Coverage is `538,695/538,695`, zero missing. These caches remain forbidden
  production inputs.
- Job 333206 output/logs exist; its checksum list passed. Execution identity is
  remediation commit `54a48f9` and source hash
  `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda`;
  JUnit reports 56 passed, zero failure/error/skip.
- The S01 review records final **PASS** at `abe5c58`, while preserving the limits
  that 332651 lacks in-job source attestation and no full trainval `t1.v2` cache
  has yet been made.

## Files and semantic changes

Integrated unchanged from reviewed S01:

- `AGENTS.md`, `fl_v3/docs/env.md`;
- `fl_v3/src/fl_v3/data/nuscenes/{__init__,conventions,dataset,info_cache,paths,zip_backend}.py/md`;
- S01 manifest/audit/benchmark/smoke/cache launchers and scripts;
- S01 data/cache tests and complete `handoffs/S01/` package.

S07-A changes:

- `fl_v3/scripts/build_gt_database.py`
  - removes the direct `nuscenes_info_*_t1.v1.pkl` bypass;
  - requires explicit `n_sweeps`, frozen expected canonical cache SHA, and exact
    physical pickle plus sidecar SHA-256 values;
  - loads only through `info_cache.load_cache(..., n_sweeps=...,
    expected_cache_hash=...)`, inheriting format, sidecar, every-record depth, and
    canonical content validation;
  - ZIP mode additionally requires and validates the logical manifest hash and
    SQLite file SHA-256, and exact trainval01..10 archive names;
  - hashes the exact depth-specific pickle/sidecar before and after load, checks
    format/version/split/depth/canonical metadata, and fails before blob opening
    or point cropping on any physical-identity drift;
  - uses one explicit `NuScenesBlobStore` for both single/multi-sweep reads and
    records physical cache/backend/manifest provenance in GT-database `meta.json`;
  - preserves directory mode without a ZIP manifest and rejects relabeling a
    directory backend with ZIP provenance.
- `fl_v3/src/fl_v3/data/nuscenes/info_cache.py`
  - adds optional `expected_cache_hash` validation after sidecar/depth/content
    verification; scientific callers can now bind a pre-frozen cache identity.
- `fl_v3/scripts/run_s01_nuscenes_zip_tests.sh`
  - expands source-state attestation to `tests/conftest.py`, the GT test/caller,
    `pyproject.toml`, requirements and lockfile;
  - disables external pytest plugin autoload/ADDOPTS drift and records installed
    dependency versions;
  - includes the new GT/cache provenance tests.
- `fl_v3/scripts/run_s07a_nuscenes_cache_t1v2.sh`
  - pending-approval launcher that validates the accepted manifest logical/file
    identities, builds only train/val `t1.v2` depth-10 caches, reloads them
    explicitly, and emits source/execution/cache/checksum identities;
  - attests the complete tracked nuScenes package and actual external local import
    dependencies, records Torch/Pillow versions, and requires a clean detached
    executor;
  - does not rebuild the manifest, profile, train, evaluate, or auto-retry.
- `fl_v3/scripts/run_s07a_provenance_tests.sh`
  - new exact one-job focused regression launcher for the P1-A hostile cases;
  - records immutable source/dependency identity and verifies checksums, but has
    not been submitted.
- `fl_v3/tests/test_build_gt_database.py` and
  `fl_v3/tests/test_nuscenes_info_cache.py`
  - cover explicit depth/canonical/physical hashes, historical format/path rejection,
    missing/mutated sidecars, content hash drift, logical/file manifest mismatch,
    derived `gt_boxes`/`sweep2keylidar` hostile mutation before blob/crop, GT caller
    behavior, and directory/ZIP preservation.
- `AGENTS.md` and `fl_v3/docs/env.md`
  - record S01/S01-R reviewed PASS, exact historical limits, S07-A migration,
    pending full cache status, explicit production provenance contract, and the
    unchanged O-009 boundary.
- `fl_v3/usenix27_orchestra/handoffs/S07/{RUN_REQUEST,RESULTS,HANDOFF}.md`
  - immutable O-009/full-cache request, execution evidence, and this handoff.

Canonical `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md` are unchanged.

## Production cache-consumer audit

An AST call search at `c1f4fbe` classified every `IC.load_cache` call. Within the
declared S07-A data-foundation ownership, all production/material entry points are
explicit:

- GT database: explicit depth plus expected cache and manifest identities;
- S01 coverage audit/benchmark: explicit depth, with actual cache/manifest hashes
  emitted as evidence rather than predeclared because these tools validate/build
  the artifacts;
- S07-A cache launcher: explicit depth and post-build canonical/file checksums;
- cache builder: explicit `n_sweeps` passed to `get_or_build_cache`.

The ambiguous production consumer in `fl_v3/src/fl_v3/training/tasks.py` and older
training/eval/diagnostic scripts are outside this phase's ownership and are listed
under DEFER below. S06 remains responsible for binding resolved `n_sweeps`, cache
hash, and manifest hash in config/provenance at every scientific entry point.

## Three-way old-interface inventory

### REMOVE NOW — deleted

Call-search command against pre-cleanup `a4ca386`:

```bash
git grep -n -E '\b(_decode_image_chw|sweeps_dir|iter_required_sensor_dirs)\b' \
  a4ca386db59a9250d3fce95209e38ac617b4ff77 -- fl_v3
```

It returned only the three definitions, with no caller:

- private `dataset._decode_image_chw`: superseded by byte decode plus the unified
  blob store used by `NuScenesMultimodalDataset`;
- `paths.sweeps_dir`: superseded by canonical metadata-relative member paths and
  unified directory/ZIP reads;
- `paths.iter_required_sensor_dirs`: superseded by `verify_dataset`'s exact
  directory/archive preflight.

The unused `info_cache` import in `dataset.py` and now-unused `typing.Iterable`
import were also removed. Job 333477's directory/ZIP raw/decoded parity,
directory-store, ZIP preflight, missing-member, and extraction-free tests cover the
replacement paths.

### KEEP / DEPRECATE — retained

- Directory `NuScenesBlobStore`, `samples_dir`, `abspath_from_relative`, and
  `relative_to_dataroot`: required for directory parity, cache portability, tests,
  and migration.
- `paths.DATAROOT`: compatibility snapshot with many active test/diagnostic callers;
  new production code should use `get_dataroot(run_config)`.
- `_infer_root_and_relative`, `_compat_blob_store`, and legacy absolute-path
  `_load_lidar`: active compatibility route covered by
  `test_legacy_absolute_lidar_and_multisweep_paths_use_zip_backend`.
- `load_cache(..., n_sweeps=None)`: retained only as fail-closed migration/test
  discovery when exactly one `t1.v2` depth exists; scientific entry points must
  pass depth and expected hash explicitly.
- S01 manifest/coverage/profile tools and historical handoff records: required
  provenance/evidence, not dead code.

### DEFER TO S07-B / S06 ownership — no edits in S07-A

- `fl_v3/src/fl_v3/training/tasks.py` production cache load;
- `arrhenius_smoke.py`, `_bench_msweep.py`, `_t4_fd_diagnose.py`,
  `p3_partition_health.py`, `t3_iid_vs_central.py`, `t4_readiness_eval.py`,
  `t5_attack_eval.py`, and `t5_mini_smoke.py` cache loads;
- remaining direct `P.DATAROOT` uses in historical calibration/partition/eval
  scripts.

These cross trainer/config/eval or historical-diagnostic ownership. Broad cleanup
here would violate the phase boundary. S06/S07-B must distinguish scientific
entry points (mandatory explicit depth/hash/config provenance) from retained
historical diagnostics and test migration paths.

## Verification and execution evidence

Local/static:

- `python3 -m py_compile` on changed Python source/scripts/tests: PASS.
- `bash -n` on all S01 ZIP launchers plus
  `run_s07a_nuscenes_cache_t1v2.sh`: PASS.
- `git diff --check`: PASS.
- Login `/usr/bin/python3 -m pytest`: unavailable (`pytest`, NumPy, and Torch not
  installed); not treated as a test result.
- Scoped remediation: `bash -n` passed for the changed cache launcher;
  `python3 -m py_compile` passed for the unchanged Python modules in its runtime
  source set; `git diff --check` passed; and the read-only source-state hash from
  the worktree exactly matched recomputation from immutable `ed31f23b...` at
  `7ddb06b3d57ef89be3b67782d90e93d64ddaa567ebd946ceda09910dc17b42f5`.
  Per owner instruction, no pytest or compute was rerun for this remediation.
- S07-A-R P1 remediation through `c8dd920cf3f8...`: `python3 -m py_compile`
  passed for `build_gt_database.py` and its test; `bash -n` passed for the full
  cache and new focused provenance launchers; `git diff --check` passed.
- Login-node `python3 -m pytest -q fl_v3/tests/test_build_gt_database.py` could
  not start because `/usr/bin/python3` has no `pytest`; this is an environment
  limitation, not a PASS or code failure. Job 335280 supplies the dependency-
  complete GH200 evidence instead.
- Locale-hardening commit `44cefd06...`: `bash -n` passed for both changed
  launchers and `git diff --check` passed. Audit found no other source-list sort
  in either S07-A attestation path.
- Under each of `C.UTF-8`, `en_US.UTF-8`, and `sv_SE.UTF-8`, the full-cache
  list/hash was exactly
  `eebaaf9528a56004b63cc2cb37fe6d312b75a52df450f374307e8e559cb1cbb5` /
  `1322c87255bc350323de108e347eea1e54daeb12b59fe1889cb15006f79c3884`;
  the focused list/hash was exactly
  `90310705f1bac3bcdfba9128deea6aed60a270e811cc62759f1204612d61d913` /
  `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531`.
- Both aggregate hashes were independently recomputed from immutable Git blobs at
  `NEW_IMPL_SHA`; full uses 23 files and focused uses 25 files.
- The full-cache root
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_44cefd06bc81`
  remains absent. The focused root
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_44cefd06bc81`
  was absent before submission and was created only by Job 335280.

O-009 Job `333477`:

- exact implementation `c1f4fbe`, source hash
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`;
- `n430`, one GH200/eight CPUs, `COMPLETED 0:0`, `00:01:23`, about 0.0231
  GPU-hours, no retry;
- `62 passed in 12.83s`, zero failures/errors/skips;
- `sha256sum -c` passed for execution identity, source list, pytest log, and JUnit;
- complete hashes/resources/limits are in `RESULTS.md`.

O-009/delegated Job `335280`:

- exact implementation `44cefd06bc815e893919d95c754896711dba3402`, focused
  source/list hashes `2710655b...` / `90310705...`;
- `n430` aarch64, one GH200/eight CPUs, `COMPLETED 0:0`, `00:01:16`,
  approximately 0.0211 elapsed GPU-hours, `Restarts=0`, no retry/requeue/
  resubmission/follow-on;
- batch `MaxRSS=540M`, `MaxVMSize=6476352K`, `TotalCPU=00:08.591`;
- JUnit `7/0/0/0`, pytest `7 passed in 1.52s`; both derived `gt_boxes` and
  `sweep2keylidar` physical-hash hostile cases were present and passed;
- in-job artifact checksum verification and independent 25-file source verification
  passed; actual Python/dependency identity and all hashes are in `RESULTS.md`;
- stderr contains only the normal module-purge notice. The approval is consumed.

No full cache job, model job, 100/1000-step job, full-data profile, metric,
scientific run, or upload was submitted. Job 335280 is the latest execution
evidence and is limited to focused real-mini/synthetic provenance validation.

## Exact cache/manifest/provenance contract for S06

For each production/scientific train or val consumer, the resolved config and run
provenance must freeze and agree on:

1. official `version` and `split`;
2. integer `n_sweeps` (primary request is 10 total including keyframe);
3. cache format exactly `t1.v2` and depth-specific filename;
4. canonical cache hash from sidecar/pickle metadata, passed as
   `expected_cache_hash` and recomputed from every record on load;
5. cache pickle file SHA-256 and sidecar file SHA-256, both required by the GT
   caller and checked before/after deserialization;
6. backend mode: `directory` or `zip`;
7. for ZIP mode, exact accepted manifest path, format, logical manifest hash,
   SQLite file SHA-256, and archive-name set;
8. locked train counts `28130` samples / `944881` boxes and val counts `6019`
   samples / `187528` boxes, checked independently against actual records and
   metadata;
9. actual Python/platform plus NumPy, nuscenes-devkit, and pyquaternion versions;
10. generated artifact hashes followed by successful in-job `sha256sum -c`;
11. resolved config hash/checkpoint provenance that includes these fields.

Loading must fail before data/model execution on missing/ambiguous depth, `t1.v1`
or other format, missing/mutated sidecar, record-depth drift, canonical hash drift,
expected cache hash mismatch, manifest logical/file/archive mismatch, or backend
relabeling. Directory mode carries null ZIP-manifest fields and must not be treated
as legacy/dead.

The remediated full cache request in `RUN_REQUEST.md` is exact and **PENDING**. Its eventual
executor must be a fresh clean UI-provisioned
`detached@44cefd06bc815e893919d95c754896711dba3402` worktree. If separately
approved and passed, it will fill the currently unknown train/val canonical cache
hashes and physical pickle/sidecar hashes in `cache_identity.json`. Until then,
S06 must not substitute the historical `t1.v1` files or claim full-data readiness.

## Gate checklist

| S07-A acceptance item | Self-assessment | Evidence / limit |
|---|---|---|
| exact worker history reachable | PASS | merge second parent is exact `abe5c58` |
| REVIEW artifact present, not implementation | PASS | cherry-only `a4ca386`; review parent remains old `ce2e772` |
| no production GT `t1.v1` bypass | PASS | explicit IC API plus hostile test |
| depth/format/sidecar/cache/manifest fail closed | PASS | job 333477, 62/62 at `c1f4fbe`; retained GT cases plus P1 changes pass Job 335280 |
| physical pickle/sidecar GT binding | PASS (worker evidence) | Job 335280: both hostile cases pass before blob/crop, 7/0/0/0 |
| directory mode supported | PASS | implementation plus directory/ZIP tests |
| focused launcher attests fixtures/config/deps | PASS (worker evidence) | Job 335280 exact SHA/hash, all 25 sources OK, runtime identity captured |
| shell/Python/diff checks | PASS | commands above |
| executor ref/worktree contract | PASS | fresh clean UI worktree at detached `NEW_IMPL_SHA` required |
| exact cache sample/box counts | PASS (implementation) | train 28130/944881; val 6019/187528; actual+meta checked |
| actual runtime dependency capture | PASS (implementation) | Python/platform + NumPy/devkit/pyquaternion/Torch/Pillow written in-job |
| generated checksum verification | PASS (implementation) | explicit in-job `sha256sum -c` after generation |
| canonical Orchestra docs unchanged | PASS | empty diff for three canonical paths |
| no S02-S06 integration | PASS | topology/name-scope audit |
| full `t1.v2` cache | PENDING | exact request prepared; not submitted |
| model/scientific work | NOT RUN / FORBIDDEN | phase boundary preserved |

## Negative results, residual risks, and interpretation limits

- Full trainval `t1.v2` train/val cache artifacts do not yet exist; their cache and
  file hashes cannot be frozen until owner-approved execution.
- The new P1-A hostile regressions executed exactly once on GH200 and passed. The
  approval is consumed; no retry/resubmit/follow-on is authorized.
- The previous c8dd/357 request was approved once but rejected before submission
  because executor locale changed its aggregate source hash. This is a preserved
  negative reproducibility finding: zero jobs, zero outputs, empty queue.
- The remediation launcher/count/runtime/checksum changes have only local/static
  plus focused GH200 validation. They remain pending independent re-review and do
  not establish full-cache/model/scientific readiness.
- Historical job 332651 proves referenced-member coverage and loader-only timing,
  but its `t1.v1` caches are forbidden and its job lacks retroactive in-job source
  attestation.
- Job 333477 is real-mini/synthetic engineering evidence, not trainval-scale
  directory/ZIP decoded parity or model-step evidence.
- Full epoch/model data wait, concurrent shared-filesystem contention, all-payload
  CRC coverage, and production model I/O remain untested.
- S06/S07-B must migrate the deferred scientific entry points and bind the exact
  post-materialization hashes; S07-A intentionally did not cross ownership.

Allowed interpretation: reviewed S01 is integrated with preserved history/review;
historical Job 333477 still supports the pre-review 62-test data-foundation tree;
and Job 335280 supports the exact P1 physical-cache provenance regressions and
locale-stable source attestation at `44cefd06`.

Forbidden interpretation: independent remediation acceptance before re-review,
permission to rerun Section C or submit pending Section B, full-data/model training readiness, architecture or
metric acceptance, mAP/NDS/model quality, FL, attack/defense, generalization,
scientific, or publication claims.

---

# S07-B HANDOFF — reviewed CL-stack history and local integration candidate

## Status and exact identity

- Session: `S07-B`, sole reviewed CL-stack integration worker.
- Base: `c9c84f8b2caebea14adc1d79d6d706695be0f50f` from
  `codex/s00-orchestra-ledger`.
- Startup observation: detached HEAD exactly at the base, empty branch name, clean
  status, top level
  `/home/gaohui/.codex/worktrees/d5e7/fl_weather_project`.
- Owner-authorized delivery branch:
  `codex/s07-b-integrated-cl-stack`; no other branch or worktree operation occurred.
- Final code/config integration candidate before the closing handoff-only commit:
  `e3cedfa984c48bc4ae28539971f5f0526ce7d916`.
- Worker self-assessment: **IMPLEMENTATION CANDIDATE DELIVERED; RUNTIME/FULL-STACK
  GATES NOT RUN; NOT PRODUCTION OR SCIENTIFIC PASS**. The final handoff commit SHA
  is returned to S00 after commit because a commit cannot embed its own SHA.
- No merge to `v3-ad-perception`, push, PR, upload, branch deletion, worktree
  add/move/remove/prune, Slurm submission, `srun`, retry, or external publication
  occurred.

## Exact owner-approved history integration

The pre-merge state was branch `codex/s07-b-integrated-cl-stack` at exact base
`c9c84f8...`. Expected overlaps were shared detector/eval/runtime interfaces and,
specifically, S05+S06 `detection_eval.py`. All five non-FF merges completed in the
required order without a textual conflict:

| Worker | Exact second parent | Resulting merge SHA |
|---|---|---|
| S02 | `3aebf2dc1d19473f29260df279421047d216d70e` | `062ee1c5596db3e77203d9d5869bc988b5beb0ed` |
| S03 | `50893839c45cd3e2ef1b72b98db6668df7030f2a` | `21d822d7ec7ff993b079f0d572bc9215164946a8` |
| S04 | `483e149b95ec891b675df825d924a96bb225b7dd` | `10fc657bbf3a3067695db4cb5c5b44c913ab0b6a` |
| S05 | `a9c801fdee378906e54d06314d0c772b6559901a` | `5f186d079a1b39133010096477b2adda8e9eeb66` |
| S06 | `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7` | `9fb1a9a9a448c90a60d75850f8146d2d4da06b80` |

Clean auto-merge was treated only as textual integration. The later scoped commits
perform the semantic wiring/audit. No reviewer branch was merged or cherry-picked.
Exact final reviewer bytes were materialized only at the five owned `REVIEW.md`
paths, then committed as provenance import
`588e9f42a3bf9aa1341fd57c5ce8a838f0e299e0`:

| Review | Git blob | SHA-256 |
|---|---|---|
| S02 | `f882a7e223ccc88084d283269ac5ba2516a482f0` | `8bb56cafc22a38dfd7b4ef4d755f1531ab081b0371fe18585d744307f5640474` |
| S03 | `09d1beb66cec07e769c3650dd9e09a942bceb674` | `01dea6fd81f14bee8ee1cdf9e4dc66488e7253075459821b2e63947fde7566c1` |
| S04 | `1caa6d01d83792736ebacbc6eecdf6b42bdadb2e` | `8673672793235ae0226d9109c73cd39577d5f40e846b17425178a7011300ea2a` |
| S05 | `d3fc2bec71fbb3206de50b3baeb3ad7db6dc9ef7` | `67b58c8e9d1d1622d1af49a2c052cbadd66580500dbf988fc1184f2d0df6736e` |
| S06 | `6df4171c0e85b4a63270af91ca18004c7db3a2e4` | `96d1996562bae4b5e2d1204cb6b51d276ad5c50dd7a75e928137b52b41ae0a59` |

## Scoped S07-B commits and semantic inventory

### `f629462c79df6ccbd491b595df1e8ad0f52bc94b` — mode-aware S01 payload I/O

- `data/nuscenes/dataset.py` accepts only exact C/L/F names. Camera-only never
  opens/reads/decodes LiDAR payloads; LiDAR-only never opens/reads/decodes image
  payloads; fusion reads both. Calibration, poses and GT remain available.
- `zip_backend.py` records camera/LiDAR/other read and byte counters and resets
  process-local handles/counters across pickle/fork lifecycle.
- `models/fusion/collate.py` omits disabled payloads, rejects mixed modes and
  unexpected disabled payloads, and preserves metadata.
- `training/tasks.py` passes exact mode and explicit production manifest to the
  dataset; the S06 synthetic raw-I/O fail seam was removed only after the real
  mode-aware implementation existed.
- `test_nuscenes_zip_dataset.py` adds hostile missing-disabled-payload directory
  and ZIP cases at depth 10 for camera-only/LiDAR-only. This original statement is
  intentionally narrowed by the remediation matrix below; the pre-review suite did
  not cover depth 1, fusion, or the full backend/mode cross-product.

### `e6ec980463b1e0aa1743df1aaedb78557ea3c65e` — reviewed detector stack

- Strict resolved construction maps `swin_t_stride8` to trainable Swin-T,
  reference image augmentation, all-level stride-8 FPN, 0.5 m depth bins and the
  common 180x180 BEV; `second_075` to the reviewed 0.075x0.075x0.2 SECOND input,
  XY stride 8, 120k/160k train/eval caps, 256-channel 180x180 output and fp16
  sparse dispatch; `pillar_020` to the repaired 0.2 m/512x512 pillar control with
  four-stage dense LiDAR backbone; and `conv_fuser_256` plus two-convolution
  six-task CenterHead to exact constructors. Missing, unknown, legacy, or
  mode-inconsistent mappings fail before model construction.
- The camera initialization choice is now an explicit resolved boolean rather
  than an implicit default. Candidate templates leave it unresolved/null.
- `MultiTaskCenterPointLoss` partitions canonical global labels by the reviewed
  name map `[(0),(1,4),(2,3),(9),(6,7),(5,8)]`, applies the reviewed S02 Gaussian
  target/regression order to each S05 task and sums task objectives.
- Production decode routes list-of-task dictionaries only through S05 forced-FP32
  candidate selection and deterministic official task-wide NMS. The old
  single-head decoder is retained only for inventoried non-production callers;
  strict config/checkpoint structure cannot select it.
- F-CBGS is hash-bound sampling, fixed at threshold 0.5/max-repeat 4.0, and fails
  if stacked with heatmap/regression class weights. Production train loaders use
  `EpochPermutationSampler` after any CBGS expansion, so the persistent trainer
  has a deterministic `set_epoch()` sampler and resume addresses an epoch rather
  than generator history.

### `8e78b643aae4ba5869af2d09d319bc81e33b56ed` — S05+S06 official eval audit

The clean auto-merged implementation already retained both reviewed semantics:
S05 duplicate-token rejection, official <=500 cap, deterministic content order,
canonical global conversion/devkit contract; and S06 actual-mode metadata,
resolved precision/autocast, forced-FP32 output boundary, one traversal, timing
neutrality and complete optional identity provenance. This commit corrects the
integrated contract documentation so evaluation cannot suggest a legacy global
`max_objects` override. It does not re-decode or introduce a second threshold.

### `2944386de19ab7d25b3ec09c77b6951dd34cea8d` — strict runtime/config/checkpoint evidence

- `resolved.py` binds exact architecture enums, explicit camera initialization,
  uniform/CBGS sampling, precision, optimizer/exposure, cache/manifest identities,
  Torch and sparse dependency identities. `t1.v1`, aliases, missing/unknown keys,
  non-SECOND spconv claims and CBGS outside F-CBGS fail closed.
- `runtime.py` verifies exact Torch build string and, for SECOND, exact installed
  spconv/cumm versions, active import roots and clean Git source HEADs. This check
  runs before physical data verification or model construction. Any approved
  launcher must still hash its immutable executable/runtime source snapshot; no
  actual Arrhenius package/source attestation was executed in this session.
- `centralized_train.py` is the one strict `--config/--out-dir/--resume`
  production entry, records dependency identity, remains world-size-one only,
  uses fixed accumulation/exposure accounting and complete boundary checkpoints.
- `test_s06_checkpoint_resume.py` adds hostile failure injection after the real
  live model or AdamW object's `load_state_dict` has mutated state; rollback is
  required to restore every component and RNG. These tests are authored but not
  run here and therefore do not establish production fail-atomic evidence.
- `test_s07_b_integration.py` covers enum constructors/geometries, unknown and
  inconsistent mappings, mocked dependency-source identity drift, deterministic
  CBGS/no-stacking, expanded-dataset sampler resume, template refusal, global-to-
  task loss mapping and gradient reachability.
- `run_s07_b_static_checks.sh` is the local reproducible static-only check. It
  contains no pytest, data, CUDA, model step, Slurm or retry action.

### `e3cedfa984c48bc4ae28539971f5f0526ce7d916` — sparse build identity closure

- Strict SECOND configs additionally require exact SHA-256 identities for the
  active spconv and cumm import package trees. Python source, generated/native
  artifacts and metadata under the import package roots are content-hashed in a
  locale-independent relative-path order; `__pycache__`/`.pyc` interpreter caches
  are excluded.
- Runtime compares both build hashes before data/model construction, in addition
  to exact versions, active import roots and clean source Git HEADs. Candidate
  templates carry explicit invalid build-hash sentinels until an approved
  Arrhenius attestation supplies real values.

## Candidate config files and explicit blocker

The five candidate files name the requested architecture/precision/head/sampling
choices, but are deliberately **NON-RUNNABLE TEMPLATES**, not resolved configs.
Each contains an unknown top-level `template_only` marker, unresolved/null camera
initialization where applicable, unmaterialized full `t1.v2` identities, and
unapproved placeholder budget/seed values. Strict loading rejects each before data
or model construction. File-byte SHA-256 values are:

| Candidate | File SHA-256 |
|---|---|
| C-STR8 `s07_b_c_str8.json` | `d2eaa46c800ebea5927359398acd88b38d90219c2f1f3841a4b1897ed05f8cc6` |
| L-P020 `s07_b_l_p020.json` | `625242234a03314010860e6026b0fbb88b774a9aeec12c7f7fe870203da07421` |
| L-S075 `s07_b_l_s075.json` | `1658cd5ec0e9c1b8945646d2e23a8db4419d16c2f644ca5a99b94c3477dcce1d` |
| F-U `s07_b_f_u.json` | `df7f36fe28e0d0c6c8275b293318cf7fae2e3c71fe3c60b7a7b81c26af69fa2e` |
| F-CBGS `s07_b_f_cbgs.json` | `bd8c57e84b34f835f3eaafe71f259a0c4131748bb27a62edf83bcd7f44bb54f0` |

The explicit blocker to a real resolved candidate is unchanged: full trainval
`t1.v2` train/val cache artifacts and their logical/pickle/sidecar identities are
absent; exact module dataroot/manifest identities, budget, seed, camera
initialization and CBGS-adjusted schedule are not owner-frozen; real spconv/cumm
build-tree hashes have not been attested on Arrhenius. Synthetic all-`a`
hashes were used only in an in-memory schema reachability check and are not written,
reported as config identities, or permitted as production inputs.

## Legacy caller/path inventory and cleanup decision

- `centralized_train.py` is now the sole resolved production entry.
- Historical `run_arrhenius_stop_e_gate.sh` supplies removed CLI flags
  (`--epochs`, `--max-steps`, `--tag`, flat overrides) and is rejected by argparse;
  it is outside S07-B ownership and was neither edited nor deleted.
- Imported `s06_synthetic_camera.json` lacks the new explicit initialization and
  sampling fields and remains synthetic; it is not silently promoted to an S07-B
  production config.
- `run_s06_runtime_tests.sh` lists `centralized_train.py` only as attested source;
  it does not call the trainer. `p3_crt_probe.py` mentions it only in prose.
- Historical tests and FL helpers calling `_det_config_from_run` without
  `s06-production-runtime` retain the old compatibility branch. No legacy symbol,
  config or launcher was bulk-deleted, and no negative evidence was erased.

## Verification performed in S07-B

PASS local/static only:

1. startup ref/top-level/clean-tree checks;
2. five non-FF merge-parent topology checks;
3. exact Git blob and SHA-256 checks for all five imported reviews;
4. `python3 -m py_compile` on every S07-B changed Python source/test listed by
   `run_s07_b_static_checks.sh`;
5. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`;
6. `python3 -m json.tool` on all five templates;
7. strict loader refusal of all five checked for `template_only` before
   construction;
8. in-memory schema reachability for all five after removing the template marker,
   supplying an explicit `camera_pretrained=False`, and substituting syntactically
   valid fake hashes; this validates enum/schema consistency only, not identity or
   run readiness;
9. `git diff --check` on every S07-B-authored working/staged diff after each
   integration stage;
10. candidate file hashes reproduced by the committed static script.

The final whole-history check
`git diff --check c9c84f8...HEAD` intentionally does not report a global PASS: it
finds one blank line at EOF in the exact imported S03 `REVIEW.md` bytes and trailing
spaces in two immutable S03 Job 336708 raw text artifacts (`scontrol.txt` and
`stdout.txt`). Those are pre-existing reviewed/raw-evidence bytes introduced by
the exact S03 worker merge. Editing them would violate history/review/artifact
preservation, so they remain recorded warnings rather than S07-B changes.

NOT RUN, with no implied PASS:

- pytest is unavailable in login `/usr/bin/python3`; Torch is also unavailable;
- every S02-S06 reviewed focused runtime test and all authored S07-B pytest cases;
- actual directory/ZIP payload/decode parity, hostile disabled-payload reads,
  persistent fork/spawn workers and resume;
- actual spconv/cumm/Torch import/build/source attestation on an aarch64 GH200;
- S04 same-instance fp16 train/eval/no-grad/concurrency/EMA/deepcopy lifecycle;
- C/L/F real construction, B=1/4/16 forward/backward, grid/dtype/gradient and
  batch-invariance cases;
- real model/optimizer rollback injection, CUDA rollback and host-memory gate;
- official devkit load/eval round trip and worst-case CPU float64 rotate-NMS
  profile;
- mini data/model steps, full trainval cache, 100/1000 steps, production-shape or
  full-data profile, mAP/NDS, DDP, matrix, seed, rerun or retry.

No new S07 `RUN_REQUEST.md` or `RESULTS.md` section was created because no compute
was requested, approved or executed. Existing S07-A request/results/evidence remain
unchanged. Jobs `341997` (45/62 failure), `342014` (bounded 66/66 pass), S05
`336731` (43/44 negative), and all S03/S04/S01 historical failures/passes retain
their original interpretation; no subset was relabeled.

## Ownership and interpretation limits

All direct S07-B edits are within the launch envelope: owned nuScenes/fusion/config/
training/eval/runtime source, central trainer, `s07_b_*` configs/script, owned tests,
the five imported review paths, and this S07 handoff. Canonical Orchestra files,
S07 `REVIEW.md`, `fl_v3/collab`, `fl_v3/docs`, `fl_v2`, historical configs/scripts
outside ownership and S07-A request/results remain untouched by S07-B direct edits.

Allowed interpretation: exact reviewed worker histories and final review bytes are
integrated; one code-level C/L/F stack candidate exists at `e3cedfa...`; static
syntax/schema/template-fail-closed checks pass; actual runtime gates are explicitly
assembled but unexecuted.

Forbidden interpretation: S07-B/full-stack/production/checkpoint/throughput/memory/
model-quality PASS; a real resolved candidate config; permission to materialize
full cache or run any model/data/profile/metric job; mAP/NDS/fusion gain; FL,
attack/defense, generalization, scientific or publication evidence.

Next action is S00 completeness audit followed by an independent S07-B-R from the
exact returned worker SHA. The reviewer must inspect actual diffs/topology and may
not substitute this worker self-assessment for a verdict.

---

## S07-B-R scoped remediation after `bcffdece`

Independent S07-B-R reviewed delivery
`df13025bc6582b9b436d1df065de75c03e92782d` and returned
`CHANGES-REQUESTED` in review commit
`bcffdece226e73207509ca86540443e7640fb6c5`. The implementation/test remediation
commit is `edc12d87b4e00e11cfdac52a7bbaab02d600bcae`. This section supersedes only
claims explicitly corrected below; the original topology, imported review bytes,
negative results, and NOT-RUN boundaries remain unchanged.

### Finding-to-remediation map

1. **P1 strict official-evaluation caller — remediated in owned scope.**
   `centralized_train.py` now re-loads the exact completed checkpoint through
   `load_checkpoint()` before evaluation, requires a hash-bound `raw`/`ema`
   checkpoint policy, constructs the complete resolved validation token set, calls
   `decode_eval_set` exactly once, rejects missing/duplicate tokens and decoded
   counts above the official 500-box cap, and then calls the official
   `DetectionEval.evaluate()` seam through `run_detection_eval`. The submission
   binds the physical checkpoint SHA-256, resolved config, train/val cache,
   manifest, actual model mode, checkpoint policy, and canonical runtime-dependency
   manifest identity. `evaluation.timing` is carried by `to_run_config()` and is
   therefore part of the canonical config hash; it controls only timing collection.
   Caller-level hostiles cover single traversal, token completeness, duplicate
   tokens, cap rejection, exact load invocation and provenance injection. No metric
   or model execution was performed.

2. **P1 six-task caller inventory — owned callers remediated; out-of-scope callers
   remain an explicit integration blocker.** The strict caller and owned
   `training/tasks.py`/`detection_eval.py` paths consume the six-task decode without
   a legacy `max_objects` override. A repository inventory still finds the following
   reachable, unowned paths and they were not edited:

   - `fl_v3/src/fl_v3/attacks/fusion_ablation.py`: passes `max_objects` to the
     six-task decoder and reads legacy top-level head fields;
   - `fl_v3/scripts/arrhenius_mini_matrix.py`: calls dict `.get()`/`.items()` on the
     six-task list;
   - `fl_v3/scripts/t4_readiness_eval.py` and
     `fl_v3/scripts/t5_attack_eval.py`: load complete S06 checkpoints as bare model
     state and T5 also reaches `fusion_ablation.py`;
   - `fl_v3/scripts/_t4_fd_diagnose.py`,
     `fl_v3/scripts/t3_trainval_reeval_fullval.py`,
     `fl_v3/scripts/p3_crt_probe.py`, and
     `fl_v3/scripts/p3_grad_conflict.py`: historical direct-checkpoint consumers
     requiring a separate live/dead and checkpoint-contract audit.

   **Exact ownership-expansion proposal:** authorize a follow-up commit over the
   four primary paths `fusion_ablation.py`, `arrhenius_mini_matrix.py`,
   `t4_readiness_eval.py`, and `t5_attack_eval.py`, plus focused
   `test_s07_b_*.py`, to remove `max_objects`, consume `task_outputs`, use the S06
   raw/EMA checkpoint loader, and add real six-task mini-telemetry/T5-condition
   tests. Separately authorize inventory-only inspection followed by either explicit
   fail-closed retirement or migration of the four historical diagnostic/P3 paths.
   Until that expansion and independent review, T4/T5/mini-matrix readiness remains
   **NOT ESTABLISHED**; no comment or handoff wording declares those callers dead.

3. **P1 executable dependency identity — remediated structurally, actual GH200
   identity NOT RUN.** `s06.v1` now requires Torch executable-build SHA-256 and
   exact source Git SHA in addition to the version. Runtime identity binds
   `torch.version.git_version`, CUDA/build config, and a per-file executable
   artifact manifest. Torch/spconv/cumm manifests include stable sorted Python,
   native and generated executable suffixes (`.py`, `.so`, `.pyd`, `.dll`,
   `.dylib`, `.cubin`, `.fatbin`, `.ptx`), exclude interpreter bytecode/cache and
   unrelated data, and record every path/size/SHA-256 plus actually loaded module
   origins. spconv/cumm explicitly import their production submodules, require
   pre/post-import file-set/hash equality, preserve exact clean source identity,
   and fail if loaded Python/native code is outside attested roots. A real temporary
   distribution fixture proves file mutation changes the digest and an injected
   outside-root native origin fails closed; the older fully mocked comparison test
   is retained only as schema-drift coverage. Candidate templates contain explicit
   invalid Torch/build/policy sentinels pending an approved GH200 attestation.

4. **P2 PID fallback — remediated.** Registered after-fork and PID-change fallback
   now share `_reset_after_process_change`, which forgets inherited SQLite/native
   descriptors without invoking inherited SQLite state and resets locks, archive
   names, location cache, aggregate counters and modality counters. A raw
   `os.fork()` hostile bypasses the multiprocessing hook and asserts child-local
   zero counters/cache while parent counters remain unchanged.

5. **P2 disabled-modality augmentation — fail closed before iteration.** Dataset
   construction rejects camera-only plus LiDAR GT-paste, camera-only plus the
   LiDAR-scene BEV augmentation, and LiDAR-only plus nonzero image flip. Fusion and
   LiDAR-only scene/GT-paste combinations retain their existing compatible
   semantics. Hostile tests construct no sample before rejection. `augment.py` and
   `gt_paste.py` were not edited.

6. **P3 coverage/wording — corrected.** New
   `test_s07_b_data_lifecycle.py` explicitly enumerates
   `depth={1,10} × backend={directory,ZIP} × mode={camera_only,lidar_only,fusion}`
   and checks enabled/disabled payload presence and exact camera/LiDAR counters.
   The raw-fork lifecycle and construction-failure hostiles are separate, explicit
   cases. Existing S01 depth-10 fork/spawn worker tests remain historical evidence;
   no claim is made that every new cross-product case has been executed under every
   worker start method.

### Remediation checks and immutable negative boundary

Actually run locally after the implementation commit, with no Torch import, data,
model, CUDA or pytest execution:

- `git diff --check`: PASS for remediation-authored changes;
- `python3 -m py_compile`: PASS for all changed Python source and tests;
- `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
- `python3 -m json.tool` on all five candidate templates: PASS;
- `run_s07_b_static_checks.sh`: PASS (compile, JSON, fail-closed template loading,
  candidate hashes only).

Explicitly **NOT RUN**: pytest; actual raw fork test; directory/ZIP/depth/mode test
matrix; Torch/spconv/cumm import or native/build identity on GH200; checkpoint
load/model construction; official devkit round trip; any Slurm/GPU/model/data
execution; full cache; 100/1000-step; profiling/metrics/DDP; rerun or scientific
cell. No `RUN_REQUEST.md` was created or changed, and no compute, merge to
`v3-ad-perception`, push or upload occurred.

The remediation is ready for S00 completeness review and a fresh independent
S07-B re-review. It is not a worker PASS. In particular, the ownership-expansion
caller block and actual GH200/runtime gate remain open; production/full-data,
metric, model-quality, fusion, FL, attack/defense and publication interpretations
remain forbidden.

### Owner-approved caller ownership expansion

After the scoped remediation above, the owner explicitly approved the exact
ownership expansion proposed in finding 2. The expansion began from clean
`codex/s07-b-integrated-cl-stack@9d9f21f2043139bbc05082acc156ba25c127ca57`
with no index lock. It authorized edits only to the four primary callers, the four
named historical callers, `test_s07_b_*.py`, the existing S07 static launcher and
this handoff. It did not authorize compute, protocol/model/head/NMS/metric changes,
RUN_REQUEST creation, merge or push. Expansion implementation commit is
`4ce2366df2925161adae8fea393d5fca64836d40`.

The earlier “ownership-expansion caller block” is now superseded as follows:

- `fl_v3/src/fl_v3/attacks/fusion_ablation.py` no longer accepts or forwards a
  legacy `max_objects` override. All condition decodes use the reviewed per-class
  K=500/no-secondary-task-K six-task decoder. The shared cond-4/cond-5a path reads
  exact `task_outputs`; camera-only invariance compares every branch of all six
  task heads.
- `fl_v3/scripts/arrhenius_mini_matrix.py` validates six task dictionaries and
  records heatmap telemetry plus branch deltas per task/branch. It no longer calls
  dict methods on the task list or silently drops five heads.
- `fl_v3/scripts/t4_readiness_eval.py` and
  `fl_v3/scripts/t5_attack_eval.py` reject bare/legacy checkpoint payloads. They
  inspect only enough payload metadata to reconstruct the embedded exact
  `ResolvedConfig`, verify caller/config drift, physical cache/manifest identity
  and actual runtime dependency identity, build matching model/optimizer/scheduler/
  scaler/EMA components, and then perform the state transition through the S06
  `load_checkpoint()` path. The embedded hash-bound raw/EMA policy selects the
  evaluated weights. Physical checkpoint, runtime-dependency, resolved/data and
  actual-mode identities propagate into official submissions. Mode-aware subset
  datasets bind exact depth, ZIP manifest and C/L/F mode.
- T5 `null-verify` now explicitly refuses to reinterpret the frozen legacy
  bare-state checksum as a complete S06 checkpoint checksum. A new scientific null
  identity requires a future owner-frozen protocol; no checksum definition was
  silently changed.

Repository call/launcher inventory found the four historical consumers only in
old executed collab evidence and legacy launcher paths. Their assumptions are
structurally incompatible with the current stack: `_t4_fd_diagnose.py` and
`t3_trainval_reeval_fullval.py` consume bare checkpoints; `p3_crt_probe.py` trains/
saves legacy bare state; `p3_grad_conflict.py` assumes one
`head.heatmap.weight[10,...]` rather than six task heads. Each now fails explicitly
at its executable entry before data/model/checkpoint use, with the historical
reason preserved in its message. Existing collab findings, old raw outputs and
negative/positive results were not edited or deleted. These scripts are retired
for the current S07-B stack, not reinterpreted as current evidence.

Focused authored tests now:

- drive T5 clean and cond-4/cond-5a helpers with an actual six-task tensor
  structure and assert no decoder receives `max_objects`;
- exercise mini-matrix telemetry/delta over all six task outputs;
- execute T4/T5 loader seams with a syntactically real resolved config and assert
  one S06 `load_checkpoint` call plus config/checkpoint/dependency provenance;
- AST-inventory every primary decoder call and assert historical callers contain
  an explicit fail-closed entry.

Expansion checks actually run: `python3 -m py_compile` over all nine changed
Python/test files, `bash -n run_s07_b_static_checks.sh`, `git diff --check`, and the
complete static launcher (compile + JSON + template refusal + candidate hashes),
all PASS. Pytest and every authored focused test remain **NOT RUN**. Also NOT RUN:
Torch/spconv/cumm imports on GH200, checkpoint/model/data construction, raw/EMA
loads, directory/ZIP payload reads, T4/T5 condition decode, official devkit,
Slurm/GPU/model steps, profile/metrics, full cache, 100/1000-step, DDP, rerun or
scientific cells. No `RUN_REQUEST.md` was created or changed.

The code-level caller ownership blocker is closed for re-review, but this remains
a static candidate only. Independent review and an exact future O-009 request are
still required before runtime evidence; all production/scientific interpretations
remain forbidden.

### S07-B-R2 sole-P1 remediation — authoritative T5 preflight

Independent S07-B-R2 reviewed exact parent
`ee5210016b072041db4956f26834ecfdffcbc206` and returned
`CHANGES-REQUESTED` at review commit
`afb81f51cdf311de215d351e92e2bf5ac6c3bd43`. Its appended review blob is
`40618498861484178a77b9096f8c0e2e79eab550`, SHA-256
`e93daac54472c568a41f06c069cc85216e8cec1914e94be48c5e33dff3c46f8b`;
S00 independently verified its exact parent, sole REVIEW path and preserved prior
47,254-byte prefix. The sole blocking finding was T5 caller ordering: `task_shard`
could seed and construct compatibility data before the complete checkpoint supplied
authoritative precision/depth/mode/cache/manifest identity.

S00 returned this exact P1 within existing O-037 ownership. Implementation commits
`2c6203c02f118678dcfb71e3b67ddc703dbd2f8a` and
`9403178ac2833e5e11e641223b728c6fa168657f` change only
`fl_v3/scripts/t5_attack_eval.py` and
`fl_v3/tests/test_s07_b_integration.py`:

- `main()` now parses the compatibility/attack-only config and, for every executable
  task except the already fail-closed legacy `null-verify`, completes poisoned and
  optional clean checkpoint preflight before output creation or task dispatch.
- Preflight requires the exact complete S06 field set and schema, resolves and
  canonicalizes the embedded `ResolvedConfig`, verifies duplicated config SHA,
  model mode, precision, data identities and checkpoint identity, checks component
  mapping/EMA presence, hashes the physical checkpoint, verifies physical
  cache/sidecar/manifest identities and obtains actual dependency identity. No
  `_device`, `_seed`, val-info/dataset, model or optimizer construction occurs in
  this phase.
- Missing strict fields in the existing compatibility `t5_attack.json`, including
  `precision`, `det-lidar-sweeps`, `model-mode`, manifest/cache identities and
  `s06-production-runtime`, are filled from the authoritative checkpoint config.
  Any overlapping scientific field outside the explicit
  `batch-size`/`num-workers`/`det-eval-limit` override allowlist must match exactly.
  T5 then forces its already-frozen batch-size-one and full-eval-limit contracts;
  attack-only fields remain separate from S06 model/data identity.
- Poisoned and optional clean checkpoints are both fully preflighted before config
  mutation. They must have identical resolved SHA, raw/EMA policy and runtime
  dependency identity. Failure leaves no registered preflight and no authoritative
  config mutation.
- Every `shard`, `aggregate`, `stealth`, `guards` and `viz` task entry requires the
  registered exact checkpoint preflight before `_device`, `_seed`, data or model.
  `_load_model` consumes that immutable preflight rather than establishing config;
  it rejects post-preflight config mutation and checkpoint byte drift before model
  construction, loads through S06 `load_checkpoint`, applies the bound raw/EMA
  policy and rehashes the checkpoint after load.
- V5/V3 visualization now consumes `model.cfg.bev` from the authoritative model
  rather than a legacy 0.4-m default, without changing model/head/NMS/metric or
  attack thresholds.

New caller-level hostile tests are authored but **NOT RUN**. They start from the
actual compatibility `fl_v3/configs/t5_attack.json` and a synthetic structurally
complete checkpoint, drive the real `main → task_shard` order with only external
runtime/data/model seams mocked, and assert parse/physical/dependency preflight
precedes device, fp32-filled seed, val-info, dataset and model. Separate cases cover
missing checkpoint, legacy/bare payload, caller/config drift, embedded metadata
drift, physical checkpoint mutation after preflight and clean/poison resolved
identity mismatch with no partial config/registry state.

Actually run after the implementation: `python3 -m py_compile` for the two changed
Python/test files, `bash -n run_s07_b_static_checks.sh`, `git diff --check`, and the
committed static launcher; all PASS. Explicitly **NOT RUN**: pytest or any authored
hostile, Torch/spconv/cumm actual import gate, physical cache/data/model/checkpoint
load, raw/EMA execution, directory/ZIP reads, T5 condition decode, official devkit,
Slurm/GPU, profile/metrics, full cache, 100/1000-step, DDP, rerun or scientific cell.
No RUN_REQUEST/RESULTS, canonical, review, collab or protocol/scientific field was
edited; no merge/push/upload occurred.

This closes the R2 ordering P1 at static code level only and returns the exact new
delivery for another independent review. It is not runtime, production or
scientific PASS, and no O-009 request should be prepared until that review accepts
the code.

### S07-B-R3 mandatory-control and fan-out artifact remediation

Independent S07-B-R3 returned `CHANGES-REQUESTED` at durable review
`d6f8ae6233c4900e63151d4ee8fab98d549695b8`, whose parent is exact delivery
`b6d132058eee9532b3563d2fe87358be3de6a0a7`. Its REVIEW blob is
`1791a1cfc56fae0f2f3093a733454762c180d335`, size 78,115 bytes, SHA-256
`8c18ed7a4b0a19604fe314b10f6fbe612a2e754e826189b0f57d0c22ab00cfd8`;
the prior 60,954-byte review prefix remained intact. R3 accepted the R2 in-process
preflight ordering but found two scientific-integrity P1s: full five-condition
shards did not require the clean occlusion checkpoint, and cross-process shard
artifacts did not carry/check the preflight/subset identity needed by aggregate.

S00 returned all R3 findings within the existing T5/test/handoff ownership. Scoped
implementation/test commits `cf99ba30c4a2edbeef99af4fc8aee85f87b65bd7`
and `b855a2a3742fdb7729a7d96667ec82e5cb60e855` change only
`t5_attack_eval.py` and `test_s07_b_integration.py`.

#### Pre-side-effect task/checkpoint contract

- Before `os.makedirs`, device, seed, data or model work, `main()` now applies an
  explicit task matrix. Full `shard` requires poison plus clean; `--cond4-only`
  shard is poison-only and rejects a supplied clean checkpoint; aggregate,
  stealth and guards are poison-only; viz conservatively requires both because it
  presents clean comparison; legacy null-verify remains fail closed.
- Every task entry repeats the same matrix assertion before consuming its stored
  preflight. Missing/extra required checkpoints and invalid shard index/count fail
  before output creation.
- The initial checkpoint preflight hashes bytes before and after `torch.load` and
  rejects replacement during deserialization. Existing complete-field/schema,
  embedded mode/precision/data/config/checkpoint identity, EMA, physical data and
  dependency checks remain mandatory.
- Full shard loads the selected raw/EMA poison and clean models immediately after
  preflight/subset validation, computes the actual selected clean trainable-weight
  checksum and requires equality to both the frozen
  `attack-clean-checkpoint-checksum` and subset checkpoint checksum before val-info,
  dataset construction or condition evaluation. The mandatory full path never
  calls `evaluate_target` with `clean=None`; an unavailable target or non-boolean
  occlusion control fails instead of becoming a false zero. Cond4-only output uses
  a separate two-condition result shape with no occlusion field.

#### Versioned shard artifact and aggregate contract

- Full and cond4-only artifacts have distinct schemas
  `s07b.t5.shard.full.v1` and `s07b.t5.shard.cond4.v1` and distinct filenames.
  Every artifact binds mode, poison and optional clean physical checkpoint SHA-256,
  resolved SHA, selected raw/EMA policy, runtime-dependency SHA, actual selected
  weight checksum, frozen subset content hash, shard index/count and results.
- Aggregate accepts only full-schema artifacts and exact-matches each poison bundle
  to the current process preflight/selected model. It requires the artifact clean
  resolved/policy/runtime identity to match poison and its actual selected checksum
  to match the frozen clean checksum. Cond4-only, stale subset/config/checkpoint,
  mixed policy/runtime or malformed/unknown fields fail closed.
- The declared `--num-shards` fixes the exact artifact count and required index set.
  Byte-identical/repeated artifacts, duplicate indices, duplicate target rows,
  wrong shard assignment, and missing/extra frozen targets fail. Each shard must
  contain exactly its deterministic frozen-target slice; the union must equal the
  unique frozen target set.
- Every full row must be evaluated, contain exactly all five boolean disappearance
  conditions and a mandatory boolean `occlusion_disappeared`. Aggregate never uses
  truthiness/defaulting for a missing control and therefore cannot convert `None`
  to a passing 0/N occlusion rate.

#### Corrected hostile coverage

The caller-order hostile still reads the repository's real compatibility T5 JSON,
but now creates an explicit nonempty matching synthetic caller dataroot, cache and
manifest. It executes real `main → task_shard → _load_model` control flow: only
external physical/dependency/data/model/checkpoint boundaries are instrumented.
Sentinels require poison and clean parse/identity checks before the first
`os.makedirs`, fp32 authoritative seed before data, and both real stored-preflight
model/load-checkpoint consumptions plus selected-clean checksum before val-info and
dataset.

Additional authored hostiles cover missing full-shard clean, the complete task
matrix, correct-schema missing/extra fields, wrong schema, independent model-mode,
precision, data-identity, checkpoint-identity and EMA drift, clean policy/runtime
drift, post-preflight config/file mutation, file replacement during initial
`torch.load`, full/cond4 mixing, missing boolean occlusion, repeated artifact/index/
row and missing shard. These tests are authored evidence only and were **NOT RUN**.

Actually run: `python3 -m py_compile` for the two changed files, `bash -n` on the
static launcher, `git diff --check`, and the committed static launcher; all PASS.
Explicitly NOT RUN: pytest, any hostile, actual checkpoint/cache/dependency import,
raw/EMA/model/data/directory/ZIP work, T5 shard/aggregate/control evaluation,
official devkit, Slurm/GPU, full cache, 100/1000-step, profile/metrics/DDP, rerun or
scientific cell. RUN_REQUEST/RESULTS/canonical/review/collab/model/head/NMS/metric/
protocol were not edited; no merge/push/upload occurred.

This is a new static delivery for independent review, not code-level self-PASS or
runtime/scientific evidence. No O-009 request should be prepared until an
independent reviewer accepts the exact delivery.

### S07-B-R4 fresh-run and sibling-artifact remediation

Independent S07-B-R4 reviewed exact parent
`098cfded362ec276d3e697e9150cd7f05de3e238` and returned
`CHANGES-REQUESTED` at durable review commit
`a1452e095ee88a0570580a612f31108aa4b9db30`. The R4 REVIEW blob is
`e8f3a818cfc892b1e2a136c7c4edaf525b898bf1`, size 94,127 bytes, SHA-256
`f10e19a51502547be1a24658d7466b3fdef1820bef3c84ca1552f18f1ca65777`;
the prior 78,115-byte review prefix remained intact. R4 retained the R3 mandatory
clean/occlusion/shard-identity closures but found that the final conjunction could
still accept stale unbound stealth/guard siblings, viz silently accepted
`--cond4-only`, aggregate did not require the canonical shard filename set, and
the hostile matrix did not support the prior completeness wording.

S00 returned those findings within the existing T5/test/handoff ownership. Exact
implementation/test commit
`efe9e7d46df3ef9feec627cf205dc197559886f7` changes only
`fl_v3/scripts/t5_attack_eval.py` and
`fl_v3/tests/test_s07_b_integration.py`. Their exact committed file SHA-256 values
are respectively
`517214daf5b3c37cd6342ff9aa52f4f9750ca55e0c0ad241ffb32a796698c199`
and
`dfba126886c7f1cf4ee5d0fd09d36a841588b14de10ea6ee3dec64b46fc12776`.

#### Immutable run/fan-out identity

- Every non-null T5 task now requires an explicit nonempty safe `--run-id` and the
  frozen subset. All task outputs live under `OUTPUT_DIR/RUN_ID`; old favorable
  fixed-name files in `OUTPUT_DIR` are never consulted.
- The first clean+poison full-shard or viz invocation atomically creates
  `t5_run_manifest.json` with exclusive-create semantics. The exact-schema
  manifest binds run ID, complete poison and clean artifact identities, frozen
  subset content hash, frozen selected-clean checksum and task plan including the
  declared full-shard count. Poison-only shard/aggregate/stealth/guards invocations
  require that existing manifest and must exact-match it. A selected clean checksum
  or clean/poison resolved SHA, raw/EMA policy, or runtime dependency mismatch
  fails before data/evaluation output.
- The complete checkpoint artifact identity is the physical checkpoint SHA-256,
  resolved SHA-256, hash-bound raw/EMA selection policy, runtime-dependency
  SHA-256 and actual selected trainable-weight checksum. Shards, stealth, guards
  and visualization binding artifacts carry the run ID and physical manifest
  SHA-256 in addition to their current checkpoint/subset identities.
- JSON result artifacts use exclusive creation and cannot overwrite an existing
  result. Stealth devkit and viz output directories are reserved exclusively
  before their first output. A partial/old run therefore requires a new run ID;
  it cannot be silently relabeled as a fresh current run.

#### Exact sibling, filename and task contracts

- Stealth and cond-5a guards now have distinct versioned exact-key/type schemas.
  Both bind the current selected poison identity and frozen subset; guards also
  record exact nonnegative counts and its subset-bound camera-only/LiDAR-invariance
  metrics. Aggregate exact-matches both complete identities and schemas before
  reading any raw control metric. Missing, favorable-but-stale, mixed-checkpoint,
  mixed-subset, wrong manifest/run ID, selected-checksum, unknown-key or wrong-type
  artifacts fail closed.
- Aggregate derives the exact canonical full-shard path set
  `ablation_shard_{i}_of_{K}.full.json` for every `i=0..K-1` and requires byte-for-
  byte set equality. Every additional `ablation_shard_*` path is rejected,
  including cond4 controls, old counts, aliases and renamed valid copies. Existing
  exact internal schema/index/target/identity/mandatory-occlusion checks remain.
- The non-shard `--cond4-only` rejection now occurs before viz and every other
  non-shard return. The exact matrix is: full shard poison+clean, cond4 shard
  poison-only, aggregate/stealth/guards poison-only, viz poison+clean, and null
  remains preflight/output-free fail-closed. Invalid shard index/count and missing
  run/subset identity are rejected before checkpoint preflight or output creation.

#### Corrected authored hostile scope and static evidence

The expanded authored tests cover illicit clean on cond4; clean on each poison-only
task; viz missing clean and viz-cond4; invalid shard index/count; null failure before
preflight/output; exact canonical shard filenames; cond4/old-count/renamed extras;
selected-clean checksum, subset, exact-key and mandatory-control drift; stale/mixed
stealth/guard poison, subset and type fields; immutable manifest poison/task-plan
drift; exclusive no-overwrite; and raw plus EMA caller ordering where selected EMA
state is applied before the actual selected checksum and data work. These remain
authored/static evidence: no pytest case was executed.

Actually run after the exact implementation commit, without data/model/GPU/Slurm:

- `python3 -m py_compile` on the changed T5 source and integration test: PASS;
- stdlib AST parse of both changed files: `AST_OK=2`;
- `git diff --check`: PASS;
- `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
- the complete static launcher (compile, five JSON parses, fail-closed template
  loading and unchanged candidate hashes): PASS.

Explicitly **NOT RUN / NO IMPLIED PASS**: pytest or any authored hostile; actual
raw/EMA checkpoint, cache/manifest/data/model/dependency import; directory/ZIP;
fork/spawn/persistent workers; T5 shard/aggregate/stealth/guards/viz; five-condition
or occlusion controls; official devkit; CUDA/Slurm/GPU; full cache; 100/1000 steps;
profile/metrics/DDP; retry/rerun or scientific cell. No RUN_REQUEST/RESULTS,
canonical, review, collab, model/head/NMS/metric/protocol file was changed, and no
merge to `v3-ad-perception`, push, upload or publication occurred.

This remediation closes the exact R4 static findings in owned code and returns a
new immutable candidate for independent review. It is not worker/code-level PASS,
runtime readiness, production/full-data evidence or scientific evidence. No O-009
request should be prepared until an independent reviewer accepts the exact new
delivery SHA.

### S07-B-R5 guard-plan, atomic-publication and contained-run remediation

Independent S07-B-R5 reviewed exact parent
`464281defc8c30f3099aa5e5e827fc907049255b` and returned
`CHANGES-REQUESTED` at durable review commit
`2176e8d2e8185af26f27d67a45838528e4390543`. The R5 REVIEW blob is
`78c05b9a1c060c82f3bff59ba2159c4675a3c9a0`, size 105,234 bytes, SHA-256
`30034cc8f649a31d3ad51fc52d1055bfc48cca8449f41fe9c3e5c5daf6d70dd2`.
R5 retained the R4 checkpoint/subset sibling, viz task-matrix and canonical shard-
set closures, but found that the existing guard sample prefix was not frozen in
the plan, manifest naming preceded complete publication, and a pre-existing run-
directory symlink could escape the output root.

S00 returned those findings under O-045 without changing the existing scientific
control. Exact implementation/test commit
`fcf36dd159bf881df300805e0934ce0ca30ea237` changes only
`fl_v3/scripts/t5_attack_eval.py` and
`fl_v3/tests/test_s07_b_integration.py`. Their exact committed SHA-256 values are
respectively
`9aad253918ef7fc47239a7d6570778ba2829302324e557ec4ee0af631ebead52`
and
`f72ed4f42c0e322e36f8f00d053a7d8476ca43066fb2b72516a6c8462b75547e`.

#### Existing guard semantics are now immutable evidence identity

- The guard scientific meaning is unchanged: the declared `--guard-samples`
  count defaults to 40 and selects the lexically sorted prefix of unique sample
  tokens in the frozen subset; all frozen targets belonging to that prefix retain
  their original frozen order. No threshold, recall definition, LiDAR-invariance
  operation, metric, model, head, NMS or protocol field changed.
- Run-manifest schema `s07b.t5.run.v2` now binds the positive declared count, exact
  selected sample-token list, exact selected `(sample_token, ann_token)` list and
  the canonical selection SHA-256. Counts at or below zero and counts exceeding
  the frozen available sample count fail before output. Every sibling invocation,
  including aggregate, must reproduce the exact same guard plan; a one-sample or
  different-prefix invocation cannot reuse the run.
- Guard schema `s07b.t5.guards.v2` carries that exact selection in addition to the
  current poison/run/manifest/whole-subset identities. Aggregate exact-matches the
  full selection before metrics and requires the invariance-check count to equal
  the selected sample count and the recall denominator to equal the exact selected
  target count. Old v1, reordered prefixes, changed target identities, smaller
  favorable slices, unknown fields or mismatched counts fail closed.

#### Complete-before-publish manifest and no stale replacement

- Manifest bytes are serialized into a random mode-0600 private file in the
  already validated run directory, written completely, flushed and `fsync`ed.
  A same-directory hard-link operation then publishes the final manifest name as
  an atomic no-replace claim; the directory is `fsync`ed after publication and
  cleanup. A reader therefore sees either no final name or complete immutable
  bytes, never the creator's partial final file.
- A lost publisher race reads the complete winner and proceeds only on exact
  manifest equality. A different winner fails; no path overwrites it. Handled
  write/link failures remove the private temporary. A later successful publisher
  also removes abandoned matching private temporaries; concurrent losers whose
  temporary was removed recognize the already published final as a lost race and
  still exact-match it. A pre-existing partial final remains invalid and is never
  replaced.
- A directory with no complete manifest may be initialized only when it contains
  no non-private run artifacts. Old favorable sibling/shard/output content is
  rejected as a stale mixed root rather than adopted by a new manifest.

#### Dirfd/no-follow containment for every run artifact

- Non-null tasks retain the lexical safe-run-ID requirement. The declared output
  root must itself be a real directory rather than a symlink. It and the direct
  run child are opened with Linux `O_DIRECTORY|O_NOFOLLOW`; the open run descriptor
  is verified as a directory and its `/proc/self/fd` resolved parent must equal
  the validated output root.
- All manifest, shard, stealth, guard and aggregate reads/writes now use direct-
  child names relative to that held run descriptor. Ablation, stealth-eval and viz
  subdirectories are likewise opened with no-follow dirfd operations. External
  devkit/viz writers receive the held `/proc/self/fd/<dirfd>` directory, so a later
  pathname substitution cannot redirect them.
- Missing poison-only roots, symlinked output/run/subdirectories, unsafe or missing
  run IDs, path traversal, pre-existing stale output directories and symlinked
  artifact names fail before artifact consumption or creation outside the declared
  root.

#### Authored hostiles and checks actually run

Focused authored tests now cover: declared guard count 40 versus 1; zero and above-
available counts; exact prefix, reordered sample list and changed target identity;
v1 guard rejection and metric count equality; a partial private writer followed by
an exception; interleaved reader before atomic publication; complete publication;
different lost-race winner/no overwrite; abandoned-private cleanup; output-root and
run-root symlinks; missing/unsafe/traversal run IDs; missing poison-only root;
partial final manifest; and favorable stale root content. These tests are authored
but were not executed with pytest.

Actually run after the exact implementation:

- `python3 -m py_compile` on the two changed Python/test files: PASS;
- stdlib AST parse of both changed files: `AST_OK=2`;
- `git diff --check`: PASS;
- `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
- an AST-extracted stdlib-only execution of the exact directory/publication/
  selection helpers exercised real dirfd creation, complete publish/read,
  no-replace winner preservation, abandoned-temp cleanup, run-symlink rejection
  and sorted-prefix target identity: `STDLIB_HELPERS_OK`.

Explicitly **NOT RUN / NO IMPLIED PASS**: pytest or any authored hostile; the full
static launcher or production-package import; Torch/spconv/cumm; checkpoint parse
or raw/EMA load; cache/manifest/data/model; directory/ZIP workers; T5 shard/
aggregate/stealth/guards/viz; official devkit; CUDA/Slurm/GPU; full cache;
100/1000 steps; profile/metrics/DDP; retry/rerun or scientific cell. No
RUN_REQUEST/RESULTS, canonical, review, collab, model/head/NMS/metric/protocol file
was changed, and no merge to `v3-ad-perception`, push, upload or publication
occurred.

This closes the exact R5 findings at static/helper level and returns a new
immutable candidate for independent review. It is not worker/code-level PASS,
runtime readiness, production/full-data evidence or scientific evidence. No O-009
request should be prepared until an independent reviewer accepts the exact final
delivery SHA.

### S07-B-R6 publisher-ownership and hostile-reachability remediation

Independent S07-B-R6 reviewed exact parent
`8cdeceb4e72042874f6ab5aa8a39e84ab67bf934` and returned
`CHANGES-REQUESTED` at durable review commit
`ef01d1cad73021acb87b01874726b83da6470e84`. The R6 REVIEW blob is
`b7a6450ec618dc5a3f40503d12a3605ed4e7c64d`, size 121,397 bytes, SHA-256
`14dd6749ec63fd473e1818109cd42553127e5e6f10daa9d9407f9c6f132190e1`.
R6 accepted the guard-plan identity, complete final/no-replace publication and
dirfd containment bodies statically, but found that the successful publisher's
wildcard orphan cleanup could unlink another still-live publisher's private temp.
It also required caller-level reachability for the remaining race, symlink, count
and interleaved-target hostiles.

S00 returned those findings under O-047. Exact implementation/test commit
`8a7b60b2dd27b1c7ba72e53ddbe67b278ea2f512` changes only
`fl_v3/scripts/t5_attack_eval.py` and
`fl_v3/tests/test_s07_b_integration.py`. Their exact committed SHA-256 values are
respectively
`19bcc9ccbea89ba363d6a6bee47449448339b1b519f792e6fdfcf99e3d08034d`
and
`3d32ed5bbde9d259ad392133d778f6686253230ca9cac93d407c61060b6f08d5`.

#### Private-temp ownership correction

- Every publisher now unlinks only the exact random private temp name that it
  created. The same `finally` ownership rule applies after a successful publish,
  `FileExistsError` lost race, or handled write/link exception. There is no
  directory-wide wildcard cleanup and no publisher can unlink another live or
  closed publisher temp.
- `FileNotFoundError` for a publisher's own temp is no longer reinterpreted as a
  legitimate lost race. Only the final-name `FileExistsError` produced by the
  no-replace hard link is a lost-publisher result. The caller then reads the
  complete final and requires exact whole-manifest equality; a different winner
  still fails.
- Existing publication semantics are unchanged: each mode-0600 private temp is
  written completely and file-`fsync`ed before same-directory hard-link publish;
  final naming is atomic/no-replace; the run directory is `fsync`ed; a partial
  final is rejected and never overwritten.
- The R5 handoff statement that a later publisher automatically removes abandoned
  matching temps is superseded. A true abrupt process death may conservatively
  leave an unreferenced private temp. It is never parsed as a manifest or artifact,
  cannot overwrite the final name, and does not authorize stale run content. Safe
  orphan reclamation is deliberately left to an explicit maintenance procedure
  with a separately proven ownership/freshness policy; correctness prefers a
  harmless retained orphan over interference with a live publisher.

No guard selection, model/head/NMS/metric/control/protocol or data ownership
semantics changed. Run v2 and guard v2 continue to bind the declared positive
guard count, exact sorted sample prefix, exact frozen-order selected targets and
their selection hash; aggregate retains both exact count equalities before metric
use.

#### Exact authored hostile additions

- A live second-publisher temp remains named and writable through another
  publisher's handled partial-write failure, successful publish and subsequent
  lost race. Each tested publisher removes only its own temp; the fixture owner
  explicitly removes the live temp afterward.
- Two caller-level cases exercise the real `_bind_run_manifest()`
  `published=False` branch: an exact complete winner is accepted, while a complete
  winner with a different selected poison identity is preserved and rejected.
  The monkeypatch only schedules which real `_atomic_publish_json_at()` call wins;
  it does not replace manifest loading, equality or the production branch under
  test.
- Symlink hostiles reach the production no-follow helpers for `ablation`,
  `stealth_det_eval` and `viz` subdirectories and for manifest, canonical shard,
  stealth, guard and aggregate-result artifact names. Each outside target remains
  unchanged.
- Guard artifacts with only `n_invariance_checks` changed or only recall `total`
  changed reach and fail their distinct aggregate validation branches.
- A six-target fixture interleaves multiple targets from three samples. Selecting
  the lexically first two samples proves the sample prefix is sorted while the four
  retained targets preserve their original frozen interleaved order.

These tests are authored but were not executed with pytest.

Actually run after the exact implementation:

- `python3 -m py_compile` on the two changed Python/test files: PASS;
- stdlib AST parse of both changed files: `AST_OK=2`;
- `git diff --check`: PASS;
- `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
- AST-extracted stdlib execution of the exact production helpers kept another
  publisher FD open and writable across complete publish and lost race, preserved
  the exact winner, rejected subdirectory/artifact symlinks and confirmed the
  multi-sample frozen target order: `STDLIB_HELPERS_OK`.

The `py_compile` check generated exactly
`fl_v3/scripts/__pycache__/t5_attack_eval.cpython-39.pyc` and
`fl_v3/tests/__pycache__/test_s07_b_integration.cpython-39.pyc`. After explicit
S00 authorization, only those two ignored files were deleted; neither cache
directory nor any other cache/artifact was removed. Final physical inspection
confirmed that both paths are absent and the worktree has no R6-generated residue.

Explicitly **NOT RUN / NO IMPLIED PASS**: pytest or any authored test; the full
static launcher or production-package import; Torch/spconv/cumm; checkpoint parse
or raw/EMA load; cache/manifest/data/model; directory/ZIP workers; T5 shard/
aggregate/stealth/guards/viz; official devkit; CUDA/Slurm/GPU; full cache;
100/1000 steps; profile/metrics/DDP; retry/rerun or scientific cell. No
RUN_REQUEST/RESULTS, canonical, review, collab, model/head/NMS/metric/protocol file
was changed, and no merge to `v3-ad-perception`, push, upload or publication
occurred.

This closes the exact R6 cleanup-ownership and authored-reachability findings at
static/helper level and returns a new immutable candidate for independent review.
It is not worker/code-level PASS, runtime readiness, production/full-data evidence
or scientific evidence. No O-009 request should be prepared until an independent
reviewer accepts the exact final delivery SHA.

### S07-B-R7 test-only real publisher-race remediation

Independent S07-B-R7 reviewed exact parent
`35a0bdca8af61172722428261024d034ecc97a50` and returned
`CHANGES-REQUESTED` at durable review commit
`e4fa439a5c09447bd8b413682772e81f9998f027`. The R7 REVIEW blob is
`b27655cf7e0cec994aada87010eae0065c5746ce`, size 134,348 bytes, SHA-256
`28164c0f692523ee4920d516ba3030052be8380b2b4cc7d96de036935bfe6f6b`.
R7 found no P0-P2 production defect. Its sole P3 was authored reachability: the
then-current bind-race test replaced the entire publisher helper, performed one
real successful publish and fabricated a `False` return instead of running two
publishers through the real hard-link loser path.

S00 authorized O-049 as test-only remediation. Exact test commit
`dd60326fd424d263ab2733fbb8353fb6a6cbb45a` changes only
`fl_v3/tests/test_s07_b_integration.py`; its committed SHA-256 is
`4349e4734c6161f0ae7bd6b6dc28450d8c9475c9b8fcc6a37462fd948f0ea551`.
No production source changed: `fl_v3/scripts/t5_attack_eval.py` remains exact
SHA-256
`19bcc9ccbea89ba363d6a6bee47449448339b1b519f792e6fdfcf99e3d08034d`.

#### Authored race now uses two real bind callers

- Exact-winner and different-winner cases each launch two real concurrent
  `_bind_run_manifest()` calls against one fresh run directory. Neither test
  replaces `_atomic_publish_json_at()` nor fabricates its return value.
- The only scheduling wrapper calls the real `_write_all()` to completion first,
  records the publisher-owned temp name from its live descriptor, and then waits
  at two barriers. Before either link is released, both callers prove that two
  distinct real publisher temps simultaneously exist as files in the run.
- After both writes, a recording wrapper is installed only around `os.link`; it
  calls the original real hard-link operation unchanged, records actual success
  or `FileExistsError`, and returns/re-raises the real result. Both callers have
  already passed the production Linux-dirfd capability check before this link-only
  observation window.
- The exact-identity race observes one real link winner and one real
  `FileExistsError` loser; both complete bind callers accept the same whole
  manifest. The different-identity race likewise observes one actual winner and
  loser; exactly one caller succeeds and the loser rejects the preserved winner at
  the production different-identity equality check.
- After each race, the final manifest is loaded through the real held-dirfd reader,
  both caller-owned private temps are absent, and no temp ending in `.tmp` remains.

All other R6 subdirectory/artifact symlink, guard-count, target-order and live-temp
non-interference hostiles are retained unchanged. The production publication,
guard, artifact, metric, model, protocol and data semantics are unchanged.

The new test is authored but was not run with pytest. Checks actually run:

- stdlib AST parse of the changed test file: `AST_OK=1`;
- `git diff --check`: PASS;
- `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
- AST-extracted stdlib execution of the exact production bind/publication helpers
  repeated both two-thread races: exact identity produced two successful callers;
  different identity produced one success and one rejection; each observed one
  hard-link success, one real loser and zero final temps:
  `STDLIB_REAL_BIND_RACES_OK`.

Explicitly **NOT RUN / NO IMPLIED PASS**: pytest; `py_compile`; the full static
launcher or production-package import; Torch/NumPy/spconv/cumm; checkpoint,
cache/data/model; directory/ZIP workers; any T5 task/devkit/model/GPU action;
Slurm; full cache; 100/1000 steps; profile/metrics/DDP; retry/rerun or scientific
cell. The two previously authorized generated-pyc paths remained absent throughout
O-049; no new pyc cleanup was needed. No RUN_REQUEST/RESULTS, canonical, review,
collab, production source, model/head/NMS/metric/protocol file changed, and no
merge to `v3-ad-perception`, push, upload or publication occurred.

This closes the exact R7 authored no-mock-bypass finding and returns a new
immutable test-only candidate for independent review. It is not worker/code-level
PASS, runtime readiness, production/full-data evidence or scientific evidence. No
O-009 request should be prepared until an independent reviewer accepts the exact
final delivery SHA.
