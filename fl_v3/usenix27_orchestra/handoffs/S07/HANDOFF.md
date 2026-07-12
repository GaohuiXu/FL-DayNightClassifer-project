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

### S07-B-R8 PASS and bounded runtime request preparation

Independent S07-B-R8 reviewed exact parent
`fdee4ba574587a9974ac6a188f2c011dc4730f75` and returned code-level **PASS** at
review commit `8a144ddaa624f3fd0605c7464eb30c1dcf6a51d9`. Its exact REVIEW blob is
`384a4a531f7967f25c75fc1282e1a7767bd4f97c`, size 145,973 bytes, SHA-256
`bdb4093a526efa22fc3f32bf99e97c5f6264b03e95b5985ee35eacc795f5876f`.
That PASS closes the static code-review sequence; it does not itself establish
GH200/runtime, production, full-data or scientific PASS.

Under O-051, S07-B prepared—but did not submit—one integrated bounded GH200
request. The separation is exact:

- launcher-only executable/archive commit `L`:
  `05b733997968b8217e1fc6dd27c3a4add34f6c98`;
- launcher SHA-256:
  `1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`;
- exact 123-file C-locale list SHA-256:
  `be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a`;
- exact aggregate source-state SHA-256:
  `d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd`.

The list/state identities were computed from an extraction of `git archive L` and
independently reproduced from the ordered immutable Git blobs. The launcher was
the only path changed in `L`; this subsequent delivery commit changes only
`RUN_REQUEST.md` and this handoff. The final docs commit SHA is returned to S00
after commit because it cannot embed its own identity.

The request now has status
`EXECUTED_ONCE_JOB_348557_FAILED_TIMEOUT_APPROVAL_CONSUMED_NO_RETRY`.
Canonical O-052 at exact Orchestra commit
`e71274b1a169c1af92fe638608785a6e479d2b3a` records S00's audit and the owner's
delegated S07-B validation authority. This is explicitly not generic O-009. It
permitted exactly one submission of the exact RUN_REQUEST Section D command.
Canonical O-053 at exact Orchestra commit
`e8ee6461ff543b258ebaf588ff36ca5591277909` records that this approval was
consumed immediately by Job `348557`. Scheduler SubmitTime was exactly
`2026-07-12T12:33:41+02:00`; StartTime was exactly
`2026-07-12T12:33:42+02:00` on Arrhenius node `n30` (Europe/Stockholm, CEST).
There is no alternate invocation, retry, rerun, requeue, replacement, follow-on
or spare job.

The immutable approval tuple is executable/archive `L`
`05b733997968b8217e1fc6dd27c3a4add34f6c98`, launcher SHA-256
`1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`,
123-file list SHA-256
`be3b9157e213b942094d290d403306aa714e82157e36ba92847e32cfef71419a`,
aggregate state SHA-256
`d8c6cc0e20ed0c8ded5a4e13dd3ae52f32a62ebbcfafd2f9cbcd469fc5b87acd`,
the 25 exactly named tests in RUN_REQUEST Section D, literal mini input
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`,
fresh output
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_integrated_05b733997968`,
fresh read-only snapshot
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_integrated_05b733997968`,
and one job/one node/one task/one GH200/eight CPUs/64 GiB/45 minutes. The exact
approved command is the Section D preflight plus `sbatch --no-requeue` command
with the recorded exact exports. Any SHA/hash/test/data/path/resource/command or
acceptance/stop-rule drift invalidates approval and requires a new canonical
decision. The one submission attempt consumed the approval regardless of outcome.

The exact Section D preflight/resources/exports/launcher command was submitted
once and unchanged. Under canonical O-054 at exact Orchestra commit
`91526456ee4d4c9d63835868b055b537d0d6655c`, Job `348557` is terminal
`FAILED 1:0`: Submit/Start/End `2026-07-12T12:33:41+02:00` /
`2026-07-12T12:33:42+02:00` / `2026-07-12T13:18:02+02:00`, elapsed `00:44:20`,
node `n30`, one GH200/eight CPUs/64 GiB, `MaxRSS=10573756K`,
`TotalCPU=01:35.363`, `Restarts=0`.

Exact source/runtime identity passed, including `L`, launcher/list/state hashes,
literal mini root, GH200, `spconv==2.3.8` and `cumm==0.7.13`. Pytest emitted
exactly `3F+4E`, hung without summary and exited `124` under the internal timeout.
JUnit, finalized counts and final `sha256sums.txt` are missing, so acceptance is
**FAIL**. Basetemp points to the truncated persistent-multiworker test name only
as a high-confidence hang-location inference, not a formal attribution. Exact
artifact/log hashes, missing evidence and interpretation limits are in
`RESULTS.md`. O-052 is consumed; no retry, replacement, diagnostic execution or
follow-on is authorized.

The exact launcher selects 25 named test files: the required S02 Gaussian/GPU,
S03 camera, S04 SECOND/fp16, S05 CenterHead/eval/NMS, five S06, two S07-B and nine
shared contract modules. Static AST inventory estimates 177 test functions / 249
collected cases; Job 348557 produced no finalized JUnit, so this estimate is not
an executed test count. Bounded GPU
forward/backward and bounded unit-level optimizer-state/step checks are preserved,
but no trainer, optimizer campaign, 100/1000-step run, full cache, trainval metric
or scientific cell is launched.

The launcher records full source/config/runtime/dependency identity, disables
plugin autoload and cacheprovider, clears `PYTEST_ADDOPTS`, constrains the pytest
subprocess to 42 minutes, and requires positive JUnit count with zero failure/
error/skip. It is designed to emit log/JUnit/exit/source/config/execution
artifacts, generate `sha256sums.txt`, and verify it with `sha256sum -c`. This
attempt stopped before JUnit/counts/final checksums; the exact present/missing set
is recorded in `RESULTS.md`. Any collision, identity, hash, resource,
CUDA/package, pytest, skip, timeout or artifact failure stops with no retry.

Preparation checks actually run, without project import or compute:

- `bash -n fl_v3/scripts/run_s07_b_runtime_tests.sh`: PASS;
- embedded stdlib Python heredoc AST parse: `EMBEDDED_AST_OK=3`;
- five candidate JSON files parsed: `JSON_OK=5`;
- `--print-source-files` static mode returned 123 existing files;
- exact archive and immutable-Git-blob list/state hashes matched;
- `git diff --check`: PASS.

Those preparation checks did not run project imports or compute. The later exact
Job `348557` did run the selected pytest suite with Torch/spconv/cumm and one
GH200, producing the preserved failure above. It did not run a trainer, full
cache/trainval scan, 100/1000-step campaign, profile/metric, DDP, matrix, retry or
scientific cell. No merge, push, upload or publication occurred.

The exact command, resources, tests, artifacts, acceptance/stop conditions and
terminal negative result are frozen in RUN_REQUEST Section D and `RESULTS.md`.
O-052 is consumed. Any diagnostic proposal is a distinct future request and has
no execution authority from O-054 or this handoff.

### O-055 S07-B diagnostic request preparation

Canonical O-055 at exact Orchestra commit
`d56e01d3b80a7dae41f90211c0be9ff565861b85` accepts the durable Job 348557
negative record and permits diagnostic preparation only. It does not approve a
job. This proposal is a distinct attribution harness, not an O-052 retry.

Launcher-only `L` is frozen at
`fd142dc1c247ed527dbf5ddb823576c817dc415a`, parent exact
`d7888a9fef615c83c8d36161bfa6d581a3dc4f0f`, with only new mode-100755 path
`fl_v3/scripts/run_s07_b_diagnostic_tests.sh`. Exact identities are:

- diagnostic launcher Git blob:
  `e41e97d31ff0a4e5555a548a63ac04d656565538`;
- diagnostic launcher SHA-256:
  `d8d7686eb727d4973591cf20186615f6bf2f3bc71ba020dec815c9b6d2d0dc1b`;
- 124-file C-locale source-list SHA-256:
  `40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8`;
- aggregate source-state SHA-256:
  `56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2`;
- parent runtime launcher SHA-256:
  `1b1c45d33b113d0c7d649e51b2ddf98a2d7822eab38d708d4bb0e223b8c334c0`.

The source set is the prior exact 123-file runtime state plus only the diagnostic
launcher. Archive extraction and ordered immutable Git blobs independently
reproduced the 124-file list and aggregate hashes. The five candidate config
hashes and exact ordered 25 test files remain unchanged and are checked in-job.

The request is
`EXECUTED_ONCE_JOB_348818_DIAGNOSTIC_COMPLETE_SUITE_FAIL_APPROVAL_CONSUMED_NO_RETRY`.
Canonical O-056 at exact Orchestra commit
`07ec16f37cbe0816be6ce102350036e8c7511e1e` records S00's independent audit and
approves exactly one submission under the owner's delegated S07-B diagnostic
scheduling authority. Canonical O-057 at exact Orchestra commit
`da9dbb4f643f9ab92f6e979e605e9ef24722963a` records that exact Job `348818` was
submitted at `2026-07-12T13:40:45+02:00`, started at
`2026-07-12T13:40:46+02:00` on node `n412`, and consumed O-056 immediately. The
immutable tuple is `L`
`fd142dc1c247ed527dbf5ddb823576c817dc415a`, launcher/list/state SHA-256
`d8d7686eb727d4973591cf20186615f6bf2f3bc71ba020dec815c9b6d2d0dc1b` /
`40c364201bda63386be614fca3710f62111e6964f9b7fdc1beffef69cb5f05d8` /
`56ddfdc66045548899cdde1ad08f7e394c300a8fc27a6c0aaf6551a8178533b2`,
the exact ordered 25 tests with 120-second isolated and 600-second combined
commands, literal mini root
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini`,
fresh snapshot/output
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s07b_diagnostic_fd142dc1c247` /
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_diagnostic_fd142dc1c247`,
one node/task/GH200, eight CPUs, 64 GiB and 30 minutes, and the exact Section E
preflight/exports/`sbatch --no-requeue` command.

Job `348818` completed its harness but failed the suite. Under canonical O-058
`348f29c3c68243ae6010ea0d017e16850081c43c`, scheduler state was `COMPLETED 0:0`
at `2026-07-12T13:57:16+02:00`, elapsed `00:16:30`, node `n412`, one GH200/eight
CPUs/64 GiB, `MaxRSS=10539927K`, `TotalCPU=05:05.306`, `Restarts=0`.
The one submission attempt consumed approval regardless of outcome. Any tuple
drift invalidates approval; no retry, requeue,
replacement, alternate invocation, follow-on or spare job is authorized. A
completed diagnostic harness is not suite PASS: `diagnostic_complete` remains
separate from `suite_pass`. The approved request uses
the same literal mini root plus output-local synthetic scratch, clears full-data
and ZIP overrides, and creates only fresh
`s07b_diagnostic_fd142dc1c247` snapshot/output paths. The old Job 348557 output
remains untouched negative evidence.

The first stage runs the same 25 files in order as isolated verbose subprocesses,
each with a 120-second GNU timeout, long traceback, 60-second faulthandler,
unique basetemp/log/JUnit/exit artifacts and continue-on-failure/timeout. The
second stage runs the same ordered list once with verbose short tracebacks and a
600-second timeout. Its log preserves the current test and prior FAILED/ERROR
names if it hangs.

The checksummed JSON summary records every isolated attempt and the combined
status. `diagnostic_complete` is separate from `suite_pass`: captured pytest
failures/timeouts do not fail the harness, while identity, missing attempts,
summary/artifact-capture or checksum failures do. Every existing formal output
artifact outside scratch is C-locale sorted into `sha256sums.txt` and verified
with `sha256sum -c`.

Preparation checks actually run, without pytest/project import/compute:

- `bash -n`: PASS;
- embedded stdlib Python heredoc AST: `EMBEDDED_AST_OK=3`;
- five JSON configs parsed: `JSON_OK=5`;
- exact ordered tests: 25;
- static source paths: 124, all present;
- archive and independent Git-blob list/state reproduction: PASS;
- `git diff --check`: PASS.

`shellcheck -x` reported only SC2034 for the readonly shell constant
`PARENT_NEGATIVE_RESULTS_COMMIT`; the same exact SHA is embedded in the execution
identity and the full launcher is source-hash bound. This is a nonfunctional
unused-metadata warning, not a runtime or tuple ambiguity; no other shellcheck
finding occurred. Because `L` is frozen, it was recorded rather than amended.

Explicitly not run: `sbatch`, `srun`, pytest, pycompile, project/package import,
Torch/spconv/cumm, data/model/CUDA/GPU, cache, model step, profile/metric, DDP,
matrix, retry or scientific cell. No fresh diagnostic output/snapshot/log exists.
No RESULTS, production source, tests, runtime launcher or canonical doc changed.
No merge, push, upload or publication occurred. Job 348818 ended with
`diagnostic_complete=true`, `suite_pass=false`: 251 isolated tests yielded 3
failures/94 errors/0 skips, including 90 missing-basetemp-parent launcher errors,
four read-only `./fl_outputs` errors, three genuine failures, and a combined exit
124/no-JUnit fork DataLoader queue hang. All 110 checksum records passed. Exact
details and hashes are preserved in `RESULTS.md`. O-056 is consumed; no retry,
replacement, intervention or follow-on is authorized.

### O-059 scoped runtime remediation delivery

Canonical O-059 `87080fd2306ab7f0961f2e23e405c3c166c49262` authorized the
scoped remediation only. Implementation/test commit `C` is
`bf480ea77ccf9ae8417c3ea58e933701dbc7222a`.

- Production/default `num_workers>0` loaders are source-attested `spawn`: both
  nuScenes `make_loader` default and detection/dummy task loaders enforce it;
  zero-worker loaders pass no multiprocessing context. Explicit `fork` remains
  only a low-level test hook and has no production config surface.
- The ZIP fork lifecycle now enters explicit fork only inside a fresh spawned,
  CUDA-hidden helper, retains two epochs/persistent workers/owner-PID/reopen/read
  assertions, and fails closed on helper timeout/exit. Default loader policy has
  an explicit spawn assertion.
- Session mini cache uses `tmp_path_factory`; model-task configs inject that exact
  path, generic multiprocessing exceptions are no longer skipped, spawn is
  asserted, and a changed-CWD hostile asserts no `./fl_outputs` write. A single
  GH200 hostile explicitly initializes CUDA in-process, then asserts the
  production loader is spawn+persistent and reads across two epochs.
- Legacy message matching now uses exact `6 task dictionaries`. LiDAR OFF locks
  the approved six-task 230-tensor topology, including the named 183-head count
  (legacy 15 plus 168), while preserving no LiDAR-backbone params, fuser width
  144, and the ON-minus-OFF 30-tensor invariant.
- Diagnostic launcher creates and verifies writable `$JOB_TMP/isolated` before
  attempts. This does not retroactively repair Job 348818 evidence.
- Dummy golden `d2d819...` and `training/loop.py` are unchanged. The dummy hash
  drift remains pending fresh GH200 attribution; no golden refresh is allowed.

Exact SHA-256 values: diagnostic launcher `663a98a57f3b66fbaec787b5c710ef180f2c8b0b0b4cc4ed2364dcaeb7e43e4c`;
dataset `719ebf749f777571fffd8a9f8eadfe124792c82cdc8932717c2b1be79c880ca6`;
tasks `b81e3ca2ac4c2b2ba543a5ce83822c59e7e423f70b1211da0de3e51edca80e99`;
conftest `fdaaa3bcd009c37927fef824610f6513c1abfc0b30ecd48f15025abbc58818b0`;
model-task test `c346838ded10b4f5ce523e84e1846ad7666be19f84ec78453a86d39ec00fac2b`;
integration test `2a820847579c5c145693f09f819113f539753c7f30d67c20683e17ba42776e55`;
LiDAR test `7a8c2909712790cb1181539be4f4bab3666c362d39981d280e62c5596064cde3`;
ZIP test `3a24613ede41cec47e5197374e2a7f0c4dc0dbd4335c9dbaef9c401c70c223d7`.

Actually run: stdlib AST parse for seven changed Python files, diagnostic
launcher `bash -n`, `git diff --check`, owned-path/source audit: PASS. Explicitly
NOT RUN: pytest, pycompile, project/package import, Torch/CUDA/spconv/data/model,
Slurm/GPU/compute. No RUN_REQUEST/RESULTS/runtime launcher/config/canonical or
forbidden production file changed. No compute, merge, push or upload occurred.

### O-060 bounded dummy-attribution request preparation

Canonical O-060
`34ee1f9672df5c907881b5c6335b6be6e204c156` permits request preparation only.
Launcher-only commit `L` is
`a9d657aebfb0f64d271fa74e312d6054eca57e1d`, parent exact
`c69befe5e8dd6397059c4d3fe1cbf906a9646836`, with only new executable path
`fl_v3/scripts/run_s07_b_dummy_attribution.sh`. Launcher Git blob/SHA-256 are
`295610fd422f3b371b8fd85e54785919903dc332` /
`bbc1293a42034540327402a5df6c1f172b76afacca7906b4f0b71f5290b5968a`;
the bound environment-bootstrap SHA-256 is
`f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf`.

The prepared job archives the fixed pre-S06
`968d81583c87ba76b7dbbb722760f8eb8eb6cd39` and current
`c69befe5e8dd6397059c4d3fe1cbf906a9646836` commits into distinct fresh
read-only snapshots. Their 78/85-file source-list/state SHA-256 tuples are
`0ec5e43e98ee6b98c949d6c3187c4484fb47842443fd3ffe54fcb61e4d777ae0` /
`dc2144cc522d20035eeff81269e45973312c10f908ddc6808bc3c2611b38c93d`
and
`104a647441ce712e83c20d32372944e48777913cf746ae19ee124894ca927e41` /
`0f2995fca7d323421e46326493b2f9cc5d0032ca3d794d422779fc10c626ee32`.
Each snapshot executes the exact seed-42 CPU dummy clean round twice in fresh
independent Python subprocesses under one locked GH200 dependency environment.

Canonical O-061 at exact Orchestra commit
`7ee0b040787bef1c26c1c3d15b0983824d42770e` changes only the approval state to
`APPROVED_ONCE_PENDING_SUBMISSION`. It approves exactly one submission of the
unchanged Section F tuple. Executable/snapshot/source/launcher/bootstrap/
dependency/workload/repetition/command/resource/path/artifact/classification or
stop-rule drift voids approval. One attempt consumes O-061 regardless of outcome;
no retry, requeue, alternate invocation, replacement, automatic resubmission or
follow-on is approved.

Canonical O-062 at exact Orchestra commit
`85798992da9837b86b731eb5b2b11ff71c7aa674` records one unchanged exact
submission as Slurm Job `349653`: submitted `2026-07-12T14:23:08+02:00`, started
`2026-07-12T14:23:09+02:00`, active on node `n530`. O-061 is consumed regardless
of eventual outcome. Current status is
`EXECUTED_ONCE_JOB_349653_ACTIVE_APPROVAL_CONSUMED_NO_RETRY`. This record does
not query or interpret scheduler/runtime results. No retry, requeue, alternate
invocation, replacement, automatic resubmission or follow-on is authorized.

The exact approved request uses one node/task/GH200, four
CPUs, 32 GiB, ten minutes, no requeue/retry. It opens no data and performs no
pytest, full cache/trainval, 100/1000-step campaign, metric, profile, DDP, matrix,
or scientific cell. Summary classification is restricted to
`stable_equal_current`, `pre_historical_current_new`, `unstable`, or
`unexpected_stable_pair`; no outcome automatically changes the golden or loop.
Exact command, paths, identities, artifacts and stop rules are in RUN_REQUEST
Section F.

Preparation checks actually run: launcher `bash -n`, stdlib AST compilation of
all four embedded Python heredocs, immutable Git-blob source-list/state
reproduction for both snapshots, launcher/bootstrap hashes, owned-path audit,
and `git diff --check`: PASS. Not run: project/package import, pytest, pycompile,
Torch/CUDA/spconv/cumm, data/model workload, Slurm/GPU/compute. No RESULTS,
production/test/golden/training-loop/config/canonical file changed; no submit,
merge, push or upload occurred.

### O-063 Job 349653 attribution and runtime-aware test remediation

Canonical O-063
`fe22ecca9bfc455c5d63ea3c9c2f4f00907a7609` accepts the exact Job 349653
attribution and authorizes only the scoped test/evidence remediation recorded
here. Results/request commit `R` is
`79be43d3920aead5068cbe90dc998c075be98a6e`; test commit `T` is
`8e2c31b0e220b30c2a0417b7da46e14b03038c08`.

Job `349653` completed `0:0` in `00:01:26` on `n530` with one GH200/four CPUs/
32 GiB, batch `MaxRSS=540M` and `TotalCPU=00:14.815`. All four exact independent
subprocesses—two at pre-S06
`968d81583c87ba76b7dbbb722760f8eb8eb6cd39` and two at current
`c69befe5e8dd6397059c4d3fe1cbf906a9646836`—returned
`4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb`.
The authoritative classification is `stable_equal_current`; all 25 artifact
checksums passed. Summary/identity/attempts/manifest SHA-256 values are
`806afbfd41eabad3d2181c7c829a74f4ded34cef91636b5bdb7018b5fbbc36fc`,
`b66bbc7400aa1acdfbaf059caea44faf3ac7bfb165b3172898a8e4d84462e9e9`,
`dfa41729753866671852071bddfc7539c44408b294c16058aeeb846fd3b15467`,
and `0c74aae4067bab74619269c16b38c8724ce38d56018d6dea035066e78528341c`.
Full raw paths, all 25 artifact hashes, scheduler-log hashes, runtime identity,
and interpretation limits are in RESULTS.

The result attributes the historical/current mismatch to runtime portability,
not S06/current source changes: pre-S06 and current agree under the same frozen
Arrhenius aarch64/CPython 3.11.15/Torch 2.11.0+cu128/NumPy 1.26.4 identity.
Historical `d2d819...` remains an explicit old-environment evidence constant.
The revised test always executes two fresh `run_clean_round` calls, requires two
valid 64-hex checksums and exact same-runtime equality, and additionally requires
exact `4fa463...` only when the full frozen Arrhenius identity matches. Unknown
runtimes are not skipped and cannot be presented as Arrhenius-golden evidence.
The test file SHA-256 is
`b74122011c5f3a06655473d9176611f2a5ab14dd1750f3c1593266fd1d801d52`.

`training/loop.py` retains exact Git blob
`881c070b1ef8affd350144cce33e508a241cf839`; all production files, launchers,
configs, canonical ledgers, REVIEW and unrelated tests are unchanged. Actually
run for remediation: stdlib
source-text compile of `test_model_task.py`, raw-artifact read/hash audit and
`git diff --check`: PASS. Not run: project/package import, pytest, pycompile,
Torch/CUDA/spconv/cumm, data/model workload, Slurm/GPU/compute. Job 349653 is
terminal and O-061 consumed; no retry, requeue, replacement, alternate
invocation, follow-on, merge, push or upload is authorized. The revised test and
evidence require independent review before any focused validation proposal.

### O-065 S07-B R9 scoped test/lifecycle remediation

Canonical O-065 at exact Orchestra commit
`8db33735a797f4b8827645c704a05f35da1eedd4` returned the R9 candidate for a
six-path, test/evidence-only remediation. The implementation worktree preflight
was clean branch `codex/s07-b-integrated-cl-stack` at exact parent
`797aaf4fa8115568692c381489928fb656f5f356`. Test-only commit `T` is
`3f3686c3fbbfd3fb1bb516a9c00f0612d9da0f04`, with that exact parent.

The independent R9 review commit is
`55f19ab1c7ef1188cfa803724b79a79b3a0d0291`; its `REVIEW.md` Git blob is
`9719ff6d35435eac00cf0f194c3032515802f148`, size `164814` bytes, SHA-256
`318a752ec30d5eb9cac07cc8dfec4b42f3f2371944f8ab51edf79c01189f646c`,
with final verdict `CHANGES-REQUESTED` for candidate `797aaf4...`. The reviewer
commit is not merged or cherry-picked into this implementation branch.

R9 findings are closed as follows, subject to a fresh independent review and
later runtime evidence:

- `test_model_overfit.py` and `test_model_viz.py` now inject the exact
  `str(mini_cache_dir)` fixture result. Both change into a fresh `tmp_path`
  before task/data construction and assert that the changed CWD has no
  `./fl_outputs`; the old CWD cache fallback is not restored.
- A real `DummyRegressionTask` two-worker loader directly asserts `spawn`,
  consumes a real batch and explicitly shuts down its workers. Separate direct
  dummy and detection zero-worker cases assert
  `multiprocessing_context is None` and consume batches. These new contract
  tests do not skip.
- Explicit ZIP `fork` remains only the pre-existing low-level lifecycle hook.
  Its outer fresh-spawn helper now enters a new POSIX session/process group and
  sends an auditable ready record proving session ID and process-group ID equal
  the helper PID. Persistent DataLoader workers are explicitly shut down,
  joined and asserted dead; the isolated helper reports their PIDs and asserts
  no live active child remains on the normal path.
- The parent owns all-path cleanup in `try/finally`. A timeout first calls
  `terminate()` and joins, then sends `SIGKILL` to the whole isolated process
  group, uses `Process.kill()` as a still-alive fallback, joins again and
  asserts final exit. Error/assertion paths also kill the isolated group before
  final direct-process fallback. Normal, error and timeout paths close and
  `join_thread()` the result queue and close the reaped `Process`, preventing
  helper/worker/queue descriptor leakage into later pytest cases.
- The LiDAR test module and function name now describe the approved six-task
  topology rather than old byte identity/62 tensors. Executable gates remain
  exact: OFF total `230`, head `183`, approved increment `183 - 15 = 168`, no
  LiDAR backbone, fuser width `144`, and ON-minus-OFF `+30` tensors.

Exact post-`T` SHA-256 values are:

- `fl_v3/tests/test_model_overfit.py`:
  `c57e88e8c3187d08d90dea6bc1e686b972e08961816e7883c7397f51ba20d2ca`;
- `fl_v3/tests/test_model_viz.py`:
  `243b41840ba34d713ec905c3888d8411bab1eb1041c309b38884b80423cfe0c6`;
- `fl_v3/tests/test_model_task.py`:
  `74d61d8122909e5bae63f83a5ad57ca7c4123236fc6a661eeff19ef6a7083e52`;
- `fl_v3/tests/test_nuscenes_zip_dataset.py`:
  `4b385dc4766bf5d83cda986602160e21c0e88db07f7d104e3da57c1fe46d6d03`;
- `fl_v3/tests/test_lidar_backbone.py`:
  `586606dfa592ddfe6f237e0390466bce49bb031325ce31ef875be08fc667b8ee`.

Actually run: stdlib AST parse and source-text `compile()` of all five changed
tests, exact source/fixture/context/cleanup/topology text audits,
`git diff --check`, and owned/forbidden-path plus Git-blob checks: PASS. The
login-node `python` command was absent; the same permitted stdlib check was run
successfully with `python3`. Explicitly NOT RUN: pytest, pycompile,
project/package import, Torch/NumPy/CUDA/spconv/cumm, data/cache/model workload,
Slurm/GPU/compute. No production/training loop/config/launcher,
`RUN_REQUEST.md`, `RESULTS.md`, `REVIEW.md`, canonical ledger or scientific
contract changed. No compute, merge, push, upload or publication occurred.

Forbidden blobs remain exact across O-065 remediation: `training/loop.py`
`881c070b1ef8affd350144cce33e508a241cf839`; `training/tasks.py`
`86ab9d0563e1636d6c4cde06986470d2559f19f7`; nuScenes `dataset.py`
`afd2707d3939d2d76205996fe94d29fcfc4ed5f3`; `RUN_REQUEST.md`
`efa5ce78eac121f2dd3e70ea75ef414023d45d13`; `RESULTS.md`
`b3b80625ef6c38e4d9382e11ded5c8534b5556ae`; local `REVIEW.md`
`cd0e0795402c2892fe199691a6a01f483d6a457f`; runtime launcher
`1e182ebc1fe883ad59702bfeb1b3db110bbf54c1`. This remediation is static/test
candidate evidence only: it is not a post-O-059 integrated runtime PASS, does
not authorize a Slurm job, and establishes no full-data, production or
scientific result.

### O-067 S07-B R10 ready/ACK and all-path cleanup remediation

Canonical O-067 at exact Orchestra commit
`cc1cbdfea8c7f0682b19c155f6c4d7bba15ffa1f` returned the R10 candidate for
a two-path, test/evidence-only remediation. The implementation worktree
preflight was clean branch `codex/s07-b-integrated-cl-stack` at exact parent
`97588f7ad556fe1ce1a5f7bd76cee19e79d16d31`. Test-only commit `T` is
`6782fa19ca2e4c021ac5215c3e85dd939f4296f9`, with that exact parent and only
`fl_v3/tests/test_nuscenes_zip_dataset.py` changed (`633` insertions, `68`
deletions).

The independent R10 review commit is
`786e31dc81a88f8250f2b5176617f1375f2afcee`; its `REVIEW.md` Git blob is
`b100c30123104063b3c1f88a6909008f3b2b888d`, size `175932` bytes, SHA-256
`1f755af1e8811253b0fec332680f06ae43dcc899cd640f4cf147d70f9863900d`,
with final verdict `CHANGES-REQUESTED` for candidate `97588f7...`. The reviewer
commit is not merged or cherry-picked into this implementation branch.

R10 findings are remediated as follows, subject to fresh independent review and
later explicitly approved runtime evidence:

- The fresh-spawn helper calls `setsid()`, synchronously sends exact
  `(PID, SID, PGID, initial_children)` readiness over a duplex `Pipe`, then
  blocks on the matching ACK. The parent validates the complete tuple, proves
  `PID == SID == PGID` again from live kernel state, proves it is distinct from
  the pytest parent identity, and only then assigns the armed process-group ID
  and sends ACK. No unvalidated/OS-only fallback may arm `killpg`; without ACK
  the child cannot enter any fork/DataLoader path.
- Cleanup no longer depends on the helper leader remaining alive: every armed
  group is probed and sent TERM then KILL as needed, joined and audited absent.
  An unarmed pre-ACK helper is terminated/killed only through its exact
  `Process`, which is safe because the ACK barrier forbids descendants. The
  control endpoints, parent get-only Queue endpoints, Queue thread and Process
  sentinel are closed and checked as closed; the reaped Process is explicitly
  closed. Parent PID/SID/PGID identity is checked unchanged.
- `_persistent_lifecycle` preserves the original exception and traceback while
  collecting iterator discovery, worker shutdown/join/liveness, dataset-close
  and GC failures as notes/aggregate evidence. The forced-error result retains
  the primary traceback, cleanup notes and exact real DataLoader worker PIDs;
  the parent independently proves each PID and the final helper group absent.
- Three executable authored hostiles were added without replacing the real
  spawn/fork/group/cleanup path. The pre-ACK ready-window failure proves no
  initial or later descendant, no ACK, no parent-group signal and complete
  resource closure. The post-ACK lifecycle failure enters a real two-worker
  fork DataLoader, preserves the forced primary failure plus forced cleanup
  evidence after real cleanup, and proves all worker PIDs gone. The post-ACK
  hang creates a real raw-fork descendant; group TERM reaps that descendant,
  group KILL removes the deliberately hanging leader, and final PID/group/FD
  audits prove no contamination.
- The normal explicit-fork contract remains two complete persistent-worker
  epochs with deterministic payload equality, worker-local owner/PID,
  reopen/read-count and archive assertions, explicit worker shutdown and no
  surviving active child.

The exact post-`T` SHA-256 of
`fl_v3/tests/test_nuscenes_zip_dataset.py` is
`0c5a4e65403ec37329503aff95c0d07bcc9c5b2dd811c9a0e598c4b4d9e2cca8`.

Actually run: stdlib AST parse and source-text `compile()` of the one changed
test; source-order checks proving full ready validation precedes arm and ACK;
required control/cleanup/hostile token audits; exact one-path ownership and
parent checks; `git diff --check`; Git blob and forbidden-path checks: PASS.
Explicitly **NOT RUN**: pytest, pycompile, project/package import,
Torch/NumPy/CUDA/spconv/cumm, data/cache/model workload, any spawn/fork runtime,
Slurm/GPU/compute. The three new hostiles are authored static evidence only,
not runtime PASS evidence.

Forbidden blobs remain exact across O-067 remediation: `training/loop.py`
`881c070b1ef8affd350144cce33e508a241cf839`; `training/tasks.py`
`86ab9d0563e1636d6c4cde06986470d2559f19f7`; nuScenes `dataset.py`
`afd2707d3939d2d76205996fe94d29fcfc4ed5f3`; `RUN_REQUEST.md`
`efa5ce78eac121f2dd3e70ea75ef414023d45d13`; `RESULTS.md`
`b3b80625ef6c38e4d9382e11ded5c8534b5556ae`; local `REVIEW.md`
`cd0e0795402c2892fe199691a6a01f483d6a457f`; runtime launcher
`1e182ebc1fe883ad59702bfeb1b3db110bbf54c1`. No production/training-loop/
config/launcher/canonical/scientific contract changed; no compute, merge, push,
upload or publication occurred. This is a static test candidate for independent
review only; it is not a post-O-059 integrated runtime PASS and authorizes no
runtime proposal or scientific interpretation.

### O-069 S07-B R11 synchronous-result and leader-dead identity remediation

Canonical O-069 at exact Orchestra commit
`4caf02ad949887b804b875fff972ebdb2c6d7fe6` returned the R11 candidate for a
two-path, test/evidence-only remediation. The implementation worktree preflight
was clean branch `codex/s07-b-integrated-cl-stack` at exact parent
`8469eb4944f164f5bd2fa1aa833ea4df0acf04b3`. Test-only commit `T` is
`2497ac11e807e5b223bfa0eaa2537fcbde1aec88`, with that exact parent and only
`fl_v3/tests/test_nuscenes_zip_dataset.py` changed (`419` insertions, `106`
deletions).

The independent R11 review commit is
`52e05ac0f500f1f671818125dc72caded9c1b4b8`; its `REVIEW.md` Git blob is
`4e0226718109e193bb09993db085422106b1dccc`, size `191368` bytes, SHA-256
`cc8922192125b054280e5b11760f801997adbb201ea2f7bd6e2564b55e0c1104`,
with final verdict `CHANGES-REQUESTED` for candidate `8469eb4...`. The reviewer
commit is not merged or cherry-picked into this implementation branch.

R11 findings are remediated as follows, subject to fresh independent review and
later explicitly approved runtime evidence:

- The child-produced result `Queue`, feeder thread and private queue endpoints
  are removed. Control remains a duplex `Pipe`; results use a separate one-way
  `Pipe`/`Connection`. The parent polls and synchronously receives the complete
  result before joining its producer. The forced-error result carries a 2 MiB
  padding payload, and its hostile requires the exact full length/content,
  primary traceback and cleanup notes, so a pipe-capacity producer/join
  deadlock cannot satisfy the test.
- Linux process identity is exact `(PID, /proc starttime)`. The parser uses the
  final `)` terminating `/proc/<pid>/stat` `comm`, and a hostile covers embedded
  spaces and parentheses. Helper readiness binds starttime before ACK. Real
  DataLoader worker identities are captured while alive, returned across the
  process boundary, and audited by `(PID,starttime)`; PID reuse therefore means
  the original instance is gone rather than falsely live.
- A new post-ACK leader-exit hostile raw-forks a descendant whose transmitted
  record contains PID, starttime, process group and session. The descendant
  inherits `SIGTERM` ignore from before `fork`, eliminating a readiness race;
  the helper restores its handler, sends the complete record synchronously and
  exits via `os._exit(0)` without `waitpid`. The parent first receives the
  record and joins/reaps the leader, then proves the exact leader identity gone
  while the verified group and exact descendant identity remain. Group TERM is
  proven insufficient and group KILL is proven, by bounded polling, to remove
  both the group and original descendant instance.
- The existing live-leader forced-hang path remains. Its raw-fork descendant is
  reported synchronously; TERM removes/reaps the descendant while the helper
  leader remains, so the bounded identity/group state forces KILL. The hostile
  requires the exact helper instance and group to be live after TERM and gone
  after KILL.
- Armed-group TERM and KILL are followed by bounded cleanup, not one-shot PID
  probes. After TERM the direct helper is first given a bounded join so a
  reaped zombie cannot cause a false KILL escalation; the remaining deadline
  polls group existence and every exact identity. After KILL the helper is
  reaped and a five-second bounded poll proves group/identity absence. Cleanup
  probe failures are collected as notes/aggregate evidence and cannot mask the
  original exception.
- All parent control/result endpoints, parent copies of child endpoints, the
  `Process` and its sentinel are explicitly closed and their file descriptors
  checked closed. The ready/ACK validation and arm ordering, pre-ACK hostile,
  forced-error primary preservation, normal two deterministic persistent-worker
  epochs, explicit worker shutdown and CUDA-hidden explicit-fork boundary are
  retained.

The exact post-`T` SHA-256 of
`fl_v3/tests/test_nuscenes_zip_dataset.py` is
`07c4c2159efbdf4fb18a95960d4ff7d8d17ac823c88f14d9184eb1cc041e3f09`;
its Git blob is `f8d4f0ee7a9ca834cbf1105562cf0c8fccb5ec38`.

Actually run: stdlib AST parse and source-text `compile()` of the changed test;
result-transport/private-endpoint text audit; exact ready/ACK, large-payload,
leader-dead, live-leader, starttime, bounded-cleanup and FD token audits;
one-path ownership/parent checks; `git diff --check`; Git-blob and forbidden-
path checks: PASS. Explicitly **NOT RUN**: pytest, pycompile, project/package
import, Torch/NumPy/CUDA/spconv/cumm, data/cache/model workload, any spawn/fork
runtime, Slurm/GPU/compute. These hostiles are authored static evidence only,
not runtime PASS evidence.

Forbidden blobs remain exact across O-069 remediation: `training/loop.py`
`881c070b1ef8affd350144cce33e508a241cf839`; `training/tasks.py`
`86ab9d0563e1636d6c4cde06986470d2559f19f7`; nuScenes `dataset.py`
`afd2707d3939d2d76205996fe94d29fcfc4ed5f3`; `RUN_REQUEST.md`
`efa5ce78eac121f2dd3e70ea75ef414023d45d13`; `RESULTS.md`
`b3b80625ef6c38e4d9382e11ded5c8534b5556ae`; local `REVIEW.md`
`cd0e0795402c2892fe199691a6a01f483d6a457f`; runtime launcher
`1e182ebc1fe883ad59702bfeb1b3db110bbf54c1`. No production/training-loop/
config/launcher/canonical/scientific contract changed; no compute, merge, push,
upload or publication occurred. This remains a static test candidate for fresh
independent review; it is not a post-O-059 integrated runtime PASS and
authorizes no runtime proposal or scientific interpretation.

### O-071 R12 static PASS and focused runtime-request preparation

Canonical O-071 at exact Orchestra commit
`bf7fd65f4b58b6981b0604647489595b4903beaf` accepts independent R12
`49735be` as code-level/static-review PASS with no P0-P3 and authorizes request
preparation only. Candidate
`c53117a889987c3070b60817e52bdb4aac4c9098` remains the exact code/test
candidate; no production, test, config, result, review, or canonical file was
changed by this preparation.

Launcher-only commit `L` is
`c36555fd9c233198b703d73741382960edcb4159`, whose sole parent is the exact
candidate and whose only changed path is
`fl_v3/scripts/run_s07_b_postremediation_focused.sh`. Launcher blob/SHA-256 are
`717ec0869d5c1207bd946fd5f5034390c208623b` /
`b32f78b76f14b8f12957d0132d8739e2ef37691c72684a022688752bb8ff185a`.
The exact 93-file C-locale source-list/state tuple is
`a0b585b40ebfef2167ad6a9e66f3b59ca719e607b01c933b33310d716a6e08a6` /
`0d519ea46dd388f80a41ed96d350e47db837f6be7976e2e583448a2975915861`.

RUN_REQUEST Section G freezes exactly four complete test files—ZIP lifecycle,
model task, LiDAR topology, and visualization consumer—and the one exact legacy
multi-task-loss node. The prepared harness uses a fresh immutable archive,
read-only snapshot, writable output CWD/basetemps, literal mini dataroot,
cleared ZIP/full-data overrides, five bounded verbose/faulthandler pytest
invocations, per-selection JUnit/log/exit/checksums, locked dependencies, final
artifact checksum verification, and suite-level zero fail/error/skip/timeout
acceptance. It excludes the 180-step overfit test, complete suite, full
cache/trainval, 100/1000-step, metrics/profile/DDP/matrix/retry.

Preparation checks actually run: launcher `bash -n`; stdlib AST parse of all
three embedded Python heredocs; exact five-entry/source-list enumeration;
launcher/source/bootstrap hashes; launcher-only parent/diff and owned-path
audit; `git diff --check`. Not run: project/package import, pytest, pycompile,
Torch/NumPy/CUDA/spconv/cumm, data/cache/model workload, Slurm/GPU/compute.
Canonical O-072 at exact Orchestra commit
`bcb45b34246ad52a45f89f6552832b3d3318b292` changes only Section G's
approval state to `APPROVED_ONCE_PENDING_SUBMISSION`. It approves exactly one
submission of the unchanged candidate/executable/launcher/source/dependency/
selection/data/path/resource/artifact/acceptance/stop-rule/command tuple frozen
in RUN_REQUEST Section G. Any drift voids approval before submission. One
attempt consumes O-072 regardless of scheduler or suite outcome; no retry,
requeue, alternate invocation, replacement, automatic resubmission, or
follow-on is approved. This docs-only approval record performs no submission,
compute, merge, push, upload, publication, or scientific interpretation.

Canonical O-073 at exact Orchestra commit
`8e61c05ce19ad7ec4eeb65d63b625f6e79d08ae2` records that the unchanged exact
Section G command was submitted once as Slurm Job `351903` at
`2026-07-12T15:56:43+02:00`, started at `2026-07-12T15:56:44+02:00`, and is
active on node `n424`. Status is
`EXECUTED_ONCE_JOB_351903_ACTIVE_APPROVAL_CONSUMED_NO_RETRY`; O-072 was
consumed immediately regardless of eventual scheduler or suite outcome. No
retry, requeue, alternate invocation, replacement, automatic resubmission, or
follow-on is authorized. This docs-only record does not query or interpret
runtime results and changes none of the exact approved tuple.

### O-074 Job 351903 negative result

Canonical O-074 at exact Orchestra commit
`e18984ac286c7d61170a77d122149ce51de8b57a` accepts the terminal raw-artifact
audit as a preserved negative result. Job `351903` ended `FAILED 1:0` after
`00:09:11` on `n424`; exact source/dependency/mini identity and all 25 produced
artifact checksums matched. The ZIP selection exited `124` after five displayed
passes, a finalized persistent-`fork` failure, and entry into persistent
`spawn`; the model-task selection exited `124` after its runtime-bound dummy
checksum PASS and entry into the real dummy multiworker loader. Neither emitted
JUnit or a finalized traceback/summary. LiDAR, visualization, and the exact
legacy-loss node completed 6/6, 1/1, and 1/1 with zero failure/error/skip.

The authoritative summary is `suite_pass=false`; the three completed entries
must not be extrapolated into a focused runtime PASS. Full scheduler, identity,
per-entry, artifact/hash, and interpretation evidence is in RESULTS. O-072 is
consumed with no retry/requeue/replacement/follow-on. O-074 authorizes only
preparation of a distinct nine-node multiworker diagnostic request; it does not
authorize that compute or any code/test/config/result interpretation change.

### O-074 distinct multiworker diagnostic request preparation

On top of the durable Job 351903 negative-results commit
`0e549ea9c34e8d19d3e55c785ca2c240f475e346`, launcher-only commit `L` is
`4b3c8474a4441a083cc4954c489c48698ee2bf2b`. Its launcher blob/SHA-256 are
`9d8dcc98259c2902443a106970665282b70044b0` /
`b995307e93026c993c3f1b3e4038e637a6a5a9437c0f52fe37b5d483bce81fbe`;
the exact 90-file source-list/state tuple is
`c9e0a4175725e59d1e4e3e3efbe3421c0d9b8480fd5161cf5147ae9184eb511f` /
`f645ae7859d2e3384e805d97bb8256ce6c90cf6540d0ca7538a1518901785d19`.
Candidate-to-`L` has no source/test/environment/dependency-manifest changes, so
the diagnostic remains bound to exact code/test candidate `c53117a...`.

RUN_REQUEST Section H freezes nine exact nodes once each: explicit persistent
fork/spawn, four ready/ACK/error/leader/hang hostiles, dummy multiworker,
detection multiworker determinism, and CUDA-initialized production
spawn/persistent. Each runs in a distinct new session/process group under a
90-second supervisor and 30-second pytest faulthandler. Timeout cleanup is
whole-group TERM, five-second grace, whole-group KILL if needed, plus exact-
identity cleanup of escaped-session descendants. A Linux subreaper runs
fixed-point scan/reap/TERM/KILL cleanup and forbids cross-node survival.

Per-node results preserve failures/timeouts while separately proving log/JUnit/
exit/identity/cleanup/checksum completeness. `diagnostic_complete` is separate
from `suite_pass`; scheduler success cannot substitute for either. The request
is mini-only, one GH200/eight CPUs/64 GiB/20 minutes and excludes overfit/full
suite/full data/cache/100-1000-step/metrics/profile/DDP/matrix/retry.

Preparation checks actually run: launcher `bash -n`, stdlib AST parse of all
four embedded Python programs, exact nine-node/source enumeration, immutable
archive source/hash reproduction, candidate-to-executable runtime-source diff,
launcher/bootstrap/ownership and `git diff --check`. Not run: project import,
pytest, pycompile, Torch/NumPy/CUDA/spconv/cumm, data/model workload, Slurm/GPU.
Canonical O-075 at exact Orchestra commit
`d617b30458401861631da63db1939876deaf5796` changes only Section H's status to
`APPROVED_ONCE_PENDING_SUBMISSION`. It approves exactly one submission of the
unchanged candidate/executable/source/launcher/dependency/nodes/supervisor/
cleanup/data/path/resource/artifact/summary/acceptance/stop-rule/command tuple.
Any drift voids approval before submission. One attempt consumes O-075
regardless of scheduler, harness, or node outcome; no retry, requeue, alternate
invocation, replacement, automatic resubmission, or follow-on is approved.
This docs-only record performs no submission, RESULTS change, compute, merge,
push, upload, publication, or scientific interpretation.

Canonical O-076 at exact Orchestra commit
`8505ceee86e160cdac26055f728f2b8215a4134e` records that the unchanged exact
Section H command was submitted once as Slurm Job `352105` at
`2026-07-12T16:23:21+02:00`, started at `2026-07-12T16:23:23+02:00`, and is
active on node `n424`. Status is
`EXECUTED_ONCE_JOB_352105_ACTIVE_APPROVAL_CONSUMED_NO_RETRY`; O-075 was
consumed immediately regardless of eventual scheduler, harness, or node
outcome. No retry, requeue, alternate invocation, replacement, automatic
resubmission, or follow-on is authorized. This docs-only record does not query
or interpret runtime results and changes none of the exact approved tuple.

### Job 352105 terminal diagnostic and scoped harness remediation candidate

Job `352105` subsequently ended scheduler `COMPLETED 0:0` in `00:09:53` on
`n424`, but its authoritative result is `diagnostic_complete=true`,
`artifact_complete=true`, `suite_pass=false`. Exact raw outcomes are two PASS,
two FAIL, and five timeouts; every supervisor cleanup predicate and all 46
global checksum records passed. `RESULTS.md` preserves the full negative result,
artifact hashes, and interpretation boundary. Scheduler success is not a suite,
runtime, production, or scientific PASS.

Independent raw-artifact reading established two harness confounds. First, the
executed launcher exported a 106-byte output-local temp directory. The
post-ACK-error log directly records four CPython multiprocessing
`OSError: AF_UNIX path too long` failures; all multiworker nodes shared that
environment, so their timeout/failure behavior cannot be attributed to the
candidate. Second, the outer per-node supervisor was itself a Linux subreaper.
After the leader-exit helper died, its resistant descendant was adopted by that
outer supervisor rather than the pytest parent. The inner test could kill but
not reap the adopted zombie, leaving conservative process-group/identity probes
live until outer cleanup. This is a nested-harness ownership conflict, not a
confirmed production-code failure.

Under S00's scoped source/test/evidence remediation authority, exact durable
code candidate `26cffb02ced50b07f93021bc48310efb68b178a9` changes no
production source. Its sole parent is the previous documentation HEAD
`f8b781dd919443fab0d9c2e6e28c0207182800d5`; its exact seven-path diff is
`439` insertions and `18` deletions:

| Path | Git blob | SHA-256 |
|---|---|---|
| `fl_v3/tests/test_nuscenes_zip_dataset.py` | `6c1d53e6aa44a3f7a1c8a0e577ead976ec62d953` | `4874b22d575b731099c56aa67ef488f8e51f03c48e076428b7d957873e3730e7` |
| `fl_v3/scripts/run_s07_b_runtime_tests.sh` | `738219c326bbf81375ecb74a2e132660167b6a43` | `7f60122b4cf84bfd356a957e963d0d73af0a1747b5ee16551a94c455163813d6` |
| `fl_v3/scripts/run_s07_b_diagnostic_tests.sh` | `53a75c302cfcd896035da8c5d8b37ccc805c4c3c` | `2816fbb42cdef927cd6ae12a7e19364ca94b41c4f4ae525d3282021a2782f510` |
| `fl_v3/scripts/run_s07_b_dummy_attribution.sh` | `b4d547fd698afb14a63a1e6b1b3380687ca2750b` | `782a7b08d1d2e7755f4f766073dcc10b29119097e91a4546a98e470e30bedbdd` |
| `fl_v3/scripts/run_s07_b_postremediation_focused.sh` | `02a7e01cca425baa401732d0db6dbd00410a4de6` | `0abb307ca9cb4149388bc6dd6c7fa452eb30158594989149219c9f82bcdd5c20` |
| `fl_v3/scripts/run_s07_b_multiworker_diagnostic.sh` | `0241bcbbdf0a86e9c0a4aaee7c39e7ad64aa1e3f` | `72c5b683d7aa664463396bf96912e072df4d1f153f72a1eee72b1b24167dfa18` |
| `fl_v3/scripts/run_s07_b_static_checks.sh` | `83f75d87af477ca503f069f40328d9319693fcee` | `05e233d36128c17354f8497c75f29ecc967d697471b3b51b80f42e888735d617` |

The exact candidate semantics are:

- all five S07-B runtime/diagnostic/attribution launchers use random
  `mktemp -d -p /tmp` job-unique mode-0700 temp directories capped at 48 bytes,
  with numeric job ID, exact 12-hex executable prefix, anchored basename,
  exact `/tmp` parent, no-symlink, and device/inode validation;
- their EXIT traps preserve the original exit code and recursively remove only
  the exact successfully acquired temp directory when its path and device/inode
  still match. Formal artifacts remain under the separately frozen output root;
- only the leader-exit hostile saves/enables Linux child-subreaper state before
  helper start, proves the exact descendant was adopted by the pytest parent,
  and reaps only that verified `(PID,starttime)` with bounded
  `waitpid(pid,WNOHANG)`. PID reuse, different PPID, or non-child state cannot
  trigger a reap. Restoration of the saved state is attempted on every
  controlled Python path; a passing leader-exit hostile proves successful
  restoration for that execution, while a restore syscall failure remains
  additive cleanup evidence rather than a claim that restoration succeeded;
- static launcher-contract checks cover shell syntax, random short-temp
  construction, trap-before-assertion order, anchored deletion, device/inode,
  length, mode, and post-environment TMP/TEMP exports. The leader-exit hostile
  asserts adoption PPID, exact wait status, and subreaper restoration.

Static checks on the exact committed tree produced:

```text
bash fl_v3/scripts/run_s07_b_static_checks.sh --launcher-contract-only
short TMPDIR contract: 5 launchers OK

python3 source-text compile of test_nuscenes_zip_dataset.py
test source compile: PASS

stdlib compile() of every PY heredoc in run_s07_b_*.sh
embedded Python heredocs: 19 PASS

git diff --check f8b781dd919443fab0d9c2e6e28c0207182800d5 \
  26cffb02ced50b07f93021bc48310efb68b178a9
<empty; PASS>
```

The one transient launcher-contract assertion observed while declarations and
the checker were being edited concurrently was an editing-state race; both S00
and the worker independently reran the stable exact candidate and obtained the
PASS above. It is not runtime evidence and not a candidate negative result.

Historical executable `4b3c8474...`, candidate `c53117a...`, Job 352105 raw
artifacts, and all earlier negative evidence remain immutable. This work has not
run pytest/project import/Torch/CUDA/data/Slurm. Code commit `26cffb02...` is
durable. Its lifecycle delivery is exact commit
`34f07994a4b3de62c7c1331d98ff03dbba98de2e`, whose sole parent is
`26cffb02...` and whose exact three changed paths are this `HANDOFF.md`,
`RUN_REQUEST.md`, and `RESULTS.md`; these files are not an uncommitted delivery.
Independent R13 review commit
`69037534352c4517e93a62b17cd8f168c0f8a24c` is review-only, is not merged,
and returns delivery `34f07994...` as `CHANGES-REQUESTED`. Current compute is
`NOT_APPROVED_DO_NOT_SUBMIT`; neither delivery nor review authorizes compute,
merge, push, upload, or scientific interpretation.

### O-079 scoped R13 evidence remediation — durable code candidate

O-079 code is durable at exact
`56c74de5bdf5463fdd6ab1a623ab0f92a35871ae`, sole parent
`34f07994a4b3de62c7c1331d98ff03dbba98de2e`. It changes exactly seven
test/launcher paths (`163` insertions, `59` deletions), no production source,
and retains Job 352105 and all prior semantics while addressing exactly the
three R13 findings:

| Path | Git blob | SHA-256 |
|---|---|---|
| `tests/test_nuscenes_zip_dataset.py` | `9db60cb07609c51d374973158380b0a003c1b1f8` | `2db06a8e6492b68ac3f645cc9bfc4b6feaa2c588ec6e2ad821b8ca5843241b3d` |
| `scripts/run_s07_b_runtime_tests.sh` | `1eef653ed6ecd8675f7603d7f1ef7771b22724a3` | `f3dc70455a06fa5f77b14232799a4c214dd6c5e38a6c5b2d1635881d2e008d04` |
| `scripts/run_s07_b_diagnostic_tests.sh` | `17dad205039334ed1a34e593c6847db47888c789` | `d1be90179426c135fb97cf57c7f162ae4f7aa77db7b279b0feee4081f4ba3edd` |
| `scripts/run_s07_b_dummy_attribution.sh` | `8755c91fa2493e6255db14bf96c75ac9daffa429` | `81a5ebd51d70c180ff0ac64bf4e8bd153be64b992550601eb96d536c557c4725` |
| `scripts/run_s07_b_postremediation_focused.sh` | `926161c3b8d10bdb7760ce3a7ec2785ed3434405` | `00d9674fcbc01ee9876508a6220e1e97a1b714e32555c863499dacec2cbc2599` |
| `scripts/run_s07_b_multiworker_diagnostic.sh` | `42bb7560d6a04995edb7ae7976906f23e3b9d4f5` | `4b09b6c6ef0f682bdb5326ca23851b45705d636e00fddd0809de74abbc37577e` |
| `scripts/run_s07_b_static_checks.sh` | `c57dc1c98e7bc07538b93f55877c30968d78eeca` | `e814bdbd3ff8c607d9aa61ebcf232b3a588f244a6d10a94e7e63b7e3af559d03` |

- the leader-exit hostile now requires the exact adopted descendant's raw
  `waitpid` status to satisfy `os.WIFSIGNALED(status)` and
  `os.WTERMSIG(status) == signal.SIGKILL`; the report also records the decoded
  signal, and the static checker locks both predicates;
- every one of the five launcher EXIT traps emits exactly one deterministic,
  non-sensitive stderr line with fixed launcher tag
  `S07B_TMP_CLEANUP_FAILURE:<launcher>` and one of
  `path_pattern/dirname/symlink/directory/stat/device_inode/rm` whenever safe
  removal is refused or fails. `stat`/`rm` raw stderr is suppressed so paths do
  not leak. A nonzero primary status remains unchanged; a successful primary
  becomes nonzero on any cleanup failure. The static checker locks all seven
  reasons, tags, redirections, and exit-status rules;
- the committed-delivery and controlled-path restore wording is corrected as
  above. A PASS hostile may prove restore success only for that exact execution.

Exact committed-tree checks were: six shell `bash -n` checks PASS;
`short TMPDIR contract: 5 launchers OK`; changed-test source compile PASS;
all 19 embedded Python heredocs compile PASS; `shellcheck -S error` PASS; and
`git diff --check` PASS. No project import, pytest, multiprocessing,
Torch/CUDA/data/model or Slurm evidence exists for O-079. The code is durable,
and its already-submitted durable delivery is exact
`e3122dbccdd252a6d89f1a4fe339b9043fe19884`, whose sole parent is code
`56c74de5bdf5463fdd6ab1a623ab0f92a35871ae` and whose only changed paths are
this `HANDOFF.md`, `RUN_REQUEST.md`, and `RESULTS.md`. Independent R14
review `9645148d3441a66a373091766c0186ea10243336` is review-only and not
merged; it accepts O-079 code at code-level/static-authored scope and requests
only this delivery-state correction. Corrected multiprocessing/GH200 runtime
remains unverified, and fresh independent acceptance is required before S00
may freeze any new exact runtime request. Current compute remains
`NOT_APPROVED_DO_NOT_SUBMIT`.

### O-083 corrected nine-node runtime request preparation

Independent R15 review
`bc587790ff3b2dfb65b12fa4469c1f5b79aea5fc` is review-only/not merged and
gives `PASS` at docs/static-authored scope for corrected delivery
`65881c5628a737eaeaf4742ab7b11a63b9d3cbc2` and O-079 code
`56c74de5bdf5463fdd6ab1a623ab0f92a35871ae`; runtime remains unverified.
O-083 therefore prepares, but does not approve or submit, a new bounded request
using the already reviewed multiworker launcher without changing any launcher,
test, production source, node, timeout, cleanup, data, or dependency semantics.

The exact request identity is:

- candidate/code `56c74de5bdf5463fdd6ab1a623ab0f92a35871ae`;
- executable/delivery `65881c5628a737eaeaf4742ab7b11a63b9d3cbc2`;
- candidate-to-executable diff under source/tests/environment/dependency inputs:
  zero paths;
- launcher blob `42bb7560d6a04995edb7ae7976906f23e3b9d4f5`, SHA-256
  `4b09b6c6ef0f682bdb5326ca23851b45705d636e00fddd0809de74abbc37577e`;
- complete 90-file source-list SHA-256
  `c9e0a4175725e59d1e4e3e3efbe3421c0d9b8480fd5161cf5147ae9184eb511f`
  and ordered source-state SHA-256
  `d64aa9c1baa28541dffc96bdfbed4bed18d85d3ae2e6e687c53e465dd67a797d`;
- environment bootstrap SHA-256
  `f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf`.

RUN_REQUEST Section I freezes the complete source list, nine exact nodes/order,
locked dependencies, numeric-job/12-hex/random mode-0700 short temp contract
with maximum 48 bytes, 90-second per-node supervisor, 30-second faulthandler,
exact group/identity cleanup, literal mini root, cleared full-data overrides,
fresh paths keyed `65881c5628a7`, one node/task/GH200, eight CPUs, 64 GiB,
20 minutes, no requeue/retry/follow-on, and strict 9/9 PASS acceptance.

Preparation checks were pure Git/hash/text/shell-preflight only: exact clean
HEAD/parent/branch, R15 direct read, candidate/executable object and runtime-diff
checks, launcher/bootstrap/source/node enumeration and hashes, path collision
checks, mini-root presence, launcher `bash -n`, and `git diff --check`. No
project/package import, pytest, multiprocessing/fork/spawn runtime, Torch/CUDA,
data read, model work, `sbatch`, `srun`, or Slurm compute occurred.

Canonical O-084 at exact Orchestra commit
`0b4e8bf089e03c093dc368d363402dd6b875bbfb` approves exactly one submission
of the unchanged Section I tuple after the clean detached execution worktree is
provisioned and the complete final preflight passes. Approval is bound to the
exact candidate/executable, launcher blob/SHA, 90-file list/state, locked
dependencies, nine nodes/order, temp/supervisor/cleanup contract, literal mini
scope, fresh paths, resources, command, strict acceptance, and stop rules.
Any drift voids approval before submission. The single attempt consumes O-084
when submitted regardless of scheduler, harness, or node outcome; no retry,
requeue, replacement, alternate invocation, automatic resubmission, or
follow-on is authorized.

Canonical O-085 at exact Orchestra commit
`6e2461dc3f20a082f6bafa0227456bf129fa342a` records that the unchanged exact
Section I command was submitted once as Slurm Job `352354`, started at
`2026-07-12T17:23:29+02:00`, and is active on node `n559`. O-084 was
consumed immediately at submission regardless of eventual scheduler, harness,
or node outcome. This execution-state record does not query or interpret
results and changes no tuple, acceptance, or stop rule.

**Compute status:
`EXECUTED_ONCE_JOB_352354_ACTIVE_APPROVAL_CONSUMED_NO_RETRY`.** No retry,
requeue, replacement, alternate invocation, automatic resubmission, or
follow-on is authorized. O-084/O-085 grant no full suite/data/cache,
100/1000-step, metrics/profile/DDP/matrix, merge, push, upload, or scientific
interpretation.

### O-086 Job 352354 terminal evidence and strict-readiness remediation

O-086 code is durable at exact
`7a3a15a13d19be87c5269966afc5fd6b1054d660`, whose sole parent is exact clean
implementation HEAD `b36bfa93da5e1b0691ad94d6ab5840a2fbd0f723`. It changes
exactly three test/launcher paths and no production source or lifecycle file:

| Path | Git blob | SHA-256 |
|---|---|---|
| `tests/test_model_task.py` | `9e8b19d6d3ff1edc02c4efe1dac8c007bdb50097` | `36abc1d447d3b1ff1feb249c2740a351b0371abe6c01da71b6665b5c0a7143c7` |
| `scripts/run_s07_b_multiworker_diagnostic.sh` | `90e477e68091276fd1c92bf914e0f0dd1fd0c1b4` | `8cb97121ada2041517f56b2c9291dea1c49771d247cd2d81a83c89d73450f5ed` |
| `scripts/run_s07_b_static_checks.sh` | `f8dda510b1e2b2f6dac2ce7924fca3ef34a8b1a8` | `f62c57252cce720d5f425142afc849766d1f14da49aa8c6c9dba5fe44b1f84f3` |

The O-086 audit independently read the raw Job `352354` artifacts rather than
relying on the scheduler status or prior summary. The job is terminal
`COMPLETED/0:0` on `n559`, ran from
`2026-07-12T17:23:29+02:00` to `17:27:45+02:00` for `00:04:16`, had zero
restarts, batch MaxRSS `1008M`, and TotalCPU `02:28.908`.

The frozen artifacts retain their formal result exactly:
`diagnostic_complete=true`, `artifact_complete=true`, `suite_pass=true`, nine
observed of nine expected nodes, aggregate JUnit `9 tests / 0 failures / 0
errors / 0 skips`, and
`all_process_groups_and_identities_cleaned=true`. Every node returned zero,
had no timeout or supervisor cleanup intervention, reported `cleanup_ok=true`,
and produced JUnit `1/0/0/0`. The 51-record root checksum manifest passes
independent `sha256sum -c` verification. Principal immutable hashes are:

| Artifact | SHA-256 |
|---|---|
| `diagnostic_summary.json` | `b8fd26b34d607510c9a3a3e90251709dce43f792b8956728845448e6837478e9` |
| `execution_identity.json` | `f97eaa87ed0b4c74706f412a186dff226b07d30430c4f846f6ff695a5f8522be` |
| `diagnostic_run_config.json` | `6a74231ec26177ff44fadde407541739fd794e6f6423efb849241a1d06595ec6` |
| source list / state | `c9e0a4175725e59d1e4e3e3efbe3421c0d9b8480fd5161cf5147ae9184eb511f` / `d64aa9c1baa28541dffc96bdfbed4bed18d85d3ae2e6e687c53e465dd67a797d` |
| `selected_nodes.tsv` | `b216a6512b5d54d58c1e9acf632ae34b4fac1b930df0cafe83c4ca7b86e6eeca` |
| `sha256sums.txt` (51 records) | `67d723b37ca3a9d36af8bde75eab13765ca05bef1bd1fc6e2f08bbf87d3527ac` |
| scheduler stdout / stderr | `0fbd5327140be93c30b69d579b28d8be23c9229d1a217bf70009d3845df398a6` / `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

That formal 9/9 is not strict readiness PASS. Node 8
`detection_loader_determinism_num_workers` emitted
`PytestUnraisableExceptionWarning` from
`_MultiProcessingDataLoaderIter.__del__`, whose traceback terminates in
`RuntimeError: DataLoader worker (pid 4081264) is killed by signal: Aborted`.
Its pytest log SHA-256 is
`fb50d32d85c1f0cc24c27727d784c0ee7ceb045caf166294fd3869fd3bb62dbb`,
JUnit SHA-256 is
`be7b4856d7c9dd2552376329fea2546dbf17f293905332e393d2f36f67ce70d3`,
and supervisor-result SHA-256 is
`55bff2a7f9907c63be5447eb50308c3ca07362e30b8aa55c41a53323441cd323`.
Because the executed pytest command did not make this warning fatal, Job
352354 strict multiworker readiness is **FAIL**, and the formal summary's
`suite_pass=true` must be cited only together with this warning-contamination
limit.

The scoped source remediation explicitly owns each multiprocessing iterator in
the two affected model-task tests and calls its private `_shutdown_workers()`
inside `finally`, without catching or suppressing shutdown errors. The CUDA
test still obtains the first batch of two successive epochs by calling
`iter(loader)` twice, asserts the persistent loader returns the same cached
iterator, and then shuts it down. The nine-node diagnostic now adds exact
`-W error::pytest.PytestUnraisableExceptionWarning` to every subprocess and
records the warning-fatal policy in both config and summary. Static checks lock
the flag and AST-verify one explicit iterator/finally-shutdown structure in
each target test.

Only shell/source/heredoc/static/diff checks are permitted for this candidate;
no pytest, project import, multiprocessing, CUDA/data/model runtime, or Slurm
has been authorized. No retry or follow-on compute is authorized. A new exact
runtime attempt would require durable SHA, independent review, a new immutable
request, and separate owner approval. Current status is
`JOB_352354_TERMINAL_FORMAL_9_OF_9_STRICT_READINESS_FAIL_NO_COMPUTE_APPROVED`.

Stable exact-code-tree checks completed within that restriction: both changed
shell files passed `bash -n`; the launcher-contract checker printed
`short TMPDIR contract: 5 launchers OK`; `test_model_task.py` passed stdlib
source `compile()`; all four embedded Python heredocs in the changed diagnostic
launcher passed stdlib `compile()`; `git diff --check` was empty; and the raw
Job 352354 root manifest independently verified all 51 records. Exact code
commit `7a3a15a13d19be87c5269966afc5fd6b1054d660` is the only O-086 code
identity; these three lifecycle documents record its separate delivery state.

### O-088 final warning-fatal runtime request

Independent R16 `d621d696d5a188189041fa73e54495eb56e8db49`
accepted code `7a3a15a13d19be87c5269966afc5fd6b1054d660` and delivery
`764aab2390940746f4409ee52a3437b5cf1d341f` at static-authored scope with no
P0-P3. The owner approved the minimal completion plan: S00 alone prepares,
submits, monitors, and audits one fresh warning-fatal nine-node job with no new
subagent and no retry. RUN_REQUEST Section K freezes the exact tuple: launcher
SHA-256 `8cb97121...`, 90-file list/state `c9e0a417...`/`ffd96718...`, node
manifest `b216a651...`, one GH200/20 minutes, and fresh paths keyed
`764aab239094`. Besides formal 9/9 and complete process/checksum evidence, raw
logs must contain no unraisable-warning, worker-abort, or temp-cleanup breach
token. No full-data, cache, model-step, metric, profile, DDP, matrix,
scientific, merge, or push scope is authorized.
