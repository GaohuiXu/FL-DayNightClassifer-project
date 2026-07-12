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
- Code/config integration candidate before this handoff-only commit:
  `2944386de19ab7d25b3ec09c77b6951dd34cea8d`.
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
  and ZIP cases at sweep depths 1/10 with read-counter and metadata assertions.

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

## Candidate config files and explicit blocker

The five candidate files name the requested architecture/precision/head/sampling
choices, but are deliberately **NON-RUNNABLE TEMPLATES**, not resolved configs.
Each contains an unknown top-level `template_only` marker, unresolved/null camera
initialization where applicable, unmaterialized full `t1.v2` identities, and
unapproved placeholder budget/seed values. Strict loading rejects each before data
or model construction. File-byte SHA-256 values are:

| Candidate | File SHA-256 |
|---|---|
| C-STR8 `s07_b_c_str8.json` | `9952a15cf5afeeb1aa058f1b48a59487cadad32451de7546d5d6bc84996a09ea` |
| L-P020 `s07_b_l_p020.json` | `c737ae2cbd26757603753c872a217d5ab1ddc99f757404ab792e60e1fdea35ad` |
| L-S075 `s07_b_l_s075.json` | `dfd7fd027b86144bd011c31c75ef3154b1aecd5af1526a52225088ba03e7f260` |
| F-U `s07_b_f_u.json` | `bad2f7ae7134b1fc2e8182bb773404822658754ab61e03b6aaa0c59b5823ccb7` |
| F-CBGS `s07_b_f_cbgs.json` | `ed3c479da0fb6ae3afe69772a97c3ec34e5b40bac5d710e6725ba25e5ac36358` |

The explicit blocker to a real resolved candidate is unchanged: full trainval
`t1.v2` train/val cache artifacts and their logical/pickle/sidecar identities are
absent; exact module dataroot/manifest identities, budget, seed, camera
initialization and CBGS-adjusted schedule are not owner-frozen. Synthetic all-`a`
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
9. `git diff --check` after every integration stage;
10. candidate file hashes reproduced by the committed static script.

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
integrated; one code-level C/L/F stack candidate exists at `2944386...`; static
syntax/schema/template-fail-closed checks pass; actual runtime gates are explicitly
assembled but unexecuted.

Forbidden interpretation: S07-B/full-stack/production/checkpoint/throughput/memory/
model-quality PASS; a real resolved candidate config; permission to materialize
full cache or run any model/data/profile/metric job; mAP/NDS/fusion gain; FL,
attack/defense, generalization, scientific or publication evidence.

Next action is S00 completeness audit followed by an independent S07-B-R from the
exact returned worker SHA. The reviewer must inspect actual diffs/topology and may
not substitute this worker self-assessment for a verdict.
