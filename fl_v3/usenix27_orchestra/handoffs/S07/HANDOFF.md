# S07-A HANDOFF — reviewed S01 data-foundation integration

## Session identity and self-assessment

- Session/phase: `S07-A`, reviewed S01 data-foundation integration.
- Worker self-assessment: **S07-A-R P1 REMEDIATIONS IMPLEMENTED; NEW FOCUSED
  EXECUTION REQUEST PENDING OWNER APPROVAL**. This is not self-approval of the
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
- **Current remediation implementation / proposed executable `INT-A_SHA`
  (`NEW_IMPL_SHA`):**
  `c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe` (adds zero-failure/error/skip
  JUnit acceptance to the focused launcher on top of `0a89ea1`).
- Current full-cache runtime source-state SHA-256 (23 tracked files, recomputed
  from immutable `NEW_IMPL_SHA`):
  `6a4ad312b41ff161aa07f7628176ab74f550768f8b15c335314c5d262cbec1c2`.
- Current focused provenance-test source-state SHA-256 (25 tracked files):
  `357da48780436aaba3cbc6735e350d446763acc9f6cb8a0bf424728e55a32d0e`.
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

## S07-A-R P1 remediation — implementation complete, execution pending

The owner authorized correction of both durable-review findings, scoped commits,
and preparation of one bounded validation request. No `sbatch`/`srun` has been
submitted in this continuation.

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
   `NEW_IMPL_SHA=c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe`.

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
- S07-A-R P1 remediation ending at `c8dd920cf3f8...`: `python3 -m py_compile`
  passed for `build_gt_database.py` and its test; `bash -n` passed for the full
  cache and new focused provenance launchers; `git diff --check` passed.
- Login-node `python3 -m pytest -q fl_v3/tests/test_build_gt_database.py` could
  not start because `/usr/bin/python3` has no `pytest`; this is an environment
  limitation, not a PASS or code failure. No login-node result is substituted for
  the pending GH200 focused evidence.
- The full-cache and focused source hashes were each recomputed both from the clean
  worktree and from immutable Git blobs at `NEW_IMPL_SHA`; they matched exactly at
  `6a4ad312...` (23 files) and `357da487...` (25 files), respectively.
- Fresh proposed roots
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_cache_t1v2_c8dd920cf3f8`
  and
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07a_provenance_tests_c8dd920cf3f8`
  were confirmed absent and were not created.

O-009 Job `333477`:

- exact implementation `c1f4fbe`, source hash
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`;
- `n430`, one GH200/eight CPUs, `COMPLETED 0:0`, `00:01:23`, about 0.0231
  GPU-hours, no retry;
- `62 passed in 12.83s`, zero failures/errors/skips;
- `sha256sum -c` passed for execution identity, source list, pytest log, and JUnit;
- complete hashes/resources/limits are in `RESULTS.md`.

No full cache job, model job, 100/1000-step job, full-data profile, metric,
scientific run, or upload was submitted. The remediation itself submitted no new
job of any kind and preserves Job 333477 as the latest execution evidence.

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
`detached@c8dd920cf3f8007c3b2ec03f48bcc3f83144ebbe` worktree. If separately
approved and passed, it will fill the currently unknown train/val canonical cache
hashes and physical pickle/sidecar hashes in `cache_identity.json`. Until then,
S06 must not substitute the historical `t1.v1` files or claim full-data readiness.

## Gate checklist

| S07-A acceptance item | Self-assessment | Evidence / limit |
|---|---|---|
| exact worker history reachable | PASS | merge second parent is exact `abe5c58` |
| REVIEW artifact present, not implementation | PASS | cherry-only `a4ca386`; review parent remains old `ce2e772` |
| no production GT `t1.v1` bypass | PASS | explicit IC API plus hostile test |
| depth/format/sidecar/cache/manifest fail closed | PASS for pre-review tree | job 333477, 62/62 at `c1f4fbe`; P1 changes need focused execution |
| physical pickle/sidecar GT binding | PASS (implementation), EXECUTION PENDING | pre/post hash checks + hostile tests at `c8dd920`; exact focused request prepared |
| directory mode supported | PASS | implementation plus directory/ZIP tests |
| focused launcher attests fixtures/config/deps | PASS (implementation) | immutable 25-file set + source hash; execution pending |
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
- The new P1-A hostile regression has not executed on GH200. The exact one-job
  focused request remains pending owner approval, with no retry/resubmit/follow-on.
- The remediation launcher/count/runtime/checksum changes have only local/static
  validation. No remediation job or test rerun was authorized or performed; the
  request remains pending independent review and exact owner approval.
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
and the two S07-A-R P1 findings have scoped implementation/static evidence plus an
exact pending focused validation request.

Forbidden interpretation: independent remediation PASS before re-review, executed
P1 hostile-test evidence, permission to submit either pending request, full-data/model training readiness, architecture or
metric acceptance, mAP/NDS/model quality, FL, attack/defense, generalization,
scientific, or publication claims.
