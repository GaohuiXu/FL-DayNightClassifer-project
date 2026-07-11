# S07-A HANDOFF — reviewed S01 data-foundation integration

## Session identity and self-assessment

- Session/phase: `S07-A`, reviewed S01 data-foundation integration.
- Worker self-assessment: **PASS FOR INDEPENDENT S07-A-R REVIEW**. This is not
  Orchestra integration/scientific acceptance.
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
- Final implementation commit:
  `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4`.
- Final implementation plus immutable run-request/results evidence commit:
  `d26ba78a4766552c7c486206556183cc04bb9dae`.
- **Proposed `INT-A_SHA`:**
  `d26ba78a4766552c7c486206556183cc04bb9dae`. The subsequent delivery commit adds
  only this `HANDOFF.md`; S00 should use the actual worker delivery HEAD for
  S07-A-R and may designate `d26ba78` as the executable/evidence integration
  candidate after review.

No merge into `v3-ad-perception`, push, PR, upload, branch deletion, or worktree
operation occurred.

## Exact approved Git topology

1. Created `codex/s07-a-data-foundation` from detached
   `953bfb57941b5a3660ed650c1a80267cd82245d4`.
2. Created non-fast-forward merge `60f603a0837a55b8bc5d56eedcbba065fcc10673`
   with parents, in order:
   `953bfb57941b5a3660ed650c1a80267cd82245d4` and
   `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`.
3. Cherry-picked review-only commit `7cf7fcc...` as
   `a4ca386db59a9250d3fce95209e38ac617b4ff77`.
4. Added scoped implementation/test/docs commit `c1f4fbe...` and evidence commit
   `d26ba78...`.

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
  - requires explicit `n_sweeps` and frozen expected canonical cache SHA;
  - loads only through `info_cache.load_cache(..., n_sweeps=...,
    expected_cache_hash=...)`, inheriting format, sidecar, every-record depth, and
    canonical content validation;
  - ZIP mode additionally requires and validates the logical manifest hash and
    SQLite file SHA-256, and exact trainval01..10 archive names;
  - uses one explicit `NuScenesBlobStore` for both single/multi-sweep reads and
    records cache/backend/manifest provenance in GT-database `meta.json`;
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
  - does not rebuild the manifest, profile, train, evaluate, or auto-retry.
- `fl_v3/tests/test_build_gt_database.py` and
  `fl_v3/tests/test_nuscenes_info_cache.py`
  - cover explicit depth/expected hash, historical format/path rejection,
    missing/mutated sidecars, content hash drift, logical/file manifest mismatch,
    GT caller behavior, and directory/ZIP preservation.
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

O-009 Job `333477`:

- exact implementation `c1f4fbe`, source hash
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`;
- `n430`, one GH200/eight CPUs, `COMPLETED 0:0`, `00:01:23`, about 0.0231
  GPU-hours, no retry;
- `62 passed in 12.83s`, zero failures/errors/skips;
- `sha256sum -c` passed for execution identity, source list, pytest log, and JUnit;
- complete hashes/resources/limits are in `RESULTS.md`.

No full cache job, model job, 100/1000-step job, full-data profile, metric,
scientific run, or upload was submitted.

## Exact cache/manifest/provenance contract for S06

For each production/scientific train or val consumer, the resolved config and run
provenance must freeze and agree on:

1. official `version` and `split`;
2. integer `n_sweeps` (primary request is 10 total including keyframe);
3. cache format exactly `t1.v2` and depth-specific filename;
4. canonical cache hash from sidecar/pickle metadata, passed as
   `expected_cache_hash` and recomputed from every record on load;
5. cache pickle file SHA-256 and sidecar file SHA-256;
6. backend mode: `directory` or `zip`;
7. for ZIP mode, exact accepted manifest path, format, logical manifest hash,
   SQLite file SHA-256, and archive-name set;
8. resolved config hash/checkpoint provenance that includes these fields.

Loading must fail before data/model execution on missing/ambiguous depth, `t1.v1`
or other format, missing/mutated sidecar, record-depth drift, canonical hash drift,
expected cache hash mismatch, manifest logical/file/archive mismatch, or backend
relabeling. Directory mode carries null ZIP-manifest fields and must not be treated
as legacy/dead.

The full cache request in `RUN_REQUEST.md` is exact and **PENDING**. If separately
approved and passed, it will fill the currently unknown train/val canonical cache
hashes and physical pickle/sidecar hashes in `cache_identity.json`. Until then,
S06 must not substitute the historical `t1.v1` files or claim full-data readiness.

## Gate checklist

| S07-A acceptance item | Self-assessment | Evidence / limit |
|---|---|---|
| exact worker history reachable | PASS | merge second parent is exact `abe5c58` |
| REVIEW artifact present, not implementation | PASS | cherry-only `a4ca386`; review parent remains old `ce2e772` |
| no production GT `t1.v1` bypass | PASS | explicit IC API plus hostile test |
| depth/format/sidecar/cache/manifest fail closed | PASS | job 333477, 62/62 |
| directory mode supported | PASS | implementation plus directory/ZIP tests |
| focused launcher attests fixtures/config/deps | PASS | source list and execution identity |
| shell/Python/diff checks | PASS | commands above |
| canonical Orchestra docs unchanged | PASS | empty diff for three canonical paths |
| no S02-S06 integration | PASS | topology/name-scope audit |
| full `t1.v2` cache | PENDING | exact request prepared; not submitted |
| model/scientific work | NOT RUN / FORBIDDEN | phase boundary preserved |

## Negative results, residual risks, and interpretation limits

- Full trainval `t1.v2` train/val cache artifacts do not yet exist; their cache and
  file hashes cannot be frozen until owner-approved execution.
- Historical job 332651 proves referenced-member coverage and loader-only timing,
  but its `t1.v1` caches are forbidden and its job lacks retroactive in-job source
  attestation.
- Job 333477 is real-mini/synthetic engineering evidence, not trainval-scale
  directory/ZIP decoded parity or model-step evidence.
- Full epoch/model data wait, concurrent shared-filesystem contention, all-payload
  CRC coverage, and production model I/O remain untested.
- S06/S07-B must migrate the deferred scientific entry points and bind the exact
  post-materialization hashes; S07-A intentionally did not cross ownership.

Allowed interpretation: reviewed S01 is integrated with preserved history/review,
the declared data-foundation/GT provenance paths fail closed under the focused
GH200 suite, and the exact pending full-cache request is ready for owner review.

Forbidden interpretation: full-data/model training readiness, architecture or
metric acceptance, mAP/NDS/model quality, FL, attack/defense, generalization,
scientific, or publication claims.
