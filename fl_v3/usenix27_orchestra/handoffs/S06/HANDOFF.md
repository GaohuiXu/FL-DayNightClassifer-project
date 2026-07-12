# S06 HANDOFF — production modes, resolved config, runtime and evaluation

## Session identity and self-assessment

- Session: `S06`.
- Base: `968d81583c87ba76b7dbbb722760f8eb8eb6cd39`.
- Source branch named by kickoff: `codex/s00-orchestra-ledger`.
- Startup state: clean detached HEAD at the exact base; branch name empty.
- Owner-authorized delivery branch: `codex/s06-production-runtime`.
- Original implementation commit: `a2c7095adfb0715decb6b21bb08977c9b39eb9a1`.
- S00 rejected/voided predecessor executables:
  `a95816b607d1ced5f07bd1136b23f36f58357a14` and
  `7d733e9b08454b059822015fcaf3eea53e8c2e56`; both are
  `REJECTED_BY_S00_NEVER_EXECUTE`.
- Remediation implementation/test commit:
  `4fe67093e0aa3b60f7e1c3804b2c8b06c3f6eeab`.
- Remediation-1 executable `6696984a6ebd4ec398d9fbfa172fb118e84e7af8`
  was executed once as failed job `341997` and is permanently retired.
- Negative-results commit: `675f152`.
- Remediation-2 implementation/launcher executable:
  `c330c72f4060348768c63fb1b7855ca56baffb95`.
- Final documentation delivery SHA is returned to S00 after committing this
  package; a commit cannot embed its own SHA without changing itself.
- Worker self-assessment: **JOB 341997 REMAINS FAILED; REMEDIATION-2 STATIC PASS,
  DEPENDENCY-BACKED TESTS NOT RUN; NEW REQUEST PENDING S00 AUDIT.** This is not an independent,
  integration, model-readiness, full-data, or scientific PASS.

Other than the one approved failed synthetic job `341997`, no Slurm job/srun,
mini/trainval traversal, cache materialization, model campaign, metric, profile,
merge, push, PR, upload, or reviewer launch occurred.
During approval-record auditing, shell interpolation accidentally invoked bare
`sbatch`; Slurm rejected the empty script before job creation with
`Batch script is empty!`. This negative control-plane event has no Job ID,
allocation or root side effect and was not the approved exact command.

## S00 remediation disposition

S00 rejected the prior request hash
`d2e302aba6cb0ed0561677f15c04601c373ebe10e9471787168bba05dcc65ef2`.
It and both predecessor executables above must never be run. Remediation stayed
inside original S06 ownership and addressed only the two blocking findings plus
S00's read-only GPU-memory follow-up:

1. fixed accumulation-window boundaries, complete attempted/invalid/discarded
   sample/window reconciliation, fail-closed partial/short batches, and
   update-boundary-only limits;
2. mutation-free checkpoint preflight followed by transactional load/rollback of
   every runtime component and global RNG;
3. rollback tensors cloned directly to detached CPU storage and production
   checkpoint load mapped to CPU, avoiding an additional live-GPU rollback copy.

The real unused-modality raw-decode proof remains an explicit S07-B seam. The
full replacement `centralized_train.py` compatibility surface and DDP refusal are
unchanged limitations, not newly claimed coverage.

After S00's terminal audit of job `341997`, remediation-2 addressed the four
traceback families without weakening their gates: frozen config values are thawed
to canonical plain containers; the runtime-tail fixture is genuinely length
opaque while known-length preflight remains; checkpoint save uses a same-directory
non-hidden `.pt` temporary path with target-preserving cleanup fixtures; and eval
retains an explicit six-camera calibration requirement. The launcher now disables
the read-only cache provider and always writes/verifies its final artifact manifest
before returning the original pytest status. No remediation-2 compute has run.

## Scope and files

Modified within S06 ownership:

- `fl_v3/src/fl_v3/models/fusion/detector.py`;
- `fl_v3/src/fl_v3/training/{tasks.py,loop.py}`;
- `fl_v3/scripts/centralized_train.py`;
- `fl_v3/src/fl_v3/eval/{detection_eval.py,provenance.py}`;
- `fl_v3/src/fl_v3/utils/runtime.py`;
- `fl_v3/tests/{test_model_task.py,test_eval_detection_eval.py,test_eval_provenance.py}`.

Added:

- `fl_v3/src/fl_v3/config/{__init__.py,resolved.py}`;
- `fl_v3/src/fl_v3/training/{checkpoint.py,runtime_state.py}`;
- `fl_v3/configs/s06_synthetic_camera.json`;
- `fl_v3/scripts/run_s06_runtime_tests.sh`;
- five `fl_v3/tests/test_s06_*.py` files;
- `fl_v3/usenix27_orchestra/handoffs/S06/{RUN_REQUEST,RESULTS,HANDOFF}.md`.

No S01-S05 source, data/cache/ZIP implementation, S05 box conversion/decode/NMS,
legacy config, canonical Orchestra file, `fl_v3/collab/`, or `fl_v2/` file was
modified. S02-S05 implementations were not copied, cherry-picked, reimplemented,
or finally wired.

## Resolved configuration contract

`s06.v1` is a strict nested schema resolved before data/model/optimizer creation.
Unknown/missing keys, unknown/legacy mode aliases, inconsistent architecture enums,
unsupported precision/optimizer, invalid batch arithmetic, noncanonical hashes,
`t1.v1`, missing train/val depth identities, and environment-only scientific
defaults fail.

The canonical UTF-8 JSON uses sorted keys, fixed separators, `allow_nan=False`,
and SHA-256. It binds:

- exact `camera_only | lidar_only | fusion` mode;
- frozen camera/LiDAR/fusion/head architecture enums;
- `fp32 | fp16`, Adam-family optimizer and successfully executed update budget;
- microbatch/world/accumulation and recomputed effective global batch;
- seed, max epochs, worker count and EMA policy;
- data root/version/train/val splits and explicit `n_sweeps`;
- separate train and val `t1.v2` logical, pickle and sidecar identities/paths;
- ZIP manifest logical/file identities/path;
- Torch version plus exact spconv/cumm versions and source SHAs for sparse modes;
- eval timing policy.

The committed synthetic config hash is
`1f06f07fc16d64e10624e98e0cad120cff63131c838244177f2e0688517ac813`.
Its paths and hashes are intentionally fake test fixtures and are not production
data identities.

At production cache load, both physical files are SHA-256 checked, the requested
canonical `info_cache.cache_paths()` path must equal the resolved path,
`load_cache(..., n_sweeps=..., expected_cache_hash=...)` rejects depth/logical
drift, and the SQLite manifest's recorded logical hash plus file SHA are checked.
The resolved hash enters memoization so a new identity cannot reuse an old cached
object.

## Model-mode construction, transfer and forward

- `DetectorConfig.model_mode` passes through `normalize_model_mode`; only the
  exact three names are accepted.
- Camera-only constructs camera/preprocess/view-transform plus a camera adapter,
  common BEV neck and head. It does not construct LiDAR or fuser modules.
- LiDAR-only constructs LiDAR plus a LiDAR adapter, common neck and head. It does
  not construct preprocess/camera/view-transform/fuser modules.
- Fusion constructs both native branches and the fuser.
- Strict checkpoint identity checks occur before `load_state_dict(strict=True)`,
  so weights from a disabled/different topology cannot be silently loaded.
- `project_batch_for_mode()` removes disabled modality tensors before host-to-
  device transfer. Detector forward requires only the active branch's keys.
- For a real production dataset, `NuScenesDetectionTask._make_loader()` currently
  fails closed because accepted S01 dataset code decodes both payload modalities.
  It never decodes both and drops one. S07-B must add the reviewed mode-aware raw
  I/O API as the sole cross-session integrator.

The hostile mode tests instantiate all three topologies with instrumented module
doubles, assert construction absence, key projection and forward absence, reject
legacy aliases, and exercise same-instance concurrent calls.

## Sparse runtime serialization and dependency boundary

Lidar/fusion resolved configs require `spconv==2.3.8`, `cumm==0.7.13` and the
accepted source SHAs. Before strict sparse construction, installed spconv must be
exactly `2.3.8`. Each detector instance owns a runtime `RLock`; individual
forward/mode calls and the `serialized_mode()` traversal context share it. Official
eval holds the context over the complete one-pass traversal; centralized training
holds it over each complete epoch call. Locks are runtime-only state and are
recreated independently during EMA/deepcopy.

This satisfies S04-R's serialization requirement at the detector entry point. It
does not claim safety for bypassing the detector, multi-process shared instances,
or an unreviewed server path. S07-B must preserve the context when it wires S04's
option-A encoder and retain `torch.no_grad()` around fp16 eval.

## Executed-update, nonfinite and overflow accounting

`TrainingState` now records cumulative attempted microbatches/samples/windows,
loss-evaluated samples, successful windows/exposure, invalid windows/samples and
their nonfinite/overflow causes, plus fail-closed discarded windows/samples. Its
validator requires the exact reconciliations:

```text
attempted_samples = exposure_samples + invalid_samples + discarded_samples
attempted_windows = successful_windows + invalid_windows + discarded_windows
optimizer_step = successful_windows
invalid_windows = nonfinite_windows + overflow_windows
```

Accumulation windows are fixed by original microbatch position. A nonfinite loss
clears gradients immediately but the remaining microbatches are still consumed
and loss-evaluated through that same boundary; the next window never shifts.
Nonfinite/overflow windows advance none of optimizer-step, successful exposure,
scheduler or EMA. A successful window must contain the same declared global
microbatch size at every position, so its exposure is the resolved effective
batch.

When loader length is known, a non-divisible epoch/`max_steps` plan is rejected
before `model.train()` or data iteration. Limits are checked before fetching the
next batch and may stop only at a window boundary. An unknown-length partial tail
or runtime short batch is cleared, fully recorded as discarded, raises, and makes
the state terminal for further production training; the caller cannot increment
the epoch or report success. No pending gradients are serialized.

Hostile fixtures place nonfinite loss at positions 1/2/3 of a three-microbatch
window, inject overflow, exercise known and unknown remainders, reject a
non-boundary `max_steps`, prove no extra fetch after update budget, and reject a
short final microbatch. They are authored but dependency-backed execution remains
pending exact S00 approval.

## Checkpoint and resume boundary

Checkpoint schema `s06.checkpoint.v1` contains exactly:

- strict model, optimizer, scheduler, GradScaler and EMA state;
- complete `TrainingState`, including epoch/update/exposure;
- Python, NumPy, Torch and all CUDA RNG states;
- full resolved config and hash, actual mode and precision;
- both train/val cache identities and ZIP-manifest identities;
- checkpoint identity field.

Save is atomic through a same-directory temporary file. Save and load reject any
pending accumulation phase/samples. Before mutating a caller object or global RNG,
load validates exact schema/config/data/component presence, TrainingState
reconciliation, complete Python/NumPy/Torch/CUDA RNG states, model keys/order and
tensor shape/dtype/layout, optimizer type/config identity, Adam/AdamW parameter
groups/state topology, and scheduler/GradScaler/EMA state structure. Legacy,
partial, mismatched and `strict=False` migration paths are absent.

Only after preflight, every caller component is snapshotted and real strict loads
begin. Any late model/optimizer/scheduler/scaler/EMA/RNG exception rolls every
component and RNG back before rethrowing. Snapshots recursively clone tensors
directly to detached CPU storage while preserving nested non-tensor containers;
production resume also loads the checkpoint payload with `map_location="cpu"`.
Thus rollback does not allocate a second full detector/optimizer/EMA copy on the
GPU. The cost is deliberately transferred to host memory: peak host storage is
proportional to the full checkpoint payload plus rollback copies of live
model/optimizer/scheduler/scaler/EMA state. No production-shape host-memory bound
or throughput claim is made; S07-B must resource that later gate.

Hostile fixtures cover corrupt model shape, optimizer topology, TrainingState,
partial/bad RNG, scheduler/scaler/EMA structural errors and real-load-only late
failures at each component position, proving caller state and all RNG streams are
unchanged. A no-alias fixture checks every snapshot tensor is detached and on CPU;
the requested GH200 inventory also exercises exact rollback of live CUDA
model/optimizer/GradScaler/EMA from CPU snapshots. The positive continuous versus
interrupted/resumed AdamW fixture guards against over-strict rejection of a legal
resume state. These PyTorch fixtures are authored, not locally run.

## Persistent loader/sampler contract

`PersistentEpochIterator` stores one loader identity and refuses any training
sampler without deterministic `set_epoch`. `EpochPermutationSampler` derives one
complete permutation solely from `(seed, epoch)` and therefore resumes at an epoch
boundary without replaying earlier generator advances. Tests assert same-loader
reuse, exact epoch calls, resume permutation equality and no duplicate/omitted
indices.

The production nuScenes loader/sampler wiring is intentionally not completed in
S06 because the current dataset lacks mode-aware I/O and S07-B owns integration.
Real ZIP-handle continuity across this new integrated sampler remains an S07-B
mini/GH200 gate; S01's prior lifecycle evidence is not relabeled as S06 evidence.

## Evaluation and provenance

- `decode_eval_set()` requires explicit resolved mode and precision, projects the
  batch before transfer, enters configured autocast, forces fp16 head outputs back
  to FP32 before decode, and traverses the loader once.
- Detector eval holds `serialized_mode(False)` and the function is
  `torch.no_grad()`, satisfying the S04 option-A entry boundary.
- Optional timing synchronizes/observes only; it writes to a separate sink and
  never changes decode/submission values. The authored fixture compares timed and
  untimed records.
- Submission `meta.use_camera/use_lidar` reflects the actual mode.
- When production identities are present, result JSON adds complete mode,
  resolved-config, checkpoint, train/val cache and ZIP-manifest provenance; a
  partial set is rejected.
- `build_s06_provenance`/`verify_s06_provenance` bind the same identities plus
  source Git SHA and reject partial/drifted records.

S05's accepted forced-FP32 multi-task decode and total content order are not in
this branch. S07-B must merge its implementation with these autocast/provenance/
single-pass changes and rerun output-neutrality plus official devkit round-trip.

## Verification actually performed

Passed locally on x86 login Python:

```text
git diff --check
python3 -m py_compile <all changed/added Python and focused test files>
bash -n fl_v3/scripts/run_s06_runtime_tests.sh
stdlib load/canonicalize/hash of configs/s06_synthetic_camera.json
stdlib hostile rejection of legacy camera/FUSION aliases
AST parse of new runtime/config/test sources
```

After S00 remediation, the exact executable additionally passed:

```text
git diff --check
python3 -m py_compile training/{runtime_state,loop,checkpoint}.py \
  scripts/centralized_train.py tests/{test_s06_training_runtime,test_s06_checkpoint_resume}.py
bash -n fl_v3/scripts/run_s06_runtime_tests.sh
```

After job `341997`, remediation-2 executable `c330c72...` passed the same static
checks for every newly changed Python/test file and launcher. A stdlib execution
of `ResolvedConfig.to_run_config()` additionally proved nested cache identities
are plain JSON-serializable dictionaries, preserve the resolved hash, and keep
the committed synthetic canonical hash
`1f06f07fc16d64e10624e98e0cad120cff63131c838244177f2e0688517ac813`.
Torch/pytest remain unavailable locally, so no remediation-2 runtime result is
claimed.

Final local dependency probe:

```text
/usr/bin/python3: pytest unavailable
/usr/bin/python3: torch unavailable
/usr/bin/python3: numpy unavailable
```

Therefore no PyTorch/pytest result is claimed. The attempted local command was:

```bash
python3 -m pytest -q fl_v3/tests/test_s06_resolved_config.py \
  fl_v3/tests/test_s06_model_modes.py \
  fl_v3/tests/test_s06_training_runtime.py \
  fl_v3/tests/test_s06_checkpoint_resume.py \
  fl_v3/tests/test_s06_loader_eval.py
```

It stopped immediately with `No module named pytest`; no test body executed. This
is preserved as an environment limitation, not a test failure or PASS.

## Failed execution and new pending validation

S00 approved exactly one remediation-1 submission. Job `341997` ended
`FAILED 1:0` after `00:01:47`: `45 passed, 17 failed, 0 errors, 0 skipped`.
`RESULTS.md` records the four root-cause families, exact environment/allocation,
raw artifact checksums, bare-sbatch no-op and missing final in-job manifest. That
tuple is permanently `FAILED_NEVER_RETRY_NEVER_EXECUTE`.

The new remediation-2 `RUN_REQUEST.md` binds executable
`c330c72f4060348768c63fb1b7855ca56baffb95`, tree
`7ce589685d15fb42c057154c3329679ada934f4b`, 25-file source aggregate
`bc19c139f773592dc085b47b3b83b1721f3c5ca0abeeeb1c6485e9e2d8f533dc`,
launcher `146f55797ec8191083f8347bcecae858785e3c64c08fc798079fa1ac53edde2d`
and request SHA-256
`9479538201ec398b1617847c5265d0dbeae8ec0db084fc6b867a435ffb5020a9`.
Its `s06_runtime_remediation2_c330c72f4060` roots are confirmed absent. Status is
`APPROVED_BY_S00_ONE_SUBMISSION_PENDING`; no remediation-2 job exists yet. S00
approved exactly one submission bound to request delivery `cae0ff59...` and the
full tuple above. Retry/requeue/resubmit, a second job, reviewer and tuple mutation
remain forbidden.

## Gate status

| Gate | Worker evidence/status |
|---|---|
| exact mode enum / alias rejection | STATIC PASS; runtime tests AUTHORED NOT RUN |
| mode-specific construction/transfer/forward | IMPLEMENTED; hostile runtime tests AUTHORED NOT RUN |
| disabled raw payload decode | FAIL-CLOSED SEAM; S07-B must add dataset API |
| canonical config/hash and fail-closed identities | 341997 NEGATIVE mappingproxy; remediation-2 STDLIB PASS, gate NOT RUN |
| separate train/val t1.v2 and manifest identities | IMPLEMENTED/STDLIB PASS; real artifacts absent |
| spconv 2.3.8 and same-instance serialization | IMPLEMENTED; Arrhenius dependency/concurrency tests NOT RUN |
| fixed-window accounting/effective batch | 341997 fixture invalid; true opaque remediation fixture AUTHORED NOT RUN |
| fail-atomic strict checkpoint/resume | 341997 BLOCKED before assertions; legal temp/atomic fixtures AUTHORED NOT RUN |
| persistent loader/set_epoch/no duplicate/omission | IMPLEMENTED; runtime test AUTHORED NOT RUN |
| real ZIP handle lifecycle across resume | NOT RUN; S07-B integrated mini gate required |
| eval autocast/single-pass/timing neutrality/metadata | 341997 fixture failed; strict six-camera fixtures AUTHORED NOT RUN |
| 100/1000-step/full profile/metrics | FORBIDDEN / NOT RUN |
| independent S06-R | NOT STARTED; worker must not launch reviewer |

## Negative results and limitations

1. Job `341997` is dependency-backed negative evidence. Remediation-2 tests are
   not run because the login runtime lacks torch/numpy/pytest and no new compute
   is approved.
2. Full trainval `t1.v2` does not exist; committed hashes are synthetic sentinels.
3. The production CLI deliberately fails before disabled-modality data decode
   until S07-B integrates a mode-aware S01 dataset API and epoch sampler.
4. DDP/world-size>1 is explicitly refused until S07-B wires the distributed
   sampler and wrapper; no concurrency or exposure claim is made for DDP.
5. S02 targets/loss, S03 camera modules, S04 SECOND and S05 head/decode are not in
   this branch. The architecture enums are provenance contracts, not final wiring.
6. Legacy task evaluation still exists outside official `decode_eval_set`; S07-B
   must route production official evaluation exclusively through the resolved path.
7. CPU rollback avoids a second live-GPU state copy but retains a full host-memory
   snapshot; production detector resume memory and rollback are not yet measured.
8. `centralized_train.py` is a full replacement entry point. Compatibility with
   historical callers is not claimed, and world-size/ DDP remains fail-closed.
9. No real checkpoint, throughput, memory, convergence, metric or scientific
   evidence was produced.
10. Job `341997` is a failed engineering gate, not a partial PASS. Remediation-2
    fixes for its four failure families are static/authored only and require the
    new exact request plus S00 approval before they can change gate status.

## Explicit S07-B integration seams

1. Add mode-aware S01 dataset/blob-store construction that does not open/read the
   disabled modality while retaining calibration/GT metadata and ZIP lifecycle.
2. Map resolved architecture enums to reviewed S02-S05 interfaces: S02 targets and
   old-target invalidation; S03 augmentation/stride8/0.5m/output dtype; S04
   `[B,256,180,180]`, 0.6m/origin, caps and option-A runtime; S05 task-list head,
   forced-FP32 decode and deterministic NMS.
3. Pass `EpochPermutationSampler` (or reviewed equivalent) into one persistent
   loader; for DDP, prove global no-duplicate/no-omission semantics and multiply
   exposure exactly once by world size.
4. Reconcile the S05 `detection_eval.py` worker diff with S06 autocast, actual-mode
   metadata, identity bundle and timing-neutral single traversal; neither branch
   may overwrite the other's correctness gates.
5. Preserve detector traversal locks around every sparse train/eval entry and
   test mode-transition contention, exception restoration and resume-to-eval.
6. Materialize/freeze real train and val `t1.v2` hashes only under separately
   approved S07-A request; replace no synthetic identity post hoc.
7. Never retry remediation-1/job `341997`. Audit the fresh remediation-2 request;
   only a later explicit approval may authorize its one exact bounded job. S06
   must not launch a reviewer; later production-shape/100/1000/full-data gates
   require separate scope.

## Interpretation boundary

Allowed:

- S06 provides an independently reviewable fail-closed runtime/config/checkpoint/
  eval implementation and an exact pending synthetic validation request;
- the listed stdlib/static checks passed on exact committed source;
- the CLI refuses unintegrated raw-modality I/O and DDP rather than silently
  violating the contract.

Forbidden:

- S06 runtime PASS before dependency-backed execution and S06-R;
- S07-B/full-stack/model/full-data/training readiness;
- real cache, ZIP-resume, sparse fp16, performance, 100/1000-step, mAP/NDS,
  fusion gain, FL, attack/defense, generalization, scientific, or publication
  claims;
- treating synthetic sentinel hashes as production identities;
- merging, pushing, uploading, or launching a reviewer from this worker handoff.
