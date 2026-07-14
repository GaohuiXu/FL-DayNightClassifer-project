# S08 independent review — immutable implementation/evidence baseline

## Verdict

```text
VERDICT: REMEDIATE
REVIEWED_SHA: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
BASE_SHA: 733c84f8e3019fe4d683663821bd86918d3875a7
BRANCH_AT_REVIEW: codex/s08-s09-cl-readiness
FOCUSED_SMOKE_VERDICT: PASS remains valid for its declared scope
Q1_READINESS: BLOCKED pending the P1 remediation and re-review
S09_READINESS: BLOCKED
```

No P0 finding was identified. One P1 and two P2 findings require linear
remediation before an exact Q1 request can be approved. They do not invalidate
Job `428112` as a focused implementation smoke and do not turn Jobs
`426619`/`427800` into model-numerical evidence.

This review did not edit implementation source, tests, configs, existing handoff
documents, external dependency checkouts, or runtime artifacts. It did not run or
submit Slurm/GPU work.

## Findings

### P1 — the Q1 fixture is recorded dynamically, not fail-closed against a predeclared complete identity

**Evidence.** The runner hard-binds the sample, scene, key-LiDAR payload and
4096-point prefix in
`tests/test_s08_precision_qualification.py:36-42,259-292`. That is useful but
not the complete model input. The full batch tensor manifest, camera/calibration/
GT content, augmentation parameters, and derived fixture identity are calculated
from the live mini tree at runtime in
`tests/test_s08_precision_qualification.py:294-335`. The derived value is then
used directly to create all resolved configs in
`tests/test_s08_precision_qualification.py:386-395`. There is no hard-coded or
launcher-supplied expected full batch/fixture/augmentation digest and no equality
check before model construction.

Consequently, a changed camera payload, calibration tensor, GT tensor, decoding
result, or augmentation implementation can produce a new internally consistent
`fixture_sha256`, run all eight cells, and be written into the evidence as though
it were the approved fixture. Post-run recording is not equivalent to binding the
input before material compute. This conflicts with the active requirement that a
material job be approved against exact immutable data/fixture identities and that
any identity change invalidate approval.

**Required remediation.** Before model or optimizer construction, compare the
complete replay fixture against predeclared exact identities, at minimum:

- the canonical full `batch_tensor_manifest_sha256` covering camera, calibration,
  GT, LiDAR prefix and all other tensor inputs;
- `augmentation_params_sha256` (and the stable field order);
- the canonical fixture-manifest SHA-256 derived from the declared fields.

The identities may be immutable constants in the reviewed runner or exact
launcher inputs, but they must be schema-validated, bound in the future Q1
snapshot/request, and rejected before any cell executes. The future request must
also bind the exact source/snapshot, command, output root, dependency state and
these same fixture values. Add a negative test showing that one changed camera or
calibration/GT tensor fails before model construction, not merely that a new hash
is recorded.

### P2 — the per-cell qualification predicate does not gate scheduler transitions that it records

**Evidence.** Window records capture `scheduler_last_epoch_before/after` in
`src/fl_v3/training/precision_diagnostics.py:360-392,541-546`, but
`counter_deltas_consistent` checks only optimizer/success/attempt/invalid fields in
`src/fl_v3/training/precision_diagnostics.py:547-553`. Q1's `_cell_pass()` relies
on that flag plus final optimizer/success/exposure state in
`tests/test_s08_precision_qualification.py:339-366`; it does not require the
scheduler to remain unchanged on a rejected window, advance exactly once on an
accepted window, or finish at `last_epoch == 3`. The summary records the final
scheduler value at line 510 but does not bind it into `qualification_pass`.

The production loop currently steps the scheduler only after an accepted update,
and the focused toy regression checks one overflow-then-accept case. Static review
therefore found no present scheduler-step defect. The issue is that the Q1 gate is
not fail-closed over one of the counters its approved envelope explicitly asks to
record.

**Required remediation.** Make the cell predicate verify the per-record scheduler
delta (`+1` only for accepted, `0` otherwise), continuity between records, and
final `scheduler.last_epoch == optimizer_step == 3`. Because Q1 explicitly uses
no EMA, also require the recorded EMA-disabled state rather than leaving it only
descriptive. Add focused positive and hostile-negative predicate tests.

The current design deliberately completes all eight cells even when a numerical
cell fails; that is appropriate and should be retained. A future Q1 request must
state explicitly that pytest/Slurm completion means **runner completeness**, not
that every precision regime qualified. Numerical acceptance is read per cell from
`qualification_pass`; `all_primary_cells_pass` is not expected to be true merely
for the job to be scientifically useful.

### P2 — active canonical documents contain stale precision and milestone state

**Evidence.** The implementation and tests make sparse precision explicit, yet
`usenix27_orchestra/ORCHESTRA.md:133-137` still says the current resolver
automatically enables sparse-conv FP16 whenever global precision is FP16. The
“Still unresolved” table at `ORCHESTRA.md:304-310` still places the hybrid-model
and Q1 diagnostic choices before implementation even though O-097/O-098 and the
sealed implementation resolved those v1 choices; only normalization and final
precision policy remain open. `usenix27_orchestra/SESSIONS.md:117-126` still calls
this a pre-implementation gate and mentions only a terminal negative result,
despite Job `428112`. Within the historical SMOKE-1 section,
`handoffs/S08/RESULTS.md:297-299` uses present-tense wording that SMOKE-2 is frozen
and unapproved even though later sections correctly record SMOKE-2 as consumed
and SMOKE-3 as PASS. Finally, the review verdict vocabulary in
`usenix27_orchestra/KICKOFFS.md:221` differs from the current S08 review contract.

These are active-authority documents, not harmless legacy notes. They can cause a
later session to infer the wrong production precision behavior or milestone gate.

**Required remediation.** Update only the active canonical/status wording:

- label automatic sparse FP16 as the pre-S08 historical behavior and describe the
  current explicit partition;
- move the already-decided v1 architecture/fixture/diagnostic choices out of the
  unresolved table while preserving sparse-normalization and final-policy
  uncertainty;
- record all three terminal smoke outcomes and current review/remediation state;
- make historical SMOKE-1/2 wording past tense; and
- harmonize the independent-review verdict vocabulary.

Do not rewrite negative evidence or imply Q1, precision-policy, gradient-root-
cause, convergence, performance, or capability acceptance.

## Evidence and adversarial checks

### Git and immutable source

- `HEAD` and the reviewed branch resolved exactly to
  `791aba97f7bbe92e7708b63f94f2e7d8599f91be`; the review began from a clean
  worktree.
- The exact comparison base was
  `733c84f8e3019fe4d683663821bd86918d3875a7`.
- `git diff --check` passed for the complete base-to-review diff.
- The S08-SMOKE-3 snapshot was independently rehashed using the documented
  path/executable-bit/size/content record format: 583 files, 4,444,941 bytes,
  zero writable entries, tree SHA-256
  `3014cab90ed88b5705367fc1dd1a21740593acc3a186c72f9073bffe15247a43`.
- Current production/config/test/runner bytes match the consumed snapshot. The
  only expected differences are the separately bound outer submit script and
  post-run canonical/handoff documents.

### Smoke artifacts

The preserved Job `428112` artifacts under
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_smoke3_3014cab90ed8`
were inspected directly. Their sizes and SHA-256 values match `RESULTS.md`:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `environment.json` | 1,139 | `2e7884a35a43fa6dc5f422602e9e75166cbc67174d48836e2ebf9bd4d88fde8d` |
| `smoke.log` | 1,577 | `5ef5db037debb49bea335d9bd9f2daea0b2f1d725aced0af5b791b66a9a36796` |
| `smoke.junit.xml` | 15,780 | `3b5dcdc2d7559b1af80446e946858d0133de1ad531802c105b43b6d128f76171` |
| `smoke.exit` | 2 | `9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa` |
| `artifact_sha256s.txt` | 318 | `a6aea314859224f2a0c238fae693a0ae5d3eabe417d11ca5131b9835311ed7b7` |
| `slurm-428112.out` | 1,602 | `d1881872129b5301cc34e5f5df4cb887b6913528452cb8c64eac41c3df171b9f` |
| `slurm-428112.err` | 123 | `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57` |

JUnit contains 106 tests, zero failures/errors/skips, and the stdout contains the
terminal `S08_PRECISION_SMOKE_PASS` marker. Slurm accounting reports
`COMPLETED 0:0`, one GH200, zero restarts, and elapsed `00:03:28`.

The negative artifacts were also checked rather than inferred from summaries:

- Job `426619`: `FAILED 1:0`, pre-pytest source-state rejection, empty redirected
  environment/stdout, and exact traceback artifact preserved. It says nothing
  about model numerics.
- Job `427800`: `FAILED 1:0`, JUnit 106 with 103 pass/3 fail, exact two
  disabled-GradScaler accessor failures plus one test-regex mismatch, and both
  tiny sparse tests passing. It remains negative overall evidence.

### Implementation semantics confirmed

- `s08.v1` requires an explicit global/sparse precision partition; the partition
  is hash-bound, bridged into production construction, and compared on checkpoint
  resume. Non-SECOND requires `not_applicable`; SECOND FP32 requires sparse FP32;
  SECOND FP16 permits sparse FP16 or the FP32 island. Sparse BF16 remains rejected.
- The FP32 island explicitly disables inherited autocast for voxelization/VFE,
  SECOND/spconv, dense collapse and `to_bev`. Eligible downstream camera/fusion/
  head work remains inside global FP16 autocast. Full sparse FP16 behavior is
  retained.
- The spconv 2.3.8 no-grad evaluation workaround remains narrowly version-gated,
  changes only sparse-convolution leaf dispatch flags, and restores every flag in
  `finally`.
- Diagnostics are opt-in, hook-free and bounded. They retain only the declared
  boundaries, summarize parameter gradients after GradScaler unscale, explicitly
  unscale retained activation gradients on an FP64 copy, perform tensor/JSON work
  before optimizer mutation, and clean references/gradients on a hostile pre-step
  failure. The default path's update, metrics and RNG equality are covered by the
  focused test.
- Q1 declares the correct eight ordered C/L/F regimes, clones an exact per-mode
  initial state, restores the same forward RNG for every attempt, uses one
  persistent scaler per FP16 cell from 512, permits bounded backoff below one, and
  preserves prior cell records when a numerical cell does not qualify.
- The observer records the large-but-finite LiDAR gradients as evidence. Neither
  source nor documents claim that their magnitude proves a head/loss defect or
  that S08 has already found the exact overflow operation.

### Coverage boundaries

- S08-SMOKE-3 contains tiny sparse and toy-loop execution, not a current full
  six-task optimizer window. That boundary is accurately stated in the terminal
  result.
- L-P020/F-CBGS have schema/template coverage in the exact smoke and inherited
  constructor/sampling regression tests in the repository, but those constructor/
  sampling cases were not among the exact Job `428112` selectors. Treat that part
  as static/inherited compatibility evidence, not newly executed GH200 evidence.
- No Q1 cell, full-data route, metric, profile, 100/1000-step run, DDP, seed
  expansion, Protocol A/B, attack or defense was executed or established.

## Re-review gate

Re-review should pin a new immutable SHA and inspect only the linear remediation:

1. pre-execution full-fixture identity enforcement plus a drift-negative test;
2. scheduler/EMA-disabled checks in the per-cell qualification predicate plus
   hostile predicate tests;
3. canonical/status wording corrections without changing prior evidence; and
4. the exact future Q1 request/launcher only after the remediated code is sealed.

No Q1 submission is authorized by this review. A new exact owner approval remains
required after remediation and re-review.

## Remediation re-review / R2

### Findings

#### P0

None.

#### P1

None. The original fixture-prebinding finding is closed.

#### P2

None. The original scheduler/EMA qualification finding is closed, and no
model/recipe/precision semantic regression was identified in the linear
remediation.

#### P3 — a few duplicated status summaries retain pre-Smoke-5 wording

The authoritative top-of-file status blocks and detailed execution sections are
current, but three duplicated summaries lag the terminal state:

- `SESSIONS.md:67` still stops at Smoke-3 and says immutable seal/review evidence
  is pending;
- `SESSIONS.md:142` says there are three consumed smoke outcomes although the
  package now preserves five; and
- `RUN_REQUEST.md:254-256` says Smoke-5 must still be approved and pass before Q1
  can be requested, while the same file's current-request block correctly records
  Job `429080` as consumed terminal PASS.

This drift is conservative: it cannot authorize compute, weaken a gate, erase a
negative result, or change a model/data/precision interpretation. It should be
cleaned up in S00's next authorized canonical/status edit, but it does not require
another implementation change, smoke job, or Q1 rerun.

### Verdict

```text
VERDICT: PASS_WITH_RESIDUAL_RISK
REVIEWED_SHA: 103c7389a47938b1f9dd0cba60251df6dce9e5bb
REMEDIATION_BASE: 791aba97f7bbe92e7708b63f94f2e7d8599f91be
BRANCH_AT_REVIEW: codex/s08-s09-cl-readiness
ORIGINAL_REVIEW_BODY_SHA256_BEFORE_R2: 4385f1696d984d50cbdc5037b0384f70453237d78597d24374c4fa6ad4e32569
SMOKE_5_VERDICT: PASS for its declared focused remediation/fixture-attestation scope
Q1_READINESS: READY_FOR_EXACT_REQUEST_PREPARATION; NOT APPROVED FOR EXECUTION
S09_READINESS: BLOCKED pending Q1 evidence, independent interpretation, and owner precision-policy acceptance
```

The P1/P2 remediation is technically sufficient to prepare a new exact Q1
request. This verdict does not approve Q1, choose a precision policy, establish
stable optimizer windows, explain the LiDAR gradient magnitude, or unlock S09.

### Closure of the original findings

#### Complete fixture identity now fails closed before model/optimizer construction

- `tests/fixtures/s08_q1_raw_input_manifest.json` predeclares 13 metadata files,
  six camera payloads, the key LiDAR payload, and nine prior sweeps: 29 unique
  regular files and 41,085,435 bytes. The runner binds both its file SHA-256 and
  canonical logical SHA-256.
- `test_s08_precision_qualification.py:122-218` schema-validates the manifest,
  normalized paths, roles/order, sizes, regular-file type, and content hashes.
  The re-review independently rehashed all 29 live mini files with zero mismatch.
- The derived identity has an exact schema and five required values: raw-input
  manifest, complete batch-tensor manifest, augmentation field order,
  augmentation values, and canonical fixture manifest. Missing, partial,
  uppercase, malformed, or unequal launcher values fail closed.
- `_prepare_fixture()` computes the candidate and calls `_require_fixture_identity()`
  before the Q1 body resolves cells, obtains the task, or builds any model,
  criterion, optimizer, scheduler, or scaler. Static call-order inspection confirms
  that no model/optimizer path precedes this equality gate.
- The hostile cases now mutate `images`, `cam_intrinsics`, and `gt_boxes` with
  in-place `add_(1.0)`, verify that both batch and fixture digests changed, and
  require the pre-model gate to raise. Job `429080` executed all three cases.
  Job `428889` remains preserved as the negative predecessor where the original
  calibration assignment was a no-op.
- The separate attestation test explicitly refuses Q1 expected-identity inputs,
  constructs neither model nor optimizer, and writes `candidate_only=true`,
  `model_constructed=false`, `optimizer_constructed=false`, and
  `q1_executed=false`. It therefore cannot silently become a Q1 cell.

#### Scheduler transitions and EMA-disabled state are genuine cell gates

- `precision_diagnostics.py:386-394,543-577` records scheduler before/after and a
  pure-Python accepted-window delta check, plus explicit EMA presence and expected
  update-state consistency.
- `_cell_pass()` now requires every scheduler-delta flag, continuity from each
  prior `after` to the next `before`, terminal
  `scheduler.last_epoch == optimizer_step == 3`, and EMA disabled with both
  expected-update values `None` and `ema_state_consistent=true`.
- The positive overflow-then-three-accept timeline passes. Hostile delta,
  continuity, terminal-epoch, and EMA-enabled records each fail. The lower-level
  loop regression also executed an overflow followed by acceptance and observed
  scheduler `(0,0)` then `(0,1)` with disabled EMA consistency.

#### No model, recipe, or precision semantics changed

The exact remediation range changes no task construction, training loop, runtime
state, checkpoint, resolved config, detector, camera/LSS, sparse encoder/backbone,
fusion, head, loss/target, optimizer recipe, metric, decode, or NMS file. The only
production-code edit is two output-only consistency fields in the already opt-in
`precision_diagnostics.py` record lifecycle. It does not alter forward values,
losses, backward, scaler decisions, optimizer/scheduler/EMA calls, or accepted
updates. The remaining changes are tests, a frozen input manifest, smoke wrappers,
authority/status documentation, and this review artifact.

### Raw evidence and adversarial checks

- Git began clean at exact branch/HEAD
  `codex/s08-s09-cl-readiness@103c7389a47938b1f9dd0cba60251df6dce9e5bb`.
  `git diff --check` passes for `791aba9..103c738`.
- The Smoke-5 snapshot was independently rehashed using the documented
  length-prefixed path/executable/size/content records: 585 files, 4,515,200
  bytes, zero writable entries, tree SHA-256
  `51daec3e860e6d412ad57d807efd78a08b03630afb37798880999fa039900a25`.
  Current remediation source/tests/manifest/runner match the snapshot byte for
  byte. The outer submit script and here-document body independently match their
  recorded hashes.
- Slurm accounting independently reports Job `429080` as `COMPLETED 0:0`, zero
  restarts, one GH200, eight CPUs, 96 GiB, and elapsed `00:03:36`.
- Phase 1 JUnit independently parses as 116 tests with zero failure/error/skip;
  Phase 2 parses as one test with zero failure/error/skip. Both exit files contain
  `0`, and stdout terminates with `S08_PRECISION_SMOKE_PASS`.
- Every listed Smoke-5 artifact size and SHA-256 matches `RESULTS.md`; the
  job-produced checksum manifest matches all ten runtime/test/fixture artifacts.
- The five fixture identities were recomputed independently from canonical JSON
  and the declared little-endian `torch.float64` augmentation bytes:

  | Identity | Recomputed SHA-256 |
  |---|---|
  | raw-input manifest | `f95c0cd141c891f69f44a0ecc792e4878946a3cdc4a1a2ce7911df074b848316` |
  | complete batch tensor manifest | `de8b8f06c8c5b14871262fe56167ac52095f8e7cac42387de157b8e247a4e9da` |
  | augmentation field order | `0495e2db0984cf3063ef5d0d84a2fd83b99b1b0cf3383f7a78534bbce8bb5de7` |
  | augmentation values | `57728184c564966e83d19214e192e8fc79fd84a2701b46b8299c237eb61dd9ea` |
  | canonical fixture manifest | `f46a79c1cefa52a65d9e402b791cfce73fa194f20e6aa7cbfb3096957b6b9c89` |

- The Smoke-4 snapshot also independently rehashes to its recorded 585-file tree
  identity. Its raw artifacts and JUnit preserve exactly 115 pass/1 fail, with the
  sole failure showing that assignment of `1.0` to the existing calibration
  diagonal did not change its digest. Smoke-4 and Smoke-5 source/test/runner bytes
  differ only by the declared `= 1.0` to `.add_(1.0)` correction.
- Shell syntax and Python AST/JSON parsing pass for the changed runner, submit
  wrapper, diagnostics, qualification tests, and raw-input manifest.

### O-107 guardrail review

The prospective mechanical-remediation rule is consistent across `AGENTS.md`,
`CLAUDE.md`, `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`: it requires explicit
opt-in on an initial exact O-009 smoke, caps the loop at three total submissions
and two GPU-hours, retains the same selectors/data/command family/resource
ceiling, and requires every replacement tuple to be frozen and recorded first.
It stops on uncertainty, a repeated blocker, or any possible model output/loss/
gradient/update, data, precision, optimizer/scheduler/EMA, metric/science, scope,
seed, or resource change. It is prospective, does not reinterpret Jobs
`426619`-`429080`, and explicitly excludes Q1 and material/scientific jobs. It
therefore does not weaken the S08 precision gate.

### Residual risk and next gate

- Q1 has not run. Smoke-5 constructed no model or optimizer and contains no
  precision-window evidence. A future exact request must bind reviewed SHA
  `103c7389...` (or an independently attested immutable snapshot), all five fixture
  identities above, exact eight-cell order/config hashes, dependency state,
  command/launcher hashes, one-GH200 resources, fresh output, bounds, and stop/
  interpretation conditions, then receive explicit owner approval.
- The replay fixture is one bounded mini sample. Even a successful Q1 can qualify
  only bounded accepted windows; it cannot establish convergence, capability,
  performance, mAP/NDS, production-data readiness, or a scientific precision
  policy by itself.
- Large finite FP32 LiDAR gradients and direct FP16 SECOND failures remain evidence,
  not a proven head/loss defect or an identified exact faulty sparse operation.
  Sparse normalization and the training recipe remain unchanged and owner-gated.
- L-P020/F-CBGS compatibility, final precision-policy acceptance, and S09 remain
  downstream gates. No full-data, profile, 100/1000-step, DDP, seed expansion,
  Protocol A/B, attack, or defense claim is established.
- The P3 duplicated-status wording above should be corrected in the next
  authorized documentation update; it is not an execution blocker.
