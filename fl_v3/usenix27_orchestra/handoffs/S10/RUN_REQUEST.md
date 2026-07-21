# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S10 Phase I-P throughput preflight
ACTIVE_DECISION: owner activates the exact Section-9.2 B16 extension resource envelope
REQUEST_STATE: B16 EXTENSION ACTIVE / SERIAL CELL A NEXT / ENVELOPE B FROZEN
EXECUTION_AUTHORITY: Section 9.2 only; base 1.20 + bug reserve 0.50 = hard 1.70 GH200-hours
ACTIVE_PHASE: Phase I-P engineering throughput preflight before C/L qualification
PLAN: HANDOFF.md Section 1 / IP-G0 closed
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at the same SHA
ENVELOPE_B: Section 7 preserved / NOT EXECUTABLE / disposition deferred to IP-G2
```

IP-G0 authorized scoped Phase I-P source/docs/tests, local validation and linear
commits. The owner subsequently activated Section 8 at implementation commit
`85c6719e4b880b198d850e16b1418c230fa5c656`, including continuous IP-WP1 ->
IP-WP2. This does not authorize any evaluation role, the Section-7 Envelope B,
merge, push or publication.

O-146 records the owner's exact activation of Envelope A at request commit
`e321aed749fd859c809199d52c30b2771dbef8b3`. S00 may execute WP0 through WP4
continuously within Section 6, including the bounded checkpoint acquisition,
data materialization, material commits and at most three serial engineering
submissions / one aggregate GH200-hour. Envelope B still requires the later
measured `P1-G1` approval.

O-147 records the owner's 2026-07-20 amendment at implementation/evidence commit
`c45e020ed16496e2acaa5f8d34b135da21fb1230`: the total submission cap is five
and the aggregate ceiling is 1.10 charged GH200-hours. It authorizes exactly one
fresh-output Camera replacement followed, only after Camera PASS, by the original
Job B. Both retain the original data, seed, configs, tolerances, performance gates,
per-job resources and prohibitions. No further derived submission is permitted;
either failure stops immediately. The fifth numerical slot is not standalone
execution authority.

O-147 execution stopped on its first allowed job, Camera Job D `521959`, which
returned `FAILED 1:0` after four seconds before creating the runner control/output
directory. Both Slurm streams are empty. Therefore the exact failing fail-closed
predicate cannot be localized retrospectively; the evidence bounds it to the
runner's pre-control checks, silent module/environment bootstrap, or Slurm-resource
assertions, before pytest, checkpoint acceptance, D_fit access, model construction,
CUDA build or calibration. Per the owner's any-failure stop rule, original Job B
was not submitted and no remaining numerical slot may be used.

O-148 superseded O-147's submission-count and per-engineering-failure stop mechanics
for the remaining WP4 smoke/qualification work. It retained concurrency one and the
unchanged `1.10` charged-GH200-hour aggregate ceiling. That authority is consumed:
12 submissions used `0.516389` GH200-hours. Camera reached an honest negative at the
unchanged pooling-promotion gate; LiDAR reached PASS. The unused budget is not
continuing authority.

O-149 consolidates the resulting collaboration contract for future explicitly
approved engineering-validation envelopes: aggregate GPU-hours and concurrency are
the default resource controls, with no numeric submission cap unless explicitly set;
unambiguous frozen-semantics defects are diagnosed, recorded, repaired and rerun
serially. It grants no compute by itself and preserves every material scientific/
resource owner gate. The canonical form is in `AGENTS.md` and the Orchestra docs.

O-145 added the independent optimized CUDA BEV-pooling/equivalent-kernel requirement
to WP2 and its forward/backward, FP16/FP32-policy, operator-timing, and end-to-end
qualification to WP4. It also authorizes drafting this exact Envelope-A request and
the O-145 documentation commit. O-146 supplies the previously missing activation;
it does not authorize capability training/evaluation, Envelope B, or any action
outside Section 6.

The full pre-O-143 per-job request ledger is preserved at Git object
`26b38612c6dd7e37bd97f8eb9e443735d36154c6:fl_v3/usenix27_orchestra/handoffs/S10/RUN_REQUEST.md`.
Accepted results, raw artifact paths and checksums remain in `RESULTS.md`.
Prior authorities are consumed and cannot be revived from the historical object.

## 2. Historical execution index

| Decision / job | Terminal state | Scientific disposition |
|---|---|---|
| O-125 / `467862` | timeout in STOP-A optimizer | no split result |
| O-126 / `468295` | site-transformed request; protection-cancelled before work | no evidence |
| O-127 / `468404` | completed | STOP-A split/evaluator accepted |
| O-128 / `477892` | parity-gate failure | no localization |
| O-129 / `478250` | baseline-instability gate | bounded negative |
| O-130 / `479667` | completed | STOP-B `INCONCLUSIVE` |
| O-131 / `492525` | failed/incomplete runner gate after two full cells | retained bounded negative/incomplete evidence |
| O-132 / `496312` | completed | C0-v2 bounded execution PASS |
| O-134 / `502456` | pre-candidate assertion failure | no gradient verdict |
| O-136 / `502572` | completed | C1-A `LOCALIZED_NORM` |
| O-137 / `502958` | pre-model test-fixture failure | no experiment |
| O-138 / `503075` | pre-model test-fixture failure | no experiment |
| O-139 / `504508` | completed | C1-B0 bounded observation PASS |
| O-140 / `504921` | one BN1d overflow broke matched exposure | C1-B1 `FAIL/INCOMPLETE` |
| O-141 / `505266` | pre-model schema-access failure | no experiment |
| O-142 / `505316` | training/eval complete; post-processing timeout | BN1d-B8 body complete, tail gate incomplete |

This index is for navigation, not reinterpretation. Exact allocations, commands,
hashes, raw paths and limitations are in `RESULTS.md` and the historical Git
object above.

## 3. Phase approval record

A phase becomes executable only after the owner approves every field:

```text
PHASE:
REQUEST_STATE: DRAFT / APPROVED / CONSUMED
OBJECTIVE_AND_EXIT_GATE:
CANDIDATES_AND_MAX_COUNT:
DATA_SPLITS_AND_EVALUATOR:
SEED_POLICY:
TRAINING_EXPOSURE_AND_CHECKPOINT_SELECTION:
AGGREGATE_GPU_HOURS:
SUBMISSION_POLICY:
MAX_CONCURRENCY:
OUTPUT_ROOT:
ENGINEERING_REMEDIATION_ALLOWED:
OWNER_ESCALATION_CONDITIONS:
ALLOWED_INTERPRETATION:
FORBIDDEN_INTERPRETATION:
OWNER_APPROVAL:
```

Owner decision O-150 (2026-07-20) explicitly amends only the future Phase-I Camera
backend and capability prerequisite: the numerically qualified PyTorch sorted
`segment_reduce` fallback is production; the CUDA kernel remains available but
unpromoted; the historical `1.25x` target no longer blocks capability. O-150 also
instructs S00 to start Envelope-B preparation. It does not itself state an aggregate
GPU-hour ceiling, so the first scientific submission still requires that exact field
to be bound.

Within an approved engineering-validation phase, S00 may derive commands/resolved
configs and repair unambiguous frozen-semantics test/fixture, config/schema,
dtype/API, runner, checkpoint-I/O, artifact/provenance or logging failures. Under
O-149 the loop is bounded by the approved aggregate GPU-hours and concurrency; a
numeric submission cap applies only when explicitly recorded. Scientific and
resource boundaries do not change.

## 4. Per-job ledger schema

Append one concise row per submitted job:

| Field | Required value |
|---|---|
| Job | Slurm ID and terminal state |
| Source | Git SHA |
| Config | resolved-config SHA-256 |
| Data | split identity |
| Seed | exact seed |
| Command | literal command or stable invocation |
| Resources | GPU/CPU/RAM/wall time and charged GPU-hours |
| Output | raw output root |
| Checkpoint | SHA-256 or `none` |
| Metrics | artifact path and SHA-256 or `missing` |
| Classification | scientific result or engineering incident |
| Follow-up | continue within phase / owner escalation |

Raw outputs are immutable. Detached snapshots, recursive artifact manifests,
command-file hashes and stdout hashes are optional rather than default.

## 5. Phase I O-144/O-145 plan record

```text
PLAN_STATE: OWNER_FROZEN / O-144 plus O-145 and O-150 amendments
PLAN_PATH: handoffs/S10/PHASE_I_PLAN.md
P1_G0: CLOSED
CANDIDATES_AND_MAX_COUNT: exact Camera ImageNet primary + exact LiDAR scratch primary; max 2
DATA: D_fit train; D_select terminal development assessment; D_audit owner-sealed
SEED_POLICY: seed 0
EXPOSURE: 20 exact-CBGS epochs; physical B4; accumulation 8; effective B32
CHECKPOINT_SELECTION: epoch-20 terminal only
WORKFLOW: 5 WPs + 3 owner gates + 2 approval envelopes
CAMERA_POOLING: O-150 PyTorch sorted segment-reduce production backend; CUDA option
                retained unpromoted; WP2/WP4 parity/policy evidence retained
EXECUTION_AUTHORITY: Envelope A consumed under O-146/O-147/O-148;
                     Phase I-P IP-E1 active under Section 8; the old 49.0-hour
                     Envelope-B request remains NOT EXECUTABLE
```

The complete graph, optimizer/scheduler, augmentation, role-bound GT-paste,
evaluation, remediation and amendment rules are normative in
`PHASE_I_PLAN.md`. This record does not duplicate them.

## 6. Envelope A — consumed engineering completion under O-148

This section preserves the complete consumed approval object. The owner's approval
bound the Git commit containing this section; deterministic `<REQUEST_SHA12>` and
`<IMPLEMENTATION_SHA12>` path substitutions did not require another approval. O-148
later amended only the remediation/submission mechanics recorded above.

```text
PHASE: S10 Phase I / Envelope A implementation and engineering calibration
REQUEST_STATE: CONSUMED / CLOSED under O-148
PLAN_SHA: 260750a76548208f62c384b0e0547744b619244c
REQUEST_COMMIT: e321aed749fd859c809199d52c30b2771dbef8b3
BRANCH: codex/s10-phase1-branch-qualification
OBJECTIVE: implement and review WP0-WP4; acquire the exact Camera initializer;
           materialize exact CBGS/GTDB identities; qualify optimized BEV pooling;
           calibrate C/L; produce the measured Envelope-B request
ONE_APPROVAL_EFFECT: continuous WP0->WP1->WP2->WP3->WP4 implementation, focused tests,
                     material linear commits, bounded acquisition/materialization,
                     and the submissions below; no per-WP approval
IMPLEMENTATION_CANDIDATES_AND_MAX_COUNT: exact frozen Camera + LiDAR graphs; max 2;
                                         no capability candidate is trained/evaluated
SCIENTIFIC_CANDIDATES_EXECUTED: none
SEED: 0 for initialization/order/calibration; no seed comparison
CALIBRATION_EXPOSURE: only the fixed warm-up/timed microbatches in Sections 6.4-6.5
CAPABILITY_METRICS_AND_SELECTION: forbidden
D_SELECT_D_AUDIT_OFFICIAL_VAL: forbidden
SCIENTIFIC_CHECKPOINT: none; all Envelope-A checkpoints are non-selectable engineering artifacts
PARITY_TOLERANCES: fixed in Section 6.4; no post-failure relaxation
PERFORMANCE_GATE: >=1.25x pooling median speedup; <=2% end-to-end time regression;
                  <=5% peak-allocation regression
ALLOWED_INTERPRETATION: implementation conformance, numerical parity, resource estimate
FORBIDDEN_INTERPRETATION: branch capability, convergence, mAP/NDS, candidate selection
OWNER_APPROVAL: 2026-07-20 — "批准激活 commit e321aed749fd859c809199d52c30b2771dbef8b3
                中的 S10 Phase I Envelope A，并按其中边界连续执行 WP0-WP4。"
OWNER_AMENDMENT: 2026-07-20 — "批准以 commit
                 c45e020ed16496e2acaa5f8d34b135da21fb1230 修订 S10 Phase I
                 Envelope A：总 submission cap 从 3 增至 5、总 GH200-hour ceiling
                 从 1.0 增至 1.10；仅允许串行执行一个 fresh-output Camera replacement
                 与原 Job B，保持原数据、seed、config、容差、性能门、单作业资源和
                 禁止项不变，不再允许任何派生重提，任一失败即停。"
OWNER_AMENDMENT_O148: 2026-07-20 — "现在，submissions不设置限制，GH200 hour还是
                      1.10，诊断并完成Job A/B WP4的任务。"
EXECUTABLE_NOW: no; authority ended at terminal WP4 outcomes
```

### 6.1 Implementation, commit, and remediation authority

Allowed edits are limited to the five frozen WPs in `PHASE_I_PLAN.md`: new Phase-I
configs; directly required files under
`fl_v3/src/fl_v3/{config,data,models,training,engine,eval,utils}`; the new in-tree
`models/ops/bev_pool/`; directly required changes to `centralized_train.py` and
`build_gt_database.py`; new `s10_phase1_*` scripts; focused tests; `pyproject.toml`
and attribution/NOTICE only for the standalone extension; and active S10 records at
material boundaries. No new network package dependency is allowed.

S00 may make the material linear commits for WP0, WP1, WP2, WP3, WP4 and the final
evidence/review boundary without further owner approval. Before a WP is conformance-
frozen, S00 may repair an implementation defect that solely restores the already-frozen
reference graph/config/data semantics and is demonstrated by the predeclared tests;
this is not permission to choose alternate math. After conformance freeze, autonomous
remediation is limited to output-neutral config consumption, tests, fixtures, runner,
checkpoint I/O, build, logging, and provenance defects.

Any alternate architecture, tensor shape, geometry, normalization, initialization,
augmentation, target/loss/decode, optimizer/scheduler, precision regime, data role or
order, evaluator/metric, seed, candidate, tolerance, performance gate, resource, or
output scope requires owner escalation. `fl_v2/`, `fl_v3/collab/`, Fusion, FL,
attack/defense, broad profiling, environment rebuild, mmdet3d/mmcv runtime dependency,
merge, push, upload, and publication remain outside the envelope.

### 6.2 Exact checkpoint acquisition

```text
ROLE: Camera-primary ImageNet-1K Swin-T backbone initialization
REFERENCE: MIT BEVFusion Camera YAML at 326653dc06e0938edf1aae7d01efcd158ba83de5
SOURCE_URL: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
LICENSE_GATE: upstream Swin repository MIT license; fail closed on conflicting asset terms
FINAL_PATH: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/pretrained/swin_tiny_patch4_window7_224.pth
QUARANTINE_PATH: same path plus .download.part
ACQUISITIONS: one HTTPS GitHub-release download after approval; final redirect host
              must be release-assets.githubusercontent.com; record final URL
ACCEPTANCE: physical SHA-256 + state-dict schema + strict tensor mapping +
            loaded/missing/unexpected report + initialization-state hash before rename/use
NUIMAGES: swint-nuimages-pretrained.pth is forbidden in Envelope A
EXISTING_TORCHVISION_CACHE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home/hub/checkpoints/swin_t-704ceda3.pth
EXISTING_TORCHVISION_SHA256:
                            704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b
SUBSTITUTION: forbidden; the torchvision object is mapping evidence only
```

The upstream YAML supplies no trusted digest, so the physical SHA-256 is an acquisition
output rather than an approval input. Bytes remain quarantined and unusable unless all
acceptance checks pass. A mapping-code defect may be repaired under pre-freeze
reference-conformance authority without a new download; a redirect outside the
allowlist, changed content/schema, conflicting license, or second download stops the
phase.

### 6.3 Frozen data and materialization

```text
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
ROLE: D_fit only / 34 logs / 494 scenes / 19,877 samples
TRAIN_CACHE_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop1_cache_t1v2_1f276b9d2cc5/info_cache_msweep10
TRAIN_CACHE: t1.v2 / cache depth 10 / canonical SHA-256
             310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a
TRAIN_CACHE_PICKLE_SHA256: 57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30
TRAIN_CACHE_SIDECAR_SHA256: f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c
ZIP_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite
ZIP_LOGICAL_SHA256: 023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6
ZIP_FILE_SHA256: 228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb
DATA_ROOT: module nuScenes-data/1.0-map-1.3-zip via NUSCENES_DATA_DIR; read-only
DATA_ARTIFACT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_data_e321aed749fd
ENGINEERING_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd
CUDA_BUILD_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_<IMPLEMENTATION_SHA12>
OUTPUT_POLICY: roots must be absent before first write; raw outputs are immutable;
               each run subdirectory binds its executable SHA and submission index
```

The accepted ten-sweep cache is metadata capacity, not permission to consume ten
sweeps. WP1 must represent cache depth and consumed point depth separately: LiDAR train
and GTDB consume exactly the keyframe (`point_sweeps=1`); evaluation remains
keyframe plus nine sweeps (`point_sweeps=10`). The identity records both values and the
loader fails if it reads a sweep outside the declared consumption depth.

CBGS uses the pinned official algorithm over D_fit; WP1 statically emits `N_cbgs`, the
expanded index/order/remainder hashes, and effective update counts before any GPU job.
GTDB uses all ten classes in official order, five-point minimum for every class, no
difficulty-policy change from `filter_by_difficulty=[-1]`, no build-time per-class
truncation, reference sample groups
`car:2, truck:3, construction_vehicle:7, bus:4, trailer:6, barrier:2,
motorcycle:6, bicycle:6, pedestrian:2, traffic_cone:2`, collision rejection at any BEV
overlap, no extra yaw jitter, keyframe five-dimensional points with intensity and ring
preserved, and D_fit source-token proof. GT-paste runs only in epochs 1-15 of the later
Envelope-B training (zero-based epoch indices 0-14). No raw dataset extraction or
duplication is allowed.

### 6.4 Frozen CUDA-pooling correctness and performance gates

The optimized backend is an independent in-tree port of the pinned MIT operation, or a
functionally equivalent kernel, with Apache-2.0 attribution and no mmdet3d/mmcv runtime.
The fallback is a sorted explicit segment-sum oracle, not atomic `scatter_add_`.
Both backends receive contiguous FP32 values and int32 geometry under FP32 and accepted
FP16-autocast regimes; pooling accumulation and output are FP32 and the downstream cast
boundary is identical.

Correctness gates, fixed now and not relaxable after failure:

- ranks, cell membership, shapes, dtypes, empty/singleton/collision behavior: exact;
- standalone forward, both regimes: `rtol=1e-5`, `atol=1e-6`;
- standalone gradient with respect to pooled point features: exact (`rtol=atol=0`);
- integrated B4 Camera BEV/loss/upstream-gradient parity: FP32
  `rtol=1e-4`, `atol=1e-6`; accepted FP16 policy `rtol=2e-3`, `atol=2e-4`;
- zero nonfinite values, identical parameter coverage, and report the worst tensor plus
  maximum absolute/relative error.

Timing uses four fixed production B4 batches from the frozen order. Operator timing is
32 warm-ups plus 128 CUDA-event samples per backend. End-to-end timing resets identical
model/optimizer/RNG/data state for each backend and uses 16 warm-up plus 64 timed B4
microbatches with accumulation 8. Report GPU operator/step median and p95, wall samples/s,
loader wait, peak allocated/reserved memory, and initialization/accepted-window state.

Production promotion requires all correctness gates, optimized operator median at most
`0.80x` fallback (at least `1.25x` speedup), optimized end-to-end median GPU-step time at
most `1.02x` fallback, and peak allocated memory at most `1.05x` fallback. P95 is a
reported guardrail, not a post-hoc selector. Failure leaves the fallback available only
for diagnosis and blocks Envelope B; it does not create another Camera candidate.

### 6.5 Exact submissions, resources, and exit

```text
ACCOUNT/PARTITION: naiss2025-22-1113-gpu / gpu
PER_SUBMISSION: 1 node, 1 task, 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
MAX_CONCURRENCY: 1
MAX_SUBMISSIONS: unlimited for serial WP4 engineering smoke under O-148
AGGREGATE_GPU_HOURS: <=1.10 charged GH200-hours across all submissions
JOB_A: Camera extension build, correctness gates, operator timing, optimized/fallback
       end-to-end calibration
JOB_B: D_fit keyframe GTDB materialization and identity checks, then LiDAR 16-warm-up /
       64-timed-B4 production calibration
DERIVED_REPLACEMENTS: under O-148, fresh-output serial replacements for diagnosed
                      in-scope engineering defects while aggregate time remains
JOB_D: O-147-authorized fresh-output Camera replacement from exact commit c45e020...;
       no replacement if it fails
```

Under O-148, each branch was repaired and resubmitted serially until an honest
terminal WP4 outcome. Engineering defects did not stop for per-job approval. The
loop stopped at a passed validation or a frozen scientific/performance gate rather
than changing math or tolerances. O-149 retains that completion-oriented pattern
for future explicitly approved validation envelopes.

Envelope A closed after WP0-WP4 implementation, focused validation, exact checkpoint/
CBGS/GTDB identities and both honest calibrations. Camera failed the frozen backend
promotion gate, so combined recipe freeze and a normal measured Envelope-B request
did not follow automatically. No capability metric, D_select, D_audit, official
validation, 20-epoch training or selectable model ran.

The single approval sentence is:

```text
批准激活 commit <REQUEST_COMMIT> 中的 S10 Phase I Envelope A，并按其中边界连续执行 WP0-WP4。
```

## 7. Envelope B — exact scientific branch-qualification request

IP-G0 supersedes this section's immediate activation path while Phase I-P is
open. The object below is preserved unchanged in scientific content as the frozen
control for cost/payback comparison; it is not executable and must be re-disposed
at IP-G2, with new source/config/resource identities if Phase I-P promotes changes.

This is the complete O-150 Envelope-B object. Its containing commit is the
activation baseline named by the owner's approval; implementation baseline
`a1d7d4fc9508875cc7559858b51b9c1fe441f69b` contains the exact runner/config
code. No job may be submitted until the owner approves both the resource envelope
and the one independent review-only subagent. The review must close before the
first scientific submission.

```text
PHASE: S10 Phase I / Envelope B independent Camera and LiDAR qualification
REQUEST_STATE: FROZEN / OWNER APPROVAL PENDING / NOT EXECUTABLE
ACTIVATION_BASELINE: the commit containing this Section 7, named verbatim by owner
IMPLEMENTATION_BASE_SHA: a1d7d4fc9508875cc7559858b51b9c1fe441f69b
IMPLEMENTATION_BASE_TREE: a13f0fd8823575ceebf8d26d265e2c3c88af8919
BRANCH: codex/s10-phase1-branch-qualification
OBJECTIVE: train the two frozen unimodal primaries to their exact terminal exposure,
           evaluate each terminal raw checkpoint once on D_select, and return both
           results to P1-G2 without post-hoc tuning or a capability threshold
CANDIDATES_AND_MAX_COUNT: exactly 2 scientific candidates — phase1_lidar_primary
                          then phase1_camera_primary; no replacement candidate
SEED_POLICY: seed 0 only; no alternate or confirmatory seed in Envelope B
CAMERA_PRODUCTION_BACKEND: pytorch_sorted_segment_reduce
CAMERA_OPTIONAL_BACKEND: optimized_cuda_unpromoted / forbidden in Envelope-B runs
DATA: accepted STOP-A train-parent split; D_fit train, D_select terminal assessment;
      D_audit owner-sealed; official validation forbidden
SPLIT_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
D_FIT: 34 logs / 494 scenes / 19,877 base samples
CBGS: official frozen artifact / expanded 87,930 / consumed 87,904 per epoch /
      drop 26 after deterministic epoch permutation
TRAINING_EXPOSURE: 20 epochs / 1,758,080 attempted sample presentations /
                   54,940 attempted effective-B32 windows per candidate
BATCH: one GH200 / physical B4 / accumulation 8 / effective B32 / workers 8
PRECISION: global FP16 autocast; Camera pool/loss FP32; LiDAR accepted sparse FP32 island;
           GradScaler initial scale 8; TF32 off
CHECKPOINT: one atomic epoch-boundary recovery retained; raw epoch-20 terminal only
            is selectable; exact RNG/optimizer/scheduler/scaler/config identity resumes
EVALUATOR: fl_v3 train-subset adapter over official nuScenes detection metric math;
           detection config SHA-256
           217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b
D_SELECT: one completed execution per terminal candidate; no re-decode for selection
D_AUDIT: resource reserve only; remains inaccessible until explicit P1-G2 OPEN D_audit
OFFICIAL_VALIDATION: forbidden
CHECKPOINT_SELECTION: epoch-20 terminal raw weights only; no best-epoch selection
AGGREGATE_GPU_HOURS: 49.0 charged GH200-hours across review-following execution,
                     engineering incidents, recovery/resume and any later P1-G2-opened
                     D_audit execution; unused Envelope-A time is not transferred
MAX_CONCURRENCY: 1
SUBMISSION_POLICY: no numeric engineering-submission cap; exactly two scientific
                   candidates; serial LiDAR then Camera; no duplicate scientific rerun
PER_JOB_RESOURCE: 1 node / 1 GH200 / 16 CPUs / 96 GiB
PLANNED_WALL: LiDAR 14:00:00; Camera 35:00:00; actual aggregate charge must remain <=49.0
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102
ENGINEERING_REMEDIATION: O-149 output-neutral config/schema/API/test/runner/checkpoint/
                         provenance/logging repairs and serial resume are autonomous
SCIENTIFIC_CONTINUATION: a weak LiDAR score does not cancel Camera unless it implicates
                         a shared data/evaluator/precision/configuration boundary
REVIEW_GATE: one independent review-only subagent; no edits and no GPU; inspect the
             activation baseline plus frozen artifacts; compute opens only with no
             open P0-P2, while any P3 is recorded as residual risk
OWNER_ESCALATION: any model math/shape, normalization, initialization, data/order/GTDB,
                  augmentation, target/loss/decode, optimizer/scheduler/precision, seed,
                  exposure/checkpoint-selection, evaluator/metric, candidate, output or
                  aggregate-resource change; ambiguous defect; same blocker after repair;
                  shared-boundary failure; or 49.0-hour exhaustion
ALLOWED_INTERPRETATION: single-seed internal C/L branch capability and engineering health
FORBIDDEN_INTERPRETATION: official-val/generalization, CUDA promotion, fusion, FL,
                          attack/defense, publication claim, or best-recipe optimality
OWNER_APPROVAL: pending
EXECUTABLE_NOW: no
```

### 7.1 Immutable candidates and executable identities

| Branch | Resolved config SHA-256 | Config file SHA-256 | Initialization/materialization |
|---|---|---|---|
| Camera | `e95e65a63a32c494296b38baf98fd913ff1ec6a168b78aabac48a8dc8f0ffe1d` | `567cb1b71535b4866193273960e531ae4b45318e56e81101e99ad186ac23ce60` | ImageNet Swin-T physical `9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3`, mapping `c87469b84b4b865aa478cc1959c400468f8aca393e53cf8dbb92a71c3a63f70f`, initialized state `814eaf5adb58ecc5b5cfe253c63002bc6cad9390c01752b890298571fed01632`; fallback only |
| LiDAR | `0efe4d6d5138e3d99ae80254a6ecf884300dd18985ab45a00425228fc3ef082e` | `c7e1fa26e1714a31c5998296cb95cbab5e8732d4bf2f06da81fd6d631c574bfc` | scratch seed 0; accepted role-bound GTDB manifest `22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5`; keyframe train |

The direct dual-branch entry is `fl_v3/scripts/s10_phase1_capability.py`
(`0f7a1a958d3b404a1c308c8dd6251cdb81bd07a0d26c6d4db98c3bb81a3ac6e5`)
through `fl_v3/scripts/run_s10_phase1_envelope_b.sh`
(`0461a96904c5258a07b79348f0de8c55e5bf38d0686fcb2f758f66b8aff2cbec`).
It performs direct config/runtime/data identity checks, one output-neutral D_fit B4
forward/backward/decode and zero-boundary checkpoint round-trip, then reconstructs
the production state before training. Recovery is epoch-boundary only. A derived
output-neutral source may resume the same config/checkpoint and is recorded per
attempt; a scientific source change is prohibited.

### 7.2 Evidence-based resource calculation

Both candidates consume `20 * 87,904 = 1,758,080` actual presentations. Job H's
accepted fallback end-to-end calibration measured `16.3513808747 samples/s`; Job
B5 measured `41.9043778095 samples/s`. This gives `29.866319` Camera hours and
`11.654046` LiDAR hours. Existing 4,626-sample D_select runs took approximately
`0.140-0.192` hours each. A conservative combined `0.80` hours covers both
D_select evaluations, checkpoint/preflight overhead and a still-sealed proportional
D_audit reserve. Therefore:

```text
training subtotal = 29.866319 + 11.654046 = 41.520365 h
with eval/checkpoint/audit reserve          = 42.320365 h
15% contingency                            = 48.668420 h
requested aggregate ceiling (rounded up)   = 49.000000 GH200-hours
```

The calibration reused prefetched batches, so it is not a sustained-loader proof;
the 15% contingency covers that residual. S09's eight-worker run observed only
`0.076%` data wait, supporting—but not proving—that compute remains dominant.
The ceiling is actual charged time, not a promise to consume all 49 hours.

### 7.3 Serial command family after approval and review

`<ACTIVATION_SHA>` is replaced only with the exact commit named in owner approval
or an O-149-recorded output-neutral derived source. First create the `slurm/` log
directory beneath the frozen output root, then submit LiDAR:

```bash
sbatch --parsable --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=96G --gpus=1 --time=14:00:00 --job-name=s10-p1b-lidar \
  --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102/slurm/lidar-%j.out \
  --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102/slurm/lidar-%j.err \
  fl_v3/scripts/run_s10_phase1_envelope_b.sh --branch lidar \
  --config fl_v3/configs/s10_phase1_lidar.json \
  --output-dir /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102/phase1_lidar_primary \
  --source-sha <ACTIVATION_SHA>
```

After LiDAR reaches a terminal result—or an independent weak result not implicating
a shared boundary—submit Camera with the same fixed resource shape and:

```bash
sbatch --parsable --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=96G --gpus=1 --time=35:00:00 --job-name=s10-p1b-camera \
  --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102/slurm/camera-%j.out \
  --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102/slurm/camera-%j.err \
  fl_v3/scripts/run_s10_phase1_envelope_b.sh --branch camera \
  --config fl_v3/configs/s10_phase1_camera.json \
  --output-dir /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_2a26c63b6102/phase1_camera_primary \
  --source-sha <ACTIVATION_SHA>
```

An exact recovery uses the same branch/config/output and adds `--resume`; its wall
request is limited to the remaining aggregate budget. No retry changes candidate,
seed, data, model, recipe, precision, terminal-selection or evaluator semantics.

The single combined approval sentence is:

```text
批准激活 commit <ACTIVATION_SHA> 中的 S10 Phase I Envelope B：授权一次独立
recipe-freeze review-only subagent；review 无 open P0-P2 后，按 Section 7 的两个
固定候选、seed 0、数据/评估/训练/checkpoint 边界、49.0 GH200-hour 总 ceiling、
最大并发 1 和 O-149 remediation 规则，串行执行 LiDAR 再 Camera；D_audit 仍封存。
```

## 8. Phase I-P IP-G0 record and exact IP-E1 request

IP-G0 freezes the Phase I-P workflow, candidate classes, measurement protocol,
topology and stop boundaries in `HANDOFF.md` Section 1. It authorized WP0 source,
docs, tests, local validation and linear commits from the unique base. The owner
then activated this phase-sized request after WP0 source/static close; its first
GH200 reference also supplies WP0's architecture-specific runtime close.

```text
PHASE: S10 Phase I-P / IP-E1 baseline and strict-output-neutral diagnosis
REQUEST_STATE: CLOSED / TERMINAL
ACTIVATION_BASELINE: 85c6719e4b880b198d850e16b1418c230fa5c656
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
BRANCH: codex/s10-phase1p-throughput-preflight
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA; do not move
OBJECTIVE_AND_EXIT_GATE: establish real sustained Camera/LiDAR B4xaccum8 baselines,
  localize whole-model bottlenecks, validate checkpoint continuation with materialized
  AdamW state, and measure only the frozen strict-output-neutral shortlist; exit with
  a trustworthy cost ledger and an IP-G1 shortlist, not a capability result
REFERENCE_VARIANTS: exactly Camera reference and LiDAR reference from Section 7.1
STRICT_OUTPUT_NEUTRAL_VARIANTS_MAX: seven named implementation groups plus at most
  one accepted per-branch combination: augmentation host/unused-return cleanup;
  training-field whitelist; fixed Camera meshgrid cache; batched Camera affine/grid
  construction while retaining per-image interpolation/grid_sample; LiDAR host
  counts/offsets; consolidated target/Hungarian D2H retaining SciPy/float64/order/ties;
  checkpoint CPU-snapshot/hash reuse
FORBIDDEN_IP_E1_VARIANTS: finite-loss window aggregation; async checkpoint; physical
  B8/B12/B16; checkpoint cadence change; Camera/LiDAR SDPA; torch.compile/CUDA graph;
  foreach/fused AdamW; activation checkpoint; persistent workers; optimized CUDA BEV pool
DEFAULTS: every candidate flag off in the reference profile and in Phase-I configs
DATA: D_fit only; exact accepted STOP-A split and physical identities
SPLIT_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
D_FIT: 34 logs / 494 scenes / 19,877 base samples
CBGS: artifact 64cc0d1d6cd82fae2787d397e610178cedd00887d98938b154fce9f8e8e115ef;
  expanded 87,930 / consumed 87,904 / drop 26 per epoch
LIDAR_GTDB: exact keyframe-only D_fit manifest
  22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5
SEED_POLICY: seed 0 only
BATCH_AND_EXPOSURE: physical B4 / accumulation 8 / effective B32; bounded profiler
  prefixes only; no scientific epoch and no change to 20-epoch Phase-I exposure
PRECISION_AND_UPDATE: frozen global FP16; Camera FP32 pool/loss; LiDAR sparse FP32
  island; real loss/backward/clip/GradScaler/AdamW/cyclic scheduler and accepted updates
MEASUREMENT: 16 accepted warm-up windows + 256 accepted measured windows; two fresh
  process repeats per reference/final candidate; third only when spread >3%; separate
  three-accepted-window torch CPU/CUDA trace; one-second system sampling
MEMORY_GATE: after AdamW state materialization, peak reserved <=85% visible memory and
  no monotonic sustained-window growth
PARITY: exact/hash exact where attainable; FP32 rtol=1e-4 atol=1e-6; FP16
  rtol=2e-3 atol=2e-4; discrete state, counters, BN/scaler/scheduler and identities exact
CHECKPOINT: real save/model hash/file hash/load timing; release the original stack;
  fresh reconstruction and exact boundary load; compare the next 8 optimizer windows
  with uninterrupted control including state/RNG/sampler identity
PER_JOB_RESOURCE: 1 node / 1 GH200 / 16 CPUs / 96 GiB / <=00:45:00 / no requeue
BASE_AGGREGATE_GPU_HOURS: 2.0 charged GH200-hours for approved WP1/WP2 cells
CODE_BUG_REMEDIATION_RESERVE: +1.0 charged GH200-hour, usable only after a diagnosed
  code-level bug; it is not candidate/science/measurement expansion authority
HARD_AGGREGATE_GPU_HOURS: 3.0 charged GH200-hours including that conditional reserve
TARGET_CONSUMPTION: <=0.8 charged GH200-hour; target is not an extra stop authority
MAX_CONCURRENCY: 1
SUBMISSION_POLICY: serial; no numeric engineering-submission cap; no blind identical
  retry; reference cells precede derived strict candidates; recurring code-level
  defects are diagnosed and repaired continuously rather than causing a mechanical stop
OUTPUT_ROOT_RULE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_<approved_sha12>/
  <camera|lidar>/<sustained|trace>_<source_sha12>_r<1|2|3>_<attempt-id>
FRESH_OUTPUT: every attempt path absent before execution; no overwrite; raw evidence immutable
DERIVED_SOURCE_POLICY: only linear source/configs implementing the seven frozen strict
  groups or O-149 repairs; every derived SHA/profile hash/options/output are ledgered
  before its first submission; default Phase-I behavior remains off; no new candidate
ENGINEERING_REMEDIATION: diagnose, make the smallest frozen-semantics runner/test/config/
  dtype/API/checkpoint/provenance/logging repair, validate, record, and resubmit serially;
  there is no retry-count stop for code bugs while the applicable layered ceiling remains
OWNER_ESCALATION: ambiguous diagnosis; scientific-boundary pressure; parity/tolerance
  failure not attributable to an unambiguous code defect; nonfinite loss or discarded
  partial window; unresolved >3% baseline instability; memory leak or >85% reserved;
  need for a non-listed candidate, changed scientific/operational semantics, or hard
  aggregate-ceiling exhaustion. A diagnosed code bug or recurring code symptom alone
  does not interrupt the workflow and may consume only the +1.0-hour reserve.
ALLOWED_INTERPRETATION: D_fit-only throughput, bottleneck, numerical-health, memory,
  checkpoint and projected charged-GH200-hour engineering evidence
FORBIDDEN_INTERPRETATION: mAP/NDS, capability, generalization, model/candidate selection,
  D_select/D_audit/official validation, fusion, FL, attack/defense, publication claim,
  or activation of Section 7 Envelope B
OWNER_APPROVAL: granted in the task on 2026-07-20 for WP0 runtime close and continuous
  IP-WP1 -> IP-WP2, with the conditional +1.0-hour code-bug reserve above
OWNER_CLOSURE: after the three ordered Camera candidates reached terminal results,
  the owner closed IP-WP2 and IP-E1 and opened IP-G1 discussion
EXECUTABLE_NOW: no; all Section-8 execution and remediation authority has expired
```

### 8.1 WP0 implementation identities

The containing activation commit carries the profiler implementation. Identities
that do not depend on that documentation commit are:

| Artifact | SHA-256 / identity |
|---|---|
| Camera source config | file `567cb1b71535b4866193273960e531ae4b45318e56e81101e99ad186ac23ce60`; resolved `e95e65a63a32c494296b38baf98fd913ff1ec6a168b78aabac48a8dc8f0ffe1d` |
| LiDAR source config | file `c7e1fa26e1714a31c5998296cb95cbab5e8732d4bf2f06da81fd6d631c574bfc`; resolved `0efe4d6d5138e3d99ae80254a6ecf884300dd18985ab45a00425228fc3ef082e` |
| IP-E1 reference profile | file `f51c7a62f93887d8cffcb125ebd1cbeb6fc3f3911ec3378fa96a82eaf01f48e0`; canonical `7b9aede3daa9b4b605ee34a64e4c672dc6136ef2e6a47ccdd60b94b26a7ed949` |
| profiler entry | `e1ffda1b0ede860c3d4886050ad9d4dcfb054f436490da8991bdfa0fe3d5955e` |
| exact environment wrapper at activation | `e4fb3b5b530ed8eea4f2980e7fdb2059dbc056e2629fd70257e50476f1770caa` |
| accumulation-aware training loop | `170b1b5614b81dae059e257c3b5de86ebeaf0362f62fccdf3f0525c50abc263c` |
| fail-closed profile parser | `fe3468ff5828fe73b13359c64414138b9fd19dcb06feed8173b8a0c538d31685` |

The reference profile binds every candidate off and the exact output prefix. The
entry imports no evaluator and rejects non-D_fit roles. It shares production stack
construction with `s10_phase1_capability.py`; sustained timing and short torch
tracing are deliberately separate.

### 8.2 Initial exact reference cells

| Order | Branch/mode | Required repeats | Purpose |
|---|---|---|---|
| 1 | LiDAR sustained | 1, 2; conditional 3 if spread >3% | real B4x8 baseline, memory, checkpoint and continuation |
| 2 | LiDAR trace | 1 | three active accepted windows after the same warm-up |
| 3 | Camera sustained | 1, 2; conditional 3 if spread >3% | real B4x8 baseline, memory, checkpoint and continuation |
| 4 | Camera trace | 1 | three active accepted windows after the same warm-up |

Strict candidate attempts follow this reference order serially. Before each first
candidate attempt, append its durable source SHA, profile file/canonical hashes,
single changed option, parity result and exact output path. A candidate that fails
parity is not timed or combined. Only an accepted per-branch combination receives
the two-repeat final sustained protocol.

### 8.3 Exact command family after owner approval

For each row set `<BRANCH>`, `<MODE>`, `<CONFIG>`, `<PROFILE_CONFIG>`, `<REPEAT>`
and `<ATTEMPT_ID>` from Section 8.2 or the pre-recorded strict-candidate entry.
The reference uses `fl_v3/configs/s10_phase1p_ip_e1.json`. `<SOURCE_SHA>` is the
activation SHA or a permitted recorded linear descendant; `<APPROVED_SHA>` remains
the exact activation SHA named by the owner. Create only `<OUTPUT_ROOT>/slurm`
before the first approved submission.

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 \
  --mem=96G --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 \
  --no-requeue --job-name=s10-ip-e1-<BRANCH>-<MODE> \
  --output=<OUTPUT_ROOT>/slurm/<BRANCH>-<MODE>-%j.out \
  --error=<OUTPUT_ROOT>/slurm/<BRANCH>-<MODE>-%j.err \
  fl_v3/scripts/run_s10_phase1p_ip_e1.sh \
  --branch <BRANCH> --mode <MODE> --config <CONFIG> \
  --profile-config <PROFILE_CONFIG> \
  --output-dir <OUTPUT_ROOT>/<BRANCH>/<MODE>_<SOURCE_SHA12>_r<REPEAT>_<ATTEMPT_ID> \
  --source-sha <SOURCE_SHA> --approved-source-sha <APPROVED_SHA> \
  --repeat <REPEAT> --attempt-id <ATTEMPT_ID>
```

The owner activated the request at implementation commit
`85c6719e4b880b198d850e16b1418c230fa5c656` and added the conditional code-bug
reserve. The normalized decision is:

```text
批准激活 commit 85c6719e4b880b198d850e16b1418c230fa5c656 中的 S10 Phase I-P
IP-E1：按 Section 8 的
D_fit-only reference cells、七个 strict-output-neutral implementation groups、
测量/parity/checkpoint/停止边界、2.0 charged-GH200-hour base ceiling，以及仅供
已诊断代码级 bug 修复的 +1.0 charged-GH200-hour reserve（总硬上限 3.0）、
最大并发 1、每 job 45 分钟和 O-149 remediation 规则串行执行 IP-WP1 -> IP-WP2；
代码 bug 不设重跑次数上限且不中断连续工作流，但禁止盲目原样重跑；
measurement-only 候选保持 default-off，D_select、D_audit、official validation、
原 Envelope B、merge 和 push 仍不授权。
```

### 8.4 Compact IP-E1 execution ledger

Budget accounting separates ordinary approved cells from the conditional code-bug
reserve. Raw Slurm logs and any created attempt directory are immutable.

| Attempt | Source / profile / output | Terminal state and diagnosis | Charged GH200-hours |
|---|---|---|---:|
| Job `525163`, LiDAR sustained reference r1 | source `07d7db10db3f2c5cf92b774b1ba0511157fa16e8`; approved source `85c6719e4b880b198d850e16b1418c230fa5c656`; profile file `f51c7a62...f48e0`, canonical `7b9aede3...ed949`; intended attempt `lidar/sustained_07d7db10db3f_r1_reference`; Slurm logs under the approved root | `FAILED_PRE_MODEL` in `00:00:05`; Slurm executed its copied batch script from `/var/lib/slurm`, so the wrapper derived a nonexistent spool-relative repository path and failed at config `realpath`; no config, runtime, data, model or update executed. Unambiguous runner-path bug; replacement must bind `source_root` to `SLURM_SUBMIT_DIR`, use a fresh attempt path and a new linear source. | reserve `0.001389`; base `0.000000` |
| Job `525168`, LiDAR sustained reference r1 path-fix replacement | source `b2ee9900cdc4968180bf90e39c10e62db94cac1b`; same approved source/profile; attempt `lidar/sustained_b2ee9900cdc4_r1_pathfix`; immutable raw artifacts retained | `FAILED_CHECKPOINT_PARITY` in `00:08:58` after the complete 16+256-window main interval. The real 100,014,315-byte checkpoint loaded with exact boundary model/optimizer/scheduler/scaler, training state and RNG. After eight continuation windows, scheduler/scaler/state/RNG remained equal but model and optimizer failed unchanged FP16 allclose: global relative L2 `1.34867e-4` and `6.42201e-5`, with 119/102 per-tensor failures. The runner did not persist the already-complete main measurement before the late gate, so no throughput verdict is recoverable. Diagnosis remains open between input-stream drift, permitted sparse-kernel nondeterminism and a comparison-protocol defect; tolerances are unchanged. | base `0.149444`; reserve `0.000000` |
| Job `525192`, LiDAR sustained r1 checkpoint diagnostic | source `a8a0b1498d020a51902d3008bfbcb65c4eaa3649`; same approved source/profile; attempt `lidar/sustained_a8a0b1498d02_r1_ckptdiag`; measurement `5208e80d...e4ae`; terminal result `62e8249c...b6b`; checkpoint `7a5e12ad...4379`; worker result `da05fada...f173` | `FAILED_CHECKPOINT_PARITY` in `00:08:42`, with valid persisted main measurement: 256/256 accepted, `40.4214` presentations/s, `1.26317` updates/s, mean loader wait `1.5169 ms/window`, peak allocated/reserved `5,534,907,904 / 6,958,350,336` bytes (`6.8215%` reserved), no scaler skip/nonfinite. All 64 continuation microbatch hashes, state and RNG were exact. Boundary restore was exact. Both same-process replay and fresh-process replay failed per-tensor FP16 allclose while remaining globally small: model relative L2 `1.32064e-4 / 6.56332e-5`, optimizer `6.17808e-5 / 6.20772e-5`. This localizes the divergence to permitted runtime-kernel nondeterminism rather than loader order, checkpoint I/O or fresh-process reconstruction. The frozen elementwise gate was not changed. One-repeat projection is `12.1294` GH200-hours for 20 epochs using the end-to-end `167.341 s` pre-training startup plus measured per-epoch checkpoint/hash cost; it is preliminary, not a final baseline. The measured save+file-hash+model-hash cost is only `0.23468 s/epoch` (`4.69 s` over 20 epochs), so lowering checkpoint cadence is not a material LiDAR speed lever. | reserve `0.145000`; base `0.000000` |
| Job `527225`, LiDAR sustained reference r2 | source `3b23a7df4818d34524ac3b01e88cba04b46ae82a`; same approved source/profile; attempt `lidar/sustained_3b23a7df4818_r2_calibrated`; measurement `55f0b2b6...0708`; result `b1645479...fe98`; complete `3d9f48dd...96f`; checkpoint `a8b4d136...500d`; worker result `0a4b61d9...ca76` | `COMPLETE_SUSTAINED` in `00:09:01`: 256/256 accepted, `38.0943` presentations/s and `1.19045` updates/s; mean/p95 loader wait `2.1058 / 2.0121 ms/window` with one `67.22 ms` maximum; peak allocated/reserved unchanged at `5,534,907,904 / 6,958,350,336` bytes; no scaler skip/nonfinite/discard. Exact boundary/input/RNG/training/discrete state and all five calibrated continuation groups PASS; per-element allclose remains diagnostic FAIL. r1-r2 spread is `5.928%`, so the frozen protocol triggers repeat 3. | base `0.150278`; reserve `0.000000` |
| Job `527229`, LiDAR sustained conditional r3 | same source/profile; attempt `lidar/sustained_3b23a7df4818_r3_calibrated`; measurement `ec4f5c66...196d`; result `31b5c5cb...6b1f`; complete `9b6eb190...2e4e`; checkpoint `4026492a...358d`; worker result `fd0cbe48...5070` | `COMPLETE_SUSTAINED` in `00:07:36`: 256/256 accepted, `36.9148` presentations/s and `1.15359` updates/s; mean/p95 loader wait `1.9870 / 4.1063 ms/window`; memory and numerical-health gates unchanged; exact boundary/input/RNG/training/discrete state and all five calibrated groups PASS. Three-repeat rates are `40.4214 / 38.0943 / 36.9148`; min-max spread/mean is `9.113%`, while same-node r2-r3 spread remains `3.145%`. Measurement-window GPU utilization/power declines `56.81% / 289.84 W -> 53.27% / 285.55 W -> 52.31% / 280.12 W` at fixed reported clocks; loader, input identity, memory and accepted-update evidence do not explain the change. Conditional repeat is exhausted and the unresolved-instability owner stop is reached. | base `0.126667`; reserve `0.000000` |
| Job `527239`, Camera sustained reference r1 | source `e3a42364236a6d4a237c3ad01d97d90656cb9de8`; same approved source/profile; attempt `camera/sustained_e3a42364236a_r1_reference`; measurement `c17b369f...3a67`; result `bf32835f...ba3`; failed `d2d8485b...6d43`; checkpoint `b45b3c4e...8b81`; worker result `a0cdc8fd...4426`; nvidia sampling `ff1f72c5...e2c1` | `FAILED_CHECKPOINT_PARITY` in `00:12:45` after a valid main interval: 256/256 accepted, `16.5390` presentations/s and `0.516845` updates/s, loader mean/p95 `2.3420 / 2.5142 ms/window`, peak allocated/reserved `16,038,963,200 / 18,723,373,056` bytes (`18.3553%`), zero overflow/nonfinite/discard. Boundary, 64 input hashes, RNG, training/discrete state are exact. The grouped gate fails without changing tolerances: fresh/control ratios reach `1.475/1.559` for Adam `exp_avg`, `1.399/1.297` for `exp_avg_sq`; BN mean max-abs, BN var relative-L2 and model-parameter max-abs also exceed their 1.25x/frozen limits. The 525,165,739-byte checkpoint's save+file/model-hash cost is `0.8053 s/epoch`; the one-repeat 20-epoch projection is about `29.56` GH200-hours but remains preliminary. This is not a code defect; Camera r2/trace stop for owner disposition. | base `0.212500`; reserve `0.000000` |
| Job `527247`, Camera trace r1 diagnostic | source `2d2f71749d27c06667030b82ceab2c4eeb16dd7c`; same approved source/profile; attempt `camera/trace_2d2f71749d27_r1_diagnostic`; measurement `8cdeabf4...21e4`; result `c353b9b9...4801`; complete `1fbc7ebe...c02`; trace `76a4c041...8df5`; summary `36a34368...fbf`; nvidia sampling `61aafaf6...37e8` | `COMPLETE_TRACE` in `00:03:48`: 16 accepted warm-up plus 3/3 active accepted windows, zero nonfinite/overflow/discard, mean loader wait `2.7439 ms/window`, peak allocated/reserved `16,027,464,704 / 18,723,373,056` bytes (`18.3553%`). Trace-mode `4.638` presentations/s is overhead-only and not a sustained comparison. CPU range localization ranks forward/backward/loss/optimizer/H2D at `53.774/29.770/12.490/3.632/0.334%` of captured stage time; preprocessing is `48.893%` of forward. Raw trace is 803,491,740 bytes. The cell is diagnosis-only and cannot waive Camera parity, promote a candidate or support a stable speed claim. | base `0.063333`; reserve `0.000000` |
| Job `527276`, Camera augmentation-cleanup r1 | source `efe767a60810c875d0a73bdc2af29f4bf426315c`; candidate profile file `9a9a48b...47cb9`, canonical `cdeed079...9d6e`; attempt `camera/sustained_efe767a60810_r1_augcleanup`; measurement `f6f635a0...766dd`; result `2e0bb207...49e0`; checkpoint `da1a827d...47c4` | `FAILED_CHECKPOINT_PARITY` in `00:14:33` after four focused tests passed and a valid 16+256-window body completed. All 256 measured windows were accepted with zero nonfinite/overflow/discard, but throughput was `14.1774` presentations/s, `14.279%` below the one-repeat reference; projected 20-epoch cost is `34.45` versus `29.53` GH200-hours. Mean loader wait was `2.4784 ms/window`; peak allocated/reserved `16,038,960,128 / 18,723,373,056` bytes was unchanged. Boundary/input/RNG/discrete state were exact; model parameters and BN mean passed the grouped continuation rule, while BN var and both Adam moment groups failed. Different-node stable-load GPU utilization/power was `51.29% / 284.84 W` versus reference `60.67% / 310.51 W`, consistent with but not proof of host stalls. Diagnosis: CPU residency left the tiny tensor in DataLoader-pinned memory and the unchanged loop then performed repeated scalar reads; derive an exact ordinary-CPU/value-table refinement inside the same candidate group before moving to fixed-grid work. | base `0.242500`; reserve `0.000000` |
| Job `527284`, Camera augmentation value-table refinement | source `200e2175e3e02932697ef13ce1368b400f663cf8`; same candidate profile; attempt `camera/sustained_200e2175e3e0_r1_augvalues`; measurement `c3283b15...212d`; result `7e8cf90d...ea39`; complete `283fbe5b...ce6e`; checkpoint `6268ff99...6e6f`; worker `a1e50118...2f78`; nvidia sampling `499e7767...a990` | `COMPLETE_SUSTAINED` in `00:14:05`: pinned-input exact pre-model test and all four focused selectors passed; 256/256 measured windows accepted with zero nonfinite/overflow/discard. One value-table conversion recovered `2.939%` over r1, but `14.5941` presentations/s remains `11.760%` below reference and projects `33.46` versus `29.53` GH200-hours. Loader mean `2.5775 ms/window` and peak allocated/reserved `16,038,960,128 / 18,723,373,056` bytes exclude loader/memory benefit. Boundary/input/RNG/discrete state and all five grouped continuation gates PASS. Stable-load GPU utilization/power was `52.69% / 287.40 W` on `n444`; cross-node variation remains a confounder, but two candidate implementations provide no promotable payback. Reject this candidate group and proceed to the independently flagged fixed grid. | base `0.234722`; reserve `0.000000` |
| Job `527313`, Camera fixed rotation-coordinate grid | source `a8e160ed14c9e4d896f7ce4842f1d0ff4021aa19`; candidate profile file `5d0c00cd...2e77`, canonical `cc5b4a59...7beb`; attempt `camera/sustained_a8e160ed14c9_r1_staticgrid`; measurement `a653cc1a...e7c0`; result `57a7095b...c42e0`; complete `c7040dfe...4f42`; checkpoint `abe0078c...0336`; worker `b08a76cf...d021`; nvidia sampling `67aa98ab...1bb0` | `COMPLETE_SUSTAINED` in `00:12:56`: three focused tests passed, including two-call elementwise exactness, data-pointer reuse and state-dict exclusion. 256/256 measured windows accepted with zero nonfinite/overflow/discard; all exact/context and five grouped continuation gates PASS. Throughput `14.7377` presentations/s is `10.891%` below reference and only `0.984%` above the different-candidate same-node Job 527284, projecting `33.14` GH200-hours. Peak active/reserved was `16,045,518,848 / 18,673,041,408` bytes: the tiny cache raises active by 6.56 MB while reducing reserved fragmentation by 50.33 MB, but gives no promotable speed evidence. Stable-load GPU utilization/power `53.66% / 291.96 W`. Reject without repeat; proceed to exact-gated batched affine/grid. | base `0.215556`; reserve `0.000000` |
| Job `527323`, Camera conservative batched affine/grid | source `9df760ac2f743207b5d631e216a7da1b1c2d3b78`; candidate profile file `fa529745...b4f1`, canonical `6a809032...a19b`; attempt `camera/sustained_9df760ac2f74_r1_batchedgrid`; measurement `11c327ce...c661`; result `02b2f6d3...b79d`; complete `664859a5...faeb`; checkpoint `84bf3f79...06cf`; worker `50ed18f5...de4`; nvidia sampling `463642fd...91cc` | `COMPLETE_SUSTAINED` in `00:13:21`: pre-model CPU+CUDA elementwise equality and exact candidate mapping passed; 256/256 measured windows accepted with zero nonfinite/overflow/discard; all exact/context and five grouped continuation gates PASS. At `15.7309` presentations/s it is the best candidate and `6.739-7.790%` above the two slower candidates, but still `4.886%` below the one-repeat reference and projects `31.04` rather than `29.53` GH200-hours. Mean loader wait `2.2156 ms/window`; peak allocated/reserved `16,038,303,744 / 18,717,081,600` bytes (`18.3491%`); stable-load utilization/power `56.98% / 305.09 W` on `n449`. Retain default-off as a later engineering anchor, but do not repeat, combine or promote. The three owner-ordered candidates are terminal; pause lower-ranked WP2 work for owner discussion. | base `0.222500`; reserve `0.000000` |

Current accounting after Job `527323`: base cells `1.617500 / 2.0`, code-bug
reserve `0.146389 / 1.0`, hard aggregate `1.763889 / 3.0` charged GH200-hours.
The owner then closes IP-WP2 and IP-E1 without running the lower-ranked field-
whitelist or target-D2H items and opens IP-G1 discussion. All unused IP-E1 budget
expires; it does not authorize IP-WP3, IP-E2 or any additional job.
The worktree-path repair is sealed at source `b2ee9900cdc4968180bf90e39c10e62db94cac1b`
with wrapper hash `4f532d1e...c564d`. The next derived diagnostic persists the
main measurement before checkpoint work, hashes all 64 continuation microbatches,
and adds a same-process replay before the fresh-process replay. These additions are
diagnostic-only; they do not alter the data, model, loss, precision, update or frozen
parity tolerances. Its runner hash is
`b8014624e3dbf44c3e11d704a0d0694646a01ad3f270e6c729082d6edb4f130d`.

The owner closed the parity disposition on 2026-07-21. Boundary, input, RNG and
discrete state remain exact. Model parameters, BN mean, BN var, Adam `exp_avg` and
Adam `exp_avg_sq` are gated separately; for every group, fresh-process relative-L2
and max-absolute error must each be no greater than
`max(frozen tolerance, 1.25 * same-process repeat-control)`. Per-element allclose is
retained as a diagnostic and is no longer the hard gate for nondeterministic kernels.
Implementation `73158b70cb853a4bac99f6fd9f7c4f1598565bc3` adds explicit floating/discrete
state identities, the grouped comparator, focused tests and a read-only reassessment
entry point; it changes no model, data, loss, precision, optimizer update or scheduler
semantics.

Read-only reassessment of immutable Job `525192` result
`62e8249c...b6b` is PASS with reassessment SHA `d883c1ef...f7fa2`; no raw artifact
was modified and no GH200 time was charged. Exact boundary/input/RNG/training state,
BN counts and Adam steps pass. Group `(fresh relative-L2 / limit; fresh max-abs /
limit)` is: model parameters `0.00105484 / 0.002; 0.000769451 / 0.000792537`,
BN mean `0.00226839 / 0.00270428; 0.0792162 / 0.103417`, BN var
`0.000640081 / 0.00200864; 0.188965 / 0.492682`, Adam `exp_avg`
`0.288651 / 0.359091; 0.00564399 / 0.0105385`, and Adam `exp_avg_sq`
`5.46709e-5 / 0.002; 7.60471e-6 / 0.0002`. The original old-gate terminal label is
preserved as historical evidence; under the amended rule LiDAR sustained r1 is an
accepted WP1 reference. Jobs `527225` and `527229` then exhausted the frozen repeat
protocol and reached the unresolved-instability stop recorded above. The owner then
directed no additional LiDAR repeat, froze the LiDAR trace, and accepted HPC
hardware/power variation as the reason not to spend further profiler budget on that
branch. This is an operational disposition, not a proven causal attribution or a
stable LiDAR speed claim. WP2 is owner-paused pending discussion; no strict-output-
neutral candidate cell may run by inference. Camera Job `527239`
then reached the grouped continuation-parity stop described in the ledger; the
owner's instruction to continue Camera WP1 did not waive that gate, so neither r2
nor Camera trace was authorized by inference. The owner then explicitly authorizes
the single predeclared Camera trace r1 only, despite that stop, to localize
whole-model bottlenecks. Job `527247` consumed and completed that authority. Camera
sustained r2 remains frozen; the trace does not waive checkpoint parity, change
numeric tolerances, support a stable speed claim, or activate any WP2 candidate.
That one-off trace authority was exhausted before the subsequent WP2 decision below.

The owner then explicitly resumes WP2 and accepts the negative Camera reference
continuation result as a known reference limitation, not as a parity waiver. The
exact serial order is augmentation transfer/unused-return cleanup, fixed coordinate
grid, then batched affine/grid construction retaining per-image resize and
`grid_sample`; only afterwards may the already-frozen smaller whitelist/target-D2H
items be considered. Each candidate remains default-off outside its explicit
profile, receives a fresh output, and retains the exact data/model/loss/precision/
update/checkpoint gates. IP-WP3, physical-batch probes, SDPA/compile, fused AdamW,
checkpoint cadence, original Envelope B, merge and push remain unauthorized.

### 8.5 WP2 Camera augmentation-cleanup candidate r1 pre-submission

```text
SOURCE: containing pre-submission commit; runtime implementation parent
        37cfde2cab9349fff7e5884d162354a18a265ab0; no runtime-file change follows
CANDIDATE_ID: camera_aug_transfer_cleanup_b4_accum8
SINGLE_CHANGED_OPTION: camera_augmentation_transfer_cleanup=true; every other
                       candidate option remains at the reference value
PROFILE: fl_v3/configs/s10_phase1p_camera_aug_cleanup.json
PROFILE_FILE_SHA256: 9a9a48b9185cbeac59d6614c6bb7567a11d5f3ae4a6f1d55145afb6f6b147cb9
PROFILE_CANONICAL_SHA256: cdeed0799bb87d8916512d90befbd2a903329a8fea69766dc611eef82a6a9d6e
LOCAL_VALIDATION: git diff --check; Python py_compile for all changed Python/tests;
                  bash -n and shellcheck for the wrapper; baseline/candidate profile
                  canonical loading and exact Camera-only single-option mapping PASS
PRE_MODEL_PARITY: four focused GH200 tests in the wrapper must PASS before profiler
                  construction: unchanged reference profile, exact candidate mapping,
                  named-field CPU residency, and elementwise-exact preprocess outputs
OUTPUT: <approved root>/camera/sustained_<containing-source-SHA12>_r1_augcleanup
RESOURCES: 1 GH200, 16 CPU, 96 GiB, 00:45:00, no requeue, concurrency one
STOP: any focused-test/profile/source/input/nonfinite/discard/memory failure; grouped
      checkpoint continuation is reported honestly and remains non-waived
```

Exact command after sealing the containing source SHA `<SOURCE>` is:

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --job-name=s10-ip-e1-camera-augcleanup \
  --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-augcleanup-%j.out \
  --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-augcleanup-%j.err \
  fl_v3/scripts/run_s10_phase1p_ip_e1.sh \
  --branch camera --mode sustained \
  --config fl_v3/configs/s10_phase1_camera.json \
  --profile-config fl_v3/configs/s10_phase1p_camera_aug_cleanup.json \
  --output-dir /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/camera/sustained_<SOURCE12>_r1_augcleanup \
  --source-sha <SOURCE> \
  --approved-source-sha 85c6719e4b880b198d850e16b1418c230fa5c656 \
  --repeat 1 --attempt-id augcleanup
```

### 8.6 WP2 Camera augmentation-cleanup value-table refinement

Job `527276` is immutable and terminal negative. It removed the GPU round trip but
left `augmentation_params` in recursively pinned DataLoader memory, after which the
unchanged Python loop read 168 scalar tensor values per B4 microbatch. The derived
implementation converts the same contiguous CPU float64 `24x7` block to Python
float values once per microbatch. It does not change the loader tensor, sampled
values, image order, resize/crop/flip/affine/grid math, returned training tensors,
loss, update, precision or checkpoint semantics.

```text
DERIVES_FROM: Job 527276 / measured negative implementation mechanism
SOURCE: containing pre-submission commit; runtime implementation parent
        57c928eac212e1604b4eeca3ed6e2c94c2083f68; no runtime-file change follows
CANDIDATE_ID: camera_aug_transfer_cleanup_b4_accum8
SINGLE_CHANGED_OPTION: camera_augmentation_transfer_cleanup=true; every other
                       candidate option remains at the reference value
PROFILE: fl_v3/configs/s10_phase1p_camera_aug_cleanup.json
PROFILE_FILE_SHA256: 9a9a48b9185cbeac59d6614c6bb7567a11d5f3ae4a6f1d55145afb6f6b147cb9
PROFILE_CANONICAL_SHA256: cdeed0799bb87d8916512d90befbd2a903329a8fea69766dc611eef82a6a9d6e
PREPROCESS_SHA256: ef3907bb2582a8c12e8fdeba7ff0badafae860bccc2d5d7fba0551984e996a6e
LOCAL_VALIDATION: git diff --check and Python py_compile PASS
PRE_MODEL_PARITY: the same four focused GH200 tests must PASS; the elementwise-
                  exact preprocess test additionally exercises DataLoader-pinned
                  augmentation parameters when CUDA is available
OUTPUT: <approved root>/camera/sustained_<containing-source-SHA12>_r1_augvalues
RESOURCES: 1 GH200, 16 CPU, 96 GiB, 00:45:00, no requeue, concurrency one
STOP: any focused-test/profile/source/input/nonfinite/discard/memory failure; grouped
      checkpoint continuation is reported honestly and remains non-waived
```

Exact command after sealing the containing source SHA `<SOURCE>` is:

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --job-name=s10-ip-e1-camera-augvalues \
  --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-augvalues-%j.out \
  --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-augvalues-%j.err \
  fl_v3/scripts/run_s10_phase1p_ip_e1.sh \
  --branch camera --mode sustained \
  --config fl_v3/configs/s10_phase1_camera.json \
  --profile-config fl_v3/configs/s10_phase1p_camera_aug_cleanup.json \
  --output-dir /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/camera/sustained_<SOURCE12>_r1_augvalues \
  --source-sha <SOURCE> \
  --approved-source-sha 85c6719e4b880b198d850e16b1418c230fa5c656 \
  --repeat 1 --attempt-id augvalues
```

### 8.7 WP2 Camera fixed rotation-coordinate grid candidate

This candidate is independent of the rejected augmentation-cleanup group. It
caches only the fixed `256x704x[x,y,1]` float64 rotation output-coordinate basis
as a non-persistent buffer. Per-image resize/crop/flip, affine construction and
inversion, sampling-grid arithmetic and `grid_sample` remain on the reference path.

```text
SOURCE: containing pre-submission commit; runtime implementation parent
        42f1168b835bab3ad64eff455df6916450cae8d2; no runtime-file change follows
CANDIDATE_ID: camera_static_grid_cache_b4_accum8
SINGLE_CHANGED_OPTION: camera_static_grid_cache=true; augmentation cleanup and every
                       other candidate option remain at the reference value
PROFILE: fl_v3/configs/s10_phase1p_camera_static_grid.json
PROFILE_FILE_SHA256: 5d0c00cd3e1ec9410b96d303bfbfce13138ef888fb50eb96734dd138f3b72e77
PROFILE_CANONICAL_SHA256: cc5b4a596dbed6e5dd3390b96909f5a6def4e584e0baa36d1c039caab3727beb
PREPROCESS_SHA256: 0ac0c55690c85f756239aa98d47b0951e5da3107227ad20081edb3ba3f8e5cf3
LOCAL_VALIDATION: git diff --check; Python py_compile; bash -n; shellcheck; exact
                  Camera binding/single-option mapping and LiDAR rejection PASS
PRE_MODEL_PARITY: three focused GH200 tests must PASS before profiler construction:
                  unchanged reference profile, exact static-grid candidate mapping,
                  and two-call elementwise-exact/non-persistent cache behavior
OUTPUT: <approved root>/camera/sustained_<containing-source-SHA12>_r1_staticgrid
RESOURCES: 1 GH200, 16 CPU, 96 GiB, 00:45:00, no requeue, concurrency one
STOP: any focused-test/profile/source/input/nonfinite/discard/memory failure; grouped
      checkpoint continuation is reported honestly and remains non-waived
```

Exact command after sealing the containing source SHA `<SOURCE>` is:

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --job-name=s10-ip-e1-camera-staticgrid \
  --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-staticgrid-%j.out \
  --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-staticgrid-%j.err \
  fl_v3/scripts/run_s10_phase1p_ip_e1.sh \
  --branch camera --mode sustained \
  --config fl_v3/configs/s10_phase1_camera.json \
  --profile-config fl_v3/configs/s10_phase1p_camera_static_grid.json \
  --output-dir /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/camera/sustained_<SOURCE12>_r1_staticgrid \
  --source-sha <SOURCE> \
  --approved-source-sha 85c6719e4b880b198d850e16b1418c230fa5c656 \
  --repeat 1 --attempt-id staticgrid
```

### 8.8 WP2 Camera conservative batched affine/grid candidate

This is the third owner-approved ordered item. To preserve the strict boundary,
per-image resize/crop/flip, affine construction and inverse, and each individual
`grid_sample` call remain unchanged. The candidate creates one fixed output basis
per microbatch and batches only the `out @ inverse.T` source-coordinate multiply.
It is independent of both rejected candidates; their flags remain false.

```text
SOURCE: containing pre-submission commit; runtime implementation parent
        834677c243659677b0a19cb3478b3057ba7c28a1; no runtime-file change follows
CANDIDATE_ID: camera_batched_affine_grid_b4_accum8
SINGLE_CHANGED_OPTION: camera_batched_affine_grid=true; augmentation cleanup,
                       static cache and every other option remain reference values
PROFILE: fl_v3/configs/s10_phase1p_camera_batched_affine_grid.json
PROFILE_FILE_SHA256: fa5297457b4453e420f950fd9b1c860bcfb70089de3600428d7b37e7a24fb4f1
PROFILE_CANONICAL_SHA256: 6a809032c9340789a08592fa8c6d887480ba9833b0da5dab8ba06dcc63d7a19b
PREPROCESS_SHA256: f48689a7aabb4ae4350a95445e197a19a1fc72c93ccd70cf153f51450ac3732f
LOCAL_VALIDATION: git diff --check; Python py_compile; bash -n; shellcheck; exact
                  Camera binding/single-option mapping and LiDAR rejection PASS
PRE_MODEL_PARITY: three focused GH200 tests must PASS before profiler construction:
                  unchanged reference profile, exact candidate mapping, and
                  elementwise-exact Camera preprocess on both CPU and CUDA
OUTPUT: <approved root>/camera/sustained_<containing-source-SHA12>_r1_batchedgrid
RESOURCES: 1 GH200, 16 CPU, 96 GiB, 00:45:00, no requeue, concurrency one
STOP: any CPU/CUDA elementwise mismatch or focused-test/profile/source/input/
      nonfinite/discard/memory failure; no tolerance relaxation or sustained timing
      after a pre-model mismatch; grouped continuation remains non-waived
```

Exact command after sealing the containing source SHA `<SOURCE>` is:

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --job-name=s10-ip-e1-camera-batchedgrid \
  --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-batchedgrid-%j.out \
  --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/slurm/camera-batchedgrid-%j.err \
  fl_v3/scripts/run_s10_phase1p_ip_e1.sh \
  --branch camera --mode sustained \
  --config fl_v3/configs/s10_phase1_camera.json \
  --profile-config fl_v3/configs/s10_phase1p_camera_batched_affine_grid.json \
  --output-dir /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e1_85c6719e4b88/camera/sustained_<SOURCE12>_r1_batchedgrid \
  --source-sha <SOURCE> \
  --approved-source-sha 85c6719e4b880b198d850e16b1418c230fa5c656 \
  --repeat 1 --attempt-id batchedgrid
```

## 9. Phase I-P IP-G1 closure and terminal IP-E2 record

```text
PHASE: S10 Phase I-P / IP-E2 capacity and numerical-runtime screening
REQUEST_STATE: IP-E2 TERMINAL / CELL 7 POSITIVE / B16 SKIPPED / IP-G2 READY
ACTIVATION_BASELINE: 3f55e635aef4f893d9fd66e7921f55ce4f7b36e8
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA; do not move
OBJECTIVE: measure Camera physical-batch scaling, Swin SDPA, scoped torch.compile
  and fused AdamW with real D_fit training mechanics; synthesize one measurement-
  only stack and projected 20-epoch payback without capability or recipe claims
BATCH_CELLS: B4x8 reference; B8x4 measurement-only; B12 deleted; B16x2 conditional
B16_PRECONDITION: B8+SDPA+compile passes capacity, sustained-health and exact checkpoint
  gates with no monotonic memory growth, and peak reserved bytes satisfy
  R8 + 2*max(R8-R4,0) <= 0.70*visible_bytes; a fresh B16 capacity process must then
  pass the unchanged <=0.85*visible hard gate before sustained measurement
NUMERICAL_CELLS: Camera Swin SDPA; scoped Camera torch.compile; fused AdamW;
  individually default-off and isolated before any combination
MATCHED_DESIGN: each comparison runs two fresh processes serially inside one Slurm
  allocation/on one GH200; confirmation reverses reference/candidate order
EXACT_SERIAL_CELLS:
  1. B4 eager reference versus B4 Camera Swin SDPA
  2. B4 eager reference versus B4 scoped torch.compile
  3. B4 eager reference versus B4 SDPA+compile
  4. B4 SDPA+compile versus B8x4 SDPA+compile
  5. B8 SDPA+compile versus B8 SDPA+compile+fused AdamW
  6. conditional B8 best-stack versus B16x2 best-stack
  7. final best-stack confirmation with reversed process order
COMPILE_SCOPE: backend=inductor, dynamic=false, mode=default; compile only forward
  callables for camera_backbone, camera_neck, decoder_backbone, decoder_neck and
  head without replacing registered modules or changing parameter/state-dict names;
  preprocess, view transform/pool, target/loss and optimizer stay eager
COMPILE_EVIDENCE: fresh candidate-local cold cache; report cold startup, graph breaks,
  compiled graphs and recompiles; unexpected steady-state recompile or checkpoint
  opacity rejects the candidate
BATCH_RNG: B8/B16 need not reproduce B4 worker assignment or per-sample augmentation
  draws; each candidate's own boundary/input/RNG/training/discrete state and
  fresh-process checkpoint structure remain exact; numerical trajectory distances
  remain diagnostic
DATA_AND_CLAIMS: D_fit only; D_select, D_audit, official validation, capability
  metrics, generalization and candidate-selection claims forbidden
PRECISION_UPDATE: frozen FP16 policy, loss, targets, GradScaler, clip, accepted
  optimizer update and scheduler semantics; physical batch preserves effective B32
MEASUREMENT: 16 accepted warm-up plus 256 accepted measured optimizer windows per
  process; one-second system sampling; compile cold-start/cache/graph/recompile cost;
  real Adam-state memory and eight-window checkpoint continuation
MEMORY_HARD_GATE: fresh process; peak reserved <=85% visible; no monotonic growth;
  OOM recorded as CAPACITY_OOM; B8 OOM skips B16
BASE_AGGREGATE_GPU_HOURS: 4.0 charged GH200-hours
CODE_BUG_REMEDIATION_RESERVE: +1.0 charged GH200-hour, code-level bug only
HARD_AGGREGATE_GPU_HOURS: 5.0 charged GH200-hours
MAX_CONCURRENCY: 1
PER_JOB_RESOURCE: 1 node / 1 GH200 / 16 CPUs / 96 GiB / <=01:00:00 / no requeue
SUBMISSION_POLICY: serial; no numeric submission cap; no blind identical retry
OUTPUT_ROOT_RULE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_e2_<activation_sha12>/
FRESH_OUTPUT: every subprocess/attempt path absent before execution; no overwrite;
  raw evidence immutable
ENGINEERING_REMEDIATION: O-149 frozen-semantics repairs only within the layered
  ceiling; compiler limitation, ambiguous numerical drift or candidate-scope change
  is not silently classified as a code bug
OWNER_ESCALATION: science-boundary pressure; ambiguous or repeated blocker;
  nonfinite/discard; exact checkpoint/context/structure failure;
  >85% memory, monotonic growth, ceiling exhaustion, or requested candidate change
OWNER_APPROVAL: 2026-07-21 — accepted the quantitative B16 gate, exact cell order,
  compile scope and Camera-only boundary; closed IP-G1 and activated IP-E2
OWNER_GATE_AMENDMENT: 2026-07-21 — grouped model/BN/Adam continuation relative-L2,
  max-absolute and allclose results are diagnostics rather than hard promotion
  gates. Exact checkpoint boundary, input, RNG, training/discrete state,
  names/shapes/dtypes, comparison integrity and finite numerical state remain hard.
  Cell-1 SDPA is promoted inside IP-E2 and the owner orders serial Cells 2-7.
  No Cell-1 rerun, Envelope-B activation or capability claim is authorized.
GATE_AMENDMENT_SHA: 89181c117d69aaa7094def38f6931623f385a691
PRE_SUBMISSION_GATE: record durable implementation SHA, profile hashes, exact paired
  command, fresh output paths and remaining budget before every first submission
IMPLEMENTATION_SHA: e6af054bfb16710355e22f6cea931368750aba89
IMPLEMENTATION_TREE: 546036d96d0d8b8f853e6c8c3fc22021b12ff25d
IMPLEMENTATION_SCOPE: Camera-only profile schema/runtime views; current Swin SDPA;
  forward-only scoped torch.compile; real fused/unfused AdamW; B4/B8/B16 effective-
  B32 binding; B8/B16 capacity/OOM terminal protocol; 16-window throughput blocks;
  monotonic-memory gate; same-allocation pair analysis; checkpoint continuation
CAPACITY_REFINEMENT: the B8+SDPA+compile stack receives a fresh 1+8-window capacity
  probe before sustained Cell 4; this is the already frozen "B8 no OOM/capacity"
  prerequisite, not a new comparison cell or candidate. Conditional B16 uses the
  same protocol only after the quantitative R4/R8 gate passes.
PROFILE_HASHES: physical SHA-256 / canonical SHA-256
  reference B4: 2f1763f13a39f6fd7a0e80c86b79381d5753f7bbfb18756d9d5ce9b75857a9ca /
    9ac0280b2eef44b36e0e24dae25c54dc1c60aa4ecc537c638d04c725f2d07fcd
  SDPA B4: 6163f1be452def8ae32cbfd2a89b99d3eea31ba90816f1b8c21e0b202d2a1273 /
    434dd9a19ba384768330ca0a4d19b20d4b746b4ad8675656e9852abaedb70a1a
  compile B4: 4509ab45da95df8fef737ccc170750e55686a77b52a5db8e52af332e17829259 /
    9e2628eebebe62fc5cdfaaad8c5c486febd284aa1294b978fbf756150877b01a
  SDPA+compile B4: 66317509d5a13690993adfbbf697183ab70e7efec6a6fb19172d16ab0b441ec5 /
    9459bd1fe8853a1030b99928feb4ddccfa2efbd53a1b06413e05a17d96e22ee0
  SDPA+compile B8: 44ab014e499005237dbfc57ea183df97ca2ef19a71c813bcc9379b18bbcd9dcb /
    c82cb49e35ef3cbeeafd3d4d6bf58966e4fb748cb257ddc32b945b06bfe0cfa3
  SDPA+compile+fused B8: 5faf9f15d70a8e790cfe242c5714db3514aee575d896aab260cf03b10522d00a /
    1ae8ec037df478e99bf16f9f2a6ca174e4ef564a350bc644539feb9bef7780bf
  SDPA+compile B16: c15b61de578dd8659f5c805773cdde10a89a2be1e790b1650fc049e66e7133d4 /
    a85f3bd5141e2fd34036f5ac77c11a83d89ab8135e918ed3b2149649a86f29fe
  SDPA+compile+fused B16: c0d919ba3eaa084f43072dab1b407eb2fa56cbf9fc926cd7386c21b49b86cf51 /
    c3ac8a0c35cc854072e8ba1069c70adc53d60ad650e6fad5c863f27494fbdbea
SCRIPT_HASHES: launcher adf3bc945ac08885eeabbf21347f30ffe5b614d15078629c14bb2009cfc6fbe6;
  profiler b4ce5df419924d63eac30070944dcba3a0bc73ccd75a5a455183ac8c4e414434;
  pair analyzer cec454e46947570244c8afdf901a406314e23071461440352e831fb1eabefc6e;
  continuation gate 6cec4646b5c1caf6a5c7145990c27220df44cf3079f0585b50f64f6e24d00d7c;
  reassessor 35aedf92df2a9c801ed54485a4b29b992fc13606185b09d298554b61e69f8b19
LOCAL_VALIDATION: Python py_compile PASS for all changed Python/tests; bash -n and
  shellcheck PASS; JSON syntax, canonical hash, patch whitespace and synthetic
  50,000-draw paired-bootstrap/B16-gate checks PASS. The x86 login Python lacks
  torch/pytest; the exact current-code FP32/FP16 Swin forward/backward and fused-
  AdamW accepted-update tests are fail-closed pretests in every IP-E2 allocation.
FIRST_CELL: Cell 1, B4 eager reference then B4 Camera Swin SDPA, each in a fresh
  Python process within one allocation; pair analyzer runs before allocation exit
FIRST_COMMAND: from a clean containing source commit, resolve SOURCE_SHA=$(git
  rev-parse HEAD), ROOT=<OUTPUT_ROOT_RULE>, then submit one sbatch with the frozen
  resource tuple and fl_v3/scripts/run_s10_phase1p_ip_e2.sh using:
  --config fl_v3/configs/s10_phase1_camera.json
  --first-profile fl_v3/configs/s10_phase1p_ip_e2_camera_reference_b4.json
  --first-mode sustained
  --first-output ${ROOT}/camera/sustained_${SOURCE_SHA:0:12}_r1_c1_ref
  --first-attempt c1_ref
  --second-profile fl_v3/configs/s10_phase1p_ip_e2_camera_sdpa_b4.json
  --second-mode sustained
  --second-output ${ROOT}/camera/sustained_${SOURCE_SHA:0:12}_r1_c1_sdpa
  --second-attempt c1_sdpa
  --pair-output ${ROOT}/pairs/cell01_sdpa_${SOURCE_SHA:0:12}_r1.json
  --pair-reference first --source-sha ${SOURCE_SHA}
  --approved-source-sha 3f55e635aef4f893d9fd66e7921f55ce4f7b36e8 --repeat 1
PRE_SUBMISSION_RECORD: before sbatch, seal the containing literal source SHA, full
  sbatch argv, resolved output paths and budget into
  <OUTPUT_ROOT_RULE>/pre_submission_cell01.json; runner independently rechecks the
  same source/branch/base/clean-output/resource identities on GH200
BUDGET_AFTER_CELL_7: base consumed 2.871389 / remaining 1.128611;
  code-bug reserve consumed 0.000000 / remaining 1.000000; hard remaining
  2.128611 GH200-hours
CELL_1_TERMINAL: Job 531766 COMPLETED 0:0 in 1541 s on n203. All eight current-
  code pretests passed. Eager and SDPA each completed 16+256 accepted windows with
  zero invalid/discard/scaler-skip, exact same-batch input anchor/CBGS prefix, stable
  memory and a real checkpoint/fresh-process continuation. SDPA measured 16.183531
  versus eager 15.143028 presentations/s: ratio 1.068712 and one-sided 95% lower
  bound 1.065611. Peak allocated/reserved fell from 16,038,963,200 /
  18,723,373,056 to 14,885,203,968 / 17,458,790,400 bytes. Projected 20-epoch
  cost fell from 32.264493 to 30.190691 GH200-hours, a 2.073802-hour screen.
  Both exact boundary/input/RNG/discrete gates passed. Eager repeated the already
  owner-accepted continuation negative in Adam exp_avg_sq max-abs only. SDPA's
  fresh-process continuation exceeded the calibrated envelope for Adam exp_avg,
  Adam exp_avg_sq and BN running mean; model parameters and BN running var passed.
  This is numerical-runtime nondeterminism, not an unambiguous code bug. Under the
  owner-amended enforcement, exact/integrity/finite hard gates PASS for both;
  numerical envelope results remain diagnostic and SDPA is promoted inside IP-E2.
CELL_1_ARTIFACTS: pre-submission f64f66aa...78949a; eager measurement/result
  60719b89...004e1 / e4d87f8d...53be9; SDPA measurement/result
  98f63f82...3b69 / 0201eaee...9185; pair summary 73463de6...55c5;
  stdout/stderr e0ecbea5...28df / 8db5d05b...e830
CELL_1_REASSESSMENT: immutable raw results unchanged. Derived v2 reassessments are
  eager `08426908dcd1e67d05c4ae18402a4cfa73c4daddaa09eac3609e873699e36512`
  and SDPA `50cc83be5f1c1e24814e45b4efa65568793b2fc5ced58dcd3a2ae2a4caf9d88c`;
  both hard PASS, with the original group-distance negatives retained.
CELL_2_TERMINAL: Job 532763 COMPLETED 0:0 in 1729 s on n204. Eager and compile
  each completed 16+256 accepted B4x8 windows with zero invalid/discard/scaler
  skip. Compile measured 16.287596 versus 15.425816 presentations/s, ratio/lower
  bound 1.055866/1.050861, and projects 1.645086 GH200-hours saved per 20 epochs.
  Peak allocated/reserved fell from 16,038,963,200/18,723,373,056 to
  15,259,009,024/17,748,197,376 bytes. Five scoped graphs compiled during warm-up;
  the measured interval had no compiler-counter delta or unexpected recompile.
  Measurement health, exact checkpoint hard gates and all numerical diagnostics
  PASS for both processes.
CELL_2_ARTIFACTS: pre-submission `1ada62a...1a788`; eager measurement/result
  `ca4ccf9...376f / 86846c8...7ec2e`; compile measurement/result
  `5f58f73...e70ac / a4697f9...04f9`; pair `01085fb...4661`;
  stdout/stderr `f06617a...fea44 / 8db5d05...1e830`.
CELL_3_TERMINAL: Job 533212 COMPLETED 0:0 in 1772 s on n69. SDPA+compile
  measured 16.527709 versus the same-node eager reference 13.961779 presentations/s,
  ratio/lower bound 1.183783/1.178471 and projected saving 5.402981 GH200-hours.
  Both completed 16+256 accepted windows with zero invalid/discard/scaler skip.
  Candidate peak allocated/reserved was 14,368,397,312/19,331,547,136 bytes with
  no monotonic growth. Twelve Swin modules plus five compile scopes were active;
  no measured compiler-counter delta or unexpected recompile occurred. Measurement
  health and exact checkpoint hard gates PASS; numerical trajectory distance is a
  retained diagnostic negative.
CELL_3_ARTIFACTS: pre-submission `0a3aacf...c0c29`; eager measurement/result
  `3a66ace...7d729 / ab65b31...7f814`; combination measurement/result
  `f0bb189...b41c1 / b381382...634d`; pair `0afc8ca...78a1e`;
  stdout/stderr `e739336...5f0ca / 8db5d05...1e830`.
B8_CAPACITY_TERMINAL: Job 533364 COMPLETED 0:0 in 262 s on n127. The fresh
  SDPA+compile B8x4 process completed 8/8 accepted windows after one warm-up with
  zero invalid/discard/scaler skip. Peak allocated/reserved was
  27,787,762,176/32,877,051,904 bytes (`32.2307%` visible), with no monotonic
  growth or unexpected recompile. The frozen <=85% capacity gate PASS.
B8_CAPACITY_ARTIFACTS: pre-submission `9df6167...6f816`; measurement/result
  `0c4781d...2453 / 734a4d7...8f8b`; stdout/stderr
  `b35ca7a...23990 / 8db5d05...1e830`.
CELL_4_TERMINAL: Job 533384 COMPLETED 0:0 in 1772 s on n463. B8x4
  SDPA+compile measured 21.554602 versus B4x8 stack reference 16.723609
  presentations/s, ratio/lower bound 1.288873/1.273372 and projected saving
  6.544602 GH200-hours. Both completed 16+256 accepted windows with zero invalid/
  discard/scaler skip, no memory growth/recompile and passing exact checkpoint
  hard gates. B8 peak allocated/reserved was 27,833,085,952/37,981,519,872 bytes
  (`37.2348%` visible); numerical trajectory distance remains diagnostic.
CELL_4_ARTIFACTS: pre-submission `941f90c...f43fa`; B4 measurement/result
  `773dd67...c578 / ffed4db...268e`; B8 measurement/result
  `8740d6d...02c75 / 796e881...8c6b0`; pair `ffa3762...6c48d`;
  stdout/stderr `14cf94e...37a5f / 8db5d05...1e830`.
B16_CONDITIONAL_STOP: `R4=19,331,547,136`, `R8=37,981,519,872`, projected
  B16 reserved `75,281,465,344` bytes = `73.8014%` visible, which exceeds the
  frozen `70%` prerequisite. `eligible_for_fresh_capacity_probe=false`; no B16
  submission is authorized or needed.
CELL_5_TERMINAL: Job 533512 COMPLETED 0:0 in 1576 s on n145. Fused AdamW
  measured 23.284372 versus the same-node unfused B8 SDPA+compile reference at
  22.348477 presentations/s, ratio/lower bound 1.041877/1.038384 and projected
  saving 0.883318 GH200-hours per 20 epochs. Both completed 16+256 accepted
  windows with zero invalid/discard/scaler skip, identical peak reserved memory
  37,981,519,872 bytes, no monotonic growth/recompile and passing exact checkpoint
  hard gates. Fused Adam exp_avg relative-L2 is a retained diagnostic negative;
  all groups are finite and structurally intact.
CELL_5_ARTIFACTS: pre-submission `f4d31e7...800d75`; unfused measurement/result
  `1c7326c...bdc3 / 011fbe9...8af`; fused measurement/result
  `e6b3ab3...e5961 / f7dcd9e...716e`; pair `5c5a728...23a85`;
  stdout/stderr `ae42dcf...24e56 / 8db5d05...1e830`.
CELL_6_STOP: B16 remains ineligible under the frozen 70% projection gate; it is
  skipped without a capacity or sustained submission.
CELL_7_TERMINAL: Job 534737 COMPLETED 0:0 in 1685 s on n411. The B8
  SDPA+compile+fused best stack ran first at 21.803544 presentations/s; original
  B4 eager ran second at 15.152073. The reversed-order ratio/lower bound is
  1.438981/1.413203. Both completed 16+256 accepted windows with zero invalid/
  discard/scaler skip, no memory growth/recompile, and passing exact checkpoint
  hard gates. Direct 20-epoch projections are 22.443901 versus 32.245064
  GH200-hours, saving 9.801163. Best-stack peak allocated/reserved is
  27,833,308,672/37,981,519,872 bytes. Final projected B16 reserved is
  76,497,813,504 bytes = 74.9938% visible, still failing the frozen 70% gate.
CELL_7_ARTIFACTS: pre-submission `6e13a8c...be777`; best measurement/result
  `3a49918...f07e9 / 8d84197...bc9da7`; B4 reference measurement/result
  `7a94b02...e62db / 503cec8...246ce`; pair `7e67036...25034`;
  stdout/stderr `2d19aaa...fa076 / 8db5d05...1e830`.
PAYBACK: IP-E2 consumed 2.871389 GH200-hours, giving a 0.292964-run break-even
  against the direct Cell-7 saving. IP-E1+IP-E2 consumed 4.635278 GH200-hours,
  giving a 0.472931-run whole-preflight break-even. The documented 29.87-hour
  Camera projection scaled by the Cell-7 point/lower-bound ratios becomes
  20.757745/21.136384 hours, a conservative 8.733616-9.112255-hour saving range.
EXECUTABLE_NOW: no; the exact IP-E2 serial sequence is terminal and unused budget
  expires. IP-G2 owner disposition is required before any candidate promotion,
  recipe freeze, new profiler cell or Envelope-B revision/activation.
```

### 9.1 IP-E2 compact execution ledger

| Cell / job | Source, pair and immutable outputs | Terminal evidence | Charged GH200-hours |
|---|---|---|---:|
| Cell 1 / Job `531766`, B4 eager -> B4 SDPA | source `b8ac61a5bc464bc1a6bf1c1e4f97b17f0b96fd54`; one `n203` allocation; eager `sustained_b8ac61a5bc46_r1_c1_ref`; SDPA `...c1_sdpa`; pair `73463de6...55c5`; v2 reassessments `08426908...36512 / 50cc83be...9d88c` | `COMPLETED 0:0`, 8/8 pretests; measurement health and amended exact/integrity/finite hard gates PASS both. SDPA ratio/lower bound `1.068712 / 1.065611`, reserved memory `-1.2646 GB`, projected `-2.0738 GH200h`; Adam/BN trajectory-distance negatives remain diagnostics. Owner promotes SDPA inside IP-E2; no rerun | base `0.428056`; reserve `0.000000` |
| Cell 2 / Job `532763`, B4 eager -> B4 compile | source `83232a770790c545a67aca3450b26ed739051515`; one `n204` allocation; eager `sustained_83232a770790_r1_c2_ref`; compile `...c2_compile`; pair `01085fbb...4661` | `COMPLETED 0:0`, 12/12 pretests; both health/checkpoint hard gates PASS. Compile ratio/lower bound `1.055866 / 1.050861`, reserved memory `-0.9752 GB`, projected `-1.6451 GH200h`; five graphs, no measured recompile; all numerical diagnostics PASS | base `0.480278`; reserve `0.000000` |
| Cell 3 / Job `533212`, B4 eager -> B4 SDPA+compile | source `a1f21878a26759df48ffe0deed24656bd6d7a316`; one `n69` allocation; eager `sustained_a1f21878a267_r1_c3_ref`; combination `...c3_sdpa_compile`; pair `0afc8ca5...78a1e` | `COMPLETED 0:0`, 12/12 pretests; health/checkpoint hard gates PASS. Ratio/lower bound `1.183783 / 1.178471`; projected `-5.4030 GH200h`; 12 SDPA modules plus five graphs, no measured recompile; numerical distance remains diagnostic negative | base `0.492222`; reserve `0.000000` |
| Pre-Cell-4 B8 capacity / Job `533364` | source `3d17ce097f0b26af2fde803f9e87d677dbc5fded`; `capacity_3d17ce097f0b_r1_c4_b8_probe`; result `734a4d70...8f8b` | `COMPLETED 0:0`; B8x4 SDPA+compile 8/8 accepted; peak allocated/reserved `27.788/32.877 GB`, `32.2307%` visible; no growth/recompile; capacity PASS | base `0.072778`; reserve `0.000000` |
| Cell 4 / Job `533384`, B4 -> B8 SDPA+compile | source `e9d4b8f378c884338b5972d244f17922a6b18826`; one `n463` allocation; B4 `sustained_e9d4b8f378c8_r1_c4_b4`; B8 `...c4_b8`; pair `ffa37629...6c48d` | `COMPLETED 0:0`; health/checkpoint hard gates PASS. B8 ratio/lower bound `1.288873 / 1.273372`, projected `-6.5446 GH200h`; R8 `37.982 GB`; B16 projection `73.8014%` fails frozen 70% prerequisite, so B16 skipped | base `0.492222`; reserve `0.000000` |
| Cell 5 / Job `533512`, B8 unfused -> fused AdamW | source `66760f45cdc3c41964ab73af48e97dbe60dd3e8d`; one `n145` allocation; unfused `sustained_66760f45cdc3_r1_c5_b8`; fused `...c5_b8_fused`; pair `5c5a728c...23a85` | `COMPLETED 0:0`; health/checkpoint hard gates PASS. Fused ratio/lower bound `1.041877 / 1.038384`, projected `-0.8833 GH200h`; identical `37.982 GB` peak reserved, no growth/recompile; Adam exp_avg distance remains diagnostic | base `0.437778`; reserve `0.000000` |
| Cell 7 / Job `534737`, best B8 first -> eager B4 second | source `cde351f99b039968133db0c273e0e0715a60b35e`; one `n411` allocation; best `sustained_cde351f99b03_r2_c7_best_first`; reference `...c7_ref_second`; pair `7e670362...25034` | `COMPLETED 0:0`; both hard gates PASS. Reversed total-stack ratio/lower bound `1.438981 / 1.413203`, projected `-9.8012 GH200h`; best reserved `37.982 GB`; final B16 projection `74.9938%`, still skipped | base `0.468056`; reserve `0.000000` |

### 9.2 IP-G2 B16 extension — active

This is an independently bounded extension. It does not reinterpret or reuse the
exhausted Section-9 IP-E2 budget. On 2026-07-21, the owner explicitly approved the
exact layered ceiling and concurrency below. Only the frozen conditional sequence
is executable; this does not activate Envelope B or authorize any capability run.

```text
PHASE: S10 Phase I-P / IP-G2 B16 capacity and matched-throughput extension
REQUEST_STATE: OWNER APPROVED / ACTIVE / CELL A CAPACITY NEXT
ACTIVATION_BASELINE: df3c17e3e6be19dcc586fdec2c6bd198c1b02d95
APPROVED_REQUEST_SHA: 1b25f1c98dabc19617fd4e2223c29b4fe076eeef
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA; do not move
OBJECTIVE: replace the withdrawn B16 projection veto with actual OOM-tolerant
  capacity evidence and, only after capacity PASS, quantify B16x2 versus the
  accepted B8x4 runtime stack on one GH200/allocation in both process orders
SCIENCE_BOUNDARY: Camera only; D_fit only; seed 0; effective B32; frozen model/head,
  precision, loss/targets/evaluator, scheduler/exposure/order and checkpoint cadence
CANDIDATES: exactly B8x4 and B16x2, both SDPA+scoped-compile+fused-AdamW; no B4,
  B12, other batch, new compile scope, optimizer or numerical-runtime candidate
RECIPE_STATUS: B8x4 BN/RNG change is owner-accepted; B16x2 remains measurement-only
  and cannot become the production recipe without a later explicit IP-G2 decision
PRODUCTION_CONFIG: unchanged during this extension; materialize the final B8 or B16
  recipe and accepted runtime defaults once, after the B16 owner decision
CLAIM_LIMIT: no capability, mAP/NDS, generalization or candidate-selection claim;
  D_select, D_audit and official validation forbidden
WITHDRAWN_GATE: projected B16 reserved <=70% visible is retained as a labelled
  diagnostic only and is not an eligibility or promotion gate
ACTUAL_MEMORY_GATE: a fresh B16 process must complete 1 warm-up plus 8 accepted
  capacity windows with peak reserved <=85% visible, no monotonic memory growth,
  nonfinite/discard/scaler skip or unexpected steady-state recompile
OOM_PROTOCOL: OOM or actual reserved >85% is terminal B16 capacity evidence; record
  it without remediation or a smaller-batch search, skip sustained B16 and retain B8
SUSTAINED_PROTOCOL: fresh processes; 16 accepted warm-up plus 256 accepted measured
  optimizer windows each; real loader/AdamW/scheduler/scaler/checkpoint mechanics;
  one-second system sampling; cold compile and measured graph/recompile accounting
PAIR_INTEGRITY: each B8/B16 pair runs serially in one Slurm allocation/on one GH200;
  exact source/base-config/data-role/seed/measurement context; order recorded; the
  physical-batch profiles intentionally differ
BATCH_RNG: B8 and B16 need not share worker assignment, per-sample augmentation draws
  or one cross-batch input anchor; boundary/input/RNG/training/discrete state remains
  exact within each candidate process and its fresh-process checkpoint continuation
HARD_GATES: accepted-window counts; zero invalid/discard/scaler skip/nonfinite;
  exact boundary/input/RNG/training/discrete state and names/shapes/dtypes; fresh-
  process checkpoint/context/structure integrity; <=85% actual peak reserved;
  no monotonic memory growth or unexpected measured recompile
NUMERICAL_DIAGNOSTICS: grouped model/BN/Adam relative-L2 and max-absolute errors plus
  elementwise allclose remain visible diagnostics, not hard gates, under the accepted
  non-deterministic-kernel policy
POSITIVE_PAIR: candidate/reference presentations-per-second ratio >1.0 and one-sided
  block-bootstrap 95% lower bound >1.0, with all hard gates PASS
DECISION_RULE: capacity failure ends B16; a non-positive first B8->B16 pair ends B16;
  a positive first pair triggers one reversed B16->B8 pair. Only two positive pairs
  support returning B16 to the owner for a recipe decision; otherwise B8 remains the
  accepted fallback. No result silently promotes B16.
EXACT_SERIAL_CELLS:
  A. fresh B16 capacity probe
  B. conditional same-allocation B8 first -> B16 second sustained pair
  C. conditional reverse B16 first -> B8 second sustained pair
BASE_AGGREGATE_GPU_HOURS: 1.20 charged GH200-hours
CODE_BUG_REMEDIATION_RESERVE: +0.50 charged GH200-hour, code-level bug only
HARD_AGGREGATE_GPU_HOURS: 1.70 charged GH200-hours
RESOURCE_RATIONALE: about 0.08 hour for capacity plus about 0.50 hour for each of
  at most two sustained pairs; 1.20 base leaves bounded scheduling/runtime margin,
  while 0.50 reserve can cover one diagnosed engineering replacement
MAX_CONCURRENCY: 1
PER_JOB_RESOURCE: 1 node / 1 GH200 / 16 CPUs / 96 GiB / <=01:00:00 / no requeue
SUBMISSION_POLICY: serial; no numeric submission cap; no blind identical retry
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_e2_df3c17e3e6be/
FRESH_OUTPUT: every process/pair/attempt path absent before execution; no overwrite;
  raw evidence immutable
ENGINEERING_REMEDIATION: O-149 frozen-semantics source/test/runner/API/checkpoint/
  logging repairs only, serially, within the layered ceiling; compiler limitations,
  OOM, capacity negatives, numerical drift or candidate changes are not code bugs
STOP_ESCALATE: ceiling exhaustion; repeated or ambiguous blocker; science-boundary
  pressure; candidate/resource/gate change; hard checkpoint/context failure;
  nonfinite/discard/scaler skip; memory growth; actual reserved >85%; OOM
IMPLEMENTATION_SHA: df3c17e3e6be19dcc586fdec2c6bd198c1b02d95
LAUNCHER_SHA256: 676dccdfc539f22c8a830e9df8eae84bc6ac8391b94361d0809fcf10a8a20bbb
PAIR_ANALYZER_SHA256: 614d7f03ea6388c7b5dd411fd4cb5f5a5e316af99587538ea9b23cadfba57f58
PAIR_TEST_SHA256: fa2aa63e6f32f25f6f7a4d11656697c0dd7e9ff8c33bee175de111245290c95e
CAMERA_CONFIG_FILE_SHA256: 567cb1b71535b4866193273960e531ae4b45318e56e81101e99ad186ac23ce60
CAMERA_RESOLVED_CONFIG_SHA256: e95e65a63a32c494296b38baf98fd913ff1ec6a168b78aabac48a8dc8f0ffe1d
B8_PROFILE_FILE_SHA256: 5faf9f15d70a8e790cfe242c5714db3514aee575d896aab260cf03b10522d00a
B8_PROFILE_CANONICAL_SHA256: 1ae8ec037df478e99bf16f9f2a6ca174e4ef564a350bc644539feb9bef7780bf
B16_PROFILE_FILE_SHA256: c0d919ba3eaa084f43072dab1b407eb2fa56cbf9fc926cd7386c21b49b86cf51
B16_PROFILE_CANONICAL_SHA256: c3ac8a0c35cc854072e8ba1069c70adc53d60ad650e6fad5c863f27494fbdbea
SOURCE_RULE: execution SOURCE_SHA is a clean linear descendant of
  IMPLEMENTATION_SHA; pass IMPLEMENTATION_SHA as --approved-source-sha; no merge
PRE_SUBMISSION_GATE: before each sbatch, record literal SOURCE_SHA, full sbatch argv,
  resolved fresh paths, exact profile hashes and remaining layered budget under ROOT
OWNER_APPROVAL: 2026-07-21 — approved base 1.20, code-bug reserve 0.50, hard 1.70
  charged GH200-hours and maximum concurrency one for this exact Section-9.2 sequence
```

Exact conditional invocations, all through
`fl_v3/scripts/run_s10_phase1p_ip_e2.sh` with the frozen resource tuple, are:

```text
COMMON:
  SOURCE_SHA=$(git rev-parse HEAD)
  SHA12=${SOURCE_SHA:0:12}
  ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
    arrhenius_fl_v3/outputs/s10_phase1p_ip_e2_df3c17e3e6be
  CONFIG=fl_v3/configs/s10_phase1_camera.json
  B8=fl_v3/configs/s10_phase1p_ip_e2_camera_sdpa_compile_fused_b8.json
  B16=fl_v3/configs/s10_phase1p_ip_e2_camera_sdpa_compile_fused_b16.json
  APPROVED_SOURCE=df3c17e3e6be19dcc586fdec2c6bd198c1b02d95

CELL_A_CAPACITY:
  --config ${CONFIG}
  --first-profile ${B16} --first-mode capacity
  --first-output ${ROOT}/camera/capacity_${SHA12}_r1_b16_probe
  --first-attempt b16_probe
  --source-sha ${SOURCE_SHA} --approved-source-sha ${APPROVED_SOURCE} --repeat 1

CELL_B_B8_THEN_B16:
  --config ${CONFIG}
  --first-profile ${B8} --first-mode sustained
  --first-output ${ROOT}/camera/sustained_${SHA12}_r1_b16_c1_b8
  --first-attempt b16_c1_b8
  --second-profile ${B16} --second-mode sustained
  --second-output ${ROOT}/camera/sustained_${SHA12}_r1_b16_c1_b16
  --second-attempt b16_c1_b16
  --pair-output ${ROOT}/pairs/b16_cell01_${SHA12}_r1.json
  --pair-reference first
  --source-sha ${SOURCE_SHA} --approved-source-sha ${APPROVED_SOURCE} --repeat 1

CELL_C_B16_THEN_B8:
  --config ${CONFIG}
  --first-profile ${B16} --first-mode sustained
  --first-output ${ROOT}/camera/sustained_${SHA12}_r2_b16_c2_b16_first
  --first-attempt b16_c2_b16
  --second-profile ${B8} --second-mode sustained
  --second-output ${ROOT}/camera/sustained_${SHA12}_r2_b16_c2_b8_second
  --second-attempt b16_c2_b8
  --pair-output ${ROOT}/pairs/b16_cell02_reverse_${SHA12}_r2.json
  --pair-reference second
  --source-sha ${SOURCE_SHA} --approved-source-sha ${APPROVED_SOURCE} --repeat 2
```

The containing `sbatch` command is fixed at account `naiss2025-22-1113-gpu`,
partition `gpu`, one node/task, `--cpus-per-task=16`, `--mem=96G`,
`--gpus-per-node=nvidia_gh200_120gb:1`, `--time=01:00:00`, `--no-requeue`.
Cells B and C each use one such allocation for both fresh processes and their pair
analysis. These commands are executable serially under the owner-approved
`1.20 + 0.50 = 1.70` layered ceiling and no broader authority.

## 10. Envelope-A compact execution ledger

This is the sole terminal ledger for Envelope A. Submission rows were appended only when
the exact durable source SHA and command are known.

| Item | Durable source / identity | State | Resources / interpretation |
|---|---|---|---|
| WP0 | `714f7a1067f375861c80e3020ab302a928983f12` | complete | local/static only; no compute |
| WP1 | `933ca6feb142bcedc2ab842b25d6a1caf242c749` | complete | exact CBGS artifact `64cc0d1d...e115ef`; no GPU submission |
| Swin acquisition 1/1 | source URL in Section 6.2; 114,342,173 bytes; SHA-256 `9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3` | accepted/read-only; quarantine absent | mapping report `c87469b8...f70f`; initialization-state identity sealed; no second download |
| WP2 Camera | implementation `5a001c96f00fffd0816492f181197e2d310a5ae1`; terminal source `8e55f5d135dbc83f2e937c4942117d9e1901f323` | terminal negative; pool backend not promoted | all correctness/e2e/memory gates passed; operator ratio `0.976174 > 0.80`; no capability inference |
| WP3 LiDAR | implementation `22138371d28e75d5218b0b888c225953fd429f0c`; terminal source `ae79a608f395fc9de841084ec4269451d826f587` | final engineering PASS | exact GTDB + collapsed sparse boundary + SECOND/SECONDFPN/TransFusion; qualified config `06e78e45...2015` |
| WP4 | initial `4c13ad736319c022d7fb6466a48a77c90ae79dde`; final `ae79a608f395fc9de841084ec4269451d826f587` | closed mixed outcome | Camera negative / LiDAR PASS; zero capability metrics and zero optimizer updates |
| WP4 checkpoint-I/O remediation | `67c1b55b59aa81a49b1ed8f4aabd07e6592e88aa` / tree `2c8812f57c3e59fce25ad1d6f3dd63044b39c714` | verified by Job F and later runs | scalar/N-D raw-byte hashing plus 0-D BatchNorm-buffer regression; no model/data/config change |
| O-148 preflight observability remediation | `125e915a0f16f8abfbfa14d73558ee518cf3170c` / tree `34840210a9d426c51973a29af4be91f06c5fe9f6` | verified beginning with Job E | names every fail-closed source/hash/module/environment/resource stage; no model/data/config/gate change |
| O-148 canonical-config remediation | `ea3cadec02cdd91f5caf5553631e916be008985f` / tree `811376a0c0eda575b7be3f87422024a6071ee02f` | verified by Job G and later runs | exact canonical bytes/physical SHA invariant plus production-path regression; no model/data/config/gate change |
| O-148 parity/sort remediation | `8e55f5d135dbc83f2e937c4942117d9e1901f323` / tree `1faebff1f495fad5cf798f7781a8371a250d432c` | verified by Job H; performance gate honestly failed | same-backend repeat control, deterministic parity isolation, exact composite-key fast sort; frozen tolerances/gates unchanged |
| Job A / `521859` | `4c13ad736319c022d7fb6466a48a77c90ae79dde`; config `f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d` | `FAILED 1:0` in focused test; engineering incident, no model/data execution | `00:01:42` = `0.028333` GH200-hour; 27 passed / 1 pooling parity failure |
| Job C / `521901` Camera derived replacement | remediation `564fb9d97c44a463ac055dc40d25b79acdc77858` / tree `a1b9f7e809708b72a927afa4ef9c3f4bae82e137` | `FAILED 1:0` in checkpoint hash; engineering incident, no checkpoint promotion/model/data execution | `00:01:48` = `0.030000` GH200-hour; pooling focused tests 29/29 passed |
| Job D / `521959` Camera O-147 replacement | `c45e020ed16496e2acaa5f8d34b135da21fb1230` / tree `3887d82545207ec67b861bf48ff49042f52cebdb`; config `f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d` | `FAILED 1:0` before runner control/output creation; exact pre-control predicate unlocalized | `00:00:04` = `0.001111` GH200-hour; no pytest/checkpoint/data/model/build/calibration execution; Job B blocked |
| Job E / `522037` Camera O-148 smoke | `125e915a0f16f8abfbfa14d73558ee518cf3170c` / tree `34840210a9d426c51973a29af4be91f06c5fe9f6`; config `f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d` | `FAILED 2:0` in explicit hash preflight; command-construction incident | `00:00:05` = `0.001389` GH200-hour; expected checkpoint-entry hash was truncated to 62 characters; no output/model/data execution |
| Job F / `522042` Camera O-148 smoke | same exact Job E source/config with mechanically derived 64-character file hashes | `FAILED 1:0` after checkpoint acceptance in resolved-config evidence write | `00:01:58` = `0.032778` GH200-hour; 30 tests passed; checkpoint accepted once; canonical bytes were incorrectly written with an extra newline before physical-hash comparison |
| Job G / `522094` Camera O-148 smoke | `ea3cadec02cdd91f5caf5553631e916be008985f` / tree `811376a0c0eda575b7be3f87422024a6071ee02f`; exact Job-G record below | `FAILED 2:0` at frozen pool-promotion gate after complete calibration | `00:04:02` = `0.067222` GH200-hour; standalone/FP32/e2e/memory passed; FP16 integrated gradient comparison and 1.25x operator-speed gate failed |
| Job H / `522113` Camera O-148 smoke | `8e55f5d135dbc83f2e937c4942117d9e1901f323` / tree `1faebff1f495fad5cf798f7781a8371a250d432c`; exact Job-H record below | complete negative qualification: `FAIL_POOL_PROMOTION_GATE` only | `00:04:15` = `0.070833` GH200-hour; all correctness/e2e/memory gates passed; optimized operator ratio 0.976174 > frozen 0.80; kernel not promoted |
| Original Job B / `522135` LiDAR O-148 smoke | `8e55f5d135dbc83f2e937c4942117d9e1901f323` / tree `1faebff1f495fad5cf798f7781a8371a250d432c`; exact Job-B record below | `FAILED 1:0` in focused test fixture before GTDB | `00:01:07` = `0.018611` GH200-hour; 30 passed/1 failed/3 skipped; test compared 3-D IoU with BEV oracle while random heights differed |
| Job B2 / `522153` LiDAR O-148 smoke | `cf53a29815a3bea6a65dbce9b6e74012f1b3e798` / tree `cac7e7d84681b654bb75e2ccbe3fe27ad93daf5b`; exact Job-B2 record below | `FAILED 1:0` after exact GTDB seal, before first calibration batch | `00:06:09` = `0.102500` GH200-hour; 31 passed/3 skipped; canonical JSON mapping-order parser defect |
| Job B3 / `522189` LiDAR O-148 smoke | `445239e965f9876c122f0b99135b0b9e8576018f` / tree `762aa2473ecad067c617195bf3892a5f04981325`; exact Job-B3 record below | `FAILED 1:0` in evaluator-schema decode after calibration | `00:04:18` = `0.071667` GH200-hour; 32 passed/3 skipped; indiscriminate FP32 helper cast discrete query indices |
| Job B4 / `522203` LiDAR O-148 smoke | `7dfc5aa173766d0f9b6a907db421f2d77882f137` / tree `7becda2242f411c8792b3de180e89c32df6b781c`; exact Job-B4 record below | `COMPLETED 0:0`; calibration passed, post-terminal path remediation required | `00:02:46` = `0.046111` GH200-hour; result paths retained vanished `.control` prefix |
| Job B5 / `522222` LiDAR O-148 smoke | `ae79a608f395fc9de841084ec4269451d826f587` / tree `2ed623162e79208c2fad8fd830f852d1a57793b3`; exact Job-B5 record below | `COMPLETED 0:0`; final LiDAR WP4 PASS | `00:02:45` = `0.045833` GH200-hour; all gates and published artifact paths passed |

Before O-148 execution, Envelope-A Slurm usage was `3 / unlimited` submissions and
`0.059444 / 1.10` charged GH200-hours. Final usage is `12 / unlimited` submissions
and `0.516389 / 1.10` GH200-hours. All jobs are terminal and the authority is consumed.

### O-147 exact amendment and Job D / Job B pre-submission record

```text
AUTHORITY_BASE_SHA: c45e020ed16496e2acaa5f8d34b135da21fb1230
AUTHORITY_BASE_TREE: 3887d82545207ec67b861bf48ff49042f52cebdb
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
SERIAL_ORDER: Job D Camera replacement PASS -> original Job B; any failure stops
SUBMISSION_USAGE_BEFORE: 2 / 5
GPU_HOURS_BEFORE: 0.058333 / 1.10 charged GH200-hours
JOB_D_SOURCE_SHA: c45e020ed16496e2acaa5f8d34b135da21fb1230
JOB_D_SOURCE_TREE: 3887d82545207ec67b861bf48ff49042f52cebdb
JOB_D_CONFIG: fl_v3/configs/s10_phase1_camera.json
JOB_D_CONFIG_FILE_SHA256: 7101578fdfa38ba364c41ebc9ccd986797fe3261492b1bb149d0f962ec134e55
JOB_D_RESOLVED_CONFIG_SHA256: f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d
JOB_D_RUNNER_SHA256: 48f962a274baf4a8205465cee6a21a596783e489e9abdf33743e1e9280c6d8a4
JOB_D_ENTRY_SHA256: 97947786cf08eca5f8baf1ab47be70030a767dc4be3079b971b95b16fde20b53
JOB_D_CHECKPOINT_ENTRY_SHA256: 7bf4d9a24687c6c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
JOB_D_DATA/SEED/GATES: exact original Job A values; D_fit first four official-CBGS B4;
                       seed 0; frozen correctness tolerances and 0.80/1.02/1.05 gates
JOB_D_OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_d_camera_c45e020ed164_d1
JOB_D_CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_c45e020ed164_d1
JOB_D_COMMAND: sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu
               --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
               --gpus-per-node=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue
               --job-name=s10-p1-d --output=<engineering-root>/slurm/job_d_%j.out
               --error=<engineering-root>/slurm/job_d_%j.err
               --export=ALL,<exact Job-D variables above>
               fl_v3/scripts/run_s10_phase1_job_a.sh
JOB_B_SOURCE_SHA/TREE: same exact c45e020... / 3887d825...
JOB_B_CONFIG: fl_v3/configs/s10_phase1_lidar.json
JOB_B_CONFIG_FILE_SHA256: 380bd6623af37241ee867b0bbe2e368abc22ec33292cb676d8189aa533dab1e1
JOB_B_RESOLVED_CONFIG_SHA256: b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf
JOB_B_RUNNER_SHA256: c69d0b53b4e20d2c078028ff31fab693dbfde103d54909bfb07b88b7a2871958
JOB_B_ENTRY_SHA256: 97947786cf08eca5f8baf1ab47be70030a767dc4be3079b971b95b16fde20b53
JOB_B_GTDB_ENTRY_SHA256: bc1136eb4ff5edc59090000d6f960632de1a8fac589f409477ef60ea54055de0
JOB_B_DATA/SEED: exact D_fit; keyframe-only training/GTDB; seed 0
JOB_B_OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_b_lidar_c45e020ed164_b1
JOB_B_COMMAND: sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu
               --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
               --gpus-per-node=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue
               --job-name=s10-p1-b --output=<engineering-root>/slurm/job_b_%j.out
               --error=<engineering-root>/slurm/job_b_%j.err
               --export=ALL,<exact Job-B variables above>
               fl_v3/scripts/run_s10_phase1_job_b.sh
PER_JOB_RESOURCES: original frozen values — 1 GH200, 16 CPU, 96 GiB,
                   <=00:30:00, no requeue
NO_MORE_DERIVED_SUBMISSIONS: true
```

### Job D `521959` terminal incident and O-147 stop

```text
TERMINAL: FAILED 1:0 / elapsed 00:00:04 / 0.001111 GH200-hour
NODE: n424
OUTPUT/CONTROL/BUILD: all absent; runner never reached control-directory creation
SLURM_STDOUT: empty, SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STDERR: empty, SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STREAM_MODE: 0444 after terminal evidence sealing
BOUND: failure occurred in pre-control source/path/hash/cleanliness checks, silent
       module/environment bootstrap, or Slurm resource assertions
NOT_EXECUTED: pytest, checkpoint mapping/promotion, D_fit read, model construction,
              CUDA build, pooling parity/timing, end-to-end calibration, evaluator
              schema, capability metrics, D_select, D_audit, official validation
CLASSIFICATION: pre-run engineering incident; exact predicate is unlocalized because
                the fail-closed runner emitted no trace before creating its control root
GTDB: absent; original Job B was not submitted
SUBMISSION_USAGE_AFTER: 3 / 5
GPU_HOURS_AFTER: 0.059444 / 1.10 charged GH200-hours
PHASE_STATE_AT_TERMINAL: STOPPED_OWNER_GATE under O-147's then-active rule;
                         superseded prospectively by O-148 engineering completion
```

### Job E exact Camera pre-submission record under O-148

```text
SOURCE_SHA: 125e915a0f16f8abfbfa14d73558ee518cf3170c
SOURCE_TREE: 34840210a9d426c51973a29af4be91f06c5fe9f6
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG: fl_v3/configs/s10_phase1_camera.json
CONFIG_FILE_SHA256: 7101578fdfa38ba364c41ebc9ccd986797fe3261492b1bb149d0f962ec134e55
RESOLVED_CONFIG_SHA256: f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d
RUNNER_SHA256: fd12524489c42530758afafc4fb69009c3591d5f81ffec65f9ebf777df378c3c
ENTRY_SHA256: 97947786cf08eca5f8baf1ab47be70030a767dc4be3079b971b95b16fde20b53
CHECKPOINT_ENTRY_SHA256: 7bf4d9a24687c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
DATA/SEED/GATES: unchanged exact Camera values; D_fit first four official-CBGS B4;
                 seed 0; frozen correctness tolerances and 0.80/1.02/1.05 gates
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_e_camera_125e915a0f16_e1
CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_125e915a0f16_e1
RESOURCES: 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
COMMAND: stable O-148 Camera invocation of run_s10_phase1_job_a.sh with the exact
         bindings above; account naiss2025-22-1113-gpu; partition gpu
USAGE_BEFORE: 3 / unlimited submissions; 0.059444 / 1.10 GH200-hours
```

### Job E `522037` terminal incident and Job F mechanical correction

```text
TERMINAL: FAILED 2:0 / elapsed 00:00:05 / 0.001389 GH200-hour / node n33
FAILURE: submitted S10_P1_EXPECTED_CHECKPOINT_ENTRY_SHA256 was a manually
         transcribed 62-character value missing `6c`; physical file SHA-256 is the
         mechanically recomputed 64-character
         7bf4d9a24687c6c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
CLASSIFICATION: command/provenance binding defect; runner correctly failed closed;
                no executable source, model, data, config, tolerance, or gate change
OUTPUT/CONTROL/BUILD: absent
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
SLURM_STDERR_SHA256: 6430e5a6c9a080495266eb6f4bd78922f12d905610bb58aef15059d5a00a6f0f
REMEDIATION: derive every expected SHA directly from the physical file in the
             pre-submission shell; assert 64-character lengths before sbatch
USAGE_AFTER: 4 / unlimited submissions; 0.060833 / 1.10 GH200-hours
```

### Job F exact Camera pre-submission record under O-148

```text
SOURCE/CONFIG/DATA/SEED/GATES: identical to Job E
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_f_camera_125e915a0f16_f1
CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_125e915a0f16_f1
HASH_BINDING: all expected source/file identities mechanically recomputed from HEAD;
              every Git/SHA-256 identity asserted to 40/64 characters before submission
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
USAGE_BEFORE: 4 / unlimited submissions; 0.060833 / 1.10 GH200-hours
```

### Job F `522042` terminal incident and canonical-config remediation

```text
TERMINAL: FAILED 1:0 / elapsed 00:01:58 / 0.032778 GH200-hour / node n415
PASSED: explicit preflight; 30/30 focused tests; one-time Swin checkpoint schema,
        tensor mapping, initialized-state identity and atomic promotion
CHECKPOINT: final physical SHA-256
            9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3;
            mapping report SHA-256
            c87469b84b4b865aa478cc1959c400468f8aca393e53cf8dbb92a71c3a63f70f;
            initialization-state SHA-256
            814eaf5adb58ecc5b5cfe253c63002bc6cad9390c01752b890298571fed01632;
            quarantine absent after atomic promotion; future jobs revalidate/reuse
FAILURE: _write_config_once wrote config.canonical_bytes plus a newline, then compared
         the physical file hash against ResolvedConfig.sha256, which is defined over
         canonical_bytes without that newline; the invariant could never pass
CLASSIFICATION: evidence-serialization implementation defect; no model/data/config/
                seed/tolerance/performance-gate change and no optimizer update
REMEDIATION: write the exact canonical byte stream and add a regression that imports
             the production calibrator, writes a resolved config, and requires both
             byte equality and physical SHA equality
NOT_EXECUTED: model construction, CUDA build, pooling parity/timing, end-to-end
              calibration, evaluator schema, capability metrics, D_select/D_audit/val
OUTPUT: immutable job_f_camera_125e915a0f16_f1
OUTPUT_MANIFEST_SHA256: 0d05d82a5e5879bf9323538801b6e076b684573477644ddeb1b5de00392b2139
USAGE_AFTER: 5 / unlimited submissions; 0.093611 / 1.10 GH200-hours
```

### Job G exact Camera pre-submission record under O-148

```text
SOURCE_SHA: ea3cadec02cdd91f5caf5553631e916be008985f
SOURCE_TREE: 811376a0c0eda575b7be3f87422024a6071ee02f
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG_FILE_SHA256: 7101578fdfa38ba364c41ebc9ccd986797fe3261492b1bb149d0f962ec134e55
RESOLVED_CONFIG_SHA256: f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d
RUNNER_SHA256: fd12524489c42530758afafc4fb69009c3591d5f81ffec65f9ebf777df378c3c
ENTRY_SHA256: 36214d881a263e40545c40b679004d7cec90dac1683df19622a67f79c76a7f2d
CHECKPOINT_ENTRY_SHA256: 7bf4d9a24687c6c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
DATA/SEED/GATES: unchanged exact Camera values
CHECKPOINT_LIFECYCLE: revalidate/reuse the accepted final checkpoint and mapping report;
                      no download or second acquisition
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_g_camera_ea3cadec02cd_g1
CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_ea3cadec02cd_g1
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 5 / unlimited submissions; 0.093611 / 1.10 GH200-hours
```

### Job G `522094` complete Camera diagnostic and bounded remediation

```text
TERMINAL: FAILED 2:0 / elapsed 00:04:02 / 0.067222 GH200-hour / node n185
PASSED: 31/31 focused tests; checkpoint revalidation; canonical materialized-config
        identity; standalone FP32/FP16 forward and exact feature-gradient parity;
        integrated FP32 output/upstream-gradient parity; identical e2e initialization;
        e2e median ratio 1.000378 <= 1.02; peak allocated ratio 0.999990 <= 1.05;
        zero optimizer/scheduler updates and unchanged parameters
FAILED_INTEGRATED_FP16: outputs exact, but 245/435 upstream parameter gradients exceeded
        rtol=2e-3/atol=2e-4; worst absolute 2.75. The serial harness ran accepted
        relaxed FP16 cuDNN/SDPA backward kernels twice without a same-backend repeat
        control, so unrelated nondeterministic upstream backward noise was attributed
        to the pooling backend despite exact standalone pool feature gradients.
FAILED_OPERATOR_GATE: fallback median 10.205792 ms; optimized median 9.791536 ms;
        ratio 0.959410 versus frozen maximum 0.80. Both dispatches paid the same stable
        int64 rank sort; the CUDA reduction saved only the remaining reduction/canvas cost.
CLASSIFICATION: complete diagnostic evidence, not capability evidence. Tolerances and
                0.80/1.02/1.05 gates remain unchanged.
REMEDIATION_PARITY: use strict deterministic backend-isolation only for serial parity
        capture while retaining FP16 autocast/scaler/FP32 pooling, add a fallback-repeat
        control under the same frozen tolerances, then restore relaxed accepted FP16
        policy for production timing.
REMEDIATION_PERFORMANCE: optimized dispatch uses a unique int64 composite
        `(rank, source_row)` key with the fast sorter; this is exactly equivalent to
        stable rank ordering and is regression-checked against the fallback order.
        Fallback implementation and all gates remain unchanged.
RESULT_SHA256: e070545de3daf9d56254be8067707d5ad8c52868cabdd0a97b207e65c6ac80b5
OUTPUT_MANIFEST_SHA256: 56cba8b7e292d5aa872a3f9fb93d03476168d9f77ed3b2ba17993e0a37378610
USAGE_AFTER: 6 / unlimited submissions; 0.160833 / 1.10 GH200-hours
```

### Job H exact Camera pre-submission record under O-148

```text
SOURCE_SHA: 8e55f5d135dbc83f2e937c4942117d9e1901f323
SOURCE_TREE: 1faebff1f495fad5cf798f7781a8371a250d432c
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG_FILE_SHA256: 7101578fdfa38ba364c41ebc9ccd986797fe3261492b1bb149d0f962ec134e55
RESOLVED_CONFIG_SHA256: f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d
RUNNER_SHA256: fd12524489c42530758afafc4fb69009c3591d5f81ffec65f9ebf777df378c3c
ENTRY_SHA256: 0df14419dbec7dba68db83cd36a43bc3db86fd86a7b906ba460b360cdf97cd71
CHECKPOINT_ENTRY_SHA256: 7bf4d9a24687c6c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
DATA/SEED/TOLERANCES/GATES: unchanged exact Camera values
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_h_camera_8e55f5d135db_h1
CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_8e55f5d135db_h1
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 6 / unlimited submissions; 0.160833 / 1.10 GH200-hours
```

### Job H `522113` final Camera WP4 result

```text
TERMINAL: FAILED 2:0 / elapsed 00:04:15 / 0.070833 GH200-hour / node n440
STATUS: complete negative Camera pooling qualification; not an engineering-smoke bug
PASSED: 32/32 focused tests; accepted-checkpoint revalidation; canonical config;
        standalone FP32/FP16 exact feature-gradient parity; integrated FP32 and FP16
        output/upstream-gradient parity; same-backend repeat controls; zero nonfinite;
        identical parameter coverage/initialization; zero optimizer/scheduler updates;
        unchanged parameters; e2e ratio 0.999519 <=1.02; peak allocated ratio
        0.999795 <=1.05
OPERATOR: fallback median/p95 10.235424/10.545619 ms; optimized median/p95
          9.991552/10.307205 ms; ratio 0.976174 > frozen 0.80
INTERPRETATION: on GH200 with PyTorch 2.11's exact segment-reduce fallback, the pinned
                MIT-style CUDA reduction provides only about 2.4% operator speedup and
                no material end-to-end improvement. Degrading the fallback, relaxing
                the gate, or altering math after observation is forbidden.
DISPOSITION: optimized kernel not promoted; fallback remains diagnostic-only; Camera
             qualified config/checkpoint is not emitted; this blocks Envelope B Camera
             capability until the owner explicitly amends the frozen production-backend
             requirement. Independent LiDAR WP4 may still be completed.
KNOWN_PROVENANCE_LIMIT: the later B4 audit found that this immutable result's sole
             materialized_config.path retains the pre-rename `.control` prefix. The
             actual final file exists at `<Job-H output>/evidence/resolved_config.materialized.json`
             with the recorded SHA256
             46e99574cd4c480b2cb6ffac07fa96c2e24d3c5e1deb9f5c75ae5eb6e0c230c9.
             Job H emitted no qualified config/checkpoint. `ae79a60` fixed final-path
             emission prospectively; rerunning the already terminal negative Camera
             gate would add no scientific information.
RESULT_SHA256: 47eacd4fb2211687703178fc4eadb027657dbdce6f79d9b3223b5e43415e7e41
OUTPUT_MANIFEST_SHA256: 87c6049f09de492a8998e4b9cdc94b2c126da81c56eca8c1956686c8e92eb19f
USAGE_AFTER: 7 / unlimited submissions; 0.231667 / 1.10 GH200-hours
```

### Original Job B exact LiDAR pre-submission record under O-148

```text
SOURCE_SHA: 8e55f5d135dbc83f2e937c4942117d9e1901f323
SOURCE_TREE: 1faebff1f495fad5cf798f7781a8371a250d432c
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG: fl_v3/configs/s10_phase1_lidar.json
CONFIG_FILE_SHA256: 380bd6623af37241ee867b0bbe2e368abc22ec33292cb676d8189aa533dab1e1
RESOLVED_CONFIG_SHA256: b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf
RUNNER_SHA256: af2a73e1de4b23aa05ff50914dc4bc4fe25cf397d05a495ec7eb56f4a93f4ddd
ENTRY_SHA256: 0df14419dbec7dba68db83cd36a43bc3db86fd86a7b906ba460b360cdf97cd71
GTDB_ENTRY_SHA256: bc1136eb4ff5edc59090000d6f960632de1a8fac589f409477ef60ea54055de0
DATA/SEED: exact D_fit; keyframe-only GTDB/training consumption; seed 0
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_b_lidar_8e55f5d135db_b1
GTDB: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_data_e321aed749fd/gtdb_keyframe_d_fit
RESOURCES: 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 7 / unlimited submissions; 0.231667 / 1.10 GH200-hours
```

### Original Job B `522135` terminal fixture incident

```text
TERMINAL: FAILED 1:0 / elapsed 00:01:07 / 0.018611 GH200-hour / node n440
PASSED: explicit preflight; 30 focused tests; 3 dependency/condition skips
FAILURE: test_phase1_vectorized_iou3d_matches_known_and_cpu_polygon_geometry
         compared pairwise 3-D IoU to an independent BEV-only polygon IoU while
         random boxes shared center-z but had different heights. Different heights
         change 3-D intersection/union, so equality with BEV IoU is mathematically false.
CLASSIFICATION: test-fixture defect; production pairwise_iou3d and TransFusion math
                remain unchanged
REMEDIATION: hold random-box height equal in the BEV-oracle comparison so 3-D IoU
             reduces exactly to BEV IoU; retain randomized x/y/l/w/yaw coverage
NOT_EXECUTED: GTDB materialization, model construction/calibration, evaluator schema,
              capability metrics, D_select/D_audit/official validation
GTDB: absent; no partial artifact
OUTPUT_MANIFEST_SHA256: 24b4e15b5e2c9cdb1e287d8159e5c0caf2cf408f0fc7f347b95877e7859b878e
USAGE_AFTER: 8 / unlimited submissions; 0.250278 / 1.10 GH200-hours
```

### Job B2 exact LiDAR pre-submission record under O-148

```text
SOURCE_SHA: cf53a29815a3bea6a65dbce9b6e74012f1b3e798
SOURCE_TREE: cac7e7d84681b654bb75e2ccbe3fe27ad93daf5b
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG_FILE_SHA256: 380bd6623af37241ee867b0bbe2e368abc22ec33292cb676d8189aa533dab1e1
RESOLVED_CONFIG_SHA256: b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf
RUNNER_SHA256: af2a73e1de4b23aa05ff50914dc4bc4fe25cf397d05a495ec7eb56f4a93f4ddd
ENTRY_SHA256: 0df14419dbec7dba68db83cd36a43bc3db86fd86a7b906ba460b360cdf97cd71
GTDB_ENTRY_SHA256: bc1136eb4ff5edc59090000d6f960632de1a8fac589f409477ef60ea54055de0
DATA/SEED/CONFIG/GATES: identical to original Job B
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_b2_lidar_cf53a29815a3_b2
GTDB: exact original Job-B path; absent before submission
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 8 / unlimited submissions; 0.250278 / 1.10 GH200-hours
```

### Job B2 `522153` terminal GTDB/config-order incident

```text
TERMINAL: FAILED 1:0 / elapsed 00:06:09 / 0.102500 GH200-hour / node n210
PASSED: explicit preflight; 31 focused tests; 3 dependency/condition skips
GTDB: exact keyframe-only D_fit database fully materialized and sealed before failure;
      321,613 objects across all ten frozen classes; manifest SHA256
      22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5;
      source cache SHA256 310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a
FAILURE_STAGE: first DataLoader batch, before model calibration/timing/result emission
FAILURE: canonical resolved-config JSON sorts mapping keys, while
         ReferenceGTDatabaseSampler incorrectly required the insertion order of the
         sample_groups mapping to equal the separate frozen class-order sequence
CLASSIFICATION: configuration parsing/order defect; JSON object order is not scientific
                state and the sampler already iterates the explicit class_names sequence
REMEDIATION: validate exact sample-group key-set equality, then reconstruct the mapping
             in the frozen class order; retain every count, class, seed, sampling loop,
             GTDB object, model operation and acceptance gate unchanged
NOT_EXECUTED: model forward/backward calibration, timing/memory, evaluator schema,
              capability metrics, D_select, D_audit, official validation
RUNNER_ARTIFACT_SHA256: d4b965f962f2c0c484b2217fc9b32a4130dc89b5fa4921ae12298c5599d21738
SLURM_STDERR_SHA256: b6f5095dceaeb557642eb6804a85d224711218f10da6475b2ae8e0e67f43eeba
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e464b934ca495991b7852b855
USAGE_AFTER: 9 / unlimited submissions; 0.352778 / 1.10 GH200-hours
```

### Job B3 exact LiDAR pre-submission record under O-148

```text
SLURM_JOB_ID: 522189
DERIVES_FROM: Job B2 522153 / one diagnosed mapping-order parser defect
SOURCE_SHA: 445239e965f9876c122f0b99135b0b9e8576018f
SOURCE_TREE: 762aa2473ecad067c617195bf3892a5f04981325
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG_FILE_SHA256: 380bd6623af37241ee867b0bbe2e368abc22ec33292cb676d8189aa533dab1e1
RESOLVED_CONFIG_SHA256: b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf
RUNNER_SHA256: af2a73e1de4b23aa05ff50914dc4bc4fe25cf397d05a495ec7eb56f4a93f4ddd
ENTRY_SHA256: 0df14419dbec7dba68db83cd36a43bc3db86fd86a7b906ba460b360cdf97cd71
GTDB_ENTRY_SHA256: bc1136eb4ff5edc59090000d6f960632de1a8fac589f409477ef60ea54055de0
DATA/SEED/CONFIG/GATES: identical to original Job B
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_b3_lidar_445239e965f9_b3
GTDB: reuse sealed exact Job-B2 database; manifest SHA256
      22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 9 / unlimited submissions; 0.352778 / 1.10 GH200-hours
```

### Job B3 `522189` terminal discrete-dtype incident

```text
TERMINAL: FAILED 1:0 / elapsed 00:04:18 / 0.071667 GH200-hour / node n451
PASSED: explicit preflight; 32 focused tests; 3 dependency/condition skips; sealed GTDB
        reuse; fixed D_fit batch materialization; LiDAR no-update calibration forward/
        backward path
FAILURE_STAGE: evaluator-schema preflight decode, before result/qualification emission
FAILURE: the shared _float_tensors precision helper converted every tensor to FP32,
         including TransFusion query_labels/query_indices; PyTorch one_hot correctly
         requires integral class indices
CLASSIFICATION: precision-plumbing dtype defect; discrete indices, labels and masks are
                outside the floating-point precision policy
REMEDIATION: upcast floating tensors only; preserve integral/boolean tensor dtypes;
             add an exact TransFusion output/decode regression
SCIENCE_EFFECT: no data, model math, floating precision, seed, recipe, evaluator schema,
                metric, tolerance, performance gate or resource change
NOT_EXECUTED: evaluator serialization completion, qualified-config/checkpoint emission,
              capability metrics, D_select, D_audit, official validation
RUNNER_ARTIFACT_SHA256: 0de55536fc3f657d0f5bbb6418ac6acb3dcaf67b8b5eb74b58afe50b7745eec4
SLURM_STDERR_SHA256: b6f5095dceaeb557642eb6804a85d224711218f10da6475b2ae8e0e67f43eeba
SLURM_STDOUT_SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e464b934ca495991b7852b855
USAGE_AFTER: 10 / unlimited submissions; 0.424445 / 1.10 GH200-hours
```

### Job B4 exact LiDAR pre-submission record under O-148

```text
SLURM_JOB_ID: 522203
DERIVES_FROM: Job B3 522189 / one diagnosed discrete-dtype precision-plumbing defect
SOURCE_SHA: 7dfc5aa173766d0f9b6a907db421f2d77882f137
SOURCE_TREE: 7becda2242f411c8792b3de180e89c32df6b781c
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG_FILE_SHA256: 380bd6623af37241ee867b0bbe2e368abc22ec33292cb676d8189aa533dab1e1
RESOLVED_CONFIG_SHA256: b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf
RUNNER_SHA256: af2a73e1de4b23aa05ff50914dc4bc4fe25cf397d05a495ec7eb56f4a93f4ddd
ENTRY_SHA256: 0df14419dbec7dba68db83cd36a43bc3db86fd86a7b906ba460b360cdf97cd71
GTDB_ENTRY_SHA256: bc1136eb4ff5edc59090000d6f960632de1a8fac589f409477ef60ea54055de0
DATA/SEED/CONFIG/GATES: identical to original Job B
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_b4_lidar_7dfc5aa17376_b4
GTDB: reuse sealed exact Job-B2 database; manifest SHA256
      22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 10 / unlimited submissions; 0.424445 / 1.10 GH200-hours
```

### Job B4 `522203` terminal calibration and provenance-path incident

```text
TERMINAL: COMPLETED 0:0 / elapsed 00:02:46 / 0.046111 GH200-hour / node n451
PASSED: 32 focused tests/3 skips; sealed GTDB reuse; fixed D_fit B4 data; 16 warm-up
        plus 64 timed microbatches; FP16 with accepted sparse FP32 island; 10 finite
        unscaled-gradient windows; unchanged parameters; zero optimizer/scheduler updates;
        evaluator decode/serialization; BN/no-GN check; exact checkpoint reload
CALIBRATION: 97.420113 ms median GPU step; 40.063358 samples/s; 5,346,498,048
             peak allocated bytes; unscaled gradient norm median 4,334.28125
SCHEMA: 29 BatchNorm1d, 15 BatchNorm2d, zero GroupNorm; sparse collapse FP32;
        decoded 200 boxes with the frozen evaluator schema
QUALIFIED_IDENTITIES: config 06e78e456793fe269c978b0e663da39e4ec3216523c54f996665bc1a6a952015;
                      zero-update recovery checkpoint
                      acf5e37f98c1e987acaad160731242637d20ecea42f076df9a922103dd099d3c
POST_TERMINAL_DEFECT: result.json recorded materialized config, qualified config and
                      checkpoint paths under the temporary `<output>.control/evidence`
                      root; atomic publish renamed that root, so all three strings became
                      stale although the files and hashes are correct in final evidence
CLASSIFICATION: output-neutral artifact/provenance publication defect
REMEDIATION: pass the immutable published output root explicitly, derive and validate
             all three final artifact paths before emission, and make runners gate them
RESULT_SHA256: 0d85faec3c31830ea09739ff19e590b165382d39d5e5e4b64da17eb70c44fbf5
RUNNER_ARTIFACT_SHA256: cabc2c5954c0b8404526df42997405d37b4caef49ec3f17fa55152943f982f84
USAGE_AFTER: 11 / unlimited submissions; 0.470556 / 1.10 GH200-hours
```

### Job B5 exact LiDAR pre-submission record under O-148

```text
SLURM_JOB_ID: 522222
DERIVES_FROM: Job B4 522203 / one diagnosed post-rename provenance-path defect
SOURCE_SHA: ae79a608f395fc9de841084ec4269451d826f587
SOURCE_TREE: 2ed623162e79208c2fad8fd830f852d1a57793b3
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
CONFIG_FILE_SHA256: 380bd6623af37241ee867b0bbe2e368abc22ec33292cb676d8189aa533dab1e1
RESOLVED_CONFIG_SHA256: b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf
RUNNER_SHA256: 3bc214c3e709ce7a1c964a0653be5b8da032dac2733f613897112743d01eae5e
ENTRY_SHA256: a879b853be8fc98cad9095363914701d02d125a10e908e2f754ea92f3234719f
GTDB_ENTRY_SHA256: bc1136eb4ff5edc59090000d6f960632de1a8fac589f409477ef60ea54055de0
DATA/SEED/CONFIG/GATES: identical to original Job B
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_b5_lidar_ae79a608f395_b5
GTDB: reuse sealed exact Job-B2 database; manifest SHA256
      22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5
RESOURCES: unchanged 1 GH200, 16 CPU, 96 GiB, <=00:30:00, no requeue
HASH_BINDING: mechanically recomputed with 40/64-character assertions
USAGE_BEFORE: 11 / unlimited submissions; 0.470556 / 1.10 GH200-hours
```

### Job B5 `522222` final LiDAR WP4 result

```text
TERMINAL: COMPLETED 0:0 / elapsed 00:02:45 / 0.045833 GH200-hour / node n451
STATUS: PASS; final directly consumable LiDAR Envelope-A calibration evidence
PASSED: 33 focused tests/3 skips; exact sealed GTDB reuse; fixed D_fit B4 data;
        16 warm-up plus 64 timed microbatches; 10 finite unscaled-gradient windows;
        unchanged parameters and zero optimizer/scheduler updates; evaluator decode/
        serialization; BN/no-GN and sparse-FP32-island checks; exact checkpoint reload;
        three final published artifact paths exist and result contains no `.control`
CALIBRATION: median/p95 GPU step 93.118401/106.594336 ms; 41.904378 samples/s;
             peak allocated/reserved 5,346,498,048/6,360,662,016 bytes;
             unscaled gradient norm median/p95 4,333.033203/4,336.105469
SCHEMA: 29 BatchNorm1d, 15 BatchNorm2d, zero GroupNorm; sparse collapse FP32
        `[4,256,180,180]`; decoded/serialized 200 boxes
DATA: exact keyframe-only D_fit training/GTDB; GTDB manifest SHA256
      22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5;
      321,613 objects across all ten frozen classes
QUALIFIED_CONFIG_SHA256: 06e78e456793fe269c978b0e663da39e4ec3216523c54f996665bc1a6a952015
ZERO_UPDATE_CHECKPOINT_SHA256: 8166d2016a560d7b572ec7d196a886f0780eb317f6a8a32f8a86f80160e92611
MODEL_STATE_SHA256: cfefa8e50e8f868bfdeddbb16dd57138ad83c5727f90a3d1be5d9461dfc3f96e
RESULT_SHA256: 5cd76d723c6cd68ef76b3bcd4d2ff1950f880524e622d14ecef4883105c608e0
RUNNER_ARTIFACT_SHA256: 2a0929a1bedb9a974196eb27f68f3e8697c6dfcb3550b1d6f0f7d346504d17ad
INTERPRETATION: implementation conformance/numerical health/resource estimate only;
                no capability, convergence, mAP/NDS or candidate-selection claim
SCOPE_NOT_EXECUTED: optimizer update, D_select, D_audit, official validation,
                    capability metrics, staged fusion, scientific checkpoint
USAGE_AFTER: 12 / unlimited submissions; 0.516389 / 1.10 GH200-hours
```

### Job A exact pre-submission record

```text
SOURCE_SHA: 4c13ad736319c022d7fb6466a48a77c90ae79dde
SOURCE_TREE: af1a582488191b0e49799ebc02b9489990ce0edf
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
SOURCE_CONFIG: fl_v3/configs/s10_phase1_camera.json
SOURCE_CONFIG_FILE_SHA256: 7101578fdfa38ba364c41ebc9ccd986797fe3261492b1bb149d0f962ec134e55
RESOLVED_CONFIG_SHA256: f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d
RUNNER_SHA256: 48f962a274baf4a8205465cee6a21a596783e489e9abdf33743e1e9280c6d8a4
ENTRY_SHA256: c56d1e9b8c79586aa1651d5d8b65706a29d71ec5a3700577374af9c95e415da8
CHECKPOINT_ENTRY_SHA256: 7bf4d9a24687c6c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
DATA: exact D_fit / official CBGS epoch-0 order / first four physical-B4 batches
SEED: 0
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_a_4c13ad736319_a1
CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_4c13ad736319
RESOURCES: account naiss2025-22-1113-gpu; partition gpu; 1 node; 1 task;
           1 nvidia_gh200_120gb; 16 CPU; 96 GiB; 00:30:00; no requeue
COMMAND: sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu
         --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
         --gpus-per-node=nvidia_gh200_120gb:1 --time=00:30:00 --no-requeue
         --job-name=s10-p1-a --output=<engineering-root>/slurm/job_a_%j.out
         --error=<engineering-root>/slurm/job_a_%j.err
         --export=<exact variables above> fl_v3/scripts/run_s10_phase1_job_a.sh
STOP: focused-test/checkpoint/content/build/parity/promotion failure; timeout; source/config/
      data/resource drift; no automatic Job C until one engineering cause is diagnosed
INTERPRETATION: implementation conformance, numerical parity and engineering timing only
```

### Job C `521901` terminal incident and phase stop

```text
TERMINAL: FAILED 1:0 / elapsed 00:01:48 / 0.030000 GH200-hour
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/
        s10_phase1_envelope_a_eng_e321aed749fd/job_c_camera_564fb9dc58a9_c1
PASSED: 29/29 focused tests, including optimized/fallback FP32/autocast forward and
        exact feature-gradient parity plus cross-cell rounding regression
FAILURE_STAGE: one-time Swin state mapping/identity, before report write and atomic rename
OBSERVED: tensor_state_sha256 called tensor.view(torch.uint8) on a scalar Long
          BatchNorm num_batches_tracked buffer; PyTorch 2.11 rejects 0-D
          cross-element-size view
ARTIFACT_LIFECYCLE: quarantine remains present/read-only with original physical SHA;
                    final checkpoint and mapping report remain absent; acquisition count
                    remains one; no second download occurred
NOT_EXECUTED: D_fit read, model calibration, operator/e2e timing, checkpoint preflight,
              evaluator schema, capability metrics, D_select, D_audit, official val
CLASSIFICATION_AT_TERMINAL: output-neutral checkpoint identity/hash implementation defect,
                distinct from Job 521859's corrected oracle defect; Job C was consumed
                and another Camera submission was not then authorized
LOCAL_REMEDIATION: `67c1b55b59aa81a49b1ed8f4aabd07e6592e88aa`; export contiguous
                   tensor bytes through NumPy for both 0-D and N-D tensors and add
                   an exact scalar-buffer identity regression test
PHASE_STATE_AT_TERMINAL: STOPPED_OWNER_GATE; O-147 later supplied one bounded Camera
                         replacement followed conditionally by Job B
```

### Job A `521859` incident and derived-replacement classification

```text
TERMINAL: FAILED 1:0 / elapsed 00:01:42 / 0.028333 GH200-hour
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/
        s10_phase1_envelope_a_eng_e321aed749fd/job_a_4c13ad736319_a1
TEST: test_bev_pool_optimized_forward_backward_and_autocast_policy
OBSERVED: 27 passed, 1 failed; 272/7854 mismatches; max absolute 2.7298927e-05;
          frozen rtol=1e-5 / atol=1e-6; no checkpoint promotion, D_fit read,
          model construction, calibration, capability metric or scientific checkpoint
DIAGNOSIS: optimized kernel matches pinned MIT per-cell sequential FP32 summation;
           the independent fallback incorrectly used one global prefix cumsum and
           cross-cell subtraction, so a preceding cell's rounded partial sum perturbed
           later cells. The mismatch is an oracle implementation defect, not evidence
           against the production kernel and not permission to relax tolerance.
REMEDIATION_CLASS: pre-conformance reference/fallback implementation repair allowed by
                   Section 6.1; replace only the fallback reduction with PyTorch's
                   length-delimited per-cell segment reduction, which uses the same
                   start-to-end FP32 order and retains an extension-independent backward.
SCIENCE_EFFECT: optimized candidate math/data/precision/config/seed/gates unchanged;
                only the diagnostic oracle is corrected to the frozen reference order
DERIVED_JOB_C: eligible after a durable remediation SHA and fresh output/build roots;
               same command family, data, seed, resources, tolerances and stop gates
```

### Job C exact derived pre-submission record

```text
DERIVES_FROM: Job A 521859 / one diagnosed fallback-oracle implementation defect
SOURCE_SHA: 564fb9d97c44a463ac055dc40d25b79acdc77858
SOURCE_TREE: a1b9f7e809708b72a927afa4ef9c3f4bae82e137
SOURCE_BRANCH: codex/s10-phase1-branch-qualification
SOURCE_CONFIG: fl_v3/configs/s10_phase1_camera.json
SOURCE_CONFIG_FILE_SHA256: 7101578fdfa38ba364c41ebc9ccd986797fe3261492b1bb149d0f962ec134e55
RESOLVED_CONFIG_SHA256: f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d
RUNNER_SHA256: 48f962a274baf4a8205465cee6a21a596783e489e9abdf33743e1e9280c6d8a4
ENTRY_SHA256: c56d1e9b8c79586aa1651d5d8b65706a29d71ec5a3700577374af9c95e415da8
CHECKPOINT_ENTRY_SHA256: 7bf4d9a24687c6c6c5ac72128f53e35cc99d1f7420bc3611a5c76576833cc402
DATA/SEED/GATES: identical to Job A; exact D_fit first four official-CBGS B4; seed 0;
                 frozen forward/backward tolerances and 0.80/1.02/1.05 gates unchanged
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_e321aed749fd/job_c_camera_564fb9dc58a9_c1
CUDA_BUILD: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_564fb9dc58a9
RESOURCES: identical to Job A — 1 GH200, 16 CPU, 96 GiB, 00:30:00, no requeue
STOP: any focused-test, content, parity, performance, resource or runtime failure;
      Job C is the sole derived replacement and cannot be retried
INTERPRETATION: implementation conformance, numerical parity and engineering timing only
```
