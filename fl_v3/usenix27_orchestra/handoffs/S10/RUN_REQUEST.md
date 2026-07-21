# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S10 Phase I-P throughput preflight
ACTIVE_DECISION: owner-approved IP-E1 under O-143/O-149; O-150 remains the frozen Phase-I control
REQUEST_STATE: IP-E1 ACTIVE / STRICT CAMERA WP2 RESUMED / WP3-ENVELOPE B FROZEN
EXECUTION_AUTHORITY: serial approved WP2 order inside the unchanged layered ceiling
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
REQUEST_STATE: ACTIVE / SERIAL EXECUTION AUTHORIZED
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
EXECUTABLE_NOW: yes, Section 8 only
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

Current accounting after Job `527284`: base cells `1.179444 / 2.0`, code-bug
reserve `0.146389 / 1.0`, hard aggregate `1.325833 / 3.0` charged GH200-hours.
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

## 9. Envelope-A compact execution ledger

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
