# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S10 Phase I-P throughput preflight
ACTIVE_DECISION: final Camera two-GH200 and LiDAR B32 recipes owner-promoted
REQUEST_STATE: REVISED SERIAL ENVELOPE B OWNER-ACCEPTED / EXECUTION DEFERRED
EXECUTION_AUTHORITY: Section 7.4 accepted at seal 1473ef67... for a later exact session
ACTIVE_PHASE: current-session no-submit hold; later execution-session startup verification
PLAN: HANDOFF.md Section 1 / IP-G0 closed
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at the same SHA
ENVELOPE_B: old Section 7 is control; revised Section 7.4 owner-accepted / current-session hold
```

IP-G0 authorized scoped Phase I-P source/docs/tests, local validation and linear
commits. The owner subsequently activated Section 8 at implementation commit
`85c6719e4b880b198d850e16b1418c230fa5c656`, including continuous IP-WP1 ->
IP-WP2. This does not authorize any evaluation role, the Section-7 Envelope B,
merge, push or publication.

After promoting the final Camera B16x2 recipe, the owner froze the next serial
order: final-B16 stage trace and conservative affine/grid screen; only then a
separately prepared same-node 2-GH200 DDP qualification. Source commit
`9233af3119857511f5f2acc310a182449e7b91a2` prepared the single-GPU Section-9.4
path. IP-E3/IP-E4 are now terminal. The owner subsequently authorized materializing
the final recipe binding and designing the exact Section-9.6 IP-E5 DDP
source/tests/resource envelope at `e51df6efa04e6d151315c72b7d7016014852078c`,
then activated that exact envelope at containing request commit
`2505db02920021663ccce7783dee483f10e638f8`.
After its terminal positive evidence, the owner explicitly accepted per-rank B16
BatchNorm and `seed + epoch*world_size + rank` worker RNG and promoted the exact
two-GH200 production recipe. Source
`2c3780bb6373ae784b41c22df072824f7a92d457` materializes that recipe; it grants
no Slurm or Envelope-B execution authority.

LiDAR IP-L-E3 subsequently closed positive and the owner promoted its exact B32
combined recipe. Source `cb2fc279b0c5e4b686525bed9da10f3ec6ad070f`
now materializes the revised dual-branch manifest, final config hashes, common fresh
output root and fail-closed launcher binding. This source still grants no compute;
independent review of remediation source
`a4f6ca86ddd966bdffc74a37af3337ac6675e83a` closed with no open P0-P2. Section
7.4 was later owner-accepted at named review seal
`1473ef67d9dc2949c49360b6826d0f30585f416f`, with serial concurrency one retained
and all submission deferred out of the accepting session.

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
EXPOSURE: 20 exact-CBGS epochs; effective global B32; revised branch-specific
          physical batch/topology is bound by Section 7.4
CHECKPOINT_SELECTION: epoch-20 terminal only
WORKFLOW: 5 WPs + 3 owner gates + 2 approval envelopes
CAMERA_POOLING: O-150 PyTorch sorted segment-reduce production backend; CUDA option
                retained unpromoted; WP2/WP4 parity/policy evidence retained
EXECUTION_AUTHORITY: Envelope A and every Phase I-P profiler envelope are consumed;
                     revised Section 7.4 is review/owner-gated and NOT EXECUTABLE;
                     the old 49.0-hour request is historical control only
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

### 7.4 Revised dual-branch Envelope B — owner-accepted, current-session hold

Sections 7.0–7.3 above remain the immutable pre-Phase-I-P B4 control. They are not
an activation option. The object below supersedes their source/config/topology/
resource/output identities while preserving the two-candidate scientific contract.

```text
PHASE: S10 Phase I / revised Envelope B independent branch qualification
REQUEST_STATE: FROZEN / REVIEW CLOSED / OWNER ACCEPTED / EXECUTION DEFERRED
EXECUTABLE_IN_ACCEPTING_SESSION: no; explicit owner no-submit hold
FUTURE_EXECUTION: allowed only from a later session after exact startup verification
MATERIALIZED_SOURCE_SHA: cb2fc279b0c5e4b686525bed9da10f3ec6ad070f
MATERIALIZED_SOURCE_TREE: 3dd9bc54a30d766f696ab752abdc1a8f4097d55c
REVIEW_BASELINE: a4f6ca86ddd966bdffc74a37af3337ac6675e83a
REVIEW_BASELINE_TREE: 48f71e4a917d5c2dc47287f110a667752e03976d
REVIEW_VERDICT: PASS_WITH_RESIDUAL_RISK; P0=0, P1=0, P2=0, P3=1; open P0-P2=0
ACTIVATION_BASELINE: 1473ef67d9dc2949c49360b6826d0f30585f416f
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA; do not move
OBJECTIVE: train the exact promoted LiDAR and Camera primaries to terminal exposure,
  evaluate each raw epoch-20 terminal checkpoint once on D_select, and return both
  results to P1-G2 without post-hoc tuning or a capability threshold
CANDIDATES_AND_MAX_COUNT: exactly 2 — phase1_lidar_primary then
  phase1_camera_primary; no replacement scientific candidate
SEED_POLICY: seed 0 only; no alternate or confirmatory seed
DATA: accepted STOP-A train-parent split; D_fit train; D_select terminal assessment
SPLIT_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
CBGS: expanded 87,930 / consumed 87,904 / drop 26 after exact epoch permutation
EXPOSURE_PER_CANDIDATE: 20 epochs / 1,758,080 consumed presentations /
  54,940 attempted effective-B32 optimizer updates
CAMERA_RECIPE: one node / world size 2 / physical B16 per rank / accumulation 1 /
  effective global B32 / ordinary rank-local B16 BN / contiguous rank halves of each
  global CBGS B32 window / seed+epoch*world_size+rank worker RNG / rank-0 canonical
  model checkpoint plus one RNG sidecar per rank; PyTorch sorted segment_reduce;
  conservative batched affine/grid, vectorized geometry, bulk input conversion,
  Camera SDPA, five-module forward-only compile and fused AdamW
LIDAR_RECIPE: one node / world size 1 / physical B32 / accumulation 1 / effective
  global B32 / ordinary physical-B32 BN / seed+epoch worker RNG; exact keyframe GTDB;
  batched target/Hungarian host plumbing, CPU point offsets and forward-only compile
  of decoder_backbone/decoder_neck/head; LiDAR SDPA and fused AdamW off
PRECISION: accepted S08 FP16 policy; Camera pool/loss FP32; LiDAR sparse FP32 island;
  GradScaler initial scale 8; TF32 off
OPTIMIZER_SCHEDULER: frozen per-branch AdamW/cyclic schedule; advance only on accepted
  effective-B32 updates; Camera fused implementation true, LiDAR false
CHECKPOINT: atomic epoch-boundary recovery cadence remains one epoch; recovery is
  non-selectable; raw epoch-20 terminal only is selectable; exact config/training/
  optimizer/scheduler/scaler/RNG identities required for resume
EVALUATOR: frozen train-subset adapter over official nuScenes detection metric math;
  detection config SHA256 217f96cca4e80f790c4674ef72257a6863ee9a85b0ce185bc56488afc32c7a0b
D_SELECT: exactly one completed execution per terminal candidate; no re-decode for
  selection
D_AUDIT: forbidden and unbudgeted; remains owner-sealed pending a separate P1-G2
  amendment
OFFICIAL_VALIDATION: forbidden
CHECKPOINT_SELECTION: epoch-20 terminal raw weights only; no best-epoch selection
SERIAL_ORDER: LiDAR terminal result first, then Camera unless a shared-boundary
  failure stops the envelope
MAX_CONCURRENCY: 1
SUBMISSION_POLICY: no numeric engineering-remediation count cap; exactly two
  scientific candidates; no duplicate scientific rerun
AGGREGATE_GPU_HOURS: 30.0 charged GH200-hours across both initial jobs, exact
  checkpoint continuation and eligible O-149 remediation
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1_envelope_b_dual_783173d6fe05
OUTPUT_RULE: one fixed candidate directory per branch; raw artifacts write once;
  every invocation writes a fresh attempt record; exact checkpoint resume reuses the
  same candidate directory; no overwrite or blind identical retry
ENGINEERING_REMEDIATION: O-149 single-correct-answer config/schema/API/test/runner/
  checkpoint/provenance/logging repairs anchored to frozen semantics; record the
  derived source, command, attempt and charge before serial continuation
SCIENTIFIC_CONTINUATION: a weak LiDAR score does not cancel Camera unless it
  implicates a shared data/evaluator/precision/configuration boundary
REVIEW_GATE: closed by one independent read-only review at REVIEW_BASELINE; no GPU
  and no edits by reviewer; open P0-P2=0; one P3 recorded and owner-accepted
OWNER_ESCALATION: model math/shape, normalization, initialization, data/order/GTDB,
  augmentation, target/loss/decode, optimizer/scheduler/precision, seed, exposure,
  selectable checkpoint, evaluator/metric, candidate, topology/resource/output-root
  change; ambiguous defect; same blocker recurrence; shared-boundary failure; or
  30.0-hour exhaustion
ALLOWED_INTERPRETATION: single-seed internal Camera/LiDAR branch capability and
  engineering health
FORBIDDEN_INTERPRETATION: official-val/generalization, fusion, FL, attack/defense,
  publication claim or best-recipe optimality
OWNER_APPROVAL: accepted; review verdict and single P3 accepted; serial concurrency
  one retained; no submission from the accepting session
```

Immutable manifest, recipe and entry identities:

| Object | File SHA-256 | Resolved SHA-256 / role |
|---|---|---|
| dual manifest `fl_v3/configs/s10_phase1_envelope_b_dual.json` | `4d83f0741d5b77d476b1e3cdd5ef10b7330b4564af271f3b2bb50ac6c1f79afd` | binds order, roots, resources and all entries |
| LiDAR config `fl_v3/configs/s10_phase1_lidar.json` | `017086bbd9a9534adf2808461da9cf881d9ef798ef3f3d7c58d3a07b2c7a15d9` | `c950d90db0833ecf5f50ddcc2f10671e4abf7a9f2b1edd640425eb52b888b1ad` |
| Camera config `fl_v3/configs/s10_phase1_camera.json` | `89a4d9982583dc213e110fcec9469be04e9b4ccf3cefb9a2ca97b294e7650014` | `63f77459fcb229155a0b1a6608d83abf3c55336d554c20f7629d57ed7122d1b3` |
| launcher `fl_v3/scripts/run_s10_phase1_envelope_b.sh` | `1daad38dba352664b1072d97774e2f24b5ed30c52a01c75b3f36752b33c4dd99` | validates manifest/config/entry/output plus allocated account/partition/node/task/CPU/memory/GPU-count identities before dispatch |
| LiDAR entry `fl_v3/scripts/s10_phase1_capability.py` | `4c93348330ee02b56a9fc282e991f391c2f986a9dbab7b704bd2195a5f79ec55` | single-GPU production entry |
| Camera entry `fl_v3/scripts/s10_phase1_camera_ddp.py` | `4b91e81c5060bec0108b99abaa6b29e6df4d4def0d04f45e54a4b20df830162e` | two-rank production DDP entry |

The Camera initialization remains physical SHA-256
`9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3`,
mapping SHA-256 `c87469b84b4b865aa478cc1959c400468f8aca393e53cf8dbb92a71c3a63f70f`
and initialized-state SHA-256
`814eaf5adb58ecc5b5cfe253c63002bc6cad9390c01752b890298571fed01632`.
LiDAR remains scratch seed 0 with GTDB manifest SHA-256
`22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5`.

#### 7.4.1 Evidence-based resource projection

Camera IP-E5 measured `64.886915` presentations/s and projects `7.581252` wall
hours on two GH200s, or `15.162504` charged GH200-hours. LiDAR IP-L-E3 measured
`59.336641` presentations/s and projects `8.261479` wall/charged hours. Both
projections include compile cold start and epoch checkpoints. A conservative `1.2`
charged hours covers the two terminal evaluations, production preflights and
recovery overhead; the original 15% contingency is then retained:

```text
training charge                           = 23.423983
evaluation/preflight/recovery reserve     =  1.200000
subtotal                                  = 24.623983
15% contingency                           =  3.693597
computed need                             = 28.317580
requested hard ceiling                    = 30.000000 GH200-hours
```

| Branch | Exact initial resource | Projected training | Initial-job limit |
|---|---|---:|---:|
| LiDAR | 1 node / 1 typed GH200 / 16 CPU / 96 GiB | 8.261479 wall/charged h | `10:00:00`, at most 10.0 charged h |
| Camera | 1 node / 2 typed GH200 / 32 CPU / 192 GiB | 7.581252 wall h / 15.162504 charged h | `09:00:00`, at most 18.0 charged h |

The two initial-job maxima total `28.0` charged hours. The aggregate ceiling's
remaining `2.0` hours is not another cell: it may cover only exact continuation or
eligible frozen-semantics remediation. D_audit has no reserve in this request.

#### 7.4.2 Exact serial command family for a later execution session

The owner has named review seal `<APPROVED_BASELINE_SHA>` as
`1473ef67d9dc2949c49360b6826d0f30585f416f`, but explicitly directed that the
accepting session create no output directory and submit no job. The commands below
are documentation for a later exact execution session only. As with prior S10
phase envelopes, `<EXECUTION_SOURCE_SHA>` is the clean descendant actually run and
recorded in the ledger; it must preserve every reviewed entry/config/manifest hash.

```bash
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_dual_783173d6fe05
mkdir -p "${ROOT}/slurm"
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=10:00:00 --no-requeue \
  --job-name=s10-p1b-lidar --output="${ROOT}/slurm/lidar-%j.out" \
  --error="${ROOT}/slurm/lidar-%j.err" \
  fl_v3/scripts/run_s10_phase1_envelope_b.sh --branch lidar \
  --envelope fl_v3/configs/s10_phase1_envelope_b_dual.json \
  --config fl_v3/configs/s10_phase1_lidar.json \
  --output-dir "${ROOT}/phase1_lidar_primary" --source-sha <EXECUTION_SOURCE_SHA>
```

Only after LiDAR reaches a terminal result, or a weak result that does not implicate
a shared boundary, may Camera be submitted:

```bash
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_b_dual_783173d6fe05
sbatch --parsable --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=192G \
  --gpus-per-node=nvidia_gh200_120gb:2 --time=09:00:00 --no-requeue \
  --job-name=s10-p1b-camera --output="${ROOT}/slurm/camera-%j.out" \
  --error="${ROOT}/slurm/camera-%j.err" \
  fl_v3/scripts/run_s10_phase1_envelope_b.sh --branch camera \
  --envelope fl_v3/configs/s10_phase1_envelope_b_dual.json \
  --config fl_v3/configs/s10_phase1_camera.json \
  --output-dir "${ROOT}/phase1_camera_primary" --source-sha <EXECUTION_SOURCE_SHA>
```

An exact checkpoint continuation uses the same branch/manifest/config/output/source
and adds `--resume`; its requested wall must fit the remaining aggregate charge. A
derived O-149 repair uses a new durable source and fresh attempt record without
overwriting raw artifacts. No command may alter candidate, seed, data, model, recipe,
precision, exposure, selection or evaluator semantics.

#### 7.4.3 Independent recipe-freeze review closure

The first read-only review pinned documentation baseline
`290995a087e0e1f982f3327a4db82c8fb8514054` and returned `REMEDIATE`: P0/P1 were
zero, one P2 identified stale active prose for the superseded 49-hour B4 object, and
one P3 identified overbroad wording for launcher resource validation. No reviewer
edit or GPU/Slurm action occurred.

Documentation-only remediation `a4f6ca86ddd966bdffc74a37af3337ac6675e83a`
(tree `48f71e4a917d5c2dc47287f110a667752e03976d`) made Section 7.4's 30-hour
object the unique current activation candidate and accurately split enforcement:
the launcher validates config/output plus allocated account, partition, node/task,
CPU, memory and GPU count; the exact `sbatch` command and ledger control wall limit,
`--no-requeue` and aggregate charge.

The same independent reviewer rechecked that clean SHA and returned
`PASS_WITH_RESIDUAL_RISK`: P0/P1/P2 are all zero, open P0-P2 is zero and the
recipe-freeze gate is closed. Manifest/config/launcher/entry file hashes, both
resolved-config hashes, recipes, science and resource limits are unchanged. The
single accepted P3 is the documented operational reliance on the exact `sbatch`
command and ledger for wall/requeue/aggregate enforcement. This review is not
compute authority.

The owner subsequently named the clean commit containing this review record,
accepted the review verdict and P3, retained serial concurrency one, and accepted
Section 7.4. The accepting session remains under an explicit no-submit hold.

#### 7.4.4 Owner acceptance and execution deferral

```text
OWNER_REVIEW_DISPOSITION: accepted PASS_WITH_RESIDUAL_RISK and the single P3
OWNER_NAMED_REVIEW_SEAL: 1473ef67d9dc2949c49360b6826d0f30585f416f
OWNER_ENVELOPE_DISPOSITION: accepted exact revised Section 7.4
ORDER_AND_CONCURRENCY: unchanged; serial LiDAR then Camera; maximum concurrency 1
PARALLEL_AMENDMENT: not adopted; may be reconsidered only in later work
CURRENT_SESSION_SUBMISSION: forbidden by explicit owner direction
JOBS_SUBMITTED: 0
CHARGED_GH200_HOURS: 0
OUTPUT_ROOT_STATE: must remain absent in the accepting session
```

This decision closes `P1-G1 SCIENTIFIC_COMPUTE_APPROVAL` for the exact reviewed
serial envelope while deferring execution. A later session may use this authority
only after reporting clean status, branch/base/topology, exact approved baseline and
unchanged manifest/config/entry hashes, confirming the output root is still absent,
and preserving every Section-7.4 stop and escalation rule. It does not authorize
parallel execution, D_audit, official validation, Fusion, merge, push or publication.

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

### 9.2 IP-G2 B16 extension — terminal positive

This was an independently bounded extension. It does not reinterpret or reuse the
exhausted Section-9 IP-E2 budget. On 2026-07-21, the owner explicitly approved the
exact layered ceiling and concurrency below. The sequence is now terminal; unused
budget expires and this does not activate Envelope B or authorize a capability run.

```text
PHASE: S10 Phase I-P / IP-G2 B16 capacity and matched-throughput extension
REQUEST_STATE: TERMINAL POSITIVE / CELL A PASS / CELLS B AND C POSITIVE
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
CELL_A_TERMINAL: Job 535315 COMPLETED 0:0 in 00:04:26 on n446; source
  a02e0a57669a76e25f28f538eb2c242ce19b5efa; 13/13 GH200 pretests passed;
  fresh B16x2 completed 8/8 measured accepted windows after one warm-up with zero
  invalid/discard/scaler-skip/nonfinite windows. Peak allocated/reserved was
  54,686,040,064 / 65,238,204,416 bytes, 63.9556% of 102,005,473,280 visible
  bytes, with 36,767,268,864 bytes reserved headroom and no monotonic growth.
  Twelve SDPA modules and five scoped compile graphs were active; measured compiler
  counter delta was empty and unexpected steady-state recompile was false. Capacity
  PASS authorizes Cell B but does not promote B16 or establish sustained throughput.
CELL_A_ARTIFACTS: pre-submission a1b99e1e...b51c08; measurement
  3370f607...92e52; result 90c3c026...22c2; complete b07bbc73...935b5;
  stdout/stderr 62449d3b...47b0 / 8db5d05b...1e830
B16_EXTENSION_USAGE_AFTER_CELL_A: base 0.073889 / 1.20, remaining 1.126111;
  code-bug reserve 0.000000 / 0.50, remaining 0.500000; hard remaining 1.626111
CELL_B_TERMINAL: Job 535343 COMPLETED 0:0 in 00:25:32 on n141; source
  441b54093b954f22ca140033f3ffcdd47abc1f58; 13/13 GH200 pretests passed.
  The same-allocation fresh B8->B16 processes each completed 16+256 accepted
  windows with zero invalid/discard/scaler-skip/nonfinite, stable memory, no
  measured recompile and checkpoint hard-gate PASS. B8 measured 22.519035 at
  37,981,519,872 peak reserved bytes; B16 measured 26.720715 presentations/s at
  65,238,204,416 bytes (`63.9556%` visible). B16/B8 ratio and one-sided 95%
  block-bootstrap lower bound were 1.186583 / 1.182178, so the frozen positive
  gate PASS and conditional reverse Cell C is executable. Projected 20-epoch cost
  was 21.730894 versus 18.320298 GH200-hours, saving 3.410596. This remains
  measurement-only evidence and does not promote the B16 recipe.
CELL_B_ARTIFACTS: pre-submission 2f04c27f...00438; B8 measurement/result
  3e8881ca...8e7f5 / cf675b93...9476f; B16 measurement/result
  d23adce1...6bf88 / eeb7c9d4...bc206; pair 9d4c54a8...93162;
  stdout/stderr ecc7a07a...8f59 / 8db5d05b...1e830
B16_EXTENSION_USAGE_AFTER_CELL_B: base 0.499445 / 1.20, remaining 0.700555;
  code-bug reserve 0.000000 / 0.50, remaining 0.500000; hard remaining 1.200555
CELL_C_INITIAL_INCIDENT: Job 536510 FAILED 1:0 in 00:01:13 on n411; source
  12f6ccbd26fae7b2784554c10ec8b7880a7f7b28; 13/13 GH200 pretests passed,
  then the first profiler failed its fail-closed output-path check before output
  directory creation, D_fit loader/model construction or training. The frozen
  output suffix was `b16_c2_b16_first` while attempt ID was `b16_c2_b16`, and the
  path invariant requires the suffix to equal the attempt ID. No scientific or
  capacity result was produced. This is an unambiguous command/provenance-plumbing
  defect eligible for the approved code-bug reserve.
CELL_C_REMEDIATION: retain candidates, order, profiles, source/runtime math, D_fit,
  seed, effective B32, windows, gates and resources; mechanically change only the
  attempt IDs to `b16_c2_b16_first` and `b16_c2_b8_second`, matching the already
  frozen descriptive output suffixes, then submit one serial fresh-output replacement.
CELL_C_INITIAL_ARTIFACTS: pre-submission ca761acb...40892c; stdout/stderr
  1d0456c8...4cf / 4fe08c77...3234; both profiler output directories absent
B16_EXTENSION_USAGE_AFTER_CELL_C_INCIDENT: base 0.499445 / 1.20, remaining
  0.700555; code-bug reserve 0.020278 / 0.50, remaining 0.479722; hard remaining
  1.180277
CELL_C_REPLACEMENT_TERMINAL: Job 536621 COMPLETED 0:0 in 00:24:52 on n421;
  source c48aef0027cd0af06f220b78889d816ead6be773; the single command blocker did
  not recur. The same-allocation fresh B16->B8 processes each completed 16+256
  accepted windows with zero invalid/discard/scaler-skip/nonfinite, stable memory,
  no measured recompile and checkpoint hard-gate PASS. B16 measured 27.031478 at
  65,238,204,416 peak reserved bytes (`63.9556%` visible); B8 measured 23.879129
  presentations/s at 37,981,519,872 bytes. B16/B8 ratio and one-sided 95%
  block-bootstrap lower bound were 1.132013 / 1.128524, so the reverse positive
  gate PASS. Projected 20-epoch cost was 18.110736 versus 20.492912 GH200-hours,
  saving 2.382176. The two-order evidence supports returning B16 to IP-G2 for an
  explicit recipe decision; it does not itself promote B16.
CELL_C_REPLACEMENT_ARTIFACTS: pre-submission 0f24e66f...9d570; B16
  measurement/result 2763a62a...41d73 / defbd731...9f360; B8 measurement/result
  4c6a7fa9...8bcd8 / ada82b93...f218d; pair ab277e8f...5a7fe;
  stdout/stderr c612a1f4...91bae / 8db5d05b...1e830
B16_EXTENSION_FINAL_USAGE: base 0.913889 / 1.20, unused 0.286111 expired;
  code-bug reserve 0.020278 / 0.50, unused 0.479722 expired; total 0.934167 /
  hard 1.70, unused 0.765833 expired. Against the conservative reverse-pair saving
  of 2.382176 GH200-hours/run, this extension breaks even after 0.392148 runs.
EXECUTABLE_NOW: no; Section 9.2 is terminal and only the B16 recipe owner decision
  remains. Production config bytes are unchanged and Envelope B remains frozen.
```

### 9.3 IP-G2 final Camera recipe decision — closed, no compute authority

The owner now promotes physical B16 x accumulation 2 and explicitly accepts the
BatchNorm-statistics and worker-RNG assignment changes relative to B8 x accumulation
4. This closes the decision left open by the immutable Section-9.2 request snapshot;
it does not retrospectively change that envelope or its raw evidence.

```text
OWNER_DECISION: promote Camera B16x2/effective B32
ACCEPTED_RUNTIME: Swin SDPA + five-module forward-only scoped torch.compile +
  fused AdamW; activation checkpoint remains off; recovery cadence remains one epoch
ACCEPTED_RECIPE_CHANGE: B16 BatchNorm statistics and worker-RNG assignment may
  differ from B8; exactness remains required within the B16 recipe/run
IMPLEMENTATION_SHA: 299277e8bdb8f60a05e8f06c2c0706e29252b51c
IMPLEMENTATION_TREE: a541ef7b003388175e6a324ac28ed8f31f3deece
CAMERA_CONFIG_SCHEMA: s10.phase1.v3
CAMERA_CONFIG_FILE_SHA256: 25f53fc554c348c329c7a9cf4b9a5c8d521d993908114fbf64a46f75b3db0bda
CAMERA_RESOLVED_CONFIG_SHA256: f6040d30c23571f049bba3602081a9ec3bbfbdafc5d5ab8b76e9dd375eb76f25
HISTORICAL_PROFILE_BINDING: Section-9.2 profiles retain B4 source file
  567cb1b7...ce60 and resolved e95e65a6...ffe1d; they are not rewritten
LOCAL_VALIDATION: strict v3 resolution, drift rejection, historical-v2 resolved
  hash reconstruction, Python syntax compilation and diff checks PASS
NOT_RUN_LOCALLY: pytest/Torch runtime unavailable on the x86 login environment;
  no GPU/Slurm validation is authorized by this decision
B32_STATUS: not tested or promoted; linear B8/B16 peak extrapolation predicts
  108.392 GB allocated and 119.752 GB reserved, above 102.005 GB visible memory
ENVELOPE_B: frozen and NOT EXECUTABLE; the current capability runner/source-branch
  binding and Section-7 identities remain historical controls
NEXT_REQUIRED_FOR_CAMERA_RUN: new exact source/config/resource projection plus the
  already-required independent recipe-freeze review and explicit owner activation
```

Exact conditional invocations, all through
`fl_v3/scripts/run_s10_phase1p_ip_e2.sh` with the frozen resource tuple, are shown
at their recorded source SHAs; they are historical and not runnable from current
HEAD:

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
  --first-attempt b16_c2_b16_first
  --second-profile ${B8} --second-mode sustained
  --second-output ${ROOT}/camera/sustained_${SHA12}_r2_b16_c2_b8_second
  --second-attempt b16_c2_b8_second
  --pair-output ${ROOT}/pairs/b16_cell02_reverse_${SHA12}_r2.json
  --pair-reference second
  --source-sha ${SOURCE_SHA} --approved-source-sha ${APPROVED_SOURCE} --repeat 2
```

The containing `sbatch` command is fixed at account `naiss2025-22-1113-gpu`,
partition `gpu`, one node/task, `--cpus-per-task=16`, `--mem=96G`,
`--gpus-per-node=nvidia_gh200_120gb:1`, `--time=01:00:00`, `--no-requeue`.
Cells B and C each used one such allocation for both fresh processes and their pair
analysis. The displayed Cell-C attempt IDs are the terminal command-plumbing
correction recorded above. All commands are now terminal and no longer executable.

### 9.4 Camera B16 follow-up IP-E3 — terminal positive

This request implements the owner-frozen first step before any DDP work. It is a
Camera-only, D_fit-only engineering profiler and cannot make capability or recipe-
selection claims. The two prepared candidates are output-neutral in intent but
remain measurement-only; the conservative candidate cannot modify production
defaults. A third implementation is conditionally in scope only under the exact
unlock below.

```text
REQUEST_STATE: TERMINAL / both exact cells completed / no further compute authority
OWNER_APPROVAL: 2026-07-21 — approved containing SHA
  1abe26b3cde2f9f1c26fca130b999d054d6782b1, the >=0.98 conservative unlock,
  base 1.50 + code-bug reserve 0.50 / hard 2.00 charged GH200-hours,
  maximum concurrency one and <=45 minutes per single-GH200 job
APPROVED_SOURCE_SHA: 1abe26b3cde2f9f1c26fca130b999d054d6782b1
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
BRANCH: codex/s10-phase1p-throughput-preflight
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA
PREPARED_IMPLEMENTATION_SHA: 9233af3119857511f5f2acc310a182449e7b91a2
PREPARED_IMPLEMENTATION_TREE: e9e2af82ba2543b01d1428407e81cdc58d131aa3
OBJECTIVE:
  1. localize the largest named Camera-forward stage on the final production B16
     stack using a short structured trace;
  2. only if preprocessing remains largest, pair fresh final-B16 reference and
     conservative batched-affine/grid processes in the same allocation;
  3. only if the conservative screen is near-neutral/positive and every hard gate
     passes, implement and pair one combined batched-rotation grid_sample plus
     non-persistent static-grid candidate under the same protocol.
DATA_ROLE: D_fit only, exact frozen train CBGS identity and seed 0
FORBIDDEN: D_select; D_audit; official validation; mAP/NDS/capability;
  candidate selection/generalization claims; LiDAR; DDP; Envelope B; merge/push
REFERENCE_RECIPE: final Camera physical B16 x accumulation 2/effective B32,
  Swin SDPA + five forward-only scoped compile graphs + fused AdamW, FP16 policy,
  activation checkpoint off, one-epoch checkpoint cadence
PREPARED_CANDIDATES:
  A. exact final-B16 reference
  B. A plus conservative camera_batched_affine_grid only
CONDITIONAL_CANDIDATE_CAP:
  C. A plus one batched rotation grid_sample call and a non-persistent static grid;
     implementation is forbidden unless B passes the unlock gate; no other
     preprocessing/model/precision/normalization/data/loss/optimizer change
DEFAULTS: B and C remain default-off outside their explicit profiler identities
TRACE_PROTOCOL: A; 16 accepted warm-up windows + 3 traced accepted windows;
  structured CPU/CUDA operator and named-stage timing/memory; trace-inflated stage
  totals are localization evidence only, never sustained-throughput estimates
TRACE_CONTINUE_GATE: fl_v3::camera::preprocess is the largest named Camera-forward
  range; otherwise record TRACE_STOP and end IP-E3 before the paired screen
PAIR_PROTOCOL: fresh processes, 16 accepted warm-up + 256 accepted measured windows
  each; same allocation/node/GPU/source/config/CBGS/input anchor; reference first;
  16-window blocks and 50,000-draw one-sided block bootstrap
HARD_GATES: both measurement-health gates; exact same-B16 first-window input/RNG
  anchor; exact boundary, discrete, optimizer-step and checkpoint-context state;
  both fresh-process checkpoint-continuation gates; zero invalid/discard/scaler-
  skip/nonfinite windows; stable memory <=85% visible; no unexpected recompile
NUMERICAL_POLICY: grouped parameters/BN mean/BN variance/Adam exp_avg/exp_avg_sq
  relative-L2 and max-absolute gates remain max(frozen tolerance, 1.25x same-process
  repeat control); elementwise allclose remains diagnostic for non-deterministic
  kernels; any clear bug or scientific drift is a hard stop
CONSERVATIVE_UNLOCK: every hard gate PASS and candidate/reference one-sided 95%
  throughput-ratio lower bound >=0.98. This authorizes local implementation of C
  and its one additional same-allocation A-versus-C pair only; it promotes nothing
CONSERVATIVE_STOP: lower bound <0.98 or any hard-gate failure ends preprocessing
  work; do not implement C
CONDITIONAL_PROMOTION: none; any B/C promotion returns to the owner with throughput,
  uncertainty, memory, checkpoint and projected-GH200-hour evidence
INITIAL_EXACT_CELL: one allocation runs A trace, then conditionally A followed by B
CONDITIONAL_EXACT_CELL: only after unlock, one new allocation runs fresh A followed
  by C with the identical sustained protocol and newly recorded exact source/profile
  identities; it is within the candidate cap but cannot be submitted before those
  immutable identities and the command are appended to this ledger
BASE_AGGREGATE_GPU_HOURS: 1.50 charged GH200-hours
CODE_BUG_REMEDIATION_RESERVE: +0.50 charged GH200-hour, code-level bug only
HARD_AGGREGATE_GPU_HOURS: 2.00 charged GH200-hours
RESOURCE_RATIONALE: at most two scientific allocations at <=0.75 hour each: the
  prepared trace+pair allocation and the conditional A-versus-C allocation; the
  separate reserve is only for diagnosed frozen-semantics engineering replacement
MAX_CONCURRENCY: 1
PER_JOB_RESOURCE: 1 node / 1 GH200 / 16 CPUs / 96 GiB / <=00:45:00 / no requeue
SUBMISSION_POLICY: serial; no numeric remediation-submission cap; no blind retry;
  aggregate layered ceiling and concurrency are binding
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_e3_<APPROVED_SOURCE_SHA12>/
FRESH_OUTPUT: every trace/process/pair/attempt path absent before execution; no
  overwrite; raw evidence immutable
ENGINEERING_REMEDIATION: O-149 unambiguous frozen-semantics source/test/runner/API/
  checkpoint/logging repair only, serially, inside the reserve; no candidate,
  math, data, precision, normalization, recipe, gate or resource change
STOP_ESCALATE: layered ceiling exhaustion; repeated/ambiguous blocker; trace
  contradiction; any hard-gate failure; lower bound below 0.98; OOM or >85% visible;
  science-boundary pressure; candidate/resource/gate change
SOURCE_RULE: each execution SOURCE_SHA must be a clean linear descendant of
  APPROVED_SOURCE_SHA;
  no merge; exact source/profile/command/output/budget is recorded before sbatch
CAMERA_CONFIG_FILE_SHA256: 25f53fc554c348c329c7a9cf4b9a5c8d521d993908114fbf64a46f75b3db0bda
CAMERA_RESOLVED_CONFIG_SHA256: f6040d30c23571f049bba3602081a9ec3bbfbdafc5d5ab8b76e9dd375eb76f25
LIDAR_TRACE_FROZEN_FILE_SHA256: c7e1fa26e1714a31c5998296cb95cbab5e8732d4bf2f06da81fd6d631c574bfc
LIDAR_TRACE_FROZEN_RESOLVED_SHA256: 0efe4d6d5138e3d99ae80254a6ecf884300dd18985ab45a00425228fc3ef082e
REFERENCE_PROFILE_FILE_SHA256: 0b655b3108680c806257c71b2df4e6f9147a63583ea7883f87fc0436b0924b4b
REFERENCE_PROFILE_CANONICAL_SHA256: 5ab0d49133248977a15acb049b98227baa21dcbd83dbf6e1a584d152922d0a5d
CONSERVATIVE_PROFILE_FILE_SHA256: bb2d423e46a84df7f3aca0b085995607f6d53c32b6ca1671d1ea23cc564e672b
CONSERVATIVE_PROFILE_CANONICAL_SHA256: cdc8aaab138940fcaa05cb9390909565174177c34befe6c3810420d8f0537d4a
LAUNCHER_SHA256: 1b22bc7d030bdb58f73b63d5a9f02d8a7d601c66e9e64617d5c0e396c30c1a53
PAIR_ANALYZER_SHA256: 189562288d95ce6d94add8e3c580e3cbf1c2c5bd4fa6077b796535725af1d521
PROFILER_SHA256: 637f00b92369ecf90bf2ef7cbb06db1aeb8a9ada8cc38dcf24a46ac3f6608ab0
LOCAL_VALIDATION: JSON/profile validation, shell syntax/shellcheck, Python syntax,
  comparison fixture and focused pure-Python gate checks PASS
NOT_RUN_LOCALLY: pytest/Torch runtime unavailable on x86 login; the launcher keeps
  six focused GH200 tests as hard pre-model gates
```

The exact initial command after explicit owner activation of the containing clean
SHA is:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=<owner-approved-containing-SHA>
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  fl_v3/scripts/run_s10_phase1p_ip_e3.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e3_camera_b16_reference.json \
  --candidate-profile \
    fl_v3/configs/s10_phase1p_ip_e3_camera_b16_batched_affine_grid.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

Initial Job `539364` is terminal positive and condition C is now prepared:

```text
INITIAL_JOB: 539364 / source b44b5d4c03a258a54a9c25eb8cbd2569a4472211 /
  tree 074a47637502b256c927db0d4bdb91b8dd30e3c0 / n89
INITIAL_TERMINAL: COMPLETED 0:0 / 00:28:26 / 0.473889 charged GH200-hour
PRETESTS: 9 passed; no pre-model incident
TRACE: COMPLETE_TRACE; every required named range present; preprocess was the
  largest Camera-forward range at 0.723759 of the named-range CPU sum; this is
  localization evidence, not a sustained wall-time fraction
REFERENCE: 25.400589 presentations/s; 256/256 measured accepted; checkpoint PASS;
  zero invalid/discard/scaler-skip/nonfinite; peak allocated/reserved
  54,663,526,912 / 65,238,204,416 bytes; no growth/recompile
CONSERVATIVE: 27.868228 presentations/s; 256/256 measured accepted; checkpoint
  PASS; zero invalid/discard/scaler-skip/nonfinite; peak allocated/reserved
  54,668,524,032 / 65,315,799,040 bytes; no growth/recompile
PAIR: same allocation/node/GPU/source/config/CBGS and exact B16 input anchor;
  candidate/reference ratio 1.097149; one-sided 95% lower bound 1.087948;
  every hard gate PASS; POSITIVE_SCREEN; conditional implementation eligible
PROJECTED_20_EPOCH: 19.269164 reference versus 17.566398 candidate GH200-hours;
  diagnostic saving 1.702766; no automatic promotion
INITIAL_ARTIFACTS: pre-submission e3aa62a1...f17c84; structured trace
  26253a19...4ac60; trace result b4f4c3e6...2e188; reference result
  a16732c1...19b4c; conservative result f7010ba1...f5055; pair
  0fdb037d...87ec; stdout/stderr f5770c06...2b104 / b7e8519a...1aa3
BUDGET_AFTER_INITIAL: base used 0.473889 / 1.50, remaining 1.026111;
  code-bug reserve used 0.000000 / 0.50, remaining 0.500000;
  hard remaining 1.526111 charged GH200-hours
CONDITIONAL_IMPLEMENTATION_SHA: 417dfefb8b37551bdd284fa30f0ef575b4a075e8
CONDITIONAL_IMPLEMENTATION_TREE: e4077302740907d31421bc0cbb1a6126364551e7
CONDITIONAL_SCOPE: retain resize/crop/flip and per-image affine math; reuse one
  non-persistent static output-coordinate grid, batch source-grid construction,
  and issue one rotation grid_sample over all rotated images; default-off
CONDITIONAL_PROFILE_FILE_SHA256: 08102445b725d0920560d94b5cd6155257170a73be9064c95db9f6dad78df40d
CONDITIONAL_PROFILE_CANONICAL_SHA256: 8313dbb4cacceec34d6fae29fc8bfd7766c67526527b4d46d9a46e43fdb2527a
CONDITIONAL_LAUNCHER_SHA256: 5ff6014085058547d94e2e19eabedc4bc593456bd9a2d208552aa33111e99e3f
CONDITIONAL_ANALYZER_SHA256: 2f2a082884d0479d2a37c38fa637e9886b59d078b8815805750a96a2b452ac07
CONDITIONAL_PROFILER_SHA256: 675cdd524e5ed400ee9a0252b4c00ad35ceb388f02300f6da1d67e7fc7751dde
CONDITIONAL_PREPROCESS_SHA256: f894206c8fbdc6b23b196e3b4ead5d9e38104d088526f798817952a11a164be2
CONDITIONAL_LOCAL_VALIDATION: shell syntax/shellcheck, Python syntax, JSON/profile
  validation, conditional-unlock replay and diff checks PASS; Torch unavailable on
  login, so exact CPU/CUDA forward-policy test remains a fail-closed GH200 pretest
CONDITIONAL_STATE_AT_SUBMISSION: exact one-pair cell executable; no result promotes
  the candidate
```

The exact conditional command from a clean containing source commit is:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=1abe26b3cde2f9f1c26fca130b999d054d6782b1
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e3_1abe26b3cde2
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --output="${ROOT}/slurm_conditional_%j.out" \
  --error="${ROOT}/slurm_conditional_%j.err" \
  fl_v3/scripts/run_s10_phase1p_ip_e3_conditional.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e3_camera_b16_reference.json \
  --candidate-profile \
    fl_v3/configs/s10_phase1p_ip_e3_camera_b16_batched_rotation_grid_sample.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

The conditional cell is terminal and closes IP-E3 compute:

```text
CONDITIONAL_JOB: 539853 / source e855320b59f8d616b6a5c44e7949850e46362184 /
  tree 9bd0d59a2736c990ff63b8b50059d79b13fbd86d / n469
CONDITIONAL_TERMINAL: COMPLETED 0:0 / 00:25:21 / 0.422500 charged GH200-hour
PRETESTS: 8 passed, including GH200 CPU/CUDA output-policy and one-call grid_sample;
  no pre-model incident
REFERENCE: 26.340313 presentations/s; 256/256 measured accepted; checkpoint PASS;
  zero invalid/discard/scaler-skip/nonfinite; peak allocated/reserved
  54,663,514,624 / 65,238,204,416 bytes; no growth/recompile
COMBINED_CANDIDATE: 27.537525 presentations/s; 256/256 measured accepted;
  checkpoint PASS; zero invalid/discard/scaler-skip/nonfinite; peak allocated/
  reserved 54,671,440,384 / 65,307,410,432 bytes (`64.0234%` visible);
  no growth/recompile
CHECKPOINT_NUMERICS: exact boundary/input/RNG/training/discrete/structure gates PASS;
  model-parameter, BN mean/variance and Adam exp_avg/exp_avg_sq grouped diagnostics
  all within max(frozen tolerance, 1.25x same-process repeat-control); elementwise
  allclose remains false and diagnostic-only under the accepted policy
PAIR: same allocation/node/GPU/source/config/CBGS and exact B16 input anchor;
  candidate/reference ratio 1.045452; one-sided 95% lower bound 1.025241;
  every hard gate PASS; POSITIVE_SCREEN; no automatic promotion
PROJECTED_20_EPOCH: 18.585048 reference versus 17.777780 candidate GH200-hours;
  diagnostic saving 0.807268
CONDITIONAL_ARTIFACTS: pre-submission 9aa68b99...aa25e5; reference result
  86c3fb8a...5164f; candidate result ab8975f0...2fd4c; pair
  245cbaba...073dc; stdout/stderr bb024f10...fdea8 / 8db5d05b...1e830
IP_E3_FINAL_USAGE: base 0.896389 / 1.50, unused 0.603611 expired;
  code-bug reserve 0.000000 / 0.50, unused 0.500000 expired;
  total 0.896389 / hard 2.00, unused 1.103611 expired
CROSS_ALLOCATION_DIAGNOSTIC: conservative-effect / combined-effect ratio 1.049450;
  10,000-draw one-sided 95% lower bound 1.032706 and two-sided interval
  [1.030356, 1.076059]. This is advisory only because allocations/nodes differ;
  it is not a new promotion gate
SYNTHESIS: both items are positive screens; conservative affine/grid is simpler,
  elementwise-exact in its focused test and showed the larger matched speedup
  (1.097149 / lower 1.087948 versus 1.045452 / lower 1.025241)
OWNER_RECIPE_DECISION: both implementations are accepted as qualified output-neutral
  paths. Conservative batched affine/grid is the production default; the combined
  static-grid plus batched-rotation path is retained as a qualified optional path
  but is not selected by the production recipe. The combined path already contains
  conservative batched affine/grid, so their measured gains are not additive
EXECUTABLE_NOW: no; final production source/config materialization and a separately
  frozen follow-up or 2-GH200 DDP implementation/resource envelope are next
```

The subsequent 2-GH200 work is intentionally not part of IP-E3. After IP-E3 is
terminal, its separate design must bind 1 GPU B16x2 against 2 GPUs B16/rank x1 in
one same-node allocation, effective global B32, and exact DDP-expanded-CBGS union
with no DDP-induced omission or duplication. Its performance gate is a one-sided
95% aggregate-throughput speed-ratio lower bound `>=1.60`. The later final
`13.285290 h` single-GPU projection makes the descriptive wall bound `<=8.303306 h`;
the fresh same-allocation reference remains binding and two-GPU charged cost must
be `<=1.25x` that reference. Model
parameters, optimizer/scheduler/scaler state and accepted/skipped counters must be
rank-consistent; BN running-buffer semantics must be explicitly frozen rather than
silently excluded. Checkpoint/resume and rank-state gates are mandatory. IP-E3
itself authorized no DDP source, resource, execution, promotion or 4-GPU cell;
the later separately designed source/request is Section 9.6.

### 9.5 Camera preprocessing IP-E4 — closed positive

The owner accepts conservative batched affine/grid as the Camera production
default and retains the combined static-grid/batched-rotation implementation as a
qualified optional path. IP-E4 is a narrow output-neutral follow-up before DDP so
that every later rank can consume any accepted per-rank gain.

```text
REQUEST_STATE: CLOSED POSITIVE / compute authority consumed
OWNER_APPROVAL_ANCHOR: 7d4bb6efdbb7b8fb61ee72243c72a5ec3ef7d451
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
BRANCH: codex/s10-phase1p-throughput-preflight
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA
OBJECTIVE: reduce the final B16 Camera preprocessing host/launch fragmentation
  without changing augmentation, image/calibration math or training semantics;
  qualify accepted changes before any 2-GH200 DDP comparison
DATA_ROLE: D_fit only; frozen exact CBGS identity/order, seed 0 and B16xaccum2/B32
REFERENCE: SDPA + scoped compile + fused AdamW + accepted conservative
  camera_batched_affine_grid; checkpoint cadence remains one epoch
INITIAL_CANDIDATE: reference plus batched construction/composition/inversion of the
  same per-image 3x3 geometry; per-image resize/crop/pad/flip and interpolation
  calls remain unchanged
CONDITIONAL_CANDIDATE: accepted initial candidate plus one bulk native-image
  uint8-to-float32 conversion; maximum added live tensor is the exact
  [96,3,900,1600] float32 batch (1,658,880,000 bytes)
CANDIDATE_CAP: reference plus the two ordered candidates above; no combination or
  extra implementation outside that chain
DEFAULTS: new candidates default-off outside their explicit IP-E4 profiles until
  the frozen >=1.02 promotion rule accepts them
TRACE: first allocation traces the final reference after 16 accepted warm-up and
  3 accepted active windows; required subranges cover parameter preparation,
  conversion/resize, crop/flip, geometry, rotation grid/sample, stack/normalize
  and calibration update
PAIR: within one allocation, fresh reference then candidate; 16 accepted warm-up
  plus 256 accepted measured windows each; exact source/config/CBGS/input anchor;
  16-window blocks and 50,000-draw one-sided bootstrap
PROMOTION_GATE: every hard gate PASS and candidate/reference one-sided 95%
  throughput-ratio lower bound >=1.02. The initial result then both accepts that
  output-neutral item and unlocks conditional implementation/execution; a lower
  positive result is retained but not selected
HARD_GATES: exact boundary/input/RNG/discrete/training state; accepted grouped
  continuation policy for parameters, BN mean/variance and Adam moments; both
  checkpoint/resume gates; zero invalid/discard/scaler-skip/nonfinite windows;
  <=85% visible reserved memory; no growth/recompile; same interpolation modes,
  sampled augmentation values/order and calibration formula
FORBIDDEN: fused single-resample image math; changed resize/crop/flip/rotation or
  border semantics; D_select; D_audit; official validation; capability metrics;
  LiDAR; DDP; Envelope B; merge/push/publication
BASE_AGGREGATE_GPU_HOURS: 1.00 charged GH200-hour
CODE_BUG_REMEDIATION_RESERVE: +0.50 charged GH200-hour, code-level bug only
HARD_AGGREGATE_GPU_HOURS: 1.50 charged GH200-hours
MAX_CONCURRENCY: 1
PER_JOB_RESOURCE: 1 node / 1 GH200 / 16 CPUs / 96 GiB / <=00:45:00 / no requeue
SUBMISSION_POLICY: serial; no numeric bug-remediation cap; no blind identical retry;
  layered aggregate ceiling and concurrency are binding
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_e4_7d4bb6efdbb7/
FRESH_OUTPUT: every trace/process/pair path absent before execution; raw outputs
  immutable and no overwrite
REMEDIATION: O-149 frozen-semantics test/runner/API/config/checkpoint/logging repairs
  only; source/command/output provenance appended before a derived submission
STOP_ESCALATE: hard/base+reserve ceiling; repeated/ambiguous blocker; hard-gate
  failure; initial lower bound <1.02 ends the conditional chain; OOM/>85%; changed
  math/data/RNG/recipe/gate/resources; any DDP or scientific-boundary pressure
SOURCE_RULE: clean linear descendants of OWNER_APPROVAL_ANCHOR and UNIQUE_BASE_SHA;
  exact source/profile/command hashes are appended before each submission
```

The initial output-neutral candidate and its fail-closed trace/pair runner are now
sealed. The containing pre-submission commit changes only this ledger relative to
the prepared implementation source; the launch command resolves that clean
containing commit as the literal execution `SOURCE_SHA`.

```text
PREPARED_IMPLEMENTATION_SHA: b909d4ee7e02375e230f2d44b193aae1d0af399b
PREPARED_IMPLEMENTATION_TREE: b3aa65547b300d8dde4c36cb1a8bb5c38887eb72
REFERENCE_PROFILE_FILE_SHA256: 675239186a53c6467fc1e9e8490e978deaf8eb3f2bf7987ee260c241490143a8
REFERENCE_PROFILE_CANONICAL_SHA256: e963b7720c778685a9de429c05a38c25636f63dc23dd88f1339a1b3756e75f6c
VECTORIZED_PROFILE_FILE_SHA256: 29d78cdfdae1cec73bb8b59d131c64282f68bb605572c5f19c63bc119b488aa6
VECTORIZED_PROFILE_CANONICAL_SHA256: 2bf1b5bd68dbcb3e89724cb857c098531688488e479c4673e7a06f7b7a871a63
LAUNCHER_SHA256: afbec29160857c33ad4509789cd43f269a1f42c3017e73dc38f9b2901a5df34b
PAIR_ANALYZER_SHA256: 79721e8401b5bbb59c5ed27071165bada5310f7f53c12f8b06b84717c2de5b2f
PROFILER_SHA256: 3c91704d3f7c2d8fe0add8a63b76bf08a7fa8e96bb814692362faf8eb9ed9611
PREPROCESS_SHA256: 0e0fedb70f3e6b4d7f3cc8ad4beaae9c6851c3bc60ec582cc29a41095ef519f9
LOCAL_VALIDATION: Python syntax, JSON syntax, strict standalone profile parsing,
  config file/resolved identities, shell syntax/shellcheck and diff checks PASS
NOT_RUN_LOCALLY: Torch/PyTorch is absent on the x86 login node; five focused
  CPU/GH200-CUDA profile/config/numerical/gate tests run before any D_fit/model
  measurement and fail closed
INITIAL_STATE_AT_SUBMISSION: one trace plus one same-allocation fresh reference ->
  vectorized-geometry pair; bulk conversion remains locked and unimplemented
```

The exact initial invocation from the containing clean source commit is:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=7d4bb6efdbb7b8fb61ee72243c72a5ec3ef7d451
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e4_7d4bb6efdbb7
mkdir -p "${ROOT}"
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --output="${ROOT}/slurm_initial_%j.out" \
  --error="${ROOT}/slurm_initial_%j.err" \
  fl_v3/scripts/run_s10_phase1p_ip_e4.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_reference.json \
  --candidate-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_vectorized_geometry.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

The initial submission exposed one pre-model command/provenance defect and is
terminal. The replacement below is a strictly derived O-149 repair: it binds each
output suffix to the same variable passed as `--attempt-id`; no profile, candidate,
data, math, test, gate, window or resource changes.

```text
INITIAL_JOB: 541217 / source 9db1ce96ebc9488ebd8c1523962bebd4029cc6cf /
  tree d754e7be48ed1b5fc0511c223fc323efe6b97fe4 / n447
INITIAL_TERMINAL: FAILED 1:0 / 00:00:59 / 0.016389 charged GH200-hour
INITIAL_PRETESTS: 9 passed in 3.73 seconds, including CPU/GH200-CUDA output policy;
  no D_fit loader, model, trace, checkpoint or measurement output was created
INCIDENT: trace attempt-id was `b16_reference_trace`, while the frozen output
  suffix was `b16_reference`; the profiler rejected this exact path mismatch
REMEDIATION: define the three attempt IDs once and derive each trace/sustained
  output suffix from the identical variable; all scientific and measurement
  inputs remain byte-identical
REPLACEMENT_LAUNCHER_SHA256: 938634e65ab36051e46cd8f901755d740f774184c857433ce71bc59183e5ac76
INITIAL_STDOUT_SHA256: 217e4a55556d0747912538778087c45dee9c6cebcbcad41226b81267c2bd2ec9
INITIAL_STDERR_SHA256: b03d595a8882a0ad9b43c874dd9aafb116ad21e6edf915a8aa945f11bfd8a67c
BUDGET_AFTER_INITIAL_INCIDENT: base used 0.000000 / 1.00; code-bug reserve used
  0.016389 / 0.50; hard total used 0.016389 / 1.50
REPLACEMENT_OUTPUTS: source-SHA-qualified fresh trace/reference/candidate paths;
  fixed pair path remains absent; Slurm logs use fresh `slurm_replacement_%j`
```

The exact serial replacement invocation from this containing clean source commit
is:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=7d4bb6efdbb7b8fb61ee72243c72a5ec3ef7d451
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e4_7d4bb6efdbb7
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --output="${ROOT}/slurm_replacement_%j.out" \
  --error="${ROOT}/slurm_replacement_%j.err" \
  fl_v3/scripts/run_s10_phase1p_ip_e4.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_reference.json \
  --candidate-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_vectorized_geometry.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

The derived replacement is terminal positive. It promotes only the scoped
vectorized geometry item under the already frozen rule and unlocks, but does not
yet execute, the one conditional bulk-conversion candidate.

```text
REPLACEMENT_JOB: 541221 / source 69272b01509335813117a482ff15543186b84e67 /
  tree 028a188c22b0289ba22c01a667a14ad425a7395d / n416
REPLACEMENT_TERMINAL: COMPLETED 0:0 / 00:26:50 / 0.447222 base GH200-hour
PRETESTS: 9 passed in 3.35 seconds; initial path defect did not recur
TRACE: COMPLETE_TRACE; every core and preprocessing subrange present; preprocess
  was the largest named Camera-forward range at 0.681777; geometry was the largest
  preprocessing subrange at 1,655,485.456 CPU-inclusive us over 6 active windows
REFERENCE: 26.934770 presentations/s; 256/256 accepted; zero invalid/discard/
  scaler-skip; checkpoint continuation PASS; peak allocated/reserved
  54,668,524,032 / 65,315,799,040 bytes (64.0317% visible); no growth/recompile
VECTORIZED_GEOMETRY: 37.482862 presentations/s; 256/256 accepted; zero invalid/
  discard/scaler-skip; checkpoint continuation PASS; peak allocated/reserved
  54,666,238,464 / 75,524,734,976 bytes (74.0399% visible); no growth/recompile
PAIR: same allocation/node/GPU/source/config/CBGS and exact B16 input anchor;
  candidate/reference ratio 1.391616; one-sided 95% lower bound 1.379987; every
  hard gate PASS; PROMOTE_AND_UNLOCK_BULK_CONVERSION
PROJECTED_20_EPOCH: 18.174655 reference versus 13.072462 vectorized-geometry
  GH200-hours; diagnostic saving 5.102194 per Camera run
PROMOTED_ITEM: batch only the existing float64 3x3 construction, left-associated
  composition and two inversion stages; resize/crop/pad/flip/interpolation and
  augmentation/calibration semantics unchanged
CONDITIONAL_UNLOCK: one candidate adding exactly one bulk native-image
  uint8-to-float32 conversion is now eligible for implementation and one fresh
  same-allocation vectorized-reference -> bulk-candidate pair
ARTIFACTS: trace result c70b03e7...9597948; trace summary 0ae1417b...fbd98;
  reference result 41db16a5...c490b; candidate result 52cf019f...7215c;
  pair 6cc4f70f...eb8ef; stdout 72052da2...5c7e; stderr b7e8519a...1aa3
BUDGET_AFTER_REPLACEMENT: base used 0.447222 / 1.00, remaining 0.552778;
  code-bug reserve used 0.016389 / 0.50, remaining 0.483611; hard total used
  0.463611 / 1.50, remaining 1.036389 charged GH200-hours
```

The unlocked conditional implementation and exact second pair are now sealed.
The candidate allocates one flattened `[96,3,900,1600]` float32 tensor, performs
the existing `/255` pointwise operation in place, and then uses the unchanged
per-image interpolation/geometry path. A `00:33:00` job limit is deliberately
stricter than the approved 45-minute per-job maximum so even a full time-limit
charge cannot exceed the remaining `0.552778` base budget.

```text
CONDITIONAL_IMPLEMENTATION_SHA: d732be28688df974fee14b5d7abc9bd00c4a07f6
CONDITIONAL_IMPLEMENTATION_TREE: 1d5685701058c152dabe0a48cb25e787843268ae
VECTORIZED_REFERENCE_PROFILE_FILE_SHA256: 29d78cdfdae1cec73bb8b59d131c64282f68bb605572c5f19c63bc119b488aa6
VECTORIZED_REFERENCE_PROFILE_CANONICAL_SHA256: 2bf1b5bd68dbcb3e89724cb857c098531688488e479c4673e7a06f7b7a871a63
BULK_PROFILE_FILE_SHA256: 9c6f6e165efb8962aba6c4ef3f996a2fa5e0529ecabf02fe20b71d3d8753ebfe
BULK_PROFILE_CANONICAL_SHA256: 2179f65fa28a4d5a6756aa62c939ac47a6db552829b395a7b8389e6fa2e811bf
CONDITIONAL_LAUNCHER_SHA256: 9ddb41b1c11e12896e90eeafc868e9a2d5aeb954e64f2a3519cf271603740085
CONDITIONAL_ANALYZER_SHA256: 0351e681b136caed66aa85727e0184fa7d54c18abd38ef84d7ac64b6b209f90c
CONDITIONAL_PROFILER_SHA256: fca4c52db0685814716c2c8de8f6acdd717f4465b8e544342ac449c898027575
CONDITIONAL_PREPROCESS_SHA256: e4e5e65af35efd6a5f5a5214eb6bc9d4d522fd787f202133addd6fafd9e9c8a3
UNLOCK_PAIR_SHA256: 6cc4f70f7691633b7e34ed1bc358ac40dc938dfedc1b61d32dce1ce1416eb8ef
CONDITIONAL_SCOPE: fresh vectorized-geometry reference followed by fresh bulk-
  conversion candidate in one allocation; repeat identity 2; 16+256 accepted
  windows and checkpoint continuation for each; same frozen >=1.02 gate
MAX_ADDED_LIVE_TENSOR: 1,658,880,000 bytes; no second full-size division tensor
LOCAL_VALIDATION: Python syntax, JSON syntax, strict standalone three-profile
  mapping/config identities, shell syntax/shellcheck and diff checks PASS
NOT_RUN_LOCALLY: Torch absent on x86 login; focused CPU/GH200-CUDA exact-output,
  one-conversion, config and comparison tests run before D_fit/model measurement
CONDITIONAL_STATE_AT_SUBMISSION: exact one-pair cell executable; no DDP, LiDAR,
  evaluation role or Envelope-B authority
```

The exact conditional invocation from the containing clean source commit is:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=7d4bb6efdbb7b8fb61ee72243c72a5ec3ef7d451
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e4_7d4bb6efdbb7
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:33:00 --no-requeue \
  --output="${ROOT}/slurm_bulk_%j.out" \
  --error="${ROOT}/slurm_bulk_%j.err" \
  fl_v3/scripts/run_s10_phase1p_ip_e4_bulk.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_vectorized_geometry.json \
  --candidate-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_bulk_input_conversion.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

The first conditional submission is a terminal pre-model engineering incident.
Both approved candidates and every resource/measurement setting remain frozen;
the derived replacement only shortens the two attempt labels and validates their
CLI grammar before invoking the profiler.

```text
CONDITIONAL_INITIAL_JOB: 541688 / source da2b559b64ec62b97df8b00a7da030d24a9e61a7 /
  tree b36ffbe6fc04b461970dae42c3379bb830494b61 / n77
CONDITIONAL_INITIAL_TERMINAL: FAILED 2:0 / 00:00:47 / 0.013056 charged
  GH200-hour from code-bug reserve
CONDITIONAL_INITIAL_PRETESTS: 8 passed in 3.57 seconds, including CPU/GH200-CUDA
  elementwise output equality and exactly one complete-native-batch conversion
CONDITIONAL_INCIDENT: descriptive attempt IDs exceeded the profiler's frozen
  32-character maximum; argparse stopped before output-directory creation,
  D_fit loader/model construction, checkpointing or measurement
CONDITIONAL_REMEDIATION: use `b16_vecgeom_ref` and `b16_vecgeom_bulk`; derive
  fresh output suffixes from those exact values and prevalidate their grammar
CONDITIONAL_REPLACEMENT_LAUNCHER_SHA256: c7538317115b51fa3415cd1bf19c5689ff27baa0b6b39d15791eb8290da3af66
CONDITIONAL_INITIAL_STDOUT_SHA256: 90ccdedfd17d04a0e3b2940ee1244ba4ae90caa1c1186c7c1a4f65c30c079633
CONDITIONAL_INITIAL_STDERR_SHA256: e1623530aa4792bb169127c3f98a6f59694f82fe86c7bdd40f141874b29a23b0
BUDGET_AFTER_CONDITIONAL_INCIDENT: base used 0.447222 / 1.00, remaining
  0.552778; code-bug reserve used 0.029444 / 0.50, remaining 0.470556;
  hard total used 0.476667 / 1.50, remaining 1.023333 charged GH200-hours
```

The exact derived replacement keeps the 33-minute base-budget cap:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=7d4bb6efdbb7b8fb61ee72243c72a5ec3ef7d451
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e4_7d4bb6efdbb7
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:33:00 --no-requeue \
  --output="${ROOT}/slurm_bulk_replacement_%j.out" \
  --error="${ROOT}/slurm_bulk_replacement_%j.err" \
  fl_v3/scripts/run_s10_phase1p_ip_e4_bulk.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_vectorized_geometry.json \
  --candidate-profile \
    fl_v3/configs/s10_phase1p_ip_e4_camera_b16_bulk_input_conversion.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

The derived conditional replacement is terminal positive. It closes IP-E4 and
promotes bulk conversion under the pre-frozen automatic rule; no additional
compute follows from this result.

```text
CONDITIONAL_REPLACEMENT_JOB: 541821 / source
  48fa78a60b3308c407fbc16b64dde188216f87e4 / tree
  4ed528542a443e7790ddbc06f68da1824b42606d / n77
CONDITIONAL_REPLACEMENT_TERMINAL: COMPLETED 0:0 / 00:20:08 /
  0.335556 base GH200-hour
PRETESTS: 8 passed, including CPU/GH200-CUDA elementwise output equality and
  exactly one complete-native-batch conversion; shortened-ID defect did not recur
REFERENCE: vectorized geometry at 36.018676 presentations/s; 256/256 accepted;
  zero invalid/discard/scaler-skip; checkpoint continuation PASS; peak allocated/
  reserved 54,666,238,464 / 75,524,734,976 bytes (74.0399% visible); no growth
BULK_CONVERSION: 36.875959 presentations/s; 256/256 accepted; zero invalid/
  discard/scaler-skip; checkpoint continuation PASS; peak allocated/reserved
  54,666,490,368 / 75,522,637,824 bytes (74.0378% visible); no growth
CONTINUATION_POLICY: boundary/input/RNG/discrete/training state and accepted
  grouped numerical gate PASS; per-element allclose remains a recorded non-hard
  diagnostic under the owner-amended non-deterministic-kernel policy
PAIR: exact same allocation/node/GPU/source/config/CBGS and B16 input anchor;
  candidate/reference ratio 1.023801; one-sided 95% lower bound 1.022026;
  every hard gate PASS; PROMOTE_BULK_INPUT_CONVERSION
PROJECTED_20_EPOCH: 13.603467 vectorized reference versus 13.285290 bulk
  GH200-hours; incremental diagnostic saving 0.318177 per Camera run
RESULT_SHA256: reference 0dc4c0fddbc09ba3808d2fdec4e502fb87bdca983c1fdf988de0b70793d1c660;
  candidate 0fe97ec71bd0f6e658696a529ea8357c0efc5bb3da4e8eee5f0fadd388d457bb
PAIR_SHA256: 69f9209b31b23ab1218eba930216c202a6c64425d30e606e6543284851b20d95
SLURM_STDOUT_SHA256: e09dac70ca2913939a31f8aa6593b070d70d1a4b59fd2b2eac37d29b6eff394b
SLURM_STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
FINAL_BUDGET: base 0.782778 / 1.00; code-bug reserve 0.029444 / 0.50;
  hard total 0.812222 / 1.50 charged GH200-hours
UNUSED_AUTHORITY: base 0.217222, bug reserve 0.470556 and hard total 0.687778
  expire at IP-E4 closure; they cannot be transferred to DDP or Envelope B
```

The final production binding is local/source-only and consumed no further GPU:

```text
PRODUCTION_FREEZE_SHA: 93cac472916e1c9c69c8910ad7034f11846e8cec
PRODUCTION_STACK: B16xaccum2; conservative batched affine/grid; vectorized
  geometry/inverses; bulk native-image conversion; SDPA; scoped forward compile;
  fused AdamW; one-epoch recovery-checkpoint cadence unchanged
CAMERA_CONFIG_FILE_SHA256:
  2e5368f96a6198e9a3b1bd43b258b53675df49f5c6ca9042fa8f72e0084c3b6a
CAMERA_RESOLVED_CONFIG_SHA256:
  0df1a19c057312923e0a8e48e81689d9ca265cc613c6f34d4795417414aa0bcf
HISTORICAL_IP_G2_RESOLVED_CONFIG_SHA256:
  f6040d30c23571f049bba3602081a9ec3bbfbdafc5d5ab8b76e9dd375eb76f25
LOCAL_VALIDATION: JSON syntax, current/historical resolved identities,
  fail-closed preprocess binding, Python byte-compilation and diff checks PASS
LOCAL_TEST_LIMIT: pytest and Torch are absent from the x86 login environment;
  the candidate paths themselves passed the eight in-job CPU/GH200-CUDA pretests
INTERPRETATION: D_fit throughput/engineering health only; no capability, mAP/NDS,
  generalization, D_select, D_audit, official validation or scientific selection
NEXT_BOUNDARY: owner accepts or rejects production promotion of the exact DDP
  BN/worker-RNG recipe; no automatic promotion or further IP-E5 compute
```

### 9.6 Camera same-node 2-GH200 DDP IP-E5 — terminal positive

The owner authorizes materializing the already selected final Camera recipe and
designing exact DDP source/tests/resources. The owner then activated this concrete
request at its containing clean request commit and full frozen resource ceiling.

```text
REQUEST_STATE: CLOSED / POSITIVE_DDP_QUALIFICATION / NOT EXECUTABLE
OWNER_ACTIVATION_REQUEST_SHA: 2505db02920021663ccce7783dee483f10e638f8
DESIGN_APPROVAL_ANCHOR: e61c486757ca5fe89340c9325014f4c3e048da2b
PREPARED_IMPLEMENTATION_SHA: e51df6efa04e6d151315c72b7d7016014852078c
PREPARED_IMPLEMENTATION_TREE: 603ecd1784fcb093532258a42dfa5dc7c563f0a3
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
BRANCH: codex/s10-phase1p-throughput-preflight
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at UNIQUE_BASE_SHA
OBJECTIVE: qualify whether same-node two-GH200 DDP gives robust Camera wall-time
  benefit without changing effective global B32, exposure, model/loss/update,
  precision, checkpoint cadence, evaluator state or data ownership
CANDIDATE_CAP: exactly two measurements in one allocation: fresh 1-GPU reference
  and fresh 2-GPU DDP candidate; no third candidate, 4-GPU cell or ablation
DATA_ROLE: exact D_fit only; official CBGS 87,930 expanded / 87,904 consumed /
  26 dropped per epoch; seed 0; no D_select/D_audit/official validation
PRODUCTION_RECIPE: Camera config file/resolved SHA256
  2e5368f96a6198e9a3b1bd43b258b53675df49f5c6ca9042fa8f72e0084c3b6a /
  0df1a19c057312923e0a8e48e81689d9ca265cc613c6f34d4795417414aa0bcf;
  B16xaccum2; conservative batched affine/grid; vectorized geometry/inverses;
  bulk native-image conversion; SDPA; scoped five-module forward compile; fused
  AdamW; FP16; activation checkpoint/telemetry off; one recovery checkpoint/epoch
REFERENCE: world size 1, physical B16, accumulation 2, effective B32
DDP_CANDIDATE: one node/world size 2, physical B16 per rank, accumulation 1,
  effective global B32; NCCL; broadcast_buffers=true; static_graph=true;
  gradient_as_bucket_view=true; find_unused_parameters=false; no SyncBatchNorm
GLOBAL_EXPOSURE: reshape each frozen global B32 permutation window, rank 0 gets
  [0:16] and rank 1 [16:32]; no DistributedSampler padding/striding; runtime epoch-0
  union plus frozen 20-epoch source identity must show no duplicate or omission
WORKER_RNG: seed + epoch*world_size + rank; exact within the DDP run and resume;
  accepted as measurement-only, not silently promoted as the production recipe
BN_POLICY: ordinary rank-local B16 BN is measured; rank-0 buffers define the
  checkpoint only after sustained timing, matching the next-forward DDP broadcast;
  cross-rank BN divergence is finite diagnostic evidence, not an equality gate;
  owner must explicitly accept the BN/RNG recipe before production promotion
CONTROL_FLOW: loss-finite, gradient-finite and GradScaler accepted/skipped decisions
  are synchronized across ranks before the corresponding branch; mismatch fails
MEASUREMENT: direct focused tests; two-rank NCCL smoke without D_fit; then fresh
  reference followed by DDP candidate in the same allocation; each sustained path
  uses 16 accepted warm-up plus 256 accepted measured B32 windows; 16-window blocks;
  50,000-draw one-sided 95% bootstrap; one-second two-GPU system samples
MEMORY_GATE: each rank peak reserved <=85% visible memory; no monotonic reserved
  growth over 64 MiB and no measured steady-state recompile
PERFORMANCE_GATE: DDP/reference sustained-rate one-sided 95% lower bound >=1.60;
  projected DDP wall <= fresh reference projection / 1.60; projected two-GPU
  charged hours <=1.25 * fresh reference projected one-GPU hours
DESCRIPTIVE_CURRENT_BOUND: Job-541821 final projection 13.285290 h implies <=8.303306 h
  DDP wall and <=16.606613 charged GH200-hours, but the fresh reference is binding
CHECKPOINT_GATE: rank-0 standard full checkpoint plus one RNG sidecar per rank;
  exact restored boundary/config/training/discrete state; exact per-rank input and
  RNG replay; same-process and fresh-process eight-window continuation; exact rank
  parameters/non-BN buffers/AdamW/scheduler/scaler/counters before canonical save
NUMERICAL_POLICY: finite/structural/group-identity checks are hard; grouped
  parameters/BN mean/BN variance/Adam exp_avg/exp_avg_sq relative-L2, max-absolute
  and per-element allclose are recorded diagnostics under the owner-amended gate
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_e5_e61c486757ca/
OUTPUT_RULE: source-SHA-qualified fresh smoke/reference/DDP directories and one
  write-once pair result; raw outputs immutable; no overwrite or identical retry
PER_JOB_RESOURCE: account naiss2025-22-1113-gpu; partition gpu; 1 node; 1 task;
  2 nvidia_gh200_120gb; 32 CPUs; 192 GiB; <=00:45:00; no requeue
BASE_AGGREGATE_GPU_HOURS: 1.50 charged GH200-hours
CODE_BUG_REMEDIATION_RESERVE: +1.50 charged GH200-hours, diagnosed code defect only
HARD_AGGREGATE_GPU_HOURS: 3.00 charged GH200-hours
MAX_CONCURRENCY: 1
SUBMISSION_POLICY: serial; no numeric remediation-count cap; base and reserve are
  separate; no blind identical retry; every derived source uses fresh output
REMEDIATION_AUTHORITY: smallest single-correct-answer test/fixture, config/schema,
  rank/dtype/API, runner, checkpoint/artifact/provenance/logging repair anchored to
  the frozen semantics; record source/command/output/charge before serial resubmit
STOP_ESCALATE: ceiling would be exceeded; same blocker recurs; ambiguous diagnosis;
  NCCL/hardware/resource/source drift; test/smoke/data-union/rank/checkpoint hard
  gate fails; OOM/>85%/growth/recompile; requested change touches math, data,
  precision, optimizer/scheduler, exposure, BN/RNG recipe, gate or interpretation
NEGATIVE_RESULT: a healthy result below any performance/payback threshold is a
  terminal negative qualification, not a code bug and not a rerun trigger
PROMOTION: never automatic; positive IP-E5 evidence still requires owner acceptance
  of the DDP BN/worker-RNG recipe and a later production-config decision
FORBIDDEN: capability/mAP/NDS/generalization/candidate-selection claims; scientific
  training; D_select; D_audit; official validation; LiDAR; 4-GPU DDP; Envelope B;
  merge, push, upload or publication
```

Prepared immutable identities:

```text
SINGLE_PROFILE_FILE_SHA256:
  2f01e83e7ccf820a76a6fe25c4ea355e4eaea060523bca755b9caa17f2da1b93
SINGLE_PROFILE_CANONICAL_SHA256:
  bb8318dd5e054ed5843402898a7ebab4aa939cc3e565c55400bad445ba0bf135
SINGLE_EFFECTIVE_RUNTIME_SHA256:
  4ffcd3a16e0b8a355aef6d17ec11da1e58418409d9b82ea17882b28dda34db58
DDP_PROFILE_FILE_SHA256:
  240118b574f32cb09b6a982a48bcb3de1206db5667b3fa32a6012a21922b1497
DDP_PROFILE_CANONICAL_SHA256:
  2d8939f15746cecfbeacb6867528b97b6796eac8e300a5ab08b330076873cfe5
DDP_EFFECTIVE_RUNTIME_SHA256:
  096415831a743368ca7529e46a3e382e1884038fcab9b7d2d8fc6d81919318f2
LAUNCHER_SHA256:
  77b84b103c96dfbe766974ad4051167108b852d0dd170e48d532b7b13ec2b129
DDP_ENTRY_SHA256:
  26fbb6d39b38cd89bc93db68df213866299fe9d16a0fe141405e5912a11e5dd3
PAIR_ANALYZER_SHA256:
  d8a57f399f2d14e0d554ef77908b095165495716bd1a8296ff8e7578cb9ae209
DDP_HELPER_SHA256:
  075df300ad1d808b57bfc6d7b9464b67b23e034e2e95dac5a076e515a8d62a2b
SINGLE_GPU_PROFILER_SHA256:
  291c6aa74183327b76d0f0180f21e2c180e8b97d720890763275d12b312aa107
PAIR_ANALYSIS_CORE_SHA256:
  35dc2354b9c53da24c3a7af324328ccde70814acc01d93d94dad3d6dc8812c82
TRAINING_LOOP_SHA256:
  4e2aeaf37c241f98a41722950e2aee34cc9c52dd444eba33bb1ce09a7de9dc82
DDP_SAMPLER_SHA256:
  86f764d4fb8b58d0d097e0b48f36fc4f474dd49f6c14abf98fdc3abab8e7832b
LOCAL_VALIDATION: shell syntax, Python byte-compilation, JSON syntax, strict direct
  profile parsing, current config/resolved identity, both effective runtime hashes,
  synthetic paired-result analysis and diff checks PASS
LOCAL_TEST_LIMIT: PyTorch/pytest are absent from the x86 login environment; the
  exact focused CPU/CUDA tests and two-rank NCCL smoke run before D_fit measurement
```

If and only if the owner activates IP-E5 at the containing clean commit, the exact
single Slurm invocation is:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=e61c486757ca5fe89340c9325014f4c3e048da2b
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_e5_e61c486757ca
mkdir -p "${ROOT}"
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=192G \
  --gpus-per-node=nvidia_gh200_120gb:2 --time=00:45:00 --no-requeue \
  --job-name=s10-p1p-ddp2 \
  --output="${ROOT}/slurm_initial_%j.out" \
  --error="${ROOT}/slurm_initial_%j.err" \
  fl_v3/scripts/run_s10_phase1p_ip_e5.sh \
  --config fl_v3/configs/s10_phase1_camera.json \
  --reference-profile \
    fl_v3/configs/s10_phase1p_ip_e5_camera_b16_single_gpu.json \
  --ddp-profile fl_v3/configs/s10_phase1p_ip_e5_camera_b16_ddp2.json \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

The owner explicitly activated this section at containing request commit
`2505db02920021663ccce7783dee483f10e638f8` with the full
`1.50 + 1.50 = 3.00` charged-GH200-hour envelope.

Terminal compact ledger:

```text
INITIAL_JOB: 543028
SOURCE_SHA: e25160f8811953c03e5805cf8c2917bc7f7ae2e0
SOURCE_TREE: 655c22e33b335e5c5deeb77a744c9f4290fe2014
NODE / RESOURCES: n418 / 2xGH200 / 32 CPU / 192 GiB / no restart or requeue
START / END / ELAPSED: 2026-07-21T20:54:22 / 21:15:02 / 00:20:40
SLURM_STATE: FAILED 1:0 only at terminal CPU pair analysis
CHARGE: 0.688889 charged GH200-hours
PRETEST: 10/10 PASS
NCCL_SMOKE: PASS; world size two / distinct devices / boolean collective /
  cross-rank gradient equality
REFERENCE: 35.469970 presentations/s; 256/256 accepted; checkpoint continuation
  PASS; projected 13.814431 h
DDP: 64.886915 presentations/s; 256/256 accepted; exact 87,904-presentation
  sampler union; all engineering/checkpoint/resume gates PASS
MEMORY: rank 0/1 peak reserved 75,807,850,496 / 75,939,971,072 bytes;
  74.32% / 74.45%; no >64 MiB monotonic growth
BN_DIAGNOSTIC: finite; relative-L2 0.0053210147; max-absolute 0.1102724075;
  cosine 0.9999869987; expected rank-local-B16 recipe distinction
FRESH_CONTINUATION_DIAGNOSTIC: exact hard gate PASS on both ranks; BN running-mean
  max-absolute 0.0054980032 exceeded diagnostic limit 0.0048224255 while its
  relative-L2 0.0004567734 passed 0.002; parameters, BN variance and both Adam
  groups passed their grouped diagnostics; elementwise allclose diagnostic false
INITIAL_TERMINAL_INCIDENT: pair analyzer compared whole source dictionaries;
  reference alone carried descriptive frozen_control_ref although all material
  source fields were identical
REPAIR_SHA / TREE: 26bf727ab36f5c3016b0c146eb8b8f3b3b66ec6d /
  a76fb348e8743f8494f682dd78b4f07ec177392a
REPAIR_SCOPE: output-neutral CPU provenance normalization plus regression test;
  no model/data/precision/optimizer/exposure/gate change and no GPU rerun
REPAIRED_PAIR_CORE_SHA256:
  ae2c4540b0faed7d46c82f904f0afd605ce323deb2eb25012f5c122daa603b2e
PAIR_RESULT_SHA256:
  0a0bd6569387c05cc170a129f9b83c94b6fefc2c5f8e6e6b0751d906d6d5a31c
THROUGHPUT_RATIO / ONE-SIDED_95_LB: 1.829347881 / 1.818635492; gate >=1.60 PASS
PROJECTED_DDP_WALL / LIMIT: 7.581252 / 8.634019 h PASS
PROJECTED_DDP_CHARGED / LIMIT: 15.162504 / 17.268039 GH200-hours PASS
CHARGED_RATIO: 1.097584; gate <=1.25 PASS
VERDICT: POSITIVE_DDP_QUALIFICATION
PRODUCTION_PROMOTION_AT_IP_E5_CLOSURE: false / owner decision was then pending
SUBSEQUENT_OWNER_DECISION: promoted exact two-GH200 recipe; ordinary per-rank B16
  BatchNorm and seed + epoch*world_size + rank worker RNG explicitly accepted
BUDGET: base used 0.688889/1.50; code-bug reserve used 0/1.50;
  total 0.688889/3.00; unused authority expired at closure
RESIDUAL: one non-contiguous 1x1-convolution grad/bucket-view stride warning may
  leave optional performance headroom; no correctness/hard-gate impact
INTERPRETATION: D_fit throughput and engineering health only; no capability,
  mAP/NDS, generalization, candidate-selection or Envelope-B claim
```

### 9.7 Owner production promotion — source/config materialized, no execution

```text
DECISION_STATE: CLOSED / TWO-GH200 CAMERA RECIPE PROMOTED / NOT EXECUTABLE
OWNER_DECISION: explicitly accepts ordinary per-rank B16 BatchNorm and
  seed + epoch*world_size + rank worker RNG
IMPLEMENTATION_SHA: 2c3780bb6373ae784b41c22df072824f7a92d457
IMPLEMENTATION_TREE: 54e26e8ef20b8fdbe56cd2c736484c309fd9c6d2
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification remains at UNIQUE_BASE_SHA
CAMERA_SCHEMA: s10.phase1.v4 / throughput decision IP-E5 / evidence commit
  5da03ffdaa29614b0bcfc5c85ace93f70acfac6a
CAMERA_CONFIG_FILE_SHA256:
  9a2cdf54a52edeb71b5335aea8445c0a8cc0c8e2e416b2f4fe3df58d7b98710c
CAMERA_RESOLVED_CONFIG_SHA256:
  e295b627551a584b460a598ee3e3f23b5ad8dda45441904d4ed526bbf3457f2b
PRODUCTION_RECIPE: one node; world size 2; B16 per rank; accumulation 1;
  effective global B32; ordinary rank-local BN; exact frozen global CBGS B32
  window split as rank0 [0:16] / rank1 [16:32]; worker generator seed
  seed + epoch*world_size + rank; FP16; SDPA; scoped five-module forward compile;
  conservative batched affine/grid; vectorized geometry/inverses; bulk native-image
  conversion; fused AdamW; activation checkpoint/telemetry off
DDP_RUNTIME: NCCL; broadcast_buffers=true; static_graph=true;
  gradient_as_bucket_view=true; find_unused_parameters=false; synchronized
  finite-loss/gradient/scaler decisions
CHECKPOINT: unchanged one-epoch cadence; rank-0 canonical model/full optimizer,
  scheduler, scaler and training state plus one exact RNG sidecar per rank;
  resume restores the sidecar for each rank before the next epoch loader stream
CAMERA_ENTRY_SHA256:
  4b91e81c5060bec0108b99abaa6b29e6df4d4def0d04f45e54a4b20df830162e
ENVELOPE_B_LAUNCHER_SHA256:
  77180d41adcab03014f13b82ba74b4f2209c084ee5ac0823e345356b51865cff
CAMERA_OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1_envelope_b_camera_ddp_5da03ffdaa29
LIDAR_DISPOSITION: config/science and single-GPU execution path unchanged
LOCAL_VALIDATION: JSON syntax; strict v4 resolve/hash and recipe-drift rejection;
  Python byte-compilation; shell syntax; ShellCheck; diff checks PASS
LOCAL_RUNTIME_LIMIT: Torch/pytest are absent from the x86 login environment;
  no local CUDA/DDP execution was claimed
EXECUTED: no GPU/Slurm, scientific training, D_select, D_audit or validation
ENVELOPE_B_STATE: Section 7 remains historical/frozen and NOT EXECUTABLE; before
  submission it must be revised to this source/config/two-GPU resource projection
  and pass the pre-existing independent recipe-freeze review with no open P0-P2
MERGE_PUSH_UPLOAD_PUBLICATION: not authorized and not performed
```

### 9.8 LiDAR IP-LG0 closure and exact IP-L-E1 request

```text
REQUEST_STATE: TERMINAL / IP-L-E1 RETURNED TO IP-LG1
PHASE: S10 Phase I-P LiDAR / L-WP1 clean capacity, sustained baseline and trace
GATE_STATE: IP-LG0 CLOSED; IP-LG1 OPEN for owner batch/cell decision
IMPLEMENTATION_SHA: 0daeee95e1a46b29fcd7bbb2338d813b798557de
IMPLEMENTATION_TREE: 26cb8aa6a91f2938cf8c1357a97b0bbf2ee92136
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification remains at UNIQUE_BASE_SHA
APPROVED_SOURCE_SHA: 8b5788d3d905cf7eb83e8f3f1e65e24df7fc15dc
ACTIVATION_DECISION: owner explicitly approved and activated IP-L-E1 at this
  containing request commit

OBJECTIVE: establish one clean current LiDAR capacity/throughput/memory/checkpoint/
  training-health baseline, localize whole-model bottlenecks, and quantify B4 versus
  the highest safe physical batch before any optimization cell is frozen
EXIT: ordered capacity ladder reaches its first rejection or B32; B4 and the highest
  accepted batch finish the sustained repeat rule and trace rule; all hard gates
  pass; compact evidence returns to IP-LG1 with no automatic recipe promotion

DATA_ROLE: exact D_fit only; existing exact CBGS and role-bound keyframe GTDB
SEED_EXPOSURE: seed 0; effective global B32; no change to epoch presentation count,
  CBGS identity, model/head, loss/target, scheduler or checkpoint cadence
PRECISION: global FP16 autocast with the accepted sparse FP32 island; sparse-conv
  FP16 is forbidden in this envelope
FORBIDDEN: D_select; D_audit; official validation; capability/mAP/NDS/generalization/
  candidate-selection claims; scientific training; Camera; Fusion; Envelope B;
  merge; push; upload; publication

SOURCE_LIDAR_CONFIG: fl_v3/configs/s10_phase1_lidar.json
SOURCE_LIDAR_FILE_SHA256:
  c7e1fa26e1714a31c5998296cb95cbab5e8732d4bf2f06da81fd6d631c574bfc
SOURCE_LIDAR_RESOLVED_SHA256:
  0efe4d6d5138e3d99ae80254a6ecf884300dd18985ab45a00425228fc3ef082e
PROFILES:
  B4x8  fl_v3/configs/s10_phase1p_lidar_e1_b4.json
    sha256 2002679dc7bd589b30b3092510faca780b47585f978ad3f388d4ce41b476883c
  B8x4  fl_v3/configs/s10_phase1p_lidar_e1_b8.json
    sha256 6fdb593ad199bb6705770520e1f6239a16abd62b40486656751beed3dc8657e4
  B16x2 fl_v3/configs/s10_phase1p_lidar_e1_b16.json
    sha256 7e7d17dd000bf63743fc8c29be50c49c784194a1b8156e8b65694e908349bf98
  B32x1 fl_v3/configs/s10_phase1p_lidar_e1_b32.json
    sha256 2d199cee7634244b84f48923a6175cc7f8e89e8516554b5727fc8c489d585e05
PROFILER_ENTRY_SHA256:
  d02f96f0400371930702fac8ea7467d904cfe9b88d202f8d2eaebdafa853a5a6
LAUNCHER: fl_v3/scripts/run_s10_phase1p_lidar_e1.sh
LAUNCHER_SHA256:
  2f4f0967ecf9ca5014e94a8f1cbef1fc6b2e193afa3ee374504d948c2ba27830

CANDIDATES: exactly four measurement-only clean profiles; every optimization flag
  is false; only physical batch/accumulation are B4x8, B8x4, B16x2 and B32x1
CAPACITY_ORDER: fresh processes B4 -> B8 -> B16 -> B32; one warm-up plus eight
  accepted optimizer windows each; stop before larger batches on first OOM,
  >85% peak-reserved-memory rejection, numerical-health rejection or other failure
SUSTAINED_ORDER: B4 repeat 1 -> highest-safe repeat 1 -> highest-safe repeat 2 ->
  B4 repeat 2 (the duplicate highest-safe entries are omitted when B4 is highest)
SUSTAINED_WINDOW: 16 accepted warm-up plus 256 accepted measured optimizer windows
CONDITIONAL_REPEAT: separately for B4 and highest-safe, add repeat 3 iff absolute
  first-two rate difference divided by their mean exceeds 3%
TRACE_ORDER: after sustained measurement, one B4 and one highest-safe trace; each
  has 16 accepted warm-up plus three active accepted windows; if B4 is highest,
  only one trace runs
PAIRING: all processes execute serially in one Slurm allocation/on one node/GPU;
  trace rates are profiler-inflated localization evidence and never throughput

HARD_GATES: exact source/config/profile/data-role/seed/effective-B32 identities;
  zero invalid, nonfinite, overflow, discarded or scaler-skipped measured windows;
  exactly accepted exposure; <=85% peak reserved memory; no sustained monotonic
  >64 MiB reserved-memory growth; finite loss and all four LiDAR criterion terms;
  exact checkpoint boundary/input/RNG/training/discrete/structure state; accepted
  grouped fresh-process continuation gate; trace core-range inventory complete
CONTINUATION_POLICY: model parameters, BN mean, BN variance, Adam exp_avg and
  exp_avg_sq each use fresh-process relative-L2 and max-absolute <=
  max(frozen tolerance, 1.25x same-process repeat-control); per-element allclose is
  diagnostic, not a hard gate
LOSS_INTERPRETATION: overall/first-quarter/last-quarter and loss_heatmap/loss_cls/
  loss_bbox/matched_iou are health observations; finite values are hard, but a
  short-window downward slope is not required and cannot support capability claims
BATCH_RECIPE_BOUNDARY: within each batch recipe, boundary/input/RNG/discrete state
  remain exact; cross-batch worker augmentation equality is not required because
  the owner already accepted that physical batch changes worker RNG assignment;
  no B8/B16/B32 promotion occurs in L-E1
OUTPUTS: immutable result/measurement/checkpoint/continuation/system-sampling and
  structured trace artifacts; report per-process throughput, repeat spread,
  utilization/power, peak allocated/reserved, 20-epoch projection and descriptive
  highest-safe-over-B4 payback at IP-LG1

RESOURCES_PER_JOB: one node; one GH200; 16 CPUs; 96 GiB; at most 01:15:00;
  no requeue/restart; maximum concurrency one
AGGREGATE_BASE: 1.25 charged GH200-hours
CODE_BUG_RESERVE: +0.50 charged GH200-hours
HARD_AGGREGATE_CEILING: 1.75 charged GH200-hours
SUBMISSION_POLICY: no numeric submission cap inside the activated O-149 envelope;
  every process is serial and aggregate charge is the controlling limit
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e1_<approved_source_sha12>
FRESH_OUTPUT_RULE: each derived source uses source-SHA-qualified child directories;
  no artifact or prior attempt is overwritten
REMEDIATION_AUTHORITY: diagnose and make only the smallest unambiguous frozen-
  semantics repair to tests/fixtures/runner/API/config parsing/discrete plumbing/
  checkpoint/provenance/logging, validate, commit linearly and resubmit serially
REMEDIATION_STOP: same blocker recurs after repair; diagnosis is ambiguous; a change
  touches data/model/math/precision/loss/target/order/exposure/optimizer/scheduler/
  gates/interpretation; or the next job could exceed the hard aggregate ceiling
OWNER_RETURN: terminal L-E1 evidence returns to IP-LG1 before any optimization
  implementation, L-E2 activation, batch promotion or scientific recipe decision
```

Exact activated command:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=8b5788d3d905cf7eb83e8f3f1e65e24df7fc15dc
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e1_${APPROVED_SOURCE_SHA:0:12}
mkdir -p "${ROOT}"
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=01:15:00 --no-requeue \
  --job-name=s10-p1p-l-e1 \
  --output="${ROOT}/slurm_initial_%j.out" \
  --error="${ROOT}/slurm_initial_%j.err" \
  fl_v3/scripts/run_s10_phase1p_lidar_e1.sh \
  --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

Active compact ledger and derived replacement record:

```text
INITIAL_JOB: 555479
SOURCE_SHA: 09b236c18811b1817b1a50a278eec943a32abedd
APPROVED_SOURCE_SHA: 8b5788d3d905cf7eb83e8f3f1e65e24df7fc15dc
NODE / RESOURCES: n58 / 1xGH200 / 16 CPU / 96 GiB / 01:15:00 /
  no restart or requeue
SUBMIT / START / END / ELAPSED: 2026-07-22T08:34:57 /
  2026-07-22T08:34:58 / 2026-07-22T08:50:59 / 00:16:01
SLURM_STATE: FAILED 1:0 in B4 sustained post-measurement memory-health check
CHARGE: 0.266944 charged GH200-hours, code-bug reserve
PRETEST: 7/7 PASS
CAPACITY_B4: COMPLETE_CAPACITY; 39.608844208 presentations/s;
  allocated/reserved 5,441,557,504 / 6,127,878,144 bytes (6.0074%);
  result sha256 e78bf4b8d8b1cf27fa83623a4d92f4a682ed068e64df8de786aaaeda9ef1bb31
CAPACITY_B8: COMPLETE_CAPACITY; 47.645146524 presentations/s;
  allocated/reserved 10,534,451,712 / 13,604,225,024 bytes (13.3368%);
  result sha256 19a1e5b0f1e919ab56b90a4f3e29930bcf7c9b27031a226d7c082980664219b6
CAPACITY_B16: COMPLETE_CAPACITY; 48.848315618 presentations/s;
  allocated/reserved 20,609,104,384 / 24,622,661,632 bytes (24.1386%);
  result sha256 847f1f7121b61fb182b4b75fbdffd5d189e540a317300b2a2808201e09f668a6
CAPACITY_B32: COMPLETE_CAPACITY; 51.784549829 presentations/s;
  allocated/reserved 40,773,322,752 / 47,886,368,768 bytes (46.9449%);
  result sha256 53c4e1f0f08027eea13fe3c9e0747a56db48c8ac2ae2001f93e450a16efc813f
CAPACITY_HEALTH: every cell 8/8 accepted with finite loss/component and update
  gates PASS; rates are short-window sanity evidence, not sustained comparisons
FAILED_SUSTAINED_ARTIFACT: no result/measurement was published by the old early
  memory require; nvidia-smi peak was only 7,405 MiB; immutable sampler sha256
  609efb4a0128497a7f23f9a753421adad799787706a6e40e0a6aede1393b5e62
DIAGNOSIS: loss-health retained every detached CUDA scalar through 256 windows,
  pinning otherwise reusable allocator blocks; the strict four-quartile >64 MiB
  monotonic-growth diagnostic fired despite large physical headroom
CLASSIFICATION: unambiguous output/science-neutral profiler-observer plumbing bug
REPAIR_IMPLEMENTATION_SHA: 3f1b8d91568a413b1f6cf1dda39a989a1be20ecf
REPAIR_TREE: 55d364f723d51b9aa696a5b98b1decb32bbeb817
REPAIR: flush identical weighted loss/component scalars once per 16-window block;
  release warm-up views before peak reset; exclude only observer-flush wall time;
  persist future IP-L-E1 memory-health failures before raising
REPAIR_LAUNCHER_SHA256:
  e79b547880b4cb547783c8324c4724531803d54734c6dfde56269dccc36f207d
REPAIR_PROFILER_ENTRY_SHA256:
  8f9ab2ac88de2e9cdbc955b27f34dc1459cd8ddd5e2ba5a72d0cd04f4ac627f4
REPLACEMENT_STATE: authorized automatically by the approved O-149 bug loop
REPLACEMENT_SOURCE: exact containing ledger commit; a linear descendant of the
  approved source and repair implementation
REPLACEMENT_OUTPUT: same approved root with source-SHA-qualified fresh child paths;
  Slurm logs use slurm_repair1_%j and overwrite nothing
REPLACEMENT_COMMAND: exact Section-9.8 command with SOURCE_SHA equal to the
  containing ledger commit and unchanged APPROVED_SOURCE_SHA/resources/cells/order
BUDGET_AFTER_INITIAL: bug reserve 0.266944/0.50 used; base 0/1.25 used;
  hard aggregate 0.266944/1.75 used; replacement remains inside every ceiling

REPLACEMENT_JOB: 555777
REPLACEMENT_SOURCE_SHA: 77f13f955e0d089db6c2ae77957f1e903ddc6410
REPLACEMENT_TREE: 984d658a847ac06ff72513c6da959b1b73e15309
NODE / RESOURCES: n64 / 1xGH200 / 16 CPU / 96 GiB / 01:15:00 /
  no restart or requeue
SUBMIT / START / END / ELAPSED: 2026-07-22T08:57:16 /
  2026-07-22T08:57:20 / 2026-07-22T09:51:25 / 00:54:05
SLURM_STATE: COMPLETED 0:0; highest tested safe batch B32
CHARGE: 0.901389 charged GH200-hours, base envelope
PRETEST: 8/8 PASS, including bounded loss-observer retention regression
PROVENANCE: approved source 8b5788d3d905cf7eb83e8f3f1e65e24df7fc15dc;
  source-resolved config 0efe4d6d5138e3d99ae80254a6ecf884300dd18985ab45a00425228fc3ef082e;
  every cell records source 77f13f95 / tree 984d658a and D_select=false,
  D_audit=false, official-validation=false, capability-metrics=false
CAPACITY_B4: COMPLETE_CAPACITY; 37.396749581 presentations/s;
  allocated/reserved 5,441,537,024 / 6,127,878,144 bytes (6.0074%);
  result sha256 c5e50c0fdf700549a61e370f389ca17e31f726da709c99abf2cc253654a6d333
CAPACITY_B8: COMPLETE_CAPACITY; 48.893392645 presentations/s;
  allocated/reserved 10,532,373,504 / 13,587,447,808 bytes (13.3203%);
  result sha256 ad3191cfb81b4f915967b1296166bb8072cb65eb73b9ed70ab4b6a142c6cffc6
CAPACITY_B16: COMPLETE_CAPACITY; 52.712682166 presentations/s;
  allocated/reserved 20,609,099,264 / 24,622,661,632 bytes (24.1386%);
  result sha256 f85316c756e71af22530b90f2911a71416486466eea9274a8feb8943e9facab7
CAPACITY_B32: COMPLETE_CAPACITY; 49.009159547 presentations/s;
  allocated/reserved 40,773,320,192 / 47,884,271,616 bytes (46.9428%);
  result sha256 8e80985fead71d0495294030d36fb9346373cdf9d3397f4b2f545ff778f4cde3
CAPACITY_HEALTH: all four cells 8/8 accepted with finite criterion components,
  no overflow/skip/discard and every memory gate PASS; short rates are not the
  sustained comparison
SUSTAINED_B4_R1: 36.223037446 presentations/s; result
  cf2b1197699e97cbbc24eff4da87fd2af8e139df7a3db9d69bacdca48ae1db3c
SUSTAINED_B32_R1: 51.688027892 presentations/s; result
  db3d9fec6c0865b1d964c5fd95afb0f19eb3ac0041ad423928ba0057a2b8c3bb
SUSTAINED_B32_R2: 52.429398150 presentations/s; result
  b9b75004b9eb5373c1bb32c5b33ca146f3a5e27d31890e9e55a52568313a1d65
SUSTAINED_B4_R2: 39.119782780 presentations/s; result
  db12741fb056c5b5bbd2102d30e5f5756d48ee0b1721214da36cfe6a919d342b
B4_FIRST_TWO_SPREAD: 0.076895060 > 0.03; exact conditional r3 triggered
SUSTAINED_B4_R3: 36.093056094 presentations/s; result
  408554a48b223f2ff0d07feb3380341e4339964a443d1103d2e1aedddd5936a1
B32_FIRST_TWO_SPREAD: 0.014241041 <= 0.03; no B32 r3
SUSTAINED_HEALTH: all 1,280/1,280 measured windows accepted; zero invalid,
  nonfinite, overflow, discard or scaler skip; B4 reserved 6,962,544,640 bytes
  (6.8257%), B32 reserved 47,886,368,768 bytes (46.9449%); no monotonic growth
LOSS_HEALTH: B4 first/last-quarter total-loss means are 43.7629/6.5967,
  44.1327/6.5837 and 43.8459/6.5401; B32 values are 43.3178/6.6660 and
  43.1046/6.7354; every total and criterion component finite
CHECKPOINT_HEALTH: every process has exact boundary/input/RNG/training/discrete/
  structure state and checkpoint gate PASS; checkpoint is about 100.0 MB;
  save plus file/model hashes median about 0.22 seconds
NUMERICAL_DIAGNOSTICS: under the pre-existing owner-amended diagnostic-only rule,
  B4 r1 BN-mean fresh relative-L2/max-absolute 0.00238861/0.0830487 narrowly
  exceeded 0.00221563/0.0794596, and B32 r2 Adam exp_avg max-absolute 0.00858848
  narrowly exceeded 0.00836905 while relative-L2 passed; neither recurred, all
  tensors were finite, and the other three processes passed every grouped
  diagnostic; these are recorded residuals, not hidden hard-gate failures
ROBUST_COMPARISON: B4 three-process median 36.223037446; B32 two-process median
  52.058713021 presentations/s; B32/B4 1.437171388 (+43.717%)
SYSTEM_COMPARISON: measurement-window nvidia-smi medians are approximately
  B4 51.47% GPU utilization / 278.91 W and B32 60.04% / 333.82 W; loader mean
  remains only 2.01-2.16 ms/B4 window and 0.61-0.62 ms/B32 window
PROJECTION: exact 87,904 consumed presentations/epoch x20, plus one median startup
  and measured per-epoch checkpoint/hash cost: B4 13.486357 GH200-hours; B32
  9.385269; descriptive saving 4.101088 hours and 30.409% wall time
TRACE_B4: COMPLETE_TRACE; 3/3 active accepted; full range inventory; result
  159eae4b88cf1bd0e05d4da921eb2b724a59f294f3d2d117c3d1055c12283450;
  structured summary 3441eff24cc3b53d99a4a2129650718e4a037a6c126686b7903c1658566bd4c7
TRACE_B32: COMPLETE_TRACE; 3/3 active accepted; full range inventory; result
  f3cbc95001863363438d97cf30edacc2063fb55bcfaf319aa14cf4ccc8db5ad5;
  structured summary 4dcddef8e41ccfddd139a91c1b0826b4230ed3f1b95580c5d35dc68ecc6595b8
TRACE_DIAGNOSIS: both traces cover 96 presentations and three optimizer windows.
  B32 versus B4 named CPU totals: forward 0.577/1.567 s, sparse-VFE-collapse
  0.498/1.059 s, loss 1.785/1.795 s, target_build 1.775/1.706 s, Hungarian cost
  0.841/0.768 s, Gaussian target 0.560/0.597 s. Finite-sync + Hungarian D2H +
  index H2D is only 69.35 ms at B32 (~3.9% of target_build); sparse batch-index/
  grouping is only 4.39 ms. Target/Hungarian/Gaussian work is the primary B32
  residual; SDPA remains plausible from ~51.95 ms cross-attention device time.
  Trace totals are nested/inflated ranking evidence, never sustained timing.
SLURM_STDOUT_SHA256:
  ca6721bc40d432816bed5d1d9dbb092024fe21bcfadb857d00f0abd68c9aa963
SLURM_STDERR_SHA256:
  e800bf6938f356fce2e7c65aa2d89b677e6d69ddf360fd536bc33b709af0ec8c
FINAL_BUDGET: base 0.901389/1.25; bug reserve 0.266944/0.50;
  aggregate 1.168333/1.75 charged GH200-hours; maximum concurrency one respected
EXIT_STATE_AT_L_E1_CLOSE: objectives met and envelope closed; B32 was then
  measurement-only and returned to IP-LG1. Section 9.9 records the later owner
  acceptance/freeze; no IP-L-E2 compute follows from either record alone
```

### 9.9 LiDAR IP-LG1 closure and frozen L-WP2 test units

```text
REQUEST_STATE: IP-LG1 CLOSED / L-WP2 CELLS FROZEN / IP-L-E2 APPROVAL PENDING
OWNER_DECISION: 2026-07-22 — accept LiDAR-only physical B32 x accumulation 1,
  including its BatchNorm and worker-RNG recipe relative to B4; freeze the five
  isolated L-WP2 paired cells below
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification remains at UNIQUE_BASE_SHA
PREDECESSOR_EVIDENCE: IP-L-E1 source 77f13f955e0d089db6c2ae77957f1e903ddc6410;
  Job 555777; robust B32/B4 ratio 1.437171388; B32 reserved 46.9449%; all
  measurement-health and exact checkpoint hard gates PASS
AUTHORITY_NOW: scoped source/docs/tests/local validation and linear commits only;
  no GPU/Slurm, D_select, D_audit, official validation, Envelope B, merge or push

ACCEPTED_BATCH_RECIPE: LiDAR-only B32 x accumulation 1/effective B32, ordinary
  single-GPU BatchNorm and its B32 worker assignment; no cross-batch per-sample
  augmentation equality requirement
UNCHANGED_SCIENCE: exact D_fit role and CBGS identity/order/count; seed 0; model/
  head; global FP16 plus sparse FP32 island; loss/targets; AdamW hyperparameters;
  scaler/clip/scheduler; one-epoch checkpoint cadence; evaluator semantics
FUSION_BOUNDARY: B32 acceptance does not apply to Camera or Fusion and makes no
  capability, mAP/NDS, convergence, generalization or candidate-selection claim

REFERENCE: current LiDAR runtime at B32x1 with every optimization flag false
PAIR_PROTOCOL: two fresh processes serially inside one Slurm allocation/node/GPU;
  same source/config/D_fit/CBGS/input anchor; 16 accepted warm-up plus 256 accepted
  measured windows per process; one-second system sampling; 16-window blocks;
  50,000-draw one-sided 95% paired block bootstrap
EXACT_SERIAL_CELLS:
  L2-1. B32 reference -> B32 hungarian_batched_d2h
  L2-2. B32 lidar_sdpa -> B32 reference
  L2-3. B32 reference -> B32 torch_compile
  L2-4. B32 lidar_host_batch_offsets -> B32 reference
  L2-5. B32 reference -> B32 fused_adamw
ISOLATION: every candidate changes exactly its named flag; no prior positive is
  carried into a later primary cell and no combined stack is part of L-WP2

TARGET_HOST_SCOPE: batch redundant GT/box validity reductions plus Hungarian
  cost/index D2H/H2D plumbing only; preserve per-sample GPU cost/IoU math, SciPy
  linear_sum_assignment inputs/results, encoded targets and Gaussian-draw math
LIDAR_SDPA_SCOPE: only the two TransFusion ReferenceMultiheadAttention cores;
  preserve Q/K/V/out projections, head count/scale, dropout probability, residuals,
  normalization, parameters/buffers and state-dict names; record changed kernel RNG
  consumption as measurement evidence
COMPILE_SCOPE: torch.compile backend=inductor, mode=default, dynamic=false;
  forward-only decoder_backbone, decoder_neck and head; sparse encoder, data,
  target/loss, backward wrapper and optimizer remain eager
SPARSE_HOST_SCOPE: collate-authored exact point offsets replace redundant sustained-
  path batch-index/grouping/stat host synchronization; preserve sample ordering,
  PointShuffle, PointToVoxel calls/caps, VFE/spconv/dense collapse and FP32 island
FUSED_ADAMW_SCOPE: PyTorch fused backend only; exact parameter groups,
  hyperparameters, GradScaler, clipping, accepted step, scheduler and checkpoint
  optimizer-state contract

HARD_GATES: exact source/config/profile/D_fit/CBGS/seed/effective-B32 identities;
  pair input anchor and CBGS prefix; zero invalid/nonfinite/overflow/discard/scaler-
  skipped measured windows; finite total loss and four criterion terms; exact
  accepted exposure; <=85% peak reserved; no monotonic reserved growth; exact
  checkpoint boundary/input/RNG/training/discrete/structure and finite state;
  candidate-specific parity/state/compile gates
NUMERICAL_POLICY: grouped model-parameter/BN-mean/BN-var/Adam-exp_avg/exp_avg_sq
  relative-L2, max-absolute and elementwise-allclose results are diagnostics under
  the latest owner amendment; they cannot alone fail a finite and structurally
  intact non-deterministic kernel
STRICT_PLUMBING_PARITY: target-host and sparse-host candidates require exact
  discrete results plus controlled FP32 and accepted-FP16-policy forward/backward/
  accepted-update parity under frozen continuous-tensor tolerances; elementwise
  exactness remains diagnostic for qualified non-deterministic kernels
SDPA_GATE: FP32/FP16 forward/backward/update checks, explicit training-dropout/RNG
  record, exact state names and checkpoint/resume
COMPILE_GATE: exactly three compiled scopes, unchanged state names, finite parity,
  graph/cold-start record and no unexpected measured steady-state recompile
FUSED_GATE: exact parameter-group identity, finite accepted update and complete
  exp_avg/exp_avg_sq checkpoint/resume

PERFORMANCE_CLASSIFICATION: positive only if every hard gate passes and the
  candidate/reference one-sided 95% throughput-ratio lower bound is >1.00;
  point estimate >1 with lower bound in [0.98,1.00] is conditional for a later
  L-WP3 decision, not promoted; every other healthy result is throughput-negative
CONTINUATION: all five independent cells may run serially inside one activated
  IP-L-E2 without per-cell owner round trips; a candidate-local negative or
  unsupported compiler path does not stop later independent cells
STOP_ESCALATE: shared hard-gate failure; nonfinite/discard; ambiguous or repeated
  blocker; science-boundary pressure; requested scope/gate/resource change; or
  hard aggregate ceiling exhaustion
L_WP3_BOUNDARY: no primary combination in L-WP2. Positive/conditional composition
  and any full-sort-to-topk, batched voxelization, batched Gaussian target,
  H2D-field-pruning or hidden-sync-cleanup cell remain IP-LG2/L-WP3 work

PROPOSED_RESOURCES_PER_JOB: one node; one GH200; 16 CPUs; 96 GiB; <=00:45:00;
  no requeue/restart; maximum concurrency one
PROPOSED_AGGREGATE_BASE: 2.00 charged GH200-hours
PROPOSED_CODE_BUG_RESERVE: +0.75 charged GH200-hour
PROPOSED_HARD_CEILING: 2.75 charged GH200-hours
PROPOSED_SUBMISSION_POLICY: serial; no numeric remediation-submission cap; no blind
  retry; aggregate layered ceiling and maximum concurrency are binding
PROPOSED_OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e2_<approved_source_sha12>/
FRESH_OUTPUT: every process/pair/attempt path is source-SHA-qualified, absent before
  execution and never overwritten; raw evidence remains immutable
REMEDIATION: O-149 smallest unambiguous frozen-semantics test/fixture/runner/API/
  config/discrete/checkpoint/provenance/logging repair only, serially inside the
  reserve; candidate math/scope, data, precision, recipe, gates and resources stay
  owner-gated
ACTIVATION_REQUIREMENT: first materialize and locally validate exact default-off
  source, tests, profiles, pair runner and commands; record their hashes and a clean
  containing source SHA here; then obtain explicit owner approval of that SHA and
  the proposed resources. This section alone is not executable authority.
```

### 9.10 LiDAR IP-L-E2 activation and execution ledger

```text
REQUEST_STATE: TERMINAL / IP-L-E2 CLOSED / OWNER RETURN AT IP-LG2
OWNER_DECISION: 2026-07-22 — "激活IP-L-E2"
APPROVAL_ANCHOR_SHA: d1789bba4804dfcdab4d26a5780a836e69b56355
ACTIVATION_INTERPRETATION: the explicit phase activation binds the already frozen
  cells, gates and resources at APPROVAL_ANCHOR_SHA and authorizes only their exact
  default-off source/tests/runner materialization as an O-149 derived descendant;
  it is not standing authority to change candidate math, gates or resources
DERIVED_IMPLEMENTATION_SHA: 914ca11db74f9f5b2f7f6836dbc566c012d3a661
DERIVED_IMPLEMENTATION_TREE: 561f8c9b4b60c73dbf307b3d3b8eb3a90add9b95
EXECUTION_SOURCE: exact clean containing activation-ledger commit, resolved at
  submission and recorded in the terminal entries below; it must be a linear
  descendant of APPROVAL_ANCHOR_SHA and UNIQUE_BASE_SHA
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification remains at UNIQUE_BASE_SHA

ACTIVATED_SCOPE: exactly the five Section-9.9 isolated B32x1 same-allocation pairs,
  in order L2-1 target/Hungarian, L2-2 SDPA, L2-3 scoped compile, L2-4 sparse host
  offsets, L2-5 fused AdamW; no candidate is composed or promoted by this envelope
CONTINUOUS_AUTHORITY: candidate-local performance negative or unsupported compiler
  path is recorded and the next independent cell proceeds; an unambiguous code bug
  may receive the smallest default-off O-149 repair and serial fresh-output rerun
OUT_OF_SCOPE: L-WP3/IP-LG2, Camera, Fusion, D_select, D_audit, official validation,
  capability metrics, revised/original Envelope B, merge, push and publication

PROFILER_ENTRY_SHA256:
  9d21d4bbb965a62589988495546434e36b8d9b3c4b173228b3b3f9c295efcc0b
COMPARATOR_ENTRY_SHA256:
  a9492deb7b0d8a884d6bb3e2246aa6bba86bc8225061430cdfddffc01fe64847
PAIR_LAUNCHER_SHA256:
  6378a59943d89dcfecae4cc77f5a3cbae317d241277e5fc718969e939847ecdd
TARGET_SOURCE_SHA256:
  c031140e0aa0b26fe4625ef93e0c08f01b9cdc4c4eb6e82409d0cd71e154fa2c
SPARSE_SOURCE_SHA256:
  badde1b9019c7353e6fecbafd116859412a204c8e6a8e94e6cc677161dc7ee38
PAIR_COMPARATOR_SHA256:
  3b610c10f1c63dd2bd63c2a10f35c38954795de8c09a15d0bddf212b126a45ec
PROFILE_SHA256:
  reference fda8eac60288a812f34c2bc2a8bbdcc3331ec155cb26e4cdbea361a0c23db928
  target    9acd85664a10dec977412b2787879da7518f8fa3d8083b5bc98ee285609639ba
  SDPA      bcd2734d6e24df99a84e938a2181dbd164828efd64f9bd289d9336a9e66c512a
  compile   bfce36632aaa503b054e3bcd4b451d2a18856e18a1ead99ac837e7f37884ac75
  offsets   eb5845322167ce3109f499d3f7833e1e28faa7dc6f822309346bcbc3d77b1012
  fused     0e4e8494dffcd80657bbb92a97358893242629b975ea5eb972ee90d6da9423e5
LOCAL_VALIDATION: Python py_compile PASS; six JSON parses PASS; bash -n PASS;
  shellcheck PASS; git diff --check PASS; login node has no pytest/PyTorch runtime,
  so exact PyTorch/spconv/SDPA/target/update tests are mandatory in-job preflight

RESOURCES_PER_JOB: one node; one GH200; 16 CPUs; 96 GiB; <=00:45:00; no requeue;
  maximum concurrency one
AGGREGATE_BASE: 2.00 charged GH200-hours
CODE_BUG_RESERVE: +0.75 charged GH200-hour
HARD_AGGREGATE_CEILING: 2.75 charged GH200-hours
SUBMISSION_POLICY: five cells serially; no numeric repair-submission cap; no blind
  retry; aggregate layered ceiling and maximum concurrency are binding
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e2_d1789bba4804/
FRESH_OUTPUT: source-SHA-qualified process/pair paths; raw evidence never overwritten
STOP_ESCALATE: shared hard-gate failure; nonfinite/discard; ambiguous or repeated
  blocker; science-boundary pressure; scope/gate/resource change; or next charge
  could exceed the hard aggregate ceiling
BUDGET_AT_ACTIVATION: base 0/2.00; bug reserve 0/0.75; aggregate 0/2.75
```

Exact command template; `CELL` is set to the next not-yet-run item in the frozen
serial order and the next submission occurs only after the prior job is terminal:

```bash
SOURCE_SHA=$(git rev-parse HEAD)
APPROVED_SOURCE_SHA=d1789bba4804dfcdab4d26a5780a836e69b56355
CELL=l2-1
ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e2_${APPROVED_SOURCE_SHA:0:12}
mkdir -p "${ROOT}"
sbatch --account=naiss2025-22-1113-gpu --partition=gpu \
  --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:45:00 --no-requeue \
  --job-name="s10-p1p-${CELL}" \
  --output="${ROOT}/slurm_${CELL}_initial_%j.out" \
  --error="${ROOT}/slurm_${CELL}_initial_%j.err" \
  fl_v3/scripts/run_s10_phase1p_lidar_e2.sh \
  --cell "${CELL}" --source-sha "${SOURCE_SHA}" \
  --approved-source-sha "${APPROVED_SOURCE_SHA}"
```

Initial L2-1 engineering incident and authorized O-149 replacement:

```text
INITIAL_L2_1_JOB: 558224
INITIAL_SOURCE_SHA: 50a95c451f6db696382398f58ef20bd3d709fbd1
INITIAL_SOURCE_TREE: edcefedadb673a9343a573b674bbed932d9bb488
NODE / RESOURCES: n30 / 1xGH200 / 16 CPU / 96 GiB / 00:45:00 /
  no restart or requeue
SUBMIT / START / END / ELAPSED: 2026-07-22T10:40:18 /
  2026-07-22T10:40:18 / 2026-07-22T10:42:20 / 00:02:02
SLURM_STATE: FAILED 1:0 in mandatory preflight; no D_fit profiler process ran
CHARGE: 0.033889 charged GH200-hours, code-bug reserve
PRETEST: 11/13 PASS; FP32 SDPA/target, sparse-offset and fused-AdamW checks PASS;
  only the two synthetic FP16 accepted-update fixture tails failed
DIAGNOSIS: the target fixture incorrectly passed FP16 autocast output leaves to
  unfused AdamW, whose 1e-8 epsilon underflows in half state and manufactured NaNs;
  the SDPA forward/backward checks passed, while its one-step Adam sign response at
  near-zero gradients produced a finite 0.000388 max parameter delta that the
  latest diagnostic-only non-deterministic-kernel policy does not make a hard gate
CLASSIFICATION: unambiguous test-fixture bug; production Phase-I model parameters
  and Adam states remain FP32, and neither candidate source nor frozen science ran
REPAIR_SHA: c8b23fd8b8b9cf352136edf8d51e17e3cf649796
REPAIR_TEST_SHA256:
  608460fe197df136cc143b17fa12794c1eb9125263ee3a5b302a5b75af4dd219
REPAIR: target accepted-update proxies now mirror FP32 master parameters; SDPA keeps
  its already-passing FP32/FP16 forward/backward tolerances and requires finite,
  structurally complete accepted Adam states rather than elementwise update veto
REPLACEMENT_AUTHORITY: smallest frozen-semantics O-149 fixture repair; submit L2-1
  serially with unchanged profiles/candidates/resources/gates and a fresh derived
  source-SHA-qualified output; no identical retry
REPLACEMENT_SOURCE: exact clean containing incident-ledger commit
BUDGET_AFTER_INITIAL: base 0/2.00; bug reserve 0.033889/0.75;
  aggregate 0.033889/2.75 charged GH200-hours
STDOUT_SHA256: fe3b258324425f4000b4a7ec23d5ebaa2b362ad48d3d89d6bf866b3e20557c15
STDERR_SHA256: 8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830
```

Terminal L-WP2 ledger:

```text
TERMINAL_SOURCE_SHA: be904876e3b42f21ce04fe453c0bb1283acb237b
APPROVED_SOURCE_SHA: d1789bba4804dfcdab4d26a5780a836e69b56355
SOURCE_CONFIG_SHA256: 0efe4d6d5138e3d99ae80254a6ecf884300dd18985ab45a00425228fc3ef082e
PRETESTS: every terminal paired job 13/13 PASS; two non-actionable dependency
  warnings only
GLOBAL_HEALTH: ten COMPLETE_SUSTAINED results; 2,560/2,560 measured optimizer
  windows and 81,920/81,920 presentations accepted; zero invalid, nonfinite,
  overflow, discard or scaler skip; every loss/component, measurement and
  checkpoint-continuation hard gate PASS; every same-batch input anchor exact
DATA/CLAIM_BOUNDARY: D_fit/CBGS/seed-0 only; D_select=false, D_audit=false,
  official-validation=false and capability-metrics=false in every result
LOSS_HEALTH: all ten first-quarter total-loss means 42.7995-44.3944 and
  last-quarter means 6.3637-6.7795; bounded health only, not convergence/capability
MEMORY: maximum reserved fraction 0.531312 in SDPA; every process below 0.85 with
  no monotonic growth; compile candidate reserved fraction 0.409046

L2-1 / JOB 558796 / NODE n30 / ELAPSED 00:16:35 / CHARGE 0.276389 base:
  reference 51.564006539 -> target/Hungarian 54.843411113 presentations/s;
  ratio 1.063598715; one-sided lower bound 1.040014880; POSITIVE; all hard gates PASS
  reference result 2e3a87105957d7d49e6089bf34b5e690f020fc2da0266323785758ff92d7d03d
  candidate result 7f805b551ed1edb6d6a8c0545471a6b077ee3d4c16fc00f8768defcc8e3b77da
  pair 42acba262b7e1fce09c86277a168f04a636d28601aaa036763e38ae9eb63b636
L2-2 / JOB 559450 / NODE n422 / ELAPSED 00:17:56 / CHARGE 0.298889 base:
  reference 54.221679924 -> SDPA 50.139856686 presentations/s;
  ratio 0.924719720; lower bound 0.920839040; NEGATIVE; all hard gates PASS;
  exact two-core/dropout/RNG scope recorded; max reserved fraction 0.531312
  reference result 27c216faad372601d16df0a847b40b91fd9bdafa1200ef1f9b0e5b4a99859249
  candidate result a36ad4c59b6426a2d67d6c5caecd381356a4cc3a7a5a671e809086cca45e29b7
  pair 7d680007ec2282ea027e7b3e0d2b985c85fd6c08b296b6dbca99c5cf582a394e
L2-3 / JOB 559566 / NODE n420 / ELAPSED 00:19:06 / CHARGE 0.318333 base:
  reference 50.918700099 -> scoped compile 55.728925044 presentations/s;
  ratio 1.094468730; lower bound 1.089391975; POSITIVE; all hard gates PASS;
  exactly decoder_backbone/decoder_neck/head, 107.274 s cold warm-up and no
  measured-interval recompile
  reference result 9f0489ef03a6c34733b6e819672d167928b33c468b938ea5aca7740493e2ef52
  candidate result b7c13b325413eeb20d157f11c6af3883faecb9890842f24c1342db500299e197
  pair 617ac9c23e619547e0474eb55f4d156d68d569400fd6ee0899b94142bfd5d4bb
L2-4 / JOB 559612 / NODE n52 / ELAPSED 00:17:40 / CHARGE 0.294444 base:
  reference 51.465626804 -> host offsets 55.470067427 presentations/s;
  ratio 1.077808061; lower bound 1.073928980; POSITIVE; all hard gates PASS
  reference result 5d658a821ec26195add3537d1dbf32ad0ebbd0c60d12fe85048ce7e8e0dbe22a
  candidate result 4064638d0086fc1afd97e2b7e4cd5bb19a13887909be38d193b316d37fa018cd
  pair e02a438400727c6bc57d9c31275bc7cafebcad80c677363e67e26eb548ebd32b
L2-5 / JOB 559662 / NODE n77 / ELAPSED 00:17:30 / CHARGE 0.291667 base:
  reference 55.175985793 -> fused AdamW 53.880671624 presentations/s;
  ratio 0.976523951; lower bound 0.972154208; NEGATIVE; all hard gates PASS;
  exact optimizer-group/state/checkpoint identity preserved
  reference result 8f1b8f680b21bdbeb67ea73417134ad9f1e0e2d63b5803746840107470c586fc
  candidate result ae1dfecd16adb6ff1ac89f37cfc461b87be6268a6319fc4e89212b828693d4d2
  pair c5607faaff6557e2499e1afd941e55e53bdc69c508e0e9c875581afc38d2630c

FINAL_BUDGET: base 1.479722/2.00; code-bug reserve 0.033889/0.75;
  aggregate 1.513611/2.75 charged GH200-hours; maximum concurrency one respected
TERMINAL_CLASSIFICATION: positive primaries are target/Hungarian host batching,
  scoped dense compile and sparse host offsets; SDPA and fused AdamW are negative
PROMOTION_STATE: none promoted or combined; no final LiDAR recipe claim
OWNER_RETURN: close IP-L-E2/L-WP2 and discuss exact L-WP3 composition at IP-LG2;
  no further compute, Camera/Fusion work or Envelope-B action is authorized
```

### 9.11 LiDAR IP-LG2 closure and IP-L-E3 activation

```text
REQUEST_STATE: TERMINAL POSITIVE / FINAL RECIPE MATERIALIZATION AUTHORIZED
OWNER_DECISION: 2026-07-22 — accept the three positive primaries and
  "批准，激活IP-L-E3"
APPROVAL_ANCHOR_SHA: 468a82bddda685fe81ece1fe0e59db35c50ba856
DERIVED_IMPLEMENTATION_SHA: 7c2b49d6cb7cc0008072530971b67123ea608748
DERIVED_IMPLEMENTATION_TREE: 1c4659bd01d9db7d628d5986a2407b3776dfddef
EXECUTION_SOURCE: exact clean containing activation-ledger commit, resolved at
  submission; it must be a linear descendant of APPROVAL_ANCHOR_SHA and
  UNIQUE_BASE_SHA
BRANCH: codex/s10-phase1p-throughput-preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
FROZEN_CONTROL: codex/s10-phase1-branch-qualification remains at UNIQUE_BASE_SHA

ACTIVATED_COMBINED_RECIPE: LiDAR-only B32xaccum1; enable exactly
  hungarian_batched_d2h, torch_compile and lidar_host_batch_offsets; compile only
  decoder_backbone/decoder_neck/head with Inductor/default/dynamic=false; keep
  lidar_sdpa=false, fused_adamw=false and every other candidate off
ABBA_ORDER: four fresh processes in one allocation/node/GH200:
  reference r1 -> combined r1 -> combined r2 -> reference r2
PROCESS_WINDOW: each process uses 16 accepted warm-up plus 256 accepted measured
  optimizer windows; one-second system sampling; 16-window blocks
STATISTICS: 50,000-draw fresh-process-stratified one-sided 95% block bootstrap
PROMOTION_GATE: every four-process loss/memory/checkpoint/runtime hard gate passes;
  pooled candidate/reference lower bound >1.00; both order-specific point ratios
  >=0.98; mean projected 20-epoch candidate cost is below reference after compile
  cold start and 20 checkpoint/hash stalls
TRACE_RULE: run one three-window combined stage trace only after the promotion gate
  passes; trace evidence ranks residual work and does not itself authorize another
  conditional implementation or profiler cell
CONTINUATION: a positive gate directly authorizes final LiDAR recipe
  materialization under the owner's IP-LG2 decision; a healthy non-positive result
  returns to the owner without subset search or retry

PROFILER_ENTRY_SHA256:
  452e220cd546abdf3bc5530a7af22d6d7c80e1d706ed4ac6d28bc7579e05df68
COMPARATOR_ENTRY_SHA256:
  7276d44ef5a229565495d4e6d79174d1c8303172403135d5b07f7268cd23d775
ABBA_LAUNCHER_SHA256:
  2c25643a24dcde84a4a15834a9c82400cde137a7a3aaae7accc423ce5faa33af
PROFILE_SCHEMA_SOURCE_SHA256:
  817b6f9ff1ec34cac563d4ee03d3161102993b139909e990499aea83680e9af5
ABBA_COMPARATOR_SOURCE_SHA256:
  4c8a435aa7447cf633ad8b4f031dd00987f7f5c889f6f31a47d3c0847c74203e
PROFILE_SHA256:
  reference 215b837e82f75766c0d53ba79594c3f0b047bf031ecdf301c63efc49cb82c2cc
  combined  52cb26f87dfe7a6cfa91d35466af9271eff75a49fa0df9fb972938baf288cd1b
LOCAL_VALIDATION: Python3 py_compile PASS; two JSON parses PASS; bash -n PASS;
  shellcheck PASS; synthetic pure-Python ABBA aggregate/gate check PASS;
  git diff --check PASS. Login Python lacks pytest/PyTorch, so the exact profile,
  checkpoint, Hungarian, sparse-offset and ABBA tests are mandatory in-job

RESOURCES_PER_JOB: one node; one GH200; 16 CPUs; 96 GiB; <=01:00:00; no requeue;
  maximum concurrency one
AGGREGATE_BASE: 1.00 charged GH200-hours
CODE_BUG_RESERVE: +0.50 charged GH200-hour
HARD_AGGREGATE_CEILING: 1.50 charged GH200-hours
SUBMISSION_POLICY: one serial ABBA/conditional-trace job; no numeric remediation
  submission cap; no blind retry; aggregate ceiling and concurrency are binding
OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/
  arrhenius_fl_v3/outputs/s10_phase1p_ip_l_e3_468a82bddda6/
FRESH_OUTPUT: source-SHA-qualified process/pair/trace paths; never overwrite raw data
STOP_ESCALATE: repeated or ambiguous blocker; hard scientific/health failure;
  requested candidate/gate/resource change; science-boundary pressure; or next
  charge could exceed 1.50 GH200-hours
OUT_OF_SCOPE: extra L-WP3 candidates, Camera/Fusion compute, D_select, D_audit,
  official validation, capability metrics, original/revised Envelope B activation,
  merge, push, upload and publication
BUDGET_AT_ACTIVATION: base 0/1.00; bug reserve 0/0.50; aggregate 0/1.50

INITIAL_ENGINEERING_INCIDENT: Job 559822 / source
  2d455acd84d1bf916b252c9a8d03cdd58ab71b41 / node n122 / FAILED 2:0 /
  elapsed 00:00:09 / no tests, data, model or optimizer execution
INCIDENT_EVIDENCE: request asked for one untyped --gpus-per-node=1, but Slurm
  allocated all four typed GH200 GRES on the otherwise idle node; runner correctly
  stopped at SLURM_GPUS_ON_NODE actual=4 expected=1
INCIDENT_CLASSIFICATION: unambiguous submission-command GPU-type plumbing bug;
  no science, candidate, gate or requested one-GPU resource change
INCIDENT_CHARGE: 4 allocated GH200 x 9 seconds = 0.010000 charged GH200-hours,
  assigned to code-bug reserve
DERIVED_REPAIR: replace only the ambiguous request with the historically validated
  --gpus-per-node=nvidia_gh200_120gb:1 spelling and restore explicit account/ntasks;
  retain the runner's exact one-visible-GPU assertion and every frozen E3 field
REPLACEMENT_SOURCE: exact clean containing this incident ledger; use fresh
  source-SHA-qualified process/pair/trace paths and a new Slurm log
BUDGET_AFTER_INITIAL: base 0/1.00; bug reserve 0.010000/0.50;
  aggregate 0.010000/1.50 charged GH200-hours

ABBA_REPLACEMENT: Job 559830 / source
  6105d70d81c1e0a496dec288e5b76217155783c5 / node n439 / FAILED 1:0 /
  elapsed 00:38:23 / exact one typed GH200 / 16 CPUs / 96 GiB
IN_JOB_PREFLIGHT: 10 passed, 2 warnings in 63.32 seconds
SUSTAINED_PROCESSES: reference A 55.776941; combined A 57.540962;
  combined B 61.247915; reference B 53.349064 presentations/s. All four
  completed 16 accepted warm-up plus 256 accepted measured windows and passed
  loss, memory, runtime and checkpoint-continuation gates.
ABBA_RESULT: POSITIVE_COMBINED_RECIPE; reference pooled 54.536029; candidate
  pooled 59.336641 presentations/s; ratio 1.088026; one-sided 95% lower bound
  1.085175; directional ratios 1.031626/1.148060; combined-recipe gate PASS.
PAYBACK: checkpoint stalls and compile cold start included; projected 20-epoch
  reference/candidate cost 8.976495/8.261479 GH200-hours; saving 0.715016.
ABBA_ARTIFACT: pairs/l3_abba_6105d70d81c1.json / SHA-256
  7f4844d65a031ce65e41c1f0b8ada946d582c40dc0b43467f77fc6fc2d9f9279
PROCESS_RESULT_SHA256_ABBA_ORDER:
  06c191d5837f2e462800ba20fc9e780a94d1260c5dfd01810b8a7ada47ce1b0c
  390dccd71f9848e864c91175995b0b343ce897f6c378740da3901b4bbde49ece
  2f20762e9a1fd019b578fe7fcacd903ebec7cd00a670eb018542f13612b4f59f
  51b989dc73d76d9954e64e2ba55f694bdbfd40e87505da15fadd6732a0894fc5

TRACE_PUBLISH_INCIDENT: after sealing the positive ABBA, the conditional trace
  completed its three windows but publication rejected missing eager internal
  head/decoder and reference-loss range names. PyTorch reported that
  record_function is ignored inside compiled regions; the active batched loss
  also intentionally emits different exact aliases. No sustained result, model
  health gate or ABBA statistic is affected.
FAILED_TRACE_RAW: lidar/trace_6105d70d81c1_r1_l3_combined_trace/torch_trace.json /
  SHA-256 fd40764917aaa4ba02defc3c24e9de052a1327653c42b79aac37a0473e2f1d56 /
  170089517 bytes; text summary SHA-256
  5a2f14bc4e976bdbb43f565113ee5280ea91026b008de4c96548d0c469266a4b
TRACE_INCIDENT_CLASSIFICATION: unambiguous profiler-instrumentation compatibility
  bug under the already frozen compiled/batched candidate; no model, math, data,
  candidate, acceptance-gate or scientific interpretation change.
TRACE_REPAIR_SHA: 9cf934020b703c8d179e33522d1455cc2da3c1ae / tree
  200844afd25bc64e25b6ddace97f4eee92601331
TRACE_REPAIR: compiled mode requires observable top-level
  second_backbone/second_fpn/transfusion_head parents instead of suppressed
  internal compiled ranges; batched mode requires its exact validation/D2H/H2D/
  Gaussian aliases and records eliminated eager sync ranges. Eager/default policy
  is unchanged. The replacement runner hash-validates and reuses the sealed ABBA,
  forbids rerunning the four sustained processes and writes a fresh trace-only path.
TRACE_REPAIR_SHA256: profiler
  6e77ca9ce1bb8033a906d3636c96786b4523e5e60c2faaa1806ac1bd6a22b027;
  runner d7d4b23bbcb071d1adb296aa0068959a3e3b27f94b1ad38f8bd5cee8ccae3754;
  focused test 594af3382fd3dcbb0bb61a2386801de9b8178b5e3c581988b161671cfb309682
TRACE_REPLACEMENT_SCOPE: one fresh three-window combined trace only; exact typed
  one-GH200 resource; <=00:20:00; no requeue; charge to diagnosed-code-bug reserve.
BUDGET_AFTER_JOB_559830: base 0.639722/1.00; bug reserve 0.010000/0.50;
  aggregate 0.649722/1.50 charged GH200-hours

TRACE_REPLACEMENT: Job 560200 / source
  2b93d0d1263bf4389dd2601d1848e44386db1641 / node n33 / COMPLETED 0:0 /
  elapsed 00:05:22 / exact one typed GH200 / 16 CPUs / 96 GiB
TRACE_RUNTIME_RESULT: 2/2 focused tests PASS; positive Job-559830 ABBA hash reused
  and attested; three accepted B32xaccum1 combined trace windows; COMPLETE_TRACE;
  all measurement-health checks PASS; no missing core range; compiled parent and
  batched Hungarian aliases present; no unexpected steady-state recompile.
TRACE_ARTIFACT_ROOT:
  lidar/trace_2b93d0d1263b_r1_l3_combined_trace_repair
TRACE_ARTIFACT_SHA256: result
  586720cd14ae17f4baedae0043dea5f23afd20195a67e8a10d6a987cd0d7ca4b;
  measurement d708b283a6ba0d6a88623c1fe20464d3a8d1ddf035aa6f182089b529b6457ea6;
  complete 33d21c3634b9cbe97a75f52e3089e4c633b616abc32b398bd5b8efe962c2a3f4;
  raw trace 7937ed1c3b7cbdeb23a3f25a87917d31300ff57cfddb3b28ba938dc48ef5c373;
  structured summary 79fe502e26cbcbb0394763c654248d302df51115f4a6631b0f7fb8ccf535fcc2;
  text summary ea438dca29d0fcae2b6b19a8269bb13d76d52abd34920abdfb164fade19884d3
TRACE_RANKING: largest named forward range is voxel/VFE/sparse-collapse. Residual
  target work ranks Hungarian GPU cost construction and batched host Gaussian
  target generation above the now-small batched D2H/SciPy wrapper. Nested trace
  totals are trace-inflated ranking evidence only and do not quantify additive
  end-to-end savings or authorize another candidate.
TERMINAL_DECISION: positive combined gate plus complete trace directly authorizes
  final LiDAR B32xaccum1 production-recipe materialization with exactly
  hungarian_batched_d2h=true, lidar_host_batch_offsets=true and scoped
  torch_compile=true; lidar_sdpa/fused_adamw remain false. No further E3 GPU cell.
PRODUCTION_RECIPE_IMPLEMENTATION: 003950166df8564a9257d7de008f4d7628836bea /
  tree f63d29b53eb07e131ee7a6fb081669558136c177
PRODUCTION_RECIPE: schema s10.phase1.v5; B32xaccum1/world-size 1; ordinary
  physical-B32 BatchNorm; seed+epoch worker RNG; batched target/Hungarian=true;
  CPU lidar_point_offsets=true; compile only decoder_backbone/decoder_neck/head
  with Inductor/default/dynamic=false; lidar_sdpa=false; fused AdamW=false;
  recovery checkpoint cadence remains one epoch.
PRODUCTION_CONFIG_SHA256: file
  683af022c053fcfcd39bbc0de4cc2753a2ba20021990347c7b19e94c0ff4838d;
  resolved a03ad08070a4081dac818965264df4fe5d27a8b76a256920f3b088e862554bf6
PRODUCTION_LOCAL_VALIDATION: config resolution and exact hash PASS; mutated B16,
  disabled batched-Hungarian, drifted compile scope and enabled SDPA rejected;
  historical B4 resolved identity reconstructed exactly; Python py_compile PASS;
  git diff --check PASS. Focused pytest is authored but not executed because the
  x86 login environment has no pytest/PyTorch and no extra Slurm job is authorized.
PRODUCTION_INTERPRETATION: materialization consumes the terminal E2/E3 parity,
  sustained, health and checkpoint evidence; it is not capability evidence and
  does not activate either the historical or revised Envelope B.
FINAL_BUDGET: base 0.639722/1.00; bug reserve 0.099444/0.50;
  aggregate 0.739166/1.50 charged GH200-hours
```

Exact submission template; `EXECUTION_SOURCE` is the clean activation-ledger SHA:

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --job-name=s10-l-e3-abba \
  --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=01:00:00 \
  --no-requeue --output=<fresh-root>/slurm/job_%j.out \
  --error=<fresh-root>/slurm/job_%j.err \
  fl_v3/scripts/run_s10_phase1p_lidar_e3.sh \
  --source-sha "${EXECUTION_SOURCE}" \
  --approved-source-sha 468a82bddda685fe81ece1fe0e59db35c50ba856
```

Exact trace-only remediation template; `EXECUTION_SOURCE` is the clean containing
repair-ledger SHA and the runner must attest the positive Job-559830 ABBA hash:

```bash
sbatch --parsable --account=naiss2025-22-1113-gpu --job-name=s10-l-e3-trace \
  --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G \
  --gpus-per-node=nvidia_gh200_120gb:1 --time=00:20:00 \
  --no-requeue --output=<fresh-root>/slurm/job_%j.out \
  --error=<fresh-root>/slurm/job_%j.err \
  fl_v3/scripts/run_s10_phase1p_lidar_e3_trace_repair.sh \
  --source-sha "${EXECUTION_SOURCE}" \
  --approved-source-sha 468a82bddda685fe81ece1fe0e59db35c50ba856
```

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
