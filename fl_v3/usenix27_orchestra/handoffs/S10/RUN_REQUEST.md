# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S00 / S10
ACTIVE_DECISION: O-146 Envelope-A activation under O-143/O-144/O-145
REQUEST_STATE: APPROVED / ACTIVE
EXECUTION_AUTHORITY: exact Envelope A at request commit e321aed749fd859c809199d52c30b2771dbef8b3
ACTIVE_PHASE: Phase I C/L independent recipe and capability — Envelope A WP0-WP4
              implementation and bounded engineering calibration active
PLAN: PHASE_I_PLAN.md / P1-G0 closed
BRANCH: codex/s10-phase1-branch-qualification
```

O-146 records the owner's exact activation of Envelope A at request commit
`e321aed749fd859c809199d52c30b2771dbef8b3`. S00 may execute WP0 through WP4
continuously within Section 6, including the bounded checkpoint acquisition,
data materialization, material commits and at most three serial engineering
submissions / one aggregate GH200-hour. Envelope B still requires the later
measured `P1-G1` approval.

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
MAX_SUBMISSIONS:
MAX_CONCURRENCY:
OUTPUT_ROOT:
ENGINEERING_REMEDIATION_ALLOWED:
OWNER_ESCALATION_CONDITIONS:
ALLOWED_INTERPRETATION:
FORBIDDEN_INTERPRETATION:
OWNER_APPROVAL:
```

Within an approved phase, S00 may derive commands/resolved configs and repair
output-neutral test, fixture, runner, checkpoint-I/O or logging failures, provided
the candidate/science/data/metric/seed/resource boundaries and submission cap do
not change.

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
PLAN_STATE: OWNER_FROZEN / O-144 plus O-145 optimized-pooling amendment
PLAN_PATH: handoffs/S10/PHASE_I_PLAN.md
P1_G0: CLOSED
CANDIDATES_AND_MAX_COUNT: exact Camera ImageNet primary + exact LiDAR scratch primary; max 2
DATA: D_fit train; D_select terminal development assessment; D_audit owner-sealed
SEED_POLICY: seed 0
EXPOSURE: 20 exact-CBGS epochs; physical B4; accumulation 8; effective B32
CHECKPOINT_SELECTION: epoch-20 terminal only
WORKFLOW: 5 WPs + 3 owner gates + 2 approval envelopes
CAMERA_POOLING: optimized in-tree CUDA/equivalent production backend plus labelled
                fallback and WP2/WP4 parity/policy/performance gates
EXECUTION_AUTHORITY: Envelope A active under O-146; Envelope B remains unauthorized
```

The complete graph, optimizer/scheduler, augmentation, role-bound GT-paste,
evaluation, remediation and amendment rules are normative in
`PHASE_I_PLAN.md`. This record does not duplicate them.

## 6. Envelope A — approved and active under O-146

This section is the complete approval object. The owner's approval binds the Git commit
containing this section; deterministic `<REQUEST_SHA12>` and `<IMPLEMENTATION_SHA12>`
path substitutions do not require another approval. After approval, S00 records that
same approval and executes WP0 through WP4 continuously without per-WP owner stops.

```text
PHASE: S10 Phase I / Envelope A implementation and engineering calibration
REQUEST_STATE: APPROVED / ACTIVE
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
EXECUTABLE_BEFORE_OWNER_APPROVAL: no; now executable only inside this approved envelope
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
MAX_SUBMISSIONS: 3 total
AGGREGATE_GPU_HOURS: <=1.0 charged GH200-hour across all submissions
JOB_A: Camera extension build, correctness gates, operator timing, optimized/fallback
       end-to-end calibration
JOB_B: D_fit keyframe GTDB materialization and identity checks, then LiDAR 16-warm-up /
       64-timed-B4 production calibration
JOB_C: not planned; at most one fresh-output derived replacement for one diagnosed
       in-scope engineering failure, only if aggregate time remains
```

There is no identical retry. A completed, validated GTDB from a timed-out Job B may be
reused by Job C; a partial/unsealed one may not. The same blocker recurring, exhausted
time/submissions, uncertain classification, checkpoint/license/content failure,
correctness/performance-gate failure requiring changed math or tolerances, or any scope
change stops at the owner boundary.

Envelope A exits after WP0-WP4 implementation, focused validation, exact checkpoint/
CBGS/GTDB identities, both calibrations, and one combined recipe-freeze review at one
durable SHA. It reports the measured Envelope-B resource tuple but runs no capability
metric, D_select, D_audit, official validation, 20-epoch training, or selectable model.

The single approval sentence is:

```text
批准激活 commit <REQUEST_COMMIT> 中的 S10 Phase I Envelope A，并按其中边界连续执行 WP0-WP4。
```

## 7. Envelope B — pending P1-G1

Envelope B is not yet a complete request. It will bind the two frozen candidates,
seed 0, exact data/evaluator identities, `N_cbgs`, accepted precision, 20-epoch
B32 exposure, terminal-only selection, measured aggregate GPU-hours, wall-time
segmentation, submission cap, output root and owner-escalation conditions.
LiDAR runs before Camera by default; `D_audit` resources may be reserved but the
data remain sealed until `P1-G2` explicitly opens them.

```text
REQUEST_STATE: NOT YET REQUESTABLE
BLOCKED_ON: Envelope-A implementation/calibration/identities and joint recipe review
EXECUTABLE_NOW: no
```
