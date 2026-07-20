# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S00 / S10
ACTIVE_DECISION: O-148 engineering-smoke completion authority under O-143/O-144/O-145/O-146
REQUEST_STATE: APPROVED / ACTIVE
EXECUTION_AUTHORITY: O-148 at b86089904a732edaaea77a446267a764f2da7073
ACTIVE_PHASE: Phase I C/L independent recipe and capability — Envelope A WP0-WP4
              implementation complete; continuous engineering qualification active
PLAN: PHASE_I_PLAN.md / P1-G0 closed
BRANCH: codex/s10-phase1-branch-qualification
```

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

O-148 supersedes O-147's submission-count and per-engineering-failure stop mechanics
for the remaining WP4 smoke/qualification work. Engineering submissions are unlimited
in count, serial at concurrency one, and bounded by the unchanged aggregate ceiling of
1.10 charged GH200-hours. S00 diagnoses an engineering failure, makes the smallest
output-/science-neutral repair, records provenance, and immediately resubmits without
another owner approval until Camera Job A and LiDAR Job B pass. Checkpoint I/O, tests,
fixtures, runner preflight, logging, build plumbing, and other non-scientific defects
are inside this remediation loop. Model math, data ownership/content, candidate/config/
seed, precision policy, evaluator/metric semantics, correctness tolerances, performance
gates, or any capability/scientific run remain outside it and require owner escalation.
The collaboration contract will be formally consolidated after both WP4 jobs complete.

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
EXECUTION_AUTHORITY: Envelope A engineering completion active under O-148;
                     Envelope B remains unauthorized
```

The complete graph, optimizer/scheduler, augmentation, role-bound GT-paste,
evaluation, remediation and amendment rules are normative in
`PHASE_I_PLAN.md`. This record does not duplicate them.

## 6. Envelope A — approved engineering completion under O-148

This section is the complete approval object. The owner's approval binds the Git commit
containing this section; deterministic `<REQUEST_SHA12>` and `<IMPLEMENTATION_SHA12>`
path substitutions do not require another approval. After approval, S00 records that
same approval and executes WP0 through WP4 continuously without per-WP owner stops.

```text
PHASE: S10 Phase I / Envelope A implementation and engineering calibration
REQUEST_STATE: APPROVED / ACTIVE under O-148
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
MAX_SUBMISSIONS: unlimited for serial WP4 engineering smoke under O-148
AGGREGATE_GPU_HOURS: <=1.10 charged GH200-hours across all submissions
JOB_A: Camera extension build, correctness gates, operator timing, optimized/fallback
       end-to-end calibration
JOB_B: D_fit keyframe GTDB materialization and identity checks, then LiDAR 16-warm-up /
       64-timed-B4 production calibration
JOB_C: not planned; at most one fresh-output derived replacement for one diagnosed
       in-scope engineering failure, only if aggregate time remains
JOB_D: O-147-authorized fresh-output Camera replacement from exact commit c45e020...;
       no replacement if it fails
```

Under O-148, Camera is repaired and resubmitted serially until PASS, then Job B is
repaired and resubmitted serially until PASS. Engineering defects do not stop for
per-job approval. Exhausting 1.10 aggregate GH200-hours, uncertain engineering versus
scientific classification, checkpoint/license/content failure that cannot be repaired
without changing the frozen science, correctness/performance-gate failure requiring
changed math or tolerances, or any scientific scope drift still stops at the owner boundary.

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

## 8. Envelope-A compact execution ledger

This is the sole active ledger for Envelope A. Submission rows are appended only when
the exact durable source SHA and command are known.

| Item | Durable source / identity | State | Resources / interpretation |
|---|---|---|---|
| WP0 | `714f7a1067f375861c80e3020ab302a928983f12` | complete | local/static only; no compute |
| WP1 | `933ca6feb142bcedc2ab842b25d6a1caf242c749` | complete | exact CBGS artifact `64cc0d1d...e115ef`; no GPU submission |
| Swin acquisition 1/1 | source URL in Section 6.2; 114,342,173 bytes; SHA-256 `9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3` | quarantined / not accepted | HTTPS 200; final host `release-assets.githubusercontent.com`; no GPU-hour charge; final path absent |
| WP2 implementation | `5a001c96f00fffd0816492f181197e2d310a5ae1` | implemented; WP4 qualification pending | login syntax/static checks only; no capability inference |
| WP3 implementation | `22138371d28e75d5218b0b888c225953fd429f0c` | complete; WP4 qualification pending | exact collapsed sparse boundary + SECOND/SECONDFPN/TransFusion; login syntax/static checks only |
| WP4 implementation | `4c13ad736319c022d7fb6466a48a77c90ae79dde` / tree `af1a582488191b0e49799ebc02b9489990ce0edf` | implemented; execution stopped before qualification | exact zero-update calibrator, checkpoint/evaluator preflight, production-input pooling parity/timing, fail-closed Job A/B runners |
| WP4 checkpoint-I/O remediation | `67c1b55b59aa81a49b1ed8f4aabd07e6592e88aa` / tree `2c8812f57c3e59fce25ad1d6f3dd63044b39c714` | locally sealed; GH200 verification remains unexecuted because Job D stopped before pytest | scalar/N-D raw-byte hashing plus 0-D BatchNorm-buffer regression; no model/data/config change |
| O-148 preflight observability remediation | `125e915a0f16f8abfbfa14d73558ee518cf3170c` / tree `34840210a9d426c51973a29af4be91f06c5fe9f6` | locally sealed; Camera Job E pending | names every fail-closed source/hash/module/environment/resource stage; no model/data/config/gate change |
| Job A / `521859` | `4c13ad736319c022d7fb6466a48a77c90ae79dde`; config `f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d` | `FAILED 1:0` in focused test; engineering incident, no model/data execution | `00:01:42` = `0.028333` GH200-hour; 27 passed / 1 pooling parity failure |
| Job C / `521901` Camera derived replacement | remediation `564fb9d97c44a463ac055dc40d25b79acdc77858` / tree `a1b9f7e809708b72a927afa4ef9c3f4bae82e137` | `FAILED 1:0` in checkpoint hash; engineering incident, no checkpoint promotion/model/data execution | `00:01:48` = `0.030000` GH200-hour; pooling focused tests 29/29 passed |
| Job D / `521959` Camera O-147 replacement | `c45e020ed16496e2acaa5f8d34b135da21fb1230` / tree `3887d82545207ec67b861bf48ff49042f52cebdb`; config `f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d` | `FAILED 1:0` before runner control/output creation; exact pre-control predicate unlocalized | `00:00:04` = `0.001111` GH200-hour; no pytest/checkpoint/data/model/build/calibration execution; Job B blocked |
| Job E / `522037` Camera O-148 smoke | `125e915a0f16f8abfbfa14d73558ee518cf3170c` / tree `34840210a9d426c51973a29af4be91f06c5fe9f6`; config `f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d` | `FAILED 2:0` in explicit hash preflight; command-construction incident | `00:00:05` = `0.001389` GH200-hour; expected checkpoint-entry hash was truncated to 62 characters; no output/model/data execution |
| Job F / `522042` Camera O-148 smoke | same exact Job E source/config with mechanically derived 64-character file hashes | `FAILED 1:0` after checkpoint acceptance in resolved-config evidence write | `00:01:58` = `0.032778` GH200-hour; 30 tests passed; checkpoint accepted once; canonical bytes were incorrectly written with an extra newline before physical-hash comparison |

Before O-148 execution, Envelope-A Slurm usage is `3 / unlimited` submissions and
`0.059444 / 1.10` charged GH200-hours. Jobs `521859`, `521901`, and `521959` are
consumed. Camera remediation/resubmission and then original Job B are active.

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
