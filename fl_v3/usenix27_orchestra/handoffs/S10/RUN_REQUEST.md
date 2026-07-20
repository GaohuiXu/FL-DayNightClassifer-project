# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S00 / S10
ACTIVE_DECISION: O-145 amendment under O-143/O-144
REQUEST_STATE: DRAFT / NOT APPROVED
EXECUTION_AUTHORITY: none
ACTIVE_PHASE: Phase I C/L independent recipe and capability — O-145 plan amended,
              Envelope A exact draft in progress and not activated
PLAN: PHASE_I_PLAN.md / P1-G0 closed
BRANCH: codex/s10-phase1-branch-qualification
```

O-143 authorizes the scientific/collaboration rebaseline only. It does not
authorize implementation, checkpoint acquisition or Slurm execution.
O-144 freezes the scientific/workflow plan in `PHASE_I_PLAN.md` but likewise
authorizes no implementation, checkpoint acquisition, GTDB materialization,
commit, or execution. Envelope A must be explicitly activated next; Envelope B
requires the later measured `P1-G1` approval.

O-145 adds the independent optimized CUDA BEV-pooling/equivalent-kernel requirement
to WP2 and its forward/backward, FP16/FP32-policy, operator-timing, and end-to-end
qualification to WP4. It also authorizes drafting this exact Envelope-A request and
the O-145 documentation commit. It does not activate Envelope A or authorize the
checkpoint download, implementation, data materialization, or GPU execution.

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
EXECUTION_AUTHORITY: none
```

The complete graph, optimizer/scheduler, augmentation, role-bound GT-paste,
evaluation, remediation and amendment rules are normative in
`PHASE_I_PLAN.md`. This record does not duplicate them.

## 6. Envelope A — O-145 exact draft / not activated

```text
PHASE: S10 Phase I / Envelope A implementation and engineering calibration
REQUEST_STATE: DRAFT / NOT APPROVED
DRAFT_BASE_SHA: c4f6d874a21b528bc7f3c187a44426992f5d101d
OBJECTIVE_AND_EXIT_GATE: implement WP0-WP4; materialize exact identities; calibrate C/L;
                         qualify optimized BEV pooling; freeze one joint-review SHA;
                         no capability result
SCIENTIFIC_CANDIDATES: none executed
DATA_ROLE: accepted STOP-A D_fit only / 34 logs / 494 scenes / 19,877 samples
D_SELECT_D_AUDIT_OFFICIAL_VAL: forbidden
SEED_POLICY: seed 0 for initialization/order checks; no seed comparison
TRAINING_EXPOSURE: engineering-only bounded microbatches; no epoch/capability exposure
CHECKPOINT_SELECTION: none; any engineering checkpoint is non-selectable
ENGINEERING_GPU_DESIGN: one GH200, 16 CPU, 96 GiB; <=1.0 aggregate GH200-hour;
                        <=3 submissions; <=30 minutes/submission; max concurrency 1
CALIBRATION: Camera optimized and fallback from identical state/order, each 16 warm-up
             + 64 timed physical-B4 microbatches; LiDAR 16 warm-up + 64 timed B4
ALLOWED_METRICS: pooling forward/backward error; dtype/finite/policy state; CUDA-event
                 operator time; loader/step/end-to-end time; samples/s; peak memory;
                 initialization and accepted-window health only
FORBIDDEN_METRICS: mAP, NDS, loss-based candidate selection, D_select/D_audit metrics
IMPLEMENTATION_COMMIT_AUTHORITY: pending owner activation; material linear WP commits only
CHECKPOINT_ACQUISITION: exact reference-YAML ImageNet-1K Swin-T only; pending activation
NUIMAGES_CHECKPOINT: excluded; no download or use
DATA_MATERIALIZATION: exact D_fit CBGS identity and D_fit-only keyframe GTDB only;
                      pending owner activation
ENGINEERING_REMEDIATION_ALLOWED: pending owner activation; output-neutral only and
                                 within aggregate submission/GPU-hour caps
OWNER_APPROVAL: pending
EXECUTABLE_NOW: no
```

### 6.1 Draft implementation boundary

After activation, S00 may execute WP0-WP4 continuously in the persistent worktree and
linear branch, including focused tests and material commits at the plan, shared recipe,
Camera, LiDAR, and production-integration/review boundaries. The allowed implementation
surface is restricted to:

- `fl_v3/pyproject.toml` and in-tree license/NOTICE records only as needed for the
  standalone extension build; no new network package dependency;
- new Phase-I configs under `fl_v3/configs/`;
- directly required code under `fl_v3/src/fl_v3/{config,data,models,training,engine,eval,utils}`;
- `fl_v3/src/fl_v3/models/ops/bev_pool/` (new) for the independent optimized CUDA
  pooling backend, fallback dispatch, and pinned Apache-2.0 attribution;
- directly required `fl_v3/scripts/{centralized_train.py,build_gt_database.py}` changes
  plus new `s10_phase1_*` entry/runner files;
- focused existing tests and new `fl_v3/tests/test_s10_phase1_*` tests; and
- the active S10/Orchestra records at material phase boundaries.

No `fl_v2/`, `fl_v3/collab/`, unrelated milestone artifact, Protocol-A/B, Fusion,
attack/defense, general environment rebuild, mmdet3d/mmcv runtime dependency, or broad
profiler scope is included. An implementation need outside this boundary returns to the
owner before editing.

### 6.2 Exact Camera checkpoint draft

```text
ROLE: Camera primary backbone initialization / ImageNet-1K Swin-T
REFERENCE_CONFIG: MIT BEVFusion camera/256x704/swint/default.yaml at
                  326653dc06e0938edf1aae7d01efcd158ba83de5
SOURCE_URL: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
LICENSE_RECORD: upstream Microsoft Swin-Transformer repository is MIT-licensed;
                verify no conflicting release-asset terms before use
DESTINATION: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/pretrained/swin_tiny_patch4_window7_224.pth
QUARANTINE: same destination plus .download.part until SHA-256/state-dict checks pass
REDIRECT_POLICY: HTTPS GitHub release redirect only; record final resolved URL
HASH_POLICY: hash quarantined bytes before atomic rename or model use; bind physical
             SHA-256 and tensor-load report in this ledger; any content drift fails closed
EXISTING_NONREFERENCE_CACHE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_home/hub/checkpoints/swin_t-704ceda3.pth
EXISTING_NONREFERENCE_SHA256: 704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b
SUBSTITUTION_POLICY: existing torchvision artifact is comparison/mapping evidence only;
                     it cannot replace the reference-YAML asset without owner amendment
ACQUIRE_DURING_DRAFT: no
```

The checkpoint URL is therefore not the MIT script's
`swint-nuimages-pretrained.pth`. NuImages remains outside the two-candidate Phase-I
envelope. Because the reference YAML publishes no trusted digest alongside the URL,
activation would authorize one quarantined acquisition; the bytes become usable only
after the physical hash, state-dict schema, strict tensor mapping, loaded/missing/
unexpected-key report, and initialization-state hash are recorded. No download has
occurred while drafting this request.

### 6.3 Data and output draft

```text
SPLIT_MANIFEST: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1/split/split_manifest.json
SPLIT_MANIFEST_SHA256: 7e84a1d4f4a099c31a1d5194f17ba77d278fdc92f52462e72517b494dc5223a8
ROLE: D_fit only / 34 logs / 494 scenes / 19,877 samples
DATA_ARTIFACT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_data_<ACTIVATION_SHA12>
ENGINEERING_OUTPUT_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_phase1_envelope_a_eng_<ACTIVATION_SHA12>
CUDA_BUILD_ROOT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/torch_extensions/s10_bev_pool_<IMPLEMENTATION_SHA12>
RAW_DATA_POLICY: read shared nuScenes ZIPs/caches only; no extraction or duplication
MATERIALIZATION: exact official-CBGS index identity plus D_fit-only keyframe GTDB;
                 record source-token proof, per-class counts, and all artifact hashes
```

`<ACTIVATION_SHA12>` and `<IMPLEMENTATION_SHA12>` are draft placeholders, not wildcard
execution authority. They must be replaced by the exact activation and executable
source SHAs before owner approval/submission. Output roots must be absent before their
first writer and raw outputs remain immutable.

### 6.4 Optimized BEV-pooling gate and submission design

WP2 uses an independent in-tree port of the pinned MIT `bev_pool` CUDA operation, or a
functionally equivalent kernel, without importing mmdet3d/mmcv. The existing
sort/cumsum or explicit segment-reduction implementation remains the labelled
reference/fallback oracle. The optimized backend cannot silently fall back in an
Envelope-B run.

Before timing, the exact production graph must pass geometry/rank/output-shape tests,
edge cases, forward-value and input/upstream-gradient comparisons in FP32 and under the
accepted FP16 policy, dtype/autocast/FP32-accumulation assertions, and finite-state
checks. Numerical tolerances must be frozen in the activation record before the first
GPU parity execution and cannot be relaxed after observing a failure.

The intended serial submissions are:

1. Camera extension build plus parity, optimized/fallback operator timing, and aligned
   B4 end-to-end calibration;
2. LiDAR production-path B4 calibration; and
3. no planned third job — reserved only for one diagnosed output-neutral replacement
   when sufficient aggregate GPU-hour budget remains.

Each job is one node/task, one GH200, 16 CPU, 96 GiB, at most 30 minutes, no requeue,
maximum concurrency one. Total charged allocation across all submissions may not exceed
`1.0` GH200-hour. Build/parity failure is an engineering incident, not a model result;
repeated root failure, exhausted caps, uncertain output-neutrality, or any requested
change to graph/math/precision/data/recipe/evaluator/candidate/resource scope stops and
returns to the owner. Capability metrics, D_select, D_audit, official validation,
20-epoch training, and scientific checkpoint selection remain forbidden.

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
