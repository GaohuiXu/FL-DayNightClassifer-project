# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S00 / S10
ACTIVE_DECISION: O-144 under O-143
REQUEST_STATE: DRAFT / NOT APPROVED
EXECUTION_AUTHORITY: none
ACTIVE_PHASE: Phase I C/L independent recipe and capability — plan frozen, envelopes inactive
PLAN: PHASE_I_PLAN.md / P1-G0 closed
BRANCH: codex/s10-cl-model-recipe
```

O-143 authorizes the scientific/collaboration rebaseline only. It does not
authorize implementation, checkpoint acquisition or Slurm execution.
O-144 freezes the scientific/workflow plan in `PHASE_I_PLAN.md` but likewise
authorizes no implementation, checkpoint acquisition, GTDB materialization,
commit, or execution. Envelope A must be explicitly activated next; Envelope B
requires the later measured `P1-G1` approval.

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

## 5. Phase I O-144 plan-freeze record

```text
PLAN_STATE: OWNER_FROZEN / O-144
PLAN_PATH: handoffs/S10/PHASE_I_PLAN.md
P1_G0: CLOSED
CANDIDATES_AND_MAX_COUNT: exact Camera ImageNet primary + exact LiDAR scratch primary; max 2
DATA: D_fit train; D_select terminal development assessment; D_audit owner-sealed
SEED_POLICY: seed 0
EXPOSURE: 20 exact-CBGS epochs; physical B4; accumulation 8; effective B32
CHECKPOINT_SELECTION: epoch-20 terminal only
WORKFLOW: 5 WPs + 3 owner gates + 2 approval envelopes
EXECUTION_AUTHORITY: none
```

The complete graph, optimizer/scheduler, augmentation, role-bound GT-paste,
evaluation, remediation and amendment rules are normative in
`PHASE_I_PLAN.md`. This record does not duplicate them.

## 6. Envelope A — draft / not activated

```text
PHASE: S10 Phase I / Envelope A implementation and engineering calibration
REQUEST_STATE: DRAFT / NOT APPROVED
OBJECTIVE_AND_EXIT_GATE: implement WP0-WP4; materialize exact identities; calibrate C/L;
                         freeze one joint-review SHA; no capability result
SCIENTIFIC_CANDIDATES: none executed
ENGINEERING_GPU_DESIGN: one GH200; <=1.0 aggregate GH200-hour; <=3 submissions;
                        <=30 minutes/submission; max concurrency 1
CALIBRATION: per branch 16 warm-up + 64 timed physical-B4 microbatches
ALLOWED_METRICS: loader/step timing, samples/s, memory, init and accepted-window health only
D_SELECT_D_AUDIT: forbidden
IMPLEMENTATION_COMMIT_AUTHORITY: pending owner activation
CHECKPOINT_ACQUISITION: exact official ImageNet-1K Swin-T only; pending owner activation
DATA_MATERIALIZATION: exact D_fit CBGS and D_fit-only GTDB only; pending owner activation
OUTPUT_ROOTS: pending exact request
ENGINEERING_REMEDIATION_ALLOWED: pending exact request; output-neutral only
OWNER_APPROVAL: pending
EXECUTABLE_NOW: no
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
