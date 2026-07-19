# S10 RUN_REQUEST — phase authority and job ledger

## 1. Current authority

```text
SESSION: persistent S00 / S10
ACTIVE_DECISION: O-143
REQUEST_STATE: DRAFT / NOT APPROVED
EXECUTION_AUTHORITY: none
ACTIVE_PHASE: Phase I C/L independent recipe and capability — design pending
BRANCH: codex/s10-cl-model-recipe
```

O-143 authorizes the scientific/collaboration rebaseline only. It does not
authorize implementation, checkpoint acquisition or Slurm execution.

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

## 5. Phase I pending request

No values are frozen yet. Before compute, owner and S00 must decide:

- coherent camera and LiDAR candidate bundles and initialization sources;
- training recipe, horizon and checkpoint selection;
- aligned internal capability gates and Alvis comparator audit;
- seed policy;
- aggregate GH200-hour and submission ceilings;
- engineering stop/escalation rules.

```text
REQUEST_STATE: DRAFT / NOT APPROVED
EXECUTABLE_NOW: no
```
