# S09 independent review — terminal record

## Final verdict

```text
FINAL_REVIEW_SEAL: ced5992ea113bd21d7d545af505debf405b556b3
VERDICT: PASS_WITH_RESIDUAL_RISK
OPEN_P0: none
OPEN_P1: none
OPEN_P2: none
OPEN_P3: none
OWNER_CLOSE: O-120
INTEGRATED_CLOSING_COMMIT: 351b7a0b8419c01d0d32ba224babbc6bdc4213ba
```

Independent reviewers read exact diffs, frozen requests, resolved configs,
source/data/dependency identities, scheduler records, immutable outputs and raw
artifacts. They did not fix implementation source or submit compute. Review was
linear; remediations were reviewed before later conditional jobs were released.

## Review chronology

| Gate | Immutable evidence/remediation | Final disposition |
|---|---|---|
| STOP-1 cache | first evidence `b35591b`; remediation `5252a591983abb0013f19547e1d6ad20d3d6661f` | PASS_WITH_RESIDUAL_RISK; no open P0-P3; owner accepted O-113 |
| STOP-2 implementation | `37aef4d6b3f4679d6702d0acef2bb5bd1b57a952` | PASS_WITH_RESIDUAL_RISK; no open P0-P2 |
| STOP-2 frozen request | `cad72621e0e3ba409ae19bb0b62829118134b2d0` | PASS_WITH_RESIDUAL_RISK; no open P0-P3 |
| STOP-2 evidence | `a67cdda`; remediation `79f87dc9accca700b5a46803d45c549b0305c6d1` | PASS_WITH_RESIDUAL_RISK; no open P0-P3; owner accepted O-116 |
| STOP-3 first failure | `4fc78d508d4ac9ad7c46b9d3ad81c87646f8f0d3` | accepted terminal negative evidence; no G100 claim |
| STOP-3 dependency attestation | `82a0e5315c9098056b6670afb490850cc71dc653` | PASS_WITH_RESIDUAL_RISK; strict Phase-B GO |
| STOP-3 G100 | `c28d09c34b0ff56fcbc3805a8361ccd26eeaccc1`; closure `84adfd0` | PASS_WITH_RESIDUAL_RISK; no open P0-P3; owner accepted O-119 |
| STOP-4A implementation/request | remediation `b509f5e527c2dd28d2db506c3f87b5a06b3b1b6a` | PASS_WITH_RESIDUAL_RISK; no open P0-P3; exact-tuple GO |
| STOP-4A evidence + STOP-4B/C implementation | `6da4bb5`; closure `1a0b7e3` | PASS_WITH_RESIDUAL_RISK; no open P0-P3 |
| STOP-4C first request | `131619f` | SUBMIT NO-GO; never submitted |
| STOP-4C replacement/evidence | source `c776990`; evidence `32b380c`; remediation `8b7542c` | PASS; no open P0-P3; STOP-4D release GO |
| STOP-4D implementation/request | `5642884cdbb16e1c9b3107f529dc70b3a1243c6a` | PASS_WITH_RESIDUAL_RISK; exact request GO |
| STOP-4D final evidence | `54e45c5fabe0a353ddd904f19cca655351ac09d1`; closure `5e5d9a9d35e04711e30bbe98512dfcd0866929b2` | PASS_WITH_RESIDUAL_RISK; no open P0-P3; S09 owner-ready |

## Material findings and closure

### STOP-1

The initial evidence package named an impossible request commit and stale current
HEAD, while active environment prose still called cache materialization pending.
Documentation-only remediation bound the real source/request/job/cache identities
and updated active state. Review independently rehashed full train/val cache
contents, sidecars, ZIP manifest and immutable output. No payload/model/training
claim was introduced.

### STOP-2

Initial implementation review found:

- per-window raw `GradScaler.get_scale()` calls could synchronize CUDA;
- one candidate-template test still expected the rejected sparse-FP16 policy;
- normal `train_eval` sampled readiness-only clocks; and
- negative readiness lifecycle behavior lacked direct tests.

Remediation removed scaler polling and unused normal-path clocks, aligned the
O-110 assertion and added fail-artifact/lifecycle tests. The exact GH200 suite
then passed 44/44. One early residual—null start scale when every loss is
nonfinite—was explicitly documented and did not compromise terminal outcome/
counter reporting.

### STOP-3

Review confirmed Job `441511` failed before data/model because of the runner's
module selector, and did not reinterpret zero GPU work as model utilization.
Because the failed editable import changed cumm native identity, O-118's separate
attestation was necessary. Review verified Phase A's stable source/build manifests,
then the strict Phase-B derivation without changing data, model, precision, recipe,
resource or gate.

For Job `446225`, review independently reconciled:

- all eight loader repeat digests and worker throughput;
- 100 successful updates in 103 attempts;
- three initial scaler overflows and 90/90 post-warm accepted windows;
- exact optimizer/scheduler/exposure/EMA and sample accounting;
- pairwise combined timing rather than sums of independent percentiles;
- memory and epoch estimates; and
- immutable artifacts and unique scheduler history.

Two documentation-only P3 issues were closed; no model rerun was required.

### STOP-4

Initial STOP-4A review found three P2 plus one P3 in profiler/config/lifecycle
plumbing; remediation closed them before Job `452520`. Review confirmed all
four capacity cells start fresh and that B2/B4 are capacity evidence only.

Source/trace review proved exactly one durable redundant-sync site: 19 per-task
loss `.item()` calls in an ordinary attempted window. Tests establish exact
loss and loss-input-gradient equality when those host scalars are suppressed.
Swin activation checkpoint-off is explicit; measured G100/G1000 behavior
discloses its memory cost. No speculative allocation/sparse/target/model rewrite
was accepted.

The first STOP-4C runner lacked frozen performance-gate enforcement and was
therefore SUBMIT NO-GO. The replacement added fail-closed gates without changing
model/data/recipe semantics and was independently reviewed before Job `455539`.
Its PASS released the conditional fresh G1000. Final review independently
recomputed 1000/1003 accounting, 990/990 post-warm acceptance, p50/p95,
throughput, stage timing, epoch estimate, memory and coarse utilization from raw
artifacts.

## Final adversarial checks

The terminal review confirms:

- exact accepted train/val `t1.v2`, ten-sweep cache and ZIP-manifest identities;
- accepted S08 global-FP16/SECOND-FP32 partition;
- one flat AdamW group at `1e-4/0.01`, constant scheduler, no EMA/clip/
  augmentation/GT paste, seed 0 and B1 for the G100/G1000 evidence;
- no output/loss/gradient/update semantic change from quiet telemetry;
- no hidden retry, array, DDP, extra seed, worker matrix, metric, checkpoint or
  official evaluation;
- all failed/NO-GO jobs remain visible; and
- every declared checksum manifest and read-only output is internally consistent.

## Residual risk and interpretation

S09 is single-seed bounded engineering evidence. It does not establish
convergence, recipe/batch quality, model quality, mAP/NDS, fusion gain, multi-seed
capability or full GH200 utilization. One-Hz telemetry and coarse stage ranges
cannot identify a specific kernel/branch bottleneck. Cross-job variation cannot
separately attribute total speedup to checkpoint removal versus scalar-sync
removal.

The true unscaled SECOND gradients remain large and unexplained. Stable GradScaler
backoff does not shrink them. B4's full-epoch tail is unresolved. The training
recipe is still a flat optimizer group plus constant scheduler and is not
scientifically selected.

O-120 accepts these residuals and closes S09. O-121 integrates `351b7a0` but
accepts only the S10 work definition; no stop design, full-run placement, compute
or S11+ role follows.

The complete 1823-line pre-compaction review chain is recoverable at Git object
`351b7a0b8419c01d0d32ba224babbc6bdc4213ba`. This compact file is the active
terminal review record.
