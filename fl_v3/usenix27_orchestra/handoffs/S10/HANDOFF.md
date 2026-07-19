# S10 HANDOFF — O-143 scientific and collaboration rebaseline

## 1. Current state and authority

```text
SESSION: persistent S00 / S10
BASE_SHA: a080d49c1c22de20ccb5b1353d4922c7df14a729
BRANCH: codex/s10-cl-model-recipe
ACTIVE_DECISION: O-143
SCIENCE_ORDER: C/L independent recipe+capability -> staged fusion -> capability gate -> profiler
CURRENT_AUTHORITY: planning/documentation only; no implementation or compute
MERGE/PUSH/UPLOAD/PUBLICATION/S11+: not authorized
```

O-143 supersedes the active six-stop execution order and S10's per-job
immutable/no-retry/multi-document/reviewer mechanics. It does not erase prior
evidence, change STOP-A data ownership/evaluator semantics, weaken metric or
provenance requirements, or authorize compute.

Current-A2 and the old C→D→E→F route are paused. The primary S10 claim remains
**absolute clean capability + fusion contribution**, but it must now be earned
through independently qualified branches followed by staged fusion.

## 2. Accepted and bounded evidence

| Evidence | Accepted fact | Must not be inferred |
|---|---|---|
| STOP-A / Job `468404` / remediation `b0478a2` | train-only scene/log-disjoint split construction, independent ownership check and evaluator parity closed `PASS_WITH_RESIDUAL_RISK`; the resulting D splits are reusable | model capability, recipe quality or global balance optimality |
| STOP-B / Job `479667` / review `02ba3b4` | camera stochasticity and LiDAR runtime variation were both observed | cause of large LiDAR gradients; STOP-B closed `INCONCLUSIVE` |
| C0-v2 / Job `496312` | bounded B4 trajectories were numerically healthy; only four initial scaler overflows; internal single-seed F-minus-L was +0.029576 mAP / +0.033423 NDS | production convergence, architecture/recipe selection, official-val or full fusion claim |
| C1-A / Job `502572` | on the fixed W0/panel, direct BN1d reduced fixed-VJP and normal-loss LiDAR-stem gradients on all batches; `LOCALIZED_NORM` | BN1d capability advantage or production promotion |
| C1-B0 / Job `504508` | GN and BN1d both completed 256 B4 updates; BN1d strongly reduced stem gradients and was about 1.41x faster | convergence or evaluator superiority |
| C1-B1 / Job `504921` | GN-B4: NDS/mAP 0.144475/0.061553, 1,538 updates, 8.4914 samples/s; BN1d-B4: 0.136705/0.053125, 1,537 updates after one first-window overflow, 12.1663 samples/s | fair winner selection because exposure differed and uncertainty was absent |
| BN1d-B8 / Job `505316` | 769/769 updates, zero overflow, 14.1569 samples/s; D_select NDS/mAP 0.078409/0.013024 | batch-size causality, capability acceptance or a complete tail evidence gate |

The bounded proxy scores are low and do not answer the owner's central question:
whether the upgraded detector is usable or improves on the historical Alvis
result. The Alvis comparator itself is not yet aligned/audited in this branch.

Exact prior jobs, raw paths, checksums and interpretation limits remain in
`RESULTS.md` and the historical sections of `RUN_REQUEST.md`. Those files are
archives; do not append duplicate narratives for routine future incidents.

## 3. Reusable STOP-A data/evaluator substrate

The accepted STOP-A train-only nested split remains the default substrate for
S10 recipe selection. Its data/cache/ZIP identities, scene/log ownership proof,
D_low/D_mid/D_select/D_audit membership, emitted hashes and evaluator-parity
artifacts are frozen in the accepted STOP-A result package.

Future phases may directly consume these artifacts. Any change to membership,
ownership, label-derived construction, evaluator semantics, class mapping or
metric implementation is a material scientific amendment requiring owner approval
and independent review. Official nuScenes validation remains held out from recipe
selection unless a future approved capability gate explicitly opens it.

## 4. Active scientific order

### Phase I — camera and LiDAR independent recipe/capability

Treat the modalities as separate training problems before fusion.

For each branch, define a small, coherent candidate set covering:

- graph and normalization;
- initialization: public ImageNet/NuImages or compatible published checkpoint
  versus random components, with exact license/source/tensor compatibility;
- optimizer parameter groups, LR, weight decay, schedule and warmup;
- augmentation, sampling/CBGS/GT-paste and EMA;
- physical batch, accumulation, exposure and checkpoint selection;
- a meaningful training horizon and aligned evaluator gate.

MIT BEVFusion's published branch pretraining, staged fusion and recipe choices are
strong external anchors. Start from a coherent reference-derived bundle. Do not
spend local compute re-proving published ablations unless the local implementation
conflicts or materially underperforms. At most one cause-directed repair per
branch should be opened by evidence; do not run a broad hyperparameter grid.

Step-level runs are only crash/numerical preflight. Capability requires meaningful
trainval-scale exposure and evaluation. Phase I exits with one reviewed camera
recipe/checkpoint and one reviewed LiDAR recipe/checkpoint, or an honest negative
result.

### Phase II — staged fusion and capability

Initialize from the qualified C/L checkpoints. Freeze/unfreeze stages and fusion
training scope must be declared before execution. Compare camera, LiDAR and fusion
under aligned data, classes, exposure, checkpoint-selection, metric and evaluator
semantics.

The gate must answer:

1. does the detector achieve useful absolute clean capability?
2. does fusion contribute beyond the qualified unimodal controls?
3. under a fair aligned audit, does the upgraded system improve on—or at least
   credibly match—the historical Alvis detector?

The final staged-fusion/full capability result requires independent review. A weak
or failed result is recorded; it does not trigger an unbounded tuning loop.

### Phase III — GH200 profiler and sustainable optimization

Begin only after Phase II capability passes and the graph/recipe is frozen.
Measure synchronization, coverage, throughput, utilization, memory and operator
cost before changing performance behavior. Optimizations must remain
output-/science-neutral and be requalified against the accepted capability result.

## 5. Observation-first in the new order

Observation-first now means:

1. run the coherent reference-led branch recipe without local model mutation;
2. inspect loss trajectory, update validity, gradient/update scale, checkpoint
   behavior and evaluator metrics over a meaningful horizon;
3. localize only a failure that is both reproducible and capability-relevant;
4. apply at most one cause-directed repair within the approved candidate cap;
5. judge the repair on optimizer behavior and capability, not gradient magnitude
   alone.

C1-A's `LOCALIZED_NORM` result is useful evidence for the LiDAR candidate set.
It is not by itself proof that GN prevents convergence or that BN1d is better.
The next LiDAR plan must connect normalization to real capability under an
appropriate branch recipe.

## 6. Simplified S10 collaboration contract

A future phase approval binds once:

- objective and exit gate;
- candidate set and maximum count;
- data splits, evaluator/metric and seed policy;
- training exposure and checkpoint-selection rule;
- aggregate GPU-hours, maximum submissions and concurrency;
- stop/escalation conditions and output root.

Inside that approved envelope, S00 may independently fix output-neutral defects
in tests, fixtures, runners, checkpoint I/O or logging and resubmit within the
same scientific/resource caps. S00 returns to the owner before changing model
math, data ownership/content, recipe candidate space, metric/evaluator, seeds,
candidate count, interpretation or aggregate resources, and when repeated
engineering failure exhausts the phase cap.

Active records are:

- this `HANDOFF.md`: compact current status, science plan and decision boundary;
- `RUN_REQUEST.md`: phase authority plus one concise job ledger.

Minimum per-run provenance is Git SHA, resolved-config hash, split, seed, command,
resources, output root, terminal state, checkpoint hash and metric hash. Raw
outputs remain immutable. Do not require detached snapshot copies, recursive
manifests, command-file/stdout hashes or duplicate write-ups unless a specific
high-risk boundary needs them.

Preflight with direct entry/config/checkpoint/one-batch checks. Broad historical
test suites, paired-statistics generation and report packaging should not occupy
the GPU training critical path unless scientifically necessary. A pre-model
runner/test failure is an engineering incident, not a scientific STOP failure.

Independent review is reserved for data/evaluator changes, each branch recipe
freeze and the final staged-fusion/full capability result. Ordinary runner bugs
do not launch reviewers. Commit at material implementation, phase-plan freeze and
phase-result closure, not after every incident.

The old C0/C1 diagnostic harness is frozen historical tooling. New capability
work should use the production `centralized_train.py`, standard checkpointing
and evaluator paths, extended only by the smallest required branch-mode seams.

## 7. Next owner discussion and current stop

Before implementation or compute, jointly freeze Phase I:

- exact camera and LiDAR candidate bundles and initialization sources;
- whether qualification is sequential or shares one aggregate budget;
- training horizon, evaluator checkpoints and capability thresholds;
- Alvis checkpoint/provenance/evaluator-alignment audit;
- seeds, aggregate GH200-hours, submission cap and escalation conditions.

No Phase-I implementation, checkpoint acquisition, Slurm submission, staged
fusion, profiler, merge, push, upload, publication or S11+ work is authorized by
O-143.
