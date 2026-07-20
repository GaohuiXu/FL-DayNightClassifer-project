# S10 HANDOFF — Phase I-P IP-E1 active; Envelope B remains frozen

## 1. Current state and authority

```text
SESSION: persistent S10 Phase I-P throughput preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
BRANCH: codex/s10-phase1p-throughput-preflight
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at f1a2babda8dafd181b5a5144ab025a3f6be21cc2
ACTIVE_DECISION: owner-approved IP-E1 under O-143/O-149; O-150 remains the Phase-I control
SCIENCE_ORDER: Phase I-P engineering preflight -> owner disposition -> still-pending C/L qualification
PHASE_I_PLAN: PHASE_I_PLAN.md; P1-G0 PLAN_FREEZE closed
CURRENT_AUTHORITY: Section-8 IP-E1; WP0 runtime closed; owner parity disposition required
EXECUTION_STATE: submissions paused after LiDAR same-process/fresh-process parity evidence
MERGE/PUSH/UPLOAD/PUBLICATION/S11+: not authorized
```

IP-G0 intentionally inserted Phase I-P before the long Camera/LiDAR qualification
runs and authorized this isolated linear branch plus WP0 implementation. The owner
then activated IP-E1 at `85c6719e4b880b198d850e16b1418c230fa5c656` for WP0's
GH200 runtime close and continuous IP-WP1 -> strict IP-WP2 execution. IP-E1 does
not authorize scientific training, an evaluation role, merge, push, or movement of
the frozen control branch.
The Section-7 Envelope-B request in `RUN_REQUEST.md` is preserved verbatim as a
historical control and is not activated by Phase I-P.

### 1.1 Frozen Phase I-P workflow

The workflow is continuous where the decision is already engineering-only; its
five WPs and three gates are not independent mini-projects:

| WP | Inputs and work | Required output / acceptance | Continuous authority and stop |
|---|---|---|---|
| IP-WP0 measurement path | frozen C/L configs and production model/data/loss/AdamW/scheduler/scaler/checkpoint paths | accumulation-aware B4x8 profiler; separate low-overhead sustained and short trace modes; exact identities; every candidate default-off; local/focused checks | IP-G0 permits scoped implementation and linear commits; stop before scientific semantics or compute |
| IP-WP1 real baseline | exact D_fit, official CBGS/GTDB, seed 0, physical B4 x accumulation 8 | two-process sustained C/L baselines, whole-model trace, system/memory evidence and real checkpoint continuation | runs only inside approved IP-E1; stop on identity drift, nonfinite/discarded windows, unresolved instability or ceiling |
| IP-WP2 output-neutral work | WP1 bottleneck evidence plus the named plumbing shortlist | individual parity, accepted-update, checkpoint/resume and sustained-throughput evidence; safe items may be combined | may proceed continuously inside IP-E1 only for frozen strict-output-neutral candidates; ambiguous or changed failure semantics go to owner |
| IP-WP3 capacity/runtime screening | IP-G1 owner-frozen cells | B8/B12/B16 capacity evidence and selected SDPA/compile/AdamW/checkpoint probes, all measurement-only by default | runs only inside separately approved IP-E2; no automatic recipe promotion |
| IP-WP4 synthesis | accepted WP1-WP3 evidence | final combination validation, GH200 payback, keep/reject/owner-gated table, revised Envelope-B projection | IP-G2 decides promotion and later Envelope-B refreeze; Phase I-P itself makes no capability claim |

Gate/envelope order is exact:

```text
IP-G0 (closed: plan/topology/local implementation)
  -> IP-WP0 (source/static closed; GH200 runtime close is first IP-E1 reference)
  -> IP-E1 (active: IP-WP1 -> strict IP-WP2 continuously)
  -> IP-G1 (baseline diagnosis and exact IP-E2 shortlist)
  -> IP-E2 (pending: IP-WP3 -> IP-WP4 continuously)
  -> IP-G2 (promotion/recipe/checkpoint/Envelope-B disposition)
```

IP-G2 is not Envelope-B activation. Any accepted production-source or config
change requires a new exact Envelope-B source/config/resource projection and the
already-required independent recipe-freeze review.

IP-E1 retains maximum concurrency one and serial execution. Its ordinary WP1/WP2
cells have a `2.0` charged-GH200-hour aggregate ceiling; a separately accounted
`+1.0` hour reserve may be consumed only by diagnosed code-level defect repair,
for a `3.0` hour hard aggregate ceiling. Code bugs have no submission-count stop
and are repaired continuously without blind identical retry. Ambiguous diagnosis,
scientific-boundary pressure, or hard-ceiling exhaustion still returns to the owner.

Job `525192` closes WP0 runtime and supplies one valid LiDAR sustained measurement:
`40.4214` presentations/s over 256/256 accepted windows, only `1.5169 ms/window`
mean loader wait, and `6.8215%` peak reserved-memory fraction. Checkpoint boundary,
64 continuation microbatches, RNG and training state were exact, but both a same-
process replay and a fresh-process replay failed the frozen per-tensor FP16 allclose
after eight windows at globally small relative L2 error. This proves the symptom is
runtime-kernel nondeterminism, not checkpoint corruption or input drift. Because a
change from elementwise allclose to a same-process-calibrated global error gate is
an acceptance-rule decision rather than a code repair, further submission awaits
explicit owner disposition; no tolerance has been relaxed.

### 1.2 Candidate classes and immutable scientific boundary

- Strict output-neutral candidates: unused training-field/augmentation transfer
  cleanup; fixed meshgrid/cache; LiDAR host counts/offsets with exact order checks;
  consolidated D2H while retaining SciPy Hungarian, float64 inputs, order and tie
  behavior; one checkpoint CPU snapshot reused for save/hash. Promotion requires
  exact where attainable plus forward, loss, gradient, accepted-update, BN/scaler/
  scheduler and checkpoint-continuation checks.
- Normal-path-equivalent but failure/control-flow-changing candidates: one finite-
  loss synchronization per B32 window and asynchronous checkpoint publication.
  These are not strict output-neutral and require explicit abnormal-path evidence
  plus owner disposition.
- Numerical runtime candidates: Camera/LiDAR SDPA, scoped `torch.compile`, and
  foreach/fused AdamW. They remain measurement-only and default-off until IP-G2.
- Recipe/operational candidates: physical B8/B16, checkpoint cadence, activation
  checkpoint, persistent-worker/RNG changes. B8x4 and B16x2 preserve effective B32
  but alter BatchNorm statistics; B12 cannot exactly realize effective B32. They
  cannot be promoted without an explicit owner recipe decision.
- Material science outside Phase I-P: normalization, precision/TF32 policy,
  model/head/math/shape, loss/target/decode/tie semantics, data ownership/order/
  exposure, scheduler mathematics, evaluator/metric, seed, or candidate count.

All Phase I-P execution is D_fit-only. `D_select`, `D_audit`, official validation,
mAP/NDS, capability, generalization and candidate-selection claims are forbidden.
The frozen Phase-I recovery contract currently writes once per epoch, retains at
most the latest two consecutive recovery checkpoints, restores RNG/sampler only at
an optimizer-window boundary, and permits selection only from epoch 20. Phase I-P
measures that cost; it does not silently lower the cadence or change eligibility.

### 1.3 Frozen measurement and decision boundaries

- Reference and final candidates use 16 accepted optimizer windows of warm-up plus
  at least 256 accepted windows (8,192 presentations) per process and two fresh
  process repeats. A throughput spread above 3% triggers a third repeat; continuing
  instability blocks a speed claim.
- A separate trace records three accepted windows with CPU/CUDA shape/memory ranges.
  One-second `nvidia-smi` sampling records utilization, memory, clocks and power;
  heavier Nsight work is added only when the bounded trace cannot explain a gap.
- Report attempted and accepted presentations/s, accepted updates/s, wall latency,
  loader wait, H2D, forward/loss/backward/update shares, allocator active/reserved/
  inactive state, checkpoint snapshot/save/model-hash/file-hash/load cost, compile
  startup/recompiles, and process-repeat spread. Do not infer compute saturation
  merely from unused VRAM.
- Memory is measured after the first accepted AdamW update has materialized state.
  Steady-state reserved memory must stay at or below 85% of visible memory and show
  no monotonic 256-window growth. Capacity probes use a fresh process; OOM is a
  recorded `CAPACITY_OOM`, B8 OOM skips larger sizes, and B12 OOM skips B16.
- Parity remains exact/hash-exact for unchanged discrete/plumbing state where
  attainable; integrated FP32 uses `rtol=1e-4, atol=1e-6`, and accepted FP16 uses
  `rtol=2e-3, atol=2e-4`. Tolerances are not relaxed inside an envelope.
- Checkpoint validation occurs after AdamW state exists at an optimizer boundary:
  real save and file/model hashes, full release, fresh stack reconstruction/load,
  exact identity/RNG/sampler/state checks, then eight D_fit optimizer windows
  compared with an uninterrupted control. The profiler-shortened epoch is not a
  scientific epoch.
- There is no arbitrary 1.25x speed gate. A measured acceleration requires the
  candidate/baseline throughput 95% confidence lower bound above 1; small strict
  improvements may be bundled and retested. Payback is
  `T20 = 1,758,080 / sustained_samples_per_second + checkpoint stalls + startup`,
  `saved_GH200h = T20_baseline - T20_candidate`, and
  `break_even_runs = actual_candidate_profiler_GH200h / saved_GH200h`.

O-143 supersedes the active six-stop execution order and S10's per-job
immutable/no-retry/multi-document/reviewer mechanics. It does not erase prior
evidence, change STOP-A data ownership/evaluator semantics, weaken metric or
provenance requirements, or authorize compute.

O-144 freezes `PHASE_I_PLAN.md` as the binding Phase I scientific and
collaboration plan: physical B4 plus accumulation 8/effective B32; one ImageNet
Camera primary and one scratch LiDAR primary; exact reference-led recipes;
role-bound D_fit-only GT-paste; seed 0; 20 epochs; terminal-only selection; two
total candidates; and five WPs, three owner gates and two approval envelopes.
O-145 amends WP2/WP4 to require an independent in-tree optimized CUDA BEV-pooling
port or functionally equivalent kernel, a labelled reference fallback, FP32/FP16
forward/backward and policy parity, and GH200 operator plus aligned end-to-end
timing. O-146 activated the exact Section-6 Envelope A recorded in
`RUN_REQUEST.md`; O-147 amended its limits; O-148 removed the numeric submission
stop while retaining serial execution and the unchanged `1.10` charged-GH200-hour
ceiling. Envelope A is now closed after 12 submissions and `0.516389` GH200-hours.
O-149 consolidates the future completion-oriented engineering-validation
contract. None of these decisions activates Envelope B or capability evaluation.

O-150 accepts Job H's numerically qualified PyTorch sorted `segment_reduce` path as
the Camera production backend and retains the CUDA kernel as an unpromoted optional
optimization. The historical `1.25x` CUDA promotion failure remains evidence but no
longer blocks Camera capability. The owner's instruction to start Envelope-B
preparation produced the exact Section-7 request in `RUN_REQUEST.md`: two fixed
seed-0 candidates, serial execution, maximum concurrency one, and a measured
`49.0` charged-GH200-hour aggregate ceiling. It is not compute authority until the
owner names and approves the containing commit. IP-G0 now postpones that
disposition through IP-G2: the old request remains a frozen comparison object and
must not be activated directly while Phase I-P is open.

Envelope-B implementation baselines `6eaafa07942a3079cb9725cf2c83a9e2e4c6c6ed`
and `a1d7d4fc9508875cc7559858b51b9c1fe441f69b` add schema-v2 production configs,
the fallback-only Camera dispatch, a clean ordered D_select loader, LiDAR's frozen
decode threshold bridge, and one direct dual-branch runner with epoch-atomic recovery.
Local `py_compile`/`compileall`, `bash -n`, `shellcheck`, canonical config resolution
and hash checks pass. No GH200 import/forward test or scientific job has run at these
commits; the independent recipe-freeze review is still required.

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
| Envelope-A Camera / Job H `522113` | checkpoint and all correctness/e2e/memory gates passed; optimized-pool ratio `0.976174` failed frozen `<=0.80`; O-150 accepts the parity-qualified fallback for production | CUDA promotion or Camera capability |
| Envelope-A LiDAR / Job B5 `522222` | exact keyframe GTDB, BN/no-GN graph, sparse FP32 island, no-update calibration/evaluator/checkpoint gates passed; qualified config emitted | training convergence, mAP/NDS, scientific checkpoint or candidate selection |

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

### Phase I-P — throughput preflight before qualification

Execute only the engineering workflow frozen in Section 1. It may measure D_fit
training mechanics and propose safe production changes, but it cannot evaluate or
select a detector. Phase I-P closes at IP-G2 with an accepted/rejected/owner-gated
optimization set and, if warranted, a newly frozen Envelope-B request.

### Phase I — camera and LiDAR independent recipe/capability

Treat the modalities as separate training problems before fusion. The complete
binding graph, initialization, optimizer/scheduler, augmentation, CBGS/GT-paste,
batch/exposure, checkpoint, evaluation and workflow specification is
`PHASE_I_PLAN.md`; summaries here do not override it.

The initial Phase I set is exactly one ImageNet-initialized standalone-reference
Camera/CenterHead primary and one scratch reference-led
SECOND/SECONDFPN/TransFusionHead LiDAR primary. Both use reference BN, seed 0,
20 exact-CBGS epochs, physical B4 plus accumulation 8/effective B32, no EMA and
epoch-20 terminal-only selection. LiDAR trains keyframe-only and evaluates with
keyframe plus nine sweeps. NuImages, GN, alternate LR/seed and automatic repair
are outside the two-candidate envelope.

Step-level runs are only crash/numerical preflight. Capability requires meaningful
trainval-scale exposure and evaluation. Phase I exits with one reviewed camera
recipe/checkpoint and one reviewed LiDAR recipe/checkpoint, or an honest negative
result. `D_audit` remains sealed until the owner explicitly opens it at `P1-G2`.

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
4. return a cause-directed repair proposal to the owner; it is not inside the
   initial two-candidate envelope;
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
- aggregate GPU-hours, submission policy and concurrency;
- stop/escalation conditions and output root.

Under O-149, an explicitly approved engineering-validation envelope is controlled
by aggregate GPU-hours and concurrency; submission count has no default numeric
cap unless the owner sets one. S00 diagnoses and fixes unambiguous
single-correct-answer defects anchored to frozen semantics—tests/fixtures,
config/schema parsing, dtype/API plumbing, runners, checkpoint I/O, artifact
publication/provenance or logging—and immediately resubmits serially with fresh
outputs. Different diagnosed bugs do not trigger mechanical owner round trips.

S00 returns to the owner before changing model/reference math, data ownership or
content, recipe/candidate space, precision, optimizer/scheduler/EMA,
metric/evaluator, seeds, gates, interpretation or aggregate resources. It also
stops at ceiling exhaustion, ambiguous diagnosis or recurrence of the same
blocker after repair. Blind identical retries remain forbidden. Scientific and
capability runs retain separate approval; O-149 creates no standing compute.

Active records are:

- this `HANDOFF.md`: compact current status, science plan and decision boundary;
- `RUN_REQUEST.md`: phase authority plus one concise job ledger.

`PHASE_I_PLAN.md` is the frozen plan specification, not a second status or job
narrative. Update it only through an explicit owner amendment.

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

The old C0/C1 diagnostic harness is frozen historical tooling. Envelope B uses one
direct two-branch entry over the exact Phase-I config/data/model/optimizer plus the
standard checkpoint and evaluator modules. It does not extend the generic FL/S09
`centralized_train.py` harness or add another reusable orchestration layer.

## 7. Envelope-A execution record

`P1-G0 PLAN_FREEZE` is closed, O-145 is incorporated, and O-146/O-147/O-148
Envelope-A execution is consumed. O-148 converted Job D's unlocalizable
four-second pre-control failure from a mechanical phase stop into a continuous,
serial, diagnosed repair loop under the unchanged `1.10` GH200-hour ceiling. That
loop is complete: 12 submissions used `0.516389` GH200-hours. Raw per-job
provenance and every diagnosis remain append-only in `RUN_REQUEST.md`; the compact
terminal outcomes below are authoritative for next-step planning.

O-149 now preserves the efficient part of that collaboration pattern for future
explicitly approved engineering-validation envelopes while retaining all
scientific owner gates. Envelope-A authority ended when WP4 reached its honest
terminal outcomes; the remaining `0.583611` GH200-hours cannot be reused.

The request-scoped roots are
`s10_phase1_envelope_a_data_e321aed749fd` and
`s10_phase1_envelope_a_eng_e321aed749fd` under the accepted Arrhenius output root.

WP0 is committed at `714f7a1067f375861c80e3020ab302a928983f12`. WP1's
mechanical comparison against the pinned MIT `use_valid_flag=True` path found that
the first static count had incorrectly used the local in-range mask. The corrected
official eligibility is `(num_lidar_pts + num_radar_pts) > 0`, derived from the
physically bound `sample_annotation.json`. This supersedes—not supplements—the
unexecuted `N_cbgs=78,470` draft.

The exact D_fit official-CBGS artifact has SHA-256
`64cc0d1d6cd82fae2787d397e610178cedd00887d98938b154fce9f8e8e115ef`:
`N_cbgs=87,930`, 87,904 consumed and 26 dropped presentations per epoch,
2,747 optimizer updates per epoch, and 54,940 over 20 overflow-free accepted
updates. The pre-materialization Camera/LiDAR resolved-config hashes are now
`f198817a4e6e021136cc1ec7c34f4079ff272341e97461458bf1715a607c658d`
and `b9b29dbabba7899ecc703fdd3566e54cca5606dfcd1a783db96c7b9efb57eddf`.
This is about 12.0% more attempted B32 exposure than the superseded draft; it is
not yet a GPU-hour estimate.

WP1 implements the immutable D_fit/ten-sweep-cache/keyframe-consumption binding,
official CBGS artifact plus epoch-order/remainder identities, reference-order
taxonomy mapping, all-class D_fit-only GTDB with per-file manifest validation,
velocity-preserving role-bound GT-paste for epochs 1-15, reference augmentation
filters/shuffle, B4 x accumulation-8 AdamW/cyclic schedule, and Phase-I checkpoint
identity/resume support. Reference `RandomFlip3D` uses both horizontal and vertical
branches; an earlier local vertical-disabled expansion was corrected before WP1
freeze. Login-node validation covers syntax, canonical config resolution, exact
physical data identities, and deterministic CBGS derivation. WP4 later passed the
focused Torch/CUDA LiDAR suite and exact GTDB binding; no capability run occurred.

WP2 implements the standalone Camera graph without modifying the historical Fusion
detector: trainable Swin-T with stage outputs `[1,2,3]` and identity-initialized output
LayerNorms; concat GeneralizedLSSFPN; pure-camera LSS; the request-scoped optimized
CUDA pooling backend plus sorted segment-sum fallback; Camera GeneralizedResNet/LSSFPN;
and six-task BatchNorm CenterHead. The Phase-I decode restores the pinned second
task-wide top-500 selection rather than inheriting the older no-starvation adaptation.
The project-wide physical `H=y,W=x` convention is retained explicitly and covered by a
non-square pooling fixture, while the CUDA segment reduction remains the pinned
operation. ImageAug3D parameters are sampled in DataLoader workers using the exact
NumPy reference draw order before the scene-3D draws, so epoch-boundary recovery also
replays Camera augmentation.

Before O-148, the single approved ImageNet Swin acquisition had completed into the read-only quarantine
path (114,342,173 bytes), redirected through the allowlisted
`release-assets.githubusercontent.com` host, with physical SHA-256
`9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3`.
At that boundary the final checkpoint path remained absent and the bytes were unusable: schema,
per-tensor mapping, loaded/missing/unexpected keys, initialized-state identity and
atomic promotion did not complete before the execution stop.

WP4 Camera Job A `521859` stopped after `00:01:42` in the focused CUDA gate:
27 tests passed and one failed because the fallback used a global prefix-cumsum whose
cross-cell rounding order did not match the pinned per-cell sequential CUDA sum. The
production kernel was unchanged; remediation `564fb9d97c44a463ac055dc40d25b79acdc77858`
replaced only the independent fallback reduction with PyTorch's length-delimited
per-cell SegmentReduce and added a rounding-order regression.

The sole derived Job C `521901` then passed all 29 focused tests, including FP32 and
autocast forward parity and exact feature-gradient parity, but stopped after
`00:01:48` while hashing the mapped Swin state: the identity helper attempted a
cross-element-size uint8 view of a scalar Long BatchNorm buffer. It failed before
writing the mapping report or renaming the quarantine, and before any D_fit read,
model calibration, operator/end-to-end timing, checkpoint preflight or evaluator
schema check. Output-neutral remediation
`67c1b55b59aa81a49b1ed8f4aabd07e6592e88aa` uses raw contiguous NumPy bytes for
both scalar and N-D tensors and adds an exact scalar-buffer identity test. O-147's
Job D stopped before verification, but O-148 Job F later verified the repair and
completed atomic checkpoint acceptance on GH200.

Under O-148, Job E `522037` exposed a manually truncated expected SHA in the sbatch
binding; Job F `522042` then passed 30/30 focused tests and completed exact Swin
acceptance. The final read-only checkpoint now has physical SHA-256
`9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3`,
the mapping report has SHA-256
`c87469b84b4b865aa478cc1959c400468f8aca393e53cf8dbb92a71c3a63f70f`,
and quarantine is absent after atomic promotion. Job F then exposed an evidence-
serialization defect: canonical config bytes were written with an extra newline and
could never match the canonical identity. Job G `522094` verified that repair, all 31
focused tests, checkpoint reuse, standalone FP32/FP16 parity, integrated FP32 parity,
and the end-to-end/memory gates. It then produced the first complete negative pooling
diagnostic: relaxed-policy FP16 serial upstream gradients lacked a same-backend
nondeterminism control, and optimized operator median was 0.959410x fallback rather
than the required <=0.80x. Job H `522113` then passed the corrected FP16
same-backend controls, all forward/backward parity checks, end-to-end ratio
`0.999519 <= 1.02` and memory ratio `0.999795 <= 1.05`. Its optimized operator
ratio was `0.976174 > 0.80`, only about a 2.4% speedup over the exact PyTorch 2.11
fallback. The frozen promotion gate therefore failed honestly: the kernel was not
promoted and no Camera qualified config was emitted. The ImageNet Swin checkpoint
itself is accepted and read-only, but it is not a scientific Camera checkpoint.
Job H's immutable result retains one stale `.control` prefix in the informational
materialized-config path; the actual final config exists under Job H's final
`evidence/` root with the recorded hash. The later publication-path repair is
prospective; no Camera rerun is warranted because the frozen performance gate is
already conclusively negative.

LiDAR Job B materialized and sealed the exact D_fit keyframe GTDB: 321,613 objects
across all ten classes, manifest SHA-256
`22e3e23c2dff19280476ee622ea062592b6b9a1712902e7e83cb4b242fafa2b5`.
The completion loop repaired an IoU test fixture, JSON mapping-order parsing,
discrete dtype promotion and post-rename artifact paths without changing frozen
science. Final Job B5 `522222` passed 33 tests/3 skips, no-update forward/backward,
evaluator serialization, BN/no-GN and sparse-FP32-island gates, plus exact
checkpoint reload. It measured median GPU step `93.118401 ms`, `41.904378`
samples/s and `5,346,498,048` peak allocated bytes. The directly consumable
qualified config is
`06e78e456793fe269c978b0e663da39e4ec3216523c54f996665bc1a6a952015`;
the zero-update recovery checkpoint is
`8166d2016a560d7b572ec7d196a886f0780eb317f6a8a32f8a86f80160e92611`
and is explicitly non-selectable/non-scientific.

Final Envelope-A usage is 12/unlimited serial submissions and
`0.516389/1.10` GH200-hours. No optimizer update, capability metric, D_select,
D_audit, official validation, scientific checkpoint or candidate selection
occurred. O-150 resolves the Camera backend disposition without revising the
historical CUDA-performance result. The exact Envelope-B tuple is now frozen at
`49.0` charged GH200-hours: measured training estimate `41.520365h`, conservative
evaluator/checkpoint/sealed-audit reserve `0.80h`, and 15% contingency gives
`48.668420h`, rounded up. Owner activation and the independent recipe-freeze review
remain before scientific submission.

WP3 implements the reference-led standalone LiDAR graph without changing the
historical Fusion detector. The existing reference-shaped sparse SECOND is reused only
through a new explicit Phase-I boundary: hard-voxel mean VFE receives
`x,y,z,intensity,time_lag`, sets keyframe lag to exact zero while retaining ring in the
source payload, preserves the frozen PointShuffle order for point/voxel caps, uses
BN1d `eps=1e-3,momentum=0.01`, and returns the unprojected FP32 dense collapse
`[B,256,180,180]`. No old `to_bev` projection or GroupNorm is instantiated.

The dense path is the pinned SECOND `[5,5]`, SECONDFPN `[1,2]`, and an independent
mmdet3d/mmcv-free one-layer, 200-query TransFusionHead. It includes physical
`H=y,W=x` query positions, canonical geometric-center box coding, vectorized rotated
3D IoU, reference focal/BEV-L1/IoU Hungarian costs, Gaussian/query/regression losses,
and no-NMS decode in the reference class order. WP4 Job B5 verified spconv
construction, forward/loss/backward, the accepted FP16+sparse-FP32 policy,
evaluator schema, checkpoint round-trip and bounded production timing.

No 20-epoch capability run, D_select/D_audit/official-val evaluation, staged fusion,
broad profiler, merge, push, upload, publication or S11+ work is authorized.
