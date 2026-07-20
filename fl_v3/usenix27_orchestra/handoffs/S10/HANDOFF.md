# S10 HANDOFF — O-146 active Phase-I Envelope A

## 1. Current state and authority

```text
SESSION: persistent S00 / S10
BASE_SHA: a080d49c1c22de20ccb5b1353d4922c7df14a729
BRANCH: codex/s10-phase1-branch-qualification
ACTIVE_DECISION: O-146 under O-143/O-144/O-145
SCIENCE_ORDER: C/L independent recipe+capability -> staged fusion -> capability gate -> profiler
PHASE_I_PLAN: PHASE_I_PLAN.md; P1-G0 PLAN_FREEZE closed
CURRENT_AUTHORITY: exact Envelope A at e321aed749fd859c809199d52c30b2771dbef8b3;
                   continuous WP0-WP4 plus bounded acquisition/materialization/calibration
MERGE/PUSH/UPLOAD/PUBLICATION/S11+: not authorized
```

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
timing. O-146 activates the exact Section-6 Envelope A recorded in
`RUN_REQUEST.md`: WP0-WP4 may proceed continuously, with at most three serial
one-GH200 submissions and at most one aggregate GH200-hour. It does not activate
Envelope B or capability evaluation.

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

The old C0/C1 diagnostic harness is frozen historical tooling. New capability
work should use the production `centralized_train.py`, standard checkpointing
and evaluator paths, extended only by the smallest required branch-mode seams.

## 7. Active Envelope-A execution

`P1-G0 PLAN_FREEZE` is closed, O-145 is incorporated, and O-146 activated Envelope A
at request commit `e321aed749fd859c809199d52c30b2771dbef8b3`. S00 is executing
WP0-WP4 continuously. The request-scoped roots are
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
physical data identities, and deterministic CBGS derivation. Torch/CUDA focused
tests remain bound to WP4's already-approved GH200 jobs; no capability run occurred.

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

The single approved ImageNet Swin acquisition completed into the read-only quarantine
path (114,342,173 bytes), redirected through the allowlisted
`release-assets.githubusercontent.com` host, with physical SHA-256
`9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3`.
The final checkpoint path remains absent and the bytes remain unusable: schema,
per-tensor mapping, loaded/missing/unexpected keys, initialized-state identity and
atomic promotion are all still gated on WP4 Job A. WP2 login-node validation is syntax
and static-contract only; it is not CUDA parity or performance qualification.

The checkpoint is the MIT Camera YAML's ImageNet
`swin_tiny_patch4_window7_224.pth`, not `swint-nuimages-pretrained.pth`; acquisition
and quarantine acceptance now belong to WP2. After Envelope A yields final GTDB
identities, resolved config hashes, graph timing and one joint recipe review,
`P1-G1` will present the measured Envelope-B scientific GPU-hour/submission
request.

No 20-epoch capability run, D_select/D_audit/official-val evaluation, staged fusion,
broad profiler, merge, push, upload, publication or S11+ work is authorized.
