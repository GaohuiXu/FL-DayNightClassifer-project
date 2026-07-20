# S10 Phase I C/L branch qualification — binding plan

## 0. Status, purpose, and authority

```text
STATUS: OWNER-FROZEN SCIENTIFIC PLAN / O-150 FALLBACK-BACKEND AMENDMENT
DATE: 2026-07-20
OWNER_DECISION: O-144 plus O-145 optimized-BEV-pooling qualification;
                O-149 collaboration-process amendment;
                O-150 production-fallback/capability-gate amendment
SCOPE: Phase I camera/LiDAR clean branch qualification
AUTHORITY: freezes the Phase I scientific choices, work-package order, gates,
           approval structure, and execution boundaries recorded below
IMPLEMENTATION: WP0-WP4 completed under consumed O-146/O-147/O-148
COMPUTE: Envelope A is terminal; exact 49.0-hour Envelope-B request is frozen,
         owner activation and independent recipe-freeze review pending
CHECKPOINT_ACQUISITION: completed once under consumed Envelope A
COMMIT: material Envelope-A implementation/result/contract closure commits authorized
MERGE/PUSH/UPLOAD/PUBLICATION: not authorized by this document
AMENDMENT: any departure from a frozen scientific field or gate requires an
           explicit owner amendment before implementation or execution
```

O-144 promotes this document from a temporary discussion draft to the binding Phase I
plan. Future implementation and execution must follow it. It complements, and does not
supersede, `HANDOFF.md`, `RUN_REQUEST.md`, `AGENTS.md`, or later owner decisions.
Plan freeze was not execution authority: O-146 later activated Envelope A, O-147
amended it, and O-148 completed it. Envelope B must still be separately approved
before scientific training or evaluation. O-145's documentation-only amendment
commit was the sole exception at plan freeze; later owner decisions are recorded in
`RUN_REQUEST.md` and the canonical Orchestra docs.

O-145 amends the frozen implementation plan without changing either scientific
candidate: WP2 must provide an independent in-tree port of the pinned MIT optimized
CUDA BEV-pooling operation, or a functionally equivalent kernel, and WP4 must qualify
its numerical and performance behavior. O-145 authorizes this documentation amendment
and drafting the exact Envelope-A request only. It does not activate Envelope A,
authorize checkpoint acquisition, or authorize implementation or GPU execution.

O-146/O-147/O-148 later activated, amended and completed Envelope A without
changing the frozen science. Its terminal outcome is Camera negative at the
optimized-pooling promotion gate and LiDAR engineering PASS. O-149 amends only
the collaboration/remediation mechanics for future explicitly approved
engineering-validation envelopes; it creates no compute authority.

O-150 accepts the numerically qualified PyTorch sorted `segment_reduce` fallback as
the Phase-I Camera production backend. The CUDA backend remains available only as an
explicit, unpromoted optimization path. Job H's historical `1.25x` promotion gate and
negative result remain intact as performance evidence, but that target is no longer a
Camera capability prerequisite and may not block Envelope B. This changes no model
graph, data, seed, recipe, evaluator, precision policy, or candidate count.

## 1. Technical summary

Phase I will qualify camera and LiDAR as independent clean perception branches before
fusion. The primary graph choices are now reference-led rather than a search over the
current shared-GN hybrid:

- **Camera:** use the exact standalone MIT Camera graph family: Swin-T,
  GeneralizedLSSFPN, pure-camera LSS, the camera-specific GeneralizedResNet/LSSFPN
  decoder, and the reference six-task CenterHead.
- **Camera BEV pooling:** use the numerically qualified PyTorch sorted
  `segment_reduce` implementation as the production backend. Retain the independent
  mmdet3d/mmcv-free CUDA port as an explicit, unpromoted optimization backend; its
  measured performance is not a capability gate.
- **LiDAR:** use the MIT `voxelnet_0p075` graph family with keyframe-only training,
  reference BatchNorm, SECOND/SECONDFPN, and TransFusionHead.
- **Fusion:** no Phase I training. Phase II will use the reference staging direction:
  the declared camera initialization plus the full qualified LiDAR checkpoint, with
  L/F sharing the SECOND/SECONDFPN/TransFusionHead family.
- **Normalization:** no GroupNorm candidate remains. Use each reference module's
  BatchNorm choice; keep Swin's native LayerNorm.
- **Sampling:** replace the local sqrt repeat-factor sampler with the exact archived
  MIT `CBGSDataset` algorithm over role-restricted `D_fit`.
- **Throughput and batch policy:** physical B4, accumulation 8, effective optimizer
  batch 32, camera activation checkpointing off, redundant scalar telemetry
  synchronization off, LiDAR keyframe-only training, and terminal-only branch
  evaluation.
- **Assessment:** no arbitrary numeric pass threshold will be invented before the
  run. Checkpoint selection, metric/evaluator semantics, and the one-time use of
  `D_audit` must nevertheless be frozen before results are inspected.

The primary Camera run uses the exact standalone reference graph with the pinned
ImageNet-1K Swin-T initialization declared by the reference YAML. The MIT README's
published reproduction command overrides that default with a NuImages checkpoint, so
the local primary must be labelled **exact standalone reference graph + reference-YAML
ImageNet initialization**, not a reproduction of the published NuImages-initialized
full recipe.

## 2. Phase I objective and exclusions

### 2.1 Objective

Produce one reviewable Camera recipe/checkpoint and one reviewable LiDAR
recipe/checkpoint with meaningful `D_fit` exposure and aligned internal evaluation.
Phase I may also return an honest negative branch result. It must not silently expand
into a hyperparameter grid after observing weak metrics.

### 2.2 Required outputs

For each branch:

1. exact local graph and normalization specification;
2. initialization source and tensor-loading identity;
3. optimizer parameter groups, LR/weight decay, scheduler/warmup, clipping, and EMA
   policy;
4. data augmentation, GT-paste, exact CBGS index identity, and sweep policy;
5. physical B4/effective B32 exposure, attempted sample presentations, accepted
   updates, and terminal-checkpoint identity;
6. one terminal `D_select` evaluation and, only after `P1-G2` owner unsealing, one
   `D_audit` result;
7. immutable checkpoint and metric-artifact hashes;
8. a branch recipe freeze review, which may be one joint review at one durable SHA.

### 2.3 Exclusions

Phase I does not include Fusion training, a general optimization/profiler campaign,
official validation publication claims, Protocol-A/B execution, federated adaptation,
attack/defense experiments, or an automatic NuImages repair cell. No current C0/C1
diagnostic graph is an automatic second candidate. The initial scientific candidate
cap is exactly two: one Camera primary and one LiDAR primary.

The completed O-145 CUDA BEV-pooling port, parity qualification, and bounded
operator/end-to-end timing are retained engineering evidence. Under O-150 the port is
not promoted and its `1.25x` target is not a Phase-I capability prerequisite.

## 3. Frozen graph direction

### 3.1 Camera — exact standalone reference graph

```text
six camera images
  -> Swin-T (native LayerNorm)
  -> reference GeneralizedLSSFPN (BatchNorm)
  -> pure-camera LSS with PyTorch sorted segment-reduce BEV pooling
  -> camera-specific GeneralizedResNet + LSSFPN decoder (BatchNorm)
  -> reference six-task CenterHead (BatchNorm)
```

The Phase-I production backend is the in-tree PyTorch sorted `segment_reduce` path
qualified by Job H. It is part of the single Camera candidate, not an alternate
scientific candidate. The independent CUDA port remains in-tree, retains its pinned
Apache-2.0 attribution, and may be selected only explicitly for later optimization or
diagnosis. It must not silently replace the production backend in Envelope B.

The completed WP2/WP4 qualification established the following reusable requirements:

1. prove exact geometry/rank/cell-membership and output-shape agreement with the
   reference/fallback path, including empty inputs, singletons, collisions, and B4;
2. compare forward values and backward gradients against the fallback under both FP32
   reference and accepted FP16 production policies, report maximum absolute/relative
   error, and freeze the tolerances before the first GPU parity run;
3. prove that autocast boundaries, FP32 accumulation where required, output dtype,
   finite-state checks, and the accepted S08 precision policy are unchanged;
4. fail closed on production-backend dispatch, dtype, architecture, or parity failure;
   a CUDA build failure cannot affect the production fallback; and
5. retain the pinned Apache-2.0 source attribution/NOTICE obligations if source is
   ported, and record the extension source/build/runtime identity.

The standalone Camera graph is intentionally not forced to be tensor-compatible with
the complete future Fusion detector. Its complete detector checkpoint establishes a
Camera capability baseline; the official Fusion staging path does not require loading
the complete Camera detector.

The rejected alternative was a project-created “fusion-compatible Camera-only” graph
using the future Fusion BEV grid/decoder/head solely to maximize checkpoint reuse. That
would not be the published Camera baseline and would reopen architecture search.

### 3.2 LiDAR — reference `voxelnet_0p075` graph

```text
keyframe LiDAR points
  -> reference-compatible hard voxelization / mean VFE
  -> sparse SECOND with BN1d (eps=1e-3, momentum=0.01)
  -> dense collapse
  -> SECOND [5,5] + SECONDFPN decoder (BatchNorm)
  -> TransFusionHead (reference BN2d/BN1d)
```

The local mmdet-free implementation may adapt framework plumbing and the accepted
sparse FP32 precision island, but it must preserve the selected graph, tensor shapes,
normalization semantics, target/loss semantics, and checkpoint mapping. The current
custom shallow `SecondFPNNeck` and shared GN CenterHead are not the Phase I LiDAR
candidate.

### 3.3 Head semantics

The Camera CenterHead partitions the official ten nuScenes classes into six detection
tasks:

1. car;
2. truck + construction vehicle;
3. bus + trailer;
4. barrier;
5. motorcycle + bicycle;
6. pedestrian + traffic cone.

This remains a ten-class detector; “multi-task” here means six class-grouped detection
heads, not six unrelated perception tasks.

TransFusionHead is different. It forms a ten-class dense heatmap, selects a bounded
set of object queries, refines them with a Transformer decoder over the BEV feature,
and trains query predictions through Hungarian assignment. It is not the old simple
global ten-class convolutional head.

The accepted mapping is therefore:

| Branch | Decoder | Detection head |
|---|---|---|
| Camera | GeneralizedResNet + LSSFPN | six-task CenterHead |
| LiDAR | SECOND `[5,5]` + SECONDFPN | TransFusionHead |
| Fusion, Phase II | SECOND `[5,5]` + SECONDFPN | TransFusionHead |

### 3.4 Frozen branch recipe bundles

Both primary branches train for 20 official-CBGS epochs with physical B4,
accumulation 8, effective optimizer batch 32, seed 0, terminal-only checkpoint
selection, no EMA, and the accepted S08 precision policy. Camera uses global FP16
autocast; LiDAR uses global FP16 with the explicit FP32 island covering voxelization,
VFE, sparse SECOND, dense collapse, and to-BEV. Any loss-scaler initial values are
qualified without updates during Envelope A and frozen in the resolved Envelope-B
config; this does not permit a precision-regime search.

The Camera primary is:

- exact standalone graph from Section 3.1;
- pinned public ImageNet-1K Swin-T initialization, with URL, license, local permitted
  path, file SHA-256, state-dict mapping, loaded/missing/unexpected tensor report, and
  initialization-state hash bound before training;
- AdamW `lr=2e-4`, `weight_decay=0.01`;
- Camera backbone `lr_mult=0.1`; no weight decay for absolute-position and
  relative-position-bias parameters;
- gradient clipping with L2 `max_norm=35`;
- one cyclic LR cycle with `target_ratio=5.0`, `step_ratio_up=0.4`, and the exact
  low-ratio interpretation inherited from MMCV 1.4;
- linear warm-up for 500 optimizer updates at `warmup_ratio=1/3`;
- cyclic momentum with the pinned MMCV 1.4 defaults;
- reference Camera augmentation from Section 7.1 and no GT-paste.

The LiDAR primary is:

- exact reference-led graph from Section 3.2, initialized from scratch;
- AdamW `lr=1e-4`, `weight_decay=0.01`, with complete/disjoint parameter groups;
- gradient clipping with L2 `max_norm=35`;
- one cyclic LR cycle with the pinned MMCV 1.4 defaults: target ratios
  `(10, 1e-4)` and `step_ratio_up=0.4`;
- cyclic momentum target ratios `(0.85 / 0.95, 1)` and `step_ratio_up=0.4`;
- no LR warm-up;
- reference LiDAR augmentation and role-bound GT-paste from Section 7.2;
- keyframe-only training and keyframe-plus-nine-sweep evaluation.

All values above are explicit fields in the local ResolvedConfig. No current-library
default may stand in for a pinned reference value.

## 4. Normalization, throughput, and the future FL boundary

### 4.1 Phase I normalization decision

The Phase I candidate set contains no GroupNorm alternative:

- Swin keeps its architectural LayerNorm;
- Camera FPN/decoder/CenterHead use reference BatchNorm;
- sparse SECOND uses reference BN1d;
- LiDAR/Fusion decoder, fuser, and TransFusionHead use their reference BatchNorm
  forms.

C1-B is bounded evidence that replacing only the sparse SECOND normalization increased
the current Fusion B4 proxy from `8.4914` to `12.1663` samples/s, about 43% higher
throughput. It did not prove a final capability advantage, but no additional GN
capability cell is requested because the owner has selected the coherent reference BN
direction.

### 4.2 Accepted throughput settings

- physical batch size: B4;
- gradient accumulation: 8 microbatches;
- effective optimizer batch: 32 samples;
- divide/average the accumulated loss so one optimizer update represents the mean
  gradient over the eight B4 microbatches;
- optimizer, cyclic LR/momentum, warm-up, and accepted-update accounting advance per
  optimizer update, not per microbatch;
- camera activation checkpointing: off;
- ordinary per-loss scalar telemetry synchronization: off unless an explicitly
  approved diagnostic requires it;
- DataLoader baseline: eight workers; no new loader campaign;
- checkpoint selection/evaluation: only the epoch-20 terminal checkpoint is eligible;
  recovery checkpoints are non-selectable, and CPU metric aggregation is separated
  from GPU inference when practical;
- no Phase I `torch.compile`/graph-compiler, fused-optimizer, TF32, or sparse-FP16
  campaign; the O-145 in-tree BEV-pooling extension is the sole custom-kernel/build
  exception;
- record natural timing and memory counters from the chosen graph, then defer a
  dedicated profiler/optimization campaign to Phase III.

MIT's published invocation uses eight distributed training processes/GPUs with B4 per
GPU, giving effective batch 32. Single-GPU B4 without accumulation would materially
change the reference optimizer-update count and cyclic schedule. Physical B4 plus
accumulation 8 preserves
the reference effective-batch target while keeping each BatchNorm observation at B4.
Exact epoch remainder/drop semantics and sampler order must be resolved, tested, and
hashed before Envelope B; they may not be chosen after metrics are observed.

Reference graph changes may raise or lower end-to-end cost. In particular, the deeper
reference decoder and TransFusion query path are capability choices rather than claimed
speedups. Their natural timing must replace the old C1-B proxy before a GPU-hour ceiling
is proposed.

### 4.3 Does BatchNorm invalidate future FL?

No, but BatchNorm requires an explicit federated state policy. The main risks under
non-IID clients are noisy small-local-batch statistics, client-specific running means
and variances, and an ill-defined server model if those buffers are naively averaged.
These risks do not justify retaining GroupNorm in Phase I.

For Protocol B, clients may use a different **training recipe** from centralized base
training while retaining the same aggregatable model graph and tensor schema. Permitted
differences may include optimizer, LR/schedule, local epochs, batch/accumulation,
precision, augmentation, and the trainable-parameter mask. All benign/attack/defense
controls must still use matched client policies.

Clients may not silently replace BN modules with GN, replace the head, or otherwise
change shared tensor parameterization while claiming ordinary FedAvg compatibility.
A preliminary low-risk Protocol-B policy to discuss later is:

- distribute the BN-based `W_base` graph unchanged;
- keep BN running statistics frozen during local tail adaptation;
- either aggregate the trainable affine `gamma/beta` parameters or freeze them as an
  explicitly matched policy;
- consider client-local BN/FedBN or clean recalibration only if later evidence shows
  that frozen centralized statistics are inadequate.

This downstream BN policy is not a Phase I experiment and is not frozen by this plan.

### 4.4 Phase I capability checkpoint is not automatically Protocol-B `W_base`

Phase I may train on `D_fit`, which can contain scenes later assigned to Protocol-B
`D_tail`. Such a checkpoint is useful for architecture/recipe qualification but is not
a valid Protocol-B initializer if it has seen future client data.

After the architecture and recipe are frozen, Protocol B must retrain the selected
model on common, broadly distributed `D_base` only to produce `W_base`. Clients then
receive `W_base` and federatively fine-tune on disjoint long-tail `D_tail`. Thus the CL
and client recipes need not be identical, while data ownership and model-state
compatibility remain strict.

## 5. LiDAR sweep policy

The accepted reference policy is asymmetric:

| Use | MIT term | Local contract | Meaning |
|---|---|---|---|
| train | `sweeps_num=0` | `n_sweeps=1` | keyframe only |
| validation/evaluation | `sweeps_num=9` | `n_sweeps=10` | keyframe + nine previous sweeps |

Keyframe-only training is expected to reduce point reads, transforms, voxelization,
active sparse sites, and input-dependent memory. It does not imply a proportional
reduction in total wall time because fixed-grid dense modules and optimizer state remain.
No percentage improvement is claimed before natural timing of the selected graph.

Implementation must bind separate train/evaluation cache identities. The frozen
scene/log ownership does not change: the keyframe-only train inputs are a subset of the
already owned scene/log data.

## 6. Data roles and evaluation protocol

The accepted train-only ownership split is:

| Role | Logs | Scenes | Unique keyframes/samples | Purpose |
|---|---:|---:|---:|---|
| `D_fit` | 34 | 494 | 19,877 | train model parameters |
| `D_select` | 8 | 115 | 4,626 | development/checkpoint selection |
| `D_audit` | 8 | 91 | 3,627 | one-time internal held-out audit |

`D_fit` is 68% of the 50 official train logs and 70.66% of the 28,130 official
train samples. Dataset membership and training exposure are different quantities.

```text
D_fit --20 exact-CBGS epochs--> epoch-20 terminal checkpoint
                                      |
                                      v
                         one D_select evaluation
                                      |
                                      v
                     P1-G2 owner branch assessment
                         | accept + OPEN D_audit
                         v
                         one D_audit evaluation
                                      |
                                      v
                            Phase-I close evidence
```

### 6.1 `D_select`

`D_select` is the development assessment set. Each branch evaluates it exactly once at
the fixed epoch-20 terminal checkpoint. It cannot choose among epochs because no
intermediate or recovery checkpoint is eligible. It may inform the owner's `P1-G2`
decision to accept the branch, retain an honest negative result, or authorize a later
cause-directed amendment. Because it influences that decision, its score remains
development evidence and is not an untouched generalization estimate.

### 6.2 `D_audit`

`D_audit` is the internal sealed audit set. It must not be used for optimizer tuning,
epoch selection, candidate construction, or repeated trial-and-error. It is opened once
only after the branch graph, recipe, and epoch-20 checkpoint have been frozen and the
owner explicitly issues `OPEN D_audit` at `P1-G2`.

No numeric pass threshold is required. The audit result is reported to the owner for a
Phase-II decision. If the model or recipe is changed because of that result, the old
`D_audit` result remains valid evidence but the set has become development evidence for
the revised design; it cannot still be described as untouched audit evidence.

Official nuScenes validation remains outside both roles and is preserved for a later
approved capability claim.

## 7. Exact official CBGS and training exposure

CBGS is class-balanced resampling, not a geometric augmentation. The Phase I sampler
must implement the archived MIT algorithm rather than the current local sqrt-RFS:

1. over the already role-restricted `D_fit`, collect the indices of samples containing
   each of the ten classes;
2. compute inverse class-mass sampling ratios targeting equal class mass;
3. draw the reference number of indices from each class pool and concatenate them;
4. bind the source sample-token order, seed/RNG behavior, expanded index list, class
   counts, duplicate counts, and SHA-256 identity;
5. construct the list once with the frozen seed, matching the reference wrapper's
   epoch-invariant index membership; epoch shuffling remains a separate sampler concern.

A sample containing several classes can appear in several class pools, so
`N_cbgs = len(expanded_indices)` is generally not `19,877`. `D_fit` ownership remains
19,877 unique samples; CBGS changes how often those samples are presented.

Training exposure must therefore be recorded as:

```text
unique D_fit samples = 19,877
sample presentations = 20 * N_cbgs, subject only to the frozen epoch-remainder rule
B4 microbatches per optimizer update = 8
effective optimizer batch = 32
attempted optimizer windows = function(sample presentations, B32 and remainder rule)
accepted optimizer updates = attempted windows minus invalid/overflow windows
```

The exact `N_cbgs`, expansion ratio, effective updates per epoch, and total reference
exposure can be computed statically before GPU execution. These values are required for
the final resource estimate even though no predeclared numeric capability threshold is
required.

### 7.1 Camera augmentation

The Camera primary freezes the reference standalone augmentation bundle:

- six-view image output size `256 x 704`;
- train resize range `[0.38, 0.55]`, image rotation `[-5.4, 5.4]` degrees, and random
  horizontal image flip; evaluation resize is fixed at `0.48` with no rotation/flip;
- 3D scale `[0.95, 1.05]`, yaw rotation `[-0.3925, 0.3925]` radians, and zero
  translation;
- reference image normalization;
- GT-paste disabled.

The pure-camera LSSTransform does not consume LiDAR point contents. Envelope A may
remove Camera-only point payload/GTDepth construction only after the production-path
parity test proves identical sample geometry, loss, prediction, and consumed fields.
That is an output-neutral I/O optimization, not permission to change calibration or
augmentation matrices.

### 7.2 LiDAR augmentation and role-bound GT-paste

The LiDAR primary freezes the reference augmentation bundle:

- 3D scale `[0.9, 1.1]`, yaw rotation `[-pi/4, pi/4]`, translation limit `0.5`,
  `RandomFlip3D`, range/name filters, and point shuffle;
- GT-paste enabled for the first 15 training epochs and disabled thereafter;
- reference ten-class sample groups and minimum-point/collision semantics;
- keyframe-only source points for both ordinary training samples and GT-database
  object crops.

GT-paste must be role-bound. The database must be built or materialized only from the
exact frozen `D_fit` sample tokens, record the split/cache/ZIP/GTDB identities and
per-class counts, and fail if any `D_select`, `D_audit`, official-val, or unknown source
token is present. A whole-train GT database may not be filtered implicitly at paste
time unless its exact per-object source provenance makes the D_fit-only projection
independently verifiable. Envelope A constructed the fresh D_fit-only keyframe GTDB;
its terminal manifest identity is recorded in `HANDOFF.md` and `RUN_REQUEST.md`.

## 8. Camera initialization and NuImages checkpoint

The Camera primary uses the pinned public ImageNet-1K Swin-T checkpoint declared by the
reference Camera YAML:
`https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth`.
It is not the NuImages checkpoint. O-144/O-145 alone did not authorize its download;
consumed Envelope A bound its URL/license/destinations/redirect policy and completed
one accepted acquisition with physical SHA-256
`9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3`. If
the pinned upstream does not publish a trusted digest, the activated acquisition must
hash the quarantined bytes and freeze the physical SHA-256 before rename, tensor
mapping, or model use; a redirect/content mismatch fails closed.

The official `swint-nuimages-pretrained.pth` remains directly downloadable from the
pinned MIT repository's `tools/download_pretrained.sh`. It is a Swin backbone
initialization, not a complete Camera detector or a ready Protocol-B `W_base`. It is
outside the initial Phase I candidate cap. Therefore:

- downloading or executing a NuImages-initialized candidate is not automatic;
- weak ImageNet-primary metrics do not automatically trigger its download or run;
- adding it requires an explicit owner amendment to candidate count, interpretation,
  resources, and checkpoint authority;
- its use requires a bound URL, local permitted storage path, file SHA-256, license
  record, state-dict inspection, and strict mmdet-Swin to local-Swin tensor mapping;
- the ImageNet primary is exact in graph and reference-YAML initialization but is not
  the NuImages-initialized published full-recipe reproduction;
- any later NuImages comparison must use the same data, evaluator, exposure, and
  checkpoint-selection semantics as its matched primary.

NuImages is a standalone 2D-annotated dataset drawn from a broader autonomous-driving
image pool covering nearly 500 logs, compared with approximately 83 logs in nuScenes.
The official public description does not itself provide a role-by-role raw-file/log
disjointness proof against this project's `D_base`, `D_tail`, `D_select`, `D_audit`, or
official validation. Consequently it is a declared external prior and requires an
ownership/interpretation decision before it could initialize the primary Protocol-B
`W_base`.

The primary Camera initialization is **frozen to ImageNet-1K Swin-T**. Fully random,
ImageNet, and NuImages must never be conflated in reporting.

## 9. Assessment without an arbitrary numeric capability threshold

The owner will assess C/L capability after observing the complete frozen results. To
avoid post-hoc metric or checkpoint selection, the following rules are nevertheless
accepted:

1. freeze evaluator implementation, class map, metric definitions, and metric artifact
   format before training;
2. keep epoch 20 as the sole selectable checkpoint and run one terminal evaluation per
   branch;
3. use `D_select` only for the declared branch-assessment purpose;
4. open `D_audit` once after recipe/checkpoint freeze;
5. report all required branch metrics and failures without retroactive thresholding;
6. if `D_audit` motivates a new recipe, explicitly consume its sealed status for that
   new design;
7. keep official validation held out until a separately approved later capability gate.

Phase I completion is therefore evidence delivery plus owner assessment, not an
automated comparison with an invented mAP/NDS threshold.

## 10. Approved collaboration simplifications

The owner has accepted the following five design changes for the Phase I workflow:

1. add one production `--preflight-only` route that exercises the same resolved config,
   assertions, model/optimizer/evaluator/checkpoint construction, and one-batch path as
   training;
2. make the `ResolvedConfig` object consumed by the trainer the single source for run
   identity, and test that every scientific field is actually consumed and optimizer
   parameter groups are complete/disjoint;
3. structure the eventual phase approval so bounded in-envelope implementation and
   linear source commits can be authorized together, while each executed source still
   has an immutable Git SHA;
4. keep all candidate/model/data/recipe/precision/evaluator/metric/seed/gate/resource
   choices owner-gated; under O-149 permit autonomous engineering remediation when
   one frozen-semantics correction is unambiguous, including config/schema parsing,
   discrete dtype/API, tests/fixtures, runners, checkpoint/artifact/provenance and
   logging defects;
5. allow one combined C/L recipe-freeze review at one durable SHA rather than two
   duplicated review chains.

These changes must be implemented without creating another permanent harness or report
stack.

### 10.1 Five binding implementation work packages

Implementation stays in one persistent S00 worktree and one linear branch. Do not
create a per-WP worktree, harness, handoff, review chain, or approval cycle.

1. **WP0 — reference specification and ResolvedConfig.** Mechanically resolve the
   pinned MIT/Torchpack/MMCV inheritance into one explicit local recipe; make the
   trainer-consumed ResolvedConfig the run identity; assert that every scientific
   field is consumed and every optimizer parameter appears in exactly one group.
2. **WP1 — shared data and training recipe.** Implement exact role-bound CBGS,
   D_fit-only GTDB/GT-paste, B4 x accumulation-8 optimizer/scheduler semantics,
   deterministic epoch order/remainder identity, checkpoint/resume, and the direct
   production `--preflight-only` path.
3. **WP2 — exact standalone Camera.** Implement the Section-3.1 graph, ImageNet tensor
   mapping, reference CenterHead recipe, Camera augmentation, the production PyTorch
   sorted-segment-reduce BEV-pooling backend plus the unpromoted CUDA option and
   forward/backward parity tests, and the tested output-neutral omission of unused
   point payloads when parity permits.
4. **WP3 — reference-led LiDAR.** Implement the Section-3.2 graph, BN, reference
   SECOND/SECONDFPN, mmdet-free TransFusionHead/target/loss/decode path, and
   keyframe-train/ten-sweep-eval separation.
5. **WP4 — production integration, qualification, and review preparation.** Run
   focused local/static tests; qualify CUDA/fallback pooling forward values, backward
   gradients, and FP16/FP32 policy; measure both pooling-operator and aligned B4
   end-to-end Camera timing; run production-path C/L engineering calibration,
   checkpoint resume, and evaluator preflight; inventory the historical Alvis
   checkpoint/config/class/evaluator provenance without performing the Phase-II aligned
   comparison; then freeze one durable C/L implementation SHA for one combined recipe
   review.

Material commits are grouped at plan freeze, shared recipe infrastructure, Camera,
LiDAR, and final production-integration/review boundaries. Ordinary fixture or runner
fixes are folded into the next material commit. O-144 itself does not grant commit
authority, and O-145 grants only its documentation amendment commit; implementation
commit authority must be explicit in Envelope A.

### 10.2 Three owner gates

1. **`P1-G0 PLAN_FREEZE` — closed by O-144.** The scientific recipe, five work
   packages, three gates, two-envelope model, and amendment boundaries in this
   document are binding. This closure does not activate Envelope A.
2. **`P1-G1 SCIENTIFIC_COMPUTE_APPROVAL` — exact request frozen; owner approval
   pending.** The Camera backend disposition is resolved in favor of the qualified
   PyTorch fallback. `RUN_REQUEST.md` Section 7 now binds the two resolved configs,
   `49.0` charged-GH200-hour aggregate ceiling, serial wall segmentation, output,
   remediation and stop rules. One owner-authorized independent recipe-freeze review
   must close with no open P0-P2 before the first 20-epoch submission.
3. **`P1-G2 SELECT_AND_AUDIT` — pending.** The owner receives both terminal
   `D_select` results and chooses, per branch, accept/freeze, honest negative, or an
   explicit cause-directed amendment. `D_audit` opens only when the owner says
   `OPEN D_audit`; a repair keeps it sealed.

### 10.3 Envelope A — implementation and engineering calibration

Envelope A was designed to authorize all five WPs in one bounded implementation
period: scoped source/docs/tests, focused local validation, material linear commits,
the exact official ImageNet-1K Swin-T acquisition, exact D_fit CBGS/GTDB
materialization, the in-tree CUDA BEV-pooling build/parity/timing work, and
production-path C/L engineering calibration. It authorizes no capability metric,
`D_select`, `D_audit`, scientific checkpoint, or 20-epoch run.

The initially adopted engineering-calibration design was:

- one GH200, maximum concurrency 1;
- aggregate ceiling at most `1.0` GH200-hour;
- at most three submissions, each at most 30 minutes;
- C and L each run 16 warm-up plus 64 timed physical-B4 microbatches through the
  production entry/config/data/model/optimizer path;
- Camera additionally compares the optimized and reference/fallback pooling backends
  from identical initialization and input order, recording CUDA-event operator timing
  and aligned end-to-end timing without capability metrics;
- report loader wait, GPU step time, samples/s, peak memory, initialization and
  accepted-window state; do not launch a broad profiler.

O-146 activated that initial design; O-147 raised the submission/resource caps;
O-148 then removed the numeric submission stop while retaining concurrency one and
the `1.10` GH200-hour aggregate ceiling. Envelope A is **CONSUMED/CLOSED** after
12 serial submissions and `0.516389` GH200-hours. Camera passed correctness/parity/
end-to-end/memory checks but failed the unchanged `<=0.80` optimized-pooling ratio
gate at `0.976174`; LiDAR engineering qualification passed. No capability metric,
optimizer update, D_select, D_audit, official validation or selectable scientific
checkpoint ran. Unused budget is not continuing authority.

### 10.4 Envelope B — scientific branch qualification

The frozen Envelope-B design contains exactly two primary candidates: one Camera and
one LiDAR. O-150 qualifies the PyTorch fallback as the Camera production backend and
removes the former CUDA-performance blocker. Each candidate uses seed 0, 20 exact-CBGS
epochs over D_fit, physical B4,
accumulation 8/effective B32, accepted S08 precision, terminal-only checkpoint
selection, one `D_select` evaluation, and conditionally one owner-unsealed `D_audit`
evaluation. No NuImages, GN, alternate LR, alternate seed, or automatic scientific
repair is inside the initial envelope.

The default serial order is LiDAR then Camera because LiDAR exercises the role-bound
GTDB and shared recipe first. A scientifically weak LiDAR result does not cancel the
independent Camera primary unless the failure implicates a shared data, evaluator,
precision, or configuration boundary. Maximum concurrency is one.

Envelope-B GPU-hours are calculated only from the materialized consumed CBGS
exposure, exact-graph production calibration, evaluator timing, checkpoint/resume
overhead, and a declared 15% aggregate contingency:

```text
N_consumed = 87,904 samples/epoch
H_B = 1.15 * (20*N_consumed/throughput_C
            + 20*N_consumed/throughput_L
            + 0.80 hours evaluator/checkpoint/sealed-audit reserve)
```

The materialized estimate uses `1,758,080` consumed presentations per candidate,
Job H fallback throughput `16.3513808747 samples/s`, Job B5 LiDAR throughput
`41.9043778095 samples/s`, a conservative combined `0.80` hours for D_select,
checkpoint/preflight and still-sealed D_audit reserve, then the frozen 15%
contingency. The result is `48.668420` hours, rounded up to an exact aggregate
ceiling of `49.0` charged GH200-hours. Planned single-GH200 wall segments are
`14:00:00` LiDAR and `35:00:00` Camera; maximum concurrency is one and no numeric
engineering-submission cap is set under O-149. The exact configs, hashes, output
root, commands and escalation rules are in `RUN_REQUEST.md` Section 7.

D_audit time is only reserved: the data remain sealed until explicit `P1-G2 OPEN
D_audit`, and that later opening still requires an exact config/provenance amendment.
The calibration reused prefetched batches rather than proving sustained loader
throughput; the 15% contingency covers this residual, supported by S09's observed
`0.076%` eight-worker data wait. The ceiling is actual charge, not planned
consumption. No estimate is derived solely from the old C1 graph.

### 10.5 In-envelope remediation and mandatory escalation

Under O-149, once an engineering-validation envelope is explicitly activated, S00
may autonomously diagnose and fix unambiguous single-correct-answer defects anchored
to frozen semantics: tests/fixtures, config/schema parsing, discrete dtype/API,
runner/Slurm plumbing, checkpoint/resume I/O, artifact publication/provenance and
logging. It records each derived source/command/fresh output and resubmits serially
inside the same aggregate GPU-hour ceiling and concurrency. Submission count has no
default numeric cap unless the owner explicitly sets one. Every scientific run still
binds a durable Git SHA and resolved-config hash; raw outputs remain immutable.

Return to the owner before changing model math or tensor shapes, normalization,
initialization source/mapping, data role/content/order or GTDB membership, augmentation,
loss/target/decode, optimizer/scheduler/EMA/precision, seed, exposure, selectable
checkpoint, evaluator/metric, candidate count, interpretation, resources, or output
scope. Also stop on uncertain classification, the same root blocker recurring after
repair, or an exhausted aggregate resource ceiling. Blind identical retries remain
forbidden. There is no automatic cause-directed scientific repair in either initial
envelope, and O-149 creates no standing compute authority.

## 11. Frozen fields and remaining activation inputs

### 11.1 Owner-frozen by O-144 plus O-145

- reference BatchNorm throughout the selected convolutional graph; Swin LayerNorm
  retained; no GN candidate;
- LiDAR keyframe-only train and ten-sweep evaluation;
- physical B4, accumulation 8, effective optimizer B32, scheduler per optimizer update,
  activation checkpointing off, and redundant telemetry synchronization off;
- exact standalone reference Camera graph;
- PyTorch sorted `segment_reduce` as the Camera production BEV-pooling backend;
  independent in-tree CUDA pooling retained as an unpromoted explicit option, with
  the completed WP2/WP4 forward, backward, FP16/FP32-policy, operator-timing, and
  end-to-end evidence retained but no `1.25x` capability prerequisite;
- Camera ImageNet-1K Swin-T primary initialization;
- Camera CenterHead; LiDAR/Fusion TransFusionHead;
- reference L/F SECOND+SECONDFPN decoder rather than the current shallow shared neck;
- the exact Camera/LiDAR optimizer, LR, weight decay, cyclic schedule, warm-up,
  clipping, augmentation, GT-paste, EMA-off, and 20-epoch bundles in Sections 3 and 7;
- exact archived MIT CBGS algorithm;
- role-bound D_fit-only LiDAR GTDB/GT-paste;
- no arbitrary predeclared numeric capability threshold;
- seed 0, two total primary candidates, epoch-20 terminal-only selection, one
  `D_select` evaluation, and owner-unsealed one-time `D_audit` use;
- NuImages checkpoint as a conditional external comparison/fallback, not an automatic
  candidate or primary initializer;
- the five collaboration simplifications, five WPs, three owner gates, and two
  approval envelopes in Section 10.

### 11.2 Pending measurements or activation records, not open recipe choices

- owner activation of the exact Envelope-B object in `RUN_REQUEST.md` Section 7;
- one independent C/L recipe-freeze review at the activation baseline, with no open
  P0-P2 before compute;
- the two terminal training/checkpoint/D_select results and actual charged time;
- the P1-G2 owner disposition for each branch and any explicit `OPEN D_audit` action;
- the later Alvis checkpoint/provenance/evaluator alignment audit for Phase II;
- later Protocol-B BN buffer/affine aggregation policy and the final `D_base/D_tail`
  construction.

## 12. Evidence and fixed external references

Local evidence:

- `HANDOFF.md` — active S10 status, O-143 order, and collaboration boundary;
- `RESULTS.md` — accepted STOP-A split, C0/C1 timing, gradient, and metric evidence;
- `../S08/MODEL_RECIPE_AUDIT.md` — current graph and fixed-reference differences;
- `../../../src/fl_v3/models/fusion/{detector,head,bev_neck}.py` — current shared-head
  implementation;
- `../../../src/fl_v3/data/nuscenes/cbgs.py` — current non-reference sqrt-RFS.

Pinned MIT BEVFusion reference, commit
`326653dc06e0938edf1aae7d01efcd158ba83de5`:

- [README and official training commands](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/README.md)
- [Camera configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml)
- [Camera parent optimizer/warm-up configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/centerhead/lssfpn/default.yaml)
- [Camera augmentation configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/centerhead/lssfpn/camera/default.yaml)
- [LiDAR configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml)
- [LiDAR GT-paste stop configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/lidar/default.yaml)
- [Fusion configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/camera%2Blidar/swint_v0p075/default.yaml)
- [nuScenes base augmentation/CBGS configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/default.yaml)
- [Detection train/eval sweep policy](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/default.yaml)
- [SECOND/SECONDFPN configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/default.yaml)
- [TransFusionHead configuration](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/default.yaml)
- [Official CBGSDataset implementation](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/datasets/dataset_wrappers.py)
- [Optimized BEV-pooling Python/autograd wrapper](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/ops/bev_pool/bev_pool.py)
- [Optimized BEV-pooling CUDA source](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/ops/bev_pool/src/bev_pool_cuda.cu)
- [Pinned BEVFusion Apache-2.0 license](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/LICENSE)
- [Official pretrained-checkpoint download script](https://raw.githubusercontent.com/mit-han-lab/bevfusion/326653dc06e0938edf1aae7d01efcd158ba83de5/tools/download_pretrained.sh)
- [Reference-YAML ImageNet Swin-T checkpoint](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)
- [Swin Transformer MIT license](https://raw.githubusercontent.com/microsoft/Swin-Transformer/main/LICENSE)
- [Official nuImages dataset description](https://www.nuscenes.org/nuimages)
- [Torchpack recursive configuration merge](https://torchpack.readthedocs.io/en/latest/_modules/torchpack/utils/config.html)
- [MMCV 1.4 cyclic LR defaults](https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/hooks/lr_updater.py)
- [MMCV 1.4 cyclic momentum defaults](https://raw.githubusercontent.com/open-mmlab/mmcv/v1.4.0/mmcv/runner/hooks/momentum_updater.py)
