# USENIX Security '27 Orchestra — strong CL backbone to federated multimodal security

> **Status:** active. The owner designated the pinned S00 task at
> `detached@f262f6bea037580065a8505008773c04fdd259f5` as the sole canonical writer
> on 2026-07-10. S01 is integrated into the dedicated S07-A branch, and S07-A is
> independently reviewed **PASS for the data-foundation phase** at delivery
> `ba1571632557c20adbda3172221694cdbecfeabe`, executable INT-A
> `44cefd06bc815e893919d95c754896711dba3402`, and review artifact
> `370ea6c0bd4d9d737a5a50b6aff1c6f742589825`. Job `335280` closed the two
> prior provenance P1 findings plus locale-stable source attestation with 7/7
> focused tests. No merge to `v3-ad-perception` or push has occurred. The full
> trainval `t1.v2` cache remains unapproved/unexecuted, and S07-B model readiness
> remains a separate gate. Wave-A S02-S05 is active from common base `372de939`:
> S02 delivered `7ad396e` after preserved failed Job `335565` and separate passing
> remediation Job `335578`; bounded GPU Job `336713` closed the sole review gap,
> and limited re-review `df142dc` now accepts S02 as a reviewed S07-B dependency.
> S03 preserved its failed infrastructure attempt, then passed exact shared-node
> Job `336708`; independent review `2f62e57` accepts it as a module-level S07-B
> dependency with production-shape limits. S04 Job `335579` closed the sparse
> composition defect but failed two final-BEV fp16 dtype assertions. S04 Job
> `336718` closed those dtype assertions and passed the B=4 gate, but remains a
> preserved 9/10 failure on a tiny-occupancy train-to-eval spconv tuner path; Job
> `336728` confirmed a universal current spconv-2.3.8 fp16-eval dispatch blocker,
> and owner decision O-025 selected the version-guarded spconv-only
> training-dispatch-under-no-grad remediation. Exact Job `341695` passed 15/15,
> and independent review `a0763c2` accepts S04 as a module-level S07-B dependency,
> with same-instance concurrency/reentrancy retained as an S06/S07-B integration
> requirement. S05 preserved failed Job
> `336731` (43/44), corrected only the tuple-valued devkit fixture, and passed the
> separately approved exact Job `336738` (44/44). Final independent review
> `1c44084` accepts worker `a9c801f` as a reviewed S07-B dependency. All failures
> remain negative evidence; no automatic retry or integration PASS has been inferred.
> All S02-S05 modules are now reviewed PASS within their stated limits. The filled
> S06 production-runtime kickoff was owner-approved under O-027 and launched at
> `xhigh` from exact base `968d81583c87ba76b7dbbb722760f8eb8eb6cd39`
> on `codex/s06-production-runtime`. S00 rejected its first two unexecuted runtime
> requests under O-028 and returned fixed-window exposure plus fail-atomic
> checkpoint blockers for scoped remediation. Remediated executable `6696984`
> then ran only the exact bounded synthetic Job `341997`, which is preserved as
> **FAILED 1:0** with 45 passed / 17 failed / 0 skipped. O-029 returns four
> engineering root-cause families plus failure-artifact preservation for a
> second scoped remediation. Exact remediation-2 Job `342014` then completed
> `0:0` with 66/66 tests, zero skips, exact source/environment attestation and
> an in-job verified final manifest. S00 completeness audit accepted final worker
> `6b7ef29` for independent review. S06-R review `ca7bbd7` found no P0-P2 and
> returned a strictly bounded **PASS** under O-031: S06 is now a reviewed S07-B
> candidate dependency, not an integration, production, full-data or scientific PASS.
> The owner approved the exact O-032 S07-B envelope on 2026-07-12. S07-B is now
> active at `xhigh` in task `019f549a-d56e-7843-8916-7ba08a6af276`, worktree
> `d5e7`, from clean `detached@c9c84f8b2caebea14adc1d79d6d706695be0f50f`;
> delivery branch `codex/s07-b-integrated-cl-stack` and the five ordered non-FF
> worker merges are authorized, but no merge has yet been accepted by S00.
> Compute remains absent at kickoff: only a new immutable O-009 request may be
> audited by S00, while full cache, 100/1000-step, profile/metrics and DDP remain
> forbidden without a new exact owner decision. Merge to `v3-ad-perception` and
> push remain forbidden.
> S07-B worker `df13025bc6582b9b436d1df065de75c03e92782d` has now delivered
> a clean static-only integration candidate after the five exact ordered merges;
> S00 independently rechecked merge parents, all five imported review blob/hash
> pairs, direct-edit ownership, and the committed static checks. No S07-B compute
> request or job exists, and the five candidate configs remain intentionally
> non-runnable. Independent S07-B-R is active from that exact worker SHA in task
> `019f54b9-f6a3-7600-b41e-d0b72613c028`, worktree `44c9`; this is not an
> integration, runtime, production, full-data or scientific PASS.
> S12 has delivered its
> evidence/proposal-only handoff for completeness/review and used no compute.
> Remaining architecture choices, cells, and run requests still require their own
> approvals.
> **Venue target:** USENIX Security '27 first submission cycle (registration 2026-08-18 AoE;
> submission 2026-08-25 AoE).
> **Canonical companions:** [`SESSIONS.md`](SESSIONS.md) and
> [`KICKOFFS.md`](KICKOFFS.md).
> **Scope rule:** this directory is a top-level `fl_v3/` workspace, parallel to
> `fl_v3/collab/`. The entire `collab/` tree is now read-only historical evidence
> for this stage; it may be read and cited but receives no new planning, handoff,
> review, or result documents.

## 1. Decision and research objective

We will sprint for the USENIX Security '27 first submission cycle. The immediate
critical path is a strong, trustworthy centralized (CL) nuScenes detector. Attack
and defense claims cannot be interpreted on a detector whose clean accuracy,
modality use, or training semantics are uncertain.

The CL work is an enabling validity requirement, not the paper's principal
security contribution. The intended paper question remains:

> During vendor-style federated adaptation of a strong camera-LiDAR detector to
> rare, geographically and environmentally non-IID fleet data, can a
> modality-localized backdoor hide among legitimate long-tail updates, and can a
> module-/modality-aware defense remove it without rejecting the rare benign
> updates that the adaptation process exists to learn?

The USENIX '27 sprint schedule therefore overlaps three streams:

1. **CL foundation:** fix correctness, choose and freeze a strong modular backbone.
2. **Security mechanism:** prepare the threat model and attack instrumentation in
   parallel; start scientific attack/FL runs as soon as the single-seed fusion
   pilot passes, without waiting for all final CL seeds.
3. **Paper/artifact:** maintain the argument, tables, provenance, and artifact from
   the beginning rather than reconstructing them at the deadline.

## 2. Current baseline: what is and is not trusted

The strongest historical CL result is `bb02d = 0.5656 mAP / 0.5733 NDS` on the full
nuScenes validation split. It is a useful engineering milestone, but it is not the
new FL handoff baseline:

- it predates the newly identified batch-global pillar-cap and Gaussian-target
  findings;
- no valid full-trainval camera-only and LiDAR-only models were independently
  trained, so branch capacity and fusion complementarity are unknown;
- the current sparse 0.075 m implementation does not have the reference SECOND
  resolution flow;
- the historical description that camera/fusion/head are essentially identical
  to MIT BEVFusion is not supported by the current code/reference comparison.

MIT BEVFusion provides orientation values, not a requirement to clone every
component: camera-only `0.3556/0.4121`, LiDAR-only `0.6468/0.6928`, and fusion
`0.6852/0.7138` mAP/NDS. Camera is expected to be weaker than LiDAR; the requirement
is that each branch is competitive for its modality and that fusion reliably
improves over the stronger branch.

### Confirmed blockers before any new full training

1. **Batch-global pillar truncation.** `max_pillars` is applied after sorting a
   batch-global key, so earlier samples are favored and later samples can lose
   LiDAR content (`models/fusion/lidar_encoder.py:105-180`).
2. **Unresolved Gaussian radius semantics.** The current formula mixes a corrected
   quadratic denominator with large roots and matches neither the official legacy
   implementation nor the geometric small-root interpretation
   (`models/fusion/losses.py:37-59`). The intended definition must be chosen,
   pinned, and covered by numerical golden tests.
3. **Dead camera FPN levels.** With `out_stride=16`, the returned tensor depends
   only on Swin levels 2/3; levels 0/1 are computed and discarded
   (`models/fusion/camera_neck.py:61-71`).
4. **Incomplete detection head/decode.** One shared 10-class heatmap and
   class-agnostic regression feed a global top-K; there is no task-wise box NMS
   (`models/fusion/head.py`, `models/fusion/detector.py:180-217`).
5. **Ambiguous architecture resolution.** Config defaults and entry points can
   select pillar versus voxel differently; unknown encoder strings can silently
   fall back to pillar (`training/tasks.py:363-426`).
6. **No production single-modality topology.** Mini diagnostics can mask branches,
   but production training/evaluation always constructs and executes both branches.
7. **Data foundation reviewed, production cache still absent.** S07-A reviewed
   PASS at `ba15716`/`44cefd0`/`370ea6c`: the S01 ZIP history is integrated,
   `build_gt_database.py` binds canonical plus physical pickle/sidecar identities,
   and source attestation is complete and locale-stable. Historical job `332651`
   still supplies only coverage/loader evidence and its `t1.v1` caches remain
   forbidden. Before any production training, the separately owner-approved full
   trainval `t1.v2` cache job must generate and freeze exact cache/sidecar/manifest
   hashes, and S06/S07-B must bind those identities at every production entry point.

### Capability limitations to resolve during architecture freeze

- Camera uses stride-16 features and 1 m depth bins; the reference camera path
  uses an effective stride 8 and 0.5 m depth bins.
- Native 1600x900 images are anisotropically stretched to 704x256. Calibration is
  updated correctly, but pretrained visual appearance is distorted and the
  reference resize/crop/rotation augmentation is absent.
- The sparse encoder downsamples only Z, densifies at the original XY grid, and
  fuses there. For 0.075 m, the current 1440x1440 grid is incorrectly treated as a
  fusion grid; in SECOND it is an input voxel grid and is reduced by approximately
  8x before densification/fusion.
- The 128-channel fuser, shallow BEV neck, and 0.15 M shared head are materially
  smaller/simpler than the reference fused detector.
- Class weights multiply both positive and negative heatmap terms. Combining them
  with CBGS double-balances classes and invalidates the old CBGS negative result.

### Performance limitations to resolve

- The current 0.075 m path uses about 59.2 GiB for a B=2 forward-only probe and
  OOMs at B=4 because it densifies/fuses at 1440x1440.
- Large BEV tensors remain fp32 after view-transform accumulation.
- Pillar encoding performs repeated stable sorts and constructs padded
  `[pillars,max_points,channels]` storage before applying the pillar cap.
- Sparse voxelization loops over samples in Python.
- The centralized trainer rebuilds the DataLoader each epoch, defeating persistent
  workers and repeatedly reopening ZIP metadata.
- Official detection evaluation does not currently enter the configured autocast
  context.
- EMA history is not restored on resume; fp32 non-finite loss handling can still
  take an optimizer step.

## 3. Owner-approved Wave-A module contract — final integration not yet frozen

The owner approved the following implementation contract for parallel S02-S05 work
on 2026-07-11 under O-017. This freezes the primary module choices that those
workers may implement; it does not approve S07-B integration, a full-trainval cache
or model run, numerical CL gates, or a final scientific architecture. S07-B and its
independent review must still reconcile the reviewed module contracts before any
production or scientific execution.

The approved Wave-A primary model is a strong, modular late-BEV detector whose modality
boundaries remain meaningful for attacks and defenses:

- **Camera:** Swin-T initially; effective multi-scale FPN output at stride 8;
  pure-camera LSS/view transform; 0.5 m depth bins; aspect-preserving resize/crop
  and reference-style image augmentation.
- **LiDAR:** SECOND-style sparse encoder at a primary candidate voxel size of
  `0.075 x 0.075 x 0.2 m`; XY downsampling in sparse space to about 1/8 resolution
  before densification.
- **Fusion:** modality-native encoders map into one common low-resolution BEV
  (about 180x180 for the 1440 input grid); 256-channel ConvFuser or an equally
  explicit late-BEV fusion block.
- **Detection:** a faithful multi-task CenterHead with task/class-aware candidate
  selection and rotate/circle NMS is the primary simple baseline. TransFusion is a
  contingency or later generalization backbone, not a prerequisite for starting.
- **Precision:** Arrhenius sparse training uses fp16 AMP plus GradScaler; fp32 is the
  debug/reference mode. Direct sparse bf16 remains unsupported.

We should not blindly copy a LiDAR-conditioned camera depth transform: it would
blur the camera/LiDAR attack boundary. Any cross-conditioned view transform must be
an explicit later experiment, not the primary security backbone.

### Wave-A execution/review state (2026-07-11)

- **S02:** durable delivery `7ad396ebe535ca468337ed44065d39354707e08b`
  contains implementation `65c83c077210469861ba722a285ab1e58e6d719f`.
  Job `335565` remains `FAILED 1:0` after 12 pytest passes because its original
  JUnit aggregation/final checksum stage failed. Separately approved parser-only
  remediation Job `335578` is `COMPLETED 0:0`, 12/0/0/0 with final checksums OK.
  S02-R at `fb17da3ea55a93d7709f6a2b5f6e4bb6adc0bf7e` independently
  found no confirmed per-sample-cap or Gaussian defect but returned one blocker:
  the binding GPU forward/backward gate was NOT RUN. Exact Job `336713` then passed
  the bounded B=3 synthetic CUDA gate at delivery `3aebf2d`, and limited re-review
  `df142dc9a391b87d05bd7becaba59459e9659f88` resolved the sole P1. S02 is
  accepted only as a reviewed S07-B dependency; integration/full-data/scientific
  readiness remain separate.
- **S03:** implementation `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`
  first encountered two preserved infrastructure failures. The first submission attempt was rejected client-side
  for missing scheduler account/partition. Corrected Job `335630` is `FAILED 1:0`
  in six seconds before environment/tests/artifacts because the compute node could
  not resolve the Codex `/home` linked-worktree Git metadata. `sacct` reported
  four allocated/billed GPUs despite a one-GPU request; a shared `/nobackup`
  immutable execution source and shared-allocation remediation produced passing
  Job `336708` at delivery `5089383`: 10/10 tests, one GH200/eight CPUs,
  `OverSubscribe=OK`, and all snapshot/source/artifact checksums. Independent
  review `2f62e570c9c24ef1e18a483888c3f28ad56a415e` accepts S03 as a
  module-level S07-B dependency only; production-shape profile/tiny-overfit/100-step
  and final cross-module opt-in remain separate.
- **S04:** Job `335566` failed five composition cases; remediation Job `335579`
  proves the composition fix (8/10 pass) but remains `FAILED 1:0` because both
  fp16-path cases returned a final fp32 BEV. The approved contract is not weakened:
  the worker must preserve both jobs, make the fp16 nonempty/empty output contract
  consistent while retaining fp32 reference behavior, and obtain a new exact
  request before any third smoke. Job `336718` then passed the original dtype
  assertions and the B=4 fp16 forward/backward/memory subgate, but is still
  `FAILED 1:0` (9/10): reusing the same fp16 model after train/backward and
  switching to eval on six active voxels caused spconv's inference tuner to find
  no suitable algorithm. Diagnostic Job `336728` completed seven isolated cells:
  all six fp16 eval variants failed, including fresh/large/cache/order controls,
  while fresh fp32 eval passed. The current spconv 2.3.8 eval dispatch therefore
  fails on fp16 features with fp32 filters/output; the cause is not low occupancy,
  backward, model reuse, or cache order. S04 remains CHANGES-REQUESTED at delivery
  `49f26de`. O-025 selected a narrowly scoped workaround: keep the encoder and
  normalization layers in eval under `no_grad`, but route only spconv sparse
  convolution modules through the installed-version training dispatch so their
  custom-fwd casts filters/features coherently. Implementation must fail closed on
  an unsupported spconv version and prove no gradients, parameter/master-weight
  mutation, state mutation, or normalization-mode drift before independent review.
  Executable `84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f` implements that
  contract. Exact Job `341695` completed `0:0` with 15/15 tests, exact spconv
  2.3.8/source/request/identity attestation, and separate B=4 fp16 train/eval
  memory evidence. Final worker delivery
  `483e149b95ec891b675df825d924a96bb225b7dd` preserves Jobs `335566`,
  `335579`, and `336718` as failures and Job `336728` as diagnostic completeness.
  Independent review `a0763c2e0b322d4ca53a92f9f69c90d9b231bbff` / REVIEW
  SHA-256 `8673672793235ae0226d9109c73cd39577d5f40e846b17425178a7011300ea2a`
  returns **PASS for the S04 bounded synthetic module gate**. S04 is accepted as
  a reviewed S07-B dependency only. The same encoder instance must be serialized
  in S06/S07-B, or receive instance-level concurrency protection plus adversarial
  reentrancy tests before any thread-safety claim.
- **S05:** worker delivery `4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9`
  (implementation `9fd3281651ef006a175ed9462e7bf1eaf3437357`) was
  independently reviewed at `c81826251349ede7c514950df785e4fe05d60192`.
  Verdict is **CHANGES-REQUESTED**, not integration PASS: decode must force FP32
  before sigmoid/top-K/threshold/NMS; submission order must be total over
  metric-relevant serialized content; exported NMS helpers must fail closed on
  single-box invalid geometry and non-positive budgets. Scoped remediation is
  durable at `705216de097ae9eeb1813de6dcdc916e2844fcde` (implementation/tests
  `753944c199ceeace160732218f1b16dfdd15ac21`) with all three findings mapped to
  hostile fixtures. Exact Job `336731` then preserved a 43/44 failure: the full
  forward/reverse content-order assertion passed, but the expected velocity
  container was a list while nuscenes-devkit returns the stable tuple `(vx, vy)`.
  Execution commit `96e509b71a3e22afb4de397132438fd3b9bbf5d8` changes only those two expected
  containers; production code, O-018 semantics, and ordering assertions are
  unchanged. Under the owner's temporary S02-S05 validation delegation—not an
  O-009 expansion—Job `336738` passed 44/44 on one shared GH200/eight CPUs with
  exact execution/source/request/launcher identity and all nine in-job checksum
  targets. Final worker delivery `a9c801fdee378906e54d06314d0c772b6559901a`
  preserves both jobs. Independent review
  `1c440843bb2b6d72f10310ff11fcde0d7d1e885c` / REVIEW SHA-256
  `67b58c8e9d1d1622d1af49a2c052cbadd66580500dbf988fc1184f2d0df6736e`
  returns **PASS for implementation and the focused synthetic runtime gate**.
  S05 is accepted only as a reviewed S07-B dependency; detector/loss/config
  integration, official end-to-end evaluation, production-volume rotate-NMS
  profiling, and scientific readiness remain separate gates.

## 4. CL scientific matrix and gates

### Pre-full-run engineering gate

All must pass before Slurm full training:

- directory/ZIP decode parity and 100% train/val image, key-LiDAR, and 10-sweep
  member coverage;
- per-sample pillar cap plus sample/batch permutation invariance;
- pinned Gaussian golden values and target-render tests;
- production `camera_only`, `lidar_only`, and `fusion` modes that skip unused I/O
  and computation;
- intended trainable parameters all receive finite gradients;
- identical resolved architecture/config hashes across CL, FL, train, resume, and
  eval entry points;
- 100-step and 1000-step capped trainval runs decrease loss without OOM, NaN,
  persistent GradScaler skips, or optimizer/scheduler/EMA step drift;
- full-data, not mini-only, throughput and memory profile.

### Candidate single-seed selection matrix — pending owner decision

The current planning proposal uses the full 28,130-sample train split, full
validation split, one screening seed, matched effective global batch/optimizer
steps/precision/decode, and approximately 20-24 epochs unless a step-based budget
is approved. These cells and budgets are not yet an approved matrix.

If approved, this matrix is the **backbone capability/architecture benchmark**. It
does not by
itself define the final federated data protocol, and its full-train weights must not
be reused as the base checkpoint of a tail-adaptation experiment if they have seen
the client-tail partition. After the architecture is frozen, the final federated
base model is retrained using only the data allowed by the approved FL protocol.

Listing a cell here is not permission to submit it. Every full run or matrix needs
an owner-approved `RUN_REQUEST.md` bound to an exact commit, resolved config, data
manifest, cells, seeds, GPU/time request, and output path.

| Cell | Purpose |
|---|---|
| `L-P020` | repaired legacy 0.2 m pillar LiDAR-only control |
| `L-S075` | SECOND-style 0.075x0.075x0.2 LiDAR-only primary candidate |
| `C-STR8` | corrected stride-8, multi-scale camera-only candidate |
| `F-U` | selected C/L fusion with uniform heatmap weighting and no CBGS |
| `F-CBGS` | CBGS replaces class weights; schedule corrected for expanded steps |

If the future FL initialization policy is not fixed before these jobs, add one
fusion initialization A/B: public/joint initialization versus CL branch warm-start.
That proposal would give 12 rather than 11 full runs after final seeds; neither run
count is currently approved.

### Draft handoff floors — not yet pre-registered

These are planning values relative to a matched-complexity reference, not approved
gates or SOTA claims. S00 may revise the recommendation using reviewed S07
engineering/profile evidence, but the owner must freeze the numerical gate before
the corresponding S08/S09/S10 scientific run is approved.

| Model | mAP / NDS gate |
|---|---:|
| Camera-only | at least `0.32 / 0.38` |
| LiDAR-only | at least `0.61 / 0.66` |
| Fusion no-go | below `0.62 / 0.66` |
| Fusion handoff target | at least `0.64 / 0.68` |
| Fusion stretch target | at least `0.67 / 0.70` |

Fusion must also exceed the independently trained paired-seed LiDAR model by at
least `+0.02 mAP / +0.01 NDS`, with scene-bootstrap 95% CI lower bound above zero.
At least 7/10 classes should not regress, and an unexplained class drop above 0.02
is a blocking finding.

Every selected checkpoint reports mAP/NDS, per-class AP, all five nuScenes TP
errors, recall, train/validation curves, and range/visibility/number-of-LiDAR-points/
day-night/rain slices.

### Two different modality tests

Both are mandatory:

1. **Independent C/L/F training** measures the capacity of each topology.
2. **Same-fusion-checkpoint intervention** (`camera-zero`, `lidar-zero`, camera
   shuffle/misalignment) measures what the fused model actually uses.

Define `fusion_gain = F - max(C,L)` globally and for each slice. Modality masking
does not replace independently trained branch models.

### Draft performance gate for FL delivery — pending owner decision

These values are an engineering proposal. The instrumentation is required, while
the numerical acceptance thresholds may be revised from reviewed S07 profiling and
then frozen before capability runs are judged.

- chosen microbatch at least 4/GH200, or accumulation reaches the matched effective
  batch without schedule drift;
- peak allocated/reserved memory below 80% of device memory;
- data wait below 15% and p95 step time below 1.25x p50 after warm-up;
- no non-finite updates and at least 99% intended optimizer steps executed;
- report p50/p95 step time, samples/s, epoch wall time, GPU utilization, data wait,
  peak memory, CPU/RSS, and evaluation time;
- reject a candidate that gains less than 1 mAP while making training/FL steps more
  than 2x slower, unless it is required for a security generalization claim;
- record parameter/upload bytes because every added parameter is multiplied by
  clients and rounds.

## 5. CL-to-FL protocols

The two protocols answer different research questions and must never share a label,
baseline, or checkpoint provenance. The owner has approved **Protocol B as the
primary security setting** and **Protocol A as the clean optimization/control
setting**. Any reversal or mixed protocol requires an explicit new owner decision.

### Protocol A — nuScenes-scratch federated training

Clients receive the **frozen architecture**, not a detector trained on nuScenes.
Every client starts from one identical initialization and jointly trains the global
model over client-partitioned nuScenes data.

“From scratch” must be resolved into one of the following before a run:

- **nuScenes-scratch/public-init:** public ImageNet or explicitly allowed NuImages
  camera pretraining may be used, while no weight has seen nuScenes detection data;
  LiDAR/fusion/head follow the declared common initialization. This is the useful
  primary version of Protocol A.
- **fully random:** all trainable detector weights are random. This is an optional
  optimization stress test, not a default realism claim.

Required controls are a centralized pooled model from the same initialization and
data, identical effective data exposure/optimizer-step accounting, and clean
FedAvg/FedAdam or another predeclared optimizer comparison. Protocol A asks whether
FL optimization can recover the strong centralized capability. If it produces a
weak model, attack/defense results on that model are diagnostic only. A utility gate
relative to the matched centralized model and an absolute clean-detection floor
must be approved before security claims are unlocked.

Protocol A is scientifically useful, but making from-scratch FL match CL is a large
optimization problem of its own. It should not silently consume or replace the
paper's security contribution.

### Protocol B — centralized base plus federated tail adaptation (primary)

This protocol closely matches a plausible autonomous-driving vendor workflow:

1. The vendor owns a broad, common-data pool `D_base` and trains a strong base
   detector `W_base` using the already validated architecture.
2. Regional fleet centers, data silos, or edge training clusters collect new rare
   classes, conditions, locations, weather, sensor regimes, or failure cases in a
   disjoint `D_tail`.
3. Every client receives the same `W_base` and fine-tunes on its local tail data;
   the server aggregates these updates.
4. A malicious client poisons its legitimate tail-adaptation stream or its model
   update. Defenses must distinguish that attack from benign tail updates.

The recommended system unit is a **regional/fleet data silo**, not a claim that each
car trains the full detector onboard. Full-model fine-tuning, selected-module
fine-tuning, and parameter-efficient updates have different bandwidth and attack
surfaces; the primary update scope must be frozen in the threat model.

Protocol B is particularly valuable for the paper because benign long-tail updates
are naturally sparse, non-IID, and sometimes module-local. They may resemble the
updates that a modality-localized attacker uses. This gives a concrete systems
security tension: a whole-model robust aggregator may either miss the attacker or
reject exactly the rare benign data the fleet needs to learn.

### Mandatory split and leakage rules for Protocol B

- Build `D_base` and `D_tail` from the official **training** split only. The
  official validation/test data remain completely held out.
- Split at least at **scene/log level**. Adjacent keyframes, sweeps, duplicated raw
  sensor files, and the same scene must not cross base, client, or evaluation
  boundaries.
- Do not create the split by moving individual annotations while the same camera/
  LiDAR frame remains in both pools. One frame can contain common and rare objects;
  the unit of ownership is the raw sample/scene, not an annotation row.
- Define “tail” before attack experiments using frozen train-only rules: class
  frequency, distance/visibility, weather/time, location, sensor condition, or a
  predeclared combination. Do not tune the definition to maximize ASR or use final
  validation outcomes.
- Reserve evaluation support for the declared tail conditions. Apply the already
  frozen tail definition to official validation for reporting; if it has
  insufficient support, reserve a scene/log-disjoint `E_tail` before training and
  exclude it from every client and centralized training pool.
- Freeze and hash the split, eligibility rules, class/condition statistics, and
  client assignment before training. Every experimental cell consumes the same
  artifacts.
- `W_base` must never have trained on `D_tail`. The full-train CL checkpoint used
  for architecture capability is therefore not a valid Protocol-B initializer.
  Retrain the frozen architecture on `D_base` to obtain the scientific `W_base`.
- The base pool must contain enough support for the target classes to make clean
  eligibility meaningful; “tail” should mean rare contexts/examples, not removal
  of all knowledge needed to define the task.
- Once the Protocol-B split/statistics are inspected, the architecture and primary
  recipe cannot be changed to exploit `D_tail` outcomes. The earlier full-train
  capability phase must be disclosed as platform development; its weights and
  scores are not Protocol-B evidence. Any tail-conditioned redesign is a separately
  labeled ablation and requires a clean re-freeze.

### Required clean baselines for Protocol B

All use the frozen split and initialization:

1. `W_base`: CL on `D_base` only.
2. `W_oracle`: centralized continuation on pooled `D_tail` (or `D_base + D_tail`,
   whichever matches the declared exposure) as the non-private upper bound.
3. Local-only client fine-tuning without aggregation.
4. Clean federated fine-tuning on the tail clients.
5. Attacked federated fine-tuning.
6. Defended federated fine-tuning under the same clean baseline and budget.

Report overall mAP/NDS, common-data retention, tail-class/condition improvement,
catastrophic forgetting, client-level dispersion, communication/compute cost, and
then ASR/false-trigger/defense FPR. A defense that suppresses tail learning is not a
successful defense even if aggregate mAP changes little.

### Relationship between the protocols

- Protocol A measures the clean federated optimization gap from a public/common
  initialization and remains an important control.
- Protocol B is the primary threat model because it starts from a
  capable deployed model and places the attacker in the realistic adaptation
  stage.
- Results must be labeled `federated training` versus `federated fine-tuning`.
  Checkpoints, data manifests, and claims cannot be mixed between them.
- The Orchestra records the approved protocol, split, client unit, update scope,
  and initialization before any full FL or security job is authorized.

### Scientific audit and novelty boundary

The deployment logic is credible, but the setting alone is not a novelty claim:

- [FedDrive](https://arxiv.org/abs/2202.13670) already treats autonomous-driving
  client data as heterogeneous visual domains, and
  [FedDrive v2](https://arxiv.org/abs/2309.13336) explicitly studies client label
  skew/class imbalance.
- [Federated Deep Learning Meets Autonomous Vehicle Perception](https://arxiv.org/abs/2206.01748)
  explicitly motivates FL using rare and occluded instances collected by vehicles
  and road sensors.
- [AutoFed](https://tianyuez.github.io/pubs/autofed.pdf) studies multimodal,
  environmental, annotation, and client-selection heterogeneity in federated
  autonomous-driving perception.
- [BalanceFL](https://research.cuhk.edu.hk/en/publications/balancefl-addressing-class-imbalance-in-long-tail-federated-learn-2/)
  shows that federated long-tail/class-imbalance learning is itself an established
  problem.

Therefore the paper contribution cannot be “federated learning learns rare driving
data.” The proposed security novelty is the interaction between **legitimate
long-tail adaptation**, **modality-localized backdoor updates**, **geographic/domain
non-IID drift**, and **structure-aware defense**. S12 must continue a systematic
literature audit before any “first” wording is approved.

## 6. USENIX '27 sprint execution calendar

The calendar is deliberately overlapping and assumes several isolated sessions/
worktrees run in parallel.

| Window | Required outcome |
|---|---|
| Jul 10-12 | owner approves architecture contract, session split, CL/FL initialization policy |
| Jul 11-16 | ZIP backend, P0 fixes, camera/LiDAR/head modules, trainer modes in parallel |
| Jul 16-20 | integration, 100/1000-step gates, full-data profiling, candidate configs frozen |
| Jul 20-29 | camera and LiDAR full single-seed runs; fusion single-seed jobs start as branch checkpoints arrive |
| Jul 27-Aug 5 | fusion recipe selection, remaining CL seeds, `CL-PILOT` then `CL-FREEZE` |
| Jul 29-Aug 12 | preliminary FL/attack mechanism and defense runs on the passing pilot/frozen model |
| Aug 8-17 | adaptive/generalization experiments, result-table freeze, artifact dry run |
| Aug 15-18 | fixed title/authors/topics/abstract approved; mandatory registration by Aug 18 AoE |
| Aug 18-25 | paper and artifact freeze; submission by Aug 25 AoE |

Two distinct unlocks keep the sprint moving:

- **`CL-PILOT`:** one fusion seed passes the absolute and fusion-gain gates. It
  unlocks preliminary FL/attack experiments.
- **`CL-FREEZE`:** C/L/F each have three seeds, the architecture/config/checkpoint
  schema is frozen, and all final paper security results must use this version.

## 7. Session collaboration rules

Detailed session contracts and dependencies are in [`SESSIONS.md`](SESSIONS.md);
copy-ready worker and independent-review startup prompts are in
[`KICKOFFS.md`](KICKOFFS.md). The following rules apply to every session:

1. The Orchestra session is the sole writer of these three canonical documents.
   Worker sessions write their durable handoff package under
   `handoffs/Sxx/` and do not edit the canonical files concurrently.
2. Every implementation session uses an owner-approved, Codex-provisioned isolated
   worktree/branch and declares its file ownership before editing. Integration is
   performed only after review. S00 must instantiate the exact `Sxx` and `Sxx-R`
   prompts from `KICKOFFS.md`, including the pinned base/worker SHA, expected branch,
   and file ownership, so every fresh session receives the required context and
   authorization boundary. Workers and reviewers verify this topology but do not
   create, move, remove, prune, or switch worktrees/branches themselves.
3. Each worker reads `AGENTS.md`, `fl_v3/docs/env.md`, this file, and its session
   contract. Historical `model_capability` conclusions are not treated as current
   requirements where they conflict with this document.
4. Mini is engineering-only. Scientific verdicts use full trainval/full val and
   record hardware, precision, seed, split, resolved config hash, and checkpoint
   checksum.
5. No worker silently changes the data split, effective batch, optimizer steps,
   precision, decoder, or metric. Required changes return to Orchestra for approval.
6. Build and review should be separate sessions for P0, geometry, metric, or
   scientific-result changes. A merge requires objective gate evidence.
7. The Orchestra, not workers, decides whether a negative result kills a candidate,
   requests a rerun, or changes the paper claim.
8. The standing O-009 policy authorizes only bounded, non-scientific engineering
   smoke jobs. No session may submit a full-data gate, full run/evaluation, matrix,
   multi-seed campaign, automatic resubmission, remote upload, or publication
   action without explicit owner permission scoped to the exact action.

### Reasoning effort and standing short-smoke policy

- S00 explicitly passes `xhigh` when creating every new worker/reviewer task; it
  must not silently inherit a host default of `ultra`.
- `ultra` is reserved for a task whose exact envelope contains a recorded reason:
  unusually complex implementation/review or unusually broad, difficult research.
  Ordinary implementation, review, orchestration, and evidence work use `xhigh`.
- Owner decision O-009 is standing authorization for a **short engineering smoke**
  on an Arrhenius compute node. It is non-scientific and must satisfy all of:
  one node, at most one GPU, at most 60 minutes requested walltime per job, at most
  one active job for the session, and at most two cumulative GPU-hours before a new
  owner decision. Job arrays, multi-node/DDP, multiple seeds/cells, full epochs,
  full trainval coverage/profile/evaluation, 100/1000-step gates, and automatic
  resubmission are outside this authorization.
- Before submission the session writes `RUN_REQUEST.md` as an audit record with the
  exact HEAD plus working-tree diff hash, command, bounded data/sample scope,
  resources, output path, and stop criteria, and cites `O-009`. It may then submit
  without waiting for another approval. Every job ID, log, exit status, retry, and
  negative result is recorded in `HANDOFF.md`/`RESULTS.md` as applicable.
- Reaching a boundary above, repeating the same failure twice, or seeking a result
  used for a gate/table/metric returns to S00/owner for exact approval. Full tests,
  full-data runs, profiles, scientific metrics, matrices, seeds, reruns, and spare-
  GPU expansion always require owner review.

### Evidence-driven Orchestra refinement

S00 is expected to refine downstream work as evidence arrives; the 15-session plan
is a controlled dependency graph, not an immutable script. It may draft a
provisional refinement immediately after reading a complete handoff, but it does
not issue a dependent technical kickoff from that refinement until independent
review accepts the evidence. For each completed worker session, S00 follows this
order:

1. inspect the actual diff/artifacts and check that the handoff package is complete;
2. prepare and show the independent `Sxx-R` launch packet from the exact worker
   SHA/diff; after owner authorization, create that review task through Codex;
3. resolve blocking findings or record an accepted review verdict;
4. update the status and decision ledgers;
5. refine the plans and kickoff envelopes of sessions that have not started;
6. request owner approval before any material or locked scientific change.

S00 may independently make **operational refinements**: launch order, dependency
edges, required reading, file ownership that avoids conflicts, requested handoff
evidence, review emphasis, and clarification of already approved requirements.
These refinements must cite the triggering handoff/review and be logged before the
new kickoff is issued.

S00 may draft from a worker handoff and may immediately delay/return work because
of a blocking review finding. It may not treat a delivered technical contract as
an accepted dependency until review approves it. Before creating any new Sxx or
Sxx-R task, S00 presents a launch packet containing the relevant upstream handoffs,
reviews, SHAs/diffs/artifact status, unresolved conflicts, and the complete filled
kickoff envelope/prompt including reasoning and compute scope. The owner reviews
that packet and explicitly authorizes launch; only then may S00 create the isolated
task directly through Codex. There is no automatic downstream or reviewer launch.

The following are **material or locked changes** and remain owner decisions:

- Protocol A/B roles, threat model, claims, or FL system assumptions;
- `D_base`/`D_tail`/client/evaluation ownership or split construction;
- primary architecture, head, decoder, metric, precision, or initialization;
- experiment cells, seeds, gates, run priority, resource budget, or stop criteria;
- Slurm/full-run/rerun permission, uploads, commits/merges/pushes, or publication.

S00 cannot weaken a gate after seeing results, erase or relabel a failed/negative
cell, change a completed session retroactively, or silently redirect an active
session. Any active-session amendment is written into the ledger, sent to the
worker, acknowledged in `HANDOFF.md`, and re-reviewed if it changes delivered
semantics.

### Worktree provisioning contract

Worktrees are provisioned by selecting `Worktree` and the starting branch in the
Codex task-creation UI, not by a worker prompt. A normal Codex-managed worktree is
detached at the selected branch's HEAD; that is the expected default. Before a task
is opened, S00 produces a kickoff envelope containing:

This policy follows the [official Codex worktree
contract](https://learn.chatgpt.com/docs/environments/git-worktrees): managed
worktrees are task-scoped, start detached at the selected branch HEAD, and a named
branch cannot be checked out in multiple worktrees at once.

```text
SESSION_ID:
BASE_SHA:
SOURCE_BRANCH: v3-ad-perception
EXPECTED_REF_MODE: detached@BASE_SHA, or exact owner-created branch
WORKTREE_PROVISIONED_BY: S00 through Codex after owner launch approval, or owner UI
FILE_OWNERSHIP:
UPSTREAM_HANDOFFS_AND_SHAS:
WORKER_SHA: pending for Sxx, or exact worker commit for Sxx-R
DELIVERY_REF: pending owner authorization for Sxx, or exact review source ref
REASONING_EFFORT: xhigh, or ultra with the recorded task-specific reason
APPROVED_COMPUTE: none | standing short-smoke policy O-009 | exact approved request
```

The session verifies its repository root, HEAD, branch/detached state, and status
before editing. A detached HEAD is valid when it matches `BASE_SHA`. If the
UI-created worktree uses the wrong base, lacks the canonical documents, or conflicts
with another checked-out branch, it stops and reports the exact mismatch; it does
not repair the worktree itself.

Parallel module workers start from one pinned integration `BASE_SHA`. When a worker
finishes, S00 first checks the handoff; the owner then explicitly authorizes any
local handoff commit and `Create branch here` action needed to make that exact
version durable. That narrow permission does not authorize merge or push. Review
tasks use a distinct UI-created worktree at the resulting `WORKER_SHA`; a unique
review branch is used only when the owner authorizes preserving `REVIEW.md` as a
commit. S07 alone receives a dedicated integration worktree based on the approved
integration SHA and integrates only accepted worker commits. Execution sessions
start from the exact frozen candidate commit named in their approved request.

A commit made in an Sxx worktree advances only that worktree's detached HEAD or its
owner-authorized scoped branch. It does not modify S00's working tree, move
`v3-ad-perception`, or integrate itself. Git objects are shared by the repository,
but the worker version becomes a review baseline only after it has a durable
`WORKER_SHA` and branch/ref. S00 keeps canonical-document changes in its own
worktree; S07 later integrates only independently accepted worker commits.

The canonical Orchestra documents must be committed/landed on the source branch
before S00 or worker worktrees are spawned. Although Codex can apply selected local
changes when it creates a managed worktree, such state has no immutable SHA and is
not a valid base for this reproducible multi-session wave.

S00 is long-lived and should use a permanent worktree or keep its managed task
pinned. Workers/reviewers normally use one managed worktree per task and remain
pinned until their handoff/review is durably recorded and accepted. Because Codex
may clean up older managed worktrees, do not archive a task or rely on worktree
retention before its accepted artifacts are landed.

### Durable session delivery

Every worker session creates
`fl_v3/usenix27_orchestra/handoffs/Sxx/` with:

- **`HANDOFF.md` (mandatory):** exact scope, worktree/branch/base/commit, files and
  semantic changes, references/equations, tests/jobs and raw outputs, gate-by-gate
  evidence, artifacts/hashes, negative results, scientific claims allowed and
  forbidden, unresolved risks, and requested Orchestra decisions.
- **`RUN_REQUEST.md` (mandatory before material compute):** for an O-009 smoke,
  record exact HEAD plus working-tree diff hash, bounded data scope, resources,
  command/output, stop criteria, and the standing-authorization citation. For full
  tests/scientific work, record the immutable commit, resolved config hash,
  data/split manifest, cells, seeds, GPU/count/time budget, command, output path,
  stop criteria, and exact owner-approval status. A changed approved scope
  invalidates full-test/scientific approval.
- **`RESULTS.md` (mandatory for execution sessions):** job IDs, exit status, raw
  artifact paths/checksums, metric and performance tables, missing/failed cells,
  and interpretation limits. Never replace or hide a negative/failed cell.
- **`REVIEW.md` (written by an independent review session):** findings first,
  exact code/data/metric references, attempted adversarial checks, gate verdict,
  residual risk, and whether integration/scientific use is approved.

Large checkpoints, logs, caches, and raw results remain under `/nobackup`; Git stores
manifests, hashes, commands, and compact reports. The Orchestra verifies the handoff
and review, and only then updates the canonical status ledger. Owner monitoring
during a worker session does not replace independent code/science review.

The independent reviewer must explicitly audit data leakage/split ownership,
coordinate/calibration/units, batch invariance, config resolution, effective
optimizer steps, precision, modality execution, metric/denominator semantics,
resume/provenance, failed cells, and any shortcut that could inflate performance or
ASR. “Tests pass” alone is not a scientific PASS.

### Worker handoff format

Every worker records this information in `HANDOFF.md` and also returns it in the
session chat:

```text
Session ID / status: PASS | CHANGES-REQUESTED | BLOCKED
Branch/worktree and commit (if authorized):
Files changed:
Commands/tests/jobs and exact outputs:
Gate checklist:
Artifacts/checkpoint/config hashes:
Scientific interpretation allowed:
Scientific interpretation NOT allowed:
Remaining risks / decisions for Orchestra:
```

## 8. Existing-file inventory and consolidation policy

The worktree was clean before this Orchestra consolidation. The immediately
preceding Arrhenius work is already committed in eight commits from `4a14e4a` to
`54a9ac3`; it touched 72 files because it included runtime implementation,
profiling harnesses, launchers, tests, configs, and long-form evidence.

The main written records are:

| Existing file | What remains valuable | New status |
|---|---|---|
| [`phase1_capability_summary.md`](../collab/model_capability/phase1_capability_summary.md) | historical capability arc and `bb02d` metrics | read-only historical evidence; architecture-gap conclusion superseded |
| [`arrhenius_bevfusion_gap_audit.md`](../collab/model_capability/arrhenius_bevfusion_gap_audit.md) | sparse precision, memory/OOM, Stop-E evidence | read-only engineering evidence |
| [`arrhenius_camera_branch_audit.md`](../collab/model_capability/arrhenius_camera_branch_audit.md) | camera mini topology/gradient/profile evidence | read-only engineering evidence |
| [`arrhenius_speedup_log.md`](../docs/arrhenius_speedup_log.md) | profiling and precision history | performance evidence |
| [`arrhenius_migration.md`](../collab/arrhenius_migration.md) and [`env.md`](../docs/env.md) | validated GH200 environment and jobs | migration is read-only history; env is active runtime documentation |
| `scripts/arrhenius_*`, `run_arrhenius_*` | executable diagnostics and Slurm harnesses | retained implementation; reuse selectively |
| `tests/test_arrhenius_*`, `test_sparse_voxel_encoder.py` | current engineering regression tests | retained tests; extend, do not summarize away |

No historical evidence or executable file is deleted. The consolidation is logical:

- **this file** is the canonical objective, decision, schedule, and gate record;
- **`SESSIONS.md`** is the canonical work breakdown and per-session contract;
- **`KICKOFFS.md`** is the canonical copy-ready worker/reviewer startup registry;
- all of `fl_v3/collab/` is linked evidence only and should not receive new
  planning, handoff, review, result, or status prose.

## 9. Current data and environment facts

- Arrhenius GH200 and the persistent fp32/fp16-spconv environment are validated.
- The user is a member of the nuScenes license group, and the shared dataset module
  is now available. The module exposes `NUSCENES_DATA_DIR`; the shared directory
  contains trainval metadata plus ten stored `trainvalXX_blobs.zip` archives and
  the test archive.
- The ten trainval archives were readable in the access check and contain the
  camera samples, `LIDAR_TOP` keyframes, and sweeps required by the current model.
- A fresh login may be needed for normal group membership; `sg
  arrhpc-dataset-nuscenes` was the temporary verification path.
- Reviewed S01 worker `abe5c58` implements the read-only ZIP backend. Job `332651`
  indexed all ten archives and resolved `538,695/538,695` declared train/val
  references with zero missing; job `333206` ran 56 focused GH200 tests with zero
  failures/errors/skips against remediation implementation `54a48f9`. The full
  `t1.v1` caches from job `332651` are historical evidence only and are forbidden
  production inputs. S07-A owns current-format caller migration and full trainval
  `t1.v2` cache materialization/freeze. The read-only
  `collab/arrhenius_migration.md` remains historical evidence.

## 10. Owner decision gates

Owner decisions are stage-gated, not one blanket prerequisite for Wave A. An
unresolved item blocks only the session/action whose latest freeze point has
arrived. S00 should bring each item back to the owner with the relevant handoff,
review, cost, or engineering evidence rather than requiring speculative decisions
now.

### Locked now

- [x] USENIX '27 sprint scope and overlapping calendar (owner-approved).
- [x] Protocol B is the primary security setting; Protocol A is the clean
      optimization/control setting (owner-approved 2026-07-10).
- [x] Per-session `RUN_REQUEST.md` audit process; bounded non-scientific smoke may
      self-submit under O-009, while full tests/runs/metrics/matrices require exact
      owner approval (owner-approved 2026-07-10).
- [x] Fifteen-session split, dependencies, file ownership, and independent review
      process in `SESSIONS.md` (owner-approved).
- [x] Copy-ready S00, S01-S15, and S01-R-S15-R kickoff registry in `KICKOFFS.md`.
- [x] Evidence-driven Orchestra refinement authority with owner-locked scientific
      decisions and a recorded change-control ledger (owner-approved 2026-07-10).
- [x] Codex-UI-provisioned worktrees with pinned SHA/ref kickoff envelopes;
      sessions verify but do not manage worktree topology (owner-approved
      2026-07-10).

### Deferred decisions and latest freeze points

Engineering diagnostics may inform these decisions. However, metric definitions,
success thresholds, cells, and exclusion rules must be frozen before the scientific
runs they judge; they cannot be selected after seeing those outcomes.

| Decision | Current status | Latest owner freeze point |
|---|---|---|
| Gaussian-radius/target equation and golden values | owner-approved for S02 under O-017: exact official CenterPoint/BEVFusion reference semantics, `min_overlap=0.1`, `min_radius=2`; deterministic golden fixtures must pin the equation | S02-R independently recomputes the fixtures; any deviation or alternative geometric formula returns to S00/owner |
| Minimum modular backbone/interface contract | owner-approved for Wave-A under O-017: S03 Swin-T/stride-8/pure-camera LSS/0.5 m bins/aspect-preserving geometry; S04 SECOND `0.075x0.075x0.2 m`/~8x sparse-XY reduction/low-resolution densification/fp16+fp32 contract | final cross-module integration contract remains an S07-B review gate |
| Multi-task CenterHead primary; TransFusion contingency/generalization | owner-approved for S05 under O-017/O-018: reference-faithful multi-task CenterHead with declared no-starvation decode and GroupNorm adaptations; TransFusion remains closed contingency | official groups/thresholds/NMS remain pinned; O-018 removes only the second task-wide K and forbids claiming exact official-decode equivalence |
| `D_base`/`D_tail`, regional/fleet client unit, and fine-tuning scope | pending | S12 may design alternatives now; freeze and hash before split materialization or any Protocol-B training in S13 |
| mAP/NDS, fusion-gain, per-class, speed, memory, and acceptance gates | pending | numerical floors may use S07 engineering/profile evidence, but freeze before the S08/S09/S10 scientific run each gate evaluates |
| Five-cell single-seed matrix and whether 11 full runs are necessary | pending | use S07 cost plus reviewed branch evidence; approve each exact wave before its `RUN_REQUEST.md`, and freeze fusion cells before S10 outcomes |
| Security thesis and `CL-PILOT`/`CL-FREEZE` semantics | pending | S12 may refine the thesis; freeze `CL-PILOT` before it unlocks S13 and freeze final-backbone semantics before final security runs |

## 11. Orchestra change-control ledger

Only S00 edits this ledger. Each downstream plan/kickoff refinement records the
evidence that caused it; material entries remain `PENDING` until the owner approves.

| ID | Date | Evidence/decision | Change and affected sessions | Class | Approval/status |
|---|---|---|---|---|---|
| O-001 | 2026-07-10 | owner decision | Protocol B primary; Protocol A clean control; all sessions | locked scientific | approved |
| O-002 | 2026-07-10 | owner decision | 15 workers, independent reviews, durable handoffs, exact run authorization | locked process | approved |
| O-003 | 2026-07-10 | owner decision | S00 may refine unstarted work from accepted handoffs/reviews within the operational boundary | operational authority | approved |
| O-004 | 2026-07-10 | Codex managed-worktree contract and owner decision | task UI creates detached managed worktrees; kickoff pins SHA/ref; S00 permanent/pinned; local handoff commit needs exact permission | workspace policy | approved |
| O-005 | 2026-07-10 | owner decision on staged review | architecture, capability gates, matrix, split details, and thesis are decided at explicit latest freeze points rather than before all Wave-A work | scientific process | approved |
| O-006 | 2026-07-10 | owner approval in active S00 task | pin `/home/gaohui/.codex/worktrees/bb67/fl_weather_project` as the sole S00 canonical writer; archive the idle `/home/gaohui/.codex/worktrees/90d4/fl_weather_project` task without deleting its worktree or branch | workspace coordination | approved and executed |
| O-007 | 2026-07-10 | owner approval in active S00 task | issue S01 ZIP-backend implementation and S12 evidence/proposal-only kickoffs from `f262f6bea037580065a8505008773c04fdd259f5`; exact ownership is recorded in `SESSIONS.md`; both had `APPROVED_COMPUTE: none` at issuance | operational launch | approved and active; S01 later amended by O-009 |
| O-008 | 2026-07-10 | owner reasoning-budget decision | S00 must explicitly create future Sxx/Sxx-R tasks at `xhigh`; use `ultra` only for a recorded unusually complex implementation/review or broad difficult research task | resource policy | approved; applies to future tasks; S01 follow-up amended to xhigh |
| O-009 | 2026-07-10 | owner compute-policy decision and direct S01 smoke approval | allow bounded non-scientific Slurm smoke without per-job waiting, subject to the standing limits and preflight record above; retain exact owner review for full tests, full-data/profile/metrics, matrices, seeds, and reruns | compute policy | standing; S07-A focused jobs `333477` and `335280` completed PASS within scope |
| O-010 | 2026-07-10 | owner clarification request | replace ambiguous worker-kickoff `WORKER_SHA: n/a` with `pending`; record that worker SHA/ref is produced only after S00 completeness checking and owner-authorized delivery commit, and is then consumed by Sxx-R | kickoff schema clarification | approved operational clarification |
| O-011 | 2026-07-11 | owner workflow clarification | before S00 directly creates any Sxx/Sxx-R, present upstream handoff/review/diff status and the complete filled kickoff for owner review and explicit launch authorization; worker sessions never launch their own reviewer; S07 remains the sole code-integration session | launch/integration workflow | approved |
| O-012 | 2026-07-11 | S01 worker `abe5c58`, review `7cf7fcc`, jobs `332651`/`333206`, and S00 raw-artifact audit | accept S01 as a reviewed dependency for S07 only: full manifest/checksums and scheduler records match; 56/56 focused tests and all listed remediation-source hashes match. Do not merge the review branch as implementation; do not use historical `t1.v1` caches or claim model/scientific readiness | integration evidence | accepted dependency; no merge/push |
| O-013 | 2026-07-11 | accepted S01/S01-R evidence and `build_gt_database.py` audit | split S07 into phase S07-A data-foundation integration and later S07-B full-stack integration. S07-A lands exact worker history plus review artifact, fixes the `t1.v1` caller and active-doc status, hardens future test attestation, and prepares a separately approved full `t1.v2` cache gate; S06 must bind `n_sweeps` and cache/manifest hashes explicitly | operational dependency refinement | executed; S07-A final review PASS at `370ea6c` |
| O-014 | 2026-07-11 | owner temporary delegation in active S00 task | S00 may coordinate S07-A/S07-A-R through completion, approve reasonable validation-only O-009 jobs, and after S07-A completion prepare parallel S02-S05 launches; no large jobs/metrics, push, or merge to `v3-ad-perception` | scoped orchestration/compute authority | approved and exercised; Job `335280` completed PASS, no retry/follow-on |
| O-015 | 2026-07-11 | S07-A delivery `ba15716`, executable `44cefd0`, review `370ea6c`, Jobs `333477`/`335280`, and S00 raw-artifact audit | accept S07-A as reviewed data-foundation dependency; preserve old c8dd locale preflight rejection and historical `t1.v1` limits; full trainval `t1.v2` cache and S07-B remain separate gates | integration evidence | accepted; no full-cache/model authorization |
| O-016 | 2026-07-11 | accepted S07-A review plus S02-S05 ownership audit | future S02-S05 workers start from one S00-frozen integration SHA; `fusion/losses.py` is exclusive to S02 during the parallel wave, while S05 treats it as read-only and returns any shared-interface change to S00; all four read S07-A handoff/review and preserve the exact data contract | operational refinement affecting S02-S05 | approved under O-003; implementation-choice dependency satisfied by O-017 |
| O-017 | 2026-07-11 | owner approval in active S00 task after review of the S02-S05 scope and full-data boundary | freeze the S02 official-reference Gaussian semantics (`min_overlap=0.1`, `min_radius=2`), S03 Swin-T stride-8 pure-camera contract, S04 SECOND `0.075x0.075x0.2 m` sparse-XY contract, and S05 multi-task CenterHead primary/TransFusion contingency; authorize S00 to commit the canonical launch ledger directly atop S07-A freeze `0249eb21a32730ac1689255491b19a158711401f`, launch S02-S05 at `xhigh` from that resulting common ledger SHA, and authorize each worker to create `codex/s0x-*` plus implementation/test/handoff commits strictly inside its envelope | locked Wave-A implementation choices plus scoped orchestration | approved and launched at `xhigh` from common base `372de9398ae435f82b83367a922fd302c0635738` on `codex/s02-cl-p0-correctness`, `codex/s03-camera-architecture`, `codex/s04-lidar-second`, and `codex/s05-centerhead-decode`; S00 may audit/approve only O-009 short non-scientific requests and schedule S02-R through S05-R after completeness checks; no full trainval, 100/1000-step gate, profile/metrics/matrix, push, merge to `v3-ad-perception`, or approval drift |
| O-018 | 2026-07-11 | S05 pre-edit audit of MIT BEVFusion archived HEAD `326653dc06e0938edf1aae7d01efcd158ba83de5`, CenterPoint v0.2 `e9ef04c3715aa3342fa42f4f4e064db987def6ad`, and owner approval in active S00 task | resolve the official coder's per-class K=500 followed by task-wide K=500 conflict with O-017 no-starvation: retain per-class K=500, remove only the second task-wide K, feed at most 500/1000 candidates for one/two-class tasks into the pinned official task-wide NMS, and use deterministic ties ordered by score descending, class ID ascending, then flattened spatial index ascending; retain official score/range, task groups, circle/rotate choice/scales, pre=1000, post=83, and IoU threshold; use GroupNorm instead of official BN while retaining the shared-conv and per-task two-layer field topology; map task-local labels to the project's devkit-global `DETECTION_NAMES` IDs explicitly by class name rather than official task-flatten offsets | locked S05 active-session amendment | approved; implementation must be labeled `reference-faithful no-starvation adaptation`, not exact official decode parity; require single-class parity plus B=1/B>1, batch/input permutation, equal-score tie, tail-candidate retention, duplicate, coordinate, explicit `construction_vehicle`/`bus`/`barrier`/`pedestrian`/`traffic_cone` label-map, and submission-conversion fixtures; all O-017 compute/Git/integration limits remain unchanged |
| O-019 | 2026-07-11 | S02 delivery `7ad396e`; Jobs `335565`/`335578`; S03 Job `335630`; S04 Jobs `335566`/`335579`; S05 review `c818262`; S00 raw-log/diff/hash audit | preserve every Wave-A negative result and apply scoped return-for-changes: launch S02-R only from exact delivery; require S03 shared `/nobackup` immutable execution provenance plus an explicit decision on observed four-GPU whole-node allocation before another job; retain the S04 final-BEV fp16 gate and repair implementation rather than tests; return S05's three review findings and require fresh re-review | operational evidence/refinement under O-017; affects S02-S05 and S07-B readiness | all scoped returns resolved by O-020 through O-026; all negatives preserved; no integration/merge/push/full-data/profile/metric/scientific authorization implied |
| O-020 | 2026-07-11 | S02 review `fb17da3`/REVIEW hash `75b6a5ed...`; S05 remediation delivery `705216d`, implementation `753944c`, HANDOFF hash `91506174...`; S00 completeness audits | accept S02-R's source audit but return the missing one-GPU forward/backward as an evidence-only remediation with implementation semantics frozen; accept S05 remediation as complete enough for a fresh exact-SHA re-review, not as PASS; keep all authored-but-unexecuted runtime cases explicitly NOT RUN | review scheduling/evidence refinement under O-017 | resolved: S02 by O-021 and S05 by O-024; their pre-execution NOT-RUN state and later exact execution evidence remain separately recorded |
| O-021 | 2026-07-11 | S02 Job `336713`, delivery `3aebf2d`, limited review `df142dc`; S03 Job `336708`, delivery `5089383`; repeated Codex reviewer-provisioning API timeouts | accept S02 as a reviewed S07-B dependency after exact one-GPU B=3 forward/backward evidence; send S03 delivery to independent review after exact 10-test one-GPU PASS; when the Codex task provisioning API times out without creating a worktree, S00 may use the owner's existing S02-S05 reviewer-launch delegation to provision an exact-SHA detached review worktree itself, while the reviewer remains forbidden to manage worktrees and retains review-only ownership | operational evidence/infrastructure fallback under O-017 | S02 accepted; S03 later accepted by O-022; fallback used only for S03-R/S05-R2 after three failed API attempts; no merge/push/scientific scope expansion |
| O-022 | 2026-07-11 | S03 review `2f62e57`/REVIEW hash `01dea6fd...`; S04 Job `336718`, delivery `80a8fbb`, raw JUnit/log/artifact audit | accept S03 as a reviewed module-level S07-B dependency with explicit production-shape/integration limits; preserve S04 Job `336718` as FAILED 9/10 despite passing final dtype and B=4 subgates; treat the fp16 train-to-eval tiny-occupancy spconv tuner failure as a potential lifecycle blocker requiring diagnosis, not as grounds to weaken/remove the eval fixture | integration evidence plus S04 return-for-diagnosis under O-017 | S03 accepted dependency; S04 CHANGES-REQUESTED/diagnosis active; no automatic retry or S04 PASS |
| O-023 | 2026-07-11 | S04 diagnostic Job `336728`, delivery `49f26de`, seven isolated lifecycle cells and installed spconv 2.3.8 source audit | confirm the blocker is universal to current sparse fp16 eval dispatch: training custom-fwd casts features/filters to half and succeeds; eval bypasses it and requests fp16-feature/fp32-filter/fp32-output, failing across small/large/fresh/reused/cache-order cells; fp32 eval succeeds. Do not implement a workaround until owner selects A) recommended spconv-only training dispatch under no-grad with encoder eval caps/GN preserved and version guard, B) fp32 sparse eval plus fp16 interface cast, C) separate fp16 eval weight copy, or D) dependency patch | locked precision/runtime decision affecting S04/S06/S07-B | owner selected A under O-025; historical diagnostic and all failed cells remain preserved |
| O-024 | 2026-07-11 | owner temporary S02-S05 coordination/validation delegation; S05 worker `a9c801f`, execution `96e509b`, Jobs `336731`/`336738`, final review `1c44084`, and S00 raw-artifact/diff/hash audit | use the delegated authority for one necessary focused S05 rerun after correcting only the devkit tuple-valued test expectation; explicitly classify it as owner-delegated validation rather than O-009, preserve Job `336731` as FAILED 43/44, and accept separately approved Job `336738` as 44/44 with exact immutable identity/checksums. Accept S05 as a reviewed S07-B dependency while retaining production wiring/profile/full-data/scientific gates | scoped compute/review evidence affecting S05/S07-B | approved and consumed; S05 reviewed PASS at worker `a9c801f` / review `1c44084`; no follow-on compute, merge, push, profile, metric, or scientific authorization; S04's former pending choice is separately resolved by O-025 |
| O-025 | 2026-07-12 | owner approval after S04 Job `336728` mechanism diagnosis and S00 presentation of options A-D | select option A for S04: during fp16 inference under `torch.no_grad()`, keep the encoder and GroupNorm/eval semantics unchanged while only spconv sparse convolution modules use the installed spconv 2.3.8 training dispatch that activates its coherent fp16 custom-fwd casts. Require an exact version guard/fail-closed behavior; fresh-small, large, and same-model before/after train-to-eval coverage; no-grad/no-parameter-gradient proof; FP32 master parameter and state-dict hash immutability; normalization/eval-cap preservation; train-dispatch-under-no-grad parity; fp32 control; and bounded B4 eval memory evidence | locked S04 precision/runtime amendment affecting S04/S06/S07-B | implemented and independently accepted under O-026; the required synthetic lifecycle test may perform one forward/backward validation but no optimizer/GradScaler/parameter update or iterative training; no full trainval, profile/metrics/matrix, merge, or push; any broader dependency patch or precision change returns to owner |
| O-026 | 2026-07-12 | S04 final worker `483e149`, executable `8498597`, Job `341695`, review `a0763c2`, and S00 raw-artifact/diff/hash audit | accept O-025 option A as the reviewed S04 bounded synthetic module gate: exact spconv 2.3.8 fail-closed dispatch, no-grad/state/master-weight/mode restoration, train-dispatch parity, fp32 control, caps/isolation, and B=4 train/eval memory all passed 15/15. Preserve Jobs `335566`/`335579`/`336718` as FAILED and Job `336728` as diagnostic completeness. Carry the review's P3 forward: same-instance concurrent/reentrant eval is unproved, so S06/S07-B must serialize the instance or add instance-level protection and adversarial tests | reviewed integration dependency affecting S04/S06/S07-B | S04 accepted as reviewed S07-B dependency only; all S02-S05 Wave-A modules now reviewed PASS within their stated limits; no production/full-data/profile/metric/scientific PASS, merge, or push |
| O-027 | 2026-07-12 | owner instruction and subsequent approval of the filled S06 kickoff after reviewed S02-S05 completion | launch the filled S06 production-runtime envelope from exact canonical S07-A/S00 base `968d81583c87ba76b7dbbb722760f8eb8eb6cd39`; bind explicit camera-only/lidar-only/fusion modes, resolved config/data identities, executed-step accounting, nonfinite/overflow synchronization, persistent loader epochs, complete boundary-safe resume, eval autocast/provenance, legacy-checkpoint rejection, and S04 spconv 2.3.8 same-instance serialization. S06 consumes reviewed S02-S05 interfaces but S07-B remains sole implementation-integration owner | scoped S06 implementation plus orchestration/review delegation | APPROVED AND LAUNCHED at `xhigh` in task `019f5434-14f1-7391-b9d6-7ff2335e1dff`, worktree `44f9`, branch `codex/s06-production-runtime`; S00 may monitor, audit/approve only bounded non-scientific validation requests, and launch S06-R after completeness; compute remains none at kickoff; no full cache/trainval, 100/1000-step, profile/metrics/matrix, merge, or push |
| O-028 | 2026-07-12 | S06 delivery `8acfbfe`, executable `7d733e9`, pending request `d2e302ab...`, monitor report, and S00 actual source/test audit | reject both unexecuted S06 runtime requests (`a95816b` and `7d733e9`) before compute. Return S06 for two scoped correctness remediations: preserve fixed accumulation-window boundaries with no successful silent tail discard and complete attempted/invalid/successful sample exposure accounting; and make checkpoint load fail-atomic so any schema/state/RNG or late component-load failure leaves model, optimizer, scheduler, scaler, EMA and RNG unchanged. Preserve production raw-decode and DDP as explicit S07-B seams | operational correctness return affecting S06/S07-B | CHANGES-REQUESTED; old requests are REJECTED/NEVER EXECUTE, no job was submitted; worker may commit scoped fixes/tests/new request under O-027 but must wait for a new exact S00 audit/approval; no compute/merge/push |
| O-029 | 2026-07-12 | S06 remediation delivery `5bbb12c`, executable `6696984`, request `e42fd060...`, approval record `1528334`, Job `341997`, raw stdout/err/JUnit/source attestations, and S00/monitor audits | preserve Job `341997` as FAILED 1:0 (45 passed / 17 failed / 0 skipped) and forbid replay of its tuple. Return S06 within existing ownership for four scoped root causes without weakening contracts: thaw nested immutable config structures for run-config JSON; retain known-length accumulation preflight but use a truly lengthless iterator for the runtime-tail test; use a legal same-directory `.pt` temporary name while preserving atomic checkpoint save/cleanup; keep the fixed six-camera nuScenes contract and correct the synthetic eval fixture. The next launcher must finalize/check all raw-artifact hashes even when pytest fails, then return the original pytest status. Preserve the accidental empty `sbatch` CLI rejection as a no-job/no-resource negative event | operational runtime remediation and evidence preservation under O-027 | CHANGES-REQUESTED; no S06-R; Job `341997` and all earlier tuples are NEVER EXECUTE; worker may prepare remediation-2 commits and a fresh exact request, but no new compute until S00 audits/approves it; no merge/push/full data/metrics/DDP |
| O-030 | 2026-07-12 | owner-delegated S06 scheduling authority; remediation-2 request delivery `cae0ff5`, executable `c330c72`, approval record `57b745a`, Job `342014`, final worker `6b7ef29`, raw sacct/JUnit/identity/source/final-manifest audits, and S00 completeness check | approve exactly one remediation-2 bounded synthetic job after independently matching the 25-file aggregate, launcher, fresh roots and unchanged resource/test scope. Accept Job `342014` as a bounded engineering gate only: COMPLETED 0:0, 66/66, zero skips, CUDA rollback executed, spconv 2.3.8, and final `sha256sum -c` all pass. Preserve Job `341997` and the bare-sbatch no-op as negative evidence. Launch independent S06-R from exact final worker SHA `6b7ef29`; reviewer owns only `handoffs/S06/REVIEW.md` and no compute | bounded validation evidence plus independent-review scheduling under O-027 | S06-R ACTIVE in task `019f546e-5661-7860-8023-61f718803e99`, worktree `90a6`, branch `codex/s06-r-production-runtime-review`; no S06 acceptance/integration until verdict and S00 audit; all S06 job tuples retired, no merge/push/full data/metrics/DDP |
| O-031 | 2026-07-12 | final S06 worker `6b7ef29`, executable `c330c72`, Jobs `341997`/`342014`, independent review `ca7bbd7`, REVIEW SHA-256 `96d19965...`, and S00 actual diff/raw-artifact/review audit | accept S06 as a reviewed candidate dependency for S07-B only. Review found no P0-P2 and validates the fail-closed C/L/F runtime/config/checkpoint/eval contract plus exact Job `342014` bounded synthetic evidence. Preserve three P3 classes in S07-B: add real model/optimizer late-load rollback injection before production checkpoint claims; re-attest actual cumm/spconv build/source rather than only package version/config strings; and run the explicitly deferred actual S04/S05 fp16, mode-aware ZIP/multi-worker/resume, concurrency/EMA and production host-memory gates. Preserve Job `341997` 45/62 and the bare-sbatch no-op as negative evidence | reviewed dependency acceptance and unstarted S07-B kickoff refinement | S06 REVIEWED PASS within bounded scope; S07-B may now be prepared for owner review but is not auto-launched; no implementation merge, full cache, 100/1000-step, profile/metric, merge to `v3-ad-perception`, or push authorized |
| O-032 | 2026-07-12 | owner agreement to prepare the next step; reviewed S02-S06 worker/review topology, actual branch parents, changed-path overlap audit, and O-031 S06 acceptance | propose the filled S07-B integration kickoff for owner review. Use a dedicated branch from the post-draft canonical SHA; non-FF merge exact final worker branches S02→S06; do not merge divergent reviewer ancestry, instead import and hash the five exact final REVIEW blobs; require explicit semantic reconciliation of S05+S06 evaluation, S01 mode-aware I/O, S02-S05 module APIs, S06 resolved runtime/checkpoint and only proven-dead legacy cleanup. Stage compute authorization: no compute at kickoff, O-009 only after S00 request audit, and full cache/100/1000/profile/metrics/DDP remain exact owner decisions | proposed S07-B scheduling/integration contract; no locked science change | PROPOSED / OWNER LAUNCH APPROVAL PENDING; no branch/worktree/merge/worker/compute created by this proposal |
| O-033 | 2026-07-12 | owner approval of the exact O-032 envelope; S00 launch record; independent startup verification of task/worktree/ref/status | launch S07-B at `xhigh` from exact clean `detached@c9c84f8b2caebea14adc1d79d6d706695be0f50f`; authorize only `codex/s07-b-integrated-cl-stack`, the listed S02→S06 non-FF worker merges, exact five-blob review provenance import, and scoped implementation/test/handoff commits. Delegate S07-B scheduling to S00, including audit/approval of a necessary new O-009 request and launch of exact-SHA independent review after completeness | owner-approved operational integration scope; no locked science change | ACTIVE in task `019f549a-d56e-7843-8916-7ba08a6af276`, worktree `d5e7`; startup verified detached and clean before branch/merge. No compute at kickoff; full cache, 100/1000-step, profile/metrics, DDP, merge to `v3-ad-perception`, push and upload remain forbidden |
| O-034 | 2026-07-12 | S07-B worker `df13025`, code candidate `e3cedfa`, five exact merge SHAs, five imported review blob/SHA-256 pairs, direct-edit ownership audit, clean status and S00 static-check replay | accept the worker package as complete enough for independent review only. Do not infer runtime PASS: all integrated pytest/GH200/data/model gates remain unexecuted, full `t1.v2` is absent, and all five `s07_b_*.json` files deliberately fail closed. Launch S07-B-R from exact worker SHA with ownership restricted to appending `handoffs/S07/REVIEW.md`; reviewer must inspect real diffs/topology and decide findings plus any necessary bounded request | completeness/review scheduling under O-033; no science or compute change | S07-B-R ACTIVE at `xhigh` in task `019f54b9-f6a3-7600-b41e-d0b72613c028`, worktree `44c9`, clean `detached@df13025` before its sole review branch. No compute request approved or submitted; merge/push/full cache/100/1000/profile/metrics/DDP remain forbidden |
