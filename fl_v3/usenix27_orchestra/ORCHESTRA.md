# USENIX Security '27 Orchestra — strong CL backbone to federated multimodal security

> **Status (2026-07-13): active cleanup and clean-foundation recovery.**
> The owner froze legacy S07-B endpoint
> `e231808e77388d69053dcbced6e754dbe3468aef` as read-only negative evidence
> and selected `4ce2366df2925161adae8fea393d5fca64836d40` as the audited code
> baseline for S07-C cleanup. The old S07-B branch must not be rewritten,
> merged, pushed, or treated as a continuation baseline.
>
> S01/S07-A and reviewed S02-S06 remain accepted clean foundations within their
> recorded limits. The next approved topology is a canonical-only preparation
> commit on `codex/s07-c-legacy-security-cleanup`, then S07-C cleanup, an
> independent review whose history is never merged, and a later
> `codex/s07-b-clean-completion` branch from the accepted cleanup worker SHA.
>
> Legacy T4 attack-readiness, T5 attack, T6/T7 defense, old defense-registry and
> harness implementations are retired from active authority. No new attack is
> allowed until the CL detector is trained/frozen and clean Protocol-B
> adaptation is established. New attack work belongs to a later owner-approved
> S13 threat model; S14 starts only after a viable undefended attack.
>
> Compute is **not approved** for canonical preparation or S07-C kickoff. Any
> bounded GH200 validation requires an exact immutable RUN_REQUEST and separate
> owner/S00 audit. Full cache, trainval, 100/1000-step, metrics, profile, DDP,
> matrix, retry, merge, push and upload remain unauthorized.
>
> S12 is deferred and its proposal is not an active implementation authority.
> `fl_v3/collab/**` and `fl_v3/docs/cycle_04/**` remain read-only historical
> evidence. Sections below retain scientific planning context; where any older
> wording conflicts with the active registry in Section 11 and O-092, O-092 wins.
>
> **Venue target:** USENIX Security '27 first submission cycle.
> **Canonical companions:** [`SESSIONS.md`](SESSIONS.md) and
> [`KICKOFFS.md`](KICKOFFS.md).
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

The USENIX '27 sprint schedule is now sequenced through three streams:

1. **CL foundation:** fix correctness, choose and freeze a strong modular backbone.
2. **Clean federated foundation:** freeze the CL detector, establish clean
   Protocol-B adaptation and the separate Protocol-A control, and only then
   return to an owner-approved S13 threat model. Legacy T5/T6/T7 code is not a
   shortcut into this stream.
3. **Paper/artifact:** maintain the argument, tables, provenance, and artifact from
   the beginning rather than reconstructing them at the deadline.

## 2. Historical pre-Wave-A baseline: retained diagnostic context

This section records the problems that motivated S02-S06. It is not a claim that
the selected S07-C base still contains every listed defect. Accepted current
foundation identities and limits are in Section 11.3; O-092 governs cleanup.

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
   hashes, and later clean S07-B completion must bind those identities at every
   production entry point.

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

## 3. Owner-approved Wave-A module contract — preserve through cleanup

The owner approved the following implementation contract for parallel S02-S05 work
on 2026-07-11 under O-017. This freezes the primary module choices that those
workers implemented; it does not approve a full-trainval cache or model run,
numerical CL gates, or a final scientific architecture. S07-C must preserve these
contracts, and later clean S07-B completion plus independent review must reconcile
them before any production or scientific execution.

The approved Wave-A primary model is a strong, modular late-BEV detector whose modality
boundaries remain meaningful for clean ablation and any later re-specified
security work:

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
non-IID drift**, and **structure-aware defense**. A later owner-approved S12
re-audit must repeat the systematic literature review before any “first” wording
or threat-model claim is approved; the old proposal is not authority.

## 6. Execution order after the 2026-07-13 cleanup decision

The earlier overlapping date calendar is retired. It assumed that legacy
attack/defense work could proceed beside CL engineering and is incompatible with
O-092. The active order is evidence-gated rather than date-promised:

| Order | Required outcome |
|---|---|
| 1 | commit canonical-only P above audited 4ce2366 |
| 2 | S07-C removes legacy attack/defense routes and preserves clean foundations |
| 3 | independent S07-C-R accepts an exact durable cleanup worker SHA |
| 4 | clean S07-B completion closes only C/L/F/runtime/eval/FedAvg engineering |
| 5 | independently review clean completion, then rebaseline S08-S11 CL work |
| 6 | train and freeze the CL detector |
| 7 | re-audit S12 and establish clean Protocol-B adaptation plus Protocol-A control |
| 8 | owner may approve a new S13 threat model/attack; S14 waits for a viable undefended attack |
| 9 | schedule paper/artifact claims only from checksummed accepted evidence |

No deadline creates compute, merge, push, attack or defense authorization.
`CL-PILOT` and `CL-FREEZE` definitions will be re-frozen after clean S07-B
completion; neither currently unlocks attack work.
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
  resources, output path, and stop criteria, and cites `O-009`. Under O-092 cleanup
  work it must stop for owner/S00 audit before submission. Every job ID, log, exit status, retry, and
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
- [x] Per-session `RUN_REQUEST.md` audit process; bounded non-scientific smoke
      still requires an exact tuple and current owner/S00 audit, while full
      tests/runs/metrics/matrices require separate exact owner approval.
- [x] Fifteen-session split, dependencies, file ownership, and independent review
      process in `SESSIONS.md` (owner-approved).
- [x] O-092 cleanup ordering and anti-recovery rule. Only S07-C is the next
      preparable worker; S12-S14 prompts are explicitly deferred/blocked.
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
| Minimum modular backbone/interface contract | owner-approved for Wave-A under O-017: S03 Swin-T/stride-8/pure-camera LSS/0.5 m bins/aspect-preserving geometry; S04 SECOND `0.075x0.075x0.2 m`/~8x sparse-XY reduction/low-resolution densification/fp16+fp32 contract | preserve through S07-C; recheck at clean S07-B completion review |
| Multi-task CenterHead primary; TransFusion contingency/generalization | owner-approved for S05 under O-017/O-018: reference-faithful multi-task CenterHead with declared no-starvation decode and GroupNorm adaptations; TransFusion remains closed contingency | official groups/thresholds/NMS remain pinned; O-018 removes only the second task-wide K and forbids claiming exact official-decode equivalence |
| `D_base`/`D_tail`, regional/fleet client unit, and fine-tuning scope | deferred | re-audit S12 only after clean CL readiness; freeze and hash before split materialization or Protocol-B training |
| mAP/NDS, fusion-gain, per-class, speed, memory, and acceptance gates | pending after cleanup | use independently accepted clean S07-B evidence; freeze before each S08/S09/S10 scientific run |
| Five-cell single-seed matrix and whether 11 full runs are necessary | pending after cleanup | re-evaluate from clean completion/cost evidence; each exact wave needs owner approval |
| Security thesis and `CL-PILOT`/`CL-FREEZE` semantics | deferred | no S13 attack unlock until CL freeze, clean Protocol-B adaptation and a new owner-approved threat model |

## 11. Canonical owner-decision registry

This section is the only active O-ledger. O identifiers are never renumbered or
reused. A row in the closed/history tables preserves provenance but grants no
current implementation, compute, scientific or scheduling authority.

### 11.1 Active decisions

| ID | Binding decision | Current authority |
|---|---|---|
| O-001 | Protocol B is the primary security setting; Protocol A is the separately labelled clean optimization/control setting. | locked scientific |
| O-002 | Material worker work requires isolated ownership, durable handoff, independent review and exact execution authorization. | locked process |
| O-003 | S00 may refine unstarted operational work from accepted evidence, but locked scientific scope returns to the owner. | active orchestration |
| O-004 | Managed worktrees start from an exact pinned SHA/ref; durable worker and reviewer baselines require explicit Git authority. | active workspace policy |
| O-005 | Architecture, capability, protocol, matrix and paper decisions freeze only at their declared evidence gates. | active scientific process |
| O-008 | Worker and reviewer sessions use xhigh reasoning unless the owner records a different budget. | active resource policy |
| O-009 | Only bounded non-scientific engineering smoke may be proposed under the standing limit; every exact tuple is still recorded and audited. Full cache/trainval, model campaigns, metrics, profiles, DDP, matrices, retries and scientific runs require separate owner approval. | active compute policy |
| O-010 | Kickoffs use WORKER_SHA=pending until a durable delivery exists; reviewers pin the exact accepted worker SHA. | active kickoff schema |
| O-011 | S00 presents upstream evidence and a filled kickoff before creating any worker/reviewer task; workers never launch their own reviewer. | active launch policy |
| O-017 | Reviewed S02-S05 camera/LiDAR/head/loss contracts remain the clean C/L/F construction foundation. | locked architecture |
| O-018 | CenterHead retains the reviewed reference-faithful no-starvation decode adaptation and explicit global class mapping. | locked head/decode |
| O-025 | spconv 2.3.8 fp16 no-grad evaluation uses the reviewed version-guarded spconv-only training-dispatch workaround while preserving encoder eval/GN/state semantics. | locked runtime |
| O-092 | Freeze the old S07-B/T5/harness chain; S07-C cleanup is independently accepted at static-review scope at canonical anchor `70bcd856f7ebb411eb2887e7ab71ef41ed13271f`; start clean S07-B completion only after owner approval from the exact docs-only packet seal containing the filled kickoff, never from e231, bf480ea or reviewer history. | active clean-completion authority |

### 11.2 O-092 exact cleanup decision (2026-07-13)

The owner approved all of the following as one binding decision:

1. Freeze `codex/s07-b-integrated-cl-stack` at
   `e231808e77388d69053dcbced6e754dbe3468aef`. Do not rewrite, merge,
   push, cherry-pick wholesale, or use it as an unquestioned continuation base.
2. Use `4ce2366df2925161adae8fea393d5fca64836d40` as the audited code
   base. A canonical-only preparation commit may sit above it, but its diff must
   be exactly ORCHESTRA.md, SESSIONS.md and KICKOFFS.md.
3. Create `codex/s07-c-legacy-security-cleanup`; remove legacy T4
   attack-readiness, failed/unreviewed T5 attack code, old defense algorithms,
   defense registry, their configs/scripts/tests/oracle fixtures and active routes.
4. Retain clean C/L/F construction, official clean DetectionEval, S01 ZIP/data
   contracts, S06 runtime/checkpoint semantics and one fixed clean FedAvg path.
   Clean FedAvg must not remain under the legacy defense namespace.
5. Do not import `bf480ea77ccf9ae8417c3ea58e933701dbc7222a` spawn-policy
   changes into S07-C. A later clean S07-B completion may independently specify
   and test a minimal spawn lifecycle if the clean runtime requires it.
6. S07-C starts with APPROVED_COMPUTE=none. GH200 validation is optional and
   requires a new immutable RUN_REQUEST; full cache/trainval, 100/1000-step,
   metrics, profile, DDP, matrix, retry and scientific claims remain forbidden.
7. Review S07-C from its exact durable worker SHA in a separate worktree/branch.
   The reviewer owns only `handoffs/S07-C/REVIEW.md`; reviewer history is never
   merged into the implementation or completion branch.
8. Only after independent acceptance may
   `codex/s07-b-clean-completion` start from the accepted cleanup SHA. It owns
   simplified clean C/L/F, S06 runtime/checkpoint, official clean evaluation and
   clean FedAvg engineering completion, not legacy attack/defense recovery.
9. S12 remains deferred. No new attack is allowed until CL is trained/frozen and
   clean Protocol-B adaptation exists. New attack work belongs to a later
   owner-approved S13 threat model; S14 begins only after a viable undefended attack.
10. `fl_v3/collab/**`, `fl_v3/docs/cycle_04/**`, O-032 through O-091,
    the frozen S07 branch/reviews and raw job artifacts are historical/negative
    evidence only. Future workers may cite them as evidence but may not import,
    copy, recover or treat their code or decisions as current authority.

#### O-092-A1 — S07-C remediation amendment (2026-07-13)

S00 audited the first uncommitted S07-C delivery at detached
`4eba37d60cbeb9c865e4eec8d5fa57c90d23f873` and found two blocking classes
before a durable worker SHA was created. The owner approved returning the same
worker for remediation with no compute:

1. `scikit-learn==1.8.0` is not FLAME-only in this environment. The builder
   installs `nuscenes-devkit==1.1.11` with `--no-deps`, while the devkit imports
   `sklearn.metrics` unconditionally. Preserve the pin/runtime identity in the
   project dependency manifests, Arrhenius audit lock, environment contract and
   import smoke; remove only FLAME/HDBSCAN-specific use and wording.
2. Remove the remaining active compatibility selectors/names
   (`NormTrackingFedAvg`, the `defense="none"` local-runner parameter) by
   migrating their clean callers. Reduce the visualization writer/tests to clean
   calibration/encoder/fusion/detection stages.
3. Delete the fail-closed but still active legacy
   `p3_grad_conflict.py`, `p3_crt_probe.py`, and
   `t3_trainval_reeval_fullval.py`; move `p3_partition_health.py` output away
   from read-only `fl_v3/collab/**`.
4. Remove active T5/T6/T7 authority wording from the package, clean model/data
   docstrings and focused tests without changing executable model/data semantics.
5. Re-run an inclusive active-surface tombstone scan and local static/focused
   checks, then correct HANDOFF/RESULTS and the exact path inventory. No worker
   commit/ref, S07-C-R, Slurm/GH200, merge or push is authorized by this
   amendment.

#### O-092-A2 — closed-session harness cleanup amendment (2026-07-13)

S00 inspected the completed A1 working diff and updated HANDOFF/RESULTS before
issuing this amendment. The A1 dependency correction, selector/alias removal,
four-stage VizWriter, three dead-script removals, output relocation and inclusive
tombstone evidence close the named A1 blockers. The owner then authorized the same
uncommitted detached S07-C worker to remove additional closed-session harness
routes that pollute automatic active-tree discovery.

Delete these 16 scripts:

```text
fl_v3/scripts/_bench_msweep.py
fl_v3/scripts/agg_overcommit_diag.py
fl_v3/scripts/arrhenius_lidar_gap_utils.py
fl_v3/scripts/arrhenius_mini_matrix.py
fl_v3/scripts/arrhenius_profile_mini.py
fl_v3/scripts/det_gate_a40.py
fl_v3/scripts/fl_gate_a40.py
fl_v3/scripts/p1_amp_smoke.py
fl_v3/scripts/p3_partition_health.py
fl_v3/scripts/run_arrhenius_mini_matrix.sh
fl_v3/scripts/run_arrhenius_profile_mini.sh
fl_v3/scripts/run_arrhenius_stop_e_gate.sh
fl_v3/scripts/run_v1_calibration.py
fl_v3/scripts/runconfig.py
fl_v3/scripts/t3_iid_vs_central.py
fl_v3/scripts/verify_levers.py
```

Delete their three dedicated tests and remove only the named residual seams from
shared clean files:

```text
REMOVE tests:
fl_v3/tests/test_arrhenius_camera_audit_controls.py
fl_v3/tests/test_arrhenius_lidar_gap_controls.py
fl_v3/tests/test_fl_gate_refuses_non_a40.py

REFACTOR-KEEP:
fl_v3/tests/test_s07_b_integration.py
fl_v3/tests/test_model_determinism.py
fl_v3/src/fl_v3/models/fusion/losses.py
fl_v3/usenix27_orchestra/handoffs/S07-C/{HANDOFF,RESULTS}.md
```

The protected active script set is exactly: Arrhenius environment/build/general
smoke (`arrhenius_env.sh`, `build_arrhenius_env.sh`,
`run_arrhenius_env_build.sh`, `arrhenius_smoke.py`,
`run_arrhenius_smoke.sh`); `centralized_train.py`; clean data builders
(`build_nuscenes_cache.py`, `build_gt_database.py`); the seven accepted S01 ZIP
scripts; `run_s06_runtime_tests.sh`; and the two accepted S07-A provenance/cache
scripts. Do not delete, repurpose or fold these into another harness.

A2 adds no scientific/runtime feature and no compute authority. Remove stale
imports/tests/wording tied to the deleted routes, update the exact inventory and
tombstone evidence, run available local compile/JSON/TOML/bash/diff checks, ACK
O-092-A2 in HANDOFF, and stop again for S00 audit. No commit/ref, reviewer launch,
Slurm/GH200, merge or push is authorized.

**S00 completeness and durable-delivery outcome.** S00 independently inspected
the cumulative A2 source diff and handoff package and re-ran the exact-script-set,
protected-byte, semantic-AST, Python compile, JSON, TOML, shell and diff checks.
No source-cleanup blocker remains. The exact linear identities are:

```text
audited code base:       4ce2366df2925161adae8fea393d5fca64836d40
canonical A1/A2 parent: f7c696345b24b0e1227b1a52f3b47fb14e9120f5
original snapshot:      9f06875e1b865734950abcf3b6de36ad06a0ac7b
worker implementation:  a16c2cdfd4e23ba08677a66c45c50dd78340cc3b
handoff seal:            f736f41371666725a11d51bc3b01c6ececb59d50
implementation patch-id: 8f89c30d21164e80ec73f6a01eab33621e984789
```

The snapshot and canonical-parent implementation have the same patch-id; the
snapshot remains provenance evidence only. The implementation is a direct child
of the canonical parent, and the handoff seal is a direct child of the
implementation. `codex/s07-c-legacy-security-cleanup` was fast-forwarded to the
handoff seal without a merge commit. The owner authorized independent S07-C-R
from the subsequent canonical review-launch seal, with the implementation and
handoff identities above pinned separately. This is not an implementation PASS or
runtime/scientific PASS; dependency-backed and GH200 checks remain explicitly
NOT RUN.

**Independent review and acceptance.** S07-C-R reviewed from launch base
`6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2` and produced review-only commit
`b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5` on
`codex/s07-c-r-legacy-security-cleanup-review`, with sole parent equal to that
launch base and tree `188a6e006e3f8c6f494258379a191f15648ae5ca`. Its only diff is the new
`handoffs/S07-C/REVIEW.md`, SHA-256
`588cfd0f91a2f70cbdcc6bf94a2279fc3cca693c9cd14f9d9909f02df769d8f5`.
The verdict is **PASS at code/source/config/test/docs static-review scope**, with
no P0-P3 finding. Dependency-backed pytest, Flower/Ray, S01/S06 lifecycle,
C/L/F runtime, official DetectionEval runtime and GH200 remain NOT RUN. Reviewer
history is separate evidence and was not merged. Canonical S07-C acceptance is
sealed at `70bcd856f7ebb411eb2887e7ab71ef41ed13271f`; this is the accepted clean-code
anchor, not itself launch authorization. The future S07-B-COMPLETE worker base is
the exact docs-only S00 packet-seal descendant containing the filled kickoff. Its
full SHA must be supplied explicitly in the task envelope, and the branch/worktree
may be created only after owner launch approval.

**Prepared clean-completion boundary.** The accepted tree still contains stale
active T3/Path-A/Path-B/4-GPU/overcommit and legacy `collab/**` authority profiles
in `fl_v3/configs/flwr_config.toml`. The filled S07-B-COMPLETE packet requires this
file to retain only CPU local smoke plus one single-GPU sequential clean profile,
with plain FedAvg and no server EMA as the validation default. Preserved FedOpt/EMA
code remains capability, not a completion default or scientific claim. Old P1/T3/
MCR configs and all scripts are read-only inputs; no harness may be recovered or
created. All other source edits require a demonstrated clean-contract failure and
remain inside the exact kickoff ownership. GH200 is not approved at kickoff; the
only candidate is the separately audited bounded sequential engineering job in
`KICKOFFS.md`. No branch, worktree, worker, compute, merge or push is authorized by
preparing or committing this packet.

### 11.3 Accepted clean-foundation evidence

Acceptance below is limited to the recorded worker/review/runtime scope. It is
not a full-data, production, performance or scientific PASS.

| Foundation | Historical decisions | Durable evidence | Accepted scope |
|---|---|---|---|
| S01 ZIP/data | O-012, O-015 | worker `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`; review `7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc` | read-only ZIP/data contract; historical t1.v1 caches forbidden |
| S07-A | O-013, O-015 | delivery `ba1571632557c20adbda3172221694cdbecfeabe`; executable `44cefd06bc815e893919d95c754896711dba3402`; review `370ea6c0bd4d9d737a5a50b6aff1c6f742589825` | reviewed data-foundation integration |
| S02 | O-017, O-021 | worker `3aebf2dc1d19473f29260df279421047d216d70e`; review `df142dc9a391b87d05bd7becaba59459e9659f88` | losses/targets module contract |
| S03 | O-017, O-022 | worker `50893839c45cd3e2ef1b72b98db6668df7030f2a`; review `2f62e570c9c24ef1e18a483888c3f28ad56a415e` | camera module contract with recorded shape limits |
| S04 | O-025, O-026 | worker `483e149b95ec891b675df825d924a96bb225b7dd`; executable `84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f`; review `a0763c2e0b322d4ca53a92f9f69c90d9b231bbff` | LiDAR/spconv module contract with recorded concurrency residual |
| S05 | O-018, O-024 | worker `a9c801fdee378906e54d06314d0c772b6559901a`; executable `96e509b71a3e22afb4de397132438fd3b9bbf5d8`; review `1c440843bb2b6d72f10310ff11fcde0d7d1e885c` | CenterHead/decode module contract |
| S06 | O-027 through O-031 | worker `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7`; executable `c330c72f4060348768c63fb1b7855ca56baffb95`; review `ca7bbd7e49e91ac2f214f39f62d5e416dd736383` | bounded C/L/F runtime/config/checkpoint candidate contract |
| S07-C | O-092, O-092-A1, O-092-A2 | implementation `a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`; handoff `f736f41371666725a11d51bc3b01c6ececb59d50`; review `b8e11bc98cfd904e9c7c259d3d6f7edc0c7922d5` | legacy active-route cleanup and protected clean-foundation preservation at static-review scope only; runtime NOT RUN |

### 11.4 Closed and consumed decision history

Every identifier listed here is closed, consumed, superseded or frozen. It does
not authorize current work.

| O-ID range | Historical purpose | Terminal disposition |
|---|---|---|
| O-006 through O-007 | initial S00 writer and S01/S12 launch coordination | closed; O-006 sole-writer topology superseded by O-092 |
| O-012 through O-016 | S01/S07-A acceptance and Wave-A preparation | consumed; accepted evidence moved to Section 11.3 |
| O-019 through O-024 | S02-S05 review/remediation scheduling and bounded evidence | consumed; accepted outcomes moved to Section 11.3 |
| O-026 through O-031 | S04/S06 acceptance, remediation and review | consumed; accepted outcomes moved to Section 11.3 |
| O-032 through O-038 | initial S07-B integration, first delivery, R1/R2 and caller migration | frozen core-integration history; 4ce2366 is the cleanup boundary |
| O-039 through O-051 | legacy T5 artifact/gate remediation and R3-R8 | retired negative evidence; never an active attack contract |
| O-052 through O-063 | initial S07-B runtime/diagnostic/attribution and spawn-policy remediation | consumed negative/attribution evidence |
| O-064 through O-071 | R9-R12 multiprocessing test-lifecycle chain | retired harness evidence |
| O-072 through O-083 | Jobs 351903/352105 and R13-R15 harness chain | retired diagnostic/harness evidence |
| O-084 through O-091 | Jobs 352354/352718, warning-fatal remediation and R16 | terminal negative evidence; no retry |
| O-092-A1 through O-092-A2 | S07-C dependency/alias remediation and closed-session harness removal | consumed by implementation `a16c2cd` and independent static review `b8e11bc`; retained in Section 11.2 as exact cleanup history |

### 11.5 Frozen S07-B Git and review anchors

These identities are retained for audit only:

| Artifact | SHA |
|---|---|
| first integrated delivery | `df13025bc6582b9b436d1df065de75c03e92782d` |
| audited cleanup code base | `4ce2366df2925161adae8fea393d5fca64836d40` |
| sustained T5 expansion start | `2c6203c02f118678dcfb71e3b67ddc703dbd2f8a` |
| T5 artifact-hardening end | `8a7b60b2dd27b1c7ba72e53ddbe67b278ea2f512` |
| last old-chain production-source change | `bf480ea77ccf9ae8417c3ea58e933701dbc7222a` |
| frozen old S07-B endpoint | `e231808e77388d69053dcbced6e754dbe3468aef` |

Reviewer commits, all unmerged review evidence:

`R1 bcffdece226e73207509ca86540443e7640fb6c5`;
`R2 afb81f51cdf311de215d351e92e2bf5ac6c3bd43`;
`R3 d6f8ae6233c4900e63151d4ee8fab98d549695b8`;
`R4 a1452e095ee88a0570580a612f31108aa4b9db30`;
`R5 2176e8d2e8185af26f27d67a45838528e4390543`;
`R6 ef01d1cad73021acb87b01874726b83da6470e84`;
`R7 e4fa439a5c09447bd8b413682772e81f9998f027`;
`R8 8a144ddaa624f3fd0605c7464eb30c1dcf6a51d9`;
`R9 55f19ab1c7ef1188cfa803724b79a79b3a0d0291`;
`R10 786e31dc81a88f8250f2b5176617f1375f2afcee`;
`R11 52e05ac0f500f1f671818125dc72caded9c1b4b8`;
`R12 49735beb1ee5e93bf2cf14dd937e24bdd4e5017e`;
`R13 69037534352c4517e93a62b17cd8f168c0f8a24c`;
`R14 9645148d3441a66a373091766c0186ea10243336`;
`R15 bc587790ff3b2dfb65b12fa4469c1f5b79aea5fc`;
`R16 d621d696d5a188189041fa73e54495eb56e8db49`.

### 11.6 Frozen raw runtime evidence

| Job | Authoritative terminal result | Preserved key SHA-256 |
|---|---|---|
| 348557 | FAILED; at least 3 failures, 4 errors and timeout; no JUnit/final manifest | pytest log `eceba3ae66efdb901626eac108200bc9f50108229a290dad39dec64bd8abad2c` |
| 348818 | diagnostic complete; suite fail; launcher/output-path noise plus real failures and fork hang | summary `892d335d528c8ea29c671a5152bbf919398882a622b6ade17e2d25b6334de9ff`; manifest `b794336a825b7a44eb8d22033bf4684fa43a93b7999f24a597b90d8d5999c835` |
| 349653 | stable_equal_current attribution only; no model/runtime readiness | summary `806afbfd41eabad3d2181c7c829a74f4ded34cef91636b5bdb7018b5fbbc36fc`; manifest `0c74aae4067bab74619269c16b38c8724ce38d56018d6dea035066e78528341c` |
| 351903 | focused gate FAILED; ZIP/model-task multiworker timeouts | summary `458d4a55b730cc375c15608d5b253752bd67454f6853e532a2d4ac66bad5a7e4`; manifest `d0d8ab44fde39f9b0149d3b1e21d375713b0fcb29da0018cba22e792a0582c3f` |
| 352105 | diagnostic complete; suite fail; AF_UNIX/subreaper harness confounds | summary `0ea391ad8f85e7567ca3473082dd1d15c3c32383ef591cb77c5d13348d104a9b`; manifest `00ada336cac0e26f2d60423d425c11439289f96154b1df4e4a6611ea7c59eb6d` |
| 352354 | formal 9/9 but hidden DataLoader worker SIGABRT warning; strict readiness FAIL | summary `b8fd26b34d607510c9a3a3e90251709dce43f792b8956728845448e6837478e9`; manifest `67d723b37ca3a9d36af8bde75eab13765ca05bef1bd1fc6e2f08bbf87d3527ac`; warning log `fb50d32d85c1f0cc24c27727d784c0ee7ceb045caf166294fd3869fd3bb62dbb` |
| 352718 | warning-fatal harness complete; suite FAIL with seven timeouts; no retry | summary `52fb107d7e5b5d9bf8655685d568574abcf95280caea19b522c36758952437d6`; manifest `fd7b9492fd05a5be418a183c42d9d3ea3a530d1c86a4920ae7dcd274e68a2a9e` |

Job 352354's manifest is additionally preserved by its authoritative full hash
in the frozen S07 RESULTS package; the canonical value is
`67d723b37ca3a9d36af8bde75eab13765ca05bef1bd1fc6e2f08bbf87d3527ac`.
No row above may be cited as full S07-B, production, model, full-data, FL,
attack/defense or scientific PASS.

### 11.7 Canonical anti-recovery rule

No current or future worker may import, copy, recover, cherry-pick or use as an
implementation/scientific contract any legacy T5/T6/T7 code, legacy defense
implementation, old cycle_04/collab decision, retired O-032-O-091 process
decision, or frozen e231 branch content. If a clean later task needs a similar
mechanism, it must be re-specified from the accepted clean foundation under a new
owner-approved envelope and independently reviewed.
