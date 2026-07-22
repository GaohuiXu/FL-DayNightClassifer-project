# S10 HANDOFF — LiDAR IP-LG0 closed / L-E1 awaiting activation

## 1. Current state and authority

```text
SESSION: persistent S10 Phase I-P throughput preflight
UNIQUE_BASE_SHA: f1a2babda8dafd181b5a5144ab025a3f6be21cc2
BRANCH: codex/s10-phase1p-throughput-preflight
FROZEN_CONTROL: codex/s10-phase1-branch-qualification at f1a2babda8dafd181b5a5144ab025a3f6be21cc2
ACTIVE_DECISION: owner approved the LiDAR preflight workflow and closed IP-LG0
SCIENCE_ORDER: complete LiDAR preflight, then refreeze/review revised Envelope B
PHASE_I_PLAN: PHASE_I_PLAN.md; P1-G0 PLAN_FREEZE closed
CURRENT_AUTHORITY: LiDAR L-WP1 source/docs/tests, local validation and linear commits
EXECUTION_STATE: L-WP1 materialized; IP-L-E1 and Envelope B are unauthorized
MERGE/PUSH/UPLOAD/PUBLICATION/S11+: not authorized
```

IP-G0 intentionally inserted Phase I-P before the long Camera/LiDAR qualification
runs and authorized this isolated linear branch plus WP0 implementation. The owner
then activated IP-E1 at `85c6719e4b880b198d850e16b1418c230fa5c656` for WP0's
GH200 runtime close and continuous IP-WP1 -> strict IP-WP2 execution. After the
three ordered Camera candidates reached terminal negative/no-promotion results, the
owner closed IP-WP2 and IP-E1 and opened IP-G1 discussion. The owner subsequently
closed IP-G1 and activated the bounded Camera-only IP-E2 authority recorded below.
Neither IP-E1 nor IP-E2 authorizes scientific training, an evaluation role, merge,
push, or movement of the frozen control branch.
The Section-7 Envelope-B request in `RUN_REQUEST.md` is preserved verbatim as a
historical control and is not activated by Phase I-P.
After terminal IP-E5 evidence, the owner explicitly accepted ordinary per-rank B16
BatchNorm and `seed + epoch*world_size + rank` worker RNG and promoted the exact
same-node two-GH200 recipe. Implementation
`2c3780bb6373ae784b41c22df072824f7a92d457` materializes that decision without
submitting a job or activating Section 7.
After the terminal B16 evidence, the owner closed IP-G2 by promoting B16x2 and
explicitly accepting its BatchNorm/worker-RNG recipe change relative to B8x4.
Implementation `299277e8bdb8f60a05e8f06c2c0706e29252b51c` materializes the exact
Camera stack; it does not activate or revise Section 7.
The owner froze the follow-up order: re-profile the final B16 stack and
screen the conservative batched affine/grid path first, then qualify same-node
2-GH200 DDP against the single-GPU B16 reference. Implementation
`9233af3119857511f5f2acc310a182449e7b91a2` prepares only the first, single-GPU
follow-up. The owner activated that exact Section-9.4 request at
`1abe26b3cde2f9f1c26fca130b999d054d6782b1`; it did not authorize DDP execution.
Initial Job `539364` passed the trace and conservative screen on one n89 allocation.
Preprocessing remained the largest named Camera-forward range at `72.3759%` of
the named-range CPU sum. Conservative batched affine/grid measured `27.868228`
versus `25.400589` presentations/s, ratio/lower bound `1.097149 / 1.087948`,
with every health/checkpoint gate PASS. This unlocked, but did not promote, the
combined batched-rotation candidate implemented at
`417dfefb8b37551bdd284fa30f0ef575b4a075e8`.

### 1.1 Frozen Phase I-P workflow

The workflow is continuous where the decision is already engineering-only; its
five WPs and three gates are not independent mini-projects:

| WP | Inputs and work | Required output / acceptance | Continuous authority and stop |
|---|---|---|---|
| IP-WP0 measurement path | frozen C/L configs and production model/data/loss/AdamW/scheduler/scaler/checkpoint paths | accumulation-aware B4x8 profiler; separate low-overhead sustained and short trace modes; exact identities; candidates default-off until an explicit owner promotion; local/focused checks | IP-G0 permits scoped implementation and linear commits; stop before scientific semantics or compute |
| IP-WP1 real baseline | exact D_fit, official CBGS/GTDB, seed 0, physical B4 x accumulation 8 | two-process sustained C/L baselines, whole-model trace, system/memory evidence and real checkpoint continuation | runs only inside approved IP-E1; stop on identity drift, nonfinite/discarded windows, unresolved instability or ceiling |
| IP-WP2 output-neutral work | WP1 bottleneck evidence plus the named plumbing shortlist | individual parity, accepted-update, checkpoint/resume and sustained-throughput evidence; safe items may be combined | may proceed continuously inside IP-E1 only for frozen strict-output-neutral candidates; ambiguous or changed failure semantics go to owner |
| IP-WP3 capacity/runtime screening | IP-G1 owner-frozen cells | B8 and conditionally B16 capacity evidence plus selected SDPA/compile/AdamW probes; Cell-1 SDPA is the explicit promoted exception | runs only inside separately approved IP-E2; remaining candidates require their frozen screens and owner-scoped interpretation |
| IP-WP4 synthesis | accepted WP1-WP3 evidence | final combination validation, GH200 payback, keep/reject/owner-gated table, revised Envelope-B projection | IP-G2 decides promotion and later Envelope-B refreeze; Phase I-P itself makes no capability claim |

Gate/envelope order is exact:

```text
IP-G0 (closed: plan/topology/local implementation)
  -> IP-WP0 (closed)
  -> IP-E1 (closed: IP-WP1 and IP-WP2 terminal)
  -> IP-G1 (closed: exact Camera cells/resources/decision boundaries)
  -> IP-E2 (terminal: Cells 1-5 and 7 complete; conditional Cell 6 skipped)
  -> IP-G2 (closed: B16x2 + SDPA + scoped compile + fused AdamW accepted)
  -> IP-E3 (closed: conservative affine/grid selected; combined path retained)
  -> IP-E4 (closed: vectorized geometry + bulk conversion promoted)
  -> IP-E5 (closed positive; exact two-GH200 BN/RNG recipe owner-promoted)
```

IP-G2 is not Envelope-B activation. Any accepted production-source or config
change requires a new exact Envelope-B source/config/resource projection and the
already-required independent recipe-freeze review.

IP-E1 retained maximum concurrency one and serial execution. Its ordinary WP1/WP2
cells have a `2.0` charged-GH200-hour aggregate ceiling; a separately accounted
`+1.0` hour reserve may be consumed only by diagnosed code-level defect repair,
for a `3.0` hour hard aggregate ceiling. Code bugs have no submission-count stop
and were repaired continuously without blind identical retry. Those ceilings and
remediation authority expired when IP-E1 closed.

Job `525192` closes WP0 runtime and supplies one valid LiDAR sustained measurement:
`40.4214` presentations/s over 256/256 accepted windows, only `1.5169 ms/window`
mean loader wait, and `6.8215%` peak reserved-memory fraction. Checkpoint boundary,
64 continuation microbatches, RNG and training state were exact, but both a same-
process replay and a fresh-process replay failed the frozen per-tensor FP16 allclose
after eight windows at globally small relative L2 error. This proves the symptom is
runtime-kernel nondeterminism, not checkpoint corruption or input drift. On
2026-07-21 the owner amended the continuation rule: boundary, input, RNG and
discrete state remain exact; model parameters, BN mean, BN var, Adam `exp_avg` and
Adam `exp_avg_sq` are gated separately, with fresh-process relative-L2 and
max-absolute error each no greater than
`max(frozen tolerance, 1.25 * same-process repeat-control)`. Per-element allclose
was diagnostic only. Implementation `73158b7` applied that rule without
changing model/data/update semantics or the frozen FP32/FP16 numeric tolerances.
Read-only reassessment of the immutable Job `525192` result passes all five groups
and every exact requirement (`d883c1ef...f7fa2`); the raw old-gate failure remains
unchanged. The later IP-E2 owner amendment described below supersedes only the
enforcement role of these numerical distances: they remain reported diagnostics.

Jobs `527225` and `527229` then completed the same LiDAR sustained reference at
`38.0943` and `36.9148` presentations/s. Both had 256/256 accepted windows, zero
nonfinite/discarded/overflow windows, the identical CBGS prefix, the same
`6.8215%` peak reserved-memory fraction, exact boundary/input/RNG/discrete state,
and PASS for all five calibrated continuation groups. Across the accepted r1-r3
measurements, rates are `40.4214 / 38.0943 / 36.9148`; min-max spread over the mean
is `9.113%`, and even same-node r2-r3 spread is `3.145%`. Loader mean stayed only
`1.517-2.106 ms/window`, while measurement-window GPU utilization/power declined
from about `56.81% / 289.84 W` to `53.27% / 285.55 W` and `52.31% / 280.12 W` at
fixed reported clocks. The data therefore point to runtime/system compute variation,
not loader, memory, acceptance or checkpoint failure, but do not identify its cause.
The frozen conditional third repeat has been consumed; unresolved >3% instability
blocks a LiDAR speed claim. The owner subsequently accepted that HPC hardware/power
variation makes more identical LiDAR measurement poor-value: no fourth sustained
repeat is permitted and the LiDAR trace is frozen. The three rates remain an
engineering interval with median `38.0943` presentations/s, not a stable promotion
baseline or scientific result. The owner disposition ends LiDAR WP1 measurement
collection without another sustained repeat or a LiDAR trace. WP2 is explicitly
paused for further owner discussion.

Camera sustained r1 Job `527239` produced a valid 256/256-window main measurement
at `16.5390` presentations/s with zero nonfinite/overflow/discarded windows, mean
loader wait `2.3420 ms/window`, and peak allocated/reserved
`16,038,963,200 / 18,723,373,056` bytes (`18.3553%` reserved). The checkpoint was
`525,165,739` bytes; save plus file/model hashing cost about `0.8053 s/epoch`, so
checkpoint cadence is not a material Camera throughput lever. Exact checkpoint
boundary, all 64 input hashes, RNG, training state and discrete state passed, but
fresh-process continuation exceeded the owner-amended same-process envelope in all
five groups: Adam moments failed both metrics; BN mean/var and model parameters each
failed one metric. This is a numerical acceptance failure, not a code defect; the
valid one-repeat projection of about `29.56` GH200-hours is preliminary and cannot
close a stable baseline. Camera sustained r2 remains frozen.

The owner then permitted exactly the predeclared Camera trace despite that stop,
solely for diagnosis. Job `527247` at source `2d2f717` completed 16 warm-up plus
3/3 active accepted B4x8 windows with zero nonfinite/overflow/discard, `18.3553%`
peak reserved memory and `2.744 ms/window` mean loader wait. Its `4.638`
presentations/s is profiler overhead and is not comparable with the sustained
`16.5390`; the trace is localization evidence only. Current accounting is base
`0.702222 / 2.0`, code-bug reserve `0.146389 / 1.0`, hard aggregate
`0.848611 / 3.0` charged GH200-hours.

The trace's CPU wall ranges are profiler-inflated, but their internal partition is
coherent enough to rank work. Of the captured stage sum, forward is `53.774%`,
backward `29.770%`, loss `12.490%`, optimizer `3.632%`, and H2D `0.334%`.
Within forward, preprocessing is `122.831 ms` per B4 microbatch (`48.893%`), ahead
of Swin `20.078%`, view-transform/pool `14.407%`, head `10.906%`, and all remaining
forward ranges together `5.716%`. This does not make those percentages sustained
wall-time shares, because tracing changes latency and asynchronous work may cross
range boundaries.

The source explains the leading hotspot. Each B4 microbatch processes 24 images
serially; loader-sampled augmentation parameters are first copied to GPU with the
batch, then synchronously copied back to CPU and cloned. Every image separately
runs resize/crop/flip, constructs float64 affine matrices, builds a full `256x704`
meshgrid, performs two small matrix inversions, and calls `grid_sample`. Across the
24 traced microbatches there are exactly 576 `grid_sampler_2d` calls, but their
self-CUDA time is only `3.613 ms` total; repeated geometry construction, resize,
copies and launch/host overhead are therefore the useful target, not replacement of
the interpolation kernel alone. `aten::copy_` is the trace's largest aggregate
self-CUDA operator (`512.233 ms`, 44,487 calls), although that total spans the whole
model and cannot be attributed only to batch transfer.

Loss is secondary but nontrivial. The six task partitions consume `17.196 ms` and
the six target/loss ranges `40.518 ms` of the `58.352 ms` mean loss range per
microbatch. The remaining `0.639 ms` is only an upper bound for stack/sum plus the
single `isfinite(loss).item()` check. The check does detect nonfinite loss before
backward, but it is not the source of most observed loss time; moving it to once per
B32 window would also change abnormal-path control flow and remains outside strict
WP2. The higher-value exact candidates are augmentation transfer/unused-return
cleanup, a fixed coordinate-grid cache, and batched affine/grid construction while
retaining per-image interpolation. Training-field whitelisting and consolidated
target D2H remain plausible smaller candidates. This ranking activates none of
them by itself, and the failed Camera checkpoint gate is unchanged.

The owner subsequently accepts that negative Camera reference continuation result
as the reference limitation and resumes strict WP2 in the exact diagnosed order.
This is not a tolerance waiver: candidate boundary/input/RNG/discrete state remains
exact, candidate continuation is still reported under the grouped rule, and no
candidate is promoted merely because the reference is negative. Execution order is
augmentation-parameter CPU residency plus unused-return cleanup first; fixed
coordinate-grid cache second; batched affine/grid construction third while retaining
per-image resize and `grid_sample`. Each item remains default-off, separately
identified and individually checked before any combination. Training-field
whitelisting and consolidated target D2H follow only after those three; full batched
preprocessing, finite-loss aggregation, physical batch, SDPA/compile, fused AdamW,
checkpoint cadence and all other measurement-only candidates remain outside the
current WP2 authority.

Camera augmentation-cleanup r1 Job `527276` then completed the full real-training
body at source `efe767a`: all four focused pre-model tests passed, 256/256 measured
windows were accepted, and there were zero nonfinite, overflow or discarded
windows. The candidate was nevertheless a clear performance negative at `14.1774`
presentations/s versus the `16.5390` one-repeat reference (`-14.279%`), projecting
about `34.45` rather than `29.53` GH200-hours for 20 consumed-CBGS epochs. Peak
allocated/reserved memory was effectively identical at `16,038,960,128 /
18,723,373,056` bytes, and loader wait remained only `2.478 ms/window`; unused
VRAM was not converted into throughput. Stable-load system samples also moved in
the wrong direction across different nodes (`60.67% / 310.51 W` reference on
`n183` versus `51.29% / 284.84 W` candidate on `n462`), so they support a host/
launch-stall mechanism but cannot by themselves separate candidate causality from
node variation.

The candidate's checkpoint boundary, input stream, RNG and all discrete state were
exact. The grouped continuation remained negative: model parameters and BN mean
passed, while BN var and both Adam moment groups failed. As already decided for the
reference, this is retained honestly and is not an automatic promotion blocker
waiver. Source inspection supplied one exact refinement hypothesis inside the same
candidate group: the profiler left `augmentation_params` in DataLoader-pinned CPU
memory, but preprocessing read 168 scalar tensor values per B4 microbatch. Job
`527284` therefore materialized the tiny `24x7` block as Python float values once.
Its pinned-input preprocess test remained elementwise exact, all 256 measured
windows were accepted, and every exact plus grouped continuation gate passed. It
recovered `2.939%` over r1 but reached only `14.5941` presentations/s, still
`11.760%` below the one-repeat reference and a projected `33.46` GH200-hours.
Accordingly the augmentation cleanup group is rejected on payback, not on numerical
correctness; no third implementation attempt or combination is justified. Current
accounting after Job `527284` is base `1.179444 / 2.0`, code-bug reserve
`0.146389 / 1.0`, hard aggregate `1.325833 / 3.0` charged GH200-hours.

Fixed-grid Job `527313` independently enabled only the non-persistent coordinate
cache. Its three pre-model tests passed, including two-call elementwise equality and
state-dict exclusion; 256/256 measured windows and every checkpoint continuation
gate also passed. Peak active increased by only `6,557,720` bytes while peak
reserved decreased by `50,331,648` bytes, confirming the expected tiny cache and
slightly lower allocator fragmentation. It nevertheless reached only `14.7377`
presentations/s (`-10.891%` versus reference; `33.14` projected GH200-hours) and
was only `0.984%` above the different-candidate run on the same `n444` node. This
does not support a repeat or promotion. Current accounting after Job `527313` is
base `1.395000 / 2.0`, code-bug reserve `0.146389 / 1.0`, hard aggregate
`1.541389 / 3.0` charged GH200-hours.

The third ordered item therefore proceeds only through an exact-gated conservative
form: per-image resize/crop/flip, affine construction and inverse, and the individual
`grid_sample` calls remain unchanged; only one output-coordinate basis per
microbatch and the source-coordinate matrix multiply are batched. Focused CPU and
CUDA elementwise equality must pass before the sustained model body. A CUDA rounding
difference is a candidate-boundary failure and defers the item; it is not authority
to relax tolerances or change interpolation math.

Batched-grid Job `527323` passed CPU and CUDA elementwise preprocess equality before
model construction, then completed 256/256 accepted windows and all checkpoint
continuation gates. It was the best candidate at `15.7309` presentations/s and
recovered `6.739-7.790%` relative to the two slower candidates, while stable-load
GPU utilization/power reached `56.98% / 305.09 W`. It still remained `4.886%`
below the `16.5390` reference and projects `31.04` rather than `29.53` presentation-
only GH200-hours. Peak allocated/reserved stayed `16,038,303,744 /
18,717,081,600` bytes, so the deliberately batched geometry did not create memory
pressure, but neither did it establish payback. It is retained default-off as the
best later engineering anchor, not promoted or repeated.

The ordered WP2 result is therefore compact: augmentation cleanup `REJECT`, static
grid `REJECT`, conservative batched grid `HOLD_FOR_LATER / NO_PROMOTION`. No
candidate earned combination or the two-repeat final protocol. The owner closes
IP-WP2 without running the lower-ranked field-whitelist or target-D2H plumbing;
their expected payback is below the already-negative ordered items. IP-E1 is also
closed. Current accounting after Job `527323` is base `1.617500 / 2.0`, code-bug
reserve `0.146389 / 1.0`, hard aggregate `1.763889 / 3.0` charged GH200-hours.
Unused budget expires with IP-E1 and is not IP-WP3, B8, SDPA/compile, fused-AdamW
or Envelope-B authority. IP-G1 is discussion-only until the owner freezes an exact
IP-E2 shortlist and resource envelope.

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
  but alter BatchNorm statistics. The owner deletes B12 because it cannot exactly
  realize effective B32. Batch candidates cannot be promoted without an explicit
  owner recipe decision.
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
  recorded `CAPACITY_OOM`; B8 OOM skips B16. B16 is considered only after the
  B8+SDPA+compile stack passes and retains an owner-frozen substantial-memory-margin
  gate.
- Parity remains hard and exact/hash-exact for checkpoint boundary, input, RNG,
  training/discrete state, state names/shapes/dtypes and comparison integrity;
  every numerical state must remain finite. Model parameters, BN mean, BN var,
  Adam `exp_avg` and Adam `exp_avg_sq` retain fresh-process relative-L2,
  max-absolute and per-element-allclose diagnostics against
  `max(frozen tolerance, 1.25 * same-process repeat-control)`, but exceeding those
  trajectory-distance envelopes alone is not a promotion stop. A direct
  forward/backward/accepted-update failure, nonfinite state, structural/context
  drift or accompanying material loss/gradient/update anomaly remains a stop.
- Same-physical-batch numerical candidates retain cross-cell input/RNG identity.
  For B8/B16 the owner accepts that DataLoader worker assignment and augmentation
  draws need not equal B4; boundary/input/RNG/discrete state must instead be exact
  within each batch candidate's repeats and fresh-process continuation.
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

### 1.4 Closed IP-G1 and active IP-E2

The owner closes IP-G1 and activates Camera-only IP-E2 from activation baseline
`3f55e635aef4f893d9fd66e7921f55ce4f7b36e8`. Each reference/candidate pair runs
as two fresh processes serially inside one Slurm allocation/on one GH200; the final
confirmation reverses process order. B12 is deleted. B8x4 and conditional B16x2
retain effective B32 and require exact boundary/input/RNG/discrete identity within
their own repeats and continuation, while cross-B4 worker assignment/augmentation
equality is intentionally not required.

The exact serial cells are B4 SDPA, B4 scoped compile, B4 SDPA+compile, B8
SDPA+compile, B8 fused-AdamW delta, conditional B16 best-stack, then reversed-order
final confirmation. Scoped compile covers only Camera backbone, Camera neck,
decoder backbone, decoder neck and head forward callables; preprocess, view
transform/pooling, target/loss and optimizer remain eager. B16 eligibility requires
a passing B8+SDPA+compile path, no monotonic memory growth, and
`R8 + 2*max(R8-R4,0) <= 0.70*V` using peak reserved bytes on the applicable stack;
its fresh capacity probe and sustained run retain the hard `0.85*V` gate.

IP-E2 has `4.0` ordinary plus `1.0` code-bug-only charged GH200-hours, hard
aggregate `5.0`, maximum concurrency one and at most 60 minutes per job. No numeric
submission cap applies inside O-149 remediation. The phase is active, but the first
submission remains fail-closed until its derived implementation SHA, profiles,
paired command and fresh output root are recorded in `RUN_REQUEST.md`.

Implementation `e6af054bfb16710355e22f6cea931368750aba89` freezes the eight
exact Camera profiles, B4/B8/B16 effective-B32 runtime identities, current-code
SDPA and fused-AdamW parity pretests, scoped forward compilation, B8/B16
capacity/OOM terminal handling, sustained memory/block evidence and one same-
allocation pair analyzer. The analyzer requires one Slurm job/node, the same CBGS
prefix, and exact first-window input identity for same-batch pairs; it computes a
deterministic 50,000-draw one-sided 95% throughput lower bound, 20-epoch payback
and the frozen R4/R8 B16 margin. Numerical and batch candidates remain
measurement-only. Local static validation passed; torch/pytest are unavailable in
the x86 login Python, so every GH200 allocation runs fail-closed current-code
numerical pretests before either paired process.

Owner decision on 2026-07-21 amends IP-E2 after Cell 1: grouped continuation
relative-L2/max-absolute/allclose results are trajectory-reproducibility
diagnostics, not hard promotion gates. Exact boundary/input/RNG/training/discrete
state, exact names/shapes/dtypes, comparison integrity and finite state remain
hard. Implementation `89181c117d69aaa7094def38f6931623f385a691` applies that
single enforcement change and adds current-schema immutable-result reassessment;
it changes no model math, data, precision, loss, optimizer update or scheduler
semantics. The owner promotes SDPA as the IP-E2 Camera runtime building block and
orders serial continuation through Cell 2 compile, Cell 3 SDPA+compile, B8,
fused AdamW, conditional B16 and the final reversed-order confirmation. This does
not modify or activate the frozen Envelope B.

Cell 1 Job `531766` completed `0:0` in `00:25:41` on one `n203` allocation and
consumed `0.428056` base GH200-hours; the code-bug reserve remains untouched. All
eight current-code pretests passed. Eager and SDPA each completed 16+256 accepted
windows with zero invalid/discard/scaler skip, identical same-batch input anchors
and CBGS prefix, stable reserved memory, and exact boundary/RNG/training/discrete
state. SDPA measured `16.183531` versus `15.143028` presentations/s: ratio
`1.068712`, one-sided 95% lower bound `1.065611`. It reduced peak allocated/
reserved from `16.039/18.723` to `14.885/17.459` GB and lowers the current
20-epoch projection by `2.073802` GH200-hours.

That is a positive throughput/memory screen and is now an owner-approved IP-E2
SDPA promotion. Eager repeated the
already owner-accepted continuation negative only in Adam `exp_avg_sq` max-abs.
SDPA passed model-parameter and BN-var groups but failed its calibrated fresh-
process continuation envelope in Adam `exp_avg`, Adam `exp_avg_sq`, and BN mean;
all exact context/integrity/finite gates passed. Immutable reassessments under the
amended rule return hard PASS for eager and SDPA while preserving those numerical
diagnostic negatives (`08426908...36512` and `50cc83be...9d88c`). No Cell-1
rerun is needed. Cell 2 and the remaining frozen sequence may proceed serially;
production Envelope-B activation remains separately forbidden.

Cell 2 Job `532763` completed `0:0` in `00:28:49` on one `n204`
allocation. Eager and scoped compile each completed 16+256 accepted B4x8 windows
with zero invalid/discard/scaler skip. Compile measured `16.287596` versus eager
`15.425816` presentations/s: ratio `1.055866`, one-sided 95% lower bound
`1.050861`, and projected 20-epoch saving `1.645086` GH200-hours. Peak allocated/
reserved fell from `16.039/18.723` to `15.259/17.748` GB with no monotonic growth.
All five forward scopes compiled into five unique graphs during warm-up; the
measured interval had no compiler-counter delta or unexpected recompile. Exact
checkpoint/context/integrity/finite hard gates and every grouped numerical
diagnostic passed. This is a positive isolated compile screen; the already frozen
Cell 3 now tests whether it composes with promoted SDPA before any stack decision.

Cell 3 Job `533212` completed `0:0` in `00:29:32` on one `n69`
allocation. The same-node eager reference was only `13.961779` presentations/s,
confirming that cross-node absolute rates are not comparable; SDPA+compile measured
`16.527709` on that node. The paired ratio was `1.183783` with one-sided 95%
lower bound `1.178471`, projecting `5.402981` GH200-hours saved relative to that
slow-node reference. Both completed 16+256 accepted windows with zero invalid/
discard/scaler skip. The combination patched 12 Swin attention modules and
compiled five forward scopes with no measured counter delta or unexpected
recompile. Candidate peak allocated/reserved was `14.368/19.332` GB with no
monotonic growth. Exact checkpoint/context/integrity/finite hard gates passed;
grouped numerical trajectory distance remained a reported negative diagnostic.
The combination is therefore eligible for the frozen fresh B8 capacity probe and
same-allocation B4-to-B8 measurement, not yet a batch-recipe promotion.

Pre-Cell-4 B8 capacity Job `533364` completed `0:0` in `00:04:22` on
`n127`. The fresh SDPA+compile B8x4 process completed one warm-up plus 8/8
accepted windows with zero invalid/discard/scaler skip. Peak allocated/reserved was
`27.788/32.877` GB, only `32.231%` of visible memory, with no monotonic growth or
unexpected recompile. This passes the frozen OOM/85% capacity prerequisite and
enables sustained Cell 4; it is not a throughput or batch-recipe claim.

Cell 4 Job `533384` completed `0:0` in `00:29:32` on one `n463`
allocation. B8x4 SDPA+compile measured `21.554602` versus B4x8 stack reference
`16.723609` presentations/s: ratio `1.288873`, one-sided 95% lower bound
`1.273372`, and projected 20-epoch saving `6.544602` GH200-hours. Both completed
16+256 accepted windows with zero invalid/discard/scaler skip, no monotonic growth
or unexpected recompile, and passing exact checkpoint/context/integrity/finite
hard gates. B8 peak allocated/reserved was `27.833/37.982` GB (`37.235%`
visible); its numerical trajectory distance remained a diagnostic negative.

The frozen conditional B16 margin does not pass: `R4=19,331,547,136`,
`R8=37,981,519,872`, and `R8 + 2*(R8-R4) = 75,281,465,344` bytes, or
`73.8014%` of visible memory, above the owner-frozen `70%` prerequisite. No B16
capacity or sustained job is therefore executable. B8 remains measurement-only
because its physical batch changes BN statistics and worker RNG assignment.

Cell 5 Job `533512` completed `0:0` in `00:26:16` on one `n145`
allocation. Fused AdamW measured `23.284372` versus the same-node unfused B8
SDPA+compile reference at `22.348477` presentations/s: ratio `1.041877`, one-sided
95% lower bound `1.038384`, and projected 20-epoch saving `0.883318` GH200-hours.
Both processes completed 16+256 accepted windows with zero invalid/discard/scaler
skip, identical peak reserved memory at `37.982` GB, no monotonic growth or measured
recompile, and passing exact checkpoint/context/integrity/finite hard gates. The
fused process's Adam `exp_avg` relative-L2 envelope is a diagnostic negative; every
group is finite and structurally intact. Fused AdamW therefore joins B8+SDPA+compile
in the measurement-only best stack for Cell 7. Cell 6/B16 remains skipped; Cell 7
runs that best stack first and the original B4 eager reference second so the final
end-to-end pair reverses the reference/candidate process order.

Cell 7 Job `534737` completed `0:0` in `00:28:05` on one `n411`
allocation. The best B8+SDPA+compile+fused stack ran first and measured
`21.803544` presentations/s; the original B4 eager reference ran second and
measured `15.152073`. The reversed-order ratio was `1.438981`, with a one-sided
95% lower bound of `1.413203`. Both processes completed 16+256 accepted windows
with zero invalid/discard/scaler skip, no memory growth or recompile, and passing
exact checkpoint/context/integrity/finite hard gates. Best-stack peak allocated/
reserved was `27.833/37.982` GB (`37.235%` visible), versus
`16.039/18.723` GB for B4 eager. The final-stack B16 projection is `76.498` GB,
or `74.994%` visible, and independently reaffirms the frozen B16 skip.

### 1.5 IP-WP4 synthesis and IP-G2 inputs

The direct same-node projection, including measured startup/compile-cold and
per-epoch checkpoint/hash costs, is `32.245064` GH200-hours for B4 eager versus
`22.443901` for the best stack over 20 epochs: `9.801163` GH200-hours saved. As a
separate scaling estimate, applying the point ratio to the documented `29.87`
GH200-hour Camera projection gives `20.757745` hours; using the throughput lower
bound gives `21.136384` hours. The corresponding conservative saving range is
therefore `8.733616-9.112255` GH200-hours, without treating cross-node absolute
rates as aligned.

IP-E2 consumed `2.871389 / 4.0` ordinary GH200-hours and no code-bug reserve. Its
direct-pair break-even is `0.292964` one 20-epoch Camera run. Including all IP-E1
and IP-E2 compute (`4.635278` GH200-hours), the whole Phase I-P preflight breaks
even after `0.472931` such a run under the direct Cell-7 saving. Unused IP-E2
budget expires with the exact sequence and is not authority for another cell.

| Candidate | Class | Terminal evidence | Final IP-G2 disposition |
|---|---|---|---|
| Camera Swin SDPA | numerical runtime, measurement-only | isolated `+6.871%`, lower bound `+6.561%`; hard gates pass | owner-accepted for the final Camera recipe |
| scoped Camera compile | numerical runtime, measurement-only | isolated `+5.587%`, lower bound `+5.086%`; five stable graphs, no measured recompile | owner-accepted for the final Camera recipe |
| SDPA+compile | numerical runtime stack, measurement-only | B4 ratio/lower bound `1.183783/1.178471`; final stack confirms composition | owner-accepted composition |
| fused AdamW | numerical runtime, measurement-only | B8 delta ratio/lower bound `1.041877/1.038384`; no extra reserved memory; hard gates pass | owner-accepted for the final Camera recipe |
| B8x4 | recipe/operational, measurement-only | stack delta ratio/lower bound `1.288873/1.273372`; effective B32 retained | owner explicitly accepts changed BN statistics and worker RNG assignment |
| B16x2 | recipe/operational, measurement-only | capacity `63.9556%` reserved; two-order ratio/lower bound `1.186583/1.182178` and `1.132013/1.128524`; all hard gates pass | promoted; owner explicitly accepts the BN/worker-RNG recipe change relative to B8x4 |
| checkpoint cadence | operational/material | measured synchronous save+hash is only about `0.84-0.89 s/epoch` | owner retains one recovery checkpoint per epoch |
| IP-WP2 augmentation/grid plumbing | strict output-neutral engineering | all ordered Camera candidates were slower than reference | reject from the Phase-I production stack |

The final Camera recipe is B16x2/effective B32 with SDPA, five-module forward-only
scoped compile, fused AdamW and one-epoch recovery cadence. The v3 production
config has file SHA-256 `25f53fc554c348c329c7a9cf4b9a5c8d521d993908114fbf64a46f75b3db0bda`
and resolved SHA-256 `f6040d30c23571f049bba3602081a9ec3bbfbdafc5d5ab8b76e9dd375eb76f25`.
Terminal profiler records retain the historical B4 file/resolved identities and
source commits; tests reconstruct the old resolved identity instead of relabelling
those runs as production-v3 evidence.

### 1.6 B16 veto withdrawal and terminal extension

The owner withdraws the `70%` projected-memory precondition. Implementation
`df3c17e3e6be19dcc586fdec2c6bd198c1b02d95` bumps paired comparison to schema v2:
the old projection and former verdict remain visible diagnostics, but B8 health,
checkpoint and no-growth checks alone make an OOM-tolerant fresh B16 capacity
probe eligible. Only the actual capacity result may pass the unchanged `<=85%`
visible-memory hard gate; it cannot authorize sustained B16 by itself.

The exact extension is fresh B16 capacity, then one same-allocation B8->B16
sustained pair if capacity passes, then a reversed B16->B8 confirmation only if
the first pair is positive. It retains D_fit, seed 0, effective B32, the promoted
SDPA+compile+fused stack, exact checkpoint/context/finite hard gates and every
claim prohibition. On 2026-07-21, the owner activated Section 9.2 with a `1.20`
base plus `0.50` code-bug reserve and `1.70` hard charged-GH200-hour ceiling at
maximum concurrency one. Any accepted stack still requires production validation
and an independently reviewed recipe freeze before Envelope B can be revised or
activated.

Cell A / Job `535315` passed the actual B16 capacity gate: B16x2 completed one
warm-up plus 8/8 accepted windows with zero invalid/discard/scaler-skip/nonfinite,
no monotonic growth or measured recompile, and peak allocated/reserved
`54.686/65.238 GB` (`63.9556%` visible). It consumed `0.073889` charged GH200-hour.
This authorizes only the same-allocation B8->B16 sustained pair; it is not a B16
recipe promotion or sustained-throughput conclusion.

Cell B / Job `535343` then passed every sustained-health and checkpoint hard gate.
On one `n141` allocation, B8 measured `22.519035` and B16 `26.720715`
presentations/s; B16/B8 point ratio and one-sided 95% lower bound were
`1.186583/1.182178`. B16 peak reserved remained `65.238 GB` (`63.9556%`). The
positive first pair triggers only the frozen reverse-order Cell C. Through Cell B,
the extension consumed `0.499445/1.20` base GH200-hours with reserve untouched.

The initial Cell C Job `536510` failed after 13/13 pretests but before output
creation, D_fit/model execution or training: descriptive output suffix
`b16_c2_b16_first` did not equal attempt ID `b16_c2_b16`. This is a single
command/provenance defect. The O-149 replacement changes only the two attempt IDs
to match the frozen output suffixes, costs `0.020278` bug-reserve GH200-hour, and
retains every candidate, scientific, measurement and resource boundary.

The derived replacement Job `536621` completed on `n421`. B16-first measured
`27.031478` and B8-second `23.879129` presentations/s; ratio/lower bound were
`1.132013/1.128524`, with every measurement/checkpoint hard gate PASS. Across both
orders, the conservative confirmed B16-over-B8 advantage is therefore `12.852%`.
B16 reserved memory was exactly `65.238 GB` (`63.9556%`) in capacity and both
sustained runs, with `36.767 GB` headroom, no growth and no measured recompile.

The extension consumed `0.913889` base plus `0.020278` bug-reserve GH200-hour,
`0.934167` total. Conservative B16-over-B8 projected saving is `2.382176`
GH200-hours per 20-epoch Camera run, so the extension breaks even after `0.392148`
runs. Unused budget expires. This is strong throughput/health evidence, not a
capability claim. The owner subsequently promoted B16x2 and explicitly accepted
its BN/worker-RNG recipe change. Production implementation is
`299277e8bdb8f60a05e8f06c2c0706e29252b51c`; Envelope B stays frozen pending a
new exact source/config/resource projection and independent recipe-freeze review.

A direct B32x1 capacity attempt is not justified by the current memory evidence.
Using the actual final-stack B8 and B16 peaks, a linear per-sample extrapolation
gives about `108.392 GB` allocated and `119.752 GB` reserved at B32, respectively
`106.26%` and `117.40%` of the `102.005 GB` visible device memory. Thus the plain
current graph is expected to OOM before accounting for safety margin; B32 would
require a separate activation-memory reduction design and new owner approval.

### 1.7 Owner-frozen Camera follow-up and DDP order

The next work is sequential rather than a new scientific search:

1. On the final B16x2/SDPA/scoped-compile/fused-AdamW stack, run one short
   structured stage trace. Preprocessing must remain the largest named Camera
   forward range before further preprocessing work proceeds.
2. In one single-GH200 allocation, compare fresh current-B16 and conservative
   B16 batched-affine/grid processes with identical source/config/CBGS/input
   anchors and the existing 16-warm-up plus 256-window sustained protocol.
3. The conservative item may unlock implementation of one later candidate only
   when every health/checkpoint gate passes and its one-sided 95% speed-ratio
   lower bound is at least `0.98`. That later implementation may combine one
   batched rotation `grid_sample` call with the non-persistent static grid; it is
   not implemented speculatively and no result promotes it automatically.
4. Only after this single-GPU follow-up is terminal may the session implement and
   qualify same-node 2-GH200 DDP. The exact comparison is 1 GPU B16xaccum2 versus
   2 GPUs, B16 per rank x accumulation 1, retaining effective global B32.
5. The DDP performance gate is a one-sided 95% aggregate-throughput speed-ratio
   lower bound of at least `1.60`. Against the now-frozen `13.285290 h` Camera
   projection, its descriptive wall/charged bounds are `<=8.303306 h` and
   `<=16.606613` charged GH200-hours; the fresh same-allocation reference
   projection is the binding denominator. Charged cost remains at most `1.25x`
   the single-GPU path.
   The union of rank shards must introduce no DDP-specific duplicate/drop relative
   to the exact expanded CBGS sequence; all ranks must agree on accepted/skipped
   updates and finite state, and checkpoint/resume plus rank state must pass.
6. Four-GPU DDP is excluded from the first qualification. B8 per rank would be a
   new BN/worker-RNG recipe; B16 per rank would change global batch to B64.

Ordinary DDP also changes BN running-buffer evolution: the single process performs
two sequential B16 BN updates per optimizer step, whereas each DDP rank performs
one. The 2-GPU profiler may measure this only under a separately frozen
measurement-only recipe; accepting that behavior or implementing an exact buffer
reconciliation remains an owner decision before any production DDP promotion.
The exact 2-GPU source/tests and proposed resource ceiling are now prepared at
`e51df6efa04e6d151315c72b7d7016014852078c`; Slurm execution and production
promotion remain unauthorized.

Current IP-E3 state: the conservative unlock passed and the exact combined
batched-rotation/static-grid candidate then completed its exact conditional pair.
Job `539853` measured `27.537525` versus `26.340313` presentations/s, ratio/lower
bound `1.045452 / 1.025241`, with every hard gate PASS. Both candidates are positive,
but the simpler conservative affine/grid screen had the larger matched speedup.
A cross-allocation ratio-of-ratios bootstrap is diagnostic only but also favors the
conservative item (`1.049450`, one-sided lower bound `1.032706`). The owner accepts
both paths as output-neutral, qualified implementations. Conservative batched
affine/grid is the production default because it showed the larger matched benefit
with the simpler path. The combined static-grid plus batched-rotation implementation
is retained as a qualified optional path and is not selected by the production
recipe. These are not additive choices: the combined implementation already
contains conservative batched affine/grid. At that boundary the production
config/source binding remained open; the IP-E4 closure recorded below now freezes
it without activating DDP or Envelope B.

The owner then inserts one narrow Camera-only IP-E4 before DDP. It first profiles
and pairs the accepted conservative path against a mathematically equivalent
batched geometry/inverse implementation. Only a hard-gate PASS with one-sided 95%
throughput lower bound `>=1.02` may unlock a second pair adding one bulk
`uint8 -> float32` conversion. IP-E4 is limited to one GH200 at a time, 45 minutes
per job, `1.00` base plus `0.50` code-bug reserve and `1.50` hard charged-GH200-hour
aggregate. Resize/crop/flip/interpolation, augmentation values/order, data, model,
loss, optimizer, precision and checkpoint cadence remain frozen. DDP is not part
of this authority. Prepared implementation source
`b909d4ee7e02375e230f2d44b193aae1d0af399b` batches only the existing float64
3x3 construction/composition/inversion sequence; the bulk-conversion candidate
remains locked until the first pair's hard gates pass and lower bound is at least
`1.02`. Initial Job `541217` stopped before D_fit/model execution because its
trace output suffix did not equal its attempt ID; the recorded one-line-derived
runner repair consumes `0.016389` GH200-hour of bug reserve and changes no
scientific or measurement input. Derived Job `541221` then passed all hard gates:
vectorized geometry measured `37.482862` versus `26.934770` presentations/s,
ratio/lower bound `1.391616 / 1.379987`, at `75.525 GB` peak reserved. The frozen
rule therefore promotes vectorized geometry and unlocks the single bulk native-
image conversion candidate. Implementation `d732be28688df974fee14b5d7abc9bd00c4a07f6`
uses one added 1,658,880,000-byte float32 native-image tensor and leaves every
per-image interpolation/geometry operation unchanged. The exact second pair uses
a 33-minute cap so the remaining base budget cannot be exceeded; no DDP authority
follows from this result. Conditional Job `541688` passed all eight pretests but
stopped before D_fit/model execution because its descriptive attempt IDs exceeded
the profiler's 32-character limit. The derived replacement shortens only those
labels; cumulative bug-reserve use is `0.029444` GH200-hour.

Derived Job `541821` completed the exact same-allocation pair on n77. The
vectorized reference measured `36.018676` presentations/s and bulk conversion
measured `36.875959`; ratio/lower bound was `1.023801 / 1.022026`, narrowly but
unambiguously above the frozen `1.02` gate. Both sides completed 256/256 accepted
windows with zero invalid/discarded/scaler-skipped windows, exact paired input
anchor, checkpoint continuation PASS and no memory growth/recompile. Candidate
peak allocated/reserved memory was `54,666,490,368 / 75,522,637,824` bytes
(`74.0378%` visible), so every hard gate passed and bulk conversion is promoted.

The final production stack is physical B16 x accumulation 2, conservative
batched affine/grid, vectorized geometry/inverses, bulk native-image conversion,
SDPA, scoped forward compile and fused AdamW; checkpoint cadence remains one
recovery checkpoint per epoch. Commit `93cac472916e1c9c69c8910ad7034f11846e8cec`
binds that stack in the production runtime. The Camera config file/resolved hashes
are `2e5368f96a6198e9a3b1bd43b258b53675df49f5c6ca9042fa8f72e0084c3b6a`
and `0df1a19c057312923e0a8e48e81689d9ca265cc613c6f34d4795417414aa0bcf`.
The final matched projection is `13.285290` GH200-hours for the 20-epoch C-only
run. Relative to the original clean B4 `16.5390` presentations/s measurement,
the final `36.875959` rate is a descriptive cross-allocation `2.2296x` (`+122.96%`),
not a matched-node confidence claim.

IP-E4 consumed `0.782778/1.00` base plus `0.029444/0.50` bug-reserve
GH200-hours, `0.812222/1.50` total. Its unused capacity expires with closure and
does not authorize DDP, LiDAR, scientific training or Envelope B. The owner
subsequently approved the exact Section-9.6 IP-E5 request at containing request
commit `2505db02920021663ccce7783dee483f10e638f8`, including its `1.50` base plus
`1.50` diagnosed-code-bug reserve and `3.00` charged-GH200-hour hard ceiling.

### 1.8 IP-E5 DDP qualification — terminal positive and subsequently promoted

Implementation `e51df6efa04e6d151315c72b7d7016014852078c` binds both IP-E5
profiles to the production Camera config at file/resolved hashes
`2e5368f96a6198e9a3b1bd43b258b53675df49f5c6ca9042fa8f72e0084c3b6a` /
`0df1a19c057312923e0a8e48e81689d9ca265cc613c6f34d4795417414aa0bcf`.
Thus the reference is exactly the Job-541821-selected stack: physical B16 x
accumulation 2, conservative batched affine/grid, vectorized geometry, bulk input
conversion, SDPA, five-module forward-only compile and fused AdamW. `36.875959`
presentations/s is its best measured descriptive rate; IP-E5 reruns that recipe as
a fresh same-allocation control rather than treating the old n77 rate as matched
DDP evidence.

The two-rank sampler preserves each frozen global B32 permutation window exactly:
rank 0 receives positions `[0:16]` and rank 1 `[16:32]`, with no padding, striding,
duplicate or omission. Per-rank worker seeds are `seed + epoch*world_size + rank`;
this was a deliberate measurement-only DDP RNG recipe during qualification.
Ordinary DDP rank-local B16 BatchNorm was measured. Rank-0 buffers define the
checkpoint boundary outside the sustained timing, matching the next-forward DDP
buffer broadcast. The owner has now explicitly accepted both behaviors as
production recipe inputs.

The approved envelope is one node, two GH200s, 32 CPUs, 192 GiB and at most 45
minutes per job; maximum concurrency one; `1.50` base plus `1.50` code-bug reserve,
`3.00` charged-GH200-hour hard ceiling. One allocation performs a two-rank NCCL
smoke, fresh 1-GPU B16x2 control, fresh 2-GPU B16/rank x1 candidate, checkpoint/
resume and the paired analysis. Section 9.6 of `RUN_REQUEST.md` is the exact
request. The owner activated it at request commit `2505db02920021663ccce7783dee483f10e638f8`.
No D_select, D_audit, official validation, capability claim, Envelope-B
activation, merge or push is authorized.

Job `543028` ran on same-node n418 from source
`e25160f8811953c03e5805cf8c2917bc7f7ae2e0`. The focused tests passed `10/10`,
the two-rank NCCL smoke passed, and both the fresh single-GPU and DDP profile plus
fresh-process continuation completed every engineering hard gate. The single-GPU
reference measured `35.469970` presentations/s; DDP measured `64.886915`, a
`1.829348x` point ratio with one-sided 95% block-bootstrap lower bound
`1.818635x`. This passes the frozen `1.60x` gate. The aligned C-only projections
are `13.814431` one-GPU hours versus `7.581252` two-GPU wall hours and
`15.162504` charged GH200-hours, a charged ratio of `1.097584` that passes the
`1.25x` limit.

The DDP sampler reconstructed all `87,904` epoch-0 presentations without padding,
duplicate or omission. Both ranks had 256/256 accepted measured windows, no
overflow/discard/invalid window or steady-state recompile, and peak reserved
memory `75.807850/75.939971` GB (`74.32/74.45%`). Parameters, non-BN buffers,
AdamW, scheduler, scaler and discrete rank state agreed exactly at the canonical
checkpoint boundaries; fresh-process eight-window continuation passed. Ordinary
rank-local BN remained finite but intentionally non-identical: relative-L2
`0.005321`, max-absolute `0.110272`, cosine `0.999987`. This is the predeclared
recipe distinction requiring owner acceptance, not a failed engineering gate.
The fresh-process grouped numerical diagnostic had one narrow miss on both ranks:
BN running-mean max-absolute error `0.005498` versus its repeat-calibrated
`0.004822` diagnostic limit, while BN-mean relative-L2 `0.000457` passed its
`0.002` limit and parameters, BN variance and both Adam groups passed both
diagnostic limits. Exact boundary/input/RNG/discrete/structure and finite-state
hard gates all passed, as frozen in the approved IP-E5 numerical policy.

The Slurm job ended `1:0` only because the terminal CPU comparator compared two
equivalent source dictionaries byte-for-byte while the single-GPU schema alone
included descriptive `frozen_control_ref`. Repair
`26bf727ab36f5c3016b0c146eb8b8f3b3b66ec6d` now requires equality of every
material source field and ignores only that non-material extra key. It re-used the
immutable completed GPU artifacts and emitted write-once positive summary SHA-256
`0a0bd6569387c05cc170a129f9b83c94b6fefc2c5f8e6e6b0751d906d6d5a31c`;
no GPU rerun was needed. Job charge was `0.688889` GH200-hours, all from the base;
bug reserve use was zero and the unused authority expires with IP-E5 closure.

One residual performance warning reported a non-contiguous 1x1-convolution
gradient stride differing from its DDP bucket view. It did not affect correctness
or any hard gate and the measured result already clears the speed/payback gates;
it is optional future headroom, not a reason to rerun IP-E5. Production promotion
is now closed by the owner's explicit acceptance of ordinary per-rank B16
BatchNorm, `seed + epoch*world_size + rank` worker RNG, and the exact two-GH200
recipe.

The production topology is world size two, physical B16 per rank, accumulation
one/effective global B32, ordinary rank-local BN, rank-addressed worker RNG,
contiguous halves of every frozen CBGS B32 window, synchronized finite/scaler
control flow, and a rank-0 model checkpoint plus one RNG sidecar per rank. Source
`2c3780bb6373ae784b41c22df072824f7a92d457` introduces fail-closed
`s10.phase1.v4` validation and a dedicated Camera DDP capability runner; LiDAR
remains on the single-GPU runner. The Camera config file/resolved SHA-256 are
`9a2cdf54a52edeb71b5335aea8445c0a8cc0c8e2e416b2f4fe3df58d7b98710c` /
`e295b627551a584b460a598ee3e3f23b5ad8dda45441904d4ed526bbf3457f2b`.
Its fresh output root is
`.../outputs/s10_phase1_envelope_b_camera_ddp_5da03ffdaa29`.

This promotion is source/recipe authority, not compute authority. No production
DDP capability run, D_select, D_audit, official validation, original Envelope-B
activation, merge or push occurred. Before any Camera capability submission,
Section 7 must be revised to the promoted SHA/config/two-GPU resource projection
and its already-required independent recipe-freeze review must close with no open
P0-P2.

### 1.9 LiDAR throughput preflight — IP-LG0 closed, L-E1 pending

The owner approved the following continuous LiDAR workflow and thereby closed
`IP-LG0`. This starts local `L-WP1` work; it does **not** activate either compute
envelope.

| WP / gate | Input and work | Output / continuous authority | Stop or owner decision |
|---|---|---|---|
| L-WP0 diagnosis | current LiDAR model/loss/loader/CBGS/checkpoint paths plus historical B4 evidence | read-only bottleneck and candidate classification; closed | IP-LG0 freezes the workflow and source scope |
| L-WP1 clean measurement | exact D_fit LiDAR recipe; clean B4/B8/B16/B32 profiles | default-off capacity ladder, B4-versus-highest-safe sustained processes, two detailed traces, checkpoint and loss-health evidence | runs only after exact IP-L-E1 activation; IP-LG1 selects the batch and exact L-WP2 cells |
| L-WP2 primary screens | IP-LG1-frozen batch and bottleneck evidence | same-allocation paired Hungarian/target-host-sync, sparse-front-end sync/stat, LiDAR SDPA, dense scoped-compile and fused-AdamW screens | continuous only inside separately approved IP-L-E2; ambiguity or science pressure returns to owner |
| L-WP3 conditional/composed screens | positive primary candidates and trace residuals | conditionally test full-sort-to-topk, batched voxelization, batched Gaussian targets, H2D-field pruning and hidden-sync cleanup; validate the final combined stack | IP-LG2 promotes/rejects the exact L-only recipe and explicitly accepts any BN/worker-RNG batch recipe |
| L-WP4 capability handoff | owner-promoted LiDAR recipe plus promoted Camera 2-GH200 v4 recipe | new dual-branch Envelope-B source/config/hash/output roots and per-branch resource projection; independent recipe-freeze review | only a no-open-P0-P2 review may lead to a later Envelope-B activation request |

The candidate classes remain distinct. Batching or removing redundant host-side
diagnostic plumbing, exact batched Hungarian transfers, output-equivalent sparse
bookkeeping, SDPA, scoped dense compile and fused AdamW are engineering candidates
that still require forward/backward/update, FP16-policy, checkpoint/resume and
sustained-throughput validation. Physical B8/B16/B32 are measurement-only until an
owner explicitly accepts their BatchNorm and worker-RNG recipe. Sparse-convolution
FP16, normalization, precision/TF32, model/loss/target/order/exposure, scheduler or
evaluator changes remain material science and are outside L-E1/L-E2 unless newly
approved.

Implementation `0daeee95e1a46b29fcd7bbb2338d813b798557de` adds four clean,
default-off effective-B32 profiles, a single-allocation orchestrator, terminal-only
loss/component health, and detailed sparse/head/decoder/Hungarian trace ranges.
It does not change model math, accepted updates, precision or the production LiDAR
config. Capacity runs use one warm-up plus eight accepted windows in fresh
processes, ordered B4 -> B8 -> B16 -> B32 and stop at the first failure. Sustained
runs use 16 warm-up plus 256 accepted windows in ABBA order for B4 and the highest
safe batch; each side has two processes and receives a conditional third only when
its first-two spread exceeds 3%. One 16+3 trace is then taken at B4 and at the
highest safe batch.

Hard evidence remains exact for source/config/data role, CBGS exposure, boundary,
input stream within a recipe, RNG/discrete state, accepted windows, finite/scaler
control and checkpoint structure. Fresh continuation uses the owner-amended grouped
parameter/BN/Adam tolerances; elementwise allclose remains diagnostic. All reported
LiDAR loss components must be finite, but a short-window loss slope is descriptive
training-health evidence rather than a speed, capability or promotion gate.
Different physical batches are not required to reproduce one another's worker RNG.

Section 9.8 of `RUN_REQUEST.md` is the exact proposed L-E1 request: one GH200,
16 CPUs, 96 GiB, at most 75 minutes, maximum concurrency one, `1.25` base plus
`0.50` diagnosed-code-bug reserve and a `1.75` charged-GH200-hour hard ceiling.
It is `OWNER APPROVAL PENDING / NOT EXECUTABLE`. No GPU/Slurm, D_select, D_audit,
official validation, capability claim, Envelope-B activation, merge or push has
been authorized or performed by IP-LG0 closure.

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
