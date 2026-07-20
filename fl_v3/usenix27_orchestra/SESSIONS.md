# USENIX Security '27 Orchestra — milestone contracts

> **Current handoff (2026-07-20).** S07-S09 are closed. S10 is active on
> `codex/s10-phase1-branch-qualification`, advanced linearly from
> `codex/s10-cl-model-recipe`. O-143 replaces the old active S10 six-stop order:
> qualify camera and LiDAR independently, perform staged fusion from qualified
> branch checkpoints, establish aligned capability/fusion contribution, and only
> then profile/optimize GH200.
>
> STOP-A's split/evaluator remains reusable. STOP-B is closed
> `INCONCLUSIVE`; C1-A's `LOCALIZED_NORM` result is bounded diagnostic
> evidence, not a recipe decision. C1-B proxy scores remain insufficient to
> establish usability or improvement over Alvis. Current-A2 and the old C→D→E→F
> route are paused.
>
> O-144 closes `P1-G0 PLAN_FREEZE`; the binding choices and five-WP/
> three-gate/two-envelope workflow are in
> `handoffs/S10/PHASE_I_PLAN.md`. O-145 adds the optimized CUDA BEV-pooling
> implementation/parity/timing contract to WP2/WP4 and authorizes exact
> Envelope-A drafting only. Envelope A is not activated.
>
> S10 now uses phase-level owner approval, one compact status file and one job
> ledger. There is no current compute authority; S11+ remains pending.

## 1. Active graph and status

```text
351b7a0 accepted and integrated S09 close
  │
  ├─ S08 model/recipe audit → owner decisions → precision qualification
  │      └─ independent review only after exact implementation/evidence SHA
  │
  ├─ S09 full-pipeline performance/readiness      [closed PASS under O-120]
  │      └─ independent review of exact profiling/evidence SHA
  │
  ├─ S10 C/L qualification → staged fusion        [Phase-I plan frozen; Envelope A pending]
  │      └─ capability gate → GH200 optimization  [pending]
  └─ S11 and later                                 [roles pending owner decision]
```

| ID | Milestone | Depends on | Status / output |
|---|---|---|---|
| S00 | Persistent owner/implementation/review coordination | — | active; sole canonical-doc writer and default implementation context |
| S01 | Stored-ZIP nuScenes data foundation | closed | reviewed PASS within recorded scope; `t1.v1` production use forbidden |
| S02-S05 | Loss/target, camera, SECOND, CenterHead/decode modules | closed | reviewed module contracts integrated into clean anchor |
| S06 | C/L/F resolved runtime/checkpoint/eval contract | closed | reviewed bounded contract integrated into clean anchor |
| S07 | Legacy cleanup plus clean completion | S01-S06 | **closed**; S07-C static review PASS and S07-B bounded FP32/FedAvg/loader gate PASS; no science/precision freeze |
| S08 | Model/recipe audit, then precision qualification | S07 | **closed PASS under O-110** at accepted seal `d31adea`; Jobs `431013`/`435151`, `00:07:58` total; R3 no P0-P2 |
| S09 | Full-pipeline engineering performance/readiness | accepted S08 policy | **closed PASS under O-120** at accepted review seal `ced5992`; STOP-1 through STOP-4 independently reviewed, no open P0-P3 |
| S10 | C/L branch recipe and capability, staged fusion, aligned clean/fusion claim, then final-architecture GH200 optimization | closed S08+S09 | O-145 pooling amendment under O-143/O-144; STOP-A reusable; Envelope A exact draft pending approval; no implementation/compute authority |
| S11+ | Not currently defined | future owner decision | pending; historical role proposals do not create scope, sequencing, full-run placement, or execution authority |

## 2. Persistent S00 contract

**Purpose.** Keep one coherent technical context, one linear branch and a small
set of evidence records. `Sxx` is an evidence namespace, not a worker/reviewer
lifecycle.

**S10 workflow under O-143/O-144/O-145.**

1. `PHASE_I_PLAN.md` freezes the Phase-I objective, two candidates, graph/recipe,
   data/metric/seed/exposure policy and the five-WP/three-gate/two-envelope model.
   Each envelope separately binds aggregate resources, submission cap, outputs and
   stop conditions before it becomes executable.
2. S00 implements and uses direct entry/config/checkpoint/one-batch preflight.
3. Inside an approved phase, S00 may fix output-neutral test, runner, checkpoint
   I/O or logging defects and resubmit within the same scientific/resource cap.
4. S00 returns to the owner before changing model math, data ownership, recipe
   search space, evaluator/metric, seeds, candidate count, interpretation or
   aggregate resources, and when repeated engineering failure exhausts the cap.
5. Record current state in one `HANDOFF.md` and runs in one `RUN_REQUEST.md`
   ledger. Preserve raw outputs and minimum provenance; do not duplicate every
   incident into canonical docs, results and review files.
6. Review only data/metric changes, each branch recipe freeze, and the final
   staged-fusion/full capability result. A reviewer reads a durable SHA/evidence
   and does not fix code.

**Authority boundary.** O-144 freezes the plan and O-145 amends WP2/WP4 and permits
the amendment commit plus exact Envelope-A drafting, but neither activates Envelope A
or B. No implementation, checkpoint acquisition, GTDB materialization, further commit,
compute, merge, push, upload or publication is authorized. Future envelope
authority must state its exact scope, aggregate resource and submission limits.
S11+ remains pending.

## 3. Closed foundation summary

The exact SHAs and accepted scopes are in ORCHESTRA Section 2. The current anchor
preserves:

- S01 read-only ZIP/cache/data contracts and S07-A provenance integration;
- S02 target/loss correctness;
- S03 independent stride-8 camera branch;
- S04 low-resolution SECOND/spconv module contract;
- S05 six-task CenterHead and deterministic decode/NMS;
- S06 resolved C/L/F config, production loop, checkpoint/resume, and official clean
  evaluation interfaces;
- S07-C removal of active legacy attack/defense/harness routes;
- one plain clean FedAvg implementation and S07-B-COMPLETE's one-step FP32/loader
  engineering evidence.

Closed S01-S07 worker prompts and remediation chains are not active session
contracts. Read their handoffs when their evidence is needed; do not relaunch them.

## 4. S08 — current six-task precision qualification

**Closed policy.** The detailed current-model, official-reference,
precision, gradient and training-recipe audit is
[`handoffs/S08/MODEL_RECIPE_AUDIT.md`](handoffs/S08/MODEL_RECIPE_AUDIT.md). It
classifies the detector as a BEVFusion-class shared-CenterHead hybrid, identifies
tiny-group sparse GroupNorm as the leading but unproven L/F gradient mechanism,
and shows that the strict runtime is not yet a frozen scientific recipe. O-097
accepts the audit boundaries and O-098 accepts the detailed implementation plan.
Architecture and normalization remain unchanged. The exact implementation, five
consumed smoke outcomes, initial `REMEDIATE` review, and remediation R2
`PASS_WITH_RESIDUAL_RISK` are in
`handoffs/S08/{HANDOFF,RUN_REQUEST,RESULTS,REVIEW}.md`. Exact Q1/Q2 Jobs
`431013`/`435151` completed in `00:07:58` total under O-109; they did not reopen
the accepted v1 architecture or training recipe. R3 reviewed evidence SHA
`c0ef86235ead753fee3b790b19d40f82f875ec59` with no P0-P2 findings. O-110 accepts
close-ready seal `d31adea`, freezes the reviewed policy, and closes S08 PASS.

**Question.** What precision regime can train the current six-task C/L/F model
stably on GH200, and why do L-S075/F-U overflow in the existing FP16 path while
older Arrhenius LiDAR evidence appeared stable?

**Current facts to preserve.**

- Job `389356`: all FP32 C/L/F gradients were finite and optimizer calls occurred.
- FP16 scale 512 overflowed C/L/F; scale 1 recovered C only.
- L-S075/F-U at scale 1 retained direct nonfinite sparse SECOND stem/stage1
  gradients. Their FP32 maximum gradient elements were approximately 1.91M/1.22M,
  while surviving FP16 elements approached its finite range.
- Historical Jobs `211502`/`211722` used an older voxel path with spconv kept in
  FP32 inside outer AMP. They do not validate the pre-S08 automatic sparse-conv
  FP16 path or either explicit S08 partition now preserved by `s09.v1`.
- Job `390576` proves only one successful FP32 update per C/L/F mode.

**Implementation/diagnostic scope.**

- make the precision choice explicit at the current production resolver rather
  than silently deriving sparse-conv FP16 from `precision=fp16`;
- preserve FP32 and direct-BF16 rejection;
- provide three clearly labelled regimes for qualification:
  `fp32`, current full FP16 AMP, and FP16 AMP with SECOND/spconv FP32 island;
- exercise the real production training loop and dynamic GradScaler continuation,
  including backoff below scale 1 when reached; do not judge AMP from one fixed
  attempt;
- use minimal opt-in window-end diagnostics: record per-window scaler before/after,
  skips, successful optimizer/exposure steps, scalar/per-task losses, and
  elementwise gradient finiteness after unscale and before clip/step/clear;
- localize gradients at the multi-task head input and SECOND stem/stage boundaries
  without changing loss/head semantics;
- cover exact C-STR8, L-S075, F-U and short L-P020/F-CBGS compatibility cells;
- keep all diagnostics bounded and output-neutral when disabled.
- do not build a general observer/hook/profiler framework; parameter-gradient
  summaries require no hook, and any boundary tensor tap must have an explicit,
  short lifecycle.

**Forbidden without a new owner decision.**

- changing task groups, head/loss equations, target construction, architecture,
  optimizer recipe, metric, data ownership, or capability thresholds;
- comparing scientific accuracy across precision regimes;
- declaring FP16 impossible from one mini batch;
- treating FP32 F1 as the final scientific policy;
- full trainval, 100/1000 steps, profile, mAP/NDS, DDP, matrix, or rerun.

**S08 acceptance evidence.** Before S09, independent review must establish:

1. exact regime resolution and provenance are fail-closed;
2. the chosen candidate executes a bounded multi-window test with finite accepted
   updates for C/L/F, not merely finite scalar loss;
3. dynamic-scaler skips and exposure/scheduler/EMA counters remain correct;
4. L/F failure or recovery is localized sufficiently to justify the policy;
5. L-P020/F-CBGS do not carry an unexamined incompatible precision route;
6. limitations explicitly exclude convergence, speed, capability, and science.

**Compute.** O-098 approved separate one-GH200 `<=1h` resource ceilings for the
focused smoke and later Q1. O-099 bound the first exact tuple; Job `426619` then
failed before pytest because the source verifier rejected a known non-executable
spconv build-metadata patch already present in S07 evidence. That request is
consumed. O-100 authorized exact state binding; O-101 then bound S08-SMOKE-2.
Job `427800` passed that attestation but failed three focused tests. Both requests
are consumed. O-102's narrow remediation passed local static validation; O-103
then bound exact S08-SMOKE-3, whose sole Job `428112` passed all 106 focused tests
and completed `0:0` with zero restarts. All three smoke authorizations were consumed;
at that point no retry was implicit, Q1 remained unapproved, and any further job
needed new approval.
The independent review of exact SHA
`791aba97f7bbe92e7708b63f94f2e7d8599f91be` returned `REMEDIATE`; O-104
allows the bounded code/test/document remediation and exact S08-SMOKE-4 request
preparation. O-105 then bound it once. Job `428889` is terminal `FAILED 1:0`:
115 passed/1 failed, with the only failure caused by assigning `1.0` to an already
`1.0` synthetic calibration element. Phase 2 did not run. No retry, source edit,
or remediation commit is authorized under O-105. A later owner message directed
Smoke-5 before its exact tuple existed; the narrow correction/request were then
frozen. O-106 approved that exact tuple; Job `429080` is terminal PASS with 116+1
tests and all five candidate fixture identities. O-108 sealed remediation at
`103c7389`; R2 returned `PASS_WITH_RESIDUAL_RISK`. O-109 then authorized and
consumed Q1 Job `431013` (`00:04:02`) and Q2 Job `435151` (`00:03:56`). Q1 found
full sparse FP16 narrowly recoverable for L-S075 but bounded-failing for F-U,
while the SECOND-FP32 island passed both. Q2 passed L-P020 global FP16 and F-CBGS
with the SECOND-FP32 island. No scientific retry or extra cell ran. The next gate
was owner precision-policy acceptance of the reviewed candidate; O-110 completes
that gate.

## 5. S09 — full-pipeline performance and readiness

**Starts when.** O-110 has accepted reviewed S08 evidence and frozen the precision
policy. O-111 accepts the engineering-focused four-stop direction; each stop still
requires one exact owner approval before S00 creates its completion goal.

**Question.** Can the selected current C/L/F training path achieve stable,
resource-efficient optimizer steps on production-shaped data before expensive CL
capability runs?

**Required plan.**

- materialize or bind the exact approved full trainval `t1.v2` cache/manifest
  identities before production-shaped execution;
- run a 100-step engineering gate first; run 1000 steps only when the 100-step
  acceptance criteria are met and the second exact request is approved;
- collect warm-up-separated p50/p95 step time, samples/s, forward/loss/backward/
  optimizer time, data wait, and epoch estimate; official evaluation is not
  invoked in this engineering gate;
- collect peak allocated/reserved memory and headroom under the 96 GiB envelope;
- compare `num_workers=0/2/4/8` on the actual ZIP/cache path without conflating
  iterator reset or first-epoch warm-up with steady state;
- determine microbatch, accumulation, effective global batch, optimizer-step and
  exposure semantics; record nonfinite/scaler skips under the S08 policy;
- establish one-GH200 performance first. Propose two-GPU DDP only if measured
  single-GPU throughput is unacceptable and the owner approves a distinct request;
- use minimal production instrumentation. Do not restore old audit/profiler
  harnesses, process matrices, source manifests, warnings-as-errors, or custom
  long-TMPDIR wrappers.

**STOP-3 gates frozen by O-117.** Require all loader digests equal; worker-8 warm
throughput at least 90% of the best warm cell; exactly 100 successful updates
within 120 attempts; zero nonfinite/discarded windows or counter drift; at least
95% accepted windows after ten successful warm-up windows; integrated
`(data_wait + CUDA H2D-through-update)` p95/p50 no greater than 1.5; measured
data-wait share no greater than 10%; peak reserved memory no greater than 86 GiB;
and both declared epoch estimates no greater than 24 hours. Aggregate loss must
be finite; monotonic convergence is not a STOP-3 gate or claim. OOM, loader
failure, unresolved identity mismatch, unusable telemetry, or an unmet bounded
update target is a terminal failure with no retry authority.

**Not S09.** mAP/NDS, per-class capability, fusion gain, recipe selection,
multi-seed science, Protocol A/B, attack, or defense.

**Engineering-optimization boundary.** S09 may use measured evidence to remove
output-neutral ZIP/cache, loader-lifecycle, H2D, redundant conversion/sync/
allocation, logging, or checkpoint overhead. It must preserve data/sample order,
model/loss/gradient/update semantics, O-110 precision, and exposure accounting.
Any candidate source change receives an exact owner-reviewed file and equivalence-
test envelope; an unsuccessful G100 does not authorize a silent optimization or
rerun. Model math, normalization and training-recipe changes remain S10 work.

**Residual large-gradient boundary.** S08 did not reduce the true optimizer
gradient: GradScaler unscales before step, and the accepted SECOND-FP32 island
avoids sparse-FP16 overflow. Job `389356` still shows approximately 1.91M/1.22M
FP32 L/F maxima. Repeated tiny-group per-voxel sparse GroupNorm is the leading but
unproven mechanism. S09 records accepted/skipped/nonfinite windows and loss/scaler
behavior without enabling S08 diagnostics or changing normalization, loss/head,
clip, optimizer, scheduler, EMA, augmentation, sampling, or initialization. A
100-step instability attributable to this residual is a stop, not permission to
perform an architecture experiment.

**Four owner stops.** The durable details live in
`handoffs/S09/{HANDOFF,RUN_REQUEST}.md`.

1. `STOP-1 DATA`: materialize/review/bind the exact full trainval `t1.v2` caches.
2. `STOP-2 IMPLEMENTATION`: add minimal hash-bound readiness mode and direct
   timing/memory accounting, verify output neutrality, seal an implementation
   commit, and run only an owner-approved O-009 focused smoke.
3. `STOP-3 G100`: execute one exact production ZIP/cache worker sweep and one F-U
   100-successful-step single-GH200 gate, then independently review it. If a
   performance threshold fails, the stop returns an exact output-neutral
   optimization proposal rather than silently patching/rerunning.
4. `STOP-4 OPTIMIZE/G1000/CLOSE`: under O-119, first profile the exact STOP-3
   B=1/checkpoint-on baseline and characterize checkpoint-off B=1/2/4 capacity;
   remove only proven output-neutral synchronization/allocation, validate an
   optimized B=1 G100, then conditionally execute a fresh B=1 1000-successful-step
   single-GH200 gate. B=2/4 do not select a recipe, and no run resumes G100 state.

**Compute.** STOP-1/3/4 material jobs require their own exact stop approval. The
STOP-2 focused smoke may use O-009/O-107 only when that stop approval explicitly
opts in. Unused quota is not retry authority. O-112 STOP-1 Job `441191` and O-115
STOP-2 Job `441293` are consumed and terminal; no replacement was used. O-117
Job `441511` consumed its sole exact submission and failed before loader/model
execution. O-118 separately authorized and consumed Phase-A Job `442152` and
conditional Phase-B Job `446225`; no retry occurred, all frozen STOP-3 gates
pass, and independent review found no open P0-P3 after remediation. O-119 accepts
STOP-3 and approves three serial STOP-4 jobs at `00:30:00`, `00:30:00`, and
`01:00:00`, one GH200/16 CPUs/96 GiB each, at most two cumulative GPU-hours and
no retry; each immutable tuple is recorded and independently reviewed before its
conditional submission.

O-119 is now fully consumed. STOP-4A Job `452520`, optimized G100 Job `455539`,
and fresh G1000 Job `456539` completed `0:0` without retry, totaling `0.345000`
GPU-hours. The final B=1 G1000 run reached 1000 accepted updates in 1003 attempts;
its p50/p95 was `178.024/203.231 ms`, throughput `5.542 samples/s`, epoch estimate
`1.409821 h`, and peak reserved memory `8.314 GiB`. Final independent review of
implementation/evidence/remediation found no open P0-P3 and marks S09 owner-ready.
This is bounded single-seed engineering evidence, not convergence, mAP/NDS,
model-quality, recipe-selection, or full-GH200-saturation evidence.
O-120 accepts review seal `ced5992` with those limits and closes S09 PASS.
O-121 later completed fast-forward-only integration at `351b7a0`; every S10
implementation or compute action still requires a fresh owner decision.

## 6. S10 O-143 scientific rebaseline

The historical A-F contract and all O-122–O-142 job outcomes remain evidence in
the S10 handoff archives, but they no longer define the active execution order.

### Phase I — independent branch qualification

O-144 freezes the complete Phase-I specification in
`handoffs/S10/PHASE_I_PLAN.md`. It contains exactly one ImageNet-initialized
standalone-reference Camera primary and one scratch reference-led LiDAR primary;
physical B4 plus accumulation 8/effective B32; reference optimizer/scheduler and
augmentation bundles; D_fit-only role-bound CBGS/GT-paste; seed 0; 20 epochs;
epoch-20 terminal-only selection; one D_select evaluation and owner-unsealed
one-time D_audit; and no automatic repair candidate.

O-145 makes the independent optimized CUDA BEV-pooling/equivalent-kernel backend
part of WP2 and binds its fallback parity, FP32/FP16 policy checks, and GH200
operator/end-to-end timing to WP4. This does not add a scientific candidate. The
Camera initialization URL is the reference YAML's ImageNet Swin-T asset, not the
optional NuImages asset.

The phase must use meaningful trainval-scale exposure and aligned internal
evaluation. Step smokes establish only that a run executes. The phase exits only
with a qualified camera checkpoint and LiDAR checkpoint, or an honest negative
result. One combined branch recipe freeze review is allowed at one durable SHA.

### Phase II — staged fusion and capability

Initialize fusion from the qualified C/L checkpoints; train the fusion-specific
components and then the approved unfrozen scope. Compare camera, LiDAR and fusion
under aligned data, class, metric, checkpoint-selection and evaluation semantics.
The gate must answer both absolute clean capability and fusion contribution, and
must include a fair comparison with the historical Alvis detector if its
checkpoint/provenance/evaluator can be aligned. The final capability result
requires independent review.

### Phase III — GH200 performance

Only a capability-passing frozen graph and recipe may enter profiling. Measure
coverage, synchronization, throughput, utilization, memory and operator-level
cost before optimization. Preserve numerical/metric behavior and do not optimize
a scientifically failed model.

No phase implementation or compute is authorized by O-144/O-145. The next owner action
is exact activation of Envelope A for WP0-WP4, official ImageNet acquisition,
D_fit CBGS/GTDB materialization, optimized-BEV-pooling build/parity/timing,
material commits and bounded engineering calibration. Envelope B remains pending
measured `P1-G1` approval.

## 7. S11 and later

All roles, ordering and execution boundaries after S10 are pending. O-143 keeps
the primary capability/fusion claim inside S10, but does not define S11 or
authorize that run. Historical S11-S15 descriptions remain context, not
authority. No Protocol A/B execution, attack, defense, paper upload, or
publication work is authorized.

## 8. S10 durable delivery and review

Under O-143, active S10 state lives in one compact `HANDOFF.md`; execution
authority and run provenance live in one `RUN_REQUEST.md` ledger. Existing
`RESULTS.md` and `REVIEW.md` are historical archives and are not updated for
every new incident.

Minimum scientific-run provenance is Git SHA, resolved-config hash, split, seed,
command, resources, output root, terminal status, checkpoint hash and metric
artifact hash. Raw outputs are immutable. Broad test suites, report generation
and recursive manifests do not belong on the GPU critical path.

Independent review occurs for data/evaluator changes, each branch recipe freeze,
and the final staged-fusion/full capability result. Review checks leakage,
coordinate/unit semantics, config and checkpoint identity, exposure, precision,
metric denominators, failed/missing cells and shortcuts that could inflate
capability or fusion gain. Ordinary fixture/runner bugs do not launch reviewers.
