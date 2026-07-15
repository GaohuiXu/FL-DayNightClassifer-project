# USENIX Security '27 Orchestra — milestone contracts

> **Current handoff (2026-07-15).** S07-S09 are closed and integrated through
> `351b7a0b8419c01d0d32ba224babbc6bdc4213ba`. S08 accepted precision policy is
> recorded under O-110; S09 accepted bounded engineering review seal is
> `ced5992ea113bd21d7d545af505debf405b556b3`. O-121 records the completed
> fast-forward. Fresh persistent S00 completed the S10 startup audit at exact
> clean base `a080d49c1c22de20ccb5b1353d4922c7df14a729` and is active on
> `codex/s10-cl-model-recipe`.
>
> O-122 accepts the six-stop S10 A-F scientific envelope, exact STOP-A
> split/evaluator gate, and primary full claim “absolute clean capability +
> fusion contribution”. It permits planning records only. STOP-A/B/C
> implementation, commits, review topology and Slurm await one bounded owner
> completion authority after GPU-budget review. STOP-D/E/F execution and S11+
> remain pending.
>
> `Sxx` is an evidence namespace, not automatically a worker, task, branch, or
> worktree. Canonical decisions: [`ORCHESTRA.md`](ORCHESTRA.md). Active launch
> state: [`KICKOFFS.md`](KICKOFFS.md).

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
  ├─ S10 A-F CL health/recipe/speed/full claim     [envelope accepted; execution pending]
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
| S10 | Centralized-model numerical/architectural health, production recipe selection, final-architecture GH200 optimization, and bounded full clean/fusion claim | closed S08+S09 | six-stop A-F envelope accepted under O-122; ABC completion authority pending GPU-budget approval |
| S11+ | Not currently defined | future owner decision | pending; historical role proposals do not create scope, sequencing, full-run placement, or execution authority |

## 2. Persistent S00 contract

**Purpose.** Maintain one coherent technical context, implement related milestones
without repeated worker onboarding, keep a linear Git topology, and preserve
independent review.

**Default workflow.**

1. S00 reads current source, canonical documents, relevant handoffs, actual diffs,
   and raw artifacts.
2. S00 discusses the exact milestone plan, scientific boundaries, file ownership,
   and compute envelope with the owner.
3. After owner scope approval, S00 implements in the persistent active worktree.
   It may use a bounded planning/research subagent before implementation; it does
   not launch parallel production implementers by default.
4. S00 runs the smallest local checks, writes a phase-sized handoff package, and
   asks separately for any commit or compute authority not already explicit.
5. After an immutable implementation/evidence SHA exists, an independent reviewer
   subagent reviews the exact diff/artifacts without fixing code. Use a separate
   review worktree for high-risk data split, metric, scientific result, conflicting
   state, exact runtime reproduction, or owner request.
6. S00 remediates accepted findings linearly and obtains re-review before sealing
   the milestone.

**Authority boundary.** Persistent context does not authorize compute, commits,
merges, pushes, uploads, scientific protocol changes, extra cells/seeds, or
publication. The owner remains the freeze point for every material scientific and
execution decision in ORCHESTRA Section 8. O-107 only lets an initial exact O-009
smoke approval opt into a capped mechanical remediation loop. Each derived job is
diagnosed, frozen, and recorded before submission; possible model/data/precision/
recipe/metric/scientific or resource changes return to the owner.

**Reasoning.** O-096 records the owner's platform-maximum reasoning override for
the current persistent S00 and bounded pre-S08 planning/research or later
independent-review subagents. It does not authorize implementation chains or
compute.

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

## 6. S10 accepted six-stop envelope; execution pending

O-122 freezes the following scientific order. The detailed split, evaluator,
candidate and claim limits are in
[`handoffs/S10/HANDOFF.md`](handoffs/S10/HANDOFF.md).

| STOP | Question and exit | Explicit non-goal |
|---|---|---|
| A — Split/Metric | Materialize the deterministic train-only nested ownership split; prove no leakage; make the internal-subset evaluator exactly agree with the unchanged official evaluator on full-val parity fixtures; freeze artifact hashes. | No model training, checkpoint selection, or official-val observation for selection. |
| B — Observation-first | On a fixed `D_low` panel and the current graph, compare FP32 with accepted FP16+SECOND-FP32; decompose targets, loss normalization, sparse occupancy, norm placement and boundary gradients without optimizer updates; exit `LOCALIZED` or `INCONCLUSIVE`. | No clipping, normalization, head, loss, view-transform or architecture amendment. |
| C — Architecture/Initialization | Run the current local family against one coherent MIT-reference-derived package, all-scratch negative control and at most two B-triggered single-factor counterfactuals through `D_low -> D_mid`; compare joint and staged-L initialization; retain at most two graph/init families. | No recipe Cartesian product, LR/WD/EMA/augmentation/batch sweep, extra seed, `D_audit`, official val, or full run. |
| D — Recipe/Production Freeze | On the accepted graph/init finalists, select optimizer groups, LR/WD, schedule/warmup, clipping, EMA, sampling/CBGS, augmentation/GT-paste, batch/accumulation and exposure on `D_select`; bind `candidate_freeze.json`; open `D_audit` exactly once; then freeze the final graph/recipe or report `INCONCLUSIVE`. | No architecture reopening, audit-driven reselection, second `D_audit`, or official-val tuning. |
| E — Final-graph GH200 Optimization | Profile only the final accepted graph/recipe, then apply sustainable output-neutral changes and requalify numerical/metric equivalence. | No one-off profile interpreted as final bottleneck; no quality-changing optimization. |
| F — Full/Official Val/Close | Use the graph/recipe already frozen by D/E; run at least one single-seed full-train primary fusion claim, evaluate absolute clean capability and matched fusion contribution on sealed official val, record negative results, and close S10. | No reuse of `D_audit`, fallback checkpoint selection after official-val failure, or multi-seed campaign unless separately approved. |

The full primary claim belongs to STOP-F rather than being inferred from limited
rungs. `A1` requires at least the primary fusion full run; if staged `A2` survives,
its matched LiDAR donor plus fusion cost is part of the claim. A bounded extra
confirmation seed may be used only at the predeclared internal confirmation gate;
it is not a second full run. Conditional `BN1d`, `TransFusion`, or
LiDAR-conditioned `DepthLSS` consideration requires B/C evidence of a current
graph defect or material capability loss and must consume, not expand, the
counterfactual cap.

Current authority is documentation only. After the owner accepts the cumulative
ABC resource envelope, one completion decision may authorize persistent S00 to
implement, validate, commit, submit serial Slurm jobs, record evidence and obtain
independent review without per-job permission, provided every exact immutable
tuple is recorded before submission and stays within the frozen cells, seeds,
derivation rules and cumulative cap. This completion authority must not be
mislabelled as O-107. D/E/F remain separate future execution gates.

O-123 rejects the first B=1-based ABC estimate. O-124 approves the revised v1 ABC
completion envelope and starts continuous implementation/execution/review under
its cumulative caps. ABC execution uses physical B=4 as the minimum microbatch
for every scientific training rung and
reuse the accepted S09 B4 evidence; B=1 is limited to a tiny paired diagnostic
decomposition. STOP-C uses an explicit fixed-batch tail policy and matched sample
exposure. B=8/16 remain bounded candidates for the later STOP-D/E batch/throughput
gate after the graph is frozen; they are not silently added to ABC. Obvious
correctness errors are isolated with bounded steps before any rung is restarted;
scientifically weak but finite training is recorded as evidence rather than
iteratively debugged.

## 7. S11 and later

All roles, ordering and execution boundaries after S10 are pending. O-122 places
the primary full clean/fusion run inside STOP-F, but does not define S11 or
authorize that run. Historical S11-S15 descriptions remain context, not
authority. No Protocol A/B execution, attack, defense, paper upload, or
publication work is authorized.

## 8. Durable delivery and independent review

Each active milestone uses only the files it needs:

| File | Required content |
|---|---|
| `HANDOFF.md` | exact base/current SHA, files/semantics, tests, gates, hashes, failures, allowed/forbidden claims, residual decisions |
| `RUN_REQUEST.md` | exact immutable snapshot/config/data/cells/resources/command/output/stop conditions and explicit approval state before material compute |
| `RESULTS.md` | every job/status, raw path/checksum, requested and missing cells, metrics/performance, negative results, interpretation limits |
| `REVIEW.md` | independent findings first, exact diff/artifact basis, adversarial checks, verdict, residual risk |

The default review sequence is:

1. S00 produces an immutable implementation or evidence SHA.
2. A reviewer subagent audits that exact state and does not fix it.
3. If findings exist, S00 remediates in a new commit and requests re-review.
4. A separate worktree is used when the review is scientifically high-risk,
   state-conflicted, runtime-reproducing, or owner-requested.
5. S00 and the owner accept the verdict and update the canonical ledger.

Every review checks, as applicable: data leakage/ownership, coordinate and unit
semantics, batch isolation, config resolution, optimizer-step/exposure accounting,
precision/scaler/resume, metric/denominator correctness, failed/missing cells,
resource accounting, and shortcuts that could inflate capability or security
claims.
