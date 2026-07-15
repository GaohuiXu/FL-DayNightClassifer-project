# USENIX Security '27 Orchestra — milestone contracts

> **Status (2026-07-15).** S07 is closed at clean engineering anchor
> `2a584053e6f6a3860b6f812681dc8d7342ca52ad`. S08 implementation/remediation is
> sealed at `103c7389a47938b1f9dd0cba60251df6dce9e5bb` with independent R2
> `PASS_WITH_RESIDUAL_RISK`. Under O-109, exact Q1 Job `431013` and Q2 Job
> `435151` are terminal and checksum-verified, consuming only `00:07:58` GPU
> elapsed. Primary evidence rejects full sparse FP16 as the unified F-capable
> route and supports global FP16 with SECOND/spconv FP32; L-P020 and F-CBGS
> compatibility cells also passed. Independent R3 reviewed exact evidence SHA
> `c0ef86235ead753fee3b790b19d40f82f875ec59` with
> `PASS_WITH_RESIDUAL_RISK` and no P0-P2 findings. O-110 accepts seal `d31adea`,
> freezes the recommended precision policy, and closes S08 PASS; closing commit
> `28f79802c0868afa6290d74ae6aeb9d23c7d088f` is fast-forward integrated. The S09
> reading gate is complete. O-111 accepts the four-stop engineering-readiness
> direction and defers branch/training-recipe selection to S10. O-112 STOP-1 Job
> `441191` completed `0:0` in `00:03:06`; exact train/val cache identities passed
> in-job and S00 post-job checks. Independent review passed raw evidence but
> returned `REMEDIATE` for P2/P3 durable provenance/status wording. Bounded
> re-review of documentation-only remediation SHA `5252a59` closed every finding
> and returned `PASS_WITH_RESIDUAL_RISK`. O-113 owner-accepts/closes STOP-1 and
> opens STOP-2 detailed planning. O-114 approves the exact STOP-2 implementation,
> local validation, linear commits, and independent review. Candidate `37aef4d`
> has independent `PASS_WITH_RESIDUAL_RISK` with no open P0-P2; exact snapshot,
> selectors, wrapper and fresh output are frozen. Closure review of request
> remediation `cad7262` found no open P0-P3. O-115 now approves that exact tuple
> and its recorded O-107 mechanical boundary. Initial Job `441293` completed
> `0:0` in `00:01:04` with 44/44 tests passing; no replacement was used and
> independent evidence review is pending.
>
> `Sxx` now names a durable evidence milestone, not necessarily a new task,
> worker, branch, or worktree. Under O-094, persistent S00 normally performs
> tightly connected implementation in one linear worktree and uses an independent
> reviewer subagent or, when risk requires, a separate review worktree.
>
> Canonical objective/decisions: [`ORCHESTRA.md`](ORCHESTRA.md). Active envelopes:
> [`KICKOFFS.md`](KICKOFFS.md).

## 1. Active graph and status

```text
28f7980 accepted S08 close / S09 base
  │
  ├─ S08 model/recipe audit → owner decisions → precision qualification
  │      └─ independent review only after exact implementation/evidence SHA
  │
  ├─ S09 full-pipeline performance/readiness
  │      └─ independent review of exact profiling/evidence SHA
  │
  ├─ S10 centralized branch/recipe ablation       [pending S08+S09]
  ├─ S11 CL capability/freeze                     [pending S10]
  ├─ S12 clean Protocol-A/B split/adaptation      [deferred pending CL freeze]
  ├─ S13 new threat model/attack                  [blocked pending clean adaptation]
  ├─ S14 new defense                              [blocked pending viable attack]
  └─ S15 paper/artifact                           [rolling, evidence-bound]
```

| ID | Milestone | Depends on | Status / output |
|---|---|---|---|
| S00 | Persistent owner/implementation/review coordination | — | active; sole canonical-doc writer and default implementation context |
| S01 | Stored-ZIP nuScenes data foundation | closed | reviewed PASS within recorded scope; `t1.v1` production use forbidden |
| S02-S05 | Loss/target, camera, SECOND, CenterHead/decode modules | closed | reviewed module contracts integrated into clean anchor |
| S06 | C/L/F resolved runtime/checkpoint/eval contract | closed | reviewed bounded contract integrated into clean anchor |
| S07 | Legacy cleanup plus clean completion | S01-S06 | **closed**; S07-C static review PASS and S07-B bounded FP32/FedAvg/loader gate PASS; no science/precision freeze |
| S08 | Model/recipe audit, then precision qualification | S07 | **closed PASS under O-110** at accepted seal `d31adea`; Jobs `431013`/`435151`, `00:07:58` total; R3 no P0-P2 |
| S09 | Full-pipeline engineering performance/readiness | accepted S08 policy | STOP-1 closed; STOP-2 Job `441293` technical PASS (44/44), no replacement; independent evidence review pending |
| S10 | Centralized branch/recipe ablation | S08+S09 | pending redefinition; no cells/gates frozen |
| S11 | Full CL capability and architecture freeze | S10 | pending redefinition; no seeds/matrix approved |
| S12 | Protocol-A/B split and clean adaptation contract | CL freeze + fresh owner review | deferred; old proposal is historical evidence only |
| S13 | Clean adaptation completion, then new attack | S12 clean PASS + new threat model | blocked; legacy T5 import forbidden |
| S14 | New defense | independently viable undefended S13 attack | blocked; legacy defenses are not active baselines |
| S15 | Paper/artifact | accepted evidence | planned/rolling; no upload/submission permission |

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

**Gate categories to freeze before execution.** Numerical values remain pending,
but the owner must approve thresholds for:

- finite accepted-step ratio and loss trend;
- peak memory/headroom;
- data-wait share and p95/p50 stability;
- samples/s or epoch wall time;
- zero unaccounted scheduler/EMA/exposure drift;
- stop-on-OOM, repeated nonfinite windows, loader failure, or unacceptable runtime.

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
4. `STOP-4 G1000/CLOSE`: only after reviewed G100 acceptance, execute a fresh
   1000-successful-step single-GH200 gate, independently review the complete S09
   evidence, and prepare a close-ready linear state. It does not resume the
   mid-epoch G100 state or imply DDP.

**Compute.** STOP-1/3/4 material jobs require their own exact stop approval. The
STOP-2 focused smoke may use O-009/O-107 only when that stop approval explicitly
opts in. Unused quota is not retry authority. O-112 approves only the one STOP-1
cache-materialization submission after its exact tuple is frozen; no later S09 job
is approved.

## 6. S10-S12 redefinition boundaries

### S10 — centralized branch and recipe ablation

Pending S08/S09. Candidate topics include `C-STR8`, `L-P020` versus `L-S075`,
`F-U` versus `F-CBGS`, and initialization. The actual cells, matched exposure,
precision, checkpoint selection, numerical gates, and compute budget are not
frozen. S10 will select a centralized recipe; it will not yet claim multi-seed
capability or define Protocol-B `W_base`.

### S11 — full CL capability and freeze

Pending a reviewed S10 selection. S11 will run only the owner-approved seeds for
the selected C/L/F recipes, produce official mAP/NDS/per-class/TP-error/slice and
performance evidence, and freeze the architecture/config/checkpoint schema.
Full-train capability checkpoints remain distinct from a Protocol-B initializer.

### S12 — clean Protocol-A/B contract and adaptation

Deferred until CL freeze and a fresh owner review. S12 will define and hash
scene/log-disjoint `D_base`, `D_tail`, clients, evaluation ownership, clean
controls, update scope, and utility/forgetting metrics. The old S12 proposal and
legacy security assumptions are not active inputs. No attack is included in the
clean adaptation gate.

## 7. S13-S15 boundaries

- **S13:** first establish accepted clean Protocol-B adaptation and the separate
  Protocol-A control. A new attack requires a later owner-approved threat model
  and clean implementation; legacy T5/T6/T7 cannot be imported.
- **S14:** begins only after an independently reviewed viable undefended S13
  attack. New defenses target measured mechanisms; legacy defense code/oracles are
  historical parity/negative evidence only.
- **S15:** paper/artifact work consumes checksummed accepted evidence. Mini, smoke,
  failed, stale, or mismatched cells never become scientific numbers. Upload,
  registration, submission, and publication require exact owner authority.

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
