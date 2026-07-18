# S10 independent review — STOP-A/B closed; STOP-C0 remediation review

## STOP-C0 review state

```text
REVIEWER: independent /root/s10_c0_reviewer subagent; read-only
INITIAL_EVIDENCE_SHA: 908fea68e320501a6e353462a828f34107fa7ebf
EXECUTION_SOURCE_SHA: 89958be504d6abaef66810695402d2a09619794b
RAW_OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_c0_89958be_o131_a1
JOB: 492525 / FAILED 1:0 / 00:47:32
INITIAL_VERDICT: REMEDIATE
REMEDIATION_SHA: pending immutable commit
FINAL_VERDICT: pending targeted re-review
REVIEWER_EDITS: none
```

The reviewer found no P0/P1 and accepted source/config/split/weight/runtime
identity, physical-B4 exposure counts, optimizer/scaler semantics, true-unscaled
gradient and realized-update observations, token-complete evaluation and metrics,
the false-positive `to_bev=Identity` gate diagnosis, missing scratch/aggregate
classification, bounded profile/telemetry interpretation, allocation arithmetic
and full-claim restrictions. Initial disposition is `REMEDIATE` for the findings
below; it is not a C0 PASS and no replacement compute is authorized.

### C0 P2 — raw dropped-token identities are not exact

The executed v1 runner derived `dropped_tokens` from
`torch.Generator(seed).randperm`, but the production DataLoader first consumes a
`_base_seed` from the same generator and only then lets RandomSampler draw its
permutation. The named three raw tokens and diagnostics-fixture identity are
therefore invalid. The remainder count of three and same-order F/L construction
remain valid; metrics and numerical comparisons are unaffected. Required
remediation is an explicit permanent limitation plus future observation of
actual collated batch tokens, without rewriting raw output or retrying C0.

### C0 P2 — remediation changes artifact semantics

The post-job changes correct the impossible required prefix and short-horizon
assertion, so calling them generically “output-neutral” is too broad. They are
model/loss/gradient/update-neutral but change diagnostic labels and artifact
contents. Required remediation is schema v2 plus precise wording and an explicit
record that the new tests were not run on GH200.

### C0 P3 — raw even-length median was an upper median

The v1 implementation used `sorted(values)[len(values)//2]`. Standard medians
are `3.09258750525629e-4` for F and `3.0088933646399e-4` for L. The harm result
does not change because maximum LiDAR update/weight is independently below the
predeclared `1e-2` threshold. The v2 remediation uses `statistics.median`.

Residual risks already accepted for re-review are: the scratch control and
aggregate summary are absent; four initial overflow/loss windows confound the
earliest trajectory; this is a one-seed internal comparison where F includes A1
and independently initialized fusion modules; the trace is early and telemetry
is mixed-phase; and large-gradient causality remains unknown.

## 1. Review envelope

```text
REVIEWER: independent /root/s10_stop_a_reviewer subagent; read-only
EVIDENCE_SHA: 2a0153be88311ce1f8d502f2593218494d579014
SCIENCE_SOURCE_SHA: 7c01cc3f1e75691339f41f101794945748f03305
RESOURCE_SOURCE_SHA: ad93c89333b0a8f19abf138c8d6816e742b51e35
REVIEW_WORKTREE: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/review_worktrees/s10_stop_a_2a0153b
WORKTREE_STATE: detached / clean / exact evidence SHA
RAW_OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_a_gate_feasible_ad93c89333b0_o127_a1
JOB: 468404
REVIEWER_EDITS: none
INITIAL_VERDICT: REMEDIATE
REMEDIATION_SHA: b0478a298a0a3b5e538bedcca63e2541d71c2146
FINAL_VERDICT: PASS_WITH_RESIDUAL_RISK / no open P0-P3
```

The review covered the exact source/evidence diff, immutable data and config
identities, real split and ownership JSON/JSONL, all checksum manifests, evaluator
fixtures/results, focused tests, Slurm accounting, runner provenance, historical
negative jobs and scientific interpretation limits.

## 2. Initial findings

### P2 — conflicting active ABC aggregate ceilings

`RUN_REQUEST.md` exposed `28.1 GH200-h` in its estimate/history while the binding
root `AGENTS.md`, O-124 and the fail-closed gate retained `27` hours. This did not
affect Job `468404` or its scientific artifacts, but it could give subsequent B/C
automation two possible resource boundaries.

Required remediation: define one active aggregate ceiling. The remediation uses
the binding, stricter 27-hour O-124 limit. O-125's additional STOP-A contingency
remains local and non-transferable; its historical 28.1 arithmetic record is
labelled non-operative rather than silently erased.

### P3 — CBGS role-binding description exceeded implementation

The GTDB caller verifies manifest SHA, role and cache/ZIP identities. The CBGS
path only requires a dataset exposing sample tokens and hashes those caller-
provided tokens plus expanded indices; it does not independently verify an
expected manifest SHA or role. CBGS is disabled in STOP-A/B/C, so this does not
affect the accepted split or evaluator.

Required remediation: describe the current component as a CBGS identity seam and
make manifest/role/expected-token binding a tested STOP-D precondition before
CBGS can be enabled.

## 3. Independently reproduced evidence

- The isolated worktree was detached, clean and exactly at evidence SHA
  `2a0153be88311ce1f8d502f2593218494d579014` before and after review.
- The source snapshot was exact, clean and read-only; train/val cache pickle and
  sidecar, ZIP manifest and detection-config physical hashes matched.
- All three checksum layers passed; the 26-file output tree had zero writable
  paths.
- Independent reconstruction of 34,149 ownership rows returned 28,130 train and
  6,019 official-val samples, with zero cross-owner overlap for log, scene,
  sample, annotation, instance and raw path. Six-camera dependencies were
  complete, `D_low ⊂ D_mid ⊂ D_fit`, and official val was separate.
- Candidate ordinal was exactly one, with no seed or reroll. Base and nested each
  used one constant-zero feasibility solve and returned `FEASIBLE_FROZEN`;
  `candidate_freeze.json` was absent.
- `P-GT` and `P-MIX` were tolerance-zero `EXACT_PARITY` over filtered identities,
  forty class-distance metric-data payloads, validity masks and finite aggregate
  metrics. Empty predictions returned exact zero mAP/NDS with `official=false`
  and `proxy_only=true`.
- Slurm independently reported Job `468404` `COMPLETED 0:0`, 479 seconds, zero
  restarts, one GH200/four CPUs/32 GiB and batch GPU memory/utilization zero.
  Execution identity proved empty `CUDA_VISIBLE_DEVICES` and PyTorch CUDA
  unavailable/device-count zero.
- Jobs `463593`, `463649`, `467862` and `468295` remained honestly negative. The
  cumulative actual allocation arithmetic was correct at `1.413333...` GH200-h.

No leakage, reroll, metric denominator/filter/order drift, parity self-comparison,
or engineering-to-model-result overclaim was found.

## 4. Initial residual risks and interpretation limits

- The parity artifacts retain full result JSON and hashes of all metric-data
  arrays, but do not separately persist the full 40×101 arrays; exact equality
  therefore also relies on the immutable runtime assertion in the source.
- The bicycle-rack adversarial count of 22 is across trainval rather than an
  explicit official-val-only count. Both evaluator paths call the same devkit
  filtering implementation, so this is low residual risk rather than a gate
  failure.
- Evidence supports only the constrained proxy split/evaluator engineering
  contract. It does not support model capability, convergence, recipe quality,
  fusion gain or official-val performance.

## 5. Remediation state

The P2/P3 remediation is documentation-only and changes no split, evaluator,
data, model, source snapshot, output or compute. STOP-A remains open until a
targeted independent re-review accepts the remediation SHA.

## 6. Targeted re-review and final verdict

The same independent reviewer reused the isolated review worktree, checked that
it was detached, clean and exactly at remediation SHA
`b0478a298a0a3b5e538bedcca63e2541d71c2146`, and reviewed only the P2/P3 fixes and
their review record.

```text
P0: none open
P1: none open
P2: closed — unique active ABC aggregate is 27 elapsed GH200-hours
P3: closed — CBGS is an identity seam with a STOP-D manifest/role hard gate
FINAL_VERDICT: PASS_WITH_RESIDUAL_RISK
STOP_A_CLOSURE: GO
STOP_B_DEPENDENCY: satisfied; this review grants no compute authority
```

The reviewer confirmed the 27-hour active ceiling, 1.413334-hour actual use and
25.586666-hour arithmetic remainder are consistent; the historical 28.1 record
is explicitly non-operative. It also confirmed the CBGS/GTDB distinction and
that the remediation diff changes only four Markdown files, leaving source,
scripts, tests and all raw hashes unchanged.

Residual risk remains limited to the two interpretation notes in §4. STOP-A may
close as a constrained split/evaluator engineering PASS. It does not establish
model quality or authorize STOP-B compute.

## 7. STOP-B independent review

```text
REVIEWER: independent /root/s10_stop_a_reviewer subagent reused; read-only
IMPLEMENTATION_SHA: 43f157b3eca7ca72633358b5a2d2dbc4c4e4684b
EVIDENCE_SHA: 36ae78521f96963c610eb94af440216396a1c8b7
RAW_OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_diag_43f157b3eca7_o129_a1
JOB: 478250
REVIEWER_EDITS: none
OPEN_P0_P1_P2_P3: none / none / none / none
FINAL_VERDICT: PASS_WITH_RESIDUAL_RISK
STOP_B_DISPOSITION: calibrated baseline-instability FAIL; localization absent; owner rebaseline required
```

The reviewer independently read the pinned implementation diff, O-129 and
`RUN_REQUEST.md` §20-§22, evidence/docs commit, exact raw files and Slurm
accounting. It found no defect that could turn a runner/gate error into the
reported baseline-instability classification.

Independently reproduced checks:

- the snapshot is detached, clean and recursively read-only; source/tree,
  runner/observer/config and diagnostic-source hashes match the frozen tuple;
- all 41 focused tests passed; Job `478250` used one GH200, 8 CPUs, 64 GiB,
  `00:04:28`, zero restarts and terminal `FAILED 1:0` as declared;
- the exact physical Job-477892 panel was loaded read-only with matching
  file/content hashes, and W0 remained byte-identical across warm-up and parity;
- disabled-0/disabled-1 used the same `P_core` B4, seed `10000` and W0. RNG and
  model-state hashes match, while output/loss differ; all 459 gradients are
  finite and present, 434 fail the fixed allclose envelope, global relative-L2
  is `3.5323887774502536`, and max-absolute error is `2422412.736328125`;
- the attribution priority correctly chooses `baseline_instability` before
  considering disabled/enabled neutrality; evidence was durable before raise;
- both artifact manifests verify, no FP16/broad/term/aggregation/summary/refine
  artifacts exist, and B-REFINE is not triggered.

The residual risk is an interpretation boundary, not an open finding. Evidence
proves only that this fixed batch/current W0/current runtime is numerically
non-repeatable after one declared warm-up. It does not identify a kernel,
module or mechanism, and cannot support a spconv, GroupNorm, loss-normalization,
large-gradient-causality, convergence or recipe claim. STOP-B may be sealed as
the bounded negative result above, but any further diagnosis or STOP-C advance
requires an owner rebaseline and new exact authority.

## 8. STOP-B O-130 B-RAND independent review

```text
REVIEWER: independent /root/s10_stop_b_rand_reviewer subagent; read-only
IMPLEMENTATION_SHA: 0bf9c0ce4148bc82d977e0d66615f606144971b6
IMPLEMENTATION_TREE: 1852db34197c142714456f3fa07e999393dc1ba9
EVIDENCE_SHA: fdf223bb1fbe6656e9c543bb3beac52aab13e6f4
RAW_OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s10_stop_b_rand_0bf9c0c_o130_a1
JOB: 479667
REVIEWER_EDITS: none
P0/P1/P2: none / none / none
P3: two documentation findings; remediation below
INITIAL_VERDICT: PASS_WITH_RESIDUAL_RISK
```

The reviewer independently checked the pinned implementation/evidence diff,
exact §23-§25 tuple, detached clean read-only snapshot, raw JSON/JSONL,
checksum manifests and Slurm accounting. It found no defect that invalidates
the B-RAND integrity PASS or the predeclared `MIXED_INCONCLUSIVE`
classification.

Independently reproduced evidence:

- source/tree, split, panel, first `P_core` token vector, config file/resolved
  hashes, F-U W0 and runtime/resource identities match;
- Job `479667` completed `0:0` in `00:07:08` with zero restarts, one GH200,
  8 CPUs and 64 GiB; both checksum layers pass and the output has no writable
  paths;
- 43 tests passed; raw cardinality is exactly 33 physical-B4
  forward/backward runs and 24 reference comparisons in the approved mode/seed
  order, with no optimizer, update or evaluator;
- loss/output/gradients are finite, missing-gradient sets are stably empty,
  fixed-seed post-run RNG hashes agree and model state remains W0;
- independent recomputation of relative-L2/cosine/max-abs, prefix ordering,
  medians, the fourfold floor-`1e-8` ratios and two-of-three rule matches the
  summary;
- C-STR8 fixed-seed is exact, while changing its seed produces camera
  RNG-dependent variation on a graph containing active stochastic-depth
  modules; L-S075 fixed/varying variation is comparable and its largest
  gradient-direction changes occur in early SECOND prefixes; F-U fusion-only
  support is loss-only and does not qualify;
- the arithmetic is correct: Job `0.118889`, STOP-B `0.272222`, cumulative ABC
  `1.685556`, and active 27-hour remainder `25.314444`.

### P3 — canonical lower-level status lag

The active top-level status and O-130 ledger were correct, but the lower graph
and unresolved-decision rows in `ORCHESTRA.md`, plus a lower STOP-B paragraph in
`SESSIONS.md`, still stopped at O-128/O-129 parity failure and owner rebaseline.
This could mislead a later reader about the latest evidence, although it did not
affect execution or raw interpretation.

Remediation: update those lower canonical records to Job `479667` integrity
PASS, route-level decomposition, large-gradient `INCONCLUSIVE`, no executable
compute and owner-gated STOP-C.

### P3 — stochastic-depth causal wording

`HANDOFF.md` and `RESULTS.md` described changing seeds as directly activating or
observing stochastic depth. Raw evidence directly proves only that the train-mode
camera graph contains twelve `StochasticDepth` modules and that camera-only
outputs/gradients vary with RNG seed. It does not retain masks or run an
SD-disabled counterfactual.

Remediation: describe the result as camera RNG-dependent variation on a graph
containing active stochastic-depth modules, consistent with intended
stochasticity but not a causal mechanism proof.

The remediation is documentation-only. It changes no source, snapshot, config,
data, panel, model, raw artifact, checksum, metric, classification or compute.
Targeted re-review of the exact remediation SHA was required before final
STOP-B disposition and is recorded in §9.

Residual risks retained by the initial verdict:

- one B4 token vector and four reference comparisons per group are bounded
  operational evidence, not a population estimate;
- no stochastic-depth mask/disable counterfactual exists;
- the classification unit test covers a unique camera-label path rather than
  the observed multi-label mixed path, although the reviewer independently
  recomputed the actual raw classification;
- no evidence explains the large true unscaled LiDAR-gradient mechanism or
  proves a specific sparse kernel/module/normalization cause.

## 9. O-130 targeted remediation re-review and final verdict

The same independent reviewer checked exact remediation SHA
`02ba3b44202092894f2c1c3e7ee53bb56ba92a1d` read-only. The diff changes only
five Markdown files. Implementation, configs, scripts, tests, the §24 tuple,
snapshot `0bf9c0c`, raw outputs, checksums, classification and compute accounting
are unchanged.

```text
P0: none open
P1: none open
P2: none open
P3: none open
NEW_FINDINGS: none
FINAL_VERDICT: PASS_WITH_RESIDUAL_RISK
STOP_B_CLOSURE: GO
```

The canonical lower graph/decision records now reflect Job `479667`, bounded
route-level decomposition, large-gradient `INCONCLUSIVE`, no executable
STOP-B compute and owner-gated STOP-C. The camera wording now states only
RNG-dependent variation on a graph containing active stochastic-depth modules
and explicitly records the absent mask capture/SD-disabled counterfactual.

STOP-B's final accepted disposition is:

```text
STOP-B: CLOSED / INCONCLUSIVE
B-RAND: integrity PASS
ACCEPTED: bounded route-level repeatability decomposition
NOT RESOLVED: large true LiDAR-gradient causality
FURTHER STOP-B COMPUTE: none
MODEL/RECIPE CHANGE: not authorized
STOP-C_AT_B_CLOSURE: separately owner-gated; O-131 later activates only C0
```

Residual risk remains exactly the bounded evidence limitations in §8. None is
an open P0-P3 or a reason to rerun Job `479667`.
