# S10 independent review — STOP-A A4 and STOP-B

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
