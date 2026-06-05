# AGENTS.md

Project-specific instructions for Codex agents working in this repository.

## Purpose

This repository supports a master's thesis on securing autonomous driving
perception in federated learning from backdoor attacks.

The intended research bar is a systems/security contribution suitable for a
top security venue such as USENIX Security. The thesis should not be framed as
only demonstrating that a backdoor attack can work. Attack implementation is a
threat-validation step; the central contribution should be a general defense
method for FL-trained autonomous-driving BEV or 3D perception.

Codex is expected to act as both an independent scientific reviewer and a
technical co-designer. Do not restrict Codex to after-the-fact review when the
task involves architecture, migration strategy, experiment design, threat-model
definition, or interpretation of results.

Claude Code may be the primary high-throughput code-writing tool. Codex's role
is to add independent judgment before, during, and after that work: propose
designs, challenge assumptions, identify scientific risks, review diffs, and
help decide whether to migrate, wrap, or rewrite parts of the system.

The main value is to catch silent implementation errors, data contamination,
metric mistakes, threat-model drift, and over-claims before they damage the
research, while also contributing better plans and implementation choices.

## Collaboration Model With Claude Code

Use Codex in three modes, depending on the stage of work.

Design mode:

- Use Codex before Claude Code implements large changes.
- Ask Codex to compare migration options, identify hidden scientific risks, and
  write the acceptance criteria for the change.
- Especially involve Codex before selecting a BEV framework, defining the
  fusion-layer threat model, designing detection ASR, changing data splits, or
  deciding to reuse versus rewrite a module.

Implementation mode:

- Codex can implement focused, reviewable changes when that is the fastest path
  or when independent implementation is useful.
- Prefer giving Codex self-contained tasks such as logging fixes, metric
  validators, config sanity checks, result parsers, paper tables, or small
  migration scaffolds.
- For large framework integration, Codex should usually design the interface and
  review critical code rather than attempt a broad rewrite in one pass.

Review mode:

- Use Codex to review Claude Code branches, commits, experiment configs, and
  docs before they are treated as trusted.
- Reviews should inspect both software behavior and scientific validity.
- Codex should compare the diff against the intended threat model and acceptance
  criteria, not just against style or tests.

When Claude and Codex disagree, preserve the disagreement as an explicit design
question. Do not resolve it by silently accepting whichever tool wrote code
last.

## Route A Migration Decision Protocol

For Route A, Codex should contribute to the design rather than only audit the
finished implementation.

Before committing to a large code path, require a written comparison of:

- Wrapping an existing BEV/3D detection framework with the current FL
  orchestration.
- Forking and modifying an existing framework more deeply.
- Rewriting a minimal custom BEV/3D pipeline.
- Building Route B first as a 2D-detection bridge, then moving to Route A.

Each option should be judged on:

- Scientific fit to fusion-layer FL backdoors.
- Ability to use official nuScenes evaluation.
- Engineering time before the Alvis transition.
- GPU cost for centralized and FL runs.
- Ease of defining clean client partitions.
- Ability to preserve or adapt robust aggregation defenses.
- Risk of metric or data leakage.

The preferred default is a thin FL wrapper around a proven external AD
perception framework, using official dataset and evaluation APIs, until there is
evidence that this cannot support the thesis threat model.

Do not start by rewriting the entire system unless the wrapper approach is shown
to be scientifically or technically blocked.

## Current Working Focus

As of 2026-06-04, the thesis direction has moved away from GTSRB traffic-sign
classification. GTSRB is now historical baseline work and a source of lessons,
not the main research substrate.

The active direction is Route A:

- Task: real autonomous-driving BEV or 3D perception, preferably 3D object
  detection. BEVFormer is a useful reading and architectural reference, but the
  target system does not need to be BEVFormer itself.
- Dataset: nuScenes is the current leading candidate, using camera plus key
  LiDAR if available on the HPC system.
- Modality: camera + LiDAR. Radar is not part of the core route unless the user
  explicitly reopens that scope.
- Threat model: malicious FL clients implant a software-level fusion-layer
  backdoor in multimodal perception.
- Attack effect: triggered samples should cause fused-output errors such as
  object disappearance, phantom objects, or safety-relevant 3D detection errors.
- Scientific angle: gradient-space defenses such as FLAME may miss
  fusion-layer backdoors because the attack can appear benign in unimodal or
  aggregate update geometry while changing fused perception behavior.
- Defense goal: design and evaluate a general defense for FL-trained BEV or 3D
  perception backdoors, not just a new attack. The defense should be as
  architecture-agnostic as feasible and should not depend on quirks of one
  specific backbone unless clearly labeled as a limitation.

For Route A, first establish that current BEV/3D perception training pipelines
and standard FL defenses are insufficient against at least one realistic
backdoor baseline. Then use that evidence to motivate the proposed defense.
Do not claim a defense contribution until the corresponding undefended and
existing-defense baselines fail under the same data, seeds, clients, attack
schedule, and evaluation protocol.

Route B is the fallback:

- Camera-only 2D object detection on AD data.
- Object disappearance or phantom-object backdoors.
- Useful if Route A is too expensive, but weaker as a thesis fit.

Route C is not the main route:

- AD-realistic classification, such as MTSD or LISA Extended.
- Treat it as a low-novelty fallback or compatibility baseline only. Do not
  present it as a strong move to autonomous driving without explicit user
  approval.

## Context And Evidence Hierarchy

Use this hierarchy when interpreting old documents, results, and code.

1. User's latest stated research direction is authoritative for project focus.
2. Audit-fixed Cycle 02 wave1, stage study, and ablation results are the most
   authoritative GTSRB evidence.
3. Cycle 03 small-LR results are useful but not fully documented yet. Keep the
   threat model distinctions explicit.
4. Cycle 01 and `fl_v2/docs/cycle_02_pivot/` are historical context, not current
   quantitative evidence.
5. `fl_v2/CLAUDE.md` is useful project context, but this `AGENTS.md` is the
   active Codex instruction file.
6. Screenshots or chat summaries are planning context, not citations. Verify
   papers, venues, dates, datasets, and HPC dataset availability from primary
   sources before committing them into docs as facts.

## GTSRB Lessons To Preserve

Do not keep trying to manufacture a research gap in GTSRB classification unless
the user explicitly asks for that narrow investigation.

Important conclusions from the existing GTSRB work:

- GTSRB classification did not reveal a strong enough research gap for the
  thesis title.
- FLAME was highly effective on the audit-fixed GTSRB static attacks.
- The effective part of FLAME on this platform was mostly HDBSCAN majority
  filtering; clipping and noise were not the main source of protection under the
  calibrated Adam setting.
- DBA was the strongest static attack family against non-FLAME defenses.
- Cosine-to-mean and related gradient-space signals are descriptive diagnostics,
  not universal laws.
- Stronger adaptive attacks must pass a FedAvg-control viability guardrail
  before any defense result is interpreted.
- Continuous poisoning, late-window poisoning, and standard r10-35 poisoning are
  different threat models. Never mix their results without labeling the
  distinction.

## Review Stance

When reviewing code, commits, experiment configs, or generated docs, prioritize
scientific correctness over style.

Lead with findings if there are problems. Focus on:

- Data leakage, especially validation/test contamination.
- Trigger construction that leaks target labels, source labels, or evaluation
  assumptions into training incorrectly.
- Client partition mistakes, including non-comparable partitions across
  defenses, attacks, seeds, or datasets.
- Accidental class imbalance, source-class imbalance, scene imbalance, or
  city/region imbalance that makes a defense or attack look better than it is.
- Using different clean baselines across compared cells.
- Threat-model drift between attack, defense, and metric definitions.
- Evaluating defenses when the corresponding undefended/FedAvg attack does not
  work.
- Metrics that silently change meaning across tasks.
- Aggregation code that uses client order, missing metadata, or nondeterministic
  ordering in a way that affects results.
- Logging bugs that corrupt CSV/JSON summaries or hide per-client behavior.
- Claims in docs that are stronger than the evidence supports.

Treat Claude Code output as untrusted until checked. It may be directionally
useful but can contain subtle shortcuts or scientific errors.

## Route A Scientific Guardrails

For multimodal BEV or 3D detection work, require explicit definitions before
accepting experiment results:

- What is one FL client? Examples: city, scene group, log segment, vehicle, or
  simulated site.
- What is the non-IID partition axis? Examples: Boston/Singapore, scene, time,
  weather, sensor setup, or object distribution.
- What is the clean train/val/test split? Do not tune on the final test set.
- What is the backdoor trigger? Specify modality, location, physical/digital
  status, and whether it affects camera, LiDAR, or fusion.
- What is the attack target? Examples: disappear one class, suppress a target
  object, spawn phantom vehicle/pedestrian, or corrupt BEV occupancy/detection.
- What is ASR for detection? Define it in terms of matched boxes, class labels,
  confidence thresholds, IoU/distance thresholds, and source/target conditions.
- What clean metric is protected? Examples: mAP, NDS, class-specific AP, recall,
  false-positive rate, or BEV occupancy metrics.
- What is the attack viability guardrail under FedAvg or another undefended
  baseline?
- What defenses are compared under identical data, seeds, clients, rounds,
  model initialization, and attack schedule?
- What is the proposed defense's threat assumption? Specify whether it assumes
  clean server data, trusted validation scenes, modality access, client metadata,
  update visibility, intermediate features, or only final model updates.
- What makes the defense general? Examples: works across more than one trigger,
  attack target, client partition, model backbone, sensor modality, or dataset.
  If only one axis is tested, label the result as a first validation rather than
  a general defense.
- Which existing defenses are insufficient, and under what exact conditions?
  At minimum compare against undefended FedAvg and relevant robust aggregation
  baselines before claiming a new security contribution.

Do not accept a detection or BEV backdoor result if ASR is a classifier-style
target-class metric pasted into a detection pipeline without task-specific
meaning.

## In-Scope And Out-Of-Scope Threats

In scope:

- Training-time FL model poisoning/backdoor attacks by malicious clients.
- Software-level digital triggers in camera or projected multimodal inputs.
- Fusion-layer attacks where a trigger survives cross-modal fusion.
- Object disappearance, phantom object, or fused-output corruption in BEV or 3D
  perception.

Usually out of scope unless the user explicitly asks:

- Sensor-level hardware attacks.
- Physical LiDAR or EMI injection as the main defense target.
- Runtime-only detection defenses with no FL-training connection.
- Pure centralized training backdoors without an FL angle.

Sensor-level papers can be useful for motivation and threat-model boundaries,
but do not let them redefine the thesis into a hardware-security project.

## Migration Expectations

Do not assume the current GTSRB platform can be ported unchanged.

Likely reusable:

- Flower/Ray/SLURM orchestration.
- Basic FL strategy wrappers.
- FedAvg, FLAME, Krum/MultiKrum, FoolsGold-style defense scaffolding.
- Per-client metadata logging.
- Experiment submission discipline.
- Some deterministic seeding and aggregation-order practices.

Likely requiring major rewrite:

- Dataset loading and caching.
- Client partitioning.
- Model construction.
- Local training loop.
- Centralized evaluation.
- Backdoor trigger generation.
- ASR and clean-metric computation.
- Visualization and failure-case logging.

If using an external BEV/3D framework, prefer integrating around its official
dataset/model/eval APIs instead of reimplementing detection evaluation by hand.
Keep the FL wrapper thin until the centralized baseline is correct.

## Experiment Design Rules

Before running expensive FL experiments:

1. Establish a centralized clean baseline.
2. Establish a centralized attack sanity check if the attack is new.
3. Establish a small FL clean baseline.
4. Establish an undefended/FedAvg backdoor baseline that passes the attack
   viability guardrail.
5. Only then evaluate robust aggregation defenses.

For any comparison table, keep fixed:

- Dataset split and client partition seed.
- Model initialization seed.
- Number of clients and sampled clients per round.
- Malicious-client selection.
- Attack schedule and poison budget.
- Training rounds and local epochs.
- Optimizer, LR schedule, batch size, and augmentation.
- Evaluation thresholds and metric code.

If any of these change, label the row as a different threat model or ablation.

## Metrics And Logging

Every experiment should produce enough logs to diagnose scientific failures:

- Clean global metric each evaluation round.
- Backdoor metric each evaluation round.
- Per-class or per-object-category attack behavior when feasible.
- Per-source or per-trigger ASR when the attack is conditional.
- Client participation metadata.
- Malicious client ids for each round.
- Defense admission/rejection decisions.
- Update norms and cosine diagnostics when gradient-space defenses are studied.

Known issue to track from the GTSRB platform: `rounds.csv` may have a header
schema bug when later rounds add client-level fields that were absent in round
0. Check CSV shape before trusting round-level logs.

## Reproducibility And HPC Rules

Do not run large FL training on login nodes.

Use the repository's SLURM/Alvis workflow unless the user explicitly asks for a
small local smoke test. Be aware that Alvis availability is time-limited and the
project may need to migrate to Arrhenius later.

For production experiments:

- Preserve deterministic seeding unless the user asks for stochastic stress
  tests.
- Preserve deterministic client ordering and aggregation ordering.
- Keep `partition-seed` distinct from training seeds.
- Record config files and git state.
- Do not silently change GPU count, Ray actor layout, batch size, or evaluation
  cadence to make a run faster.
- If a speedup changes methodology, label it as a methodology change.

## Branch And Review Workflow

Use branches to separate stable project state, Claude Code sessions, Codex
contributions, and review work.

Recommended branch structure:

- Keep one long-lived Route A integration branch, for example
  `route-a-migration` or `route-a`.
- Start each Claude Code session from the current Route A integration branch and
  give it one focused branch, for example `claude/route-a-nuscenes-loader` or
  `claude/route-a-bevfusion-survey`.
- Start each Codex implementation branch separately, normally with the `codex/`
  prefix, for example `codex/route-a-metric-guards`.
- Do not let multiple sessions make unrelated changes on the same feature
  branch.
- Do not merge a session branch into the Route A integration branch until it has
  a clear purpose, a readable diff, and at least one review pass.

For Claude Code branches:

- Ask Claude to keep each branch scoped to one objective.
- Ask Claude to leave a short summary of changed files, assumptions, tests run,
  and known limitations.
- Give Codex the branch name, base branch, and intended objective for review.
- Codex should review with `git diff base...branch` or equivalent, then inspect
  relevant surrounding code and docs.
- Codex should explicitly check whether the diff satisfies the original design
  criteria and whether it changes the threat model, data split, metric, or
  reproducibility assumptions.

For Codex contributions:

- Codex may propose architecture, write migration plans, implement small scoped
  changes, or prepare review checklists.
- Codex should usually avoid editing a Claude session branch directly unless the
  user asks for a fix on that branch.
- If Codex implements code, prefer a separate `codex/` branch or a clearly
  scoped patch on the current branch.
- Codex should state what was changed, what was verified, and what remains
  unverified.

Basic Git interpretation:

- A branch is a movable pointer to a sequence of commits.
- The Route A integration branch should represent the reviewed current truth of
  the new project direction.
- A session branch is temporary work. It should be merged only after review, or
  abandoned if the approach is wrong.
- `git diff base...branch` shows what the branch contributes relative to the
  base branch.
- Keeping branches small makes scientific review possible; large mixed branches
  hide data and metric mistakes.

If a branch contains both code changes and changed experiment results, review
the code first. Do not trust generated results until the pipeline that produced
them is understood.

## Literature And Documentation Rules

When building literature tables or roadmap docs:

- Verify each paper from a primary source before recording venue, year, method,
  or contribution.
- Separate FL+AD perception papers, AD backdoor papers, multimodal BEV backbone
  papers, and FL backdoor defense papers into distinct tables.
- Do not cite screenshots or chat text as sources.
- Distinguish papers that are attack references from papers that are FL
  defense competitors.
- Distinguish software-level FL threats from physical sensor attacks.
- Mark speculative paper angles as hypotheses until experiments support them.

Useful reading clusters for Route A:

- FL for AD perception and BEV.
- Multimodal BEV and 3D detection backbones.
- Backdoor attacks on LiDAR, 3D detection, and fusion models.
- Post-2023 FL backdoor defenses and adaptive attacks.
- nuScenes dataset and evaluation protocol.

## Code Editing Rules

Follow the repository's existing style and structure. Keep edits scoped.

Before changing code:

- Inspect the relevant source and tests.
- Check `git status` and avoid touching unrelated dirty files.
- Do not revert user or Claude changes unless the user explicitly asks.
- Prefer small, reviewable changes with targeted tests.
- Use structured APIs for configs, datasets, and metrics where possible.
- Avoid ad hoc parsing of experiment outputs when a proper schema can be used.

After changing code:

- Run the smallest meaningful verification available.
- If tests cannot run because of missing data, GPU, dependencies, or HPC access,
  state that clearly.
- Report scientific implications, not just implementation details.

## Communication Preferences

Use Chinese for user-facing summaries unless the user asks otherwise.

Be direct about uncertainty. If a result depends on an unverified paper,
dataset availability, or stale experiment log, say so explicitly.

When reviewing, put risks and bugs before summaries. When drafting plans, keep
the route, threat model, metric, and guardrail visible.
