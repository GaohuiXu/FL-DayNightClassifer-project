# S12 HANDOFF — Protocol-B contract, Protocol-A control, tail split, threat model, novelty audit, and paper skeleton

> **交付状态：PROPOSAL ONLY / NOT APPROVED / NOT FROZEN。** 这是只读证据审计与 owner 决策提案，
> 不是 Protocol-B split manifest，不是 client assignment，不是实验批准，也不是 scientific/integration PASS。
> 任何下游采用前必须经过独立 **S12-R** review，并由 owner 对 material scientific decisions 逐项批准。

## 0. Session identity, scope, and preflight

| Field | Exact value |
|---|---|
| `SESSION_ID` | `S12` |
| `BASE_SHA` / observed `HEAD` | `f262f6bea037580065a8505008773c04fdd259f5` |
| `SOURCE_BRANCH` | `v3-ad-perception` |
| Expected / observed ref mode | `detached@f262f6bea037580065a8505008773c04fdd259f5`; `git branch --show-current` was empty |
| Worktree root | `/home/gaohui/.codex/worktrees/aada/fl_weather_project` |
| Initial worktree status | clean; `git status --short` produced no output |
| File ownership | only `fl_v3/usenix27_orchestra/handoffs/S12/**` |
| Upstream handoffs | none |
| Approved compute | none |
| Material compute actually run | none |
| External actions | none: no Slurm, training, upload, registration, publication, commit, merge, push, or PR |
| Review required | independent `S12-R`, based on a durable worker SHA if/when owner separately authorizes a local handoff commit |

The required preflight was run before any edit:

```text
$ git rev-parse --show-toplevel
/home/gaohui/.codex/worktrees/aada/fl_weather_project
$ git rev-parse HEAD
f262f6bea037580065a8505008773c04fdd259f5
$ git branch --show-current
<empty>
$ git status --short
<empty>
```

No mismatch was found. No branch/worktree operation was performed.

### Binding decisions preserved

- **Protocol B** is the primary security protocol: a vendor trains `W_base` only on common `D_base`, then
  regional/fleet data-silo clients federatively adapt that checkpoint on disjoint `D_tail` data.
- **Protocol A** is a clean optimization/control setting: nuScenes-scratch federated training from one
  identical declared initialization. It is not a source of Protocol-B checkpoints or security claims.
- These roles are not reopened or mixed here. Every other split, client, update-scope, threat, metric,
  claim, resource, paper, and artifact detail below is explicitly **unapproved**.

## 1. Executive verdict

Protocol B is scientifically defensible only if the project first builds a new immutable ownership
manifest and a new provenance/evaluation contract. The current full-train `log_group` partition is a useful
engineering substrate, but it does **not** express `D_base`/`D_tail`/held-out-tail ownership, raw-asset and
sweep closure, per-client held-out utility, Protocol IDs, `W_base` lineage, update-scope hashes, false-trigger
rates, defense FPR, or secure-aggregation mode. It must not be relabeled as Protocol B.

The recommended high-level contract is:

1. fit every tail rule using official **training metadata only** and before any attack/ASR observation;
2. split indivisible connected components whose primary boundary is the complete nuScenes **log**, expanded
   by every referenced sensor/sweep/raw-file dependency;
3. identity-hash and seal train-origin evaluation components before tail statistics, fit the rule on the
   remaining design pool, then apply the frozen rule once to the blind reserve and official validation;
   official validation/test never train or select a model;
4. represent clients as synthetic proxies for regional/fleet **data silos**, not individual cars;
5. train a new `W_base` on `D_base` only, then initialize every Protocol-B control from the identical checksum;
6. choose one exact symmetric adaptation allowlist using clean development evidence only, then hash it before
   any attack run;
7. compare fixed-denominator ASR with clean-input target coverage, benign false-trigger, tail/common utility,
   honest-client defense FPR, per-client dispersion, and complete compute/communication accounting;
8. state either a visible-update server or secure aggregation. Current FLAME/FoolsGold/Krum/median/server-side
   norm-clipping implementations require individual updates and are not compatible with standard secure-sum;
9. make novelty rest on a demonstrated systems-security mechanism and utility-preserving defense in this
   Protocol-B setting—not on an unsupported “first federated/multimodal/3D/backdoor” claim.

Materialization remains blocked by owner decisions, independent S12-R, reviewed S01 ZIP/coverage evidence,
and the eventual S07 architecture/S11 metric contracts. No split or client assignment was generated here.

## 2. Evidence scope and method

### 2.1 Repository evidence read at the pinned base

Read completely:

- root `AGENTS.md`;
- all current canonical Orchestra files: `ORCHESTRA.md`, `SESSIONS.md`, and `KICKOFFS.md`;
- `fl_v3/docs/env.md` and `fl_v3/docs/roadmap/INDEX.md`;
- the relevant data schema, loader, cache, path, partition, training-task, sampling, attack-routing,
  aggregation/defense, ASR, detection-evaluation, report, and provenance source/tests;
- read-only legacy evidence under `fl_v3/collab/`, especially T1/T3/T4/T5 specs and reviews,
  `fl_baseline/phase3_fl_baseline.md`, findings, and historical partition artifacts;
- the active cycle-04 roadmap and historical decisions only to identify superseded assumptions.

Key source locations are cited in Sections 8–9. Historical `fl_v2` was not modified and was not treated as
scientific evidence; defense parity remains implementation equivalence only.

### 2.2 Literature search protocol

Search date: **2026-07-10**. The audit screened primary paper/venue pages and full papers across these axes:

```text
(federated OR decentralized) AND (autonomous driving OR vehicle perception)
(federated) AND (3D object detection OR BEV OR camera LiDAR OR multimodal perception)
(long-tail OR rare class OR tail adaptation) AND (federated OR fine-tuning)
(backdoor OR poisoning) AND (federated learning)
(backdoor OR adversarial) AND (camera LiDAR OR sensor fusion OR 3D detection)
(layer OR module OR prompt) AND (federated backdoor OR poisoning)
(secure aggregation) AND (poisoning OR robust aggregation OR backdoor defense)
```

Primary sources were preferred from USENIX, IJCAI, ICLR/OpenReview, CVF, PMLR, ACM/IEEE DOI/publisher pages,
and arXiv for current preprints. The screen included title/abstract/claim comparison and, for the closest work,
method/threat-model inspection. This is a systematic **screen**, not a certified exhaustive systematic review:
forward/backward citation chasing, DBLP/Scopus/Web of Science deduplication, patents, non-English literature,
and papers released after the search date remain gaps. Therefore this document authorizes no “first” claim.

### 2.3 Official venue/rules verification

Verified on **2026-07-10** against:

- [USENIX Security '27 Preliminary CFP](https://www.usenix.org/conference/usenixsecurity27/call-for-papers),
  including the [official CFP PDF](https://www.usenix.org/sites/default/files/sec27_cfp-070826.pdf);
- [Submitting ML Work to USENIX Security](https://www.usenix.org/conference/usenixsecurity27/submitting-ml-work-usenix-security);
- [USENIX Authorship Policy](https://www.usenix.org/conferences/authorship-policy) and general submission policy;
- [Cycle-1 HotCRP](https://sec27cycle1.usenix.hotcrp.com/), which was present but closed at search time;
- the SEC'26 ethics page only as the example framework linked by SEC'27, not as overriding SEC'27 policy.

The CFP still said **Preliminary**; PC, SEC'27 artifact-call/AEC details, final-paper instructions, and a
venue-specific statistics checklist were not yet published. These must be rechecked rather than inferred from
SEC'26.

## 3. Proposed data and ownership contract — unapproved

Everything in this section is a recommendation for owner/S00 decision. Keywords `MUST`, `SHOULD`, and `FAIL`
describe the proposed future contract, not current repository behavior.

### 3.1 Names and allowed flows

Let `U_train` be only the official nuScenes training split. First freeze the architecture, sensor dependency
scope, update-key allowlist, and primary base/adaptation recipes. The future builder may then construct the
all-split dependency graph; before exposing any new Protocol-B component/tail/support statistics, it assigns
complete training components by a precommitted identity-only hash into a rule-development pool `U_design` and
a sealed blind evaluation pool `H_eval_blind`. A future immutable manifest SHOULD then express:

| Symbol | Role | May influence |
|---|---|---|
| `D_base_fit` | vendor common-data fit set | gradients for `W_base` only |
| `D_base_dev` | vendor-only, log-disjoint clean development | predeclared `W_base` monitoring/checkpoint selection inside a frozen architecture/recipe; never a final claim set |
| `D_tail_adapt,k` | adaptation data owned by client silo `k` | local Protocol-B updates for client `k` only |
| `H_eval_blind` | identity-hash-selected, sealed train-origin components | no tail-rule/support inspection until the rule and support floor are frozen |
| `E_tail_train` | the frozen tail rule applied once to unsealed `H_eval_blind` | final tail evaluation only; no fitting, selection, poisoning, early stopping, or fallback reselection |
| `E_val_all` | complete official validation | final overall clean/security evaluation only |
| `E_val_tail` / `E_val_common` | fixed predicate applied blindly to official validation | complementary held-out tail/common slices; not criterion selection |
| official test | untouched | at most one owner-approved final evaluation; never development |

`D_base = D_base_fit ∪ D_base_dev` denotes vendor ownership, but the checkpoint gradient path MUST record that
only `D_base_fit` trained `W_base`. `D_tail = ⋃k D_tail_adapt,k` denotes only the client adaptation pool.
`D_base` and `D_tail` are allocated only from `U_design`. `E_tail_train` is a separate blind reserve and never
enters a client's training shard. Any held-out strata used for client-utility dispersion MUST come from that
reserve (or the blindly derived official-validation slice), never from `D_tail` itself.

The exact number/fractions of components are intentionally absent. They require a small metadata-only
feasibility audit and owner approval; a trainval scan was not authorized in S12.

### 3.2 Indivisible ownership unit and dependency closure

**Recommendation:** use a connected component whose initial edge is “same `log_token`,” not a sample or
annotation. Expand the component through every data dependency used by the resolved model/config. If any raw
asset appears in two nominal logs, merge those logs into one component before assignment. Assign a component
exactly once to base, one client, development, or held-out evaluation.

Construct the isolation graph against metadata from **all official splits**, although allocation candidates come
only from `U_train`. If a nominal training component shares a log, scene dependency, sample-data/raw asset, or
confirmed content identity with official validation/test, quarantine the entire training component; do not
absorb validation/test into training and do not cut the component to save its training frames.

For every component, the manifest MUST enumerate and validate at least:

- `log_token` and every contained `scene_token`;
- every keyframe `sample_token` and timestamp;
- every referenced keyframe `sample_data_token` for each enabled camera/LiDAR channel;
- the complete configured LiDAR-sweep `prev` closure, including sweep `sample_data_token`, source
  sample/scene/log, timestamp, and transform-to-keyframe;
- normalized raw identity: archive ID + normalized member name for ZIP mode, or canonical relative path for
  directory mode; member size/CRC where available;
- content identity for duplicate detection (stream SHA-256, or size/CRC candidate detection followed by
  SHA-256 confirmation), without extracting/duplicating the dataset;
- annotations/instances used to derive criteria or metrics and their owning sample;
- exact sensor channels, `n_sweeps`, loader/backend schema version, source metadata/archive coverage hashes,
  tail-rule version, random seed/solver version, and canonical manifest hash.

The validator MUST hard-fail on every pairwise crossing among base, tail clients, dev, and evaluation at the
levels of log, scene, sample, sample-data token, sweep dependency, normalized raw path/member, and confirmed
raw content. It MUST also fail on missing, duplicate, or extra expected entities. “Whole logs probably contain
their sweeps” is not evidence; closure is checked from metadata.

Public maps, class definitions, calibration schema, and the frozen architecture may be common resources.
Sharing a sensor raw sample, its sweep window, its label-derived tail decision, or a checkpoint trained on it
is not allowed.

### 3.3 Tail criterion and allocation

**Recommended primary criterion family:** a semantically declared, context-conditioned rarity rule fitted
only from `U_design` metadata after the blind-holdout commitment. `U_train` outside `U_design` is used only for
dependency-graph identity/closure and the identity-only holdout assignment. Candidate rule features are location,
train-only scene weather/time descriptors,
class/context support, target distance/visibility/point support, and their predeclared combinations. Because
ownership is log-atomic, scene/sample evidence is summarized to the log/component by a declared aggregation
rule; it never licenses extracting just the convenient frames from a log.

The process SHOULD be:

1. before revealing exact tail/split statistics, freeze the architecture, exact update-key allowlist, primary
   `W_base` and adaptation recipes/exposure, candidate metadata fields, statistical unit (log/scene, never
   frame-IID), binning/quantile procedure, and target/evaluation support floors;
2. after forming the dependency components, precommit the count/fraction and seed/salt for an identity-only
   hash split; seal `H_eval_blind` (S00 may hold the salt/membership commitment) and expose only `U_design` to
   tail-rule development;
3. use only `U_design` feature/count statistics to select one exact rule whose remaining components support
   `D_base`, multiple client silos, and the declared development procedure;
4. allocate whole `U_design` dependency components by a deterministic constrained solver, with exact objective,
   tie-breakers, seed, and negative/failed feasibility attempts recorded;
5. freeze and hash rule + allocation + raw closure and the already predeclared minimum evaluation support;
6. only then unseal `H_eval_blind` and apply the rule once to form `E_tail_train`; independently apply it to
   official validation to form `E_val_tail/common`.

The primary rule MUST NOT use model loss/error, embeddings, clean-detection eligibility, trigger-placement
success, ASR, defense response, or an attack-favorable client. Model-error mining may be studied only as a
clearly labeled secondary adaptive setting after the primary contract is frozen.

`D_base_fit` MUST retain enough instances of the attack target class and broad operating conditions to learn a
competent detector; the security result cannot rely on `W_base` simply never learning the target. Exact support
and clean-detection gates await S11/owner approval. If either blindly derived evaluation slice is underpowered,
do not relax the rule, change the holdout, or select a better reserve: narrow/abandon the claim. A tail set chosen
with knowledge of its own feature/support statistics may appear only as `transductive-selected/secondary`, never
as the primary independent `E_tail_train`.

**Rejected primary alternatives:** sample-IID splitting; annotation-level ownership; selecting rare cases by
`W_base` errors; assigning all tail examples to training clients; defining tail after ASR; splitting scenes from
the same log without a proved dependency closure; reusing the historical 25-client full-train partition.

### 3.4 Client unit and assignment

nuScenes does not document actual commercial fleet ownership. Therefore each client SHOULD be described as a
**synthetic regional/fleet data-silo proxy**, built from complete capture components. It MUST NOT be described
as one car, an on-board full-detector trainer, or a real company/fleet.

Recommended properties:

- client count `K` is derived after reserving base/dev/evaluation support; it is not inherited as 25 or 50;
- one component belongs to one client; a client may own multiple components, preferably location-coherent;
- client identities/roster and assignment hashes are frozen before any attack;
- distributions retain real train-only heterogeneity (region, weather/time, class/context) and report volume,
  target support, missing classes, and tail support per client;
- held-out client-stratum evaluation is reserved where support permits; otherwise performance dispersion is
  explicitly unavailable rather than replaced by in-sample loss; the client↔held-out-stratum mapping rule is
  frozen before `H_eval_blind` is unsealed, not chosen from performance;
- an attacker/client is selected by a public deterministic rule or an exhaustive predeclared sweep, never the
  client with best observed ASR.

### 3.5 Participation and communication model

**Recommendation:** use authenticated synchronous cross-silo FL with full participation as the primary causal
mechanism cell. With a small stable silo cohort, this removes participant-sampling variance and permits exact
matched exposure. Add one deterministic partial-participation schedule only as a separately budgeted realism
check if `K`, tail support, and compute permit it. This recommendation is unapproved.

Every run MUST bind `K`, clients sampled per round, minimum cohort, schedule/seed/hash, local examples and
optimizer steps, aggregation weights, dropouts/retries, and whether the malicious client actually participated.
For partial participation, report both per-participating-malicious-round and unconditional attack outcomes.
Do not silently transplant historical D10 full-participation or `rho=0.2`.

The communication model is a central coordinator exchanging only the exact trainable parameter allowlist (or
its declared compressed form), over authenticated encrypted transport. Data remain in each synthetic silo.
Transport encryption is not secure aggregation and must not be described as such.

### 3.6 Fine-tuning/update scope

**Recommended primary candidate:** freeze heavy camera and LiDAR encoders; update symmetric modality adapters
or projection/view-transform blocks plus fusion, BEV neck, and detection head. The exact parameter names cannot
be set until S07 freezes the architecture. Owner must choose and hash one primary candidate, together with the
primary `W_base`/adaptation recipes and exposure rules, **before any exact Protocol-B component, split, tail, or
support statistics are revealed**—not merely before materialization/S13. The choice may use S07 interface/
parameter/communication evidence and disclosed historical full-train engineering evidence, but not the new
`U_design`, `H_eval_blind`, `D_tail`, attack, or defense outcomes.

The selected scope MUST be an exact ordered parameter-key allowlist with tensor shapes, total bytes, and hash.
It MUST be identical for pooled oracle, local-only, clean FL, attacked FL, and defended FL. Optimizer state and
buffers included/excluded must also be explicit. A full-model scope is the principal alternative/generalization
cell; PEFT/prompt/adapters are a distinct protocol because Fed3D and federated prompt literature already make
that design scientifically non-neutral.

S13's clean phase then tests whether that frozen scope meets predeclared pooled-tail and clean-FL utility gates.
If it fails, security work stops. The owner may abandon/narrow Protocol B or explicitly amend and re-freeze a
full-model/other scope as a new protocol version before rerunning all clean controls; it cannot silently promote
the best post-tail scope. A security paper must not call a defense successful when the scope itself prevents
benign adaptation.

## 4. Checkpoint and control contract — unapproved

### 4.1 `W_base`

`W_base` MUST be retrained after the architecture and manifest are frozen, using only `D_base_fit`; selection
may use only `D_base_dev` through the single checkpoint/early-stop rule frozen before Protocol-B statistics were
revealed. It cannot choose a new architecture, optimizer recipe, or exposure. The full-train CL capability
checkpoint and every historical full-train FL checkpoint have seen data that may become `D_tail` and are invalid
Protocol-B initializers.

Record architecture SHA, public initialization provenance, `D_base` manifest hash, train/dev ownership hashes,
precision, seed, resolved config, exact optimizer/exposure, checkpoint SHA-256, and clean overall/common/tail/
target capability. Every matched Protocol-B cell begins from the identical `W_base` checksum for its seed.

### 4.2 Required Protocol-B controls

| Control | Purpose | Matching constraints |
|---|---|---|
| `B0 W_base` | no-adaptation anchor | exact frozen checkpoint |
| `B1 pooled-tail oracle` | upper bound for learnability absent client drift | pooled `D_tail_adapt`; same trainable scope, precision, optimizer family, example/step exposure; aggregation overhead separately zero |
| `B2 local-only,k` | benefit without collaboration | each client starts at `W_base`; same local recipe/exposure; no aggregation |
| `B3 clean FL` | primary benign adaptation baseline | exact manifest/scope/schedule |
| `B4 attacked FL` | undefended attack viability | matched to B3 except declared poisoning/attacker |
| `B5 clean defended FL` | defense FPR and benign harm | matched defense with no attacker |
| `B6 attacked defended FL` | defense efficacy | matched B4 + defense; same attack knowledge class |

Recommended secondary anchors, subject to resource approval, are a centralized `D_base + D_tail_adapt` joint
retention oracle and full-scope generalization. A defense cell is interpretable only if its matched undefended
attack is viable and B5 still learns useful tail behavior.

B1–B6 use a fixed round/step/checkpoint rule and matched exposure; there is no per-cell tail early stopping.
If a predeclared failure rule terminates a run, the cell is failed/missing rather than compared at a favorable
checkpoint. A later recipe amendment creates a new protocol version and reruns all affected clean controls.

### 4.3 Protocol A boundary

Protocol A starts every client from one identical declared **non-nuScenes-detector** initialization. A public
ImageNet/NuImages initialization and fully random initialization are separate settings; public pretraining must
list data/tasks and prove it has not trained on the Protocol-A nuScenes detection split. The matched centralized
control uses the same architecture, data union, precision, augmentations, optimizer-step/example exposure, and
initialization.

For NuImages or any driving-domain pretraining, “no detector weights” is insufficient: audit log/sample/token,
normalized path/member, and confirmed raw-content overlap against Protocol-A nuScenes ownership. If raw frames
overlap, label the cell `nuScenes-related public 2D pretraining`, keep it separate from ImageNet/random controls,
and do not call it nuScenes-scratch. ImageNet-only camera initialization plus seeded non-camera modules is the
recommended primary Protocol-A control precisely to avoid that ambiguity.

Protocol A reports only the clean optimization gap and per-client/rare-class behavior unless a later owner
decision creates a separate, viable security protocol. It never uses `W_base`, `D_base`, or tail-adaptation
language and cannot be mixed into Protocol-B tables as another seed/control.

## 5. Threat model and secure-aggregation contract — unapproved

### 5.1 Recommended primary attacker

The primary attacker SHOULD be one authenticated but compromised regional/fleet silo in a synchronous
Protocol-B adaptation federation:

- it receives `W_base` and each global model sent to honest clients and knows the architecture, optimizer,
  update scope, global schedule, its own silo data, and the published defense family;
- it controls only its legitimate local `D_tail_adapt,k` stream and can digitally add the declared trigger and
  change target labels within that stream;
- it sends a protocol-shaped update from the allowed parameter keys; the primary data-poisoning cell does not
  grant arbitrary server control, other clients' data, evaluation labels, secure-aggregation keys, or split
  selection;
- it cannot choose its client, tail rule, target set, trigger thresholds, or malicious round after seeing ASR;
- the server is non-malicious. Server privacy posture and update visibility are separately declared below.

**Unapproved target recommendation:** use digitally triggered `car` disappearance as the primary goal because
the class is expected to retain broad base/tail/evaluation support and the existing identity-based evaluator can
be repaired rather than replaced. Freeze this only after the authorized train-only support audit; box shift,
phantom insertion, rarer classes, and alternate modality triggers are distinct secondary goals, not pooled ASR.

One compromised silo is the recommended lower-bound primary cell; after `K` is known, the manifest must report
`m/K` and per-round malicious participation. A predeclared multi-silo fraction is a separate **compromise/
collusion among already registered silos** stress cell. A Sybil attack is different: it grants the adversary new
identities or enrollment influence and therefore needs an identity/credential/admission threat model; it is out
of the recommended primary scope. The historical formula `m=floor(rho*N)` and `rho=0.2` are not approved
defaults. If selecting a single roster member materially changes support, use a deterministic stratified choice
or a budgeted all-one-client sweep rather than reporting the most favorable client.

An **adaptive model-poisoning upper bound** may additionally allow arbitrary manipulation inside the same
allowlist, norm/scale constraints, knowledge of the defense algorithm, and colluding clients. It must never be
conflated with the protocol-compliant data-poisoning result. Every capability delta gets its own cell.

### 5.2 Trigger and deployment scope

The primary claim is **offline digital sensor/data poisoning**. Current trigger placement reads ground-truth
boxes, calibration/projection, and LiDAR evidence, so the exact attacker oracle must be disclosed. A weaker
realistic variant would replace ground truth with attacker-observable 2D/tracking estimates and must be tested,
not asserted.

No physical sticker/material, real vehicle, live road, end-to-end driving-policy, collision-risk, or safety
guarantee follows from nuScenes digital replay. Physical claims require a sensor/rendering/placement model,
viewpoint/distance/weather transformations, print/material realization, and a physical or defensible simulator
evaluation under separate ethical/compute approval. Until then use “digital trigger” and “3D detector output,”
not “causes crashes” or “works in the physical world.”

### 5.3 Server visibility profiles

Two profiles are mutually exclusive and MUST be named in every cell:

| Profile | Server learns | Compatible current behavior | Allowed claim |
|---|---|---|---|
| `VIS` visible updates | each pseudonymous client update, weight, and decision telemetry | current Flower path and current robust defenses | trusted/authorized aggregator sees individual updates; **no secure-aggregation privacy claim** |
| `SA` standard secure aggregation | only a threshold cohort aggregate/sum, plus declared public metadata | plain aggregate/FedAvg attack study; current individual-update defenses do not run | attack remains possible under secure-sum; defense compatibility only if implemented cryptographically/verifiably |

Recommendation: use `VIS` for the primary mechanism/defense experiment because the current proposed
module-aware detection and baseline defenses inspect individual deltas; include an `SA` compatibility analysis
and, if resources allow, an aggregate-only attacked FedAvg cell. If secure aggregation is a required primary
deployment property, the defense must be redesigned and reviewed before a claim.

In both profiles the coordinator follows the declared protocol, does not poison the model, and does not collude
with malicious clients. `VIS` authorizes inspection of pseudonymous individual updates for aggregation/defense
but makes no cryptographic privacy promise; `SA` must name its server/client collusion threshold, dropout model,
weight validation, and residual aggregate leakage. Any trusted validation/probe/reference data used by a
defense must have a separately legal owner and hash (for example an approved `D_base_dev` or synthetic/public
set) and may never come from `E_tail`, official validation/test, or another client's private data.

Compatibility matrix:

| Operation | Standard secure-sum compatibility | Reason/condition |
|---|---|---|
| FedAvg / weighted sum | yes, subject to securely supplied/validated weights | needs only aggregate |
| data-poisoning attacker | yes | poisoned update is hidden in the same sum |
| server FLAME, FoolsGold, MultiKrum, coordinate median | no in current form | compare/filter individual updates |
| server per-client norm clipping | no in current form | needs each norm/update |
| client-side clipping | only with trust or verifiable proof/TEE/MPC | a malicious client can otherwise ignore it |
| module-wise norm/direction/spectrum defense | no in current form | needs individual module deltas |
| RFLPA-like secure computation | potentially, but it is a new protocol | requires implementation, cryptographic assumptions, leakage and overhead evaluation |

Bonawitz-style secure aggregation reveals the aggregate, not the individual vectors. RFLPA explicitly motivates
cryptographic machinery because plaintext robust aggregators conflict with secure aggregation. Transport TLS,
pseudonyms, deleting logs after use, or running the experimenter's telemetry out-of-band does not make a
defense secure-aggregation compatible.

## 6. Metric and evaluation contract — unapproved

### 6.1 Clean capability, tail utility, retention, and dispersion

For each checkpoint/seed, report complete official-validation mAP, NDS, per-class AP, TP errors, recall, and
target-class clean detection. Additionally report the frozen `E_val_common`, `E_val_tail`, and `E_tail_train`
slices with sample/scene/log/instance counts and uncertainty at the scene/log unit.

For metric `M` where higher is better:

```text
tail_gain(M)       = M(W_after, E_tail)   - M(W_base, E_tail)
common_change(M)   = M(W_after, E_common) - M(W_base, E_common)
forgetting(M)      = M(W_base, E_common)  - M(W_after, E_common)
pooled_oracle_gap  = M(W_oracle, E_tail)  - M(W_clean_FL, E_tail)
```

Report signed changes; do not clip negative “forgetting” to zero. Include overall official-val change so a
carefully chosen tail slice cannot hide global collapse. Exact gates belong to S11/owner, but order of evidence
is binding in spirit: `W_base` capability → pooled-tail learnability → local-only/clean-FL utility and retention
→ attack viability → defense.

Client utility dispersion SHOULD evaluate the global and local-only models on frozen held-out strata associated
with each regional/fleet proxy, reporting all values plus median, IQR, 10th percentile, and worst client. If the
log budget cannot reserve honest held-out support, mark this unavailable. Update-norm/cosine dispersion and
in-sample training loss are diagnostics, not substitutes for held-out client utility.

Final claim cells SHOULD use at least three predeclared seeds, paired manifests/schedules, scene/log-level
bootstrap confidence intervals and effect sizes. The SEC'27 CFP does not mandate a particular seed count or
test; this is the proposed scientific contract. Failed/missing seeds and every omitted cell remain visible.

### 6.2 ASR and exact coverage

Define and hash a geometry-only target universe `U_geom` independently of attack/model output. For each seed,
derive (a) `U_B3`, targets clean-detected by matched undefended clean FL `B3`, and (b) for every defense `d`,
`U_B5,d`, targets clean-detected by its clean-defense checkpoint `B5,d`. Bind sorted `(sample_token, ann_token)`,
all thresholds, manifest, evaluator/config, and reference checkpoint hashes before B4/B6 evaluation.

Report three complementary views rather than one denominator:

1. **B3-fixed cross-cell panel.** On the identical `U_B3`, report for B3/B4/B5/B6 both untriggered miss rate,
   triggered miss rate, and `fixed_trigger_delta = triggered_miss - untriggered_miss`. This exposes clean model
   loss separately from trigger effect and prevents a defense from shrinking its denominator.
2. **Defense-matched panel.** On `U_B5,d`, report B5,d and B6,d untriggered/triggered outcomes and their paired
   trigger deltas. This is the required B5→B6 clean-defense anchor; B5's own response to the trigger is the
   defense's no-attacker false-activation baseline.
3. **Checkpoint-paired ASR.** For each attacked checkpoint `W` (B4 or B6), restrict `U_geom` to targets that the
   same `W` detects on the untriggered paired input, then report the proportion achieving the attack goal after
   triggering plus `clean_coverage = N_W / |U_geom|`. Its denominator may differ and must never be hidden.

The B3-fixed view supports cross-defense comparison; the B5-matched view isolates each defense's clean baseline;
the checkpoint-paired view measures trigger causality without rewarding a model that already lost the target.
The primary poisoning-specific quantities are paired differences-in-differences on the same targets/scenes:

```text
excess_B4      = fixed_trigger_delta(B4)   - fixed_trigger_delta(B3)       on U_B3
excess_B6,d    = matched_trigger_delta(B6,d) - matched_trigger_delta(B5,d) on U_B5,d
```

Compute these from per-target paired indicators, then use scene/log-cluster intervals; do not merely subtract
two rounded aggregate percentages. A viable backdoor requires a predeclared positive `excess_B4` gate, so an
ordinary patch/occlusion effect already present in B3 cannot be relabeled as poisoning success. Defense efficacy
uses the matched `excess_B6,d` plus cross-cell B3-fixed panels. A defense succeeds only if all views, untriggered
target coverage, tail gain, and common utility meet their gates.

Every evaluator MUST require exactly one decode for every expected token and every frozen target. Missing,
duplicate, extra, wrong-checkpoint, or wrong-threshold entries invalidate the cell; missing samples never count
as disappearance. Match eligible GT only against the declared candidate set so an ineligible GT cannot consume
a detection before eligibility filtering. Batch-size/mode and prediction completeness are provenance fields.

Report numerator/denominator with scene/log cluster uncertainty, target distance/visibility/point bins, client/
round participation, and target-support counts—not ASR alone. Do not pool raw numerators across seeds whose
clean-reference eligibility sets differ; report per-seed values and a declared paired aggregate/interval.

### 6.3 False triggers and collateral effects

Before B4/B6 outputs are opened, freeze and hash every sample/target/placement list, pattern generator, clean-
detection eligibility rule, matching threshold, class map, and numerator condition. At minimum report:

| Metric | Frozen denominator | Numerator |
|---|---|---|
| `FDR_repeat` | targets in the declared clean-reference eligibility set | targets lost by an identical unmodified second decode |
| `FTR_lookalike` | clean-detected targets in a predeclared natural/benign-pattern list where the exact trigger is absent | targets achieving the attack goal without the exact trigger |
| `FTR_offtarget` | clean-detected intended targets under a hashed set of exact-trigger placements away from the target | intended targets achieving the attack goal despite off-target placement |
| `FTR_nonclass` | clean-detected non-target-class objects receiving the trigger under predeclared eligible placements | those objects suffering the declared disappearance/misclassification effect |
| `collateral_FN` | all clean-detected non-target objects in the primary triggered scenes | objects lost under the same matcher/threshold; also report new phantoms per scene |
| `occlusion_control` | the primary fixed/paired target sets with a same-area, predeclared nonaligned patch | attack-goal outcomes, reported beside—not subtracted silently from—the exact trigger |

Report each numerator, denominator, rate, unavailable/underpowered list, and scene/log-cluster interval. Natural
look-alikes may not be selected after observing their activation; if a reproducible pre-outcome selection rule
cannot be defined, mark that metric unavailable rather than substitute hand-picked examples.

The existing “false disappearance baseline” is only clean redecoding stability, and the historical occlusion
control is not a natural false-trigger rate. Neither may be relabeled.

### 6.4 Defense and system metrics

For every round and globally, record malicious-client TPR/FNR and honest-client FPR/TNR. For hard filters,
honest FPR is rejected honest submissions divided by participating honest submissions. For soft weighting,
predeclare a weight/action threshold and also report the full honest/malicious weight distributions; do not
invent a threshold after results. Slice honest FPR by rare/tail client because benign tail updates are the likely
outliers. Report clean-defense tail/common harm from B5.

System accounting includes:

- trainable and total parameters; bytes uploaded/downloaded per client/round and in total;
- participants, successful/dropout rounds, local examples, optimizer steps, effective exposure;
- client/server GPU-hours, CPU-hours, peak memory, wall-clock, and storage;
- aggregation/defense latency and any profiling overhead excluded from deployment;
- secure-aggregation/cryptographic expansion, proofs/TEE assumptions, and failure threshold when applicable.

## 7. Leakage and comparability audit checklist

A future manifest/run is valid only if all rows are demonstrated from the actual immutable artifact:

| Gate | Required evidence | Failure interpretation |
|---|---|---|
| official split isolation | no train-owned log/scene/sample/sample-data/raw content in official val/test | data leakage; invalidate all claims |
| base/tail/eval isolation | pairwise zero intersections at all ownership levels | `W_base` or clients saw evaluation/tail data |
| sweep closure | every configured sweep and adjacency dependency has same owner as keyframe | temporal/raw leakage |
| duplicate raw assets | normalized member/path and confirmed content duplicates have one owner | same sensor evidence crosses sets |
| exact cover | every expected entity owned once; no missing/extra/duplicate token | manifest incomplete or ambiguous |
| criterion independence | rule/search log predates attack/ASR; uses train-only fields | post-selection/attack-conditioned tail |
| blind tail evaluation | identity-only holdout commitment predates `U_design` stats; no `H_eval_blind` feature/support access in rule-selection log | selected/transductive evaluation masquerades as held-out |
| target support | frozen base/tail/eval support floors pass | ASR or utility denominator underpowered |
| client/eval reserve | tail held-out support not assigned to any client | in-sample tail claim |
| `W_base` lineage | only `D_base_fit`; exact checksum shared by matched cells | Protocol-B initializer contamination |
| matched controls | same manifest/scope/init/precision/schedule/exposure except named factor | causal comparison invalid |
| evaluator completeness | exact tokens/targets and no missing-as-success | metric inflation/deflation |
| class/box semantics | class-map hash; center/dimension order, meters, yaw/quaternion convention and box frame fixtures | wrong class/unit/yaw silently changes AP/ASR |
| calibration/projection | sensor→ego→global and sweep→key-LiDAR transforms; image resize/crop/flip→projection parity fixtures | trigger/box projected into the wrong frame or pixels |
| decode/batch invariance | exact score/NMS/range/mode hash; batch-1 reference; declared parity tolerance across supported batching | batch/mode artifact masquerades as attack/clean effect |
| sparse/sweep edge cases | voxel/index semantics, configured sweep order/time channel, empty/missing sensor behavior | ownership-correct inputs still enter a different model geometry |
| precision/resume | same declared fp32 or fp16-AMP regime; full state/checksum | changed training dynamics |

No current artifact passes this complete table; this is not a claim that the data themselves are contaminated,
only that the necessary audit has not yet been materialized.

## 8. Current repository/schema audit at `BASE_SHA`

### 8.1 Reusable substrate and missing ownership fields

- `info_cache.build_keyframe_info` resolves sample → scene → log and sensor/calibration/pose records
  (`fl_v3/src/fl_v3/data/nuscenes/info_cache.py:45-49,73-75`). Persisted entries include sample/scene/log,
  location/timestamp, six camera paths, LiDAR path, transforms, GT annotation/instance identity, and optional
  sweep paths/transforms (`:96-120,199-237`). This is a useful base.
- It does not persist camera/LiDAR keyframe `sample_data_token`, sweep source sample/scene/log or adjacency,
  ZIP member CRC, or raw-content digest. Cache hashing binds many paths/transforms/labels and sweeps
  (`:260-292`), but `n_sweeps` is not bound in the cache filename/sidecar (`:338-376`). Different-name same-
  content duplication and sweep ownership therefore cannot be audited from the cache alone.
- `NuScenesMultimodalDataset(..., sample_tokens=...)` can materialize an explicit token shard
  (`dataset.py:80-119`), which is the correct future seam after a manifest is validated.
- The production loader still uses ordinary `Image.open`/`np.fromfile` paths (`dataset.py:38-54`), and dataset
  verification requires a directory (`paths.py:147-170`). The “fully extracted” wording in `paths.py:6-8`
  conflicts with the current Arrhenius ZIP data reality. Protocol-B materialization depends on reviewed S01
  ZIP member coverage, parity, handle lifecycle, and multi-worker gates.

### 8.2 Existing partitions are not Protocol B

- Official split construction collects samples by official scene names (`info_cache.py:298-312`). Tests prove
  train/val sample-token disjointness (`tests/test_nuscenes_partition.py:146-154`) but not log, sample-data,
  sweep, raw-path, or content disjointness.
- The main partition builds location-coherent log groups with scene/sample/class counts and emits clients with
  `location`, `log_tokens`, and `sample_tokens` (`data/nuscenes/partition.py:53-83,251-273`). It partitions all
  train data directly among clients and has no `D_base`, `D_tail`, dev, or held-out-tail roles.
- The IID fallback is sample-level (`partition.py:367-372`) and can split adjacent frames/overlapping sweep
  windows from the same scene/log. It is forbidden for Protocol-B ownership.
- The historical health script checks sample union/overlap and log ownership, but not the full closure; its
  report omits exact full assignment. Thus the historical “GO” is not a Protocol-B leakage PASS.
- A silent nomenclature bug matters for future sizing: `derive_max_clients` relies on token-order greedy
  packing (`partition.py:141-158,183-186`), which is not a mathematical maximum. For floor 10 and log sizes
  `[6,6,4,4]`, it finds one group although two are feasible. Its output must not be used as a client-count upper
  bound. `build_log_group_partition(..., seed=...)` does not use the seed, and fallback candidates 25/20 are
  hard-coded (`:35-36,304-309`); neither transfers to a smaller tail pool.

### 8.3 Metrics/provenance do not yet enforce this proposal

- Existing ASR uses target-class/frustum/point/range/clean-score/center-match eligibility and hashes sorted
  target identities plus thresholds/checkpoint (`eval/asr.py:1-24,145-179,227-286`). That identity/hash seam
  is reusable.
- The matcher assigns detections across all target GT before eligibility filtering (`:105-133,156-168`), so an
  ineligible GT can consume a detection. Frozen-subset construction does not require complete/unique decodes.
  `disappearance_asr` counts a missing sample as disappeared (`:327-362`), while the T5 aggregation script
  treats unevaluated targets differently and does not put coverage in the gate. This can bias ASR in either
  direction and must be fixed before Protocol-B claims.
- The current false-disappearance measure (`asr.py:365-388`) is clean redecoding stability, not benign/natural
  false trigger. Phantom is only a placeholder, and no complete false-trigger suite exists.
- Detection evaluation provides official mAP/NDS/per-class metrics (`eval/detection_eval.py:162-187`), but
  `eval/report.py:23-84` lacks common/tail retention, forgetting, held-out client dispersion, uncertainty,
  communication, and structured defense-FPR fields. Clients currently have no validation loader
  (`training/tasks.py:657-666`).
- Provenance is the historical D10 full-train/log-group/full-participation clean regime
  (`eval/provenance.py:18-37,52-94`). It does not bind Protocol ID, ownership manifest, `W_base`, control role,
  update keys, tail criterion, participation schedule, server visibility, or secure aggregation. Attack
  provenance checks only a subset of D10 + poison rate/roster (`:105-132`) and cannot support this contract.

### 8.4 Current attack/defenses and visibility

- The current attacker uses a seed-derived `m=floor(rho*N)` honest-majority roster and deterministic data
  poisoning in its own shard (`attacks/poisoned_client.py:35-92`); it is not arbitrary model replacement.
  Placement reads GT/calibration/LiDAR evidence and is digital (`attacks/poison.py`, `fusion_ablation.py`,
  `trigger.py`), so its oracle and physical limitation must be explicit.
- FL aggregation unpacks every client array, sorts by partition ID, computes individual norms/diagnostics, then
  feeds individual vectors to defenses (`strategy/flower_strategies.py:291-329,430-475`). FLAME, FoolsGold,
  MultiKrum, FedMedian, and server norm clipping all consume these vectors (`:490-552`; aggregation/gradient
  cores). No secure-aggregation implementation is present.
- `DefenseDecision` can expose admitted/selected/coefficients, but production does not compare these against
  honest/malicious truth to produce FPR/TPR, and soft weights have no predeclared decision threshold. Existing
  defense cores are implementation-parity evidence only, not AD-domain validity.

### 8.5 Historical negative evidence that must remain visible

- The old full-train partition used 50 logs/28,130 keyframes grouped into 25 clients, with location/weather/
  class skew and absent rare classes in some clients. It is useful as a heterogeneity hypothesis, not an
  approved Protocol-B assignment.
- Historical clean FL severely underperformed the centralized model and exhibited tail confidence collapse;
  a FedAdam server-LR mistake and later cRT recovery still left large tail gaps
  (`collab/fl_baseline/phase3_fl_baseline.md:213-250,224-229,408-452`). A weak clean model blocks security
  interpretation.
- T4 found batch-dependent ASR (28/60 target differences) and a 9.4% false-disappearance issue before batch-1
  evaluation stabilized (`collab/T4/SPEC.md:134-146`). Exact batch/decode mode is therefore a provenance field
  and parity gate, not an implementation detail.
- The T5 camera-only attack was historically non-viable, but independent review was `CHANGES-REQUESTED` and
  three issues remain: shifted targets can leave the head grid; “nonaligned” does not prove LiDAR sparsity;
  integer minimum side can exceed the area budget (`collab/T5/REVIEW.md:7-33` and current trigger/poison code).
  This negative run cannot establish the final mechanism or novelty boundary.
- Historical checkpoints saw the full training pool; none is `W_base`. Old `/mimer` paths, bf16 choices, D10,
  and “first credible comparison” wording are superseded or unverified on Arrhenius.

## 9. Primary-literature novelty audit

### 9.1 Closest-work matrix

| Axis | Primary work screened | What it already establishes | Claim blocked / remaining boundary |
|---|---|---|---|
| FL for AD domains | [FedDrive](https://arxiv.org/abs/2202.13670), [FedDrive v2](https://arxiv.org/abs/2309.13336), [Federated Deep Learning Meets Autonomous Vehicle Perception](https://arxiv.org/abs/2206.01748) | AD perception FL under geographic/domain/label heterogeneity | no “first FL for AD perception”; our task/security protocol differs |
| federated multimodal AD | [AutoFed, MobiCom'23](https://doi.org/10.1145/3570361.3592517) | federated multimodal AD with sensor/label/environment/client heterogeneity | no “first multimodal FL for autonomous driving” |
| federated BEV/3D detection | [FedBEVT](https://arxiv.org/abs/2304.01534), [BEV-FePNet/DP-DeceFL](https://doi.org/10.1016/j.neucom.2024.127476), [Personalized FedM2former](https://doi.org/10.3390/pr13020449), [Fed3D, Apr. 2026 preprint](https://arxiv.org/abs/2604.15795) | federated camera BEV; decentralized camera-LiDAR 3D detection on nuScenes; personalized multimodal 3D detection; prompt-efficient federated 3D detection (Fed3D uses indoor datasets) | no “first federated camera-LiDAR/BEV/3D detector,” nuScenes multimodal FL, or prompt-efficient 3D FL |
| multimodal FL generally | [FedAFD, CVPR'26](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_FedAFD_Multimodal_Federated_Learning_via_Adversarial_Fusion_and_Distillation_CVPR_2026_paper.html) | current multimodal FL/fusion/distillation | multimodal federation alone is not novelty |
| long-tail FL | [BalanceFL, IPSN'22](https://doi.org/10.1109/IPSN54338.2022.00029), [FedLF](https://proceedings.mlr.press/v260/lu25a.html) | global/local long-tail and class imbalance in FL | no “first long-tail FL” |
| long-tail 3D detection | [Towards Long-Tailed 3D Detection](https://proceedings.mlr.press/v205/peri23a.html), [FOMO-3D](https://proceedings.mlr.press/v305/yang25e.html) | rare-class 3D detection and multimodal benefit | no “first long-tail/rare 3D detection”; nuScenes 10-class tail definition needs care |
| long-tail federated fine-tuning | [FedPuReL, CVPR'26](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_Fine-Tuning_Impairs_the_Balancedness_of_Foundation_Models_in_Long-tailed_Personalized_CVPR_2026_paper.html) | fine-tuning can erode balanced pretrained knowledge under long-tailed personalized FL | no “first long-tail federated fine-tuning” or generic forgetting observation |
| FL backdoors | [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459), [Can You Really Backdoor FL?](https://research.google/pubs/can-you-really-backdoor-federated-learning/), [FLAME](https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen), [PFedBA/Lurking in the Shadows, USENIX'24](https://www.usenix.org/conference/usenixsecurity24/presentation/lyu) | model replacement/data poisoning, realism constraints, defenses/utility, and persistent personalized-FL backdoors | no generic FL-backdoor, fine-tuning/personalized-backdoor, or robust-aggregation novelty |
| tail + FL backdoors | [Attack of the Tails, NeurIPS'20](https://proceedings.neurips.cc/paper_files/paper/2020/hash/b8ffa41d4e492f0fad2f13e29e1762eb-Abstract.html) | uses tail/edge inputs to create FL backdoors | no “first tail federated backdoor”; distinguish attack-chosen edge cases from frozen benign Protocol-B ownership |
| AD-FL poisoning/defense | [Bandit-based poisoning for AD FL](https://doi.org/10.1016/j.eswa.2023.120295), [Poisoning Attacks on FL for AD](https://ecp.ep.liu.se/index.php/sais/article/view/994), [SecFedDrive](https://doi.org/10.1109/CNS62487.2024.10735501) | poisoning/backdoor/defense in steering or trajectory AD federations | no “first AD federated poisoning, backdoor, or defense”; 3D multimodal Protocol-B mechanism is the narrower distinction |
| layer-local FL attacks | [Backdoor Federated Learning by Poisoning Backdoor-Critical Layers](https://arxiv.org/abs/2308.04466) | critical-layer poisoning can be compact/stealthy and evade defenses | no generic “first layer/module-localized FL attack” or dilution mechanism claim |
| layer/structure defenses and non-IID | [FedCPA, ICCV'23](https://openaccess.thecvf.com/content/ICCV2023/papers/Han_Towards_Attack-tolerant_Federated_Learning_via_Critical_Parameter_Analysis_ICCV_2023_paper.pdf), [FreqFed, NDSS'24](https://www.ndss-symposium.org/wp-content/uploads/2024-620-paper.pdf), [adaptive layer-wise alignment, ICCV'25](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_Stealthy_Backdoor_Attack_in_Federated_Learning_via_Adaptive_Layer-wise_Gradient_ICCV_2025_paper.pdf), [Breaking the Illusion, TIFS'25](https://doi.org/10.1109/TIFS.2025.3643155), [DoBlock, AAAI'26](https://ojs.aaai.org/index.php/AAAI/article/download/39778/43739) | critical parameters/frequency/layer dynamics; non-IID/domain skew can hide attacks, degrade defenses, and motivate block-restricted aggregation | no “first structural defense,” “first layer stealth,” or “first domain-skew hides/breaks defenses”; evaluate benign-tail suppression explicitly |
| multimodal backdoors | [Backdooring Multimodal Learning, IEEE S&P'24](https://doi.org/10.1109/SP54263.2024.00031), [SABRE-FL, ICLR'26](https://openreview.net/pdf?id=n1HBsszaY6), [BadPromptFL preprint](https://arxiv.org/abs/2508.08040) | modality contribution/localization; attack/defense in multimodal federated prompt learning | no “first multimodal or modality-localized backdoor,” nor broad “first multimodal federated backdoor/defense” |
| secure aggregation + robustness | [Practical Secure Aggregation, CCS'17](https://doi.org/10.1145/3133956.3133982), [SAFELearning](https://doi.org/10.1109/TIFS.2023.3280032), [RFLPA, NeurIPS'24](https://openreview.net/pdf?id=js74ZCddxG) | secure-sum hides individual updates; detectability/robustness requires protocol co-design or secure computation | no claim that plaintext client filtering works under secure aggregation or that compatible detection is new |
| fusion adversarial security | [Drift with Devil, USENIX'20](https://www.usenix.org/system/files/sec20-shen.pdf), [Security Analysis of Camera-LiDAR Fusion, USENIX'22](https://www.usenix.org/system/files/sec22-hallyburton.pdf), [Fusion Is Not Enough, ICLR'24](https://openreview.net/forum?id=3VD4PNEt5q) | multi-sensor localization and camera-LiDAR detection under spoofing/camera-only adversarial weaknesses | no broad “first sensor-fusion/camera-LiDAR security analysis” |
| 3D/fusion backdoors | [LiDAR 3D backdoor, SenSys'22](https://doi.org/10.1145/3560905.3568539), [BadFusion, IJCAI'24](https://www.ijcai.org/proceedings/2024/0039.pdf) | LiDAR physical backdoors; camera-oriented backdoors against camera-LiDAR 3D detection | no “first 3D detector/fusion/camera-trigger backdoor” |
| newest LiDAR backdoors | [MOBA, AAAI'26](https://doi.org/10.1609/aaai.v40i42.40842), [Mirage, Jun. 2026 preprint](https://arxiv.org/abs/2606.20752) | material-oriented physical and clean-label/black-box LiDAR 3D backdoors | no broad physical, clean-label, black-box, or practical 3D-backdoor novelty |

### 9.2 Defensible candidate novelty boundary

The closest defensible thesis is a **combination and mechanism claim**, subject to positive evidence:

> In a leakage-audited vendor-base → regional/fleet tail-adaptation workflow for camera-LiDAR 3D detection,
> measure whether malicious modality-localized tail updates overlap with benign rare-domain update structure,
> explain when standard client-update defenses reject useful tail adaptation or miss the backdoor, and evaluate
> a structure-aware defense under explicit update-visibility and utility constraints.

This is not automatically novel merely because the nouns are combined. The paper needs (a) a viable matched
attack, (b) causal module/block evidence beyond norms, (c) a clean strong detector and useful clean FL, (d) a
defense that beats tuned baselines and adaptive attacks without suppressing benign tails, and (e) a systems-
security argument about realistic silo compromise, communication, privacy visibility, and deployment limits.

No “first” should appear in title, abstract, introduction, or contributions unless a refreshed search near
submission supports a precisely scoped statement and reviewers can reproduce the search boundary. Safer
language is “we study,” “we characterize,” or “we provide evidence in this defined Protocol-B setting.”

### 9.3 Literature gaps before paper freeze

- complete backward/forward citation chasing from Fed3D, FedM2former, AutoFed, BadFusion, SABRE-FL, MOBA,
  Mirage, and RFLPA;
- confirm venue/final versions and corrections for 2026 preprints/current papers;
- search patents/industry systems and 2026 papers through the registration/submission dates;
- inspect any 2026 “cross-modal multi-level backdoor poisoning in multimodal FL” final proceedings, not only
  program/secondary listings;
- compare the eventual exact update scope/defense against layer-wise robust aggregation, PEFT/prompt attacks,
  and secure-computation defenses;
- verify overlap with every author team's own concurrent/accepted work before SEC'27 registration.

## 10. Research questions and claim–evidence map

### 10.1 Proposed RQs

- **RQ1 — Clean protocol validity.** Can the frozen detector produce a capable `W_base`, and can clean
  Protocol-B federation learn held-out tail conditions without unacceptable common-data forgetting, relative
  to pooled-tail and local-only controls? What does separate Protocol A reveal about the clean optimization gap?
- **RQ2 — Viability and scope.** Can one compromised regional/fleet silo implant a digital modality-localized
  backdoor during Protocol-B adaptation on a clean-capable detector, under matched poison/update exposure,
  while preserving untriggered overall/common/tail/target utility and low false activation?
- **RQ3 — Causal mechanism.** Which exact modules, directions, and spectral/energy components carry benign
  tail adaptation and the backdoor? Do block removal/replacement/retraining interventions causally change ASR
  rather than merely correlate with it?
- **RQ4 — Defense under heterogeneity.** Why do fairly tuned generic defenses accept malicious updates or
  reject benign tail clients? Can the minimum structure-aware defense reduce ASR while meeting tail/common
  utility, honest-FPR, adaptive-attacker, and overhead gates?
- **RQ5 — System/privacy boundary.** How do participation, update scope, attack modality, and visible-update vs
  secure-aggregate server profiles alter the result and communication/privacy assumptions? Which conclusions
  generalize, and which remain specific to this synthetic regional/fleet nuScenes protocol?

### 10.2 Claim–evidence table

All claims are tentative. “Status” is the evidence state at this handoff, not a prediction.

| ID | Tentative bounded claim | Required experiment/artifact | Falsifier / narrowing rule | Status |
|---|---|---|---|---|
| C0 | Protocol B has no base/client/eval raw or temporal leakage and tail evaluation was blind | canonical ownership manifest, raw/sweep closure, pre-stat holdout commitment/reveal log, adversarial validator tests, independent review | any crossing/missing/extra entity, mutable runtime split, evaluation support used to select the rule | **blocked; no manifest** |
| C1 | `W_base` is a capable common-data detector that never saw tail | D_base-only training provenance/checksum; overall/common/tail/target metrics | full-train init, weak target clean eligibility, unknown lineage | **unproven** |
| C2 | clean federated tail adaptation is useful | B0/B1/B2/B3, paired seeds, tail gain/common retention/forgetting/client dispersion/cost | no tail gain, unacceptable forgetting, large unexplained oracle gap | **unproven; must precede security** |
| C3 | one compromised silo can implant the declared digital backdoor | positive predeclared `excess_B4 = Δtrigger(B4)-Δtrigger(B3)` with per-target pairing/cluster CI, plus fixed/paired panels, exact coverage, clean utility and false-trigger controls | no excess beyond clean patch/occlusion, weak detection, missing coverage, client/seed cherry-pick | **unproven; old T5 is negative/invalid for final claim** |
| C4 | malicious and benign tail updates overlap in whole-model space but separate in a specific structure | paired per-module norm/direction/spectrum, benign-tail strata, block interventions and counterfactuals | only post-hoc t-SNE/norm correlation; effect disappears under intervention | **hypothesis only; prior layer/domain-skew work limits novelty** |
| C5 | generic defenses face an ASR–tail utility/FPR tradeoff here | tuned equal-budget baselines; B5/B6; honest tail-client FPR and adaptive attack | untuned baseline, unmatched clean model/exposure, attack not viable | **unproven** |
| C6 | proposed structure-aware defense improves that tradeoff | B3-fixed plus `excess_B6,d = Δtrigger(B6,d)-Δtrigger(B5,d)`, checkpoint-paired panels, predeclared threshold, seeds, strongest fair baseline, adaptive attacker, tail/common/FPR/overhead | tail suppression, high honest FPR, no poisoning-specific excess reduction, clean-defense miss conflated with ASR, defense-aware evasion | **unproven; depends on S13** |
| C7 | attack/defense has a stated deployment/privacy profile | `VIS`/`SA` manifest, communication measurements and compatibility tests | plaintext per-client inspection presented as secure aggregation | **current code supports VIS only** |
| C8 | result generalizes beyond one accidental split/config | approved alternate split/roster, modality/scope or architecture check with same protocol discipline | only one partition/seed/model; negative check hidden | **optional, resource-dependent** |
| C9 | Protocol A diagnoses clean optimization rather than supporting security | matched CL vs FL initialization/data/exposure and clean metrics | nuScenes-pretrained detector, mixed checkpoints/names, security interpretation on weak FL | **unproven clean control** |

No current claim is paper-ready. In particular, C3–C6 are logically blocked until C0–C2 pass.

## 11. Tentative paper skeleton and visual inventory

### 11.1 Title/abstract boundary

Tentative title candidates, all **unapproved**:

1. `Backdoors in Federated Tail Adaptation of Camera–LiDAR 3D Detectors`
2. `Securing Federated Tail Adaptation for Camera–LiDAR 3D Detection`
3. only if C4 is demonstrated: `When Benign Tail Updates Hide Backdoors in Federated 3D Perception`

**Tentative nonblank registration abstract (unapproved; contains no result claim):**

> Autonomous-driving vendors may train a camera–LiDAR detector on common data and later adapt it with rare
> regional data held by fleet silos. This planned study investigates backdoor risk in that federated adaptation
> stage. It specifies a train-only, scene/log/raw-asset-disjoint base/tail protocol; separates clean optimization
> controls from the primary security setting; and evaluates digital modality-localized poisoning against a
> capable 3D detector. The evaluation is designed to connect attack success to module-level update structure
> while jointly measuring tail learning, common-data retention, false triggers, honest-client defense errors,
> communication cost, and update-visibility constraints. It asks whether a structure-aware defense can reduce
> attack impact without suppressing the benign rare updates that motivate federation.

The author list/order/ORCIDs are **UNRESOLVED OWNER INPUT**; S12 cannot infer authorship from repository access,
funding, supervision, or tool use. Candidate primary topic is `Security of ML`, with CPS security only as an
owner-approved secondary topic. Neither the abstract nor these topics may be registered from this handoff.

Avoid “first,” “real fleets,” “physical,” “privacy-preserving,” and “secure aggregation” in the title unless the
corresponding exact evidence exists. A safe abstract sequence is: system workflow and adversary → leakage-
audited protocol → clean capability → measured attack/mechanism → defense with tail/FPR cost → explicit
digital/server-visibility/generalization limits. No result number enters the abstract without a raw-artifact hash.

### 11.2 Proposed body (13-page initial main-body budget)

1. **Introduction and security problem.** Vendor-base/fleet-tail workflow, threat, why benign tail learning
   creates a security tension, bounded contributions.
2. **System, data ownership, and threat model.** Protocol B primary; Protocol A clean control; component graph,
   clients, server visibility, attacker capabilities, digital scope.
3. **Detector and federated adaptation.** Frozen architecture, `W_base`, exact update scope, communication and
   controls—engineering only as needed to make the security study reproducible.
4. **Attack and hypotheses.** Poison channel, target/trigger/eligibility, data vs adaptive model-poison variants.
5. **Mechanism study.** Module geometry, benign-tail overlap, causal block interventions.
6. **Structure-aware defense.** Minimum rule, visibility requirement, threshold selection, complexity.
7. **Evaluation.** RQs, leakage evidence, clean controls, attack/false triggers, generic defenses/FPR, adaptive
   attacker, overhead and approved generalization.
8. **Discussion, ethics, and limitations.** Synthetic silo proxy, licensed data, digital-to-physical gap,
   secure-aggregation incompatibility, external validity and dual use.
9. **Related work.** AD FL/long tail; FL backdoors/domain skew/layer defenses; multimodal/3D security; secure
   aggregation—woven into positioning, not a novelty dump.
10. **Conclusion.** Only claims supported by the final claim–artifact ledger.

The appendix plan includes the mandatory Open Science appendix and, separately, a strongly encouraged Ethics
appendix warranted by this work, plus exact manifests/thresholds, extra checks, and artifact instructions.
Reviewers are not required to read optional appendices, so threat model, primary metrics, essential controls,
and key limitations stay in the body.

### 11.3 Figures and tables

| ID | Proposed content | Evidence source |
|---|---|---|
| Fig. 1 | Protocol-B ownership/system diagram: vendor `D_base` → `W_base` → regional/fleet proxies; separate held-out evaluation; attacker/server visibility | frozen manifest and threat contract |
| Fig. 2 | exact detector/update-scope and trigger path; trainable/frozen parameter bytes | S07 model + scope hash |
| Fig. 3 | clean utility: W_base/oracle/local/clean-FL tail gain vs common forgetting, with client dispersion | B0–B3 raw metrics |
| Fig. 4 | benign vs malicious per-module geometry plus causal intervention result | S13 paired updates/interventions |
| Fig. 5 | defense Pareto: ASR vs tail utility/common retention, encoded by honest FPR/overhead | B4–B6 and baselines |
| Table 1 | Protocol A/B, ownership, client, init, update, visibility, attacker contract | owner-approved contract after S12-R |
| Table 2 | closest-work feature matrix without first claims | refreshed primary-literature audit |
| Table 3 | clean controls, all seeds, overall/common/tail/client/cost | S13 |
| Table 4 | attack/false-trigger/coverage/stealth by modality/roster | S13 |
| Table 5 | defense/adaptive attack/FPR/utility/SA compatibility | S14 |
| Table 6 | communication, compute, memory, and cryptographic overhead | run manifests/profiles |

Every plot/table generator should consume only checksummed raw records and fail on a missing requested cell.

## 12. Artifact, ethics, and venue implications

### 12.1 Artifact plan

The anonymous artifact SHOULD contain, subject to later owner/publication approval:

- source and exact dependency/environment lock; Arrhenius activation recipe without private paths/secrets;
- metadata-only ownership builder/validator, canonical schema, approved manifest or redistributable hash/index,
  and tests that inject every leakage class;
- resolved Protocol-A/B configs, update-key hashes, W/checkpoint hashes, participant schedules, and provenance;
- scripts that reconstruct tables/figures from checksummed raw outputs and refuse missing cells;
- evaluation/ASR/false-trigger/FPR code with synthetic or mini engineering fixtures clearly marked non-science;
- raw-result index/checksums, resource/communication accounting, failure/negative-cell ledger, and reproduction
  instructions for license holders.

Do not redistribute licensed nuScenes raw data, extracted ZIP members, or checkpoints if the license forbids it.
The owner should obtain a license review for sample tokens/manifests and model weights; if an item cannot be
shared, the Open Science appendix must name it and explain why, while sharing all lawful surrounding code,
hashes, synthetic fixtures, and result derivations. Dataset availability is not upload authorization.

Attack release requires a dual-use review: consider staged release of evaluation/defense and digital replay
code, omit turnkey physical deployment assets, disclose current oracle assumptions, and state residual risks.
No new participant recruitment/interaction or live-vehicle/public-system experiment is authorized. nuScenes
nevertheless contains captured public-road users, so data-subject privacy, licensing, stakeholder harm, and
release ethics remain in scope rather than being dismissed as “no human-subject data.”

### 12.2 SEC'27 rules and deadline risk (verified 2026-07-10)

- The official page remains a **Preliminary CFP** and must be rechecked.
- Cycle-1 mandatory registration is **2026-08-18 AoE**; paper **2026-08-25 AoE**; submission artifact
  **2026-08-28 AoE**. From this search date those were 39/46/49 days away.
- Registration fixes title, complete author list, and topics; the abstract may be tentative but must be nonblank.
  All real authors need the required intellectual/writing/approval/accountability role; ORCID/HotCRP terms and
  conflicts need early completion. AI is not an author, and humans remain responsible for text/code/data/cites.
- Initial main body is at most 13 pages; references/appendices are outside that count but reviewers need not read
  them. Camera-ready is at most 20 total pages including references and appendices. Use the official template,
  preserve its formatting, and make plots legible in grayscale; SEC'27 has no replication-submission track.
- Double anonymity covers paper and artifact: remove names, organizations, usernames, repository/commit
  history, private paths, and tracking. Self-cite in the third person.
- A mandatory **Open Science Appendix** must identify artifacts/access or specific non-release reasons. Its
  anonymous URL must already be in the Aug. 25 paper and is then final. The three-day grace permits only
  artifact upload/anonymization; the paper/URL cannot change, and the artifact freezes Aug. 28. The anonymous
  link must remain available through shepherd approval. Shareable artifacts must be made public upon acceptance;
  license, legal, subject-harm, or adversarial-risk exceptions require itemized explanation, not a blanket waiver.
- Ethics appendix is no longer mandatory but is strongly encouraged; ethics is still reviewed and can cause
  rejection. This project should cover road users, data licensors/subjects, vendors/silos, dual use, digital-
  only experimentation, attack release, and residual harms.
- ML security work must explain attacker, attack surface, generality, practicality, and broader systems-security
  relevance. Backdoors/poisoning are in scope; ordinary noise robustness without a deliberate adversary is not.
- The natural tentative primary topic is `Security of ML`; CPS/emerging-system security is a possible secondary
  topic, both owner decisions. Topics freeze at registration.
- SEC'27 PC/AEC call/final instructions/statistical checklist were not published at verification time. Cycle-1
  rejection cannot be resubmitted to SEC'27 Cycle 2, increasing the cost of an under-evidenced submission.
- Do not submit the same work simultaneously to another proceedings venue. Disclose overlapping/concurrent
  related work as required; refresh author COIs when the PC appears. The current limit is seven papers per
  author per cycle.

**Recommended safe internal dates, not venue rules:** by Aug. 10–11 settle qualified authors/order, candidate
title/topics, ORCIDs/basic COIs and send rule questions; Aug. 18 is the hard title/authors/topics freeze; Aug. 25
freezes paper, claim–evidence ledger, Ethics/Open Science and anonymous URL; Aug. 28 freezes artifact. Recheck
the preliminary CFP and any new AEC/PC guidance at each point.

## 13. Owner decision docket

No row is approved by this handoff. “Latest safe freeze” means before the listed downstream action; a changed
decision creates a new protocol version and invalidates prior authorization/comparability.

| ID | Decision | Recommendation | Alternatives | Evidence still needed | Latest safe freeze |
|---|---|---|---|---|---|
| D12-01 | ownership unit | log-rooted dependency-connected component with sample-data/sweep/raw-content closure | scene component only if closure proves no shared log/adjacency; never sample-IID | reviewed S01 backend/coverage; metadata schema tests | schema and sensor scope before any new Protocol-B component/statistics reveal |
| D12-02 | tail rule | predeclare semantic context-conditioned family/support floors; fit exact rule on `U_design` only | class-only, context-only, or predeclared composite; model-error mining only secondary | owner-approved `U_design`-only metadata statistics; sealed `H_eval_blind`; no ASR | family/procedure/floors before stats; exact rule before holdout unseal/allocation hash |
| D12-03 | base/tail/dev/eval allocation | identity-hash blind holdout first; reserve `D_base_dev`; derive `E_tail_train` only after rule freeze | a support-selected train evaluation set only as transductive/secondary | component closure, holdout commitment, U_design feasibility, blind support and solver negatives | holdout rule before stats; allocation/rule hash before unseal/`W_base` |
| D12-04 | client unit/count | synthetic regional/fleet silo proxies; `K` derived from log/support constraints | coarser region silos; no “one car” interpretation | per-component location/context/target support | assignment hash before S13 |
| D12-05 | update scope | symmetric modality adapters/projections + fusion/BEV/head; freeze heavy encoders | full-model; PEFT/prompt as distinct protocol/generalization | reviewed S07 module names/shapes/bytes and disclosed pre-tail engineering evidence | exact key hash **before any new Protocol-B component/tail/split/support statistics reveal** |
| D12-06 | `W_base` and adaptation recipes | D_base-only `W_base`; fixed primary optimizer/exposure/checkpoint rules and same checksum per matched seed | public backbone init variants, declared separately | S07/CL-freeze architecture; owner precision/gates/resource estimate | architecture/recipes/exposure **before any new Protocol-B statistics reveal**; exact run request later |
| D12-06A | Protocol-A initialization | ImageNet-only camera initialization plus declared seeded initialization for all other modules; identical full tensor checksum for CL/clients; fully random secondary | fully random primary; NuImages/driving-domain init as a separately labeled non-scratch setting | exact pretraining dataset/task/license, detector-weight provenance, log/token/path/content-overlap audit, exposure estimate | before Protocol-A config/run request |
| D12-07 | participation | synchronous full participation primary; one deterministic partial schedule secondary | partial primary if owner prioritizes realism and funds sampling variance | `K`, compute/communication estimate, secure cohort minimum | before S13 clean run manifest |
| D12-08 | attacker/roster | one compromised registered silo primary data poisoning; fixed/exhaustive roster; adaptive model poisoning separate | multiple already registered silos compromised/colluding; Sybil enrollment is a separate identity threat and out of primary scope | client support, identity/admission assumption, exact trigger viability, budget | before attack code/config and attacked `RUN_REQUEST` |
| D12-09 | server privacy | `VIS` primary for current defenses; honest statement of no SecAgg; `SA` attack/FedAvg compatibility branch | redesign around SAFELearning/MPC/TEE/RFLPA-like protocol | owner privacy requirement, implementation/overhead scope | before S14 defense design; before any privacy claim |
| D12-10 | target/trigger scope | `car` disappearance, digital-only, exact eligibility/oracle disclosed, with false-trigger/collateral controls | box shift/phantom/rarer class; weaker attacker-observable placement; physical only in separately approved project | authorized train-only target support, S13 trigger bug fixes and reviewed evaluator | before attacked `RUN_REQUEST` |
| D12-11 | metrics/gates/statistics | poisoning-specific paired difference-in-differences (`excess_B4`, `excess_B6,d`) plus B3-fixed, B5-matched and checkpoint-paired panels; exact coverage, common/tail/client/FPR/cost; ≥3 paired seeds/log bootstrap final | owner-approved narrower design with explicit limitation | S11 evaluator/coordinate/batch fixtures and resource estimate | definitions/floors before holdout unseal and the runs they judge |
| D12-12 | novelty/contribution | mechanism + defense under strict Protocol B; no first | protocol/measurement paper if attack or defense fails | refreshed closest-work full-text/citation audit and S13/S14 outcomes | contribution wording before Aug. 18 title; numbers/claims Aug. 25 |
| D12-13 | title/authors/topics | candidate title 1; genuine authors only; Security of ML primary | candidates 2/3; CPS secondary | owner author list/order/ORCID/COI; final evidence trajectory | **2026-08-18 AoE** |
| D12-14 | artifact/ethics release | anonymous partial artifact, license-aware hashes/scripts, explicit ethics/dual-use plan | narrower release with itemized justification | license and release review; AEC update; anonymity dry run | URL/paper Aug. 25; artifact Aug. 28 |

### Recommended approval sequence

1. S12-R independently reviews this proposal and actual source evidence.
2. After reviewed S01/S07/S11 evidence, and **before any new exact Protocol-B component/tail/split statistics
   are revealed**, owner freezes the ownership/sensor schema, architecture, update-key hash, primary W_base/
   adaptation recipes and exposure, tail-rule family/procedure/support floors, blind-holdout rule, and metrics.
3. An exactly authorized metadata tool commits/seals `H_eval_blind` and reports only `U_design` feasibility.
   Owner then freezes the exact tail rule and base/client allocation/hash; only afterward may the tool unseal
   blind evaluation support and apply the unchanged rule to official validation.
4. Owner also freezes participation/privacy/attacker decisions. S13 gets a separate clean-only request for
   `W_base` and B0–B3. Underpowered blind support or a failed clean gate stops security work; neither triggers
   evaluation reselection or post-tail scope/recipe tuning.
5. Only after reviewed clean gates, owner approves a distinct B4 request; only after viable attack evidence does
   S14 obtain defense/adaptive requests.
6. Venue/title/artifact decisions follow the hard dates above; no training or upload authority is implied.

## 14. Proposed future durable paths — not created, ownership required

If S00/owner accepts the contract, the following exact artifacts are recommended for later owned sessions.
They are proposals, not authorization to edit these paths:

| Proposed path | Minimum contents |
|---|---|
| `fl_v3/docs/protocol_b_contract.md` | owner-approved symbols, allowed data flows, client/system/threat/update/metric contract, decision IDs and effective protocol version; no results |
| `fl_v3/src/fl_v3/data/nuscenes/ownership_manifest.py` | typed raw-asset/sample-data/sweep/log component schema; canonical serialization/hash; exact-cover, closure, official-split, raw-duplicate and pairwise-disjoint validators |
| `fl_v3/scripts/build_protocol_b_manifest.py` | identity-only precommitted blind holdout; sealed evaluation commitment; `U_design`-only rule fit; post-freeze unseal/application; deterministic allocation and negative-feasibility report; no runtime split recomputation |
| `fl_v3/manifests/protocol_b/<manifest_id>.json` | source/archive/cache/backend hashes, blind-holdout commitment/reveal, rule/support stats by allowed pool, complete base/tail/dev/eval/client ownership, raw dependencies, solver/seed, audit verdicts and canonical hash |
| `fl_v3/tests/test_protocol_b_ownership.py` | adversarial injections for log/scene/sample/sample-data/sweep/raw path/content/official split leakage, adjacency, missing/duplicate/extra tokens, deterministic hash and tamper failure |
| `fl_v3/src/fl_v3/strategy/update_scope.py` | ordered parameter/buffer allowlist, shapes/bytes/hash, strict load/aggregation guards |
| `fl_v3/src/fl_v3/eval/provenance.py` v2 | Protocol A/B, role, manifest, init/`W_base`, update scope, precision/recipe/exposure, participant schedule, attacker/roster, visibility/SecAgg and all artifact hashes |
| `fl_v3/src/fl_v3/eval/protocol_b_metrics.py` and `eval/report.py` v2 | frozen overall/common/tail/client metrics, fixed/paired ASR with exact coverage, false triggers/collateral effects, defense FPR/TPR, uncertainty and system cost |
| `fl_v3/src/fl_v3/strategy/secure_aggregation.py` or capability schema | explicit `requires_individual_updates` per aggregator/defense and configuration hard-fail; cryptographic code only if separately scoped |

The manifest builder should be a one-time auditable tool. Training must consume `manifest-path + expected-hash +
role` and hard-fail on mismatch, never recreate a “close enough” partition at runtime. S00 alone may update the
canonical Orchestra decision/change-control ledgers.

## 15. Downstream dependencies and blockers

| Dependency | Why S12/S13 needs it | Current implication |
|---|---|---|
| independent S12-R | adversarial review of leakage, threat, metrics, novelty, and decision docket | no accepted downstream use before review |
| owner decisions D12-01…14 | material scientific choices remain unlocked | no split, clients, training, attack, paper claim, or upload |
| reviewed S01 | ZIP-aware loader, complete member coverage, directory/ZIP parity, multi-worker lifecycle | full-data ownership/materialization not ready |
| reviewed S07 / CL architecture freeze | exact module topology, head, transforms, parameter names/bytes | update allowlist and W_base architecture cannot yet be hashed |
| reviewed S10/CL evidence | capable centralized architecture and exact `CL-PILOT`/`CL-FREEZE` semantics | security work cannot start from a weak/unstable detector |
| reviewed S11 | metric correctness, modes, provenance and acceptance gates | tail/common/ASR/FPR contract not executable yet |
| S13 clean phase | new D_base-only `W_base`, oracle/local/clean FL/Protocol-A evidence | attacked cells remain blocked until clean gate passes |
| S13 attack/mechanism | viable matched attack and causal module evidence | S14 defense remains blocked until attack viability |
| S14 | fair defenses, adaptive attacker, FPR/utility/overhead/generalization | defense and final security claims unavailable |
| S15 + venue refresh | anonymous paper/artifact, license/ethics/reference/anonymity audits | current skeleton is not a submission |

S01 trainval availability is not compute permission. Every future metadata scan that is material, and every
clean/attack/defense job, requires an exact `RUN_REQUEST.md` and explicit owner authorization bound to commit,
manifest/config, cells/seeds, command, resources, and output location.

## 16. Allowed and forbidden interpretations

### Allowed now

- “S12 proposes, but the owner has not approved, a log-rooted dependency-connected Protocol-B ownership and
  regional/fleet-silo contract.”
- “At the pinned SHA, current partition/provenance/evaluation code lacks several fields/checks required by that
  proposed contract.”
- “The historical full-train checkpoint cannot be Protocol-B `W_base` because it may have seen future tail.”
- “Current defenses inspect individual client updates; under standard secure-sum they are incompatible without
  protocol redesign.”
- “Primary literature already covers federated AD/multimodal/3D perception, AD-FL poisoning, long-tail FL,
  multimodal/3D backdoors, layer/domain-skew defenses, and secure-aggregation-aware detection; broad first
  claims are unsafe.”
- “Historical weak clean FL and non-viable/changes-requested T5 attack are negative engineering evidence that
  motivates stronger gates; they are not new Protocol-B scientific results.”

### Forbidden now

- any statement that `D_base`, `D_tail`, `E_tail`, tail criterion, client assignment, update scope, roster,
  malicious fraction, participation, metric threshold, title/authors/topics, or paper claim is frozen/approved;
- any leakage PASS, split hash, sufficient support, trainval readiness, scientific PASS, attack viability,
  defense efficacy, physical feasibility, real-fleet realism, privacy preservation, or secure-aggregation claim;
- use of the full-train CL/FL checkpoints as `W_base`, or reuse of old D10/25-client artifacts as Protocol B;
- security claims from mini/smoke, single weak checkpoint, missing evaluation cells, or an undefended attack that
  is not viable;
- “first” claims for FL in AD, multimodal/3D/nuScenes FL, long-tail/tail FL backdoors, AD-FL poisoning/defense,
  camera/LiDAR/fusion backdoors, module/layer-aware attack/defense, domain-skew defense failure, or
  secure-aggregation-compatible detection;
- paper/result/artifact upload, registration, publication, commit, merge, push, or compute based on this handoff.

## 17. Deliverable completeness and review request

This session created only this `HANDOFF.md`. It did not create `RUN_REQUEST.md` because no material compute was
requested or authorized, and it did not create `RESULTS.md` because no execution occurred. No split statistics,
assignment, manifest, checkpoint, result table, or paper source exists from S12.

Final Git verification immediately before handoff still reported the expected root and
`HEAD=f262f6bea037580065a8505008773c04fdd259f5`, an empty branch name (detached HEAD), and:

```text
$ git status --short --untracked-files=all
?? fl_v3/usenix27_orchestra/handoffs/S12/HANDOFF.md
```

Thus the only working-tree delta is the authorized S12 deliverable; no source/canonical/legacy file changed.

S12-R should independently challenge at least:

1. whether log-rooted connected components cover every configured camera/LiDAR/sweep/raw dependency;
2. whether the proposed train-only tail-selection sequence leaks official validation or permits attack-driven
   post-selection;
3. whether 50 historical train logs can support the proposed base/dev/tail/client/eval roles without an
   underpowered or artificial claim;
4. whether a synthetic regional/fleet proxy is stated honestly and whether full participation is defensible;
5. whether the update-scope recommendation is symmetric, frozen pre-tail, and capable of benign learning;
6. whether B3/B5-matched poisoning difference-in-differences, fixed/paired panels, false-trigger, eligibility
   matching, missing-token rules, and defense FPR prevent patch, denominator, or utility gaming;
7. whether every baseline/defense actually matches init, scope, precision, exposure, and participant schedule;
8. whether `VIS`/`SA` boundaries and overhead are accurate for every proposed defense;
9. whether BEV-FePNet, FedM2former, Fed3D, BadFusion, Attack of the Tails, BadPromptFL/SABRE-FL, FedCPA,
   adaptive layer-wise attacks, DoBlock, SAFELearning/RFLPA, and 2026 updates further narrow novelty;
10. whether each tentative paper claim has a falsifier, raw artifact, and ethical/release boundary.

**Gate verdict:** `NOT READY / REVIEW REQUIRED`. This is the expected outcome for an evidence/proposal-only
kickoff. There is no frozen split and no scientific PASS. Independent S12-R plus explicit owner decisions are
required before S13 can materialize or train under Protocol B.
