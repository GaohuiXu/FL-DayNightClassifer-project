# T4 — Build-session kickoff prompt (paste into a fresh Claude Code session)

You are the **build session for Cycle-04 task T4** of a thesis project on securing federated learning for
autonomous-driving perception. **T0–T3 are complete and Codex-PASSed** (the fl_v3 skeleton + determinism +
defenses; the nuScenes data module + log-group partitioner + V1; the deterministic BEVFusion-class model +
loss/decode + V2/V3; the real Flower/Ray FedAvg on the A40 + the platform milestone; 167 tests). **T4 is
the metric layer** — the official nuScenes `DetectionEval` (mAP/NDS) + the strict ASR-eligibility harness +
the benchmark-readiness gate + V4 — that every later attack×defense result (T5/T6/T7) is reported in.
**There is no attack yet** (the trigger is T5); T4 builds + validates the ASR machinery on clean data.

**Read first, in order:**
1. `fl_v3/docs/cycle_04/tasks/T4_SPEC.md` — **your contract. Read §0 FIRST** — the ASR *semantics* are
   correct, but the GATE has self-certifying / undeclared-threshold / partition-mismatch holes that §0
   closes (anchor to the devkit, bind readiness to the log-group checkpoint, pin numeric floors, denominator
   = eligible-clean-detected, hash + bind the frozen subset).
2. `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md` — the plan (T4 task entry + GATE, §Attack spec
   "ASR-eligibility" + "Evaluation protocol & splits", §Defense "utility/ASR 2×2", Viz V4).
3. `fl_v3/docs/cycle_04/decisions.md` — D4 (disappearance primary), D8 (car), D7 (δ deferred — T4 produces
   the clean NDS baseline), D9 (compute tiers for the reference run).
4. The seams you build on: `fl_v3/src/fl_v3/models/fusion/detector.py` (`decode` — the single decode),
   `fl_v3/src/fl_v3/training/tasks.py` (`center_distance_proxy` — "NOT the T4 DetectionEval"; the greedy
   matching primitive to reuse), `fl_v3/src/fl_v3/data/nuscenes/{transforms.py, conventions.md §3/§4/§6,
   info_cache.py}` (the verified forward to **invert**, the raw global annotation = your independent oracle).
5. Skim `fl_v3/collab/T3/SPEC.md §6` — the weak clean model (trainval **log_group** recall@2m **0.1455** /
   IID 0.3528 at 4 rounds) + the `FL_TRAINABLE_CHECKSUM` pattern + the D9 Path-A multi-GPU launcher.

**Your job:** execute T4 to its GATE — wire the official `DetectionEval` (mAP/NDS) onto the platform's
decoded boxes (a correct canonical→global submission conversion), build the 6-criterion ASR-eligibility
harness + the frozen held-out ASR subset + the disappearance metric + the false-disappearance baseline +
the denominator-N, establish a converged reference clean checkpoint + emit an honest benchmark-readiness
go/no-go for T5, and render V4 — all on **clean** data (no attack).

**Hard rules (the §0 non-negotiables):**
- **Anchor the canonical→global conversion to the RAW devkit annotation, NOT to T1's own forward** — a
  self-inverse round-trip certifies a shared yaw/`wlh`/ego bug (the T1-viz / T2-BEV trap). The GATE: ≥200
  real mini boxes match the raw devkit global annotation, AND the `DetectionEval` GT-as-pred AP≈1 sanity
  uses the **devkit's own `load_gt`** as the GT side. Reuse `transforms.*` so only the inverse is new.
  Mind the **velocity** inverse (order `R(ep)·R(cs)`, `v_z=0`, check direction) and the devkit submission
  rules (builtin `float()` score, no NaN/inf in trans/size/rot, `size=wlh`, `attribute_name` in the class
  vocab). Call `DetectionEval(...).evaluate()` **not** `main()` (which shuffles + writes PDFs).
- **Train the reference checkpoint at FULL participation (D10), then bind readiness to it.** Per **D10**
  (read `decisions.md`), the clean baseline runs at `fraction-train=1.0` (all N/round), NOT the T3 5-of-25
  sampling — that sampling is what made the log-group model look weak (recall@2m 0.1455 conflates
  partial-participation variance + ~0.8 selections/shard under-training; the same model hits 0.35–0.50
  IID). Build a `t4_reference.json` + Path-A launcher (fraction-train=1.0, log-group, trainval, rounds
  bumped to convergence; generalize `fl_stamp_supernodes` to stamp `local-simulation-gpu-4x` or use
  single-GPU). **Bind readiness to THAT full-participation log-group checkpoint by `FL_TRAINABLE_CHECKSUM`**
  (not IID, not a sampled checkpoint). The T3 `0.1455`/`+0.2073` are `scale=trainval-sampled` — not the
  anchor. **Pin defensible floors in config** (`N_min ≥ 150`; `recall_floor` on **official** car recall
  > 0.20, not the proxy; `d_clean=2.0`, `τ_clean`/`τ_pts` declared; false-disappearance `< 0.02` valid only
  when denominator ≥ N_min). **Strict sequence:** full-participation retrain FIRST → re-judge readiness →
  architecture strengthening (deeper PFN / full-model on A100 per D9) ONLY if STILL NOT-READY. Report the
  5-of-25 and full-participation recall side by side; do not lower a floor to pass.
- **The ASR denominator is `eligible-CLEAN-DETECTED`** (all 6 criteria incl. clean detect+match), not
  all-eligible — assert it + test a criteria-1–4-but-undetected target is excluded. **The frozen ASR subset
  is content-hashed + bound to the checkpoint checksum + thresholds**, built on the full `val` split, with
  a load-time re-verify T5 reuses.
- **Evaluator + V4 consume the SAME decoded boxes** (same thresholds, no V4 re-threshold); V4 agreement
  must include the lowest-scoring eligible boxes, not a high-score sample.
- **Mini = engineering smoke** (harness validated: evaluator runs, round-trip, false-disappearance
  near-zero); the **real mAP/NDS + readiness are trainval-`val`-scale** (≥ the declared scene/sample floor),
  `scale`-stamped — a mini_val (2-scene) verdict is NOT a go/no-go. Heavy eval runs go through SLURM; run
  code via `fl_v3/scripts/run_in_venv.sh`. Consume T2/T1/T0 unchanged (if T1 needs full-3D velocity, flag
  it — don't mutate T1).
- Write your SPEC to `fl_v3/collab/T4/SPEC.md` (the conversion convention + the **pinned** thresholds +
  attribute defaults + the mAP/NDS + readiness numbers + the frozen-subset hash + the attacked-checkpoint
  checksum), add the tests, drive the GATE to green, append to `findings_log.md`.

**When the GATE is green:** summarize what landed, paste the test count + the trainval mAP/NDS + the
benchmark-readiness verdict (READY/NOT-READY with eligible_count + official clean car recall + the pinned
floors) + the frozen-subset hash + the attacked-checkpoint checksum, list the 2–3 things the Codex reviewer
should scrutinize hardest (the SPEC predicts: the devkit-anchored round-trip + GT-as-pred AP≈1; the velocity
inverse; the eligible-clean-detected denominator; the readiness bound to the log-group checkpoint with
floors on official recall), and stop — the Codex review session reviews before T5 starts. **If the verdict
is NOT-READY, say so plainly** — that is a valid outcome that gates T5 and needs a model-strengthening
decision before the attack benchmark. Do not commit/push unless the user asks.
