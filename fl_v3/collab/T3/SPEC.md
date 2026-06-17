# T3 — SPEC: FL integration + clean FedAvg baseline (PLATFORM MILESTONE) + real Ray run on the A40

> Build-session copy. Contract: `fl_v3/docs/cycle_04/tasks/T3_SPEC.md` (read its **§0** first).
> Plan: task **T3** in `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`. Decisions: **D1**
> (frozen ImageNet camera backbone; FL-train the rest).

## 1. Scientific intent

Stand up the real federated loop end-to-end and prove the platform is a sound, reproducible
instrument: the T2 deterministic detector trained by **sequential single-actor FedAvg**
(`num-gpus=1.0`) on the T1 partitions, via Flower's Ray simulation on the **A40 (SLURM)**. The
load-bearing property is **full-loop bit-determinism** — two same-seed FedAvg runs (≥3 rounds,
sampled) produce a **byte-identical final global model** — delivered by the T0 machinery
(`derive_seed`, partition-id aggregation sort, single GPU actor) **plus the new deterministic
sampler (DT3-B)** on the heavy AD model + live Ray path. On top of it the milestone measures
**(a) IID-mini FedAvg ≈ centralized** (falsifiably) and **(b) the non-IID/geographic gap**
(a completed trainval-scale run). Clean FedAvg only — the baseline every attack×defense cell
later compares against.

## 2. Scope

**In scope (delivered):**
- **DT3-A — trainable-only update vector** (the 62 non-backbone param tensors) at the 4 seams
  (`initial_arrays`, client reply, server-eval load, final checkpoint), `strict=False` loads with
  clean-load asserts, `det-pretrained-backbone=True` required for a frozen backbone, final
  checkpoint = self-contained FULL model.
- **DT3-B — deterministic client sampler** (`strategy/sampling.py` + a one-time `@query` discovery
  probe) replacing Flower's `random.sample(get_node_ids())`; symmetric on the evaluate side with
  its own `min-evaluate-nodes`.
- **Hardened flwr-1.27 launcher** (`scripts/run_alvis.sh` + `scripts/_fl_env.sh`): `-A/-p`, per-job
  auto-SuperLink + Ray ports + tmp, `WANDB_MODE=offline`, offline preflight, silent-exit guard,
  derived N → `num-supernodes`; **no manual `flower-superlink`**.
- **FL bit-determinism gate** (`scripts/run_fedavg_a40.sh` + `scripts/fl_gate_a40.py`): `assert_a40`,
  two same-seed Ray runs → byte-identical final global, local_runner cross-check, substrate stability.
- **`local_runner.run_clean_rounds`** — the multi-round, sampled, trainable-only cross-check substrate.
- **Milestone measurements** (`scripts/t3_iid_vs_central.py` + the trainval run).

**Out of scope / deferred:** official mAP/NDS + ASR + V4 (T4); attacks/V5 (T5); defense-behavior
benchmark + per-module gradient logging + V6 (T6); controlled `m_r`/`f_r` (T5/T6); D7 `δ` (deferred,
a provisional δ is declared for the IID≈central check only).

**Files created/changed:** `strategy/sampling.py` (new), `strategy/flower_strategies.py` (DT3-B
discovery + deterministic configure_train/evaluate), `client_app.py` (trainable reply + strict
load + `@query`), `server_app.py` (trainable initial arrays + strict-load eval + full-model
checkpoint + `min-evaluate-nodes` + pretrained guard + FL checksum), `engine/local_runner.py`
(trainable-only + `run_clean_rounds`), `training/tasks.py` (`trainable_state_dict` /
`load_trainable_state_dict` / `assert_trainable_layout` / `TRAINABLE_MODULE_SLICE_MAP`),
`pyproject.toml` (`min-evaluate-nodes`), `scripts/{_fl_env.sh, run_alvis.sh (harden),
run_fedavg_a40.sh, fl_gate_a40.py, runconfig.py, t3_iid_vs_central.py, run_t3_milestone_a40.sh}`,
`configs/t3_fl_gate.json`, `tests/test_fl_{sampling,trainable_only,local_runner_multiround,gate_refuses_non_a40}.py`.
**Consume-only:** T2 `models/fusion/**` + `scripts/det_gate_a40.assert_a40`, T1 `data/nuscenes/**`,
T0 `strategy/defenses/**` + `utils/runtime.py`. `fl_v2/` untouched.

## 3. The frozen DT3-A update-vector layout contract (T3→T6)

The FL update vector = the model's **trainable** tensors only (`requires_grad`-filter, NOT a
prefix-drop). For the headline detector this is **exactly 62 param tensors**, in this module order
with these per-module counts (asserted by `assert_trainable_layout` + `test_fl_trainable_only.py`):

| module | trainable tensors |
|---|---:|
| camera_neck | 15 |
| view_transform | 5 |
| lidar_encoder | 3 |
| fusion | 6 |
| bev_neck | 18 |
| head | 15 |
| **TOTAL** | **62** |

- **All frozen *params* live under `camera_backbone`** (0 outside) and are EXCLUDED + reconstructed
  byte-identically per node (`pretrained=True`).
- **Buffers are excluded.** There are **4 non-backbone buffers** (`preprocess._mean/_std`,
  `view_transform.frustum/depth_values`) — VERIFIED node-invariant constants AND static through
  training, so excluding them is lossless (each node rebuilds them from config). The 6 trainable
  modules use GroupNorm (D6), no running-stat buffers. *(This refines the T3_SPEC claim that "all
  buffers live in the backbone"; see findings_log.)*
- Gradient-space metrics are trainable-only by construction (they read `current_arrays` + the
  trainable replies — no mask added).

## 4. Invariants (must hold; Codex checks each)

- **Full-loop bit-determinism (crown jewel):** two same-seed FedAvg runs (≥3 rounds, fraction<1,
  real model) → byte-identical final global on the A40; the `local_runner` multi-round
  same-sampler two-run checksum is identical (same-A40 cross-check). [A40 GATE — see §6]
- **D1 + DT3-A:** only the 62 trainable tensors aggregated; frozen backbone reconstructed
  byte-identically across clients AND server (`pretrained=True`, cross-node test); final checkpoint
  is a self-contained full model (loads `strict=True`).
- **DT3-B:** participant set drawn over the fixed `0..N-1` partition-id space via
  `derive_seed(seed, SALT, round)`; byte-identical across drivers at fraction<1; no selection flows
  through Flower's `random.sample(get_node_ids())`.
- **Substrate stability:** per-round participant set + `norm_log` byte-reproducible at fraction<1.
- **Falsifiable IID-mini≈central:** both sides clear `R_floor>0` + anti-collapse + recall agreement
  within `δ`.
- **Mini vs trainval boundary:** mini = engineering smoke; the non-IID gap number is a completed
  trainval-scale run; ≤20 rounds.
- **Task-agnostic preserved:** `dummy_regression` FL still deterministic after DT3-A/DT3-B
  (trainable filter degenerates to identity on TinyMLP; `test_fl_round_smoke` + the multi-round
  tests pass).

## 5. Scientific failure modes checked (point Codex here)

- Determinism passing on a trivial config (1 round / fraction=1 / CPU / Tesla-T4) — the gate is
  A40 + ≥3 rounds + fraction<1 + real model + `assert_a40` (test `test_fl_gate_refuses_non_a40`).
- Sampling non-determinism — replaced by the DT3-B partition-id sampler (test
  `test_configure_train_same_pids_across_drivers_at_fraction_lt_1`).
- Frozen-backbone divergence under `pretrained=False` — required `pretrained=True` + cross-node
  identity test; server guard `_require_reconstructible_frozen_backbone`.
- Partial-load crash / silent wrong-weights — `strict=False` + clean-load asserts; full-model merge.
- IID-mini "matching" central as a zero-decode collapse — `R_floor` + anti-collapse asserted.
- Hand-waved trainval run — the gap is a COMPLETED trainval-scale run with measured wall-clock.
- SLURM/Ray startup — `-A/-p`, per-job ports/tmp, silent-exit guard, offline preflight.
- Cross-check is a scalar — the cross-check compares the aggregated-WEIGHT checksum, not a loss.

## 6. GATE — status

- [x] **Code + login tests green:** `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests`
      → **167 passed** (T0+T1+T2's 148 + 19 new T3). New: `test_fl_sampling` (5),
      `test_fl_trainable_only` (6), `test_fl_local_runner_multiround` (4),
      `test_fl_gate_refuses_non_a40` (3), +1 (counts: 5+6+4+3=18; the +1 is the discovery-coverage
      case). The A40 SLURM gate is NOT in pytest.
- [x] **DT3-A:** trainable-only 62-tensor vector; `strict=False` loads (asserted key sets); frozen
      backbone byte-identical across nodes under `pretrained=True` (diverges under `False`); final
      checkpoint loads `strict=True`; per-module slice map frozen (§3).
- [x] **DT3-B deterministic sampling:** same partition-ids across two fake drivers (different
      node_ids) at fraction<1; Flower's `sample_nodes` monkeypatched-to-raise and the path still works.
- [x] **In-process detection FL path** validated end-to-end on the real detector (CPU, two-run
      byte-identical). [`/tmp/det_fl_smoke.py` — see §6.1]
- [x] **A40 FL bit-determinism gate** (`sbatch fl_v3/scripts/run_fedavg_a40.sh`, job **6764008**,
      NVIDIA A40): two same-seed Ray runs → **byte-identical `FL_TRAINABLE_CHECKSUM`**, 3 rounds,
      fraction-train=0.5 (4/8 clients), real `nuscenes_detection` model. **OVERALL: PASS.** DT3-B
      participants identical across runs (r1=[1,2,3,4], r2=[0,1,4,6], r3=[0,2,3,6]); per-round server
      eval byte-identical across runA/runB. `assert_a40`-guarded (test: exit-2 off the A40).
- [x] **IID-mini ≈ central (falsifiable)** (`sbatch fl_v3/scripts/run_t3_milestone_a40.sh`, job
      **6763999**, A40): **PASS** — recall@2m fed **0.4961** vs central **0.4653** (gap **0.0308** <
      δ); both clear `R_floor=0.05`; anti-collapse both sides; all 5 checks True.
- [x] **Non-IID gap (completed trainval-scale)** (`sbatch fl_v3/scripts/run_trainval_gap_a40.sh`, job
      **6764226**, A40, COMPLETED): headline frozen **Swin-T** on v1.0-**trainval**, derived **N=25**,
      fraction-train=0.2 (5/25), **4 rounds** (≤20), batch=16, run to completion both modes.
      **Authoritative gap (re-eval on the FULL val split, 6019 samples, `det-eval-limit=0`, job
      6764280 — the Codex-F1 fix):** recall@2m **IID=0.3528** vs **log_group=0.1455** → **non-IID gap
      = +0.2073** (`scale`-stamped `trainval-scientific`, reported, NOT required small — and it is not).
      *(The in-training server eval was a fixed 256-sample subset `sorted(sample_token)[:256]` giving
      IID 0.369/log_group 0.146/gap +0.224 — same direction + magnitude, so the gap is robust to eval
      scope.)* Measured wall-clock ≈ **3088 s (IID) / 3113 s (log_group)** ≈ 13 min/round at batch 16
      (the headline Swin-T number). Final-model checksums: IID `a4408cdb…`, log_group `47635bb6…`. The
      mini log-group run is methodology smoke only.

### 6.1 Declared thresholds
- **R_floor = 0.05** (absolute recall@2m, car; >0 — a zero-decode collapse fails).
- **δ = 0.15** (absolute recall@2m agreement, IID-mini FedAvg vs central) — provisional, this check
  only; the benchmark D7 `δ` is deferred to T6/T7.

### 6.2 A40 artifacts
- **FL gate** device = **NVIDIA A40**, job **6764008**, elapsed 488 s (2 local_runner + 2 Ray runs).
- **`FL_TRAINABLE_CHECKSUM` runA == runB ==**
  `d82ef5001b88bd157161fe7b3eb658a9493fd169b4be28d3b5d3d9c34c08b236` (byte-identical — crown jewel).
- **`LOCAL_RUNNER_CHECKSUM` ==** the same string — the login-node↔Ray cross-check is **byte-identical
  on the same A40** (not just allclose).
- **Substrate** (`norm_log.json`): byte-identical across runA/runB (participant set + cosine/energy
  arrays) — PASS.
- **Gate eval curve** (both runs identical): eval_loss 12.42 → 6.08 → 4.10 → 3.61 (r0→r3);
  recall@2m 0 → 0 → 0.086 → 0.204; n_decoded 0→442 (the loop trains, anti-collapse).
- **Final checkpoint (real artifact):** `t3_gate_runA/final_model.pt` loads `strict=True` into a fresh
  full model (182 state_dict entries) — the DT3-A self-contained full-model checkpoint, confirmed on
  an actual A40 run output (not just the unit test).
- **IID-mini ≈ central** (job 6763999, 51 min on mini, N=4 IID, 15 rounds): recall@2m fed
  **0.4961** / central **0.4653** / |gap| **0.0308**; checks all True.
- **Measured wall-clock**: bring-up resnet18 gate (N=8, 3 rounds) — Ray run ≈ 2–3 min/run incl.
  bring-up; mini milestone (N=4, 15 rounds, num_workers=0, I/O-bound) — 3072 s. Headline Swin-T
  per-round wall-clock: **from the trainval run (§ below) when complete.**
- **Trainval-scale non-IID gap** (job 6764226, A40, COMPLETED, headline frozen **Swin-T**, derived
  **N=25**, fraction-train=0.2 → 5/25, **4 rounds**, batch=16, v1.0-trainval train/val): authoritative
  recall@2m on the **FULL val split** (6019 samples; re-eval job 6764280, Codex-F1 fix)
  **IID=0.3528** / **log_group=0.1455** → **non-IID gap +0.2073**; per-run wall-clock
  **IID 3088 s / log_group 3113 s** (≈ 13 min/round at batch 16); final checksums IID `a4408cdb6b…`,
  log_group `47635bb66f…`. *(In-training eval used a fixed 256-sample subset → +0.2235; full-val
  re-eval gives +0.2073 — robust.)* *(Earlier batch=4 attempt 6764191 cancelled to switch to the faster
  batch=16/fraction-0.2 config; 6764225 hit the documented Ray same-node bring-up race — caught by
  the silent-exit guard — and was resubmitted on a fresh node as 6764226.)*
- **Parallelism follow-up (T5–T7 speedup; see decisions.md D9):** both parallelism methods are
  **byte-identical to the single-actor reference `d82ef500…`** (atomic-free model ⇒ interleaving
  changes timing, not values):
  - **Path A — multi-GPU** (`local-simulation-gpu-4x`, `num-gpus=1.0`, 1 client/GPU): jobs
    6764253/6764255 — PASS-STRONG. *Multiplies compute ⇒ ~N× wall-clock (the real per-cell speedup).*
  - **Path B — concurrent / shared-GPU** (`local-simulation-gpu-shared`, `num-gpus=0.25`, N actors/GPU):
    job 6764256 — PASS-STRONG. *Shares one GPU ⇒ gap-filler only; ≈ no gain when one client already
    saturates the GPU (headline Swin-T batch≥16 ~100% SM).*
  Recommendation: matrix → across-cell fan-out, 1 GPU/cell (+ optional Path B to hide latency); single
  heavy run → Path A. Validated at bring-up scale; re-confirm + measure speedup at Swin-T/trainval
  scale before full T5–T7 reliance. Harness: `run_parallel_validation_a40.sh`.

## 7. Self-review — what to attack hardest (for Codex)
1. **DT3-B sampler truly replaces Flower's random selection at fraction<1** — the discovery probe
   (`@query`) + `select_partition_ids` over `range(N)`; participant set byte-identical across drivers
   (`test_configure_train_same_pids_across_drivers_at_fraction_lt_1`). Scrutinize: the discovery
   coverage assert (`num-supernodes==N`), and that no path falls back to `random.sample(node_ids)`.
2. **Full-loop A40 bit-identity at ≥3 rounds + fraction<1 on the real model** — the committed
   `FL_TRAINABLE_CHECKSUM` (runA==runB), via the `assert_a40`-guarded gate (not a CPU/T4 pass).
3. **Trainable-only transmit + pretrained frozen-backbone cross-node identity** — the 62-tensor
   layout, `strict=False`/merge plumbing, and the 4-non-backbone-constant-buffers nuance (§3 +
   findings_log) being genuinely lossless.
4. **IID≈central falsifiable + the trainval gap a completed run** — `R_floor`/anti-collapse, and the
   trainval number coming from a finished trainval(-subset) Swin-T run (not a projection).
