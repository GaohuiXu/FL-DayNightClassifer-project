# T3 — SPEC: FL integration + clean FedAvg baseline (PLATFORM MILESTONE) + real Ray run on the A40

Plan: `../../roadmap/cycle_04_fusion_layer_backdoors.md` (task **T3**; §FL setup, §determinism "How
enforced per scope", Architecture). Decisions: `../decisions.md` — **D1** (frozen ImageNet camera
backbone; FL-train LSS-depth + LiDAR-enc + fusion + neck + head; full-model FL = later ablation).
Contract for the **T3 build session**. Fill `fl_v3/collab/T3/SPEC.md` from the template.

> **This gate = "the platform works."** T3 is where the deterministic model (T2), the nuScenes data +
> log-group partitioner (T1), and the task-agnostic FL skeleton + FedAvg + determinism machinery (T0)
> first run as a **real Flower/Ray sequential FedAvg on the A40 via SLURM** — proven **bit-identical
> across two same-seed runs**, with the IID-mini≈central check, the measured non-IID gap, and acceptable
> wall-clock. No attack, no defense-behavior study yet (T5/T6) — just the clean baseline.

> **The Ray/Flower/determinism design below was verified against the installed flwr 1.27 + ray 2.51.1
> source and the actual model state_dict** (a 5-agent pass, workflow `wf_7ea15066-8db`). It overturned
> three of my first-draft assumptions — **read §0 first.**

---

## 0. CRITICAL realities (read before scoping) — three first-draft assumptions were wrong

1. **flwr 1.27 AUTO-MANAGES the local SuperLink — do NOT carry the fl_v2 SuperLink/SuperExec startup-race
   waits.** `flwr run . local-simulation-gpu` is correct, but in 1.27 `flwr run` is only a Control-API
   client that *itself* auto-starts + manages the local SuperLink (`cli/local_superlink.py
   ensure_local_superlink`, with its own readiness probe) and runs the whole simulation (ServerApp + Ray
   actor pool) in **one** auto-spawned `flwr-simulation` subprocess. There are **no** manual
   `flower-superlink` / `flwr-serverapp` / `flwr-supernode` processes to launch or grep-wait on.
   Manually starting `flower-superlink` (the fl_v2 pattern) **collides** on the default ports and
   re-creates the silent race. **The ONLY fl_v2 SuperLink hardening that still applies:** per-job
   `FLWR_LOCAL_CONTROL_API_PORT` / `FLWR_LOCAL_SIMULATIONIO_API_PORT` (the CLI binds its auto-SuperLink
   there; default 39093/39094 collide across concurrent same-node jobs).
2. **Client SAMPLING is non-deterministic by construction — it must be HARD-overridden, not gated away.**
   Flower's `FedAvg.configure_train` selects participants via `random.sample(list(grid.get_node_ids()),
   …)` over node_ids built from **random public-key bytes**, so two same-seed drivers pick **different
   subsets** at `fraction-train < 1` even though the global RNG is seeded; `FedAvg` has no seed arg.
   fl_v3's current `configure_train` override only stamps `server-round`. **DT3-B is therefore a required
   strategy-level override** (a deterministic selector over the fixed `0..N-1` partition-id space), NOT
   "run the gate at fraction=1.0." The fraction=1.0 escape is a FALSE PASS — it never exercises the
   sampling drift the gate exists to catch, and T5/T6 will run at fraction<1.
3. **The determinism gate must be A40 + ≥3 rounds + fraction<1 on the REAL model.** Same-seed-twice on
   one GPU at `num-server-rounds=1, fraction-train=1.0, all clients` is bit-identical *even with a latent
   non-determinism* — it never exercises per-round sampling drift or the fl_v2 "diverges at round 2"
   carry-over. The FL bit-identity gate must reuse `det_gate_a40.assert_a40` (loud exit-2 on
   non-A40/CPU), run **≥3 rounds at fraction-train<1 on the `nuscenes_detection` model**, and commit the
   checksum. A login-node (Tesla T4) / CPU pass is a FALSE PASS.

> **Orchestrator decisions (CONFIRMED 2026-06-16 — locked, do NOT re-litigate):**
> - **DT3-A — update-vector = trainable-only (CONFIRMED; adopt as primary).** The FL apps
>   currently transmit the **full** `model.state_dict()` (`client_app.py:84`), so the frozen 27.5M-param
>   backbone (94% of params, zero update) rides along every round, wasting ~16× serialization AND
>   **contaminating the gradient-space metrics** (cosine/norm over all params → swamped by 94% zeros) —
>   the substrate T6 + Q2 consume. **Verified clean:** the trainable vector is exactly the **62
>   non-backbone param tensors**; ALL frozen params + ALL buffers live inside `camera_backbone`; the 6
>   trainable modules use GroupNorm (D6) with zero buffers — so trainable-only loses nothing. Adopt it
>   (the fallback "full transmit + mask the numpy cores" is *more* invasive and still pays serialization).
> - **The metric at T3 is the T2 provisional proxy, NOT mAP/NDS** (official `DetectionEval` is **T4**):
>   "IID-mini ≈ central" and "the non-IID gap" use the **T2 center-distance proxy (recall@2m, car) +
>   val-loss**; the official-metric comparison is re-run at T4+.

---

## 1. Scientific intent

Stand up the **real federated training loop** end-to-end and prove the platform is a sound, reproducible
instrument: the T2 deterministic detector trained by **sequential single-actor FedAvg** (`num-gpus=1.0`)
on the T1 log-group/IID client partitions, via Flower's Ray simulation on the **A40 (SLURM)**. The
load-bearing property is **full-loop bit-determinism** — two same-seed FedAvg runs (≥3 rounds, sampled)
produce a **byte-identical final global model** — which the T0 machinery (per-client/per-round
`derive_seed`, `partition-id` aggregation sort, single GPU actor) plus the **new deterministic sampler
(DT3-B)** must deliver on the heavy AD model + live Ray path. On top of that the milestone measures **(a)
IID-mini FedAvg ≈ centralized** (falsifiably — the loop truly trains, not a zero-decode collapse) and
**(b) the non-IID/geographic gap** (location-coherent log-group vs IID — *measured, not required small*;
the Q2-heterogeneity substrate, established here, analyzed at T7). Clean FedAvg only — the **baseline
trajectory** every later attack×defense cell compares against; mini is the engineering proof, a
**trainval-scale run** produces the real gap + wall-clock.

## 2. Scope

**In scope (deliver):**

### 2.1 The hardened SLURM launcher (`run_alvis.sh` + a `run_fedavg_a40.sh`) — flwr-1.27-correct
- **SBATCH header MUST include `-A NAISS2025-22-1113` and `-p alvis`** (mirror `run_det_gate_a40.sh`) or
  the job is rejected at submit; `--gpus-per-node=A40:1`; `--time` budgeted for **two** same-seed runs
  (T2: ~3.1 min/round headline *train-only*, excl. per-round eval + Ray bring-up; 4 h is safe for
  2×≤20 rounds).
- **Wire `flwr run . local-simulation-gpu`.** Do **NOT** manually launch `flower-superlink` and do **NOT**
  carry fl_v2's SuperExec grep/sleep waits (§0.1) — the CLI auto-manages the SuperLink.
- **Per-job isolation env (the real hardening; all reach `ray.init`/the CLI via env inheritance):**
  `FLWR_HOME=/tmp/flwr_$SLURM_JOB_ID` (present) + per-job **`FLWR_LOCAL_CONTROL_API_PORT` /
  `FLWR_LOCAL_SIMULATIONIO_API_PORT`** (derived from `$SLURM_JOB_ID`, as fl_v2:63-68) + per-job **Ray
  ports** `RAY_GCS_SERVER_PORT / RAY_DASHBOARD_PORT / RAY_NODE_MANAGER_PORT / RAY_OBJECT_MANAGER_PORT /
  RAY_RUNTIME_ENV_AGENT_PORT` + **`RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID`** (cleaned on the EXIT trap). Keep
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `TORCH_HOME` (present). Add `RAY_DEDUP_LOGS=0` + `PYTHONHASHSEED=0`
  (cheap defense-in-depth).
- **Offline-safe:** export **`WANDB_MODE=offline`** (wandb is a hard dep and will block on egress on an
  offline compute node); assert `TORCH_HOME` weights + the nuScenes info-cache are pre-built; the run
  makes **zero network calls**.
- **Silent-exit guard (carry fl_v2:249-263 — the single most valuable hardening):** `flwr run` returns
  **exit 0 even on a zero-round run** (the documented Ray-under-SLURM fail mode). Stream the run, assert
  a **completion/round-count marker** (the ServerApp emits the final-round summary / the committed
  checksum line) before exiting 0; absence → **force non-zero exit**. A zero-round run that exits 0 is a
  FALSE PASS for both the determinism gate and IID≈central.
- Stamp the **derived N** (not the placeholder 50) into `num-supernodes`.

### 2.2 DT3-B — the deterministic client sampler (REQUIRED override)
Override the strategy so the per-round participant set is drawn by `random.Random(derive_seed(seed,
"sample", server_round)).sample(range(N), n_r)` over the **fixed `0..N-1` partition-id space** (recommend
`n_r` = 8–12 via `fraction-train`), then mapped chosen-partition-id → node_id via the node_config
`partition-id`; **no selection may flow through Flower's `random.sample(grid.get_node_ids())`.** Keep the
`m_r`/`f_r` threat-model split out of scope (T5/T6) but the seam compatible (sampling is `derive_seed`-
driven, not Flower RNG). Same for the evaluate side; **read `min-evaluate-nodes` from its own key** (the
current `server_app.py:126` reuses `min-train-nodes`).

### 2.3 DT3-A — trainable-only update vector + the T3→T6 layout contract
- A **`requires_grad`-based** filter (NOT a `camera_backbone` prefix-drop):
  `trainable_state_dict(model) = {k:v for k,v in model.state_dict().items() if k in {n for n,p in
  model.named_parameters() if p.requires_grad}}`. Apply it at the **4 seams**: `initial_arrays`
  (`server_app.py:133`), the client reply `ArrayRecord` (`client_app.py:84`), the server-eval load, the
  final checkpoint. **Loads use `strict=False`** (asserting `unexpected_keys==[]` and `missing_keys ⊆
  backbone keys` — a partial 62-key dict into the full model RAISES under the current `strict=True`).
- **Frozen backbone reconstructed byte-identically per node** — verified true **only for
  `pretrained=True`** (cached ImageNet weights). **REQUIRE `det-pretrained-backbone=True` for ALL FL runs
  + the FL gate;** if the non-pretrained path is ever used, seed the frozen reconstruction from a
  **node-invariant** constant (`derive_seed(seed, "frozen_backbone")`, no `client_id`/round), because
  `build_model` runs *after* the per-client `derive_seed` (`client_app.py:60→65`) so a random-init
  "frozen" backbone otherwise **differs per node** and corrupts eval + the no-op-aggregation claim.
- **Final checkpoint = a self-contained FULL model:** merge aggregated-trainable into a freshly-built
  full model (frozen backbone reconstructed) and `torch.save` the **full** state_dict, so `final_model.pt`
  loads `strict=True` for T4. Server-eval loads trainable-only into its cached full model (built
  `pretrained=True`).
- **Gradient metrics need NO mask** — they read `current_arrays` + the replies, so trainable-only inputs
  ⇒ trainable-only metrics structurally (do not add redundant masking to the numpy cores).
- **Freeze the per-module update-vector layout (the T3→T6 contract):** the 62 ordered trainable tensors,
  module order `camera_neck → view_transform → lidar_encoder → fusion → bev_neck → head`, per-module
  counts **`{camera_neck:15, view_transform:5, lidar_encoder:3, fusion:6, bev_neck:18, head:15}`** —
  declared + asserted in `collab/T3/SPEC.md` so T6 defenses + Q2 dilution slice correctly.
- **`local_runner` applies the SAME filter** (a shared helper) so the in-process proxy aggregates the
  identical 62-tensor vector as Ray (else the cross-check compares different vectors); `dummy_regression`
  is unaffected (the `requires_grad` filter degenerates to identity on TinyMLP).

### 2.4 The milestone runs + measurements
- **IID-mini FedAvg ≈ centralized (falsifiable):** FedAvg on the IID mini partition vs T2 central, in the
  **T2 proxy (recall@2m, car) + val-loss**, with a **declared ABSOLUTE recall floor `R_floor` (>0) both
  sides clear** + an **anti-collapse** assertion (`n_decoded > 0`, `matched_gt > 0`) + agreement on
  **recall** within `δ` (loss reported as secondary, never the sole match) — so a zero-decode dead loop
  (recall 0 ≈ 0) FAILS. Report the curve.
- **Non-IID gap measured (trainval-scale, a COMPLETED artifact):** a **trainval-scale FedAvg run** —
  frozen Swin-T headline config, derived N, fraction<1, ≤20 rounds, run to **completion** on the A40 —
  commits **(i)** the measured per-round + total wall-clock **from that run** (NOT a ResNet projection),
  **(ii)** the IID-vs-log-group gap in `recall@2m`, **(iii)** the run's final-model checksum;
  `scale`-stamped `trainval-scientific`, **reported, NOT required small**. *(If `v1.0-trainval` is
  unavailable, a sufficiently-large fixed trainval subset is allowed per plan §Verification — record
  which.)* The mini log-group run (≤6 clients) is **methodology smoke only** and cannot stand in.
- **Wall-clock acceptable:** real per-round + total on the A40; ≤20 rounds; confirm the sequential
  single-actor + sampling + frozen-backbone mitigations fit the SLURM budget.

### 2.5 The bit-identity gate + cross-check
- **FL bit-identity gate (`run_fedavg_a40.sh` / a gate script):** **import + call
  `det_gate_a40.assert_a40()`** (loud exit-2 on non-A40/CPU), run **two same-seed FedAvg runs at
  num-server-rounds ≥3 AND fraction-train<1 on the `nuscenes_detection` model** → byte-identical final
  global model; **commit both checksums** (annotated with the A40 device name). A unit test
  (`test_fl_gate_refuses_non_a40`) monkeypatches `get_device_name→"Tesla T4"` / no-CUDA and asserts
  exit-2.
- **Login-node↔Ray cross-check:** `local_runner` (extended to a **`run_clean_rounds(num_rounds,
  fraction_train)`** loop with the **same deterministic sampler**) and the Ray run produce a
  **byte-identical aggregated-WEIGHT checksum** (`numpy_state_checksum`, not a scalar loss) on a tiny
  shared config (≥2 rounds, sampled). Run the cross-check **on the same A40** for byte-identity (CPU↔A40
  float drift otherwise) — or assert tight `allclose` + identical participant/order metadata, documented.
- **Substrate stability:** the per-round participant partition-id set + `norm_log.json` (cosine/energy
  arrays, FoolsGold-relevant history) are **byte-reproducible across two same-seed runs at fraction<1**
  (commit both, assert identical) — so the T3→T6 substrate the layout contract freezes is itself stable.

**Out of scope / deferred:** official `DetectionEval` mAP/NDS + 6-criterion ASR + V4 (**T4**); attacks/V5
(**T5**); the **defense-behavior** benchmark + assumption cards + per-module gradient *logging* + V6
(**T6** — T3 only confirms the clean FedAvg path + freezes the update-vector layout); controlled `m_r`/
`f_r` (**T5/T6**); full-model FL ablation; the Q2 IID-vs-skew-vs-location *analysis* (**T7**); D7 `δ`
(DEFERRED). 

**Files created/changed:** `client_app.py`/`server_app.py` (trainable-only filter + strict=False +
merge/checkpoint), `strategy/flower_strategies.py` (the DT3-B deterministic sampler override;
`min-evaluate-nodes` key), `engine/local_runner.py` (multi-round loop + same sampler + trainable filter),
`training/tasks.py` (a `trainable_state_dict` helper / sampler hook if cleaner there), `scripts/{run_alvis.sh
(harden), run_fedavg_a40.sh (new)}` + the FL gate, `configs/flwr_config.toml` + `pyproject.toml`
(`num-server-rounds`, `fraction-train`, `min-evaluate-nodes`, `det-pretrained-backbone`), `tests/test_fl_*.py`,
`collab/T3/SPEC.md`. **Consume-only:** T2 `models/fusion/**` (+ `scripts/det_gate_a40.assert_a40` reused),
T1 `data/nuscenes/**`, T0 `strategy/defenses/**` + `utils/runtime.py`. `fl_v2/` untouched (read
`fl_v2/run_alvis.sh` only as the *port-derivation* oracle — NOT the SuperLink-wait pattern).

## 3. Invariants (must hold; Codex checks each)

- **Full-loop bit-determinism (crown jewel):** two same-seed FedAvg runs (**≥3 rounds, fraction<1, real
  model**) → **byte-identical final global model on the A40** (committed checksum, via
  `assert_a40`-guarded gate) AND the `local_runner` multi-round same-sampler two-run checksum is identical
  (same-A40 cross-check). Holds under: single Ray actor (`num-gpus=1.0`), `partition-id` aggregation sort,
  per-client/per-round `derive_seed`, the **DT3-B deterministic sampler** (participant set byte-identical
  across drivers), deterministic DataLoader worker seeding, and the T2 banned-op contract inside local
  training.
- **D1 + DT3-A:** only the 62 trainable tensors are aggregated; the frozen ImageNet backbone is
  reconstructed **byte-identically across all clients AND the server** (`pretrained=True`; test
  cross-node identity) and **excluded** from the update vector; the final checkpoint is a self-contained
  full model.
- **Update-vector layout frozen (T3→T6):** the ordered 62 tensors + the `{camera_neck:15,
  view_transform:5, lidar_encoder:3, fusion:6, bev_neck:18, head:15}` slice map are asserted; gradient
  metrics are trainable-only (inherited, no mask).
- **Substrate stability:** per-round participant set + `norm_log` byte-reproducible at fraction<1.
- **The loop truly federates + trains (falsifiable):** IID-mini FedAvg clears `R_floor` (>0) both sides +
  anti-collapse + recall-agreement within `δ` — not a zero-decode/no-op/diverging loop.
- **Mini vs trainval boundary:** mini = engineering smoke (incl. the mini log-group methodology smoke);
  the **non-IID gap number is a completed trainval-scale run**, `scale`-stamped; no scientific claim on
  mini. ≤20 rounds.
- **Task-agnostic preserved:** the `dummy_regression` CPU FL smoke runs + stays deterministic after
  DT3-A/DT3-B.
- **No false oracle:** clean FedAvg is the platform **baseline**; correctness = determinism + falsifiable
  IID≈central + the measured gap + wall-clock.

## 4. Reference (ground truth for the review)

- **Determinism contract:** `fl_v3/docs/determinism.md` ("How enforced per scope" + the §T2 banned-op
  reality — the model's contract holds inside FL local training too).
- **flwr 1.27 (verified):** `flwr run` auto-manages the SuperLink (`cli/local_superlink.py
  ensure_local_superlink`, ports `FLWR_LOCAL_CONTROL_API_PORT`/`…SIMULATIONIO…`); `FedAvg.configure_train`
  samples via `random.sample(grid.get_node_ids())` (`strategy_utils.py:189`) over random-pubkey node_ids
  (the DT3-B target); the simulation backend `ray.init(**{})` honors `RAY_*` env (`raybackend.py`);
  `num-gpus=1.0` ⇒ 1 actor (`ray_actor.py`). **fl_v2/run_alvis.sh** = the *port-derivation + silent-exit
  guard* oracle ONLY (its manual `flower-superlink`/SuperExec-wait is stale).
- **FL seams (T0):** `client_app.py` (per-call seeding, `reply-meta/partition-id`, the
  `ArrayRecord(model.state_dict())` → trainable-only), `server_app.py` (`initial_arrays`, `_server_eval_fn`,
  checkpoint), `strategy/flower_strategies.py` (`NormTrackingFedAvg`, `partition_sort_key`, the
  `global_keys==reply_keys` guard that *protects* DT3-A, gradient logging), `engine/local_runner.py`
  (`run_clean_round` → multi-round; `numpy_state_checksum`), `configs/flwr_config.toml`,
  `scripts/{run_alvis.sh, det_gate_a40.py (assert_a40 + wall_clock), run_det_gate_a40.sh}`.
- **Model + task (T2):** `NuScenesDetectionTask` (`evaluate`=center-distance proxy + loss; `valloader=None`
  for the AD task — utility is server-side), `detector.param_table()`, the T2 wall-clock table + the A40
  determinism-gate pattern.
- **Data/partition (T1):** `partition.{build_log_group_partition, iid_sample_partition}`,
  `info_cache.load_cache`, the official splits.

## 5. Scientific failure modes to check (point Codex here)

- **Determinism passes on a trivial config** (1 round / fraction=1.0 / all clients / CPU / Tesla T4) and
  drifts on the real A40 ≥3-round sampled run — the §0.3 trap. The gate must be A40 + ≥3 rounds +
  fraction<1 + real model + `assert_a40`.
- **Sampling non-determinism** (§0.2): Flower's `random.sample(get_node_ids())` over random node_ids →
  non-reproducible participant set → broken bit-identity AND a run-dependent gradient/FoolsGold substrate.
  Must be the DT3-B partition-id-space sampler.
- **Frozen-backbone divergence:** the no-op-aggregation / eval-correctness claim breaks under
  `pretrained=False` (per-node seed-init) — require `pretrained=True` (or a node-invariant frozen seed)
  and assert cross-node byte-identity.
- **Partial-load crash / silent wrong-weights:** `strict=True` on a 62-key dict RAISES; a trainable-only
  final checkpoint can't `load_state_dict(strict=True)` at T4 — pin `strict=False` + the full-model merge.
- **IID-mini "matches" central as a zero-decode collapse** (recall 0 ≈ 0, loss ≈ loss) — the
  T2-overfit-collapse analog; require `R_floor` + anti-collapse + recall-agreement.
- **Hand-waved trainval run:** reporting a ResNet bring-up projection instead of a completed trainval
  Swin-T run; the degenerate mini log-group standing in for the gap.
- **SLURM/Ray startup:** missing `-A/-p` (won't submit); default Ray/SuperLink ports colliding across
  concurrent jobs; `flwr run` exit-0 on a zero-round run (silent-exit guard); wandb egress on an offline
  node.
- **Cross-check is a scalar:** comparing `local_runner` vs Ray on `eval_loss` not the aggregated-weight
  checksum lets a Ray-only weight divergence slip.

## 6. GATE (objective pass criteria — plan's T3 gate, made objective)

- [ ] **Real Flower/Ray FedAvg on the A40 (SLURM):** `run_alvis.sh` hardened — `-A NAISS2025-22-1113`/`-p
      alvis`, `flwr run` wired (no manual SuperLink), per-job `FLWR_LOCAL_*`/`RAY_*` ports + `RAY_TMPDIR`,
      `WANDB_MODE=offline`, the **silent-exit guard** (round-count marker or non-zero exit), derived N →
      `num-supernodes`; a clean `nuscenes_detection` FedAvg run completes.
- [ ] **Two same-seed FedAvg runs → byte-identical final global on the A40, at ≥3 rounds + fraction<1 on
      the real model;** the gate **reuses `det_gate_a40.assert_a40`** (test: it exits-2 on non-A40/CPU);
      commit both checksums + the device name. The `local_runner` multi-round same-sampler two-run
      checksum matches on the same A40.
- [ ] **DT3-B deterministic sampling:** a test runs `configure_train` twice (different driver ⇒ different
      node_ids) and asserts the selected **partition-id** sets are identical per round **at
      fraction<1** (the fraction=1.0 case is NOT sufficient); no selection flows through Flower's
      `random.sample(get_node_ids())`.
- [ ] **DT3-A:** trainable-only (62-tensor) update vector; `strict=False` loads (asserted key sets);
      frozen backbone reconstructed **byte-identically across clients + server** (`pretrained=True`
      asserted); final checkpoint is a self-contained full model (loads `strict=True`); gradient metrics
      trainable-only; the per-module slice map frozen + documented (T3→T6 contract).
- [ ] **Substrate stability:** per-round participant set + `norm_log.json` byte-reproducible across two
      same-seed runs at fraction<1 (commit both).
- [ ] **IID-mini ≈ central (falsifiable):** both sides clear `R_floor` (>0) + anti-collapse + recall
      agreement within `δ` (a zero-decode model FAILS — tested); curve reported.
- [ ] **Non-IID gap (completed trainval-scale artifact):** a frozen-Swin-T trainval(-subset) FedAvg run
      ran to completion on the A40 (derived N, fraction<1, ≤20 rounds); commit the **measured** per-round
      + total wall-clock from that run, the IID-vs-log-group `recall@2m` gap (reported, NOT required
      small, `scale`-stamped), and the final checksum. The mini log-group run is methodology smoke only.
- [ ] **Login-node↔Ray cross-check:** byte-identical **aggregated-weight** checksum on a shared ≥2-round
      sampled config (same A40), not a scalar.
- [ ] **Task-agnostic intact:** `dummy_regression` CPU FL smoke runs + deterministic after DT3-A/DT3-B.
- [ ] **Tests green** — `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` (T0+T1+T2's 148 +
      new T3 tests); record the count; note which require the A40 SLURM job.
- [ ] **`collab/T3/SPEC.md` filled** (the 62-tensor update-vector layout contract + the A40 FL checksum +
      measured wall-clock + the IID/non-IID numbers + R_floor/δ) + `findings_log.md`; 2–3 least-certain
      items flagged for Codex.

## 7. Self-review — to be filled by the build session
(Predicted hardest review targets: (a) the **DT3-B deterministic sampler** truly replacing Flower's
random selection — participant set byte-identical across drivers at fraction<1; (b) **full-loop
bit-identity on the A40 at ≥3 rounds + fraction<1** on the real model (not a trivial-config or CPU pass);
(c) the **trainable-only transmit + pretrained frozen-backbone cross-node identity** + the strict=False/
merge plumbing; (d) IID≈central being falsifiable (R_floor + anti-collapse) and the trainval gap being a
completed run, not a projection. Point Codex at the committed A40 FL checksum, the sampling-determinism
test, the 62-tensor layout, and the trainval run artifacts.)
