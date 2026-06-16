# T3 — Build-session kickoff prompt (paste into a fresh Claude Code session)

You are the **build session for Cycle-04 task T3** of a thesis project on securing federated learning for
autonomous-driving perception. **T0, T1, and T2 are complete and Codex-PASSed** (the fl_v3 skeleton +
determinism harness + defense family + FedAvg machinery; the bit-deterministic nuScenes data module +
log-group partitioner + V1; the deterministic BEVFusion-class model + detection loss/decode + V2/V3; 148
tests green). **T3 is the PLATFORM MILESTONE** — wire it all into a real Flower/Ray sequential FedAvg run
on the A40 via SLURM and prove the platform is a sound, reproducible instrument. This gate = "the platform
works."

**Read first, in order:**
1. `fl_v3/docs/cycle_04/tasks/T3_SPEC.md` — **your contract. Read §0 FIRST** — it overturns three
   first-draft assumptions (flwr 1.27 auto-manages the SuperLink; client sampling is non-deterministic by
   construction and must be hard-overridden; the determinism gate must be A40 + ≥3 rounds + fraction<1 on
   the real model). The DT3-A/DT3-B decisions, the invariants, and the GATE are load-bearing.
2. `fl_v3/docs/determinism.md` — "How determinism is enforced per scope" (the FL determinism machinery)
   + the §T2 banned-op reality (it holds inside FL local training too).
3. `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md` — the plan (T3 task entry + GATE, §FL setup D1,
   §determinism).
4. `fl_v3/docs/cycle_04/decisions.md` — D1 (frozen ImageNet backbone, FL-train the rest; binds T3).
5. Skim `fl_v3/collab/T2/SPEC.md` (the model, the per-module param table, the wall-clock numbers, the
   `det_gate_a40.assert_a40` A40-gate pattern you reuse) and the actual FL seams: `client_app.py`,
   `server_app.py`, `strategy/flower_strategies.py`, `engine/local_runner.py`, `configs/flwr_config.toml`,
   `scripts/{run_alvis.sh, det_gate_a40.py, run_det_gate_a40.sh}`.

**Your job:** execute T3 to its GATE — harden `run_alvis.sh` into a working flwr-1.27 launcher, run a real
clean `nuscenes_detection` FedAvg on the A40, and prove **two same-seed runs (≥3 rounds, sampled) produce
a byte-identical final global model on the A40**; measure **IID-mini ≈ central** (falsifiably) and the
**non-IID gap** (a completed trainval-scale run); confirm wall-clock + ≤20 rounds. Clean FedAvg only — no
attack, no defense-behavior study (that's T5/T6).

**Hard rules:**
- **§0 is the spine.** (1) flwr 1.27 `flwr run` auto-starts + manages the local SuperLink — do **NOT**
  manually launch `flower-superlink` or carry fl_v2's SuperExec grep/sleep waits; the only fl_v2 SuperLink
  hardening that still applies is per-job `FLWR_LOCAL_CONTROL_API_PORT`/`…SIMULATIONIO…` ports. (2) Client
  sampling is non-deterministic by construction (Flower `random.sample(grid.get_node_ids())` over
  random-pubkey node_ids) — you **must** add a deterministic sampler over the fixed `0..N-1` partition-id
  space (`derive_seed(seed, "sample", server_round)`); the fraction=1.0 escape is a FALSE PASS. (3) The FL
  determinism gate must **reuse `det_gate_a40.assert_a40`** (loud exit-2 on non-A40/CPU) and run **≥3
  rounds at fraction<1 on the real model** — a 1-round / CPU / Tesla-T4 pass is a FALSE PASS.
- **DT3-A (trainable-only update vector):** transmit/aggregate only the **62 trainable tensors** (the
  frozen 27.5M-param backbone is reconstructed identically per node from the pinned ImageNet cache and
  excluded). Use a `requires_grad`-based filter at 4 seams (initial_arrays, client reply, eval load, final
  checkpoint), `load_state_dict(strict=False)`, **require `det-pretrained-backbone=True`** (a random-init
  "frozen" backbone differs per node), and make the final checkpoint a self-contained FULL model. Freeze
  the per-module slice map `{camera_neck:15, view_transform:5, lidar_encoder:3, fusion:6, bev_neck:18,
  head:15}` as the T3→T6 contract. Gradient metrics inherit trainable-only-ness — do NOT add a mask.
- **SLURM launcher must actually submit + not silently die:** include `#SBATCH -A NAISS2025-22-1113` +
  `-p alvis`; per-job `RAY_*` ports + `RAY_TMPDIR`; `WANDB_MODE=offline` (offline node); the **silent-exit
  guard** (assert a completion marker or force non-zero exit — `flwr run` returns 0 even on a zero-round
  run). The run must make zero network calls (TORCH_HOME weights + nuScenes cache pre-built).
- **Falsifiable milestone, real artifacts:** IID-mini≈central needs an absolute recall floor `R_floor`>0
  both sides + anti-collapse (a zero-decode loop must FAIL); the non-IID gap is a **completed
  trainval-scale Swin-T run** (measured wall-clock from that run, not a ResNet projection; the degenerate
  mini log-group is methodology smoke only). The metric is the T2 center-distance proxy (recall@2m) +
  val-loss — official mAP/NDS is T4.
- Heavy runs go through SLURM, never the login node; run code via `fl_v3/scripts/run_in_venv.sh`. Consume
  T2/T1/T0 unchanged (don't mutate the model or the schema; if something's missing, raise a finding).
- Write your SPEC to `fl_v3/collab/T3/SPEC.md` (the 62-tensor update-vector layout contract + the A40 FL
  checksum + measured wall-clock + the IID/non-IID numbers + R_floor/δ), add the tests, drive the GATE to
  green, append to `findings_log.md`.

**When the GATE is green:** summarize what landed, paste the test count + the A40 FL determinism checksum +
the measured wall-clock + the IID/non-IID numbers, list the 2–3 things the Codex reviewer should scrutinize
hardest (the SPEC predicts: the deterministic sampler replacing Flower's random selection at fraction<1;
full-loop A40 bit-identity at ≥3 rounds; the trainable-only transmit + pretrained frozen-backbone
cross-node identity; IID≈central being falsifiable + the trainval gap being a completed run), and stop —
the Codex review session reviews before T4 starts. Do not commit/push unless the user asks.
