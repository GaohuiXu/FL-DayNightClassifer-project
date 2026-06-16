# T2 — Build-session kickoff prompt (paste into a fresh Claude Code session)

You are the **build session for Cycle-04 task T2** of a thesis project on securing federated learning for
autonomous-driving perception. **T0 and T1 are complete and Codex-PASSed** (the fl_v3 skeleton +
determinism harness + defense family + viz scaffold, and the bit-deterministic nuScenes multimodal data
module + log-group partitioner + V1; 120 tests green). You build the **deterministic BEVFusion-class model
+ the detection loss/decode + V2/V3 (clean)** on top of them. **This is the most technically complex and
most determinism-critical task in the cycle** — the model is the instrument every later attack/defense
result is measured on.

**Read first, in order:**
1. `fl_v3/docs/cycle_04/tasks/T2_SPEC.md` — **your contract. Read §0 FIRST** — it overturns a stale
   project assumption (strict mode does NOT catch most banned ops on torch 2.7) and reshapes how you prove
   determinism. The conventions, the frozen schema, and the GATE are load-bearing.
2. `fl_v3/docs/determinism.md` — the corrected banned-ops contract (note the §0-aligned correction).
3. `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md` — the plan (Architecture table, §FL setup, the
   T2 task entry + GATE, Viz V2/V3).
4. `fl_v3/docs/cycle_04/decisions.md` — D1 (frozen ImageNet backbone), D3 (BEV-concat `ConvFuser`), D6
   (frozen-backbone BN eval; new modules GroupNorm/LayerNorm).
5. `fl_v3/src/fl_v3/data/nuscenes/conventions.md` + `collab/T1/SPEC.md` §schema — the **frozen T1↔T2
   sample schema** you consume (you do the resize/normalization T1 left out; you must NOT mutate T1).
6. Skim `fl_v3/collab/T1/SPEC.md` §7 + `findings_log.md` for the bar and the carried-over conventions.

**Your job:** execute T2 to its GATE — a **bit-deterministic** BEVFusion-class detector (frozen Swin-T
camera backbone + LSS `cumsum_trick` camera→BEV + dense PointPillars LiDAR + BEV-concat `ConvFuser` +
SECOND-FPN + CenterPoint head), a deterministic detection loss + decode (boxes in the T1 canonical
convention), the `NuScenesDetectionTask` wired into the FL skeleton (generalizing `loop.py` additively
without breaking `dummy_regression` byte-identity), and the V2/V3 clean renders — proving the model is
**deterministic on the A40**, **learns** (falsifiable overfit), and is **calibrated** (ground-truth-anchored
BEV, not eyeball). T2 has **no bit-parity oracle** — correctness is earned by determinism + overfit + V2/V3
+ per-module sanity, not by matching an external model.

**Hard rules:**
- **§0 is the spine.** `enforce_determinism(strict=True)` only RAISES on `grid_sample`/adaptive-pool
  backward on this torch — `scatter_add`/`index_add`/non-stable `topk`/`sort` pass silently. Prove
  determinism with **(1) a static AST ban test** over `models/fusion/**`, **(2) a permutation-invariance
  test**, **(3) the #76176 `index_copy_` GPU test + the float-cumsum GPU guard**, and **(4) a dedicated
  A40-pinned SLURM job** that asserts the device, runs two same-seed K-step trainings to `torch.equal`
  weights, and commits the per-param SHA-256 checksum into `collab/T2/SPEC.md`. **CPU-only / login-node
  (Tesla T4) determinism is a FALSE PASS** — the gate must run on the A40 via SLURM and error loud (never
  green) if there's no CUDA.
- **The exact deterministic recipes are in the SPEC — follow them:** LSS splat with `argsort(stable=True)`
  (the reference is non-stable) + the `QuickCumsum` custom autograd; pillar scatter via
  `canvas.index_copy_(1, idx, voxels)` (**never** `canvas[:, idx] = voxels` — #76176 silent no-op);
  sparse-pillar PFN (not a dense `[H*W,…]` 2 GiB tensor) with `torch.max(...).values`; deterministic decode
  topk (no `stable` kwarg exists — use the max-pool-mask + monotone-tiebreak composite); the **corrected**
  3-case `gaussian_radius` (not the CenterNet `/2` bug); no `Adaptive*Pool2d` in trainable modules.
- **D1/D6:** backbone `requires_grad=False` AND `.eval()` that **survives `model.train()`** (assert BN
  running-stat bytes unchanged after a step — test this on the **ResNet-18 fallback**, the BN path, not
  only Swin); GroupNorm/LayerNorm in new modules. **DT2-A:** add `torchvision==0.22.1 --no-deps`,
  **pre-cache the Swin/ResNet ImageNet weights on the LOGIN node** at build time (compute nodes are
  offline) + pin `TORCH_HOME`; no SDPA pin is needed (torchvision Swin uses manual math attention).
- **Freeze the T2↔T4↔T5 contract:** the BEV grid + the single `(x,y)→(row,col)` mapping shared by
  splat/scatter/target/decode (anchor it to T1's geometry with a **ground-truth** test + an
  injected-corruption negative — NOT a self-consistency check, the trap T1 defeated), and the decode box
  convention (== T1 canonical; **no** mmdet3d `−π/2`/`(l,w,h)` swap; pin it with an encode→decode round-trip
  golden). Write these into `collab/T2/SPEC.md`.
- **Scope:** the official `DetectionEval` mAP/NDS + ASR is **T4** — T2 uses an overfit + a provisional
  center-distance proxy; do NOT reimplement T4. The real Ray FedAvg run is **T3** — T2 trains *centrally*,
  but **report wall-clock for BOTH the bring-up AND the headline Swin-T config** so T3 can plan. Heavy runs
  go through SLURM, never the login node; run code via `fl_v3/scripts/run_in_venv.sh`.
- Write your SPEC to `fl_v3/collab/T2/SPEC.md` (per-module param table + the BEV/decode convention + the
  A40 weight checksum + wall-clock numbers), add the tests, drive the GATE to green, append to
  `findings_log.md`.

**When the GATE is green:** summarize what landed, paste the test count + the A40 determinism checksum +
the wall-clock numbers, list the 2–3 things the Codex reviewer should scrutinize hardest (the SPEC
predicts: the splat/scatter/topk being summation-order-independent on the A40, the single shared BEV
convention anchored to T1, the frozen-backbone BN freeze on the ResNet path, and `dummy_regression`
byte-identity after the loop generalization), and stop — the Codex review session reviews before T3 starts.
Do not commit/push unless the user asks.
