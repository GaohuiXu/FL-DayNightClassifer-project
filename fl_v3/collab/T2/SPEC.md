# T2 — SPEC: deterministic BEVFusion-class model + detection loss/decode + V2/V3 (clean)

> Build-session copy, filled from `fl_v3/collab/SPEC_TEMPLATE.md`.
> Contract: `fl_v3/docs/cycle_04/tasks/T2_SPEC.md` (read its **§0** first). Plan: task **T2** in
> `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md`. Decisions: D1 (frozen ImageNet camera
> backbone), D3 (BEV-concat `ConvFuser`), D6 (frozen-backbone BN eval; new modules GroupNorm/LayerNorm).

## 1. Scientific intent

Build the **bit-deterministic BEVFusion-class multimodal detector** that is the platform's model — the
instrument every later attack/defense result is measured on. It consumes the **frozen T1 canonical
schema** (doing the resize/normalization T1 left out), registers as a `NuScenesDetectionTask`, and is
proven **deterministic on the A40**, **learns** (a falsifiable overfit), and is **calibrated**
(ground-truth-anchored BEV, not eyeball). T2 has **no bit-parity oracle** (the BEVFusion/LSS/CenterPoint
references use the very ops we reject) — correctness is earned by (a) bit-identical weights on the A40,
(b) overfit-learns + anti-collapse, (c) V2/V3 + ground-truth-anchored calibration, (d) per-module sanity
+ a static banned-op gate. No FL run, no attack yet (T3/T5); T2 trains **centrally** on mini.

## 2. Scope

**Delivered — `models/fusion/` package of cleanly-named sub-networks** (per-module param counts + gradient
slicing for T5/T6 without surgery), all obeying the single shared BEV convention:

- `bev_grid.py` — the **single** metric↔grid convention (the T2↔T4↔T5 contract; §convention below).
- `preprocess.py` — deterministic image resize + ImageNet norm; **half-pixel-correct** rescale of
  `cam_intrinsics` and **recompute `lidar2img`** (exact consistency, ≤1px gate). RNG-free; 256×704.
- `camera_backbone.py` — **Swin-T** (torchvision `IMAGENET1K_V1`, frozen, manual fp32 attention) +
  **ResNet-18** fallback. Freeze **survives `model.train()`** (overridden `train()` keeps the backbone in
  eval → BN running stats never update; D1/D6).
- `camera_neck.py` — GeneralizedLSSFPN analog (GroupNorm; nearest upsample) → one feature map at stride 16.
- `view_transform.py` — **LSS** depth-softmax + outer-product lift + **`cumsum_trick`** splat
  (`QuickCumsum` autograd; **canonical `(rank, geom_id)` sort** → permutation-invariant + architecture-
  portable; `index_copy_` assignment of unique-cell sums). No `scatter_add`, no `grid_sample`.
- `lidar_encoder.py` — **dense PointPillars**, sparse-pillar PFN (Linear+GroupNorm+ReLU+`torch.max`);
  **per-point features only** (no within-pillar cluster-mean → permutation-invariant by construction);
  STABLE-sort grouping via `unique_consecutive`; **`index_copy_`** scatter (unique cells; #76176-safe).
- `fusion.py` — **`ConvFuser`** (D3): concat camera-BEV ⊕ LiDAR-BEV → Conv-GroupNorm-ReLU; the named
  `model.fusion` sub-network.
- `bev_neck.py` — SECOND-FPN (GroupNorm; nearest upsample; no adaptive pool) → head resolution.
- `head.py` — CenterPoint SeparateHead (per-class heatmap + reg: offset/z/log-dim/sin-cos/vel;
  heatmap bias −2.19).
- `losses.py` — Gaussian focal heatmap + L1 reg; the **corrected 3-case `gaussian_radius`** (`/(2·a)`
  denominators, not the CenterNet `/2` bug); target rendering RNG-free + atomic-free (`torch.maximum`).
- `detector.py` — `BEVFusionDetector` wiring + **deterministic `decode()`** (3×3 max-pool local-max mask
  + `torch.sort(stable=True)`; boxes in the **T1 canonical** convention — no `−π/2`, no `(l,w,h)` swap;
  NMS-free). `param_table()` (the Q2 seed).
- `collate.py` — `detection_collate_fn` (RNG-free; LiDAR points as a **batch-index-column** `[TotalP,6]`).
- `training/tasks.py` — `NuScenesDetectionTask` (registered `nuscenes_detection`); `num_clients` returns
  the **derived** partition N; `evaluate` = loss + **center-distance proxy**.
- `training/loop.py` — **additive** device-move + batch-size protocol (tensor OR multimodal dict;
  `criterion(model_output, targets)`); `Criterion` alias widened in `loop.py` **and** `tasks.py`.
  `dummy_regression` stays **byte-identical** (committed golden, below).
- `viz/encoder.py` (V2) + `viz/fusion.py` (V3 clean) — mirror `viz/calibration.py`.
- Tests `tests/test_model_*.py` (+ `tests/_det_fixtures.py`); env: `torchvision==0.22.1 --no-deps` +
  `TORCH_HOME` pin + login-node weight pre-cache in `scripts/{build_venv,run_in_venv,run_alvis}.sh`;
  `scripts/{det_gate_a40.py,run_det_gate_a40.sh,build_nuscenes_cache.py}`; `docs/env.md` +
  `docs/determinism.md` (§T2 enforcement); this SPEC + `findings_log.md`.

**Out of scope / deferred (unchanged from contract):** official `DetectionEval` mAP/NDS + 6-criterion ASR
+ V4 (T4); real Ray FedAvg + IID/non-IID gap (T3); attacks/V5/trigger-diff V3 (T5); defenses/V6 (T6);
full-model FL ablation; LiDAR sweeps, radar, map priors, TTA; within-pillar cluster-mean PFN feature
(dropped for permutation-invariance — a canonical-order variant is a later refinement).

**Consume-only (unmodified):** T0 `strategy/`, `utils/runtime.py`; T1 `data/nuscenes/**` (schema frozen —
not mutated). `fl_v2/` untouched.

## 3. The frozen T2↔T4↔T5 contract (BEV grid + convention)

**Single mapping** (`bev_grid.py`, the only place it is defined; splat/scatter/target/decode all call it):

```
BEV tensor layout [B, C, H, W]:   W (cols) ← x (fwd, +x),   H (rows) ← y (left, +y)
col = floor((x − x_min)/vx)        row = floor((y − y_min)/vy)        flat = row·W + col
decode (CenterPoint, +offset, NO half-cell): cx = (col + offset_x)·head_vx + x_min  (offset = fx − floor(fx))
```

- **Headline grid:** `point_cloud_range = [−51.2,−51.2,−5.0, 51.2,51.2,3.0]`, `bev_voxel = (0.4,0.4)` →
  fine grid **256×256** (camera-BEV + LiDAR-BEV + fused all on it); `out_size_factor = 2` → head grid
  **128×128** (0.8 m/cell). Splat/scatter use the fine grid; target/decode use the head grid; the
  **convention** (`W→x`, `H→y`, floor, `flat=row·W+col`) is identical at both.
- **Box convention == T1 canonical** `(cx,cy,cz,dx=l,dy=w,dz=h,yaw)`, `yaw = atan2(sin,cos)`. **No mmdet3d
  `−π/2` offset, no `(l,w,h)` swap.** Pinned by the **encode→decode round-trip golden** (recovers cx/cy ≤
  half a head cell, z/dims ≤1e-2/1e-3, yaw |Δ|<1e-3 incl. sign over yaw ∈ {±0.7, ±2.5, 0}).
- **Anchored to T1 geometry, NOT self-consistency** (`test_model_bev_convention.py`): a real GT car's
  box center and its physical LiDAR returns fall in the SAME fine-grid cell (median within ≤2 cells); an
  **injected row↔col swap FAILS** the anchor (proving discriminating power). Resize/intrinsic/`lidar2img`
  consistency is ≤1px vs T1's projector (measured **~1e-4 px**; half-pixel-correct affine).

## 4. Invariants (must hold; Codex checks each)

- **Bit-determinism (crown jewel):** same-seed two K-step central trainings → `torch.equal` on every
  parameter **on the A40 (SLURM)** — PASS, checksum below. Backed by: a **static AST ban** over
  `models/fusion/**`; **permutation-invariance** of splat + pillar-scatter (byte-identical under input
  permutation); a **decode tie** test; the **#76176** `index_copy_` GPU guard; the **float-cumsum** GPU
  guard. All RNG via `derive_seed`/`seed_everything`; determinism holds at `num_workers` 0 and >0
  (loader byte-identical); `preprocess` resize is RNG-free.
- **D1 (frozen backbone):** `requires_grad=False` AND `.eval()` **survives `model.train()`** (BN
  running-stat bytes unchanged after a train step — verified on the **ResNet BN path**); no Adam state for
  backbone params (optimizer set is disjoint from backbone params).
- **D6:** new modules GroupNorm/LayerNorm; the only BatchNorm is inside the frozen backbone (eval mode);
  no `Adaptive*Pool2d` anywhere.
- **Schema/convention contracts:** consume the T1 schema unchanged; the §3 BEV grid + mapping + decode
  convention are frozen here; the encode→decode golden + ground-truth anchor + corruption negative pin
  them numerically.
- **The model learns (falsifiable):** on a fixed single real scene of K=180 steps, `final_loss ≤
  0.2·initial` **AND** recall@2m (center-distance proxy, car) **0 → ≥0.5** **AND** anti-collapse
  (decoded-above-threshold > 0 AND ≥0.5·#GT matched to DISTINCT GT).
- **Task-agnostic skeleton preserved:** `dummy_regression` determinism/regression pass byte-for-byte
  (committed-golden checksum assert); no detection assumption leaks into the generic loop.
- **No false oracle:** determinism + overfit + V2/V3 + anchored calibration + per-module sanity — NOT
  bit-parity with any external model.

## 5. Scientific failure modes checked (point Codex here)

- A banned op slips in & strict mode misses it (§0) — caught by the **static AST ban** + permutation-
  invariance + #76176 + cumsum guards, on the **A40**.
- BEV index-convention self-consistent-but-wrong — caught by the **ground-truth anchor + injected
  corruption** (NOT self-consistency).
- Resize/intrinsic/`lidar2img` inconsistency — numeric ≤1px test anchored to T1.
- D1/D6 via `requires_grad`-only freeze (BN drift), BN in new modules, adaptive pooling — all asserted.
- Decode convention drift (`−π/2`/swap/yaw sign) — caught by the encode→decode golden.
- Degenerate "learning" (collapse-to-background) — caught by the falsifiable overfit + anti-collapse.
- `gaussian_radius /2` bug — corrected 3-case used; the overfit converging (loss ratio 0.029) is the
  positive evidence it is not stalled.
- Wall-clock blow-up — the **headline** config is measured on the A40 (below), not only the bring-up.

## 6. GATE — status (all met)

- [x] **Bit-identical weights on the A40 (SLURM):** `scripts/run_det_gate_a40.sh` → device **NVIDIA
      A40**, `CUBLAS_WORKSPACE_CONFIG=:4096:8` pre-CUDA, two same-seed **12-step** trainings → `torch.equal`
      on every param. **Errors LOUD (exit 2) on no-CUDA / non-A40** (verified: the login Tesla T4 run
      exits 2). Also bit-identical on CPU (`test_full_forward_same_seed_bit_identical_cpu`).
      **`A40_WEIGHT_CHECKSUM = 31f23465bef5b46c5aa241b23d7b0726eb7a22502f3fdeb3a0191353a75afcd5`**
      (per-parameter SHA-256, ResNet bring-up config, 12 steps, seed 7; reproducible on any A40;
      job 6763718, node alvis9-07).
- [x] **Static AST ban test** over `models/fusion/**` (no `scatter_add`/`index_add`/
      `index_put(accumulate=True)`/`grid_sample`/`topk`/non-stable sort).
- [x] **Permutation-invariance:** splat + pillar-scatter byte-identical under input permutation (CPU
      `max|Δ|=0`; A40 bit-identity covered by the gate). **Decode tie:** canonical, reproducible.
- [x] **#76176 + cumsum guards (GPU):** `index_copy_` scatter correct under strict mode; float-CUDA
      `cumsum` does not raise & is `torch.equal` fwd+bwd (run on T4 in the suite; **re-run on ARM/H200**).
- [x] **Trains centrally on mini end-to-end:** full multimodal forward + loss + backward + decode on real
      mini batches; non-degenerate detections; center-distance proxy > 0. *(Official mAP/NDS = T4.)*
- [x] **Overfit (falsifiable):** single real mini scene, K=180, lr 3e-3, frozen ImageNet ResNet-18:
      `init=53.7 → final=1.58` (**ratio 0.029** ≤ 0.2); recall@2m **0 → 0.615** (≥0.5); 70 decoded boxes,
      8/13 distinct GT cars matched (anti-collapse holds).
- [x] **BEV ground-truth-anchored convention** + injected-corruption negative; **resize ≤1px** (~1e-4 px);
      **encode→decode golden** (incl. yaw sign; no `−π/2`/swap).
- [x] **D1/D6:** backbone frozen through `train()` (ResNet BN bytes unchanged after a step); no Adam state
      for backbone; Swin-T `IMAGENET1K_V1` loads + forwards deterministically; new modules GN/LN; no
      adaptive pooling.
- [x] **Per-module param counts** (headline frozen Swin-T; see table) — param-count asserts (Swin full
      **28,288,354** / ResNet-18 full **11,689,512**) double as the weights-loaded check.
- [x] **V2 + V3 (clean) renders** (≥3 mini samples each; manifest written; the V3 BEV-alignment overlay).
- [x] **Task-agnostic intact:** `dummy_regression` byte-for-byte (golden
      `d2d819fee9a54fc302a9d6c9d0ac4e4d875629a0a16e75f2328f28b7f63cd7cc`); detection determinism identical
      at `num_workers` 0 and >0; a 1-step central smoke runs through the generalized loop.
- [x] **Wall-clock for BOTH configs on the A40** (below) — recorded for T3 planning.
- [x] **Env:** `torchvision==0.22.1 --no-deps` pinned; login-node weights pre-cached; `TORCH_HOME` pinned
      in build + run scripts; post-install asserts (`numpy==1.26.4`, both `.pth` exist, `torch.cuda`);
      `docs/env.md` + `docs/determinism.md` §T2 enforcement landed.
- [x] **Tests green:** `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` → **147 passed**
      (T0+T1's 120 + **27** T2). GPU-requiring set (overfit, central smoke, #76176, cumsum) runs on any
      CUDA in the suite; the **A40 SLURM job is the authoritative bit-identity gate** (NOT in pytest).
- [x] **`collab/T2/SPEC.md`** filled (this file) + `findings_log.md`; least-certain items flagged (§7).

### Per-module parameter table — headline config (frozen Swin-T, 256×704, fine BEV 256², head 128²)

| module | total | trainable | % of trainable |
|---|---:|---:|---:|
| preprocess | 0 | 0 | 0.00% |
| camera_backbone (Swin-T, **frozen**) | 27,517,818 | 0 | 0.00% |
| camera_neck (LSS-FPN) | 333,056 | 333,056 | 18.02% |
| view_transform (LSS depthnet) | 165,643 | 165,643 | 8.96% |
| lidar_encoder (PointPillars PFN) | 576 | 576 | 0.03% |
| **fusion (`ConvFuser`, D3)** | 313,856 | 313,856 | **16.98%** |
| bev_neck (SECOND-FPN) | 886,272 | 886,272 | 47.95% |
| head (CenterPoint) | 148,884 | 148,884 | 8.06% |
| **TOTAL** | **29,366,105** | **1,848,287** | |

Frozen camera backbone = 27,517,818 params (feature-extractor of the 28,288,354-param Swin-T; the dropped
770,536 = the discarded ImageNet classifier head/norm). **Fusion = 16.98% of trainable** — the Q2-dilution
seed (will be re-measured under each trained-component config; do NOT assume a fixed %). *Note:* the
PointPillars PFN is a single Linear (576 params) — minimal by design (the LiDAR-BEV is mostly geometric:
pillar scatter + `torch.max`); deepening it is a later refinement.

### Wall-clock (A40, job 6763718, batch=1; warmup'd; per-call mean over 10 iters)

| config | forward | full train step | projected /round (25 clients × 100 steps) |
|---|---:|---:|---:|
| bring-up ResNet-18 | 16.0 ms | 29.2 ms | **1.2 min** |
| **headline frozen Swin-T (6 cam, 256×704, real grid)** | 54.4 ms | 76.0 ms | **3.2 min** |

→ T3 headroom: a ≤20-round FedAvg at the headline config ≈ **~64 min** of compute (25 clients × 100
local steps/round). The bring-up backbone is the recommended T3 first-light config (~24 min for 20
rounds). These are A40 single-actor numbers; T3 owns client-sampling mitigations.

## 7. Self-review — what I'm least sure about (attack these hardest)

1. **Splat/scatter/topk summation-order independence on the A40 (§0 spine).** The crown-jewel claim is
   that the LSS `cumsum_trick` and the PointPillars pool have **zero summation-order/atomic dependence by
   construction** — proven by the static AST ban + permutation-invariance (CPU `max|Δ|=0`) + the A40
   bit-identity gate (checksum above), NOT by the now-hollow `strict`-mode raise. **Scrutinize:** (a) the
   splat's canonical `(rank, geom_id)` lexsort really makes the per-cell `cumsum` order input-permutation-
   independent (vs the LSS reference's non-stable `argsort`); (b) dropping the PointPillars within-pillar
   **cluster-mean** to keep the PFN per-point (a float-mean over points would be order-dependent) — is
   that the right determinism/accuracy trade, or should it be a canonical-order mean? Also: the over-cap
   truncation keeps the first `max_points` in **file order** — confirm that is acceptable (a permutation
   changes the kept subset only when a pillar exceeds the cap; the permutation test runs without
   truncation). (c) the float-CUDA `cumsum` guard passed on the A40 here but the **ARM/H200 rebuild must
   re-run it** (torch docstring still lists it as potentially raising).
2. **The single shared BEV `(x,y)→(row,col)` binding, anchored to T1 (not self-consistency).** The
   no-oracle trap. **Scrutinize:** the ground-truth anchor (real GT car center vs its physical LiDAR
   returns in the same fine cell, ≤2-cell median) + the injected row↔col-swap negative are genuinely
   independent of the mapping under test (a self-consistent-but-wrong mapping should fail the anchor); and
   the half-pixel-correct resize affine is the right intrinsic/`lidar2img` rescale (≤1px, measured ~1e-4).
3. **Frozen-backbone BN freeze on the ResNet path (D1/D6).** The `train()`-override keeps the backbone in
   eval so BN running stats never update mid-`model.train()`. **Scrutinize:** `test_resnet_bn_frozen_
   through_train_step` asserts the running_mean/var/num_batches_tracked **bytes** unchanged after a real
   train step — is this the complete freeze (the failure mode a `requires_grad`-only freeze hides)?
4. **`dummy_regression` byte-identity after the loop generalization.** The tensor path is *provably* the
   same op sequence (`X.to/y.to → model(X) → criterion(out,y) → y.size(0)`); the committed golden
   `d2d819…cd7cc` pins it and 21 T0 determinism/task-agnostic tests pass. **Scrutinize:** the additive
   `_unpack_batch`/`_move_to_device`/`_batch_size` helpers introduce nothing into the tensor path.
5. *(Lower)* The **center-distance proxy** is a deliberate strict subset of T4's matching (car-class, BEV
   L2 ≤2m, greedy distinct) — it is the overfit/learning signal, NOT mAP/NDS (T4). The lidar_encoder PFN
   is minimal (1 layer) — adequate for the determinism+overfit instrument, flagged for later deepening.
