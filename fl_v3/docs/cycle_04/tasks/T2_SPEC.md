# T2 — SPEC: deterministic BEVFusion-class model + detection loss/decode + V2/V3 (clean)

Plan: `../../roadmap/cycle_04_fusion_layer_backdoors.md` (task **T2**; Architecture table, §FL setup,
Viz **V2/V3**, §determinism). Decisions: `../decisions.md` — **D1** (frozen ImageNet camera backbone;
FL-train LSS-depth + LiDAR-enc + fusion + neck + head), **D3** (BEV-concat `ConvFuser`), **D6**
(frozen-backbone BN eval-mode; new modules GroupNorm/LayerNorm). Contract for the **T2 build session**.
Fill `fl_v3/collab/T2/SPEC.md` from the template.

> **The determinism design below was empirically verified on the Alvis GPU node** (a 6-agent pass,
> workflow `wf_f35d6cff-9be`, which *ran* the ops under `enforce_determinism(strict=True)` on CUDA).
> That pass overturned a stale project assumption — **read §0 first.**

---

## 0. CRITICAL determinism reality (read before anything else) — strict mode is NOT your banned-op detector

The platform's determinism contract assumed `use_deterministic_algorithms(True)` makes the banned ops
**raise**. **On the pinned torch 2.7.1 / CUDA 12.6 build this is FALSE** (verified empirically, CPU and
CUDA): `scatter_add` / `index_add` / `index_put(accumulate=True)` / **non-stable `topk`** / non-stable
`sort` **do NOT raise** — recent PyTorch registered deterministic CUDA kernels for them. **Only
`grid_sample` *backward* still raises.** Consequences that reshape T2:

1. **`enforce_determinism(strict=True)` cannot be your proof that no banned op is present.** A stray
   `scatter_add` in the splat/pillar-scatter passes silently. → **You MUST add a static AST/grep ban
   test** over `models/fusion/**` (fail on any `scatter_add` / `index_add` / `index_put(...,
   accumulate=True)` / `F.grid_sample` / bare `argsort`/`sort`/`topk` without a stable/deterministic
   recipe). This static test is the real banned-op gate; the runtime `strict` pass is necessary but not
   sufficient.
2. **Same-seed-twice-on-one-GPU is NOT a sufficient determinism test.** It is bit-identical *even with a
   `scatter_add` present* (verified). The float-summation-order / topk-tie nondeterminism only surfaces
   **cross-architecture** (the login Tesla T4 ≠ the production A40 ≠ the future ARM H200) — which is the
   whole Arrhenius-portability bet. → add a **permutation-invariance** test (permute the input point
   order → byte-identical splat/scatter output; a `scatter_add` fails this, `cumsum_trick` +
   unique-index assignment passes) and a **tie-permutation** test for the decode.
3. **The standard `pytest fl_v3/tests` runs on the LOGIN node (Tesla T4), not the A40.** Zero current
   tests touch CUDA. → the bit-identical-weights gate must be a **dedicated A40-pinned SLURM job** that
   asserts `torch.cuda.get_device_name()` contains `A40`, **skips LOUD (xfail/error, never green) if no
   CUDA**, exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` *before* first CUDA use, and **commits the
   resulting per-parameter SHA-256 weight checksum** into `collab/T2/SPEC.md` as the evidence artifact.

The **reason we still ban** `scatter_add`/`index_add` is therefore NOT "it raises" but: (a) bit-identity
of the deterministic-scatter path is **not guaranteed across the ARM rebuild**, and (b) defense-in-depth
— the model should have **zero atomic-ordering / summation-order dependence by construction**. *(This SPEC
also corrects `fl_v3/docs/determinism.md`, whose "strict mode makes them RAISE" wording is now stale.)*

> **Orchestrator setup notes:**
> - **DT2-A (camera-backbone weights, CONFIRMED 2026-06-16):** `torchvision` is NOT installed (verified)
>   and not in the manifest, yet D1's backbone is an ImageNet Swin-T. **Decision (locked):** add
>   **`torchvision==0.22.1`** (the exact pairing for torch 2.7.1) installed **`--no-deps`** against the
>   module torch — the same footgun-avoiding pattern as `nuscenes-devkit` (its PyPI metadata pins
>   `torch==2.7.1`+`numpy`, which would shadow the CUDA/numpy-matched module build). cp312 manylinux wheels
>   exist for **x86_64 AND aarch64** → stays Arrhenius-portable. **No SDPA/flash-attn pin needed:**
>   torchvision `swin_t` uses **manual math attention** (`matmul → +bias → F.softmax → matmul`), it does
>   **not** call `scaled_dot_product_attention`, so no flash/mem-efficient kernel can be reached (verified
>   vs v0.22.1 source). **Offline-weights (blocker):** `torch.hub` downloads weights on first instantiation
>   and Alvis *compute* nodes are offline — pre-cache on the *login* node at build time (see §2). *(The
>   vendor-a-deterministic-Swin/ResNet alternative was considered and rejected as heavier with no upside —
>   do NOT pursue it.)*
> - **"mAP/NDS>0" vs "DetectionEval is T4":** the official nuScenes `DetectionEval` + the 6-criterion ASR
>   eligibility is **T4**. T2 proves *learning* via an **overfit** + a **provisional center-distance
>   proxy** (a deliberate strict subset of T4's center-distance matching) and does **NOT** reimplement
>   `DetectionEval`.
> - **T2 has NO bit-parity oracle** (the BEVFusion/LSS/CenterPoint references use the very ops we reject).
>   Correctness = (a) bit-identical weights, (b) overfit-learns, (c) V2/V3 + ground-truth-anchored
>   calibration, (d) per-module sanity. Do **not** invent a false oracle.

---

## 1. Scientific intent

Build the **bit-deterministic BEVFusion-class multimodal detector** that is the platform's model — the
instrument every later attack/defense result is measured on. Per the Architecture table: **Swin-T**
camera backbone (ImageNet-pretrained, **frozen** — D1; manual fp32 attention) + **LSS-FPN** neck, **LSS
depth-softmax + `cumsum_trick` splat** for camera→BEV (NOT `scatter_add`, NOT `grid_sample`), a **dense
PointPillars** LiDAR encoder (sparse-pillar PFN + `torch.max` + **collision-free index-copy** scatter,
NOT spconv/atomicAdd), a **BEV-concat `ConvFuser`** (D3; GroupNorm/LayerNorm — D6; a cleanly named
sub-network), a SECOND-FPN BEV neck, and a **CenterPoint dense head** with deterministic Gaussian-target
loss and a deterministic decode. It consumes the **frozen T1 canonical sample schema** (doing the
resize/normalization T1 left out) and registers as a `NuScenesDetectionTask`. **Bit-determinism is the
load-bearing property** (T3's FL determinism gate depends on it): same-seed two runs → byte-identical
weights, achievable only if **every op is deterministic AND summation-order-independent** (§0). No FL run
and no attack yet — T2 trains *centrally* on mini and proves the model is deterministic, learns, and
renders inspectably (V2/V3).

## 2. Scope

**In scope (deliver):** a `models/fusion/` package of cleanly-named sub-networks (so per-module param
counts + gradient slicing for T6 Q2 / the T5 fusion-aware attack work without surgery):

- **`preprocess.py`** — deterministic image **resize + ImageNet normalization** (T2 owns this; T1 stores
  native uint8 1600×900). Rescale `cam_intrinsics` for resize/crop and **recompute `lidar2img`
  post-resize** (load-bearing — a mismatch silently mis-aligns camera-BEV vs LiDAR-BEV). Clean baseline =
  **deterministic resize, RNG-free by construction, no random aug** (aug optional + seeded via
  `derive_seed`, off by default). Recommended input **256×704** (BEVFusion-MIT); configurable.
- **`camera_backbone.py`** — **Swin-T** (torchvision `Swin_T_Weights.IMAGENET1K_V1`, **frozen**:
  `requires_grad=False`, `.eval()`, **manual fp32 attention** — no SDPA call). **ResNet-18** fallback
  (`ResNet18_Weights.IMAGENET1K_V1`; the recommended T3 wall-clock bring-up backbone, 11.7M vs 28.3M
  params). **Frozen means frozen through mode switches:** override the detector's `train()` (or re-assert
  `backbone.eval()` each forward) so `model.train()` does **not** flip the backbone's BN to update running
  stats (Swin uses LayerNorm so this is moot for Swin, but the **ResNet-18 fallback is all BatchNorm** —
  this is where a `requires_grad`-only freeze silently breaks D1/D6). Config `camera-backbone ∈
  {swin_t, resnet18}`, `freeze-camera-backbone` (default true; flip = the full-model generality ablation).
- **`camera_neck.py`** — LSS-FPN (GeneralizedLSSFPN analog).
- **`view_transform.py`** — **LSS**: per-pixel **depth softmax** over bins, lift features into the frustum
  (outer product — determinism-clean), **splat into BEV via the `cumsum_trick`** (the LSS `QuickCumsum`
  custom `autograd.Function`: sort by BEV rank → float cumsum → segment-boundary diff). **Determinism
  musts:** the LSS reference uses a **non-stable `ranks.argsort()` — you MUST use `argsort(stable=True)`**
  (float-add is non-associative, so a tie permutation drifts the pooled value at ULP level; `strict` mode
  does **not** catch this). The `cumsum` is **float-on-CUDA** which the torch-2.7 docstring still lists as
  raising but **empirically does not** on this build — pin it with a guard test (below) and a note that
  the **ARM/H200 rebuild must re-run it**; if a future torch raises, the fallback is an int/index
  reformulation or segment-reduce, **NOT `scatter_add`**. **BANNED here: `scatter_add`/`index_add`/atomic
  scatter, `grid_sample` (the lift is grid_sample-free by design — the ban is belt-and-suspenders).**
- **`lidar_encoder.py`** — **dense PointPillars, sparse-pillar PFN**: pillarize points
  (`pillar_id = floor((xy − min)/voxel)`, **int64**); materialize **only the `P_nonempty` occupied
  pillars** as `[P, max_points, C]` (NOT a dense `[H*W, max_points, C]` — that's ~2 GiB at 512², a
  blow-up); config `max-pillars`, `max-points-per-pillar`; over-cap truncation takes the **first
  max_points in file order via a STABLE int64 sort** (`unique_consecutive(return_counts=True)` for segment
  boundaries — **forbid float `cumsum`, weighted `bincount`, and `unique(return_inverse)` for slot
  assignment**; those raise or are tie-unstable on CUDA). PFN = Linear + GroupNorm + ReLU +
  **`torch.max(x, dim=points).values`** (value path — argmax-index nondeterminism is irrelevant; **no
  `adaptive_*pool`** — its CUDA backward raises). **Scatter to the dense `[C,H,W]` canvas via
  `canvas.index_copy_(1, idx, voxels)` (or explicit `index_put_(..., accumulate=False)`) — NOT
  `canvas[:, idx] = voxels`** (that `__setitem__` form **silently no-ops on CUDA under deterministic
  mode** — PyTorch #76176; a determinism-crown-jewel disaster that even passes same-seed equality because
  both runs are zero). Each occupied pillar maps to a **unique** cell (pillar identity == cell identity ⇒
  flat index injective) → assignment, not accumulation; **assert index uniqueness** at runtime.
- **`fusion.py`** — **`ConvFuser`** (D3): channel-concat camera-BEV ⊕ LiDAR-BEV → Conv2d–**GroupNorm/
  LayerNorm**(D6)–ReLU. A **named** module (`model.fusion`) for T5/T6 slicing.
- **`bev_neck.py`** — SECOND-FPN convs (GroupNorm; **no adaptive pooling**). **`head.py`** — CenterPoint
  dense head (per-class heatmap + regression: offset, z, log-dim, rot `sin/cos`, velocity).
- **`losses.py`** — Gaussian-target focal heatmap loss + L1 regression on matched centers. **Pin the
  corrected 3-case `gaussian_radius`** (`r = max(0, int(min(r1,r2,r3)))`, denominators `2·a` with
  `a1=1, a2=4, a3=4·min_overlap`, `min_overlap=0.1`) — **NOT** the historical CenterNet `/2`
  missing-denominator bug (it halves the radius and stalls the overfit). Target rendering is RNG-free +
  atomic-free (`torch.max` overlay + sliced assignment).
- **`detector.py`** — `BEVFusionDetector(nn.Module)` wiring the above; `forward` returns head outputs (+
  intermediate BEV features for V2/V3); **`decode()`** → 3D boxes+scores **in the T1 canonical convention**
  (`(cx,cy,cz,dx=l,dy=w,dz=h,yaw)`, yaw `atan2(sin,cos)` with T1's **MINUS** cross-term; **forbid the
  mmdet3d `−π/2` offset and `(l,w,h)` swap** the CenterPoint reference carries and T1 already banned).
  **Decode peak extraction = fixed 3×3 `max_pool2d` local-max mask (`keep=(hmax==heat)`) + a DETERMINISTIC
  top-k** — **`torch.topk` has NO `stable` kwarg and its CUDA tie order is not reproducible**, so use a
  composite recipe: zero non-peak cells via the max-pool mask, then break remaining ties with a monotone
  secondary key (`score − ε·flat_index`) before `topk`, **or** `torch.sort(stable=True)` + slice. **No
  post-decode box NMS** in the clean baseline (CenterPoint is NMS-free; circle/rotated NMS is deprecated
  AND non-deterministic — circle_nms is numba/CPU, `nms_bev` is a CUDA kernel — **banned**). The
  `(row,col)→(x,y)` mapping (`xs=ind%W, ys=ind//W → metric via voxel/out_factor/pc_range`) must be the
  **SAME** binding used by the splat, the pillar scatter, and the target rendering (declare which of
  `{W→x, W→y}` holds).
- **`training/tasks.py` — `NuScenesDetectionTask(Task)`** (register, `name="nuscenes_detection"`):
  `build_model`; `build_criterion` (the detection-loss module — note the loop calls
  `criterion(model_output, targets)` where `targets` is the batch); **`client_data`**: resolve `info_list`
  via `info_cache.load_cache(run_config["nuscenes-cache-dir"], run_config["nuscenes-version"],
  run_config["nuscenes-train-split"])` (**cache pre-built by T1; raise if absent — do NOT build the devkit
  on the login node**), partition via `build_log_group_partition(...)` → `partition["clients"][client_id]
  ["sample_tokens"]` **or** `iid_sample_partition(info_list, n, seed)[client_id]` (the two return
  **different shapes** — wire each explicitly), materialize `NuScenesMultimodalDataset(info_list, dataroot,
  sample_tokens=...)`, `make_loader(collate_fn=detection_collate)`. **`num_clients(run_config)` returns the
  DERIVED `partition["num_clients"]`** (which may have fallen back to N∈{20,25}), NOT the requested
  `nuscenes-num-clients`. `eval_loader` uses `nuscenes-val-split`. `evaluate` = overfit/training metrics +
  the center-distance proxy (matched on **metric BEV center L2 in the canonical frame**, decoded boxes).
- **`models/fusion/collate.py` — `detection_collate_fn`** (RNG-free): stack `images`/calibration `[B,6,…]`;
  batch LiDAR points (list-of-`[P_i,5]` **or** concat with a batch-index column — declare which); keep
  ragged GT as per-sample lists.
- **Generalize `training/loop.py`** (T1 authorized): an **additive** device-move + batch-size protocol
  handling a tensor **or** the multimodal batch; `criterion(model_output, targets)`. **Widen the
  `Criterion` type alias in BOTH `loop.py:22` and `tasks.py:32`.** Batch-size protocol: `targets.size(0)`
  for a tensor batch, `len(batch["gt_boxes"])` for the detection batch. **`dummy_regression` must stay
  BYTE-IDENTICAL** — capture the determinism-smoke `agg_checksum` *before* the edit and assert it
  unchanged *after*. Keep **Adam**, trainable-params-only (frozen backbone → no Adam state — D1, enforced
  by `loop.py:82-86`).
- **V2 renderers** (`viz/encoder.py`, stage `encoder`=V2): per-cam feature-norm, LSS depth-prob, cam→BEV
  norm, pillar occupancy, LiDAR BEV norm, response-at-GT. **V3 clean** (`viz/fusion.py`, stage
  `fusion`=V3): `camera_BEV_norm`, `lidar_BEV_norm`, `fused_BEV_norm` + the side-by-side cam/LiDAR/fused
  BEV-alignment overlay. *(Trigger-diff V3 panels = T5.)* Mirror the `viz/calibration.py` sibling-module
  pattern; do not rewrite `writer.py`.
- **Tests** `fl_v3/tests/test_model_*.py`; **build/env:** pin `torchvision==0.22.1 --no-deps` in
  `build_venv.sh` (after the nuscenes-devkit line) + a **login-node weights pre-cache** step
  (`swin_t(weights=…); resnet18(weights=…)`) + `export TORCH_HOME=/cephyr/users/gaohui/Alvis/.cache/torch`
  in `build_venv.sh` AND `run_in_venv.sh`/`run_alvis.sh` + an import-sanity assert that the two `.pth`
  exist and `numpy==1.26.4`/`torch.cuda` survive the install; config keys in `pyproject.toml`; this SPEC +
  `findings_log.md` + `docs/env.md` + the `determinism.md` correction.

**Out of scope / deferred:** the real Ray FedAvg run + wall-clock mitigations + IID-vs-non-IID gap
(**T3**); official `DetectionEval` mAP/NDS + 6-criterion ASR + V4 (**T4**); attacks/V5/V3-trigger-diff
(**T5**); defenses/V6 (**T6**); full-model FL ablation; LiDAR sweep accumulation, radar, map priors, TTA.

**Files:** `fl_v3/src/fl_v3/models/fusion/**` (new), `training/{tasks.py,loop.py}` (register + additive
generalize), `viz/{encoder.py,fusion.py}` (new), `tests/test_model_*.py`, `pyproject.toml`,
`scripts/{build_venv.sh,run_in_venv.sh,run_alvis.sh}` (torchvision + TORCH_HOME), `docs/env.md`,
`docs/determinism.md` (§0 correction), `collab/T2/SPEC.md`. **Consume-only (unmodified):** T0 `strategy/`,
`utils/runtime.py`; T1 `data/nuscenes/**` (**the schema is frozen — if a field is missing, raise a finding,
do NOT mutate T1**). `fl_v2/` untouched.

## 3. Invariants (must hold; Codex checks each)

- **Bit-determinism (crown jewel):** same-seed two central runs → `torch.equal` on every parameter after K
  steps, **on the A40 (SLURM)** and on CPU; backed by (a) a **static AST ban test** (no
  `scatter_add`/`index_add`/`index_put(accumulate=True)`/`F.grid_sample`/non-stable sort-topk in
  `models/fusion/**`), (b) a **permutation-invariance** test (permuted point order → byte-identical
  splat+scatter), (c) a **decode tie-permutation** test, (d) the **#76176** index-copy GPU test (scatter a
  known pillar → cell is non-zero & correct), (e) the **cumsum guard** (float CUDA cumsum doesn't raise &
  is `torch.equal` fwd+bwd on the GPU). All RNG via `derive_seed`/`seed_everything`; loaders via
  `seeded_worker_init`; **detection determinism holds at `num_workers=0` AND `>0` identically**;
  `preprocess` resize is RNG-free.
- **D1 (frozen backbone):** backbone `requires_grad=False`, `.eval()` **survives `model.train()`** (assert
  BN `running_mean`/`running_var` **bytes unchanged after a train step** — run this on the **ResNet-18 BN
  path**, not only Swin), no Adam state for backbone params (assert the optimizer param set excludes them).
- **D6 (normalization):** new modules GroupNorm/LayerNorm; any BN's policy declared; frozen backbone BN
  eval-mode; **no `AdaptiveAvg/MaxPool2d` in any trainable module** (CUDA backward raises).
- **Schema/convention contracts:** consume the T1 schema unchanged; **freeze T2's BEV grid + `(x,y)→
  (row,col)` mapping + decode convention** (== T1 canonical) in `collab/T2/SPEC.md` as the T2↔T4↔T5
  contract. An **encode→decode round-trip golden test** (render a known GT box to head targets, decode
  those exact targets, recover the box to tight tol incl. yaw sign) pins it numerically — a layout-only
  "obeys convention" check misses a paired encode/decode swap.
- **The model learns (falsifiable, not "monotonically-ish"):** on a fixed 1–2-scene overfit of K steps,
  `final_loss ≤ 0.2·initial_loss` **AND** the GT-proximity metric improves by a stated margin (e.g.
  recall@2m from ~0 to **≥0.5**, or mean best-match center-distance **halves**) **AND** an **anti-collapse**
  assertion (decoded-boxes-above-the-production-threshold > 0; ≥ a stated fraction of the scene's GT
  matched to **distinct** GT) — so collapse-to-background (loss↓, zero/one-coincidental box) fails.
- **Task-agnostic skeleton preserved:** `dummy_regression` determinism/regression tests pass byte-for-byte
  (captured-checksum assertion); no detection assumption leaks into the generic loop.
- **No false oracle:** T2 claims determinism + overfit-learns + V2/V3 + ground-truth-anchored calibration +
  per-module sanity — NOT bit-parity with any external model.

## 4. Reference (read-only, Apache-2.0; do NOT import mmdet3d/mmcv/spconv)

- **LSS:** nv-tlabs/lift-splat-shoot `models.py`/`tools.py` (`QuickCumsum`/`cumsum_trick`, `voxel_pooling`)
  — reproduce the algorithm; **the reference `ranks.argsort()` is non-stable (must change to
  `stable=True`)**. BEVDet `bev_pool_v2`.
- **Camera backbone/neck:** BEVFusion-MIT `SwinTransformer`+`GeneralizedLSSFPN`; torchvision `swin_t`/
  `resnet18` v0.22.1 (manual math attention — verified no SDPA). Weights: `swin_t-704ceda3.pth`
  (28,288,354 params), `resnet18-f37072fd.pth` (11,689,512 params).
- **LiDAR:** PointPillars (Lang 2019) + mmdet3d `PointPillarsScatter` (the `canvas[:,idx]=voxels` reference
  — **reimplement as `index_copy_` per #76176**; sparse-pillar tensor, not dense).
- **Fusion/neck/head:** BEVFusion `fusers/conv.py`; SECOND-FPN; CenterPoint (Yin 2021) `SeparateHead` +
  `_topk`/`gaussian_radius` (reimplement deterministic-topk + the **corrected** radius).
- **Determinism:** `fl_v3/docs/determinism.md` (after the §0 correction) + `utils/runtime.py`. PyTorch 2.7
  `use_deterministic_algorithms` op lists (the source of the §0 reality).
- **fl_v3 seams (verified):** T1 `data/nuscenes/{dataset.NuScenesMultimodalDataset, make_loader,
  info_cache.load_cache, partition.{build_log_group_partition, iid_sample_partition, coerce_partition_seed},
  paths, conventions.md}`; T0 `training/{loop, tasks.Task/ClientData/register_task}`, `viz/writer.VizWriter`
  (stages `encoder`=V2, `fusion`=V3), `client_app.py`/`engine/local_runner.py` (both call `train_local`).

## 5. Scientific failure modes to check (point Codex here)

- **A banned op slips in & strict mode misses it** (§0): `scatter_add`/`index_add` in splat/scatter,
  non-stable `argsort`/`topk`, the `canvas[:,idx]=` #76176 silent no-op → run-to-run-on-one-GPU passes,
  cross-arch/ARM drifts. Caught only by the **static AST ban + permutation-invariance + #76176 + cumsum**
  tests, on the **A40**.
- **BEV index-convention self-consistent-but-wrong** (the no-oracle trap T1 defeated with an independent
  projector): if splat+scatter+target+decode share the SAME wrong `(x,y)→(row,col)` mapping the
  self-consistency test passes while both BEVs are wrong. → **ground-truth-anchored** test (a real GT car's
  box center projected via T1's verified `lidar2img` AND its LiDAR points land in the SAME independently
  computed BEV cell) + an **injected-corruption negative** (row/col swap in ONE path → test FAILS).
- **Resize/intrinsic/`lidar2img` inconsistency** (no oracle, eyeball-only otherwise) → camera features
  mis-placed in BEV. → numeric ≤1px test anchored to T1's geometry.
- **D1/D6 violation** via `requires_grad`-only freeze (BN running stats update on `train()`); BN in new
  modules; adaptive pooling.
- **Decode convention drift** (mmdet3d `−π/2`/swap reintroduced; yaw sign) — caught by the encode→decode
  round-trip golden, not a layout check.
- **Degenerate "learning"** (collapse-to-background) — caught by the falsifiable overfit + anti-collapse.
- **gaussian_radius `/2` bug** stalls the overfit; **adaptive pooling** in trainable graph raises on GPU;
  **`num_workers>0`** drift if any preprocess/collate RNG isn't seeded.
- **Wall-clock blow-up** — the heavy **headline** config (frozen Swin-T + 6 cams + 256×704 + real grid) is
  the one T3/T5 need; reporting only the toy bring-up time hides T3 infeasibility.

## 6. GATE (objective pass criteria)

- [ ] **Bit-identical weights on the A40 (SLURM, the real gate):** a dedicated SLURM job asserts
      `get_device_name()∋"A40"`, exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` pre-CUDA, runs two same-seed K-step
      central trainings → `torch.equal` on every param; **commit the per-param SHA-256 checksum** into
      `collab/T2/SPEC.md`. Errors LOUD (never green) if no CUDA. Also passes on CPU.
- [ ] **Static AST ban test:** no `scatter_add`/`index_add`/`index_put(accumulate=True)`/`F.grid_sample`/
      bare `sort`/`argsort`/`topk` (without the stable/deterministic recipe) in `models/fusion/**`.
- [ ] **Permutation-invariance:** permuted input point order → byte-identical splat + pillar-scatter output
      (GPU). **Decode tie-permutation:** a heatmap with tied peaks decodes invariantly under row permutation.
- [ ] **#76176 + cumsum guards (GPU):** `index_copy_`/`index_put_` scatter of a known pillar yields the
      correct non-zero cell under strict mode; float CUDA `cumsum` (the splat) does not raise & is
      `torch.equal` fwd+bwd. *(Re-run on the ARM/H200 rebuild.)*
- [ ] **Trains centrally on mini end-to-end:** full multimodal forward + loss + backward + decode on real
      mini batches; non-degenerate detections; the **center-distance proxy > 0** (metric BEV center L2,
      canonical frame). *(Official mAP/NDS = T4.)*
- [ ] **Overfit (falsifiable):** fixed K; `final_loss ≤ 0.2·initial`; GT-proximity improves by the stated
      margin (recall@2m ~0→≥0.5 or center-dist halves); **anti-collapse** holds; reported as numbers/curve.
- [ ] **BEV ground-truth-anchored convention test** + injected-corruption negative; **resize ≤1px** numeric
      test; **encode→decode round-trip golden** (incl. yaw sign; no `−π/2`/`(l,w,h)`-swap).
- [ ] **D1/D6:** backbone frozen through `train()` (BN bytes unchanged after a step — **on the ResNet path**);
      no Adam state for backbone; Swin-T `IMAGENET1K_V1` loads + forwards deterministically; new modules
      GN/LN; no adaptive pooling in trainable modules.
- [ ] **Per-module param counts** (frozen vs trainable; **fusion as % of trainable** — the Q2 dilution
      seed); param-count asserts (Swin 28,288,354 / ResNet-18 11,689,512) double as the weights-loaded check.
- [ ] **V2 + V3(clean) renders** (≥3 mini samples each; manifest written); the V3 BEV-alignment overlay.
- [ ] **Task-agnostic intact:** `dummy_regression` determinism/regression pass **byte-for-byte**
      (captured-checksum assert); detection determinism identical at `num_workers` 0 and >0; a 1-step
      central smoke runs.
- [ ] **Wall-clock for BOTH the bring-up AND the headline D1 config** (frozen Swin-T, 6 cams, 256×704, real
      grid) — forward + full-train-step per mini batch on the A40 + a projected per-round estimate; recorded
      in `collab/T2/SPEC.md` for T3 planning.
- [ ] **Env:** `torchvision==0.22.1 --no-deps` pinned; login-node weights pre-cached; `TORCH_HOME` pinned in
      build + run scripts; post-install `numpy==1.26.4`/`torch.cuda` asserts; `docs/env.md` + the
      `determinism.md` §0 correction landed.
- [ ] **Tests green:** `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests` (T0+T1's **120** +
      new T2 tests) — record the count; note which tests REQUIRE the A40 SLURM job (the GPU-determinism set).
- [ ] **`collab/T2/SPEC.md` filled** (per-module param table + BEV/decode convention + A40 weight checksum +
      wall-clock) + `findings_log.md`; 2–3 least-certain items flagged for Codex.

## 7. Self-review — to be filled by the build session
(Predicted hardest review targets: (a) **§0** — the splat/scatter/topk being provably summation-order-
independent and GPU-deterministic on the **A40**, proven by the static-ban + permutation-invariance +
#76176 + cumsum tests, NOT by the now-hollow strict-mode raise; (b) the single shared BEV `(x,y)→(row,col)`
binding, anchored to T1 geometry (ground-truth, not self-consistency); (c) the frozen-backbone BN freeze on
the ResNet path; (d) the encode→decode convention round-trip; (e) `dummy_regression` byte-identity after the
loop generalization. Point Codex at the exact ops + the committed A40 weight checksum.)
