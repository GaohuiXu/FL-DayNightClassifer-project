# T1 — Build-session kickoff prompt (paste into a fresh Claude Code session)

You are the **build session for Cycle-04 task T1** of a thesis project on securing federated learning
for autonomous-driving perception. **T0 is complete and Codex-PASSed** (fl_v3 skeleton + determinism
harness + defense family + viz scaffold; 62 tests green). You build the **nuScenes multimodal data
module + the geographic log-group partitioner + the V1 calibration visualizations** on top of it.

**Read first, in order:**
1. `fl_v3/docs/roadmap/cycle_04_fusion_layer_backdoors.md` — the approved plan (focus: §FL setup,
   §Threat model, **§Attack spec "Client construction" + "Evaluation protocol & splits"**, Architecture,
   the Viz **V1** row, and task **T1** + its GATE).
2. `fl_v3/docs/cycle_04/README.md` — the orchestration/session model.
3. `fl_v3/docs/cycle_04/decisions.md` — confirmed D1–D8 (**D8 car-primary** binds T1; D1 is T3, but
   note the frozen-backbone direction).
4. `fl_v3/docs/cycle_04/tasks/T1_SPEC.md` — **your contract.** It is detailed and was verified against
   the installed `nuscenes-devkit` source; the conventions section + the frozen sample-schema table are
   load-bearing — honor them exactly.
5. Skim `fl_v3/collab/T0/SPEC.md` to see the bar for a filled SPEC, and `fl_v3/collab/findings_log.md`.

**Your job:** execute T1 to its GATE — a **bit-deterministic** nuScenes loader returning the frozen
canonical sample schema (synchronized 6-cam + `LIDAR_TOP` + full calibration + canonical-frame 3D-box
GT), the **location-coherent log-group partitioner** (N **derived** from a justified
`min-keyframes-per-client` floor — never hard-coded to 50), and the **V1 calibration renders** (a hard
pre-trust gate). Treat the **`nuscenes-devkit` as the geometry oracle** (it is your `fl_v2`-equivalent):
derive boxes/transforms from it, then unit-test your own reimplementation against it.

**Hard rules:**
- The dataset is **already fully extracted, read-only**, at **`/mimer/NOBACKUP/Datasets/NuScenes_v1.0/`**
  — **extract/copy nothing.** Ignore the stale "ZIPs at `/.../nuScenes`; extract mini first" text in the
  plan / CLAUDE.md (a *different*, wrong-layout `/.../nuScenes` dir also exists — do NOT point at it).
  Write **nothing** under `DATAROOT`; the info-cache lives under `nuscenes-cache-dir` (below `fl_outputs/`).
- **Bit-determinism is sacred.** Any RNG via `derive_seed`/`seed_everything`; `DataLoader` with
  `seeded_worker_init`; order by `sample_token` (across samples) + `ann_token` (within a sample); pin
  the image decoder (PIL convert-RGB); cache hash must be **host-portable** (DATAROOT-relative paths,
  fixed little-endian dtypes) so the Arrhenius rebuild reproduces it. No atomic scatter / `grid_sample`
  backward / non-stable sort/topk.
- **No `mmdet3d`/`mmcv`/`spconv`.** And **do not transplant mmdet3d's `-π/2` yaw offset or `(w,l,h)→(l,w,h)`
  swap** — the native nuScenes convention has neither (the SPEC cites the devkit proof). Derive every
  convention against the devkit; the SPEC gives the exact numeric parity tolerances.
- **Honor the frozen T1↔T2 schema table** (dtype·unit·frame·resolution per field): native uint8
  1600×900 images, **no resize/normalization in T1** (that's T2); pixel-space intrinsics; yaw in
  radians about +z; velocity rotated into the LiDAR frame; `gt_in_range`/`gt_num_lidar_pts`/`gt_visibility`
  carried with the exact semantics T4 needs. The dict schema deliberately does **not** fit the T0
  2-tuple loop/default-collate — **do not modify `training/loop.py` or `ClientData`**; the `collate_fn`
  is a T2 deliverable.
- **Avoid the gameable-GATE traps the SPEC calls out:** V1 must include an **independent-projector**
  numeric check (not just eyeballing — a wrong-but-self-consistent transform passes an eyeball); the
  partition logic must be **unit-tested on the real trainval log table** (mini is degenerate: ~6 logs,
  the N=20/25 fallback can't fire there); the floor value must be **justified**.
- **Mini = engineering smoke; trainval = science.** Stamp every partition/stats artifact with `scale`.
  Trainval is only *indexed/partitioned* (metadata-only, login-node) in T1 — no training, no scientific
  claim. Heavy work goes through SLURM, never the login node; run code via
  `fl_v3/scripts/run_in_venv.sh`.
- Write your SPEC to `fl_v3/collab/T1/SPEC.md` (from `fl_v3/collab/SPEC_TEMPLATE.md`, **including the
  frozen schema table**), add the tests, drive the `T1_SPEC.md` GATE checklist to green, and append to
  `findings_log.md`.

**When the GATE is green:** summarize what landed, paste the test count, list the 2–3 things the Codex
reviewer should scrutinize hardest (the SPEC predicts: the yaw/box convention + cam↔LiDAR ego-motion
composition, cache-hash cross-machine portability, and whether the V1 independent-projector check is
genuinely independent), and stop — the Codex review session reviews before T2 starts. Do not commit/push
unless the user asks.
