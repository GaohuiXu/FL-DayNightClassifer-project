# S03-R independent review — camera branch architecture

## Verdict

**PASS — the S03 module-level camera contract is accepted as a reviewed S07-B
dependency.**

This verdict applies to worker delivery
`50893839c45cd3e2ef1b72b98db6668df7030f2a`, implementation
`6dfd2c775f54e488f3930996b303ce21f9b8e8b7`, and the exact focused execution
snapshot `5c83daa1dffea6920e9918a3befec96e6db767c9`. It does not approve production
wiring, a full model, a 100-step gate, a performance/profile gate, trainval use,
metrics, or scientific claims.

## Findings, ordered by severity

No P0, P1, or P2 implementation/scientific-correctness finding remains in the
reviewed scope.

The following are required downstream boundaries, not findings against S03:

1. **S07-B must opt into the new path explicitly.** The current read-only detector
   still constructs `ImagePreprocessor` without `ImageAugmentationConfig` and its
   config/task defaults remain stride 16 with 1.0 m depth bins. S03 deliberately
   leaves that compatibility path in place because `detector.py` and `tasks.py`
   were outside ownership. Integration must bind reference augmentation, stride 8,
   0.5 m bins, the selected `bev_output_dtype`, and the common camera/S04 BEV grid;
   merely merging S03 does not satisfy O-017
   (`preprocess.py:17-20,254-271,339-342`; `view_transform.py:112-145,286-313`).
2. **The focused gate is not a production-shape/profile gate.** Job `336708`
   executes all module classes and one CUDA fp16 forward/backward, but the chained
   runtime fixture uses `64x96`, four depth bins, one camera, and a small BEV. The
   primary `256x704`, six-camera, 118-bin contract is constructor/arithmetic tested,
   not forward/backward profiled (`test_s03_camera_contract.py:226-246,249-312`).
   Therefore primary-shape peak memory, throughput, tiny-overfit, and the canonical
   100-step loss-decrease gate remain for separately authorized S07-B work.
3. **Reference-faithfulness is bounded, not byte-equivalence.** The archived MIT
   config uses Swin outputs at strides 8/16/32, BN, its own top-down
   `GeneralizedLSSFPN`, and `ImageAug3D` continuous-coordinate conventions. S03
   intentionally also consumes the existing stride-4 tap, uses GroupNorm and one
   all-level sum, and uses a self-consistent half-pixel pixel-centre convention.
   Those choices satisfy O-017's no-dead-level, batch-robust, framework-independent
   contract, but downstream prose must not claim exact MIT neck/augmentation
   equivalence (`camera_neck.py:31-119`; `preprocess.py:77-140`).

## Review identity and scope

- Review session: `S03-R`, independent from the worker session.
- Preflight: clean detached HEAD
  `50893839c45cd3e2ef1b72b98db6668df7030f2a`; branch name empty.
- Worker base: `372de9398ae435f82b83367a922fd302c0635738`.
- Worker branch/ref: `codex/s03-camera-architecture` at the exact delivery SHA.
- Reviewed diff:
  `372de9398ae435f82b83367a922fd302c0635738..50893839c45cd3e2ef1b72b98db6668df7030f2a`.
- Executed implementation: `6dfd2c775f54e488f3930996b303ce21f9b8e8b7`.
- Successful executable/tree:
  `5c83daa1dffea6920e9918a3befec96e6db767c9` /
  `d7125faffc6cb30c792ac2635dfda285f1dbe43c`.
- Read completely: repository `AGENTS.md`, active environment/roadmap, canonical
  Orchestra S03 contract/kickoff/review prompt, S03 `HANDOFF.md`,
  `RUN_REQUEST.md`, `RESULTS.md`, all changed camera source/tests/launcher,
  committed and `/nobackup` raw artifacts, the read-only historical camera audit,
  and the archived MIT BEVFusion reference at
  `326653dc06e0938edf1aae7d01efcd158ba83de5`.

The 19 changed paths are limited to the four owned camera modules, one focused S03
test, the S03 handoff/launcher package, and exact job evidence. No canonical
Orchestra file, `detector.py`, `tasks.py`, shared BEV geometry, data/split code,
`fl_v3/collab/`, or `fl_v2/` changed. The implementation source files are unchanged
between `6dfd2c7` and delivery; later commits add only approved launcher/evidence
material.

## Architecture and adversarial audit

### Swin taps and no dead levels

- `CameraBackbone` returns the four real torchvision Swin-T stage taps at strides
  4/8/16/32 with channels 96/192/384/768 (`camera_backbone.py:63-71,92-99,
  121-152`). This matches the actual torchvision stage order; no fabricated scale
  is advertised.
- `GeneralizedLSSFPN` validates equal nonempty channel/stride lists, uniqueness,
  increasing strides, feature count, batch, and channel shapes. Every declared tap
  enters its own lateral Conv/GN/ReLU, is resampled to stride 8, summed, and enters
  the smoothing block (`camera_neck.py:38-69,71-108`). There is no branch by which
  a declared tap is computed and then discarded.
- The hostile B=2 test checks nonzero finite gradients for each level input and a
  non-`None`, finite gradient for every neck parameter
  (`test_s03_camera_contract.py:141-164`). The dependency-complete GH200 case checks
  every trainable Swin/FPN/LSS parameter receives a finite gradient and that a
  changed image pixel changes BEV (`test_s03_camera_contract.py:249-312`).

### Image geometry and calibration

- The composed native-to-augmented affine order is resize, crop/pad, horizontal
  flip, then rotation. Resize uses realized integer dimensions plus the exact
  `align_corners=False` half-pixel translation; flip is `W-1-u`; rotation uses the
  declared pixel-centre convention (`preprocess.py:77-140`).
- The actual image path applies the same realized resize/crop/pad/flip and derives
  the inverse-sampling rotation grid from the final affine
  (`preprocess.py:365-444`). Both `K` and `lidar2img` are updated from that affine in
  float64 before returning in the calibration input dtype
  (`preprocess.py:436-454`). Negative crop origins are handled by explicit zero
  padding rather than accidental negative slicing (`preprocess.py:203-216`).
- Independent scalar projection fixtures cover crop, pad, flip, positive rotation,
  negative rotation, updated intrinsics, and updated `lidar2img`; the maximum
  declared residual threshold is `1e-10` (`test_s03_camera_contract.py:35-94`).
- Validation does not touch RNG and fixes resize 0.48, centered horizontal crop,
  bottom crop, no flip, and no rotation. The native nuScenes fixture produces
  `900x1600 -> 432x768 -> crop (32,176) -> 256x704`; repeated validation outputs are
  exact. Training replay is bound to an explicit CPU generator
  (`preprocess.py:150-200`; `test_s03_camera_contract.py:97-138`).

### Pure-camera boundary, BEV geometry, dtype, and memory

- The view-transform signature accepts camera features, calibration, `B`, and `N`
  only. It has no LiDAR points, LiDAR BEV, projected-depth, or cross-conditioned
  input; `LIDAR_TOP` is the output coordinate frame. Shape and floating calibration
  validation fail closed (`view_transform.py:104-145,179-195`). A hostile LiDAR
  keyword is rejected and unrelated LiDAR tensors cannot enter the graph
  (`test_s03_camera_contract.py:184-223`).
- Frustum construction matches the archived reference's `linspace(0, size-1,
  feature_size)` and `[1,60)` / 0.5 m contract. Metric mapping uses the shared
  read-only `BEVConfig` convention `W=x`, `H=y`, metres, floor binning
  (`view_transform.py:158-168,207-220`). The module does not invent a second box,
  yaw, or class convention.
- Strict and relaxed splats accumulate in fp32. Output is either fp32 or explicitly
  cast back to the input feature dtype; the contract exposes that choice and the
  exact BEV layout/shape (`view_transform.py:255-313`). S07-B must select it rather
  than inherit an accidental default.
- The primary unmasked lift arithmetic independently recomputes to
  `6*80*118*32*88 = 159,498,240` elements, i.e. 304.22 MiB fp16 or 608.44 MiB fp32.
  This is correctly labeled arithmetic, not a measured memory claim.

No data loader, dataset, split, annotation, optimizer, scheduler, EMA, metric,
decode, ASR denominator, or checkpoint is touched or executed. Consequently there
is no scene/sample/sweep leakage surface and no path by which LiDAR content can
silently condition the S03 camera module.

## Execution and provenance reconciliation

| Stage | Independent result |
|---|---|
| First client submission | Slurm rejected the command for invalid account/partition before returning a job ID. There is necessarily no scheduler/output artifact; the package correctly treats this as zero-GPU client-side negative evidence, not a camera result. |
| Job `335630` | Committed `sacct`/`scontrol`/stdout/stderr hashes verify. State `FAILED`, exit `1:0`, six seconds, no output root; stderr is exactly `fatal: not a git repository...`. Requested TRES was 1 GH200/8 CPU but allocated TRES was 4 GH200/288 CPU with `OverSubscribe=NO`. This is preserved as a provenance/resource failure before environment, CUDA, pytest, or camera execution. |
| Commit `2496fec` | Git inspection confirms its launcher still contains `#SBATCH --nodes=1` and legacy `--gres`; it was not executed and is not represented as evidence. |
| Job `336708` | Exact executable `5c83daa`, tree `d7125fa`, implementation ancestry, branch, request, launcher, source list/state, and read-only snapshot all reconcile. `sacct`/`scontrol` report `COMPLETED 0:0`, 89 s, one GH200/eight CPU requested and allocated, `OverSubscribe=OK`, `Restarts=0`, 504 MiB batch MaxRSS. JUnit and summary are exactly `10/0/0/0`; CUDA-visible count is one; environment is aarch64, Python 3.11.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128. |

Job `336708` externally approved identities independently match:

- RUN_REQUEST SHA-256:
  `0e6a22ecfd0d9f28d9a91e62bd23c425fac96a4e8d9ddeab81e2242e8225d615`;
- launcher SHA-256:
  `dc61bd2ebd2a88c8be717c8deb2bdfb848971bcf29fe3995e43e1f139f2bfaee`;
- source-list SHA-256:
  `d4eb8d29da926c88bbcf5c9bbbf9b3e9197f9eda4478ea956ec4c7cfaf664742`;
- source-state SHA-256:
  `197e5692e6d3c4477a3595cff39d831240b4419954bf929c7ff61e55b65a687e`;
- `execution_identity.json`:
  `37d70e7ada0de4cf7e2e197d8755ee090292efca2db766bd7384327c30b1bd28`;
- JUnit:
  `d1da1948fb71a0203e7dfa57a49d93c0de1cd3880c6ae87f9b62c76c38a652c5`;
- pytest log:
  `00f06042f110e6d11a21d87bf7334df22805392813f56b03879e7ee0467742f8`;
- allocation record:
  `48cacecd5ab0b70224347adb4e2cc04ba43c814fd33ffad94141bafcb5c8104d`;
- snapshot identity:
  `b990ac6e9064cdfa1f557f3e57eb7b307bebe9e9a82182b7c291e15fd0fdb615`;
- test summary:
  `fa9cdd0ad6e7402b511a3d2f4003179b99699ac15a1536233a2f931d3c6fc35a`.

All ten output artifacts pass `sha256sum -c`; all 15 source files pass the runtime
source manifest; the execution snapshot contains no writable path. The sole pytest
warning is the expected inability to create `.pytest_cache` inside that read-only
snapshot and did not affect test count/status. Committed raw scheduler/log copies
also pass their checksum manifests. Exact raw `scontrol`/stdout bytes contain a
trailing space, so a final whole-diff `git diff --check` reports those evidence
lines; normalizing them would corrupt the preserved raw hashes and is not requested.

Independent local read-only/static checks: Python byte-compilation and launcher
`bash -n` pass. No Slurm job or runtime/model rerun was performed by S03-R.

## Gate verdict

| Gate | Verdict | Evidence / limit |
|---|---|---|
| Swin-T taps, stride-8 multi-scale output | PASS | Source graph plus Job `336708` |
| No permanently dead declared FPN level | PASS | Source graph; nonzero input-level gradients; all neck params receive gradients |
| 0.5 m depth-bin contract | PASS | D=118 constructor/arithmetic fixture; exact archived config comparison |
| Resize/crop/pad/flip/rotation calibration | PASS | Independent scalar fixtures and source audit |
| Deterministic validation / seeded train replay | PASS | Job `336708`, exact repeats |
| Pure-camera, no LiDAR conditioning | PASS | API/source audit and hostile keyword rejection |
| Intended module finite-gradient/pixel sensitivity | PASS for focused synthetic shape | GH200 fp16 F/B; primary production shape remains downstream |
| Shape/dtype/memory contract | PASS as an interface/arithmetic contract | No measured primary-shape profile |
| Tiny-overfit / 100-step loss decrease | NOT RUN / NOT APPROVED | Required later before full-run readiness |
| Production integration and common S04 grid | NOT IN S03 SCOPE | S07-B must wire and revalidate |

## Allowed and forbidden interpretations

Allowed:

- accept `50893839...` as the independently reviewed S03 module dependency for
  S07-B;
- state that the focused synthetic geometry/interface/gradient suite passed on the
  attested GH200 fp16 runtime;
- state that S03 exposes a pure-camera stride-8/0.5 m contract with deterministic
  validation and self-consistent calibration updates;
- preserve both failed execution stages as scheduler/provenance negative evidence.

Forbidden:

- claim that merging S03 alone activates O-017 in production;
- claim exact MIT FPN/augmentation byte equivalence;
- claim measured primary-shape memory, throughput, tiny-overfit, 100-step, model,
  full-data, or training stability readiness;
- claim mAP, NDS, fusion gain, FL, attack/defense, generalization, scientific, or
  publication evidence;
- relabel Job `335630` as a one-GPU allocation or a negative camera/model result.

## Residual risk and integration requirements

1. S07-B must remove the legacy compatibility choice from every production entry
   point and bind one resolved config/hash for image augmentation, stride, depth
   bins, output dtype, and the reviewed S04 common BEV grid.
2. The first integrated gate must check camera-only topology skips LiDAR I/O and
   computation, not only that this isolated module lacks a LiDAR argument.
3. The separately authorized primary-shape forward/backward, memory/profile,
   tiny-overfit, 100-step, and later 1000-step gates remain mandatory before a
   full-data/model run.
4. Old stride-16/1 m/stretch checkpoints are not valid initializers for the new
   camera semantics without an explicit owner-approved migration policy; retraining
   remains the safe default.
5. Mini/synthetic execution remains engineering evidence only.

