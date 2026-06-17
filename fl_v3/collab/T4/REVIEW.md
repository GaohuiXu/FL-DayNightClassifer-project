# T4 — REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code. Copy to
> `fl_v3/collab/T<N>/REVIEW.md`.

## Verdict
`CHANGES-REQUESTED`

The core metric implementation looks scientifically sound: canonical-to-global conversion is devkit-anchored, `DetectionEval.evaluate()` is used directly, ASR uses the 6 criteria with `N = eligible-clean-detected`, and V4 consumes the shared decode. One gate invariant still needs hardening before T5 can rely on the READY artifact: the readiness verdict is checksum-bound, but not provenance-bound to the required D10 full-participation log-group training regime.

Reviewer checks run:
- `bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests/test_eval_asr.py fl_v3/tests/test_eval_box_to_global.py fl_v3/tests/test_eval_detection_eval.py fl_v3/tests/test_eval_frustum.py fl_v3/tests/test_eval_report.py fl_v3/tests/test_viz_detection.py -q` -> 24 passed.
- `bash fl_v3/scripts/run_in_venv.sh python -m py_compile fl_v3/scripts/t4_readiness_eval.py fl_v3/scripts/_t4_fd_diagnose.py fl_v3/scripts/t3_trainval_reeval_fullval.py` -> passed.
- `git diff --check a5fa47b..HEAD -- fl_v3` -> one trailing-whitespace style issue in `fl_v3/collab/T4/SPEC.md`.
- Inspected the installed `nuscenes-devkit==1.1.11` sources for `DetectionEval.evaluate`, `load_gt`, `filter_eval_boxes`, `DetectionBox`, and `algo.accumulate`.

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- Nothing found.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- invariant-violation · `fl_v3/scripts/t4_readiness_eval.py:202` / `fl_v3/scripts/run_t4_reference_a40.sh:52` · The READY predicate checks only metric floors plus `scale=trainval-scientific`, and the reference launcher only prints a warning for `fraction-train != 1.0` instead of failing. The warning is also rounded through `printf '%.0f'`, so values such as `0.9` would not warn. T4_SPEC §0.2 says the readiness checkpoint **must** be the full-participation log-group trainval FedAvg checkpoint and that a verdict computed on IID or sampled `fraction<1` is **INVALID** (`fl_v3/docs/cycle_04/tasks/T4_SPEC.md:36-46`). As written, an overridden `CONFIG`/`CKPT` can emit a READY `benchmark_readiness.json` for a sampled or IID checkpoint if the metric floors pass; the JSON records the checksum but not verified D10 provenance. Minimal fix: hard-fail the reference and readiness paths unless the loaded training provenance is `task-type=nuscenes_detection`, `nuscenes-version=v1.0-trainval`, `nuscenes-train-split=train`, `nuscenes-val-split=val`, `nuscenes-partition-mode=log_group`, `fraction-train == 1.0`, and `defense-type=none`; persist the exact training run config/provenance beside `final_model.pt`, verify it in `t4_readiness_eval.py`, and record the verified provenance in `benchmark_readiness.json`.

### question
- question · `fl_v3/tests/test_eval_box_to_global.py:148` · The durable contract asks for yaw parity on yaw-only-flattened boxes at `|wrap(Δyaw)| < 1e-4` (`fl_v3/docs/cycle_04/tasks/T4_SPEC.md:30-35`), while the implemented raw-annotation heading check accepts `< 0.02` because T1 stores pyquaternion `yaw_pitch_roll[0]` and devkit AOE uses `quaternion_yaw`. The independent lift-equivalence check at `fl_v3/tests/test_eval_box_to_global.py:147` and GT-as-pred AP sanity make this non-blocking for AP/ASR, and the documented ~0.004 rad AOE floor is negligible for NDS. Minimal fix if this is intentional: update the durable T4 contract or add a short note there mirroring `collab/T4/SPEC.md` §3a, so later reviewers do not treat the relaxed evaluator-heading tolerance as an accidental gate weakening.

### style
(deprioritized — note only, do not block on these)
- style · `fl_v3/collab/T4/SPEC.md:207` · `git diff --check a5fa47b..HEAD -- fl_v3` reports trailing whitespace. Minimal fix: remove the trailing space. Non-blocking.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: no blocking parity bug found. `box_to_global` composes `ego2global_lidar @ lidar2ego`, emits `size=(dy,dx,dz)`, rotates as `R(lidar2global) @ Rz(yaw)`, and the tests anchor it to raw devkit annotations plus devkit `load_gt` GT-as-pred. The only open item is the non-blocking yaw-tolerance contract question above.
- Invariants (determinism, null-config): one blocking gate invariant above. Decode avoids unstable `topk`, results JSON uses content-defined box order, readiness forces `batch-size=1`, frozen subset hashes and re-verifies targets/thresholds/checksum, and the T4-focused tests pass.
- Calibration/units: nothing found. The ASR thresholds are pinned in config, `tau_clean` matches the production decode threshold, `d_clean=2.0m` matches devkit `dist_th_tp`, `tau_pts=10` is a declared LiDAR-support floor, and coordinate/velocity conversion order matches the devkit oracle checks.
- Metric correctness: nothing found. Official mAP/NDS use `DetectionEval.evaluate()` with `detection_cvpr_2019`; ASR eligibility implements the 6 criteria; `N` is eligible-clean-detected; false-disappearance is undefined below `N_min`; and V4 reuses the same ASR matcher/decode.
