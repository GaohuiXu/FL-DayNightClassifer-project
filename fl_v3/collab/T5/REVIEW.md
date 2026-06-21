# T5 -- REVIEW (Codex)

> Written by the **Codex review session**. Reviews the build session's diff against `SPEC.md` + the
> paper/reference for **scientific correctness only**. Codex does **not** commit code. Copy to
> `fl_v3/collab/T<N>/REVIEW.md`.

## Verdict
`CHANGES-REQUESTED`

The reported trainval result is correctly gated as **not green**: cond-4 is non-viable, occlusion fails,
and the placement sub-gate is not passed in the saved JSON. I do not see a false positive fusion-aware
claim. I do see two mechanism/control implementation bugs and one hard-budget invariant bug that should be
fixed before treating the T5 negative mechanism explanation as final.

Reviewer checks run:
- `PYTHONPATH="$PWD/fl_v3/src" bash fl_v3/scripts/run_in_venv.sh python -m pytest fl_v3/tests/test_attack_trigger.py fl_v3/tests/test_attack_poison.py fl_v3/tests/test_attack_roster.py fl_v3/tests/test_attack_ablation.py fl_v3/tests/test_attack_provenance.py` -> 38 passed.
- The same pytest command without the explicit `PYTHONPATH` imported a stale T4 worktree package and failed collection; this is not counted as a scientific finding because the T5 SLURM launchers set `PYTHONPATH`, but the local test command needs that guard.

## Findings (severity-tagged)

For each: severity · exact file:line · why it's wrong (cite the SPEC/paper) · minimal fix.

### scientific-error
- Nothing found.

### correctness-bug
- correctness-bug · `fl_v3/src/fl_v3/attacks/poison.py:236` / `fl_v3/src/fl_v3/models/fusion/losses.py:137` · `relocation` always shifts the GT center by `(+delta_reloc, 0)` and never checks that the shifted center is still inside the CenterPoint head grid. `CenterPointLoss.build_targets` skips boxes whose shifted center is out of grid, so boundary cars silently become "trigger + no positive supervision" rather than the SPEC's center-relocation mechanism. T5_SPEC says the primary operator must keep the box so supervision survives at the wrong BEV cell (`fl_v3/docs/cycle_04/tasks/T5_SPEC.md:23-32`, `:117-119`); BadFusion also motivates relocation because label removal is ineffective for disappearance (arXiv 2405.03884, Sec. 5.1, lines 168-170 in the HTML). Minimal fix: before selecting/poisoning a target, compute the relocated center and require it to map to a valid head-grid cell; either choose a deterministic alternate offset direction that still exceeds `d_clean`, or skip/mark that target relocation-ineligible. Add a boundary test with a car near `x_max` proving `build_targets` still renders a relocated positive.

- correctness-bug · `fl_v3/src/fl_v3/attacks/trigger.py:294` / `fl_v3/src/fl_v3/attacks/trigger.py:317` · `compute_nonaligned_placement` implements the cond-2 control as "IoU==0 with target 2D boxes", but it never checks whether the patch footprint is actually LiDAR-sparse. The T5 contract requires non-aligned cond-2 to be a "LiDAR-sparse / no-target region" (`fl_v3/docs/cycle_04/tasks/T5_SPEC.md:114-115`, `:147-150`), while BadFusion's reference mechanism is specifically about dense 2D LiDAR projection regions (arXiv 2405.03884, Sec. 4.2 / Algorithm 1). As written, cond-2 can land on a dense non-target LiDAR projection, making it an uncontrolled fusion condition. This did not create a false green result in the saved run because cond-4 is already non-viable, but it would invalidate a future positive fusion-awareness claim. Minimal fix: pass/project the sample LiDAR points into the non-aligned resolver, require a pinned low projected-point count inside the patch footprint, record that count in the placement objective JSON, and re-run cond-2.

### invariant-violation
(bit-determinism / null-config / oracle-parity / banned-op / metric-definition)
- invariant-violation · `fl_v3/src/fl_v3/attacks/trigger.py:241` · `_patch_half` applies `max(spec.min_side_px, round(side / 2.0))` after capping `side` by the 0.30 budget. For small projected boxes, the minimum side can violate the hard trigger budget; e.g. a 20x20 px box gives a 12x12 patch, area ratio 0.36 > 0.30. T5_SPEC makes `patch area <= 0.3` a mandatory anti-occlusion gate (`fl_v3/docs/cycle_04/tasks/T5_SPEC.md:60-61`, `:106-108`, `:263-265`), and `collab/T5/SPEC.md:49-50` calls it hard-capped. Minimal fix: enforce the integer patch rectangle after rounding/min-side, reducing it to the largest budget-compliant size or returning `None`/invalid placement when `min_side_px` cannot satisfy the budget. Add a small-box regression test.

### question
- Nothing found.

### style
(deprioritized -- note only, do not block on these)
- Nothing found.

## Per-category "nothing found" (state explicitly)
- Reference/oracle parity: findings above. No public BadFusion code or `fl_v2` oracle exists for this new T5 attack, so parity is against the paper/SPEC semantics. The center-relocation and cond-2 sparse-control gaps are the parity issues I found; cond-5a zero-LiDAR readout matches the declared `ConvFuser` hook semantics.
- Invariants (determinism, null-config): one trigger-budget invariant above. I found no T5-specific banned-op/RNG issue in the reviewed attack code: roster and poison selection use `derive_seed`, rate-0/non-roster paths return the base dataset, ablation decodes are RNG-free, and the saved headline verdict is provenance-bound trainval.
- Calibration/units: no additional finding beyond the relocation grid-range issue above. I found no yaw/class-mapping or optimizer-scale constant transplant in T5.
- Metric correctness: nothing found. The attack eval loads the frozen subset with literal hash/checksum pins, forces `batch-size=1`, uses the frozen subset `N` denominator, scores disappearance on unedited val GT at inference, and uses official nuScenes `DetectionEval` for poisoned clean utility.
