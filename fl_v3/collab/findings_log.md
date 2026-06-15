# Cycle 04 — Findings Log (fl_v3)

Running, append-only log of triaged decisions and resolved review findings across
tasks T0–T7. One entry per resolved item.

Format:
```
## [T<N>] <date> — <short title>
- **Finding (severity):** …
- **Decision/fix:** …
- **Rationale:** …
```

---

## [T0] 2026-06-15 — nuscenes-devkit caps matplotlib<3.6 (manifest portability)
- **Finding (correctness-bug, build):** `nuscenes-devkit==1.1.11` publishes
  `matplotlib<3.6.0` (via the abandoned `descartes` map-rendering dep, which has
  no Python-3.12 wheel), conflicting with a modern matplotlib and breaking the
  portable manifest build.
- **Decision/fix:** install `nuscenes-devkit --no-deps` and supply its real
  runtime deps at modern py3.12/numpy-1.x versions; keep matplotlib 3.10.8. Build
  asserts the data + `DetectionEval` APIs import without `descartes`.
- **Rationale:** routing around an abandoned transitive is the Arrhenius
  portability posture; we don't use map rendering. Documented in docs/env.md.

## [T0] 2026-06-15 — self-verification sweep (8 module skeptics + completeness critic)
Ran an adversarial parity/determinism verification workflow before the Codex
handoff. FLAME, NormClip+FedMedian+metrics, and Partition verdicts = clean. Fixed
the actionable findings:
- **`seed_everything` masked numpy only** → now masks all three RNGs with one
  32-bit value (no stream desync for a seed ≥ 2**32; no-op for derive_seed leaves).
- **`enforce_determinism` docstring falsely claimed it "mirrors fl_v2"** (fl_v2 uses
  `warn_only=True`) → corrected to state the strict-raise default is an INTENTIONAL
  invariant strengthening for the AD model's no-banned-op requirement.
- **Client/server loaded arrays positionally** → switched to by-name
  `load_state_dict(arrays.to_torch_state_dict())` (matches oracle, robust to key order).
- **Strategy wrapper keyed output from the client reply** → now keys from the GLOBAL
  arrays (matching `new_global` order) + asserts reply key order == global key order.
- **`local_runner` FoolsGold ignored `foolsgold-head-index`** → forwarded.
- **`krum_scores` returned garbage if called directly with invalid f** → guard raises.
- **`DummyRegressionTask.evaluate` duplicated the eval loop** → delegates to `loop.evaluate`.
- Added `tests/test_defense_edge_cases.py` (FLAME n<4 / small-majority, NormClip
  scale<1 + decision aggregate, FedMedian odd-n, FoolsGold n==1).
**Documented (not bugs) for the Codex reviewer** — see SPEC §7.

## [T0] 2026-06-15 — unified fp64 aggregation core (parity scope, design decision)
- **Finding (correctness-bug per critic):** fl_v3 routes plain FedAvg + NormClip
  through the same fp64 `aggregate_weighted_updates` core as FLAME/FoolsGold, whereas
  the fl_v2 oracle's *clean*-FedAvg/NormClip aggregation delegates to Flower's fp32
  `aggregate_arrayrecords`. Algebraically equal (Σcoef=1) but not bit-identical.
- **Decision:** keep the unified fp64 core (one source of truth, higher precision).
  Bit-parity is CLAIMED only for FLAME + FoolsGold (the oracle computes those via the
  SAME fp64 core → exact). Clean/clip-path agreement with Flower's fp32 weighting is a
  tolerance-level T3 (real-Ray) check, not a T0 bit-identity claim.
- **Rationale:** the GATE requires FLAME + ≥1 other parity (both bit-identical);
  unifying avoids a second divergent aggregation path. Documented in
  `aggregation_core.py` + SPEC §7; flagged for Codex.

## [T0] 2026-06-15 — `--system-site-packages` numpy shadowing
- **Finding (invariant-violation risk):** an unpinned transitive pulled numpy
  2.4.6 into the venv, shadowing the module's CUDA-matched numpy 1.26.4 (breaks
  scipy + CUDA + determinism).
- **Decision/fix:** `constraints.txt` pins numpy/scipy to the module versions; the
  build asserts `numpy == 1.26.4` post-install.
- **Rationale:** the exact hazard CLAUDE.md warns about; the assert makes a
  regression loud.
