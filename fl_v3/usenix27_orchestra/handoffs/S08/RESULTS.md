# S08 precision qualification — terminal results

## Owner disposition

O-110 accepts close-ready seal
`d31adea049c84e47a0e4f82f38f22a2ca91a5a6f`, freezes the precision policy, and
closes S08 PASS. R3 reviewed exact evidence
`c0ef86235ead753fee3b790b19d40f82f875ec59` with
`PASS_WITH_RESIDUAL_RISK` and no P0-P2.

## Primary Q1 result

```text
JOB: 431013 / COMPLETED 0:0 / zero restarts
NODE/ELAPSED: n451 / 00:04:02
SOURCE: e6e28bea43f7757347da2e460cdf24e9a32b791f
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q1_dbeee35dcd6d
ARTIFACT_MANIFEST_SHA256: 5f606fc73b67fdbb188f20eb970c5040636960440b6f0cc093c2b98fe58202e2
WINDOW_RECORDS: 66
```

| Cell | Route | Attempts | Accepted | First accepted scale | Verdict |
|---|---|---:|---:|---:|---|
| C1 | C-STR8 FP32 | 3 | 3 | 1 | PASS |
| C2 | C-STR8 FP16 | 7 | 3 | 32 | PASS |
| L1 | L-S075 FP32/sparse FP32 | 3 | 3 | 1 | PASS |
| L2 | L-S075 FP16/sparse FP16 | 17 | 3 | 0.03125 | PASS after 14 overflows |
| L3 | L-S075 FP16/sparse FP32 island | 7 | 3 | 32 | PASS |
| F1 | F-U FP32/sparse FP32 | 3 | 3 | 1 | PASS |
| F2 | F-U FP16/sparse FP16 | 18 | 0 | none through 0.00390625 | bounded FAIL |
| F3 | F-U FP16/sparse FP32 island | 8 | 3 | 16 | PASS |

All accepted windows have finite scalar and six-task losses, complete finite
parameter gradients, finite retained boundary gradients, and exact
optimizer/scheduler/EMA/exposure accounting. Skipped windows advance none of
those counters; no accepted cell skipped after first acceptance. Per-mode
initial-state, fixture and replayed RNG identities match across precision regimes.

### Sparse-overflow localization

- L2 at scale 1 still has 158 nonfinite parameter-gradient elements, beginning at
  `lidar_encoder.backbone.stem.0.weight`. At scale 0.03125 the stem is finite,
  with unscaled maximum about 1.29M.
- F2's final attempted scale 0.00390625 retains ten nonfinite elements in that same
  first bad stem weight. Loss, head input, SECOND output/stage1/stem activation
  gradients are finite. The largest surviving unscaled finite element is about
  16.63M; its scaled magnitude is about 64.96K, next to the FP16 finite limit.
  The final backoff produced 0.001953125 but that scale was not attempted.
- L3/F3 keep SECOND in FP32 and accept at scales 32/16. This supports the FP32
  island as the stable current partition and localizes the practical failure to
  sparse-convolution weight-gradient dynamic range.
- FP32 references also show unusually large finite gradients: first L1 sparse-stem
  maximum about 1.92M and first F1 global maximum about 218K. S08 does not prove
  why they are large or that the head/loss is defective.

Key Q1 raw evidence:

| Artifact | SHA-256 |
|---|---|
| `raw/resolved_configs.json` | `7a8e142d40a750954958fba5bdada651aa5389f5d5dffe1c8e5612aa83795cbe` |
| `raw/window_records.jsonl` | `6e8b6f676bebfe67c6808f8d478be018188284473d6cfbcbe62752b756827bef` |
| `raw/q1_summary.json` | `3c30b017d689eb4fc32bf01f2c391d4647485adb30a96091a59b35c2b62e00de` |
| `raw/fixture_identity.json` | `6f44f71692a79a443b4fdce4abe528184e8eab7c61f018960815024ffab709b2` |

## Compatibility Q2 result

```text
JOB: 435151 / COMPLETED 0:0 / zero restarts
NODE/ELAPSED: n207 / 00:03:56
SOURCE: 3bb10d39c60e6fd2d0bfe480bb03a7c8cfc76fe9
OUTPUT: /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s08_q2_1d9191c2f623
ARTIFACT_MANIFEST_SHA256: 36b9cbf1eab30f54799cf7abbe83056ac009b301a7817d604a0c8b9abea5fb2f
WINDOW_RECORDS: 13
Q1+Q2_ELAPSED: 00:07:58
```

| Cell | Route | Attempts | Accepted | Scale | Verdict |
|---|---|---:|---:|---:|---|
| P1 | L-P020/global FP16/not-applicable/uniform | 7 | 1 | 8 | PASS |
| B1 | F-CBGS/global FP16/SECOND FP32 island/CBGS identity | 6 | 1 | 16 | PASS |

Only the accepted window advanced optimizer, scheduler and exposure. Every
parameter and applicable named boundary gradient was finite. Independent raw
inspection also found zero missing parameter gradients, although the now-retired
Q2 automatic predicate did not explicitly gate that count. B1 proves bounded
precision/config compatibility, not CBGS distribution or loader quality.

Key Q2 raw evidence:

| Artifact | SHA-256 |
|---|---|
| `raw/resolved_configs.json` | `529a8ea44f595edf8ee86b19e1632643c315237987a20d5c6dddbd226de03925` |
| `raw/window_records.jsonl` | `47dfa3407204f36f0da002334304fdb1c0795dc54d0e729fafb7f64a595a3f81` |
| `raw/q2_summary.json` | `211c2560ab207525e3ceeb66e0b73c3100b70473ce02cfed09ec21e91fb383e1` |
| `raw/fixture_identity.json` | `6f44f71692a79a443b4fdce4abe528184e8eab7c61f018960815024ffab709b2` |

## Engineering-smoke record

| Job | Result | Meaning |
|---:|---|---|
| 426619 | FAIL before pytest | provenance-policy defect; no model result |
| 427800 | 103 pass / 3 fail | diagnostics/test compatibility defects |
| 428112 | 106/106 PASS | pre-review focused implementation smoke |
| 428889 | 115/116 FAIL | no-op hostile calibration mutation; no fixture phase |
| 429080 | 116/116 + 1/1 PASS | final fixture-attestation smoke |

These failures are preserved and were not reinterpreted as environment/model
failures. Exact output roots are in `RUN_REQUEST.md`.

## Accepted conclusion and limits

The accepted policy is global FP16 for camera/dense-pillar, global FP16 with the
complete SECOND sparse route in FP32 for L-S075/F-U/F-CBGS, and FP32 as
reference/fallback. Full sparse FP16 is rejected as the current unified
fusion-capable route; the evidence does not prove AMP impossible at every lower
scale.

This is one replay-frozen mini fixture, random initialization, batch one, a
constant base recipe, at most three accepted primary updates and one compatibility
update. It is not convergence, performance, recipe quality, mAP/NDS, full-data,
multi-seed, Protocol A/B, attack, defense, or scientific-result evidence. The
large true SECOND gradient remains an explicit S10 work-definition input.
