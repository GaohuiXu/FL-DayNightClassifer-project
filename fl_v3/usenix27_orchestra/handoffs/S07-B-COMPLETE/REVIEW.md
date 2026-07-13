# S07-B-COMPLETE-R independent review

## Gate verdict

**PASS at the exact bounded clean-engineering scope** for candidate
`c615b6471a04b91a09c6ac6d487ff39a1501ceee`, executable test commit
`29ca6637bcd0a4e9a6422f3b820fb43d5295ad2c`, immutable F1 snapshot
`s07b_fp32_final_1b72abf2f8aa`, and Job `390576`.

No P0/P1/P2/P3 finding was identified. Independent raw-artifact inspection
establishes exactly five passing cases: clean identity FedAvg construction,
one real-mini FP32 optimizer step for each of C-STR8/L-S075/F-U, and exact
fusion first-batch equality for `num_workers=0` versus `2`.

This verdict is deliberately narrow. It is not a multi-step stability,
convergence, detector-capability, full-trainval, mAP/NDS, fusion-gain,
Protocol-A/B, performance, reproducibility, attack/defense, or scientific
PASS. It does not select the precision policy for later scientific training.

## Review identity and baseline

- accepted S07-C cleanup anchor:
  `70bcd856f7ebb411eb2887e7ab71ef41ed13271f`;
- S07-B-COMPLETE worker base:
  `4aa2b133d1d33382bf1514f7a3c86fcb03cf83e5`;
- focused executable commit:
  `34cbe02b7b72114e3a2d61f6f797c8dec022798c`;
- D1 diagnostic test commit:
  `1900fe3bcb52ade22f0b947a2aca44d5ece12b2f`;
- F1 test commit:
  `29ca6637bcd0a4e9a6422f3b820fb43d5295ad2c`;
- reviewed candidate / startup HEAD:
  `c615b6471a04b91a09c6ac6d487ff39a1501ceee`;
- branch ref `codex/s07-b-clean-completion` points to the same candidate;
  reviewer worktree itself was detached at that SHA.

The ancestry is exact: `70bcd856 -> 4aa2b133 -> ... -> 29ca6637 ->
c615b647`. Relative to `4aa2b133`, the candidate changes only
`fl_v3/configs/flwr_config.toml`, the new focused completion test, and S00
orchestration/handoff documents. `fl_v3/src/**` has no diff. Relative to the
accepted cleanup anchor, executable changes are still only the Flower config
and focused test.

Startup `git status --short` contained six pre-existing, uncommitted S00
document changes (`ORCHESTRA.md`, `SESSIONS.md`, `KICKOFFS.md`, plus this
session's `HANDOFF.md`, `RESULTS.md`, and `RUN_REQUEST.md`). They were read-only
review leads, not review authority. This review derives F1/D1 conclusions from
the durable test/config bytes, immutable snapshots, raw job artifacts and Slurm
accounting. No source, config, test, canonical, handoff, results, request, Git
topology, or compute state was changed by the reviewer.

## Findings

### P0 / P1 / P2 / P3

No findings.

The uncommitted terminal-ledger edits are a sealing step, not evidence used to
manufacture this verdict. S00 must make the independently verified terminal
record and this review durable in the later canonical acceptance seal; until
then, `c615b647` alone still describes F1 as approved but not yet submitted.

## Candidate diff and clean-only boundary

`git diff --check 4aa2b133..c615b647` passes. The Flower config removes the
stale supergrid, shared-GPU, four-GPU and overcommit profiles and leaves only:

- default `local-simulation-cpu`: eight supernodes, one CPU and zero GPU each;
- `local-simulation-gpu`: eight supernodes, one CPU and one whole GPU each.

The focused construction test calls the production `_build_strategy` and
independently asserts `CleanFedAvgStrategy`, identity `fedavg` server optimizer,
and zero server EMA. It does not execute a Flower/Ray round, so the result is a
clean-FedAvg construction/foundation check rather than an FL training claim.

Targeted scans of active `src/configs/scripts/tests/pyproject` found no attack,
backdoor, malicious-client, Neurotoxin, robust-defense, Krum, trimmed-mean,
coordinate-median, T5/T6/T7, attack-eval, or old launcher route. The removed
`attacks/**`, `defenses/**`, T5 config/launcher/test paths remain absent. No
legacy security implementation or scientific decision was recovered by this
candidate.

## F1 immutable identity and raw evidence

The F1 snapshot is:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/
  execution_snapshots/s07b_fp32_final_1b72abf2f8aa
```

Independent checks found 567 files, zero writable files, zero writable
directories, and reproduced tree SHA-256
`76c0bf5ba7ba0118a0150c3956cb5d8e9645a98adc54649a46a529bb96620d1c`.
The focused test is byte-identical to Git commit `29ca6637`; all four pinned
hashes match:

```text
test_s07_b_clean_completion.py  1b72abf2f8aaa9c98db9cabe994792187f976c5fbb267483967a58103b61c79f
flwr_config.toml                 2f459f816ad1bfcc9d1f9c1c2de9cc6491f5ea564eee633290e47665ff2003ab
arrhenius_env.sh                 f57befbb5082aaf4d4bb186958a88420ea873e0fdee5c65da1091b73f566c2bf
pyproject.toml                   29c5e81e56fdcb40a2caefdc8a91563ffcd1596df64fed6f4997eef3d58bab72
```

The retained exact temporary scripts pass `bash -n`; their bytes match the
request hashes `db4a5249...` (job body) and `015f701c...` (submit wrapper).
Their command exposes one GH200, uses `/tmp`, the persistent Arrhenius
environment, read-only mini data, exactly the three selected test functions,
and no D1/AMP/profile/metric/full-suite/extra-step cell.

Raw root:

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/
  outputs/s07b_fp32_final_1b72abf2f8aa
```

All six raw artifact hashes match the recorded values. JUnit independently
parses as exactly `5 passed / 0 failed / 0 error / 0 skipped` in 173.217 s on
`n105`. The log contains exactly three `S07_B_CLEAN_MODE_EVIDENCE` records; the
Slurm stdout contains exactly one `S07B_FP32_FINAL_GATE_PASS`; and
`final_gate.exit` is zero. `sacct` reports Job `390576` as `COMPLETED 0:0`,
elapsed `00:04:24`, one GH200, eight CPUs, 96 GiB, node `n105`, zero restarts.

The environment record is aarch64, Python 3.11.15, Torch 2.11.0+cu128/CUDA
12.8, cumm 0.7.13, spconv 2.3.8, one NVIDIA GH200 120GB and `TMPDIR=/tmp`.

| Mode | loss | finite gradient norm | optimizer / exposure | precision |
|---|---:|---:|---:|---|
| C-STR8 | 945.8136 | 58,677.4648 | 1 / 1 | FP32, scaler off |
| L-S075 | 1,562.8792 | 9,707,248.0 | 1 / 1 | FP32, scaler off |
| F-U | 1,122.4043 | 5,889,270.0 | 1 / 1 | FP32, scaler off |

Each record also has `nonfinite_loss_steps=0`, `grad_scaler_skips=0`, B=1,
ten sweeps and `num_workers=0`. L/F use the declared 4,096-point bound; camera
correctly has no LiDAR payload. The fifth runtime case recursively compares the
entire fusion batch and passed for worker counts zero and two.

## Production-loop and wrong-sample/shortcut checks

The C/L/F case does not substitute a toy model or ad-hoc loss. It constructs
`NuScenesDetectionTask`, whose production-runtime resolver fail-closes the
mode/architecture mapping, then builds `BEVFusionDetector` and
`MultiTaskCenterPointLoss`. It calls production `train_one_epoch` with AdamW,
where a finite loss is backpropagated, gradient finiteness is checked and the
real `optimizer.step()` is called before `optimizer_step`, successful-window
and exposure counters advance. A positive finite gradient norm plus all three
counters equal to one therefore establishes one accepted optimizer call/update
through the actual training seam. It does not establish useful learning or a
nontrivial parameter-distance threshold.

The fixture builds official `mini_train` infos with `n_sweeps=10` and selects a
real sample with nine prior sweeps. D1 used the same helper/sample and source
tree (the F1 snapshot differs only in the focused test) and reports `n_gt=15`
for all nine diagnostic cells, with six task-term records each. This rules out
an empty-target/background-only shortcut for the sample. F1 itself does not
persist the token or per-task target counts, so this target conclusion is an
inference from the byte-matched D1 fixture and artifacts, not a new F1 field.

## D1 and earlier negative evidence

Job `389356` artifacts independently reproduce all recorded hashes, exactly
nine JUnit cases, nine strict-JSON diagnostic records, one final marker and
`diagnostic.exit=0`. Slurm reports `COMPLETED 0:0`, `00:04:05`, one GH200 on
`n101`, zero restarts. The records confirm:

- all three FP32 controls have finite gradients and call the optimizer;
- FP16 scale 512 produces direct nonfinite elements and skips in C/L/F;
- FP16 scale 1 recovers C only, while L/F retain nonfinite sparse-SECOND
  stem/stage1 gradients and skip.

Thus D1 is diagnostic evidence, not an AMP PASS. The owner-directed F1 did not
quietly change production precision, GradScaler, model, loss, data, environment
or dependency code. FP16/scaler remediation remains explicitly outside this
session.

The recorded key hashes for Jobs `372819`, `373363`, `374142`, and `380806`
were also located under their exact raw roots and independently recomputed.
They preserve the Git-variable bootstrap failure, warning-fatal wrapper
failure, long-`TMPDIR` AF_UNIX failure, and first real six-task FP16 gradient
failure as negative evidence; none is relabeled as a training PASS.

## Reviewed inheritance and non-rerun boundaries

Because the candidate has no `fl_v3/src/**` diff, it does not alter the
previously reviewed S01 ZIP/data contracts, S02-S05 clean camera/LiDAR/fusion
constructors and six-task head/loss, official clean DetectionEval path, or S06
runtime/checkpoint/resume behavior. The S07-C cleanup review already checked
that these protected clean foundations survived legacy deletion.

Those are reviewed inheritance, not F1 re-execution. In particular, F1 did not
run the S01 real-mini ZIP/fork/spawn suite, S06 checkpoint/resume suite, official
evaluation, full trainval `t1.v2` materialization, or an FL round. Full trainval
cache materialization remains pending separate exact authorization.

## Residual risk and interpretation limits

- only one fixed mini sample and one optimizer step per mode were executed;
- LiDAR was capped at 4,096 points and no throughput/profile claim is valid;
- the F1 test proves an optimizer call and accepted runtime-state transition,
  not convergence, accuracy or multi-step numerical stability;
- FP16 sparse-path overflow remains an unresolved future precision-policy
  issue, not a failure hidden by this FP32 scope;
- the non-fatal spconv multidimensional-indexing warning is future-compatibility
  debt, not a failure of the pinned Torch/spconv runtime used here;
- no clean federated round, Protocol A, Protocol B, data-ownership split,
  checkpoint handoff, mAP/NDS, attack or defense was executed.

## Final acceptance recommendation

Accept S07-B-COMPLETE as the bounded clean integration engineering gate at
`c615b647`, with Job `390576` as exact one-batch/one-step FP32 runtime evidence.
S00 may now seal the terminal handoff/results/request facts and this independent
review in Git, then update the canonical ledger. Do not merge frozen
`e231808...`, do not restore T5/T6/T7 routes, do not treat FP32 F1 as the later
scientific precision decision, and do not schedule attack/defense or additional
compute from this PASS.
