# S07-C HANDOFF — legacy-security cleanup（O-092-A1 + O-092-A2）

## 结论与交付状态

S07-C 明确 ACK owner 于 2026-07-13 批准的 `O-092-A1` 与
`O-092-A2`。两次 amendment 都在同一个 detached worktree、同一份未提交
diff 上完成，没有 reset、checkout、branch/ref/worktree 创建或代码恢复。

累计结果：

- active attack、T4 attack-readiness、T5、old defense/registry/gradient
  telemetry/oracle surface 已删除；
- clean aggregation 只有 `CleanFedAvgStrategy` 与固定顺序、按样本数加权的
  `fp32_weighted_average`；local runner 无 selector/compatibility keyword；
- scikit-learn `1.8.0` 作为 clean nuScenes runtime dependency 保留；没有
  HDBSCAN/FLAME import/check/code；
- VizWriter 只有 calibration/encoder/fusion/detection；
- O-092-A1 的 3 个 dead harness 与 O-092-A2 的 16 个 old/closed-session
  scripts、3 个专用 tests 全部删除；
- `fl_v3/scripts/` 现在精确只含 owner 指定的 18 个 active foundation files；
- `test_s07_b_integration.py` 只额外删除 obsolete mini-matrix helper/test seam；
  其余 clean integrated tests 保留；
- `losses.py` 与 `test_model_determinism.py` 仅改历史 authority 文案，
  executable AST 保持不变。

completed worker tree 已按 owner 的 durable-materialization authorization 封存：

```text
ORIGINAL_WORKTREE_BASE_SHA: 4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
CANONICAL_PARENT_SHA: f7c696345b24b0e1227b1a52f3b47fb14e9120f5
SNAPSHOT_SHA: 9f06875e1b865734950abcf3b6de36ad06a0ac7b
WORKER_IMPLEMENTATION_SHA: a16c2cdfd4e23ba08677a66c45c50dd78340cc3b
REVIEW_BASE: pending S00 canonical launch seal
DELIVERY_REF: pending S00 branch fast-forward/launch seal
APPROVED_COMPUTE: none
```

`SNAPSHOT_SHA` 是 detached original-base worker snapshot；该 commit 被 cleanly
cherry-pick 到 exact canonical parent，生成单 parent 的
`WORKER_IMPLEMENTATION_SHA`。没有创建 branch/ref、merge commit 或 push。
本 session 完成 docs-only handoff seal 后停止；不启动 reviewer。

## Git 身份与 amendment 启动复核

```text
SESSION_ID: S07-C
ORIGINAL_WORKTREE_BASE_SHA: 4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
AUDITED_CODE_BASE_SHA: 4ce2366df2925161adae8fea393d5fca64836d40
CANONICAL_PARENT_SHA: f7c696345b24b0e1227b1a52f3b47fb14e9120f5
SNAPSHOT_SHA: 9f06875e1b865734950abcf3b6de36ad06a0ac7b
WORKER_IMPLEMENTATION_SHA: a16c2cdfd4e23ba08677a66c45c50dd78340cc3b
SOURCE_BRANCH: codex/s07-c-legacy-security-cleanup
REF_MODE: detached@a16c2cdfd4e23ba08677a66c45c50dd78340cc3b before handoff seal
TOPLEVEL: /home/gaohui/.codex/worktrees/ab38/fl_weather_project
```

O-092-A2 编辑前：

```text
HEAD = 4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
branch = <empty; detached>
S07-C+A1 tracked diff = 117 files, +681/-8502
status = 51 deleted + 66 modified + untracked S07-C handoff directory
```

snapshot 前的累计 worker tree：

```text
tracked diff = 134 files, +674/-12941
status = 70 deleted + 64 modified + untracked S07-C handoff directory
untracked handoff files = HANDOFF.md, RUN_REQUEST.md, RESULTS.md
```

durable materialization evidence：

```text
4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
  -> 9f06875e1b865734950abcf3b6de36ad06a0ac7b  snapshot side commit

4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
  -> f7c696345b24b0e1227b1a52f3b47fb14e9120f5  canonical-only parent
  -> a16c2cdfd4e23ba08677a66c45c50dd78340cc3b  worker implementation
```

- snapshot 与 implementation 的 changed-path/status manifests 完全相同：
  本文件 inventory 的 70 deleted + 64 modified + 3 handoff additions；
- 两个 worker diffs 的 stable patch-id 均为
  `8f89c30d21164e80ec73f6a01eab33621e984789`；
- canonical parent 相对 original base 只改
  `ORCHESTRA.md`、`SESSIONS.md`、`KICKOFFS.md`；
- static verification 在 snapshot 前的 identical worker tree 上完成；canonical-only
  parent 不改变任何 worker code、config、test、script 或 handoff content。

没有 import/copy/cherry-pick `bf480ea...`、`e231808...` 或 legacy
T5/T6/T7 implementation。

## 上游证据与绑定身份

第一版已完整阅读 repository `AGENTS.md`、canonical `ORCHESTRA.md`、
`SESSIONS.md`、`KICKOFFS.md`、`docs/env.md`、accepted S01/S07-A 与 S02–S06
HANDOFF/REVIEW、实际 Git diffs，以及 frozen S07 evidence/raw artifacts。A1/A2
沿用该已核对上下文。

- S01 worker/review: `abe5c58b174dbbe1f7045ce91c8b15168d97b87b` /
  `7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc`。
- S07-A delivery/executable/review: `ba1571632557c20adbda3172221694cdbecfeabe` /
  `44cefd06bc815e893919d95c754896711dba3402` /
  `370ea6c0bd4d9d737a5a50b6aff1c6f742589825`。
- S02 worker/review: `3aebf2dc1d19473f29260df279421047d216d70e` /
  `df142dc9a391b87d05bd7becaba59459e9659f88`。
- S03 worker/review: `50893839c45cd3e2ef1b72b98db6668df7030f2a` /
  `2f62e570c9c24ef1e18a483888c3f28ad56a415e`。
- S04 worker/executable/review: `483e149b95ec891b675df825d924a96bb225b7dd` /
  `84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f` /
  `a0763c2e0b322d4ca53a92f9f69c90d9b231bbff`。
- S05 worker/executable/review: `a9c801fdee378906e54d06314d0c772b6559901a` /
  `96e509b71a3e22afb4de397132438fd3b9bbf5d8` /
  `1c440843bb2b6d72f10310ff11fcde0d7d1e885c`。
- S06 worker/executable/review: `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7` /
  `c330c72f4060348768c63fb1b7855ca56baffb95` /
  `ca7bbd7e49e91ac2f214f39f62d5e416dd736383`。
- frozen evidence only: `e231808e77388d69053dcbced6e754dbe3468aef`。
- read-only spawn-policy reference only:
  `bf480ea77ccf9ae8417c3ea58e933701dbc7222a`。

## 累计语义改动

### Clean security tombstone 与 FedAvg

- 删除 `fl_v3.attacks`、attack-specific eval/viz/config/script/test；删除 legacy
  defense namespace、registry、gradient metrics、defense tests 与 oracle fixtures。
- `training/tasks.py` 不再 wrap poisoned dataset；dummy/nuScenes、C/L/F、clean
  loader、partition/augmentation 保留。
- `flower_strategies.py` 只提供 `CleanFedAvgStrategy`；无
  `NormTrackingFedAvg` alias。
- `run_clean_round` / `run_clean_rounds` 无 defense 参数/validator/`**kwargs`；
  clean callers 直接调用 fixed runner。
- deterministic partition identity/order、num-example FP32 weighting、sampling、
  FedOpt、EMA、checkpoint/resume 与 trainable-only state 语义保留。

### scikit-learn dependency 纠错

- `pyproject.toml`、`requirements.txt`、`requirements.lock.txt` 均固定
  `scikit-learn==1.8.0`；lock 与 BASE 相同，故最终无 tracked diff。
- builder 对 `nuscenes-devkit==1.1.11` 使用 `--no-deps`。
- validated prefix raw source：`nuscenes/__init__.py` 导入 `.nuscenes`；
  `nuscenes/nuscenes.py:16` 无条件 `import sklearn.metrics`。
- builder/smoke 保留 sklearn import/version；HDBSCAN smoke 未恢复。

### O-092-A2 script/test subtraction

O-092-A2 删除 16 个 scripts：

```text
_bench_msweep.py
agg_overcommit_diag.py
arrhenius_lidar_gap_utils.py
arrhenius_mini_matrix.py
arrhenius_profile_mini.py
det_gate_a40.py
fl_gate_a40.py
p1_amp_smoke.py
p3_partition_health.py
run_arrhenius_mini_matrix.sh
run_arrhenius_profile_mini.sh
run_arrhenius_stop_e_gate.sh
run_v1_calibration.py
runconfig.py
t3_iid_vs_central.py
verify_levers.py
```

删除 3 个 dedicated tests：

```text
test_arrhenius_camera_audit_controls.py
test_arrhenius_lidar_gap_controls.py
test_fl_gate_refuses_non_a40.py
```

`test_s07_b_integration.py` 仅删除
`test_mini_matrix_six_task_telemetry_and_delta_use_every_branch` 及其只被该 test
使用的 `_six_tasks` helper。`test_model_determinism.py` 删除 A40 det-gate script
authority 文案；`losses.py` 删除 `verify_levers.py` historical proof 文案。

O-092-A1 曾把 partition diagnostic output 从 collab 移到 `fl_outputs`；A2 随后
删除整个 closed-session `p3_partition_health.py`，所以 A1 的 output-route check
已被 A2 的 file tombstone supersede。

## 18 个 protected scripts（精确全集）

```text
arrhenius_env.sh
build_arrhenius_env.sh
run_arrhenius_env_build.sh
arrhenius_smoke.py
run_arrhenius_smoke.sh
centralized_train.py
build_nuscenes_cache.py
build_gt_database.py
run_s01_nuscenes_zip_full_gate.sh
run_s01_nuscenes_zip_smoke.sh
run_s01_nuscenes_zip_tests.sh
s01_nuscenes_zip_audit.py
s01_nuscenes_zip_benchmark.py
s01_nuscenes_zip_manifest.py
s01_nuscenes_zip_smoke.py
run_s06_runtime_tests.sh
run_s07a_nuscenes_cache_t1v2.sh
run_s07a_provenance_tests.sh
```

目录 set equality 为真。除 A1 已授权修改的 `build_arrhenius_env.sh` 与
`arrhenius_smoke.py` 外，其余 16 个与 BASE byte-identical。

## 精确累计 path inventory

### 删除（70）

```text
fl_v3/configs/t4_a100_detgate.json
fl_v3/configs/t4_mini_smoke.json
fl_v3/configs/t4_reference.json
fl_v3/configs/t5_attack.json
fl_v3/configs/t5_mini_smoke.json
fl_v3/scripts/_bench_msweep.py
fl_v3/scripts/_t4_fd_diagnose.py
fl_v3/scripts/agg_overcommit_diag.py
fl_v3/scripts/arrhenius_lidar_gap_utils.py
fl_v3/scripts/arrhenius_mini_matrix.py
fl_v3/scripts/arrhenius_profile_mini.py
fl_v3/scripts/det_gate_a40.py
fl_v3/scripts/fl_gate_a40.py
fl_v3/scripts/p1_amp_smoke.py
fl_v3/scripts/p3_crt_probe.py
fl_v3/scripts/p3_grad_conflict.py
fl_v3/scripts/p3_partition_health.py
fl_v3/scripts/run_arrhenius_mini_matrix.sh
fl_v3/scripts/run_arrhenius_profile_mini.sh
fl_v3/scripts/run_arrhenius_stop_e_gate.sh
fl_v3/scripts/run_s07_b_static_checks.sh
fl_v3/scripts/run_v1_calibration.py
fl_v3/scripts/runconfig.py
fl_v3/scripts/t3_iid_vs_central.py
fl_v3/scripts/t3_trainval_reeval_fullval.py
fl_v3/scripts/t4_readiness_eval.py
fl_v3/scripts/t5_attack_eval.py
fl_v3/scripts/t5_mini_smoke.py
fl_v3/scripts/verify_levers.py
fl_v3/src/fl_v3/attacks/__init__.py
fl_v3/src/fl_v3/attacks/fusion_ablation.py
fl_v3/src/fl_v3/attacks/poison.py
fl_v3/src/fl_v3/attacks/poisoned_client.py
fl_v3/src/fl_v3/attacks/trigger.py
fl_v3/src/fl_v3/eval/asr.py
fl_v3/src/fl_v3/eval/frustum_visibility.py
fl_v3/src/fl_v3/eval/report.py
fl_v3/src/fl_v3/strategy/defenses/__init__.py
fl_v3/src/fl_v3/strategy/defenses/base.py
fl_v3/src/fl_v3/strategy/defenses/fed_median.py
fl_v3/src/fl_v3/strategy/defenses/fedavg.py
fl_v3/src/fl_v3/strategy/defenses/flame.py
fl_v3/src/fl_v3/strategy/defenses/foolsgold.py
fl_v3/src/fl_v3/strategy/defenses/multi_krum.py
fl_v3/src/fl_v3/strategy/gradient_metrics.py
fl_v3/src/fl_v3/viz/attack.py
fl_v3/src/fl_v3/viz/detection.py
fl_v3/tests/_attack_fixtures.py
fl_v3/tests/fixtures/make_oracle_fixtures.py
fl_v3/tests/fixtures/oracle_arrays.npz
fl_v3/tests/fixtures/oracle_decisions.json
fl_v3/tests/fixtures/oracle_inputs.npz
fl_v3/tests/test_arrhenius_camera_audit_controls.py
fl_v3/tests/test_arrhenius_lidar_gap_controls.py
fl_v3/tests/test_attack_ablation.py
fl_v3/tests/test_attack_poison.py
fl_v3/tests/test_attack_provenance.py
fl_v3/tests/test_attack_roster.py
fl_v3/tests/test_attack_trigger.py
fl_v3/tests/test_defense_edge_cases.py
fl_v3/tests/test_defense_parity_flame.py
fl_v3/tests/test_defense_parity_foolsgold.py
fl_v3/tests/test_defense_parity_normclip_median.py
fl_v3/tests/test_eval_asr.py
fl_v3/tests/test_eval_frustum.py
fl_v3/tests/test_eval_report.py
fl_v3/tests/test_fl_gate_refuses_non_a40.py
fl_v3/tests/test_gradient_metrics_parity.py
fl_v3/tests/test_multikrum.py
fl_v3/tests/test_viz_detection.py
```

### 修改（64）

```text
fl_v3/README.md
fl_v3/configs/fl_1client_sanity.json
fl_v3/configs/fl_bb02d_fedadam.json
fl_v3/configs/p1_bb02.json
fl_v3/configs/p1_bb02d.json
fl_v3/configs/p1_bb02d_voxel.json
fl_v3/configs/p1_bb02h.json
fl_v3/configs/p1_bb04.json
fl_v3/configs/p1_cbgs.json
fl_v3/configs/p1_exp3.json
fl_v3/configs/p1_gtpaste.json
fl_v3/configs/p1_msweep.json
fl_v3/configs/p1_msweep_aug.json
fl_v3/configs/p1_msweep_aug2.json
fl_v3/configs/p1_unfrozen.json
fl_v3/configs/t3_fl_gate.json
fl_v3/configs/t3_trainval.json
fl_v3/docs/determinism.md
fl_v3/docs/env.md
fl_v3/docs/roadmap/INDEX.md
fl_v3/pyproject.toml
fl_v3/requirements.txt
fl_v3/scripts/arrhenius_smoke.py
fl_v3/scripts/build_arrhenius_env.sh
fl_v3/src/fl_v3/__init__.py
fl_v3/src/fl_v3/data/nuscenes/conventions.md
fl_v3/src/fl_v3/data/nuscenes/gt_database.py
fl_v3/src/fl_v3/data/nuscenes/partition.py
fl_v3/src/fl_v3/engine/local_runner.py
fl_v3/src/fl_v3/eval/__init__.py
fl_v3/src/fl_v3/eval/box_to_global.py
fl_v3/src/fl_v3/eval/detection_eval.py
fl_v3/src/fl_v3/eval/provenance.py
fl_v3/src/fl_v3/models/fusion/__init__.py
fl_v3/src/fl_v3/models/fusion/bev_grid.py
fl_v3/src/fl_v3/models/fusion/fusion.py
fl_v3/src/fl_v3/models/fusion/losses.py
fl_v3/src/fl_v3/server_app.py
fl_v3/src/fl_v3/strategy/__init__.py
fl_v3/src/fl_v3/strategy/aggregation_core.py
fl_v3/src/fl_v3/strategy/flower_strategies.py
fl_v3/src/fl_v3/strategy/server_opt.py
fl_v3/src/fl_v3/training/tasks.py
fl_v3/src/fl_v3/viz/calibration.py
fl_v3/src/fl_v3/viz/fusion.py
fl_v3/src/fl_v3/viz/writer.py
fl_v3/tests/conftest.py
fl_v3/tests/test_determinism_smoke.py
fl_v3/tests/test_eval_provenance.py
fl_v3/tests/test_fl_config_keys_registered.py
fl_v3/tests/test_fl_local_runner_multiround.py
fl_v3/tests/test_fl_round_smoke.py
fl_v3/tests/test_fl_sampling.py
fl_v3/tests/test_fl_server_opt_integration.py
fl_v3/tests/test_fl_trainable_only.py
fl_v3/tests/test_flower_fp32_parity.py
fl_v3/tests/test_flower_strategies_construct.py
fl_v3/tests/test_model_bev_convention.py
fl_v3/tests/test_model_determinism.py
fl_v3/tests/test_model_params.py
fl_v3/tests/test_model_task.py
fl_v3/tests/test_s07_b_integration.py
fl_v3/tests/test_task_agnostic.py
fl_v3/tests/test_viz_writer.py
```

`requirements.lock.txt` 已恢复为 BASE pin，最终无 tracked diff。新增 handoff
package 为 `HANDOFF.md`、`RUN_REQUEST.md`、`RESULTS.md`。全部 tracked paths
均在 original kickoff、O-092-A1 或 O-092-A2 ownership 内，
`OWNERSHIP_UNKNOWN=0`。

## Protected-foundation 证明

- 18 个 protected scripts 全部存在；16 个 byte-identical，2 个只有 A1 已授权
  dependency-smoke 改动。
- S01 ZIP/cache/path implementation 未改；`gt_database.py`、`partition.py` 仅改
  docstring，semantic AST 与 audited base 相同。
- S02–S05 camera/LiDAR/fusion/head executable semantics 未改；fusion package、
  fuser、BEV grid、losses 均通过 semantic AST equality。
- S06 config resolution、precision、loop、checkpoint/resume/runtime source 未改。
- 五个 clean C/L/F templates byte-identical。
- official `box_to_global.py`、`detection_eval.py` executable AST 与 audited base
  相同。
- `server_opt.py` 与 `fp32_weighted_average` executable AST 保持相同。
- `test_model_determinism.py` executable AST 与 BASE 相同；
  `test_s07_b_integration.py` 不再含 mini-matrix seam，但保留其它 clean tests。

## 验证与 inclusive reference scan

详细命令/结果见 `RESULTS.md`。PASS：

- scripts set equality = exact protected 18；
- removed-module active AST import/runtime-string references = 0；
- active runtime/docs launcher routes for 16 removed A2 names = 0；
- Python compileall（external temp pycache）、27 JSON、TOML、17 shell
  `bash -n`、`git diff --check`；
- 11 个 foundation files semantic AST equal；losses included；
- clean runner/strategy/callers、VizWriter、FedAvg arithmetic、five-template checks。

inclusive scan 覆盖 `src/scripts/configs/tests/README/active docs`。只排除 canonical
Orchestra/handoff、`collab/**`、`docs/cycle_04/**` 与由 roadmap INDEX 明确标记的
frozen history。没有 active file import/invoke 任一 removed module。

唯一 active-tree 字面 residual 是
`tests/test_nuscenes_dataset.py:16` 对 `run_v1_calibration` 的 Alvis→ARM golden
decoder migration 注释。它位于 owner 未授权修改的 S01 foundation test，只是已
完成迁移的 historical provenance；不在 AST 中，不 import/invoke/launch，也不
构成当前 authority。其余 A2 deleted names 在 active tests/docs/runtime 为零。

## 明确 NOT RUN

login `/usr/bin/python3` 为 x86_64 Python 3.9.25，缺少 pytest、numpy、torch、
Flower、nuScenes；validated Python 是 ARM aarch64。`APPROVED_COMPUTE=none`，
所以以下均为 `NOT RUN`：

- dependency-backed pytest、Flower parity/sampling/FedOpt/EMA/checkpoint；
- S01 real-mini/ZIP/spawn tests；
- C/L/F construction/loss/head/decode/sparse LiDAR/fusion runtime；
- official DetectionEval runtime；
- S06 resolved-config/precision/loop/checkpoint/resume runtime；
- Ray/Flower/spconv、mini/trainval、100/1000-step、metrics/profile/DDP/matrix/retry；
- Slurm/GH200、Protocol-B、attack/defense/scientific execution。

没有安装替代依赖或修改 persistent environment。

## Residual risk 与 reviewer focus

1. owner 创建 exact worker commit/ref 后，reviewer 应独立重跑 exact-18 script
   set、removed-name AST/reference scan 与 cumulative security tombstone。
2. 另行批准的 Arrhenius run 应补 Flower parity、S01、C/L/F、official
   DetectionEval、S06 regressions；当前全部 NOT RUN。
3. 核对 sklearn direct pin 与 devkit `--no-deps` 的 clean runtime necessity，且
   确认 HDBSCAN/FLAME 没有回流。
4. 核对 `test_s07_b_integration.py` diff 只移除 mini-matrix test/helper，
   `losses.py` 和 `test_model_determinism.py` 只有文案变化。
5. large deletion 的 import-time residual risk 只能由 independent review 与批准后
   dependency-backed tests 关闭。

## 交接动作

本 identity update 只允许由一个 docs-only handoff seal commit 封存；其 diff
必须仅为 `HANDOFF.md` 与 `RESULTS.md`。向 S00 报告 seal SHA、完整 parent chain
与 clean detached status。`REVIEW_BASE`、`DELIVERY_REF` 仍等待 S00 launch
seal；不创建 branch/ref，不 push，不启动 reviewer。
