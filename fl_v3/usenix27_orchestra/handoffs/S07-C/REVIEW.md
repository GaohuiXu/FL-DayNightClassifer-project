# S07-C-R 独立审查 — legacy-security cleanup

## Gate verdict

**PASS at code/source/config/test/docs static-review scope** for worker
`a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`, handoff seal
`f736f41371666725a11d51bc3b01c6ececb59d50`, and review-launch base
`6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2`.

没有 P0/P1/P2/P3 finding。实现 diff 与 O-092、O-092-A1、O-092-A2 的精确删除面、
clean-foundation 保护面和 compute 边界一致；没有发现旧 attack/defense/harness 通过改名、
兼容 alias、registry、config selector、launcher 或 shared-test seam 回流。

这是**静态 cleanup PASS**，不是 dependency-backed、GH200、runtime、detector capability、
FL quality 或 scientific PASS。当前 login 环境缺少 NumPy、Torch、Flower、pytest、nuScenes、
spconv/cumm；本 session `APPROVED_COMPUTE=none`，因此所有 dependency-backed 与 GH200 检查
仍明确为 **NOT RUN**。

## Review identity 与基线

- session：`S07-C-R`；唯一写入为本文件；没有修改实现、canonical、handoff、结果或请求文件。
- audited clean foundation：`4ce2366df2925161adae8fea393d5fca64836d40`。
- canonical preparation：`4eba37d60cbeb9c865e4eec8d5fa57c90d23f873`，sole parent 为 audited
  foundation；相对 foundation 只改三份 canonical Orchestra 文档。
- canonical A1/A2 parent：`f7c696345b24b0e1227b1a52f3b47fb14e9120f5`，sole parent 为
  canonical preparation；相对 preparation 只改三份 canonical Orchestra 文档。
- implementation：`a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`，sole parent 为 exact
  canonical A1/A2 parent；tree `3a1a4fa49b3afc8cd0a919982746ef27b4dea487`。
- handoff seal：`f736f41371666725a11d51bc3b01c6ececb59d50`，sole parent 为 exact
  implementation；只改 `handoffs/S07-C/HANDOFF.md` 与 `RESULTS.md`。
- review-launch seal / startup HEAD：
  `6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2`，sole parent 为 exact handoff seal；
  只改 `ORCHESTRA.md`、`SESSIONS.md`、`KICKOFFS.md`。
- original detached snapshot（provenance only）：
  `9f06875e1b865734950abcf3b6de36ad06a0ac7b`；其 implementation patch-id 与 worker
  均为 `8f89c30d21164e80ec73f6a01eab33621e984789`。
- frozen old S07-B endpoint `e231808e77388d69053dcbced6e754dbe3468aef` 与 old spawn-policy
  reference `bf480ea77ccf9ae8417c3ea58e933701dbc7222a` 均不是 worker ancestor；只作
  read-only evidence，未 merge/import/copy/cherry-pick。

启动 preflight 实际结果：

```text
git status --short          = <empty>
git rev-parse HEAD          = 6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2
git branch --show-current   = <empty; detached>
git rev-parse --show-toplevel
  = /home/gaohui/.codex/worktrees/5d23/fl_weather_project
```

基线与 `EXPECTED_REF_MODE=detached@6d42e954...` 完全匹配。

## Findings（按严重性排序）

### P0 / P1 / P2 / P3

无 finding。

以下各节记录实际检查结论和未关闭的 runtime residual；它们没有被升级为代码 finding，原因是
受影响 executable semantics 要么与 accepted foundation 相同，要么正是 owner 明确批准的
cleanup，而未执行边界已被交接文件诚实标为 NOT RUN。

## 实现 diff、inventory 与 anti-recovery

审查对象严格分层：

1. `f7c696345b24b0e1227b1a52f3b47fb14e9120f5..a16c2cdfd4e23ba08677a66c45c50dd78340cc3b`
   为 implementation diff；
2. `a16c2cdfd4e23ba08677a66c45c50dd78340cc3b..f736f41371666725a11d51bc3b01c6ececb59d50`
   为 handoff-only seal；
3. `f736f41371666725a11d51bc3b01c6ececb59d50..6d42e9543bafb6bd971d5e0e8c36043ec8c64bd2`
   为 canonical review-launch seal；
4. protected foundation 直接与 `4ce2366...` 比较；没有把 `9f06875...` 当 continuation
   baseline。

Implementation 实际为 **137 paths = 70 deleted + 64 modified + 3 added**，
`1303 insertions / 12941 deletions`。`HANDOFF.md` 的删除、修改、新增 inventory 与
`git diff --name-status` 做 set equality 后无 missing/extra/duplicate；三条新增路径正好是
`HANDOFF.md`、`RESULTS.md`、`RUN_REQUEST.md`。

删除面结论：

- `fl_v3.attacks`、attack-specific ASR/frustum/report、attack/detection viz、T4/T5 configs/
  launchers/tests 已删除；
- defense namespace、gradient metrics、registry/strategy wrappers、oracle fixtures 与 defense tests
  已删除；
- O-092-A1 的三个旧 harness 与 O-092-A2 的 16 个 A40/T3/MCR/Stop-E/mini-matrix/profile
  scripts 均不存在；三个 A2 dedicated tests 亦不存在；
- 对全部被删除 script basename 的 `rglob` 为零；deleted code blob 在当前 `fl_v3/**` 中无
  exact SHA-256 survivor；新增文件均为 S07-C durable docs，因此没有 code rename/copy recovery；
- active `src/scripts/configs/tests/README/active docs` 的 AST import/import-from/runtime-string
  scan 对 removed modules 为零；`rg` 对 defense selector/registry、robust clipping/clustering/
  reweight/noise、malicious count、attack imports 与旧 launcher route 为零。

唯一保留的 deleted-name 字面量仍是
`fl_v3/tests/test_nuscenes_dataset.py:16` 对 `run_v1_calibration` 的已完成 ARM migration
provenance 注释。它不在 executable AST、不 import/invoke/launch，且该 S01 foundation test
不在 worker 修改授权内；不构成 active authority。

`t3_fl_gate.json`、`t3_trainval.json` 依 canonical kickoff 的 **REFACTOR-KEEP** 明确保留；
它们只删除 `defense-type`，没有指向已删除 launcher。相同地，retained clean capability 中的
训练侧 `grad-clip-norm` 是 `train_local` client optimizer 稳定性旋钮
（`engine/local_runner.py:90-104`），不是 FedAvg aggregator 的 robust clipping。Clean strategy
可执行树中不存在 `clip_norm`、clustering、reweight、noise 或 malicious-count identifier。

## 精确 18-script foundation

`fl_v3/scripts/` 当前文件集合与 kickoff 的 18 项 protected set 完全相等。相对 audited
`4ce2366...`：

- 16 个 protected scripts byte-identical；
- 只有 owner 在 O-092-A1 明确授权的 `arrhenius_smoke.py` 和
  `build_arrhenius_env.sh` 有变化；
- `centralized_train.py` byte-identical；没有残留旧 default launcher、profile、matrix 或
  compatibility harness。

## Clean FedAvg contract

实现满足 fixed clean path：

- `strategy/aggregation_core.py:15-46` 的 `fp32_weighted_average` executable AST 与
  audited foundation 完全相同，按 fixed client order 使用 `num-examples` 权重并以 FP32
  累加；原 defense-only fp64 aggregate 与 coordinate median 已移除；
- `strategy/flower_strategies.py:72-300` 只有一个 strategy class：
  `CleanFedAvgStrategy`。稳定 partition identity discovery 在 `:127-161`，seeded sampling 在
  `:163-225`，reply 依 partition-id 排序、num-example weighting、FedOpt 与 EMA 在
  `:227-287`；无 `DEFENSE_REGISTRY` 或 compatibility alias；
- `server_app.py:64-87` 无 defense/type 参数，只构造 `CleanFedAvgStrategy`；
  `:250-310` 仍用 trainable-only initial/result arrays，保存 full checkpoint、trainable
  checksum 与可选 server EMA；
- `engine/local_runner.py:107-186` 与 `:189-280` 的 public signature 均无 `defense`、无
  `**kwargs` selector；两条 runner 都对 replies 依 partition-id 排序，以 `num_examples`
  调用同一 FP32 primitive，再应用保留的 server optimizer；
- `training/tasks.py:93-134` 保持 requires-grad 过滤、ordered trainable state 与 fail-closed
  partial load；clean nuScenes loader `:832-882` 不再含 poison wrapper；
- `strategy/server_opt.py` 去 docstring 后与 audited foundation semantic-AST equal。

保留测试面包括：Flower FP32 bit-parity、strategy construction、deterministic identity/sampling、
clean runner API/determinism、FedAvg/FedAvgM/FedAdam、EMA/state knobs、trainable-only/full checkpoint。
这些文件存在且源码契约未被错误删除，但由于 login 缺依赖，本 review 未执行它们。

## Shared integration tests 与 clean test preservation

`tests/test_s07_b_integration.py` 当前保留 14 个 tests，覆盖：

- camera-only、pillar/SECOND LiDAR、fusion resolved construction 和 fail-closed mode/arch；
- runtime dependency/source manifest identity；
- deterministic CBGS/epoch-addressable sampler；
- official eval token completeness、duplicate/missing、K=500 guard 与 provenance；
- 五个 clean C/L/F candidate templates；
- six-task loss/global label mapping 与 legacy single-head refusal。

Cumulative diff 删除的四项都是已授权 legacy seam：T5 condition decode、mini-matrix telemetry、
old T4/T5 caller inventory、T4/T5 checkpoint caller。O-092-A2 特有删除为 mini-matrix test 与其
`_six_tasks` helper；原始 cleanup/A1 删除其余 attack/readiness seams。`_script_module` 只服务已删
caller test，亦随之清理。其余 14 tests 保留。

Clean FedAvg coverage 位于仍保留的专门 tests（`test_flower_fp32_parity.py`、
`test_flower_strategies_construct.py`、`test_fl_sampling.py`、`test_fl_round_smoke.py`、
`test_fl_local_runner_multiround.py`、`test_fl_server_opt_integration.py`、
`test_fl_trainable_only.py`），并非被挪入或藏入 deleted harness。没有发现误删未获授权的
clean-foundation test；三个看似 clean 的 Arrhenius camera/lidar/A40 dedicated files 是
O-092-A2 明确点名删除的 closed-session harness tests，核心 C/L/F contracts 仍由 accepted
S02-S05 tests 与上述 shared integration tests 覆盖。

## Protected foundation 审查结论

### S01 ZIP/data 与 centralized training

- `data/nuscenes/archive.py`、`dataset.py`、`info_cache.py` 与 audited foundation byte-identical；
- `gt_database.py`、`partition.py` 仅改 docstring，去 docstring 后 semantic AST equal；
- S07-A cache/provenance launchers均在 exact protected 18 中；
- `scripts/centralized_train.py` byte-identical。

因此没有改变 stored-ZIP routing、cache depth/sidecar/manifest identity、partition ownership、
centralized data/training path。实际 mini/ZIP/fork/spawn runtime 未重跑。

### S02-S05 clean C/L/F、head 与 official eval

- `models/fusion/{__init__,fusion,bev_grid,losses}.py` 去 docstring 后与 audited foundation
  semantic-AST equal；其余 camera/LiDAR/head/decode executable paths未改；
- 五个 `s07_b_{c_str8,l_p020,l_s075,f_u,f_cbgs}.json` 与 audited foundation byte-identical；
- `eval/box_to_global.py` 与 `eval/detection_eval.py` 去 docstring 后 semantic-AST equal；
- clean VizWriter 只保留 calibration/encoder/fusion/detection 四 stage，attack/defense stages
  与 trigger renderer 已移除。

没有发现 coordinate frame、yaw/class map、unit、K=500/no-starvation、official
DetectionEval 或 clean C/L/F constructor 语义漂移。Dependency-backed construction/loss/decode/
official eval 未执行。

### S06 runtime/config/checkpoint/resume

`config.py`、`training/loop.py`、`training/checkpoint.py`、`utils/runtime.py` 与 audited
foundation byte-identical；S06 provenance 保留完整 resolved config、data identity、mode、precision、
checkpoint hash 与 source SHA 绑定，删除的是旧 D10/attack-specific provenance API。没有把旧
security metadata 伪装成 clean resume authority。S06 runtime tests 未执行。

### Dependency contract

- `pyproject.toml:24-32` 直接固定 `scikit-learn==1.8.0`；
- `requirements.txt:32-35` 与 `requirements.lock.txt:104` 同样固定该版本；
- `scripts/build_arrhenius_env.sh:70-74` 对 `nuscenes-devkit==1.1.11` 使用 `--no-deps`，
  requirements 先提供其实际 clean runtime deps；
- builder/smoke 保留 sklearn import/version；active HDBSCAN import/check 为零。

这只证明 source/manifest contract，不证明 GH200 import 成功。

## HANDOFF / RESULTS / RUN_REQUEST 与 frozen evidence

三份 S07-C durable 文档与实际 Git/环境一致：

- exact identities、parent chain、patch-id、137-path inventory、18-script set 与 source counts
  均可重现；
- `a16c2cd..f736f41` 只改 HANDOFF/RESULTS，`f736f41..6d42e95` 只改 canonical launch docs；
- `/usr/bin/python3` 实测为 x86_64 Python 3.9.25，`numpy/torch/flwr/pytest/nuscenes/sklearn/
  spconv/pyquaternion` 均不可 import；文档将 dependency-backed/GH200 标为 NOT RUN 是诚实的；
- `RUN_REQUEST.md` 为 `NOT REQUESTED / NOT APPROVED`，没有 job ID、model output 或 scientific
  result；本 review 未发现 Slurm、dataset scan、environment mutation、merge/push/upload 行为；
- 交接只声称 static/source preservation，不把 compile/AST checks 升格为 runtime/scientific PASS。

作为 read-only anti-recovery 核验，本 review 还直接读取 frozen old S07-B 最终包与 R16，
并检查仍存在的 Job 352718 raw root：

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/
  outputs/s07b_mw_diag_764aab239094
```

`diagnostic_summary.json`、`execution_identity.json`、`diagnostic_run_config.json`、
`sha256sums.txt` 的 SHA-256 分别匹配 frozen RESULTS 的
`52fb107d...`、`0842ee3b...`、`886536a5...`、`fd7b9492...`；独立
`sha256sum -c sha256sums.txt` 全部 47 records 为 OK。Raw summary 仍为
`diagnostic_complete=true`、`artifact_complete=true`、`suite_pass=false`、9/9 nodes、
warning-fatal=true、all identities cleaned。它仍是 seven timeouts / five present JUnit failures /
four missing JUnit 的负面证据，不是当前 cleanup baseline，也没有被改写成 PASS。

## Adversarial checks actually run

以下命令均在 startup exact `6d42e954...` worktree 执行；未安装依赖、未运行 project import、
pytest、Slurm 或 GPU：

```bash
git status --short
git rev-parse HEAD
git branch --show-current
git rev-parse --show-toplevel

git show -s --format='%H | %P | %T | %s' \
  4ce2366 4eba37d f7c6963 9f06875 a16c2cd f736f41 6d42e95 e231808 bf480ea
git diff --name-status f7c6963..a16c2cd
git diff --stat f7c6963..a16c2cd
git diff f7c6963..a16c2cd | git patch-id --stable
git diff 4eba37d..9f06875 | git patch-id --stable
git diff --name-only a16c2cd..f736f41
git diff --name-only f736f41..6d42e95
git merge-base --is-ancestor e231808 a16c2cd
git merge-base --is-ancestor bf480ea a16c2cd
```

结果：`M=64/D=70/A=3`；两个 patch-id 都是 `8f89c30...`；handoff/launch seal
路径严格分离；两个 old refs 的 ancestor check 均为 false。

```bash
# Python 脚本分别重算 HANDOFF inventory set equality、exact script set、
# protected byte equality、deleted-blob SHA-256 survivor、AST signatures/imports、
# semantic-AST equality 与 retained test function inventory。
python3 - <<'PY'
# ast.parse / compile / git show / hashlib / pathlib checks
PY
```

结果：inventory `70/64/3` exact；scripts `18/18` exact；16 byte-identical + 2 authorized；
deleted exact blob survivor 0；removed-module executable/runtime-string hit 0；11 个 foundation
semantic-AST equal；`test_model_determinism.py` executable AST equal；
`fp32_weighted_average` executable AST equal；clean runner 无 selector/`**kwargs`；strategy class
只有 `CleanFedAvgStrategy`；shared integration 保留 14 tests。

```bash
rg -n -i '(defense-type|defense_type|DEFENSE_REGISTRY|NormTrackingFedAvg|...|t5_attack_eval)' \
  fl_v3/src fl_v3/scripts fl_v3/configs fl_v3/tests fl_v3/README.md fl_v3/docs/env.md
find fl_v3 -type f -name '*.json' | python3 -c '...json.load...'
python3 -c '...tomli.load("fl_v3/pyproject.toml")...'
python3 -c '...compile(each fl_v3/**/*.py, ..., "exec")...'
while read -r p; do bash -n "$p"; done < <(find fl_v3 -type f -name '*.sh' | sort)
git diff --check f7c6963..a16c2cd
```

结果：active legacy runtime route 0；Python source compile `135/135`；JSON `27/27`；TOML
parse + direct sklearn pin PASS；shell `17/17`；`git diff --check` empty；workspace 未产生
`__pycache__`。

```bash
sha256sum <Job-352718 four principal artifacts>
(cd <Job-352718 raw-root> && sha256sum -c sha256sums.txt)
```

结果：四个 principal hash exact；47/47 manifest records OK；没有改变 raw root。

## Explicit NOT RUN 与 residual risks

明确 **NOT RUN / NO IMPLIED PASS**：

- dependency-backed pytest、Flower FP32 parity、strategy construction、sampling、FedOpt/EMA、
  local runner 与 trainable/full-checkpoint tests；
- S01 real-mini/ZIP/cache/path/partition/fork/spawn lifecycle；
- camera/LiDAR/fusion construction、loss/head/decode、spconv sparse path、fp16/fp32 runtime；
- official nuScenes `DetectionEval` runtime；
- S06 resolved-config/precision/loop/checkpoint/resume runtime；
- Ray/Flower simulation、mini model step、full cache/trainval、100/1000-step、tiny-overfit、
  mAP/NDS、profile、DDP、matrix、seed/rerun/retry；
- Slurm/GH200、Protocol A/B execution、attack/defense/security/scientific cell。

Residual risks：

1. large deletion 的真实 import/runtime closure 仍需未来 exact accepted SHA 上的 dependency-backed
   focused tests；当前 AST/string/compile 不能替代 import 与 execution；
2. Flower 1.27 FP32 parity、partition discovery/sampling、FedOpt/EMA/checkpoint/trainable-only state
   虽有 retained tests 和 preserved code，未在本环境执行；
3. sklearn/nuScenes/spconv/cumm、C/L/F、official DetectionEval 与 S01/S06 lifecycle 需 Arrhenius
   target environment 才能关闭 runtime residual；
4. frozen S07-B 的多进程负面证据仍为 FAIL；S07-C 删除旧 harness 不构成对其 runtime failure 的
   修复或 clean S07-B completion；
5. retained source/test 中的 T3/MCR/A40 字样是 clean capability provenance/历史验证说明，不是
   executable launcher route；未来文档清理若要进一步改写这些 protected comments，仍需独立
   owner scope，不能借本 review 擅改。

## 允许与禁止的解释

允许解释：

- exact worker diff 静态删除了 O-092 指定的 legacy attack/defense/readiness/harness active routes；
- fixed clean FedAvg source contract、protected 18 scripts、clean C/L/F、official eval、S01、S06、
  centralized training 与 dependency manifest 在本 review 的 source/AST/byte/config 范围内通过；
- S07-C handoff 的 counts、hashes、negative boundary 与 NOT RUN 状态诚实一致；
- S00 可把本 verdict 作为是否接受 exact cleanup SHA、再准备后续 clean S07-B completion 的
  静态 gate 输入。

禁止解释：

- 不得称 dependency-backed tests、Flower/Ray、GH200、spconv、S01/S06 runtime、C/L/F runtime
  或 official DetectionEval 已在 exact worker 上 PASS；
- 不得称 centralized detector 已有 capability、mAP/NDS、fusion gain、FL quality、Protocol A/B、
  attack viability、ASR、defense、generalization、performance 或 reproducibility 科学证据；
- 不得把 frozen S07-B Job 352718 或更早负面结果改称 PASS，也不得称 cleanup 修复了其
  multiprocessing/runtime blocker；
- 本 review 不授权 remediation、clean S07-B completion、compute request/job、cache/trainval、
  commit/branch/merge/push/PR/upload/publication。

## Final verdict

**PASS — S07-C cleanup is accepted at independent code/source/config/test/docs static-review scope.**

Runtime/dependency/GH200 remains **NOT RUN**. 完成本文件后本 session 停止；不启动 remediation、
不启动 S07-B clean completion、不提交 commit。
