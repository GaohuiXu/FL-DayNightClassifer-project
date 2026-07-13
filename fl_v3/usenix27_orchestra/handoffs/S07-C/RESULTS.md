# S07-C RESULTS — O-092-A2 cumulative local/static verification

## 修订声明

本文件 ACK `O-092-A2` 并 supersede A1 的 inventory/counts。A1 已纠正 sklearn
dependency、clean runner compatibility 与 VizWriter；A2 进一步删除 active-tree
old/closed-session script pollution。A1 对 `p3_partition_health.py` output route 的
验证被 A2 的 file deletion supersede，不再列作 retained behavior。

## 环境与 compute 边界

```text
login architecture: x86_64
login python: /usr/bin/python3 (3.9.25)
pytest/numpy/torch/flwr/nuscenes: UNAVAILABLE
validated prefix: present
validated prefix python: ARM aarch64 ELF; not executable on login node
APPROVED_COMPUTE: none
```

没有安装/uninstall dependency、修改 persistent environment、提交 Slurm/GH200、
运行模型或读取 full dataset。

## PASS — identity 与 cumulative diff

```text
show-toplevel = /home/gaohui/.codex/worktrees/ab38/fl_weather_project
HEAD = 4eba37d60cbeb9c865e4eec8d5fa57c90d23f873
branch = <empty; detached>
A2 startup S07-C+A1 diff = 117 files, +681/-8502
current tracked diff = 134 files, +674/-12941
current status = 70 deleted + 64 modified + untracked handoff directory
WORKER_SHA = pending
DELIVERY_REF = pending owner authorization
```

没有 reset、switch、new worktree/ref、commit、merge、push、upload、copy 或
cherry-pick。

## PASS — scripts 精确收敛与保护

```text
SCRIPTS_EXACT_PROTECTED_18=1
PROTECTED_SCRIPT_BYTES_EQUAL_BASE=16
PROTECTED_A1_EDITED_SCRIPTS_PRESENT=2
REMOVED_MODULE_ACTIVE_AST_REFERENCES=0
REMOVED_NAME_ACTIVE_RUNTIME_DOC_ROUTES=0
```

当前 `fl_v3/scripts/` 的完整文件集合：

```text
arrhenius_env.sh
arrhenius_smoke.py
build_arrhenius_env.sh
build_gt_database.py
build_nuscenes_cache.py
centralized_train.py
run_arrhenius_env_build.sh
run_arrhenius_smoke.sh
run_s01_nuscenes_zip_full_gate.sh
run_s01_nuscenes_zip_smoke.sh
run_s01_nuscenes_zip_tests.sh
run_s06_runtime_tests.sh
run_s07a_nuscenes_cache_t1v2.sh
run_s07a_provenance_tests.sh
s01_nuscenes_zip_audit.py
s01_nuscenes_zip_benchmark.py
s01_nuscenes_zip_manifest.py
s01_nuscenes_zip_smoke.py
```

AST scan 遍历 active `src/scripts/tests` 的 imports、import-from 与 runtime string
constants，没有任何 removed module reference。retained shell/runtime/docs 路由 scan
同样为零。

字面 scan 唯一 residual：

```text
fl_v3/tests/test_nuscenes_dataset.py:16
# Regenerate at the ARM migration with scripts/run_v1_calibration's decoder.
```

这是未授权 S01 foundation test 中已完成 Alvis→ARM golden migration 的 historical
provenance comment；不在 AST、不是 import/invoke/launcher/default route/current
authority。除此之外，16 个 A2 names 在 active tree 的允许范围内无命中。

## PASS — source/config/shell hygiene

```text
PYTHON_COMPILEALL_OK=1
JSON_OK=27
TOML_OK=1
BASH_N_OK=17
GIT_DIFF_CHECK_OK=1
workspace __pycache__ left behind = 0
```

命令：external `PYTHONPYCACHEPREFIX=<tmp>` compileall；全量 JSON parse；pyproject
TOML parse；全部 retained `fl_v3/**/*.sh` 执行 `bash -n`；`git diff --check`。

## PASS — focused semantic/AST preservation

```text
SEMANTIC_AST_EQUAL_AUDITED=11
MODEL_DETERMINISM_EXECUTABLE_AST_EQUAL_BASE=1
FEDAVG_ARITHMETIC_AST_EQUAL=1
CLEAN_RUNNER_NO_SELECTOR_OR_KWARGS=1
CLEAN_STRATEGY_ONLY_NO_ALIAS=1
CLEAN_CALLERS_NO_SELECTOR=1
VIZ_CLEAN_STAGES_AND_MANIFEST_OK=1
S07B_MINI_MATRIX_SEAM_ABSENT=1
CUMULATIVE_EXTRA_REMOVED_SCRIPTS_OK=19
A2_DEDICATED_TESTS_REMOVED_OK=3
FIVE_CLEAN_TEMPLATES_BYTE_UNCHANGED=1
```

11 个 semantic-AST-equal files（去 docstring 后相对 audited base）：

```text
src/fl_v3/__init__.py
src/fl_v3/viz/calibration.py
src/fl_v3/models/fusion/__init__.py
src/fl_v3/models/fusion/fusion.py
src/fl_v3/models/fusion/bev_grid.py
src/fl_v3/models/fusion/losses.py
src/fl_v3/data/nuscenes/gt_database.py
src/fl_v3/data/nuscenes/partition.py
src/fl_v3/eval/box_to_global.py
src/fl_v3/eval/detection_eval.py
src/fl_v3/strategy/server_opt.py
```

`test_model_determinism.py` 去 docstring 后与 BASE 相同。integration source 不含
`arrhenius_mini_matrix`、obsolete test 或 `_six_tasks`；其余 A1 clean integrated
tests 保留。VizWriter focused smoke 实际创建临时 clean manifest，stage set 只有
calibration/encoder/fusion/detection。

19 个 cumulative extra removed scripts = A1 的 3 个 dead harness + A2 的 16 个
old/closed scripts；不包括第一版已经删除的 T4/T5/security scripts。

## PASS — cumulative inclusive tombstone

扫描覆盖 active `src/scripts/configs/tests/README/docs`；只排除 canonical/handoff、
`collab/**`、`docs/cycle_04/**` 与 INDEX 明确标记的 frozen roadmap history。

```text
INCLUSIVE_LEGACY_TOMBSTONE_OK=1
ACTIVE security registry/import/config/launcher/viz route = 0
NormTrackingFedAvg = absent
local runner defense selector/kwargs = absent
legacy active filenames = absent
scripts outside protected 18 = absent
```

active broad security-word residual 仍只有否定性 assertions/docs 与明确 frozen rows，
不是 runtime route。O-092-A2 deleted-name scan 的唯一额外字面 residual是上述 S01
historical migration comment。

## PASS — corrected sklearn runtime dependency

```text
pyproject.toml: scikit-learn==1.8.0
requirements.txt: scikit-learn==1.8.0 under nuScenes runtime deps
requirements.lock.txt: scikit-learn==1.8.0
builder/smoke: sklearn import/version retained
builder devkit install: --no-deps
validated nuscenes/nuscenes.py:16: import sklearn.metrics
HDBSCAN/FLAME active import/check: absent
```

这是 source/lock evidence，不是 GH200 import PASS。

## NOT RUN — dependency-backed/runtime verification

以下全部明确 `NOT RUN`：

- Flower parity、strategy construction、sampling、local runner、FedOpt/EMA/
  checkpoint/trainable-state pytest；
- S01 ZIP/cache/path/partition real-mini/spawn tests；
- C/L/F construction/loss/head/decode/sparse LiDAR/fusion pytest/runtime；
- official DetectionEval runtime；
- S06 config/precision/loop/checkpoint/resume runtime；
- full pytest、Ray/Flower/spconv、mini/trainval；
- 100/1000-step、metrics/profile/DDP/matrix/retry；
- Slurm/GH200、attack/defense、Protocol-B/scientific execution。

原因：login 缺 dependency、target Python 为 aarch64、`APPROVED_COMPUTE=none`。

## Scientific interpretation

结果只支持 source cleanup、static preservation、corrected dependency contract 与
待 S00/independent review 的 worker diff。不支持 detector capability、mAP/NDS、
FL quality、performance、ASR、security behavior、Protocol A/B 或 reproducibility
scientific claim。
