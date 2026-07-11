# S07-A-R 独立审查 — reviewed data foundation 与待批准 full-cache 请求

## 审查身份与最终结论

- Session: `S07-A-R`，独立实现/集成拓扑/来源证明/执行请求/科学工程审查。
- 审查交付基线（`BASE_SHA = WORKER_SHA`）:
  `d41150692e0be40ac87a6a9346ef36f13c0eb3a7`。
- 待执行实现树（`EXECUTABLE_INT_A_SHA`）:
  `ed31f23b2ee1b193b5dd3600c00570e40a888ce9`。它与交付 SHA 不同；
  `d411506` 只在其后增加 `HANDOFF.md`/`RUN_REQUEST.md` 文档修订。
- S01 worker/review:
  `abe5c58b174dbbe1f7045ce91c8b15168d97b87b` /
  `7cf7fcc4b17d43806f1a134cf8c8a7b6868aa5bc`。
- `APPROVED_COMPUTE`: none。本审查未运行 `sbatch`/`srun`，未创建执行
  worktree，未修改请求，未替 owner/S00 批准执行。
- **S07-A implementation gate: CHANGES-REQUESTED。** ZIP/backend/depth/目录模式、
  集成拓扑与既有 focused evidence 通过；GT-database 尚未绑定能覆盖派生几何
  tensor 的物理 pickle/sidecar SHA。
- **Pending full trainval `t1.v2` cache request gate: CHANGES-REQUESTED。** 请求的
  commit/manifest/count/output/checksum/resource 边界基本正确，但声明为完整的
  runtime source-state 集合遗漏了 Python 实际执行的本地导入闭包。
- **Latest final verdict: CHANGES-REQUESTED。** 当前不能向 owner 提议按现有
  `ed31f23`/`7ddb06...`/output-root 三元组执行；本结论不否定已核验的 S01 历史
  coverage 或 Job 333477 focused-test 结果。

## 强制 preflight

审查开始前实际结果为：

```text
git rev-parse --show-toplevel
/home/gaohui/.codex/worktrees/2752/fl_weather_project

git rev-parse HEAD
d41150692e0be40ac87a6a9346ef36f13c0eb3a7

git branch --show-current
<empty; detached HEAD>

git status --short
<empty; clean>
```

与 kickoff 完全一致；未执行任何 Git/worktree 拓扑修复。

## Findings（按严重性排序）

### P1 — GT-database 只绑定 raw-input canonical hash，不能发现派生几何内容漂移

`build_gt_database.py:62-77` 只接收 `expected_cache_hash` 并调用
`IC.load_cache(..., expected_cache_hash=...)`；输出 `meta.json` 在
`build_gt_database.py:251-265` 也只记录 canonical `cache_hash`，没有记录或验证
待物化请求已经计划冻结的 cache pickle 与 sidecar 文件 SHA-256。

这个 canonical hash 不是完整 pickle/served-schema hash：
`info_cache.py:290-322` 只散列 `_cam_raw`、`_lidar_raw`、`_raw_boxes` 和 sweep
`_raw` 输入，不散列已序列化的 `gt_boxes`、`gt_velocity`、`lidar2img`、
`cam_intrinsics` 或 `lidar_sweeps[*].sweep2keylidar` 等派生数组；
`load_cache` 在 `info_cache.py:437-478` 也不会重新派生并比较这些数组。现有测试
甚至明确注明“hash alone only covers raw inputs”
(`tests/test_nuscenes_info_cache.py:71-72`)。

因此，只要 pickle 中某个派生 box/calibration/sweep transform 被改写、pickle meta
和 sidecar 保持一致，canonical hash 与 `expected_cache_hash` 都可继续通过；
GT-database 随后会在 `build_gt_database.py:205-230` 使用错误的点云变换或 box
裁剪对象。这直接触及坐标、sweep 与科学 provenance，而不只是文档字段。
`run_s07a_nuscenes_cache_t1v2.sh:207-227` 已经会产生物理 pickle/sidecar SHA，
`HANDOFF.md:274-293` 也把它们列为 production contract，但当前 GT caller 没有
消费它们；`HANDOFF.md:130-140` 所称的 cache provenance/fail-closed 范围因而过宽。

Required resolution:

1. 给 GT-database caller 增加必填的预冻结 pickle SHA-256 与 sidecar SHA-256
   （目录和 ZIP backend 都适用），在读取前后 fail closed，并把绝对路径、字节数、
   两个实际 hash 与 canonical hash 一并写入 `meta.json`；
2. 增加 hostile regression：只改派生 `gt_boxes` 或
   `sweep2keylidar`、保持 raw canonical inputs/meta/sidecar 不变，必须因物理
   artifact hash 不匹配而在任何点云裁剪前失败；
3. 后续 S06 resolved config 仍须独立绑定 version/split/depth、canonical hash、
   pickle/sidecar SHA、backend/manifest 与 resolved-config hash；修复 GT caller
   不能替代 S06 的全入口审计。

### P1 — pending launcher 的 source-state attestation 未覆盖实际 Python 导入闭包

`run_s07a_nuscenes_cache_t1v2.sh:50-63` 的 `runtime_source_files` 只列出五个
nuScenes 模块及 builder/launcher/environment/config 文件，并据此在
`run_s07a_nuscenes_cache_t1v2.sh:65-71` 计算
`7ddb06b3d57ef89be3b67782d90e93d64ddaa567ebd946ceda09910dc17b42f5`。
该值在当前 worktree 与 immutable `ed31f23` object 上都可精确重算，但集合本身
不完整。

执行 `from fl_v3.data.nuscenes import info_cache` 前，Python 必须运行 package
`fl_v3/src/fl_v3/data/nuscenes/__init__.py:8-15`。该文件又 eager-import
`dataset.py` 和 `partition.py`；`dataset.py:26-33` 继续导入 Torch、Pillow 与
`fl_v3/utils/runtime.py`，`partition.py:29-33` 继续导入
`fl_v3/data/partition.py`。这些实际执行的本地源码均未进入 source-state hash，
Torch/Pillow 版本也未进入 `execution_identity.json` 的 dependency set
(`run_s07a_nuscenes_cache_t1v2.sh:128-165`)。

虽然待提交命令在 `RUN_REQUEST.md:145-153` 提交前检查 clean detached HEAD，精确
HEAD 也正确，但未散列的已跟踪文件可在排队/执行之间发生 working-tree drift，
仍不会改变 `EXPECTED_S07A_STATE_HASH`；其中 `__init__.py` 是任意 import-time
代码入口。因而该 hash 不能满足 kickoff 所要求的“cache construction runtime
source file set is complete”，也不能作为完整 in-job source attestation。

Required resolution:

1. 要么消除 package 的无关 eager imports 并记录实际最小 import closure，要么
   至少散列整个 `fl_v3/src/fl_v3/data/nuscenes/`、
   `fl_v3/src/fl_v3/data/partition.py`、`fl_v3/src/fl_v3/utils/runtime.py` 及所有
   仍会执行的本地依赖；
2. 若仍实际 import Torch/Pillow，则在 runtime identity 中记录其安装版本；
3. 该修复会改变 executable tree 与 launcher source hash，因此必须生成新的
   executable SHA、source-state hash、以新 SHA 命名且确认不存在的 output root，
   并同步重写 exact command/`EXPECTED_S07A_SHA`/`RUN_REQUEST.md`。不能只改文档，
   也不能沿用 `ed31f23`/`7ddb06...` 的执行批准请求。

## Git topology 与 scope 审核

- Merge `60f603a0837a55b8bc5d56eedcbba065fcc10673` 的 parent 顺序精确为：
  1. `953bfb57941b5a3660ed650c1a80267cd82245d4`；
  2. `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`。
- S01 历史按顺序可达：
  `011e4640d26330e2c8145fcdb56833fe19e7b67d` →
  `1fe651700bd06a07707307c60ad4e31cc9d1e0ba` →
  `ce2e77284b290de4c9faa6b2f971c0bd52f98eff` →
  `54a48f9102fd0de9a9abe97701550740b547e769` →
  `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`。
- 原 review commit `7cf7fcc...` 不是 `d411506` 的 ancestor；它与
  cherry-pick `a4ca386db59a9250d3fce95209e38ac617b4ff77` 都只含
  `handoffs/S01/REVIEW.md`，且两个 commit 中该文件 blob 完全相同。review branch
  没有作为 implementation merge 进入。
- `a4ca386..c1f4fbe` 只触及声明的 S07-A data/docs/tests；
  `c7d5751..ed31f23` 只改 pending cache launcher；
  `ed31f23..d411506` 只改 S07 handoff/request 文档。
- `953bfb5..d411506` 没有修改 canonical
  `ORCHESTRA.md`/`SESSIONS.md`/`KICKOFFS.md`，没有进入 S02-S06 model/training/eval
  implementation。`d411506` 与 `60f603a` 均不是 `v3-ad-perception` ancestor，
  也没有 remote branch 包含 `d411506`；未见 merge/push 到长期分支的证据。
- 交付 SHA `d411506` 与 executable SHA `ed31f23` 的区分在
  `HANDOFF.md:20-37,88-90` 和 `RUN_REQUEST.md:91-102` 中正确。

## 数据基础实现与调用方审核

### S01 ZIP/backend

- v2 manifest 保留所有 `(path, archive)` occurrence，拒绝 archive 内重复及
  跨 archive size/CRC 冲突；正常路由选择最低 archive ID，audit 使用
  `read_archive_bytes` 读取精确 occurrence。local header signature、compression、
  flags、filename、payload length 与 CRC 均 fail closed
  (`zip_backend.py:148-331,528-598,600-645,684-781,830-869`)。
- 每个进程独立 lazy SQLite/FD state，fork hook、PID fallback、pickle state 与一次
  descriptor reopen 逻辑相互一致 (`zip_backend.py:403-527,737-781`)。Job 333206
  的 fork/spawn/persistent-worker JUnit 覆盖实际执行且零 skip。
- directory mode 在存在完整 sensor directories 且未显式传 manifest 时优先；
  shell 中 module manifest 不会劫持 extracted mini
  (`zip_backend.py:403-439`)。目录 byte/decode parity 和 writable LiDAR 在
  Job 333206/333477 的 real-mini 测试中通过。
- `t1.v2` 把 `n_sweeps` 绑定到 filename/meta/every record/canonical hash；load
  校验 sidecar、每记录 depth 与 canonical hash，dataset 再校验选中记录
  (`info_cache.py:353-478`, `dataset.py:187-219`)。历史 `t1.v1` 文件名不能由
  当前 API 解析为 production cache。

### 旧接口与 deferred consumers

- 在 `a4ca386` 对 `_decode_image_chw`、`sweeps_dir`、
  `iter_required_sensor_dirs` 的 call search 只找到定义；在 executable tree 中
  三个定义已删除，未发现实际 caller。移除不构成接口回归。
- `_infer_root_and_relative`、`_compat_blob_store`、legacy absolute `_load_lidar`
  仍有测试/兼容路径；directory helpers 和 `P.DATAROOT` 仍有历史调用方，保留合理。
- `training/tasks.py:580` 及 `arrhenius_smoke.py`、旧 benchmark/eval/attack scripts
  仍有未显式传 depth/hash 的 cache load。它们与
  `HANDOFF.md:229-241,342-343` 一致地属于 S06/S07-B；这是 S07-A scope 的
  可接受 defer，但在迁移完成前任何这些入口都不得成为 scientific production
  caller。

### GT-database 坐标/数据语义

- keyframe 与历史 sweep 都通过同一个显式 `NuScenesBlobStore`；历史 sweep
  由 cache 的 `sweep2keylidar` 变换到 keyframe LIDAR_TOP，再以同一 frame 的
  `(cx,cy,cz,dx,dy,dz,yaw)` box 裁剪，当前 frame/unit/column 顺序一致
  (`build_gt_database.py:189-230`, `dataset.py:128-154`,
  `gt_database.py:28-44`)。
- ZIP backend 绑定 manifest logical hash、SQLite file SHA 与 exact ten-archive
  set；directory backend 拒绝伪装 ZIP provenance
  (`build_gt_database.py:80-134`)。
- 但受上面 P1 影响，GT input 的派生几何内容尚未被物理 cache/sidecar identity
  保护。另一个明确边界是：默认 whole-official-train GT DB 只适用于 full-train
  CL capability。它没有 Protocol-B `D_base`/`D_tail`/client scene-log ownership
  subset manifest；不得把该 DB 用于 Protocol-B `W_base` 或 client adaptation，
  否则会把其他 owner 的对象点云复制进训练集。

## Raw artifact、hash 与 scheduler 独立核验

### Job 332651 — accepted historical full ZIP gate

- `sacct`: `COMPLETED`, `0:0`, node `n574`, `00:05:29` /
  `01:35:00`, one GH200, 32 CPUs；batch `MaxRSS=11048512K`,
  `MaxVMSize=71370624K`, `MaxDiskRead=21564396.27K`,
  `MaxDiskWrite=6545771.25K`, `TotalCPU=05:40.946`。
- 对原始 `sha256sums.txt` 执行 `sha256sum -c`：manifest、coverage、profile、
  两个 `t1.v1` pickle 与两个 sidecar 全部 `OK`。stdout/stderr SHA 分别为
  `5836dfe4ce50f67dca1adfb3d694531dcb35dc949f69fecdf219315aec4c727e` /
  `8db5d05b4abfa9c9cc1bd7028c410675c3e2d697af110ce6c6d9aa51f2e1e830`。
- Manifest file 为 633,106,432 bytes，SHA
  `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`；
  logical hash
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`。
  SQLite 独立查询得到 2,631,093 occurrences、2,631,084 unique paths；唯一重复
  path 是十份 `LICENSE`，size 25,319、CRC `0x48f670e8`，九个额外 occurrence。
- Coverage internal hash `cd0e298d...e50`、profile internal hash
  `397bfeaf...47e` 均重算一致；coverage 为 `538695/538695`、zero missing，
  camera/key/sweep 为 `204894/34149/299652`。十个 sentinel 是十个不同的
  archive-specific hidden text member。
- 历史 cache meta/count 为 train `28130/944881`、val `6019/187528`；这些值与
  builder stdout、coverage cache meta 和 split/reference counts 一致。其 cache
  格式仍是 `t1.v1`，且作业内没有 Git/source identity；只接受为 historical
  coverage/count/loader evidence，不是 production cache。

### Job 333206 — S01 remediation focused gate

- `sacct`: `COMPLETED`, `0:0`, node `n405`, `00:01:27`，one GH200/eight CPUs；
  batch `MaxRSS=552960K`, `MaxVMSize=6279744K`, `TotalCPU=00:19.221`。
- execution identity 精确为 `54a48f9102fd0de9a9abe97701550740b547e769` /
  source hash
  `260560ef3c5904825ad384825ec6755877748bbb403f65b5d5d907f1b7db1cda`。
  source list 18 个文件逐一从 immutable commit object 重算，0 mismatch。
- JUnit: 56 tests, 0 failures/errors/skips；包含 real-mini parity、fork、spawn、
  persistent lifecycle、2-vs-10 depth、local-header mutation 与 exact duplicate
  sentinel。artifact checksum list 与 stdout/stderr hash 全部匹配。

### Job 333477 — S07-A focused gate

- `sacct`: `COMPLETED`, `0:0`, node `n430`, `00:01:23`，one GH200/eight CPUs；
  batch `MaxRSS=36864K`, `MaxVMSize=6017600K`, `TotalCPU=00:19.544`。
- execution identity 精确为 `c1f4fbeade20975fd648e8d6c109f50d27f2bbf4` /
  source hash
  `dddca872e681a3616c279d9d41fb957f80ef1e780eb9a26705207bdf4269e544`；
  source list 逐一从 immutable `c1f4fbe` object 重算，0 mismatch。
- JUnit: 62 tests, 0 failures/errors/skips；包含五个 GT/cache provenance case 和
  保留的 57 个 S01/cache case。`sha256sum -c` 对 identity/source-list/log/JUnit
  全部通过；stdout/stderr SHA 与 RESULTS 一致。
- `c1f4fbe..ed31f23` 唯一 executable 差异是 cache launcher 的 43-line 修订。
  因而 Job 333477 **没有**执行或验证后来的 `ed31f23` launcher；该修订仅有
  `bash -n`、`py_compile`（未变 Python modules）、`git diff --check` 与 source
  hash 静态证据。不得把 62 tests 描述为对 `ed31f23` 请求 remediations 的执行
  evidence。

## Pending full-cache 请求的对抗性检查

| 检查项 | Verdict | 独立证据/边界 |
|---|---|---|
| approval state | PASS | `RUN_REQUEST.md:84-90` 明确 `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT`；本 review 不批准 compute。 |
| fresh executor topology | PASS (request text) | exact command 检查 empty branch、clean status、`detached@ed31f23`；未来仍须由 owner/UI 新建并在提交时重验。 |
| executable/delivery distinction | PASS | executor 用 `ed31f23`，review/delivery 用 `d411506`。 |
| command/SHA/source/output binding | **FAIL** | command 数值彼此一致，但 source-state set 不完整；修复后 SHA/hash/output 三者都必须更新。 |
| proposed output root | PASS NOW | `/.../s07a_cache_t1v2_ed31f23b2ee1` 独立检查为 absent；launcher 在任何 `mkdir` 前拒绝存在路径。修复后不能复用此旧-SHA 名称。 |
| accepted manifest logical/file/format/archive set | PASS | 原始 file checksum、SQLite metadata/rows 与 exact trainval01..10 已独立验证；launcher 在 output creation 前比较 logical/file/name set。 |
| manifest rebuild/mutation | PASS | launcher 只有 read/summary/SHA；没有 manifest builder 或 `--force`。 |
| runtime source completeness | **FAIL / BLOCKING** | `__init__.py` 及其 eager import closure 未散列；详见 P1。 |
| Python/platform/dependency capture | CHANGES-REQUESTED | Python/platform + NumPy/devkit/pyquaternion 已记录；当前实际 eager imports 还需记录 Torch/Pillow，或消除这些 imports。 |
| train/val counts | PASS (implementation; execution pending) | exact 28130/944881、6019/187528 与历史实际记录/meta 一致；launcher 将对新 `t1.v2` 的 actual-record sum 与 meta 分别比较。尚无新 artifact。 |
| `t1.v2`, depth 10, every-record/content validation | PASS (implementation) | explicit builder/load depth；`load_cache` 验证 format/sidecar/every-record depth/raw canonical hash。物理 served content 由 post-build pickle SHA 冻结。 |
| cache/sidecar identities | PASS (launcher design) | `cache_identity.json` 记录 canonical hash、路径、bytes、pickle/sidecar SHA，随后 checksum list 覆盖全部文件。 |
| checksum generation then verification | PASS (implementation) | `run_s07a...sh:238-249` 先生成、后独立 `sha256sum -c`。 |
| output reuse/retry/array/DDP/follow-on | PASS | fresh-root guard；单 job；没有 retry/array/DDP/model/profile/metric/follow-on code。 |
| resource scope | PASS | one GH200/eight CPUs/30 min/0.5 GPU-hour；GH200 是已验证 aarch64 prefix 的执行平台，历史 metadata/cache gate 显示 30 min 足够保守。 |
| request may be proposed unchanged | **NO** | 两个 P1 必须先修复并重新形成 immutable request；当前请求不能送 owner 执行批准。 |

## S07-A implementation gate-by-gate verdict

| Gate | Verdict | Evidence / limit |
|---|---|---|
| exact S01 history/review topology | PASS | merge parents、ancestor chain、review-only blob 均独立验证。 |
| no canonical/S02-S06/v3 merge/push contamination | PASS | actual range/name/reachability/remote-ref audit。 |
| ZIP routing/local-header/CRC/handle lifecycle | PASS WITH HISTORICAL LIMITS | code + jobs 332651/333206；并非 every-payload CRC 或 long-epoch contention。 |
| directory mode preservation | PASS | explicit mode logic + real-mini directory/ZIP byte/decoded parity。 |
| `t1.v2` depth/sidecar/raw canonical fail-closed | PASS | code + 56/62-test JUnit；历史 `t1.v1` 禁止 production。 |
| old-interface removal | PASS | pre-removal only definitions, no callers；replacement tests executed。 |
| GT caller removes hardcoded `t1.v1` | PASS | explicit current API/depth/manifest path。 |
| GT caller full artifact provenance | **CHANGES-REQUESTED / BLOCKING** | canonical hash 不覆盖派生 tensor，caller 未绑定 pickle/sidecar SHA。 |
| focused test evidence | PASS FOR `c1f4fbe` | 62/62；不能外推为 `ed31f23` launcher execution 或 trainval/model evidence。 |
| deferred production consumers | NOT YET READY | 合理属于 S06/S07-B；完成前不能 scientific run。 |
| full trainval `t1.v2` artifacts | NOT EXECUTED | 正确保留为 pending；没有 cache hash/physical SHA 可接受。 |
| model/full-data/scientific readiness | NOT ESTABLISHED | 本 phase 没有运行模型、100/1000-step、profile 或 metric。 |

## Required return package

1. 修复 GT-database 的 pickle/sidecar physical-hash 输入、双向验证、meta 输出与
   derived-field hostile regression。
2. 修复 pending launcher 的完整 local import/source closure 与实际依赖版本记录。
3. 形成新的 executable commit；重新计算 runtime source-state hash；使用新 SHA
   生成并确认不存在的新 output root；同步更新 `RUN_REQUEST.md` exact command、
   acceptance 和 `HANDOFF.md`。旧 `ed31f23` 请求不得被口头补丁式批准。
4. 对新增测试建立新的 durable focused evidence。是否提交新的 GH200 focused job
   需要 owner 的明确授权；本 review 不授权 rerun。Job 333477 保持历史
   `c1f4fbe` evidence，不得被重标。
5. S00 检查修复交付后重新发起独立 re-review；任何 full-cache execution 仍需之后
   单独的 owner exact approval。

## Allowed interpretations

- Reviewed S01 implementation history和独立 review artifact 已按正确 topology 集成。
- 历史 Job 332651 支持 exact ten-archive manifest、100% declared train/val
  six-camera/key-LiDAR/10-sweep path coverage、十 archive sentinel 和 loader-only
  determinism/timing；其 `t1.v1` caches 仅是历史证据。
- Job 333206 支持 `54a48f9` 上 real-mini directory/ZIP parity、fork/spawn lifecycle、
  depth 与 ZIP-integrity remediations；Job 333477 支持 `c1f4fbe` 上声明的 62 项
  data-foundation focused regressions。
- 当前代码保存 directory backend，并在 reviewed API 中 fail closed 拒绝旧格式、
  depth drift、sidecar drift、raw canonical drift 与 manifest mismatch。
- 两个 findings 都是可局部修复的 provenance/attestation 问题；它们不推翻原始
  full-reference coverage 计数。

## Forbidden interpretations

- 不得称 S07-A 已被独立 PASS/accepted，或称当前 exact cache request 已可提交。
- 不得提交 `ed31f23`/`7ddb06...`/`s07a_cache_t1v2_ed31f23b2ee1` 当前请求；
  `RUN_REQUEST.md` 仍是 pending，O-009 不覆盖 full trainval cache。
- 不得把 332651 的 `t1.v1` cache 作为任何 production input，也不得给该 job
  retroactive in-job source attestation。
- 不得把 Job 333477 的 62 tests 当成 `ed31f23` launcher 执行证据、trainval-scale
  decoded parity、all-payload CRC、model-step 或 scientific evidence。
- 不得从 mini/synthetic/coverage/loader evidence 推断 mAP/NDS、模型质量、FL、
  attack/defense、generalization 或 publication claim。
- full-official-train GT database 不得进入 Protocol-B `D_base`/`D_tail`/client
  训练；未来必须使用 scene/log/raw-sensor ownership-compatible 的 subset artifact。
- 即使将来 cache materialization PASS，也只会建立 data artifact readiness；不会
  自动建立 S06/S07-B model/full-data/scientific readiness，也不会授权 merge、push、
  PR、upload 或其他 follow-on。

## Residual risks（修复后仍需后续 session 处理）

- Job 332651 未内嵌 source attestation；其历史边界永久保留。
- 现有 full-data profile 只读取 2,432 个 sample、batch size 1；未测 full epoch、
  concurrent shared-filesystem contention 或 model-step data wait。
- CRC32 是 per-payload error detection，不是 archive cryptographic identity；后续
  production 必须同时保留 manifest SQLite SHA、cache pickle/sidecar SHA 和 source/
  config provenance。
- `training/tasks.py` 与若干历史 scripts 的 implicit cache-depth/hash 调用尚未迁移；
  S06/S07-B 必须逐入口 fail closed，不能靠目录内“只有一个 cache”作为 scientific
  provenance。
- Protocol-B split/client ownership 尚未冻结；当前 full-train cache/GT tooling 不做
  scene/log/client leakage enforcement。
- full `t1.v2` materialization 尚未发生；新 artifacts 的 canonical/file hashes、
  runtime identity、counts 与 checksum verification 仍需未来执行后独立审查。

## Final verdict

**CHANGES-REQUESTED**。

S07-A 还不能作为 S00 已接受的 reviewed data-foundation dependency，现有 pending
cache request 也不能提交 owner 作执行批准。修复两个 P1、建立新的 immutable
executable/request 并完成独立 re-review 后，才可重新判断 PASS。该 verdict 不授权
compute、commit、merge、push、PR、upload、model/full-data run 或 scientific claim。
