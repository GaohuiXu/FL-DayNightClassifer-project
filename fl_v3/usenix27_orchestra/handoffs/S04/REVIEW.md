# S04-R 独立审查 — SECOND 稀疏 LiDAR 与 O-025 fp16 eval 分派

## Findings（按严重度）

### P0/P1/P2

未发现 P0、P1 或 P2 级可执行缺陷。实际源码、15 项测试、五次执行记录和
原始工件支持在下述严格边界内接受 S04：O-025 option A 的实现仅在
`spconv==2.3.8`、fp16 CUDA eval、`torch.no_grad()` 下临时切换稀疏卷积叶子
的分派标志，并保持 encoder、GroupNorm、非 spconv 模块、参数、注册 buffer、
state dict、几何、cap 与训练路径不变。

### P3 — 同一模型的并发/重入 eval 尚未被证明安全

`_spconv_training_dispatch_for_fp16_eval()` 直接修改每个
`SparseConvolution.training`，退出时再恢复
（`fl_v3/src/fl_v3/models/fusion/sparse_voxel_encoder.py:50-78`）。该修改在
单线程串行 forward 中由 `finally` 正确保护；Job 341695 也验证了正常路径和
显式异常路径恢复。但该状态是 module 实例上的进程内可变状态，没有锁或重入
计数；15 项清单没有覆盖同一 encoder 实例的并发 forward、递归 forward hook，
或与并发 `.train()`/`.eval()` 交错。

这不阻断当前 S04 串行合成门，也不是 O-025 已批准范围的偏离。它是 S06/S07-B
的生产集成约束：官方 eval/推理入口应保证同一模型实例串行调用，或在引入并发
服务/线程执行前增加实例级互斥/明确拒绝与相应重入、异常恢复测试。不得把当前
证据解释为通用线程安全保证。

## 建议 verdict

**PASS — 仅限 S04 模块级、bounded synthetic engineering gate；可作为 S07-B
候选依赖交给 S00 接受。**

该 verdict 不等于 Orchestra 集成 PASS、生产 detector PASS、完整数据 PASS、
吞吐/profile PASS、指标/收敛 PASS 或科学 PASS。S06/S07-B 仍须完成下文列出的
版本、配置、checkpoint、入口和全栈集成门。

## 审查身份、范围与实际 diff

- Session：`S04-R`。
- 审查起点/worker SHA：
  `483e149b95ec891b675df825d924a96bb225b7dd`。
- worker tree：`ee4b40eb3bb6d6bb6b116fc81ef002b3d5ab40fb`。
- 初始 Wave-A base：
  `372de9398ae435f82b83367a922fd302c0635738`。
- O-025 executable：
  `84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f`，tree
  `913fee67d405ed554b3f7df37c3c137f6f577c2d`。
- Canonical O-025：
  `f413b837f07846a667f91b265016448771e4f99b`；验证边界澄清：
  `04569c6fe26e5f0737777c3bb4bca8c8a1f4e6a6`。两者仅只读核验，未合并或
  cherry-pick。
- `372de93..483e149` binary diff SHA-256：
  `18af82e3199d6c4ef61c1a61c91a8684a8f99f4cb90cb447017707785fb92b47`。
- pre-O-025 delivery `49f26de..8498597` binary diff SHA-256：
  `675e15d2fac4f5e0656cea6c79c675a57fc5ba027ef5903ac6c24abb921afc3a`。
- executable 后 `8498597..483e149` binary diff SHA-256：
  `546cdeb1c1653d8c5ed0d7f04293da63a35fe0f50a815295412e09028c2e1c27`；
  该 diff 仅修改 `HANDOFF.md`、`RUN_REQUEST.md`、`RESULTS.md`，未再修改
  executable、测试、launcher 或依赖。
- 最终交付文档 SHA-256：
  - `HANDOFF.md`：
    `dc2c8dc719dd18f465dcd3056cd76319de9ff63a10819eed341c3117471821ec`；
  - `RUN_REQUEST.md`：
    `a14d774a5405b1de3eed30fbc523831379ef11d12b782db63e0c965f583f91af`；
  - `RESULTS.md`：
    `9016154412e69dd4993f6f50111a9e53e67b3ad8a6e6e9d3c7761d67c0402619`。

从 base 到 worker 的实际文件集合为两个稀疏源码文件、四个 pytest 文件、一个
诊断脚本、三个 launcher 及 S04 的 HANDOFF/RUN_REQUEST/RESULTS。未发现对
`detector.py`、`tasks.py`、trainer、canonical Orchestra、依赖源码、`fl_v2/`
或只读 `fl_v3/collab/` 的修改。

## O-025 option A 源码审计

### 范围与 fail-closed 行为

- `SPCONV_FP16_EVAL_VERSION = "2.3.8"`，通过
  `importlib.metadata.version("spconv")` 做精确字符串匹配；缺包或任何非
  `2.3.8` 版本均抛 `RuntimeError`
  （`sparse_voxel_encoder.py:33-47`）。
- fp16 CUDA eval 在进入 voxelization/空输入返回之前就检查
  `torch.is_grad_enabled()` 和版本，因此空输入不能绕过版本/`no_grad` 门
  （`sparse_voxel_encoder.py:271-281,330-344`）。
- 只有 `backbone.modules()` 中 `isinstance(..., SparseConvolution)` 的 21 个
  稀疏卷积叶子被直接设为 `training=True`；未调用递归 `.train()`。所有叶子
  必须先处于 eval，否则 fail closed；`finally` 逐个恢复先前值
  （`sparse_voxel_encoder.py:50-78`）。
- 临时状态仅包住 `self.backbone(x)`；`dense()`、z collapse 后的 BEV projection
  和 metadata 记录在恢复之后执行
  （`sparse_voxel_encoder.py:364-415`）。
- `active_max_voxels` 在上下文之前由 encoder 自身的 `self.training` 选择；因此
  仍为 eval cap。GroupNorm 与其他非 spconv 模块没有被写入训练态。
- fp32 path、CPU path和真正的训练 path均不进入 option-A 上下文；训练计算语义
  未改变。

### 未发现的禁止项

实际 `49f26de..8498597` diff 中没有：spconv/cumm patch、fp32 sparse eval
fallback、fp16 副本/secondary weights、optimizer/GradScaler/参数更新、迭代训练、
新 architecture/config、cap、几何、通道、densification 或 checkpoint schema
变化。唯一生产源码改动是版本门、临时 leaf dispatch 上下文和只读 debug
metadata。

### 状态、autograd 与异常语义

- 非 `no_grad` 的 fp16 eval 明确失败，而不是偷偷构建 training-dispatch graph。
- 状态测试对排序后的完整 `state_dict()`（参数与注册 buffer）做字节 SHA-256，
  并在 fresh eval、大输入、空输入、真实 train/backward 后再次 eval 的生命周期
  中验证不变；所有 master 参数保持 fp32。
- eval 前后所有 `.grad` 为 `None`；真实 backward 仅用于许可内生命周期覆盖，
  随后 `zero_grad(set_to_none=True)`，没有 optimizer/GradScaler/参数 step。
- 异常测试在上下文中主动抛异常并验证全部 leaf flag 与 encoder eval 状态恢复。
- 私有 spconv autotuner/indice 运行时 cache 不属于 `state_dict()`，当前测试不能
  证明它们“完全无进程内状态变化”；因此本 verdict 将 immutability 限定为参数、
  注册 buffer、state dict 与 checkpoint-visible state，不扩展为无 runtime cache
  副作用。

## 架构、坐标、cap 与 densification 审计

官方参考配置中的 voxel/range/cap/channel 组合与实现一致：
`0.075x0.075x0.2 m`、`[-54,-54,-5,54,54,3]`、`[120000,160000]`、
`SparseEncoder output_channels=128`、四段 channel groups。参考代码声明坐标为
`(batch,z,y,x)`，最后一次 sparse z reduction 后才 `dense()` 并把 z 折入通道。
S04 在内部统一转置为显式 `(z,y,x)` spatial shape，因而把参考的 z 方向
padding/kernel 同步转置，而不是改变几何。

- 输入 `(41,1440,1440)`，三次 sparse XY stride-2 后到 `(5,180,180)`，z-only
  conv 后 `(2,180,180)`；实际 `SECONDSparseBackbone` 内没有 `.dense()`。
- encoder 只在最终 sparse 输出处调用一次 `.dense()`，B=4 边界为
  `[4,128,2,180,180]`，BEV 为 `[4,256,180,180]`，未出现 1440-square dense/
  fusion tensor。
- sparse index 为 `(b,z,y,x)`，dense BEV 为 `[B,C,H=y,W=x]`；输出 stride 8、
  cell `0.6m`、origin `(-54,-54)`，固定中心映射与参考 shape 测试一致。
- canonical per-sample voxelization 在 batch grouping 后逐 sample 执行；train/eval
  cap 为 `120000/160000`，并记录 input/valid/unique/kept/dropped/point-cap drops。
- 既有 over-cap、极端 occupancy、空 sample、batch permutation、sample isolation
  和 point-order fixture 均保留并通过。option A 不改变这些路径。

## 15 项测试清单与“未弱化”核验

Job 341695 的 JUnit 清单恰好包含 15 项：

1. reference shape/stride/channel/RF golden；
2. metric/camera-fusion alignment golden；
3. reduced-resolution-only densification；
4. official stage-channel fail-closed；
5. bf16 sparse precision rejection；
6. sparse shape/stats/backward/reduced occupancy；
7. per-sample train/eval cap、extreme occupancy、point permutation；
8. empty/sample isolation/batch permutation；
9. fp32/fp16 train output、gradient及 train→eval empty/nonempty；
10. unsupported spconv version rejection；
11. empty fp16 eval 不能绕过 version guard；
12. exception 时 dispatch flag 恢复；
13. fresh 6-voxel、256-active-voxel、same-model pre/post real backward、mode/
    state/grad、train-dispatch-under-no-grad parity 与 fp32 control；
14. B=4 fp16 train/forward/backward/memory；
15. B=4 fp16 option-A eval dtype/finite/dispatch/memory。

相对 Job 336718 的 10 项清单，没有删除测试：新增四项 option-A fixture 和一项
B=4 eval fixture；原生命周期 fixture 仅加入 O-025 要求的 `torch.no_grad()`，
并新增 grad/dispatch/version 断言。该修改是将原来违反已锁定推理契约的调用修正
为批准语义，不是弱化 native failure。旧 native path 仍由 Job 336728 的七个隔离
cell 完整保存。

## 五次执行记录与原始工件复核

对五个 output root 均重新执行了只读 `sha256sum -c sha256sums.txt`，所有条目
为 `OK`；另行重算了 manifest 自身及 Slurm stdout/stderr。每个
`runtime_source_sha256s.txt` 中的逐文件 hash 均与其记录的 executable Git tree
重新比对一致。

| Job | scheduler / JUnit 或 matrix | 关键原始证据 SHA-256 |
|---|---|---|
| `335566` | `FAILED 1:0`；10/5 fail/0 error/0 skip；5 个 real-spconv case 均为 residual block 收到 Tensor | identity `4b57ff440d678d83615c50249ecb1b42982eb38897b30eee64590042936c5659`；source `4816f0de0a653b667e20a79d20b11862bb56423428c374f88e3a66fb6d6209df`；pytest `c6d27aa2e14f2535ccb5c0e6ea1fe39e305dceb8af2886889a70835a52bda5ee`；JUnit `a130a7ae347de462f49e802be4ca2d3aefad705d36e780adf16991e4ba591ada`；manifest `a9115f43e3c539867ec6c5d4440c5b07f00315bb7aeefc9a84ba264ac6bb040f`；stdout `a8bd24753f806fb244e06c1090c43361c9773b7362c4b93c87aac8a2af0187c7` |
| `335579` | `FAILED 1:0`；10/2/0/0；composition 已过，两个 fp16 final dtype assertion 失败 | identity `e9e2a513a2ece734c98bc7ad4866368b780f732c47a865bc8b60505f90912dc2`；source `2e5755522cff0aa2899a035f45440fb5ecdb71f2cb5156c96403dd818bba9886`；pytest `4b30beadf77a822c9b8edd4b5a6010c403cb81b4c4226e881b544e8ddfc5bf01`；JUnit `0c97e228bdaac48a423c14532771191d2c3953e195c25eb2c7b209905538f1f8`；manifest `8ff9002e4045360ddb5d50187fce36e2769da8c84bbc7c876bdcde399edf509f`；stdout `c309169861d7fa83cf05bd3f5c6b7a2849390b0d2b7cf2284951d79bc2278576` |
| `336718` | `FAILED 1:0`；10/1/0/0；9 项及 B4/dtype 通过，native same-model fp16 eval tuner 失败 | identity `a0d59d11bc16b801fba625d6ecadec9beba2b46c3438cbdd8d553376b8dd73e3`；source `a9b6fd7f6a5d72cc7691cb6118b001ac4221d6d5cffe4b6799d75ef32fa58c06`；pytest `3353d78a6f73ea38093b2a19a7453dba3f0fde46cee56fe27af097a858d96265`；JUnit `0c7756f1ed801b0268bd4b32fda0224bf55bedd973e3d6df749f57f5a95d7439`；manifest `373dfb5063c060dc1fd4f7b407f0de759f60b2edeeb81894633e2a2e16e35730`；stdout `af6ae66979f98ad12e083e6959fcb54d660437f20f10d36afc6d500aa5e8d303` |
| `336728` | `COMPLETED 0:0` 仅表示 matrix 完整；七个隔离 cell 为 6 个 fp16 error、1 个 fp32 success | identity `0a40265c36c6854f777e7909c2a422f185e3222970dcc71b95be7ae3c6f66119`；repo source `d2a5041c5177279f874bd788320053df679c5b8ad060f95d729e29ae0ebfbf63`；dependency source `e7e162a1f10b4e66c42c1bc07fae19248c42a5e198fbee2c546f3dc0a0d43141`；matrix `3257e16b7bf8ed9b7afcfc252b284ece81595b5c45c83c296e9434e412e346e4`；manifest `038b0a93e4a8e084b2d4a9d06381361e8e4bea30ba89e25fb59e1073f5b102d0`；stdout `ddfe1ec1a2b99509c33cef4188b3fc80dc4ae27d1073f35e22e8357d421e17f8` |
| `341695` | `COMPLETED 0:0`；JUnit 15/0/0/0，`78.714s` | identity `bdca744bf8d8380a6bb67f9ea48603e240278adc267b3ceb8c6d7d722c2e3342`；source `a2608664abd6b69f09b96f19b915cdefe1431aa8b503985f2184b94817e92463`；pytest `63ad1d176074b56020e072d56f93fcd496f7c59694a4ddde2ad954be39190a34`；JUnit `8969a3b40f39f65853d5fdae488a9f7c9e1acd22c6b28b2dffb7159afee466d6`；manifest `efc70b39763053662c82907ff7901488119f9d7b8c5b081a021f726c8958105d`；stdout `f4e0bcee0900a9bfba727ec5d94b67f6a16c601da9cc7ac9fbd592802592661d` |

五个 stderr 均为同一 123-byte module-purge notice，SHA-256：
`ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`。

Jobs 335566、335579、336718 必须继续标记为 **FAILED**，不得由后续 PASS 覆盖。
Job 336728 必须继续标记为 diagnostic completeness，六个 native fp16 error 是
O-025 的因果依据，不得改写成通过。前两次历史执行的 committed executable tree
中尚无 `RUN_REQUEST.md`；原始工件只保留批准 request hash，不能从 Git 重建当时
完整 request bytes。该历史 provenance 限制不影响其负结果用途，也不能让它们
承担正向 acceptance 证据。

## Job 341695 独立调度、身份与 gate 审计

独立 `sacct` 返回：

- `flv3_s04_option_a`，node `n412`；submit/start/end
  `2026-07-12T03:22:00 / 03:22:02 / 03:24:43`；
- state/exit `COMPLETED / 0:0`，elapsed/timelimit `00:02:41 / 00:20:00`，
  restarts `0`；
- `NumNodes=1`、`AllocCPUS=8`、
  `gres/gpu:nvidia_gh200_120gb=1`、generic GPU `1`、`mem=11672M`；
- batch MaxRSS/MaxVMSize `1,141,888K / 18,892,480K`，TotalCPU `00:38.021`。

执行身份与 approved request 完全一致：

- delivery/tree：`2350335166e8f2407ff58f99e9aa5ca98c8acb23` /
  `671611cdd743d0b1d63bde5c861cc76ec55236db`；
- executable/tree：`84985970f0f4b4acb8704ddbbd6ae9b2bf94ca9f` /
  `913fee67d405ed554b3f7df37c3c137f6f577c2d`；
- request SHA-256：
  `b242336e1696a68b6a01a90492ca0e58b7216ef8c4ab9a0eced1f983bf5d2110`；
  已从 preserved snapshot 与 delivery commit 各自独立重算一致；
- snapshot identity SHA-256：
  `35d5547e68d2132d9bec1dc77202154b7faf6d2e1d767d68944f722bb0849d2c`；
- runtime source aggregate：
  `a2608664abd6b69f09b96f19b915cdefe1431aa8b503985f2184b94817e92463`；
  18 个逐文件 hash 与 executable tree 全部一致；
- runtime：CPython `3.11.15`、Torch `2.11.0+cu128`、spconv `2.3.8`、
  cumm `0.7.13`、NumPy `1.26.4`、pytest `9.1.1`，machine `aarch64`；
- preserved snapshot 无任一 file/directory write bit；command、WorkDir、
  `SLURM_SUBMIT_DIR` 均指向 exact snapshot；
- JUnit 精确 15 tests、0 failure/error/skip；最终 `sha256sum -c` 全部通过。

原始 stdout 中可独立重建：

- B=4 train/backward：`[4,256,180,180]` fp16，dense
  `[4,128,2,180,180]`，loss `0.010624694637954235`，peak allocated/reserved
  `962274304 / 1069547520` bytes；
- B=4 option-A eval：相同 shape/dtype，dispatch count `21`，peak
  `371167232 / 411041792` bytes，device total `102005473280` bytes；
- 无 optimizer/parameter update。

warnings 与交付一致：一条 Torch FX API warning；一条
`locale.getdefaultlocale` Python 3.15 deprecation；七次 spconv 非 tuple
多维 indexing warning。它们不影响本次精确 gate，但必须在未来 Torch/Python/
dependency upgrade 中重新验证，不能依赖当前 PASS 推断兼容性。

## 本地只读/静态复核

- 七个相关 Python 源/测试使用内存 `compile()`：PASS；未生成 pyc。
- 三个 S04 launcher `bash -n`：PASS。
- `git diff --check 372de93..483e149`：PASS。
- 精确 pytest function inventory：15。
- sparse backbone + encoder 的 `.dense()` 源码调用总数：1，且只在 reduced
  boundary。
- 五个 artifact manifests 重新 `sha256sum -c`：全部 PASS。
- 五个 runtime source lists 与对应 Git executable tree：全部 PASS。
- 当前 login-node `/usr/bin/python3` 无 `pytest`、无 `torch`；按 Arrhenius
  contract 未把该环境当成 GH200 runtime，也未运行/伪称本地 pytest。
- `APPROVED_COMPUTE: none`：本审查没有 `sbatch`/`srun`、GPU、训练、数据读取、
  重跑、retry、profile 或其他 material compute。

## S06 / S07-B 必须处理的集成含义

1. **版本与 fail-closed 配置。** resolved config/provenance 必须显式记录
   sparse precision、`sparse_conv_fp16`、option-A eval dispatch policy、
   `spconv==2.3.8`，并记录 Torch/cumm/spconv build/source identity。任何版本或
   dependency source 变化都需要重新走 lifecycle/parity/memory gate；不得静默
   fallback 到 fp32、secondary fp16 weights 或 dependency patch。
2. **官方 eval 上下文。** 所有 fp16 sparse eval/official metric entry point 必须
   明确包在 `torch.no_grad()` 或经验证的 inference-mode 等价上下文；否则当前
   实现按设计 fail closed。S06 应增加 config/eval entry-point rejection test。
3. **checkpoint/resume。** option A 不增加权重或 buffer，现有 state dict schema
   未改变；但 checkpoint manifest 必须绑定 precision/dispatch policy 与 exact
   dependency identity。训练 resume 后再 eval、eval 后再 resume train 均需在集成
   stack 上验证 optimizer/scaler/EMA/scheduler state 不漂移。
4. **串行/并发语义。** 当前 leaf flag mutation 只在串行调用下通过。S06/S07-B
   必须保证同一 encoder instance 不并发 forward/切换 mode，或补实例级保护与
   对抗测试后再声明并发安全。
5. **全栈 dtype/geometry。** S07-B 必须验证 detector 的 camera/LiDAR/fusion
   接口真正消费 `[B,256,180,180]`、0.6 m、origin `(-54,-54)`，并在
   camera-only/lidar-only/fusion、B=1/B=4、empty/nonempty、train/eval、checkpoint
   reload 下保持 dtype/cap/shape/provenance 一致。
6. **性能边界。** 371 MB eval 与 962 MB train peaks 只来自每 sample 4096 个
   synthetic points 的 standalone encoder；不是 160k eval voxel 上限、完整
   detector、真实 nuScenes occupancy、吞吐或 full-data profile。S07-B 的生产
   memory/performance gate仍完全未满足。

## 允许与禁止的解释

允许：在 executable `84985970`、已核验的 spconv 2.3.8 GH200 runtime、精确
15 项 synthetic fixtures 范围内，option A 关闭了已诊断的 native fp16 eval
dispatch blocker，同时保持 checkpoint-visible state、fp32 master weights、
GroupNorm/eval cap、fp32 control 与训练路径不变。

禁止：把本 review 称为最终 integration/production/full-data/model/metric/
scientific PASS；声称已证明 worst-case memory、throughput、convergence、mAP/NDS、
fusion gain、最佳 voxel size、FL、attack/defense、generalization、线程安全或未来
dependency version 兼容。Jobs 335566/335579/336718 的 FAILED 与 Job 336728 的
六个 native fp16 errors 必须永久保留。

## 残余风险

- option A 依赖 spconv 2.3.8 的私有 training/custom-fwd 分派行为；精确版本门
  降低但不能消除同版本源码/build 漂移风险。
- 同一模型并发/重入 forward 尚未验证，见 P3。
- state hash 覆盖 checkpoint-visible state，不覆盖 spconv 全局 autotuner/
  runtime caches。
- 只验证 bounded synthetic occupancy；真实数据、上限 occupancy、全 detector、
  official eval、resume 和 full-data 性能仍由 S06/S07-B 门负责。
- 当前 dependency warnings 指向未来 Python/Torch 兼容风险。

## 最终状态

审查稿已在 clean detached worker worktree 上完成；本文件是唯一预期修改。
未创建/切换分支，未 commit、merge、push、上传或提交 compute。等待 S00/owner
审阅本稿；只有在其显式授权后，才可创建 `codex/s04-r-lidar-review` 并提交一个
仅含本 `REVIEW.md` 的 review commit。
