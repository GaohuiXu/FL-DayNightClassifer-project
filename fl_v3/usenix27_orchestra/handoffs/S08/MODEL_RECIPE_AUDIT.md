# S08 前置模型、数值与训练配方审计

日期：2026-07-14
代码锚点：`2a584053e6f6a3860b6f812681dc8d7342ca52ad`
官方对照：MIT BEVFusion `326653dc06e0938edf1aae7d01efcd158ba83de5`
状态：**前置分析已封存；其后 owner 已批准 envelope v1、开始实现、本地验证及验证通过后的一次 immutable implementation commit。一个 smoke 与后续 Q1 的各自 `<=1h` GH200 资源上限已获方向性批准，但精确不可变 tuple 仍须分别绑定；当前没有 Slurm/GH200 作业执行。**

本报告正文保留 2026-07-14 pre-implementation 审计时点。后续实施状态、精确
source/request 和未满足 gate 以同目录的 `HANDOFF.md` 与 `RUN_REQUEST.md` 为准；
Section 17--18 中“尚未启动/compute none”是当时记录，不得覆盖后续显式 owner 决策。

## 1. 结论先行

当前工程不是 MIT BEVFusion 某一个公开配置的直接复现，而是经过 S02-S07
逐模块审查后形成的 **BEVFusion-class 混合适配模型**：它采用了官方配置族中的
Swin-T、LSS、0.075 m SECOND 几何、六任务 CenterHead 与 CenterPoint target/loss
语义，同时在 camera neck、稀疏归一化、fuser、BEV neck、pillar control 和统一
head 组合上做了本项目自己的设计选择。

这不是天然错误；共享 CenterHead 和统一输出接口有利于做 C/L/F matched control。
但必须停止使用“当前模型已经按 BEVFusion reference 完成，因此只剩精度选择”这种
过强表述。现有证据证明模块契约、构造、单批梯度与一个 FP32 update，不证明：

- 当前混合架构等价于官方 fusion detector；
- 当前 training recipe 合理或收敛；
- 百万量级 LiDAR 梯度健康；
- 全 sparse-conv FP16 或 FP32-island 已被接受；
- S07 五个 JSON 是可运行的科学配置。

本审计的最高优先级发现是：当前 SECOND 把官方稀疏特征上的
`BN1d(eps=1e-3, momentum=0.01)` 替换成了对每个 active voxel 独立执行的
`GroupNorm(eps=1e-5)`。在 stem/stage1 的 16 通道上，8 个 group 意味着每组只有
两个数，并连续出现五次归一化。该机制与 Job `389356` 的百万量级 FP32 梯度、
scale-1 FP16 在 stem/stage1 仍非有限高度一致，是当前第一根因假设；但它尚未被
逐层 variance/gradient 边界证据确证，不能提前写成定论。

另一个同等级结论是：strict centralized runtime 是强的 fail-closed 运行/恢复壳，
但不是已冻结 scientific recipe。当前实际是单一参数组、常数 LR、无 gradient
clip、模板 EMA 关闭、camera 2D augmentation 隐式开启而 scene-level 3D/BEV
augmentation 与 GT paste 关闭。`1e-4` 对官方 0.075 LiDAR 有出处，但对当前 C/F
只是模板复用。

因此建议把进入 S08 的顺序改成两个连续门：

1. **数值/架构健康门**：用最小、无通用 hook 框架的诊断确认梯度在哪里开始异常，
   并判断 tiny-group sparse GN 是否为主因；
2. **精度资格门**：在 owner 明确接受的诊断夹具或 provisional recipe 上，比较 FP32、
   full FP16 和 SECOND-FP32 island，验证多个 accepted optimizer windows。

即使 FP32 island 能更新，只要百万级 FP32 梯度仍未解释，也不能直接把“可执行”升级为
“模型/recipe 健康”并进入 S09 production readiness。

## 2. 审计范围与证据等级

本审计完整追踪了当前生产构造、camera/LiDAR/fusion/head/loss、resolved config、
centralized training loop、optimizer/scheduler/EMA/checkpoint、S02-S07 handoff/review、
Job `389356` 原始日志和历史 Job `211502/211722` 边界；并只使用固定提交的 MIT
BEVFusion 官方仓库做外部结构/recipe 对照。

证据等级如下：

| 等级 | 含义 | 本报告中的例子 |
|---|---|---|
| 已证实源码事实 | 当前锚点可直接追踪的构造/运算/配置 | global `precision` 只有 `fp32/fp16`；SECOND FP16 被隐式启用 |
| 已执行 bounded evidence | 精确 Job/测试在声明范围内通过或失败 | Job `389356` 的九个诊断 cell；Job `390576` 的一个 FP32 update |
| 高优先机制假设 | 源码机制与观测吻合，但尚无因果对照 | tiny-group sparse GN 导致梯度放大 |
| 未知 | 需要新证据才能回答 | 最终 recipe、预训练 policy、多步收敛、mAP/NDS |

历史 `fl_v3/collab/**` 只用于解释设计来源和候选经验；它不重新获得当前科学权威。

## 3. 当前五个候选的真实计算图

### 3.1 C-STR8

```text
6-camera images
  -> model-side aspect-preserving 2D image augmentation / normalization
  -> trainable torchvision Swin-T, taps at strides 4/8/16/32
  -> custom all-level sum GN-FPN, 128 ch at stride 8
  -> pure-camera LSS, 118 depth bins, 80 ch
  -> direct 180 x 180 BEV at 0.6 m
  -> 1x1 camera adapter, 80 -> 256
  -> custom shallow GN SECOND-FPN neck
  -> six-task CenterHead
```

### 3.2 L-S075

```text
[batch,x,y,z,intensity,ring,dt]
  -> deterministic per-sample hard voxelization in FP32
  -> mean VFE over x,y,z,intensity,dt in FP32
  -> sparse SECOND: 5 -> 16 -> 32 -> 64 -> 128
  -> three XY stride-2 reductions; z collapse to [B,128,2,180,180]
  -> one final densification and z-to-channel reshape [B,256,180,180]
  -> 1x1 LiDAR adapter, 256 -> 256
  -> same custom BEV neck
  -> same six-task CenterHead
```

几何为 `[-54,-54,-5,54,54,3]`、voxel `0.075 x 0.075 x 0.2 m`、sparse shape
`41 x 1440 x 1440`、train/eval cap `120000/160000`、每 voxel 最多 10 点。输出 cell
为 `0.6 m`，理论 XY receptive field 为约 `137 x 0.075 = 10.275 m`。

### 3.3 L-P020

```text
[batch,x,y,z,intensity,ring,dt]
  -> deterministic pillar grouping on 512 x 512 @ 0.2 m
  -> per-point 8-d feature
  -> Linear 8 -> 64 + GN + ReLU + max over <=32 points
  -> custom 4-stage dense 2D backbone, 64 -> 128/256/256/256
  -> nearest-add FPN -> [B,128,512,512]
  -> 1x1 LiDAR adapter, 128 -> 256
  -> BEV neck with factor 2 -> [B,256,256,256]
  -> same six-task CenterHead
```

它是确定性/能力搜索导向的项目自研 control，不是官方 PointPillars parity。当前没有
fusion+pillar 模板；`F-U` 和 `F-CBGS` 都使用 `second_075`。

### 3.4 F-U 与 F-CBGS

```text
C-STR8 camera BEV [B,80,180,180]
                    +
L-S075 LiDAR BEV [B,256,180,180]
  -> concat [B,336,180,180]
  -> custom two-layer Conv-GN-ReLU fuser -> 256 ch
  -> same custom BEV neck
  -> same six-task CenterHead
```

`F-CBGS` 与 `F-U` 的网络参数完全相同；差别只在 dataset sampling。当前名为
`cbgs` 的实现实际是 seeded sqrt repeat-factor sampling，而非 MIT 官方
`CBGSDataset`。

## 4. 参数量、shape 与来源清单

下表参数量由当前源码结构静态核算；C/L/F 总数与 Job `389356` 的 gradient element
count 完全一致。

| 模块 | 当前参数量 | 输出/作用 | 来源判断 |
|---|---:|---|---|
| image preprocess | 0 | 256x704、标定同步变换、ImageNet normalize | MIT-style policy，经 S03 自洽实现 |
| trainable Swin-T feature stages | 27,517,818 | strides 4/8/16/32，96/192/384/768 ch | torchvision Swin-T；并非官方 mmdet Swin 输出完全等价 |
| custom camera FPN | 333,056 | all-level sum，128 ch @ stride 8 | S03 项目适配 |
| pure-camera LSS | 173,254 | 118 depth bins，80 ch，FP32 splat accumulation | LSS family，S03 framework-independent 实现 |
| camera adapter | 20,736 | 80 -> 256 | 项目统一接口 |
| PointPillars PFN | 640 | 8 -> 64 | 项目确定性简化 |
| P020 dense backbone | 3,732,224 | four-stage 2D conv/FPN，128 ch | 历史能力搜索所得项目模块 |
| P020 LiDAR adapter | 33,024 | 128 -> 256 | 项目统一接口 |
| SECOND encoder | 2,694,352 | sparse 5 -> 128；collapse 256 ch @ 180² | 官方几何/通道映射 + 项目 GN/执行适配 |
| SECOND LiDAR adapter | 65,792 | 256 -> 256 | 项目统一接口 |
| two-layer ConvFuser | 1,364,992 | 336 -> 256 -> 256 | 比官方多一层，并以 GN 替代 BN |
| custom BEV neck | 1,033,728 | 256 -> 256，nearest upsample | 浅层 deterministic SECOND-FPN family 适配 |
| six-task CenterHead | 1,519,686 | shared 64 + 6 task x 6 fields | 官方 task/topology + O-018 GN/no-starvation 适配 |

| 候选 | 总可训练参数 | head grid | 备注 |
|---|---:|---|---|
| C-STR8 | 30,598,278 | 180x180 @ 0.6 m | 与 D1 30,598,278 gradient elements 一致 |
| L-S075 | 5,313,558 | 180x180 @ 0.6 m | 与 D1 5,313,558 一致 |
| L-P020 | 6,319,302 | 256x256 @ 0.4 m | 静态源码核算；当前未做 integrated GH200 optimizer gate |
| F-U | 34,636,886 | 180x180 @ 0.6 m | 与 D1 34,636,886 一致 |
| F-CBGS | 34,636,886 | 180x180 @ 0.6 m | 与 F-U 同模型，仅 sampler 不同 |

## 5. 与固定 MIT BEVFusion reference 的逐模块差异

### 5.1 Camera backbone 与 neck

共同点：Swin-T 的 embed dim、stage depth、heads、window size 等核心骨架对应官方
Swin-T family；输入为 256x704；训练图使用多尺度特征。

关键差异：

1. 官方 camera config 使用 strides 8/16/32 的 `[192,384,768]`；当前还取 stride-4
   `[96]`，然后把四层全部投影到 128 ch、在 stride 8 相加。
2. 官方 neck 输出 256 ch、使用 BN 和自己的 top-down/concat 结构；当前输出 128 ch、
   使用 GN 和 all-level sum。
3. torchvision `net.features` 的 raw stage tap 被直接输出；`net.norm` 与 classifier
   被丢弃。固定 MIT/mmdet Swin 为每个 `out_index` 建立并应用独立 LayerNorm。
   当前 lateral GN 不是该 LayerNorm 的数学等价替代。S03 已证明当前图自洽和梯度覆盖，
   但没有验证预训练 feature distribution/parity。
4. 当前 production 强制 `activation_checkpoint=True`，却没有把该选择写进 resolved
   architecture schema。

因此 camera path 的正确标签是“reviewed independent stride-8 Swin/LSS adaptation”，
而不是“official BEVFusion camera encoder replica”。

### 5.2 LSS 与 camera-to-BEV geometry

当前使用 pure-camera depth distribution，不把 LiDAR projected depth 注入 view transform。
这与官方 camera-only LSS family一致，但不同于官方 camera+LiDAR fusion 的
`DepthLSSTransform` 路径。

固定官方 `swint_v0p075` fusion config 先在 `0.3 m` 的 360² camera BEV 上构造，
再学习下采样到与 LiDAR 对齐的 180²/0.6 m；当前直接把 camera splat 到
180²/0.6 m，没有该中间高分辨率/learned-downsample 路径。

当前 strict/relaxed splat 都以 FP32 累加，这是有明确数值稳定性理由的项目改造。

### 5.3 SECOND

高度匹配的部分：range、voxel、caps、sparse shape、stage channel groups、三次 XY
reduction、stage-3 z padding、128 ch 输出和最终 z collapse。

重大差异是 normalization：官方 SparseEncoder 用稀疏 active rows 上的
`BN1d(eps=1e-3,momentum=0.01)`；当前用每 active voxel 的 GroupNorm，默认
`eps=1e-5`。这是为了避免跨 sample batch statistics、满足早期 batch-invariance/
framework-independent 目标而做的 S04 选择，但当时的 focused synthetic loss 并未覆盖
当前六任务梯度尺度。

### 5.4 P020 control

与官方 PointPillars 相比，当前 max points 为 32（官方 20）、cap 为 120k（官方
30k/60k）、PFN 为单层 `8->64` GN 且删除 cluster mean（官方两层/标准 decoration），
2D backbone/FPN 深度和 head factor 也不同。因此它有历史项目经验依据，但不是 paper
parity cell。

### 5.5 Fusion、BEV decoder 与 head

- 官方 ConvFuser：一层 `Conv3x3 -> BN -> ReLU`；当前为两层
  `Conv3x3 -> GN -> ReLU`。
- 官方 detection decoder：SECOND `[5,5]` blocks + SECONDFPN/deconv，head 输入
  512 ch；当前是更浅的 two-branch nearest-upsample neck，head 输入 256 ch。
- 固定官方 Swin+0.075 fusion 主配置使用 TransFusionHead；当前 C/L/F 全部使用另一个
  官方配置族的六任务 CenterHead。

统一 CenterHead 使三分支输出语义和 matched comparison 更干净，是可辩护的研究设计；
但报告中应称它为“shared CenterHead control”，不能称官方 BEVFusion fusion-head parity。

### 5.6 Head 与 loss

当前 task order、class grouping、字段、两层 branch、heatmap bias `-2.19`、Gaussian
radius/min-radius、code weights 和 `0.25 x L1` 来自官方 CenterHead/CenterPoint 语义。
O-018 把 BN 换成 GN，并移除了会导致多类 task candidate starvation 的第二次 task-wide
top-K；这些是已记录的 intentional deviations。

六个 task loss 在 `MultiTaskCenterPointLoss` 中直接求和。每个 task 的 heatmap focal
按自己的 positive count 归一化；empty task 仍训练 dense negatives，分母 clamp 到 1。
这与 CenterHead 训练意图一致，不构成已证实 semantic bug，但会使随机初始化时某些
empty task 对总 loss/梯度贡献很大。

## 6. 精度路径逐段追踪

### 6.1 当前 global 选项

`s06.v1` 已明确支持全局 `precision = fp32 | fp16`。当前缺少的不是第三种浮点类型，
而是第二个、显式、fail-closed 的 **precision partition**：全局 FP16 下，SECOND/
spconv 是跟随 AMP 还是强制 FP32。

当前 resolver 隐式执行：

```text
lidar_arch == second_075 && precision == fp16
    => sparse_conv_fp16 = true
```

所以用户在 JSON 中选择 `fp16` 时，同时、不显式地选择了 full sparse-conv FP16。

### 6.2 FP32

- 参数/master weights、forward、loss、gradients 均为 FP32；
- camera calibration inverse 临时用 FP64 后回到 calibration dtype；
- LSS splat accumulation FP32；
- voxelization/mean VFE FP32；
- SECOND/spconv/dense FP32；
- 无 GradScaler。

### 6.3 当前 full FP16 AMP

```text
outer model autocast FP16
camera preprocess: explicit/observed FP32 normalization boundary
Swin/FPN/LSS depthnet: AMP-eligible execution
LSS splat: explicit FP32 accumulation -> cast back to feature dtype
voxelization + mean VFE: explicit FP32
voxel features: explicit cast FP16
SECOND/spconv + dense collapse: FP16 autocast region
LiDAR BEV: forced FP16 interface
adapter/fuser/neck/head: outer autocast
head outputs: recursively promoted FP32
target/loss: FP32
scaled backward: crosses the preceding AMP graph
master params and final .grad buffers: FP32
```

把 head output 升成 FP32 能保护 target/loss 方程，不能保护 head/neck/SECOND backward
中的 FP16 activation-gradient 边界。

固定 MIT 实现还对 BaseTransform geometry/BEV pooling 以及 LSS/DepthLSS camera-lift
路径使用 `force_fp32`。当前实现没有等价的整体边界：calibration inverse 临时 FP64、
splat contribution/reduction 显式 FP32，但 depthnet、softmax、geometry einsum 和
metric/grid 运算仍位于全局 autocast 上下文，具体 op dtype 由 PyTorch policy 决定，
最后 camera BEV 又按 context dtype cast 回去。因此“外围 camera AMP”也不是官方
precision boundary 的同义词。

### 6.4 候选 SECOND-FP32 island

对当前 exact `second_075`，建议语义应是：voxelization、mean VFE、整个 SECOND/
spconv、最终 dense collapse 均为 FP32；返回 `[B,256,180,180]` 后，adapter 或 fuser
再进入外围 AMP。当前 `collapsed_channels == out_channels == 256`，`to_bev` 是 Identity，
所以边界可以非常明确。

该语义必须写入 resolved config、checkpoint/provenance，不能继续由 `precision` 和
`lidar_arch` 推断。

### 6.5 GradScaler 与状态

production loop 的 scaler 是持久对象，init scale 512；默认 backoff 0.5，可低于 1。
overflow window 不推进 optimizer、scheduler、EMA、successful exposure；checkpoint
保存 scaler 与完整 runtime state。这部分 S06 契约是当前实现的强项。

## 7. Job 389356：LiDAR 梯度为什么必须单独处理

原始日志 SHA-256 为
`6921efe9e39d25d7dc5fa6dfcab87a748d5db6040a4a49ab5a1fb3d5849edc16`。

| 模式 | FP32 loss | 梯度元素 | stable FP64 L2 | 每元素 RMS | max abs |
|---|---:|---:|---:|---:|---:|
| C-STR8 | 945.87 | 30,598,278 | 58,689.6 | 10.61 | 1,983.7 |
| L-S075 | 1,557.71 | 5,313,558 | 8,326,751.0 | 3,612.3 | 1,910,373.9 |
| F-U | 1,121.04 | 34,636,886 | 5,773,409.1 | 981.0 | 1,217,219.4 |

这不是“LiDAR 参数较多所以 total norm 大”：L-S075 参数只有 camera 约 17%，每元素
RMS 却约高 340 倍；F-U 的 RMS 约高 92 倍。owner 对“不正常”的判断成立。

scale 1 时：

- L 有 4,740 个非有限 gradient elements，注册顺序中的前几个参数位于 stem/stem GN/
  stage1；
- F 有 1,870 个，前几个同样位于 sparse stem/stage1；
- C 全有限并执行 optimizer step。

但 `first_nonfinite_parameters` 是 `named_parameters()` 注册顺序，不是 backward 时间上的
第一条故障指令。它把问题定位到 early sparse path，却没有确定某一个 conv/GN kernel
就是起点。

### 7.1 Loss 组成

同一批共有 15 个 GT。FP32 task loss 中：

- C 的两个 empty tasks 贡献约 `216.69 + 55.84`；
- L 的两个 empty tasks 贡献约 `562.80 + 44.26`；
- F 的两个 empty tasks 贡献约 `737.98 + 131.75`。

这是 dense negative heatmap loss 的结果，不自动证明 loss 错；但它说明 head logits/
初始化会显著改变向 backbone 注入的梯度。六任务 loss 与 SECOND 数值条件必须分开诊断。

### 7.2 跨模式比较的 RNG 混杂

C/L/F 虽重设同一 seed，但在创建 shared neck/head 前构造的模块数不同，RNG 被消耗的
序列不同。因此三种模式的 neck/head 初始权重并不相同。D1 的 C/L/F raw loss/norm
不能被解释为只替换输入 branch 后的 matched attribution。

未来三种 precision regime 必须从同一个 exact mode state_dict 克隆，而不是只依赖
“相同 seed 重新构造”。

### 7.3 4096-point bound 的混杂

D1 对 collated point tensor 做前 4096 点截断；当前数据顺序可能使该前缀的 sweep/dt
组成偏向 keyframe，日志没有记录每 sweep 构成。该 bound 可作为重现夹具，但不能代表
完整 10-sweep occupancy。

## 8. Tiny-group sparse GroupNorm 假设

当前 sparse GN 规则是：

```python
groups = min(8, channels // 2)
GroupNorm(groups, channels, eps=1e-5)
```

输入 shape 是 `[N_active, C]`，没有 H/W 维；所以每一行（每个 active voxel）独立
归一化。

| C | groups | 每组数值 | 该宽度 GN 层数 |
|---:|---:|---:|---:|
| 16 | 8 | 2 | stem + stage1 四层 = 5 |
| 32 | 8 | 4 | down1 + stage2 四层 = 5 |
| 64 | 8 | 8 | down2 + stage3 四层 = 5 |
| 128 | 8 | 16 | down3 + stage4 四层 + conv_out = 6 |

对于 C16 的两值 group `a,b`，variance 为 `(a-b)^2/4`。当前 epsilon floor 对应
`1/sqrt(1e-5)=316.2`；近零 variance 时单坐标 Jacobian 的尺度约可到 158。官方
`eps=1e-3` 的 floor 是 31.6，而且 BN1d variance 来自大量 active rows。

不能把最坏 Jacobian 简单逐层相乘：实际卷积权重、ReLU、residual 和 variance 都会
改变结果。但它精确给出了一条能同时解释以下观测的机制：forward/loss 仍有限、
backward 剧烈放大、FP32 容纳百万梯度、scale-1 FP16 在 stem/stage1 失败。

head 也有 `GN(32,64)`，但 head feature 含 180² 或 256² 空间维，每组统计数不是 2，
因此不具有同样的 tiny-group 条件。它仍应记录，但不是当前第一嫌疑。

严格结论是：

> repeated tiny-group sparse GroupNorm 是与源码及现象最一致的高优先根因假设；
> 尚未通过 per-group variance/gain 或 normalization 对照确证。

## 9. Dynamic scaling 能回答什么、不能回答什么

若只用 FP32 max element 做线性估计，为把 scaled gradient 压在 FP16 最大有限值
65504 内：

- L 需要 scale `< 0.034`；
- F 需要 scale `< 0.054`。

二者都可能需要到 `0.03125`。从 512 开始要连续 backoff 14 次，第 15 个 attempt
才首次使用该 scale。因此固定 512/1 不能回答 full FP16 是否最终能恢复；一个小于约
15 attempts 的 dynamic-scaler 请求也可能在数学上还没走到可回答的位置。

这只是参数梯度估计；内部 half activation-gradient 可能要求更低，也可能在 FP32 island
中完全消失。即使 scale 0.03125 能接受，也不能由此证明百万级原始梯度健康。

## 10. 当前 strict centralized recipe

| 项目 | 当前实际行为 | 是否已科学冻结 |
|---|---|---|
| optimizer | Adam/AdamW；所有 trainable params 一个 group | 否 |
| S07 template | AdamW，LR `1e-4`，WD `.01` | 只是 template |
| AdamW 隐含值 | PyTorch defaults：betas `.9/.999`、eps `1e-8`、amsgrad false、非 fused | 未显式入 schema |
| scheduler | `LambdaLR(lambda=1.0)`，成功 update 后推进但 LR 恒定 | runtime 行为固定；科学合理性未证 |
| grad clip | loop 支持，strict central 未传入，实际 off | 否 |
| EMA | nullable decay；成功 update 后更新；templates 为 null | 实现/恢复正确；policy 未定 |
| precision | global FP32/FP16；FP16 scaler init 512 | partition 未显式化 |
| camera initialization | bool 只能选 random 或 torchvision ImageNet1K V1 | templates 为 null；权重文件 digest 未绑定 |
| LiDAR/fusion warm-start | 不支持 model-only/submodule load | 否 |
| camera 2D augmentation | model-side 默认隐式开启 | 数值有 MIT 来源，但未入 resolved manifest |
| scene 3D/BEV augmentation | strict central 没接线，实际 off | 否 |
| GT paste | schema 无开关，实际 off | 否 |
| sampling | uniform 或名为 cbgs 的 sqrt-RFS | F-CBGS 语义未准确命名 |
| world size | strict central 只接受 1 | 当前 runtime boundary |
| batch/exposure | schema 严格核算 micro/world/accumulation | 机制正确，数值未冻结 |
| checkpoint/resume | model/optimizer/scheduler/scaler/EMA/RNG/config/data 全量严格恢复 | 强工程契约 |

完整 epoch 还有一个已知边界：loader `drop_last=False`，而 loop 拒绝与声明 microbatch
不同的短尾 batch。100/1000 step 若在 epoch 尾前停止不触发，但 full epoch policy
必须在以后明确。

## 11. 与官方训练 recipe 的准确比较

固定官方训练说明/配置为 8 GPU：

| 模式 | MIT 官方 | 当前 strict/template |
|---|---|---|
| camera-only | 4 samples/GPU；AdamW `2e-4/.01`；clip 35；cyclic（Swin override）/对应 config schedule；backbone LR x0.1；20 epochs；NuImages-pretrained Swin | world1/B1 template；`1e-4/.01`；constant；无 clip；单 group；初始化未冻结 |
| LiDAR 0.075 | 4/GPU；AdamW `1e-4/.01`；clip 35；cyclic LR/momentum；20 epochs；object paste | LR/WD 数值相同；其余未接线或不同 |
| Swin+0.075 fusion | 4/GPU；AdamW `2e-4/.01`；clip 35；cosine + 500 linear warmup；6 epochs；NuImages Swin；加载 LiDAR-only checkpoint | `1e-4/.01`；constant；无 clip；无 model-only warm-start；随机或 ImageNet bool |

官方 fusion 不是随机 C+L 同时起跑，而是 NuImages camera + 已训练 LiDAR detector。
当前 strict resume 只允许同 config/data/precision run 的完整恢复，不等价于 warm-start。

可直接继承的只是“候选依据”，不是数值照抄：

- AdamW/WD `.01` 有清晰官方依据；
- `1e-4` 对 L-S075 有直接依据；
- C/F 的 `1e-4` 没有对应官方依据；
- clip35、动态 schedule、3D aug/GT paste 有 reference 依据，但必须结合 batch、初始化、
  总 steps 重新冻结；
- EMA `.9997` 来自历史项目 recipe bundle，不是 MIT 官方字段，也没有当前模型的独立
  因果证据。

## 12. 历史 capability recipe 的正确用法

旧 `p1_bb02d` 使用 AdamW、peak LR `3e-3`、WD `.01`、backbone LR x0.1、clip35、
OneCycle、EMA `.9997`、10 sweeps、BEV augmentation、训练 Swin 和 dense P020
backbone，并产生过较强 capability 结果。

它能说明：训练 backbone、空间 LiDAR capacity、augmentation 与一套更完整 recipe
在旧架构上有经验价值。它不能说明：

- 任一单独组件的因果收益；
- `3e-3` 适合当前 SECOND+six-task head；
- BF16/旧 precision 适合当前 spconv；
- 当前 constant/no-clip recipe 已被否定或某个旧 bundle 应直接恢复。

S06 用 fail-closed production trainer 替换旧 centralized path 时，保留了 runtime
state/checkpoint/accounting，却没有把旧 recipe 的 scheduler、clip、backbone group、
augmentation、GT paste 和 warm-start 字段迁入 strict schema。这是当前 recipe 断层的
历史来源。

## 13. Sampling：F-CBGS 的名称冲突

当前算法是：

```text
r_c = clamp(sqrt(0.5 / f_c), 1, 4)
r_i = max class repeat factor in sample i
seeded stochastic rounding
```

MIT `CBGSDataset` 则为每类建立样本池，以目标 class mass `1/C` 与实际 class mass 的
比率做带替换抽样，然后连接各类样本。两者的 sample probability、epoch length 和
multi-class sample 重复语义不同。

后续应二选一：把当前枚举准确改名为 `sqrt_rfs`；或另加并验证
`official_cbgs`。在此之前不能把 F-CBGS 写成“BEVFusion 官方 CBGS”。

## 14. 现有测试到底覆盖了什么

| 层级 | 已覆盖 | 未覆盖 |
|---|---|---|
| S02 loss/targets | official Gaussian/target 边界、synthetic gradients | current full model optimizer dynamics |
| S03 camera | geometry、all-level participation、synthetic FP16 F/B | pretrained stage-norm parity、完整 data/optimizer recipe、收敛 |
| S04 SECOND | shape/caps/empty/permutation、focused sparse FP16 F/B、eval workaround | six-task loss 下的梯度尺度、persistent scaler/optimizer windows |
| S05 head/decode | six-task topology、mapping、decode/NMS | integrated branch training quality |
| S06 runtime | fail-closed config、accounting、resume/rollback、eval contracts | current six-task real resume、scientific recipe |
| Job 389356 | exact mini batch 上九个 precision/scale diagnostics | dynamic continuation、根因、收敛 |
| Job 390576 | production loop 一次 FP32 C/L/F update；worker0/2 batch equality | multi-window、FP16 policy、schedule/EMA/clip、performance/science |

S04 的 sparse backward 使用极小 synthetic active set 与简单
`output.square().mean()`；通过它不能预测当前 six-task heatmap loss 的梯度幅值。

Job `390576` 也没有端到端运行 `centralized_train.main()`：测试直接构造 model/criterion/
AdamW 和 one-batch loader，所以 constant scheduler、EMA、strict loader、完整 checkpoint
与 augmentation recipe 并未一起获得集成接受。

## 15. 关于 opt-in window observer

本审计不建议实现一个通用的“observer framework”。这个名字容易让范围膨胀为：长期
module hooks、activation graph 保存、复杂生命周期、序列化 schema 和多模式兼容；这正是
过去会把精力从模型问题转移到检测设施上的风险。

S08 真正需要的最小功能可改称 **opt-in window-end diagnostics**：

- 默认关闭时不创建 hook、不保留 activation、不改变输出；
- 在 scaler `unscale_` 后、clip/step/zero_grad 前，直接遍历已有 `.grad`；
- 按稳定的参数名前缀聚合 finite count、FP64 norm、RMS、max、`||g||/||w||`；
- scaler/counter/loss 从 loop 已有对象直接读取；
- head-input 只保留一个明确生命周期的 dense tensor tap；
- 只有第一阶段证据确实指向 GN 时，才临时采集 stem/stage1 五个 GN 输入的
  per-group variance/`rsqrt(var+eps)`，并在 `finally` 中清理。

最稳妥的实现不是在 21 个 spconv 模块上挂通用 backward hook，而是让 detector 在
diagnostic mode 显式返回/保存少数 named boundary tensors，对它们 `retain_grad()`；window
结束立即释放。参数梯度统计完全不需要 hook。

因此困难度取决于边界：最小 window-end diagnostics 是局部、可单测的工程改动；通用
observer/hook 系统则高风险且没有必要。本轮明确建议后者不进入 S08。

## 16. 推荐的最小诊断顺序

1. 对每个 mode，从一个 exact initialized `state_dict` 克隆 FP32/full-AMP/island 三个
   regime；使用同一 batch、同一 point order、同一 augmentation parameters。
2. 先使用现有 `last_voxel_stats/last_sparse_meta`，补齐 point/dt/sweep composition、active
   voxels、cap drops 与 dtype；不加 hook。
3. 在 unscale 后按模块前缀统计参数 gradient：head、neck、adapter/fuser、conv_out、
   stage4 到 stem，conv 与 norm affine 分开。
4. 记录一个 head-input gradient，回答梯度在进入 shared head 前是否已经异常。
5. 只有若 stage1->stem 出现明确跃升，才记录五个 C16 GN 的 group variance 和 gain
   分位数。
6. full FP16 dynamic scaler 至少允许走到 scale `0.03125` 后第一次 attempt；随后还需
   多个 accepted windows，而不是一次成功即停止。
7. 第一轮保持 clip off，以免掩盖 pre-clip 现象；clip35 是否进入最终 recipe另行决策。
8. 真正的 GN/BN/LayerNorm/epsilon 对照属于 architecture experiment，必须由 owner
   单独批准，不能偷偷塞入 precision qualification。

## 17. Owner 决策点

### 2026-07-14 决策记录

O-097 接受本报告推荐的 v1 方向：当前 hybrid 架构在第一轮保持不变；D1-style
夹具只用于 numerical isolation；全局 `fp32/fp16` 与 sparse partition 分开记录；
只实现最小 window-end diagnostics；camera/LSS precision boundary 不变；任何
normalization amendment 必须根据新 boundary evidence 另行返回 owner。owner 同时授权
创建线性交付分支和封存当前 audit baseline，但要求在生产实现前再审阅
一次多 subagent 详细执行计划。计算仍为 `none`。

### S08 实现前必须明确

1. **模型标签**：接受当前模型作为“BEVFusion-class shared-CenterHead hybrid”，还是要求
   在 precision work 前向某一官方配置收敛。建议接受准确标签，不做 wholesale rewrite，
   先诊断 GN。
2. **S08 夹具目标**：建议先精确复现 D1（random init、同 mini sample、4096-point bound、
   AdamW `1e-4/.01`、constant、EMA off、clip off、3D aug/GT paste off），只作 numerical
   isolation，不称科学 recipe。
3. **精度 schema**：保留 global `fp32/fp16`，新增独立 sparse partition 字段；它不是
   第三种 precision。字段名/枚举需 owner 审核后再实施。
4. **GN 健康门**：若 boundary evidence 支持 tiny-group GN 根因，是否允许单独提出 S04
   architecture amendment。precision island 不能自动授权 normalization 改动。
5. **诊断设施边界**：是否接受上述最小 window-end diagnostics，并明确禁止通用 observer/
   长期 hook 框架。
6. **Swin stage-output policy**：为 D1 可归因性，建议 S08 numerical isolation 暂时冻结
   当前 raw torchvision taps，不在同一工作包里更改架构；但 owner 必须明确选择“接受为
   custom interface”或另行批准 per-output LayerNorm amendment。后者会使旧 D1/F1 只保留
   为历史工程证据。
7. **Camera LSS precision boundary**：建议第一轮 S08 保持现状，只改变 sparse partition，
   因为 C 在 scale 1 已恢复；同时在结论中明确它不同于官方 force-FP32 LSS。若要增加
   LSS/geometry FP32 island，应作为单独 owner-approved regime，而不是暗中并入
   SECOND-island。

### S09 前必须明确

- 100-step 所称的 production recipe 是否包含 3D/BEV augmentation、GT paste、clip、EMA、
  optimizer groups 和非 constant scheduler；
- sampling 是 uniform、准确改名后的 sqrt-RFS，还是官方 CBGS；
- short-tail/drop-last policy；
- 这些字段必须进入 resolved config/provenance，否则 S09 只能称“base uniform pipeline
  readiness”，不能称最终 production readiness。

当前 dataset 还会拒绝 camera-only 的 scene-level BEV augmentation。若未来只给 fusion
开启 3D augmentation，C/F 将不再是 matched recipe；需要实现 camera-only-compatible
的 box/calibration transform，或由 owner 明确接受 branch-specific recipe。

### 可推迟到 capability/ablation gate

- C/L/F 各自 LR/schedule/epochs；
- matched-control recipe 与 branch-specific official recipe 的取舍；
- ImageNet、NuImages、LiDAR warm-start 及其 Protocol A/B 合法角色；
- EMA decay、CBGS/augmentation/GT paste 的质量收益；
- 多 seed、mAP/NDS、fusion gain。

## 18. 建议状态

在 owner 审阅本报告并回答 Section 17 之前：

- S08 implementation 保持未启动；
- 不修改 model/head/loss/optimizer/normalization；
- 不准备或提交 GH200 job；
- S09 只保留依赖轮廓，不进入 executable planning。

若 owner 接受推荐顺序，下一步应先把 S08 exact implementation envelope 缩减为：
显式 precision partition、最小 window-end diagnostics、同 state/batch 的 bounded dynamic
scaler 资格验证，以及一个独立 reviewer；training recipe 扩展和 normalization 对照分开
决策、分开记账。

## 19. 可追溯审计索引

当前生产代码的主要落点：

- 构造与 precision 解析：`src/fl_v3/training/tasks.py`、`src/fl_v3/config/resolved.py`、
  `src/fl_v3/utils/runtime.py`；
- 训练状态：`src/fl_v3/training/loop.py`、`runtime_state.py`、`checkpoint.py`；
- 整体图：`src/fl_v3/models/fusion/detector.py`；
- camera：`camera_backbone.py`、`camera_neck.py`、`view_transform.py`、`preprocess.py`；
- LiDAR：`sparse_voxel_encoder.py`、`second_sparse_backbone.py`、`lidar_encoder.py`、
  `lidar_backbone.py`；
- fusion/head/loss：`fusion.py`、`bev_neck.py`、`head.py`、`losses.py`。

Job `389356` 原始日志：

```text
/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/
outputs/s07b_grad_diag_0ca44717e978/diagnostic.log
SHA-256 6921efe9e39d25d7dc5fa6dfcab87a748d5db6040a4a49ab5a1fb3d5849edc16
```

官方对照固定在 MIT BEVFusion commit
[`326653dc`](https://github.com/mit-han-lab/bevfusion/tree/326653dc06e0938edf1aae7d01efcd158ba83de5)：

- [paper](https://arxiv.org/abs/2205.13542) 与
  [official repository README/training commands](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/README.md)；
- [Swin camera config](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml)；
- [0.075 m LiDAR config](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml)；
- [Swin + 0.075 m fusion override](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/camera%2Blidar/swint_v0p075/default.yaml)
  与 [fusion base config](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/configs/nuscenes/det/transfusion/secfpn/camera%2Blidar/default.yaml)；
- [official sparse encoder normalization](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/models/backbones/sparse_encoder.py)
  与 [official CBGS wrapper](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/datasets/dataset_wrappers.py)。

官方资料只用于确定结构和 recipe 来源；本项目的接受状态仍由当前源码、
Orchestra 合同和已审核证据决定，不由上游论文自动继承。
