# S06-R 独立审查 — production modes、resolved config、runtime/checkpoint/eval

## Findings（按严重度）

### P0 / P1 / P2

**未发现 P0、P1 或 P2 级可执行缺陷。** 在下述严格边界内，源码、敌对测试、
Job 341997 的完整负证据和 Job 342014 的独立可重建正证据共同支持接受 S06：

- 三种 mode 只构造、传输并 forward 已启用的分支；生产 raw payload decode 在
  S01 dataset 尚无 mode-aware API 时明确失败，而不是先双模态解码再丢弃；
- `s06.v1` 配置是嵌套不可变、顺序/locale 稳定、严格枚举和严格字段集合，绑定
  effective batch、成功 optimizer-update budget、train/val `t1.v2` 与 manifest
  身份；
- accumulation window 固定在原始 microbatch 边界，nonfinite/overflow 不推进
  optimizer step、scheduler、EMA 或成功 exposure，partial/short window 被审计后
  fail closed；
- checkpoint 在 mutation-free preflight 后事务式加载，失败时从 CPU snapshot
  回滚 model/optimizer/scheduler/scaler/EMA 与 Python/NumPy/Torch/CUDA RNG；保存使用
  同目录临时文件和 `os.replace`；
- persistent epoch/sampler、eval autocast/FP32 decode boundary、单遍历、six-camera
  calibration、actual-mode metadata 和 provenance 接口在合成边界内一致；
- 同一 detector 实例的 forward/mode 操作由一个 runtime `RLock` 串行化，deepcopy
  重建独立锁，且 sparse mode 在构造前要求安装包版本精确为 `spconv==2.3.8`。

这不是 production detector 或 S07-B PASS。S06 有意不包含 S02-S05 实现、真实
mode-aware dataset、DDP、真实 `t1.v2`、S04 option-A encoder 与 S05 official decode
的最终接线；这些 seam 仍是 S07-B 的阻断门，不能从本 verdict 推断完成。

### P3 — handoff 对“每个组件位置的 real late-load failure”表述宽于实际测试

事务实现本身把 `model.load_state_dict`、`optimizer.load_state_dict`、scheduler、
GradScaler、EMA 和 RNG restore 全部包在同一个 rollback 区间内
（`fl_v3/src/fl_v3/training/checkpoint.py:388-402`），并按固定顺序从 detached CPU
snapshot 回滚（同文件 `:237-288`）。因此静态审查没有确认 fail-atomic 实现缺陷。

但 Job 342014 中真正越过 preflight、在 `load_state_dict` 内部先修改再抛异常的
fixture 只参数化了 `scheduler`、`scaler`、`ema`
（`fl_v3/tests/test_s06_checkpoint_resume.py:291-316`）。`model_shape` 与
`optimizer` hostile cases 在 mutation-free preflight 被拒绝
（同文件 `:257-288`），没有直接注入 model/AdamW 的 real-load-only late failure。
所以 `HANDOFF.md` 所称“real-load-only late failures at each component position”不得
按字面解释为五个组件均有运行时故障注入证据。

这不阻断当前 bounded synthetic gate：标准 AdamW legal resume、前置拒绝、
三种后置组件 late failure、完整 CPU/CUDA rollback 和 RNG 不变性均已执行通过，
且通用事务代码覆盖 model/optimizer 异常。S07-B 仍应在真实集成模型上加入至少
一个 model load hook/extra-state 的 late failure，并在 production optimizer
状态上保留 legal resume 与 rollback gate，才能扩展到“完整生产 checkpoint
fail-atomic 已实测”的表述。

### P3 — dependency source/build 身份被配置绑定，但 S06 job 只实测 package version

resolved config 要求 lidar/fusion 声明 `spconv=2.3.8`、`cumm=0.7.13` 和两个
40 字符 source SHA（`fl_v3/src/fl_v3/config/resolved.py:274-289`）；detector 在
sparse 构造前调用 `require_spconv_238()`，后者只对安装包版本做精确字符串检查
（`fl_v3/src/fl_v3/utils/runtime.py:81-88`；
`fl_v3/src/fl_v3/models/fusion/detector.py:92-97`）。Job 342014 的 identity 和
非 skip fixture 因而证明了运行环境报告 `spconv==2.3.8`，但没有重新读取并证明
实际 cumm/spconv build 来自 config 声明的两个 Git source SHA。

这不是当前 S06 gate 的 P1/P2：canonical contract 只锁定了 S04 的精确
`spconv==2.3.8` 版本，并且 S04 自身已有 exact-source gate；S06 的职责是让该
依赖身份进入 resolved hash/provenance。但 S07-B/任何科学 launcher 必须重新做
实际 dependency build/source attestation，不能把“config 中写入 accepted SHA”
解释为 S06 已独立验证安装内容；版本、source 或 build 漂移必须重新走 S04
lifecycle gate。

### P3 — fp16/S04/worker-resume/生产内存仍是明确未执行的集成证据

- `decode_eval_set()` 的 autocast、FP32 head conversion 和 single traversal 在
  `detection_eval.py:108-168` 静态成立，但执行 fixture 使用 CPU+fp32
  （`test_s06_loader_eval.py:67-83`）；Job 342014 没有跑实际 fp16 S04+S05 eval。
- mode 构造测试使用 instrumented doubles，same-instance test证明并发 forward
  被串行化且 `copy.deepcopy` 获得独立锁
  （`test_s06_model_modes.py:91-146`），但没有接入 S04 option-A encoder，也没有
  对真实 EMA `AveragedModel` 与并发 `.train()`/`.eval()` 交错作运行时压力测试。
- sampler 证明 `(seed,epoch)` 全排列和 loader object 重用
  （`test_s06_loader_eval.py:23-38`），但未运行多 worker augmentation RNG、真实
  ZIP handle 跨 epoch/resume 生命周期或 DDP global ownership。
- CPU snapshot 避免第二份 live-GPU rollback state，但 checkpoint payload 加全部
  live component snapshot 的 production host-memory 峰值完全未测；这与 handoff
  的限制一致。

这些均已在 worker handoff 中交给 S07-B，故不是本次返回修改的理由；它们是任何
production/full-stack PASS 前不可删除的必做门。

## Verdict

**PASS — 仅限 S06 fail-closed production-runtime contract 与 Job 342014 bounded
synthetic engineering gate；可作为 S07-B 的 reviewed candidate dependency。**

不批准 S07-B/full-stack/model readiness、真实数据、production checkpoint、DDP、
100/1000-step、profile、mAP/NDS、FL/security、merge、push 或任何科学解释。

## 审查身份、拓扑与实际 diff

- Session：`S06-R`。
- 启动 worktree：
  `/home/gaohui/.codex/worktrees/90a6/fl_weather_project`。
- 启动 HEAD / `WORKER_SHA`：
  `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7`。
- 启动 branch：空（符合 `detached@WORKER_SHA`）；启动 status：clean。
- Owner-authorized review branch：`codex/s06-r-production-runtime-review`。
- Worker tree：`8b16c40c8e8e9552789d7d940d0f778754ba06b2`。
- Worker base：`968d81583c87ba76b7dbbb722760f8eb8eb6cd39`。
- Implementation executable / tree：
  `c330c72f4060348768c63fb1b7855ca56baffb95` /
  `7ce589685d15fb42c057154c3329679ada934f4b`。
- `968d815..c330c72` binary diff SHA-256：
  `6f196001c8144806ff5b71c52b87154bdd7ecbe704b21bce2f1e770df3c09963`。
- `968d815..6b7ef29` binary diff SHA-256：
  `0dc00bd798e854e7cb802a16454c2aa7d91b79f2f1bf8d449d17f6f8d665e249`。
- executable 后 `c330c72..6b7ef29` binary diff SHA-256：
  `aad47924eaa6b6120c403fd38ecfd64d59488a33e85204fff8271dbc1bf843a4`；
  该 diff 只修改 S06 的 `HANDOFF.md`、`RUN_REQUEST.md`、`RESULTS.md`，未再修改
  executable、测试或 launcher。
- 最终交付文档 SHA-256 与 kickoff 一致：
  - `HANDOFF.md`：
    `c409a3ac8f23b62f69332271c5b74fefe23c23825b82ab46ad9c2a494dac002f`；
  - `RUN_REQUEST.md`：
    `b696ca6f7735dbf301b7e8b6c548c26f22ffc46444cc1e569acb86b0ff2380f2`；
  - `RESULTS.md`：
    `1ff68c9d8dc9e03ebebfd0dc9864d2cbdb3233fae559baff3be09af97792df1f`。

从 base 到 executable 的实际变更为 24 个路径：19 个 Python source/test、一个
JSON config、一个 shell launcher、三个 S06 handoff 文件；`c330c72` 后仅更新
handoff 三件套。未发现 S01-S05 source、data/cache builder、canonical Orchestra、
`fl_v3/collab/`、`fl_v2/` 或其他越权修改。

## 源码审计

### Fail-closed resolved config 与 data identity

- 根对象及每个 nested object 使用精确字段集合；unknown/missing key、legacy
  mode、错误 architecture/mode 组合、非法 precision/optimizer、错误 effective
  batch、非 `t1.v2`、缺失 train/val/manifest identity 均失败
  （`resolved.py:30-99,206-297`）。
- canonical JSON 使用 UTF-8、sorted keys、固定 separators、`allow_nan=False`；
  normalized graph 递归冻结，`to_run_config()` 递归 thaw 成 plain container，
  关闭了 Job 341997 的 `mappingproxy` 失败（`:102-124,159-203`）。
- physical preflight 对 train/val pickle、sidecar 和 manifest 文件重新 SHA-256
  （`:306-325`）；task bridge 同时要求 canonical `cache_paths(...,n_sweeps)` 与
  resolved path 相等，调用 `load_cache(..., n_sweeps=...,
  expected_cache_hash=...)` 并验证 manifest logical hash
  （`training/tasks.py:572-642`）。
- cache memoization key包含 depth、logical identity 与 resolved config hash，不会让
  新 identity 复用旧 unpickle。
- committed synthetic config 的独立 stdlib 重算 hash 为
  `1f06f07fc16d64e10624e98e0cad120cff63131c838244177f2e0688517ac813`，
  与 handoff 一致；其中路径/哈希只是 sentinel，绝非 production identity。

### C/L/F construction、transfer、forward 与 raw decode

- exact mode 枚举位于 `runtime.py:74-78`；legacy/大小写 alias 不会归一化成功。
- detector 仅对启用分支赋实例；disabled camera/LiDAR/fuser 为 `None`，单模态通过
  adapter 进入共同 neck/head（`detector.py:86-172`）。forward 只要求活动分支键，
  不会执行另一分支（`:197-239`）。
- `project_batch_for_mode()` 在 `_move_to_device()` 前删除 disabled tensor
  （`runtime_state.py:117-142`；`training/loop.py:257-259`；
  `detection_eval.py:129-131`）。
- 当前 S01 dataset 仍双模态解码，因此 production `_make_loader()` 在 dataset
  构造前无条件抛出明确 S07-B seam（`training/tasks.py:682-698`）。这是真实的
  fail-closed raw-I/O 证明，不是“实际 production camera/lidar payload skip 已实现”。
- resolved architecture enums 尚未映射到 S02-S05 concrete module flags；
  `ResolvedConfig.to_run_config()` 明示该 mapper 归 S07-B
  （`resolved.py:162-203`）。S07-B 必须完成并 hostile-test 映射，不能只解除 loader
  fail-closed 后让 `tasks.py:378` 的 legacy `pillar` default 静默生效。

### Fixed accumulation、effective batch 与 executed-update accounting

- preflight 拒绝非完整 window 的 `max_steps`、已落后于 state 的 update limit、已知
  loader remainder、resolved local/global microbatch 不一致
  （`training/loop.py:166-205`）。
- update limit 在 `next(batch_iterator)` 之前检查（`:243-252`），成功 update 只能在
  window boundary 发生。
- loss nonfinite 后立即清 grad，但仍消费、forward 并 loss-evaluate 原窗口剩余
  microbatch；下一窗口不左移（`:269-303`）。
- GradScaler overflow 通过 scale backoff 识别；optimizer/scheduler/EMA/成功 exposure
  只在 successful window 推进（`:304-360`）。FP32 gradient nonfinite 也在 step 前
  拒绝。
- partial/short runtime tail 全量记为 discarded、清 grad、终止 state 后抛错；不能
  继续 epoch 或保存 pending grad（`:228-239,362-374`）。
- `TrainingState.validate()` 强制 attempted/exposure/invalid/discarded、window cause、
  optimizer/successful window 完整守恒（`runtime_state.py:11-71`）。
- Job 342014 覆盖 nonfinite 在 window 第 1/2/3 位、一次 overflow、known/opaque
  remainder、non-boundary `max_steps`、optimizer budget、short microbatch；66 项中
  均通过。EMA 没有单独 hostile counter fixture，但源码只在成功分支调用
  `update_parameters`，静态控制流无歧义。

### Checkpoint preflight、事务 rollback 与 atomic save

- schema 精确绑定 model/optimizer/scheduler/scaler/EMA、完整 TrainingState、四类 RNG、
  full resolved config/hash、mode/precision、train/val/manifest identity 与 checkpoint
  identity（`checkpoint.py:17-23,291-319`）。
- preflight 检查 model key/order/shape/dtype/layout、Adam/AdamW type/config/param-group/
  state topology、component presence/structure、TrainingState reconciliation、四类 RNG
  完整性；合法 post-step AdamW state 不会因 fresh target state 为空而被错误拒绝
  （`:46-217,333-386`）。
- preflight 后才创建 CPU detached snapshots 并开始 strict load；任何 load/RNG restore
  异常进入全组件和 RNG rollback（`:237-288,388-402`）。production caller 明确用
  `map_location="cpu"`（`centralized_train.py:104-110`）。
- save 先写同目录 `s06-ckpt-*.pt`，成功后 `os.replace`；partial write、replace
  failure 都保持旧 target 并清理 temp（`checkpoint.py:320-330`；
  `test_s06_checkpoint_resume.py:92-142`）。
- Job 342014 运行了 legal continuous/resume、8 类 preflight corruption、3 类
  late-component rollback、CPU no-alias 与 live CUDA model/AdamW/GradScaler/EMA rollback；
  CUDA case 未 skip。
- snapshot host-memory 复杂度约为完整 incoming checkpoint 加 live component rollback
  copies；Job 的 36 MiB batch MaxRSS 不是 production detector host-memory evidence。

### Persistent loader、S04 serialization 与 eval/provenance

- `PersistentEpochIterator` 固定 loader identity 并强制 sampler 有 `set_epoch`；
  `EpochPermutationSampler` 只由 `(seed,epoch)` 生成完整 permutation
  （`runtime_state.py:75-114`）。
- central trainer复用一个 loader/stream、按成功 update budget 训练、epoch boundary
  checkpoint；DDP/world-size>1 明确失败
  （`centralized_train.py:63-76,82-153`）。当前 production loader 会更早在 raw-I/O
  seam 失败，故本 branch 不能运行真实训练。
- detector 的 `RLock` 同时保护 `.train()`、forward 和 complete traversal context；
  `__getstate__/__setstate__` 保证 deepcopy/EMA 获得独立锁
  （`detector.py:83-97,174-243`）。
- eval 在 projected transfer 后进入 resolved precision autocast，fp16 head 在 decode
  前递归转 FP32，只遍历 loader 一次；timing 只同步/观察并写独立 sink
  （`detection_eval.py:108-168`）。
- 每个 sample 强制 `(6,4,4)` calibration；submission meta 按 actual mode 写 camera/
  lidar flags。identity bundle 若出现任一字段即要求全部 config/checkpoint/train-cache/
  val-cache/manifest 字段，否则失败（`:48-105,171-209`）。
- `build_s06_provenance`/`verify_s06_provenance` 精确绑定 full resolved config、mode、
  precision、data identity、checkpoint SHA 和 source SHA，并拒绝 partial/drift
  （`eval/provenance.py:42-80`）。
- legacy `NuScenesDetectionTask.evaluate()` 仍存在且不是 official resolved path；
  S07-B 必须使 production official eval 只经过合并后的 S05+S06 单遍历入口。

## Job 341997 / 342014 原始证据 reconciliation

### Job 341997 — 必须保留的失败门

- executable：`6696984a6ebd4ec398d9fbfa172fb118e84e7af8`；request SHA-256：
  `e42fd06051fc8fa7ce1531fb8151d150c2395d2ea89aaf7a6249257f2aeddf08`；
  source aggregate：
  `7be6c0c58b42dbef005ccf0ed52f152c06179701c3205bb607a0007ffa098aae`。
- `sacct/scontrol`：submit/start/end
  `05:29:09 / 05:29:10 / 05:30:57 +02:00`，node `n405`，
  `FAILED 1:0`，elapsed `00:01:47`，1 node、8 CPU、16 GiB、1 GH200，
  requeue/restarts `0/0`；batch MaxRSS `36M`。
- JUnit：62 tests、45 pass、17 failure、0 error、0 skip，host `n405`，
  time `23.338s`。四个根因族与 raw traceback 一致：nested `mappingproxy`、假
  opaque iterator、Torch 2.11 hidden temp filename、one-camera eval fixture。
- raw SHA-256：identity `e7846f77...`，JUnit `168d84d0...`，pytest log/stdout
  `1ff31c03...`，source list `9afd0ce0...`，source hashes `7be6c0c5...`，stderr
  `ae633085...`。
- failing pytest pipeline 因 `set -euo pipefail` 停止，**没有**生成最终 in-job
  `sha256sums.txt`；交付中的单件 checksum 是 post-job 复算，标注正确。
- CUDA case 被收集且未 skip，但在 `torch.save` 前置失败，不能作为 rollback PASS。
  该 job 只能解释为完整负证据，绝不是 45 项 partial acceptance。

### Job 342014 — 单独的 remediation-2 bounded PASS

- approved delivery / request：
  `cae0ff59ce3e215ba950be6a76167d2dd716c940` /
  `9479538201ec398b1617847c5265d0dbeae8ec0db084fc6b867a435ffb5020a9`；
  从 delivery commit 独立重算 request bytes 一致。
- executable/tree：`c330c72f...` / `7ce58968...`；base diff：`6f196001...`；
  launcher：`146f5579...`；25-file source aggregate：`bc19c139...`。
- 25 个逐文件 hash 全部与 executable Git tree 重新比对一致；不存在 source-list
  漏配或 snapshot 后源码漂移。
- `sacct/scontrol`：submit/start/end
  `05:39:37 / 05:39:38 / 05:39:54 +02:00`，node `n405`，
  `COMPLETED 0:0`，elapsed `00:00:16`，1 node、8 CPU、16 GiB、1 GH200，
  requeue/restarts `0/0`；batch MaxRSS `36M`。
- identity：aarch64、Python `3.11.15`、Torch `2.11.0+cu128`、spconv `2.3.8`、
  `CUDA_VISIBLE_DEVICES=0`、`synthetic_only=true`、job `342014`。
- JUnit：66 tests、66 pass、0 failure/error/skip，time `2.930s`；pytest log：
  `66 passed in 2.94s`。live CUDA rollback case为 PASS 而非 skip。
- final manifest SHA-256 `2429764a33ff574b5de2623da137c816500889c8afa20e421464edeab155997b`
  绑定 identity、source list、source hashes、pytest log、JUnit、exitcode；独立
  `sha256sum -c` 六项全 `OK`。
- 该 PASS 修复并重跑了同一 inventory，加上 plain-container、真正 opaque tail、
  三个 atomic save 和 six-camera negative fixtures；没有删除 341997 的失败证据。

### Authorization compliance

- 仅查到了两次已记录 job：341997 与 342014；两者资源均在 exact 15-minute、
  one-node/one-GPU/eight-CPU/16-GiB request 内，无 restart/requeue。
- remediation-2 exact command 只提交一次；approval 已消耗，未见 retry、array、
  DDP、extra cell/seed 或 follow-on。
- 先前 bare `sbatch` 被 Slurm 以 `Batch script is empty!` 拒绝，无 Job ID、root 或
  allocation；作为 control-plane negative 保留，不计成功执行。
- 本 reviewer 只运行只读 Git/hash/JUnit/XML/scheduler 查询和本地静态检查；没有
  `sbatch`、`srun`、数据读取、训练、模型 step、GPU job、上传、merge 或 push。

## Gate-by-gate verdict

| Gate | 独立结论 |
|---|---|
| exact mode enum / alias rejection | **PASS — bounded synthetic** |
| disabled branch construction/transfer/forward | **PASS — instrumented doubles；真实 S02-S05 接线待 S07-B** |
| disabled raw payload decode | **FAIL-CLOSED SEAM — production dataset 构造被拒；S07-B 必须实现并验证实际 skip** |
| nested immutable config / canonical hash | **PASS — static + Job 342014** |
| train/val `t1.v2` / manifest identities | **PASS synthetic contract；真实 artifact ABSENT/NOT RUN** |
| fixed window、constant effective batch、limits/no extra `next()` | **PASS — static + hostile fixtures** |
| nonfinite/overflow accounting与 optimizer/scheduler/EMA sync | **PASS bounded；EMA 分支为静态审计，真实模型待集成** |
| legal AdamW continuous/resume | **PASS CPU synthetic** |
| preflight/transaction rollback / CPU snapshots | **PASS bounded CPU+CUDA；P3 late model/optimizer fixture 限制见上** |
| atomic save temp/cleanup | **PASS — save/replace failure + success fixtures** |
| production checkpoint host memory | **NOT RUN / UNMEASURED** |
| persistent loader/set_epoch/permutation | **PASS synthetic；multi-worker ZIP/augmentation resume NOT RUN** |
| exact spconv package `2.3.8` | **PASS runtime version fixture；actual source/build re-attestation待 S07-B** |
| same-instance serialization/deepcopy | **PASS synthetic lock contract；actual S04/EMA/concurrent mode integration NOT RUN** |
| eval autocast/FP32 decode/single-pass/timing/six-camera/meta | **PASS static + CPU fp32 synthetic；actual fp16 S04/S05 eval NOT RUN** |
| provenance drift rejection | **PASS synthetic；production eval/FL entry integration待 S07-B** |
| full `centralized_train.py` replacement compatibility | **DEFERRED/UNCLAIMED；S07-B 必须审计历史 caller 与唯一 production entry** |
| DDP | **FAIL CLOSED / NOT IMPLEMENTED** |
| 100/1000-step、profile、metric、full data | **FORBIDDEN / NOT RUN** |
| compute authorization与 negative preservation | **PASS** |

## 本地只读/静态复核

1. 完整读取 AGENTS、三份 canonical Orchestra、env/roadmap、S06 三件套、base 到
   executable/worker 的实际 diff、全部 changed implementation/test、launcher、
   config、source inventory、两个 job 的 raw log/JUnit/identity/source/checksum。
2. `git diff --check 968d815..6b7ef29`：PASS。
3. 19 个 changed Python 文件使用内存 `ast.parse`：PASS；未生成 pyc。
4. `bash -n fl_v3/scripts/run_s06_runtime_tests.sh`：PASS。
5. committed synthetic config stdlib load/canonical/hash/thaw：PASS，hash
   `1f06f07f...`。
6. 两个 pre-approval RUN_REQUEST bytes 与 delivery commit 独立重算：分别匹配
   `e42fd060...`、`94795382...`。
7. executable launcher、tree、base diff、25-file source list/aggregate逐项重算：PASS。
8. Job 342014 final manifest 独立 `sha256sum -c`：六项全 PASS。
9. 两个 JUnit 逐 testcase 解析：341997 为 62/17/0/0，342014 为 66/0/0/0；
   CUDA rollback 在后者明确 PASS。
10. `sacct`/`scontrol` 对 state/exit/timestamps/resources/node/MaxRSS/restarts 的只读
    复核：与 RESULTS 一致。
11. login `/usr/bin/python3` 无 Torch/pytest/NumPy，遵守 Arrhenius x86/aarch64
    runtime contract，未伪称本地 pytest；依赖执行证据仅来自 exact Job 342014。

## S07-B 必须保留的 integration blockers

1. 实现并测试 mode-aware S01 dataset/blob-store，使 disabled payload 不 open/read/
   decode，同时仍提供 GT、calibration、pose metadata；camera/lidar/fusion 分别做
   read counters 和 hostile missing-disabled-payload 测试。
2. 将 resolved camera/lidar/fusion/head enum 一一映射到 reviewed S02-S05 API；任何
   unknown/legacy/missing mapper 必须在解除 loader fail-closed 之前失败。
3. 合并 S03 stride-8/0.5m、S04 `[B,256,180,180]`/0.6m/origin/cap/option-A、S05
   multi-task forced-FP32 decode/NMS；不得让 S06 legacy detector decode 覆盖 S05。
4. 在实际 S04 encoder 上保留 complete traversal lock、`torch.no_grad()` fp16 eval、
   exception/mode restoration，并加入真实 EMA deepcopy、concurrent mode/forward 与
   train→resume→eval lifecycle gate。
5. 用一个 persistent production DataLoader + epoch sampler 验证 multi-worker
   augmentation RNG、无重复/遗漏、ZIP PID handle 跨 epoch/resume；DDP 如开放则必须
   独立证明全局 ownership 与 exposure 只乘一次。
6. 真实 train/val `t1.v2` 与 manifest 只能来自单独批准的 cache materialization；
   在每个 train/resume/eval/FL entry 复验 physical/logical/depth/config identity，
   不得 post hoc 替换 sentinel。
7. 重新证明实际 cumm/spconv build/source identity；跑 production-shape checkpoint
   save/load/rollback host-memory、fp16 sparse eval、official devkit round-trip、
   100/1000-step 和 profile，均需各自授权。
8. 审计 full-replacement `centralized_train.py` 与所有历史 caller/launcher；确定唯一
   production入口并移除/拒绝能绕过 resolved config、official eval 或 provenance 的
   legacy 路径。

## 允许与禁止的解释

允许：

- executable `c330c72...` 上的 S06 runtime/config/checkpoint/eval 合成契约经独立
  静态审查，Job 342014 的 66/66 bounded synthetic evidence 与源码/请求/调度身份
  可重建；
- Job 341997 是完整、不可删除的 45/62 失败证据，Job 342014 是其后独立批准且
  source-bound 的 remediation-2 PASS；
- S06 可以作为 S07-B 的 reviewed candidate dependency，由 S07-B 继续完成明确
  integration seams。

禁止：

- 把本 PASS 写成 S07-B、production detector、full-stack、full-data、checkpoint
  memory、throughput、100/1000-step、model quality 或 scientific PASS；
- 声称真实 camera-only/lidar-only 已跳过 raw payload decode；当前事实是 production
  loader fail closed；
- 声称 S02-S05 已接线、S04 option-A 已在本 detector 中运行、S05 official decode
  已与 S06 eval 合并，或 DDP 已支持；
- 把 synthetic sentinel identity 当作真实 `t1.v2`/manifest，或把 package version
  check 当作 actual dependency source/build attestation；
- 从 341997 的 45 个通过项推断 partial gate PASS，或隐去其 17 failures/缺失 final
  manifest；
- 推断 mAP/NDS、fusion gain、FL、attack/defense、generalization、publication，或
  任何 merge/push/upload/compute 权限。
