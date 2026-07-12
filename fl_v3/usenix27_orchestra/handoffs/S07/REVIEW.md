# S07-A-R2 最终独立审查 — data foundation remediation/integration

## Findings（按严重性排序）

### 无新增 P0/P1/P2 finding

本次对 `BASE_SHA=WORKER_SHA=ba1571632557c20adbda3172221694cdbecfeabe`
及可执行实现 `44cefd06bc815e893919d95c754896711dba3402` 的独立复审没有发现新的
阻塞问题。旧审查 `976206405ccf7d2c864d318f5ee27302bdf59059` 的两项 P1 及随后
暴露的 locale 不稳定性均保留为负面历史；它们的最新处置如下。

| 历史问题 | 最终处置 | 独立证据 |
|---|---|---|
| P1-A：GT caller 只绑定 raw canonical hash，派生 `gt_boxes`/`sweep2keylidar` 漂移可逃逸 | **CLOSED** | caller 现在要求 exact version/split/depth 的 canonical hash、pickle SHA-256、sidecar SHA-256，按 `info_cache.cache_paths` 解析唯一物理路径，反序列化前后各散列一次，并把实际路径/字节数/hash 写入 GT DB metadata（`fl_v3/scripts/build_gt_database.py:65-163,223-293,349-364`）。hostile cases 保持 raw canonical inputs、pickle meta、sidecar 一致，只改派生几何，并在 `_open_blob_store`/crop 前失败（`fl_v3/tests/test_build_gt_database.py:193-271`）。Job 335280 JUnit 中两个参数化 case 均实际执行并通过。 |
| P1-B：full-cache source attestation 遗漏实际 eager import closure 及 Torch/Pillow runtime identity | **CLOSED** | full/focused launchers覆盖整个 tracked nuScenes Python package、包 initializer、`data/partition.py`、`utils/runtime.py`、实际 builder/test/launcher、环境与依赖配置（`run_s07a_nuscenes_cache_t1v2.sh:54-80`; `run_s07a_provenance_tests.sh:44-69`）；identity 分别记录 NumPy、nuscenes-devkit、pyquaternion、Torch、Pillow，以及 focused 的 pytest（full: `run_s07a_nuscenes_cache_t1v2.sh:138-181`; focused: `run_s07a_provenance_tests.sh:103-145`）。独立 AST closure 重建得到 full 14 个、focused 17 个实际 eager local files，全部是各自 23/25-file attested set 的子集，无 missing local source。直接非 stdlib imports 为 NumPy、nuscenes-devkit、Torch、Pillow，focused 另含 pytest；pyquaternion 作为 devkit 几何依赖也被记录。 |
| locale：`c8dd920` 的 ambient-locale `sort -u` 使 source aggregate 不可复现 | **CLOSED，负面结果保留** | 独立重建复现：旧 full set 在 C/C.UTF-8 为 `6a4ad312...`、在 en_US.UTF-8/sv_SE.UTF-8 为 `59da9492...`；旧 focused set 分别为 `357da487...` 与 `8de319b6...`。`c8dd920..44cefd0` 的 executable diff 只有两处 `sort -u` 改为 `LC_ALL=C sort -u`（`run_s07a_nuscenes_cache_t1v2.sh:73`; `run_s07a_provenance_tests.sh:61`），未 export 或改变 Python/pytest/cache 的全局 locale。`44cefd0` 下三种 ambient locale 的 effective full list/aggregate 均为 `eebaaf95...`/`1322c872...`，focused 均为 `90310705...`/`2710655b...`。旧 c8dd preflight 在 `sbatch` 前拒绝；对应 output root 不存在，调度记录中当天该 job name 只有后来的 335280。 |

因此旧 P1 的原始严重性和因果分析没有被删除或降级；最终结论变化来自新的实现、
新的 immutable identity 和新的 focused runtime evidence，而不是对旧证据的重新解释。

## 审查身份、preflight 与权限边界

- Session: `S07-A-R2`，最终独立 S07-A remediation/integration reviewer。
- Kickoff `BASE_SHA` / `WORKER_SHA`:
  `ba1571632557c20adbda3172221694cdbecfeabe`。
- 可执行 `INT-A_SHA`:
  `44cefd06bc815e893919d95c754896711dba3402`。
- 旧独立 review:
  `976206405ccf7d2c864d318f5ee27302bdf59059`，其 `REVIEW.md` SHA-256
  `7264cd63bd7d807d6ac4490b63d6686ec5e83f6668dd7e2ddefcbb70ac1ce8d9`，
  verdict `CHANGES-REQUESTED`。
- `APPROVED_COMPUTE: none`。本 review 未运行 `sbatch`/`srun`，未提交 full cache，
  未运行模型、profile 或指标，未 merge/push/upload，未改 canonical Orchestra 文件。

行动前 preflight 为：

```text
git rev-parse --show-toplevel
/home/gaohui/.codex/worktrees/f036/fl_weather_project

git rev-parse HEAD
ba1571632557c20adbda3172221694cdbecfeabe

git branch --show-current
<empty; detached HEAD>

git status --short
<empty; clean>
```

它与 kickoff 完全一致。随后只按明确授权从该 detached SHA 创建
`codex/s07-a-r2-data-foundation-review`；本 review 唯一写入文件是
`fl_v3/usenix27_orchestra/handoffs/S07/REVIEW.md`。

## Git topology、review separation 与 scope 审核

- Integration merge `60f603a0837a55b8bc5d56eedcbba065fcc10673` 的 parent 顺序精确为
  `953bfb57941b5a3660ed650c1a80267cd82245d4`、
  `abe5c58b174dbbe1f7045ce91c8b15168d97b87b`；S01 worker 是当前 worker 的
  ancestor。
- 对 `f262f6b..abe5c58` 的全部 23 个 S01-changed paths 逐一比较 blob，merge tree
  与 exact S01 worker tree 为 `23/23` 相同，0 mismatch；S01 历史
  `011e464 → 1fe6517 → ce2e772 → 54a48f9 → abe5c58` 完整可达。
- 原 review commit `7cf7fcc4...` 的 parent 是旧 baseline `ce2e772...`，它不是
  `ba15716` 的 ancestor；其唯一 diff 是 `handoffs/S01/REVIEW.md`。cherry-pick
  `a4ca386...` 中该文件 blob 与 `7cf7fcc...` 完全相同。review branch 没有作为
  implementation merge 进入（拓扑声明亦见 `S07/HANDOFF.md:184-211`）。
- `44cefd0..ba15716` 只修改 S07 的 `HANDOFF.md`/`RUN_REQUEST.md`/`RESULTS.md`；
  executable tree 保持 `44cefd0`。`953bfb5..ba15716` 未修改
  `ORCHESTRA.md`/`SESSIONS.md`/`KICKOFFS.md`，未进入 S02-S06 的 model/training/eval
  implementation。
- `ba15716` 没有成为 `v3-ad-perception` ancestor；未见 merge/push 到长期分支。

## P1-A：物理 cache identity 与 hostile geometry 审核

### Caller contract

- 三个 digest 均为必填 CLI 参数并接受严格 64-hex 校验；cache pickle/sidecar 只能由
  exact `(cache_dir, version, split, n_sweeps)` 的 `IC.cache_paths` 得到
  (`build_gt_database.py:58-98,223-243`)。
- 第一次 snapshot 在反序列化前检查文件存在、字节数和预冻结物理 SHA；
  `IC.load_cache(..., n_sweeps=..., expected_cache_hash=...)` 再检查 format、sidecar、
  every-record depth 与 raw canonical hash；第二次 snapshot 检查 in-flight change
  (`build_gt_database.py:100-150`; `info_cache.py:353-478`)。
- `_load_info_list` 完成之后才调用 `_open_blob_store`，之后才进入 multisweep transform/
  crop (`build_gt_database.py:277-319`)。因此 physical mismatch 的失败边界在任何
  ZIP blob 打开和对象点裁剪之前。
- GT DB `meta.json` 合并 cache provenance 与 backend/manifest provenance，包含
  absolute pickle/sidecar paths、bytes、physical hashes、canonical hash、format/version/
  split/depth，以及 ZIP logical/file/archive identities (`build_gt_database.py:151-220,
  349-364`)。

### Hostile tests 与几何语义

- `gt_boxes` hostile 将 box center x 改 `+0.25`；`sweep2keylidar` hostile 将 transform
  x translation 改 `+0.25`。两者保留 raw canonical contract、pickle meta、JSON
  sidecar，并先证明 `IC.load_cache` 仍接受 canonical contract，再要求 GT caller 因
  frozen pickle SHA 不匹配而失败；monkeypatch 明确禁止 `_open_blob_store` 和
  `crop_object_points` 被触达 (`test_build_gt_database.py:193-271`)。
- keyframe 与 previous sweeps 由同一 blob store 读取；previous sweep 使用
  `sweep2keylidar` 变换至 keyframe `LIDAR_TOP`，columns 为
  `x,y,z,intensity,ring,dt` (`dataset.py:128-154`)。box 为同一 keyframe frame 的
  `(cx,cy,cz,dx,dy,dz,yaw)`，crop 使用 `R(-yaw)` 与半尺寸边界
  (`gt_database.py:28-44`)。未发现 frame、dimension、yaw、unit 或 sweep 顺序漂移。
- cache raw canonical hash 本来就不覆盖全部 derived tensors
  (`info_cache.py:290-322`)；本修复正确地保留 canonical identity，同时另加物理
  artifact identity，而不是虚称 canonical hash 已改变语义。

### Job 335280 runtime closure

Job 335280 的 JUnit 有 7 tests、0 failures、0 errors、0 skipped；两个 hostile
参数化 case 的 testcase 名均存在。launcher 还要求至少一个 test 且任何 skip 都失败
(`run_s07a_provenance_tests.sh:151-187`)。这足以关闭旧 P1-A；它仍只是 real-mini/
synthetic focused provenance evidence，不是 full-cache 或科学证据。

## P1-B：实际 import closure、source sets 与 runtime identity

独立静态 closure 重建显式包含 Python package initializer 语义：

```text
fl_v3 -> fl_v3.data -> fl_v3.data.nuscenes.__init__
  -> class_map, dataset, info_cache, partition, paths, transforms, zip_backend
dataset -> fl_v3.utils.__init__ -> fl_v3.utils.runtime
nuscenes.partition -> fl_v3.data.partition
```

- Full-cache eager local closure 为 14 files；23-file attested set 完整包含它，并保守地
  加入其余 tracked nuScenes package source、launcher、environment 和 dependency/
  config metadata。immutable `44cefd0` 重建的 file-list hash 为
  `eebaaf9528a56004b63cc2cb37fe6d312b75a52df450f374307e8e559cb1cbb5`，aggregate
  为 `1322c87255bc350323de108e347eea1e54daeb12b59fe1889cb15006f79c3884`。
- Focused eager local closure 为 17 files；25-file set 完整包含它，并包括
  `tests/conftest.py`、test module 与 effective config/dependency inputs。Job 335280
  的实际 list hash 为 `90310705f1bac3bcdfba9128deea6aed60a270e811cc62759f1204612d61d913`，
  aggregate 为 `2710655b166a78e3af39d6537a5098c916463415d27dd9f5503bb79a533c1531`。
- 对 Job 335280 的 `runtime_source_sha256s.txt` 逐项从 immutable `44cefd0` Git blob
  重算：25/25 `OK`，aggregate/list hash 精确匹配。没有依赖当前 documentation HEAD
  的未散列 executable input。
- 实际 direct non-stdlib import roots 与 identity set 一致：NumPy、nuscenes-devkit、
  Torch、Pillow；focused 另有 pytest，devkit 几何依赖 pyquaternion 也被记录。
  Job 335280 实际记录 CPython 3.11.15、NumPy 1.26.4、nuscenes-devkit 1.1.11、
  pyquaternion 0.9.9、Pillow 12.2.0、pytest 9.1.1、Torch 2.11.0+cu128。

## Locale negative 与 `44cefd0` 修复核验

- 旧 `c8dd920` clean detached executor 仍存在且当前干净；旧 focused output root
  `.../s07a_provenance_tests_c8dd920cf3f8` 不存在。
- 独立按 immutable c8dd blobs 重建时，C/C.UTF-8 与 en_US.UTF-8/sv_SE.UTF-8 的
  list 顺序和 aggregate 均按上方 finding 所列发生变化，完整复现 pre-submit guard
  的 `357da487...` vs `8de319b6...` 失败。
- `44cefd0` 只改两行 source-list sort；未改 Python、tests、cache builder、GT caller，
  也未设置全局 `LC_ALL`。三种 ambient locale 下，两个 launcher 的 effective list 与
  aggregate 都精确相同。
- 2026-07-11 的 `sacct --name=flv3_s07a_provenance` 仅返回 Job 335280；旧 c8dd
  请求产生 0 job，旧 output 也不存在。这个负面结果没有被重标为 Job 335280 的
  尝试或 PASS。

## Raw scheduler、artifact 与 hash reconciliation

本 review 直接读取了 scheduler、stdout/stderr、SQLite/JSON/JUnit/identity/source-list
和 checksum 文件，而不是仅依赖 handoff prose。

| Job | 独立 scheduler 结果 | Raw artifact 结论 |
|---|---|---|
| 332651 | `COMPLETED 0:0`, n574, `00:05:29/01:35:00`, 1 GH200, 32 CPU, `Restarts=0`; batch `MaxRSS=11048512K`, `TotalCPU=05:40.946` | `sha256sum -c` 对 manifest、coverage、profile、两个 pickle/sidecar 全部 OK；manifest file SHA `228e2f5b...`，logical `023f72b4...`，2,631,093 occurrences / 2,631,084 unique，唯一重复为十份 matching `LICENSE`；coverage `538695/538695`, zero missing；stdout/stderr `5836dfe4...`/`8db5d05b...`。缓存是 `t1.v1`，永久只作历史 coverage/loader evidence。 |
| 333206 | `COMPLETED 0:0`, n405, `00:01:27/00:20:00`, 1 GH200, 8 CPU, `Restarts=0` | identity `54a48f9` / `260560ef...`；18/18 immutable sources 匹配；JUnit `56/0/0/0`；artifact checksum list 与 logs 全匹配。它支持 reviewed S01 real-mini parity、fork/spawn、depth、local-header 与 exact-sentinel remediations。 |
| 333477 | `COMPLETED 0:0`, n430, `00:01:23/00:20:00`, 1 GH200, 8 CPU, `Restarts=0` | identity `c1f4fbe` / `dddca872...`；24/24 immutable sources 匹配；JUnit `62/0/0/0`；artifact/log hashes 与 `sha256sum -c` 匹配。它不执行后来的 P1/locale commits。 |
| 335280 | `COMPLETED 0:0`, n430, `00:01:16/00:15:00`, 1 GH200, 8 CPU, `Restarts=0`; batch `MaxRSS=540M`, `MaxVMSize=6476352K`, `TotalCPU=00:08.591` | identity exact `44cefd0` / `2710655b...`；25/25 Git-source checks OK；list hash `90310705...`；JUnit `7/0/0/0` 且两个 hostile case 存在；in-job `sha256sum -c` 全部 OK。artifact hashes：checksum `bae54e1d...`、JUnit `c6f28882...`、pytest log `40f69658...`、identity `56d4c10...`；stdout/stderr `9db6bc86...`/`ae633085...`。 |

Job 335280 的 scheduler 中该 job name 只有一次 submission，`Restarts=0`，无 retry/
requeue/resubmit/follow-on。保留的 executor worktree
`/home/gaohui/.codex/worktrees/5810/fl_weather_project` 当前仍是 clean detached
`44cefd06...`；launcher 在 allocation 内也会在任何 test 前检查 exact SHA、empty branch
和 clean status (`run_s07a_provenance_tests.sh:32-42`)。因此“final clean detached
executor”与 durable execution identity 一致。

## Pending Section B full trainval `t1.v2` cache request

### Authorization 与 immutable inputs

- `S07/RUN_REQUEST.md:84-90` 明确为
  `PENDING_OWNER_APPROVAL_DO_NOT_SUBMIT`，并明确该 full trainval cache 超出 O-009。
  本 review 不批准也未提交它。
- exact executable/source/list 为 `44cefd06...` / `1322c872...` / `eebaaf95...`
  (`RUN_REQUEST.md:91-106`)；23-file set 已由 immutable blobs 及三种 locale 独立重建。
- accepted manifest 原文件存在且 SHA-256 为 `228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb`；
  SQLite metadata 为 format `s01.nuscenes-zip.v2`、logical hash
  `023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6`、
  exact `trainval01..10` archive set、2,631,093/2,631,084/9 counts
  (`RUN_REQUEST.md:133-146`)。
- proposed output `.../s07a_cache_t1v2_44cefd06bc81` 当前不存在；旧 ed31 和 c8dd
  full-cache outputs 也不存在。launcher 在 output creation 前验证 clean detached SHA、
  source aggregate、backend、manifest logical/file/archive identities
  (`run_s07a_nuscenes_cache_t1v2.sh:43-130`)。

### Request design

- 请求只构建 official `v1.0-trainval` 的 train/val、`n_sweeps=10`，使用 depth-specific
  `t1.v2` filenames (`RUN_REQUEST.md:118-119,171-178`；launcher
  `run_s07a_nuscenes_cache_t1v2.sh:183-189`)。
- train `28130/944881` 与 val `6019/187528` 的 sample/box counts 同时对 actual
  records 和 cache metadata fail closed；它们与 Job 332651 原始 metadata/counts
  独立吻合 (`run_s07a_nuscenes_cache_t1v2.sh:191-252`;
  `RUN_REQUEST.md:180-198`)。
- `cache_identity.json` 将记录 canonical hash、absolute physical paths、bytes、pickle/
  sidecar SHA；`execution_identity.json` 将记录实际 interpreter/platform 与全部上述
  dependency versions。之后先生成 checksum list，再在 job 内执行独立
  `sha256sum -c` (`run_s07a_nuscenes_cache_t1v2.sh:138-181,191-265`)。
- 资源为 one node/one GH200/eight CPUs/30 min/0.5 GPU-hour、one job、无 array/DDP/
  retry/follow-on (`RUN_REQUEST.md:148-168`)。没有 manifest rebuild、payload coverage、
  model、profile、metric 或自动后续代码。

**Section B request 可在本 S07-A review PASS 后由 S00 作为独立执行请求提交 owner
审阅，但仍不能执行，直到 owner 对该 exact SHA/hash/manifest/command/resource/output
给出新批准。** 即使未来执行 PASS，也只建立 trainval cache artifact readiness，仍不
自动建立 S07-B/model/scientific readiness。

## ZIP/cache/directory/geometry 与 deferred seams

- S01-R 最终 PASS 保留于 `S01/REVIEW.md:352-401,528-579`。`t1.v2` 把 depth 绑定
  filename/meta/every record/canonical hash，并在 load 时验证 sidecar、every-record depth
  和 expected hash (`info_cache.py:353-478`)；dataset 对选中每条 record 再验证
  runtime depth (`dataset.py:201-219`)。
- ZIP v2 保留每个 `(archive,path)` occurrence，冲突/within-archive duplicate/local-header/
  flags/CRC fail closed；Job 332651/333206 的 raw evidence 支持 exact ten-archive routing、
  full reference coverage 和 real-mini lifecycle，边界不扩展为 all-payload CRC 或
  trainval directory/ZIP decoded parity。
- Directory backend 仍是无 manifest 的一等路径；GT caller 拒绝用 ZIP provenance
  relabel directory (`build_gt_database.py:166-220`)，Job 333477/335280 的 focused cases
  覆盖该路径。
- S07-A 没有迁移 `training/tasks.py` 和历史 scientific/diagnostic consumers；这些明确
  defer 给 S06/S07-B (`S07/HANDOFF.md:358-370`)。在迁移完成并绑定 resolved config、
  depth、canonical/physical cache hashes、backend/manifest 前，它们不得成为 production
  scientific entry point。
- 当前 whole-official-train GT DB tooling 不带 Protocol-B scene/log/raw-sensor ownership
  subset manifest。它可用于 full-train CL capability 的后续批准流程，但不得用于
  Protocol-B `W_base`、`D_tail` 或 client adaptation；未来必须按冻结的 ownership split
  构建 subset artifact。

## Gate checklist

| S07-A gate | 最终 verdict | Evidence / boundary |
|---|---|---|
| exact reviewed S01 implementation history | PASS | merge parents、23/23 worker blobs、ancestor chain 精确匹配 |
| review-only separation | PASS | `7cf7fcc` 非 ancestor；`a4ca386` 只带 identical REVIEW blob |
| canonical/S02-S06/model contamination | PASS | canonical/model/training/eval range audit为空 |
| ZIP routing/integrity/worker lifecycle | PASS WITH HISTORICAL LIMITS | Jobs 332651/333206；不外推 all-payload/full-epoch contention |
| directory backend preservation | PASS | code path + focused runtime cases |
| `t1.v2` depth/format/sidecar/canonical fail closed | PASS | reviewed S01 code + Jobs 333206/333477 |
| GT caller canonical + physical pickle/sidecar identity | PASS | code + Job 335280 hostile cases |
| full/focused source closure and dependency identity | PASS | 14/17 closure subsets；23/25 immutable sets；Torch/Pillow captured |
| locale-stable source attestation | PASS | c8dd mismatch reproduced；44ce 三 locale identical；仅 sort locale changed |
| Job 335280 authorization/artifacts | PASS | exact single submission，sacct/log/JUnit/source/checksum all reconciled |
| Section B request completeness | PASS AS A PENDING REQUEST | exact SHA/hash/manifest/count/resource/output/checksum contract；仍未批准 |
| full trainval `t1.v2` artifacts | NOT EXECUTED / SEPARATE GATE | output absent；canonical/physical cache hashes 尚未知 |
| S06/S07-B model/full-data readiness | NOT ESTABLISHED / SEPARATE GATE | deferred consumers、100/1000-step、model profile/eval/metrics 均未执行 |

## Allowed interpretations

- S07-A 在 `ba15716` 的 data-foundation integration 与在 `44cefd0` 的 executable
  remediation 可被 S00 接受为 independently reviewed dependency。
- 旧两项 P1 已由 exact code/runtime evidence 关闭；Job 335280 支持 physical cache
  provenance rejection、complete focused source identity 和 locale-stable attestation。
- Job 332651 继续支持 exact ten-archive manifest、`538695/538695` declared train/val
  six-camera/key-LiDAR/10-sweep reference coverage、十 archive sentinels 与 loader-only
  determinism/timing；其 `t1.v1` cache 只作历史证据。
- Job 333206 支持 S01 的 real-mini directory/ZIP parity、fork/spawn lifecycle、cache depth
  与 ZIP integrity remediations；Job 333477 支持 pre-P1 S07-A focused regression tree。
- S02-S05 可在 S00 形成并冻结一个未来 integration SHA 后按各自完整 kickoff/owner
  冻结选择启动；该未来 SHA可以消费本 S07-A reviewed dependency。此句不是自动 task、
  compute、commit、merge 或 push 授权。

## Forbidden interpretations

- 不得称 full trainval `t1.v2` cache 已生成、已获批或已被 review；Section B 仍需 exact
  owner approval、执行和独立 artifact review。
- 不得使用 Job 332651 的 `t1.v1` pickle/sidecar 作为 production input，也不得给该 job
  retroactive in-job Git/source attestation。
- 不得把 Job 333206/333477/335280 的 mini/synthetic focused evidence当成 trainval-scale
  decoded parity、all-payload CRC、model step、100/1000-step、profile 或 scientific
  evidence。
- 不得称 S07/S07-B full stack、S06 production consumers、C/L/F modes、geometry seams、
  precision/resume/eval/config、model/full-data readiness 已 PASS。
- 不得从本阶段推断 mAP/NDS、模型质量、fusion gain、FL、attack/defense、generalization
  或 publication claim。
- 不得把 whole-train GT DB 用于 Protocol-B base/tail/client 数据所有权边界。
- 本 PASS 不授权 Slurm、full cache、model run、commit（本 review 的单次授权提交除外）、
  merge、push、PR、upload 或 publication。

## Residual risks 与后续独立 gates

- Full trainval `t1.v2` cache 尚不存在；其 train/val canonical hash、pickle SHA、sidecar
  SHA 和 actual runtime identity 必须由未来 approved job 产生并独立复核。
- Job 332651 永久缺少 in-job source attestation；它的历史边界不能补写。
- Full-data profile 仍只是 2,432 个 batch-size-1 loader reads；未测 full epoch、并发
  shared-filesystem contention 或 model-step data wait。
- CRC32 不是 cryptographic archive identity；production provenance 必须继续同时保存
  manifest SQLite SHA、cache pickle/sidecar SHA、canonical hash、source/config identity。
- S06/S07-B 仍须逐入口迁移 cache consumer，并独立审查 resolved config、modality I/O、
  precision、resume、effective steps、geometry、official eval 与 profile。
- Protocol-B split/client ownership 尚未冻结；S07-A cache/GT tooling 本身不执行 scene/log/
  sensor ownership enforcement。

## Final phase verdict

**PASS for S07-A**。

`ba1571632557c20adbda3172221694cdbecfeabe` 可作为 S07-A 的 reviewed delivery，
`44cefd06bc815e893919d95c754896711dba3402` 可作为其 executable `INT-A_SHA`。
S00 可以接受 S07-A 为 reviewed data-foundation dependency，并可在未来由 S00 冻结的
integration SHA 上为 S02-S05 准备/启动各自获批工作。

Full-cache Section B execution 仍是未批准、未执行、需独立 review 的 separate gate；
完整 S07/S07-B model readiness 仍是之后由 S02-S06 reviewed outputs、S06 consumer
migration、model gates/profile/eval 共同决定的另一 separate gate。本 verdict 不授权
任何计算、merge、push、upload 或 scientific claim。

---

# S07-B-R 独立审查 — reviewed CL stack integration

## Findings（按严重性排序）

### P1 — 严格 centralized entry 没有接入 val/official evaluation，当前不能产出候选比较所需的单遍 decode、mAP/NDS 或完整 provenance

`fl_v3/scripts/centralized_train.py:68-115` 解析 config、验证 dependency/data identity、
构建 train loader/model/optimizer 并可 resume；`119-163` 只训练、保存 checkpoint 及三个
identity 文件后退出。该入口没有构建 val loader，没有调用
`decode_eval_set`/`run_detection_eval`，也没有把实际 checkpoint file SHA-256 注入
`build_results_dict`。同时，严格 schema 虽接受 `evaluation.timing`
(`fl_v3/src/fl_v3/config/resolved.py:322-324`)，`to_run_config()` 在
`168-222` 完全丢弃该字段。于是 `detection_eval.py:111-173` 的一次遍历、autocast、
forced-FP32 decode 和 timing-neutrality，以及 `176-221` 的 actual-mode/完整 identity
provenance，只是孤立库能力，不是严格 CL executable path。

这不是可由静态 schema test 代替的缺口。现存 S06 test 只用 synthetic `EvalModel`
直接调用库函数 (`fl_v3/tests/test_s06_loader_eval.py:41-73`)；没有测试从 strict config、
完整 checkpoint、真实 task/val loader 一直走到 official devkit submission。并且历史
`t4_readiness_eval.py:137-139` 仍把 `torch.load()` 返回值直接传给
`model.load_state_dict`，而新的 S06 checkpoint 是含 `model/optimizer/scheduler/scaler/ema/...`
的完整 payload (`fl_v3/src/fl_v3/training/checkpoint.py:291-319`)，因此它不能作为这条
缺失生产路径的替代。

**Required remediation:** 在 strict resolved entry（或一个同样只接受 `s06.v1` 的
strict eval entry）中使用 `load_checkpoint()` 语义加载 exact model/EMA policy，构建
token-complete val loader，只调用一次 `decode_eval_set`，把实际 checkpoint file hash、
resolved/data/dependency identities 和 actual mode 注入 submission，再调用 official
`DetectionEval.evaluate()`；`evaluation.timing` 必须 hash-bound 地决定 timing 收集而不改变
输出。必须有一个从 strict config 到结果/provenance 的 caller-level hostile test，证明无
第二 decode/threshold、无缺 token/重复 token、每样本 `<=500`、且 timing on/off 输出相同。

### P1 — 六任务 CenterHead 已替换真实 head contract，但多个仍可达的历史 caller 没有迁移；“legacy decoder retained for inventoried callers”在当前模型上不可达

`CenterPointHead.forward()` 现在无条件返回六个 task dict 的 list
(`fl_v3/src/fl_v3/models/fusion/head.py:147-197`)；detector 的 intermediate path 只把它放在
`task_outputs` (`detector.py:249-260`)，且 multi-task decode 明确拒绝 `max_objects`
(`detector.py:274-286`)。但相关 caller 仍按旧单头字典运行：

- `fl_v3/src/fl_v3/attacks/fusion_ablation.py:110-125` 向 multi-task decode 传
  `max_objects`，并从 intermediate result 顶层读取 `heatmap/reg/...`；前者必定
  `ValueError`，后者必定 `KeyError`。
- `fl_v3/scripts/arrhenius_mini_matrix.py:345-359` 对 list 调用 `.get()`/`.items()`，第一
  次 head telemetry 即 `AttributeError`。
- `fl_v3/scripts/t4_readiness_eval.py:137-139` 还有上一 finding 所述完整 checkpoint
  incompatibility；T5 通过上述 fusion-ablation helper 消费模型，也未形成新 contract 的
  可运行端到端证据。

这些路径没有被证明为 dead；其中 T4/T5 是后续 clean readiness/attack scientific contract
的直接 consumer，mini matrix 还是已列出的工程门。`detector.py:10-14` 与 S07-B handoff
`613-616,695-708` 所称的旧 single-head decoder “retained for inventoried
non-production callers”并不能解决问题，因为当前 `CenterPointHead` 没有产生 legacy dict
的构造选项。严格 config 确实不能选旧 head，这是正确的 fail-closed 行为；但保留损坏 caller
不能算完成迁移。

**Required remediation:** 逐个列出并迁移所有仍授权保留的 caller 到
`task_outputs`/reviewed no-starvation decode，删除 `max_objects` override 和旧顶层 field
访问；若某路径确实废弃，需以调用/launcher inventory 证明 dead 后显式 fail closed 或移除，
不能靠注释宣告。至少为 strict official eval、mini matrix 的 head telemetry 和 T5 condition
decode 各加一个真实六任务 contract test。

### P1 — dependency identity 不能证明 kickoff 要求的 Torch build/source 或实际执行的 spconv/cumm native/generated code

严格 schema 对 Torch 只有一个 version string (`fl_v3/src/fl_v3/config/resolved.py:51-54`)；
runtime 也只比较 `torch.__version__` (`fl_v3/src/fl_v3/utils/runtime.py:174-180`)。两套二进制
内容不同但都报告 `2.11.0+cu128` 的 Torch 安装会得到完全相同的 attestation，故当前实现
从结构上无法完成 “Torch build/source attestation”。

对 spconv/cumm，`_source_checkout_identity()` 只绑定 top-level import origin 与 clean tracked
Git checkout (`runtime.py:96-136`)，`_runtime_package_sha256()` 只递归散列
`find_spec(import_name).submodule_search_locations` 下的普通文件 (`139-164`)。它没有 import/
枚举实际 `spconv.pytorch`、`cumm` native extension origins，也没有绑定 package root 外的
generated/JIT/cache/native artifact。对 editable、wheel、namespace/multi-root 或 generated
native layout，top-level `__init__.py` tree 相同而实际加载 `.so`/生成 kernel 不同的情形不能
被该 hash 区分；相反，如果 runtime 首次 import 会在 package root 内生成文件，pre-import
hash 又可能在 import 后漂移。当前结果因此不能跨这些 installed layouts 被信任。

S07-B 新 test 名称声称绑定 packages/sources/imports，但在
`fl_v3/tests/test_s07_b_integration.py:98-125` 把 version、source identity 和整个 package-hash
函数全部 mock 掉，只验证预制字符串比较，无法发现上述遗漏。S07-B handoff 自己也正确记录
actual GH200 attestation **NOT RUN** (`739-750`)；该负面边界必须保留。

**Required remediation:** schema/manifest 必须绑定 Torch 的可重算 executable/build identity
（以及可获得的 source/build provenance）；对 spconv/cumm 必须在 import 后记录并散列实际
Python 与 native module origins、distribution-owned/generated executable artifact set，并定义
稳定的 include/exclude 规则。应做 pre/post-import equality、editable/wheel/native-outside-root
hostile fixtures和 GH200 实际路径清单，不得仅 mock helper 返回值。

### P2 — PID fallback 没有恢复 process-local counter/cache 语义，mandatory lifecycle contract 只在 registered hook 恰好执行时成立

registered `_after_fork` 会清空 location cache、总计数和 modality 计数
(`fl_v3/src/fl_v3/data/nuscenes/zip_backend.py:477-497`)；但注释明确作为 hook 未执行时兜底的
`_ensure_process()` (`514-533`) 只丢弃 connection/FD/name，保留父进程继承的
`_locations`、`_read_count/_byte_count` 和 modality counters。用 raw fork、禁用/绕过
`multiprocessing.util` hook 后首次 `read_many()` 即可复现：child 的 debug counters 从 parent
值继续增长，而不是 process-local 从零开始；location cache 也不是文档所称的 child-local
重新建立。

现有 persistent fork/spawn test (`fl_v3/tests/test_nuscenes_zip_dataset.py:257-315`) 只覆盖
正常 multiprocessing hook，并没有强制 fallback；它也不检查 parent counters 没被继承或
camera/LiDAR counters 的 lifecycle。应让两条 reset path 共享同一 reset primitive，并增加
hook-skipped PID-change hostile test。

### P2 — 保留的 compatibility data path 对 disabled modality 不闭合：camera-only 的 GT-paste/BEV augmentation 会访问不存在的 LiDAR payload

dataset 正确地在 `camera_only` 不读取/构造 `lidar_points` (`dataset.py:231-279`)，但随后仍
无条件执行 configured GT-paste 与 BEV augmentation (`280-288`)。`paste_sample()` 第一条
有效路径直接取 `sample["lidar_points"]` (`gt_paste.py:41-49`)；`augment_sample()` 同样在
`augment.py:107-119` 无条件变换 points，且 LiDAR-only 若 `img_flip>0` 又访问不存在的
`images`。旧 flat config branch仍允许 `_aug_from_run`/`_gtpaste_from_run`
(`training/tasks.py:583-615`)，所以 retained compatibility caller 可复现 `KeyError`。

五个 strict S07-B 模板/schema 当前不暴露这些字段，故这不是五模板 fail-before-construction
性质的反例，也不证明 strict candidate 已运行；它是“保留兼容路径但没有完成 mode-aware
迁移”的独立缺陷。修复应按 mode 明确定义 GT-only scene transform、camera appearance
transform 与 LiDAR-only GT-paste 行为，或对不支持组合在 config resolution 时 fail closed，
并加入 directory/ZIP 两 backend 的 hostile test。

### P3 — authored tests/worker handoff 对 mode-depth 与 build attestation 的表述超过实际覆盖

- S07-B handoff `595-596` 称新增 disabled-payload test 覆盖 sweep depths 1/10；实际 test
  只有 backend×C/L 参数化，dataset 固定 `n_sweeps=10`
  (`fl_v3/tests/test_nuscenes_zip_dataset.py:342-373`)。它没有 fusion、depth=1、fork/spawn/
  persistent lifecycle 与 mode counter 的笛卡尔 hostile cases。
- sparse identity test 如 P1 所述完整 mock 掉 hash/source实现；它不是 build-tree soundness
  evidence。
- S07-B suite 没有从 `centralized_train.py` 到 official eval 的测试，也没有遍历上述旧 caller。

这些不把未运行测试变成失败测试，但会导致未来只跑现有 suite 时产生 false confidence。
应修正 handoff/测试矩阵，且 gate 报告必须逐项写实际参数与 skip，而不是只报总数。

### 无 P0 finding

未发现数据泄漏、静默修改 canonical Oracle 文件、未授权 compute/外部动作或把历史失败改写为
PASS 的 P0 证据。上述 P1 已足以阻止静态 PASS、production readiness 和 scientific use。

## 审查身份、startup 与权限边界

- Session: `S07-B-R`。
- Kickoff `BASE_SHA=WORKER_SHA`:
  `df13025bc6582b9b436d1df065de75c03e92782d`。
- Reviewed integration base:
  `c9c84f8b2caebea14adc1d79d6d706695be0f50f`。
- Source branch: `codex/s07-b-integrated-cl-stack`；owner-authorized delivery branch:
  `codex/s07-b-r-integrated-cl-stack-review`。
- `APPROVED_COMPUTE: none`。本 review 未运行 Slurm/GPU/data/model/pytest，未编辑
  implementation/canonical docs，未 merge/push/upload/manage worktree。

行动前 startup 原样为：

```text
git rev-parse --show-toplevel
/home/gaohui/.codex/worktrees/44c9/fl_weather_project
git rev-parse HEAD
df13025bc6582b9b436d1df065de75c03e92782d
git branch --show-current

git status --short

```

HEAD exact、branch empty、status clean 后，只创建了 kickoff 授权的
`codex/s07-b-r-integrated-cl-stack-review`。审查前 `S07/REVIEW.md` 为 22,715 bytes，
SHA-256 `d9bbc63c9b5c52963ad4e8cbdd9af248aac5f371c43bd6e7627a20d87bda9952`；本段仅追加于
其后，旧 S07-A-R2 前缀必须以该 byte count/hash 独立复核。

## 五路 non-FF topology 与 review blob 独立核验

从 integration base 沿 first parent 的五个 merge 顺序与 parent 精确为：

| 顺序 | Worker second parent | Merge SHA | First parent |
|---|---|---|---|
| S02 | `3aebf2dc1d19473f29260df279421047d216d70e` | `062ee1c5596db3e77203d9d5869bc988b5beb0ed` | `c9c84f8b2caebea14adc1d79d6d706695be0f50f` |
| S03 | `50893839c45cd3e2ef1b72b98db6668df7030f2a` | `21d822d7ec7ff993b079f0d572bc9215164946a8` | `062ee1c5596db3e77203d9d5869bc988b5beb0ed` |
| S04 | `483e149b95ec891b675df825d924a96bb225b7dd` | `10fc657bbf3a3067695db4cb5c5b44c913ab0b6a` | `21d822d7ec7ff993b079f0d572bc9215164946a8` |
| S05 | `a9c801fdee378906e54d06314d0c772b6559901a` | `5f186d079a1b39133010096477b2adda8e9eeb66` | `10fc657bbf3a3067695db4cb5c5b44c913ab0b6a` |
| S06 | `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7` | `9fb1a9a9a448c90a60d75850f8146d2d4da06b80` | `5f186d079a1b39133010096477b2adda8e9eeb66` |

五个 worker second parent 均为 `WORKER_SHA` ancestor，顺序吻合 kickoff。reviewer branch
tip 均不是 `WORKER_SHA` ancestor（`merge-base --is-ancestor` exit 1）：

| Review | Reviewer tip | merge-base with WORKER_SHA | Imported blob | SHA-256 |
|---|---|---|---|---|
| S02 | `df142dc9a391b87d05bd7becaba59459e9659f88` | `7ad396ebe535ca468337ed44065d39354707e08b` | `f882a7e223ccc88084d283269ac5ba2516a482f0` | `8bb56cafc22a38dfd7b4ef4d755f1531ab081b0371fe18585d744307f5640474` |
| S03 | `2f62e570c9c24ef1e18a483888c3f28ad56a415e` | `50893839c45cd3e2ef1b72b98db6668df7030f2a` | `09d1beb66cec07e769c3650dd9e09a942bceb674` | `01dea6fd81f14bee8ee1cdf9e4dc66488e7253075459821b2e63947fde7566c1` |
| S04 | `a0763c2e0b322d4ca53a92f9f69c90d9b231bbff` | `483e149b95ec891b675df825d924a96bb225b7dd` | `1caa6d01d83792736ebacbc6eecdf6b42bdadb2e` | `8673672793235ae0226d9109c73cd39577d5f40e846b17425178a7011300ea2a` |
| S05 | `1c440843bb2b6d72f10310ff11fcde0d7d1e885c` | `705216de097ae9eeb1813de6dcdc916e2844fcde` | `d3fc2bec71fbb3206de50b3baeb3ad7db6dc9ef7` | `67b58c8e9d1d1622d1af49a2c052cbadd66580500dbf988fc1184f2d0df6736e` |
| S06 | `ca7bbd7e49e91ac2f214f39f62d5e416dd736383` | `6b7ef29b49c23f206c07ea60c2f15e3ffd9aeef7` | `6df4171c0e85b4a63270af91ca18004c7db3a2e4` | `96d1996562bae4b5e2d1204cb6b51d276ad5c50dd7a75e928137b52b41ae0a59` |

每个 reviewer tip 的 `REVIEW.md` blob 与 `WORKER_SHA` 对应路径 blob 完全相同；import commit
`588e9f42a3bf9aa1341fd57c5ce8a838f0e299e0` 为单亲提交，只新增五个 `REVIEW.md`，没有
合并/cherry-pick reviewer history。其后的 S07-B semantic commits依次为
`f629462`（mode I/O）、`e6ec980`（detector）、`8e78b64`（eval audit）、`2944386`
（runtime/config）、`9e5a3e3`（handoff）、`e3cedfa`（sparse hash）、`6b1a6be`
（handoff closure）、`df13025`（whitespace evidence）。

Reviewed diff `c9c84f8..df13025` 为 94 files、16,537 insertions、1,077 deletions。
`git diff --check` 的三项输出只来自 exact imported S03 review/raw artifacts：S03
`REVIEW.md` EOF blank line，以及 Job 336708 `scontrol.txt`/`stdout.txt` trailing spaces；它们
与 worker handoff 记录一致，未被本 reviewer 改写。

## Adversarial contract audit（除 findings 外）

### Mode-aware directory/ZIP I/O 与 t1.v2

- Dataset 的 primary C/L/F branch 在 raw read/decode 前分流，disabled payload 不进入
  `read_many`；calibration/pose/GT 保留。collate 拒绝 mixed mode、missing enabled payload
  和 unexpected disabled payload。此静态主路径成立。
- `n_sweeps` 仍由 constructor 对每条 `_cache_n_sweeps`、presence/maximum sweep list fail
  closed；production `_load_info` 继续验证 canonical/physical t1.v2 cache 与 manifest
  identities。没有发现把历史 t1.v1 重新合法化的路径。
- fork/spawn hook、spawn pickle reset 与 persistent loader 的已有实现未被静态发现破坏；但
  P2 fallback 与所有 actual runtime cross-product 仍未验证。

### C/L/F construction、geometry、loss/head 与 batch contract

- Strict enum mapping把 C-STR8 固定到 Swin-T trainable backbone、all-level stride-8 FPN、
  0.5 m depth bins与 reference camera geometry；L-S075 使用 0.075 m input/XY stride 8 输出
  180x180，L-P020 保持 0.2 m/512 grid与 dense backbone；F 使用同一 selected BEV config，
  shape mismatch在 concat前失败。未发现第二套 x/y row/col 或 yaw/dimension swap。
- `MultiTaskCenterPointLoss` 按 name map `(0),(1,4),(2,3),(9),(6,7),(5,8)` 分离 GT，复用
  reviewed S02 Gaussian/focal/regression字段并对六任务求和；F-CBGS 与 class/reg weights
  fail closed，不叠加。S05 decode保留 per-class K=500、task-wide NMS、post=83，六任务上限
  498，forced-FP32 在 sigmoid/threshold/top-K/regression/NMS 前完成。
- 以上是 static wiring，不是实际 construction/forward/backward evidence。B=1/4/16、dtype、
  gradients、batch permutation/invariance 和真实 sparse empty/cap path全部 NOT RUN。

### Evaluation reconciliation

- `detection_eval.py` 本身拒绝 duplicate decoded/eval tokens，构造完整 token key set，使用
  deterministic content order、global conversion、actual mode，默认不做第二 threshold；timing
  包围 forward+decode且 timing dict 不影响 decode内容。official per-sample cap由 six-task
  `6*83=498` 与 conversion guard双重约束。
- 仅发现一次 `model.decode()` 的库内 traversal，没有第二 decoder overwrite；但 P1 表明
  strict caller没有消费该实现，历史 T4/T5 caller又与新 checkpoint/head contract不兼容。

### Runtime/checkpoint/exposure

- Dependency check在 strict entry 中先于 physical data与 model；DDP 对实际/声明 world-size
  drift及 `world_size != 1` 均 fail closed。
- `TrainingState` 对 attempted/loss-evaluated/success/invalid/discarded sample/window reconciliation
  fail closed；scheduler/EMA 只在 successful update后推进，GradScaler overflow记为 invalid
  window；persistent sampler由 epoch寻址，checkpoint只在 accumulation boundary写入。
- Checkpoint preflight验证 full field/config/data/model/optimizer/scheduler/scaler/EMA/RNG
  structure；late load failure有 snapshot/rollback实现。新增 real model/optimizer mutation test是
  合理 hostile case，但本 reviewer 环境没有 Torch，故未执行；真实 live model/optimizer、CUDA
  rollback与 host-memory cost仍不得标 PASS。

### 五个 candidate templates

五个 `s07_b_*.json` 均含 strict root 不允许的 `template_only`，所以在 data/model前失败；
C/F 的 camera initialization 为 `null`，SECOND build hash使用非 64-hex sentinel，cache/
manifest identity也为不可通过 placeholder。只删除一个 marker不会把 placeholder误认为 actual
identity。它们是有意 non-runnable architecture templates，不是 resolved run configs；没有
运行、批准或生成任何真实 identity。

## Tests/checks actually run 与明确 NOT RUN

本 reviewer 只运行无 Torch/数据/GPU 的 read-only checks：

1. startup ref/branch/status/top-level：PASS；
2. first-parent/parent/ancestor topology、review branch non-ancestry、五个 blob 与 SHA-256：PASS；
3. `git diff --shortstat/name-only` 与 `git diff --check`：PASS with the three preserved S03
   whitespace warnings described above；
4. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`：PASS；
5. stdlib AST parse 11 个 S07-B核心 Python 文件、stdlib JSON parse五个模板：PASS；
6. runtime availability probe：`ModuleNotFoundError: No module named 'torch'`。因此没有运行
   `run_s07_b_static_checks.sh` 的 `py_compile`/strict-loader段，也没有运行 pytest；probe后
   worktree仍 clean。

以下全部 **NOT RUN / NO IMPLIED PASS**，与 worker negative list合并保留：

- integrated tree 上所有 S02-S06 focused tests 与 S07-B pytest suite；
- actual directory/ZIP disabled-payload/decode parity、depth 1/10、counter、fork/spawn/
  persistent-worker lifecycle；
- GH200 上实际 Torch/spconv/cumm import、native/build/source pre/post-import attestation；
- S04 actual fp16 train/eval/no-grad/concurrency/EMA/deepcopy lifecycle；
- C/L/F construction，B=1/4/16 forward/backward，grid/dtype/gradient/batch invariance；
- real live model/optimizer late-load injection、CUDA rollback与 host-memory gate；
- strict caller到 official devkit round trip、GT-as-pred、single traversal、provenance与
  worst-case CPU float64 rotate-NMS profile；
- mini model steps、full t1.v2 cache materialization、100/1000 steps、production/full-data
  profile、mAP/NDS、DDP、matrix、seed/rerun/automatic retry。

历史 negative/positive evidence不变：S06 Job 341997 `45/62` failure 与 bare-sbatch no-op
保留；Job 342014 `66/66` 仅 bounded synthetic；S05 Job 336731 `43/44` failure 与 336738
`44/44` bounded pass并存；S04 的 335566/335579/336718 failures、336728 diagnostic six-error
negative与 O-025 Job 341695 `15/15` option-A pass均保留；S03 335630 failure 与 336708
10-test pass、S01 332648/332651/333206 的各自边界均未扩张。mini/synthetic evidence不是
production/scientific evidence。

## O-009 后续 bounded engineering request 建议（当前不批准、不得提交）

由于存在以上静态 P1/P2，不能为 `df13025` 准备一个用运行来“验收”缺陷的请求。先在 scoped
S07-B remediation 中修复 caller/eval/attestation/lifecycle并完成独立 code review；随后 S00
才可把下列内容写成 **一个 exact immutable** `RUN_REQUEST.md` 并请求 owner 审批：

- HEAD/source diff：修复后的 exact SHA，clean worktree，列出从 `df13025` 的 exact diff；
- command：一个 committed、`bash -n` 通过、无 retry/array/follow-on 的 S07-B gate launcher；
- resources：one node、one GH200、one concurrent job、8 CPUs、`00:60:00` hard limit、最多
  1.0 GPU-hour，output root唯一且预先声明；
- data：仅现有 real-mini directory和由测试临时构造的 stored-ZIP fixture；不得扫描/build
  full trainval manifest/cache，不得 materialize full data；
- gates：integrated S02-S06 focused suites + S07-B suite；actual C/L/F fp16 lifecycle；
  production constructors的 B=1/4/16 分阶段 forward/backward（每阶段显式 VRAM/time stop，OOM
  记失败不自动降 batch）；disabled-payload directory/ZIP depth1/10 fork/spawn/persistent cases；
  Torch/spconv/cumm实际 module/native origin及 pre/post-import hash；CPU+CUDA rollback injection；
  mini official devkit round trip与 declared worst-case rotate-NMS profile；
- stop：任一 identity/config/source mismatch、unexpected skip、nonfinite、OOM、timeout、第二
  traversal、payload counter或rollback mismatch立即非零退出；不自动修改 config、降 batch、
  rerun或提交后续 job。

该建议不是 approval。full t1.v2 cache、100/1000 steps、production/full-data profile、metrics、
DDP、scientific matrix/seeds仍超出 O-009，必须另行 exact owner approval。

## Gate verdict、allowed/forbidden interpretation 与 residual risk

| Layer | Verdict |
|---|---|
| exact worker topology/order + imported review bytes | **PASS** |
| S02-S06 reviewed component history preservation | **PASS WITH EACH REVIEW'S ORIGINAL BOUNDARY** |
| S07-B local/static syntax/config inventory | **PASS, STATIC ONLY** |
| S07-B implementation completeness | **CHANGES-REQUESTED (P1/P2 above)** |
| integrated runtime/GH200 engineering evidence | **NOT RUN / NOT ESTABLISHED** |
| production/full-data readiness | **NOT ESTABLISHED** |
| scientific capability/metric/FL/attack-defense evidence | **ABSENT / FORBIDDEN** |

允许解释：五路 worker merge topology与五个最终 review blob provenance精确；核心 C/L/F、
multi-task loss/decode、S06 runtime/checkpoint代码已经以可审查形式集成；五个 candidate只是
fail-closed templates；历史 reviewed component evidence可继续在各自边界内引用。

禁止解释：不得称 strict centralized CL stack可完成 official evaluation，不得称 T4/T5/mini
caller已迁移，不得称 Torch/spconv/cumm executable identity已证明，不得把 authored/mocked
tests当实际 GH200 evidence；不得声称 production ready、full trainval ready、100/1000-step、
mAP/NDS、fusion gain、FL、attack/defense、generalization或publication claim。不得提交
O-009、full-cache/model job、merge/push/upload。

Residual risks包括：full trainval t1.v2仍不存在；real native package layout/first-import hash
行为未知；S04 fp16 option-A尚未在integrated detector lifecycle中执行；真实 batch/gradient/
memory/rotate-NMS性能未知；strict official evaluator与完整 checkpoint/EMA policy未冻结；
Protocol-B split/client ownership仍未进入本 CL capability候选。

## Final verdict

**CHANGES-REQUESTED for S07-B at
`df13025bc6582b9b436d1df065de75c03e92782d`.**

Topology/review provenance可以接受，但 P1 的 strict official-eval缺口、六任务 caller迁移断点
和不可充分重算的 executable dependency identity阻止 static PASS；P2 lifecycle/mode
compatibility也必须修复。当前不应提交工程计算来绕过这些代码问题。修复后的 exact SHA需
重新独立 review，之后才可按上节冻结并请求一次 bounded O-009 gate；所有 material/full-data/
scientific execution继续需要独立 owner approval。

---

# S07-B-R2 独立复审 — O-036/O-037 remediation

## Findings（按严重性排序）

### P1 — T5 load-bearing shard caller 仍在权威 checkpoint/config 前构造数据；现存 config 会直接失败，测试没有覆盖真实顺序

fl_v3/scripts/t5_attack_eval.py:223-237 的 task_shard() 先调用
_seed(cfg)、_val_info(cfg) 和 _val_dataset(cfg, ...)，到第 236 行才调用
_load_model()。S06 checkpoint 内嵌的 exact ResolvedConfig 只在
_load_model():110-124 才被解析并回填；caller-drift 检查 :115-119 还明确跳过
caller 中缺失的 strict 字段。这会产生三个实际问题：

- 当前 fl_v3/configs/t5_attack.json 没有 det-lidar-sweeps、
  nuscenes-zip-manifest、model-mode、s06-production-runtime 或 precision。
  因而 _val_info():189-193 先走 compatibility cache path，
  _val_dataset():196-204 随即读取 cfg["det-lidar-sweeps"] 并抛 KeyError，根本到不了
  新的完整 checkpoint loader。
- 即使外部 caller 临时补齐这些字段，cache/dataset 已在 checkpoint 绑定的 t1.v2
  cache/manifest/depth/mode identities 验证前构造；之后 cfg.update(strict) 不能追溯证明
  已选取的 val_info/dataset 属于 exact resolved identity。
- T5 各任务在 _load_model() 前执行 _seed(cfg)（例如 :223、:426、:469）。
  t5_attack.json 缺 precision，故先按默认 fp16 设置 backend；若 checkpoint 绑定 fp32，
  缺失字段不会触发 drift，进程级 numeric policy 已经错误。

fl_v3/tests/test_s07_b_integration.py:527-569 只直接调用并大幅 mock _load_model()；
它没有执行 task_shard，也没有以现存 t5_attack.json 走 dataset，因此对“完整 T5 caller
已迁移”的结论产生假阳性。task_shard 是五条件 ablation 主 fan-out，不是 reporting
旁路，故阻断 O-037 closure。

Required remediation：先解析并验证 complete S06 checkpoint 的 exact ResolvedConfig、
physical cache/manifest 和 dependency identity，再以该 config 设置 seed/precision/runtime；
之后才能加载 val info 和构造 mode-aware dataset。允许的 batch/worker/eval-limit override
应在同一 preflight 明确处理；缺失或漂移 scientific 字段必须 fail closed。增加从现存兼容
T5 config + synthetic complete checkpoint 进入 task_shard 的 caller-level hostile，证明
任何 data constructor 前已完成 strict preflight，并覆盖 fp32 checkpoint 对 caller 缺失
precision 的回填。修复后必须重新独立 review；不得用 runtime job 绕过此静态缺陷。

### P0

未发现 P0：没有 split/leakage 改动、canonical/collab 篡改、未授权 compute、
RUN_REQUEST/RESULTS mutation、merge/push/upload，历史失败也未被改写为 PASS。

### P2

除上述 P1 外，没有确认新的 P2。prior PID fallback 与 disabled-modality augmentation
在代码层已通过 shared reset primitive 和 construction-time fail-closed 修复；其 authored
runtime cases仍未执行。

### P3 — dependency/caller hostile coverage 不能替代真实 GH200 gate

fl_v3/tests/test_s07_b_integration.py:121-160 仍 mock source/build/artifact helper；
真实临时 distribution case :172-198 确实证明同版本 native bytes 改变会改变 digest，
loaded origin 越出 attested root 会失败，但没有运行 editable/wheel 多根布局、
first-import artifact 漂移或 Arrhenius spconv/cumm operator dispatch。实现
fl_v3/src/fl_v3/utils/runtime.py:154-255,263-355 在结构上补齐稳定 suffix include、
cache/data exclude、pre/post import equality、Torch source/config/executable manifest
及 loaded-origin 检查，因此 prior P1 可在静态结构层关闭；actual GH200
Python/native/JIT layout仍为 NOT RUN。

## Review identity、紧急更正与精确 provenance

- Session: S07-B-R2。
- BASE/WORKER SHA: ee5210016b072041db4956f26834ecfdffcbc206。
- REMEDIATION_BASE: df13025bc6582b9b436d1df065de75c03e92782d。
- O-036 handoff: 9d9f21f2043139bbc05082acc156ba25c127ca57。
- PRIOR_REVIEW_SHA: bcffdece226e73207509ca86540443e7640fb6c5。
- Source branch: codex/s07-b-integrated-cl-stack；唯一创建/使用的 review branch:
  codex/s07-b-r2-integrated-cl-stack-review。
- APPROVED_COMPUTE: none；未运行 pytest、Slurm/srun、GPU、data、model、devkit
  metric、training/evaluation 或 dependency import gate。

Kickoff 的 prior-review blob 原值 3c8985f... 被 S00 紧急更正。更正到达时尚未物化任何
bytes；本 reviewer 独立核验并只使用：

- exact blob dc879423d18c2448619b50fd7e819165e7dad995；
- byte size 47,254；
- SHA-256 eb836e1400102a55798f23cfabbd29d2d379a7bb91f673ee999f42b5cc52a73c。

物化后、追加本 R2 段前再次得到相同 blob/SHA-256。该 prefix 原样包含全部
S07-A/S07-B-R bytes；没有 merge/cherry-pick prior reviewer history。
PRIOR_REVIEW_SHA 不是 WORKER_SHA ancestor（exit 1）。

Startup 满足门禁：top-level 为
/home/gaohui/.codex/worktrees/5dcc/fl_weather_project，HEAD exact 为 WORKER_SHA，
branch name empty、status clean。只在此后创建上述唯一 review branch；未进行其他
branch/worktree 操作。

## Commit topology、diff 与 ownership

Remediation 是单亲线性链：

    df13025bc6582b9b436d1df065de75c03e92782d
      -> edc12d87b4e00e11cfdac52a7bbaab02d600bcae
      -> 9d9f21f2043139bbc05082acc156ba25c127ca57
      -> 4ce2366df2925161adae8fea393d5fca64836d40
      -> ee5210016b072041db4956f26834ecfdffcbc206

df13025..9d9f21f 为 17 paths、863 insertions/61 deletions；
9d9f21f..WORKER_SHA 为 11 paths、480 insertions/96 deletions。逐路径对照：

- 第一段只在 O-035/O-036 scoped runtime/config/eval/data/test/handoff ownership；
- 第二段只在 O-037 明列的四个 primary、四个 historical caller、test_s07_b_*.py、
  static launcher 和 S07 HANDOFF；
- 两段都未改 canonical、fl_v3/collab、fl_v2、S07/RUN_REQUEST.md、
  S07/RESULTS.md 或 prior S07/REVIEW.md；
- git diff --check df13025..WORKER_SHA 无 remediation-authored warning；
- WORKER_SHA parent exact 为 4ce2366df2925161adae8fea393d5fca64836d40。

WORKER_SHA 内 canonical snapshot 只到 O-032；本 review 只读使用
codex/s00-orchestra-ledger@6bea4f5b3d2eda723606d8b456655ae007397e86
核验 O-033 至 O-037，未合入或写回该 ref。

## Prior finding closure matrix

| Prior finding | R2 独立结论 | 证据/边界 |
|---|---|---|
| P1 strict official-eval caller | CLOSED — static code level | centralized_train.py:69-171,277-283 使用完整 loader重载 checkpoint、hash-bound raw/EMA、token-complete val、一次 decode、cap<=500、actual mode/config/data/checkpoint/dependency provenance并进入 official seam；detection_eval.py:111-229,260-301 不做第二 threshold。Actual devkit/GH200 NOT RUN。 |
| P1 six-task caller migration | PARTIAL / NEW P1 ABOVE | fusion_ablation.py:110-125,268-307 和 arrhenius_mini_matrix.py:208-242,337-385 已消费六任务且无 max_objects；T4/T5 helper使用完整 checkpoint。但 T5 task_shard 顺序仍不可运行且先走未绑定 data path。 |
| P1 dependency identity | CLOSED STRUCTURALLY; RUNTIME OPEN | Torch version/source/build config/executable manifest及 spconv/cumm sources/files/origins/pre-post checks存在；same-version/different-binary和 outside-root fixture有效。Actual GH200 set NOT RUN。 |
| P2 PID fallback | CLOSED STATICALLY | zip_backend.py:477-533 两条 path共享 reset，重置 location/cache/read/byte/modality state；raw-fork test test_s07_b_data_lifecycle.py:116-154 检查 child zero、parent unchanged。Test NOT RUN。 |
| P2 disabled-modality augmentation | CLOSED STATICALLY | dataset.py:189-211 在 blob/data iteration 前拒绝 C+GTpaste、C+BEV aug、L+image flip；hostiles :157-171。Test NOT RUN。 |
| P3 coverage/wording | CLOSED FOR AUTHORED MATRIX, EXECUTION OPEN | test_s07_b_data_lifecycle.py:80-113 明列 depth 1/10 × directory/ZIP × C/L/F counters；handoff明确未执行。 |

## Adversarial checks

### Strict official evaluation

- s06.v1 将 evaluation.timing 与 checkpoint_weights=raw|ema 纳入 canonical hash
  （resolved.py:31-57,170-228,332-340）。
- checkpoint.py:333-402 拒绝 field/schema/config/data/mode/precision/EMA presence drift及
  legacy/partial payload。
- strict caller构造一个 non-shuffled完整 val loader，只调用一次 decoder，拒绝 cache/decode
  duplicate、missing/extra token和 >500 boxes。
- decode_eval_set 只传 score_threshold，不传 global top-K；run_detection_eval 直接调用
  DetectionEval.evaluate()。
- timing只写独立 artifact，不进入 decode/official output；CPU synthetic test存在但未执行，
  actual CUDA timing-neutrality仍需 runtime gate。

### Six-task与 historical disposition

- O-037 primary caller 的 model.decode 调用均不再传 max_objects；fusion ablation 的
  cond4/cond5a 读取 task_outputs，invariance遍历六任务全部 field；mini matrix telemetry/
  delta遍历全部任务。
- reviewed decoder的 task-local name -> devkit-global ID、per-class K=500、task-wide pinned
  NMS/post=83及 deterministic order未被 remediation 修改；head/NMS/coordinate/metric/
  protocol均未改变。
- current-repo inventory未找到四个 historical scripts 的 active shell launcher；它们仍可
  直接执行，故不能称 dead。实际控制流证明 _t4_fd_diagnose.py:15-19 顶层立即失败；
  t3_trainval_reeval_fullval.py:18-22、p3_crt_probe.py:65-69、
  p3_grad_conflict.py:25-29 在 main首行失败，均早于 config/data/model/checkpoint。
  collab旧结果和所有 historical negative evidence未改。

### Scientific guardrails

- 五个 strict templates仍含 unknown template_only，camera init/hash/cache/manifest sentinels
  没有变成真实身份；full trainval t1.v2仍缺失。
- C/L/F model/head/decode/coordinate、DDP fail-closed、100/1000/profile/full metric/
  scientific forbidden scope不变。
- T4/T5 helper本身使用 S06 model/optimizer/scheduler/scaler/EMA topology与
  load_checkpoint，没有 bare-state shortcut；但 T5 caller-order P1 阻止完整 closure。
- S06 341997 45/62 failure、bare-sbatch no-job、342014 bounded 66/66；S05
  336731 43/44与336738 44/44；S04 failures/diagnostic/341695 15/15；S03
  335630 failure/336708 pass；S01 332648 failure及332651/333206各自边界均保留。

## Tests/checks actually run 与 NOT RUN

本 R2 实际只运行 read-only/static checks：

1. startup root/HEAD/detached/status与唯一 branch门禁：PASS；
2. ancestry、parents、两段 diff、ownership、RUN_REQUEST/RESULTS unchanged：PASS；
3. corrected prior blob/size/SHA-256与物化 prefix：PASS；
4. bash -n run_s07_b_static_checks.sh：PASS；
5. 内存 ast.parse 18 files：AST_OK=18；
6. stdlib JSON parse五个 templates：JSON_OK=5；
7. committed static launcher：PASS，五个 hashes 为 d2eaa46c...、bd8c57e8...、
   df7f36fe...、62524223...、1658cd5e...；
8. caller/launcher inventory与 git diff --check：PASS，并发现上述 T5 P1。

明确 NOT RUN / NO IMPLIED PASS：

- pytest，包括所有新 T5 caller、dependency、raw-fork和交叉矩阵 tests；
- Torch/spconv/cumm actual import、native/JIT/build/source identity；
- checkpoint raw/EMA load、model/optimizer construction、CPU/CUDA rollback；
- directory/ZIP payload、fork/spawn/persistent workers；
- C/L/F forward/backward、fp16 S04 option-A、concurrency/EMA、memory；
- official devkit、GT-as-pred、T4/T5 five-condition decode、rotate-NMS profile；
- Slurm/srun/GPU、full cache、100/1000、profile、mAP/NDS、DDP、matrix、seed/rerun。

## Interpretation、residual risk 与 O-009

允许解释：strict centralized official-eval、executable identity、PID fallback、mode
augmentation及大部分 fusion/mini/T4/T5 helper remediation已形成可审查静态实现。除 T5
shard caller-order 外，prior直接代码缺口已关闭。五个 templates仍只是 fail-closed templates。

禁止解释：不得称 T5 five-condition caller complete、S07-B static/integration PASS；
不得把 authored/mocked/static tests当 GH200 evidence；不得称 production/full-data/
checkpoint/performance/scientific ready，亦不得声称 mAP/NDS、fusion gain、FL、
attack/defense、generalization或publication evidence。不得提交 O-009/full-cache/model job、
merge/push/upload。

Residual risk：full t1.v2 artifact不存在；T5 preflight顺序必须修复；actual Torch/spconv/cumm
first-import/operator layout未知；integrated fp16、same-instance lifecycle、batch/gradient/
memory、official eval与rotate-NMS性能均未执行；Protocol-B split/client ownership不在本
CL capability review内。

因存在静态 P1，当前不应创建或提交 O-009 request。修复并通过新一轮独立 code review 后，
S00 才可基于 exact new SHA 按 prior review 的单节点/单 GH200/<=60分钟、real-mini +
temporary ZIP、无 retry/array/follow-on bounded gate 建议准备 immutable request。
这仍不是 approval；full cache、100/1000、full-data profile/metric、DDP/scientific cells
始终需要新的 exact owner decision。

## Final verdict

**CHANGES-REQUESTED for S07-B at
ee5210016b072041db4956f26834ecfdffcbc206.**

Strict official-eval、dependency identity、PID fallback、augmentation和大部分 six-task caller
remediation在静态层可接受；但 load-bearing T5 task_shard 仍在权威 checkpoint/config
preflight前构造 compatibility data，并会被当前 t5_attack.json 的缺失 strict fields直接
击穿。现有 mock helper test未覆盖真实 caller顺序。修复 exact SHA必须重新独立 review；
在此之前没有 code-level PASS、runtime gate或任何 production/scientific授权。

---

# S07-B-R3 独立复审 — O-039 authoritative T5 preflight remediation

## Findings（按严重性排序）

### P1 — 完整五条件 shard 没有强制 clean checkpoint，mandatory occlusion control 可被缺省参数静默判 PASS

O-039 的 poison/optional-clean preflight 本身已提前，但 task requirement 仍未 fail closed：

- fl_v3/scripts/t5_attack_eval.py:172-178 仅在 clean_checkpoint 非空时 preflight clean；
  main():634-650 对所有非 null task 都保留 clean 为可选参数，没有按 task/cond4-only 建立
  requirement matrix。
- task_shard():322-341 即使 cond4_only=False 且 clean checkpoint 缺失也继续构造
  val info/dataset/model；clean 最终为 None。
- fl_v3/src/fl_v3/attacks/fusion_ablation.py:248-252 在 clean_model=None 时把
  occlusion_disappeared 留为 None；shard 输出没有把这是缺失 mandatory control 标成失败。
- task_aggregate():403-450 只对 truthy occlusion_disappeared 计数。None 因而被计作
  0/N，并使 passes_not_occlusion = (0 < 0.02) 成为 True。它没有要求每个完整五条件
  row 的 occlusion value 必须是 boolean，也没有拒绝 cond4-only shard。

这违反 active T5 contract 中“clean pre-attack checkpoint 对同一 triggered image 的
occlusion control 必须存在且小于 0.02”的硬门。一个没有 clean checkpoint 的完整 shard
可以伪造 not-occlusion 子门为绿色；若同一 output root 已有 stealth/guards，最终 conjunction
可能接受一个从未执行 clean control 的结果。

此外，现有 preflight 只要求 clean 与 poison 具有相同 resolved SHA、raw/EMA policy 和
runtime dependency identity（t5_attack_eval.py:179-188），并不证明 clean 的实际 selected
raw/EMA 权重就是冻结 subset 所声明的 clean checkpoint。_load_subset():277-288 只比较
subset JSON 内的 checkpoint_checksum 与 config 常量；它没有比较实际加载后的 clean model
checksum。任意同 config 的 complete checkpoint 都可充当所谓 clean control。物理文件 hash
被记录并不能替代与冻结 clean identity 的比对。

Required remediation：

1. 在任何 os.makedirs/device/seed/data/model 前建立显式 task requirement matrix：
   full shard 必须同时有 poison+clean；cond4-only control 明确只允许 poison；aggregate/
   stealth/guards 只需 poison；viz 是否需要 clean 必须显式冻结并测试；null-verify继续在
   preflight前 fail closed。
2. full shard 缺 clean 必须在无副作用边界失败；evaluate_target 的 mandatory path 不应接受
   None clean model。aggregate 必须拒绝 missing/None occlusion values和 cond4-only artifacts，
   不能把缺失 control 当 0。
3. 按 selected raw/EMA policy 将实际 clean weights 绑定到现存冻结
   attack-clean-checkpoint-checksum，或在 owner 冻结的 replacement complete-checkpoint
   identity contract 下绑定等价不可变 identity；不得只比较 clean/poison config。

### P1 — process-local preflight 没有进入 shard artifact，aggregate 可消费另一 checkpoint、另一 config 或重复的 rows

_CHECKPOINT_PREFLIGHTS 只保护当前 Python 进程。task_shard():369-372 写出的 JSON 只有
shard、num_shards、n_targets 和 results；没有 poison/clean physical hash、resolved SHA、
raw/EMA policy、runtime dependency identity、subset identity或 cond4-only/full mode。
task_aggregate():394-400 随后读取 output directory 中所有匹配文件并直接拼接 rows，不核验
这些 artifacts 是否来自本 aggregate 进程刚 preflight 的 checkpoint/config。

同一目录中的旧 shard、不同 poison/clean pair 或 cond4-only control shards 因而可以被当前
aggregate 使用。seen 仅用于 coverage；evaluated 和每条件 counts 仍保留重复 rows
（:399-417），所以重复 artifact 还可让 ASR 分子超过 frozen N。aggregate 最终只验证自己
加载的 poison checkpoint provenance，而不是生成 rows 的 checkpoint。这使“每个 task 都
preflight”在跨进程 fan-out/fan-in 边界失效，并提供可抬高 ASR 或替换 clean-control 的静默
路径。

Required remediation：每个 shard 必须写入并由 aggregate exact-match 验证完整 preflight
identity bundle、subset hash、task mode、shard index/count和 artifact schema；aggregate 必须
要求 exact shard set、唯一 (sample_token, ann_token)、无重复/缺失/额外 target，并把 row
identity绑定到当前 poison/clean preflight。cond4-only controls必须有独立 artifact/result
schema，不能进入 full fusion-aware verdict。

### P2 — 新 caller-order hostile 在执行 main 前就会因无效 strict config 失败，当前不能证明所声称的真实顺序

test_t5_main_preflights_existing_compat_config_before_device_seed_or_data 在
fl_v3/tests/test_s07_b_integration.py:600-616 读取真实 t5_attack.json 后，把
compatibility["nuscenes-dataroot"] 原样写入 synthetic S06 config。当前 config 的该值是空串
（fl_v3/configs/t5_attack.json:6），而 strict resolver 明确拒绝空 dataroot
（fl_v3/src/fl_v3/config/resolved.py:284-286）。因此 resolved =
config_module.resolve_config(raw) 会在 test 调用 module.main() 之前抛 ConfigError。

这不是未执行环境的不确定性，而是静态可证的 authored-test defect。测试目前无法到达
events sentinel，也无法证明 fp32 backfill、preflight-before-os.makedirs/data/model 或
current compatibility config 的 caller order。它还整体 mock _load_model
（test_s07_b_integration.py:644-645），所以即使修正 dataroot，也不会在同一 caller test 中
证明 stored preflight 被实际 model loader 消费。

Required remediation：继续读取真实 compatibility file，但用显式、非空且与 synthetic
checkpoint 相同的 dataroot/cache caller inputs（或明确声明的匹配 CLI inputs），然后让
strict resolved config 合法；给 os.makedirs、task dispatch、device、seed、val-info、
dataset、build_model/load_checkpoint 加顺序 sentinel。只 mock 外部 runtime/data/model
边界，不要用一个 _load_model mock 同时跳过本次要证明的 preflight-consumption contract。

### P3 — hostile matrix 未覆盖 handoff 所列全部拒绝类，mock coverage 仍可能产生假阳性

已 authored 的 hostiles 确实覆盖：missing checkpoint、无 schema 的 legacy/bare payload、
一个 caller precision drift、一个 duplicated resolved SHA drift、post-preflight file mutation、
以及 clean/poison resolved SHA mismatch。但仍缺：

- 有正确 schema 但缺字段/多字段的 partial payload；
- wrong schema；
- 独立的 model_mode、precision、data_identities、checkpoint_identity 和 EMA-presence drift；
- clean/poison raw-vs-EMA policy drift及 runtime dependency identity drift；
- full shard clean absence、cond4-only/full artifact混用、重复 shard/row；
- os.makedirs 必须在 poison+required-clean preflight 后；
- post-preflight derived config/provenance field drift；
- preflight 首次 torch.load 期间文件被替换的 before/after byte-stability hostile。

静态实现对其中若干项确实已有拒绝分支（t5_attack_eval.py:114-168,179-188,221-264），所以
本 finding 不把未执行 test 直接等同于实现失败；但当前 suite无法支持 handoff中“完整
missing/legacy/partial/embedded/file/pair drift coverage”的表述，也不能作为未来 O-009
gate 的充分回归集合。

### 无 P0 finding

未发现本 remediation 修改 split/leakage、model/head/NMS/metric/protocol，未发现
RUN_REQUEST/RESULTS/canonical/collab/fl_v2 mutation、未授权 compute、merge/push/upload，
也未见把历史 failed job 重写为 PASS。上述 P1 足以阻止 code-level PASS 和 O-009 request，
但没有证据支持升级为 P0。

## Review identity、provenance 与 startup

- Session: S07-B-R3。
- BASE_SHA / WORKER_SHA:
  b6d132058eee9532b3563d2fe87358be3de6a0a7。
- REMEDIATION_BASE:
  ee5210016b072041db4956f26834ecfdffcbc206。
- PRIOR_REVIEW_SHA:
  afb81f51cdf311de215d351e92e2bf5ac6c3bd43。
- Source branch: codex/s07-b-integrated-cl-stack；唯一创建的 delivery branch:
  codex/s07-b-r3-integrated-cl-stack-review。
- APPROVED_COMPUTE: none；未运行 pytest、Torch import/runtime、GPU、data、model、
  checkpoint load、devkit metric、Slurm/srun。

Startup 原样满足门禁：top-level 为
/home/gaohui/.codex/worktrees/4cc6/fl_weather_project，HEAD exact 为 WORKER_SHA，
branch name empty、status clean。之后只创建上述 owner-authorized delivery branch；
未进行 worktree add/move/remove/prune、其他 branch switch/create/delete。

Prior review provenance 独立核验：

- afb81f5 parent exact 为 ee5210016b072041db4956f26834ecfdffcbc206；
- commit 唯一 changed path 为 fl_v3/usenix27_orchestra/handoffs/S07/REVIEW.md；
- blob 40618498861484178a77b9096f8c0e2e79eab550；
- size 60,954 bytes；
- SHA-256 e93daac54472c568a41f06c069cc85216e8cec1914e94be48c5e33dff3c46f8b。

当前 worker tree 原有 22,715-byte S07-A prefix与 prior blob前 22,715 bytes 的
SHA-256均为 d9bbc63c9b5c52963ad4e8cbdd9af248aac5f371c43bd6e7627a20d87bda9952。
随后只物化 exact 60,954-byte prior blob，再追加本 R3；没有合入/cherry-pick prior
reviewer ancestry。

WORKER_SHA 内 canonical snapshot只到 O-032。本 reviewer 只读核验
codex/s00-orchestra-ledger@e48029829c9dea0e1f3a93f159b8f6e12a5acfc5 中 O-033 至
O-040，尤其 O-039 return和 O-040 R3 envelope；没有合入或写回该 ref。

## Exact diff、ownership 与 caller-order tracing

ee52100..WORKER_SHA 是线性四提交：

    ee5210016b072041db4956f26834ecfdffcbc206
      -> 2c6203c02f118678dcfb71e3b67ddc703dbd2f8a
      -> c114fd58fc4070a19aaf712e1140b4bc8ade0c3d
      -> 9403178ac2833e5e11e641223b728c6fa168657f
      -> b6d132058eee9532b3563d2fe87358be3de6a0a7

Exact diff仅三路径：t5_attack_eval.py、test_s07_b_integration.py、S07/HANDOFF.md。
RUN_REQUEST blob保持 0ef42edc0d69f3805dc960fafa06d6b9edba438d，RESULTS blob保持
f3ac930d87cbcf3a9ce056ad34af2153c2890ddf。models/eval/config/training/data/configs
各 tree ID在 remediation base与 WORKER_SHA完全相同；没有 model/head/NMS/metric/
protocol/full-t1.v2/template/DDP/compute boundary变化。

静态 main→task trace：

1. main 只先解析 args和 compatibility/attack config；null-verify在任何 preflight/
   output副作用前固定失败。
2. 其余六选项中的五个 executable task都在 main():650 调
   _preflight_t5_checkpoints，随后才在 :651 os.makedirs、:652-653 dispatch。
3. poison preflight完整执行 schema/exact field set/resolved config/embedded metadata/
   checkpoint identity/component shape/EMA presence/caller drift/physical data/dependency identity/
   checkpoint file hash；若提供 clean，clean完整执行同一路径，pair comparison完成后才更新
   cfg与registry。失败前没有 partial cfg/registry mutation。
4. task_shard、task_aggregate、task_stealth、task_guards、task_viz各自第一条 executable
   statement均为 _require_preflight；其后才 device/seed/subset/val-info/dataset/model。
   fp32 authoritative precision在 _seed 前进入 cfg。_load_model只消费 stored resolved/
   strict/hash，先拒绝 cfg strict drift和checkpoint byte drift，再构建 model/components并用
   complete load_checkpoint，按 stored raw/EMA policy选权重，load后再次hash。

因此 R2 原 caller-order P1 的“preflight晚于 seed/data”核心因果已在静态代码层修复。
但上述 P1 表明 task-specific clean requirement与跨进程 artifact binding仍允许绕过该
preflight的科学目的，故不能把核心顺序修复扩张为完整 T5 closure。

## Prior finding closure matrix

| Prior finding | R3 conclusion | Evidence / remaining boundary |
|---|---|---|
| R2 P1 main→task_shard authoritative ordering | CLOSED FOR IN-PROCESS ORDER; NEW P1 ABOVE | main preflight在 os.makedirs/dispatch前，各 task首行require；fp32在seed前。clean requirement和fan-out artifact identity未闭合。 |
| strict official-eval caller | RETAINS STATIC CLOSURE | 本 diff未改 centralized/eval tree；actual devkit/GH200仍 NOT RUN。 |
| six-task caller migration | RETAINS STATIC CLOSURE | 本 diff未改 model/head/decode/fusion ablation；无 max_objects/NMS/metric change。full T5 science仍受新 P1阻断。 |
| dependency identity | RETAINS STRUCTURAL CLOSURE; RUNTIME OPEN | preflight实际调用既有 verifier；真实 Arrhenius Torch/spconv/cumm path/operator/first-import仍 NOT RUN。 |
| PID fallback | RETAINS STATIC CLOSURE | data tree未变；raw-fork authored test仍 NOT RUN。 |
| disabled-modality augmentation | RETAINS STATIC CLOSURE | dataset tree未变；cross-product authored tests仍 NOT RUN。 |
| mode/depth/build coverage wording | EXECUTION OPEN | 本次没有运行任何相关 test；不得扩张 prior边界。 |

## Checks actually run 与 NOT RUN

本 R3 只运行安全、read-only/static checks：

1. startup root/HEAD/detached/status与branch absence：PASS；
2. prior review parent/path/blob/size/SHA-256及60,954-byte prefix：PASS；
3. O-033..O-040只读 canonical ledger核验：PASS；
4. remediation commit parents、三路径 diff、ownership、RUN_REQUEST/RESULTS及核心 tree
   immutability：PASS；
5. git diff --check ee52100..WORKER_SHA：PASS；
6. bash -n run_s07_b_static_checks.sh：PASS；
7. stdlib ast.parse两个 changed Python files与stdlib JSON parse当前 t5 config：
   AST_OK=2 / JSON_OK=1；
8. 手工+AST级 caller-order、task requirement、artifact schema和test reachability追踪：
   发现上述 P1/P2/P3。

没有运行 committed static launcher本体，因为它会执行 py_compile及导入 production config
package；kickoff明确禁止 Torch/runtime execution，本 reviewer使用无 import的 AST/JSON
替代检查。

明确 NOT RUN / NO IMPLIED PASS：

- pytest，以及所有新/旧 S07-B hostiles；
- 当前 compatibility config + synthetic complete checkpoint 的真实 main execution；
- Torch/spconv/cumm import/native/JIT/build/source identity；
- physical cache/manifest load、directory/ZIP payload、fork/spawn/persistent workers；
- checkpoint raw/EMA load、model/optimizer/scaler/scheduler/EMA construction与rollback；
- C/L/F forward/backward、fp16 S04 option-A、same-instance concurrency、memory；
- T5 shard/aggregate/stealth/guards/viz、five-condition decode、clean occlusion control；
- official devkit/GT-as-pred/rotate-NMS profile；
- Slurm/srun/GPU、full t1.v2 cache、100/1000 steps、profile、mAP/NDS、DDP、matrix、
  seed/rerun/automatic retry。

历史 evidence边界全部保留：S06 Job 341997 45/62 failure和342014 bounded 66/66；
S05 336731 43/44与336738 44/44；S04 failures/diagnostic与341695 15/15；
S03 335630 failure/336708 pass；S01 332648 failure及332651/333206各自限制。

## Interpretation、residual risk 与 O-009

允许解释：O-039 已把 complete poison/optional-clean S06 config/data/dependency preflight
移到 os.makedirs/device/seed/data/model之前；missing strict fields由authoritative checkpoint
回填，overlapping scientific drift失败，三项declared eval override保留；_load_model消费stored
preflight并执行双hash与complete loader。以上仅为静态实现结论。

禁止解释：不得称 mandatory clean control、full T5 five-condition caller或fan-out/fan-in
provenance已闭合；不得称新 tests可运行或已PASS；不得称 actual checkpoint/cache/dependency/
raw/EMA/runtime ready；不得声称 production/full-data、100/1000、mAP/NDS、fusion gain、
FL、attack/defense、generalization或publication evidence。

Residual risk：full trainval t1.v2 artifact仍不存在；实际 Torch/spconv/cumm first-import/
operator layout未知；checkpoint首次 torch.load 期间的before/after byte stability未测；
integrated fp16、mode-aware ZIP workers、rollback、official eval、rotate-NMS、host memory均未执行；
Protocol-B split/client ownership不在本 CL capability review内。

由于存在静态 P1/P2，当前不应准备或提交 O-009 request。先修复 task-specific clean/identity/
artifact contract与load-bearing test，再从 exact new SHA做新一轮独立 review。只有 code-level
review PASS 后，S00才可按 prior envelope准备一个 exact bounded O-009 request：one node/one
GH200、<=60 min、real-mini+temporary ZIP、无array/retry/follow-on。即使该未来 runtime gate
PASS，也只可能建立bounded engineering evidence，不是 production/full-data/scientific PASS。

## Final verdict

**CHANGES-REQUESTED for S07-B at
b6d132058eee9532b3563d2fe87358be3de6a0a7.**

R2 的核心 caller-order P1 已在单进程静态顺序层修复；但完整 shard仍可缺失或替换 mandatory
clean control并让not-occlusion误判为PASS，且fan-out artifacts没有绑定各自preflight identity。
新caller-order hostile还会因真实compatibility config的空dataroot在进入main前失败。修复并
重新独立review前，没有code-level PASS、O-009 request、runtime/production或scientific授权。
