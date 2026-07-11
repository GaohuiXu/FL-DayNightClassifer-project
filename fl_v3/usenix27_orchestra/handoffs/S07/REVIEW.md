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
