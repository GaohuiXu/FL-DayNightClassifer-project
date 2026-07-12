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

---

# S07-B-R4 independent review — O-041 mandatory-control/artifact remediation

## Findings (severity ordered)

### P1 — the final aggregate still trusts unversioned, unbound stealth/guard siblings, so stale evidence from another checkpoint/subset can turn the current gate green

The new full-shard loader correctly binds each ablation row to the current poison
preflight and frozen clean/subset identities, but the load-bearing final conjunction
still consumes two other cross-process artifacts through the permissive `_load_json`
helper (`fl_v3/scripts/t5_attack_eval.py:93-97,654-662`):

- `stealth.json` is used only by reading `poisoned_clean_car_recall`; aggregate does
  not require its producer's `checkpoint_checksum` or `verified_attack_provenance`
  to match the current selected poison weights (`:650-660,687-700`).
- `cond5a_guards.json` contains no checkpoint/config/runtime/subset identity at all
  (`:790-794`), and aggregate accepts any JSON object whose two favorable values
  make `cond5a_valid` true (`:654-662`).
- Neither file has a versioned exact-key/type schema, and neither is tied to the
  current resolved SHA, physical checkpoint SHA, raw/EMA policy, runtime dependency
  SHA, selected poison checksum, frozen subset hash, or shard mode/count.

Consequently, a current poison checkpoint plus correctly bound current shards can
reuse favorable `stealth.json` and `cond5a_guards.json` left in the same output root
by a different checkpoint/subset/run. Those two stale values can satisfy
`stealth_ok` and `cond5a_guards_valid`, allowing the overall conjunction to become
green even though the corresponding controls were never run for the current
artifact identity. The current checkpoint's separate `verify_attack_provenance`
call at `:624-631` does not authenticate those sibling measurements.

Required remediation: make stealth and guards versioned exact-schema artifacts,
bind their current selected poison identity bundle (and the frozen subset identity
for guards), and exact-match those identities in aggregate before using any raw
metric. A single immutable fan-out/run manifest or equivalent fresh-output contract
may be used, but fixed filenames plus `exist_ok=True` are insufficient. Add hostile
fixtures proving that favorable stale/mixed sibling files cannot make the current
gate green.

### P2 — `viz --cond4-only` bypasses the claimed cond4-only task restriction

`_validate_task_requirements()` handles `viz` at
`fl_v3/scripts/t5_attack_eval.py:196-199` and returns immediately after requiring a
clean checkpoint. The generic `--cond4-only is valid only for shard` rejection is
below that return and covers only aggregate/stealth/guards (`:200-205`). Therefore
`--task viz --cond4-only --checkpoint P --clean-checkpoint C` is accepted and the
flag is silently ignored.

This contradicts the required task matrix: cond4-only is the distinct poison-only
shard schema, while viz is a separately frozen poison+clean task. Move the cond4
restriction ahead of all non-shard task branches or reject it explicitly in viz,
and add a before-side-effect hostile. This is fail-closed contract drift rather
than a metric-result P1, but it prevents accepting the claimed complete matrix.

### P2 — aggregate does not require the canonical filename set for the declared shard tuple

The writer defines canonical tuple- and mode-specific paths at
`fl_v3/scripts/t5_attack_eval.py:332-336`, but aggregate ignores that helper. It
instead accepts any `ablation_shard_*.json` names (`:354-358`) as long as their count
equals `num_shards` and their internal indices/identities pass. Thus two valid
current artifacts renamed to arbitrary matching-prefix names are accepted, and the
absence of the exact expected `ablation_shard_{i}_of_{K}.full.json` files is not
detected. The internal index and duplicate-row checks are valuable, but they do not
close the mandatory filename-level stale/mixed/duplicate contract.

Required remediation: derive the exact expected full-artifact path set for
`i=0..K-1`, require that set byte-for-byte, and reject every extra shard-prefixed
file (including cond4, aliases, old counts, and renamed copies). Add hostile cases
for a renamed valid artifact and a stale extra artifact.

### P3 — the authored hostile suite still does not support the handoff's “complete task matrix” and complete artifact-guard wording

`test_t5_task_requirement_matrix_fails_before_output_side_effect`
(`fl_v3/tests/test_s07_b_integration.py:783-808`) tests one invalid case (full shard
without clean) and then only valid task combinations. It does not exercise:

- cond4 shard with an illicit clean checkpoint;
- clean checkpoints on aggregate/stealth/guards;
- viz without clean or viz with `--cond4-only`;
- invalid shard index/count;
- null-verify failing before preflight/output creation.

The artifact hostile at `:891-962` covers missing boolean occlusion, cond4/full
mixing, duplicate index, duplicate row, and a missing shard, but not canonical
filename aliases, stale/mixed stealth/guard siblings, extra shard files, selected
clean checksum drift, subset identity drift, or the exact-key/type branches added
in the final hardening commit. No authored caller-order case exercises selected EMA
weights; the reachable order test uses raw policy only. These are test-evidence
gaps, not proof that every corresponding implementation branch is wrong, but the
handoff's “complete task matrix” wording at `HANDOFF.md:1097-1107` is too broad.

### No P0 finding

The remediation does not alter split/data ownership, model/head/NMS/metric/protocol
semantics, canonical/collab/fl_v2 content, compute authorization, or historical
negative results. No unauthorized runtime/Slurm/GPU action, merge, push, upload, or
publication occurred. The findings above are sufficient to block code-level PASS,
but do not justify P0.

## Review identity, exact prefix, startup, and topology

- Session: `S07-B-R4`.
- BASE/WORKER SHA: `098cfded362ec276d3e697e9150cd7f05de3e238`.
- REMEDIATION_BASE: `b6d132058eee9532b3563d2fe87358be3de6a0a7`.
- PRIOR_REVIEW_SHA: `d6f8ae6233c4900e63151d4ee8fab98d549695b8`.
- Delivery branch: `codex/s07-b-r4-integrated-cl-stack-review`.
- APPROVED_COMPUTE: none.

Startup was exact and clean before the only authorized branch creation: repository
root `/home/gaohui/.codex/worktrees/8249/fl_weather_project`, detached HEAD equal to
WORKER_SHA, empty branch name, empty status. No worktree add/move/remove/prune,
other branch creation/switch/deletion, or external action occurred.

The prior review was independently verified and materialized without reviewer
ancestry:

- `d6f8ae6` parent is exact REMEDIATION_BASE and its sole changed path is this
  `REVIEW.md`;
- exact blob `1791a1cfc56fae0f2f3093a733454762c180d335`;
- exact size 78,115 bytes;
- exact SHA-256
  `8c18ed7a4b0a19604fe314b10f6fbe612a2e754e826189b0f57d0c22ab00cfd8`;
- the prior reviewer commit is not an ancestor of WORKER_SHA.

Before this R4 appendix was added, the working file reproduced all three values
exactly. Binding canonical decisions through O-041 were read from the read-only S00
ledger snapshot; no canonical commit was merged or written back.

The remediation is an exact linear four-commit chain:

    b6d132058eee9532b3563d2fe87358be3de6a0a7
      -> cf99ba30c4a2edbeef99af4fc8aee85f87b65bd7
      -> 4c22ba1eed26fef998629070a3304a57f5fcafe4
      -> b855a2a3742fdb7729a7d96667ec82e5cb60e855
      -> 098cfded362ec276d3e697e9150cd7f05de3e238

The exact diff is 599 insertions / 29 deletions over only:

1. `fl_v3/scripts/t5_attack_eval.py`;
2. `fl_v3/tests/test_s07_b_integration.py`;
3. `fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md`.

`RUN_REQUEST.md` remains blob
`0ef42edc0d69f3805dc960fafa06d6b9edba438d`; `RESULTS.md` remains blob
`f3ac930d87cbcf3a9ce056ad34af2153c2890ddf`. Model, eval, config, training, data,
and `fl_v3/configs` tree IDs are identical between REMEDIATION_BASE and WORKER_SHA.
Ownership/topology therefore match the envelope.

## R3 requirement and closure matrix

| Requirement / prior finding | R4 conclusion | Static evidence and remaining boundary |
|---|---|---|
| Full shard poison+clean; cond4 poison-only; aggregate/stealth/guards poison; viz poison+clean; null preflight-free failure | **PARTIAL — P2 above** | Full/cond4/aggregate/stealth/guards/null logic is fail-closed at `t5_attack_eval.py:180-205,881-889`; viz requires clean, but silently accepts `--cond4-only`. |
| Complete checkpoint preflight and before/after initial `torch.load` byte stability | **CLOSED STATICALLY** | `:112-177` validates exact schema/field set/config/data/checkpoint/EMA/physical/runtime identity and hashes bytes around `torch.load`; hostile replacement exists at test `:811-856`. Runtime NOT RUN. |
| Poison/clean pair compatibility and no partial registry/config mutation | **CLOSED STATICALLY** | `:209-239` compares resolved SHA, raw/EMA policy and runtime identity before mutating cfg/registry; mismatch tests retain empty state. |
| Stored-preflight consumption and post-preflight byte/config stability | **CLOSED STATICALLY** | `_load_model` at `:252-303` requires stored preflight, rejects config/file drift, uses complete S06 loader, applies selected EMA when declared and rehashes after load. Actual load NOT RUN. |
| Actual selected clean checksum bound to config and subset before val data/evaluation | **CLOSED STATICALLY FOR FULL SHARD** | `:519-537` loads selected poison/clean models and checks clean checksum against both pins before `_val_info`/`_val_dataset` at `:545-546`; full path refuses `clean=None` at `:561-564`. |
| Distinct full/cond4 schema and full aggregate exclusion | **CLOSED STATICALLY** | Separate version/mode constants and filenames at `:34-37,332-344`; full aggregate exact schema/mode checks at `:392-395`. Cond4 result-schema hostile depth remains P3. |
| Exact poison/clean shard identity, subset, mode, index/count | **CLOSED EXCEPT FILENAME P2** | Full top-level and identity exact keys, current poison equality, clean selected checksum/policy/runtime/resolved checks and subset/index/count validation at `:376-426`. Canonical path set is not enforced. |
| Exact unique shard set/targets/rows/coverage | **CLOSED STATICALLY EXCEPT FILENAME P2** | `:369-371,398-463` rejects bad indices, duplicates, wrong row counts/assignment and missing/extra target coverage. Membership in deterministic `targets[index::count]` is exact. |
| Mandatory boolean occlusion and no `None -> 0` | **CLOSED STATICALLY** | Every full row must be evaluated and boolean at `:439-458`; aggregate increments only after validation (`:631-650`). |
| Current compatibility-config caller order | **CLOSED FOR ORDER, STATIC/AUTHORED ONLY** | Test `:590-711` now supplies nonempty matching data/cache/manifest paths, runs real `main -> task_shard -> _load_model`, and mocks only external physical/dependency/data/model/checkpoint boundaries. It proves both preflights precede first output creation and selected clean checksum precedes val-info/dataset. It deliberately uses an empty target slice, so it is not five-condition runtime evidence. |
| R3 P3 hostile matrix | **PARTIAL — P3 above** | Checkpoint field/schema/metadata/file/pair branches are substantially expanded, but task/artifact/sibling/EMA cases remain absent and no test was executed. |

## Retained R2 closures and scientific boundaries

The exact remediation does not touch the trees underlying the prior static
closures for strict centralized official evaluation, six-task decode/caller
migration, executable dependency structure, PID fallback, disabled-modality
augmentation, or the depth/backend/mode authored matrix. Those conclusions retain
only their previous static limits. Actual devkit evaluation, Arrhenius
Torch/spconv/cumm layout, raw fork/spawn, mode-aware ZIP workers, model/optimizer
rollback, integrated fp16/S04 dispatch, concurrency/EMA, host memory, and real C/L/F
construction remain open.

Historical evidence remains distinct and unmodified: S06 Job 341997 is 45/62
FAILED and Job 342014 is bounded 66/66 PASS; S05 336731 is 43/44 and 336738 is
44/44; S04 failures/diagnostic and 341695 15/15, S03 335630 failure/336708 pass,
and S01 332648 failure plus 332651/333206 bounded evidence retain their original
interpretation. Full trainval `t1.v2` still does not exist.

## Checks actually run and explicit NOT RUN

R4 performed only safe read-only/static checks:

1. exact startup root/HEAD/detached/status and sole delivery-branch check: PASS;
2. REMEDIATION_BASE ancestry and all four single-parent links: PASS;
3. prior review parent/sole path/non-ancestry/blob/size/SHA-256 and exact prefix: PASS;
4. exact three-path ownership, 599/29 diff, unchanged RUN_REQUEST/RESULTS blobs and unchanged core tree IDs: PASS;
5. `git diff --check b6d1320..098cfde`: PASS;
6. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
7. stdlib-only AST parse of the two changed Python files and JSON parse of current `t5_attack.json`: `AST_OK=2`, `JSON_OK=1`;
8. manual/AST-level task, preflight, clean-checksum, fan-out/fan-in, sibling-artifact and hostile reachability trace: produced the findings above.

The committed static launcher itself was not run because it performs `py_compile`
and imports the production config package; the envelope forbids Torch/runtime
execution. No pytest was run.

Explicitly **NOT RUN / NO IMPLIED PASS**:

- pytest, including every new or retained S07-B hostile;
- Torch/spconv/cumm imports, native/JIT/build/source verification or operator dispatch;
- physical cache/manifest reads, directory/ZIP payloads, fork/spawn/persistent workers;
- checkpoint raw/EMA loading, model/optimizer/scaler/scheduler/EMA construction or rollback;
- selected-clean checksum on a real checkpoint, T5 shard/aggregate/stealth/guards/viz, five-condition or occlusion evaluation;
- C/L/F forward/backward, fp16 S04 option-A, concurrency, host memory or rotate-NMS profile;
- official devkit, Slurm/srun/GPU, full cache, 100/1000 steps, full-data profile, mAP/NDS, DDP, matrix, seed, rerun, retry, FL, attack or defense cell.

## Interpretation, residual risk, and O-009 boundary

Allowed interpretation: the two R3 scientific-integrity P1s are substantially
closed for the core full-shard path at static code level. Mandatory clean selected
weights, boolean occlusion, exact current poison/full-shard identity, deterministic
target coverage, and the corrected caller order are now present and reviewable.

Forbidden interpretation: this is not a code-level S07-B PASS because the final
gate can still consume stale/unbound sibling evidence, the task/filename contracts
remain incomplete, and no runtime test ran. It is not production, full-data,
checkpoint, performance, mAP/NDS, fusion-gain, FL, attack/defense, generalization,
or publication evidence.

No O-009 request should be prepared or submitted from this SHA. First bind the
stealth/guard artifacts to the current aggregate identity, close the viz-cond4 and
canonical shard-filename seams, correct the handoff/test matrix, and obtain another
independent static review. Only after a code-level review PASS may S00 prepare a
new exact bounded request under the prior one-node/one-GH200/<=60-minute/no-array/
no-retry envelope. Even a future bounded runtime PASS would not establish
production/full-data/scientific readiness.

## Final verdict

**CHANGES-REQUESTED for S07-B at
`098cfded362ec276d3e697e9150cd7f05de3e238`.**

The R3 mandatory-clean, selected-checksum, boolean-occlusion, fan-out identity and
caller-order defects are closed in the central full-shard path at static code
level. However, the final conjunction still admits stale/mismatched stealth and
guard evidence, viz silently accepts the cond4-only flag, and aggregate does not
require the canonical shard filename set. The claimed hostile matrix is also
incomplete and entirely unexecuted. A new exact remediation SHA and independent
review are required before code-level PASS or any O-009 request.

---

# S07-B-R5 independent review — O-043 run/artifact remediation

## Findings (severity ordered)

### P1 — the immutable plan does not bind the guard sample selection, so an under-scoped favorable guard can satisfy the final gate

The manifest task plan is constructed only from `num_shards`
(`fl_v3/scripts/t5_attack_eval.py:395-400`) and inserted at `:417`. The
load-bearing guard has a separate mutable `--guard-samples` parameter (`:1169`),
which selects only a sorted prefix of the frozen subset (`:1024-1029`). Neither
the manifest nor the guard artifact records that parameter or the exact selected
sample/target set (`:1043-1060`), and aggregate validates nonnegative reported
counts but never compares them with a frozen selection policy (`:637-670,853-886`).

Therefore a run initialized with the declared full-shard plan can execute
`guards --guard-samples 1`; its artifact still has the current run, manifest,
checkpoint and whole-subset identities and is accepted by a later default
aggregate. A favorable one-sample control can stand in for the intended 40-sample
control and make `cond5a_guards_valid` true. This changes load-bearing evidence
without changing any checked identity.

Required remediation: freeze and validate the existing guard-selection policy
without changing its scientific meaning, bind the declared count and exact
selected sample/target identity into the manifest and guard schema, and require
aggregate equality before metrics. Reject zero/negative/out-of-plan counts and
add hostiles proving a 1-sample or differently selected guard cannot satisfy the
declared run. A semantic change to the control rather than enforcement of the
existing declaration returns to the owner.

### P2 — the manifest is exclusively named but not atomically published as a complete file

`_write_json_exclusive()` opens the final pathname with `O_CREAT|O_EXCL` and then
writes/fsyncs its bytes (`fl_v3/scripts/t5_attack_eval.py:347-358`). A reader
treats pathname existence as publication and immediately parses it
(`:361-368,430-450`). A concurrent task can therefore observe the final manifest
after the creating `open()` but before complete JSON; a creator failure can leave
a permanently partial final file. This fails closed rather than mixing identity,
but it does not meet the requested atomic exact-manifest contract, and the atomic
claim in `fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md:1172-1179` is too strong.

Required remediation: fully write/fsync a private same-directory temporary,
publish completed bytes with a no-replace atomic primitive, fsync the directory,
and exact-match only a complete winner on a lost race. Add interleaved reader/
creator and partial-writer/crash hostiles.

### P2 — a safe lexical run ID can still escape through a symlinked run directory

The regex rejects slashes and lexical `..` traversal
(`fl_v3/scripts/t5_attack_eval.py:42,178-188`), but `_run_root()` simply joins the
output directory and run ID (`:337-338`). Initialization uses
`os.makedirs(..., exist_ok=True)` and all later opens follow that directory
(`:430-446`). A pre-existing `OUTPUT_DIR/RUN_ID` symlink redirects the manifest
and all artifacts outside `OUTPUT_DIR`; no `lstat`, no-symlink, containment, or
dirfd-relative guard prevents it.

Required remediation: create/validate the run directory without following
symlinks, prove resolved containment under the declared root, and operate relative
to that validated directory, including poison-only readers. Add symlink-root and
unsafe/missing-run-ID hostiles before output.

### P3 — the hostile suite and handoff still overstate exact R4 branch coverage

The new tests cover useful branches: viz-cond4 and invalid task tuples, canonical
cond4/old-count/renamed extras, one manifest poison-checksum and shard-count drift,
one stale stealth selected checksum, one guard subset drift, one guard wrong type,
helper-level exclusive create, and raw/EMA ordering
(`fl_v3/tests/test_s07_b_integration.py:591-725,797-852,935-1160`). They do not
cover the P1/P2 paths above, partial/concurrent publication, a symlinked root,
missing/unsafe run IDs or subset, or field-wise sibling run/manifest/file/resolved/
policy/runtime drift. The three sibling mutations at `:1124-1138` do not support
the broad “stale/mixed stealth/guard poison, subset and type fields” wording in
`HANDOFF.md:1212-1219`. No authored test was executed.

### No P0 finding

No model/head/NMS/metric/protocol, split/data ownership, canonical/collab/fl_v2,
RUN_REQUEST/RESULTS, or historical-result change occurred. No pytest, Torch/data/
model runtime, Slurm/GPU action, merge, push, upload, or publication occurred.

## Identity, startup, exact prefix, topology, and ownership

- Session: `S07-B-R5`.
- BASE/WORKER SHA: `464281defc8c30f3099aa5e5e827fc907049255b`.
- REMEDIATION_BASE: `098cfded362ec276d3e697e9150cd7f05de3e238`.
- PRIOR_REVIEW_SHA: `a1452e095ee88a0570580a612f31108aa4b9db30`.
- Sole delivery branch: `codex/s07-b-r5-integrated-cl-stack-review`.
- APPROVED_COMPUTE: none.

Startup was clean detached at exact WORKER_SHA with empty branch/status and root
`/home/gaohui/.codex/worktrees/c79d/fl_weather_project` before the sole authorized
branch creation. No worktree management or other branch/external action occurred.

Prior review materialization was ancestry-free and exact: `a1452e0` has parent
REMEDIATION_BASE and only changes this REVIEW; blob
`e8f3a818cfc892b1e2a136c7c4edaf525b898bf1`, size 94,127, SHA-256
`f10e19a51502547be1a24658d7466b3fdef1820bef3c84ca1552f18f1ca65777`.
It is not a WORKER_SHA ancestor. Before this appendix, the working review exactly
reproduced those bytes.

The worker snapshot's canonical files stop at O-032. O-033 through O-043 were
read from read-only S00 ledger
`codex/s00-orchestra-ledger@5fb3be722f74a337cb6f000222f4b38f392e392a`;
O-044 additionally records this exact R5 launch. Nothing canonical was merged.

The remediation is the linear chain:

    098cfded362ec276d3e697e9150cd7f05de3e238
      -> efe9e7d46df3ef9feec627cf205dc197559886f7
      -> 464281defc8c30f3099aa5e5e827fc907049255b

Its exact diff is 666 insertions / 76 deletions over only
`fl_v3/scripts/t5_attack_eval.py`, `fl_v3/tests/test_s07_b_integration.py`, and
`fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md`. RUN_REQUEST remains blob
`0ef42edc0d69f3805dc960fafa06d6b9edba438d`; RESULTS remains blob
`f3ac930d87cbcf3a9ce056ad34af2153c2890ddf`.

## R4 and retained closure matrix

| Requirement / prior finding | R5 conclusion | Evidence / boundary |
|---|---|---|
| R4 stale stealth sibling | **CLOSED STATICALLY** | Exact schema plus run/manifest/current poison/whole-subset equality at `t5_attack_eval.py:607-634`; aggregate validates before metrics. |
| R4 stale guard sibling | **PARTIAL — P1** | Current checkpoint and whole subset bind at `:637-670`; the `guard_samples`-derived subset does not. |
| R4 viz-cond4 bypass | **CLOSED STATICALLY** | Non-shard rejection precedes viz at `:189-205`; hostile at test `:826-843`. |
| R4 canonical shard path set | **CLOSED STATICALLY** | Exact set equality at `:485-504` rejects missing/cond4/old-count/renamed/alias-prefixed extras; uniqueness/coverage remains at `:505-600`. |
| Run identity/task plan | **PARTIAL — P1/P2** | Run/checkpoint/whole-subset/clean checksum/shard count bind; guard selection is absent and publication is not atomic-complete. |
| Exclusive final writes | **CLOSED STATICALLY** | `O_EXCL` at `:347-358`, exclusive stealth/viz directories at `:971-975,1104-1108`, and exclusive result writes. No concurrency ran. |
| Run-root containment | **OPEN — P2** | Lexical name is safe; directory symlinks are followed. |
| R3 mandatory clean/checksum/occlusion/fan-in | **RETAINED STATICALLY** | Exact selected clean, boolean occlusion, identity, index/row/target coverage and full/cond4 schemas remain. |
| Raw/EMA ordering | **RETAINED STATICALLY/AUTHORED** | `_load_model` applies EMA before return/checksum (`:295-308`); raw/EMA full-shard test exists at test `:591-725`. |
| R4 hostile completeness | **PARTIAL — P3** | Several requested hostiles exist, but race/path/guard-policy and field-wise sibling cases do not. |

Earlier static closures for official evaluation, six-task migration, dependency
structure, PID fallback, disabled-modality augmentation, and mode/depth construction
retain only their recorded limits. Historical failures are unchanged; full
trainval `t1.v2` remains absent.

## Static checks actually run and explicit NOT RUN

Only safe static/read-only checks preceded this review edit:

1. exact startup and sole delivery branch: PASS;
2. prior parent/sole path/non-ancestry/blob/size/SHA-256/prefix: PASS;
3. read-only decisions through O-043 and R5 launch record: PASS;
4. remediation chain, exact ownership, unchanged request/results blobs and
   `git diff --check`: PASS;
5. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`: PASS;
6. stdlib `ast.parse` of two changed Python files and JSON parse of
   `fl_v3/configs/t5_attack.json`: `AST_OK=2`, `JSON_OK=1`;
7. manual/AST-level task/manifest/artifact/aggregate/hostile tracing: findings above.

Explicitly **NOT RUN / NO IMPLIED PASS**: pytest or any hostile; `py_compile` or
the static launcher itself; Torch/spconv/cumm import/native/JIT/build attestation;
checkpoint parse/raw/EMA load; cache/manifest/data, directory/ZIP, workers; model/
optimizer/scaler/scheduler/EMA construction; C/L/F, fp16, concurrency, rollback,
memory or NMS; T5 tasks/controls/devkit; Slurm/GPU; full cache, 100/1000 steps,
profile, mAP/NDS, DDP, matrix, seed, retry/rerun, FL, attack/defense, upload or
publication.

## Interpretation, residual risk, and O-009 boundary

Allowed: R4's direct stale checkpoint/subset sibling seam, viz-cond4 bypass, and
canonical shard-path defect are closed statically. Exact run-scoped checkpoint/
whole-subset identity and exclusive final result claiming are present.

Forbidden: the guard is not bound to its declared selected slice, the manifest is
not atomically complete at publication, the run root is not symlink-contained, and
no hostile ran. This is not runtime, production, full-data, metric, fusion-gain,
FL, attack/defense, generalization, or publication evidence.

No O-009 request should be prepared or submitted. First bind the guard slice,
make manifest publication atomic-complete, close root containment, add focused
hostiles, correct handoff wording, and obtain another exact-SHA review. A future
review PASS would permit only preparation of the prior bounded one-node/one-GH200/
<=60-minute proposal for S00 audit, not execution or scientific readiness.

## Final verdict

**CHANGES-REQUESTED for S07-B at
`464281defc8c30f3099aa5e5e827fc907049255b`.**

R4 checkpoint/subset sibling binding, viz task matrix, and canonical shard set
close statically. The final gate can still accept a favorable guard over an unbound
smaller sample prefix, and the promised atomic manifest/path-contained run root are
not implemented. Remediation and another independent review are required before
code-level PASS or any O-009 request.

---

# S07-B-R6 独立复审 — O-045 guard/atomic-publication/containment remediation

## Findings（按严重性排序）

### P2 — manifest 成功发布后的通配 cleanup 会删除仍在写入的另一发布者 private temp，违反 publisher non-interference 契约

private temp 名包含当前 PID 和随机 token，且自身以 mode `0600`、`O_EXCL|O_NOFOLLOW`
创建；但是成功 `link()` 最终名并 `fsync` 目录后，publisher 会枚举整个 run 目录，并删除
所有匹配 `.t5_run_manifest.json.*.tmp` 的名字，而不是只删除自己的 temp，也没有证明其他
temp 已 abandoned（`fl_v3/scripts/t5_attack_eval.py:474-510`）。因此另一 publisher 即使仍持有
打开且正在写入的 FD，其 private 名也会被先完成的 publisher 删除。后者最终执行 `link()`
时进入 `FileNotFoundError`；只要 final 已存在，代码就将它解释为 lost race
（`fl_v3/scripts/t5_attack_eval.py:494-501`）。final 不会被覆盖，随后 manifest exact-match 仍会
拒绝不同 winner；但 cleanup 本身已经驱动并删除了另一活跃 publisher 的命名状态，未满足
kickoff 明确要求的“crash/temp cleanup cannot delete/overwrite another publisher”。

本 review 用 stdlib AST 只提取 exact production helpers，保持另一 temp FD 打开并写入后调用
真实 `_atomic_publish_json_at()`；结果为：

```text
PUBLISHED True
OTHER_ACTIVE_FD_OPEN True
OTHER_TEMP_NAME_SURVIVES False
STDLIB_ACTIVE_PUBLISHER_RACE_OK
```

这不是 partial-final 或 identity 混入：complete final、no-replace、winner equality 的主体设计
成立；它是并发 publisher non-interference/cleanup ownership 缺口。修复应让每个 publisher
只清理自己创建的 temp；若要回收 abandoned temp，必须使用不会误删 live publisher 的独立、
可证明 ownership/lease 机制，或将无害 orphan 保留给显式维护流程。hostile 必须保持另一个
publisher 的 FD 与名字处于 live 状态，证明其名字不会被第三方 cleanup 删除，并覆盖 exact
winner 与 different winner 两种后续结果。

### P3 — authored hostile suite 尚未命中全部 R6 必需 race/containment/count 分支

1. `test_t5_manifest_publication_is_complete_noreplace_and_cleans_failures` 对 crash temp 的
   writer 已关闭 FD，然后才将其当 orphan 清理；第二次 publisher 也是在 final 已存在后串行
   调用 helper（`fl_v3/tests/test_s07_b_integration.py:1145-1171`）。它没有构造 live publisher，
   也没有让 `_bind_run_manifest()` 的实际 `published=False` exact-winner/different-winner 分支
   在并发窗口中执行（production 分支位于
   `fl_v3/scripts/t5_attack_eval.py:624-631`）。interleaved reader 只证明 link 前 final 不可见，
   没有证明 lost-race caller 的完整 exact-match 行为。
2. pathname hostile 只覆盖 final output-root symlink、run-root symlink、unsafe/missing run ID、
   partial final 与 stale sibling root（`fl_v3/tests/test_s07_b_integration.py:1175-1227`）。没有
   创建 symlinked `ablation`/`stealth_det_eval`/`viz` subdirectory，也没有创建 symlinked
   manifest/shard/stealth/guard/aggregate artifact name，故未实际到达
   `_open_subdirectory()`、`_reserve_subdirectory()`、`_load_required_json_at()` 和
   `_write_json_exclusive_at()` 的相应 no-follow 拒绝分支
   (`fl_v3/scripts/t5_attack_eval.py:400-466`)。这些生产分支静态上是 dirfd-relative 且
   fail-closed，但当前 authored evidence 不支持“每个 containment hostile 已覆盖”的门槛。
3. guard tests 已覆盖 default 40 对 1、zero、greater-than-available、old v1、sample reorder 与
   target drift（`fl_v3/tests/test_s07_b_integration.py:1077-1122,1280-1318`），但没有直接改变
   `n_invariance_checks` 或 `total` 来命中两个 exact count-mismatch 拒绝分支
   (`fl_v3/scripts/t5_attack_eval.py:865-868`)。所有 selection fixture 还是一 sample 对一 target，
   没有以交错的多-target samples 证明 selected targets 保持原 frozen target order。

这些 gap 没有证明相应生产分支本身错误，因此定为 P3；但它们是 kickoff 明列的 adversarial
gate evidence，不能用 test 名称或未执行的 broad prose 替代。由于上方 P2 已阻断 code-level
PASS，本轮不得以未来 pytest 来绕过实现修复。

### 无 P0/P1 finding

本 remediation 没有修改 split/data ownership、model/head/NMS/metric/protocol、canonical/
collab/fl_v2、RUN_REQUEST/RESULTS 或历史 job 结果；也没有发生 pytest、Torch/data/model、
Slurm/GPU、merge、push、upload 或 publication。guard identity 与 manifest/run containment
没有发现能让不同 checkpoint/subset/selection 直接变绿的 P1；确认的问题是 P2 并发 cleanup
ownership 和 P3 hostile reachability。

## Review identity、startup、prior prefix 与 exact topology

- Session: `S07-B-R6`。
- BASE/WORKER SHA: `8cdeceb4e72042874f6ab5aa8a39e84ab67bf934`。
- REMEDIATION_BASE: `464281defc8c30f3099aa5e5e827fc907049255b`。
- IMPLEMENTATION_SHA: `fcf36dd159bf881df300805e0934ce0ca30ea237`。
- PRIOR_REVIEW_SHA: `2176e8d2e8185af26f27d67a45838528e4390543`。
- Sole delivery branch: `codex/s07-b-r6-integrated-cl-stack-review`。
- APPROVED_COMPUTE: none。

Startup 在创建唯一授权 branch 前精确满足门禁：repository root
`/home/gaohui/.codex/worktrees/d133/fl_weather_project`，detached HEAD exact 等于
WORKER_SHA，branch name empty，status clean。未进行 worktree add/move/remove/prune、
unrelated branch switch/create/delete 或外部动作。

Prior review 独立核验并 ancestry-free materialize：

- `2176e8d` parent 精确为 REMEDIATION_BASE；
- commit 唯一 changed path 为本 `S07/REVIEW.md`；
- exact blob `78c05b9a1c060c82f3bff59ba2159c4675a3c9a0`；
- exact size `105234` bytes；
- exact SHA-256
  `30034cc8f649a31d3ad51fc52d1055bfc48cca8449f41fe9c3e5c5daf6d70dd2`；
- prior reviewer commit 未进入 WORKER_SHA ancestry。

worker tree 原有 22,715-byte S07-A prefix与 prior blob前缀完全相同；追加本 R6 前，working
`REVIEW.md` 再次精确得到上述 blob/size/SHA-256。没有 merge/cherry-pick reviewer ancestry。

Implementation/handoff 是精确线性链：

```text
464281defc8c30f3099aa5e5e827fc907049255b
  -> fcf36dd159bf881df300805e0934ce0ca30ea237
  -> 8cdeceb4e72042874f6ab5aa8a39e84ab67bf934
```

`fcf36dd` parent 与 `8cdeceb` parent 均精确；REMEDIATION_BASE..WORKER_SHA 仅三路径、
629 insertions / 122 deletions：`t5_attack_eval.py`、`test_s07_b_integration.py`、
`S07/HANDOFF.md`。`RUN_REQUEST.md` blob 在两端均为
`0ef42edc0d69f3805dc960fafa06d6b9edba438d`，`RESULTS.md` 均为
`f3ac930d87cbcf3a9ce056ad34af2153c2890ddf`。当前 source/test SHA-256 精确为 handoff
声明的 `9aad253918ef7fc47239a7d6570778ba2829302324e557ec4ee0af631ebead52` /
`f72ed4f42c0e322e36f8f00d053a7d8476ca43066fb2b72516a6c8462b75547e`。

worker snapshot 的 canonical ledger 非当前版本；本 review 从只读
`codex/s00-orchestra-ledger@272b4a52d0ca0e068dd76946208b0d577cabcd9f` 完整核验
ORCHESTRA/SESSIONS/KICKOFFS 至 O-046，没有 merge 或写回该 ref。

## Guard scientific semantics 与 immutable identity trace

R5 的 guard P1 在静态实现层关闭，且未发现科学语义漂移：

- CLI default 仍是 40 (`t5_attack_eval.py:1379`)；`_guard_selection()` 要求 exact positive
  integer，拒绝超过 available，按 Python locale-independent lexical order 取 unique sample
  token prefix，并按 subset 原 targets 顺序过滤 selected targets
  (`t5_attack_eval.py:542-562`)。
- run schema/guard schema 升级为 `s07b.t5.run.v2` / `s07b.t5.guards.v2`
  (`t5_attack_eval.py:41-43`)；plan 同时携带 declared count、exact sample list、exact target
  list 与对前三者 canonical bytes 的 SHA-256 (`t5_attack_eval.py:554-570`)。
- 每个 invocation 都从当前 frozen subset 重建同一 selection 并参与 complete manifest exact
  equality；one-sample、shard-count、checkpoint/subset/target/order drift 在 artifact/data work
  前不能复用现有 run (`t5_attack_eval.py:574-632`)。
- guard producer 在写 output 前再次 exact-match manifest selection，并要求每个 selected
  sample/target 都实际被计数 (`t5_attack_eval.py:1224-1271`)；aggregate 在读取 raw metric 前
  要求 guard v2 exact schema、checkpoint/run/manifest/whole-subset/selection equality、
  invariance count等于 declared sample count、recall total等于 exact selected target count
  (`t5_attack_eval.py:829-869`)。

因此 default/declaration 40、sorted unique prefix、original target order、selection hash 与
aggregate denominator 的 production semantics 可静态接受；本 review 不把相应 tests 未运行
解释为 runtime PASS。

## Manifest publication、dirfd containment 与 stale-root trace

除 finding 的 cleanup ownership 外，R5 atomicity/containment 主体静态成立：

- private temp 以 random direct-child name、mode 0600、`O_EXCL|O_NOFOLLOW` 创建，complete
  write、file `fsync` 后才通过 same-directory hard link claim final；成功 link 后 directory
  `fsync` (`t5_attack_eval.py:469-502`)。reader 只能看到无 final 或 complete hard-linked inode；
  partial final JSON 在 `_load_required_json_at()` 解码处失败且不会被替换
  (`t5_attack_eval.py:451-466,603-623`)。
- lost `FileExistsError` 不覆盖 winner；caller 读取 complete final，并仅在整个 manifest 与
  current expected exact 相等时继续，不同 winner失败 (`t5_attack_eval.py:492-501,624-632`)。
- final output root 用 `lstat` 拒绝 symlink，root/run 通过 `O_DIRECTORY|O_NOFOLLOW` 打开，
  run 的 real parent 必须等于 held root；run、ablation、stealth-eval、viz descriptors 在所有
  artifact/devkit/viz I/O 期间保持打开 (`t5_attack_eval.py:356-427,917-1023,
  1156-1202,1210-1272,1281-1346`)。
- manifest/shard/stealth/guard/aggregate JSON 使用 direct-child basename检查和 held dirfd
  `O_NOFOLLOW` read/exclusive write；canonical shard set、siblings与 aggregate output不再通过
  mutable run pathname (`t5_attack_eval.py:430-466,665-869,1029-1139`)。external devkit/viz
  只收到 held `/proc/self/fd/<fd>` alias，fd 在同步 writer完成后才关闭。
- clean+poison initializer 在无 complete manifest 时只允许 private manifest temp；任何其他
  run artifact使初始化失败。poison-only invocation以 `create=False` 打开已有 direct child，
  缺 run/complete manifest即失败，不会初始化 state；所有 sibling/result writes保持 `O_EXCL`
  或 exclusive directory reservation (`t5_attack_eval.py:601-635,1171-1202,1314-1346`)。

所以 symlink substitution不能把 held descriptor重定向到另一个 pathname；partial final、
different identity、mixed stale root 与 overwrite 都 fail closed。P2 仅否定“cleanup不会删除
另一个 publisher temp”的更强结论，不否定 complete final/no-replace/exact-match本身。

## R4/R3 与更早 closure matrix

| Requirement | R6 conclusion | Static evidence / remaining boundary |
|---|---|---|
| R5 guard count/sample/target identity | **CLOSED STATICALLY** | 上述 run v2/guard v2/aggregate exact-match；P3仅为未命中 count branch。 |
| R5 complete-before-publish/no-replace | **PARTIAL — P2 cleanup** | final bytes/link/fsync/winner equality成立；live temp non-interference失败。 |
| R5 output/run containment | **CLOSED STATICALLY; HOSTILE GAP P3** | held no-follow dirfds及direct-child I/O成立；subdir/artifact symlink tests缺失。 |
| R4 stale stealth/guard siblings | **RETAINED STATICALLY** | checkpoint/run/manifest/subset schema exact-match；guards再绑定selection。 |
| R4 viz-cond4 | **RETAINED STATICALLY** | non-shard cond4在任何 task分支前拒绝 (`t5_attack_eval.py:180-213`)。 |
| R4 canonical shard set | **RETAINED STATICALLY** | exact canonical filename set及identity/index/target/row checks (`:665-793`)。 |
| R3 mandatory clean/checksum/occlusion/fan-in | **RETAINED STATICALLY** | full shard需clean；actual selected clean checksum在data前绑定；boolean occlusion、unique rows/exact targets保留 (`:917-1021,665-793`)。 |
| Raw/EMA order | **RETAINED STATICALLY/AUTHORED** | EMA state在model return/checksum前应用 (`:297-310`)；producer均先load model再checksum。 |
| Strict official eval/six-task/dependency/PID/mode augmentation | **UNCHANGED FROM PRIOR STATIC CLOSURES** | 本 diff未修改对应 model/head/NMS/eval/runtime/data trees；actual gates仍 NOT RUN。 |

没有 model/head/NMS/metric/protocol、split、template、DDP 或 scientific field drift。
历史 failures/passes 保持原边界；full trainval `t1.v2` 仍不存在。

## Checks actually run 与 explicit NOT RUN

本 R6 只执行 kickoff 允许的 read-only/static/stdlib checks：

1. startup root/HEAD/detached/status与唯一 branch：PASS；
2. prior review parent/sole path/non-ancestry/blob/size/SHA-256及 exact materialized prefix：PASS；
3. canonical ledger through O-046、implementation/handoff parents、three-path ownership、
   unchanged request/results blobs与source/test hashes：PASS；
4. `git diff --check 464281d..8cdeceb`：PASS；
5. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`：PASS；
6. stdlib `ast.parse` 两个 changed Python files与 JSON parse当前 T5 config：
   `AST_OK=2`, `JSON_OK=1`；
7. 完整 source/test逐行 trace guard、manifest、dirfd、stale-root、R3/R4 closures与hostile
   reachability：产生上述 findings；
8. stdlib AST-extracted exact helper执行：complete publish成功，并独立复现 live publisher
   temp name被cleanup删除。

明确 **NOT RUN / NO IMPLIED PASS**：

- pytest及任何 authored hostile；
- `py_compile`、完整 static launcher或production package import；
- Torch/spconv/cumm import、native/JIT/build/source identity与operator dispatch；
- checkpoint parse、raw/EMA load、model/optimizer/scaler/scheduler/EMA construction/rollback；
- cache/manifest/data、directory/ZIP payload、fork/spawn/persistent workers；
- C/L/F forward/backward、fp16 S04 option-A、same-instance concurrency、memory/NMS；
- T5 shard/aggregate/stealth/guards/viz、five-condition/occlusion、official devkit；
- Slurm/srun/GPU、full cache、100/1000 steps、profile、mAP/NDS、DDP、matrix、seed、
  retry/rerun、FL、attack/defense、upload或publication。

## Interpretation、residual risk 与 O-009 boundary

允许解释：R5 guard scientific semantics 已被 exact run/guard identity静态冻结；manifest final
具备 complete-before-name、no-replace、directory durability与winner exact-match主体；所有当前
T5 artifact/devkit/viz路径已迁移到held no-follow dirfds；R3/R4直接scientific-identity closure
保持。

禁止解释：不得称并发 temp cleanup具备publisher non-interference，不得称全部必需 race/
subdir/artifact/count hostiles已覆盖或PASS；不得称S07-B runtime/production/full-data/checkpoint/
performance/scientific ready，亦不得声称mAP/NDS、fusion gain、FL、attack/defense、
generalization或publication evidence。

由于存在 atomicity P2，当前不得准备或提交 O-009 request。先修复 cleanup ownership，并补齐
live publisher + `_bind_run_manifest` exact/different winner、subdir/artifact symlink、guard count/
order hostiles，再从 exact new SHA 做独立复审。未来 code-level PASS 也只允许为 S00 审计准备
一个 separate bounded O-009 proposal；它不是执行授权，更不是 runtime/production/full-data/
scientific PASS。

## Final verdict

**CHANGES-REQUESTED for S07-B at
`8cdeceb4e72042874f6ab5aa8a39e84ab67bf934`.**

Guard-plan equality、complete final/no-replace publication、dirfd containment及R3/R4 identities
在主体静态路径上成立；但 successful publisher会删除另一live publisher private temp，且必需
hostile suite尚未覆盖真实lost-race caller、subdir/artifact symlinks和count/order分支。修复并
重新独立review前，没有code-level PASS、O-009 proposal、runtime/production或scientific授权。

# S07-B-R7 独立复审 — O-047 publisher ownership / hostile reachability remediation

## Findings（按严重性排序）

### P3 — authored “real lost race” 测试仍通过替换整个 publisher helper 伪造 `False`，没有让两个真实 publisher 竞争 final hard link

R7 实现已把通配 cleanup 和 `link()` 的 `FileNotFoundError` lost-race shortcut
删除：`_atomic_publish_json_at()` 在 success、`FileExistsError` loser 和异常三类路径都只在
`finally` 中 unlink 自己生成的 `temp_name`，同时保留 complete write/file-fsync、
same-directory hard-link no-replace 和 directory-fsync
(`fl_v3/scripts/t5_attack_eval.py:469-503`)。这一生产修复本身未发现 P0-P2 缺口。

但是新测试 `test_t5_bind_manifest_real_lost_race_accepts_only_exact_complete_winner`
保存 `original_publish`，随后用 monkeypatch 整体替换
`module._atomic_publish_json_at`。替身只执行一次真实成功 publish，然后直接
`return False` (`fl_v3/tests/test_s07_b_integration.py:1178-1215`)。所以它确实达到
`_bind_run_manifest()` 的 `published=False` 与 exact/different equality 分支
(`fl_v3/scripts/t5_attack_eval.py:608-615`)，却没有第二个真实
`_atomic_publish_json_at()` 创建自己的 temp、与 winner 并发到达 `os.link()`、获得
真实 `FileExistsError` 并执行自己的 loser cleanup。

相邻的 ownership 测试手工 `os.open()` 一个名字类似 publisher temp 的文件，
并在另一个 helper 的 failure/success/lost 路径中确认该 open FD 仍可写、名字仍
存在 (`fl_v3/tests/test_s07_b_integration.py:1125-1174`)。这能发现原先的 glob
unlink 回归，但它仍不是另一个处在真实 publication 窗口内的 publisher。

本 review 用 stdlib AST 提取精确生产 helper，然后以两个真实
`_bind_run_manifest()` 线程在完成 private-temp write 后同步到同一 hard-link 窗口。
结果是 exact identity 两个 caller 都成功，different identity 恰好一个成功、一个在
winner exact-match 处拒绝，并且竞争期间两个真实 temp 同时存在、结束后都只被
各自清理：

```text
REAL_BIND_EXACT_RACE_OK 2 ['ok', 'ok']
REAL_BIND_DIFFERENT_RACE_OK 2 ['err', 'ok']
```

这说明当前生产代码的 race 语义静态/helper 层成立，但 reviewer 自建 harness
不能替代 O-047/O-048 要求的 authored hostile。修复只需改测试：使两个真实
`_bind_run_manifest()` 并发调用，可仅在执行完真实 `_write_all()` 后用 barrier
排程；不得替换 `_atomic_publish_json_at()`、伪造其返回值或绕过真实
`FileExistsError` loser 路径。对 exact/different winner 各做一例，并在 barrier
窗口内证明两个 publisher-owned temp 都存在且互不删除。

这是 authored-evidence reachability 缺口，不是已确认的生产 atomicity/scientific
错误，因此定为 P3；但它正是 R7 kickoff 明确的 no-mock-bypass gate，本轮不能
给 code-level PASS。

### 无 P0/P1/P2 finding

R7 diff 没有修改 model/head/NMS/metric/protocol/split、candidate configs、
`RUN_REQUEST.md`、`RESULTS.md`、canonical/collab/fl_v2 或历史 job 记录。上述
P3 之外，publisher cleanup、held-dirfd containment、guard identity/count/order 及 R3/R4
scientific closures 未发现新回归。

## Review identity、prior prefix、topology 与 ownership

- Session: `S07-B-R7`。
- BASE/WORKER SHA: `35a0bdca8af61172722428261024d034ecc97a50`。
- REMEDIATION_BASE: `8cdeceb4e72042874f6ab5aa8a39e84ab67bf934`。
- IMPLEMENTATION_SHA: `8a7b60b2dd27b1c7ba72e53ddbe67b278ea2f512`。
- PRIOR_REVIEW_SHA: `ef01d1cad73021acb87b01874726b83da6470e84`。
- Sole delivery branch: `codex/s07-b-r7-integrated-cl-stack-review`。
- APPROVED_COMPUTE: none。

Startup 在创建唯一授权 branch 前精确为 repository root
`/home/gaohui/.codex/worktrees/c556/fl_weather_project`、clean
`detached@35a0bdca8af61172722428261024d034ecc97a50`、empty branch name。未进行
worktree add/move/remove/prune、未创建其他 branch、未 merge/push/upload。

Prior R6 review 已独立核验并 ancestry-free materialize：

- `ef01d1c` parent 精确为 `8cdeceb4e72042874f6ab5aa8a39e84ab67bf934`；
- commit 唯一 changed path 为本 `S07/REVIEW.md`；
- exact blob `b7a6450ec618dc5a3f40503d12a3605ed4e7c64d`；
- exact size `121397` bytes；
- exact SHA-256
  `14dd6749ec63fd473e1818109cd42553127e5e6f10daa9d9407f9c6f132190e1`；
- prior reviewer commit 未进入 WORKER SHA ancestry。

R7 以上述 exact bytes 作为本段前缀，没有 merge/cherry-pick reviewer ancestry。
Implementation/handoff 是精确线性链：

```text
8cdeceb4e72042874f6ab5aa8a39e84ab67bf934
  -> 8a7b60b2dd27b1c7ba72e53ddbe67b278ea2f512
  -> 35a0bdca8af61172722428261024d034ecc97a50
```

`8cdeceb..35a0bdc` 只有三个授权路径：
`fl_v3/scripts/t5_attack_eval.py`、
`fl_v3/tests/test_s07_b_integration.py` 和
`fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md`，共264 insertions / 23
deletions。`RUN_REQUEST.md` blob 两端均为
`0ef42edc0d69f3805dc960fafa06d6b9edba438d`，`RESULTS.md` 两端均为
`f3ac930d87cbcf3a9ce056ad34af2153c2890ddf`。当前 source/test SHA-256 与
HANDOFF 一致：

- `t5_attack_eval.py`:
  `19bcc9ccbea89ba363d6a6bee47449448339b1b519f792e6fdfcf99e3d08034d`；
- `test_s07_b_integration.py`:
  `3d32ed5bbde9d259ad392133d778f6686253230ca9cac93d407c61060b6f08d5`。

Canonical ledger 从只读
`codex/s00-orchestra-ledger@f02d8b5ec345719e28c952c965b7c4b4e5063fd2`
完整核验至 O-048；未 merge 或写回该 ref。

## R6 / R5 / retained closure matrix

| Requirement | R7 conclusion | Static/authored evidence and remaining boundary |
|---|---|---|
| R6 successful/lost/error cleanup ownership | **CLOSED STATICALLY** | `t5_attack_eval.py:469-503` 无 glob，无 `link()`-`FileNotFoundError` shortcut，只 unlink 自己 `temp_name`。 |
| R6 exact/different lost-race caller | **PRODUCTION CLOSED; AUTHORED GAP P3** | reviewer 的两线程精确 helper 证明真实 race 成立；authored test 替换整个 publisher 并伪造 `False` (`test_s07_b_integration.py:1178-1215`)。 |
| R6 live-publisher non-interference | **PRODUCTION CLOSED; AUTHORED GAP P3** | 无 wildcard cleanup；authored fixture 仅手工 open temp，未同时运行两个 publisher (`:1125-1174`)。 |
| R6 subdirectory symlinks | **CLOSED AUTHORED, NOT EXECUTED** | `ablation` 达到 `_open_subdirectory()`，`stealth_det_eval`/`viz` 达到 `_reserve_subdirectory()`，均是 held dirfd/no-follow (`:1277-1298`)。 |
| R6 artifact symlinks | **CLOSED AUTHORED, NOT EXECUTED** | manifest/shard/stealth/guard/aggregate 名称分别到达 `_bind_run_manifest()`、bound loaders 或 `_write_json_exclusive_at()`；outside bytes 不变 (`:1300-1341`)。 |
| R6 guard count mismatches | **CLOSED AUTHORED, NOT EXECUTED** | 分别改变 `n_invariance_checks` 和 `total`，命中两个 production rejection (`:1433-1442`)。 |
| R6 multi-sample/target frozen order | **CLOSED AUTHORED, NOT EXECUTED** | 3 samples/6 interleaved targets，选中2 samples 后保留原冻结目标顺序 (`:1446-1466`)。 |
| R5 guard run-v2 identity/count | **RETAINED STATICALLY** | 本 production diff 只删 publisher cleanup；declared count/sample/target/selection hash 与 aggregate exact counts 未变。 |
| R5 complete final/no-replace/durability | **RETAINED STATICALLY** | complete private write+fsync、hard-link no-replace、dir fsync、exact winner 均保留。 |
| R5 stale root/contained run | **RETAINED STATICALLY** | held output/run/subdir descriptors、direct-child artifact I/O 与 stale-root rejection未变。 |
| R4 siblings/shards/viz-cond4 | **RETAINED STATICALLY** | checkpoint/run/manifest/subset identities、canonical shard filename set、viz-cond4 fail-closed 未改。 |
| R3 clean/checksum/occlusion/fan-in/raw-EMA | **RETAINED STATICALLY** | mandatory clean identity/checksum/boolean occlusion/unique rows/exact targets、EMA-before-checksum ordering未改。 |
| Earlier official eval/six-task/dependency/PID/mode closures | **UNCHANGED FROM PRIOR STATIC CLOSURES** | 本 remediation 没有修改对应生产树；actual runtime gates 仍 NOT RUN。 |

## Generated-pyc cleanup audit

Git 可证明的 R7 范围是：`8a7b60b` 只修改 source/test 两个 tracked
files，`35a0bdc` 只修改 HANDOFF；没有 tracked deletion、broad cleanup 或其他路径
改动。当前物理检查证明下列两个 HANDOFF 声明的 generated files 均不存在：

```text
fl_v3/scripts/__pycache__/t5_attack_eval.cpython-39.pyc
fl_v3/tests/__pycache__/test_s07_b_integration.cpython-39.pyc
```

但它们是 ignored/untracked runtime files；Git 无法回溯证明物理删除操作当时“只删了这
两个”。本 review 能独立接受的精确边界是：当前两个 exact path 不存在、Git
范围没有 broad tracked deletion；物理历史的更强声明依赖 S00/O-048 的同步观测，
不由本 review 倒推或扩大。

## Checks actually run 与 explicit NOT RUN

本 R7 只执行 kickoff 允许的 read-only/static/stdlib helper checks：

1. startup root/HEAD/detached/status 与唯一 review branch：PASS；
2. prior review parent/sole path/non-ancestry/blob/size/SHA-256 与 exact materialized prefix：PASS；
3. canonical ledger through O-048、implementation/handoff parents、three-path ownership、
   unchanged request/results blobs 与 source/test hashes：PASS；
4. `git diff --check 8cdeceb..35a0bdc`：PASS；
5. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`：PASS；
6. stdlib `ast.parse` 当前 source/test：`AST_OK=2`；当前五个 S07-B JSON
   template parse：`JSON_OK=5`；
7. 完整逐行 tracing publisher、manifest/bound loaders、task producer/aggregate、guard、
   subdir/artifact no-follow 路径与 authored hostiles：产生上述唯一 finding；
8. stdlib AST-extracted exact production helper 的两线程真实 race：exact winner
   2/2 caller 成功，different winner 1/2 成功、1/2 exact-identity 拒绝，竞争期间
   2 temps、结束后0 temps；
9. 首个 reviewer race harness 尝试在调用 `_open_run_directory()` 前直接替换
   `os.link`，因生产 `os.supports_dir_fd` fail-closed guard 而在创建 run child 前拒绝。
   这是 preserved harness-setup negative，不是代码失败；后续 helper 使用真实
   `_open_run_directory()` 先创建 held run，再仅对 hard-link window 排程。
10. exact generated-pyc absence、commit stat 和无 tracked deletion：PASS，限制如上。

明确 **NOT RUN / NO IMPLIED PASS**：

- pytest 或任何 authored hostile；
- `py_compile`、完整 static launcher 或 production-package import；
- Torch/NumPy/spconv/cumm import、native/JIT/build/source identity 或 operator dispatch；
- checkpoint parse、raw/EMA load、model/optimizer/scaler/scheduler/EMA construction/rollback；
- cache/manifest/data、directory/ZIP payload、fork/spawn/persistent workers；
- C/L/F forward/backward、fp16 S04 option-A、same-instance concurrency、memory/NMS；
- T5 real shard/aggregate/stealth/guards/viz、five-condition/occlusion、official devkit；
- Slurm/srun/GPU、full cache、100/1000 steps、profile、mAP/NDS、DDP、matrix、seed、
  retry/rerun、FL、attack/defense、upload或publication。

## Interpretation、residual risk 与 O-009 boundary

允许解释：R6 wildcard cleanup 和 `FileNotFoundError` shortcut 已在当前生产源码
中删除；真实 helper-level exact/different race 与 publisher-owned temp 清理在本 review
的 stdlib harness 中成立；subdirectory/artifact symlink、guard count 与 interleaved
order 的 authored code 已到达对应 production helpers；R3-R5 identities 保留。

禁止解释：不得称 authored suite 已无 mock 地复现真实 publisher race，不得称
pytest/runtime/production/full-data/checkpoint/performance/scientific ready，不得声称 mAP/NDS、
fusion gain、FL、attack/defense、generalization 或 publication evidence。

由于 O-048 明列的 authored no-mock-bypass hostile 尚未满足，当前仍不应准备或
提交 O-009 request。只需在现有 test ownership 内修复该并发 fixture，保持
production 源码和科学语义不变，然后从 exact new SHA 再做一次独立静态复审。
未来 code-level PASS 也只允许 S00 准备 separate bounded O-009 proposal；它不是
自动执行授权，更不是 runtime/production/full-data/scientific PASS。

## Final verdict

**CHANGES-REQUESTED for S07-B at
`35a0bdca8af61172722428261024d034ecc97a50`.**

Publisher cleanup 生产修复本身在静态与 reviewer 真实 helper race 中成立，且
symlink/count/order 测试已到达相应 production helpers；但 exact/different lost-race
authored test 仍通过替换整个 publisher 与伪造 `False` 来越过真实 loser，没有满足
R7 明确的 no-mock-bypass gate。修复该唯一 authored-evidence P3 并再审前，没有
code-level PASS、O-009 proposal 或任何 runtime/scientific 授权。

# S07-B-R8 独立复审 — O-049 real-caller publisher-race fixture

## Findings（按严重性排序）

### 无 P0/P1/P2/P3 finding

R8 候选把 R7 唯一 P3 的 mock-bypass fixture 替换为两个真实 concurrent
`_bind_run_manifest()` caller，未发现新的 correctness、atomicity、identity、scientific
semantics 或 evidence-reachability defect。

新测试的 exact 与 different 两例各创建一个 fresh direct-child run directory，随后各启动
两个线程；每个线程都调用真实 `_bind_run_manifest()`，后者依次到达真实
`_atomic_publish_json_at()`、随机 publisher-owned private temp、真实 `_write_all()`、file
`fsync`、真实 same-directory hard link、`FileExistsError` loser 与 exact whole-manifest
comparison (`fl_v3/tests/test_s07_b_integration.py:1179-1285`；
`fl_v3/scripts/t5_attack_eval.py:469-503,558-620`)。测试没有替换
`_atomic_publish_json_at()`，也没有伪造其 boolean return。

排程 wrapper 先调用并完成真实 `_write_all()`，从仍打开的 `/proc/self/fd/<fd>` 读取每个
真实 private temp basename；两个 caller 在 write barrier 后均确认 snapshot 中有两个不同
temp，且它们在 held run 下同时是 regular files。link barrier 仅在这个观察完成后释放。
`os.link` wrapper 调用原始 hard-link operation，不改变参数、返回或异常；它只记录真实
success 或重新抛出的真实 `FileExistsError`。因此两个 publisher 都已越过真实 private
write，两个 temp 在 link 前同时存在，而且 winner/loser 都由 production hard link 决定。

exact identity case 要求两个 bind caller 都成功并读取相同完整 manifest；different identity
case 要求恰好一个 caller 成功、另一个在 production winner equality check 以
`different identity` 拒绝。两例均要求真实 link outcomes 排序后为 `[False, True]`，并在
结束时枚举 run directory 证明没有 `.tmp` 残留。R8 的 stdlib AST/helper 独立执行还以
thread profiler 观察到精确调用计数：

```text
R8_AUTHORED_FIXTURE_AST_HELPER_PASS
REAL_PRODUCTION_CALL_COUNTS {"_atomic_publish_json_at": 4, "_bind_run_manifest": 4, "_write_all": 4}
```

该 helper 直接 AST 提取 authored test 与 exact production functions；只把测试的
`_script_module()` loader 换成 AST-extracted production namespace，以避免任何 Torch/pytest/
package import。它没有替换 publisher、bind、write 或 link 结果。四次计数对应 exact 与
different 各两个真实 caller；fixture 自身所有 two-live-temp、real link outcome、caller
result 与 zero-temp assertions 均通过。故 R7 唯一 authored no-mock-bypass P3 已关闭。

## Review identity、prior prefix、topology 与 ownership

- Session: `S07-B-R8`。
- BASE/WORKER SHA: `fdee4ba574587a9974ac6a188f2c011dc4730f75`。
- REMEDIATION_BASE: `35a0bdca8af61172722428261024d034ecc97a50`。
- TEST_SHA: `dd60326fd424d263ab2733fbb8353fb6a6cbb45a`。
- PRIOR_REVIEW_SHA: `e4fa439a5c09447bd8b413682772e81f9998f027`。
- Sole delivery branch: `codex/s07-b-r8-integrated-cl-stack-review`。
- APPROVED_COMPUTE: none。

Startup 在创建唯一授权 branch 前精确为 repository root
`/home/gaohui/.codex/worktrees/1459/fl_weather_project`、clean
`detached@fdee4ba574587a9974ac6a188f2c011dc4730f75`、empty branch name。未执行
worktree add/move/remove/prune，未创建其他 branch，未 merge/push/upload。

Prior R7 review 已独立核验并 ancestry-free materialize：

- `e4fa439` parent 精确为 `35a0bdca8af61172722428261024d034ecc97a50`；
- commit 唯一 changed path 为本 `S07/REVIEW.md`；
- exact blob `b27655cf7e0cec994aada87010eae0065c5746ce`；
- exact size `134348` bytes；
- exact SHA-256
  `28164c0f692523ee4920d516ba3030052be8380b2b4cc7d96de036935bfe6f6b`；
- prior reviewer commit 未进入 WORKER SHA ancestry。

R8 以上述 exact bytes 作为本段前缀，没有 merge/cherry-pick reviewer ancestry。
Implementation/handoff 是精确线性链：

```text
35a0bdca8af61172722428261024d034ecc97a50
  -> dd60326fd424d263ab2733fbb8353fb6a6cbb45a
  -> fdee4ba574587a9974ac6a188f2c011dc4730f75
```

`35a0bdc..fdee4ba` 只改两个授权路径：
`fl_v3/tests/test_s07_b_integration.py` 与
`fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md`，共170 insertions / 31
deletions。没有 production source/config/model/head/NMS/metric/protocol、canonical、collab、
`RUN_REQUEST.md` 或 `RESULTS.md` 变化。Production T5 blob 两端均为
`940b7eb002accc504eb18c796e448df849a65876`；`RUN_REQUEST.md` 两端均为
`0ef42edc0d69f3805dc960fafa06d6b9edba438d`，`RESULTS.md` 两端均为
`f3ac930d87cbcf3a9ce056ad34af2153c2890ddf`。

当前 committed SHA-256 与 HANDOFF 精确一致：

- `test_s07_b_integration.py`:
  `4349e4734c6161f0ae7bd6b6dc28450d8c9475c9b8fcc6a37462fd948f0ea551`；
- unchanged `t5_attack_eval.py`:
  `19bcc9ccbea89ba363d6a6bee47449448339b1b519f792e6fdfcf99e3d08034d`。

Canonical ledger 从只读
`codex/s00-orchestra-ledger@57be167b79e15fce47e689f710fa7d72367abb1a`
完整核验至 O-050；未 merge 或写回该 ref。HANDOFF 对唯一 test-only remediation、未执行
pytest/pycompile、production/RUN_REQUEST/RESULTS 不变、解释边界及 no-compute 状态的陈述
与 actual diff/commands 一致。

## R7 / R6 / R5 / R4 / R3 closure matrix

| Requirement | R8 conclusion | Static/authored evidence and remaining boundary |
|---|---|---|
| R7 exact lost race no-mock-bypass | **CLOSED AUTHORED + STDLIB HELPER** | 两个真实 bind caller、两个真实 private writes、两个 link 前 live temps、一个 real winner/一个 `FileExistsError` loser；exact 2 success；结束0 temp。 |
| R7 different lost race no-mock-bypass | **CLOSED AUTHORED + STDLIB HELPER** | 同样两个真实 caller/write/link；different winner 精确1 success/1 identity reject；winner保留；结束0 temp。 |
| R6 cleanup ownership/live publisher non-interference | **RETAINED** | production仅在 `finally` unlink自己随机 `temp_name`，无glob/foreign-temp cleanup；相邻 live-temp hostile保留。 |
| R6 subdirectory/artifact symlinks | **RETAINED AUTHORED, NOT PYTEST-EXECUTED** | ablation/stealth/viz及manifest/shard/stealth/guard/aggregate仍到达held-dirfd/no-follow production helpers。 |
| R6 guard count/order hostiles | **RETAINED AUTHORED, NOT PYTEST-EXECUTED** | invariance/recall count mismatch及multi-sample interleaved frozen target order测试未改。 |
| R5 guard run-v2 identity/count | **RETAINED STATICALLY** | declared count、sorted sample prefix、frozen-order targets、selection hash与aggregate exact counts未改。 |
| R5 complete-final/no-replace/durability | **RETAINED STATICALLY + R8 RACE REACHABILITY** | complete private write/file-fsync、hard-link no-replace、directory-fsync、exact winner未改；R8真实到达winner/loser。 |
| R5 contained fresh run/artifact naming | **RETAINED STATICALLY** | real output root、held no-follow run/subdir dirfds、direct-child artifact I/O和stale-root rejection未改。 |
| R4 siblings/shards/viz-cond4 | **RETAINED STATICALLY** | run/checkpoint/subset identities、canonical shard filename set、stale sibling及viz-cond4 fail-before-side-effect未改。 |
| R3 clean/checksum/occlusion/fan-in/raw-EMA | **RETAINED STATICALLY** | mandatory clean identity/selected checksum、boolean occlusion、unique rows/exact targets、EMA-before-selected-checksum顺序未改。 |
| Earlier official eval/six-task/dependency/PID/mode closures | **UNCHANGED FROM PRIOR STATIC CLOSURES** | 本 test-only diff未修改对应生产树；actual runtime gates仍 NOT RUN。 |

没有 model/head/NMS/metric、Protocol A/B、split/data ownership、candidate config、precision、
checkpoint schema、run cell、seed、gate或resource drift。历史 failed/passed jobs及其限制均保留；
full trainval `t1.v2` cache仍未执行/不存在。

## Checks actually run 与 explicit NOT RUN

本 R8 只执行 kickoff 允许的 read-only/static/stdlib checks：

1. startup root/HEAD/detached/status与唯一授权 review branch：PASS；
2. prior review parent/sole path/non-ancestry/blob/size/SHA-256与exact materialized prefix：PASS；
3. canonical AGENTS/env/roadmap与ledger through O-050、implementation/handoff parents、two-path
   ownership、unchanged production/request/results blobs及source/test hashes：PASS；
4. `git diff --check 35a0bdc..fdee4ba`：PASS；
5. `bash -n fl_v3/scripts/run_s07_b_static_checks.sh`：PASS；
6. stdlib `ast.parse` production T5与integration test：`AST_OK=2`；当前五个
   `s07_b_*.json` parse：`JSON_OK=5`；
7. complete source/test/HANDOFF逐行 trace，确认 fixture 两个真实 caller、private write、
   two-live-temp pre-link observation、real link winner/loser、exact/different结果与cleanup；
8. stdlib AST-extracted authored fixture/exact production helper执行：PASS；thread profiler精确
   记录 bind/publish/write 各4次；
9. 初次调用 `python` 得到 `command not found`；这是login PATH没有该别名的preserved
   harness-launch negative，不是代码/test失败。随后同一stdlib helper以`python3`原样通过；
10. tracked diff无pyc；最终物理 `find` 未发现 `fl_v3` 下任何 `.pyc/.pyo`，没有执行cleanup。

明确 **NOT RUN / NO IMPLIED PASS**：

- pytest或任何完整 authored test suite；
- `py_compile`、完整 static launcher或production-package import；
- Torch/NumPy/spconv/cumm import、native/JIT/build/source identity或operator dispatch；
- checkpoint parse、raw/EMA load、model/optimizer/scaler/scheduler/EMA construction/rollback；
- cache/manifest/data、directory/ZIP payload、fork/spawn/persistent workers；
- C/L/F forward/backward、fp16 S04 option-A、same-instance concurrency、memory/NMS；
- T5真实 shard/aggregate/stealth/guards/viz、five-condition/occlusion、official devkit；
- Slurm/srun/GPU、full cache、100/1000 steps、profile、mAP/NDS、DDP、matrix、seed、
  retry/rerun、FL、attack/defense、upload或publication。

## Interpretation、residual risk 与 O-009 boundary

允许解释：R7 唯一 test-evidence P3 已关闭；当前 authored race fixture无mock-bypass地到达
真实 bind/private-write/hard-link winner+loser/cleanup，production publisher ownership与R3-R6
static closures保留。S07-B 因此达到 **code-level/static-review PASS**，S00 可以准备一个新的、
独立、immutable、bounded O-009 validation proposal供审计。

禁止解释：本 PASS 不表示 O-009 自动获批或已执行，不表示pytest/runtime/production/
full-data/checkpoint/performance/scientific readiness；不得声称full trainval cache、100/1000
steps、profile、mAP/NDS、fusion gain、FL、attack/defense、generalization或publication evidence。
所有这些仍需各自 exact request、执行artifact与独立review。

Residual risk 是完整 authored suite及真实 GH200/Torch/spconv/data/model/checkpoint/T5 task路径
均未在本静态复审执行。该风险应由后续 separate bounded O-009 proposal精确列出并在S00审计
后决定是否执行；不能用本 review 回填runtime证据。

## Final verdict

**PASS at code-level/static-review scope for S07-B candidate
`fdee4ba574587a9974ac6a188f2c011dc4730f75`.**

R7 唯一 authored no-mock-bypass gap已由真实 concurrent bind/write/link/cleanup fixture关闭，
R3-R6 production/scientific identity closures未回归。此 verdict 只允许 S00 准备 separate
bounded O-009 proposal；它不是执行许可、runtime/production/full-data/scientific PASS，也不
授权merge到`v3-ad-perception`、push、upload或publication。

# S07-B-R9 独立复审 — O-059 spawn/runtime remediation 与 Job 349653 归因

## Findings（按严重性排序）

### P2 — session-scoped mini cache 迁移遗漏两个既有 fixture consumer，clean/read-only CWD 下会绕过新 cache 并使现有模型门失败

O-059 正确地把共享 `mini_cache_dir` 从 CWD-relative
`./fl_outputs/nuscenes/info_cache` 移到 `tmp_path_factory` 创建的 session scratch，并返回
该 exact 路径（`fl_v3/tests/conftest.py:66-74`）。`test_model_task.py` 也通过
`_cfg_with_cache()` 把该路径显式注入当前选中的 model-task cases
（`fl_v3/tests/test_model_task.py:77-90,107-186`）。但是该 fixture 还有两个现存 consumer，
它们声明了 `mini_cache_dir` 参数却完全不使用返回值：

- `test_overfit_single_scene_falsifiable(mini_cache_dir)` 仍把
  `nuscenes-cache-dir` 固定为 `./fl_outputs/nuscenes/info_cache`，并在构造模型前先调用
  `task.client_data()`（`fl_v3/tests/test_model_overfit.py:20,26-39`）；
- `test_v2_v3_render(tmp_path, mini_cache_dir)` 同样保留该固定路径，并从它构造 client
  loader（`fl_v3/tests/test_model_viz.py:12,22-33`）。

因此在 fresh/read-only snapshot 中，fixture 会在 pytest scratch 成功生成 cache，而这两个
测试仍访问不存在的 CWD cache；overfit gate 在其 pretrained-weight skip 边界之前即失败，
viz smoke 也直接失败。若开发 CWD 恰有旧 `./fl_outputs`，它们反而可能消费未由本次 fixture
生成、未绑定本次 temp identity 的 stale cache。Job 348557/348818/未来同一 25-file
launcher都没有选择这两个文件，所以当前 bounded suite 即使以后变绿也不会暴露该回归。

这不是 production data loader 缺陷，但它破坏了共享 fixture 的完整调用契约，并使已有
model-learns/viz gate 在 clean/read-only CWD 不可运行，属于明确的测试基础设施与 gate
完整性问题。修复不能恢复 CWD fallback；应在 owner 扩展 exact test ownership 后，把两个
consumer 都显式注入 `str(mini_cache_dir)`，并保留/扩展 changed-CWD hostile，证明没有
`./fl_outputs` 读取或写入。应再以 exact new SHA 独立复审。

### P3 — multiprocessing hostile 尚未直接锁住 dummy/zero-worker 两条生产分支，显式 fork timeout 的失败路径也没有完整回收证明

生产实现本身静态正确：

- `NuScenesMultimodalDataset.make_loader()` 在 `num_workers>0` 且 caller 未显式给低层
  test hook 时固定 `spawn`，而 `num_workers=0` 时不向 `DataLoader` 传
  `multiprocessing_context`（`fl_v3/src/fl_v3/data/nuscenes/dataset.py:363-383`）；
- `DummyRegressionTask._loader()` 仅在 worker 数大于零时传 `spawn`
  （`fl_v3/src/fl_v3/training/tasks.py:315-330`）；
- `NuScenesDetectionTask._make_loader()` 对实际 production task 显式传 `spawn`，零 worker
  传 `None`，随后 `make_loader` 不把 context 传给 PyTorch
  （`fl_v3/src/fl_v3/training/tasks.py:834-870`）。

但 authored assertions 只直接检查 nuScenes default/detection multi-worker spawn
（`test_nuscenes_zip_dataset.py:351-359`；`test_model_task.py:147-175`）。没有 case 构造
真实 `DummyRegressionTask` 的 `num_workers>0` loader 并检查 spawn，也没有直接断言 dummy
与 detection 的 zero-worker loader `multiprocessing_context is None`。这不会把静态实现变成
错误，但 O-059 的两条真实 caller 与 zero-worker no-context 契约缺少对称回归保护。

另外，显式 ZIP fork 已被正确移入 CUDA-hidden 的 fresh spawned helper；helper 内仍执行完整
parent-open、2 workers、persistent workers、两个完整 epoch、owner PID/reopen/read-growth 与
parent-state checks（`test_nuscenes_zip_dataset.py:257-316,322-348`）。正常路径的 loader
`finally` cleanup 也存在。不过 timeout 分支只对 helper 调用 `terminate()` 和一次十秒
`join()`；若它仍存活，没有 `kill()`/second join，也没有关闭 `result_queue`、关闭
`Process` 或证明其 forked DataLoader descendants 已退出（`:335-348`）。这正是 hostile
为 fork hang 设置的失败边界；未回收 helper/descendant/queue FD 可能污染同一 pytest
进程的后续 case 或在解释器 shutdown 再次等待。建议对 timeout/error/success 均使用
`try/finally`：terminate 后仍活则 kill，join 并检查最终退出，close/join queue 与 process，
并以 child PID/active-child 或等价可审计证据证明无残留。

### P3 — LiDAR test 的模块级说明仍声称旧 62-tensor layout，和本次正确的六任务断言冲突

`test_default_off_byte_identical()` 的 executable assertions 已正确更新为 total 230、head
183（legacy 15 + 168）、无 LiDAR backbone、fuser width 144；相邻 ON path 继续锁定
`+30` tensors（`fl_v3/tests/test_lidar_backbone.py:53-75`）。legacy-head regex 也由宽松的
`six task` 改为 exact `6 task dictionaries`
（`fl_v3/tests/test_s07_b_integration.py:432-438`），没有 gate weakening。

但文件顶层说明仍写着“DEFAULT-OFF byte-identity”与“62-tensor trainable layout intact”
（`fl_v3/tests/test_lidar_backbone.py:1-6`），函数名也保留旧 `byte_identical`。这不会改变
assertion 结果，但与 O-059 已批准的六任务 topology 相冲突，容易在后续 handoff/review 中
重新制造“62 是 gate”的错误解释。应只修正文档/名称，不改变 230/183/168/+30 数值门。

### 无 P0/P1 finding

未发现 data split/leakage、coordinate/class/metric/protocol 变化，未发现把 mini 或 attribution
结果外推为科学证据，未发现 canonical/collab/fl_v2 篡改、未授权 compute、merge、push、
upload 或 publication。Jobs 348557/348818 的失败与诊断仍原样保留；上方 P2/P3 足以阻止
本 exact candidate 的 code-level acceptance，但没有证据支持 P0/P1。

## Review identity、R8 prefix、candidate/import topology 与 ownership

- Session：`S07-B-R9`。
- R8-reviewed baseline：`fdee4ba574587a9974ac6a188f2c011dc4730f75`。
- Candidate `WORKER_SHA`：`797aaf4fa8115568692c381489928fb656f5f356`。
- Review import commit：`546ca61736a4484747c38377a032b98e169e6fe4`，唯一 parent 精确为
  candidate；commit 仅导入 exact prior `REVIEW.md`。
- Review branch：`codex/s07-b-r9-integrated-cl-stack-review`。
- `APPROVED_COMPUTE: none`。

Startup 在任何写入前精确为 review worktree
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/review_worktrees/s07b_r9_797aaf4`、
branch `codex/s07-b-r9-integrated-cl-stack-review`、HEAD `546ca617...`、clean status。
没有创建/切换/删除 branch 或管理 worktree。

追加 R9 前的 prior review 精确为：

- Git blob `384a4a531f7967f25c75fc1282e1a7767bd4f97c`；
- size `145973` bytes；
- SHA-256 `bdb4093a526efa22fc3f32bf99e97c5f6264b03e95b5985ee35eacc795f5876f`。

本 R9 只在这些 bytes 末尾追加；没有重写或删除 R8 prefix，也没有合入 prior reviewer
ancestry。权威 O-064 从 read-only canonical commit
`5fb0cc725aaaa89f840f1e6437cebc256f9016fb` 读取；review worktree 中滞后的 canonical
文件未被修改。

`fdee4ba..797aaf4` 的完整历史包括三个 bounded launcher、O-052/O-056/O-061 请求与终态
证据、O-059 implementation/test 与 O-063 runtime-aware test，共 13 changed paths；它们都在
相应 owner ledger ownership 内。O-063 terminal submission 记录之后的 exact remediation
`da262ff..797aaf4` 只有四路径：`test_model_task.py` 与 S07 的
`HANDOFF/RUN_REQUEST/RESULTS`。Candidate parent 链为
`da262ff -> 79be43d -> 8e2c31b -> 797aaf4`，均为单亲线性提交。

`fl_v3/src/fl_v3/training/loop.py` 在 R8 baseline 与 candidate 的 Git blob 都精确为
`881c070b1ef8affd350144cce33e508a241cf839`；五个 `s07_b_*.json` blob 也完全不变。
没有 production loop/config/model/head/NMS/metric/protocol 漂移。`git diff --check` 对
`fdee4ba..797aaf4` 与 exact O-063 四路径范围都无 warning。

## O-059 remediation 独立核验

| 项目 | R9 结论 | 证据与边界 |
|---|---|---|
| production workers>0 固定 spawn | **STATICALLY CLOSED** | dataset default、dummy 与 detection actual callers 如上；无 user config 可选 fork。 |
| zero-worker 不传 multiprocessing context | **STATICALLY CLOSED / DIRECT TEST GAP P3** | 三个实现均只在 `>0` 组装 kwarg；无对称 authored direct assertion。 |
| 显式 ZIP fork 隔离 | **STATICALLY CLOSED / CLEANUP GAP P3** | fresh outer spawn、CUDA hidden/assert unavailable、inner explicit fork、两个完整 epoch及 PID/reopen/read semantics保留；timeout有90+10秒界限，但最终 kill/descendant/queue cleanup未证明。 |
| CUDA-initialized production hostile | **AUTHORED, NOT EXECUTED IN THIS REVIEW** | exact process 先建立 CUDA tensor，再构造 detection spawn+persistent loader并取两个 iterator epoch heads；无 generic exception skip。 |
| tmp cache/read-only CWD | **PARTIAL / P2** | model-task四个原 PermissionError consumers已显式注入；另外两个共享 fixture consumers仍错用 CWD。 |
| diagnostic isolated basetemp parent | **CLOSED STATICALLY** | launcher在任何 isolated attempt前创建并验证 `$JOB_TMP/isolated` writable；不篡改 Job 348818 raw evidence。 |
| legacy message regex | **CLOSED** | exact `6 task dictionaries`。 |
| six-task LiDAR OFF topology | **EXECUTABLE ASSERTIONS CLOSED / DOC P3** | exact 230 total、183 head、183-15=168、ON-OFF=30、fuser 144与 no-backbone均保留。 |

所有 O-059 changed Python source/test 的 stdlib AST parse 为 `AST_OK=7`；三个 S07-B
launcher 的 `bash -n` 均通过。candidate source SHA-256 精确匹配 HANDOFF 所列值，包括
dataset `719ebf74...`、tasks `b81e3ca...`、conftest `fdaaa3bc...`、model-task
`b7412201...`、integration `2a820847...`、LiDAR test `7a8c2909...`、ZIP test
`3a24613e...` 与 diagnostic launcher `663a98a5...`。

## Jobs 348557/348818 negative evidence preservation

- Job 348557 仍明确为 `FAILED 1:0`、internal timeout 124、至少 `3F+4E`、无 JUnit/counts/
  final checksum manifest；没有从其 progress dots 推导任何 component PASS。
- Job 348818 仍明确区分 `COMPLETED 0:0` harness 与 `suite_pass=false`：251 isolated tests、
  3 failures、94 errors、0 skips；90 missing-parent launcher errors、4 read-only CWD errors、
  3 genuine failures和 combined fork queue hang均保留。110 checksum records仍是原始诊断证据。
- O-059 修复没有回写、删除或重标这两个 job 的 outputs/logs，也没有称其为修复后执行。

因此负面证据保留满足要求；本 R9 的 P2 正是静态发现尚未被 25-file diagnostic 选择覆盖的
共享 fixture 影响，不能被 Job 349653 的 attribution PASS 掩盖。

## Job 349653 raw artifact、source 与 attribution audit

本 review 直接读取 output root
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_dummy_attr_a9d657aebfb0`
及 scheduler logs，而不是只接受 HANDOFF summary。

### Exact request/source/dependency

- Executable launcher commit `a9d657aebfb0f64d271fa74e312d6054eca57e1d`；launcher
  blob/SHA-256 为 `295610fd422f3b371b8fd85e54785919903dc332` /
  `bbc1293a42034540327402a5df6c1f172b76afacca7906b4f0b71f5290b5968a`；environment
  bootstrap SHA-256 为 `f57befbb...`。
- pre-S06 `968d815...` 的 78-file list/state 为 `0ec5e43e...` / `dc2144cc...`；current
  `c69befe...` 的 85-file list/state 为 `104a6474...` / `0f2995fc...`。本 review 对两个
  `*_source_sha256s.txt` 的全部 `78/78` 与 `85/85` entries 从 exact Git blobs 重新散列，
  mismatch 均为 0；list/state aggregate 也精确一致。
- `execution_identity.json` 精确记录 n530/aarch64、CPython 3.11.15、同一 interpreter，
  NumPy 1.26.4、SciPy 1.13.1、Torch 2.11.0+cu128、torchvision 0.26.0+cu128、spconv
  2.3.8、cumm 0.7.13、nuscenes-devkit 1.1.11、pyquaternion 0.9.9、Pillow 12.2.0，
  one node/task/GH200、four CPUs。四个 result JSON 都重复记录 Python 与 NumPy/Torch
  identity和 exact workload config。

### Four independent subprocesses 与 classification

`attempts.tsv` 精确有四行：pre/current 各 repetition 1/2，exit 均为 0；四个 result JSON
分别绑定 exact snapshot commit、same seed-42 CPU dummy config、`defense=none`、
`server_round=1`、`n_clients=4`、`decision_valid=true`。四个 checksum 全部精确为
`4fa46307bab67f2a836102b23b1ad2abc331702e83d16c65e11a09330c3d9edb`。

`attribution_summary.json` 因而正确给出 `diagnostic_complete=true` 与
`classification=stable_equal_current`，同时保留 historical
`d2d819fee9a54fc302a9d6c9d0ac4e4d875629a0a16e75f2328f28b7f63cd7cc`，并明确
`automatic_code_or_golden_change_authorized=false`。该结果支持：在 exact frozen Arrhenius
runtime 中，pre-S06 与 current source 都稳定产生 `4fa463...`，所以 Job 348818 相对旧
`d2d819...` 的差异不能归因于 S06/current source change；它不单独定位某个库/硬件原因，
也不建立跨平台 universal checksum。

### Artifact completeness

`sha256sums.txt` 的 SHA-256 为 `0c74aae4...`，包含且验证全部 25 个 formal artifacts；本
review 再次执行 read-only `sha256sum -c`，25/25 均 OK。summary、attempts、identity 分别为
`806afbfd...`、`dfa41729...`、`b66bbc74...`。scheduler stdout/stderr SHA-256 为
`48f6a06c...` / `ae633085...`，stdout逐项显示25次 OK，stderr仅为已记录的 module-purge
notice。RESULTS 中这些路径、哈希、resources和 interpretation limits均与 raw bytes一致。

## Runtime-aware dummy test contract

`test_model_task.py` 保留旧 `DUMMY_AGG_GOLDEN=d2d819...` 作为 historical evidence，新增
exact `ARRHENIUS_DUMMY_AGG_GOLDEN=4fa463...` 与 runtime tuple
`(aarch64, CPython, 3.11.15, torch 2.11.0+cu128, numpy 1.26.4)`。测试在所有 runtime 都
执行两个新的 `run_clean_round()`，要求两个 checksum 都是 64-lower-hex 且 same-runtime
exact equality；只有 tuple 完整相等时才额外断言 `4fa463...`
（`fl_v3/tests/test_model_task.py:22-73`）。unknown runtime 没有 skip，也不再错误断言旧
cross-environment golden；frozen Arrhenius exact-4fa branch与 Job 349653 的
`importlib.metadata` identity定义一致。

这项变更没有修改 `training/loop.py` 或任何 production source。其局限是本 R9 未执行该
pytest；若未来在能 import Torch/NumPy 但缺 distribution metadata 的非标准 source-only
environment运行，`importlib.metadata.version()` 会 error而不是把它归为 unknown runtime，
但这不影响当前冻结 Arrhenius install identity。可把该点作为 portability residual，不应
据此扩张 Arrhenius golden 的适用范围。

## Checks actually run 与 explicit NOT RUN

本 R9 严格只执行允许的 Git/hash/stdlib/static/read-only checks：

1. startup root/HEAD/branch/status、import-parent/candidate关系：PASS；
2. R8 prefix blob/size/SHA-256：PASS；
3. root AGENTS、权威 O-064 canonical docs、S07 四份交付与全部 prior review完整读取：PASS；
4. `fdee4ba..797aaf4` 与 O-063 四路径 diff/topology/ownership/`diff --check`：PASS；
5. 七个 changed Python files 的 stdlib AST parse：`AST_OK=7`；
6. 三个 launcher 的 `bash -n`：`SHELL_OK=3`；
7. candidate source SHA-256、training-loop/config blobs、generated artifact absence：PASS；
8. Job 349653 raw JSON/TSV/log/manifest读取，25/25 checksums与163个 immutable Git-source
   records重算：PASS；
9. fixture consumer、DataLoader caller/context、explicit-fork control flow、topology/regex
   assertion逐行静态 tracing：产生上述 findings。

明确 **NOT RUN / NO IMPLIED PASS**：

- pytest、pycompile、project/package import、Torch/NumPy/CUDA/spconv/cumm import；
- 任何 directory/ZIP/data/cache/model/checkpoint、worker process、fork/spawn runtime；
- Slurm/srun/GPU/job query或新 compute；
- C/L/F forward/backward、fp16 option-A、rollback、official devkit、T5 task；
- full trainval `t1.v2` cache、100/1000-step、profile、mAP/NDS、DDP、matrix、seed/rerun、
  FL/attack/defense/scientific cell；
- merge到`v3-ad-perception`、push、upload、publication。

## Gate verdict、allowed/forbidden interpretation 与 residual risk

| Layer | R9 verdict |
|---|---|
| R8 exact prefix/import/candidate lineage | **PASS** |
| Job 349653 source/dependency/four-process/checksum attribution | **PASS within bounded attribution scope** |
| production spawn policy for dataset/dummy/detection | **PASS STATICALLY** |
| explicit ZIP fork isolation/two-epoch semantics | **PASS STATICALLY; cleanup/test gaps P3** |
| diagnostic-parent/regex/six-task executable assertions | **PASS STATICALLY** |
| shared mini-cache fixture migration | **CHANGES-REQUESTED (P2)** |
| integrated GH200 runtime suite | **FAILED/DIAGNOSTIC ONLY; no post-fix run exists** |
| production/full-data/scientific readiness | **NOT ESTABLISHED / FORBIDDEN** |

允许解释：Jobs 348557/348818 的原始负面结果保留；O-059 production spawn policy和主要
hostile主体已形成可审查静态实现；Job 349653 在 exact frozen runtime 下可靠地把 dummy
差异分类为 `stable_equal_current`；runtime-aware test不改 production loop并保留旧 hash。

禁止解释：不得称 O-059 后 integrated 25-file suite 已运行或 PASS，不得忽略两个被共享
fixture 破坏的现有 tests，不得称 fork timeout cleanup、dummy/zero-worker direct regression
已完整覆盖；不得称 production/full cache/full trainval/100/1000/profile/metric ready，不得
声称 mAP/NDS、fusion gain、FL、attack/defense、generalization或publication evidence。

Residual risks还包括：full `t1.v2` cache不存在；actual post-O-059 spawn/ZIP/CUDA/model suite
未执行；S04 integrated fp16/concurrency/EMA与official eval仍未得到新 runtime evidence；
Job 349653只隔离source-vs-runtime attribution，不定位具体依赖/CPU/BLAS原因；Protocol-B
split/client ownership不在本 CL engineering review内。

## Final verdict

**CHANGES-REQUESTED for S07-B candidate
`797aaf4fa8115568692c381489928fb656f5f356`.**

Job 349653 的 exact attribution、runtime-aware dummy contract、production spawn主路径、
CUDA-hidden explicit-fork设计、diagnostic-parent、regex和六任务数值断言可以在各自静态/
bounded边界内接受；但 shared `mini_cache_dir` 迁到 tmp 后遗漏
`test_model_overfit.py` 与 `test_model_viz.py`，使两个现有 gate在clean/read-only CWD必然
绕过新cache并失败或误用stale CWD cache。修复全部fixture consumers并处理上方P3回归保护/
cleanup/doc问题后，exact new SHA必须再次独立review；在此之前没有code-level acceptance、
后续runtime proposal、production/full-data或scientific授权。

# S07-B-R10 独立复审 — O-065 fixture/lifecycle remediation

## Findings（按严重性排序）

### P2 — ready record 没有父子握手；父侧在取得 PGID 前失败时只杀 helper，异步 `Queue.put()` 后已经 fork 的 DataLoader descendants 可能泄漏

helper 先 `setsid()`，随后把 ready tuple 放进 `multiprocessing.Queue`，并立即进入
`_persistent_lifecycle(..., "fork")`（`fl_v3/tests/test_nuscenes_zip_dataset.py:325-330`）。
父进程只有在 `result_queue.get(timeout=10)` 返回并通过 tuple assertions 后才把
`group_id` 视为可用（`:367-376`）。问题是 `multiprocessing.Queue.put()` 只把对象交给
helper 本地 feeder；它不是父进程已经收到并确认 PGID 的同步握手。helper 因而可以在 ready
record 尚未到达父进程时继续启动两个 fork DataLoader workers。

若父侧 ready `get()` 超时、queue feeder/pipe 异常，或在 `group_id` 赋值前的 ready
assertion 失败，`finally` 中 `group_id is None`，所以跳过 whole-group `killpg`
（`:399-407`）。它只对 `multiprocessing.Process` 代表的 helper 执行
`terminate()/kill()/join()`；这不等价于终止 helper 已经 fork 的 grandchildren。helper 被
SIGTERM/SIGKILL 时也不会执行 Python/DataLoader 的 orderly daemon-child shutdown。因此本次
remediation 对“ready 前失败”并非 all-path descendant cleanup，仍可能把 fork workers 留给
后续 pytest case/解释器 shutdown。normal path 的 `_shutdown_workers()`、worker join/dead
assertion和 `active_children()==[]`（`:309-339,392-398`）不能覆盖这个窗口；ready 已确认后的
90 秒 timeout path 才具备正确的 isolated-group kill。

同时，代码在验证 `ready[2]`/`ready[3]` 前先执行 `group_id = ready[3]`（`:374-376`）。exact
helper 的正常 `setsid()` record 会令 SID/PGID 等于 helper PID，因此本 review 没有确认正常
路径会误杀 pytest parent；parent 本身也不在 helper 的新 session/group 中。但 fail-closed
cleanup 不应在 record 尚未验证时 arm 一个将传给 `os.killpg()` 的值。一个 malformed/error-
injection ready record会在 assertion failure 的 `finally` 中使未验证 group 成为 kill target。

**Required remediation:** 使用双向 ready/ack handshake：helper 在父进程确认 exact helper
PID、SID、PGID 后才允许进入 `_persistent_lifecycle`/fork；或者使用等价的、父侧可安全独立
验证的 group ownership primitive。父侧必须先在局部变量中验证 complete ready record，必要
时再用 OS-level PGID 查询交叉确认，然后才设置 armed `group_id`。所有 ready-timeout、malformed
ready、helper error、90 秒 timeout和 success路径都必须终止并审计 whole isolated group，随后
关闭/join queue thread、reap/close `Process`。增加一个 authored hostile，在 ready delivery/ack
窗口强制失败并让 helper 尝试生成 descendant，证明 parent 不会被 signal 且 helper/group/
worker PID 均不残留。仅有正常执行路径不能关闭此 finding。

### P3 — cleanup 可以遮蔽原始 lifecycle assertion，且没有 authored error/timeout case 证明所声称的诊断与回收顺序

`_persistent_lifecycle()` 把 `_shutdown_workers()`、逐 worker `join(5)`、dead assertion、
dataset close 和 GC 直接放进一个 `finally`（`test_nuscenes_zip_dataset.py:279-322`）。若主体的
epoch/PID/reopen/read assertion 已经失败，而这些 cleanup 操作中的任一个再失败，Python 会把
cleanup exception 作为当前异常；helper 随后只发送 `repr(exc)`（`:335-339`）。这样原始
lifecycle assertion/traceback被遮蔽，父侧只做 `assert outcome[1] is None`，也不会在 error
outcome 上独立验证所报告的 `live_children` 已在 group kill 后消失（`:392-407`）。

本 diff 没有新增强制 helper assertion failure、worker-shutdown failure或90 秒 timeout的
authored case；所以 HANDOFF `:1950-1956` 对 normal/error/timeout 全部“preventing leakage”的
措辞超过实际可达证据。应保留主体 exception/traceback，单独收集 cleanup failure（例如明确
chaining/`ExceptionGroup`），并让 parent 在失败报告中同时保留 original failure、cleanup
failure、worker PID与最终 group-reaped proof。上述分支需要独立 hostile，不得靠静态存在的
`finally` 与正常路径外推。

### 无 P0/P1 finding

未发现数据 split/leakage、coordinate/class/metric/protocol 或 production source变化；未发现
canonical/collab/fl_v2、RUN_REQUEST/RESULTS、training loop、runtime launcher或五个 candidate
config漂移。没有未授权 compute、merge、push、upload 或 publication。上述 P2/P3 是测试
lifecycle/all-path cleanup与证据完整性问题，不改变模型或科学协议。

## Review identity、R9 prefix、import topology 与 exact ownership

- Session：`S07-B-R10`。
- Remediation base：`797aaf4fa8115568692c381489928fb656f5f356`。
- Candidate `WORKER_SHA`：`97588f7ad556fe1ce1a5f7bd76cee19e79d16d31`。
- Test commit：`3f3686c3fbbfd3fb1bb516a9c00f0612d9da0f04`，parent exact 为 remediation base。
- Startup/import HEAD：`a900cab039d18985c03ea0e53bc39d4a9f2f6904`，branch
  `codex/s07-b-r10-integrated-cl-stack-review`，clean。
- 权威 canonical：`589b2a99965de9031a79f66768a1c6821857e69b`（O-066）。
- `APPROVED_COMPUTE: none`。

Import topology 精确为：candidate `97588f7` → review-only `fe35e47` → review-only
startup `a900cab`。`fe35e47` 的唯一 diff 是把 prior review bytes 补至 R8；`a900cab` 的唯一
diff 是追加 R9。candidate 与 startup 除
`fl_v3/usenix27_orchestra/handoffs/S07/REVIEW.md` 外 tree 内容完全相同；没有把 prior reviewer
ancestry当 implementation merge。

追加 R10 前的 exact R9 prefix 为：Git blob
`9719ff6d35435eac00cf0f194c3032515802f148`、size `164814` bytes、SHA-256
`318a752ec30d5eb9cac07cc8dfec4b42f3f2371944f8ab51edf79c01189f646c`。本段只追加在这些
bytes末尾，没有重写、删除或重新编码 prior prefix。

`797aaf4..97588f7` 是线性两提交、exact 六路径、214 insertions/20 deletions：五个获批
test文件与 `S07/HANDOFF.md`。没有第七个 changed path；`git diff --check` 无 warning。
HANDOFF 中五个 post-test SHA-256 与 candidate bytes精确一致。

Forbidden blobs在 remediation base与 candidate完全相同：

- `training/loop.py`：`881c070b1ef8affd350144cce33e508a241cf839`；
- `training/tasks.py`：`86ab9d0563e1636d6c4cde06986470d2559f19f7`；
- nuScenes `dataset.py`：`afd2707d3939d2d76205996fe94d29fcfc4ed5f3`；
- `RUN_REQUEST.md`：`efa5ce78eac121f2dd3e70ea75ef414023d45d13`；
- `RESULTS.md`：`b3b80625ef6c38e4d9382e11ded5c8534b5556ae`；
- candidate-local `REVIEW.md`：`cd0e0795402c2892fe199691a6a01f483d6a457f`；
- runtime launcher：`1e182ebc1fe883ad59702bfeb1b3db110bbf54c1`。

## R9 finding closure matrix

| R9 requirement | R10 独立结论 | Evidence / boundary |
|---|---|---|
| overfit exact mini cache + changed CWD/no `fl_outputs` | **CLOSED STATICALLY** | `test_model_overfit.py:20-42` 注入 exact `str(mini_cache_dir)`，在 task/data前 chdir，并在 `client_data()` 后证明无 CWD output。没有恢复 fallback。 |
| viz exact mini cache + changed CWD/no `fl_outputs` | **CLOSED STATICALLY** | `test_model_viz.py:12-47` 注入同一 session cache；data construction后及render/manifest后均证明无 `tmp_path/fl_outputs`。Viz artifacts显式写 `tmp_path`，不是 stale cache fallback。 |
| real dummy workers=2 spawn/batch/shutdown | **CLOSED AUTHORED, NOT EXECUTED** | `test_model_task.py:76-90` 真实 `DummyRegressionTask.client_data`、spawn assertion、真实 `(8,4)/(8,1)` batch及 explicit iterator shutdown。 |
| dummy/detection workers=0 no context + batch | **CLOSED AUTHORED, NOT EXECUTED** | dummy train/val均断言 `None`并取 batch（`:93-103`）；detection eval loader断言 `None`并取 dict batch（`:191-199`）。 |
| explicit fork normal path DataLoader shutdown/PID/active children | **CLOSED STATICALLY/AUTHORED, NOT EXECUTED** | persistent iterator显式 shutdown/join/dead；normal helper返回 worker PIDs并要求 `active_children()==[]`。 |
| explicit fork error/timeout all-path cleanup | **OPEN — P2/P3** | ready-before-ack window可跳过 group kill；error cleanup可遮蔽 original assertion；无对应 hostile。 |
| LiDAR doc/name and numerical gates | **CLOSED** | module/function wording改为 six-task；OFF 230、head 183、`183-15=168`、no-backbone、fuser 144及 ON-OFF `+30` 均未弱化。 |
| runtime/negative evidence preservation | **PASS** | Jobs 348557/348818仍是 failure/diagnostic suite-fail；Job 349653仍只支持 `stable_equal_current` bounded attribution；无 post-O-059 integrated runtime run。 |

## Checks actually run 与 explicit NOT RUN

本 R10 只执行允许的 Git/hash/static/artifact checks：

1. startup root/branch/HEAD/status、candidate/import parents与 non-review tree equality：PASS；
2. R9 prefix blob/size/SHA-256：PASS；
3. root AGENTS、权威 O-066 canonical三文件、S07 HANDOFF/RUN_REQUEST/RESULTS/完整 prior REVIEW、actual diff与相关 source/test完整读取：PASS；
4. exact two-commit/six-path ownership、`git diff --check`、forbidden blobs/config不变：PASS；
5. 五个 changed test SHA-256与 HANDOFF：PASS；
6. cache consumers、dummy/detection context、fork session/group/queue/process/worker控制流及LiDAR数值门逐行追踪：产生上述 findings。

明确 **NOT RUN / NO IMPLIED PASS**：pytest、pycompile、project/package import、Torch/NumPy/
CUDA/spconv/cumm、data/cache/model/checkpoint、任何 worker/fork/spawn runtime、Slurm/srun/GPU/
compute、full trainval `t1.v2`、100/1000 steps、profile、mAP/NDS、DDP、matrix、seed/rerun、
FL/attack/defense/scientific cell。未 merge/push/upload/publication。

## Interpretation、residual risk 与 final verdict

允许解释：R9 的两个 mini-cache consumer、dummy/detection直接 context contracts、LiDAR
wording/numerical gates以及 explicit-fork normal-path shutdown已形成正确的静态/authored
remediation；exact ownership、forbidden blobs、Jobs 348557/348818负面证据和 Job 349653
bounded attribution均保留。

禁止解释：不得称 explicit-fork helper 的 ready/error/timeout all-path descendant cleanup 已
关闭，不得称 post-O-059 integrated suite已运行或PASS；不得称 production/full-cache/full-
trainval/100/1000/profile/metric ready，亦不得声称 mAP/NDS、fusion gain、FL、attack/defense、
generalization或publication evidence。

Residual risk还包括所有新增test均未执行、full `t1.v2` cache仍不存在、integrated fp16/
concurrency/EMA/official eval仍无新 runtime evidence。上述 P2 需要先修复握手和安全 arm/kill
顺序，P3 需要保留原始failure并补齐error/timeout hostiles；然后 exact new SHA再次独立review。

**CHANGES-REQUESTED for S07-B candidate
`97588f7ad556fe1ce1a5f7bd76cee19e79d16d31`.**

本 verdict 不授权任何 runtime proposal、compute、merge、push、upload 或科学解释。

---

# S07-B-R11 独立复审 — O-067 ready/ACK 与 all-path cleanup remediation

## Findings（按严重性排序）

### P2 — “leader 已退出、group/descendant 仍存活”路径没有被等待或 hostile 覆盖；当前 cleanup 会把对已退出 leader 的 `join()` 误当成 group 等待

候选正确地把 `armed_group` 的赋值放到完整 ready tuple 与 live kernel
`getsid/getpgid` 验证之后，并用 ACK 阻止 child 在此之前进入任何 fork/DataLoader
路径（`fl_v3/tests/test_nuscenes_zip_dataset.py:433-466,549-631`）。但 O-068 明列的
**leader-dead remaining-group** 分支仍未闭合。

cleanup 在发现 `armed_group` 存在后发送 SIGTERM，随后调用
`process.join(2)`；若 group 仍存在则发送 SIGKILL，再调用 `process.join(5)`
（`:675-727`）。这两个 `join()` 只等待 `multiprocessing.Process` 所代表的 helper
leader，不等待 process group。若 leader 在 ACK 后已经退出、只留下 fork worker/raw
descendant，它们都会立即返回；代码随后立刻执行最终 `killpg(group, 0)` 检查
（`:741-759`），没有独立于 leader 的 bounded group-disappearance poll。TERM/KILL 后的
descendant 尚未被调度退出、正成为 orphan/zombie，或正等待 init/subreaper 回收时，当前
代码会把短暂存在的原 group 报成 cleanup failure。更重要的是，它没有可执行证据证明
这一分支最终清除原 descendant。

现有 `forced_hang` 不能覆盖该问题。`_hang_with_forked_descendant()` 给 helper leader
安装 SIGTERM handler；handler 阻塞 `waitpid(descendant_pid, 0)`，随后 leader 继续
`signal.pause()`，特意保持存活直到父进程发送 SIGKILL（`:411-430`）。因此该 hostile
证明的是“live leader 先 reap child、随后被 KILL”，而不是“leader 已经退出、remaining
group 由 parent 清理”。对应 test 只断言这条规避分支
（`:957-978`）。HANDOFF 所称“cleanup no longer depends on the helper leader remaining
alive”及“every armed group ... joined and audited absent”
（`fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md:2025-2026`）超过实际实现与 authored
coverage。

**Required remediation:**

1. 增加真实 `leader_exit_with_descendant` hostile：helper 在 ACK 后 raw-fork 一个保持在
   exact isolated PGID 的 descendant，向 parent 发送其不可歧义 identity，然后 helper
   自己正常或异常退出而不等待 descendant；parent 必须先证明 direct `Process` 已退出而
   `killpg(pgid, 0)` 仍为真，才能进入被测 cleanup。
2. 把 direct-child reap 与 group cleanup 分开：`Process.join()` 只用于回收 helper；每次
   TERM/KILL 后用 monotonic deadline 独立轮询原 group/descendant identity。TERM deadline
   到期且原 group仍存在才发送 KILL；KILL 后必须再次 bounded wait，不能靠再次 join 已退出
   leader。最终才关闭 `Process` sentinel。
3. 对 orphan descendant 的“gone/reaped”证明必须识别原进程实例，而不是只看裸 PID；在
   Linux/CPython 3.11 环境可在 descendant 活着时记录 `/proc/<pid>/stat` starttime 或打开
   pidfd。若测试要声称由 pytest parent **reap**，则需显式、安全地采用并恢复 subreaper
   contract；否则措辞应精确为 original descendant exited and no matching process/group
   remains，而不能宣称 parent `waitpid` 了非 child。
4. hostile 必须继续证明 parent PID/SID/PGID 未变、无 parent-group signal、control/Queue/
   sentinel 全闭合，并在后续测试中无残留。

在这一修复与新 exact-SHA 独立复审前，R10 的 all-path descendant cleanup 仍为
`CHANGES-REQUESTED`。

### P2 — parent 先 `join()` producer、后读取 `multiprocessing.Queue`，仍存在 feeder backpressure deadlock，可能把 primary traceback/cleanup notes 重新降格成 timeout

child 的正常/error outcome 经 `result_queue.put(...)` 发送；其 `finally` 随即执行
`result_queue.close(); result_queue.join_thread()`（`test_nuscenes_zip_dataset.py:470-499`）。
parent 却先 `process.join(run_timeout)`，只有确认 helper 退出后才
`result_queue.get(timeout=5)`（`:638-645`）。这正是 CPython `multiprocessing.Queue`
producer 的经典 deadlock 顺序：`put()` 只进入 child feeder buffer，child 的
`join_thread()` 等 feeder 把全部 pickle bytes 写入 pipe；若 outcome（尤其完整 traceback、
多个 cleanup notes 或较长 exception repr）超过 pipe 可用容量，parent 又在 join 中不读，
双方互相等待直到 `run_timeout`。此时本应回传的 forced primary error 会被父侧
“helper timed out”替代，R10 所要求的跨进程 primary traceback preservation 实际仍不具备
all-size/all-cleanup-path保证。

本轮 fixed `forced_error` payload 通常较小，因此静态审查不能断言该 authored case必然
触发 deadlock；但 all-path harness 不能把正确性建立在 traceback/notes 小于平台 pipe
容量的隐含条件上。CPython 3.11 的 `Queue` 实现也明确显示 producer 的
`close()`只向 feeder buffer排 sentinel，而 `join_thread()`等待 feeder；parent get-only
Queue 本身没有 feeder finalizer。这使上述等待环真实存在，不是 style issue。

**Required remediation:** parent 必须在 producer join 之前持续 drain outcome（例如 normal/
error path 先按 deadline `get()` 完整 outcome，再 bounded join/reap helper；hang path在无
outcome deadline后进入 group cleanup），或改用一个具有明确 framing/parent-read-before-
child-exit contract 的同步 channel。不得用 `cancel_join_thread()`静默丢失 primary report。
新增一个超过 pipe capacity 的 traceback/cleanup-note hostile，证明 parent 收到完整 primary
traceback和notes、helper正常reap、Queue/control/Process资源闭合且不超时。

### P3 — worker/descendant 的裸 PID `kill(pid, 0)` audit 不区分 PID reuse；private Queue/FD 断言仅在冻结的 CPython 3.11 layout 下成立

forced-error/forced-hang 最终只保存 integer PID，parent 稍后用 `os.kill(pid, 0)` 判断存活
（`:761-803`）。原 worker/descendant 已退出后若 PID 被快速复用给无关进程，这会把“原进程
已消失”误报为残留；它也不能证明所见 task 是原 child。当前代码不会向这些裸 worker PID
直接发 signal，因此主要后果是 flaky false failure；但在测试 process cleanup 与 PID-reuse
审计中不能把它称为 exact per-PID proof。应按上一 finding 记录 starttime/pidfd identity，
将 ESRCH 或 identity 改变都解释为“原进程 gone”，仅同一 identity 存活才失败。

parent 对 get-only Queue 的处理在**当前冻结 CPython 3.11**语义下静态可行：parent从未
`put()`，所以 `_thread/_close/_jointhread` 均未启动；`Queue.close()`不会关闭 pipe，随后
显式关闭 `_reader/_writer` 不构成同一 parent object 的 double-close；child producer的
normal/error path则由自己的 feeder close/join。`Process.close()`在 direct child 已停止后
关闭 sentinel。这里没有确认新的 CPython-3.11 必败点。不过 `_reader/_writer` 是 private
API，且最终只对早先记录的 integer FD做 `fstat(...)=EBADF`（`:805-847`）；若未来 Python
layout改变或 FD number在检查前被复用，证据会变脆。该限制应保留为版本绑定的 P3，不能
外推到其他 Python/runtime。上方 Queue drain-order P2 则即使在冻结 CPython 3.11 也成立。

### 无 P0/P1 finding

本 diff 只有获批 test 与 HANDOFF 两路径；没有 production/model/data/config/metric/protocol、
canonical/collab/fl_v2、RUN_REQUEST/RESULTS、negative artifacts 或 scientific contract
变化。没有未授权 pytest/import/compute、merge、push、upload 或 publication。上述两项 P2
是 explicit-fork test harness 的 all-path cleanup/diagnostic correctness blocker，不是模型或
科学协议变化。

## Review identity、prefix、import topology 与 exact ownership

- Session：`S07-B-R11`。
- Remediation base：`97588f7ad556fe1ce1a5f7bd76cee19e79d16d31`。
- Candidate `WORKER_SHA`：`8469eb4944f164f5bd2fa1aa833ea4df0acf04b3`。
- Test commit：`6782fa19ca2e4c021ac5215c3e85dd939f4296f9`，parent exact 为
  remediation base；该 commit 只改
  `fl_v3/tests/test_nuscenes_zip_dataset.py`，exact numstat `633/68`。
- Handoff-only candidate commit：`8469eb4944f164f5bd2fa1aa833ea4df0acf04b3`，parent exact
  为 test commit。
- Startup/import HEAD：`1839c94b28b016ed690165b15c013c0e72544f8b`，branch
  `codex/s07-b-r11-integrated-cl-stack-review`，startup clean。
- 权威 canonical：`e999b5d3854d21ee4e22f709fadc6bcd297b6535`（O-068）。
- `APPROVED_COMPUTE: none`。

Import topology 精确为 candidate `8469eb4` → review-only `935734d` →
review-only `a7e97aa` → review-only startup `1839c94`；这三个后续 commit 的唯一 changed
path 都是 `fl_v3/usenix27_orchestra/handoffs/S07/REVIEW.md`。candidate 是 startup 的
ancestor；除该 REVIEW path外，candidate与startup tree一致，没有把 reviewer ancestry
当 implementation merge。

追加 R11 前 exact R10 prefix 为 Git blob
`b100c30123104063b3c1f88a6909008f3b2b888d`、size `175932` bytes、SHA-256
`1f755af1e8811253b0fec332680f06ae43dcc899cd640f4cf147d70f9863900d`。本段只追加在这些
bytes末尾，未重写、删除或重新编码 prior prefix。

`97588f7..8469eb4` 精确只有两个 changed paths：

- `fl_v3/tests/test_nuscenes_zip_dataset.py`：test commit `633` insertions / `68`
  deletions，candidate SHA-256
  `0c5a4e65403ec37329503aff95c0d07bcc9c5b2dd811c9a0e598c4b4d9e2cca8`；
- `fl_v3/usenix27_orchestra/handoffs/S07/HANDOFF.md`：仅末尾追加 O-067 remediation
  record，candidate SHA-256
  `c74a0ba576e388c361f4ad31b8459e716a9a2de5d44f03667c7330a2e571f164`。

`git diff --check 97588f7..8469eb4` 无 warning。HANDOFF 的 test hash、test commit、
parent、R10 prefix blob/size/SHA-256与 candidate bytes精确一致；但其 leader-dead/all-path
措辞受首项 P2 限制。

Forbidden blobs在 remediation base与candidate精确相同：

- `training/loop.py`：`881c070b1ef8affd350144cce33e508a241cf839`；
- `training/tasks.py`：`86ab9d0563e1636d6c4cde06986470d2559f19f7`；
- nuScenes `dataset.py`：`afd2707d3939d2d76205996fe94d29fcfc4ed5f3`；
- `RUN_REQUEST.md`：`efa5ce78eac121f2dd3e70ea75ef414023d45d13`；
- `RESULTS.md`：`b3b80625ef6c38e4d9382e11ded5c8534b5556ae`；
- candidate-local `REVIEW.md`：`cd0e0795402c2892fe199691a6a01f483d6a457f`；
- runtime launcher：`1e182ebc1fe883ad59702bfeb1b3db110bbf54c1`。

## R10 requirement closure matrix

| R10/O-068 requirement | R11 独立结论 | Evidence / boundary |
|---|---|---|
| duplex ready/ACK；pre-ACK禁止 fork | **CLOSED STATICALLY/AUTHORED, NOT EXECUTED** | child `send→recv ACK→fork`顺序明确；parent完整tuple、process.pid、kernel SID/PGID、parent-distinct后才arm/ACK；pre-ACK hostile在verified ready窗口失败。 |
| unvalidated/unarmed path不 `killpg` parent group | **CLOSED STATICALLY** | `armed_group`只在validator返回后赋值；`None`分支只调用 exact `Process.terminate/kill`；parent identity末尾核对。现有 pre-ACK hostile本身是validated-and-armed-before-ACK，不是malformed-unarmed case，但结构上无 unvalidated killpg。 |
| leader-dead remaining group/descendant TERM/KILL/gone | **OPEN — P2** | code检查group，但等待只join leader；无group deadline；forced-hang故意保持leader活着并由leader reap descendant。 |
| primary traceback不被 lifecycle cleanup覆盖 | **CLOSED IN CHILD STATICALLY; CROSS-PROCESS QUEUE P2** | `_persistent_lifecycle`保存原 traceback并附notes；fixed forced-error检查notes/PIDs。producer join-before-drain仍可把大report变成timeout。 |
| forced-error进入真实fork DataLoader并回传worker PIDs | **CLOSED AUTHORED, NOT EXECUTED** | error在第一个真实batch后触发；iterator workers被记录；parent逐PID检查。PID identity精度受P3限制。 |
| forced-hang进入真实 raw-fork descendant | **CLOSED FOR LIVE-LEADER CASE ONLY** | descendant真实fork、TERM后由leader `waitpid`、leader再KILL；不能外推leader-dead。 |
| Queue/control/sentinel/Process.close | **STRUCTURALLY VALID FOR CPYTHON 3.11, NOT EXECUTED; QUEUE ORDER P2** | endpoint ownership/close语义成立且无明显double-close；但join-before-drain可hang，private API/FD-number证据保留P3。 |
| normal two epochs/PID/reopen/read-count | **RETAINED STATICALLY/AUTHORED, NOT EXECUTED** | 两epoch payload equality、同一两worker PID set、owner/current PID、reopen不增/read增、archive set、explicit shutdown仍在。 |
| only test+HANDOFF；negative/results/forbidden blobs不变 | **PASS** | exact two-path diff与blob audit如上。 |

## Checks actually run 与 explicit NOT RUN

本 R11 只执行 kickoff允许的 Git/hash/static/artifact检查：

1. startup root/HEAD/branch/status、candidate/import parent chain与唯一 review-only paths：PASS；
2. R10 prefix blob/size/SHA-256：PASS；
3. root AGENTS、三份 canonical、权威 O-068、S07 HANDOFF、完整 prior REVIEW、candidate
   actual diff及相关 test/CPython 3.11 multiprocessing source读取：完成；
4. exact two-commit/two-path ownership、`633/68` numstat、candidate hashes、
   `git diff --check`与forbidden blobs：PASS；
5. ready/ACK、kernel validation、group signal/join、primary exception、Queue feeder/endpoint、
   Process sentinel、worker/descendant PID与normal two-epoch控制流逐行追踪：产生上述 findings。

明确 **NOT RUN / NO IMPLIED PASS**：pytest、pycompile、AST/source-text compile、project/
package import、Torch/NumPy/CUDA/spconv/cumm、data/cache/model/checkpoint、任何 worker/fork/
spawn runtime、Slurm/srun/GPU/compute、full trainval `t1.v2`、100/1000 steps、profile、mAP/
NDS、DDP、matrix、seed/rerun、FL/attack/defense/scientific cell。未 merge/push/upload/
publication。

## Interpretation、residual risk 与 final verdict

允许解释：R10 的 ready/ACK ordering、full-validation-before-arm、pre-ACK no-fork结构、
child primary traceback/cleanup-note收集、真实 forced-error/forced-hang路径、CPython 3.11
endpoint/sentinel close结构以及normal two-epoch语义均有实质静态/authored改进；exact ownership、
forbidden blobs、Jobs 348557/348818负面证据和Job 349653 bounded attribution保持不变。

禁止解释：不得称 leader-dead all-path group cleanup、arbitrary-size primary report传输或
per-PID exact gone proof已关闭；不得称新增hostiles已执行或 post-O-059 integrated suite已PASS；
不得称 production/full-cache/full-trainval/100/1000/profile/metric readiness，亦不得推断
mAP/NDS、fusion gain、FL、attack/defense、generalization或publication evidence。

Residual risk还包括所有新增test均未执行、full `t1.v2` cache仍不存在、integrated fp16/
concurrency/EMA/official eval仍无新runtime evidence，以及private multiprocessing API只绑定
当前CPython 3.11。应先修复leader-dead group deadline/identity hostile与Queue drain-before-
join，再从exact new SHA独立复审；之后才可讨论任何新的bounded runtime proposal。

**CHANGES-REQUESTED for S07-B candidate
`8469eb4944f164f5bd2fa1aa833ea4df0acf04b3`.**

本 verdict 不授权任何 runtime proposal、compute、merge、push、upload 或科学解释。

---

# S07-B-R12 独立复审 — O-069 synchronous-result / leader-dead identity remediation

## Findings（按严重性排序）

### 无 P0/P1/P2/P3 finding

本轮没有根据 HANDOFF 总结直接给 PASS，而是逐行核对 exact candidate
`c53117a889987c3070b60817e52bdb4aac4c9098` 的 Pipe 方向、每种 mode 的
send/recv/join 顺序、leader-dead 与 live-leader 控制流、Linux procfs identity、
TERM/KILL deadlines、endpoint/sentinel ownership 和 exception preservation。未确认新的
correctness、liveness、identity、cleanup、evidence-reachability 或权限 finding；R11 的
两项 P2 与一项 P3 均在当前候选的静态/authored 范围内关闭。

1. **Queue backpressure P2 — CLOSED STATICALLY/AUTHORED，未执行 runtime。**
   child-produced `Queue`、feeder、`join_thread()` 和 private queue endpoints 已完全移除。
   冻结 CPython 3.11 POSIX `Pipe(False)` 的第一端只读、第二端只写；candidate 正确赋予
   parent `result_receiver` 与 child `result_sender`
   (`fl_v3/tests/test_nuscenes_zip_dataset.py:780-799`)。ACK 后 parent 严格执行
   `poll → recv complete frame → join producer` (`:835-862`)。normal/error/hang/
   leader-exit 各只有一条同步 result frame；raw descendant 关闭继承的 result/control
   connection，不依赖 EOF 区分消息。forced-error 同步发送 2 MiB padding，test 要求完整
   length/content、primary traceback、cleanup notes、exact worker identities 和无 live child
   (`:630-652,1173-1207`)。旧 producer-join/feeder-backpressure 等待环不存在。

2. **leader-dead remaining-group P2 — CLOSED STATICALLY/AUTHORED，未执行 runtime。**
   resistant path 在 `fork()` 前安装 `SIG_IGN`，child 从创建时即继承，parent branch随后
   恢复原 disposition (`:483-512`)。helper 发送 `(pid,starttime,pgid,sid)` 后关闭
   endpoints，以 `os._exit(0)` 退出且不 `waitpid` descendant (`:547-571`)。parent 先完整
   recv，再 join/probe/reap direct helper，证明 helper exact identity gone、group 与 exact
   descendant仍 live且SID/PGID均为 armed group (`:858-885`)。finally 将 direct-child reap
   与 group wait 分离：TERM 后 bounded join，再独立 poll group+all exact identities；仅在
   TERM deadline 后仍存在目标才 KILL，KILL 后再 bounded poll五秒，最后才关闭 Process
   (`:924-1007`)。leader-dead hostile要求TERM不足、KILL后group/descendant original
   instance均gone；existing live-leader hostile继续要求TERM后helper exact identity live、
   KILL后gone (`:1209-1291`)。

3. **bare PID / PID reuse P3 — CLOSED FOR ORIGINAL INSTANCES。**
   `/proc/<pid>/stat` parser 使用最后一个 `)` 结束可能含空格/括号的 `comm`；其后
   `remaining[0]` 是field 3，因此field 22 `starttime` 位于index 19 (`:423-454`)。
   hostile含嵌套括号并得到exact field-22值 (`:1246-1253`)。helper在ready/ACK前、
   DataLoader workers在live/shutdown前、raw descendant在record前捕获 `(pid,starttime)`；
   final audit和deadline poll均比较exact tuple。ESRCH或同PID不同starttime都表示original
   instance gone；同starttime zombie则保守地保持alive直到被reap，不会false PASS。

4. **FD、exception 与 retained gates — CLOSED STATICALLY/AUTHORED。**
   parent在`Process.start()`后立即关闭child-control/result-sender copies；raw descendant
   与helper各自关闭继承/拥有的connection (`:475-501,653-655,805-809`)。parent finally
   关闭control/result receiver，仅在child不alive后`Process.close()`，并检查四个Pipe FD
   与Process sentinel均EBADF (`:1069-1094`)；未发现方向反转、double-close或cleanup
   masking。parent cleanup errors仍作为notes附到primary；child traceback/notes/worker
   identities完整进入forced-error report。ready/ACK full-validation-before-arm、pre-ACK
   no-fork、parent PID/SID/PGID不变、normal fork/spawn两完整persistent epochs、reopen/
   read/archive assertions与explicit worker shutdown均保留。

## Review identity、prefix、topology 与 ownership

- Session：`S07-B-R12`；`APPROVED_COMPUTE: none`。
- Remediation base：`8469eb4944f164f5bd2fa1aa833ea4df0acf04b3`。
- Candidate：`c53117a889987c3070b60817e52bdb4aac4c9098`。
- Test commit：`2497ac11e807e5b223bfa0eaa2537fcbde1aec88`，parent exact为base，
  只改`fl_v3/tests/test_nuscenes_zip_dataset.py`。
- Handoff-only commit：`c53117a...`，parent exact为test commit。
- Startup/import HEAD：`8e294576d624abc2c0681c44657e000ccbf62f3f`，branch
  `codex/s07-b-r12-integrated-cl-stack-review`，startup clean。
- 权威 canonical：`4026fe4dfe0bb93f59852f4b7d604ce6e2f7aef9`（O-070）。

Candidate之后到startup的四个commit仅修改`S07/REVIEW.md`；candidate是startup ancestor，
其余tree一致。追加R12前exact R11 prefix为Git blob
`4e0226718109e193bb09993db085422106b1dccc`、size `191368`、SHA-256
`cc8922192125b054280e5b11760f801997adbb201ea2f7bd6e2564b55e0c1104`；本段只追加在
这些bytes末尾。

`8469eb4..c53117a`精确两路径：test为`419/106`、blob
`f8d4f0ee7a9ca834cbf1105562cf0c8fccb5ec38`、SHA-256
`07c4c2159efbdf4fb18a95960d4ff7d8d17ac823c88f14d9184eb1cc041e3f09`；HANDOFF只追加
O-069 record，SHA-256
`59c2e190e3ceaefff927a371d170c62ffb7ed8c1c7d8e2715475792ae8bdb741`。
`git diff --check`无warning。Forbidden blobs在base/candidate相同：
`training/loop.py=881c070b...`、`training/tasks.py=86ab9d05...`、
`dataset.py=afd2707d...`、`RUN_REQUEST.md=efa5ce78...`、
`RESULTS.md=b3b80625...`、runtime launcher=`1e182ebc...`。

## Gate matrix

| R11/O-069 requirement | R12 verdict |
|---|---|
| one-way result Pipe direction/ownership | **CLOSED STATICALLY** |
| complete 2 MiB result recv before join | **CLOSED STATICALLY/AUTHORED** |
| all authored modes无send block/错序/EOF依赖 | **CLOSED STATICALLY** |
| real leader-dead SIGTERM-resistant orphan | **CLOSED STATICALLY/AUTHORED** |
| TERM insufficient，KILL+deadline gone | **CLOSED STATICALLY/AUTHORED** |
| existing live-leader forced-hang | **RETAINED** |
| final-`)` field22 parser与PID reuse | **CLOSED STATICALLY/AUTHORED** |
| zombie/orphan semantics | **CONSERVATIVE / ACCEPTABLE** |
| direct reap与group/identity deadlines分离 | **CLOSED STATICALLY** |
| Pipe/control/Process/sentinel closure | **CLOSED STATICALLY/AUTHORED** |
| primary traceback/notes/worker identities | **CLOSED STATICALLY/AUTHORED** |
| ready/ACK、pre-ACK、normal two epochs | **RETAINED** |
| exact two paths/forbidden evidence | **PASS** |

## Checks actually run 与 explicit NOT RUN

实际只执行Git/hash/text/static/artifact读取：startup/ancestry/review-only topology、prefix
blob/size/hash、AGENTS/env/三份canonical/O-070、完整S07 HANDOFF/REVIEW、actual diff、
冻结CPython 3.11 multiprocessing source、candidate blobs/hashes、forbidden blobs和
`git diff --check`；逐行trace上述transport/process/group/identity/FD/exception路径。

为定位stdlib source，曾执行一次`python3 -c 'import sysconfig; ...'`；它只导入宿主
Python 3.9 stdlib `sysconfig`，没有导入project、Torch、NumPy或数据/模型代码。冻结
Arrhenius CPython 3.11 source随后均直接按文件读取。此命令不是candidate runtime evidence，
也不应被误报为完全零import。

明确 **NOT RUN / NO IMPLIED PASS**：pytest、pycompile、AST/source-text execution、project/
package import、Torch/NumPy/CUDA/spconv/cumm、data/cache/model/checkpoint、任何worker/fork/
spawn runtime、Slurm/srun/GPU、full `t1.v2`、100/1000 steps、profile、mAP/NDS、DDP、
matrix、seed/rerun、FL/attack/defense/scientific cell。未merge到`v3-ad-perception`、push、
upload或publication。

## Residual risk、interpretation 与 final verdict

全部新hostile仍只是authored/static evidence。leader-dead test绑定Linux `/proc`、`fork`、
`setsid`、`killpg`及外部init/subreaper在五秒内回收被KILL的orphan zombie；raw-fork
descendant可能继承冻结CPython spawn内部sentinel writer，使timeout join保守等到deadline，
随后由`is_alive()`/`waitpid` probe回收leader，但当前bounded路径不会因此死锁或误判。
DataLoader worker发现和`_shutdown_workers`仍绑定当前PyTorch private lifecycle。numeric PGID
在liveness probe与signal间理论上仍有极窄reuse TOCTOU；当前leader-dead/live-leader/
pre-ACK authored paths在实际signal时至少有一个verified original identity存活，所以本轮
未确认误杀，但不能外推为任意Linux/PID压力下的形式化无误杀证明。

允许解释：R11的Queue backpressure、leader-dead remaining-group与bare-PID findings已在
candidate静态/authored层关闭；ready/ACK、live-leader、forced-error、normal two epochs和
exact ownership未回归。S07-B重新达到**code-level/static-review PASS**，S00可决定是否
准备一个新的、独立、immutable、bounded runtime proposal。

禁止解释：没有runtime proposal获批/执行，没有pytest/integrated GH200、production、
full-cache/full-trainval、checkpoint、performance或scientific PASS；不得声称100/1000、
mAP/NDS、fusion gain、FL、attack/defense、generalization或publication evidence。本verdict
不授权compute、merge、push、upload或publication。

**PASS at code-level/static-review scope for S07-B candidate
`c53117a889987c3070b60817e52bdb4aac4c9098`.**

该PASS只关闭exact O-069/R11 test-harness findings并保留上述Linux/CPython/runtime residual；
它不是post-O-059 integrated runtime PASS，也不自动允许任何后续job。

---

# S07-B-R13 独立复审 — Job 352105 harness-confound remediation

## Findings（按严重性排序）

### P2 — leader-exit hostile没有验证“reaped wait status来自SIGKILL”，交付却宣称已断言exact wait status

`fl_v3/tests/test_nuscenes_zip_dataset.py:1378-1389` 只验证发送记录为
`[SIGTERM, SIGKILL]`、SIGKILL后group/identity消失，以及`waitpid`返回的raw status
大于等于零。这个谓词对正常退出status `0`、其他signal或非预期退出同样成立；因此它不能证明
被reap的exact descendant确实在TERM无效后由预期SIGKILL终止。实现的identity、PPID、
bounded `waitpid(exact_pid,WNOHANG)`与TERM/KILL/reap顺序在静态路径上是正确的，但hostile
缺少`os.WIFSIGNALED(status)`及`os.WTERMSIG(status) == signal.SIGKILL`的最终语义断言。

这也使`HANDOFF.md:2354`“asserts ... exact wait status”的交付陈述强于实际test。请保存
该exact wait status并断言WIFSIGNALED/WTERMSIG；同时让static contract或source-text检查覆盖
这一谓词。修复只能获得authored/static closure，仍需后续独立runtime evidence。

### P3 — 五个EXIT trap在primary已失败时可静默吞掉path/identity cleanup failure

五个launcher的`cleanup_job_tmp`都在path regex、exact parent、directory/no-symlink或
device/inode检查失败时仅设置`cleanup_status=1`。当主流程status已经非零时，trap按要求保留
原status，但没有为这些predicate failures输出确定性stderr或durable record；`if`条件失败本身
也不打印诊断。因此例如pytest timeout同时发生temp path replacement/removal时，launcher只返回
原timeout code，cleanup breach可完全不可见。`rm`/`stat`自身报错并不覆盖纯predicate mismatch。

请在所有五个trap中对每个未获准删除的原因至少输出明确stderr（不改变非零primary status；
primary为零时仍因cleanup failure转为非零），并把该observability token纳入static contract。
无需将temp内容复制进durable artifacts，但不能静默声称cleanup contract已完成。

### P3 — durable HANDOFF把已经提交的delivery仍写成“uncommitted documentation diff”

`HANDOFF.md:2380-2383`称code SHA durable而“this updated lifecycle prose remains an
uncommitted documentation diff”。实际delivery commit
`34f07994a4b3de62c7c1331d98ff03dbba98de2e`正是把HANDOFF/RUN_REQUEST/RESULTS作为
durable commit提交，review baseline `8089a61...`的parent也精确为该delivery。这是可复现状态
错误，不是措辞偏好。请改为记录exact delivery SHA/parent与三路径ownership；同时把
“saved state is restored on every path”改为“restore is attempted on every controlled path”，
因为restore syscall本身失败时只能保留additive cleanup evidence，不能声称状态已恢复。

## 已确认没有finding的部分

1. 五个launcher均验证numeric job ID与12-hex executable prefix，使用随机
   `/tmp/flv3-s07b-...XXXXXX`、运行时长度上限48、mode 0700、anchored regex、exact
   `/tmp` parent、no-symlink及device/inode identity；环境激活后重新export
   `TMPDIR/TMP/TEMP`，formal artifacts仍位于独立output root。未发现output-local temp
   回归、非anchored `rm -rf`或source-attestation绕过。
2. Linux `prctl` constants/ABI在目标aarch64 LP64上匹配；subreaper仅在
   `post_ack_leader_exit`启用，且在helper start前生效。pytest parent成为nearest
   subreaper后，代码要求exact `(pid,starttime,PPID)`，只调用
   `waitpid(pid,WNOHANG)`，未使用`waitpid(-1)`；primary与cleanup exception仍按notes/
   `BaseExceptionGroup`保留，FD/Process/sentinel与parent SID/PGID predicates未回退。
3. Job 352105原始证据支持delivery的负面解释：manifest 46/46通过；summary为
   diagnostic/artifact complete但suite FAIL（2 PASS / 2 FAIL / 5 timeout）；旧路径确为
   106 bytes；post-ACK log有四条`AF_UNIX path too long`；leader descendant被outer
   supervisor PID `401513`收养。历史candidate/executable/hash与negative shape没有被改写。

## Review identity、prefix、topology 与 ownership

- Session：`S07-B-R13`；`APPROVED_COMPUTE: none`。
- Exact code commit：`26cffb02ced50b07f93021bc48310efb68b178a9`，parent
  `f8b781dd919443fab0d9c2e6e28c0207182800d5`。
- Delivery：`34f07994a4b3de62c7c1331d98ff03dbba98de2e`，parent exact为code commit。
- Review startup/import：`8089a61edb6d2a8a61df3c6271cdfb7f9cfe2501`，branch
  `codex/s07-b-r13-runtime-harness-review`，startup clean。
- 追加前R12 prefix：Git blob `7fbd6d7f2559ee36eab57295a217b1816655f214`、size
  `200971`、SHA-256
  `d457f19bdd87bc8cfbed54f337674b548a3be520bd17003a0dd03bc0fe48f0f1`；
  R13只追加在这些exact bytes之后。

`f8b781d..26cffb0`精确七路径，`439/18`；无production source。七个blob/SHA-256与
HANDOFF/RESULTS/RUN_REQUEST表格逐项一致。`26cffb0..34f0799`只改三份S07 durable docs。
`git diff --check f8b781d..34f0799`无warning。

## Checks actually run 与 explicit NOT RUN

实际执行：startup branch/HEAD/parent/clean与R12 prefix blob/size/SHA-256；commit ancestry、
name-status/numstat、七个blob及SHA-256；逐行读取test/五launcher/static checker/delivery docs；
六个相关shell `bash -n`；`python3 -m py_compile`仅针对changed test；launcher-contract-only
checker（`short TMPDIR contract: 5 launchers OK`）；stdlib `compile()`检查19个shell Python
heredoc；`git diff --check`；Job 352105全局`sha256sum -c` 46/46、summary/identity/config/
node logs与supervisor JSON的直接读取。py_compile只生成ignored cache且review tree仍clean。

明确 **NOT RUN / NO IMPLIED PASS**：project/package import、pytest、任何multiprocessing/
fork/spawn runtime、Torch/NumPy/CUDA/spconv/cumm、data/cache/model/checkpoint、Slurm/srun/GPU、
full `t1.v2`、full trainval、100/1000 steps、profile、metrics、DDP、matrix、seed/rerun、
FL/attack/defense/scientific cell。未merge/push/upload/publication。

## Interpretation、residual risk 与 final verdict

允许解释：short-temp与nearest-subreaper remediation方向在static层成立；Job 352105是
artifact-complete但harness-confounded的negative diagnostic，不能归因production candidate。

禁止解释：不得称leader-exit exact SIGKILL/reap semantics已被hostile完整锁定、所有cleanup
failure均可观察、durable handoff状态完全准确，亦不得称任何corrected runtime、integrated
suite、production/full-data/checkpoint/performance/scientific gate PASS。

修复上述三项后应从exact new worker/delivery SHA做独立复审。即使复审static PASS，也只能由
S00另行冻结并批准全新bounded runtime request；本review不授权compute、merge、push或upload。

**CHANGES-REQUESTED for S07-B delivery
`34f07994a4b3de62c7c1331d98ff03dbba98de2e` / code candidate
`26cffb02ced50b07f93021bc48310efb68b178a9`.**

---

# S07-B-R14 独立复审 — O-079 R13 harness-evidence remediation

## Findings（按严重性排序）

### P3 — exact delivery 再次把已提交的三份 durable docs 写成“pending/requires delivery”

O-079 documentation delivery 已经是 exact commit
`e3122dbccdd252a6d89f1a4fe339b9043fe19884`，其唯一parent是code
`56c74de5bdf5463fdd6ab1a623ab0f92a35871ae`，且只修改
`HANDOFF.md`、`RUN_REQUEST.md`、`RESULTS.md`。但这三份已提交文件仍分别称：

- `HANDOFF.md:2431-2433`：lifecycle update “remains a documentation diff pending
  delivery”；
- `RESULTS.md:721-723`：lifecycle update “still requires delivery”；
- `RUN_REQUEST.md:1351-1353`：在当前delivery中仍要求“A documentation delivery”。

这不是历史段落对旧candidate的准确陈述，而是O-079新段落对自身当前状态的陈述；因此exact
delivery一创建就使其不可复现。它直接复现R13第三项finding，只是从旧delivery
`34f07994...`迁移到了新delivery `e3122db...`。请追加一个durable docs-only commit，明确记录
O-079 code SHA、exact delivery SHA/parent、三条delivery-owned路径，以及“本delivery已提交、
R14 review-only且未合并、仍需新的independent acceptance/runtime request”；不要再次把同一份
已提交文档称作uncommitted/pending delivery。该修复无需改code、launcher或test，也不授权
runtime。

### 无 P0/P1/P2 finding；R13前两项在authored/static范围内关闭

1. **exact SIGKILL wait-status predicate — CLOSED STATICALLY/AUTHORED。**
   leader-exit cleanup从exact `(pid,starttime)` key取得raw `waitpid` status，并记录raw与
   decoded signal；hostile明确要求`os.WIFSIGNALED(reaped_status)`及
   `os.WTERMSIG(reaped_status) == signal.SIGKILL`
   (`test_nuscenes_zip_dataset.py:1399-1407`)。static checker也锁定这两个谓词。没有把普通
   exit status、其他signal或仅“status >= 0”误当SIGKILL证据。

2. **五launcher cleanup observability/status — CLOSED STATICALLY/AUTHORED。**
   五个EXIT trap对`path_pattern/dirname/symlink/directory/stat/device_inode/rm`各有唯一固定
   `S07B_TMP_CLEANUP_FAILURE:<launcher> reason=<reason>` stderr token；message不含temp path，
   `stat`与`rm`原始stderr均被抑制。每个trap保存entry `$?`，仅当primary为0且cleanup失败时
   将状态改为1；primary非零时保留原值。实际脚本与contract checker一致，未发现纯predicate
   mismatch继续静默、cleanup覆盖primary或primary成功时cleanup失败仍返回0。

3. **short temp / identity / rebind — RETAINED STATICALLY。**
   五launcher继续使用numeric job ID、12-hex executable prefix、随机短`/tmp` mode-0700目录、
   anchored regex、exact parent、no-symlink与captured device:inode；trap仍在assertions之前安装，
   environment activation之后重新export `TMPDIR/TMP/TEMP`。formal output仍与job temp分离。

4. **subreaper/reap/exception semantics — RETAINED STATICALLY。**
   O-079未改nearest-subreaper的enable/restore控制流、exact starttime+PPID adoption、bounded
   `waitpid(exact_pid,WNOHANG)`、TERM/KILL顺序、primary exception与additive cleanup notes；
   文件中不存在`waitpid(-1)`。文档已正确降级为“restore is attempted on every controlled
   path”，passing hostile才可证明该次restore成功。

5. **Job 352105 negative与compute boundary — RETAINED。**
   O-079没有改生产source、旧executable/artifact或Job 352105的2 PASS / 2 FAIL / 5 timeout
   negative解释，也没有冻结新command或批准compute。short-TMP/subreaper与本轮predicate仍只有
   authored/static evidence。

## Review identity、prefix、topology 与 ownership

- Session：`S07-B-R14`；`APPROVED_COMPUTE: none`。
- Exact O-079 code：`56c74de5bdf5463fdd6ab1a623ab0f92a35871ae`，parent exact
  `34f07994a4b3de62c7c1331d98ff03dbba98de2e`。
- Exact delivery：`e3122dbccdd252a6d89f1a4fe339b9043fe19884`，parent exact为code。
- Review startup/import：`4891fb59397275a88211b9ed4100e3e85144ed35`，parent exact为
  delivery，branch `codex/s07-b-r14-runtime-harness-review`，startup clean。
- 追加前R13 prefix：Git blob
  `a5ac4a62e31e431d0cf5f5729ac439f205ead4c8`、size `207926`、SHA-256
  `a6b4de09ad5fcecd8442167e8455f41294160347c67bed9480ae485480dd4140`；R14仅追加在
  这些exact bytes之后。

`34f0799..56c74de`精确七个test/launcher路径、`163/59`，无production source；七个Git
blob与SHA-256逐项匹配三份delivery表。`56c74de..e3122db`精确三份S07 durable docs、
`143/12`。`git diff --check 34f0799..e3122db`无warning。review commit与R13历史均未合并进
implementation lineage。

## Checks actually run 与 explicit NOT RUN

实际执行：startup branch/HEAD/parent/clean；R13 prefix blob/size/SHA-256；ancestry、commit
parents、name-status/numstat；七个candidate blob/SHA-256；逐行读取changed test、五launcher、
static checker与三份delivery docs；六个相关shell `bash -n`；launcher-contract-only checker
（`short TMPDIR contract: 5 launchers OK`）；changed-test source-text `compile()`；
`shellcheck -S error`；`git diff --check`；以及waitpid/source-text与stale-delivery wording搜索。
所有检查后review tree仍clean。

明确 **NOT RUN / NO IMPLIED PASS**：project/package import、pytest、pycompile、任何
multiprocessing/fork/spawn runtime、Torch/NumPy/CUDA/spconv/cumm、data/cache/model/checkpoint、
Slurm/srun/GPU、full `t1.v2`、full trainval、100/1000 steps、profile、metrics、DDP、matrix、
seed/rerun、FL/attack/defense/scientific cell。未merge、push、upload或publication。

## Interpretation、residual risk 与 final verdict

允许解释：R13的SIGKILL wait-status与cleanup-observability findings在exact O-079 code的
static/authored层关闭；short-temp、post-env rebind、exact subreaper/reap、primary-error与历史
negative evidence未回归。

禁止解释：durable delivery state尚不准确；没有corrected pytest/multiprocessing/GH200 runtime、
integrated suite、production/full-data/checkpoint/performance/scientific PASS。不得从本review推导
任何新的compute、merge、push、upload或科学结论。

请仅修复三份durable docs的当前delivery状态后，从exact new docs-only delivery SHA进行独立
复审。即使后续static review PASS，也必须由S00另行冻结并批准exact bounded runtime request。

**CHANGES-REQUESTED for S07-B delivery
`e3122dbccdd252a6d89f1a4fe339b9043fe19884`; O-079 code candidate
`56c74de5bdf5463fdd6ab1a623ab0f92a35871ae` is accepted only at
code-level/static-authored scope.**
