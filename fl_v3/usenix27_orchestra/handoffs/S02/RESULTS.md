# S02 RESULTS — focused CL P0 correctness validation

## Overall result

**GPU forward/backward remediation PASS; manual parser remediation PASS; initial
failed job preserved.** Initial Slurm Job
`335565` ran the twelve approved synthetic CPU tensor tests and reported
`12 passed in 17.31s`, but failed `1:0` afterward because the launcher misread a
`testsuites` JUnit root and exited before its final checksum manifest. That job
remains overall FAILED and its missing artifact is not rewritten.

After an explicit new S00 approval, manual parser-remediation Job `335578` ran the
unchanged twelve tests and scope from executable `840e8bee...`; it completed
`COMPLETED 0:0`, reported `12 passed in 19.86s`, aggregated JUnit to 12/0/0/0, and
verified the complete in-job checksum manifest. No retry/requeue/resubmission or
follow-on occurred after either exact submission.

Independent S02-R then requested only the missing canonical GPU forward/backward
evidence. Exact synthetic Job `336713` completed `0:0`, passed its sole CUDA test,
matched one-GPU allocation and all cap/isolation/empty/gradient assertions, and
verified source plus final artifacts in-job. Limited evidence re-review remains
required before independent/integration PASS.

## Approved identity and submission

- S00-audited request SHA-256:
  `60b0b923d527b60a34449ddb7d24678e85e68ca187d453c18809368637ed50c9`.
- Exact executable HEAD:
  `a877ea0ecdc510350e03843ec66b9a679cdb6f37`.
- Implementation commit:
  `65c83c077210469861ba722a285ab1e58e6d719f`.
- Runtime source-state SHA-256:
  `5ff316b81233d4a367ded2928ebacb2f90ae240485003af2f701c54f22c560fa`.
- Branch: `codex/s02-cl-p0-correctness`.
- Before submission, HEAD/branch/request hash matched, tracked diff was empty,
  output was absent, and no active `flv3_s02*` job existed.
- Submitted once as Job `335565`; approval consumed.

## Scheduler and resource result

| Field | Value |
|---|---|
| job / name | `335565` / `flv3_s02_cpu_tests` |
| submit / start / end | `2026-07-11T18:12:44` / `18:12:45` / `18:14:20` Europe/Stockholm |
| node / machine | `n507` / `aarch64` |
| state / exit | `FAILED` / `1:0` |
| elapsed / limit | `00:01:35` / `00:10:00` |
| allocation | one node, one `nvidia_gh200_120gb`, four CPUs, 5,836 MiB memory |
| actual elapsed allocation | about `0.0264` GPU-hours |
| restarts | `0` |
| batch MaxRSS / MaxVMSize | `540M` / `6410816K` |
| batch disk read / write | `57.54M` / `0.28M` |
| batch TotalCPU | `00:05.863` |

`CUDA_VISIBLE_DEVICES` was the empty string before Python imports and pytest. No
GPU tensor path, forward/backward model smoke, optimizer step, data load, profile,
or metric ran.

## Pytest and JUnit evidence

Pytest stdout:

```text
............                                                             [100%]
12 passed in 17.31s
```

The JUnit structure independently parsed after termination was:

```text
root: testsuites, attributes: {name: "pytest tests"}
child: testsuite, tests=12, failures=0, errors=0, skipped=0, time=17.306
```

All approved cases are present:

- four official Gaussian root numerical goldens, including the discriminating
  `(4,8): official radius 2 versus old mixed radius 4`, `(10,20): 6 versus 10`,
  and `(6,16): 4 versus 8` cases;
- exact official float32 radius-2 patch and clipped maximum-overlay heatmap;
- exact `CenterPointLoss.build_targets` heatmap and GT-order invariance;
- per-sample cap/selection/occupancy/truncation diagnostics;
- B=1 versus B>1 sample isolation and batch permutation;
- input-point permutation under simultaneous point and pillar overflow;
- empty batch and empty-sample behavior;
- the two retained PointPillars point-permutation tests from
  `test_model_determinism.py`.

## Runtime identity and source verification

`execution_identity.json` records:

- Git SHA `a877ea0ecdc510350e03843ec66b9a679cdb6f37`;
- source state `5ff316b81233d4a367ded2928ebacb2f90ae240485003af2f701c54f22c560fa`;
- CPython `3.11.15`, NumPy `1.26.4`, pytest `9.1.1`, Torch `2.11.0+cu128`;
- host `n507`, machine `aarch64`, and empty `CUDA_VISIBLE_DEVICES`;
- Slurm Job `335565`.

After termination, `sha256sum -c runtime_source_sha256s.txt` passed for all 16
listed runtime inputs, including the two changed model files, focused tests,
readonly BEV/runtime dependencies, effective test/dependency configuration, and
the launcher itself.

## Raw artifacts and hashes

Output root:

`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_a877ea0ecdc5`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 747 | `0d8f227349c14d2d0d70a64a7a9f52d140b0bf4342670001afa54c0a5d72472e` |
| `runtime_source_sha256s.txt` | 1,620 | `5ff316b81233d4a367ded2928ebacb2f90ae240485003af2f701c54f22c560fa` |
| `pytest.log` | 100 | `4b3433f6e44e0605b5b292df5b1d77de14cf8870eb955f146f6ebfe1dba2e59d` |
| `pytest.junit.xml` | 1,895 | `7413f7555bc82a3d3593bbaf234c9c4725d59bb4a6337ef4f32d8201c6b20ebf` |

Logs:

- stdout:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_cpu_tests_335565.out`,
  100 bytes, SHA-256
  `4b3433f6e44e0605b5b292df5b1d77de14cf8870eb955f146f6ebfe1dba2e59d`;
- stderr:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s02_cpu_tests_335565.err`,
  SHA-256
  `b2a3b12ebc6c8dc943472d3ec956e92285730be0221af5203162978ea64a54b9`.

Stderr contains the normal module-purge notice followed by the exact negative:

```text
unexpected JUnit counts: tests=0 failures=0 errors=0 skipped=0
```

No `sha256sums.txt` exists: the fail-closed parser exited before that final stage.
The hashes above were computed read-only after scheduler termination and are not
misrepresented as an in-job checksum-manifest PASS.

## Negative result and disposition

The failure is in the handoff-owned validation launcher, not in a model assertion:
the parser used `ET.parse(...).getroot().attrib`, but the root was `testsuites` and
the counts were on its child. The exact request required scheduler `COMPLETED`,
exit `0:0`, and in-job final checksum verification, so the overall job gate is
**FAIL** despite twelve passing tests.

Under the first approval, S02 stopped without modifying or retrying. The separate
manual remediation below occurred only after a new exact S00 authorization; it is
not reclassified as an automatic retry.

## Manual parser-remediation Job 335578 — PASS

### Approval and immutable identity

- S00-audited second request SHA-256:
  `48aa43079bcca7bbc9f9005862149d99968a76b149c6f3f7482f37bd8e125a0b`.
- Exact executable HEAD:
  `840e8bee8d1157c71b7752d3937c6cb8e75201e7`.
- Implementation remained:
  `65c83c077210469861ba722a285ab1e58e6d719f`.
- Launcher SHA-256:
  `35798c3956e1cb4fcf54288f34ead04687d2b22d2dcdff4186b865d5261b452b`.
- Runtime source state:
  `2ff7d74246e55332305e92a83dc028a42ce3c1e60993c28c24ece868784e580a`.
- New output was absent and no S02 job was active before exact-once submission.

The only executable delta from the preserved negative package was the JUnit suite
aggregator. Model source, tests/nodes, expected count, data scope, resources, and
CUDA-hidden CPU execution remained unchanged.

### Scheduler and resources

| Field | Value |
|---|---|
| job / name | `335578` / `flv3_s02_cpu_tests` |
| submit / start / end | `2026-07-11T18:20:56` / `18:20:57` / `18:22:30` Europe/Stockholm |
| node / machine | `n534` / `aarch64` |
| state / exit | `COMPLETED` / `0:0` |
| elapsed / limit | `00:01:33` / `00:10:00` |
| allocation | one node, one `nvidia_gh200_120gb`, four CPUs, 5,836 MiB memory |
| actual elapsed allocation | about `0.0258` GPU-hours |
| restarts | `0` |
| batch MaxRSS / MaxVMSize | `612M` / `6935104K` |
| batch disk read / write | `65.59M` / `0.23M` |
| batch TotalCPU | `00:06.434` |

Cumulative S02 elapsed allocation for both exact jobs was about `0.0522`
GPU-hours. Both hid CUDA before Python/Torch imports and ran no GPU tensor/model
path.

### Test, JUnit, source, and checksum result

```text
............                                                             [100%]
12 passed in 19.86s
```

Independent JUnit parsing found a `testsuites` root and aggregate counts
`tests=12, failures=0, errors=0, skipped=0`, with all twelve approved testcase
nodes present. Execution identity records CPython `3.11.15`, NumPy `1.26.4`,
pytest `9.1.1`, Torch `2.11.0+cu128`, exact Git/source identities, host `n534`,
aarch64, Job `335578`, and empty `CUDA_VISIBLE_DEVICES`.

All sixteen source entries verified against the immutable worktree. The final
in-job checksum manifest exists and `sha256sum -c` passed for every listed
artifact.

### Raw artifacts and hashes

Output root:

`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_cpu_tests_840e8bee8d11`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 747 | `c2663500bbfb63acb25a34eb589da8312f4382ceacec3cbabe0e58ffbeb86d01` |
| `runtime_source_sha256s.txt` | 1,620 | `2ff7d74246e55332305e92a83dc028a42ce3c1e60993c28c24ece868784e580a` |
| `pytest.log` | 100 | `8a9822a691fbe40121b9c0412f86f2a2810e58023be05f03cc61e0d6afc482fc` |
| `pytest.junit.xml` | 1,895 | `21c48e0af6352459b4407ee4cb57b40d773b6d3fcf744690dd0615319770a256` |
| `sha256sums.txt` | 759 | `ea14f3418876135ad4a1b57b9de4c8897c0c3a2fea39eee0e18c45e549c8383a` |

Logs:

- stdout: 611 bytes, SHA-256
  `375479797b332715a3d5a49803adbe32b43479428d6b1ca178352cc424333e4a`;
- stderr: 123 bytes, SHA-256
  `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`.

Stderr contains only the normal module-purge notice. Stdout contains the pytest
PASS plus four in-job checksum `OK` lines.

## Interpretation limits

Allowed:

- the exact twelve listed CPU tensor tests executed and passed on the recorded
  aarch64 dependency environment in both jobs;
- JUnit independently confirms 12/0/0/0 and contains every approved testcase;
- all runtime-source identities match the immutable executable commit;
- the negative post-pytest parser behavior is fully preserved.
- the manual parser-only remediation Job 335578 completed `0:0` with final in-job
  checksum-manifest verification.

Forbidden:

- calling Job `335565` an overall PASS or `COMPLETED` job;
- claiming in-job final checksum-manifest verification;
- GPU forward/backward correctness, mini/trainval readiness, performance/memory
  quality, mAP/NDS, fusion gain, FL, attack/defense, generalization, scientific, or
  publication conclusions;
- any inference that a retry or launcher change is authorized.

## GPU forward/backward remediation Job 336713 — PASS

### Approval and immutable identity

S00 approved one evidence-only submission against the prior independent
`CHANGES-REQUESTED` review. The immutable tuple was:

| Field | Value |
|---|---|
| executable / tree | `b6f815d01b5374eb4b922559b83b1d28c208e2b9` / `8ee64fbde2d022f468def7d24cddf0f87e08a3fb` |
| unchanged implementation | `65c83c077210469861ba722a285ab1e58e6d719f` |
| review / REVIEW hash | `fb17da3ea55a93d7709f6a2b5f6e4bb6adc0bf7e` / `75b6a5ed589c1f29ba847750a915732be8826562c055a7fa1cecd5a749e63497` |
| RUN_REQUEST SHA-256 | `aaec2fdf8662edcccf1dde6cec68737fdadf18461c087c46eee84ae312ce3769` |
| launcher SHA-256 | `78715618936c1469da37d3bbe5582ff84964a04bb9c8521bbdc8573023a797ed` |
| test SHA-256 | `f45cc992bde4b1353713fdf906578c22a684c8c0b000f8a4eb115199288e5fee` |
| runtime source state | `5f5cc459a149483120bafc91cacb1c8a19bf500c45844a60891fe47ee28e1e49` |

The final clean branch/HEAD/hash/root/active-job preflight matched. The execution
snapshot was created with `git archive` at
`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/snapshots/s02_gpu_fb_b6f815d01b53`.
The compute job's command and working directory were both inside that snapshot; it
did not access the `/home` linked worktree.

### Scheduler and actual allocation

| Field | Value |
|---|---|
| job / name | `336713` / `flv3_s02_gpu_fb` |
| submit / start / end | `2026-07-11T18:48:05` / `18:48:06` / `18:49:24` Europe/Stockholm |
| state / exit | `COMPLETED` / `0:0` |
| elapsed / limit | `00:01:18` / `00:10:00` |
| node / machine | `n580` / `aarch64` |
| requested TRES | one GH200, four CPUs, 16 GiB, one node, billing one |
| allocated TRES | exactly matched requested TRES |
| sharing / restarts | `OverSubscribe=OK` / `0` |
| batch MaxRSS / MaxVMSize | `36M` / `6738496K` |
| batch disk read / write | `65.52M` / `0.26M` |
| batch TotalCPU | `00:07.844` |

The scheduler command contained no `--nodes=1` or exclusive request. Runtime
identity records `SLURM_GPUS_ON_NODE=1`, `SLURM_JOB_GPUS=1`,
`CUDA_VISIBLE_DEVICES=0`, Torch CUDA device count one, device
`NVIDIA GH200 120GB`, capability `9.0`. This job used about `0.0217` elapsed
GPU-hour; all three S02 exact jobs total about `0.0739` elapsed GPU-hour.

### Exact synthetic CUDA result

One and only one test ran:

```text
tests/test_s02_gpu_forward_backward.py::
  test_s02_cuda_b3_overcap_empty_isolation_forward_backward
1 passed in 5.50s
JUnit: tests=1 failures=0 errors=0 skipped=0
```

The fixture used B=3: two populated samples each exceeded the point and pillar
caps; sample 1 was empty. Exact diagnostics were:

```text
input/in-range             [12, 0, 9]
occupied pillars           [3, 0, 4]
selected pillars           [2, 0, 2]
truncated pillars          [1, 0, 2]
kept points                [5, 0, 4]
point-cap drops            [3, 0, 2]
pillar-cap drops           [4, 0, 3]
selected batch ids         [0, 0, 2, 2]
selected local cell keys   [0, 1, 4, 6]
```

CUDA output shape was `[3,32,4,6]`, dtype fp32, finite, with empty sample output
zero and sample 0 exactly equal to its isolated B=1 output. Loss was finite and
positive at `0.027591658756136894`. Backward produced finite nonzero gradients for
every intended parameter:

| Parameter | Gradient norm |
|---|---:|
| `linear.weight` | `0.015205752104520798` |
| `norm.weight` | `0.012320908717811108` |
| `norm.bias` | `0.012358705513179302` |

No optimizer or GradScaler step ran.

### Raw artifacts and hashes

Output root:

`/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s02_gpu_fb_b6f815d01b53`

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `execution_identity.json` | 1,472 | `b8f63aba3e898c11da56fb4ab4193e1bc9f832199f80418dacff6f4e4d448d55` |
| `runtime_source_sha256s.txt` | 1,546 | `5f5cc459a149483120bafc91cacb1c8a19bf500c45844a60891fe47ee28e1e49` |
| `pytest.log` | 789 | `f2228e25fdf14dadbcf15cdb1c73fd6760703321971eda928d8b75759dab42ad` |
| `pytest.junit.xml` | 378 | `38990046c2ab3f855bae48cbbb3a0ff917b578800bc92a0f6517187d8cc6e7f1` |
| `sha256sums.txt` | 747 | `87c62e780c526f0407c2c50ee192694ad1c61d85f759e517c802634b311a6d39` |

Logs:

- stdout: 3,024 bytes, SHA-256
  `1a0264a829a98f64891f5c01f85fe542b32d714fba3a7a59aa8aa7eb621be21f`;
- stderr: 123 bytes, SHA-256
  `ae6330855ac405b2e19691ca1681d7f9eeedc6216718d1516023d9376d891b57`.

Stderr contains only the normal module-purge notice. Runtime source manifests
passed `sha256sum -c` before and after pytest. The final in-job manifest passed
`sha256sum -c` for execution identity, source manifest, pytest log, and JUnit.

### Interpretation and disposition

Allowed: the unchanged reviewed S02 PointPillars implementation completed the
exact bounded synthetic CUDA forward/backward fixture with finite intended
gradients, exact per-sample diagnostics, and one-GPU source-attested execution.

Forbidden: performance/memory claims, full-stack or S07-B readiness by this job
alone, mini/trainval readiness, target-frequency claims, old-checkpoint
compatibility, mAP/NDS, fusion gain, FL/security, generalization, science, or
publication claims.

Job `336713` closes only the P1 evidence gap identified by S02-R. Final reviewed
acceptance requires the permitted limited re-review of this request/results/handoff
and exact executable diff. No further compute is authorized.
