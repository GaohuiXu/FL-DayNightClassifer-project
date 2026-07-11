# S05-R2 RUN_REQUEST — exact synthetic CenterHead re-review runtime

## Approval state

- **Status:** `PENDING_S00_EXACT_O009_APPROVAL_DO_NOT_SUBMIT`.
- This request exists because the x86_64 login interpreter has no `torch`,
  `pytest`, or `numpy`; all 44 authored S05 cases remain `NOT RUN`.
- The requested job is a dependency/runtime engineering check only. It uses no
  dataset, model checkpoint, optimizer or parameter update, training step,
  scientific metric, profile, seed matrix, array, DDP, retry, or follow-on.
- Preparing or committing this file is not permission to run it. S05-R2 must not
  invoke `sbatch` until S00 approves the exact tuple below, including the final
  immutable request-file SHA-256 and external launcher SHA-256.

## Immutable source identity

- Reviewed worker/delivery SHA:
  `705216de097ae9eeb1813de6dcdc916e2844fcde`.
- Worker tree: `2d5cd99c004e3ebd83a748f84141c03739e8fd4b`.
- Worker branch which must still resolve to that exact commit at submission:
  `codex/s05-centerhead-decode`.
- Original reviewed delivery: `4561d3ef4d5dd1dcbfe71fdf0ca1eb38d61257d9`.
- Remediation implementation: `753944c199ceeace160732218f1b16dfdd15ac21`.
- Runtime source-list SHA-256 (31 C-locale-sorted paths):
  `bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857`.
- Runtime source-state SHA-256 (SHA-256 of the 31 `sha256sum` lines):
  `2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766`.
- Runtime source includes all selected tests; head/decode/NMS/BEV/eval sources;
  the complete eagerly imported `data/nuscenes/*.py` package; package
  initializers; environment bootstrap; and dependency/config manifests.
  `tests/conftest.py` is intentionally excluded because the exact invocation uses
  `--noconftest` and none of the selected cases needs its data fixtures.
- No working-tree path is executed. The launcher exports the exact Git object into
  a new immutable `/nobackup` snapshot, recomputes both source hashes there, and
  fails before environment activation on any mismatch.

## Exact bounded scope

The job runs exactly these four files and must collect exactly 44 cases:

1. `fl_v3/tests/test_head_capacity.py` — 6 cases;
2. `fl_v3/tests/test_s05_centerhead_decode.py` — 9 cases;
3. `fl_v3/tests/test_s05_nms.py` — 22 cases after parametrization;
4. `fl_v3/tests/test_s05_eval_roundtrip.py` — 7 cases.

The cases cover the six-task/GN topology; B=1/B>1 isolation; explicit name-to-
devkit ID mapping; per-class K=500/no second task K; fp16 adjacent-logit and strict
0.1 forced-FP32 boundaries; FP32 score/velocity output; canonical box/yaw/velocity
round trip; circle/rotate NMS geometry, units, budgets, input permutation and
cross-class suppression; submission conversion/order including the equal-score,
same-class, same-geometry velocity/attribute collision; invalid inputs; duplicate
samples; and the official 500-box cap.

## Resources and paths

- Account: `naiss2025-22-1113-gpu`.
- Partition: `gpu`.
- Nodes: scheduler-selected single node; the submission deliberately omits
  `--nodes=1` so it does not request an exclusive whole GH200 node.
- GPU: exactly one `nvidia_gh200_120gb`, allocation and CUDA visibility both
  verified fail-closed inside the batch step.
- CPUs: 8.
- Walltime: `00:15:00` (maximum 0.25 GPU-hours).
- Concurrency: one S05-R2 job maximum; no array, DDP, retry, resubmission, or
  follow-on.
- Common Git object store (read-only input):
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git`.
- Fresh immutable snapshot root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a`.
- Fresh output root:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a`.
- Slurm logs:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r2_centerhead_%j.{out,err}`.
- S00-provisioned immutable external launcher:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_centerhead_705216de097a.sh`.
- S00-provisioned immutable copy of this exact request:
  `/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_RUN_REQUEST_705216de097a.md`.

S00 must materialize the launcher exactly from the body below, make launcher and
request copy read-only, compute their SHA-256 values, verify both fresh roots are
absent and no `flv3_s05r2_centerhead` job is queued/running, and bind those literal
hashes in its approval and submission environment. This external binding avoids a
launcher/request self-hash cycle.

## Exact launcher body

SHA-256 of the exact code-block body below (from `#!` through the final
`exit "$pytest_status"` newline):
`7ea5e8128fac4ddb471c27030b2d18b7e133297fca6a50fb336f27ee007a9e10`.

```bash
#!/usr/bin/env bash
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s05r2_centerhead
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r2_centerhead_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s05r2_centerhead_%j.err
set -euo pipefail

required=(
  EXPECTED_S05R2_SHA EXPECTED_S05R2_TREE EXPECTED_S05R2_BRANCH
  EXPECTED_S05R2_SOURCE_LIST_SHA EXPECTED_S05R2_SOURCE_SHA
  EXPECTED_S05R2_LAUNCHER_SHA EXPECTED_S05R2_RUN_REQUEST_SHA
  S05R2_REQUEST_COPY S05R2_SNAPSHOT_ROOT S05R2_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then echo "$name is required" >&2; exit 2; fi
done

readonly COMMON_GIT_DIR=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/.git
readonly APPROVED_SHA=705216de097ae9eeb1813de6dcdc916e2844fcde
readonly APPROVED_TREE=2d5cd99c004e3ebd83a748f84141c03739e8fd4b
readonly APPROVED_BRANCH=codex/s05-centerhead-decode
readonly APPROVED_SOURCE_LIST_SHA=bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857
readonly APPROVED_SOURCE_SHA=2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766
readonly APPROVED_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a
readonly APPROVED_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a
readonly SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

[[ "$EXPECTED_S05R2_SHA" == "$APPROVED_SHA" ]]
[[ "$EXPECTED_S05R2_TREE" == "$APPROVED_TREE" ]]
[[ "$EXPECTED_S05R2_BRANCH" == "$APPROVED_BRANCH" ]]
[[ "$EXPECTED_S05R2_SOURCE_LIST_SHA" == "$APPROVED_SOURCE_LIST_SHA" ]]
[[ "$EXPECTED_S05R2_SOURCE_SHA" == "$APPROVED_SOURCE_SHA" ]]
[[ "$S05R2_SNAPSHOT_ROOT" == "$APPROVED_SNAPSHOT_ROOT" ]]
[[ "$S05R2_OUTPUT_ROOT" == "$APPROVED_OUTPUT_ROOT" ]]
[[ -d "$COMMON_GIT_DIR/objects" ]]
[[ ! -e "$S05R2_SNAPSHOT_ROOT" ]]
[[ ! -e "$S05R2_OUTPUT_ROOT" ]]
[[ -f "$S05R2_REQUEST_COPY" ]]
[[ "$(sha256sum "$SCRIPT_PATH" | awk '{print $1}')" == "$EXPECTED_S05R2_LAUNCHER_SHA" ]]
[[ "$(sha256sum "$S05R2_REQUEST_COPY" | awk '{print $1}')" == "$EXPECTED_S05R2_RUN_REQUEST_SHA" ]]

readonly JOB_DESC="$(scontrol show job -o "${SLURM_JOB_ID:?SLURM_JOB_ID required}")"
python3 - "$JOB_DESC" <<'PY'
import re, sys
desc = sys.argv[1]
for token in ("NumNodes=1 ", "NumCPUs=8 "):
    if token not in desc:
        raise SystemExit(f"allocation mismatch: missing {token!r}: {desc}")
m = re.search(r"AllocTRES=([^ ]+)", desc)
if m is None:
    raise SystemExit(f"missing AllocTRES: {desc}")
tres = dict(item.split("=", 1) for item in m.group(1).split(",") if "=" in item)
if tres.get("gres/gpu") != "1" or tres.get("gres/gpu:nvidia_gh200_120gb") != "1":
    raise SystemExit(f"expected exactly one GH200 allocation: {m.group(1)}")
PY
: "${CUDA_VISIBLE_DEVICES:?exactly one Slurm-visible GPU is required}"
IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
[[ "${#visible_devices[@]}" -eq 1 ]]
[[ -n "${visible_devices[0]}" && "${visible_devices[0]}" != "-1" ]]

readonly ACTUAL_BRANCH_SHA="$(git --git-dir="$COMMON_GIT_DIR" rev-parse "refs/heads/$APPROVED_BRANCH^{commit}")"
readonly ACTUAL_TREE="$(git --git-dir="$COMMON_GIT_DIR" rev-parse "$APPROVED_SHA^{tree}")"
[[ "$ACTUAL_BRANCH_SHA" == "$APPROVED_SHA" ]]
[[ "$ACTUAL_TREE" == "$APPROVED_TREE" ]]

readonly TMP_SNAPSHOT="${S05R2_SNAPSHOT_ROOT}.tmp.${SLURM_JOB_ID}"
[[ ! -e "$TMP_SNAPSHOT" ]]
mkdir -p "$(dirname "$S05R2_SNAPSHOT_ROOT")"
mkdir "$TMP_SNAPSHOT"
cleanup() { if [[ -d "$TMP_SNAPSHOT" ]]; then chmod -R u+w "$TMP_SNAPSHOT"; rm -rf -- "$TMP_SNAPSHOT"; fi; }
trap cleanup EXIT
git --git-dir="$COMMON_GIT_DIR" archive --format=tar "$APPROVED_SHA" | tar -xf - -C "$TMP_SNAPSHOT"

runtime_source_files() {
  {
    printf '%s\n' \
      fl_v3/pyproject.toml \
      fl_v3/requirements.lock.txt \
      fl_v3/requirements.txt \
      fl_v3/scripts/arrhenius_env.sh \
      fl_v3/src/fl_v3/__init__.py \
      fl_v3/src/fl_v3/data/__init__.py \
      fl_v3/src/fl_v3/eval/__init__.py \
      fl_v3/src/fl_v3/eval/box_to_global.py \
      fl_v3/src/fl_v3/eval/detection_eval.py \
      fl_v3/src/fl_v3/models/__init__.py \
      fl_v3/src/fl_v3/models/fusion/__init__.py \
      fl_v3/src/fl_v3/models/fusion/bev_grid.py \
      fl_v3/src/fl_v3/models/fusion/centerhead_decode.py \
      fl_v3/src/fl_v3/models/fusion/head.py \
      fl_v3/src/fl_v3/models/fusion/nms_deterministic.py \
      fl_v3/tests/test_head_capacity.py \
      fl_v3/tests/test_s05_centerhead_decode.py \
      fl_v3/tests/test_s05_eval_roundtrip.py \
      fl_v3/tests/test_s05_nms.py
    git --git-dir="$COMMON_GIT_DIR" ls-tree -r --name-only "$APPROVED_SHA" fl_v3/src/fl_v3/data/nuscenes | grep '\.py$'
  } | LC_ALL=C sort -u
}
readonly ACTUAL_SOURCE_LIST_SHA="$(runtime_source_files | sha256sum | awk '{print $1}')"
readonly ACTUAL_SOURCE_SHA="$({
  while IFS= read -r path; do sha256sum "$TMP_SNAPSHOT/$path" | sed "s|  $TMP_SNAPSHOT/|  |"; done < <(runtime_source_files)
} | sha256sum | awk '{print $1}')"
[[ "$ACTUAL_SOURCE_LIST_SHA" == "$APPROVED_SOURCE_LIST_SHA" ]]
[[ "$ACTUAL_SOURCE_SHA" == "$APPROVED_SOURCE_SHA" ]]

cat > "$TMP_SNAPSHOT/.s05r2_snapshot_identity" <<EOF
schema=s05r2.centerhead-synthetic.v1
worker_sha=$APPROVED_SHA
worker_tree=$APPROVED_TREE
source_list_sha256=$ACTUAL_SOURCE_LIST_SHA
source_sha256=$ACTUAL_SOURCE_SHA
launcher_sha256=$EXPECTED_S05R2_LAUNCHER_SHA
run_request_sha256=$EXPECTED_S05R2_RUN_REQUEST_SHA
slurm_job_id=$SLURM_JOB_ID
EOF
chmod -R a-w "$TMP_SNAPSHOT"
mv "$TMP_SNAPSHOT" "$S05R2_SNAPSHOT_ROOT"
trap - EXIT
readonly REPO="$S05R2_SNAPSHOT_ROOT"
if find "$REPO" -xdev -perm /222 -print -quit | grep -q .; then
  echo "snapshot is writable" >&2; exit 2
fi

cd "$REPO"
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
arrhenius_activate_env
export PYTHONPATH="$REPO/fl_v3/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTEST_ADDOPTS
python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GH200, available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
PY

mkdir -p "$S05R2_OUTPUT_ROOT/tmp"
export TMPDIR="$S05R2_OUTPUT_ROOT/tmp"
cp "$SCRIPT_PATH" "$S05R2_OUTPUT_ROOT/approved_launcher.sh"
cp "$S05R2_REQUEST_COPY" "$S05R2_OUTPUT_ROOT/approved_run_request.md"
cp "$REPO/.s05r2_snapshot_identity" "$S05R2_OUTPUT_ROOT/snapshot_identity.txt"
runtime_source_files > "$S05R2_OUTPUT_ROOT/runtime_source_files.txt"
while IFS= read -r path; do sha256sum "$REPO/$path" | sed "s|  $REPO/|  |"; done \
  < "$S05R2_OUTPUT_ROOT/runtime_source_files.txt" > "$S05R2_OUTPUT_ROOT/runtime_source_sha256s.txt"
[[ "$(sha256sum "$S05R2_OUTPUT_ROOT/runtime_source_files.txt" | awk '{print $1}')" == "$APPROVED_SOURCE_LIST_SHA" ]]
[[ "$(sha256sum "$S05R2_OUTPUT_ROOT/runtime_source_sha256s.txt" | awk '{print $1}')" == "$APPROVED_SOURCE_SHA" ]]

python - "$S05R2_OUTPUT_ROOT/execution_identity.json" "$JOB_DESC" <<'PY'
import importlib.metadata, json, os, platform, socket, sys
out, job_desc = sys.argv[1:]
record = {
    "schema": "s05r2.centerhead-synthetic.v1",
    "worker_sha": os.environ["EXPECTED_S05R2_SHA"],
    "worker_tree": os.environ["EXPECTED_S05R2_TREE"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S05R2_SOURCE_LIST_SHA"],
    "runtime_source_sha256": os.environ["EXPECTED_S05R2_SOURCE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S05R2_LAUNCHER_SHA"],
    "run_request_sha256": os.environ["EXPECTED_S05R2_RUN_REQUEST_SHA"],
    "snapshot_root": os.environ["S05R2_SNAPSHOT_ROOT"],
    "output_root": os.environ["S05R2_OUTPUT_ROOT"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm_job_description": job_desc,
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in ("torch", "numpy", "pytest", "nuscenes-devkit", "Pillow")
    },
    "synthetic_only": True,
    "dataset_access": False,
    "optimizer_or_parameter_update": False,
    "scientific_metric": False,
}
with open(out, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
PY
printf '%s\n' "$JOB_DESC" > "$S05R2_OUTPUT_ROOT/slurm_allocation.txt"

set +e
python -m pytest -q -ra --noconftest -p no:cacheprovider \
  fl_v3/tests/test_head_capacity.py \
  fl_v3/tests/test_s05_centerhead_decode.py \
  fl_v3/tests/test_s05_nms.py \
  fl_v3/tests/test_s05_eval_roundtrip.py \
  --junitxml="$S05R2_OUTPUT_ROOT/pytest.junit.xml" \
  2>&1 | tee "$S05R2_OUTPUT_ROOT/pytest.log"
pytest_status=${PIPESTATUS[0]}
set -e

if [[ "$pytest_status" -eq 0 ]]; then
  set +e
  python - "$S05R2_OUTPUT_ROOT/pytest.junit.xml" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
counts = {k: sum(int(s.attrib.get(k, "0")) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
if counts != {"tests": 44, "failures": 0, "errors": 0, "skipped": 0}:
    raise SystemExit(f"S05-R2 JUnit acceptance failed: {counts}")
print(f"S05-R2 JUnit acceptance PASS: {counts}")
PY
  junit_status=$?
  set -e
  if [[ "$junit_status" -ne 0 ]]; then pytest_status="$junit_status"; fi
fi

sha256sum \
  "$S05R2_OUTPUT_ROOT/approved_launcher.sh" \
  "$S05R2_OUTPUT_ROOT/approved_run_request.md" \
  "$S05R2_OUTPUT_ROOT/snapshot_identity.txt" \
  "$S05R2_OUTPUT_ROOT/runtime_source_files.txt" \
  "$S05R2_OUTPUT_ROOT/runtime_source_sha256s.txt" \
  "$S05R2_OUTPUT_ROOT/execution_identity.json" \
  "$S05R2_OUTPUT_ROOT/slurm_allocation.txt" \
  "$S05R2_OUTPUT_ROOT/pytest.log" \
  "$S05R2_OUTPUT_ROOT/pytest.junit.xml" \
  > "$S05R2_OUTPUT_ROOT/sha256sums.txt"
sha256sum -c "$S05R2_OUTPUT_ROOT/sha256sums.txt"
exit "$pytest_status"
```

S00 must independently recompute the launcher hash after materializing it and
include the same literal value in the approval. Any change to the body invalidates
the request.

## Exact submission form after approval

S00 must replace the two shell variables below with the literal hashes it has
approved; the remaining tuple is already immutable. S05-R2 may then submit at
most once:

```bash
LAUNCHER_SHA=<S00-approved-exact-launcher-sha256>
REQUEST_SHA=<S00-approved-exact-request-sha256>
export LAUNCHER_SHA REQUEST_SHA
test "$(sha256sum /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_centerhead_705216de097a.sh | awk '{print $1}')" = "$LAUNCHER_SHA" && \
test "$(sha256sum /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_RUN_REQUEST_705216de097a.md | awk '{print $1}')" = "$REQUEST_SHA" && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a && \
test ! -e /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a && \
test -z "$(squeue -u "$USER" -h -o '%j' | awk '$1 == "flv3_s05r2_centerhead"')" && \
sbatch --export=ALL,EXPECTED_S05R2_SHA=705216de097ae9eeb1813de6dcdc916e2844fcde,EXPECTED_S05R2_TREE=2d5cd99c004e3ebd83a748f84141c03739e8fd4b,EXPECTED_S05R2_BRANCH=codex/s05-centerhead-decode,EXPECTED_S05R2_SOURCE_LIST_SHA=bea19dd528010020a462b18cfaeedd2642fd0e0a147ac458e215bdb8718b1857,EXPECTED_S05R2_SOURCE_SHA=2ff6389f0a556663e0cd2284c76c9fa11741bb0f44adb28eda4aebd33765c766,EXPECTED_S05R2_LAUNCHER_SHA="$LAUNCHER_SHA",EXPECTED_S05R2_RUN_REQUEST_SHA="$REQUEST_SHA",S05R2_REQUEST_COPY=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_RUN_REQUEST_705216de097a.md,S05R2_SNAPSHOT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots/s05r2_centerhead_705216de097a,S05R2_OUTPUT_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s05r2_centerhead_705216de097a \
  /nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/launchers/s05r2_centerhead_705216de097a.sh
```

## Acceptance and stop conditions

PASS requires all of the following:

- exact worker branch/SHA/tree, launcher/request hashes, source-list hash, and
  source-state hash;
- a fresh read-only `/nobackup` snapshot and fresh output root;
- scheduler allocation of exactly one node, one GH200 and eight CPUs, plus exactly
  one CUDA-visible GPU after environment activation;
- actual dependency versions and allocation identity recorded;
- exactly 44 tests, zero failures, errors, and skips;
- generated artifact checksums and a successful in-job `sha256sum -c`.

Stop/fail immediately on any identity/hash/allocation/output collision, missing
dependency, collection mismatch, failure/error/skip, exception, or walltime. There
is no automatic retry. Any negative result is preserved and returned to S00 before
any new request.

## Interpretation boundary

An accepted PASS can close the S05 authored synthetic runtime gate and support the
independent S05-R2 source verdict. It cannot establish production detector/loss/
config integration, official CUDA-kernel parity, CPU NMS performance, mini or
trainval model quality, mAP/NDS, full-run readiness, FL/security behavior, or any
scientific/publication claim.
