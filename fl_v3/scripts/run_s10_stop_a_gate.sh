#!/bin/bash
# One-shot CPU-only S10 STOP-A split/ownership/evaluator gate. No model/training path.
set -euo pipefail
umask 077

: "${S10_STOPA_SNAPSHOT:?required}"
: "${S10_STOPA_OUTPUT:?required}"
: "${S10_STOPA_EXPECTED_SOURCE_SHA:?required}"
: "${S10_STOPA_EXPECTED_TREE:?required}"
: "${S10_STOPA_EXPECTED_RUNNER_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s10_stop_a_gate.sh"
GATE_REL="fl_v3/scripts/s10_stop_a_gate.py"
CACHE_DIR="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s09_stop1_cache_t1v2_1f276b9d2cc5/info_cache_msweep10"
ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"
WORK="${S10_STOPA_OUTPUT}.control"

test -d "${S10_STOPA_SNAPSHOT}"
test ! -e "${S10_STOPA_OUTPUT}"
test ! -e "${WORK}"
test -f "${S10_STOPA_SNAPSHOT}/${RUNNER_REL}"
test -f "${S10_STOPA_SNAPSHOT}/${GATE_REL}"
test -d "${CACHE_DIR}"
test -f "${ZIP_MANIFEST}"

actual_source_sha="$(git -C "${S10_STOPA_SNAPSHOT}" rev-parse HEAD)"
actual_tree="$(git -C "${S10_STOPA_SNAPSHOT}" rev-parse 'HEAD^{tree}')"
actual_runner_sha256="$(sha256sum "${S10_STOPA_SNAPSHOT}/${RUNNER_REL}" | cut -d' ' -f1)"
test "${actual_source_sha}" = "${S10_STOPA_EXPECTED_SOURCE_SHA}"
test "${actual_tree}" = "${S10_STOPA_EXPECTED_TREE}"
test "${actual_runner_sha256}" = "${S10_STOPA_EXPECTED_RUNNER_SHA256}"
test -z "$(git -C "${S10_STOPA_SNAPSHOT}" status --short --untracked-files=all)"
test "$(git -C "${S10_STOPA_SNAPSHOT}" branch --show-current)" = ""

# shellcheck disable=SC1091
source "${S10_STOPA_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S10_STOPA_SNAPSHOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="${ZIP_MANIFEST}"

test "${NUSCENES_DATAROOT}" = "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
test "${SLURM_JOB_PARTITION:-}" = "gpu"
test "${SLURM_CPUS_PER_TASK:-}" = "4"
test "${SLURM_MEM_PER_NODE:-}" = "32768"
test -n "${SLURM_JOB_GPUS:-}"
test "${SLURM_GPUS_ON_NODE:-0}" = "1"
mkdir -p "${WORK}"
runner_complete=0

finalize() {
  local status="${1:-1}"
  local target="${S10_STOPA_OUTPUT}"
  if [[ ! -d "${target}" ]]; then
    mv "${WORK}" "${target}"
  elif [[ -d "${WORK}" ]]; then
    mkdir -p "${target}/control"
    find "${WORK}" -mindepth 1 -maxdepth 1 -exec mv -t "${target}/control" {} +
    rmdir "${WORK}"
  fi
  printf '%s\n' "${status}" > "${target}/final.exit"
  local temporary="${target}/.runner_artifact_sha256s.tmp"
  find "${target}" -type f ! -name runner_artifact_sha256s.txt ! -name .runner_artifact_sha256s.tmp \
    -printf '%P\0' | sort -z | while IFS= read -r -d '' relative; do
      sha256sum "${target}/${relative}" | sed "s#  ${target}/#  #"
    done > "${temporary}"
  mv "${temporary}" "${target}/runner_artifact_sha256s.txt"
  find "${target}" -type f -exec chmod 0444 {} +
  find "${target}" -type d -exec chmod 0555 {} +
}

handle_signal() {
  exit "$1"
}

handle_exit() {
  local status=$?
  trap - EXIT TERM INT HUP QUIT
  if (( status == 0 && runner_complete != 1 )); then
    status=125
  fi
  finalize "${status}"
  exit "${status}"
}

trap 'handle_signal 143' TERM
trap 'handle_signal 130' INT
trap 'handle_signal 129' HUP
trap 'handle_signal 131' QUIT
trap handle_exit EXIT

python - "${S10_STOPA_SNAPSHOT}" "${actual_source_sha}" "${actual_tree}" \
  "${actual_runner_sha256}" <<'PY' > "${WORK}/execution_identity.json"
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import scipy
import torch

snapshot, source_sha, source_tree, runner_sha256 = sys.argv[1:]
if platform.machine() != "aarch64":
    raise RuntimeError("STOP-A validated environment requires an aarch64 compute node")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("STOP-A CPU-only gate requires empty CUDA_VISIBLE_DEVICES")
if not os.environ.get("SLURM_JOB_GPUS"):
    raise RuntimeError("STOP-A replacement requires one explicitly reserved GH200")
if os.environ.get("SLURM_GPUS_ON_NODE", "0") != "1":
    raise RuntimeError("STOP-A replacement requires exactly one allocated GPU on node")
if torch.cuda.is_available() or torch.cuda.device_count() != 0:
    raise RuntimeError("STOP-A CPU-only gate unexpectedly exposes a CUDA device")
print(json.dumps({
    "schema": "fl_v3.s10.stop_a_execution_identity.v3",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "scipy": scipy.__version__,
    "nuscenes_devkit": version("nuscenes-devkit"),
    "allocation": "one_gh200_reserved_cpu_only_process",
    "device": "CPU-only aarch64 process; one reserved GH200 hidden",
    "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS", ""),
    "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE", "0"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device_count": torch.cuda.device_count(),
    "snapshot": str(Path(snapshot).resolve(strict=True)),
    "source_sha": source_sha,
    "source_tree": source_tree,
    "runner_sha256": runner_sha256,
    "dataroot": os.environ["NUSCENES_DATAROOT"],
    "zip_manifest": os.environ["NUSCENES_ZIP_MANIFEST"],
}, indent=2, sort_keys=True))
PY

set +e
python -m pytest -q -p no:cacheprovider \
  "${S10_STOPA_SNAPSHOT}/fl_v3/tests/test_s10_internal_split.py" \
  "${S10_STOPA_SNAPSHOT}/fl_v3/tests/test_s10_subset_eval.py" \
  "${S10_STOPA_SNAPSHOT}/fl_v3/tests/test_build_gt_database.py" \
  "${S10_STOPA_SNAPSHOT}/fl_v3/tests/test_eval_detection_eval.py" \
  "${S10_STOPA_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_f_cbgs_is_deterministic_and_cannot_stack_loss_weights" \
  "${S10_STOPA_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_production_sampler_is_epoch_addressable_over_expanded_cbgs_dataset" \
  > "${WORK}/focused_tests.stdout" 2> "${WORK}/focused_tests.stderr"
test_status=$?
set -e
printf '%s\n' "${test_status}" > "${WORK}/focused_tests.exit"
if (( test_status != 0 )); then
  exit "${test_status}"
fi

set +e
python "${S10_STOPA_SNAPSHOT}/${GATE_REL}" \
  --cache-dir "${CACHE_DIR}" \
  --dataroot "${NUSCENES_DATAROOT}" \
  --zip-manifest "${ZIP_MANIFEST}" \
  --output-dir "${S10_STOPA_OUTPUT}" \
  --train-cache-hash 310e1bba8f65912450e864b634a47b4ca2ea4feb20ed26018e087c93299eed0a \
  --train-cache-file-sha256 57fce20f035a99c0c0ab96fdef418c1b0e04e28bd3e32d191a8298f99919be30 \
  --train-cache-sidecar-sha256 f4c45dd12ea0db8ec35d9235de52e51981870b91f175c376d5c34747da661b6c \
  --val-cache-hash bb692de4c1eb8b66e8c74f4e807eb208ad891b45ce8f233e8017dc4f3a3b6e2f \
  --val-cache-file-sha256 d4ed7aee9978c2294e2087c917006cbb3d69276453266d0f9c92591340084837 \
  --val-cache-sidecar-sha256 4f5390815720e14625be31b20fb1596cafe9869ad95b08dc098aea65413be432 \
  --zip-manifest-hash 023f72b4220bb0db587be00920308bf9074384740fe186d243be92f9a53119f6 \
  --zip-manifest-file-sha256 228e2f5bab30007acb06eb61393d1fbacc88979490668ff800f8f7f9752a47fb \
  --source-sha "${actual_source_sha}" \
  --source-tree "${actual_tree}" \
  > "${WORK}/gate.stdout" 2> "${WORK}/gate.stderr"
gate_status=$?
set -e
printf '%s\n' "${gate_status}" > "${WORK}/gate.exit"
runner_complete=1
exit "${gate_status}"
