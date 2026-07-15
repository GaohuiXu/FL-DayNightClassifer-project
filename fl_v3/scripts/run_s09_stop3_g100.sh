#!/bin/bash
# Exact one-shot S09 STOP-3 loader sweep plus F-U G100 readiness job.
# Submission is valid only for the immutable tuple frozen in RUN_REQUEST.md.
set -euo pipefail
umask 077

: "${S09_STOP3_SNAPSHOT:?required}"
: "${S09_STOP3_OUTPUT:?required}"
: "${S09_STOP3_EXPECTED_SOURCE_SHA:?required}"
: "${S09_STOP3_EXPECTED_TREE:?required}"
: "${S09_STOP3_EXPECTED_CONFIG_SHA256:?required}"

CONFIG_REL="fl_v3/configs/s09_stop3_f_u_g100.json"
RUNNER_REL="fl_v3/scripts/run_s09_stop3_g100.sh"
TRAIN_REL="fl_v3/scripts/centralized_train.py"
READINESS_DIR="${S09_STOP3_OUTPUT}/readiness"

test -d "${S09_STOP3_SNAPSHOT}"
test ! -e "${S09_STOP3_OUTPUT}"
test -f "${S09_STOP3_SNAPSHOT}/${CONFIG_REL}"
test -f "${S09_STOP3_SNAPSHOT}/${RUNNER_REL}"
test -f "${S09_STOP3_SNAPSHOT}/${TRAIN_REL}"

actual_source_sha="$(git -C "${S09_STOP3_SNAPSHOT}" rev-parse HEAD)"
actual_tree="$(git -C "${S09_STOP3_SNAPSHOT}" rev-parse 'HEAD^{tree}')"
test "${actual_source_sha}" = "${S09_STOP3_EXPECTED_SOURCE_SHA}"
test "${actual_tree}" = "${S09_STOP3_EXPECTED_TREE}"
test -z "$(git -C "${S09_STOP3_SNAPSHOT}" status --short --untracked-files=all)"
test "$(git -C "${S09_STOP3_SNAPSHOT}" branch --show-current)" = ""

# shellcheck disable=SC1091
source "${S09_STOP3_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
# Editable cumm/spconv imports may run ccimport/ninja checks; env.md therefore
# requires the CUDA build module stack even for runtime/training jobs.
arrhenius_load_modules build
# arrhenius_load_modules purges modules; the licensed data module must load later.
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S09_STOP3_SNAPSHOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s01_zip_full_gate_v2_1fe651700bd0/nuscenes_trainval_zip_manifest.sqlite"

test "${NUSCENES_DATAROOT}" = "/dataset/easybuild/data/nuScenes-data/1.0-map-1.3-zip"
test -f "${NUSCENES_ZIP_MANIFEST}"
test "${SLURM_GPUS_ON_NODE:-1}" != "0"

mkdir -p "${S09_STOP3_OUTPUT}"
cd "${S09_STOP3_OUTPUT}"

python - "${S09_STOP3_SNAPSHOT}/${CONFIG_REL}" <<'PY' > config_identity.json
import hashlib
import json
import os
import sys
from pathlib import Path

from fl_v3.config.resolved import load_resolved_config

path = Path(sys.argv[1]).resolve(strict=True)
resolved = load_resolved_config(path)
expected = os.environ["S09_STOP3_EXPECTED_CONFIG_SHA256"]
if resolved.sha256 != expected:
    raise RuntimeError(
        f"resolved config identity drift: expected={expected}, actual={resolved.sha256}"
    )
print(json.dumps({
    "config_path": str(path),
    "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "resolved_config_sha256": resolved.sha256,
    "execution": resolved.as_dict()["execution"],
    "data_identities": resolved.data_identities,
}, indent=2, sort_keys=True))
PY

python - "${S09_STOP3_SNAPSHOT}" "${actual_source_sha}" "${actual_tree}" \
  "${S09_STOP3_SNAPSHOT}/${RUNNER_REL}" <<'PY' > execution_identity.json
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch

snapshot, source_sha, tree, runner_text = sys.argv[1:]
runner = Path(runner_text).resolve(strict=True)
if platform.machine() != "aarch64":
    raise RuntimeError("S09 STOP-3 requires an aarch64 compute node")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("S09 STOP-3 requires exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GH200 120GB":
    raise RuntimeError("S09 STOP-3 requires the reviewed NVIDIA GH200 120GB target")
print(json.dumps({
    "schema": "s09.stop3-execution-identity.v1",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": platform.node(),
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "spconv": version("spconv"),
    "cumm": version("cumm"),
    "cuda_device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0),
    "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "snapshot": str(Path(snapshot).resolve(strict=True)),
    "source_sha": source_sha,
    "source_tree": tree,
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "resolved_config_sha256": os.environ["S09_STOP3_EXPECTED_CONFIG_SHA256"],
    "dataroot": os.environ["NUSCENES_DATAROOT"],
    "zip_manifest": os.environ["NUSCENES_ZIP_MANIFEST"],
    "world_size": os.environ["WORLD_SIZE"],
    "omp_num_threads": os.environ["OMP_NUM_THREADS"],
    "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
}, indent=2, sort_keys=True))
PY

telemetry_query="timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu"
nvidia-smi --query-gpu="${telemetry_query}" --format=csv,noheader,nounits \
  > gpu_telemetry_preflight.csv
test "$(wc -l < gpu_telemetry_preflight.csv)" -eq 1

telemetry_loop() {
  printf '%s\n' 'epoch_seconds,index,name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_watts,sm_clock_mhz,memory_clock_mhz,temperature_celsius'
  while true; do
    epoch_seconds="$(date '+%s.%N')"
    sample="$(nvidia-smi --query-gpu="${telemetry_query#timestamp,}" --format=csv,noheader,nounits)"
    printf '%s,%s\n' "${epoch_seconds}" "${sample}"
    sleep 1
  done
}

telemetry_loop > gpu_telemetry.csv &
telemetry_pid=$!
centralized_start_ns="$(date '+%s%N')"
set +e
python "${S09_STOP3_SNAPSHOT}/${TRAIN_REL}" \
  --config "${S09_STOP3_SNAPSHOT}/${CONFIG_REL}" \
  --out-dir "${READINESS_DIR}" \
  > centralized_train.stdout 2> centralized_train.stderr
centralized_status=$?
set -e
centralized_end_ns="$(date '+%s%N')"
kill "${telemetry_pid}" 2>/dev/null || true
wait "${telemetry_pid}" 2>/dev/null || true
telemetry_lines="$(wc -l < gpu_telemetry.csv)"
if (( telemetry_lines < 2 )); then
  printf '%s\n' \
    "GPU telemetry is unusable: expected a header and at least one sample" \
    >> centralized_train.stderr
  if (( centralized_status == 0 )); then
    centralized_status=3
  fi
fi
printf '%s\n' "${centralized_status}" > centralized_train.exit

python - "${centralized_start_ns}" "${centralized_end_ns}" \
  "${READINESS_DIR}/readiness.json" <<'PY' > telemetry_alignment.json
import json
import sys
from pathlib import Path

start_ns, end_ns = map(int, sys.argv[1:3])
readiness_path = Path(sys.argv[3])
result = {
    "schema": "s09.stop3-telemetry-alignment.v1",
    "centralized_command_start_unix_ns": start_ns,
    "centralized_command_end_unix_ns": end_ns,
    "training_interval_definition": (
        "when readiness.json exists, the training interval ends no later than the "
        "captured command end and starts training_wall_seconds earlier; 1 Hz samples "
        "have up to one sampling interval of boundary uncertainty"
    ),
    "sampling_interval_seconds": 1,
}
if readiness_path.is_file():
    report = json.loads(readiness_path.read_text(encoding="utf-8"))
    wall_seconds = float(report["training_wall_seconds"])
    result.update({
        "training_wall_seconds": wall_seconds,
        "estimated_training_start_unix_ns": end_ns - round(wall_seconds * 1e9),
        "estimated_training_end_unix_ns": end_ns,
    })
print(json.dumps(result, indent=2, sort_keys=True))
PY

artifact_manifest_tmp="${S09_STOP3_OUTPUT}.artifact_sha256s.$$"
find . -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${artifact_manifest_tmp}"
mv "${artifact_manifest_tmp}" artifact_sha256s.txt
find . -type f -exec chmod 0444 {} +
find . -type d -exec chmod 0555 {} +

exit "${centralized_status}"
