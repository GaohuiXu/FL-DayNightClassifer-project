#!/bin/bash
# One-shot optimized S09 STOP-4C F-U G100. No loader matrix or operator profiler.
set -euo pipefail
umask 077

: "${S09_STOP4C_SNAPSHOT:?required}"
: "${S09_STOP4C_OUTPUT:?required}"
: "${S09_STOP4C_EXPECTED_SOURCE_SHA:?required}"
: "${S09_STOP4C_EXPECTED_TREE:?required}"
: "${S09_STOP4C_EXPECTED_CONFIG_SHA256:?required}"

CONFIG_REL="fl_v3/configs/s09_stop4c_f_u_g100.json"
RUNNER_REL="fl_v3/scripts/run_s09_stop4c_g100.sh"
TRAIN_REL="fl_v3/scripts/centralized_train.py"
READINESS_DIR="${S09_STOP4C_OUTPUT}/readiness"

test -d "${S09_STOP4C_SNAPSHOT}"
test ! -e "${S09_STOP4C_OUTPUT}"
for relative in "${CONFIG_REL}" "${RUNNER_REL}" "${TRAIN_REL}"; do
  test -f "${S09_STOP4C_SNAPSHOT}/${relative}"
done

actual_source_sha="$(git -C "${S09_STOP4C_SNAPSHOT}" rev-parse HEAD)"
actual_tree="$(git -C "${S09_STOP4C_SNAPSHOT}" rev-parse 'HEAD^{tree}')"
test "${actual_source_sha}" = "${S09_STOP4C_EXPECTED_SOURCE_SHA}"
test "${actual_tree}" = "${S09_STOP4C_EXPECTED_TREE}"
test -z "$(git -C "${S09_STOP4C_SNAPSHOT}" status --short --untracked-files=all)"
test "$(git -C "${S09_STOP4C_SNAPSHOT}" branch --show-current)" = ""

# shellcheck disable=SC1091
source "${S09_STOP4C_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S09_STOP4C_SNAPSHOT}/fl_v3/src"
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

mkdir -p "${S09_STOP4C_OUTPUT}"
cd "${S09_STOP4C_OUTPUT}"

python - "${S09_STOP4C_SNAPSHOT}/${CONFIG_REL}" <<'PY' > config_identity.json
import hashlib
import json
import os
import sys
from pathlib import Path

from fl_v3.config.resolved import load_resolved_config

path = Path(sys.argv[1]).resolve(strict=True)
resolved = load_resolved_config(path)
expected = os.environ["S09_STOP4C_EXPECTED_CONFIG_SHA256"]
if resolved.sha256 != expected:
    raise RuntimeError(
        f"resolved config identity drift: expected={expected}, actual={resolved.sha256}"
    )
data = resolved.as_dict()
if data["model"]["camera_activation_checkpoint"] is not False:
    raise RuntimeError("STOP-4C requires camera activation checkpointing disabled")
if data["execution"]["loader_profile"] is not None:
    raise RuntimeError("STOP-4C forbids the retired worker matrix")
if data["execution"]["operator_profile"] is not None:
    raise RuntimeError("STOP-4C forbids an operator profiler")
print(json.dumps({
    "config_path": str(path),
    "config_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "resolved_config_sha256": resolved.sha256,
    "model": data["model"],
    "training": data["training"],
    "execution": data["execution"],
    "data_identities": resolved.data_identities,
}, indent=2, sort_keys=True))
PY

python - "${S09_STOP4C_SNAPSHOT}" "${actual_source_sha}" "${actual_tree}" \
  "${S09_STOP4C_SNAPSHOT}/${RUNNER_REL}" <<'PY' > execution_identity.json
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
    raise RuntimeError("S09 STOP-4C requires an aarch64 compute node")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("S09 STOP-4C requires exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GH200 120GB":
    raise RuntimeError("S09 STOP-4C requires the reviewed NVIDIA GH200 120GB target")
print(json.dumps({
    "schema": "s09.stop4c-execution-identity.v1",
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
    "resolved_config_sha256": os.environ["S09_STOP4C_EXPECTED_CONFIG_SHA256"],
    "dataroot": os.environ["NUSCENES_DATAROOT"],
    "zip_manifest": os.environ["NUSCENES_ZIP_MANIFEST"],
    "world_size": os.environ["WORLD_SIZE"],
}, indent=2, sort_keys=True))
PY

set +e
python -m pytest -q \
  "${S09_STOP4C_SNAPSHOT}/fl_v3/tests/test_s06_resolved_config.py" \
  "${S09_STOP4C_SNAPSHOT}/fl_v3/tests/test_s09_readiness.py" \
  "${S09_STOP4C_SNAPSHOT}/fl_v3/tests/test_s08_precision_diagnostics.py" \
  "${S09_STOP4C_SNAPSHOT}/fl_v3/tests/test_s08_precision_partition.py" \
  "${S09_STOP4C_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_resolved_camera_checkpoint_switch_is_explicit_and_boolean" \
  "${S09_STOP4C_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_multitask_loss_term_recording_is_output_and_gradient_neutral" \
  > focused_tests.stdout 2> focused_tests.stderr
focused_test_status=$?
set -e
printf '%s\n' "${focused_test_status}" > focused_tests.exit
if (( focused_test_status != 0 )); then
  exit "${focused_test_status}"
fi

telemetry_query="index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu"
telemetry_loop() {
  printf '%s\n' 'epoch_seconds,index,name,utilization_gpu_percent,utilization_memory_percent,memory_used_mib,memory_total_mib,power_draw_watts,sm_clock_mhz,memory_clock_mhz,temperature_celsius'
  while true; do
    sample="$(nvidia-smi --query-gpu="${telemetry_query}" --format=csv,noheader,nounits)"
    printf '%s,%s\n' "$(date '+%s.%N')" "${sample}"
    sleep 1
  done
}

telemetry_loop > gpu_telemetry.csv &
telemetry_pid=$!
cleanup_telemetry() {
  if [[ -n "${telemetry_pid:-}" ]]; then
    kill "${telemetry_pid}" 2>/dev/null || true
    wait "${telemetry_pid}" 2>/dev/null || true
    telemetry_pid=""
  fi
}
trap cleanup_telemetry EXIT

start_ns="$(date '+%s%N')"
set +e
python "${S09_STOP4C_SNAPSHOT}/${TRAIN_REL}" \
  --config "${S09_STOP4C_SNAPSHOT}/${CONFIG_REL}" \
  --out-dir "${READINESS_DIR}" \
  > centralized_train.stdout 2> centralized_train.stderr
centralized_status=$?
set -e
end_ns="$(date '+%s%N')"
cleanup_telemetry
printf '%s\n' "${centralized_status}" > centralized_train.exit

validation_status=0
if (( centralized_status == 0 )); then
  set +e
  python - "${S09_STOP4C_SNAPSHOT}" "${READINESS_DIR}/readiness.json" <<'PY' \
    > readiness_validation.json
import json
import sys
from pathlib import Path

snapshot, report_text = sys.argv[1:]
sys.path.insert(0, str(Path(snapshot) / "fl_v3" / "scripts"))
from centralized_train import readiness_evidence_errors, readiness_performance_gate

report = json.loads(Path(report_text).read_text(encoding="utf-8"))
errors = readiness_evidence_errors(report, expect_operator_profile=False)
performance = readiness_performance_gate(
    report,
    expected_train_samples=28130,
    accepted_ratio_min=0.95,
    window_p95_p50_max=1.5,
    data_wait_share_max=0.10,
    peak_reserved_bytes_max=92_341_796_864,
    epoch_hours_max=24.0,
    combined_p50_limit_ms=229.620313,
    combined_p95_limit_ms=246.759346,
)
errors.extend(performance["errors"])
result = {
    "schema": "s09.stop4c-readiness-validation.v2",
    "status": "PASS" if report.get("status") == "PASS" and not errors else "FAIL",
    "evidence_validation_errors": errors,
    "performance_gates": {
        **performance["metrics"],
        "reference_combined_p50_ms": 208.745739,
        "reference_combined_p95_ms": 224.326678,
    },
}
print(json.dumps(result, indent=2, sort_keys=True))
if result["status"] != "PASS":
    raise SystemExit(4)
PY
  validation_status=$?
  set -e
fi
printf '%s\n' "${validation_status}" > readiness_validation.exit

python - "${start_ns}" "${end_ns}" "${READINESS_DIR}/readiness.json" <<'PY' \
  > telemetry_alignment.json
import json
import sys
from pathlib import Path

start_ns, end_ns = map(int, sys.argv[1:3])
report_path = Path(sys.argv[3])
result = {
    "schema": "s09.stop4c-telemetry-alignment.v1",
    "centralized_command_start_unix_ns": start_ns,
    "centralized_command_end_unix_ns": end_ns,
    "sampling_interval_seconds": 1,
}
if report_path.is_file():
    wall_seconds = float(json.loads(report_path.read_text())["training_wall_seconds"])
    result.update({
        "training_wall_seconds": wall_seconds,
        "estimated_training_start_unix_ns": end_ns - round(wall_seconds * 1e9),
        "estimated_training_end_unix_ns": end_ns,
    })
print(json.dumps(result, indent=2, sort_keys=True))
PY

final_status="${centralized_status}"
if (( final_status == 0 && validation_status != 0 )); then
  final_status="${validation_status}"
fi
if (( $(wc -l < gpu_telemetry.csv) < 2 && final_status == 0 )); then
  final_status=3
fi
printf '%s\n' "${final_status}" > final.exit
artifact_manifest_tmp="${S09_STOP4C_OUTPUT}.artifact_sha256s.$$"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "${artifact_manifest_tmp}"
mv "${artifact_manifest_tmp}" artifact_sha256s.txt
find . -type f -exec chmod 0444 {} +
find . -type d -exec chmod 0555 {} +
exit "${final_status}"
