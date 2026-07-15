#!/bin/bash
# One-shot S09 STOP-4A baseline profiler plus checkpoint-off B=1/2/4 capacity cells.
# Submission is valid only for the immutable tuple frozen in RUN_REQUEST.md.
set -euo pipefail
umask 077

: "${S09_STOP4A_SNAPSHOT:?required}"
: "${S09_STOP4A_OUTPUT:?required}"
: "${S09_STOP4A_EXPECTED_SOURCE_SHA:?required}"
: "${S09_STOP4A_EXPECTED_TREE:?required}"
: "${S09_STOP4A_EXPECTED_B1_PROFILE_SHA256:?required}"
: "${S09_STOP4A_EXPECTED_B1_NO_CKPT_SHA256:?required}"
: "${S09_STOP4A_EXPECTED_B2_NO_CKPT_SHA256:?required}"
: "${S09_STOP4A_EXPECTED_B4_NO_CKPT_SHA256:?required}"

RUNNER_REL="fl_v3/scripts/run_s09_stop4a_profile_capacity.sh"
TRAIN_REL="fl_v3/scripts/centralized_train.py"
CONFIG_B1_PROFILE="fl_v3/configs/s09_stop4a_f_u_b1_profile.json"
CONFIG_B1_NO_CKPT="fl_v3/configs/s09_stop4a_f_u_b1_no_ckpt.json"
CONFIG_B2_NO_CKPT="fl_v3/configs/s09_stop4a_f_u_b2_no_ckpt.json"
CONFIG_B4_NO_CKPT="fl_v3/configs/s09_stop4a_f_u_b4_no_ckpt.json"

test -d "${S09_STOP4A_SNAPSHOT}"
test ! -e "${S09_STOP4A_OUTPUT}"
for relative in \
  "${RUNNER_REL}" "${TRAIN_REL}" \
  "${CONFIG_B1_PROFILE}" "${CONFIG_B1_NO_CKPT}" \
  "${CONFIG_B2_NO_CKPT}" "${CONFIG_B4_NO_CKPT}"
do
  test -f "${S09_STOP4A_SNAPSHOT}/${relative}"
done

actual_source_sha="$(git -C "${S09_STOP4A_SNAPSHOT}" rev-parse HEAD)"
actual_tree="$(git -C "${S09_STOP4A_SNAPSHOT}" rev-parse 'HEAD^{tree}')"
test "${actual_source_sha}" = "${S09_STOP4A_EXPECTED_SOURCE_SHA}"
test "${actual_tree}" = "${S09_STOP4A_EXPECTED_TREE}"
test -z "$(git -C "${S09_STOP4A_SNAPSHOT}" status --short --untracked-files=all)"
test "$(git -C "${S09_STOP4A_SNAPSHOT}" branch --show-current)" = ""

# shellcheck disable=SC1091
source "${S09_STOP4A_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONPATH="${S09_STOP4A_SNAPSHOT}/fl_v3/src"
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

mkdir -p "${S09_STOP4A_OUTPUT}"
cd "${S09_STOP4A_OUTPUT}"

python - "${S09_STOP4A_SNAPSHOT}" <<'PY' > config_identities.json
import hashlib
import json
import os
import sys
from pathlib import Path

from fl_v3.config.resolved import load_resolved_config

snapshot = Path(sys.argv[1]).resolve(strict=True)
entries = [
    ("b1_profile", "fl_v3/configs/s09_stop4a_f_u_b1_profile.json",
     "S09_STOP4A_EXPECTED_B1_PROFILE_SHA256"),
    ("b1_no_ckpt", "fl_v3/configs/s09_stop4a_f_u_b1_no_ckpt.json",
     "S09_STOP4A_EXPECTED_B1_NO_CKPT_SHA256"),
    ("b2_no_ckpt", "fl_v3/configs/s09_stop4a_f_u_b2_no_ckpt.json",
     "S09_STOP4A_EXPECTED_B2_NO_CKPT_SHA256"),
    ("b4_no_ckpt", "fl_v3/configs/s09_stop4a_f_u_b4_no_ckpt.json",
     "S09_STOP4A_EXPECTED_B4_NO_CKPT_SHA256"),
]
result = {}
for label, relative, expected_name in entries:
    path = (snapshot / relative).resolve(strict=True)
    resolved = load_resolved_config(path)
    expected = os.environ[expected_name]
    if resolved.sha256 != expected:
        raise RuntimeError(
            f"{label} resolved config drift: expected={expected}, actual={resolved.sha256}"
        )
    result[label] = {
        "path": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resolved_config_sha256": resolved.sha256,
        "model": resolved.as_dict()["model"],
        "training": resolved.as_dict()["training"],
        "execution": resolved.as_dict()["execution"],
        "data_identities": resolved.data_identities,
    }
print(json.dumps(result, indent=2, sort_keys=True))
PY

python - "${S09_STOP4A_SNAPSHOT}" "${actual_source_sha}" "${actual_tree}" \
  "${S09_STOP4A_SNAPSHOT}/${RUNNER_REL}" <<'PY' > execution_identity.json
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
    raise RuntimeError("S09 STOP-4A requires an aarch64 compute node")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("S09 STOP-4A requires exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GH200 120GB":
    raise RuntimeError("S09 STOP-4A requires the reviewed NVIDIA GH200 120GB target")
print(json.dumps({
    "schema": "s09.stop4a-execution-identity.v1",
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
    "dataroot": os.environ["NUSCENES_DATAROOT"],
    "zip_manifest": os.environ["NUSCENES_ZIP_MANIFEST"],
    "world_size": os.environ["WORLD_SIZE"],
}, indent=2, sort_keys=True))
PY

set +e
python -m pytest -q \
  "${S09_STOP4A_SNAPSHOT}/fl_v3/tests/test_s06_resolved_config.py" \
  "${S09_STOP4A_SNAPSHOT}/fl_v3/tests/test_s09_readiness.py" \
  "${S09_STOP4A_SNAPSHOT}/fl_v3/tests/test_s06_model_modes.py::test_operator_profile_ranges_are_opt_in_scoped_and_output_neutral" \
  "${S09_STOP4A_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_resolved_camera_constructor_is_stride8_half_metre_and_180_grid" \
  "${S09_STOP4A_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_resolved_camera_checkpoint_switch_is_explicit_and_boolean" \
  "${S09_STOP4A_SNAPSHOT}/fl_v3/tests/test_s08_precision_partition.py" \
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

printf '%s\n' 'label,start_unix_ns,end_unix_ns,exit_status' > cell_windows.csv
labels=(b1_profile b1_no_ckpt b2_no_ckpt b4_no_ckpt)
configs=("${CONFIG_B1_PROFILE}" "${CONFIG_B1_NO_CKPT}" "${CONFIG_B2_NO_CKPT}" "${CONFIG_B4_NO_CKPT}")
for index in "${!labels[@]}"; do
  label="${labels[${index}]}"
  config_relative="${configs[${index}]}"
  start_ns="$(date '+%s%N')"
  set +e
  python "${S09_STOP4A_SNAPSHOT}/${TRAIN_REL}" \
    --config "${S09_STOP4A_SNAPSHOT}/${config_relative}" \
    --out-dir "${S09_STOP4A_OUTPUT}/${label}" \
    > "${label}.stdout" 2> "${label}.stderr"
  status=$?
  set -e
  end_ns="$(date '+%s%N')"
  printf '%s,%s,%s,%s\n' "${label}" "${start_ns}" "${end_ns}" "${status}" \
    >> cell_windows.csv
  printf '%s\n' "${status}" > "${label}.exit"
done

cleanup_telemetry
telemetry_lines="$(wc -l < gpu_telemetry.csv)"
final_status=0
if (( telemetry_lines < 2 )); then
  final_status=3
fi

set +e
python - "${S09_STOP4A_SNAPSHOT}" <<'PY' > cell_statuses.json
import csv
import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(snapshot / "fl_v3" / "scripts"))
from centralized_train import readiness_evidence_errors

allowed_capacity_limits = {"b2_no_ckpt", "b4_no_ckpt"}
cuda_oom_signatures = ("cuda out of memory", "cuda error: out of memory")
rows = list(csv.DictReader(Path("cell_windows.csv").open(encoding="utf-8")))
result = {"schema": "s09.stop4a-cell-statuses.v1", "cells": [], "status": "PASS"}
for row in rows:
    label = row["label"]
    status = int(row["exit_status"])
    readiness_path = Path(label) / "readiness.json"
    stderr = Path(f"{label}.stderr").read_text(encoding="utf-8", errors="replace")
    classification = "PASS"
    evidence_errors = []
    readiness = None
    if readiness_path.is_file():
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if status == 0:
        if readiness is None:
            classification = "UNCLASSIFIED_FAILURE"
            evidence_errors.append("readiness.json is missing")
        else:
            evidence_errors = readiness_evidence_errors(
                readiness,
                expect_operator_profile=label == "b1_profile",
                verify_profile_artifacts=True,
            )
            if readiness.get("status") != "PASS":
                evidence_errors.append(
                    f"readiness status is {readiness.get('status')!r}, not PASS"
                )
            if evidence_errors:
                classification = "UNCLASSIFIED_FAILURE"
    elif (
        label in allowed_capacity_limits
        and any(signature in stderr.lower() for signature in cuda_oom_signatures)
    ):
        classification = "CAPACITY_LIMIT"
    else:
        classification = "UNCLASSIFIED_FAILURE"
    if classification == "UNCLASSIFIED_FAILURE":
        result["status"] = "FAIL"
    result["cells"].append({
        **row,
        "exit_status": status,
        "classification": classification,
        "readiness_status": None if readiness is None else readiness.get("status"),
        "evidence_validation_errors": evidence_errors,
    })
print(json.dumps(result, indent=2, sort_keys=True))
if result["status"] != "PASS":
    raise SystemExit(4)
PY
classification_status=$?
set -e
if (( classification_status != 0 )); then
  final_status="${classification_status}"
fi

printf '%s\n' "${final_status}" > final.exit
artifact_manifest_tmp="${S09_STOP4A_OUTPUT}.artifact_sha256s.$$"
find . -type f -print0 | sort -z | xargs -0 sha256sum > "${artifact_manifest_tmp}"
mv "${artifact_manifest_tmp}" artifact_sha256s.txt
find . -type f -exec chmod 0444 {} +
find . -type d -exec chmod 0555 {} +
exit "${final_status}"
