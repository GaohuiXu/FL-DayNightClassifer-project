#!/bin/bash
# One-shot S08 implementation smoke.  Submission remains bound by RUN_REQUEST.md.
set -euo pipefail
umask 077

: "${S08_SNAPSHOT:?required}"
: "${S08_OUTPUT:?required}"

test -d "${S08_SNAPSHOT}"
test -d "${S08_OUTPUT}"
test -f "${S08_SNAPSHOT}/fl_v3/tests/test_s08_precision_diagnostics.py"
test -f "${S08_SNAPSHOT}/fl_v3/tests/test_s08_precision_partition.py"

source "${S08_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONPATH="${S08_SNAPSHOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
export TMPDIR=/tmp
unset PYTHONWARNINGS
unset NUSCENES_DATAROOT ARRHENIUS_NUSCENES_DATAROOT NUSCENES_DATA_DIR
unset NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST

cd "${S08_OUTPUT}"
python - <<'PY' > environment.json
import json
import platform
from importlib.metadata import version

import torch
from fl_v3.utils.runtime import verify_runtime_dependency_identity

if platform.machine() != "aarch64":
    raise RuntimeError("S08 smoke requires an aarch64 compute node")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("S08 smoke requires exactly one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GH200 120GB":
    raise RuntimeError("S08 smoke requires the reviewed NVIDIA GH200 120GB target")

verified = verify_runtime_dependency_identity({
    "det-lidar-arch": "second_075",
    "dependency-torch": "2.11.0+cu128",
    "dependency-torch-build-sha256": "a58ba749ac7947ce123a6af8d4cdc595d2aff5dccccec5d6e10bcfe522040f10",
    "dependency-torch-source-sha": "70d99e998b4955e0049d13a98d77ae1b14db1f45",
    "dependency-spconv": "2.3.8",
    "dependency-spconv-build-sha256": "74934de877e07a8eef8edacd4e31ec0f06eff030b3bc7e06d01f41b1444687d8",
    "dependency-spconv-source-sha": "263d6b47425ef843c82f997b12d8b714013d216c",
    "dependency-spconv-source-state": {
        "format": "git-tracked-regular-files.v1",
        "changes": [{
            "status": " M",
            "path": "pyproject.toml",
            "sha256": "e2c84544b5b5d6fd8e149d88539c3a6e989a1824637fd6b0006891955cb7a7e9",
        }],
        "sha256": "499efdbb5ab31c43109d48f11ee0ff79af847a3d378fd48bf9c79f8672da28db",
    },
    "dependency-cumm": "0.7.13",
    "dependency-cumm-build-sha256": "0a7e3c1a8c3e8d41b3b40c4fb77d05bdec8ca2dfce5dbb8863626c4b45d8296d",
    "dependency-cumm-source-sha": "4dedaf43ff801e417c60c6bd7536a29d83d29ee0",
    "dependency-cumm-source-state": {
        "format": "git-tracked-regular-files.v1",
        "changes": [],
        "sha256": "f835ee22d539bbf0ab486fecf1188c3883c3cde5860913434cbcf945ee325662",
    },
})
identity_keys = (
    "torch", "torch_build_sha256", "torch_source_sha", "torch_build_config_sha256",
    "spconv_version", "spconv_build_sha256", "spconv_source_sha",
    "spconv_source_state_sha256",
    "cumm_version", "cumm_build_sha256", "cumm_source_sha",
    "cumm_source_state_sha256",
)

print(json.dumps({
    "machine": platform.machine(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "spconv": version("spconv"),
    "cumm": version("cumm"),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "runtime_dependencies": {key: verified[key] for key in identity_keys},
}, indent=2, sort_keys=True))
PY

set +e
timeout --signal=TERM --kill-after=30s 20m \
  python -m pytest -q -p no:cacheprovider \
    -c "${S08_SNAPSHOT}/fl_v3/pyproject.toml" \
    --rootdir="${S08_SNAPSHOT}" \
    --basetemp="${S08_OUTPUT}/pytest-tmp" \
    --junitxml="${S08_OUTPUT}/smoke.junit.xml" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s08_precision_partition.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s08_source_identity.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s08_precision_diagnostics.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s06_training_runtime.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s06_resolved_config.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s06_checkpoint_resume.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_eval_provenance.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s06_model_modes.py" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_candidate_templates_name_exact_choices_and_fail_closed" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_runtime_sparse_identity_binds_torch_packages_sources_and_imports" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_multitask_loss_maps_global_labels_and_reaches_every_task_head" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_s07_b_integration.py::test_multitask_loss_rejects_legacy_single_head_output" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_sparse_voxel_encoder.py::test_fp32_and_fp16_sparse_paths_have_finite_outputs_and_gradients" \
    "${S08_SNAPSHOT}/fl_v3/tests/test_sparse_voxel_encoder.py::test_second_fp32_island_overrides_outer_autocast_and_exposes_named_boundaries" \
    > "${S08_OUTPUT}/smoke.log" 2>&1
pytest_rc=$?
set -e

printf '%s\n' "${pytest_rc}" > "${S08_OUTPUT}/smoke.exit"
sed -n '1,20000p' "${S08_OUTPUT}/smoke.log"
test "${pytest_rc}" = 0
sha256sum environment.json smoke.log smoke.junit.xml smoke.exit > artifact_sha256s.txt
printf '%s\n' S08_PRECISION_SMOKE_PASS
