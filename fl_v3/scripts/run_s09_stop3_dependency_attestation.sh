#!/bin/bash
# Rebuild/warm and attest the exact editable sparse runtime before STOP-3 G100.
# This runner never loads nuScenes data or constructs/trains a model.
set -euo pipefail
umask 077

: "${S09_STOP3_DEP_SNAPSHOT:?required}"
: "${S09_STOP3_DEP_OUTPUT:?required}"
: "${S09_STOP3_DEP_EXPECTED_SOURCE_SHA:?required}"
: "${S09_STOP3_DEP_EXPECTED_TREE:?required}"
: "${S09_STOP3_DEP_EXPECTED_SPCONV_HEAD:?required}"
: "${S09_STOP3_DEP_EXPECTED_SPCONV_STATE:?required}"
: "${S09_STOP3_DEP_EXPECTED_CUMM_HEAD:?required}"
: "${S09_STOP3_DEP_EXPECTED_CUMM_STATE:?required}"

RUNNER_REL="fl_v3/scripts/run_s09_stop3_dependency_attestation.sh"
CONFIG_REL="fl_v3/configs/s09_stop3_f_u_g100.json"
SPCONV_SOURCE="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/spconv"
CUMM_SOURCE="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/src/cumm"

test -d "${S09_STOP3_DEP_SNAPSHOT}"
test ! -e "${S09_STOP3_DEP_OUTPUT}"
test -f "${S09_STOP3_DEP_SNAPSHOT}/${RUNNER_REL}"
test -f "${S09_STOP3_DEP_SNAPSHOT}/${CONFIG_REL}"
test -d "${SPCONV_SOURCE}/.git"
test -d "${CUMM_SOURCE}/.git"

actual_source_sha="$(git -C "${S09_STOP3_DEP_SNAPSHOT}" rev-parse HEAD)"
actual_tree="$(git -C "${S09_STOP3_DEP_SNAPSHOT}" rev-parse 'HEAD^{tree}')"
test "${actual_source_sha}" = "${S09_STOP3_DEP_EXPECTED_SOURCE_SHA}"
test "${actual_tree}" = "${S09_STOP3_DEP_EXPECTED_TREE}"
test -z "$(git -C "${S09_STOP3_DEP_SNAPSHOT}" status --short --untracked-files=all)"
test -z "$(git -C "${S09_STOP3_DEP_SNAPSHOT}" branch --show-current)"

# shellcheck disable=SC1091
source "${S09_STOP3_DEP_SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONPATH="${S09_STOP3_DEP_SNAPSHOT}/fl_v3/src"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1

test "$(uname -m)" = "aarch64"
test "${SLURM_GPUS_ON_NODE:-1}" != "0"
command -v nvcc >/dev/null
nvcc --version >/dev/null

mkdir -p "${S09_STOP3_DEP_OUTPUT}"
cd "${S09_STOP3_DEP_OUTPUT}"

module -t list > modules.txt 2>&1
nvcc --version > nvcc.txt 2>&1
git -C "${SPCONV_SOURCE}" diff --name-only --no-renames | LC_ALL=C sort \
  > spconv_tracked_changes_before.txt
git -C "${CUMM_SOURCE}" diff --name-only --no-renames | LC_ALL=C sort \
  > cumm_tracked_changes_before.txt
git -C "${SPCONV_SOURCE}" status --short --untracked-files=all \
  > spconv_status_before.txt
git -C "${CUMM_SOURCE}" status --short --untracked-files=all \
  > cumm_status_before.txt

cat > dependency_probe.py <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from fl_v3.config.resolved import load_resolved_config
from fl_v3.source_identity import inspect_tracked_source_state
from fl_v3.utils.runtime import (
    _executable_artifact_records,
    _runtime_build_identity,
    _source_checkout_identity,
)


def git_head(root: str) -> str:
    return subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sparse_identity(
    label: str,
    source: str,
    expected_head: str,
    expected_state: str,
) -> dict[str, object]:
    distribution = import_name = label
    version = importlib.metadata.version(distribution)
    if (label, version) not in {("spconv", "2.3.8"), ("cumm", "0.7.13")}:
        raise RuntimeError(f"unexpected {label} version: {version}")
    if git_head(source) != expected_head:
        raise RuntimeError(f"{label} source HEAD drift")
    state_before = inspect_tracked_source_state(source)
    if state_before["sha256"] != expected_state:
        raise RuntimeError(f"{label} tracked source-state drift before import")
    head, origin, resolved_state = _source_checkout_identity(distribution, import_name)
    if head != expected_head or resolved_state != state_before:
        raise RuntimeError(f"{label} editable-source provenance drift")
    targets = (
        ("spconv", "spconv.pytorch")
        if label == "spconv"
        else ("cumm", "cumm.tensorview")
    )
    build, origins = _runtime_build_identity(
        distribution,
        import_name,
        targets,
        {"version": version, "source_sha": head},
    )
    state_after = inspect_tracked_source_state(source)
    if state_after != state_before:
        raise RuntimeError(f"{label} tracked source changed during stable probe")
    return {
        "version": version,
        "source_head": head,
        "source_state": state_after,
        "import_origin": origin,
        "import_origins": origins,
        "build_sha256": build,
        "executable_artifacts": _executable_artifact_records(
            distribution, import_name
        ),
    }


config_path, spconv_source, cumm_source, output_path = sys.argv[1:]
resolved = load_resolved_config(config_path)
run = resolved.to_run_config()
torch_metadata = {
    "version": str(torch.__version__),
    "git_version": str(getattr(torch.version, "git_version", "") or ""),
    "cuda": str(getattr(torch.version, "cuda", "")),
    "config": str(torch.__config__.show()),
}
torch_build, torch_origins = _runtime_build_identity(
    "torch", "torch", ("torch",), torch_metadata
)
if torch.__version__ != run["dependency-torch"]:
    raise RuntimeError("Torch version drift")
if torch_metadata["git_version"] != run["dependency-torch-source-sha"]:
    raise RuntimeError("Torch source drift")
if torch_build != run["dependency-torch-build-sha256"]:
    raise RuntimeError("Torch executable-build drift")
if platform.machine() != "aarch64":
    raise RuntimeError("dependency attestation requires aarch64")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("dependency attestation requires one visible CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA GH200 120GB":
    raise RuntimeError("dependency attestation requires NVIDIA GH200 120GB")

result = {
    "schema": "s09.stop3-dependency-attestation.v1",
    "machine": platform.machine(),
    "device": torch.cuda.get_device_name(0),
    "torch": {
        "version": torch.__version__,
        "source_sha": torch_metadata["git_version"],
        "build_sha256": torch_build,
        "import_origins": torch_origins,
        "build_config_sha256": hashlib.sha256(
            torch_metadata["config"].encode("utf-8")
        ).hexdigest(),
    },
    "cumm": sparse_identity(
        "cumm",
        cumm_source,
        os.environ["S09_STOP3_DEP_EXPECTED_CUMM_HEAD"],
        os.environ["S09_STOP3_DEP_EXPECTED_CUMM_STATE"],
    ),
    "spconv": sparse_identity(
        "spconv",
        spconv_source,
        os.environ["S09_STOP3_DEP_EXPECTED_SPCONV_HEAD"],
        os.environ["S09_STOP3_DEP_EXPECTED_SPCONV_STATE"],
    ),
}
Path(output_path).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

restore_generated_stubs() {
  local source="$1"
  local baseline="$2"
  local label="$3"
  local current new_changes path
  current="${S09_STOP3_DEP_OUTPUT}/${label}_tracked_changes_current.txt"
  new_changes="${S09_STOP3_DEP_OUTPUT}/${label}_new_tracked_changes.txt"
  git -C "${source}" diff --name-only --no-renames | LC_ALL=C sort > "${current}"
  comm -13 "${baseline}" "${current}" > "${new_changes}"
  while IFS= read -r path; do
    test -n "${path}" || continue
    case "${path}" in
      "${label}"/core_cc/*.pyi)
        git -C "${source}" restore --source=HEAD --worktree -- "${path}"
        ;;
      *)
        printf 'refusing to restore unexpected generated path: %s/%s\n' \
          "${label}" "${path}" >&2
        return 1
        ;;
    esac
  done < "${new_changes}"
}

seal_on_exit() {
  local status=$?
  local cleanup_status=0
  trap - EXIT
  set +e
  restore_generated_stubs \
    "${SPCONV_SOURCE}" "${S09_STOP3_DEP_OUTPUT}/spconv_tracked_changes_before.txt" spconv \
    || cleanup_status=$?
  restore_generated_stubs \
    "${CUMM_SOURCE}" "${S09_STOP3_DEP_OUTPUT}/cumm_tracked_changes_before.txt" cumm \
    || cleanup_status=$?
  if (( status == 0 && cleanup_status != 0 )); then
    status="${cleanup_status}"
  fi
  git -C "${SPCONV_SOURCE}" status --short --untracked-files=all \
    > spconv_status_after.txt 2>&1
  git -C "${CUMM_SOURCE}" status --short --untracked-files=all \
    > cumm_status_after.txt 2>&1
  printf '%s\n' "${status}" > dependency_attestation.exit
  artifact_manifest_tmp="${S09_STOP3_DEP_OUTPUT}.artifact_sha256s.$$"
  find . -type f ! -name artifact_sha256s.txt -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > "${artifact_manifest_tmp}"
  mv "${artifact_manifest_tmp}" artifact_sha256s.txt
  find . -type f -exec chmod 0444 {} +
  find . -type d -exec chmod 0555 {} +
  exit "${status}"
}
trap seal_on_exit EXIT

python - "${SPCONV_SOURCE}" "${CUMM_SOURCE}" <<'PY' > source_identity_before.json
import json
import os
import subprocess
import sys

from fl_v3.source_identity import inspect_tracked_source_state


def record(label: str, source: str, expected_head: str, expected_state: str):
    head = subprocess.run(
        ["git", "-C", source, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    state = inspect_tracked_source_state(source)
    if head != expected_head:
        raise RuntimeError(f"{label} source HEAD drift before warm import")
    if state["sha256"] != expected_state:
        raise RuntimeError(f"{label} tracked source-state drift before warm import")
    return {"source": source, "head": head, "tracked_state": state}


spconv_source, cumm_source = sys.argv[1:]
print(json.dumps({
    "schema": "s09.stop3-dependency-source-preflight.v1",
    "spconv": record(
        "spconv",
        spconv_source,
        os.environ["S09_STOP3_DEP_EXPECTED_SPCONV_HEAD"],
        os.environ["S09_STOP3_DEP_EXPECTED_SPCONV_STATE"],
    ),
    "cumm": record(
        "cumm",
        cumm_source,
        os.environ["S09_STOP3_DEP_EXPECTED_CUMM_HEAD"],
        os.environ["S09_STOP3_DEP_EXPECTED_CUMM_STATE"],
    ),
}, indent=2, sort_keys=True))
PY

python - <<'PY' > warm_import.stdout 2> warm_import.stderr
import importlib.metadata
import json

import cumm
import cumm.tensorview
import spconv
import spconv.pytorch

print(json.dumps({
    "schema": "s09.stop3-dependency-warm-import.v1",
    "cumm_version": importlib.metadata.version("cumm"),
    "cumm_origin": cumm.__file__,
    "spconv_version": importlib.metadata.version("spconv"),
    "spconv_origin": spconv.__file__,
}, indent=2, sort_keys=True))
PY

restore_generated_stubs \
  "${SPCONV_SOURCE}" "${S09_STOP3_DEP_OUTPUT}/spconv_tracked_changes_before.txt" spconv
restore_generated_stubs \
  "${CUMM_SOURCE}" "${S09_STOP3_DEP_OUTPUT}/cumm_tracked_changes_before.txt" cumm

python dependency_probe.py \
  "${S09_STOP3_DEP_SNAPSHOT}/${CONFIG_REL}" "${SPCONV_SOURCE}" "${CUMM_SOURCE}" \
  dependency_probe_a.json > dependency_probe_a.stdout 2> dependency_probe_a.stderr
python dependency_probe.py \
  "${S09_STOP3_DEP_SNAPSHOT}/${CONFIG_REL}" "${SPCONV_SOURCE}" "${CUMM_SOURCE}" \
  dependency_probe_b.json > dependency_probe_b.stdout 2> dependency_probe_b.stderr
cmp dependency_probe_a.json dependency_probe_b.json

git -C "${SPCONV_SOURCE}" diff --name-only --no-renames | LC_ALL=C sort \
  | cmp - spconv_tracked_changes_before.txt
git -C "${CUMM_SOURCE}" diff --name-only --no-renames | LC_ALL=C sort \
  | cmp - cumm_tracked_changes_before.txt

python - dependency_probe_a.json "${actual_source_sha}" "${actual_tree}" \
  "${S09_STOP3_DEP_SNAPSHOT}/${RUNNER_REL}" <<'PY' > acceptance.json
import hashlib
import json
import os
import sys
from pathlib import Path

probe_path, source_sha, source_tree, runner_text = sys.argv[1:]
probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))
runner = Path(runner_text).resolve(strict=True)
print(json.dumps({
    "schema": "s09.stop3-dependency-acceptance.v1",
    "accepted": True,
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "source_sha": source_sha,
    "source_tree": source_tree,
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
    "torch_build_sha256": probe["torch"]["build_sha256"],
    "spconv_build_sha256": probe["spconv"]["build_sha256"],
    "spconv_source_state_sha256": probe["spconv"]["source_state"]["sha256"],
    "cumm_build_sha256": probe["cumm"]["build_sha256"],
    "cumm_source_state_sha256": probe["cumm"]["source_state"]["sha256"],
    "stable_fresh_processes": 2,
    "data_loaded": False,
    "model_constructed": False,
    "training_attempts": 0,
}, indent=2, sort_keys=True))
PY
