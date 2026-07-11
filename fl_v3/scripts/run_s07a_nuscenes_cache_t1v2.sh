#!/bin/bash
# Full trainval t1.v2 cache materialization using the already accepted S01
# manifest. DO NOT SUBMIT while handoffs/S07/RUN_REQUEST.md is PENDING.
# This launcher does not rebuild/replace the manifest, profile a model, or train.
#
# Required immutable submission variables:
#   EXPECTED_S07A_SHA=<approved-implementation-commit>
#   EXPECTED_S07A_STATE_HASH=<approved-runtime-source-hash>
#   S07A_ACCEPTED_MANIFEST=/nobackup/.../nuscenes_trainval_zip_manifest.sqlite
#   S07A_ACCEPTED_MANIFEST_HASH=<logical-hash>
#   S07A_ACCEPTED_MANIFEST_FILE_SHA256=<sqlite-file-sha256>
#   S07A_OUTPUT_ROOT=/nobackup/.../immutable-output
#     sbatch fl_v3/scripts/run_s07a_nuscenes_cache_t1v2.sh
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --job-name=flv3_s07a_cache_t1v2
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_cache_t1v2_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07a_cache_t1v2_%j.err
set -euo pipefail

required=(
  EXPECTED_S07A_SHA
  EXPECTED_S07A_STATE_HASH
  S07A_ACCEPTED_MANIFEST
  S07A_ACCEPTED_MANIFEST_HASH
  S07A_ACCEPTED_MANIFEST_FILE_SHA256
  S07A_OUTPUT_ROOT
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done
if [ -e "${S07A_OUTPUT_ROOT}" ]; then
  echo "Refusing to reuse/overwrite S07A_OUTPUT_ROOT=${S07A_OUTPUT_ROOT}" >&2
  exit 2
fi

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"
ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "${ACTUAL_SHA}" != "${EXPECTED_S07A_SHA}" ]; then
  echo "SHA mismatch: expected=${EXPECTED_S07A_SHA} actual=${ACTUAL_SHA}" >&2
  exit 2
fi
if [ -n "$(git branch --show-current)" ] || [ -n "$(git status --short)" ]; then
  echo "S07-A cache execution requires a clean detached worktree" >&2
  exit 2
fi
runtime_source_files() {
  # Importing ``fl_v3.data.nuscenes`` executes its package initializer and eager
  # imports dataset/partition.  Keep the complete tracked package plus every
  # local dependency reached outside it in the attested set.  The explicit
  # package initializers matter because Python executes them before submodules.
  {
    git ls-files -- 'fl_v3/src/fl_v3/data/nuscenes/*.py'
    printf '%s\n' \
      fl_v3/src/fl_v3/__init__.py \
      fl_v3/src/fl_v3/data/__init__.py \
      fl_v3/src/fl_v3/data/partition.py \
      fl_v3/src/fl_v3/utils/__init__.py \
      fl_v3/src/fl_v3/utils/runtime.py \
      fl_v3/scripts/build_nuscenes_cache.py \
      fl_v3/scripts/run_s07a_nuscenes_cache_t1v2.sh \
      fl_v3/scripts/arrhenius_env.sh \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt
  } | LC_ALL=C sort -u
}
S07A_STATE_HASH="$(runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done | sha256sum | awk '{print $1}')"
if [ "${S07A_STATE_HASH}" != "${EXPECTED_S07A_STATE_HASH}" ]; then
  echo "S07-A execution-state hash mismatch: expected=${EXPECTED_S07A_STATE_HASH} actual=${S07A_STATE_HASH}" >&2
  exit 2
fi

# shellcheck disable=SC1091
source fl_v3/scripts/arrhenius_env.sh
arrhenius_load_modules build
module load nuScenes-data/1.0-map-1.3-zip
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUSCENES_DATAROOT="${NUSCENES_DATA_DIR}"
export NUSCENES_ZIP_MANIFEST="$(realpath "${S07A_ACCEPTED_MANIFEST}")"
CACHE_DIR="${S07A_OUTPUT_ROOT}/info_cache_msweep10"
IDENTITY_JSON="${S07A_OUTPUT_ROOT}/execution_identity.json"
CACHE_IDENTITY_JSON="${S07A_OUTPUT_ROOT}/cache_identity.json"
SOURCE_HASHES="${S07A_OUTPUT_ROOT}/runtime_source_sha256s.txt"

python - "${S07A_OUTPUT_ROOT}" "${NUSCENES_DATAROOT}" "${NUSCENES_ZIP_MANIFEST}" \
  "${S07A_ACCEPTED_MANIFEST_HASH}" "${S07A_ACCEPTED_MANIFEST_FILE_SHA256}" <<'PY'
import hashlib
import os
import sys

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import TRAINVAL_ARCHIVE_NAMES, manifest_summary

output, dataroot, manifest, expected_logical, expected_file = sys.argv[1:]
P.resolve_writable(output, dataroot)
report = P.verify_dataset("v1.0-trainval", dataroot)
if report["blob_backend"] != "zip":
    raise SystemExit(f"expected ZIP backend, got {report['blob_backend']!r}")
summary = manifest_summary(manifest)
if tuple(summary["archive_names"]) != TRAINVAL_ARCHIVE_NAMES:
    raise SystemExit("accepted manifest does not declare exact trainval01..trainval10 set")
if summary["manifest_hash"] != expected_logical:
    raise SystemExit(
        f"manifest logical hash mismatch: expected={expected_logical} "
        f"actual={summary['manifest_hash']}"
    )
digest = hashlib.sha256()
with open(manifest, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
actual_file = digest.hexdigest()
if actual_file != expected_file:
    raise SystemExit(
        f"manifest file SHA-256 mismatch: expected={expected_file} actual={actual_file}"
    )
PY

mkdir -p "${CACHE_DIR}"
runtime_source_files | while IFS= read -r path; do
  sha256sum "${path}"
done > "${SOURCE_HASHES}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${S07A_STATE_HASH}"

python - "${IDENTITY_JSON}" "${ACTUAL_SHA}" "${S07A_STATE_HASH}" \
  "${NUSCENES_DATAROOT}" "${NUSCENES_ZIP_MANIFEST}" \
  "${S07A_ACCEPTED_MANIFEST_HASH}" "${S07A_ACCEPTED_MANIFEST_FILE_SHA256}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

output, git_sha, source_hash, dataroot, manifest, manifest_hash, manifest_file_hash = sys.argv[1:]
record = {
    "schema": "s07a.nuscenes-cache-t1v2-execution.v1",
    "git_sha": git_sha,
    "runtime_source_sha256": source_hash,
    "runtime_source_list": "runtime_source_sha256s.txt",
    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "python_runtime": sys.version,
    "dependency_versions": {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "nuscenes-devkit",
            "pyquaternion",
            "torch",
            "Pillow",
        )
    },
    "dataroot": os.path.abspath(dataroot),
    "manifest_path": os.path.abspath(manifest),
    "manifest_hash": manifest_hash,
    "manifest_file_sha256": manifest_file_hash,
    "n_sweeps": 10,
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

python fl_v3/scripts/build_nuscenes_cache.py \
  --dataroot "${NUSCENES_DATAROOT}" \
  --version v1.0-trainval \
  --splits train val \
  --n-sweeps 10 \
  --cache-dir "${CACHE_DIR}" \
  --rebuild

python - "${CACHE_IDENTITY_JSON}" "${CACHE_DIR}" <<'PY'
import hashlib
import json
import os
import sys

from fl_v3.data.nuscenes import info_cache as IC

output, cache_dir = sys.argv[1:]
expected_counts = {
    "train": {"n_samples": 28130, "n_boxes": 944881},
    "val": {"n_samples": 6019, "n_boxes": 187528},
}
records = {}
for split in ("train", "val"):
    infos, meta = IC.load_cache(
        cache_dir, "v1.0-trainval", split, n_sweeps=10
    )
    actual_counts = {
        "n_samples": len(infos),
        "n_boxes": sum(len(info["gt_ann_tokens"]) for info in infos),
    }
    expected = expected_counts[split]
    meta_counts = {
        "n_samples": meta.get("n_samples"),
        "n_boxes": meta.get("n_boxes"),
    }
    if actual_counts != expected or meta_counts != expected:
        raise SystemExit(
            f"{split} cache count mismatch: expected={expected}, "
            f"actual={actual_counts}, metadata={meta_counts}"
        )
    pkl_path, sidecar_path = IC.cache_paths(
        cache_dir, "v1.0-trainval", split, n_sweeps=10
    )
    files = {}
    for label, path in (("pickle", pkl_path), ("sidecar", sidecar_path)):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[label] = {
            "path": os.path.abspath(path),
            "bytes": os.path.getsize(path),
            "sha256": digest.hexdigest(),
        }
    records[split] = {
        "expected_counts": expected,
        "actual_counts": actual_counts,
        "metadata_counts": meta_counts,
        "meta": meta,
        "files": files,
    }
with open(output, "w", encoding="utf-8") as stream:
    json.dump(
        {"schema": "s07a.nuscenes-cache-t1v2-identity.v1", "splits": records},
        stream,
        indent=2,
        sort_keys=True,
    )
    stream.write("\n")
PY

sha256sum \
  "${IDENTITY_JSON}" \
  "${SOURCE_HASHES}" \
  "${CACHE_IDENTITY_JSON}" \
  "${CACHE_DIR}"/*.pkl \
  "${CACHE_DIR}"/*.meta.json \
  > "${S07A_OUTPUT_ROOT}/sha256sums.txt"
echo "[S07-A cache] generated checksums:"
cat "${S07A_OUTPUT_ROOT}/sha256sums.txt"
echo "[S07-A cache] verifying generated files against sha256sums.txt"
sha256sum -c "${S07A_OUTPUT_ROOT}/sha256sums.txt"
echo "[S07-A cache] checksum verification completed"
