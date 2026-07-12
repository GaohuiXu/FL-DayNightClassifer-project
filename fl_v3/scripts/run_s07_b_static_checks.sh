#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO}"

python3 -m py_compile \
  fl_v3/src/fl_v3/config/resolved.py \
  fl_v3/src/fl_v3/data/nuscenes/dataset.py \
  fl_v3/src/fl_v3/data/nuscenes/zip_backend.py \
  fl_v3/src/fl_v3/eval/detection_eval.py \
  fl_v3/src/fl_v3/models/fusion/collate.py \
  fl_v3/src/fl_v3/models/fusion/detector.py \
  fl_v3/src/fl_v3/models/fusion/losses.py \
  fl_v3/src/fl_v3/training/tasks.py \
  fl_v3/src/fl_v3/utils/runtime.py \
  fl_v3/scripts/centralized_train.py \
  fl_v3/tests/test_nuscenes_zip_dataset.py \
  fl_v3/tests/test_s06_checkpoint_resume.py \
  fl_v3/tests/test_s06_loader_eval.py \
  fl_v3/tests/test_s06_resolved_config.py \
  fl_v3/tests/test_s07_b_data_lifecycle.py \
  fl_v3/tests/test_s07_b_integration.py

for config in fl_v3/configs/s07_b_*.json; do
  python3 -m json.tool "${config}" >/dev/null
done

PYTHONPATH=fl_v3/src python3 - <<'PY'
from pathlib import Path

from fl_v3.config import ConfigError, load_resolved_config

paths = sorted(Path("fl_v3/configs").glob("s07_b_*.json"))
assert len(paths) == 5
for path in paths:
    try:
        load_resolved_config(path)
    except ConfigError as exc:
        assert "template_only" in str(exc), (path, exc)
    else:
        raise AssertionError(f"non-runnable template unexpectedly resolved: {path}")
PY

sha256sum fl_v3/configs/s07_b_*.json
