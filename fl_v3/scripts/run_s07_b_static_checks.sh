#!/usr/bin/env bash
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO}"

readonly -a S07_B_RUNTIME_LAUNCHERS=(
  fl_v3/scripts/run_s07_b_runtime_tests.sh
  fl_v3/scripts/run_s07_b_diagnostic_tests.sh
  fl_v3/scripts/run_s07_b_dummy_attribution.sh
  fl_v3/scripts/run_s07_b_postremediation_focused.sh
  fl_v3/scripts/run_s07_b_multiworker_diagnostic.sh
)
for launcher in "${S07_B_RUNTIME_LAUNCHERS[@]}"; do
  bash -n "${launcher}"
done

python3 - "${S07_B_RUNTIME_LAUNCHERS[@]}" <<'PY'
from pathlib import Path
import sys

required = (
    '[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]',
    '[[ "${SHORT_EXECUTABLE}" =~ ^[0-9a-f]{12}$ ]]',
    'JOB_TMP_PREVIOUS_UMASK="$(umask)"',
    'readonly JOB_TMP_PREVIOUS_UMASK',
    'mktemp -d -p /tmp "flv3-s07b-${SLURM_JOB_ID}-${SHORT_EXECUTABLE}.XXXXXX"',
    'umask "${JOB_TMP_PREVIOUS_UMASK}"',
    'readonly JOB_TMP_PATTERN="^/tmp/flv3-s07b-${SLURM_JOB_ID}-${SHORT_EXECUTABLE}\\\\.[A-Za-z0-9]{6}$"',
    'JOB_TMP_IDENTITY="$(stat -c \'%d:%i\' "${JOB_TMP}")"',
    'readonly JOB_TMP_IDENTITY',
    'trap cleanup_job_tmp EXIT',
    'test "${#JOB_TMP}" -le 48',
    '[[ "${JOB_TMP}" =~ ${JOB_TMP_PATTERN} ]]',
    'test "$(dirname -- "${JOB_TMP}")" = "/tmp"',
    'test ! -L "${JOB_TMP}"',
    'test "$(stat -c \'%a\' "${JOB_TMP}")" = "700"',
    '[ "${current_identity}" != "${JOB_TMP_IDENTITY}" ]',
    'rm -rf -- "${JOB_TMP}" 2>/dev/null',
    'if [ "${status}" -eq 0 ] && [ "${cleanup_status}" -ne 0 ]; then',
    'status="${cleanup_status}"',
    'exit "${status}"',
    'export TMPDIR="${JOB_TMP}"',
    'export TMP="${JOB_TMP}"',
    'export TEMP="${JOB_TMP}"',
)
for name in sys.argv[1:]:
    text = Path(name).read_text(encoding="utf-8")
    missing = [token for token in required if token not in text]
    assert not missing, (name, missing)
    identity = text.index("JOB_TMP_IDENTITY=")
    trap = text.index("trap cleanup_job_tmp EXIT")
    assertions = text.index('test "${#JOB_TMP}" -le 48')
    activate = text.index("arrhenius_activate_env")
    export = text.index('export TMPDIR="${JOB_TMP}"')
    assert identity < trap < assertions, name
    assert activate < export, name
    assert '/outputs/' not in text[text.index('JOB_TMP="$(mktemp'):trap], name
    expected_tag = (
        f'readonly JOB_TMP_CLEANUP_TAG="S07B_TMP_CLEANUP_FAILURE:'
        f'{Path(name).stem}"'
    )
    assert expected_tag in text, (name, expected_tag)
    for reason in (
        "path_pattern", "dirname", "symlink", "directory", "stat",
        "device_inode", "rm",
    ):
        token = (
            'printf \'%s\\n\' "${JOB_TMP_CLEANUP_TAG} '
            f'reason={reason}" >&2'
        )
        assert text.count(token) == 1, (name, reason)
    assert 'stat -c \'%d:%i\' "${JOB_TMP}" 2>/dev/null' in text, name

test_text = Path("fl_v3/tests/test_nuscenes_zip_dataset.py").read_text(
    encoding="utf-8"
)
for token in (
    'reaped_status = report["post_sigkill_state"]["reaped_identities"]',
    'assert os.WIFSIGNALED(reaped_status)',
    'assert os.WTERMSIG(reaped_status) == signal.SIGKILL',
):
    assert token in test_text, token
print(f"short TMPDIR contract: {len(sys.argv) - 1} launchers OK")
PY

if [ "${1:-}" = "--launcher-contract-only" ]; then
  exit 0
fi

python3 -m py_compile \
  fl_v3/src/fl_v3/config/resolved.py \
  fl_v3/src/fl_v3/data/nuscenes/dataset.py \
  fl_v3/src/fl_v3/data/nuscenes/zip_backend.py \
  fl_v3/src/fl_v3/eval/detection_eval.py \
  fl_v3/src/fl_v3/attacks/fusion_ablation.py \
  fl_v3/src/fl_v3/models/fusion/collate.py \
  fl_v3/src/fl_v3/models/fusion/detector.py \
  fl_v3/src/fl_v3/models/fusion/losses.py \
  fl_v3/src/fl_v3/training/tasks.py \
  fl_v3/src/fl_v3/utils/runtime.py \
  fl_v3/scripts/centralized_train.py \
  fl_v3/scripts/arrhenius_mini_matrix.py \
  fl_v3/scripts/t4_readiness_eval.py \
  fl_v3/scripts/t5_attack_eval.py \
  fl_v3/scripts/_t4_fd_diagnose.py \
  fl_v3/scripts/t3_trainval_reeval_fullval.py \
  fl_v3/scripts/p3_crt_probe.py \
  fl_v3/scripts/p3_grad_conflict.py \
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
