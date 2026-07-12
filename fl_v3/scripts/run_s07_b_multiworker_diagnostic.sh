#!/bin/bash
# Prepared S07-B multiworker diagnostic. This launcher is not execution
# authority; RUN_REQUEST Section H requires separate exact S00 approval.
#SBATCH -A naiss2025-22-1113-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=nvidia_gh200_120gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --job-name=flv3_s07b_mw_diag
#SBATCH --output=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_mw_diag_%j.out
#SBATCH --error=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/logs/s07b_mw_diag_%j.err

set -euo pipefail

readonly MINI_ROOT=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project/data/nuscenes_mini
readonly REPO=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/fl_weather_project
readonly SNAPSHOT_BASE=/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/execution_snapshots
readonly APPROVAL_SCOPE=s07b-multiworker-diagnostic-only
readonly SUPERVISOR_TIMEOUT_SECONDS=90
readonly PYTEST_FAULTHANDLER_SECONDS=30
readonly TERM_GRACE_SECONDS=5

readonly -a NODE_LABELS=(
  zip_persistent_fork
  zip_persistent_spawn
  zip_pre_ack
  zip_post_ack_error
  zip_leader_exit
  zip_post_ack_hang
  dummy_multiworker
  detection_loader_determinism_num_workers
  cuda_initialized_production_loader_spawn_persistent
)
readonly -a NODE_IDS=(
  'fl_v3/tests/test_nuscenes_zip_dataset.py::test_repeated_persistent_multiworker_reads_are_deterministic[fork]'
  'fl_v3/tests/test_nuscenes_zip_dataset.py::test_repeated_persistent_multiworker_reads_are_deterministic[spawn]'
  'fl_v3/tests/test_nuscenes_zip_dataset.py::test_explicit_fork_pre_ack_failure_never_forks_or_touches_parent_group'
  'fl_v3/tests/test_nuscenes_zip_dataset.py::test_explicit_fork_post_ack_error_preserves_primary_and_cleanup_evidence'
  'fl_v3/tests/test_nuscenes_zip_dataset.py::test_explicit_fork_post_ack_leader_exit_cleans_verified_orphan_group'
  'fl_v3/tests/test_nuscenes_zip_dataset.py::test_explicit_fork_post_ack_hang_kills_verified_group_and_descendant'
  'fl_v3/tests/test_model_task.py::test_dummy_multiworker_loader_is_spawn_and_consumes_batch'
  'fl_v3/tests/test_model_task.py::test_loader_determinism_num_workers'
  'fl_v3/tests/test_model_task.py::test_cuda_initialized_production_loader_is_spawn_persistent'
)

runtime_source_files() {
  {
    find fl_v3/src/fl_v3 -type f -name '*.py' -print
    printf '%s\n' \
      fl_v3/scripts/arrhenius_env.sh \
      fl_v3/scripts/run_s07_b_multiworker_diagnostic.sh \
      fl_v3/tests/conftest.py \
      fl_v3/tests/test_nuscenes_zip_dataset.py \
      fl_v3/tests/test_model_task.py \
      fl_v3/pyproject.toml \
      fl_v3/requirements.txt \
      fl_v3/requirements.lock.txt
  } | LC_ALL=C sort -u
}

if [ "${1:-}" = "--print-source-files" ]; then
  runtime_source_files
  exit 0
fi
if [ "${1:-}" = "--print-nodes" ]; then
  paste <(printf '%s\n' "${NODE_LABELS[@]}") <(printf '%s\n' "${NODE_IDS[@]}")
  exit 0
fi

required=(
  EXPECTED_S07B_MW_CANDIDATE_SHA
  EXPECTED_S07B_MW_EXECUTABLE_SHA
  EXPECTED_S07B_MW_LAUNCHER_SHA256
  EXPECTED_S07B_MW_SOURCE_SHA256
  EXPECTED_S07B_MW_SOURCE_LIST_SHA256
  S07B_MW_MINI_DATAROOT
  S07B_MW_OUTPUT_ROOT
  S07B_MW_APPROVAL_SCOPE
)
for name in "${required[@]}"; do
  if [ -z "${!name:-}" ]; then
    echo "${name} is required" >&2
    exit 2
  fi
done

test "${S07B_MW_APPROVAL_SCOPE}" = "${APPROVAL_SCOPE}"
test "${S07B_MW_MINI_DATAROOT}" = "${MINI_ROOT}"
test -d "${MINI_ROOT}"
test "$(readlink -f "${S07B_MW_MINI_DATAROOT}")" = "$(readlink -f "${MINI_ROOT}")"
test "${SLURM_NNODES:-1}" = "1"
test "${SLURM_NTASKS:-1}" = "1"
test "${SLURM_CPUS_PER_TASK:-8}" = "8"
test "$(uname -m)" = "aarch64"
test "$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" = "${EXPECTED_S07B_MW_LAUNCHER_SHA256}"

git -C "${REPO}" cat-file -e "${EXPECTED_S07B_MW_CANDIDATE_SHA}^{commit}"
git -C "${REPO}" cat-file -e "${EXPECTED_S07B_MW_EXECUTABLE_SHA}^{commit}"
test -z "$(
  git -C "${REPO}" diff --name-only \
    "${EXPECTED_S07B_MW_CANDIDATE_SHA}" \
    "${EXPECTED_S07B_MW_EXECUTABLE_SHA}" -- \
    fl_v3/src/fl_v3 fl_v3/tests fl_v3/scripts/arrhenius_env.sh \
    fl_v3/pyproject.toml fl_v3/requirements.txt fl_v3/requirements.lock.txt
)"

readonly SHORT_EXECUTABLE="${EXPECTED_S07B_MW_EXECUTABLE_SHA:0:12}"
readonly SNAPSHOT="${SNAPSHOT_BASE}/s07b_mw_diag_${SHORT_EXECUTABLE}"
readonly EXPECTED_OUTPUT_ROOT="/nobackup/proj/disk/naiss2024-22-991/personal/gaohui/arrhenius_fl_v3/outputs/s07b_mw_diag_${SHORT_EXECUTABLE}"
test "${S07B_MW_OUTPUT_ROOT}" = "${EXPECTED_OUTPUT_ROOT}"
test ! -e "${S07B_MW_OUTPUT_ROOT}"
test ! -e "${SNAPSHOT}"
mkdir -p "${SNAPSHOT_BASE}"
mkdir "${SNAPSHOT}"
git -C "${REPO}" archive "${EXPECTED_S07B_MW_EXECUTABLE_SHA}" | tar -x -C "${SNAPSHOT}"
cd "${SNAPSHOT}"

mapfile -t SOURCE_FILES < <(runtime_source_files)
test "${#NODE_LABELS[@]}" -eq 9
test "${#NODE_IDS[@]}" -eq 9
test "${#SOURCE_FILES[@]}" -gt 9
for path in "${SOURCE_FILES[@]}"; do
  test -f "${path}"
done
ACTUAL_SOURCE_LIST_SHA256="$(printf '%s\n' "${SOURCE_FILES[@]}" | sha256sum | awk '{print $1}')"
ACTUAL_SOURCE_SHA256="$(
  printf '%s\n' "${SOURCE_FILES[@]}" |
    while IFS= read -r path; do sha256sum "${path}"; done |
    sha256sum | awk '{print $1}'
)"
test "${ACTUAL_SOURCE_LIST_SHA256}" = "${EXPECTED_S07B_MW_SOURCE_LIST_SHA256}"
test "${ACTUAL_SOURCE_SHA256}" = "${EXPECTED_S07B_MW_SOURCE_SHA256}"
test "$(sha256sum fl_v3/scripts/run_s07_b_multiworker_diagnostic.sh | awk '{print $1}')" = \
  "${EXPECTED_S07B_MW_LAUNCHER_SHA256}"

chmod -R a-w "${SNAPSHOT}"
mkdir "${S07B_MW_OUTPUT_ROOT}"
readonly NODE_ROOT="${S07B_MW_OUTPUT_ROOT}/nodes"
readonly WORK_CWD="${S07B_MW_OUTPUT_ROOT}/work"
mkdir "${NODE_ROOT}" "${WORK_CWD}"
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]
[[ "${SHORT_EXECUTABLE}" =~ ^[0-9a-f]{12}$ ]]
JOB_TMP_PREVIOUS_UMASK="$(umask)"
readonly JOB_TMP_PREVIOUS_UMASK
umask 077
JOB_TMP="$(mktemp -d -p /tmp "flv3-s07b-${SLURM_JOB_ID}-${SHORT_EXECUTABLE}.XXXXXX")"
umask "${JOB_TMP_PREVIOUS_UMASK}"
readonly JOB_TMP
JOB_TMP_IDENTITY="$(stat -c '%d:%i' "${JOB_TMP}")"
readonly JOB_TMP_IDENTITY
readonly JOB_TMP_PATTERN="^/tmp/flv3-s07b-${SLURM_JOB_ID}-${SHORT_EXECUTABLE}\\.[A-Za-z0-9]{6}$"
cleanup_job_tmp() {
  local status=$?
  local cleanup_status=0
  local current_identity=""
  trap - EXIT
  if [[ "${JOB_TMP}" =~ ${JOB_TMP_PATTERN} ]] &&
    [ "$(dirname -- "${JOB_TMP}")" = "/tmp" ] &&
    test -d "${JOB_TMP}" && test ! -L "${JOB_TMP}"; then
    current_identity="$(stat -c '%d:%i' "${JOB_TMP}")" || cleanup_status=$?
    if [ "${cleanup_status}" -eq 0 ] &&
      [ "${current_identity}" = "${JOB_TMP_IDENTITY}" ]; then
      rm -rf -- "${JOB_TMP}" || cleanup_status=$?
    else
      cleanup_status=1
    fi
  else
    cleanup_status=1
  fi
  if [ "${status}" -eq 0 ] && [ "${cleanup_status}" -ne 0 ]; then
    status="${cleanup_status}"
  fi
  exit "${status}"
}
trap cleanup_job_tmp EXIT
test "${#JOB_TMP}" -le 48
[[ "${JOB_TMP}" =~ ${JOB_TMP_PATTERN} ]]
test "$(dirname -- "${JOB_TMP}")" = "/tmp"
test -d "${JOB_TMP}"
test ! -L "${JOB_TMP}"
test "$(stat -c '%a' "${JOB_TMP}")" = "700"

# shellcheck disable=SC1091
source "${SNAPSHOT}/fl_v3/scripts/arrhenius_env.sh"
arrhenius_load_modules build
arrhenius_activate_env

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
unset PYTEST_ADDOPTS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WORLD_SIZE=1
export NUSCENES_DATAROOT="${MINI_ROOT}"
export ARRHENIUS_NUSCENES_DATAROOT="${MINI_ROOT}"
unset NUSCENES_DATA_DIR NUSCENES_ZIP_MANIFEST ARRHENIUS_NUSCENES_ZIP_MANIFEST
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export PYTHONPATH="${SNAPSHOT}/fl_v3/src"

readonly SOURCE_LIST="${S07B_MW_OUTPUT_ROOT}/runtime_source_files.txt"
readonly SOURCE_HASHES="${S07B_MW_OUTPUT_ROOT}/runtime_source_sha256s.txt"
readonly NODES_TSV="${S07B_MW_OUTPUT_ROOT}/selected_nodes.tsv"
readonly EXECUTION_JSON="${S07B_MW_OUTPUT_ROOT}/execution_identity.json"
readonly RUN_CONFIG="${S07B_MW_OUTPUT_ROOT}/diagnostic_run_config.json"
readonly SUMMARY_JSON="${S07B_MW_OUTPUT_ROOT}/diagnostic_summary.json"
readonly CHECKSUMS="${S07B_MW_OUTPUT_ROOT}/sha256sums.txt"

printf '%s\n' "${SOURCE_FILES[@]}" > "${SOURCE_LIST}"
while IFS= read -r path; do sha256sum "${path}"; done < "${SOURCE_LIST}" > "${SOURCE_HASHES}"
paste <(printf '%s\n' "${NODE_LABELS[@]}") \
  <(printf '%s\n' "${NODE_IDS[@]}") > "${NODES_TSV}"
test "$(sha256sum "${SOURCE_LIST}" | awk '{print $1}')" = "${EXPECTED_S07B_MW_SOURCE_LIST_SHA256}"
test "$(sha256sum "${SOURCE_HASHES}" | awk '{print $1}')" = "${EXPECTED_S07B_MW_SOURCE_SHA256}"

python - "${EXECUTION_JSON}" "${NODES_TSV}" <<'PY'
import importlib.metadata
import json
import os
import platform
import socket
import sys

import cumm
import spconv
import torch

output, nodes_path = sys.argv[1:]
expected = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "pytest": "9.1.1",
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "spconv": "2.3.8",
    "cumm": "0.7.13",
    "nuscenes-devkit": "1.1.11",
    "pyquaternion": "0.9.9",
    "Pillow": "12.2.0",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"dependency identity mismatch: expected={expected!r}, actual={actual!r}")
if platform.python_version() != "3.11.15":
    raise SystemExit(f"Python identity mismatch: {platform.python_version()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("multiworker diagnostic requires exactly one visible CUDA device")
with open(nodes_path, encoding="utf-8") as stream:
    nodes = [line.rstrip("\n").split("\t", 1) for line in stream]
if len(nodes) != 9 or any(len(item) != 2 for item in nodes):
    raise SystemExit("exactly nine labelled pytest nodes are required")
record = {
    "schema": "s07b.multiworker-diagnostic-identity.v1",
    "approval_scope": os.environ["S07B_MW_APPROVAL_SCOPE"],
    "candidate_sha": os.environ["EXPECTED_S07B_MW_CANDIDATE_SHA"],
    "executable_sha": os.environ["EXPECTED_S07B_MW_EXECUTABLE_SHA"],
    "launcher_sha256": os.environ["EXPECTED_S07B_MW_LAUNCHER_SHA256"],
    "runtime_source_sha256": os.environ["EXPECTED_S07B_MW_SOURCE_SHA256"],
    "runtime_source_list_sha256": os.environ["EXPECTED_S07B_MW_SOURCE_LIST_SHA256"],
    "host": socket.gethostname(),
    "machine": platform.machine(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "torch_cuda_version": torch.version.cuda,
    "dependencies": actual,
    "selected_nodes": nodes,
    "mini_dataroot": os.environ["S07B_MW_MINI_DATAROOT"],
    "output_root": os.environ["S07B_MW_OUTPUT_ROOT"],
    "tmpdir": os.environ["TMPDIR"],
    "tmpdir_bytes": len(os.fsencode(os.environ["TMPDIR"])),
    "slurm": {
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "nnodes": os.environ.get("SLURM_NNODES", ""),
        "ntasks": os.environ.get("SLURM_NTASKS", ""),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
    },
    "pytest_disable_plugin_autoload": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"),
    "pytest_addopts": os.environ.get("PYTEST_ADDOPTS", ""),
    "zip_and_full_data_overrides_cleared": all(
        not os.environ.get(name)
        for name in (
            "NUSCENES_DATA_DIR",
            "NUSCENES_ZIP_MANIFEST",
            "ARRHENIUS_NUSCENES_ZIP_MANIFEST",
        )
    ),
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

python - "${RUN_CONFIG}" "${NODES_TSV}" \
  "${SUPERVISOR_TIMEOUT_SECONDS}" "${PYTEST_FAULTHANDLER_SECONDS}" \
  "${TERM_GRACE_SECONDS}" <<'PY'
import json
import sys

output, nodes_path, supervisor_timeout, faulthandler_timeout, term_grace = sys.argv[1:]
with open(nodes_path, encoding="utf-8") as stream:
    nodes = [line.rstrip("\n").split("\t", 1) for line in stream]
record = {
    "schema": "s07b.multiworker-diagnostic-config.v1",
    "selected_nodes": nodes,
    "supervisor_timeout_seconds": int(supervisor_timeout),
    "pytest_faulthandler_timeout_seconds": int(faulthandler_timeout),
    "term_to_kill_grace_seconds": int(term_grace),
    "subprocess_start_new_session": True,
    "continue_all_nine_without_retry": True,
    "pytest_flags": [
        "-vv", "--tb=long", "-p", "no:cacheprovider", "-o",
        f"faulthandler_timeout={faulthandler_timeout}",
    ],
    "snapshot_read_only": True,
    "output_cwd_writable": True,
    "mini_only": True,
    "diagnostic_complete_separate_from_suite_pass": True,
    "forbidden": [
        "test_model_overfit.py",
        "full-suite",
        "full-cache-or-trainval",
        "100-or-1000-step",
        "metrics-or-profile",
        "DDP-or-matrix",
        "retry-or-requeue",
    ],
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

python - "${NODES_TSV}" "${NODE_ROOT}" "${WORK_CWD}" "${SNAPSHOT}" \
  "${SUPERVISOR_TIMEOUT_SECONDS}" "${PYTEST_FAULTHANDLER_SECONDS}" \
  "${TERM_GRACE_SECONDS}" <<'PY'
import ctypes
import datetime as dt
import errno
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

(
    nodes_path,
    node_root_text,
    work_cwd,
    snapshot,
    supervisor_timeout_text,
    faulthandler_timeout_text,
    term_grace_text,
) = sys.argv[1:]
node_root = pathlib.Path(node_root_text)
supervisor_timeout = float(supervisor_timeout_text)
faulthandler_timeout = int(faulthandler_timeout_text)
term_grace = float(term_grace_text)
with open(nodes_path, encoding="utf-8") as stream:
    nodes = [line.rstrip("\n").split("\t", 1) for line in stream]
if len(nodes) != 9:
    raise SystemExit("supervisor requires exactly nine nodes")

PR_SET_CHILD_SUBREAPER = 36
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error))
supervisor_pid = os.getpid()
supervisor_pgid = os.getpgrp()


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def proc_record(pid):
    try:
        text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = text.rfind(")")
    if close < 0:
        return None
    fields = text[close + 2 :].split()
    if len(fields) < 20:
        return None
    return {
        "pid": int(pid),
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "starttime": int(fields[19]),
    }


def process_table():
    table = {}
    for entry in pathlib.Path("/proc").iterdir():
        if entry.name.isdigit():
            record = proc_record(int(entry.name))
            if record is not None:
                table[record["pid"]] = record
    return table


def descendant_records(root_pid, table):
    children = {}
    for record in table.values():
        children.setdefault(record["ppid"], []).append(record["pid"])
    found = []
    pending = list(children.get(root_pid, ()))
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        record = table.get(pid)
        if record is not None:
            found.append(record)
            pending.extend(children.get(pid, ()))
    return found


def identity_key(record):
    return (record["pid"], record["starttime"])


def identity_alive(record):
    current = proc_record(record["pid"])
    return current is not None and current["starttime"] == record["starttime"]


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def safe_killpg(pgid, sig, owned_records):
    if pgid == supervisor_pgid:
        raise RuntimeError("refusing to signal supervisor process group")
    if not any(
        record["pgid"] == pgid and identity_alive(record)
        for record in owned_records
    ):
        raise RuntimeError("refusing to signal process group without a live exact owned identity")
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False


def signal_exact(records, sig):
    sent = []
    for record in records:
        if not identity_alive(record):
            continue
        try:
            os.kill(record["pid"], sig)
            sent.append(identity_key(record))
        except ProcessLookupError:
            pass
    return sent


def reap_adopted():
    reaped = []
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid == 0:
            break
        reaped.append({"pid": pid, "status": status})
    return reaped


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def junit_counts(path):
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def wait_until(predicate, seconds, scan):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        scan()
        if predicate():
            return True
        time.sleep(0.05)
    scan()
    return predicate()


previous_identities = []
for label, node_id in nodes:
    reap_adopted()
    if any(identity_alive(record) for record in previous_identities):
        raise RuntimeError("previous node left a live process identity")
    table = process_table()
    adopted_before = [
        record for record in table.values()
        if record["ppid"] == supervisor_pid
    ]
    if adopted_before:
        raise RuntimeError(f"cross-node adopted descendants remain: {adopted_before!r}")

    attempt_root = node_root / label
    basetemp = attempt_root / "basetemp"
    attempt_root.mkdir()
    basetemp.mkdir()
    log_path = attempt_root / "pytest.log"
    junit_path = attempt_root / "pytest.junit.xml"
    exit_path = attempt_root / "pytest.exitcode"
    result_path = attempt_root / "supervisor_result.json"
    manifest_path = attempt_root / "sha256sums.txt"
    command = [
        sys.executable,
        "-X",
        "faulthandler",
        "-m",
        "pytest",
        "-vv",
        "--tb=long",
        "-p",
        "no:cacheprovider",
        "-o",
        f"faulthandler_timeout={faulthandler_timeout}",
        f"--basetemp={basetemp}",
        f"--junitxml={junit_path}",
        f"{snapshot}/{node_id}",
    ]
    tracked = {}
    reaped = []
    timeout = False
    term_group_sent = False
    kill_group_sent = False
    exact_term_sent = []
    exact_kill_sent = []
    started_at = utc_now()
    started_monotonic = time.monotonic()
    with open(log_path, "wb", buffering=0) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=work_cwd,
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        root_record = proc_record(process.pid)
        if root_record is None:
            raise RuntimeError("pytest root disappeared before identity capture")
        if root_record["pid"] != root_record["pgid"] or root_record["pid"] != root_record["sid"]:
            raise RuntimeError(f"start_new_session identity mismatch: {root_record!r}")
        if root_record["pgid"] == supervisor_pgid:
            raise RuntimeError("pytest process group aliases supervisor")

        def scan():
            table_now = process_table()
            for record in descendant_records(process.pid, table_now):
                tracked.setdefault(identity_key(record), record)
            for record in table_now.values():
                if record["ppid"] == supervisor_pid and record["pid"] != process.pid:
                    tracked.setdefault(identity_key(record), record)

        deadline = time.monotonic() + supervisor_timeout
        while process.poll() is None and time.monotonic() < deadline:
            scan()
            time.sleep(0.05)
        scan()
        if process.poll() is None:
            timeout = True
            owned_records = [root_record, *tracked.values()]
            term_group_sent = safe_killpg(
                root_record["pgid"], signal.SIGTERM, owned_records
            )
            wait_until(
                lambda: process.poll() is not None
                and not group_exists(root_record["pgid"]),
                term_grace,
                scan,
            )
            if process.poll() is None or group_exists(root_record["pgid"]):
                owned_records = [root_record, *tracked.values()]
                kill_group_sent = safe_killpg(
                    root_record["pgid"], signal.SIGKILL, owned_records
                )
            wait_until(lambda: process.poll() is not None, term_grace, scan)
        else:
            scan()

        if process.poll() is None:
            process.kill()
        returncode = process.wait(timeout=term_grace)
        scan()
        reaped.extend(reap_adopted())

        def cleanup_scan_and_reap():
            scan()
            reaped.extend(reap_adopted())
            scan()

        term_signalled = set()
        term_deadline = time.monotonic() + term_grace
        term_empty_streak = 0
        while time.monotonic() < term_deadline:
            cleanup_scan_and_reap()
            live_now = [
                record for record in tracked.values()
                if identity_alive(record) and identity_key(record) not in term_signalled
            ]
            if live_now:
                sent = signal_exact(live_now, signal.SIGTERM)
                exact_term_sent.extend(sent)
                term_signalled.update(tuple(item) for item in sent)
            all_live = [record for record in tracked.values() if identity_alive(record)]
            adopted_now = [
                record for record in process_table().values()
                if record["ppid"] == supervisor_pid
            ]
            if not all_live and not adopted_now:
                term_empty_streak += 1
                if term_empty_streak >= 3:
                    break
            else:
                term_empty_streak = 0
            time.sleep(0.05)

        kill_signalled = set()
        kill_deadline = time.monotonic() + term_grace
        kill_empty_streak = 0
        while time.monotonic() < kill_deadline:
            cleanup_scan_and_reap()
            live_now = [
                record for record in tracked.values()
                if identity_alive(record) and identity_key(record) not in kill_signalled
            ]
            if live_now:
                sent = signal_exact(live_now, signal.SIGKILL)
                exact_kill_sent.extend(sent)
                kill_signalled.update(tuple(item) for item in sent)
            all_live = [record for record in tracked.values() if identity_alive(record)]
            adopted_now = [
                record for record in process_table().values()
                if record["ppid"] == supervisor_pid
            ]
            if not all_live and not adopted_now:
                kill_empty_streak += 1
                if kill_empty_streak >= 3:
                    break
            else:
                kill_empty_streak = 0
            time.sleep(0.05)
        cleanup_scan_and_reap()

    root_group_absent = wait_until(
        lambda: not group_exists(root_record["pgid"]), term_grace, scan
    )
    identities_absent = not any(identity_alive(record) for record in tracked.values())
    table_after = process_table()
    adopted_after = [
        record for record in table_after.values()
        if record["ppid"] == supervisor_pid
    ]
    cleanup_ok = root_group_absent and identities_absent and not adopted_after
    ended_at = utc_now()
    elapsed = time.monotonic() - started_monotonic
    exit_path.write_text(f"{returncode}\n", encoding="utf-8")
    counts = junit_counts(junit_path)
    result = {
        "schema": "s07b.multiworker-node-supervisor.v1",
        "label": label,
        "node_id": node_id,
        "command": command,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "supervisor_timeout_seconds": supervisor_timeout,
        "pytest_faulthandler_timeout_seconds": faulthandler_timeout,
        "term_grace_seconds": term_grace,
        "root_identity": root_record,
        "returncode": returncode,
        "timed_out": timeout,
        "term_whole_root_group_sent": term_group_sent,
        "kill_whole_root_group_sent": kill_group_sent,
        "supervisor_cleanup_intervened": bool(
            term_group_sent or kill_group_sent or exact_term_sent or exact_kill_sent
        ),
        "tracked_descendant_identities": sorted(tracked.values(), key=lambda item: (item["pid"], item["starttime"])),
        "exact_descendant_term_sent": exact_term_sent,
        "exact_descendant_kill_sent": exact_kill_sent,
        "reaped_adopted_children": reaped,
        "root_group_absent_final": root_group_absent,
        "tracked_identities_absent_final": identities_absent,
        "adopted_children_absent_final": not adopted_after,
        "cleanup_ok": cleanup_ok,
        "junit_present": junit_path.is_file(),
        "junit_counts": counts,
        "log_bytes": log_path.stat().st_size,
        "log_sha256": sha256(log_path),
        "junit_sha256": sha256(junit_path) if junit_path.is_file() else None,
        "cross_node_descendants_permitted": False,
    }
    with open(result_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    artifacts = [exit_path, log_path, result_path]
    if junit_path.is_file():
        artifacts.append(junit_path)
    with open(manifest_path, "w", encoding="utf-8") as stream:
        for artifact in sorted(artifacts, key=lambda item: item.name):
            stream.write(f"{sha256(artifact)}  {artifact.name}\n")
    subprocess.run(
        ["sha256sum", "-c", manifest_path.name],
        cwd=attempt_root,
        check=True,
    )
    previous_identities = [root_record, *tracked.values()]

reap_adopted()
if any(identity_alive(record) for record in previous_identities):
    raise RuntimeError("final node left a live process identity")
PY

set +e
python - "${SUMMARY_JSON}" "${NODES_TSV}" "${NODE_ROOT}" <<'PY'
import json
import pathlib
import sys

output, nodes_path, node_root_text = sys.argv[1:]
node_root = pathlib.Path(node_root_text)
with open(nodes_path, encoding="utf-8") as stream:
    nodes = [line.rstrip("\n").split("\t", 1) for line in stream]
entries = []
artifact_complete = True
for label, node_id in nodes:
    attempt_root = node_root / label
    required = [
        attempt_root / "pytest.exitcode",
        attempt_root / "pytest.log",
        attempt_root / "supervisor_result.json",
        attempt_root / "sha256sums.txt",
    ]
    artifact_complete = artifact_complete and all(path.is_file() for path in required)
    with open(attempt_root / "supervisor_result.json", encoding="utf-8") as stream:
        result = json.load(stream)
    entries.append(result)

def node_passed(entry):
    counts = entry["junit_counts"]
    return (
        entry["returncode"] == 0
        and not entry["timed_out"]
        and not entry["supervisor_cleanup_intervened"]
        and entry["cleanup_ok"]
        and entry["junit_present"]
        and counts is not None
        and counts["tests"] > 0
        and counts["failures"] == 0
        and counts["errors"] == 0
        and counts["skipped"] == 0
    )

diagnostic_complete = (
    len(entries) == 9
    and artifact_complete
    and all(entry["cleanup_ok"] for entry in entries)
    and all(entry["log_bytes"] > 0 for entry in entries)
)
suite_pass = diagnostic_complete and all(node_passed(entry) for entry in entries)
record = {
    "schema": "s07b.multiworker-diagnostic-summary.v1",
    "diagnostic_complete": diagnostic_complete,
    "suite_pass": suite_pass,
    "expected_nodes": 9,
    "observed_nodes": len(entries),
    "artifact_complete": artifact_complete,
    "all_process_groups_and_identities_cleaned": all(entry["cleanup_ok"] for entry in entries),
    "failures_and_timeouts_are_preserved_diagnostics": True,
    "scheduler_success_is_harness_completion_only": True,
    "entries": entries,
    "totals_from_present_junit": {
        key: sum((entry["junit_counts"] or {}).get(key, 0) for entry in entries)
        for key in ("tests", "failures", "errors", "skipped")
    },
}
with open(output, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2, sort_keys=True)
    stream.write("\n")
if not diagnostic_complete:
    raise SystemExit(1)
PY
summary_exit=$?
set -e

find "${S07B_MW_OUTPUT_ROOT}" -type f \
  ! -path "${JOB_TMP}/*" \
  ! -path '*/basetemp/*' \
  -printf '%P\n' | LC_ALL=C sort | while IFS= read -r path; do
    sha256sum "${S07B_MW_OUTPUT_ROOT}/${path}"
  done > "${JOB_TMP}/sha256sums.pending"
mv "${JOB_TMP}/sha256sums.pending" "${CHECKSUMS}"
(
  cd "${S07B_MW_OUTPUT_ROOT}"
  sed "s#  ${S07B_MW_OUTPUT_ROOT}/#  #" "${CHECKSUMS}" > "${CHECKSUMS}.relative"
  mv "${CHECKSUMS}.relative" "${CHECKSUMS}"
  sha256sum -c "${CHECKSUMS}"
)

test "${summary_exit}" -eq 0
