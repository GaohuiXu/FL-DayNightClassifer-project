"""S01 directory/ZIP byte, decoded-array, multi-sweep, and worker parity."""
from __future__ import annotations

import copy
import errno
import gc
import multiprocessing
import os
import signal
import time
import traceback
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes.zip_backend import (
    NuScenesBlobStore,
    TRAINVAL_ARCHIVE_NAMES,
    build_zip_manifest,
)


def _toy_info(cam_paths, lidar_path, sweep_paths):
    eye4 = np.eye(4, dtype=np.float64)
    sweeps = []
    for index, rel_path in enumerate(sweep_paths, start=1):
        transform = eye4.copy()
        transform[0, 3] = float(index)
        sweeps.append(
            {
                "rel_path": rel_path,
                "sweep2keylidar": transform,
                "dt": index * 0.05,
                "_raw": (rel_path, [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [1, 0, 0, 0], index),
            }
        )
    return {
        "sample_token": "sample-0",
        "_cache_n_sweeps": 10,
        "scene_token": "scene-0",
        "log_token": "log-0",
        "location": "boston-seaport",
        "timestamp": 123,
        "cam_order": tuple(P.CAMERA_CHANNELS),
        "cam_rel_paths": list(cam_paths),
        "lidar_rel_path": lidar_path,
        "lidar_sweeps": sweeps,
        "cam_intrinsics": np.repeat(np.eye(3, dtype=np.float64)[None], 6, axis=0),
        "lidar2img": np.repeat(eye4[None], 6, axis=0),
        "cam2ego": np.repeat(eye4[None], 6, axis=0),
        "ego2global_cam": np.repeat(eye4[None], 6, axis=0),
        "lidar2ego": eye4.copy(),
        "ego2global_lidar": eye4.copy(),
        "gt_boxes": np.zeros((0, 7), dtype=np.float64),
        "gt_velocity": np.zeros((0, 2), dtype=np.float64),
        "gt_labels": np.zeros((0,), dtype=np.int64),
        "gt_names": [],
        "gt_num_lidar_pts": np.zeros((0,), dtype=np.int64),
        "gt_visibility": np.zeros((0,), dtype=np.int64),
        "gt_in_range": np.zeros((0,), dtype=bool),
        "gt_attribute": [],
        "gt_instance_tokens": [],
        "gt_ann_tokens": [],
    }


class _DebugStateDataset(torch.utils.data.Dataset):
    """Test-only wrapper that exposes the worker-local blob-store lifecycle."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["_zip_debug_state"] = self.dataset.blob_store.debug_state()
        return sample


@pytest.fixture()
def directory_and_zip(tmp_path):
    directory_root = tmp_path / "directory"
    cam_paths = []
    all_paths = []
    for index, channel in enumerate(P.CAMERA_CHANNELS):
        rel = f"samples/{channel}/{index}.jpg"
        path = directory_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[..., 0] = index * 20
        image[..., 1] = np.arange(5, dtype=np.uint8)[None]
        Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=95)
        cam_paths.append(rel)
        all_paths.append(rel)

    lidar_path = "samples/LIDAR_TOP/key.pcd.bin"
    sweep_paths = [f"sweeps/LIDAR_TOP/sweep{index}.pcd.bin" for index in range(1, 10)]
    clouds = {
        lidar_path: np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.float32),
    }
    for index, rel in enumerate(sweep_paths, start=1):
        clouds[rel] = np.array(
            [[index * 10 + 1, index * 10 + 2, index * 10 + 3, index * 10 + 4, index * 10 + 5]],
            dtype=np.float32,
        )
    for rel, cloud in clouds.items():
        path = directory_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cloud.tobytes())
        all_paths.append(rel)

    zip_root = tmp_path / "zip"
    zip_root.mkdir()
    buckets = {name: [] for name in TRAINVAL_ARCHIVE_NAMES}
    for index, rel in enumerate(all_paths):
        buckets[TRAINVAL_ARCHIVE_NAMES[index % len(TRAINVAL_ARCHIVE_NAMES)]].append(
            (rel, (directory_root / rel).read_bytes())
        )
    for name, members in buckets.items():
        with zipfile.ZipFile(zip_root / name, "w", compression=zipfile.ZIP_STORED) as archive:
            for rel, payload in members:
                archive.writestr(rel, payload)
    manifest = tmp_path / "manifest.sqlite"
    build_zip_manifest(str(zip_root), str(manifest))
    info = _toy_info(cam_paths, lidar_path, sweep_paths)
    return directory_root, zip_root, manifest, info, all_paths


def test_raw_bytes_match_directory_for_every_referenced_member(directory_and_zip):
    directory_root, zip_root, manifest, _info, paths = directory_and_zip
    directory = NuScenesBlobStore(str(directory_root))
    zipped = NuScenesBlobStore(str(zip_root), manifest_path=str(manifest))
    assert directory.read_many(paths) == zipped.read_many(paths)


def test_real_mini_directory_zip_bytes_and_decoded_arrays_match(
    nusc_mini, dataroot, tmp_path
):
    """Package only two real mini samples (one scene start, one full history)."""
    chosen = {}
    for token in IC.split_sample_tokens(nusc_mini, "mini_train"):
        sample = nusc_mini.get("sample", token)
        sample_data = nusc_mini.get("sample_data", sample["data"][P.LIDAR_CHANNEL])
        depth = 0
        current = sample_data
        while depth < 9 and current["prev"]:
            current = nusc_mini.get("sample_data", current["prev"])
            depth += 1
        if depth == 0 and "scene_start" not in chosen:
            chosen["scene_start"] = token
        if depth == 9 and "full_history" not in chosen:
            chosen["full_history"] = token
        if len(chosen) == 2:
            break
    assert set(chosen) == {"scene_start", "full_history"}
    tokens = [chosen["scene_start"], chosen["full_history"]]
    infos = IC.build_info_list(nusc_mini, tokens, dataroot, n_sweeps=10)
    depths = {info["sample_token"]: len(info["lidar_sweeps"]) for info in infos}
    assert depths == {chosen["scene_start"]: 0, chosen["full_history"]: 9}

    members = []
    for info in infos:
        members.extend(info["cam_rel_paths"])
        members.append(info["lidar_rel_path"])
        members.extend(sweep["rel_path"] for sweep in info["lidar_sweeps"])
    members = list(dict.fromkeys(members))
    zip_root = tmp_path / "mini_zip"
    zip_root.mkdir()
    buckets = {name: [] for name in TRAINVAL_ARCHIVE_NAMES}
    for index, rel in enumerate(members):
        with open(P.abspath_from_relative(rel, dataroot), "rb") as stream:
            payload = stream.read()
        buckets[TRAINVAL_ARCHIVE_NAMES[index % 10]].append((rel, payload))
    for archive_name, archive_members in buckets.items():
        with zipfile.ZipFile(
            zip_root / archive_name, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for rel, payload in archive_members:
                archive.writestr(rel, payload)
    manifest = tmp_path / "mini_manifest.sqlite"
    build_zip_manifest(str(zip_root), str(manifest))

    directory_store = NuScenesBlobStore(dataroot)
    zip_store = NuScenesBlobStore(str(zip_root), manifest_path=str(manifest))
    directory_payloads = directory_store.read_many(members)
    zip_payloads = zip_store.read_many(members)
    assert directory_payloads == zip_payloads

    directory_ds = DS.NuScenesMultimodalDataset(
        infos, dataroot, sample_tokens=tokens, n_sweeps=10
    )
    zip_ds = DS.NuScenesMultimodalDataset(
        infos,
        str(zip_root),
        sample_tokens=tokens,
        n_sweeps=10,
        zip_manifest=str(manifest),
    )
    for index in range(2):
        directory_sample = directory_ds[index]
        zip_sample = zip_ds[index]
        assert DS.sample_image_sha256(directory_sample) == DS.sample_image_sha256(zip_sample)
        assert torch.equal(directory_sample["images"], zip_sample["images"])
        assert torch.equal(directory_sample["lidar_points"], zip_sample["lidar_points"])
        assert torch.equal(directory_sample["gt_boxes"], zip_sample["gt_boxes"])
    directory_ds.close()
    zip_ds.close()
    directory_store.close()
    zip_store.close()


@pytest.mark.parametrize("n_sweeps", [1, 10])
def test_decoded_image_and_lidar_arrays_match_directory(directory_and_zip, n_sweeps):
    directory_root, zip_root, manifest, info, _paths = directory_and_zip
    runtime_info = copy.deepcopy(info)
    runtime_info["_cache_n_sweeps"] = n_sweeps
    if n_sweeps == 1:
        runtime_info.pop("lidar_sweeps")
    directory_ds = DS.NuScenesMultimodalDataset(
        [runtime_info], str(directory_root), n_sweeps=n_sweeps
    )
    zip_ds = DS.NuScenesMultimodalDataset(
        [runtime_info], str(zip_root), n_sweeps=n_sweeps, zip_manifest=str(manifest)
    )
    expected = directory_ds[0]
    actual = zip_ds[0]
    assert actual["cam_order"] == tuple(P.CAMERA_CHANNELS)
    assert torch.equal(actual["images"], expected["images"])
    assert torch.equal(actual["lidar_points"], expected["lidar_points"])
    assert torch.equal(actual["lidar2img"], expected["lidar2img"])
    assert actual["images"].dtype == torch.uint8
    assert actual["lidar_points"].dtype == torch.float32
    assert actual["lidar_points"].numpy().flags.writeable
    if n_sweeps == 10:
        # key (2 points) + nine one-point sweeps; dt is appended.
        assert tuple(actual["lidar_points"].shape) == (11, 6)
        expected_dt = torch.tensor([0.0, 0.0, *[index * 0.05 for index in range(1, 10)]])
        assert torch.equal(actual["lidar_points"][:, -1], expected_dt)


def test_legacy_absolute_lidar_and_multisweep_paths_use_zip_backend(
    directory_and_zip, monkeypatch
):
    directory_root, zip_root, manifest, info, _paths = directory_and_zip
    expected_key = DS._load_lidar(str(directory_root / info["lidar_rel_path"]))
    expected_sweeps = DS._load_multisweep(info, str(directory_root), 10)
    monkeypatch.setenv("NUSCENES_ZIP_MANIFEST", str(manifest))
    actual_key = DS._load_lidar(str(zip_root / info["lidar_rel_path"]))
    assert np.array_equal(actual_key, expected_key)
    actual_sweeps = DS._load_multisweep(info, str(zip_root), 10)
    assert np.array_equal(actual_sweeps, expected_sweeps)


def _add_cleanup_notes(primary, cleanup_errors):
    for label, error in cleanup_errors:
        primary.add_note(f"cleanup failure [{label}]: {error!r}")


def _persistent_lifecycle(
    zip_root,
    manifest,
    base_info,
    start_method,
    *,
    force_lifecycle_error=False,
    force_cleanup_evidence=False,
):
    infos = []
    for index in range(4):
        info = copy.deepcopy(base_info)
        info["sample_token"] = f"sample-{index}"
        infos.append(info)
    dataset = DS.NuScenesMultimodalDataset(
        infos, str(zip_root), n_sweeps=10, zip_manifest=str(manifest)
    )
    # Open parent state first; forked workers must not reuse it.
    parent_sample = dataset[0]
    debug_dataset = _DebugStateDataset(dataset)
    loader = DS.make_loader(
        debug_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        seed=123,
        multiprocessing_context=start_method,
    )
    primary = None
    primary_traceback = None
    workers = []
    worker_identities = []
    cleanup_errors = []
    try:
        epochs = []
        lifecycle_epochs = []
        for epoch_index in range(2):
            epoch = []
            lifecycle = {}
            for batch in loader:
                sample = batch[0]
                state = sample.pop("_zip_debug_state")
                assert state["owner_pid"] == state["current_pid"]
                lifecycle[state["owner_pid"]] = state
                epoch.append(
                    (
                        sample["sample_token"],
                        DS.sample_image_sha256(sample),
                        sample["lidar_points"].numpy().tobytes(),
                    )
                )
                if force_lifecycle_error and epoch_index == 0:
                    raise RuntimeError("forced post-ACK lifecycle error")
            epochs.append(epoch)
            lifecycle_epochs.append(lifecycle)
        assert epochs[0] == epochs[1]
        assert len(lifecycle_epochs[0]) == 2
        assert set(lifecycle_epochs[0]) == set(lifecycle_epochs[1])
        for pid in lifecycle_epochs[0]:
            first = lifecycle_epochs[0][pid]
            second = lifecycle_epochs[1][pid]
            assert second["reopen_count"] == first["reopen_count"]
            assert second["read_count"] > first["read_count"]
            assert set(second["open_archives"]) == set(first["open_archives"])
        assert torch.equal(parent_sample["images"], dataset[0]["images"])
    except BaseException as exc:
        primary = exc
        primary_traceback = exc.__traceback__
    finally:
        iterator = None
        try:
            iterator = getattr(loader, "_iterator", None)
        except BaseException as exc:
            cleanup_errors.append(("iterator_discovery", exc))
        try:
            workers = (
                list(getattr(iterator, "_workers", ()))
                if iterator is not None
                else []
            )
        except BaseException as exc:
            cleanup_errors.append(("worker_discovery", exc))
            workers = []
        for worker in workers:
            try:
                identity = _process_identity(worker.pid)
                if identity is None:
                    raise RuntimeError(
                        f"worker {worker.pid} disappeared before identity capture"
                    )
                worker_identities.append(identity)
            except BaseException as exc:
                cleanup_errors.append((f"worker_identity:{worker.pid}", exc))
        if iterator is not None:
            try:
                iterator._shutdown_workers()
            except BaseException as exc:
                cleanup_errors.append(("shutdown_workers", exc))
        for worker in workers:
            try:
                worker.join(5)
            except BaseException as exc:
                cleanup_errors.append((f"worker_join:{worker.pid}", exc))
        live_workers = []
        for worker in workers:
            try:
                if worker.is_alive():
                    live_workers.append(worker.pid)
            except BaseException as exc:
                cleanup_errors.append((f"worker_liveness:{worker.pid}", exc))
        if live_workers:
            cleanup_errors.append(
                (
                    "worker_liveness",
                    RuntimeError(
                        "persistent DataLoader workers survived explicit shutdown: "
                        f"{live_workers}"
                    ),
                )
            )
        try:
            del loader
        except BaseException as exc:
            cleanup_errors.append(("delete_loader", exc))
        try:
            dataset.close()
        except BaseException as exc:
            cleanup_errors.append(("dataset_close", exc))
        try:
            gc.collect()
        except BaseException as exc:
            cleanup_errors.append(("gc_collect", exc))
        if force_cleanup_evidence:
            cleanup_errors.append(
                (
                    "forced_after_real_cleanup",
                    RuntimeError("forced cleanup evidence after real worker cleanup"),
                )
            )
    if primary is not None:
        try:
            primary._s07_worker_pids = tuple(worker.pid for worker in workers)
            primary._s07_worker_identities = tuple(worker_identities)
            primary._s07_live_worker_pids_after_cleanup = tuple(live_workers)
        except BaseException as exc:
            cleanup_errors.append(("primary_evidence_attachment", exc))
        _add_cleanup_notes(primary, cleanup_errors)
        raise primary.with_traceback(primary_traceback)
    if cleanup_errors:
        raise BaseExceptionGroup(
            "persistent lifecycle cleanup failed",
            [error for _label, error in cleanup_errors],
        )
    return tuple(worker_identities)


def _proc_starttime(pid):
    """Return Linux procfs starttime (field 22), or None once PID is gone."""
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="ascii") as stream:
            stat = stream.read()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return _parse_proc_starttime(stat, pid)


def _parse_proc_starttime(stat, pid):
    # comm is parenthesized and may itself contain spaces or parentheses. The
    # final ')' precedes field 3; starttime is index 19 in the remaining fields.
    close_paren = stat.rfind(")")
    if close_paren < 0:
        raise RuntimeError(f"malformed /proc/{pid}/stat: missing comm terminator")
    remaining = stat[close_paren + 1 :].split()
    if len(remaining) <= 19:
        raise RuntimeError(f"malformed /proc/{pid}/stat: missing starttime")
    return int(remaining[19])


def _process_identity(pid):
    starttime = _proc_starttime(pid)
    return None if starttime is None else (pid, starttime)


def _identity_alive(identity):
    if identity is None:
        return False
    pid, expected_starttime = identity
    return _proc_starttime(pid) == expected_starttime


def _wait_for_cleanup(group_id, identities, timeout):
    deadline = time.monotonic() + timeout
    while True:
        state = {
            "group_alive": (
                _process_group_exists(group_id) if group_id is not None else False
            ),
            "identities_alive": {
                identity: _identity_alive(identity) for identity in identities
            },
        }
        if not state["group_alive"] and not any(state["identities_alive"].values()):
            return state
        if time.monotonic() >= deadline:
            return state
        time.sleep(0.01)


def _close_descendant_connections(*connections):
    for connection in connections:
        try:
            connection.close()
        except BaseException:
            pass


def _fork_descendant(control_connection, result_connection, *, resist_sigterm):
    previous_sigterm = None
    if resist_sigterm:
        # Install before fork so there is no child-startup window in which the
        # parent can observe/report the descendant before SIGTERM resistance.
        previous_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        descendant_pid = os.fork()
    except BaseException:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        raise
    if descendant_pid == 0:
        _close_descendant_connections(control_connection, result_connection)
        signal.signal(
            signal.SIGTERM, signal.SIG_IGN if resist_sigterm else signal.SIG_DFL
        )
        while True:
            signal.pause()
    if previous_sigterm is not None:
        signal.signal(signal.SIGTERM, previous_sigterm)

    deadline = time.monotonic() + 2
    while True:
        descendant_identity = _process_identity(descendant_pid)
        if descendant_identity is not None:
            return descendant_pid, descendant_identity
        if time.monotonic() >= deadline:
            raise RuntimeError("forked descendant identity capture timed out")
        time.sleep(0.01)


def _hang_with_forked_descendant(control_connection, result_connection):
    descendant_pid, descendant_identity = _fork_descendant(
        control_connection, result_connection, resist_sigterm=False
    )

    def _reap_descendant(_signum, _frame):
        try:
            os.waitpid(descendant_pid, 0)
        except ChildProcessError:
            pass

    # A group SIGTERM must reach and terminate the real forked descendant, while
    # the helper deliberately remains alive so parent cleanup must escalate to
    # a verified-group SIGKILL for the session/process-group leader.
    signal.signal(signal.SIGTERM, _reap_descendant)
    result_connection.send(
        {
            "kind": "descendant",
            "descendant_identity": descendant_identity,
            "descendant_pid": descendant_pid,
            "descendant_record": (
                descendant_pid,
                descendant_identity[1],
                os.getpgid(descendant_pid),
                os.getsid(descendant_pid),
            ),
        }
    )
    while True:
        signal.pause()


def _exit_leader_with_resistant_descendant(control_connection, result_connection):
    descendant_pid, descendant_identity = _fork_descendant(
        control_connection, result_connection, resist_sigterm=True
    )
    result_connection.send(
        {
            "kind": "result",
            "status": "leader_exit",
            "worker_identities": (),
            "descendant_pid": descendant_pid,
            "descendant_identity": descendant_identity,
            "descendant_record": (
                descendant_pid,
                descendant_identity[1],
                os.getpgid(descendant_pid),
                os.getsid(descendant_pid),
            ),
            "live_children": (),
        }
    )
    control_connection.close()
    result_connection.close()
    # Deliberately do not waitpid: the process-group leader must be gone while
    # its exact, SIGTERM-resistant descendant instance remains parent-auditable.
    os._exit(0)


def _fresh_spawn_then_fork(
    zip_root, manifest, base_info, control_connection, result_connection, mode
):
    os.setsid()
    pid = os.getpid()
    ready = (
        "ready",
        pid,
        _proc_starttime(pid),
        os.getsid(0),
        os.getpgrp(),
        tuple(
            child.pid
            for child in multiprocessing.active_children()
            if child.is_alive()
        ),
    )
    try:
        # Connection.send is synchronous with the pipe transport. Unlike
        # Queue.put, the helper cannot pass this point and fork until the exact
        # parent ACK arrives over the same duplex control channel.
        control_connection.send(ready)
        acknowledgement = control_connection.recv()
        assert acknowledgement == (
            "ack",
            pid,
            _proc_starttime(pid),
            pid,
            pid,
        )
        assert torch.cuda.is_available() is False
        if mode == "forced_hang":
            _hang_with_forked_descendant(control_connection, result_connection)
        if mode == "post_ack_leader_exit":
            _exit_leader_with_resistant_descendant(
                control_connection, result_connection
            )
        worker_identities = _persistent_lifecycle(
            zip_root,
            manifest,
            base_info,
            "fork",
            force_lifecycle_error=mode == "forced_error",
            force_cleanup_evidence=mode == "forced_error",
        )
        live_children = [
            child.pid for child in multiprocessing.active_children() if child.is_alive()
        ]
        result_connection.send(
            {
                "kind": "result",
                "status": "ok",
                "worker_identities": worker_identities,
                "live_children": live_children,
            }
        )
    except BaseException as exc:
        live_children = [
            child.pid for child in multiprocessing.active_children() if child.is_alive()
        ]
        padding = "R" * (2 * 1024 * 1024) if mode == "forced_error" else ""
        result_connection.send(
            {
                "kind": "result",
                "status": "error",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "notes": tuple(getattr(exc, "__notes__", ())),
                "worker_identities": tuple(
                    getattr(exc, "_s07_worker_identities", ())
                ),
                "live_worker_pids_after_cleanup": tuple(
                    getattr(exc, "_s07_live_worker_pids_after_cleanup", ())
                ),
                "live_children": live_children,
                "padding": padding,
                "padding_length": len(padding),
            }
        )
    finally:
        control_connection.close()
        result_connection.close()


class _ForkHelperFailure(AssertionError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _kill_process_group(group_id, sig):
    try:
        os.killpg(group_id, sig)
    except ProcessLookupError:
        pass


def _process_group_exists(group_id):
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_process_alive(process, cleanup_errors, label):
    try:
        return process.is_alive()
    except BaseException as exc:
        cleanup_errors.append((f"{label}_process_probe", exc))
        return True


def _cleanup_group_exists(group_id, cleanup_errors, label):
    try:
        return _process_group_exists(group_id)
    except BaseException as exc:
        cleanup_errors.append((f"{label}_group_probe", exc))
        return True


def _cleanup_wait_for_targets(group_id, identities, timeout, cleanup_errors, label):
    try:
        return _wait_for_cleanup(group_id, identities, timeout)
    except BaseException as exc:
        cleanup_errors.append((f"{label}_target_wait", exc))
        return {
            "group_alive": True,
            "identities_alive": {identity: True for identity in identities},
        }


def _cleanup_identity_alive(identity, cleanup_errors, label):
    try:
        return _identity_alive(identity)
    except BaseException as exc:
        cleanup_errors.append((f"{label}_identity_probe", exc))
        return True


def _fd_is_closed(fd):
    try:
        os.fstat(fd)
    except OSError as exc:
        return exc.errno == errno.EBADF
    return False


def _validate_ready_record(process, ready, parent_pid, parent_sid, parent_group):
    assert isinstance(ready, tuple) and len(ready) == 6
    (
        tag,
        helper_pid,
        helper_starttime,
        helper_sid,
        helper_group,
        initial_children,
    ) = ready
    assert tag == "ready"
    assert helper_pid == process.pid
    assert helper_pid not in {parent_pid, parent_sid, parent_group}
    assert helper_pid == helper_sid == helper_group
    assert helper_starttime == _proc_starttime(helper_pid)
    assert initial_children == ()
    # Cross-check the child-supplied tuple against live kernel state before the
    # value is eligible to arm killpg or the ACK is sent.
    assert os.getsid(helper_pid) == helper_pid
    assert os.getpgid(helper_pid) == helper_pid
    return helper_pid, (helper_pid, helper_starttime)


def _run_fresh_spawn_fork_helper(
    zip_root,
    manifest,
    base_info,
    *,
    mode="normal",
    ready_timeout=10,
    run_timeout=90,
):
    assert mode in {
        "normal",
        "pre_ack_failure",
        "forced_error",
        "forced_hang",
        "post_ack_leader_exit",
    }
    parent_pid = os.getpid()
    parent_sid = os.getsid(0)
    parent_group = os.getpgrp()
    report = {
        "mode": mode,
        "parent_pid": parent_pid,
        "parent_sid": parent_sid,
        "parent_group": parent_group,
        "ack_sent": False,
        "armed_group": None,
        "descendant_pid": None,
        "descendant_identity": None,
        "descendant_record": None,
        "worker_identities": (),
        "cleanup_signals": [],
        "closed_fds": {},
    }
    ctx = multiprocessing.get_context("spawn")
    parent_control, child_control = ctx.Pipe(duplex=True)
    result_receiver, result_sender = ctx.Pipe(duplex=False)
    process = ctx.Process(
        target=_fresh_spawn_then_fork,
        args=(
            str(zip_root),
            str(manifest),
            base_info,
            child_control,
            result_sender,
            mode,
        ),
    )
    tracked_fds = {
        "parent_control": parent_control.fileno(),
        "child_control": child_control.fileno(),
        "result_receiver": result_receiver.fileno(),
        "result_sender": result_sender.fileno(),
    }
    primary = None
    primary_traceback = None
    cleanup_errors = []
    armed_group = None
    helper_identity = None
    try:
        process.start()
        tracked_fds["process_sentinel"] = process.sentinel
        child_control.close()
        result_sender.close()
        assert parent_control.poll(ready_timeout), "helper ready record timed out"
        ready = parent_control.recv()
        report["ready"] = ready
        verified_group, helper_identity = _validate_ready_record(
            process, ready, parent_pid, parent_sid, parent_group
        )
        # Assignment is deliberately after every tuple and kernel-state check.
        armed_group = verified_group
        report["armed_group"] = armed_group
        if mode == "pre_ack_failure":
            raise _ForkHelperFailure(
                "forced ready-window failure before ACK", report
            )
        report["helper_identity"] = helper_identity
        parent_control.send(
            (
                "ack",
                process.pid,
                helper_identity[1],
                armed_group,
                armed_group,
            )
        )
        report["ack_sent"] = True

        # Drain the complete result payload before joining its producer. This is
        # required for payloads larger than pipe capacity and prevents the classic
        # producer-join/result-recv deadlock.
        result_timeout = max(run_timeout, 5) if mode == "forced_hang" else run_timeout
        assert result_receiver.poll(result_timeout), "helper result record timed out"
        outcome = result_receiver.recv()
        report["outcome"] = outcome
        report["worker_identities"] = tuple(outcome.get("worker_identities", ()))
        report["descendant_pid"] = outcome.get("descendant_pid")
        report["descendant_identity"] = outcome.get("descendant_identity")
        report["descendant_record"] = outcome.get("descendant_record")
        if mode == "forced_hang":
            assert outcome["kind"] == "descendant"
            process.join(run_timeout)
            if process.is_alive():
                raise _ForkHelperFailure(
                    "fresh spawn helper timed out during explicit fork lifecycle",
                    report,
                )
            raise _ForkHelperFailure(
                "forced-hang helper exited unexpectedly", report
            )

        process.join(run_timeout)
        if process.is_alive():
            raise _ForkHelperFailure(
                "fresh spawn helper timed out during explicit fork lifecycle", report
            )
        assert outcome["kind"] == "result"
        if mode == "post_ack_leader_exit":
            assert outcome["status"] == "leader_exit"
            assert process.exitcode == 0
            assert not _identity_alive(helper_identity)
            descendant_identity = outcome["descendant_identity"]
            descendant_pid = outcome["descendant_pid"]
            descendant_record = outcome["descendant_record"]
            assert _identity_alive(descendant_identity)
            assert descendant_identity[0] == descendant_pid
            assert os.getsid(descendant_pid) == armed_group
            assert os.getpgid(descendant_pid) == armed_group
            assert descendant_record == (
                descendant_pid,
                descendant_identity[1],
                armed_group,
                armed_group,
            )
            report["leader_exited_before_cleanup"] = True
            report["group_alive_after_leader_exit"] = _process_group_exists(
                armed_group
            )
            report["descendant_alive_after_leader_exit"] = True
        elif outcome["status"] != "ok":
            failure = _ForkHelperFailure(
                "fresh spawn helper lifecycle failed: "
                f"{outcome['error']}\n{outcome['traceback']}",
                report,
            )
            for note in outcome["notes"]:
                failure.add_note(f"child cleanup evidence: {note}")
            if process.exitcode != 0:
                failure.add_note(
                    f"helper exitcode after reporting primary: {process.exitcode}"
                )
            raise failure
        assert process.exitcode == 0
        if mode == "post_ack_leader_exit":
            assert outcome["worker_identities"] == ()
        else:
            assert outcome["worker_identities"], (
                "fork lifecycle did not report worker identities"
            )
        assert not outcome["live_children"], (
            f"forked descendants survived normal cleanup: {outcome['live_children']}"
        )
    except BaseException as exc:
        primary = exc
        primary_traceback = exc.__traceback__
    finally:
        if not child_control.closed:
            try:
                child_control.close()
            except BaseException as exc:
                cleanup_errors.append(("child_control_close", exc))
        if not result_sender.closed:
            try:
                result_sender.close()
            except BaseException as exc:
                cleanup_errors.append(("result_sender_close", exc))

        if process.pid is not None:
            group_needs_cleanup = (
                armed_group is not None
                and _cleanup_group_exists(
                    armed_group, cleanup_errors, "initial_cleanup"
                )
            )
            helper_alive = _cleanup_process_alive(
                process, cleanup_errors, "initial_cleanup"
            )
            if armed_group is not None and (helper_alive or group_needs_cleanup):
                group_safe = (
                    armed_group == process.pid
                    and armed_group not in {parent_pid, parent_sid, parent_group}
                )
                if not group_safe:
                    cleanup_errors.append(
                        (
                            "verified_group_identity",
                            RuntimeError("armed helper group lost its safety invariant"),
                        )
                    )
                else:
                    try:
                        _kill_process_group(armed_group, signal.SIGTERM)
                        report["cleanup_signals"].append("SIGTERM")
                    except BaseException as exc:
                        cleanup_errors.append(("verified_group_sigterm", exc))
                tracked_identities = tuple(
                    identity
                    for identity in (
                        helper_identity,
                        report.get("descendant_identity"),
                        *report["worker_identities"],
                    )
                    if identity is not None
                )
                try:
                    # Reap a helper that honored TERM before polling procfs/group
                    # state; otherwise its zombie PID would force a false KILL.
                    process.join(1)
                except BaseException as exc:
                    cleanup_errors.append(("verified_group_term_reap", exc))
                term_state = _cleanup_wait_for_targets(
                    armed_group,
                    tracked_identities,
                    1,
                    cleanup_errors,
                    "post_sigterm",
                )
                report["post_sigterm_state"] = term_state
                if group_safe and (
                    term_state["group_alive"]
                    or any(term_state["identities_alive"].values())
                ):
                    try:
                        _kill_process_group(armed_group, signal.SIGKILL)
                        report["cleanup_signals"].append("SIGKILL")
                    except BaseException as exc:
                        cleanup_errors.append(("verified_group_sigkill", exc))
                    try:
                        # Reap a killed helper leader before auditing whether its
                        # process group and exact instances disappeared.
                        process.join(1)
                    except BaseException as exc:
                        cleanup_errors.append(("verified_group_kill_reap", exc))
                    kill_state = _cleanup_wait_for_targets(
                        armed_group,
                        tracked_identities,
                        5,
                        cleanup_errors,
                        "post_sigkill",
                    )
                    report["post_sigkill_state"] = kill_state
                try:
                    process.join(0)
                except BaseException as exc:
                    cleanup_errors.append(("verified_group_final_join", exc))
                if _cleanup_process_alive(process, cleanup_errors, "post_sigkill"):
                    try:
                        process.kill()
                        process.join(5)
                    except BaseException as exc:
                        cleanup_errors.append(("verified_helper_direct_kill", exc))
            elif armed_group is None and helper_alive:
                try:
                    process.terminate()
                    process.join(5)
                except BaseException as exc:
                    cleanup_errors.append(("unarmed_helper_termination", exc))
                if _cleanup_process_alive(process, cleanup_errors, "unarmed_post_term"):
                    try:
                        process.kill()
                        process.join(5)
                    except BaseException as exc:
                        cleanup_errors.append(("unarmed_helper_direct_kill", exc))

        if process.pid is not None and _cleanup_process_alive(
            process, cleanup_errors, "final"
        ):
            cleanup_errors.append(
                ("helper_liveness", RuntimeError("fresh spawn helper survived cleanup"))
            )
        if armed_group is not None:
            report["group_alive_after_cleanup"] = _cleanup_group_exists(
                armed_group, cleanup_errors, "final"
            )
            if report["group_alive_after_cleanup"]:
                cleanup_errors.append(
                    (
                        "group_liveness",
                        RuntimeError(
                            f"verified helper process group {armed_group} survived cleanup"
                        ),
                    )
                )

        descendant_identity = report["descendant_identity"]
        report["descendant_alive_after_cleanup"] = _cleanup_identity_alive(
            descendant_identity, cleanup_errors, "descendant_final"
        )
        if report["descendant_alive_after_cleanup"]:
            cleanup_errors.append(
                (
                    "descendant_liveness",
                    RuntimeError(
                        f"forked descendant {descendant_identity} survived cleanup"
                    ),
                )
            )

        report["worker_alive_after_cleanup"] = {
            identity: _cleanup_identity_alive(
                identity, cleanup_errors, f"worker_final:{identity[0]}"
            )
            for identity in report["worker_identities"]
        }
        if any(report["worker_alive_after_cleanup"].values()):
            cleanup_errors.append(
                (
                    "worker_identity_liveness",
                    RuntimeError("forked DataLoader worker instance survived cleanup"),
                )
            )

        for label, resource in (
            ("parent_control", parent_control),
            ("result_receiver", result_receiver),
        ):
            try:
                resource.close()
            except BaseException as exc:
                cleanup_errors.append((f"{label}_close", exc))
        process_alive = process.pid is not None and _cleanup_process_alive(
            process, cleanup_errors, "pre_close"
        )
        if not process_alive:
            try:
                process.close()
                report["process_closed"] = True
            except BaseException as exc:
                cleanup_errors.append(("process_close", exc))
        else:
            report["process_closed"] = False

        for label, fd in tracked_fds.items():
            report["closed_fds"][label] = _fd_is_closed(fd)
            if not report["closed_fds"][label]:
                cleanup_errors.append(
                    ("fd_close", RuntimeError(f"{label} fd {fd} remained open"))
                )

        try:
            report["parent_identity_after_cleanup"] = (
                os.getpid(),
                os.getsid(0),
                os.getpgrp(),
            )
        except BaseException as exc:
            cleanup_errors.append(("parent_identity_probe", exc))
            report["parent_identity_after_cleanup"] = None
        if report["parent_identity_after_cleanup"] != (
            parent_pid,
            parent_sid,
            parent_group,
        ):
            cleanup_errors.append(
                ("parent_identity", RuntimeError("pytest parent session/group changed"))
            )

    if primary is not None:
        if isinstance(primary, _ForkHelperFailure):
            primary.report = report
        _add_cleanup_notes(primary, cleanup_errors)
        raise primary.with_traceback(primary_traceback)
    if cleanup_errors:
        raise BaseExceptionGroup(
            "fresh spawn helper cleanup failed",
            [error for _label, error in cleanup_errors],
        )
    return report


@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_repeated_persistent_multiworker_reads_are_deterministic(
    directory_and_zip, start_method, monkeypatch
):
    assert start_method in multiprocessing.get_all_start_methods()
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    if start_method == "spawn":
        _persistent_lifecycle(str(zip_root), str(manifest), base_info, "spawn")
        return
    # Explicit fork is lifecycle evidence only. Enter it from a fresh spawned,
    # CUDA-hidden interpreter so no CUDA/threaded pytest state is inherited.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = _run_fresh_spawn_fork_helper(zip_root, manifest, base_info)
    assert report["outcome"]["status"] == "ok"
    assert report["process_closed"] is True
    assert report["group_alive_after_cleanup"] is False
    assert all(report["closed_fds"].values())


def test_explicit_fork_pre_ack_failure_never_forks_or_touches_parent_group(
    directory_and_zip, monkeypatch
):
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    parent_identity = (os.getpid(), os.getsid(0), os.getpgrp())
    with pytest.raises(_ForkHelperFailure, match="ready-window") as caught:
        _run_fresh_spawn_fork_helper(
            zip_root,
            manifest,
            base_info,
            mode="pre_ack_failure",
            ready_timeout=5,
            run_timeout=1,
        )
    report = caught.value.report
    assert report["ready"][5] == ()
    assert report["ack_sent"] is False
    assert report["descendant_pid"] is None
    assert report["armed_group"] == report["ready"][1]
    assert report["armed_group"] not in parent_identity
    assert report["group_alive_after_cleanup"] is False
    assert report["parent_identity_after_cleanup"] == parent_identity
    assert report["process_closed"] is True
    assert all(report["closed_fds"].values())


def test_explicit_fork_post_ack_error_preserves_primary_and_cleanup_evidence(
    directory_and_zip, monkeypatch
):
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(_ForkHelperFailure, match="forced post-ACK lifecycle error") as caught:
        _run_fresh_spawn_fork_helper(
            zip_root,
            manifest,
            base_info,
            mode="forced_error",
            ready_timeout=5,
            run_timeout=20,
        )
    report = caught.value.report
    outcome = report["outcome"]
    assert report["ack_sent"] is True
    assert "forced post-ACK lifecycle error" in outcome["error"]
    assert "raise RuntimeError" in outcome["traceback"]
    assert outcome["padding_length"] >= 1024 * 1024
    assert len(outcome["padding"]) == outcome["padding_length"]
    assert set(outcome["padding"]) == {"R"}
    assert any("forced cleanup evidence" in note for note in outcome["notes"])
    assert any("child cleanup evidence" in note for note in caught.value.__notes__)
    assert outcome["worker_identities"]
    assert outcome["live_worker_pids_after_cleanup"] == ()
    assert set(report["worker_alive_after_cleanup"]) == set(
        outcome["worker_identities"]
    )
    assert not any(report["worker_alive_after_cleanup"].values())
    assert outcome["live_children"] == []
    assert report["group_alive_after_cleanup"] is False
    assert report["process_closed"] is True
    assert all(report["closed_fds"].values())


def test_explicit_fork_post_ack_leader_exit_cleans_verified_orphan_group(
    directory_and_zip, monkeypatch
):
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = _run_fresh_spawn_fork_helper(
        zip_root,
        manifest,
        base_info,
        mode="post_ack_leader_exit",
        ready_timeout=5,
        run_timeout=5,
    )
    assert report["ack_sent"] is True
    assert report["leader_exited_before_cleanup"] is True
    assert report["group_alive_after_leader_exit"] is True
    assert report["descendant_alive_after_leader_exit"] is True
    assert report["descendant_identity"][0] == report["descendant_pid"]
    assert report["descendant_record"] == (
        report["descendant_pid"],
        report["descendant_identity"][1],
        report["armed_group"],
        report["armed_group"],
    )
    assert report["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
    assert report["post_sigterm_state"]["group_alive"] is True
    assert report["post_sigterm_state"]["identities_alive"][
        report["descendant_identity"]
    ] is True
    assert report["post_sigkill_state"]["group_alive"] is False
    assert report["post_sigkill_state"]["identities_alive"][
        report["descendant_identity"]
    ] is False
    assert report["descendant_alive_after_cleanup"] is False
    assert report["group_alive_after_cleanup"] is False
    assert report["process_closed"] is True
    assert all(report["closed_fds"].values())


def test_proc_starttime_parser_uses_final_comm_parenthesis():
    fields_3_through_21 = ["S", *[str(index) for index in range(4, 22)]]
    stat = "123 (hostile ) comm ( value) " + " ".join(
        [*fields_3_through_21, "987654", "0", "0"]
    )
    assert _parse_proc_starttime(stat, 123) == 987654


def test_explicit_fork_post_ack_hang_kills_verified_group_and_descendant(
    directory_and_zip, monkeypatch
):
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(_ForkHelperFailure, match="timed out") as caught:
        _run_fresh_spawn_fork_helper(
            zip_root,
            manifest,
            base_info,
            mode="forced_hang",
            ready_timeout=5,
            run_timeout=0.25,
        )
    report = caught.value.report
    assert report["ack_sent"] is True
    assert report["descendant_pid"] is not None
    assert report["descendant_record"] == (
        report["descendant_pid"],
        report["descendant_identity"][1],
        report["armed_group"],
        report["armed_group"],
    )
    assert report["cleanup_signals"] == ["SIGTERM", "SIGKILL"]
    assert report["post_sigterm_state"]["group_alive"] is True
    assert report["post_sigterm_state"]["identities_alive"][
        report["helper_identity"]
    ] is True
    assert report["post_sigkill_state"]["group_alive"] is False
    assert report["post_sigkill_state"]["identities_alive"][
        report["helper_identity"]
    ] is False
    assert report["descendant_alive_after_cleanup"] is False
    assert report["group_alive_after_cleanup"] is False
    assert report["process_closed"] is True
    assert all(report["closed_fds"].values())


def test_default_loader_context_is_spawn(directory_and_zip):
    _directory_root, zip_root, manifest, base_info, _paths = directory_and_zip
    dataset = DS.NuScenesMultimodalDataset(
        [base_info], str(zip_root), n_sweeps=10, zip_manifest=str(manifest)
    )
    loader = DS.make_loader(dataset, num_workers=1)
    assert loader.multiprocessing_context.get_start_method() == "spawn"
    del loader
    dataset.close()


def test_zip_root_without_manifest_fails_before_any_archive_scan(tmp_path):
    root = tmp_path / "zip"
    root.mkdir()
    (root / TRAINVAL_ARCHIVE_NAMES[0]).write_bytes(b"not-even-opened")
    with pytest.raises(FileNotFoundError, match="no external member manifest"):
        NuScenesBlobStore(str(root))


def _mode_zip(tmp_path, directory_root, members):
    root = tmp_path / "mode_zip"
    root.mkdir()
    buckets = {name: [] for name in TRAINVAL_ARCHIVE_NAMES}
    for index, rel in enumerate(members):
        buckets[TRAINVAL_ARCHIVE_NAMES[index % len(TRAINVAL_ARCHIVE_NAMES)]].append(rel)
    for name, paths in buckets.items():
        with zipfile.ZipFile(root / name, "w", compression=zipfile.ZIP_STORED) as archive:
            for rel in paths:
                archive.writestr(rel, (directory_root / rel).read_bytes())
    manifest = tmp_path / "mode_manifest.sqlite"
    build_zip_manifest(str(root), str(manifest))
    return root, manifest


@pytest.mark.parametrize("backend", ["directory", "zip"])
@pytest.mark.parametrize("model_mode", ["camera_only", "lidar_only"])
def test_mode_aware_io_never_reads_missing_disabled_payload(
    directory_and_zip, tmp_path, backend, model_mode
):
    directory_root, _zip_root, _manifest, info, _paths = directory_and_zip
    enabled = (
        list(info["cam_rel_paths"])
        if model_mode == "camera_only"
        else [info["lidar_rel_path"], *(s["rel_path"] for s in info["lidar_sweeps"])]
    )
    if backend == "zip":
        root, manifest = _mode_zip(tmp_path, directory_root, enabled)
        kwargs = {"zip_manifest": str(manifest)}
    else:
        root = tmp_path / "mode_directory"
        for rel in enabled:
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((directory_root / rel).read_bytes())
        kwargs = {}

    dataset = DS.NuScenesMultimodalDataset(
        [info], str(root), n_sweeps=10, model_mode=model_mode, **kwargs
    )
    sample = dataset[0]
    counts = dataset.payload_read_counts["reads"]
    assert sample["model_mode"] == model_mode
    assert "images" in sample is (model_mode == "camera_only")
    assert "lidar_points" in sample is (model_mode == "lidar_only")
    assert counts["camera"] == (6 if model_mode == "camera_only" else 0)
    assert counts["lidar"] == (10 if model_mode == "lidar_only" else 0)
    assert sample["cam_intrinsics"].shape == (6, 3, 3)
    assert sample["lidar2ego"].shape == (4, 4)
    assert sample["gt_boxes"].shape == (0, 7)
    dataset.close()
