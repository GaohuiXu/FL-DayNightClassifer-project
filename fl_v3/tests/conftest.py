"""Shared pytest fixtures for nuScenes devkit/data tests."""
from __future__ import annotations

import pytest

from fl_v3.data.nuscenes import paths as _P


# ---------------------------------------------------------------------------
# nuScenes devkit + data fixtures (T1). Skip cleanly if the read-only dataset
# or the devkit is unavailable (e.g. a CI box without the mounted dataset).
# ---------------------------------------------------------------------------
def _dataset_available(version: str) -> bool:
    try:
        _P.verify_dataset(version)
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def dataroot() -> str:
    return _P.DATAROOT


@pytest.fixture(scope="session")
def nusc_mini():
    if not _dataset_available("v1.0-mini"):
        pytest.skip("v1.0-mini not available at DATAROOT")
    from nuscenes import NuScenes

    return NuScenes(version="v1.0-mini", dataroot=_P.DATAROOT, verbose=False)


@pytest.fixture(scope="session")
def nusc_trainval():
    """Heavy (~46 s, ~2.5 GB metadata). Used only by the trainval-log-table tests."""
    if not _dataset_available("v1.0-trainval"):
        pytest.skip("v1.0-trainval not available at DATAROOT")
    from nuscenes import NuScenes

    return NuScenes(version="v1.0-trainval", dataroot=_P.DATAROOT, verbose=False)


@pytest.fixture(scope="session")
def mini_train_info(nusc_mini):
    """Full mini_train info-list (8 scenes) — the canonical schema for fast tests."""
    from fl_v3.data.nuscenes import info_cache as IC

    tokens = IC.split_sample_tokens(nusc_mini, "mini_train")
    return IC.build_info_list(nusc_mini, tokens, _P.DATAROOT)


@pytest.fixture(scope="session")
def mini_val_info(nusc_mini):
    from fl_v3.data.nuscenes import info_cache as IC

    tokens = IC.split_sample_tokens(nusc_mini, "mini_val")
    return IC.build_info_list(nusc_mini, tokens, _P.DATAROOT)


@pytest.fixture(scope="session")
def mini_cache_dir(nusc_mini):
    """Ensure the mini info-cache exists at the CWD-relative ``./fl_outputs/...`` path the
    detection-task config defaults to (T2). Builds it from the devkit if absent — this is
    TEST infrastructure (allowed to use the devkit), not ``client_data`` (which must raise
    if the cache is missing). Skips cleanly if the dataset/devkit is unavailable (via the
    ``nusc_mini`` dependency). Makes the T2 task tests reproducible from ANY pytest CWD."""
    from fl_v3.data.nuscenes import info_cache as IC

    cache_dir = "./fl_outputs/nuscenes/info_cache"
    for split in ("mini_train", "mini_val"):
        IC.get_or_build_cache(nusc_mini, cache_dir, "v1.0-mini", split, "mini-smoke", _P.DATAROOT)
    return cache_dir
