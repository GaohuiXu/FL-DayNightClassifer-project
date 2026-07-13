"""Guard: every key an FL config sets MUST be registered in pyproject [tool.flwr.app.config].

``flwr run --run-config`` hard-rejects any key not present in the pyproject config dictionary ("Key 'X' is
not present in the main dictionary") AND rejects array values (scalars / str / dict only). The bb02d FL
config (fl_bb02d_fedadam.json) adds the whole MCR capability + recipe surface; this test catches an
unregistered or array-typed key at unit-test time instead of at SLURM-submit time. Also covers
``_normalize_weights`` (the str|list→None uniform mapping that keeps the default loss byte-identical).
"""
from __future__ import annotations

import json
import os

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from fl_v3.training.tasks import _normalize_weights

_HERE = os.path.dirname(__file__)
_PYPROJECT = os.path.join(_HERE, "..", "pyproject.toml")
_FL_CONFIGS = ["fl_bb02d_fedadam.json"]


def _declared_keys() -> set:
    with open(_PYPROJECT, "rb") as f:
        data = tomllib.load(f)
    return set(data["tool"]["flwr"]["app"]["config"].keys())


def test_fl_config_keys_are_registered_in_pyproject():
    declared = _declared_keys()
    for cfg_name in _FL_CONFIGS:
        cfg = json.load(open(os.path.join(_HERE, "..", "configs", cfg_name)))
        missing = sorted(k for k in cfg if k not in declared)
        assert not missing, (
            f"{cfg_name}: keys not registered in pyproject [tool.flwr.app.config] "
            f"(flwr run would reject them): {missing}")


def test_no_array_valued_keys_in_fl_config():
    # flwr 1.27 rejects array config values; the class-weights must be a comma-string, not a JSON list.
    for cfg_name in _FL_CONFIGS:
        cfg = json.load(open(os.path.join(_HERE, "..", "configs", cfg_name)))
        arrays = sorted(k for k, v in cfg.items() if isinstance(v, list))
        assert not arrays, f"{cfg_name}: flwr rejects array values; encode as a comma-string: {arrays}"


def test_normalize_weights_str_list_uniform():
    assert _normalize_weights(None) is None
    assert _normalize_weights("") is None
    assert _normalize_weights([]) is None
    assert _normalize_weights("1,1,1,1") is None              # uniform string ⇒ None (byte-identical default)
    assert _normalize_weights([1.0, 1.0, 1.0]) is None        # uniform list ⇒ None
    assert _normalize_weights("1.0,1.3,2.5") == [1.0, 1.3, 2.5]
    assert _normalize_weights([1.0, 1.3, 2.5]) == [1.0, 1.3, 2.5]


def test_fl_bb02d_is_clean_full_participation_trainval():
    """The bb02d FL config remains a clean, scene-aware trainval control."""
    cfg = json.load(open(os.path.join(_HERE, "..", "configs", "fl_bb02d_fedadam.json")))
    assert float(cfg["fraction-train"]) == 1.0
    assert cfg["nuscenes-partition-mode"] == "log_group"
    assert cfg["nuscenes-version"] == "v1.0-trainval"
    assert cfg["nuscenes-train-split"] == "train" and cfg["nuscenes-val-split"] == "val"
    assert cfg["server-optimizer"] == "fedadam"
