"""nuScenes 10-class detection taxonomy + eval ranges (fl_v3 T1).

Thin, deterministic wrappers over the devkit's authoritative definitions so the
whole platform keys off the SAME taxonomy ``DetectionEval`` uses (T4):

  * ``DETECTION_NAMES`` order == class id (car == 0, the D8 primary target).
  * ``category_to_detection_name`` == the devkit fn (``None`` categories dropped).
  * ``class_range`` == ``config_factory("detection_cvpr_2019").class_range``.

We re-expose (not re-derive) these to avoid a hand table drifting from the devkit —
the unit test asserts our mapping is identical to ``category_to_detection_name`` on
**all** categories in the version (including the ``None``/dropped set), and that our
``class_range`` equals the devkit config's. Keeping the devkit as the single source
means a devkit version bump cannot silently desync our ids.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from nuscenes.eval.detection.constants import DETECTION_NAMES as _DEVKIT_DETECTION_NAMES
from nuscenes.eval.detection.utils import category_to_detection_name as _devkit_cat2name
from nuscenes.eval.common.config import config_factory as _config_factory

# Frozen at import from the devkit (id order == DETECTION_NAMES). car == 0 (D8).
DETECTION_NAMES: List[str] = list(_DEVKIT_DETECTION_NAMES)
NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(DETECTION_NAMES)}
NUM_CLASSES: int = len(DETECTION_NAMES)

# D8 primary target class — keyed by the DETECTION NAME, never raw category.
PRIMARY_TARGET_NAME: str = "car"
PRIMARY_TARGET_ID: int = NAME_TO_ID[PRIMARY_TARGET_NAME]  # == 0

# Official per-class radial eval ranges (metres). Frozen from the devkit config so
# T4's eligibility (criterion 4) uses the exact thresholds DetectionEval uses.
_DETECTION_CONFIG = _config_factory("detection_cvpr_2019")
CLASS_RANGE: Dict[str, float] = {
    name: float(_DETECTION_CONFIG.class_range[name]) for name in DETECTION_NAMES
}

assert PRIMARY_TARGET_ID == 0, "D8: car must be detection id 0"


def category_to_detection_name(category_name: str) -> Optional[str]:
    """Map a raw nuScenes category to a detection name, or ``None`` if dropped.

    Delegates to the exact devkit fn ``DetectionEval`` uses — categories like
    ``animal``, ``human.pedestrian.{stroller,wheelchair,personal_mobility}``,
    ``movable_object.{debris,pushable_pullable}``, ``static_object.bicycle_rack``,
    and ``vehicle.emergency.*`` return ``None`` (dropped, not mis-mapped).
    """
    return _devkit_cat2name(category_name)


def category_to_detection_id(category_name: str) -> Optional[int]:
    """Map a raw category to a detection id 0..9, or ``None`` if dropped."""
    name = category_to_detection_name(category_name)
    return None if name is None else NAME_TO_ID[name]


def class_range_for(detection_name: str) -> float:
    """Official eval radial range (m) for a detection name."""
    return CLASS_RANGE[detection_name]
