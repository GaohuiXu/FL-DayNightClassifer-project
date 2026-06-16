"""T1 — detection taxonomy parity vs the devkit (class_map.py)."""
from __future__ import annotations

from fl_v3.data.nuscenes import class_map as CM


def test_detection_names_id_order():
    from nuscenes.eval.detection.constants import DETECTION_NAMES

    assert CM.DETECTION_NAMES == list(DETECTION_NAMES)
    assert CM.PRIMARY_TARGET_NAME == "car" and CM.PRIMARY_TARGET_ID == 0  # D8


def test_mapping_identical_to_devkit_on_all_categories(nusc_mini):
    from nuscenes.eval.detection.utils import category_to_detection_name as dk

    for c in nusc_mini.category:
        name = c["name"]
        assert CM.category_to_detection_name(name) == dk(name), name


def test_dropped_categories_are_none():
    # Spot-check the None/dropped set (must be dropped, not mis-mapped).
    for dropped in (
        "animal",
        "human.pedestrian.stroller",
        "human.pedestrian.wheelchair",
        "human.pedestrian.personal_mobility",
        "movable_object.debris",
        "movable_object.pushable_pullable",
        "static_object.bicycle_rack",
        "vehicle.emergency.ambulance",
        "vehicle.emergency.police",
    ):
        assert CM.category_to_detection_name(dropped) is None, dropped


def test_class_range_matches_devkit_config():
    from nuscenes.eval.common.config import config_factory

    cfg = config_factory("detection_cvpr_2019")
    assert CM.CLASS_RANGE == {k: float(v) for k, v in cfg.class_range.items()}
    assert CM.CLASS_RANGE["car"] == 50.0 and CM.CLASS_RANGE["traffic_cone"] == 30.0


def test_barrier_cone_empty_attribute_tolerated(nusc_mini, dataroot):
    """A sample containing static classes (barrier/cone) loads with '' attribute and
    does not crash."""
    from fl_v3.data.nuscenes import info_cache as IC

    found = False
    for s in nusc_mini.sample:
        info = IC.build_keyframe_info(nusc_mini, s, dataroot)
        for name, attr in zip(info["gt_names"], info["gt_attribute"]):
            if name in ("barrier", "traffic_cone"):
                found = True
                assert isinstance(attr, str)  # may be "" — tolerated
        if found:
            break
    assert found, "no barrier/cone sample found in mini"
