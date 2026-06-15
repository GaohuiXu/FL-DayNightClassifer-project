"""Viz writer: lays down the V1–V6 tree and a deterministic manifest."""
from __future__ import annotations

import json
import os

from fl_v3.viz.writer import VIZ_STAGES, VizWriter


def test_viz_tree_created(tmp_path):
    w = VizWriter(str(tmp_path / "run"))
    for stage in VIZ_STAGES:
        assert os.path.isdir(os.path.join(str(tmp_path / "run"), "viz", stage))


def test_write_json_and_manifest_are_deterministic(tmp_path):
    def build(run_dir):
        w = VizWriter(run_dir)
        # Register out of order; manifest must sort.
        w.write_json("fusion", "b_artifact", {"z": 1, "a": 2})
        w.write_json("calibration", "a_artifact", {"k": [3, 2, 1]})
        w.figure_path("attack", "trigger_overlay")  # reserve a figure path
        return w.write_manifest()

    m1 = build(str(tmp_path / "r1"))
    m2 = build(str(tmp_path / "r2"))
    with open(m1) as f:
        man1 = f.read()
    with open(m2) as f:
        man2 = f.read()
    assert man1 == man2  # byte-identical manifests across runs

    data = json.loads(man1)
    assert set(data["stages"].keys()) == set(VIZ_STAGES)
    assert data["stages"]["calibration"]["viz_id"] == "V1"
    assert data["stages"]["defense"]["viz_id"] == "V6"
    # Registered artifacts are recorded (sorted).
    assert "viz/fusion/b_artifact.json" in data["stages"]["fusion"]["artifacts"]
    assert "viz/attack/trigger_overlay.png" in data["stages"]["attack"]["artifacts"]


def test_unknown_stage_rejected(tmp_path):
    w = VizWriter(str(tmp_path / "run"))
    try:
        w.write_json("not_a_stage", "x", {})
        assert False, "expected ValueError"
    except ValueError:
        pass
