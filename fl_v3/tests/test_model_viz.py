"""T2 V2/V3 clean renderers smoke — figures + manifest written deterministically."""
from __future__ import annotations

import os

import pytest
import torch

from fl_v3.utils.runtime import enforce_determinism, seed_everything


def test_v2_v3_render(tmp_path, mini_cache_dir, monkeypatch):
    from fl_v3.training.tasks import get_task
    from fl_v3.training.loop import _move_to_device
    from fl_v3.viz.writer import VizWriter
    from fl_v3.viz import encoder as V2
    from fl_v3.viz import fusion as V3
    from fl_v3.models.fusion.bev_grid import BEVConfig

    monkeypatch.chdir(tmp_path)
    enforce_determinism(strict=True); seed_everything(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        "nuscenes-cache-dir": str(mini_cache_dir), "nuscenes-version": "v1.0-mini",
        "nuscenes-train-split": "mini_train", "nuscenes-val-split": "mini_val",
        "nuscenes-partition-mode": "iid", "nuscenes-num-clients": 27, "seed": 0,
        "batch-size": 3, "num-workers": 0, "det-camera-backbone": "resnet18",
        "det-pretrained-backbone": False, "det-neck-channels": 32, "det-context-channels": 16,
        "det-lidar-channels": 16, "det-fusion-channels": 32, "det-bev-neck-channels": 64,
        "det-head-channels": 16,
    }
    task = get_task("nuscenes_detection")
    model = task.build_model(cfg).to(dev)
    batch = _move_to_device(next(iter(task.client_data(0, cfg).trainloader)), dev)
    assert cfg["nuscenes-cache-dir"] == str(mini_cache_dir)
    assert not (tmp_path / "fl_outputs").exists()

    w = VizWriter(str(tmp_path))
    p2 = V2.render_v2(w, model, batch, BEVConfig(), max_samples=3)
    p3 = V3.render_v3(w, model, batch, BEVConfig(), max_samples=3)
    manifest = w.write_manifest()

    assert len(p2) >= 6 and len(p3) >= 3  # ≥3 samples each (2 V2 panels per sample)
    for p in p2 + p3:
        assert os.path.getsize(p) > 1000, f"empty figure {p}"
    assert os.path.exists(manifest)
    assert not (tmp_path / "fl_outputs").exists()
