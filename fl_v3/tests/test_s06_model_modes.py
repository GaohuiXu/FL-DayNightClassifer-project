from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest
import torch

from fl_v3.models.fusion.detector import DetectorConfig
from fl_v3.training.runtime_state import project_batch_for_mode
from fl_v3.utils.runtime import normalize_model_mode
from fl_v3.utils.runtime import require_spconv_238


@pytest.mark.parametrize("mode", ["camera_only", "lidar_only", "fusion"])
def test_exact_modes(mode):
    assert normalize_model_mode(mode) == mode


@pytest.mark.parametrize("bad", ["camera", "lidar", "both", "", None, "FUSION"])
def test_legacy_and_unknown_modes_fail(bad):
    with pytest.raises(ValueError):
        normalize_model_mode(bad)


def test_disabled_modality_is_removed_before_device_transfer():
    batch = {
        "images": torch.ones(1), "lidar2img": torch.ones(1), "cam_intrinsics": torch.ones(1),
        "lidar_points": torch.ones(1), "gt_boxes": [], "sample_token": ["x"],
    }
    camera = project_batch_for_mode(batch, "camera_only")
    lidar = project_batch_for_mode(batch, "lidar_only")
    assert "lidar_points" not in camera
    assert not ({"images", "lidar2img", "cam_intrinsics"} & set(lidar))
    assert set(project_batch_for_mode(batch, "fusion")) == set(batch)


def test_detector_config_default_is_exact_fusion_name():
    assert DetectorConfig().model_mode == "fusion"


def test_reviewed_spconv_dependency_is_exact_in_arrhenius_runtime():
    require_spconv_238()


class _Pre(torch.nn.Module):
    def __init__(self, **kw): super().__init__()
    def forward(self, images, lidar2img, cam_intrinsics):
        return {"images": images, "lidar2img": lidar2img}


class _Camera(torch.nn.Module):
    out_channels = (1,); strides = (1,)
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return [x[:, :1]]


class _CameraNeck(torch.nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, feats): return feats[0]


class _VT(torch.nn.Module):
    def __init__(self, context_channels, **kw): super().__init__(); self.c=context_channels
    def forward(self, feat, lidar2img, B, N):
        bev=torch.ones(B,self.c,4,4); return {"bev":bev,"depth_prob":bev,"context":bev}


class _Lidar(torch.nn.Module):
    def __init__(self, out_channels, **kw): super().__init__(); self.c=out_channels
    def forward(self, points, B): return torch.ones(B,self.c,4,4)


class _Fuser(torch.nn.Module):
    def __init__(self, out_channels, **kw): super().__init__(); self.c=out_channels
    def forward(self, camera, lidar): return torch.ones(camera.shape[0],self.c,4,4)


class _BevNeck(torch.nn.Module):
    out_channels=4
    def __init__(self, out_channels, **kw): super().__init__(); self.out_channels=out_channels
    def forward(self, x): return x


class _Head(torch.nn.Module):
    def __init__(self, **kw): super().__init__()
    def forward(self, x): return {"heatmap": x}


@pytest.mark.parametrize("mode", ["camera_only", "lidar_only", "fusion"])
def test_mode_constructs_and_executes_only_enabled_branches(monkeypatch, mode):
    import fl_v3.models.fusion.detector as d
    monkeypatch.setattr(d, "ImagePreprocessor", _Pre)
    monkeypatch.setattr(d, "CameraBackbone", _Camera)
    monkeypatch.setattr(d, "GeneralizedLSSFPN", _CameraNeck)
    monkeypatch.setattr(d, "DepthLSSTransform", _VT)
    monkeypatch.setattr(d, "PointPillarsEncoder", _Lidar)
    monkeypatch.setattr(d, "ConvFuser", _Fuser)
    monkeypatch.setattr(d, "SecondFPNNeck", _BevNeck)
    monkeypatch.setattr(d, "CenterPointHead", _Head)
    monkeypatch.setattr(d, "require_spconv_238", lambda: None)
    model = d.BEVFusionDetector(DetectorConfig(
        model_mode=mode, required_spconv_version=(None if mode == "camera_only" else "2.3.8"),
        pretrained_backbone=False, context_channels=2, lidar_channels=3,
        fusion_channels=4, bev_neck_channels=4,
    ))
    assert (model.camera_backbone is not None) == (mode != "lidar_only")
    assert (model.lidar_encoder is not None) == (mode != "camera_only")
    assert (model.fusion is not None) == (mode == "fusion")
    batch = {"batch_size": 1, "images": torch.ones(1,1,3,2,2),
             "lidar2img": torch.eye(4).reshape(1,1,4,4),
             "cam_intrinsics": torch.eye(3).reshape(1,1,3,3),
             "lidar_points": torch.ones(1,6)}
    projected = project_batch_for_mode(batch, mode)
    assert model(projected)["heatmap"].shape == (1,4,4,4)
    cloned = copy.deepcopy(model)
    assert cloned._runtime_lock is not model._runtime_lock


def test_same_detector_instance_serializes_sparse_forward_and_mode_change(monkeypatch):
    import fl_v3.models.fusion.detector as d
    active = 0; maximum = 0; guard = threading.Lock()

    class SlowLidar(_Lidar):
        def forward(self, points, B):
            nonlocal active, maximum
            with guard: active += 1; maximum = max(maximum, active)
            time.sleep(0.02)
            with guard: active -= 1
            return super().forward(points, B)

    monkeypatch.setattr(d, "PointPillarsEncoder", SlowLidar)
    monkeypatch.setattr(d, "SecondFPNNeck", _BevNeck)
    monkeypatch.setattr(d, "CenterPointHead", _Head)
    monkeypatch.setattr(d, "require_spconv_238", lambda: None)
    model = d.BEVFusionDetector(DetectorConfig(
        model_mode="lidar_only", required_spconv_version="2.3.8",
        lidar_channels=3, fusion_channels=4, bev_neck_channels=4,
    ))
    batch = {"batch_size": 1, "lidar_points": torch.ones(1, 6)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(model, batch), pool.submit(model, batch)]
        [future.result() for future in futures]
    assert maximum == 1
    model.eval(); assert model.training is False
