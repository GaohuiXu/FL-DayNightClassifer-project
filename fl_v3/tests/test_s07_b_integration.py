"""S07-B seams across reviewed S02-S06 components.

These tests are deliberately synthetic and local.  They validate enum-to-constructor
mapping and tensor contracts; they are not substitutes for an approved GH200/nuScenes
execution gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fl_v3.config import ConfigError, load_resolved_config
from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.losses import MultiTaskCenterPointLoss
from fl_v3.training.tasks import (
    _apply_production_sampling,
    _det_config_from_run,
    _production_sampler,
)
from fl_v3.utils import runtime


def _run_config(mode: str, camera: str, lidar: str, fusion: str) -> dict:
    return {
        "s06-production-runtime": True,
        "model-mode": mode,
        "det-camera-arch": camera,
        "det-camera-pretrained": False if camera != "none" else None,
        "det-lidar-arch": lidar,
        "det-fusion-arch": fusion,
        "det-head-arch": "centerhead_multitask",
        "det-lidar-sweeps": 10,
        "precision": "fp16",
    }


def test_resolved_camera_constructor_is_stride8_half_metre_and_180_grid():
    cfg = _det_config_from_run(_run_config("camera_only", "swin_t_stride8", "none", "none"))
    assert cfg.model_mode == "camera_only"
    assert cfg.camera_backbone == "swin_t"
    assert cfg.image_hw == (256, 704)
    assert cfg.feat_stride == 8
    assert cfg.depth_bins == (1.0, 60.0, 0.5)
    assert cfg.reference_camera is True
    assert cfg.camera_bev_output_dtype == "input"
    assert (cfg.bev.head_ny, cfg.bev.head_nx) == (180, 180)
    assert cfg.head_conv_layers == 2 and cfg.fusion_channels == 256


def test_resolved_pillar_and_second_constructors_keep_distinct_reviewed_grids():
    pillar = _det_config_from_run(_run_config("lidar_only", "none", "pillar_020", "none"))
    assert pillar.lidar_encoder == "pillar"
    assert pillar.required_spconv_version is None
    assert pillar.bev.bev_voxel == (0.2, 0.2)
    assert (pillar.bev.ny, pillar.bev.nx) == (512, 512)
    assert (pillar.bev.head_ny, pillar.bev.head_nx) == (256, 256)
    assert pillar.lidar_backbone and pillar.lidar_backbone_stages == 4

    second = _det_config_from_run(_run_config("lidar_only", "none", "second_075", "none"))
    assert second.lidar_encoder == "voxel"
    assert second.required_spconv_version == "2.3.8"
    assert second.lidar_input_bev is not None
    assert second.lidar_input_bev.bev_voxel == (0.075, 0.075)
    assert second.lidar_input_bev.out_size_factor == 8
    assert (second.lidar_input_bev.ny, second.lidar_input_bev.nx) == (1440, 1440)
    assert (second.bev.ny, second.bev.nx) == (180, 180)
    assert second.max_voxels_train == 120000
    assert second.max_voxels_eval == 160000
    assert second.sparse_conv_fp16 is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("det-camera-arch", "legacy_camera"),
        ("det-lidar-arch", "legacy_lidar"),
        ("det-fusion-arch", "legacy_fuser"),
        ("det-head-arch", "legacy_head"),
    ],
)
def test_unknown_resolved_architecture_fails_before_model_construction(field, value):
    run = _run_config("fusion", "swin_t_stride8", "second_075", "conv_fuser_256")
    run[field] = value
    with pytest.raises(ValueError, match="unknown resolved"):
        _det_config_from_run(run)


def test_mode_architecture_mismatch_fails_before_model_construction():
    run = _run_config("camera_only", "swin_t_stride8", "second_075", "none")
    with pytest.raises(ValueError, match="inconsistent"):
        _det_config_from_run(run)


def test_runtime_sparse_identity_binds_torch_packages_sources_and_imports(monkeypatch):
    run = _run_config("lidar_only", "none", "second_075", "none")
    run.update({
        "dependency-torch": torch.__version__,
        "dependency-spconv": "2.3.8",
        "dependency-spconv-build-sha256": "a" * 64,
        "dependency-spconv-source-sha": "2" * 40,
        "dependency-cumm": "0.7.13",
        "dependency-cumm-build-sha256": "b" * 64,
        "dependency-cumm-source-sha": "3" * 40,
    })
    versions = {"spconv": "2.3.8", "cumm": "0.7.13"}
    sources = {"spconv": ("2" * 40, "/src/spconv/spconv/__init__.py"),
               "cumm": ("3" * 40, "/src/cumm/cumm/__init__.py")}
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(runtime, "_source_checkout_identity", lambda dist, _imp: sources[dist])
    monkeypatch.setattr(
        runtime, "_runtime_package_sha256",
        lambda name: {"spconv": "a" * 64, "cumm": "b" * 64}[name],
    )
    identity = runtime.verify_runtime_dependency_identity(run)
    assert identity["spconv_source_sha"] == "2" * 40
    assert identity["cumm_source_sha"] == "3" * 40
    assert identity["spconv_build_sha256"] == "a" * 64
    assert identity["torch"] == torch.__version__
    run["dependency-spconv-source-sha"] = "4" * 40
    with pytest.raises(RuntimeError, match="source identity drift"):
        runtime.verify_runtime_dependency_identity(run)


class _InfoDataset:
    def __init__(self):
        self._infos = [
            {"gt_labels": np.asarray([0]), "gt_in_range": np.asarray([True])}
            for _ in range(9)
        ] + [{"gt_labels": np.asarray([4]), "gt_in_range": np.asarray([True])}]

    def __len__(self):
        return len(self._infos)

    def __getitem__(self, index):
        return index


def test_f_cbgs_is_deterministic_and_cannot_stack_loss_weights():
    dataset = _InfoDataset()
    run = {
        "s06-production-runtime": True, "det-cbgs": True,
        "det-cbgs-thresh": 0.5, "det-cbgs-max-repeat": 4.0,
        "det-class-weights": None, "det-reg-class-weights": None, "seed": 7,
    }
    a = _apply_production_sampling(dataset, run, shuffle=True)
    b = _apply_production_sampling(dataset, run, shuffle=True)
    assert np.array_equal(a.indices, b.indices)
    assert len(a) > len(dataset)
    assert _apply_production_sampling(dataset, run, shuffle=False) is dataset
    bad = dict(run, **{"det-class-weights": [1.0] * 10})
    with pytest.raises(ValueError, match="replaces rather than stacks"):
        _apply_production_sampling(dataset, bad, shuffle=True)


def test_production_sampler_is_epoch_addressable_over_expanded_cbgs_dataset():
    dataset = _InfoDataset()
    run = {
        "s06-production-runtime": True, "det-cbgs": True,
        "det-cbgs-thresh": 0.5, "det-cbgs-max-repeat": 4.0,
        "det-class-weights": None, "det-reg-class-weights": None, "seed": 7,
    }
    expanded = _apply_production_sampling(dataset, run, shuffle=True)
    sampler = _production_sampler(expanded, run, shuffle=True)
    sampler.set_epoch(3)
    continuous = list(sampler)
    resumed = _production_sampler(expanded, run, shuffle=True)
    resumed.set_epoch(3)
    assert list(resumed) == continuous
    assert sorted(continuous) == list(range(len(expanded)))
    assert _production_sampler(dataset, run, shuffle=False) is None


@pytest.mark.parametrize(
    "name,mode,lidar,sampling",
    [
        ("s07_b_c_str8.json", "camera_only", "none", "uniform"),
        ("s07_b_l_p020.json", "lidar_only", "pillar_020", "uniform"),
        ("s07_b_l_s075.json", "lidar_only", "second_075", "uniform"),
        ("s07_b_f_u.json", "fusion", "second_075", "uniform"),
        ("s07_b_f_cbgs.json", "fusion", "second_075", "cbgs"),
    ],
)
def test_candidate_templates_name_exact_choices_and_fail_closed(name, mode, lidar, sampling):
    path = Path(__file__).resolve().parents[1] / "configs" / name
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["model"]["mode"] == mode
    assert raw["model"]["lidar_arch"] == lidar
    assert raw["model"]["head_arch"] == "centerhead_multitask"
    assert raw["precision"] == "fp16"
    assert raw["training"]["sampling"] == sampling
    assert raw["model"]["camera_pretrained"] is None
    with pytest.raises(ConfigError, match="template_only"):
        load_resolved_config(path)


def _task_output(classes: int, size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "heatmap": torch.zeros(1, classes, size, size, requires_grad=True),
        "reg": torch.zeros(1, 2, size, size, requires_grad=True),
        "height": torch.zeros(1, 1, size, size, requires_grad=True),
        "dim": torch.zeros(1, 3, size, size, requires_grad=True),
        "rot": torch.zeros(1, 2, size, size, requires_grad=True),
        "vel": torch.zeros(1, 2, size, size, requires_grad=True),
    }


def test_multitask_loss_maps_global_labels_and_reaches_every_task_head():
    bev = BEVConfig(
        point_cloud_range=(-2.0, -2.0, -1.0, 2.0, 2.0, 1.0),
        bev_voxel=(1.0, 1.0),
        out_size_factor=1,
    )
    outputs = [_task_output(n) for n in (1, 2, 2, 1, 2, 2)]
    batch = {
        "gt_boxes": [torch.tensor([[0.25, 0.25, 0.0, 1.0, 1.0, 1.0, 0.0]])],
        "gt_labels": [torch.tensor([4])],
        "gt_velocity": [torch.tensor([[0.5, -0.25]])],
    }
    criterion = MultiTaskCenterPointLoss(bev)
    loss = criterion(outputs, batch)
    assert torch.isfinite(loss)
    assert criterion.last_terms["n_gt"] == 1
    loss.backward()
    assert all(task["heatmap"].grad is not None for task in outputs)
    assert criterion.losses[1].last_terms["n_gt"] == 1
    assert sum(item.last_terms["n_gt"] for item in criterion.losses) == 1


def test_multitask_loss_rejects_legacy_single_head_output():
    criterion = MultiTaskCenterPointLoss(BEVConfig())
    with pytest.raises(ValueError, match="six task"):
        criterion({"heatmap": torch.zeros(1, 10, 1, 1)}, {
            "gt_boxes": [], "gt_labels": [],
        })
