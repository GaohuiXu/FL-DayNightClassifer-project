"""S07-B seams across reviewed S02-S06 components.

These tests are deliberately synthetic and local.  They validate enum-to-constructor
mapping and tensor contracts; they are not substitutes for an approved GH200/nuScenes
execution gate.
"""
from __future__ import annotations

import json
import ast
import copy
import inspect
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

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


def _centralized_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "centralized_train.py"
    spec = importlib.util.spec_from_file_location("s07b_centralized_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script_module(filename: str, module_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


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
        "dependency-torch-build-sha256": "f" * 64,
        "dependency-torch-source-sha": "1" * 40,
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
    monkeypatch.setattr(runtime.torch.version, "git_version", "1" * 40)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(runtime, "_source_checkout_identity", lambda dist, _imp: sources[dist])
    monkeypatch.setattr(
        runtime, "_runtime_build_identity",
        lambda distribution, _name, _targets, _metadata: (
            {"torch": "f" * 64, "spconv": "a" * 64, "cumm": "b" * 64}[distribution],
            [f"/attested/{distribution}.so"],
        ),
    )
    monkeypatch.setattr(
        runtime, "_executable_artifact_records",
        lambda distribution, _name: [{"path": f"/attested/{distribution}.so", "bytes": 1,
                                      "sha256": "0" * 64}],
    )
    identity = runtime.verify_runtime_dependency_identity(run)
    assert identity["spconv_source_sha"] == "2" * 40
    assert identity["cumm_source_sha"] == "3" * 40
    assert identity["spconv_build_sha256"] == "a" * 64
    assert identity["torch"] == torch.__version__
    assert identity["torch_build_sha256"] == "f" * 64
    run["dependency-spconv-source-sha"] = "4" * 40
    with pytest.raises(RuntimeError, match="source identity drift"):
        runtime.verify_runtime_dependency_identity(run)


def _write_dist_info(root: Path, name: str) -> None:
    dist = root / f"{name}-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n", encoding="utf-8"
    )
    (dist / "RECORD").write_text(f"{name}/__init__.py,,\n", encoding="utf-8")


def test_executable_manifest_uses_real_files_and_rejects_loaded_native_outside_root(
    tmp_path, monkeypatch,
):
    name = "s07b_hostile_dist"
    package = tmp_path / name
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    native = package / "inside.so"
    native.write_bytes(b"native-a")
    _write_dist_info(tmp_path, name)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    first, origins = runtime._runtime_build_identity(name, name, (name,), {"version": "1.0"})
    assert len(first) == 64 and any(path.endswith("__init__.py") for path in origins)
    native.write_bytes(b"native-b")
    second, _ = runtime._runtime_build_identity(name, name, (name,), {"version": "1.0"})
    assert second != first

    outside = tmp_path / "outside.so"
    outside.write_bytes(b"outside")
    injected = ModuleType(name + ".native")
    injected.__file__ = str(outside)
    monkeypatch.setitem(sys.modules, name + ".native", injected)
    with pytest.raises(RuntimeError, match="outside attested roots"):
        runtime._runtime_build_identity(name, name, (name,), {"version": "1.0"})


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


class _StrictEvalTask:
    def __init__(self):
        self.infos = [{"sample_token": "token-b"}, {"sample_token": "token-a"}]
        self.loader_calls = 0

    def _load_info(self, _run, split):
        assert split == "mini_val"
        return self.infos, {}

    def _make_loader(self, _run, infos, tokens, shuffle):
        self.loader_calls += 1
        assert infos is self.infos and tokens == ["token-a", "token-b"] and shuffle is False
        return object()


def test_strict_official_eval_is_token_complete_single_decode_and_provenance_bound(
    tmp_path, monkeypatch,
):
    entry = _centralized_train_module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"exact checkpoint")
    config = SimpleNamespace(sha256="c" * 64)
    run = {
        "evaluation-checkpoint-weights": "raw", "evaluation-timing": True,
        "nuscenes-version": "v1.0-mini", "nuscenes-val-split": "mini_val",
        "nuscenes-dataroot": str(tmp_path), "model-mode": "camera_only",
        "resolved-config-sha256": config.sha256,
        "nuscenes-train-cache-logical-sha256": "1" * 64,
        "nuscenes-train-cache-pickle-sha256": "2" * 64,
        "nuscenes-train-cache-sidecar-sha256": "3" * 64,
        "nuscenes-val-cache-logical-sha256": "4" * 64,
        "nuscenes-val-cache-pickle-sha256": "5" * 64,
        "nuscenes-val-cache-sidecar-sha256": "6" * 64,
        "nuscenes-zip-manifest-logical-sha256": "7" * 64,
        "nuscenes-zip-manifest-file-sha256": "8" * 64,
    }
    load_calls = []
    monkeypatch.setattr(
        entry, "load_checkpoint",
        lambda path, **kwargs: (load_calls.append((path, kwargs)) or (object(), config.sha256)),
    )
    decode_calls = []

    def decode(_model, _loader, _device, decode_run, timing):
        decode_calls.append(dict(decode_run))
        timing.update({"batches": 1, "total_seconds": 0.0, "batch_seconds": [0.0]})
        return [SimpleNamespace(sample_token="token-a", boxes=np.zeros((0, 7))),
                SimpleNamespace(sample_token="token-b", boxes=np.zeros((0, 7)))]

    captured = {}

    def official(_nusc, decodes, eval_set, version, _out, _names, **kwargs):
        captured.update(kwargs)
        assert len(decodes) == 2 and eval_set == "mini_val" and version == "v1.0-mini"
        assert kwargs["all_eval_tokens"] == ["token-a", "token-b"]
        provenance = kwargs["run_config"]
        assert provenance["checkpoint-sha256"] == entry._checkpoint_sha256(checkpoint)
        assert provenance["checkpoint-weights"] == "raw"
        assert len(provenance["runtime-dependencies-sha256"]) == 64
        return {"mAP": 0.0, "NDS": 0.0}

    task = _StrictEvalTask()
    model = torch.nn.Linear(1, 1)
    metrics = entry.run_strict_official_evaluation(
        config=config, run_config=run, runtime_dependencies={"torch": "exact"},
        task=task, model=model, optimizer=object(), scheduler=object(), scaler=object(),
        ema=None, checkpoint=checkpoint, device=torch.device("cpu"), output_dir=tmp_path,
        decode_fn=decode, official_eval_fn=official,
        nusc_factory=lambda *args, **kwargs: object(),
    )
    assert metrics == {"mAP": 0.0, "NDS": 0.0}
    assert len(load_calls) == len(decode_calls) == task.loader_calls == 1
    assert (tmp_path / "evaluation_timing.json").is_file()


@pytest.mark.parametrize("tokens,match", [(["token-a"], "not token-complete"),
                                            (["token-a", "token-a"], "more than once")])
def test_strict_official_eval_rejects_missing_or_duplicate_decode_tokens(
    tmp_path, monkeypatch, tokens, match,
):
    entry = _centralized_train_module()
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(sha256="d" * 64)
    run = {
        "evaluation-checkpoint-weights": "raw", "evaluation-timing": False,
        "nuscenes-version": "v1.0-mini", "nuscenes-val-split": "mini_val",
        "nuscenes-dataroot": str(tmp_path),
    }
    monkeypatch.setattr(entry, "load_checkpoint", lambda *args, **kwargs: (object(), config.sha256))
    with pytest.raises(RuntimeError, match=match):
        entry.run_strict_official_evaluation(
            config=config, run_config=run, runtime_dependencies={}, task=_StrictEvalTask(),
            model=torch.nn.Linear(1, 1), optimizer=object(), scheduler=object(), scaler=object(),
            ema=None, checkpoint=checkpoint, device=torch.device("cpu"), output_dir=tmp_path,
            decode_fn=lambda *_args: [SimpleNamespace(sample_token=token, boxes=np.zeros((0, 7)))
                                      for token in tokens],
            official_eval_fn=lambda *_args, **_kwargs: pytest.fail("official eval reached"),
            nusc_factory=lambda *args, **kwargs: object(),
        )


def test_strict_official_eval_rejects_over_500_before_devkit(tmp_path, monkeypatch):
    entry = _centralized_train_module()
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(sha256="e" * 64)
    run = {
        "evaluation-checkpoint-weights": "raw", "evaluation-timing": False,
        "nuscenes-version": "v1.0-mini", "nuscenes-val-split": "mini_val",
        "nuscenes-dataroot": str(tmp_path),
    }
    monkeypatch.setattr(entry, "load_checkpoint", lambda *args, **kwargs: (object(), config.sha256))
    records = [SimpleNamespace(sample_token="token-a", boxes=np.zeros((501, 7))),
               SimpleNamespace(sample_token="token-b", boxes=np.zeros((0, 7)))]
    with pytest.raises(RuntimeError, match="per-sample box cap"):
        entry.run_strict_official_evaluation(
            config=config, run_config=run, runtime_dependencies={}, task=_StrictEvalTask(),
            model=torch.nn.Linear(1, 1), optimizer=object(), scheduler=object(), scaler=object(),
            ema=None, checkpoint=checkpoint, device=torch.device("cpu"), output_dir=tmp_path,
            decode_fn=lambda *_args: records,
            official_eval_fn=lambda *_args, **_kwargs: pytest.fail("official eval reached"),
            nusc_factory=lambda *args, **kwargs: object(),
        )


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


def _six_tasks():
    return [_task_output(count, size=2) for count in (1, 2, 2, 1, 2, 2)]


def _synthetic_s06_payload(resolved):
    from fl_v3.training.checkpoint import CHECKPOINT_SCHEMA
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": {}, "optimizer": {}, "scheduler": {}, "grad_scaler": {},
        "ema": {} if resolved.data["training"]["ema_decay"] is not None else None,
        "training_state": {}, "rng": {},
        "resolved_config_sha256": resolved.sha256,
        "resolved_config": resolved.as_dict(),
        "model_mode": resolved.model_mode,
        "precision": resolved.precision,
        "data_identities": resolved.data_identities,
        "checkpoint_identity": resolved.sha256,
    }


def test_t5_condition_decode_consumes_task_outputs_without_legacy_global_k(monkeypatch):
    from fl_v3.attacks import fusion_ablation as ablation

    calls = []

    class Model:
        fusion = staticmethod(lambda camera, lidar: camera + lidar)
        bev_neck = staticmethod(lambda value: value)
        head = staticmethod(lambda _value: _six_tasks())

        def __call__(self, _batch, return_intermediates=False):
            if return_intermediates:
                return {
                    "task_outputs": _six_tasks(),
                    "_camera_bev": torch.ones(1, 1, 2, 2),
                    "_lidar_bev": torch.ones(1, 1, 2, 2),
                }
            return _six_tasks()

        def decode(self, head, **kwargs):
            calls.append((head, kwargs))
            assert isinstance(head, list) and len(head) == 6
            assert set(kwargs) == {"score_threshold"}
            return [{"boxes": torch.zeros(0, 7), "scores": torch.zeros(0),
                     "labels": torch.zeros(0, dtype=torch.long), "velocity": torch.zeros(0, 2)}]

    monkeypatch.setattr(ablation, "_collate_one", lambda *_args: {})
    model = Model()
    ablation._decode(model, {}, torch.device("cpu"), 0.1)
    ablation._decode_cond4_and_cond5a(model, {}, torch.device("cpu"), 0.1)
    assert len(calls) == 3


def test_mini_matrix_six_task_telemetry_and_delta_use_every_branch(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "arrhenius_mini_matrix.py"
    spec = importlib.util.spec_from_file_location("s07b_mini_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    base = _six_tasks()
    changed = [
        {key: value.detach().clone() for key, value in task.items()}
        for task in base
    ]
    changed[5]["heatmap"] += 1.0
    stats = module._head_task_stats(base)
    delta = module._head_delta_stats(changed, base)
    assert len(stats) == 6 and set(delta) == {f"task_{index}" for index in range(6)}
    assert delta["task_5"]["heatmap"]["nonzero"] > 0
    assert delta["task_0"]["heatmap"]["nonzero"] == 0


def test_primary_and_historical_caller_inventory_is_fail_closed_and_no_legacy_k():
    root = Path(__file__).resolve().parents[1]
    paths = {
        "fusion": root / "src/fl_v3/attacks/fusion_ablation.py",
        "t4": root / "scripts/t4_readiness_eval.py",
        "t5": root / "scripts/t5_attack_eval.py",
        "mini": root / "scripts/arrhenius_mini_matrix.py",
    }
    for path in paths.values():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "decode":
                    assert all(keyword.arg != "max_objects" for keyword in node.keywords)
    for label in ("t4", "t5"):
        source = paths[label].read_text(encoding="utf-8")
        assert "load_checkpoint(" in source and "CHECKPOINT_SCHEMA" in source
        assert "refuses legacy/bare checkpoints" in source
    for name in (
        "_t4_fd_diagnose.py", "t3_trainval_reeval_fullval.py",
        "p3_crt_probe.py", "p3_grad_conflict.py",
    ):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "frozen historical" in source or "frozen legacy" in source
        assert "RuntimeError" in source


@pytest.mark.parametrize(
    "filename,function_name",
    [("t4_readiness_eval.py", "_load_s06_eval_model"),
     ("t5_attack_eval.py", "_load_model")],
)
def test_t4_t5_callers_use_complete_checkpoint_loader_and_bind_provenance(
    tmp_path, monkeypatch, filename, function_name,
):
    from test_s06_resolved_config import valid_config
    import fl_v3.config as config_module
    import fl_v3.training.checkpoint as checkpoint_module
    import fl_v3.training.tasks as tasks_module
    import fl_v3.utils.runtime as runtime_module

    module = _script_module(filename, "s07b_" + filename.replace(".py", ""))
    raw = valid_config(tmp_path)
    resolved = config_module.resolve_config(raw)
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"exact-s06-checkpoint")
    payload = _synthetic_s06_payload(resolved)
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(config_module, "verify_physical_data_identities", lambda _cfg: None)
    monkeypatch.setattr(runtime_module, "verify_runtime_dependency_identity",
                        lambda _run: {"torch": "attested"})
    monkeypatch.setattr(runtime_module, "make_grad_scaler", lambda *_args: object())

    class Task:
        @staticmethod
        def build_model(_run):
            return torch.nn.Linear(2, 2)

    monkeypatch.setattr(tasks_module, "get_task", lambda _name: Task())
    load_calls = []
    monkeypatch.setattr(
        checkpoint_module, "load_checkpoint",
        lambda path, **kwargs: (load_calls.append((path, kwargs)) or (object(), resolved.sha256)),
    )
    caller = {"batch-size": 1, "num-workers": 0, "det-eval-limit": 0}
    if filename.startswith("t5"):
        module._preflight_t5_checkpoints(caller, str(checkpoint), None)
    model = getattr(module, function_name)(caller, str(checkpoint), torch.device("cpu"))
    assert isinstance(model, torch.nn.Module) and len(load_calls) == 1
    assert caller["resolved-config-sha256"] == resolved.sha256
    expected_checkpoint_hash = (
        module._sha256_file(str(checkpoint)) if filename.startswith("t4")
        else module._checkpoint_file_sha256(str(checkpoint))
    )
    assert caller["checkpoint-sha256"] == expected_checkpoint_hash
    assert caller["checkpoint-weights"] == "raw"
    assert len(caller["runtime-dependencies-sha256"]) == 64


@pytest.mark.parametrize("checkpoint_weights", ["raw", "ema"])
def test_t5_main_preflights_existing_compat_config_before_device_seed_or_data(
    tmp_path, monkeypatch, checkpoint_weights,
):
    from test_s06_resolved_config import valid_config
    import fl_v3.config as config_module
    import fl_v3.eval.asr as asr_module
    import fl_v3.attacks.trigger as trigger_module
    import fl_v3.training.checkpoint as checkpoint_module
    import fl_v3.training.tasks as tasks_module
    import fl_v3.utils.runtime as runtime_module

    module = _script_module("t5_attack_eval.py", "s07b_t5_order")
    repo = Path(__file__).resolve().parents[1]
    compatibility = json.loads((repo / "configs/t5_attack.json").read_text(encoding="utf-8"))
    compatibility["nuscenes-dataroot"] = str(tmp_path / "explicit_data")
    compatibility["nuscenes-cache-dir"] = str(tmp_path / "explicit_cache")
    compatibility["nuscenes-zip-manifest"] = str(tmp_path / "explicit_manifest.sqlite")
    compatibility_path = tmp_path / "t5_compatibility.json"
    compatibility_path.write_text(json.dumps(compatibility), encoding="utf-8")
    raw = valid_config(tmp_path)
    raw["evaluation"]["checkpoint_weights"] = checkpoint_weights
    raw["precision"] = "fp32"
    raw["optimizer"]["learning_rate"] = float(compatibility["learning-rate"])
    raw["optimizer"]["weight_decay"] = float(compatibility["weight-decay"])
    raw["training"]["seed"] = int(compatibility["seed"])
    raw["data"].update(
        dataroot=str(compatibility["nuscenes-dataroot"]),
        version=str(compatibility["nuscenes-version"]),
        train_split=str(compatibility["nuscenes-train-split"]),
        val_split=str(compatibility["nuscenes-val-split"]),
    )
    raw["data"]["zip_manifest"]["path"] = compatibility["nuscenes-zip-manifest"]
    cache_dir = Path(str(compatibility["nuscenes-cache-dir"]))
    for role in ("train", "val"):
        raw["data"]["caches"][role]["path"] = str(cache_dir / f"{role}.pkl")
        raw["data"]["caches"][role]["sidecar_path"] = str(cache_dir / f"{role}.meta.json")
    resolved = config_module.resolve_config(raw)
    checkpoint = tmp_path / "poison.pt"; checkpoint.write_bytes(b"synthetic-poison")
    clean_checkpoint = tmp_path / "clean.pt"; clean_checkpoint.write_bytes(b"synthetic-clean")
    events = []
    monkeypatch.setattr(
        module.torch, "load",
        lambda path, **kwargs: (
            events.append("parse_clean" if str(path) == str(clean_checkpoint) else "parse_poison")
            or _synthetic_s06_payload(resolved)
        ),
    )
    monkeypatch.setattr(
        config_module, "verify_physical_data_identities",
        lambda _cfg: events.append("physical"),
    )
    monkeypatch.setattr(
        runtime_module, "verify_runtime_dependency_identity",
        lambda _run: (events.append("dependency") or {"torch": "attested"}),
    )
    monkeypatch.setattr(runtime_module, "make_grad_scaler", lambda *_args: object())
    monkeypatch.setattr(module, "_device", lambda cfg: (events.append("device") or torch.device("cpu")))

    def seeded(cfg):
        events.append("seed")
        assert cfg["precision"] == "fp32"
        assert cfg["s06-production-runtime"] is True
        assert cfg["model-mode"] == "camera_only"

    monkeypatch.setattr(module, "_seed", seeded)
    frozen_clean = str(compatibility["attack-clean-checkpoint-checksum"])
    frozen_targets = [[f"sample-{index:02d}", f"ann-{index:02d}"] for index in range(40)]
    monkeypatch.setattr(
        module, "_load_subset",
        lambda *_args: {"targets": frozen_targets, "n": len(frozen_targets),
                        "content_hash": "9" * 64,
                        "checkpoint_checksum": frozen_clean},
    )
    monkeypatch.setattr(asr_module, "thresholds_from_subset", lambda _subset: object())
    monkeypatch.setattr(trigger_module, "trigger_spec_from_run_config", lambda _cfg: object())
    monkeypatch.setattr(module, "_val_info", lambda _cfg: (events.append("val_info") or []))
    monkeypatch.setattr(module, "_val_dataset",
                        lambda *_args: (events.append("dataset") or []))

    class Task:
        @staticmethod
        def build_model(_run):
            events.append("build_model")
            class TrackedLinear(torch.nn.Linear):
                def load_state_dict(self, *args, **kwargs):
                    events.append("select_ema")
                    return super().load_state_dict(*args, **kwargs)

            model = TrackedLinear(2, 2)
            model.selected_checksum = None
            return model

    monkeypatch.setattr(tasks_module, "get_task", lambda _name: Task())

    def load_checkpoint(path, **kwargs):
        events.append("load_clean" if str(path) == str(clean_checkpoint) else "load_poison")
        kwargs["model"].selected_checksum = (
            frozen_clean if str(path) == str(clean_checkpoint) else "8" * 64
        )
        return object(), resolved.sha256

    monkeypatch.setattr(checkpoint_module, "load_checkpoint", load_checkpoint)
    def selected_checksum(model):
        events.append("checksum_clean" if model.selected_checksum == frozen_clean else "checksum_poison")
        return model.selected_checksum

    monkeypatch.setattr(module, "_trainable_checksum", selected_checksum)
    original_makedirs = module.os.makedirs

    def tracked_makedirs(path, *args, **kwargs):
        events.append("makedirs")
        return original_makedirs(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "makedirs", tracked_makedirs)
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "t5_attack_eval.py", "--task", "shard", "--config",
        str(compatibility_path), "--checkpoint", str(checkpoint),
        "--clean-checkpoint", str(clean_checkpoint),
        "--subset", str(tmp_path / "subset.json"), "--output-dir", str(output),
        "--run-id", "unit-run",
    ])
    module.main()
    assert events.index("parse_poison") < events.index("parse_clean")
    assert events.index("parse_clean") < events.index("makedirs")
    assert max(index for index, event in enumerate(events) if event == "dependency") \
        < events.index("makedirs")
    assert events.index("dependency") < events.index("device") < events.index("seed")
    assert events.index("seed") < events.index("build_model") < events.index("load_poison")
    assert events.index("load_clean") < events.index("checksum_clean") < events.index("val_info")
    if checkpoint_weights == "ema":
        assert events.count("select_ema") == 2
        assert events.index("load_poison") < events.index("select_ema") < events.index("checksum_poison")
    else:
        assert "select_ema" not in events
    assert events.index("val_info") < events.index("dataset")


def test_t5_preflight_rejects_missing_legacy_caller_drift_and_embedded_drift(
    tmp_path, monkeypatch,
):
    from test_s06_resolved_config import valid_config
    import fl_v3.config as config_module
    import fl_v3.utils.runtime as runtime_module

    module = _script_module("t5_attack_eval.py", "s07b_t5_hostiles")
    with pytest.raises(RuntimeError, match="explicit complete S06 checkpoint"):
        module._preflight_t5_checkpoints({}, None, None)

    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {})
    with pytest.raises(RuntimeError, match="legacy/bare checkpoints"):
        module._preflight_t5_checkpoints({}, str(checkpoint), None)

    resolved = config_module.resolve_config(valid_config(tmp_path))
    payload = _synthetic_s06_payload(resolved)
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(config_module, "verify_physical_data_identities", lambda _cfg: None)
    monkeypatch.setattr(runtime_module, "verify_runtime_dependency_identity",
                        lambda _run: {"torch": "attested"})
    with pytest.raises(RuntimeError, match="caller/checkpoint resolved-config drift"):
        module._preflight_t5_checkpoints({"precision": "fp16"}, str(checkpoint), None)

    embedded_drift = dict(payload); embedded_drift["resolved_config_sha256"] = "0" * 64
    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: embedded_drift)
    with pytest.raises(RuntimeError, match="embedded config/data metadata drift"):
        module._preflight_t5_checkpoints({}, str(checkpoint), None)

    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: payload)
    authoritative = {}
    module._preflight_t5_checkpoints(authoritative, str(checkpoint), None)
    original_mode = authoritative["model-mode"]
    authoritative["model-mode"] = "fusion"
    with pytest.raises(RuntimeError, match="authoritative config changed after preflight"):
        module._load_model(authoritative, str(checkpoint), torch.device("cpu"))
    authoritative["model-mode"] = original_mode
    checkpoint.write_bytes(b"changed-after-preflight")
    with pytest.raises(RuntimeError, match="changed after authoritative preflight"):
        module._load_model(authoritative, str(checkpoint), torch.device("cpu"))


def test_t5_preflight_rejects_clean_poison_resolved_identity_mismatch(tmp_path, monkeypatch):
    from test_s06_resolved_config import valid_config
    import fl_v3.config as config_module
    import fl_v3.utils.runtime as runtime_module

    module = _script_module("t5_attack_eval.py", "s07b_t5_pair")
    poison_raw = valid_config(tmp_path)
    clean_raw = valid_config(tmp_path); clean_raw["evaluation"]["timing"] = True
    poison = config_module.resolve_config(poison_raw)
    clean = config_module.resolve_config(clean_raw)
    poison_path = tmp_path / "poison.pt"; poison_path.write_bytes(b"poison")
    clean_path = tmp_path / "clean.pt"; clean_path.write_bytes(b"clean")
    payloads = {
        str(poison_path): _synthetic_s06_payload(poison),
        str(clean_path): _synthetic_s06_payload(clean),
    }
    monkeypatch.setattr(module.torch, "load", lambda path, **kwargs: payloads[str(path)])
    monkeypatch.setattr(config_module, "verify_physical_data_identities", lambda _cfg: None)
    monkeypatch.setattr(runtime_module, "verify_runtime_dependency_identity",
                        lambda _run: {"torch": "attested"})
    caller = {}
    with pytest.raises(RuntimeError, match="clean/poison checkpoint resolved identities differ"):
        module._preflight_t5_checkpoints(caller, str(poison_path), str(clean_path))
    assert caller == {} and module._CHECKPOINT_PREFLIGHTS == {}


def test_t5_task_requirement_matrix_fails_before_output_side_effect(tmp_path, monkeypatch):
    module = _script_module("t5_attack_eval.py", "s07b_t5_requirements")
    config = tmp_path / "config.json"; config.write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr(module.os, "makedirs", lambda *_args, **_kwargs: called.append("makedirs"))
    monkeypatch.setattr(module, "_preflight_t5_checkpoints",
                        lambda *_args, **_kwargs: called.append("preflight"))
    monkeypatch.setattr(sys, "argv", [
        "t5_attack_eval.py", "--task", "shard", "--config", str(config),
        "--checkpoint", str(tmp_path / "poison.pt"), "--subset", "subset.json",
        "--output-dir", str(tmp_path / "out"), "--run-id", "matrix-run",
    ])
    with pytest.raises(RuntimeError, match="requires both poison and clean"):
        module.main()
    assert called == []

    base = SimpleNamespace(task="shard", checkpoint="poison", clean_checkpoint=None,
                           cond4_only=True, shard=0, num_shards=1,
                           run_id="matrix-run", subset="subset.json")
    assert module._validate_task_requirements(base) == module._SHARD_MODE_COND4
    for task, clean, expected in (
        ("aggregate", None, "poison_only"), ("stealth", None, "poison_only"),
        ("guards", None, "poison_only"), ("viz", "clean", module._SHARD_MODE_FULL),
    ):
        args = SimpleNamespace(task=task, checkpoint="poison", clean_checkpoint=clean,
                               cond4_only=False, shard=0, num_shards=1,
                               run_id="matrix-run", subset="subset.json")
        assert module._validate_task_requirements(args) == expected

    for task, clean, cond4, message in (
        ("shard", "clean", True, "clean checkpoint is forbidden"),
        ("aggregate", "clean", False, "poison-only"),
        ("stealth", "clean", False, "poison-only"),
        ("guards", "clean", False, "poison-only"),
        ("viz", None, False, "requires both poison and clean"),
        ("viz", "clean", True, "valid only for shard"),
        ("aggregate", None, True, "valid only for shard"),
        ("stealth", None, True, "valid only for shard"),
        ("guards", None, True, "valid only for shard"),
    ):
        args = SimpleNamespace(
            task=task, checkpoint="poison", clean_checkpoint=clean,
            cond4_only=cond4, shard=0, num_shards=1,
            run_id="matrix-run", subset="subset.json",
        )
        with pytest.raises(RuntimeError, match=message):
            module._validate_task_requirements(args)

    for bad_index, count in ((-1, 2), (2, 2), (0, 0)):
        args = SimpleNamespace(
            task="shard", checkpoint="poison", clean_checkpoint="clean",
            cond4_only=False, shard=bad_index, num_shards=count,
            run_id="matrix-run", subset="subset.json",
        )
        with pytest.raises(RuntimeError, match="index/count"):
            module._validate_task_requirements(args)

    for bad_run_id in (None, "", "../escape", "a/b"):
        args = SimpleNamespace(
            task="aggregate", checkpoint="poison", clean_checkpoint=None,
            cond4_only=False, shard=0, num_shards=1,
            run_id=bad_run_id, subset="subset.json",
        )
        with pytest.raises(RuntimeError, match="nonempty immutable --run-id"):
            module._validate_task_requirements(args)
    args = SimpleNamespace(
        task="aggregate", checkpoint="poison", clean_checkpoint=None,
        cond4_only=False, shard=0, num_shards=1,
        run_id="matrix-run", subset=None,
    )
    with pytest.raises(RuntimeError, match="requires the frozen subset"):
        module._validate_task_requirements(args)


def test_t5_checkpoint_preflight_rejects_complete_schema_and_metadata_hostiles(
    tmp_path, monkeypatch,
):
    from test_s06_resolved_config import valid_config
    import fl_v3.config as config_module
    import fl_v3.training.checkpoint as checkpoint_module
    import fl_v3.utils.runtime as runtime_module

    module = _script_module("t5_attack_eval.py", "s07b_t5_payload_matrix")
    resolved = config_module.resolve_config(valid_config(tmp_path))
    valid = _synthetic_s06_payload(resolved)
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"stable")
    monkeypatch.setattr(config_module, "verify_physical_data_identities", lambda _cfg: None)
    monkeypatch.setattr(runtime_module, "verify_runtime_dependency_identity",
                        lambda _run: {"torch": "attested"})

    cases = []
    missing = copy.deepcopy(valid); missing.pop("optimizer")
    cases.append((missing, "partial checkpoints"))
    extra = copy.deepcopy(valid); extra["unexpected"] = 1
    cases.append((extra, "partial checkpoints"))
    wrong_schema = copy.deepcopy(valid); wrong_schema["schema"] = "wrong.v1"
    cases.append((wrong_schema, "unsupported checkpoint schema"))
    for key, value in (
        ("model_mode", "fusion"), ("precision", "fp16"),
        ("data_identities", {"drift": True}),
    ):
        payload = copy.deepcopy(valid); payload[key] = value
        cases.append((payload, "embedded config/data metadata drift"))
    bad_identity = copy.deepcopy(valid); bad_identity["checkpoint_identity"] = "f" * 64
    cases.append((bad_identity, "does not equal embedded resolved-config"))
    bad_ema = copy.deepcopy(valid); bad_ema["ema"] = None
    cases.append((bad_ema, "EMA presence differs"))

    for payload, message in cases:
        monkeypatch.setattr(module.torch, "load", lambda *args, _p=payload, **kwargs: _p)
        with pytest.raises(RuntimeError, match=message):
            module._checkpoint_preflight({}, str(checkpoint))

    def replacing_load(*_args, **_kwargs):
        checkpoint.write_bytes(b"replaced-during-load")
        return valid

    monkeypatch.setattr(module.torch, "load", replacing_load)
    with pytest.raises(RuntimeError, match="changed during authoritative preflight"):
        module._checkpoint_preflight({}, str(checkpoint))


@pytest.mark.parametrize("drift_key", ["checkpoint_weights", "runtime_dependencies_sha256"])
def test_t5_preflight_rejects_clean_policy_or_runtime_drift(tmp_path, monkeypatch, drift_key):
    module = _script_module("t5_attack_eval.py", "s07b_t5_pair_" + drift_key)
    poison_path = os.path.abspath(tmp_path / "poison.pt")
    clean_path = os.path.abspath(tmp_path / "clean.pt")
    base = {
        "resolved_sha256": "1" * 64, "checkpoint_weights": "raw",
        "runtime_dependencies_sha256": "2" * 64, "strict": {},
        "checkpoint_sha256": "3" * 64,
    }
    poison = {**base, "path": poison_path}
    clean = {**base, "path": clean_path, drift_key: ("ema" if drift_key == "checkpoint_weights" else "4" * 64)}
    monkeypatch.setattr(
        module, "_checkpoint_preflight",
        lambda _cfg, path: poison if os.path.abspath(path) == poison_path else clean,
    )
    with pytest.raises(RuntimeError, match=("raw/EMA policies" if drift_key == "checkpoint_weights"
                                            else "runtime dependency identities")):
        module._preflight_t5_checkpoints({}, poison_path, clean_path)
    assert module._CHECKPOINT_PREFLIGHTS == {}


def _full_shard_row(module, target):
    from fl_v3.attacks import fusion_ablation as ablation
    return {
        "sample_token": target[0], "ann_token": target[1], "evaluated": True,
        "disappeared": {condition: False for condition in ablation.CONDITIONS},
        "occlusion_disappeared": False, "placement_aligned_ok": True,
        "placement_nonaligned_iou0": True, "area_ratio": 0.1,
    }


def test_t5_full_shard_aggregate_binds_identity_and_rejects_mixed_duplicate_rows(
    tmp_path,
):
    module = _script_module("t5_attack_eval.py", "s07b_t5_artifacts")
    output = tmp_path / "out"; run_id = "artifact-run"
    ablation_dir = output / run_id / "ablation"; ablation_dir.mkdir(parents=True)
    poison_path = tmp_path / "poison.pt"; poison_path.write_bytes(b"poison")
    poison_selected = "8" * 64
    frozen_clean = "9" * 64
    preflight = {
        "checkpoint_sha256": "1" * 64, "resolved_sha256": "2" * 64,
        "checkpoint_weights": "raw", "runtime_dependencies_sha256": "3" * 64,
    }
    module._CHECKPOINT_PREFLIGHTS[os.path.abspath(poison_path)] = preflight
    poison_identity = {
        "checkpoint_file_sha256": preflight["checkpoint_sha256"],
        "resolved_sha256": preflight["resolved_sha256"],
        "checkpoint_weights": preflight["checkpoint_weights"],
        "runtime_dependencies_sha256": preflight["runtime_dependencies_sha256"],
        "selected_weights_checksum": poison_selected,
    }
    clean_identity = {
        "checkpoint_file_sha256": "4" * 64, "resolved_sha256": "2" * 64,
        "checkpoint_weights": "raw", "runtime_dependencies_sha256": "3" * 64,
        "selected_weights_checksum": frozen_clean,
    }
    targets = [("sample-a", "ann-a"), ("sample-b", "ann-b")]
    subset = {"content_hash": "5" * 64, "checkpoint_checksum": frozen_clean,
              "targets": [list(target) for target in targets], "n": 2}
    args = SimpleNamespace(
        num_shards=2, output_dir=str(output), checkpoint=str(poison_path), run_id=run_id,
    )
    manifest_sha = "6" * 64
    manifest = {"clean": clean_identity}

    def artifact(index, target):
        return {
            "schema_version": module._SHARD_SCHEMA_FULL, "run_id": run_id,
            "run_manifest_sha256": manifest_sha, "mode": module._SHARD_MODE_FULL,
            "poison": copy.deepcopy(poison_identity), "clean": copy.deepcopy(clean_identity),
            "subset_content_hash": subset["content_hash"], "shard_index": index,
            "num_shards": 2, "results": [_full_shard_row(module, target)],
        }

    paths = []
    for index, target in enumerate(targets):
        path = ablation_dir / f"ablation_shard_{index}_of_2.full.json"
        path.write_text(json.dumps(artifact(index, target)), encoding="utf-8")
        paths.append(path)
    cfg = {"attack-clean-checkpoint-checksum": frozen_clean}
    run_fd = module._open_run_directory(args, create=False)

    def load_shards():
        return module._load_bound_full_shards(
            args, cfg, subset, poison_selected, manifest, manifest_sha, run_fd,
        )

    rows = load_shards()
    assert [(row["sample_token"], row["ann_token"]) for row in rows] == targets

    wrong_clean = artifact(1, targets[1])
    wrong_clean["clean"]["selected_weights_checksum"] = "7" * 64
    paths[1].write_text(json.dumps(wrong_clean), encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean identity differs|actual clean selected"):
        load_shards()
    wrong_subset = artifact(1, targets[1]); wrong_subset["subset_content_hash"] = "0" * 64
    paths[1].write_text(json.dumps(wrong_subset), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen-subset identity mismatch"):
        load_shards()
    extra_key = artifact(1, targets[1]); extra_key["unknown"] = True
    paths[1].write_text(json.dumps(extra_key), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or incomplete"):
        load_shards()

    missing_control = artifact(1, targets[1])
    missing_control["results"][0]["occlusion_disappeared"] = None
    paths[1].write_text(json.dumps(missing_control), encoding="utf-8")
    with pytest.raises(RuntimeError, match="occlusion control must be boolean"):
        load_shards()

    mixed = artifact(1, targets[1]); mixed["schema_version"] = module._SHARD_SCHEMA_COND4
    mixed["mode"] = module._SHARD_MODE_COND4; mixed["clean"] = None
    paths[1].write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid or incomplete|cond4-only/mixed"):
        load_shards()

    duplicate = artifact(0, targets[0])
    paths[1].write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate shard index|repeated byte-identical"):
        load_shards()

    duplicate_row = artifact(1, targets[0])
    paths[1].write_text(json.dumps(duplicate_row), encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate \(sample_token, ann_token\)"):
        load_shards()

    paths[1].unlink()
    with pytest.raises(RuntimeError, match="exact canonical full-shard filename set"):
        load_shards()

    paths[1].write_text(json.dumps(artifact(1, targets[1])), encoding="utf-8")
    for extra_name in (
        "ablation_shard_0_of_2.cond4.json",
        "ablation_shard_0_of_1.full.json",
        "ablation_shard_renamed.json",
    ):
        extra = ablation_dir / extra_name
        extra.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="exact canonical full-shard filename set"):
            load_shards()
        extra.unlink()
    os.close(run_fd)


def _t5_identity(selected: str, checkpoint_file: str = "1" * 64):
    return {
        "checkpoint_file_sha256": checkpoint_file,
        "resolved_sha256": "2" * 64,
        "checkpoint_weights": "raw",
        "runtime_dependencies_sha256": "3" * 64,
        "selected_weights_checksum": selected,
    }


def test_t5_run_manifest_is_immutable_exact_and_poison_only_reuses_it(tmp_path):
    module = _script_module("t5_attack_eval.py", "s07b_t5_manifest")
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"), run_id="fresh-run", num_shards=3,
        guard_samples=40,
    )
    subset = {
        "content_hash": "5" * 64,
        "targets": [[f"sample-{index:02d}", f"ann-{index:02d}"] for index in range(40)],
    }
    cfg = {"attack-clean-checkpoint-checksum": "9" * 64}
    poison = _t5_identity("8" * 64)
    clean = _t5_identity("9" * 64, checkpoint_file="4" * 64)

    manifest, manifest_sha, run_fd = module._bind_run_manifest(args, cfg, subset, poison, clean)
    assert manifest["run_id"] == "fresh-run"
    assert manifest["poison"] == poison and manifest["clean"] == clean
    assert manifest["task_plan"]["full_num_shards"] == 3
    assert manifest["task_plan"]["guard_selection"]["declared_sample_count"] == 40
    assert len(manifest_sha) == 64
    os.close(run_fd)
    reread, reread_sha, reread_fd = module._bind_run_manifest(args, cfg, subset, poison)
    assert reread == manifest and reread_sha == manifest_sha
    os.close(reread_fd)

    changed = dict(poison); changed["selected_weights_checksum"] = "7" * 64
    with pytest.raises(RuntimeError, match="does not exactly match"):
        module._bind_run_manifest(args, cfg, subset, changed)
    args.guard_samples = 1
    with pytest.raises(RuntimeError, match="does not exactly match"):
        module._bind_run_manifest(args, cfg, subset, poison)
    args.guard_samples = 40
    args.num_shards = 4
    with pytest.raises(RuntimeError, match="does not exactly match"):
        module._bind_run_manifest(args, cfg, subset, poison)

    args.num_shards = 3
    run_fd = module._open_run_directory(args, create=False)
    with pytest.raises(FileExistsError):
        module._write_json_exclusive_at(run_fd, "t5_run_manifest.json", manifest)
    os.close(run_fd)

    with pytest.raises(RuntimeError, match="exceeds frozen available"):
        module._guard_selection(subset, 41)
    with pytest.raises(RuntimeError, match="positive integer"):
        module._guard_selection(subset, 0)


def test_t5_manifest_publication_is_complete_noreplace_and_owns_only_its_temp(
    tmp_path, monkeypatch,
):
    module = _script_module("t5_attack_eval.py", "s07b_t5_atomic_manifest")
    args = SimpleNamespace(output_dir=str(tmp_path / "out"), run_id="atomic-run")
    run_fd = module._open_run_directory(args, create=True)
    value = {"schema_version": module._RUN_MANIFEST_SCHEMA, "payload": "x" * 256}
    other_live = ".t5_run_manifest.json.live-writer.deadbeef.tmp"
    other_fd = os.open(other_live, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=run_fd)
    os.write(other_fd, b"live-private-temp")

    original_write_all = module._write_all

    def partial_then_fail(fd, payload):
        os.write(fd, payload[:17])
        raise RuntimeError("simulated creator crash")

    monkeypatch.setattr(module, "_write_all", partial_then_fail)
    with pytest.raises(RuntimeError, match="simulated creator crash"):
        module._atomic_publish_json_at(run_fd, "t5_run_manifest.json", value)
    assert os.listdir(run_fd) == [other_live]

    monkeypatch.setattr(module, "_write_all", original_write_all)
    original_link = module.os.link
    observed = []

    def observing_link(*link_args, **link_kwargs):
        with pytest.raises(RuntimeError, match="missing or unsafe"):
            module._load_required_json_at(run_fd, "t5_run_manifest.json")
        observed.append("reader_saw_no_partial_final")
        return original_link(*link_args, **link_kwargs)

    monkeypatch.setattr(module.os, "link", observing_link)
    assert module._atomic_publish_json_at(run_fd, "t5_run_manifest.json", value) is True
    assert observed == ["reader_saw_no_partial_final"]
    loaded, loaded_sha = module._load_required_json_at(run_fd, "t5_run_manifest.json")
    assert loaded == value and len(loaded_sha) == 64
    assert other_live in os.listdir(run_fd)
    os.write(other_fd, b"-still-owned-by-other-publisher")

    monkeypatch.setattr(module.os, "link", original_link)
    assert module._atomic_publish_json_at(
        run_fd, "t5_run_manifest.json", {"different": True},
    ) is False
    winner, _winner_sha = module._load_required_json_at(run_fd, "t5_run_manifest.json")
    assert winner == value
    assert other_live in os.listdir(run_fd)
    os.close(other_fd)
    os.unlink(other_live, dir_fd=run_fd)
    assert not any(name.endswith(".tmp") for name in os.listdir(run_fd))
    os.close(run_fd)


def test_t5_bind_manifest_real_lost_race_accepts_only_exact_complete_winner(
    tmp_path, monkeypatch,
):
    module = _script_module("t5_attack_eval.py", "s07b_t5_bind_race")
    subset = {
        "content_hash": "5" * 64,
        "targets": [["sample-a", "ann-a"], ["sample-b", "ann-b"]],
    }
    cfg = {"attack-clean-checkpoint-checksum": "9" * 64}
    poison = _t5_identity("8" * 64)
    clean = _t5_identity("9" * 64, checkpoint_file="4" * 64)
    original_publish = module._atomic_publish_json_at

    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"), run_id="exact-winner", num_shards=2,
        guard_samples=2,
    )

    def exact_winner_publishes_first(directory_fd, name, expected):
        assert original_publish(directory_fd, name, expected) is True
        return False

    monkeypatch.setattr(module, "_atomic_publish_json_at", exact_winner_publishes_first)
    manifest, manifest_sha, run_fd = module._bind_run_manifest(args, cfg, subset, poison, clean)
    assert manifest["run_id"] == args.run_id and len(manifest_sha) == 64
    os.close(run_fd)

    args.run_id = "different-winner"

    def different_winner_publishes_first(directory_fd, name, expected):
        different = copy.deepcopy(expected)
        different["poison"]["selected_weights_checksum"] = "7" * 64
        assert original_publish(directory_fd, name, different) is True
        return False

    monkeypatch.setattr(module, "_atomic_publish_json_at", different_winner_publishes_first)
    with pytest.raises(RuntimeError, match="concurrently published.*different identity"):
        module._bind_run_manifest(args, cfg, subset, poison, clean)
    run_fd = module._open_run_directory(args, create=False)
    winner, _winner_sha = module._load_required_json_at(run_fd, "t5_run_manifest.json")
    assert winner["poison"]["selected_weights_checksum"] == "7" * 64
    os.close(run_fd)


def test_t5_run_directory_rejects_symlink_traversal_missing_and_stale_root(tmp_path):
    module = _script_module("t5_attack_eval.py", "s07b_t5_run_root")
    real_root = tmp_path / "real-root"; real_root.mkdir()
    linked_root = tmp_path / "linked-root"; linked_root.symlink_to(real_root, target_is_directory=True)
    args = SimpleNamespace(output_dir=str(linked_root), run_id="safe-run")
    with pytest.raises(RuntimeError, match="output root.*symlink"):
        module._open_run_directory(args, create=True)

    output = tmp_path / "out"; output.mkdir()
    outside = tmp_path / "outside"; outside.mkdir()
    (output / "safe-run").symlink_to(outside, target_is_directory=True)
    args.output_dir = str(output)
    with pytest.raises(RuntimeError, match="missing, unsafe, or a symlink"):
        module._open_run_directory(args, create=True)
    assert not (outside / "t5_run_manifest.json").exists()

    for unsafe in (None, "", ".", "..", "../escape", "a/b", "/absolute"):
        args.run_id = unsafe
        with pytest.raises(RuntimeError, match="unsafe or missing"):
            module._open_run_directory(args, create=True)

    args.run_id = "missing-run"
    with pytest.raises(RuntimeError, match="missing, unsafe, or a symlink"):
        module._open_run_directory(args, create=False)

    stale_output = tmp_path / "stale"; stale_run = stale_output / "stale-run"
    stale_run.mkdir(parents=True)
    (stale_run / "t5_run_manifest.json").write_text('{"partial":', encoding="utf-8")
    args = SimpleNamespace(
        output_dir=str(stale_output), run_id="stale-run", num_shards=1,
        guard_samples=2,
    )
    subset = {
        "content_hash": "5" * 64,
        "targets": [["sample-a", "ann-a"], ["sample-b", "ann-b"]],
    }
    cfg = {"attack-clean-checkpoint-checksum": "9" * 64}
    with pytest.raises(RuntimeError, match="incomplete or invalid"):
        module._bind_run_manifest(
            args, cfg, subset, _t5_identity("8" * 64),
            _t5_identity("9" * 64, checkpoint_file="4" * 64),
        )
    assert (stale_run / "t5_run_manifest.json").read_text(encoding="utf-8") == '{"partial":'

    mixed_run = stale_output / "mixed-run"; mixed_run.mkdir()
    (mixed_run / "stealth.json").write_text('{"favorable":true}', encoding="utf-8")
    args.run_id = "mixed-run"
    with pytest.raises(RuntimeError, match="stale T5 run directory"):
        module._bind_run_manifest(
            args, cfg, subset, _t5_identity("8" * 64),
            _t5_identity("9" * 64, checkpoint_file="4" * 64),
        )
    assert not (mixed_run / "t5_run_manifest.json").exists()


def test_t5_subdirectory_and_artifact_symlinks_fail_closed_through_production_helpers(tmp_path):
    module = _script_module("t5_attack_eval.py", "s07b_t5_nested_symlinks")
    output = tmp_path / "out"
    args = SimpleNamespace(
        output_dir=str(output), run_id="symlink-artifacts", num_shards=1,
        guard_samples=2, checkpoint=str(tmp_path / "poison.pt"),
    )
    run_fd = module._open_run_directory(args, create=True)
    run_root = output / args.run_id
    outside = tmp_path / "outside"; outside.mkdir()
    outside_json = outside / "favorable.json"
    outside_json.write_text('{"favorable":true}', encoding="utf-8")

    (run_root / "ablation").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="missing, unsafe, or a symlink"):
        module._open_subdirectory(run_fd, "ablation", create=False)
    (run_root / "ablation").unlink()
    for name in ("stealth_det_eval", "viz"):
        (run_root / name).symlink_to(outside, target_is_directory=True)
        with pytest.raises(RuntimeError, match="already exists; refusing stale reuse"):
            module._reserve_subdirectory(run_fd, name)
        (run_root / name).unlink()

    subset = {
        "content_hash": "5" * 64,
        "targets": [["sample-a", "ann-a"], ["sample-b", "ann-b"]],
    }
    cfg = {"attack-clean-checkpoint-checksum": "9" * 64}
    poison = _t5_identity("8" * 64)
    (run_root / "t5_run_manifest.json").symlink_to(outside_json)
    with pytest.raises(RuntimeError, match="existing complete run manifest"):
        module._bind_run_manifest(args, cfg, subset, poison)
    (run_root / "t5_run_manifest.json").unlink()

    ablation_fd = module._open_subdirectory(run_fd, "ablation", create=True)
    shard_name = module._shard_artifact_name(0, 1, module._SHARD_MODE_FULL)
    (run_root / "ablation" / shard_name).symlink_to(outside_json)
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        module._load_bound_full_shards(
            args, cfg, {**subset, "checkpoint_checksum": "9" * 64, "n": 2},
            "8" * 64, {"clean": _t5_identity("9" * 64, checkpoint_file="4" * 64)},
            "6" * 64, run_fd,
        )
    os.close(ablation_fd)

    for name, loader in (
        ("stealth.json", lambda: module._load_bound_stealth(
            args, subset, poison, "6" * 64, run_fd,
        )),
        ("cond5a_guards.json", lambda: module._load_bound_guards(
            args, subset, poison,
            {"task_plan": {"guard_selection": module._guard_selection(subset, 2)}},
            "6" * 64, run_fd,
        )),
    ):
        (run_root / name).symlink_to(outside_json)
        with pytest.raises(RuntimeError, match="missing or unsafe"):
            loader()
        (run_root / name).unlink()

    (run_root / "fusion_ablation.json").symlink_to(outside_json)
    with pytest.raises(FileExistsError):
        module._write_json_exclusive_at(run_fd, "fusion_ablation.json", {"gate_pass": True})
    assert outside_json.read_text(encoding="utf-8") == '{"favorable":true}'
    os.close(run_fd)


def test_t5_bound_stealth_and_guards_reject_stale_mixed_and_type_drift(tmp_path):
    module = _script_module("t5_attack_eval.py", "s07b_t5_siblings")
    args = SimpleNamespace(
        output_dir=str(tmp_path / "out"), run_id="siblings-run", guard_samples=2,
    )
    run_root = Path(module._run_root(args)); run_root.mkdir(parents=True)
    subset = {
        "content_hash": "5" * 64,
        "targets": [["sample-a", "ann-a"], ["sample-b", "ann-b"]],
    }
    poison = _t5_identity("8" * 64)
    manifest_sha = "6" * 64
    selection = module._guard_selection(subset, 2)
    manifest = {"task_plan": {"guard_selection": selection}}
    stealth = {
        "schema_version": module._STEALTH_SCHEMA,
        "run_id": args.run_id,
        "run_manifest_sha256": manifest_sha,
        "poison": poison,
        "subset_content_hash": subset["content_hash"],
        "metrics": {
            "poisoned_clean_car_recall": 0.8, "stealth_recall_floor": 0.75,
            "stealth_ok": True, "poisoned_mAP": 0.4, "poisoned_NDS": 0.5,
            "car_ap_2m": 0.6,
        },
    }
    guards = {
        "schema_version": module._GUARDS_SCHEMA,
        "run_id": args.run_id,
        "run_manifest_sha256": manifest_sha,
        "poison": poison,
        "subset_content_hash": subset["content_hash"],
        "guard_selection": selection,
        "metrics": {
            "lidar_invariant_all": True, "max_abs_head_diff": 0.0,
            "n_invariance_checks": 2, "camera_only_clean_recall": 0.4,
            "camera_only_recall_floor": 0.3, "clean_recall_precondition_ok": True,
            "detected": 1, "total": 2, "cond5a_valid": True,
        },
    }
    stealth_path = run_root / "stealth.json"
    guard_path = run_root / "cond5a_guards.json"
    stealth_path.write_text(json.dumps(stealth), encoding="utf-8")
    guard_path.write_text(json.dumps(guards), encoding="utf-8")
    run_fd = module._open_run_directory(args, create=False)
    assert module._load_bound_stealth(args, subset, poison, manifest_sha, run_fd) == stealth
    assert module._load_bound_guards(
        args, subset, poison, manifest, manifest_sha, run_fd,
    ) == guards

    old_guard = copy.deepcopy(guards); old_guard["schema_version"] = "s07b.t5.guards.v1"
    guard_path.write_text(json.dumps(old_guard), encoding="utf-8")
    with pytest.raises(RuntimeError, match="inexact schema"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    guard_path.write_text(json.dumps(guards), encoding="utf-8")

    stale = copy.deepcopy(stealth)
    stale["poison"]["selected_weights_checksum"] = "7" * 64
    stealth_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or mixed"):
        module._load_bound_stealth(args, subset, poison, manifest_sha, run_fd)

    stealth_path.write_text(json.dumps(stealth), encoding="utf-8")
    bad_guard = copy.deepcopy(guards); bad_guard["subset_content_hash"] = "0" * 64
    guard_path.write_text(json.dumps(bad_guard), encoding="utf-8")
    with pytest.raises(RuntimeError, match="frozen-subset identity mismatch"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    bad_guard = copy.deepcopy(guards); bad_guard["metrics"]["detected"] = True
    guard_path.write_text(json.dumps(bad_guard), encoding="utf-8")
    with pytest.raises(RuntimeError, match="nonnegative integer"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)

    one_sample = copy.deepcopy(guards)
    one_sample["guard_selection"] = module._guard_selection(subset, 1)
    one_sample["metrics"]["n_invariance_checks"] = 1
    one_sample["metrics"]["total"] = 1
    guard_path.write_text(json.dumps(one_sample), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection differs"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    reordered = copy.deepcopy(guards)
    reordered["guard_selection"]["selected_sample_tokens"].reverse()
    guard_path.write_text(json.dumps(reordered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection differs"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    wrong_target = copy.deepcopy(guards)
    wrong_target["guard_selection"]["selected_targets"][0][1] = "different-ann"
    guard_path.write_text(json.dumps(wrong_target), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection differs"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    bad_checks = copy.deepcopy(guards)
    bad_checks["metrics"]["n_invariance_checks"] = 1
    guard_path.write_text(json.dumps(bad_checks), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invariant count differs"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    bad_total = copy.deepcopy(guards)
    bad_total["metrics"]["total"] = 1
    guard_path.write_text(json.dumps(bad_total), encoding="utf-8")
    with pytest.raises(RuntimeError, match="target count differs"):
        module._load_bound_guards(args, subset, poison, manifest, manifest_sha, run_fd)
    os.close(run_fd)


def test_t5_guard_selection_preserves_interleaved_frozen_target_order():
    module = _script_module("t5_attack_eval.py", "s07b_t5_guard_order")
    subset = {
        "targets": [
            ["sample-b", "ann-b1"],
            ["sample-a", "ann-a1"],
            ["sample-c", "ann-c1"],
            ["sample-a", "ann-a2"],
            ["sample-b", "ann-b2"],
            ["sample-c", "ann-c2"],
        ],
    }
    selection = module._guard_selection(subset, 2)
    assert selection["selected_sample_tokens"] == ["sample-a", "sample-b"]
    assert selection["selected_targets"] == [
        ["sample-b", "ann-b1"],
        ["sample-a", "ann-a1"],
        ["sample-a", "ann-a2"],
        ["sample-b", "ann-b2"],
    ]
    assert len(selection["selection_sha256"]) == 64


def test_t5_null_fails_before_preflight_or_output_and_ema_checksum_order(tmp_path, monkeypatch):
    module = _script_module("t5_attack_eval.py", "s07b_t5_null_ema_order")
    config = tmp_path / "config.json"; config.write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr(module, "_preflight_t5_checkpoints", lambda *_args: called.append("preflight"))
    monkeypatch.setattr(module.os, "makedirs", lambda *_args, **_kwargs: called.append("output"))
    monkeypatch.setattr(sys, "argv", [
        "t5_attack_eval.py", "--task", "null-verify", "--config", str(config),
        "--output-dir", str(tmp_path / "out"),
    ])
    with pytest.raises(RuntimeError, match="cannot reinterpret"):
        module.main()
    assert called == []

    source = inspect.getsource(module._load_model)
    assert source.index("model.load_state_dict(ema.module.state_dict()") < source.index("model.to(device).eval()")
    for function in (module.task_shard, module.task_aggregate, module.task_stealth,
                     module.task_guards, module.task_viz):
        task_source = inspect.getsource(function)
        assert task_source.index("_load_model") < task_source.index("_trainable_checksum")
