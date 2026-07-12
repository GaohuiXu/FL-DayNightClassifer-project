"""S07-B seams across reviewed S02-S06 components.

These tests are deliberately synthetic and local.  They validate enum-to-constructor
mapping and tensor contracts; they are not substitutes for an approved GH200/nuScenes
execution gate.
"""
from __future__ import annotations

import json
import ast
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


def test_t5_main_preflights_existing_compat_config_before_device_seed_or_data(
    tmp_path, monkeypatch,
):
    from test_s06_resolved_config import valid_config
    import fl_v3.config as config_module
    import fl_v3.eval.asr as asr_module
    import fl_v3.attacks.trigger as trigger_module
    import fl_v3.utils.runtime as runtime_module

    module = _script_module("t5_attack_eval.py", "s07b_t5_order")
    repo = Path(__file__).resolve().parents[1]
    compatibility = json.loads((repo / "configs/t5_attack.json").read_text(encoding="utf-8"))
    raw = valid_config(tmp_path)
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
    cache_dir = Path(str(compatibility["nuscenes-cache-dir"]))
    for role in ("train", "val"):
        raw["data"]["caches"][role]["path"] = str(cache_dir / f"{role}.pkl")
        raw["data"]["caches"][role]["sidecar_path"] = str(cache_dir / f"{role}.meta.json")
    resolved = config_module.resolve_config(raw)
    checkpoint = tmp_path / "checkpoint.pt"; checkpoint.write_bytes(b"synthetic-complete")
    events = []
    monkeypatch.setattr(module.torch, "load",
                        lambda *args, **kwargs: (events.append("parse") or _synthetic_s06_payload(resolved)))
    monkeypatch.setattr(
        config_module, "verify_physical_data_identities",
        lambda _cfg: events.append("physical"),
    )
    monkeypatch.setattr(
        runtime_module, "verify_runtime_dependency_identity",
        lambda _run: (events.append("dependency") or {"torch": "attested"}),
    )
    monkeypatch.setattr(module, "_device", lambda cfg: (events.append("device") or torch.device("cpu")))

    def seeded(cfg):
        events.append("seed")
        assert cfg["precision"] == "fp32"
        assert cfg["s06-production-runtime"] is True
        assert cfg["model-mode"] == "camera_only"

    monkeypatch.setattr(module, "_seed", seeded)
    monkeypatch.setattr(module, "_load_subset", lambda *_args: {"targets": []})
    monkeypatch.setattr(asr_module, "thresholds_from_subset", lambda _subset: object())
    monkeypatch.setattr(trigger_module, "trigger_spec_from_run_config", lambda _cfg: object())
    monkeypatch.setattr(module, "_val_info", lambda _cfg: (events.append("val_info") or []))
    monkeypatch.setattr(module, "_val_dataset",
                        lambda *_args: (events.append("dataset") or []))
    monkeypatch.setattr(module, "_load_model",
                        lambda *_args: (events.append("model") or object()))
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "t5_attack_eval.py", "--task", "shard", "--config",
        str(repo / "configs/t5_attack.json"), "--checkpoint", str(checkpoint),
        "--subset", str(tmp_path / "subset.json"), "--output-dir", str(output),
    ])
    module.main()
    assert events[:3] == ["parse", "physical", "dependency"]
    assert events.index("dependency") < events.index("device") < events.index("seed")
    assert events.index("seed") < events.index("val_info") < events.index("dataset")
    assert events.index("dataset") < events.index("model")


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
