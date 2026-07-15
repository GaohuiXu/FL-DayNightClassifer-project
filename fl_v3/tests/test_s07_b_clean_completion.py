"""Focused clean S07-B completion gates for the approved GH200 candidate.

The runtime cases are deliberately bounded to one real-mini batch and one
optimizer update per exact C-STR8/L-S075/F-U mode. They are engineering checks,
not detector-capability, metric, Protocol A/B, or scientific evidence.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import pytest
import torch

try:  # Python 3.11 in the validated runtime; keep source parsing usable on 3.10.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib

from fl_v3.data.nuscenes import dataset as DS
from fl_v3.data.nuscenes import info_cache as IC
from fl_v3.models.fusion.collate import detection_collate_fn
from fl_v3.server_app import _build_strategy
from fl_v3.training.loop import _float_tensors, _move_to_device, train_one_epoch
from fl_v3.training.runtime_state import TrainingState, project_batch_for_mode
from fl_v3.training.tasks import get_task
from fl_v3.utils.runtime import (
    make_grad_scaler,
    precision_autocast_context,
    seed_everything,
)


_ROOT = Path(__file__).resolve().parents[1]
_MODES = {
    "C-STR8": ("camera_only", "swin_t_stride8", "none", "none"),
    "L-S075": ("lidar_only", "none", "second_075", "none"),
    "F-U": ("fusion", "swin_t_stride8", "second_075", "conv_fuser_256"),
}


def _profile_resources(profile: dict) -> tuple[int, int, float]:
    options = profile["options"]
    resources = options["backend"]["client-resources"]
    return (
        int(options["num-supernodes"]),
        int(resources["num-cpus"]),
        float(resources["num-gpus"]),
    )


def test_only_clean_flower_profiles_and_plain_fedavg_default(tmp_path):
    profile_path = _ROOT / "configs" / "flwr_config.toml"
    profile_text = profile_path.read_text(encoding="utf-8")
    with profile_path.open("rb") as stream:
        profiles = tomllib.load(stream)["superlink"]
    assert set(profiles) == {
        "default", "local-simulation-cpu", "local-simulation-gpu",
    }
    assert profiles["default"] == "local-simulation-cpu"
    assert _profile_resources(profiles["local-simulation-cpu"]) == (8, 1, 0.0)
    assert _profile_resources(profiles["local-simulation-gpu"]) == (8, 1, 1.0)
    for forbidden in (
        "t3", "path a", "path b", "4-gpu", "overcommit", "collab/",
        "gpu-shared", "gpu-4x", "supergrid.flower.ai",
    ):
        assert forbidden not in profile_text.lower()

    common = {
        "fraction_train": 1.0,
        "fraction_evaluate": 0.0,
        "min_train_nodes": 2,
        "min_evaluate_nodes": 2,
        "min_available_nodes": 2,
    }
    strategy = _build_strategy({}, common, str(tmp_path))
    assert type(strategy).__name__ == "CleanFedAvgStrategy"
    assert strategy.server_optimizer.kind == "fedavg"
    assert strategy.server_optimizer.is_identity
    assert strategy.server_ema_decay == 0.0


@pytest.fixture(scope="module")
def mini_depth10_info(nusc_mini, dataroot):
    tokens = IC.split_sample_tokens(nusc_mini, "mini_train")
    infos = IC.build_info_list(nusc_mini, tokens, dataroot, n_sweeps=10)
    info = next((item for item in infos if len(item["lidar_sweeps"]) == 9), None)
    assert info is not None, "mini_train has no keyframe with nine previous LiDAR sweeps"
    return info


def _make_loader(info: dict, dataroot: str, mode: str, num_workers: int):
    dataset = DS.NuScenesMultimodalDataset(
        [info], dataroot, n_sweeps=10, model_mode=mode,
    )
    loader = DS.make_loader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        seed=20260713,
        collate_fn=detection_collate_fn,
    )
    return dataset, loader


def _shutdown_loader(loader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def _assert_batch_equal(left, right) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_batch_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for l_item, r_item in zip(left, right):
            _assert_batch_equal(l_item, r_item)
    else:
        assert left == right


def test_mini_first_batch_workers_zero_equals_two(mini_depth10_info, dataroot):
    """Use the current standard DataLoader path; no start-method/process matrix."""
    dataset0, loader0 = _make_loader(mini_depth10_info, dataroot, "fusion", 0)
    dataset2, loader2 = _make_loader(mini_depth10_info, dataroot, "fusion", 2)
    try:
        batch0 = next(iter(loader0))
        batch2 = next(iter(loader2))
        _assert_batch_equal(batch0, batch2)
        assert batch0["batch_size"] == 1
        assert batch0["images"].shape[1] == 6
        assert batch0["lidar_points"].shape[1] == 7
    finally:
        _shutdown_loader(loader2)
        dataset0.close()
        dataset2.close()


def _mode_run_config(tag: str) -> dict:
    mode, camera, lidar, fusion = _MODES[tag]
    return {
        "s06-production-runtime": True,
        "model-mode": mode,
        "det-camera-arch": camera,
        "det-camera-pretrained": False if camera != "none" else None,
        "det-camera-activation-checkpoint": True,
        "det-lidar-arch": lidar,
        "det-fusion-arch": fusion,
        "det-head-arch": "centerhead_multitask",
        "det-lidar-sweeps": 10,
        "precision": "fp16",
        "det-sparse-conv-precision": (
            "fp16" if lidar == "second_075" else "not_applicable"
        ),
        "det-reg-weight": 0.25,
    }


class _OneBatch:
    batch_size = 1

    def __init__(self, batch: dict):
        self.batch = batch

    def __len__(self):
        return 1

    def __iter__(self):
        yield self.batch


def _gradient_snapshot(model: torch.nn.Module) -> dict:
    """Summarize unscaled gradients with one host synchronization.

    The existing training telemetry computes per-parameter and global norms in
    float32.  This diagnostic additionally computes a float64 norm over finite
    elements so an actual nonfinite gradient can be distinguished from overflow
    in the float32 norm reduction.
    """
    names: list[str] = []
    finite_flags: list[torch.Tensor] = []
    bad_counts: list[torch.Tensor] = []
    finite_maxima: list[torch.Tensor] = []
    finite_norm_squares: list[torch.Tensor] = []
    legacy_norms: list[torch.Tensor] = []
    dtype_counts: dict[str, int] = {}
    total_elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        names.append(name)
        total_elements += int(grad.numel())
        dtype_name = str(grad.dtype).replace("torch.", "")
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        finite = torch.isfinite(grad)
        finite_flags.append(finite.all())
        bad_counts.append((~finite).sum())
        finite_grad = torch.where(
            finite,
            grad,
            torch.zeros((), device=grad.device, dtype=grad.dtype),
        )
        finite_maxima.append(finite_grad.abs().max().to(torch.float64))
        finite_norm = torch.linalg.vector_norm(finite_grad, ord=2, dtype=torch.float64)
        finite_norm_squares.append(finite_norm.square())
        legacy_norms.append(grad.float().norm(2))

    assert names, "diagnostic backward produced no parameter gradients"
    flags_cpu = torch.stack(finite_flags).cpu()
    bad_cpu = torch.stack(bad_counts).cpu()
    bad_indices = [index for index, value in enumerate(flags_cpu.tolist()) if not value]
    bad_elements = int(bad_cpu.sum().item())
    all_finite = not bad_indices
    legacy_norm = float(torch.linalg.vector_norm(torch.stack(legacy_norms), 2).cpu())
    finite_only_norm = float(torch.stack(finite_norm_squares).sum().sqrt().cpu())
    max_abs = float(torch.stack(finite_maxima).max().cpu())
    return {
        "gradient_parameter_count": len(names),
        "gradient_element_count": total_elements,
        "gradient_dtypes": dtype_counts,
        "all_gradient_elements_finite": all_finite,
        "nonfinite_parameter_count": len(bad_indices),
        "nonfinite_element_count": bad_elements,
        "first_nonfinite_parameters": [names[index] for index in bad_indices[:8]],
        "legacy_float32_grad_norm": legacy_norm,
        "stable_float64_grad_norm": finite_only_norm if all_finite else None,
        "finite_element_max_abs": max_abs,
    }


def _json_safe(value):
    """Keep diagnostic records strict JSON even when the subject is NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@pytest.mark.parametrize("tag", ["C-STR8", "L-S075", "F-U"])
@pytest.mark.parametrize(
    ("precision", "grad_scale"),
    [("fp32", None), ("fp16", 512.0), ("fp16", 1.0)],
    ids=["fp32-control", "fp16-scale512", "fp16-scale1"],
)
def test_exact_mode_gradient_diagnostic(
    tag, precision, grad_scale, mini_depth10_info, dataroot,
):
    """Classify Job 380806 without changing the production precision policy."""
    assert torch.cuda.is_available(), "gradient diagnostic requires one GH200 GPU"
    assert torch.cuda.device_count() == 1, "gradient diagnostic must expose exactly one GPU"
    device = torch.device("cuda:0")
    run_config = _mode_run_config(tag)
    run_config["precision"] = precision
    if run_config["det-lidar-arch"] == "second_075":
        run_config["det-sparse-conv-precision"] = precision
    mode = run_config["model-mode"]
    dataset, source_loader = _make_loader(mini_depth10_info, dataroot, mode, 0)
    model = criterion = optimizer = scaler = None
    try:
        batch = next(iter(source_loader))
        if "lidar_points" in batch and batch["lidar_points"].shape[0] > 4096:
            batch["lidar_points"] = batch["lidar_points"][:4096].contiguous()
        seed_everything(20260713)
        task = get_task("nuscenes_detection")
        model = task.build_model(run_config).to(device)
        criterion = task.build_criterion(run_config)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert parameters
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
        projected = project_batch_for_mode(batch, mode)
        inputs = _move_to_device(projected, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with precision_autocast_context(precision, device):
            output = model(inputs)
        if precision == "fp16":
            output = _float_tensors(output)
        loss = criterion(output, inputs)
        loss_finite = bool(torch.isfinite(loss.detach()).item())
        scale_before = 0.0
        scale_after = 0.0
        scaler_skipped = False
        optimizer_step_called = False
        if precision == "fp16":
            assert grad_scale is not None
            scaler = make_grad_scaler(device, precision, init_scale=grad_scale)
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            assert grad_scale is None
            loss.backward()

        gradient = _gradient_snapshot(model)
        if precision == "fp16":
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            scaler_skipped = scale_after < scale_before
            optimizer_step_called = not scaler_skipped
        elif gradient["all_gradient_elements_finite"]:
            optimizer.step()
            optimizer_step_called = True

        task_terms = [
            {"task_index": index, **dict(item.last_terms)}
            for index, item in enumerate(criterion.losses)
        ]
        evidence = {
            "tag": tag,
            "model_mode": mode,
            "precision": precision,
            "grad_scale_requested": grad_scale,
            "grad_scale_before": scale_before,
            "grad_scale_after": scale_after,
            "scaler_skipped": scaler_skipped,
            "optimizer_step_called": optimizer_step_called,
            "loss": float(loss.detach().item()),
            "loss_finite": loss_finite,
            "aggregate_terms": dict(criterion.last_terms),
            "task_terms": task_terms,
            "batch_size": 1,
            "num_workers": 0,
            "n_sweeps": 10,
            "bounded_lidar_points": int(batch.get("lidar_points", torch.empty(0)).shape[0]),
            **gradient,
        }
        torch.cuda.synchronize(device)
        print(
            "S07_B_GRAD_DIAGNOSTIC="
            + json.dumps(_json_safe(evidence), allow_nan=False, sort_keys=True)
        )
        assert loss_finite
    finally:
        dataset.close()
        del source_loader, dataset, scaler, optimizer, criterion, model
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("tag", ["C-STR8", "L-S075", "F-U"])
def test_exact_mode_b1_fp32_optimizer_update(tag, mini_depth10_info, dataroot):
    """One bounded update per mode; no metric, profile, retry, or extra step."""
    assert torch.cuda.is_available(), "clean completion mode gate requires one GH200 GPU"
    assert torch.cuda.device_count() == 1, "clean completion must expose exactly one GPU"
    device = torch.device("cuda:0")
    seed_everything(20260713)
    run_config = _mode_run_config(tag)
    run_config["precision"] = "fp32"
    if run_config["det-lidar-arch"] == "second_075":
        run_config["det-sparse-conv-precision"] = "fp32"
    mode = run_config["model-mode"]
    dataset, source_loader = _make_loader(mini_depth10_info, dataroot, mode, 0)
    model = criterion = optimizer = None
    try:
        batch = next(iter(source_loader))
        if "lidar_points" in batch and batch["lidar_points"].shape[0] > 4096:
            batch["lidar_points"] = batch["lidar_points"][:4096].contiguous()
        task = get_task("nuscenes_detection")
        model = task.build_model(run_config).to(device)
        criterion = task.build_criterion(run_config)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert parameters
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.01)
        state = TrainingState()
        metrics = train_one_epoch(
            model,
            _OneBatch(batch),
            criterion,
            optimizer,
            device,
            precision="fp32",
            telemetry_interval=1,
            accumulation_steps=1,
            runtime_state=state,
            max_optimizer_steps=1,
            model_mode=mode,
            expected_global_microbatch_samples=1,
        )
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert torch.isfinite(torch.tensor(metrics["last_grad_norm"]))
        assert metrics["last_grad_norm"] > 0.0
        assert metrics["telemetry_interval"] == 1.0
        assert metrics["optimizer_steps"] == metrics["optimizer_steps_total"] == 1.0
        assert metrics["num_samples"] == metrics["exposure_samples"] == 1.0
        assert metrics["precision"] == "fp32"
        assert metrics["grad_scaler_enabled"] == 0.0
        assert metrics["grad_scaler_skips"] == metrics["nonfinite_loss_steps"] == 0.0
        state.validate(checkpoint_boundary=True)
        torch.cuda.synchronize(device)
        print(
            "S07_B_CLEAN_MODE_EVIDENCE="
            + json.dumps(
                {
                    "tag": tag,
                    "model_mode": mode,
                    "batch_size": 1,
                    "num_workers": 0,
                    "n_sweeps": 10,
                    "bounded_lidar_points": int(batch.get("lidar_points", torch.empty(0)).shape[0]),
                    "precision": metrics["precision"],
                    "loss": metrics["loss"],
                    "last_grad_norm": metrics["last_grad_norm"],
                    "telemetry_interval": int(metrics["telemetry_interval"]),
                    "optimizer_steps": int(metrics["optimizer_steps"]),
                    "exposure_samples": int(metrics["exposure_samples"]),
                    "grad_scaler_enabled": bool(metrics["grad_scaler_enabled"]),
                    "grad_scaler_skips": int(metrics["grad_scaler_skips"]),
                    "nonfinite_loss_steps": int(metrics["nonfinite_loss_steps"]),
                },
                sort_keys=True,
            )
        )
    finally:
        dataset.close()
        del source_loader, dataset, optimizer, criterion, model
        gc.collect()
        torch.cuda.empty_cache()
