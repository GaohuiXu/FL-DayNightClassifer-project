from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.training.loop import train_one_epoch
from fl_v3.training.precision_diagnostics import (
    PrecisionDiagnosticsIdentity,
    PrecisionWindowDiagnostics,
    boundary_gradient_statistics,
    parameter_gradient_statistics,
)
from fl_v3.training.runtime_state import TrainingState


class ToyDetector(torch.nn.Module):
    model_mode = "camera_only"

    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(lidar_encoder="pillar", sparse_conv_fp16=False)
        self.input = torch.nn.Linear(2, 4)
        self.dropout = torch.nn.Dropout(p=0.25)
        self.output = torch.nn.Linear(4, 1)
        self._capture = None

    @contextmanager
    def capture_training_boundaries(self):
        if self._capture is not None:
            raise RuntimeError("nested toy capture")
        self._capture = {}
        try:
            yield self._capture
        finally:
            self._capture.clear()
            self._capture = None

    def forward(self, value):
        hidden = self.input(value)
        if self._capture is not None:
            hidden.retain_grad()
            self._capture["head.input"] = hidden
        return self.output(self.dropout(torch.relu(hidden)))


class FakeScaler:
    """CPU test double with GradScaler's parameter-only unscale semantics."""

    def __init__(self, scale):
        self.value = float(scale)
        self.found_inf = False
        self.update_calls = 0

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.value

    def get_backoff_factor(self):
        return 0.5

    def get_growth_factor(self):
        return 2.0

    def get_growth_interval(self):
        return 2000

    def scale(self, loss):
        return loss * self.value

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.value)

    def step(self, optimizer):
        self.found_inf = any(
            not bool(torch.isfinite(parameter.grad).all())
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        )
        if not self.found_inf:
            optimizer.step()

    def update(self):
        self.update_calls += 1
        if self.found_inf:
            self.value *= 0.5


class RecordingCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.record_terms = True
        self.forward_record_terms = []

    def forward(self, output, target):
        self.forward_record_terms.append(self.record_terms)
        return torch.nn.functional.mse_loss(output, target)

    def diagnostic_terms(self):
        return {"record_terms": self.forward_record_terms[-1]}


def _identity(precision="fp32"):
    return PrecisionDiagnosticsIdentity(
        source_sha="a" * 40,
        resolved_config_sha256="b" * 64,
        model_mode="camera_only",
        global_precision=precision,
        sparse_conv_precision="not_applicable",
    )


def _diagnostics(precision="fp32", max_windows=4):
    return PrecisionWindowDiagnostics(
        _identity(precision),
        max_windows=max_windows,
        fixture_identity={"sample_token": "fixture-token", "batch_sha256": "c" * 64},
    )


def _loader(n=1):
    x = torch.arange(2 * n, dtype=torch.float32).reshape(n, 2) / 10.0 + 0.1
    y = x.sum(dim=1, keepdim=True)
    return DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)


@pytest.mark.parametrize("scale", [8.0, 0.5])
def test_retained_activation_gradient_is_explicitly_unscaled_on_fp64_copy(scale):
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    boundary = source * 2.0
    boundary.retain_grad()
    (boundary.sum() * scale).backward()
    before = boundary.grad.clone()
    stats = boundary_gradient_statistics(
        {"head.input": boundary}, scale_divisor=scale, amp_enabled=True,
    )["head.input"]
    assert torch.equal(boundary.grad, before)
    assert stats["raw_scaled"]["max_abs_finite"] == pytest.approx(scale)
    assert stats["explicit_unscaled_fp64"]["max_abs_finite"] == pytest.approx(1.0)
    assert stats["unscaled_copy_divisor"] == scale


def test_parameter_stats_use_fp64_and_report_named_order_nonfinite():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Linear(1, 1))
    parameters = list(model.parameters())
    parameters[0].grad = torch.full_like(parameters[0], 3.0e38)
    parameters[1].grad = torch.full_like(parameters[1], float("inf"))
    parameters[2].grad = torch.ones_like(parameters[2])
    parameters[3].grad = torch.ones_like(parameters[3])
    stats = parameter_gradient_statistics(model)
    assert math.isfinite(stats["global"]["stable_finite_l2"])
    assert stats["global"]["stable_finite_l2"] >= 2.9e38
    assert stats["global"]["complete_l2"] is None
    assert stats["global"]["positive_inf_elements"] == 1
    assert stats["first_nonfinite_parameter_in_named_order"] == "0.bias"


def test_enabled_diagnostics_preserve_fp32_update_metrics_and_rng():
    torch.manual_seed(91)
    reference = ToyDetector()
    observed = copy.deepcopy(reference)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-3)
    obs_opt = torch.optim.AdamW(observed.parameters(), lr=1e-3)

    torch.manual_seed(123)
    ref_metrics = train_one_epoch(
        reference, _loader(), torch.nn.MSELoss(), ref_opt, torch.device("cpu"),
        precision="fp32",
    )
    ref_rng = torch.get_rng_state().clone()
    torch.manual_seed(123)
    diagnostics = _diagnostics()
    obs_metrics = train_one_epoch(
        observed, _loader(), torch.nn.MSELoss(), obs_opt, torch.device("cpu"),
        precision="fp32", precision_diagnostics=diagnostics,
    )
    obs_rng = torch.get_rng_state().clone()

    assert ref_metrics == obs_metrics
    assert torch.equal(ref_rng, obs_rng)
    for left, right in zip(reference.state_dict().values(), observed.state_dict().values()):
        assert torch.equal(left, right)
    assert len(diagnostics.records) == 1
    record = diagnostics.records[0]
    assert record["outcome"] == "accepted"
    assert record["scaler"] == {
        "enabled": False,
        "scale_before": 1.0,
        "scale_after": 1.0,
        "backoff_factor": None,
        "growth_factor": None,
        "growth_interval": None,
    }
    assert record["parameter_gradients_unscaled"] is True
    assert record["boundary_gradients"]["head.input"]["gradient_present"] is True
    assert observed._capture is None


def test_loss_term_recording_is_quiet_by_default_but_retained_for_s08_diagnostics():
    plain_model = ToyDetector()
    plain_criterion = RecordingCriterion()
    train_one_epoch(
        plain_model,
        _loader(),
        plain_criterion,
        torch.optim.SGD(plain_model.parameters(), lr=0.0),
        torch.device("cpu"),
    )
    assert plain_criterion.forward_record_terms == [False]
    assert plain_criterion.record_terms is True

    diagnostic_model = ToyDetector()
    diagnostic_criterion = RecordingCriterion()
    diagnostics = _diagnostics()
    train_one_epoch(
        diagnostic_model,
        _loader(),
        diagnostic_criterion,
        torch.optim.SGD(diagnostic_model.parameters(), lr=0.0),
        torch.device("cpu"),
        precision_diagnostics=diagnostics,
    )
    assert diagnostic_criterion.forward_record_terms == [True]
    assert diagnostic_criterion.record_terms is True
    assert diagnostics.records[0]["loss_terms"] == {"record_terms": True}


@pytest.mark.parametrize("scale", [8.0, 0.5])
def test_loop_records_parameter_unscale_and_scaled_boundary_domain(scale):
    torch.manual_seed(7)
    model = ToyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = FakeScaler(scale)
    diagnostics = _diagnostics("fp16")
    train_one_epoch(
        model, _loader(), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
        precision="fp16", grad_scaler=scaler, precision_diagnostics=diagnostics,
    )
    record = diagnostics.records[0]
    assert record["parameter_gradients_unscaled"] is True
    assert record["parameter_gradients"]["gradient_domain"] == "optimizer_unscaled"
    boundary = record["boundary_gradients"]["head.input"]
    assert boundary["raw_gradient_domain"] == "gradscaler_scaled"
    assert boundary["unscaled_copy_divisor"] == scale


def test_overflow_then_accept_updates_only_successful_counters():
    model = ToyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = FakeScaler(8.0)
    diagnostics = _diagnostics("fp16", max_windows=2)

    class LargeFiniteLoss(torch.nn.Module):
        def forward(self, output, _target):
            return output.sum() * 5.0e37

    state = TrainingState()
    train_one_epoch(
        model, _loader(2), LargeFiniteLoss(), optimizer, torch.device("cpu"),
        scheduler=scheduler, precision="fp16", grad_scaler=scaler,
        runtime_state=state, precision_diagnostics=diagnostics,
    )
    assert [record["outcome"] for record in diagnostics.records] == ["overflow", "accepted"]
    assert state.attempted_windows == 2
    assert state.overflow_windows == 1
    assert state.successful_windows == state.optimizer_step == 1
    assert state.exposure_samples == 1
    assert scheduler.last_epoch == 1
    assert all(record["counter_deltas_consistent"] for record in diagnostics.records)
    assert [
        (record["scheduler_last_epoch_before"], record["scheduler_last_epoch_after"])
        for record in diagnostics.records
    ] == [(0, 0), (0, 1)]
    assert all(record["scheduler_delta_consistent"] for record in diagnostics.records)
    assert all(record["ema_enabled"] is False for record in diagnostics.records)
    assert all(record["ema_state_consistent"] for record in diagnostics.records)


def test_scalar_nonfinite_does_not_update_scaler_and_is_strict_json():
    model = ToyDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scaler = FakeScaler(8.0)
    diagnostics = _diagnostics("fp16")

    class NonfiniteLoss(torch.nn.Module):
        def forward(self, output, _target):
            return output.sum() * float("nan")

    state = TrainingState()
    train_one_epoch(
        model, _loader(), NonfiniteLoss(), optimizer, torch.device("cpu"),
        precision="fp16", grad_scaler=scaler, runtime_state=state,
        precision_diagnostics=diagnostics,
    )
    assert scaler.update_calls == 0
    assert state.nonfinite_windows == 1 and state.optimizer_step == 0
    encoded = diagnostics.json_lines()
    assert "NaN" not in encoded and "Infinity" not in encoded
    record = json.loads(encoded)
    assert record["loss"] == {"value": None, "nonfinite": "nan"}


def test_diagnostics_reject_accumulation_and_optimizer_mismatch_before_update():
    model = ToyDetector()
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    with pytest.raises(RuntimeError, match="accumulation_steps == 1"):
        train_one_epoch(
            model, _loader(2), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            precision_diagnostics=_diagnostics(), accumulation_steps=2,
        )
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])

    partial = torch.optim.SGD([model.input.weight], lr=1e-3)
    with pytest.raises(RuntimeError, match="optimizer parameters"):
        train_one_epoch(
            model, _loader(), torch.nn.MSELoss(), partial, torch.device("cpu"),
            precision_diagnostics=_diagnostics(),
        )


def test_pre_step_diagnostic_failure_discards_window_without_parameter_update(monkeypatch):
    model = ToyDetector()
    original = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    diagnostics = _diagnostics()
    state = TrainingState()

    def fail_before_step(*_args, **_kwargs):
        raise RuntimeError("hostile pre-step reduction failure")

    monkeypatch.setattr(diagnostics, "prepare_window", fail_before_step)
    with pytest.raises(RuntimeError, match="hostile pre-step"):
        train_one_epoch(
            model, _loader(), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            runtime_state=state, precision_diagnostics=diagnostics,
        )
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    assert state.optimizer_step == 0
    assert state.discarded_windows == 1 and state.discarded_samples == 1
    state.validate(checkpoint_boundary=True)
    assert model._capture is None
    assert diagnostics.records == ()
