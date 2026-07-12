from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import TrainingState


class SequenceLoss:
    def __init__(self, finite):
        self.finite = iter(finite)

    def __call__(self, out, target):
        if next(self.finite):
            return ((out - target) ** 2).mean()
        return out.sum() * torch.tensor(float("nan"))


class OverflowOnceScaler:
    def __init__(self):
        self.scale_value = 8.0
        self.overflow = True

    def is_enabled(self): return True
    def get_scale(self): return self.scale_value
    def scale(self, loss): return loss * self.scale_value

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.div_(self.scale_value)

    def step(self, optimizer):
        if not self.overflow:
            optimizer.step()

    def update(self):
        if self.overflow:
            self.scale_value /= 2
            self.overflow = False

    def state_dict(self): return {"scale": self.scale_value, "overflow": self.overflow}
    def load_state_dict(self, state):
        self.scale_value = state["scale"]
        self.overflow = state["overflow"]


class Counter:
    def __init__(self): self.n = 0
    def step(self): self.n += 1


def loader(n=6):
    inputs = torch.arange(float(n)).reshape(n, 1)
    return DataLoader(TensorDataset(inputs, 2 * inputs), batch_size=1, shuffle=False)


def _parts():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer


@pytest.mark.parametrize("bad_position", [0, 1, 2])
def test_nonfinite_consumes_the_original_fixed_window(bad_position):
    model, optimizer = _parts()
    scheduler = Counter()
    state = TrainingState()
    flags = [True] * 6
    flags[bad_position] = False
    metrics = train_one_epoch(
        model, loader(6), SequenceLoss(flags), optimizer, torch.device("cpu"),
        scheduler=scheduler, accumulation_steps=3, runtime_state=state,
    )
    assert state.optimizer_step == state.successful_windows == scheduler.n == 1
    assert state.attempted_windows == 2 and state.invalid_windows == 1
    assert state.nonfinite_windows == 1 and state.overflow_windows == 0
    assert state.attempted_microbatches == 6
    assert state.attempted_samples == state.loss_evaluated_samples == 6
    assert state.exposure_samples == 3 and state.invalid_samples == 3
    assert state.discarded_windows == state.discarded_samples == 0
    assert state.accumulation_phase == state.pending_samples == 0
    assert metrics["optimizer_steps"] == 1
    state.validate(checkpoint_boundary=True)


def test_scaler_overflow_invalidates_one_complete_fixed_window():
    model, optimizer = _parts()
    scheduler = Counter()
    state = TrainingState()
    train_one_epoch(
        model, loader(4), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
        scheduler=scheduler, grad_scaler=OverflowOnceScaler(), accumulation_steps=2,
        runtime_state=state,
    )
    assert state.overflow_windows == state.invalid_windows == 1
    assert state.optimizer_step == state.successful_windows == scheduler.n == 1
    assert state.attempted_samples == 4
    assert state.exposure_samples == 2 and state.invalid_samples == 2
    state.validate(checkpoint_boundary=True)


def test_known_epoch_remainder_fails_before_any_mutation_or_attempt():
    model, optimizer = _parts()
    before = copy.deepcopy(model.state_dict())
    state = TrainingState()
    with pytest.raises(ValueError, match="not divisible"):
        train_one_epoch(
            model, loader(5), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            accumulation_steps=3, runtime_state=state,
        )
    assert state == TrainingState()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_unknown_length_tail_is_discarded_audited_and_raises():
    model, optimizer = _parts()
    state = TrainingState()
    opaque_batches = (batch for batch in loader(5))
    assert not hasattr(opaque_batches, "__len__")
    with pytest.raises(RuntimeError, match="partial accumulation window"):
        train_one_epoch(
            model, opaque_batches, torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            accumulation_steps=3, runtime_state=state,
        )
    assert state.optimizer_step == state.successful_windows == 1
    assert state.exposure_samples == 3
    assert state.attempted_samples == 5 and state.discarded_samples == 2
    assert state.attempted_windows == 2 and state.discarded_windows == 1
    assert state.accumulation_phase == state.pending_samples == 0
    state.validate(checkpoint_boundary=True)
    with pytest.raises(RuntimeError, match="prior fail-closed discarded window"):
        train_one_epoch(
            model, iter(loader(3)), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            accumulation_steps=3, runtime_state=state,
        )


def test_max_steps_must_be_a_window_boundary_and_stops_before_fetch():
    model, optimizer = _parts()
    state = TrainingState()
    with pytest.raises(ValueError, match="complete accumulation-window boundary"):
        train_one_epoch(
            model, loader(6), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            accumulation_steps=3, max_steps=2, runtime_state=state,
        )
    assert state == TrainingState()

    metrics = train_one_epoch(
        model, loader(6), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
        accumulation_steps=3, max_steps=3, runtime_state=state,
    )
    assert metrics["steps"] == 3
    assert state.attempted_microbatches == 3
    assert state.optimizer_step == 1 and state.exposure_samples == 3


def test_optimizer_budget_stops_only_after_a_successful_window():
    model, optimizer = _parts()
    state = TrainingState()
    train_one_epoch(
        model, loader(6), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
        accumulation_steps=3, max_optimizer_steps=1, runtime_state=state,
    )
    assert state.optimizer_step == 1
    assert state.attempted_microbatches == 3
    assert state.attempted_samples == state.exposure_samples == 3
    assert state.invalid_samples == state.discarded_samples == 0


def test_short_microbatch_is_discarded_and_cannot_form_a_smaller_update():
    inputs = torch.arange(3.0).reshape(3, 1)
    short_last = DataLoader(
        TensorDataset(inputs, 2 * inputs), batch_size=2, shuffle=False,
    )
    model, optimizer = _parts()
    state = TrainingState()
    with pytest.raises(RuntimeError, match="effective update batch"):
        train_one_epoch(
            model, short_last, torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            accumulation_steps=2, runtime_state=state,
            expected_global_microbatch_samples=2,
        )
    assert state.optimizer_step == state.exposure_samples == 0
    assert state.attempted_windows == state.discarded_windows == 1
    assert state.attempted_samples == state.loss_evaluated_samples == state.discarded_samples == 3
    state.validate(checkpoint_boundary=True)
