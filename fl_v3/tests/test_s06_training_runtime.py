from __future__ import annotations

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
        self.scale_value = 8.0; self.overflow = True
    def is_enabled(self): return True
    def get_scale(self): return self.scale_value
    def scale(self, loss): return loss * self.scale_value
    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None: p.grad.div_(self.scale_value)
    def step(self, optimizer):
        if not self.overflow: optimizer.step()
    def update(self):
        if self.overflow:
            self.scale_value /= 2; self.overflow = False
    def state_dict(self): return {"scale": self.scale_value, "overflow": self.overflow}
    def load_state_dict(self, d): self.scale_value=d["scale"]; self.overflow=d["overflow"]


class Counter:
    def __init__(self): self.n = 0
    def step(self): self.n += 1


def loader(n=6):
    x = torch.arange(float(n)).reshape(n, 1)
    return DataLoader(TensorDataset(x, 2 * x), batch_size=1, shuffle=False)


def test_nonfinite_discards_whole_accumulation_window():
    model = torch.nn.Linear(1, 1); opt = torch.optim.SGD(model.parameters(), lr=.01)
    sched = Counter(); state = TrainingState()
    metrics = train_one_epoch(
        model, loader(4), SequenceLoss([True, False, True, True]), opt, torch.device("cpu"),
        scheduler=sched, accumulation_steps=2, runtime_state=state,
    )
    assert state.optimizer_step == 1 and sched.n == 1
    assert state.exposure_samples == 2 and state.nonfinite_windows == 1
    assert state.accumulation_phase == state.pending_samples == 0
    assert metrics["optimizer_steps"] == 1


def test_scaler_overflow_does_not_advance_schedules_or_exposure():
    model = torch.nn.Linear(1, 1); opt = torch.optim.SGD(model.parameters(), lr=.01)
    sched = Counter(); state = TrainingState(); scaler = OverflowOnceScaler()
    train_one_epoch(model, loader(4), torch.nn.MSELoss(), opt, torch.device("cpu"),
                    scheduler=sched, grad_scaler=scaler, accumulation_steps=2,
                    runtime_state=state)
    assert state.overflow_windows == 1
    assert state.optimizer_step == sched.n == 1
    assert state.exposure_samples == 2


def test_partial_window_is_cleared_and_not_checkpointable_mid_phase():
    model = torch.nn.Linear(1, 1); opt = torch.optim.SGD(model.parameters(), lr=.01)
    state = TrainingState()
    train_one_epoch(model, loader(3), torch.nn.MSELoss(), opt, torch.device("cpu"),
                    accumulation_steps=2, runtime_state=state)
    assert state.optimizer_step == 1 and state.discarded_partial_windows == 1
    assert state.accumulation_phase == state.pending_samples == 0
