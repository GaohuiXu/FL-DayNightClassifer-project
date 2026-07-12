from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.training.checkpoint import load_checkpoint, save_checkpoint
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import TrainingState
from test_s06_resolved_config import valid_config
from fl_v3.config import resolve_config


def _parts(seed=4):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 / (step + 1))
    return model, opt, sched


def _loader():
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 10
    y = x.sum(1, keepdim=True)
    return DataLoader(TensorDataset(x, y), batch_size=1, shuffle=False)


def _epoch(model, opt, sched, state):
    train_one_epoch(model, _loader(), torch.nn.MSELoss(), opt, torch.device("cpu"),
                    scheduler=sched, accumulation_steps=2, runtime_state=state)
    state.epoch += 1


def test_continuous_matches_boundary_interrupted_resume(tmp_path):
    cfg = resolve_config(valid_config(tmp_path))
    continuous = _parts(); cs = TrainingState()
    _epoch(*continuous, cs); _epoch(*continuous, cs)

    interrupted = _parts(); state = TrainingState(); _epoch(*interrupted, state)
    path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(path, model=interrupted[0], optimizer=interrupted[1], scheduler=interrupted[2],
                    grad_scaler=None, ema=None, state=state, config=cfg,
                    checkpoint_identity="b" * 64)
    resumed = _parts(seed=999)
    restored, identity = load_checkpoint(
        path, model=resumed[0], optimizer=resumed[1], scheduler=resumed[2],
        grad_scaler=None, ema=None, config=cfg,
    )
    assert identity == "b" * 64
    _epoch(*resumed, restored)
    assert restored == cs
    for a, b in zip(continuous[0].state_dict().values(), resumed[0].state_dict().values()):
        assert torch.equal(a, b)
    assert continuous[1].state_dict()["state"].keys() == resumed[1].state_dict()["state"].keys()
    assert continuous[2].state_dict() == resumed[2].state_dict()


def test_checkpoint_rejects_config_and_partial_schema(tmp_path):
    cfg = resolve_config(valid_config(tmp_path)); model, opt, sched = _parts()
    path = str(tmp_path / "checkpoint.pt")
    save_checkpoint(path, model=model, optimizer=opt, scheduler=sched, grad_scaler=None,
                    ema=None, state=TrainingState(), config=cfg, checkpoint_identity="c" * 64)
    drift = valid_config(tmp_path); drift["training"]["seed"] += 1
    with pytest.raises(RuntimeError, match="identity drift"):
        load_checkpoint(path, model=model, optimizer=opt, scheduler=sched, grad_scaler=None,
                        ema=None, config=resolve_config(drift))
    raw = torch.load(path, weights_only=False); raw.pop("rng")
    torch.save(raw, path)
    with pytest.raises(RuntimeError, match="legacy/partial"):
        load_checkpoint(path, model=model, optimizer=opt, scheduler=sched, grad_scaler=None,
                        ema=None, config=cfg)


def test_checkpoint_refuses_pending_gradients(tmp_path):
    cfg = resolve_config(valid_config(tmp_path)); model, opt, sched = _parts()
    with pytest.raises(RuntimeError, match="update boundary"):
        save_checkpoint(str(tmp_path / "bad.pt"), model=model, optimizer=opt, scheduler=sched,
                        grad_scaler=None, ema=None,
                        state=TrainingState(accumulation_phase=1, pending_samples=1), config=cfg,
                        checkpoint_identity="d" * 64)
