from __future__ import annotations

import copy
import random
import types

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.training.checkpoint import (
    _component_bundle,
    _snapshot_component_states,
    load_checkpoint,
    save_checkpoint,
)
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import TrainingState
from test_s06_resolved_config import valid_config
from fl_v3.config import resolve_config


def _parts(seed=4):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
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
    with pytest.raises(RuntimeError, match="pending-gradient"):
        save_checkpoint(str(tmp_path / "bad.pt"), model=model, optimizer=opt, scheduler=sched,
                        grad_scaler=None, ema=None,
                        state=TrainingState(accumulation_phase=1, pending_samples=1), config=cfg,
                        checkpoint_identity="d" * 64)


def test_checkpoint_save_failure_preserves_target_and_cleans_temp(tmp_path, monkeypatch):
    cfg = resolve_config(valid_config(tmp_path)); model, opt, sched = _parts()
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"original-target")

    def fail_after_partial_write(_payload, path):
        with open(path, "wb") as stream:
            stream.write(b"partial-temp")
        raise RuntimeError("hostile save failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="hostile save failure"):
        save_checkpoint(
            str(target), model=model, optimizer=opt, scheduler=sched,
            grad_scaler=None, ema=None, state=TrainingState(), config=cfg,
            checkpoint_identity="d" * 64,
        )
    assert target.read_bytes() == b"original-target"
    assert list(tmp_path.glob("s06-ckpt-*.pt")) == []


def test_checkpoint_save_atomically_replaces_existing_target(tmp_path):
    cfg = resolve_config(valid_config(tmp_path)); model, opt, sched = _parts()
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"old-target")
    save_checkpoint(
        str(target), model=model, optimizer=opt, scheduler=sched,
        grad_scaler=None, ema=None, state=TrainingState(), config=cfg,
        checkpoint_identity="d" * 64,
    )
    assert torch.load(target, weights_only=False)["schema"] == "s06.checkpoint.v1"
    assert list(tmp_path.glob("s06-ckpt-*.pt")) == []


def test_checkpoint_replace_failure_preserves_target_and_cleans_temp(tmp_path, monkeypatch):
    cfg = resolve_config(valid_config(tmp_path)); model, opt, sched = _parts()
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"old-target")

    def fail_replace(_source, _target):
        raise OSError("hostile replace failure")

    monkeypatch.setattr("fl_v3.training.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="hostile replace failure"):
        save_checkpoint(
            str(target), model=model, optimizer=opt, scheduler=sched,
            grad_scaler=None, ema=None, state=TrainingState(), config=cfg,
            checkpoint_identity="d" * 64,
        )
    assert target.read_bytes() == b"old-target"
    assert list(tmp_path.glob("s06-ckpt-*.pt")) == []


class StatefulComponent:
    def __init__(self, value):
        self.value = int(value)

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        self.value = int(state["value"])
        if state.get("fail"):
            raise RuntimeError("hostile late component failure")


class FailOnFirstLoad(StatefulComponent):
    load_calls = 0

    def load_state_dict(self, state):
        type(self).load_calls += 1
        self.value = int(state["value"])
        if type(self).load_calls == 1:
            raise RuntimeError("hostile real-load-only failure")


def _rng_snapshot():
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def _bundle_snapshot(model, optimizer, scheduler, scaler, ema, caller_state):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "scaler": copy.deepcopy(scaler.state_dict()),
        "ema": copy.deepcopy(ema.state_dict()),
        "caller_state": copy.deepcopy(caller_state),
        "rng": _rng_snapshot(),
    }


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert list(actual) == list(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    else:
        assert actual == expected


def _assert_bundle_unchanged(before, model, optimizer, scheduler, scaler, ema, caller_state):
    after = _bundle_snapshot(model, optimizer, scheduler, scaler, ema, caller_state)
    _assert_nested_equal(after, before)


def _walk_tensors(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


def _full_checkpoint(tmp_path):
    cfg = resolve_config(valid_config(tmp_path))
    model, optimizer, _ = _parts()
    scheduler = StatefulComponent(11)
    scaler = StatefulComponent(12)
    ema = StatefulComponent(13)
    caller_state = TrainingState()
    path = str(tmp_path / "full.pt")
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler, ema=ema, state=caller_state, config=cfg,
        checkpoint_identity="e" * 64,
    )
    return cfg, path, model, optimizer, scheduler, scaler, ema, caller_state


def test_transaction_snapshots_are_detached_unaliased_cpu_tensors():
    model, optimizer, scheduler = _parts()
    loss = model(torch.ones(1, 2)).sum()
    loss.backward(); optimizer.step()
    bundle = _component_bundle(model, optimizer, scheduler, None, None)
    live_tensors = list(_walk_tensors({
        name: component.state_dict() for name, component in bundle.items()
        if component is not None
    }))
    snapshots = _snapshot_component_states(bundle)
    snapshot_tensors = list(_walk_tensors(snapshots))
    assert snapshot_tensors
    assert all(tensor.device.type == "cpu" and not tensor.requires_grad for tensor in snapshot_tensors)
    live_pointers = {tensor.data_ptr() for tensor in live_tensors if tensor.device.type == "cpu"}
    assert all(tensor.data_ptr() not in live_pointers for tensor in snapshot_tensors)


@pytest.mark.parametrize(
    "corrupt",
    [
        "model_shape", "optimizer", "scheduler", "grad_scaler", "ema",
        "training_state", "rng_partial", "rng_bad_numpy",
    ],
)
def test_hostile_checkpoint_failure_is_non_mutating(tmp_path, corrupt):
    cfg, path, model, optimizer, scheduler, scaler, ema, caller_state = _full_checkpoint(tmp_path)
    raw = torch.load(path, weights_only=False)
    if corrupt == "model_shape":
        key = next(iter(raw["model"]))
        raw["model"][key] = raw["model"][key].reshape(-1)[:-1]
    elif corrupt == "optimizer":
        raw["optimizer"]["param_groups"][0]["params"].append(999999)
    elif corrupt in {"scheduler", "grad_scaler", "ema"}:
        raw[corrupt] = {"value": 99, "fail": True}
    elif corrupt == "training_state":
        raw["training_state"]["optimizer_step"] = 1
    elif corrupt == "rng_partial":
        raw["rng"].pop("torch")
    elif corrupt == "rng_bad_numpy":
        raw["rng"]["numpy"] = ("invalid",)
    torch.save(raw, path)

    before = _bundle_snapshot(model, optimizer, scheduler, scaler, ema, caller_state)
    with pytest.raises(RuntimeError):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, config=cfg,
        )
    _assert_bundle_unchanged(before, model, optimizer, scheduler, scaler, ema, caller_state)


@pytest.mark.parametrize("late_component", ["scheduler", "scaler", "ema"])
def test_real_late_component_failure_rolls_back_every_component_and_rng(
    tmp_path, late_component,
):
    cfg, path, model, optimizer, _scheduler, scaler, ema, caller_state = _full_checkpoint(tmp_path)
    scheduler = StatefulComponent(21)
    if late_component == "scheduler":
        scheduler = FailOnFirstLoad(21)
    elif late_component == "scaler":
        scaler = FailOnFirstLoad(22)
    else:
        ema = FailOnFirstLoad(23)
    raw = torch.load(path, weights_only=False)
    first = next(iter(raw["model"]))
    raw["model"][first] = raw["model"][first] + 1
    raw[late_component if late_component != "scaler" else "grad_scaler"] = {"value": 99}
    torch.save(raw, path)

    FailOnFirstLoad.load_calls = 0
    before = _bundle_snapshot(model, optimizer, scheduler, scaler, ema, caller_state)
    with pytest.raises(RuntimeError, match="rolled back"):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, config=cfg,
        )
    _assert_bundle_unchanged(before, model, optimizer, scheduler, scaler, ema, caller_state)


@pytest.mark.parametrize("late_component", ["model", "optimizer"])
def test_real_model_or_optimizer_load_failure_rolls_back_every_component_and_rng(
    tmp_path, late_component,
):
    """Inject after the real live object's load has mutated it, not in preflight."""
    cfg, path, model, optimizer, scheduler, scaler, ema, caller_state = _full_checkpoint(tmp_path)
    raw = torch.load(path, weights_only=False)
    first = next(iter(raw["model"]))
    raw["model"][first] = raw["model"][first] + 1
    raw["optimizer"]["param_groups"][0]["lr"] *= 0.5
    torch.save(raw, path)

    component = model if late_component == "model" else optimizer
    real_load = component.load_state_dict
    calls = {"count": 0}

    def fail_after_real_load(_self, state, *args, **kwargs):
        calls["count"] += 1
        result = real_load(state, *args, **kwargs)
        if calls["count"] == 1:
            raise RuntimeError(f"hostile real {late_component} load failure")
        return result

    component.load_state_dict = types.MethodType(fail_after_real_load, component)
    before = _bundle_snapshot(model, optimizer, scheduler, scaler, ema, caller_state)
    with pytest.raises(RuntimeError, match="rolled back"):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, config=cfg,
        )
    assert calls["count"] == 2
    _assert_bundle_unchanged(before, model, optimizer, scheduler, scaler, ema, caller_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rollback validation")
def test_cuda_live_objects_roll_back_exactly_from_cpu_snapshots(tmp_path):
    raw_config = valid_config(tmp_path); raw_config["precision"] = "fp16"
    cfg = resolve_config(raw_config)
    model = torch.nn.Linear(2, 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    scheduler = FailOnFirstLoad(31)
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
    ema = torch.optim.swa_utils.AveragedModel(model).cuda()
    caller_state = TrainingState()
    loss = model(torch.ones(2, 2, device="cuda")).sum()
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
    optimizer.zero_grad(set_to_none=True); ema.update_parameters(model)
    path = str(tmp_path / "cuda.pt")
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        grad_scaler=scaler, ema=ema, state=caller_state, config=cfg,
        checkpoint_identity="f" * 64,
    )
    snapshots = _snapshot_component_states(
        _component_bundle(model, optimizer, scheduler, scaler, ema)
    )
    assert all(tensor.device.type == "cpu" for tensor in _walk_tensors(snapshots))
    raw = torch.load(path, map_location="cpu", weights_only=False)
    raw["model"][next(iter(raw["model"]))] += 1
    raw["scheduler"] = {"value": 99}
    torch.save(raw, path)
    FailOnFirstLoad.load_calls = 0
    before = _bundle_snapshot(model, optimizer, scheduler, scaler, ema, caller_state)
    with pytest.raises(RuntimeError, match="rolled back"):
        load_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            grad_scaler=scaler, ema=ema, config=cfg, map_location="cpu",
        )
    _assert_bundle_unchanged(before, model, optimizer, scheduler, scaler, ema, caller_state)
