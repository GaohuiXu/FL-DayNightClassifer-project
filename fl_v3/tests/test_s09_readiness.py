from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.training.loop import train_one_epoch
from fl_v3.training.runtime_state import EpochPermutationSampler, TrainingState
from fl_v3.utils.runtime import make_grad_scaler


def _centralized_train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "centralized_train.py"
    spec = importlib.util.spec_from_file_location("s09_centralized_train", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loader(n: int = 6) -> DataLoader:
    inputs = torch.arange(float(n)).reshape(n, 1) / 5.0
    targets = 1.5 * inputs - 0.25
    return DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)


def _assert_nested_equal(left, right) -> None:
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def _run_output_neutral_pair(device: torch.device, precision: str):
    torch.manual_seed(711)
    base = torch.nn.Sequential(
        torch.nn.Linear(1, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 1),
    ).to(device)
    initial = copy.deepcopy(base.state_dict())

    def run(timing: bool):
        model = copy.deepcopy(base)
        model.load_state_dict(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        ema = torch.optim.swa_utils.AveragedModel(model, use_buffers=False)
        scaler = make_grad_scaler(device, precision)
        state = TrainingState()
        cpu_loader = _loader()
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        metrics = train_one_epoch(
            model,
            cpu_loader,
            torch.nn.MSELoss(),
            optimizer,
            device,
            scheduler=scheduler,
            ema_model=ema,
            precision=precision,
            grad_scaler=scaler,
            accumulation_steps=1,
            runtime_state=state,
            max_steps=5,
            max_optimizer_steps=4,
            expected_global_microbatch_samples=1,
            readiness_timing=timing,
            readiness_warmup_successful_windows=1 if timing else 0,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "ema": copy.deepcopy(ema.state_dict()),
            "scaler": copy.deepcopy(scaler.state_dict()),
            "state": state.checkpoint_dict(),
            "metrics": metrics,
            "rng_before": rng,
            "rng_after": torch.get_rng_state().clone(),
            "cuda_rng_before": cuda_rng,
            "cuda_rng_after": (
                torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
            ),
        }

    # Restore the same RNG boundary before each execution; timing itself must not
    # consume either host or device RNG state.
    host_rng = torch.get_rng_state().clone()
    device_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    torch.set_rng_state(host_rng)
    if device_rng is not None:
        torch.cuda.set_rng_state(device_rng, device)
    plain = run(False)
    torch.set_rng_state(host_rng)
    if device_rng is not None:
        torch.cuda.set_rng_state(device_rng, device)
    timed = run(True)
    return plain, timed


def _assert_output_neutral_pair(plain, timed) -> None:
    for key in ("model", "optimizer", "scheduler", "ema", "scaler", "state"):
        _assert_nested_equal(plain[key], timed[key])
    for key, value in plain["metrics"].items():
        assert key != "readiness_timing"
        _assert_nested_equal(value, timed["metrics"][key])
    assert "readiness_timing" in timed["metrics"]
    _assert_nested_equal(plain["rng_before"], timed["rng_before"])
    _assert_nested_equal(plain["rng_after"], timed["rng_after"])
    if plain["cuda_rng_before"] is not None:
        _assert_nested_equal(plain["cuda_rng_before"], timed["cuda_rng_before"])
        _assert_nested_equal(plain["cuda_rng_after"], timed["cuda_rng_after"])

    report = timed["metrics"]["readiness_timing"]
    assert report["warmup_boundary_reached"] is True
    assert report["warmup_boundary_attempted_window"] == 1
    assert report["measured_attempted_windows"] == 3
    assert report["measured_accepted_windows"] == 3
    assert report["measurement_counter_delta"]["successful_windows"] == 3
    assert report["measurement_counter_delta"]["exposure_samples"] == 3
    assert report["component_counters"]["optimizer_step"] == 4
    assert report["component_counters"]["ema_n_averaged"] == 4
    assert report["accepted_stage_ms"]["window"]["n"] == 3
    assert len(report["records"]) == 4
    json.dumps(report, allow_nan=False)


def test_cpu_direct_timing_is_output_neutral_and_json_finite():
    plain, timed = _run_output_neutral_pair(torch.device("cpu"), "fp32")
    _assert_output_neutral_pair(plain, timed)
    assert timed["metrics"]["readiness_timing"]["clock"] == "host_perf_counter"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA event path requires a GPU")
def test_cuda_event_timing_is_output_neutral_and_json_finite():
    device = torch.device("cuda")
    plain, timed = _run_output_neutral_pair(device, "fp16")
    _assert_output_neutral_pair(plain, timed)
    report = timed["metrics"]["readiness_timing"]
    assert report["clock"] == "cuda_event"
    assert report["memory"]["peak_reserved_bytes"] > 0
    assert report["memory"]["reserved_headroom_bytes"] >= 0


class OverflowOnceScaler:
    def __init__(self):
        self.scale_value = 8.0
        self.overflow = True
        self.get_scale_calls = 0

    def is_enabled(self): return True
    def get_scale(self):
        self.get_scale_calls += 1
        return self.scale_value
    def scale(self, loss): return loss * self.scale_value

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.scale_value)

    def step(self, optimizer):
        if not self.overflow:
            optimizer.step()

    def update(self):
        if self.overflow:
            self.scale_value /= 2.0
            self.overflow = False


def test_attempted_window_cap_records_overflow_without_relabeling_success():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    state = TrainingState()
    scaler = OverflowOnceScaler()
    observed_windows = []
    metrics = train_one_epoch(
        model,
        _loader(5),
        torch.nn.MSELoss(),
        optimizer,
        torch.device("cpu"),
        grad_scaler=scaler,
        precision="fp16",
        accumulation_steps=1,
        runtime_state=state,
        max_steps=3,
        max_optimizer_steps=3,
        expected_global_microbatch_samples=1,
        readiness_timing=True,
        readiness_warmup_successful_windows=0,
        attempted_window_callback=lambda: observed_windows.append(state.attempted_windows),
    )
    assert state.attempted_windows == 3
    assert state.successful_windows == state.optimizer_step == 2
    assert state.invalid_windows == state.overflow_windows == 1
    assert state.nonfinite_windows == state.discarded_windows == 0
    report = metrics["readiness_timing"]
    assert [record["outcome"] for record in report["records"]] == [
        "overflow", "accepted", "accepted",
    ]
    assert report["component_counters"]["scaler_scale_at_start"] == 8.0
    assert report["component_counters"]["scaler_scale_at_end"] == 4.0
    assert report["component_counters"]["scaler_skips"] == 1
    # Two reads per enabled-scaler optimizer attempt plus the loop's pre-existing
    # terminal metrics read; readiness timing must not add its own scale syncs.
    assert scaler.get_scale_calls == 2 * state.attempted_windows + 1
    assert observed_windows == [1, 2, 3]


def test_readiness_timing_fails_closed_on_incompatible_state_or_options():
    def parts():
        model = torch.nn.Linear(1, 1)
        return model, torch.optim.SGD(model.parameters(), lr=0.01)

    model, optimizer = parts()
    with pytest.raises(RuntimeError, match="fresh non-resumed"):
        train_one_epoch(
            model, _loader(2), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            runtime_state=TrainingState(epoch=1), readiness_timing=True,
            max_optimizer_steps=1,
        )

    model, optimizer = parts()
    with pytest.raises(RuntimeError, match="requires accumulation_steps == 1"):
        train_one_epoch(
            model, _loader(2), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            accumulation_steps=2, readiness_timing=True, max_steps=2,
            max_optimizer_steps=1,
        )

    model, optimizer = parts()
    with pytest.raises(RuntimeError, match="cannot enable the S08 precision observer"):
        train_one_epoch(
            model, _loader(2), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            readiness_timing=True, precision_diagnostics=object(),
            max_optimizer_steps=1,
        )

    model, optimizer = parts()
    with pytest.raises(ValueError, match="warm-up must be below"):
        train_one_epoch(
            model, _loader(2), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            readiness_timing=True, readiness_warmup_successful_windows=1,
            max_optimizer_steps=1,
        )

    model, optimizer = parts()
    with pytest.raises(RuntimeError, match="restricted to S09 readiness"):
        train_one_epoch(
            model, _loader(2), torch.nn.MSELoss(), optimizer, torch.device("cpu"),
            attempted_window_callback=lambda: None, max_optimizer_steps=1,
        )


def _profile_collate(batch):
    return {
        "batch_size": len(batch),
        "value": torch.stack([item[0] for item in batch]),
    }


class ProfileTask:
    def __init__(self, *, drift: bool = False):
        self.calls = []
        self.drift = drift

    def _make_loader(self, run_config, _infos, _tokens, shuffle):
        assert shuffle is True
        workers = int(run_config["num-workers"])
        self.calls.append(workers)
        offset = 1000 * workers if self.drift else 0
        dataset = TensorDataset(torch.arange(20) + offset)
        sampler = EpochPermutationSampler(dataset, seed=int(run_config["seed"]))
        # Worker mechanics are covered by the production data tests. This bounded
        # unit fixture checks cell orchestration/content gating without processes.
        return DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            num_workers=0,
            collate_fn=_profile_collate,
        )


def test_loader_profile_is_bounded_observational_and_content_gated():
    entry = _centralized_train_module()
    task = ProfileTask()
    run_config = {"num-workers": 2, "seed": 17}
    spec = {
        "workers": [0, 2],
        "repeats": 2,
        "determinism_batches": 2,
        "warmup_batches": 1,
        "measured_batches": 2,
    }
    report = entry.run_production_loader_profile(
        task=task,
        run_config=run_config,
        infos=[],
        tokens=[],
        profile_spec=spec,
    )
    assert task.calls == [0, 2]
    assert run_config == {"num-workers": 2, "seed": 17}
    assert report["training_num_workers"] == 2
    assert report["content_sha256_identical"] is True
    assert [cell["num_workers"] for cell in report["profiles"]] == [0, 2]
    assert all(len(cell["repeats"]) == 2 for cell in report["profiles"])
    assert all(
        repeat["measured_batches"] == 2
        for cell in report["profiles"]
        for repeat in cell["repeats"]
    )
    json.dumps(report, allow_nan=False)

    drift = entry.run_production_loader_profile(
        task=ProfileTask(drift=True),
        run_config=run_config,
        infos=[],
        tokens=[],
        profile_spec=spec,
    )
    assert drift["content_sha256_identical"] is False


def test_json_evidence_writer_refuses_nonfinite_values(tmp_path):
    entry = _centralized_train_module()
    with pytest.raises(ValueError, match="Out of range float values"):
        entry._write_json(tmp_path / "bad.json", {"value": float("nan")})


def test_bounded_operator_profiler_emits_one_trace_and_summary(monkeypatch, tmp_path):
    entry = _centralized_train_module()

    class Event:
        key = "fl_v3::camera.backbone"
        count = 3
        self_cpu_time_total = 4.0
        cpu_time_total = 5.0
        self_device_time_total = 6.0
        device_time_total = 7.0
        self_cpu_memory_usage = 8
        cpu_memory_usage = 9
        self_device_memory_usage = 10
        device_memory_usage = 11

    class FakeProfiler:
        def __init__(self, **kwargs):
            self.handler = kwargs["on_trace_ready"]
            self.steps = 0
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def step(self):
            self.steps += 1
            if self.steps == 3:
                self.handler(self)
        def export_chrome_trace(self, path):
            Path(path).write_text("{}\n", encoding="utf-8")
        def key_averages(self, **_kwargs): return [Event()]

    monkeypatch.setattr(entry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(entry.torch.profiler, "schedule", lambda **kwargs: kwargs)
    monkeypatch.setattr(entry.torch.profiler, "profile", FakeProfiler)
    spec = {
        "wait_attempted_windows": 1,
        "warmup_attempted_windows": 1,
        "active_attempted_windows": 1,
        "record_shapes": True,
        "profile_memory": True,
        "row_limit": 10,
    }
    profiler = entry.BoundedOperatorProfiler(spec, tmp_path / "operator")
    with profiler:
        profiler.step()
        profiler.step()
        profiler.step()
    report = profiler.report()
    assert report["attempted_window_step_calls"] == 3
    summary = json.loads((tmp_path / "operator" / "summary.json").read_text())
    assert summary["all_row_count"] == 1
    assert summary["rows"][0]["key"] == "fl_v3::camera.backbone"
    assert summary["rows"][0]["self_device_time_total_us"] == 6.0
    json.dumps(report, allow_nan=False)


class _ReadinessConfigFixture:
    def __init__(self):
        self.data = {
            "schema_version": "s09.v1",
            "precision": "fp32",
            "sparse_conv_precision": "not_applicable",
            "model": {"mode": "camera_only"},
            "optimizer": {
                "name": "adamw",
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
            },
            "training": {
                "max_optimizer_steps": 3,
                "micro_batch_size": 1,
                "world_size": 1,
                "accumulation_steps": 1,
                "effective_global_batch": 1,
                "seed": 23,
                "max_epochs": 1,
                "num_workers": 0,
                "ema_decay": None,
                "sampling": "uniform",
            },
            "execution": {
                "mode": "readiness",
                "max_attempted_windows": 4,
                "timing_warmup_successful_windows": 1,
                "loader_profile": None,
            },
        }
        self.canonical_bytes = json.dumps(
            self.data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.sha256 = hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def execution_mode(self): return "readiness"
    @property
    def precision(self): return "fp32"
    @property
    def model_mode(self): return "camera_only"
    @property
    def sparse_conv_precision(self): return "not_applicable"
    @property
    def data_identities(self): return {"fixture": "bounded-toy-no-nuscenes"}

    def as_dict(self):
        return copy.deepcopy(self.data)

    def to_run_config(self):
        return {
            "nuscenes-train-split": "toy_train",
            "num-workers": 0,
            "seed": 23,
            "det-camera-activation-checkpoint": True,
        }


class _ReadinessTaskFixture:
    def _load_info(self, _run_config, split):
        assert split == "toy_train"
        return [{"sample_token": str(index)} for index in range(4)], {}

    def _partition(self, _run_config):
        return {
            "mode": "fixture",
            "num_clients": 1,
            "client_tokens": {0: ["0", "1", "2", "3"]},
        }

    def _make_loader(self, run_config, _infos, _tokens, shuffle):
        assert shuffle is True
        inputs = torch.arange(4.0).reshape(4, 1)
        dataset = TensorDataset(inputs, 2.0 * inputs)
        sampler = EpochPermutationSampler(dataset, seed=int(run_config["seed"]))
        return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

    def build_model(self, _run_config):
        return torch.nn.Linear(1, 1)

    def build_criterion(self, _run_config):
        return torch.nn.MSELoss()


def _patch_readiness_main(entry, config, task, monkeypatch):
    import fl_v3.training.tasks as tasks_module

    monkeypatch.setattr(entry, "load_resolved_config", lambda _path: config)
    monkeypatch.setattr(
        entry,
        "verify_runtime_dependency_identity",
        lambda _run: {"torch": "bounded-test-fixture"},
    )
    monkeypatch.setattr(entry, "verify_physical_data_identities", lambda _config: None)
    monkeypatch.setattr(entry, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(entry, "enforce_determinism", lambda **_kwargs: None)
    monkeypatch.setattr(tasks_module, "get_task", lambda _name: task)
    monkeypatch.setattr(
        entry,
        "save_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("readiness wrote a checkpoint"),
    )
    monkeypatch.setattr(
        entry,
        "run_strict_official_evaluation",
        lambda **_kwargs: pytest.fail("readiness ran official evaluation"),
    )


def test_readiness_lifecycle_writes_terminal_artifact_without_checkpoint_or_eval(
    tmp_path, monkeypatch,
):
    entry = _centralized_train_module()
    config = _ReadinessConfigFixture()
    task = _ReadinessTaskFixture()
    _patch_readiness_main(entry, config, task, monkeypatch)
    out_dir = tmp_path / "fresh-readiness-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["centralized_train.py", "--config", "fixture.json", "--out-dir", str(out_dir)],
    )

    entry.main()

    assert (out_dir / "resolved_config.json").read_bytes() == config.canonical_bytes + b"\n"
    assert (out_dir / "runtime_dependencies.json").is_file()
    assert not (out_dir / "checkpoint.pt").exists()
    assert not (out_dir / "checkpoint.sha256").exists()
    assert not (out_dir / "official_metrics.json").exists()
    report = json.loads((out_dir / "readiness.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["resolved_config_sha256"] == config.sha256
    assert report["terminal_training_state"]["epoch"] == 0
    assert report["terminal_training_state"]["optimizer_step"] == 3
    assert report["terminal_training_state"]["attempted_windows"] == 3
    assert report["checkpoint_written"] is False
    assert report["official_evaluation_executed"] is False
    assert report["training_metrics"]["readiness_timing"]["measured_accepted_windows"] == 2
    json.dumps(report, allow_nan=False)


def test_readiness_lifecycle_rejects_resume_and_existing_output(tmp_path, monkeypatch):
    entry = _centralized_train_module()
    config = _ReadinessConfigFixture()
    _patch_readiness_main(entry, config, _ReadinessTaskFixture(), monkeypatch)
    out_dir = tmp_path / "readiness-output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "centralized_train.py", "--config", "fixture.json",
            "--out-dir", str(out_dir), "--resume",
        ],
    )
    with pytest.raises(RuntimeError, match="non-resumable"):
        entry.main()
    assert not out_dir.exists()

    out_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["centralized_train.py", "--config", "fixture.json", "--out-dir", str(out_dir)],
    )
    with pytest.raises(RuntimeError, match="fresh absent output directory"):
        entry.main()
    assert not (out_dir / "readiness.json").exists()


def test_readiness_unmet_target_writes_fail_artifact_before_nonzero_exit(
    tmp_path, monkeypatch,
):
    entry = _centralized_train_module()
    config = _ReadinessConfigFixture()
    config.data["training"]["max_optimizer_steps"] = 5
    config.data["execution"]["max_attempted_windows"] = 6
    # Rebind the fake config's canonical identity after its legitimate bounded
    # lifecycle fields change.
    config.canonical_bytes = json.dumps(
        config.data, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    config.sha256 = hashlib.sha256(config.canonical_bytes).hexdigest()
    _patch_readiness_main(entry, config, _ReadinessTaskFixture(), monkeypatch)
    out_dir = tmp_path / "unmet-readiness-output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["centralized_train.py", "--config", "fixture.json", "--out-dir", str(out_dir)],
    )

    with pytest.raises(RuntimeError, match="target not reached"):
        entry.main()

    report = json.loads((out_dir / "readiness.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["terminal_training_state"]["optimizer_step"] == 4
    assert report["terminal_training_state"]["attempted_windows"] == 4
    assert report["target_successful_windows"] == 5
    assert report["checkpoint_written"] is False
    assert report["official_evaluation_executed"] is False
    assert not (out_dir / "checkpoint.pt").exists()
    json.dumps(report, allow_nan=False)
