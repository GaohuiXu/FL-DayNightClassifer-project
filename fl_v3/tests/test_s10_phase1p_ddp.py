from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fl_v3.data.nuscenes.cbgs import Phase1DistributedWindowSampler
from fl_v3.data.nuscenes.phase1 import phase1_worker_seed
from fl_v3.training.loop import train_one_epoch
from fl_v3.training.phase1p_ddp import (
    BLOCK_WINDOWS,
    EFFECTIVE_BATCH,
    MEASURED_WINDOWS,
    aggregate_rank_measurements,
)
from fl_v3.training.runtime_state import TrainingState


def test_distributed_window_sampler_reconstructs_each_global_b32_without_padding():
    dataset = list(range(67))
    common = {
        "dataset": dataset,
        "seed": 7,
        "consumed_samples": 64,
        "world_size": 2,
        "local_batch_size": 16,
        "effective_global_batch": 32,
        "epochs": 2,
    }
    rank0 = Phase1DistributedWindowSampler(rank=0, **common)
    rank1 = Phase1DistributedWindowSampler(rank=1, **common)
    assert len(rank0) == len(rank1) == 32
    for epoch in range(2):
        global_positions = rank0.global_epoch_positions(epoch).reshape(-1, 32)
        left = rank0.rank_epoch_positions(epoch).reshape(-1, 16)
        right = rank1.rank_epoch_positions(epoch).reshape(-1, 16)
        reconstructed = torch.cat(
            [torch.from_numpy(left), torch.from_numpy(right)], dim=1
        ).numpy()
        assert (reconstructed == global_positions).all()
        assert len(set(reconstructed.reshape(-1).tolist())) == 64


def test_distributed_window_sampler_rejects_nonexact_global_windows():
    with pytest.raises(ValueError, match="do not form effective global batch"):
        Phase1DistributedWindowSampler(
            list(range(64)),
            seed=0,
            consumed_samples=64,
            rank=0,
            world_size=2,
            local_batch_size=8,
            effective_global_batch=32,
        )
    with pytest.raises(ValueError, match="not global-window aligned"):
        Phase1DistributedWindowSampler(
            list(range(65)),
            seed=0,
            consumed_samples=63,
            rank=0,
            world_size=2,
            local_batch_size=16,
            effective_global_batch=32,
        )


def test_phase1_worker_seed_is_rank_addressed_and_epoch_resume_exact():
    assert [phase1_worker_seed(0, epoch, 2, rank) for epoch in range(3) for rank in range(2)] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert phase1_worker_seed(17, 9, 2, 1) == 36
    with pytest.raises(ValueError, match="out of range"):
        phase1_worker_seed(0, 0, 2, 2)


def test_training_loop_synchronizes_default_off_ddp_boolean_decisions():
    inputs = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    targets = inputs.sum(dim=1, keepdim=True)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    calls = []

    def global_and(name, value):
        calls.append((name, value))
        return value

    state = TrainingState()
    train_one_epoch(
        model,
        loader,
        torch.nn.MSELoss(),
        optimizer,
        torch.device("cpu"),
        accumulation_steps=1,
        runtime_state=state,
        exposure_multiplier=2,
        expected_global_microbatch_samples=4,
        distributed_boolean_and=global_and,
    )
    assert state.successful_windows == 2
    assert state.exposure_samples == 8
    assert calls == [
        ("finite_loss", True),
        ("gradients_finite", True),
        ("finite_loss", True),
        ("gradients_finite", True),
    ]


def _rank_measurement(rank: int, *, wall: float, peak_fraction: float = 0.5):
    block_wall = wall / (MEASURED_WINDOWS // BLOCK_WINDOWS)
    return {
        "rank": rank,
        "startup_seconds": {"before_training_total": 4.0 + rank},
        "training_wall_seconds_including_warmup": wall + 3.0 + rank,
        "device": {"name": "NVIDIA GH200 120GB", "total_memory_bytes": 100_000},
        "compile_evidence": {"unexpected_steady_state_recompile": False},
        "metrics": {
            "readiness_timing": {
                "measured_accepted_windows": MEASURED_WINDOWS,
                "measured_attempted_windows": MEASURED_WINDOWS,
                "measurement_wall_seconds": wall,
                "measurement_counter_delta": {
                    "exposure_samples": MEASURED_WINDOWS * EFFECTIVE_BATCH,
                    "invalid_windows": 0,
                    "discarded_windows": 0,
                    "overflow_windows": 0,
                },
                "memory": {
                    "peak_reserved_fraction": peak_fraction,
                    "monotonic_reserved_growth_over_64mib": False,
                },
                "throughput_blocks": {
                    "records": [
                        {
                            "accepted_windows": BLOCK_WINDOWS,
                            "exposure_samples": BLOCK_WINDOWS * EFFECTIVE_BATCH,
                            "wall_seconds": block_wall,
                        }
                        for _ in range(MEASURED_WINDOWS // BLOCK_WINDOWS)
                    ]
                },
            }
        },
    }


def test_rank_aggregation_uses_slower_rank_and_preserves_health():
    result = aggregate_rank_measurements(
        [_rank_measurement(0, wall=10.0), _rank_measurement(1, wall=12.0)]
    )
    assert result["measurement_wall_seconds"] == 12.0
    assert result["exposure_samples_per_second"] == (
        MEASURED_WINDOWS * EFFECTIVE_BATCH / 12.0
    )
    assert result["startup_seconds"]["before_training_total"] == 5.0
    assert result["compile_evidence"]["warmup_including_compile_seconds"] == 4.0
    assert result["gate_pass"] is True
    assert [item["rank"] for item in result["rank_devices"]] == [0, 1]

    unhealthy = copy.deepcopy(_rank_measurement(1, wall=12.0))
    unhealthy["metrics"]["readiness_timing"]["memory"][
        "peak_reserved_fraction"
    ] = 0.9
    result = aggregate_rank_measurements(
        [_rank_measurement(0, wall=10.0), unhealthy]
    )
    assert result["gate_pass"] is False
