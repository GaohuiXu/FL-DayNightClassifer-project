"""Bounded synthetic CUDA evidence for the reviewed S02 pillar implementation.

This test intentionally covers only the missing S02 GPU forward/backward gate.  It
does not perform an optimizer/GradScaler step and does not access nuScenes data.
"""
from __future__ import annotations

import json

import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder


_PILLAR_CFG = BEVConfig(
    point_cloud_range=(0.0, 0.0, -1.0, 6.0, 4.0, 1.0),
    bev_voxel=(1.0, 1.0),
    out_size_factor=1,
)


def _sample_points(cell_counts: list[tuple[int, int, int]]) -> torch.Tensor:
    rows = []
    for col, row, count in cell_counts:
        for point_idx in range(count):
            jitter = 0.01 * point_idx
            rows.append(
                [
                    0.0,
                    col + 0.10 + jitter,
                    row + 0.20 + jitter,
                    -0.20 + 0.01 * point_idx,
                    0.05 * (point_idx + 1),
                    0.0,
                ]
            )
    return torch.tensor(rows, dtype=torch.float32)


def _with_batch(points: torch.Tensor, batch_idx: int) -> torch.Tensor:
    out = points.clone()
    out[:, 0] = float(batch_idx)
    return out


def _meta_list(encoder: PointPillarsEncoder, key: str) -> list[int]:
    value = encoder.last_pillar_meta[key]
    assert isinstance(value, torch.Tensor)
    return value.cpu().tolist()


def test_s02_cuda_b3_overcap_empty_isolation_forward_backward():
    assert torch.cuda.is_available(), "S02 GPU gate requires CUDA"
    assert torch.cuda.device_count() == 1, "S02 GPU gate requires exactly one visible GPU"
    device = torch.device("cuda:0")

    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    encoder = PointPillarsEncoder(
        out_channels=32,
        max_points=3,
        max_pillars=2,
        cfg=_PILLAR_CFG,
    ).to(device)
    encoder.train()

    # Both populated samples exceed both caps. Batch element 1 is deliberately empty.
    sample_a = _sample_points([(0, 0, 6), (1, 0, 2), (2, 0, 4)])
    sample_b = _sample_points([(4, 0, 1), (0, 1, 5), (1, 1, 1), (2, 1, 2)])

    isolated_a = encoder(_with_batch(sample_a, 0).to(device), B=1)[0].detach().clone()
    batched_points = torch.cat(
        [_with_batch(sample_a, 0), _with_batch(sample_b, 2)]
    ).to(device)
    output = encoder(batched_points, B=3)

    assert output.is_cuda
    assert output.shape == (3, 32, 4, 6)
    assert torch.isfinite(output).all()
    assert torch.equal(output[0], isolated_a)
    assert torch.count_nonzero(output[1]).item() == 0

    expected_meta = {
        "input_points_per_sample": [12, 0, 9],
        "in_range_points_per_sample": [12, 0, 9],
        "occupied_pillars_per_sample": [3, 0, 4],
        "selected_pillars_per_sample": [2, 0, 2],
        "truncated_pillars_per_sample": [1, 0, 2],
        "points_kept_after_caps_per_sample": [5, 0, 4],
        "points_dropped_by_point_cap_per_sample": [3, 0, 2],
        "points_dropped_by_pillar_cap_per_sample": [4, 0, 3],
        "selected_pillar_batch_ids": [0, 0, 2, 2],
        "selected_local_pillar_keys": [0, 1, 4, 6],
    }
    actual_meta = {key: _meta_list(encoder, key) for key in expected_meta}
    assert actual_meta == expected_meta

    loss = output.square().mean()
    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()

    gradient_norms: dict[str, float] = {}
    for name, parameter in encoder.named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None, f"missing intended gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient: {name}"
        gradient_norm = float(parameter.grad.norm().detach().cpu())
        assert gradient_norm > 0.0, f"zero intended gradient: {name}"
        gradient_norms[name] = gradient_norm

    torch.cuda.synchronize()
    print(
        "S02_GPU_DIAGNOSTICS="
        + json.dumps(
            {
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "output_shape": list(output.shape),
                "output_dtype": str(output.dtype),
                "loss": float(loss.detach().cpu()),
                "gradient_norms": gradient_norms,
                "pillar_meta": actual_meta,
            },
            sort_keys=True,
        )
    )
