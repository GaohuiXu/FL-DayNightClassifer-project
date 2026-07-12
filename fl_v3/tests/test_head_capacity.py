"""S05 O-018 multi-task CenterHead topology and batch-isolation fixtures."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from fl_v3.models.fusion.head import (
    CenterPointHead,
    NUSCENES_CENTERHEAD_TASKS,
    REG_CHANNELS,
)


IN, HW = 32, 8


def test_official_six_task_topology_and_fields():
    head = CenterPointHead(in_channels=IN, n_classes=10, head_channels=16, shared_channels=16)
    assert head.class_names == NUSCENES_CENTERHEAD_TASKS
    outputs = head(torch.randn(2, IN, HW, HW))
    assert len(outputs) == 6
    for output, task in zip(outputs, NUSCENES_CENTERHEAD_TASKS):
        assert output["heatmap"].shape == (2, len(task), HW, HW)
        for name, channels in REG_CHANNELS.items():
            assert output[name].shape == (2, channels, HW, HW)


def test_every_task_field_is_an_independent_two_conv_branch():
    head = CenterPointHead(in_channels=IN, head_channels=16, shared_channels=16)
    parameter_ids = set()
    for task_head in head.task_heads:
        for name in ("heatmap", *REG_CHANNELS.keys()):
            layers = task_head.branches[name].layers
            convs = [layer for layer in layers if isinstance(layer, nn.Conv2d)]
            assert len(convs) == 2
            assert convs[0].kernel_size == (3, 3)
            assert convs[1].kernel_size == (3, 3)
            ids = {id(param) for param in task_head.branches[name].parameters()}
            assert parameter_ids.isdisjoint(ids), f"{name} unexpectedly shares branch parameters"
            parameter_ids.update(ids)


def test_heatmap_bias_is_centerpoint_prior_for_every_task():
    head = CenterPointHead(in_channels=IN, head_channels=16, shared_channels=16)
    for task_head in head.task_heads:
        bias = task_head.branches["heatmap"].layers[-1].bias
        assert torch.equal(bias, torch.full_like(bias, -2.19))


def test_groupnorm_makes_sample_output_batch_independent():
    torch.manual_seed(17)
    head = CenterPointHead(in_channels=IN, head_channels=16, shared_channels=16).train()
    sample = torch.randn(1, IN, HW, HW)
    distractor = torch.randn(1, IN, HW, HW) * 100.0 + 50.0
    alone = head(sample)
    paired = head(torch.cat((sample, distractor), dim=0))
    for task_alone, task_paired in zip(alone, paired):
        for field in task_alone:
            assert torch.allclose(task_alone[field][0], task_paired[field][0], atol=1e-5, rtol=1e-5)


def test_o018_rejects_non_two_conv_task_fields():
    with pytest.raises(ValueError, match="two convolutions"):
        CenterPointHead(in_channels=IN, conv_layers=1)


def test_task_classes_must_be_unique_and_match_n_classes():
    with pytest.raises(ValueError, match="unique"):
        CenterPointHead(in_channels=IN, tasks=(("car",), ("car",)), n_classes=2)
    with pytest.raises(ValueError, match="task total"):
        CenterPointHead(in_channels=IN, n_classes=9)
