from __future__ import annotations

import copy

import pytest
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.losses import MultiTaskCenterPointLoss
from fl_v3.models.fusion.second_sparse_backbone import _ObservedGroupNorm
from fl_v3.training.s10_observation import (
    StopBObservationRecorder,
    attribute_term_gradients,
    compare_parameter_gradient_tensors,
    loss_term_snapshot,
    recompose_from_sample_terms,
)


def _outputs(batch_size: int, size: int = 4):
    result = []
    for classes in (1, 2, 2, 1, 2, 2):
        result.append({
            "heatmap": torch.randn(batch_size, classes, size, size, requires_grad=True),
            "reg": torch.randn(batch_size, 2, size, size, requires_grad=True),
            "height": torch.randn(batch_size, 1, size, size, requires_grad=True),
            "dim": torch.randn(batch_size, 3, size, size, requires_grad=True),
            "rot": torch.randn(batch_size, 2, size, size, requires_grad=True),
            "vel": torch.randn(batch_size, 2, size, size, requires_grad=True),
        })
    return result


def _batch():
    boxes, labels, velocity = [], [], []
    for sample in range(4):
        sample_boxes = []
        sample_labels = []
        sample_velocity = []
        for label in range(10):
            sample_boxes.append([
                -1.5 + 0.25 * ((label + sample) % 10),
                -1.5 + 0.25 * ((2 * label + sample) % 10),
                0.0, 0.5, 0.5, 0.5, 0.1 * label,
            ])
            sample_labels.append(label)
            sample_velocity.append([0.1 * label, -0.1 * label])
        boxes.append(torch.tensor(sample_boxes, dtype=torch.float32))
        labels.append(torch.tensor(sample_labels, dtype=torch.int64))
        velocity.append(torch.tensor(sample_velocity, dtype=torch.float32))
    return {"gt_boxes": boxes, "gt_labels": labels, "gt_velocity": velocity}


def _bev():
    return BEVConfig(
        point_cloud_range=(-2.0, -2.0, -1.0, 2.0, 2.0, 1.0),
        bev_voxel=(1.0, 1.0),
        out_size_factor=1,
    )


def test_s10_loss_capture_is_output_and_gradient_neutral_and_recomposes_b4():
    torch.manual_seed(7)
    plain_outputs = _outputs(4)
    observed_outputs = [
        {name: value.detach().clone().requires_grad_() for name, value in task.items()}
        for task in plain_outputs
    ]
    plain = MultiTaskCenterPointLoss(_bev())
    observed = copy.deepcopy(plain)
    batch = _batch()

    plain_loss = plain(plain_outputs, batch)
    with observed.capture_s10_terms():
        observed_loss = observed(observed_outputs, batch)
    bundle = observed.s10_term_bundle()
    recomposed, task_values = recompose_from_sample_terms(bundle)

    assert torch.equal(plain_loss, observed_loss)
    assert torch.allclose(recomposed, observed_loss, rtol=1e-5, atol=1e-7)
    assert len(task_values) == 6
    snapshot = loss_term_snapshot(bundle)
    assert len(snapshot["tasks"]) == 6
    assert all(len(task["hm_sample_numerators"]) == 4 for task in snapshot["tasks"])

    plain_loss.backward()
    observed_loss.backward()
    for left_task, right_task in zip(plain_outputs, observed_outputs, strict=True):
        for name in left_task:
            assert torch.equal(left_task[name].grad, right_task[name].grad)


def test_observed_group_norm_keeps_state_output_and_gradient_identical():
    torch.manual_seed(11)
    reference = torch.nn.GroupNorm(8, 16)
    observed = _ObservedGroupNorm(8, 16, observation_label="second.stem.norm")
    observed.load_state_dict(reference.state_dict(), strict=True)
    assert reference.state_dict().keys() == observed.state_dict().keys()

    recorder = StopBObservationRecorder(expected_boundaries=(), expected_group_norm_count=1)
    observed._s10_observer = recorder
    x_reference = torch.randn(32, 16, requires_grad=True)
    x_observed = x_reference.detach().clone().requires_grad_()
    y_reference = reference(x_reference)
    y_observed = observed(x_observed)
    assert torch.equal(y_reference, y_observed)
    y_reference.square().sum().backward()
    y_observed.square().sum().backward()
    assert torch.equal(x_reference.grad, x_observed.grad)
    assert torch.equal(reference.weight.grad, observed.weight.grad)
    assert torch.equal(reference.bias.grad, observed.bias.grad)
    recorder.validate_forward()
    stats = recorder.group_norms["second.stem.norm"]
    assert stats["values_per_group"] == 2
    assert stats["group_instances"] == 32 * 8


def test_term_projection_reconstructs_aggregate_boundary_gradient():
    torch.manual_seed(19)
    outputs = _outputs(4)
    criterion = MultiTaskCenterPointLoss(_bev())
    # One explicit upstream boundary shared by all six task predictions.
    boundary = torch.randn(4, 3, 4, 4, requires_grad=True)
    for task in outputs:
        for name, value in tuple(task.items()):
            # Preserve each field's independent leaf while making the synthetic
            # boundary part of every exact task/term graph.
            task[name] = value + boundary[:, :1].expand_as(value) * 0.01
    recorder = StopBObservationRecorder(
        expected_boundaries=("head.input",), expected_group_norm_count=0
    )
    recorder.capture_dense_boundary("head.input", boundary)
    with criterion.capture_s10_terms():
        criterion(outputs, _batch())
    attribution = attribute_term_gradients(criterion.s10_term_bundle(), recorder)
    check = attribution["aggregate_gradient_reconstruction"]["head.input"]
    assert check["allclose_rtol_1e-5_atol_1e-7"]
    shares = sum(
        source["head.input"]["projection_share"]
        for source in attribution["sources"].values()
    )
    assert shares == pytest.approx(1.0, rel=1e-5, abs=1e-6)


def test_fixed_gradient_parity_gate_separates_hash_drift_from_numerical_drift():
    reference = {
        "lidar_encoder.backbone.stem.weight": torch.tensor([64.0, -128.0]),
        "head.bias": torch.tensor([0.0]),
        "unused": None,
    }
    numerically_neutral = {
        "lidar_encoder.backbone.stem.weight": torch.tensor([64.00001, -128.0]),
        "head.bias": torch.tensor([0.0]),
        "unused": None,
    }
    accepted = compare_parameter_gradient_tensors(
        reference, numerically_neutral, scale_divisor=64.0
    )
    assert accepted["gate_pass"]
    assert accepted["missing_gradient_sets_equal"]
    assert accepted["global"]["relative_l2_error"] <= 1e-6

    drifted = dict(numerically_neutral)
    drifted["head.bias"] = torch.tensor([1.0])
    rejected = compare_parameter_gradient_tensors(
        reference, drifted, scale_divisor=64.0
    )
    assert not rejected["gate_pass"]
    assert "head.bias" in rejected["allclose_failure_parameters"]
