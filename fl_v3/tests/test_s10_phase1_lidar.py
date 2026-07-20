import math

import pytest
import torch
import torch.nn as nn

from fl_v3.config import load_resolved_config
from fl_v3.models.fusion.nms_deterministic import rotated_iou_bev
from fl_v3.models.fusion.sparse_voxel_encoder import select_xyzi_time_features
from fl_v3.models.phase1_lidar import (
    Phase1SECOND,
    Phase1SECONDFPN,
    build_phase1_lidar_model,
)
from fl_v3.models.phase1_transfusion import (
    Phase1TransFusionHead,
    Phase1TransFusionLoss,
    TransFusionGeometry,
    create_bev_position_grid,
    decode_transfusion_tensors,
    encode_transfusion_boxes,
    focal_loss_cost,
    pairwise_iou3d,
)


def _mini_geometry() -> TransFusionGeometry:
    # 0.6 m output cells -> H=4, W=5 while preserving all production equations.
    return TransFusionGeometry(
        point_cloud_range=(-1.5, -1.2, -5.0, 1.5, 1.2, 3.0),
        voxel_size=(0.075, 0.075, 0.2),
        out_size_factor=8,
        bev_hw=(4, 5),
    )


def test_phase1_keyframe_and_multisweep_features_preserve_ring_only_in_source():
    keyframe = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 0.4, 17.0], [1.0, -1.0, -2.0, -3.0, 0.8, 2.0]]
    )
    selected = select_xyzi_time_features(keyframe)
    assert selected.shape == (2, 5)
    assert torch.equal(selected[:, :4], keyframe[:, 1:5])
    assert torch.equal(selected[:, 4], torch.zeros(2))
    multisweep = torch.cat((keyframe, torch.tensor([[0.0], [-0.25]])), dim=1)
    selected_sweep = select_xyzi_time_features(multisweep)
    assert torch.equal(selected_sweep[:, :4], keyframe[:, 1:5])
    assert torch.equal(selected_sweep[:, 4], multisweep[:, 6])
    with pytest.raises(ValueError, match="keyframe.*multi-sweep"):
        select_xyzi_time_features(torch.zeros(1, 5))


def test_phase1_second_and_secondfpn_exact_module_inventory_and_shapes():
    backbone = Phase1SECOND().eval()
    assert len(backbone.blocks) == 2
    assert [len(block) for block in backbone.blocks] == [6, 6]
    convs = [module for module in backbone.modules() if isinstance(module, nn.Conv2d)]
    norms = [module for module in backbone.modules() if isinstance(module, nn.BatchNorm2d)]
    assert len(convs) == len(norms) == 12
    assert [(convs[0].in_channels, convs[0].out_channels, convs[0].stride)] == [
        (256, 128, (1, 1))
    ]
    assert (convs[6].in_channels, convs[6].out_channels, convs[6].stride) == (
        128,
        256,
        (2, 2),
    )
    assert all(module.eps == 1e-3 and module.momentum == 0.01 for module in norms)
    fine, coarse = backbone(torch.randn(2, 256, 8, 10))
    assert fine.shape == (2, 128, 8, 10)
    assert coarse.shape == (2, 256, 4, 5)

    neck = Phase1SECONDFPN().eval()
    output = neck((fine, coarse))
    assert output.shape == (2, 512, 8, 10)
    assert isinstance(neck.deblocks[0][0], nn.Conv2d)
    assert neck.deblocks[0][0].kernel_size == (1, 1)
    assert isinstance(neck.deblocks[1][0], nn.ConvTranspose2d)
    assert neck.deblocks[1][0].kernel_size == (2, 2)
    assert all(
        module.eps == 1e-3 and module.momentum == 0.01
        for module in neck.modules()
        if isinstance(module, nn.BatchNorm2d)
    )


def test_phase1_transfusion_position_grid_tracks_h_equals_y_w_equals_x():
    grid = create_bev_position_grid(_mini_geometry())
    assert grid.shape == (1, 20, 2)
    assert torch.equal(
        grid[0, :6],
        torch.tensor(
            [
                [0.5, 0.5],
                [1.5, 0.5],
                [2.5, 0.5],
                [3.5, 0.5],
                [4.5, 0.5],
                [0.5, 1.5],
            ]
        ),
    )


def test_phase1_vectorized_iou3d_matches_known_and_cpu_polygon_geometry():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0],
            [10.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0],
            [2.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2],
        ],
        dtype=torch.float64,
    )
    observed = pairwise_iou3d(boxes[:1], boxes)
    assert observed[0, 0].item() == pytest.approx(1.0, abs=1e-12)
    assert observed[0, 1].item() == 0.0
    assert observed[0, 2].item() == pytest.approx(1.0 / 3.0, abs=1e-10)
    assert observed[0, 3].item() == pytest.approx(1.0 / 3.0, abs=1e-10)

    generator = torch.Generator().manual_seed(31)
    random_boxes = torch.zeros(6, 7, dtype=torch.float64)
    random_boxes[:, :2] = torch.randn(6, 2, generator=generator)
    random_boxes[:, 2] = 0.25
    random_boxes[:, 3:6] = torch.rand(6, 3, generator=generator) + 0.5
    # This block compares 3-D IoU with the independent BEV-only polygon oracle.
    # Equal centers alone are insufficient when heights differ: the 3-D union
    # then includes the taller box outside the shared vertical interval.  Hold
    # height equal so the 3-D ratio reduces exactly to the BEV ratio being tested.
    random_boxes[:, 5] = 1.0
    random_boxes[:, 6] = torch.rand(6, generator=generator) * 2 * math.pi - math.pi
    actual = pairwise_iou3d(random_boxes[:3], random_boxes[3:])
    for row in range(3):
        for column in range(3):
            expected = rotated_iou_bev(
                random_boxes[row].tolist(), random_boxes[column + 3].tolist()
            )
            assert actual[row, column].item() == pytest.approx(expected, abs=2e-8)


def test_phase1_focal_assignment_cost_matches_independent_equation():
    logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=torch.float64)
    labels = torch.tensor([2, 0])
    observed = focal_loss_cost(logits, labels)
    probability = logits.sigmoid()
    negative = -(1 - probability + 1e-12).log() * 0.75 * probability.square()
    positive = -(probability + 1e-12).log() * 0.25 * (1 - probability).square()
    expected = (positive[:, labels] - negative[:, labels]) * 0.15
    assert torch.equal(observed, expected)


def test_phase1_transfusion_box_code_roundtrip_keeps_geometric_center_z():
    geometry = _mini_geometry()
    boxes = torch.tensor(
        [
            [-0.9, -0.3, 0.7, 4.0, 1.8, 1.6, 0.4, 2.5, -0.2],
            [0.3, 0.3, -0.1, 0.8, 0.6, 1.7, -1.2, 0.0, 0.0],
        ]
    )
    code = encode_transfusion_boxes(boxes, geometry)
    decoded = decode_transfusion_tensors(
        code[:, :2].T.unsqueeze(0),
        code[:, 2:3].T.unsqueeze(0),
        code[:, 3:6].T.unsqueeze(0),
        code[:, 6:8].T.unsqueeze(0),
        code[:, 8:10].T.unsqueeze(0),
        geometry,
    )[0]
    assert torch.allclose(decoded, boxes, rtol=1e-6, atol=1e-6)
    assert torch.equal(decoded[:, 2], boxes[:, 2])


def test_phase1_transfusion_forward_loss_backward_small_exact_geometry():
    torch.manual_seed(0)
    geometry = _mini_geometry()
    head = Phase1TransFusionHead(geometry=geometry)
    criterion = Phase1TransFusionLoss(geometry=geometry)
    features = torch.randn(2, 512, 4, 5, requires_grad=True)
    output = head(features)
    assert output["dense_heatmap"].shape == (2, 10, 4, 5)
    assert output["heatmap"].shape == (2, 10, 200)
    assert output["center"].shape == (2, 2, 200)
    assert output["query_labels"].shape == (2, 200)
    assert torch.equal(
        head.prediction_head.heads["heatmap"][-1].bias,
        torch.full((10,), -2.19),
    )
    assert not any(isinstance(module, nn.GroupNorm) for module in head.modules())

    batch = {
        "gt_boxes": [
            torch.tensor([[-0.9, -0.6, 0.0, 0.8, 0.5, 1.5, 0.2]]),
            torch.tensor([[0.3, 0.3, -0.1, 1.0, 0.6, 1.7, -0.4]]),
        ],
        "gt_velocity": [torch.tensor([[1.0, 0.2]]), torch.tensor([[0.0, 0.0]])],
        "gt_labels": [torch.tensor([0]), torch.tensor([8])],
    }
    terms = criterion.loss_terms(output, batch)
    assert set(terms) == {"loss_heatmap", "loss_cls", "loss_bbox", "matched_iou"}
    assert all(torch.isfinite(value) for value in terms.values())
    loss = terms["loss_heatmap"] + terms["loss_cls"] + terms["loss_bbox"]
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_phase1_full_lidar_builder_has_only_reference_normalization_when_spconv_available():
    pytest.importorskip("spconv.pytorch")
    torch.manual_seed(0)
    config = load_resolved_config("fl_v3/configs/s10_phase1_lidar.json")
    model = build_phase1_lidar_model(config)
    assert model.lidar_encoder.output_mode == "collapsed"
    assert model.lidar_encoder.point_feature_mode == "xyzi_time"
    assert model.lidar_encoder.n_pt_feat == 5
    assert model.lidar_encoder.to_bev is None
    assert model.lidar_encoder.second_normalization == "batch_norm_1d"
    assert not any(isinstance(module, nn.GroupNorm) for module in model.modules())
    sparse_bn = [
        module
        for module in model.lidar_encoder.backbone.modules()
        if isinstance(module, nn.BatchNorm1d)
    ]
    assert sparse_bn
    assert all(module.eps == 1e-3 and module.momentum == 0.01 for module in sparse_bn)
