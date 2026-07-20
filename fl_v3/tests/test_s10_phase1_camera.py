from __future__ import annotations

import numpy as np
import torch

from fl_v3.models.fusion.preprocess import (
    sample_reference_image_augmentation_parameters,
)
from fl_v3.models.fusion.centerhead_decode import (
    CenterHeadDecodeConfig,
    select_task_candidates,
)
from fl_v3.models.phase1_camera import (
    Phase1CameraDetector,
    Phase1GeneralizedLSSFPN,
)
from fl_v3.models.phase1_swin import original_swin_destination_key


def test_reference_image_augmentation_uses_exact_numpy_draw_order():
    np.random.seed(417)
    observed = sample_reference_image_augmentation_parameters(
        camera_count=2,
        native_height=900,
        native_width=1600,
    )
    np.random.seed(417)
    rows = []
    for _ in range(2):
        resize = np.random.uniform(0.38, 0.55)
        resized = (int(1600 * resize), int(900 * resize))
        crop_top = int((1 - np.random.uniform(0.0, 0.0)) * resized[1]) - 256
        crop_left = int(np.random.uniform(0, max(0, resized[0] - 704)))
        flip = bool(np.random.choice([0, 1]))
        rotate = np.random.uniform(-5.4, 5.4)
        rows.append(
            [resize, resized[1], resized[0], crop_left, crop_top, flip, rotate]
        )
    assert torch.equal(observed, torch.tensor(rows, dtype=torch.float64))


def test_original_swin_key_mapping_is_complete_by_family():
    cases = {
        "patch_embed.proj.weight": "_swin_features.0.0.weight",
        "patch_embed.norm.bias": "_swin_features.0.2.bias",
        "layers.0.blocks.1.attn.qkv.weight": "_swin_features.1.1.attn.qkv.weight",
        "layers.2.blocks.5.mlp.fc1.bias": "_swin_features.5.5.mlp.0.bias",
        "layers.3.blocks.1.mlp.fc2.weight": "_swin_features.7.1.mlp.3.weight",
        "layers.1.downsample.reduction.weight": "_swin_features.4.reduction.weight",
        "norm.weight": None,
        "head.bias": None,
        "layers.0.blocks.1.attn_mask": None,
    }
    assert {
        key: original_swin_destination_key(key) for key in cases
    } == cases


def test_reference_camera_fpn_concat_shapes_and_stride8_selection():
    neck = Phase1GeneralizedLSSFPN().eval()
    with torch.no_grad():
        outputs = neck(
            (
                torch.zeros((1, 192, 32, 88)),
                torch.zeros((1, 384, 16, 44)),
                torch.zeros((1, 768, 8, 22)),
            )
        )
    assert [tuple(value.shape) for value in outputs] == [
        (1, 256, 32, 88),
        (1, 256, 16, 44),
    ]


def test_phase1_camera_topology_has_bn_and_only_swin_layernorm():
    torch.manual_seed(0)
    model = Phase1CameraDetector(pool_backend="fallback")
    assert model.camera_backbone.out_indices == (1, 2, 3)
    assert model.camera_backbone.output_layer_norm is True
    assert model.camera_backbone.activation_checkpoint is False
    assert model.view_transform.depth_bins == 118
    assert tuple(int(value) for value in model.view_transform.nx) == (256, 256, 1)
    assert model.head.class_names == (
        ("car",),
        ("truck", "construction_vehicle"),
        ("bus", "trailer"),
        ("barrier",),
        ("motorcycle", "bicycle"),
        ("pedestrian", "traffic_cone"),
    )
    assert not any(isinstance(module, torch.nn.GroupNorm) for module in model.modules())
    convolutional_bn = [
        module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)
    ]
    assert convolutional_bn
    assert all(module.eps == 1e-5 and module.momentum == 0.1 for module in convolutional_bn)


def test_phase1_reference_decode_restores_second_task_wide_topk():
    task_class_counts = (1, 2, 2, 1, 2, 2)
    outputs = []
    for classes in task_class_counts:
        outputs.append(
            {
                "heatmap": torch.zeros((1, classes, 128, 128)),
                "reg": torch.zeros((1, 2, 128, 128)),
                "height": torch.zeros((1, 1, 128, 128)),
                "dim": torch.zeros((1, 3, 128, 128)),
                "rot": torch.cat(
                    (
                        torch.zeros((1, 1, 128, 128)),
                        torch.ones((1, 1, 128, 128)),
                    ),
                    dim=1,
                ),
                "vel": torch.zeros((1, 2, 128, 128)),
            }
        )
    reference = select_task_candidates(
        outputs,
        config=CenterHeadDecodeConfig(task_pre_max=500),
    )[0]
    adapted = select_task_candidates(outputs)[0]
    assert [item["scores"].numel() for item in reference] == [500] * 6
    assert [item["scores"].numel() for item in adapted] == [500, 1000, 1000, 500, 1000, 1000]
