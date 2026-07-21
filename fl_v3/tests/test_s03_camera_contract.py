"""S03 camera architecture, geometry, independence, and memory fixtures."""
from __future__ import annotations

from contextlib import nullcontext
import inspect
import math

import pytest
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.camera_backbone import CameraBackbone
from fl_v3.models.fusion.camera_neck import GeneralizedLSSFPN
from fl_v3.models.fusion.preprocess import (
    AUGMENTATION_PARAM_FIELDS,
    ImageAugmentationConfig,
    ImagePreprocessor,
)
from fl_v3.models.fusion.view_transform import DepthLSSTransform


def _calibration(dtype=torch.float64):
    lidar2img = torch.tensor(
        [
            [12.0, 0.0, 7.0, 0.0],
            [0.0, 10.0, 5.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    ).view(1, 1, 4, 4)
    return lidar2img, lidar2img[:, :, :3, :3].clone()


@pytest.mark.parametrize(
    "crop_left,crop_top,flip,angle",
    [
        (2, 3, False, 0.0),
        (2, 3, True, 0.0),
        (2, 3, False, 11.0),
        (-2, 5, True, -5.4),
    ],
)
def test_s03_projection_residual_for_every_image_transform(
    crop_left, crop_top, flip, angle
):
    """Native and augmented projections agree with the exact composed homography."""
    H_in, W_in = 20, 32
    H_out, W_out = 12, 20
    images = torch.arange(3 * H_in * W_in, dtype=torch.int64)
    images = images.remainder(256).to(torch.uint8).view(1, 1, 3, H_in, W_in)
    lidar2img, K = _calibration()
    pre = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    ).eval()
    params = torch.tensor(
        [[[[0.75, 15.0, 24.0, crop_left, crop_top, float(flip), angle]]]],
        dtype=torch.float64,
    ).reshape(1, 1, len(AUGMENTATION_PARAM_FIELDS))
    out = pre(images, lidar2img, K, augmentation_params=params)

    points = torch.tensor(
        [[1.0, 0.5, 4.0, 1.0], [2.0, -0.25, 5.0, 1.0], [-0.5, 1.0, 3.0, 1.0]],
        dtype=torch.float64,
    )
    native_h = points @ lidar2img[0, 0].T
    native_uv = native_h[:, :2] / native_h[:, 2:3]
    # Independent scalar fixture: do not reuse the implementation homography to
    # calculate the expected point locations.
    expected_u = (native_uv[:, 0] + 0.5) * (24.0 / W_in) - 0.5 - crop_left
    expected_v = (native_uv[:, 1] + 0.5) * (15.0 / H_in) - 0.5 - crop_top
    if flip:
        expected_u = (W_out - 1) - expected_u
    centre_u, centre_v = (W_out - 1) / 2.0, (H_out - 1) / 2.0
    co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    du, dv = expected_u - centre_u, expected_v - centre_v
    expected_uv = torch.stack(
        (co * du + si * dv + centre_u, -si * du + co * dv + centre_v), dim=1
    )
    actual_h = points @ out["lidar2img"][0, 0].T
    actual_uv = actual_h[:, :2] / actual_h[:, 2:3]
    residual = (actual_uv - expected_uv).abs().max()
    assert float(residual) < 1e-10
    native_uv1 = torch.cat(
        (native_uv, torch.ones((points.shape[0], 1), dtype=torch.float64)), dim=1
    )
    affine_uv = native_uv1 @ out["image_aug_matrix"][0, 0].T
    assert torch.allclose(affine_uv[:, :2], expected_uv, atol=1e-10, rtol=0.0)
    assert torch.allclose(
        out["cam_intrinsics"][0, 0],
        out["image_aug_matrix"][0, 0] @ K[0, 0],
        atol=1e-12,
        rtol=0.0,
    )


def test_s10_phase1p_augmentation_transfer_cleanup_is_output_neutral():
    H_in, W_in = 20, 32
    H_out, W_out = 12, 20
    images = torch.arange(2 * 3 * H_in * W_in, dtype=torch.int64)
    images = images.remainder(256).to(torch.uint8).view(1, 2, 3, H_in, W_in)
    lidar2img, K = _calibration()
    lidar2img = lidar2img.repeat(1, 2, 1, 1)
    K = K.repeat(1, 2, 1, 1)
    params = torch.tensor(
        [
            [0.75, 15.0, 24.0, 2.0, 3.0, 0.0, 11.0],
            [0.75, 15.0, 24.0, -2.0, 5.0, 1.0, -5.4],
        ],
        dtype=torch.float64,
    ).view(1, 2, len(AUGMENTATION_PARAM_FIELDS))
    params_before = params.clone()

    reference = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    ).eval()
    candidate = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    ).eval()
    candidate.set_phase1p_augmentation_transfer_cleanup(True)

    expected = reference(images, lidar2img, K, augmentation_params=params)
    actual = candidate(images, lidar2img, K, augmentation_params=params)
    for key in ("images", "lidar2img", "cam_intrinsics", "image_aug_matrix"):
        assert torch.equal(actual[key], expected[key]), key
    assert actual["image_hw"] == expected["image_hw"]
    assert actual["geometry_mode"] == expected["geometry_mode"]
    assert "augmentation_params" not in actual
    assert "augmentation_param_fields" not in actual
    assert torch.equal(params, params_before)

    if torch.cuda.is_available():
        pinned_params = params.pin_memory()
        assert pinned_params.is_pinned()
        pinned_actual = candidate(
            images,
            lidar2img,
            K,
            augmentation_params=pinned_params,
        )
        for key in ("images", "lidar2img", "cam_intrinsics", "image_aug_matrix"):
            assert torch.equal(pinned_actual[key], expected[key]), key
        assert torch.equal(pinned_params, params_before)

    with pytest.raises(TypeError, match="float64"):
        candidate(
            images,
            lidar2img,
            K,
            augmentation_params=params.to(torch.float32),
        )


def test_s10_phase1p_static_grid_cache_is_output_neutral_and_nonpersistent():
    H_in, W_in = 20, 32
    H_out, W_out = 12, 20
    images = torch.arange(2 * 3 * H_in * W_in, dtype=torch.int64)
    images = images.remainder(256).to(torch.uint8).view(1, 2, 3, H_in, W_in)
    lidar2img, K = _calibration()
    lidar2img = lidar2img.repeat(1, 2, 1, 1)
    K = K.repeat(1, 2, 1, 1)
    params = torch.tensor(
        [
            [0.75, 15.0, 24.0, 2.0, 3.0, 0.0, 11.0],
            [0.75, 15.0, 24.0, -2.0, 5.0, 1.0, -5.4],
        ],
        dtype=torch.float64,
    ).view(1, 2, len(AUGMENTATION_PARAM_FIELDS))

    reference = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    ).eval()
    candidate = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    ).eval()
    candidate.set_phase1p_static_grid_cache(True)
    coordinates = candidate._phase1p_rotation_output_coordinates
    assert coordinates.shape == (H_out, W_out, 3)
    assert coordinates.dtype == torch.float64
    assert "_phase1p_rotation_output_coordinates" not in candidate.state_dict()
    coordinates_pointer = coordinates.data_ptr()

    expected = reference(images, lidar2img, K, augmentation_params=params)
    for _ in range(2):
        actual = candidate(images, lidar2img, K, augmentation_params=params)
        assert actual.keys() == expected.keys()
        for key in (
            "images",
            "lidar2img",
            "cam_intrinsics",
            "image_aug_matrix",
            "augmentation_params",
        ):
            assert torch.equal(actual[key], expected[key]), key
        assert actual["augmentation_param_fields"] == expected[
            "augmentation_param_fields"
        ]
        assert candidate._phase1p_rotation_output_coordinates.data_ptr() == (
            coordinates_pointer
        )

    candidate.set_phase1p_static_grid_cache(False)
    assert candidate._phase1p_rotation_output_coordinates.numel() == 0


def test_s10_phase1p_batched_affine_grid_is_elementwise_exact_on_cpu_and_cuda():
    H_in, W_in = 20, 32
    H_out, W_out = 12, 20
    base_images = torch.arange(3 * 3 * H_in * W_in, dtype=torch.int64)
    base_images = base_images.remainder(256).to(torch.uint8).view(
        1, 3, 3, H_in, W_in
    )
    base_lidar2img, base_K = _calibration()
    base_lidar2img = base_lidar2img.repeat(1, 3, 1, 1)
    base_K = base_K.repeat(1, 3, 1, 1)
    base_params = torch.tensor(
        [
            [0.75, 15.0, 24.0, 2.0, 3.0, 0.0, 11.0],
            [0.75, 15.0, 24.0, -2.0, 5.0, 1.0, -5.4],
            [0.75, 15.0, 24.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    ).view(1, 3, len(AUGMENTATION_PARAM_FIELDS))

    def check(device: torch.device) -> None:
        reference = ImagePreprocessor(
            (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
        ).to(device).eval()
        candidate = ImagePreprocessor(
            (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
        ).to(device).eval()
        candidate.set_phase1p_batched_affine_grid(True)
        images = base_images.to(device)
        lidar2img = base_lidar2img.to(device)
        intrinsics = base_K.to(device)
        params = base_params.to(device)
        expected = reference(
            images,
            lidar2img,
            intrinsics,
            augmentation_params=params,
        )
        actual = candidate(
            images,
            lidar2img,
            intrinsics,
            augmentation_params=params,
        )
        assert actual.keys() == expected.keys()
        for key in (
            "images",
            "lidar2img",
            "cam_intrinsics",
            "image_aug_matrix",
            "augmentation_params",
        ):
            assert torch.equal(actual[key], expected[key]), (device, key)
        assert actual["augmentation_param_fields"] == expected[
            "augmentation_param_fields"
        ]

    check(torch.device("cpu"))
    if torch.cuda.is_available():
        check(torch.device("cuda", 0))


def test_s10_phase1p_vectorized_geometry_preserves_output_policy_and_batches_inverses(
    monkeypatch,
):
    H_in, W_in = 20, 32
    H_out, W_out = 12, 20
    base_images = torch.arange(3 * 3 * H_in * W_in, dtype=torch.int64)
    base_images = base_images.remainder(256).to(torch.uint8).view(
        1, 3, 3, H_in, W_in
    )
    base_lidar2img, base_K = _calibration()
    base_lidar2img = base_lidar2img.repeat(1, 3, 1, 1)
    base_K = base_K.repeat(1, 3, 1, 1)
    base_params = torch.tensor(
        [
            [0.75, 15.0, 24.0, 2.0, 3.0, 0.0, 11.0],
            [0.75, 15.0, 24.0, -2.0, 5.0, 1.0, -5.4],
            [0.75, 15.0, 24.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    ).view(1, 3, len(AUGMENTATION_PARAM_FIELDS))

    incomplete = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    )
    with pytest.raises(ValueError, match="requires conservative"):
        incomplete.set_phase1p_vectorized_geometry(True)

    def check(device: torch.device) -> None:
        reference = ImagePreprocessor(
            (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
        ).to(device).eval()
        reference.set_phase1p_batched_affine_grid(True)
        candidate = ImagePreprocessor(
            (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
        ).to(device).eval()
        candidate.set_phase1p_batched_affine_grid(True)
        candidate.set_phase1p_vectorized_geometry(True)
        assert tuple(candidate.state_dict()) == tuple(reference.state_dict())

        images = base_images.to(device)
        lidar2img = base_lidar2img.to(device)
        intrinsics = base_K.to(device)
        params = base_params.to(device)
        expected = reference(
            images,
            lidar2img,
            intrinsics,
            augmentation_params=params,
        )

        original_inv = torch.linalg.inv
        inverse_shapes = []

        def counted_inv(value, *args, **kwargs):
            inverse_shapes.append(tuple(value.shape))
            return original_inv(value, *args, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(torch.linalg, "inv", counted_inv)
            actual = candidate(
                images,
                lidar2img,
                intrinsics,
                augmentation_params=params,
            )
        assert inverse_shapes == [(3, 3, 3), (3, 3, 3)]
        assert actual.keys() == expected.keys()
        torch.testing.assert_close(
            actual["images"], expected["images"], rtol=0.002, atol=0.0002
        )
        for key in ("lidar2img", "cam_intrinsics", "image_aug_matrix"):
            torch.testing.assert_close(
                actual[key], expected[key], rtol=1e-4, atol=1e-6
            )
        assert torch.equal(
            actual["augmentation_params"], expected["augmentation_params"]
        )
        assert actual["augmentation_param_fields"] == expected[
            "augmentation_param_fields"
        ]

    check(torch.device("cpu"))
    if torch.cuda.is_available():
        check(torch.device("cuda", 0))


def test_s10_phase1p_batched_rotation_grid_sample_is_output_neutral_and_single_call(
    monkeypatch,
):
    H_in, W_in = 20, 32
    H_out, W_out = 12, 20
    base_images = torch.arange(3 * 3 * H_in * W_in, dtype=torch.int64)
    base_images = base_images.remainder(256).to(torch.uint8).view(
        1, 3, 3, H_in, W_in
    )
    base_lidar2img, base_K = _calibration()
    base_lidar2img = base_lidar2img.repeat(1, 3, 1, 1)
    base_K = base_K.repeat(1, 3, 1, 1)
    base_params = torch.tensor(
        [
            [0.75, 15.0, 24.0, 2.0, 3.0, 0.0, 11.0],
            [0.75, 15.0, 24.0, -2.0, 5.0, 1.0, -5.4],
            [0.75, 15.0, 24.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    ).view(1, 3, len(AUGMENTATION_PARAM_FIELDS))

    def check(device: torch.device) -> None:
        reference = ImagePreprocessor(
            (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
        ).to(device).eval()
        candidate = ImagePreprocessor(
            (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
        ).to(device).eval()
        candidate.set_phase1p_static_grid_cache(True)
        candidate.set_phase1p_batched_affine_grid(True)
        candidate.set_phase1p_batched_preprocess(True)
        assert "_phase1p_rotation_output_coordinates" not in candidate.state_dict()

        images = base_images.to(device)
        lidar2img = base_lidar2img.to(device)
        intrinsics = base_K.to(device)
        params = base_params.to(device)
        expected = reference(
            images,
            lidar2img,
            intrinsics,
            augmentation_params=params,
        )

        original_grid_sample = torch.nn.functional.grid_sample
        calls = []

        def counted_grid_sample(input_tensor, grid, **kwargs):
            calls.append((tuple(input_tensor.shape), tuple(grid.shape)))
            return original_grid_sample(input_tensor, grid, **kwargs)

        with monkeypatch.context() as scoped:
            scoped.setattr(
                torch.nn.functional,
                "grid_sample",
                counted_grid_sample,
            )
            actual = candidate(
                images,
                lidar2img,
                intrinsics,
                augmentation_params=params,
            )
        assert calls == [((2, 3, H_out, W_out), (2, H_out, W_out, 2))]
        assert actual.keys() == expected.keys()
        torch.testing.assert_close(
            actual["images"],
            expected["images"],
            rtol=0.002,
            atol=0.0002,
        )
        for key in (
            "lidar2img",
            "cam_intrinsics",
            "image_aug_matrix",
            "augmentation_params",
        ):
            assert torch.equal(actual[key], expected[key]), (device, key)
        assert actual["augmentation_param_fields"] == expected[
            "augmentation_param_fields"
        ]

    incomplete = ImagePreprocessor(
        (H_out, W_out), augmentation=ImageAugmentationConfig(enabled=True)
    )
    with pytest.raises(ValueError, match="requires static grid"):
        incomplete.set_phase1p_batched_preprocess(True)

    check(torch.device("cpu"))
    if torch.cuda.is_available():
        check(torch.device("cuda", 0))


def test_s03_validation_geometry_is_deterministic_and_training_is_seed_replayable():
    images = torch.arange(2 * 3 * 20 * 32, dtype=torch.int64)
    images = images.remainder(256).to(torch.uint8).view(1, 2, 3, 20, 32)
    lidar2img, K = _calibration(torch.float32)
    lidar2img = lidar2img.repeat(1, 2, 1, 1)
    K = K.repeat(1, 2, 1, 1)
    pre = ImagePreprocessor(
        (12, 20), augmentation=ImageAugmentationConfig(enabled=True)
    )

    pre.eval()
    first = pre(images, lidar2img, K)
    second = pre(images, lidar2img, K)
    for key in ("images", "lidar2img", "cam_intrinsics", "image_aug_matrix", "augmentation_params"):
        assert torch.equal(first[key], second[key]), key
    assert first["augmentation_params"][..., 0].tolist() == [[0.48, 0.48]]
    assert first["augmentation_params"][..., 5:].tolist() == [[[0.0, 0.0], [0.0, 0.0]]]

    pre.train()
    g1 = torch.Generator(device="cpu").manual_seed(1703)
    g2 = torch.Generator(device="cpu").manual_seed(1703)
    seeded_a = pre(images, lidar2img, K, generator=g1)
    seeded_b = pre(images, lidar2img, K, generator=g2)
    assert torch.equal(seeded_a["augmentation_params"], seeded_b["augmentation_params"])
    assert torch.equal(seeded_a["images"], seeded_b["images"])
    assert torch.equal(seeded_a["lidar2img"], seeded_b["lidar2img"])


def test_s03_native_nuscenes_validation_geometry_matches_reference_fixture():
    images = torch.zeros((1, 1, 3, 900, 1600), dtype=torch.uint8)
    lidar2img, K = _calibration(torch.float32)
    pre = ImagePreprocessor(
        (256, 704), augmentation=ImageAugmentationConfig(enabled=True)
    ).eval()
    out = pre(images, lidar2img, K)
    params = out["augmentation_params"][0, 0]
    assert params.tolist() == [0.48, 432.0, 768.0, 32.0, 176.0, 0.0, 0.0]
    assert out["images"].shape == (1, 1, 3, 256, 704)
    contract = pre.output_contract()
    assert contract["geometry_mode"] == "aspect_preserving_reference"
    assert contract["validation_deterministic"] is True
    assert contract["resize_limits"] == [0.38, 0.55]


def test_s03_fpn_consumes_every_declared_level_with_finite_gradients():
    neck = GeneralizedLSSFPN(
        in_channels=[4, 8, 16, 32],
        in_strides=[4, 8, 16, 32],
        out_channels=8,
        out_stride=8,
    )
    feats = [
        torch.randn(2, 4, 16, 24, requires_grad=True),
        torch.randn(2, 8, 8, 12, requires_grad=True),
        torch.randn(2, 16, 4, 6, requires_grad=True),
        torch.randn(2, 32, 2, 3, requires_grad=True),
    ]
    out = neck(feats)
    assert out.shape == (2, 8, 8, 12)
    out.square().mean().backward()
    for level, feat in enumerate(feats):
        assert feat.grad is not None, level
        assert torch.isfinite(feat.grad).all(), level
        assert torch.count_nonzero(feat.grad) > 0, level
    for name, parameter in neck.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert neck.output_contract()["all_levels_consumed"] is True


def _small_view_transform() -> DepthLSSTransform:
    cfg = BEVConfig(
        point_cloud_range=(0.0, 0.0, 0.0, 32.0, 32.0, 8.0),
        bev_voxel=(2.0, 2.0),
        out_size_factor=2,
    )
    return DepthLSSTransform(
        in_channels=4,
        context_channels=3,
        depth_bins=(1.0, 3.0, 0.5),
        image_hw=(16, 24),
        feat_stride=8,
        cfg=cfg,
        bev_output_dtype="input",
    )


def test_s03_pure_camera_pixel_sensitivity_lidar_invariance_and_dtype_contract():
    vt = _small_view_transform().eval()
    assert "lidar_points" not in inspect.signature(vt.forward).parameters
    assert vt.requires_lidar_input is False
    lidar2img = torch.eye(4).view(1, 1, 4, 4)
    feature = torch.randn(1, 4, 2, 3)
    with pytest.raises(TypeError):
        vt(feature, lidar2img, B=1, N=1, lidar_points=torch.zeros(1, 6))

    old_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.deterministic = True
        base = vt(feature, lidar2img, B=1, N=1)
        repeated = vt(feature, lidar2img, B=1, N=1)
        changed_feature = feature.clone()
        changed_feature[..., 0, 0] += 1.0
        changed_pixels = vt(changed_feature, lidar2img, B=1, N=1)
        independent_a = vt(feature, lidar2img, B=1, N=1)["bev"]
        independent_b = vt(feature, lidar2img, B=1, N=1)["bev"]
    finally:
        torch.backends.cudnn.deterministic = old_deterministic

    assert torch.equal(base["bev"], repeated["bev"])
    assert not torch.equal(base["bev"], changed_pixels["bev"])
    # Two unrelated LiDAR tensors cannot affect the result because neither enters
    # the camera API.  This pins the modality boundary for S07-B.
    lidar_a = torch.randn(17, 6)
    lidar_b = torch.randn(31, 6)
    assert not torch.equal(lidar_a, lidar_b[:17])
    assert torch.equal(independent_a, independent_b)

    contract = vt.output_contract(torch.float16)
    assert contract["input_stride"] == 8
    assert contract["depth_bins"] == [1.0, 3.0, 0.5]
    assert contract["depth_bin_count"] == 4
    assert contract["bev_output_dtype_policy"] == "input"
    assert contract["example_bev_dtype"] == "torch.float16"
    assert contract["requires_lidar_input"] is False
    assert base["bev"].shape == (1, 3, 16, 16)
    assert base["bev"].dtype == feature.dtype


def test_s03_stride8_half_meter_lift_memory_multiplier_is_explicit():
    primary = DepthLSSTransform(
        in_channels=8,
        context_channels=80,
        depth_bins=(1.0, 60.0, 0.5),
        image_hw=(256, 704),
        feat_stride=8,
    )
    legacy = DepthLSSTransform(
        in_channels=8,
        context_channels=80,
        depth_bins=(1.0, 60.0, 1.0),
        image_hw=(256, 704),
        feat_stride=16,
    )
    primary_elements = primary.theoretical_lift_elements(B=1, N=6)
    legacy_elements = legacy.theoretical_lift_elements(B=1, N=6)
    assert primary.D == 118
    assert (primary.fH, primary.fW) == (32, 88)
    assert primary_elements == 6 * 80 * 118 * 32 * 88
    assert primary_elements == 8 * legacy_elements


def test_s03_swin_camera_path_every_intended_parameter_has_finite_gradient():
    """Dependency-complete gate: Swin taps -> all-level FPN -> pure-camera LSS."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def amp_context():
        return (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )

    backbone = CameraBackbone("swin_t", frozen=False, pretrained=False).eval().to(device)
    neck = GeneralizedLSSFPN(
        backbone.out_channels,
        backbone.strides,
        out_channels=16,
        out_stride=8,
    ).eval().to(device)
    cfg = BEVConfig(
        point_cloud_range=(0.0, 0.0, 0.0, 64.0, 64.0, 8.0),
        bev_voxel=(4.0, 4.0),
        out_size_factor=2,
    )
    vt = DepthLSSTransform(
        in_channels=16,
        context_channels=4,
        depth_bins=(1.0, 3.0, 0.5),
        image_hw=(64, 96),
        feat_stride=8,
        cfg=cfg,
        bev_output_dtype="input",
    ).eval().to(device)
    image = torch.randn(1, 3, 64, 96, device=device)
    calibration = torch.eye(4, device=device).view(1, 1, 4, 4)
    with amp_context():
        features = backbone(image)
        assert [tuple(f.shape[-2:]) for f in features] == [(16, 24), (8, 12), (4, 6), (2, 3)]
        camera_feature = neck(features)
        result = vt(camera_feature, calibration, B=1, N=1)
    loss = camera_feature.float().square().mean() + result["bev"].float().square().mean()
    loss.backward()

    for module_name, module in (("backbone", backbone), ("neck", neck), ("view_transform", vt)):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None, f"{module_name}.{name}"
            assert torch.isfinite(parameter.grad).all(), f"{module_name}.{name}"

    changed_image = image.detach().clone()
    changed_image[..., 17, 29] += 8.0
    old_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.backends.cudnn.deterministic = True
        with torch.no_grad():
            with amp_context():
                base_bev = vt(
                    neck(backbone(image.detach())), calibration, B=1, N=1
                )["bev"]
                changed_bev = vt(
                    neck(backbone(changed_image)), calibration, B=1, N=1
                )["bev"]
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
    assert not torch.equal(base_bev, changed_bev)
