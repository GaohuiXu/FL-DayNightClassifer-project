"""Camera image geometry and ImageNet normalization.

The production S03 path is an aspect-preserving resize followed by crop/pad,
optional horizontal flip, and optional in-plane rotation.  Every operation is
represented by one pixel-space homography ``A`` and the exact same transform is
left-multiplied into both ``cam_intrinsics`` and ``lidar2img``::

    K_aug = A @ K
    lidar2img_aug = embed4(A) @ lidar2img

Pixel coordinates use integer pixel centres, ``u`` right and ``v`` down.  Resize
uses ``align_corners=False``; therefore its half-pixel translation is part of
``A``.  Flip uses ``u' = (W - 1) - u`` and rotation is around
``((W - 1) / 2, (H - 1) / 2)``.  These details make projected points and sampled
image content share one geometry rather than relying on an approximate scale.

``augmentation=None`` retains the pre-S03 anisotropic resize for callers that
have not yet been migrated.  S07-B must explicitly construct
``ImageAugmentationConfig`` for the O-017 reference-style path; this lets S03
publish the new module contract without editing ``detector.py`` or ``tasks.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet-1k RGB statistics (torchvision Swin_T IMAGENET1K_V1 transforms).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMAGE_HW = (256, 704)
AUGMENTATION_PARAM_FIELDS = (
    "resize",
    "resized_h",
    "resized_w",
    "crop_left",
    "crop_top",
    "flip",
    "rotate_degrees",
)


@dataclass(frozen=True)
class ImageAugmentationConfig:
    """MIT-BEVFusion-style 2D geometry with an explicit deterministic eval path.

    ``enabled`` controls random *training* augmentation.  Validation always uses
    ``validation_resize``, mean bottom crop, no flip, and no rotation.  A caller
    can also supply explicit per-camera parameters to :meth:`ImagePreprocessor.forward`
    for fixtures or replay.
    """

    enabled: bool = True
    resize_limits: Tuple[float, float] = (0.38, 0.55)
    validation_resize: float = 0.48
    bottom_crop_limits: Tuple[float, float] = (0.0, 0.0)
    rotation_limits_degrees: Tuple[float, float] = (-5.4, 5.4)
    random_flip: bool = True

    def __post_init__(self) -> None:
        r0, r1 = self.resize_limits
        b0, b1 = self.bottom_crop_limits
        a0, a1 = self.rotation_limits_degrees
        if not (0.0 < r0 <= r1 and self.validation_resize > 0.0):
            raise ValueError("resize factors must be positive and ordered")
        if not (0.0 <= b0 <= b1 <= 1.0):
            raise ValueError("bottom_crop_limits must lie in [0, 1]")
        if a0 > a1:
            raise ValueError("rotation_limits_degrees must be ordered")


def resize_affine(
    H_in: int, W_in: int, H_out: int, W_out: int, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return 3x3/4x4 half-pixel resize transforms.

    This helper remains the compatibility primitive for the legacy direct-resize
    path.  The aspect-preserving path composes the same affine with crop/flip/
    rotation in :func:`image_augmentation_affine`.
    """
    if min(H_in, W_in, H_out, W_out) <= 0:
        raise ValueError("image dimensions must be positive")
    sx = W_out / W_in
    sy = H_out / H_in
    tx = 0.5 * sx - 0.5
    ty = 0.5 * sy - 0.5
    A = torch.tensor(
        [[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    M = torch.eye(4, device=device, dtype=dtype)
    M[:3, :3] = A
    return A, M


def image_augmentation_affine(
    H_in: int,
    W_in: int,
    H_out: int,
    W_out: int,
    resized_h: int,
    resized_w: int,
    crop_left: int,
    crop_top: int,
    flip: bool,
    rotate_degrees: float,
    *,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Compose native-pixel -> augmented-pixel affine ``A`` exactly."""
    A_resize, _ = resize_affine(
        H_in, W_in, resized_h, resized_w, device=device, dtype=dtype
    )
    A_crop = torch.eye(3, device=device, dtype=dtype)
    A_crop[0, 2] = -float(crop_left)
    A_crop[1, 2] = -float(crop_top)

    A_flip = torch.eye(3, device=device, dtype=dtype)
    if flip:
        A_flip[0, 0] = -1.0
        A_flip[0, 2] = float(W_out - 1)

    theta = math.radians(float(rotate_degrees))
    co, si = math.cos(theta), math.sin(theta)
    # Positive angle follows MIT ImageAug3D/PIL in image coordinates (v down).
    R = torch.tensor([[co, si], [-si, co]], device=device, dtype=dtype)
    centre = torch.tensor(
        [(W_out - 1) / 2.0, (H_out - 1) / 2.0], device=device, dtype=dtype
    )
    A_rotate = torch.eye(3, device=device, dtype=dtype)
    A_rotate[:2, :2] = R
    A_rotate[:2, 2] = centre - R @ centre
    return A_rotate @ A_flip @ A_crop @ A_resize


def _uniform(low: float, high: float, generator: Optional[torch.Generator]) -> float:
    value = torch.rand((), generator=generator, device="cpu", dtype=torch.float64)
    return float(low + (high - low) * value.item())


def _sample_parameters(
    config: ImageAugmentationConfig,
    training: bool,
    B: int,
    N: int,
    H_in: int,
    W_in: int,
    H_out: int,
    W_out: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("image augmentation requires a CPU torch.Generator")
    rows = []
    random_train = bool(training and config.enabled)
    for _ in range(B * N):
        if random_train:
            resize = _uniform(*config.resize_limits, generator)
            resized_w = max(1, int(W_in * resize))
            resized_h = max(1, int(H_in * resize))
            bottom = _uniform(*config.bottom_crop_limits, generator)
            crop_top = int((1.0 - bottom) * resized_h) - H_out
            max_left = max(0, resized_w - W_out)
            crop_left = int(_uniform(0.0, float(max_left), generator))
            flip = bool(
                config.random_flip
                and int(torch.randint(0, 2, (), generator=generator, device="cpu").item())
            )
            rotate = _uniform(*config.rotation_limits_degrees, generator)
        else:
            resize = float(config.validation_resize)
            resized_w = max(1, int(W_in * resize))
            resized_h = max(1, int(H_in * resize))
            bottom = 0.5 * sum(config.bottom_crop_limits)
            crop_top = int((1.0 - bottom) * resized_h) - H_out
            max_left = max(0, resized_w - W_out)
            crop_left = int(max_left / 2)
            rotate = 0.0
            flip = False

        # Match the reference's scalar aspect-preserving resize choice.  The
        # realized sx/sy (after integer rounding) are used by calibration.
        rows.append(
            [
                resize,
                float(resized_h),
                float(resized_w),
                float(crop_left),
                float(crop_top),
                float(flip),
                rotate,
            ]
        )
    return torch.tensor(rows, dtype=torch.float64).view(B, N, len(AUGMENTATION_PARAM_FIELDS))


def sample_reference_image_augmentation_parameters(
    *,
    camera_count: int,
    native_height: int,
    native_width: int,
    output_height: int = 256,
    output_width: int = 704,
    resize_limits: Tuple[float, float] = (0.38, 0.55),
    bottom_crop_limits: Tuple[float, float] = (0.0, 0.0),
    rotation_limits_degrees: Tuple[float, float] = (-5.4, 5.4),
    random_flip: bool = True,
) -> torch.Tensor:
    """Sample MIT ``ImageAug3D`` parameters with its exact NumPy RNG order.

    This runs inside the DataLoader worker before scene-3D augmentation.  Merely
    delaying application of the sampled image transform until the model
    preprocessor is output-equivalent because image augmentation left-multiplies
    ``lidar2img`` while scene augmentation right-multiplies it.
    """
    if camera_count <= 0 or min(native_height, native_width, output_height, output_width) <= 0:
        raise ValueError("image augmentation dimensions and camera_count must be positive")
    rows = []
    for _ in range(int(camera_count)):
        resize = float(np.random.uniform(*resize_limits))
        resized_width = max(1, int(native_width * resize))
        resized_height = max(1, int(native_height * resize))
        bottom = float(np.random.uniform(*bottom_crop_limits))
        crop_top = int((1.0 - bottom) * resized_height) - int(output_height)
        maximum_left = max(0, resized_width - int(output_width))
        crop_left = int(np.random.uniform(0, maximum_left))
        flip = bool(random_flip and np.random.choice([0, 1]))
        rotation = float(np.random.uniform(*rotation_limits_degrees))
        rows.append(
            [
                resize,
                float(resized_height),
                float(resized_width),
                float(crop_left),
                float(crop_top),
                float(flip),
                rotation,
            ]
        )
    return torch.tensor(rows, dtype=torch.float64)


def _crop_or_pad(
    image: torch.Tensor, crop_left: int, crop_top: int, H_out: int, W_out: int
) -> torch.Tensor:
    """Crop a CHW image, zero-padding when the reference crop extends outside."""
    _, H, W = image.shape
    pad_left = max(-crop_left, 0)
    pad_top = max(-crop_top, 0)
    pad_right = max(crop_left + W_out - W, 0)
    pad_bottom = max(crop_top + H_out - H, 0)
    if pad_left or pad_right or pad_top or pad_bottom:
        image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom))
    left = crop_left + pad_left
    top = crop_top + pad_top
    return image[:, top : top + H_out, left : left + W_out]


def _rotate_image(image: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Apply only the rotation component of ``affine`` to an already-cropped CHW image."""
    C, H, W = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float64),
        torch.arange(W, device=image.device, dtype=torch.float64),
        indexing="ij",
    )
    out = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)
    inv = torch.linalg.inv(affine.to(device=image.device, dtype=torch.float64))
    src = out @ inv.T
    grid_x = (2.0 * (src[..., 0] + 0.5) / W) - 1.0
    grid_y = (2.0 * (src[..., 1] + 0.5) / H) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).to(image.dtype).unsqueeze(0)
    return F.grid_sample(
        image.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape(C, H, W)


class ImagePreprocessor(nn.Module):
    """Native RGB tensors + calibration -> normalized augmented camera tensors.

    Reference-style forward output:

    - ``images``: ``[B,N,3,H_out,W_out]`` float32;
    - ``cam_intrinsics`` / ``lidar2img``: input calibration dtype;
    - ``image_aug_matrix``: ``[B,N,3,3]`` calibration dtype;
    - ``augmentation_params``: ``[B,N,7]`` float64 and the stable field names in
      ``augmentation_param_fields``.
    """

    def __init__(
        self,
        image_hw: Tuple[int, int] = DEFAULT_IMAGE_HW,
        augmentation: Optional[ImageAugmentationConfig] = None,
    ):
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        if min(self.image_hw) <= 0:
            raise ValueError("image_hw must be positive")
        self.augmentation = augmentation
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)
        self._phase1p_augmentation_transfer_cleanup = False

    def set_phase1p_augmentation_transfer_cleanup(self, enabled: bool) -> None:
        """Enable the profiler-only, output-neutral augmentation transfer cleanup.

        The production/default path remains unchanged.  The candidate requires
        loader-sampled parameters to remain contiguous CPU float64 tensors, avoids
        cloning them, and omits the two diagnostic-only augmentation fields from
        the returned mapping.  Image/calibration tensors and all augmentation math
        are unchanged.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P augmentation transfer cleanup must be boolean")
        self._phase1p_augmentation_transfer_cleanup = enabled

    @property
    def geometry_mode(self) -> str:
        return "legacy_stretch" if self.augmentation is None else "aspect_preserving_reference"

    def output_contract(self) -> dict:
        config = self.augmentation
        return {
            "geometry_mode": self.geometry_mode,
            "output_image_hw": list(self.image_hw),
            "image_dtype": "torch.float32",
            "calibration_dtype": "preserve_input",
            "calibration_update": "K'=A@K; lidar2img'=embed4(A)@lidar2img",
            "augmentation_param_fields": list(AUGMENTATION_PARAM_FIELDS),
            "training_random_augmentation": bool(config is not None and config.enabled),
            "validation_deterministic": True,
            "resize_limits": list(config.resize_limits) if config is not None else None,
            "validation_resize": config.validation_resize if config is not None else None,
            "bottom_crop_limits": list(config.bottom_crop_limits) if config is not None else None,
            "rotation_limits_degrees": (
                list(config.rotation_limits_degrees) if config is not None else None
            ),
            "random_flip": config.random_flip if config is not None else False,
        }

    def _legacy_forward(
        self,
        images_u8: torch.Tensor,
        lidar2img: torch.Tensor,
        cam_intrinsics: torch.Tensor,
    ) -> dict:
        B, N, C, H_in, W_in = images_u8.shape
        H_out, W_out = self.image_hw
        x = images_u8.reshape(B * N, C, H_in, W_in).to(torch.float32) / 255.0
        x = F.interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=False)
        x = x.reshape(B, N, C, H_out, W_out)
        x = (x - self._mean) / self._std
        A, M = resize_affine(
            H_in, W_in, H_out, W_out, device=images_u8.device, dtype=torch.float64
        )
        A_batch = A.view(1, 1, 3, 3).expand(B, N, 3, 3)
        M_batch = M.view(1, 1, 4, 4).expand(B, N, 4, 4)
        K2 = (A_batch @ cam_intrinsics.to(torch.float64)).to(cam_intrinsics.dtype)
        l2i2 = (M_batch @ lidar2img.to(torch.float64)).to(lidar2img.dtype)
        return {
            "images": x.contiguous(),
            "cam_intrinsics": K2.contiguous(),
            "lidar2img": l2i2.contiguous(),
            "image_aug_matrix": A_batch.to(cam_intrinsics.dtype).contiguous(),
            "image_hw": (H_out, W_out),
            "geometry_mode": self.geometry_mode,
        }

    def forward(
        self,
        images_u8: torch.Tensor,
        lidar2img: torch.Tensor,
        cam_intrinsics: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        augmentation_params: Optional[torch.Tensor] = None,
    ) -> dict:
        if images_u8.ndim != 5 or images_u8.shape[2] != 3:
            raise ValueError("images_u8 must have shape [B,N,3,H,W]")
        B, N, C, H_in, W_in = images_u8.shape
        if lidar2img.shape != (B, N, 4, 4):
            raise ValueError(f"lidar2img must have shape {(B, N, 4, 4)}")
        if cam_intrinsics.shape != (B, N, 3, 3):
            raise ValueError(f"cam_intrinsics must have shape {(B, N, 3, 3)}")
        if not (lidar2img.is_floating_point() and cam_intrinsics.is_floating_point()):
            raise TypeError("calibration tensors must be floating point")
        if self.augmentation is None:
            if augmentation_params is not None:
                raise ValueError("augmentation_params require ImageAugmentationConfig")
            return self._legacy_forward(images_u8, lidar2img, cam_intrinsics)

        H_out, W_out = self.image_hw
        if augmentation_params is None:
            params = _sample_parameters(
                self.augmentation,
                self.training,
                B,
                N,
                H_in,
                W_in,
                H_out,
                W_out,
                generator,
            )
        else:
            expected = (B, N, len(AUGMENTATION_PARAM_FIELDS))
            if tuple(augmentation_params.shape) != expected:
                raise ValueError(f"augmentation_params must have shape {expected}")
            if self._phase1p_augmentation_transfer_cleanup:
                if augmentation_params.device.type != "cpu":
                    raise ValueError(
                        "Phase I-P augmentation transfer cleanup requires CPU parameters"
                    )
                if augmentation_params.dtype != torch.float64:
                    raise TypeError(
                        "Phase I-P augmentation transfer cleanup requires float64 parameters"
                    )
                if not augmentation_params.is_contiguous():
                    raise ValueError(
                        "Phase I-P augmentation transfer cleanup requires contiguous parameters"
                    )
                params = augmentation_params.detach()
            else:
                params = augmentation_params.detach().to(
                    device="cpu", dtype=torch.float64
                ).clone()
        if not torch.isfinite(params).all():
            raise ValueError("augmentation_params must be finite")

        images = []
        affines = []
        flat = images_u8.reshape(B * N, C, H_in, W_in)
        for idx in range(B * N):
            resize, resized_h_f, resized_w_f, left_f, top_f, flip_f, angle = params.view(-1, 7)[idx]
            if float(resize) <= 0:
                raise ValueError("resize must be positive")
            for field_name, value in (
                ("resized_h", resized_h_f),
                ("resized_w", resized_w_f),
                ("crop_left", left_f),
                ("crop_top", top_f),
                ("flip", flip_f),
            ):
                if float(value) != float(int(value)):
                    raise ValueError(f"{field_name} must be integer-valued")
            if int(flip_f) not in (0, 1):
                raise ValueError("flip must be 0 or 1")
            resized_h, resized_w = int(resized_h_f), int(resized_w_f)
            crop_left, crop_top = int(left_f), int(top_f)
            flip = bool(int(flip_f))
            if min(resized_h, resized_w) <= 0:
                raise ValueError("resized dimensions must be positive")

            image = flat[idx].to(torch.float32).unsqueeze(0) / 255.0
            image = F.interpolate(
                image,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            image = _crop_or_pad(image, crop_left, crop_top, H_out, W_out)
            if flip:
                image = torch.flip(image, dims=(-1,))

            A = image_augmentation_affine(
                H_in,
                W_in,
                H_out,
                W_out,
                resized_h,
                resized_w,
                crop_left,
                crop_top,
                flip,
                float(angle),
                device=images_u8.device,
                dtype=torch.float64,
            )
            if float(angle) != 0.0:
                # At this point resize/crop/flip are already applied.  Supply only
                # the final rotation to the inverse-sampling grid.
                A_before_rotation = image_augmentation_affine(
                    H_in,
                    W_in,
                    H_out,
                    W_out,
                    resized_h,
                    resized_w,
                    crop_left,
                    crop_top,
                    flip,
                    0.0,
                    device=images_u8.device,
                    dtype=torch.float64,
                )
                A_rotation = A @ torch.linalg.inv(A_before_rotation)
                image = _rotate_image(image, A_rotation)
            images.append(image)
            affines.append(A)

        x = torch.stack(images, dim=0).reshape(B, N, C, H_out, W_out)
        x = (x - self._mean) / self._std
        A_batch = torch.stack(affines, dim=0).reshape(B, N, 3, 3)
        M_batch = torch.eye(
            4, device=images_u8.device, dtype=torch.float64
        ).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        M_batch[:, :, :3, :3] = A_batch
        K2 = (A_batch @ cam_intrinsics.to(torch.float64)).to(cam_intrinsics.dtype)
        l2i2 = (M_batch @ lidar2img.to(torch.float64)).to(lidar2img.dtype)
        result = {
            "images": x.contiguous(),
            "cam_intrinsics": K2.contiguous(),
            "lidar2img": l2i2.contiguous(),
            "image_aug_matrix": A_batch.to(cam_intrinsics.dtype).contiguous(),
            "image_hw": (H_out, W_out),
            "geometry_mode": self.geometry_mode,
        }
        if not self._phase1p_augmentation_transfer_cleanup:
            result["augmentation_params"] = params.to(device=images_u8.device)
            result["augmentation_param_fields"] = AUGMENTATION_PARAM_FIELDS
        return result
