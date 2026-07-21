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

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class _DecodedAugmentationRow:
    resize: float
    resized_h: int
    resized_w: int
    crop_left: int
    crop_top: int
    flip: bool
    rotate_degrees: float


def _decode_augmentation_row(values: Sequence[object]) -> _DecodedAugmentationRow:
    if len(values) != len(AUGMENTATION_PARAM_FIELDS):
        raise ValueError("augmentation parameter row has the wrong length")
    resize, resized_h_f, resized_w_f, left_f, top_f, flip_f, angle = values
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
    if min(resized_h, resized_w) <= 0:
        raise ValueError("resized dimensions must be positive")
    return _DecodedAugmentationRow(
        resize=float(resize),
        resized_h=resized_h,
        resized_w=resized_w,
        crop_left=int(left_f),
        crop_top=int(top_f),
        flip=bool(int(flip_f)),
        rotate_degrees=float(angle),
    )


def _batched_image_augmentation_geometry(
    H_in: int,
    W_in: int,
    H_out: int,
    W_out: int,
    rows: Sequence[_DecodedAugmentationRow],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch the unchanged 3x3 affine/inverse operation sequence.

    Per-row scalar choices and ``math.sin``/``math.cos`` remain on the host, as in
    :func:`image_augmentation_affine`.  One packed float64 transfer supplies the
    primitive matrices; composition and both inversions retain the reference
    left-associated operation order but execute over the complete image batch.
    """
    if not rows:
        raise ValueError("batched augmentation geometry requires at least one row")
    scalar_rows = []
    for row in rows:
        sx = row.resized_w / W_in
        sy = row.resized_h / H_in
        theta = math.radians(row.rotate_degrees)
        co, si = math.cos(theta), math.sin(theta)
        scalar_rows.append(
            (
                sx,
                sy,
                0.5 * sx - 0.5,
                0.5 * sy - 0.5,
                -float(row.crop_left),
                -float(row.crop_top),
                -1.0 if row.flip else 1.0,
                float(W_out - 1) if row.flip else 0.0,
                co,
                si,
            )
        )
    scalars = torch.tensor(scalar_rows, dtype=torch.float64).to(device=device)
    count = len(rows)

    A_resize = torch.zeros((count, 3, 3), device=device, dtype=torch.float64)
    A_resize[:, 0, 0] = scalars[:, 0]
    A_resize[:, 1, 1] = scalars[:, 1]
    A_resize[:, 0, 2] = scalars[:, 2]
    A_resize[:, 1, 2] = scalars[:, 3]
    A_resize[:, 2, 2] = 1.0

    identity = torch.eye(3, device=device, dtype=torch.float64).expand(
        count, -1, -1
    )
    A_crop = identity.clone()
    A_crop[:, 0, 2] = scalars[:, 4]
    A_crop[:, 1, 2] = scalars[:, 5]
    A_flip = identity.clone()
    A_flip[:, 0, 0] = scalars[:, 6]
    A_flip[:, 0, 2] = scalars[:, 7]
    A_rotate = identity.clone()
    A_rotate[:, 0, 0] = scalars[:, 8]
    A_rotate[:, 0, 1] = scalars[:, 9]
    A_rotate[:, 1, 0] = -scalars[:, 9]
    A_rotate[:, 1, 1] = scalars[:, 8]
    centre = torch.tensor(
        [(W_out - 1) / 2.0, (H_out - 1) / 2.0],
        device=device,
        dtype=torch.float64,
    )
    A_rotate[:, :2, 2] = centre - torch.bmm(
        A_rotate[:, :2, :2],
        centre.view(1, 2, 1).expand(count, -1, -1),
    ).squeeze(-1)

    def compose(rotation: torch.Tensor) -> torch.Tensor:
        value = torch.bmm(rotation, A_flip)
        value = torch.bmm(value, A_crop)
        return torch.bmm(value, A_resize)

    affine = compose(A_rotate)
    before_rotation = compose(identity)
    rotation = torch.bmm(affine, torch.linalg.inv(before_rotation))
    inverse_rotation = torch.linalg.inv(rotation)
    return affine, inverse_rotation


def _bulk_uint8_to_float01(images_u8: torch.Tensor) -> torch.Tensor:
    """Convert one native-image batch with exactly one added float32 tensor."""
    if images_u8.dtype != torch.uint8:
        raise TypeError("bulk native-image conversion requires torch.uint8 input")
    return images_u8.to(dtype=torch.float32).div_(255.0)


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


def _rotation_output_coordinates(
    height: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float64),
        torch.arange(width, device=device, dtype=torch.float64),
        indexing="ij",
    )
    return torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)


def _rotate_image(
    image: torch.Tensor,
    affine: torch.Tensor,
    *,
    output_coordinates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply only the rotation component of ``affine`` to an already-cropped CHW image."""
    C, H, W = image.shape
    if output_coordinates is None:
        out = _rotation_output_coordinates(H, W, device=image.device)
    else:
        if output_coordinates.shape != (H, W, 3):
            raise ValueError("cached rotation coordinates have the wrong shape")
        if output_coordinates.device != image.device:
            raise ValueError("cached rotation coordinates have the wrong device")
        if output_coordinates.dtype != torch.float64:
            raise TypeError("cached rotation coordinates must be float64")
        out = output_coordinates
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
        self.register_buffer(
            "_phase1p_rotation_output_coordinates",
            torch.empty(0, dtype=torch.float64),
            persistent=False,
        )
        self._phase1p_augmentation_transfer_cleanup = False
        self._phase1p_static_grid_cache = False
        self._phase1p_batched_affine_grid = False
        self._phase1p_batched_preprocess = False
        self._phase1p_vectorized_geometry = False
        self._phase1p_bulk_input_conversion = False
        self._operator_profile_ranges = False

    @contextmanager
    def operator_profile_ranges(self):
        """Enable bounded preprocessing subranges for a short torch trace."""
        if self._operator_profile_ranges:
            raise RuntimeError("Camera preprocessing profiler ranges are already active")
        self._operator_profile_ranges = True
        try:
            yield self
        finally:
            self._operator_profile_ranges = False

    def _profile_range(self, name: str):
        if not self._operator_profile_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"fl_v3::camera_preprocess::{name}")

    def set_phase1p_augmentation_transfer_cleanup(self, enabled: bool) -> None:
        """Enable the profiler-only, output-neutral augmentation transfer cleanup.

        The production/default path remains unchanged.  The candidate requires
        loader-sampled parameters to remain contiguous CPU float64 tensors, avoids
        their GPU round trip, materializes one Python value table instead of
        repeatedly reading pinned-memory scalar tensors, and omits the two
        diagnostic-only augmentation fields from the returned mapping.
        Image/calibration tensors and all augmentation math are unchanged.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P augmentation transfer cleanup must be boolean")
        self._phase1p_augmentation_transfer_cleanup = enabled

    def set_phase1p_static_grid_cache(self, enabled: bool) -> None:
        """Enable the profiler-only fixed rotation-coordinate cache.

        The cached tensor is non-persistent and therefore absent from checkpoints.
        Only the fixed output coordinate basis is reused; per-image augmentation
        matrices, their inverse, sampling grids and interpolation remain unchanged.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P static grid cache control must be boolean")
        self._phase1p_static_grid_cache = enabled
        if enabled:
            height, width = self.image_hw
            coordinates = _rotation_output_coordinates(
                height,
                width,
                device=self._mean.device,
            )
        else:
            coordinates = torch.empty(
                0,
                device=self._mean.device,
                dtype=torch.float64,
            )
        self._phase1p_rotation_output_coordinates = coordinates

    def set_phase1p_batched_affine_grid(self, enabled: bool) -> None:
        """Batch only rotation-grid coordinate construction for profiling.

        Resize/crop/flip, per-image affine construction and inversion, and each
        image's ``grid_sample`` call retain the reference path.  The candidate
        shares one output-coordinate basis per microbatch and batches only the
        ``out @ inverse.T`` source-coordinate multiplication.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P batched affine/grid control must be boolean")
        self._phase1p_batched_affine_grid = enabled

    def set_phase1p_batched_preprocess(self, enabled: bool) -> None:
        """Batch the final rotation ``grid_sample`` for the scoped IP-E3 probe.

        This flag deliberately covers only the already-isolated rotation stage;
        resize/crop/flip and per-image affine construction remain unchanged.  The
        candidate requires both the shared non-persistent output-coordinate grid
        and batched affine/grid construction, then issues one ``grid_sample`` for
        all rotated images in the microbatch instead of one call per image.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P batched preprocess control must be boolean")
        if enabled and not (
            self._phase1p_static_grid_cache
            and self._phase1p_batched_affine_grid
        ):
            raise ValueError(
                "batched rotation grid_sample requires static grid and batched affine/grid"
            )
        self._phase1p_batched_preprocess = enabled

    def set_phase1p_vectorized_geometry(self, enabled: bool) -> None:
        """Batch the reference affine composition and inverse sequence.

        Resize/crop/pad/flip and interpolation remain per image.  The candidate
        requires conservative batched affine/grid so only the small-matrix host/
        launch fragmentation changes.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P vectorized geometry control must be boolean")
        if enabled and not self._phase1p_batched_affine_grid:
            raise ValueError(
                "vectorized geometry requires conservative batched affine/grid"
            )
        self._phase1p_vectorized_geometry = enabled

    def set_phase1p_bulk_input_conversion(self, enabled: bool) -> None:
        """Convert the complete native uint8 image batch once before resizing.

        The conditional IP-E4 candidate retains every per-image interpolation and
        geometry operation.  Its only added live tensor is the flattened native
        image batch in float32; in-place division avoids a second full-size
        temporary.
        """
        if not isinstance(enabled, bool):
            raise TypeError("Phase I-P bulk input conversion control must be boolean")
        if enabled and not self._phase1p_vectorized_geometry:
            raise ValueError(
                "bulk input conversion requires promoted vectorized geometry"
            )
        self._phase1p_bulk_input_conversion = enabled

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
        with self._profile_range("parameter_prepare"):
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
        rotation_image_indices: list[int] = []
        rotation_inverse_affines: list[torch.Tensor] = []
        flat = images_u8.reshape(B * N, C, H_in, W_in)
        bulk_float_images: torch.Tensor | None = None
        if self._phase1p_bulk_input_conversion:
            with self._profile_range("convert_resize"):
                bulk_float_images = _bulk_uint8_to_float01(flat)
        parameter_rows = params.view(-1, 7)
        if self._phase1p_augmentation_transfer_cleanup:
            # DataLoader pinning applies recursively to augmentation_params.  A
            # Python loop over individual scalar tensors then performs poorly on
            # pinned host memory.  Convert the tiny B*N x 7 block once; Python
            # float conversion is exact for the required float64 values and the
            # downstream resize/affine/grid_sample path remains unchanged.
            parameter_rows = parameter_rows.tolist()
        vectorized_rows: list[_DecodedAugmentationRow] | None = None
        vectorized_affines: torch.Tensor | None = None
        vectorized_inverse_affines: torch.Tensor | None = None
        if self._phase1p_vectorized_geometry:
            with self._profile_range("geometry"):
                vectorized_rows = [
                    _decode_augmentation_row(parameter_rows[idx])
                    for idx in range(B * N)
                ]
                vectorized_affines, vectorized_inverse_affines = (
                    _batched_image_augmentation_geometry(
                        H_in,
                        W_in,
                        H_out,
                        W_out,
                        vectorized_rows,
                        device=images_u8.device,
                    )
                )
        for idx in range(B * N):
            if vectorized_rows is None:
                resize, resized_h_f, resized_w_f, left_f, top_f, flip_f, angle = (
                    parameter_rows[idx]
                )
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
                angle_value = float(angle)
                if min(resized_h, resized_w) <= 0:
                    raise ValueError("resized dimensions must be positive")
            else:
                row = vectorized_rows[idx]
                resized_h, resized_w = row.resized_h, row.resized_w
                crop_left, crop_top = row.crop_left, row.crop_top
                flip = row.flip
                angle_value = row.rotate_degrees

            with self._profile_range("convert_resize"):
                if bulk_float_images is None:
                    image = flat[idx].to(torch.float32).unsqueeze(0) / 255.0
                else:
                    image = bulk_float_images[idx].unsqueeze(0)
                image = F.interpolate(
                    image,
                    size=(resized_h, resized_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            with self._profile_range("crop_pad_flip"):
                image = _crop_or_pad(image, crop_left, crop_top, H_out, W_out)
                if flip:
                    image = torch.flip(image, dims=(-1,))

            A_rotation = None
            with self._profile_range("geometry"):
                if vectorized_affines is not None:
                    if vectorized_inverse_affines is None:
                        raise RuntimeError("vectorized inverse geometry is absent")
                    A = vectorized_affines[idx]
                    if angle_value != 0.0:
                        rotation_image_indices.append(idx)
                        rotation_inverse_affines.append(
                            vectorized_inverse_affines[idx]
                        )
                else:
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
                        angle_value,
                        device=images_u8.device,
                        dtype=torch.float64,
                    )
                if angle_value != 0.0 and vectorized_affines is None:
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
                    if self._phase1p_batched_affine_grid:
                        rotation_image_indices.append(idx)
                        rotation_inverse_affines.append(
                            torch.linalg.inv(
                                A_rotation.to(
                                    device=image.device,
                                    dtype=torch.float64,
                                )
                            )
                        )
            if A_rotation is not None and not self._phase1p_batched_affine_grid:
                with self._profile_range("rotation_grid_sample"):
                    output_coordinates = (
                        self._phase1p_rotation_output_coordinates
                        if self._phase1p_static_grid_cache
                        else None
                    )
                    image = _rotate_image(
                        image,
                        A_rotation,
                        output_coordinates=output_coordinates,
                    )
            images.append(image)
            affines.append(A)

        with self._profile_range("rotation_grid_sample"):
            if rotation_image_indices:
                if len(rotation_image_indices) != len(rotation_inverse_affines):
                    raise RuntimeError("batched rotation bookkeeping drift")
                output_coordinates = (
                    self._phase1p_rotation_output_coordinates
                    if self._phase1p_static_grid_cache
                    else _rotation_output_coordinates(
                        H_out,
                        W_out,
                        device=images_u8.device,
                    )
                )
                if vectorized_inverse_affines is None:
                    inverse_affines = torch.stack(rotation_inverse_affines, dim=0)
                else:
                    rotation_indices = torch.tensor(
                        rotation_image_indices,
                        device=images_u8.device,
                        dtype=torch.long,
                    )
                    inverse_affines = vectorized_inverse_affines.index_select(
                        0, rotation_indices
                    )
                src = (
                    output_coordinates.reshape(1, H_out * W_out, 3)
                    @ inverse_affines.transpose(1, 2)
                ).reshape(len(rotation_image_indices), H_out, W_out, 3)
                grid_x = (2.0 * (src[..., 0] + 0.5) / W_out) - 1.0
                grid_y = (2.0 * (src[..., 1] + 0.5) / H_out) - 1.0
                grids = torch.stack((grid_x, grid_y), dim=-1).to(images[0].dtype)
                if self._phase1p_batched_preprocess:
                    rotated_images = F.grid_sample(
                        torch.stack(
                            [images[image_index] for image_index in rotation_image_indices],
                            dim=0,
                        ),
                        grids,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    if tuple(rotated_images.shape) != (
                        len(rotation_image_indices),
                        C,
                        H_out,
                        W_out,
                    ):
                        raise RuntimeError("batched rotation grid_sample shape drift")
                    for grid_index, image_index in enumerate(rotation_image_indices):
                        images[image_index] = rotated_images[grid_index]
                else:
                    for grid_index, image_index in enumerate(rotation_image_indices):
                        image = images[image_index]
                        images[image_index] = F.grid_sample(
                            image.unsqueeze(0),
                            grids[grid_index].unsqueeze(0),
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=False,
                        ).reshape(C, H_out, W_out)

        with self._profile_range("stack_normalize"):
            x = torch.stack(images, dim=0).reshape(B, N, C, H_out, W_out)
            x = (x - self._mean) / self._std
        with self._profile_range("calibration_update"):
            if vectorized_affines is None:
                A_batch = torch.stack(affines, dim=0).reshape(B, N, 3, 3)
            else:
                A_batch = vectorized_affines.reshape(B, N, 3, 3)
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
