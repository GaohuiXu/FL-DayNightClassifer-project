"""S10 Phase-I standalone Camera candidate.

This is the mmdet/mmcv-free realization of the camera graph frozen by O-144:
Swin-T -> GeneralizedLSSFPN -> LSS -> GeneralizedResNet/LSSFPN -> six-task
CenterHead.  Convolutional normalization is BatchNorm; Swin retains LayerNorm.

The pinned MIT implementation stores the two symmetric BEV axes in its own
internal order.  This port keeps the project's already-frozen physical layout
``[B,C,H=y,W=x]``.  The pooling reduction itself is unchanged and both the
fallback and optimized backends consume integer coordinates ``[x,y,z,b]``.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_v3.config.phase1 import (
    FROZEN_CAMERA_MODEL,
    FROZEN_CAMERA_MODEL_V2,
    PHASE1_SCHEMA_V2,
    REFERENCE_OBJECT_CLASSES,
)
from fl_v3.models.fusion.bev_grid import BEVConfig
from fl_v3.models.fusion.camera_backbone import CameraBackbone
from fl_v3.models.fusion.centerhead_decode import (
    CenterHeadDecodeConfig,
    decode_centerhead,
)
from fl_v3.models.fusion.head import CenterPointHead, NUSCENES_CENTERHEAD_TASKS
from fl_v3.models.fusion.losses import MultiTaskCenterPointLoss
from fl_v3.models.fusion.preprocess import ImageAugmentationConfig, ImagePreprocessor
from fl_v3.models.ops.bev_pool import bev_pool


CAMERA_BEV = BEVConfig(
    point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
    bev_voxel=(0.4, 0.4),
    out_size_factor=2,
)


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    *,
    stride: int = 1,
    padding: int = 0,
    inplace: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        nn.ReLU(inplace=inplace),
    )


class Phase1GeneralizedLSSFPN(nn.Module):
    """Exact concat top-down FPN used by the pinned standalone Swin recipe."""

    def __init__(
        self,
        in_channels: Sequence[int] = (192, 384, 768),
        out_channels: int = 256,
        *,
        start_level: int = 0,
        num_outs: int = 3,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in in_channels)
        if channels != (192, 384, 768):
            raise ValueError("Phase-I Camera FPN channels must be [192,384,768]")
        if start_level != 0 or num_outs != 3 or out_channels != 256:
            raise ValueError("Phase-I Camera GeneralizedLSSFPN recipe drift")
        self.in_channels = channels
        self.out_channels = int(out_channels)
        self.start_level = int(start_level)
        self.num_outs = int(num_outs)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        last_fused_level = len(channels) - 2
        for index in range(len(channels) - 1):
            upper_channels = channels[index + 1] if index == last_fused_level else out_channels
            self.lateral_convs.append(
                _conv_bn_relu(channels[index] + upper_channels, out_channels, 1)
            )
            self.fpn_convs.append(
                _conv_bn_relu(out_channels, out_channels, 3, padding=1)
            )

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"Camera FPN needs {len(self.in_channels)} inputs, got {len(inputs)}"
            )
        laterals = list(inputs)
        for index, (tensor, channels) in enumerate(zip(laterals, self.in_channels, strict=True)):
            if tensor.ndim != 4 or tensor.shape[1] != channels:
                raise ValueError(
                    f"Camera FPN input {index} must be NCHW with C={channels}, "
                    f"got {tuple(tensor.shape)}"
                )
        for index in range(len(laterals) - 2, -1, -1):
            upper = F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[index] = self.fpn_convs[index](
                self.lateral_convs[index](torch.cat((laterals[index], upper), dim=1))
            )
        # The reference implementation returns only fused levels, not the raw
        # coarsest input.  LSSTransform consumes output zero (stride 8).
        return tuple(laterals[:-1])


class Phase1LSSTransform(nn.Module):
    """Pure-camera LSS with explicit FP32 optimized/fallback pooling."""

    def __init__(
        self,
        *,
        in_channels: int = 256,
        out_channels: int = 80,
        image_size: tuple[int, int] = (256, 704),
        feature_size: tuple[int, int] = (32, 88),
        xbound: tuple[float, float, float] = (-51.2, 51.2, 0.4),
        ybound: tuple[float, float, float] = (-51.2, 51.2, 0.4),
        zbound: tuple[float, float, float] = (-10.0, 10.0, 20.0),
        dbound: tuple[float, float, float] = (1.0, 60.0, 0.5),
        downsample: int = 2,
        pool_backend: str = "fallback",
        pool_build_directory: str | None = None,
    ) -> None:
        super().__init__()
        if (
            in_channels != 256
            or out_channels != 80
            or image_size != (256, 704)
            or feature_size != (32, 88)
            or xbound != (-51.2, 51.2, 0.4)
            or ybound != (-51.2, 51.2, 0.4)
            or zbound != (-10.0, 10.0, 20.0)
            or dbound != (1.0, 60.0, 0.5)
            or downsample != 2
        ):
            raise ValueError("Phase-I standalone Camera LSSTransform recipe drift")
        if pool_backend not in {"optimized", "fallback"}:
            raise ValueError("pool_backend must be optimized or fallback")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.pool_backend = pool_backend
        self.pool_build_directory = pool_build_directory

        bounds = (xbound, ybound, zbound)
        dx = torch.tensor([row[2] for row in bounds], dtype=torch.float32)
        bx = torch.tensor([row[0] + row[2] / 2.0 for row in bounds], dtype=torch.float32)
        nx = torch.tensor(
            [round((row[1] - row[0]) / row[2]) for row in bounds],
            dtype=torch.int64,
        )
        self.register_buffer("dx", dx)
        self.register_buffer("bx", bx)
        self.register_buffer("nx", nx)

        depths = torch.arange(*dbound, dtype=torch.float32)
        f_height, f_width = feature_size
        xs = torch.linspace(0.0, image_size[1] - 1.0, f_width, dtype=torch.float32)
        ys = torch.linspace(0.0, image_size[0] - 1.0, f_height, dtype=torch.float32)
        dd = depths.view(-1, 1, 1).expand(-1, f_height, f_width)
        xx = xs.view(1, 1, -1).expand(depths.numel(), f_height, f_width)
        yy = ys.view(1, -1, 1).expand(depths.numel(), f_height, f_width)
        frustum = torch.stack((xx * dd, yy * dd, dd, torch.ones_like(dd)), dim=-1)
        self.register_buffer("frustum", frustum)
        self.depth_bins = int(depths.numel())

        self.depthnet = nn.Conv2d(in_channels, self.depth_bins + out_channels, 1)
        self.downsample = nn.Sequential(
            _conv_bn_relu(out_channels, out_channels, 3, padding=1, inplace=True),
            _conv_bn_relu(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                inplace=True,
            ),
            _conv_bn_relu(out_channels, out_channels, 3, padding=1, inplace=True),
        )

    def _geometry(self, lidar2img: torch.Tensor) -> torch.Tensor:
        if lidar2img.ndim != 4 or lidar2img.shape[-2:] != (4, 4):
            raise ValueError("lidar2img must have shape [B,N,4,4]")
        # Calibration and integer binning are explicitly outside autocast.  The
        # input matrix already contains both scene-3D and image-2D transforms.
        with torch.autocast(device_type=lidar2img.device.type, enabled=False):
            image_to_lidar = torch.linalg.inv(lidar2img.to(torch.float32))
            geometry = torch.einsum(
                "bnij,dhwj->bndhwi", image_to_lidar, self.frustum
            )[..., :3]
        return geometry

    def prepare_pool_inputs(
        self,
        lifted: torch.Tensor,
        geometry: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        """Return the exact production operator inputs and output dimensions.

        The public diagnostic boundary deliberately shares this implementation
        with :meth:`_pool`: WP4 operator timing must measure the real B4
        frustum/geometry workload, not a synthetic proxy with different rank or
        collision structure.
        """
        batch, cameras, depth, height, width, channels = lifted.shape
        expected = (
            batch,
            cameras,
            self.depth_bins,
            self.feature_size[0],
            self.feature_size[1],
            self.out_channels,
        )
        if tuple(lifted.shape) != expected or geometry.shape != expected[:-1] + (3,):
            raise ValueError(
                f"LSS lifted/geometry contract drift: {tuple(lifted.shape)}, "
                f"{tuple(geometry.shape)}"
            )
        integer = torch.floor(
            (geometry - (self.bx - self.dx / 2.0)) / self.dx
        ).to(torch.int64)
        valid = (
            (integer[..., 0] >= 0)
            & (integer[..., 0] < self.nx[0])
            & (integer[..., 1] >= 0)
            & (integer[..., 1] < self.nx[1])
            & (integer[..., 2] >= 0)
            & (integer[..., 2] < self.nx[2])
        )
        batch_index = torch.arange(batch, device=lifted.device, dtype=torch.int64)
        batch_index = batch_index.view(batch, 1, 1, 1, 1).expand_as(valid)
        coordinates = torch.cat((integer, batch_index.unsqueeze(-1)), dim=-1)
        values = lifted.reshape(-1, channels)[valid.reshape(-1)].to(torch.float32).contiguous()
        coordinates = coordinates.reshape(-1, 4)[valid.reshape(-1)].to(torch.int32).contiguous()
        dimensions = (
            batch,
            int(self.nx[2]),
            int(self.nx[1]),  # H is metric y
            int(self.nx[0]),  # W is metric x
        )
        return values, coordinates, dimensions

    def operator_inputs(
        self,
        features: torch.Tensor,
        lidar2img: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[int, int, int, int],
        dict[str, torch.Tensor],
    ]:
        """Prepare the production pool inputs without executing the pool/decoder."""
        if features.ndim != 5:
            raise ValueError("LSS features must have shape [B,N,C,H,W]")
        batch, cameras, channels, height, width = features.shape
        if (channels, height, width) != (
            self.in_channels,
            self.feature_size[0],
            self.feature_size[1],
        ):
            raise ValueError(f"unexpected LSS feature shape {tuple(features.shape)}")
        if lidar2img.shape != (batch, cameras, 4, 4):
            raise ValueError("LSS calibration batch/camera dimensions differ")
        projected = self.depthnet(
            features.reshape(batch * cameras, channels, height, width)
        )
        depth_probability = projected[:, : self.depth_bins].softmax(dim=1)
        context = projected[:, self.depth_bins :]
        lifted = depth_probability.unsqueeze(1) * context.unsqueeze(2)
        lifted = lifted.view(
            batch,
            cameras,
            self.out_channels,
            self.depth_bins,
            height,
            width,
        ).permute(0, 1, 3, 4, 5, 2)
        geometry = self._geometry(lidar2img)
        values, coordinates, dimensions = self.prepare_pool_inputs(lifted, geometry)
        return values, coordinates, dimensions, {
            "depth_probability": depth_probability,
            "context": context,
            "geometry": geometry,
        }

    def _pool(
        self,
        lifted: torch.Tensor,
        geometry: torch.Tensor,
        *,
        backend: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
        values, coordinates, dimensions = self.prepare_pool_inputs(lifted, geometry)
        pooled = bev_pool(
            values,
            coordinates,
            *dimensions,
            backend=backend,
            build_directory=self.pool_build_directory,
        )
        # Reference collapses height bins by channel concatenation.  The frozen
        # z-bound has one bin, but retain the general operation and assertion.
        collapsed = torch.cat(pooled.unbind(dim=2), dim=1)
        if collapsed.shape[1] != self.out_channels * int(self.nx[2]):
            raise RuntimeError("unexpected LSS height-bin collapse")
        return collapsed, values, coordinates, dimensions

    def forward(
        self,
        features: torch.Tensor,
        lidar2img: torch.Tensor,
        *,
        pool_backend: str | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        backend = self.pool_backend if pool_backend is None else pool_backend
        if backend not in {"optimized", "fallback"}:
            raise ValueError("pool_backend must be optimized or fallback")
        pool_values, pool_geometry, pool_dimensions, diagnostics = self.operator_inputs(
            features, lidar2img
        )
        pooled_5d = bev_pool(
            pool_values,
            pool_geometry,
            *pool_dimensions,
            backend=backend,
            build_directory=self.pool_build_directory,
        )
        pooled = torch.cat(pooled_5d.unbind(dim=2), dim=1)
        if pooled.shape[1] != self.out_channels * int(self.nx[2]):
            raise RuntimeError("unexpected LSS height-bin collapse")
        bev = self.downsample(pooled)
        if not return_intermediates:
            return bev
        return {
            "bev": bev,
            "pooled_fp32": pooled,
            **diagnostics,
            "pool_values_fp32": pool_values,
            "pool_geometry_int32": pool_geometry,
            "pool_dimensions": pool_dimensions,
        }


class Phase1BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Phase1GeneralizedResNet(nn.ModuleList):
    """Reference decoder backbone with blocks [[2,128,2],[2,256,2],[2,512,1]]."""

    def __init__(
        self,
        in_channels: int = 80,
        blocks: Sequence[Sequence[int]] = ((2, 128, 2), (2, 256, 2), (2, 512, 1)),
    ) -> None:
        normalized = tuple(tuple(int(value) for value in row) for row in blocks)
        if in_channels != 80 or normalized != ((2, 128, 2), (2, 256, 2), (2, 512, 1)):
            raise ValueError("Phase-I Camera GeneralizedResNet recipe drift")
        modules: list[nn.Module] = []
        current = int(in_channels)
        for count, output, stride in normalized:
            stage = [Phase1BasicBlock(current, output, stride)]
            stage.extend(Phase1BasicBlock(output, output, 1) for _ in range(count - 1))
            modules.append(nn.Sequential(*stage))
            current = output
        super().__init__(modules)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for stage in self:
            x = stage(x)
            outputs.append(x)
        return outputs


class Phase1LSSFPN(nn.Module):
    """Reference Camera decoder neck (coarse+fine concat and x2 upsample)."""

    def __init__(self) -> None:
        super().__init__()
        self.in_indices = (-1, 0)
        self.in_channels = (512, 128)
        self.fuse = nn.Sequential(
            _conv_bn_relu(640, 256, 1, inplace=True),
            _conv_bn_relu(256, 256, 3, padding=1, inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            _conv_bn_relu(256, 256, 3, padding=1, inplace=True),
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 3:
            raise ValueError("Camera decoder LSSFPN needs three backbone levels")
        coarse = inputs[-1]
        fine = inputs[0]
        if coarse.shape[1] != 512 or fine.shape[1] != 128:
            raise ValueError("Camera decoder channel contract drift")
        coarse = F.interpolate(
            coarse, size=fine.shape[-2:], mode="bilinear", align_corners=True
        )
        return self.upsample(self.fuse(torch.cat((coarse, fine), dim=1)))


class Phase1CameraDetector(nn.Module):
    """Complete frozen standalone Camera candidate used by Envelope A/B."""

    def __init__(
        self,
        *,
        pool_backend: str = "fallback",
        pool_build_directory: str | None = None,
    ) -> None:
        super().__init__()
        self.preprocess = ImagePreprocessor(
            image_hw=(256, 704),
            augmentation=ImageAugmentationConfig(
                enabled=True,
                resize_limits=(0.38, 0.55),
                validation_resize=0.48,
                bottom_crop_limits=(0.0, 0.0),
                rotation_limits_degrees=(-5.4, 5.4),
                random_flip=True,
            ),
        )
        self.camera_backbone = CameraBackbone(
            "swin_t",
            frozen=False,
            pretrained=False,
            activation_checkpoint=False,
            sdpa_attention=False,
            out_indices=(1, 2, 3),
            output_layer_norm=True,
        )
        self.camera_neck = Phase1GeneralizedLSSFPN()
        self.view_transform = Phase1LSSTransform(
            pool_backend=pool_backend,
            pool_build_directory=pool_build_directory,
        )
        self.decoder_backbone = Phase1GeneralizedResNet()
        self.decoder_neck = Phase1LSSFPN()
        self.head = CenterPointHead(
            in_channels=256,
            n_classes=10,
            head_channels=64,
            conv_layers=2,
            tasks=NUSCENES_CENTERHEAD_TASKS,
            shared_channels=64,
            normalization="batch_norm",
        )
        self._operator_profile_ranges = False

    @contextmanager
    def operator_profile_ranges(self):
        """Enable output-neutral, bounded ranges for a short torch trace."""
        if self._operator_profile_ranges:
            raise RuntimeError("Phase-I Camera profiler ranges are already active")
        self._operator_profile_ranges = True
        try:
            yield self
        finally:
            self._operator_profile_ranges = False

    def _profile_range(self, name: str):
        if not self._operator_profile_ranges:
            return nullcontext()
        return torch.profiler.record_function(f"fl_v3::camera::{name}")

    def forward(
        self,
        batch: dict,
        *,
        pool_backend: str | None = None,
        return_intermediates: bool = False,
    ):
        for field in ("images", "lidar2img", "cam_intrinsics"):
            if field not in batch:
                raise KeyError(f"Phase-I Camera batch is missing {field!r}")
        with self._profile_range("preprocess"):
            preprocessed = self.preprocess(
                batch["images"],
                batch["lidar2img"],
                batch["cam_intrinsics"],
                augmentation_params=batch.get("augmentation_params"),
            )
        images = preprocessed["images"]
        batch_size, cameras = images.shape[:2]
        with self._profile_range("swin_backbone"):
            features = self.camera_backbone(
                images.reshape(batch_size * cameras, *images.shape[2:])
            )
        with self._profile_range("camera_neck"):
            camera_pyramid = self.camera_neck(features)
        stride8 = camera_pyramid[0].view(
            batch_size, cameras, 256, *camera_pyramid[0].shape[-2:]
        )
        with self._profile_range("view_transform_and_pool"):
            view = self.view_transform(
                stride8,
                preprocessed["lidar2img"],
                pool_backend=pool_backend,
                return_intermediates=return_intermediates,
            )
        camera_bev = view["bev"] if isinstance(view, dict) else view
        with self._profile_range("decoder_backbone"):
            decoded_levels = self.decoder_backbone(camera_bev)
        with self._profile_range("decoder_neck"):
            decoded = self.decoder_neck(decoded_levels)
        with self._profile_range("head"):
            task_outputs = self.head(decoded)
        if not return_intermediates:
            return task_outputs
        assert isinstance(view, dict)
        return {
            "task_outputs": task_outputs,
            "camera_bev": camera_bev,
            "decoder_feature": decoded,
            "pool_output_fp32": view["pooled_fp32"],
            "depth_probability": view["depth_probability"],
            "camera_context": view["context"],
            "pool_values_fp32": view["pool_values_fp32"],
            "pool_geometry_int32": view["pool_geometry_int32"],
            "pool_dimensions": view["pool_dimensions"],
        }

    @staticmethod
    def build_criterion() -> MultiTaskCenterPointLoss:
        return MultiTaskCenterPointLoss(
            cfg=CAMERA_BEV,
            reg_weight=0.25,
            global_class_names=REFERENCE_OBJECT_CLASSES,
        )

    @torch.no_grad()
    def decode(self, output, *, score_threshold: float = 0.1) -> list[dict]:
        if isinstance(output, dict):
            output = output["task_outputs"]
        return decode_centerhead(
            output,
            bev=CAMERA_BEV,
            config=CenterHeadDecodeConfig(
                score_threshold=float(score_threshold),
                per_class_pre_max=500,
                task_pre_max=500,
                nms_pre_max=1000,
                nms_post_max=83,
                rotate_iou_threshold=0.2,
            ),
        )

    def parameter_groups(self) -> dict[str, Iterable[nn.Parameter]]:
        """Named module boundary used only for conformance diagnostics."""
        return {
            "camera_backbone": self.camera_backbone.parameters(),
            "camera_neck": self.camera_neck.parameters(),
            "view_transform": self.view_transform.parameters(),
            "decoder_backbone": self.decoder_backbone.parameters(),
            "decoder_neck": self.decoder_neck.parameters(),
            "head": self.head.parameters(),
        }


def build_phase1_camera_model(
    config,
    *,
    pool_backend: str | None = None,
    pool_build_directory: str | None = None,
    require_accepted_initialization: bool = True,
    allow_unpromoted_backend: bool = False,
) -> Phase1CameraDetector:
    """Construct from one validated ResolvedConfig and bind accepted weights."""
    if not getattr(config, "is_phase1", False):
        raise ValueError("Phase-I Camera construction requires a Phase-I ResolvedConfig")
    raw = config.as_dict()
    if raw["contract"]["branch"] != "camera":
        raise ValueError("Phase-I Camera construction received the LiDAR recipe")
    model_spec = raw["model"]
    schema_version = str(raw["schema_version"])
    frozen_model = (
        FROZEN_CAMERA_MODEL_V2
        if schema_version == PHASE1_SCHEMA_V2
        else FROZEN_CAMERA_MODEL
    )
    expected_pool_identity = (
        "pytorch_sorted_segment_reduce"
        if schema_version == PHASE1_SCHEMA_V2
        else "optimized_cuda"
    )
    if (
        model_spec != frozen_model
        or model_spec["view_transform"]["pool_backend"] != expected_pool_identity
    ):
        raise ValueError("resolved Phase-I Camera graph contract drift")
    selected_backend = (
        "fallback"
        if pool_backend is None and schema_version == PHASE1_SCHEMA_V2
        else "optimized"
        if pool_backend is None
        else str(pool_backend)
    )
    if (
        schema_version == PHASE1_SCHEMA_V2
        and selected_backend != "fallback"
        and not allow_unpromoted_backend
    ):
        raise ValueError(
            "O-150 Envelope-B production construction forbids implicit CUDA promotion"
        )
    model = Phase1CameraDetector(
        pool_backend=selected_backend,
        pool_build_directory=pool_build_directory,
    )
    initialization = raw["initialization"]
    if initialization["status"] != "accepted":
        if require_accepted_initialization:
            raise RuntimeError("Phase-I Camera initialization is not accepted")
        return model

    from fl_v3.models.phase1_swin import (
        sha256_file,
        validate_and_load_original_swin,
    )

    checkpoint = Path(initialization["final_path"]).resolve()
    report_path = Path(initialization["mapping_report_path"]).resolve()
    if sha256_file(checkpoint) != initialization["physical_sha256"]:
        raise RuntimeError("accepted Camera checkpoint physical identity drift")
    if sha256_file(report_path) != initialization["mapping_report_sha256"]:
        raise RuntimeError("accepted Camera mapping-report identity drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("physical_sha256", "initialization_state_sha256"):
        if report[key] != initialization[key]:
            raise RuntimeError(f"accepted Camera mapping report {key} drift")
    observed = validate_and_load_original_swin(
        model,
        checkpoint,
        expected_physical_sha256=initialization["physical_sha256"],
    )
    if observed["initialization_state_sha256"] != initialization[
        "initialization_state_sha256"
    ]:
        raise RuntimeError("accepted Camera initialized model state drift")
    return model


__all__ = [
    "CAMERA_BEV",
    "Phase1BasicBlock",
    "Phase1CameraDetector",
    "Phase1GeneralizedLSSFPN",
    "Phase1GeneralizedResNet",
    "Phase1LSSFPN",
    "Phase1LSSTransform",
    "build_phase1_camera_model",
]
