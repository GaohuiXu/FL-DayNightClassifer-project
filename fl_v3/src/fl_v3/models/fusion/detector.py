"""Resolved BEVFusion-class detector wiring and decode.

The optional camera path is preprocess → configurable trainable/frozen backbone →
camera FPN → LSS view transform. The optional LiDAR path is either the deterministic
pillar encoder (with an optional dense backbone) or the sparse SECOND voxel encoder.
Camera-only/LiDAR-only modes use adapters; fusion concatenates both BEVs through
``ConvFuser``. Every mode then uses the shared BEV neck and six-task CenterHead on
one :mod:`bev_grid` convention. ``forward(batch)`` returns that reviewed task list
(plus named intermediate BEVs when requested); ``decode(head_out)`` returns boxes
and scores in the **T1 canonical** convention.

Production decode delegates to the reviewed S05 reference-faithful no-starvation
path: forced-FP32 fields, per-class K=500, deterministic global-class/spatial tie
order, official task-wide circle/rotate NMS, and post=83.  The older single-head
stable-sort decoder remains only for inventoried non-production callers and is
unreachable from the strict ``centerhead_multitask`` constructor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_v3.models.fusion.bev_grid import BEVConfig, flat_to_colrow, head_decode_to_metric
from fl_v3.models.fusion.preprocess import (
    ImageAugmentationConfig,
    ImagePreprocessor,
    DEFAULT_IMAGE_HW,
)
from fl_v3.models.fusion.camera_backbone import CameraBackbone
from fl_v3.models.fusion.camera_neck import GeneralizedLSSFPN
from fl_v3.models.fusion.view_transform import DepthLSSTransform
from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder
from fl_v3.models.fusion.lidar_backbone import LidarBackbone2D
from fl_v3.models.fusion.fusion import ConvFuser
from fl_v3.models.fusion.bev_neck import SecondFPNNeck
from fl_v3.models.fusion.head import CenterPointHead
from fl_v3.models.fusion.centerhead_decode import (
    CenterHeadDecodeConfig,
    decode_centerhead,
)
from fl_v3.utils.runtime import normalize_model_mode, require_spconv_238


@dataclass
class DetectorConfig:
    """All architecture knobs (config-injected from run_config)."""

    model_mode: str = "fusion"
    required_spconv_version: str | None = None
    camera_backbone: str = "swin_t"      # swin_t | resnet18
    freeze_camera_backbone: bool = True  # D1
    pretrained_backbone: bool = True
    activation_checkpoint: bool = False  # MCR P1 (D16 envelope): grad checkpointing on a TRAINED backbone
    swin_sdpa: bool = False              # MCR P2 (D16 envelope): SDPA core for Swin windowed attention
    image_hw: tuple = DEFAULT_IMAGE_HW
    feat_stride: int = 16                # LSS-FPN output stride
    neck_channels: int = 128
    context_channels: int = 80           # camera-BEV channels
    depth_bins: tuple = (1.0, 60.0, 1.0)
    reference_camera: bool = False
    camera_bev_output_dtype: str = "float32"
    lidar_channels: int = 64
    max_points_per_pillar: int = 32
    max_pillars: int = 30000
    lidar_sweeps: int = 1                 # MCR P1: >1 ⇒ multi-sweep input (+dt channel in the PFN)
    lidar_encoder: str = "pillar"         # pillar (PFN, default) | voxel (spconv sparse 3D, Rule#2-relaxed; z-res)
    lidar_z_voxel: float | None = None     # voxel only: z voxel size; None keeps historical cubic xyz voxels
    lidar_sparse_z_size: int | None = None # voxel only: optional sparse z shape override for parity probes
    # Internal mapping of S08's explicit sparse_conv_precision enum.  ``False``
    # keeps voxelization/VFE/SECOND/to_bev in the FP32 island; ``True`` enables
    # the reviewed sparse-FP16 path (voxelization/VFE themselves remain FP32).
    sparse_conv_fp16: bool = False
    second_normalization: str = "group_norm"
    lidar_input_bev: BEVConfig | None = None
    max_voxels_train: int = 120000
    max_voxels_eval: int = 160000
    lidar_backbone: bool = False          # MCR P1 capacity lever: dense 2D conv LiDAR backbone (default OFF)
    lidar_backbone_out: int = 128         # backbone Cout → widens ConvFuser.lidar_channels when ON
    lidar_backbone_checkpoint: bool = False  # activation-checkpoint the stages (for the 0.2m/512² leg)
    lidar_backbone_stages: int = 3        # down-stages; 4 adds an H/8 level (large-object RF at 0.2m)
    fusion_channels: int = 256
    bev_neck_channels: int = 256
    head_channels: int = 64
    head_conv_layers: int = 2
    n_classes: int = 10
    bev: BEVConfig = field(default_factory=BEVConfig)
    # decode
    max_objects: int = 200
    score_threshold: float = 0.1


class BEVFusionDetector(nn.Module):
    """The platform's deterministic BEVFusion-class detector."""

    def __init__(self, cfg: Optional[DetectorConfig] = None):
        super().__init__()
        self.cfg = cfg or DetectorConfig()
        c = self.cfg
        self.model_mode = normalize_model_mode(c.model_mode)
        self._runtime_lock = threading.RLock()
        self._training_boundary_tensors: dict[str, torch.Tensor] | None = None
        self._s10_observer = None
        self._operator_profile_ranges = False
        use_camera = self.model_mode in {"camera_only", "fusion"}
        use_lidar = self.model_mode in {"lidar_only", "fusion"}
        if use_lidar and c.required_spconv_version is not None:
            if c.required_spconv_version != "2.3.8":
                raise ValueError("required_spconv_version must be exactly '2.3.8'")
            require_spconv_238()

        if use_camera:
            self.preprocess = ImagePreprocessor(
                image_hw=c.image_hw,
                augmentation=(ImageAugmentationConfig() if c.reference_camera else None),
            )
            self.camera_backbone = CameraBackbone(
                c.camera_backbone, frozen=c.freeze_camera_backbone, pretrained=c.pretrained_backbone,
                activation_checkpoint=c.activation_checkpoint, sdpa_attention=c.swin_sdpa,
            )
            self.camera_neck = GeneralizedLSSFPN(
                in_channels=self.camera_backbone.out_channels,
                in_strides=self.camera_backbone.strides,
                out_channels=c.neck_channels,
                out_stride=c.feat_stride,
            )
            self.view_transform = DepthLSSTransform(
                in_channels=c.neck_channels,
                context_channels=c.context_channels,
                depth_bins=c.depth_bins,
                image_hw=c.image_hw,
                feat_stride=c.feat_stride,
                cfg=c.bev,
                bev_output_dtype=c.camera_bev_output_dtype,
            )
        else:
            self.preprocess = self.camera_backbone = self.camera_neck = self.view_transform = None

        if use_lidar and c.lidar_encoder == "voxel":
            from fl_v3.models.fusion.sparse_voxel_encoder import SparseVoxelEncoder
            self.lidar_encoder = SparseVoxelEncoder(
                out_channels=c.lidar_channels, cfg=(c.lidar_input_bev or c.bev),
                use_timestamp=(c.lidar_sweeps > 1),
                max_voxels_train=c.max_voxels_train,
                max_voxels_eval=c.max_voxels_eval,
                max_points_per_voxel=c.max_points_per_pillar,
                z_voxel=c.lidar_z_voxel, sparse_z_size=c.lidar_sparse_z_size,
                sparse_conv_fp16=c.sparse_conv_fp16,
                second_normalization=c.second_normalization,
            )
        elif use_lidar:
            self.lidar_encoder = PointPillarsEncoder(
                out_channels=c.lidar_channels,
                max_points=c.max_points_per_pillar,
                max_pillars=c.max_pillars,
                cfg=c.bev,
                use_timestamp=(c.lidar_sweeps > 1),
            )
        else:
            self.lidar_encoder = None
        # MCR P1 capacity lever (default OFF ⇒ None ⇒ byte-identical): dense 2D conv backbone on the
        # scattered pillar BEV BEFORE fusion — gives the LiDAR branch the receptive field the PFN lacks.
        self.lidar_backbone = (
            LidarBackbone2D(in_channels=c.lidar_channels, out_channels=c.lidar_backbone_out,
                            activation_checkpoint=c.lidar_backbone_checkpoint,
                            num_stages=c.lidar_backbone_stages)
            if use_lidar and c.lidar_backbone else None)
        # When the backbone is ON it widens the LiDAR feature into the fuser; OFF keeps lidar_channels=64
        # ⇒ ConvFuser in_channels unchanged ⇒ fusion byte-identical to the pre-backbone baseline.
        lidar_out = c.lidar_backbone_out if c.lidar_backbone else c.lidar_channels
        self.fusion = None
        self.camera_adapter = None
        self.lidar_adapter = None
        if self.model_mode == "fusion":
            self.fusion = ConvFuser(
                camera_channels=c.context_channels, lidar_channels=lidar_out,
                out_channels=c.fusion_channels,
            )
        elif self.model_mode == "camera_only":
            self.camera_adapter = nn.Conv2d(c.context_channels, c.fusion_channels, kernel_size=1)
        else:
            self.lidar_adapter = nn.Conv2d(lidar_out, c.fusion_channels, kernel_size=1)
        self.bev_neck = SecondFPNNeck(
            in_channels=c.fusion_channels,
            out_channels=c.bev_neck_channels,
            out_size_factor=c.bev.out_size_factor,
        )
        self.head = CenterPointHead(
            in_channels=self.bev_neck.out_channels,
            n_classes=c.n_classes,
            head_channels=c.head_channels,
            conv_layers=c.head_conv_layers,
        )

    def __getstate__(self):
        """Locks are runtime-only; allow EMA/deepcopy to create its own instance lock."""
        if self._training_boundary_tensors is not None:
            raise RuntimeError("cannot copy detector while training-boundary capture is active")
        if self._s10_observer is not None:
            raise RuntimeError("cannot copy detector while STOP-B observation is active")
        state = self.__dict__.copy()
        state.pop("_runtime_lock", None)
        state.pop("_training_boundary_tensors", None)
        state.pop("_s10_observer", None)
        state.pop("_operator_profile_ranges", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._runtime_lock = threading.RLock()
        self._training_boundary_tensors = None
        self._s10_observer = None
        self._operator_profile_ranges = False

    # --- forward ---
    def train(self, mode: bool = True):
        """Serialize mode transitions with forward for the instance's sparse path."""
        with self._runtime_lock:
            return super().train(mode)

    @contextmanager
    def serialized_mode(self, training: bool):
        """Hold the instance lock across a complete train/eval traversal."""
        with self._runtime_lock:
            super().train(bool(training))
            yield self

    @contextmanager
    def operator_profile_ranges(self):
        """Enable bounded, output-neutral branch ranges for ``torch.profiler``."""
        with self._runtime_lock:
            if self._operator_profile_ranges:
                raise RuntimeError("nested operator-profile ranges are forbidden")
            self._operator_profile_ranges = True
            try:
                yield self
            finally:
                self._operator_profile_ranges = False

    def _profiled(self, name: str, function, *args, **kwargs):
        if not self._operator_profile_ranges:
            return function(*args, **kwargs)
        with torch.profiler.record_function(f"fl_v3::{name}"):
            return function(*args, **kwargs)

    @contextmanager
    def capture_training_boundaries(self):
        """Retain four explicit S08 tensors for one training window."""
        with self._runtime_lock:
            if self._training_boundary_tensors is not None:
                raise RuntimeError("nested training-boundary capture is forbidden")
            if not self.training:
                raise RuntimeError("training-boundary capture requires model.train()")
            tensors: dict[str, torch.Tensor] = {}
            self._training_boundary_tensors = tensors
            sparse = self.lidar_encoder if hasattr(self.lidar_encoder, "record_debug") else None
            old_debug = None if sparse is None else bool(sparse.record_debug)
            old_meta = None if sparse is None else sparse.last_sparse_meta
            if sparse is not None:
                sparse.record_debug = True
            try:
                yield tensors
            finally:
                tensors.clear()
                self._training_boundary_tensors = None
                if sparse is not None:
                    sparse.record_debug = old_debug
                    sparse.last_sparse_meta = old_meta

    def _capture_training_boundary(self, name: str, tensor: torch.Tensor) -> None:
        tensors = self._training_boundary_tensors
        if tensors is None:
            return
        if name in tensors:
            raise RuntimeError(f"duplicate training-boundary name {name!r}")
        if not torch.is_grad_enabled() or not tensor.requires_grad:
            raise RuntimeError(f"training boundary {name!r} does not carry autograd")
        tensor.retain_grad()
        tensors[name] = tensor

    @contextmanager
    def capture_s10_observation(self, observer):
        """Attach one explicit STOP-B recorder without changing model outputs."""
        with self._runtime_lock:
            if self._s10_observer is not None:
                raise RuntimeError("nested STOP-B observation is forbidden")
            if self._training_boundary_tensors is not None:
                raise RuntimeError("STOP-B and S08 boundary capture cannot overlap")
            if not self.training:
                raise RuntimeError("STOP-B observation requires model.train()")
            self._s10_observer = observer
            sparse_backbone = None
            sparse_encoder = None
            old_sparse_debug = None
            old_sparse_meta = None
            if self.cfg.lidar_encoder == "voxel" and self.lidar_encoder is not None:
                sparse_encoder = self.lidar_encoder
                old_sparse_debug = bool(sparse_encoder.record_debug)
                old_sparse_meta = sparse_encoder.last_sparse_meta
                sparse_encoder.record_debug = True
                sparse_backbone = self.lidar_encoder.backbone
                sparse_backbone.set_s10_observer(observer)
            try:
                yield observer
            finally:
                if sparse_backbone is not None:
                    sparse_backbone.set_s10_observer(None)
                if sparse_encoder is not None:
                    sparse_encoder.record_debug = old_sparse_debug
                    sparse_encoder.last_sparse_meta = old_sparse_meta
                self._s10_observer = None

    def _capture_s10_dense(self, name: str, tensor: torch.Tensor) -> None:
        if self._s10_observer is not None:
            self._s10_observer.capture_dense_boundary(name, tensor)

    def _forward_locked(self, batch: dict, return_intermediates: bool):
        c = self.cfg
        camera_bev = lidar_bev = vt = None
        if self.model_mode in {"camera_only", "fusion"}:
            for key in ("images", "lidar2img", "cam_intrinsics"):
                if key not in batch:
                    raise KeyError(f"{self.model_mode} forward requires batch[{key!r}]")
            pre = self._profiled(
                "camera.preprocess",
                self.preprocess,
                batch["images"],
                batch["lidar2img"],
                batch["cam_intrinsics"],
                augmentation_params=batch.get("augmentation_params"),
            )
            imgs = pre["images"]
            B, N = imgs.shape[0], imgs.shape[1]
            feats = self._profiled(
                "camera.backbone",
                self.camera_backbone,
                imgs.reshape(B * N, *imgs.shape[2:]),
            )
            camfeat = self._profiled("camera.neck", self.camera_neck, feats)
            vt = self._profiled(
                "camera.view_transform",
                self.view_transform,
                camfeat,
                pre["lidar2img"],
                B,
                N,
            )
            camera_bev = vt["bev"]
        else:
            B = int(batch.get("batch_size", len(batch.get("gt_boxes", ()))))
            if B <= 0:
                raise ValueError("lidar_only forward requires positive batch_size or gt_boxes length")
        if self.model_mode in {"lidar_only", "fusion"}:
            if "lidar_points" not in batch:
                raise KeyError(f"{self.model_mode} forward requires batch['lidar_points']")
            if c.lidar_encoder == "voxel":
                lidar_bev = self._profiled(
                    "lidar.encoder",
                    self.lidar_encoder,
                    batch["lidar_points"],
                    B,
                    boundary_capture=(
                        self._capture_training_boundary
                        if self._training_boundary_tensors is not None
                        else None
                    ),
                )
            else:
                lidar_bev = self._profiled(
                    "lidar.encoder", self.lidar_encoder, batch["lidar_points"], B
                )
            if self.lidar_backbone is not None:
                lidar_bev = self._profiled(
                    "lidar.backbone", self.lidar_backbone, lidar_bev
                )
        if self.model_mode == "fusion":
            if camera_bev.shape[-2:] != lidar_bev.shape[-2:]:
                raise RuntimeError(
                    "camera/LiDAR BEV geometry mismatch: "
                    f"camera={tuple(camera_bev.shape[-2:])}, lidar={tuple(lidar_bev.shape[-2:])}"
                )
            self._capture_s10_dense("fusion.camera_input", camera_bev)
            self._capture_s10_dense("fusion.lidar_input", lidar_bev)
            fused = self._profiled(
                "fusion.fuser", self.fusion, camera_bev, lidar_bev
            )
        elif self.model_mode == "camera_only":
            fused = self._profiled("fusion.camera_adapter", self.camera_adapter, camera_bev)
        else:
            fused = self._profiled("fusion.lidar_adapter", self.lidar_adapter, lidar_bev)
        self._capture_s10_dense("bev_neck.input", fused)
        neck = self._profiled("shared.bev_neck", self.bev_neck, fused)
        self._capture_training_boundary("head.input", neck)
        self._capture_s10_dense("head.input", neck)
        out = self._profiled("shared.head", self.head, neck)
        if return_intermediates:
            out = {"task_outputs": out} if isinstance(out, list) else dict(out)
            if camera_bev is not None:
                out["_camera_bev"] = camera_bev
            if lidar_bev is not None:
                out["_lidar_bev"] = lidar_bev
            out["_fused_bev"] = fused
            if vt is not None:
                out["_depth_prob"] = vt["depth_prob"]
                out["_camera_context"] = vt["context"]
        return out

    def forward(self, batch: dict, return_intermediates: bool = False):
        with self._runtime_lock:
            return self._forward_locked(batch, return_intermediates)

    # --- deterministic decode ---
    @torch.no_grad()
    def decode(
        self, head_out, score_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
    ) -> List[dict]:
        cfg = self.cfg.bev
        thr = self.cfg.score_threshold if score_threshold is None else score_threshold
        if isinstance(head_out, dict) and "task_outputs" in head_out:
            head_out = head_out["task_outputs"]
        if isinstance(head_out, (list, tuple)):
            if max_objects is not None:
                raise ValueError(
                    "multi-task CenterHead uses the frozen per-class/NMS budgets; "
                    "max_objects is a legacy single-head override"
                )
            return decode_centerhead(
                head_out,
                bev=cfg,
                config=CenterHeadDecodeConfig(score_threshold=float(thr)),
            )
        if not isinstance(head_out, dict):
            raise TypeError("head_out must be six task dictionaries or a legacy head dictionary")
        K = self.cfg.max_objects if max_objects is None else max_objects
        heat = head_out["heatmap"].sigmoid()              # [B, C, H, W]
        B, C, H, W = heat.shape
        # 3×3 local-max mask (fixed kernel; deterministic forward)
        hmax = F.max_pool2d(heat, kernel_size=3, stride=1, padding=1)
        keep = (hmax == heat).to(heat.dtype)
        heat_masked = heat * keep                          # non-peaks → 0

        reg = head_out["reg"]; height = head_out["height"]; dim = head_out["dim"]
        rot = head_out["rot"]; vel = head_out["vel"]
        results: List[dict] = []
        for b in range(B):
            flat = heat_masked[b].reshape(-1)              # [C*H*W]
            # deterministic top-K: stable descending sort + slice (no torch.topk tie issue)
            vals, idx = torch.sort(flat, descending=True, stable=True)
            vals, idx = vals[:K], idx[:K]
            cls = torch.div(idx, H * W, rounding_mode="floor")
            cell = idx % (H * W)
            col, row = flat_to_colrow(cell, W)
            # gather regression at (row, col)
            off = reg[b, :, row, col]                       # [2, K]
            cx, cy = head_decode_to_metric(col, off[0], row, off[1], cfg)
            cz = height[b, 0, row, col].to(torch.float64)
            ldim = dim[b, :, row, col]                       # [3, K] = log(l,w,h)
            dl, dw, dh = torch.exp(ldim[0]), torch.exp(ldim[1]), torch.exp(ldim[2])
            s, co = rot[b, 0, row, col], rot[b, 1, row, col]
            yaw = torch.atan2(s, co)                         # T1 canonical (NO -π/2, NO swap)
            vx, vy = vel[b, 0, row, col], vel[b, 1, row, col]
            box7 = torch.stack(
                [cx, cy, cz, dl.to(torch.float64), dw.to(torch.float64),
                 dh.to(torch.float64), yaw.to(torch.float64)], dim=1
            ).to(torch.float32)                              # [K, 7]
            sel = vals >= thr
            results.append({
                "boxes": box7[sel],
                "scores": vals[sel],
                "labels": cls[sel],
                "velocity": torch.stack([vx, vy], dim=1)[sel],
            })
        return results

    # --- per-module parameter accounting (the Q2-dilution seed) ---
    def param_table(self) -> Dict[str, dict]:
        modules = {
            "bev_neck": self.bev_neck,
            "head": self.head,
        }
        for name in ("preprocess", "camera_backbone", "camera_neck", "view_transform",
                     "lidar_encoder", "fusion", "camera_adapter", "lidar_adapter"):
            module = getattr(self, name, None)
            if module is not None:
                modules[name] = module
        if self.lidar_backbone is not None:           # guarded: keeps Q2-dilution accounting complete when ON
            modules["lidar_backbone"] = self.lidar_backbone
        table = {}
        for name, m in modules.items():
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            table[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        return table
