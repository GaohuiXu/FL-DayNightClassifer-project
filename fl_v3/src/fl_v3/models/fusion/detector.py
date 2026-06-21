"""BEVFusion-class detector wiring + deterministic decode (fl_v3 T2).

Wires preprocess → frozen camera backbone → LSS-FPN neck → LSS view transform
(camera-BEV) ⊕ PointPillars (LiDAR-BEV) → ``ConvFuser`` → SECOND-FPN neck →
CenterPoint head, all on the single :mod:`bev_grid` convention. ``forward(batch)``
returns the head dict (+ intermediate BEV features for V2/V3 when requested);
``decode(head_out)`` returns 3D boxes+scores in the **T1 canonical** convention.

## Deterministic decode (SPEC §0 + detector bullet)

``torch.topk`` has **no ``stable`` kwarg** and its CUDA tie order is not reproducible, so
peak extraction is: (1) a fixed **3×3 ``max_pool2d`` local-max mask** (``keep = (hmax ==
heat)``) zeroes non-peak cells; (2) the surviving scores are ranked with
**``torch.sort(stable=True)``** (descending) and sliced to ``max_objects`` — a canonical,
reproducible tie order (ascending flat index among ties). **No post-decode box NMS**
(CenterPoint is NMS-free; circle/rotated NMS is non-deterministic and banned). The box is
``(cx,cy,cz,dx=l,dy=w,dz=h,yaw)`` with ``yaw = atan2(sin,cos)`` — **NO** mmdet3d ``-π/2``
offset and **NO** ``(l,w,h)`` swap (the encode→decode golden pins it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_v3.models.fusion.bev_grid import BEVConfig, flat_to_colrow, head_decode_to_metric
from fl_v3.models.fusion.preprocess import ImagePreprocessor, DEFAULT_IMAGE_HW
from fl_v3.models.fusion.camera_backbone import CameraBackbone
from fl_v3.models.fusion.camera_neck import GeneralizedLSSFPN
from fl_v3.models.fusion.view_transform import DepthLSSTransform
from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder
from fl_v3.models.fusion.fusion import ConvFuser
from fl_v3.models.fusion.bev_neck import SecondFPNNeck
from fl_v3.models.fusion.head import CenterPointHead


@dataclass
class DetectorConfig:
    """All architecture knobs (config-injected from run_config)."""

    camera_backbone: str = "swin_t"      # swin_t | resnet18
    freeze_camera_backbone: bool = True  # D1
    pretrained_backbone: bool = True
    activation_checkpoint: bool = False  # MCR P1 (D16 envelope): grad checkpointing on a TRAINED backbone
    image_hw: tuple = DEFAULT_IMAGE_HW
    feat_stride: int = 16                # LSS-FPN output stride
    neck_channels: int = 128
    context_channels: int = 80           # camera-BEV channels
    depth_bins: tuple = (1.0, 60.0, 1.0)
    lidar_channels: int = 64
    max_points_per_pillar: int = 32
    max_pillars: int = 30000
    fusion_channels: int = 128
    bev_neck_channels: int = 256
    head_channels: int = 64
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
        self.preprocess = ImagePreprocessor(image_hw=c.image_hw)
        self.camera_backbone = CameraBackbone(
            c.camera_backbone, frozen=c.freeze_camera_backbone, pretrained=c.pretrained_backbone,
            activation_checkpoint=c.activation_checkpoint,
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
        )
        self.lidar_encoder = PointPillarsEncoder(
            out_channels=c.lidar_channels,
            max_points=c.max_points_per_pillar,
            max_pillars=c.max_pillars,
            cfg=c.bev,
        )
        self.fusion = ConvFuser(
            camera_channels=c.context_channels,
            lidar_channels=c.lidar_channels,
            out_channels=c.fusion_channels,
        )
        self.bev_neck = SecondFPNNeck(
            in_channels=c.fusion_channels,
            out_channels=c.bev_neck_channels,
            out_size_factor=c.bev.out_size_factor,
        )
        self.head = CenterPointHead(
            in_channels=self.bev_neck.out_channels,
            n_classes=c.n_classes,
            head_channels=c.head_channels,
        )

    # --- forward ---
    def forward(self, batch: dict, return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        c = self.cfg
        pre = self.preprocess(batch["images"], batch["lidar2img"], batch["cam_intrinsics"])
        imgs = pre["images"]                       # [B,6,3,H,W]
        B, N = imgs.shape[0], imgs.shape[1]
        feats = self.camera_backbone(imgs.reshape(B * N, *imgs.shape[2:]))
        camfeat = self.camera_neck(feats)          # [B*N, C, fH, fW]
        vt = self.view_transform(camfeat, pre["lidar2img"], B, N)
        camera_bev = vt["bev"]                     # [B, Cc, ny, nx]
        lidar_bev = self.lidar_encoder(batch["lidar_points"], B)  # [B, Cl, ny, nx]
        fused = self.fusion(camera_bev, lidar_bev)
        neck = self.bev_neck(fused)
        out = self.head(neck)
        if return_intermediates:
            out = dict(out)
            out["_camera_bev"] = camera_bev
            out["_lidar_bev"] = lidar_bev
            out["_fused_bev"] = fused
            out["_depth_prob"] = vt["depth_prob"]
            out["_camera_context"] = vt["context"]
        return out

    # --- deterministic decode ---
    @torch.no_grad()
    def decode(
        self, head_out: Dict[str, torch.Tensor], score_threshold: Optional[float] = None,
        max_objects: Optional[int] = None,
    ) -> List[dict]:
        cfg = self.cfg.bev
        thr = self.cfg.score_threshold if score_threshold is None else score_threshold
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
            "preprocess": self.preprocess,
            "camera_backbone": self.camera_backbone,
            "camera_neck": self.camera_neck,
            "view_transform": self.view_transform,
            "lidar_encoder": self.lidar_encoder,
            "fusion": self.fusion,
            "bev_neck": self.bev_neck,
            "head": self.head,
        }
        table = {}
        for name, m in modules.items():
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            table[name] = {"total": total, "trainable": trainable, "frozen": total - trainable}
        return table
