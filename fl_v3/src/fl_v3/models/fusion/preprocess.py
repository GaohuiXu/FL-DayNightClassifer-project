"""Deterministic image resize + ImageNet normalization (fl_v3 T2).

T1 stores **native uint8 1600×900 RGB** images and the full-resolution
``cam_intrinsics`` / ``lidar2img`` (T1 schema). T2 owns the resize + normalization
T1 deliberately left out — and, critically, **rescales the calibration consistently**
so camera features land in the right BEV cell (the SPEC's "resize/intrinsic/lidar2img
inconsistency" failure mode → camera features mis-placed in BEV).

## The half-pixel-correct resize (exact consistency, not "≈")

We resize each camera image from ``(H_in, W_in)`` to ``(H_out, W_out)`` with
``F.interpolate(mode="bilinear", align_corners=False)`` — the **forward** of which is
deterministic on CUDA (there is no backward path: images are input data, not
parameters, even in the full-model ablation). ``align_corners=False`` maps an output
pixel ``i`` to the input continuous coordinate ``(i + 0.5)/s - 0.5`` (``s = W_out/W_in``),
i.e. content at input pixel ``u`` appears at output ``u·s + (0.5·s − 0.5)``.

We rescale the intrinsics with the **same affine** so the projection matches the
resized image to float precision (not just "<1px"):

    u_out = fx·u_in + tx,   tx = 0.5·fx − 0.5      (fx = W_out / W_in)
    v_out = fy·v_in + ty,   ty = 0.5·fy − 0.5      (fy = H_out / H_in)

As a 3×3 pixel-space affine on ``K`` and a 4×4 left-multiply on ``lidar2img``
(``lidar2img`` maps a lidar point to the homogeneous ``[u·d, v·d, d, 1]``, so the
``tx,ty`` ride on the depth component ``d = proj[2]``)::

    A = [[fx, 0, tx], [0, fy, ty], [0, 0, 1]]            cam_intrinsics' = A @ K
    M = [[fx,0,tx,0],[0,fy,ty,0],[0,0,1,0],[0,0,0,1]]    lidar2img'      = M @ lidar2img

This is **exact** (the ≤1px gate measures ~1e-4 px residual vs the resize center
mapping). Aspect ratio is NOT preserved (``fx ≠ fy`` for 1600×900→704×256) — that is
fine because ``fx, fy`` are handled independently and the calibration follows exactly;
aspect-preserving resize-then-crop (BEVFusion-MIT) is a later refinement, hooked via
``crop`` but off by default. RNG-free by construction; no random augmentation in the
clean baseline (seeded aug is a deferred hook).
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet-1k RGB statistics (torchvision Swin_T / ResNet18 IMAGENET1K_V1 transforms).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Recommended input (BEVFusion-MIT): H=256, W=704.
DEFAULT_IMAGE_HW = (256, 704)


def resize_affine(
    H_in: int, W_in: int, H_out: int, W_out: int, device=None, dtype=torch.float64
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A_3x3, M_4x4)`` for the half-pixel-correct resize (see module doc).

    ``A`` rescales a pixel-space ``K``; ``M`` left-multiplies ``lidar2img``. float64
    by default for calibration accuracy (cast to f32 when stored)."""
    fx = W_out / W_in
    fy = H_out / H_in
    tx = 0.5 * fx - 0.5
    ty = 0.5 * fy - 0.5
    A = torch.tensor([[fx, 0.0, tx], [0.0, fy, ty], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    M = torch.tensor(
        [[fx, 0.0, tx, 0.0], [0.0, fy, ty, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    )
    return A, M


class ImagePreprocessor(nn.Module):
    """uint8 native images + native calibration → normalized float images + rescaled
    calibration. Stateless (no parameters), RNG-free, deterministic.

    Forward consumes the collated batch tensors and returns a dict with
    ``images`` ``[B,6,3,H_out,W_out]`` f32 (ImageNet-normalized), ``lidar2img``
    ``[B,6,4,4]`` f32 (rescaled), and ``cam_intrinsics`` ``[B,6,3,3]`` f32 (rescaled)."""

    def __init__(self, image_hw: Tuple[int, int] = DEFAULT_IMAGE_HW):
        super().__init__()
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 1, 3, 1, 1)
        # Buffers so .to(device)/.cuda() move them and they're not trainable params.
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def forward(
        self,
        images_u8: torch.Tensor,      # [B,6,3,H_in,W_in] uint8
        lidar2img: torch.Tensor,      # [B,6,4,4] f32 (native resolution)
        cam_intrinsics: torch.Tensor,  # [B,6,3,3] f32 (native resolution)
    ) -> dict:
        B, N, C, H_in, W_in = images_u8.shape
        H_out, W_out = self.image_hw
        device = images_u8.device

        # --- resize (bilinear, align_corners=False; forward is deterministic) ---
        x = images_u8.reshape(B * N, C, H_in, W_in).to(torch.float32) / 255.0
        x = F.interpolate(x, size=(H_out, W_out), mode="bilinear", align_corners=False)
        x = x.reshape(B, N, C, H_out, W_out)
        # --- ImageNet normalization ---
        x = (x - self._mean) / self._std

        # --- rescale calibration with the same affine (half-pixel-correct) ---
        A, M = resize_affine(H_in, W_in, H_out, W_out, device=device, dtype=torch.float32)
        # broadcast matmul over [B,6]
        K2 = torch.matmul(A.view(1, 1, 3, 3), cam_intrinsics)        # [B,6,3,3]
        l2i2 = torch.matmul(M.view(1, 1, 4, 4), lidar2img)           # [B,6,4,4]
        return {
            "images": x.contiguous(),
            "cam_intrinsics": K2.contiguous(),
            "lidar2img": l2i2.contiguous(),
            "image_hw": (H_out, W_out),
        }
