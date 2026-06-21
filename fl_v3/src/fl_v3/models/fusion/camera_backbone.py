"""Camera backbone — frozen ImageNet Swin-T (D1) + ResNet-18 fallback (fl_v3 T2).

Per the Architecture table + D1: the camera backbone is an **ImageNet-pretrained
Swin-T**, **frozen** in the primary FL setting (all AD-specific learning is in the
FL-trained LSS-depth / LiDAR / fusion / neck / head). **ResNet-18** is the fast
bring-up fallback (11.7M vs 28.3M params; recommended for the T3 wall-clock bring-up).

## "Frozen means frozen through mode switches" (the D1/D6 trap)

A ``requires_grad=False``-only freeze is **not enough**: ``model.train()`` recurses
into children and flips every ``BatchNorm`` to training mode, which **updates running
mean/var on the next forward** even with no gradient. Swin-T uses ``LayerNorm`` (so
this is moot for Swin), but the **ResNet-18 fallback is all BatchNorm** — that is
exactly where a grad-only freeze silently breaks D1/D6 (the running stats drift,
making the "frozen" backbone non-deterministic across rounds). We therefore override
:meth:`train` on this module so that **when frozen it stays in eval mode regardless of
any parent ``.train()`` call** — the BN running-stat bytes are unchanged after a train
step (asserted on the ResNet path by ``test_model_freeze``).

## Determinism

torchvision ``swin_t`` v0.22.1 uses **manual fp32 math attention**
(``matmul → +rel-pos-bias → softmax → matmul``) — it does NOT call
``scaled_dot_product_attention``, so no flash/mem-efficient kernel is reachable
(verified vs source; SPEC DT2-A). No ``scatter_add`` / ``grid_sample``. Weights are
``IMAGENET1K_V1``, pre-cached on the login node (``TORCH_HOME`` pinned) since compute
nodes are offline. The param-count asserts (28,288,354 / 11,689,512) double as the
weights-loaded check.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint  # MCR P1: gradient checkpointing on a trained Swin backbone

SWIN_T_PARAMS = 28_288_354
RESNET18_PARAMS = 11_689_512


class CameraBackbone(nn.Module):
    """Multi-scale 2D feature extractor over per-camera images.

    ``forward(x)`` takes ``[BN, 3, H, W]`` (cameras already folded into the batch)
    and returns a list of feature maps ``[BN, C_i, H/s_i, W/s_i]`` at strides
    ``(4, 8, 16, 32)`` — the inputs to the LSS-FPN neck.
    """

    def __init__(self, name: str = "swin_t", frozen: bool = True, pretrained: bool = True,
                 activation_checkpoint: bool = False, sdpa_attention: bool = False):
        super().__init__()
        self.name = name
        self.frozen = bool(frozen)
        # MCR P2 (D16 envelope): route Swin windowed attention through F.scaled_dot_product_attention
        # (rel-pos-bias as attn_mask → EFFICIENT backend; deterministic MATH under precision=fp32).
        # Numerically equivalent to torchvision's manual core (validated). swin_t only; no-op for resnet18.
        self.sdpa_attention = bool(sdpa_attention)
        # MCR P1 (D16 envelope): gradient/activation checkpointing on a TRAINED Swin backbone — trades
        # ~20-30% recompute for the VRAM headroom that lets bf16 backbone-training fit a useful batch.
        # No effect when frozen (no backward through the backbone) or in eval; off ⇒ byte-identical.
        self.activation_checkpoint = bool(activation_checkpoint)
        self.strides = (4, 8, 16, 32)
        if name == "swin_t":
            self._build_swin(pretrained)
            self.out_channels = [96, 192, 384, 768]
        elif name == "resnet18":
            self._build_resnet18(pretrained)
            self.out_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"unknown camera-backbone {name!r} (use swin_t|resnet18)")

        if self.frozen:
            for p in self.parameters():
                p.requires_grad_(False)
            # Force eval now AND keep it there through any parent .train() (below).
            super().train(False)

    # --- builders ---
    def _build_swin(self, pretrained: bool) -> None:
        from torchvision.models import swin_t, Swin_T_Weights

        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        net = swin_t(weights=weights)
        # Full-model param count = the weights-loaded check (with weights=
        # IMAGENET1K_V1, torchvision RAISES if the .pth is not cached — offline
        # compute nodes — so a successful build + the canonical count proves the
        # ImageNet weights loaded). The detector uses only the feature-extractor
        # stages (drops net.norm + net.head: the 1000-class classifier); the used
        # count is reported separately by param_count().
        self.full_backbone_params = sum(p.numel() for p in net.parameters())
        # torchvision Swin `.features` is a Sequential; outputs are NHWC. We tap
        # after each stage (odd indices are the SwinBlock stages, even are
        # PatchEmbed/PatchMerging): taps at features[1,3,5,7] → strides 4,8,16,32.
        self._swin_features = net.features
        self._swin_tap_after = {1, 3, 5, 7}
        if self.sdpa_attention:
            from fl_v3.models.fusion.swin_sdpa import apply_sdpa_to_swin
            apply_sdpa_to_swin(self._swin_features)

    def _build_resnet18(self, pretrained: bool) -> None:
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.full_backbone_params = sum(p.numel() for p in net.parameters())
        self._stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self._layer1 = net.layer1  # stride 4, 64ch
        self._layer2 = net.layer2  # stride 8, 128ch
        self._layer3 = net.layer3  # stride 16, 256ch
        self._layer4 = net.layer4  # stride 32, 512ch

    # --- the freeze that survives model.train() (D1/D6) ---
    def train(self, mode: bool = True):  # type: ignore[override]
        if self.frozen:
            # Stay in eval forever; recurses eval into all children (incl. ResNet BN),
            # so running stats never update on a forward inside model.train().
            return super().train(False)
        return super().train(mode)

    # --- forward ---
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.name == "swin_t":
            feats: List[torch.Tensor] = []
            h = x
            # Checkpoint per Swin stage only when the backbone is TRAINED + in train mode (otherwise there is
            # no backward to recompute for, and eval/frozen stay byte-identical). use_reentrant=False is the
            # modern API (composes with autocast + supports the no-grad eval pass).
            ckpt = self.activation_checkpoint and (not self.frozen) and self.training
            for i, stage in enumerate(self._swin_features):
                h = torch.utils.checkpoint.checkpoint(stage, h, use_reentrant=False) if ckpt else stage(h)
                if i in self._swin_tap_after:
                    feats.append(h.permute(0, 3, 1, 2).contiguous())  # NHWC→NCHW
            return feats
        # resnet18
        h = self._stem(x)
        f1 = self._layer1(h)
        f2 = self._layer2(f1)
        f3 = self._layer3(f2)
        f4 = self._layer4(f3)
        return [f1, f2, f3, f4]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
