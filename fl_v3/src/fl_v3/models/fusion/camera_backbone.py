"""Camera backbone — O-017 Swin-T taps plus legacy ResNet-18 support.

Swin-T exposes the four real feature stages at strides ``(4, 8, 16, 32)``.  The
S03 FPN consumes every declared tap, so there is no computed-but-permanently-dead
stage.  Whether the backbone is frozen or trainable remains an explicit constructor
choice for later protocol integration; S03 does not silently encode a CL/FL policy.

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

## Determinism and initialization

The validated Arrhenius environment and exact torchvision version are recorded in
``docs/env.md`` and the S03 evidence. The stock path uses torchvision's manual Swin
attention core rather than ``scaled_dot_product_attention``; operation dtypes still
follow the surrounding PyTorch autocast policy. ``sdpa_attention=True`` is a
separate, explicit implementation path.

``pretrained=True`` asks torchvision for ``IMAGENET1K_V1`` and therefore requires
the corresponding cached checkpoint on offline compute nodes. Parameter counts
check architecture construction only; they do not attest which weight bytes were
loaded. Runs that rely on pretrained initialization must record that choice and
checkpoint/cache identity separately.
"""
from __future__ import annotations

from typing import List, Sequence

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
                 activation_checkpoint: bool = False, sdpa_attention: bool = False,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 output_layer_norm: bool = False):
        super().__init__()
        self.name = name
        self.frozen = bool(frozen)
        # MCR P2 (D16 envelope): route Swin windowed attention through F.scaled_dot_product_attention
        # (rel-pos-bias as attn_mask → EFFICIENT backend; deterministic MATH under precision=fp32).
        # Numerically equivalent to torchvision's manual core (validated). swin_t only; no-op for resnet18.
        self.sdpa_attention = bool(sdpa_attention)
        # MCR P1 (D16 envelope): gradient/activation checkpointing on a TRAINED Swin backbone — trades
        # ~20-30% recompute for the VRAM headroom that lets fp16 backbone-training fit a useful batch.
        # No effect when frozen (no backward through the backbone) or in eval; off ⇒ byte-identical.
        self.activation_checkpoint = bool(activation_checkpoint)
        self.out_indices = tuple(int(index) for index in out_indices)
        if not self.out_indices or len(self.out_indices) != len(set(self.out_indices)):
            raise ValueError("camera backbone out_indices must be non-empty and unique")
        if any(index < 0 or index > 3 for index in self.out_indices):
            raise ValueError("camera backbone out_indices must be within [0,3]")
        self.output_layer_norm = bool(output_layer_norm)
        all_strides = (4, 8, 16, 32)
        if name == "swin_t":
            self._build_swin(pretrained)
            all_channels = [96, 192, 384, 768]
        elif name == "resnet18":
            self._build_resnet18(pretrained)
            all_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"unknown camera-backbone {name!r} (use swin_t|resnet18)")
        self.strides = tuple(all_strides[index] for index in self.out_indices)
        self.out_channels = [all_channels[index] for index in self.out_indices]
        self.stage_output_norms = nn.ModuleDict()
        if self.output_layer_norm:
            if self.name != "swin_t":
                raise ValueError("output_layer_norm is defined only for Swin")
            self.stage_output_norms.update({
                str(index): nn.LayerNorm(all_channels[index])
                for index in self.out_indices
            })

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
        # Record the full architecture size before dropping net.norm/net.head.
        # This count is an architecture check, not evidence that a particular
        # pretrained checkpoint was loaded. The used feature-stage count is
        # reported separately by param_count().
        self.full_backbone_params = sum(p.numel() for p in net.parameters())
        # torchvision Swin `.features` is a Sequential; outputs are NHWC. We tap
        # after each stage (odd indices are the SwinBlock stages, even are
        # PatchEmbed/PatchMerging): taps at features[1,3,5,7] → strides 4,8,16,32.
        self._swin_features = net.features
        self._swin_tap_after = {1: 0, 3: 1, 5: 2, 7: 3}
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
                    stage_index = self._swin_tap_after[i]
                    if stage_index not in self.out_indices:
                        continue
                    if self.output_layer_norm:
                        h_out = self.stage_output_norms[str(stage_index)](h)
                    else:
                        h_out = h
                    feats.append(h_out.permute(0, 3, 1, 2).contiguous())  # NHWC→NCHW
            return feats
        # resnet18
        h = self._stem(x)
        f1 = self._layer1(h)
        f2 = self._layer2(f1)
        f3 = self._layer3(f2)
        f4 = self._layer4(f3)
        all_features = [f1, f2, f3, f4]
        return [all_features[index] for index in self.out_indices]

    def output_contract(self) -> dict:
        """Return the immutable feature interface consumed by the camera FPN."""
        return {
            "backbone": self.name,
            "layout": "NCHW",
            "strides": list(self.strides),
            "channels": list(self.out_channels),
            "n_levels": len(self.strides),
            "all_declared_levels_are_returned": True,
            "out_indices": list(self.out_indices),
            "output_layer_norm": self.output_layer_norm,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
