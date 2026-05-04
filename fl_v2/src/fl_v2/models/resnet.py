from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class GTSRBResNet18(nn.Module):
    """ResNet18 backbone for GTSRB.

    Modified for 32x32 inputs: conv1 replaced with 3x3 stride-1,
    maxpool removed. Standard practice for small-image datasets
    (CIFAR-10/GTSRB ResNet variants).

    With ``pretrained=True`` (Cycle 02 onward), ImageNet weights are loaded
    for ``bn1``, ``layer1``-``layer4``, and ``avgpool`` (no params). The
    swapped 3x3 ``conv1`` and the 43-class ``fc`` are randomly initialized
    by their constructors because ImageNet's 7x7 stride-2 conv1 and 1000-class
    head are architecturally / class-count-incompatible with GTSRB.
    """

    def __init__(
        self,
        num_classes: int = 43,
        pretrained: bool = False,
        canonical_conv1: bool = False,
    ) -> None:
        super().__init__()
        # Load ImageNet weights first; the conv1 swap below (when enabled)
        # replaces conv1 with a freshly-initialized 3x3 layer, leaving
        # bn1/layer1-4 with ImageNet weights. Order matters.
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        if not canonical_conv1:
            # Cycle 01 default: 3x3 stride-1 conv1, maxpool absent. Tuned for
            # 32x32 GTSRB inputs. Replacement conv1 is random-initialized
            # regardless of the `pretrained` flag.
            base.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )
        else:
            # Canonical mode: keep torchvision's 7x7 stride-2 conv1 +
            # 3x3 stride-2 maxpool (ImageNet-shaped first stage). Use with
            # image-size 64+. State-dict keys differ from the default
            # mode (extra `features.3` for maxpool); cross-loading between
            # modes is intentionally not supported.
            self.features = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4,
            )

        self.avgpool = base.avgpool  # AdaptiveAvgPool2d(1, 1)
        self.fc = nn.Linear(512, num_classes)

        conv1_kind = "canonical 7x7+maxpool" if canonical_conv1 else "modified 3x3"
        if pretrained:
            conv1_src = "ImageNet" if canonical_conv1 else "random-init"
            print(
                f"[model] resnet18 pretrained=True conv1={conv1_kind} "
                f"(conv1 {conv1_src}, bn1+layer1-4 from ImageNet, fc random-init)",
                flush=True,
            )
        else:
            print(
                f"[model] resnet18 pretrained=False conv1={conv1_kind} "
                "(random init)",
                flush=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-dim features before the classification head."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
