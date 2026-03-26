from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class GTSRBResNet18(nn.Module):
    """ResNet18 backbone for GTSRB, trained from scratch.

    Modified for 32x32 inputs: conv1 replaced with 3x3 stride-1,
    maxpool removed. Standard practice for small-image datasets
    (CIFAR-10/GTSRB ResNet variants).
    """

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()
        base = resnet18(weights=None)

        # Replace 7x7 stride-2 conv1 with 3x3 stride-1 for 32x32 inputs
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.features = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )
        self.avgpool = base.avgpool  # AdaptiveAvgPool2d(1, 1)
        self.fc = nn.Linear(512, num_classes)

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
