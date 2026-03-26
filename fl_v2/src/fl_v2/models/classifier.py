from __future__ import annotations

import torch
import torch.nn as nn


class GTSRBClassifier(nn.Module):
    """A simple CNN baseline for GTSRB classification."""

    def __init__(self, num_classes: int = 43) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32 -> 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # 16x16 -> 8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 8x8 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),                  # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim penultimate features before the classification head."""
        x = self.features(x)
        # classifier = [Flatten, Linear(2048,256), ReLU, Dropout, Linear(256,43)]
        for i, layer in enumerate(self.classifier):
            x = layer(x)
            if i == 3:  # after Dropout — 256-dim features
                return x
        return x