from __future__ import annotations

from torchvision import transforms

# 训练集做轻微增强，测试集只做确定性的预处理

def get_train_transforms(image_size: int = 32) -> transforms.Compose:
    """Return training transforms for GTSRB."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )


def get_eval_transforms(image_size: int = 32) -> transforms.Compose:
    """Return evaluation transforms for GTSRB."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )