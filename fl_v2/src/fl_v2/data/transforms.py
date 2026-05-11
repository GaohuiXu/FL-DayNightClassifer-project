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


# Split variants for `TransformedSubset`'s pre_transform cache.
#
# `Resize` is purely deterministic (PIL bilinear is byte-stable for fixed
# input + fixed output size). Pre-applying it once at __init__ produces
# bit-equivalent output to applying it per __getitem__: the stochastic
# ops downstream (RandomRotation, ColorJitter) see the same input pixels
# they would have seen with the fused chain, and the seeded RNG state
# at their entry is unchanged. The bit-equality unit test in
# `tests/test_dataloader_determinism.py` enforces this contract.

def get_train_transforms_split(image_size: int = 32):
    """Return ``(pre, post)`` for the train chain.

    ``pre`` is ``Resize`` (deterministic, run once and cached).
    ``post`` is ``RandomRotation`` + ``ColorJitter`` + ``ToTensor`` +
    ``Normalize`` (the stochastic + final tensor ops, run per
    ``__getitem__``).
    """
    pre = transforms.Resize((image_size, image_size))
    post = transforms.Compose(
        [
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
    return pre, post


def get_eval_transforms_split(image_size: int = 32):
    """Return ``(pre, post)`` for the eval chain.

    Both halves are deterministic; we still split on ``Resize`` for
    cache-residency consistency with the train path.
    """
    pre = transforms.Resize((image_size, image_size))
    post = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ]
    )
    return pre, post