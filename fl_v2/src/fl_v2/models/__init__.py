import torch.nn as nn

from fl_v2.models.classifier import GTSRBClassifier
from fl_v2.models.resnet import GTSRBResNet18


def create_model(
    model_type: str,
    num_classes: int = 43,
    pretrained: bool = False,
    canonical_conv1: bool = False,
) -> nn.Module:
    """Factory function for model selection.

    ``pretrained=True`` and ``canonical_conv1=True`` are only supported by
    ``model_type='resnet18'``; the small custom CNN has no ImageNet analog
    and explicitly rejects either flag.
    """
    if model_type == "cnn":
        if pretrained:
            raise ValueError(
                "pretrained-init=true is not supported for model-type='cnn'. "
                "Use model-type='resnet18' or set pretrained-init=false."
            )
        if canonical_conv1:
            raise ValueError(
                "canonical-conv1=true is not supported for model-type='cnn'."
            )
        return GTSRBClassifier(num_classes=num_classes)
    elif model_type == "resnet18":
        return GTSRBResNet18(
            num_classes=num_classes,
            pretrained=pretrained,
            canonical_conv1=canonical_conv1,
        )
    else:
        raise ValueError(
            f"Unknown model-type: {model_type!r}. Expected 'cnn' or 'resnet18'."
        )


__all__ = ["GTSRBClassifier", "GTSRBResNet18", "create_model"]
