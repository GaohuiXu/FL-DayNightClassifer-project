import torch.nn as nn

from fl_v2.models.classifier import GTSRBClassifier
from fl_v2.models.resnet import GTSRBResNet18


def create_model(model_type: str, num_classes: int = 43) -> nn.Module:
    """Factory function for model selection."""
    if model_type == "cnn":
        return GTSRBClassifier(num_classes=num_classes)
    elif model_type == "resnet18":
        return GTSRBResNet18(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model-type: {model_type!r}. Expected 'cnn' or 'resnet18'."
        )


__all__ = ["GTSRBClassifier", "GTSRBResNet18", "create_model"]
