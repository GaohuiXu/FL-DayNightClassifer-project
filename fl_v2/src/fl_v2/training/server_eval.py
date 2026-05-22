"""Combined server-side evaluation — single dataloader pass for all metrics."""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def server_evaluate(
    model: nn.Module,
    testloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_label: int = 0,
    trigger_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    """Evaluate test accuracy, target-class accuracy, and ASR in one pass.

    Instead of iterating the testloader 3 times (evaluate + TCA + ASR),
    this does 1 iteration with up to 2 forward passes per batch
    (clean + triggered).
    """
    model.eval()

    # Global accuracy accumulators
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Target-class accuracy accumulators
    tca_correct = 0
    tca_total = 0

    # ASR accumulators
    asr_success = 0
    asr_total = 0

    for images, labels in testloader:
        images_dev = images.to(device)
        labels_dev = labels.to(device)

        # --- Forward pass 1: clean images ---
        logits = model(images_dev)
        loss = criterion(logits, labels_dev)
        preds = logits.argmax(dim=1)

        batch_size = labels_dev.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels_dev).sum().item()
        total_samples += batch_size

        # --- TCA: extract from same predictions ---
        target_mask = labels_dev == target_label
        if target_mask.sum() > 0:
            tca_correct += (preds[target_mask] == target_label).sum().item()
            tca_total += target_mask.sum().item()

        # --- Forward pass 2: triggered images (only if backdoor active) ---
        if trigger_fn is not None:
            # Mask + index on CPU because trigger_fn operates on the CPU
            # `images` tensor (the global testloader yields CPU tensors);
            # the masked subset is moved to device only for the forward pass.
            # Bit-equivalent to indexing `images_dev[labels_dev != target_label]`
            # because the pixel trigger is a constant assignment, but
            # keeping the masking on CPU avoids an unnecessary device
            # round-trip on every batch.
            non_target_mask = labels != target_label
            if non_target_mask.sum() > 0:
                triggered = trigger_fn(images[non_target_mask]).to(device)
                asr_preds = model(triggered).argmax(dim=1)
                asr_success += (asr_preds == target_label).sum().item()
                asr_total += non_target_mask.sum().item()

    result: Dict[str, float] = {
        "test_loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "test_accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "num-test-examples": float(total_samples),
        "target_class_clean_accuracy": tca_correct / tca_total if tca_total > 0 else 0.0,
        "target_class_num_samples": float(tca_total),
    }

    if trigger_fn is not None:
        result["asr"] = asr_success / asr_total if asr_total > 0 else 0.0
        result["asr_num_samples"] = float(asr_total)

    return result
