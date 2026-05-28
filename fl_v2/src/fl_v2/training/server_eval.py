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

    When ``trigger_fn`` is provided, two backdoor-attribution metrics
    extend the returned dict (Cycle-03 WS-A):

    - ``asr_src<c>`` / ``asr_src<c>_n`` per non-target source class
      ``c``: lets offline analysis split global ASR into the trained-on
      and unseen-source subsets (Wave-1's single ASR averaged over
      heterogeneous source classes).
    - ``clean_floor_to_target`` — fraction of non-target test samples
      classified as ``target_label`` WITHOUT the trigger applied.
      Derived ``backdoor_attribute_asr = asr - clean_floor_to_target``
      is the exact trigger-attributable ASR. Both reuse pass-1
      predictions; no extra forward pass.
    """
    model.eval()

    # Global accuracy accumulators
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Target-class accuracy accumulators
    tca_correct = 0
    tca_total = 0

    # ASR accumulators (overall + per-source-class + clean-floor)
    asr_success = 0
    asr_total = 0
    per_class_success: dict[int, int] = {}
    per_class_total: dict[int, int] = {}
    clean_floor_success = 0
    clean_floor_total = 0

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
                non_target_labels_cpu = labels[non_target_mask]
                triggered = trigger_fn(images[non_target_mask]).to(device)
                asr_preds = model(triggered).argmax(dim=1)
                success_mask_cpu = (asr_preds == target_label).cpu()

                asr_success += int(success_mask_cpu.sum().item())
                asr_total += int(non_target_mask.sum().item())

                # Per-source-class ASR breakdown.
                for src_class_t in non_target_labels_cpu.unique():
                    c = int(src_class_t.item())
                    sm = non_target_labels_cpu == src_class_t
                    per_class_success[c] = per_class_success.get(c, 0) + int(
                        success_mask_cpu[sm].sum().item()
                    )
                    per_class_total[c] = per_class_total.get(c, 0) + int(
                        sm.sum().item()
                    )

                # Clean-floor: fraction of non-target samples classified as
                # ``target_label`` WITHOUT the trigger. Reuses pass-1 preds
                # (same image set, no second forward) so the only added
                # cost is two CPU-side reductions.
                non_target_mask_dev = labels_dev != target_label
                clean_non_target_preds = preds[non_target_mask_dev]
                clean_floor_success += int(
                    (clean_non_target_preds == target_label).sum().item()
                )
                clean_floor_total += int(non_target_mask_dev.sum().item())

    result: Dict[str, float] = {
        "test_loss": total_loss / total_samples if total_samples > 0 else 0.0,
        "test_accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        "num-test-examples": float(total_samples),
        "target_class_clean_accuracy": tca_correct / tca_total if tca_total > 0 else 0.0,
        "target_class_num_samples": float(tca_total),
    }

    if trigger_fn is not None:
        asr_value = asr_success / asr_total if asr_total > 0 else 0.0
        clean_floor = (
            clean_floor_success / clean_floor_total if clean_floor_total > 0 else 0.0
        )
        result["asr"] = asr_value
        result["asr_num_samples"] = float(asr_total)
        # Per-source-class breakdown (sorted for stable column order in CSV).
        for c in sorted(per_class_total.keys()):
            n_c = per_class_total[c]
            result[f"asr_src{c}"] = per_class_success[c] / n_c if n_c > 0 else 0.0
            result[f"asr_src{c}_n"] = float(n_c)
        # Clean floor + derived trigger-attributable ASR.
        result["clean_floor_to_target"] = clean_floor
        result["clean_floor_num_samples"] = float(clean_floor_total)
        result["backdoor_attribute_asr"] = asr_value - clean_floor

    return result
