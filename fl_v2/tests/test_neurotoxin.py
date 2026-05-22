"""WS3 — Neurotoxin gradient-masking tests.

Covers `topk_mask_from_proxy` (global magnitude ranking, ratio extremes)
and `project_grad_away_from_topk` (zeroing the masked coordinates),
including the null config: neurotoxin-topk-ratio 0.0 masks nothing, so a
Neurotoxin run at ratio 0.0 is bit-identical to the plain pixel backdoor.
"""

import torch
import torch.nn as nn

from fl_v2.attacks_defenses.attacks.neurotoxin import (
    project_grad_away_from_topk,
    topk_mask_from_proxy,
)


def _proxy(values):
    return {"w": torch.tensor(values, dtype=torch.float32)}


def test_ratio_zero_masks_nothing():
    mask = topk_mask_from_proxy(_proxy([5.0, -4.0, 3.0, 1.0]), 0.0)
    assert not mask["w"].any()


def test_ratio_one_masks_everything():
    mask = topk_mask_from_proxy(_proxy([5.0, -4.0, 3.0, 1.0]), 1.0)
    assert mask["w"].all()


def test_topk_selects_largest_magnitude():
    # top-2 of |[5, -4, 3, 1]| -> coordinates 0 and 1
    mask = topk_mask_from_proxy(_proxy([5.0, -4.0, 3.0, 1.0]), 0.5)
    assert mask["w"].tolist() == [True, True, False, False]


def test_global_ranking_across_tensors():
    # The mask is a fraction of the whole model, not of each tensor.
    proxy = {
        "a": torch.tensor([10.0, 9.0], dtype=torch.float32),
        "b": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    mask = topk_mask_from_proxy(proxy, 0.5)  # top-2 of 4 -> both 'a' coords
    assert mask["a"].tolist() == [True, True]
    assert mask["b"].tolist() == [False, False]


def test_project_zeros_masked_grad_coords():
    lin = nn.Linear(4, 1, bias=False)
    lin.weight.grad = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    project_grad_away_from_topk(
        lin, {"weight": torch.tensor([[True, False, True, False]])}
    )
    assert lin.weight.grad.tolist() == [[0.0, 2.0, 0.0, 4.0]]


def test_ratio_zero_mask_is_a_noop_when_applied():
    """Null config: neurotoxin-topk-ratio 0.0 leaves every gradient intact."""
    lin = nn.Linear(3, 1, bias=False)
    original = torch.tensor([[1.0, 2.0, 3.0]])
    lin.weight.grad = original.clone()
    mask = topk_mask_from_proxy({"weight": torch.tensor([[5.0, 4.0, 3.0]])}, 0.0)
    project_grad_away_from_topk(lin, mask)
    assert torch.equal(lin.weight.grad, original)
