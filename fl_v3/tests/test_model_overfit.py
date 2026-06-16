"""T2 falsifiable overfit + anti-collapse (the "model learns" gate).

On a fixed single real scene of K steps: ``final_loss ≤ 0.2·initial`` AND the GT-proximity
proxy improves (recall@2m ~0 → ≥0.5) AND an anti-collapse assertion (decoded boxes above
threshold > 0 AND ≥ a fraction of the scene's GT matched to DISTINCT GT). Collapse-to-
background (loss↓ with zero/one-coincidental box) fails. GPU-gated (heavy on CPU); the
device-agnostic *learning* signal is the point — the bit-identity gate is the A40 job.
"""
from __future__ import annotations

import pytest
import torch

from fl_v3.utils.runtime import enforce_determinism, seed_everything

CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="overfit is heavy; run on a GPU")
def test_overfit_single_scene_falsifiable(mini_cache_dir):
    from fl_v3.training.tasks import get_task, center_distance_proxy
    from fl_v3.training.loop import _move_to_device

    enforce_determinism(strict=True); seed_everything(42)
    dev = torch.device("cuda")
    cfg = {
        "nuscenes-cache-dir": "./fl_outputs/nuscenes/info_cache", "nuscenes-version": "v1.0-mini",
        "nuscenes-train-split": "mini_train", "nuscenes-val-split": "mini_val",
        "nuscenes-partition-mode": "iid", "nuscenes-num-clients": 4, "seed": 42,
        "batch-size": 1, "num-workers": 0,
        # D1 setting: ImageNet-pretrained FROZEN backbone (cached on the login node) —
        # the FL-trained LSS/LiDAR/fusion/head learn on top of it. This is the real
        # headline configuration, not a random backbone.
        "det-camera-backbone": "resnet18", "det-pretrained-backbone": True,
        "det-neck-channels": 64, "det-context-channels": 32, "det-lidar-channels": 32,
        "det-fusion-channels": 64, "det-bev-neck-channels": 128, "det-head-channels": 32,
    }
    task = get_task("nuscenes_detection")
    cdata = task.client_data(0, cfg)
    seed_everything(42)
    try:
        model = task.build_model(cfg).to(dev)
    except Exception as e:  # offline / pretrained weights uncached
        pytest.skip(f"pretrained backbone weights unavailable (pre-cache on login node): {e}")
    crit = task.build_criterion(cfg)
    batch = _move_to_device(next(iter(cdata.trainloader)), dev)

    n_gt_car = int((batch["gt_labels"][0] == 0).sum().item())
    if n_gt_car < 3:
        pytest.skip(f"scene has too few cars ({n_gt_car}) for a falsifiable overfit")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    model.train()
    losses = []
    for _ in range(180):
        opt.zero_grad(); loss = crit(model(batch), batch); loss.backward(); opt.step()
        losses.append(float(loss))

    model.eval()
    with torch.no_grad():
        decoded = model.decode(model(batch), score_threshold=0.1)
        proxy = center_distance_proxy(decoded, batch, target_class=0)

    # (1) loss collapses to ≤ 0.2× initial
    assert losses[-1] <= 0.2 * losses[0], f"loss did not drop: {losses[0]:.3f}→{losses[-1]:.3f}"
    # (2) GT-proximity improves: recall@2m ≥ 0.5 (from ~0)
    assert proxy["recall_at"] >= 0.5, f"recall@2m {proxy['recall_at']:.3f} < 0.5"
    # (3) anti-collapse: decoded boxes above threshold > 0 AND ≥ half the GT matched DISTINCT
    assert proxy["n_decoded"] > 0, "collapse: no decoded boxes above threshold"
    assert proxy["matched_gt"] >= 0.5 * proxy["n_gt"], "collapse: too few DISTINCT GT matched"
