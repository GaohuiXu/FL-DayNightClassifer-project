"""T2 task integration + skeleton-preservation + GPU op guards.

Covers: ``dummy_regression`` byte-identity after the loop generalization (committed
golden), ``NuScenesDetectionTask`` registration / derived N / client materialization, the
generalized loop training a detection dict-batch end-to-end, loader determinism across
``num_workers``, and the GPU op guards (#76176 ``index_copy_`` + float-CUDA ``cumsum``).
The full-training **bit-identical-weights** gate is the A40 SLURM job — NOT here.
"""
from __future__ import annotations

import pytest
import torch

from fl_v3.utils.runtime import enforce_determinism, seed_everything

CUDA = torch.cuda.is_available()

# Committed golden: the dummy_regression aggregated-state checksum. The loop's tensor
# path is provably byte-identical after the T2 generalization (same op sequence:
# X.to/y.to → model(X) → criterion(out,y) → y.size(0)); this pins it so any future drift
# fails loudly.
DUMMY_AGG_GOLDEN = "d2d819fee9a54fc302a9d6c9d0ac4e4d875629a0a16e75f2328f28b7f63cd7cc"

_DUMMY_CFG = {
    "task-type": "dummy_regression", "seed": 42, "device": "cpu", "num-clients": 4,
    "num-local-epochs": 1, "batch-size": 8, "learning-rate": 0.01, "weight-decay": 0.0,
    "num-workers": 0, "loss": "mse",
}


def test_dummy_regression_byte_identity_golden():
    """dummy_regression aggregated checksum unchanged after the loop generalization."""
    from fl_v3.engine.local_runner import run_clean_round

    r = run_clean_round(dict(_DUMMY_CFG), defense="none", server_round=1)
    assert r["agg_checksum"] == DUMMY_AGG_GOLDEN, (
        f"dummy_regression byte-identity broke: {r['agg_checksum']} != golden"
    )


# --- detection task wiring ---
_DET_CFG = {
    "nuscenes-cache-dir": "./fl_outputs/nuscenes/info_cache", "nuscenes-version": "v1.0-mini",
    "nuscenes-train-split": "mini_train", "nuscenes-val-split": "mini_val",
    "nuscenes-partition-mode": "iid", "nuscenes-num-clients": 8, "seed": 42,
    "batch-size": 1, "num-workers": 0,
    "det-camera-backbone": "resnet18", "det-pretrained-backbone": False,
    "det-neck-channels": 32, "det-context-channels": 16, "det-lidar-channels": 16,
    "det-fusion-channels": 32, "det-bev-neck-channels": 64, "det-head-channels": 16,
    "det-eval-limit": 2,
}


def test_detection_task_registered():
    from fl_v3.training.tasks import get_task, available_tasks

    assert "nuscenes_detection" in available_tasks()
    assert get_task("nuscenes_detection").name == "nuscenes_detection"


def test_detection_config_rejects_legacy_model_mode_alias():
    from fl_v3.training.tasks import _det_config_from_run

    with pytest.raises(ValueError, match="model-mode"):
        _det_config_from_run({**_DET_CFG, "model-mode": "camera"})


def test_num_clients_iid_is_requested(mini_cache_dir):
    from fl_v3.training.tasks import get_task

    assert get_task("nuscenes_detection").num_clients(dict(_DET_CFG)) == 8


def test_client_data_materializes_dict_batch(mini_cache_dir):
    from fl_v3.training.tasks import get_task

    task = get_task("nuscenes_detection")
    cdata = task.client_data(0, dict(_DET_CFG))
    assert cdata.num_train > 0
    batch = next(iter(cdata.trainloader))
    assert isinstance(batch, dict)
    assert batch["images"].shape[1:] == (6, 3, 900, 1600)
    assert batch["lidar_points"].shape[1] == 6  # batch-index column
    assert isinstance(batch["gt_boxes"], list)


@pytest.mark.skipif(not CUDA, reason="central smoke needs a GPU (heavy on CPU)")
def test_generalized_loop_trains_detection_batch(mini_cache_dir):
    """The generalized loop trains the multimodal dict-batch end-to-end (train + eval)."""
    from fl_v3.training.tasks import get_task
    from fl_v3.training.loop import train_one_epoch, _move_to_device

    enforce_determinism(strict=True); seed_everything(0)
    dev = torch.device("cuda")
    task = get_task("nuscenes_detection")
    cfg = dict(_DET_CFG)
    model = task.build_model(cfg).to(dev)
    crit = task.build_criterion(cfg)
    loader = task.eval_loader(cfg)  # 2-sample shuffle=False loader
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    m = train_one_epoch(model, loader, crit, opt, dev)
    assert "loss" in m and m["num_samples"] == 2
    out = task.evaluate(model, loader, crit, dev, cfg)
    assert "eval_loss" in out and "proxy_recall_at_2m" in out
    assert out["num-eval-examples"] == 2


def test_loader_determinism_num_workers(mini_cache_dir):
    """The eval loader yields a byte-identical first batch at num_workers 0 and 2."""
    from fl_v3.training.tasks import get_task

    task = get_task("nuscenes_detection")
    b0 = next(iter(task.eval_loader(dict(_DET_CFG, **{"num-workers": 0}))))
    try:
        b2 = next(iter(task.eval_loader(dict(_DET_CFG, **{"num-workers": 2}))))
    except Exception as e:  # multiprocessing unavailable in this env
        pytest.skip(f"num_workers>0 unavailable: {e}")
    assert torch.equal(b0["images"], b2["images"])
    assert torch.equal(b0["lidar_points"], b2["lidar_points"])
    assert torch.equal(b0["gt_boxes"][0], b2["gt_boxes"][0])


# --- GPU op guards (#76176 + float cumsum). Run on any CUDA (T4 ok); the cross-arch
#     bit-identity gate is the A40 SLURM job. ---
@pytest.mark.skipif(not CUDA, reason="GPU guard requires CUDA")
def test_index_copy_scatter_76176_gpu():
    """index_copy_ scatter of a known pillar yields the correct non-zero cell under strict
    deterministic mode (the #76176 silent-no-op would leave it zero)."""
    enforce_determinism(strict=True)
    dev = torch.device("cuda")
    C, HW = 4, 100
    canvas = torch.zeros(HW, C, device=dev)
    idx = torch.tensor([37], device=dev)
    val = torch.arange(1, C + 1, device=dev, dtype=torch.float32).view(1, C)
    canvas.index_copy_(0, idx, val)
    assert torch.equal(canvas[37], val[0]), "index_copy_ did not write (the #76176 no-op)"
    assert canvas.sum() == val.sum(), "extra cells written"
    # the BROKEN advanced-index form would silently no-op on CUDA under det-mode:
    broken = torch.zeros(HW, C, device=dev)
    broken[idx] = val  # may no-op (#76176) — we DON'T rely on this; document the hazard
    # (no assertion on `broken`; the point is index_copy_ above is the correct path)


@pytest.mark.skipif(not CUDA, reason="GPU guard requires CUDA")
def test_float_cumsum_gpu_deterministic():
    """float-CUDA cumsum does NOT raise under strict mode and is torch.equal fwd+bwd
    (the LSS splat relies on this; the ARM/H200 rebuild must re-run this guard)."""
    enforce_determinism(strict=True)
    dev = torch.device("cuda")

    def run():
        seed_everything(0)
        x = torch.randn(5000, 8, device=dev, requires_grad=True)
        y = x.cumsum(0)
        y.sum().backward()
        return y.detach().clone(), x.grad.detach().clone()

    y1, g1 = run()
    y2, g2 = run()
    assert torch.equal(y1, y2), "float CUDA cumsum forward non-deterministic"
    assert torch.equal(g1, g2), "float CUDA cumsum backward non-deterministic"
