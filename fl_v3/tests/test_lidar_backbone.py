"""Dense 2D LiDAR backbone tests (MCR P1 capacity lever).

Guards: (a) the module shape contract (H,W preserved so ConvFuser can concat); (b) DEFAULT-OFF six-task
topology (disabled ⇒ no LiDAR-backbone params, fuser width unchanged, 230-tensor layout intact); (c) the ON-path wiring
(fuser width threaded to 80+Cout, trainable count rises by the backbone's tensors); (d) finite/deterministic
forward. The static-AST ban over models/fusion/** auto-covers lidar_backbone.py via the existing sweep test."""
import torch

from fl_v3.models.fusion.lidar_backbone import LidarBackbone2D
from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
from fl_v3.training.tasks import trainable_state_dict


def _first_conv_in_channels(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m.in_channels
    raise AssertionError("no Conv2d in module")


def test_backbone_shape_preserved():
    bb = LidarBackbone2D(in_channels=64, out_channels=128)
    for H, W in [(256, 256), (128, 256), (512, 512)]:
        y = bb(torch.randn(2, 64, H, W))
        assert y.shape == (2, 128, H, W), f"{y.shape} for input {(H, W)} — H,W must be preserved"


def test_backbone_finite_and_deterministic():
    bb = LidarBackbone2D(in_channels=64, out_channels=128).eval()
    x = torch.randn(2, 64, 128, 128)
    y1, y2 = bb(x), bb(x)
    assert torch.isfinite(y1).all(), "backbone produced non-finite output"
    assert torch.equal(y1, y2), "backbone forward not deterministic"
    # empty-cloud canvas (all zeros, as lidar_encoder returns) must be handled
    assert torch.isfinite(bb(torch.zeros(1, 64, 64, 64))).all()


def test_param_count_meaningful():
    bb = LidarBackbone2D(in_channels=64, out_channels=128)
    n = sum(p.numel() for p in bb.parameters())
    assert 2.0e6 < n < 3.0e6, f"expected ~2.5M params (vs the 640-param PFN), got {n}"


def test_deep_4stage_backbone():
    """num_stages=4 adds an H/8 down-stage (large-object RF at 0.2m): shape preserved, +9 tensors, more params."""
    bb3 = LidarBackbone2D(in_channels=64, out_channels=128, num_stages=3)
    bb4 = LidarBackbone2D(in_channels=64, out_channels=128, num_stages=4)
    assert len(list(bb3.parameters())) + 9 == len(list(bb4.parameters())), "4-stage = +9 tensors (s4=6 + l4=3)"
    y = bb4(torch.randn(2, 64, 512, 512))          # the 0.2m grid
    assert y.shape == (2, 128, 512, 512) and torch.isfinite(y).all()


def test_default_off_preserves_six_task_topology():
    """Backbone OFF preserves six-task head topology and excludes LiDAR backbone."""
    off = BEVFusionDetector(DetectorConfig(pretrained_backbone=False))
    names = [n for n, _ in off.named_parameters()]
    assert not any(n.startswith("lidar_backbone") for n in names), "OFF must not construct lidar_backbone"
    assert off.lidar_backbone is None
    assert _first_conv_in_channels(off.fusion) == 80 + 64, "OFF fuser width must be unchanged (144)"
    state = trainable_state_dict(off)
    assert len(state) == 230
    head_count = sum(name.startswith("head.") for name in state)
    assert head_count == 183
    assert head_count - 15 == 168  # approved six-task increment over legacy head


def test_on_path_wiring():
    """Backbone ON ⇒ fuser width 80+128=208, params present, trainable count rises by the backbone's 30 tensors."""
    # 30 = 3 stages × (2 conv-weights + 2 GN weight/bias pairs = 6) + 3 laterals × (conv + GN = 3) + smooth (3).
    off = BEVFusionDetector(DetectorConfig(pretrained_backbone=False))
    on = BEVFusionDetector(DetectorConfig(pretrained_backbone=False, lidar_backbone=True))
    assert on.lidar_backbone is not None
    assert _first_conv_in_channels(on.fusion) == 80 + 128, "ON fuser must widen to 80+Cout=208"
    n_off, n_on = len(trainable_state_dict(off)), len(trainable_state_dict(on))
    assert n_on == n_off + 30, f"backbone adds 30 trainable tensors (got {n_on - n_off})"
    # param_table accounts for the backbone when ON
    assert "lidar_backbone" in on.param_table()
    assert "lidar_backbone" not in off.param_table()
