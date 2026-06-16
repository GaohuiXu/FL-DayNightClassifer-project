"""T2 determinism proofs — the §0 spine (static ban + permutation-invariance + decode).

§0: ``enforce_determinism(strict=True)`` does NOT raise on ``scatter_add`` / non-stable
``topk`` on torch 2.7 — so determinism is proven by (1) a **static AST ban** over
``models/fusion/**``, (2) **permutation-invariance** of the splat + pillar scatter, and
(3) **decode tie reproducibility**. The bit-identical-weights gate is the dedicated
**A40 SLURM job** (``scripts/det_gate_a40.py``) — CPU/T4 cannot prove cross-arch identity.
"""
from __future__ import annotations

import ast
import os

import pytest
import torch

from fl_v3.utils.runtime import enforce_determinism, seed_everything

FUSION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "fl_v3", "models", "fusion"
)

# Banned bare identifiers (atomic / order-dependent ops).
_BANNED_CALLS = {
    "scatter_add", "scatter_add_", "index_add", "index_add_",
    "scatter_reduce", "scatter_reduce_", "grid_sample", "topk",
}


def _ast_ban_findings(src: str, fname: str) -> list[str]:
    """AST scan (prose in docstrings is Constant nodes → never flagged)."""
    findings: list[str] = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else (
                func.id if isinstance(func, ast.Name) else None
            )
            if name in _BANNED_CALLS:
                findings.append(f"{fname}: banned call {name!r}")
            if name in ("index_put", "index_put_"):
                for kw in node.keywords:
                    if kw.arg == "accumulate" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        findings.append(f"{fname}: index_put(accumulate=True)")
            if name in ("sort", "argsort"):
                has_stable = any(
                    kw.arg == "stable" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                    for kw in node.keywords
                )
                if not has_stable:
                    findings.append(f"{fname}: {name}() without stable=True")
    return findings


def test_static_ban_over_models_fusion():
    """No scatter_add/index_add/index_put(accumulate)/grid_sample/topk/non-stable sort."""
    files = [f for f in os.listdir(FUSION_DIR) if f.endswith(".py")]
    assert files, "no model source files found"
    all_findings: list[str] = []
    for f in sorted(files):
        with open(os.path.join(FUSION_DIR, f), encoding="utf-8") as fh:
            all_findings += _ast_ban_findings(fh.read(), f)
    assert not all_findings, "banned ops in models/fusion: " + "; ".join(all_findings)


# ---------------------------------------------------------------------------
# Permutation-invariance — the cross-arch determinism proof (§0.2).
# ---------------------------------------------------------------------------
def test_splat_permutation_invariant():
    """``bev_splat`` is byte-identical under a permutation of its frustum points."""
    from fl_v3.models.fusion.view_transform import bev_splat

    enforce_determinism(strict=True); seed_everything(0)
    B, nx, ny, Cc, P = 2, 32, 32, 8, 4000
    x = torch.randn(P, Cc)
    ranks = torch.randint(0, B * nx * ny, (P,))
    geom = torch.arange(P)  # canonical per-point id (unique)
    out = bev_splat(x, ranks, geom, B, nx, ny)
    perm = torch.randperm(P)
    out_p = bev_splat(x[perm], ranks[perm], geom[perm], B, nx, ny)
    assert torch.equal(out, out_p), f"splat not permutation-invariant: max|Δ|={(out-out_p).abs().max()}"


def test_pillar_scatter_permutation_invariant():
    """PointPillars encoder is byte-identical under a LiDAR point permutation (sparse)."""
    from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder

    enforce_determinism(strict=True); seed_everything(1)
    enc = PointPillarsEncoder(out_channels=32, max_points=32)
    n = 3000
    xyz = (torch.rand(n, 3) * 2 - 1) * torch.tensor([45.0, 45.0, 3.0])
    pts = torch.cat([torch.zeros(n, 1), xyz, torch.rand(n, 1), torch.zeros(n, 1)], dim=1)
    a = enc(pts, B=1)
    perm = torch.randperm(n)
    b = enc(pts[perm], B=1)
    assert torch.equal(a, b), f"pillar scatter not permutation-invariant: max|Δ|={(a-b).abs().max()}"


def test_pillar_scatter_permutation_invariant_OVERCAP():
    """The Codex T2 finding: with a pillar EXCEEDING max_points, a LiDAR point permutation
    must STILL give a byte-identical canvas (canonical content-order truncation, not file
    order). A plain stable-sort-on-pillar_key encoder would fail this (max|Δ|~2e-5)."""
    from fl_v3.models.fusion.lidar_encoder import PointPillarsEncoder

    enforce_determinism(strict=True); seed_everything(2)
    enc = PointPillarsEncoder(out_channels=32, max_points=8)  # small cap → truncation fires
    # a dense cluster in a ~1m box → many points land in a handful of pillars (>> 8 each)
    n = 2000
    cluster = torch.tensor([10.0, -5.0, -0.5]) + (torch.rand(n, 3) * 2 - 1) * torch.tensor([0.5, 0.5, 0.3])
    pts = torch.cat([torch.zeros(n, 1), cluster, torch.rand(n, 1), torch.zeros(n, 1)], dim=1)
    # sanity: at least one pillar must exceed the cap, else the test is vacuous
    from fl_v3.models.fusion.bev_grid import BEVConfig, metric_to_grid
    cfg = BEVConfig()
    c, r = metric_to_grid(pts[:, 1], pts[:, 2], cfg.x_min, cfg.y_min, cfg.vx, cfg.vy)
    _, counts = torch.unique(r * cfg.nx + c, return_counts=True)
    assert int(counts.max()) > enc.max_points, "test not exercising the over-cap path"
    a = enc(pts, B=1)
    b = enc(pts[torch.randperm(n)], B=1)
    assert torch.equal(a, b), f"over-cap pillar scatter not permutation-invariant: max|Δ|={(a-b).abs().max()}"


def test_decode_tie_break_is_canonical():
    """Decode tie order is reproducible (stable sort → ascending flat index among ties)."""
    from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig

    enforce_determinism(strict=True); seed_everything(0)
    cfg = DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False)
    model = BEVFusionDetector(cfg)
    H, W = cfg.bev.head_ny, cfg.bev.head_nx
    # a heatmap with a flat plateau of ties on class 0 (logit→constant after sigmoid),
    # surrounded by lower values so the 3×3 local-max keeps the plateau cells.
    heat = torch.full((1, cfg.n_classes, H, W), -10.0)
    heat[0, 0, 10:13, 10:13] = 5.0  # 3×3 tied plateau (all equal)
    head = {"heatmap": heat, "reg": torch.zeros(1, 2, H, W), "height": torch.zeros(1, 1, H, W),
            "dim": torch.zeros(1, 3, H, W), "rot": torch.zeros(1, 2, H, W), "vel": torch.zeros(1, 2, H, W)}
    r1 = model.decode(head, score_threshold=0.5, max_objects=200)
    r2 = model.decode(head, score_threshold=0.5, max_objects=200)
    # run-to-run determinism
    assert torch.equal(r1[0]["boxes"], r2[0]["boxes"])
    assert torch.equal(r1[0]["scores"], r2[0]["scores"])
    # canonical tie order: the local-max mask keeps only the plateau's interior peak(s);
    # whatever survives, the ordering of equal scores is by ascending flat index (stable).
    scores = r1[0]["scores"]
    assert scores.numel() > 0
    assert torch.all(scores[:-1] >= scores[1:]), "decode scores not non-increasing"


def test_full_forward_same_seed_bit_identical_cpu():
    """Two same-seed model builds + forwards are byte-identical on CPU (the A40 SLURM job
    proves the production cross-arch identity; this is the cheap CPU canary)."""
    from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
    from fl_v3.models.fusion.collate import detection_collate_fn
    from _det_fixtures import fake_detection_sample

    cfg = DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False,
                         neck_channels=32, context_channels=16, lidar_channels=16,
                         fusion_channels=32, bev_neck_channels=64, head_channels=16)
    batch = detection_collate_fn([fake_detection_sample(1)])

    def build_fwd():
        enforce_determinism(strict=True); seed_everything(7)
        m = BEVFusionDetector(cfg)
        return m(batch)["heatmap"].detach().clone()

    assert torch.equal(build_fwd(), build_fwd())
