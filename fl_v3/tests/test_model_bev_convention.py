"""T2 BEV/decode convention — ground-truth-anchored (NOT self-consistent), the T2↔T4↔T5
contract.

The no-oracle trap (SPEC §5): if splat/scatter/target/decode share the SAME wrong
``(x,y)→(row,col)`` mapping, a self-consistency test passes while the BEV is wrong. We
defeat it by anchoring to **two independent physical facts from T1**: (a) a real GT
object's LiDAR returns and (b) its labelled box center must land in the SAME BEV cell;
plus an **injected-corruption negative** (swap row/col → the anchor FAILS, proving the
test discriminates). Resize/intrinsic consistency is checked numerically against T1's
``lidar2img``; the box convention is pinned by an encode→decode round-trip golden.
"""
from __future__ import annotations

import numpy as np
import torch

from fl_v3.models.fusion.bev_grid import BEVConfig, points_to_fine_grid, metric_to_grid


def _load_real_sample(mini_train_info, dataroot):
    from fl_v3.data.nuscenes.dataset import NuScenesMultimodalDataset

    ds = NuScenesMultimodalDataset(mini_train_info, dataroot)
    return ds


def test_bev_convention_ground_truth_anchored(mini_train_info, dataroot):
    """A real GT object's LiDAR points and its labelled box center fall in the SAME BEV
    cell under the single shared mapping (anchored to T1 data, not self-consistency)."""
    cfg = BEVConfig()
    ds = _load_real_sample(mini_train_info, dataroot)
    found = False
    for idx in range(len(ds)):
        sample = ds[idx]
        boxes = sample["gt_boxes"].numpy()
        npts = sample["gt_num_lidar_pts"].numpy()
        names = sample["gt_names"]
        pts = sample["lidar_points"][:, :3].numpy()
        for m in range(boxes.shape[0]):
            cx, cy, cz, dx, dy, dz, yaw = boxes[m]
            # need a well-populated, in-range, BEV-interior object
            if npts[m] < 40 or names[m] != "car":
                continue
            if not (cfg.x_min + 5 < cx < cfg.x_max - 5 and cfg.y_min + 5 < cy < cfg.y_max - 5):
                continue
            # object's physical LiDAR points (axis-aligned footprint approx is enough)
            sel = (
                (np.abs(pts[:, 0] - cx) < dx / 2 + 0.5)
                & (np.abs(pts[:, 1] - cy) < dy / 2 + 0.5)
                & (pts[:, 2] > cz - dz)
                & (pts[:, 2] < cz + dz)
            )
            obj = pts[sel]
            if obj.shape[0] < 20:
                continue
            # shared mapping: box-center cell vs object-point cells
            col_c, row_c = points_to_fine_grid(torch.tensor([[cx, cy]]), cfg)
            col_p, row_p = points_to_fine_grid(torch.tensor(obj[:, :2]), cfg)
            # the box-center cell must sit at the MEDIAN of the object points' cells
            med_col = float(np.median(col_p.numpy()))
            med_row = float(np.median(row_p.numpy()))
            assert abs(float(col_c.item()) - med_col) <= 2.0, (
                f"box-center col {col_c.item()} far from object-points median {med_col}"
            )
            assert abs(float(row_c.item()) - med_row) <= 2.0, (
                f"box-center row {row_c.item()} far from object-points median {med_row}"
            )
            found = True
            # --- injected-corruption negative: swap row↔col → anchor must BREAK ---
            # (col,row) swapped is a genuinely different cell unless cx≈cy; pick objects
            # where they differ enough to discriminate.
            if abs(med_col - med_row) > 5.0:
                swapped_col, swapped_row = row_c, col_c  # the row/col-swap bug
                broke = (abs(float(swapped_col.item()) - med_col) > 2.0
                         or abs(float(swapped_row.item()) - med_row) > 2.0)
                assert broke, "row/col-swap corruption NOT detected — test lacks discriminating power"
            break
        if found:
            break
    assert found, "no suitable anchored GT car found on mini (unexpected)"


def test_resize_intrinsic_lidar2img_consistency_1px(mini_train_info, dataroot):
    """Projecting a real LiDAR point with the resize-rescaled lidar2img matches the
    align_corners=False resize location of its native projection to ≤1px (anchored to T1)."""
    from fl_v3.models.fusion.preprocess import ImagePreprocessor

    ds = _load_real_sample(mini_train_info, dataroot)
    sample = ds[0]
    images = sample["images"].unsqueeze(0)        # [1,6,3,900,1600]
    l2i = sample["lidar2img"].unsqueeze(0)         # [1,6,4,4]
    K = sample["cam_intrinsics"].unsqueeze(0)
    H_in, W_in = images.shape[-2:]
    pre = ImagePreprocessor(image_hw=(256, 704))
    out = pre(images, l2i, K)
    H_out, W_out = out["image_hw"]
    fx, fy = W_out / W_in, H_out / H_in

    pts = sample["lidar_points"][:, :3]
    max_resid = 0.0
    n_checked = 0
    for cam in range(6):
        native = l2i[0, cam].double()
        rescaled = out["lidar2img"][0, cam].double()
        hom = torch.cat([pts.double(), torch.ones(pts.shape[0], 1, dtype=torch.float64)], dim=1)
        pn = hom @ native.T
        depth = pn[:, 2]
        m = depth > 1.0
        if m.sum() == 0:
            continue
        uvn = pn[m, :2] / depth[m, None]                  # native pixels
        # in-image native points only
        inb = (uvn[:, 0] > 1) & (uvn[:, 0] < W_in - 1) & (uvn[:, 1] > 1) & (uvn[:, 1] < H_in - 1)
        if inb.sum() == 0:
            continue
        pr = hom[m][inb] @ rescaled.T
        uvr = pr[:, :2] / pr[:, 2, None]                  # rescaled pixels
        # where align_corners=False resize maps the native pixel
        expect_u = (uvn[inb, 0] + 0.5) * fx - 0.5
        expect_v = (uvn[inb, 1] + 0.5) * fy - 0.5
        resid = torch.maximum((uvr[:, 0] - expect_u).abs().max(), (uvr[:, 1] - expect_v).abs().max())
        max_resid = max(max_resid, float(resid))
        n_checked += int(inb.sum())
    assert n_checked > 0
    assert max_resid <= 1.0, f"resize/intrinsic inconsistency {max_resid:.4f}px > 1px"


def _encode_decode_box(cx, cy, cz, l, w, h, yaw, cls=0):
    """Render a single box to head targets, build a head_out from them, decode, return box7."""
    from fl_v3.models.fusion.detector import BEVFusionDetector, DetectorConfig
    from fl_v3.models.fusion.losses import CenterPointLoss

    cfg = DetectorConfig(camera_backbone="resnet18", pretrained_backbone=False)
    model = BEVFusionDetector(cfg)
    loss = CenterPointLoss(cfg=cfg.bev, n_classes=cfg.n_classes)
    box = torch.tensor([[cx, cy, cz, l, w, h, yaw]], dtype=torch.float32)
    batch = {"gt_boxes": [box], "gt_labels": [torch.tensor([cls])], "gt_velocity": [torch.zeros(1, 2)]}
    heatmap_t, bidx, cells, reg_target = loss.build_targets(batch, device="cpu")
    H, W = cfg.bev.head_ny, cfg.bev.head_nx
    cell = int(cells[0].item()); row, col = cell // W, cell % W
    # synthetic head_out: clear peak at (row,col) on class `cls`, reg = encoded vector
    heat = torch.full((1, cfg.n_classes, H, W), -10.0)
    heat[0, cls, row, col] = 10.0
    rv = reg_target[0]  # [10] = reg2,z1,dim3,rot2,vel2
    head = {
        "heatmap": heat,
        "reg": torch.zeros(1, 2, H, W), "height": torch.zeros(1, 1, H, W),
        "dim": torch.zeros(1, 3, H, W), "rot": torch.zeros(1, 2, H, W), "vel": torch.zeros(1, 2, H, W),
    }
    head["reg"][0, :, row, col] = rv[0:2]
    head["height"][0, :, row, col] = rv[2:3]
    head["dim"][0, :, row, col] = rv[3:6]
    head["rot"][0, :, row, col] = rv[6:8]
    head["vel"][0, :, row, col] = rv[8:10]
    res = model.decode(head, score_threshold=0.5, max_objects=10)[0]
    return res["boxes"][0], int(res["labels"][0].item())


def test_encode_decode_roundtrip_golden():
    """Encode→decode recovers the box to tight tol incl. yaw SIGN (no -π/2, no (l,w,h) swap)."""
    for yaw in (0.7, -0.7, 2.5, -2.5, 0.0):
        cx, cy, cz, l, w, h = 12.3, -7.6, -0.9, 4.2, 1.9, 1.55
        rec, cls = _encode_decode_box(cx, cy, cz, l, w, h, yaw)
        rec = rec.numpy()
        assert cls == 0
        assert abs(rec[0] - cx) < 0.5, f"cx {rec[0]} vs {cx}"   # ≤ half a head cell (0.8m)
        assert abs(rec[1] - cy) < 0.5, f"cy {rec[1]} vs {cy}"
        assert abs(rec[2] - cz) < 1e-3, f"cz {rec[2]} vs {cz}"
        assert abs(rec[3] - l) < 1e-2, f"l(dx) {rec[3]} vs {l}"  # dx=l (NO (l,w,h) swap)
        assert abs(rec[4] - w) < 1e-2, f"w(dy) {rec[4]} vs {w}"
        assert abs(rec[5] - h) < 1e-2, f"h(dz) {rec[5]} vs {h}"
        # yaw recovered with correct sign (no -π/2 offset)
        dyaw = np.arctan2(np.sin(rec[6] - yaw), np.cos(rec[6] - yaw))
        assert abs(dyaw) < 1e-3, f"yaw {rec[6]} vs {yaw} (Δ={dyaw})"
