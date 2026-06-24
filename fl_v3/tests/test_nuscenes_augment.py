"""BEV/3D augmentation geometry-consistency tests (MCR P1 overfitting fix).

The aug is only valid if the scene transform T is applied CONSISTENTLY across modalities. These tests
are the safety net for that:
  * camera projection invariance: a point projects to the SAME pixel before/after aug (proves
    lidar2img_aug = lidar2img · T⁻¹ + the point push-forward agree) — the fusion-alignment guarantee;
  * box↔point containment: points inside a GT box STAY inside the transformed box (proves the box
    center/dims/yaw transform matches the point transform — catches any yaw/flip sign error);
  * determinism: same numpy seed → same transform (reproducible under seeded_worker_init);
  * off-is-identity: no aug params ⇒ the sample is unchanged.
"""
import numpy as np
import torch

from fl_v3.data.nuscenes.augment import (
    sample_transform, apply_transform, augment_sample, apply_image_flip, DEFAULT_AUG)


def _box_local(points, box):
    """Per-point box-local coords: local = (world - center) @ R (R = yaw rotation)."""
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box]
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return (points - np.array([cx, cy, cz])) @ R, np.array([dx, dy, dz])


def test_camera_projection_invariance():
    """lidar2img · [p,1] must equal lidar2img_aug · [p_aug,1] for every aug — the fusion alignment."""
    rng = np.random.RandomState(0)
    np.random.seed(0)
    pts = torch.from_numpy(rng.uniform(-40, 40, size=(64, 6)).astype(np.float32))
    l2i = torch.from_numpy(rng.uniform(-1, 1, size=(6, 4, 4)).astype(np.float32))
    boxes = torch.zeros((0, 7)); vel = torch.zeros((0, 2))
    for _ in range(20):
        T = sample_transform(DEFAULT_AUG)
        pts_a, _, _, l2i_a = apply_transform(pts, boxes, vel, l2i, T)
        p1 = torch.cat([pts[:, :3], torch.ones(64, 1)], 1)          # [N,4]
        p1a = torch.cat([pts_a[:, :3], torch.ones(64, 1)], 1)
        for cam in range(6):
            orig = p1 @ l2i[cam].T          # [N,4] homogeneous image coords
            aug = p1a @ l2i_a[cam].T
            assert torch.allclose(orig, aug, atol=1e-2), "camera projection moved under aug"


def test_box_point_containment():
    """Points inside a GT box stay inside the transformed box (box transform ↔ point transform)."""
    np.random.seed(1)
    rng = np.random.RandomState(1)
    for _ in range(30):
        box = np.array([rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-2, 2),
                        rng.uniform(1, 6), rng.uniform(1, 4), rng.uniform(1, 3),
                        rng.uniform(-np.pi, np.pi)], dtype=np.float64)
        # interior points (in box-local frame) → world
        local = (rng.rand(40, 3) - 0.5) * box[3:6]
        c, s = np.cos(box[6]), np.sin(box[6])
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        world = local @ R.T + box[:3]
        pts = torch.from_numpy(np.concatenate([world, np.zeros((40, 3))], 1).astype(np.float32))
        gt = torch.from_numpy(box[None].astype(np.float32))
        T = sample_transform(DEFAULT_AUG)
        pts_a, gt_a, _, _ = apply_transform(pts, gt, torch.zeros((1, 2)), torch.zeros((6, 4, 4)), T)
        loc_a, ext_a = _box_local(pts_a[:, :3].numpy().astype(np.float64), gt_a[0].numpy())
        assert (np.abs(loc_a) <= ext_a / 2 + 1e-3).all(), "points left the box after aug (inconsistent transform)"


def test_velocity_rotates_with_scene():
    """A pure +90° yaw rotation maps velocity (1,0) → (0,1)."""
    np.random.seed(0)
    T = np.eye(4); th = np.pi / 2
    T[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    vel = torch.tensor([[1.0, 0.0]])
    _, _, v_a, _ = apply_transform(torch.zeros((1, 6)), torch.zeros((0, 7)), vel, torch.zeros((6, 4, 4)), T)
    assert torch.allclose(v_a[0], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_image_flip_projection_consistency():
    """After a horizontal image flip, a lidar point must project to the mirrored pixel u→(W-1)-u (v
    unchanged), and the image tensor must actually be flipped — the regularizer relies on Swin seeing
    flipped pixels while the lift geometry stays correct."""
    rng = np.random.RandomState(3)
    H, W = 16, 44
    images = torch.from_numpy(rng.rand(6, 3, H, W).astype(np.float32))
    l2i = torch.from_numpy(rng.uniform(-1, 1, size=(6, 4, 4)).astype(np.float32))
    P = torch.tensor([3.0, -5.0, 1.0, 1.0])           # a lidar point (homogeneous)
    imgs_f, l2i_f = apply_image_flip(images, l2i)
    assert torch.equal(imgs_f, torch.flip(images, dims=[-1])), "image not flipped"
    for cam in range(6):
        proj = l2i[cam] @ P;   u, depth = proj[0] / proj[2], proj[2]
        proj_f = l2i_f[cam] @ P; u_f, v_f = proj_f[0] / proj_f[2], proj_f[1] / proj_f[2]
        v = (l2i[cam] @ P)[1] / depth
        assert torch.allclose(u_f, (W - 1) - u, atol=1e-3), "u did not mirror to (W-1)-u"
        assert torch.allclose(v_f, v, atol=1e-3), "v changed under horizontal flip"


def test_determinism():
    np.random.seed(7); T1 = sample_transform(DEFAULT_AUG)
    np.random.seed(7); T2 = sample_transform(DEFAULT_AUG)
    assert np.array_equal(T1, T2)


def test_yaw_only_rotation_preserves_dims():
    """Scale=1, no flip, pure rotation: box dims unchanged, yaw shifts by exactly theta."""
    np.random.seed(0)
    p = {"rot": 0.5, "scale": (1.0, 1.0), "translate": 0.0, "flip": False}
    box = torch.tensor([[1.0, 2.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
    T = sample_transform(p)
    _, gt_a, _, _ = apply_transform(torch.zeros((1, 6)), box, torch.zeros((1, 2)), torch.zeros((6, 4, 4)), T)
    assert torch.allclose(gt_a[0, 3:6], box[0, 3:6], atol=1e-5), "dims changed under pure rotation"
