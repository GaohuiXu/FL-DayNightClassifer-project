"""V1 calibration renderers (fl_v3 T1) — a hard pre-trust gate.

Renders into the existing ``VizWriter`` ``"calibration"`` stage (= V1). No later
stage (model, attack, defense) may be trusted until calibrated projection renders
correctly on real samples. Renderers:

  (i)   cam image + **projected LiDAR** (depth-colored)
  (ii)  cam image + **projected 3D GT boxes**
  (iii) **BEV point cloud + GT boxes** (top-down)
  (iv)  **partition plots** (per-client class histogram + location)

**Independent-projector check (anti "viz-shares-the-bug").** A wrong-but-self-
consistent transform passes an eyeball, so each projected render overlays **our**
projection (built from raw ``calibrated_sensor``/``ego_pose`` records via our
``transforms``) against the **devkit** projector (``get_sample_data`` +
``view_points``) in a distinct color, and :func:`independent_projector_agreement`
returns the max pixel disagreement that a numeric test asserts is **≤1 px**. Our
side never calls ``get_sample_data`` to *build* the transform — only as the
independent reference to compare against.

Figure filenames derive from ``sample_token`` (not an enumeration counter), so the
artifact set is run-stable. Matplotlib uses the headless ``Agg`` backend.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from fl_v3.data.nuscenes import paths as P
from fl_v3.data.nuscenes import transforms as T
from fl_v3.data.nuscenes.class_map import DETECTION_NAMES

_FIG_DPI = 100


# ---------------------------------------------------------------------------
# Geometry helpers (schema box7 → corners; devkit reference projection).
# ---------------------------------------------------------------------------
def box7_to_corners(box7: np.ndarray) -> np.ndarray:
    """``(cx,cy,cz,dx,dy,dz,yaw)`` (LIDAR_TOP) → ``(8,3)`` corners.

    Matches the devkit corner ordering (x fwd, y left, z up; first four face
    forward). dx=l on x, dy=w on y, dz=h on z; rotation about +z by yaw (yaw-only,
    the schema's BEV-oriented parameterization — pitch/roll are intentionally
    dropped, matching nuScenes detection).
    """
    cx, cy, cz, dx, dy, dz, yaw = [float(v) for v in box7]
    x = dx / 2.0 * np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    y = dy / 2.0 * np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64)
    z = dz / 2.0 * np.array([1, 1, -1, -1, 1, 1, -1, -1], dtype=np.float64)
    corners = np.vstack([x, y, z])  # (3,8)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    corners = R @ corners
    corners += np.array([[cx], [cy], [cz]])
    return corners.T  # (8,3)


def _devkit_records(nusc, sample_token: str, cam_channel: str):
    sample = nusc.get("sample", sample_token)
    sd_l = nusc.get("sample_data", sample["data"][P.LIDAR_CHANNEL])
    cs_l = nusc.get("calibrated_sensor", sd_l["calibrated_sensor_token"])
    ep_l = nusc.get("ego_pose", sd_l["ego_pose_token"])
    sd_c = nusc.get("sample_data", sample["data"][cam_channel])
    cs_c = nusc.get("calibrated_sensor", sd_c["calibrated_sensor_token"])
    ep_c = nusc.get("ego_pose", sd_c["ego_pose_token"])
    return sample, (cs_l, ep_l), (cs_c, ep_c, sd_c)


def independent_projector_agreement(
    nusc, sample_token: str, cam_channel: str
) -> Dict[str, float]:
    """Max pixel disagreement between OUR projector and the devkit's, for one cam.

    Returns ``{"lidar_px": ..., "box_corner_px": ..., "n_lidar": ..., "n_box": ...}``.
    Our projector is built from the raw records via ``transforms`` (never
    ``get_sample_data``); the devkit reference uses ``view_points`` + the stepwise
    pose chain (for LiDAR) and ``get_sample_data`` cam-frame boxes (for corners).
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import view_points
    from pyquaternion import Quaternion

    sample, (cs_l, ep_l), (cs_c, ep_c, sd_c) = _devkit_records(nusc, sample_token, cam_channel)
    K = np.asarray(cs_c["camera_intrinsic"], dtype=np.float64)
    W, H = sd_c["width"], sd_c["height"]

    # OUR projector (raw records → transforms).
    lidar2cam = T.compose_lidar2cam(cs_l, ep_l, ep_c, cs_c)
    lidar2img = T.build_lidar2img(K, lidar2cam)

    # --- LiDAR points ---
    lidar_path = nusc.get_sample_data_path(sample["data"][P.LIDAR_CHANNEL])
    pts = LidarPointCloud.from_file(lidar_path).points  # (4,N) — only used as common 3D input
    our_uv, our_d = T.project_to_image(pts[:3].T, lidar2img)

    # DEVKIT reference path (stepwise float32).
    pc = LidarPointCloud.from_file(lidar_path)
    pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix); pc.translate(np.array(cs_l["translation"]))
    pc.rotate(Quaternion(ep_l["rotation"]).rotation_matrix); pc.translate(np.array(ep_l["translation"]))
    pc.translate(-np.array(ep_c["translation"])); pc.rotate(Quaternion(ep_c["rotation"]).rotation_matrix.T)
    pc.translate(-np.array(cs_c["translation"])); pc.rotate(Quaternion(cs_c["rotation"]).rotation_matrix.T)
    dk_uv = view_points(pc.points[:3, :], K, normalize=True)[:2].T  # (N,2)
    dk_d = pc.points[2, :]

    m = (dk_d > 1.0) & (dk_uv[:, 0] > 1) & (dk_uv[:, 0] < W - 1) & (dk_uv[:, 1] > 1) & (dk_uv[:, 1] < H - 1)
    lidar_px = float(np.abs(our_uv[m] - dk_uv[m]).max()) if m.any() else 0.0

    # --- Box corners: our lidar2img on devkit lidar-frame corners vs devkit cam-frame view_points ---
    _, boxes_lidar, _ = nusc.get_sample_data(sample["data"][P.LIDAR_CHANNEL])
    _, boxes_cam, K_cam = nusc.get_sample_data(sample["data"][cam_channel])
    lidar_by_tok = {b.token: b for b in boxes_lidar}
    box_px = 0.0
    n_box = 0
    for bc in boxes_cam:
        bl = lidar_by_tok.get(bc.token)
        if bl is None:
            continue
        our = T.project_to_image(bl.corners().T, lidar2img)[0]  # (8,2)
        dk = view_points(bc.corners(), np.asarray(K_cam), normalize=True)[:2].T
        box_px = max(box_px, float(np.abs(our - dk).max()))
        n_box += 1
    return {
        "lidar_px": lidar_px,
        "box_corner_px": box_px,
        "n_lidar": int(m.sum()),
        "n_box": n_box,
    }


# ---------------------------------------------------------------------------
# Renderers.
# ---------------------------------------------------------------------------
def _front_mask(uv, depth, W, H):
    return (depth > 1.0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)


def render_cam_lidar(writer, sample: dict, nusc, cam_index: int = 0,
                     overlay_devkit: bool = True) -> str:
    """(i) cam image + projected LiDAR (depth-colored); optional devkit overlay."""
    cam = sample["cam_order"][cam_index]
    img = sample["images"][cam_index].numpy().transpose(1, 2, 0)  # (H,W,3)
    H, W = img.shape[:2]
    lidar2img = sample["lidar2img"][cam_index].numpy()
    pts = sample["lidar_points"][:, :3].numpy()
    uv, depth = T.project_to_image(pts, lidar2img)
    m = _front_mask(uv, depth, W, H)

    fig, ax = plt.subplots(figsize=(W / _FIG_DPI, H / _FIG_DPI), dpi=_FIG_DPI)
    ax.imshow(img)
    sc = ax.scatter(uv[m, 0], uv[m, 1], c=depth[m], s=1.5, cmap="viridis", alpha=0.7)
    if overlay_devkit and nusc is not None:
        agree = independent_projector_agreement(nusc, sample["sample_token"], cam)
        ax.set_title(f"{cam} | ours=viridis  devkit=red×  max|Δ|={agree['lidar_px']:.3f}px", fontsize=8)
        # devkit overlay (independent reference) as sparse red crosses
        from nuscenes.utils.data_classes import LidarPointCloud
        from nuscenes.utils.geometry_utils import view_points
        from pyquaternion import Quaternion
        sample_rec, (cs_l, ep_l), (cs_c, ep_c, sd_c) = _devkit_records(nusc, sample["sample_token"], cam)
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(sample_rec["data"][P.LIDAR_CHANNEL]))
        pc.rotate(Quaternion(cs_l["rotation"]).rotation_matrix); pc.translate(np.array(cs_l["translation"]))
        pc.rotate(Quaternion(ep_l["rotation"]).rotation_matrix); pc.translate(np.array(ep_l["translation"]))
        pc.translate(-np.array(ep_c["translation"])); pc.rotate(Quaternion(ep_c["rotation"]).rotation_matrix.T)
        pc.translate(-np.array(cs_c["translation"])); pc.rotate(Quaternion(cs_c["rotation"]).rotation_matrix.T)
        K = np.asarray(cs_c["camera_intrinsic"])
        dkuv = view_points(pc.points[:3], K, normalize=True)[:2].T
        dkd = pc.points[2]
        dm = _front_mask(dkuv, dkd, W, H)
        # subsample for legibility (stable stride, no RNG)
        ax.scatter(dkuv[dm][::13, 0], dkuv[dm][::13, 1], marker="x", s=10, c="red", linewidths=0.5)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    path = writer.figure_path("calibration", f"cam_lidar__{cam}__{sample['sample_token']}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


def render_cam_boxes(writer, sample: dict, nusc, cam_index: int = 0,
                     overlay_devkit: bool = True) -> str:
    """(ii) cam image + projected 3D GT boxes; optional devkit-box overlay."""
    cam = sample["cam_order"][cam_index]
    img = sample["images"][cam_index].numpy().transpose(1, 2, 0)
    H, W = img.shape[:2]
    lidar2img = sample["lidar2img"][cam_index].numpy()
    fig, ax = plt.subplots(figsize=(W / _FIG_DPI, H / _FIG_DPI), dpi=_FIG_DPI)
    ax.imshow(img)
    boxes = sample["gt_boxes"].numpy()
    for i in range(boxes.shape[0]):
        corners = box7_to_corners(boxes[i])  # (8,3)
        uv, depth = T.project_to_image(corners, lidar2img)
        if np.all(depth <= 1.0):
            continue  # behind camera
        _draw_box_2d(ax, uv, depth, color="lime")
    title = f"{cam} | ours=lime"
    if overlay_devkit and nusc is not None:
        from nuscenes.utils.geometry_utils import view_points
        _, _, _ = None, None, None
        sample_rec = nusc.get("sample", sample["sample_token"])
        _, boxes_cam, K_cam = nusc.get_sample_data(sample_rec["data"][cam])
        for bc in boxes_cam:
            uv = view_points(bc.corners(), np.asarray(K_cam), normalize=True)[:2].T
            _draw_box_2d(ax, uv, np.ones(8) * 2.0, color="red", lw=0.6, ls="--")
        agree = independent_projector_agreement(nusc, sample["sample_token"], cam)
        title += f"  devkit=red-- max|Δ|={agree['box_corner_px']:.2e}px"
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    path = writer.figure_path("calibration", f"cam_boxes__{cam}__{sample['sample_token']}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return path


_BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),  # front/back faces
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _draw_box_2d(ax, uv, depth, color="lime", lw=1.0, ls="-"):
    for a, b in _BOX_EDGES:
        if depth[a] > 1.0 and depth[b] > 1.0:
            ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], c=color, lw=lw, ls=ls)


def render_bev(writer, sample: dict, max_range: float = 55.0) -> str:
    """(iii) BEV point cloud (top-down) + GT box rectangles in LIDAR_TOP frame."""
    pts = sample["lidar_points"][:, :2].numpy()
    fig, ax = plt.subplots(figsize=(7, 7), dpi=_FIG_DPI)
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="0.5", alpha=0.5)
    boxes = sample["gt_boxes"].numpy()
    names = sample["gt_names"]
    in_range = sample["gt_in_range"].numpy()
    for i in range(boxes.shape[0]):
        corners = box7_to_corners(boxes[i])[:4, :2]  # top 4 corners' xy (BEV footprint)
        col = "red" if names[i] == "car" else ("orange" if in_range[i] else "blue")
        ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=col, linewidth=1.0))
    ax.set_xlim(-max_range, max_range); ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal"); ax.set_xlabel("x fwd (m)"); ax.set_ylabel("y left (m)")
    ax.set_title(f"BEV LIDAR_TOP | {sample['sample_token']} | car=red in-range=orange oor=blue", fontsize=8)
    ax.plot(0, 0, "k+", markersize=10)  # ego/lidar origin
    path = writer.figure_path("calibration", f"bev__{sample['sample_token']}")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return path


def render_partition(writer, report: dict, name: str = "partition") -> Tuple[str, str]:
    """(iv) partition plots: per-client class histogram (stacked) + location bars."""
    clients = report["clients"]
    cids = [c["client_id"] for c in clients]
    # class histogram heat: rows=clients, cols=detection classes
    mat = np.zeros((len(clients), len(DETECTION_NAMES)), dtype=np.int64)
    for r, c in enumerate(clients):
        for k, v in c["class_hist"].items():
            mat[r, int(k)] = v
    fig, ax = plt.subplots(figsize=(10, max(3, 0.28 * len(clients))), dpi=_FIG_DPI)
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(DETECTION_NAMES)))
    ax.set_xticklabels(DETECTION_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(clients)))
    ax.set_yticklabels([f"c{c['client_id']}·{c['location'][:3]}·{c['n_keyframes']}kf" for c in clients], fontsize=6)
    ax.set_title(f"per-client class histogram | scale={report['scale']} | N={report['num_clients']} | {report['mode']}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025)
    hist_path = writer.figure_path("calibration", f"{name}_class_hist")
    fig.savefig(hist_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # location bar: keyframes per client, colored by location
    loc_color = {loc: c for loc, c in zip(sorted(set(c["location"] for c in clients)),
                                          ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])}
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=_FIG_DPI)
    ax.bar(cids, [c["n_keyframes"] for c in clients],
           color=[loc_color[c["location"]] for c in clients])
    handles = [plt.Rectangle((0, 0), 1, 1, color=loc_color[l]) for l in loc_color]
    ax.legend(handles, list(loc_color), fontsize=7, title="location")
    ax.set_xlabel("client_id"); ax.set_ylabel("#keyframes")
    ax.set_title(f"keyframes/client by location | floor={report['floor']} | N@band={report['n_at_floor_band']}", fontsize=8)
    loc_path = writer.figure_path("calibration", f"{name}_location")
    fig.savefig(loc_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return hist_path, loc_path
