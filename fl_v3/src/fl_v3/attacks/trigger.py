"""The camera trigger — ONE deterministic generator (BadFusion-style, D2) (fl_v3 T5).

A digital patch blitted onto a native-resolution (900×1600) camera frame, used by **both** the
train-time poison wrapper AND the inference-time ablation decode (a golden sha256 test asserts the
patched image is byte-identical at both call sites). **No global RNG** — the pattern, colour, and
opacity are config-pinned and the placement is a pure function of the target box + the per-camera
``lidar2img`` projection.

## Placement (the load-bearing knob — §Attack-spec "Fusion-awareness")

The primary threat model is **camera-only**: cam-vs-LiDAR *spatial alignment*, not a LiDAR
perturbation, is what makes fusion-dependence testable.

  * **target-aligned** (cond-4) = at the **densest cluster of the target's projected LiDAR points**
    (BadFusion's actual criterion). The box's enclosed LiDAR points are projected via the reused
    ``transforms.project_to_image`` / ``build_lidar2img`` geometry; the camera is chosen by the
    smallest positive box-centre depth (tie → lowest ``CAM_ORDER`` index); the patch lands at the 2-D
    densest-bin centroid (fallback: the box-centre projection when ``< K`` enclosed points). The
    centre is clamped to ``≤ trigger_center_max_px`` of the box-centre projection so the **objective
    placement test** (cond-4 centre ≤ 20 px from the box centre) holds.
  * **non-aligned** (cond-2) = a deterministic LiDAR-sparse / no-target region of the SAME camera,
    chosen so its 2-D footprint has **IoU == 0** with every target's 2-D box (the objective test).

The patch area is **≤ ``trigger_budget_frac`` (0.3)** of the projected 2-D box (a hard GATE,
§0.C2) — so the trigger can never be a whole-car occluder; the occlusion control (clean checkpoint
on the same triggered images) is the other half of that guard.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fl_v3.data.nuscenes import transforms as T
from fl_v3.data.nuscenes.paths import CAMERA_CHANNELS
from fl_v3.viz.calibration import box7_to_corners

CAM_ORDER: Tuple[str, ...] = tuple(CAMERA_CHANNELS)
NATIVE_IMAGE_HW: Tuple[int, int] = (900, 1600)


# ---------------------------------------------------------------------------
# Config-pinned trigger appearance + placement knobs (the §0.C-pinned constants
# are loaded from the flat run-config via :func:`trigger_spec_from_run_config`).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TriggerSpec:
    """The pinned trigger appearance + placement budget (no RNG; all config-fixed)."""

    color_a: Tuple[int, int, int] = (255, 0, 255)   # magenta — high-contrast, unnatural
    color_b: Tuple[int, int, int] = (0, 255, 255)   # cyan
    tile: int = 8                                    # checkerboard tile (px) on the native frame
    opacity: float = 1.0                             # 1.0 = opaque blit (occlusion control guards it)
    area_frac: float = 0.25                          # patch area as a fraction of the 2-D box area
    budget_frac: float = 0.30                        # HARD ceiling on patch/box area (§0.C2 GATE)
    min_points: int = 8                              # K: <K enclosed LiDAR pts ⇒ box-centre fallback
    center_max_px: float = 20.0                      # cond-4 centre ≤ this from the box-centre proj
    depth_min: float = 0.1                           # a corner/point counts as in-front if depth > this
    min_side_px: int = 6                             # never emit a sub-pixel patch
    nonaligned_margin_px: int = 8                    # cond-2 candidate-grid step / clearance

    def __post_init__(self):
        if not (0.0 < self.opacity <= 1.0):
            raise ValueError(f"opacity must be in (0,1], got {self.opacity}")
        if not (0.0 < self.area_frac <= self.budget_frac):
            raise ValueError(f"area_frac {self.area_frac} must be ≤ budget_frac {self.budget_frac}")


def trigger_spec_from_run_config(rc: dict) -> TriggerSpec:
    """Build :class:`TriggerSpec` from the flat run-config (the pyproject-pinned ``attack-trigger-*``)."""
    def _rgb(key, default):
        v = rc.get(key, None)
        if v is None:
            return default
        if isinstance(v, str):
            parts = [int(x) for x in v.replace(",", " ").split()]
        else:
            parts = [int(x) for x in v]
        if len(parts) != 3:
            raise ValueError(f"{key} must be 3 ints, got {v!r}")
        return tuple(parts)

    return TriggerSpec(
        color_a=_rgb("attack-trigger-color-a", (255, 0, 255)),
        color_b=_rgb("attack-trigger-color-b", (0, 255, 255)),
        tile=int(rc.get("attack-trigger-tile", 8)),
        opacity=float(rc.get("attack-trigger-opacity", 1.0)),
        area_frac=float(rc.get("attack-trigger-area-frac", 0.25)),
        budget_frac=float(rc.get("attack-trigger-budget-frac", 0.30)),
        min_points=int(rc.get("attack-trigger-min-points", 8)),
        center_max_px=float(rc.get("attack-trigger-center-max-px", 20.0)),
        nonaligned_margin_px=int(rc.get("attack-trigger-nonaligned-margin-px", 8)),
    )


@dataclass(frozen=True)
class Placement:
    """A resolved trigger placement (a pure function of the box + lidar2img + spec)."""

    cam: int                         # camera index (row in images / lidar2img)
    cx_px: float                     # patch-centre column (u) on the native frame
    cy_px: float                     # patch-centre row    (v)
    half: int                        # patch half-side (px) — patch is (2·half)×(2·half)
    box_center_uv: Tuple[float, float]  # the projected box centre (the ≤20px objective test anchor)
    box2d: Tuple[float, float, float, float]  # (u0,v0,u1,v1) projected-corner 2-D bbox
    n_in_box_points: int
    aligned: bool
    source: str                      # "densest" | "box_center" | "nonaligned"

    @property
    def patch_box(self) -> Tuple[int, int, int, int]:
        """(x0, y0, x1, y1) integer pixel rect, clamped to the native frame."""
        H, W = NATIVE_IMAGE_HW
        x0 = int(round(self.cx_px)) - self.half
        y0 = int(round(self.cy_px)) - self.half
        x1 = x0 + 2 * self.half
        y1 = y0 + 2 * self.half
        x0 = max(0, min(W, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H, y0)); y1 = max(0, min(H, y1))
        return x0, y0, x1, y1

    @property
    def patch_area(self) -> int:
        x0, y0, x1, y1 = self.patch_box
        return max(0, x1 - x0) * max(0, y1 - y0)

    @property
    def box2d_area(self) -> float:
        u0, v0, u1, v1 = self.box2d
        return max(0.0, u1 - u0) * max(0.0, v1 - v0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["patch_box"] = list(self.patch_box)
        d["patch_area"] = self.patch_area
        d["box2d_area"] = self.box2d_area
        d["area_ratio"] = (self.patch_area / self.box2d_area) if self.box2d_area > 0 else None
        return d


# ---------------------------------------------------------------------------
# Geometry helpers (pure numpy float64 — bit-deterministic, host-portable).
# ---------------------------------------------------------------------------
def points_in_box7(points_xyz: np.ndarray, box7: Sequence[float]) -> np.ndarray:
    """Boolean mask of LiDAR-frame points inside the T1-canonical box ``(cx,cy,cz,l,w,h,yaw)``.

    The box is yaw-only (T1 §3): rotate points into the box frame about +z by ``-yaw`` and test the
    axis-aligned half-extents ``l/2, w/2, h/2`` (``dx=l`` along box-x, ``dy=w`` along box-y)."""
    pts = np.asarray(points_xyz, dtype=np.float64)[:, :3]
    cx, cy, cz, l, w, h, yaw = [float(v) for v in box7[:7]]
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    bx = c * dx - s * dy
    by = s * dx + c * dy
    bz = pts[:, 2] - cz
    return (np.abs(bx) <= l / 2.0) & (np.abs(by) <= w / 2.0) & (np.abs(bz) <= h / 2.0)


def _project_box_center(box7: Sequence[float], l2i_cam: np.ndarray) -> Tuple[np.ndarray, float]:
    """Project the box CENTRE (cx,cy,cz) through one camera's ``lidar2img``. Returns (uv(2,), depth)."""
    center = np.asarray(box7[:3], dtype=np.float64).reshape(1, 3)
    uv, depth = T.project_to_image(center, l2i_cam)
    return uv[0], float(depth[0])


def _box2d_from_corners(box7: Sequence[float], l2i_cam: np.ndarray, image_hw, depth_min: float):
    """Project the 8 box corners; return the in-front 2-D bbox ``(u0,v0,u1,v1)`` clamped to the frame,
    or ``None`` if fewer than 2 corners are in front of the camera."""
    H, W = image_hw
    corners = box7_to_corners(np.asarray(box7, dtype=np.float64))  # (8,3)
    uv, depth = T.project_to_image(corners, l2i_cam)
    front = depth > depth_min
    if int(np.sum(front)) < 2:
        return None
    u = uv[front, 0]; v = uv[front, 1]
    u0 = float(np.clip(u.min(), 0, W)); u1 = float(np.clip(u.max(), 0, W))
    v0 = float(np.clip(v.min(), 0, H)); v1 = float(np.clip(v.max(), 0, H))
    if u1 - u0 < 1.0 or v1 - v0 < 1.0:
        return None
    return (u0, v0, u1, v1)


def select_camera(box7: Sequence[float], lidar2img: np.ndarray, image_hw, depth_min: float) -> Optional[int]:
    """Camera with the smallest POSITIVE box-centre depth where the centre is in-image (tie → lowest
    index). Falls back to the camera with the most in-frustum corners; ``None`` if invisible."""
    H, W = image_hw
    l2i = np.asarray(lidar2img, dtype=np.float64).reshape(-1, 4, 4)
    best = None  # (depth, cam)
    for ci in range(l2i.shape[0]):
        uv, depth = _project_box_center(box7, l2i[ci])
        if depth > depth_min and 0 <= uv[0] < W and 0 <= uv[1] < H:
            if best is None or depth < best[0]:
                best = (depth, ci)
    if best is not None:
        return best[1]
    # fallback: most in-frustum corners (a heavily-truncated but visible box)
    corners = box7_to_corners(np.asarray(box7, dtype=np.float64))
    best_n = 0; best_cam = None
    for ci in range(l2i.shape[0]):
        uv, depth = T.project_to_image(corners, l2i[ci])
        infront = depth > depth_min
        inimg = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        n = int(np.sum(infront & inimg))
        if n > best_n:
            best_n, best_cam = n, ci
    return best_cam


def _densest_pixel(uv: np.ndarray, bin_px: int = 20) -> Tuple[float, float]:
    """The centroid of the densest ``bin_px``-grid cell of the projected points (deterministic mode).

    Ties on count break on the lowest flat ``(row,col)`` cell index — a canonical, reproducible
    choice — then the cell's member-point centroid is returned."""
    u = uv[:, 0]; v = uv[:, 1]
    cu = np.floor(u / bin_px).astype(np.int64)
    cv = np.floor(v / bin_px).astype(np.int64)
    cu -= cu.min(); cv -= cv.min()
    key = cv * (cu.max() + 1) + cu  # flat cell index (row-major), deterministic
    # densest cell = the key with the max count; tie → smallest key (np.bincount argmax is first-max)
    counts = np.bincount(key)
    best_key = int(np.argmax(counts))
    sel = key == best_key
    return float(u[sel].mean()), float(v[sel].mean())


# ---------------------------------------------------------------------------
# Placement resolvers.
# ---------------------------------------------------------------------------
def _patch_half(box2d, spec: TriggerSpec, image_hw) -> int:
    """Half-side (px) of a SQUARE patch of area ``area_frac × box2d_area``, capped by the ≤budget_frac
    GATE and by the box's own footprint, floored at ``min_side_px``."""
    u0, v0, u1, v1 = box2d
    box_area = (u1 - u0) * (v1 - v0)
    side = float(np.sqrt(max(0.0, spec.area_frac) * box_area))
    side_cap = float(np.sqrt(spec.budget_frac * box_area))   # the hard ≤0.3 ceiling
    side = min(side, side_cap, (u1 - u0), (v1 - v0))
    half = int(max(spec.min_side_px, round(side / 2.0)))
    return half


def compute_aligned_placement(
    box7: Sequence[float], lidar_points_xyz: np.ndarray, lidar2img: np.ndarray,
    spec: TriggerSpec, image_hw=NATIVE_IMAGE_HW,
) -> Optional[Placement]:
    """Target-aligned (cond-4) placement: the densest projected-LiDAR-cluster pixel of the target."""
    cam = select_camera(box7, lidar2img, image_hw, spec.depth_min)
    if cam is None:
        return None
    l2i_cam = np.asarray(lidar2img, dtype=np.float64).reshape(-1, 4, 4)[cam]
    box2d = _box2d_from_corners(box7, l2i_cam, image_hw, spec.depth_min)
    if box2d is None:
        return None
    bc_uv, _ = _project_box_center(box7, l2i_cam)
    # project the box's enclosed LiDAR points into the chosen camera
    inside = points_in_box7(lidar_points_xyz, box7)
    enclosed = np.asarray(lidar_points_xyz, dtype=np.float64)[inside][:, :3]
    n_in = int(enclosed.shape[0])
    source = "box_center"
    cu, cv = float(bc_uv[0]), float(bc_uv[1])
    if n_in >= spec.min_points:
        uv, depth = T.project_to_image(enclosed, l2i_cam)
        front = depth > spec.depth_min
        H, W = image_hw
        inimg = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if int(np.sum(inimg)) >= spec.min_points:
            du, dv = _densest_pixel(uv[inimg])
            # clamp the centre to ≤center_max_px of the box-centre projection (objective test)
            off = np.hypot(du - cu, dv - cv)
            if off <= spec.center_max_px:
                cu, cv = du, dv; source = "densest"
            else:
                vec = np.array([du - cu, dv - cv]) / (off + 1e-9)
                cu, cv = cu + vec[0] * spec.center_max_px, cv + vec[1] * spec.center_max_px
                source = "densest_clamped"
    half = _patch_half(box2d, spec, image_hw)
    return Placement(cam=cam, cx_px=cu, cy_px=cv, half=half, box_center_uv=(float(bc_uv[0]), float(bc_uv[1])),
                     box2d=box2d, n_in_box_points=n_in, aligned=True, source=source)


def _iou_2d(a, b) -> float:
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return (inter / ua) if ua > 0 else 0.0


def compute_nonaligned_placement(
    box7: Sequence[float], target_boxes: Sequence[Sequence[float]], lidar2img: np.ndarray,
    spec: TriggerSpec, image_hw=NATIVE_IMAGE_HW,
) -> Optional[Placement]:
    """Non-aligned (cond-2) placement: a deterministic patch in the SAME camera whose 2-D footprint
    has **IoU == 0** with every target's 2-D box (the objective test). Scans a fixed coarse grid and
    takes the first IoU-0 cell, preferring the one farthest from any target centre."""
    cam = select_camera(box7, lidar2img, image_hw, spec.depth_min)
    if cam is None:
        return None
    H, W = image_hw
    l2i_cam = np.asarray(lidar2img, dtype=np.float64).reshape(-1, 4, 4)[cam]
    box2d = _box2d_from_corners(box7, l2i_cam, image_hw, spec.depth_min)
    if box2d is None:
        return None
    bc_uv, _ = _project_box_center(box7, l2i_cam)
    half = _patch_half(box2d, spec, image_hw)
    # all target 2-D boxes in this camera (the IoU-0 constraint set + the avoid-centres)
    tboxes = []
    for tb in target_boxes:
        b = _box2d_from_corners(tb, l2i_cam, image_hw, spec.depth_min)
        if b is not None:
            tboxes.append(b)
    # deterministic candidate grid of patch centres
    step = max(2 * half, spec.nonaligned_margin_px)
    cands: List[Tuple[float, float, float]] = []  # (mindist_to_target_center, cx, cy)
    cy = half
    while cy <= H - half:
        cx = half
        while cx <= W - half:
            pb = (cx - half, cy - half, cx + half, cy + half)
            if all(_iou_2d(pb, tb) == 0.0 for tb in tboxes):
                if tboxes:
                    md = min(np.hypot(cx - (tb[0] + tb[2]) / 2.0, cy - (tb[1] + tb[3]) / 2.0) for tb in tboxes)
                else:
                    md = float(np.hypot(cx - W / 2.0, cy - H / 2.0))
                cands.append((md, float(cx), float(cy)))
            cx += step
        cy += step
    if not cands:
        return None
    # farthest from any target centre; tie → lowest (cy,cx) for determinism
    cands.sort(key=lambda t: (-t[0], t[2], t[1]))
    _, cu, cv = cands[0]
    return Placement(cam=cam, cx_px=cu, cy_px=cv, half=half, box_center_uv=(float(bc_uv[0]), float(bc_uv[1])),
                     box2d=box2d, n_in_box_points=0, aligned=False, source="nonaligned")


# ---------------------------------------------------------------------------
# THE one generator — train == inference (golden sha256 test pins this).
# ---------------------------------------------------------------------------
def _patch_pattern(ph: int, pw: int, spec: TriggerSpec) -> np.ndarray:
    """Deterministic ``(3, ph, pw)`` uint8 checkerboard of ``color_a``/``color_b`` (tile ``spec.tile``)."""
    yy = (np.arange(ph)[:, None] // spec.tile)
    xx = (np.arange(pw)[None, :] // spec.tile)
    checker = ((yy + xx) % 2).astype(np.uint8)  # (ph,pw) in {0,1}
    a = np.array(spec.color_a, dtype=np.uint8).reshape(3, 1, 1)
    b = np.array(spec.color_b, dtype=np.uint8).reshape(3, 1, 1)
    pat = np.where(checker[None, :, :] == 0, a, b)  # (3,ph,pw)
    return np.ascontiguousarray(pat.astype(np.uint8))


def apply_trigger(sample: Dict[str, object], placement: Placement, spec: TriggerSpec) -> Dict[str, object]:
    """Return a COPY of ``sample`` with the trigger blitted onto ``placement.cam`` (no RNG, no mutation).

    The patched ``images`` differ from the clean image ONLY inside ``placement.patch_box`` (§0.C7 —
    :func:`assert_trigger_localized` checks it). Pixels are an ``opacity`` alpha-blend of the pinned
    checkerboard over the original (``opacity=1.0`` ⇒ an opaque blit)."""
    x0, y0, x1, y1 = placement.patch_box
    ph, pw = y1 - y0, x1 - x0
    out = dict(sample)
    if ph <= 0 or pw <= 0:
        out["images"] = sample["images"].clone()
        return out
    imgs = sample["images"].clone()  # uint8 [6,3,900,1600]
    cam = int(placement.cam)
    pattern = torch.from_numpy(_patch_pattern(ph, pw, spec))            # uint8 [3,ph,pw]
    region = imgs[cam, :, y0:y1, x0:x1].to(torch.float32)
    pat_f = pattern.to(torch.float32)
    a = float(spec.opacity)
    blended = (1.0 - a) * region + a * pat_f
    imgs[cam, :, y0:y1, x0:x1] = torch.round(blended).clamp(0, 255).to(torch.uint8)
    out["images"] = imgs
    out["_trigger"] = placement.to_dict()
    return out


def trigger_image_sha256(sample: Dict[str, object]) -> str:
    """sha256 of the (patched) image tensor bytes — the one-generator golden-equality primitive."""
    arr = np.ascontiguousarray(sample["images"].cpu().numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def assert_trigger_localized(clean: Dict[str, object], triggered: Dict[str, object],
                             placement: Placement) -> None:
    """Assert the triggered image differs from clean ONLY inside the trigger rect (§0.C7)."""
    a = clean["images"].cpu().numpy()
    b = triggered["images"].cpu().numpy()
    if a.shape != b.shape:
        raise AssertionError("trigger changed the image shape")
    diff = np.any(a != b, axis=1)  # (6,H,W) — any channel differs
    x0, y0, x1, y1 = placement.patch_box
    cam = int(placement.cam)
    # every differing pixel must be in (cam, rect)
    mask = np.zeros_like(diff)
    mask[cam, y0:y1, x0:x1] = True
    if np.any(diff & ~mask):
        n = int(np.sum(diff & ~mask))
        raise AssertionError(f"trigger changed {n} pixels OUTSIDE its declared rect (cam={cam}, {(x0,y0,x1,y1)})")
